import sys
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add parent directory to sys.path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from rag_core import _chunk_text, _load_and_chunk_faqs, ask_faq_core
from api_server import app

client = TestClient(app)

# --- Unit Tests ---

def test_chunk_text():
    # Recursive splitting: Small enough text stays intact
    text = "Paragraph 1.\nStill paragraph 1.\n\nParagraph 2."
    chunks = _chunk_text(text, size=100)
    assert chunks == ["Paragraph 1.\nStill paragraph 1.\n\nParagraph 2."]

def test_chunk_text_splits_large():
    # Force split by smaller size
    text = "Para1-Line1.\nPara1-Line2.\n\nPara2."
    # If size=20, it should split. 
    # "Para1-Line1." is ~12 chars. "\n" is 1.
    chunks = _chunk_text(text, size=20)
    # Recursion logic:
    # 1. \n\n -> "Para1-Line1.\nPara1-Line2." (Length ~25 > 20) -> Recurse on this
    #    -> Split by \n -> "Para1-Line1." (Okay), "Para1-Line2." (Okay)
    # 2. "Para2." (Okay)
    expected = ["Para1-Line1.", "Para1-Line2.", "Para2."]
    assert chunks == expected

def test_chunk_text_qa():
    # Verify Q&A structure is preserved if it fits
    text = "Q: Test?\nA: Yes.\n\nNext topic."
    chunks = _chunk_text(text, size=100)
    assert chunks == ["Q: Test?\nA: Yes.\n\nNext topic."]

def test_chunk_text_empty():
    assert _chunk_text("") == []

def test_missing_faq_dir(monkeypatch):
    """Test handling of non-existent FAQ directory."""
    # We test the internal loader directly
    with pytest.raises(ValueError, match="must be a directory"):
        _load_and_chunk_faqs("non_existent_directory_12345")

def test_empty_faq_dir(tmp_path):
    """Test handling of empty FAQ directory."""
    chunks, sources = _load_and_chunk_faqs(str(tmp_path))
    assert chunks == []
    assert sources == []

def test_whitespace_file_ignored(tmp_path):
    """Test that files with only whitespace are ignored."""
    d = tmp_path / "faqs"
    d.mkdir()
    f = d / "whitespace.md"
    f.write_text("   \n  \t  ")
    
    chunks, sources = _load_and_chunk_faqs(str(d))
    # If the current implementation chunks whitespace, this currently FAILS (which confirms the issue)
    # We want this to be empty eventually.
    # For now, let's assert what we expect to happen (it includes it) to demonstrate, 
    # or assert the desired behavior to confirm failure.
    # Let's assert the DESIRED behavior.
    assert chunks == []
    assert sources == []

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API Key required")
def test_ask_faq_core_sanity():
    """Smoke test for the core logic (requires API key)."""
    # Assuming the environment is set up correctly with the sample FAQs
    result = ask_faq_core("password", top_k=1)
    # The result should contain an answer and sources
    assert "answer" in result
    assert "sources" in result
    # We might get the placeholder if setup failed or empty, 
    # but with the current valid setup, we expect real results if preloaded.
    
# --- API Tests ---

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API Key required")
def test_ask_endpoint():
    payload = {"question": "How do I reset my password?", "top_k": 2}
    response = client.post("/ask", json=payload)
    if response.status_code == 500:
        # If server fails (e.g. key issue), fail test with message
        pytest.fail(f"Server error: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    # Check if we got the expected source
    assert any("faq_auth.md" in s for s in data["sources"])
