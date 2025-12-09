# FAQ RAG + Local API (Starter Skeleton)

This is a minimal starting point for the HTTP API option.

## Contents
- `rag_core.py` — RAG core
- `api_server.py` — FastAPI app exposing `/health` and `/ask`
- `faqs/` — tiny sample corpus
- `requirements.txt`

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...

# Optional overrides
# export FAQ_DIR=./faqs
# export EMBED_MODEL=text-embedding-ada-002
# export LLM_MODEL=gpt-3.5-turbo

# Run locally (bind to loopback for safety)
uvicorn api_server:app --host 127.0.0.1 --port 8000

# Test
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json'   -d '{"question":"How do I reset my password?","top_k":4}'
```