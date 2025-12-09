import os
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Config ---
FAQ_DIR = os.getenv("FAQ_DIR", str(Path(__file__).parent / "faqs"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.60"))

# Initialize the OpenAI client (fail fast if key missing)
_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=_API_KEY)

# Globals (preloaded at import)
_CHUNKS: List[str] = []
_SOURCES: List[str] = []
_CHUNK_EMBEDS: Optional[np.ndarray] = None  # shape: (N, d)

# ---------------- Core utilities ----------------
# ---------------- Core utilities ----------------
def _chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    """Recursively split text into smaller chunks to respect 'size' limit.
    Priority:
    1. Split by double newlines (\n\n) - Paragraphs
    2. Split by single newlines (\n) - Lines
    3. Split by sentences (. )
    4. Split by characters (hard slice)
    """
    if not text or not text.strip():
        return []
    
    # Base case: if text fits, return it
    if len(text) <= size:
        return [text.strip()]
    
    # Try splitting by delimiters in order of preference
    separators = ["\n\n", "\n", ". ", " "]
    
    for sep in separators:
        # If this separator exists in the text
        if sep in text:
            splits = text.split(sep)
            chunks = []
            current_chunk = []
            current_len = 0
            
            for split in splits:
                # Re-add separator length unless it's the last element (approximate)
                # Actually, simple recursion is easier: split all, then recurse on pieces that are too big?
                # Better strategy: Accumulate splits into a chunk until 'size' is reached.
                
                # Let's use a simpler accumulation approach for clarity and robustness
                # If a single split is huge, we recurse on IT.
                pass
            
            # Let's restart logic with a cleaner recursive accumulation pattern
            final_chunks = []
            sub_chunks = text.split(sep)
            
            current_buffer = ""
            for sub in sub_chunks:
                # If sub-chunk itself is massive, we must recurse on it deeper
                if len(sub) > size:
                    if current_buffer:
                        final_chunks.append(current_buffer)
                        current_buffer = ""
                    # Recurse on this specific sub-segment with the NEXT separator
                    # (Logic tricky here without keeping index. simplified: just recurse)
                    # Actually, simply calling _chunk_text on a large sub-segment might use the SAME separator again if we aren't careful.
                    # Standard recursive chunkers use a list of separators passed down.
                    pass 
                
            # Re-implementing a standard iterative-recursive approach
            chunks = [] 
            splits = text.split(sep)
            current_chunk = ""
            
            for split in splits:
                # Re-attach separator for readability if it's not whitespace
                # For \n\n and \n, we usually want to keep structure or just use the gap.
                # For split() behavior, the separator is gone.
                
                # Check if adding this split exceeds size
                if len(current_chunk) + len(split) + len(sep) <= size:
                    current_chunk += (sep if current_chunk else "") + split
                else:
                    # Current chunk is full, save it
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Now handle the new split. 
                    # If the split itself is larger than size, we must recurse on it!
                    if len(split) > size:
                        # But we must use a strictly *smaller* separator to avoid infinite recursion
                        # We can't easily recurse with _chunk_text unless we enable passing separator index.
                        # For this skeleton, let's implement a hard slice fallback.
                        if sep == " ": # Final separator
                             # Hard slice
                             chunks.extend([split[i:i+size] for i in range(0, len(split), size)])
                             current_chunk = "" # Consumed
                        else:
                             # Recurse with *full* function? No, might infinite loop.
                             # Simple fallback: Hard slice if logical split fails?
                             # Let's implement a simplified version for RAG Skeleton:
                             chunks.extend([split[i:i+size] for i in range(0, len(split), size)])
                             current_chunk = ""
                    else:
                        current_chunk = split
            
            if current_chunk:
                chunks.append(current_chunk)
                
            return chunks

    # If no separators found or all logic fell through (unlikely with " "), hard slice
    return [text[i : i + size] for i in range(0, len(text), size)]


# Let's Provide a cleaner, verified implementation of recursive splitting.
def _recursive_chunk(text: str, size: int) -> List[str]:
    """Helper for recursive chunking."""
    if not text or not text.strip():
        return []

    if len(text) <= size:
        return [text]
    
    separators = ["\n\n", "\n", ". ", " "]
    for sep in separators:
        if sep in text:
            splits = text.split(sep)
            chunks = []
            current = ""
            for s in splits:
                if not s: # Handle empty split artifacts
                    continue
                    
                # Re-add separator if not the last one (approximate logic for reconstruction)
                # Actually, for RAG, slight loss of separator format is acceptable if it means clean chunks.
                # But strictly, we should try to preserve. For simplicity here:
                # If we split by '\n\n', we effectively consume it.
                
                # Check if we can add 's' to 'current'
                # If current is empty, just take s.
                # If current has content, we need a separator in betwen.
                prefix = (sep if current else "") 
                candidate = current + prefix + s
                
                if len(candidate) <= size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                        current = ""
                    
                    # Now handle 's'. 
                    # If 's' itself is huge, recurse on it.
                    if len(s) > size:
                        # Recursive call
                        sub_chunks = _recursive_chunk(s, size)
                        chunks.extend(sub_chunks)
                        # Current remains empty
                    else:
                        current = s
            
            if current:
                chunks.append(current)
            return chunks
            
    # Fallback hard slice if no separator worked or text is just huge single word
    return [text[i : i + size] for i in range(0, len(text), size)]

def _chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
    # Initial whitespace check
    if not text or not text.strip():
        return []
    return _recursive_chunk(text, size)

def _load_and_chunk_faqs(faq_dir: str, chunk_size: int = CHUNK_SIZE) -> Tuple[List[str], List[str]]:
    """Load *.md files, chunk each, and return (chunks, matching_source_filenames)."""
    if not faq_dir:
        raise ValueError("faq_dir is required")
    
    faq_dir = Path(faq_dir)
    if not faq_dir.is_dir():
        raise ValueError("faq_dir must be a directory")
    
    chunks: List[str] = []
    sources: List[str] = []
    
    for faq_file in faq_dir.glob("*.md"):
        with open(faq_file, "r") as f:
            text = f.read()
        file_chunks = _chunk_text(text, size=chunk_size)
        chunks.extend(file_chunks)
        sources.extend([faq_file.name] * len(file_chunks))
        # print(file_chunks) # Reduced verbosity
        # print(sources)
    return chunks, sources

def _embed_texts(texts: List[str]) -> np.ndarray:
    """Create embeddings for texts and return a (N, d) float32 numpy array."""
    if not texts:
        return np.array([])
    return np.array([client.embeddings.create(input=text, model=EMBED_MODEL).data[0].embedding for text in texts])

def _embed_query(q: str) -> np.ndarray:
    """Create an embedding for the query and return a (d,) float32 vector."""
    if not q:
        raise ValueError("q is required")
    return np.array(client.embeddings.create(input=q, model=EMBED_MODEL).data[0].embedding)


def _generate_answer(context: str, question: str) -> str:
    """Call the chat model to answer using only context and cite filenames."""
    if not context or not question:
        raise ValueError("context and question are required")
    return client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
        ],
    ).choices[0].message.content


# ---------------- Public API ----------------
def ask_faq_core(
    question: str, 
    top_k: int = TOP_K_DEFAULT, 
    similarity_threshold: Optional[float] = None,
    chunk_size: Optional[int] = None
) -> Dict[str, object]:
    
    q = (question or "").strip()
    if not q:
        raise ValueError("question is required")
    if top_k <= 0:
        top_k = TOP_K_DEFAULT
        
    # Determine effective threshold
    eff_threshold = similarity_threshold if similarity_threshold is not None else SIMILARITY_THRESHOLD

    # Handle dynamic re-indexing if chunk_size changed
    if chunk_size is not None and chunk_size > 0 and chunk_size != CHUNK_SIZE:
        print(f"Re-indexing due to chunk size change: {CHUNK_SIZE} -> {chunk_size}")
        _preload(target_chunk_size=chunk_size)

    # If not yet implemented, return a safe placeholder so wrappers run.
    if _CHUNK_EMBEDS is None or len(_CHUNKS) == 0:
        if _API_KEY:
            try:
                # Initial load if empty
                _preload(target_chunk_size=chunk_size if chunk_size else CHUNK_SIZE)
            except Exception as e:
                print(f"Lazy load failed: {e}")
        
    if _CHUNK_EMBEDS is None or len(_CHUNKS) == 0:
        return {
            "answer": "System is not ready (no FAQs loaded or no API key).",
            "sources": []
        }

    try:
        q_emb = _embed_query(q)
        
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 1e-9:
            q_emb = q_emb / q_norm
            
        sims = _CHUNK_EMBEDS @ q_emb  # cosine if rows are normalized
        top_idx = np.argsort(sims)[-top_k:][::-1]
        
        print(f"\n--- Similarity Search (Threshold: {eff_threshold}, ChunkSize: {CHUNK_SIZE}) ---")
        for idx in top_idx:
            score = sims[idx]
            is_relevant = score >= eff_threshold
            status = "[PASS]" if is_relevant else "[FAIL]"
            snippet = _CHUNKS[idx][:60].replace('\n', ' ') + "..."
            print(f"{status} Score: {score:.4f} | File: {_SOURCES[idx]} | Text: {snippet}")
        print("-" * 60 + "\n")

        # Filter by threshold
        valid_idx = [i for i in top_idx if sims[i] >= eff_threshold]
        if not valid_idx:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": []
            }
            
        top_files = [_SOURCES[i] for i in valid_idx]
        context_parts = [f"From {_SOURCES[i]}:\n{_CHUNKS[i]}" for i in valid_idx]
        context = "\n\n".join(context_parts)
    
        answer = _generate_answer(context, q)
        distinct_sources = sorted(list({f for f in top_files}))
        # Limit distinct sources if desired, or skip limit since max is top_k
        sources_out = distinct_sources[:2] if len(distinct_sources) >= 2 else distinct_sources
        return {"answer": answer, "sources": sources_out}
        
    except Exception as e:
        return {"answer": f"Error processing request: {str(e)}", "sources": []}

# ---------------- Module preload ----------------

def _preload(target_chunk_size: int = CHUNK_SIZE) -> None:
    """Load and chunk FAQs, compute embeddings, L2-normalize rows, assign globals."""
    global _CHUNKS, _SOURCES, _CHUNK_EMBEDS, CHUNK_SIZE
    
    # Update global chunk size tracking
    CHUNK_SIZE = target_chunk_size
    
    # 1. Load chunks (now returns local lists)
    chunks, sources = _load_and_chunk_faqs(FAQ_DIR, chunk_size=target_chunk_size)
    
    if not chunks:
        print("Warning: No FAQ chunks found.")
        # Ensure globals are at least empty lists/None to avoid stale state if re-run
        _CHUNKS = []
        _SOURCES = []
        _CHUNK_EMBEDS = None
        return

    # 2. Embed chunks
    # Note: For production, we'd batch this more carefully or cache embeddings.
    print(f"Embedding {len(chunks)} chunks (Size: {target_chunk_size})...")
    embeds = _embed_texts(chunks)

    # 3. Normalize for cosine similarity
    # (x . y) / (|x| |y|) == (x/|x|) . (y/|y|)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    
    # Avoid zero-division if any embedding is all-zeros
    # Replace zero norms with 1.0 (so 0/1 = 0 instead of NaN)
    norms[norms < 1e-9] = 1.0
    
    embeds_norm = embeds / norms

    # 4. Assign globals
    _CHUNKS = chunks
    _SOURCES = sources
    _CHUNK_EMBEDS = embeds_norm
    print(f"Preloaded {len(_CHUNKS)} chunks from {len(set(_SOURCES))} files.")

# Run preload at import time (enable after implementation)
_preload()

# ---------------- Optional CLI runner ----------------
def main_cli():
    q = input("Enter your question: ")
    print(json.dumps(ask_faq_core(q), indent=2))

if __name__ == "__main__":
    main_cli()
