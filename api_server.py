from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

# Import the core logic
from rag_core import ask_faq_core

app = FastAPI()

# Mount static directory to serve HTML
# Ensure the directory exists to avoid errors on start
if not os.path.exists("static"):
    os.makedirs("static")
    
app.mount("/static", StaticFiles(directory="static"), name="static")

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4
    chunk_size: Optional[int] = None
    similarity_threshold: Optional[float] = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    from fastapi.responses import FileResponse
    return FileResponse('static/index.html')

@app.post("/ask")
def ask_faq(req: QuestionRequest):
    try:
        # Call the core logic
        result = ask_faq_core(
            req.question, 
            top_k=req.top_k, 
            similarity_threshold=req.similarity_threshold, 
            chunk_size=req.chunk_size
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
