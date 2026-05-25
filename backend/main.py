import json
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()  # picks up ../.env when running locally

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict

from rag import RAGPipeline
from agent import SmartDocAgent

app = FastAPI(title="SmartDoc Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

rag = RAGPipeline(chroma_path=CHROMA_PATH)
agent = SmartDocAgent(rag=rag, api_key=DEEPSEEK_API_KEY)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []


class ChatResponse(BaseModel):
    answer: str
    history: List[Dict[str, str]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = rag.index_document(tmp_path, file.filename)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        os.unlink(tmp_path)

    return {
        "message": f"Indexed '{file.filename}' into {result['chunks']} chunks.",
        **result,
    }


@app.get("/documents")
def list_documents():
    return {"documents": rag.list_documents()}


@app.delete("/documents/{filename:path}")
def delete_document(filename: str):
    deleted = rag.delete_document(filename)
    if deleted == 0:
        raise HTTPException(404, f"Document '{filename}' not found.")
    return {"message": f"Deleted '{filename}' ({deleted} chunks removed)."}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")
    result = agent.chat(req.history, req.message)
    return ChatResponse(answer=result["answer"], history=result["history"])


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")

    def generate():
        for event in agent.stream_chat(req.history, req.message):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
