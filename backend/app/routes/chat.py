from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.embedder import embed_single
from app.services.vector_store import search_similar
from app.services.prompt_builder import build_messages
from app.services.llm import ask_llm


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    top_k: int = 10


@router.post("/")
async def chat(body: ChatRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 1: Embed the question into a vector
    query_vector = embed_single(body.question)

    # Step 2: Retrieve relevant chunks from ChromaDB
    chunks = search_similar(query_vector, top_k=body.top_k)

    if not chunks:
        return {
            "question": body.question,
            "answer": "No documents found. Please upload a PDF first.",
            "sources": [],
        }

    # Step 3: Build the prompt (system + context + question)
    messages = build_messages(body.question, chunks)

    # Step 4: Send to LLM and get the answer
    answer = ask_llm(messages)

    # Step 5: Return answer with source references + confidence score
    sources = [
        {"page_number": c["page_number"], "file_id": c["file_id"], "distance": c["distance"]}
        for c in chunks
    ]

    # Confidence = average of (1 - distance) across all chunks.
    # ChromaDB cosine distance: 0 = identical, 2 = opposite.
    # So (1 - distance) gives us a 0-to-1 confidence per chunk.
    avg_confidence = sum(1 - c["distance"] for c in chunks) / len(chunks)
    # Clamp between 0 and 1 (just in case)
    avg_confidence = max(0.0, min(1.0, avg_confidence))

    return {
        "question": body.question,
        "answer": answer,
        "confidence": round(avg_confidence, 4),
        "sources": sources,
    }
