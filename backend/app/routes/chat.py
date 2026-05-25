from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.embedder import embed_single
from app.services.vector_store import search_similar, search_balanced
from app.services.prompt_builder import build_messages
from app.services.llm import ask_llm


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    top_k: int = 10
    balanced: bool = False


@router.post("/")
async def chat(body: ChatRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 1: Embed the question into a vector
    query_vector = embed_single(body.question)

    # Step 2: Retrieve relevant chunks from ChromaDB
    # Use balanced search for summaries to cover all documents equally
    if body.balanced:
        chunks = search_balanced(query_vector, per_doc=body.top_k)
    else:
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

        # Step 5: Return answer with source references
    sources = [
        {"page_number": c["page_number"], "file_id": c["file_id"], "distance": c["distance"]}
        for c in chunks
    ]

    return {
        "question": body.question,
        "answer": answer,
        "sources": sources,
    }
