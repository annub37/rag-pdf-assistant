from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.embedder import embed_single
from app.services.vector_store import search_similar


router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/")
async def search_documents(body: SearchRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Step 1: convert the question into an embedding vector
    query_vector = embed_single(body.query)

    # Step 2: find the most similar chunks in ChromaDB
    results = search_similar(query_vector, top_k=body.top_k)

    return {
        "query": body.query,
        "top_k": body.top_k,
        "results": results,
    }
