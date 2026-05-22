import chromadb
from app.config import settings

# Create a persistent ChromaDB client — data lives on disk at chroma_dir
_client = chromadb.PersistentClient(path=settings.chroma_dir)

# One collection for all document chunks
_collection = _client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},  # use cosine similarity
)


def store_chunks(chunks: list[dict]) -> int:
    """
    Store chunks with their embeddings in ChromaDB.

    Each chunk dict must have:
      - chunk_id: str
      - text: str
      - file_id: str
      - page_number: int
      - embedding: list[float]

    Returns the number of chunks stored.
    """
    _collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"file_id": c["file_id"], "page_number": c["page_number"]}
            for c in chunks
        ],
    )
    return len(chunks)


def search_similar(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Find the top_k most similar chunks to the query embedding.

    Returns a list of dicts with: chunk_id, text, file_id, page_number, distance
    """
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "file_id": results["metadatas"][0][i]["file_id"],
            "page_number": results["metadatas"][0][i]["page_number"],
            "distance": results["distances"][0][i],
        })
    return hits
