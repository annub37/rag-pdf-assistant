import chromadb
from app.config import settings

# Create a persistent ChromaDB client — data lives on disk at chroma_dir
_client = chromadb.PersistentClient(path=settings.chroma_dir)

# One collection for all document chunks
_collection = _client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},  # use cosine similarity
)


# ChromaDB recommends adding no more than ~500 items per call.
# Larger batches can cause memory spikes and slow indexing.
_CHROMA_BATCH_SIZE = 500


def store_chunks(chunks: list[dict], file_hash: str = "") -> int:
    """
    Store chunks with their embeddings in ChromaDB in batches.

    Each chunk dict must have:
      - chunk_id: str
      - text: str
      - file_id: str
      - page_number: int
      - embedding: list[float]

    Returns the number of chunks stored.
    """
    for i in range(0, len(chunks), _CHROMA_BATCH_SIZE):
        batch = chunks[i : i + _CHROMA_BATCH_SIZE]
        _collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "file_id": c["file_id"],
                    "page_number": c["page_number"],
                    "file_hash": file_hash,
                }
                for c in batch
            ],
        )
    return len(chunks)


def find_by_hash(file_hash: str) -> dict | None:
    """
    Check if a file with the given SHA-256 hash already exists in the collection.

    Returns a dict with file_id and chunk_count if found, otherwise None.
    """
    results = _collection.get(
        where={"file_hash": file_hash},
        include=["metadatas"],
    )
    if results["ids"]:
        file_id = results["metadatas"][0]["file_id"]
        return {"file_id": file_id, "chunk_count": len(results["ids"])}
    return None


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


def search_balanced(query_embedding: list[float], per_doc: int = 15) -> list[dict]:
    """
    Retrieve chunks balanced across ALL uploaded documents.

    Fetches up to `per_doc` chunks per document so that every PDF
    is represented in the results, regardless of similarity ranking.
    """
    # Get all unique file_ids in the collection
    all_data = _collection.get(include=["metadatas"])
    file_ids = list({m["file_id"] for m in all_data["metadatas"]})

    if not file_ids:
        return []

    hits = []
    for fid in file_ids:
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=per_doc,
            where={"file_id": fid},
        )
        for i in range(len(results["ids"][0])):
            hits.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "file_id": results["metadatas"][0][i]["file_id"],
                "page_number": results["metadatas"][0][i]["page_number"],
                "distance": results["distances"][0][i],
            })

    # Sort all hits by distance (most relevant first)
    hits.sort(key=lambda h: h["distance"])
    return hits
