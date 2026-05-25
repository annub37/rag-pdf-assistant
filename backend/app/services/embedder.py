from sentence_transformers import SentenceTransformer

from app.config import settings

# Load the model once when this module is first imported.
# The model stays in memory so we don't reload it on every request.
_model = SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        texts: ["chunk text 1", "chunk text 2", ...]

    Returns:
        [[0.021, -0.045, ...], [0.019, -0.042, ...], ...]
        One vector per input text.
    """
    # batch_size=64: process 64 chunks at a time instead of all at once.
    # This keeps memory low and is faster for large documents.
    # normalize_embeddings: pre-normalizes vectors so cosine similarity
    # becomes a simple dot product (faster search later).
    vectors = _model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def embed_single(text: str) -> list[float]:
    """
    Convert a single text string into an embedding vector.
    Used for embedding user queries at search time.
    """
    vector = _model.encode(text)
    return vector.tolist()
