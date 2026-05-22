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
    vectors = _model.encode(texts)
    return vectors.tolist()


def embed_single(text: str) -> list[float]:
    """
    Convert a single text string into an embedding vector.
    Used for embedding user queries at search time.
    """
    vector = _model.encode(text)
    return vector.tolist()
