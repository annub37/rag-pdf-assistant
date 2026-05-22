import uuid


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a single text string into overlapping pieces.

    Args:
        text:       the full text to split
        chunk_size: max number of characters per chunk
        overlap:    how many characters to repeat between consecutive chunks

    Returns:
        list of text pieces
    """
    if not text or not text.strip():
        return []

    # Step size = how far we move forward each time
    # Example: chunk_size=500, overlap=100 → step=400
    step = chunk_size - overlap

    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)

    return chunks


def chunk_pages(
    pages: list[dict],
    file_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    Take the output of pdf_extractor (list of pages) and split
    each page's text into overlapping chunks with metadata.

    Args:
        pages:         [{"page_number": 1, "text": "..."}, ...]
        file_id:       the document's unique ID (for metadata)
        chunk_size:    max characters per chunk
        chunk_overlap: overlap between chunks

    Returns:
        [
            {
                "chunk_id": "abc123",
                "file_id": "xyz789",
                "page_number": 1,
                "text": "chunk text here...",
            },
            ...
        ]
    """
    all_chunks = []

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]

        pieces = chunk_text(text, chunk_size, chunk_overlap)

        for piece in pieces:
            all_chunks.append({
                "chunk_id": uuid.uuid4().hex,
                "file_id": file_id,
                "page_number": page_number,
                "text": piece,
            })

    return all_chunks
