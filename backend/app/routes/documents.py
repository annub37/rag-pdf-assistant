import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import settings
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.chunker import chunk_pages
from app.services.embedder import embed_texts
from app.services.vector_store import store_chunks

# PDF files always start with these 5 bytes: %PDF-
PDF_MAGIC_BYTES = b"%PDF-"

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(file: UploadFile):
    # ── Guard 1: check the content type header ───────────
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    # ── Guard 2: read file and enforce size limit ────────
    contents = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {settings.max_upload_size_mb} MB.",
        )

    # ── Guard 3: verify actual file bytes (magic bytes) ──
    if not contents[:5].startswith(PDF_MAGIC_BYTES):
        raise HTTPException(
            status_code=400,
            detail="File is not a valid PDF. Content does not match PDF format.",
        )

    # ── Generate a unique filename to avoid collisions ───
    file_id = uuid.uuid4().hex
    safe_name = f"{file_id}.pdf"
    save_path = Path(settings.upload_dir) / safe_name

    # ── Write the uploaded bytes to disk ─────────────────
    save_path.write_bytes(contents)

    # ── Extract text from the saved PDF ──────────────────
    pages = extract_text_from_pdf(save_path)

    # ── Split extracted text into overlapping chunks ─────
    chunks = chunk_pages(
        pages=pages,
        file_id=file_id,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # ── Generate embeddings for each chunk ───────────────
    chunk_texts = [c["text"] for c in chunks]
    vectors = embed_texts(chunk_texts)

    # Attach the embedding vector to each chunk
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    # ── Store chunks + embeddings in ChromaDB ────────────
    stored_count = store_chunks(chunks)

    return {
        "file_id": file_id,
        "original_name": file.filename,
        "saved_to": str(save_path),
        "size_bytes": len(contents),
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "stored_in_vectordb": stored_count,
        "embedding_dimensions": len(vectors[0]) if vectors else 0,
    }
