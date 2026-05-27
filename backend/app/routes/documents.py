import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import settings
from app.logging_config import logger
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.chunker import chunk_pages
from app.services.embedder import embed_texts
from app.services.vector_store import (
    find_by_hash,
    store_chunks,
    list_documents,
    delete_by_file_id,
)

# PDF files always start with these 5 bytes: %PDF-
PDF_MAGIC_BYTES = b"%PDF-"

router = APIRouter(prefix="/documents", tags=["documents"])


async def _process_single_pdf(file: UploadFile) -> dict:
    """Validate, save, extract, embed, and store a single PDF. Returns a result dict."""
    # ── Guard 1: check the content type header ───────────
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are accepted. Got '{file.content_type}' for '{file.filename}'.",
        )

    # ── Guard 2: read file and enforce size limit ────────
    contents = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File '{file.filename}' too large. Maximum allowed size is {settings.max_upload_size_mb} MB.",
        )

    # ── Guard 3: verify actual file bytes (magic bytes) ──
    if not contents[:5].startswith(PDF_MAGIC_BYTES):
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' is not a valid PDF. Content does not match PDF format.",
        )

    # ── Guard 4: check for duplicate file ────────────────
    file_hash = hashlib.sha256(contents).hexdigest()
    existing = find_by_hash(file_hash)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"This PDF has already been uploaded (file_id: {existing['file_id']}, {existing['chunk_count']} chunks). Skipping duplicate.",
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
    stored_count = store_chunks(chunks, file_hash=file_hash)

    logger.info(
        "Processed PDF: name=%s file_id=%s pages=%d chunks=%d",
        file.filename, file_id, len(pages), len(chunks),
    )

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


@router.post("/upload")
async def upload_document(file: UploadFile):
    result = await _process_single_pdf(file)
    return result


@router.post("/upload-multiple")
async def upload_multiple_documents(files: list[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []
    errors = []
    for file in files:
        try:
            result = await _process_single_pdf(file)
            results.append(result)
        except HTTPException as e:
            errors.append({"filename": file.filename, "error": e.detail})

    return {
        "total_files": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


@router.get("/")
async def list_all_documents():
    """List all uploaded documents with their chunk counts."""
    docs = list_documents()

    # Enrich with filename from disk if the PDF file still exists
    upload_dir = Path(settings.upload_dir)
    for doc in docs:
        pdf_path = upload_dir / f"{doc['file_id']}.pdf"
        doc["file_exists"] = pdf_path.exists()

    return {"total_documents": len(docs), "documents": docs}


@router.delete("/{file_id}")
async def delete_document(file_id: str):
    """Delete a document and its chunks from ChromaDB and disk."""
    deleted_chunks = delete_by_file_id(file_id)
    if deleted_chunks == 0:
        raise HTTPException(status_code=404, detail=f"No document found with file_id '{file_id}'.")

    # Also remove the PDF file from disk if it exists
    pdf_path = Path(settings.upload_dir) / f"{file_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()

    logger.info("Deleted document: file_id=%s chunks=%d", file_id, deleted_chunks)
    return {"file_id": file_id, "deleted_chunks": deleted_chunks}
