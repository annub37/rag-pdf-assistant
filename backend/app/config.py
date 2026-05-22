import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "RAG PDF Assistant API")
    app_env: str = os.getenv("APP_ENV", "development")
    upload_dir: str = os.getenv("UPLOAD_DIR", "uploads")
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chroma_dir: str = os.getenv("CHROMA_DIR", "storage/chroma")


settings = Settings()

# Ensure upload directory exists when the app starts
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)