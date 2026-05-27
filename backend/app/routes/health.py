from fastapi import APIRouter

from app.config import settings
from app.logging_config import logger

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    """Basic health check."""
    return {"status": "ok", "environment": settings.app_env}


@router.get("/health/detailed")
def detailed_health_check() -> dict:
    """Detailed health check with dependency status."""
    checks = {
        "status": "ok",
        "environment": settings.app_env,
        "dependencies": {},
    }

    # Check ChromaDB
    try:
        from app.services.vector_store import get_collection_stats
        stats = get_collection_stats()
        checks["dependencies"]["chromadb"] = {"status": "ok", **stats}
    except Exception as e:
        logger.error("ChromaDB health check failed: %s", e)
        checks["dependencies"]["chromadb"] = {"status": "error", "detail": str(e)}
        checks["status"] = "degraded"

    # Check LLM connectivity
    try:
        from app.services.llm import _get_client
        _get_client()
        checks["dependencies"]["llm"] = {"status": "ok", "provider": "azure_openai" if settings.azure_openai_api_key else "openai"}
    except Exception as e:
        logger.error("LLM health check failed: %s", e)
        checks["dependencies"]["llm"] = {"status": "error", "detail": str(e)}
        checks["status"] = "degraded"

    # Check embedding model
    try:
        from app.services.embedder import _model
        checks["dependencies"]["embedding_model"] = {"status": "ok", "model": settings.embedding_model}
    except Exception as e:
        logger.error("Embedding model health check failed: %s", e)
        checks["dependencies"]["embedding_model"] = {"status": "error", "detail": str(e)}
        checks["status"] = "degraded"

    return checks
