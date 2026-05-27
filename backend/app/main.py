import truststore
truststore.inject_into_ssl()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logging_config import logger
from app.middleware import RequestIdMiddleware
from app.routes.health import router as health_router
from app.routes.documents import router as documents_router
from app.routes.search import router as search_router
from app.routes.chat import router as chat_router


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware (order matters: first added = outermost) ──────────
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)

logger.info("RAG PDF Assistant API started [env=%s]", settings.app_env)