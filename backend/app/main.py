import truststore
truststore.inject_into_ssl()

from fastapi import FastAPI

from app.config import settings
from app.routes.health import router as health_router
from app.routes.documents import router as documents_router
from app.routes.search import router as search_router
from app.routes.chat import router as chat_router


app = FastAPI(title=settings.app_name)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)