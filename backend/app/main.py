from fastapi import FastAPI

from app.config import settings
from app.routes.health import router as health_router
from app.routes.documents import router as documents_router


app = FastAPI(title=settings.app_name)

app.include_router(health_router)
app.include_router(documents_router)