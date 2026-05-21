import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "RAG PDF Assistant API")
    app_env: str = os.getenv("APP_ENV", "development")


settings = Settings()