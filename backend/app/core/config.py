"""Environment-backed application settings."""

from functools import lru_cache
from os import getenv

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()


class Settings(BaseModel):
    """Validated settings used by the backend application."""

    model_config = ConfigDict(frozen=True)

    app_name: str = "Life Archive AI"
    app_version: str = "0.0.0"
    api_prefix: str = "/api/v1"
    environment: str = "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable settings instance for the process."""
    return Settings(environment=getenv("APP_ENV", "development"))
