"""Environment-backed application settings."""

from functools import lru_cache
from os import getenv
from pathlib import Path

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
    openai_model: str = "gpt-5.6-sol"
    openai_embedding_model: str = "text-embedding-3-small"
    sqlite_database_path: Path = Path("data/db/life_archive.sqlite3")
    chroma_persist_directory: Path = Path("data/indexes/chroma")
    transcript_upload_directory: Path = Path("data/raw/transcripts")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable settings instance for the process."""
    return Settings(
        environment=getenv("APP_ENV", "development"),
        openai_model=getenv("OPENAI_MODEL", "gpt-5.6-sol"),
        openai_embedding_model=getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        sqlite_database_path=Path(
            getenv("SQLITE_DATABASE_PATH", "data/db/life_archive.sqlite3")
        ),
        chroma_persist_directory=Path(
            getenv("CHROMA_PERSIST_DIRECTORY", "data/indexes/chroma")
        ),
        transcript_upload_directory=Path(
            getenv("TRANSCRIPT_UPLOAD_DIRECTORY", "data/raw/transcripts")
        ),
    )
