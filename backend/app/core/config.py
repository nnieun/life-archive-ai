"""Environment-backed application settings."""

from functools import lru_cache
from os import getenv
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# Resolve `.env` from the repository root instead of the process working
# directory. This keeps the setting reliable when Uvicorn is started elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=True)


class Settings(BaseModel):
    """Validated settings used by the backend application."""

    model_config = ConfigDict(frozen=True)

    app_name: str = "Life Archive AI"
    app_version: str = "0.0.0"
    api_prefix: str = "/api/v1"
    environment: str = "development"
    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_model: str = "gemma4:e2b"
    ollama_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-5.6-sol"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_provider: Literal['openai', 'ollama'] = 'openai'
    ollama_embedding_model: str = 'bge-m3'

    @property
    def embedding_model(self) -> str:
        return self.ollama_embedding_model if self.embedding_provider == 'ollama' else self.openai_embedding_model

    @property
    def embedding_index_directory(self) -> Path:
        if self.embedding_provider == 'openai':
            return self.chroma_persist_directory
        from hashlib import sha256
        suffix = sha256(self.ollama_embedding_model.encode()).hexdigest()[:12]
        return self.chroma_persist_directory.parent / f'{self.chroma_persist_directory.name}-ollama-{suffix}'
    sqlite_database_path: Path = Path("data/db/life_archive.sqlite3")
    chroma_persist_directory: Path = Path("data/indexes/chroma")
    transcript_upload_directory: Path = Path("data/raw/transcripts")
    qa_combined: bool = True
    qa_compact: bool = True
    qa_cache_enabled: bool = True
    qa_max_rewrites: int = Field(default=0, ge=0, le=1)
    qa_max_tokens: int = Field(default=768, ge=128, le=8192)
    ollama_keep_alive: str = "15m"
    qa_native_ollama: bool = True
    qa_verification_model: str = ""


    @property
    def chat_model(self) -> str:
        return self.ollama_model if self.llm_provider == "ollama" else self.openai_model
    @property
    def chat_api_key(self) -> str:
        return "ollama" if self.llm_provider == "ollama" else self.openai_api_key

    @property
    def chat_base_url(self) -> str | None:
        return self.ollama_base_url if self.llm_provider == "ollama" else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable settings instance for the process."""
    return Settings(
        llm_provider=getenv("LLM_PROVIDER", "ollama"),
        embedding_provider=getenv('EMBEDDING_PROVIDER', 'openai'),
        ollama_embedding_model=getenv('OLLAMA_EMBEDDING_MODEL', 'bge-m3'),
        qa_combined=getenv('QA_COMBINED', 'true').lower() == 'true',
        qa_compact=getenv('QA_COMPACT', 'true').lower() == 'true',
        qa_cache_enabled=getenv('QA_CACHE_ENABLED', 'true').lower() == 'true',
        qa_max_rewrites=int(getenv('QA_MAX_REWRITES', '0')),
        qa_max_tokens=int(getenv('QA_MAX_TOKENS', '768')),
        ollama_keep_alive=getenv('OLLAMA_KEEP_ALIVE', '15m'),
        qa_native_ollama=getenv('QA_NATIVE_OLLAMA', 'true').lower() == 'true',
        qa_verification_model=getenv('QA_VERIFICATION_MODEL', ''),
        ollama_model=getenv("OLLAMA_MODEL", "gemma4:e2b"),
        ollama_base_url=getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        environment=getenv("APP_ENV", "development"),
        openai_api_key=getenv("OPENAI_API_KEY", "").strip(),
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
