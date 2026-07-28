"""Transcript ingestion and structured-memory read APIs."""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as Base64Error
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from backend.app.core.config import get_settings
from backend.app.models.ingestion import IngestionResult
from backend.app.services.ingestion import (
    IngestionError,
    InvalidUploadError,
    TranscriptIngestionService,
    UploadConflictError,
)
from backend.app.services.memory_extraction import (
    MemoryExtractionError,
    build_openai_memory_model,
)
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import CitationRecord, MemoryRecord
from backend.app.storage.repository import SQLiteRepository, StorageError

router = APIRouter(tags=["memories"])


class IngestTranscriptRequest(BaseModel):
    """Base64 transport keeps the original TXT bytes unchanged."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1, max_length=20_000_000)
    language: str | None = Field(default=None, max_length=32)
    recorded_at: AwareDatetime | None = None

    @field_validator("filename", "language")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text fields must not be blank")
        return value


class MemoryView(BaseModel):
    """Structured memory plus traceable SQLite source offsets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory: MemoryRecord
    citations: list[CitationRecord]


@lru_cache(maxsize=1)
def get_memory_repository() -> SQLiteRepository:
    """Build the shared SQLite-backed memory reader lazily."""

    database = SQLiteDatabase(get_settings().sqlite_database_path)
    database.initialize()
    return SQLiteRepository(database)


@lru_cache(maxsize=1)
def get_ingestion_service() -> TranscriptIngestionService:
    """Build ingestion dependencies without exposing them to Streamlit."""

    settings = get_settings()
    repository = get_memory_repository()
    return TranscriptIngestionService(
        settings.transcript_upload_directory,
        repository,
        build_openai_memory_model(settings.openai_model),
        MemoryVectorIndex(
            repository,
            settings.chroma_persist_directory,
            embedding_model=settings.openai_embedding_model,
        ),
    )


@router.post("/memories/ingest", response_model=IngestionResult)
def ingest_transcript(
    request: IngestTranscriptRequest,
    service: TranscriptIngestionService = Depends(get_ingestion_service),
) -> IngestionResult:
    """Upload and process one immutable UTF-8 TXT transcript."""

    try:
        content = b64decode(request.content_base64, validate=True)
    except (Base64Error, ValueError) as exception:
        raise HTTPException(
            status_code=422,
            detail="TXT upload encoding is invalid",
        ) from exception
    try:
        return service.ingest(
            filename=request.filename,
            content=content,
            language=request.language,
            recorded_at=request.recorded_at,
        )
    except UploadConflictError as exception:
        raise HTTPException(
            status_code=409,
            detail="A transcript with this filename already exists",
        ) from exception
    except InvalidUploadError as exception:
        raise HTTPException(status_code=422, detail=str(exception)) from exception
    except (IngestionError, MemoryExtractionError, StorageError) as exception:
        raise HTTPException(
            status_code=503,
            detail="Transcript processing is unavailable",
        ) from exception


@router.get("/memories", response_model=list[MemoryView])
def list_memories(
    transcript_id: str | None = Query(default=None, min_length=1),
    repository: SQLiteRepository = Depends(get_memory_repository),
) -> list[MemoryView]:
    """Return active structured memories with source offsets."""

    return [
        MemoryView(
            memory=memory,
            citations=[
                CitationRecord(
                    memory_id=source.memory_id,
                    transcript_id=source.transcript_id,
                    segment_id=source.segment_id,
                    start_offset=source.start_offset,
                    end_offset=source.end_offset,
                )
                for source in repository.list_memory_sources(memory.memory_id)
            ],
        )
        for memory in repository.list_memories(transcript_id=transcript_id)
    ]
