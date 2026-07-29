"""Privacy and transcript-deletion API."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from backend.app.core.config import get_settings
from backend.app.models.privacy import TranscriptDeletionResult
from backend.app.services.privacy import (
    PrivacyDeletionError,
    TranscriptDeletionService,
)
from backend.app.services.retrieval import BM25MemoryIndex
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.repository import (
    SQLiteRepository,
    StorageNotFoundError,
)

router = APIRouter(tags=["privacy"])


@lru_cache(maxsize=1)
def get_transcript_deletion_service() -> TranscriptDeletionService:
    """Build persistent deletion dependencies lazily."""

    settings = get_settings()
    database = SQLiteDatabase(settings.sqlite_database_path)
    database.initialize()
    repository = SQLiteRepository(database)
    return TranscriptDeletionService(
        repository,
        MemoryVectorIndex(
            repository,
            settings.chroma_persist_directory,
            embedding_model=settings.openai_embedding_model,
        ),
        BM25MemoryIndex(repository),
    )


@router.delete(
    "/transcripts/{transcript_id}",
    response_model=TranscriptDeletionResult,
)
def delete_transcript(
    transcript_id: Annotated[str, Path(min_length=1, max_length=200)],
    service: TranscriptDeletionService = Depends(
        get_transcript_deletion_service
    ),
) -> TranscriptDeletionResult:
    """Remove a transcript from application access but preserve its raw file."""

    try:
        return service.delete_transcript(transcript_id)
    except StorageNotFoundError as exception:
        raise HTTPException(
            status_code=404,
            detail="Transcript not found",
        ) from exception
    except PrivacyDeletionError as exception:
        raise HTTPException(
            status_code=503,
            detail="Transcript indexes could not be fully cleaned",
        ) from exception
