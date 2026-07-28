"""Grounded autobiography generation and retrieval API."""

from __future__ import annotations

from functools import lru_cache
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.core.config import get_settings
from backend.app.models.autobiography import (
    AutobiographyGenerationResult,
    AutobiographyInput,
)
from backend.app.services.autobiography import (
    AutobiographyService,
    build_openai_autobiography_models,
)
from backend.app.services.retrieval import BM25MemoryIndex, HybridMemoryRetriever
from backend.app.services.timeline import TimelineService
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import AutobiographyRecord
from backend.app.storage.repository import (
    SQLiteRepository,
    StorageConflictError,
)

router = APIRouter(tags=["autobiographies"])


class AutobiographyRequest(BaseModel):
    """Validated public autobiography generation request."""

    model_config = ConfigDict(extra="forbid")

    autobiography_id: str | None = Field(default=None, min_length=1)
    title: str = Field(min_length=1, max_length=200)
    request: str = Field(min_length=1, max_length=4000)
    target_period: str | None = Field(default=None, max_length=200)
    target_topics: list[str] = Field(default_factory=list, max_length=20)
    chapter_count: int = Field(default=1, ge=1, le=3)
    top_k: int = Field(default=10, ge=1, le=30)

    @field_validator(
        "autobiography_id",
        "title",
        "request",
        "target_period",
    )
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    @field_validator("target_topics")
    @classmethod
    def validate_topics(cls, value: list[str]) -> list[str]:
        if any(not topic.strip() for topic in value):
            raise ValueError("target_topics must not contain blanks")
        if len(value) != len(set(value)):
            raise ValueError("target_topics must be unique")
        return value


@lru_cache(maxsize=1)
def get_autobiography_service() -> AutobiographyService:
    """Build persistent generation dependencies lazily."""

    settings = get_settings()
    database = SQLiteDatabase(settings.sqlite_database_path)
    database.initialize()
    repository = SQLiteRepository(database)
    vector_index = MemoryVectorIndex(
        repository,
        settings.chroma_persist_directory,
        embedding_model=settings.openai_embedding_model,
    )
    vector_index.sync_from_sqlite()
    bm25_index = BM25MemoryIndex(repository)
    bm25_index.rebuild_from_sqlite()
    retriever = HybridMemoryRetriever(
        repository,
        vector_index,
        bm25_index,
    )
    return AutobiographyService(
        repository,
        retriever,
        TimelineService(repository),
        build_openai_autobiography_models(settings.openai_model),
    )


@router.post(
    "/autobiographies",
    response_model=AutobiographyGenerationResult,
)
def generate_autobiography(
    request: AutobiographyRequest,
    service: AutobiographyService = Depends(get_autobiography_service),
) -> AutobiographyGenerationResult:
    """Generate and persist a maximum-three-chapter grounded draft."""

    autobiography_id = (
        request.autobiography_id or f"autobio_{uuid4().hex}"
    )
    try:
        return service.generate(
            AutobiographyInput(
                autobiography_id=autobiography_id,
                title=request.title,
                request=request.request,
                target_period=request.target_period,
                target_topics=request.target_topics,
                chapter_count=request.chapter_count,
                top_k=request.top_k,
            )
        )
    except StorageConflictError as exception:
        raise HTTPException(
            status_code=409,
            detail="Autobiography already exists",
        ) from exception


@router.get(
    "/autobiographies/{autobiography_id}",
    response_model=AutobiographyRecord,
)
def get_autobiography(
    autobiography_id: str,
    service: AutobiographyService = Depends(get_autobiography_service),
) -> AutobiographyRecord:
    """Return one stored autobiography draft or completed result."""

    record = service.get(autobiography_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Autobiography not found")
    return record
