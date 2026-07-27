"""Grounded question-answering API."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.core.config import get_settings
from backend.app.models.qa import QAResult
from backend.app.services.qa import GroundedQAService, QAError, build_openai_qa_models
from backend.app.services.retrieval import BM25MemoryIndex, HybridMemoryRetriever
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.repository import SQLiteRepository, StorageError

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    """Validated public chat input."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("session_id", "question")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


@lru_cache(maxsize=1)
def get_qa_service() -> GroundedQAService:
    """Build the persistent production Q&A dependencies lazily."""

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
    return GroundedQAService(
        repository,
        retriever,
        build_openai_qa_models(settings.openai_model),
    )


@router.post("/chat", response_model=QAResult)
def chat(
    request: ChatRequest,
    service: GroundedQAService = Depends(get_qa_service),
) -> QAResult:
    """Answer from retrieved memories and persist the conversation."""

    try:
        return service.answer_question(
            session_id=request.session_id,
            question=request.question,
            top_k=request.top_k,
        )
    except (QAError, StorageError) as exception:
        raise HTTPException(
            status_code=503,
            detail="Grounded question answering is unavailable",
        ) from exception
