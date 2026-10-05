"""Grounded question-answering API."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
import sqlite3
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.core.config import get_settings
from backend.app.models.qa import QAResult
from backend.app.services.qa import GroundedQAService, QAError, build_openai_qa_models
from backend.app.services.retrieval import BM25MemoryIndex, HybridMemoryRetriever
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.repository import SQLiteRepository, StorageError
from backend.app.models.chat_job import ChatJob
from backend.app.services.chat_jobs import ChatJobStore
from backend.app.services.qa_failures import classify_exception, MESSAGES
from pydantic import ValidationError

router = APIRouter(tags=["chat"])


@lru_cache(maxsize=1)
def get_chat_job_store() -> ChatJobStore:
    database = SQLiteDatabase(get_settings().sqlite_database_path)
    database.initialize()
    return ChatJobStore(database)


def run_chat_job(job_id: str, request: "ChatRequest", store: ChatJobStore) -> None:
    if not store.start(job_id):
        return
    stage = "initialization"
    try:
        service = get_qa_service()
        stage = "execution"
        result = service.answer_question(
            session_id=request.session_id, question=request.question, top_k=request.top_k,
            progress=lambda payload: store.progress(job_id, payload),
        )
    except Exception as exception:
        code = "storage_error" if isinstance(exception, (StorageError, sqlite3.Error)) else classify_exception(exception)
        if stage == "initialization" and isinstance(exception, (ValidationError, ValueError)):
            code = "configuration_invalid"
        if code == "model_call_failed":
            code = "internal_error"
        message = MESSAGES.get(code, "모델 설정을 확인해 주세요.")
        prefix = "서비스 초기화" if stage == "initialization" else "질문 처리"
        store.finish(job_id, failure_code=code, failure_stage=stage, error_message=f"{prefix} 단계: {message}")
    else:
        store.finish(job_id, result)


class ChatRequest(BaseModel):
    """Validated public chat input."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=20)

    @field_validator("session_id", "question")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


@router.post("/chat/jobs", response_model=ChatJob, status_code=202)
def submit_chat_job(request: ChatRequest, tasks: BackgroundTasks,
                    store: ChatJobStore = Depends(get_chat_job_store)) -> ChatJob:
    try:
        job = store.create(request.session_id)
    except sqlite3.IntegrityError as exception:
        raise HTTPException(409, "A question is already being processed") from exception
    tasks.add_task(run_chat_job, job.job_id, request, store)
    return job


@router.get("/chat/jobs/{job_id}", response_model=ChatJob)
def read_chat_job(job_id: str, session_id: str,
                  store: ChatJobStore = Depends(get_chat_job_store)) -> ChatJob:
    job = store.get(job_id, session_id)
    if job is None:
        raise HTTPException(404, "Chat job was not found")
    return job


@lru_cache(maxsize=1)
def get_qa_service() -> GroundedQAService:
    """Build the persistent production Q&A dependencies lazily."""

    settings = get_settings()
    database = SQLiteDatabase(settings.sqlite_database_path)
    database.initialize()
    repository = SQLiteRepository(database)
    vector_index = MemoryVectorIndex(
        repository,
        settings.embedding_index_directory,
        embedding_model=settings.embedding_model,
        embedding_provider=settings.embedding_provider,
        embedding_base_url=settings.ollama_base_url,
        api_key=settings.openai_api_key,
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
        build_openai_qa_models(
            settings.chat_model,
            api_key=settings.chat_api_key,
            base_url=settings.chat_base_url,
            combined=settings.qa_combined,
            max_tokens=settings.qa_max_tokens,
            keep_alive=settings.ollama_keep_alive,
            native_ollama=settings.qa_native_ollama and settings.llm_provider == 'ollama',
            verification_model=settings.qa_verification_model or None,
        ),
        provider=settings.llm_provider,
        model_name=settings.chat_model,
        combined=settings.qa_combined,
        compact=settings.qa_compact,
        cache_enabled=settings.qa_cache_enabled,
        max_rewrites=settings.qa_max_rewrites,
        performance_enabled=True,
        configuration=f'qa-speed-v1:{settings.llm_provider}:{settings.chat_model}:{settings.embedding_provider}:{settings.embedding_model}:{settings.embedding_index_directory}:{settings.qa_combined}:{settings.qa_compact}:{settings.qa_max_rewrites}:{settings.qa_max_tokens}:{settings.qa_native_ollama}:{settings.qa_verification_model}',
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
