"""Memory-gap discovery, reconstruction, clue, and confirmation APIs."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.api.memories import get_memory_repository
from backend.app.core.config import get_settings
from backend.app.models.gap import (
    MemoryGapCandidateRecord,
    MemoryGapCandidateStatus,
    MemoryGapRecord,
    MemoryGapReconstructionResult,
    MemoryGapResolutionResult,
    MemoryGapStatus,
    MemoryGapUpdate,
)
from backend.app.services.gap_reconstruction import (
    MemoryGapAgentPolicyError,
    MemoryGapCandidateMismatchError,
    MemoryGapCandidateNotFoundError,
    MemoryGapCandidateOutputError,
    MemoryGapCandidateSourceError,
    MemoryGapClosedError,
    MemoryGapConfirmationRequiredError,
    MemoryGapModelUnavailableError,
    MemoryGapNotFoundError,
    MemoryGapReconstructionError,
    MemoryGapReconstructionService,
    MemoryGapResolutionService,
    MemoryGapResolutionUnsupportedError,
    MemoryGapTargetChangedError,
    MemoryGapToolExecutionError,
    build_openai_memory_gap_models,
)
from backend.app.services.gap_tools import build_memory_gap_tools
from backend.app.services.retrieval import BM25MemoryIndex, HybridMemoryRetriever
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.repository import SQLiteRepository

router = APIRouter(prefix="/memory-gaps", tags=["memory-gaps"])


class MemoryGapView(BaseModel):
    """One gap and every candidate currently available for user review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gap: MemoryGapRecord
    candidates: list[MemoryGapCandidateRecord]


class MemoryGapReconstructionRequest(BaseModel):
    web_search_consent: bool = False


class MemoryGapListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: list[MemoryGapView]


class MemoryGapClueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clue: str = Field(min_length=1, max_length=2000)

    @field_validator("clue")
    @classmethod
    def reject_blank_clue(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("clue must not be blank")
        return value.strip()


class MemoryGapResolutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1, max_length=200)
    user_confirmed: bool

    @field_validator("candidate_id")
    @classmethod
    def reject_blank_candidate(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate_id must not be blank")
        return value.strip()


@lru_cache(maxsize=1)
def get_memory_gap_reconstruction_service() -> MemoryGapReconstructionService:
    """Build the production local-search graph and OpenAI Tool Calling model."""

    settings = get_settings()
    repository = get_memory_repository()
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
    retriever = HybridMemoryRetriever(repository, vector_index, bm25_index)
    tools = build_memory_gap_tools(repository, retriever)
    return MemoryGapReconstructionService(
        repository,
        tools,
        build_openai_memory_gap_models(
            tools,
            settings.chat_model,
            api_key=settings.chat_api_key,
            base_url=settings.chat_base_url,
        ),
    )


@lru_cache(maxsize=1)
def get_memory_gap_resolution_service() -> MemoryGapResolutionService:
    """Build the non-LLM confirmation writer."""

    return MemoryGapResolutionService(get_memory_repository())


def _gap_view(repository: SQLiteRepository, gap: MemoryGapRecord) -> MemoryGapView:
    return MemoryGapView(
        gap=gap,
        candidates=repository.list_memory_gap_candidates(gap.gap_id),
    )


def _gap_http_error(exception: MemoryGapReconstructionError) -> HTTPException:
    """Map distinct reconstruction failures to stable, user-safe error codes."""

    if isinstance(exception, MemoryGapNotFoundError):
        return HTTPException(
            status_code=404,
            detail={
                "code": "memory_gap_not_found",
                "message": "기억 빈칸을 찾을 수 없습니다.",
            },
        )
    if isinstance(exception, MemoryGapCandidateNotFoundError):
        return HTTPException(
            status_code=404,
            detail={
                "code": "memory_gap_candidate_not_found",
                "message": "선택한 복원 후보를 찾을 수 없습니다.",
            },
        )
    if isinstance(exception, MemoryGapConfirmationRequiredError):
        return HTTPException(
            status_code=428,
            detail={
                "code": "memory_gap_confirmation_required",
                "message": "후보를 반영하려면 사용자가 직접 동의해야 합니다.",
            },
        )
    if isinstance(exception, MemoryGapClosedError):
        return HTTPException(
            status_code=409,
            detail={
                "code": "memory_gap_already_closed",
                "message": "이미 확인을 마친 기억 빈칸입니다.",
            },
        )
    if isinstance(exception, MemoryGapCandidateMismatchError):
        return HTTPException(
            status_code=409,
            detail={
                "code": "memory_gap_candidate_mismatch",
                "message": "이 기억 빈칸에서 만든 후보가 아닙니다.",
            },
        )
    if isinstance(exception, MemoryGapTargetChangedError):
        return HTTPException(
            status_code=409,
            detail={
                "code": "memory_gap_target_changed",
                "message": "원래 기억이 이미 수정되어 이 후보를 바로 반영할 수 없습니다.",
            },
        )
    if isinstance(exception, MemoryGapResolutionUnsupportedError):
        return HTTPException(
            status_code=422,
            detail={
                "code": "memory_gap_resolution_unsupported",
                "message": "선택한 후보를 이 기억 항목에 안전하게 반영할 수 없습니다.",
            },
        )
    if isinstance(exception, MemoryGapCandidateSourceError):
        return HTTPException(
            status_code=422,
            detail={
                "code": "memory_gap_candidate_source_invalid",
                "message": "후보를 뒷받침하는 내부 근거를 확인할 수 없습니다.",
            },
        )
    if isinstance(exception, MemoryGapAgentPolicyError):
        return HTTPException(
            status_code=503,
            detail={
                "code": "memory_gap_tool_policy_violation",
                "message": "안전하지 않은 검색 순서가 감지되어 복원을 중단했습니다.",
            },
        )
    if isinstance(exception, MemoryGapCandidateOutputError):
        return HTTPException(
            status_code=503,
            detail={
                "code": "memory_gap_candidate_output_invalid",
                "message": "복원 후보가 내부 근거와 일치하지 않아 저장하지 않았습니다.",
            },
        )
    if isinstance(exception, MemoryGapModelUnavailableError):
        return HTTPException(
            status_code=503,
            detail={
                "code": "memory_gap_model_unavailable",
                "message": "기억 복원 모델을 현재 사용할 수 없습니다.",
            },
        )
    if isinstance(exception, MemoryGapToolExecutionError):
        return HTTPException(
            status_code=503,
            detail={
                "code": "memory_gap_search_unavailable",
                "message": "기억 빈칸에 필요한 내부 검색을 현재 사용할 수 없습니다.",
            },
        )
    return HTTPException(
        status_code=503,
        detail={
            "code": "memory_gap_reconstruction_unavailable",
            "message": "기억 빈칸 복원을 현재 사용할 수 없습니다.",
        },
    )


@router.get("", response_model=MemoryGapListResponse)
def list_memory_gaps(
    include_closed: bool = False,
    repository: SQLiteRepository = Depends(get_memory_repository),
) -> MemoryGapListResponse:
    """List actionable gaps by default, optionally including closed history."""

    statuses = None
    if not include_closed:
        statuses = {
            MemoryGapStatus.OPEN,
            MemoryGapStatus.SEARCHING,
            MemoryGapStatus.CANDIDATE_FOUND,
            MemoryGapStatus.WAITING_USER,
        }
    return MemoryGapListResponse(
        items=[
            _gap_view(repository, gap)
            for gap in repository.list_memory_gaps(statuses=statuses)
        ]
    )


@router.get("/{gap_id}", response_model=MemoryGapView)
def get_memory_gap(
    gap_id: str,
    repository: SQLiteRepository = Depends(get_memory_repository),
) -> MemoryGapView:
    gap = repository.get_memory_gap(gap_id)
    if gap is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "memory_gap_not_found",
                "message": "기억 빈칸을 찾을 수 없습니다.",
            },
        )
    return _gap_view(repository, gap)


@router.post(
    "/{gap_id}/reconstruct",
    response_model=MemoryGapReconstructionResult,
)
def reconstruct_memory_gap(
    gap_id: str,
    request: MemoryGapReconstructionRequest = Body(default_factory=MemoryGapReconstructionRequest),
    service: MemoryGapReconstructionService = Depends(
        get_memory_gap_reconstruction_service
    ),
) -> MemoryGapReconstructionResult:
    try:
        if request.web_search_consent:
            repository = get_memory_repository()
            repository.update_memory_gap(gap_id, MemoryGapUpdate(web_search_consent=True))
        if request.web_search_consent:
            return service.reconstruct(gap_id, force_web_search=True)
        return service.reconstruct(gap_id)
    except MemoryGapReconstructionError as exception:
        raise _gap_http_error(exception) from exception


@router.post("/{gap_id}/clues", response_model=MemoryGapView)
def add_memory_gap_clue(
    gap_id: str,
    request: MemoryGapClueRequest,
    repository: SQLiteRepository = Depends(get_memory_repository),
) -> MemoryGapView:
    gap = repository.get_memory_gap(gap_id)
    if gap is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "memory_gap_not_found",
                "message": "기억 빈칸을 찾을 수 없습니다.",
            },
        )
    if gap.status in {MemoryGapStatus.RESOLVED, MemoryGapStatus.DISMISSED}:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "memory_gap_already_closed",
                "message": "이미 확인을 마친 기억 빈칸입니다.",
            },
        )
    if request.clue in gap.user_clues:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "memory_gap_clue_duplicate",
                "message": "이미 저장한 단서입니다.",
            },
        )
    updated = repository.update_memory_gap(
        gap_id,
        MemoryGapUpdate(
            status=MemoryGapStatus.OPEN,
            user_clues=[*gap.user_clues, request.clue],
        ),
    )
    return _gap_view(repository, updated)


@router.post(
    "/{gap_id}/resolve",
    response_model=MemoryGapResolutionResult,
)
def resolve_memory_gap(
    gap_id: str,
    request: MemoryGapResolutionRequest,
    service: MemoryGapResolutionService = Depends(
        get_memory_gap_resolution_service
    ),
) -> MemoryGapResolutionResult:
    try:
        return service.resolve(
            gap_id=gap_id,
            candidate_id=request.candidate_id,
            user_confirmed=request.user_confirmed,
        )
    except MemoryGapReconstructionError as exception:
        raise _gap_http_error(exception) from exception


@router.post("/{gap_id}/dismiss", response_model=MemoryGapView)
def dismiss_memory_gap(
    gap_id: str,
    repository: SQLiteRepository = Depends(get_memory_repository),
) -> MemoryGapView:
    gap = repository.get_memory_gap(gap_id)
    if gap is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "memory_gap_not_found",
                "message": "기억 빈칸을 찾을 수 없습니다.",
            },
        )
    if gap.status in {MemoryGapStatus.RESOLVED, MemoryGapStatus.DISMISSED}:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "memory_gap_already_closed",
                "message": "이미 확인을 마친 기억 빈칸입니다.",
            },
        )
    updated = repository.update_memory_gap(
        gap_id,
        MemoryGapUpdate(
            status=MemoryGapStatus.DISMISSED,
            resolved_at=datetime.now(UTC),
        ),
    )
    return _gap_view(repository, updated)
