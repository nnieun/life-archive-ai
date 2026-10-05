"""Typed HTTP client used by every Streamlit page."""

from __future__ import annotations

from base64 import b64encode
from datetime import date, datetime
from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_API_URL: Final = "http://127.0.0.1:8000/api/v1"
DEFAULT_TIMEOUT_SECONDS: Final = 60.0
INGEST_TIMEOUT_SECONDS: Final = 180.0
CHAT_TIMEOUT_SECONDS: Final = 900.0


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class HealthStatus(ApiModel):
    status: Literal["ok"]
    service: str
    version: str


class Citation(ApiModel):
    memory_id: str
    transcript_id: str
    segment_id: str | None = None
    start_offset: int
    end_offset: int
    start_line: int | None = None
    end_line: int | None = None


class IngestionResult(ApiModel):
    transcript_id: str
    filename: str
    segment_count: int
    memory_count: int
    gap_count: int = 0
    indexed_memory_count: int
    memory_ids: list[str]
    gap_ids: list[str] = Field(default_factory=list)


class TranscriptDeletionResult(ApiModel):
    transcript_id: str
    deleted_segment_count: int
    deleted_memory_count: int
    dismissed_gap_count: int = 0
    deleted_vector_count: int
    bm25_memory_count: int
    invalidated_conversation_message_count: int
    invalidated_autobiography_count: int
    raw_file_deleted: bool = False


class MemoryData(ApiModel):
    memory_id: str
    transcript_id: str
    title: str
    summary: str
    people: list[str]
    location: str | None = None
    event_date: str | None = None
    date_precision: str
    emotion: str | None = None
    confidence: float
    uncertainty_notes: str | None = None
    status: str


class MemoryView(ApiModel):
    memory: MemoryData
    citations: list[Citation]
    source_filename: str


class MemoryGapData(ApiModel):
    gap_id: str
    memory_id: str | None = None
    gap_type: str
    clue_text: str
    missing_field: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    location: str | None = None
    people: list[str] = Field(default_factory=list)
    confidence: float
    importance_score: float
    status: str
    web_search_consent: bool = False
    user_clues: list[str] = Field(default_factory=list)
    resolved_candidate_id: str | None = None
    resolved_memory_id: str | None = None


class MemoryGapCandidateData(ApiModel):
    candidate_id: str
    gap_id: str
    value: str
    explanation: str
    deterministic_score: float
    llm_relation: str
    supporting_source_ids: list[str] = Field(default_factory=list)
    status: str
    external_sources: list[dict[str, object]] = Field(default_factory=list)


class MemoryGapView(ApiModel):
    gap: MemoryGapData
    candidates: list[MemoryGapCandidateData] = Field(default_factory=list)


class MemoryGapListResponse(ApiModel):
    items: list[MemoryGapView]


class MemoryGapReconstructionResult(ApiModel):
    gap: MemoryGapData
    candidates: list[MemoryGapCandidateData]
    searched_tools: list[str]
    tool_call_count: int
    needs_more_clues: bool = False
    user_question: str | None = None
    message: str


class MemoryGapResolutionResult(ApiModel):
    gap: MemoryGapData
    candidate: MemoryGapCandidateData
    resolved_memory_id: str


class QAValidation(ApiModel):
    stage: str
    passed: bool
    reason: str
    failure_code: str | None = None
    exception_type: str | None = None


class ChatResult(ApiModel):
    session_id: str
    question: str
    retrieved_memory_ids: list[str]
    final_answer: str
    citations: list[Citation]
    validation_result: QAValidation
    retry_count: int
    error: str | None = None
    elapsed_ms: float = 0
    cache_hit: bool = False


class TimelineEvent(ApiModel):
    memory_id: str
    title: str
    description: str
    event_date: str | None = None
    date_precision: str
    date_label: str
    confidence: float
    uncertainty_notes: str | None = None
    citations: list[Citation]


class TimelineResult(ApiModel):
    events: list[TimelineEvent]
    undated_events: list[TimelineEvent]
    start_date: date | None = None
    end_date: date | None = None


class AutobiographyChapter(ApiModel):
    title: str
    content: str
    citations: list[Citation]


class AutobiographyContent(ApiModel):
    chapters: list[AutobiographyChapter] = Field(max_length=3)


class AutobiographyData(ApiModel):
    autobiography_id: str
    title: str
    content: AutobiographyContent
    status: str


class AutobiographyResult(ApiModel):
    autobiography: AutobiographyData
    completed: bool
    retrieved_memory_ids: list[str]
    citations: list[Citation]
    retry_count: int
    error: str | None = None
    important_unresolved_gaps: list[MemoryGapData] = Field(default_factory=list)
    requires_gap_confirmation: bool = False


class AutobiographyGapCheckResult(ApiModel):
    important_unresolved_gaps: list[MemoryGapData]
    retrieved_memory_ids: list[str]
    requires_gap_confirmation: bool


class ChatJob(ApiModel):
    job_id: str
    session_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: datetime
    result: ChatResult | None = None
    error: str | None = None
    failure_code: str | None = None
    failure_stage: str | None = None
    progress: dict = Field(default_factory=dict)


class IngestionJob(ApiModel):
    job_id: str
    session_id: str
    filename: str
    created_at: datetime
    status: Literal['queued', 'running', 'completed', 'failed', 'cancelled']
    progress: dict = Field(default_factory=dict)
    result: IngestionResult | None = None
    error: str | None = None
    failure_code: str | None = None


class ApiClientError(RuntimeError):
    """User-safe backend communication failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        error_code: str | None = None,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.error_code = error_code
        self.user_message = user_message


def _api_error(
    response: httpx.Response,
    message: str,
) -> ApiClientError:
    """Read only the safe request ID from a backend error response."""

    request_id = response.headers.get("X-Request-ID")
    error_code: str | None = None
    response_message: str | None = None
    try:
        error = response.json().get("error", {})
        if isinstance(error, dict):
            if isinstance(error.get("request_id"), str):
                request_id = error["request_id"]
            if isinstance(error.get("code"), str):
                error_code = error["code"]
            if isinstance(error.get("message"), str):
                response_message = error["message"]
    except (ValueError, AttributeError):
        pass
    return ApiClientError(
        message,
        status_code=response.status_code,
        request_id=request_id,
        error_code=error_code,
        user_message=response_message,
    )


class LifeArchiveApiClient:
    """Small synchronous client suitable for Streamlit reruns."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = f"{base_url.rstrip('/')}/"
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def _request(
        self,
        method: str,
        path: str,
        response_model: type[ApiModel],
        *,
        error_message: str,
        timeout_seconds: float | None = None,
        **kwargs: object,
    ) -> ApiModel:
        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=timeout_seconds or self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = client.request(method, path, **kwargs)
                response.raise_for_status()
                return response_model.model_validate(response.json())
        except httpx.HTTPStatusError as exception:
            raise _api_error(exception.response, error_message) from exception
        except httpx.TimeoutException as exception:
            raise ApiClientError(
                error_message,
                error_code="request_timeout",
                user_message="응답 대기 시간이 초과됐습니다. 모델 처리에 시간이 걸릴 수 있습니다. 잠시 후 다시 확인해 주세요.",
            ) from exception
        except (httpx.HTTPError, ValueError, ValidationError) as exception:
            raise ApiClientError(error_message) from exception

    def get_health(self) -> HealthStatus:
        return HealthStatus.model_validate(
            self._request(
                "GET",
                "health",
                HealthStatus,
                error_message="Backend health check failed",
            )
        )

    def ingest_transcript(
        self,
        filename: str,
        content: bytes,
        *,
        language: str | None = None,
        recorded_at: datetime | None = None,
    ) -> IngestionResult:
        payload = {
            "filename": filename,
            "content_base64": b64encode(content).decode("ascii"),
            "language": language,
            "recorded_at": recorded_at.isoformat() if recorded_at else None,
        }
        return IngestionResult.model_validate(
            self._request(
                "POST",
                "memories/ingest",
                IngestionResult,
                error_message="Transcript upload failed",
                timeout_seconds=INGEST_TIMEOUT_SECONDS,
                json=payload,
            )
        )

    def submit_ingestion_job(self, filename: str, content: bytes, *, session_id: str,
                             language: str | None = None, recorded_at: datetime | None = None) -> IngestionJob:
        return IngestionJob.model_validate(self._request('POST', 'memories/ingest/jobs', IngestionJob,
            error_message='Upload submission failed', timeout_seconds=INGEST_TIMEOUT_SECONDS,
            json={'session_id': session_id, 'filename': filename, 'content_base64': b64encode(content).decode('ascii'),
                  'language': language, 'recorded_at': recorded_at.isoformat() if recorded_at else None}))

    def get_ingestion_job(self, job_id: str, session_id: str) -> IngestionJob:
        return IngestionJob.model_validate(self._request('GET', f'memories/ingest/jobs/{job_id}', IngestionJob,
            error_message='Upload status lookup failed', params={'session_id': session_id}))

    def list_memories(self) -> list[MemoryView]:
        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = client.get("memories")
                response.raise_for_status()
                return [
                    MemoryView.model_validate(item)
                    for item in response.json()
                ]
        except httpx.HTTPStatusError as exception:
            raise _api_error(exception.response, "Memory lookup failed") from exception
        except (httpx.HTTPError, ValueError, ValidationError) as exception:
            raise ApiClientError("Memory lookup failed") from exception

    def delete_transcript(self, transcript_id: str) -> TranscriptDeletionResult:
        return TranscriptDeletionResult.model_validate(
            self._request(
                "DELETE",
                f"transcripts/{transcript_id}",
                TranscriptDeletionResult,
                error_message="Transcript deletion failed",
            )
        )

    def list_memory_gaps(self, *, include_closed: bool = False) -> list[MemoryGapView]:
        result = MemoryGapListResponse.model_validate(
            self._request(
                "GET",
                "memory-gaps",
                MemoryGapListResponse,
                error_message="Memory gap lookup failed",
                params={"include_closed": str(include_closed).lower()},
            )
        )
        return result.items

    def reconstruct_memory_gap(
        self,
        gap_id: str,
        *,
        web_search_consent: bool = False,
    ) -> MemoryGapReconstructionResult:
        return MemoryGapReconstructionResult.model_validate(
            self._request(
                "POST",
                f"memory-gaps/{gap_id}/reconstruct",
                MemoryGapReconstructionResult,
                error_message="Memory gap reconstruction failed",
                timeout_seconds=INGEST_TIMEOUT_SECONDS,
                json={"web_search_consent": web_search_consent},
            )
        )

    def add_memory_gap_clue(self, gap_id: str, clue: str) -> MemoryGapView:
        return MemoryGapView.model_validate(
            self._request(
                "POST",
                f"memory-gaps/{gap_id}/clues",
                MemoryGapView,
                error_message="Memory gap clue update failed",
                json={"clue": clue},
            )
        )

    def resolve_memory_gap(
        self,
        gap_id: str,
        candidate_id: str,
        *,
        user_confirmed: bool,
    ) -> MemoryGapResolutionResult:
        return MemoryGapResolutionResult.model_validate(
            self._request(
                "POST",
                f"memory-gaps/{gap_id}/resolve",
                MemoryGapResolutionResult,
                error_message="Memory gap resolution failed",
                json={
                    "candidate_id": candidate_id,
                    "user_confirmed": user_confirmed,
                },
            )
        )

    def dismiss_memory_gap(self, gap_id: str) -> MemoryGapView:
        return MemoryGapView.model_validate(
            self._request(
                "POST",
                f"memory-gaps/{gap_id}/dismiss",
                MemoryGapView,
                error_message="Memory gap dismissal failed",
            )
        )

    def chat(
        self,
        *,
        session_id: str,
        question: str,
        top_k: int = 3,
    ) -> ChatResult:
        return ChatResult.model_validate(
            self._request(
                "POST",
                "chat",
                ChatResult,
                error_message="Chat request failed",
                timeout_seconds=CHAT_TIMEOUT_SECONDS,
                json={
                    "session_id": session_id,
                    "question": question,
                    "top_k": top_k,
                },
            )
        )
    def get_timeline(
        self,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> TimelineResult:
        return TimelineResult.model_validate(
            self._request(
                "POST",
                "timeline",
                TimelineResult,
                error_message="Timeline lookup failed",
                json={
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                },
            )
        )

    def generate_autobiography(
        self,
        *,
        title: str,
        request: str,
        target_period: str | None,
        target_topics: list[str],
        chapter_count: int,
        proceed_with_unresolved_gaps: bool = False,
    ) -> AutobiographyResult:
        return AutobiographyResult.model_validate(
            self._request(
                "POST",
                "autobiographies",
                AutobiographyResult,
                error_message="Autobiography generation failed",
                json={
                    "title": title,
                    "request": request,
                    "target_period": target_period,
                    "target_topics": target_topics,
                    "chapter_count": chapter_count,
                    "proceed_with_unresolved_gaps": proceed_with_unresolved_gaps,
                },
            )
        )

    def submit_chat_job(self, *, session_id: str, question: str, top_k: int = 3) -> ChatJob:
        return ChatJob.model_validate(self._request(
            "POST", "chat/jobs", ChatJob, error_message="Chat submission failed",
            json={"session_id": session_id, "question": question, "top_k": top_k},
        ))

    def get_chat_job(self, job_id: str, session_id: str) -> ChatJob:
        return ChatJob.model_validate(self._request(
            "GET", f"chat/jobs/{job_id}", ChatJob, error_message="Chat status lookup failed",
            params={"session_id": session_id},
        ))

    def check_autobiography_gaps(
        self,
        *,
        title: str,
        request: str,
        target_period: str | None,
        target_topics: list[str],
        chapter_count: int,
    ) -> AutobiographyGapCheckResult:
        return AutobiographyGapCheckResult.model_validate(
            self._request(
                "POST",
                "autobiographies/gap-check",
                AutobiographyGapCheckResult,
                error_message="Autobiography memory gap check failed",
                json={
                    "title": title,
                    "request": request,
                    "target_period": target_period,
                    "target_topics": target_topics,
                    "chapter_count": chapter_count,
                },
            )
        )
