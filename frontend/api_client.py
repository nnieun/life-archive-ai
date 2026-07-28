"""Typed HTTP client used by every Streamlit page."""

from __future__ import annotations

from base64 import b64encode
from datetime import date, datetime
from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_API_URL: Final = "http://127.0.0.1:8000/api/v1"


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


class IngestionResult(ApiModel):
    transcript_id: str
    filename: str
    segment_count: int
    memory_count: int
    indexed_memory_count: int
    memory_ids: list[str]


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


class QAValidation(ApiModel):
    stage: str
    passed: bool
    reason: str


class ChatResult(ApiModel):
    session_id: str
    question: str
    retrieved_memory_ids: list[str]
    final_answer: str
    citations: list[Citation]
    validation_result: QAValidation
    retry_count: int
    error: str | None = None


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


class ApiClientError(RuntimeError):
    """User-safe backend communication failure."""


class LifeArchiveApiClient:
    """Small synchronous client suitable for Streamlit reruns."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        timeout_seconds: float = 60.0,
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
        **kwargs: object,
    ) -> ApiModel:
        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = client.request(method, path, **kwargs)
                response.raise_for_status()
                return response_model.model_validate(response.json())
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
                json=payload,
            )
        )

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
        except (httpx.HTTPError, ValueError, ValidationError) as exception:
            raise ApiClientError("Memory lookup failed") from exception

    def chat(
        self,
        *,
        session_id: str,
        question: str,
        top_k: int = 5,
    ) -> ChatResult:
        return ChatResult.model_validate(
            self._request(
                "POST",
                "chat",
                ChatResult,
                error_message="Chat request failed",
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
                },
            )
        )
