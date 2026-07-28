"""Validated state and Structured Outputs for autobiography generation."""

from __future__ import annotations

from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.models.qa import CitedClaim, QAEvidence
from backend.app.models.timeline import TimelineEvent
from backend.app.storage.models import (
    AutobiographyContent,
    AutobiographyRecord,
    CitationRecord,
)


class AutobiographyModel(BaseModel):
    """Strict immutable base for autobiography workflow data."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AutobiographyInput(AutobiographyModel):
    autobiography_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    request: str = Field(min_length=1)
    target_period: str | None = None
    target_topics: list[str] = Field(default_factory=list)
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


class ChapterPlanItem(AutobiographyModel):
    title: str = Field(min_length=1)
    focus: str = Field(min_length=1)
    memory_ids: list[str] = Field(min_length=1)

    @field_validator("memory_ids")
    @classmethod
    def require_unique_memory_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("chapter memory_ids must be unique")
        return value


class ChapterPlan(AutobiographyModel):
    chapters: list[ChapterPlanItem] = Field(min_length=1, max_length=3)


class ChapterDraft(AutobiographyModel):
    title: str = Field(min_length=1)
    paragraphs: list[CitedClaim] = Field(min_length=1)


class ChapterReview(AutobiographyModel):
    passed: bool
    reason: str = Field(min_length=1)
    unsupported_paragraph_indexes: list[
        Annotated[int, Field(ge=0)]
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unsupported(self) -> ChapterReview:
        if self.passed and self.unsupported_paragraph_indexes:
            raise ValueError("passed review cannot list unsupported paragraphs")
        return self


class AutobiographyGenerationResult(AutobiographyModel):
    autobiography: AutobiographyRecord
    completed: bool
    retrieved_memory_ids: list[str]
    citations: list[CitationRecord]
    retry_count: int = Field(ge=0)
    error: str | None = None


class AutobiographyState(TypedDict):
    """Shared LangGraph state for a maximum-three-chapter workflow."""

    autobiography_id: str
    title: str
    request: str
    retrieval_query: str
    target_period: str | None
    target_topics: list[str]
    chapter_count: int
    top_k: int
    retrieved_memory_ids: list[str]
    evidence: list[QAEvidence]
    timeline: list[TimelineEvent]
    chapter_plan: list[ChapterPlanItem]
    current_chapter_index: int
    current_draft: ChapterDraft | None
    chapter_drafts: list[ChapterDraft]
    review_result: ChapterReview | None
    citations: list[CitationRecord]
    final_content: AutobiographyContent | None
    retry_count: int
    chapter_retry_count: int
    error: str | None
