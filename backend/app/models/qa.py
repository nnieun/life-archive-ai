"""Validated state and structured outputs for grounded question answering."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.storage.models import CitationRecord


class QAModel(BaseModel):
    """Strict immutable base for Q&A data."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class QAEvidence(QAModel):
    """One SQLite memory and its traceable transcript sources."""

    memory_id: str
    transcript_id: str
    title: str
    summary: str
    people: list[str] = Field(default_factory=list)
    location: str | None = None
    event_date: str | None = None
    uncertainty_notes: str | None = None
    sources: list[CitationRecord] = Field(min_length=1)


class EvidenceAssessment(QAModel):
    """Structured judgment about whether retrieved evidence can answer a question."""

    sufficient: bool
    reason: str = Field(min_length=1)
    selected_memory_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selection(self) -> EvidenceAssessment:
        if self.sufficient and not self.selected_memory_ids:
            raise ValueError("sufficient evidence requires selected_memory_ids")
        if not self.sufficient and self.selected_memory_ids:
            raise ValueError("insufficient evidence cannot select memories")
        return self


class CitedClaim(QAModel):
    """One answer claim supported by one or more retrieved memories."""

    text: str = Field(min_length=1)
    memory_ids: list[str] = Field(min_length=1)

    @field_validator("memory_ids")
    @classmethod
    def require_unique_memory_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("claim memory_ids must be unique")
        return value


class GroundedAnswerDraft(QAModel):
    """Structured answer whose every claim carries explicit evidence IDs."""

    claims: list[CitedClaim] = Field(min_length=1)


class AnswerVerification(QAModel):
    """Structured citation and support verification result."""

    passed: bool
    reason: str = Field(min_length=1)
    unsupported_claim_indexes: list[Annotated[int, Field(ge=0)]] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_unsupported_claims(self) -> AnswerVerification:
        if self.passed and self.unsupported_claim_indexes:
            raise ValueError("passed verification cannot list unsupported claims")
        return self


class QAValidationResult(QAModel):
    """Latest deterministic or model-assisted graph validation decision."""

    stage: Literal["evidence", "answer"]
    passed: bool
    reason: str


class QAResult(QAModel):
    """Public result returned after graph execution and persistence."""

    session_id: str
    question: str
    retrieved_memory_ids: list[str]
    final_answer: str
    citations: list[CitationRecord]
    validation_result: QAValidationResult
    retry_count: int = Field(ge=0, le=1)
    error: str | None = None


class QAState(TypedDict):
    """Shared LangGraph state for the bounded grounded-Q&A workflow."""

    session_id: str
    question: str
    top_k: int
    retrieved_memory_ids: list[str]
    selected_evidence: list[QAEvidence]
    answer_draft: GroundedAnswerDraft | None
    draft_answer: str
    citations: list[CitationRecord]
    validation_result: QAValidationResult
    final_answer: str
    retry_count: int
    error: str | None
