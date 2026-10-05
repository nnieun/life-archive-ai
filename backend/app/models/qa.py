"""Validated state and structured outputs for grounded question answering."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.storage.models import CitationRecord
from backend.app.models.qa_diagnostics import QAStepDiagnostic, QASchemaIssue


class QAModel(BaseModel):
    """Strict immutable base for Q&A data."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class QAEvidence(QAModel):
    """One SQLite memory and its traceable transcript sources."""

    memory_id: str
    transcript_id: str
    title: str
    summary: str
    emotion: str | None = None
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


class EvidenceSelection(QAModel):
    """Local model output with one authoritative selection, not two flags."""

    reason: str = Field(min_length=1, description="Why the selected memories answer the question, or why none can.")
    selected_memory_ids: list[str] = Field(
        description="Exact IDs from supplied evidence that support an answer. Empty if insufficient."
    )


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


class AnswerProposal(QAModel):
    """An empty claim list means the retrieved evidence cannot answer."""

    reason: str = Field(min_length=1)
    claims: list[CitedClaim]


class AnswerVerification(QAModel):
    """Structured citation and support verification result."""

    passed: bool
    reason: str = Field(min_length=1)
    unsupported_claim_indexes: list[Annotated[int, Field(ge=0)]] = Field(
        default_factory=list
    )
    missing_required_memory_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unsupported_claims(self) -> AnswerVerification:
        if self.passed and (self.unsupported_claim_indexes or self.missing_required_memory_ids):
            raise ValueError("passed verification cannot list unsupported claims or missing memories")
        return self


class QAQueryPlan(QAModel):
    """Deterministic retrieval and coverage requirements shared by graph nodes."""

    mode: Literal["standard", "entity_overview", "composite"] = "standard"
    matched_people: list[str] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list)
    required_memory_ids: list[str] = Field(default_factory=list)


class QAValidationResult(QAModel):
    """Latest deterministic or model-assisted graph validation decision."""

    stage: Literal["evidence", "answer"]
    passed: bool
    reason: str
    failure_code: str | None = None
    exception_type: str | None = None
    schema_issues: list[QASchemaIssue] = Field(default_factory=list)


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
    elapsed_ms: float = Field(default=0, ge=0)
    cache_hit: bool = False
    steps: list[QAStepDiagnostic] = Field(default_factory=list)


class QAState(TypedDict):
    """Shared LangGraph state for the bounded grounded-Q&A workflow."""

    session_id: str
    question: str
    top_k: int
    query_plan: QAQueryPlan
    retrieved_memory_ids: list[str]
    selected_evidence: list[QAEvidence]
    answer_draft: GroundedAnswerDraft | None
    draft_answer: str
    citations: list[CitationRecord]
    validation_result: QAValidationResult
    final_answer: str
    retry_count: int
    error: str | None
    diagnostic_steps: list[QAStepDiagnostic]
