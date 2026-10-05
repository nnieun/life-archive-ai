"""Validated memory-gap and reconstruction-candidate contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class GapModel(BaseModel):
    """Strict immutable base shared by gap domain records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryGapType(StrEnum):
    MISSING_LOCATION = "MISSING_LOCATION"
    MISSING_PERSON = "MISSING_PERSON"
    MISSING_DATE = "MISSING_DATE"
    UNCERTAIN_EVENT = "UNCERTAIN_EVENT"
    CONFLICTING_FACT = "CONFLICTING_FACT"
    WEAK_PROVENANCE = "WEAK_PROVENANCE"


class MemoryGapStatus(StrEnum):
    OPEN = "OPEN"
    SEARCHING = "SEARCHING"
    CANDIDATE_FOUND = "CANDIDATE_FOUND"
    WAITING_USER = "WAITING_USER"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class MemoryGapSourceType(StrEnum):
    MEMORY = "MEMORY"
    TRANSCRIPT_SEGMENT = "TRANSCRIPT_SEGMENT"
    EXTERNAL = "EXTERNAL"


class MemoryGapSearchSourceType(StrEnum):
    """Read-only evidence kinds returned by reconstruction search tools."""

    MEMORY = "MEMORY"
    TRANSCRIPT_SEGMENT = "TRANSCRIPT_SEGMENT"
    MEMORY_GAP = "MEMORY_GAP"
    EXTERNAL = "EXTERNAL"


class MemoryGapCandidateStatus(StrEnum):
    PROPOSED = "PROPOSED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class MemoryGapCandidateRelation(StrEnum):
    SUPPORTS = "SUPPORTS"
    RELATED = "RELATED"
    CONFLICTS = "CONFLICTS"
    UNKNOWN = "UNKNOWN"


class ExternalSource(GapModel):
    """Public provenance retained for a future web-derived candidate."""

    url: str = Field(min_length=1, pattern=r"^https?://")
    title: str = Field(min_length=1)
    source_domain: str = Field(min_length=1)
    published_date: str | None = None
    snippet: str = Field(min_length=1, repr=False)
    retrieved_at: AwareDatetime

    @field_validator("title", "source_domain", "published_date", "snippet")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text fields must not be blank")
        return value


class MemoryGapCreate(GapModel):
    gap_id: str = Field(min_length=1)
    memory_id: str | None = Field(default=None, min_length=1)
    gap_type: MemoryGapType
    clue_text: str = Field(min_length=1, repr=False)
    missing_field: str | None = Field(default=None, min_length=1)
    period_start: str | None = Field(default=None, min_length=1)
    period_end: str | None = Field(default=None, min_length=1)
    location: str | None = Field(default=None, min_length=1)
    people: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    status: MemoryGapStatus = MemoryGapStatus.OPEN
    source_type: MemoryGapSourceType
    source_id: str = Field(min_length=1)
    web_search_consent: bool = False
    user_clues: list[str] = Field(default_factory=list)

    @field_validator("clue_text")
    @classmethod
    def reject_blank_clue(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("clue_text must not be blank")
        return value

    @field_validator("people", "user_clues")
    @classmethod
    def validate_text_lists(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("text lists must not contain blank values")
        if len(value) != len(set(value)):
            raise ValueError("text lists must not contain duplicates")
        return value


class MemoryGapRecord(MemoryGapCreate):
    resolved_candidate_id: str | None = None
    resolved_memory_id: str | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    resolved_at: AwareDatetime | None = None


class MemoryGapUpdate(GapModel):
    status: MemoryGapStatus | None = None
    web_search_consent: bool | None = None
    user_clues: list[str] | None = None
    resolved_candidate_id: str | None = None
    resolved_memory_id: str | None = None
    resolved_at: AwareDatetime | None = None

    @field_validator("user_clues")
    @classmethod
    def validate_user_clues(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if any(not clue.strip() for clue in value):
            raise ValueError("user_clues must not contain blank values")
        if len(value) != len(set(value)):
            raise ValueError("user_clues must not contain duplicates")
        return value


class MemoryGapCandidateCreate(GapModel):
    candidate_id: str = Field(min_length=1)
    gap_id: str = Field(min_length=1)
    value: str = Field(min_length=1)
    explanation: str = Field(min_length=1, repr=False)
    deterministic_score: float = Field(ge=0.0, le=1.0)
    llm_relation: MemoryGapCandidateRelation = MemoryGapCandidateRelation.UNKNOWN
    supporting_source_ids: list[str] = Field(default_factory=list)
    external_sources: list[ExternalSource] = Field(default_factory=list)
    status: MemoryGapCandidateStatus = MemoryGapCandidateStatus.PROPOSED

    @field_validator("value", "explanation")
    @classmethod
    def reject_blank_candidate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate text must not be blank")
        return value

    @field_validator("supporting_source_ids")
    @classmethod
    def validate_source_ids(cls, value: list[str]) -> list[str]:
        if any(not source_id.strip() for source_id in value):
            raise ValueError("supporting_source_ids must not contain blanks")
        if len(value) != len(set(value)):
            raise ValueError("supporting_source_ids must be unique")
        return value


class MemoryGapCandidateRecord(MemoryGapCandidateCreate):
    created_at: AwareDatetime
    updated_at: AwareDatetime


class MemoryGapSearchSource(GapModel):
    """One privacy-safe, local search result exposed to the reconstruction model."""

    source_id: str = Field(min_length=1)
    source_type: MemoryGapSearchSourceType
    title: str = Field(min_length=1)
    content: str = Field(min_length=1, repr=False)
    score: float = Field(ge=0.0, le=1.0)
    memory_id: str | None = Field(default=None, min_length=1)
    transcript_id: str | None = Field(default=None, min_length=1)
    event_date: str | None = Field(default=None, min_length=1)
    location: str | None = Field(default=None, min_length=1)
    people: list[str] = Field(default_factory=list)
    url: str | None = Field(default=None, pattern=r"^https?://")
    source_domain: str | None = None
    published_date: str | None = None

    @field_validator("title", "content", "event_date", "location")
    @classmethod
    def reject_blank_search_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("search text fields must not be blank")
        return value

    @field_validator("people")
    @classmethod
    def validate_search_people(cls, value: list[str]) -> list[str]:
        if any(not person.strip() for person in value):
            raise ValueError("people must not contain blank values")
        if len(value) != len(set(value)):
            raise ValueError("people must not contain duplicates")
        return value


class MemoryGapToolPayload(GapModel):
    """Validated JSON payload shared between a ToolNode and graph state."""

    tool_name: Literal[
        "search_memory",
        "search_uploaded_documents",
        "search_web",
        "search_memory_gaps",
        "request_more_clues",
    ]
    query: str = Field(min_length=1)
    sources: list[MemoryGapSearchSource] = Field(default_factory=list)
    needs_user_input: bool = False
    question: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_question(self) -> MemoryGapToolPayload:
        if self.needs_user_input != (self.question is not None):
            raise ValueError("question is required only when user input is needed")
        return self


class MemoryGapCandidateProposal(GapModel):
    """Strict LLM proposal that must point back to exact returned evidence."""

    value: str = Field(min_length=1, max_length=500)
    evidence_text: str = Field(min_length=1, max_length=2000, repr=False)
    explanation: str = Field(min_length=1, max_length=1000, repr=False)
    llm_relation: MemoryGapCandidateRelation
    supporting_source_ids: list[str] = Field(min_length=1, max_length=3)

    @field_validator(
        "value",
        "evidence_text",
        "explanation",
    )
    @classmethod
    def reject_blank_proposal_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("proposal text must not be blank")
        return value

    @field_validator("supporting_source_ids")
    @classmethod
    def validate_proposal_sources(cls, value: list[str]) -> list[str]:
        if any(not source_id.strip() for source_id in value):
            raise ValueError("supporting source ids must not be blank")
        if len(value) != len(set(value)):
            raise ValueError("supporting source ids must be unique")
        return value


class MemoryGapCandidateProposalBatch(GapModel):
    """Structured Output envelope for at most three ranked candidates."""

    candidates: list[MemoryGapCandidateProposal] = Field(max_length=3)


class MemoryGapReconstructionResult(GapModel):
    """Public result of one bounded internal reconstruction run."""

    gap: MemoryGapRecord
    candidates: list[MemoryGapCandidateRecord]
    searched_tools: list[str]
    tool_call_count: int = Field(ge=0)
    needs_more_clues: bool = False
    user_question: str | None = None
    message: str = Field(min_length=1)


class MemoryGapResolutionResult(GapModel):
    """Result of a server-authorized, user-confirmed candidate resolution."""

    gap: MemoryGapRecord
    candidate: MemoryGapCandidateRecord
    resolved_memory_id: str = Field(min_length=1)
