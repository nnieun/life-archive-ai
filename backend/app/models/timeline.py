"""Validated chronological timeline results."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.models.memory import DatePrecision
from backend.app.storage.models import CitationRecord, MemoryStatus


class TimelineModel(BaseModel):
    """Strict immutable base for timeline output."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TimelineEvent(TimelineModel):
    """One active, traceable memory displayed on the timeline."""

    memory_id: str
    transcript_id: str
    title: str
    description: str
    event_date: str | None
    date_precision: DatePrecision
    date_label: str
    people: list[str] = Field(default_factory=list)
    location: str | None = None
    emotion: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_notes: str | None = None
    status: MemoryStatus
    citations: list[CitationRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_citations(self) -> TimelineEvent:
        if any(
            citation.memory_id != self.memory_id
            or citation.transcript_id != self.transcript_id
            for citation in self.citations
        ):
            raise ValueError("timeline citations must match the memory")
        return self


class TimelineResult(TimelineModel):
    """Dated and unknown-date events returned as separate collections."""

    events: list[TimelineEvent]
    undated_events: list[TimelineEvent]
    start_date: date | None = None
    end_date: date | None = None

    @property
    def total(self) -> int:
        return len(self.events) + len(self.undated_events)
