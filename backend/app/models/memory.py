"""Structured memory extraction models."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LOW_CONFIDENCE_THRESHOLD = 0.5

_YEAR_PATTERN = re.compile(r"^\d{4}$")
_MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_DAY_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-\d{2}$")


class DatePrecision(StrEnum):
    """How precisely an event date is supported by transcript evidence."""

    EXACT = "exact"
    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


def event_date_matches_precision(
    event_date: str | None,
    date_precision: DatePrecision,
) -> bool:
    """Check one event_date against the precision that is claimed for it."""

    if date_precision is DatePrecision.UNKNOWN:
        return event_date is None
    if event_date is None:
        return False
    if date_precision is DatePrecision.YEAR:
        return _YEAR_PATTERN.fullmatch(event_date) is not None
    if date_precision is DatePrecision.MONTH:
        return _MONTH_PATTERN.fullmatch(event_date) is not None
    if date_precision is DatePrecision.DAY:
        if _DAY_PATTERN.fullmatch(event_date) is None:
            return False
        try:
            datetime.strptime(event_date, "%Y-%m-%d")
        except ValueError:
            return False
        return True
    if date_precision is DatePrecision.EXACT:
        try:
            parsed = datetime.fromisoformat(event_date)
        except ValueError:
            return False
        return parsed.tzinfo is not None and parsed.utcoffset() is not None
    return bool(event_date.strip())


class ExtractedMemory(BaseModel):
    """One model-proposed memory before transcript context is attached."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1, repr=False)
    people: list[str]
    location: str | None
    event_date: str | None
    date_precision: DatePrecision
    emotion: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_start_offset: int = Field(ge=0)
    evidence_end_offset: int = Field(gt=0)
    uncertainty_notes: str | None

    @field_validator("title", "summary", "location", "emotion", "uncertainty_notes")
    @classmethod
    def reject_blank_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    @field_validator("people")
    @classmethod
    def validate_people(cls, value: list[str]) -> list[str]:
        if any(not person.strip() for person in value):
            raise ValueError("people must not contain blank names")
        if len(set(value)) != len(value):
            raise ValueError("people must not contain duplicate names")
        return value

    @model_validator(mode="after")
    def validate_evidence_and_uncertainty(self) -> ExtractedMemory:
        if self.evidence_end_offset <= self.evidence_start_offset:
            raise ValueError("evidence_end_offset must follow evidence_start_offset")
        if (
            self.confidence < LOW_CONFIDENCE_THRESHOLD
            and self.uncertainty_notes is None
        ):
            raise ValueError("low-confidence memories require uncertainty_notes")
        if (
            self.date_precision is DatePrecision.APPROXIMATE
            and self.uncertainty_notes is None
        ):
            raise ValueError("approximate dates require uncertainty_notes")
        return self

    @model_validator(mode="after")
    def validate_event_date(self) -> ExtractedMemory:
        if self.date_precision is DatePrecision.UNKNOWN:
            if self.event_date is not None:
                raise ValueError("unknown date precision requires a null event_date")
            return self
        if self.event_date is None:
            raise ValueError("known date precision requires event_date")
        if not event_date_matches_precision(self.event_date, self.date_precision):
            raise ValueError("event_date does not match date_precision")
        return self


class MemoryExtractionBatch(BaseModel):
    """Strict Structured Output envelope for zero or more memories."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memories: list[ExtractedMemory]


class MemoryCorrection(BaseModel):
    """Human-supplied changes that replace one stored memory.

    Only the fields present in ``model_fields_set`` are applied, so an explicit
    ``null`` clears a value while an omitted field inherits the original.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str | None = Field(default=None, min_length=1)
    summary: str | None = Field(default=None, min_length=1, repr=False)
    people: list[str] | None = None
    location: str | None = None
    event_date: str | None = None
    date_precision: DatePrecision | None = None
    emotion: str | None = None
    uncertainty_notes: str | None = None

    @field_validator("people")
    @classmethod
    def validate_people(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if any(not person.strip() for person in value):
            raise ValueError("people must not contain blank names")
        if len(set(value)) != len(value):
            raise ValueError("people must not contain duplicate names")
        return value

    @model_validator(mode="after")
    def validate_correction(self) -> MemoryCorrection:
        changed = self.model_fields_set
        if not changed:
            raise ValueError("a correction must change at least one field")
        if ("event_date" in changed) != ("date_precision" in changed):
            raise ValueError("event_date and date_precision must be corrected together")
        if "date_precision" in changed and not event_date_matches_precision(
            self.event_date,
            self.date_precision or DatePrecision.UNKNOWN,
        ):
            raise ValueError("event_date does not match date_precision")
        return self
