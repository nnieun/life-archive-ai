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

        if self.date_precision is DatePrecision.YEAR:
            valid = _YEAR_PATTERN.fullmatch(self.event_date) is not None
        elif self.date_precision is DatePrecision.MONTH:
            valid = _MONTH_PATTERN.fullmatch(self.event_date) is not None
        elif self.date_precision is DatePrecision.DAY:
            valid = _DAY_PATTERN.fullmatch(self.event_date) is not None
            if valid:
                try:
                    datetime.strptime(self.event_date, "%Y-%m-%d")
                except ValueError:
                    valid = False
        elif self.date_precision is DatePrecision.EXACT:
            try:
                parsed = datetime.fromisoformat(self.event_date)
                valid = parsed.tzinfo is not None and parsed.utcoffset() is not None
            except ValueError:
                valid = False
        else:
            valid = bool(self.event_date.strip())

        if not valid:
            raise ValueError("event_date does not match date_precision")
        return self


class MemoryExtractionBatch(BaseModel):
    """Strict Structured Output envelope for zero or more memories."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memories: list[ExtractedMemory]
