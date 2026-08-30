"""Structured memory extraction models."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LOW_CONFIDENCE_THRESHOLD = 0.5

_YEAR_PATTERN = re.compile(r"^\d{4}$")
_MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_DAY_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-\d{2}$")
_FLEXIBLE_MONTH_PATTERN = re.compile(r"^(\d{4})[./-](\d{1,2})$")
_FLEXIBLE_DAY_PATTERN = re.compile(
    r"^(\d{4})[./-](\d{1,2})[./-](\d{1,2})$"
)
_KOREAN_YEAR_PATTERN = re.compile(r"^(\d{4})년$")
_KOREAN_MONTH_PATTERN = re.compile(r"^(\d{4})년\s*(\d{1,2})월$")
_KOREAN_DAY_PATTERN = re.compile(
    r"^(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일$"
)


def _normalize_event_date_text(value: str) -> str:
    """Normalize supported Korean date notation and surrounding whitespace."""

    stripped = value.strip()
    match = _KOREAN_DAY_PATTERN.fullmatch(stripped)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    match = _KOREAN_MONTH_PATTERN.fullmatch(stripped)
    if match:
        year, month = match.groups()
        return f"{year}-{int(month):02d}"
    match = _KOREAN_YEAR_PATTERN.fullmatch(stripped)
    if match:
        return match.group(1)
    match = _FLEXIBLE_DAY_PATTERN.fullmatch(stripped)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    match = _FLEXIBLE_MONTH_PATTERN.fullmatch(stripped)
    if match:
        year, month = match.groups()
        return f"{year}-{int(month):02d}"
    return stripped


class DatePrecision(StrEnum):
    """How precisely an event date is supported by transcript evidence."""

    EXACT = "exact"
    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


def _canonical_date_and_precision(
    event_date: str,
) -> tuple[str, DatePrecision] | None:
    """Derive a canonical supported date and its actual precision."""

    normalized_date = _normalize_event_date_text(event_date)
    if _YEAR_PATTERN.fullmatch(normalized_date):
        return normalized_date, DatePrecision.YEAR
    if _MONTH_PATTERN.fullmatch(normalized_date):
        return normalized_date, DatePrecision.MONTH
    if _DAY_PATTERN.fullmatch(normalized_date):
        try:
            datetime.strptime(normalized_date, "%Y-%m-%d")
        except ValueError:
            return None
        return normalized_date, DatePrecision.DAY
    try:
        parsed = datetime.fromisoformat(normalized_date)
    except ValueError:
        return None
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        return parsed.isoformat(), DatePrecision.EXACT
    # A date-time without a timezone cannot satisfy exact precision. Retain
    # only its supported calendar date instead of inventing a timezone.
    return parsed.date().isoformat(), DatePrecision.DAY


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
    """One validated memory with chunk-relative evidence offsets."""

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

    @model_validator(mode="before")
    @classmethod
    def normalize_event_date_precision(cls, values: object) -> object:
        """Derive precision from a valid date instead of trusting the model label."""

        if not isinstance(values, Mapping):
            return values
        event_date = values.get("event_date")
        date_precision = values.get("date_precision")
        if not isinstance(event_date, str) or date_precision in (
            DatePrecision.APPROXIMATE,
            DatePrecision.APPROXIMATE.value,
        ):
            return values

        canonical = _canonical_date_and_precision(event_date)
        if canonical is None:
            return values

        normalized_date, normalized_precision = canonical
        normalized_values = dict(values)
        normalized_values["event_date"] = normalized_date
        normalized_values["date_precision"] = normalized_precision
        return normalized_values

    @field_validator("title", "summary", "location", "emotion", "uncertainty_notes")
    @classmethod
    def reject_blank_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    @field_validator("event_date", mode="before")
    @classmethod
    def normalize_korean_date(cls, value: str | None) -> str | None:
        """Normalize explicit Korean year/month/day notation to ISO text."""
        if value is None or not isinstance(value, str):
            return value
        return _normalize_event_date_text(value)

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


class MemoryExtractionProposal(BaseModel):
    """Model-facing memory proposal with verbatim transcript evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1, repr=False)
    people: list[str]
    location: str | None
    event_date: str | None
    date_precision: DatePrecision
    emotion: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str = Field(
        min_length=1,
        repr=False,
        description="Exact contiguous quote copied from segment_content",
    )
    uncertainty_notes: str | None

    @field_validator("evidence_text")
    @classmethod
    def reject_blank_evidence(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence_text must not be blank")
        return value


class MemoryExtractionProposalBatch(BaseModel):
    """Strict Structured Output envelope returned by the extraction model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memories: list[MemoryExtractionProposal]


class LocalizedMemoryFields(BaseModel):
    """Korean display fields returned after evidence validation is complete."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_index: int = Field(ge=0)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1, repr=False)
    people: list[str]
    location: str | None
    emotion: str | None
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


class MemoryLocalizationBatch(BaseModel):
    """Strict envelope for optional Korean display-field localization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memories: list[LocalizedMemoryFields]


class MemoryExtractionBatch(BaseModel):
    """Validated internal batch whose evidence offsets are already resolved."""

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
