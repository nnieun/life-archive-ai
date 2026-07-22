"""Validated transcript loader inputs and outputs."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TranscriptLoadRequest(BaseModel):
    """Metadata supplied when loading one immutable STT transcript."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: Path = Field(exclude=True, repr=False)
    uploaded_at: datetime
    recorded_at: datetime | None = None
    recording_id: str | None = None
    language: str | None = None
    source_type: str = "stt_text"

    @field_validator("uploaded_at", "recorded_at")
    @classmethod
    def require_timezone(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        """Reject ambiguous timestamps while allowing an unknown recording time."""
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamps must include a timezone")
        return value

    @field_validator("recording_id", "language", "source_type")
    @classmethod
    def reject_blank_metadata(cls, value: str | None) -> str | None:
        """Keep optional metadata unknown instead of accepting blank values."""
        if value is not None and not value.strip():
            raise ValueError("metadata values must not be blank")
        return value


class LoadedTranscript(BaseModel):
    """Original and normalized transcript representations for persistence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_id: str
    filename: str = Field(repr=False)
    recording_id: str | None = None
    language: str | None = None
    source_type: str
    uploaded_at: datetime
    recorded_at: datetime | None = None
    content_hash: str
    raw_content: str = Field(repr=False)
    normalized_content: str = Field(repr=False)
