"""Pydantic-validated records stored in SQLite."""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from backend.app.models.memory import DatePrecision


class StorageModel(BaseModel):
    """Strict immutable base for persistence inputs and outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    CORRECTED = "corrected"
    DELETED = "deleted"


class AutobiographyStatus(StrEnum):
    DRAFT = "draft"
    COMPLETED = "completed"
    DELETED = "deleted"


class TranscriptRecord(StorageModel):
    transcript_id: str
    filename: str = Field(repr=False)
    recording_id: str | None = None
    language: str | None = None
    source_type: str
    uploaded_at: AwareDatetime
    recorded_at: AwareDatetime | None = None
    content_hash: str
    raw_content: str = Field(repr=False)
    normalized_content: str = Field(repr=False)
    created_at: AwareDatetime
    updated_at: AwareDatetime
    deleted_at: AwareDatetime | None = None


class TranscriptMetadataUpdate(StorageModel):
    recording_id: str | None = None
    language: str | None = None
    source_type: str | None = Field(default=None, min_length=1)
    recorded_at: AwareDatetime | None = None


class TranscriptSegmentCreate(StorageModel):
    segment_id: str
    transcript_id: str
    chunk_index: int = Field(ge=0)
    content: str = Field(repr=False)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> TranscriptSegmentCreate:
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must not precede start_offset")
        return self


class TranscriptSegmentRecord(TranscriptSegmentCreate):
    created_at: AwareDatetime
    updated_at: AwareDatetime
    deleted_at: AwareDatetime | None = None


class MemoryCreate(StorageModel):
    memory_id: str
    transcript_id: str
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1, repr=False)
    people: list[str] = Field(default_factory=list)
    location: str | None = None
    event_date: str | None = None
    date_precision: DatePrecision = DatePrecision.UNKNOWN
    emotion: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_notes: str | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    supersedes_memory_id: str | None = None

    @model_validator(mode="after")
    def validate_event_date(self) -> MemoryCreate:
        if (self.event_date is None) != (
            self.date_precision is DatePrecision.UNKNOWN
        ):
            raise ValueError("event_date and date_precision must agree")
        return self


class MemoryUpdate(StorageModel):
    title: str | None = Field(default=None, min_length=1)
    summary: str | None = Field(default=None, min_length=1, repr=False)
    people: list[str] | None = None
    location: str | None = None
    event_date: str | None = None
    date_precision: DatePrecision | None = None
    emotion: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty_notes: str | None = None
    status: MemoryStatus | None = None
    supersedes_memory_id: str | None = None


class MemoryRecord(StorageModel):
    memory_id: str
    transcript_id: str
    title: str
    summary: str = Field(repr=False)
    people: list[str]
    location: str | None = None
    event_date: str | None = None
    date_precision: DatePrecision
    emotion: str | None = None
    confidence: float
    uncertainty_notes: str | None = None
    status: MemoryStatus
    supersedes_memory_id: str | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    deleted_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_event_date(self) -> MemoryRecord:
        if (self.event_date is None) != (
            self.date_precision is DatePrecision.UNKNOWN
        ):
            raise ValueError("event_date and date_precision must agree")
        return self


class MemorySourceCreate(StorageModel):
    memory_source_id: str
    memory_id: str
    transcript_id: str
    segment_id: str | None = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> MemorySourceCreate:
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must not precede start_offset")
        return self


class MemorySourceRecord(MemorySourceCreate):
    created_at: AwareDatetime
    updated_at: AwareDatetime


class CitationRecord(StorageModel):
    memory_id: str
    transcript_id: str
    segment_id: str | None = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> CitationRecord:
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must not precede start_offset")
        return self


class ConversationSessionCreate(StorageModel):
    session_id: str
    title: str | None = None


class ConversationSessionRecord(ConversationSessionCreate):
    created_at: AwareDatetime
    updated_at: AwareDatetime
    deleted_at: AwareDatetime | None = None


class ConversationMessageCreate(StorageModel):
    message_id: str
    session_id: str
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, repr=False)
    citations: list[CitationRecord] = Field(default_factory=list)


class ConversationMessageRecord(ConversationMessageCreate):
    created_at: AwareDatetime
    updated_at: AwareDatetime
    deleted_at: AwareDatetime | None = None


class AutobiographyChapter(StorageModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1, repr=False)
    citations: list[CitationRecord] = Field(default_factory=list)


class AutobiographyContent(StorageModel):
    chapters: list[AutobiographyChapter] = Field(max_length=3)


class AutobiographyCreate(StorageModel):
    autobiography_id: str
    title: str = Field(min_length=1)
    content: AutobiographyContent
    status: AutobiographyStatus = AutobiographyStatus.DRAFT


class AutobiographyRecord(AutobiographyCreate):
    created_at: AwareDatetime
    updated_at: AwareDatetime
    deleted_at: AwareDatetime | None = None
