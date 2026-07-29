"""Validated results for privacy-safe transcript deletion."""

from pydantic import BaseModel, ConfigDict, Field


class SQLiteTranscriptDeletion(BaseModel):
    """SQLite rows affected by one atomic logical deletion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_id: str
    memory_ids: list[str]
    deleted_segment_count: int = Field(ge=0)
    deleted_memory_count: int = Field(ge=0)
    invalidated_conversation_message_count: int = Field(ge=0)
    invalidated_autobiography_count: int = Field(ge=0)


class TranscriptDeletionResult(BaseModel):
    """Public result including disposable-index cleanup."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_id: str
    deleted_segment_count: int = Field(ge=0)
    deleted_memory_count: int = Field(ge=0)
    deleted_vector_count: int = Field(ge=0)
    bm25_memory_count: int = Field(ge=0)
    invalidated_conversation_message_count: int = Field(ge=0)
    invalidated_autobiography_count: int = Field(ge=0)
    raw_file_deleted: bool = False
