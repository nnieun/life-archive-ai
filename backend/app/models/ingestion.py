"""Validated transcript ingestion results."""

from pydantic import BaseModel, ConfigDict, Field


class IngestionResult(BaseModel):
    """Public processing and indexing summary for one immutable upload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_id: str
    filename: str
    segment_count: int = Field(ge=0)
    memory_count: int = Field(ge=0)
    gap_count: int = Field(default=0, ge=0)
    indexed_memory_count: int = Field(ge=0)
    memory_ids: list[str]
    gap_ids: list[str] = Field(default_factory=list)
