"""Traceable transcript chunk models and configuration."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkUnit(StrEnum):
    """Units supported by the deterministic fixed-size chunker."""

    CHARACTER = "character"
    TOKEN = "token"


class ChunkingConfig(BaseModel):
    """Validated fixed-size chunking settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    unit: ChunkUnit = ChunkUnit.CHARACTER

    @model_validator(mode="after")
    def validate_overlap(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class TranscriptChunk(BaseModel):
    """One chunk whose offsets refer to normalized transcript content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_id: str
    transcript_id: str
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1, repr=False)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> TranscriptChunk:
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must follow start_offset")
        if len(self.content) != self.end_offset - self.start_offset:
            raise ValueError("content length must match its offset range")
        return self
