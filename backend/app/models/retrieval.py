"""Validated sparse and hybrid memory retrieval results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.storage.models import MemoryRecord


class RetrievalModel(BaseModel):
    """Strict immutable base for retrieval models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BM25IndexResult(RetrievalModel):
    """Outcome of synchronizing one memory with the in-memory BM25 index."""

    memory_id: str
    content_hash: str | None = None
    indexed: bool = False
    deleted: bool = False


class BM25SearchHit(RetrievalModel):
    """One keyword-search result reloaded from SQLite."""

    memory_id: str
    score: float = Field(gt=0.0)
    memory: MemoryRecord

    @model_validator(mode="after")
    def validate_memory_id(self) -> BM25SearchHit:
        if self.memory_id != self.memory.memory_id:
            raise ValueError("memory_id must match the SQLite memory")
        return self


class RetrievalHit(RetrievalModel):
    """One deduplicated RRF result with source-specific rank details."""

    memory_id: str
    score: float = Field(gt=0.0)
    memory: MemoryRecord
    dense_rank: int | None = Field(default=None, ge=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    dense_distance: float | None = Field(default=None, ge=0.0)
    bm25_score: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_sources(self) -> RetrievalHit:
        if self.memory_id != self.memory.memory_id:
            raise ValueError("memory_id must match the SQLite memory")
        if self.dense_rank is None and self.bm25_rank is None:
            raise ValueError("at least one retrieval rank is required")
        if (self.dense_rank is None) != (self.dense_distance is None):
            raise ValueError("dense rank and distance must appear together")
        if (self.bm25_rank is None) != (self.bm25_score is None):
            raise ValueError("BM25 rank and score must appear together")
        return self
