"""Validated vector-index operation and search results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from backend.app.storage.models import MemoryRecord


class VectorModel(BaseModel):
    """Strict immutable base for vector-index result models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryIndexResult(VectorModel):
    """Outcome of synchronizing one SQLite memory with Chroma."""

    memory_id: str
    content_hash: str | None = None
    indexed: bool = False
    deleted: bool = False


class MemoryIndexMetadata(VectorModel):
    """Minimal rebuildable metadata stored alongside a vector."""

    memory_id: str
    embedding_version: str
    content_hash: str


class MemoryVectorSearchHit(VectorModel):
    """A vector score paired with the current SQLite source-of-truth record."""

    memory_id: str
    distance: float = Field(ge=0.0)
    memory: MemoryRecord
