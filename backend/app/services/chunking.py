"""Deterministic fixed-size transcript chunking with source offsets."""

from __future__ import annotations

import hashlib
import re

from backend.app.models.chunk import (
    ChunkingConfig,
    ChunkUnit,
    TranscriptChunk,
)
from backend.app.storage.models import TranscriptSegmentCreate
from backend.app.storage.repository import SQLiteRepository

CHUNK_SIZE_CANDIDATES = (256, 512, 1024)
EVENT_AWARE_CANDIDATE = "event_aware"

_TOKEN_WITH_TRAILING_WHITESPACE = re.compile(r"\S+\s*")


class ChunkingError(ValueError):
    """Base class for deterministic chunking failures."""


class ChunkingTranscriptNotFoundError(ChunkingError):
    """The requested active transcript is not stored in SQLite."""


def _unit_boundaries(text: str, unit: ChunkUnit) -> list[int]:
    if unit is ChunkUnit.CHARACTER:
        return list(range(len(text) + 1))

    boundaries = [0]
    boundaries.extend(
        match.end() for match in _TOKEN_WITH_TRAILING_WHITESPACE.finditer(text)
    )
    return boundaries


def _segment_id(
    transcript_id: str,
    chunk_index: int,
    start_offset: int,
    end_offset: int,
) -> str:
    identity = (
        f"{transcript_id}:{chunk_index}:{start_offset}:{end_offset}".encode("utf-8")
    )
    return f"seg_{hashlib.sha256(identity).hexdigest()[:24]}"


def chunk_transcript(
    transcript_id: str,
    text: str,
    config: ChunkingConfig | None = None,
) -> list[TranscriptChunk]:
    """Split normalized text while keeping exact Python string offsets."""
    settings = config or ChunkingConfig()
    if not transcript_id.strip():
        raise ChunkingError("transcript_id must not be blank")
    if not text or not text.strip():
        return []

    boundaries = _unit_boundaries(text, settings.unit)
    unit_count = len(boundaries) - 1
    chunks: list[TranscriptChunk] = []
    unit_start = 0

    while unit_start < unit_count:
        unit_end = min(unit_start + settings.chunk_size, unit_count)
        start_offset = boundaries[unit_start]
        end_offset = boundaries[unit_end]
        content = text[start_offset:end_offset]
        chunk_index = len(chunks)
        chunks.append(
            TranscriptChunk(
                segment_id=_segment_id(
                    transcript_id,
                    chunk_index,
                    start_offset,
                    end_offset,
                ),
                transcript_id=transcript_id,
                chunk_index=chunk_index,
                content=content,
                start_offset=start_offset,
                end_offset=end_offset,
            )
        )
        if unit_end == unit_count:
            break
        unit_start = unit_end - settings.chunk_overlap

    return chunks


def chunk_and_store_transcript(
    repository: SQLiteRepository,
    transcript_id: str,
    config: ChunkingConfig | None = None,
) -> list[TranscriptChunk]:
    """Chunk one active SQLite transcript and atomically persist its segments."""
    transcript = repository.get_transcript(transcript_id)
    if transcript is None:
        raise ChunkingTranscriptNotFoundError("Active transcript was not found")

    chunks = chunk_transcript(
        transcript_id,
        transcript.normalized_content,
        config,
    )
    repository.create_segments(
        [
            TranscriptSegmentCreate(
                segment_id=chunk.segment_id,
                transcript_id=chunk.transcript_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
            )
            for chunk in chunks
        ]
    )
    return chunks
