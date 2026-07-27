"""Traceable transcript chunking unit and SQLite integration tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.app.models.chunk import ChunkingConfig, ChunkUnit, TranscriptChunk
from backend.app.models.transcript import LoadedTranscript
from backend.app.services.chunking import (
    CHUNK_SIZE_CANDIDATES,
    EVENT_AWARE_CANDIDATE,
    ChunkingTranscriptNotFoundError,
    chunk_and_store_transcript,
    chunk_transcript,
)
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import TranscriptSegmentCreate
from backend.app.storage.repository import SQLiteRepository, StorageIntegrityError


def _transcript(text: str) -> LoadedTranscript:
    return LoadedTranscript(
        transcript_id="tr_chunking",
        filename="private-transcript.txt",
        language="ko",
        source_type="stt_text",
        uploaded_at=datetime(2026, 7, 27, tzinfo=UTC),
        content_hash="c" * 64,
        raw_content=text,
        normalized_content=text,
    )


def test_character_chunking_preserves_offsets_and_overlap() -> None:
    text = "가나다라마바사아자차카타파하"
    chunks = chunk_transcript(
        "tr_001",
        text,
        ChunkingConfig(chunk_size=6, chunk_overlap=2),
    )

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 6),
        (4, 10),
        (8, 14),
    ]
    assert all(
        chunk.content == text[chunk.start_offset : chunk.end_offset]
        for chunk in chunks
    )
    assert chunks[0].content[-2:] == chunks[1].content[:2]
    assert chunks[1].content[-2:] == chunks[2].content[:2]


def test_short_and_empty_transcripts() -> None:
    short_text = "짧은 기록"
    chunks = chunk_transcript(
        "tr_short",
        short_text,
        ChunkingConfig(chunk_size=256, chunk_overlap=32),
    )

    assert len(chunks) == 1
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(short_text)
    assert chunks[0].content == short_text
    assert chunk_transcript("tr_empty", "") == []
    assert chunk_transcript("tr_blank", " \n\t") == []


def test_very_long_transcript_has_contiguous_indexes_and_full_tail() -> None:
    text = "".join(str(index % 10) for index in range(10_000))
    chunks = chunk_transcript(
        "tr_long",
        text,
        ChunkingConfig(chunk_size=1024, chunk_overlap=128),
    )

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[-1].end_offset == len(text)
    assert all(
        chunk.content == text[chunk.start_offset : chunk.end_offset]
        for chunk in chunks
    )
    assert all(
        previous.end_offset - current.start_offset == 128
        for previous, current in zip(chunks, chunks[1:], strict=False)
    )


def test_token_chunking_uses_whitespace_delimited_source_spans() -> None:
    text = "하나 둘 셋\n넷 다섯 여섯"
    chunks = chunk_transcript(
        "tr_tokens",
        text,
        ChunkingConfig(
            chunk_size=3,
            chunk_overlap=1,
            unit=ChunkUnit.TOKEN,
        ),
    )

    assert len(chunks) == 3
    assert chunks[0].content == "하나 둘 셋\n"
    assert chunks[1].content == "셋\n넷 다섯 "
    assert chunks[2].content == "다섯 여섯"
    assert all(
        chunk.content == text[chunk.start_offset : chunk.end_offset]
        for chunk in chunks
    )


def test_chunk_configuration_and_model_reject_invalid_ranges() -> None:
    assert CHUNK_SIZE_CANDIDATES == (256, 512, 1024)
    assert EVENT_AWARE_CANDIDATE == "event_aware"

    with pytest.raises(ValidationError, match="smaller than chunk_size"):
        ChunkingConfig(chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValidationError, match="content length"):
        TranscriptChunk(
            segment_id="seg_invalid",
            transcript_id="tr_001",
            chunk_index=0,
            content="abc",
            start_offset=0,
            end_offset=2,
        )


def test_chunking_persists_segments_in_sqlite_atomically() -> None:
    text = "0123456789abcdef"
    with SQLiteDatabase(":memory:") as database:
        repository = SQLiteRepository(database)
        repository.create_transcript(_transcript(text))

        with pytest.raises(StorageIntegrityError, match="offset range"):
            repository.create_segments(
                [
                    TranscriptSegmentCreate(
                        segment_id="seg_invalid",
                        transcript_id="tr_chunking",
                        chunk_index=0,
                        content="wrong",
                        start_offset=0,
                        end_offset=5,
                    )
                ]
            )
        assert repository.list_segments("tr_chunking") == []

        chunks = chunk_and_store_transcript(
            repository,
            "tr_chunking",
            ChunkingConfig(chunk_size=6, chunk_overlap=2),
        )
        stored = repository.list_segments("tr_chunking")

        assert len(stored) == len(chunks)
        assert [
            (
                segment.segment_id,
                segment.chunk_index,
                segment.content,
                segment.start_offset,
                segment.end_offset,
            )
            for segment in stored
        ] == [
            (
                chunk.segment_id,
                chunk.chunk_index,
                chunk.content,
                chunk.start_offset,
                chunk.end_offset,
            )
            for chunk in chunks
        ]

        with pytest.raises(StorageIntegrityError):
            chunk_and_store_transcript(
                repository,
                "tr_chunking",
                ChunkingConfig(chunk_size=6, chunk_overlap=2),
            )
        assert repository.list_segments("tr_chunking") == stored


def test_chunking_requires_an_active_stored_transcript() -> None:
    with SQLiteDatabase(":memory:") as database:
        repository = SQLiteRepository(database)

        with pytest.raises(ChunkingTranscriptNotFoundError):
            chunk_and_store_transcript(repository, "tr_missing")
