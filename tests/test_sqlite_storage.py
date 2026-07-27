"""Temporary SQLite schema and CRUD integration tests."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.app.models.memory import DatePrecision
from backend.app.models.transcript import LoadedTranscript
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import (
    AutobiographyChapter,
    AutobiographyContent,
    AutobiographyCreate,
    AutobiographyStatus,
    CitationRecord,
    ConversationMessageCreate,
    ConversationSessionCreate,
    MemoryCreate,
    MemorySourceCreate,
    MemoryStatus,
    MemoryUpdate,
    TranscriptMetadataUpdate,
    TranscriptSegmentCreate,
)
from backend.app.storage.repository import (
    SQLiteRepository,
    StorageConflictError,
    StorageIntegrityError,
)

EXPECTED_TABLES = {
    "transcripts",
    "transcript_segments",
    "memories",
    "memory_sources",
    "conversation_sessions",
    "conversation_messages",
    "autobiographies",
}


@pytest.fixture
def storage(tmp_path: Path):
    database_path = tmp_path / "storage-test.sqlite3"
    database = SQLiteDatabase(database_path)
    database.initialize()
    repository = SQLiteRepository(database)
    yield database, repository, database_path
    database.close()
    database_path.unlink(missing_ok=True)
    assert not database_path.exists()


def _loaded_transcript(
    transcript_id: str = "tr_001",
    content_hash: str = "a" * 64,
) -> LoadedTranscript:
    return LoadedTranscript(
        transcript_id=transcript_id,
        filename="private-transcript.txt",
        recording_id="recording_001",
        language="ko",
        source_type="stt_text",
        uploaded_at=datetime(2026, 7, 27, 10, 0, tzinfo=UTC),
        recorded_at=datetime(2025, 1, 2, 9, 30, tzinfo=UTC),
        content_hash=content_hash,
        raw_content="원본 기억",
        normalized_content="원본 기억",
    )


def test_schema_creates_all_tables_and_enables_foreign_keys(storage) -> None:
    database, _repository, _path = storage

    assert database.table_names() == EXPECTED_TABLES
    assert database.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    with pytest.raises(sqlite3.IntegrityError):
        with database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO transcript_segments (
                    segment_id, transcript_id, chunk_index, content,
                    start_offset, end_offset, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "seg_orphan",
                    "tr_missing",
                    0,
                    "orphan",
                    0,
                    6,
                    datetime.now(UTC).isoformat(),
                    datetime.now(UTC).isoformat(),
                ),
            )


def test_schema_migrates_task_006_memory_columns(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy-storage.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE memories (
            memory_id TEXT PRIMARY KEY,
            transcript_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            people_json TEXT NOT NULL,
            location TEXT,
            event_date TEXT,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            supersedes_memory_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            deleted_at TEXT
        );
        INSERT INTO memories (
            memory_id, transcript_id, summary, people_json, event_date,
            confidence, status, created_at, updated_at
        ) VALUES (
            'mem_legacy', 'tr_legacy', 'Legacy memory', '[]',
            '2010-05-01T00:00:00+00:00', 0.8, 'active',
            '2026-07-27T00:00:00+00:00', '2026-07-27T00:00:00+00:00'
        );
        """
    )
    connection.close()

    database = SQLiteDatabase(database_path)
    database.initialize()
    columns = {
        row["name"]
        for row in database.connection.execute(
            "PRAGMA table_info(memories)"
        ).fetchall()
    }
    migrated = database.connection.execute(
        "SELECT title, date_precision, emotion, uncertainty_notes "
        "FROM memories WHERE memory_id = 'mem_legacy'"
    ).fetchone()
    database.close()

    assert {
        "title",
        "date_precision",
        "emotion",
        "uncertainty_notes",
    } <= columns
    assert migrated["title"] == "Legacy memory"
    assert migrated["date_precision"] == "exact"


def test_transcript_crud_and_duplicate_hash_prevention(storage) -> None:
    _database, repository, _path = storage
    loaded = _loaded_transcript()

    created = repository.create_transcript(loaded)
    fetched = repository.get_transcript(loaded.transcript_id)

    assert fetched == created
    assert fetched is not None
    assert fetched.raw_content == "원본 기억"
    assert fetched.recorded_at != fetched.uploaded_at

    updated = repository.update_transcript_metadata(
        loaded.transcript_id,
        TranscriptMetadataUpdate(language="ko-KR"),
    )
    assert updated.language == "ko-KR"
    assert updated.content_hash == loaded.content_hash

    duplicate = _loaded_transcript(
        transcript_id="tr_002",
        content_hash=loaded.content_hash,
    ).model_copy(update={"recording_id": "recording_002"})
    with pytest.raises(StorageConflictError, match="already exists"):
        repository.create_transcript(duplicate)

    assert repository.soft_delete_transcript(loaded.transcript_id) is True
    assert repository.get_transcript(loaded.transcript_id) is None
    assert repository.get_transcript(
        loaded.transcript_id,
        include_deleted=True,
    ).deleted_at is not None


def test_memory_crud_json_validation_and_deleted_filter(storage) -> None:
    database, repository, _path = storage
    repository.create_transcript(_loaded_transcript())
    memory = MemoryCreate(
        memory_id="mem_001",
        transcript_id="tr_001",
        title="가족과의 기억",
        summary="가족과 함께한 기억",
        people=["가족", "친구"],
        location="서울",
        event_date="2010-05-01",
        date_precision=DatePrecision.DAY,
        confidence=0.8,
    )

    created = repository.create_memory(memory)
    assert created.people == ["가족", "친구"]
    assert created.status is MemoryStatus.ACTIVE
    assert created.event_date != repository.get_transcript("tr_001").recorded_at

    updated = repository.update_memory(
        memory.memory_id,
        MemoryUpdate(
            summary="수정된 가족 기억",
            people=["가족"],
            status=MemoryStatus.CORRECTED,
        ),
    )
    assert updated.summary == "수정된 가족 기억"
    assert updated.people == ["가족"]
    assert updated.status is MemoryStatus.CORRECTED
    assert updated.updated_at >= updated.created_at

    deleted = repository.soft_delete_memory(memory.memory_id)
    assert deleted.status is MemoryStatus.DELETED
    assert deleted.deleted_at is not None
    assert repository.get_memory(memory.memory_id) is None
    assert repository.list_memories() == []
    assert repository.get_memory(memory.memory_id, include_deleted=True) == deleted

    with pytest.raises(sqlite3.IntegrityError):
        with database.transaction() as connection:
            connection.execute(
                "UPDATE memories SET people_json = ? WHERE memory_id = ?",
                ("not-json", memory.memory_id),
            )


def test_memory_requires_an_active_transcript(storage) -> None:
    _database, repository, _path = storage

    with pytest.raises(StorageIntegrityError, match="active transcript"):
        repository.create_memory(
            MemoryCreate(
                memory_id="mem_orphan",
                transcript_id="tr_missing",
                title="연결되지 않은 기억",
                summary="연결되지 않은 기억",
                confidence=0.5,
            )
        )


def test_segments_and_memory_sources_round_trip(storage) -> None:
    _database, repository, _path = storage
    repository.create_transcript(_loaded_transcript())
    segment = repository.create_segment(
        TranscriptSegmentCreate(
            segment_id="seg_001",
            transcript_id="tr_001",
            chunk_index=0,
            content="원본 기억",
            start_offset=0,
            end_offset=5,
        )
    )
    repository.create_memory(
        MemoryCreate(
            memory_id="mem_001",
            transcript_id="tr_001",
            title="구조화된 기억",
            summary="구조화된 기억",
            confidence=0.9,
        )
    )

    source = repository.create_memory_source(
        MemorySourceCreate(
            memory_source_id="src_001",
            memory_id="mem_001",
            transcript_id="tr_001",
            segment_id=segment.segment_id,
            start_offset=0,
            end_offset=5,
        )
    )

    assert source.segment_id == "seg_001"
    assert repository.list_memory_sources("mem_001") == [source]
    assert repository.delete_memory_source(source.memory_source_id) is True
    assert repository.list_memory_sources("mem_001") == []


def test_conversation_and_autobiography_json_round_trip(storage) -> None:
    _database, repository, _path = storage
    session = repository.create_conversation_session(
        ConversationSessionCreate(session_id="session_001", title="기억 대화")
    )
    citation = CitationRecord(
        memory_id="mem_001",
        transcript_id="tr_001",
        segment_id="seg_001",
        start_offset=0,
        end_offset=5,
    )
    message = repository.add_conversation_message(
        ConversationMessageCreate(
            message_id="message_001",
            session_id=session.session_id,
            role="assistant",
            content="근거가 있는 답변",
            citations=[citation],
        )
    )
    assert message.citations == [citation]
    assert repository.list_conversation_messages(session.session_id) == [message]
    renamed = repository.update_conversation_session_title(
        session.session_id,
        "수정된 기억 대화",
    )
    assert renamed.title == "수정된 기억 대화"
    assert repository.soft_delete_conversation_message(message.message_id) is True
    assert repository.list_conversation_messages(session.session_id) == []
    assert repository.get_conversation_message(
        message.message_id,
        include_deleted=True,
    ).deleted_at is not None

    autobiography = repository.create_autobiography(
        AutobiographyCreate(
            autobiography_id="book_001",
            title="나의 기억",
            content=AutobiographyContent(
                chapters=[
                    AutobiographyChapter(
                        title="첫 장",
                        content="근거가 있는 내용",
                        citations=[citation],
                    )
                ]
            ),
        )
    )
    assert autobiography.content.chapters[0].citations == [citation]

    completed = repository.update_autobiography(
        autobiography.autobiography_id,
        status=AutobiographyStatus.COMPLETED,
    )
    assert completed.status is AutobiographyStatus.COMPLETED

    deleted = repository.update_autobiography(
        autobiography.autobiography_id,
        status=AutobiographyStatus.DELETED,
    )
    assert deleted.deleted_at is not None
    assert repository.get_autobiography(autobiography.autobiography_id) is None
