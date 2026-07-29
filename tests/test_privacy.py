"""Privacy deletion tests with temporary SQLite and Chroma storage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.api.privacy import get_transcript_deletion_service
from backend.app.main import app
from backend.app.models.memory import DatePrecision
from backend.app.models.privacy import TranscriptDeletionResult
from backend.app.models.transcript import LoadedTranscript
from backend.app.services.privacy import TranscriptDeletionService
from backend.app.services.retrieval import BM25MemoryIndex
from backend.app.services.timeline import TimelineService
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.models import (
    AutobiographyChapter,
    AutobiographyContent,
    AutobiographyCreate,
    CitationRecord,
    ConversationMessageCreate,
    ConversationSessionCreate,
    MemoryCreate,
    MemorySourceCreate,
    MemoryStatus,
    TranscriptSegmentCreate,
)
from backend.app.storage.repository import (
    SQLiteRepository,
    StorageNotFoundError,
)
from tests.conftest import DeterministicEmbeddings


def _create_memory_bundle(
    repository: SQLiteRepository,
    *,
    suffix: str,
    content: str,
    title: str,
) -> CitationRecord:
    transcript_id = f"tr_{suffix}"
    segment_id = f"seg_{suffix}"
    memory_id = f"mem_{suffix}"
    repository.create_transcript(
        LoadedTranscript(
            transcript_id=transcript_id,
            filename=f"synthetic-{suffix}.txt",
            language="ko",
            source_type="stt_text",
            uploaded_at=datetime(2026, 7, 29, tzinfo=UTC),
            content_hash=(suffix[0] * 64),
            raw_content=content,
            normalized_content=content,
        )
    )
    repository.create_segments(
        [
            TranscriptSegmentCreate(
                segment_id=segment_id,
                transcript_id=transcript_id,
                chunk_index=0,
                content=content,
                start_offset=0,
                end_offset=len(content),
            )
        ]
    )
    repository.create_memory(
        MemoryCreate(
            memory_id=memory_id,
            transcript_id=transcript_id,
            title=title,
            summary=content,
            people=[],
            event_date="2020",
            date_precision=DatePrecision.YEAR,
            confidence=0.9,
        )
    )
    repository.create_memory_source(
        MemorySourceCreate(
            memory_source_id=f"src_{suffix}",
            memory_id=memory_id,
            transcript_id=transcript_id,
            segment_id=segment_id,
            start_offset=0,
            end_offset=len(content),
        )
    )
    return CitationRecord(
        memory_id=memory_id,
        transcript_id=transcript_id,
        segment_id=segment_id,
        start_offset=0,
        end_offset=len(content),
    )


@pytest.mark.integration
def test_transcript_deletion_cleans_indexes_and_invalidates_derivatives(
    sqlite_repository: SQLiteRepository,
    tmp_path: Path,
) -> None:
    target_content = "학교 졸업식에서 가족과 사진을 찍었다."
    other_content = "바다 여행에서 가족과 해변을 걸었다."
    target_citation = _create_memory_bundle(
        sqlite_repository,
        suffix="target",
        content=target_content,
        title="학교 졸업식",
    )
    other_citation = _create_memory_bundle(
        sqlite_repository,
        suffix="other",
        content=other_content,
        title="바다 여행",
    )
    sqlite_repository.create_conversation_session(
        ConversationSessionCreate(session_id="privacy_session")
    )
    sqlite_repository.add_conversation_message(
        ConversationMessageCreate(
            message_id="msg_target",
            session_id="privacy_session",
            role="assistant",
            content="졸업식에 관한 답변",
            citations=[target_citation],
        )
    )
    sqlite_repository.add_conversation_message(
        ConversationMessageCreate(
            message_id="msg_other",
            session_id="privacy_session",
            role="assistant",
            content="바다 여행에 관한 답변",
            citations=[other_citation],
        )
    )
    sqlite_repository.create_autobiography(
        AutobiographyCreate(
            autobiography_id="autobio_target",
            title="삭제 대상 자서전",
            content=AutobiographyContent(
                chapters=[
                    AutobiographyChapter(
                        title="졸업",
                        content="졸업식에 관한 내용",
                        citations=[target_citation],
                    )
                ]
            ),
        )
    )
    sqlite_repository.create_autobiography(
        AutobiographyCreate(
            autobiography_id="autobio_other",
            title="보존 대상 자서전",
            content=AutobiographyContent(
                chapters=[
                    AutobiographyChapter(
                        title="여행",
                        content="바다 여행에 관한 내용",
                        citations=[other_citation],
                    )
                ]
            ),
        )
    )

    raw_path = tmp_path / "raw" / "synthetic-target.txt"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(target_content.encode("utf-8"))

    vector_index = MemoryVectorIndex(
        sqlite_repository,
        tmp_path / "chroma",
        embeddings=DeterministicEmbeddings(),
        embedding_version="privacy-test:v1",
    )
    vector_index.rebuild_from_sqlite()
    bm25_index = BM25MemoryIndex(sqlite_repository)
    bm25_index.rebuild_from_sqlite()
    service = TranscriptDeletionService(
        sqlite_repository,
        vector_index,
        bm25_index,
    )

    result = service.delete_transcript("tr_target")

    assert result.deleted_segment_count == 1
    assert result.deleted_memory_count == 1
    assert result.deleted_vector_count == 1
    assert result.bm25_memory_count == 1
    assert result.invalidated_conversation_message_count == 1
    assert result.invalidated_autobiography_count == 1
    assert result.raw_file_deleted is False
    assert raw_path.read_bytes() == target_content.encode("utf-8")

    assert sqlite_repository.get_transcript("tr_target") is None
    assert sqlite_repository.list_segments("tr_target") == []
    deleted_memory = sqlite_repository.get_memory(
        "mem_target",
        include_deleted=True,
    )
    assert deleted_memory is not None
    assert deleted_memory.status is MemoryStatus.DELETED
    assert vector_index.get_metadata("mem_target") is None
    assert bm25_index.search("학교") == []
    assert [hit.memory_id for hit in bm25_index.search("바다")] == [
        "mem_other"
    ]
    assert [event.memory_id for event in TimelineService(
        sqlite_repository
    ).get_timeline().events] == ["mem_other"]
    assert [
        message.message_id
        for message in sqlite_repository.list_conversation_messages(
            "privacy_session"
        )
    ] == ["msg_other"]
    assert sqlite_repository.get_autobiography("autobio_target") is None
    assert sqlite_repository.get_autobiography("autobio_other") is not None

    with pytest.raises(StorageNotFoundError):
        service.delete_transcript("tr_target")


def test_delete_api_returns_safe_result_and_not_found() -> None:
    service = Mock()
    service.delete_transcript.return_value = TranscriptDeletionResult(
        transcript_id="tr_api",
        deleted_segment_count=1,
        deleted_memory_count=1,
        deleted_vector_count=1,
        bm25_memory_count=0,
        invalidated_conversation_message_count=1,
        invalidated_autobiography_count=1,
        raw_file_deleted=False,
    )
    app.dependency_overrides[get_transcript_deletion_service] = lambda: service
    client = TestClient(app)
    try:
        response = client.delete("/api/v1/transcripts/tr_api")
        service.delete_transcript.side_effect = StorageNotFoundError(
            "private detail"
        )
        missing = client.delete("/api/v1/transcripts/tr_missing")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["raw_file_deleted"] is False
    assert missing.status_code == 404
    assert missing.json()["error"]["message"] == "Transcript not found"
    assert "private detail" not in missing.text
