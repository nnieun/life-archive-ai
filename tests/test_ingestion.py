"""TXT ingestion and memory API tests without real OpenAI calls."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.api.memories import (
    get_ingestion_service,
    get_memory_repository,
)
from backend.app.main import app
from backend.app.models.ingestion import IngestionResult
from backend.app.models.memory import (
    DatePrecision,
    ExtractedMemory,
    MemoryExtractionBatch,
)
from backend.app.models.vector import MemoryIndexResult
from backend.app.services.ingestion import (
    IngestionError,
    TranscriptIngestionService,
    UploadConflictError,
)
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.repository import SQLiteRepository


class ExtractionModel:
    def invoke(self, _input: object) -> MemoryExtractionBatch:
        return MemoryExtractionBatch(
            memories=[
                ExtractedMemory(
                    title="첫 기억",
                    summary="친구와 공원에서 만났다.",
                    people=["친구"],
                    location="공원",
                    event_date=None,
                    date_precision=DatePrecision.UNKNOWN,
                    emotion="반가움",
                    confidence=0.9,
                    evidence_start_offset=0,
                    evidence_end_offset=12,
                    uncertainty_notes=None,
                )
            ]
        )


class VectorIndex:
    def __init__(self) -> None:
        self.memory_ids: list[str] = []
        self.deleted_memory_ids: list[str] = []

    def index_memory(self, memory_id: str) -> MemoryIndexResult:
        self.memory_ids.append(memory_id)
        return MemoryIndexResult(
            memory_id=memory_id,
            content_hash="a" * 64,
            indexed=True,
        )

    def delete_memory(self, memory_id: str) -> bool:
        if memory_id not in self.memory_ids:
            return False
        self.memory_ids.remove(memory_id)
        self.deleted_memory_ids.append(memory_id)
        return True


class TwoMemoryExtractionModel:
    """Return two candidates so a mid-loop index failure leaves one indexed."""

    def invoke(self, _input: object) -> MemoryExtractionBatch:
        return MemoryExtractionBatch(
            memories=[
                ExtractedMemory(
                    title="첫 기억",
                    summary="친구와 공원에서 만났다.",
                    people=["친구"],
                    location="공원",
                    event_date=None,
                    date_precision=DatePrecision.UNKNOWN,
                    emotion="반가움",
                    confidence=0.9,
                    evidence_start_offset=0,
                    evidence_end_offset=12,
                    uncertainty_notes=None,
                ),
                ExtractedMemory(
                    title="두번째 기억",
                    summary="즐거운 하루였다.",
                    people=[],
                    location=None,
                    event_date=None,
                    date_precision=DatePrecision.UNKNOWN,
                    emotion=None,
                    confidence=0.9,
                    evidence_start_offset=13,
                    evidence_end_offset=21,
                    uncertainty_notes=None,
                ),
            ]
        )


class GapExtractionModel:
    def invoke(self, _input: object) -> MemoryExtractionBatch:
        evidence = (
            "2001년 대구 동성로에서 친구들과 영화를 봤지만 "
            "극장 이름은 기억나지 않는다."
        )
        return MemoryExtractionBatch(
            memories=[
                ExtractedMemory(
                    title="동성로 영화관",
                    summary=evidence,
                    people=["친구"],
                    location="대구 동성로",
                    event_date="2001",
                    date_precision=DatePrecision.YEAR,
                    emotion=None,
                    confidence=0.8,
                    evidence_start_offset=0,
                    evidence_end_offset=len(evidence),
                    uncertainty_notes="극장 이름을 기억하지 못한다.",
                )
            ]
        )


@pytest.fixture
def ingestion_storage(tmp_path: Path):
    database = SQLiteDatabase(tmp_path / "ingestion.sqlite3")
    database.initialize()
    repository = SQLiteRepository(database)
    vector_index = VectorIndex()
    service = TranscriptIngestionService(
        tmp_path / "raw" / "transcripts",
        repository,
        ExtractionModel(),
        vector_index,  # type: ignore[arg-type]
    )
    yield service, repository, vector_index, tmp_path / "raw" / "transcripts"
    database.close()


def test_txt_upload_is_immutable_and_indexes_extracted_memory(
    ingestion_storage,
) -> None:
    service, repository, vector_index, raw_root = ingestion_storage
    original = "친구와 공원에서 만났다.\r\n즐거운 하루였다.".encode()

    result = service.ingest(
        filename="memory.txt",
        content=original,
        language="ko",
    )

    assert (raw_root / "memory.txt").read_bytes() == original
    assert result.segment_count == 1
    assert result.memory_count == 1
    assert result.gap_count == 0
    assert result.indexed_memory_count == 1
    assert vector_index.memory_ids == result.memory_ids
    assert repository.get_transcript(result.transcript_id) is not None
    assert len(repository.list_memories(result.transcript_id)) == 1


def test_upload_detects_gap_and_returns_gap_ids(tmp_path: Path) -> None:
    content = (
        "2001년 대구 동성로에서 친구들과 영화를 봤지만 "
        "극장 이름은 기억나지 않는다."
    )
    database = SQLiteDatabase(tmp_path / "gap-ingestion.sqlite3")
    database.initialize()
    repository = SQLiteRepository(database)
    vector_index = VectorIndex()
    service = TranscriptIngestionService(
        tmp_path / "raw" / "transcripts",
        repository,
        GapExtractionModel(),
        vector_index,  # type: ignore[arg-type]
    )

    result = service.ingest(
        filename="unknown-cinema.txt",
        content=content.encode("utf-8"),
        language="ko",
    )

    assert result.memory_count == 1
    assert result.gap_count == 1
    assert len(result.gap_ids) == 1
    assert repository.get_memory_gap(result.gap_ids[0]) is not None
    database.close()


def test_existing_raw_filename_is_not_overwritten(ingestion_storage) -> None:
    service, _repository, _vector_index, raw_root = ingestion_storage
    original = "친구와 공원에서 만났다.".encode()
    (raw_root / "memory.txt").write_bytes(original)

    with pytest.raises(UploadConflictError):
        service.ingest(filename="memory.txt", content="다른 내용입니다.".encode())

    assert (raw_root / "memory.txt").read_bytes() == original


def test_chroma_failure_becomes_safe_ingestion_error(
    ingestion_storage,
) -> None:
    service, _repository, vector_index, _raw_root = ingestion_storage
    vector_index.index_memory = Mock(  # type: ignore[method-assign]
        side_effect=RuntimeError(r"index missing C:\private\chroma")
    )

    with pytest.raises(IngestionError, match="Memory index update failed"):
        service.ingest(
            filename="index-failure.txt",
            content="친구와 공원에서 만나서 즐거운 하루였다.".encode(),
        )


def test_partial_index_failure_purges_already_indexed_vectors(
    tmp_path: Path,
) -> None:
    """A failure on the second of two memories must not strand the first vector.

    delete_transcript hard-deletes the SQLite rows, so once cleanup runs there
    is no way to rediscover which memory_ids had already reached Chroma. The
    service has to remember them as it goes and purge exactly those ids.
    """

    database = SQLiteDatabase(tmp_path / "ingestion.sqlite3")
    database.initialize()
    repository = SQLiteRepository(database)
    vector_index = VectorIndex()
    service = TranscriptIngestionService(
        tmp_path / "raw" / "transcripts",
        repository,
        TwoMemoryExtractionModel(),
        vector_index,  # type: ignore[arg-type]
    )
    real_index_memory = vector_index.index_memory
    call_count = 0

    def flaky_index_memory(memory_id: str) -> MemoryIndexResult:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError(r"index missing C:\private\chroma")
        return real_index_memory(memory_id)

    vector_index.index_memory = flaky_index_memory  # type: ignore[method-assign]

    with pytest.raises(IngestionError, match="Memory index update failed"):
        service.ingest(
            filename="partial-index-failure.txt",
            content="친구와 공원에서 만났다. 즐거운 하루였다.".encode(),
        )

    assert vector_index.memory_ids == []
    assert len(vector_index.deleted_memory_ids) == 1
    assert repository.list_transcripts(include_deleted=False) == []
    database.close()


def test_failed_ingestion_can_be_retried_with_same_content(
    ingestion_storage,
) -> None:
    service, repository, vector_index, raw_root = ingestion_storage
    content = "다시 시도할 내용입니다.".encode()
    vector_index.index_memory = Mock(  # type: ignore[method-assign]
        side_effect=[
            RuntimeError("index missing C:\private\chroma"),
            MemoryIndexResult(
                memory_id="mem_retry",
                content_hash="a" * 64,
                indexed=True,
            ),
        ]
    )

    with pytest.raises(IngestionError, match="Memory index update failed"):
        service.ingest(filename="retry.txt", content=content)

    assert not (raw_root / "retry.txt").exists()
    assert repository.list_transcripts(include_deleted=False) == []

    result = service.ingest(filename="retry.txt", content=content)

    assert result.memory_count == 1
    assert result.transcript_id is not None
    assert (raw_root / "retry.txt").read_bytes() == content
    assert repository.get_transcript(result.transcript_id) is not None


def test_reupload_after_soft_delete_succeeds_with_a_fresh_transcript_id(
    ingestion_storage,
) -> None:
    """Re-uploading content whose transcript was soft-deleted must not 500.

    transcript_id is derived from content_hash, and a soft-deleted row keeps
    that id around for audit purposes, so a second upload of the same bytes
    used to collide on the transcripts primary key with a confusing
    generic IngestionError instead of succeeding.
    """

    service, repository, _vector_index, _raw_root = ingestion_storage
    content = "삭제 후 재업로드되는 내용입니다.".encode()
    first = service.ingest(filename="first-upload.txt", content=content)
    repository.soft_delete_transcript_cascade(first.transcript_id)

    second = service.ingest(filename="second-upload.txt", content=content)

    assert second.transcript_id != first.transcript_id
    assert repository.get_transcript(first.transcript_id) is None
    assert repository.get_transcript(second.transcript_id) is not None
    assert len(repository.list_memories(second.transcript_id)) == 1


def test_same_content_is_blocked_as_duplicate_after_success(
    ingestion_storage,
) -> None:
    service, _repository, _vector_index, _raw_root = ingestion_storage
    content = "중복으로 막혀야 하는 내용입니다.".encode()

    service.ingest(filename="duplicate.txt", content=content)

    with pytest.raises(UploadConflictError, match="already exists"):
        service.ingest(filename="duplicate-again.txt", content=content)


def test_ingest_and_memory_list_api_return_citations(
    ingestion_storage,
) -> None:
    service, repository, _vector_index, _raw_root = ingestion_storage
    app.dependency_overrides[get_ingestion_service] = lambda: service
    app.dependency_overrides[get_memory_repository] = lambda: repository
    client = TestClient(app)
    try:
        from base64 import b64encode

        response = client.post(
            "/api/v1/memories/ingest",
            json={
                "filename": "api-memory.txt",
                "content_base64": b64encode(
                    "친구와 공원에서 만났다. 즐거운 하루였다.".encode()
                ).decode(),
                "language": "ko",
            },
        )
        memories = client.get("/api/v1/memories")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["indexed_memory_count"] == 1
    assert memories.status_code == 200
    assert memories.json()[0]["memory"]["title"] == "첫 기억"
    assert memories.json()[0]["source_filename"] == "api-memory.txt"
    assert memories.json()[0]["citations"][0]["start_offset"] == 0


def test_ingest_api_decodes_original_bytes_before_calling_service() -> None:
    service = Mock()
    service.ingest.return_value = IngestionResult(
        transcript_id="tr_api",
        filename="upload.txt",
        segment_count=1,
        memory_count=0,
        indexed_memory_count=0,
        memory_ids=[],
    )
    app.dependency_overrides[get_ingestion_service] = lambda: service
    client = TestClient(app)
    from base64 import b64encode

    try:
        response = client.post(
            "/api/v1/memories/ingest",
            json={
                "filename": "upload.txt",
                "content_base64": b64encode(b"original\r\nbytes").decode(),
                "recorded_at": datetime(2020, 1, 1, tzinfo=UTC).isoformat(),
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert service.ingest.call_args.kwargs["content"] == b"original\r\nbytes"
