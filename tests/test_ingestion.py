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

    def index_memory(self, memory_id: str) -> MemoryIndexResult:
        self.memory_ids.append(memory_id)
        return MemoryIndexResult(
            memory_id=memory_id,
            content_hash="a" * 64,
            indexed=True,
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
    assert result.indexed_memory_count == 1
    assert vector_index.memory_ids == result.memory_ids
    assert repository.get_transcript(result.transcript_id) is not None
    assert len(repository.list_memories(result.transcript_id)) == 1


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
