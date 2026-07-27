"""BM25, RRF, and hybrid retrieval tests without real OpenAI calls."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.app.models.memory import DatePrecision
from backend.app.models.transcript import LoadedTranscript
from backend.app.services.retrieval import (
    BM25MemoryIndex,
    HybridMemoryRetriever,
    reciprocal_rank_fusion,
    tokenize_for_bm25,
)
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import MemoryCreate, MemoryUpdate
from backend.app.storage.repository import SQLiteRepository


class FakeEmbeddings:
    """Deterministic semantic groups for persistent Chroma tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("학교")) + 0.01,
            float(lowered.count("바다")) + 0.01,
            float(lowered.count("요리")) + 0.01,
        ]


@pytest.fixture
def retrieval_storage(tmp_path: Path):
    database = SQLiteDatabase(tmp_path / "retrieval.sqlite3")
    database.initialize()
    repository = SQLiteRepository(database)
    repository.create_transcript(
        LoadedTranscript(
            transcript_id="tr_001",
            filename="private.txt",
            language="ko",
            source_type="stt_text",
            uploaded_at=datetime(2026, 7, 27, tzinfo=UTC),
            content_hash="b" * 64,
            raw_content="원본",
            normalized_content="원본",
        )
    )
    yield repository, tmp_path
    database.close()


def _memory(memory_id: str, title: str, summary: str) -> MemoryCreate:
    return MemoryCreate(
        memory_id=memory_id,
        transcript_id="tr_001",
        title=title,
        summary=summary,
        people=[],
        location=None,
        event_date=None,
        date_precision=DatePrecision.UNKNOWN,
        emotion=None,
        confidence=0.9,
    )


def _create_memories(repository: SQLiteRepository) -> None:
    repository.create_memory(
        _memory(
            "mem_school",
            "학교 졸업식",
            "친구들과 학교 운동장에서 졸업 사진을 찍었다.",
        )
    )
    repository.create_memory(
        _memory(
            "mem_beach",
            "여름 바다",
            "가족과 해변에서 파도를 보며 쉬었다.",
        )
    )
    repository.create_memory(
        _memory(
            "mem_cooking",
            "첫 요리 수업",
            "주말에 가족과 김치찌개 요리를 배웠다.",
        )
    )


def test_korean_tokenizer_normalizes_and_adds_character_bigrams() -> None:
    tokens = tokenize_for_bm25("ＡＢＣ 졸업식")

    assert "abc" in tokens
    assert "졸업식" in tokens
    assert "2:졸업" in tokens
    assert "2:업식" in tokens


def test_bm25_search_supports_korean_partial_terms_and_top_k(
    retrieval_storage,
) -> None:
    repository, _tmp_path = retrieval_storage
    _create_memories(repository)
    index = BM25MemoryIndex(repository)
    results = index.rebuild_from_sqlite()

    hits = index.search("졸업", top_k=1)

    assert index.count == 3
    assert all(result.indexed for result in results)
    assert [hit.memory_id for hit in hits] == ["mem_school"]
    assert hits[0].memory == repository.get_memory("mem_school")
    assert index.search("우주선") == []


def test_bm25_skips_stale_content_until_synchronized(retrieval_storage) -> None:
    repository, _tmp_path = retrieval_storage
    repository.create_memory(
        _memory("mem_001", "학교 입학", "학교에서 새 친구를 만났다.")
    )
    index = BM25MemoryIndex(repository)
    index.rebuild_from_sqlite()
    repository.update_memory(
        "mem_001",
        MemoryUpdate(title="바다 여행", summary="해변에서 파도를 봤다."),
    )

    assert index.search("학교") == []

    results = index.sync_from_sqlite()

    assert results[0].indexed is True
    assert [hit.memory_id for hit in index.search("바다")] == ["mem_001"]


def test_bm25_excludes_deleted_memories_and_removes_stale_entries(
    retrieval_storage,
) -> None:
    repository, _tmp_path = retrieval_storage
    repository.create_memory(
        _memory("mem_001", "학교 입학", "학교에서 새 친구를 만났다.")
    )
    index = BM25MemoryIndex(repository)
    index.rebuild_from_sqlite()
    repository.soft_delete_memory("mem_001")

    assert index.search("학교") == []

    assert index.sync_from_sqlite() == []
    assert index.count == 0
    assert index.search("학교") == []


def test_rrf_combines_ranks_and_deduplicates_within_each_list() -> None:
    scores = reciprocal_rank_fusion(
        [
            ["mem_a", "mem_b", "mem_b"],
            ["mem_b", "mem_c"],
        ],
        rank_constant=10,
    )

    assert scores["mem_b"] == pytest.approx((1 / 12) + (1 / 11))
    assert scores["mem_a"] == pytest.approx(1 / 11)
    assert scores["mem_c"] == pytest.approx(1 / 12)
    assert scores["mem_b"] > scores["mem_a"] > scores["mem_c"]


def test_hybrid_retrieval_fuses_dense_and_bm25_without_duplicate_ids(
    retrieval_storage,
) -> None:
    repository, tmp_path = retrieval_storage
    _create_memories(repository)
    vector_index = MemoryVectorIndex(
        repository,
        tmp_path / "chroma",
        embeddings=FakeEmbeddings(),
        embedding_version="fake-ko:v1",
    )
    vector_index.rebuild_from_sqlite()
    bm25_index = BM25MemoryIndex(repository)
    bm25_index.rebuild_from_sqlite()
    retriever = HybridMemoryRetriever(repository, vector_index, bm25_index)

    hits = retriever.search("학교 졸업", top_k=2)

    assert len(hits) == 2
    assert hits[0].memory_id == "mem_school"
    assert hits[0].dense_rank is not None
    assert hits[0].bm25_rank == 1
    assert len({hit.memory_id for hit in hits}) == len(hits)
    assert all(hit.memory == repository.get_memory(hit.memory_id) for hit in hits)


def test_hybrid_retrieval_excludes_memory_deleted_after_indexing(
    retrieval_storage,
) -> None:
    repository, tmp_path = retrieval_storage
    repository.create_memory(
        _memory("mem_001", "학교 입학", "학교에서 새 친구를 만났다.")
    )
    vector_index = MemoryVectorIndex(
        repository,
        tmp_path / "chroma",
        embeddings=FakeEmbeddings(),
        embedding_version="fake-ko:v1",
    )
    vector_index.rebuild_from_sqlite()
    bm25_index = BM25MemoryIndex(repository)
    bm25_index.rebuild_from_sqlite()
    retriever = HybridMemoryRetriever(repository, vector_index, bm25_index)
    repository.soft_delete_memory("mem_001")

    assert retriever.search("학교") == []


def test_retrieval_validates_blank_query_and_top_k(retrieval_storage) -> None:
    repository, _tmp_path = retrieval_storage
    index = BM25MemoryIndex(repository)

    with pytest.raises(ValueError, match="blank"):
        index.search(" ")
    with pytest.raises(ValueError, match="top_k"):
        index.search("학교", top_k=0)
