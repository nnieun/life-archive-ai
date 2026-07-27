"""Korean-friendly BM25 and reciprocal-rank-fusion retrieval."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from rank_bm25 import BM25Plus

from backend.app.models.retrieval import (
    BM25IndexResult,
    BM25SearchHit,
    RetrievalHit,
)
from backend.app.models.vector import MemoryVectorSearchHit
from backend.app.storage.models import MemoryRecord
from backend.app.storage.repository import SQLiteRepository

DEFAULT_RRF_CONSTANT = 60
_TERM_PATTERN = re.compile(r"[0-9a-z가-힣]+")


class DenseRetriever(Protocol):
    """Dense-search interface implemented by MemoryVectorIndex."""

    def similarity_search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[MemoryVectorSearchHit]:
        """Return Chroma-ranked memories reloaded from SQLite."""


@dataclass(frozen=True)
class _BM25Document:
    memory_id: str
    content_hash: str
    tokens: tuple[str, ...]


class BM25MemoryIndex:
    """Maintain a rebuildable in-memory keyword index from active SQLite rows."""

    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
    ) -> None:
        self._repository = repository
        self._k1 = k1
        self._b = b
        self._delta = delta
        self._documents: dict[str, _BM25Document] = {}
        self._ordered_ids: list[str] = []
        self._engine: BM25Plus | None = None

    @property
    def count(self) -> int:
        return len(self._documents)

    def index_memory(self, memory_id: str) -> BM25IndexResult:
        """Insert, refresh, or remove one record using current SQLite state."""

        memory = self._repository.get_memory(memory_id)
        if memory is None:
            deleted = self._documents.pop(memory_id, None) is not None
            if deleted:
                self._rebuild_engine()
            return BM25IndexResult(memory_id=memory_id, deleted=deleted)

        content = _memory_content(memory)
        content_hash = _content_hash(content)
        existing = self._documents.get(memory_id)
        if existing is not None and existing.content_hash == content_hash:
            return BM25IndexResult(
                memory_id=memory_id,
                content_hash=content_hash,
            )

        self._documents[memory_id] = _BM25Document(
            memory_id=memory_id,
            content_hash=content_hash,
            tokens=tuple(tokenize_for_bm25(content)),
        )
        self._rebuild_engine()
        return BM25IndexResult(
            memory_id=memory_id,
            content_hash=content_hash,
            indexed=True,
        )

    def sync_from_sqlite(self) -> list[BM25IndexResult]:
        """Synchronize active memories and remove deleted or orphaned entries."""

        memories = self._repository.list_memories()
        active_ids = {memory.memory_id for memory in memories}
        stale_ids = set(self._documents) - active_ids
        for stale_id in stale_ids:
            del self._documents[stale_id]

        results: list[BM25IndexResult] = []
        changed = bool(stale_ids) or set(self._documents) != active_ids
        for memory in memories:
            content = _memory_content(memory)
            content_hash = _content_hash(content)
            existing = self._documents.get(memory.memory_id)
            if existing is not None and existing.content_hash == content_hash:
                results.append(
                    BM25IndexResult(
                        memory_id=memory.memory_id,
                        content_hash=content_hash,
                    )
                )
                continue
            self._documents[memory.memory_id] = _BM25Document(
                memory_id=memory.memory_id,
                content_hash=content_hash,
                tokens=tuple(tokenize_for_bm25(content)),
            )
            changed = True
            results.append(
                BM25IndexResult(
                    memory_id=memory.memory_id,
                    content_hash=content_hash,
                    indexed=True,
                )
            )
        if changed or self._engine is None:
            self._rebuild_engine()
        return results

    def rebuild_from_sqlite(self) -> list[BM25IndexResult]:
        """Discard the keyword index and recreate it from SQLite."""

        self._documents.clear()
        self._ordered_ids.clear()
        self._engine = None
        return self.sync_from_sqlite()

    def search(self, query: str, *, top_k: int = 5) -> list[BM25SearchHit]:
        """Rank keyword matches and reload accepted memories from SQLite."""

        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_tokens = tokenize_for_bm25(query)
        if self._engine is None or not query_tokens:
            return []

        query_token_set = set(query_tokens)
        scores = self._engine.get_scores(query_tokens)
        candidates: list[tuple[str, float]] = []
        for memory_id, raw_score in zip(self._ordered_ids, scores, strict=True):
            document = self._documents[memory_id]
            if query_token_set.isdisjoint(document.tokens):
                continue
            score = float(raw_score)
            if score > 0.0:
                candidates.append((memory_id, score))
        candidates.sort(key=lambda item: (-item[1], item[0]))

        hits: list[BM25SearchHit] = []
        for memory_id, score in candidates:
            memory = self._repository.get_memory(memory_id)
            document = self._documents[memory_id]
            if (
                memory is None
                or document.content_hash
                != _content_hash(_memory_content(memory))
            ):
                continue
            hits.append(
                BM25SearchHit(
                    memory_id=memory_id,
                    score=score,
                    memory=memory,
                )
            )
            if len(hits) == top_k:
                break
        return hits

    def _rebuild_engine(self) -> None:
        self._ordered_ids = sorted(self._documents)
        corpus = [
            list(self._documents[memory_id].tokens)
            for memory_id in self._ordered_ids
        ]
        self._engine = (
            BM25Plus(
                corpus,
                k1=self._k1,
                b=self._b,
                delta=self._delta,
            )
            if corpus
            else None
        )


class HybridMemoryRetriever:
    """Fuse Chroma dense ranks and BM25 keyword ranks with RRF."""

    def __init__(
        self,
        repository: SQLiteRepository,
        dense_retriever: DenseRetriever,
        bm25_index: BM25MemoryIndex,
        *,
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
        candidate_multiplier: int = 2,
    ) -> None:
        if rrf_constant < 1:
            raise ValueError("rrf_constant must be at least 1")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        self._repository = repository
        self._dense_retriever = dense_retriever
        self._bm25_index = bm25_index
        self._rrf_constant = rrf_constant
        self._candidate_multiplier = candidate_multiplier

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        """Return unique active memories ordered by combined dense/sparse rank."""

        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        candidate_k = top_k * self._candidate_multiplier
        dense_hits = self._dense_retriever.similarity_search(
            query,
            top_k=candidate_k,
        )
        bm25_hits = self._bm25_index.search(query, top_k=candidate_k)
        scores = reciprocal_rank_fusion(
            [
                [hit.memory_id for hit in dense_hits],
                [hit.memory_id for hit in bm25_hits],
            ],
            rank_constant=self._rrf_constant,
        )
        dense_by_id = {
            hit.memory_id: (rank, hit)
            for rank, hit in enumerate(dense_hits, start=1)
        }
        bm25_by_id = {
            hit.memory_id: (rank, hit)
            for rank, hit in enumerate(bm25_hits, start=1)
        }

        ordered_ids = sorted(
            scores,
            key=lambda memory_id: (
                -scores[memory_id],
                _best_rank(memory_id, dense_by_id, bm25_by_id),
                memory_id,
            ),
        )
        results: list[RetrievalHit] = []
        for memory_id in ordered_ids:
            memory = self._repository.get_memory(memory_id)
            if memory is None:
                continue
            dense = dense_by_id.get(memory_id)
            sparse = bm25_by_id.get(memory_id)
            results.append(
                RetrievalHit(
                    memory_id=memory_id,
                    score=scores[memory_id],
                    memory=memory,
                    dense_rank=dense[0] if dense else None,
                    dense_distance=dense[1].distance if dense else None,
                    bm25_rank=sparse[0] if sparse else None,
                    bm25_score=sparse[1].score if sparse else None,
                )
            )
            if len(results) == top_k:
                break
        return results


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    rank_constant: int = DEFAULT_RRF_CONSTANT,
) -> dict[str, float]:
    """Combine ranked ID lists while counting each ID once per ranking."""

    if rank_constant < 1:
        raise ValueError("rank_constant must be at least 1")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, memory_id in enumerate(ranking, start=1):
            if memory_id in seen:
                continue
            seen.add(memory_id)
            scores[memory_id] = scores.get(memory_id, 0.0) + (
                1.0 / (rank_constant + rank)
            )
    return scores


def tokenize_for_bm25(text: str) -> list[str]:
    """Normalize text and add character bigrams for lightweight Korean search."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    terms = _TERM_PATTERN.findall(normalized)
    if not terms and normalized.strip():
        return [normalized.strip()]
    tokens = list(terms)
    for term in terms:
        tokens.extend(f"2:{term[index:index + 2]}" for index in range(len(term) - 1))
    return tokens


def _memory_content(memory: MemoryRecord) -> str:
    return f"{memory.title.strip()}\n{memory.summary.strip()}"


def _content_hash(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _best_rank(
    memory_id: str,
    dense_by_id: dict[str, tuple[int, MemoryVectorSearchHit]],
    bm25_by_id: dict[str, tuple[int, BM25SearchHit]],
) -> int:
    ranks = [
        source[memory_id][0]
        for source in (dense_by_id, bm25_by_id)
        if memory_id in source
    ]
    return min(ranks)
