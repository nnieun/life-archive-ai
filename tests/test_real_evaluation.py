"""Tests for the opt-in real-embedding evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

from evaluation.real_runner import REAL_SEARCH_METHODS, evaluate_real
from evaluation.runner import TOP_K_VALUES, load_dataset


class TinyEmbeddings:
    """A local test double; this is not used by the real evaluation script."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        value = sum(ord(character) for character in text)
        return [float(value % 101), float((value // 101) % 101), 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def test_real_evaluation_is_separate_and_complete() -> None:
    dataset = load_dataset(Path("evaluation/dataset.json"))

    rows = evaluate_real(dataset, TinyEmbeddings())

    assert len(rows) == (
        len(dataset.queries)
        * 4
        * len(REAL_SEARCH_METHODS)
        * len(TOP_K_VALUES)
    )
    assert {row["search_method"] for row in rows} == set(REAL_SEARCH_METHODS)
    assert {int(row["top_k"]) for row in rows} == set(TOP_K_VALUES)
    assert all(0.0 <= float(row["recall_at_k"]) <= 1.0 for row in rows)
