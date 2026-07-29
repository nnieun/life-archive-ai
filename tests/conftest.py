"""Shared privacy-safe test doubles and SQLite fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.repository import SQLiteRepository


class QueueStructuredModel:
    """Return predefined structured outputs and fail on unexpected calls."""

    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.inputs: list[object] = []

    def invoke(self, input: object) -> object:
        self.inputs.append(input)
        if not self.outputs:
            raise AssertionError("Unexpected model call")
        return self.outputs.pop(0)


class DeterministicEmbeddings:
    """Small semantic groups for Chroma tests without network access."""

    _terms = ("학교", "졸업", "직장", "바다", "요리", "도서관")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        lowered = text.casefold()
        return [
            float(lowered.count(term)) + 0.01
            for term in cls._terms
        ]


@pytest.fixture
def sqlite_repository(tmp_path: Path) -> Iterator[SQLiteRepository]:
    """Provide a fresh initialized SQLite source of truth per test."""

    database = SQLiteDatabase(tmp_path / "test.sqlite3")
    database.initialize()
    try:
        yield SQLiteRepository(database)
    finally:
        database.close()
