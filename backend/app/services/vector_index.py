"""Persistent Chroma index rebuilt from SQLite memory records."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings

from backend.app.models.vector import (
    MemoryIndexMetadata,
    MemoryIndexResult,
    MemoryVectorSearchHit,
)
from backend.app.storage.models import MemoryRecord
from backend.app.storage.repository import SQLiteRepository

DEFAULT_COLLECTION_NAME = "life_archive_memories"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_VERSION = f"{DEFAULT_EMBEDDING_MODEL}:v1"


class EmbeddingProvider(Protocol):
    """Small interface shared by OpenAI embeddings and test doubles."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for indexing."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one similarity-search query."""


def create_openai_embeddings(
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> OpenAIEmbeddings:
    """Create the production OpenAI embedding provider."""

    return OpenAIEmbeddings(model=model)


class MemoryVectorIndex:
    """Maintain a disposable Chroma index whose source of truth is SQLite."""

    def __init__(
        self,
        repository: SQLiteRepository,
        persist_directory: Path | str,
        *,
        embeddings: EmbeddingProvider | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_version: str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self._repository = repository
        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._embeddings = embeddings or create_openai_embeddings(embedding_model)
        self._embedding_version = (
            embedding_version or f"{embedding_model}:v1"
        )
        self._collection_name = collection_name
        self._client: ClientAPI = chromadb.PersistentClient(
            path=self._persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._get_or_create_collection()

    @property
    def count(self) -> int:
        """Return the number of vectors, not the number of SQLite memories."""

        return self._collection.count()

    def index_memory(
        self,
        memory_id: str,
        *,
        force: bool = False,
    ) -> MemoryIndexResult:
        """Insert, refresh, or remove one vector based on current SQLite state."""

        memory = self._repository.get_memory(memory_id)
        if memory is None:
            deleted = self.delete_memory(memory_id)
            return MemoryIndexResult(memory_id=memory_id, deleted=deleted)

        content = _index_content(memory)
        content_hash = _content_hash(content)
        existing = self.get_metadata(memory_id)
        if (
            not force
            and existing is not None
            and existing.content_hash == content_hash
            and existing.embedding_version == self._embedding_version
        ):
            return MemoryIndexResult(
                memory_id=memory_id,
                content_hash=content_hash,
            )

        vectors = self._embeddings.embed_documents([content])
        if len(vectors) != 1 or not vectors[0]:
            raise ValueError("Embedding provider must return one non-empty vector")

        metadata = MemoryIndexMetadata(
            memory_id=memory_id,
            embedding_version=self._embedding_version,
            content_hash=content_hash,
        )
        self._collection.upsert(
            ids=[memory_id],
            embeddings=vectors,
            documents=[content],
            metadatas=[metadata.model_dump()],
        )
        return MemoryIndexResult(
            memory_id=memory_id,
            content_hash=content_hash,
            indexed=True,
        )

    def reindex_memory(self, memory_id: str) -> MemoryIndexResult:
        """Force regeneration of one memory vector."""

        return self.index_memory(memory_id, force=True)

    def sync_from_sqlite(self) -> list[MemoryIndexResult]:
        """Index active records and delete vectors absent from active SQLite data."""

        memories = self._repository.list_memories()
        active_ids = {memory.memory_id for memory in memories}
        indexed_ids = set(self._collection.get(include=[]).get("ids", []))
        stale_ids = sorted(indexed_ids - active_ids)
        if stale_ids:
            self._collection.delete(ids=stale_ids)
        return [self.index_memory(memory.memory_id) for memory in memories]

    def rebuild_from_sqlite(self) -> list[MemoryIndexResult]:
        """Drop the disposable collection and rebuild it from active SQLite rows."""

        try:
            self._client.delete_collection(self._collection_name)
        except Exception as exception:
            if "does not exist" not in str(exception).lower():
                raise
        self._collection = self._get_or_create_collection()
        return self.sync_from_sqlite()

    def delete_memory(self, memory_id: str) -> bool:
        """Delete one Chroma vector without changing SQLite."""

        if self.get_metadata(memory_id) is None:
            return False
        self._collection.delete(ids=[memory_id])
        return True

    def get_metadata(self, memory_id: str) -> MemoryIndexMetadata | None:
        """Return only the minimal index metadata for one Chroma ID."""

        result = self._collection.get(ids=[memory_id], include=["metadatas"])
        if not result["ids"]:
            return None
        metadata = result["metadatas"][0] if result["metadatas"] else None
        if metadata is None:
            return None
        return MemoryIndexMetadata.model_validate(metadata)

    def similarity_search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[MemoryVectorSearchHit]:
        """Search vectors and rehydrate every accepted hit from current SQLite."""

        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        vector_count = self._collection.count()
        if vector_count == 0:
            return []

        query_vector = self._embeddings.embed_query(query)
        if not query_vector:
            raise ValueError("Embedding provider returned an empty query vector")
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=vector_count,
            include=["metadatas", "distances"],
        )

        ids = _first_query_result(result.get("ids"))
        distances = _first_query_result(result.get("distances"))
        metadatas = _first_query_result(result.get("metadatas"))
        hits: list[MemoryVectorSearchHit] = []
        for memory_id, distance, raw_metadata in zip(
            ids,
            distances,
            metadatas,
            strict=True,
        ):
            memory = self._repository.get_memory(str(memory_id))
            if memory is None or raw_metadata is None:
                continue
            metadata = MemoryIndexMetadata.model_validate(raw_metadata)
            if (
                metadata.memory_id != memory.memory_id
                or metadata.embedding_version != self._embedding_version
                or metadata.content_hash != _content_hash(_index_content(memory))
            ):
                continue
            hits.append(
                MemoryVectorSearchHit(
                    memory_id=memory.memory_id,
                    distance=float(distance),
                    memory=memory,
                )
            )
            if len(hits) == top_k:
                break
        return hits

    def _get_or_create_collection(self) -> Collection:
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def _index_content(memory: MemoryRecord) -> str:
    """Return the intentionally small, deterministic text embedded for retrieval."""

    return f"{memory.title.strip()}\n{memory.summary.strip()}"


def _content_hash(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _first_query_result(values: Sequence[Sequence[object]] | None) -> Sequence[object]:
    if not values:
        return ()
    return values[0]
