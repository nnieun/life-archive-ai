"""Opt-in retrieval evaluation using a real embedding provider.

This module is intentionally separate from the deterministic evaluation. It
may make paid external API calls and must never run as part of the test suite.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Protocol

from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Plus

from backend.app.services.retrieval import reciprocal_rank_fusion, tokenize_for_bm25
from evaluation.runner import (
    CHUNK_STRATEGIES,
    TOP_K_VALUES,
    EvaluationChunk,
    EvaluationDataset,
    build_chunks,
    build_transcript,
    load_dataset,
)

REAL_SEARCH_METHODS = ("dense", "mmr", "bm25", "hybrid")
EMBEDDING_SEARCH_METHODS = frozenset({"dense", "mmr", "hybrid"})
EMBEDDING_INPUT_USD_PER_MILLION_TOKENS = 0.02


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _dense_rank(
    chunks: list[EvaluationChunk],
    query_vector: list[float],
    document_vectors: list[list[float]],
) -> list[int]:
    return sorted(
        range(len(chunks)),
        key=lambda index: (-_cosine(query_vector, document_vectors[index]), index),
    )


def _mmr_rank(
    chunks: list[EvaluationChunk],
    query_vector: list[float],
    document_vectors: list[list[float]],
    *,
    lambda_mult: float = 0.7,
) -> list[int]:
    relevance = [
        _cosine(query_vector, vector) for vector in document_vectors
    ]
    remaining = set(range(len(chunks)))
    selected: list[int] = []
    while remaining:
        index = max(
            remaining,
            key=lambda candidate: (
                lambda_mult * relevance[candidate]
                - (1.0 - lambda_mult)
                * max(
                    (
                        _cosine(
                            document_vectors[candidate],
                            document_vectors[other],
                        )
                        for other in selected
                    ),
                    default=0.0,
                ),
                -candidate,
            ),
        )
        selected.append(index)
        remaining.remove(index)
    return selected


def _bm25_rank(chunks: list[EvaluationChunk], query: str) -> list[int]:
    corpus = [tokenize_for_bm25(chunk.text) for chunk in chunks]
    engine = BM25Plus(corpus)
    scores = engine.get_scores(tokenize_for_bm25(query))
    return sorted(range(len(chunks)), key=lambda index: (-scores[index], index))


def _rank_chunks(
    chunks: list[EvaluationChunk],
    query: str,
    method: str,
    embeddings: EmbeddingProvider,
    document_vectors: list[list[float]],
    query_vector: list[float] | None = None,
) -> list[EvaluationChunk]:
    if query_vector is None:
        query_vector = embeddings.embed_query(query)
    if method == "dense":
        indexes = _dense_rank(chunks, query_vector, document_vectors)
    elif method == "mmr":
        indexes = _mmr_rank(chunks, query_vector, document_vectors)
    elif method == "bm25":
        indexes = _bm25_rank(chunks, query)
    else:
        dense = _dense_rank(chunks, query_vector, document_vectors)
        sparse = _bm25_rank(chunks, query)
        dense_ids = [chunks[index].chunk_id for index in dense]
        sparse_ids = [chunks[index].chunk_id for index in sparse]
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids])
        indexes = sorted(
            range(len(chunks)),
            key=lambda index: (-fused[chunks[index].chunk_id], index),
        )
    return [chunks[index] for index in indexes]


def _memory_ranking(
    ranked_chunks: list[EvaluationChunk],
    spans: list,
) -> list[str]:
    spans_by_id = {span.memory_id: span for span in spans}
    scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    for rank, chunk in enumerate(ranked_chunks, start=1):
        for memory_id in chunk.memory_ids:
            span = spans_by_id[memory_id]
            overlap = max(
                0,
                min(chunk.end_offset, span.end_offset)
                - max(chunk.start_offset, span.start_offset),
            )
            coverage = overlap / (span.end_offset - span.start_offset)
            score = (1.0 / rank) * coverage
            scores[memory_id] = max(scores.get(memory_id, 0.0), score)
            first_rank.setdefault(memory_id, rank)
    return sorted(scores, key=lambda item: (-scores[item], first_rank[item], item))


def evaluate_real(
    dataset: EvaluationDataset,
    embeddings: EmbeddingProvider,
) -> list[dict[str, object]]:
    """Evaluate retrieval only; generation remains in the deterministic track."""

    transcript, spans = build_transcript(dataset.events)
    rows: list[dict[str, object]] = []
    for strategy in CHUNK_STRATEGIES:
        chunks = build_chunks(transcript, spans, strategy)
        document_started = time.perf_counter()
        document_vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
        document_embedding_latency_ms = (time.perf_counter() - document_started) * 1000
        if len(document_vectors) != len(chunks):
            raise ValueError("Embedding provider returned an unexpected vector count")
        query_vectors: dict[str, list[float]] = {}
        query_embedding_latency_ms: dict[str, float] = {}
        for query in dataset.queries:
            query_started = time.perf_counter()
            query_vectors[query.query_id] = embeddings.embed_query(query.question)
            query_embedding_latency_ms[query.query_id] = (
                time.perf_counter() - query_started
            ) * 1000
        for method in REAL_SEARCH_METHODS:
            for query in dataset.queries:
                started = time.perf_counter()
                ranked_chunks = _rank_chunks(
                    chunks,
                    query.question,
                    method,
                    embeddings,
                    document_vectors,
                    query_vectors[query.query_id],
                )
                ranked_memories = _memory_ranking(ranked_chunks, spans)
                ranking_latency_ms = (time.perf_counter() - started) * 1000
                relevant = set(query.relevant_memory_ids)
                query_input_length = (
                    len(query.question) if method in EMBEDDING_SEARCH_METHODS else 0
                )
                estimated_input_tokens = math.ceil(
                    (sum(len(chunk.text) for chunk in chunks) + query_input_length) / 4
                )
                estimated_cost_usd = (
                    estimated_input_tokens
                    / 1_000_000
                    * EMBEDDING_INPUT_USD_PER_MILLION_TOKENS
                )
                for top_k in TOP_K_VALUES:
                    selected = ranked_memories[:top_k]
                    matched = relevant.intersection(selected)
                    rows.append(
                        {
                            "dataset_id": dataset.dataset_id,
                            "query_id": query.query_id,
                            "chunk_strategy": strategy,
                            "search_method": method,
                            "top_k": top_k,
                            "retrieved_memory_ids": "|".join(selected),
                            "recall_at_k": len(matched) / len(relevant),
                            "contains_answer": int(bool(matched)),
                            "retrieval_latency_ms": (
                                document_embedding_latency_ms
                                + query_embedding_latency_ms[query.query_id]
                                + ranking_latency_ms
                            ),
                            "document_embedding_latency_ms": document_embedding_latency_ms,
                            "query_embedding_latency_ms": query_embedding_latency_ms[
                                query.query_id
                            ],
                            "ranking_latency_ms": ranking_latency_ms,
                            "estimated_input_tokens": estimated_input_tokens,
                            "estimated_embedding_cost_usd": estimated_cost_usd,
                        }
                    )
    return rows


def run_real_evaluation(
    dataset_path: Path,
    reports_directory: Path,
    *,
    embedding_model: str = "text-embedding-3-small",
) -> Path:
    """Run the opt-in matrix and write a CSV plus a reproducibility manifest."""

    dataset = load_dataset(dataset_path)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    rows = evaluate_real(dataset, embeddings)
    reports_directory.mkdir(parents=True, exist_ok=True)
    csv_path = reports_directory / "real_retrieval_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = reports_directory / "real_evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset.dataset_id,
                "embedding_provider": "OpenAIEmbeddings",
                "embedding_model": embedding_model,
                "search_methods": REAL_SEARCH_METHODS,
                "top_k": TOP_K_VALUES,
                "personal_data_used": False,
                "api_key_required": True,
                "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path
