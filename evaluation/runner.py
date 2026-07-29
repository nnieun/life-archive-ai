"""Deterministic, privacy-safe retrieval and generation evaluation."""

from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rank_bm25 import BM25Plus

from backend.app.models.chunk import ChunkingConfig
from backend.app.services.chunking import chunk_transcript
from backend.app.services.retrieval import reciprocal_rank_fusion, tokenize_for_bm25

ChunkStrategy = Literal["256", "512", "1024", "event_aware"]
SearchMethod = Literal["dense", "mmr", "bm25", "hybrid"]

CHUNK_STRATEGIES: tuple[ChunkStrategy, ...] = (
    "256",
    "512",
    "1024",
    "event_aware",
)
SEARCH_METHODS: tuple[SearchMethod, ...] = (
    "dense",
    "mmr",
    "bm25",
    "hybrid",
)
TOP_K_VALUES = (3, 5, 10)

_SEMANTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "school_finish": ("졸업", "마친", "졸업식"),
    "first_job": ("첫 출근", "처음 회사", "첫 직장"),
    "seaside": ("바닷가", "해변", "모래사장"),
    "relocation": ("이사", "이삿짐", "거처를 옮긴", "새 동네"),
    "library": ("도서관", "책을 분류", "서가"),
    "online_learning": ("온라인", "비대면", "강의", "수업"),
    "programming": ("코딩", "프로그래밍", "파이썬"),
    "gardening": ("텃밭", "재배", "수확", "심었다"),
    "running": ("달리기", "마라톤", "완주"),
}


class EvaluationEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str = Field(pattern=r"^eval_[a-z0-9_]+$")
    text: str = Field(min_length=1)


class EvaluationQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(pattern=r"^q_[a-z0-9_]+$")
    question: str = Field(min_length=1)
    relevant_memory_ids: list[str] = Field(min_length=1)

    @field_validator("relevant_memory_ids")
    @classmethod
    def require_unique_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("relevant_memory_ids must be unique")
        return value


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    description: str
    events: list[EvaluationEvent] = Field(min_length=1)
    queries: list[EvaluationQuery] = Field(min_length=1)

    @field_validator("events")
    @classmethod
    def require_unique_memories(
        cls,
        value: list[EvaluationEvent],
    ) -> list[EvaluationEvent]:
        memory_ids = [event.memory_id for event in value]
        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError("memory_id values must be unique")
        return value


@dataclass(frozen=True)
class EventSpan:
    memory_id: str
    text: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class EvaluationChunk:
    chunk_id: str
    text: str
    start_offset: int
    end_offset: int
    memory_ids: tuple[str, ...]


@dataclass(frozen=True)
class RankedChunk:
    chunk: EvaluationChunk
    score: float


def load_dataset(path: Path) -> EvaluationDataset:
    """Load and validate a JSON dataset without executing its contents."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    dataset = EvaluationDataset.model_validate(raw)
    event_ids = {event.memory_id for event in dataset.events}
    missing = {
        memory_id
        for query in dataset.queries
        for memory_id in query.relevant_memory_ids
        if memory_id not in event_ids
    }
    if missing:
        raise ValueError("Queries reference unknown evaluation memories")
    return dataset


def build_transcript(
    events: list[EvaluationEvent],
) -> tuple[str, list[EventSpan]]:
    """Join synthetic events and retain exact source offsets."""

    pieces: list[str] = []
    spans: list[EventSpan] = []
    cursor = 0
    for index, event in enumerate(events):
        if index:
            pieces.append("\n\n")
            cursor += 2
        start = cursor
        pieces.append(event.text)
        cursor += len(event.text)
        spans.append(
            EventSpan(
                memory_id=event.memory_id,
                text=event.text,
                start_offset=start,
                end_offset=cursor,
            )
        )
    return "".join(pieces), spans


def build_chunks(
    transcript: str,
    spans: list[EventSpan],
    strategy: ChunkStrategy,
) -> list[EvaluationChunk]:
    """Apply fixed or event-aware candidates to the same transcript."""

    if strategy == "event_aware":
        return [
            EvaluationChunk(
                chunk_id=f"event_{index:02d}",
                text=span.text,
                start_offset=span.start_offset,
                end_offset=span.end_offset,
                memory_ids=(span.memory_id,),
            )
            for index, span in enumerate(spans)
        ]

    chunk_size = int(strategy)
    overlap = {256: 32, 512: 64, 1024: 128}[chunk_size]
    fixed_chunks = chunk_transcript(
        "evaluation_transcript",
        transcript,
        ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        ),
    )
    return [
        EvaluationChunk(
            chunk_id=chunk.segment_id,
            text=chunk.content,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            memory_ids=tuple(
                span.memory_id
                for span in spans
                if span.start_offset < chunk.end_offset
                and span.end_offset > chunk.start_offset
            ),
        )
        for chunk in fixed_chunks
    ]


def _semantic_tokens(text: str) -> list[str]:
    tokens = tokenize_for_bm25(text)
    normalized = text.casefold()
    for concept, aliases in _SEMANTIC_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            tokens.append(f"concept:{concept}")
    return tokens


def _cosine(left: list[str], right: list[str]) -> float:
    left_counts = Counter(left)
    right_counts = Counter(right)
    numerator = sum(
        count * right_counts.get(token, 0)
        for token, count in left_counts.items()
    )
    left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
    right_norm = math.sqrt(
        sum(count * count for count in right_counts.values())
    )
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _dense_scores(
    chunks: list[EvaluationChunk],
    query: str,
) -> list[RankedChunk]:
    query_tokens = _semantic_tokens(query)
    ranked = [
        RankedChunk(
            chunk=chunk,
            score=_cosine(query_tokens, _semantic_tokens(chunk.text)),
        )
        for chunk in chunks
    ]
    return sorted(ranked, key=lambda item: (-item.score, item.chunk.chunk_id))


def _mmr_scores(
    chunks: list[EvaluationChunk],
    query: str,
    *,
    lambda_mult: float = 0.7,
) -> list[RankedChunk]:
    query_tokens = _semantic_tokens(query)
    document_tokens = [_semantic_tokens(chunk.text) for chunk in chunks]
    relevance = [
        _cosine(query_tokens, tokens)
        for tokens in document_tokens
    ]
    remaining = set(range(len(chunks)))
    selected: list[int] = []
    ranked: list[RankedChunk] = []
    while remaining:
        candidate = max(
            remaining,
            key=lambda index: (
                lambda_mult * relevance[index]
                - (1.0 - lambda_mult)
                * max(
                    (
                        _cosine(document_tokens[index], document_tokens[other])
                        for other in selected
                    ),
                    default=0.0,
                ),
                -index,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
        ranked.append(
            RankedChunk(
                chunk=chunks[candidate],
                score=1.0 / len(selected),
            )
        )
    return ranked


def _bm25_scores(
    chunks: list[EvaluationChunk],
    query: str,
) -> list[RankedChunk]:
    corpus = [tokenize_for_bm25(chunk.text) for chunk in chunks]
    engine = BM25Plus(corpus)
    scores = engine.get_scores(tokenize_for_bm25(query))
    ranked = [
        RankedChunk(chunk=chunk, score=float(score))
        for chunk, score in zip(chunks, scores, strict=True)
    ]
    return sorted(ranked, key=lambda item: (-item.score, item.chunk.chunk_id))


def rank_chunks(
    chunks: list[EvaluationChunk],
    query: str,
    method: SearchMethod,
) -> list[RankedChunk]:
    """Rank chunks with one deterministic evaluation candidate."""

    if method == "dense":
        return _dense_scores(chunks, query)
    if method == "mmr":
        return _mmr_scores(chunks, query)
    if method == "bm25":
        return _bm25_scores(chunks, query)

    dense = _dense_scores(chunks, query)
    sparse = _bm25_scores(chunks, query)
    scores = reciprocal_rank_fusion(
        [
            [item.chunk.chunk_id for item in dense],
            [item.chunk.chunk_id for item in sparse],
        ]
    )
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [
        RankedChunk(chunk=chunks_by_id[chunk_id], score=scores[chunk_id])
        for chunk_id in sorted(
            scores,
            key=lambda item: (-scores[item], item),
        )
    ]


def rank_memories(
    ranked_chunks: list[RankedChunk],
    spans: list[EventSpan],
) -> list[str]:
    """Aggregate chunk scores into unique source-memory rankings."""

    spans_by_id = {span.memory_id: span for span in spans}
    scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    for rank, item in enumerate(ranked_chunks, start=1):
        for memory_id in item.chunk.memory_ids:
            span = spans_by_id[memory_id]
            overlap = max(
                0,
                min(item.chunk.end_offset, span.end_offset)
                - max(item.chunk.start_offset, span.start_offset),
            )
            coverage = overlap / (span.end_offset - span.start_offset)
            weighted_score = item.score * coverage
            scores[memory_id] = max(scores.get(memory_id, 0.0), weighted_score)
            first_rank.setdefault(memory_id, rank)
    return sorted(
        scores,
        key=lambda memory_id: (
            -scores[memory_id],
            first_rank[memory_id],
            memory_id,
        ),
    )


def _join_ids(values: list[str]) -> str:
    return "|".join(values)


def evaluate(dataset: EvaluationDataset) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all specified retrieval and deterministic generation conditions."""

    transcript, spans = build_transcript(dataset.events)
    retrieval_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []

    for strategy in CHUNK_STRATEGIES:
        chunks = build_chunks(transcript, spans, strategy)
        for method in SEARCH_METHODS:
            for query in dataset.queries:
                started = time.perf_counter()
                ranked_chunks = rank_chunks(chunks, query.question, method)
                ranked_memories = rank_memories(ranked_chunks, spans)
                retrieval_ms = (time.perf_counter() - started) * 1000
                relevant = set(query.relevant_memory_ids)

                for top_k in TOP_K_VALUES:
                    selected = ranked_memories[:top_k]
                    matched = relevant.intersection(selected)
                    recall = len(matched) / len(relevant)
                    retrieval_rows.append(
                        {
                            "dataset_id": dataset.dataset_id,
                            "query_id": query.query_id,
                            "chunk_strategy": strategy,
                            "search_method": method,
                            "top_k": top_k,
                            "relevant_memory_ids": _join_ids(
                                query.relevant_memory_ids
                            ),
                            "retrieved_memory_ids": _join_ids(selected),
                            "recall_at_k": f"{recall:.4f}",
                            "contains_answer": int(bool(matched)),
                            "retrieval_latency_ms": f"{retrieval_ms:.4f}",
                        }
                    )

                    generation_started = time.perf_counter()
                    cited = selected[:1]
                    correct_citation = bool(cited) and set(cited).issubset(relevant)
                    unsupported = not correct_citation
                    generated_answer = (
                        "검색된 합성 기억을 근거로 답변했습니다."
                        if correct_citation
                        else "검색 결과가 정답 기억을 뒷받침하지 못했습니다."
                    )
                    generation_ms = (
                        time.perf_counter() - generation_started
                    ) * 1000
                    generation_rows.append(
                        {
                            "dataset_id": dataset.dataset_id,
                            "query_id": query.query_id,
                            "chunk_strategy": strategy,
                            "search_method": method,
                            "top_k": top_k,
                            "answer": generated_answer,
                            "cited_memory_ids": _join_ids(cited),
                            "citation_correctness": (
                                "1.0000" if correct_citation else "0.0000"
                            ),
                            "unsupported_answer": int(unsupported),
                            "end_to_end_latency_ms": (
                                f"{retrieval_ms + generation_ms:.4f}"
                            ),
                        }
                    )

    return retrieval_rows, generation_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Evaluation rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    retrieval_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    generation_by_key = {
        (
            row["query_id"],
            row["chunk_strategy"],
            row["search_method"],
            row["top_k"],
        ): row
        for row in generation_rows
    }
    grouped: dict[tuple[str, str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row in retrieval_rows:
        key = (
            row["chunk_strategy"],
            row["search_method"],
            int(row["top_k"]),
        )
        generation = generation_by_key[
            (
                row["query_id"],
                row["chunk_strategy"],
                row["search_method"],
                row["top_k"],
            )
        ]
        grouped.setdefault(key, []).append((row, generation))

    aggregates: list[dict[str, Any]] = []
    for key, pairs in sorted(grouped.items()):
        retrieval = [pair[0] for pair in pairs]
        generation = [pair[1] for pair in pairs]
        aggregates.append(
            {
                "chunk_strategy": key[0],
                "search_method": key[1],
                "top_k": key[2],
                "recall": statistics.fmean(
                    float(row["recall_at_k"]) for row in retrieval
                ),
                "hit_rate": statistics.fmean(
                    int(row["contains_answer"]) for row in retrieval
                ),
                "citation": statistics.fmean(
                    float(row["citation_correctness"])
                    for row in generation
                ),
                "unsupported": statistics.fmean(
                    int(row["unsupported_answer"]) for row in generation
                ),
                "retrieval_ms": statistics.fmean(
                    float(row["retrieval_latency_ms"]) for row in retrieval
                ),
                "e2e_ms": statistics.fmean(
                    float(row["end_to_end_latency_ms"])
                    for row in generation
                ),
            }
        )
    return aggregates


def write_summary(
    path: Path,
    dataset: EvaluationDataset,
    retrieval_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
) -> None:
    """Write a compact comparison with limitations and reproduction details."""

    aggregates = _aggregate(retrieval_rows, generation_rows)
    best = max(
        aggregates,
        key=lambda row: (
            row["recall"],
            row["citation"],
            -row["unsupported"],
            -row["retrieval_ms"],
        ),
    )
    lines = [
        "# Retrieval Evaluation Summary",
        "",
        "## Scope",
        "",
        f"- Dataset: `{dataset.dataset_id}`",
        f"- Synthetic memories: {len(dataset.events)}",
        f"- Evaluation queries: {len(dataset.queries)}",
        "- Chunk candidates: 256, 512, 1024 characters, event-aware",
        "- Search candidates: dense similarity, MMR, BM25, hybrid RRF",
        "- Top-K values: 3, 5, 10",
        "- No OpenAI API or personal transcript was used.",
        "",
        "## Aggregate Results",
        "",
        "| Chunk | Search | K | Recall@K | Citation | Unsupported | Retrieval ms | E2E ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['chunk_strategy']} | {row['search_method']} | "
            f"{row['top_k']} | {row['recall']:.3f} | "
            f"{row['citation']:.3f} | {row['unsupported']:.3f} | "
            f"{row['retrieval_ms']:.4f} | {row['e2e_ms']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best Observed Configuration",
            "",
            f"- Chunk: `{best['chunk_strategy']}`",
            f"- Search: `{best['search_method']}`",
            f"- Top-K: `{best['top_k']}`",
            f"- Recall@K: `{best['recall']:.3f}`",
            f"- Citation correctness: `{best['citation']:.3f}`",
            f"- Unsupported answer rate: `{best['unsupported']:.3f}`",
            "",
            "## Reproduction",
            "",
            "```powershell",
            r".\.venv\Scripts\python.exe scripts\run_evaluation.py",
            "```",
            "",
            f"- Python: `{platform.python_version()}`",
            "- Rankings and quality metrics are deterministic for this dataset.",
            "- Latency is measured locally and can vary by machine and background load.",
            "- Failed query rows remain in both CSV files with zero quality scores.",
            "",
            "## Limitations",
            "",
            "- Dense similarity uses deterministic lexical-semantic aliases instead of an external embedding API.",
            "- Generation evaluation uses a deterministic grounded-answer simulator, not an LLM.",
            "- Results compare MVP settings on a small synthetic corpus and are not a claim about production accuracy.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(
    dataset_path: Path,
    reports_directory: Path,
) -> tuple[Path, Path, Path]:
    """Execute the matrix and persist all required TASK-015 artifacts."""

    dataset = load_dataset(dataset_path)
    retrieval_rows, generation_rows = evaluate(dataset)
    retrieval_path = reports_directory / "retrieval_results.csv"
    generation_path = reports_directory / "generation_results.csv"
    summary_path = reports_directory / "experiment_summary.md"
    _write_csv(retrieval_path, retrieval_rows)
    _write_csv(generation_path, generation_rows)
    write_summary(
        summary_path,
        dataset,
        retrieval_rows,
        generation_rows,
    )
    return retrieval_path, generation_path, summary_path
