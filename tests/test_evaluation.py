"""TASK-015 evaluation tests."""

from __future__ import annotations

import csv
from pathlib import Path

from evaluation.runner import (
    CHUNK_STRATEGIES,
    SEARCH_METHODS,
    TOP_K_VALUES,
    build_chunks,
    build_transcript,
    evaluate,
    load_dataset,
    run_evaluation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "dataset.json"


def test_synthetic_dataset_has_valid_references_and_no_raw_paths() -> None:
    dataset = load_dataset(DATASET_PATH)
    serialized = DATASET_PATH.read_text(encoding="utf-8")

    assert dataset.dataset_id == "synthetic-life-memories-v1"
    assert len(dataset.events) == 12
    assert len(dataset.queries) == 8
    assert "data/raw" not in serialized
    assert "C:\\" not in serialized


def test_all_chunk_candidates_preserve_event_references() -> None:
    dataset = load_dataset(DATASET_PATH)
    transcript, spans = build_transcript(dataset.events)
    expected_ids = {event.memory_id for event in dataset.events}

    for strategy in CHUNK_STRATEGIES:
        chunks = build_chunks(transcript, spans, strategy)
        referenced_ids = {
            memory_id
            for chunk in chunks
            for memory_id in chunk.memory_ids
        }

        assert chunks
        assert referenced_ids == expected_ids
        assert all(
            chunk.text == transcript[chunk.start_offset:chunk.end_offset]
            for chunk in chunks
        )


def test_evaluation_matrix_is_complete_and_keeps_failures() -> None:
    dataset = load_dataset(DATASET_PATH)

    retrieval, generation = evaluate(dataset)

    expected_rows = (
        len(dataset.queries)
        * len(CHUNK_STRATEGIES)
        * len(SEARCH_METHODS)
        * len(TOP_K_VALUES)
    )
    assert len(retrieval) == expected_rows
    assert len(generation) == expected_rows
    assert {row["search_method"] for row in retrieval} == set(SEARCH_METHODS)
    assert {int(row["top_k"]) for row in retrieval} == set(TOP_K_VALUES)
    assert all("retrieval_latency_ms" in row for row in retrieval)
    assert all("end_to_end_latency_ms" in row for row in generation)
    assert any(int(row["unsupported_answer"]) == 1 for row in generation)


def test_evaluation_outputs_are_reproducible_and_parseable(
    tmp_path: Path,
) -> None:
    retrieval_path, generation_path, summary_path = run_evaluation(
        DATASET_PATH,
        tmp_path / "reports",
    )

    with retrieval_path.open(encoding="utf-8-sig", newline="") as source:
        retrieval_rows = list(csv.DictReader(source))
    with generation_path.open(encoding="utf-8-sig", newline="") as source:
        generation_rows = list(csv.DictReader(source))
    summary = summary_path.read_text(encoding="utf-8")

    assert len(retrieval_rows) == 384
    assert len(generation_rows) == 384
    assert "Best Observed Configuration" in summary
    assert "No OpenAI API or personal transcript was used." in summary
