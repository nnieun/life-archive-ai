"""Isolated evaluation tests with deterministic providers."""

import pytest

from evaluation.embedding_comparison import compare_embeddings, retrieval_metrics
from evaluation.runner import EvaluationDataset
from backend.app.models.qa import GroundedAnswerDraft, CitedClaim, AnswerVerification


class Embeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0] if "school" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_metrics_use_labeled_relevant_memories():
    metrics = retrieval_metrics(["wrong", "right"], ["right", "missing"])
    assert metrics == {"recall_at_k": 0.5, "precision_at_k": 0.5, "mrr": 0.5}


def test_two_indexes_are_isolated_and_existing_runs_are_preserved(tmp_path):
    dataset = EvaluationDataset(
        dataset_id="test", description="synthetic",
        events=[{"memory_id": "eval_school", "text": "school"},
                {"memory_id": "eval_sea", "text": "sea"}],
        queries=[{"query_id": "q_school", "question": "school", "relevant_memory_ids": ["eval_school"]}],
    )
    output = tmp_path / "run"
    rows = compare_embeddings(dataset, {"openai": Embeddings(), "ollama": Embeddings()}, output, top_k=1, repeats=2)
    assert len(rows) == 8
    assert all(row["recall_at_k"] == 1 for row in rows)
    assert (output / "openai/chroma.sqlite3").exists()
    assert (output / "ollama/chroma.sqlite3").exists()
    assert (output / "evaluation.sqlite3").exists()
    before = (output / "results.json").read_bytes()
    with pytest.raises(FileExistsError):
        compare_embeddings(dataset, {"openai": Embeddings()}, output)
    assert (output / "results.json").read_bytes() == before


def test_citation_ids_and_llm_judgment_are_reported_separately(tmp_path):
    class Answer:
        def invoke(self, messages):
            return GroundedAnswerDraft(claims=[CitedClaim(text="unsupported", memory_ids=["eval_missing"])])

    class Verifier:
        def invoke(self, messages):
            return AnswerVerification(passed=True, reason="test judgment")

    dataset = EvaluationDataset(
        dataset_id="test", description="synthetic",
        events=[{"memory_id": "eval_school", "text": "school"}],
        queries=[{"query_id": "q_school", "question": "school", "relevant_memory_ids": ["eval_school"]}],
    )
    rows = compare_embeddings(dataset, {"ollama": Embeddings()}, tmp_path / "answers", repeats=1,
                              answer_model=Answer(), verifier_model=Verifier())
    assert all(row["citation_in_retrieved_rate"] == 0 for row in rows)
    assert all(row["citation_gold_match_rate"] == 0 for row in rows)
    assert all(row["llm_verifier_passed"] for row in rows)
