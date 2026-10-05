"""Compare OpenAI and Ollama without touching production indexes."""

import argparse
import json
from hashlib import sha256
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings
from backend.app.models.qa import AnswerVerification, GroundedAnswerDraft
from backend.app.services.vector_index import create_openai_embeddings
from evaluation.embedding_comparison import OllamaEmbeddings, compare_embeddings
from evaluation.runner import load_dataset
from langchain_openai import ChatOpenAI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--answers", action="store_true", help="Also generate and verify answers locally")
    parser.add_argument("--query-limit", type=int, help="Limit questions for a short answer smoke comparison")
    args = parser.parse_args()
    settings = get_settings()
    if not settings.openai_api_key:
        parser.error("OPENAI_API_KEY is required for the OpenAI comparison arm")
    dataset = load_dataset(PROJECT_ROOT / "evaluation/dataset.json")
    if args.query_limit is not None:
        if args.query_limit < 1:
            parser.error("--query-limit must be positive")
        dataset = dataset.model_copy(update={"queries": dataset.queries[:args.query_limit]})
    output = PROJECT_ROOT / "data/processed/embedding_comparison" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    models = {}
    if args.answers:
        chat = ChatOpenAI(model=settings.ollama_model, api_key="ollama", base_url=settings.ollama_base_url,
                          timeout=300, max_retries=0, temperature=0, max_tokens=512, use_responses_api=False)
        models = {"answer_model": chat.with_structured_output(GroundedAnswerDraft, method="json_schema"),
                  "verifier_model": chat.with_structured_output(AnswerVerification, method="json_schema")}
    print("Using synthetic memories; OpenAI embedding calls incur costs. Production indexes are untouched.", flush=True)
    openai_embeddings = create_openai_embeddings(settings.openai_embedding_model, api_key=settings.openai_api_key)
    # Synthetic memories are short. Send strings directly without downloading
    # tokenization assets, and fail promptly on network problems.
    openai_embeddings.check_embedding_ctx_length = False
    openai_embeddings.request_timeout = 30
    openai_embeddings.max_retries = 0
    rows = compare_embeddings(dataset, {
        "openai": openai_embeddings,
        "ollama": OllamaEmbeddings(base_url=settings.ollama_base_url.removesuffix("/v1")),
    }, output, top_k=args.top_k, repeats=args.repeats, **models)
    (output / "manifest.json").write_text(json.dumps({
        "dataset_id": dataset.dataset_id,
        "dataset_sha256": sha256((PROJECT_ROOT / "evaluation/dataset.json").read_bytes()).hexdigest(),
        "openai_embedding_model": settings.openai_embedding_model,
        "ollama_embedding_model": "embeddinggemma",
        "answer_model": settings.ollama_model if args.answers else None,
        "top_k": args.top_k, "repeats": args.repeats,
        "evaluated_query_ids": [query.query_id for query in dataset.queries],
    }, indent=2), encoding="utf-8")
    for provider in ("openai", "ollama"):
        for method in ("dense", "hybrid"):
            group = [row for row in rows if row["provider"] == provider and row["method"] == method]
            means = {key: sum(row[key] for row in group) / len(group) for key in ("recall_at_k", "precision_at_k", "mrr", "retrieval_ms")}
            print(provider, method, json.dumps(means))
    print(output.relative_to(PROJECT_ROOT) / "results.json")


if __name__ == "__main__":
    main()
