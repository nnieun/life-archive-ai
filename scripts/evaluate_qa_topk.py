"""Compare top-k on isolated synthetic memories, with real local retrieval and QA."""

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.core.config import get_settings
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
from backend.app.models.transcript import LoadedTranscript
from backend.app.services.qa import GroundedQAService, QAModels, build_openai_qa_models
from backend.app.services.retrieval import BM25MemoryIndex, HybridMemoryRetriever
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import MemoryCreate, MemorySourceCreate, TranscriptSegmentCreate
from backend.app.storage.repository import SQLiteRepository
from evaluation.qa_topk import grade_result, paired_order, summarize
from scripts.benchmark_qa import TimedModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    settings = get_settings()
    if settings.llm_provider != 'ollama' or settings.embedding_provider != 'ollama':
        raise SystemExit('This experiment requires local generation and embeddings.')
    output = ROOT / 'data/processed/qa_topk' / (datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:6])
    output.mkdir(parents=True)
    database = SQLiteDatabase(output / 'evaluation.sqlite3')
    database.initialize()
    repository = SQLiteRepository(database)
    dataset = json.loads((ROOT / 'evaluation/dataset.json').read_text(encoding='utf-8'))
    cases_path = ROOT / 'evaluation/qa_topk_cases.json'
    cases = json.loads(cases_path.read_text(encoding='utf-8'))['cases']
    events = dataset['events'] + [{'memory_id': 'eval_radio',
        'text': '가상의 영수는 1967년 또는 1968년경 중학생 때 집에 있던 고장난 라디오를 수리했다. 정확한 연도는 기억하지 못한다.'}]
    for i, event in enumerate(events):
        mid, text = event['memory_id'], event['text']
        tid, sid = f'tr_{i}', f'seg_{i}'
        repository.create_transcript(LoadedTranscript(transcript_id=tid, filename=f'synthetic_{i}.txt',
            source_type='stt_text', uploaded_at=datetime.now(UTC), content_hash=sha256(text.encode()).hexdigest(),
            raw_content=text, normalized_content=text))
        repository.create_segments([TranscriptSegmentCreate(segment_id=sid, transcript_id=tid, chunk_index=0,
            content=text, start_offset=0, end_offset=len(text))])
        repository.create_memory(MemoryCreate(memory_id=mid, transcript_id=tid, title=mid,
            summary=text, confidence=.95))
        repository.create_memory_source(MemorySourceCreate(memory_source_id=f'src_{i}', memory_id=mid,
            transcript_id=tid, segment_id=sid, start_offset=0, end_offset=len(text)))
    index = MemoryVectorIndex(repository, output / 'chroma', embedding_model=settings.embedding_model,
        embedding_provider='ollama', embedding_base_url=settings.ollama_base_url)
    index.sync_from_sqlite()
    bm25 = BM25MemoryIndex(repository)
    bm25.rebuild_from_sqlite()
    retriever = HybridMemoryRetriever(repository, index, bm25)
    raw_models = build_openai_qa_models(settings.chat_model, base_url=settings.chat_base_url,
        combined=settings.qa_combined, native_ollama=True, max_tokens=settings.qa_max_tokens,
        keep_alive=settings.ollama_keep_alive, verification_model=settings.qa_verification_model or None)
    calls = []
    models = QAModels(**{name: TimedModel(getattr(raw_models, name), name, calls)
                        for name in ('evidence', 'answer', 'verification', 'rewrite')})
    service = GroundedQAService(repository, retriever, models, provider='ollama', model_name=settings.chat_model,
        combined=settings.qa_combined, compact=settings.qa_compact, max_rewrites=0, cache_enabled=False)
    manifest = {'created_at': datetime.now(UTC).isoformat(), 'synthetic_only': True,
        'model': settings.chat_model, 'embedding': settings.embedding_model, 'repeats': args.repeats,
        'cache': False, 'verification': True, 'max_rewrites': 0, 'max_tokens': settings.qa_max_tokens,
        'combined': settings.qa_combined, 'compact': settings.qa_compact,
        'cases_sha256': sha256(cases_path.read_bytes()).hexdigest(),
        'dataset_sha256': sha256((ROOT / 'evaluation/dataset.json').read_bytes()).hexdigest(),
        'order': 'alternating 5/3 and 3/5 by case and repetition; sequential',
        'scope': 'warm QA end-to-end including actual BGE-M3 retrieval; index build and warmup excluded'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Synthetic index ready; warmup then paired runs.', flush=True)
    rows = []
    try:
        service.answer_question(session_id='warmup', question=cases[0]['question'], top_k=5)
        for repeat in range(args.repeats):
            for case_index, case in enumerate(cases):
                for top_k in paired_order(repeat, case_index):
                    calls.clear()
                    started = perf_counter()
                    result = service.answer_question(session_id=f"run_{repeat}_{case['id']}_{top_k}",
                        question=case['question'], top_k=top_k)
                    row = {'case_id': case['id'], 'kind': case['kind'], 'repeat': repeat+1, 'top_k': top_k,
                        'seconds': round(perf_counter()-started, 3), 'passed_verification': result.validation_result.passed,
                        'failure_code': result.validation_result.failure_code, 'cache_hit': result.cache_hit,
                        'answer': result.final_answer, 'retrieved_ids': result.retrieved_memory_ids,
                        'cited_ids': sorted({citation.memory_id for citation in result.citations}),
                        'steps': [{'node': step.node, 'seconds': round(step.elapsed_ms/1000, 3)} for step in result.steps],
                        'model_calls': list(calls), **grade_result(case, result)}
                    rows.append(row)
                    with (output / 'results.jsonl').open('a', encoding='utf-8') as file:
                        file.write(json.dumps(row, ensure_ascii=False)+'\n')
                    print(json.dumps({key: row[key] for key in ('case_id','repeat','top_k','seconds','acceptance_passed','failure_code')}), flush=True)
        summary = summarize(rows)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps({'output': str(output.relative_to(ROOT)), 'summary': summary}), flush=True)
    finally:
        for client in {getattr(model, 'client', None) for model in
                       (raw_models.evidence, raw_models.answer, raw_models.verification, raw_models.rewrite)}:
            if client is not None:
                client.close()
        database.close()


if __name__ == '__main__':
    main()
