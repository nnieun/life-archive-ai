"""Compare the previous hybrid path with entity-aware retrieval on synthetic data."""

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
from backend.app.services.qa import GroundedQAService, build_openai_qa_models
from backend.app.services.retrieval import BM25MemoryIndex, HybridMemoryRetriever
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import MemoryCreate, MemorySourceCreate, TranscriptSegmentCreate
from backend.app.storage.repository import SQLiteRepository
from evaluation.entity_aware_qa import score, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    settings = get_settings()
    source_path = ROOT / 'evaluation/entity_aware_cases.json'
    dataset = json.loads(source_path.read_text(encoding='utf-8'))
    output = ROOT / 'data/processed/entity_aware_qa' / (datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:6])
    output.mkdir(parents=True)
    database = SQLiteDatabase(output / 'evaluation.sqlite3')
    database.initialize()
    repository = SQLiteRepository(database)
    for index, event in enumerate(dataset['events']):
        text, mid = event['text'], event['memory_id']
        tid, sid = f'tr_{mid}', f'seg_{mid}'
        repository.create_transcript(LoadedTranscript(transcript_id=tid, filename=f'{mid}.txt',
            source_type='stt_text', uploaded_at=datetime.now(UTC), content_hash=sha256(text.encode()).hexdigest(),
            raw_content=text, normalized_content=text))
        repository.create_segments([TranscriptSegmentCreate(segment_id=sid, transcript_id=tid,
            chunk_index=0, content=text, start_offset=0, end_offset=len(text))])
        repository.create_memory(MemoryCreate(memory_id=mid, transcript_id=tid, title=event['title'],
            summary=text, people=event['people'], confidence=.95))
        repository.create_memory_source(MemorySourceCreate(memory_source_id=f'src_{mid}', memory_id=mid,
            transcript_id=tid, segment_id=sid, start_offset=0, end_offset=len(text)))
    index = MemoryVectorIndex(repository, output/'chroma', embedding_model=settings.embedding_model,
        embedding_provider='ollama', embedding_base_url=settings.ollama_base_url)
    index.sync_from_sqlite()
    bm25 = BM25MemoryIndex(repository)
    bm25.rebuild_from_sqlite()
    retriever = HybridMemoryRetriever(repository, index, bm25)
    models = build_openai_qa_models(settings.chat_model, base_url=settings.chat_base_url,
        combined=True, native_ollama=True, max_tokens=settings.qa_max_tokens,
        keep_alive=settings.ollama_keep_alive)
    services = {variant: GroundedQAService(repository, retriever, models, provider='ollama',
        model_name=settings.chat_model, combined=True, compact=True, max_rewrites=0,
        cache_enabled=False, entity_aware=enabled)
        for variant, enabled in (('baseline', False), ('entity_aware', True))}
    manifest = {'created_at': datetime.now(UTC).isoformat(), 'synthetic_only': True,
        'model': settings.chat_model, 'embedding': settings.embedding_model, 'repeats': args.repeats,
        'verification': True, 'cache': False, 'dataset_sha256': sha256(source_path.read_bytes()).hexdigest(),
        'order': 'variant order alternates by case and repeat',
        'scope': 'end-to-end warm requests; initial indexing and one warmup excluded'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows = []
    try:
        services['baseline'].answer_question(session_id='warmup', question=dataset['cases'][1]['question'], top_k=3)
        for repeat in range(args.repeats):
            for case_index, case in enumerate(dataset['cases']):
                variants = ('baseline','entity_aware') if (repeat+case_index)%2 == 0 else ('entity_aware','baseline')
                for variant in variants:
                    started = perf_counter()
                    result = services[variant].answer_question(session_id=f'{variant}_{repeat}_{case["id"]}',
                        question=case['question'], top_k=3)
                    row = {'variant': variant, 'repeat': repeat+1, 'case_id': case['id'],
                        'kind': case['kind'], 'seconds': round(perf_counter()-started, 3),
                        'failure_code': result.validation_result.failure_code,
                        'retrieved_ids': result.retrieved_memory_ids,
                        'cited_ids': sorted({citation.memory_id for citation in result.citations}),
                        'steps': [{'node': step.node, 'seconds': round(step.elapsed_ms/1000,3)} for step in result.steps],
                        **score(case, result)}
                    rows.append(row)
                    with (output/'results.jsonl').open('a', encoding='utf-8') as file:
                        file.write(json.dumps(row, ensure_ascii=False)+'\n')
                    print(json.dumps({key:row[key] for key in ('variant','case_id','repeat','seconds','acceptance_passed','failure_code')}), flush=True)
        summary = summarize(rows)
        (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps({'output':str(output.relative_to(ROOT)),'summary':summary}), flush=True)
    finally:
        services['baseline'].close()
        database.close()


if __name__ == '__main__':
    main()
