"""Measure local QA on isolated synthetic memories; never use personal data."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.core.config import get_settings
import os
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
from backend.app.models.memory import DatePrecision
from backend.app.models.retrieval import RetrievalHit
from backend.app.models.transcript import LoadedTranscript
from backend.app.services.qa import GroundedQAService, QAModels, build_openai_qa_models
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import MemoryCreate, MemorySourceCreate, TranscriptSegmentCreate
from backend.app.storage.repository import SQLiteRepository


class TimedModel:
    def __init__(self, model, label, calls):
        self.model, self.label, self.calls = model, label, calls
        self.client = getattr(model, 'client', None)

    def invoke(self, messages):
        start = perf_counter()
        result = None
        try:
            result = self.model.invoke(messages)
            return result
        finally:
            measurement = {'stage': self.label, 'seconds': round(perf_counter()-start, 3),
                           'input_chars': sum(len(message.content) for message in messages)}
            if isinstance(result, dict):
                metadata = getattr(result.get('raw'), 'response_metadata', {})
                for key in ('load_duration', 'prompt_eval_duration', 'eval_duration', 'prompt_eval_count', 'eval_count'):
                    if key in metadata:
                        measurement[key] = metadata[key]
            self.calls.append(measurement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--combined', action='store_true')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--max-rewrites', type=int, default=1)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--model')
    parser.add_argument('--verification-model')
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--keep-alive', default='15m')
    parser.add_argument('--question', choices=['answerable', 'insufficient', 'uncertainty', 'location', 'identity_uncertain'], default='answerable')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    settings = get_settings()
    if settings.llm_provider != 'ollama':
        raise SystemExit('This benchmark is local-only.')
    root = Path('data/processed/qa_performance')
    root.mkdir(parents=True, exist_ok=True)
    database = SQLiteDatabase(root / f'benchmark_{uuid4().hex}.sqlite3')
    database.initialize()
    repo = SQLiteRepository(database)
    facts = [('영수', '고장난 라디오를 수리했다.'), ('민수', '학교 운동회에 참가했다.'),
             ('지수', '부산으로 여행을 갔다.'), ('철수', '정원에서 꽃을 심었다.')]
    if args.question == 'uncertainty':
        facts[0] = ('영수', '1967년 또는 1968년경 중학생 때 고장난 라디오를 수리했다.')
    if args.question == 'identity_uncertain':
        facts[0] = ('영수 오빠', '1967년 또는 1968년경 중학생 때 고장난 라디오를 수리했다.')
    hits = []
    for index, (person, event) in enumerate(facts):
        tid, mid, sid = f'tr_demo_{index}', f'mem_demo_{index}', f'seg_demo_{index}'
        text = f'{person}는 {event}'
        repo.create_transcript(LoadedTranscript(transcript_id=tid, filename=f'demo{index}.txt', language='ko',
            source_type='stt_text', uploaded_at=datetime.now(UTC), content_hash=str(index)*64,
            raw_content=text, normalized_content=text))
        repo.create_segments([TranscriptSegmentCreate(segment_id=sid, transcript_id=tid, chunk_index=0,
            content=text, start_offset=0, end_offset=len(text))])
        memory = repo.create_memory(MemoryCreate(memory_id=mid, transcript_id=tid, title=f'{person}의 기억',
            summary=text, people=[person], date_precision=DatePrecision.UNKNOWN, confidence=0.95))
        repo.create_memory_source(MemorySourceCreate(memory_source_id=f'src_demo_{index}', memory_id=mid,
            transcript_id=tid, segment_id=sid, start_offset=0, end_offset=len(text)))
        hits.append(RetrievalHit(memory_id=mid, memory=memory, score=1/(index+1), bm25_rank=index+1, bm25_score=1.0))

    class Retriever:
        def search(self, query, *, top_k=5):
            return hits[:top_k]

    options = {}
    # Supports recording the unmodified baseline before optimizations exist.
    if args.combined: options['combined'] = True
    if args.compact: options['compact'] = True
    if args.cache: options['cache_enabled'] = True
    if args.max_rewrites != 1: options['max_rewrites'] = args.max_rewrites
    model_options = {'combined': True} if args.combined else {}
    if args.max_tokens: model_options['max_tokens'] = args.max_tokens
    if args.verification_model: model_options['verification_model'] = args.verification_model
    if args.native:
        model_options.update(native_ollama=True, keep_alive=args.keep_alive, think=args.thinking)
    model_name = args.model or settings.chat_model
    models = build_openai_qa_models(model_name, api_key=settings.chat_api_key,
                                   base_url=settings.chat_base_url, **model_options)
    calls = []
    models = QAModels(**{name: TimedModel(getattr(models, name), name, calls)
                        for name in ('evidence', 'answer', 'verification', 'rewrite')})
    service = GroundedQAService(repo, Retriever(), models, provider='ollama', model_name=model_name, **options)
    question = {'answerable': '영수가 누구야?', 'insufficient': '영수의 생일은 언제야?',
                'uncertainty': '영수는 언제 라디오를 수리했어?', 'location': '지수는 어디로 여행을 갔어?',
                'identity_uncertain': '영수가 누구야?'}[args.question]
    expected_id = 'mem_demo_2' if args.question == 'location' else 'mem_demo_0'
    for iteration in range(args.repeat):
        calls.clear()
        start = perf_counter()
        result = service.answer_question(session_id='benchmark', question=question)
        record = {'recorded_at': datetime.now(UTC).isoformat(), 'label': args.label, 'model': model_name,
                  'iteration': iteration+1, 'question_case': args.question, 'options': vars(args),
                  'seconds': round(perf_counter()-start, 3), 'passed': result.validation_result.passed,
                  'failure_code': result.validation_result.failure_code,
                  'citation_ids': sorted({item.memory_id for item in result.citations}),
                  'expected_citation': any(item.memory_id == expected_id for item in result.citations),
                  'retry_count': result.retry_count, 'model_calls': list(calls)}
        record['cache_hit'] = result.cache_hit
        if args.question in ('uncertainty', 'identity_uncertain'):
            record['uncertainty_preserved'] = '1967' in result.final_answer and '1968' in result.final_answer
        with (root / 'results.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(json.dumps(record, ensure_ascii=True), flush=True)
    service.close()
    database.close()


if __name__ == '__main__':
    main()
