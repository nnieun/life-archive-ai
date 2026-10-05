"""Replay latest completed QA locally in an in-memory DB; save timings only."""

import ctypes
import json
import os
from pathlib import Path
import sqlite3
import sys
from time import perf_counter
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.core.config import get_settings
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
from backend.app.models.retrieval import RetrievalHit
from backend.app.services.qa import GroundedQAService, QAModels, build_openai_qa_models
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.repository import SQLiteRepository
import httpx


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in
        ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]


def memory_status():
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {'load_percent': status.load, 'available_gb': round(status.available / 1024**3, 2)}
    return {}


class Timed:
    def __init__(self, model, label, calls):
        self.model, self.label, self.calls = model, label, calls

    def invoke(self, messages):
        started = perf_counter()
        result = self.model.invoke(messages)
        metadata = result['raw'].response_metadata
        item = {'stage': self.label, 'wall_seconds': round(perf_counter()-started, 3),
                'input_chars': sum(len(message.content) for message in messages)}
        for key in ('load_duration', 'prompt_eval_duration', 'eval_duration'):
            item[key.replace('_duration', '_seconds')] = round(metadata[key] / 1e9, 3)
        for key in ('prompt_eval_count', 'eval_count'):
            item[key] = metadata[key]
        item['tokens_per_second'] = round(metadata['eval_count'] / max(metadata['eval_duration']/1e9, .001), 2)
        self.calls.append(item)
        return result


def main():
    settings = get_settings()
    if settings.llm_provider != 'ollama':
        raise SystemExit('Local Ollama is required.')
    source = sqlite3.connect(f'file:{settings.sqlite_database_path.as_posix()}?mode=ro', uri=True)
    row = source.execute("SELECT result_json FROM chat_jobs WHERE status='completed' AND result_json IS NOT NULL ORDER BY rowid DESC LIMIT 1").fetchone()
    original = json.loads(row[0])
    database = SQLiteDatabase(':memory:')
    source.backup(database.connection)
    source.close()
    repository = SQLiteRepository(database)
    if '--retrieval-only' in sys.argv:
        from backend.app.services.vector_index import create_openai_embeddings, DEFAULT_COLLECTION_NAME
        from backend.app.services.retrieval import BM25MemoryIndex
        import chromadb
        from chromadb.config import Settings
        bm25 = BM25MemoryIndex(repository)
        bm25.rebuild_from_sqlite()
        client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory),
                                          settings=Settings(anonymized_telemetry=False))
        collection = client.get_collection(DEFAULT_COLLECTION_NAME)
        embeddings = create_openai_embeddings(settings.openai_embedding_model, api_key=settings.openai_api_key)
        measurements = []
        for _ in range(2):
            started = perf_counter()
            vector = embeddings.embed_query(original['question'])
            embed_seconds = perf_counter() - started
            started = perf_counter()
            collection.query(query_embeddings=[vector], n_results=min(10, collection.count()), include=['distances'])
            dense_seconds = perf_counter() - started
            started = perf_counter()
            bm25.search(original['question'], top_k=10)
            measurements.append({'embedding_seconds': round(embed_seconds, 3),
                'vector_search_seconds': round(dense_seconds, 3), 'bm25_seconds': round(perf_counter()-started, 3)})
        path = Path('data/processed/qa_performance/recent_diagnosis.json')
        report = json.loads(path.read_text(encoding='utf-8'))
        report['retrieval'] = measurements
        report['embedding_model'] = settings.openai_embedding_model
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(measurements))
        database.close()
        return
    hits = [RetrievalHit(memory_id=mid, memory=repository.get_memory(mid), score=1/(i+1),
                         bm25_rank=i+1, bm25_score=1.0) for i, mid in enumerate(original['retrieved_memory_ids'])]
    models = build_openai_qa_models(settings.chat_model, base_url=settings.chat_base_url,
        combined=settings.qa_combined, native_ollama=True, max_tokens=settings.qa_max_tokens,
        keep_alive=settings.ollama_keep_alive, verification_model=settings.qa_verification_model or None)
    report = {'memory_before': memory_status(), 'model': settings.chat_model,
              'max_tokens': settings.qa_max_tokens, 'keep_alive': settings.ollama_keep_alive,
              'retrieved_count': len(hits), 'runs': []}
    with httpx.Client(trust_env=False) as client:
        report['ollama_before'] = client.get(settings.chat_base_url.removesuffix('/v1') + '/api/ps').json()
    for run in range(2):
        calls = []
        wrapped = QAModels(**{name: Timed(getattr(models, name), name, calls)
                             for name in ('evidence', 'answer', 'verification', 'rewrite')})
        service = GroundedQAService(repository, SimpleNamespace(search=lambda question, **kwargs: hits), wrapped,
            provider='ollama', model_name=settings.chat_model, combined=settings.qa_combined,
            compact=settings.qa_compact, max_rewrites=settings.qa_max_rewrites)
        result = service.answer_question(session_id=original['session_id'], question=original['question'])
        report['runs'].append({'elapsed_seconds': round(result.elapsed_ms/1000, 3),
                              'passed': result.validation_result.passed, 'calls': calls})
        print(json.dumps(report['runs'][-1]), flush=True)
    report['memory_after'] = memory_status()
    path = Path('data/processed/qa_performance/recent_diagnosis.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'memory_before': report['memory_before'], 'memory_after': report['memory_after']}))
    for client in {getattr(model, 'client', None) for model in (models.evidence, models.answer, models.verification, models.rewrite)}:
        if client is not None:
            client.close()
    database.close()


if __name__ == '__main__':
    main()
