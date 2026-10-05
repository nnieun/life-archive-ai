"""Background upload completion, navigation, privacy, and recovery tests."""

from base64 import b64encode
from pathlib import Path
from threading import Event, Thread
from unittest.mock import Mock
import sqlite3
import pytest
from fastapi.testclient import TestClient
from backend.app.api import memories
from backend.app.main import app
from backend.app.models.ingestion import IngestionResult
from backend.app.services.ingestion_jobs import IngestionJobStore, IngestionJobCancelled
from frontend.api_client import IngestionJob
from frontend.upload_background import collect_upload_result


def result():
    return IngestionResult(transcript_id='synthetic', filename='demo.txt', segment_count=1, memory_count=1,
                           indexed_memory_count=1, memory_ids=['m'])


def test_api_background_completes_and_scopes_lookup(sqlite_repository, monkeypatch):
    store = IngestionJobStore(sqlite_repository._database)
    service = Mock()
    service.ingest.return_value = result()
    monkeypatch.setattr(memories, 'get_ingestion_service', lambda: service)
    app.dependency_overrides[memories.get_ingestion_job_store] = lambda: store
    try:
        client = TestClient(app)
        response = client.post('/api/v1/memories/ingest/jobs', json={'session_id': 's', 'filename': 'demo.txt',
                                'content_base64': b64encode(b'demo text').decode()})
        assert response.status_code == 202 and response.json()['status'] == 'queued'
        job_id = response.json()['job_id']
        assert client.get(f'/api/v1/memories/ingest/jobs/{job_id}', params={'session_id': 's'}).json()['status'] == 'completed'
        assert client.get(f'/api/v1/memories/ingest/jobs/{job_id}', params={'session_id': 'other'}).status_code == 404
        assert service.ingest.call_args.kwargs['content'] == b'demo text'
        assert client.post('/api/v1/memories/ingest/jobs', json={'session_id': 's', 'filename': '../private.txt',
                          'content_base64': b64encode(b'demo').decode()}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_duplicate_pending_jobs_recovery_and_safe_error(sqlite_repository, monkeypatch):
    store = IngestionJobStore(sqlite_repository._database)
    job = store.create('s', 'demo.txt', 'hash')
    for args in [('s', 'other.txt', 'other'), ('other', 'demo.txt', 'other'), ('other', 'other.txt', 'hash')]:
        with pytest.raises(sqlite3.IntegrityError):
            store.create(*args)
    monkeypatch.setattr(memories, 'get_ingestion_service', Mock(side_effect=RuntimeError('private key and path')))
    request = memories.SubmitIngestionRequest(session_id='s', filename='demo.txt', content_base64='x')
    memories.run_ingestion_job(job.job_id, request, b'demo', store)
    failed = store.get(job.job_id, 's')
    assert failed.status == 'failed' and 'private' not in failed.error
    next_job = store.create('s', 'demo.txt', 'hash')
    store.start(next_job.job_id)
    store.recover_interrupted()
    assert store.get(next_job.job_id, 's').failure_code == 'server_restarted'
    store.finish(next_job.job_id, result())
    assert store.get(next_job.job_id, 's').result is None


def test_worker_is_independent_of_frontend_navigation(sqlite_repository, monkeypatch):
    store = IngestionJobStore(sqlite_repository._database)
    job = store.create('s', 'demo.txt', 'hash')
    started, release = Event(), Event()
    def ingest(**kwargs):
        kwargs['progress']({'stage': 'extracting', 'total_segments': 1, 'completed_segments': 0})
        started.set()
        assert release.wait(5)
        return result()
    monkeypatch.setattr(memories, 'get_ingestion_service', lambda: Mock(ingest=ingest))
    request = memories.SubmitIngestionRequest(session_id='s', filename='demo.txt', content_base64='x')
    worker = Thread(target=memories.run_ingestion_job, args=(job.job_id, request, b'demo', store))
    worker.start()
    try:
        assert started.wait(5)
        assert store.get(job.job_id, 's').progress['stage'] == 'extracting'
        # No frontend page participates in or retains the worker.
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and store.get(job.job_id, 's').status == 'completed'


def test_completion_collected_once_on_other_page(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from frontend import upload_background
    client = Mock()
    client.get_ingestion_job.return_value = IngestionJob(job_id='job', session_id='s', filename='demo.txt',
        created_at='2026-10-05T00:00:00Z', status='completed', result=result().model_dump())
    state = {'pending_upload_job': {'job_id': 'job', 'session_id': 's'}}
    assert collect_upload_result(state, client)
    assert state['last_upload_job']['result']['memory_count'] == 1
    assert not collect_upload_result(state, client)
    monkeypatch.setattr(upload_background, 'get_api_client', lambda: client)
    frontend_root = Path(__file__).resolve().parents[1] / 'frontend'
    # AppTest resets legacy pages discovery between runs. A temporary entry
    # point with absolute page paths exercises st.navigation without that reset.
    source = (frontend_root / 'app.py').read_text(encoding='utf-8').replace('"pages/', f'"{frontend_root.as_posix()}/pages/')
    application = AppTest.from_string(source)
    application.run()
    application.session_state['pending_upload_job'] = {'job_id': 'job', 'session_id': 's'}
    application.switch_page(str(frontend_root / 'pages/chat.py')).run()
    assert not application.exception
    assert application.session_state['upload_completion_notice']['success']


def test_pending_upload_progress_is_shown_on_upload_page(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from frontend import upload_background
    client = Mock()
    client.get_ingestion_job.return_value = IngestionJob(job_id='job', session_id='s', filename='demo.txt',
        created_at='2026-10-05T00:00:00Z', status='running', progress={'stage': 'indexing'})
    monkeypatch.setattr(upload_background, 'get_api_client', lambda: client)
    application = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'frontend/app.py'))
    application.session_state['pending_upload_job'] = {'job_id': 'job', 'session_id': 's', 'created_at': '2026-10-05T00:00:00+00:00'}
    application.run()
    assert not application.exception
    assert len([item for item in application.caption if '검색 인덱스를 만드는 중' in item.value]) == 1
    client.get_ingestion_job.assert_called_once()


def test_indexing_timeout_is_classified_as_embedding_failure(sqlite_repository, monkeypatch):
    import httpx
    store = IngestionJobStore(sqlite_repository._database)
    job = store.create('s', 'demo.txt', 'hash')
    def ingest(**kwargs):
        kwargs['progress']({'stage': 'indexing'})
        raise httpx.ReadTimeout('private detail')
    monkeypatch.setattr(memories, 'get_ingestion_service', lambda: Mock(ingest=ingest))
    memories.run_ingestion_job(job.job_id,
        memories.SubmitIngestionRequest(session_id='s', filename='demo.txt', content_base64='x'), b'demo', store)
    failed = store.get(job.job_id, 's')
    assert failed.failure_code == 'embedding_timeout' and 'private' not in failed.error


def test_failure_diagnostic_keeps_types_but_not_library_messages():
    from backend.app.api.memories import _failure_diagnostic
    try:
        try:
            raise ValueError("secret diary text")
        except ValueError as cause:
            raise RuntimeError("wrapped") from cause
    except RuntimeError as error:
        result = _failure_diagnostic(error, {'stage': 'extracting', 'completed_segments': 2, 'total_segments': 3})
    assert result['stage'] == 'extracting' and result['completed_segments'] == 2
    assert [item['type'] for item in result['causes']] == ['RuntimeError', 'ValueError']
    assert all(item['message'] is None for item in result['causes'])
