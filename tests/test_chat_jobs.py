"""Background completion, isolation, recovery, and polling tests."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.api import chat
from backend.app.main import app
from backend.app.models.qa import QAResult, QAValidationResult
from backend.app.services.chat_jobs import ChatJobStore
from frontend.api_client import ChatJob as ClientJob
from frontend.chat_background import collect_chat_result


def result():
    return QAResult(session_id="s", question="synthetic", retrieved_memory_ids=[], final_answer="answer",
                    citations=[], validation_result=QAValidationResult(stage="answer", passed=True, reason="ok"), retry_count=0)


def test_submission_completes_in_background_and_is_session_scoped(sqlite_repository, monkeypatch):
    store = ChatJobStore(sqlite_repository._database)
    service = Mock()
    service.answer_question.return_value = result()
    monkeypatch.setattr(chat, "get_qa_service", lambda: service)
    app.dependency_overrides[chat.get_chat_job_store] = lambda: store
    try:
        client = TestClient(app)
        submitted = client.post("/api/v1/chat/jobs", json={"session_id": "s", "question": "synthetic"})
        assert submitted.status_code == 202
        assert submitted.json()["status"] == "queued"
        job_id = submitted.json()["job_id"]
        fetched = client.get(f"/api/v1/chat/jobs/{job_id}", params={"session_id": "s"})
        assert fetched.json()["status"] == "completed"
        assert fetched.json()["result"]["final_answer"] == "answer"
        assert service.answer_question.call_args.kwargs['top_k'] == 3
        assert client.get(f"/api/v1/chat/jobs/{job_id}", params={"session_id": "other"}).status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_duplicate_pending_job_recovery_and_cancellation(sqlite_repository):
    import sqlite3
    store = ChatJobStore(sqlite_repository._database)
    job = store.create("s")
    with pytest.raises(sqlite3.IntegrityError):
        store.create("s")
    assert store.start(job.job_id)
    store.recover_interrupted()
    assert store.get(job.job_id, "s").status == "failed"
    store.finish(job.job_id, result())
    assert store.get(job.job_id, "s").result is None
    assert store.create("s").status == "queued"


def test_progress_is_session_scoped_and_cannot_return_after_cancel(sqlite_repository):
    store = ChatJobStore(sqlite_repository._database)
    job = store.create('s')
    store.start(job.job_id)
    store.progress(job.job_id, {'stage': 'verify_answer', 'memories': [{'memory_id': 'm', 'title': 'synthetic'}]})
    assert store.get(job.job_id, 's').progress['stage'] == 'verify_answer'
    assert store.get(job.job_id, 'other') is None
    # Job creation precedes session creation in the background worker.
    from backend.app.storage.models import ConversationSessionCreate
    sqlite_repository.create_conversation_session(ConversationSessionCreate(session_id='s', title='synthetic'))
    sqlite_repository.soft_delete_conversation_session('s')
    store.progress(job.job_id, {'stage': 'late', 'memories': []})
    assert store.get(job.job_id, 's').progress == {}


def test_worker_failure_uses_safe_message(sqlite_repository, monkeypatch):
    store = ChatJobStore(sqlite_repository._database)
    job = store.create("s")
    monkeypatch.setattr(chat, "get_qa_service", Mock(side_effect=RuntimeError("secret key")))
    chat.run_chat_job(job.job_id, chat.ChatRequest(session_id="s", question="synthetic"), store)
    failure = store.get(job.job_id, "s")
    assert failure.status == "failed"
    assert "secret" not in failure.error
    assert failure.failure_stage == "initialization"
    assert failure.failure_code == "internal_error"


def test_result_collected_after_navigation_once():
    client = Mock()
    client.list_memories.return_value = []
    job = ClientJob(job_id="job", session_id="s", status="completed", created_at="2026-10-05T00:00:00Z",
                    result=result().model_dump())
    client.get_chat_job.return_value = job
    state = {"pending_chat_job": {"job_id": "job", "session_id": "s"},
             "chat_messages": [{"role": "user", "content": "synthetic"}]}
    assert collect_chat_result(state, client)
    assert state["chat_completion_notice"]["text"] == "답변이 완료되었습니다."
    assert state["chat_messages"][-1]["content"] == "answer"
    assert not collect_chat_result(state, client)
    assert len(state["chat_messages"]) == 2


def test_shared_app_collects_completion_on_upload_page(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from frontend import chat_background
    client = Mock()
    client.list_memories.return_value = []
    client.get_chat_job.return_value = ClientJob(
        job_id="job", session_id="s", status="completed", created_at="2026-10-05T00:00:00Z",
        result=result().model_dump(),
    )
    monkeypatch.setattr(chat_background, "get_api_client", lambda: client)
    application = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"))
    application.session_state["pending_chat_job"] = {"job_id": "job", "session_id": "s"}
    application.session_state["chat_messages"] = [{"role": "user", "content": "synthetic"}]
    application.run()
    assert not application.exception
    assert application.session_state["chat_messages"][-1]["content"] == "answer"
    assert application.session_state["chat_completion_notice"]["text"] == "답변이 완료되었습니다."


def test_shared_app_hides_pending_progress(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from frontend import chat_background
    client = Mock()
    client.get_chat_job.return_value = ClientJob(
        job_id="job", session_id="s", status="running", created_at="2026-10-05T00:00:00Z",
    )
    monkeypatch.setattr(chat_background, "get_api_client", lambda: client)
    application = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"))
    application.session_state["pending_chat_job"] = {
        "job_id": "job", "session_id": "s", "created_at": "2026-10-05T00:00:00+00:00",
    }
    application.run()
    assert not application.exception
    assert len([caption for caption in application.caption if "관련 기억을 찾는 중" in caption.value]) == 0
    client.get_chat_job.assert_called_once()
