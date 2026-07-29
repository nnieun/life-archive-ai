"""TASK-018 privacy-safe error handling and logging tests."""

from __future__ import annotations

import json
import logging
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.chat import get_qa_service
from backend.app.api.memories import get_memory_repository
from backend.app.core.safe_logging import SafeJsonFormatter
from backend.app.main import create_app
from backend.app.services.qa import QAError
from backend.app.storage.repository import StorageError


def _assert_safe_error(response, status_code: int) -> dict[str, str]:
    assert response.status_code == status_code
    detail = response.json()["error"]
    assert detail["request_id"] == response.headers["X-Request-ID"]
    serialized = response.text
    assert "sk-private-secret" not in serialized
    assert r"C:\Users\person\archive.sqlite3" not in serialized
    return detail


def test_request_id_is_returned_and_untrusted_id_is_replaced() -> None:
    client = TestClient(create_app())

    accepted = client.get(
        "/api/v1/missing",
        headers={"X-Request-ID": "req-safe-123"},
    )
    rejected = client.get(
        "/api/v1/missing",
        headers={"X-Request-ID": "unsafe request id"},
    )

    assert accepted.headers["X-Request-ID"] == "req-safe-123"
    assert accepted.json()["error"]["request_id"] == "req-safe-123"
    assert rejected.headers["X-Request-ID"] != "unsafe request id"
    assert len(rejected.headers["X-Request-ID"]) == 32


def test_sqlite_failure_returns_safe_common_error() -> None:
    application = create_app()
    repository = Mock()
    repository.list_memories.side_effect = StorageError(
        r"failed at C:\Users\person\archive.sqlite3 sk-private-secret"
    )
    application.dependency_overrides[get_memory_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get("/api/v1/memories")

    detail = _assert_safe_error(response, 503)
    assert detail["code"] == "storage_error"


@pytest.mark.parametrize(
    "private_failure",
    [
        "OpenAI 401 for sk-private-secret",
        r"Chroma index missing at C:\Users\person\archive.sqlite3",
    ],
)
def test_external_service_failure_does_not_expose_details(
    private_failure: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    application = create_app()
    service = Mock()
    service.answer_question.side_effect = QAError(private_failure)
    application.dependency_overrides[get_qa_service] = lambda: service
    caplog.set_level(logging.INFO, logger="life_archive")

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "session_id": "session-safe",
                "question": "기억을 찾아줘",
            },
        )

    detail = _assert_safe_error(response, 503)
    assert detail["message"] == "Grounded question answering is unavailable"
    assert private_failure not in response.text
    failure_record = next(
        record
        for record in caplog.records
        if record.getMessage() == "service_request_failed"
    )
    assert failure_record.error_type == "QAError"
    assert private_failure not in SafeJsonFormatter().format(failure_record)


def test_invalid_upload_and_model_validation_are_safe() -> None:
    client = TestClient(create_app())

    invalid_file = client.post(
        "/api/v1/memories/ingest",
        json={
            "filename": r"C:\Users\person\archive.txt",
            "content_base64": "not-base64!",
        },
    )
    invalid_model = client.post(
        "/api/v1/chat",
        json={
            "session_id": " ",
            "question": "sk-private-secret",
            "top_k": 999,
        },
    )

    assert _assert_safe_error(invalid_file, 422)["code"] == "http_error"
    assert _assert_safe_error(invalid_model, 422)["code"] == "validation_error"


def test_unexpected_failure_hides_exception_message_and_path() -> None:
    application = create_app()

    @application.get("/test/unexpected")
    def unexpected() -> None:
        raise RuntimeError(
            r"sk-private-secret C:\Users\person\archive.sqlite3"
        )

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/test/unexpected")

    detail = _assert_safe_error(response, 500)
    assert detail["code"] == "internal_error"


def test_safe_json_formatter_masks_api_credentials() -> None:
    record = logging.LogRecord(
        name="life_archive",
        level=logging.ERROR,
        pathname=r"C:\private\module.py",
        lineno=1,
        msg="provider failed sk-private-secret Bearer abc.def-123",
        args=(),
        exc_info=None,
    )

    payload = json.loads(SafeJsonFormatter().format(record))

    assert payload["event"] == "provider failed sk-*** Bearer ***"
    assert "pathname" not in payload
    assert "private-secret" not in json.dumps(payload)


def test_request_log_uses_route_template_without_query_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    application = create_app()
    repository = Mock()
    repository.list_memories.return_value = []
    application.dependency_overrides[get_memory_repository] = lambda: repository
    caplog.set_level(logging.INFO, logger="life_archive")

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/memories",
            params={"transcript_id": "private-transcript-value"},
        )

    assert response.status_code == 200
    completion = next(
        record
        for record in caplog.records
        if record.getMessage() == "request_completed"
    )
    payload = SafeJsonFormatter().format(completion)
    assert json.loads(payload)["route"] == "/memories"
    assert "private-transcript-value" not in payload
