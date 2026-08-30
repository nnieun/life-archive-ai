"""Streamlit MVP page tests using mocked FastAPI client responses."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from frontend.api_client import (
    ApiClientError,
    AutobiographyResult,
    ChatResult,
    IngestionResult,
    MemoryView,
    TimelineResult,
)
from frontend import ui

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _page_path(page_name: str) -> str:
    """Resolve Streamlit pages from the project root, not the tests directory."""
    return str(PROJECT_ROOT / "frontend" / "pages" / page_name)


def _citation() -> dict[str, object]:
    return {
        "memory_id": "mem_001",
        "transcript_id": "tr_001",
        "segment_id": "seg_001",
        "start_offset": 0,
        "end_offset": 12,
    }


@pytest.fixture
def api_client(monkeypatch) -> Mock:
    client = Mock()
    client.list_memories.return_value = []
    monkeypatch.setattr(ui, "get_api_client", lambda: client)
    return client


def _has_memory_citation_button(app: AppTest, memory_id: str) -> bool:
    return any(
        button.key == f"open-memory-{memory_id}-0"
        for button in app.button
    )


def test_backend_failure_shows_user_safe_message(api_client: Mock) -> None:
    api_client.list_memories.side_effect = ApiClientError("private detail")

    app = AppTest.from_file(_page_path("memories.py")).run()

    assert len(app.error) == 1
    assert "백엔드가 실행 중인지" in app.error[0].value
    assert "private detail" not in app.error[0].value


def test_service_failure_shows_request_id_without_private_detail(
    api_client: Mock,
) -> None:
    api_client.list_memories.side_effect = ApiClientError(
        r"sk-private C:\Users\person\memory.txt",
        status_code=503,
        request_id="req-ui-123",
    )

    app = AppTest.from_file(_page_path("memories.py")).run()

    assert len(app.error) == 1
    assert "서비스를 현재 사용할 수 없습니다" in app.error[0].value
    assert any("req-ui-123" in caption.value for caption in app.caption)
    assert "sk-private" not in str(app)
    assert r"C:\Users\person" not in str(app)


def test_txt_upload_displays_processing_and_index_result(
    api_client: Mock,
) -> None:
    api_client.ingest_transcript.return_value = IngestionResult(
        transcript_id="tr_upload",
        filename="memory.txt",
        segment_count=2,
        memory_count=1,
        indexed_memory_count=1,
        memory_ids=["mem_001"],
    )
    app = AppTest.from_file(_page_path("upload.py")).run()

    app.file_uploader[0].upload(
        "memory.txt",
        b"\xec\xb6\x94\xec\x96\xb5",
        "text/plain",
    ).run()
    app.button[0].click().run()

    assert api_client.ingest_transcript.call_args.args[0] == "memory.txt"
    assert len(app.success) == 1
    assert "1개의 기억" in app.success[0].value


def test_duplicate_txt_upload_shows_conflict_message(
    api_client: Mock,
) -> None:
    api_client.ingest_transcript.side_effect = ApiClientError(
        "conflict",
        status_code=409,
    )
    app = AppTest.from_file(_page_path("upload.py")).run()

    app.file_uploader[0].upload(
        "memory.txt",
        b"duplicate",
        "text/plain",
    ).run()
    app.button[0].click().run()

    assert len(app.error) == 1
    assert "이미 등록된 TXT" in app.error[0].value
    assert "백엔드가 실행 중인지" not in app.error[0].value


def test_chat_displays_answer_and_citation(api_client: Mock) -> None:
    api_client.list_memories.return_value = [
        MemoryView.model_validate(
            {
                "memory": {
                    "memory_id": "mem_001",
                    "transcript_id": "tr_001",
                    "title": "첫 만남",
                    "summary": "공원에서 친구를 만났다.",
                    "people": ["친구"],
                    "location": "공원",
                    "event_date": None,
                    "date_precision": "unknown",
                    "emotion": None,
                    "confidence": 0.9,
                    "uncertainty_notes": None,
                    "status": "active",
                },
                "citations": [_citation()],
                "source_filename": "memory.txt",
            }
        )
    ]
    api_client.chat.return_value = ChatResult.model_validate(
        {
            "session_id": "session_ui",
            "question": "어디에서 만났어?",
            "retrieved_memory_ids": ["mem_001"],
            "final_answer": "공원에서 만났습니다.",
            "citations": [_citation()],
            "validation_result": {
                "stage": "answer",
                "passed": True,
                "reason": "근거 확인",
            },
            "retry_count": 0,
        }
    )
    app = AppTest.from_file(_page_path("chat.py")).run()

    app.chat_input[0].set_value("어디에서 만났어?").run()

    assert "공원에서 만났습니다." in str(app)
    assert any("첫 만남" in button.label for button in app.button)
    assert _has_memory_citation_button(app, "mem_001")


def test_timeline_displays_precision_and_citation(api_client: Mock) -> None:
    event = {
        "memory_id": "mem_001",
        "title": "첫 만남",
        "description": "친구와 공원에서 만났다.",
        "event_date": "2020",
        "date_precision": "year",
        "date_label": "2020년",
        "confidence": 0.9,
        "citations": [_citation()],
    }
    api_client.get_timeline.return_value = TimelineResult.model_validate(
        {"events": [event], "undated_events": []}
    )
    app = AppTest.from_file(_page_path("timeline.py")).run()

    app.button[0].click().run()

    assert any(
        "날짜 정밀도: 연도만 확인" in caption.value
        for caption in app.caption
    )
    assert _has_memory_citation_button(app, "mem_001")


def test_autobiography_displays_each_chapter_citation(
    api_client: Mock,
) -> None:
    api_client.generate_autobiography.return_value = (
        AutobiographyResult.model_validate(
            {
                "autobiography": {
                    "autobiography_id": "auto_001",
                    "title": "나의 기억",
                    "status": "completed",
                    "content": {
                        "chapters": [
                            {
                                "title": "첫 장",
                                "content": "공원에서 친구를 만났다.",
                                "citations": [_citation()],
                            }
                        ]
                    },
                },
                "completed": True,
                "retrieved_memory_ids": ["mem_001"],
                "citations": [_citation()],
                "retry_count": 0,
            }
        )
    )
    app = AppTest.from_file(_page_path("autobiography.py")).run()
    app.text_area[0].set_value("친구에 대한 기억을 써 주세요.")
    app.button[0].click().run()

    assert any("1장. 첫 장" in header.value for header in app.header)
    assert _has_memory_citation_button(app, "mem_001")
