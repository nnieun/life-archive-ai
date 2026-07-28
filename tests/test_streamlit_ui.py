"""Streamlit MVP page tests using mocked FastAPI client responses."""

from __future__ import annotations

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
    monkeypatch.setattr(ui, "get_api_client", lambda: client)
    return client


def test_backend_failure_shows_user_safe_message(api_client: Mock) -> None:
    api_client.list_memories.side_effect = ApiClientError("private detail")

    app = AppTest.from_file("frontend/pages/memories.py").run()

    assert len(app.error) == 1
    assert "백엔드가 실행 중인지" in app.error[0].value
    assert "private detail" not in app.error[0].value


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
    app = AppTest.from_file("frontend/pages/upload.py").run()

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
    app = AppTest.from_file("frontend/pages/upload.py").run()

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
    app = AppTest.from_file("frontend/pages/chat.py").run()

    app.chat_input[0].set_value("어디에서 만났어?").run()

    assert "공원에서 만났습니다." in str(app)
    assert "mem_001" in str(app)


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
    app = AppTest.from_file("frontend/pages/timeline.py").run()

    app.button[0].click().run()

    assert any(
        "날짜 정밀도: year" in caption.value
        for caption in app.caption
    )
    assert any("mem_001" in code.value for code in app.code)


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
    app = AppTest.from_file("frontend/pages/autobiography.py").run()
    app.text_area[0].set_value("친구에 대한 기억을 써 주세요.")
    app.button[0].click().run()

    assert any("1장. 첫 장" in header.value for header in app.header)
    assert any("mem_001" in code.value for code in app.code)
