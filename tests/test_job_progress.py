"""Progress belongs to its originating page, even with concurrent jobs."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize('page,expected,hidden', [
    ('chat.py', '답변 근거를 검증하는 중', '검색 인덱스를 만드는 중'),
    ('upload.py', '검색 인덱스를 만드는 중', '답변 근거를 검증하는 중'),
])
def test_each_page_only_renders_its_own_job(page, expected, hidden):
    root = Path(__file__).resolve().parents[1] / 'frontend/pages'
    application = AppTest.from_file(str(root / page))
    application.session_state['pending_chat_job'] = {
        'created_at': '2026-10-05T00:00:00+00:00', 'progress': {'stage': 'verify_answer'},
    }
    application.session_state['pending_upload_job'] = {
        'created_at': '2026-10-05T00:00:00+00:00', 'progress': {'stage': 'indexing'},
    }
    application.run()
    assert not application.exception
    assert sum(expected in item.value for item in application.caption) == 1
    assert not any(hidden in item.value for item in application.caption)
