from types import SimpleNamespace

from backend.app.services.memory_text import memory_search_text
from backend.app.models.qa import QAEvidence


def test_structured_fields_are_in_search_text():
    memory = SimpleNamespace(title="저녁", summary="TV를 보았다.", people=["도현"],
                             location="집", event_date="2026-04-12", emotion="그리움")
    text = memory_search_text(memory)
    for value in ("저녁", "TV를 보았다.", "도현", "집", "2026-04-12", "그리움"):
        assert value in text
    assert "emotion" in QAEvidence.model_fields
