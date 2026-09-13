from types import SimpleNamespace
from unittest.mock import Mock

from frontend import citations
from frontend.privacy_state import clear_derived_state


def test_citation_buttons_are_scoped_and_deduplicated(monkeypatch):
    ui = Mock()
    ui.button.return_value = False
    monkeypatch.setattr(citations, "st", ui)
    source = SimpleNamespace(memory_id="memory-one")
    citations.render_citations([source, source], scope="chat-1")
    citations.render_citations([source], scope="chat-3")
    keys = [call.kwargs["key"] for call in ui.button.call_args_list]
    assert len(keys) == len(set(keys)) == 2


def test_delete_clears_derived_state_without_removing_unrelated_preferences():
    state = {"chat_messages": ["old answer"], "chat_session_id": "old",
             "pending_autobiography": {}, "selected_memory_id": "old",
             "theme": "dark"}
    clear_derived_state(state)
    clear_derived_state(state)
    assert state == {"theme": "dark"}
