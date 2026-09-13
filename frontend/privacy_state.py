"""삭제 후 현재 브라우저 세션에 남은 파생 정보를 비운다."""

from collections.abc import MutableMapping
from typing import Any


def clear_derived_state(state: MutableMapping[str, Any]) -> None:
    """이전 답변을 다시 표시하지 않도록 대화와 작성 대기를 초기화한다."""
    for key in (
        "chat_messages", "chat_session_id", "selected_memory_id",
        "pending_autobiography", "pending_autobiography_gap_count",
        "autobiography-unresolved-confirm",
    ):
        state.pop(key, None)
