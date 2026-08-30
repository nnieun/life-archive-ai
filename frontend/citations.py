"""사용자 화면에 연결된 기억을 표시한다."""

from __future__ import annotations

import re

import streamlit as st

from frontend.api_client import Citation


_INTERNAL_CITATION_PATTERN = re.compile(
    r"\s*\[mem_[^\]|\s]+\|tr_[^:\]\s]+:\d+-\d+\]"
)


def remove_internal_citation_markers(answer: str) -> str:
    """답변에 포함된 내부 출처 식별자를 사용자 화면에서 숨긴다."""

    return _INTERNAL_CITATION_PATTERN.sub("", answer).strip()


def render_citations(
    citations: list[Citation],
    memory_labels: dict[str, str] | None = None,
) -> None:
    """출처를 클릭 가능한 연결된 기억으로 표시한다."""

    if not citations:
        st.caption("표시할 연결된 기억이 없습니다.")
        return

    st.markdown("**연결된 기억**")
    for index, citation in enumerate(citations):
        label = (memory_labels or {}).get(citation.memory_id, "연결된 기억")
        if st.button(
            f"연결된 기억: {label}",
            key=f"open-memory-{citation.memory_id}-{index}",
        ):
            st.session_state["selected_memory_id"] = citation.memory_id
            st.switch_page("pages/memories.py")
