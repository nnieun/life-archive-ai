"""Shared presentation helpers for Streamlit pages."""

from __future__ import annotations

from os import getenv

import streamlit as st

from frontend.api_client import Citation, LifeArchiveApiClient


def get_api_client() -> LifeArchiveApiClient:
    return LifeArchiveApiClient(
        base_url=getenv(
            "LIFE_ARCHIVE_API_URL",
            "http://127.0.0.1:8000/api/v1",
        )
    )


def show_backend_error(action: str) -> None:
    st.error(
        f"{action}에 실패했습니다. 백엔드가 실행 중인지 확인한 뒤 다시 시도해 주세요."
    )


def render_citations(citations: list[Citation]) -> None:
    if not citations:
        st.caption("표시할 출처가 없습니다.")
        return
    st.markdown("**출처**")
    for citation in citations:
        segment = citation.segment_id or "전체 원문"
        st.code(
            f"{citation.memory_id} · {citation.transcript_id} · "
            f"{segment} · {citation.start_offset}-{citation.end_offset}"
        )
