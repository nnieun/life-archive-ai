"""Shared presentation helpers for Streamlit pages."""

from __future__ import annotations

from os import getenv

import streamlit as st

from frontend.api_client import ApiClientError, Citation, LifeArchiveApiClient


def get_api_client() -> LifeArchiveApiClient:
    return LifeArchiveApiClient(
        base_url=getenv(
            "LIFE_ARCHIVE_API_URL",
            "http://127.0.0.1:8000/api/v1",
        )
    )


def show_backend_error(action: str, exception: ApiClientError) -> None:
    """Show status-aware guidance without displaying backend details."""

    if exception.status_code is None:
        message = (
            f"{action}에 실패했습니다. 백엔드가 실행 중인지 확인한 뒤 "
            "다시 시도해 주세요."
        )
    elif exception.status_code == 422:
        message = f"{action} 요청 내용을 확인해 주세요."
    elif exception.status_code == 409:
        message = f"{action} 요청이 기존 데이터와 충돌했습니다."
    elif exception.status_code == 503:
        message = (
            f"{action}에 필요한 서비스를 현재 사용할 수 없습니다. "
            "잠시 후 다시 시도해 주세요."
        )
    else:
        message = f"{action} 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
    st.error(message)
    if exception.request_id:
        st.caption(f"문제가 계속되면 요청 ID를 알려 주세요: {exception.request_id}")


def render_citations(citations: list[Citation]) -> None:
    if not citations:
        st.caption("표시할 출처가 없습니다.")
        return
    st.markdown("**출처**")
    for index, citation in enumerate(citations, start=1):
        st.markdown(
            f"- **출처 {index}**: 업로드한 원문의 "
            f"{citation.start_offset + 1}~{citation.end_offset}번째 글자"
        )
