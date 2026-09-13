"""Grounded autobiography generation page with important-gap preflight."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.api_client import ApiClientError, AutobiographyResult
from frontend.citations import render_citations
from frontend.ui import get_api_client, show_backend_error


def _render_result(result: AutobiographyResult) -> None:
    if result.error and not result.requires_gap_confirmation:
        st.warning(result.error)
    for index, chapter in enumerate(
        result.autobiography.content.chapters,
        start=1,
    ):
        st.header(f"{index}장. {chapter.title}")
        st.write(chapter.content)
        render_citations(chapter.citations, scope=f"autobiography-{index}")


def _generate(payload: dict[str, Any], *, proceed: bool) -> AutobiographyResult | None:
    with st.spinner("관련 기억을 찾고 장별 초안을 생성하는 중입니다."):
        try:
            return get_api_client().generate_autobiography(
                **payload,
                proceed_with_unresolved_gaps=proceed,
            )
        except ApiClientError as exception:
            show_backend_error("자서전 생성", exception)
            return None


st.title("자서전 초안")
st.caption(
    "기간과 주제를 바탕으로 최대 3장의 근거 있는 초안을 생성합니다. "
    "중요한 기억 빈칸은 작성 전에 먼저 알려 드립니다."
)

with st.form("autobiography_form"):
    title = st.text_input("자서전 제목", value="나의 기억")
    request = st.text_area(
        "작성 요청",
        placeholder="예: 학창 시절의 중요한 변화와 사람들을 중심으로 써 주세요.",
    )
    target_period = st.text_input(
        "대상 기간",
        placeholder="예: 2008년부터 2012년",
    )
    topics_text = st.text_input(
        "주제",
        placeholder="쉼표로 구분해 주세요. 예: 학교, 우정, 성장",
    )
    chapter_count = st.slider("장 수", min_value=1, max_value=3, value=1)
    submitted = st.form_submit_button("초안 생성", type="primary")

generated_result: AutobiographyResult | None = None
if submitted:
    if not title.strip() or not request.strip():
        st.warning("제목과 작성 요청을 입력해 주세요.")
    else:
        topics = [
            topic.strip()
            for topic in topics_text.split(",")
            if topic.strip()
        ]
        payload = {
            "title": title.strip(),
            "request": request.strip(),
            "target_period": target_period.strip() or None,
            "target_topics": list(dict.fromkeys(topics)),
            "chapter_count": chapter_count,
        }
        with st.spinner("이 자서전과 관련된 중요한 기억 빈칸을 확인하는 중입니다."):
            try:
                gap_check = get_api_client().check_autobiography_gaps(**payload)
            except ApiClientError as exception:
                show_backend_error("자서전 기억 빈칸 확인", exception)
            else:
                if gap_check.requires_gap_confirmation:
                    st.session_state["pending_autobiography"] = payload
                    st.session_state["pending_autobiography_gap_count"] = len(
                        gap_check.important_unresolved_gaps
                    )
                else:
                    st.session_state.pop("pending_autobiography", None)
                    st.session_state.pop("pending_autobiography_gap_count", None)
                    generated_result = _generate(payload, proceed=False)

pending = st.session_state.get("pending_autobiography")
if isinstance(pending, dict):
    gap_count = st.session_state.get("pending_autobiography_gap_count", 0)
    st.warning(
        f"이 자서전과 관련된 중요한 기억 빈칸이 {gap_count}개 있습니다. "
        "먼저 채우면 더 구체적으로 쓸 수 있습니다."
    )
    first, second = st.columns(2)
    if first.button("기억을 먼저 채우기", type="primary"):
        st.switch_page("pages/gaps.py")
    proceed_confirmed = second.checkbox(
        "미확인 세부사항을 만들지 않고 작성하는 데 동의합니다.",
        key="autobiography-unresolved-confirm",
    )
    if second.button(
        "현재 자료로 생성",
        disabled=not proceed_confirmed,
    ):
        generated_result = _generate(pending, proceed=True)
        if generated_result is not None:
            st.session_state.pop("pending_autobiography", None)
            st.session_state.pop("pending_autobiography_gap_count", None)

if generated_result is not None:
    _render_result(generated_result)
