"""Grounded autobiography generation page."""

import streamlit as st

from frontend.api_client import ApiClientError
from frontend.ui import get_api_client, render_citations, show_backend_error

st.title("자서전 초안")
st.caption("기간과 주제를 바탕으로 최대 3장의 근거 있는 초안을 생성합니다.")

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

if submitted:
    if not title.strip() or not request.strip():
        st.warning("제목과 작성 요청을 입력해 주세요.")
    else:
        topics = [
            topic.strip()
            for topic in topics_text.split(",")
            if topic.strip()
        ]
        with st.spinner("관련 기억을 찾고 장별 초안을 생성하는 중입니다."):
            try:
                result = get_api_client().generate_autobiography(
                    title=title.strip(),
                    request=request.strip(),
                    target_period=target_period.strip() or None,
                    target_topics=list(dict.fromkeys(topics)),
                    chapter_count=chapter_count,
                )
            except ApiClientError as exception:
                show_backend_error("자서전 생성", exception)
            else:
                if result.error:
                    st.warning(result.error)
                for index, chapter in enumerate(
                    result.autobiography.content.chapters,
                    start=1,
                ):
                    st.header(f"{index}장. {chapter.title}")
                    st.write(chapter.content)
                    render_citations(chapter.citations)
