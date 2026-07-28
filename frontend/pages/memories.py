"""Structured-memory browsing page."""

import streamlit as st

from frontend.api_client import ApiClientError
from frontend.ui import get_api_client, render_citations, show_backend_error

st.title("구조화된 기억")
st.caption("SQLite에 저장된 기억, 불확실성, 원문 위치를 확인합니다.")

try:
    memories = get_api_client().list_memories()
except ApiClientError:
    show_backend_error("기억 조회")
else:
    if not memories:
        st.info("아직 저장된 기억이 없습니다. 먼저 TXT 파일을 업로드해 주세요.")
    for item in memories:
        memory = item.memory
        with st.expander(memory.title, expanded=False):
            st.write(memory.summary)
            left, right = st.columns(2)
            left.metric("신뢰도", f"{memory.confidence:.0%}")
            right.write(f"날짜: {memory.event_date or '알 수 없음'}")
            right.caption(f"정밀도: {memory.date_precision}")
            if memory.people:
                st.write("인물:", ", ".join(memory.people))
            if memory.location:
                st.write("장소:", memory.location)
            if memory.uncertainty_notes:
                st.warning(f"불확실성: {memory.uncertainty_notes}")
            else:
                st.caption("별도로 기록된 불확실성 없음")
            render_citations(item.citations)
