"""Chronological memory timeline page."""

from datetime import date

import streamlit as st

from frontend.api_client import ApiClientError, TimelineEvent
from frontend.ui import get_api_client, render_citations, show_backend_error


def render_event(event: TimelineEvent) -> None:
    with st.container(border=True):
        st.subheader(event.title)
        st.caption(f"{event.date_label} · 날짜 정밀도: {event.date_precision}")
        st.write(event.description)
        if event.uncertainty_notes:
            st.warning(f"불확실성: {event.uncertainty_notes}")
        render_citations(event.citations)


st.title("기억 타임라인")
st.caption("날짜가 있는 사건을 순서대로 보고, 날짜 정밀도와 불확실성을 확인합니다.")

use_filter = st.checkbox("기간 필터 사용")
start_date: date | None = None
end_date: date | None = None
if use_filter:
    first, second = st.columns(2)
    start_date = first.date_input("시작일", value=date(1900, 1, 1))
    end_date = second.date_input("종료일", value=date.today())

if st.button("타임라인 조회", type="primary"):
    if start_date and end_date and start_date > end_date:
        st.warning("시작일은 종료일보다 늦을 수 없습니다.")
    else:
        try:
            result = get_api_client().get_timeline(
                start_date=start_date,
                end_date=end_date,
            )
        except ApiClientError:
            show_backend_error("타임라인 조회")
        else:
            st.subheader("날짜가 있는 기억")
            if not result.events:
                st.info("조건에 맞는 날짜 기억이 없습니다.")
            for event in result.events:
                render_event(event)
            st.subheader("날짜를 알 수 없는 기억")
            if not result.undated_events:
                st.caption("날짜 미상 기억이 없습니다.")
            for event in result.undated_events:
                render_event(event)
