"""날짜순 기억 Timeline 페이지."""

from datetime import date

import streamlit as st

from frontend.api_client import ApiClientError, TimelineEvent
from frontend.citations import render_citations
from frontend.ui import get_api_client, show_backend_error

DATE_PRECISION_LABELS = {
    "exact": "정확한 날짜와 시간",
    "day": "일 단위까지 확인",
    "month": "월 단위까지 확인",
    "year": "연도만 확인",
    "approximate": "대략적인 날짜",
    "unknown": "날짜를 확인할 수 없음",
}


def render_event(event: TimelineEvent) -> None:
    with st.container(border=True):
        st.subheader(event.title)
        precision = DATE_PRECISION_LABELS.get(event.date_precision, "확인 필요")
        st.caption(f"{event.date_label} · 날짜 정밀도: {precision}")
        st.write(event.description)
        if event.uncertainty_notes:
            st.warning(f"불확실성: {event.uncertainty_notes}")
        else:
            st.caption("추가로 기록된 불확실성이 없습니다.")
        render_citations(event.citations, scope=f"timeline-{event.memory_id}")


st.title("기억 Timeline")
st.caption("날짜가 있는 기억을 시간순으로 확인하고, 날짜가 불확실한 기억은 따로 봅니다.")

use_filter = st.checkbox("기간 필터 사용")
start_date: date | None = None
end_date: date | None = None
if use_filter:
    first, second = st.columns(2)
    start_date = first.date_input("시작일", value=date(1900, 1, 1))
    end_date = second.date_input("종료일", value=date.today())

if st.button("Timeline 조회", type="primary"):
    if start_date and end_date and start_date > end_date:
        st.warning("시작일은 종료일보다 늦을 수 없습니다.")
    else:
        try:
            result = get_api_client().get_timeline(
                start_date=start_date,
                end_date=end_date,
            )
        except ApiClientError as exception:
            show_backend_error("Timeline 조회", exception)
        else:
            st.subheader("날짜가 있는 기억")
            if not result.events:
                st.info("조건에 맞는 날짜 기억이 없습니다.")
            for event in result.events:
                render_event(event)
            st.subheader("날짜를 확인할 수 없는 기억")
            if not result.undated_events:
                st.caption("날짜를 확인할 수 없는 기억이 없습니다.")
            for event in result.undated_events:
                render_event(event)
