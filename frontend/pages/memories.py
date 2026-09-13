"""구조화된 기억 조회 및 원본별 삭제 페이지."""

import streamlit as st

from frontend.api_client import ApiClientError
from frontend.privacy_state import clear_derived_state
from frontend.ui import get_api_client, show_backend_error

st.title("구조화된 기억")
st.caption("저장된 기억과 연결된 원본 파일, 원문 위치를 확인합니다.")

try:
    api_client = get_api_client()
    memories = api_client.list_memories()
except ApiClientError as exception:
    show_backend_error("기억 조회", exception)
else:
    if not memories:
        st.info("아직 저장된 기억이 없습니다. 먼저 TXT 파일을 업로드해 주세요.")

    selected_memory_id = st.session_state.pop("selected_memory_id", None)
    shown_transcripts: set[str] = set()
    for item in memories:
        memory = item.memory
        with st.expander(
            memory.title,
            expanded=memory.memory_id == selected_memory_id,
        ):
            st.write(memory.summary)
            st.info(f"원본 파일: {item.source_filename}")
            for citation in item.citations:
                start_line = getattr(citation, "start_line", None)
                end_line = getattr(citation, "end_line", None)
                if start_line is not None and end_line is not None:
                    if start_line == end_line:
                        line_label = f"{start_line}번째 줄"
                    else:
                        line_label = f"{start_line}~{end_line}번째 줄"
                    location_label = (
                        "PDF 추출 텍스트 위치" if item.source_filename.lower().endswith(".pdf")
                        else "원문 위치"
                    )
                    st.caption(f"{location_label}: {line_label}")
            precision_labels = {
                "exact": "정확한 날짜와 시간",
                "day": "일 단위까지 확인",
                "month": "월 단위까지 확인",
                "year": "연도만 확인",
                "approximate": "대략적인 날짜",
                "unknown": "날짜를 확인할 수 없음",
            }
            left, right = st.columns(2)
            left.metric("기억 추출 신뢰도", f"{memory.confidence:.0%}")
            right.write(f"날짜: {memory.event_date or '알 수 없음'}")
            right.caption(
                "날짜 정밀도: "
                f"{precision_labels.get(memory.date_precision, '확인 필요')}"
            )
            if memory.people:
                st.write("인물:", ", ".join(memory.people))
            if memory.location:
                st.write("장소:", memory.location)
            if memory.uncertainty_notes:
                st.write("불확실한 점:", memory.uncertainty_notes)
                st.warning(f"불확실성: {memory.uncertainty_notes}")
            else:
                st.caption("추가로 기록된 불확실성이 없습니다.")
            if memory.transcript_id not in shown_transcripts:
                shown_transcripts.add(memory.transcript_id)
                st.divider()
                confirmed = st.checkbox(
                    "이 원본 파일과 연결된 기억 전체를 삭제하는 데 동의합니다.",
                    key=f"confirm-delete-{memory.transcript_id}",
                )
                if st.button(
                    "원본 연결 기억 삭제",
                    key=f"delete-transcript-{memory.transcript_id}",
                    disabled=not confirmed,
                ):
                    try:
                        result = api_client.delete_transcript(memory.transcript_id)
                    except ApiClientError as exception:
                        show_backend_error("기억 삭제", exception)
                    else:
                        clear_derived_state(st.session_state)
                        st.success(
                            f"{result.deleted_memory_count}개의 기억을 삭제했습니다. "
                            "원본 파일은 보존됩니다."
                        )
                        st.rerun()
