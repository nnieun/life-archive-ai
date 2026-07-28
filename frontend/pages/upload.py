"""Immutable TXT upload page."""

import streamlit as st

from frontend.api_client import ApiClientError
from frontend.ui import get_api_client, show_backend_error

st.title("TXT 기억 업로드")
st.caption("외부 STT로 만든 UTF-8 TXT 파일을 등록합니다. 기존 원본은 덮어쓰지 않습니다.")

uploaded_file = st.file_uploader("TXT 파일", type=["txt"])
language = st.text_input("언어", value="ko", max_chars=32)

if st.button("처리 및 인덱싱", type="primary", disabled=uploaded_file is None):
    assert uploaded_file is not None
    with st.status("TXT를 처리하고 기억을 인덱싱하는 중입니다.", expanded=True) as status:
        try:
            result = get_api_client().ingest_transcript(
                uploaded_file.name,
                uploaded_file.getvalue(),
                language=language.strip() or None,
            )
        except ApiClientError:
            status.update(label="처리에 실패했습니다.", state="error")
            show_backend_error("TXT 업로드")
        else:
            status.write(f"세그먼트 {result.segment_count}개 처리")
            status.write(f"구조화 기억 {result.memory_count}개 생성")
            status.update(label="처리와 인덱싱이 완료되었습니다.", state="complete")
            st.success(
                f"{result.indexed_memory_count}개의 기억이 검색 인덱스에 반영되었습니다."
            )
            st.code(result.transcript_id)
