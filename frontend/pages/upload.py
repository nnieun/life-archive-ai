"""Immutable TXT/PDF upload page."""

import streamlit as st

from frontend.api_client import ApiClientError
from frontend.ui import get_api_client, show_backend_error

st.title("TXT/PDF 기억 업로드")
st.caption("STT TXT 또는 텍스트가 포함된 PDF를 업로드합니다. 원본 파일은 변경하지 않습니다.")

uploaded_file = st.file_uploader("TXT 또는 PDF 파일", type=["txt", "pdf"])
language = st.text_input("언어", value="ko", max_chars=32)

if st.button("처리 및 인덱싱", type="primary", disabled=uploaded_file is None):
    assert uploaded_file is not None
    with st.status("파일을 처리하고 기억을 인덱싱하는 중입니다.", expanded=True) as status:
        try:
            result = get_api_client().ingest_transcript(
                uploaded_file.name,
                uploaded_file.getvalue(),
                language=language.strip() or None,
            )
        except ApiClientError as exception:
            status.update(label="처리에 실패했습니다.", state="error")
            if exception.status_code == 409:
                st.error("이미 등록된 TXT 또는 PDF 파일이거나 같은 내용의 파일입니다.")
            else:
                show_backend_error("파일 업로드", exception)
        else:
            status.write(f"세그먼트 {result.segment_count}개 처리")
            status.write(f"구조화된 기억 {result.memory_count}개 생성")
            status.update(label="처리와 인덱싱이 완료되었습니다.", state="complete")
            st.success(f"{result.indexed_memory_count}개의 기억을 검색 인덱스에 반영했습니다.")
            st.code(result.transcript_id)
