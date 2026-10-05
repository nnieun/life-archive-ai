"""Submit immutable uploads and render background results."""

import streamlit as st
from uuid import uuid4

from frontend.api_client import ApiClientError
from frontend.ui import get_api_client, show_backend_error
from frontend.job_progress import upload_job_progress

st.title("TXT/PDF 기억 업로드")
st.caption("STT TXT 또는 텍스트가 포함된 PDF를 업로드합니다. 원본 파일은 변경하지 않습니다.")

if 'upload_session_id' not in st.session_state:
    st.session_state.upload_session_id = 'upload_session_' + uuid4().hex
pending = bool(st.session_state.get('pending_upload_job'))
uploaded_file = st.file_uploader('TXT 또는 PDF 파일', type=['txt', 'pdf'], disabled=pending)
language = st.text_input('언어', value='ko', max_chars=32, disabled=pending)

if st.button('처리 및 인덱싱', type='primary', disabled=uploaded_file is None or pending):
    try:
        job = get_api_client().submit_ingestion_job(uploaded_file.name, uploaded_file.getvalue(),
            session_id=st.session_state.upload_session_id, language=language.strip() or None)
    except ApiClientError as exception:
        if exception.status_code == 409:
            st.error('이미 등록된 TXT 또는 PDF 파일이거나 같은 내용의 파일이 처리 중입니다.')
        else:
            show_backend_error('파일 업로드', exception)
    else:
        st.session_state.pending_upload_job = {'job_id': job.job_id, 'session_id': job.session_id,
                                              'created_at': job.created_at.isoformat(), 'filename': job.filename}
        st.session_state.pop('last_upload_job', None)
        st.session_state.pop('upload_completion_notice', None)
        st.rerun()

upload_job_progress()

last = st.session_state.get('last_upload_job')
if last:
    result = last.get('result')
    if result:
        st.success(f"{result['indexed_memory_count']}개의 기억을 검색 인덱스에 반영했습니다.")
        st.write(f"세그먼트 {result['segment_count']}개 처리 · 기억 {result['memory_count']}개 생성 · 기억 빈칸 {result['gap_count']}개 발견")
        if result['gap_count']:
            st.warning(f"기억에서 확인이 필요한 빈칸 {result['gap_count']}개를 찾았습니다.")
        if result['gap_count'] and st.button('기억 빈칸 확인', key='open-memory-gaps'):
            st.switch_page('pages/gaps.py')
        st.caption(f"저장된 원본 파일: {result['filename']}")
    else:
        st.error(last.get('error') or '업로드 처리에 실패했습니다.')
        if last.get('failure_code'):
            st.caption(f"오류 유형: {last['failure_code']}")
