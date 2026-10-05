"""Observe upload jobs from the shared shell, independently of the upload page."""

from html import escape
from time import time
import streamlit as st
from frontend.api_client import ApiClientError
from frontend.ui import get_api_client


def collect_upload_result(state, client) -> bool:
    pending = state.get('pending_upload_job')
    if not pending:
        return False
    job = client.get_ingestion_job(pending['job_id'], pending['session_id'])
    if job.status in ('queued', 'running'):
        pending['progress'] = job.progress
        return False
    state['last_upload_job'] = job.model_dump(mode='json')
    state.pop('pending_upload_job', None)
    state['upload_completion_notice'] = {
        'text': '업로드 처리와 인덱싱이 완료되었습니다.' if job.result is not None else job.error or '업로드 처리가 중단되었습니다.',
        'expires_at': time() + 10, 'success': job.result is not None,
    }
    return True


@st.fragment(run_every=1)
def upload_job_notifications():
    if st.session_state.get('pending_upload_job'):
        try:
            completed = collect_upload_result(st.session_state, get_api_client())
        except ApiClientError:
            st.caption('업로드 처리 상태를 확인할 수 없습니다. 연결되면 다시 확인합니다.')
        else:
            if completed:
                st.rerun()
    notice = st.session_state.get('upload_completion_notice')
    if notice and notice['expires_at'] > time():
        color = '#166534' if notice['success'] else '#b91c1c'
        st.html('<div role="status" style="position:fixed;bottom:84px;right:24px;z-index:9999;'
                f'padding:16px 24px;border-radius:12px;background:{color};color:white;'
                'box-shadow:0 4px 16px #0003">' + escape(notice['text']) + '</div>')
