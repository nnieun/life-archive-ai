"""Render progress only on the page that submitted the job."""

from datetime import UTC, datetime

import streamlit as st


def elapsed_seconds(pending: dict) -> float:
    return max(0, (datetime.now(UTC) - datetime.fromisoformat(pending['created_at'])).total_seconds())


@st.fragment(run_every=1)
def chat_job_progress():
    pending = st.session_state.get('pending_chat_job')
    if not pending:
        return
    stages = {
        'retrieve': '관련 기억을 찾는 중입니다.',
        'evidence_sufficient': '답변 근거를 확인하는 중입니다.',
        'generate_answer': '기억을 바탕으로 답변을 작성하는 중입니다.',
        'verify_answer': '답변 근거를 검증하는 중입니다.',
        'rewrite_once': '답변을 다시 작성하는 중입니다.',
    }
    progress = pending.get('progress', {})
    label = stages.get(progress.get('stage'), '관련 기억을 찾는 중입니다.')
    st.caption(f'{label} · {elapsed_seconds(pending):.0f}초 경과')


@st.fragment(run_every=1)
def upload_job_progress():
    pending = st.session_state.get('pending_upload_job')
    if not pending:
        return
    stages = {
        'loading': '파일을 읽는 중입니다.',
        'chunking': '텍스트를 나누는 중입니다.',
        'extracting': '기억을 추출하는 중입니다.',
        'gaps': '기억 빈칸을 확인하는 중입니다.',
        'indexing': '검색 인덱스를 만드는 중입니다.',
        'completed': '처리 결과를 저장하는 중입니다.',
    }
    progress = pending.get('progress', {})
    label = stages.get(progress.get('stage'), '업로드 처리를 준비하는 중입니다.')
    count = f" · {progress.get('completed_segments', 0)}/{progress['total_segments']} 구간" if progress.get('total_segments') else ''
    st.caption(f'{label}{count} · {elapsed_seconds(pending):.0f}초 경과')
