"""Grounded chat page."""

from uuid import uuid4

import streamlit as st

from frontend.api_client import ApiClientError
from frontend.citations import remove_internal_citation_markers, render_citations
from frontend.ui import get_api_client, show_backend_error
from frontend.job_progress import chat_job_progress

st.title("기억과 대화")
st.caption("검색된 기억에 근거한 답변만 생성하며, 답변 아래에 출처를 표시합니다.")

if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = f"session_{uuid4().hex}"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for message_index, message in enumerate(st.session_state.chat_messages):
    with st.chat_message(message["role"]):
        st.write(remove_internal_citation_markers(message["content"]))
        if message.get("failure_code"):
            st.caption(f"오류 유형: {message['failure_code']}")
        if message.get('elapsed_ms') is not None:
            label = '저장된 검증 답변' if message.get('cache_hit') else '답변 처리'
            st.caption(f"{label} · {message['elapsed_ms']/1000:.2f}초")
        if message.get("citations"):
            render_citations(
                message["citations"],
                message.get("memory_labels"),
                scope=f"chat-{message_index}",
            )

chat_job_progress()

question = st.chat_input("기억에 대해 질문해 보세요", disabled=bool(st.session_state.get("pending_chat_job")))
if question:
    try:
        job = get_api_client().submit_chat_job(
            session_id=st.session_state.chat_session_id, question=question,
        )
    except ApiClientError as exception:
        show_backend_error("질문 제출", exception)
    else:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        st.session_state.pending_chat_job = {
            "job_id": job.job_id, "session_id": job.session_id,
            "created_at": job.created_at.isoformat(),
        }
        st.session_state.pop("chat_completion_notice", None)
        st.rerun()
