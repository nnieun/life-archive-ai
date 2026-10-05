"""Poll chat jobs from the shared app shell across navigation changes."""

from time import time

import streamlit as st

from frontend.api_client import ApiClientError
from frontend.ui import get_api_client


def collect_chat_result(state, client) -> bool:
    pending = state.get("pending_chat_job")
    if not pending:
        return False
    job = client.get_chat_job(pending["job_id"], pending["session_id"])
    if job.status in ("queued", "running"):
        pending['progress'] = job.progress
        return False
    if job.result is not None:
        result = job.result
        labels = {}
        try:
            labels = {item.memory.memory_id: item.memory.title for item in client.list_memories()}
        except ApiClientError:
            pass
        if not any(message.get("job_id") == job.job_id for message in state.get("chat_messages", [])):
            state.setdefault("chat_messages", []).append({
                "role": "assistant", "content": result.final_answer,
                "citations": result.citations, "memory_labels": labels, "job_id": job.job_id,
                "failure_code": result.validation_result.failure_code,
                'elapsed_ms': result.elapsed_ms, 'cache_hit': result.cache_hit,
            })
        notification = "답변이 완료되었습니다."
    else:
        notification = job.error or "질문 처리가 중단되었습니다."
        state.setdefault("chat_messages", []).append({"role": "assistant", "content": notification,
                                                      "job_id": job.job_id, "failure_code": job.failure_code})
    state.pop("pending_chat_job", None)
    state["chat_completion_notice"] = {"text": notification, "expires_at": time() + 10}
    return True


@st.fragment(run_every=1)
def chat_job_notifications():
    if st.session_state.get("pending_chat_job"):
        try:
            completed = collect_chat_result(st.session_state, get_api_client())
        except ApiClientError:
            st.caption("질문 처리 상태를 확인할 수 없습니다. 연결되면 다시 확인합니다.")
        else:
            if completed:
                st.rerun()
    notice = st.session_state.get("chat_completion_notice")
    if notice and notice["expires_at"] > time():
        from html import escape
        st.html(
            '<div role="status" style="position:fixed;bottom:24px;right:24px;z-index:9999;'
            'padding:16px 24px;border-radius:12px;background:#166534;color:white;'
            'box-shadow:0 4px 16px #0003">' + escape(notice["text"]) + '</div>'
        )
