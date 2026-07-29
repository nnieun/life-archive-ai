"""Grounded chat page."""

from uuid import uuid4

import streamlit as st

from frontend.api_client import ApiClientError
from frontend.ui import get_api_client, render_citations, show_backend_error

st.title("기억과 대화")
st.caption("검색된 기억에 근거한 답변만 생성하며, 답변 아래에 출처를 표시합니다.")

if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = f"session_{uuid4().hex}"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("citations"):
            render_citations(message["citations"])

question = st.chat_input("기억에 대해 질문해 보세요")
if question:
    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("관련 기억을 찾는 중입니다."):
            try:
                result = get_api_client().chat(
                    session_id=st.session_state.chat_session_id,
                    question=question,
                )
            except ApiClientError as exception:
                show_backend_error("질문 처리", exception)
            else:
                st.write(result.final_answer)
                render_citations(result.citations)
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": result.final_answer,
                        "citations": result.citations,
                    }
                )
