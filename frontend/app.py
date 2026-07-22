"""Streamlit entry point for the MVP interface."""

from os import getenv

import streamlit as st
from dotenv import load_dotenv

from frontend.api_client import ApiClientError, LifeArchiveApiClient

load_dotenv()

st.set_page_config(page_title="Life Archive AI", page_icon="🧠")
st.title("Life Archive AI")
st.caption("Memory-Centric RAG")

api_url = getenv("LIFE_ARCHIVE_API_URL", "http://127.0.0.1:8000/api/v1")
st.subheader("API 상태")

try:
    health = LifeArchiveApiClient(base_url=api_url).get_health()
except ApiClientError:
    st.error("Backend API에 연결할 수 없습니다.")
else:
    st.success(f"{health.service} API가 정상입니다.")
    st.json(health.model_dump())

if st.button("상태 새로고침"):
    st.rerun()
