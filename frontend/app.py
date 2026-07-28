"""Streamlit entry point for the Life Archive AI MVP."""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Life Archive AI",
    page_icon="🗃️",
    layout="wide",
)

navigation = st.navigation(
    [
        st.Page("pages/upload.py", title="업로드", icon=":material/upload_file:"),
        st.Page("pages/memories.py", title="기억", icon=":material/book_2:"),
        st.Page("pages/chat.py", title="대화", icon=":material/chat:"),
        st.Page("pages/timeline.py", title="타임라인", icon=":material/timeline:"),
        st.Page(
            "pages/autobiography.py",
            title="자서전",
            icon=":material/auto_stories:",
        ),
    ]
)
navigation.run()
