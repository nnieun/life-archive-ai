"""Smoke tests for the minimum TASK-002 dependency set."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "chromadb",
        "dotenv",
        "fastapi",
        "httpx",
        "langchain",
        "langchain_chroma",
        "langchain_community",
        "langchain_core",
        "langchain_openai",
        "langgraph",
        "duckduckgo_search",
        "pypdf",
        "pydantic",
        "rank_bm25",
        "streamlit",
        "uvicorn",
    ],
)
def test_minimum_dependency_imports(module_name: str) -> None:
    """Each required runtime dependency can be imported."""
    import_module(module_name)
