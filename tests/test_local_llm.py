"""Provider routing checks without external model calls."""

from unittest.mock import Mock

import pytest

from backend.app.core.config import Settings
from backend.app.services import autobiography, gap_reconstruction, memory_extraction, qa


def test_local_settings_do_not_use_openai_credentials():
    settings = Settings(openai_api_key="unused-secret")
    assert settings.chat_model == "gemma4:e2b"
    assert settings.chat_api_key == "ollama"
    assert settings.chat_base_url == "http://localhost:11434/v1"


def test_openai_settings_remain_available():
    settings = Settings(llm_provider="openai", openai_model="test", openai_api_key="key")
    assert settings.chat_model == "test"
    assert settings.chat_api_key == "key"
    assert settings.chat_base_url is None


def test_native_qa_can_use_separate_verifier_and_shared_connection(monkeypatch):
    from backend.app.services import ollama_qa
    constructor = Mock()
    monkeypatch.setattr(ollama_qa, 'OllamaQAModel', constructor)
    models = qa.build_openai_qa_models('generator', base_url='http://localhost:11434/v1',
        native_ollama=True, combined=True, verification_model='verifier')
    calls = constructor.call_args_list
    assert calls[1].args[0] == 'generator'
    assert calls[2].args[0] == 'verifier'
    assert calls[1].kwargs['client'] is calls[2].kwargs['client']
    assert calls[1].kwargs['think'] is False
    calls[1].kwargs['client'].close()


@pytest.mark.parametrize("module,builder", [
    (memory_extraction, memory_extraction.build_openai_memory_model),
    (memory_extraction, memory_extraction.build_openai_memory_localization_model),
    (qa, qa.build_openai_qa_models),
    (autobiography, autobiography.build_openai_autobiography_models),
    (gap_reconstruction, gap_reconstruction.build_openai_memory_gap_models),
])
def test_all_generation_stages_route_to_local_chat_completions(monkeypatch, module, builder):
    constructor = Mock()
    monkeypatch.setattr(module, "ChatOpenAI", constructor)
    args = ([], "gemma4:e2b") if module is gap_reconstruction else ("gemma4:e2b",)
    builder(*args, api_key="ollama", base_url="http://localhost:11434/v1")
    options = constructor.call_args.kwargs
    assert options["base_url"] == "http://localhost:11434/v1"
    assert options["api_key"] == "ollama"
    assert options["use_responses_api"] is False
    assert constructor.return_value.with_structured_output.called
