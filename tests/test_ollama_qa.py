"""Native local inference request, truncation, and HTTP failure contracts."""

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from backend.app.models.qa import AnswerProposal
from backend.app.services.ollama_qa import OllamaQAModel
from backend.app.services.qa import _invoke_structured, QAError


def test_native_options_disable_thinking_keep_residency_and_enforce_schema(monkeypatch):
    captured = {}
    def post(self, url, **kwargs):
        captured.update(url=url, **kwargs)
        return httpx.Response(200, request=httpx.Request('POST', url),
            json={'message': {'content': '{"reason":"unsupported","claims":[]}'}, 'done_reason': 'stop', 'load_duration': 123})
    monkeypatch.setattr('backend.app.services.ollama_qa.httpx.Client.post', post)
    model = OllamaQAModel('demo', 'http://localhost:11434/v1/', AnswerProposal)
    result = _invoke_structured(model, [SystemMessage(content='rule'), HumanMessage(content='data')], AnswerProposal)
    assert result.claims == []
    assert captured['url'] == 'http://localhost:11434/api/chat'
    assert captured['json']['think'] is False
    assert captured['json']['keep_alive'] == '15m'
    assert captured['json']['options']['num_predict'] == 768
    assert captured['json']['format'] == AnswerProposal.model_json_schema()
    assert [item['role'] for item in captured['json']['messages']] == ['system', 'user']


@pytest.mark.parametrize('payload,expected', [
    ({'done_reason': 'length', 'message': {'content': '{'}}, 'output_truncated'),
    ({'message': {'content': '{"claims":[]}'}}, 'output_schema_invalid'),
])
def test_bad_native_outputs_are_rejected(monkeypatch, payload, expected):
    monkeypatch.setattr('backend.app.services.ollama_qa.httpx.Client.post',
        lambda self, url, **kw: httpx.Response(200, json=payload, request=httpx.Request('POST', url)))
    with pytest.raises(QAError) as error:
        _invoke_structured(OllamaQAModel('demo', 'http://localhost:11434/v1', AnswerProposal), [], AnswerProposal)
    assert error.value.code == expected


def test_native_missing_model_is_classified(monkeypatch):
    monkeypatch.setattr('backend.app.services.ollama_qa.httpx.Client.post',
        lambda self, url, **kw: httpx.Response(404, request=httpx.Request('POST', url)))
    with pytest.raises(QAError) as error:
        _invoke_structured(OllamaQAModel('demo', 'http://localhost:11434/v1', AnswerProposal), [], AnswerProposal)
    assert error.value.code == 'model_not_found'
