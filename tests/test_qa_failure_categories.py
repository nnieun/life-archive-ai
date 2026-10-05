"""Failure categories distinguish transport, parsing, and semantic failures."""

from unittest.mock import Mock

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage

from backend.app.models.qa import EvidenceAssessment, EvidenceSelection
from backend.app.services.qa import _invoke_structured, QAError
import json


@pytest.mark.parametrize("error,code", [
    (httpx.ReadTimeout("sensitive"), "model_timeout"),
    (httpx.ConnectError("sensitive"), "model_connection"),
    (RuntimeError("sensitive"), "model_call_failed"),
])
def test_transport_categories_do_not_store_exception_messages(error, code):
    model = Mock()
    model.invoke.side_effect = error
    with pytest.raises(QAError) as caught:
        _invoke_structured(model, [], EvidenceAssessment)
    assert caught.value.code == code
    assert "sensitive" not in str(caught.value)


@pytest.mark.parametrize("status,code", [(401,"model_authentication"),(403,"model_permission"),
    (404,"model_not_found"),(429,"model_rate_limit"),(500,"model_server_error"),(400,"model_request_invalid")])
def test_http_status_categories(status, code):
    model = Mock()
    response = httpx.Response(status, request=httpx.Request("POST", "http://localhost"))
    model.invoke.side_effect = openai.APIStatusError("private", response=response, body=None)
    with pytest.raises(QAError) as caught:
        _invoke_structured(model, [], EvidenceAssessment)
    assert caught.value.code == code


@pytest.mark.parametrize("output,code", [
    ("invalid", "output_envelope_invalid"),
    ({"parsed": None}, "output_missing"),
    ({"parsed": {"sufficient": True, "reason": "test", "selected_memory_ids": []}}, "output_schema_invalid"),
    ({"parsed": None, "raw": AIMessage(content="", additional_kwargs={"refusal": "private"})}, "model_refusal"),
    ({"parsed": None, "raw": AIMessage(content="", response_metadata={"finish_reason":"length"})}, "output_truncated"),
    ({"parsed": None, "parsing_error": json.JSONDecodeError("bad", "secret", 0)}, "output_parse_failed"),
])
def test_structured_output_categories(output, code):
    model = Mock()
    model.invoke.return_value = output
    with pytest.raises(QAError) as caught:
        _invoke_structured(model, [], EvidenceAssessment)
    assert caught.value.code == code


@pytest.mark.parametrize("ids,sufficient", [([], False), (["memory-1"], True)])
def test_local_selection_has_one_authoritative_sufficiency_value(ids, sufficient):
    model = Mock()
    model.invoke.return_value = {"parsed": EvidenceSelection(reason="test", selected_memory_ids=ids)}
    assessed = _invoke_structured(model, [], EvidenceAssessment)
    assert assessed.sufficient is sufficient
    assert assessed.selected_memory_ids == ids


def test_schema_details_exclude_private_values_and_identify_conflicting_rule():
    model = Mock()
    model.invoke.return_value = {"parsed": {"sufficient": False, "reason": "private",
                                           "selected_memory_ids": ["private-memory"]}}
    with pytest.raises(QAError) as caught:
        _invoke_structured(model, [], EvidenceAssessment)
    issue = caught.value.schema_issues[0]
    assert issue.rule == "selection_empty_when_insufficient"
    assert "private" not in issue.model_dump_json()


def test_missing_field_diagnostic_names_field_without_recording_input():
    model = Mock()
    model.invoke.return_value = {"parsed": {"sufficient": False, "selected_memory_ids": []}}
    with pytest.raises(QAError) as caught:
        _invoke_structured(model, [], EvidenceAssessment)
    assert caught.value.schema_issues[0].field == "reason"
    assert caught.value.schema_issues[0].error_type == "missing"
