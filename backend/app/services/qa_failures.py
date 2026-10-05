"""Safe failure classification without retaining exception messages."""

import json
import httpx
import openai
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError


def classify_exception(error: BaseException) -> str:
    if isinstance(error, (openai.APITimeoutError, httpx.TimeoutException, TimeoutError)):
        return "model_timeout"
    if isinstance(error, (openai.APIConnectionError, httpx.ConnectError)):
        return "model_connection"
    if isinstance(error, (openai.APIStatusError, httpx.HTTPStatusError)):
        status_code = error.status_code if isinstance(error, openai.APIStatusError) else error.response.status_code
        return {401: "model_authentication", 403: "model_permission", 404: "model_not_found",
                429: "model_rate_limit"}.get(status_code, "model_server_error" if status_code >= 500 else "model_request_invalid")
    if isinstance(error, ValidationError):
        return "output_schema_invalid"
    if isinstance(error, (OutputParserException, json.JSONDecodeError)):
        return "output_parse_failed"
    return "model_call_failed"


MESSAGES = {
    "storage_error": "저장된 기억을 읽거나 저장하는 중 오류가 발생했습니다.",
    "internal_error": "질문 처리 중 예상하지 못한 내부 오류가 발생했습니다.",
    "embedding_timeout": "검색 임베딩 응답 대기 시간이 초과됐습니다.",
    "embedding_connection": "검색 임베딩 서버에 연결하지 못했습니다.",
    "embedding_authentication": "검색 임베딩 API 인증에 실패했습니다.",
    "embedding_permission": "검색 임베딩 API 접근 권한이 없습니다.",
    "embedding_rate_limit": "검색 임베딩 API 사용량 제한에 도달했습니다.",
    "embedding_not_found": "검색 임베딩 모델 또는 API 경로를 찾지 못했습니다.",
    "embedding_server_error": "검색 임베딩 서버 오류가 발생했습니다.",
    "embedding_request_invalid": "검색 임베딩 요청 형식이 올바르지 않습니다.",
    "model_timeout": "모델 응답 대기 시간이 초과됐습니다.",
    "model_connection": "모델 서버에 연결하지 못했습니다. Ollama 실행 상태를 확인해 주세요.",
    "model_authentication": "모델 API 인증에 실패했습니다.",
    "model_permission": "모델 API 접근 권한이 없습니다.",
    "model_not_found": "요청한 모델 또는 API 경로를 찾지 못했습니다.",
    "model_rate_limit": "모델 API 사용량 제한에 도달했습니다.",
    "model_server_error": "모델 서버에서 오류가 발생했습니다.",
    "model_request_invalid": "모델 서버가 요청 형식이나 옵션을 처리하지 못했습니다.",
    "output_schema_invalid": "모델 응답이 필요한 필드나 값 형식을 지키지 않았습니다.",
    "output_parse_failed": "모델 응답을 구조화된 데이터로 읽지 못했습니다.",
    "output_truncated": "모델 응답이 길이 제한으로 중단됐습니다.",
    "model_refusal": "모델이 응답을 거절했습니다.",
    "output_missing": "모델이 사용할 수 있는 구조화 응답을 반환하지 않았습니다.",
    "output_envelope_invalid": "모델 응답 형식이 올바르지 않습니다.",
    "unknown_memory_selected": "모델이 검색 결과에 없는 기억을 선택했습니다.",
    "citation_outside_evidence": "답변이 선택한 근거에 없는 기억을 인용했습니다.",
    "verification_index_invalid": "검증 모델이 잘못된 주장 번호를 반환했습니다.",
    "verification_memory_invalid": "검증 모델이 필수 기억 목록에 없는 기억을 반환했습니다.",
    "incomplete_answer": "질문에 필요한 관련 기억이 답변에서 누락됐습니다.",
    "unsupported_number": "답변의 숫자나 날짜가 인용한 기억에서 확인되지 않았습니다.",
    "unsupported_person": "답변의 인물이 인용한 기억에서 확인되지 않았습니다.",
    "source_untraceable": "인용한 기억의 원문 근거를 확인하지 못했습니다.",
    "verifier_rejected": "근거 검증에서 뒷받침되지 않는 주장이 발견됐습니다.",
    "draft_missing": "검증할 답변 초안이 없습니다.",
    "retrieval_failed": "기억 검색 처리에 실패했습니다.",
    "model_call_failed": "모델 호출 중 분류되지 않은 오류가 발생했습니다.",
}
