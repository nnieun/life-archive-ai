# 오류 처리와 안전한 로깅

## 공통 오류 응답

API 오류는 다음 형식을 사용한다.

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "request_id": "요청 추적 ID"
  }
}
```

응답의 `X-Request-ID` 헤더와 본문의 `request_id`는 같다. 안전한
`X-Request-ID` 요청 헤더가 있으면 유지하고, 없거나 허용되지 않는 문자가
포함되면 서버가 새 ID를 만든다.

## 로그 원칙

- 로그는 JSON 한 줄 형식으로 기록한다.
- 메서드, 라우트 템플릿, 상태 코드, 처리 시간, 요청 ID만 기록한다.
- 요청 본문, 쿼리 값, 원문, 개인정보, 로컬 경로는 기록하지 않는다.
- OpenAI 키와 Bearer 토큰은 마스킹한다.
- 예외 메시지와 스택 대신 예외 타입만 기록한다.

## 장애 처리

- 요청 모델 검증 실패: `422 validation_error`
- SQLite 장애: `503 storage_error`
- OpenAI 또는 검색 인덱스 장애: 각 API의 안전한 `503` 메시지
- 예상하지 못한 장애: `500 internal_error`

Streamlit 화면은 내부 예외 내용을 표시하지 않는다. 재현되는 장애에는
화면에 표시된 요청 ID를 사용해 안전한 구조화 로그를 찾는다.
