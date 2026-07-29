# Life Archive AI API 명세

## 기본 정보

- Base URL: `http://127.0.0.1:8000/api/v1`
- Content-Type: `application/json`
- OpenAPI UI: `http://127.0.0.1:8000/docs`
- 모든 응답에 `X-Request-ID` 헤더를 반환한다.

## 공통 오류

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "request_id": "7d42e599dfac482ca8907c79957bc333"
  }
}
```

| 상태 | 의미 |
|---:|---|
| 404 | 리소스 없음 |
| 409 | 중복 파일·내용 또는 ID 충돌 |
| 422 | 요청 형식·필드 검증 실패 |
| 500 | 예상하지 못한 내부 오류 |
| 503 | SQLite, OpenAI 또는 검색 인덱스 사용 불가 |

오류 메시지는 내부 경로, 원문, API 키나 stack trace를 포함하지 않는다.

## 엔드포인트 요약

| Method | Path | 설명 |
|---|---|---|
| GET | `/health` | 서비스 상태 |
| POST | `/memories/ingest` | TXT 수집·기억 추출·색인 |
| GET | `/memories` | 활성 기억과 출처 |
| POST | `/chat` | 근거 기반 질문 답변 |
| POST | `/timeline` | 타임라인 |
| POST | `/autobiographies` | 자서전 생성 |
| GET | `/autobiographies/{autobiography_id}` | 자서전 조회 |
| DELETE | `/transcripts/{transcript_id}` | 기록 논리 삭제 |

## GET `/health`

응답:

```json
{
  "status": "ok",
  "service": "Life Archive AI",
  "version": "0.0.0"
}
```

버전은 TASK-020에서 `0.1.0`으로 변경할 예정이다.

## POST `/memories/ingest`

Streamlit client는 원본 bytes를 base64로 전송한다.

```json
{
  "filename": "synthetic-memory.txt",
  "content_base64": "7ZWp7ISxIOuNsOydtO2EsA==",
  "language": "ko",
  "recorded_at": "2020-01-01T09:00:00+09:00"
}
```

규칙:

- `filename`: 경로가 아닌 `.txt` 파일명, 최대 255자
- `content_base64`: UTF-8 TXT bytes, 최대 20,000,000자 transport field
- `language`: 선택, 최대 32자
- `recorded_at`: 선택, timezone-aware datetime

성공 응답:

```json
{
  "transcript_id": "tr_example",
  "filename": "synthetic-memory.txt",
  "segment_count": 1,
  "memory_count": 1,
  "indexed_memory_count": 1,
  "memory_ids": ["mem_example"]
}
```

같은 파일명 또는 같은 내용은 `409`다. 빈 파일, 비 TXT, 경로형 파일명,
잘못된 base64나 비 UTF-8은 `422`다.

## GET `/memories`

선택 query:

- `transcript_id`: 특정 transcript의 활성 기억만 조회

응답 항목:

```json
{
  "memory": {
    "memory_id": "mem_example",
    "transcript_id": "tr_example",
    "title": "공원에서 친구를 만난 날",
    "summary": "친구를 만나 이야기를 나눴다.",
    "people": ["친구"],
    "location": "공원",
    "event_date": "2020",
    "date_precision": "year",
    "emotion": "반가움",
    "confidence": 0.9,
    "uncertainty_notes": null,
    "status": "active",
    "supersedes_memory_id": null,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "deleted_at": null
  },
  "citations": [
    {
      "memory_id": "mem_example",
      "transcript_id": "tr_example",
      "segment_id": "seg_example",
      "start_offset": 0,
      "end_offset": 20
    }
  ]
}
```

실제 응답은 위 항목의 배열이다.

## POST `/chat`

요청:

```json
{
  "session_id": "session_demo",
  "question": "친구를 어디에서 만났어?",
  "top_k": 5
}
```

- `session_id`: 1~200자
- `question`: 1~4,000자
- `top_k`: 1~20, 기본값 5

응답은 `session_id`, 원 질문, `retrieved_memory_ids`, `final_answer`,
`citations`, 마지막 `validation_result`, `retry_count`(0~1), 선택적
`error`를 포함한다. 근거가 부족하거나 최종 검증에 실패하면 안전한 거절
답변을 반환한다.

## POST `/timeline`

요청:

```json
{
  "start_date": "2019-01-01",
  "end_date": "2021-12-31"
}
```

두 날짜는 선택 사항이며 범위 양 끝을 포함한다. 시작일이 종료일보다 늦으면
`422`다. 응답은 날짜가 해석된 `events`, 날짜 미상 `undated_events`,
적용된 `start_date`와 `end_date`를 포함한다.

## POST `/autobiographies`

요청:

```json
{
  "title": "나의 기억",
  "request": "친구와 성장에 관한 이야기를 작성해 줘",
  "target_period": "2018년부터 2022년",
  "target_topics": ["친구", "성장"],
  "chapter_count": 2,
  "top_k": 10
}
```

- `autobiography_id`: 선택; 생략하면 서버 생성
- `title`: 1~200자
- `request`: 1~4,000자
- `target_period`: 선택, 최대 200자
- `target_topics`: 중복·빈 문자열 없는 최대 20개
- `chapter_count`: 1~3
- `top_k`: 1~30

응답은 저장된 `autobiography`, 완료 여부, 검색 기억 ID, 인용,
`retry_count`, 선택적 `error`를 포함한다. 같은 ID는 `409`다.

## GET `/autobiographies/{autobiography_id}`

저장된 `draft`, `completed` 또는 보이는 상태의 자서전을 반환한다. 없으면
`404`다.

## DELETE `/transcripts/{transcript_id}`

응답:

```json
{
  "transcript_id": "tr_example",
  "deleted_segment_count": 1,
  "deleted_memory_count": 1,
  "deleted_vector_count": 1,
  "bm25_memory_count": 0,
  "invalidated_conversation_message_count": 1,
  "invalidated_autobiography_count": 1,
  "raw_file_deleted": false
}
```

SQLite soft deletion이 먼저 commit되며, 관련 Chroma vector를 제거하고
BM25를 rebuild한다. raw 원본은 삭제하지 않는다. transcript가 없으면
`404`, 인덱스 정리가 실패하면 SQLite 삭제 상태를 유지하고 `503`을
반환한다.
