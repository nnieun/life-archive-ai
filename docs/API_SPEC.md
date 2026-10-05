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
| 428 | 사용자의 명시적 확인 필요 |
| 500 | 예상하지 못한 내부 오류 |
| 503 | SQLite, OpenAI 또는 검색 인덱스 사용 불가 |

오류 메시지는 내부 경로, 원문, API 키나 stack trace를 포함하지 않는다.

## 엔드포인트 요약

| Method | Path | 설명 |
|---|---|---|
| GET | `/health` | 서비스 상태 |
| POST | `/memories/ingest` | TXT/PDF 수집·기억 추출·색인 |
| GET | `/memories` | 활성 기억과 출처 |
| POST | `/memories/{memory_id}/corrections` | 기억 정정 |
| GET | `/memory-gaps` | 확인이 필요한 기억 빈칸 목록 |
| GET | `/memory-gaps/{gap_id}` | 기억 빈칸과 후보 조회 |
| POST | `/memory-gaps/{gap_id}/reconstruct` | 내부 기록에서 복원 후보 탐색 |
| POST | `/memory-gaps/{gap_id}/clues` | 사용자 단서 추가 |
| POST | `/memory-gaps/{gap_id}/resolve` | 명시적으로 확인한 후보 반영 |
| POST | `/memory-gaps/{gap_id}/dismiss` | 빈칸 확인 종료 |
| POST | `/chat` | 근거 기반 질문 답변 |
| POST | `/timeline` | 타임라인 |
| POST | `/autobiographies/gap-check` | 관련 중요 빈칸 사전 확인 |
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

- `filename`: 경로가 아닌 `.txt` 또는 `.pdf` 파일명, 최대 255자
- `content_base64`: 원본 파일 bytes, 최대 20,000,000자 transport field
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
  "memory_ids": ["mem_example"],
  "gap_count": 1,
  "gap_ids": ["gap_example"]
}
```

같은 파일명 또는 같은 내용은 `409`다. 빈 파일, 지원하지 않는 확장자, 경로형
파일명, 잘못된 base64, 비 UTF-8 TXT 또는 추출할 텍스트가 없는 PDF는 `422`다.

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

실제 응답은 위 항목의 배열이다. 정정으로 대체된 기억은 이 목록에 나오지
않는다. 타임라인·검색·자서전도 같은 조회 경로를 쓰므로 동일하게 제외된다.

## POST `/memories/{memory_id}/corrections`

기억을 제자리에서 고치지 않고, 원본을 대체하는 새 기억을 덧붙인다. 원본
행은 SQLite에 그대로 남아 추출 이력을 감사할 수 있고, 원본의 `memory_sources`
는 정정본에 그대로 승계되어 인용 추적성이 유지된다.

요청 (모든 필드 선택, 최소 한 개 필요):

```json
{
  "title": "공원에서 민수를 만난 날",
  "summary": "민수를 만나 이야기를 나눴다.",
  "people": ["민수"],
  "location": "공원",
  "event_date": "2021",
  "date_precision": "year",
  "emotion": null,
  "uncertainty_notes": null
}
```

- 생략한 필드는 원본 값을 물려받고, 명시한 `null`은 값을 지운다.
- `event_date`와 `date_precision`은 반드시 함께 보내야 한다.
- 정정본은 `status: "corrected"`, `supersedes_memory_id: "<원본 id>"`,
  `confidence: 1.0`으로 저장된다.

응답: `201`, `GET /memories` 항목과 같은 형태.

오류:

- `404` 원본 기억 없음 또는 삭제됨
- `409` 이미 정정된 기억 (기억 하나당 정정은 한 번)
- `422` 요청 본문이 위 규칙에 어긋남
- `503` 저장소 사용 불가

## 기억 빈칸 API

기억 빈칸은 추출된 기억에서 장소·사람·날짜 등이 빠졌거나 서로 충돌하는
지점을 뜻한다. 후보를 찾는 것과 기억을 수정하는 것은 분리되어 있다. 복원
에이전트는 후보만 `PROPOSED` 상태로 저장하며, 사용자가 확인하기 전에는
기억·타임라인·자서전에 반영하지 않는다.

빈칸 종류:

- `MISSING_LOCATION`, `MISSING_PERSON`, `MISSING_DATE`
- `UNCERTAIN_EVENT`, `CONFLICTING_FACT`, `WEAK_PROVENANCE`

상태 흐름:

```text
OPEN → SEARCHING → CANDIDATE_FOUND → WAITING_USER → RESOLVED
   └──────────────────────────────────────────────→ DISMISSED
```

### GET `/memory-gaps`

기본적으로 `OPEN`, `SEARCHING`, `CANDIDATE_FOUND`, `WAITING_USER` 상태만
반환한다. `include_closed=true`를 지정하면 `RESOLVED`, `DISMISSED` 이력도
포함한다.

```json
{
  "items": [
    {
      "gap": {
        "gap_id": "gap_example",
        "memory_id": "mem_example",
        "gap_type": "MISSING_LOCATION",
        "missing_field": "location",
        "status": "WAITING_USER",
        "importance_score": 0.8
      },
      "candidates": []
    }
  ]
}
```

실제 `gap`에는 단서, 기간, 인물, 근거 ID, 사용자 추가 단서, 생성·수정 시각
등 전체 `MemoryGapRecord`가 포함된다. `GET /memory-gaps/{gap_id}`는 같은
형태의 항목 하나를 반환한다.

### POST `/memory-gaps/{gap_id}/reconstruct`

요청 본문은 없다. 서버가 다음 로컬 도구를 정해진 순서와 호출 예산 안에서
사용한다.

1. `search_memory`
2. `search_uploaded_documents`
3. `search_memory_gaps`
4. `request_more_clues`

반환된 후보는 검색 결과에 실제로 포함된 source ID와 원문 그대로의 근거를
가져야 한다. 응답에는 `gap`, `candidates`, `searched_tools`,
`tool_call_count`, `needs_more_clues`, `user_question`, `message`가 포함된다.

### POST `/memory-gaps/{gap_id}/clues`

```json
{
  "clue": "그때 대구에 살고 있었어요."
}
```

중복 단서는 `409 memory_gap_clue_duplicate`, 이미 종료된 빈칸은
`409 memory_gap_already_closed`다. 단서를 추가하면 상태가 `OPEN`으로
돌아가므로 다시 후보를 탐색할 수 있다.

### POST `/memory-gaps/{gap_id}/resolve`

```json
{
  "candidate_id": "candidate_example",
  "user_confirmed": true
}
```

`user_confirmed`가 `true`일 때만 서버가 후보를 반영한다. 기억을 덮어쓰지
않고 append-only 정정본을 만든 뒤, 정정본·후보 상태·빈칸 상태를 하나의
SQLite transaction으로 확정한다. 성공 응답에는 `gap`, `candidate`,
`resolved_memory_id`가 포함된다.

주요 오류:

| 상태 | 코드 | 의미 |
|---:|---|---|
| 404 | `memory_gap_not_found` | 빈칸을 찾을 수 없음 |
| 404 | `memory_gap_candidate_not_found` | 후보를 찾을 수 없음 |
| 409 | `memory_gap_already_closed` | 이미 해결하거나 닫은 빈칸 |
| 409 | `memory_gap_candidate_mismatch` | 다른 빈칸의 후보 |
| 409 | `memory_gap_target_changed` | 후보 생성 뒤 원래 기억이 변경됨 |
| 422 | `memory_gap_candidate_source_invalid` | 후보 근거가 내부 기록과 일치하지 않음 |
| 422 | `memory_gap_resolution_unsupported` | 해당 필드에 안전하게 반영할 수 없음 |
| 428 | `memory_gap_confirmation_required` | 사용자의 명시적 확인이 없음 |
| 503 | `memory_gap_tool_policy_violation` | 허용한 검색 순서·횟수를 위반함 |
| 503 | `memory_gap_candidate_output_invalid` | 모델 후보의 구조 또는 근거가 잘못됨 |
| 503 | `memory_gap_model_unavailable` | 복원 모델을 사용할 수 없음 |
| 503 | `memory_gap_search_unavailable` | 내부 검색을 사용할 수 없음 |

### POST `/memory-gaps/{gap_id}/dismiss`

요청 본문은 없다. 빈칸을 `DISMISSED`로 닫으며 원래 기억은 수정하지 않는다.

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
  "top_k": 10,
  "proceed_with_unresolved_gaps": false
}
```

- `autobiography_id`: 선택; 생략하면 서버 생성
- `title`: 1~200자
- `request`: 1~4,000자
- `target_period`: 선택, 최대 200자
- `target_topics`: 중복·빈 문자열 없는 최대 20개
- `chapter_count`: 1~3
- `top_k`: 1~30
- `proceed_with_unresolved_gaps`: 관련 중요 빈칸을 확인한 뒤에도 현재 기록만으로
  진행할지 여부. 기본값은 `false`다.

응답은 저장된 `autobiography`, 완료 여부, 검색 기억 ID, 인용,
`retry_count`, 선택적 `error`, `important_unresolved_gaps`,
`requires_gap_confirmation`을 포함한다. 관련 중요 빈칸이 있고 진행 동의가
없으면 초안을 만들거나 저장하지 않고 `requires_gap_confirmation: true`를
반환한다. 같은 ID는 `409`다.

## POST `/autobiographies/gap-check`

`POST /autobiographies`와 같은 본문을 받으며 쓰기 작업 없이 관련도가 있고
중요도 `0.65` 이상인 미해결 빈칸을 미리 확인한다.

```json
{
  "important_unresolved_gaps": [],
  "retrieved_memory_ids": ["mem_example"],
  "requires_gap_confirmation": false
}
```

확인 작업 자체를 수행할 수 없으면 `503 autobiography_gap_check_unavailable`을
반환한다.

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
# 비동기 대화 작업

QA 결과의 `validation_result`에는 선택적 `failure_code`, `exception_type`이
추가됩니다. 그래프 밖 실패 시 작업 응답의 `failure_code`, `failure_stage`를
확인합니다. 기존 클라이언트 응답 필드는 유지합니다.

- `POST /api/v1/chat/jobs`: 기존 chat 요청 형식으로 제출, 202와 작업 ID 반환
- `GET /api/v1/chat/jobs/{job_id}?session_id=...`: 상태와 완료 결과 조회
- 같은 세션의 진행 작업이 있으면 제출은 409, 작업/세션 불일치는 조회 404

상세 동작은 [비동기 대화](ASYNC_CHAT.md)를 참고합니다.

## QA 속도와 진행 정보

업로드 백그라운드 API는 [비동기 업로드](ASYNC_UPLOAD.md)를 참고한다.
`POST /memories/ingest/jobs` 접수(202), `GET /memories/ingest/jobs/{job_id}` 상태 조회를 제공한다.

`QAResult`에는 `elapsed_ms`, `cache_hit`, `steps`가 추가된다.
`steps`는 단계별 시간과 검증 결과이며 캐시 응답에서는 빈 목록이다.
`elapsed_ms`는 QA 처리 시간으로, 큐 대기·서비스 최초 초기화·화면 폴링 시간은 별도다.

작업 응답의 `progress`는 `stage`, `elapsed_ms`, `memories`를 제공한다.
`memories`에는 검색된 기억 ID와 제목만 있고 검증 전 답변 초안은 없다.
진행 정보 갱신은 작업 실행 중일 때만 허용하며 개인정보 삭제 시 진행 정보도 지운다.

시간과 상태는 성공 여부와 관계없이 SQLite `qa_performance`에 기록한다.
실패의 상세 진단과 별도이며 문답 내용은 성능 기록에 저장하지 않는다.
