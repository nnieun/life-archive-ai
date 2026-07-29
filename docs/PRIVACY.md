# 개인정보 보호와 삭제 정책

## 1. 범위

Life Archive AI는 사용자가 제공한 STT transcript를 민감한 로컬 데이터로
취급한다. transcript, 질문, 추출 기억과 생성 자서전은 신뢰하지 않는
사용자 데이터이며 코드나 지시문으로 실행하지 않는다.

## 2. 로컬 저장 위치

| 데이터 | 기본 위치 | Git |
|---|---|---|
| 불변 raw TXT | `data/raw/transcripts/` | 제외 |
| SQLite 기준 데이터 | `data/db/` | 제외 |
| Chroma 검색 인덱스 | `data/indexes/chroma/` | 제외 |
| 생성 export | `data/exports/` | 제외 |
| API 키 | `.env` | 제외 |

`.env.example`에는 키 이름과 빈 값만 포함한다. 실제 개인 파일, DB와
인덱스는 공개 저장소에 추가하지 않는다.

## 3. 최소 저장

- SQLite에는 서비스 기능에 필요한 transcript, memory, source,
  conversation과 autobiography만 저장한다.
- Chroma에는 제목+요약 임베딩과 `memory_id`, `embedding_version`,
  `content_hash`만 저장한다.
- BM25는 메모리 내 폐기 가능한 인덱스다.
- Chroma와 BM25 결과는 항상 SQLite 활성 상태로 다시 검증한다.

## 4. 안전한 로그

로그에 다음 내용을 기록하지 않는다.

- transcript·질문·답변·자서전 본문
- memory summary, 사람, 장소
- request body, query 값과 실제 URL path parameter
- API 키, Bearer token
- 로컬 경로, SQL 문, exception message와 stack trace

허용된 JSON 필드는 timestamp, level, event, request ID, HTTP method,
route template, status code, duration과 exception type이다. 알려진 OpenAI
키와 Bearer token 패턴은 formatter에서 한 번 더 마스킹한다.

## 5. 요청 추적

각 API 응답은 동일한 `request_id`를 오류 본문과 `X-Request-ID` 헤더에
포함한다. 호출자가 보낸 ID는 영문·숫자와 `._-`로 구성된 최대 64자만
수락하며, 그 외에는 서버가 새 ID를 만든다. Streamlit은 내부 예외 대신
이 ID만 사용자에게 안내한다.

## 6. 애플리케이션 삭제

`DELETE /api/v1/transcripts/{transcript_id}`는 다음 순서로 처리한다.

1. transcript와 segment를 하나의 SQLite transaction에서 논리 삭제한다.
2. 관련 memory를 `deleted` 상태로 변경한다.
3. 해당 memory를 인용한 conversation message를 숨긴다.
4. 해당 인용을 가진 autobiography를 `deleted`로 변경한다.
5. 관련 Chroma vector를 삭제한다.
6. 활성 SQLite memory로 BM25를 다시 만든다.

타임라인은 요청 시 활성 기억에서 계산되므로 삭제된 기억은 다음 조회부터
표시되지 않는다. 인덱스 정리가 실패해도 이미 commit된 SQLite가 기준
상태이며 API는 `503`으로 후속 정리 실패를 알린다.

## 7. Raw 원본과 보존 한계

raw transcript 파일은 불변이며 API가 자동 삭제하지 않는다. 삭제 결과의
`raw_file_deleted`는 항상 `false`다. 사용자가 raw 원본을 물리 삭제하려면
정확한 파일을 직접 확인해 별도로 제거해야 하며 이는 애플리케이션 workflow
밖의 작업이다.

MVP 삭제는 soft deletion이다. 삭제된 SQLite row는 추적성을 위해 로컬에
남지만 일반 조회, 검색, 타임라인, 대화와 자서전에서 제외된다. 보존 기간,
관리자 hard delete, 백업 삭제, 암호화와 다중 사용자 권한은 MVP 범위 밖이다.
