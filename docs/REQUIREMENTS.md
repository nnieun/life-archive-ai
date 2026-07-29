# Life Archive AI 요구사항

## 1. 목적

Life Archive AI는 STT TXT를 구조화된 장기 기억으로 변환하고, 저장된
기억만 검색하여 출처가 있는 답변·타임라인·자서전을 생성해야 한다.

## 2. 기능 요구사항

### FR-01 기록 수집

- UTF-8 `.txt` 파일만 받는다.
- 파일명과 내용 해시 중복을 거부한다.
- raw 원본은 생성 후 덮어쓰거나 자동 수정하지 않는다.
- 정규화된 본문과 raw 본문을 분리해 SQLite에 저장한다.

### FR-02 청킹과 기억 추출

- 세그먼트는 정규화 본문의 반열림 구간 `[start_offset, end_offset)`을
  유지해야 한다.
- 기억은 `memory_id`, `transcript_id`, 제목, 요약, 인물, 장소, 사건
  날짜, 날짜 정밀도, 감정, 신뢰도와 불확실성을 포함해야 한다.
- 기억은 하나 이상의 `memory_sources` 출처로 원문 위치에 연결되어야 한다.
- 모델 출력은 Pydantic Structured Output으로 검증하며 이름·날짜·대화를
  임의로 만들지 않는다.

### FR-03 저장과 인덱스

- SQLite를 유일한 기준 저장소로 사용한다.
- Chroma에는 검색용 텍스트, 임베딩과 최소 재생성 메타데이터만 저장한다.
- BM25는 활성 SQLite 기억으로 다시 만들 수 있어야 한다.
- 인덱싱은 동일 내용과 버전에서 멱등이어야 한다.

### FR-04 검색

- Chroma Similarity, MMR, BM25와 Hybrid RRF를 지원한다.
- Hybrid 결과는 중복 `memory_id`를 제거하고 Top-K를 적용한다.
- 모든 검색 결과는 SQLite에서 다시 읽어 삭제·오래된 레코드를 거부한다.

### FR-05 근거 기반 QA

- 검색 → 근거 충분성 판단 → 생성 → 검증 → 최대 1회 재작성 흐름을
  LangGraph로 실행한다.
- 검색 근거가 부족하면 답변을 거절한다.
- 각 생성 주장에는 선택된 `memory_id`가 하나 이상 있어야 한다.
- 최종 답변에 `transcript_id`와 원문 offset을 포함한 인용을 반환한다.
- 최종 검증에 실패한 초안은 사용자에게 반환하지 않는다.

### FR-06 타임라인

- LLM 없이 SQLite 기억을 날짜순으로 정렬한다.
- 부분 날짜의 정밀도를 보존하고 누락된 날짜 부분을 채우지 않는다.
- 해석할 수 없는 날짜는 `undated_events`로 분리한다.
- 선택적 시작일·종료일 필터와 출처를 지원한다.

### FR-07 자서전

- 검색 → 타임라인 → 1~3장 계획 → 장 작성 → 검증 → 최대 1회 수정
  흐름을 LangGraph로 실행한다.
- 검증된 장만 SQLite draft에 저장한다.
- 모든 요청 장이 통과했을 때만 `completed` 상태로 변경한다.
- 각 장에 SQLite 기반 출처를 포함한다.

### FR-08 Streamlit UI

- 업로드, 기억, 대화, 타임라인, 자서전의 5개 화면을 제공한다.
- UI는 typed HTTP client를 통해서만 FastAPI와 통신한다.
- 중복 업로드와 상태별 장애를 사용자가 이해할 수 있는 문구로 표시한다.
- 내부 예외 대신 요청 ID만 문제 추적 정보로 표시한다.

### FR-09 개인정보 보호 삭제

- `DELETE /api/v1/transcripts/{transcript_id}`로 관련 SQLite 레코드를
  논리 삭제한다.
- 관련 대화와 자서전을 무효화하고 Chroma를 정리하며 BM25를 재구성한다.
- raw 원본은 API가 자동 삭제하지 않는다.

## 3. 비기능 요구사항

### 보안·개인정보

- transcript는 신뢰하지 않는 데이터이며 내부 지시로 실행하지 않는다.
- API 키, Bearer 토큰, 원문, 질문, 개인정보와 로컬 경로를 로그에 남기지
  않는다.
- `.env`, 개인 raw 파일, SQLite DB, Chroma 인덱스와 개인 export를
  커밋하지 않는다.
- 모든 오류 응답은 공통 envelope와 `request_id`를 포함한다.

### 신뢰성

- SQLite 외 저장소가 없어져도 인덱스를 재생성할 수 있어야 한다.
- 데이터베이스 작업은 트랜잭션과 foreign key 검사를 사용한다.
- 신규 기능은 실제 OpenAI 호출 없는 테스트를 포함해야 한다.

### 환경

- Python `>=3.13,<3.14`
- Windows PowerShell
- FastAPI, Streamlit, LangChain v1, LangGraph, ChromaDB, SQLite,
  Pydantic v2, pytest

## 4. 평가 요구사항

- 청크: 256, 512, 1024자, 이벤트 인식
- 검색: Dense, MMR, BM25, Hybrid RRF
- Top-K: 3, 5, 10
- 지표: Recall@K, 인용 정확도, 미지원 답변 비율, 검색·E2E 지연
- 합성 데이터와 결정적 가짜 모델을 사용해 개인정보와 외부 호출을 배제한다.

## 5. 인수 조건

- 전체 기능과 API가 문서 및 현재 폴더 구조와 일치한다.
- 전체 pytest 회귀 테스트가 통과한다.
- README 설치·실행·평가 명령이 PowerShell에서 실행 가능하다.
- 답변, 타임라인과 자서전에서 출처를 추적할 수 있다.
- Git에 비밀값, 실제 개인정보와 로컬 생성 데이터가 없다.
