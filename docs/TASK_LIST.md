# Life Archive AI Development Tasks

## Development Strategy

프로젝트를 한 번에 구현하지 않는다.

각 TASK를 순서대로 완료하고, 테스트가 통과한 뒤 다음 TASK로 이동한다.

각 TASK는 다음 조건을 만족해야 한다.

- 구현 완료
- 관련 테스트 통과
- 필요한 문서 업데이트
- 변경 파일 검토
- 커밋 후 다음 TASK 진행

## Git Commit Rules

커밋 전 다음 명령으로 변경 사항을 확인한다.

```powershell
git status
git diff
```

테스트가 통과한 경우에만 커밋한다.

```powershell
git add .
git commit -m "<recommended commit message>"
```

`.env`, API 키, 실제 개인 녹취, SQLite DB 파일, ChromaDB 인덱스는 커밋하지 않는다.

---

# TASK-001 — Project Initialization

## Goal

Life Archive AI의 프로젝트 골격과 기본 문서를 구성한다.

## Requirements

- `AGENTS.md`
- `README.md`
- `.gitignore`
- `backend/`
- `frontend/`
- `docs/`
- `data/`
- `scripts/`
- `tests/`

아래 데이터 폴더도 생성한다.

```text
data/
├─ raw/
│  └─ transcripts/
├─ processed/
├─ db/
├─ indexes/
│  ├─ chroma/
│  └─ bm25/
└─ exports/
```

빈 폴더를 Git에 포함해야 한다면 `.gitkeep`을 사용한다.

## Deliverables

- 프로젝트 폴더 구조
- 기본 문서
- 안전한 `.gitignore`
- Git 저장소 초기화

## Validation

- 폴더와 문서가 설계와 일치하는지 확인
- 민감한 데이터가 Git 추적 대상이 아닌지 확인
- `git status`로 최초 커밋 파일 확인

## Recommended Commit

```powershell
git commit -m "chore: initialize project structure and documentation"
```

---

# TASK-002 — Environment Setup

## Goal

Python 3.13 개발 환경과 최소 의존성을 구성한다.

## Requirements

- Python 3.13
- 가상환경 `.venv`
- 루트 `pyproject.toml`
- `.env.example`

최소 의존성:

- FastAPI
- Uvicorn
- Streamlit
- LangChain Core
- LangGraph
- `langchain-openai`
- `langchain-chroma`
- ChromaDB
- Pydantic v2
- `python-dotenv`
- pytest
- HTTPX
- BM25 구현에 필요한 경량 라이브러리

불필요한 패키지는 추가하지 않는다.

## Deliverables

- 설치 가능한 `pyproject.toml`
- 정상 작동하는 가상환경
- 기본 import smoke test
- Python 3.13 호환성 확인

## Validation

```powershell
python --version
python -c "import fastapi, streamlit, langgraph, chromadb, pydantic"
pytest
```

## Recommended Commit

```powershell
git commit -m "chore: configure Python 3.13 development environment"
```

---

# TASK-003 — Backend and Frontend Skeleton

## Goal

FastAPI와 Streamlit의 최소 실행 골격을 만든다.

## Requirements

FastAPI:

- `backend/app/main.py`
- `GET /api/v1/health`
- 설정 모듈
- 기본 오류 응답

Streamlit:

- `frontend/app.py`
- API 상태 확인 화면
- `frontend/api_client.py`

PowerShell 실행 스크립트:

- `scripts/run_backend.ps1`
- `scripts/run_frontend.ps1`

## Deliverables

- 실행 가능한 FastAPI
- 실행 가능한 Streamlit
- Frontend에서 health API 확인
- health API 테스트

## Validation

```powershell
uvicorn backend.app.main:app --reload
streamlit run frontend/app.py
pytest
```

## Recommended Commit

```powershell
git commit -m "feat: add FastAPI and Streamlit application skeletons"
```

---

# TASK-004 — Transcript Dataset Inspection

## Goal

실제 TXT와 `upload_manifest.jsonl` 구조를 확인하고 입력 규칙을 확정한다.

## Requirements

- 파일명 패턴 확인
- 인코딩 확인
- 빈 파일 확인
- 중복 파일 확인
- 평균 및 최대 텍스트 길이 확인
- manifest와 TXT 연결 확인
- 원본 파일 수정 금지

검사 스크립트:

```text
scripts/inspect_data.py
```

검사 결과:

```text
data/processed/dataset_report.json
```

개인정보가 포함된 원문 자체는 보고서에 복사하지 않는다.

## Deliverables

- 데이터 검사 스크립트
- 데이터 구조 보고서
- Loader 입력 규칙
- 데이터 예외 목록

## Validation

- 모든 TXT 파일을 읽을 수 있음
- manifest 참조 오류 확인
- 원본 해시가 변경되지 않음

## Recommended Commit

```powershell
git commit -m "chore: add transcript dataset inspection workflow"
```

---

# TASK-005 — Transcript Loader and Normalization

## Goal

STT TXT 파일을 안전하게 읽고 정규화한다.

## Requirements

- UTF-8 TXT 읽기
- 원본 보존
- 줄바꿈과 공백 정규화
- 지나친 정제 금지
- 파일 해시 생성
- 업로드일과 사건 발생일 분리
- 동일 파일 중복 적재 방지
- `pathlib` 사용

정규화 과정에서 다음 정보는 제거하지 않는다.

- 말 더듬음
- 정정 표현
- 날짜 불확실성
- 인물 별칭
- 반복 기억

## Deliverables

- Transcript Loader
- Normalization 함수
- Pydantic 입력 모델
- 단위 테스트

## Validation

- 정상 TXT 로딩
- 빈 파일 오류 처리
- 잘못된 인코딩 오류 처리
- 원본 파일 불변 확인
- 같은 파일의 중복 감지

## Recommended Commit

```powershell
git commit -m "feat: implement transcript loading and normalization"
```

---

# TASK-006 — SQLite Schema and Storage

## Goal

애플리케이션의 정식 저장소인 SQLite를 구축한다.

## Requirements

최소 테이블:

- `transcripts`
- `transcript_segments`
- `memories`
- `memory_sources`
- `conversation_sessions`
- `conversation_messages`
- `autobiographies`

필수 원칙:

- SQLite가 유일한 Source of Truth
- 외래 키 활성화
- 생성·수정 시간 기록
- 기억 상태 관리
- 삭제 상태 관리
- 사건 발생일과 기록일 분리
- JSON 컬럼은 TEXT로 직렬화하되 Pydantic으로 검증

기억 상태:

- `active`
- `corrected`
- `deleted`

## Deliverables

- DB 초기화 코드
- CRUD 함수
- ERD 업데이트
- 임시 DB 기반 테스트

## Validation

- 테이블 생성
- 외래 키 제약 확인
- Transcript 저장·조회
- Memory 저장·조회·수정
- 삭제된 Memory 필터링
- 테스트 종료 후 임시 DB 정리

## Recommended Commit

```powershell
git commit -m "feat: add SQLite schema and persistence layer"
```

---

# TASK-007 — Chunking Pipeline

## Goal

원문 출처를 추적할 수 있는 Chunking을 구현한다.

## Requirements

기본 구현:

- Fixed-size Chunking
- Chunk Overlap
- 문자 또는 토큰 기준 설정
- `start_offset`
- `end_offset`
- `chunk_index`
- `transcript_id`

비교용 구현:

- 256
- 512
- 1024
- Event-aware Chunking 후보

Chunk가 원문 범위를 정확하게 참조해야 한다.

## Deliverables

- Chunking 모듈
- Chunk 모델
- SQLite segment 저장
- 단위 테스트

## Validation

- 원문 누락 여부
- offset 정확성
- overlap 정확성
- 매우 짧은 문서
- 매우 긴 문서
- 빈 문서

## Recommended Commit

```powershell
git commit -m "feat: implement traceable transcript chunking"
```

---

# TASK-008 — Structured Memory Extraction

## Goal

녹취 Chunk에서 구조화된 기억을 추출한다.

## Requirements

OpenAI 모델의 Structured Output과 Pydantic v2를 사용한다.

최소 출력:

- `title`
- `summary`
- `people`
- `location`
- `event_date`
- `date_precision`
- `emotion`
- `confidence`
- `evidence_start_offset`
- `evidence_end_offset`
- `uncertainty_notes`

날짜 정밀도:

- `exact`
- `day`
- `month`
- `year`
- `approximate`
- `unknown`

추출 결과는 자동으로 사실 확정하지 않는다.

낮은 신뢰도는 그대로 저장한다.

## Deliverables

- Memory Pydantic 모델
- Extraction Prompt
- Structured Output 처리
- SQLite 저장
- Mock LLM 테스트

## Validation

- 정상 구조화 결과
- 날짜 없음
- 인물 없음
- 날짜 충돌
- 모델 출력 검증 실패
- 근거 offset 범위 확인

## Recommended Commit

```powershell
git commit -m "feat: extract structured memories from transcripts"
```

---

# TASK-009 — ChromaDB Vector Index

## Goal

기억 검색을 위한 벡터 인덱스를 구축한다.

## Requirements

- OpenAI Embedding
- ChromaDB Persistent Client
- Memory 단위 색인
- `memory_id`를 Chroma ID와 연결
- 최소 metadata만 저장
- `embedding_version`
- `content_hash`
- 삭제·재색인 지원

ChromaDB는 정식 데이터 저장소가 아니다.

답변 근거는 Chroma metadata가 아니라 SQLite에서 다시 조회한다.

## Deliverables

- Vector Store 모듈
- Index/Reindex 함수
- Similarity Search
- 임시 Chroma 테스트

## Validation

- Memory 색인
- 검색 결과 ID 조회
- 중복 색인 방지
- 삭제 반영
- Chroma 삭제 후 SQLite에서 재구축 가능

## Recommended Commit

```powershell
git commit -m "feat: add ChromaDB memory vector index"
```

---

# TASK-010 — BM25 and Hybrid Retrieval

## Goal

Dense Search와 Keyword Search를 결합한 Hybrid Retrieval을 구현한다.

## Requirements

- BM25 검색
- Chroma Similarity Search
- Reciprocal Rank Fusion
- 중복 Memory 제거
- Top-K 설정
- 삭제된 Memory 제외
- 선택적으로 MMR 지원

MVP 한국어 검색은 무거운 형태소 분석기 대신 단순 정규화 또는 문자 n-gram을 우선 검토한다.

## Deliverables

- BM25 Index
- Hybrid Retriever
- RRF 구현
- RetrievalHit 모델
- 단위 및 통합 테스트

## Validation

- Dense 결과
- BM25 결과
- RRF 순위 병합
- 동일 ID 중복 제거
- 결과 없음
- Top-K 동작
- 삭제 상태 필터링

## Recommended Commit

```powershell
git commit -m "feat: implement BM25 and hybrid memory retrieval"
```

---

# TASK-011 — Grounded Q&A LangGraph

## Goal

검색된 기억만으로 답변하는 Q&A 그래프를 구현한다.

## Workflow

```text
retrieve
→ evidence_sufficient
   ├─ false → insufficient_answer
   └─ true  → generate_answer
              → verify_answer
                 ├─ pass → finalize
                 └─ fail → rewrite_once
                            → finalize_or_reject
```

## Requirements

- `QAState`
- Retrieval Node
- Evidence Sufficiency Node
- Answer Generation Node
- Citation Verification Node
- 최대 1회 Rewrite
- 출처 부족 시 답변 거부
- 대화 저장
- Prompt Injection 방어

최소 State:

- `session_id`
- `question`
- `retrieved_memory_ids`
- `selected_evidence`
- `draft_answer`
- `citations`
- `validation_result`
- `final_answer`
- `retry_count`
- `error`

## Deliverables

- Q&A LangGraph
- Q&A Prompt
- Verification Prompt
- Rewrite Prompt
- Chat API
- 테스트

## Validation

- 충분한 근거
- 검색 결과 없음
- 불충분한 근거
- 검증 실패 후 재작성
- 재작성 후 실패
- 모든 주요 주장에 citation 존재

## Recommended Commit

```powershell
git commit -m "feat: add grounded question answering graph"
```

---

# TASK-012 — Timeline Service

## Goal

구조화된 기억을 시간순으로 표시한다.

## Requirements

- 일반 Python 서비스로 구현
- LangGraph 사용 금지
- 사건 발생일 기준 정렬
- 날짜 정밀도 표시
- 날짜가 없는 사건 별도 처리
- corrected Memory 우선
- deleted Memory 제외
- Citation 포함
- 기간 필터 지원

## Deliverables

- Timeline Service
- Timeline API
- TimelineEvent 모델
- 테스트

## Validation

- 정확한 날짜 정렬
- 연도만 있는 기억
- approximate 날짜
- unknown 날짜
- corrected 기억
- 동일 날짜 사건
- 삭제된 기억 제외

## Recommended Commit

```powershell
git commit -m "feat: implement chronological memory timeline"
```

---

# TASK-013 — Autobiography LangGraph

## Goal

검색된 기억을 근거로 최대 3장의 짧은 자서전 초안을 생성한다.

## Workflow

```text
analyze_request
→ retrieve_memories
→ build_timeline
→ create_chapter_plan
→ write_chapter
→ verify_chapter
   ├─ pass → next_chapter
   └─ fail → revise_once
→ assemble_autobiography
→ save
```

## Requirements

- `AutobiographyState`
- 최대 3장
- 각 장마다 Citation
- 근거 없는 문장 제거
- 날짜 불확실성 보존
- 자료에 없는 대화·감정 창작 금지
- 장별 생성 결과 저장
- 최종 결과 SQLite 저장

최소 State:

- `request`
- `target_period`
- `target_topics`
- `retrieved_memory_ids`
- `timeline`
- `chapter_plan`
- `current_chapter_index`
- `chapter_drafts`
- `review_result`
- `citations`
- `final_content`
- `retry_count`
- `error`

## Deliverables

- Autobiography LangGraph
- Chapter Plan Prompt
- Chapter Writing Prompt
- Chapter Verification Prompt
- API
- 테스트

## Validation

- 1장 생성
- 3장 생성
- 근거 부족
- 장 검증 실패 후 재작성
- 모든 장 Citation 확인
- 결과 DB 저장·조회

## Recommended Commit

```powershell
git commit -m "feat: generate grounded autobiography drafts"
```

---

# TASK-014 — Streamlit MVP Interface

## Goal

핵심 기능을 하나의 Streamlit UI에서 사용할 수 있게 한다.

## Pages

- Upload
- Memories
- Chat
- Timeline
- Autobiography

## Requirements

Upload:

- TXT 업로드
- 처리 상태
- 적재 결과

Memories:

- 구조화 기억 확인
- 신뢰도 확인
- 출처 확인

Chat:

- 질문 입력
- 답변 출력
- Citation 표시

Timeline:

- 날짜순 사건
- 날짜 불확실성 표시

Autobiography:

- 기간·주제 입력
- 생성 요청
- 최대 3장 결과
- 장별 출처 표시

UI에서 비즈니스 로직을 직접 실행하지 않고 FastAPI를 호출한다.

## Deliverables

- Streamlit UI
- API Client
- 사용자 오류 메시지
- 핵심 화면 통합

## Validation

- Backend 연결 실패
- TXT 업로드
- Chat 질의
- Timeline 조회
- Autobiography 생성
- Citation 표시

## Recommended Commit

```powershell
git commit -m "feat: build Streamlit MVP interface"
```

---

# TASK-015 — Retrieval Evaluation

## Goal

검색 설정에 따른 품질 차이를 비교한다.

## Experiments

Chunk Size:

- 256
- 512
- 1024
- Event-aware

Search:

- Dense Similarity
- MMR
- BM25
- Hybrid

Top-K:

- 3
- 5
- 10

## Measures

- Recall@K 또는 정답 Memory 포함 여부
- Citation correctness
- Unsupported answer rate
- Retrieval latency
- End-to-end response time

## Outputs

```text
reports/
├─ retrieval_results.csv
├─ generation_results.csv
└─ experiment_summary.md
```

실험 데이터는 실제 개인 정보가 아닌 평가용 데이터셋을 사용한다.

## Deliverables

- 평가 질의셋
- 평가 스크립트
- CSV 결과
- 비교 요약

## Validation

- 동일 데이터셋·질의로 조건 비교
- 실험 설정 기록
- 재현 가능한 결과
- 실패 결과도 삭제하지 않음

## Recommended Commit

```powershell
git commit -m "test: add retrieval and generation evaluation"
```

---

# TASK-016 — Test Suite Hardening

## Goal

핵심 기능의 회귀 오류를 방지한다.

## Requirements

Unit Tests:

- normalization
- chunking
- offsets
- date precision
- memory validation
- RRF
- timeline sorting
- citation checking

Integration Tests:

- TXT 적재
- SQLite 저장
- Chroma 검색
- Hybrid Retrieval
- Q&A Graph
- Timeline
- Autobiography

실제 OpenAI 호출은 기본 pytest에서 제외한다.

## Deliverables

- 안정적인 pytest suite
- 테스트 fixture
- Mock LLM 응답
- 핵심 실패 시나리오

## Validation

```powershell
pytest
```

모든 테스트가 통과해야 한다.

## Recommended Commit

```powershell
git commit -m "test: strengthen unit and integration coverage"
```

---

# TASK-017 — Privacy and Deletion Workflow

## Goal

개인 녹취의 저장·삭제 규칙을 구현한다.

## Requirements

- `.env` 및 비밀키 보호
- 실제 개인 녹취 Git 제외
- 로그에 원문·API 키 기록 금지
- Transcript 삭제
- 관련 Segment 삭제
- 관련 Memory 삭제
- Chroma ID 삭제
- BM25 재구축
- 파생 Timeline·Autobiography 무효화
- 원본 자동 삭제 금지 정책 명시

## Deliverables

- 삭제 서비스
- 삭제 API
- Privacy 문서
- 관련 테스트

## Validation

- 삭제된 기억 검색 제외
- Chroma 잔여 데이터 없음
- BM25 잔여 데이터 없음
- 파생 결과 invalidation
- 실제 원문 로그 미노출

## Recommended Commit

```powershell
git commit -m "feat: add privacy-safe deletion workflow"
```

---

# TASK-018 — Error Handling and Logging

## Goal

사용자와 개발자가 오류 원인을 파악할 수 있도록 한다.

## Requirements

- 공통 오류 응답
- 구조화 로그
- API Key 마스킹
- 원문 개인정보 로그 금지
- OpenAI 오류 처리
- Chroma 오류 처리
- SQLite 오류 처리
- Frontend 사용자 메시지
- 요청 추적 ID

## Deliverables

- 공통 예외 처리
- 안전한 로깅
- 실패 시나리오 테스트

## Validation

- 잘못된 파일
- DB 실패
- API 인증 실패
- 검색 인덱스 없음
- 모델 출력 검증 실패
- 사용자에게 내부 경로가 노출되지 않음

## Recommended Commit

```powershell
git commit -m "refactor: improve error handling and safe logging"
```

---

# TASK-019 — Final Documentation

## Goal

GitHub 포트폴리오용 문서를 완성한다.

## Requirements

업데이트:

- `README.md`
- `docs/PROJECT_PLAN.md`
- `docs/REQUIREMENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/DATA_MODEL.md`
- `docs/ERD.md`
- `docs/TASK_LIST.md`
- `docs/API_SPEC.md`
- `docs/PRIVACY.md`

README 포함 항목:

- 프로젝트 배경
- 핵심 문제
- 주요 기능
- 아키텍처
- 기술 스택
- 실행 방법
- 평가 결과
- 화면 이미지
- 한계와 향후 개선

## Deliverables

- 최신 문서
- Mermaid 다이어그램
- 실행 가능한 설치 안내
- 프로젝트 스크린샷

## Validation

- README 명령으로 새 환경에서 실행 가능
- 문서와 실제 API·폴더 구조가 일치
- 깨진 링크 없음
- 실제 개인정보 없음

## Recommended Commit

```powershell
git commit -m "docs: finalize portfolio documentation"
```

---

# TASK-020 — Release Preparation

## Goal

MVP를 재현 가능한 최종 버전으로 정리한다.

## Requirements

- 전체 테스트 통과
- Git status clean
- `.gitignore` 최종 확인
- 새 가상환경 설치 테스트
- Backend/Frontend 실행 테스트
- 샘플 데이터 실행
- 평가 결과 포함
- 버전 `0.1.0`
- Git tag 준비

## Deliverables

- 재현 가능한 MVP
- 최종 GitHub 상태
- Release Notes
- 태그 `v0.1.0`

## Validation

```powershell
git status
pytest
uvicorn backend.app.main:app
streamlit run frontend/app.py
```

최종 커밋 후 태그를 생성한다.

## Recommended Commit

```powershell
git commit -m "release: prepare Life Archive AI MVP v0.1.0"
```

## Recommended Tag

```powershell
git tag -a v0.1.0 -m "Life Archive AI MVP v0.1.0"
git push origin main
git push origin v0.1.0
```

---

# Completion Criteria

MVP는 다음 조건을 모두 만족해야 한다.

- TXT 업로드가 작동한다.
- 원본 TXT가 수정되지 않는다.
- SQLite 저장이 작동한다.
- 구조화 Memory 추출이 작동한다.
- ChromaDB 검색이 작동한다.
- BM25 검색이 작동한다.
- Hybrid Retrieval이 작동한다.
- Grounded Q&A가 출처와 함께 작동한다.
- Timeline이 날짜 불확실성을 보존한다.
- 최대 3장 자서전이 근거와 함께 생성된다.
- Streamlit UI에서 핵심 기능을 사용할 수 있다.
- 삭제된 기억이 검색 결과에서 제외된다.
- 모든 핵심 pytest가 통과한다.
- GitHub에 비밀키와 개인 데이터가 포함되지 않는다.
- 문서가 실제 구현과 일치한다.
- `v0.1.0` 태그를 생성할 수 있다.
