# 포트폴리오·데모 가이드

## 한 줄 소개

Life Archive AI는 STT 텍스트를 구조화된 기억으로 저장하고, 각 답변을 원문 offset과 함께 검증하는 Memory-Centric RAG MVP입니다.

## 보여줄 핵심 문제와 해결

| 문제 | 구현한 해결책 |
|---|---|
| 긴 기록에서 사건을 다시 찾기 어려움 | 기억 추출, chunking, BM25·dense·MMR·hybrid 검색 |
| LLM이 사실을 만들어낼 위험 | 검색된 memory만 사용하고 citation/offset 검증 |
| 원문과 파생 결과의 불일치 | SQLite를 단일 진실 공급원으로 사용하고 Chroma/BM25는 재생성 가능한 index로 분리 |
| 정정·삭제 후 이전 결과 노출 | superseded/deleted 상태와 index 동기화 |
| 개인 기록 유출 위험 | immutable raw 파일, 안전한 오류 응답, 로컬 단일 사용자 범위 |

## 3분 데모 순서

1. `docs/OPERATIONS.md`의 명령으로 backend와 Streamlit을 실행합니다.
2. 샘플 TXT를 업로드하고 처리 결과와 citation을 확인합니다.
3. Chat에서 기록에 존재하는 질문과 존재하지 않는 질문을 각각 입력합니다.
4. 존재하는 질문은 답변·memory ID·원문 offset을 보여주고, 근거가 부족한 질문은 답변을 거절하는지 확인합니다.
5. Timeline에서 부분 날짜와 미상 날짜가 분리되는지 확인합니다.
6. Autobiography에서 여러 chapter와 chapter별 supporting memory를 확인합니다.
7. 필요하면 transcript 삭제 후 검색·timeline·autobiography에서 파생 결과가 사라지는지 시연합니다.

## 평가 결과를 설명하는 방법

- `scripts/run_evaluation.py`는 API 비용이 없는 결정적 회귀 평가입니다.
- `scripts/run_real_evaluation.py`는 합성 데이터에 실제 임베딩 모델을 연결하는 선택적 평가입니다.
- 실제 평가는 `text-embedding-3-small`, chunk 256/512/1024/oracle, dense/MMR/BM25/hybrid, top-k 3/5/10을 비교합니다.
- Recall@K는 검색된 정답 memory 비율이며, citation 검증 결과와 동일한 지표가 아닙니다.
- 작은 합성 데이터셋 결과를 운영 정확도로 포장하지 않고, 설정 선택과 실패 양상을 비교하는 자료로 제시합니다.

## 기술적 차별점

- QA와 autobiography만 LangGraph workflow로 구성하고, CRUD·chunking·timeline은 일반 서비스로 단순하게 유지했습니다.
- 원문을 수정하지 않고 source offset을 끝까지 보존합니다.
- 검증 실패 시 한 번만 재작성하고, 두 번째 실패는 안전한 거절로 처리합니다.
- 실제 평가 러너는 같은 chunk 전략의 query embedding을 검색 방식 사이에서 재사용해 비교 비용을 줄입니다.

## 프로젝트 요약

> “문서 요약기가 아니라 기억 저장소를 만들었습니다. SQLite를 source of truth로 두고, 검색은 파생 index로 제한했습니다. 생성된 답변은 memory와 transcript offset으로 검증하고, 근거가 없으면 거절합니다. 결정적 평가와 실제 임베딩 평가를 분리해 품질·비용·지연을 각각 확인할 수 있게 했습니다.”

## 명시할 한계

이 버전은 로컬 단일 사용자 MVP이며, STT·OCR·인증·다중 사용자 권한·클라우드 배포·백그라운드 작업 큐는 포함하지 않습니다. 실제 임베딩 평가는 합성 데이터만 사용하며, 개인 transcript의 운영 성능을 의미하지 않습니다.
