# 실제 임베딩 평가 운영 가이드

## 목적

기본 평가는 외부 API 없이 재현 가능한 결정적 평가입니다. 실제 임베딩 평가는 별도 명령으로 실행하며, 합성 데이터만 사용합니다. 개인 transcript는 평가에 포함하지 않습니다.

## 실행

`.env`에 `OPENAI_API_KEY`를 설정한 뒤 PowerShell에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_real_evaluation.py
```

이 명령은 OpenAI 임베딩 API를 호출하므로 비용이 발생할 수 있습니다. 결과는 다음 파일에 저장됩니다.

- `reports/real_retrieval_results.csv`
- `reports/real_evaluation_manifest.json`

생성 결과 파일은 개인 데이터가 아니지만, API 사용량과 실행 시점에 따라 재생성되므로 Git에 커밋하지 않습니다.

## 측정 항목

CSV의 `retrieval_latency_ms`는 문서 임베딩, 질의 임베딩, 검색 랭킹 시간을 합산한 값입니다. 세부 시간은 다음 열로 분리됩니다.

- `document_embedding_latency_ms`: chunk 전략별 문서 임베딩 시간
- `query_embedding_latency_ms`: 질의 임베딩 시간
- `ranking_latency_ms`: dense/MMR/BM25/hybrid 랭킹 시간
- `estimated_input_tokens`: 문자 수를 기준으로 한 보수적 추정치
- `estimated_embedding_cost_usd`: `text-embedding-3-small`의 입력 단가를 적용한 추정 비용

질의 임베딩은 같은 chunk 전략 안에서 검색 방식별로 재사용합니다. 따라서 검색 방식 비교가 임베딩 API 재호출 횟수 때문에 왜곡되지 않습니다. 비용 열은 청구서가 아니라 평가 전 입력량 추정치입니다.

## 해석 규칙

- `recall_at_k`와 `contains_answer`는 합성 데이터의 정답 memory ID 기준입니다.
- `oracle`은 실제 서비스 chunking이 아니라 경계 상한선 비교용입니다.
- `alias`는 결정적 기준선이며 실제 임베딩 검색이 아닙니다.
- 실제 운영 품질을 주장하려면 동일한 데이터셋, chunk, top-k, 모델을 고정하고 반복 실행해야 합니다.
- API 잔액과 사용량은 반영 지연이 있을 수 있으므로 실행 직후 청구 화면만으로 비용을 판단하지 않습니다.

## 현재 결과 해석

현재 합성 데이터는 256 또는 512자 chunk에서 1024자보다 안정적이며, 실제 임베딩 결과에서는 BM25 또는 hybrid를 함께 비교해야 합니다. 소규모 평가의 Recall@K는 방향성을 확인하는 지표이지 운영 정확도 보증이 아닙니다.
