# 임베딩 비교

현재 앱의 SQLite와 Chroma를 열거나 수정하지 않고, 기존 합성 평가 데이터의
기억을 평가용 SQLite에 저장하고 OpenAI/Ollama 인덱스를 각각 생성합니다.
생성 모델과 실제 앱의 임베딩 설정은 변경하지 않습니다.

```powershell
ollama pull embeddinggemma
.\.venv-review\Scripts\python.exe scripts/compare_embeddings.py
# 답변 생성과 근거 검증까지 비교
.\.venv-review\Scripts\python.exe scripts/compare_embeddings.py --answers
# 짧은 답변 비교 (첫 질문만)
.\.venv-review\Scripts\python.exe scripts/compare_embeddings.py --answers --query-limit 1
```

OpenAI 비교군은 `.env`의 `OPENAI_EMBEDDING_MODEL`과 API 키를 사용하며
합성 기록과 질문에 대한 유료 임베딩 호출이 발생합니다. 로컬 비교군은
`embeddinggemma`를 사용합니다. 답변은 두 비교군 모두 같은 로컬 생성 모델을
사용합니다. 한 번의 답변 생성은 실제 QA 그래프 전체 평가를 대체하지 않습니다.

결과는 `data/processed/embedding_comparison/<실행별 폴더>/results.json`에
저장합니다. 각 실행은 새 폴더를 사용하고 이전 결과는 덮어쓰지 않습니다.

- Recall@K: 정답 기억 중 검색한 비율
- Precision@K: 검색한 기억 중 정답 기억 비율
- MRR: 첫 정답 기억 순위의 역수
- citation_in_retrieved_rate: 인용 ID가 검색된 기억에 속하는 비율
- citation_gold_match_rate: 인용 ID가 정답 기억에 속하는 비율
- llm_verifier_passed: 같은 로컬 LLM의 근거 검증 판정. 객관적 사실 판정이 아님
- index_build_ms: 최초 문서 임베딩과 인덱스 저장 시간
- query_embedding_ms / retrieval_ms: 질문 임베딩 / 검색 시간
- answer_and_verification_ms: 답변 생성과 검증 시간

검색 시간은 첫 실행과 이후 실행을 `repeat`로 구분합니다. 모델 로딩과 API
네트워크 상태에 따라 변동하므로 반복 실행해 비교하세요. 답변은 반복 0에서만
생성합니다. 실제 한국어 개인 기록 성능을 대표하려면 별도의 정답 라벨 데이터가
필요합니다. 인용 ID 적합률만으로 문장 전체가 근거 있다고 판단할 수 없습니다.

## 최초 검색 실험

2026-10-04, 합성 기억 12개·질문 8개·Top-K 3·반복 3회 기준입니다.

| 모델 | 검색 | Recall@3 | MRR | 평균 검색 시간 |
|---|---|---:|---:|---:|
| text-embedding-3-small | Dense | 1.000 | 0.875 | 271ms |
| text-embedding-3-small | Hybrid | 1.000 | 0.854 | 272ms |
| embeddinggemma | Dense | 1.000 | 1.000 | 2096ms |
| embeddinggemma | Hybrid | 1.000 | 0.938 | 2096ms |

검색과 로컬 답변 생성이 동시에 실행된 측정입니다. GPU 가속 없이 CPU로
Gemma가 실행됐으며, 속도는 독립 실행으로 재측정해야 합니다. 작은 합성
데이터의 결과만으로 개인 기록에서 로컬 모델이 더 정확하다고 결론내릴 수 없습니다.
검색 결과 파일: `data/processed/embedding_comparison/20261004-184148-817233/results.json`.
