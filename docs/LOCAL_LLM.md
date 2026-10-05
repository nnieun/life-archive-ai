# Ollama 생성 모델

QA의 기본 흐름은 충분성 판단과 답변 생성을 합치고, 인용된 주장을 별도로 검증합니다.
근거가 부족하면 빈 주장 목록을 반환합니다. 로컬 QA는 Ollama Native Chat API로
내부 추론을 끄고, 출력 상한과 모델 유지 시간을 지정합니다.
인용·원문·숫자·인물·LLM 검증은 유지합니다.

기억 추출, 한국어 표현 정리, 자서전, 기억 빈칸 Agent는 기존 OpenAI 호환
Chat Completions API를 계속 사용합니다. 패키지를 추가할 필요는 없습니다.
[단계별 속도 개선 결과와 설정](QA_SPEED_RESULTS.md)을 참고하세요.

```powershell
ollama pull gemma4:e2b
```

`.env` 설정:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma4:e2b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

설정 변경 후 백엔드를 재시작합니다. Ollama도 실행 중이어야 합니다.
첫 호출은 모델 로딩 시간이 추가됩니다. 로컬 모델 요청은 300초까지 기다리고
자동 네트워크 재시도는 하지 않습니다.

QA는 여러 번의 모델 호출을 순차 실행하므로 프론트엔드는 최대 900초까지
응답을 기다립니다. 대기 시간 초과는 백엔드 연결 실패와 구분해 표시합니다.
대화 화면의 로딩 표시 옆에는 요청을 기다린 경과 시간이 실시간으로 표시됩니다.
검색뿐 아니라 답변 생성과 검증을 포함한 대기 시간이며, 남은 시간 예측은 아닙니다.
최종 실패 요청은 SQLite에 진단 기록을 남깁니다.
[실패 기록 조회](QA_FAILURE_DIAGNOSTICS.md)를 참고하세요.

임베딩은 기존 `OPENAI_EMBEDDING_MODEL`을 유지하므로 검색과 업로드에는
`OPENAI_API_KEY`가 여전히 필요합니다. 기존 SQLite와 Chroma는 변경하지
않습니다. 임베딩까지 로컬로 전환하려면 별도의 인덱스 재구축이 필요합니다.
기억 빈칸 인터넷 검색은 사용자 동의 후 기존 웹 검색 기능을 사용합니다.

OpenAI 생성 모델로 돌아가려면 `LLM_PROVIDER=openai`를 설정하고
`OPENAI_MODEL`, `OPENAI_API_KEY`를 지정합니다.

공식 API 안내: https://docs.ollama.com/api/openai-compatibility
