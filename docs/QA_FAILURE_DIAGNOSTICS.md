# 실패한 QA 진단 기록

성능 측정과 상세 실패 진단은 별도입니다. 상세 진단은 기존처럼 실패할 때만 저장합니다.
속도 비교를 위해 성공/실패의 시간·단계·상태만 `qa_performance`에 기록합니다.
질문·답변·원문은 성능 기록에 넣지 않습니다. `scripts/inspect_qa_performance.py --limit 20`으로 조회합니다.

구조화 출력 실패에는 단계별 `schema_issues`도 남깁니다. 필드 이름,
Pydantic 오류 유형, 알려진 교차 필드 규칙 이름만 저장합니다. 원래 입력값과
예외 메시지는 저장하지 않으며, 알 수 없는 필드 이름은 가립니다.

최종 QA 결과가 근거 부족, 검증 거절, 처리 오류이면 SQLite의
`qa_failure_diagnostics`에 한 건을 저장합니다. 성공한 요청이나 첫 검증에
실패했지만 재작성 후 성공한 요청은 진단 기록을 저장하지 않습니다.
기존 질문·답변 대화 저장은 유지합니다.

저장 항목은 진단 ID, 세션 ID, 기존 질문 메시지 ID, 시간, 제공자·모델,
최종 상태·이유, 검색/선택 기억 ID, 재작성 횟수, 총 처리 시간 및 단계별
검증 결과·처리 시간입니다. 질문과 원문, 답변 초안, 프롬프트, 내부 추론,
API 키는 진단 테이블에 중복 저장하지 않습니다. 모델의 검증 이유는
개인정보가 포함될 수 있는 로컬 진단 데이터로 취급합니다.

조회:

```powershell
.\.venv-review\Scripts\python.exe scripts/inspect_qa_failures.py <session_id>
```

세션 ID는 `/api/v1/chat` 응답의 `session_id` 또는 SQLite의
`conversation_sessions`에서 확인할 수 있습니다. 별도 공개 API는 추가하지
않습니다. 백엔드를 재시작하면 초기화 시 테이블이 자동 생성됩니다.

관련 transcript 삭제 시 검색/선택 기억 ID가 연결된 진단은 실제 삭제됩니다.
질문 메시지 또는 대화 세션 삭제 시 해당 진단도 실제 삭제됩니다.
DB 장애, 서비스 의존성 초기화 실패, 서버 프로세스 종료로 완료하지 못한
요청은 이 테이블에 남길 수 없습니다. 과거 실패를 소급 복원하지 않습니다.

## 오류 구분

단계와 함께 `failure_code`, `exception_type`을 기록합니다. 예외 종류만
저장하며 예외 원문(키, 로컬 경로, 모델 입력이 포함될 수 있음)은 저장하지 않습니다.

| 구분 | 코드 예시 |
|---|---|
| 검색 결과 없음 / 원문 출처 없음 / 근거 부족 | no_memories_found, no_traceable_sources, insufficient_evidence |
| 생성 모델 연결·시간 초과·인증·권한·사용량 제한 | model_connection, model_timeout, model_authentication, model_permission, model_rate_limit |
| 모델 없음·잘못된 요청·서버 오류 | model_not_found, model_request_invalid, model_server_error |
| 검색 임베딩 API 오류 | 위 model 코드에 대응하는 embedding_* 코드 |
| 구조화 응답 형식·파싱·스키마·응답 없음 | output_envelope_invalid, output_parse_failed, output_schema_invalid, output_missing |
| 응답 길이 제한 / 모델 거절 | output_truncated, model_refusal |
| 잘못된 기억 선택·인용·검증 주장 번호 | unknown_memory_selected, citation_outside_evidence, verification_index_invalid |
| 검증 거절 / 원문과 맞지 않는 숫자·인물 / 출처 추적 실패 | verifier_rejected, unsupported_number, unsupported_person, source_untraceable |
| 초안 없음 / 저장소 / 분류되지 않은 오류 | draft_missing, storage_error, internal_error, model_call_failed |

화면에는 실패한 단계와 안전한 한국어 설명, 오류 코드를 표시합니다.
출력 길이 제한은 모델이 finish_reason=length를 반환한 경우에만 구분합니다.
모델이 이유 없이 빈 응답을 주면 output_missing으로 남기며 원인을 추측하지 않습니다.
근거 검증의 자유 서술 이유는 모델의 판정이며 절대적인 사실 판정은 아닙니다.

서비스 초기화·그래프 밖 오류는 QA 진단을 만들기 전에 발생할 수 있습니다.
이때는 `chat_jobs`의 failure_stage, failure_code에 별도 기록합니다.
서버 재시작은 server_restarted, 잘못된 초기 설정은 configuration_invalid입니다.
기존 일반 실패 기록에는 새 코드가 없으며 소급 추정하지 않습니다.
