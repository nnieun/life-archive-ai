# unsupported_number 오거절 수정 기록 (2026-10-05)

## 증상
채팅에서 답변이 나오지 않고 아래 메시지만 표시됨 (처리 15.50초).

> 답변 검증 단계: 답변의 숫자나 날짜가 인용한 기억에서 확인되지 않았습니다. / 오류 유형: unsupported_number

## 원인
`qa_failure_diagnostics`의 최근 기록:

- reason: `Answer claim used the number 790 that its cited memories do not contain`
- 검색된 기억 3건은 정상 (`mem_790be3ff…`, `mem_e4f2b04d…`, `mem_6d9b6a4e…`)
- `790`은 메모리 ID `mem_790be3ff…`의 앞부분과 일치

모델이 claim 본문에 메모리 ID를 적은 것으로 추정된다(본문은 진단에 저장되지 않아 직접 확인하지는 못함). 숫자 검증(`_grounding_error`)이 본문의 모든 숫자열(`\d+`)을 근거 텍스트와 대조하므로, ID 속 숫자가 "근거 없는 숫자"로 판정되어 답변 전체가 거절됐다.

## 수정 전후 (`backend/app/services/qa.py`)

| | 수정 전 | 수정 후 |
|---|---|---|
| 숫자 검사 | `_DIGIT_RUN.findall(claim.text)` | `_DIGIT_RUN.findall(_strip_memory_ids(claim.text))` |
| 화면 답변 본문 | `claim.text.strip()` 그대로 | `_strip_memory_ids(claim.text)` (ID 제거, 출처는 뒤의 인용 표기가 담당) |
| 본문 `mem_790be3…` | 숫자 790·1768·89749 등이 검증 대상 | 검증 대상에서 제외 |
| 진짜 숫자 (`1967년`) | 검증함 | 그대로 검증함 (변화 없음) |

추가된 코드:

```python
_MEMORY_ID_TOKEN = re.compile(r"\[?\bmem_[0-9A-Za-z]+\b[^\s\]]*\]?")

def _strip_memory_ids(text: str) -> str:
    return " ".join(_MEMORY_ID_TOKEN.sub(" ", text).split())
```

## 수정 중 발생한 실수
처음 정규식을 스크립트로 쓰면서 `\b`가 백스페이스 문자(0x08)로 저장되어 패턴이 아무것도 매칭하지 못했다. 회귀 테스트가 실패해서 발견했고, 해당 문자를 제거해 고쳤다.

## 검증
- 회귀 테스트 `test_memory_id_digits_in_claim_text_are_not_unsupported_numbers` 추가: ID가 섞인 본문에서 숫자가 검출되지 않고, `1967년` 같은 일반 숫자는 보존됨을 확인
- 전체 테스트 295개 통과
- 사용자가 같은 질문을 다시 보내 정상 답변이 나오는 것을 확인

## 한계
- 이 수정은 "ID 속 숫자" 오탐만 해결한다. 모델이 인용 기억에 없는 날짜·숫자를 실제로 지어내면 `unsupported_number` 거절은 여전히 나오며, 이는 의도된 동작이다.
- 거절 진단에 claim 본문이 저장되지 않아 원인 추정이 필요했다. 필요하면 진단에 거절된 claim 텍스트를 남기도록 개선할 수 있다.
