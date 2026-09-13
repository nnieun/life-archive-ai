# 프로젝트 검토 후 구현 기록

2026-09-13 검토 항목을 다음 문서로 나누어 구현했다.

1. [Python 3.13 실행 환경](01_실행환경.md)
2. [출처 버튼과 삭제 후 화면](02_출처와삭제화면.md)
3. [업로드 임베딩 일괄 처리](03_임베딩일괄처리.md)
4. [오래된 벡터 제외 후 검색 후보 보충](04_검색후보보충.md)
5. [원문 줄 번호 복원](05_원문줄번호.md)
6. [감정과 구조화 필드의 검색 연결](06_감정검색연결.md)
7. [PDF 업로드 지원 검증](07_PDF지원검증.md)
8. [자서전 출력과 사용자 표시](08_자서전출력.md)
9. [기억 추출 증거 문구 공백 복원](09_증거문구공백복원.md)

## 검증 방법

최종 결과: Python 3.13.14에서 **207 passed, 2 warnings (35.83초)**.
경고는 Starlette/AnyIO 및 langchain-community의 사용 중단 예고다.
`git diff --check`도 통과했다. 반복 대화 출처와 잘못된 임베딩 배치 응답 회귀를 포함한다.

실제 API와 사용자 원본을 사용하지 않고 임시 SQLite·Chroma와 모의 모델로 검증한다.
테스트 실행 전에 dotenv 로딩 함수를 차단하여 로컬 `.env`를 읽지 않는다.

```powershell
.\.venv-review\Scripts\python.exe -c "import dotenv; dotenv.load_dotenv=lambda *a, **k: False; import pytest; raise SystemExit(pytest.main(['-q','-o','addopts=','-p','no:cacheprovider','--tb=short']))"
```

## 남아 있는 경계

- 기존 `.venv`는 보존했다. 새 실행 환경의 명령은 01 문서에 있다.
- 실제 API 인증과 과거 503 오류의 운영 재현은 검증하지 않았다.
- 실제 업로드 지연 시간과 비용 절감률을 측정하지 않았다.
- PDF 페이지 매핑은 없으며 추출 텍스트 줄 번호로 명시한다.
- 삭제 시 현재 브라우저 세션만 초기화한다.
