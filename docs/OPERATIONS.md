# 로컬 운영 런북

## 환경 설정

1. `.env.example`을 복사해 `.env`를 만듭니다.
2. 로컬 `.env` 파일에만 `OPENAI_API_KEY`를 설정합니다.
3. API 키를 이슈, 로그, 스크린샷 또는 커밋에 포함하지 않습니다.

선택형 실제 임베딩 평가 스크립트는 키를 확인하기 전에 프로젝트 루트의
`.env`를 자동으로 불러옵니다.

## MVP 실행

PowerShell 창을 두 개 열고 각각 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

백엔드 상태는 `http://127.0.0.1:8000/api/v1/health`에서 확인하고, UI는
`http://localhost:8501`에서 엽니다.

## 데이터 및 복구

- `data/raw/transcripts/`: 변경하지 않는 원본 업로드 파일
- `data/db/`: 유일한 원본 데이터 저장소인 SQLite
- `data/indexes/`: 재생성 가능한 ChromaDB 및 BM25 인덱스
- 개인정보 보호 API로 전사록을 삭제하면 파생 데이터도 무효화됩니다.
- 인덱스가 손상되면 SQLite에서 다시 생성합니다. ChromaDB만으로 데이터를
  복구하지 않습니다.

## 데모 전 검증

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

실제 임베딩 평가는 선택 사항이며 API 비용이 발생할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_real_evaluation.py
```

일반적인 회귀 검증에는 비용이 없는 결정적 평가를 사용합니다. 임베딩 모델을
비교할 때만 실제 평가를 실행하고, 결과 manifest와 CSV를 함께 확인합니다.

## 릴리스 범위

이 MVP는 로컬 단일 사용자 애플리케이션입니다. 인증, 다중 사용자 권한,
요청 제한, 백그라운드 작업, 클라우드 배포, 원본 파일 자동 보존 기간 삭제는
1주 MVP 범위에 포함하지 않습니다.
