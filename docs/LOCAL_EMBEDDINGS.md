# BGE-M3 로컬 임베딩

`.env`의 `EMBEDDING_PROVIDER=ollama`, `OLLAMA_EMBEDDING_MODEL=bge-m3`로 선택한다.
Ollama에 `ollama pull bge-m3`로 모델을 설치하고 백엔드를 재시작한다.
질문과 기억 임베딩은 Ollama `/api/embed`를 사용하며 외부 API로 자동 대체하지 않는다.
응답 생성 모델은 기존 Gemma 설정을 유지한다.

기존 `CHROMA_PERSIST_DIRECTORY`는 OpenAI 인덱스용으로 보존한다.
로컬 인덱스는 같은 부모 폴더에 `기존이름-ollama-모델명해시`로 분리한다.
질문 서비스를 처음 초기화할 때 SQLite의 현재 기억을 새 인덱스로 동기화한다.
TXT/PDF를 다시 업로드할 필요는 없다. 처음 인덱싱과 모델 로딩에는 추가 시간이 필요하다.
업로드·대화·기억 수정·자서전·삭제 서비스는 같은 공급자 설정을 사용한다.
QA 캐시 키에도 공급자와 모델·인덱스 경로를 반영한다.

`EMBEDDING_PROVIDER=openai`로 되돌리고 백엔드를 재시작하면 기존 인덱스를 사용한다.
기존 인덱스는 비교용 데이터가 남으므로 개인정보 삭제 정책에도 함께 포함해야 한다.
SQLite가 원본이며 삭제된 기억은 과거 벡터가 남아 있어도 검색 결과에서 제외한다.

여기서는 BGE-M3의 dense 벡터와 기존 BM25를 결합한다.
BGE-M3의 sparse·ColBERT 기능은 이번 변경에 포함하지 않는다.
속도와 한국어 검색 품질은 동일 질문으로 별도 평가해야 한다.

공식 자료: https://ollama.com/library/bge-m3, https://docs.ollama.com/api/embed
