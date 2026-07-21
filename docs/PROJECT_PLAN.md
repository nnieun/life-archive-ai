# Life Archive AI

## 1. Project Overview

Life Archive AI는 사용자의 생애 기록을 장기 기억(Long-term Memory)으로 저장하고,
관련 기억을 검색하여 질문에 답변하며,
타임라인과 자서전을 생성하는 Memory-Centric RAG 시스템이다.

본 프로젝트는 외부 STT 시스템으로 변환된 TXT 파일을 입력으로 사용한다.

AI는 단순히 문서를 검색하는 것이 아니라,
사용자의 기억을 구조화하고,
관련 기억만을 근거로 응답을 생성한다.

---

# 2. Project Goal

본 프로젝트의 목표는 다음과 같다.

- STT 기록을 장기 기억으로 저장
- Hybrid Search를 이용한 기억 검색
- 근거 기반 Q&A
- Timeline 생성
- 자서전 초안 생성
- 모든 응답에 출처(Citation) 제공

---

# 3. Motivation

일반적인 RAG 시스템은 PDF나 문서를 검색한다.

하지만 사람의 기억은

- 시간
- 장소
- 사람
- 감정
- 사건

등이 함께 저장된다.

Life Archive AI는 이러한 기억을 구조화하여
장기 기억 저장소를 구축하는 것을 목표로 한다.

---

# 4. Scope

## Included

- STT TXT Upload
- Memory Extraction
- Chunking
- Embedding
- ChromaDB
- BM25
- Hybrid Search
- Grounded QA
- Timeline
- Autobiography

## Excluded

- Speech Recognition
- Whisper
- OCR
- Image Analysis
- Video Analysis
- Email Import
- Calendar Import
- Multi-user
- Authentication

---

# 5. Technology Stack

Language

- Python 3.13

Backend

- FastAPI

Frontend

- Streamlit

AI

- OpenAI API
- LangChain
- LangGraph

Retrieval

- ChromaDB
- BM25

Database

- SQLite

Validation

- Pydantic v2

Testing

- pytest

---

# 6. System Flow

User

↓

Upload TXT

↓

Memory Extraction

↓

Chunking

↓

Embedding

↓

SQLite

+

ChromaDB

↓

Hybrid Retrieval

↓

LangGraph

↓

Grounded Answer

↓

Timeline

↓

Autobiography

---

# 7. Core Features

## Memory Ingestion

TXT 업로드

↓

구조화 기억 추출

↓

저장

---

## Retrieval

Hybrid Search

- Chroma Similarity
- BM25

---

## QA

관련 기억 검색

↓

답변 생성

↓

근거 검증

↓

출처 제공

---

## Timeline

사건을 시간순으로 정렬하여 출력

---

## Autobiography

관련 기억 검색

↓

목차 생성

↓

챕터 생성

↓

근거 검증

↓

자서전 초안 생성

---

# 8. Expected Outcomes

- Memory-Centric RAG 구현
- Grounded QA
- Timeline 생성
- 자서전 생성
- LangGraph 활용
- Hybrid Search 비교
- RAG 포트폴리오 완성

---

# 9. Development Schedule

Week 1

TASK-001
Project Setup

TASK-002
Memory Ingestion

TASK-003
Chunking

TASK-004
Embedding

TASK-005
Hybrid Retrieval

TASK-006
Grounded QA

TASK-007
Timeline

TASK-008
Autobiography

TASK-009
Streamlit UI

TASK-010
Testing & Evaluation

---

# 10. Success Criteria

프로젝트가 성공했다고 판단하는 기준

- TXT 업로드 가능
- Memory 저장 가능
- Hybrid Search 가능
- 출처 기반 답변 생성
- Timeline 생성
- 자서전 생성
- GitHub 공개 가능