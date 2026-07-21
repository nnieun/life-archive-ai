# 기억함 (Life Archive AI)

> Memory-Centric Retrieval-Augmented Generation (RAG)

기억함 (Life Archive AI)는 STT(음성→텍스트)로 변환된 생애 기록을 장기 기억(Long-term Memory)으로 저장하고, 관련 기억만을 검색하여 질문에 답변하며, 타임라인과 자서전을 생성하는 AI 시스템입니다.

---

# Features

- STT TXT Upload
- Structured Memory Extraction
- ChromaDB Embedding
- BM25 + Semantic Hybrid Search
- Grounded Question Answering
- Timeline Generation
- Autobiography Generation
- Citation-based Answers

---

# Architecture

```text
TXT Upload

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

Grounded QA

↓

Timeline

↓

Autobiography
```

---

# Technology Stack

## Language

- Python 3.13

## Backend

- FastAPI

## Frontend

- Streamlit

## AI

- OpenAI API
- LangChain
- LangGraph

## Database

- SQLite

## Vector Store

- ChromaDB

## Retrieval

- BM25
- Similarity Search
- MMR

## Validation

- Pydantic v2

## Testing

- pytest

---

# Project Structure

```text
life-archive-ai/

AGENTS.md
README.md
pyproject.toml

backend/
frontend/
data/
docs/
scripts/
tests/
```

---

# Development Roadmap

- Project Setup
- SQLite Database
- Transcript Loader
- Chunking
- Embedding
- Memory Extraction
- Hybrid Retrieval
- Grounded QA
- Timeline
- Autobiography
- Streamlit UI
- Evaluation

---

# Running

## Backend

```bash
uvicorn backend.app.main:app --reload
```

## Frontend

```bash
streamlit run frontend/app.py
```

---

# Evaluation

The project compares:

- Chunk Size
- Similarity Search
- MMR
- Hybrid Search
- Top-K

Evaluation metrics

- Retrieval Accuracy
- Citation Accuracy
- Response Time

---

# Future Work

- Voice Upload
- Photo Memory
- Relationship Graph
- Knowledge Graph
- Multi-user
- Cloud Deployment

---

# License

This project was developed as an educational portfolio project.