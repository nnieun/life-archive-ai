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

# Development Setup

Python 3.13 and PowerShell are required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
pytest
```

Keep `.env` local and never commit API keys or personal transcript data.

---

# Running

## Backend

```powershell
.\scripts\run_backend.ps1
```

Health check: `GET http://127.0.0.1:8000/api/v1/health`

## Frontend

```powershell
.\scripts\run_frontend.ps1
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

# Dataset Inspection

Inspect transcript structure without copying raw text into the report:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_data.py
```

See `docs/DATA_INPUT_RULES.md` for the accepted loader input contract and the
current dataset exceptions.

## Memory Vector Index

Structured memories can be indexed with OpenAI `text-embedding-3-small` and a
persistent Chroma client. Chroma stores only rebuildable retrieval data; SQLite
remains the source of truth and search hits are reloaded from SQLite.

The local index path defaults to `data/indexes/chroma` and is ignored by Git.
Set `OPENAI_EMBEDDING_MODEL` or `CHROMA_PERSIST_DIRECTORY` in `.env` to override
the defaults. Tests use deterministic fake embeddings and never call OpenAI.

## Hybrid Memory Retrieval

Keyword retrieval uses BM25 over normalized words and character bigrams so the
MVP can search Korean text without a separate morphological analyzer. Hybrid
retrieval combines the BM25 and Chroma rankings with Reciprocal Rank Fusion
(RRF), removes duplicate memory IDs, applies Top-K, and reloads every result
from SQLite. Deleted or stale memories are never returned.

## Grounded Question Answering

`POST /api/v1/chat` runs a bounded LangGraph workflow:

```text
retrieve -> assess evidence -> generate cited claims -> verify
                                      ^                 |
                                      |-- rewrite once -|
```

The graph refuses questions without sufficient retrieved evidence. Each answer
claim must name supporting `memory_id` values, which the application resolves
to SQLite transcript IDs and source offsets. Failed verification can trigger
one rewrite; a second failure returns a safe rejection instead of the draft.
User and assistant messages, including validated citations, are saved to
SQLite. Retrieved transcript content is always treated as untrusted data.

## Memory Timeline

`POST /api/v1/timeline` returns traceable memories in chronological order.
Exact, day, month, year, and parseable approximate dates are sorted without
inventing missing date parts. Unknown or unparseable dates are returned in a
separate `undated_events` list. Optional inclusive `start_date` and `end_date`
filters use the supported date interval for partial dates.

Corrected memories replace the records they supersede, deleted memories are
excluded, and every returned event includes SQLite-backed transcript offsets.
The timeline is a normal Python service and does not use LangGraph or an LLM.

## Grounded Autobiography

`POST /api/v1/autobiographies` generates one to three chapters from retrieved
memories. LangGraph plans the chapters, writes one chapter at a time, verifies
every cited paragraph, and allows at most one revision per chapter. Verified
chapters are saved immediately as a SQLite draft; the record becomes
`completed` only after every requested chapter passes verification.

Every paragraph carries SQLite-backed transcript offsets. Unsupported creative
detail, dialogue, emotion, or false date precision is rejected. A failed final
review leaves only previously verified chapters in the draft. Stored results
can be read with `GET /api/v1/autobiographies/{autobiography_id}`.

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
