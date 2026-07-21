# Codex가 프로젝트 전체를 이해하는 설계 문서

# ARCHITECTURE.md

# Life Archive AI Architecture

---

# 1. System Overview

Life Archive AI is a Memory-Centric Retrieval-Augmented Generation (RAG) system.

Unlike a traditional PDF chatbot, this project stores personal memories as structured long-term memories.

The system retrieves only relevant memories and generates grounded responses.

Main Features

- Memory Ingestion
- Hybrid Retrieval
- Grounded Question Answering
- Timeline Generation
- Autobiography Generation

---

# 2. System Architecture

```mermaid
flowchart TD

User

--> Streamlit

Streamlit

--> FastAPI

FastAPI

--> Memory Service

Memory Service

--> SQLite

Memory Service

--> ChromaDB

FastAPI

--> LangGraph

LangGraph

--> Retriever

Retriever

--> SQLite

Retriever

--> ChromaDB

LangGraph

--> OpenAI

LangGraph

--> FastAPI

FastAPI

--> Streamlit
```

---

# 3. Data Flow

## Memory Ingestion

```text
TXT Upload

↓

Read Transcript

↓

Normalize Text

↓

Chunking

↓

Embedding

↓

Structured Memory Extraction

↓

SQLite

↓

ChromaDB
```

---

## Question Answering

```text
User Question

↓

Hybrid Retrieval

↓

Relevant Memories

↓

Grounded Answer

↓

Citation Verification

↓

Final Answer
```

---

## Timeline Generation

```text
Stored Memories

↓

Sort by Event Date

↓

Timeline Events

↓

Timeline Output
```

---

## Autobiography Generation

```text
Retrieve Memories

↓

Build Timeline

↓

Create Chapter Plan

↓

Write Chapters

↓

Verify Citations

↓

Generate Final Draft
```

---

# 4. Folder Structure

```text
life-archive-ai/

backend/

frontend/

data/

docs/

scripts/

tests/
```

Detailed Structure

```text
backend/

app/

api/

graphs/

prompts/

frontend/

app.py

api_client.py

data/

raw/

processed/

db/

indexes/

exports/
```

---

# 5. Backend Architecture

The backend consists of four major components.

## API Layer

Responsibilities

- Receive requests
- Validate inputs
- Return responses

Files

```text
api/

health.py

chat.py

timeline.py

memories.py

autobiographies.py
```

---

## Service Layer

Business Logic

```text
ingestion.py

retrieval.py

timeline.py

autobiography.py
```

Responsibilities

- Memory processing
- Retrieval
- Timeline creation
- Autobiography generation

---

## Storage Layer

SQLite

Stores

- transcripts
- memories
- conversations
- timeline
- autobiography

ChromaDB

Stores

- embeddings
- vector metadata

SQLite is the source of truth.

---

## AI Layer

LangChain

↓

LangGraph

↓

OpenAI

Responsible for

- Memory extraction
- Retrieval
- Answer generation
- Chapter generation

---

# 6. Database Architecture

SQLite

```text
transcripts

↓

transcript_segments

↓

memories

↓

conversation_sessions

↓

conversation_messages

↓

autobiographies
```

SQLite stores structured information.

---

# 7. ChromaDB

Stores

- embedding
- memory_id
- metadata

Never store business data only in ChromaDB.

If ChromaDB is deleted,

it must be rebuildable from SQLite.

---

# 8. Hybrid Retrieval

The retrieval pipeline combines

Semantic Search

+

BM25

↓

Reciprocal Rank Fusion

↓

Optional MMR

↓

Top-K Memories

Supported Experiments

Similarity

MMR

Top-K

Chunk Size

---

# 9. LangGraph

LangGraph is used ONLY for

Grounded QA

and

Autobiography.

---

## QA Graph

```mermaid
flowchart LR

START

--> Retrieve

Retrieve

--> Generate

Generate

--> Verify

Verify

--> Rewrite

Rewrite

--> END
```

---

## Autobiography Graph

```mermaid
flowchart LR

START

--> Retrieve Memories

Retrieve Memories

--> Timeline

Timeline

--> Chapter Planning

Chapter Planning

--> Chapter Writing

Chapter Writing

--> Verification

Verification

--> Save

Save

--> END
```

---

# 10. State Design

QAState

```text
session_id

question

retrieved_memories

draft_answer

validation_result

final_answer

retry_count
```

AutobiographyState

```text
request

timeline

chapter_plan

current_chapter

draft

review_result

final_book
```

---

# 11. Prompt Architecture

Prompt Categories

Memory Extraction

Question Answering

Verification

Autobiography

Prompt files

```text
prompts/

extraction.py

qa.py

verification.py

autobiography.py
```

---

# 12. API

```text
GET

/api/v1/health

POST

/api/v1/memories/ingest

POST

/api/v1/chat

POST

/api/v1/timeline

POST

/api/v1/autobiographies

GET

/api/v1/autobiographies/{id}
```

---

# 13. Development Principles

SQLite

↓

Single Source of Truth

Chroma

↓

Retrieval Only

LangGraph

↓

Workflow Only

Raw Data

↓

Never Modify

---

# 14. Future Extensions

Possible improvements

Photo Memories

Voice Upload

Video Memories

Knowledge Graph

Relationship Graph

Multi-user

Cloud Deployment

Mobile App