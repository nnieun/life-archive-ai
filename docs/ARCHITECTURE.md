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

Structured Memory Extraction

↓

SQLite

↓

Embedding

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

The Q&A workflow is a bounded `StateGraph`:

```text
retrieve
  -> evidence_sufficient
     -> insufficient_answer -> END
     -> generate_answer
        -> verify_answer
           -> finalize -> END
           -> rewrite_once
              -> verify_answer
                 -> finalize_or_reject -> END
```

Model outputs are strict Pydantic Structured Outputs. The generated answer is a
list of claims, and every claim must carry one or more selected `memory_id`
values. Application code resolves those IDs to SQLite `memory_sources`, renders
transcript IDs and half-open source offsets, and rejects IDs outside the
selected evidence before model verification.

Retrieved memories are wrapped as escaped JSON and explicitly treated as
untrusted data. Instructions inside transcripts cannot alter the workflow. The
graph performs at most one rewrite and never returns a draft that fails final
verification.

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

Timeline generation reads active and corrected memories directly from SQLite.
It removes records superseded by visible corrections, excludes deleted and
untraceable memories, interprets partial dates as supported intervals, and
keeps unknown dates in a separate collection. It is a deterministic Python
service with no LangGraph or model call.

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

Memory extraction is a normal service, not a LangGraph workflow. It uses an
OpenAI native JSON Schema Structured Output mapped to a strict Pydantic model,
validates evidence offsets against the stored transcript segment, and writes
the memory plus source reference atomically to SQLite.

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
- minimal metadata: embedding_version and content_hash

Never store business data only in ChromaDB.

If ChromaDB is deleted,

it must be rebuildable from SQLite.

The persistent memory collection uses each SQLite `memory_id` as its Chroma ID.
Indexing is idempotent: an unchanged content hash and embedding version skips
another embedding call, while changed memories are upserted. Deleted SQLite
memories are removed during synchronization.

Similarity search uses Chroma only to rank candidate IDs. Before returning a
result, the service reloads the current memory from SQLite and rejects deleted
or stale candidates. The indexed text is limited to the memory title and
summary and is always rebuildable.

---

# 8. Hybrid Retrieval

The retrieval service combines two independent rankings:

```text
query
  -> Chroma similarity ranking
  -> BM25 word + character bigram ranking
  -> Reciprocal Rank Fusion
  -> deduplicate memory_id
  -> reload active memories from SQLite
  -> Top-K results
```

RRF combines rank positions rather than incomparable raw dense and sparse
scores. The in-memory BM25 index is disposable and can be rebuilt from active
SQLite memories. It verifies content hashes at search time so changed records
must be synchronized before they can be returned.

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

```text
analyze_request
  -> retrieve_memories
  -> build_timeline
  -> create_chapter_plan (1 to 3 chapters)
  -> write_chapter
  -> verify_chapter
     -> pass: save_chapter -> next chapter
     -> fail: revise_once -> verify_chapter
        -> second failure: stop with verified draft only
  -> assemble_autobiography
  -> mark completed
```

Every generated paragraph names its supporting `memory_id` values. Application
code rejects citations outside the chapter plan, resolves accepted IDs to
SQLite source offsets, and saves only verified chapters. Each successful
chapter update persists the accumulated draft; final assembly changes status
to `completed` only after every requested chapter passes.

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

retrieval_query

target_period

target_topics

retrieved_memory_ids

timeline

chapter_plan

current_chapter_index

chapter_drafts

review_result

citations

final_content

retry_count

error
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

GET

/api/v1/memories

POST

/api/v1/chat

POST

/api/v1/timeline

POST

/api/v1/autobiographies

GET

/api/v1/autobiographies/{id}
```

`POST /api/v1/chat` accepts `session_id`, `question`, and `top_k` (1 to 20).
The response includes the final answer, retrieved memory IDs, SQLite-backed
citations, validation result, and bounded retry count.

`POST /api/v1/memories/ingest` accepts an immutable UTF-8 TXT upload as
base64-encoded bytes. FastAPI saves a new raw original, then calls the normal
loader, chunker, structured extraction, SQLite persistence, and Chroma indexing
services. Existing raw filenames and duplicate transcript content are rejected.
`GET /api/v1/memories` returns active SQLite memories with their source offsets.

Streamlit never invokes backend business services. Its five pages communicate
only through the typed HTTP client and FastAPI endpoints.

`POST /api/v1/timeline` accepts optional inclusive ISO `start_date` and
`end_date` values. The response separates chronologically sorted `events` from
`undated_events`, and every event contains source citations.

`POST /api/v1/autobiographies` accepts a title, generation request, optional
period and topics, and a chapter count from one to three. It returns the
persisted draft or completed autobiography plus retrieval and validation
status. `GET /api/v1/autobiographies/{id}` reads the stored result.

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
