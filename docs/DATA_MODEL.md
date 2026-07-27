# SQLite, ChromaDB, LangChain Document, Pydantic 모델 정의

# DATA_MODEL.md

# Life Archive AI Data Model

---

# 1. Overview

Life Archive AI stores information in two different storage systems.

SQLite

↓

Business Data

ChromaDB

↓

Retrieval Index

SQLite is the only Source of Truth.

ChromaDB can always be rebuilt from SQLite.

---

# 2. Data Pipeline

```text
Transcript

↓

Chunk

↓

Memory

↓

Embedding

↓

Hybrid Retrieval

↓

Timeline

↓

Autobiography
```

---

# 3. Transcript Model

Original STT transcript.

```python
class Transcript(BaseModel):

    transcript_id: str

    filename: str

    recording_id: str | None

    language: str | None

    source_type: str

    uploaded_at: datetime

    recorded_at: datetime | None

    content_hash: str

    raw_content: str

    normalized_content: str
```

`uploaded_at` records ingestion time. `recorded_at` is source recording metadata.
Neither field is treated as the remembered event date. Event dates are extracted
later into structured Memory records and may remain unknown.

The raw content is preserved exactly as decoded from UTF-8. Normalized content is
stored separately for downstream processing.

SQLite Table

```text
transcripts

----------------------------

transcript_id

filename

recording_id

language

source_type

uploaded_at

recorded_at

content_hash

raw_content

normalized_content

created_at

updated_at

deleted_at
```

---

# 4. Chunk Model

Each transcript is divided into chunks.

```python
class Chunk(BaseModel):

    segment_id: str

    transcript_id: str

    chunk_index: int

    content: str

    start_offset: int

    end_offset: int
```

SQLite

```text
transcript_segments

----------------------------

segment_id

transcript_id

chunk_index

content

start_offset

end_offset

created_at

updated_at

deleted_at
```

`start_offset` is inclusive and `end_offset` is exclusive. Both offsets refer
to `Transcript.normalized_content`, and every persisted chunk must satisfy:

```python
chunk.content == transcript.normalized_content[
    chunk.start_offset:chunk.end_offset
]
```

The fixed-size chunker supports character units by default and
whitespace-delimited token units when explicitly selected. Validated
`chunk_size` and `chunk_overlap` settings ensure overlap is smaller than the
chunk size. Initial comparison sizes are 256, 512, and 1024 units.
Event-aware chunking is recorded as a later evaluation candidate and is not
used to infer event boundaries during deterministic ingestion.

---

# 5. Memory Model

Core model of this project.

```python
class Memory(BaseModel):

    memory_id: str

    transcript_id: str

    title: str

    summary: str

    people: list[str]

    location: str | None

    event_date: str | None

    date_precision: str

    emotion: str | None

    confidence: float

    uncertainty_notes: str | None

    status: str
```

SQLite

```text
memories

-----------------------------------

memory_id

transcript_id

title

summary

people_json

location

event_date

date_precision

emotion

confidence

uncertainty_notes

status

supersedes_memory_id

created_at

updated_at

deleted_at
```

Status

- active

- corrected

- deleted

`people` is serialized to `people_json` TEXT and validated as `list[str]` by
Pydantic. Deleted memories remain stored but are excluded from default queries.

`date_precision` is one of `exact`, `day`, `month`, `year`, `approximate`, or
`unknown`. Unknown dates keep `event_date` null. Partial and approximate dates
remain strings so ingestion never invents missing month, day, or time values.
Low-confidence and approximate extractions retain an `uncertainty_notes` value.

Memory extraction uses OpenAI Structured Outputs with a strict Pydantic batch
schema. Evidence offsets returned for a segment are validated and converted to
absolute normalized-transcript offsets before the memory and its
`memory_sources` row are saved in one SQLite transaction.

---

# 5.1 Memory Source Model

Memory evidence is stored separately so every structured memory remains
traceable to transcript offsets and, when available, one segment.

```python
class MemorySource(BaseModel):

    memory_source_id: str

    memory_id: str

    transcript_id: str

    segment_id: str | None

    start_offset: int

    end_offset: int
```

SQLite table: `memory_sources`.

---

# 6. Citation Model

Every generated answer must contain citations.

```python
class Citation(BaseModel):

    memory_id: str

    transcript_id: str

    chunk_id: str

    start_offset: int

    end_offset: int
```

---

# 7. Timeline Model

```python
class TimelineEvent(BaseModel):

    memory_id: str

    transcript_id: str

    event_date: str | None

    date_precision: DatePrecision

    date_label: str

    title: str

    description: str

    status: MemoryStatus

    citations: list[CitationRecord]
```

Timeline responses contain `events` and `undated_events`. Dated events are
sorted by the earliest supported date without filling in unknown date parts.
Unknown dates and approximate values that cannot be interpreted are kept in
`undated_events`.

Corrected memories suppress the records named by `supersedes_memory_id`.
Deleted memories and memories without a traceable source are not returned.
Every citation includes the transcript ID and half-open source offset range.

---

# 8. Conversation Model

```python
class ConversationSession(BaseModel):

    session_id: str

    created_at: datetime
```

```python
class Message(BaseModel):

    message_id: str

    session_id: str

    role: str

    content: str

    created_at: datetime
```

SQLite

```text
conversation_sessions

conversation_messages
```

Conversation message citations are stored as JSON TEXT and validated as a list
of typed citation records before writing and after reading.

---

# 9. Autobiography Model

```python
class Chapter(BaseModel):

    title: str

    content: str

    citations: list[Citation]
```

```python
class Autobiography(BaseModel):

    autobiography_id: str

    title: str

    chapters: list[Chapter]

    created_at: datetime
```

SQLite

```text
autobiographies

-----------------------------------

autobiography_id

title

content_json

created_at

updated_at

status

deleted_at
```

Autobiography content is JSON TEXT validated by Pydantic. MVP content is limited
to at most three typed chapters.

---

# 10. LangChain Document

Each chunk becomes one Document.

```python
Document(

    page_content="...",

    metadata={

        "memory_id": "...",

        "transcript_id": "...",

        "chunk_id": "...",

        "event_date": "...",

        "people": [...],

        "location": "...",

    }

)
```

---

# 11. Chroma Metadata

Only retrieval metadata is stored.

```text
id

embedding

memory_id

transcript_id

chunk_id

event_date

people

location
```

Never store business logic only inside Chroma.

---

# 12. SQLite Relationships

```text
Transcript

↓

Transcript Segment

↓

Memory

↓

Conversation

↓

Autobiography
```

---

# 13. Source of Truth

SQLite

Stores

✓ Transcript

✓ Memory

✓ Timeline

✓ Conversation

✓ Autobiography

ChromaDB

Stores

✓ Embedding

✓ Retrieval Metadata

Nothing else.

---

# 14. Data Lifecycle

TXT Upload

↓

Transcript

↓

Chunk

↓

Memory Extraction

↓

SQLite

↓

Embedding

↓

ChromaDB

↓

Retrieval

↓

QA

↓

Timeline

↓

Autobiography

---

# 15. Deletion Policy

Deleting a transcript

↓

Delete Memory

↓

Delete Embedding

↓

Rebuild BM25

↓

Invalidate Timeline

↓

Invalidate Autobiography

Raw transcript files are never modified automatically.

---

# 16. Design Principles

SQLite

↓

Permanent Storage

Chroma

↓

Search Index

Memory

↓

Single Source

Citation

↓

Mandatory

Everything generated by AI must be traceable back to original transcript.
