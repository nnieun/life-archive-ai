# Life Archive AI ERD

SQLite is the only source of truth. ChromaDB is a rebuildable retrieval index
and is intentionally absent from this relational model.

```mermaid
erDiagram
    transcripts ||--o{ transcript_segments : contains
    transcripts ||--o{ memories : owns
    transcripts ||--o{ memory_sources : anchors
    memories ||--o{ memory_sources : cites
    transcript_segments o|--o{ memory_sources : supports
    memories o|--o{ memories : supersedes
    conversation_sessions ||--o{ conversation_messages : contains

    transcripts {
        TEXT transcript_id PK
        TEXT filename
        TEXT recording_id UK
        TEXT language
        TEXT source_type
        TEXT uploaded_at
        TEXT recorded_at
        TEXT content_hash UK
        TEXT raw_content
        TEXT normalized_content
        TEXT created_at
        TEXT updated_at
        TEXT deleted_at
    }

    transcript_segments {
        TEXT segment_id PK
        TEXT transcript_id FK
        INTEGER chunk_index
        TEXT content
        INTEGER start_offset
        INTEGER end_offset
        TEXT created_at
        TEXT updated_at
        TEXT deleted_at
    }

    memories {
        TEXT memory_id PK
        TEXT transcript_id FK
        TEXT summary
        TEXT people_json
        TEXT location
        TEXT event_date
        REAL confidence
        TEXT status
        TEXT supersedes_memory_id FK
        TEXT created_at
        TEXT updated_at
        TEXT deleted_at
    }

    memory_sources {
        TEXT memory_source_id PK
        TEXT memory_id FK
        TEXT transcript_id FK
        TEXT segment_id FK
        INTEGER start_offset
        INTEGER end_offset
        TEXT created_at
        TEXT updated_at
    }

    conversation_sessions {
        TEXT session_id PK
        TEXT title
        TEXT created_at
        TEXT updated_at
        TEXT deleted_at
    }

    conversation_messages {
        TEXT message_id PK
        TEXT session_id FK
        TEXT role
        TEXT content
        TEXT citations_json
        TEXT created_at
        TEXT updated_at
        TEXT deleted_at
    }

    autobiographies {
        TEXT autobiography_id PK
        TEXT title
        TEXT content_json
        TEXT status
        TEXT created_at
        TEXT updated_at
        TEXT deleted_at
    }
```

## Integrity Rules

- Every SQLite connection enables `PRAGMA foreign_keys = ON`.
- Transcript `content_hash` is unique and prevents duplicate ingestion.
- Segment indexes are unique within a transcript.
- Offsets are non-negative and `end_offset >= start_offset`.
- JSON values are stored as TEXT, checked with SQLite `json_valid`, and validated
  with Pydantic before writes and after reads.
- Memory status is restricted to `active`, `corrected`, or `deleted`.
- Deleted memories are hidden by default and retained for traceability.
- `uploaded_at` and `recorded_at` belong to transcripts; `event_date` belongs to
  memories. They are never substituted for one another.
- Creation and modification timestamps use timezone-aware ISO 8601 values.

## Deletion Behavior

Application CRUD uses soft deletion for transcripts, segments, memories,
conversation sessions, messages, and autobiographies. Foreign-key cascade rules
protect consistency if a future administrative hard deletion is explicitly
performed. Raw transcript files are never changed or deleted automatically.
