"""Cross-service MVP integration tests without real OpenAI calls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.app.models.autobiography import (
    AutobiographyInput,
    ChapterDraft,
    ChapterPlan,
    ChapterPlanItem,
    ChapterReview,
)
from backend.app.models.memory import (
    DatePrecision,
    ExtractedMemory,
    MemoryExtractionBatch,
)
from backend.app.models.qa import (
    AnswerVerification,
    CitedClaim,
    EvidenceAssessment,
    GroundedAnswerDraft,
)
from backend.app.services.autobiography import (
    AutobiographyModels,
    AutobiographyService,
)
from backend.app.services.ingestion import TranscriptIngestionService
from backend.app.services.qa import (
    INSUFFICIENT_ANSWER,
    GroundedQAService,
    QAModels,
)
from backend.app.services.retrieval import (
    BM25MemoryIndex,
    HybridMemoryRetriever,
)
from backend.app.services.timeline import TimelineService
from backend.app.services.vector_index import MemoryVectorIndex
from backend.app.storage.models import MemoryStatus
from backend.app.storage.repository import SQLiteRepository
from tests.conftest import DeterministicEmbeddings, QueueStructuredModel


@dataclass(frozen=True)
class IngestedArchive:
    repository: SQLiteRepository
    retriever: HybridMemoryRetriever
    timeline: TimelineService
    memory_id: str
    transcript_id: str
    original_path: Path
    original_bytes: bytes


@pytest.fixture
def ingested_archive(
    sqlite_repository: SQLiteRepository,
    tmp_path: Path,
) -> IngestedArchive:
    text = (
        "2010년 겨울, 가상의 화자는 새봄중학교 졸업식에서 "
        "가족과 사진을 찍고 선생님께 감사 인사를 전했다."
    )
    original_bytes = text.encode("utf-8")
    vector_index = MemoryVectorIndex(
        sqlite_repository,
        tmp_path / "chroma",
        embeddings=DeterministicEmbeddings(),
        embedding_version="deterministic-test:v1",
    )
    extraction_model = QueueStructuredModel(
        MemoryExtractionBatch(
            memories=[
                ExtractedMemory(
                    title="중학교 졸업식",
                    summary="졸업식에서 가족과 사진을 찍고 선생님께 인사했다.",
                    people=["가족", "선생님"],
                    location="새봄중학교",
                    event_date="2010",
                    date_precision=DatePrecision.YEAR,
                    emotion=None,
                    confidence=0.95,
                    evidence_start_offset=0,
                    evidence_end_offset=len(text),
                    uncertainty_notes=None,
                )
            ]
        )
    )
    service = TranscriptIngestionService(
        tmp_path / "raw" / "transcripts",
        sqlite_repository,
        extraction_model,
        vector_index,
    )

    result = service.ingest(
        filename="synthetic-school-memory.txt",
        content=original_bytes,
        language="ko",
    )
    bm25 = BM25MemoryIndex(sqlite_repository)
    bm25.rebuild_from_sqlite()
    retriever = HybridMemoryRetriever(
        sqlite_repository,
        vector_index,
        bm25,
    )
    return IngestedArchive(
        repository=sqlite_repository,
        retriever=retriever,
        timeline=TimelineService(sqlite_repository),
        memory_id=result.memory_ids[0],
        transcript_id=result.transcript_id,
        original_path=tmp_path
        / "raw"
        / "transcripts"
        / "synthetic-school-memory.txt",
        original_bytes=original_bytes,
    )


@pytest.mark.integration
def test_grounded_workflows_share_one_ingested_sqlite_source(
    ingested_archive: IngestedArchive,
) -> None:
    archive = ingested_archive

    hits = archive.retriever.search("학교 졸업식에서 무엇을 했나?", top_k=3)
    assert [hit.memory_id for hit in hits[:1]] == [archive.memory_id]

    qa_service = GroundedQAService(
        archive.repository,
        archive.retriever,
        QAModels(
            evidence=QueueStructuredModel(
                EvidenceAssessment(
                    sufficient=True,
                    reason="졸업식 행동을 직접 뒷받침합니다.",
                    selected_memory_ids=[archive.memory_id],
                )
            ),
            answer=QueueStructuredModel(
                GroundedAnswerDraft(
                    claims=[
                        CitedClaim(
                            text="가족과 사진을 찍고 선생님께 인사했습니다.",
                            memory_ids=[archive.memory_id],
                        )
                    ]
                )
            ),
            verification=QueueStructuredModel(
                AnswerVerification(
                    passed=True,
                    reason="주장이 검색된 기억과 일치합니다.",
                )
            ),
            rewrite=QueueStructuredModel(),
        ),
    )
    qa_result = qa_service.answer_question(
        session_id="integration_session",
        question="학교 졸업식에서 무엇을 했나?",
    )

    timeline_result = archive.timeline.get_timeline()

    autobiography_service = AutobiographyService(
        archive.repository,
        archive.retriever,
        archive.timeline,
        AutobiographyModels(
            plan=QueueStructuredModel(
                ChapterPlan(
                    chapters=[
                        ChapterPlanItem(
                            title="졸업의 날",
                            focus="중학교 졸업식",
                            memory_ids=[archive.memory_id],
                        )
                    ]
                )
            ),
            write=QueueStructuredModel(
                ChapterDraft(
                    title="졸업의 날",
                    paragraphs=[
                        CitedClaim(
                            text="졸업식에서 가족과 사진을 찍었습니다.",
                            memory_ids=[archive.memory_id],
                        )
                    ],
                )
            ),
            verify=QueueStructuredModel(
                ChapterReview(
                    passed=True,
                    reason="모든 문단이 근거와 일치합니다.",
                )
            ),
            revise=QueueStructuredModel(),
        ),
    )
    autobiography_result = autobiography_service.generate(
        AutobiographyInput(
            autobiography_id="integration_autobiography",
            title="합성 기억 자서전",
            request="학교 졸업식에 관한 한 장을 작성해 줘",
            target_period="2010년",
            target_topics=["학교"],
            chapter_count=1,
            top_k=3,
        )
    )

    assert archive.original_path.read_bytes() == archive.original_bytes
    assert qa_result.citations[0].memory_id == archive.memory_id
    assert qa_result.citations[0].transcript_id == archive.transcript_id
    assert timeline_result.events[0].citations == qa_result.citations
    assert autobiography_result.completed is True
    assert (
        autobiography_result.autobiography.content.chapters[0]
        .citations[0]
        .memory_id
        == archive.memory_id
    )
    assert archive.repository.get_autobiography(
        "integration_autobiography"
    ) is not None


@pytest.mark.integration
def test_deleted_memory_is_rejected_by_all_read_paths(
    ingested_archive: IngestedArchive,
) -> None:
    archive = ingested_archive
    deleted = archive.repository.soft_delete_memory(archive.memory_id)
    evidence_model = QueueStructuredModel()
    qa_service = GroundedQAService(
        archive.repository,
        archive.retriever,
        QAModels(
            evidence=evidence_model,
            answer=QueueStructuredModel(),
            verification=QueueStructuredModel(),
            rewrite=QueueStructuredModel(),
        ),
    )

    qa_result = qa_service.answer_question(
        session_id="deleted_memory_session",
        question="학교 졸업식에서 무엇을 했나?",
    )

    assert deleted.status is MemoryStatus.DELETED
    assert archive.retriever.search("학교 졸업식") == []
    assert archive.timeline.get_timeline().events == []
    assert qa_result.final_answer == INSUFFICIENT_ANSWER
    assert qa_result.citations == []
    assert evidence_model.inputs == []
