"""Tool-governed reconstruction and user-confirmed resolution tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from backend.app.models.gap import (
    MemoryGapCandidateCreate,
    MemoryGapCandidateProposal,
    MemoryGapCandidateProposalBatch,
    MemoryGapCandidateRelation,
    MemoryGapCandidateStatus,
    MemoryGapCreate,
    MemoryGapSourceType,
    MemoryGapStatus,
    MemoryGapType,
)
from backend.app.models.memory import DatePrecision
from backend.app.models.retrieval import RetrievalHit
from backend.app.models.transcript import LoadedTranscript
from backend.app.services.gap_reconstruction import (
    MemoryGapAgentPolicyError,
    MemoryGapCandidateOutputError,
    MemoryGapConfirmationRequiredError,
    MemoryGapModels,
    MemoryGapReconstructionService,
    MemoryGapResolutionService,
    MemoryGapResolutionUnsupportedError,
)
from backend.app.services.gap_tools import build_memory_gap_tools
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import (
    MemoryCreate,
    MemorySourceCreate,
    TranscriptSegmentCreate,
)
from backend.app.storage.repository import SQLiteRepository

CONTENT = (
    "2001년 대구 동성로에서 친구와 영화를 봤지만 극장 이름은 기억나지 않는다.\n"
    "다른 기록에는 같은 해 대구 동성로 아카데미극장 앞에서 그 친구를 만났다고 적혀 있다."
)


class QueueAgentModel:
    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.inputs: list[object] = []

    def invoke(self, input: object) -> object:
        self.inputs.append(input)
        if not self.outputs:
            raise AssertionError("Unexpected model call")
        return self.outputs.pop(0)


class StaticRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        self.queries.append((query, top_k))
        return self.hits[:top_k]


@pytest.fixture
def reconstruction_storage(tmp_path: Path):
    database = SQLiteDatabase(tmp_path / "reconstruction.sqlite3")
    database.initialize()
    repository = SQLiteRepository(database)
    repository.create_transcript(
        LoadedTranscript(
            transcript_id="tr_reconstruction",
            filename="recording_001.txt",
            language="ko",
            source_type="stt_text",
            uploaded_at=datetime(2026, 8, 31, tzinfo=UTC),
            content_hash=hashlib.sha256(CONTENT.encode("utf-8")).hexdigest(),
            raw_content=CONTENT,
            normalized_content=CONTENT,
        )
    )
    repository.create_segment(
        TranscriptSegmentCreate(
            segment_id="seg_reconstruction",
            transcript_id="tr_reconstruction",
            chunk_index=0,
            content=CONTENT,
            start_offset=0,
            end_offset=len(CONTENT),
        )
    )
    gap_memory = repository.create_memory(
        MemoryCreate(
            memory_id="mem_gap",
            transcript_id="tr_reconstruction",
            title="친구와 영화 관람",
            summary=(
                "2001년 대구 동성로에서 친구와 영화를 봤지만 "
                "극장 이름은 기억나지 않는다."
            ),
            people=["친구"],
            location="대구 동성로",
            event_date="2001",
            date_precision=DatePrecision.YEAR,
            confidence=0.8,
            uncertainty_notes="극장의 정확한 이름이 불확실하다.",
        )
    )
    support_memory = repository.create_memory(
        MemoryCreate(
            memory_id="mem_support",
            transcript_id="tr_reconstruction",
            title="아카데미극장 앞 만남",
            summary=(
                "2001년 대구 동성로 아카데미극장 앞에서 "
                "그 친구를 만났다."
            ),
            people=["친구"],
            location="대구 동성로 아카데미극장",
            event_date="2001",
            date_precision=DatePrecision.YEAR,
            confidence=0.95,
        )
    )
    for memory, suffix in ((gap_memory, "gap"), (support_memory, "support")):
        repository.create_memory_source(
            MemorySourceCreate(
                memory_source_id=f"src_{suffix}",
                memory_id=memory.memory_id,
                transcript_id="tr_reconstruction",
                segment_id="seg_reconstruction",
                start_offset=0,
                end_offset=len(CONTENT),
            )
        )
    gap = repository.create_memory_gap(
        MemoryGapCreate(
            gap_id="gap_theater",
            memory_id="mem_gap",
            gap_type=MemoryGapType.MISSING_LOCATION,
            clue_text=gap_memory.summary,
            missing_field="location_detail",
            period_start="2001",
            period_end="2001",
            location="대구 동성로",
            people=["친구"],
            confidence=0.92,
            importance_score=0.9,
            source_type=MemoryGapSourceType.MEMORY,
            source_id="mem_gap",
        )
    )
    hit = RetrievalHit(
        memory_id=support_memory.memory_id,
        score=0.03,
        memory=support_memory,
        dense_rank=1,
        dense_distance=0.1,
    )
    yield repository, gap, hit
    database.close()


def _tool_call(name: str, call_id: str, **args: object) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _candidate_batch(value: str = "아카데미극장") -> MemoryGapCandidateProposalBatch:
    return MemoryGapCandidateProposalBatch(
        candidates=[
            MemoryGapCandidateProposal(
                value=value,
                evidence_text="대구 동성로 아카데미극장",
                explanation="같은 해와 같은 거리, 같은 친구가 언급된 기록입니다.",
                llm_relation=MemoryGapCandidateRelation.SUPPORTS,
                supporting_source_ids=["mem_support"],
            )
        ]
    )


def _reconstruction_service(
    repository: SQLiteRepository,
    retriever: StaticRetriever,
    agent: QueueAgentModel,
    candidate: QueueAgentModel,
) -> MemoryGapReconstructionService:
    tools = build_memory_gap_tools(repository, retriever)
    return MemoryGapReconstructionService(
        repository,
        tools,
        MemoryGapModels(agent=agent, candidate=candidate),
    )


def test_agent_searches_memory_and_persists_only_grounded_candidate(
    reconstruction_storage,
) -> None:
    repository, gap, hit = reconstruction_storage
    retriever = StaticRetriever([hit])
    agent = QueueAgentModel(
        _tool_call(
            "search_memory",
            "call_memory",
            query="2001년 동성로 극장 친구",
            top_k=5,
        ),
        AIMessage(content="내부 기억에 충분한 근거가 있습니다."),
    )
    candidate = QueueAgentModel(_candidate_batch())

    result = _reconstruction_service(
        repository,
        retriever,
        agent,
        candidate,
    ).reconstruct(gap.gap_id)

    assert result.gap.status is MemoryGapStatus.CANDIDATE_FOUND
    assert result.searched_tools == ["search_memory"]
    assert result.tool_call_count == 1
    assert result.needs_more_clues is False
    assert len(result.candidates) == 1
    assert result.candidates[0].value == "아카데미극장"
    assert result.candidates[0].supporting_source_ids == ["mem_support"]
    assert result.candidates[0].deterministic_score > 0.5
    assert retriever.queries == [("2001년 동성로 극장 친구", 5)]
    assert [tool.name for tool in build_memory_gap_tools(repository, retriever)] == [
        "search_memory",
        "search_uploaded_documents",
        "search_memory_gaps",
        "request_more_clues",
    ]
    assert repository.get_memory("mem_gap").location == "대구 동성로"
    assert repository.get_correction_of("mem_gap") is None


def test_server_rejects_out_of_order_tool_call_without_executing_it(
    reconstruction_storage,
) -> None:
    repository, gap, _hit = reconstruction_storage
    retriever = StaticRetriever([])
    service = _reconstruction_service(
        repository,
        retriever,
        QueueAgentModel(
            _tool_call(
                "search_uploaded_documents",
                "call_document",
                query="극장",
                top_k=5,
            )
        ),
        QueueAgentModel(),
    )

    with pytest.raises(MemoryGapAgentPolicyError, match="search_memory"):
        service.reconstruct(gap.gap_id)

    assert retriever.queries == []
    assert repository.get_memory_gap(gap.gap_id).status is MemoryGapStatus.OPEN
    assert repository.list_memory_gap_candidates(gap.gap_id) == []


def test_candidate_quote_must_exist_in_tool_evidence(
    reconstruction_storage,
) -> None:
    repository, gap, hit = reconstruction_storage
    service = _reconstruction_service(
        repository,
        StaticRetriever([hit]),
        QueueAgentModel(
            _tool_call(
                "search_memory",
                "call_memory",
                query="동성로 극장",
                top_k=5,
            ),
            AIMessage(content="검색 완료"),
        ),
        QueueAgentModel(
            MemoryGapCandidateProposalBatch(
                candidates=[
                    MemoryGapCandidateProposal(
                        value="중앙극장",
                        evidence_text="중앙극장",
                        explanation="근거 없이 만든 후보",
                        llm_relation=MemoryGapCandidateRelation.UNKNOWN,
                        supporting_source_ids=["mem_support"],
                    )
                ]
            )
        ),
    )

    with pytest.raises(MemoryGapCandidateOutputError, match="evidence quote"):
        service.reconstruct(gap.gap_id)

    assert repository.get_memory_gap(gap.gap_id).status is MemoryGapStatus.OPEN
    assert repository.list_memory_gap_candidates(gap.gap_id) == []


def test_agent_can_exhaust_local_search_then_request_one_user_clue(
    reconstruction_storage,
) -> None:
    repository, gap, _hit = reconstruction_storage
    query = "존재하지않는단서XYZ"
    service = _reconstruction_service(
        repository,
        StaticRetriever([]),
        QueueAgentModel(
            _tool_call("search_memory", "call_1", query=query, top_k=5),
            _tool_call(
                "search_uploaded_documents",
                "call_2",
                query=query,
                top_k=5,
            ),
            _tool_call("search_memory_gaps", "call_3", query=query, top_k=5),
            _tool_call(
                "request_more_clues",
                "call_4",
                question="그 극장 근처에 기억나는 건물이 있었나요?",
            ),
        ),
        QueueAgentModel(),
    )

    result = service.reconstruct(gap.gap_id)

    assert result.gap.status is MemoryGapStatus.WAITING_USER
    assert result.candidates == []
    assert result.needs_more_clues is True
    assert result.user_question == "그 극장 근처에 기억나는 건물이 있었나요?"
    assert result.searched_tools == [
        "search_memory",
        "search_uploaded_documents",
        "search_memory_gaps",
        "request_more_clues",
    ]
    assert result.tool_call_count == 4


def test_resolution_requires_real_confirmation_and_appends_correction(
    reconstruction_storage,
) -> None:
    repository, gap, hit = reconstruction_storage
    reconstruction = _reconstruction_service(
        repository,
        StaticRetriever([hit]),
        QueueAgentModel(
            _tool_call(
                "search_memory",
                "call_memory",
                query="2001년 동성로 극장 친구",
                top_k=5,
            ),
            AIMessage(content="검색 완료"),
        ),
        QueueAgentModel(_candidate_batch()),
    ).reconstruct(gap.gap_id)
    candidate = reconstruction.candidates[0]
    repository.create_memory_gap_candidate(
        MemoryGapCandidateCreate(
            candidate_id="gcan_alternative",
            gap_id=gap.gap_id,
            value="다른극장",
            explanation="비교용 후보",
            deterministic_score=0.2,
            llm_relation=MemoryGapCandidateRelation.RELATED,
            supporting_source_ids=["mem_support"],
        )
    )
    resolver = MemoryGapResolutionService(repository)

    with pytest.raises(MemoryGapConfirmationRequiredError):
        resolver.resolve(
            gap_id=gap.gap_id,
            candidate_id=candidate.candidate_id,
            user_confirmed=False,
        )
    assert repository.get_memory_gap(gap.gap_id).status is (
        MemoryGapStatus.CANDIDATE_FOUND
    )
    assert repository.get_correction_of("mem_gap") is None

    result = resolver.resolve(
        gap_id=gap.gap_id,
        candidate_id=candidate.candidate_id,
        user_confirmed=True,
    )

    corrected = repository.get_memory(result.resolved_memory_id)
    assert result.gap.status is MemoryGapStatus.RESOLVED
    assert result.candidate.status is MemoryGapCandidateStatus.ACCEPTED
    assert corrected is not None
    assert corrected.supersedes_memory_id == "mem_gap"
    assert corrected.location == "대구 동성로 · 아카데미극장"
    assert repository.get_memory("mem_gap") is not None
    assert [memory.memory_id for memory in repository.list_memories()] == [
        "mem_support",
        corrected.memory_id,
    ]
    assert repository.get_memory_gap_candidate("gcan_alternative").status is (
        MemoryGapCandidateStatus.REJECTED
    )
    assert repository.list_memory_sources(corrected.memory_id)[0].segment_id == (
        "seg_reconstruction"
    )


def test_unsupported_candidate_does_not_change_candidate_or_gap_status(
    reconstruction_storage,
) -> None:
    repository, _gap, _hit = reconstruction_storage
    date_gap = repository.create_memory_gap(
        MemoryGapCreate(
            gap_id="gap_date",
            memory_id="mem_gap",
            gap_type=MemoryGapType.MISSING_DATE,
            clue_text="정확한 시기가 기억나지 않는다.",
            missing_field="event_date",
            confidence=0.8,
            source_type=MemoryGapSourceType.MEMORY,
            source_id="mem_gap",
            status=MemoryGapStatus.CANDIDATE_FOUND,
        )
    )
    candidate = repository.create_memory_gap_candidate(
        MemoryGapCandidateCreate(
            candidate_id="gcan_bad_date",
            gap_id=date_gap.gap_id,
            value="어느 봄날",
            explanation="날짜 형식이 아닌 후보",
            deterministic_score=0.4,
            llm_relation=MemoryGapCandidateRelation.RELATED,
            supporting_source_ids=["seg_reconstruction"],
        )
    )

    with pytest.raises(MemoryGapResolutionUnsupportedError, match="date format"):
        MemoryGapResolutionService(repository).resolve(
            gap_id=date_gap.gap_id,
            candidate_id=candidate.candidate_id,
            user_confirmed=True,
        )

    assert repository.get_memory_gap(date_gap.gap_id).status is (
        MemoryGapStatus.CANDIDATE_FOUND
    )
    assert repository.get_memory_gap_candidate(candidate.candidate_id).status is (
        MemoryGapCandidateStatus.PROPOSED
    )
    assert repository.get_correction_of("mem_gap") is None
