"""Grounded autobiography LangGraph tests without real OpenAI calls."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.api.autobiographies import get_autobiography_service
from backend.app.main import app
from backend.app.models.autobiography import (
    AutobiographyInput,
    ChapterDraft,
    ChapterPlan,
    ChapterPlanItem,
    ChapterReview,
)
from backend.app.models.gap import (
    MemoryGapCandidateCreate,
    MemoryGapCandidateRelation,
    MemoryGapCreate,
    MemoryGapSourceType,
    MemoryGapStatus,
    MemoryGapType,
)
from backend.app.models.memory import DatePrecision
from backend.app.models.qa import CitedClaim
from backend.app.models.retrieval import RetrievalHit
from backend.app.models.transcript import LoadedTranscript
from backend.app.services import autobiography
from backend.app.services.autobiography import (
    AutobiographyModels,
    AutobiographyService,
)
from backend.app.services.gap_reconstruction import MemoryGapResolutionService
from backend.app.services.timeline import TimelineService
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.models import (
    AutobiographyStatus,
    MemoryCreate,
    MemorySourceCreate,
    TranscriptSegmentCreate,
)
from backend.app.storage.repository import SQLiteRepository


class QueueModel:
    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.inputs: list[object] = []

    def invoke(self, input: object) -> object:
        self.inputs.append(input)
        if not self.outputs:
            raise AssertionError("Unexpected model call")
        return self.outputs.pop(0)


class FakeRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, top_k: int = 10) -> list[RetrievalHit]:
        self.calls.append((query, top_k))
        return self.hits[:top_k]


@pytest.fixture
def autobiography_storage(tmp_path: Path):
    database = SQLiteDatabase(tmp_path / "autobiography.sqlite3")
    database.initialize()
    repository = SQLiteRepository(database)
    content = "나" * 300
    repository.create_transcript(
        LoadedTranscript(
            transcript_id="tr_001",
            filename="private.txt",
            language="ko",
            source_type="stt_text",
            uploaded_at=datetime(2026, 7, 27, tzinfo=UTC),
            content_hash="e" * 64,
            raw_content=content,
            normalized_content=content,
        )
    )
    repository.create_segments(
        [
            TranscriptSegmentCreate(
                segment_id="seg_001",
                transcript_id="tr_001",
                chunk_index=0,
                content=content,
                start_offset=0,
                end_offset=len(content),
            )
        ]
    )
    hits: list[RetrievalHit] = []
    for index, (memory_id, title, year) in enumerate(
        [
            ("mem_childhood", "어린 시절", "2000"),
            ("mem_school", "학교 생활", "2010"),
            ("mem_work", "첫 직장", "2020"),
        ]
    ):
        memory = repository.create_memory(
            MemoryCreate(
                memory_id=memory_id,
                transcript_id="tr_001",
                title=title,
                summary=f"{year}년에 있었던 {title}의 기억",
                people=[],
                location=None,
                event_date=year,
                date_precision=DatePrecision.YEAR,
                emotion=None,
                confidence=0.9,
            )
        )
        repository.create_memory_source(
            MemorySourceCreate(
                memory_source_id=f"src_{memory_id}",
                memory_id=memory_id,
                transcript_id="tr_001",
                segment_id="seg_001",
                start_offset=index * 20,
                end_offset=(index + 1) * 20,
            )
        )
        hits.append(
            RetrievalHit(
                memory_id=memory_id,
                score=1 / (index + 1),
                memory=memory,
                bm25_rank=index + 1,
                bm25_score=float(3 - index),
            )
        )
    yield repository, hits
    database.close()


def _models(
    *,
    plan: QueueModel | None = None,
    write: QueueModel | None = None,
    verify: QueueModel | None = None,
    revise: QueueModel | None = None,
) -> AutobiographyModels:
    return AutobiographyModels(
        plan=plan or QueueModel(),
        write=write or QueueModel(),
        verify=verify or QueueModel(),
        revise=revise or QueueModel(),
    )


def _request(
    autobiography_id: str,
    *,
    chapter_count: int = 1,
    proceed_with_unresolved_gaps: bool = False,
) -> AutobiographyInput:
    return AutobiographyInput(
        autobiography_id=autobiography_id,
        title="나의 이야기",
        request="기억을 바탕으로 자서전을 써 줘",
        target_period=None,
        target_topics=[],
        chapter_count=chapter_count,
        top_k=10,
        proceed_with_unresolved_gaps=proceed_with_unresolved_gaps,
    )


def _plan_item(index: int, memory_id: str) -> ChapterPlanItem:
    return ChapterPlanItem(
        title=f"{index + 1}장",
        focus=f"{memory_id}에 관한 시기",
        memory_ids=[memory_id],
    )


def _draft(index: int, memory_id: str, text: str | None = None) -> ChapterDraft:
    return ChapterDraft(
        title=f"{index + 1}장",
        paragraphs=[
            CitedClaim(
                text=text or f"{memory_id}에 기록된 일을 겪었습니다.",
                memory_ids=[memory_id],
            )
        ],
    )


def test_generates_one_grounded_chapter_and_saves_completed_record(
    autobiography_storage,
) -> None:
    repository, hits = autobiography_storage
    service = AutobiographyService(
        repository,
        FakeRetriever(hits),
        TimelineService(repository),
        _models(
            plan=QueueModel(
                ChapterPlan(chapters=[_plan_item(0, "mem_childhood")])
            ),
            write=QueueModel(_draft(0, "mem_childhood")),
            verify=QueueModel(
                ChapterReview(passed=True, reason="모든 문단이 근거와 일치합니다.")
            ),
        ),
    )

    result = service.generate(_request("autobio_one"))

    assert result.completed is True
    assert result.autobiography.status is AutobiographyStatus.COMPLETED
    assert len(result.autobiography.content.chapters) == 1
    chapter = result.autobiography.content.chapters[0]
    assert "[mem_childhood|tr_001:0-20]" not in chapter.content
    assert chapter.citations[0].memory_id == "mem_childhood"
    assert chapter.citations[0].memory_id == "mem_childhood"
    assert result.autobiography == repository.get_autobiography("autobio_one")


def test_generates_maximum_three_chapters_in_plan_order(
    autobiography_storage,
) -> None:
    repository, hits = autobiography_storage
    memory_ids = ["mem_childhood", "mem_school", "mem_work"]
    service = AutobiographyService(
        repository,
        FakeRetriever(hits),
        TimelineService(repository),
        _models(
            plan=QueueModel(
                ChapterPlan(
                    chapters=[
                        _plan_item(index, memory_id)
                        for index, memory_id in enumerate(memory_ids)
                    ]
                )
            ),
            write=QueueModel(
                *[
                    _draft(index, memory_id)
                    for index, memory_id in enumerate(memory_ids)
                ]
            ),
            verify=QueueModel(
                *[
                    ChapterReview(passed=True, reason="근거와 일치합니다.")
                    for _ in memory_ids
                ]
            ),
        ),
    )

    result = service.generate(
        _request("autobio_three", chapter_count=3)
    )

    assert result.completed is True
    assert [
        chapter.title for chapter in result.autobiography.content.chapters
    ] == ["1장", "2장", "3장"]
    assert len(result.citations) == 3
    assert {
        citation.memory_id for citation in result.citations
    } == set(memory_ids)


def test_insufficient_evidence_stores_empty_draft_without_model_calls(
    autobiography_storage,
) -> None:
    repository, _hits = autobiography_storage
    plan_model = QueueModel()
    service = AutobiographyService(
        repository,
        FakeRetriever([]),
        TimelineService(repository),
        _models(plan=plan_model),
    )

    result = service.generate(_request("autobio_empty"))

    assert result.completed is False
    assert result.error == "Insufficient traceable evidence"
    assert result.autobiography.status is AutobiographyStatus.DRAFT
    assert result.autobiography.content.chapters == []
    assert plan_model.inputs == []


def test_failed_review_revises_once_and_saves_only_verified_text(
    autobiography_storage,
) -> None:
    repository, hits = autobiography_storage
    unsupported = _draft(
        0,
        "mem_childhood",
        "2000년 여름에 매우 행복하게 여행했습니다.",
    )
    corrected = _draft(
        0,
        "mem_childhood",
        "2000년에 어린 시절의 일을 겪었습니다.",
    )
    revise_model = QueueModel(corrected)
    service = AutobiographyService(
        repository,
        FakeRetriever(hits),
        TimelineService(repository),
        _models(
            plan=QueueModel(
                ChapterPlan(chapters=[_plan_item(0, "mem_childhood")])
            ),
            write=QueueModel(unsupported),
            verify=QueueModel(
                ChapterReview(
                    passed=False,
                    reason="여름, 감정, 여행은 근거에 없습니다.",
                    unsupported_paragraph_indexes=[0],
                ),
                ChapterReview(passed=True, reason="수정 후 근거와 일치합니다."),
            ),
            revise=revise_model,
        ),
    )

    result = service.generate(_request("autobio_revised"))

    assert result.completed is True
    assert result.retry_count == 1
    content = result.autobiography.content.chapters[0].content
    assert "여름" not in content
    assert "매우 행복" not in content
    assert len(revise_model.inputs) == 1


def test_second_review_failure_keeps_safe_empty_draft(
    autobiography_storage,
) -> None:
    repository, hits = autobiography_storage
    unsupported = _draft(
        0,
        "mem_childhood",
        "근거에 없는 긴 대화를 나누었습니다.",
    )
    service = AutobiographyService(
        repository,
        FakeRetriever(hits),
        TimelineService(repository),
        _models(
            plan=QueueModel(
                ChapterPlan(chapters=[_plan_item(0, "mem_childhood")])
            ),
            write=QueueModel(unsupported),
            verify=QueueModel(
                ChapterReview(
                    passed=False,
                    reason="대화가 근거에 없습니다.",
                    unsupported_paragraph_indexes=[0],
                ),
                ChapterReview(
                    passed=False,
                    reason="수정본에도 대화가 남았습니다.",
                    unsupported_paragraph_indexes=[0],
                ),
            ),
            revise=QueueModel(unsupported),
        ),
    )

    result = service.generate(_request("autobio_rejected"))

    assert result.completed is False
    assert result.error == "Chapter failed final verification"
    assert result.autobiography.status is AutobiographyStatus.DRAFT
    assert result.autobiography.content.chapters == []
    assert result.citations == []


def test_verified_earlier_chapter_remains_saved_when_later_chapter_fails(
    autobiography_storage,
) -> None:
    repository, hits = autobiography_storage
    second = _draft(
        1,
        "mem_school",
        "근거에 없는 대화를 길게 나누었습니다.",
    )
    service = AutobiographyService(
        repository,
        FakeRetriever(hits),
        TimelineService(repository),
        _models(
            plan=QueueModel(
                ChapterPlan(
                    chapters=[
                        _plan_item(0, "mem_childhood"),
                        _plan_item(1, "mem_school"),
                    ]
                )
            ),
            write=QueueModel(_draft(0, "mem_childhood"), second),
            verify=QueueModel(
                ChapterReview(passed=True, reason="첫 장은 근거와 일치합니다."),
                ChapterReview(
                    passed=False,
                    reason="둘째 장의 대화는 근거에 없습니다.",
                    unsupported_paragraph_indexes=[0],
                ),
                ChapterReview(
                    passed=False,
                    reason="수정 후에도 근거가 없습니다.",
                    unsupported_paragraph_indexes=[0],
                ),
            ),
            revise=QueueModel(second),
        ),
    )

    result = service.generate(
        _request("autobio_partial", chapter_count=2)
    )

    assert result.completed is False
    assert result.autobiography.status is AutobiographyStatus.DRAFT
    assert len(result.autobiography.content.chapters) == 1
    assert result.autobiography.content.chapters[0].title == "1장"
    assert result.citations[0].memory_id == "mem_childhood"


def _important_location_gap(repository: SQLiteRepository):
    gap = repository.create_memory_gap(
        MemoryGapCreate(
            gap_id="gap_childhood_place",
            memory_id="mem_childhood",
            gap_type=MemoryGapType.MISSING_LOCATION,
            clue_text="어린 시절에 갔던 극장 이름은 기억나지 않는다.",
            missing_field="location_detail",
            period_start="2000",
            period_end="2000",
            confidence=0.9,
            importance_score=0.9,
            status=MemoryGapStatus.CANDIDATE_FOUND,
            source_type=MemoryGapSourceType.MEMORY,
            source_id="mem_childhood",
        )
    )
    candidate = repository.create_memory_gap_candidate(
        MemoryGapCandidateCreate(
            candidate_id="gcan_childhood_place",
            gap_id=gap.gap_id,
            value="아카데미극장",
            explanation="관련 내부 기록에서 찾은 후보입니다.",
            deterministic_score=0.8,
            llm_relation=MemoryGapCandidateRelation.RELATED,
            supporting_source_ids=["seg_001"],
        )
    )
    return gap, candidate


def test_important_unresolved_gap_blocks_before_any_writing_model(
    autobiography_storage,
) -> None:
    repository, hits = autobiography_storage
    gap, _candidate = _important_location_gap(repository)
    plan_model = QueueModel()
    service = AutobiographyService(
        repository,
        FakeRetriever(hits),
        TimelineService(repository),
        _models(plan=plan_model),
    )

    preview = service.check_important_gaps(_request("preview_only"))
    result = service.generate(_request("autobio_blocked_gap"))

    assert preview.requires_gap_confirmation is True
    assert [item.gap_id for item in preview.important_unresolved_gaps] == [
        gap.gap_id
    ]
    assert result.completed is False
    assert result.requires_gap_confirmation is True
    assert result.error == "Important unresolved memory gaps require confirmation"
    assert result.autobiography.content.chapters == []
    assert plan_model.inputs == []


def test_unconfirmed_candidate_is_removed_when_user_proceeds_with_current_data(
    autobiography_storage,
) -> None:
    repository, hits = autobiography_storage
    gap, candidate = _important_location_gap(repository)
    hallucinated = _draft(
        0,
        "mem_childhood",
        "2000년에 아카데미극장에서 어린 시절을 보냈습니다.",
    )
    guarded = _draft(
        0,
        "mem_childhood",
        "2000년의 정확한 극장 이름은 현재 기록에서 확인되지 않습니다.",
    )
    write_model = QueueModel(hallucinated)
    verify_model = QueueModel(
        ChapterReview(passed=True, reason="미확인 장소를 사실로 쓰지 않았습니다.")
    )
    service = AutobiographyService(
        repository,
        FakeRetriever(hits),
        TimelineService(repository),
        _models(
            plan=QueueModel(
                ChapterPlan(chapters=[_plan_item(0, "mem_childhood")])
            ),
            write=write_model,
            verify=verify_model,
            revise=QueueModel(guarded),
        ),
    )

    result = service.generate(
        _request(
            "autobio_guarded_gap",
            proceed_with_unresolved_gaps=True,
        )
    )

    assert result.completed is True
    assert result.requires_gap_confirmation is False
    assert result.retry_count == 1
    content = result.autobiography.content.chapters[0].content
    assert candidate.value not in content
    assert "현재 기록에서 확인되지 않습니다" in content
    assert gap.gap_id in str(write_model.inputs[0])
    assert candidate.value not in str(write_model.inputs[0])
    assert len(verify_model.inputs) == 1


def test_confirmed_gap_correction_is_used_by_the_next_autobiography(
    autobiography_storage,
) -> None:
    repository, _hits = autobiography_storage
    gap, candidate = _important_location_gap(repository)
    resolution = MemoryGapResolutionService(repository).resolve(
        gap_id=gap.gap_id,
        candidate_id=candidate.candidate_id,
        user_confirmed=True,
    )
    corrected = repository.get_memory(resolution.resolved_memory_id)
    assert corrected is not None
    corrected_hit = RetrievalHit(
        memory_id=corrected.memory_id,
        score=1.0,
        memory=corrected,
        bm25_rank=1,
        bm25_score=1.0,
    )
    service = AutobiographyService(
        repository,
        FakeRetriever([corrected_hit]),
        TimelineService(repository),
        _models(
            plan=QueueModel(
                ChapterPlan(chapters=[_plan_item(0, corrected.memory_id)])
            ),
            write=QueueModel(
                _draft(
                    0,
                    corrected.memory_id,
                    "2000년에 아카데미극장에서 어린 시절의 일을 겪었습니다.",
                )
            ),
            verify=QueueModel(
                ChapterReview(passed=True, reason="확인된 수정 기억과 일치합니다.")
            ),
        ),
    )

    preview = service.check_important_gaps(_request("resolved_preview"))
    result = service.generate(_request("autobio_resolved_gap"))

    assert preview.requires_gap_confirmation is False
    assert result.completed is True
    assert "아카데미극장" in result.autobiography.content.chapters[0].content
    assert result.retrieved_memory_ids == [corrected.memory_id]
    assert repository.get_memory_gap(gap.gap_id).status is MemoryGapStatus.RESOLVED


def test_openai_models_use_strict_structured_outputs(monkeypatch) -> None:
    structured = [Mock(), Mock(), Mock(), Mock()]
    chat_model = Mock()
    chat_model.with_structured_output.side_effect = structured
    chat_openai = Mock(return_value=chat_model)
    monkeypatch.setattr(autobiography, "ChatOpenAI", chat_openai)

    built = autobiography.build_openai_autobiography_models("test-model")

    chat_openai.assert_called_once_with(model="test-model")
    assert [
        built.plan,
        built.write,
        built.verify,
        built.revise,
    ] == structured
    for call in chat_model.with_structured_output.call_args_list:
        assert call.kwargs == {
            "method": "json_schema",
            "include_raw": True,
            "strict": True,
        }


def test_autobiography_generate_and_get_api(autobiography_storage) -> None:
    repository, hits = autobiography_storage
    service = AutobiographyService(
        repository,
        FakeRetriever(hits),
        TimelineService(repository),
        _models(
            plan=QueueModel(
                ChapterPlan(chapters=[_plan_item(0, "mem_childhood")])
            ),
            write=QueueModel(_draft(0, "mem_childhood")),
            verify=QueueModel(
                ChapterReview(passed=True, reason="근거와 일치합니다.")
            ),
        ),
    )
    app.dependency_overrides[get_autobiography_service] = lambda: service
    client = TestClient(app)
    try:
        generated = client.post(
            "/api/v1/autobiographies",
            json={
                "autobiography_id": "autobio_api",
                "title": "API 자서전",
                "request": "내 이야기를 써 줘",
                "chapter_count": 1,
            },
        )
        gap_check = client.post(
            "/api/v1/autobiographies/gap-check",
            json={
                "title": "API 자서전",
                "request": "내 이야기를 써 줘",
                "chapter_count": 1,
            },
        )
        fetched = client.get("/api/v1/autobiographies/autobio_api")
        missing = client.get("/api/v1/autobiographies/missing")
    finally:
        app.dependency_overrides.clear()

    assert generated.status_code == 200
    assert generated.json()["completed"] is True
    assert fetched.status_code == 200
    assert fetched.json()["autobiography_id"] == "autobio_api"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "http_error"
    assert gap_check.status_code == 200
    assert gap_check.json()["requires_gap_confirmation"] is False
