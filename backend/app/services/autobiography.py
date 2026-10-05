"""Bounded LangGraph workflow for grounded autobiography drafts."""

from __future__ import annotations

import re

import re
from dataclasses import dataclass
from typing import Any, Protocol, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from backend.app.models.autobiography import (
    AutobiographyGenerationResult,
    AutobiographyGapCheckResult,
    AutobiographyInput,
    AutobiographyState,
    ChapterDraft,
    ChapterPlan,
    ChapterPlanItem,
    ChapterReview,
)
from backend.app.models.gap import (
    MemoryGapCandidateStatus,
    MemoryGapRecord,
    MemoryGapStatus,
    MemoryGapType,
)
from backend.app.models.qa import QAEvidence
from backend.app.models.retrieval import RetrievalHit
from backend.app.models.timeline import TimelineEvent
from backend.app.prompts.autobiography import (
    CHAPTER_PLAN_SYSTEM_PROMPT,
    CHAPTER_REVISION_SYSTEM_PROMPT,
    CHAPTER_VERIFICATION_SYSTEM_PROMPT,
    CHAPTER_WRITING_SYSTEM_PROMPT,
    build_autobiography_context,
    build_chapter_input,
    build_plan_input,
    build_review_input,
    build_revision_input,
)
from backend.app.services.qa import (
    QAError,
    StructuredQAModel,
    _invoke_structured,
)
from backend.app.services.timeline import TimelineService
from backend.app.storage.models import (
    AutobiographyChapter,
    AutobiographyContent,
    AutobiographyCreate,
    AutobiographyRecord,
    AutobiographyStatus,
    CitationRecord,
)
from backend.app.storage.repository import SQLiteRepository, StorageConflictError

DEFAULT_OPENAI_MODEL = "gpt-5.6-sol"
IMPORTANT_GAP_THRESHOLD = 0.65

_DIGIT_RUN = re.compile(r"\d+")
_UNCERTAINTY_MARKERS = (
    "확인되지",
    "불확실",
    "정확하지",
    "알 수 없",
    "기록마다",
    "서로 다",
    "uncertain",
    "not confirmed",
    "records differ",
)


class AutobiographyRetriever(Protocol):
    def search(self, query: str, *, top_k: int = 10) -> list[RetrievalHit]:
        """Return active SQLite-backed memories."""


@dataclass(frozen=True)
class AutobiographyModels:
    plan: StructuredQAModel
    write: StructuredQAModel
    verify: StructuredQAModel
    revise: StructuredQAModel


class AutobiographyGapCheckError(RuntimeError):
    """Important-gap preflight could not retrieve related memories."""


def build_openai_autobiography_models(
    model_name: str = DEFAULT_OPENAI_MODEL,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> AutobiographyModels:
    """Create strict Structured Output models for all autobiography stages."""

    model_kwargs: dict[str, object] = {"model": model_name}
    if api_key:
        model_kwargs["api_key"] = api_key
    if base_url:
        model_kwargs.update(base_url=base_url, timeout=300, max_retries=0, use_responses_api=False)
    model = ChatOpenAI(**model_kwargs)
    options = {
        "method": "json_schema",
        "include_raw": True,
        "strict": True,
    }
    return AutobiographyModels(
        plan=model.with_structured_output(ChapterPlan, **options),
        write=model.with_structured_output(ChapterDraft, **options),
        verify=model.with_structured_output(ChapterReview, **options),
        revise=model.with_structured_output(ChapterDraft, **options),
    )


class AutobiographyService:
    """Plan, write, verify, incrementally save, and assemble up to three chapters."""

    def __init__(
        self,
        repository: SQLiteRepository,
        retriever: AutobiographyRetriever,
        timeline_service: TimelineService,
        models: AutobiographyModels,
    ) -> None:
        self._repository = repository
        self._retriever = retriever
        self._timeline_service = timeline_service
        self._models = models
        self.graph = self._build_graph()

    def generate(
        self,
        request: AutobiographyInput,
    ) -> AutobiographyGenerationResult:
        """Create the draft row, run the graph, and return persisted state."""

        if (
            self._repository.get_autobiography(
                request.autobiography_id,
                include_deleted=True,
            )
            is not None
        ):
            raise StorageConflictError("Autobiography already exists")
        self._repository.create_autobiography(
            AutobiographyCreate(
                autobiography_id=request.autobiography_id,
                title=request.title,
                content=AutobiographyContent(chapters=[]),
            )
        )
        initial: AutobiographyState = {
            "autobiography_id": request.autobiography_id,
            "title": request.title,
            "request": request.request,
            "retrieval_query": request.request,
            "target_period": request.target_period,
            "target_topics": request.target_topics,
            "chapter_count": request.chapter_count,
            "top_k": request.top_k,
            "proceed_with_unresolved_gaps": request.proceed_with_unresolved_gaps,
            "retrieved_memory_ids": [],
            "evidence": [],
            "timeline": [],
            "important_unresolved_gaps": [],
            "chapter_plan": [],
            "current_chapter_index": 0,
            "current_draft": None,
            "chapter_drafts": [],
            "review_result": None,
            "citations": [],
            "final_content": None,
            "retry_count": 0,
            "chapter_retry_count": 0,
            "error": None,
        }
        final_state = cast(AutobiographyState, self.graph.invoke(initial))
        record = self._repository.get_autobiography(request.autobiography_id)
        if record is None:
            raise RuntimeError("Autobiography draft disappeared during generation")
        return AutobiographyGenerationResult(
            autobiography=record,
            completed=record.status is AutobiographyStatus.COMPLETED,
            retrieved_memory_ids=final_state["retrieved_memory_ids"],
            citations=final_state["citations"],
            retry_count=final_state["retry_count"],
            error=final_state["error"],
            important_unresolved_gaps=final_state["important_unresolved_gaps"],
            requires_gap_confirmation=bool(
                final_state["important_unresolved_gaps"]
                and not request.proceed_with_unresolved_gaps
            ),
        )

    def check_important_gaps(
        self,
        request: AutobiographyInput,
    ) -> AutobiographyGapCheckResult:
        """Preview only important gaps related to retrieved request memories."""

        query = _retrieval_query(request)
        try:
            hits = self._retriever.search(query, top_k=request.top_k)
        except Exception as exception:
            raise AutobiographyGapCheckError(
                "Autobiography gap preflight retrieval failed"
            ) from exception
        memory_ids = [
            hit.memory_id
            for hit in hits
            if self._repository.list_memory_sources(hit.memory_id)
        ]
        gaps = self._important_gaps_for_memory_ids(memory_ids)
        return AutobiographyGapCheckResult(
            important_unresolved_gaps=gaps,
            retrieved_memory_ids=memory_ids,
            requires_gap_confirmation=bool(gaps),
        )

    def get(self, autobiography_id: str) -> AutobiographyRecord | None:
        """Return one persisted non-deleted autobiography."""

        return self._repository.get_autobiography(autobiography_id)

    def _build_graph(self) -> Any:
        builder = StateGraph(AutobiographyState)
        builder.add_node("analyze_request", self._analyze_request)
        builder.add_node("retrieve_memories", self._retrieve_memories)
        builder.add_node("check_unresolved_gaps", self._check_unresolved_gaps)
        builder.add_node("build_timeline", self._build_timeline)
        builder.add_node("create_chapter_plan", self._create_chapter_plan)
        builder.add_node("write_chapter", self._write_chapter)
        builder.add_node("verify_chapter", self._verify_chapter)
        builder.add_node("revise_once", self._revise_once)
        builder.add_node("save_chapter", self._save_chapter)
        builder.add_node("assemble_autobiography", self._assemble)
        builder.add_node("stop", self._stop)

        builder.add_edge(START, "analyze_request")
        builder.add_edge("analyze_request", "retrieve_memories")
        builder.add_conditional_edges(
            "retrieve_memories",
            self._route_error,
            {"continue": "check_unresolved_gaps", "stop": "stop"},
        )
        builder.add_conditional_edges(
            "check_unresolved_gaps",
            self._route_error,
            {"continue": "build_timeline", "stop": "stop"},
        )
        builder.add_edge("build_timeline", "create_chapter_plan")
        builder.add_conditional_edges(
            "create_chapter_plan",
            self._route_error,
            {"continue": "write_chapter", "stop": "stop"},
        )
        builder.add_conditional_edges(
            "write_chapter",
            self._route_error,
            {"continue": "verify_chapter", "stop": "stop"},
        )
        builder.add_conditional_edges(
            "verify_chapter",
            self._route_review,
            {
                "save": "save_chapter",
                "revise": "revise_once",
                "stop": "stop",
            },
        )
        builder.add_conditional_edges(
            "revise_once",
            self._route_error,
            {"continue": "verify_chapter", "stop": "stop"},
        )
        builder.add_conditional_edges(
            "save_chapter",
            self._route_next_chapter,
            {
                "write": "write_chapter",
                "assemble": "assemble_autobiography",
            },
        )
        builder.add_edge("assemble_autobiography", END)
        builder.add_edge("stop", END)
        return builder.compile()

    @staticmethod
    def _analyze_request(state: AutobiographyState) -> dict[str, object]:
        return {"retrieval_query": _state_retrieval_query(state)}

    def _retrieve_memories(self, state: AutobiographyState) -> dict[str, object]:
        try:
            hits = self._retriever.search(
                state["retrieval_query"],
                top_k=state["top_k"],
            )
        except Exception:
            return {"error": "Memory retrieval failed"}
        evidence: list[QAEvidence] = []
        for hit in hits:
            sources = self._repository.list_memory_sources(hit.memory_id)
            if not sources:
                continue
            evidence.append(
                QAEvidence(
                    memory_id=hit.memory_id,
                    transcript_id=hit.memory.transcript_id,
                    title=hit.memory.title,
                    summary=hit.memory.summary,
                    people=hit.memory.people,
                    location=hit.memory.location,
                    event_date=hit.memory.event_date,
                    uncertainty_notes=hit.memory.uncertainty_notes,
                    sources=[
                        CitationRecord(
                            memory_id=source.memory_id,
                            transcript_id=source.transcript_id,
                            segment_id=source.segment_id,
                            start_offset=source.start_offset,
                            end_offset=source.end_offset,
                        )
                        for source in sources
                    ],
                )
            )
        if not evidence:
            return {
                "retrieved_memory_ids": [hit.memory_id for hit in hits],
                "error": "Insufficient traceable evidence",
            }
        return {
            "retrieved_memory_ids": [hit.memory_id for hit in hits],
            "evidence": evidence,
        }

    def _check_unresolved_gaps(
        self,
        state: AutobiographyState,
    ) -> dict[str, object]:
        gaps = self._important_gaps_for_memory_ids(
            [item.memory_id for item in state["evidence"]]
        )
        update: dict[str, object] = {"important_unresolved_gaps": gaps}
        if gaps and not state["proceed_with_unresolved_gaps"]:
            update["error"] = "Important unresolved memory gaps require confirmation"
        return update

    def _important_gaps_for_memory_ids(
        self,
        memory_ids: list[str],
    ) -> list[MemoryGapRecord]:
        selected = set(memory_ids)
        active_statuses = {
            MemoryGapStatus.OPEN,
            MemoryGapStatus.SEARCHING,
            MemoryGapStatus.CANDIDATE_FOUND,
            MemoryGapStatus.WAITING_USER,
        }
        return [
            gap
            for gap in self._repository.list_memory_gaps(statuses=active_statuses)
            if gap.memory_id in selected
            and gap.importance_score >= IMPORTANT_GAP_THRESHOLD
        ]

    def _build_timeline(self, state: AutobiographyState) -> dict[str, object]:
        result = self._timeline_service.get_timeline()
        selected_ids = {item.memory_id for item in state["evidence"]}
        timeline = [
            event
            for event in (*result.events, *result.undated_events)
            if event.memory_id in selected_ids
        ]
        return {"timeline": timeline}

    def _create_chapter_plan(
        self,
        state: AutobiographyState,
    ) -> dict[str, object]:
        try:
            context = self._context(state)
            plan = _invoke_structured(
                self._models.plan,
                [
                    SystemMessage(content=CHAPTER_PLAN_SYSTEM_PROMPT),
                    HumanMessage(
                        content=build_plan_input(
                            context=context,
                            chapter_count=state["chapter_count"],
                        )
                    ),
                ],
                ChapterPlan,
            )
            if len(plan.chapters) != state["chapter_count"]:
                raise QAError("Plan did not match requested chapter count")
            available = {item.memory_id for item in state["evidence"]}
            if any(
                memory_id not in available
                for chapter in plan.chapters
                for memory_id in chapter.memory_ids
            ):
                raise QAError("Plan referenced unavailable evidence")
            return {"chapter_plan": plan.chapters}
        except QAError:
            return {"error": "Chapter planning failed"}

    def _write_chapter(self, state: AutobiographyState) -> dict[str, object]:
        plan = state["chapter_plan"][state["current_chapter_index"]]
        try:
            draft = _invoke_structured(
                self._models.write,
                [
                    SystemMessage(content=CHAPTER_WRITING_SYSTEM_PROMPT),
                    HumanMessage(
                        content=build_chapter_input(
                            context=self._context(state),
                            plan=plan,
                        )
                    ),
                ],
                ChapterDraft,
            )
            error = _draft_error(draft, plan, state["evidence"])
            if error:
                raise QAError(error)
            return {
                "current_draft": draft,
                "review_result": None,
                "chapter_retry_count": 0,
            }
        except QAError:
            return {"error": "Chapter writing failed"}

    def _verify_chapter(self, state: AutobiographyState) -> dict[str, object]:
        draft = state["current_draft"]
        if draft is None:
            return {"error": "Chapter draft is missing"}
        plan = state["chapter_plan"][state["current_chapter_index"]]
        error = self._draft_error(draft, plan, state)
        if error:
            return {
                "review_result": ChapterReview(passed=False, reason=error)
            }
        try:
            review = _invoke_structured(
                self._models.verify,
                [
                    SystemMessage(content=CHAPTER_VERIFICATION_SYSTEM_PROMPT),
                    HumanMessage(
                        content=build_review_input(
                            context=self._context(state),
                            plan=plan,
                            draft=draft,
                        )
                    ),
                ],
                ChapterReview,
            )
            if any(
                index >= len(draft.paragraphs)
                for index in review.unsupported_paragraph_indexes
            ):
                raise QAError("Review returned an invalid paragraph index")
            return {"review_result": review}
        except QAError:
            return {"error": "Chapter verification failed"}

    def _revise_once(self, state: AutobiographyState) -> dict[str, object]:
        draft = state["current_draft"]
        review = state["review_result"]
        if draft is None or review is None:
            return {"error": "Chapter revision input is missing"}
        plan = state["chapter_plan"][state["current_chapter_index"]]
        try:
            revised = _invoke_structured(
                self._models.revise,
                [
                    SystemMessage(content=CHAPTER_REVISION_SYSTEM_PROMPT),
                    HumanMessage(
                        content=build_revision_input(
                            context=self._context(state),
                            plan=plan,
                            draft=draft,
                            reason=review.reason,
                        )
                    ),
                ],
                ChapterDraft,
            )
            error = self._draft_error(revised, plan, state)
            if error:
                raise QAError(error)
            return {
                "current_draft": revised,
                "review_result": None,
                "chapter_retry_count": 1,
                "retry_count": state["retry_count"] + 1,
            }
        except QAError:
            return {"error": "Chapter revision failed"}

    def _save_chapter(self, state: AutobiographyState) -> dict[str, object]:
        draft = state["current_draft"]
        if draft is None:
            return {"error": "Verified chapter draft is missing"}
        drafts = [*state["chapter_drafts"], draft]
        content = _content_from_drafts(drafts, state["evidence"])
        self._repository.update_autobiography(
            state["autobiography_id"],
            content=content,
            status=AutobiographyStatus.DRAFT,
        )
        citations = [
            citation
            for chapter in content.chapters
            for citation in chapter.citations
        ]
        return {
            "chapter_drafts": drafts,
            "current_draft": None,
            "current_chapter_index": state["current_chapter_index"] + 1,
            "chapter_retry_count": 0,
            "citations": _deduplicate_citations(citations),
        }

    def _assemble(self, state: AutobiographyState) -> dict[str, object]:
        content = _content_from_drafts(
            state["chapter_drafts"],
            state["evidence"],
        )
        self._repository.update_autobiography(
            state["autobiography_id"],
            content=content,
            status=AutobiographyStatus.COMPLETED,
        )
        return {"final_content": content}

    @staticmethod
    def _stop(state: AutobiographyState) -> dict[str, object]:
        if state["error"] is not None:
            return {}
        review = state["review_result"]
        if review is not None and not review.passed:
            return {"error": "Chapter failed final verification"}
        return {"error": "Autobiography generation stopped"}

    def _context(self, state: AutobiographyState) -> str:
        return build_autobiography_context(
            request=state["request"],
            target_period=state["target_period"],
            target_topics=state["target_topics"],
            evidence=state["evidence"],
            timeline=state["timeline"],
            unresolved_gaps=state["important_unresolved_gaps"],
        )

    def _draft_error(
        self,
        draft: ChapterDraft,
        plan: ChapterPlanItem,
        state: AutobiographyState,
    ) -> str | None:
        structural_error = _draft_error(draft, plan, state["evidence"])
        if structural_error is not None:
            return structural_error
        evidence_by_id = {
            item.memory_id: item for item in state["evidence"]
        }
        gaps = [
            gap
            for gap in state["important_unresolved_gaps"]
            if gap.memory_id in plan.memory_ids
        ]
        forbidden_values = {
            candidate.value.casefold()
            for gap in gaps
            for candidate in self._repository.list_memory_gap_candidates(
                gap.gap_id,
                statuses={
                    MemoryGapCandidateStatus.PROPOSED,
                    MemoryGapCandidateStatus.REJECTED,
                },
            )
        }
        for paragraph in draft.paragraphs:
            lowered = paragraph.text.casefold()
            if any(value in lowered for value in forbidden_values):
                return "Chapter used an unconfirmed memory-gap candidate"
            grounding = "\n".join(
                _evidence_grounding(evidence_by_id[memory_id])
                for memory_id in paragraph.memory_ids
            )
            for number in _DIGIT_RUN.findall(paragraph.text):
                if not _grounded_number(number, grounding):
                    return "Chapter used a number absent from cited memories"
            for gap in gaps:
                if (
                    gap.memory_id in paragraph.memory_ids
                    and gap.gap_type is MemoryGapType.CONFLICTING_FACT
                    and _DIGIT_RUN.search(paragraph.text)
                    and not any(marker in lowered for marker in _UNCERTAINTY_MARKERS)
                ):
                    return "Chapter stated a conflicting fact without uncertainty"
        return None

    @staticmethod
    def _route_error(state: AutobiographyState) -> str:
        return "stop" if state["error"] else "continue"

    @staticmethod
    def _route_review(state: AutobiographyState) -> str:
        if state["error"]:
            return "stop"
        review = state["review_result"]
        if review is not None and review.passed:
            return "save"
        if state["chapter_retry_count"] == 0:
            return "revise"
        return "stop"

    @staticmethod
    def _route_next_chapter(state: AutobiographyState) -> str:
        return (
            "assemble"
            if state["current_chapter_index"] >= len(state["chapter_plan"])
            else "write"
        )


def _draft_error(
    draft: ChapterDraft,
    plan: ChapterPlanItem,
    evidence: list[QAEvidence],
) -> str | None:
    allowed = set(plan.memory_ids)
    available = {item.memory_id for item in evidence}
    for paragraph in draft.paragraphs:
        if any(
            memory_id not in allowed or memory_id not in available
            for memory_id in paragraph.memory_ids
        ):
            return "Chapter cited evidence outside its plan"
    return None


def _content_from_drafts(
    drafts: list[ChapterDraft],
    evidence: list[QAEvidence],
) -> AutobiographyContent:
    evidence_by_id = {item.memory_id: item for item in evidence}
    chapters: list[AutobiographyChapter] = []
    for draft in drafts:
        lines: list[str] = []
        citations: list[CitationRecord] = []
        for paragraph in draft.paragraphs:
            markers: list[str] = []
            for memory_id in paragraph.memory_ids:
                source = evidence_by_id[memory_id].sources[0]
                markers.append(
                    f"[{source.memory_id}|{source.transcript_id}:"
                    f"{source.start_offset}-{source.end_offset}]"
                )
                citations.append(source)
            lines.append(_remove_internal_citation_markers(paragraph.text.strip()))
        chapters.append(
            AutobiographyChapter(
                title=draft.title,
                content="\n\n".join(lines),
                citations=_deduplicate_citations(citations),
            )
        )
    return AutobiographyContent(chapters=chapters)


_INTERNAL_CITATION_MARKER = re.compile(
    r"\s*\[mem_[^\]|\s]+\|tr_[^:\]\s]+:\d+-\d+\]"
)


def _remove_internal_citation_markers(text: str) -> str:
    """Keep citations as structured data, never as user-facing prose."""
    return _INTERNAL_CITATION_MARKER.sub("", text).strip()


def _deduplicate_citations(
    citations: list[CitationRecord],
) -> list[CitationRecord]:
    result: list[CitationRecord] = []
    seen: set[tuple[object, ...]] = set()
    for citation in citations:
        key = (
            citation.memory_id,
            citation.transcript_id,
            citation.segment_id,
            citation.start_offset,
            citation.end_offset,
        )
        if key not in seen:
            seen.add(key)
            result.append(citation)
    return result


def _retrieval_query(request: AutobiographyInput) -> str:
    parts = [request.request]
    if request.target_period:
        parts.append(request.target_period)
    parts.extend(request.target_topics)
    return " ".join(part.strip() for part in parts)


def _state_retrieval_query(state: AutobiographyState) -> str:
    parts = [state["request"]]
    if state["target_period"]:
        parts.append(state["target_period"])
    parts.extend(state["target_topics"])
    return " ".join(part.strip() for part in parts)


def _evidence_grounding(evidence: QAEvidence) -> str:
    return "\n".join(
        value
        for value in (
            evidence.title,
            evidence.summary,
            evidence.location,
            evidence.event_date,
            *evidence.people,
            evidence.uncertainty_notes,
        )
        if value
    )


def _grounded_number(number: str, grounding: str) -> bool:
    if number in grounding:
        return True
    unpadded = number.lstrip("0")
    return bool(unpadded) and unpadded in grounding
