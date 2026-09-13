"""Bounded Tool Calling graph and user-confirmed memory-gap resolution."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, TypedDict, TypeVar, cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ValidationError

from backend.app.models.gap import (
    MemoryGapCandidateCreate,
    ExternalSource,
    MemoryGapCandidateProposal,
    MemoryGapCandidateProposalBatch,
    MemoryGapCandidateRecord,
    MemoryGapCandidateStatus,
    MemoryGapRecord,
    MemoryGapReconstructionResult,
    MemoryGapResolutionResult,
    MemoryGapSearchSource,
    MemoryGapSearchSourceType,
    MemoryGapStatus,
    MemoryGapToolPayload,
    MemoryGapType,
    MemoryGapUpdate,
)
from backend.app.models.memory import (
    DatePrecision,
    MemoryCorrection,
    event_date_matches_precision,
)
from backend.app.prompts.gap_reconstruction import (
    GAP_AGENT_SYSTEM_PROMPT,
    GAP_CANDIDATE_SYSTEM_PROMPT,
    build_gap_agent_input,
    build_gap_candidate_input,
)
from backend.app.services.corrections import (
    MemoryAlreadyCorrectedError,
    MemoryCorrectionService,
    MemoryNotFoundError,
    PreparedMemoryCorrection,
    UntraceableMemoryError,
)
from backend.app.services.retrieval import tokenize_for_bm25
from backend.app.storage.models import MemoryRecord
from backend.app.storage.repository import SQLiteRepository

DEFAULT_OPENAI_MODEL = "gpt-5.6-sol"
MAX_TOOL_CALLS = 6
MAX_REQUEST_MORE_CLUES = 2

_TOOL_SEQUENCE = (
    "search_memory",
    "search_uploaded_documents",
    "search_web",
    "search_memory_gaps",
    "request_more_clues",
)
_GENERIC_CLUE_QUESTION = (
    "저장된 기록만으로는 빈칸을 확인하기 어렵습니다. "
    "당시의 장소, 함께 있던 사람, 또는 시기를 한 가지 더 알려 주세요."
)

StructuredT = TypeVar("StructuredT", bound=BaseModel)


class GapAgentModel(Protocol):
    """Minimal interface shared by a bound OpenAI model and test doubles."""

    def invoke(self, input: object) -> object:
        """Return one AI message or structured output envelope."""


@dataclass(frozen=True)
class MemoryGapModels:
    """Models used for tool selection and grounded candidate judgment."""

    agent: GapAgentModel
    candidate: GapAgentModel


class MemoryGapGraphState(TypedDict):
    gap_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    searched_tools: list[str]
    search_sources: list[MemoryGapSearchSource]
    candidate_drafts: list[MemoryGapCandidateCreate]
    tool_call_count: int
    request_more_clues_count: int
    needs_more_clues: bool
    user_question: str | None
    web_search_consent: bool
    web_search_only: bool


class MemoryGapReconstructionError(RuntimeError):
    """Base privacy-safe reconstruction failure."""


class MemoryGapNotFoundError(MemoryGapReconstructionError):
    """The requested gap does not exist."""


class MemoryGapClosedError(MemoryGapReconstructionError):
    """The gap is already resolved or dismissed."""


class MemoryGapAgentPolicyError(MemoryGapReconstructionError):
    """The model requested a forbidden, repeated, or out-of-order tool."""


class MemoryGapModelUnavailableError(MemoryGapReconstructionError):
    """A model call failed before returning a usable response."""


class MemoryGapCandidateOutputError(MemoryGapReconstructionError):
    """Candidate Structured Output did not match returned evidence."""


class MemoryGapToolExecutionError(MemoryGapReconstructionError):
    """A read-only reconstruction tool failed."""


class MemoryGapConfirmationRequiredError(MemoryGapReconstructionError):
    """A candidate cannot be applied without actual user confirmation."""


class MemoryGapCandidateNotFoundError(MemoryGapReconstructionError):
    """The selected candidate does not exist."""


class MemoryGapCandidateMismatchError(MemoryGapReconstructionError):
    """The selected candidate belongs to another gap or is already decided."""


class MemoryGapCandidateSourceError(MemoryGapReconstructionError):
    """The selected candidate has missing or unauthorized provenance."""


class MemoryGapResolutionUnsupportedError(MemoryGapReconstructionError):
    """The selected candidate cannot safely update the target memory field."""


class MemoryGapTargetChangedError(MemoryGapReconstructionError):
    """The gap's original memory was independently replaced before confirmation."""


def build_openai_memory_gap_models(
    tools: Sequence[BaseTool],
    model_name: str = DEFAULT_OPENAI_MODEL,
    *,
    api_key: str | None = None,
) -> MemoryGapModels:
    """Build actual OpenAI Tool Calling and strict candidate-output models."""

    model_kwargs: dict[str, object] = {"model": model_name}
    if api_key:
        model_kwargs["api_key"] = api_key
    model = ChatOpenAI(**model_kwargs)
    return MemoryGapModels(
        agent=model.bind_tools(
            list(tools),
            tool_choice="auto",
            strict=True,
            parallel_tool_calls=False,
        ),
        candidate=model.with_structured_output(
            MemoryGapCandidateProposalBatch,
            method="json_schema",
            include_raw=True,
            strict=True,
        ),
    )


class MemoryGapReconstructionService:
    """Run a server-governed local search loop and persist grounded candidates."""

    def __init__(
        self,
        repository: SQLiteRepository,
        tools: Sequence[BaseTool],
        models: MemoryGapModels,
    ) -> None:
        tool_names = tuple(tool.name for tool in tools)
        if tool_names != _TOOL_SEQUENCE:
            raise ValueError("Memory-gap tools must use the enforced sequence")
        self._repository = repository
        self._tools = tuple(tools)
        self._models = models
        self.graph = self._build_graph()

    def reconstruct(self, gap_id: str, *, force_web_search: bool = False) -> MemoryGapReconstructionResult:
        """Find local evidence without changing the underlying memory."""

        gap = self._repository.get_memory_gap(gap_id)
        if gap is None:
            raise MemoryGapNotFoundError("Memory gap was not found")
        if gap.status in {MemoryGapStatus.RESOLVED, MemoryGapStatus.DISMISSED}:
            raise MemoryGapClosedError("Memory gap is already closed")

        existing = self._repository.list_memory_gap_candidates(
            gap_id,
            statuses={MemoryGapCandidateStatus.PROPOSED},
        )
        if existing and gap.status is MemoryGapStatus.CANDIDATE_FOUND and not force_web_search:
            return MemoryGapReconstructionResult(
                gap=gap,
                candidates=existing,
                searched_tools=[],
                tool_call_count=0,
                message="이미 확인을 기다리는 복원 후보가 있습니다.",
            )

        if force_web_search:
            for candidate in existing:
                self._repository.update_memory_gap_candidate_status(
                    candidate.candidate_id, MemoryGapCandidateStatus.REJECTED
                )
        self._repository.update_memory_gap(
            gap_id,
            MemoryGapUpdate(status=MemoryGapStatus.SEARCHING),
        )
        initial: MemoryGapGraphState = {
            "gap_id": gap_id,
            "messages": [
                SystemMessage(content=GAP_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=build_gap_agent_input(gap)),
            ],
            "searched_tools": [],
            "search_sources": [],
            "candidate_drafts": [],
            "tool_call_count": 0,
            "request_more_clues_count": 0,
            "needs_more_clues": False,
            "user_question": None,
            "web_search_consent": gap.web_search_consent,
            "web_search_only": (
                gap.web_search_consent and gap.missing_field == "food_place_name"
            ),
        }
        try:
            final = cast(
                MemoryGapGraphState,
                self.graph.invoke(initial, {"recursion_limit": MAX_TOOL_CALLS * 3}),
            )
            candidates = self._persist_candidates(final["candidate_drafts"])
            needs_more_clues = final["needs_more_clues"] or not candidates
            target_status = (
                MemoryGapStatus.WAITING_USER
                if needs_more_clues
                else MemoryGapStatus.CANDIDATE_FOUND
            )
            updated_gap = self._repository.update_memory_gap(
                gap_id,
                MemoryGapUpdate(status=target_status),
            )
        except MemoryGapReconstructionError:
            self._restore_open_status(gap_id)
            raise
        except Exception as exception:
            self._restore_open_status(gap_id)
            raise MemoryGapToolExecutionError(
                "Memory-gap reconstruction tool failed"
            ) from exception

        question = final["user_question"]
        if needs_more_clues and question is None:
            question = _GENERIC_CLUE_QUESTION
        message = (
            "복원 후보를 찾았습니다. 내용을 확인한 뒤 직접 선택해 주세요."
            if candidates
            else "저장된 기록만으로 후보를 확인하지 못했습니다. 단서를 더 알려 주세요."
        )
        return MemoryGapReconstructionResult(
            gap=updated_gap,
            candidates=candidates,
            searched_tools=final["searched_tools"],
            tool_call_count=final["tool_call_count"],
            needs_more_clues=needs_more_clues,
            user_question=question,
            message=message,
        )

    def _build_graph(self) -> Any:
        builder = StateGraph(MemoryGapGraphState)
        builder.add_node("agent", self._call_agent)
        builder.add_node(
            "tool",
            ToolNode(self._tools, handle_tool_errors=False),
        )
        builder.add_node("record_tool", self._record_tool_result)
        builder.add_node("propose_candidates", self._propose_candidates)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tool": "tool", "propose": "propose_candidates"},
        )
        builder.add_edge("tool", "record_tool")
        builder.add_conditional_edges(
            "record_tool",
            self._route_after_tool,
            {"agent": "agent", "finish": END},
        )
        builder.add_edge("propose_candidates", END)
        return builder.compile()

    def _call_agent(self, state: MemoryGapGraphState) -> dict[str, object]:
        if state["tool_call_count"] >= MAX_TOOL_CALLS:
            raise MemoryGapAgentPolicyError("Tool call budget was exhausted")
        try:
            output = self._models.agent.invoke(state["messages"])
        except Exception as exception:
            raise MemoryGapModelUnavailableError(
                "Memory-gap agent model call failed"
            ) from exception
        if not isinstance(output, AIMessage):
            raise MemoryGapCandidateOutputError(
                "Memory-gap agent did not return an AI message"
            )
        if output.invalid_tool_calls:
            raise MemoryGapAgentPolicyError("Agent returned an invalid tool call")
        if len(output.tool_calls) > 1:
            raise MemoryGapAgentPolicyError(
                "Agent may call only one tool per response"
            )
        if not output.tool_calls:
            if not state["searched_tools"]:
                raise MemoryGapAgentPolicyError(
                    "Agent stopped before the required memory search"
                )
            return {"messages": [output]}

        name = output.tool_calls[0]["name"]
        searched_count = len(state["searched_tools"])
        if searched_count >= len(_TOOL_SEQUENCE):
            raise MemoryGapAgentPolicyError("Agent requested an extra tool call")
        allowed_sequence = (
            ("search_web",)
            if state["web_search_only"]
            else tuple(
                tool_name
                for tool_name in _TOOL_SEQUENCE
                if tool_name != "search_web" or state["web_search_consent"]
            )
        )
        expected = allowed_sequence[searched_count]
        if name != expected:
            raise MemoryGapAgentPolicyError(
                f"Agent must call {expected} before {name}"
            )
        if (
            name == "request_more_clues"
            and state["request_more_clues_count"] >= MAX_REQUEST_MORE_CLUES
        ):
            raise MemoryGapAgentPolicyError("Clue request budget was exhausted")
        return {"messages": [output]}

    @staticmethod
    def _route_after_agent(state: MemoryGapGraphState) -> str:
        message = state["messages"][-1]
        if isinstance(message, AIMessage) and message.tool_calls:
            return "tool"
        return "propose"

    @staticmethod
    def _record_tool_result(
        state: MemoryGapGraphState,
    ) -> dict[str, object]:
        message = state["messages"][-1]
        if not isinstance(message, ToolMessage) or not isinstance(
            message.content, str
        ):
            raise MemoryGapToolExecutionError("Tool returned an invalid message")
        try:
            payload = MemoryGapToolPayload.model_validate_json(message.content)
        except (ValidationError, ValueError) as exception:
            raise MemoryGapToolExecutionError(
                "Tool returned an invalid payload"
            ) from exception

        preceding = state["messages"][-2]
        if not isinstance(preceding, AIMessage) or len(preceding.tool_calls) != 1:
            raise MemoryGapToolExecutionError("Tool call context is missing")
        called_name = preceding.tool_calls[0]["name"]
        if payload.tool_name != called_name:
            raise MemoryGapToolExecutionError("Tool payload name does not match")

        by_id = {source.source_id: source for source in state["search_sources"]}
        for source in payload.sources:
            current = by_id.get(source.source_id)
            if current is None or source.score > current.score:
                by_id[source.source_id] = source
        searched_tools = [*state["searched_tools"], payload.tool_name]
        clue_count = state["request_more_clues_count"] + int(
            payload.tool_name == "request_more_clues"
        )
        return {
            "searched_tools": searched_tools,
            "search_sources": sorted(
                by_id.values(),
                key=lambda source: (-source.score, source.source_id),
            ),
            "tool_call_count": state["tool_call_count"] + 1,
            "request_more_clues_count": clue_count,
            "needs_more_clues": payload.needs_user_input,
            "user_question": payload.question,
        }

    @staticmethod
    def _route_after_tool(state: MemoryGapGraphState) -> str:
        return "finish" if state["needs_more_clues"] else "agent"

    def _propose_candidates(
        self,
        state: MemoryGapGraphState,
    ) -> dict[str, object]:
        sources = state["search_sources"]
        if not sources:
            return {
                "candidate_drafts": [],
                "needs_more_clues": True,
                "user_question": _GENERIC_CLUE_QUESTION,
            }
        gap = self._repository.get_memory_gap(state["gap_id"])
        if gap is None:
            raise MemoryGapNotFoundError("Memory gap disappeared during search")
        batch = _invoke_structured_gap_model(
            self._models.candidate,
            [
                SystemMessage(content=GAP_CANDIDATE_SYSTEM_PROMPT),
                HumanMessage(content=build_gap_candidate_input(gap, sources)),
            ],
            MemoryGapCandidateProposalBatch,
        )
        available = {source.source_id: source for source in sources}
        drafts: list[MemoryGapCandidateCreate] = []
        seen_values: set[str] = set()
        for proposal in batch.candidates:
            # A food-place gap must be filled from a web result, never by
            # promoting generic local phrases such as “ate tteokbokki”.
            if gap.missing_field == "food_place_name" and not any(
                available[source_id].source_type is MemoryGapSearchSourceType.EXTERNAL
                for source_id in proposal.supporting_source_ids
                if source_id in available
            ):
                continue
            draft = _validated_candidate(gap, proposal, available)
            normalized_value = draft.value.casefold()
            if normalized_value not in seen_values:
                seen_values.add(normalized_value)
                drafts.append(draft)
        return {
            "candidate_drafts": drafts,
            "needs_more_clues": not drafts,
            "user_question": _GENERIC_CLUE_QUESTION if not drafts else None,
        }

    def _persist_candidates(
        self,
        drafts: list[MemoryGapCandidateCreate],
    ) -> list[MemoryGapCandidateRecord]:
        records: list[MemoryGapCandidateRecord] = []
        for draft in drafts:
            existing = self._repository.get_memory_gap_candidate(
                draft.candidate_id
            )
            if existing is None:
                records.append(
                    self._repository.create_memory_gap_candidate(draft)
                )
            elif existing.gap_id == draft.gap_id and existing.value == draft.value:
                records.append(existing)
            else:
                raise MemoryGapCandidateOutputError(
                    "Candidate identifier collision"
                )
        return sorted(
            records,
            key=lambda candidate: (
                -candidate.deterministic_score,
                candidate.candidate_id,
            ),
        )

    def _restore_open_status(self, gap_id: str) -> None:
        try:
            current = self._repository.get_memory_gap(gap_id)
            if current is not None and current.status is MemoryGapStatus.SEARCHING:
                self._repository.update_memory_gap(
                    gap_id,
                    MemoryGapUpdate(status=MemoryGapStatus.OPEN),
                )
        except Exception:
            pass


class MemoryGapResolutionService:
    """Apply only an API-supplied, user-confirmed grounded candidate."""

    def __init__(self, repository: SQLiteRepository) -> None:
        self._repository = repository
        self._corrections = MemoryCorrectionService(repository)

    def resolve(
        self,
        *,
        gap_id: str,
        candidate_id: str,
        user_confirmed: bool,
    ) -> MemoryGapResolutionResult:
        """Resolve one gap; this method is deliberately not an LLM-bound tool."""

        if not user_confirmed:
            raise MemoryGapConfirmationRequiredError(
                "Actual user confirmation is required"
            )
        gap = self._repository.get_memory_gap(gap_id)
        if gap is None:
            raise MemoryGapNotFoundError("Memory gap was not found")
        if gap.status in {MemoryGapStatus.RESOLVED, MemoryGapStatus.DISMISSED}:
            raise MemoryGapClosedError("Memory gap is already closed")
        candidate = self._repository.get_memory_gap_candidate(candidate_id)
        if candidate is None:
            raise MemoryGapCandidateNotFoundError("Candidate was not found")
        if (
            candidate.gap_id != gap_id
            or candidate.status is not MemoryGapCandidateStatus.PROPOSED
        ):
            raise MemoryGapCandidateMismatchError(
                "Candidate does not belong to this open gap"
            )
        self._validate_candidate_sources(gap, candidate)
        if gap.memory_id is None:
            raise MemoryGapResolutionUnsupportedError(
                "Gap has no memory that can be resolved"
            )
        memory = self._repository.get_memory(gap.memory_id)
        if memory is None:
            raise MemoryGapResolutionUnsupportedError(
                "Gap memory was not found"
            )

        correction = _candidate_correction(gap, memory, candidate.value)
        prepared: PreparedMemoryCorrection | None = None
        if correction is not None:
            try:
                prepared = self._corrections.prepare_memory_correction(
                    memory.memory_id,
                    correction,
                )
            except MemoryAlreadyCorrectedError as exception:
                raise MemoryGapTargetChangedError(
                    "Gap memory was already replaced"
                ) from exception
            except MemoryNotFoundError as exception:
                raise MemoryGapTargetChangedError(
                    "Gap memory is no longer active"
                ) from exception
            except UntraceableMemoryError as exception:
                raise MemoryGapCandidateSourceError(
                    "Gap memory has no traceable transcript source"
                ) from exception
        resolved_gap, accepted, resolved_memory = (
            self._repository.resolve_memory_gap_candidate(
                gap_id,
                candidate_id,
                correction=prepared.memory if prepared else None,
                sources=list(prepared.sources) if prepared else None,
            )
        )
        return MemoryGapResolutionResult(
            gap=resolved_gap,
            candidate=accepted,
            resolved_memory_id=resolved_memory.memory_id,
        )

    def _validate_candidate_sources(
        self,
        gap: MemoryGapRecord,
        candidate: MemoryGapCandidateRecord,
    ) -> None:
        if not candidate.supporting_source_ids:
            raise MemoryGapCandidateSourceError(
                "Candidate has no supporting source"
            )
        if candidate.external_sources and not gap.web_search_consent:
            raise MemoryGapCandidateSourceError(
                "External evidence has no user consent"
            )
        for source_id in candidate.supporting_source_ids:
            if (
                self._repository.get_memory(source_id) is None
                and self._repository.get_segment(source_id) is None
                and self._repository.get_memory_gap(source_id) is None
            ):
                raise MemoryGapCandidateSourceError(
                    "Candidate references an unavailable source"
                )


def _invoke_structured_gap_model(
    model: GapAgentModel,
    messages: list[object],
    schema: type[StructuredT],
) -> StructuredT:
    try:
        output = model.invoke(messages)
    except Exception as exception:
        raise MemoryGapModelUnavailableError(
            "Candidate model call failed"
        ) from exception
    if isinstance(output, schema):
        return output
    if not isinstance(output, Mapping):
        raise MemoryGapCandidateOutputError(
            "Candidate model returned an invalid envelope"
        )
    parsing_error = output.get("parsing_error")
    if parsing_error is not None:
        if isinstance(parsing_error, BaseException):
            raise MemoryGapCandidateOutputError(
                "Candidate output did not match its schema"
            ) from parsing_error
        raise MemoryGapCandidateOutputError(
            "Candidate output did not match its schema"
        )
    parsed = output.get("parsed")
    if parsed is None:
        raise MemoryGapCandidateOutputError("Candidate model returned no output")
    try:
        return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
    except ValidationError as exception:
        raise MemoryGapCandidateOutputError(
            "Candidate output did not match its schema"
        ) from exception


def _validated_candidate(
    gap: MemoryGapRecord,
    proposal: MemoryGapCandidateProposal,
    sources: Mapping[str, MemoryGapSearchSource],
) -> MemoryGapCandidateCreate:
    try:
        selected_sources = [
            sources[source_id] for source_id in proposal.supporting_source_ids
        ]
    except KeyError as exception:
        raise MemoryGapCandidateOutputError(
            "Candidate cited a source outside tool results"
        ) from exception

    evidence = proposal.evidence_text
    value = proposal.value.strip()
    if not _contains_text(evidence, value):
        raise MemoryGapCandidateOutputError(
            "Candidate value was not found in its evidence quote"
        )
    if not any(_contains_text(source.content, evidence) for source in selected_sources):
        raise MemoryGapCandidateOutputError(
            "Candidate evidence quote was not found in cited sources"
        )
    identity = json.dumps(
        {
            "gap_id": gap.gap_id,
            "value": value.casefold(),
            "source_ids": sorted(proposal.supporting_source_ids),
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return MemoryGapCandidateCreate(
        candidate_id=f"gcan_{hashlib.sha256(identity).hexdigest()[:24]}",
        gap_id=gap.gap_id,
        value=value,
        explanation=proposal.explanation.strip(),
        deterministic_score=_deterministic_candidate_score(
            gap,
            selected_sources,
        ),
        llm_relation=proposal.llm_relation,
        supporting_source_ids=proposal.supporting_source_ids,
        external_sources=[
            ExternalSource(
                url=source.url,
                title=source.title,
                source_domain=source.source_domain or "외부 검색",
                published_date=source.published_date,
                snippet=source.content,
                retrieved_at=datetime.now(UTC),
            )
            for source in selected_sources
            if source.url and source.source_domain
        ],
    )


def _contains_text(source: str, value: str) -> bool:
    """Accept harmless whitespace differences without accepting paraphrases."""
    if value in source:
        return True
    return " ".join(source.split()).find(" ".join(value.split())) >= 0


def _deterministic_candidate_score(
    gap: MemoryGapRecord,
    sources: list[MemoryGapSearchSource],
) -> float:
    semantic = max(source.score for source in sources)
    date_match = max(_date_match(gap, source) for source in sources)
    location_match = max(_location_match(gap, source) for source in sources)
    people_match = max(_people_match(gap, source) for source in sources)
    event_match = max(_event_match(gap, source) for source in sources)
    return round(
        min(
            1.0,
            semantic * 0.50
            + date_match * 0.20
            + location_match * 0.15
            + people_match * 0.10
            + event_match * 0.05,
        ),
        6,
    )


def _date_match(gap: MemoryGapRecord, source: MemoryGapSearchSource) -> float:
    if source.event_date is None:
        return 0.0
    periods = {value for value in (gap.period_start, gap.period_end) if value}
    if source.event_date in periods:
        return 1.0
    source_year = source.event_date[:4]
    return 0.5 if any(period[:4] == source_year for period in periods) else 0.0


def _location_match(gap: MemoryGapRecord, source: MemoryGapSearchSource) -> float:
    if gap.location is None or source.location is None:
        return 0.0
    left = "".join(gap.location.casefold().split())
    right = "".join(source.location.casefold().split())
    return 1.0 if left in right or right in left else 0.0


def _people_match(gap: MemoryGapRecord, source: MemoryGapSearchSource) -> float:
    if not gap.people or not source.people:
        return 0.0
    return len(set(gap.people) & set(source.people)) / len(set(gap.people))


def _event_match(gap: MemoryGapRecord, source: MemoryGapSearchSource) -> float:
    gap_tokens = set(tokenize_for_bm25(gap.clue_text))
    source_tokens = set(tokenize_for_bm25(source.title))
    return 1.0 if gap_tokens & source_tokens else 0.0


def _candidate_correction(
    gap: MemoryGapRecord,
    memory: MemoryRecord,
    value: str,
) -> MemoryCorrection | None:
    missing_field = gap.missing_field
    if missing_field in {"location", "location_detail"}:
        corrected_location = value
        if (
            missing_field == "location_detail"
            and memory.location
            and value not in memory.location
        ):
            corrected_location = f"{memory.location} · {value}"
        if corrected_location == memory.location:
            raise MemoryGapResolutionUnsupportedError(
                "Candidate does not change the location"
            )
        return MemoryCorrection(location=corrected_location)
    if missing_field in {"people", "person_detail"}:
        if value in memory.people:
            raise MemoryGapResolutionUnsupportedError(
                "Candidate person already exists in the memory"
            )
        return MemoryCorrection(people=[*memory.people, value])
    if missing_field == "event_date":
        precision = _supported_date_precision(value)
        if precision is None:
            raise MemoryGapResolutionUnsupportedError(
                "Candidate date format is unsupported"
            )
        if value == memory.event_date and precision is memory.date_precision:
            raise MemoryGapResolutionUnsupportedError(
                "Candidate does not change the event date"
            )
        return MemoryCorrection(event_date=value, date_precision=precision)
    if gap.gap_type is MemoryGapType.WEAK_PROVENANCE:
        return None
    raise MemoryGapResolutionUnsupportedError(
        "Gap does not identify a safely correctable field"
    )


def _supported_date_precision(value: str) -> DatePrecision | None:
    for precision in (
        DatePrecision.YEAR,
        DatePrecision.MONTH,
        DatePrecision.DAY,
        DatePrecision.EXACT,
    ):
        if event_date_matches_precision(value, precision):
            return precision
    return None
