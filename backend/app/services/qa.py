"""Bounded LangGraph workflow for evidence-grounded question answering."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast
from uuid import uuid4
from time import perf_counter
from hashlib import sha256

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from backend.app.models.qa import (
    AnswerVerification,
    AnswerProposal,
    EvidenceAssessment,
    EvidenceSelection,
    GroundedAnswerDraft,
    QAEvidence,
    QAQueryPlan,
    QAResult,
    QAState,
    QAValidationResult,
)
from backend.app.models.retrieval import RetrievalHit
from backend.app.prompts.qa import (
    ANSWER_REWRITE_SYSTEM_PROMPT,
    ANSWER_VERIFICATION_SYSTEM_PROMPT,
    EVIDENCE_ASSESSMENT_SYSTEM_PROMPT,
    GROUNDED_ANSWER_SYSTEM_PROMPT,
    COMBINED_ANSWER_SYSTEM_PROMPT,
    build_evidence_input,
    build_rewrite_input,
    build_verification_input,
)
from backend.app.storage.models import (
    CitationRecord,
    ConversationMessageCreate,
    ConversationSessionCreate,
)
from backend.app.storage.repository import SQLiteRepository, StorageError
import sqlite3
from backend.app.models.qa_diagnostics import QAFailureDiagnostic, QAStepDiagnostic, QASchemaIssue
from backend.app.services.qa_failures import classify_exception, MESSAGES

DEFAULT_OPENAI_MODEL = "gpt-5.6-sol"
INSUFFICIENT_ANSWER = "질문에 답할 수 있는 충분한 근거를 찾지 못했습니다."
REJECTED_ANSWER = "근거 검증을 통과하지 못해 답변할 수 없습니다."

UNTRACEABLE_CITATION = "A cited memory could not be traced back to transcript text"

StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)

_DIGIT_RUN = re.compile(r"\d+")
_SHORTEST_CHECKED_NAME = 2
_OVERVIEW_TERMS = ("누구", "어떤 사람", "관련 기억", "관련된 기억", "함께한 일", "있었던 일")
_COMPOSITE_TERMS = ("각각", "모두", "비교")
_COMPOSITE_SPLIT = re.compile(r"[,;，]|\s+(?:그리고|및)\s+")


class StructuredQAModel(Protocol):
    """Minimal structured-output interface used by production and test models."""

    def invoke(self, input: object) -> object:
        """Return a structured model response."""


class HybridRetriever(Protocol):
    """Minimal hybrid-search interface required by the retrieval node."""

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        """Return active SQLite-backed retrieval hits."""


@dataclass(frozen=True)
class QAModels:
    """Structured models used at each semantic judgment stage."""

    evidence: StructuredQAModel
    answer: StructuredQAModel
    verification: StructuredQAModel
    rewrite: StructuredQAModel


class QAError(RuntimeError):
    """Privacy-safe Q&A workflow failure."""

    def __init__(self, message: str, *, code: str = "model_call_failed") -> None:
        super().__init__(message)
        self.code = code
        self.schema_issues: list[QASchemaIssue] = []


class QAOutputError(QAError):
    """A structured model response failed validation."""


def build_openai_qa_models(
    model_name: str = DEFAULT_OPENAI_MODEL,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    combined: bool = False,
    max_tokens: int | None = None,
    keep_alive: str | None = None,
    native_ollama: bool = False,
    think: bool = False,
    verification_model: str | None = None,
) -> QAModels:
    """Create strict OpenAI Structured Output models for all Q&A stages."""

    if native_ollama and base_url:
        import httpx
        from backend.app.services.ollama_qa import OllamaQAModel
        client = httpx.Client(timeout=300, trust_env=False)
        def local(schema, selected_model: str = model_name):
            return OllamaQAModel(selected_model, base_url, schema, max_tokens=max_tokens or 768,
                                keep_alive=keep_alive or '15m', think=think, client=client)
        return QAModels(evidence=local(EvidenceSelection),
                        answer=local(AnswerProposal if combined else GroundedAnswerDraft),
                        verification=local(AnswerVerification, verification_model or model_name), rewrite=local(GroundedAnswerDraft))

    model_kwargs: dict[str, object] = {"model": model_name}
    if api_key:
        model_kwargs["api_key"] = api_key
    if base_url:
        model_kwargs.update(base_url=base_url, timeout=300, max_retries=0, use_responses_api=False)
    if max_tokens is not None:
        model_kwargs['max_tokens'] = max_tokens
    if base_url:
        model_kwargs['temperature'] = 0
    model = ChatOpenAI(**model_kwargs)
    options = {
        "method": "json_schema",
        "include_raw": True,
        "strict": True,
    }
    return QAModels(
        evidence=model.with_structured_output(EvidenceSelection if base_url else EvidenceAssessment, **options),
        answer=model.with_structured_output(AnswerProposal if combined else GroundedAnswerDraft, **options),
        verification=(ChatOpenAI(**{**model_kwargs, 'model': verification_model}) if verification_model else model).with_structured_output(AnswerVerification, **options),
        rewrite=model.with_structured_output(GroundedAnswerDraft, **options),
    )


class GroundedQAService:
    """Execute, validate, and persist one bounded grounded-Q&A graph."""

    def __init__(
        self,
        repository: SQLiteRepository,
        retriever: HybridRetriever,
        models: QAModels,
        *,
        provider: str = "unknown",
        model_name: str = "unknown",
        combined: bool = False,
        compact: bool = False,
        cache_enabled: bool = False,
        max_rewrites: int = 1,
        performance_enabled: bool = False,
        configuration: str = "",
        entity_aware: bool = True,
    ) -> None:
        self._repository = repository
        if max_rewrites not in (0, 1):
            raise ValueError('max_rewrites must be 0 or 1')
        self._retriever = retriever
        self._models = models
        self._provider = provider
        self._model_name = model_name
        self._combined = combined
        self._compact = compact
        self._max_rewrites = max_rewrites
        self._entity_aware = entity_aware
        self._configuration = configuration or f'qa-speed-v1:{provider}:{model_name}:{combined}:{compact}:{max_rewrites}'
        self._configuration += f':entity-aware={entity_aware}'
        prompt_version = sha256('\n'.join((EVIDENCE_ASSESSMENT_SYSTEM_PROMPT, GROUNDED_ANSWER_SYSTEM_PROMPT,
            COMBINED_ANSWER_SYSTEM_PROMPT, ANSWER_VERIFICATION_SYSTEM_PROMPT, ANSWER_REWRITE_SYSTEM_PROMPT)).encode()).hexdigest()
        self._configuration += ':' + prompt_version
        from backend.app.services.qa_performance import QAPerformanceStore
        self._performance = QAPerformanceStore(repository) if cache_enabled or performance_enabled else None
        self._cache_enabled = cache_enabled
        self.graph = self._build_graph()

    def close(self) -> None:
        clients = {getattr(model, 'client', None) for model in
                   (self._models.evidence, self._models.answer, self._models.verification, self._models.rewrite)}
        for client in clients:
            if client is not None:
                client.close()

    def answer_question(
        self,
        *,
        session_id: str,
        question: str,
        top_k: int = 5,
        progress: Callable[[dict], None] | None = None,
    ) -> QAResult:
        """Run the graph, save the exchange, and return the safe final result."""

        if not session_id.strip():
            raise ValueError("session_id must not be blank")
        if not question.strip():
            raise ValueError("question must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        started = perf_counter()
        cache_key = self._performance.key(session_id, question, top_k, self._configuration) if self._cache_enabled else None
        if cache_key:
            cached = self._performance.get(cache_key, session_id)
            if cached is not None:
                result = cached.model_copy(update={'question': question, 'cache_hit': True, 'steps': [],
                                                  'elapsed_ms': (perf_counter()-started)*1000})
                self._persist_exchange(result)
                self._performance.measure(result, self._configuration)
                return result
        initial_state: QAState = {
            "session_id": session_id,
            "question": question,
            "top_k": top_k,
            "query_plan": QAQueryPlan(),
            "retrieved_memory_ids": [],
            "selected_evidence": [],
            "answer_draft": None,
            "draft_answer": "",
            "citations": [],
            "validation_result": QAValidationResult(
                stage="evidence",
                passed=False,
                reason="Evidence has not been assessed",
            ),
            "final_answer": "",
            "retry_count": 0,
            "error": None,
            "diagnostic_steps": [],
        }
        if progress is None:
            final_state = cast(QAState, self.graph.invoke(initial_state))
        else:
            progress({'stage': 'retrieve', 'memories': []})
            final_state = initial_state
            for event in self.graph.stream(initial_state, stream_mode='updates'):
                for node, update in event.items():
                    final_state.update(update)
                    stage = {'retrieve': 'generate_answer' if self._combined else 'evidence_sufficient',
                             'evidence_sufficient': 'generate_answer', 'generate_answer': 'verify_answer',
                             'rewrite_once': 'verify_answer'}.get(node, node)
                    progress({'stage': stage, 'elapsed_ms': (perf_counter()-started)*1000,
                              'memories': [{'memory_id': item.memory_id, 'title': item.title}
                                           for item in final_state['selected_evidence']]})
        result = QAResult(
            session_id=final_state["session_id"],
            question=final_state["question"],
            retrieved_memory_ids=final_state["retrieved_memory_ids"],
            final_answer=final_state["final_answer"],
            citations=final_state["citations"],
            validation_result=final_state["validation_result"],
            retry_count=final_state["retry_count"],
            error=final_state["error"],
            elapsed_ms=(perf_counter()-started)*1000,
            steps=final_state['diagnostic_steps'],
        )
        user_message_id = self._persist_exchange(result)
        if self._performance is not None:
            if cache_key:
                self._performance.save(cache_key, result, top_k, self._configuration)
            self._performance.measure(result, self._configuration)
        if not result.validation_result.passed or result.error is not None:
            self._repository.save_qa_failure(QAFailureDiagnostic(
                session_id=session_id,
                user_message_id=user_message_id,
                provider=self._provider,
                model=self._model_name,
                status="error" if result.error else (
                    "insufficient" if result.validation_result.stage == "evidence" else "rejected"
                ),
                retrieved_memory_ids=result.retrieved_memory_ids,
                selected_memory_ids=[item.memory_id for item in final_state["selected_evidence"]],
                retry_count=result.retry_count,
                reason=result.validation_result.reason,
                error=result.error,
                elapsed_ms=(perf_counter() - started) * 1000,
                steps=final_state["diagnostic_steps"],
                failure_code=result.validation_result.failure_code,
            ))
        return result

    def _diagnostic_node(
        self,
        name: str,
        function: Callable[[QAState], dict[str, object]],
    ) -> Callable[[QAState], dict[str, object]]:
        """Keep per-request node outcomes in graph state, never shared state."""
        def run(state: QAState) -> dict[str, object]:
            started = perf_counter()
            try:
                update = function(state)
            except Exception as exception:
                code = "storage_error" if isinstance(exception, (StorageError, sqlite3.Error)) else "internal_error"
                update = {
                    "error": "QA step failed",
                    "validation_result": QAValidationResult(
                        stage="evidence" if name in ("retrieve", "evidence_sufficient") else "answer",
                        passed=False, reason="QA step failed", failure_code=code,
                        exception_type=type(exception).__name__,
                    ),
                }
            validation = cast(QAValidationResult | None, update.get("validation_result"))
            if validation is not None and not validation.passed and not validation.failure_code:
                reason = validation.reason
                code = None
                if name == "evidence_sufficient":
                    code = ("no_traceable_sources" if state["retrieved_memory_ids"] else "no_memories_found") if not state["selected_evidence"] else "insufficient_evidence"
                elif name == "verify_answer":
                    if "number " in reason and "cited memories" in reason:
                        code = "unsupported_number"
                    elif "without citing the memory" in reason:
                        code = "unsupported_person"
                    elif reason == UNTRACEABLE_CITATION:
                        code = "source_untraceable"
                    elif "outside the selected evidence" in reason:
                        code = "citation_outside_evidence"
                    elif reason == "Answer draft is missing":
                        code = "draft_missing"
                    else:
                        code = "verifier_rejected"
                if code:
                    validation = validation.model_copy(update={"failure_code": code})
                    update["validation_result"] = validation
            step = QAStepDiagnostic(
                node=name,
                elapsed_ms=(perf_counter() - started) * 1000,
                passed=validation.passed if validation is not None else None,
                reason=validation.reason if validation is not None else cast(str | None, update.get("error")),
                failure_code=validation.failure_code if validation is not None else None,
                exception_type=validation.exception_type if validation is not None else None,
                schema_issues=validation.schema_issues if validation is not None else [],
            )
            update["diagnostic_steps"] = [*state.get("diagnostic_steps", []), step]
            return update
        return run

    def _build_graph(self) -> Any:
        builder = StateGraph(QAState)
        builder.add_node("retrieve", self._diagnostic_node("retrieve", self._retrieve))
        builder.add_node("evidence_sufficient", self._diagnostic_node("evidence_sufficient", self._assess_evidence))
        builder.add_node("insufficient_answer", self._diagnostic_node("insufficient_answer", self._insufficient_answer))
        builder.add_node("generate_answer", self._diagnostic_node("generate_answer", self._generate_answer))
        builder.add_node("verify_answer", self._diagnostic_node("verify_answer", self._verify_answer))
        builder.add_node("rewrite_once", self._diagnostic_node("rewrite_once", self._rewrite_once))
        builder.add_node("finalize", self._diagnostic_node("finalize", self._finalize))
        builder.add_node("reject", self._diagnostic_node("reject", self._reject))

        builder.add_edge(START, "retrieve")
        if self._combined:
            builder.add_conditional_edges('retrieve', self._route_retrieved,
                                          {'generate': 'generate_answer', 'insufficient': 'insufficient_answer', 'reject': 'reject'})
        else:
            builder.add_edge("retrieve", "evidence_sufficient")
        builder.add_conditional_edges(
            "evidence_sufficient",
            self._route_evidence,
            {
                "generate": "generate_answer",
                "insufficient": "insufficient_answer",
                "reject": "reject",
            },
        )
        builder.add_edge("insufficient_answer", END)
        builder.add_conditional_edges(
            "generate_answer",
            self._route_generation,
            {"verify": "verify_answer", "reject": "reject", "insufficient": "insufficient_answer"},
        )
        builder.add_conditional_edges(
            "verify_answer",
            self._route_verification,
            {
                "finalize": "finalize",
                "rewrite": "rewrite_once",
                "reject": "reject",
            },
        )
        builder.add_conditional_edges(
            "rewrite_once",
            self._route_generation,
            {"verify": "verify_answer", "reject": "reject"},
        )
        builder.add_edge("finalize", END)
        builder.add_edge("reject", END)
        return builder.compile()

    def _retrieve(self, state: QAState) -> dict[str, object]:
        try:
            plan = self._plan_query(state["question"]) if self._entity_aware else QAQueryPlan()
            base_hits = self._retriever.search(state["question"], top_k=state["top_k"])
            hits_by_id = {hit.memory_id: hit for hit in base_hits}
            ordered_ids: list[str] = []
            required_ids: list[str] = []

            if plan.mode == "composite":
                for subquery in plan.subqueries:
                    sub_hits = self._retriever.search(subquery, top_k=1)
                    if sub_hits:
                        hit = sub_hits[0]
                        hits_by_id[hit.memory_id] = hit
                        if hit.memory_id not in required_ids:
                            required_ids.append(hit.memory_id)
                        if hit.memory_id not in ordered_ids:
                            ordered_ids.append(hit.memory_id)

            entity_memories = []
            if plan.mode == "entity_overview":
                matched = set(plan.matched_people)
                entity_memories = [memory for memory in self._repository.list_memories()
                                   if matched.intersection(memory.people)]
                required_ids = [memory.memory_id for memory in entity_memories]
                ordered_ids.extend(required_ids)

            for hit in base_hits:
                if hit.memory_id not in ordered_ids:
                    ordered_ids.append(hit.memory_id)
            plan = plan.model_copy(update={"required_memory_ids": required_ids})
        except Exception as exception:
            code = classify_exception(exception)
            code = code.replace("model_", "embedding_", 1) if code != "model_call_failed" else "retrieval_failed"
            if isinstance(exception, (StorageError, sqlite3.Error)):
                code = "storage_error"
            return {
                "error": "Memory retrieval failed",
                "validation_result": QAValidationResult(
                    stage="evidence",
                    passed=False,
                    reason="Memory retrieval failed",
                    failure_code=code,
                    exception_type=type(exception).__name__,
                ),
            }

        records = {hit.memory_id: hit.memory for hit in hits_by_id.values()}
        records.update({memory.memory_id: memory for memory in entity_memories})
        evidence = [item for memory_id in ordered_ids
                    if (item := self._evidence_for_memory(records[memory_id])) is not None]
        visible_ids = {item.memory_id for item in evidence}
        plan = plan.model_copy(update={"required_memory_ids": [memory_id for memory_id in
                                    plan.required_memory_ids if memory_id in visible_ids]})
        return {
            "query_plan": plan,
            "retrieved_memory_ids": [item.memory_id for item in evidence],
            "selected_evidence": evidence,
        }

    def _plan_query(self, question: str) -> QAQueryPlan:
        normalized = self._normalize_text(question)
        people = sorted({person for memory in self._repository.list_memories()
                         for person in memory.people}, key=len, reverse=True)
        matched = [person for person in people if self._normalize_text(person) in normalized]
        if matched and any(term in question for term in _OVERVIEW_TERMS):
            return QAQueryPlan(mode="entity_overview", matched_people=matched)
        subqueries = self._decompose_question(question)
        if subqueries:
            return QAQueryPlan(mode="composite", matched_people=matched, subqueries=subqueries)
        return QAQueryPlan(matched_people=matched)

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())

    @staticmethod
    def _decompose_question(question: str) -> list[str]:
        if not any(term in question for term in _COMPOSITE_TERMS):
            return []
        parts = _COMPOSITE_SPLIT.split(question)
        cleaned: list[str] = []
        for part in parts:
            item = re.sub(r"^(?:기록된|기억 속)\s*", "", part.strip())
            item = re.sub(r"(?:을|를)?\s*(?:각각|모두|비교).*$", "", item).strip(" .?")
            if len(item) >= 2 and item not in cleaned:
                cleaned.append(item)
        return cleaned[:6] if len(cleaned) >= 2 else []

    def _evidence_for_memory(self, memory) -> QAEvidence | None:
        sources = self._repository.list_memory_sources(memory.memory_id)
        if not sources:
            return None
        return QAEvidence(memory_id=memory.memory_id, transcript_id=memory.transcript_id,
            title=memory.title, summary=memory.summary, emotion=memory.emotion,
            people=memory.people, location=memory.location, event_date=memory.event_date,
            uncertainty_notes=memory.uncertainty_notes,
            sources=[CitationRecord(memory_id=source.memory_id, transcript_id=source.transcript_id,
                segment_id=source.segment_id, start_offset=source.start_offset,
                end_offset=source.end_offset) for source in sources])

    def _assess_evidence(self, state: QAState) -> dict[str, object]:
        if state["error"] is not None:
            return {}
        evidence = state["selected_evidence"]
        if not evidence:
            return {
                "validation_result": QAValidationResult(
                    stage="evidence",
                    passed=False,
                    reason="No traceable retrieved evidence was found",
                )
            }
        try:
            assessment = _invoke_structured(
                self._models.evidence,
                [
                    SystemMessage(content=EVIDENCE_ASSESSMENT_SYSTEM_PROMPT),
                    HumanMessage(
                        content=build_evidence_input(state["question"], evidence, compact=self._compact,
                                                     query_plan=state["query_plan"])
                    ),
                ],
                EvidenceAssessment,
            )
            available = {item.memory_id: item for item in evidence}
            if any(
                memory_id not in available
                for memory_id in assessment.selected_memory_ids
            ):
                raise QAOutputError("Evidence model selected an unknown memory", code="unknown_memory_selected")
            selected = [
                available[memory_id]
                for memory_id in assessment.selected_memory_ids
            ]
            return {
                "selected_evidence": selected,
                "validation_result": QAValidationResult(
                    stage="evidence",
                    passed=assessment.sufficient,
                    reason=assessment.reason,
                ),
            }
        except QAError as exception:
            return {
                "error": "Evidence assessment failed",
                "validation_result": QAValidationResult(
                    stage="evidence",
                    passed=False,
                    reason="Evidence assessment failed",
                    failure_code=exception.code,
                    exception_type=type(exception.__cause__ or exception).__name__,
                    schema_issues=exception.schema_issues,
                ),
            }

    def _generate_answer(self, state: QAState) -> dict[str, object]:
        try:
            draft = _invoke_structured(
                self._models.answer,
                [
                    SystemMessage(content=COMBINED_ANSWER_SYSTEM_PROMPT if self._combined else GROUNDED_ANSWER_SYSTEM_PROMPT),
                    HumanMessage(
                        content=build_evidence_input(
                            state["question"],
                            state["selected_evidence"],
                            compact=self._compact,
                            query_plan=state["query_plan"],
                        )
                    ),
                ],
                AnswerProposal if self._combined else GroundedAnswerDraft,
            )
            if isinstance(draft, AnswerProposal):
                if not draft.claims:
                    return {'answer_draft': None, 'validation_result': QAValidationResult(
                        stage='evidence', passed=False, reason=draft.reason, failure_code='insufficient_evidence')}
                draft = GroundedAnswerDraft(claims=draft.claims)
            return self._draft_update(draft, state["selected_evidence"])
        except QAError as exception:
            return {
                "error": "Answer generation failed",
                "validation_result": QAValidationResult(
                    stage="answer",
                    passed=False,
                    reason="Answer generation failed",
                    failure_code=exception.code,
                    exception_type=type(exception.__cause__ or exception).__name__,
                    schema_issues=exception.schema_issues,
                ),
            }

    def _verify_answer(self, state: QAState) -> dict[str, object]:
        draft = state["answer_draft"]
        if draft is None:
            return {
                "error": "Answer draft is missing",
                "validation_result": QAValidationResult(
                    stage="answer",
                    passed=False,
                    reason="Answer draft is missing",
                ),
            }
        structural_error = _draft_validation_error(
            draft,
            state["selected_evidence"],
        )
        if structural_error is None and self._combined:
            structural_error = self._grounding_error(draft, state['selected_evidence'])
        if structural_error is not None:
            return {
                "validation_result": QAValidationResult(
                    stage="answer",
                    passed=False,
                    reason=structural_error,
                )
            }
        cited_ids = {memory_id for claim in draft.claims for memory_id in claim.memory_ids}
        missing_ids = [memory_id for memory_id in state["query_plan"].required_memory_ids
                       if memory_id not in cited_ids]
        if missing_ids:
            return {"validation_result": QAValidationResult(stage="answer", passed=False,
                reason="Required retrieved memories were omitted from the answer",
                failure_code="incomplete_answer")}
        try:
            verification = _invoke_structured(
                self._models.verification,
                [
                    SystemMessage(content=ANSWER_VERIFICATION_SYSTEM_PROMPT),
                    HumanMessage(
                        content=build_verification_input(
                            state["question"],
                            state["selected_evidence"],
                            draft,
                            compact=self._compact,
                            query_plan=state["query_plan"],
                        )
                    ),
                ],
                AnswerVerification,
            )
            if any(
                index >= len(draft.claims)
                for index in verification.unsupported_claim_indexes
            ):
                raise QAOutputError("Verifier returned an invalid claim index", code="verification_index_invalid")
            required_ids = set(state["query_plan"].required_memory_ids)
            if any(memory_id not in required_ids for memory_id in verification.missing_required_memory_ids):
                raise QAOutputError("Verifier returned an unknown required memory", code="verification_memory_invalid")
            if not verification.passed:
                return {
                    "validation_result": QAValidationResult(
                        stage="answer",
                        passed=False,
                        reason=verification.reason,
                        failure_code="incomplete_answer" if verification.missing_required_memory_ids else None,
                    )
                }
            grounding_error = self._grounding_error(
                draft,
                state["selected_evidence"],
            )
            if grounding_error is not None:
                return {
                    "validation_result": QAValidationResult(
                        stage="answer",
                        passed=False,
                        reason=grounding_error,
                    )
                }
            return {
                "validation_result": QAValidationResult(
                    stage="answer",
                    passed=True,
                    reason=verification.reason,
                )
            }
        except QAError as exception:
            return {
                "error": "Answer verification failed",
                "validation_result": QAValidationResult(
                    stage="answer",
                    passed=False,
                    reason="Answer verification failed",
                    failure_code=exception.code,
                    exception_type=type(exception.__cause__ or exception).__name__,
                    schema_issues=exception.schema_issues,
                ),
            }

    def _rewrite_once(self, state: QAState) -> dict[str, object]:
        draft = state["answer_draft"]
        if draft is None:
            return {"error": "Answer draft is missing", "retry_count": 1}
        try:
            rewritten = _invoke_structured(
                self._models.rewrite,
                [
                    SystemMessage(content=ANSWER_REWRITE_SYSTEM_PROMPT),
                    HumanMessage(
                        content=build_rewrite_input(
                            state["question"],
                            state["selected_evidence"],
                            draft,
                            state["validation_result"].reason,
                            compact=self._compact,
                            query_plan=state["query_plan"],
                        )
                    ),
                ],
                GroundedAnswerDraft,
            )
            update = self._draft_update(
                rewritten,
                state["selected_evidence"],
            )
            update["retry_count"] = 1
            update["error"] = None
            return update
        except QAError as exception:
            return {
                "error": "Answer rewrite failed",
                "retry_count": 1,
                "validation_result": QAValidationResult(
                    stage="answer",
                    passed=False,
                    reason="Answer rewrite failed",
                    failure_code=exception.code,
                    exception_type=type(exception.__cause__ or exception).__name__,
                    schema_issues=exception.schema_issues,
                ),
            }

    def _grounding_error(
        self,
        draft: GroundedAnswerDraft,
        evidence: list[QAEvidence],
    ) -> str | None:
        """Re-check a verifier-approved draft against the transcript itself.

        ``_draft_validation_error`` only proves the cited IDs were retrieved, and
        ``AnswerVerification`` is the model grading its own answer. Neither stops
        a real memory_id from carrying an invented sentence, so the tokens that
        can be checked without a model - numbers and known people - are checked
        against the cited memories and their verbatim transcript excerpts here.
        """

        grounding: dict[str, str] = {}
        for item in evidence:
            text = self._grounding_text(item)
            if text is None:
                return UNTRACEABLE_CITATION
            grounding[item.memory_id] = text
        people_by_memory = {item.memory_id: set(item.people) for item in evidence}
        known_people = {
            person
            for people in people_by_memory.values()
            for person in people
            if len(person) >= _SHORTEST_CHECKED_NAME
        }

        for claim in draft.claims:
            cited_text = "\n".join(
                grounding[memory_id] for memory_id in claim.memory_ids
            )
            for number in _DIGIT_RUN.findall(_strip_memory_ids(claim.text)):
                if not _is_grounded_number(number, cited_text):
                    return (
                        f"Answer claim used the number {number} "
                        "that its cited memories do not contain"
                    )
            cited_people = {
                person
                for memory_id in claim.memory_ids
                for person in people_by_memory[memory_id]
            }
            for person in sorted(known_people - cited_people):
                if any(person in cited for cited in cited_people):
                    continue
                if person in claim.text:
                    return (
                        f"Answer claim named {person} without citing the memory "
                        "that records the name"
                    )
        return None

    def _grounding_text(self, item: QAEvidence) -> str | None:
        """Join one memory's stored fields with its verbatim transcript excerpts."""

        excerpts: list[str] = []
        for citation in item.sources:
            segment = self._repository.get_segment(citation.segment_id)
            if segment is None:
                return None
            start = max(citation.start_offset - segment.start_offset, 0)
            end = max(citation.end_offset - segment.start_offset, start)
            excerpts.append(segment.content[start:end])
        return "\n".join(
            [
                item.title,
                item.summary,
                item.emotion or "",
                item.location or "",
                item.event_date or "",
                item.uncertainty_notes or "",
                *item.people,
                *excerpts,
            ]
        )

    def _draft_update(
        self,
        draft: GroundedAnswerDraft,
        evidence: list[QAEvidence],
    ) -> dict[str, object]:
        structural_error = _draft_validation_error(draft, evidence)
        if structural_error is not None:
            raise QAOutputError(structural_error, code="citation_outside_evidence")
        answer, citations = _render_draft(draft, evidence)
        return {
            "answer_draft": draft,
            "draft_answer": answer,
            "citations": citations,
            "validation_result": QAValidationResult(
                stage="answer",
                passed=False,
                reason="Answer is awaiting verification",
            ),
        }

    @staticmethod
    def _route_evidence(state: QAState) -> str:
        if state["error"] is not None:
            return "reject"
        return (
            "generate"
            if state["validation_result"].passed
            else "insufficient"
        )

    @staticmethod
    def _route_generation(state: QAState) -> str:
        if state['error'] is not None:
            return 'reject'
        return 'insufficient' if state['answer_draft'] is None else 'verify'

    @staticmethod
    def _route_retrieved(state: QAState) -> str:
        if state['error'] is not None:
            return 'reject'
        return 'generate' if state['selected_evidence'] else 'insufficient'

    def _route_verification(self, state: QAState) -> str:
        if state["validation_result"].passed:
            return "finalize"
        if state["error"] is not None or state["retry_count"] >= self._max_rewrites:
            return "reject"
        return "rewrite"

    @staticmethod
    def _insufficient_answer(_state: QAState) -> dict[str, object]:
        return {
            "final_answer": INSUFFICIENT_ANSWER,
            "citations": [],
        }

    @staticmethod
    def _finalize(state: QAState) -> dict[str, object]:
        return {"final_answer": state["draft_answer"]}

    @staticmethod
    def _reject(_state: QAState) -> dict[str, object]:
        validation = _state["validation_result"]
        message = REJECTED_ANSWER
        if validation.failure_code in MESSAGES:
            stages = {"evidence_sufficient": "근거 충분성 판단", "generate_answer": "답변 생성",
                      "verify_answer": "답변 검증", "rewrite_once": "답변 재작성", "retrieve": "기억 검색"}
            steps = _state.get("diagnostic_steps", [])
            failed = next((step for step in reversed(steps) if step.failure_code), None)
            stage = stages.get(failed.node, "질문 처리") if failed else "질문 처리"
            message = f"{stage} 단계: {MESSAGES[validation.failure_code]}"
        return {
            "final_answer": message,
            "citations": [],
        }

    def _persist_exchange(self, result: QAResult) -> str:
        if self._repository.get_conversation_session(result.session_id) is None:
            self._repository.create_conversation_session(
                ConversationSessionCreate(
                    session_id=result.session_id,
                    title=result.question[:80],
                )
            )
        user_message_id = f"msg_{uuid4().hex}"
        self._repository.add_conversation_message(
            ConversationMessageCreate(
                message_id=user_message_id,
                session_id=result.session_id,
                role="user",
                content=result.question,
            )
        )
        self._repository.add_conversation_message(
            ConversationMessageCreate(
                message_id=f"msg_{uuid4().hex}",
                session_id=result.session_id,
                role="assistant",
                content=result.final_answer,
                citations=result.citations,
            )
        )
        return user_message_id


def _invoke_structured(
    model: StructuredQAModel,
    messages: list[object],
    schema: type[StructuredOutputT],
) -> StructuredOutputT:
    try:
        output = model.invoke(messages)
    except Exception as exception:
        failure = QAError("Q&A model call failed", code=classify_exception(exception))
        failure.schema_issues = _schema_issues(exception)
        raise failure from exception
    if schema is EvidenceAssessment and isinstance(output, EvidenceSelection):
        return cast(StructuredOutputT, EvidenceAssessment(
            sufficient=bool(output.selected_memory_ids), reason=output.reason,
            selected_memory_ids=output.selected_memory_ids,
        ))
    if isinstance(output, schema):
        return output
    if not isinstance(output, Mapping):
        raise QAOutputError("Model returned an invalid output envelope", code="output_envelope_invalid")
    raw = output.get("raw")
    metadata = getattr(raw, "response_metadata", {})
    if isinstance(metadata, Mapping) and metadata.get("finish_reason") == "length":
        raise QAOutputError("Model output was truncated", code="output_truncated")
    parsing_error = output.get("parsing_error")
    if parsing_error is not None:
        if isinstance(parsing_error, BaseException):
            failure = QAOutputError("Model output did not match its schema", code=classify_exception(parsing_error))
            failure.schema_issues = _schema_issues(parsing_error)
            raise failure from parsing_error
        raise QAOutputError("Model output did not match its schema", code="output_schema_invalid")
    parsed = output.get("parsed")
    if parsed is None:
        raw = output.get("raw")
        additional_kwargs = getattr(raw, "additional_kwargs", {})
        refusal = (
            additional_kwargs.get("refusal")
            if isinstance(additional_kwargs, Mapping)
            else None
        )
        message = "Model refused the Q&A request" if refusal else "No parsed output"
        raise QAOutputError(message, code="model_refusal" if refusal else "output_missing")
    try:
        if schema is EvidenceAssessment and isinstance(parsed, EvidenceSelection):
            return cast(StructuredOutputT, EvidenceAssessment(
                sufficient=bool(parsed.selected_memory_ids), reason=parsed.reason,
                selected_memory_ids=parsed.selected_memory_ids,
            ))
        return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
    except ValidationError as exception:
        failure = QAOutputError("Model output did not match its schema", code="output_schema_invalid")
        failure.schema_issues = _schema_issues(exception)
        raise failure from exception


def _schema_issues(exception: BaseException) -> list[QASchemaIssue]:
    """Retain schema fields and rule names, never input values or error text."""
    if not isinstance(exception, ValidationError):
        return []
    allowed = {"sufficient", "reason", "selected_memory_ids", "claims", "text",
               "memory_ids", "passed", "unsupported_claim_indexes"}
    rules = {
        "sufficient evidence requires selected_memory_ids": "selection_required_when_sufficient",
        "insufficient evidence cannot select memories": "selection_empty_when_insufficient",
        "passed verification cannot list unsupported claims": "passed_verification_has_no_unsupported_claims",
    }
    issues = []
    for error in exception.errors():
        location = error["loc"]
        safe = [str(item) if isinstance(item, int) or item in allowed else "<unknown_field>" for item in location]
        rule = rules.get(str(error.get("ctx", {}).get("error", "")))
        issues.append(QASchemaIssue(field=".".join(safe) or "<model>", error_type=error["type"], rule=rule))
    return issues


_MEMORY_ID_TOKEN = re.compile(r"\[?mem_[0-9A-Za-z]+[^\s\]]*\]?")


def _strip_memory_ids(text: str) -> str:
    """Remove memory IDs a model copied into prose; citations carry them already."""

    return " ".join(_MEMORY_ID_TOKEN.sub(" ", text).split())


def _is_grounded_number(number: str, grounding: str) -> bool:
    """Accept a digit run that occurs in the grounding text, zero-padded or not."""

    if number in grounding:
        return True
    unpadded = number.lstrip("0")
    return bool(unpadded) and unpadded in grounding


def _draft_validation_error(
    draft: GroundedAnswerDraft,
    evidence: list[QAEvidence],
) -> str | None:
    available_ids = {item.memory_id for item in evidence}
    for claim in draft.claims:
        if any(memory_id not in available_ids for memory_id in claim.memory_ids):
            return "Answer cited a memory outside the selected evidence"
    return None


def _render_draft(
    draft: GroundedAnswerDraft,
    evidence: list[QAEvidence],
) -> tuple[str, list[CitationRecord]]:
    evidence_by_id = {item.memory_id: item for item in evidence}
    answer_lines: list[str] = []
    citations: list[CitationRecord] = []
    citation_keys: set[tuple[object, ...]] = set()
    for claim in draft.claims:
        markers: list[str] = []
        for memory_id in claim.memory_ids:
            source = evidence_by_id[memory_id].sources[0]
            markers.append(
                f"[{source.memory_id}|{source.transcript_id}:"
                f"{source.start_offset}-{source.end_offset}]"
            )
            key = (
                source.memory_id,
                source.transcript_id,
                source.segment_id,
                source.start_offset,
                source.end_offset,
            )
            if key not in citation_keys:
                citation_keys.add(key)
                citations.append(source)
        answer_lines.append(f"{_strip_memory_ids(claim.text)} {' '.join(markers)}")
    return "\n".join(answer_lines), citations
