"""Bounded LangGraph workflow for evidence-grounded question answering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from backend.app.models.qa import (
    AnswerVerification,
    EvidenceAssessment,
    GroundedAnswerDraft,
    QAEvidence,
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
    build_evidence_input,
    build_rewrite_input,
    build_verification_input,
)
from backend.app.storage.models import (
    CitationRecord,
    ConversationMessageCreate,
    ConversationSessionCreate,
)
from backend.app.storage.repository import SQLiteRepository

DEFAULT_OPENAI_MODEL = "gpt-5.6-sol"
INSUFFICIENT_ANSWER = "질문에 답할 수 있는 충분한 근거를 찾지 못했습니다."
REJECTED_ANSWER = "근거 검증을 통과하지 못해 답변할 수 없습니다."

StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


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


class QAOutputError(QAError):
    """A structured model response failed validation."""


def build_openai_qa_models(
    model_name: str = DEFAULT_OPENAI_MODEL,
) -> QAModels:
    """Create strict OpenAI Structured Output models for all Q&A stages."""

    model = ChatOpenAI(model=model_name)
    options = {
        "method": "json_schema",
        "include_raw": True,
        "strict": True,
    }
    return QAModels(
        evidence=model.with_structured_output(EvidenceAssessment, **options),
        answer=model.with_structured_output(GroundedAnswerDraft, **options),
        verification=model.with_structured_output(AnswerVerification, **options),
        rewrite=model.with_structured_output(GroundedAnswerDraft, **options),
    )


class GroundedQAService:
    """Execute, validate, and persist one bounded grounded-Q&A graph."""

    def __init__(
        self,
        repository: SQLiteRepository,
        retriever: HybridRetriever,
        models: QAModels,
    ) -> None:
        self._repository = repository
        self._retriever = retriever
        self._models = models
        self.graph = self._build_graph()

    def answer_question(
        self,
        *,
        session_id: str,
        question: str,
        top_k: int = 5,
    ) -> QAResult:
        """Run the graph, save the exchange, and return the safe final result."""

        if not session_id.strip():
            raise ValueError("session_id must not be blank")
        if not question.strip():
            raise ValueError("question must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        initial_state: QAState = {
            "session_id": session_id,
            "question": question,
            "top_k": top_k,
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
        }
        final_state = cast(QAState, self.graph.invoke(initial_state))
        result = QAResult(
            session_id=final_state["session_id"],
            question=final_state["question"],
            retrieved_memory_ids=final_state["retrieved_memory_ids"],
            final_answer=final_state["final_answer"],
            citations=final_state["citations"],
            validation_result=final_state["validation_result"],
            retry_count=final_state["retry_count"],
            error=final_state["error"],
        )
        self._persist_exchange(result)
        return result

    def _build_graph(self) -> Any:
        builder = StateGraph(QAState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("evidence_sufficient", self._assess_evidence)
        builder.add_node("insufficient_answer", self._insufficient_answer)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("verify_answer", self._verify_answer)
        builder.add_node("rewrite_once", self._rewrite_once)
        builder.add_node("finalize", self._finalize)
        builder.add_node("reject", self._reject)

        builder.add_edge(START, "retrieve")
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
            {"verify": "verify_answer", "reject": "reject"},
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
            hits = self._retriever.search(
                state["question"],
                top_k=state["top_k"],
            )
        except Exception:
            return {
                "error": "Memory retrieval failed",
                "validation_result": QAValidationResult(
                    stage="evidence",
                    passed=False,
                    reason="Memory retrieval failed",
                ),
            }

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
        return {
            "retrieved_memory_ids": [hit.memory_id for hit in hits],
            "selected_evidence": evidence,
        }

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
                        content=build_evidence_input(state["question"], evidence)
                    ),
                ],
                EvidenceAssessment,
            )
            available = {item.memory_id: item for item in evidence}
            if any(
                memory_id not in available
                for memory_id in assessment.selected_memory_ids
            ):
                raise QAOutputError("Evidence model selected an unknown memory")
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
        except QAError:
            return {
                "error": "Evidence assessment failed",
                "validation_result": QAValidationResult(
                    stage="evidence",
                    passed=False,
                    reason="Evidence assessment failed",
                ),
            }

    def _generate_answer(self, state: QAState) -> dict[str, object]:
        try:
            draft = _invoke_structured(
                self._models.answer,
                [
                    SystemMessage(content=GROUNDED_ANSWER_SYSTEM_PROMPT),
                    HumanMessage(
                        content=build_evidence_input(
                            state["question"],
                            state["selected_evidence"],
                        )
                    ),
                ],
                GroundedAnswerDraft,
            )
            return self._draft_update(draft, state["selected_evidence"])
        except QAError:
            return {
                "error": "Answer generation failed",
                "validation_result": QAValidationResult(
                    stage="answer",
                    passed=False,
                    reason="Answer generation failed",
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
        if structural_error is not None:
            return {
                "validation_result": QAValidationResult(
                    stage="answer",
                    passed=False,
                    reason=structural_error,
                )
            }
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
                        )
                    ),
                ],
                AnswerVerification,
            )
            if any(
                index >= len(draft.claims)
                for index in verification.unsupported_claim_indexes
            ):
                raise QAOutputError("Verifier returned an invalid claim index")
            return {
                "validation_result": QAValidationResult(
                    stage="answer",
                    passed=verification.passed,
                    reason=verification.reason,
                )
            }
        except QAError:
            return {
                "error": "Answer verification failed",
                "validation_result": QAValidationResult(
                    stage="answer",
                    passed=False,
                    reason="Answer verification failed",
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
        except QAError:
            return {
                "error": "Answer rewrite failed",
                "retry_count": 1,
                "validation_result": QAValidationResult(
                    stage="answer",
                    passed=False,
                    reason="Answer rewrite failed",
                ),
            }

    def _draft_update(
        self,
        draft: GroundedAnswerDraft,
        evidence: list[QAEvidence],
    ) -> dict[str, object]:
        structural_error = _draft_validation_error(draft, evidence)
        if structural_error is not None:
            raise QAOutputError(structural_error)
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
        return "reject" if state["error"] is not None else "verify"

    @staticmethod
    def _route_verification(state: QAState) -> str:
        if state["validation_result"].passed:
            return "finalize"
        if state["error"] is not None or state["retry_count"] >= 1:
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
        return {
            "final_answer": REJECTED_ANSWER,
            "citations": [],
        }

    def _persist_exchange(self, result: QAResult) -> None:
        if self._repository.get_conversation_session(result.session_id) is None:
            self._repository.create_conversation_session(
                ConversationSessionCreate(
                    session_id=result.session_id,
                    title=result.question[:80],
                )
            )
        self._repository.add_conversation_message(
            ConversationMessageCreate(
                message_id=f"msg_{uuid4().hex}",
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


def _invoke_structured(
    model: StructuredQAModel,
    messages: list[object],
    schema: type[StructuredOutputT],
) -> StructuredOutputT:
    try:
        output = model.invoke(messages)
    except Exception as exception:
        raise QAError("Q&A model call failed") from exception
    if isinstance(output, schema):
        return output
    if not isinstance(output, Mapping):
        raise QAOutputError("Model returned an invalid output envelope")
    parsing_error = output.get("parsing_error")
    if parsing_error is not None:
        if isinstance(parsing_error, BaseException):
            raise QAOutputError("Model output did not match its schema") from parsing_error
        raise QAOutputError("Model output did not match its schema")
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
        raise QAOutputError(message)
    try:
        return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
    except ValidationError as exception:
        raise QAOutputError("Model output did not match its schema") from exception


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
        answer_lines.append(f"{claim.text.strip()} {' '.join(markers)}")
    return "\n".join(answer_lines), citations
