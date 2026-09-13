"""Read-only local tools available to the memory reconstruction agent."""

from __future__ import annotations

from typing import Annotated, Protocol
from urllib.parse import urlparse

from duckduckgo_search import DDGS

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from pydantic import Field

from backend.app.models.gap import (
    MemoryGapSearchSource,
    MemoryGapSearchSourceType,
    MemoryGapStatus,
    MemoryGapToolPayload,
)
from backend.app.models.retrieval import RetrievalHit
from backend.app.services.retrieval import tokenize_for_bm25
from backend.app.storage.repository import SQLiteRepository

MAX_SEARCH_RESULTS = 10


def search_public_web(query: str, *, top_k: int = 5) -> list[MemoryGapSearchSource]:
    """Search public web pages for an explicit, user-approved query."""
    sources: list[MemoryGapSearchSource] = []
    with DDGS() as client:
        for item in client.text(query, max_results=top_k):
            url = item.get("href") or item.get("url")
            if not url:
                continue
            title = item.get("title") or url
            body = item.get("body") or ""
            sources.append(
                MemoryGapSearchSource(
                    source_id=f"web:{url}",
                    source_type=MemoryGapSearchSourceType.EXTERNAL,
                    title=title,
                    # Search engines often put the business name only in the
                    # result title, so retain title and snippet together as
                    # one verifiable evidence block.
                    content=f"{title}\n{body}".strip(),
                    score=0.5,
                    url=url,
                    source_domain=urlparse(url).netloc,
                )
            )
    return sources


class GapMemoryRetriever(Protocol):
    """Minimal hybrid retrieval interface required by the memory tool."""

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        """Return active SQLite-backed memory hits."""


def build_memory_gap_tools(
    repository: SQLiteRepository,
    retriever: GapMemoryRetriever,
) -> tuple[BaseTool, ...]:
    """Build the bounded local-only tools exposed to the LLM."""

    @tool("search_memory")
    def search_memory(
        query: Annotated[str, Field(min_length=1, max_length=2000)],
        runtime: ToolRuntime,
        top_k: Annotated[int, Field(ge=1, le=MAX_SEARCH_RESULTS)] = 5,
    ) -> str:
        """Search structured memories first; returns evidence and never writes data."""

        hits = retriever.search(query, top_k=top_k)
        sources = [
            _memory_source(hit, rank)
            for rank, hit in enumerate(hits, start=1)
        ]
        return MemoryGapToolPayload(
            tool_name="search_memory",
            query=query,
            sources=sources,
        ).model_dump_json()

    @tool("search_uploaded_documents")
    def search_uploaded_documents(
        query: Annotated[str, Field(min_length=1, max_length=2000)],
        runtime: ToolRuntime,
        top_k: Annotated[int, Field(ge=1, le=MAX_SEARCH_RESULTS)] = 5,
    ) -> str:
        """Search immutable uploaded transcript chunks; returns no local file paths."""

        query_tokens = set(tokenize_for_bm25(query))
        ranked: list[tuple[float, str, MemoryGapSearchSource]] = []
        for transcript in repository.list_transcripts():
            for segment in repository.list_segments(transcript.transcript_id):
                score = _token_overlap(query_tokens, segment.content)
                if score <= 0.0:
                    continue
                source = MemoryGapSearchSource(
                    source_id=segment.segment_id,
                    source_type=MemoryGapSearchSourceType.TRANSCRIPT_SEGMENT,
                    title=transcript.filename,
                    content=segment.content,
                    score=score,
                    transcript_id=transcript.transcript_id,
                )
                ranked.append((score, segment.segment_id, source))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return MemoryGapToolPayload(
            tool_name="search_uploaded_documents",
            query=query,
            sources=[item[2] for item in ranked[:top_k]],
        ).model_dump_json()

    @tool("search_web")
    def search_web(
        query: Annotated[str, Field(min_length=1, max_length=500)],
        runtime: ToolRuntime,
        top_k: Annotated[int, Field(ge=1, le=5)] = 5,
    ) -> str:
        """Search public web pages; results remain unconfirmed candidates."""
        if not runtime.state.get("web_search_consent", False):
            return MemoryGapToolPayload(
                tool_name="search_web", query=query, sources=[]
            ).model_dump_json()
        sources = search_public_web(query, top_k=top_k)
        return MemoryGapToolPayload(
            tool_name="search_web", query=query, sources=sources,
        ).model_dump_json()

    @tool("search_memory_gaps")
    def search_memory_gaps(
        query: Annotated[str, Field(min_length=1, max_length=2000)],
        runtime: ToolRuntime,
        top_k: Annotated[int, Field(ge=1, le=MAX_SEARCH_RESULTS)] = 5,
    ) -> str:
        """Search similar unresolved gaps for related clues; never resolves either gap."""

        current_gap_id = _gap_id_from_runtime(runtime)
        query_tokens = set(tokenize_for_bm25(query))
        ranked: list[tuple[float, str, MemoryGapSearchSource]] = []
        for gap in repository.list_memory_gaps(
            statuses={
                MemoryGapStatus.OPEN,
                MemoryGapStatus.SEARCHING,
                MemoryGapStatus.CANDIDATE_FOUND,
                MemoryGapStatus.WAITING_USER,
            }
        ):
            if gap.gap_id == current_gap_id:
                continue
            content = "\n".join([gap.clue_text, *gap.user_clues])
            score = _token_overlap(query_tokens, content)
            if score <= 0.0:
                continue
            source = MemoryGapSearchSource(
                source_id=gap.gap_id,
                source_type=MemoryGapSearchSourceType.MEMORY_GAP,
                title=gap.gap_type.value,
                content=content,
                score=score,
                memory_id=gap.memory_id,
                event_date=gap.period_start,
                location=gap.location,
                people=gap.people,
            )
            ranked.append((score, gap.gap_id, source))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return MemoryGapToolPayload(
            tool_name="search_memory_gaps",
            query=query,
            sources=[item[2] for item in ranked[:top_k]],
        ).model_dump_json()

    @tool("request_more_clues")
    def request_more_clues(
        question: Annotated[str, Field(min_length=1, max_length=500)],
        runtime: ToolRuntime,
    ) -> str:
        """Ask the user for one focused clue when all local searches are insufficient."""

        del runtime
        return MemoryGapToolPayload(
            tool_name="request_more_clues",
            query=question,
            needs_user_input=True,
            question=question,
        ).model_dump_json()

    return (
        search_memory,
        search_uploaded_documents,
        search_web,
        search_memory_gaps,
        request_more_clues,
    )


def _memory_source(hit: RetrievalHit, rank: int) -> MemoryGapSearchSource:
    memory = hit.memory
    content = "\n".join(
        value
        for value in (
            memory.title,
            memory.summary,
            memory.location,
            memory.event_date,
            *memory.people,
            memory.uncertainty_notes,
        )
        if value
    )
    return MemoryGapSearchSource(
        source_id=memory.memory_id,
        source_type=MemoryGapSearchSourceType.MEMORY,
        title=memory.title,
        content=content,
        score=round(1.0 / rank, 6),
        memory_id=memory.memory_id,
        transcript_id=memory.transcript_id,
        event_date=memory.event_date,
        location=memory.location,
        people=memory.people,
    )


def _token_overlap(query_tokens: set[str], content: str) -> float:
    if not query_tokens:
        return 0.0
    content_tokens = set(tokenize_for_bm25(content))
    return round(
        min(1.0, len(query_tokens & content_tokens) / len(query_tokens)),
        6,
    )


def _gap_id_from_runtime(runtime: ToolRuntime) -> str:
    state = runtime.state
    if not isinstance(state, dict):
        raise ValueError("Memory-gap tools require graph state")
    gap_id = state.get("gap_id")
    if not isinstance(gap_id, str) or not gap_id.strip():
        raise ValueError("Memory-gap tools require a gap_id")
    return gap_id
