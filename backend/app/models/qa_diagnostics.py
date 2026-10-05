"""Minimal diagnostics for unsuccessful QA requests."""

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class QASchemaIssue(BaseModel):
    field: str
    error_type: str
    rule: str | None = None


class QAStepDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: str
    elapsed_ms: float = Field(ge=0)
    passed: bool | None = None
    reason: str | None = None
    failure_code: str | None = None
    exception_type: str | None = None
    schema_issues: list[QASchemaIssue] = Field(default_factory=list)


class QAFailureDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diagnostic_id: str = Field(default_factory=lambda: f"qaf_{uuid4().hex}")
    session_id: str
    user_message_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider: str
    model: str
    status: Literal["insufficient", "rejected", "error"]
    retrieved_memory_ids: list[str]
    selected_memory_ids: list[str]
    retry_count: int = Field(ge=0, le=1)
    reason: str
    error: str | None = None
    elapsed_ms: float = Field(ge=0)
    steps: list[QAStepDiagnostic]
    failure_code: str | None = None
