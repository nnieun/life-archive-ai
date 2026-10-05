"""Persisted asynchronous chat request state."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from backend.app.models.qa import QAResult


class ChatJob(BaseModel):
    job_id: str
    session_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: datetime
    result: QAResult | None = None
    error: str | None = None
    failure_code: str | None = None
    failure_stage: str | None = None
    progress: dict = Field(default_factory=dict)
