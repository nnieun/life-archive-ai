"""Persisted upload processing state, scoped to a browser session."""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field
from backend.app.models.ingestion import IngestionResult


class IngestionJob(BaseModel):
    job_id: str
    session_id: str
    filename: str
    created_at: datetime
    status: Literal['queued', 'running', 'completed', 'failed', 'cancelled']
    progress: dict = Field(default_factory=dict)
    result: IngestionResult | None = None
    error: str | None = None
    failure_code: str | None = None

