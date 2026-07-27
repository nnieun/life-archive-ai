"""Chronological memory timeline API."""

from __future__ import annotations

from datetime import date
from functools import lru_cache

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, model_validator

from backend.app.core.config import get_settings
from backend.app.models.timeline import TimelineResult
from backend.app.services.timeline import TimelineService
from backend.app.storage.database import SQLiteDatabase
from backend.app.storage.repository import SQLiteRepository

router = APIRouter(tags=["timeline"])


class TimelineRequest(BaseModel):
    """Optional inclusive date range for timeline filtering."""

    model_config = ConfigDict(extra="forbid")

    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_range(self) -> TimelineRequest:
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError("start_date must not follow end_date")
        return self


@lru_cache(maxsize=1)
def get_timeline_service() -> TimelineService:
    """Build the SQLite-backed timeline service lazily."""

    database = SQLiteDatabase(get_settings().sqlite_database_path)
    database.initialize()
    return TimelineService(SQLiteRepository(database))


@router.post("/timeline", response_model=TimelineResult)
def timeline(
    request: TimelineRequest,
    service: TimelineService = Depends(get_timeline_service),
) -> TimelineResult:
    """Return dated and unknown-date memories with citations."""

    return service.get_timeline(
        start_date=request.start_date,
        end_date=request.end_date,
    )
