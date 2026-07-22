"""Health-check endpoint."""

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.app.core.config import Settings, get_settings

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    """Public service health information."""

    status: Literal["ok"] = "ok"
    service: str
    version: str


@router.get("/health", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Report whether the API process is ready to accept requests."""
    return HealthResponse(service=settings.app_name, version=settings.app_version)
