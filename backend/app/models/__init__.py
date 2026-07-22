"""Pydantic models shared by backend services."""

from backend.app.models.transcript import LoadedTranscript, TranscriptLoadRequest

__all__ = ["LoadedTranscript", "TranscriptLoadRequest"]
