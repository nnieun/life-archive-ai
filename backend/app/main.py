"""FastAPI application entry point."""

from fastapi import FastAPI

from backend.app.api.health import router as health_router
from backend.app.core.config import get_settings
from backend.app.core.errors import register_error_handlers


def create_app() -> FastAPI:
    """Create and configure the backend application."""
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Memory-centric retrieval and grounded generation API",
    )
    register_error_handlers(application)
    application.include_router(health_router, prefix=settings.api_prefix)
    return application


app = create_app()
