"""FastAPI application entry point."""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from backend.app.api.chat import get_chat_job_store


@asynccontextmanager
async def lifespan(application: FastAPI):
    get_chat_job_store().recover_interrupted()
    from backend.app.api.memories import get_ingestion_job_store
    get_ingestion_job_store().recover_interrupted()
    yield
    from backend.app.api.chat import get_qa_service
    if get_qa_service.cache_info().currsize:
        get_qa_service().close()
        get_qa_service.cache_clear()

from backend.app.api.autobiographies import router as autobiographies_router
from backend.app.api.chat import router as chat_router
from backend.app.api.health import router as health_router
from backend.app.api.memories import router as memories_router
from backend.app.api.memory_gaps import router as memory_gaps_router
from backend.app.api.privacy import router as privacy_router
from backend.app.api.timeline import router as timeline_router
from backend.app.core.config import get_settings
from backend.app.core.errors import register_error_handlers
from backend.app.core.request_context import RequestContextMiddleware


def create_app() -> FastAPI:
    """Create and configure the backend application."""
    settings = get_settings()
    application = FastAPI(
        lifespan=lifespan,
        title=settings.app_name,
        version=settings.app_version,
        description="Memory-centric retrieval and grounded generation API",
    )
    application.add_middleware(RequestContextMiddleware)
    register_error_handlers(application)
    application.include_router(health_router, prefix=settings.api_prefix)
    application.include_router(memories_router, prefix=settings.api_prefix)
    application.include_router(memory_gaps_router, prefix=settings.api_prefix)
    application.include_router(privacy_router, prefix=settings.api_prefix)
    application.include_router(chat_router, prefix=settings.api_prefix)
    application.include_router(timeline_router, prefix=settings.api_prefix)
    application.include_router(autobiographies_router, prefix=settings.api_prefix)
    return application


app = create_app()
