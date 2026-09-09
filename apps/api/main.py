from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from apps.api.core.config import get_settings
from apps.api.core.database import init_db
from apps.api.core.logging_config import setup_logging
from apps.api.middleware.request_id import RequestTimingAndCorrelationMiddleware
from apps.api.routers import (
    agents_router,
    auth_router,
    datasets_router,
    experiments_router,
    health_router,
    ml_router,
    nlp_router,
    projects_router,
    rag_router,
    vision_router,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context for startup and graceful shutdown."""
    # Setup structured logging
    setup_logging()
    logger.info(f"Starting {settings.PROJECT_NAME} in [{settings.ENVIRONMENT}] mode")

    # Initialize Database tables
    try:
        await init_db()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")

    yield

    logger.info(f"Shutting down {settings.PROJECT_NAME}")


def create_application() -> FastAPI:
    """Factory function for FastAPI application."""
    app = FastAPI(
        title="OmniForge Intelligence Platform API",
        description=(
            "Production-Grade Multimodal AI/ML Intelligence Platform unifying "
            "Classical ML, Computer Vision, NLP, RAG, and Agentic Orchestration."
        ),
        version="0.1.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else [settings.CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID and Latency Middleware
    app.add_middleware(RequestTimingAndCorrelationMiddleware)

    # Standardized Exception Handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(f"Unhandled exception | req_id={request_id} error={str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": str(exc) if settings.DEBUG else "An unexpected error occurred. Please contact system admin.",
                "request_id": request_id,
            },
        )

    # Register API v1 Routers
    app.include_router(health_router, prefix=settings.API_V1_STR)
    app.include_router(auth_router, prefix=settings.API_V1_STR)
    app.include_router(projects_router, prefix=settings.API_V1_STR)
    app.include_router(datasets_router, prefix=settings.API_V1_STR)
    app.include_router(experiments_router, prefix=settings.API_V1_STR)
    app.include_router(ml_router, prefix=settings.API_V1_STR)
    app.include_router(vision_router, prefix=settings.API_V1_STR)
    app.include_router(nlp_router, prefix=settings.API_V1_STR)
    app.include_router(rag_router, prefix=settings.API_V1_STR)
    app.include_router(agents_router, prefix=settings.API_V1_STR)

    @app.get("/", tags=["Root"])
    async def root() -> dict:
        return {
            "platform": settings.PROJECT_NAME,
            "version": "0.1.0",
            "status": "online",
            "docs": "/docs",
            "health": f"{settings.API_V1_STR}/health",
        }

    return app


app = create_application()
