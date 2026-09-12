from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from apps.api.core.config import get_settings
from apps.api.core.database import init_db
from apps.api.core.logging_config import setup_logging
from apps.api.middleware.request_id import RequestTimingAndCorrelationMiddleware
from apps.api.middleware.security_guardrails import SecurityGuardrailMiddleware
from apps.api.routers import (
    agents_router,
    auth_router,
    datasets_router,
    experiments_router,
    health_router,
    ml_router,
    mlops_router,
    nlp_router,
    observability_router,
    projects_router,
    rag_router,
    security_router,
    vision_router,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Setup global unified logging
    setup_logging(log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)
    logger.info(f"Initializing {settings.PROJECT_NAME} application state...")

    # Initialize SQLite / PostgreSQL Database tables asynchronously
    await init_db()
    logger.info("Database schema migration and tables initialized.")

    yield

    logger.info(f"Shutting down {settings.PROJECT_NAME} cleanly...")


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="0.1.0",
        description="Production-Grade Multimodal AI/ML Intelligence Platform API Gateway.",
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

    # Security Guardrails & Rate Limiting Middleware
    app.add_middleware(SecurityGuardrailMiddleware)

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
    app.include_router(mlops_router, prefix=settings.API_V1_STR)
    app.include_router(observability_router, prefix=settings.API_V1_STR)
    app.include_router(security_router, prefix=settings.API_V1_STR)

    @app.get("/metrics", tags=["Observability"], include_in_schema=False)
    async def prometheus_metrics() -> Response:
        """Root Prometheus exposition scrape endpoint."""
        from observability.metrics import metrics_registry

        return Response(
            content=metrics_registry.generate_prometheus_text(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

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
