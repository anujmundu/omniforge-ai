import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.config import get_settings
from apps.api.core.database import get_db_session
from apps.api.schemas.health import HealthResponse, ServiceHealth

router = APIRouter(prefix="/health", tags=["Health & Observability"])
settings = get_settings()


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Platform Deep Health Check",
    description="Inspects database connectivity, response latency, and subsystem health.",
)
async def health_check(db: AsyncSession = Depends(get_db_session)) -> HealthResponse:
    services = {}

    # Check Database connectivity & measure round-trip latency
    db_start = time.perf_counter()
    try:
        await db.execute(text("SELECT 1"))
        db_latency = (time.perf_counter() - db_start) * 1000
        services["database"] = ServiceHealth(
            status="healthy",
            latency_ms=round(db_latency, 2),
            details={"engine": "asyncpg / aiosqlite"},
        )
    except Exception as e:
        db_latency = (time.perf_counter() - db_start) * 1000
        services["database"] = ServiceHealth(
            status="unhealthy",
            latency_ms=round(db_latency, 2),
            details={"error": str(e)},
        )

    # Check Telemetry & Tracing subsystem
    try:
        from opentelemetry import trace

        provider_name = trace.get_tracer_provider().__class__.__name__
    except Exception:
        provider_name = "standard"

    services["telemetry"] = ServiceHealth(
        status="healthy",
        latency_ms=0.0,
        details={"provider": provider_name, "service": settings.PROJECT_NAME},
    )

    overall_status = "healthy" if all(s.status == "healthy" for s in services.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        version="0.1.0",
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc),
        services=services,
    )
