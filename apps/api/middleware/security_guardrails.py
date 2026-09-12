"""FastAPI ASGI middleware for rate limiting and OWASP security response headers."""

from __future__ import annotations

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from security.rate_limiter import rate_limiter


class SecurityGuardrailMiddleware(BaseHTTPMiddleware):
    """Intercept incoming HTTP requests to enforce rate limit quotas and attach security headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Determine client identity (API key header, Authorization, or client IP)
        client_id = (
            request.headers.get("X-API-Key")
            or request.headers.get("X-Forwarded-For")
            or (request.client.host if request.client else "127.0.0.1")
        )

        tier = request.headers.get("X-Client-Tier", "free")

        # Skip rate limiting for static docs / openapi schema / test environments (unless explicitly testing rate limiting)
        path = request.url.path
        is_testing = os.getenv("ENVIRONMENT") == "testing" and not request.headers.get("X-Enforce-Rate-Limit")
        if not path.startswith("/docs") and not path.startswith("/openapi.json") and not is_testing:
            status = rate_limiter.check_and_consume(client_id, tier=tier)
            if status.is_limited:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "client_id": client_id,
                        "retry_after_seconds": status.reset_seconds,
                    },
                    headers={
                        "X-RateLimit-Limit": str(status.limit),
                        "X-RateLimit-Remaining": str(status.remaining),
                        "X-RateLimit-Reset": str(status.reset_seconds),
                        "Retry-After": str(status.reset_seconds),
                    },
                )

        response = await call_next(request)

        # Attach enterprise OWASP security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response
