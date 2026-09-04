import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from loguru import logger


class RequestTimingAndCorrelationMiddleware(BaseHTTPMiddleware):
    """Middleware for injecting X-Request-ID and recording latency."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"Request failed | method={request.method} path={request.url.path} "
                f"req_id={request_id} duration={duration_ms:.2f}ms error={str(exc)}"
            )
            raise exc

        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

        logger.info(
            f"Request completed | status={response.status_code} method={request.method} "
            f"path={request.url.path} req_id={request_id} duration={duration_ms:.2f}ms"
        )
        return response
