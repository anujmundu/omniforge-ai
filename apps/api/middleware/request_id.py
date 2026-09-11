import time
import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from observability.metrics import HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_TOTAL


class RequestTimingAndCorrelationMiddleware(BaseHTTPMiddleware):
    """Middleware for injecting X-Request-ID, recording latency, and emitting Prometheus telemetry."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_s = time.perf_counter() - start_time
            duration_ms = duration_s * 1000
            logger.error(
                f"Request failed | method={request.method} path={request.url.path} "
                f"req_id={request_id} duration={duration_ms:.2f}ms error={str(exc)}"
            )
            # Record Prometheus error telemetry
            try:
                endpoint_label = request.url.path
                HTTP_REQUESTS_TOTAL.inc(
                    labels={"method": request.method, "endpoint": endpoint_label, "status_code": "500"}
                )
                HTTP_REQUEST_DURATION_SECONDS.observe(
                    duration_s, labels={"method": request.method, "endpoint": endpoint_label}
                )
            except Exception:
                pass
            raise exc

        duration_s = time.perf_counter() - start_time
        duration_ms = duration_s * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

        logger.info(
            f"{request.method} {request.url.path} [{response.status_code}] "
            f"req_id={request_id} duration={duration_ms:.2f}ms"
        )

        # Record Prometheus telemetry
        try:
            endpoint_label = request.url.path
            status_str = str(response.status_code)
            HTTP_REQUESTS_TOTAL.inc(
                labels={"method": request.method, "endpoint": endpoint_label, "status_code": status_str}
            )
            HTTP_REQUEST_DURATION_SECONDS.observe(
                duration_s, labels={"method": request.method, "endpoint": endpoint_label}
            )
        except Exception:
            pass

        return response
