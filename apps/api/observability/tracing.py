import os
from loguru import logger

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


def init_tracing():
    """Initialize OpenTelemetry tracing based on environment variables.

    Expected env vars:
        OTEL_EXPORTER_OTLP_ENDPOINT - OTLP collector endpoint (e.g., http://jaeger-collector:4317)
        OTEL_SERVICE_NAME - Logical name of the service (default: "omniforge")
    """
    if not _HAS_OTEL:
        logger.info("OpenTelemetry OTLP exporter packages not installed; skipping tracing initialization.")
        return None

    try:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        service_name = os.getenv("OTEL_SERVICE_NAME", "omniforge")

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        return trace.get_tracer(__name__)
    except Exception as exc:
        logger.warning(f"OpenTelemetry initialization encountered an issue: {exc}. Tracing disabled.")
        return None

