# ADR-017: Prometheus & Grafana Observability Stack Architecture

## Status
Accepted

## Context
OmniForge operates high-throughput multimodal pipelines spanning Classical ML, Computer Vision (YOLO/DeepSORT), NLP (MiniLM/NER), Hybrid RAG (Dense + BM25 + Cross-Encoder), and Autonomous Multi-Agent Reasoning. To maintain 99.9% uptime and operational reliability in production environments, the platform requires unified, low-overhead, multi-dimensional time-series metrics collection, latency histograms, error rate tracking, and automated dashboard visualization.

## Decision
1. **Thread-Safe Telemetry Registry**: Implement an in-memory `PrometheusRegistry` supporting monotonically increasing `Counter`, instantaneous `Gauge`, and cumulative bucketed `Histogram` collectors.
2. **Prometheus Exposition Format**: Expose standard text-format metrics via `/metrics` (for Prometheus scrapers) and `/api/v1/observability/metrics` (for authenticated gateway consumers).
3. **Automated Request Instrumentation**: Integrate `RequestIDMiddleware` in FastAPI to automatically track inbound HTTP request throughput (`omniforge_http_requests_total`) and request duration distributions (`omniforge_http_request_duration_seconds`).
4. **Declarative Dashboard Provisioning**: Provide native Grafana datasource and dashboard provisioning (`deploy/grafana/`) with pre-configured visualization panels for all multimodal engines and system resource utilization.

## Consequences
### Positive
- Standardized Prometheus metrics format compatible with Kubernetes Prometheus Operator, AWS Managed Prometheus, and Datadog.
- Sub-millisecond in-memory metrics overhead with zero blocking on critical inference paths.
- Single-pane-of-glass observability into ML inference latencies, Vision streaming FPS, NLP token throughput, RAG search timings, and agent steps.

### Negative
- Requires storage configuration for Prometheus time-series retention in large-scale production deployments.
