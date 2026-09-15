# OmniForge Production Deployment & Hardening Guide

This document specifies the operational runbook and architecture requirements for deploying the OmniForge Multimodal AI/ML Intelligence Platform in production environments.

---

## 1. Secrets Management & Cryptographic Security
- **JWT Signing Key (`SECRET_KEY`)**: Must be a high-entropy 256-bit (32-byte) hex string generated via a cryptographically secure random number generator:
  ```bash
  python -c "import secrets; print(secrets.token_hex(32))"
  ```
- **Environment Isolation**: Set `ENVIRONMENT=production` and ensure `DEBUG=false` to prevent stack trace disclosures.
- **CORS Protection**: Explicitly restrict `CORS_ORIGINS` to trusted frontend domains and administrative dashboards.

---

## 2. Infrastructure & Data Layer Architecture
- **Relational Storage**: Deploy PostgreSQL 16+ using pooled connection handling via SQLAlchemy 2.0 and `asyncpg`.
- **In-Memory Cache & Message Broker**: Deploy Redis 7+ for Celery distributed task execution, token-bucket rate limiting, and embedding cache.
- **Artifact & Dataset Volumes**: Persist model weights, vector embeddings, and datasets onto persistent volume claims (`/app/storage/artifacts`, `/app/storage/datasets`).

---

## 3. Distributed Telemetry & Observability
- **OpenTelemetry & OTLP**: Ingest distributed traces via the standard OpenTelemetry collector at `http://jaeger-collector:4317` (gRPC) or `http://jaeger-collector:4318` (HTTP).
- **Service Name**: Configured through `OTEL_SERVICE_NAME=omniforge-api`.
- **Metrics**: Expose Prometheus metrics endpoint at `/metrics` for scrape targets.

---

## 4. Container & Kubernetes Orchestration
- **Docker Compose**: Standard multi-container development and single-node production runtime utilizing health checks (`postgres`, `redis`, `api`, `mlflow`).
- **Helm 3 Charts**: Cloud-native deployment into Kubernetes clusters with automated Horizontal Pod Autoscaling (HPA) and ingress routing.
