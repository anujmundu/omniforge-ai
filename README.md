# OmniForge — Production-Grade Multimodal AI/ML Intelligence Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791.svg?logo=postgresql)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **OmniForge** is an enterprise-grade multimodal AI/ML intelligence platform engineered from first principles. It unifies **Classical Machine Learning**, **Deep Learning / Computer Vision**, **Natural Language Processing**, **Enterprise Retrieval-Augmented Generation (RAG)**, and **Multi-Agent Orchestration** behind a resilient, observable asynchronous API layer.

---

## Architecture Overview

```
                         ┌──────────────────────────┐
                         │       Web / API Client   │
                         └────────────┬─────────────┘
                                      │
                              API Gateway / Auth
                                      │
                         ┌────────────▼─────────────┐
                         │        FastAPI            │
                         │     AI/ML API Layer       │
                         └────────────┬─────────────┘
                                      │
             ┌────────────────────────┼────────────────────────┐
             │                        │                        │
             ▼                        ▼                        ▼
      Classical ML             Computer Vision            GenAI / RAG
      ─────────────             ──────────────            ──────────
      Classification            YOLO                    Documents
      Regression                OpenCV                  Chunking
      Anomaly Detection         Tracking                Embeddings
      Forecasting               OCR                     Retrieval
                                                        Reranking
                                                        LLM
             │                        │                        │
             └────────────────────────┼────────────────────────┘
                                      │
                              AI Orchestration
                                      │
                         ┌────────────▼────────────┐
                         │   Model / AI Router     │
                         └────────────┬────────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
                 ▼                    ▼                    ▼
              MLflow              Vector DB             LLMs
              Registry            PostgreSQL             Local/API
                 │                    │                    │
                 └────────────────────┼────────────────────┘
                                      │
                                Data Layer
                                      │
                    ┌─────────────────┼────────────────┐
                    ▼                 ▼                ▼
                  SQL              Object Store      Redis
                    │                 │
                    └────────┬────────┘
                             │
                        ML Pipelines
                             │
                  ┌──────────▼──────────┐
                  │ Training / Evaluation│
                  └──────────┬──────────┘
                             │
                       MLOps / CI/CD
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
          Docker         GitHub Actions      Cloud
                                                │
                                           Monitoring
```

---

## The 12 Production Quality Gates

Every module in AIForge passes through 12 engineering quality gates:

| # | Quality Gate | AIForge Implementation |
|---|---|---|
| **1** | **Real Problem Definition** | Solves high-impact industrial problems (churn, fraud, visual tracking, doc QA). |
| **2** | **Technical Depth** | Modular abstractions (`BaseEstimator`, `BaseVisionPipeline`, `BaseRAG`). |
| **3** | **Defensible Architecture** | Documented [Architectural Decision Records (ADRs)](docs/adr/). |
| **4** | **Measurable Evaluation** | Quantitative benchmarks (F1, ROC-AUC, mAP, Recall@K, Groundedness, Latency). |
| **5** | **Failure & Boundary Analysis** | Intentional error handling, data drift detection, prompt injection defenses. |
| **6** | **Zero-Friction Reproducibility** | Deterministic single-command startup with `docker compose up --build`. |
| **7** | **RESTful API Contracts** | Strict Pydantic v2 schemas, automated OpenAPI documentation (`/docs`). |
| **8** | **Containerization** | Multi-stage Docker builds, isolated network bridges, healthchecks. |
| **9** | **Automated Test Suite** | Unit, integration, schema validation, and end-to-end API tests with `pytest`. |
| **10** | **Living Documentation** | End-to-end system design diagrams, ER diagrams, and API guides. |
| **11** | **Observability** | Structured JSON logging, correlation IDs (`X-Request-ID`), latency tracking. |
| **12** | **Interview Defensibility** | Documented trade-offs, scalability bottlenecks, and engineering decisions. |

---

## 10-Phase Roadmap

- [x] **Phase 1: Foundation** — Core FastAPI framework, Async PostgreSQL/SQLAlchemy 2.0, JWT + RBAC auth, Project/Dataset/Experiment tracking, Docker environment, automated tests.
- [x] **Phase 2: Classical ML Engine** — Classification, Regression, Anomaly Detection, Forecasting, Automated Preprocessing, Model Registry, Real-time REST Inference Serving.
- [ ] **Phase 3: Computer Vision Engine** — YOLO detection, ByteTrack tracking, OCR, frame extraction pipelines.
- [ ] **Phase 4: NLP Pipeline** — Transformer embeddings, NER, text classification, semantic similarity.
- [ ] **Phase 5: Enterprise RAG** — Document ingestion, semantic chunking, vector indexing, reranking, Ragas evaluation.
- [ ] **Phase 6: Multi-Agent Orchestrator** — Intent routing, tool calling (SQL, ML, Vision, RAG agents).
- [ ] **Phase 7: MLOps & CI/CD** — DVC dataset versioning, MLflow stage promotion, automated release pipelines.
- [ ] **Phase 8: Production Observability** — Prometheus metrics, Evidently data drift monitoring, Grafana dashboards.
- [ ] **Phase 9: Adversarial Security & Red-Teaming** — Prompt injection testing, rate-limiting, RBAC penetration auditing.
- [ ] **Phase 10: Cloud Deployment & Scaling** — Kubernetes/Helm charts, Celery/Redis asynchronous worker mesh.

---

## Quickstart

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (Optional for containerized run)

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anujmundu/omniforge-ai.git
   cd omniforge-ai
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the test suite:**
   ```bash
   pytest
   ```

5. **Run the Phase 2 Classical ML Demonstration:**
   ```bash
   python scripts/demo_phase2_ml.py
   ```

6. **Start the API server:**
   ```bash
   uvicorn apps.api.main:app --reload --port 8000
   ```

7. **Explore interactive documentation:**
   - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
   - Health Check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

### Running with Docker Compose

```bash
docker compose up --build -d
```
This boots up:
- **FastAPI Core Gateway**: [http://localhost:8000](http://localhost:8000)
- **PostgreSQL Database**: `localhost:5432`
- **Redis Cache & Broker**: `localhost:6379`
- **MLflow Tracking Server**: [http://localhost:5000](http://localhost:5000)

---

## Architectural Decision Records (ADRs)
- [ADR-001: Selection of FastAPI for Core API & Inference Gateway](docs/adr/ADR-001-fastapi-framework.md)
- [ADR-002: Relational Metadata Storage with PostgreSQL, AsyncPG & SQLAlchemy 2.0](docs/adr/ADR-002-postgresql-sqlalchemy2-asyncpg.md)
- [ADR-003: Stateless Authentication with JWT & Role-Based Access Control (RBAC)](docs/adr/ADR-003-jwt-rbac-security.md)
- [ADR-004: Configuration & Secrets Management with Pydantic Settings](docs/adr/ADR-004-configuration-secrets-management.md)
- [ADR-005: Unified Classical ML Pipeline & Estimator Interface](docs/adr/ADR-005-classical-ml-pipeline-architecture.md)
- [ADR-006: Feature Store & Pipeline Serialization Strategy](docs/adr/ADR-006-feature-store-and-model-registry.md)

---

## License
MIT License. Created with enterprise engineering standards by the OmniForge Team.
