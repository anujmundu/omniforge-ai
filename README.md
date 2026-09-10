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
- [x] **Phase 3: Computer Vision Engine** — Object Detection (YOLO), Multi-Object Tracking (ByteTrack), Spatial OCR, Async Video Frame Stream Ingestion, and REST APIs.
- [x] **Phase 4: NLP Pipeline** — Dense Transformer Embeddings, Span-Level NER, Text Classification, Cross-Document Semantic Similarity, and REST APIs.
- [x] **Phase 5: Enterprise RAG Engine** — Document Ingestion (Markdown/JSON/HTML), Recursive Semantic Chunking, Dense Vector Store Collections, Cross-Encoder Reranking, Citation-Backed Q&A Generation, and Automated Evaluation.
- [x] **Phase 7: MLOps & CI/CD Pipelines** — DVC Data Versioning & Pipeline DAGs, MLflow Central Registry & Experiment Tracking, Automated Candidate vs. Champion Regression Evaluation Gates, Zero-Downtime Rollback Safety, and GitHub Actions CI/CD Workflows.
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

4. **Run the automated test suite (93 tests):**
   ```bash
   pytest
   ```

5. **Run the Live Demonstrations:**
   ```bash
   # Phase 1: End-to-End Foundation & RBAC Lifecycle
   python scripts/demo_e2e_flow.py

   # Phase 2: Classical ML Engine (4 Paradigms + Inference Serving)
   python scripts/demo_phase2_ml.py

   # Phase 3: Computer Vision (Detection, Video Tracking, Spatial OCR, Streaming)
   python scripts/demo_phase3_vision.py

   # Phase 4: NLP Pipeline (Embeddings, Span NER, Classification, Semantic Search)
   python scripts/demo_phase4_nlp.py

   # Phase 5: Enterprise RAG Engine (Ingestion, Chunking, Vector Store, Reranker, Grounded Q&A)
   python scripts/demo_phase5_rag.py

   # Phase 6: Multi-Agent Orchestrator (Intent Routing, Tool Introspection, ReAct Loops)
   python scripts/demo_phase6_agents.py
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
- [ADR-007: Deep Learning Object Detection & Spatial Inference Architecture](docs/adr/ADR-007-deep-learning-object-detection-architecture.md)
- [ADR-008: Real-Time Multi-Object Tracking & Video Frame Ingestion Architecture](docs/adr/ADR-008-real-time-multi-object-tracking-and-video-stream-architecture.md)
- [ADR-009: Dense Text Embeddings & Dimensionality Reduction Architecture](docs/adr/ADR-009-dense-text-embeddings-and-dimensionality-architecture.md)
- [ADR-010: Named Entity Recognition & Contextual Sequence Classification](docs/adr/ADR-010-named-entity-recognition-and-sequence-classification.md)
- [ADR-011: Enterprise RAG Ingestion, Chunking & Parsing Architecture](docs/adr/ADR-011-enterprise-rag-ingestion-and-chunking-architecture.md)
- [ADR-012: Hybrid Vector Retrieval & Cross-Encoder Reranking Architecture](docs/adr/ADR-012-hybrid-vector-retrieval-and-cross-encoder-reranking.md)
- [ADR-013: Multi-Agent ReAct Orchestration & Intent Routing Architecture](docs/adr/ADR-013-multi-agent-react-orchestration-and-intent-routing.md)
- [ADR-014: Declarative Tool Calling & Agent Execution Mesh](docs/adr/ADR-014-declarative-tool-calling-and-agent-execution-mesh.md)

---

## License
MIT License. Created with enterprise engineering standards by the OmniForge Team.
