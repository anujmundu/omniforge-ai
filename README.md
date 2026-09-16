# OmniForge — Production-Grade Multimodal AI/ML Intelligence Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://anujmundu-omniforge-ai.streamlit.app)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg?logo=docker)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-Live%20API-46E3B7.svg?logo=render)](https://omniforge-ai.onrender.com/docs)
[![API Docs](https://img.shields.io/badge/Swagger%20UI-Live%20Interactive-success.svg?logo=swagger)](https://omniforge-ai.onrender.com/docs)
[![Health Check](https://img.shields.io/badge/API%20Health-Passing%20200%20OK-brightgreen.svg)](https://omniforge-ai.onrender.com/api/v1/health)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **OmniForge** is an enterprise-grade multimodal AI/ML intelligence platform engineered from first principles. It unifies **Classical Machine Learning**, **Deep Learning / Computer Vision**, **Natural Language Processing**, **Enterprise Retrieval-Augmented Generation (RAG)**, and **Multi-Agent Orchestration** behind a resilient, observable asynchronous API layer.

---

### 🚀 Live Production Deployment

OmniForge is deployed in production and actively serving live inferences:

| Service | Live URL | Description |
| :--- | :--- | :--- |
| **Interactive UI Control Center** | [https://anujmundu-omniforge-ai.streamlit.app](https://anujmundu-omniforge-ai.streamlit.app) | Streamlit multi-agent playground & visual AI demos |
| **Interactive Swagger Docs** | [https://omniforge-ai.onrender.com/docs](https://omniforge-ai.onrender.com/docs) | Test all NLP, Vision, ML, & Agent endpoints live |
| **ReDoc API Contract** | [https://omniforge-ai.onrender.com/redoc](https://omniforge-ai.onrender.com/redoc) | Formal OpenAPI 3.1 specification & schema viewer |
| **Deep Health & Telemetry** | [https://omniforge-ai.onrender.com/api/v1/health](https://omniforge-ai.onrender.com/api/v1/health) | Real-time database latency & subsystem telemetry |
| **Production API Gateway** | [https://omniforge-ai.onrender.com](https://omniforge-ai.onrender.com) | Core asynchronous FastAPI gateway |

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
- [x] **Phase 6: Multi-Agent Orchestration** — ReAct reasoning loops, dynamic tool calling, multi-domain autonomous solvers, supervisor intent routing, and AST-secured code execution.
- [x] **Phase 7: MLOps & CI/CD Pipelines** — DVC Data Versioning & Pipeline DAGs, MLflow Central Registry & Experiment Tracking, Automated Candidate vs. Champion Regression Evaluation Gates, Zero-Downtime Rollback Safety, and GitHub Actions CI/CD Workflows.
- [x] **Phase 8: Production Observability** — Prometheus metrics, Evidently data drift monitoring, Grafana dashboards.
- [x] **Phase 9: Adversarial Security & Red-Teaming** — Multi-layer prompt injection defense, PII/secrets redaction, token-bucket rate limiting, and automated 32-vector red-team audit battery.
- [x] **Phase 10: Cloud Deployment & Scaling** — Cloud-native Kubernetes Helm 3.x charts, distributed Celery/Redis priority task mesh, worker pool autoscaling, and real-time HPA telemetry.

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

5. **Run the Live Demonstrations (All 10 Phases):**
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

   # Phase 7: MLOps & CI/CD Pipelines (DVC DAGs, MLflow Registry, Promotion Gates)
   python scripts/demo_phase7_mlops.py

   # Phase 8: Production Observability (Prometheus Metrics, Evidently Data Drift, Grafana)
   python scripts/demo_phase8_observability.py

   # Phase 9: Adversarial Security & Red-Teaming (Prompt Injection Defenses, PII Redaction, Rate Limiting)
   python scripts/demo_phase9_security.py

   # Phase 10: Cloud Deployment & Scaling (Celery/Redis Task Mesh, Worker Pool Autoscaling, Helm)
   python scripts/demo_phase10_scaling.py
   ```

6. **Start the API Server and Streamlit Dashboard:**
   ```bash
   # Terminal 1: Launch FastAPI Backend Gateway
   uvicorn apps.api.main:app --reload --port 8000

   # Terminal 2: Launch Interactive Streamlit Control Center
   streamlit run apps/dashboard/app.py
   ```

7. **Explore interactive documentation & live UI:**
   - **Live Streamlit Dashboard**: [https://anujmundu-omniforge-ai.streamlit.app](https://anujmundu-omniforge-ai.streamlit.app)
   - **Production Swagger UI**: [https://omniforge-ai.onrender.com/docs](https://omniforge-ai.onrender.com/docs)
   - **Production ReDoc**: [https://omniforge-ai.onrender.com/redoc](https://omniforge-ai.onrender.com/redoc)
   - **Production Health Check**: [https://omniforge-ai.onrender.com/api/v1/health](https://omniforge-ai.onrender.com/api/v1/health)
   - Local Streamlit Dashboard: [http://localhost:8501](http://localhost:8501)
   - Local Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

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
- [ADR-007: Deep Learning Object Detection & Spatial Inference Architecture](docs/adr/ADR-007-vision-detection-and-tracking-architecture.md)
- [ADR-008: Real-Time Multi-Object Tracking & Video Frame Ingestion Architecture](docs/adr/ADR-008-video-stream-and-spatial-ocr-pipeline.md)
- [ADR-009: Dense Text Embeddings & Dimensionality Reduction Architecture](docs/adr/ADR-009-transformer-embedding-and-tokenization-architecture.md)
- [ADR-010: Named Entity Recognition & Contextual Sequence Classification](docs/adr/ADR-010-named-entity-recognition-and-semantic-similarity-pipeline.md)
- [ADR-011: Enterprise RAG Ingestion, Chunking & Parsing Architecture](docs/adr/ADR-011-enterprise-rag-ingestion-and-chunking-architecture.md)
- [ADR-012: Hybrid Vector Retrieval & Cross-Encoder Reranking Architecture](docs/adr/ADR-012-hybrid-vector-retrieval-and-cross-encoder-reranking.md)
- [ADR-013: Multi-Agent ReAct Orchestration & Intent Routing Architecture](docs/adr/ADR-013-multi-agent-react-orchestration-and-intent-routing.md)
- [ADR-014: Declarative Tool Calling & Agent Execution Mesh](docs/adr/ADR-014-declarative-tool-calling-and-agent-execution-mesh.md)
- [ADR-015: DVC Data & ML Pipeline Versioning Architecture](docs/adr/ADR-015-dvc-data-and-pipeline-versioning.md)
- [ADR-016: MLflow Model Registry & Automated Promotion Gates](docs/adr/ADR-016-mlflow-registry-and-automated-promotion-gates.md)
- [ADR-017: Prometheus & Grafana Production Observability Stack](docs/adr/ADR-017-prometheus-and-grafana-observability-stack.md)
- [ADR-018: Data Drift & Statistical Monitoring with Evidently AI](docs/adr/ADR-018-data-drift-and-evidently-monitoring-architecture.md)
- [ADR-019: Adversarial Prompt Injection & LLM Guardrails Architecture](docs/adr/ADR-019-adversarial-prompt-injection-and-llm-security-guardrails.md)
- [ADR-020: Token-Bucket Rate Limiting & DDoS Defense Architecture](docs/adr/ADR-020-token-bucket-rate-limiting-and-ddos-defense.md)
- [ADR-021: Kubernetes & Helm 3 Cloud-Native Deployment Architecture](docs/adr/ADR-021-kubernetes-and-helm-deployment-architecture.md)
- [ADR-022: Distributed Task Mesh & Asynchronous Worker Pool Architecture](docs/adr/ADR-022-celery-redis-distributed-task-mesh-and-async-scaling.md)

---

## 👨‍💻 Author

**Anuj Mundu**  
Master of Computer Applications (MCA)  
Maulana Azad National Institute of Technology (MANIT), Bhopal  

### Areas of Interest
- Artificial Intelligence
- Agentic AI
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Machine Learning & Deep Learning
- Full-Stack AI Engineering
- AI Systems Design & MLOps

**Connect & Follow:**
- **GitHub**: [https://github.com/anujmundu](https://github.com/anujmundu)
- **LinkedIn**: [Anuj Mundu | LinkedIn](https://www.linkedin.com/in/anujmundu/)

---

## ⭐ Support the Project

If you found this project useful:
- ⭐ **Star** the repository
- 🍴 **Fork** the project
- 🛠️ **Share** suggestions and improvements
- 💬 **Open issues** for bugs or feature requests

*Every contribution helps improve the project.*

---

## 📄 License

MIT License. Designed and engineered by **Anuj Mundu**.

