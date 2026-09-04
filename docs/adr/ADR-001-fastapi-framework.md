# ADR-001: Selection of FastAPI for Core API & Inference Gateway

## Status
Accepted

## Context
AIForge requires a high-performance, asynchronous web framework capable of handling concurrent ML/CV inference requests, streaming LLM responses, running background tasks, and enforcing strict schema validation with minimal serialization overhead.

Alternative frameworks considered:
1. **Flask**: Synchronous by default, requires manual OpenAPI generation, lack of native async I/O.
2. **Django / Django REST Framework**: Heavyweight ORM and monolithic overhead; unnecessarily rigid for dynamic model metadata and vector workflows.
3. **Tornado / Sanic**: Good async performance, but lacks the rich ecosystem, native Pydantic v2 validation, and automatic interactive documentation of FastAPI.

## Decision
We chose **FastAPI** as the core API framework for the AIForge intelligence platform.

### Key Architectural Rationale:
1. **High Concurrency & Async I/O**: Native `async`/`await` support on top of Starlette and AnyIO ensures the server does not block the event loop during vector database queries, database transactions, or external model API calls.
2. **Pydantic v2 Data Validation**: Eliminates repetitive payload validation and provides compiled C-speed parsing for large feature tensors and metadata payloads.
3. **Automated OpenAPI & JSON Schema**: Generates standard interactive Swagger (`/docs`) and ReDoc documentation out of the box, speeding up frontend and client integration.
4. **Dependency Injection System**: Allows modular injection of database sessions, authentication guards, and ML engine singletons across endpoints.

## Consequences
- Requires developers to understand async programming and avoid blocking CPU-bound calls in the main event loop (offloading heavy model training to background workers/executors).
