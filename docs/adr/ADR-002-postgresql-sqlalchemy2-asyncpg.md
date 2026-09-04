# ADR-002: Relational Metadata Storage with PostgreSQL, AsyncPG & SQLAlchemy 2.0

## Status
Accepted

## Context
AIForge needs a rock-solid metadata, relational tracking, and persistence layer for:
- User identities, roles, and tenant isolation
- Project hierarchies and dataset version metadata
- Model experiment parameters, run metrics, and artifact references
- Future vector embedding search via `pgvector`

Alternatives considered:
1. **MongoDB / NoSQL Document Stores**: Schemaless flexibility, but lacks strict relational foreign keys, transactional ACID guarantees for audit logs, and requires a separate vector DB.
2. **SQLite**: Excellent for local testing and self-contained pipelines, but lacks production concurrent write support and enterprise RBAC capabilities.

## Decision
We chose **PostgreSQL** paired with **SQLAlchemy 2.0** (Async API) and **asyncpg** driver for production, with transparent **aiosqlite** fallback for lightweight offline unit testing.

### Key Architectural Rationale:
1. **Strict Relational Integrity**: Ensures cascade deletes, foreign keys (e.g. `datasets` belongs to `projects`, `experiments` belongs to `projects`), and unique constraints.
2. **SQLAlchemy 2.0 Async**: Standardized modern type-annotated mapped columns (`Mapped[str]`, `mapped_column`), pure async queries (`select()`, `scalars()`), and clean transaction lifecycles.
3. **pgvector Extensibility**: PostgreSQL can be extended with pgvector in Phase 5 for unified relational + semantic hybrid search without introducing extra database infrastructure if desired.

## Consequences
- Database migrations must be tracked and applied via Alembic.
