# ADR-003: Stateless Authentication with JWT & Role-Based Access Control (RBAC)

## Status
Accepted

## Context
AIForge serves multiple personas (Data Scientists, ML Engineers, Admins, Read-Only API consumers). The platform requires secure, scalable authentication that can verify permissions at line-rate across distributed microservices without hitting a central database on every micro-call.

## Decision
We implemented **JSON Web Tokens (JWT)** utilizing **HMAC-SHA256 (HS256)** for token signatures, coupled with **bcrypt** for robust password hashing, and granular **Role-Based Access Control (RBAC)** guards on API routers.

### Key Roles:
1. **Admin (`ADMIN`)**: Full platform control, user management, audit review, project/dataset deletion.
2. **ML Engineer (`ML_ENGINEER`)**: Create/edit projects, register datasets, run training pipelines, deploy models.
3. **Data Scientist (`DATA_SCIENTIST`)**: Read datasets, execute experiments, log metrics, run inference.
4. **Viewer (`VIEWER`)**: Read-only access to dashboards, experiment metrics, and model evaluation reports.

## Consequences
- Tokens are stateless; revocation lists for compromised tokens require short expiration windows (e.g. 60 minutes) backed by Redis token blocklisting when required.
