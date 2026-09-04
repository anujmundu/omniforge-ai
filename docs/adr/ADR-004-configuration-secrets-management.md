# ADR-004: Configuration & Secrets Management with Pydantic Settings

## Status
Accepted

## Context
Production ML systems require rigorous configuration management across multiple environments (Local development, CI automated testing, Staging, and Production) adhering to the 12-Factor App methodology.

## Decision
We utilize **Pydantic Settings (`BaseSettings`)** with strict type validation, default fallbacks, and multi-tier `.env` file resolution.

### Architectural Features:
1. **Strict Type Coercion**: Ensures port numbers are integers, secret keys meet minimum byte lengths, and database URLs conform to RFC URI standards.
2. **Deterministic Defaults**: Sensible local defaults for offline unit testing without requiring external cloud secrets.
3. **Immutability & Singleton Pattern**: Application settings are parsed once at startup and injected as a cached dependency.

## Consequences
- Missing required production environment variables immediately halts application boot with clear human-readable validation errors instead of runtime crashes.
