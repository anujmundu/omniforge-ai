# ADR-020: Token-Bucket Rate Limiting & DDoS Defense Architecture

## Status
Accepted

## Context
Production LLM endpoints and multi-modal pipeline services incur significant compute, GPU, and API token costs. To prevent denial-of-wallet attacks, algorithmic complexity exhaustion, and abusive scraping, OmniForge requires deterministic, tenant-aware, and tier-specific rate limiting and automated red-teaming harnesses.

## Decision
1. **Thread-Safe In-Memory Token Bucket**: Implement `TokenBucketRateLimiter` utilizing monotonic time clocks and sliding token refills per client identity (API key or IP).
2. **Subscription Tier Quotas**: Support tiered rate capacities:
   - `free`: 60 requests/min, burst capacity 10.
   - `pro`: 600 requests/min, burst capacity 50.
   - `enterprise`: 3,000 requests/min, burst capacity 200.
3. **Automated Red-Teaming Harness**: Implement `RedTeamEngine` equipped with 32 curated adversarial attack vectors mapped directly to the OWASP Top 10 for LLM Applications (LLM01 Prompt Injection, LLM02 Sensitive Information Disclosure, LLM06 Excessive Agency, etc.).

## Consequences
### Positive
- Strict isolation of misbehaving clients without noisy neighbor impact.
- Standardized HTTP 429 response headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `Retry-After`).
- Continuous automated compliance auditing against OWASP LLM Top 10 vulnerabilities.

### Negative
- Local in-memory store is single-process; multi-node clusters will require distributed Redis backend in Phase 10.
