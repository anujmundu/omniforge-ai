# ADR-022: Distributed Task Mesh & Asynchronous Worker Pool Architecture

## Status
Accepted

## Context
High-compute tasks such as classical ML model fitting, cross-encoder document reranking, high-resolution video object tracking, and automated red-team security audits cannot execute synchronously within the HTTP request/response lifecycle without causing client timeouts, thread starvation, and head-of-line blocking. OmniForge requires a distributed, resilient, and prioritized asynchronous task execution mesh.

## Decision
1. **Priority Task Queueing (`DistributedTaskQueue`)**:
   - Multi-priority scheduling (`CRITICAL`, `HIGH`, `DEFAULT`, `LOW`, `BATCH`).
   - Exponential backoff retry policies with configurable maximum retry attempts.
   - Dedicated Dead-Letter Queue (DLQ) for unrecoverable task failures with diagnostic tracing.
2. **Worker Pool Execution (`AsyncWorkerPool`)**:
   - Thread-safe, non-blocking asynchronous task consumption.
   - Dynamic worker concurrency with heartbeat telemetry, load balancing, and timeout protection.
   - Pluggable storage backend supporting Redis Pub/Sub in production and zero-dependency in-memory queues for local testing and CI/CD pipelines.
3. **Cluster Health & Telemetry (`ClusterScalingManager`)**:
   - Real-time aggregation of node states, queue depth, average latency, and resource saturation.
   - Automated autoscaling recommendation engine evaluating HPA scaling triggers.

## Consequences
### Positive
- Prevents HTTP gateway timeout bottlenecks for long-running AI/ML operations.
- Full observability into queued, running, completed, failed, and dead-lettered jobs.
- Clean fallback between Redis and in-memory backends.

### Negative
- Asynchronous task completion requires polling or WebSocket notifications from frontend clients.
