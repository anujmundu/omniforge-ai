# ADR-021: Kubernetes & Helm 3 Cloud-Native Deployment Architecture

## Status
Accepted

## Context
As OmniForge reaches enterprise readiness across Multimodal ML, Vision, NLP, RAG, MLOps, Observability, and Adversarial Security, single-server or single-node Docker Compose deployments become insufficient for high-availability production workloads. Production workloads demand automated self-healing, zero-downtime rolling updates, independent horizontal autoscaling (HPA) for compute vs. worker pods, ingress routing with TLS termination, and distributed persistent storage for model checkpoints.

## Decision
1. **Cloud-Native Topology**: Package the complete OmniForge infrastructure into a standardized **Helm 3.x chart** (`deploy/helm/omniforge/`) with configurable values:
   - **API Gateway Service**: Stateless FastAPI pods scaled dynamically (min 2, max 10) behind a Kubernetes `ClusterIP` service and NGINX Ingress controller.
   - **Worker Pod Mesh**: Dedicated compute pods running Celery/asynchronous worker pools listening to Redis queues for long-running jobs (model training, video inference, vector indexing).
   - **Horizontal Pod Autoscaling (HPA v2)**: Autoscaling triggered at 75% CPU and 80% Memory utilization thresholds.
   - **Persistent Storage (PVC)**: Standardized `ReadWriteMany` / `ReadWriteOnce` persistent volume claims for shared model registry artifacts (`storage/artifacts`) and dataset storage (`storage/datasets`).
2. **Probes & Lifecycle Management**:
   - `livenessProbe` checking HTTP `/api/v1/health` with 10s initial delay.
   - `readinessProbe` verifying database and redis connectivity before admitting traffic.
   - `preStop` hook enabling 30s graceful connection draining.

## Consequences
### Positive
- Fully reproducible deployments across AWS EKS, GCP GKE, Azure AKS, and local Minikube/k3s.
- Clear separation of concerns between interactive API requests and compute-heavy background tasks.
- Zero-downtime rolling rollouts with configurable `maxUnavailable` and `maxSurge`.

### Negative
- Requires Kubernetes orchestration tooling (kubectl, Helm 3) and cloud infrastructure management.
