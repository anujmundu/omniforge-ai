# ADR-016: MLflow Registry, Experiment Tracking, and Automated Promotion Gates

## Status
Accepted

## Context
Deploying machine learning models to production without continuous evaluation and strict governance risks silent model degradation, performance regressions, increased inference latency, and breaking schema changes.

We need a centralized model management architecture that provides:
1. **Experiment Tracking**: Fine-grained logging of hyperparameters, performance metrics, system tags, and model artifacts per training run.
2. **Model Registry & Semantic Versioning**: Tracking model versions (`v1`, `v2`, ...) with explicit operational stages (`None` -> `Staging` -> `Production` -> `Archived`).
3. **Automated Evaluation Gates**: Objective benchmark criteria (e.g., F1-score delta >= +0.01, accuracy delta >= 0.0, latency p95 delta <= +5%) comparing candidate models against active champion models before promotion.
4. **Auditability and Rollback Safety**: Automated archival of superseded champion models with instant one-click rollback capabilities.

## Decision
We implement a unified MLflow registry and evaluation gate engine (`mlops.mlflow_registry.MLflowRegistryManager` and `mlops.eval_gate.ModelEvaluationGate`):
1. **Stage State Machine**:
   - `NONE`: Freshly logged model candidate.
   - `STAGING`: Model passed initial sanity and integration testing, undergoing shadow/canary evaluation.
   - `PRODUCTION`: Active champion model serving live traffic.
   - `ARCHIVED`: Superseded model preserved for rollback and audit compliance.
2. **Automated Promotion Gating**:
   - Before a model transitions from `STAGING` to `PRODUCTION`, the `ModelEvaluationGate` evaluates the candidate against the current production baseline on a golden validation benchmark dataset.
   - If performance improves or meets parity thresholds without violating latency SLAs, promotion is automatically approved; otherwise, promotion is rejected with a detailed diagnostic report.
3. **Audit Trail**:
   - Every transition records the timestamp, transitioning user/system, reason, and evaluation gate report ID.

## Consequences
### Positive
- **Zero-Downtime Governance**: Enforces that only rigorously verified models reach production.
- **Auditable Lifecycle**: Full visibility into why, when, and by whom any model was promoted or demoted.
- **Automated Regression Prevention**: Eliminates human error during model releases.

### Negative
- Requires maintaining representative golden benchmark datasets for each registered model family.
