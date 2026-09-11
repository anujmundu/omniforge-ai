# ADR-018: Data Drift & Statistical Monitoring Architecture

## Status
Accepted

## Context
In production machine learning systems, statistical properties of input data inevitably evolve over time (covariate shift, concept drift, upstream schema changes), degrading model inference performance silently without throwing explicit runtime exceptions. OmniForge requires automated statistical drift detection across both numerical and categorical feature distributions to proactively alert operators and trigger model evaluation/retraining workflows.

## Decision
1. **Statistical Drift Engine**: Implement a vectorized drift calculation engine supporting:
   - **Two-Sample Kolmogorov-Smirnov (KS) Test**: For continuous numerical features, evaluating empirical cumulative distribution function (eCDF) maximum discrepancy ($D$) with asymptotic p-value estimation ($\alpha = 0.05$).
   - **Population Stability Index (PSI)**: For categorical and binned numerical distributions ($PSI \ge 0.20$ denotes significant drift).
   - **Missing Value Tracking**: Continuous quantification of feature-level null rates between reference training baselines and live inference batches.
2. **Dataset-Level Aggregation**: Compute dataset drift status when the proportion of drifted features exceeds the configurable dataset threshold (default: 33%).
3. **Automated SLA Alerting**: Integrate drift metrics directly into `AlertManager` and expose `/api/v1/observability/drift/calculate` for scheduled batch evaluation and pipeline integration.

## Consequences
### Positive
- Early detection of model degradation before downstream business KPIs are impacted.
- High-performance vectorized computation in NumPy/Pandas without heavyweight external server dependencies.
- Direct lineage coupling with MLOps pipeline retraining gates.

### Negative
- Requires maintaining baseline reference distribution datasets in storage.
