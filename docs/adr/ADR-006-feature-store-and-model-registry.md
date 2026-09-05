# ADR-006: Feature Store & Pipeline Serialization Strategy

## Status
Accepted

## Context
Deploying trained models to production often suffers from serialization mismatches, missing feature headers, or deserialization vulnerabilities. Furthermore, high-throughput REST APIs require sub-10ms inference latencies, which cannot afford re-parsing raw pipeline configs on every request.

## Decision
We implement a unified **`ModelRegistry`** and **Artifact Packaging** standard:
1. **Atomic Artifact Bundling**: Models are saved as composite packages containing:
   - Estimator weights (Joblib / Scikit-learn serialized estimators)
   - Fitted `AutoColumnTransformer` preprocessors
   - Feature schema signatures (names, expected types, default fallbacks)
   - Evaluation metrics & parameter manifests (JSON)
2. **In-Memory Model Caching**: The API layer loads and caches models into memory on first request or application boot, reducing per-request prediction latency to <5ms.
3. **Strict Input Coercion**: Raw incoming JSON payloads are automatically converted to typed DataFrames and validated against the model's feature schema before preprocessing.

## Consequences
- Requires storage space for artifact archives, but ensures 100% deterministic reproducibility across training, validation, and inference environments.
