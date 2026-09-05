# ADR-005: Unified Classical ML Pipeline & Estimator Interface

## Status
Accepted

## Context
OmniForge supports multiple machine learning paradigms (supervised classification, regression, unsupervised anomaly detection, and time-series forecasting). Without a unified contract, code becomes fragmented, data preprocessing differs between training and serving (causing training-serving skew), and inference APIs require custom routing logic for each model type.

## Decision
We establish an abstract `BaseMLEstimator` interface and a unified `AutoColumnTransformer` preprocessor in `ml/base.py` and `ml/preprocessing/pipeline.py`.

### Architectural Guarantees:
1. **Standardized Lifecycle Methods**:
   - `fit(X, y)`: Trains the preprocessor and estimator pipeline.
   - `predict(X)`: Generates predictions with automatic preprocessing.
   - `predict_proba(X)`: Produces calibrated class probabilities (for classification/anomaly scoring).
   - `evaluate(X, y)`: Computes task-specific production metrics (F1, ROC-AUC, RMSE, MAE, R2, Contamination).
   - `save(path)` / `load(path)`: Serializes model weights, feature names, data types, and preprocessors together.
2. **Elimination of Training-Serving Skew**:
   - The preprocessor (imputation values, categorical encodings, scaling parameters) is fitted strictly on training data and bundled directly inside the serialized artifact.

## Consequences
- Every new estimator developed for OmniForge must implement the `BaseMLEstimator` protocol, ensuring plug-and-play compatibility with the model registry and inference API.
