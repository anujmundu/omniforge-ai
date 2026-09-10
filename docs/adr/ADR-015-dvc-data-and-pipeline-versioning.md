# ADR-015: DVC Data and Pipeline Versioning Architecture

## Status
Accepted

## Context
OmniForge unifies classical tabular ML, computer vision models, NLP pipelines, RAG retrieval engines, and multi-agent reasoning systems. In enterprise machine learning, training datasets, raw inputs, preprocessed features, and model weight artifacts are too large to track via Git, leading to reproducibility drift, untracked dataset modifications, and opaque pipelines.

We need a standardized data versioning and reproducible pipeline execution framework that:
1. Computes deterministic cryptographic hashes (SHA-256 / MD5) for raw and preprocessed data splits.
2. Defines declarative, deterministic pipeline execution stages (`load_data` -> `preprocess` -> `train` -> `evaluate`) with explicit dependencies (`deps`) and outputs (`outs`).
3. Supports pipeline step caching so un-modified upstream stages are never re-computed redundantly.
4. Integrates seamlessly with local development environments, CI/CD runners, and remote storage backends (S3, MinIO, Azure Blob, Google Cloud Storage).

## Decision
We adopt **Data Version Control (DVC)** with programmatic Python abstractions (`mlops.dvc_pipeline.DVCPipelineManager`):
1. **Declarative Stage Contracts**:
   - Every pipeline stage explicitly declares its inputs (`deps`), outputs (`outs`), parameters (`params`), and execution metrics (`metrics`).
2. **Deterministic Hash Registry**:
   - File and directory states are indexed by content hashes. If input hashes are unchanged, the pipeline reuses cached stage outputs.
3. **Storage-Agnostic Remote Support**:
   - DVC manages pointers (`.dvc` files and `.dvcignore`) while actual large blobs reside in remote storage.
4. **Programmatic Python API**:
   - In addition to CLI tooling (`dvc repro`, `dvc push`), the platform provides a native Python interface for automated trigger execution, lineage querying, and DAG validation.

## Consequences
### Positive
- **Guaranteed Reproducibility**: Exact data states and pipeline executions can be recreated across any environment or historical timestamp.
- **Compute Optimization**: Intelligent stage caching prevents redundant retraining on static datasets.
- **Data Lineage**: Unbroken audit trail linking model artifacts back to the exact training dataset version.

### Negative
- Developers must maintain `dvc.yaml` and `params.yaml` contracts when altering data ingestion schemas.
