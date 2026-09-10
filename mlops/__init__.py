"""OmniForge MLOps & CI/CD Pipelines Package.

Provides DVC data versioning, MLflow experiment tracking & model registry,
automated evaluation gates, and CI/CD automation.
"""

from mlops.base import (
    EvalGateResult,
    ExperimentRun,
    MetricComparison,
    ModelStage,
    ModelVersion,
    PipelineRunResult,
    PipelineStatus,
    RegisteredModel,
    StageDefinition,
)
from mlops.dvc_pipeline import (
    DVCPipelineManager,
    compute_data_fingerprint,
    compute_file_hash,
    dvc_pipeline,
)
from mlops.eval_gate import ModelEvaluationGate, eval_gate
from mlops.mlflow_registry import MLflowRegistryManager, mlflow_registry

__all__ = [
    "ModelStage",
    "PipelineStatus",
    "ExperimentRun",
    "ModelVersion",
    "RegisteredModel",
    "MetricComparison",
    "EvalGateResult",
    "StageDefinition",
    "PipelineRunResult",
    "DVCPipelineManager",
    "compute_file_hash",
    "compute_data_fingerprint",
    "dvc_pipeline",
    "MLflowRegistryManager",
    "mlflow_registry",
    "ModelEvaluationGate",
    "eval_gate",
]
