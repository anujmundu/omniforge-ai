"""OmniForge MLOps Pydantic v2 API Schemas.

Defines request and response schemas for runs, models, registry transitions,
evaluation gates, and pipeline execution triggers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StartRunRequest(BaseModel):
    """Request schema for starting an experiment run."""

    experiment_name: str = Field(default="default_experiment", description="Name of the experiment")
    tags: Dict[str, str] = Field(default_factory=dict, description="Run tags and metadata")


class LogMetricsRequest(BaseModel):
    """Request schema for logging metrics to an active or specified run."""

    run_id: str = Field(..., description="Target experiment run ID")
    metrics: Dict[str, float] = Field(..., description="Dictionary of metric name to float value")


class LogParamsRequest(BaseModel):
    """Request schema for logging parameters to an active or specified run."""

    run_id: str = Field(..., description="Target experiment run ID")
    parameters: Dict[str, Any] = Field(..., description="Dictionary of parameter name to value")


class EndRunRequest(BaseModel):
    """Request schema for ending an experiment run."""

    run_id: str = Field(..., description="Target experiment run ID")
    status: str = Field(default="SUCCESS", description="Final run status (SUCCESS or FAILED)")


class ExperimentRunResponse(BaseModel):
    """Response schema representing an experiment run."""

    run_id: str
    experiment_name: str
    status: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)
    artifact_uris: List[str] = Field(default_factory=list)
    start_time: datetime
    end_time: Optional[datetime] = None


class RegisterModelRequest(BaseModel):
    """Request schema for registering a model version from an experiment run."""

    name: str = Field(..., description="Registered model name")
    run_id: str = Field(..., description="Run ID containing model artifacts and metrics")
    description: str = Field(default="", description="Model version description")
    artifact_uri: str = Field(default="", description="URI or path to the saved model artifact")
    tags: Dict[str, str] = Field(default_factory=dict, description="Version tags")


class ModelVersionResponse(BaseModel):
    """Response schema representing a registered model version."""

    model_name: str
    version: int
    run_id: str
    stage: str
    description: str = ""
    metrics: Dict[str, float] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    artifact_uri: str = ""
    created_at: datetime
    updated_at: datetime
    tags: Dict[str, str] = Field(default_factory=dict)


class RegisteredModelResponse(BaseModel):
    """Response schema representing a named registered model and all its versions."""

    name: str
    description: str = ""
    latest_version: int = 0
    versions: List[ModelVersionResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    tags: Dict[str, str] = Field(default_factory=dict)


class TransitionStageRequest(BaseModel):
    """Request schema for transitioning a model version to a target lifecycle stage."""

    model_name: str = Field(..., description="Name of the registered model")
    version: int = Field(..., description="Version number to transition")
    target_stage: str = Field(..., description="Target stage: None, Staging, Production, Archived")
    archive_existing: bool = Field(
        default=True, description="Whether to automatically archive existing production version"
    )


class MetricComparisonSchema(BaseModel):
    """Schema representing comparison details for a single evaluation metric."""

    metric_name: str
    candidate_value: float
    champion_value: Optional[float] = None
    delta: Optional[float] = None
    threshold: float
    passed: bool
    description: str = ""


class EvalGateRequest(BaseModel):
    """Request schema for executing an automated model regression evaluation gate."""

    model_name: str = Field(..., description="Name of the registered model")
    candidate_version: int = Field(..., description="Candidate version number")
    golden_dataset_metrics: Optional[Dict[str, float]] = Field(
        default=None, description="Optional override metrics evaluated on golden dataset"
    )
    auto_promote: bool = Field(default=False, description="Automatically promote to Production if gate passes")


class EvalGateResponse(BaseModel):
    """Response schema representing the outcome of an automated evaluation gate."""

    gate_id: str
    model_name: str
    candidate_version: int
    champion_version: Optional[int] = None
    passed: bool
    promoted: bool
    decision_reason: str
    comparisons: List[MetricComparisonSchema] = Field(default_factory=list)
    timestamp: datetime


class RunPipelineRequest(BaseModel):
    """Request schema for executing a DVC data & training pipeline."""

    force: bool = Field(default=False, description="Force re-execution of cached stages")


class PipelineRunResponse(BaseModel):
    """Response schema summarizing a full DVC pipeline execution."""

    pipeline_id: str
    status: str
    executed_stages: List[str] = Field(default_factory=list)
    cached_stages: List[str] = Field(default_factory=list)
    duration_seconds: float
    stage_results: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
