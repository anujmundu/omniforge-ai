"""OmniForge MLOps Core Data Models and Domain Enums.

Defines schemas and state representations for experiment runs, model versions,
registry stages, automated evaluation gates, and DVC pipeline stages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ModelStage(str, Enum):
    """Operational lifecycle stages for registered models."""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class PipelineStatus(str, Enum):
    """Execution status for MLOps and DVC pipeline stages."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ExperimentRun(BaseModel):
    """Represents an experiment tracking run recording parameters, metrics, and artifacts."""

    run_id: str = Field(default_factory=lambda: f"run_{uuid4().hex[:10]}")
    experiment_name: str = "default_experiment"
    status: PipelineStatus = PipelineStatus.SUCCESS
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)
    artifact_uris: List[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None


class ModelVersion(BaseModel):
    """Represents a specific version of a registered machine learning model."""

    model_name: str
    version: int
    run_id: str
    stage: ModelStage = ModelStage.NONE
    description: str = ""
    metrics: Dict[str, float] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    artifact_uri: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = Field(default_factory=dict)


class RegisteredModel(BaseModel):
    """Represents a named model entity in the model registry containing multiple versions."""

    name: str
    description: str = ""
    latest_version: int = 0
    versions: List[ModelVersion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = Field(default_factory=dict)

    def get_version(self, version: int) -> Optional[ModelVersion]:
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def get_stage_model(self, stage: ModelStage) -> Optional[ModelVersion]:
        for v in reversed(self.versions):
            if v.stage == stage:
                return v
        return None


class MetricComparison(BaseModel):
    """Detailed comparison between candidate and champion baseline for a specific metric."""

    metric_name: str
    candidate_value: float
    champion_value: Optional[float] = None
    delta: Optional[float] = None
    threshold: float
    passed: bool
    description: str = ""


class EvalGateResult(BaseModel):
    """Result of an automated model regression and promotion evaluation gate."""

    gate_id: str = Field(default_factory=lambda: f"gate_{uuid4().hex[:10]}")
    model_name: str
    candidate_version: int
    champion_version: Optional[int] = None
    passed: bool
    promoted: bool = False
    decision_reason: str
    comparisons: List[MetricComparison] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StageDefinition(BaseModel):
    """Declarative specification of a pipeline execution stage."""

    name: str
    command: str
    deps: List[str] = Field(default_factory=list)
    outs: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)


class PipelineRunResult(BaseModel):
    """Summary of a full DVC pipeline execution."""

    pipeline_id: str = Field(default_factory=lambda: f"pipe_{uuid4().hex[:10]}")
    status: PipelineStatus
    executed_stages: List[str] = Field(default_factory=list)
    cached_stages: List[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    stage_results: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
