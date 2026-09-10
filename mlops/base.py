"""OmniForge MLOps Core Data Models and Domain Enums."""
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
