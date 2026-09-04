from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field
from apps.api.models.experiment import ExperimentDomain, ExperimentStatus


class ExperimentBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=150)
    domain: ExperimentDomain = Field(default=ExperimentDomain.CLASSICAL_ML)
    model_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ExperimentCreate(ExperimentBase):
    project_id: str


class ExperimentUpdate(BaseModel):
    status: Optional[ExperimentStatus] = None
    metrics: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None


class ExperimentResponse(ExperimentBase):
    id: str
    project_id: str
    status: ExperimentStatus
    metrics: Optional[Dict[str, Any]]
    duration_seconds: Optional[float]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
