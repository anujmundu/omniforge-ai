from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ml.base import ModelEvaluationResult, TaskType


class TrainClassificationRequest(BaseModel):
    project_id: str
    model_id: str = Field(..., min_length=2, max_length=100)
    algorithm: str = Field(default="random_forest")
    dataset_records: List[Dict[str, Any]] = Field(..., min_length=10)
    target_column: str = Field(default="target")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
    hyperparameters: Optional[Dict[str, Any]] = None


class TrainRegressionRequest(BaseModel):
    project_id: str
    model_id: str = Field(..., min_length=2, max_length=100)
    algorithm: str = Field(default="random_forest")
    dataset_records: List[Dict[str, Any]] = Field(..., min_length=10)
    target_column: str = Field(default="target")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
    hyperparameters: Optional[Dict[str, Any]] = None


class TrainAnomalyRequest(BaseModel):
    project_id: str
    model_id: str = Field(..., min_length=2, max_length=100)
    algorithm: str = Field(default="isolation_forest")
    dataset_records: List[Dict[str, Any]] = Field(..., min_length=10)
    contamination: float = Field(default=0.05, ge=0.001, le=0.5)
    hyperparameters: Optional[Dict[str, Any]] = None


class TrainForecastingRequest(BaseModel):
    project_id: str
    model_id: str = Field(..., min_length=2, max_length=100)
    dataset_records: List[Dict[str, Any]] = Field(..., min_length=15)
    target_column: str = Field(default="value")
    date_column: Optional[str] = None
    lags: int = Field(default=7, ge=1, le=30)
    hyperparameters: Optional[Dict[str, Any]] = None


class TrainJobResponse(BaseModel):
    status: str
    model_id: str
    task_type: TaskType
    artifact_uri: str
    evaluation: ModelEvaluationResult


class InferenceRequest(BaseModel):
    model_id: str
    records: List[Dict[str, Any]] = Field(..., min_length=1)
    horizon: Optional[int] = Field(default=7, ge=1, le=365)


class InferenceResponse(BaseModel):
    model_id: str
    task_type: TaskType
    predictions: List[Union[float, int, str]]
    probabilities: Optional[List[List[float]]] = None
    anomaly_scores: Optional[List[float]] = None
    latency_ms: float
    num_records: int


class ModelInfoResponse(BaseModel):
    model_id: str
    artifact_path: str
    metadata: Dict[str, Any]
