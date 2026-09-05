from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    FORECASTING = "FORECASTING"


class ModelEvaluationResult(BaseModel):
    task_type: TaskType
    model_name: str
    primary_metric_name: str
    primary_metric_value: float
    metrics: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    parameters: Optional[Dict[str, Any]] = None
    inference_latency_ms: Optional[float] = None
    dataset_rows: Optional[int] = None
    dataset_features: Optional[int] = None


class BasePreprocessor(ABC):
    """Abstract base class for all feature preprocessing transformers."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BasePreprocessor":
        """Fit preprocessor parameters on training data."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input features using fitted parameters."""
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform in a single pass."""
        return self.fit(X, y).transform(X)


class BaseMLEstimator(ABC):
    """Abstract base class for all OmniForge machine learning models."""

    def __init__(self, model_id: str, task_type: TaskType):
        self.model_id = model_id
        self.task_type = task_type
        self.is_fitted: bool = False
        self.feature_names: List[str] = []
        self.target_name: Optional[str] = None
        self.preprocessor: Optional[BasePreprocessor] = None
        self.estimator: Any = None
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> "BaseMLEstimator":
        """Train model estimator on input data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions."""
        pass

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate prediction probabilities (if supported)."""
        return None

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> ModelEvaluationResult:
        """Evaluate model performance and return structured metrics."""
        pass

    @abstractmethod
    def save(self, directory: Union[str, Path]) -> str:
        """Persist model bundle and metadata to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, artifact_path: Union[str, Path]) -> "BaseMLEstimator":
        """Load model bundle from disk."""
        pass
