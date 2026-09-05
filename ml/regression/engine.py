import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from ml.base import BaseMLEstimator, ModelEvaluationResult, TaskType
from ml.preprocessing.pipeline import AutoColumnTransformer


class RegressionEngine(BaseMLEstimator):
    """Production regression engine for continuous value prediction."""

    SUPPORTED_ALGORITHMS = {
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "ridge": Ridge,
    }

    def __init__(
        self,
        model_id: str,
        algorithm: str = "random_forest",
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_id=model_id, task_type=TaskType.REGRESSION)
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Supported: {list(self.SUPPORTED_ALGORITHMS.keys())}"
            )
        self.algorithm_name = algorithm
        self.hyperparameters = hyperparameters or {}
        self.preprocessor = AutoColumnTransformer()
        self._init_estimator()

    def _init_estimator(self) -> None:
        """Instantiate underlying scikit-learn regressor."""
        cls = self.SUPPORTED_ALGORITHMS[self.algorithm_name]
        default_params = {"random_state": 42}
        if self.algorithm_name in ("random_forest", "gradient_boosting"):
            default_params["n_estimators"] = 100
        elif self.algorithm_name == "ridge":
            default_params["alpha"] = 1.0

        params = {**default_params, **self.hyperparameters}
        self.estimator = cls(**params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str = "target",
        validation_split: float = 0.2,
    ) -> "RegressionEngine":
        """Train regressor with automated preprocessing."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=target_name)

        self.feature_names = list(X.columns)
        self.target_name = target_name

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_val_transformed = self.preprocessor.transform(X_val)

        start_time = time.perf_counter()
        self.estimator.fit(X_train_transformed, y_train)
        training_time = time.perf_counter() - start_time

        self.is_fitted = True
        self.metadata = {
            "algorithm": self.algorithm_name,
            "training_duration_sec": round(training_time, 4),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "num_raw_features": len(self.feature_names),
            "num_transformed_features": len(self.preprocessor.transformed_feature_names),
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict continuous target values."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.model_id}' is not fitted.")
        X_transformed = self.preprocessor.transform(X)
        return self.estimator.predict(X_transformed)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelEvaluationResult:
        """Calculate regression evaluation metrics."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        start_time = time.perf_counter()
        y_pred = self.predict(X)
        latency_ms = ((time.perf_counter() - start_time) / max(len(X), 1)) * 1000

        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae = float(mean_absolute_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))
        evs = float(explained_variance_score(y, y_pred))
        
        try:
            mape = float(mean_absolute_percentage_error(y, y_pred))
        except Exception:
            mape = 0.0

        # Feature importances
        feature_importance: Optional[Dict[str, float]] = None
        if hasattr(self.estimator, "feature_importances_"):
            importances = self.estimator.feature_importances_
            names = self.preprocessor.transformed_feature_names
            if len(names) == len(importances):
                sorted_idx = np.argsort(importances)[::-1][:15]
                feature_importance = {
                    names[i]: round(float(importances[i]), 4) for i in sorted_idx
                }

        metrics = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4),
            "mape": round(mape, 4),
            "explained_variance": round(evs, 4),
        }

        return ModelEvaluationResult(
            task_type=TaskType.REGRESSION,
            model_name=f"{self.algorithm_name}_regressor",
            primary_metric_name="r2_score",
            primary_metric_value=metrics["r2_score"],
            metrics=metrics,
            feature_importance=feature_importance,
            parameters=self.hyperparameters,
            inference_latency_ms=round(latency_ms, 3),
            dataset_rows=len(X),
            dataset_features=len(self.feature_names),
        )

    def save(self, directory: Union[str, Path]) -> str:
        """Serialize model, preprocessor, and metadata."""
        out_dir = Path(directory) / self.model_id
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = out_dir / "model.joblib"
        meta_path = out_dir / "metadata.json"

        bundle = {
            "model_id": self.model_id,
            "algorithm_name": self.algorithm_name,
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "is_fitted": self.is_fitted,
            "preprocessor": self.preprocessor,
            "estimator": self.estimator,
        }
        joblib.dump(bundle, bundle_path)

        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        return str(bundle_path)

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "RegressionEngine":
        """Load serialized model bundle."""
        path = Path(artifact_path)
        if path.is_dir():
            path = path / "model.joblib"

        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")

        bundle = joblib.load(path)
        engine = cls(
            model_id=bundle["model_id"],
            algorithm=bundle["algorithm_name"],
            hyperparameters=bundle["hyperparameters"],
        )
        engine.feature_names = bundle["feature_names"]
        engine.target_name = bundle["target_name"]
        engine.is_fitted = bundle["is_fitted"]
        engine.preprocessor = bundle["preprocessor"]
        engine.estimator = bundle["estimator"]

        meta_path = path.parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                engine.metadata = json.load(f)

        return engine
