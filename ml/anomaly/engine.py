import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from ml.base import BaseMLEstimator, ModelEvaluationResult, TaskType
from ml.preprocessing.pipeline import AutoColumnTransformer


class AnomalyEngine(BaseMLEstimator):
    """Production unsupervised anomaly and outlier detection engine."""

    SUPPORTED_ALGORITHMS = {
        "isolation_forest": IsolationForest,
        "local_outlier_factor": LocalOutlierFactor,
        "one_class_svm": OneClassSVM,
    }

    def __init__(
        self,
        model_id: str,
        algorithm: str = "isolation_forest",
        contamination: float = 0.05,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_id=model_id, task_type=TaskType.ANOMALY_DETECTION)
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Supported: {list(self.SUPPORTED_ALGORITHMS.keys())}"
            )
        self.algorithm_name = algorithm
        self.contamination = contamination
        self.hyperparameters = hyperparameters or {}
        self.preprocessor = AutoColumnTransformer(scaler_type="robust")
        self._init_estimator()

    def _init_estimator(self) -> None:
        """Instantiate underlying anomaly detection estimator."""
        cls = self.SUPPORTED_ALGORITHMS[self.algorithm_name]
        default_params: Dict[str, Any] = {}
        if self.algorithm_name == "isolation_forest":
            default_params = {
                "contamination": self.contamination,
                "random_state": 42,
                "n_estimators": 100,
            }
        elif self.algorithm_name == "local_outlier_factor":
            default_params = {
                "contamination": self.contamination,
                "novelty": True,
                "n_neighbors": 20,
            }
        elif self.algorithm_name == "one_class_svm":
            default_params = {
                "nu": min(max(self.contamination, 0.01), 0.5),
                "kernel": "rbf",
                "gamma": "scale",
            }

        params = {**default_params, **self.hyperparameters}
        self.estimator = cls(**params)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs,
    ) -> "AnomalyEngine":
        """Fit unsupervised anomaly detector."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names = list(X.columns)

        X_transformed = self.preprocessor.fit_transform(X)

        start_time = time.perf_counter()
        self.estimator.fit(X_transformed)
        training_time = time.perf_counter() - start_time

        self.is_fitted = True
        self.metadata = {
            "algorithm": self.algorithm_name,
            "contamination": self.contamination,
            "training_duration_sec": round(training_time, 4),
            "samples_analyzed": len(X),
            "num_raw_features": len(self.feature_names),
            "num_transformed_features": len(self.preprocessor.transformed_feature_names),
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict outlier labels: 1 for inlier/normal, -1 for outlier/anomaly."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.model_id}' is not fitted.")
        X_transformed = self.preprocessor.transform(X)
        return self.estimator.predict(X_transformed)

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """Compute anomaly score for each sample. Lower score = more anomalous."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.model_id}' is not fitted.")
        X_transformed = self.preprocessor.transform(X)
        if hasattr(self.estimator, "score_samples"):
            return self.estimator.score_samples(X_transformed)
        elif hasattr(self.estimator, "decision_function"):
            return self.estimator.decision_function(X_transformed)
        raise NotImplementedError("Estimator does not support continuous anomaly scoring.")

    def evaluate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> ModelEvaluationResult:
        """Evaluate anomaly distribution and outlier statistics."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        start_time = time.perf_counter()
        preds = self.predict(X)
        scores = self.score_samples(X)
        latency_ms = ((time.perf_counter() - start_time) / max(len(X), 1)) * 1000

        num_anomalies = int(np.sum(preds == -1))
        num_inliers = int(np.sum(preds == 1))
        detected_rate = float(num_anomalies / max(len(X), 1))

        metrics: Dict[str, Any] = {
            "total_samples": len(X),
            "detected_anomalies": num_anomalies,
            "detected_inliers": num_inliers,
            "anomaly_percentage": round(detected_rate * 100, 2),
            "mean_anomaly_score": round(float(np.mean(scores)), 4),
            "min_anomaly_score": round(float(np.min(scores)), 4),
            "max_anomaly_score": round(float(np.max(scores)), 4),
        }

        # If ground truth labels are provided (e.g. 1 for fraud, 0 for normal)
        if y is not None:
            # Map sklearn (-1 anomaly, 1 normal) to (1 anomaly, 0 normal)
            y_binary_pred = np.where(preds == -1, 1, 0)
            from sklearn.metrics import f1_score, precision_score, recall_score
            metrics["ground_truth_f1"] = round(float(f1_score(y, y_binary_pred, zero_division=0)), 4)
            metrics["ground_truth_precision"] = round(float(precision_score(y, y_binary_pred, zero_division=0)), 4)
            metrics["ground_truth_recall"] = round(float(recall_score(y, y_binary_pred, zero_division=0)), 4)

        return ModelEvaluationResult(
            task_type=TaskType.ANOMALY_DETECTION,
            model_name=f"{self.algorithm_name}_detector",
            primary_metric_name="anomaly_percentage",
            primary_metric_value=metrics["anomaly_percentage"],
            metrics=metrics,
            parameters={"contamination": self.contamination, **self.hyperparameters},
            inference_latency_ms=round(latency_ms, 3),
            dataset_rows=len(X),
            dataset_features=len(self.feature_names),
        )

    def save(self, directory: Union[str, Path]) -> str:
        """Serialize anomaly detection pipeline."""
        out_dir = Path(directory) / self.model_id
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = out_dir / "model.joblib"
        meta_path = out_dir / "metadata.json"

        bundle = {
            "model_id": self.model_id,
            "algorithm_name": self.algorithm_name,
            "contamination": self.contamination,
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "preprocessor": self.preprocessor,
            "estimator": self.estimator,
        }
        joblib.dump(bundle, bundle_path)

        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        return str(bundle_path)

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "AnomalyEngine":
        """Load serialized anomaly model bundle."""
        path = Path(artifact_path)
        if path.is_dir():
            path = path / "model.joblib"

        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")

        bundle = joblib.load(path)
        engine = cls(
            model_id=bundle["model_id"],
            algorithm=bundle["algorithm_name"],
            contamination=bundle["contamination"],
            hyperparameters=bundle["hyperparameters"],
        )
        engine.feature_names = bundle["feature_names"]
        engine.is_fitted = bundle["is_fitted"]
        engine.preprocessor = bundle["preprocessor"]
        engine.estimator = bundle["estimator"]

        meta_path = path.parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                engine.metadata = json.load(f)

        return engine
