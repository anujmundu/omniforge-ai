import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml.base import BaseMLEstimator, ModelEvaluationResult, TaskType
from ml.preprocessing.pipeline import AutoColumnTransformer


class ClassificationEngine(BaseMLEstimator):
    """Production classification engine with automated preprocessing and benchmarking."""

    SUPPORTED_ALGORITHMS = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
    }

    def __init__(
        self,
        model_id: str,
        algorithm: str = "random_forest",
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_id=model_id, task_type=TaskType.CLASSIFICATION)
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Supported: {list(self.SUPPORTED_ALGORITHMS.keys())}"
            )
        self.algorithm_name = algorithm
        self.hyperparameters = hyperparameters or {}
        self.classes_: Optional[np.ndarray] = None
        self.preprocessor = AutoColumnTransformer()
        self._init_estimator()

    def _init_estimator(self) -> None:
        """Instantiate underlying scikit-learn classifier."""
        cls = self.SUPPORTED_ALGORITHMS[self.algorithm_name]
        default_params = {"random_state": 42}
        if self.algorithm_name == "logistic_regression":
            default_params["max_iter"] = 1000
        elif self.algorithm_name in ("random_forest", "gradient_boosting"):
            default_params["n_estimators"] = 100

        params = {**default_params, **self.hyperparameters}
        self.estimator = cls(**params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str = "target",
        validation_split: float = 0.2,
    ) -> "ClassificationEngine":
        """Train classifier with automated preprocessing."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=target_name)

        self.feature_names = list(X.columns)
        self.target_name = target_name

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )

        # Fit preprocessor strictly on training features
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_val_transformed = self.preprocessor.transform(X_val)

        # Fit classifier
        start_time = time.perf_counter()
        self.estimator.fit(X_train_transformed, y_train)
        training_time = time.perf_counter() - start_time

        self.classes_ = self.estimator.classes_
        self.is_fitted = True

        # Store training metadata
        self.metadata = {
            "algorithm": self.algorithm_name,
            "training_duration_sec": round(training_time, 4),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "num_raw_features": len(self.feature_names),
            "num_transformed_features": len(self.preprocessor.transformed_feature_names),
            "classes": [str(c) for c in self.classes_],
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for input samples."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.model_id}' is not fitted.")
        X_transformed = self.preprocessor.transform(X)
        return self.estimator.predict(X_transformed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for input samples."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.model_id}' is not fitted.")
        X_transformed = self.preprocessor.transform(X)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X_transformed)
        elif hasattr(self.estimator, "decision_function"):
            decision = self.estimator.decision_function(X_transformed)
            prob = 1 / (1 + np.exp(-decision))
            return np.vstack([1 - prob, prob]).T
        raise NotImplementedError("Estimator does not support probability estimation.")

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelEvaluationResult:
        """Calculate comprehensive classification metrics."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        start_time = time.perf_counter()
        y_pred = self.predict(X)
        latency_ms = ((time.perf_counter() - start_time) / max(len(X), 1)) * 1000

        acc = float(accuracy_score(y, y_pred))
        f1_macro = float(f1_score(y, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y, y_pred, average="weighted", zero_division=0))
        prec = float(precision_score(y, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y, y_pred, average="weighted", zero_division=0))

        # ROC-AUC calculation
        roc_auc_val = 0.0
        try:
            y_proba = self.predict_proba(X)
            if len(self.classes_) == 2 and y_proba is not None:
                roc_auc_val = float(roc_auc_score(y, y_proba[:, 1]))
            elif len(self.classes_) > 2 and y_proba is not None:
                roc_auc_val = float(roc_auc_score(y, y_proba, multi_class="ovr"))
        except Exception:
            roc_auc_val = 0.0

        # Feature importances if available
        feature_importance: Optional[Dict[str, float]] = None
        if hasattr(self.estimator, "feature_importances_"):
            importances = self.estimator.feature_importances_
            names = self.preprocessor.transformed_feature_names
            if len(names) == len(importances):
                sorted_idx = np.argsort(importances)[::-1][:15]
                feature_importance = {names[i]: round(float(importances[i]), 4) for i in sorted_idx}

        metrics = {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "precision_weighted": round(prec, 4),
            "recall_weighted": round(rec, 4),
            "roc_auc": round(roc_auc_val, 4),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

        return ModelEvaluationResult(
            task_type=TaskType.CLASSIFICATION,
            model_name=f"{self.algorithm_name}_classifier",
            primary_metric_name="f1_macro",
            primary_metric_value=metrics["f1_macro"],
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
            "classes": self.classes_,
            "is_fitted": self.is_fitted,
            "preprocessor": self.preprocessor,
            "estimator": self.estimator,
        }
        joblib.dump(bundle, bundle_path)

        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        return str(bundle_path)

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "ClassificationEngine":
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
        engine.classes_ = bundle["classes"]
        engine.is_fitted = bundle["is_fitted"]
        engine.preprocessor = bundle["preprocessor"]
        engine.estimator = bundle["estimator"]

        meta_path = path.parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                engine.metadata = json.load(f)

        return engine
