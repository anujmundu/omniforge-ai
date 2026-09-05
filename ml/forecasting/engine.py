import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.base import BaseMLEstimator, ModelEvaluationResult, TaskType


class ForecastingEngine(BaseMLEstimator):
    """Production Time-Series Forecasting Engine with automated lag & rolling feature generation."""

    def __init__(
        self,
        model_id: str,
        lags: int = 7,
        rolling_windows: Optional[List[int]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_id=model_id, task_type=TaskType.FORECASTING)
        self.lags = lags
        self.rolling_windows = rolling_windows or [3, 7]
        self.hyperparameters = hyperparameters or {}
        self.date_col: Optional[str] = None
        self.target_name: str = "demand"
        self.last_known_history: Optional[pd.Series] = None
        self._init_estimator()

    def _init_estimator(self) -> None:
        """Instantiate underlying regression estimator for forecasting."""
        params = {"n_estimators": 100, "random_state": 42, **self.hyperparameters}
        self.estimator = GradientBoostingRegressor(**params)

    def _create_features(self, series: pd.Series, dates: Optional[pd.Series] = None) -> pd.DataFrame:
        """Generate lag, rolling window, and temporal features from time series."""
        df = pd.DataFrame({"target": series.values})

        # Lags
        for lag in range(1, self.lags + 1):
            df[f"lag_{lag}"] = df["target"].shift(lag)

        # Rolling statistics
        for w in self.rolling_windows:
            df[f"rolling_mean_{w}"] = df["target"].shift(1).rolling(window=w).mean()
            df[f"rolling_std_{w}"] = df["target"].shift(1).rolling(window=w).std()

        # Date features
        if dates is not None:
            dt_series = pd.to_datetime(dates)
            df["day_of_week"] = dt_series.dt.dayofweek.values
            df["day_of_month"] = dt_series.dt.day.values
            df["month"] = dt_series.dt.month.values

        return df

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        date_col: Optional[str] = None,
        target_col: str = "value",
        **kwargs,
    ) -> "ForecastingEngine":
        """Fit forecasting engine on historical time series data."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.date_col = date_col
        self.target_name = target_col

        if target_col in X.columns:
            target_series = X[target_col]
        elif y is not None:
            target_series = y
        else:
            raise ValueError(f"Target column '{target_col}' not found in input DataFrame.")

        date_series = X[date_col] if date_col and date_col in X.columns else None

        # Build feature matrix
        df_feat = self._create_features(target_series, date_series)
        df_clean = df_feat.dropna().reset_index(drop=True)

        y_train = df_clean["target"]
        X_train = df_clean.drop(columns=["target"])
        self.feature_names = list(X_train.columns)

        start_time = time.perf_counter()
        self.estimator.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time

        self.last_known_history = target_series.tail(max(self.lags, max(self.rolling_windows)) * 2)
        self.is_fitted = True

        self.metadata = {
            "lags": self.lags,
            "rolling_windows": self.rolling_windows,
            "training_duration_sec": round(training_time, 4),
            "historical_points": len(target_series),
            "generated_features": self.feature_names,
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict directly on generated feature DataFrame."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.model_id}' is not fitted.")
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.feature_names if c in X.columns]
            if len(cols) == len(self.feature_names):
                return self.estimator.predict(X[self.feature_names])

        # If scalar or int passed (e.g. horizon)
        return self.forecast_horizon(horizon=len(X))

    def forecast_horizon(self, horizon: int = 7) -> np.ndarray:
        """Recursive multi-step future forecasting for `horizon` periods."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.model_id}' is not fitted.")

        history = list(self.last_known_history.values)
        forecasts = []

        for _ in range(horizon):
            row: Dict[str, float] = {}
            # Compute lags from rolling history
            for lag in range(1, self.lags + 1):
                row[f"lag_{lag}"] = float(history[-lag])

            # Compute rolling windows
            for w in self.rolling_windows:
                window_slice = history[-w:]
                row[f"rolling_mean_{w}"] = float(np.mean(window_slice))
                row[f"rolling_std_{w}"] = float(np.std(window_slice))

            # Optional calendar defaults
            for cal in ["day_of_week", "day_of_month", "month"]:
                if cal in self.feature_names:
                    row[cal] = 0.0

            df_row = pd.DataFrame([row])[self.feature_names]
            pred = float(self.estimator.predict(df_row)[0])
            forecasts.append(pred)
            history.append(pred)

        return np.array(forecasts)

    def evaluate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> ModelEvaluationResult:
        """Evaluate forecasting accuracy against test series."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        target_series = X[self.target_name] if self.target_name in X.columns else (y if y is not None else X.iloc[:, 0])
        date_series = X[self.date_col] if self.date_col and self.date_col in X.columns else None

        df_feat = self._create_features(target_series, date_series).dropna().reset_index(drop=True)
        y_true = df_feat["target"]
        X_eval = df_feat.drop(columns=["target"])

        start_time = time.perf_counter()
        y_pred = self.estimator.predict(X_eval)
        latency_ms = ((time.perf_counter() - start_time) / max(len(X_eval), 1)) * 1000

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        
        # Weighted Absolute Percentage Error (WAPE)
        wape = float(np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1e-6))

        metrics = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4),
            "wape": round(wape, 4),
        }

        return ModelEvaluationResult(
            task_type=TaskType.FORECASTING,
            model_name="gradient_boosting_forecaster",
            primary_metric_name="wape",
            primary_metric_value=metrics["wape"],
            metrics=metrics,
            parameters={"lags": self.lags, "rolling_windows": self.rolling_windows},
            inference_latency_ms=round(latency_ms, 3),
            dataset_rows=len(X),
            dataset_features=len(self.feature_names),
        )

    def save(self, directory: Union[str, Path]) -> str:
        """Serialize forecasting engine and historical context."""
        out_dir = Path(directory) / self.model_id
        out_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = out_dir / "model.joblib"
        meta_path = out_dir / "metadata.json"

        bundle = {
            "model_id": self.model_id,
            "lags": self.lags,
            "rolling_windows": self.rolling_windows,
            "hyperparameters": self.hyperparameters,
            "feature_names": self.feature_names,
            "date_col": self.date_col,
            "target_name": self.target_name,
            "last_known_history": self.last_known_history,
            "is_fitted": self.is_fitted,
            "estimator": self.estimator,
        }
        joblib.dump(bundle, bundle_path)

        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        return str(bundle_path)

    @classmethod
    def load(cls, artifact_path: Union[str, Path]) -> "ForecastingEngine":
        """Load serialized forecasting engine."""
        path = Path(artifact_path)
        if path.is_dir():
            path = path / "model.joblib"

        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")

        bundle = joblib.load(path)
        engine = cls(
            model_id=bundle["model_id"],
            lags=bundle["lags"],
            rolling_windows=bundle["rolling_windows"],
            hyperparameters=bundle["hyperparameters"],
        )
        engine.feature_names = bundle["feature_names"]
        engine.date_col = bundle["date_col"]
        engine.target_name = bundle["target_name"]
        engine.last_known_history = bundle["last_known_history"]
        engine.is_fitted = bundle["is_fitted"]
        engine.estimator = bundle["estimator"]

        meta_path = path.parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                engine.metadata = json.load(f)

        return engine
