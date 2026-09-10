from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from ml.base import BasePreprocessor


class AutoColumnTransformer(BasePreprocessor):
    """Production-grade automated preprocessing pipeline for tabular data."""

    def __init__(
        self,
        scaler_type: str = "standard",
        handle_unknown: str = "ignore",
    ):
        self.scaler_type = scaler_type
        self.handle_unknown = handle_unknown
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.pipeline: Optional[ColumnTransformer] = None
        self.transformed_feature_names: List[str] = []

    def _detect_column_types(self, X: pd.DataFrame) -> None:
        """Automatically detect numeric and categorical column names."""
        self.numeric_features = []
        self.categorical_features = []

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.numeric_features.append(str(col))
            else:
                self.categorical_features.append(str(col))

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AutoColumnTransformer":
        """Fit preprocessors on feature DataFrame."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self._detect_column_types(X)

        transformers = []

        # Numerical pipeline
        if self.numeric_features:
            scaler = RobustScaler() if self.scaler_type == "robust" else StandardScaler()
            num_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", scaler),
                ]
            )
            transformers.append(("numeric", num_pipe, self.numeric_features))

        # Categorical pipeline
        if self.categorical_features:
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoder", OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False)),
                ]
            )
            transformers.append(("categorical", cat_pipe, self.categorical_features))

        if not transformers:
            raise ValueError("Input dataset contains no recognizable numeric or categorical features.")

        self.pipeline = ColumnTransformer(transformers=transformers, remainder="drop")
        self.pipeline.fit(X)

        # Extract output feature names
        self.transformed_feature_names = []
        if self.numeric_features:
            self.transformed_feature_names.extend(self.numeric_features)
        if self.categorical_features:
            cat_encoder = self.pipeline.named_transformers_["categorical"].named_steps["encoder"]
            encoded_cats = cat_encoder.get_feature_names_out(self.categorical_features)
            self.transformed_feature_names.extend(encoded_cats.tolist())

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input features using fitted pipelines."""
        if self.pipeline is None:
            raise RuntimeError("AutoColumnTransformer must be fitted before transform().")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Ensure all expected columns exist, filling missing with NaN if absent
        expected_cols = self.numeric_features + self.categorical_features
        for col in expected_cols:
            if col not in X.columns:
                X[col] = np.nan

        return self.pipeline.transform(X[expected_cols])
