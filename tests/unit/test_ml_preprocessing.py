import numpy as np
import pandas as pd
import pytest
from ml.preprocessing.pipeline import AutoColumnTransformer


def test_auto_column_transformer_mixed_types():
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 45, 50],
        "salary": [50000.0, 60000.0, 75000.0, np.nan, 95000.0],
        "department": ["Sales", "Engineering", "HR", "Sales", None],
    })

    transformer = AutoColumnTransformer()
    transformed = transformer.fit_transform(df)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 5
    # Numerical: age, salary (2) + Categorical one-hot features (>=3)
    assert transformed.shape[1] >= 5
    assert not np.isnan(transformed).any()


def test_auto_column_transformer_unseen_categories():
    train_df = pd.DataFrame({
        "feature_num": [1.0, 2.0, 3.0],
        "category": ["A", "B", "A"],
    })
    test_df = pd.DataFrame({
        "feature_num": [4.0],
        "category": ["Unseen_C"],
    })

    transformer = AutoColumnTransformer()
    transformer.fit(train_df)
    transformed_test = transformer.transform(test_df)

    assert transformed_test.shape[0] == 1
    assert not np.isnan(transformed_test).any()
