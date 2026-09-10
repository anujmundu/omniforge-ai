import tempfile

import numpy as np
import pandas as pd
import pytest

from ml.base import TaskType
from ml.classification.engine import ClassificationEngine


@pytest.fixture
def synthetic_churn_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "tenure": np.random.randint(1, 72, size=n),
            "monthly_charges": np.random.uniform(20, 120, size=n),
            "contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
            "tech_support": np.random.choice(["Yes", "No"], size=n),
        }
    )
    # Target correlated with tenure and contract
    churn = ((df["tenure"] < 20) & (df["contract"] == "Month-to-month")).astype(int)
    return df, churn


def test_classification_engine_fit_evaluate_predict(synthetic_churn_data):
    X, y = synthetic_churn_data
    engine = ClassificationEngine(model_id="test_churn_rf", algorithm="random_forest")
    engine.fit(X, y, target_name="churn")

    assert engine.is_fitted is True

    # Predictions
    preds = engine.predict(X)
    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})

    # Probabilities
    probs = engine.predict_proba(X)
    assert probs is not None
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Evaluation metrics
    eval_result = engine.evaluate(X, y)
    assert eval_result.task_type == TaskType.CLASSIFICATION
    assert "accuracy" in eval_result.metrics
    assert "f1_macro" in eval_result.metrics
    assert "roc_auc" in eval_result.metrics
    assert eval_result.metrics["accuracy"] >= 0.7


def test_classification_engine_save_and_load(synthetic_churn_data):
    X, y = synthetic_churn_data
    engine = ClassificationEngine(model_id="test_save_load", algorithm="logistic_regression")
    engine.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_path = engine.save(tmp_dir)
        loaded = ClassificationEngine.load(saved_path)

        assert loaded.model_id == "test_save_load"
        assert loaded.is_fitted is True

        orig_preds = engine.predict(X)
        loaded_preds = loaded.predict(X)
        assert np.array_equal(orig_preds, loaded_preds)
