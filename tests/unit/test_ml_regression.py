import tempfile
import numpy as np
import pandas as pd
import pytest
from ml.base import TaskType
from ml.regression.engine import RegressionEngine


@pytest.fixture
def synthetic_price_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "sqft": np.random.uniform(500, 3500, size=n),
        "bedrooms": np.random.randint(1, 6, size=n),
        "neighborhood": np.random.choice(["Downtown", "Suburbs", "Rural"], size=n),
    })
    # Target price
    price = df["sqft"] * 150 + df["bedrooms"] * 10000 + np.random.normal(0, 5000, size=n)
    return df, price


def test_regression_engine_fit_evaluate_predict(synthetic_price_data):
    X, y = synthetic_price_data
    engine = RegressionEngine(model_id="test_price_rf", algorithm="random_forest")
    engine.fit(X, y, target_name="price")

    assert engine.is_fitted is True

    preds = engine.predict(X)
    assert len(preds) == len(X)
    assert np.all(preds > 0)

    eval_result = engine.evaluate(X, y)
    assert eval_result.task_type == TaskType.REGRESSION
    assert "rmse" in eval_result.metrics
    assert "r2_score" in eval_result.metrics
    assert eval_result.metrics["r2_score"] > 0.7


def test_regression_engine_save_load(synthetic_price_data):
    X, y = synthetic_price_data
    engine = RegressionEngine(model_id="test_ridge_reg", algorithm="ridge")
    engine.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_path = engine.save(tmp_dir)
        loaded = RegressionEngine.load(saved_path)

        assert loaded.model_id == "test_ridge_reg"
        assert loaded.is_fitted is True

        orig_preds = engine.predict(X)
        loaded_preds = loaded.predict(X)
        assert np.allclose(orig_preds, loaded_preds, rtol=1e-4)
