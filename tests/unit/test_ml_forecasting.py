import tempfile
import numpy as np
import pandas as pd
import pytest
from ml.base import TaskType
from ml.forecasting.engine import ForecastingEngine


@pytest.fixture
def synthetic_demand_series():
    np.random.seed(42)
    n = 60
    trend = np.linspace(100, 200, n)
    seasonality = 15 * np.sin(np.linspace(0, 8 * np.pi, n))
    noise = np.random.normal(0, 3, n)
    values = trend + seasonality + noise

    dates = pd.date_range(start="2026-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "demand": values,
    })
    return df


def test_forecasting_engine_fit_horizon(synthetic_demand_series):
    df = synthetic_demand_series
    engine = ForecastingEngine(model_id="test_demand_forecaster", lags=5, rolling_windows=[3, 5])
    engine.fit(df, date_col="timestamp", target_col="demand")

    assert engine.is_fitted is True

    # 7-day horizon forecast
    horizon_preds = engine.forecast_horizon(horizon=7)
    assert len(horizon_preds) == 7
    assert np.all(horizon_preds > 50)

    # Evaluate
    eval_res = engine.evaluate(df)
    assert eval_res.task_type == TaskType.FORECASTING
    assert "rmse" in eval_res.metrics
    assert "wape" in eval_res.metrics
    assert eval_res.metrics["wape"] < 0.2


def test_forecasting_engine_save_load(synthetic_demand_series):
    df = synthetic_demand_series
    engine = ForecastingEngine(model_id="test_forecast_save", lags=3)
    engine.fit(df, date_col="timestamp", target_col="demand")

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_path = engine.save(tmp_dir)
        loaded = ForecastingEngine.load(saved_path)

        assert loaded.model_id == "test_forecast_save"
        orig_preds = engine.forecast_horizon(horizon=5)
        loaded_preds = loaded.forecast_horizon(horizon=5)
        assert np.allclose(orig_preds, loaded_preds, rtol=1e-4)
