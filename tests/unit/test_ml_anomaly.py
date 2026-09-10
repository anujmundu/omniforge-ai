import tempfile

import numpy as np
import pandas as pd
import pytest

from ml.anomaly.engine import AnomalyEngine
from ml.base import TaskType


@pytest.fixture
def synthetic_transaction_data():
    np.random.seed(42)
    n = 100
    # 95 normal, 5 anomalous
    normal_amounts = np.random.normal(50, 10, size=95)
    anomalous_amounts = np.random.uniform(500, 1000, size=5)
    amounts = np.concatenate([normal_amounts, anomalous_amounts])

    df = pd.DataFrame(
        {
            "amount": amounts,
            "tx_count_1h": np.random.randint(1, 5, size=n),
            "location": np.random.choice(["Domestic", "International"], size=n),
        }
    )
    return df


def test_anomaly_engine_fit_evaluate_predict(synthetic_transaction_data):
    df = synthetic_transaction_data
    engine = AnomalyEngine(
        model_id="test_iforest_tx",
        algorithm="isolation_forest",
        contamination=0.05,
    )
    engine.fit(df)

    assert engine.is_fitted is True

    preds = engine.predict(df)
    assert len(preds) == len(df)
    # Sklearn output: 1 for inlier, -1 for outlier
    assert set(np.unique(preds)).issubset({1, -1})
    assert -1 in preds

    scores = engine.score_samples(df)
    assert len(scores) == len(df)

    eval_result = engine.evaluate(df)
    assert eval_result.task_type == TaskType.ANOMALY_DETECTION
    assert "detected_anomalies" in eval_result.metrics
    assert eval_result.metrics["detected_anomalies"] > 0


def test_anomaly_engine_save_load(synthetic_transaction_data):
    df = synthetic_transaction_data
    engine = AnomalyEngine(model_id="test_anomaly_save", algorithm="isolation_forest")
    engine.fit(df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_path = engine.save(tmp_dir)
        loaded = AnomalyEngine.load(saved_path)

        assert loaded.model_id == "test_anomaly_save"
        orig_preds = engine.predict(df)
        loaded_preds = loaded.predict(df)
        assert np.array_equal(orig_preds, loaded_preds)
