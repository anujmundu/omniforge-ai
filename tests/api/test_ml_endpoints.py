import pytest
from httpx import AsyncClient


@pytest.fixture
async def ml_test_project_id(client: AsyncClient, engineer_headers) -> str:
    res = await client.post(
        "/api/v1/projects",
        json={"name": "ML Engine Test Project", "slug": "ml-engine-test-proj"},
        headers=engineer_headers,
    )
    assert res.status_code == 201
    return res.json()["id"]


@pytest.mark.asyncio
async def test_ml_train_classification_and_predict(client: AsyncClient, engineer_headers, ml_test_project_id):
    # 1. Train classification
    records = []
    for i in range(50):
        records.append(
            {
                "tenure": i + 1,
                "monthly_charges": 30.0 + i * 1.5,
                "contract": "Month-to-month" if i % 2 == 0 else "Two year",
                "churn": 1 if i < 20 else 0,
            }
        )

    train_payload = {
        "project_id": ml_test_project_id,
        "model_id": "api_churn_rf_v1",
        "algorithm": "random_forest",
        "dataset_records": records,
        "target_column": "churn",
        "validation_split": 0.2,
    }
    train_res = await client.post("/api/v1/ml/train/classification", json=train_payload, headers=engineer_headers)
    assert train_res.status_code == 201
    train_data = train_res.json()
    assert train_data["status"] == "COMPLETED"
    assert train_data["model_id"] == "api_churn_rf_v1"
    assert "accuracy" in train_data["evaluation"]["metrics"]

    # 2. Real-time Inference
    infer_payload = {
        "model_id": "api_churn_rf_v1",
        "records": [
            {"tenure": 5, "monthly_charges": 85.0, "contract": "Month-to-month"},
            {"tenure": 60, "monthly_charges": 45.0, "contract": "Two year"},
        ],
    }
    infer_res = await client.post("/api/v1/ml/predict", json=infer_payload, headers=engineer_headers)
    assert infer_res.status_code == 200
    infer_data = infer_res.json()
    assert len(infer_data["predictions"]) == 2
    assert infer_data["probabilities"] is not None
    assert infer_data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_ml_train_regression_and_predict(client: AsyncClient, engineer_headers, ml_test_project_id):
    records = []
    for i in range(40):
        records.append(
            {
                "sqft": 1000 + i * 50,
                "bedrooms": 2 + (i % 3),
                "price": 200000 + i * 10000,
            }
        )

    train_payload = {
        "project_id": ml_test_project_id,
        "model_id": "api_price_ridge_v1",
        "algorithm": "ridge",
        "dataset_records": records,
        "target_column": "price",
    }
    train_res = await client.post("/api/v1/ml/train/regression", json=train_payload, headers=engineer_headers)
    assert train_res.status_code == 201
    assert train_res.json()["evaluation"]["primary_metric_name"] == "r2_score"

    infer_res = await client.post(
        "/api/v1/ml/predict",
        json={"model_id": "api_price_ridge_v1", "records": [{"sqft": 1500, "bedrooms": 3}]},
        headers=engineer_headers,
    )
    assert infer_res.status_code == 200
    assert len(infer_res.json()["predictions"]) == 1


@pytest.mark.asyncio
async def test_ml_train_anomaly_and_predict(client: AsyncClient, engineer_headers, ml_test_project_id):
    records = []
    for i in range(50):
        records.append(
            {
                "amount": 20.0 + i * 2.0 if i < 48 else 9999.0,
                "frequency": 1 + (i % 4),
            }
        )

    train_payload = {
        "project_id": ml_test_project_id,
        "model_id": "api_fraud_iforest_v1",
        "algorithm": "isolation_forest",
        "dataset_records": records,
        "contamination": 0.05,
    }
    train_res = await client.post("/api/v1/ml/train/anomaly", json=train_payload, headers=engineer_headers)
    assert train_res.status_code == 201
    assert "detected_anomalies" in train_res.json()["evaluation"]["metrics"]

    infer_res = await client.post(
        "/api/v1/ml/predict",
        json={"model_id": "api_fraud_iforest_v1", "records": [{"amount": 10000.0, "frequency": 1}]},
        headers=engineer_headers,
    )
    assert infer_res.status_code == 200
    assert infer_res.json()["anomaly_scores"] is not None


@pytest.mark.asyncio
async def test_ml_train_forecasting_and_predict(client: AsyncClient, engineer_headers, ml_test_project_id):
    records = []
    for i in range(30):
        records.append(
            {
                "step": i,
                "demand": 100 + i * 3 + (i % 5),
            }
        )

    train_payload = {
        "project_id": ml_test_project_id,
        "model_id": "api_demand_forecast_v1",
        "dataset_records": records,
        "target_column": "demand",
        "lags": 3,
    }
    train_res = await client.post("/api/v1/ml/train/forecasting", json=train_payload, headers=engineer_headers)
    assert train_res.status_code == 201

    infer_res = await client.post(
        "/api/v1/ml/predict",
        json={"model_id": "api_demand_forecast_v1", "records": [{"dummy": 1}], "horizon": 5},
        headers=engineer_headers,
    )
    assert infer_res.status_code == 200
    assert len(infer_res.json()["predictions"]) == 5


@pytest.mark.asyncio
async def test_list_registered_models(client: AsyncClient, engineer_headers):
    res = await client.get("/api/v1/ml/models", headers=engineer_headers)
    assert res.status_code == 200
    assert isinstance(res.json(), list)
