import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_experiment_tracking_and_artifacts_lifecycle(client: AsyncClient, engineer_headers):
    # 1. Create a project
    proj_res = await client.post(
        "/api/v1/projects",
        json={"name": "ML Experiment Project", "slug": "ml-exp-proj"},
        headers=engineer_headers,
    )
    assert proj_res.status_code == 201
    project_id = proj_res.json()["id"]

    # 2. Create an experiment run
    exp_payload = {
        "project_id": project_id,
        "name": "XGBoost_Hyperopt_Run_01",
        "domain": "CLASSICAL_ML",
        "model_name": "xgboost.XGBClassifier",
        "parameters": {
            "max_depth": 5,
            "learning_rate": 0.03,
            "n_estimators": 250,
            "subsample": 0.8,
        },
    }
    exp_res = await client.post("/api/v1/experiments", json=exp_payload, headers=engineer_headers)
    assert exp_res.status_code == 201
    exp_data = exp_res.json()
    experiment_id = exp_data["id"]
    assert exp_data["status"] == "RUNNING"
    assert exp_data["parameters"]["max_depth"] == 5

    # 3. Update experiment with evaluation metrics & completion status
    update_payload = {
        "status": "COMPLETED",
        "metrics": {
            "f1_score": 0.912,
            "roc_auc": 0.954,
            "precision": 0.901,
            "recall": 0.923,
            "inference_latency_ms": 4.82,
        },
        "duration_seconds": 18.45,
    }
    update_res = await client.patch(
        f"/api/v1/experiments/{experiment_id}",
        json=update_payload,
        headers=engineer_headers,
    )
    assert update_res.status_code == 200
    updated_exp = update_res.json()
    assert updated_exp["status"] == "COMPLETED"
    assert updated_exp["metrics"]["f1_score"] == 0.912
    assert updated_exp["duration_seconds"] == 18.45

    # 4. Register an output model artifact
    artifact_payload = {
        "name": "xgboost_churn_model.onnx",
        "artifact_type": "ONNX_MODEL",
        "uri": "s3://aiforge-models/churn/xgb_run_01.onnx",
        "size_bytes": 1048576,
        "checksum": "a1b2c3d4e5f607182930415263748596a1b2c3d4e5f607182930415263748596",
    }
    art_res = await client.post(
        f"/api/v1/experiments/{experiment_id}/artifacts",
        json=artifact_payload,
        headers=engineer_headers,
    )
    assert art_res.status_code == 201
    art_data = art_res.json()
    assert art_data["name"] == "xgboost_churn_model.onnx"
    assert art_data["artifact_type"] == "ONNX_MODEL"

    # 5. List artifacts for experiment
    list_art_res = await client.get(
        f"/api/v1/experiments/{experiment_id}/artifacts",
        headers=engineer_headers,
    )
    assert list_art_res.status_code == 200
    artifacts = list_art_res.json()
    assert len(artifacts) == 1
    assert artifacts[0]["name"] == "xgboost_churn_model.onnx"

    # 6. Query experiments by domain and status
    filter_res = await client.get(
        "/api/v1/experiments?domain=CLASSICAL_ML&status=COMPLETED",
        headers=engineer_headers,
    )
    assert filter_res.status_code == 200
    filtered = filter_res.json()
    assert len(filtered) >= 1
    assert any(e["id"] == experiment_id for e in filtered)
