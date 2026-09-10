"""API Integration tests for /api/v1/mlops REST endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_mlops_runs_lifecycle(client: AsyncClient, engineer_headers: dict):
    # 1. Start Run
    resp = await client.post(
        "/api/v1/mlops/runs",
        json={"experiment_name": "customer_churn", "tags": {"team": "mlops"}},
        headers=engineer_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    run_id = data["run_id"]
    assert data["experiment_name"] == "customer_churn"

    # 2. Log Params
    resp = await client.post(
        "/api/v1/mlops/runs/params",
        json={"run_id": run_id, "parameters": {"lr": 0.01, "max_depth": 6}},
        headers=engineer_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["parameters"]["lr"] == 0.01

    # 3. Log Metrics
    resp = await client.post(
        "/api/v1/mlops/runs/metrics",
        json={"run_id": run_id, "metrics": {"accuracy": 0.94, "f1_score": 0.93, "latency_p95_ms": 25.0}},
        headers=engineer_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["metrics"]["f1_score"] == 0.93

    # 4. End Run
    resp = await client.post(
        "/api/v1/mlops/runs/end",
        json={"run_id": run_id, "status": "SUCCESS"},
        headers=engineer_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "SUCCESS"

    # 5. Get Run by ID
    resp = await client.get(f"/api/v1/mlops/runs/{run_id}", headers=engineer_headers)
    assert resp.status_code == 200
    assert resp.json()["run_id"] == run_id

    # 6. List Runs
    resp = await client.get("/api/v1/mlops/runs", headers=engineer_headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


@pytest.mark.asyncio
async def test_mlops_model_registry_and_evaluation_gate(client: AsyncClient, engineer_headers: dict):
    # 1. Create Run 1 and register Champion
    r1 = await client.post("/api/v1/mlops/runs", json={"experiment_name": "propensity"}, headers=engineer_headers)
    run1_id = r1.json()["run_id"]
    await client.post(
        "/api/v1/mlops/runs/metrics",
        json={"run_id": run1_id, "metrics": {"f1_score": 0.90, "accuracy": 0.91, "latency_p95_ms": 30.0}},
        headers=engineer_headers,
    )
    await client.post("/api/v1/mlops/runs/end", json={"run_id": run1_id, "status": "SUCCESS"}, headers=engineer_headers)

    # Register Model v1
    resp_reg = await client.post(
        "/api/v1/mlops/models/register",
        json={"name": "propensity_model", "run_id": run1_id, "description": "Baseline v1"},
        headers=engineer_headers,
    )
    assert resp_reg.status_code == 201
    assert resp_reg.json()["version"] == 1

    # Transition v1 to Production
    resp_trans = await client.post(
        "/api/v1/mlops/models/transition",
        json={"model_name": "propensity_model", "version": 1, "target_stage": "Production"},
        headers=engineer_headers,
    )
    assert resp_trans.status_code == 200
    assert resp_trans.json()["stage"] == "Production"

    # 2. Create Run 2 and register Candidate v2
    r2 = await client.post("/api/v1/mlops/runs", json={"experiment_name": "propensity"}, headers=engineer_headers)
    run2_id = r2.json()["run_id"]
    await client.post(
        "/api/v1/mlops/runs/metrics",
        json={"run_id": run2_id, "metrics": {"f1_score": 0.95, "accuracy": 0.96, "latency_p95_ms": 31.0}},
        headers=engineer_headers,
    )
    await client.post("/api/v1/mlops/runs/end", json={"run_id": run2_id, "status": "SUCCESS"}, headers=engineer_headers)

    resp_reg2 = await client.post(
        "/api/v1/mlops/models/register",
        json={"name": "propensity_model", "run_id": run2_id, "description": "Candidate v2 with XGBoost"},
        headers=engineer_headers,
    )
    assert resp_reg2.status_code == 201
    assert resp_reg2.json()["version"] == 2

    # 3. Evaluate Gate with Auto Promote
    resp_gate = await client.post(
        "/api/v1/mlops/evaluate-gate",
        json={"model_name": "propensity_model", "candidate_version": 2, "auto_promote": True},
        headers=engineer_headers,
    )
    assert resp_gate.status_code == 200
    gate_data = resp_gate.json()
    assert gate_data["passed"] is True
    assert gate_data["promoted"] is True

    # 4. Get Registered Model & Verify v2 is Production and v1 is Archived
    resp_m = await client.get("/api/v1/mlops/models/propensity_model", headers=engineer_headers)
    assert resp_m.status_code == 200
    m_data = resp_m.json()
    v1_info = [v for v in m_data["versions"] if v["version"] == 1][0]
    v2_info = [v for v in m_data["versions"] if v["version"] == 2][0]
    assert v1_info["stage"] == "Archived"
    assert v2_info["stage"] == "Production"


@pytest.mark.asyncio
async def test_mlops_pipeline_trigger(client: AsyncClient, engineer_headers: dict):
    resp = await client.post(
        "/api/v1/mlops/pipelines/run",
        json={"force": True},
        headers=engineer_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert len(data["executed_stages"]) >= 1
