"""Integration tests for Observability REST endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_metrics_endpoint(client: AsyncClient):
    # Root /metrics scrape endpoint
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "omniforge_http_requests_total" in resp.text

    # Router /api/v1/observability/metrics endpoint
    resp2 = await client.get("/api/v1/observability/metrics")
    assert resp2.status_code == 200
    assert "text/plain" in resp2.headers["content-type"]


@pytest.mark.asyncio
async def test_drift_calculate_endpoint(client: AsyncClient, admin_headers: dict):
    payload = {
        "dataset_name": "api_drift_test",
        "reference_data": [{"age": 25, "income": 50000}, {"age": 30, "income": 60000}],
        "current_data": [{"age": 55, "income": 120000}, {"age": 60, "income": 130000}],
        "drift_share_threshold": 0.5,
    }

    resp = await client.post("/api/v1/observability/drift/calculate", json=payload, headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset_name"] == "api_drift_test"
    assert data["reference_rows"] == 2
    assert "age" in data["feature_results"]
    assert "income" in data["feature_results"]


@pytest.mark.asyncio
async def test_alert_rules_and_evaluation_endpoints(client: AsyncClient, admin_headers: dict):
    # List default rules
    resp = await client.get("/api/v1/observability/alerts/rules", headers=admin_headers)
    assert resp.status_code == 200
    rules = resp.json()
    assert len(rules) >= 6

    # Evaluate metric
    eval_payload = {"metric_name": "omniforge_http_request_duration_seconds", "value": 1.25}
    eval_resp = await client.post("/api/v1/observability/alerts/evaluate", json=eval_payload, headers=admin_headers)
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert len(eval_data["triggered_alerts"]) >= 1

    # List active alerts
    alerts_resp = await client.get("/api/v1/observability/alerts", headers=admin_headers)
    assert alerts_resp.status_code == 200
    active_alerts = alerts_resp.json()
    assert len(active_alerts) >= 1

    # Resolve alert
    alert_id = active_alerts[0]["alert_id"]
    resolve_resp = await client.post(f"/api/v1/observability/alerts/{alert_id}/resolve", headers=admin_headers)
    assert resolve_resp.status_code == 200
    assert resolve_resp.json()["success"] is True
