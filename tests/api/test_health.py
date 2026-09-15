import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_endpoint(client: AsyncClient):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "version" in data
    assert "environment" in data
    assert "database" in data["services"]
    assert data["services"]["database"]["status"] == "healthy"
    assert "telemetry" in data["services"]
    assert data["services"]["telemetry"]["status"] == "healthy"
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["docs"] == "/docs"
