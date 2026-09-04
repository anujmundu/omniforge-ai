import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_user_registration_and_login_lifecycle(client: AsyncClient):
    # 1. Register first user (automatically ADMIN)
    reg_payload = {
        "email": "lead_ml@aiforge.dev",
        "password": "SecurePassword123!",
        "full_name": "Lead ML Architect",
    }
    reg_res = await client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_res.status_code == 201
    reg_data = reg_res.json()
    assert reg_data["user"]["email"] == "lead_ml@aiforge.dev"
    assert reg_data["user"]["role"] == "ADMIN"
    assert "access_token" in reg_data["tokens"]
    assert "refresh_token" in reg_data["tokens"]

    # 2. Login with correct credentials
    login_payload = {
        "email": "lead_ml@aiforge.dev",
        "password": "SecurePassword123!",
    }
    login_res = await client.post("/api/v1/auth/login", json=login_payload)
    assert login_res.status_code == 200
    login_data = login_res.json()
    access_token = login_data["tokens"]["access_token"]
    refresh_token = login_data["tokens"]["refresh_token"]

    # 3. Test /me profile endpoint
    headers = {"Authorization": f"Bearer {access_token}"}
    me_res = await client.get("/api/v1/auth/me", headers=headers)
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["email"] == "lead_ml@aiforge.dev"

    # 4. Test token refresh
    refresh_res = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert refresh_res.status_code == 200
    refresh_data = refresh_res.json()
    assert "access_token" in refresh_data


@pytest.mark.asyncio
async def test_login_invalid_password(client: AsyncClient, admin_user):
    res = await client.post(
        "/api/v1/auth/login",
        json={"email": admin_user.email, "password": "WrongPassword999!"},
    )
    assert res.status_code == 401
    assert "Invalid email or password" in res.json()["detail"]


@pytest.mark.asyncio
async def test_unauthorized_access(client: AsyncClient):
    res = await client.get("/api/v1/auth/me")
    assert res.status_code == 401
