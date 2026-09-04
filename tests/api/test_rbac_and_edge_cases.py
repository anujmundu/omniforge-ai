import pytest
from httpx import AsyncClient
from apps.api.core.security import create_access_token, get_password_hash
from apps.api.models.user import User, UserRole


@pytest.fixture
async def viewer_user(db_session):
    user = User(
        email="viewer@omniforge.dev",
        hashed_password=get_password_hash("ViewerPass123!"),
        full_name="Platform Viewer",
        role=UserRole.VIEWER,
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def inactive_user(db_session):
    user = User(
        email="inactive@omniforge.dev",
        hashed_password=get_password_hash("InactivePass123!"),
        full_name="Inactive User",
        role=UserRole.DATA_SCIENTIST,
        is_active=False,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def viewer_headers(viewer_user):
    token = create_access_token(subject=viewer_user.id, role=viewer_user.role.value)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def inactive_headers(inactive_user):
    token = create_access_token(subject=inactive_user.id, role=inactive_user.role.value)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_rbac_viewer_cannot_create_project(client: AsyncClient, viewer_headers):
    # Viewer role should be denied 403 when attempting to create a project
    res = await client.post(
        "/api/v1/projects",
        json={"name": "Unauthorized Project", "slug": "unauthorized-proj"},
        headers=viewer_headers,
    )
    assert res.status_code == 403
    assert "Operation not permitted" in res.json()["detail"]


@pytest.mark.asyncio
async def test_inactive_user_cannot_access_api(client: AsyncClient, inactive_headers):
    res = await client.get("/api/v1/auth/me", headers=inactive_headers)
    assert res.status_code == 403
    assert "Inactive user" in res.json()["detail"]


@pytest.mark.asyncio
async def test_oauth2_swagger_token_endpoint(client: AsyncClient, admin_user):
    res = await client.post(
        "/api/v1/auth/token",
        data={"username": admin_user.email, "password": "AdminPass123!"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_missing_resource_404_handling(client: AsyncClient, engineer_headers):
    # Missing Project
    res = await client.get("/api/v1/projects/non-existent-uuid", headers=engineer_headers)
    assert res.status_code == 404

    # Missing Dataset
    res = await client.get("/api/v1/datasets/non-existent-uuid", headers=engineer_headers)
    assert res.status_code == 404

    # Missing Experiment
    res = await client.get("/api/v1/experiments/non-existent-uuid", headers=engineer_headers)
    assert res.status_code == 404
