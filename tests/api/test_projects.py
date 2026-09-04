import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_project_crud_lifecycle(client: AsyncClient, engineer_headers, admin_headers):
    # 1. Create project as ML Engineer
    project_payload = {
        "name": "Customer Churn Prediction",
        "slug": "customer-churn-v1",
        "description": "Production churn prediction pipeline",
    }
    create_res = await client.post("/api/v1/projects", json=project_payload, headers=engineer_headers)
    assert create_res.status_code == 201
    created_data = create_res.json()
    project_id = created_data["id"]
    assert created_data["name"] == "Customer Churn Prediction"
    assert created_data["slug"] == "customer-churn-v1"

    # 2. List projects
    list_res = await client.get("/api/v1/projects", headers=engineer_headers)
    assert list_res.status_code == 200
    projects = list_res.json()
    assert len(projects) >= 1
    assert any(p["id"] == project_id for p in projects)

    # 3. Get single project by ID and by slug
    get_res = await client.get(f"/api/v1/projects/{project_id}", headers=engineer_headers)
    assert get_res.status_code == 200
    assert get_res.json()["id"] == project_id

    get_slug_res = await client.get("/api/v1/projects/customer-churn-v1", headers=engineer_headers)
    assert get_slug_res.status_code == 200
    assert get_slug_res.json()["id"] == project_id

    # 4. Update project metadata
    update_res = await client.patch(
        f"/api/v1/projects/{project_id}",
        json={"name": "Updated Churn Pipeline", "description": "Updated description"},
        headers=engineer_headers,
    )
    assert update_res.status_code == 200
    assert update_res.json()["name"] == "Updated Churn Pipeline"

    # 5. Delete project
    del_res = await client.delete(f"/api/v1/projects/{project_id}", headers=engineer_headers)
    assert del_res.status_code == 204

    # 6. Verify 404 after deletion
    get_after_del = await client.get(f"/api/v1/projects/{project_id}", headers=engineer_headers)
    assert get_after_del.status_code == 404


@pytest.mark.asyncio
async def test_duplicate_project_slug_rejected(client: AsyncClient, engineer_headers):
    payload = {"name": "Vision Tracking", "slug": "vision-tracking-pipe"}
    res1 = await client.post("/api/v1/projects", json=payload, headers=engineer_headers)
    assert res1.status_code == 201

    res2 = await client.post("/api/v1/projects", json=payload, headers=engineer_headers)
    assert res2.status_code == 400
    assert "already exists" in res2.json()["detail"]
