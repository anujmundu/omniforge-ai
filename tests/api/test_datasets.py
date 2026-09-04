import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_dataset_registration_and_query_lifecycle(client: AsyncClient, engineer_headers):
    # 1. Create a project first
    proj_res = await client.post(
        "/api/v1/projects",
        json={"name": "Dataset Test Project", "slug": "dataset-test-proj"},
        headers=engineer_headers,
    )
    assert proj_res.status_code == 201
    project_id = proj_res.json()["id"]

    # 2. Register dataset
    dataset_payload = {
        "project_id": project_id,
        "name": "telecom_customer_churn",
        "version": "1.0.0",
        "file_format": "CSV",
        "storage_path": "/storage/datasets/telecom_churn.csv",
        "row_count": 7043,
        "checksum_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "schema_metadata": {
            "features": ["tenure", "MonthlyCharges", "TotalCharges"],
            "target": "Churn",
        },
        "description": "Baseline telecom churn dataset with 7043 records",
    }
    ds_res = await client.post("/api/v1/datasets", json=dataset_payload, headers=engineer_headers)
    assert ds_res.status_code == 201
    ds_data = ds_res.json()
    dataset_id = ds_data["id"]
    assert ds_data["name"] == "telecom_customer_churn"
    assert ds_data["row_count"] == 7043

    # 3. Prevent duplicate dataset version in same project
    dup_res = await client.post("/api/v1/datasets", json=dataset_payload, headers=engineer_headers)
    assert dup_res.status_code == 400
    assert "already exists" in dup_res.json()["detail"]

    # 4. List datasets by project
    list_res = await client.get(f"/api/v1/datasets?project_id={project_id}", headers=engineer_headers)
    assert list_res.status_code == 200
    datasets = list_res.json()
    assert len(datasets) == 1
    assert datasets[0]["id"] == dataset_id

    # 5. Get dataset details
    get_res = await client.get(f"/api/v1/datasets/{dataset_id}", headers=engineer_headers)
    assert get_res.status_code == 200
    assert get_res.json()["schema_metadata"]["target"] == "Churn"
