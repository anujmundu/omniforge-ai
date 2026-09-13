"""Integration tests for Cloud Scaling and Distributed Task Mesh API endpoints."""

from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def test_submit_async_job_endpoint():
    response = client.post(
        "/api/v1/scaling/jobs/submit",
        json={
            "task_type": "ml_training",
            "payload": {"model_name": "random_forest_regressor", "epochs": 20},
            "priority": 1,
            "max_retries": 3,
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["task_type"] == "ml_training"
    assert data["status"] == "queued"


def test_get_job_detail_endpoint():
    # Submit job first
    submit_res = client.post(
        "/api/v1/scaling/jobs/submit",
        json={
            "task_type": "nlp_embedding_batch",
            "payload": {"texts": ["sentence A", "sentence B"]},
            "priority": 2,
        },
    )
    job_id = submit_res.json()["job_id"]

    # Query job
    get_res = client.get(f"/api/v1/scaling/jobs/{job_id}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["job"]["job_id"] == job_id
    assert data["job"]["task_type"] == "nlp_embedding_batch"


def test_process_next_job_endpoint():
    # Submit job
    submit_res = client.post(
        "/api/v1/scaling/jobs/submit",
        json={
            "task_type": "rag_document_indexing",
            "payload": {"collection_name": "k8s_docs", "documents": ["doc1", "doc2"]},
            "priority": 0,
        },
    )
    job_id = submit_res.json()["job_id"]

    # Process next
    proc_res = client.post("/api/v1/scaling/workers/process-next?worker_id=worker-cpu-01")
    assert proc_res.status_code == 200
    data = proc_res.json()
    assert data["status"] == "processed"
    assert data["job_id"] == job_id
    assert data["job_status"] == "completed"


def test_cluster_status_endpoint():
    response = client.get("/api/v1/scaling/cluster/status")
    assert response.status_code == 200
    data = response.json()
    assert data["cluster"]["cluster_name"] == "omniforge-production-cluster"
    assert data["cluster"]["total_nodes"] >= 2
    assert data["cluster"]["active_workers"] >= 2
    assert "hpa_recommended_replicas" in data["cluster"]


def test_missing_job_404_handling():
    response = client.get("/api/v1/scaling/jobs/non_existent_job_12345")
    assert response.status_code == 404
