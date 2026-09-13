"""Unit tests for AsyncWorkerPool node registration and workload execution."""

import pytest

from deploy.scaling.base import ClusterNodeRole, JobStatus, TaskType
from deploy.scaling.task_queue import DistributedTaskQueue
from deploy.scaling.worker_pool import AsyncWorkerPool


@pytest.fixture
def worker_harness():
    q = DistributedTaskQueue()
    q.clear()
    pool = AsyncWorkerPool(queue=q)
    return pool, q


def test_worker_registration_and_unregistration(worker_harness):
    pool, _ = worker_harness
    initial_count = len(pool.get_workers())

    node = pool.register_worker(role=ClusterNodeRole.WORKER_GPU, concurrency=8)
    assert node.role == ClusterNodeRole.WORKER_GPU
    assert node.concurrency_capacity == 8
    assert len(pool.get_workers()) == initial_count + 1

    removed = pool.unregister_worker(node.worker_id)
    assert removed is True
    assert len(pool.get_workers()) == initial_count


def test_worker_executes_ml_training_workload(worker_harness):
    pool, queue = worker_harness
    job = queue.enqueue(TaskType.ML_TRAINING, {"model_name": "xgboost_churn", "epochs": 50})

    processed = pool.process_next_job("worker-cpu-01")
    assert processed is not None
    assert processed.job_id == job.job_id
    assert processed.status == JobStatus.COMPLETED
    assert processed.result["accuracy"] == 0.945


def test_worker_executes_rag_indexing_workload(worker_harness):
    pool, queue = worker_harness
    job = queue.enqueue(
        TaskType.RAG_DOCUMENT_INDEXING,
        {"collection_name": "tech_docs", "documents": ["doc1", "doc2"]},
    )

    processed = pool.process_next_job("worker-cpu-01")
    assert processed is not None
    assert processed.status == JobStatus.COMPLETED
    assert processed.result["indexed_chunks"] == 8


def test_worker_heartbeat_update(worker_harness):
    pool, _ = worker_harness
    pool.update_heartbeat("worker-cpu-01", cpu_pct=65.5, mem_pct=72.0)
    workers = {w.worker_id: w for w in pool.get_workers()}

    assert workers["worker-cpu-01"].cpu_utilization_pct == 65.5
    assert workers["worker-cpu-01"].memory_utilization_pct == 72.0
