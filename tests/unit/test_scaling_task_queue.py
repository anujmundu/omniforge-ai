"""Unit tests for DistributedTaskQueue with priority scheduling and DLQ."""

import pytest

from deploy.scaling.base import JobPriority, JobStatus, TaskType
from deploy.scaling.task_queue import DistributedTaskQueue


@pytest.fixture
def queue():
    q = DistributedTaskQueue()
    q.clear()
    return q


def test_enqueue_and_dequeue_fifo_order(queue):
    j1 = queue.enqueue(TaskType.ML_TRAINING, {"model": "A"}, priority=JobPriority.DEFAULT)
    j2 = queue.enqueue(TaskType.ML_TRAINING, {"model": "B"}, priority=JobPriority.DEFAULT)

    popped1 = queue.dequeue("worker-01")
    popped2 = queue.dequeue("worker-02")

    assert popped1.job_id == j1.job_id
    assert popped2.job_id == j2.job_id
    assert popped1.status == JobStatus.RUNNING
    assert popped1.assigned_worker_id == "worker-01"


def test_priority_scheduling_preempts_lower_priority(queue):
    # Enqueue low priority first, then critical priority
    j_low = queue.enqueue(TaskType.DATA_DRIFT_COMPUTATION, {}, priority=JobPriority.BATCH)
    j_crit = queue.enqueue(TaskType.RED_TEAM_AUDIT_BATTERY, {}, priority=JobPriority.CRITICAL)

    first_popped = queue.dequeue("worker-01")
    assert first_popped.job_id == j_crit.job_id
    assert first_popped.priority == JobPriority.CRITICAL


def test_job_completion_updates_status(queue):
    job = queue.enqueue(TaskType.NLP_EMBEDDING_BATCH, {"texts": ["hello"]})
    popped = queue.dequeue("worker-01")

    completed = queue.complete_job(popped.job_id, {"embedded_count": 1}, execution_time_ms=12.5)
    assert completed.status == JobStatus.COMPLETED
    assert completed.result == {"embedded_count": 1}
    assert completed.execution_time_ms == 12.5


def test_job_retry_and_dead_letter_queue(queue):
    job = queue.enqueue(TaskType.ML_TRAINING, {}, max_retries=2)
    popped = queue.dequeue("worker-01")

    # Retry 1
    r1 = queue.fail_job(popped.job_id, "OOM Error 1")
    assert r1.status == JobStatus.QUEUED
    assert r1.retry_count == 1

    # Retry 2
    popped2 = queue.dequeue("worker-02")
    r2 = queue.fail_job(popped2.job_id, "OOM Error 2")
    assert r2.status == JobStatus.QUEUED
    assert r2.retry_count == 2

    # Retry 3 (exceeds max_retries=2) -> Dead Letter Queue
    popped3 = queue.dequeue("worker-03")
    r3 = queue.fail_job(popped3.job_id, "OOM Error 3 (Fatal)")
    assert r3.status == JobStatus.DEAD_LETTERED

    stats = queue.get_queue_stats()
    assert stats["dead_lettered"] == 1
