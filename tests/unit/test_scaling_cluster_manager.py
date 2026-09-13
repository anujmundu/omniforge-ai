"""Unit tests for ClusterScalingManager and HPA evaluations."""

import pytest

from deploy.scaling.base import TaskType
from deploy.scaling.cluster_manager import ClusterScalingManager
from deploy.scaling.task_queue import DistributedTaskQueue
from deploy.scaling.worker_pool import AsyncWorkerPool


@pytest.fixture
def cluster_harness():
    q = DistributedTaskQueue()
    q.clear()
    pool = AsyncWorkerPool(queue=q)
    mgr = ClusterScalingManager(pool=pool, queue=q, hpa_cpu_target_pct=75.0, min_replicas=2, max_replicas=8)
    return mgr, pool, q


def test_cluster_health_baseline(cluster_harness):
    mgr, _, _ = cluster_harness
    health = mgr.get_cluster_health()

    assert health.cluster_name == "omniforge-production-cluster"
    assert health.total_nodes == 2
    assert health.active_workers == 2
    assert health.total_concurrency_slots == 8
    assert health.hpa_recommended_replicas == 2


def test_cluster_hpa_recommends_scale_up_on_high_cpu(cluster_harness):
    mgr, pool, _ = cluster_harness
    # Set high CPU load on workers (e.g. 90% each)
    for w in pool.get_workers():
        pool.update_heartbeat(w.worker_id, cpu_pct=95.0, mem_pct=85.0)

    health = mgr.get_cluster_health()
    assert health.avg_cpu_utilization_pct == 95.0
    # 95 / 75 * 2 ~ ceil(2.53) -> 3 replicas
    assert health.hpa_recommended_replicas >= 3


def test_cluster_autoscale_execution(cluster_harness):
    mgr, pool, _ = cluster_harness
    assert len(pool.get_workers()) == 2

    # Scale to 5 replicas
    scaled = mgr.autoscale_pool(target_replicas=5)
    assert scaled.total_nodes == 5
    assert len(pool.get_workers()) == 5

    # Scale down to 2 replicas
    scaled_down = mgr.autoscale_pool(target_replicas=2)
    assert scaled_down.total_nodes == 2
    assert len(pool.get_workers()) == 2


def test_queue_backlog_triggers_scale_recommendation(cluster_harness):
    mgr, _, q = cluster_harness
    # Current capacity is 8 slots. Enqueue 20 jobs
    for i in range(20):
        q.enqueue(TaskType.ML_TRAINING, {"id": i})

    health = mgr.get_cluster_health()
    assert health.queue_depth_pending == 20
    assert health.hpa_recommended_replicas > 2
