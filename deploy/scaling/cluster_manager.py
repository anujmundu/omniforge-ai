"""Cluster Scaling Manager: monitors node metrics, queue saturation, and HPA autoscaling thresholds."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from deploy.scaling.base import ClusterHealthSummary, ClusterNodeRole
from deploy.scaling.task_queue import DistributedTaskQueue, task_queue
from deploy.scaling.worker_pool import AsyncWorkerPool, worker_pool


class ClusterScalingManager:
    """Orchestrates Kubernetes HPA evaluations and distributed cluster node topology."""

    def __init__(
        self,
        pool: Optional[AsyncWorkerPool] = None,
        queue: Optional[DistributedTaskQueue] = None,
        hpa_cpu_target_pct: float = 75.0,
        min_replicas: int = 2,
        max_replicas: int = 10,
    ):
        self.pool = pool or worker_pool
        self.queue = queue or task_queue
        self.hpa_cpu_target_pct = hpa_cpu_target_pct
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

    def get_cluster_health(self) -> ClusterHealthSummary:
        """Aggregate health, concurrency, utilization, and HPA recommendation."""
        workers = self.pool.get_workers()
        stats = self.queue.get_queue_stats()

        total_nodes = len(workers)
        active_workers = sum(1 for w in workers if w.is_active)
        total_slots = sum(w.concurrency_capacity for w in workers if w.is_active)

        if workers:
            avg_cpu = sum(w.cpu_utilization_pct for w in workers) / len(workers)
            avg_mem = sum(w.memory_utilization_pct for w in workers) / len(workers)
        else:
            avg_cpu = 0.0
            avg_mem = 0.0

        # Calculate recommended replicas according to Kubernetes HPA v2 standard formula:
        # desiredReplicas = ceil(currentReplicas * (currentMetricValue / desiredMetricValue))
        # Account for queue depth pressure as well
        if avg_cpu > 0:
            scale_factor = avg_cpu / self.hpa_cpu_target_pct
            recommended = math.ceil(max(1, total_nodes) * scale_factor)
        else:
            recommended = total_nodes

        # Add queue pressure scale up if pending jobs exceed current capacity
        if stats["queued"] > total_slots:
            recommended += math.ceil((stats["queued"] - total_slots) / 4)

        recommended = max(self.min_replicas, min(self.max_replicas, recommended))

        return ClusterHealthSummary(
            timestamp=datetime.now(timezone.utc),
            cluster_name="omniforge-production-cluster",
            total_nodes=total_nodes,
            active_workers=active_workers,
            total_concurrency_slots=total_slots,
            queue_depth_pending=stats["queued"],
            queue_depth_running=stats["running"],
            queue_depth_dlq=stats["dead_lettered"],
            avg_cpu_utilization_pct=round(avg_cpu, 2),
            avg_memory_utilization_pct=round(avg_mem, 2),
            hpa_recommended_replicas=recommended,
            workers=workers,
        )

    def autoscale_pool(self, target_replicas: Optional[int] = None) -> ClusterHealthSummary:
        """Adjust worker pool size to target count or HPA recommended count."""
        health = self.get_cluster_health()
        target = target_replicas if target_replicas is not None else health.hpa_recommended_replicas
        target = max(self.min_replicas, min(self.max_replicas, target))

        current_count = len(self.pool.get_workers())

        if target > current_count:
            # Scale up
            for _ in range(target - current_count):
                self.pool.register_worker(role=ClusterNodeRole.WORKER_CPU, concurrency=4)
        elif target < current_count:
            # Scale down
            workers = self.pool.get_workers()
            to_remove = workers[target:]
            for w in to_remove:
                self.pool.unregister_worker(w.worker_id)

        return self.get_cluster_health()


# Global Cluster Manager Instance
cluster_manager = ClusterScalingManager()
