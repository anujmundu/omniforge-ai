"""Distributed asynchronous priority task queue with retries and dead-letter queue (DLQ).

Supports priority ordering, atomic popping, retry backoff, and DLQ management.
"""

from __future__ import annotations

import heapq
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from deploy.scaling.base import JobPriority, JobStatus, TaskJob, TaskType


class DistributedTaskQueue:
    """Thread-safe in-memory priority queue with Redis-compatible abstractions."""

    def __init__(self):
        # Priority heap: list of (priority_int, created_timestamp, job_id)
        self._heap: List[tuple[int, float, str]] = []
        self._jobs: Dict[str, TaskJob] = {}
        self._dead_letter_queue: List[str] = []
        self._lock = threading.Lock()

    def enqueue(
        self,
        task_type: TaskType,
        payload: dict,
        priority: JobPriority = JobPriority.DEFAULT,
        max_retries: int = 3,
    ) -> TaskJob:
        """Enqueue a new job into the priority queue."""
        with self._lock:
            job = TaskJob(
                task_type=task_type,
                priority=priority,
                payload=payload,
                status=JobStatus.QUEUED,
                max_retries=max_retries,
            )
            self._jobs[job.job_id] = job
            heapq.heappush(self._heap, (priority.value, job.created_at.timestamp(), job.job_id))
            return job

    def dequeue(self, worker_id: str) -> Optional[TaskJob]:
        """Atomically pop the highest priority job from the queue and assign to worker."""
        with self._lock:
            while self._heap:
                _, _, job_id = heapq.heappop(self._heap)
                job = self._jobs.get(job_id)
                if job and job.status == JobStatus.QUEUED:
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.now(timezone.utc)
                    job.assigned_worker_id = worker_id
                    return job
            return None

    def complete_job(self, job_id: str, result_data: dict, execution_time_ms: float) -> Optional[TaskJob]:
        """Mark job as successfully completed."""