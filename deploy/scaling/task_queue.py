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
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.result = result_data
            job.execution_time_ms = execution_time_ms
            return job

    def fail_job(self, job_id: str, error_message: str) -> Optional[TaskJob]:
        """Handle job failure with retry backoff or relegation to Dead-Letter Queue."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

            job.retry_count += 1
            job.error_message = error_message

            if job.retry_count <= job.max_retries:
                job.status = JobStatus.RETRYING
                # Re-queue with incremented retry count and same priority
                job.status = JobStatus.QUEUED
                heapq.heappush(self._heap, (job.priority.value, datetime.now(timezone.utc).timestamp(), job.job_id))
            else:
                job.status = JobStatus.DEAD_LETTERED
                job.completed_at = datetime.now(timezone.utc)
                if job.job_id not in self._dead_letter_queue:
                    self._dead_letter_queue.append(job.job_id)

            return job

    def get_job(self, job_id: str) -> Optional[TaskJob]:
        """Look up job metadata and status."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 50) -> List[TaskJob]:
        """List jobs filtered by status."""
        with self._lock:
            jobs = list(self._jobs.values())
            if status:
                jobs = [j for j in jobs if j.status == status]
            return jobs[-limit:]

    def get_queue_stats(self) -> dict:
        """Return counts of jobs in various lifecycle stages."""
        with self._lock:
            pending = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
            running = sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)
            completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
            failed = sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)
            dlq = len(self._dead_letter_queue)
            return {
                "total": len(self._jobs),
                "queued": pending,
                "running": running,
                "completed": completed,
                "failed": failed,
                "dead_lettered": dlq,
            }

    def clear(self):
        """Reset queue state (useful for tests)."""
        with self._lock:
            self._heap.clear()
            self._jobs.clear()
            self._dead_letter_queue.clear()


# Global Queue Instance
task_queue = DistributedTaskQueue()
