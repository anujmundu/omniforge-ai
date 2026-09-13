"""API router exposing distributed job submission, worker management, and cluster scaling telemetry."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from apps.api.core.dependencies import require_roles
from apps.api.models.user import UserRole
from apps.api.schemas.scaling import (
    AutoscaleRequest,
    ClusterStatusResponse,
    JobDetailResponse,
    JobListResponse,
    JobSubmitRequest,
    JobSubmitResponse,
)
from deploy.scaling.base import JobStatus
from deploy.scaling.cluster_manager import cluster_manager
from deploy.scaling.task_queue import task_queue
from deploy.scaling.worker_pool import worker_pool

router = APIRouter(prefix="/scaling", tags=["Cloud Scaling & Distributed Task Mesh"])


@router.post(
    "/jobs/submit",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an asynchronous background job",
)
async def submit_job(request: JobSubmitRequest) -> JobSubmitResponse:
    """Enqueue an asynchronous AI/ML job into the distributed priority task mesh."""
    job = task_queue.enqueue(
        task_type=request.task_type,
        payload=request.payload,
        priority=request.priority,
        max_retries=request.max_retries,
    )
    return JobSubmitResponse(
        job_id=job.job_id,
        task_type=job.task_type,
        priority=job.priority,
        status=job.status,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobDetailResponse,
    summary="Get job execution status and results",
)
async def get_job(job_id: str) -> JobDetailResponse:
    """Retrieve execution lifecycle status, results, or error diagnostics for a job."""
    job = task_queue.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found in task queue.",
        )
    return JobDetailResponse(job=job)


@router.get(
    "/jobs",
    response_model=JobListResponse,
    summary="List distributed jobs",
)
async def list_jobs(
    job_status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=100),
) -> JobListResponse:
    """List recent jobs dispatched through the mesh."""
    jobs = task_queue.list_jobs(status=job_status, limit=limit)
    return JobListResponse(total_count=len(jobs), jobs=jobs)


@router.post(
    "/workers/process-next",
    summary="Trigger worker to process the next pending job",
)
async def process_next_job(
    worker_id: str = Query("worker-cpu-01", description="Worker identifier"),
) -> dict:
    """Worker trigger endpoint to dequeue and execute the next prioritized workload."""
    job = worker_pool.process_next_job(worker_id)
    if not job:
        return {"status": "idle", "message": "No pending jobs in queue"}
    return {
        "status": "processed",
        "job_id": job.job_id,
        "job_status": job.status.value,
        "execution_time_ms": job.execution_time_ms,
    }


@router.get(
    "/cluster/status",
    response_model=ClusterStatusResponse,
    summary="Get real-time Kubernetes cluster health and telemetry",
)
async def get_cluster_status() -> ClusterStatusResponse:
    """Inspect active worker nodes, queue depth, resource utilization, and HPA recommendation."""
    health = cluster_manager.get_cluster_health()
    return ClusterStatusResponse(cluster=health)


@router.post(
    "/cluster/autoscale",
    response_model=ClusterStatusResponse,
    summary="Trigger HPA worker pool autoscaling (Admin only)",
)
async def autoscale_cluster(
    request: Optional[AutoscaleRequest] = None,
    current_admin=Depends(require_roles(UserRole.ADMIN)),
) -> ClusterStatusResponse:
    """Scale worker pool up/down based on explicit target or automated HPA thresholds."""
    target = request.target_replicas if request else None
    health = cluster_manager.autoscale_pool(target_replicas=target)
    return ClusterStatusResponse(cluster=health)
