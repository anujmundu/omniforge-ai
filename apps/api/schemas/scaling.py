"""Pydantic v2 schemas for distributed task scaling, job queues, and cluster management."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from deploy.scaling.base import (
    ClusterHealthSummary,
    ClusterNodeRole,
    JobPriority,
    JobStatus,
    TaskJob,
    TaskType,
)


class JobSubmitRequest(BaseModel):
    """Request schema for dispatching an async background job."""

    task_type: TaskType = Field(..., description="Category of background AI/ML task")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Workload parameters")
    priority: JobPriority = Field(JobPriority.DEFAULT, description="Execution priority (0=CRITICAL, 4=BATCH)")
    max_retries: int = Field(3, ge=0, le=10, description="Max retries upon failure")


class JobSubmitResponse(BaseModel):
    """Response returned upon successfully queueing a job."""

    job_id: str = Field(...)
    task_type: TaskType = Field(...)
    priority: JobPriority = Field(...)
    status: JobStatus = Field(...)
    message: str = Field("Job queued successfully into distributed task mesh")


class JobDetailResponse(BaseModel):
    """Response containing full job execution details and results."""

    job: TaskJob


class JobListResponse(BaseModel):
    """Response listing jobs filtered by status."""

    total_count: int = Field(..., ge=0)
    jobs: List[TaskJob] = Field(default_factory=list)


class WorkerRegisterRequest(BaseModel):
    """Request schema for registering a worker node."""

    worker_id: Optional[str] = Field(None)
    role: ClusterNodeRole = Field(ClusterNodeRole.WORKER_CPU)
    concurrency: int = Field(4, ge=1, le=32)


class AutoscaleRequest(BaseModel):
    """Request schema for triggering an autoscale event."""

    target_replicas: Optional[int] = Field(None, ge=1, le=50, description="Explicit target replica count")


class ClusterStatusResponse(BaseModel):
    """Response schema containing cluster health and scaling summary."""

    cluster: ClusterHealthSummary
