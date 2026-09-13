"""Core domain models, enums, and data contracts for OmniForge Cloud Scaling and Distributed Task Mesh."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle status of an asynchronous distributed task."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTERED = "dead_lettered"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Priority weights for task queue scheduling (lower integer = higher priority)."""

    CRITICAL = 0
    HIGH = 1
    DEFAULT = 2
    LOW = 3
    BATCH = 4


class TaskType(str, Enum):
    """Supported distributed asynchronous workload categories."""

    ML_TRAINING = "ml_training"
    ML_INFERENCE_BATCH = "ml_inference_batch"
    VISION_TRACKING_BATCH = "vision_tracking_batch"
    NLP_EMBEDDING_BATCH = "nlp_embedding_batch"
    RAG_DOCUMENT_INDEXING = "rag_document_indexing"
    RED_TEAM_AUDIT_BATTERY = "red_team_audit_battery"
    DATA_DRIFT_COMPUTATION = "data_drift_computation"


class ClusterNodeRole(str, Enum):
    """Role classification of a cluster node/pod."""

    API_GATEWAY = "api_gateway"
    WORKER_CPU = "worker_cpu"
    WORKER_GPU = "worker_gpu"
    STORAGE_CACHE = "storage_cache"


class TaskJob(BaseModel):
    """Representation of an asynchronous job dispatched to the task mesh."""

    job_id: str = Field(default_factory=lambda: f"job_{uuid4().hex[:12]}")
    task_type: TaskType = Field(..., description="Category of workload")
    priority: JobPriority = Field(JobPriority.DEFAULT, description="Execution priority")
    status: JobStatus = Field(JobStatus.PENDING, description="Current execution lifecycle state")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Task execution parameters")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = Field(0, ge=0, description="Number of attempted retries")
    max_retries: int = Field(3, ge=0, description="Max allowed retries before moving to DLQ")
    assigned_worker_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


class TaskExecutionResult(BaseModel):
    """Outcome of a worker processing a TaskJob."""

    job_id: str = Field(...)
    success: bool = Field(...)
    result_data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = Field(..., ge=0.0)


class WorkerNodeInfo(BaseModel):
    """Telemetry information for a worker node in the mesh."""

    worker_id: str = Field(default_factory=lambda: f"worker_{uuid4().hex[:8]}")
    role: ClusterNodeRole = Field(ClusterNodeRole.WORKER_CPU)
    is_active: bool = Field(True)
    concurrency_capacity: int = Field(4, ge=1)
    active_jobs_count: int = Field(0, ge=0)
    total_completed_jobs: int = Field(0, ge=0)
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cpu_utilization_pct: float = Field(0.0, ge=0.0, le=100.0)
    memory_utilization_pct: float = Field(0.0, ge=0.0, le=100.0)


class ClusterHealthSummary(BaseModel):
    """Real-time cluster topology and scaling telemetry."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cluster_name: str = Field("omniforge-production-cluster")
    total_nodes: int = Field(..., ge=0)
    active_workers: int = Field(..., ge=0)
    total_concurrency_slots: int = Field(..., ge=0)
    queue_depth_pending: int = Field(..., ge=0)
    queue_depth_running: int = Field(..., ge=0)
    queue_depth_dlq: int = Field(..., ge=0)
    avg_cpu_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    avg_memory_utilization_pct: float = Field(..., ge=0.0, le=100.0)
    hpa_recommended_replicas: int = Field(..., ge=1)
    workers: List[WorkerNodeInfo] = Field(default_factory=list)
