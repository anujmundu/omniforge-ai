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
