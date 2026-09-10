"""OmniForge MLOps Core Data Models and Domain Enums."""
from __future__ import annotations
from enum import Enum

class ModelStage(str, Enum):
    """Operational lifecycle stages for registered models."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

class PipelineStatus(str, Enum):
    """Execution status for MLOps and DVC pipeline stages."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
