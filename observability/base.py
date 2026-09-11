"""OmniForge Observability Core Data Models, Enums, and Schemas.

Defines schemas and state representations for time-series metrics, statistical data drift
reports, alert rules, and real-time SLA incident monitoring.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """Supported Prometheus metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Operational alert severity tiers."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(str, Enum):
    """Lifecycle state of an alert."""

    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


class DriftMethod(str, Enum):
    """Statistical algorithms for distribution drift detection."""

    KS_TEST = "ks_test"
    PSI = "psi"
    CHI_SQUARE = "chi_square"
    WASSERSTEIN = "wasserstein"


class FeatureDriftResult(BaseModel):
