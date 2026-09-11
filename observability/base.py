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
    """Statistical drift quantification for an individual dataset feature."""

    feature_name: str
    feature_type: str = "numerical"  # numerical | categorical | text
    method: DriftMethod = DriftMethod.KS_TEST
    test_statistic: float
    p_value: Optional[float] = None
    threshold: float = 0.05
    drift_detected: bool = False
    reference_mean: Optional[float] = None
    current_mean: Optional[float] = None
    reference_missing_rate: float = 0.0
    current_missing_rate: float = 0.0
    description: str = ""


class DatasetDriftReport(BaseModel):
    """Comprehensive dataset-level statistical distribution and schema drift report."""

    report_id: str = Field(default_factory=lambda: f"drift_{uuid4().hex[:10]}")
    dataset_name: str = "default_dataset"
    reference_rows: int
    current_rows: int
    drift_detected: bool = False
    share_of_drifted_features: float = 0.0
    number_of_features: int = 0
    drifted_features_count: int = 0
    drift_threshold: float = 0.33  # If >33% features drift, dataset is drifted
    feature_results: Dict[str, FeatureDriftResult] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AlertRule(BaseModel):
