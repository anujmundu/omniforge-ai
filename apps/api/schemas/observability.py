"""OmniForge Observability API Schemas (Pydantic v2).

Defines request/response models for statistical data drift analysis,
Prometheus metrics queries, and SLA alerting lifecycle management.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FeatureDriftSchema(BaseModel):
    """Schema representing statistical drift metrics for a single feature."""

    feature_name: str
    feature_type: str = "numerical"
    method: str = "ks_test"
    test_statistic: float
    p_value: Optional[float] = None
    threshold: float = 0.05
    drift_detected: bool = False
    reference_mean: Optional[float] = None
    current_mean: Optional[float] = None
    reference_missing_rate: float = 0.0
    current_missing_rate: float = 0.0
    description: str = ""


class DriftCalculationRequest(BaseModel):
    """Request payload to calculate statistical distribution drift."""

    dataset_name: str = "production_inference"
    reference_data: List[Dict[str, Any]]
    current_data: List[Dict[str, Any]]
    drift_share_threshold: float = Field(default=0.33, ge=0.0, le=1.0)
    significance_level: float = Field(default=0.05, gt=0.0, lt=1.0)
    psi_threshold: float = Field(default=0.20, gt=0.0)


class DriftCalculationResponse(BaseModel):
    """Response containing complete dataset-level drift report."""

    report_id: str
    dataset_name: str
    reference_rows: int
    current_rows: int
    drift_detected: bool
    share_of_drifted_features: float
    number_of_features: int
    drifted_features_count: int
    drift_threshold: float
    feature_results: Dict[str, FeatureDriftSchema]
    timestamp: datetime


class AlertRuleCreateRequest(BaseModel):
    """Request payload to register a new SLA threshold rule."""

    name: str
    description: str = ""
    metric_name: str
    condition: str = Field(default="gt", pattern="^(gt|lt|gte|lte|eq)$")
    threshold: float
    severity: str = Field(default="warning", pattern="^(info|warning|error|critical)$")
    labels: Dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class AlertRuleResponse(BaseModel):
    """Schema representing an active SLA threshold rule."""

    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str
    threshold: float
    severity: str
    labels: Dict[str, str]
    enabled: bool


class ActiveAlertResponse(BaseModel):
    """Schema representing an operational alert instance."""

    alert_id: str
    rule_id: str
    rule_name: str
    severity: str
    state: str
    current_value: float
    threshold: float
    message: str
    labels: Dict[str, str]
    fired_at: datetime
    resolved_at: Optional[datetime] = None


class EvaluateMetricRequest(BaseModel):
    """Request payload to evaluate a single metric observation against registered SLA rules."""

    metric_name: str
    value: float
    labels: Optional[Dict[str, str]] = None


class EvaluateMetricResponse(BaseModel):
    """Response from metric SLA evaluation."""

    evaluated_rules_count: int
    triggered_alerts: List[ActiveAlertResponse]


class ResolveAlertResponse(BaseModel):
    """Response confirming manual alert resolution."""

    success: bool
    alert_id: str
    resolved_at: datetime
