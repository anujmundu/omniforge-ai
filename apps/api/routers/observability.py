"""OmniForge Observability and Telemetry API Router.

Exposes REST endpoints for Prometheus metrics exposition, statistical data drift
calculation, SLA rule management, and active incident alerting.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response, status

from apps.api.core.dependencies import get_current_user, require_roles
from apps.api.models.user import User, UserRole
from apps.api.schemas.observability import (
    ActiveAlertResponse,
    AlertRuleCreateRequest,
    AlertRuleResponse,
    DriftCalculationRequest,
    DriftCalculationResponse,
    EvaluateMetricRequest,
    EvaluateMetricResponse,
    FeatureDriftSchema,
    ResolveAlertResponse,
)
from observability.alerts import alert_manager
from observability.base import AlertRule, AlertSeverity
from observability.drift import drift_engine
from observability.metrics import DATA_DRIFT_SCORE_GAUGE, metrics_registry

router = APIRouter(prefix="/observability", tags=["Observability & Telemetry"])


@router.get("/metrics", response_class=Response, summary="Prometheus metrics scrape endpoint")
async def get_prometheus_metrics() -> Response:
    """Expose real-time platform telemetry metrics formatted in Prometheus exposition format."""
    metrics_text = metrics_registry.generate_prometheus_text()
    return Response(content=metrics_text, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.post(
    "/drift/calculate",
    response_model=DriftCalculationResponse,
    summary="Compute statistical data drift between reference and current dataset batches",
)
async def calculate_drift(
    request: DriftCalculationRequest,
    current_user: User = Depends(get_current_user),
) -> DriftCalculationResponse:
    """Execute two-sample KS tests and PSI distribution analysis across aligned dataset features."""
    if not request.reference_data or not request.current_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both reference_data and current_data must contain non-empty record lists.",
        )

    drift_report = drift_engine.calculate_dataset_drift(
        reference_data=request.reference_data,
        current_data=request.current_data,
        dataset_name=request.dataset_name,
        drift_share_threshold=request.drift_share_threshold,
    )

    # Update gauge
    DATA_DRIFT_SCORE_GAUGE.set(drift_report.share_of_drifted_features, labels={"dataset_name": request.dataset_name})

    # Evaluate metric in alert manager
    alert_manager.evaluate_metric(
        "omniforge_data_drift_score",
        drift_report.share_of_drifted_features,
        labels={"dataset_name": request.dataset_name},
    )

    feature_schemas = {
        name: FeatureDriftSchema(
            feature_name=f.feature_name,
            feature_type=f.feature_type,
            method=f.method.value,
            test_statistic=f.test_statistic,
            p_value=f.p_value,
            threshold=f.threshold,
            drift_detected=f.drift_detected,
            reference_mean=f.reference_mean,
            current_mean=f.current_mean,
            reference_missing_rate=f.reference_missing_rate,
            current_missing_rate=f.current_missing_rate,
            description=f.description,
        )
        for name, f in drift_report.feature_results.items()
    }

    return DriftCalculationResponse(
        report_id=drift_report.report_id,
        dataset_name=drift_report.dataset_name,
        reference_rows=drift_report.reference_rows,
        current_rows=drift_report.current_rows,
        drift_detected=drift_report.drift_detected,
        share_of_drifted_features=drift_report.share_of_drifted_features,
        number_of_features=drift_report.number_of_features,
        drifted_features_count=drift_report.drifted_features_count,
        drift_threshold=drift_report.drift_threshold,
        feature_results=feature_schemas,
        timestamp=drift_report.timestamp,
    )


@router.get("/alerts", response_model=List[ActiveAlertResponse], summary="List all active SLA alerts")
async def list_active_alerts(
    current_user: User = Depends(get_current_user),
) -> List[ActiveAlertResponse]:
    """Retrieve all currently firing threshold violations and SLA breach incidents."""
    alerts = alert_manager.list_active_alerts()
    return [
        ActiveAlertResponse(
            alert_id=a.alert_id,
            rule_id=a.rule_id,
            rule_name=a.rule_name,
            severity=a.severity.value,
            state=a.state.value,
            current_value=a.current_value,
            threshold=a.threshold,
            message=a.message,
            labels=a.labels,
            fired_at=a.fired_at,
            resolved_at=a.resolved_at,
        )
        for a in alerts
    ]


@router.get("/alerts/rules", response_model=List[AlertRuleResponse], summary="List configured SLA alert rules")
async def list_alert_rules(
    current_user: User = Depends(get_current_user),
) -> List[AlertRuleResponse]:
    rules = alert_manager.list_rules()
    return [
        AlertRuleResponse(
            rule_id=r.rule_id,
            name=r.name,
            description=r.description,
            metric_name=r.metric_name,
            condition=r.condition,
            threshold=r.threshold,
            severity=r.severity.value,
            labels=r.labels,
            enabled=r.enabled,
        )
        for r in rules
    ]


@router.post("/alerts/rules", response_model=AlertRuleResponse, summary="Create a new SLA alert rule")
async def create_alert_rule(
    request: AlertRuleCreateRequest,
    current_admin: User = Depends(require_roles(UserRole.ADMIN)),
) -> AlertRuleResponse:
    rule = AlertRule(
        name=request.name,
        description=request.description,
        metric_name=request.metric_name,
        condition=request.condition,
        threshold=request.threshold,
        severity=AlertSeverity(request.severity),
        labels=request.labels,
        enabled=request.enabled,
    )
    created = alert_manager.add_rule(rule)
    return AlertRuleResponse(
        rule_id=created.rule_id,
        name=created.name,
        description=created.description,
        metric_name=created.metric_name,
        condition=created.condition,
        threshold=created.threshold,
        severity=created.severity.value,
        labels=created.labels,
        enabled=created.enabled,
    )


@router.post("/alerts/evaluate", response_model=EvaluateMetricResponse, summary="Evaluate metric against SLA rules")
async def evaluate_metric(
    request: EvaluateMetricRequest,
    current_user: User = Depends(get_current_user),
) -> EvaluateMetricResponse:
    triggered = alert_manager.evaluate_metric(
        metric_name=request.metric_name,
        value=request.value,
        labels=request.labels,
    )
    return EvaluateMetricResponse(
        evaluated_rules_count=len(alert_manager.list_rules()),
        triggered_alerts=[
            ActiveAlertResponse(
                alert_id=a.alert_id,
                rule_id=a.rule_id,
                rule_name=a.rule_name,
                severity=a.severity.value,
                state=a.state.value,
                current_value=a.current_value,
                threshold=a.threshold,
                message=a.message,
                labels=a.labels,
                fired_at=a.fired_at,
                resolved_at=a.resolved_at,
            )
            for a in triggered
        ],
    )


@router.post(
    "/alerts/{alert_id}/resolve", response_model=ResolveAlertResponse, summary="Manually resolve an active alert"
)
async def resolve_alert(
    alert_id: str,
    current_admin: User = Depends(require_roles(UserRole.ADMIN)),
) -> ResolveAlertResponse:
    resolved = alert_manager.resolve_alert_by_id(alert_id)
    if not resolved:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Active alert '{alert_id}' not found.")
    return ResolveAlertResponse(success=True, alert_id=alert_id, resolved_at=datetime.now(timezone.utc))
