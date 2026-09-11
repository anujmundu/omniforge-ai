"""Unit tests for operational SLA alerting engine."""

from observability.alerts import AlertManager
from observability.base import AlertRule, AlertSeverity, AlertState


def test_alert_rule_registration_and_evaluation():
    manager = AlertManager()
    rule = AlertRule(
        rule_id="custom_cpu_alert",
        name="Custom High CPU",
        metric_name="custom_cpu",
        condition="gt",
        threshold=85.0,
        severity=AlertSeverity.WARNING,
    )
    manager.add_rule(rule)

    # Below threshold -> No alert
    triggered = manager.evaluate_metric("custom_cpu", 70.0)
    assert len(triggered) == 0

    # Above threshold -> Alert fires
    triggered = manager.evaluate_metric("custom_cpu", 92.5)
    assert len(triggered) == 1
    assert triggered[0].rule_id == "custom_cpu_alert"
    assert triggered[0].state == AlertState.FIRING
    assert triggered[0].current_value == 92.5

    # Evaluation drops below threshold -> Auto-resolves
    manager.evaluate_metric("custom_cpu", 60.0)
    assert len(manager.list_active_alerts()) == 0


def test_manual_alert_resolution():
    manager = AlertManager()
    manager.evaluate_metric("omniforge_http_request_duration_seconds", 0.95)
    active = manager.list_active_alerts()
    assert len(active) == 1

    alert_id = active[0].alert_id
    success = manager.resolve_alert_by_id(alert_id)
    assert success is True
    assert len(manager.list_active_alerts()) == 0
