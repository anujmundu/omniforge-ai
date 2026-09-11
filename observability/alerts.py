"""OmniForge Operational SLA and Threshold Alerting Engine.

Manages alert rule registration, real-time metric evaluation, and active incident state transitions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from loguru import logger

from observability.base import ActiveAlert, AlertRule, AlertSeverity, AlertState


class AlertManager:
    """Production SLA and anomaly alerting engine."""

    def __init__(self) -> None:
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, ActiveAlert] = {}
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register standard out-of-the-box SLA monitoring rules."""
        self.add_rule(
            AlertRule(
                rule_id="sla_p99_latency",
                name="High P99 HTTP Latency",
                description="API gateway latency exceeded 500ms SLA target",
                metric_name="omniforge_http_request_duration_seconds",
                condition="gt",
                threshold=0.5,
                severity=AlertSeverity.CRITICAL,
            )
        )
        self.add_rule(
            AlertRule(
                rule_id="sla_error_rate",
                name="High API Error Rate",
                description="Gateway 5xx error percentage breached 5% threshold",
                metric_name="omniforge_http_error_rate",
                condition="gt",
                threshold=0.05,
                severity=AlertSeverity.CRITICAL,
            )
        )
        self.add_rule(
            AlertRule(
                rule_id="drift_dataset_anomaly",
                name="High Feature Drift Ratio",
                description="More than 25% of model input features have significantly drifted",
                metric_name="omniforge_data_drift_score",
                condition="gt",
                threshold=0.25,
                severity=AlertSeverity.WARNING,
            )
        )
        self.add_rule(
            AlertRule(
                rule_id="vision_low_fps",
                name="Degraded Vision Streaming FPS",
                description="Real-time video pipeline FPS dropped below 15 FPS",
                metric_name="omniforge_vision_fps",
                condition="lt",
                threshold=15.0,
                severity=AlertSeverity.WARNING,
            )
        )
        self.add_rule(
            AlertRule(
                rule_id="system_cpu_saturation",
                name="Host CPU Saturation",
                description="Host CPU usage sustained above 90%",
                metric_name="omniforge_system_cpu_usage_percent",
                condition="gt",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
            )
        )
        self.add_rule(
            AlertRule(
                rule_id="system_memory_exhaustion",
                name="Host Memory Exhaustion",
                description="Host memory usage sustained above 90%",
                metric_name="omniforge_system_memory_usage_percent",
                condition="gt",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
            )
        )

    def add_rule(self, rule: AlertRule) -> AlertRule:
        """Register or update an alerting rule."""
        self._rules[rule.rule_id] = rule
        return rule

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        return self._rules.get(rule_id)

    def list_rules(self) -> List[AlertRule]:
        return list(self._rules.values())

    def delete_rule(self, rule_id: str) -> bool:
        return self._rules.pop(rule_id, None) is not None

    def evaluate_metric(
        self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> List[ActiveAlert]:
        """Evaluate observed metric against all matching rules and fire/resolve alerts."""
        labels = labels or {}
        triggered: List[ActiveAlert] = []

        for rule in self._rules.values():
            if not rule.enabled or rule.metric_name != metric_name:
                continue

            # Check matching labels
            if rule.labels and not all(labels.get(k) == v for k, v in rule.labels.items()):
                continue

            breached = self._check_condition(value, rule.condition, rule.threshold)
            alert_key = f"{rule.rule_id}:{sorted(labels.items())}"

            if breached:
                msg = f"Threshold breached: {value} {rule.condition} {rule.threshold}"
                if alert_key not in self._active_alerts:
                    alert = ActiveAlert(
                        alert_id=f"alert_{uuid4().hex[:10]}",
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        state=AlertState.FIRING,
                        current_value=value,
                        threshold=rule.threshold,
                        message=msg,
                        labels=labels,
                    )
                    self._active_alerts[alert_key] = alert
                    logger.warning(f"ALERT FIRING [{alert.severity.value.upper()}]: {rule.name} -> {msg}")
                    triggered.append(alert)
                else:
                    self._active_alerts[alert_key].current_value = value
            else:
                # Auto-resolve if previously firing
                if alert_key in self._active_alerts:
                    alert = self._active_alerts.pop(alert_key)
                    alert.state = AlertState.RESOLVED
                    alert.resolved_at = datetime.now(timezone.utc)
                    logger.info(f"ALERT RESOLVED: {rule.name}")

        return triggered

    def resolve_alert_by_id(self, alert_id: str) -> bool:
        """Manually resolve an active alert by ID."""
        for k, v in list(self._active_alerts.items()):
            if v.alert_id == alert_id:
                del self._active_alerts[k]
                return True
        return False

    def list_active_alerts(self) -> List[ActiveAlert]:
        return list(self._active_alerts.values())

    @staticmethod
    def _check_condition(val: float, cond: str, thresh: float) -> bool:
        if cond == "gt":
            return val > thresh
        elif cond == "gte":
            return val >= thresh
        elif cond == "lt":
            return val < thresh
        elif cond == "lte":
            return val <= thresh
        elif cond == "eq":
            return abs(val - thresh) < 1e-6
        return False


# Global Alert Manager Instance
alert_manager = AlertManager()
