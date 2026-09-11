"""OmniForge Thread-Safe Prometheus Metrics Registry and Telemetry Collectors.

Provides high-performance in-memory metric collectors (Counter, Gauge, Histogram)
and standard Prometheus exposition text formatting for continuous platform observability.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

from observability.base import MetricType


def _format_labels(labels: Dict[str, str]) -> str:
    """Format dictionary labels into Prometheus label string."""
    if not labels:
        return ""
    formatted = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{{{formatted}}}"


class MetricCollector:
    """Base class for thread-safe Prometheus metrics."""

    def __init__(self, name: str, description: str, metric_type: MetricType, label_names: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.metric_type = metric_type
        self.label_names = tuple(label_names or [])
        self._lock = threading.Lock()

    def _validate_labels(self, labels: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
        labels = labels or {}
        return tuple(sorted((k, str(v)) for k, v in labels.items()))


class Counter(MetricCollector):
    """Monotonically increasing counter metric."""

    def __init__(self, name: str, description: str, label_names: Optional[List[str]] = None):
        super().__init__(name, description, MetricType.COUNTER, label_names)
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter by specified amount (must be non-negative)."""
        if amount < 0:
            raise ValueError("Counter increments must be non-negative.")
        key = self._validate_labels(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Retrieve current counter value for given label set."""
        key = self._validate_labels(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> List[Tuple[Dict[str, str], float]]:
        with self._lock:
            return [({k: v for k, v in key}, val) for key, val in self._values.items()]


class Gauge(MetricCollector):
