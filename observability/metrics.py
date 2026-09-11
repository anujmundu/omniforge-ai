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
    """Instantaneous numerical gauge metric."""

    def __init__(self, name: str, description: str, label_names: Optional[List[str]] = None):
        super().__init__(name, description, MetricType.GAUGE, label_names)
        self._values: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge to exact value."""
        key = self._validate_labels(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment gauge by amount."""
        key = self._validate_labels(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement gauge by amount."""
        key = self._validate_labels(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) - amount

    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Retrieve current gauge value."""
        key = self._validate_labels(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> List[Tuple[Dict[str, str], float]]:
        with self._lock:
            return [({k: v for k, v in key}, val) for key, val in self._values.items()]


class Histogram(MetricCollector):
    """Cumulative histogram metric with configurable upper-bound buckets."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__(name, description, MetricType.HISTOGRAM, label_names)
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: Dict[Tuple[Tuple[str, str], ...], int] = {}
        self._sums: Dict[Tuple[Tuple[str, str], ...], float] = {}
        self._bucket_counts: Dict[Tuple[Tuple[str, str], ...], Dict[float, int]] = {}

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe an execution duration or numerical observation."""
        key = self._validate_labels(labels)
        val = float(value)
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1
            self._sums[key] = self._sums.get(key, 0.0) + val
            if key not in self._bucket_counts:
                self._bucket_counts[key] = {b: 0 for b in self.buckets}

            for b in self.buckets:
                if val <= b:
                    self._bucket_counts[key][b] += 1

    def get_count(self, labels: Optional[Dict[str, str]] = None) -> int:
        key = self._validate_labels(labels)
        with self._lock:
            return self._counts.get(key, 0)

    def get_sum(self, labels: Optional[Dict[str, str]] = None) -> float:
        key = self._validate_labels(labels)
        with self._lock:
            return self._sums.get(key, 0.0)

    def collect(self) -> List[Tuple[Dict[str, str], int, float, Dict[float, int]]]:
        with self._lock:
            results = []
            for key in self._counts:
                labels = {k: v for k, v in key}
                results.append((labels, self._counts[key], self._sums[key], self._bucket_counts[key].copy()))
            return results


class PrometheusRegistry:
