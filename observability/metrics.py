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
    """Central registry storing all metrics and exporting Prometheus text format."""

    def __init__(self) -> None:
        self._collectors: Dict[str, MetricCollector] = {}
        self._lock = threading.Lock()

    def register(self, collector: MetricCollector) -> MetricCollector:
        """Register a new metric collector."""
        with self._lock:
            if collector.name in self._collectors:
                return self._collectors[collector.name]
            self._collectors[collector.name] = collector
            return collector

    def counter(self, name: str, description: str, label_names: Optional[List[str]] = None) -> Counter:
        c = Counter(name, description, label_names)
        return self.register(c)  # type: ignore

    def gauge(self, name: str, description: str, label_names: Optional[List[str]] = None) -> Gauge:
        g = Gauge(name, description, label_names)
        return self.register(g)  # type: ignore

    def histogram(
        self,
        name: str,
        description: str,
        label_names: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        h = Histogram(name, description, label_names, buckets)
        return self.register(h)  # type: ignore

    def get(self, name: str) -> Optional[MetricCollector]:
        with self._lock:
            return self._collectors.get(name)

    def generate_prometheus_text(self) -> str:
        """Generate official Prometheus exposition text."""
        lines: List[str] = []
        with self._lock:
            collectors = list(self._collectors.values())

        for c in collectors:
            lines.append(f"# HELP {c.name} {c.description}")
            lines.append(f"# TYPE {c.name} {c.metric_type.value}")

            if isinstance(c, (Counter, Gauge)):
                for labels, val in c.collect():
                    lbl_str = _format_labels(labels)
                    lines.append(f"{c.name}{lbl_str} {val}")
            elif isinstance(c, Histogram):
                for labels, count, total_sum, b_counts in c.collect():
                    # Cumulative buckets
                    for b in sorted(b_counts.keys()):
                        b_labels = labels.copy()
                        b_labels["le"] = str(b)
                        lines.append(f"{c.name}_bucket{_format_labels(b_labels)} {b_counts[b]}")
                    inf_labels = labels.copy()
                    inf_labels["le"] = "+Inf"
                    lines.append(f"{c.name}_bucket{_format_labels(inf_labels)} {count}")
                    lines.append(f"{c.name}_sum{_format_labels(labels)} {total_sum}")
                    lines.append(f"{c.name}_count{_format_labels(labels)} {count}")

        return "\n".join(lines) + "\n"


# Global Platform Metrics Registry
metrics_registry = PrometheusRegistry()

# -----------------------------------------------------------------------------
# Pre-Configured Multimodal Telemetry Metrics
# -----------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = metrics_registry.counter(
    "omniforge_http_requests_total",
    "Total HTTP requests handled by OmniForge API gateway",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = metrics_registry.histogram(
    "omniforge_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

ML_INFERENCE_DURATION_SECONDS = metrics_registry.histogram(
    "omniforge_ml_inference_duration_seconds",
    "Classical ML model inference latency in seconds",
    ["model_type", "model_id"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

VISION_PROCESSING_DURATION_SECONDS = metrics_registry.histogram(
    "omniforge_vision_processing_duration_seconds",
    "Computer vision inference and tracking duration in seconds",
    ["task"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
)

VISION_FPS_GAUGE = metrics_registry.gauge(
    "omniforge_vision_fps",
    "Real-time processed frames per second for video pipelines",
    ["stream_id"],
)

NLP_TOKEN_THROUGHPUT_TOTAL = metrics_registry.counter(
    "omniforge_nlp_tokens_total",
    "Total NLP tokens embedded, classified, or extracted",
    ["operation"],
)

NLP_PROCESSING_DURATION_SECONDS = metrics_registry.histogram(
    "omniforge_nlp_processing_duration_seconds",
    "NLP transformer processing duration in seconds",
    ["operation"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

RAG_RETRIEVAL_DURATION_SECONDS = metrics_registry.histogram(
    "omniforge_rag_retrieval_duration_seconds",
    "RAG hybrid vector search and reranking latency in seconds",
    ["collection_name", "stage"],
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

AGENT_STEP_DURATION_SECONDS = metrics_registry.histogram(
    "omniforge_agent_step_duration_seconds",
    "Autonomous agent reasoning and tool execution duration in seconds",
    ["agent_name", "tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

DATA_DRIFT_SCORE_GAUGE = metrics_registry.gauge(
    "omniforge_data_drift_score",
    "Continuous statistical dataset drift ratio (0.0 to 1.0)",
    ["dataset_name"],
)

SYSTEM_CPU_USAGE_PERCENT = metrics_registry.gauge(
    "omniforge_system_cpu_usage_percent",
    "System CPU usage percentage",
)

SYSTEM_MEMORY_USAGE_PERCENT = metrics_registry.gauge(
    "omniforge_system_memory_usage_percent",
    "System memory utilization percentage",
)
