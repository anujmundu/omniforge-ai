"""Unit tests for thread-safe Prometheus metrics registry and collectors."""

from observability.metrics import PrometheusRegistry


def test_counter_increments_and_labels():
    registry = PrometheusRegistry()
    counter = registry.counter("test_requests_total", "Test counter", ["method", "endpoint"])

    counter.inc(1.0, {"method": "GET", "endpoint": "/api/v1/health"})
    counter.inc(2.5, {"method": "GET", "endpoint": "/api/v1/health"})
    counter.inc(1.0, {"method": "POST", "endpoint": "/api/v1/predict"})

    assert counter.get_value({"method": "GET", "endpoint": "/api/v1/health"}) == 3.5
    assert counter.get_value({"method": "POST", "endpoint": "/api/v1/predict"}) == 1.0
    assert counter.get_value({"method": "DELETE", "endpoint": "/api/v1/health"}) == 0.0


def test_gauge_set_inc_dec():
    registry = PrometheusRegistry()
    gauge = registry.gauge("test_cpu_usage", "Test gauge", ["node"])

    gauge.set(45.0, {"node": "worker-1"})
    assert gauge.get_value({"node": "worker-1"}) == 45.0

    gauge.inc(5.5, {"node": "worker-1"})
    assert gauge.get_value({"node": "worker-1"}) == 50.5

    gauge.dec(10.5, {"node": "worker-1"})
    assert gauge.get_value({"node": "worker-1"}) == 40.0


def test_histogram_observations_and_buckets():
    registry = PrometheusRegistry()
    hist = registry.histogram("test_latency_seconds", "Test histogram", ["method"], buckets=(0.01, 0.05, 0.1, 0.5))

    hist.observe(0.005, {"method": "GET"})
    hist.observe(0.04, {"method": "GET"})
    hist.observe(0.08, {"method": "GET"})
    hist.observe(0.6, {"method": "GET"})

    assert hist.get_count({"method": "GET"}) == 4
    assert round(hist.get_sum({"method": "GET"}), 3) == 0.725


def test_prometheus_registry_generation():
    registry = PrometheusRegistry()
    c = registry.counter("demo_ops_total", "Demo operations")
    g = registry.gauge("demo_queue_depth", "Demo queue")

    c.inc(10.0)
    g.set(3.0)

    text = registry.generate_prometheus_text()
    assert "# HELP demo_ops_total Demo operations" in text
    assert "# TYPE demo_ops_total counter" in text
    assert "demo_ops_total 10.0" in text
    assert "# HELP demo_queue_depth Demo queue" in text
    assert "demo_queue_depth 3.0" in text
