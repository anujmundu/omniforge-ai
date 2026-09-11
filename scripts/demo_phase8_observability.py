"""OmniForge Phase 8 Demonstration: Production Observability, Prometheus Metrics & Drift Monitoring."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from observability.alerts import alert_manager
from observability.drift import drift_engine
from observability.metrics import (
    AGENT_STEP_DURATION_SECONDS,
    DATA_DRIFT_SCORE_GAUGE,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    ML_INFERENCE_DURATION_SECONDS,
    NLP_PROCESSING_DURATION_SECONDS,
    NLP_TOKEN_THROUGHPUT_TOTAL,
    RAG_RETRIEVAL_DURATION_SECONDS,
    SYSTEM_CPU_USAGE_PERCENT,
    SYSTEM_MEMORY_USAGE_PERCENT,
    VISION_FPS_GAUGE,
    VISION_PROCESSING_DURATION_SECONDS,
    metrics_registry,
)

console = Console()


def run_phase8_observability_demo() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]OmniForge Phase 8: Production Observability, Prometheus Metrics & Statistical Drift[/bold cyan]\n"
            "[dim]Demonstrating real-time telemetry instrumentation, KS-test/PSI data drift analysis, and SLA alerting[/dim]",
            border_style="cyan",
        )
    )

    # 1. Emit Multimodal Telemetry Metrics
    console.print("\n[bold yellow]Step 1: Emitting Real-Time Multimodal Telemetry Metrics...[/bold yellow]")

    # HTTP requests
    HTTP_REQUESTS_TOTAL.inc(
        amount=250.0, labels={"method": "GET", "endpoint": "/api/v1/projects", "status_code": "200"}
    )
    HTTP_REQUESTS_TOTAL.inc(
        amount=120.0, labels={"method": "POST", "endpoint": "/api/v1/ml/predict", "status_code": "200"}
    )
    HTTP_REQUESTS_TOTAL.inc(
        amount=5.0, labels={"method": "POST", "endpoint": "/api/v1/ml/predict", "status_code": "500"}
    )

    # Latencies
    HTTP_REQUEST_DURATION_SECONDS.observe(0.012, labels={"method": "GET", "endpoint": "/api/v1/projects"})
    HTTP_REQUEST_DURATION_SECONDS.observe(0.045, labels={"method": "POST", "endpoint": "/api/v1/ml/predict"})
    ML_INFERENCE_DURATION_SECONDS.observe(0.008, labels={"model_type": "classification", "model_id": "churn_xgb_v1"})
    VISION_PROCESSING_DURATION_SECONDS.observe(0.022, labels={"task": "yolo_object_detection"})
    VISION_FPS_GAUGE.set(29.8, labels={"stream_id": "camera_front_01"})
    NLP_TOKEN_THROUGHPUT_TOTAL.inc(amount=14500.0, labels={"operation": "minilm_embeddings"})
    NLP_PROCESSING_DURATION_SECONDS.observe(0.018, labels={"operation": "ner_extraction"})
    RAG_RETRIEVAL_DURATION_SECONDS.observe(
        0.014, labels={"collection_name": "enterprise_docs", "stage": "vector_search"}
    )
    AGENT_STEP_DURATION_SECONDS.observe(0.125, labels={"agent_name": "SupervisorAgent", "tool_name": "ml_predict"})

    # System gauges
    SYSTEM_CPU_USAGE_PERCENT.set(42.5)
    SYSTEM_MEMORY_USAGE_PERCENT.set(68.2)

    console.print(
        "[green][OK] Telemetry metrics recorded across API, Classical ML, Vision, NLP, RAG, and Agents.[/green]"
    )

    # 2. Simulate Statistical Data Drift Analysis
    console.print("\n[bold yellow]Step 2: Executing Statistical Data Drift Detection (KS-Test & PSI)...[/bold yellow]")
    np.random.seed(42)

    # Baseline training reference
    ref_df = pd.DataFrame(
        {
            "monthly_charges": np.random.normal(65.0, 12.0, 500),
            "tenure_months": np.random.uniform(1, 72, 500),
            "total_calls": np.random.poisson(25, 500),
            "contract_type": ["monthly"] * 300 + ["one-year"] * 200,
        }
    )

    # Production current inference batch (with deliberate billing distribution shift)
    curr_df = pd.DataFrame(
        {
            "monthly_charges": np.random.normal(118.0, 18.0, 500),  # HEAVILY DRIFTED
            "tenure_months": np.random.uniform(1, 72, 500),  # STABLE
            "total_calls": np.random.poisson(26, 500),  # STABLE
            "contract_type": ["monthly"] * 290 + ["one-year"] * 210,  # STABLE
        }
    )

    drift_report = drift_engine.calculate_dataset_drift(
        reference_data=ref_df,
        current_data=curr_df,
        dataset_name="customer_churn_production",
        drift_share_threshold=0.25,
    )

    # Update platform drift score gauge
    DATA_DRIFT_SCORE_GAUGE.set(
        drift_report.share_of_drifted_features, labels={"dataset_name": "customer_churn_production"}
    )

    # Display Drift Table
    drift_table = Table(title="Feature-Level Statistical Drift Report")
    drift_table.add_column("Feature", style="cyan")
    drift_table.add_column("Type", style="magenta")
    drift_table.add_column("Method", style="dim")
    drift_table.add_column("Test Stat", justify="right")
    drift_table.add_column("P-Value", justify="right")
    drift_table.add_column("Drift Status", justify="center")

    for f_name, f_res in drift_report.feature_results.items():
        status_style = "[bold red]DRIFTED[/bold red]" if f_res.drift_detected else "[green]STABLE[/green]"
        pval_str = f"{f_res.p_value:.4f}" if f_res.p_value is not None else "N/A"
        drift_table.add_row(
            f_name,
            f_res.feature_type,
            f_res.method.value.upper(),
            f"{f_res.test_statistic:.4f}",
            pval_str,
            status_style,
        )

    console.print(drift_table)
    console.print(
        f"[bold]Dataset Level Assessment:[/bold] {drift_report.drifted_features_count}/{drift_report.number_of_features} "
        f"features drifted ({drift_report.share_of_drifted_features:.1%}) -> "
        f"{'[bold red]DATASET DRIFT DETECTED[/bold red]' if drift_report.drift_detected else '[green]DATASET STABLE[/green]'}"
    )

    # 3. Trigger SLA Alerting Engine
    console.print("\n[bold yellow]Step 3: Evaluating SLA Alerting Rules...[/bold yellow]")

    # Evaluate drift score
    alert_manager.evaluate_metric(
        "omniforge_data_drift_score",
        drift_report.share_of_drifted_features,
        labels={"dataset": "customer_churn_production"},
    )

    # Evaluate artificial high latency
    alert_manager.evaluate_metric("omniforge_http_request_duration_seconds", 0.785)

    alerts_table = Table(title="Active Operational SLA Alerts")
    alerts_table.add_column("Alert ID", style="dim")
    alerts_table.add_column("Rule Name", style="bold cyan")
    alerts_table.add_column("Severity", justify="center")
    alerts_table.add_column("State", justify="center")
    alerts_table.add_column("Trigger Value", justify="right")
    alerts_table.add_column("Message", style="yellow")

    for a in alert_manager.list_active_alerts():
        sev_style = (
            "[bold red]CRITICAL[/bold red]" if a.severity.value == "critical" else "[bold yellow]WARNING[/bold yellow]"
        )
        alerts_table.add_row(
            a.alert_id,
            a.rule_name,
            sev_style,
            f"[bold red]{a.state.value.upper()}[/bold red]",
            str(a.current_value),
            a.message,
        )

    console.print(alerts_table)

    # 4. Prometheus Scrape Output Preview
    console.print("\n[bold yellow]Step 4: Prometheus Exposition Text (/metrics Preview)...[/bold yellow]")
    prom_text = metrics_registry.generate_prometheus_text()
    sample_lines = [line for line in prom_text.splitlines() if line and not line.startswith("#")][:8]
    console.print(
        Panel("\n".join(sample_lines) + "\n...", title="Prometheus Telemetry Scrape (Sample)", border_style="green")
    )

    console.print(
        Panel.fit(
            "[bold green][OK] Phase 8: Production Observability Stack Fully Operational![/bold green]\n"
            "- Metrics Registry: Thread-safe Counter, Gauge, and Histogram with `/metrics` exporter\n"
            "- Statistical Drift: Two-sample KS-test, PSI, missingness tracking, and dataset reports\n"
            "- SLA Alerting: Configurable thresholds, active alert state machine, and auto-resolution\n"
            "- Dashboards: Prometheus scrape config & Grafana overview dashboard provisioned",
            border_style="green",
        )
    )


if __name__ == "__main__":
    run_phase8_observability_demo()
