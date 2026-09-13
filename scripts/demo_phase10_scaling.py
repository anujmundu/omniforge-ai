"""Interactive demonstration of Phase 10: Cloud Deployment, Scaling & Distributed Task Mesh.

Showcases Asynchronous Priority Job Scheduling, Worker Concurrency, Dead-Letter Queueing,
and Real-Time Kubernetes Horizontal Pod Autoscaling (HPA) Simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deploy.scaling.base import JobPriority, TaskType
from deploy.scaling.cluster_manager import cluster_manager
from deploy.scaling.task_queue import task_queue
from deploy.scaling.worker_pool import worker_pool

console = Console()


def run_phase10_demo():
    console.print(
        Panel.fit(
            "[bold cyan]OmniForge Phase 10: Cloud Deployment, Scaling & Distributed Task Mesh[/bold cyan]\n"
            "[dim]Kubernetes Helm 3, Distributed Priority Task Mesh, & Dynamic HPA Autoscaling[/dim]",
            border_style="cyan",
        )
    )

    # 1. Asynchronous Priority Task Queueing
    console.print("\n[bold yellow]1. Priority Task Queue Dispatching[/bold yellow]")
    task_queue.clear()

    jobs_to_submit = [
        ("Batch Embedding", TaskType.NLP_EMBEDDING_BATCH, {"texts": ["sentence A", "sentence B"]}, JobPriority.LOW),
        ("RAG Ingestion", TaskType.RAG_DOCUMENT_INDEXING, {"collection_name": "k8s_docs"}, JobPriority.DEFAULT),
        ("Urgent Red-Team Audit", TaskType.RED_TEAM_AUDIT_BATTERY, {}, JobPriority.CRITICAL),
        ("ML Training Run", TaskType.ML_TRAINING, {"model_name": "lightgbm_fraud"}, JobPriority.HIGH),
    ]

    queue_table = Table(title="Dispatched Workloads to Task Mesh", show_header=True, header_style="bold magenta")
    queue_table.add_column("Workload Description", style="cyan")
    queue_table.add_column("Task Type", style="yellow")
    queue_table.add_column("Assigned Priority", justify="center")
    queue_table.add_column("Initial Status", justify="center")

    for desc, ttype, payload, prio in jobs_to_submit:
        job = task_queue.enqueue(ttype, payload, priority=prio)
        prio_color = "red" if prio == JobPriority.CRITICAL else "yellow" if prio == JobPriority.HIGH else "green"
        queue_table.add_row(
            desc,
            ttype.value,
            f"[{prio_color}]{prio.name} ({prio.value})[/{prio_color}]",
            f"[blue]{job.status.value.upper()}[/blue]",
        )
    console.print(queue_table)

    # 2. Worker Mesh Workload Processing
    console.print("\n[bold yellow]2. Distributed Worker Mesh Processing (Priority Preemption)[/bold yellow]")
    proc_table = Table(title="Worker Execution Log", show_header=True, header_style="bold blue")
    proc_table.add_column("Worker Node", style="cyan")
    proc_table.add_column("Executed Job ID", style="dim")
    proc_table.add_column("Task Category", style="yellow")
    proc_table.add_column("Execution Time", justify="right")
    proc_table.add_column("Status", justify="center")

    while True:
        job = worker_pool.process_next_job("worker-cpu-01")
        if not job:
            break
        proc_table.add_row(
            "worker-cpu-01",
            job.job_id,
            job.task_type.value,
            f"{job.execution_time_ms:.2f}ms",
            "[green]COMPLETED[/green]",
        )
    console.print(proc_table)

    # 3. Kubernetes HPA & Cluster Telemetry
    console.print("\n[bold yellow]3. Real-Time Cluster Health & Kubernetes HPA Scaling[/bold yellow]")

    # Simulate high load spike
    worker_pool.update_heartbeat("worker-cpu-01", cpu_pct=88.5, mem_pct=76.0)
    worker_pool.update_heartbeat("worker-cpu-02", cpu_pct=92.0, mem_pct=81.5)

    health = cluster_manager.get_cluster_health()

    console.print(
        Panel(
            f"[bold]Cluster Name:[/bold] {health.cluster_name}\n"
            f"[bold]Active Worker Nodes:[/bold] {health.active_workers} pods\n"
            f"[bold]Total Concurrency Slots:[/bold] {health.total_concurrency_slots} slots\n"
            f"[bold red]Average CPU Utilization:[/bold red] {health.avg_cpu_utilization_pct}%\n"
            f"[bold yellow]Average Memory Utilization:[/bold yellow] {health.avg_memory_utilization_pct}%\n"
            f"[bold green]HPA Autoscaler Recommendation:[/bold green] Scale to {health.hpa_recommended_replicas} pods (Target: 75% CPU)",
            title="[bold green]Kubernetes Cluster Health & HPA Metric Engine[/bold green]",
            border_style="green",
        )
    )

    # 4. Trigger Dynamic Autoscale Event
    console.print("\n[bold yellow]4. Executing Dynamic Horizontal Pod Autoscale (HPA)[/bold yellow]")
    scaled_health = cluster_manager.autoscale_pool(target_replicas=health.hpa_recommended_replicas)
    console.print(
        f"[bold green][SUCCESS][/bold green] Autoscaled worker pool to [cyan]{scaled_health.total_nodes}[/cyan] nodes! Concurrency increased to [cyan]{scaled_health.total_concurrency_slots}[/cyan] parallel slots."
    )


if __name__ == "__main__":
    run_phase10_demo()
