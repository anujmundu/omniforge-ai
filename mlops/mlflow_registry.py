"""OmniForge Central MLflow Model Registry and Experiment Tracker."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from loguru import logger
from mlops.base import ExperimentRun, PipelineStatus

class MLflowRegistryManager:
    def __init__(self) -> None:
        self.experiments: Dict[str, List[str]] = {}
        self.runs: Dict[str, ExperimentRun] = {}
        self._active_run_id: Optional[str] = None

    def start_run(self, experiment_name: str = "default_experiment", tags: Optional[Dict[str, str]] = None) -> ExperimentRun:
        run = ExperimentRun(experiment_name=experiment_name, tags=tags or {})
        self.runs[run.run_id] = run
        if experiment_name not in self.experiments:
            self.experiments[experiment_name] = []
        self.experiments[experiment_name].append(run.run_id)
        self._active_run_id = run.run_id
        return run

    def log_params(self, params: Dict[str, Any], run_id: Optional[str] = None) -> None:
        target_id = run_id or self._active_run_id
        if not target_id or target_id not in self.runs:
            raise KeyError(f"Run ID '{target_id}' not found.")
        self.runs[target_id].parameters.update(params)

    def log_metrics(self, metrics: Dict[str, float], run_id: Optional[str] = None) -> None:
        target_id = run_id or self._active_run_id
        if not target_id or target_id not in self.runs:
            raise KeyError(f"Run ID '{target_id}' not found.")
        self.runs[target_id].metrics.update(metrics)

    def log_artifact(self, artifact_uri: str, run_id: Optional[str] = None) -> None:
        target_id = run_id or self._active_run_id
        if not target_id or target_id not in self.runs:
            raise KeyError(f"Run ID '{target_id}' not found.")
        self.runs[target_id].artifact_uris.append(artifact_uri)

    def end_run(self, run_id: Optional[str] = None, status: PipelineStatus = PipelineStatus.SUCCESS) -> ExperimentRun:
        target_id = run_id or self._active_run_id
        if not target_id or target_id not in self.runs:
            raise KeyError(f"Run ID '{target_id}' not found.")
        run = self.runs[target_id]
        run.status = status
        run.end_time = datetime.now(timezone.utc)
        if self._active_run_id == target_id:
            self._active_run_id = None
        return run

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        return self.runs.get(run_id)

    def list_runs(self, experiment_name: Optional[str] = None) -> List[ExperimentRun]:
        if experiment_name:
            run_ids = self.experiments.get(experiment_name, [])
            return [self.runs[r] for r in run_ids if r in self.runs]
        return list(self.runs.values())
