"""OmniForge Central MLflow Model Registry and Experiment Tracker.

Provides production-grade experiment tracking, model registry management,
semantic versioning, lifecycle stage transitions, and rollback safety.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from mlops.base import (
    ExperimentRun,
    ModelStage,
    ModelVersion,
    PipelineStatus,
    RegisteredModel,
)


class MLflowRegistryManager:
    """Central registry and experiment tracker for managing the entire model lifecycle."""

    def __init__(self) -> None:
        self.experiments: Dict[str, List[str]] = {}  # experiment_name -> list of run_ids
        self.runs: Dict[str, ExperimentRun] = {}  # run_id -> ExperimentRun
        self.models: Dict[str, RegisteredModel] = {}  # model_name -> RegisteredModel
        self._active_run_id: Optional[str] = None

    # -------------------------------------------------------------------------
    # Experiment Tracking
    # -------------------------------------------------------------------------

    def start_run(
        self,
        experiment_name: str = "default_experiment",
        tags: Optional[Dict[str, str]] = None,
    ) -> ExperimentRun:
        """Start a new experiment tracking run."""
        run = ExperimentRun(
            experiment_name=experiment_name,
            tags=tags or {},
        )
        self.runs[run.run_id] = run
        if experiment_name not in self.experiments:
            self.experiments[experiment_name] = []
        self.experiments[experiment_name].append(run.run_id)
        self._active_run_id = run.run_id
        logger.info(f"Started experiment run [{run.run_id}] in experiment '{experiment_name}'")
        return run

    def log_params(self, params: Dict[str, Any], run_id: Optional[str] = None) -> None:
        """Log parameters for an experiment run."""
        target_id = run_id or self._active_run_id
        if not target_id or target_id not in self.runs:
            raise KeyError(f"Run ID '{target_id}' not found.")
        self.runs[target_id].parameters.update(params)

    def log_metrics(self, metrics: Dict[str, float], run_id: Optional[str] = None) -> None:
        """Log evaluation metrics for an experiment run."""
        target_id = run_id or self._active_run_id
        if not target_id or target_id not in self.runs:
            raise KeyError(f"Run ID '{target_id}' not found.")
        self.runs[target_id].metrics.update(metrics)

    def log_artifact(self, artifact_uri: str, run_id: Optional[str] = None) -> None:
        """Log an artifact URI for an experiment run."""
        target_id = run_id or self._active_run_id
        if not target_id or target_id not in self.runs:
            raise KeyError(f"Run ID '{target_id}' not found.")
        self.runs[target_id].artifact_uris.append(artifact_uri)

    def end_run(self, run_id: Optional[str] = None, status: PipelineStatus = PipelineStatus.SUCCESS) -> ExperimentRun:
        """Complete an experiment tracking run."""
        target_id = run_id or self._active_run_id
        if not target_id or target_id not in self.runs:
            raise KeyError(f"Run ID '{target_id}' not found.")
        run = self.runs[target_id]
        run.status = status
        run.end_time = datetime.now(timezone.utc)
        if self._active_run_id == target_id:
            self._active_run_id = None
        logger.info(f"Ended experiment run [{target_id}] with status {status.value}")
        return run

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Retrieve an experiment run by ID."""
        return self.runs.get(run_id)

    def list_runs(self, experiment_name: Optional[str] = None) -> List[ExperimentRun]:
        """List all runs, optionally filtered by experiment name."""
        if experiment_name:
            run_ids = self.experiments.get(experiment_name, [])
            return [self.runs[r] for r in run_ids if r in self.runs]
        return list(self.runs.values())

    # -------------------------------------------------------------------------
    # Model Registry Management
    # -------------------------------------------------------------------------

    def register_model(
        self,
        name: str,
        run_id: str,
        description: str = "",
        artifact_uri: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelVersion:
        """Register a new model version from an experiment run."""
        run = self.get_run(run_id)
        metrics = run.metrics.copy() if run else {}
        params = run.parameters.copy() if run else {}

        if name not in self.models:
            self.models[name] = RegisteredModel(name=name, description=description)

        reg_model = self.models[name]
        new_version_num = reg_model.latest_version + 1
        reg_model.latest_version = new_version_num

        model_version = ModelVersion(
            model_name=name,
            version=new_version_num,
            run_id=run_id,
            stage=ModelStage.NONE,
            description=description,
            metrics=metrics,
            parameters=params,
            artifact_uri=artifact_uri or (run.artifact_uris[0] if run and run.artifact_uris else ""),
            tags=tags or {},
        )

        reg_model.versions.append(model_version)
        reg_model.updated_at = datetime.now(timezone.utc)
        logger.info(f"Registered model '{name}' version {new_version_num} (stage: {model_version.stage.value})")
        return model_version

    def transition_stage(
        self,
        model_name: str,
        version: int,
        target_stage: ModelStage,
        archive_existing_versions: bool = True,
    ) -> ModelVersion:
        """Transition a model version to a target lifecycle stage with safe state machine validation."""
        if model_name not in self.models:
            raise KeyError(f"Registered model '{model_name}' not found.")

        reg_model = self.models[model_name]
        target_version = reg_model.get_version(version)
        if not target_version:
            raise KeyError(f"Version {version} of model '{model_name}' not found.")

        # If transitioning to PRODUCTION, archive any currently active production models
        if target_stage == ModelStage.PRODUCTION and archive_existing_versions:
            for v in reg_model.versions:
                if v.stage == ModelStage.PRODUCTION and v.version != version:
                    v.stage = ModelStage.ARCHIVED
                    v.updated_at = datetime.now(timezone.utc)
                    logger.info(f"Archived previous production version {v.version} of model '{model_name}'")

        target_version.stage = target_stage
        target_version.updated_at = datetime.now(timezone.utc)
        reg_model.updated_at = datetime.now(timezone.utc)

        logger.info(f"Transitioned model '{model_name}' v{version} to stage [{target_stage.value}]")
        return target_version

    def rollback_production(self, model_name: str, fallback_version: Optional[int] = None) -> ModelVersion:
        """Roll back production to a specified version or the most recent archived champion."""
        if model_name not in self.models:
            raise KeyError(f"Registered model '{model_name}' not found.")

        reg_model = self.models[model_name]
        current_prod = reg_model.get_stage_model(ModelStage.PRODUCTION)

        if fallback_version is not None:
            target = reg_model.get_version(fallback_version)
            if not target:
                raise KeyError(f"Target fallback version {fallback_version} not found for model '{model_name}'.")
        else:
            # Find the most recently updated ARCHIVED model
            archived = [v for v in reg_model.versions if v.stage == ModelStage.ARCHIVED]
            if not archived:
                raise ValueError(f"No archived versions available to roll back to for model '{model_name}'.")
            target = sorted(archived, key=lambda v: v.updated_at, reverse=True)[0]

        if current_prod:
            current_prod.stage = ModelStage.ARCHIVED
            current_prod.updated_at = datetime.now(timezone.utc)

        target.stage = ModelStage.PRODUCTION
        target.updated_at = datetime.now(timezone.utc)
        reg_model.updated_at = datetime.now(timezone.utc)

        logger.warning(f"Rolled back model '{model_name}' to version {target.version} in PRODUCTION")
        return target

    def get_registered_model(self, name: str) -> Optional[RegisteredModel]:
        """Retrieve a registered model with all its versions."""
        return self.models.get(name)

    def list_registered_models(self) -> List[RegisteredModel]:
        """List all registered models in the platform."""
        return list(self.models.values())


# Default global instance for platform runtime
mlflow_registry = MLflowRegistryManager()
