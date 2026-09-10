"""OmniForge MLOps & CI/CD REST API Router.

Exposes endpoints for experiment tracking, model registry management,
lifecycle stage transitions, automated regression evaluation gates,
and DVC pipeline triggering.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from apps.api.core.dependencies import get_current_user
from apps.api.models.user import User
from apps.api.schemas.mlops import (
    EndRunRequest,
    EvalGateRequest,
    EvalGateResponse,
    ExperimentRunResponse,
    LogMetricsRequest,
    LogParamsRequest,
    MetricComparisonSchema,
    ModelVersionResponse,
    PipelineRunResponse,
    RegisteredModelResponse,
    RegisterModelRequest,
    RunPipelineRequest,
    StartRunRequest,
    TransitionStageRequest,
)
from mlops.base import ModelStage, PipelineStatus
from mlops.dvc_pipeline import dvc_pipeline
from mlops.eval_gate import eval_gate
from mlops.mlflow_registry import mlflow_registry

router = APIRouter(prefix="/mlops", tags=["MLOps & CI/CD"])


# ---------------------------------------------------------------------------
# Experiment Tracking Endpoints
# ---------------------------------------------------------------------------


@router.post("/runs", response_model=ExperimentRunResponse, status_code=status.HTTP_201_CREATED)
async def start_experiment_run(
    payload: StartRunRequest,
    current_user: User = Depends(get_current_user),
) -> ExperimentRunResponse:
    """Start a new experiment tracking run."""
    run = mlflow_registry.start_run(
        experiment_name=payload.experiment_name,
        tags={**payload.tags, "user_id": current_user.id},
    )
    return ExperimentRunResponse(
        run_id=run.run_id,
        experiment_name=run.experiment_name,
        status=run.status.value,
        parameters=run.parameters,
        metrics=run.metrics,
        tags=run.tags,
        artifact_uris=run.artifact_uris,
        start_time=run.start_time,
        end_time=run.end_time,
    )


@router.post("/runs/metrics", response_model=ExperimentRunResponse)
async def log_run_metrics(
    payload: LogMetricsRequest,
    current_user: User = Depends(get_current_user),
) -> ExperimentRunResponse:
    """Log performance metrics to an active experiment run."""
    try:
        mlflow_registry.log_metrics(payload.metrics, run_id=payload.run_id)
        run = mlflow_registry.get_run(payload.run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        return ExperimentRunResponse(
            run_id=run.run_id,
            experiment_name=run.experiment_name,
            status=run.status.value,
            parameters=run.parameters,
            metrics=run.metrics,
            tags=run.tags,
            artifact_uris=run.artifact_uris,
            start_time=run.start_time,
            end_time=run.end_time,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/runs/params", response_model=ExperimentRunResponse)
async def log_run_parameters(
    payload: LogParamsRequest,
    current_user: User = Depends(get_current_user),
) -> ExperimentRunResponse:
    """Log parameters or hyperparameters to an active experiment run."""
    try:
        mlflow_registry.log_params(payload.parameters, run_id=payload.run_id)
        run = mlflow_registry.get_run(payload.run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        return ExperimentRunResponse(
            run_id=run.run_id,
            experiment_name=run.experiment_name,
            status=run.status.value,
            parameters=run.parameters,
            metrics=run.metrics,
            tags=run.tags,
            artifact_uris=run.artifact_uris,
            start_time=run.start_time,
            end_time=run.end_time,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/runs/end", response_model=ExperimentRunResponse)
async def end_experiment_run(
    payload: EndRunRequest,
    current_user: User = Depends(get_current_user),
) -> ExperimentRunResponse:
    """End an active experiment run."""
    try:
        run_status = PipelineStatus.SUCCESS if payload.status == "SUCCESS" else PipelineStatus.FAILED
        run = mlflow_registry.end_run(run_id=payload.run_id, status=run_status)
        return ExperimentRunResponse(
            run_id=run.run_id,
            experiment_name=run.experiment_name,
            status=run.status.value,
            parameters=run.parameters,
            metrics=run.metrics,
            tags=run.tags,
            artifact_uris=run.artifact_uris,
            start_time=run.start_time,
            end_time=run.end_time,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/runs/{run_id}", response_model=ExperimentRunResponse)
async def get_run_by_id(
    run_id: str,
    current_user: User = Depends(get_current_user),
) -> ExperimentRunResponse:
    """Retrieve details for a specific experiment run."""
    run = mlflow_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return ExperimentRunResponse(
        run_id=run.run_id,
        experiment_name=run.experiment_name,
        status=run.status.value,
        parameters=run.parameters,
        metrics=run.metrics,
        tags=run.tags,
        artifact_uris=run.artifact_uris,
        start_time=run.start_time,
        end_time=run.end_time,
    )


@router.get("/runs", response_model=List[ExperimentRunResponse])
async def list_runs(
    experiment_name: Optional[str] = Query(None, description="Optional experiment name filter"),
    current_user: User = Depends(get_current_user),
) -> List[ExperimentRunResponse]:
    """List all experiment runs."""
    runs = mlflow_registry.list_runs(experiment_name=experiment_name)
    return [
        ExperimentRunResponse(
            run_id=r.run_id,
            experiment_name=r.experiment_name,
            status=r.status.value,
            parameters=r.parameters,
            metrics=r.metrics,
            tags=r.tags,
            artifact_uris=r.artifact_uris,
            start_time=r.start_time,
            end_time=r.end_time,
        )
        for r in runs
    ]


# ---------------------------------------------------------------------------
# Model Registry Endpoints
# ---------------------------------------------------------------------------


@router.post("/models/register", response_model=ModelVersionResponse, status_code=status.HTTP_201_CREATED)
async def register_model_version(
    payload: RegisterModelRequest,
    current_user: User = Depends(get_current_user),
) -> ModelVersionResponse:
    """Register a new version of a model from an existing experiment run."""
    try:
        ver = mlflow_registry.register_model(
            name=payload.name,
            run_id=payload.run_id,
            description=payload.description,
            artifact_uri=payload.artifact_uri,
            tags={**payload.tags, "registered_by": current_user.email},
        )
        return ModelVersionResponse(
            model_name=ver.model_name,
            version=ver.version,
            run_id=ver.run_id,
            stage=ver.stage.value,
            description=ver.description,
            metrics=ver.metrics,
            parameters=ver.parameters,
            artifact_uri=ver.artifact_uri,
            created_at=ver.created_at,
            updated_at=ver.updated_at,
            tags=ver.tags,
        )
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models", response_model=List[RegisteredModelResponse])
async def list_registered_models(
    current_user: User = Depends(get_current_user),
) -> List[RegisteredModelResponse]:
    """List all registered models and their versions."""
    models = mlflow_registry.list_registered_models()
    result = []
    for m in models:
        versions_list = [
            ModelVersionResponse(
                model_name=v.model_name,
                version=v.version,
                run_id=v.run_id,
                stage=v.stage.value,
                description=v.description,
                metrics=v.metrics,
                parameters=v.parameters,
                artifact_uri=v.artifact_uri,
                created_at=v.created_at,
                updated_at=v.updated_at,
                tags=v.tags,
            )
            for v in m.versions
        ]
        result.append(
            RegisteredModelResponse(
                name=m.name,
                description=m.description,
                latest_version=m.latest_version,
                versions=versions_list,
                created_at=m.created_at,
                updated_at=m.updated_at,
                tags=m.tags,
            )
        )
    return result


@router.get("/models/{model_name}", response_model=RegisteredModelResponse)
async def get_registered_model(
    model_name: str,
    current_user: User = Depends(get_current_user),
) -> RegisteredModelResponse:
    """Retrieve details for a specific registered model."""
    m = mlflow_registry.get_registered_model(model_name)
    if not m:
        raise HTTPException(status_code=404, detail=f"Registered model '{model_name}' not found.")
    versions_list = [
        ModelVersionResponse(
            model_name=v.model_name,
            version=v.version,
            run_id=v.run_id,
            stage=v.stage.value,
            description=v.description,
            metrics=v.metrics,
            parameters=v.parameters,
            artifact_uri=v.artifact_uri,
            created_at=v.created_at,
            updated_at=v.updated_at,
            tags=v.tags,
        )
        for v in m.versions
    ]
    return RegisteredModelResponse(
        name=m.name,
        description=m.description,
        latest_version=m.latest_version,
        versions=versions_list,
        created_at=m.created_at,
        updated_at=m.updated_at,
        tags=m.tags,
    )


@router.post("/models/transition", response_model=ModelVersionResponse)
async def transition_model_stage(
    payload: TransitionStageRequest,
    current_user: User = Depends(get_current_user),
) -> ModelVersionResponse:
    """Transition a model version to a new lifecycle stage (None, Staging, Production, Archived)."""
    try:
        target_stage = ModelStage(payload.target_stage)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage '{payload.target_stage}'. Allowed stages: {[s.value for s in ModelStage]}",
        )

    try:
        ver = mlflow_registry.transition_stage(
            model_name=payload.model_name,
            version=payload.version,
            target_stage=target_stage,
            archive_existing_versions=payload.archive_existing,
        )
        return ModelVersionResponse(
            model_name=ver.model_name,
            version=ver.version,
            run_id=ver.run_id,
            stage=ver.stage.value,
            description=ver.description,
            metrics=ver.metrics,
            parameters=ver.parameters,
            artifact_uri=ver.artifact_uri,
            created_at=ver.created_at,
            updated_at=ver.updated_at,
            tags=ver.tags,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Automated Evaluation Gate & Regression Benchmarking
# ---------------------------------------------------------------------------


@router.post("/evaluate-gate", response_model=EvalGateResponse)
async def evaluate_candidate_gate(
    payload: EvalGateRequest,
    current_user: User = Depends(get_current_user),
) -> EvalGateResponse:
    """Evaluate candidate model against champion baseline and optional auto-promotion."""
    try:
        result = eval_gate.evaluate_candidate(
            model_name=payload.model_name,
            candidate_version=payload.candidate_version,
            golden_dataset_metrics=payload.golden_dataset_metrics,
            auto_promote=payload.auto_promote,
        )
        comparisons_out = [
            MetricComparisonSchema(
                metric_name=c.metric_name,
                candidate_value=c.candidate_value,
                champion_value=c.champion_value,
                delta=c.delta,
                threshold=c.threshold,
                passed=c.passed,
                description=c.description,
            )
            for c in result.comparisons
        ]
        return EvalGateResponse(
            gate_id=result.gate_id,
            model_name=result.model_name,
            candidate_version=result.candidate_version,
            champion_version=result.champion_version,
            passed=result.passed,
            promoted=result.promoted,
            decision_reason=result.decision_reason,
            comparisons=comparisons_out,
            timestamp=result.timestamp,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluation gate execution failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# DVC Pipeline Execution Endpoints
# ---------------------------------------------------------------------------


@router.post("/pipelines/run", response_model=PipelineRunResponse)
async def trigger_dvc_pipeline(
    payload: RunPipelineRequest,
    current_user: User = Depends(get_current_user),
) -> PipelineRunResponse:
    """Execute reproducible DVC pipeline stages with caching."""
    try:
        # Register standard default stages if none registered
        if not dvc_pipeline.stages:
            dvc_pipeline.register_stage(
                name="load_data",
                deps=["storage/datasets/"],
                outs=["storage/processed/raw_split.parquet"],
                params={"train_ratio": 0.8},
                callback=lambda p: {"samples_loaded": 1000},
            )
            dvc_pipeline.register_stage(
                name="preprocess",
                deps=["storage/processed/raw_split.parquet"],
                outs=["storage/processed/features.parquet"],
                params={"strategy": "median"},
                callback=lambda p: {"features_generated": 15},
            )
            dvc_pipeline.register_stage(
                name="train",
                deps=["storage/processed/features.parquet"],
                outs=["storage/models/candidate_model.joblib"],
                params={"n_estimators": 100},
                callback=lambda p: {"train_f1": 0.94, "train_accuracy": 0.95},
            )

        res = dvc_pipeline.run_pipeline(force=payload.force)
        return PipelineRunResponse(
            pipeline_id=res.pipeline_id,
            status=res.status.value,
            executed_stages=res.executed_stages,
            cached_stages=res.cached_stages,
            duration_seconds=res.duration_seconds,
            stage_results=res.stage_results,
            timestamp=res.timestamp,
        )
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
