import time
from typing import Any, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.database import get_db_session
from apps.api.core.dependencies import get_current_user, require_roles
from apps.api.models.artifact import Artifact, ArtifactType
from apps.api.models.experiment import Experiment, ExperimentDomain, ExperimentStatus
from apps.api.models.project import Project
from apps.api.models.user import User, UserRole
from apps.api.schemas.ml import (
    InferenceRequest,
    InferenceResponse,
    ModelInfoResponse,
    TrainAnomalyRequest,
    TrainClassificationRequest,
    TrainForecastingRequest,
    TrainJobResponse,
    TrainRegressionRequest,
)
from ml.anomaly.engine import AnomalyEngine
from ml.base import TaskType
from ml.classification.engine import ClassificationEngine
from ml.forecasting.engine import ForecastingEngine
from ml.registry import registry
from ml.regression.engine import RegressionEngine

router = APIRouter(prefix="/ml", tags=["Classical Machine Learning Engine"])


@router.post(
    "/train/classification",
    response_model=TrainJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Train and benchmark classification model",
)
async def train_classification(
    req: TrainClassificationRequest,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.ML_ENGINEER, UserRole.DATA_SCIENTIST)),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    # 1. Verify project exists
    proj_stmt = select(Project).where(Project.id == req.project_id)
    project = (await db.execute(proj_stmt)).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    df = pd.DataFrame(req.dataset_records)
    if req.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{req.target_column}' not found in provided dataset.",
        )

    X = df.drop(columns=[req.target_column])
    y = df[req.target_column]

    # 2. Train model
    engine = ClassificationEngine(
        model_id=req.model_id,
        algorithm=req.algorithm,
        hyperparameters=req.hyperparameters,
    )
    engine.fit(X=X, y=y, target_name=req.target_column, validation_split=req.validation_split)
    eval_res = engine.evaluate(X, y)
    artifact_path = registry.register_and_save(engine)

    # 3. Record in Experiment tracking DB
    exp = Experiment(
        project_id=req.project_id,
        name=f"Classification_{req.model_id}",
        domain=ExperimentDomain.CLASSICAL_ML,
        status=ExperimentStatus.COMPLETED,
        model_name=req.algorithm,
        parameters=req.hyperparameters,
        metrics=eval_res.metrics,
        duration_seconds=engine.metadata.get("training_duration_sec", 0.0),
    )
    db.add(exp)
    await db.commit()
    await db.refresh(exp)

    art = Artifact(
        experiment_id=exp.id,
        name=f"{req.model_id}.joblib",
        artifact_type=ArtifactType.MODEL_WEIGHTS,
        uri=artifact_path,
    )
    db.add(art)
    await db.commit()

    return TrainJobResponse(
        status="COMPLETED",
        model_id=req.model_id,
        task_type=TaskType.CLASSIFICATION,
        artifact_uri=artifact_path,
        evaluation=eval_res,
    )


@router.post(
    "/train/regression",
    response_model=TrainJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Train and benchmark regression model",
)
async def train_regression(
    req: TrainRegressionRequest,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.ML_ENGINEER, UserRole.DATA_SCIENTIST)),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    proj_stmt = select(Project).where(Project.id == req.project_id)
    project = (await db.execute(proj_stmt)).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    df = pd.DataFrame(req.dataset_records)
    if req.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{req.target_column}' not found in provided dataset.",
        )

    X = df.drop(columns=[req.target_column])
    y = df[req.target_column]

    engine = RegressionEngine(
        model_id=req.model_id,
        algorithm=req.algorithm,
        hyperparameters=req.hyperparameters,
    )
    engine.fit(X=X, y=y, target_name=req.target_column, validation_split=req.validation_split)
    eval_res = engine.evaluate(X, y)
    artifact_path = registry.register_and_save(engine)

    exp = Experiment(
        project_id=req.project_id,
        name=f"Regression_{req.model_id}",
        domain=ExperimentDomain.CLASSICAL_ML,
        status=ExperimentStatus.COMPLETED,
        model_name=req.algorithm,
        parameters=req.hyperparameters,
        metrics=eval_res.metrics,
        duration_seconds=engine.metadata.get("training_duration_sec", 0.0),
    )
    db.add(exp)
    await db.commit()
    await db.refresh(exp)

    return TrainJobResponse(
        status="COMPLETED",
        model_id=req.model_id,
        task_type=TaskType.REGRESSION,
        artifact_uri=artifact_path,
        evaluation=eval_res,
    )


@router.post(
    "/train/anomaly",
    response_model=TrainJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Fit unsupervised anomaly detection model",
)
async def train_anomaly(
    req: TrainAnomalyRequest,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.ML_ENGINEER, UserRole.DATA_SCIENTIST)),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    proj_stmt = select(Project).where(Project.id == req.project_id)
    project = (await db.execute(proj_stmt)).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    df = pd.DataFrame(req.dataset_records)
    engine = AnomalyEngine(
        model_id=req.model_id,
        algorithm=req.algorithm,
        contamination=req.contamination,
        hyperparameters=req.hyperparameters,
    )
    engine.fit(df)
    eval_res = engine.evaluate(df)
    artifact_path = registry.register_and_save(engine)

    exp = Experiment(
        project_id=req.project_id,
        name=f"Anomaly_{req.model_id}",
        domain=ExperimentDomain.CLASSICAL_ML,
        status=ExperimentStatus.COMPLETED,
        model_name=req.algorithm,
        parameters={"contamination": req.contamination, **(req.hyperparameters or {})},
        metrics=eval_res.metrics,
        duration_seconds=engine.metadata.get("training_duration_sec", 0.0),
    )
    db.add(exp)
    await db.commit()
    await db.refresh(exp)

    return TrainJobResponse(
        status="COMPLETED",
        model_id=req.model_id,
        task_type=TaskType.ANOMALY_DETECTION,
        artifact_uri=artifact_path,
        evaluation=eval_res,
    )


@router.post(
    "/train/forecasting",
    response_model=TrainJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Fit multi-horizon time-series forecasting model",
)
async def train_forecasting(
    req: TrainForecastingRequest,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.ML_ENGINEER, UserRole.DATA_SCIENTIST)),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    proj_stmt = select(Project).where(Project.id == req.project_id)
    project = (await db.execute(proj_stmt)).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    df = pd.DataFrame(req.dataset_records)
    engine = ForecastingEngine(
        model_id=req.model_id,
        lags=req.lags,
        hyperparameters=req.hyperparameters,
    )
    engine.fit(df, date_col=req.date_column, target_col=req.target_column)
    eval_res = engine.evaluate(df)
    artifact_path = registry.register_and_save(engine)

    exp = Experiment(
        project_id=req.project_id,
        name=f"Forecasting_{req.model_id}",
        domain=ExperimentDomain.CLASSICAL_ML,
        status=ExperimentStatus.COMPLETED,
        model_name="gradient_boosting_forecaster",
        parameters={"lags": req.lags, **(req.hyperparameters or {})},
        metrics=eval_res.metrics,
        duration_seconds=engine.metadata.get("training_duration_sec", 0.0),
    )
    db.add(exp)
    await db.commit()
    await db.refresh(exp)

    return TrainJobResponse(
        status="COMPLETED",
        model_id=req.model_id,
        task_type=TaskType.FORECASTING,
        artifact_uri=artifact_path,
        evaluation=eval_res,
    )


@router.post(
    "/predict",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute real-time low-latency ML inference",
)
async def predict(
    req: InferenceRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    try:
        model = registry.get_model(req.model_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found in registry.")

    df = pd.DataFrame(req.records)
    start_time = time.perf_counter()

    predictions: List[Any] = []
    probabilities: Any = None
    anomaly_scores: Any = None

    if model.task_type == TaskType.FORECASTING:
        forecaster: ForecastingEngine = model  # type: ignore
        preds = forecaster.forecast_horizon(horizon=req.horizon or len(req.records))
        predictions = [float(p) for p in preds]
    elif model.task_type == TaskType.ANOMALY_DETECTION:
        anomaly_engine: AnomalyEngine = model  # type: ignore
        preds = anomaly_engine.predict(df)
        scores = anomaly_engine.score_samples(df)
        predictions = [int(p) for p in preds]
        anomaly_scores = [float(s) for s in scores]
    else:
        preds = model.predict(df)
        predictions = [
            (
                float(p)
                if isinstance(p, (np.floating, float))
                else (int(p) if isinstance(p, (np.integer, int)) else str(p))
            )
            for p in preds
        ]
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(df)
                if probs is not None:
                    probabilities = probs.tolist()
            except Exception:
                probabilities = None

    latency_ms = (time.perf_counter() - start_time) * 1000

    return InferenceResponse(
        model_id=req.model_id,
        task_type=model.task_type,
        predictions=predictions,
        probabilities=probabilities,
        anomaly_scores=anomaly_scores,
        latency_ms=round(latency_ms, 2),
        num_records=len(predictions),
    )


@router.get(
    "/models",
    response_model=List[ModelInfoResponse],
    status_code=status.HTTP_200_OK,
    summary="List all registered models in the serving cache",
)
async def list_models(current_user: User = Depends(get_current_user)) -> Any:
    return registry.list_models()
