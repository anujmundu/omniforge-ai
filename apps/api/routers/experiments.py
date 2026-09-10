from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.database import get_db_session
from apps.api.core.dependencies import get_current_user, require_roles
from apps.api.models.artifact import Artifact
from apps.api.models.experiment import Experiment, ExperimentDomain, ExperimentStatus
from apps.api.models.project import Project
from apps.api.models.user import User, UserRole
from apps.api.schemas.artifact import ArtifactCreate, ArtifactResponse
from apps.api.schemas.experiment import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentUpdate,
)

router = APIRouter(prefix="/experiments", tags=["Experiments & Model Tracking"])


@router.post(
    "",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create and start a new experiment run",
)
async def create_experiment(
    experiment_in: ExperimentCreate,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.ML_ENGINEER, UserRole.DATA_SCIENTIST)),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    project_stmt = select(Project).where(Project.id == experiment_in.project_id)
    project = (await db.execute(project_stmt)).scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found.",
        )

    experiment = Experiment(
        project_id=experiment_in.project_id,
        name=experiment_in.name,
        domain=experiment_in.domain,
        status=ExperimentStatus.RUNNING,
        model_name=experiment_in.model_name,
        parameters=experiment_in.parameters,
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    return experiment


@router.get(
    "",
    response_model=List[ExperimentResponse],
    status_code=status.HTTP_200_OK,
    summary="List experiment runs with filters",
)
async def list_experiments(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    domain: Optional[ExperimentDomain] = Query(None, description="Filter by domain"),
    status: Optional[ExperimentStatus] = Query(None, description="Filter by run status"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    stmt = select(Experiment).order_by(Experiment.created_at.desc())
    if project_id:
        stmt = stmt.where(Experiment.project_id == project_id)
    if domain:
        stmt = stmt.where(Experiment.domain == domain)
    if status:
        stmt = stmt.where(Experiment.status == status)

    result = await db.execute(stmt)
    return result.scalars().all()


@router.get(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get details, metrics, and parameters for an experiment run",
)
async def get_experiment(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    stmt = select(Experiment).where(Experiment.id == experiment_id)
    experiment = (await db.execute(stmt)).scalar_one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found.",
        )
    return experiment


@router.patch(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Update experiment status and record evaluation metrics",
)
async def update_experiment(
    experiment_id: str,
    update_in: ExperimentUpdate,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.ML_ENGINEER, UserRole.DATA_SCIENTIST)),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    stmt = select(Experiment).where(Experiment.id == experiment_id)
    experiment = (await db.execute(stmt)).scalar_one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found.",
        )

    if update_in.status is not None:
        experiment.status = update_in.status
    if update_in.metrics is not None:
        # Merge metrics if already present
        if experiment.metrics:
            merged_metrics = dict(experiment.metrics)
            merged_metrics.update(update_in.metrics)
            experiment.metrics = merged_metrics
        else:
            experiment.metrics = update_in.metrics
    if update_in.parameters is not None:
        experiment.parameters = update_in.parameters
    if update_in.duration_seconds is not None:
        experiment.duration_seconds = update_in.duration_seconds
    if update_in.error_message is not None:
        experiment.error_message = update_in.error_message

    await db.commit()
    await db.refresh(experiment)
    return experiment


@router.post(
    "/{experiment_id}/artifacts",
    response_model=ArtifactResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register an output artifact for an experiment run",
)
async def register_artifact(
    experiment_id: str,
    artifact_in: ArtifactCreate,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.ML_ENGINEER, UserRole.DATA_SCIENTIST)),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    stmt = select(Experiment).where(Experiment.id == experiment_id)
    experiment = (await db.execute(stmt)).scalar_one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found.",
        )

    artifact = Artifact(
        experiment_id=experiment_id,
        name=artifact_in.name,
        artifact_type=artifact_in.artifact_type,
        uri=artifact_in.uri,
        size_bytes=artifact_in.size_bytes,
        checksum=artifact_in.checksum,
    )
    db.add(artifact)
    await db.commit()
    await db.refresh(artifact)
    return artifact


@router.get(
    "/{experiment_id}/artifacts",
    response_model=List[ArtifactResponse],
    status_code=status.HTTP_200_OK,
    summary="List all artifacts for an experiment run",
)
async def list_artifacts(
    experiment_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    stmt = select(Artifact).where(Artifact.experiment_id == experiment_id).order_by(Artifact.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()
