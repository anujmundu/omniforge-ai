from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.database import get_db_session
from apps.api.core.dependencies import get_current_user, require_roles
from apps.api.models.dataset import Dataset
from apps.api.models.project import Project
from apps.api.models.user import User, UserRole
from apps.api.schemas.dataset import DatasetCreate, DatasetResponse

router = APIRouter(prefix="/datasets", tags=["Dataset Management"])


@router.post(
    "",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new dataset version under a project",
)
async def register_dataset(
    dataset_in: DatasetCreate,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.ML_ENGINEER, UserRole.DATA_SCIENTIST)),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    # Verify project exists and user has access
    project_stmt = select(Project).where(Project.id == dataset_in.project_id)
    project = (await db.execute(project_stmt)).scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target project not found.",
        )

    if current_user.role != UserRole.ADMIN and project.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to add datasets to this project.",
        )

    # Check for duplicate version in this project
    dup_stmt = select(Dataset).where(
        Dataset.project_id == dataset_in.project_id,
        Dataset.name == dataset_in.name,
        Dataset.version == dataset_in.version,
    )
    if (await db.execute(dup_stmt)).scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset '{dataset_in.name}' with version '{dataset_in.version}' already exists in this project.",
        )

    dataset = Dataset(
        project_id=dataset_in.project_id,
        name=dataset_in.name,
        version=dataset_in.version,
        file_format=dataset_in.file_format,
        storage_path=dataset_in.storage_path,
        row_count=dataset_in.row_count,
        checksum_sha256=dataset_in.checksum_sha256,
        schema_metadata=dataset_in.schema_metadata,
        description=dataset_in.description,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset


@router.get(
    "",
    response_model=List[DatasetResponse],
    status_code=status.HTTP_200_OK,
    summary="List registered datasets with optional project filtering",
)
async def list_datasets(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    stmt = select(Dataset).order_by(Dataset.created_at.desc())
    if project_id:
        stmt = stmt.where(Dataset.project_id == project_id)

    result = await db.execute(stmt)
    return result.scalars().all()


@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    status_code=status.HTTP_200_OK,
    summary="Get dataset details, schema, and checksum",
)
async def get_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    dataset = (await db.execute(stmt)).scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        )
    return dataset


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a dataset entry",
)
async def delete_dataset(
    dataset_id: str,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.ML_ENGINEER)),
    db: AsyncSession = Depends(get_db_session),
) -> None:
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    dataset = (await db.execute(stmt)).scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found.",
        )

    await db.delete(dataset)
    await db.commit()
