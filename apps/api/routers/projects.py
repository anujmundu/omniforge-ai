import re
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from apps.api.core.database import get_db_session
from apps.api.core.dependencies import get_current_user, require_roles
from apps.api.models.project import Project
from apps.api.models.user import User, UserRole
from apps.api.schemas.project import ProjectCreate, ProjectResponse, ProjectUpdate

router = APIRouter(prefix="/projects", tags=["Project Workspaces"])


def generate_slug(name: str) -> str:
    """Generate a clean URL slug from project name."""
    clean = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"[-\s]+", "-", clean)


@router.post(
    "",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new ML project workspace",
)
async def create_project(
    project_in: ProjectCreate,
    current_user: User = Depends(
        require_roles(UserRole.ADMIN, UserRole.ML_ENGINEER, UserRole.DATA_SCIENTIST)
    ),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    base_slug = project_in.slug or generate_slug(project_in.name)

    # Ensure unique slug
    existing_stmt = select(Project).where(Project.slug == base_slug)
    if (await db.execute(existing_stmt)).scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Project with slug '{base_slug}' already exists. Please specify a unique slug.",
        )

    project = Project(
        name=project_in.name,
        slug=base_slug,
        description=project_in.description,
        owner_id=current_user.id,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.get(
    "",
    response_model=List[ProjectResponse],
    status_code=status.HTTP_200_OK,
    summary="List all accessible projects",
)
async def list_projects(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    if current_user.role == UserRole.ADMIN:
        stmt = select(Project).order_by(Project.created_at.desc())
    else:
        stmt = select(Project).where(Project.owner_id == current_user.id).order_by(Project.created_at.desc())

    result = await db.execute(stmt)
    return result.scalars().all()


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve project workspace by ID or slug",
)
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    stmt = select(Project).where((Project.id == project_id) | (Project.slug == project_id))
    project = (await db.execute(stmt)).scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found.",
        )

    if current_user.role != UserRole.ADMIN and project.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this project.",
        )

    return project


@router.patch(
    "/{project_id}",
    response_model=ProjectResponse,
    status_code=status.HTTP_200_OK,
    summary="Update project metadata",
)
async def update_project(
    project_id: str,
    project_update: ProjectUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found.",
        )

    if current_user.role != UserRole.ADMIN and project.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to modify this project.",
        )

    if project_update.name is not None:
        project.name = project_update.name
    if project_update.description is not None:
        project.description = project_update.description

    await db.commit()
    await db.refresh(project)
    return project


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a project workspace",
)
async def delete_project(
    project_id: str,
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.ML_ENGINEER)),
    db: AsyncSession = Depends(get_db_session),
) -> None:
    stmt = select(Project).where(Project.id == project_id)
    project = (await db.execute(stmt)).scalar_one_or_none()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found.",
        )

    if current_user.role != UserRole.ADMIN and project.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this project.",
        )

    await db.delete(project)
    await db.commit()
