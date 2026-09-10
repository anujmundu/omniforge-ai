from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.config import get_settings
from apps.api.core.database import get_db_session
from apps.api.core.dependencies import get_current_user
from apps.api.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
    verify_password,
)
from apps.api.models.user import User, UserRole
from apps.api.schemas.auth import AuthResponse, LoginRequest, RefreshTokenRequest, Token
from apps.api.schemas.user import UserCreate, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication & Access Control"])
settings = get_settings()


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new platform user",
)
async def register_user(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    # Check if email is already taken
    stmt = select(User).where(User.email == user_in.email.lower())
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    # First user automatically becomes ADMIN if none exist
    user_count_stmt = select(User)
    user_count = len((await db.execute(user_count_stmt)).scalars().all())
    assigned_role = UserRole.ADMIN if user_count == 0 else (user_in.role or UserRole.DATA_SCIENTIST)

    db_user = User(
        email=user_in.email.lower(),
        hashed_password=get_password_hash(user_in.password),
        full_name=user_in.full_name,
        role=assigned_role,
        is_active=True,
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)

    access_token = create_access_token(subject=db_user.id, role=db_user.role.value)
    refresh_token = create_refresh_token(subject=db_user.id, role=db_user.role.value)

    return AuthResponse(
        user=UserResponse.model_validate(db_user),
        tokens=Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=refresh_token,
        ),
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user and issue JWT tokens",
)
async def login(
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    stmt = select(User).where(User.email == login_data.email.lower())
    user = (await db.execute(stmt)).scalar_one_or_none()

    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account.",
        )

    access_token = create_access_token(subject=user.id, role=user.role.value)
    refresh_token = create_refresh_token(subject=user.id, role=user.role.value)

    return AuthResponse(
        user=UserResponse.model_validate(user),
        tokens=Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=refresh_token,
        ),
    )


@router.post(
    "/token",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="OAuth2 compatible token endpoint for Swagger docs",
    include_in_schema=True,
)
async def oauth2_token_login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    stmt = select(User).where(User.email == form_data.username.lower())
    user = (await db.execute(stmt)).scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(subject=user.id, role=user.role.value)
    refresh_token = create_refresh_token(subject=user.id, role=user.role.value)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        refresh_token=refresh_token,
    )


@router.post(
    "/refresh",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="Renew access token with valid refresh token",
)
async def refresh_access_token(
    refresh_in: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db_session),
) -> Any:
    try:
        payload = decode_token(refresh_in.refresh_token)
        user_id = payload.get("sub")
        token_type = payload.get("type")
        if not user_id or token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token.",
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    stmt = select(User).where(User.id == user_id)
    user = (await db.execute(stmt)).scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer active.",
        )

    new_access_token = create_access_token(subject=user.id, role=user.role.value)
    new_refresh_token = create_refresh_token(subject=user.id, role=user.role.value)

    return Token(
        access_token=new_access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        refresh_token=new_refresh_token,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current authenticated user identity",
)
async def get_me(current_user: User = Depends(get_current_user)) -> Any:
    return current_user
