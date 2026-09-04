import os
import pytest
from typing import AsyncGenerator, Dict
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Force testing configuration
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key-at-least-32-characters-long"

from apps.api.core.config import get_settings
from apps.api.core.database import Base, get_db_session
from apps.api.core.security import create_access_token, get_password_hash
from apps.api.main import app
from apps.api.models.user import User, UserRole

settings = get_settings()

test_engine: AsyncEngine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    future=True,
)

TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
async def prepare_database() -> AsyncGenerator[None, None]:
    """Create in-memory database tables for each test and drop them afterwards."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide isolated async DB session."""
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client with overridden DB session dependency."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
async def admin_user(db_session: AsyncSession) -> User:
    """Create test admin user."""
    user = User(
        email="admin@aiforge.dev",
        hashed_password=get_password_hash("AdminPass123!"),
        full_name="Platform Administrator",
        role=UserRole.ADMIN,
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def ml_engineer_user(db_session: AsyncSession) -> User:
    """Create test ML engineer user."""
    user = User(
        email="engineer@aiforge.dev",
        hashed_password=get_password_hash("EngineerPass123!"),
        full_name="ML Systems Engineer",
        role=UserRole.ML_ENGINEER,
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def admin_headers(admin_user: User) -> Dict[str, str]:
    """Bearer token headers for admin."""
    token = create_access_token(subject=admin_user.id, role=admin_user.role.value)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def engineer_headers(ml_engineer_user: User) -> Dict[str, str]:
    """Bearer token headers for ML engineer."""
    token = create_access_token(subject=ml_engineer_user.id, role=ml_engineer_user.role.value)
    return {"Authorization": f"Bearer {token}"}
