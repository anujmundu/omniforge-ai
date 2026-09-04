from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from apps.api.models.user import User
    from apps.api.models.dataset import Dataset
    from apps.api.models.experiment import Experiment


class Project(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Project workspace grouping datasets and experiments."""
    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(120), unique=True, index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="projects")
    datasets: Mapped[List["Dataset"]] = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    experiments: Mapped[List["Experiment"]] = relationship("Experiment", back_populates="project", cascade="all, delete-orphan")
