import enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import JSON, Enum, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from apps.api.models.artifact import Artifact
    from apps.api.models.project import Project


class ExperimentDomain(str, enum.Enum):
    CLASSICAL_ML = "CLASSICAL_ML"
    COMPUTER_VISION = "COMPUTER_VISION"
    NLP = "NLP"
    GENAI_RAG = "GENAI_RAG"
    AGENTIC = "AGENTIC"


class ExperimentStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Experiment(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Experiment and model training run tracking entity."""

    __tablename__ = "experiments"

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    domain: Mapped[ExperimentDomain] = mapped_column(
        Enum(ExperimentDomain, name="experiment_domain_enum", native_enum=False),
        default=ExperimentDomain.CLASSICAL_ML,
        nullable=False,
    )
    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(ExperimentStatus, name="experiment_status_enum", native_enum=False),
        default=ExperimentStatus.PENDING,
        nullable=False,
    )
    model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="experiments")
    artifacts: Mapped[List["Artifact"]] = relationship(
        "Artifact", back_populates="experiment", cascade="all, delete-orphan"
    )
