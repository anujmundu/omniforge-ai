import enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from apps.api.models.experiment import Experiment


class ArtifactType(str, enum.Enum):
    MODEL_WEIGHTS = "MODEL_WEIGHTS"
    ONNX_MODEL = "ONNX_MODEL"
    METRICS_PLOT = "METRICS_PLOT"
    CONFUSION_MATRIX = "CONFUSION_MATRIX"
    EMBEDDINGS = "EMBEDDINGS"
    REPORT = "REPORT"


class Artifact(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Artifact produced by an experiment or pipeline run."""

    __tablename__ = "artifacts"

    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    artifact_type: Mapped[ArtifactType] = mapped_column(
        Enum(ArtifactType, name="artifact_type_enum", native_enum=False),
        default=ArtifactType.MODEL_WEIGHTS,
        nullable=False,
    )
    uri: Mapped[str] = mapped_column(String(500), nullable=False)
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    checksum: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Relationships
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="artifacts")
