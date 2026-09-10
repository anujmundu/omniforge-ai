import enum
from typing import TYPE_CHECKING, Any, Dict, Optional

from sqlalchemy import JSON, BigInteger, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from apps.api.models.project import Project


class DatasetFormat(str, enum.Enum):
    CSV = "CSV"
    PARQUET = "PARQUET"
    JSON = "JSON"
    IMAGE_DIRECTORY = "IMAGE_DIRECTORY"
    AUDIO_DIRECTORY = "AUDIO_DIRECTORY"
    TEXT_CORPUS = "TEXT_CORPUS"


class Dataset(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Dataset registration and metadata tracking."""

    __tablename__ = "datasets"

    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(20), default="1.0.0", nullable=False)
    file_format: Mapped[DatasetFormat] = mapped_column(
        Enum(DatasetFormat, name="dataset_format_enum", native_enum=False),
        default=DatasetFormat.CSV,
        nullable=False,
    )
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    row_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    checksum_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    schema_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="datasets")
