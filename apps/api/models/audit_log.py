from typing import TYPE_CHECKING, Any, Dict, Optional

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.api.core.database import Base
from apps.api.models.base import TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from apps.api.models.user import User


class AuditLog(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Platform audit log tracking security, inference, and operational events."""

    __tablename__ = "audit_logs"

    user_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")
