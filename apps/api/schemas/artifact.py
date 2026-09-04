from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from apps.api.models.artifact import ArtifactType


class ArtifactBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=150)
    artifact_type: ArtifactType = Field(default=ArtifactType.MODEL_WEIGHTS)
    uri: str = Field(...)
    size_bytes: Optional[int] = Field(None, ge=0)
    checksum: Optional[str] = None


class ArtifactCreate(ArtifactBase):
    experiment_id: Optional[str] = None


class ArtifactResponse(ArtifactBase):
    id: str
    experiment_id: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
