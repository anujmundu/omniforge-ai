from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field
from apps.api.models.dataset import DatasetFormat


class DatasetBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    version: str = Field(default="1.0.0")
    file_format: DatasetFormat = Field(default=DatasetFormat.CSV)
    description: Optional[str] = None


class DatasetCreate(DatasetBase):
    project_id: str
    storage_path: str = Field(...)
    row_count: Optional[int] = Field(None, ge=0)
    checksum_sha256: Optional[str] = Field(None, min_length=64, max_length=64)
    schema_metadata: Optional[Dict[str, Any]] = None


class DatasetResponse(DatasetBase):
    id: str
    project_id: str
    storage_path: str
    row_count: Optional[int]
    checksum_sha256: Optional[str]
    schema_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
