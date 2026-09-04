from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field


class ServiceHealth(BaseModel):
    status: str = Field(...)
    latency_ms: Optional[float] = Field(None)
    details: Optional[Dict[str, str]] = None


class HealthResponse(BaseModel):
    status: str = Field(...)
    version: str = Field(...)
    environment: str = Field(...)
    timestamp: datetime = Field(...)
    services: Dict[str, ServiceHealth]
