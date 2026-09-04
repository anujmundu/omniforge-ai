import os
from functools import lru_cache
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Core Application
    ENVIRONMENT: str = Field(default="development", description="Current environment mode")
    PROJECT_NAME: str = Field(default="OmniForge Multimodal Platform", description="Platform name")
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = Field(default=False)
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Security & JWT
    SECRET_KEY: str = Field(
        default="omniforge-insecure-development-secret-key-change-in-production-32bytes",
        description="JWT signature secret key"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./omniforge.db",
        description="Async database connection string"
    )

    # Redis & Storage
    REDIS_URL: str = "redis://localhost:6379/0"
    ARTIFACT_STORAGE_PATH: str = "./storage/artifacts"
    DATASET_STORAGE_PATH: str = "./storage/datasets"

    # CORS
    CORS_ORIGINS: Union[List[str], str] = ["http://localhost:3000", "http://localhost:8000"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)


@lru_cache()
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    return Settings()
