"""Pydantic v2 API request/response schemas for OmniForge Security Endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from security.base import (
    DefenseAction,
    PIIFinding,
    RateLimitStatus,
    RedTeamAuditReport,
    ThreatCategory,
    ThreatSeverity,
)


class PromptScanRequest(BaseModel):
    """Request schema for scanning an adversarial input prompt."""

    prompt: str = Field(..., min_length=1, max_length=50000, description="Prompt text to analyze")
    tenant_id: Optional[str] = Field("default", description="Tenant or workspace ID")


class PromptScanResponse(BaseModel):
    """Response schema containing prompt security analysis."""

    is_safe: bool = Field(..., description="True if prompt is allowed")
    action: DefenseAction = Field(..., description="Defensive action taken")
    severity: ThreatSeverity = Field(..., description="Assessed threat severity")
    threat_score: float = Field(..., ge=0.0, le=1.0, description="Threat risk score")
    detected_threats: List[ThreatCategory] = Field(default_factory=list)
    matched_rules: List[str] = Field(default_factory=list)
    sanitized_prompt: Optional[str] = Field(None)
    scan_time_ms: float = Field(..., ge=0.0)


class PIIRedactRequest(BaseModel):
    """Request schema for PII/secret scanning and redaction."""

    text: str = Field(..., min_length=1, max_length=100000, description="Input string to inspect and sanitize")


class PIIRedactResponse(BaseModel):
    """Response schema containing redacted text and detected findings."""

    contains_pii: bool = Field(..., description="True if any PII or secrets were found")
    findings_count: int = Field(0, ge=0)
    findings: List[PIIFinding] = Field(default_factory=list)
    redacted_text: str = Field(...)


class RateLimitCheckResponse(BaseModel):
    """Response schema for rate limit telemetry."""

    status: RateLimitStatus


class RateLimitResetRequest(BaseModel):
    """Request schema for resetting client rate limit bucket."""

    client_id: str = Field(..., min_length=1, description="Client ID or IP to reset")


class RedTeamAuditRequest(BaseModel):
    """Request schema for triggering an automated red-team audit."""

    include_payloads: bool = Field(True, description="Whether to include attack payloads in report")


class SecurityAuditLogsResponse(BaseModel):
    """Response schema listing security telemetry events."""

    events_count: int = Field(..., ge=0)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    report: Optional[RedTeamAuditReport] = Field(None)
