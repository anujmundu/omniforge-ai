"""Core domain enums, data contracts, and threat models for OmniForge Security.

Defines state models for adversarial prompt scanning, PII/secret redaction,
token-bucket rate limiting, and automated red-team vulnerability auditing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ThreatCategory(str, Enum):
    """Categorization of adversarial threats and policy violations."""

    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    DATA_EXFILTRATION = "data_exfiltration"
    OBFUSCATION = "obfuscation"
    PII_LEAK = "pii_leak"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class ThreatSeverity(str, Enum):
    """Threat risk severity levels."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DefenseAction(str, Enum):
    """Defensive mitigation action taken by guardrails."""

    ALLOW = "allow"
    FLAG = "flag"
    SANITIZE = "sanitize"
    BLOCK = "block"


class PIIType(str, Enum):
    """Categories of sensitive personal and credential data."""

    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    PRIVATE_KEY = "private_key"


class RedTeamAttackType(str, Enum):
    """Vulnerability taxonomy for adversarial red-teaming vectors."""

    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK_DAN = "jailbreak_dan"
    ROLE_REVERSAL = "role_reversal"
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"
    BASE64_OBFUSCATION = "base64_obfuscation"
    ROT13_OBFUSCATION = "rot13_obfuscation"
    UNICODE_HOMOGLYPH = "unicode_homoglyph"
    PII_EXFILTRATION = "pii_exfiltration"
    RECURSIVE_EXPANSION = "recursive_expansion"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class ThreatScanResult(BaseModel):
    """Result of scanning an input prompt for adversarial threats."""

    is_safe: bool = Field(..., description="True if prompt satisfies security policy")
    action: DefenseAction = Field(..., description="Mitigation action taken")
    severity: ThreatSeverity = Field(..., description="Assessed threat severity")
    threat_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score of threat")
    detected_threats: List[ThreatCategory] = Field(default_factory=list, description="Categories identified")
    matched_rules: List[str] = Field(default_factory=list, description="Rule identifiers triggered")
    sanitized_prompt: Optional[str] = Field(None, description="Sanitized prompt text if action was sanitize")
    scan_time_ms: float = Field(..., ge=0.0, description="Latency of defense scanning pipeline")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic scanning metadata")


class PIIFinding(BaseModel):
    """Individual PII entity or secret discovered in text."""

    pii_type: PIIType = Field(..., description="Type of PII or secret entity")
    value_preview: str = Field(..., description="Masked snippet preview")
    start_pos: int = Field(..., ge=0, description="Start character index")
    end_pos: int = Field(..., ge=0, description="End character index")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Detection confidence")


class PIIScanResult(BaseModel):
    """Result of PII and credential scanning and redaction."""

    contains_pii: bool = Field(..., description="True if any sensitive entity found")
    findings_count: int = Field(0, ge=0, description="Total number of entities detected")
    findings: List[PIIFinding] = Field(default_factory=list, description="List of detected PII findings")
    redacted_text: str = Field(..., description="Safe redacted string replacing secrets with masks")
    redaction_map: Dict[str, str] = Field(default_factory=dict, description="Mapping of masks to redacted types")


class RateLimitStatus(BaseModel):
    """Rate limit quota status for a client identifier."""

    client_id: str = Field(..., description="Client identifier or API token")
    tier: str = Field("free", description="Client subscription tier")
    limit: int = Field(..., description="Maximum allowed requests per window")
    remaining: int = Field(..., description="Remaining requests available in current window")
    reset_seconds: int = Field(..., description="Seconds until bucket is fully refilled")
    is_limited: bool = Field(..., description="True if client exceeded request capacity")
