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
