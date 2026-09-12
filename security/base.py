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
