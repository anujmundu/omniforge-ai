"""Adversarial prompt injection, jailbreak, and policy evasion defense scanner.

Implements multi-layer heuristic pattern scanning, character normalization,
homoglyph resolution, zero-width stripping, and Base64/Rot13 obfuscation decoders.
"""

from __future__ import annotations

import base64
import codecs
import re
import time
import unicodedata
from typing import List, Optional, Tuple

from security.base import DefenseAction, ThreatCategory, ThreatScanResult, ThreatSeverity


class PromptDefenseScanner:
    """Enterprise-grade multi-layer adversarial prompt injection and jailbreak detector."""

    # Heuristic signature rules: (rule_id, regex, threat_weight, category)
    RULES: List[Tuple[str, re.Pattern, float, ThreatCategory]] = [
        # Direct override / System prompt overwrite
        (
            "direct_override_instruction",
            re.compile(
                r"(?i)\b(ignore|disregard|forget|override|bypass)\b.*\b(previous|prior|system|all)\b.*\b(instructions|prompts|rules|guidelines)\b"
            ),
            0.95,
            ThreatCategory.PROMPT_INJECTION,
        ),
        (
            "system_prompt_leakage",
            re.compile(
                r"(?i)\b(repeat|print|output|display|show|reveal|echo)\b.*\b(system\s+prompt|initial\s+prompt|core\s+instructions|system\s+message)\b"
            ),
            0.90,
            ThreatCategory.SYSTEM_PROMPT_LEAK,
        ),
        (
            "role_reversal_authority",
            re.compile(
                r"(?i)\b(you\s+are\s+now|from\s+now\s+on\s+you\s+are|act\s+as|pretend\s+to\s+be)\b.*\b(an\s+unrestricted|unfiltered|jailbroken|godmode|dan|stan|aim)\b"
            ),