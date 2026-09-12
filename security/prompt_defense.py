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
            0.95,
            ThreatCategory.JAILBREAK,
        ),
        (
            "dan_jailbreak_signature",
            re.compile(
                r"(?i)\b(DAN\s+Mode|Do\s+Anything\s+Now|Developer\s+Mode\s+v2|Maximum\s+Bypass|always\s+comply\s+without\s+filter)\b"
            ),
            0.98,
            ThreatCategory.JAILBREAK,
        ),
        (
            "context_delimiter_injection",
            re.compile(
                r"(?i)(<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|Assistant:\s*|Human:\s*|System:\s*|\bBEGIN\s+SYSTEM\s+PROMPT\b|\bEND\s+SYSTEM\s+PROMPT\b)"
            ),
            0.85,
            ThreatCategory.PROMPT_INJECTION,
        ),
        (
            "hypothetical_evil_bypass",
            re.compile(
                r"(?i)\b(in\s+a\s+fictional\s+world|for\s+educational\s+purposes\s+only|hypothetically\s+speaking|purely\s+academic)\b.*\b(how\s+to\s+hack|exploit|build\s+a\s+bomb|synthesize|steal)\b"
            ),
            0.80,
            ThreatCategory.JAILBREAK,
        ),
        (
            "rbac_and_privilege_escalation",
            re.compile(
                r"(?i)\b(sudo\s+su|chmod\s+777|elevated\s+to\s+role=admin|bypass_all_auth=true|grant\s+superuser)\b"
            ),
            0.90,
            ThreatCategory.UNAUTHORIZED_ACCESS,
        ),
        (
            "output_format_hijack",
            re.compile(
                r"(?i)\b(start\s+your\s+response\s+with|respond\s+only\s+with|always\s+prefix\s+your\s+answer)\b\s*[\"']{1,3}(yes|sure|agreed|i\s+can\s+do\s+that)[\"']{1,3}"
            ),
            0.65,
            ThreatCategory.PROMPT_INJECTION,
        ),
    ]

    # Suspicious Base64 matching pattern
    BASE64_PATTERN = re.compile(r"(?:^|[\s:=,])([A-Za-z0-9+/]{12,}={0,2})(?:[\s;,\.]|$)")

    def __init__(self, block_threshold: float = 0.70, flag_threshold: float = 0.40):
        self.block_threshold = block_threshold
        self.flag_threshold = flag_threshold

    def _normalize_text(self, text: str) -> str:
        """Remove zero-width spaces, invisible characters, and normalize unicode."""
        # Convert fullwidth/homoglyph unicode into normalized ASCII (NFKC)