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
        normalized = unicodedata.normalize("NFKC", text)
        # Strip zero-width characters (ZWSP, ZWNJ, ZWJ, etc.)
        cleaned = re.sub(r"[\u200B-\u200D\uFEFF\u2060\u180E\u00AD]", "", normalized)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _inspect_obfuscation(self, text: str) -> List[Tuple[str, str, float, ThreatCategory]]:
        """Detect and decode Base64 / Rot13 obfuscated attack payloads."""
        findings: List[Tuple[str, str, float, ThreatCategory]] = []

        # 1. Inspect potential Base64 payloads
        for match in self.BASE64_PATTERN.finditer(text):
            token = match.group(1).strip()
            try:
                decoded = base64.b64decode(token).decode("utf-8", errors="ignore")
                if len(decoded) > 6:
                    for rule_id, pattern, weight, cat in self.RULES:
                        if pattern.search(decoded):
                            findings.append((f"obfuscated_base64_{rule_id}", decoded, weight, cat))
            except Exception:
                pass

        # 2. Inspect Rot13 deciphered text
        try:
            rot13_text = codecs.decode(text, "rot_13")
            for rule_id, pattern, weight, cat in self.RULES:
                if pattern.search(rot13_text):
                    findings.append((f"obfuscated_rot13_{rule_id}", rot13_text, weight, cat))
        except Exception:
            pass

        return findings

    def scan(self, prompt: str) -> ThreatScanResult:
        """Scan an input prompt against adversarial injection and jailbreak guardrails."""
        start_time = time.perf_counter()

        if not prompt or not prompt.strip():
            return ThreatScanResult(
                is_safe=True,
                action=DefenseAction.ALLOW,
                severity=ThreatSeverity.SAFE,
                threat_score=0.0,
                detected_threats=[],
                matched_rules=[],
                sanitized_prompt=prompt,
                scan_time_ms=(time.perf_counter() - start_time) * 1000,
                details={"status": "empty_prompt"},
            )

        normalized = self._normalize_text(prompt)
        matched_rules: List[str] = []
        detected_threats: List[ThreatCategory] = []
        max_threat_score = 0.0

        # Step 1: Scan normalized text with heuristic rules
        for rule_id, pattern, weight, category in self.RULES:
            if pattern.search(normalized):
                matched_rules.append(rule_id)
                if category not in detected_threats:
                    detected_threats.append(category)
                max_threat_score = max(max_threat_score, weight)

        # Step 2: Check for obfuscated encodings
        obfuscated_findings = self._inspect_obfuscation(normalized)
        if obfuscated_findings:
            if ThreatCategory.OBFUSCATION not in detected_threats:
                detected_threats.append(ThreatCategory.OBFUSCATION)
            for rule_id, _, weight, cat in obfuscated_findings:
                matched_rules.append(rule_id)
                if cat not in detected_threats:
                    detected_threats.append(cat)
                max_threat_score = max(max_threat_score, weight)

        # Step 3: Determine Severity and Defensive Action
        if max_threat_score >= 0.85:
            severity = ThreatSeverity.CRITICAL
        elif max_threat_score >= 0.70:
            severity = ThreatSeverity.HIGH
        elif max_threat_score >= 0.40:
            severity = ThreatSeverity.MEDIUM
        elif max_threat_score > 0.0:
            severity = ThreatSeverity.LOW
        else:
            severity = ThreatSeverity.SAFE

        sanitized_prompt: Optional[str] = None
        if max_threat_score >= self.block_threshold:
            action = DefenseAction.BLOCK
            is_safe = False
        elif max_threat_score >= self.flag_threshold:
            action = DefenseAction.FLAG
            is_safe = False
        elif max_threat_score > 0.0:
            action = DefenseAction.SANITIZE
            is_safe = True
            sanitized_prompt = self._sanitize(normalized)
        else:
            action = DefenseAction.ALLOW
            is_safe = True
            sanitized_prompt = normalized

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return ThreatScanResult(
            is_safe=is_safe,
            action=action,
            severity=severity,
            threat_score=round(max_threat_score, 4),
            detected_threats=detected_threats,
            matched_rules=matched_rules,
            sanitized_prompt=sanitized_prompt,
            scan_time_ms=round(elapsed_ms, 3),
            details={
                "normalized_length": len(normalized),
                "original_length": len(prompt),
                "obfuscation_detected": len(obfuscated_findings) > 0,
            },
        )

    def _sanitize(self, text: str) -> str:
        """Strip matched dangerous delimiters and control tokens."""
        sanitized = text
        for _, pattern, _, _ in self.RULES:
            sanitized = pattern.sub("[REDACTED_ADVERSARIAL_INPUT]", sanitized)
        return sanitized


# Global Scanner Instance
prompt_scanner = PromptDefenseScanner()
