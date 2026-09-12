"""Sensitive PII, credential, and private key redaction and masking engine.

Implements Luhn-validated credit card detection, SSN matching, email / phone masking,
and pattern matching for AWS / OpenAI keys, JWT tokens, and cryptographic keys.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from security.base import PIIFinding, PIIScanResult, PIIType


class PIIRedactor:
    """Enterprise PII and Secret Redactor with Luhn validation for payment cards."""

    PATTERNS: List[Tuple[PIIType, re.Pattern, str]] = [
        # Social Security Number (US)
        (
            PIIType.SSN,
            re.compile(r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b"),
            "[REDACTED_SSN]",
        ),
        # Email Addresses
        (
            PIIType.EMAIL,
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
            "[REDACTED_EMAIL]",
        ),
        # International & US Phone Numbers
        (
            PIIType.PHONE,
            re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            "[REDACTED_PHONE]",
        ),
        # AWS Access Key ID
        (
            PIIType.API_KEY,
            re.compile(r"\b(AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b"),
            "[REDACTED_AWS_KEY]",
        ),
        # OpenAI / Anthropic API Keys
        (
            PIIType.API_KEY,
            re.compile(r"\b(sk-[a-zA-Z0-9_-]{20,64}|anthropic-api-key-[a-zA-Z0-9_-]{20,64})\b"),
            "[REDACTED_API_KEY]",
        ),
        # JWT Bearer Tokens
        (
            PIIType.JWT_TOKEN,
            re.compile(r"\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b"),
            "[REDACTED_JWT]",
        ),
        # RSA / EC / OpenSSH Private Keys
        (
            PIIType.PRIVATE_KEY,
            re.compile(
                r"-----BEGIN (RSA|EC|DSA|OPENSSH|ENCRYPTED)? PRIVATE KEY-----[\s\S]+?-----END \1 PRIVATE KEY-----"
            ),
            "[REDACTED_PRIVATE_KEY]",
        ),
    ]

    # Regex candidate for potential credit cards (13 to 19 digits)
    CREDIT_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){13,19}\b")

    @staticmethod
    def _luhn_check(card_number_str: str) -> bool:
        """Validate credit card number using Luhn checksum algorithm."""
        digits = [int(c) for c in card_number_str if c.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False

        checksum = 0
        reverse_digits = digits[::-1]
        for idx, digit in enumerate(reverse_digits):
            if idx % 2 == 1:
                doubled = digit * 2
                checksum += doubled - 9 if doubled > 9 else doubled
            else:
                checksum += digit

        return checksum % 10 == 0

    def scan_and_redact(self, text: str) -> PIIScanResult:
        """Scan input text for sensitive PII/secrets, redact them, and return findings."""
        if not text:
            return PIIScanResult(
                contains_pii=False,
                findings_count=0,
                findings=[],
                redacted_text="",
                redaction_map={},
            )