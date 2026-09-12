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