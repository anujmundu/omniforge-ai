"""Unit tests for PII and credential redactor."""

import pytest

from security.base import PIIType
from security.pii_redactor import PIIRedactor


@pytest.fixture
def redactor():
    return PIIRedactor()


def test_ssn_redaction(redactor):
    text = "User SSN is 123-45-6789 and needs processing."
    res = redactor.scan_and_redact(text)
    assert res.contains_pii is True
    assert res.findings_count == 1
    assert res.findings[0].pii_type == PIIType.SSN
    assert "[REDACTED_SSN]" in res.redacted_text
    assert "123-45-6789" not in res.redacted_text


def test_email_and_phone_redaction(redactor):
    text = "Contact alice@example.com or call +1 (555) 234-5678."
    res = redactor.scan_and_redact(text)
    assert res.contains_pii is True
    assert res.findings_count == 2
    assert "[REDACTED_EMAIL]" in res.redacted_text
    assert "[REDACTED_PHONE]" in res.redacted_text


def test_luhn_valid_credit_card_redaction(redactor):
    # Valid Visa card candidate with valid Luhn checksum: 4532015112830366
    text = "Payment details: 4532 0151 1283 0366"
    res = redactor.scan_and_redact(text)
    assert res.contains_pii is True
    assert res.findings_count == 1
    assert res.findings[0].pii_type == PIIType.CREDIT_CARD
    assert "[REDACTED_CREDIT_CARD]" in res.redacted_text


def test_invalid_credit_card_not_redacted(redactor):
    # Invalid card number (fails Luhn)
    text = "Order tracking sequence: 1234 5678 9012 3456"
    res = redactor.scan_and_redact(text)
    assert "[REDACTED_CREDIT_CARD]" not in res.redacted_text


def test_aws_api_key_redaction(redactor):
    text = "AWS config: AKIAIOSFODNN7EXAMPLE"
    res = redactor.scan_and_redact(text)
    assert res.contains_pii is True
    assert res.findings[0].pii_type == PIIType.API_KEY
    assert "[REDACTED_AWS_KEY]" in res.redacted_text


def test_jwt_bearer_token_redaction(redactor):
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.do_not_leak_signature_payload"
    text = f"Authorization: Bearer {jwt}"
    res = redactor.scan_and_redact(text)
    assert res.contains_pii is True
    assert "[REDACTED_JWT]" in res.redacted_text
