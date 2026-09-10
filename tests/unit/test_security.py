from datetime import timedelta

import pytest

from apps.api.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
    verify_password,
)


def test_password_hashing_and_verification():
    raw_password = "SuperSecretPassword123!"
    hashed = get_password_hash(raw_password)

    assert hashed != raw_password
    assert verify_password(raw_password, hashed) is True
    assert verify_password("WrongPassword123!", hashed) is False


def test_access_token_encoding_and_decoding():
    user_id = "test-user-uuid-123"
    role = "ADMIN"
    token = create_access_token(subject=user_id, role=role, expires_delta=timedelta(minutes=15))

    payload = decode_token(token)
    assert payload["sub"] == user_id
    assert payload["role"] == role
    assert payload["type"] == "access"
    assert "exp" in payload


def test_refresh_token_encoding_and_decoding():
    user_id = "test-user-uuid-456"
    role = "DATA_SCIENTIST"
    token = create_refresh_token(subject=user_id, role=role, expires_delta=timedelta(days=1))

    payload = decode_token(token)
    assert payload["sub"] == user_id
    assert payload["role"] == role
    assert payload["type"] == "refresh"


def test_invalid_token_decoding():
    with pytest.raises(ValueError, match="Invalid or expired token"):
        decode_token("this.is.a.completely.invalid.token")
