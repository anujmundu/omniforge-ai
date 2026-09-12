"""Unit tests for TokenBucketRateLimiter."""

import time

from security.rate_limiter import TokenBucketRateLimiter


def test_rate_limiter_allows_under_quota():
    limiter = TokenBucketRateLimiter()
    client_id = "test_client_allow"

    status = limiter.check_and_consume(client_id, tier="free", tokens=1)
    assert status.is_limited is False
    assert status.remaining == 9
    assert status.limit == 10


def test_rate_limiter_exhaustion_blocks():
    limiter = TokenBucketRateLimiter()
    client_id = "test_client_burst"

    # Free tier capacity is 10
    for _ in range(10):
        status = limiter.check_and_consume(client_id, tier="free", tokens=1)
        assert status.is_limited is False

    # 11th request should exceed limit
    over_status = limiter.check_and_consume(client_id, tier="free", tokens=1)
    assert over_status.is_limited is True
    assert over_status.remaining == 0


def test_rate_limiter_refills_over_time():
    limiter = TokenBucketRateLimiter()
    client_id = "test_client_refill"

    # Consume all 10 tokens
    for _ in range(10):
        limiter.check_and_consume(client_id, tier="free", tokens=1)

    assert limiter.peek(client_id, tier="free").is_limited is True

    # Sleep 1.1 seconds (refills 1+ tokens at 1 token/sec)
    time.sleep(1.1)

    status = limiter.check_and_consume(client_id, tier="free", tokens=1)
    assert status.is_limited is False


def test_rate_limiter_reset():
    limiter = TokenBucketRateLimiter()
    client_id = "test_client_reset"

    for _ in range(10):
        limiter.check_and_consume(client_id, tier="free", tokens=1)

    assert limiter.peek(client_id, tier="free").is_limited is True

    limiter.reset_client(client_id)
    status = limiter.check_and_consume(client_id, tier="free", tokens=1)
    assert status.is_limited is False
    assert status.remaining == 9
