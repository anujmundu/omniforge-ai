"""Thread-safe Token Bucket Rate Limiter with tenant and tier capacity management.

Supports smooth sliding token refills, burst quotas, reset counters, and HTTP headers.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Tuple

from security.base import RateLimitStatus


class TokenBucketRateLimiter:
    """Enterprise In-Memory Token-Bucket Rate Limiter with thread-safe locking."""

    # Tier specifications: (capacity, refill_rate_per_sec)
    TIER_CONFIGS: Dict[str, Tuple[int, float]] = {
        "free": (10, 1.0),  # 60 req/min, burst 10
        "pro": (50, 10.0),  # 600 req/min, burst 50
        "enterprise": (200, 50.0),  # 3000 req/min, burst 200
    }

    def __init__(self):
        self._buckets: Dict[str, float] = {}
        self._last_updated: Dict[str, float] = {}
        self._lock = threading.Lock()

    def check_and_consume(self, client_id: str, tier: str = "free", tokens: int = 1) -> RateLimitStatus:
        """Check rate limit for client and consume tokens if within capacity."""
        with self._lock:
            now = time.monotonic()
            capacity, refill_rate = self.TIER_CONFIGS.get(tier, self.TIER_CONFIGS["free"])

            current_tokens = self._buckets.get(client_id, float(capacity))
            last_time = self._last_updated.get(client_id, now)

            # Refill tokens based on elapsed monotonic time
            elapsed = now - last_time
            current_tokens = min(float(capacity), current_tokens + (elapsed * refill_rate))
            self._last_updated[client_id] = now

            if current_tokens >= tokens:
                current_tokens -= tokens
                self._buckets[client_id] = current_tokens
                is_limited = False
            else:
                self._buckets[client_id] = current_tokens
                is_limited = True