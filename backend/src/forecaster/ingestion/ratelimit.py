"""Rate limiting and circuit breaking for external providers.

Free API tiers are unforgiving -- Alpha Vantage allows 25 calls *per day*, and
yfinance will start serving empty frames if you hammer it. Two mechanisms:

* :class:`TokenBucket` -- smooths request rate, awaits rather than erroring.
* :class:`CircuitBreaker` -- after N consecutive failures, stop calling a dead
  provider for a cooldown instead of burning the whole retry budget on it.

In-memory by default; pass a Redis client to share limits across processes.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum

from forecaster.exceptions import ProviderRateLimitError
from forecaster.logging import get_logger

log = get_logger(__name__)


class TokenBucket:
    """Async token bucket.

    Tokens refill continuously at ``rate`` per second up to ``capacity``.
    :meth:`acquire` waits for a token; :meth:`try_acquire` never blocks.
    """

    def __init__(self, rate_per_minute: float, capacity: int | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self.rate = rate_per_minute / 60.0
        self.capacity = float(capacity if capacity is not None else max(1, int(rate_per_minute)))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    async def acquire(self, tokens: float = 1.0, *, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            if deadline is not None and time.monotonic() + wait > deadline:
                raise ProviderRateLimitError(
                    f"Token bucket timeout after {timeout}s", provider="<bucket>"
                )
            await asyncio.sleep(min(wait, 1.0))

    async def try_acquire(self, tokens: float = 1.0) -> bool:
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False


class BreakerState(StrEnum):
    CLOSED = "closed"  # healthy, requests flow
    OPEN = "open"  # failing, requests short-circuit
    HALF_OPEN = "half_open"  # cooldown elapsed, probing with one request


@dataclass
class CircuitBreaker:
    """Trips after ``threshold`` consecutive failures, recovers after cooldown."""

    name: str
    threshold: int = 5
    cooldown_seconds: float = 300.0
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if time.monotonic() - self._opened_at >= self.cooldown_seconds:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    @property
    def is_available(self) -> bool:
        return self.state is not BreakerState.OPEN

    def record_success(self) -> None:
        if self._failures or self._opened_at:
            log.info("circuit_closed", provider=self.name)
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            log.warning(
                "circuit_opened",
                provider=self.name,
                failures=self._failures,
                cooldown_seconds=self.cooldown_seconds,
            )

    def reset(self) -> None:
        self._failures = 0
        self._opened_at = None


class DailyQuota:
    """Hard per-day call cap (Alpha Vantage: 25, FMP: 250).

    Resets on a rolling 24-hour window from first use.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._used = 0
        self._window_start = time.monotonic()

    def _maybe_reset(self) -> None:
        if time.monotonic() - self._window_start >= 86_400:
            self._used = 0
            self._window_start = time.monotonic()

    @property
    def remaining(self) -> int:
        self._maybe_reset()
        return max(0, self.limit - self._used)

    def consume(self, n: int = 1) -> bool:
        self._maybe_reset()
        if self._used + n > self.limit:
            return False
        self._used += n
        return True
