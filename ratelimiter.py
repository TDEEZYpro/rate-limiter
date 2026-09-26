"""Token bucket rate limiter (standard library only).

Design
------
* Per-key buckets keyed by an arbitrary string (API key or IP).
* Capacity and refill rate are configured globally but may be changed at
  runtime via :meth:`RateLimiter.update_config`. Changing the config never
  drops existing bucket state; it only affects how future refills are
  computed.
* Refill is computed lazily on access from elapsed real time, so there is no
  background thread keeping buckets fresh.
* The clock is injected so tests can advance time deterministically without
  sleeping.

Thread safety
-------------
A single :class:`threading.Lock` guards the bucket table and all mutations.
Each access does at most one lock acquisition, which keeps the critical section
tiny under contention.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class RateConfig:
    """Bucket capacity and refill rate (tokens per second)."""

    capacity: float
    rate: float  # tokens added per second


# time.monotonic on CPython reports nanosecond precision; we treat its raw
# return value as nanoseconds so all maths is integer/float arithmetic.
_NSEC_PER_SECOND = 1_000_000_000.0


class _Bucket:
    __slots__ = ("tokens", "last_ns", "last_access_ns")

    def __init__(self, tokens: float, last_ns: int, now_ns: int) -> None:
        self.tokens = tokens
        self.last_ns = last_ns  # timestamp of last refill accounting
        self.last_access_ns = now_ns  # for eviction liveness tracking


class RateLimiter:
    """A thread-safe, per-key token bucket rate limiter."""

    def __init__(
        self,
        capacity: float,
        rate: float,
        *,
        eviction_window: float = 300.0,
        clock: time.monotonic.__class__ | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if rate <= 0:
            raise ValueError("rate must be positive")

        self._config = RateConfig(capacity=capacity, rate=rate)
        self._eviction_window = eviction_window
        self._clock = clock if clock is not None else time.monotonic

        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}

    # -- configuration -----------------------------------------------------
    @property
    def config(self) -> RateConfig:
        with self._lock:
            return self._config  # immutable dataclass; safe to share

    def update_config(
        self, *, capacity: float | None = None, rate: float | None = None
    ) -> None:
        """Update capacity and/or rate without dropping bucket state.

        Existing buckets are kept as-is; only future refill calculations use
        the new parameters. Lowering capacity below a bucket's current token
        count briefly creates a deficit that refills away; raising it grants
        extra headroom — both are expected for an in-flight config change.
        """
        new_capacity = self._config.capacity if capacity is None else capacity
        new_rate = self._config.rate if rate is None else rate
        if new_capacity <= 0:
            raise ValueError("capacity must be positive")
        if new_rate <= 0:
            raise ValueError("rate must be positive")
        with self._lock:
            self._config = RateConfig(capacity=new_capacity, rate=new_rate)

    # -- core --------------------------------------------------------------
    def allow(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        """Try to consume ``cost`` tokens from ``key``'s bucket.

        Returns ``(allowed, retry_after)``. ``retry_after`` is seconds until
        enough tokens are available (0.0 when allowed). Cost must be positive.
        """
        if cost <= 0:
            raise ValueError("cost must be positive")

        now_ns = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                self._buckets[key] = bucket = _Bucket(
                    tokens=self._config.capacity, last_ns=now_ns, now_ns=now_ns
                )
            else:
                elapsed_s = (now_ns - bucket.last_ns) / _NSEC_PER_SECOND
                if elapsed_s > 0:
                    bucket.tokens += elapsed_s * self._config.rate
                    bucket.last_ns = now_ns
                    if bucket.tokens > self._config.capacity:
                        bucket.tokens = self._config.capacity
                bucket.last_access_ns = now_ns

            return self._charge_locked(bucket, cost)

    def _charge_locked(self, bucket: _Bucket, cost: float) -> tuple[bool, float]:
        """Consume tokens from an already-selected bucket (lock held)."""
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return True, 0.0

        deficit = cost - bucket.tokens
        need_s = deficit / self._config.rate
        if self._config.rate == 0.0:  # defensive; construction disallows it
            return False, float("inf")
        return False, need_s

    def _evict_locked(self) -> None:
        """Remove buckets idle longer than the eviction window (lock held)."""
        now_ns = self._clock()
        cutoff = now_ns - self._eviction_window * _NSEC_PER_SECOND
        dead = [k for k, b in self._buckets.items() if b.last_access_ns < cutoff]
        for k in dead:
            del self._buckets[k]

    def snapshot(self) -> dict[str, float]:
        """Return a copy of current token level per live bucket."""
        with self._lock:
            self._evict_locked()
            return dict(self._buckets)

    def snapshot_report(self) -> dict[str, object]:
        """Bucket count and per-key token levels for the /stats endpoint."""
        with self._lock:
            self._evict_locked()
            return {
                "bucket_count": len(self._buckets),
                "buckets": {k: round(v.tokens, 9) for k, v in self._buckets.items()},
            }

    @property
    def bucket_count(self) -> int:
        with self._lock:
            self._evict_locked()
            return len(self._buckets)
