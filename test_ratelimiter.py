"""Deterministic tests for the token-bucket rate limiter and HTTP service.

No sleeps longer than 0.01s (in fact none at all): time is a fake clock we
advance directly, so every timing assertion is exact.
"""

from __future__ import annotations

import http.client
import json
import threading
import unittest

from ratelimiter import RateLimiter
from service import build_server


class FakeClock:
    """A monotonic clock measured in nanoseconds that we advance by hand."""

    def __init__(self, start_ns: float = 0.0) -> None:
        self._t = start_ns

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds * 1_000_000_000.0


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        # capacity 10, refills 2 tokens/second.
        self.lim = RateLimiter(10, 2.0, clock=self.clock)

    def test_exhaustion(self) -> None:
        for _ in range(10):
            allowed, retry_after = self.lim.allow("k")
            self.assertTrue(allowed)
            self.assertEqual(retry_after, 0.0)

        allowed, retry_after = self.lim.allow("k")
        self.assertFalse(allowed)
        # One token short at 2/s -> wait exactly 0.5s.
        self.assertAlmostEqual(retry_after, 0.5, places=9)

    def test_isolated_buckets(self) -> None:
        for _ in range(10):
            self.lim.allow("a")
        _, retry_a = self.lim.allow("a")  # now denied, fully drained
        allowed_b, _ = self.lim.allow("b")
        self.assertNotEqual(retry_a, 0.0)  # 'a' is drained
        self.assertTrue(allowed_b)  # 'b' starts full and independent

    def test_refill_over_time(self) -> None:
        for _ in range(10):
            self.lim.allow("k")
        _, retry = self.lim.allow("k")
        self.assertFalse(retry == 0.0)
        self.clock.advance(1.0)  # +2 tokens -> exactly 2
        for _ in range(2):
            allowed, _ = self.lim.allow("k")
            self.assertTrue(allowed)
        _, retry_after = self.lim.allow("k")
        self.assertEqual(retry_after, 0.5)  # 1 token short at 2/s

    def test_refill_exact(self) -> None:
        self.lim.allow("k")  # tokens 9
        self.clock.advance(0.5)  # +1 token -> exactly 10 (capped)
        allowed, _ = self.lim.allow("k")
        self.assertTrue(allowed)

    def test_fractional_accumulation(self) -> None:
        clock = FakeClock()
        lim = RateLimiter(1, 4.0, clock=clock)  # capacity 1, 4 tokens/s
        allowed, _ = lim.allow("k")
        self.assertTrue(allowed)  # tokens now exactly 0
        clock.advance(0.1)  # +0.4 fractional token
        allowed, retry_after = lim.allow("k", cost=1)
        self.assertFalse(allowed)
        # deficit 0.6 at 4/s -> exactly 0.15s
        self.assertAlmostEqual(retry_after, 0.15, places=9)

    def test_cost_greater_than_one(self) -> None:
        lim = RateLimiter(5, 1.0, clock=FakeClock())
        allowed, _ = lim.allow("k", cost=3)
        self.assertTrue(allowed)  # tokens 2
        allowed, retry_after = lim.allow("k", cost=3)
        self.assertFalse(allowed)
        self.assertAlmostEqual(retry_after, 1.0, places=9)  # need 1 more at 1/s

    def test_eviction(self) -> None:
        clock = FakeClock()
        lim = RateLimiter(10, 2.0, eviction_window=300.0, clock=clock)
        lim.allow("a")
        lim.allow("b")
        self.assertEqual(lim.bucket_count, 2)
        # Advance past the eviction window; buckets should be swept on next query.
        clock.advance(301.0)
        self.assertEqual(lim.bucket_count, 0)

    def test_eviction_preserves_recent(self) -> None:
        clock = FakeClock()
        lim = RateLimiter(10, 2.0, eviction_window=300.0, clock=clock)
        lim.allow("old")
        clock.advance(200.0)
        lim.allow("fresh")
        # 'old' idle >300s from t=0 while 'fresh' (t=200) stays recent.
        clock.advance(101.0)  # now t=301; old last at 0 < cutoff(1e9)
        self.assertEqual(lim.bucket_count, 1)

    def test_config_update_keeps_state(self) -> None:
        lim = RateLimiter(10, 2.0, clock=FakeClock())
        for _ in range(10):
            lim.allow("k")
        _, retry_before = lim.allow("k")  # fully drained -> denied
        self.assertTrue(retry_before > 0)
        self.assertAlmostEqual(retry_before, 0.5, places=9)
        # Lower capacity; existing fractional/bucket state persists (no reset).
        lim.update_config(capacity=20, rate=5.0)
        self.assertEqual(lim.config.capacity, 20)
        self.assertEqual(lim.config.rate, 5.0)
        self.assertEqual(lim.bucket_count, 1)  # bucket not dropped

    def test_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            RateLimiter(0, 1.0)
        with self.assertRaises(ValueError):
            RateLimiter(10, -1.0)
        with self.assertRaises(ValueError):
            self.lim.update_config(rate=0)

    def test_concurrent_access_50_threads(self) -> None:
        # Frozen clock => no refill, so exactly `capacity` successes total.
        lim = RateLimiter(10, 2.0, clock=FakeClock())
        results: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            allowed, _ = lim.allow("shared")
            with lock:
                results.append(allowed)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 50)
        self.assertEqual(sum(results), 10)  # exactly capacity admitted


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.lim = RateLimiter(10, 2.0, clock=self.clock)
        self.httpd = build_server("127.0.0.1", 0, limiter=self.lim)
        self.host, self.port = self.httpd.server_address
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def _request(self, method: str, path: str, body: object | None = None,
                 headers: dict[str, str] | None = None) -> tuple[int, dict, bytes]:
        conn = http.client.HTTPConnection(self.host, self.port)
        payload = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=payload, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        status = resp.status
        hdrs = dict(resp.getheaders())
        conn.close()
        return status, hdrs, data

    def test_limit_returns_429_with_retry_after(self) -> None:
        headers = {"X-API-Key": "key-1"}
        for _ in range(10):
            status, _, _ = self._request("GET", "/", headers=headers)
            self.assertEqual(status, 200)
        status, hdrs, data = self._request("GET", "/", headers=headers)
        self.assertEqual(status, 429)
        self.assertIn("Retry-After", hdrs)
        payload = json.loads(data)
        self.assertIn("error", payload)

    def test_ip_fallback(self) -> None:
        for _ in range(10):
            status, _, _ = self._request("GET", "/", headers={})
            self.assertEqual(status, 200)
        status, hdrs, _ = self._request("GET", "/", headers={})
        self.assertEqual(status, 429)
        self.assertIn("Retry-After", hdrs)

    def test_stats_endpoint(self) -> None:
        self._request("GET", "/", headers={"X-API-Key": "k"})
        status, _, data = self._request("GET", "/stats")
        self.assertEqual(status, 200)
        report = json.loads(data)
        self.assertIn("bucket_count", report)
        self.assertIn("buckets", report)
        self.assertEqual(report["bucket_count"], 1)

    def test_config_update_via_http(self) -> None:
        # Warm up a bucket so we can prove state survives a config change.
        for _ in range(9):
            self._request("GET", "/", headers={"X-API-Key": "persist"})
        _, _, before = self._request("GET", "/stats")
        stats_before = json.loads(before)

        status, _, data = self._request(
            "POST", "/config", body={"capacity": 50, "rate": 10.0}, headers={"X-API-Key": "admin"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["config"], {"capacity": 50, "rate": 10.0})

        # Bucket for 'persist' still present (not dropped by config change).
        stats_after = json.loads(self._request("GET", "/stats")[2])
        self.assertIn("persist", stats_after["buckets"])

    def test_config_bad_value(self) -> None:
        status, _, data = self._request(
            "POST", "/config", body={"rate": -1}, headers={"X-API-Key": "admin"}
        )
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(data))


if __name__ == "__main__":
    unittest.main(verbosity=2)
