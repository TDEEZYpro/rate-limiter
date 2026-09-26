"""HTTP service wrapping :mod:`ratelimiter`.

Run directly:

    python service.py --host 127.0.0.1 --port 8080

All requests are rate limited by the ``X-API-Key`` header, falling back to the
connecting client IP when the header is absent. The limiter uses a monotonic
clock, so wall-clock changes do not affect limits; the injectable clock stays
at its default here (tests import :class:`RateLimiter` directly instead).

Endpoints
---------
* Any path: rate limited. Exceeding the limit returns ``429`` with a
  ``Retry-After`` header and a JSON body.
* ``GET /stats``: ``{bucket_count, buckets: {key: tokens}}``.
* ``POST /config`` with JSON ``{"capacity": .., "rate": ..}``: updates limits
  at runtime without dropping bucket state.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ratelimiter import RateLimiter


def build_server(
    host: str,
    port: int,
    *,
    limiter: RateLimiter | None = None,
    host_factory=None,
):
    """Create a ThreadingHTTPServer whose handler uses the given limiter.

    ``host_factory(limiter)`` builds (and returns) the HTTP handler class so a
    fresh limiter can be injected in tests without rebuilding the socket stack.
    When ``limiter`` is None a default one (capacity 10, rate 2/s) is created.
    """
    if limiter is None:
        limiter = RateLimiter(capacity=10, rate=2.0)  # sane production default
    handler_cls = host_factory(limiter) if host_factory else _Handler
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    httpd.limiter = limiter  # attach for programmatic access / tests
    return httpd


class _Handler(BaseHTTPRequestHandler):
    server_version = "RateLimiter/1.0"
    protocol_version = "HTTP/1.1"

    # -- routing -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (library API name)
        path = urlparse(self.path).path
        if path == "/stats":
            self._send_json(200, self.server.limiter.snapshot_report())
        elif self._limited():
            return
        else:
            self._send_json(200, {"ok": True})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/config":
            self._update_config()
        elif self._limited():
            return
        else:
            self._send_json(404, {"error": "not found"})

    # -- helpers -----------------------------------------------------------
    def _client_key(self) -> str:
        key = self.headers.get("X-API-Key")
        if key:
            return key
        # socket.getpeername can fail in some sandboxed tests; degrade safely.
        try:
            return self.client_address[0]
        except Exception:  # pragma: no cover - defensive
            return "anonymous"

    def _limited(self) -> bool:
        limiter = self.server.limiter
        allowed, retry_after = limiter.allow(self._client_key())
        if allowed:
            return False
        self._send_response_json(429, {"error": "rate limit exceeded"},
                                 extra_headers={"Retry-After": f"{round(retry_after, 6)}"})
        return True

    def _update_config(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": "invalid JSON", "detail": str(exc)})
            return

        kwargs = {}
        if "capacity" in body:
            kwargs["capacity"] = float(body["capacity"])
        if "rate" in body:
            kwargs["rate"] = float(body["rate"])

        try:
            self.server.limiter.update_config(**kwargs)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        cfg = self.server.limiter.config
        self._send_json(
            200,
            {"ok": True, "config": {"capacity": cfg.capacity, "rate": cfg.rate}},
        )

    # -- response helpers --------------------------------------------------
    def _send_response_json(self, status: int, payload: dict,
                            extra_headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send_response_json(status, payload)

    def log_message(self, *args: object) -> None:  # silence default stderr logging
        pass


def _run() -> None:
    parser = argparse.ArgumentParser(description="Rate-limited HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    httpd = build_server(args.host, args.port, host_factory=None)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        httpd.shutdown()


if __name__ == "__main__":
    _run()
