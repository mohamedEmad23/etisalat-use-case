"""In-process rate limiting for POST /chat (OWASP API4/API6 control).

A fixed sliding-window counter keyed by the bearer token's digest.
Single-instance service, so an in-process counter is honest and exact.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from telco_churn.api.errors import error_body


class RateLimiter:
    """One deque of call timestamps per client key."""

    def __init__(self, *, max_calls: int, window_seconds: float) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        """True when the call is inside the window budget."""
        now = time.monotonic() if now is None else now
        queue = self._hits.setdefault(key, deque())
        while queue and now - queue[0] > self.window_seconds:
            queue.popleft()
        if len(queue) >= self.max_calls:
            return False
        queue.append(now)
        return True


def client_key(request: Request) -> str:
    """Key by bearer token digest, falling back to client address."""
    authorization = request.headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ")
        digest = hashlib.sha256(token.encode()).hexdigest()[:16]
        return f"token:{digest}"
    forwarded: str | None = getattr(request.client, "host", None)
    return f"addr:{forwarded or 'unknown'}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """429 with the stable error body once the per-key budget is spent."""

    def __init__(self, app: Any, limiter: RateLimiter, *, path: str = "/chat") -> None:
        super().__init__(app)
        self._limiter = limiter
        self._path = path

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if request.url.path != self._path:
            return await call_next(request)
        if not self._limiter.allow(client_key(request)):
            body = error_body(
                "rate_limited", "too many requests — slow down and retry later"
            )
            return Response(
                content=json.dumps(body),
                status_code=429,
                headers={"Retry-After": str(int(self._limiter.window_seconds))},
                media_type="application/json",
            )
        return await call_next(request)
