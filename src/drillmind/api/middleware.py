"""
DrillMind — HTTP middleware
============================
ASGI middleware that:
* assigns a request-ID to every request,
* records Prometheus latency + count,
* logs structured request lines with loguru.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from drillmind.observability import record_request


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach an ``X-Request-ID`` header to every response."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        response: Response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response


class MetricsAndLoggingMiddleware(BaseHTTPMiddleware):
    """Records Prometheus metrics + emits a structured access line."""

    async def dispatch(self, request: Request, call_next):
        from loguru import logger
        t0 = time.time()
        try:
            response: Response = await call_next(request)
            status = response.status_code
        except Exception:  # noqa: BLE001
            duration = time.time() - t0
            record_request(request.method, _safe_path(request), 500, duration)
            logger.exception(
                "request_failed method={} path={} duration_ms={}",
                request.method, _safe_path(request), int(duration * 1000),
            )
            raise

        duration = time.time() - t0
        path = _safe_path(request)
        record_request(request.method, path, status, duration)
        # Only emit DEBUG for fast healthy reads to keep stdout signal-to-noise high
        if duration > 1.0 or status >= 500:
            logger.warning(
                "request method={} path={} status={} duration_ms={} req_id={}",
                request.method, path, status, int(duration * 1000),
                getattr(request.state, "request_id", "-"),
            )
        else:
            logger.debug(
                "request method={} path={} status={} duration_ms={}",
                request.method, path, status, int(duration * 1000),
            )
        return response


def _safe_path(request: Request) -> str:
    """Use the route template (if matched) so we don't blow up label cardinality."""
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return request.url.path
