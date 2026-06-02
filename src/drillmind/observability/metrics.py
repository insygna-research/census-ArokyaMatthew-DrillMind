"""
DrillMind — Prometheus metrics
==============================
Thin wrapper around prometheus_client that degrades gracefully when the
package is not installed (returns a static text response). Every metric
exposed here is consumed by ``GET /metrics``.

Metrics
-------
* ``drillmind_app_info``                — build / version (Info)
* ``drillmind_ready``                   — readiness flag (Gauge)
* ``drillmind_http_requests_total``     — by method/path/status (Counter)
* ``drillmind_http_request_seconds``    — request duration (Histogram)
* ``drillmind_websocket_clients``       — active WS clients (Gauge)
* ``drillmind_alerts_total``            — alerts created (Counter)
* ``drillmind_copilot_queries_total``   — copilot queries (Counter)
* ``drillmind_copilot_query_seconds``   — copilot query duration (Histogram)
* ``drillmind_anomaly_score``           — last anomaly score (Gauge)
"""

from __future__ import annotations

import time
from typing import Any, Iterable

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:  # noqa: WPS433
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


# ---------------------------------------------------------------------------
# Registry + metric singletons
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    REGISTRY = CollectorRegistry(auto_describe=True)

    APP_INFO = Info(
        "drillmind_app",
        "DrillMind build information",
        registry=REGISTRY,
    )
    READY = Gauge(
        "drillmind_ready",
        "1 if all subsystems are loaded and the API is ready, else 0",
        registry=REGISTRY,
    )
    HTTP_REQUESTS = Counter(
        "drillmind_http_requests_total",
        "HTTP request count",
        labelnames=("method", "path", "status"),
        registry=REGISTRY,
    )
    HTTP_LATENCY = Histogram(
        "drillmind_http_request_seconds",
        "HTTP request duration in seconds",
        labelnames=("method", "path"),
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
        registry=REGISTRY,
    )
    WS_CLIENTS = Gauge(
        "drillmind_websocket_clients",
        "Currently connected WebSocket clients",
        registry=REGISTRY,
    )
    ALERTS_TOTAL = Counter(
        "drillmind_alerts_total",
        "Alerts created (post-dedup)",
        labelnames=("severity", "event_type"),
        registry=REGISTRY,
    )
    COPILOT_QUERIES = Counter(
        "drillmind_copilot_queries_total",
        "Copilot / agent queries",
        labelnames=("intent",),
        registry=REGISTRY,
    )
    COPILOT_LATENCY = Histogram(
        "drillmind_copilot_query_seconds",
        "Copilot query duration in seconds",
        labelnames=("intent",),
        buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
        registry=REGISTRY,
    )
    ANOMALY_SCORE = Gauge(
        "drillmind_anomaly_score",
        "Most recent combined anomaly score",
        registry=REGISTRY,
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def register_app_info(version: str, well: str, field_name: str) -> None:
    if PROMETHEUS_AVAILABLE:
        APP_INFO.info({"version": version, "well": well, "field": field_name})


def set_readiness(ready: bool) -> None:
    if PROMETHEUS_AVAILABLE:
        READY.set(1 if ready else 0)


def record_request(method: str, path: str, status: int, duration_s: float) -> None:
    if not PROMETHEUS_AVAILABLE:
        return
    HTTP_REQUESTS.labels(method=method, path=path, status=str(status)).inc()
    HTTP_LATENCY.labels(method=method, path=path).observe(max(duration_s, 0.0))


def set_websocket_clients(n: int) -> None:
    if PROMETHEUS_AVAILABLE:
        WS_CLIENTS.set(max(0, int(n)))


def inc_alert(severity: str, event_type: str) -> None:
    if PROMETHEUS_AVAILABLE:
        ALERTS_TOTAL.labels(severity=severity, event_type=event_type).inc()


def observe_query(intent: str, duration_s: float) -> None:
    if PROMETHEUS_AVAILABLE:
        COPILOT_QUERIES.labels(intent=intent).inc()
        COPILOT_LATENCY.labels(intent=intent).observe(max(duration_s, 0.0))


def record_anomaly_score(score: float) -> None:
    if PROMETHEUS_AVAILABLE:
        try:
            ANOMALY_SCORE.set(float(score))
        except (TypeError, ValueError):
            pass


def metrics_text() -> tuple[bytes, str]:
    if PROMETHEUS_AVAILABLE:
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
    msg = (
        "# prometheus_client is not installed.\n"
        "# Run: pip install prometheus_client to enable metrics.\n"
    ).encode("utf-8")
    return msg, "text/plain; charset=utf-8"
