"""DrillMind observability — Prometheus metrics + health probes.

Optional dependency: ``prometheus_client``. If not installed, the
metrics endpoint returns a clear ``application/text`` message instead
of failing the import.
"""

from drillmind.observability.metrics import (
    PROMETHEUS_AVAILABLE,
    metrics_text,
    record_request,
    set_readiness,
    register_app_info,
    inc_alert,
    observe_query,
    set_websocket_clients,
    record_anomaly_score,
)

__all__ = [
    "PROMETHEUS_AVAILABLE",
    "metrics_text",
    "record_request",
    "set_readiness",
    "register_app_info",
    "inc_alert",
    "observe_query",
    "set_websocket_clients",
    "record_anomaly_score",
]
