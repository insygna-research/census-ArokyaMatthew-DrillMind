"""
DrillMind — Health, Readiness, Liveness, Metrics endpoints
==========================================================
These are deliberately kept dependency-light and isolated from the rest
of the FastAPI surface so a container orchestrator (Kubernetes, Nomad,
ECS) can hit them without taking dependencies on the heavy data layers.

Routes
------

* ``GET /health``  — always 200 once the process is up.
* ``GET /live``    — same semantics as ``/health``.
* ``GET /ready``   — 200 only once :func:`set_ready` has been called.
  Returns 503 with a JSON payload listing which subsystems are
  pending while we are still warming up.
* ``GET /metrics`` — Prometheus exposition (text format).

The router is mounted at the application root, NOT under ``/api`` so
operators don't need to grep through Swagger to find the probes.
"""

from __future__ import annotations

import os
import platform
import time
from typing import Any

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from drillmind.observability import metrics_text, set_readiness

router = APIRouter(tags=["health"])

_START_TIME = time.time()
_READY_FLAGS: dict[str, bool] = {
    "time_log_loaded": False,
    "features_built": False,
    "models_ready": False,
    "anomaly_pipeline_ready": False,
    "rig_state_ready": False,
    "kpis_ready": False,
    "quality_report_ready": False,
    "rag_store_ready": False,
    "alert_store_ready": False,
    "agents_ready": False,
}


def mark_ready(component: str, ready: bool = True) -> None:
    """Toggle a sub-component readiness flag. Updates the Prometheus gauge."""
    if component not in _READY_FLAGS:
        _READY_FLAGS[component] = ready
    else:
        _READY_FLAGS[component] = ready
    set_readiness(all(_READY_FLAGS.values()))


def mark_all_ready() -> None:
    for k in _READY_FLAGS:
        _READY_FLAGS[k] = True
    set_readiness(True)


def reset_readiness() -> None:
    for k in _READY_FLAGS:
        _READY_FLAGS[k] = False
    set_readiness(False)


def is_ready() -> bool:
    return all(_READY_FLAGS.values())


def readiness_state() -> dict[str, Any]:
    return {
        "ready": is_ready(),
        "subsystems": dict(_READY_FLAGS),
        "uptime_seconds": round(time.time() - _START_TIME, 1),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness — returns 200 if the process is running."""
    return {
        "status": "ok",
        "service": "drillmind",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "python": platform.python_version(),
    }


@router.get("/live")
async def live() -> dict[str, Any]:
    return await health()


@router.get("/ready")
async def ready() -> Response:
    state = readiness_state()
    if state["ready"]:
        return JSONResponse(status_code=200, content=state)
    return JSONResponse(status_code=503, content=state)


@router.get("/metrics")
async def metrics() -> Response:
    payload, content_type = metrics_text()
    return Response(content=payload, media_type=content_type)
