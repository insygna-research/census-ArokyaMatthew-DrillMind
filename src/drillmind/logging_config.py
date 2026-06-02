"""
DrillMind — Centralised logging setup
======================================
A single loguru sink configured for production:

* JSON sink when ``DRILLMIND_LOG_JSON=1`` (best for shipping to ELK / Loki).
* Coloured human-readable sink otherwise (best for local dev).
* Log level driven by ``DRILLMIND_LOG_LEVEL`` (default ``INFO``).
* Rotates files at 50 MB, keeps 7 days.

This module is safe to call ``configure_logging()`` multiple times —
it idempotently re-binds the sinks.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from loguru import logger

_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
    "<level>{level: <7}</level> "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    "{message}"
)


def _json_formatter(record) -> str:
    payload = {
        "ts": record["time"].isoformat(),
        "level": record["level"].name,
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    if record["exception"] is not None:
        payload["exception"] = str(record["exception"])
    extra = record.get("extra") or {}
    if extra:
        payload["extra"] = {k: v for k, v in extra.items() if k != "ctx"}
    return json.dumps(payload, default=str) + "\n"


def configure_logging(
    level: str | None = None,
    log_dir: str | None = None,
    use_json: bool | None = None,
) -> None:
    """Configure loguru sinks (stdout + optional file)."""
    level = level or os.environ.get("DRILLMIND_LOG_LEVEL", "INFO")
    use_json = use_json if use_json is not None else os.environ.get("DRILLMIND_LOG_JSON", "0") == "1"
    log_dir = log_dir or os.environ.get("DRILLMIND_LOG_DIR", "")

    logger.remove()

    if use_json:
        logger.add(sys.stdout, level=level, format=_json_formatter, enqueue=True, colorize=False)
    else:
        logger.add(sys.stdout, level=level, format=_DEFAULT_FORMAT, enqueue=True, colorize=True)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        logger.add(
            f"{log_dir}/drillmind.log",
            level=level,
            rotation="50 MB",
            retention="7 days",
            compression="zip",
            format=_json_formatter if use_json else _DEFAULT_FORMAT,
            enqueue=True,
        )
    logger.debug("Logging configured: level={} json={} dir={}", level, use_json, log_dir or "-")
