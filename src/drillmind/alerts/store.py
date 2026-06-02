"""
DrillMind — Alert Persistence (SQLite)
======================================
Lightweight, dependency-free storage for drilling alerts.

Why SQLite
----------
* Zero extra service to deploy — file-on-disk inside the data/ tree.
* WAL mode for safe concurrent reads while the API writes.
* Schema migrations are trivial because there are only two tables.
* Survives container restarts; can be backed up with `cp` or rsync.

Tables
------
* ``alerts``                — every distinct alert seen on the rig.
* ``alert_resolutions``     — append-only audit trail for resolution actions.

Every write goes through a short-lived ``sqlite3.connect`` so we are
safe from "SQLite objects created in a thread can only be used in that
same thread" errors when the FastAPI worker pool spawns tasks.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class AlertStatus(str, Enum):
    """Lifecycle of a single alert."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """A single persisted alert.

    Attributes
    ----------
    id : str
        UUID4 hex — primary key.
    event_type : str
        Drilling event class (kick, lost_circulation, …) or "system".
    severity : str
        One of ``low|medium|high|critical``.
    score : float
        Anomaly score (0..1).
    timestamp : str
        ISO-8601 timestamp of the underlying telemetry sample.
    dedup_key : str
        Deterministic key used to detect duplicates within a window.
    description : str
        Human-readable description.
    recommended_action : str
        Operator action text.
    status : AlertStatus
        Current lifecycle state.
    created_at : str
        ISO timestamp the alert was created in the database.
    updated_at : str
        ISO timestamp of the latest mutation.
    resolved_at : str | None
        ISO timestamp of resolution (None while open).
    resolved_by : str | None
        Operator id / username who resolved (None while open).
    contributing_channels : dict[str, float]
        Sensor channel deviations at peak.
    metadata : dict[str, Any]
        Any additional structured fields (rig_state, bit_depth, …).
    """

    id: str
    event_type: str
    severity: str
    score: float
    timestamp: str
    dedup_key: str
    description: str
    recommended_action: str
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: str = ""
    updated_at: str = ""
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None
    contributing_channels: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, AlertStatus) else str(self.status)
        return d


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    id                     TEXT PRIMARY KEY,
    event_type             TEXT NOT NULL,
    severity               TEXT NOT NULL,
    score                  REAL NOT NULL,
    timestamp              TEXT NOT NULL,
    dedup_key              TEXT NOT NULL,
    description            TEXT NOT NULL,
    recommended_action     TEXT NOT NULL,
    status                 TEXT NOT NULL DEFAULT 'active',
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL,
    resolved_at            TEXT,
    resolved_by            TEXT,
    contributing_channels  TEXT NOT NULL DEFAULT '{}',
    metadata               TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_alerts_status      ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity    ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_event_type  ON alerts(event_type);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at  ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_dedup_key   ON alerts(dedup_key);

CREATE TABLE IF NOT EXISTS alert_resolutions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id      TEXT NOT NULL,
    action        TEXT NOT NULL,
    actor         TEXT,
    note          TEXT,
    occurred_at   TEXT NOT NULL,
    FOREIGN KEY(alert_id) REFERENCES alerts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_resolutions_alert_id ON alert_resolutions(alert_id);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AlertStore:
    """SQLite-backed alert persistence layer.

    Parameters
    ----------
    db_path : str | Path
        Filesystem location of the SQLite database file.
    """

    def __init__(self, db_path: str | Path = "data/alerts.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._migrate()
        logger.info("AlertStore initialised at {}", self._path)

    # ---- internal helpers -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._path),
            timeout=10.0,
            isolation_level=None,  # autocommit; manage tx explicitly when needed
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        # WAL — concurrent reads while writing
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _migrate(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _row_to_alert(self, row: sqlite3.Row) -> Alert:
        return Alert(
            id=row["id"],
            event_type=row["event_type"],
            severity=row["severity"],
            score=float(row["score"]),
            timestamp=row["timestamp"],
            dedup_key=row["dedup_key"],
            description=row["description"],
            recommended_action=row["recommended_action"],
            status=AlertStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            resolved_at=row["resolved_at"],
            resolved_by=row["resolved_by"],
            contributing_channels=json.loads(row["contributing_channels"] or "{}"),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    # ---- public API -------------------------------------------------------

    def insert(self, alert: Alert) -> Alert:
        """Insert a fresh alert. Caller guarantees ``id`` is unique."""
        if not alert.id:
            alert.id = uuid.uuid4().hex
        if not alert.created_at:
            alert.created_at = _utcnow_iso()
        alert.updated_at = alert.created_at

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alerts (
                    id, event_type, severity, score, timestamp,
                    dedup_key, description, recommended_action, status,
                    created_at, updated_at, resolved_at, resolved_by,
                    contributing_channels, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.id,
                    alert.event_type,
                    alert.severity,
                    float(alert.score),
                    alert.timestamp,
                    alert.dedup_key,
                    alert.description,
                    alert.recommended_action,
                    alert.status.value,
                    alert.created_at,
                    alert.updated_at,
                    alert.resolved_at,
                    alert.resolved_by,
                    json.dumps(alert.contributing_channels, default=float),
                    json.dumps(alert.metadata, default=str),
                ),
            )
        return alert

    def update_status(
        self,
        alert_id: str,
        status: AlertStatus,
        actor: str | None = None,
        note: str | None = None,
    ) -> Optional[Alert]:
        now = _utcnow_iso()
        resolved_at = now if status == AlertStatus.RESOLVED else None
        resolved_by = actor if status == AlertStatus.RESOLVED else None

        with self._lock, self._connect() as conn:
            cur = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
            row = cur.fetchone()
            if not row:
                return None

            # If already resolved we still allow re-resolution (idempotent) but keep the
            # original resolved_at if it was set, to preserve the audit trail.
            keep_resolved_at = row["resolved_at"]
            keep_resolved_by = row["resolved_by"]
            if status == AlertStatus.RESOLVED and keep_resolved_at:
                resolved_at = keep_resolved_at
                resolved_by = keep_resolved_by or actor

            conn.execute(
                """
                UPDATE alerts
                   SET status      = ?,
                       updated_at  = ?,
                       resolved_at = ?,
                       resolved_by = ?
                 WHERE id = ?
                """,
                (status.value, now, resolved_at, resolved_by, alert_id),
            )
            conn.execute(
                """
                INSERT INTO alert_resolutions (alert_id, action, actor, note, occurred_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (alert_id, status.value, actor, note, now),
            )

            cur = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
            row = cur.fetchone()
            return self._row_to_alert(row) if row else None

    def get(self, alert_id: str) -> Optional[Alert]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return self._row_to_alert(row) if row else None

    def find_active_by_dedup(
        self,
        dedup_key: str,
        window_seconds: int,
    ) -> Optional[Alert]:
        """Return an active alert with the same dedup key emitted within the window."""
        cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat(timespec="seconds")

        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM alerts
                 WHERE dedup_key = ?
                   AND status IN ('active', 'acknowledged')
                   AND created_at >= ?
                 ORDER BY created_at DESC
                 LIMIT 1
                """,
                (dedup_key, cutoff_iso),
            ).fetchone()
        return self._row_to_alert(row) if row else None

    def list_active(
        self,
        severity: str | None = None,
        event_type: str | None = None,
        limit: int = 200,
    ) -> list[Alert]:
        query = "SELECT * FROM alerts WHERE status IN ('active','acknowledged')"
        params: list[Any] = []
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY datetime(created_at) DESC LIMIT ?"
        params.append(int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_alert(r) for r in rows]

    def list_history(
        self,
        limit: int = 200,
        status: str | None = None,
    ) -> list[Alert]:
        query = "SELECT * FROM alerts"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY datetime(created_at) DESC LIMIT ?"
        params.append(int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_alert(r) for r in rows]

    def summary(self) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE status IN ('active','acknowledged')"
            ).fetchone()[0]
            by_severity = {
                r["severity"]: r["c"]
                for r in conn.execute(
                    "SELECT severity, COUNT(*) AS c FROM alerts "
                    "WHERE status IN ('active','acknowledged') GROUP BY severity"
                )
            }
            by_type = {
                r["event_type"]: r["c"]
                for r in conn.execute(
                    "SELECT event_type, COUNT(*) AS c FROM alerts "
                    "WHERE status IN ('active','acknowledged') GROUP BY event_type"
                )
            }
        return {
            "total": int(total),
            "active": int(active),
            "by_severity": by_severity,
            "by_type": by_type,
        }

    def bulk_insert(self, alerts: Iterable[Alert]) -> int:
        n = 0
        for a in alerts:
            self.insert(a)
            n += 1
        return n
