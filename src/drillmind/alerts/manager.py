"""
DrillMind — Alert Manager
=========================
Operational layer over :class:`AlertStore`.

Responsibilities
----------------
* Convert internal :class:`AnomalyEvent` objects into persisted alerts.
* Deduplicate identical alerts emitted inside a sliding time window
  (default: 120 seconds — matches typical RTOC quiet-period policy).
* Broadcast newly-created alerts to all subscribed WebSocket clients
  without coupling the manager to FastAPI/Starlette internals.

This module is intentionally framework-agnostic. The FastAPI layer
only sees ``AlertBroadcaster.subscribe`` / ``unsubscribe`` and the
``AlertManager`` public API.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from loguru import logger

from drillmind.alerts.store import Alert, AlertStatus, AlertStore


# ---------------------------------------------------------------------------
# Broadcaster
# ---------------------------------------------------------------------------

class AlertBroadcaster:
    """Async fan-out of alert events to multiple subscribers.

    Subscribers register a callable that accepts a single ``dict`` payload.
    The broadcaster is thread-safe enough for the typical FastAPI worker
    pattern: subscriber registration is guarded by an asyncio.Lock and
    delivery is awaited per-subscriber so a slow consumer cannot starve
    others.
    """

    def __init__(self) -> None:
        self._subs: set[Callable[[dict[str, Any]], "asyncio.Future | None"]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self, callback) -> None:
        async with self._lock:
            self._subs.add(callback)

    async def unsubscribe(self, callback) -> None:
        async with self._lock:
            self._subs.discard(callback)

    async def publish(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            subs = list(self._subs)
        for cb in subs:
            try:
                result = cb(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:  # noqa: BLE001 — never let one subscriber crash others
                logger.warning("Alert subscriber raised: {}", e)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

@dataclass
class _DedupConfig:
    """Tunables for alert deduplication."""

    window_seconds: int = 120          # 2 minutes
    score_bucket: float = 0.05         # bucket size for score-based dedup
    use_score: bool = False            # if True, fold score bucket into dedup key


class AlertManager:
    """High-level alert pipeline used by the API and WebSocket layers.

    Typical usage::

        store = AlertStore("data/alerts.db")
        manager = AlertManager(store)
        await manager.create_from_event(event)            # called when an event fires
        active = manager.list_active()                    # served at GET /api/alerts/active
        await manager.resolve(alert_id, actor="aon")      # POST /api/alerts/{id}/resolve

    The constructor accepts an optional :class:`AlertBroadcaster`; if
    omitted, a fresh one is allocated so the FastAPI layer can subscribe.
    """

    def __init__(
        self,
        store: AlertStore,
        broadcaster: AlertBroadcaster | None = None,
        dedup_window_seconds: int = 120,
    ) -> None:
        self._store = store
        self._broadcaster = broadcaster or AlertBroadcaster()
        self._dedup = _DedupConfig(window_seconds=dedup_window_seconds)

    # ---- accessors --------------------------------------------------------

    @property
    def store(self) -> AlertStore:
        return self._store

    @property
    def broadcaster(self) -> AlertBroadcaster:
        return self._broadcaster

    # ---- internal helpers -------------------------------------------------

    @staticmethod
    def _dedup_key(event_type: str, severity: str, timestamp_iso: str, score: float | None = None) -> str:
        """Compute a deterministic dedup key.

        We coarsen ``timestamp`` to the nearest minute so two anomalies fired
        a few seconds apart with the same event type / severity collapse.
        """
        coarse_ts = (timestamp_iso or "")[:16]  # YYYY-MM-DDTHH:MM
        base = f"{event_type}|{severity}|{coarse_ts}"
        if score is not None:
            base += f"|{round(score, 2)}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:24]

    # ---- public mutations -------------------------------------------------

    async def create_alert(
        self,
        *,
        event_type: str,
        severity: str,
        score: float,
        timestamp: str,
        description: str,
        recommended_action: str,
        contributing_channels: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Alert | None:
        """Create a new alert with dedup. Returns None if deduplicated."""
        dedup_key = self._dedup_key(event_type, severity, timestamp, score)

        existing = self._store.find_active_by_dedup(dedup_key, self._dedup.window_seconds)
        if existing is not None:
            logger.debug(
                "Alert deduplicated event_type={} severity={} dedup_key={}",
                event_type, severity, dedup_key,
            )
            return None

        alert = Alert(
            id=uuid.uuid4().hex,
            event_type=event_type,
            severity=severity,
            score=float(score),
            timestamp=timestamp,
            dedup_key=dedup_key,
            description=description,
            recommended_action=recommended_action,
            contributing_channels=contributing_channels or {},
            metadata=metadata or {},
        )

        try:
            stored = self._store.insert(alert)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to persist alert: {}", e)
            return None

        await self._broadcaster.publish({
            "type": "alert",
            "event": "created",
            "alert": stored.to_dict(),
        })
        return stored

    async def create_from_event(self, event: Any) -> Alert | None:
        """Convert an :class:`AnomalyEvent` into a persisted alert."""
        ev_type = (
            event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        )
        sev = event.severity.value if hasattr(event.severity, "value") else str(event.severity)
        ts = str(event.timestamp)
        contributing = dict(event.contributing_channels) if event.contributing_channels else {}
        meta = {
            "duration_rows": int(getattr(event, "duration_rows", 1)),
        }
        return await self.create_alert(
            event_type=ev_type,
            severity=sev,
            score=float(event.score),
            timestamp=ts,
            description=event.description,
            recommended_action=event.recommended_action,
            contributing_channels=contributing,
            metadata=meta,
        )

    async def bulk_seed_from_events(self, events: Iterable[Any]) -> int:
        """Seed the store from a batch of historical events (no broadcasting)."""
        events = list(events)
        inserted = 0
        for ev in events:
            ev_type = (
                ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
            )
            sev = ev.severity.value if hasattr(ev.severity, "value") else str(ev.severity)
            ts = str(ev.timestamp)
            dedup_key = self._dedup_key(ev_type, sev, ts, float(ev.score))
            existing = self._store.find_active_by_dedup(dedup_key, self._dedup.window_seconds)
            if existing is not None:
                continue
            alert = Alert(
                id=uuid.uuid4().hex,
                event_type=ev_type,
                severity=sev,
                score=float(ev.score),
                timestamp=ts,
                dedup_key=dedup_key,
                description=ev.description,
                recommended_action=ev.recommended_action,
                contributing_channels=dict(ev.contributing_channels or {}),
                metadata={"duration_rows": int(getattr(ev, "duration_rows", 1))},
            )
            try:
                self._store.insert(alert)
                inserted += 1
            except Exception:  # noqa: BLE001
                continue
        logger.info("Seeded {} alerts from {} events", inserted, len(events))
        return inserted

    async def acknowledge(self, alert_id: str, actor: str | None = None, note: str | None = None) -> Alert | None:
        alert = self._store.update_status(alert_id, AlertStatus.ACKNOWLEDGED, actor=actor, note=note)
        if alert is not None:
            await self._broadcaster.publish({"type": "alert", "event": "acknowledged", "alert": alert.to_dict()})
        return alert

    async def resolve(self, alert_id: str, actor: str | None = None, note: str | None = None) -> Alert | None:
        alert = self._store.update_status(alert_id, AlertStatus.RESOLVED, actor=actor, note=note)
        if alert is not None:
            await self._broadcaster.publish({"type": "alert", "event": "resolved", "alert": alert.to_dict()})
        return alert

    async def suppress(self, alert_id: str, actor: str | None = None, note: str | None = None) -> Alert | None:
        alert = self._store.update_status(alert_id, AlertStatus.SUPPRESSED, actor=actor, note=note)
        if alert is not None:
            await self._broadcaster.publish({"type": "alert", "event": "suppressed", "alert": alert.to_dict()})
        return alert

    # ---- reads ------------------------------------------------------------

    def list_active(self, severity: str | None = None, event_type: str | None = None, limit: int = 200) -> list[Alert]:
        return self._store.list_active(severity=severity, event_type=event_type, limit=limit)

    def list_history(self, limit: int = 200, status: str | None = None) -> list[Alert]:
        return self._store.list_history(limit=limit, status=status)

    def get(self, alert_id: str) -> Alert | None:
        return self._store.get(alert_id)

    def summary(self) -> dict[str, Any]:
        return self._store.summary()
