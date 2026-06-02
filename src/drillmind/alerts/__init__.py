"""
DrillMind Alert Management
==========================
Persistent, deduplicated, broadcastable alert pipeline.

Subpackage exports the public surface:

    from drillmind.alerts import AlertManager, AlertStore, Alert

The manager is the only object the FastAPI layer should touch. The
underlying SQLite store is thread-safe via short-lived connections.
"""

from drillmind.alerts.store import Alert, AlertStatus, AlertStore
from drillmind.alerts.manager import AlertManager, AlertBroadcaster

__all__ = [
    "Alert",
    "AlertStatus",
    "AlertStore",
    "AlertManager",
    "AlertBroadcaster",
]
