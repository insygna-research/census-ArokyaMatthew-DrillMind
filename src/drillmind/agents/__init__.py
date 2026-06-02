"""DrillMind agent layer.

Two orchestrators are exported:
* :class:`AgentOrchestrator` — original single-loop tool-calling engine
  (kept for backwards compatibility and used as the ``general`` fallback).
* :class:`MultiAgentOrchestrator` — state-machine that routes to one of
  the four specialised agents (drilling / safety / historical /
  reporting) and supports hand-offs.
"""

from drillmind.agents.orchestrator import AgentOrchestrator
from drillmind.agents.router import Intent, IntentRouter, RouterDecision
from drillmind.agents.multi_orchestrator import MultiAgentOrchestrator, MultiAgentResult
from drillmind.agents.base import AgentResult, BaseAgent

__all__ = [
    "AgentOrchestrator",
    "MultiAgentOrchestrator",
    "MultiAgentResult",
    "AgentResult",
    "BaseAgent",
    "Intent",
    "IntentRouter",
    "RouterDecision",
]
