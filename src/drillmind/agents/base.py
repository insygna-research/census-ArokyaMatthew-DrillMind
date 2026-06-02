"""
DrillMind — Multi-Agent Base Types
====================================
Common dataclasses shared by every specialised agent.

The multi-agent layer sits ABOVE the existing tool-calling engine. Each
agent is allowed to invoke the same tools the legacy orchestrator uses
(:mod:`drillmind.agents.tools`) plus the new RAG retriever and alert
manager. Agents return a structured :class:`AgentResult` that the
orchestrator concatenates into the final response.

This module intentionally has no FastAPI / Starlette dependency so the
agents are trivially unit-testable.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Output of a single agent invocation."""
    agent: str
    answer: str
    intent: str
    tools_called: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    handoff_to: Optional[str] = None    # if set, orchestrator chains
    confidence: float = 1.0
    elapsed_ms: int = 0
    grounded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "intent": self.intent,
            "answer": self.answer,
            "tools_called": self.tools_called,
            "citations": self.citations,
            "evidence_count": len(self.evidence),
            "handoff_to": self.handoff_to,
            "confidence": self.confidence,
            "elapsed_ms": self.elapsed_ms,
            "grounded": self.grounded,
        }


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base. Subclasses implement :py:meth:`_handle`.

    The :py:meth:`run` wrapper handles timing and exception isolation so
    one misbehaving agent cannot break the orchestrator.
    """

    name: str = "base"
    intent: str = "general"

    def __init__(self, state: dict, llm_fn=None) -> None:
        self._state = state
        self._llm_fn = llm_fn

    async def run(self, question: str, ctx: dict | None = None) -> AgentResult:
        from loguru import logger
        ctx = ctx or {}
        t0 = time.time()
        try:
            result = await self._handle(question, ctx)
        except Exception as e:  # noqa: BLE001
            logger.error("Agent {} failed: {}", self.name, e)
            return AgentResult(
                agent=self.name,
                intent=self.intent,
                answer=f"_Agent {self.name} could not complete due to an internal error: {e}_",
                grounded=False,
                confidence=0.0,
                elapsed_ms=int((time.time() - t0) * 1000),
            )
        result.elapsed_ms = int((time.time() - t0) * 1000)
        return result

    @abstractmethod
    async def _handle(self, question: str, ctx: dict) -> AgentResult:
        ...
