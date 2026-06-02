"""
DrillMind — Multi-Agent Orchestrator
=====================================
State machine that routes a single user question to one or more
specialised agents and chains them via *hand-off*.

States
------

    start ──▶ ROUTING ──▶ AGENT_RUN ──▶ HANDOFF? ──▶ AGENT_RUN ──▶ DONE
                                ▲                       │
                                └───── chain (≤2 hops) ─┘

* ``ROUTING``  — :class:`IntentRouter` decides the first agent.
* ``AGENT_RUN`` — the chosen agent executes and returns
  :class:`AgentResult` plus an optional ``handoff_to``.
* ``HANDOFF?`` — if a hand-off is requested AND we have hops left we
  loop back; otherwise we fall through to ``DONE``.

The orchestrator deliberately caps the number of hops (default 2 +
primary) to keep latency predictable. An LLM, when configured, is used
only for the final synthesis step — never for routing — so the routing
behaviour stays deterministic and auditable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from drillmind.agents.base import AgentResult, BaseAgent
from drillmind.agents.drilling_agent import DrillingAgent
from drillmind.agents.historical_agent import HistoricalAgent
from drillmind.agents.reporting_agent import ReportingAgent
from drillmind.agents.router import Intent, IntentRouter, RouterDecision
from drillmind.agents.safety_agent import SafetyAgent
# Legacy single-loop tool-calling orchestrator — used as fallback for
# Intent.GENERAL so we never regress the original behaviour.
from drillmind.agents.orchestrator import AgentOrchestrator


@dataclass
class MultiAgentResult:
    """Final orchestrator output that the API returns to the client."""
    answer: str
    intent: str
    agents_run: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    handoffs: list[tuple[str, str]] = field(default_factory=list)
    elapsed_ms: int = 0
    grounded: bool = True
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "intent": self.intent,
            "agents_run": self.agents_run,
            "tools_called": self.tools_called,
            "citations": self.citations,
            "handoffs": [list(h) for h in self.handoffs],
            "elapsed_ms": self.elapsed_ms,
            "grounded": self.grounded,
            "confidence": self.confidence,
        }


class MultiAgentOrchestrator:
    """Top-level orchestrator.

    Parameters
    ----------
    state : dict
        Application state shared with all agents (DataFrames, stores).
    llm_fn : callable | None
        Optional async function ``(system_prompt, user_prompt) -> str``.
        When provided, used to *synthesise* a coherent final answer that
        merges the individual agent outputs into a single narrative.
    max_hops : int
        Maximum number of hand-offs allowed after the primary agent.
        Default 2 (primary + up to 2 chained agents).
    """

    def __init__(
        self,
        state: dict,
        llm_fn=None,
        max_hops: int = 2,
    ) -> None:
        self._state = state
        self._llm_fn = llm_fn
        self._max_hops = max(0, int(max_hops))

        # Agent registry
        self._agents: dict[str, BaseAgent] = {
            "drilling_agent":  DrillingAgent(state=state, llm_fn=llm_fn),
            "safety_agent":    SafetyAgent(state=state, llm_fn=llm_fn),
            "historical_agent": HistoricalAgent(state=state, llm_fn=llm_fn),
            "reporting_agent": ReportingAgent(state=state, llm_fn=llm_fn),
        }
        # Legacy tool-loop for Intent.GENERAL
        self._legacy = AgentOrchestrator(state=state, llm_fn=llm_fn)
        self._router = IntentRouter()

    # ---- public ----------------------------------------------------------

    async def query(self, question: str) -> MultiAgentResult:
        t0 = time.time()
        decision = self._router.classify(question)
        logger.info(
            "MultiAgent route: intent={} score={} secondary={}",
            decision.intent.value, decision.score, [s.value for s in decision.secondary],
        )

        # GENERAL → fall back to legacy single-loop tool calling — keeps the
        # original behaviour for "what is happening right now?" type asks.
        if decision.intent == Intent.GENERAL:
            legacy = await self._legacy.query(question)
            return MultiAgentResult(
                answer=legacy.answer,
                intent=legacy.intent,
                agents_run=["legacy_tool_loop"],
                tools_called=legacy.tools_called,
                citations=[],
                handoffs=[],
                elapsed_ms=int((time.time() - t0) * 1000),
                grounded=legacy.grounded,
                confidence=0.75,
            )

        # Resolve first agent
        primary_name = self._agent_for_intent(decision.intent)
        if primary_name is None:
            # Should never happen — defensive
            legacy = await self._legacy.query(question)
            return MultiAgentResult(
                answer=legacy.answer,
                intent=legacy.intent,
                agents_run=["legacy_tool_loop"],
                tools_called=legacy.tools_called,
                handoffs=[],
                elapsed_ms=int((time.time() - t0) * 1000),
                grounded=legacy.grounded,
                confidence=0.7,
            )

        # State machine ----------------------------------------------------
        results: list[AgentResult] = []
        agents_run: list[str] = []
        tools_called: list[str] = []
        citations: list[str] = []
        handoffs: list[tuple[str, str]] = []

        next_agent = primary_name
        ctx = {
            "hybrid_retriever": self._state.get("hybrid_retriever"),
            "alert_manager":    self._state.get("alert_manager"),
            "well_meta":        self._state.get("well_meta", {}),
        }

        hop = 0
        seen: set[str] = set()
        while next_agent and hop <= self._max_hops:
            if next_agent in seen:
                # Cycle guard
                break
            seen.add(next_agent)
            agent = self._agents.get(next_agent)
            if agent is None:
                break

            res = await agent.run(question, ctx=ctx)
            results.append(res)
            agents_run.append(res.agent)
            tools_called.extend(res.tools_called)
            citations.extend([c for c in res.citations if c])

            if res.handoff_to and res.handoff_to in self._agents and hop < self._max_hops:
                handoffs.append((res.agent, res.handoff_to))
                next_agent = res.handoff_to
                hop += 1
                continue
            break

        # Also chain a secondary intent agent if router suggests one and
        # we haven't already used it.
        for sec in decision.secondary:
            sec_name = self._agent_for_intent(sec)
            if sec_name and sec_name not in seen and len(seen) <= self._max_hops:
                agent = self._agents[sec_name]
                res = await agent.run(question, ctx=ctx)
                results.append(res)
                agents_run.append(res.agent)
                tools_called.extend(res.tools_called)
                citations.extend([c for c in res.citations if c])
                handoffs.append((primary_name, sec_name))
                seen.add(sec_name)

        # Synthesise final answer
        if self._llm_fn is not None:
            try:
                answer = await self._synthesize_with_llm(question, results)
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM synthesis failed: {} — falling back to deterministic merge", e)
                answer = self._merge(question, results)
        else:
            answer = self._merge(question, results)

        avg_conf = (sum(r.confidence for r in results) / len(results)) if results else 0.5
        return MultiAgentResult(
            answer=answer,
            intent=decision.intent.value,
            agents_run=agents_run,
            tools_called=tools_called,
            citations=list(dict.fromkeys(citations))[:20],
            handoffs=handoffs,
            elapsed_ms=int((time.time() - t0) * 1000),
            grounded=all(r.grounded for r in results) if results else False,
            confidence=round(avg_conf, 2),
        )

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _agent_for_intent(intent: Intent) -> Optional[str]:
        return {
            Intent.DRILLING: "drilling_agent",
            Intent.SAFETY: "safety_agent",
            Intent.HISTORICAL: "historical_agent",
            Intent.REPORTING: "reporting_agent",
        }.get(intent)

    @staticmethod
    def _merge(question: str, results: list[AgentResult]) -> str:
        if not results:
            return "_No agent produced output._"
        if len(results) == 1:
            return results[0].answer
        parts = [f"## Multi-agent answer (`{len(results)}` agents)"]
        for r in results:
            parts.append(f"\n#### `{r.agent}` _({r.elapsed_ms} ms · confidence {r.confidence})_")
            parts.append(r.answer)
        return "\n\n".join(parts)

    async def _synthesize_with_llm(self, question: str, results: list[AgentResult]) -> str:
        system = (
            "You are DrillMind, an RTOC drilling analyst. You will be given a user "
            "question and the outputs of several specialised drilling agents that have "
            "already gathered evidence. Your job is to synthesise ONE coherent answer "
            "for an RTOC engineer. RULES:\n"
            "- Never invent numbers. Use only the values that appear in the agent outputs.\n"
            "- Lead with the answer in 1–2 sentences.\n"
            "- Cite tool / DDR sources inline when relevant (e.g. [get_anomaly_status], DDR #37).\n"
            "- If a safety risk is flagged, surface it BEFORE drilling performance discussion.\n"
            "- Keep the answer under 250 words unless the question explicitly asks for a report.\n"
        )
        evidence_blob = "\n\n".join(
            f"### {r.agent} (intent={r.intent}, confidence={r.confidence})\n{r.answer}"
            for r in results
        )
        user = f"QUESTION:\n{question}\n\nAGENT OUTPUTS:\n{evidence_blob}"
        return await self._llm_fn(system, user)
