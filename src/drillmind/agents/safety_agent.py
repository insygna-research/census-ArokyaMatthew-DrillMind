"""
DrillMind — Safety Agent (Well Control + Anomaly)
==================================================
Owns kick detection, lost circulation, stuck pipe and BOP-related
questions. Always reads the live anomaly status AND the active alerts
table so it can answer "is anything dangerous right now?" with proof.

Tools called
------------
* ``get_anomaly_status``  — current ensemble score + top events
* ``get_current_sensors`` — pit / flow / gas / SPP snapshot
* ``get_rig_state``       — required context for any well-control claim
* (RAG hybrid)            — DDR history on the same problem class

The agent escalates an alert to ``critical`` if the live score crosses
0.5 and the current rig state is DRILLING or REAMING.
"""

from __future__ import annotations

from typing import Any

from drillmind.agents.base import AgentResult, BaseAgent
from drillmind.agents.tools import execute_tool


class SafetyAgent(BaseAgent):
    name = "safety_agent"
    intent = "safety"

    CRITICAL_SCORE = 0.5

    async def _handle(self, question: str, ctx: dict) -> AgentResult:
        tools_called: list[str] = []
        evidence: list[dict] = []
        citations: list[str] = []

        anom = execute_tool("get_anomaly_status", self._state)
        evidence.append({"tool": "get_anomaly_status", "data": anom})
        tools_called.append("get_anomaly_status")

        sensors = execute_tool("get_current_sensors", self._state)
        evidence.append({"tool": "get_current_sensors", "data": sensors})
        tools_called.append("get_current_sensors")

        rig = execute_tool("get_rig_state", self._state)
        evidence.append({"tool": "get_rig_state", "data": rig})
        tools_called.append("get_rig_state")

        # Hybrid RAG — DDR precedent for the same class
        hybrid = ctx.get("hybrid_retriever") or self._state.get("hybrid_retriever")
        rag_hits: list[dict] = []
        if hybrid is not None:
            try:
                results = hybrid.search(query=question, top_k=3)
                rag_hits = [r.to_dict() for r in results]
                evidence.append({"tool": "rag_hybrid_search", "data": {"results": rag_hits}})
                tools_called.append("rag_hybrid_search")
                for r in rag_hits:
                    citations.append(r.get("source", ""))
            except Exception:  # noqa: BLE001
                rag_hits = []

        # Active alerts from store
        alert_summary: dict[str, Any] = {}
        alert_manager = ctx.get("alert_manager") or self._state.get("alert_manager")
        if alert_manager is not None:
            try:
                alert_summary = alert_manager.summary()
                evidence.append({"tool": "alert_summary", "data": alert_summary})
                tools_called.append("alert_summary")
            except Exception:  # noqa: BLE001
                pass

        answer = self._format(question, anom, sensors, rig, rag_hits, alert_summary)
        return AgentResult(
            agent=self.name,
            intent=self.intent,
            answer=answer,
            tools_called=tools_called,
            citations=citations,
            evidence=evidence,
            confidence=0.9,
        )

    # ---------------------------------------------------------------- helpers

    def _format(
        self,
        question: str,
        anom: dict,
        sensors: dict,
        rig: dict,
        rag_hits: list[dict],
        alert_summary: dict,
    ) -> str:
        lines = ["### Safety / Well Control Assessment"]

        score = anom.get("score", 0.0) if isinstance(anom, dict) else 0.0
        active = anom.get("anomaly_active", False) if isinstance(anom, dict) else False
        rig_state = rig.get("current_state", "unknown") if isinstance(rig, dict) else "unknown"

        if active and float(score) >= self.CRITICAL_SCORE and rig_state in ("drilling", "reaming"):
            lines.append(f"\n🚨 **Critical anomaly active** while {rig_state.upper()} — score = {round(float(score),3)}.")
        elif active:
            lines.append(f"\n⚠ **Anomaly active** — score = {round(float(score),3)}, rig state = {rig_state}.")
        else:
            lines.append(f"\n✅ No anomaly active — score = {round(float(score),3)}, rig state = {rig_state}.")

        # Top events
        events = anom.get("recent_events", []) if isinstance(anom, dict) else []
        if events:
            lines.append("\n**Top recent events**")
            for e in events[:5]:
                lines.append(
                    f"- [{e['severity'].upper()}] {e['type']} at {e['timestamp']} "
                    f"(score {e['score']}) — {e['description']}"
                )

        # Sensor snapshot relevant to well control
        s = sensors.get("sensors", {}) if isinstance(sensors, dict) else {}
        if s:
            lines.append("\n**Well-control sensors**")
            for k in ("pit_volume_active", "flow_pumps", "spp", "gas_total", "mud_weight_in", "mud_weight_out"):
                if k in s and s[k] is not None:
                    lines.append(f"- `{k}` = **{s[k]}**")

        # Active alerts roll-up
        if alert_summary:
            lines.append(
                f"\n**Open alerts**: {alert_summary.get('active', 0)} "
                f"(by severity: { dict(alert_summary.get('by_severity', {})) })"
            )

        # DDR precedent
        if rag_hits:
            lines.append("\n**Operational precedent (DDRs)**")
            for r in rag_hits:
                ssrc = r.get("source", "DDR")
                snippet = (r.get("text", "") or "")[:180].replace("\n", " ")
                lines.append(f"- {ssrc}: {snippet}…")

        return "\n".join(lines)
