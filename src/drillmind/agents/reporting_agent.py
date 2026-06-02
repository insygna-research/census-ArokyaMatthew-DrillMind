"""
DrillMind — Reporting Agent
===========================
Generates short, ready-to-send shift / DDR summaries from live state.

It aggregates:
* well metadata + interval length
* rig-state time breakdown (operational efficiency)
* KPI averages (MSE, d-exp)
* anomaly stats + most-recent critical events
* open alerts roll-up

Intended consumer is a Drilling Engineer producing the next morning
report — the agent's output should be paste-ready into Word / Email.
"""

from __future__ import annotations

from typing import Any

from drillmind.agents.base import AgentResult, BaseAgent
from drillmind.agents.tools import execute_tool


class ReportingAgent(BaseAgent):
    name = "reporting_agent"
    intent = "reporting"

    async def _handle(self, question: str, ctx: dict) -> AgentResult:
        tools_called: list[str] = []
        evidence: list[dict] = []
        citations: list[str] = []

        rig = execute_tool("get_rig_state", self._state)
        evidence.append({"tool": "get_rig_state", "data": rig})
        tools_called.append("get_rig_state")

        kpis = execute_tool("get_drilling_kpis", self._state)
        evidence.append({"tool": "get_drilling_kpis", "data": kpis})
        tools_called.append("get_drilling_kpis")

        anom = execute_tool("get_anomaly_status", self._state)
        evidence.append({"tool": "get_anomaly_status", "data": anom})
        tools_called.append("get_anomaly_status")

        qa = execute_tool("get_data_quality", self._state)
        evidence.append({"tool": "get_data_quality", "data": qa})
        tools_called.append("get_data_quality")

        alert_summary = {}
        alert_manager = ctx.get("alert_manager") or self._state.get("alert_manager")
        if alert_manager is not None:
            try:
                alert_summary = alert_manager.summary()
                evidence.append({"tool": "alert_summary", "data": alert_summary})
                tools_called.append("alert_summary")
            except Exception:  # noqa: BLE001
                pass

        well_meta = ctx.get("well_meta", {})

        answer = self._format(question, well_meta, rig, kpis, anom, qa, alert_summary)
        return AgentResult(
            agent=self.name,
            intent=self.intent,
            answer=answer,
            tools_called=tools_called,
            citations=citations,
            evidence=evidence,
            confidence=0.85,
        )

    def _format(
        self,
        question: str,
        well: dict,
        rig: dict,
        kpis: dict,
        anom: dict,
        qa: dict,
        alerts: dict,
    ) -> str:
        lines = ["### Shift / DDR Summary"]
        if well:
            lines.append(
                f"**Well**: {well.get('well', '—')} · **Field**: {well.get('field', '—')} · "
                f"**Operator**: {well.get('operator', '—')}"
            )

        # Time-in-state breakdown — the core of a morning report
        breakdown = rig.get("breakdown", {}) if isinstance(rig, dict) else {}
        if breakdown:
            lines.append("\n**Time in state**")
            for s, info in sorted(breakdown.items(), key=lambda x: -x[1]["pct"])[:8]:
                lines.append(f"- {s}: {info['pct']}% ({info['count']} samples)")

        # KPIs
        kp = kpis.get("kpis", {}) if isinstance(kpis, dict) else {}
        if kp:
            lines.append("\n**Drilling KPIs (interval)**")
            for name in ("mse", "d_exponent", "d_exponent_corrected"):
                obj = kp.get(name)
                if isinstance(obj, dict):
                    lines.append(
                        f"- {name}: current {obj.get('current')} · mean {obj.get('mean')} · "
                        f"min {obj.get('min')} / max {obj.get('max')}"
                    )

        # Anomalies
        if isinstance(anom, dict):
            lines.append(
                f"\n**Anomalies**: {anom.get('total_events', 0)} total, "
                f"latest score = {anom.get('score', 0.0)}"
            )
            evts = anom.get("recent_events", [])
            for e in evts[:5]:
                lines.append(
                    f"  - [{e['severity'].upper()}] {e['type']} at {e['timestamp']} — {e['description'][:120]}"
                )

        # Open alerts
        if alerts:
            lines.append(
                f"\n**Open alerts**: {alerts.get('active', 0)} "
                f"(by severity: { dict(alerts.get('by_severity', {})) })"
            )

        # Data quality footer
        if isinstance(qa, dict):
            lines.append(
                "\n**Data quality** — "
                f"rows: {qa.get('total_rows', '?')} | "
                f"gaps: {qa.get('time_gaps', '?')} | "
                f"spikes: {qa.get('spikes_detected', '?')} | "
                f"flatlines: {qa.get('flatline_segments', '?')} | "
                f"sparse cols: {qa.get('sparse_columns', '?')}"
            )

        return "\n".join(lines)
