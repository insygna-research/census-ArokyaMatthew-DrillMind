"""
DrillMind — Drilling Performance Agent
=======================================
Answers parameter / KPI / efficiency questions.

Tools called
------------
* ``get_current_sensors`` — latest WITS values
* ``get_rig_state``       — drilling / circulating / tripping …
* ``get_drilling_kpis``   — MSE, d-exponent, corrected d-exp
* ``get_rop_formation``   — ROP vs petrophysics (only when relevant)

The agent ALWAYS hands off to the SAFETY agent when MSE is well above
typical North-Sea norms (>200 MPa) AND a non-trivial anomaly is active.
"""

from __future__ import annotations

from typing import Any

from drillmind.agents.base import AgentResult, BaseAgent
from drillmind.agents.tools import execute_tool


class DrillingAgent(BaseAgent):
    name = "drilling_agent"
    intent = "drilling"

    HIGH_MSE_MPA = 200.0  # founder-point heuristic for 12¼" section

    async def _handle(self, question: str, ctx: dict) -> AgentResult:
        tools_called: list[str] = []
        evidence: list[dict] = []
        citations: list[str] = []

        sensors = execute_tool("get_current_sensors", self._state)
        evidence.append({"tool": "get_current_sensors", "data": sensors})
        tools_called.append("get_current_sensors")

        rig = execute_tool("get_rig_state", self._state)
        evidence.append({"tool": "get_rig_state", "data": rig})
        tools_called.append("get_rig_state")

        kpis = execute_tool("get_drilling_kpis", self._state)
        evidence.append({"tool": "get_drilling_kpis", "data": kpis})
        tools_called.append("get_drilling_kpis")

        # Only call rop_formation if the question mentions ROP / formation / petrophysics
        ql = question.lower()
        rop_data = None
        if any(k in ql for k in ("rop", "formation", "porosity", "permeab", "vsh", "klogh")):
            rop_data = execute_tool("get_rop_formation", self._state)
            evidence.append({"tool": "get_rop_formation", "data": rop_data})
            tools_called.append("get_rop_formation")

        # Hand-off decision
        handoff = self._decide_handoff(kpis)

        answer = self._format(question, sensors, rig, kpis, rop_data, handoff)
        return AgentResult(
            agent=self.name,
            intent=self.intent,
            answer=answer,
            tools_called=tools_called,
            citations=citations,
            evidence=evidence,
            handoff_to=handoff,
            confidence=0.85,
        )

    # ---------------------------------------------------------------- helpers

    def _decide_handoff(self, kpis: dict) -> str | None:
        try:
            mse_obj = kpis.get("kpis", {}).get("mse")
            if not isinstance(mse_obj, dict):
                return None
            cur = mse_obj.get("current")
            if cur is not None and float(cur) > self.HIGH_MSE_MPA:
                # Founder-point regime — let safety verify drillability
                return "safety_agent"
        except (TypeError, ValueError, AttributeError):
            return None
        return None

    def _format(
        self,
        question: str,
        sensors: dict,
        rig: dict,
        kpis: dict,
        rop_data: dict | None,
        handoff: str | None,
    ) -> str:
        lines = ["### Drilling Performance Analysis"]

        # Current sensor snapshot
        s = sensors.get("sensors", {}) if isinstance(sensors, dict) else {}
        if s:
            lines.append("\n**Live sensors**")
            for k in ("rop", "wob_avg", "rpm_avg", "torque_averaged", "spp", "weight_on_hook", "bit_depth"):
                if k in s and s[k] is not None:
                    lines.append(f"- `{k}` = **{s[k]}**")

        # Rig state
        if isinstance(rig, dict) and rig.get("current_state"):
            lines.append(f"\n**Rig state**: {rig['current_state']}")

        # KPIs
        if isinstance(kpis, dict):
            kp = kpis.get("kpis", {})
            mse = kp.get("mse")
            d_exp = kp.get("d_exponent")
            d_exp_c = kp.get("d_exponent_corrected")
            lines.append("\n**Drilling KPIs**")
            if isinstance(mse, dict):
                cur = mse.get("current")
                mean = mse.get("mean")
                tag = " ⚠ above founder-point heuristic" if cur and float(cur) > self.HIGH_MSE_MPA else ""
                lines.append(f"- MSE (Teale): current **{cur}** MPa · mean {mean} MPa{tag}")
            elif mse is None:
                lines.append("- MSE: not available (no active drilling in the loaded window)")
            if isinstance(d_exp, dict):
                lines.append(f"- d-exponent: current **{d_exp.get('current')}** · mean {d_exp.get('mean')}")
            if isinstance(d_exp_c, dict):
                lines.append(f"- d-exp corrected: current **{d_exp_c.get('current')}** · mean {d_exp_c.get('mean')}")

        # Formation context
        if isinstance(rop_data, dict) and rop_data.get("total_rows"):
            lines.append("\n**ROP / formation overlap**")
            dr = rop_data.get("depth_range", {})
            lines.append(f"- depth range: {dr.get('min')}–{dr.get('max')} m MD over {rop_data['total_rows']} samples")

        # Hand-off
        if handoff == "safety_agent":
            lines.append(
                "\n> _Founder-point conditions detected — flagging the safety agent to "
                "verify drillability and well-control risk._"
            )

        return "\n".join(lines)
