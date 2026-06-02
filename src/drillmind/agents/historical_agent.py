"""
DrillMind — Historical / Offset Well Agent
============================================
Owns offset-well comparisons and DDR lookups.

Tools called
------------
* (RAG hybrid)           — DDR semantic + lexical search (RRF fused)
* ``query_production``   — production data per well
* ``compare_wells``      — cross-well numeric comparison
* ``get_depth_log``      — depth-indexed LWD/MWD (only on petrophysics asks)
"""

from __future__ import annotations

from typing import Any

from drillmind.agents.base import AgentResult, BaseAgent
from drillmind.agents.tools import execute_tool


class HistoricalAgent(BaseAgent):
    name = "historical_agent"
    intent = "historical"

    async def _handle(self, question: str, ctx: dict) -> AgentResult:
        tools_called: list[str] = []
        evidence: list[dict] = []
        citations: list[str] = []
        ql = question.lower()

        # Hybrid RAG primary
        hybrid = ctx.get("hybrid_retriever") or self._state.get("hybrid_retriever")
        rag_hits: list[dict] = []
        if hybrid is not None:
            try:
                results = hybrid.search(query=question, top_k=6)
                rag_hits = [r.to_dict() for r in results]
                evidence.append({"tool": "rag_hybrid_search", "data": {"results": rag_hits}})
                tools_called.append("rag_hybrid_search")
                for r in rag_hits:
                    citations.append(r.get("source", ""))
            except Exception:  # noqa: BLE001
                rag_hits = []

        # Production + cross-well comparison
        compare = execute_tool("compare_wells", self._state)
        evidence.append({"tool": "compare_wells", "data": compare})
        tools_called.append("compare_wells")

        prod = execute_tool("query_production", self._state)
        evidence.append({"tool": "query_production", "data": prod})
        tools_called.append("query_production")

        depth_data = None
        if any(k in ql for k in ("gamma", "resistivity", "porosity", "lwd", "mwd", "formation")):
            depth_data = execute_tool("get_depth_log", self._state)
            evidence.append({"tool": "get_depth_log", "data": depth_data})
            tools_called.append("get_depth_log")

        answer = self._format(question, rag_hits, compare, prod, depth_data)
        return AgentResult(
            agent=self.name,
            intent=self.intent,
            answer=answer,
            tools_called=tools_called,
            citations=citations,
            evidence=evidence,
            confidence=0.8,
        )

    def _format(
        self,
        question: str,
        rag_hits: list[dict],
        compare: dict,
        prod: dict,
        depth_data: dict | None,
    ) -> str:
        lines = ["### Historical / Offset-Well Analysis"]

        if rag_hits:
            lines.append("\n**DDR matches (hybrid BM25 + vector, RRF fused)**")
            for r in rag_hits[:6]:
                ssrc = r.get("source", "DDR")
                snippet = (r.get("text", "") or "")[:220].replace("\n", " ")
                lines.append(f"- {ssrc}\n  > {snippet}…")
        else:
            lines.append("\n_No DDR hits — RAG store may not be initialised yet._")

        if isinstance(compare, dict) and compare.get("comparison"):
            lines.append("\n**Cross-well comparison**")
            for well, data in list(compare["comparison"].items())[:8]:
                metrics = data.get("metrics", {})
                metric_str = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:4])
                lines.append(f"- **{well}** ({data.get('records', 0)} rec): {metric_str}")

        if isinstance(prod, dict) and prod.get("wells"):
            lines.append(f"\n**Production wells available**: { ', '.join(prod['wells'][:8]) }")

        if isinstance(depth_data, dict) and depth_data.get("total_rows"):
            dr = depth_data.get("depth_range", {})
            lines.append(
                f"\n**LWD/MWD depth log**: {depth_data['total_rows']} rows, "
                f"{dr.get('min')}–{dr.get('max')} m MD"
            )

        return "\n".join(lines)
