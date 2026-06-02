"""
DrillMind — Multi-Agent Intent Router
======================================
Pluggable intent classifier. The default implementation is a keyword
scorer that matches the heuristics RTOC analysts use when triaging a
question ("kick" → safety, "MSE" → drilling, "DDR" → historical,
"summary" → reporting). The orchestrator can fall back to a LLM-driven
router when a model is configured.

Public surface
--------------
* :class:`Intent` — string enum used across the multi-agent layer.
* :class:`IntentRouter` — ``classify(question) -> tuple[Intent, dict]``
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    DRILLING = "drilling"        # parameter / KPI questions
    SAFETY = "safety"            # well control, kick, lost circ, BOP
    HISTORICAL = "historical"    # offset wells, DDR lookups, mud weight history
    REPORTING = "reporting"      # DDR / shift summaries
    GENERAL = "general"          # everything else


@dataclass(frozen=True)
class RouterDecision:
    intent: Intent
    score: int
    secondary: tuple[Intent, ...] = ()


class IntentRouter:
    """Keyword-scoring intent router.

    The router returns a :class:`RouterDecision` with the primary intent
    plus any secondary intents that scored above zero. The orchestrator
    can use the secondary list to chain a safety agent after a drilling
    agent when the question touches both topics (e.g. "is the MSE high
    enough to risk a stuck pipe?").
    """

    SAFETY_KW = (
        "kick", "well control", "shut in", "shut-in", "kill", "bop",
        "gas influx", "pit gain", "flow check", "loss", "lost circulation",
        "stuck", "blowout", "emergency", "danger", "risk", "pp/fg",
        "pore pressure", "fracture gradient", "lcm",
    )
    DRILLING_KW = (
        "mse", "specific energy", "d-exponent", "d_exp", "rop",
        "rate of penetration", "wob", "rpm", "torque", "spp", "standpipe",
        "flow", "hookload", "ecd", "bha", "bit", "performance", "efficiency",
        "drilling", "circulat", "reaming", "rig state", "doing", "current",
    )
    HISTORICAL_KW = (
        "ddr", "report", "history", "historical", "past",
        "previous", "offset", "compare", "when did", "last time",
        "mud weight", "formation", "casing", "cement", "whipstock",
        "sidetrack", "production", "well comparison",
    )
    REPORTING_KW = (
        "summary", "summarise", "summarize", "report", "brief",
        "overview", "morning report", "shift", "ddr summary",
        "executive", "highlight", "wrap up",
    )

    def classify(self, question: str) -> RouterDecision:
        q = question.lower()

        safety_score = sum(1 for k in self.SAFETY_KW if k in q)
        drilling_score = sum(1 for k in self.DRILLING_KW if k in q)
        historical_score = sum(1 for k in self.HISTORICAL_KW if k in q)
        reporting_score = sum(1 for k in self.REPORTING_KW if k in q)

        candidates = {
            Intent.SAFETY: safety_score,
            Intent.DRILLING: drilling_score,
            Intent.HISTORICAL: historical_score,
            Intent.REPORTING: reporting_score,
        }

        # Safety has priority bias — a single hit is enough to dominate
        # when the next-best intent only has 1 hit too, because a wrong
        # answer on a well-control question is the most expensive class
        # of error in this product.
        if safety_score > 0 and safety_score >= max(candidates.values()) - 1:
            primary = Intent.SAFETY
        else:
            primary = max(candidates, key=lambda i: candidates[i])

        if candidates[primary] == 0:
            return RouterDecision(intent=Intent.GENERAL, score=0)

        secondary = tuple(
            i for i, s in sorted(candidates.items(), key=lambda x: -x[1])
            if i != primary and s > 0
        )

        return RouterDecision(
            intent=primary,
            score=candidates[primary],
            secondary=secondary,
        )
