"""
DrillMind — BM25 Index (in-memory)
==================================
Lexical retrieval over the same DDR chunks ChromaDB stores semantically.
This complements the vector store: BM25 is unbeatable for exact-phrase
hits like "lost circulation" or "13⅜\" casing shoe set at 822 m".

We use :mod:`rank_bm25` (no native deps, pure Python). If the package is
not installed, the index transparently falls back to a degraded
"contains" search so the API still returns something useful.

The index is built from the same :class:`DDRChunk` objects that flow
into the vector store, so the two retrievers stay perfectly aligned and
reciprocal-rank-fusion produces stable results.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from loguru import logger

from drillmind.rag.chunker import DDRChunk


# ---------------------------------------------------------------------------
# Tokenisation — drilling-aware
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/]*")

# Tiny domain stop list: common English connectives. We DO NOT drop short
# drilling acronyms ("BOP", "ROP", "WOB", "MD", "TVD", "LWD", "MWD").
_STOPWORDS = {
    "a", "an", "and", "or", "the", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "is", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "as", "it", "its",
}


def tokenize(text: str) -> list[str]:
    """Lowercase + drilling-aware token extraction."""
    if not text:
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BM25Result:
    chunk_id: str
    text: str
    score: float
    well_name: str
    report_index: int
    chunk_type: str
    operations: str
    source: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(float(self.score), 4),
            "well_name": self.well_name,
            "report_index": self.report_index,
            "chunk_type": self.chunk_type,
            "operations": self.operations,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class BM25Index:
    """In-memory BM25 index over DDR chunks.

    The index is rebuilt from scratch each time :py:meth:`build` is
    called. For a corpus of ~6 000 chunks this takes <1 s on a laptop,
    so we don't bother persisting it — it can be recreated at startup
    from the same data that feeds ChromaDB.
    """

    def __init__(self) -> None:
        self._chunks: list[DDRChunk] = []
        self._bm25 = None
        self._tokens: list[list[str]] = []
        self._using_fallback = False

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: Sequence[DDRChunk]) -> None:
        """Build (or rebuild) the index from a list of chunks."""
        if not chunks:
            logger.warning("BM25Index.build called with empty chunk list")
            self._chunks = []
            self._tokens = []
            self._bm25 = None
            return

        t0 = time.time()
        self._chunks = list(chunks)
        self._tokens = [tokenize(c.text) for c in self._chunks]

        try:
            from rank_bm25 import BM25Okapi  # noqa: WPS433 — optional dep
            self._bm25 = BM25Okapi(self._tokens)
            self._using_fallback = False
            logger.info(
                "BM25 index built: {} chunks tokenised in {:.2f}s",
                len(self._chunks), time.time() - t0,
            )
        except ImportError:
            self._bm25 = None
            self._using_fallback = True
            logger.warning(
                "rank_bm25 not installed — BM25 retrieval will use a "
                "naive containment fallback. `pip install rank-bm25` "
                "for proper lexical scoring."
            )

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        if not self._chunks or not query.strip():
            return []

        if self._bm25 is not None:
            return self._search_bm25(query, top_k)
        return self._search_fallback(query, top_k)

    # ---- internal --------------------------------------------------------

    def _search_bm25(self, query: str, top_k: int) -> list[BM25Result]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        # Pick top-k indices
        idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[: max(top_k, 1)]
        out: list[BM25Result] = []
        for i in idxs:
            if scores[i] <= 0:
                continue
            chunk = self._chunks[i]
            source = (
                f"DDR #{chunk.report_index} — Well: {chunk.well_name} — Type: {chunk.chunk_type}"
            )
            if chunk.operations:
                source += f" — Ops: {','.join(chunk.operations)}"
            out.append(BM25Result(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(scores[i]),
                well_name=chunk.well_name,
                report_index=chunk.report_index,
                chunk_type=chunk.chunk_type,
                operations=",".join(chunk.operations) if chunk.operations else "",
                source=source,
            ))
        return out

    def _search_fallback(self, query: str, top_k: int) -> list[BM25Result]:
        q = query.lower()
        q_tokens = tokenize(query)
        matches: list[tuple[float, int]] = []
        for i, chunk in enumerate(self._chunks):
            text_lower = chunk.text.lower()
            score = 0.0
            for t in q_tokens:
                if not t:
                    continue
                if t in text_lower:
                    score += 1.0
            if q in text_lower:
                score += 2.0
            if score > 0:
                matches.append((score, i))
        matches.sort(reverse=True)
        out: list[BM25Result] = []
        for score, i in matches[:top_k]:
            chunk = self._chunks[i]
            source = (
                f"DDR #{chunk.report_index} — Well: {chunk.well_name} — Type: {chunk.chunk_type}"
            )
            if chunk.operations:
                source += f" — Ops: {','.join(chunk.operations)}"
            out.append(BM25Result(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),
                well_name=chunk.well_name,
                report_index=chunk.report_index,
                chunk_type=chunk.chunk_type,
                operations=",".join(chunk.operations) if chunk.operations else "",
                source=source,
            ))
        return out
