"""
DrillMind — Hybrid Retriever (BM25 + Vector with RRF Fusion)
=============================================================
Combines lexical (BM25) and semantic (ChromaDB cosine) retrievers via
Reciprocal Rank Fusion. The vector store stays the primary retriever
(weight 0.7 by default) and BM25 acts as a secondary lexical booster
(weight 0.3). Either retriever may be missing — the fuser degrades
gracefully and uses whichever side is available.

RRF formula
-----------
    score(d) = Σ_r  w_r / (k + rank_r(d))

We use the standard ``k = 60`` from Cormack et al. (2009).

API surface
-----------
* :class:`HybridRetriever` — drop-in replacement for direct calls to
  :py:meth:`DDRVectorStore.search`. The :func:`search` method returns
  a unified list of :class:`RAGResult` instances (same shape used by
  the rest of the codebase, so the agent tools stay unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from loguru import logger

from drillmind.rag.bm25 import BM25Index, BM25Result
from drillmind.rag.chunker import DDRChunk
from drillmind.rag.store import DDRVectorStore, RAGResult


@dataclass(frozen=True)
class HybridConfig:
    """Weights for Reciprocal Rank Fusion."""
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    rrf_k: int = 60


class HybridRetriever:
    """Vector + BM25 retriever with RRF fusion.

    Parameters
    ----------
    vector_store : DDRVectorStore
        Existing ChromaDB-backed store.
    bm25_index : BM25Index | None
        Lexical index. Build with the same chunks fed to ``vector_store``.
    config : HybridConfig
        Fusion weights and ``k`` constant.
    """

    def __init__(
        self,
        vector_store: DDRVectorStore | None,
        bm25_index: BM25Index | None = None,
        config: HybridConfig | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._bm25 = bm25_index
        self._cfg = config or HybridConfig()

    # ---- public surface --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        well_filter: Optional[str] = None,
        operation_filter: Optional[str] = None,
        chunk_type: Optional[str] = None,
        mode: str = "hybrid",
    ) -> list[RAGResult]:
        """Hybrid search.

        Parameters
        ----------
        query : str
            Free-text query.
        top_k : int
            Number of fused results to return.
        well_filter, operation_filter, chunk_type : str | None
            Metadata filters (only applied to the vector branch — BM25 is
            applied across the whole corpus and post-filtered).
        mode : ``"hybrid" | "vector" | "bm25"``
            Selects which retrievers contribute. Default hybrid.
        """
        if not query.strip():
            return []

        vec_results: list[RAGResult] = []
        bm25_results: list[BM25Result] = []

        if mode in ("hybrid", "vector") and self._vector_store is not None:
            try:
                vec_results = self._vector_store.search(
                    query=query,
                    top_k=max(top_k * 3, 10),  # over-fetch for fusion
                    well_filter=well_filter,
                    operation_filter=operation_filter,
                    chunk_type=chunk_type,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Vector retrieval failed: {}", e)
                vec_results = []

        if mode in ("hybrid", "bm25") and self._bm25 is not None:
            try:
                bm25_raw = self._bm25.search(query, top_k=max(top_k * 3, 10))
                # Apply post-filters
                bm25_results = [r for r in bm25_raw if self._passes_filters(
                    r,
                    well_filter=well_filter,
                    operation_filter=operation_filter,
                    chunk_type=chunk_type,
                )]
            except Exception as e:  # noqa: BLE001
                logger.warning("BM25 retrieval failed: {}", e)
                bm25_results = []

        if mode == "vector":
            return vec_results[:top_k]
        if mode == "bm25":
            return [self._bm25_to_rag(r) for r in bm25_results[:top_k]]

        # Hybrid: RRF
        fused = self._reciprocal_rank_fusion(vec_results, bm25_results)
        return fused[:top_k]

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _passes_filters(
        r: BM25Result,
        well_filter: str | None,
        operation_filter: str | None,
        chunk_type: str | None,
    ) -> bool:
        if well_filter and r.well_name != well_filter:
            return False
        if chunk_type and r.chunk_type != chunk_type:
            return False
        if operation_filter and operation_filter.lower() not in r.operations.lower():
            return False
        return True

    @staticmethod
    def _bm25_to_rag(r: BM25Result) -> RAGResult:
        return RAGResult(
            chunk_id=r.chunk_id,
            text=r.text,
            score=float(r.score),
            well_name=r.well_name,
            report_index=r.report_index,
            chunk_type=r.chunk_type,
            operations=r.operations,
            source=r.source,
        )

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[RAGResult],
        bm25_results: list[BM25Result],
    ) -> list[RAGResult]:
        """Fuse two ranked lists via RRF.

        score(d) = w_vec / (k + rank_vec(d)) + w_bm25 / (k + rank_bm25(d))
        """
        cfg = self._cfg
        scores: dict[str, float] = {}
        items: dict[str, RAGResult] = {}

        for rank, r in enumerate(vector_results, start=1):
            inc = cfg.vector_weight / (cfg.rrf_k + rank)
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + inc
            items[r.chunk_id] = r

        for rank, r in enumerate(bm25_results, start=1):
            inc = cfg.bm25_weight / (cfg.rrf_k + rank)
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + inc
            if r.chunk_id not in items:
                items[r.chunk_id] = self._bm25_to_rag(r)

        ordered = sorted(items.values(), key=lambda x: -scores.get(x.chunk_id, 0.0))
        # Replace the raw distance-based score with the RRF score for downstream display.
        out: list[RAGResult] = []
        for r in ordered:
            r2 = RAGResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=float(scores.get(r.chunk_id, 0.0)),
                well_name=r.well_name,
                report_index=r.report_index,
                chunk_type=r.chunk_type,
                operations=r.operations,
                source=r.source,
            )
            out.append(r2)
        return out


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_hybrid_retriever(
    vector_store: DDRVectorStore | None,
    chunks: Iterable[DDRChunk] | None = None,
) -> HybridRetriever:
    """Convenience constructor that builds the BM25 side from chunks."""
    bm25 = None
    if chunks is not None:
        bm25 = BM25Index()
        bm25.build(list(chunks))
    return HybridRetriever(vector_store=vector_store, bm25_index=bm25)
