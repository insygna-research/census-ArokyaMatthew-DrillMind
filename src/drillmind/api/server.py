"""
DrillMind REST API
==================
FastAPI backend that serves real, replay-driven drilling telemetry from
the Equinor Volve field. The API includes:

* Health, readiness, liveness, and Prometheus metrics endpoints
  (``/health``, ``/ready``, ``/live``, ``/metrics``).
* Request-ID middleware for tracing.
* SQLite-backed alert pipeline with WebSocket broadcast.
* Multi-agent + tool-calling copilot with deterministic fallback.
* Hybrid RAG (BM25 + ChromaDB) with Reciprocal Rank Fusion.
* Configurable replay speed up to 1000×.

The architecture is deliberately simple: one process, one FastAPI app,
one in-memory dataset.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from drillmind.agents.multi_orchestrator import MultiAgentOrchestrator
from drillmind.agents.orchestrator import AgentOrchestrator
from drillmind.alerts import AlertBroadcaster, AlertManager, AlertStore
from drillmind.api.health import (
    mark_all_ready,
    mark_ready,
    readiness_state,
    reset_readiness,
    router as health_router,
)
from drillmind.api.middleware import MetricsAndLoggingMiddleware, RequestIdMiddleware
from drillmind.config import get_project_root, get_settings
from drillmind.copilot.engine import CopilotEngine
from drillmind.data.quality import run_quality_check
from drillmind.logging_config import configure_logging
from drillmind.models.anomaly_detection import (
    AutoencoderConfig,
    AutoencoderDetector,
    EnsembleDetector,
    IsolationForestConfig,
    IsolationForestDetector,
)
from drillmind.models.drilling_kpis import compute_drilling_kpis
from drillmind.models.event_classifier import classify_anomalies
from drillmind.models.feature_engineering import build_feature_matrix
from drillmind.models.rig_state import classify_rig_state, compute_state_transitions
from drillmind.observability import (
    inc_alert,
    observe_query,
    record_anomaly_score,
    register_app_info,
    set_websocket_clients,
)
from drillmind.parsers.production_parser import load_production_data
from drillmind.parsers.time_log_parser import load_time_log
from drillmind.rag.retriever import HybridRetriever, build_hybrid_retriever


# ---------------------------------------------------------------------------
# Application state (loaded once at startup)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}

# WebSocket subscriber tracking for /ws/alerts and /ws/stream
_alert_ws_clients: set[WebSocket] = set()
_stream_ws_clients: set[WebSocket] = set()


def _serialize_value(val: Any) -> Any:
    """Safely serialize numpy/pandas types to JSON-compatible Python types."""
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return None if np.isnan(val) else round(float(val), 6)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, pd.Timestamp):
        return str(val)
    if pd.isna(val):
        return None
    return val


def _df_to_records(df: pd.DataFrame, max_rows: int = 10000) -> list[dict]:
    """Convert a DataFrame to a list of dicts with safe serialization."""
    records = []
    for idx, row in df.head(max_rows).iterrows():
        record = {"timestamp": str(idx)} if isinstance(idx, pd.Timestamp) else {"index": idx}
        for col in df.columns:
            record[col] = _serialize_value(row[col])
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: C901 — startup pipeline is by nature sequential
    """Load data, train (or restore) models, then mark the API ready."""
    configure_logging()
    logger.info("=== DrillMind API starting ===")
    settings = get_settings()
    register_app_info(version="0.4.0", well=settings.well, field_name=settings.field_name)
    reset_readiness()

    # ----------- Time log -------------------------------------------------
    max_rows = int(os.environ.get("DRILLMIND_MAX_ROWS", "0")) or None
    logger.info("Loading time log... (max_rows={})", max_rows or "ALL")
    try:
        time_df = load_time_log(nrows=max_rows)
    except FileNotFoundError as e:
        logger.error("Time log not found: {}", e)
        # Allow process to come up so /health/metrics still work; ready stays false.
        _state["time_df"] = pd.DataFrame()
        _state["startup_error"] = str(e)
        yield
        return
    _state["time_df"] = time_df
    mark_ready("time_log_loaded", True)
    logger.info("Time log: {} rows", len(time_df))

    # ----------- Features -------------------------------------------------
    logger.info("Building feature matrix...")
    features = build_feature_matrix(time_df)
    _state["features"] = features
    mark_ready("features_built", True)
    logger.info("Features: {} rows x {} cols", *features.shape)

    # ----------- Models (load or train) -----------------------------------
    model_dir = Path(settings.data.processed_dir).parent / "models"
    force_retrain = os.environ.get("DRILLMIND_RETRAIN", "0") == "1"
    checkpoint_exists = (model_dir / "autoencoder_weights.pt").exists()

    if checkpoint_exists and not force_retrain:
        logger.info("Loading saved models from {} ...", model_dir)
        ae = AutoencoderDetector(AutoencoderConfig())
        ae.load(model_dir)
        _state["ae"] = ae

        ifo = IsolationForestDetector(IsolationForestConfig())
        ifo.load(model_dir)
        _state["ifo"] = ifo

        ensemble = EnsembleDetector(ae, ifo)
        ensemble.load(model_dir)
        _state["ensemble"] = ensemble

        try:
            from drillmind.models.lstm_detector import LSTMConfig, LSTMDetector
            if (model_dir / "lstm_weights.pt").exists():
                lstm = LSTMDetector(LSTMConfig())
                lstm.load(model_dir)
                _state["lstm"] = lstm
        except Exception as e:  # noqa: BLE001
            logger.warning("LSTM load failed (non-fatal): {}", e)

        logger.info("All models loaded from checkpoint")
    else:
        if force_retrain:
            logger.info("DRILLMIND_RETRAIN=1 — forcing full retrain")

        logger.info("Training autoencoder...")
        ae = AutoencoderDetector(AutoencoderConfig(epochs=30, batch_size=512))
        ae.fit(features)
        _state["ae"] = ae

        logger.info("Training isolation forest...")
        ifo = IsolationForestDetector(IsolationForestConfig(n_estimators=200))
        ifo.fit(features)
        _state["ifo"] = ifo

        logger.info("Calibrating ensemble...")
        ensemble = EnsembleDetector(ae, ifo)
        ensemble.calibrate(features)
        _state["ensemble"] = ensemble

        try:
            from drillmind.models.lstm_detector import LSTMConfig, LSTMDetector
            logger.info("Training LSTM temporal model...")
            lstm = LSTMDetector(LSTMConfig(seq_len=60, epochs=20))
            lstm_result = lstm.fit(time_df)
            if lstm_result.get("status") == "trained":
                lstm_scores = lstm.score(time_df)
                offset = len(time_df) - len(features)
                lstm_aligned = lstm_scores[offset:] if offset > 0 else lstm_scores[: len(features)]
                ensemble.attach_lstm(lstm_aligned)
                ensemble.calibrate(features)
                _state["lstm"] = lstm
        except Exception as e:  # noqa: BLE001
            logger.warning("LSTM training failed (non-fatal, using AE+IF only): {}", e)

        logger.info("Saving models to {} ...", model_dir)
        ae.save(model_dir)
        ifo.save(model_dir)
        ensemble.save(model_dir)
        if "lstm" in _state:
            _state["lstm"].save(model_dir)
        logger.info("Models saved — next startup will load from checkpoint")
    mark_ready("models_ready", True)

    # ----------- Score and classify --------------------------------------
    logger.info("Scoring and classifying anomalies...")
    details = ensemble.score_with_details(features)
    _state["anomaly_details"] = details

    events = classify_anomalies(
        features=features,
        anomaly_scores=details["combined"],
        anomaly_mask=details["is_anomaly"],
    )
    _state["events"] = events
    mark_ready("anomaly_pipeline_ready", True)

    # ----------- Rig state, KPIs, quality --------------------------------
    logger.info("Classifying rig states...")
    rig_states = classify_rig_state(time_df)
    _state["rig_states"] = rig_states
    _state["transitions"] = compute_state_transitions(rig_states)
    mark_ready("rig_state_ready", True)

    logger.info("Computing drilling KPIs...")
    _state["kpi_df"] = compute_drilling_kpis(time_df)
    mark_ready("kpis_ready", True)

    logger.info("Running data quality check...")
    _state["quality_report"] = run_quality_check(time_df)
    mark_ready("quality_report_ready", True)

    # ----------- Optional secondary data ---------------------------------
    try:
        _state["production_df"] = load_production_data()
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load production data: {}", e)
        _state["production_df"] = None

    try:
        from drillmind.parsers.depth_log_parser import load_depth_log
        _state["depth_df"] = load_depth_log()
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load depth log (non-fatal): {}", e)
        _state["depth_df"] = None

    try:
        from drillmind.parsers.rop_parser import load_rop_data
        _state["rop_df"] = load_rop_data()
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load ROP data (non-fatal): {}", e)
        _state["rop_df"] = None

    # ----------- DDR + RAG (vector + BM25 hybrid) ------------------------
    try:
        from drillmind.parsers.ddr_parser import load_ddrs_from_huggingface
        from drillmind.rag.chunker import chunk_all_ddrs
        from drillmind.rag.store import DDRVectorStore

        logger.info("Loading DDRs from HuggingFace...")
        ddrs = load_ddrs_from_huggingface()
        chunks = chunk_all_ddrs(ddrs)

        rag_store = DDRVectorStore(persist_dir="data/chromadb")
        if rag_store.count == 0:
            logger.info("Indexing DDR chunks into ChromaDB...")
            rag_store.index_chunks(chunks)
        else:
            logger.info("ChromaDB already has {} docs, skipping re-index", rag_store.count)
        _state["rag_store"] = rag_store
        _state["ddrs"] = ddrs
        _state["ddr_chunks"] = chunks

        # Build BM25 + hybrid retriever
        hybrid = build_hybrid_retriever(vector_store=rag_store, chunks=chunks)
        _state["hybrid_retriever"] = hybrid
        logger.info("Hybrid retriever ready (vector + BM25, RRF fusion)")
    except Exception as e:  # noqa: BLE001
        logger.warning("DDR/RAG initialisation failed (non-fatal): {}", e)
        _state["rag_store"] = None
        _state["hybrid_retriever"] = HybridRetriever(vector_store=None, bm25_index=None)
        _state["ddrs"] = []
        _state["ddr_chunks"] = []
    mark_ready("rag_store_ready", True)

    # ----------- Alert store + manager + broadcaster ---------------------
    alert_db_path = Path(settings.data.processed_dir).parent / "alerts.db"
    alert_store = AlertStore(db_path=alert_db_path)
    alert_broadcaster = AlertBroadcaster()
    alert_manager = AlertManager(
        store=alert_store,
        broadcaster=alert_broadcaster,
        dedup_window_seconds=120,
    )
    _state["alert_store"] = alert_store
    _state["alert_manager"] = alert_manager

    # Subscribe to broadcaster — fan-out to /ws/alerts subscribers
    async def _broadcast_to_ws(payload: dict):
        await _fanout_alert(payload)

    await alert_broadcaster.subscribe(_broadcast_to_ws)

    # Seed alerts from the offline-detected events (one-time)
    if events:
        try:
            seeded = await alert_manager.bulk_seed_from_events(events)
            logger.info("Seeded {} historical alerts from classified events", seeded)
        except Exception as e:  # noqa: BLE001
            logger.warning("Alert seeding failed (non-fatal): {}", e)
    mark_ready("alert_store_ready", True)

    # ----------- Copilot + Agents ----------------------------------------
    copilot_provider = os.environ.get("DRILLMIND_LLM_PROVIDER", "fallback")
    copilot_model = os.environ.get("DRILLMIND_LLM_MODEL", None)
    _state["copilot"] = CopilotEngine(provider=copilot_provider, model=copilot_model)

    # Well metadata for agents
    _state["well_meta"] = {
        "well": settings.well,
        "field": settings.field_name,
        "operator": settings.operator,
    }
    mark_ready("agents_ready", True)

    # ----------- Final readiness ------------------------------------------
    mark_all_ready()
    logger.info("=== DrillMind API ready (state={}) ===", readiness_state())
    yield
    logger.info("=== DrillMind API shutting down ===")
    # Drain WebSocket clients
    for ws in list(_alert_ws_clients):
        try:
            await ws.close(code=1001)
        except Exception:  # noqa: BLE001
            pass
    for ws in list(_stream_ws_clients):
        try:
            await ws.close(code=1001)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DrillMind API",
    description="Real-time drilling analytics and monitoring API",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MetricsAndLoggingMiddleware)
app.add_middleware(RequestIdMiddleware)

app.include_router(health_router)


# ---------------------------------------------------------------------------
# Exception handlers — always return a JSON envelope with the request id
# ---------------------------------------------------------------------------
from fastapi import Request as _FRequest

@app.exception_handler(HTTPException)
async def _http_exc_handler(request: _FRequest, exc: HTTPException):
    rid = getattr(request.state, "request_id", "-")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status": exc.status_code,
            "request_id": rid,
            "path": request.url.path,
        },
        headers={"x-request-id": rid},
    )


@app.exception_handler(Exception)
async def _unhandled_handler(request: _FRequest, exc: Exception):
    rid = getattr(request.state, "request_id", "-")
    logger.exception("Unhandled error rid={} path={}: {}", rid, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "status": 500,
            "request_id": rid,
            "path": request.url.path,
        },
        headers={"x-request-id": rid},
    )


# Serve dashboard static files
_dashboard_path = get_project_root() / "dashboard"
if _dashboard_path.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_dashboard_path), html=True), name="dashboard")


# ===========================================================================
# Well info
# ===========================================================================
@app.get("/api/well/info")
async def well_info():
    """Return well metadata."""
    settings = get_settings()
    time_df: pd.DataFrame = _state.get("time_df", pd.DataFrame())
    return {
        "well": settings.well,
        "field": settings.field_name,
        "operator": settings.operator,
        "total_rows": len(time_df),
        "time_start": str(time_df.index.min()) if len(time_df) else None,
        "time_end": str(time_df.index.max()) if len(time_df) else None,
        "columns": list(time_df.columns),
        "n_features": _state.get("features", pd.DataFrame()).shape[1] if _state.get("features") is not None else 0,
        "n_events": len(_state.get("events", [])),
        "max_replay_speed": int(settings.replay.speed_multiplier_max if hasattr(settings.replay, "speed_multiplier_max") else 1000),
    }


# ===========================================================================
# Time-series data
# ===========================================================================
@app.get("/api/data/timeseries")
async def get_timeseries(
    start: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
    columns: str = Query(None),
):
    time_df: pd.DataFrame = _state["time_df"]
    end = min(start + limit, len(time_df))
    subset = time_df.iloc[start:end]
    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip() in time_df.columns]
        if cols:
            subset = subset[cols]
    return {
        "start": start,
        "end": end,
        "total": len(time_df),
        "columns": list(subset.columns),
        "data": _df_to_records(subset),
    }


@app.get("/api/data/timedepth")
async def time_depth(
    start: int = Query(0, ge=0),
    limit: int = Query(5000, ge=10, le=50000),
):
    """Time vs depth chart data (the staple RTOC plot)."""
    time_df: pd.DataFrame = _state["time_df"]
    end = min(start + limit, len(time_df))
    candidate_cols = ["bit_depth", "hole_depth_md", "tvd"]
    cols = [c for c in candidate_cols if c in time_df.columns]
    subset = time_df.iloc[start:end][cols]
    points = []
    # Downsample for plot performance: every Nth row to cap at ~2000 points
    step = max(1, len(subset) // 2000)
    for idx, row in subset.iloc[::step].iterrows():
        points.append({
            "timestamp": str(idx),
            **{c: _serialize_value(row[c]) for c in cols},
        })
    return {"start": start, "end": end, "total": len(time_df), "columns": cols, "data": points}


# ===========================================================================
# Anomaly Events / Scores / Summary
# ===========================================================================
@app.get("/api/anomalies/events")
async def get_anomaly_events(
    severity: str = Query(None),
    event_type: str = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    events = _state.get("events", [])
    if severity:
        events = [e for e in events if e.severity.value == severity]
    if event_type:
        events = [e for e in events if e.event_type.value == event_type]
    return {"total": len(events), "events": [e.to_dict() for e in events[:limit]]}


@app.get("/api/anomalies/scores")
async def get_anomaly_scores(start: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=10000)):
    details = _state["anomaly_details"]
    features: pd.DataFrame = _state["features"]
    end = min(start + limit, len(features))
    scores = []
    for i in range(start, end):
        scores.append({
            "timestamp": str(features.index[i]),
            "combined": round(float(details["combined"][i]), 4),
            "autoencoder": round(float(details["autoencoder_norm"][i]), 4),
            "isolation_forest": round(float(details["isolation_forest_norm"][i]), 4),
            "is_anomaly": int(details["is_anomaly"][i]),
        })
    return {"start": start, "end": end, "total": len(features), "scores": scores}


@app.get("/api/anomalies/summary")
async def get_anomaly_summary():
    from collections import Counter
    events = _state.get("events", [])
    details = _state["anomaly_details"]
    type_counts = Counter(e.event_type.value for e in events)
    severity_counts = Counter(e.severity.value for e in events)
    return {
        "total_events": len(events),
        "total_anomalous_samples": int(details["is_anomaly"].sum()),
        "total_samples": len(details["is_anomaly"]),
        "anomaly_rate": round(float(details["is_anomaly"].mean()), 4),
        "by_type": dict(type_counts),
        "by_severity": dict(severity_counts),
    }


# ===========================================================================
# Data quality
# ===========================================================================
@app.get("/api/quality/report")
async def get_quality_report():
    report = _state["quality_report"]
    return {
        "total_rows": report.total_rows,
        "total_columns": report.total_columns,
        "time_range_start": str(report.time_range_start),
        "time_range_end": str(report.time_range_end),
        "n_gaps": len(report.gaps),
        "n_spikes": len(report.spikes),
        "n_flatlines": len(report.flatlines),
        "n_sparse_columns": len(report.sparse_columns),
        "sparse_columns": report.sparse_columns,
        "gaps": [
            {"start": str(g.start), "end": str(g.end), "duration_seconds": g.duration_seconds}
            for g in report.gaps[:50]
        ],
    }


# ===========================================================================
# Production / Depth / ROP
# ===========================================================================
@app.get("/api/data/production")
async def get_production_data(well: str = Query(None), limit: int = Query(500, ge=1, le=5000)):
    prod_df = _state.get("production_df")
    if prod_df is None:
        return JSONResponse(status_code=404, content={"error": "Production data not loaded"})
    if well:
        prod_df = prod_df[prod_df["wellbore_code"].str.contains(well, case=False, na=False)]
    return {
        "total": len(prod_df),
        "wells": list(prod_df["wellbore_code"].unique()) if "wellbore_code" in prod_df.columns else [],
        "data": _df_to_records(prod_df.head(limit)),
    }


@app.get("/api/data/depth")
async def get_depth_data(limit: int = Query(500, ge=1, le=5000)):
    depth_df = _state.get("depth_df")
    if depth_df is None:
        return JSONResponse(status_code=404, content={"error": "Depth log not loaded"})
    return {
        "total": len(depth_df),
        "columns": list(depth_df.columns),
        "depth_range": {"min": round(float(depth_df.index.min()), 2), "max": round(float(depth_df.index.max()), 2)},
        "data": _df_to_records(depth_df.head(limit)),
    }


@app.get("/api/data/rop")
async def get_rop_data(limit: int = Query(500, ge=1, le=5000)):
    rop_df = _state.get("rop_df")
    if rop_df is None:
        return JSONResponse(status_code=404, content={"error": "ROP data not loaded"})
    return {
        "total": len(rop_df),
        "columns": list(rop_df.columns),
        "depth_range": {"min": round(float(rop_df.index.min()), 2), "max": round(float(rop_df.index.max()), 2)},
        "data": _df_to_records(rop_df.head(limit)),
    }


# ===========================================================================
# Formation tops (derived from gamma-ray of the depth log)
# ===========================================================================
@app.get("/api/well/formations")
async def get_formations():
    """Return formation tops derived from the depth-indexed gamma-ray log.

    We do NOT hard-code the Volve geology. Instead we look at the depth
    log (if available) and identify likely formation tops as inflection
    points in the gamma-ray channel. Output is a list of tops with
    depths in m MD. This stays faithful to the "no synthetic data" rule
    — every value comes from a real LWD measurement.
    """
    depth_df = _state.get("depth_df")
    if depth_df is None or depth_df.empty:
        return {"tops": []}

    # Find a gamma-ray column heuristically
    gr_col = None
    for c in depth_df.columns:
        if "gamma" in c.lower() or c.lower().startswith("gr"):
            gr_col = c
            break
    if gr_col is None:
        return {"tops": []}

    df = depth_df[[gr_col]].dropna().sort_index()
    if len(df) < 20:
        return {"tops": []}

    # Rolling z-score to find inflection points
    win = max(5, len(df) // 40)
    rolling = df[gr_col].rolling(win, min_periods=1).mean()
    diff = rolling.diff().abs()
    threshold = float(diff.quantile(0.9))
    candidates = diff[diff > threshold].index.tolist()
    # De-duplicate candidates that are within 50 m of each other
    tops = []
    last = -1e9
    for d in candidates:
        if d - last > 50:
            tops.append({"name": f"Top @ {round(float(d), 1)} m MD", "depth_md": round(float(d), 1)})
            last = float(d)
        if len(tops) >= 12:
            break
    return {"tops": tops, "source": gr_col}


# ===========================================================================
# WebSocket — real-time streaming (replay with configurable speed up to 1000x)
# ===========================================================================
@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """Stream drilling data in real-time via WebSocket.

    The client may send JSON control messages: ``{"action":"set_speed","speed":1000}``,
    ``{"action":"pause"}``, ``{"action":"resume"}``, ``{"action":"seek","index":12345}``.
    Speed is clamped to ``[1, 1000]``.
    """
    await ws.accept()
    _stream_ws_clients.add(ws)
    set_websocket_clients(len(_stream_ws_clients))

    time_df: pd.DataFrame = _state["time_df"]
    details = _state["anomaly_details"]
    features: pd.DataFrame = _state["features"]
    settings = get_settings()

    state = {
        "speed": int(settings.replay.speed_multiplier),
        "paused": False,
        "i": 0,
    }
    logger.info("WebSocket /ws/stream connected (active={}, speed={}x)", len(_stream_ws_clients), state["speed"])

    async def _control_loop():
        try:
            while True:
                msg = await ws.receive_json()
                action = msg.get("action")
                if action == "set_speed":
                    raw = msg.get("speed", state["speed"])
                    try:
                        v = int(raw)
                        state["speed"] = max(1, min(v, 1000))
                    except (TypeError, ValueError):
                        pass
                elif action == "pause":
                    state["paused"] = True
                elif action == "resume":
                    state["paused"] = False
                elif action == "seek":
                    try:
                        idx = int(msg.get("index", 0))
                        state["i"] = max(0, min(idx, len(time_df) - 1))
                    except (TypeError, ValueError):
                        pass
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001
            return

    control_task = asyncio.create_task(_control_loop())
    try:
        await ws.send_json({
            "type": "meta",
            "well": settings.well,
            "total_rows": len(time_df),
            "speed": state["speed"],
            "max_speed": 1000,
        })

        while state["i"] < len(time_df):
            if state["paused"]:
                await asyncio.sleep(0.1)
                continue
            i = state["i"]
            row = time_df.iloc[i]
            data: dict[str, Any] = {
                "type": "data",
                "index": i,
                "timestamp": str(time_df.index[i]),
            }
            for col in time_df.columns:
                data[col] = _serialize_value(row[col])

            offset = len(time_df) - len(features)
            feat_idx = i - offset
            if 0 <= feat_idx < len(details["combined"]):
                score = float(details["combined"][feat_idx])
                data["anomaly_score"] = round(score, 4)
                data["is_anomaly"] = int(details["is_anomaly"][feat_idx])
                record_anomaly_score(score)

            await ws.send_json(data)

            # Compute sleep from time delta + current speed
            if i + 1 < len(time_df):
                delta = (time_df.index[i + 1] - time_df.index[i]).total_seconds()
                delta = max(0.0, min(delta, 60.0))
                sleep = delta / max(1, state["speed"])
                if sleep > 0:
                    await asyncio.sleep(sleep)
            state["i"] += 1

        await ws.send_json({"type": "control", "status": "completed"})

    except WebSocketDisconnect:
        logger.info("WebSocket /ws/stream client disconnected")
    except Exception as e:  # noqa: BLE001
        logger.error("WebSocket /ws/stream error: {}", e)
    finally:
        control_task.cancel()
        _stream_ws_clients.discard(ws)
        set_websocket_clients(len(_stream_ws_clients))


# ===========================================================================
# Rig State
# ===========================================================================
@app.get("/api/rig/state")
async def get_rig_state_endpoint(start: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=10000)):
    states = _state["rig_states"]
    end = min(start + limit, len(states))
    time_df = _state["time_df"]
    data = []
    for i in range(start, end):
        val = states.iloc[i]
        data.append({
            "timestamp": str(time_df.index[i]),
            "state": val.value if hasattr(val, "value") else str(val),
        })
    return {"start": start, "end": end, "total": len(states), "data": data}


@app.get("/api/rig/summary")
async def get_rig_summary():
    from collections import Counter
    states = _state["rig_states"]
    counts = Counter(s.value if hasattr(s, "value") else str(s) for s in states)
    total = len(states)
    return {
        "total_samples": total,
        "states": {
            state: {"count": count, "pct": round(100 * count / total, 2)}
            for state, count in sorted(counts.items(), key=lambda x: -x[1])
        },
    }


@app.get("/api/rig/transitions")
async def get_rig_transitions(limit: int = Query(100, ge=1, le=1000)):
    trans = _state["transitions"]
    records = []
    for _, row in trans.tail(limit).iterrows():
        records.append({
            "start": str(row["start"]),
            "end": str(row["end"]),
            "state": row["state"],
            "duration_samples": int(row["duration_samples"]),
            "duration_seconds": float(row["duration_seconds"]),
        })
    return {"total": len(trans), "transitions": records}


# ===========================================================================
# Drilling KPIs
# ===========================================================================
@app.get("/api/kpi/values")
async def get_kpi_values(start: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=10000)):
    kpi_df = _state["kpi_df"]
    end = min(start + limit, len(kpi_df))
    subset = kpi_df.iloc[start:end]
    data = []
    for idx, row in subset.iterrows():
        record = {"timestamp": str(idx)}
        for col in kpi_df.columns:
            val = row[col]
            record[col] = round(float(val), 4) if pd.notna(val) and np.isfinite(val) else None
        data.append(record)
    return {"start": start, "end": end, "total": len(kpi_df), "data": data}


@app.get("/api/kpi/summary")
async def get_kpi_summary():
    kpi_df = _state["kpi_df"]
    summary: dict[str, dict[str, Any]] = {}
    for col in kpi_df.columns:
        series = kpi_df[col].dropna()
        if len(series) == 0:
            summary[col] = {"available": False}
        else:
            finite = series[np.isfinite(series)]
            summary[col] = {
                "available": True,
                "count": len(finite),
                "mean": round(float(finite.mean()), 4),
                "std": round(float(finite.std()), 4),
                "min": round(float(finite.min()), 4),
                "max": round(float(finite.max()), 4),
                "median": round(float(finite.median()), 4),
            }
    return summary


# ===========================================================================
# Copilot / Multi-Agent
# ===========================================================================
class CopilotQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field(default="multi", description="multi | tools | legacy")


def _llm_fn():
    copilot: CopilotEngine | None = _state.get("copilot")
    if copilot and copilot._llm.name != "fallback":
        return copilot._llm.generate
    return None


@app.post("/api/copilot/query")
async def copilot_query(query: CopilotQuery):
    """Process a natural language question.

    Modes:
    * ``multi``   — multi-agent state machine (default; agents + handoffs)
    * ``tools``   — legacy single-loop tool-calling orchestrator
    * ``legacy``  — alias for ``tools``
    * ``copilot`` — grounded single-shot copilot over the full live context
    """
    import time
    if not _state.get("time_df", pd.DataFrame()).shape[0]:
        raise HTTPException(status_code=503, detail="data not loaded")

    t0 = time.time()
    state_for_agents = {
        "time_df":        _state["time_df"],
        "events":         _state["events"],
        "anomaly_details": _state["anomaly_details"],
        "features":       _state["features"],
        "rig_states":     _state["rig_states"],
        "transitions":    _state["transitions"],
        "kpi_df":         _state["kpi_df"],
        "production_df":  _state.get("production_df"),
        "quality_report": _state.get("quality_report"),
        "rag_store":      _state.get("rag_store"),
        "hybrid_retriever": _state.get("hybrid_retriever"),
        "depth_df":       _state.get("depth_df"),
        "rop_df":         _state.get("rop_df"),
        "alert_manager":  _state.get("alert_manager"),
        "well_meta":      _state.get("well_meta", {}),
        "settings":       get_settings(),
    }

    mode = (query.mode or "multi").lower()
    if mode in ("tools", "legacy"):
        agent = AgentOrchestrator(state=state_for_agents, llm_fn=_llm_fn())
        result = await agent.query(query.question)
        copilot: CopilotEngine = _state.get("copilot")
        provider_name = copilot._llm.name if copilot else "fallback"
        model_name = copilot._llm.model_name if copilot else "rule-based-v2"
        elapsed = time.time() - t0
        observe_query(result.intent, elapsed)
        return {
            "answer": result.answer,
            "provider": provider_name,
            "model": model_name,
            "grounded": result.grounded,
            "context_summary": {
                "intent": result.intent,
                "tools_called": result.tools_called,
                "evidence_count": len(result.evidence),
                "total_time_ms": round(result.total_time * 1000),
            },
        }

    if mode in ("copilot", "single", "grounded"):
        copilot: CopilotEngine = _state.get("copilot")
        if copilot is None:
            raise HTTPException(status_code=503, detail="copilot not ready")
        resp = await copilot.query(query.question, state_for_agents)
        elapsed = time.time() - t0
        observe_query("grounded", elapsed)
        return {
            "answer": resp.answer,
            "provider": resp.provider,
            "model": resp.model,
            "grounded": resp.grounded,
            "context_summary": {
                "intent": "grounded",
                "agents_run": ["grounded_copilot"],
                "tools_called": [],
                "anomaly_score": resp.context_summary.get("anomaly_score"),
                "rig_state": resp.context_summary.get("rig_state"),
                "total_events": resp.context_summary.get("total_events"),
                "total_time_ms": round(elapsed * 1000),
            },
        }

    multi = MultiAgentOrchestrator(state=state_for_agents, llm_fn=_llm_fn(), max_hops=2)
    multi_result = await multi.query(query.question)
    elapsed = time.time() - t0
    observe_query(multi_result.intent, elapsed)
    copilot: CopilotEngine = _state.get("copilot")
    return {
        "answer": multi_result.answer,
        "provider": copilot._llm.name if copilot else "fallback",
        "model": copilot._llm.model_name if copilot else "rule-based-v2",
        "grounded": multi_result.grounded,
        "context_summary": {
            "intent": multi_result.intent,
            "agents_run": multi_result.agents_run,
            "tools_called": multi_result.tools_called,
            "citations": multi_result.citations,
            "handoffs": [list(h) for h in multi_result.handoffs],
            "evidence_count": len(multi_result.tools_called),
            "total_time_ms": multi_result.elapsed_ms,
            "confidence": multi_result.confidence,
        },
    }


# ===========================================================================
# RAG (Hybrid) Search
# ===========================================================================
class RAGSearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    well_filter: str | None = None
    mode: str = Field(default="hybrid", description="hybrid | vector | bm25")


@app.post("/api/rag/search")
async def rag_search(search: RAGSearchQuery):
    hybrid: HybridRetriever | None = _state.get("hybrid_retriever")
    if hybrid is None:
        return {"error": "Hybrid retriever not initialized", "results": []}
    results = hybrid.search(
        query=search.query,
        top_k=search.top_k,
        well_filter=search.well_filter,
        mode=search.mode,
    )
    return {
        "query": search.query,
        "mode": search.mode,
        "results": [r.to_dict() for r in results],
        "total_indexed_vector": (_state.get("rag_store").count if _state.get("rag_store") else 0),
    }


# ===========================================================================
# Alerts — CRUD + WebSocket broadcast
# ===========================================================================
class AlertResolveBody(BaseModel):
    actor: str | None = None
    note: str | None = None


def _alert_manager() -> AlertManager:
    am: AlertManager | None = _state.get("alert_manager")
    if am is None:
        raise HTTPException(status_code=503, detail="alert manager not ready")
    return am


@app.get("/api/alerts/active")
async def alerts_active(
    severity: str = Query(None),
    event_type: str = Query(None),
    limit: int = Query(200, ge=1, le=1000),
):
    am = _alert_manager()
    items = am.list_active(severity=severity, event_type=event_type, limit=limit)
    return {"total": len(items), "items": [a.to_dict() for a in items]}


@app.get("/api/alerts/history")
async def alerts_history(
    status: str = Query(None),
    limit: int = Query(200, ge=1, le=1000),
):
    am = _alert_manager()
    items = am.list_history(status=status, limit=limit)
    return {"total": len(items), "items": [a.to_dict() for a in items]}


@app.get("/api/alerts/{alert_id}")
async def alert_get(alert_id: str):
    am = _alert_manager()
    a = am.get(alert_id)
    if a is None:
        raise HTTPException(status_code=404, detail="alert not found")
    return a.to_dict()


@app.get("/api/alerts/summary/overview")
async def alerts_summary_overview():
    return _alert_manager().summary()


@app.post("/api/alerts/{alert_id}/acknowledge")
async def alert_acknowledge(alert_id: str, body: AlertResolveBody):
    am = _alert_manager()
    a = await am.acknowledge(alert_id, actor=body.actor, note=body.note)
    if a is None:
        raise HTTPException(status_code=404, detail="alert not found")
    return a.to_dict()


@app.post("/api/alerts/{alert_id}/resolve")
async def alert_resolve(alert_id: str, body: AlertResolveBody):
    am = _alert_manager()
    a = await am.resolve(alert_id, actor=body.actor, note=body.note)
    if a is None:
        raise HTTPException(status_code=404, detail="alert not found")
    inc_alert(a.severity, a.event_type)
    return a.to_dict()


@app.post("/api/alerts/{alert_id}/suppress")
async def alert_suppress(alert_id: str, body: AlertResolveBody):
    am = _alert_manager()
    a = await am.suppress(alert_id, actor=body.actor, note=body.note)
    if a is None:
        raise HTTPException(status_code=404, detail="alert not found")
    return a.to_dict()


@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    """Broadcasts new alerts as they are created/resolved/acknowledged."""
    await ws.accept()
    _alert_ws_clients.add(ws)
    logger.info("WebSocket /ws/alerts connected (active={})", len(_alert_ws_clients))
    try:
        # Send initial active snapshot
        am: AlertManager | None = _state.get("alert_manager")
        if am is not None:
            snapshot = [a.to_dict() for a in am.list_active(limit=200)]
            await ws.send_json({"type": "snapshot", "items": snapshot})

        # Keep the socket alive — broadcasts happen via _fanout_alert
        while True:
            try:
                # Echo any ping/keep-alive sent by the client
                msg = await ws.receive_text()
                if msg.strip().lower() == "ping":
                    await ws.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        logger.error("WebSocket /ws/alerts error: {}", e)
    finally:
        _alert_ws_clients.discard(ws)


async def _fanout_alert(payload: dict) -> None:
    """Push a single alert payload to all subscribed /ws/alerts clients."""
    dead: list[WebSocket] = []
    for ws in list(_alert_ws_clients):
        try:
            await ws.send_json(payload)
        except Exception:  # noqa: BLE001
            dead.append(ws)
    for d in dead:
        _alert_ws_clients.discard(d)
    try:
        sev = payload.get("alert", {}).get("severity", "unknown")
        etype = payload.get("alert", {}).get("event_type", "unknown")
        if payload.get("event") == "created":
            inc_alert(sev, etype)
    except Exception:  # noqa: BLE001
        pass


# ===========================================================================
# Legacy compatibility — /api/health returns the same shape as v0.3
# ===========================================================================
@app.get("/api/health")
async def health_compat():
    return {
        "status": "ok",
        "version": "0.4.0",
        "features": {
            "anomaly_detection": True,
            "rig_state": True,
            "drilling_kpis": True,
            "rag_ddr": _state.get("rag_store") is not None,
            "hybrid_retriever": _state.get("hybrid_retriever") is not None,
            "agent_orchestrator": True,
            "multi_agent": True,
            "alerts": _state.get("alert_manager") is not None,
            "metrics": True,
        },
    }


@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard/index.html")
