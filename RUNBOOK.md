# DrillMind RUNBOOK — first-time setup → live dashboard

This is the complete, copy-paste guide to run DrillMind v0.4 on a fresh machine.
No prior project knowledge required.

> ⚠ DrillMind runs on **real** drilling telemetry from the Equinor Volve
> field. There is **no** synthetic data generator. You must download the
> dataset once before the first run.

---

## 0. Prerequisites

| Requirement      | Why                                                |
| ---------------- | -------------------------------------------------- |
| **Python 3.11+** | All ML and FastAPI deps target 3.11+               |
| **pip ≥ 23**     | Editable installs                                  |
| **4 GB disk**    | Volve CSV (~408 MB) + ChromaDB index (~150 MB)     |
| **8 GB RAM**     | Models + 419 745-row DataFrame in memory           |
| **CUDA GPU**     | _Optional_ — speeds up first-time training ~10×    |
| **Git**          | Only if cloning; this guide assumes you have the zip |

Windows, macOS, and Linux all work.

---

## 1. Unzip the project

```bash
unzip DrillMind.zip -d ~/DrillMind
cd ~/DrillMind
```

You should now see:

```
DrillMind/
├── config/
├── dashboard/
├── data/
│   └── raw/        # ← we'll fill this in step 3
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
├── RUNBOOK.md
├── scripts/
├── src/
└── tests/
```

---

## 2. Create a virtual environment and install

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]"
```

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[all]"
```

`[all]` pulls in:

* ML stack: PyTorch (CPU build by default — install GPU build separately if needed), scikit-learn
* RAG stack: ChromaDB, sentence-transformers, datasets
* Dev tools: pytest, ruff
* Extra packages: `prometheus_client`, `rank_bm25`

> First install downloads the `all-MiniLM-L6-v2` sentence-transformer
> model (~80 MB) the first time RAG is actually used.

---

## 3. Download the Volve dataset (one-time)

DrillMind reads real CSVs from `data/raw/`. The Volve field data is
free to download from Equinor — DrillMind does **not** redistribute it.

1. Go to **[https://www.equinor.com/energy/volve-data-sharing](https://www.equinor.com/energy/volve-data-sharing)**
2. Accept the Equinor Open Data License.
3. From the **WITSML Realtime Drilling Data** archive, download for
   well `NO 15/9-F-9 A`:

   | File                                   | Size  | Required? |
   | -------------------------------------- | ----- | --------- |
   | `Norway-NA-15_47_9-F-9 A time.csv`     | ~408 MB | **Yes**   |
   | `Norway-NA-15_47_9-F-9 A depth.csv`    | ~5 MB | Optional  |
   | `ROP data .csv`                        | ~10 KB | Optional |
   | `Volve production data.xlsx`           | ~2 MB | Optional |

4. Move all downloaded files into `data/raw/`:

   ```bash
   mkdir -p data/raw
   mv ~/Downloads/"Norway-NA-15_47_9-F-9 A time.csv" data/raw/
   mv ~/Downloads/"Norway-NA-15_47_9-F-9 A depth.csv" data/raw/   # if you have it
   mv ~/Downloads/"ROP data .csv" data/raw/                       # if you have it
   mv ~/Downloads/"Volve production data.xlsx" data/raw/          # if you have it
   ```

5. Verify the filenames match exactly — DrillMind reads them by name:

   ```bash
   ls -lh data/raw/
   ```

The Daily Drilling Reports (DDRs) are auto-loaded from HuggingFace
(`bengsoon/volve_alpaca`) on first run — nothing to do here.

---

## 4. First run — trains models then serves the dashboard

The very first time you start, DrillMind trains three ML models on the
Volve data and saves them to `data/models/`. This takes ~15 min on a
CUDA GPU, ~30–45 min on CPU. After that, every subsequent start is
**~5 seconds** (models are loaded from disk).

### Linux / macOS

```bash
export DRILLMIND_RETRAIN=1                 # force training on the first run
export DRILLMIND_MAX_ROWS=50000            # optional: load only 50K rows for a fast first start
python -m uvicorn drillmind.api.server:app --host 0.0.0.0 --port 8000
```

### Windows (PowerShell)

```powershell
$env:DRILLMIND_RETRAIN = "1"
$env:DRILLMIND_MAX_ROWS = "50000"
python -m uvicorn drillmind.api.server:app --host 0.0.0.0 --port 8000
```

When you see this line you are ready:

```
=== DrillMind API ready ===
```

Open your browser at **[http://localhost:8000](http://localhost:8000)** — the dashboard streams data immediately.

---

## 5. Subsequent runs

After the first run, you do **not** need `DRILLMIND_RETRAIN=1` anymore.
Just:

```bash
source .venv/bin/activate           # (Windows: .\.venv\Scripts\Activate.ps1)
python -m uvicorn drillmind.api.server:app --host 0.0.0.0 --port 8000
```

To use the **full dataset** (419 745 rows) instead of the 50K dev slice:

```bash
unset DRILLMIND_MAX_ROWS            # Linux/macOS
$env:DRILLMIND_MAX_ROWS = $null      # Windows
```

---

## 6. Run via Docker (no Python required)

```bash
docker compose up --build
```

This:
* Builds the container with all CPU dependencies.
* Mounts `data/` and `config/` as volumes so your CSVs are visible inside the container.
* Exposes the API + dashboard on `http://localhost:8000`.
* Configures the health probe (`curl /health`).

> ⚠ The container uses the CPU build of PyTorch and `DRILLMIND_MAX_ROWS=50000` by default — for full-dataset training, increase RAM and unset that variable in `docker-compose.yml`.

---

## 7. The dashboard at a glance

URL: **`http://localhost:8000/dashboard/index.html`** (auto-redirected from `/`).

| Section               | What it shows                                                                                |
| --------------------- | -------------------------------------------------------------------------------------------- |
| **Top bar**           | Well metadata · live timestamp · connection status · replay slider (1–1000×)                |
| **Alarm bar**         | Active alert counters (Critical/High/Medium/Low) + scrolling live ticker                     |
| **KPI strip**         | 11 live tiles with sparklines: Depth, WOB, SPP, Hookload, Torque, RPM, Flow, MSE, Anomaly, Rig State, Bit Status |
| **Time vs. Depth**    | Industry-standard "drilling progress" chart — bit/hole depth & TVD over time                |
| **Wellbore schematic**| Casing string + live bit indicator + formation tops derived from real LWD gamma-ray         |
| **Active Alerts**     | Persistent, dedup'd, with Ack / Resolve buttons. Live updates via WebSocket                 |
| **Drilling params**   | Quad-chart: SPP, Hookload, Torque, Pit Volume                                               |
| **Anomaly timeline**  | Combined anomaly score + threshold line                                                     |
| **Events / Summary / Offset** | Classified events list · detection summary · cumulative production by well        |
| **AI Assistant (right pane)** | Multi-agent (Drilling / Safety / Historical / Reporting) with hand-off, or tool-loop mode |

### Replay speed

The slider in the top-right scrolls from **1× → 1000×** in real-time.
Changes are sent to the server over the same WebSocket and take effect
on the very next sample.

---

## 8. Operational endpoints

| Method | URL                                | Purpose                                             |
| ------ | ---------------------------------- | --------------------------------------------------- |
| GET    | `/health`                          | Liveness — always 200 once the process is up        |
| GET    | `/live`                            | Alias of `/health`                                  |
| GET    | `/ready`                           | 200 only when all subsystems are loaded             |
| GET    | `/metrics`                         | Prometheus exposition (text)                        |
| WS     | `/ws/stream`                       | Live telemetry. Control msgs: `set_speed`, `pause`, `resume`, `seek` |
| WS     | `/ws/alerts`                       | Snapshot + live alert events                        |
| GET    | `/api/alerts/active`               | List active alerts                                  |
| POST   | `/api/alerts/{id}/acknowledge`     | Mark as acknowledged                                |
| POST   | `/api/alerts/{id}/resolve`         | Mark as resolved                                    |
| POST   | `/api/alerts/{id}/suppress`        | Suppress (won't re-alert in 2 min window)           |
| POST   | `/api/copilot/query`               | Multi-agent (default) or tool-loop copilot          |
| POST   | `/api/rag/search`                  | Hybrid (BM25 + vector, RRF) DDR search              |
| GET    | `/api/data/timedepth`              | Time-vs-depth chart data                            |
| GET    | `/api/well/formations`             | Formation tops derived from real LWD gamma          |
| GET    | `/api/well/info`                   | Well metadata                                       |
| GET    | _all of the v0.3 endpoints_        | Unchanged — full backwards compatibility            |

---

## 9. Optional: enable a real LLM for the assistant

The assistant defaults to a deterministic rule-based fallback. To enable a real LLM:

### Ollama (local, free)

```bash
# install ollama from https://ollama.com
ollama pull mistral
export DRILLMIND_LLM_PROVIDER=ollama
export DRILLMIND_LLM_MODEL=mistral
python -m uvicorn drillmind.api.server:app
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
export DRILLMIND_LLM_PROVIDER=openai
export DRILLMIND_LLM_MODEL=gpt-4o-mini
python -m uvicorn drillmind.api.server:app
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export DRILLMIND_LLM_PROVIDER=anthropic
export DRILLMIND_LLM_MODEL=claude-sonnet-4-20250514
python -m uvicorn drillmind.api.server:app
```

---

## 10. Verifying everything is up

In a second terminal, while the API is running:

```bash
# 1. Liveness
curl http://localhost:8000/health

# 2. Readiness — all subsystems should report true
curl http://localhost:8000/ready | python -m json.tool

# 3. Prometheus metrics
curl http://localhost:8000/metrics | head -40

# 4. Active alerts (will be seeded from offline-detected events)
curl http://localhost:8000/api/alerts/active | python -m json.tool

# 5. Hybrid RAG search
curl -X POST http://localhost:8000/api/rag/search \
     -H "Content-Type: application/json" \
     -d '{"query":"lost circulation","top_k":3,"mode":"hybrid"}' | python -m json.tool

# 6. Multi-agent copilot
curl -X POST http://localhost:8000/api/copilot/query \
     -H "Content-Type: application/json" \
     -d '{"question":"What is the current rig state and are there any anomalies?","mode":"multi"}' | python -m json.tool
```

---

## 11. Troubleshooting

| Symptom                                              | Fix                                                                       |
| ---------------------------------------------------- | ------------------------------------------------------------------------- |
| `FileNotFoundError: Norway-NA-15_47_9-F-9 A time.csv`| Re-check filename in `data/raw/` — note the **double space** in `ROP data .csv` |
| `/ready` returns 503                                 | Wait — first run trains models. Check stdout for the current step.        |
| `ChromaDB query failed`                              | Delete `data/chromadb/` and restart — the index will be rebuilt           |
| Models keep retraining every restart                 | Make sure `data/models/` is writable and not gitignored as `**/models/**` |
| Dashboard shows "Disconnected"                       | Ensure no firewall is blocking the WebSocket upgrade on port 8000          |
| OOM on first run                                     | Set `DRILLMIND_MAX_ROWS=50000` to slice the dataset                       |
| Want JSON logs to ship to ELK / Loki                 | `export DRILLMIND_LOG_JSON=1`                                             |

---

## 12. What's running where

```
┌─────────────────────────────────────────────────────────────────────┐
│                   DrillMind v0.4 (single process)                   │
│                                                                     │
│  FastAPI (uvicorn)                                                  │
│  ├─ /health · /live · /ready · /metrics  (operations)               │
│  ├─ /api/* (REST surface, v0.3 + v0.4 additions)                    │
│  ├─ /ws/stream  (replay telemetry, 1×–1000×)                        │
│  ├─ /ws/alerts  (live alert broadcast)                              │
│  └─ /dashboard  (vanilla JS RTOC HMI)                               │
│                                                                     │
│  In-memory:                                                         │
│  ├─ time_df (419K rows)                                             │
│  ├─ features (261-d)                                                │
│  ├─ Autoencoder + Isolation Forest + LSTM ensemble                  │
│  ├─ Classified events + rig states + KPIs                           │
│  ├─ Hybrid retriever (ChromaDB vector + BM25)                       │
│  └─ MultiAgentOrchestrator (drilling / safety / historical / report)│
│                                                                     │
│  Persistent on disk:                                                │
│  ├─ data/models/         (ML checkpoints)                           │
│  ├─ data/chromadb/       (DDR vector store)                         │
│  └─ data/alerts.db       (SQLite alert log + resolutions audit)     │
└─────────────────────────────────────────────────────────────────────┘
```

That's everything. Welcome to the RTOC.
