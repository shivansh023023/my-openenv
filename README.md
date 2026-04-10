# 🛡️ DataPipeline-Sentry

**Financial Data Lake Reliability Environment** — Built for the [Meta PyTorch OpenEnv Hackathon](https://pytorch.org/)

An AI agent acts as a **Data Reliability Engineer** managing a real-time Financial Data Lake. It must detect schema drift, handle late-arriving trade data, and quarantine poisoned data sources to prevent multi-million dollar trading errors.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-blue)](https://huggingface.co/spaces/Hello12334/openenv-data-cleaner)

---

## 🎯 Tasks

| Difficulty | Task | Description | Key Actions |
|:---:|---|---|---|
| 🟢 Easy | **Schema Drift** | Upstream API renamed fields (`trade_price`→`px`, `trade_volume`→`vol`). Fix the schema. | `inspect_schema` → `apply_mapping_patch` |
| 🟡 Medium | **Late-Arriving Data** | Trade records arriving out of order corrupt moving averages. Re-align the timeline. | `inspect_schema` → `watermark_reprocess` |
| 🔴 Hard | **Poisoned Records** | One of 5 data sources is injecting shifted data into a 10k-row baseline. Find and quarantine it. | `statistical_traceback` → `quarantine_source` |

Each task features **partial reward curves** (not binary pass/fail) and **deterministic grading** via `np.random.seed(42)`.

---

## 📁 Project Structure

```
├── app/
│   ├── __init__.py
│   ├── models.py      # Pydantic models (Observation, Actions, LatencyMetrics, SchemaHealth, BufferLog)
│   ├── env.py          # Core environment logic with 3 tasks & partial rewards
│   └── server.py       # FastAPI server (OpenEnv-compliant endpoints)
├── inference.py         # Baseline LLM agent with structured [START]/[STEP]/[END] logging
├── openenv.yaml         # Environment metadata
├── Dockerfile           # Runs as user:1000 on port 7860
├── pyproject.toml       # Package config
└── validate.py          # Pre-submission validation script
```

---

## 🚀 Quick Start

### Run Locally

```bash
pip install fastapi uvicorn pydantic pandas numpy scipy
uvicorn app.server:app --host 0.0.0.0 --port 7860
```

### Run with Docker

```bash
docker build -t datapipeline-sentry .
docker run -p 7860:7860 datapipeline-sentry
```

### Test Endpoints

```bash
# Health check
curl http://localhost:7860/

# List tasks
curl http://localhost:7860/tasks

# Reset to a task
curl -X POST http://localhost:7860/reset -H 'Content-Type: application/json' \
  -d '{"task": "schema_drift"}'

# Take an action
curl -X POST http://localhost:7860/step -H 'Content-Type: application/json' \
  -d '{"action": {"action": "inspect_schema"}}'

# Check score
curl http://localhost:7860/grader
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/reset` | Reset environment (`{"task": "schema_drift\|late_data\|poisoned_records"}`) |
| `POST` | `/step` | Execute an action |
| `GET` | `/state` | Current environment state |
| `GET` | `/grader` | Current grading score (0.0–1.0) |
| `GET` | `/baseline` | Baseline performance info |

---

## 📝 Structured Logging (Phase 2)

The `inference.py` script outputs evaluator-parsable logs:

```
[START] task=schema_drift
[STEP] step=1 reward=0.2
[STEP] step=2 reward=0.95
[END] task=schema_drift score=0.95 steps=2
```

All prints use `flush=True` for real-time evaluator compatibility.

---

## 🏗️ Observation Model

Each observation includes rich telemetry:

- **`latency_metrics`** — `ingestion_lag_ms`, `processing_time_ms`, `queue_depth`
- **`schema_health`** — `expected_fields`, `actual_fields`, `drift_detected`, `drift_details`
- **`buffer_logs`** — Per-source ingestion buffer with `pending`/`processed`/`quarantined` status
- **`data_snapshot`** — Human-readable pipeline state preview
- **`available_actions`** — Context-aware action list

---

## 🏆 Reward Curves

| Task | Action Flow | Reward Progression |
|------|------------|-------------------|
| Schema Drift | `inspect_schema` → `apply_mapping_patch` → `submit` | 0 → 0.2 → 0.95 |
| Late Data | `inspect_schema` → `watermark_reprocess` → `submit` | 0 → 0.1 → 0.95 |
| Poisoned Records | `statistical_traceback` → `quarantine_source` → `submit` | 0 → 0.2 → 0.95 |

---

## 📜 License

MIT
