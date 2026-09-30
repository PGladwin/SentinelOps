# 🛡️ SentinelOps - Network Intrusion Detection System

> **A production-grade ML pipeline for real-time detection and explanation of network intrusions using XGBoost with SHAP interpretability.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Threat Classification](#-threat-classification)
- [Quick Start](#-quick-start)
- [Live Traffic Monitor](#-live-traffic-monitor)
- [Phase 1: Data Pipeline](#-phase-1-data-pipeline)
- [Phase 2: Model Training](#-phase-2-model-training)
- [Phase 3: API & Deployment](#-phase-3-api--deployment)
- [Frontend Dashboard](#-frontend-dashboard)
- [Analyzing Your Own Traffic](#-analyzing-your-own-traffic)
- [Deployment](#-deployment)
- [CI/CD & Automation](#-cicd--automation)
- [API Endpoints](#-api-endpoints)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Model Performance](#-model-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**SentinelOps** is an end-to-end MLOps project that detects and classifies malicious network traffic with explainable AI (XAI). It processes network flow data from the **CIC-IDS2017 dataset**, trains ensemble and neural network models, selects a champion model, and deploys it as a production-ready REST API with a modern React frontend.

### Use Cases
- 🏢 **Enterprise Security** - Real-time threat detection for network traffic
- 🔍 **Security Analytics** - SHAP explanations for threat investigation
- 📊 **Model Explainability** - Understand why traffic is flagged as malicious
- 🚨 **Incident Response** - Batch analyze suspicious traffic flows

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔴 **Multiclass Classification** | 8-class threat taxonomy (BENIGN, DoS, DDoS, PortScan, BruteForce, Botnet, WebAttack, Infiltration) |
| 📡 **Live Traffic Feed** | Server-Sent Events stream classifying flows in real time, with running accuracy against ground truth |
| 🎯 **Real-time Inference** | Single flow prediction via REST API |
| 📤 **Batch Processing** | CSV upload for threat analysis at scale |
| 🔍 **SHAP Explainability** | Local & global feature importance on every decision, batch and live |
| 🏆 **Governed Promotion** | Champion/challenger gate on F1 *and* a per-class false-negative ceiling |
| 🔁 **Reproducible Pipeline** | DVC DAG — `dvc repro` rebuilds the model from raw data |
| 📊 **Experiment Tracking** | MLflow runs and model registry, exported for the dashboard |
| 📉 **Drift Monitoring** | Evidently comparison against the training reference, on a schedule |
| ⚙️ **CI/CD** | GitHub Actions: tests, serving-contract check, image builds, drift-triggered retraining |
| 🐳 **Containerized** | Dockerfiles + compose for API and console; Render/Vercel/Netlify blueprints |
| 📥 **Any Traffic Export** | CSV/TSV/JSON/JSONL/Parquet, with column names matched across case and separator variants |
| 🎨 **Interactive Dashboard** | React + Vite console: Live, Detect, MLOps |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SentinelOps Architecture                     │
└─────────────────────────────────────────────────────────────────┘

    CIC-IDS2017 Dataset (8 CSV files)
            ↓
    ┌───────────────────────────────────┐
    │  Phase 1: Data Pipeline           │
    │  • Ingest (Parquet conversion)    │
    │  • Clean (Label normalization)    │
    │  • Preprocess (Scale, split)      │
    └───────────────────────────────────┘
            ↓
    Train/Test Splits (70/30)
            ↓
    ┌───────────────────────────────────┐
    │  Phase 2: Model Training          │
    │  • Random Forest                  │
    │  • XGBoost                        │
    │  • MLP Neural Network             │
    │  • SHAP Analysis                  │
    │  • Feature Experiments            │
    └───────────────────────────────────┘
            ↓
    ┌───────────────────────────────────┐
    │  Governance Gate                  │
    │  • Macro F1 must improve          │
    │  • Per-class FNR ceiling enforced │
    │  → Champion promoted, or rejected │
    └───────────────────────────────────┘
            ↓
    ┌───────────────────────────────────┐
    │  Phase 3: Deployment              │
    │  • FastAPI REST API               │
    │  • SHAP Explainer Setup           │
    │  • SSE live traffic feed          │
    └───────────────────────────────────┘
            ↓
    ┌──────────────────────────────────────────────────┐
    │  Console (React + Vite)                          │
    │  Live feed | Upload & detect | MLOps lifecycle   │
    └──────────────────────────────────────────────────┘
            ↓
    ┌──────────────────────────────────────────────────┐
    │  Drift Monitoring (Evidently, scheduled)         │
    │  Breach → retrain → back to the governance gate  │
    └──────────────────────────────────────────────────┘
            └─────────────── ↺ ────────────────┘
```

---

## 📁 Project Structure

```
MLOPs-proj/
├── 📂 api/                          # FastAPI backend
│   ├── main.py                      # Application, routes, schema contract
│   ├── config.py                    # Env-driven runtime settings
│   ├── model_service.py             # Inference, SHAP, upload parsing
│   ├── traffic_stream.py            # Scenario replay & window scoring
│   ├── stream_routes.py             # SSE live feed (/stream/*)
│   ├── mlops_routes.py              # Lifecycle state (/mlops/*)
│   └── schemas.py                   # Pydantic request/response models
│
├── 📂 src/                          # Core ML pipeline
│   ├── data/
│   │   ├── ingest.py                # Load & convert Parquet files
│   │   ├── clean.py                 # Label normalization & validation
│   │   ├── preprocess.py            # Scaling, splitting, feature engineering
│   │   └── inference_prep.py        # Column resolution & raw→scaled bundle
│   ├── models/
│   │   ├── train.py                 # Model training orchestration
│   │   ├── evaluate.py              # Metrics computation & evaluation
│   │   └── explain.py               # SHAP analysis & explanations
│   └── mlops/
│       ├── tracking.py              # MLflow runs & registry
│       ├── governance.py            # Champion/challenger gate
│       ├── drift.py                 # Evidently drift detection
│       └── export_state.py          # Writes reports/mlops_state.json
│
├── 📂 frontend/                     # React + Vite console
│   ├── src/
│   │   ├── components/
│   │   │   ├── LiveFeed.jsx         # SSE live traffic monitor
│   │   │   ├── SocDashboard.jsx     # Upload & batch detection
│   │   │   ├── MlopsPanel.jsx       # Lifecycle console
│   │   │   ├── MlopsWorkflow.jsx    # The loop, with live per-stage state
│   │   │   ├── explanation.jsx      # Shared SHAP drawer & distribution
│   │   │   ├── ui.jsx               # Panels, metrics, badges
│   │   │   └── Navbar.jsx           # Navigation component
│   │   ├── api.js                   # API client
│   │   └── App.jsx                  # Main app component
│   ├── Dockerfile / nginx.conf      # Static bundle image
│   └── vite.config.js
│
├── 📂 scripts/                      # Pipeline stages & operational tooling
│   ├── stage_*.py                   # DVC stage entrypoints
│   ├── run_experiment_sweep.py      # Training sweep + governance gate
│   ├── generate_demo_traffic.py     # Builds the live-feed scenario
│   ├── check_drift.py               # Drift check (--fail-on-drift → exit 2)
│   ├── check_serving_contract.py    # Validates the committed Champion
│   └── smoke_api.py                 # Boots the API, exercises every route
│
├── 📂 .github/workflows/            # CI and drift-triggered retraining
│   ├── ci.yml
│   └── drift-retrain.yml
│
├── 📂 models/                       # Trained artifacts (committed, not DVC-cached)
│   ├── champion.pkl                 # Whatever governance last promoted
│   ├── champion_features.json       # Ordered feature contract
│   ├── champion_preprocessor.json   # Raw→scaled inference bundle
│   └── ...                          # Per-model artifacts and metadata
│
├── 📂 samples/                      # Replayable / uploadable traffic
│   ├── live_traffic_stream.csv      # 6,000 flows, ~8% hostile
│   └── demo_traffic_mixed.csv       # 50% hostile — trips the drift monitor
│
├── 📂 data/
│   ├── raw/                         # Original CIC-IDS2017 CSV files
│   ├── processed/                   # Cleaned & preprocessed data
│   │   ├── train.parquet
│   │   ├── test.parquet
│   │   ├── scaler.pkl
│   │   ├── feature_names.json
│   │   ├── class_encoding.json
│   │   └── preprocessing_metadata.json
│   └── reference/
│
├── 📂 metrics/                      # Model evaluation results
│   ├── xgboost_metrics.json
│   ├── xgboost_top40_metrics.json
│   ├── random_forest_metrics.json
│   └── mlp_metrics.json
│
├── 📂 reports/                      # Analysis & visualizations
│   ├── phase1_ingestion_report.json
│   ├── phase1_cleaning_report.json
│   ├── phase2_training_report.json
│   ├── confusion_matrix_*.png       # Model confusion matrices
│   ├── shap_summary_*.png           # SHAP explanations
│   └── feature_set_comparison.json
│
├── 📂 tests/                        # Unit & integration tests
│   ├── test_phase1.py
│   ├── test_phase2.py
│   └── test_api.py
│
├── run_phase1.py                    # Phase 1 pipeline orchestration
├── run_phase2.py                    # Phase 2 training orchestration
├── params.yaml                      # Configuration parameters
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🔴 Threat Classification

SentinelOps detects and classifies **7 types of network threats** from the CIC-IDS2017 dataset:

| Class | Type | Description | Example Attack |
|-------|------|-------------|-----------------|
| 🟢 **BENIGN** | Legitimate | Normal enterprise traffic | HTTPS browsing, DNS queries |
| 🔴 **DoS** | Denial of Service | Single-source attack | HTTP Flood (HULK) |
| 🔴 **DDoS** | Distributed DoS | Multi-source volumetric attack | Distributed packet flood (LOIC) |
| 🔴 **PortScan** | Reconnaissance | Network enumeration | SYN port scan (Nmap-style) |
| 🔴 **BruteForce** | Credential Attack | Authentication bypass | SSH brute-force dictionary attack |
| 🔴 **Botnet** | C2 Communication | Command & control beaconing | Botnet C2 heartbeat (Ares) |
| 🔴 **WebAttack** | Application Layer | Web exploit attack | XSS injection (Cross-site scripting) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+ (for frontend)
- Git
- 8GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/prithish47/Mlops-proj.git
cd Mlops-proj
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Install frontend dependencies**
```bash
cd frontend
npm install
cd ..
```

### Run Complete Pipeline (Recommended)

```bash
# Phase 1: Data preprocessing (5-10 minutes)
python run_phase1.py

# Phase 2: Model training & evaluation (15-20 minutes)
python run_phase2.py
```

### Start API Server
```bash
# From project root
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend
```bash
cd frontend
npm run dev
# Navigate to http://localhost:5173
```

### Or run the whole stack in containers
```bash
docker compose up --build
# Console: http://localhost:5173   API: http://localhost:8000
```

> **Presenting this?** [`DEMO.md`](DEMO.md) is a timed walkthrough with the
> exact commands, what to say at each step, and a troubleshooting table.

### Verify the install

```bash
python scripts/check_serving_contract.py   # 16 checks: artifacts agree and score
python scripts/smoke_api.py                # 34 checks: every route, incl. the live feed
pytest tests/ -q                           # 170 tests
```

---

## 📡 Live Traffic Monitor

The **Live** tab replays a traffic capture as a paced, classified feed — the
demo centrepiece, and the view that shows the system doing its job rather than
reporting on work already done.

Each flow is scored by the promoted Champion **at the moment it is emitted**,
through the same model, preprocessing, and SHAP path that serves `/predict`.
Nothing is precomputed.

```bash
# Regenerate the stream (6,000 flows, ~8% hostile — a realistic enterprise mix)
python scripts/generate_demo_traffic.py

# Custom composition
python scripts/generate_demo_traffic.py --rows 12000 --attack-rate 0.12
```

**What is genuine and what is not.** Every feature value and every label is
sampled verbatim from the cleaned CIC-IDS2017 corpus — the detections are the
model's real behaviour on real attack traffic, and because ground truth travels
with each flow, the console reports **measured accuracy**, not confidence.

The addressing — `src_ip`, `dst_ip`, `dst_port`, `protocol_name` — is
reconstructed from the published CIC-IDS2017 testbed topology. The cleaning
stage drops identifier columns because they leak the label, so the flows carry
no addressing of their own and a packet-log view would have nothing to render.
That envelope is presentation only, it never reaches the model, and the UI says
so on screen.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /stream/scenarios` | Replayable captures, with their true class mix |
| `GET /stream/live` | SSE feed: `ready` → `batch`* → `complete` |

```bash
# Watch 40 flows go by
curl -N "http://localhost:8000/stream/live?rate=20&limit=40&loop=false"
```

Query parameters: `scenario`, `rate` (flows/sec), `loop`, `limit`, `start`.

SSE rather than WebSockets: the feed is one-directional, and plain HTTP
survives the proxies and free-tier routing that often break WebSocket upgrades.

---

## 📊 Phase 1: Data Pipeline

**Purpose**: Ingest raw network traffic data, clean labels, preprocess features

### Run Phase 1
```bash
# Development mode (sample 150k rows)
python run_phase1.py

# Full dataset mode
python run_phase1.py --full

# Custom configuration
python run_phase1.py --params custom_params.yaml
```

### Outputs
- `data/processed/train.parquet` - Scaled training data
- `data/processed/test.parquet` - Scaled test data
- `data/processed/feature_names.json` - Ordered feature list
- `data/processed/class_encoding.json` - Label encoding mapping
- `data/processed/preprocessing_metadata.json` - Pipeline metadata
- `reports/phase1_ingestion_report.json` - Data statistics
- `reports/phase1_cleaning_report.json` - Cleaning summary

### Process
```
Raw CSV Files (8 CSVs)
    ↓ [ingest.py]
Convert to Parquet + Aggregate
    ↓ [clean.py]
Normalize Labels: "ddos" → "DDoS", "bruteforce" → "BruteForce"
Validate Data Quality
    ↓ [preprocess.py]
Handle Missing Values (mean imputation)
Remove Correlated Features
Train/Test Split (70/30)
StandardScaler Normalization
    ↓
Processed Data (70 features × 169k samples)
```

---

## 🧠 Phase 2: Model Training

**Purpose**: Train multiple models, evaluate performance, select champion, generate SHAP explanations

### Run Phase 2
```bash
# Train all models
python run_phase2.py

# Train specific model
python run_phase2.py --model xgb          # XGBoost only
python run_phase2.py --model rf           # Random Forest only
python run_phase2.py --model mlp          # MLP only

# Skip SHAP analysis (faster)
python run_phase2.py --no-shap

# Skip feature experiments
python run_phase2.py --no-feature-exp
```

### Trained Models

The last sweep, as logged to MLflow and exported to `reports/mlops_state.json`:

| Run | Accuracy | Macro F1 | Attack FNR | Outcome |
|---|---:|---:|---:|---|
| `rf_baseline` | 0.9952 | 0.8403 | 0.0511 | Rejected |
| **`xgb_baseline`** | 0.9989 | 0.9308 | **0.0295** | 🏆 **Champion** |
| `xgb_deep` | 0.9990 | 0.9414 | 0.0304 | Rejected |
| `mlp_baseline` | 0.9760 | 0.7701 | 0.3295 | Rejected |
| `xgb_top40` | 0.9988 | 0.9127 | 0.0298 | Rejected |
| `xgb_top40_tuned` | 0.9989 | 0.9334 | 0.0301 | Rejected |

**Note `xgb_deep`.** It has the best macro F1 in the sweep (0.9414 against the
champion's 0.9308) and was still rejected — its attack FNR is worse (0.0304 vs
0.0295). The gate refuses a challenger that regresses attack FNR no matter what
it gains elsewhere, because in intrusion detection a missed attack costs far
more than a false alarm. This is the gate doing its job, not a bug.

Which model holds the title is not fixed: whatever the gate last promoted is
written to `models/champion.pkl`, and the API serves that.

### Outputs
- `models/champion.pkl` — the promoted model, whichever it is
- `models/champion_features.json` — its ordered feature contract
- `models/champion_preprocessor.json` — the raw→scaled inference bundle
- `models/champion_metadata.json` — hyperparameters, version, class mapping
- `reports/mlops_state.json` — runs, registry, gate decisions, metrics
- `reports/confusion_matrix_*.png`, `reports/shap_summary_*.png`

### Feature Engineering

Features are reduced from 69 by a correlation filter during preprocessing;
`params.yaml:preprocessing.correlation_threshold` controls the cut. The current
champion consumes **47**:

- Packet counts & sizes (forward/backward)
- Flow duration, inter-arrival time, and rate statistics
- TCP flag counts (FIN, RST, PSH, ACK, URG)
- Initial window sizes and segment-size minimums
- Active/idle period statistics

The sweep also trains explicit SHAP-selected variants (`xgb_top40*`) so feature
selection is a measured comparison rather than an assumption — and on this
corpus the correlation-filtered 47 beat the SHAP-selected 40.

---

## 🔌 Phase 3: API & Deployment

### API Overview

**Framework**: FastAPI + Uvicorn  
**Base URL**: `http://localhost:8000`  
**Docs**: `http://localhost:8000/docs` (Swagger UI)

### Start API

```bash
# Development mode (with reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🔌 API Endpoints

### 1. Health Check
```http
GET /health
```
**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model": "xgboost_top40",
  "feature_count": 40
}
```

### 2. Get Model Info
```http
GET /model-info
```
**Response**:
```json
{
  "model_name": "xgboost_top40",
  "model_type": "XGBoost",
  "feature_count": 40,
  "features": ["Fwd Packet Length Mean", "Bwd Packet Length Mean", ...],
  "class_mapping": {
    "BENIGN": 0,
    "DoS": 1,
    "DDoS": 2,
    ...
  },
  "metadata": {
    "training_accuracy": 0.987,
    "training_f1": 0.985,
    ...
  }
}
```

### 3. Get Demo Samples
```http
GET /demo-samples
```
**Response**:
```json
[
  {
    "id": "benign_flow_01",
    "label": "BENIGN",
    "description": "Legitimate HTTPS browsing & DNS query traffic",
    "features": {
      "Fwd Packet Length Mean": 45.23,
      "Bwd Packet Length Mean": 128.45,
      ...
    }
  },
  ...
]
```

### 4. Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "features": [45.23, 128.45, 0.5, 1024, ...]  # 40 values in order
}
```

**Response**:
```json
{
  "prediction": "BENIGN",
  "prediction_proba": {
    "BENIGN": 0.98,
    "DoS": 0.01,
    "DDoS": 0.005,
    ...
  },
  "shap_explanation": {
    "base_value": 0.45,
    "shap_values": [-0.32, 0.15, -0.08, ...],
    "feature_names": ["Fwd Packet Length Mean", ...],
    "feature_values": [45.23, ...]
  },
  "top_contributing_features": [
    {"name": "Fwd Packet Length Mean", "impact": 0.32},
    ...
  ]
}
```

### 5. Batch Analysis
```http
POST /analyze
Content-Type: multipart/form-data

file=flows.csv    # or .tsv .txt .json .jsonl .parquet
```

Classifies every flow, computes batched SHAP, and returns dashboard aggregates.
Column names are matched across case and separator variants — see
[Analyzing Your Own Traffic](#-analyzing-your-own-traffic).

**Response** (abridged):
```json
{
  "summary": {
    "threat_level": "MEDIUM",
    "total_connections": 6000,
    "total_attacks": 487,
    "attack_rate": 0.081167,
    "features_expected": 47,
    "features_matched": 47,
    "schema_coverage": 1.0,
    "columns_supplied": 71,
    "n_unused_columns": 22,
    "ground_truth_available": true,
    "ground_truth_accuracy": 0.9985,
    "ground_truth_scored": 6000
  },
  "class_breakdown":       { "BENIGN": 5513, "DoS": 149, "DDoS": 125, "...": 0 },
  "confidence_histogram":  [ { "bucket": "90%-100%", "count": 5904 } ],
  "global_shap":           [ { "rank": 1, "feature": "Init Bwd Win Bytes", "mean_abs_shap": 1.83 } ],
  "top_suspicious":        [ "…ten highest-confidence attacks…" ],
  "rows":                  [ { "flow_id": 1, "prediction": "BENIGN", "confidence": 0.9999,
                               "is_attack": false, "top_features": [] } ]
}
```

`ground_truth_accuracy` is always paired with `ground_truth_scored`: labels that
do not resolve to the taxonomy are excluded, so compare the two before reading
the figure as whole-file.

### 6. Upload Contract
```http
GET /schema                 # features, aliases, formats, limits
GET /schema/template.csv    # header-only CSV in the exact schema
```

### 7. Live Traffic Feed
```http
GET /stream/scenarios       # replayable captures and their class mix
GET /stream/live?scenario=live_traffic_stream&rate=8&loop=true
```

Server-Sent Events: `ready` → `batch`\* → `complete`. Each flow carries its
prediction, confidence, full class distribution, top-3 SHAP contributions, and
its ground-truth label.

### 8. MLOps Lifecycle
```http
GET /mlops/state            # everything the control panel renders, in one call
GET /mlops/runs             # MLflow experiment runs
GET /mlops/registry         # model versions and the champion pointer
GET /mlops/governance       # champion/challenger decision history
GET /mlops/metrics          # per-class metrics for the production model
GET /mlops/pipeline         # DVC DAG and stage freshness
GET /mlops/drift            # drift status
GET /mlops/drift/report     # the Evidently HTML report
```

Interactive docs for every route: **http://localhost:8000/docs**

---

## 🎨 Frontend Dashboard

Three tabs, each answering a different question: *what is happening now*, *what
is in this file*, and *how did this model get here*.

**Live** (`LiveFeed.jsx`) — the SOC console
- Streams classified flows over SSE, at 4–50 flows/sec
- Running counters: flows, threats, attack rate, threat level
- **Live accuracy** measured against ground truth, not confidence
- Per-second timeline, benign stacked against detected threats
- Click any flow for its SHAP drawer, with the full class distribution

**Detect** (`SocDashboard.jsx`) — bring your own traffic
- Drop any flow export: CSV, TSV, JSON, JSONL, or Parquet
- Column names matched regardless of case, spacing, or separators
- A **schema match** report showing exactly what was recognised
- Threat summary, class breakdown, confidence histogram
- Global feature importance, paginated log, export flagged flows

**MLOps** (`MlopsPanel.jsx`) — lifecycle control panel
- **Lifecycle workflow**: the whole loop as eight stages, each carrying its
  own live state, closing from drift monitoring back to the governance gate
- DVC stage DAG, MLflow runs, and registry versions
- Champion/challenger decision history
- Drift status and the Evidently report
- Live GitHub Actions history

Shared pieces live in `ui.jsx` (panels, metrics, badges), `explanation.jsx`
(SHAP drawer, contribution bars, class distribution) and `MlopsWorkflow.jsx`,
so the batch and live views present a verdict identically.

> A fourth **Explain** tab was removed: it scored one pre-baked sample at a
> time, and both Live and Detect now open a full SHAP investigation — class
> probabilities included — on any flow. Its one unique element moved into the
> shared drawer rather than being dropped.

### Access Frontend
```bash
cd frontend
npm run dev
# Open http://localhost:5173
```

---

## 📥 Analyzing Your Own Traffic

The **Detect** tab takes whatever your exporter produced. You should not have
to reshape a capture before you can scan it.

### Formats

`.csv` · `.tsv` · `.txt` · `.json` · `.jsonl` / `.ndjson` · `.parquet`

Delimiters are sniffed from the header, so comma, tab, semicolon and pipe files
all work without being declared. JSON may be a bare array of records, or nested
under a `flows` / `data` / `records` / `rows` key.

### Column names

Headers are matched **case- and separator-insensitively**, so every one of
these resolves to the same feature:

```
Flow Duration    flow_duration    FLOW-DURATION    Flow.Duration    flowduration
```

The CIC-IDS2017 CSV-release aliases are resolved on top of that
(`Total Length of Fwd Packets` → `Fwd Packets Length Total`, `Init_Win_bytes_forward`
→ `Init Fwd Win Bytes`, and so on).

Word order is deliberately *not* normalised. `Packet Length Min` and
`Min Packet Length` are related through the explicit alias table rather than
guessed at, because a wrong mapping feeds one feature's values into another's
slot — worse than a missing column, since it is silently plausible instead of
visibly absent.

Identifier columns (`Source IP`, `Destination Port`, `Timestamp`, `Flow ID`, …)
are ignored if present. A `Label` column, if you have one, is used only to
report measured accuracy — never for inference.

### What you get back

Every scan reports a **schema match**: how many of the model's features were
found, which were missing, and how many extra columns were carried but unused.
This matters because absent features are filled from training medians — without
the report, a scan driven largely by defaults would look identical to a clean
one.

An upload missing more than 25% of the schema is **refused** rather than
scored. A confident "no threats detected" describing training medians is worse
than an error, because someone would act on it.

### The contract

```bash
curl http://localhost:8000/schema                  # features, aliases, formats, limits
curl -O http://localhost:8000/schema/template.csv  # header-only CSV in the exact schema
```

Values must be **raw**, in the units of the source dataset (packet counts, byte
rates, microsecond durations). Scaling is applied server-side.

Need test data? `python scripts/generate_demo_traffic.py --rows 5000 --attack-rate 0.15`.

---

## 🚢 Deployment

The API and the console deploy as two independent services. The model is only
3.6 MB and the serving image installs `requirements-api.txt` — no MLflow, DVC,
or Evidently — so it fits comfortably on a free tier.

### Docker

```bash
docker compose up --build        # both services
docker build -t sentinelops-api .
docker run -p 8000:8000 sentinelops-api
```

### Render (both services, one blueprint)

[`render.yaml`](render.yaml) defines the API (Docker) and the console (static).

1. Push to GitHub → Render dashboard → **New → Blueprint** → pick the repo.
2. After the first deploy, set these and redeploy:
   - API: `SENTINELOPS_CORS_ORIGINS` = the console's URL
   - Console: `VITE_API_BASE_URL` = the API's full origin, **scheme included**

### Vercel / Netlify (console only)

Config is committed at [`frontend/vercel.json`](frontend/vercel.json) and
[`frontend/netlify.toml`](frontend/netlify.toml). Point the project at the
`frontend/` directory and set `VITE_API_BASE_URL` **before the first build** —
Vite inlines it into the bundle, so setting it afterwards has no effect until
the site is rebuilt.

### Two things that bite

**CORS.** The API allows only the origins in `SENTINELOPS_CORS_ORIGINS`. A
console deployed to a URL that isn't listed fails every request, in the browser
only — the API's own health check stays green.

**Cold starts.** Free instances suspend when idle and take ~50 seconds to wake,
during which the live feed cannot connect. Hit `/health` before a demo.

---

## ⚙️ CI/CD & Automation

### `CI` — every push and pull request

| Job | What it does |
|---|---|
| **Backend tests** | 170 pytest tests |
| **Serving contract** | Validates the committed Champion, then boots the API and exercises every route |
| **Console build** | oxlint + production Vite build |
| **Image builds** | Builds both images and asserts the API container reports `model_loaded: true` (PRs only) |

The serving-contract check earns its place: it verifies `champion.pkl`, its
feature list, and its preprocessing bundle actually agree. A partial commit
that leaves them out of sync passes every unit test — they exercise code, not
artifacts — and then fails at container startup. Run it yourself with
`python scripts/check_serving_contract.py`.

### `Drift Monitor` — scheduled, and on demand

Daily at 03:17 UTC, or **Actions → Drift Monitor → Run workflow**:

1. Score a traffic batch against the committed training reference.
2. **No drift** → record and stop.
3. **Drift** → retrain, put the challenger through the same governance gate,
   and open a PR *only if the gate approves*. Drift means the world moved, not
   that the new model is better — the gate still decides, and a model change is
   reviewed rather than pushed to main.
4. Either way, a drift finding is filed as a tracked issue with the Evidently
   report attached to the run.

Try both outcomes:

```bash
# Normal traffic → no drift (8 of 47 features, under the 20% threshold)
python scripts/check_drift.py --batch samples/live_traffic_stream.csv

# Sustained attack campaign → DRIFT DETECTED (31 of 47 features)
python scripts/check_drift.py --batch samples/demo_traffic_mixed.csv
```

`--fail-on-drift` exits **2** on a drift verdict, leaving exit 1 to mean the
check itself failed — so CI can tell "the model needs attention" apart from
"the monitoring is broken".

**Enabling automatic retraining.** Retraining needs the dataset, which is
DVC-tracked and not in the repository. Set the repository variable
`ENABLE_AUTO_RETRAIN=true` and the secret `DVC_REMOTE_URL` (plus credentials for
your remote). Without them the retrain job is skipped with an explanation rather
than failing, and the drift finding is still reported.

---

## ⚙️ Configuration

### `params.yaml`

```yaml
# Phase 1 Configuration
phase1:
  sample_size: 150000        # Use 0 for full dataset
  test_size: 0.3
  random_state: 42
  scale_method: "standard"   # StandardScaler

# Phase 2 Configuration  
phase2:
  models:
    - random_forest
    - xgboost
    - mlp
  random_forest:
    n_estimators: 100
    max_depth: 15
  xgboost:
    n_estimators: 100
    max_depth: 7
    learning_rate: 0.1
  mlp:
    hidden_layer_sizes: [128, 64]
    learning_rate_init: 0.001
```

### Environment Variables

Every setting has a working default, so `uvicorn api.main:app` from the project
root needs none of these. They exist so the same image runs anywhere. Defined
in [`api/config.py`](api/config.py).

**API**

| Variable | Default | Purpose |
|---|---|---|
| `SENTINELOPS_CORS_ORIGINS` | localhost 5173/3000 | Comma-separated allowed origins. **Must list the deployed console's URL**, or every browser request fails. |
| `SENTINELOPS_MODELS_DIR` | `models` | Champion artifacts |
| `SENTINELOPS_REPORTS_DIR` | `reports` | Where `/mlops/*` reads exported state |
| `SENTINELOPS_SAMPLES_DIR` | `samples` | Traffic scenarios the live feed replays |
| `SENTINELOPS_PROCESSED_DIR` | `data/processed` | Optional; serving falls back to the champion metadata |
| `SENTINELOPS_DEFAULT_SCENARIO` | `live_traffic_stream` | Scenario the Live tab opens on |
| `SENTINELOPS_STREAM_RATE` | `8` | Default flows/sec |
| `SENTINELOPS_STREAM_MAX_RATE` | `200` | Ceiling — each flow costs a SHAP pass |
| `SENTINELOPS_STREAM_TICK` | `0.4` | Seconds between emitted batches |
| `SENTINELOPS_MAX_UPLOAD_MB` | `50` | `/analyze` upload cap |
| `SENTINELOPS_MAX_ANALYSIS_ROWS` | `100000` | Rows scored per upload |
| `SENTINELOPS_MAX_SHAP_ROWS` | `20000` | Rows explained per upload |
| `SENTINELOPS_MAX_RETURNED_ROWS` | `20000` | Rows echoed in the response |
| `PORT` | `8000` | Bind port (set by most PaaS platforms) |

**Console** — read by Vite at **build** time and inlined into the bundle.
Changing either one requires a rebuild, not a restart.

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | API origin, **scheme included** |
| `VITE_GITHUB_REPO` | `prithish47/Mlops-proj` | Repo behind the CI/CD panel |

**CI** — repository settings, for the drift workflow

| Name | Kind | Purpose |
|---|---|---|
| `ENABLE_AUTO_RETRAIN` | variable | `true` switches on the retrain job |
| `DVC_REMOTE_URL` | secret | Remote holding the tracked dataset |

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_phase1.py
pytest tests/test_api.py

# With coverage
pytest tests/ --cov=src --cov=api

# Verbose output
pytest tests/ -v
```

### Test Files

- `tests/test_phase1.py` - Data pipeline validation
- `tests/test_phase2.py` - Model training verification
- `tests/test_api.py` - API endpoint testing

---

## 📈 Model Performance

Current champion: **`xgb_baseline`** (XGBoost, 47 features), evaluated on
446,362 held-out flows. These figures are read from
`reports/mlops_state.json` — the same document the MLOps tab serves, exported
by the pipeline rather than transcribed by hand. Regenerate them with
`dvc repro train`.

### Per-class results

| Class | Support | Recall | F1 | FNR |
|---|---:|---:|---:|---:|
| BENIGN | 379,064 | 0.9988 | 0.9994 | 0.0012 |
| DoS | 38,751 | 0.9997 | 0.9971 | 0.0003 |
| DDoS | 25,603 | 0.9999 | 0.9996 | 0.0001 |
| BruteForce | 1,830 | 0.9984 | 0.9970 | 0.0016 |
| WebAttack | 429 | 0.9883 | 0.9792 | 0.0117 |
| PortScan | 391 | 0.9744 | 0.9466 | 0.0256 |
| Botnet | 287 | 0.9756 | 0.7273 | 0.0244 |
| Infiltration | 7 | 0.8571 | 0.8000 | 0.1429 |

### Headline metrics

| Metric | Value |
|---|---|
| Accuracy | 0.9989 |
| Weighted F1 | 0.9989 |
| **Macro F1** | **0.9308** |
| Macro recall | 0.9740 |
| Mean attack FNR | 0.0295 |
| Max per-class FNR | 0.1429 (Infiltration) |

**Read the macro F1, not the accuracy.** 85% of the corpus is benign, so
accuracy is dominated by the easy majority class; the 0.999 → 0.931 gap between
weighted and macro F1 is where the real difficulty lives. Botnet is the honest
weak spot — recall is fine at 0.976, but precision drags F1 to 0.73, meaning
benign flows are being pulled into that class. Infiltration has 7 test flows
in the entire corpus, so its numbers are indicative at best.

Governance gates on **per-class FNR** for exactly this reason: a model that
improves overall accuracy while quietly missing more Botnet traffic is not an
improvement for a SOC.

### Feature importance

Global mean |SHAP| is computed live over whatever batch you analyse — see the
**Detect** tab, or `reports/shap_summary_*.png` from the last training run. It
is not hardcoded here, because it changes with the promoted champion.

---

## 🔧 Troubleshooting

### Issue: Model fails to load on API startup

**Solution**:
```bash
# Check the artifacts agree with each other -- this reports exactly which
# one is missing or out of sync, rather than leaving you to guess
python scripts/check_serving_contract.py

# Rebuild the champion if the artifacts are stale or absent
dvc repro train
```

### Issue: API connection refused

**Solution**:
```bash
# Check if port is in use
netstat -tuln | grep 8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Use different port
uvicorn api.main:app --port 8001
```

### Issue: Frontend cannot reach API

**Solution** (frontend/.env):
```bash
VITE_API_URL=http://localhost:8000
```

### Issue: Out of memory during Phase 2

**Solution**:
```bash
# Use smaller sample
python run_phase1.py --sample-size 50000
python run_phase2.py
```

### Issue: CSV batch upload fails

**Ensure CSV**:
- Has exactly 40 columns
- Columns match feature names in order
- No missing values
- Numeric format

---

## 📚 Dependencies

### Python Packages
- **ML & Data**: numpy, pandas, scikit-learn, xgboost
- **Explainability**: shap
- **API**: fastapi, uvicorn, pydantic
- **Visualization**: matplotlib
- **Config**: pyyaml

See `requirements.txt` for full list with versions.

### Frontend Dependencies
- **React** 18+
- **Vite** (build tool)
- **Tailwind CSS** (optional styling)

See `frontend/package.json` for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Style
- Follow PEP 8 for Python
- Use type hints
- Add docstrings to functions
- Write tests for new features

---

## 📞 Support

For issues, questions, or suggestions:
1. Check [Troubleshooting](#-troubleshooting)
2. Search existing [GitHub Issues](https://github.com/prithish47/Mlops-proj/issues)
3. Create a new issue with detailed description

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) - Canadian Institute for Cybersecurity
- **Libraries**: FastAPI, XGBoost, SHAP, scikit-learn, React
- **Inspiration**: MLOps best practices & network security research

---

## 📊 Project Stats

- **Total Lines of Code**: 3,000+
- **Python Modules**: 8
- **Frontend Components**: 5
- **API Endpoints**: 5
- **Test Coverage**: 70%+
- **Supported Threat Classes**: 7
- **Dataset Samples**: 169,000+

---

**Last Updated**: August 2026  
**Status**: ✅ Production Ready  
**Champion Model Accuracy**: 98.7%

---

<div align="center">

### 🚀 Ready to detect threats?

[Start API Server](#start-api-server) | [Run Full Pipeline](#run-complete-pipeline-recommended) | [View Frontend](#start-frontend)

⭐ If this project helped you, please give it a star!

</div>
