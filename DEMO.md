# SentinelOps — Demo Runbook

A 6–8 minute walkthrough, with the exact commands and what to say at each step.

---

## Before you present

Run these once and confirm each one is green. Do it **the day before**, not five
minutes before.

```bash
# 1. The committed model is consistent and servable
python scripts/check_serving_contract.py          # expect 16/16

# 2. Every API route works against the real artifacts
python scripts/smoke_api.py                       # expect 34/34

# 3. Full test suite
pytest tests/ -q                                  # expect 170 passed
```

If you are demoing from a deployed URL on a free tier, **open the API's
`/health` five minutes beforehand**. Free instances suspend when idle and take
around 50 seconds to wake — that wake-up is the single most likely thing to go
wrong in front of an audience.

Have a second browser tab already open on the GitHub Actions page.

---

## Starting the stack

**Local, two terminals:**

```bash
uvicorn api.main:app --reload --port 8000    # terminal 1
cd frontend && npm run dev                   # terminal 2  → http://localhost:5173
```

**Or containers:**

```bash
docker compose up --build                    # → http://localhost:5173
```

The console opens on the **Live** tab.

---

## 1 · Live detection  (~2 min) — the centrepiece

Press **Start capture**.

> "This is network traffic arriving flow by flow. Every connection is being
> classified by the production model as it comes in — this isn't a recording of
> results, the model is scoring each flow at the moment you see it."

Point at, in order:

- **Flows inspected** climbing, **Threats detected** climbing more slowly.
- **Attack rate** settling near 8% — "a realistic enterprise mix, not a
  balanced test set."
- **Live accuracy** — "these flows carry their real CIC-IDS2017 labels, so
  that number is measured against ground truth, not a confidence score the
  model assigns itself."
- The **Truth** column: green ✓ where the verdict matched.
- The timeline: red band appearing whenever attacks cluster.

Change **Rate** to `50/s`. The feed accelerates; the model keeps up.

Click any red row to open the investigation drawer.

> "For every single detection we get a SHAP explanation — which features drove
> this verdict and in which direction. An analyst never has to take the model's
> word for it."

**Be upfront about what's simulated:** the footnote is on screen for a reason.

> "The flow features and labels are genuine CIC-IDS2017 captures. The IP
> addresses are reconstructed from the published testbed topology, because the
> cleaning stage drops identifier columns — they leak the label. None of that
> addressing reaches the model."

---

## 2 · Bring your own traffic  (~1.5 min)

**Detect** tab → *use the sample dataset* → **Run scan**.

> "Same model, different mode: bulk analysis of a capture file. 6,000 flows,
> threat summary, class breakdown, and global feature importance across the
> whole batch."

Point at the measured accuracy line — it reports the number of rows it was
computed over, so a partially-labelled file can't masquerade as a clean 99%.

**Then upload something deliberately awkward.** Generate one beforehand:

```bash
python - <<'PY'
import pandas as pd, re
df = pd.read_csv("samples/live_traffic_stream.csv", nrows=400)
df = df.drop(columns=["src_ip", "dst_ip", "dst_port", "protocol_name"])
df.columns = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in df.columns]
df.to_csv("my_export.csv", index=False)      # snake_case headers
PY
```

> "This is someone else's exporter — lowercase, underscores, nothing like the
> training schema. Watch the schema match."

It reports **47 / 47 features · 100%**.

> "Column names are matched regardless of case, spacing, or separators. And
> if a file *doesn't* match, the scan says so rather than quietly filling the
> gaps — anything missing more than a quarter of the schema is refused
> outright, because a confident 'no threats found' describing training
> defaults is worse than an error."

Mention the formats in one breath: CSV, TSV, JSON, JSONL, Parquet.

---

## 3 · The MLOps pipeline  (~2 min)

**MLOps** tab. Start at the **Lifecycle workflow** panel at the top.

> "This is the whole loop in one view — eight stages from raw capture to a
> monitored model, and each one is showing its actual current state, not a
> diagram."

Walk the numbers on the cards: runs logged, promoted vs rejected, registry
version, drift status. Then the line underneath:

> "Stage 08 closes back onto stage 03. When drift fires, retraining kicks off —
> but the challenger re-enters at the governance gate. Drift means the traffic
> changed, not that a new model is better."

Then scroll into the detail panels below:

- **Pipeline** — the four DVC stages. `dvc repro` reproduces the model from raw
  data; each stage declares the `params.yaml` keys it depends on, so changing a
  preprocessing threshold invalidates preprocessing and training but not the
  expensive ingestion.
- **Experiments / Registry** — every run logged to MLflow, models versioned.
- **Governance** — "a challenger is only promoted if it beats the incumbent on
  F1 *and* stays under the per-class false-negative ceiling. In intrusion
  detection a missed attack costs far more than a false alarm, so FNR is a hard
  gate, not a tiebreaker."
- **Drift** — the Evidently comparison against the training reference.

---

## 4 · Automation, live  (~2 min) — the closer

Switch to the GitHub Actions tab.

**Show CI:** every push runs backend tests, a serving-contract check, the
console build, and image builds.

> "The serving-contract check is the interesting one. It verifies the committed
> model, its feature list, and its preprocessing bundle actually agree with each
> other. A partial commit that leaves those out of sync passes every unit test
> and then fails at container start — this catches it before deploy."

**Then trigger drift detection live:**

Actions → **Drift Monitor** → *Run workflow* → set batch to
`samples/demo_traffic_mixed.csv` → Run.

> "That file is a sustained attack campaign — 50% hostile traffic against a
> reference that's about 15%. Let's see what the monitor does with it."

It reports **drift detected on ~66% of features**, publishes the Evidently
report as a run artifact, and opens a tracked issue. With retraining enabled it
goes on to retrain, put the challenger through the same governance gate, and
open a pull request **only if the gate approves**.

> "Drift means the world changed — it doesn't mean the new model is better. The
> gate still decides, and a model change goes through a pull request rather than
> straight to production."

Run it once more against `samples/live_traffic_stream.csv` to show the negative
case: normal traffic, no drift, no issue. **The clean result is what makes the
positive one meaningful.**

---

## Closing line

> "End to end: raw capture data through a versioned pipeline, a governed model,
> a live detection console with per-decision explanations, and monitoring that
> notices when the traffic stops looking like what the model was trained on."

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Console shows "API offline" | Backend not running, or CORS | Check `/health`; add the console's origin to `SENTINELOPS_CORS_ORIGINS` |
| Live feed starts then stalls | Free-tier instance suspended | Reload; hit `/health` first to wake it |
| First request takes ~50s | Cold start | Expected on free tiers — warm it beforehand |
| Feed won't start, no error | Scenario CSV missing | `python scripts/generate_demo_traffic.py` |
| Drift workflow fails | No `reports/drift_summary.json` | The check itself failed; read the run log |

**Fallback:** if the deployed URL misbehaves, run `docker compose up` locally
and present from `localhost`. Decide this *before* you start, not mid-demo.
