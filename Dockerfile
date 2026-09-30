# =============================================================================
# SentinelOps - Inference API
# =============================================================================
# Serves the promoted Champion: /predict, /analyze, /stream/live, /mlops/*.
#
# Build from the repo root so the model artifacts and the exported MLOps state
# are in context:
#     docker build -t sentinelops-api .
#     docker run -p 8000:8000 sentinelops-api
# =============================================================================

FROM python:3.12-slim AS base

# Fail fast and log straight through, so a crash-looping container shows its
# traceback in `docker logs` rather than dying with an empty buffer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp is XGBoost's OpenMP runtime. It is not in python:slim, and without it
# `import xgboost` fails at load with a bare shared-object error.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached across every source-only change,
# and the ML wheels are by far the slowest part of the build.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code and the artifacts it serves. models/ is committed to git
# (dvc.yaml declares it cache: false) precisely so this build needs no DVC
# remote; samples/ carries the traffic the live feed replays.
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY samples/ ./samples/
COPY params.yaml ./

# The MLOps panel reads the state the pipeline exported. Copied separately
# because reports/ also holds large PNGs that the API never serves.
COPY reports/mlops_state.json reports/drift_summary.json ./reports/

# Run unprivileged: nothing here needs root, and the model artifacts should not
# be writable by the process serving them.
RUN useradd --create-home --uid 10001 sentinel && chown -R sentinel:sentinel /app
USER sentinel

# The serving image has no processed dataset; ModelService falls back to the
# metadata baked into the champion artifacts for its class encoding.
ENV SENTINELOPS_MODELS_DIR=/app/models \
    SENTINELOPS_REPORTS_DIR=/app/reports \
    SENTINELOPS_SAMPLES_DIR=/app/samples \
    PORT=8000

EXPOSE 8000

# /health reports model_loaded, so this checks the model actually deserialized
# rather than only that the port is bound.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Shell form so ${PORT} is expanded: PaaS platforms assign the port at runtime.
# Single worker by design -- the model and its SHAP explainer are held in
# process memory, so each extra worker duplicates them.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75
