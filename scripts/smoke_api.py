"""
SentinelOps - API Smoke Test
============================
Boots the real FastAPI application in-process and exercises every route a
deployed instance must serve, including the SSE live feed.

This complements the unit tests rather than repeating them: it runs the app
through its actual lifespan (model load, SHAP explainer init, scenario
discovery) against the committed artifacts, which is the part that breaks when
an artifact, a path, or an environment default is wrong -- and the part a test
suite using fixtures and mocks cannot observe.

Exit code 0 if every route behaves, 1 otherwise.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

failures: list[str] = []
checks = 0


def check(condition: bool, description: str, detail: str = "") -> bool:
    global checks
    checks += 1
    if condition:
        print(f"  PASS  {description}")
        return True
    print(f"  FAIL  {description}")
    if detail:
        print(f"        {detail}")
    failures.append(description)
    return False


def main() -> int:
    from fastapi.testclient import TestClient

    from api.main import app

    print("Booting SentinelOps API in-process...\n")

    # The context manager runs the lifespan handler, so the model is loaded
    # exactly as it would be in a container rather than lazily on first call.
    with TestClient(app) as client:
        # -- Health ------------------------------------------------------
        print("Health & introspection")
        r = client.get("/health")
        check(r.status_code == 200, "GET /health returns 200", f"got {r.status_code}")
        health = r.json() if r.status_code == 200 else {}
        check(health.get("model_loaded") is True, "health reports model_loaded=true", str(health))

        r = client.get("/model-info")
        check(r.status_code == 200, "GET /model-info returns 200")
        info = r.json() if r.status_code == 200 else {}
        feature_count = info.get("feature_count", 0)
        check(feature_count > 0, "model-info reports a feature count", str(feature_count))

        r = client.get("/demo-samples")
        samples = r.json() if r.status_code == 200 else []
        check(r.status_code == 200 and len(samples) > 0, "GET /demo-samples returns samples")

        # -- Prediction --------------------------------------------------
        print("\nInference")
        if samples:
            r = client.post("/predict", json={"features": samples[0]["features"]})
            check(r.status_code == 200, "POST /predict returns 200", r.text[:200])
            if r.status_code == 200:
                body = r.json()
                check(bool(body.get("prediction")), "prediction carries a class label")
                check(len(body.get("explanation", [])) > 0, "prediction carries a SHAP explanation")

        # A request missing required features must be rejected as a client
        # error, not surface as a 500 from deep inside the model.
        r = client.post("/predict", json={"features": {"Protocol": 6}})
        check(r.status_code == 422, "POST /predict rejects an incomplete flow with 422", f"got {r.status_code}")

        # -- Batch analysis ----------------------------------------------
        print("\nSOC analysis")
        scenario_csv = PROJECT_ROOT / "samples" / "live_traffic_stream.csv"
        if scenario_csv.exists():
            # Only the header and a small slice: this asserts the route works,
            # and batch SHAP over the whole file would dominate CI runtime.
            lines = scenario_csv.read_text(encoding="utf-8").splitlines()
            excerpt = "\n".join(lines[:201]).encode("utf-8")
            r = client.post(
                "/analyze",
                files={"file": ("excerpt.csv", io.BytesIO(excerpt), "text/csv")},
            )
            check(r.status_code == 200, "POST /analyze returns 200", r.text[:200])
            if r.status_code == 200:
                summary = r.json()["summary"]
                check(summary["total_connections"] == 200, "analysis covers every submitted row")
                check(
                    summary["n_imputed_columns"] == 0,
                    "no Champion feature had to be imputed",
                    f"imputed: {summary.get('imputed_columns')}",
                )
                # Guards the label-resolution path: a regression there silently
                # scores a fraction of the file and still reports high accuracy.
                check(
                    summary.get("ground_truth_scored") == 200,
                    "ground truth resolved for every labelled row",
                    f"scored {summary.get('ground_truth_scored')} of 200",
                )

        # .txt is a supported container (delimited text); a PDF is not.
        r = client.post("/analyze", files={"file": ("report.pdf", io.BytesIO(b"%PDF"), "application/pdf")})
        check(r.status_code == 400, "POST /analyze rejects an unsupported file type", f"got {r.status_code}")

        r = client.post("/analyze", files={"file": ("junk.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
        check(
            r.status_code == 422,
            "POST /analyze refuses a file that is not network data",
            f"got {r.status_code} -- an unrelated file must not be scored against training medians",
        )

        # -- Upload contract ---------------------------------------------
        print("\nSchema contract")
        r = client.get("/schema")
        check(r.status_code == 200, "GET /schema returns 200")
        if r.status_code == 200:
            schema = r.json()
            check(schema["feature_count"] == len(schema["features"]), "schema feature count is consistent")
            check(".parquet" in schema["supported_formats"], "schema advertises the accepted formats")

        r = client.get("/schema/template.csv")
        check(r.status_code == 200, "GET /schema/template.csv returns 200")
        check(
            r.headers["content-type"].startswith("text/csv"),
            "template is served as CSV",
            r.headers.get("content-type", ""),
        )

        # -- Live feed ---------------------------------------------------
        print("\nLive feed")
        r = client.get("/stream/scenarios")
        check(r.status_code == 200, "GET /stream/scenarios returns 200")
        scenarios = r.json().get("scenarios", []) if r.status_code == 200 else []
        check(len(scenarios) > 0, "at least one replay scenario is discoverable")

        with client.stream("GET", "/stream/live?rate=20&limit=20&loop=false") as stream:
            check(stream.status_code == 200, "GET /stream/live opens")
            check(
                stream.headers["content-type"].startswith("text/event-stream"),
                "live feed advertises text/event-stream",
                stream.headers.get("content-type", ""),
            )
            body = "".join(stream.iter_text())

        check("event: ready" in body, "stream emits a ready event")
        check("event: batch" in body, "stream emits classified batches")
        check('"prediction"' in body, "streamed flows carry predictions")
        check('"top_features"' in body, "streamed flows carry SHAP attributions")
        check("event: complete" in body, "a bounded stream terminates cleanly")

        r = client.get("/stream/live?scenario=does-not-exist&limit=1&loop=false")
        check("event: error" in r.text, "unknown scenario reports an error event")

        # -- MLOps panel -------------------------------------------------
        print("\nMLOps control panel")
        for route in ["/mlops/state", "/mlops/registry", "/mlops/governance", "/mlops/pipeline"]:
            r = client.get(route)
            check(r.status_code == 200, f"GET {route} returns 200", f"got {r.status_code}")

    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("\nFailed:")
        for name in failures:
            print(f"  - {name}")
        return 1

    print("\nAPI smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
