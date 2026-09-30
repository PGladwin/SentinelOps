"""
SentinelOps Upload Robustness Test Suite
========================================
Covers the "bring your own network data" path: whatever a user's exporter
produced should be analyzable without them reshaping it first.

  - Column resolution across case, separator and alias variants
  - Ingestion of CSV, TSV, semicolon-delimited text, JSON, JSONL and Parquet
  - The schema contract endpoints
  - Coverage reporting, and refusal when too little of the schema is present

The header-spelling tests matter most: exact matching recognises exactly one
spelling of each feature, and a file that matches nothing is not rejected --
every feature is filled from a training median and scored, producing confident
verdicts about defaults. These assert the resolver keeps that from happening.
"""

import io
import json
import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.data.inference_prep import build_column_resolver, canonical_key, normalize_columns

client = TestClient(app)

SAMPLE = "samples/live_traffic_stream.csv"
ENVELOPE = ["src_ip", "dst_ip", "dst_port", "protocol_name"]


@pytest.fixture(scope="module")
def flows() -> pd.DataFrame:
    """A small slice of the demo stream, without the display envelope."""
    return pd.read_csv(SAMPLE, nrows=120).drop(columns=ENVELOPE)


@pytest.fixture(scope="module")
def features() -> list[str]:
    return client.get("/schema").json()["features"]


def post(name: str, payload: bytes) -> dict:
    response = client.post("/analyze", files={"file": (name, io.BytesIO(payload), "application/octet-stream")})
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Column key normalization
# ---------------------------------------------------------------------------

def test_canonical_key_collapses_case_and_separators():
    keys = {
        canonical_key("Flow Duration"),
        canonical_key("flow_duration"),
        canonical_key("FLOW-DURATION"),
        canonical_key("Flow.Duration"),
        canonical_key("  flow   duration  "),
    }
    assert len(keys) == 1


def test_canonical_key_keeps_genuinely_different_names_apart():
    # Word order is not normalised: these are related through the alias table,
    # where the mapping is stated rather than guessed.
    assert canonical_key("Packet Length Min") != canonical_key("Min Packet Length")
    # The duplicated CIC-IDS2017 header must stay distinguishable.
    assert canonical_key("Fwd Header Length.1") != canonical_key("Fwd Header Length")


def test_resolver_prefers_the_champion_schema_over_an_alias():
    """A target feature must never be redirected away by an alias entry."""
    resolver = build_column_resolver(["Fwd Header Length"])
    assert resolver[canonical_key("Fwd Header Length")] == "Fwd Header Length"


def test_normalize_columns_resolves_snake_case(features):
    df = pd.DataFrame(columns=[re.sub(r"[^a-z0-9]+", "_", f.lower()).strip("_") for f in features])
    resolved = normalize_columns(df, target_features=features)
    assert list(resolved.columns) == features


def test_normalize_columns_leaves_unknown_headers_alone(features):
    """
    An unrecognised column passes through unchanged rather than being forced
    onto a near match. A wrong mapping feeds one feature's values into
    another's slot -- worse than a missing column, because it is silently
    plausible instead of visibly absent.
    """
    df = pd.DataFrame(columns=["totally_unrelated_column"])
    resolved = normalize_columns(df, target_features=features)
    assert list(resolved.columns) == ["totally_unrelated_column"]


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

def test_schema_endpoint_describes_the_contract():
    body = client.get("/schema").json()
    assert body["feature_count"] == len(body["features"])
    assert body["value_contract"] == "raw"
    assert ".csv" in body["supported_formats"]
    assert ".parquet" in body["supported_formats"]
    assert "BENIGN" in body["classes"]


def test_schema_template_matches_the_feature_contract(features):
    response = client.get("/schema/template.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.text.strip().split(",") == features


# ---------------------------------------------------------------------------
# Header spellings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rename,label",
    [
        (lambda c: re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_"), "snake_case"),
        (lambda c: re.sub(r"[^A-Z0-9]+", "-", c.upper()).strip("-"), "UPPER-HYPHEN"),
        (lambda c: c.replace(" ", ""), "nospaces"),
        (lambda c: f"  {c}  ", "padded"),
    ],
)
def test_analyze_matches_every_feature_regardless_of_spelling(flows, rename, label):
    df = flows.copy()
    df.columns = [rename(c) for c in df.columns]

    summary = post(f"{label}.csv", df.to_csv(index=False).encode())["summary"]

    assert summary["features_matched"] == summary["features_expected"], (
        f"{label} headers failed to resolve; "
        f"missing {summary['imputed_columns'][:5]}"
    )
    assert summary["schema_coverage"] == 1.0
    assert summary["n_imputed_columns"] == 0


# ---------------------------------------------------------------------------
# File formats
# ---------------------------------------------------------------------------

def test_analyze_accepts_tab_separated(flows):
    summary = post("flows.tsv", flows.to_csv(index=False, sep="\t").encode())["summary"]
    assert summary["total_connections"] == len(flows)
    assert summary["n_imputed_columns"] == 0


def test_analyze_accepts_semicolon_delimited(flows):
    summary = post("flows.csv", flows.to_csv(index=False, sep=";").encode())["summary"]
    assert summary["total_connections"] == len(flows)
    assert summary["n_imputed_columns"] == 0


def test_analyze_accepts_json_records(flows):
    payload = json.dumps(flows.to_dict(orient="records")).encode()
    assert post("flows.json", payload)["summary"]["total_connections"] == len(flows)


def test_analyze_accepts_json_wrapped_in_a_key(flows):
    """Exporters commonly nest the rows under "flows"/"data"; unwrap them."""
    payload = json.dumps({"flows": flows.to_dict(orient="records")}).encode()
    summary = post("wrapped.json", payload)["summary"]
    assert summary["total_connections"] == len(flows)
    assert summary["n_imputed_columns"] == 0


def test_analyze_accepts_newline_delimited_json(flows):
    payload = flows.to_json(orient="records", lines=True).encode()
    assert post("flows.jsonl", payload)["summary"]["total_connections"] == len(flows)


def test_analyze_accepts_parquet(flows):
    buffer = io.BytesIO()
    flows.to_parquet(buffer, index=False)
    assert post("flows.parquet", buffer.getvalue())["summary"]["total_connections"] == len(flows)


def test_every_format_reaches_the_same_verdicts(flows):
    """
    Format is a container, not a signal. The same flows must produce identical
    detections whichever way they were written -- otherwise a user's choice of
    exporter quietly changes the security answer.
    """
    csv_result = post("a.csv", flows.to_csv(index=False).encode())
    json_result = post("a.json", json.dumps(flows.to_dict(orient="records")).encode())

    buffer = io.BytesIO()
    flows.to_parquet(buffer, index=False)
    parquet_result = post("a.parquet", buffer.getvalue())

    verdicts = [
        [row["prediction"] for row in result["rows"]]
        for result in (csv_result, json_result, parquet_result)
    ]
    assert verdicts[0] == verdicts[1] == verdicts[2]


# ---------------------------------------------------------------------------
# Coverage reporting and refusal
# ---------------------------------------------------------------------------

def test_summary_reports_full_coverage_for_a_complete_file(flows):
    summary = post("complete.csv", flows.to_csv(index=False).encode())["summary"]
    assert summary["schema_coverage"] == 1.0
    assert summary["columns_supplied"] == len(flows.columns)


def test_extra_columns_are_carried_and_reported_as_unused(flows):
    baseline = post("base.csv", flows.to_csv(index=False).encode())["summary"]

    df = flows.copy()
    df["my_custom_metric"] = 1.0
    df["another_unknown"] = 2.0
    summary = post("extra.csv", df.to_csv(index=False).encode())["summary"]

    # Extra columns never block an analysis, and both are counted as unused.
    # Asserted on the count rather than the listed sample, which is capped at
    # 20 entries and so may not contain any given name.
    assert summary["n_unused_columns"] == baseline["n_unused_columns"] + 2
    assert summary["columns_supplied"] == baseline["columns_supplied"] + 2
    assert summary["features_matched"] == summary["features_expected"]


def test_unused_columns_are_the_norm_for_a_full_dataset_export(flows):
    """
    A complete CIC-IDS2017 export carries more columns than the Champion uses,
    because the correlation filter dropped some during training. Those must be
    reported as unused rather than flagged as unknown -- a perfectly-formed
    file should not look like a malformed one.
    """
    summary = post("full.csv", flows.to_csv(index=False).encode())["summary"]

    assert summary["columns_supplied"] > summary["features_expected"]
    assert summary["n_unused_columns"] > 0
    # The file is still a complete match for what the model actually needs.
    assert summary["schema_coverage"] == 1.0


def test_identifier_columns_are_ignored_without_warning():
    """IPs, ports and timestamps are expected in real exports and are not features."""
    df = pd.read_csv(SAMPLE, nrows=60)
    df = df.rename(columns={"src_ip": "Source IP", "dst_ip": "Destination IP", "dst_port": "Destination Port"})

    summary = post("with_ids.csv", df.to_csv(index=False).encode())["summary"]
    assert summary["features_matched"] == summary["features_expected"]
    assert "Source IP" not in summary["unused_columns"]
    assert "Destination Port" not in summary["unused_columns"]


def test_analyze_refuses_a_file_that_is_not_network_data():
    """
    An unrelated CSV must be refused, not scored.

    Missing features are imputed from training medians, so without this guard
    an arbitrary file returns a confident "no threats detected" describing the
    medians rather than the upload -- and a user would act on it.
    """
    payload = b"foo,bar,baz\n1,2,3\n4,5,6\n"
    response = client.post("/analyze", files={"file": ("junk.csv", payload, "text/csv")})

    assert response.status_code == 422
    detail = response.json()["detail"].lower()
    assert "missing" in detail
