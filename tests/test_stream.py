"""
SentinelOps Live Feed Test Suite
================================
Covers the traffic replay that backs the live SOC console:
  - GET /stream/scenarios (catalogue, class mix, error handling)
  - GET /stream/live (SSE protocol, classified flows, ground truth, limits)
  - Replay window arithmetic (paced_windows) at the file boundary
  - Ground-truth label resolution, including already-canonical spellings

The window arithmetic is tested directly rather than only through the endpoint:
its edge cases live at the wrap point, and reaching them through a timed SSE
stream would make the suite slow and flaky.
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.traffic_stream import ReplayStats, paced_windows, threat_level
from src.data.inference_prep import build_truth_resolver, resolve_truth_labels

client = TestClient(app)

LABEL_MAP = {
    "Benign": "BENIGN",
    "DoS Hulk": "DoS",
    "Web Attack - XSS": "WebAttack",
    "FTP-Patator": "BruteForce",
}


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE response body into (event, payload) pairs."""
    events = []
    for frame in text.split("\n\n"):
        name, data = None, []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if name and data:
            events.append((name, json.loads("\n".join(data))))
    return events


# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------

def test_scenarios_lists_replayable_traffic():
    response = client.get("/stream/scenarios")
    assert response.status_code == 200

    body = response.json()
    assert body["scenarios"], "no replay scenarios discovered"
    assert body["max_rate"] > 0

    scenario = body["scenarios"][0]
    assert scenario["total_flows"] > 0
    assert "class_counts" in scenario


def test_default_scenario_is_present_and_labelled():
    body = client.get("/stream/scenarios").json()
    names = {s["name"] for s in body["scenarios"]}
    assert body["default"] in names

    default = next(s for s in body["scenarios"] if s["name"] == body["default"])
    assert default["labelled"] is True
    # The demo stream is built to a realistic hostile share; a file that has
    # drifted to a balanced mix would misrepresent what a SOC console sees.
    assert 0.0 < default["attack_rate"] < 0.25
    # Every Champion feature must be present, or the feed is silently scoring
    # imputed medians.
    assert default["imputed_columns"] == 0


# ---------------------------------------------------------------------------
# Live feed
# ---------------------------------------------------------------------------

def test_live_feed_streams_classified_flows():
    with client.stream("GET", "/stream/live?rate=40&limit=40&loop=false") as stream:
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        events = parse_sse("".join(stream.iter_text()))

    names = [name for name, _ in events]
    assert names[0] == "ready"
    assert "batch" in names
    assert names[-1] == "complete"

    flows = [f for name, payload in events if name == "batch" for f in payload["flows"]]
    assert flows

    for flow in flows:
        assert flow["prediction"]
        assert 0.0 <= flow["confidence"] <= 1.0
        assert isinstance(flow["is_attack"], bool)
        assert flow["is_attack"] == (flow["prediction"] != "BENIGN")
        assert flow["ts"].endswith("Z")
        # Presentation envelope, reconstructed by the generator.
        assert flow["src_ip"] and flow["dst_ip"]
        assert isinstance(flow["dst_port"], int)


def test_live_feed_reports_accuracy_against_ground_truth():
    with client.stream("GET", "/stream/live?rate=40&limit=60&loop=false") as stream:
        events = parse_sse("".join(stream.iter_text()))

    batches = [payload for name, payload in events if name == "batch"]
    assert batches

    stats = batches[-1]["stats"]
    assert stats["flows"] > 0
    # The default scenario is fully labelled, so every emitted flow must count
    # toward accuracy. A regression in label resolution shows up here as a
    # denominator far below the flow count.
    assert stats["labelled"] == stats["flows"]
    assert stats["accuracy"] is not None
    # The champion scores ~99% on this corpus; a collapse to chance means the
    # truth column stopped lining up with predictions.
    assert stats["accuracy"] > 0.90

    for flow in batches[-1]["flows"]:
        assert flow["actual"] is not None
        assert flow["correct"] == (flow["actual"] == flow["prediction"])


def test_live_feed_honours_the_flow_limit():
    with client.stream("GET", "/stream/live?rate=40&limit=20&loop=false") as stream:
        events = parse_sse("".join(stream.iter_text()))

    flows = [f for name, payload in events if name == "batch" for f in payload["flows"]]
    # The limit is enforced per emitted window, so the final window may overrun
    # it; what matters is that the stream stops promptly rather than replaying
    # the whole file.
    assert 0 < len(flows) < 100


def test_unknown_scenario_reports_an_error_event():
    response = client.get("/stream/live?scenario=no-such-file&limit=1&loop=false")
    events = parse_sse(response.text)
    assert events
    assert events[0][0] == "error"
    assert "not found" in events[0][1]["detail"].lower()


def test_scenario_name_cannot_escape_the_samples_directory():
    """A traversal in the scenario name must not read outside samples/."""
    response = client.get("/stream/live?scenario=../params&limit=1&loop=false")
    events = parse_sse(response.text)
    assert events[0][0] == "error"


# ---------------------------------------------------------------------------
# Replay window arithmetic
# ---------------------------------------------------------------------------

def test_paced_windows_covers_every_row_without_looping():
    windows = list(paced_windows(total=10, flows_per_tick=3, start=0, loop=False, limit=None))
    assert [(s, e) for s, e, _ in windows] == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert not any(wrapped for _, _, wrapped in windows)


def test_paced_windows_wraps_and_flags_the_restart():
    windows = list(paced_windows(total=5, flows_per_tick=2, start=0, loop=True, limit=8))
    assert any(w for _, _, w in windows), "a looping replay must signal when it restarts"
    # Every window stays inside the file after the wrap.
    assert all(0 <= s <= e <= 5 for s, e, _ in windows)


def test_paced_windows_does_not_repeat_the_opening_window_after_a_wrap():
    """
    Regression: the wrap branch reset the cursor to 0 without advancing it past
    the window it had just yielded, so the first rows were emitted twice on
    every loop -- a visible stutter at each restart of a long replay.
    """
    windows = list(paced_windows(total=5, flows_per_tick=2, start=0, loop=True, limit=8))

    wrap_index = next(i for i, (_, _, w) in enumerate(windows) if w)
    assert windows[wrap_index][:2] == (0, 2)
    # Whatever follows the wrap must move on, not re-emit rows 0-2.
    assert all(start >= 2 for start, _, _ in windows[wrap_index + 1:])


def test_paced_windows_resumes_from_a_cursor():
    windows = list(paced_windows(total=10, flows_per_tick=4, start=8, loop=False, limit=None))
    assert windows[0][0] == 8


def test_paced_windows_terminates_at_the_end_without_loop():
    windows = list(paced_windows(total=4, flows_per_tick=10, start=0, loop=False, limit=None))
    assert windows == [(0, 4, False)]


# ---------------------------------------------------------------------------
# Session statistics
# ---------------------------------------------------------------------------

def test_replay_stats_track_rates_and_accuracy():
    stats = ReplayStats()
    stats.observe("BENIGN", False, "BENIGN")
    stats.observe("DDoS", True, "DDoS")
    stats.observe("PortScan", True, "DoS")       # a miss
    stats.observe("BENIGN", False, None)         # unlabelled: excluded

    snapshot = stats.snapshot(position=4, total=100, thresholds={"high": 0.20, "medium": 0.05})
    assert snapshot["flows"] == 4
    assert snapshot["attacks"] == 2
    assert snapshot["labelled"] == 3
    assert snapshot["accuracy"] == pytest.approx(2 / 3, abs=1e-6)
    assert snapshot["by_class"]["BENIGN"] == 2


def test_threat_level_thresholds():
    thresholds = {"high": 0.20, "medium": 0.05}
    assert threat_level(0.30, thresholds) == "HIGH"
    assert threat_level(0.20, thresholds) == "HIGH"
    assert threat_level(0.10, thresholds) == "MEDIUM"
    assert threat_level(0.01, thresholds) == "LOW"


# ---------------------------------------------------------------------------
# Ground-truth label resolution
# ---------------------------------------------------------------------------

def test_resolver_maps_raw_dataset_spellings():
    resolver = build_truth_resolver(LABEL_MAP)
    assert resolver["dos hulk"] == "DoS"
    assert resolver["ftp-patator"] == "BruteForce"


def test_resolver_maps_already_canonical_labels_to_themselves():
    """
    The regression this guards: label_map holds only RAW spellings, so a CSV
    exported after cleaning ("BENIGN", "WebAttack") resolved to nothing and was
    dropped from scoring -- yielding a confident accuracy computed over a small,
    unrepresentative slice of the file.
    """
    resolver = build_truth_resolver(LABEL_MAP)
    assert resolver["benign"] == "BENIGN"
    assert resolver["webattack"] == "WebAttack"
    assert resolver["brutef" + "orce"] == "BruteForce"


def test_resolve_truth_labels_handles_mixed_spellings():
    truth = pd.Series(["BENIGN", "Benign", "DoS Hulk", "DoS", "Web Attack � XSS", "nonsense"])
    resolved = resolve_truth_labels(truth, LABEL_MAP)

    assert list(resolved[:5]) == ["BENIGN", "BENIGN", "DoS", "DoS", "WebAttack"]
    # An unmappable label yields NaN so the caller excludes it, rather than
    # counting it as a misclassification.
    assert pd.isna(resolved.iloc[5])
