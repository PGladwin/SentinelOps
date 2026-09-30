"""
SentinelOps API - Live Traffic Feed (Server-Sent Events)
=========================================================
Streams classified network flows to the SOC console as they are scored.

Why SSE rather than WebSockets: the feed is strictly one-directional, and SSE
is plain HTTP. It survives proxies and free-tier PaaS routing that often break
WebSocket upgrades, reconnects on its own, and needs no client library.

Protocol
--------
The stream opens with a ``ready`` event describing the scenario, then emits a
``batch`` event on every tick carrying the flows that came due plus the running
session stats. A ``heartbeat`` keeps idle intermediaries from closing the
connection, and ``complete`` closes a non-looping run.

    event: ready      {"scenario": {...}, "rate": 8, "tick_seconds": 0.4}
    event: batch      {"flows": [...], "stats": {...}}
    event: heartbeat  {"t": "2026-09-01T12:04:31.240Z"}
    event: complete   {"stats": {...}}
    event: error      {"detail": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

from anyio import to_thread
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from api.config import settings
from api.model_service import ModelService
from api.traffic_stream import (
    ReplayStats,
    TrafficReplayEngine,
    paced_windows,
    synthetic_timestamps,
)

logger = logging.getLogger("sentinelops.api.stream")

router = APIRouter(prefix="/stream", tags=["Live Feed"])

_engine: Optional[TrafficReplayEngine] = None

# Emitted if no batch has gone out for this long, so a low replay rate does not
# look like a dead connection to either the browser or an intermediate proxy.
HEARTBEAT_SECONDS = 15.0


def get_engine() -> TrafficReplayEngine:
    """Lazily build the replay engine over the loaded Champion."""
    global _engine
    if _engine is None:
        _engine = TrafficReplayEngine(ModelService.get_instance())
    return _engine


def reset_engine() -> None:
    """Drop cached scenarios. Used by tests and after a model reload."""
    global _engine
    _engine = None


def sse(event: str, payload: dict[str, Any]) -> str:
    """Frame one Server-Sent Event. Compact separators keep the feed light."""
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------

@router.get("/scenarios", summary="Traffic scenarios available for replay")
def list_scenarios() -> dict[str, Any]:
    """
    Describe every replayable CSV in the samples directory.

    Each entry is loaded and preprocessed to report its true class mix, so the
    picker shows what a scenario actually contains rather than its filename.
    """
    engine = get_engine()
    scenarios = []

    for name in engine.available():
        try:
            scenarios.append(engine.get(name).describe())
        except Exception as e:
            # One malformed CSV must not blank the whole picker.
            logger.warning(f"Skipping scenario '{name}': {e}")
            scenarios.append({"name": name, "error": str(e), "total_flows": 0})

    return {
        "scenarios": scenarios,
        "default": settings.default_scenario,
        "max_rate": settings.stream_max_rate,
        "default_rate": settings.stream_default_rate,
    }


# ---------------------------------------------------------------------------
# Live feed
# ---------------------------------------------------------------------------

async def replay_events(
    request: Request,
    scenario_name: Optional[str],
    rate: float,
    loop: bool,
    limit: Optional[int],
    start: int,
) -> AsyncIterator[str]:
    """Produce the SSE event sequence for one subscriber."""
    engine = get_engine()

    try:
        scenario = await to_thread.run_sync(engine.get, scenario_name)
    except FileNotFoundError as e:
        yield sse("error", {"detail": str(e)})
        return

    tick = settings.stream_tick_seconds
    # At least one flow per tick, so a rate below 1/tick still advances.
    per_tick = max(1, round(rate * tick))
    thresholds = {"high": 0.20, "medium": 0.05}
    try:
        from api.main import get_soc_thresholds
        thresholds = get_soc_thresholds()
    except Exception:
        pass  # defaults are correct; params.yaml is optional in a slim image

    stats = ReplayStats()
    yield sse("ready", {
        "scenario": scenario.describe(),
        "rate": rate,
        "flows_per_tick": per_tick,
        "tick_seconds": tick,
        "loop": loop,
        "limit": limit,
    })

    since_emit = 0.0

    for window_start, window_end, wrapped in paced_windows(
        scenario.total, per_tick, start, loop, limit
    ):
        # Stop promptly when the browser navigates away or hits Pause, rather
        # than scoring windows nobody is listening to.
        if await request.is_disconnected():
            logger.info(f"Live feed subscriber disconnected after {stats.flows:,} flows.")
            return

        await asyncio.sleep(tick)

        if wrapped:
            stats.loops += 1

        # Inference is CPU-bound and would otherwise block every other request
        # served by this worker for the duration of the window.
        flows = await to_thread.run_sync(engine.score_window, scenario, window_start, window_end)

        if not flows:
            since_emit += tick
            if since_emit >= HEARTBEAT_SECONDS:
                since_emit = 0.0
                yield sse("heartbeat", {"t": synthetic_timestamps(1, tick)[0]})
            continue

        for flow, timestamp in zip(flows, synthetic_timestamps(len(flows), tick)):
            flow["ts"] = timestamp
            stats.observe(flow["prediction"], flow["is_attack"], flow["actual"])

        since_emit = 0.0
        yield sse("batch", {
            "flows": flows,
            "stats": stats.snapshot(window_end, scenario.total, thresholds),
        })

    yield sse("complete", {"stats": stats.snapshot(scenario.total, scenario.total, thresholds)})


@router.get("/live", summary="Live classified traffic feed (SSE)")
async def live_feed(
    request: Request,
    scenario: Optional[str] = Query(None, description="Scenario name; defaults to the configured stream."),
    rate: float = Query(None, gt=0, description="Flows per second."),
    loop: bool = Query(True, description="Restart from the top when the scenario ends."),
    limit: Optional[int] = Query(None, gt=0, description="Stop after this many flows."),
    start: int = Query(0, ge=0, description="Row offset to begin replay from."),
) -> StreamingResponse:
    """
    Replay a traffic scenario as a live, classified feed.

    Every flow is scored by the promoted Champion at the moment it is emitted --
    the same model, preprocessing, and SHAP path that back ``/predict``. Flows
    carry their true CIC-IDS2017 label alongside the prediction so the console
    can report live accuracy rather than confidence alone.
    """
    try:
        ModelService.get_instance()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service unavailable: {e}",
        )

    resolved_rate = min(rate or settings.stream_default_rate, settings.stream_max_rate)

    return StreamingResponse(
        replay_events(request, scenario, resolved_rate, loop, limit, start),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Nginx and several PaaS routers buffer responses by default, which
            # would hold every event until the stream ended.
            "X-Accel-Buffering": "no",
        },
    )
