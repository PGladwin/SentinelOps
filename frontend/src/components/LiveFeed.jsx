import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Activity, AlertTriangle, Pause, Play, RotateCcw, ShieldAlert, Radio } from "lucide-react";
import { streamUrl } from "../api";
import { tooltipStyle, useTheme } from "../theme-context";
import { ChartCard, MetricCard, Panel, SectionHeader, StatusBadge, StatusIndicator, Th } from "./ui";
import { FlowDetailDrawer } from "./explanation";

/**
 * Live SOC console.
 *
 * Subscribes to the server's SSE feed, where each flow is scored by the
 * production Champion at the moment it is emitted. Nothing here is precomputed
 * or mocked: the verdicts, confidences and SHAP attributions arrive from the
 * same model path that serves /predict.
 */

// The console keeps a bounded window of flows. An unbounded log would grow the
// DOM without limit during a long demo and eventually stall the tab.
const MAX_ROWS = 250;
const TIMELINE_BUCKETS = 40;

const RATE_PRESETS = [4, 8, 20, 50];

const THREAT_TONE = { HIGH: "danger", MEDIUM: "warn", LOW: "safe" };
const THREAT_TEXT = { HIGH: "text-danger", MEDIUM: "text-warn", LOW: "text-safe" };

const fmt = (n) => (typeof n === "number" ? n.toLocaleString() : n ?? "—");
const clockOf = (iso) => (iso ? iso.slice(11, 23) : "—");

/**
 * Consume an SSE stream via fetch rather than EventSource.
 *
 * EventSource cannot be aborted mid-response in a way that reliably closes the
 * upstream connection, and it offers no hook for surfacing a non-200 status --
 * a failed stream would simply retry forever behind the scenes. Reading the
 * body ourselves makes both Pause and error reporting deterministic.
 */
async function consumeStream(url, signal, onEvent) {
  const response = await fetch(url, { signal, headers: { Accept: "text/event-stream" } });
  if (!response.ok || !response.body) {
    throw new Error(`Stream refused: ${response.status} ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) return;

    buffer += decoder.decode(value, { stream: true });

    // Events are separated by a blank line; a partial trailing frame stays in
    // the buffer until the rest of it arrives.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      let event = "message";
      const data = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trim());
      }
      if (!data.length) continue;
      try {
        onEvent(event, JSON.parse(data.join("\n")));
      } catch {
        /* a truncated frame is dropped rather than tearing down the feed */
      }
    }
  }
}

/** Rolling per-second counts, derived from the flow window. */
function buildTimeline(flows) {
  if (!flows.length) return [];

  const buckets = new Map();
  for (const flow of flows) {
    const key = (flow.ts || "").slice(11, 19);
    const bucket = buckets.get(key) || { t: key, benign: 0, attacks: 0 };
    if (flow.is_attack) bucket.attacks += 1;
    else bucket.benign += 1;
    buckets.set(key, bucket);
  }

  const ordered = Array.from(buckets.values());

  // Both end buckets are partial and would misreport the rate: the newest
  // second is still being filled, and the oldest is clipped by the row window
  // (or by whenever the capture happened to start). Plotting either draws a
  // cliff that reads as a traffic drop rather than an artefact of framing.
  if (ordered.length > 2) {
    ordered.pop();
    ordered.shift();
  }

  // Oldest first, and only the most recent buckets: the chart is a "what is
  // happening now" view, not a session history.
  return ordered.slice(-TIMELINE_BUCKETS);
}

function ControlBar({ running, connecting, rate, setRate, scenario, scenarios, setScenario, onToggle, onReset }) {
  return (
    <div className="card p-4 flex flex-wrap items-center gap-x-6 gap-y-4">
      <button
        onClick={onToggle}
        className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-opacity hover:opacity-90 ${
          running ? "border border-line text-ink" : "bg-accent text-white"
        }`}
      >
        {running ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        {running ? "Pause" : connecting ? "Connecting…" : "Start capture"}
      </button>

      <button
        onClick={onReset}
        className="flex items-center gap-1.5 px-3 py-2 rounded-md border border-line text-xs text-muted hover:text-ink hover:bg-elevated transition-colors"
      >
        <RotateCcw className="h-3.5 w-3.5" />
        Reset
      </button>

      <label className="flex items-center gap-2 text-xs">
        <span className="text-faint uppercase tracking-[0.1em] text-[10px] font-medium">Source</span>
        <select
          value={scenario}
          onChange={(e) => setScenario(e.target.value)}
          disabled={running}
          className="bg-elevated border border-line rounded-md px-2 py-1.5 text-xs text-ink disabled:opacity-50 max-w-[16rem]"
        >
          {scenarios.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name} ({fmt(s.total_flows)} flows
              {s.attack_rate != null ? `, ${(s.attack_rate * 100).toFixed(1)}% hostile` : ""})
            </option>
          ))}
        </select>
      </label>

      <div className="flex items-center gap-2">
        <span className="text-faint uppercase tracking-[0.1em] text-[10px] font-medium">Rate</span>
        <div className="flex rounded-md border border-line overflow-hidden">
          {RATE_PRESETS.map((preset) => (
            <button
              key={preset}
              onClick={() => setRate(preset)}
              className={`px-2.5 py-1.5 text-xs tabular-nums transition-colors ${
                rate === preset ? "bg-accent text-white" : "text-muted hover:text-ink hover:bg-elevated"
              }`}
            >
              {preset}/s
            </button>
          ))}
        </div>
      </div>

      <StatusIndicator
        tone={running ? "safe" : connecting ? "warn" : "neutral"}
        label={running ? "Capturing" : connecting ? "Connecting" : "Idle"}
        pulse={running || connecting}
      />
    </div>
  );
}

export default function LiveFeed() {
  const { chart } = useTheme();

  const [scenarios, setScenarios] = useState([]);
  const [scenario, setScenario] = useState("");
  const [rate, setRate] = useState(8);

  const [running, setRunning] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState(null);

  const [flows, setFlows] = useState([]);
  const [stats, setStats] = useState(null);
  const [selected, setSelected] = useState(null);
  const [threatsOnly, setThreatsOnly] = useState(false);

  const abortRef = useRef(null);
  // Cursor so Pause/Resume continues where the capture stopped instead of
  // replaying the same opening flows every time.
  const cursorRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    fetch(streamUrl("/stream/scenarios"))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`Scenario list unavailable (${r.status})`))))
      .then((data) => {
        if (cancelled) return;
        const usable = (data.scenarios || []).filter((s) => !s.error && s.total_flows > 0);
        setScenarios(usable);
        setScenario(data.default && usable.some((s) => s.name === data.default) ? data.default : usable[0]?.name || "");
        setRate(data.default_rate || 8);
      })
      .catch((err) => !cancelled && setError(err.message));
    return () => { cancelled = true; };
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunning(false);
    setConnecting(false);
  }, []);

  // Tear the connection down if the tab unmounts mid-capture, so the server
  // stops scoring windows for a subscriber that has gone away.
  useEffect(() => stop, [stop]);

  const start = useCallback(() => {
    if (!scenario) return;

    const controller = new AbortController();
    abortRef.current = controller;
    setConnecting(true);
    setError(null);

    const url = streamUrl(
      `/stream/live?scenario=${encodeURIComponent(scenario)}&rate=${rate}&start=${cursorRef.current}&loop=true`
    );

    consumeStream(url, controller.signal, (event, payload) => {
      if (event === "ready") {
        setConnecting(false);
        setRunning(true);
        return;
      }
      if (event === "error") {
        setError(payload.detail);
        stop();
        return;
      }
      if (event === "batch") {
        setStats(payload.stats);
        cursorRef.current = payload.stats.position;
        setFlows((prev) => [...payload.flows].reverse().concat(prev).slice(0, MAX_ROWS));
      }
    })
      .catch((err) => {
        if (err.name === "AbortError") return;
        setError(err.message);
        stop();
      })
      .finally(() => {
        if (abortRef.current === controller) stop();
      });
  }, [scenario, rate, stop]);

  const toggle = () => (running || connecting ? stop() : start());

  const reset = () => {
    stop();
    cursorRef.current = 0;
    setFlows([]);
    setStats(null);
    setError(null);
  };

  // Restarting on a settings change keeps the controls honest: a rate or source
  // picked mid-capture takes effect immediately rather than silently at the
  // next manual restart.
  useEffect(() => {
    if (!running) return;
    stop();
    const id = setTimeout(start, 60);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rate, scenario]);

  const timeline = useMemo(() => buildTimeline([...flows].reverse()), [flows]);
  const visible = useMemo(
    () => (threatsOnly ? flows.filter((f) => f.is_attack) : flows),
    [flows, threatsOnly]
  );

  const threatLevel = stats?.threat_level ?? "LOW";
  const recentAttacks = useMemo(() => flows.filter((f) => f.is_attack).slice(0, 5), [flows]);

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Live"
        title="Live Traffic Monitor"
        description="Network flows replayed as a real-time feed. Every connection is classified and explained by the production Champion at the moment it arrives."
        action={
          stats && (
            <div className="flex items-center gap-3">
              <ShieldAlert className={`h-6 w-6 ${THREAT_TEXT[threatLevel]}`} strokeWidth={1.75} />
              <StatusBadge tone={THREAT_TONE[threatLevel]}>{threatLevel}</StatusBadge>
            </div>
          )
        }
      />

      <ControlBar
        running={running}
        connecting={connecting}
        rate={rate}
        setRate={setRate}
        scenario={scenario}
        scenarios={scenarios}
        setScenario={setScenario}
        onToggle={toggle}
        onReset={reset}
      />

      {error && (
        <div className="flex items-start gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger-soft text-danger text-xs">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          {error}
        </div>
      )}

      {/* Session counters */}
      <div className="card p-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-x-8 gap-y-4">
        <MetricCard label="Flows inspected" value={fmt(stats?.flows ?? 0)} dense />
        <MetricCard label="Threats detected" value={fmt(stats?.attacks ?? 0)} tone="danger" dense />
        <MetricCard
          label="Attack rate"
          value={stats ? `${(stats.attack_rate * 100).toFixed(1)}%` : "—"}
          dense
        />
        <MetricCard
          label="Live accuracy"
          value={stats?.accuracy != null ? `${(stats.accuracy * 100).toFixed(2)}%` : "—"}
          tone="safe"
          dense
          sub={stats?.labelled ? `vs ground truth on ${fmt(stats.labelled)} flows` : "no labels in source"}
        />
        <MetricCard
          label="Replay position"
          value={stats ? `${fmt(stats.position)} / ${fmt(stats.total)}` : "—"}
          dense
          sub={stats?.loops ? `${stats.loops} loop${stats.loops > 1 ? "s" : ""} completed` : "single pass"}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2">
          <ChartCard
            title="Traffic timeline"
            description="Flows per second, split benign against detected threats"
          >
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={timeline} margin={{ left: -22, right: 6, top: 6 }}>
                <defs>
                  <linearGradient id="benignFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={chart.safe} stopOpacity={0.35} />
                    <stop offset="100%" stopColor={chart.safe} stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="attackFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={chart.danger} stopOpacity={0.5} />
                    <stop offset="100%" stopColor={chart.danger} stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="2 4" stroke={chart.grid} vertical={false} />
                <XAxis dataKey="t" tick={{ fill: chart.axis, fontSize: 9 }} tickLine={false} axisLine={{ stroke: chart.grid }} minTickGap={28} />
                <YAxis tick={{ fill: chart.axis, fontSize: 10 }} tickLine={false} axisLine={false} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle(chart)} />
                {/* Step, not a spline: these are discrete per-second counts, and
                    monotone smoothing invents intermediate values that were never
                    measured -- including dips below zero between two busy seconds. */}
                <Area type="stepAfter" dataKey="benign" stackId="1" stroke={chart.safe} fill="url(#benignFill)" strokeWidth={1.5} isAnimationActive={false} name="Benign" />
                <Area type="stepAfter" dataKey="attacks" stackId="1" stroke={chart.danger} fill="url(#attackFill)" strokeWidth={1.5} isAnimationActive={false} name="Threats" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        <Panel title="Latest threats" description="Most recent hostile verdicts" bodyClassName="p-0">
          {recentAttacks.length === 0 ? (
            <p className="px-5 py-8 text-xs text-faint text-center">
              {running ? "No threats in the current window." : "Start the capture to see detections."}
            </p>
          ) : (
            <ul className="divide-y divide-line">
              {recentAttacks.map((flow) => (
                <li key={flow.index + flow.ts}>
                  <button
                    onClick={() => setSelected(flow)}
                    className="w-full text-left px-5 py-3 hover:bg-elevated transition-colors"
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="text-danger text-[13px] font-medium">{flow.prediction}</span>
                      <span className="text-[11px] font-mono tabular-nums text-muted shrink-0">
                        {(flow.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-[11px] text-faint font-mono truncate mt-0.5">
                      {flow.src_ip} → {flow.dst_ip}:{flow.dst_port}
                    </p>
                    <p className="text-[11px] text-muted truncate mt-0.5">
                      {flow.top_features?.[0]?.feature ?? "—"}
                    </p>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      {/* Flow console */}
      <Panel
        title="Connection stream"
        description={`${fmt(visible.length)} flows in view · newest first · select a row to investigate`}
        bodyClassName="p-0"
        action={
          <div className="flex items-center gap-2">
            <button
              onClick={() => setThreatsOnly((v) => !v)}
              className={`px-3 py-1.5 rounded-md text-xs border transition-colors ${
                threatsOnly
                  ? "border-danger text-danger bg-danger-soft"
                  : "border-line text-muted hover:text-ink hover:bg-elevated"
              }`}
            >
              Threats only
            </button>
          </div>
        }
      >
        {visible.length === 0 ? (
          <div className="flex flex-col items-center justify-center text-center py-14 px-6">
            <Radio className="h-7 w-7 text-faint mb-3" strokeWidth={1.5} />
            <p className="text-sm font-medium text-ink">
              {running ? "Waiting for flows…" : "Capture stopped"}
            </p>
            <p className="text-xs text-muted mt-1 max-w-sm leading-relaxed">
              {running
                ? "The sensor is connected; classified flows will appear here as they arrive."
                : "Press Start capture to begin replaying traffic through the production model."}
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto max-h-[30rem] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface z-10">
                <tr className="border-b border-line">
                  <Th>Time</Th>
                  <Th>Source</Th>
                  <Th>Destination</Th>
                  <Th>Proto</Th>
                  <Th>Verdict</Th>
                  <Th right>Confidence</Th>
                  <Th>Truth</Th>
                </tr>
              </thead>
              <tbody>
                {visible.map((flow) => (
                  <tr
                    key={`${flow.index}-${flow.ts}`}
                    onClick={() => setSelected(flow)}
                    className={`border-b border-line last:border-0 cursor-pointer transition-colors ${
                      flow.is_attack ? "bg-danger-soft/40 hover:bg-danger-soft" : "hover:bg-elevated"
                    }`}
                  >
                    <td className="py-2 px-4 font-mono text-[11px] text-faint tabular-nums whitespace-nowrap">
                      {clockOf(flow.ts)}
                    </td>
                    <td className="py-2 px-4 font-mono text-[11px] text-muted whitespace-nowrap">{flow.src_ip}</td>
                    <td className="py-2 px-4 font-mono text-[11px] text-muted whitespace-nowrap">
                      {flow.dst_ip}:{flow.dst_port}
                    </td>
                    <td className="py-2 px-4 text-[11px] text-faint">{flow.protocol}</td>
                    <td className="py-2 px-4 whitespace-nowrap">
                      <span className="inline-flex items-center gap-1.5">
                        <span className={`h-1.5 w-1.5 rounded-full ${flow.is_attack ? "bg-danger" : "bg-safe"}`} />
                        <span className={flow.is_attack ? "text-danger font-medium text-[13px]" : "text-ink text-[13px]"}>
                          {flow.prediction}
                        </span>
                      </span>
                    </td>
                    <td className="py-2 px-4 text-right font-mono text-[11px] tabular-nums text-muted">
                      {(flow.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-4 text-[11px] whitespace-nowrap">
                      {flow.actual == null ? (
                        <span className="text-faint">—</span>
                      ) : flow.correct ? (
                        <span className="text-safe">✓ {flow.actual}</span>
                      ) : (
                        <span className="text-warn">✗ {flow.actual}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <p className="flex items-start gap-2 text-[11px] text-faint leading-relaxed">
        <Activity className="h-3.5 w-3.5 shrink-0 mt-0.5" />
        <span>
          Flow features and ground-truth labels are genuine CIC-IDS2017 records. Addressing
          (source, destination, port) is reconstructed from the published testbed topology for
          display only — the cleaning stage drops identifier columns, and none of them reach
          the model.
        </span>
      </p>

      <FlowDetailDrawer
        flow={selected}
        onClose={() => setSelected(null)}
        eyebrow={selected ? `${clockOf(selected.ts)} · flow #${selected.index}` : "Flow"}
        meta={
          selected
            ? [
                { label: "Source", value: selected.src_ip },
                { label: "Destination", value: `${selected.dst_ip}:${selected.dst_port}` },
                { label: "Protocol", value: selected.protocol },
                ...(selected.actual != null
                  ? [{
                      label: "Ground truth",
                      value: selected.correct ? `${selected.actual} ✓` : `${selected.actual} ✗`,
                      tone: selected.correct ? "safe" : "danger",
                    }]
                  : []),
              ]
            : []
        }
      />
    </div>
  );
}
