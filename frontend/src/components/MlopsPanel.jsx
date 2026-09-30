import React, { useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { ChevronDown, ChevronRight, ExternalLink, GitBranch, Lock, RefreshCw } from "lucide-react";
import { driftReportUrl, fetchGithubRuns, fetchMlopsState } from "../api";
import { tooltipStyle, useTheme } from "../theme-context";
import { MetricCard, Panel, PipelineStage, SectionHeader, StatusBadge, Th } from "./ui";
import MlopsWorkflow from "./MlopsWorkflow";

const num = (v, d = 4) => (typeof v === "number" ? v.toFixed(d) : "—");
const when = (iso) => (iso ? new Date(iso).toLocaleString() : "—");

const BRANCH_STAGES = ["train", "drift_reference"];

/** Human-readable reason a challenger was rejected — the first gate it failed. */
function decisionReason(h) {
  if (h.promoted) return "Cleared every governance gate.";
  const failed = (h.gates || []).find((g) => !g.passed);
  return failed ? failed.detail : "Did not clear governance.";
}

export default function MlopsPanel({ modelInfo }) {
  const { chart } = useTheme();
  const [state, setState] = useState(null);
  const [github, setGithub] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(null);
  const [showFeatures, setShowFeatures] = useState(false);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      setState(await fetchMlopsState());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
    setGithub(await fetchGithubRuns(8));
  };

  useEffect(() => { load(); }, []);

  const runs = state?.runs || [];
  const governance = state?.governance || {};
  const registry = state?.registry || {};
  const production = useMemo(() => state?.production_metrics || {}, [state]);
  const pipeline = state?.pipeline || {};
  const drift = state?.drift || {};

  const fnrData = useMemo(
    () =>
      (production.classes || [])
        .filter((c) => c.fnr != null)
        .map((c) => ({ name: c.class, fnr: c.fnr, status: c.status, support: c.support })),
    [production]
  );

  if (loading) {
    return <p className="text-sm text-faint py-12 text-center">Loading lifecycle state…</p>;
  }

  if (error) {
    return (
      <div className="card p-6 space-y-3">
        <p className="text-sm text-danger">Lifecycle state unavailable</p>
        <p className="text-xs text-muted">{error}</p>
        <button onClick={load} className="px-3 py-1.5 rounded-md border border-line text-xs hover:bg-elevated">
          Retry
        </button>
      </div>
    );
  }

  const ceiling = production.max_per_class_fnr ?? 0.3;
  const worstFnr = Math.max(...fnrData.map((d) => d.fnr), 0.001);
  // Only give the ceiling axis room when it is within reach of the data.
  // Pinning the domain to a distant ceiling spends most of the chart on empty
  // headroom and squashes every class onto the baseline, which hides the
  // comparison the chart exists to make.
  const ceilingInFrame = ceiling <= worstFnr * 2;
  const axisMax = (ceilingInFrame ? Math.max(ceiling, worstFnr) : worstFnr) * 1.15;

  const sequentialStages = (pipeline.stages || []).filter((s) => !BRANCH_STAGES.includes(s.name));
  const branchStages = (pipeline.stages || []).filter((s) => BRANCH_STAGES.includes(s.name));

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="MLOps"
        title="Model Lifecycle Console"
        description={`Every figure below comes from an actual pipeline run — the MLflow store and the append-only governance log. Exported ${when(state?.generated_at)}.`}
        action={
          <button
            onClick={load}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-line text-xs text-muted hover:text-ink hover:bg-elevated transition-colors"
          >
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </button>
        }
      />

      {/* Production model */}
      {modelInfo && (
        <div className="card p-6">
          <div className="flex flex-wrap items-start justify-between gap-6">
            <div className="flex items-center gap-4">
              <span className="h-2.5 w-2.5 rounded-full bg-safe shrink-0" />
              <div>
                <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-faint">Production</p>
                <p className="text-2xl font-semibold tracking-tight mt-0.5">{modelInfo.model_type}</p>
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-10 gap-y-3">
              <MetricCard label="Features" value={modelInfo.feature_count} dense />
              <MetricCard label="Macro F1" value={num(production.summary?.macro_f1)} dense />
              <MetricCard label="Attack FNR" value={num(production.summary?.mean_attack_fnr)} tone="danger" dense />
              <MetricCard label="Registry version" value={modelInfo.metadata?.version ?? "—"} dense />
            </div>
          </div>

          {modelInfo.metadata?.hyperparameters && (
            <div className="mt-6 pt-4 border-t border-line">
              <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-faint mb-2">Hyperparameters</p>
              <div className="flex flex-wrap gap-x-6 gap-y-1.5 text-xs font-mono text-muted">
                {Object.entries(modelInfo.metadata.hyperparameters).map(([k, v]) => (
                  <span key={k}>
                    {k}=<span className="text-ink">{String(v)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="mt-4 pt-4 border-t border-line">
            <button
              onClick={() => setShowFeatures((v) => !v)}
              className="flex items-center gap-1.5 text-xs text-muted hover:text-ink"
            >
              {showFeatures ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
              Ordered feature contract ({modelInfo.feature_count})
            </button>
            {showFeatures && (
              <ol className="mt-3 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-x-6 gap-y-1 text-[11px] font-mono text-muted">
                {modelInfo.features.map((f, i) => (
                  <li key={f}>
                    <span className="text-faint">{String(i + 1).padStart(2, "0")}</span> {f}
                  </li>
                ))}
              </ol>
            )}
          </div>
        </div>
      )}

      {/* The loop, before the panels that detail each of its stages. */}
      <MlopsWorkflow
        pipeline={pipeline}
        runs={runs}
        governance={governance}
        registry={registry}
        drift={drift}
        apiHealthy={Boolean(modelInfo)}
      />

      {/* Section 1: Model lifecycle */}
      <Panel
        title="Model lifecycle"
        description="Every training run logged to MLflow, and the champion/challenger decision that followed. A challenger is promoted only if it clears the absolute FNR ceilings and beats the incumbent on macro F1 without regressing attack FNR."
        action={
          <span className="text-xs text-faint font-mono whitespace-nowrap">
            {governance.promotions ?? 0} promoted · {governance.rejections ?? 0} rejected
          </span>
        }
        bodyClassName="p-0"
      >
        <div className="px-5 pt-4 pb-2">
          <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-faint mb-2">Experiment runs ({runs.length})</p>
        </div>
        <div className="overflow-x-auto -mt-1">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line">
                <Th>Run</Th><Th>Model</Th><Th right>Macro F1</Th>
                <Th right>Attack FNR</Th><Th right>Accuracy</Th>
                <Th right>Train (s)</Th><Th>Started</Th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id} className="border-b border-line last:border-0">
                  <td className="py-2 px-5 font-medium">{r.run_name}</td>
                  <td className="py-2 px-5 text-muted text-xs">{r.model_type}</td>
                  <td className="py-2 px-5 text-right font-mono text-xs tabular-nums">{num(r.macro_f1)}</td>
                  <td className="py-2 px-5 text-right font-mono text-xs tabular-nums">{num(r.mean_attack_fnr)}</td>
                  <td className="py-2 px-5 text-right font-mono text-xs tabular-nums text-muted">{num(r.accuracy, 5)}</td>
                  <td className="py-2 px-5 text-right font-mono text-xs tabular-nums text-muted">{num(r.training_time_seconds, 1)}</td>
                  <td className="py-2 px-5 text-xs text-faint">{when(r.started_at)}</td>
                </tr>
              ))}
              {!runs.length && (
                <tr><td colSpan={7} className="py-6 text-center text-xs text-faint">No runs logged.</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {governance.thresholds && (
          <div className="flex flex-wrap gap-x-6 gap-y-1 text-[11px] font-mono text-muted px-5 pt-5 pb-1">
            {Object.entries(governance.thresholds).map(([k, v]) => (
              <span key={k}>{k}=<span className="text-ink">{v}</span></span>
            ))}
          </div>
        )}

        <div className="px-5 pt-3 pb-2 border-t border-line mt-2">
          <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-faint mb-2">Champion / challenger decisions</p>
        </div>
        <div className="space-y-2 px-5 pb-5">
          {(governance.history || []).map((h, i) => (
            <div key={i} className="rounded-md border border-line overflow-hidden">
              <button
                onClick={() => setExpanded(expanded === i ? null : i)}
                className="w-full flex flex-wrap items-center justify-between gap-3 px-4 py-3 hover:bg-elevated transition-colors text-left"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <StatusBadge tone={h.promoted ? "safe" : "danger"}>
                    {h.promoted ? "Promoted" : "Rejected"}
                  </StatusBadge>
                  <div className="min-w-0">
                    <p className="text-sm font-medium truncate">
                      {h.challenger} <span className="text-faint font-normal">vs {h.champion || "—"}</span>
                    </p>
                    <p className="text-[11px] text-muted truncate">{decisionReason(h)}</p>
                  </div>
                </div>
                <div className="flex items-center gap-4 shrink-0">
                  <span className="text-[11px] font-mono tabular-nums text-muted">
                    F1 {num(h.challenger_macro_f1)}/{num(h.champion_macro_f1)} · FNR {num(h.challenger_attack_fnr)}/{num(h.champion_attack_fnr)}
                  </span>
                  <span className="text-[11px] text-faint">{when(h.timestamp)}</span>
                  {expanded === i ? <ChevronDown className="h-3.5 w-3.5 text-faint" /> : <ChevronRight className="h-3.5 w-3.5 text-faint" />}
                </div>
              </button>
              {expanded === i && (
                <ul className="px-4 py-3 border-t border-line bg-elevated space-y-1.5">
                  {(h.gates || []).map((g) => (
                    <li key={g.gate} className="text-xs flex gap-2">
                      <span className={`shrink-0 font-medium ${g.passed ? "text-safe" : "text-danger"}`}>
                        {g.passed ? "PASS" : "FAIL"}
                      </span>
                      <span className="text-muted">
                        <span className="font-mono text-ink">{g.gate}</span> — {g.detail}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
          {!governance.history?.length && (
            <p className="text-xs text-faint py-4 text-center">No governance decisions logged.</p>
          )}
        </div>
      </Panel>

      {/* Section 2: Production registry */}
      <Panel
        title="Production registry"
        description={`Versions of "${registry.model_name || "—"}". The champion alias points at whatever governance last approved, and the API serves that version.`}
      >
        <div className="flex flex-wrap gap-2">
          {(registry.versions || []).map((v) => (
            <div
              key={v.version}
              className={`px-3 py-2 rounded-md border text-xs ${v.is_champion ? "border-safe bg-safe-soft" : "border-line"}`}
            >
              <span className="font-mono font-medium">v{v.version}</span>
              <span className={`ml-2 ${v.is_champion ? "text-safe" : "text-faint"}`}>
                {v.is_champion ? "production" : String(v.stage || "").toLowerCase()}
              </span>
              {v.governance === "REJECTED" && <span className="ml-2 text-danger">rejected</span>}
            </div>
          ))}
          {!registry.versions?.length && <p className="text-xs text-faint">No registered versions.</p>}
        </div>
      </Panel>

      {/* Section 3: Security performance */}
      <Panel
        title="Security performance"
        description={`Per-class false-negative rate — the share of that class the model misses. ${production.model || "—"} on ${production.test_samples?.toLocaleString() || "—"} held-out flows.`}
      >
        {fnrData.length > 0 && (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={fnrData} margin={{ left: -12, right: ceilingInFrame ? 56 : 8, top: 8 }}>
              <CartesianGrid strokeDasharray="2 4" stroke={chart.grid} vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fill: chart.axis, fontSize: 10 }}
                angle={-30}
                textAnchor="end"
                height={58}
                tickLine={false}
                axisLine={{ stroke: chart.grid }}
              />
              {/* Domain pinned to the governance ceiling so the limit stays in
                  frame; auto-fit topped out below it and the ReferenceLine
                  silently fell outside the chart. */}
              <YAxis
                domain={[0, axisMax]}
                tick={{ fill: chart.axis, fontSize: 10 }}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={tooltipStyle(chart)}
                cursor={{ fill: chart.grid, opacity: 0.4 }}
                formatter={(v, _n, p) => [`${(v * 100).toFixed(2)}% (n=${p.payload.support})`, "FNR"]}
              />
              {ceilingInFrame && (
                <ReferenceLine
                  y={ceiling}
                  stroke={chart.danger}
                  strokeDasharray="3 3"
                  label={{
                    value: `ceiling ${(ceiling * 100).toFixed(0)}%`,
                    fill: chart.danger,
                    fontSize: 9,
                    position: "right",
                  }}
                />
              )}
              {/* minPointSize: FNR spans three orders of magnitude here, so
                  without a floor most bars round to zero pixels and the chart
                  reads as empty. */}
              <Bar dataKey="fnr" radius={[2, 2, 0, 0]} minPointSize={3}>
                {fnrData.map((d) => (
                  <Cell
                    key={d.name}
                    fill={d.status === "BREACH" ? chart.danger : d.fnr > 0.1 ? chart.warn : chart.safe}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}

        {fnrData.length > 0 && !ceilingInFrame && (
          <p className="text-[11px] text-faint mt-2">
            Governance ceiling is {(ceiling * 100).toFixed(0)}% — the worst class sits at{" "}
            {(worstFnr * 100).toFixed(1)}%, so the axis is scaled to the data rather than
            the limit.
          </p>
        )}

        <div className="overflow-x-auto -mx-5 mt-4">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-line">
                <Th>Class</Th><Th right>Support</Th><Th right>Recall</Th>
                <Th right>F1</Th><Th right>FNR</Th><Th>Status</Th>
              </tr>
            </thead>
            <tbody>
              {(production.classes || []).map((c) => (
                <tr key={c.class} className="border-b border-line last:border-0">
                  <td className="py-2 px-5 font-medium">{c.class}</td>
                  <td className="py-2 px-5 text-right font-mono text-xs tabular-nums text-muted">
                    {c.support?.toLocaleString() ?? "—"}
                  </td>
                  <td className="py-2 px-5 text-right font-mono text-xs tabular-nums text-muted">{num(c.recall)}</td>
                  <td className="py-2 px-5 text-right font-mono text-xs tabular-nums text-muted">{num(c.f1)}</td>
                  <td className="py-2 px-5 text-right font-mono text-xs tabular-nums">{num(c.fnr)}</td>
                  <td className="py-2 px-5">
                    <StatusBadge tone={c.status === "BREACH" ? "danger" : c.status === "NO_TEST_SUPPORT" ? "neutral" : "safe"}>
                      {c.status === "NO_TEST_SUPPORT" ? "no test support" : String(c.status).toLowerCase()}
                    </StatusBadge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {(production.classes || []).some((c) => c.support > 0 && c.support < 50) && (
          <p className="text-[11px] text-warn mt-4 leading-relaxed">
            Classes with very few held-out samples carry wide error bars — one additional miss
            moves their FNR by many points. Read those rows as directional.
          </p>
        )}
      </Panel>

      {/* Section 4: Data pipeline */}
      <Panel
        title="Data pipeline"
        description="Reproduce end to end with dvc repro. Each stage declares the params.yaml keys it depends on, so changing the correlation threshold re-runs preprocessing without repeating ingestion."
      >
        {pipeline.available ? (
          <div>
            {sequentialStages.map((s) => (
              <PipelineStage
                key={s.name}
                name={s.name}
                cmd={s.cmd}
                outs={s.outs}
                locked={s.locked}
                last={false}
              />
            ))}
            {branchStages.length > 0 && (
              <div className="flex gap-3">
                <div className="flex flex-col items-center pt-0.5">
                  <GitBranch className="h-3.5 w-3.5 text-faint" />
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 flex-1 pb-2">
                  {branchStages.map((s) => (
                    <div key={s.name} className="rounded-md border border-line px-3.5 py-3">
                      <div className="flex items-center gap-2">
                        <span className={`h-2 w-2 rounded-full shrink-0 border-2 ${s.locked ? "bg-safe border-safe" : "bg-transparent border-line"}`} />
                        <span className="font-mono text-[13px]">{s.name}</span>
                        {s.locked && <Lock className="h-3 w-3 text-faint ml-auto" />}
                      </div>
                      <p className="text-[11px] text-muted font-mono mt-1.5 truncate">{s.cmd}</p>
                      {s.outs?.length > 0 && <p className="text-[11px] text-faint mt-0.5">→ {s.outs.join(", ")}</p>}
                      <p className={`text-[10px] uppercase tracking-wide mt-1.5 ${s.locked ? "text-safe" : "text-faint"}`}>
                        {s.locked ? "tracked" : "pending"}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <p className="text-xs text-faint">{pipeline.reason || "Not available."}</p>
        )}
      </Panel>

      {/* Section 5: Drift */}
      <Panel
        title="Data drift"
        description="Incoming traffic compared against the training distribution. One shifted feature is noise; a large share shifting means the world changed and the model should be retrained."
        action={
          drift.available && (
            <a
              href={driftReportUrl()}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-xs text-accent hover:underline whitespace-nowrap"
            >
              Full report <ExternalLink className="h-3 w-3" />
            </a>
          )
        }
      >
        {!drift.available ? (
          <p className="text-xs text-faint">{drift.reason || "No drift check has been run."}</p>
        ) : (
          <div className="space-y-5">
            <div
              className={`flex flex-wrap items-center justify-between gap-6 rounded-md border px-5 py-4 ${
                drift.drift_detected ? "border-danger/30 bg-danger-soft" : "border-safe/30 bg-safe-soft"
              }`}
            >
              <div>
                <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-faint">Verdict</p>
                <p className={`text-2xl font-semibold tracking-tight mt-1 ${drift.drift_detected ? "text-danger" : "text-safe"}`}>
                  {drift.drift_detected ? "Drift detected" : "No drift"}
                </p>
                {drift.current_source && <p className="text-xs text-muted mt-1">{drift.current_source}</p>}
              </div>
              <div className="grid grid-cols-3 gap-x-8 gap-y-1">
                <MetricCard label="Affected features" value={`${drift.n_drifted_features}/${drift.n_features}`} dense />
                <MetricCard label="Drift share" value={`${(drift.drift_share * 100).toFixed(1)}%`} dense />
                <MetricCard label="Threshold" value={`${(drift.drift_share_threshold * 100).toFixed(0)}%`} dense />
              </div>
            </div>
            <p className="text-[11px] text-faint -mt-2">Statistical test: {drift.stattest}</p>

            {drift.top_drifted_features?.length > 0 && (
              <div>
                <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-faint mb-2">Top drifted features</p>
                <div className="space-y-1">
                  {drift.top_drifted_features.slice(0, 8).map((f) => (
                    <div key={f.feature} className="flex justify-between text-xs">
                      <span className="text-muted truncate pr-3">{f.feature}</span>
                      <span className="font-mono tabular-nums shrink-0">
                        <span className="text-danger">{f.score?.toFixed(3)}</span>
                        <span className="text-faint"> / {f.threshold}</span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </Panel>

      {/* Section 6: CI/CD */}
      <Panel
        title="CI/CD"
        description="Queried live from the GitHub REST API rather than exported with the rest of this page, so it reflects the current state of the repository."
        action={
          github?.repo && (
            <a
              href={`https://github.com/${github.repo}/actions`}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-xs text-accent hover:underline whitespace-nowrap"
            >
              {github.repo} <ExternalLink className="h-3 w-3" />
            </a>
          )
        }
      >
        {!github?.available ? (
          <p className="text-xs text-faint">{github?.reason || "Unavailable."}</p>
        ) : github.runs.length === 0 ? (
          <p className="text-xs text-faint">
            No workflow runs yet — CI has not been triggered on this repository.
          </p>
        ) : (
          <div className="overflow-x-auto -mx-5">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-line">
                  <Th>Run</Th><Th>Workflow</Th><Th>Trigger</Th><Th>Branch</Th><Th>Result</Th><Th>When</Th>
                </tr>
              </thead>
              <tbody>
                {github.runs.map((r) => (
                  <tr key={r.id} className="border-b border-line last:border-0">
                    <td className="py-2 px-5">
                      <a
                        href={r.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-accent hover:underline font-mono text-xs"
                      >
                        {r.number}
                      </a>
                    </td>
                    <td className="py-2 px-5 text-xs">{r.name}</td>
                    <td className="py-2 px-5 text-xs text-muted">{r.event}</td>
                    <td className="py-2 px-5 text-xs text-muted font-mono">{r.branch}</td>
                    <td className="py-2 px-5">
                      <StatusBadge tone={r.conclusion === "success" ? "safe" : r.conclusion === "failure" ? "danger" : "neutral"}>
                        {r.conclusion || r.status}
                      </StatusBadge>
                    </td>
                    <td className="py-2 px-5 text-xs text-faint">{when(r.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
