import React, { useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  ChevronLeft, ChevronRight, Download, FileWarning, ShieldAlert, UploadCloud,
} from "lucide-react";
import { fetchSchema, postAnalyze, schemaTemplateUrl } from "../api";
import { tooltipStyle, useTheme } from "../theme-context";
import { ChartCard, EmptyState, MetricCard, Panel, SectionHeader, StatusBadge, Th } from "./ui";
import { FlowDetailDrawer } from "./explanation";

const ROWS_PER_PAGE = 20;

const THREAT_TONE = { HIGH: "danger", MEDIUM: "warn", LOW: "safe" };
const THREAT_TEXT = { HIGH: "text-danger", MEDIUM: "text-warn", LOW: "text-safe" };

const fmt = (n) => (typeof n === "number" ? n.toLocaleString() : n);

const SAMPLE_DATASET_URL = "/demo_traffic_mixed.csv";
const SAMPLE_DATASET_NAME = "demo_traffic_mixed.csv";

// Kept in step with SUPPORTED_UPLOAD_SUFFIXES in api/main.py. Used only for the
// file picker's filter; the server validates for real.
const FALLBACK_FORMATS = [".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".parquet"];

/**
 * How much of the model's schema the upload actually supplied.
 *
 * A partial match still produces verdicts -- absent features are filled from
 * training medians -- so the difference between a complete file and a thin one
 * has to be visible. Without this, a scan driven mostly by defaults looks
 * identical to a clean one.
 */
function SchemaMatch({ summary }) {
  const expected = summary.features_expected ?? 0;
  const matched = summary.features_matched ?? 0;
  if (!expected) return null;

  const coverage = summary.schema_coverage ?? matched / expected;
  const complete = matched >= expected;
  const tone = complete ? "safe" : coverage >= 0.9 ? "warn" : "danger";

  // Written out rather than interpolated: Tailwind generates utilities by
  // scanning source text, so a class built at runtime is never emitted.
  const TEXT = { safe: "text-safe", warn: "text-warn", danger: "text-danger" };
  const BAR = { safe: "bg-safe", warn: "bg-warn", danger: "bg-danger" };

  return (
    <div className="rounded-md border border-line bg-elevated px-4 py-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-faint">
          Schema match
        </p>
        <p className={`text-xs font-mono tabular-nums ${TEXT[tone]}`}>
          {matched} / {expected} features · {(coverage * 100).toFixed(0)}%
        </p>
      </div>

      <div className="h-1 rounded-full bg-canvas overflow-hidden mt-2">
        <div
          className={`h-full rounded-full ${BAR[tone]}`}
          style={{ width: `${Math.max(coverage * 100, 1)}%` }}
        />
      </div>

      <p className="text-[11px] text-muted mt-2 leading-relaxed">
        {complete ? (
          <>
            Every feature the model consumes was found in your file
            {summary.columns_supplied ? ` (${summary.columns_supplied} columns supplied)` : ""}.
            Column names were matched regardless of case, spacing, or separators.
          </>
        ) : (
          <span className="text-warn">
            {summary.n_imputed_columns} feature(s) were absent and filled from training
            medians, so these verdicts are partly driven by defaults rather than by your
            traffic: {summary.imputed_columns?.slice(0, 6).join(", ")}
            {summary.imputed_columns?.length > 6 ? "…" : ""}
          </span>
        )}
      </p>

      {summary.n_unused_columns > 0 && (
        // Informational, not a warning: extra columns are the normal case for a
        // full dataset export, since the champion consumes a filtered subset.
        <p className="text-[11px] text-faint mt-1.5 leading-relaxed">
          {summary.n_unused_columns} further column(s) were carried but not used — this
          model consumes a filtered subset of the dataset's features.
        </p>
      )}
    </div>
  );
}

function Dropzone({ file, setFile, setError, loading, onAnalyze, schema }) {
  const [dragging, setDragging] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);

  // Prefer what the server reports it accepts; the constants are only a
  // fallback for when /schema has not loaded yet.
  const formats = schema?.supported_formats?.length ? schema.supported_formats : FALLBACK_FORMATS;
  const maxMb = schema?.max_upload_mb ?? 50;

  const accept = (f) => {
    if (!f) return;
    setFile(f);
    setError(null);
  };

  const useSampleDataset = async () => {
    setLoadingSample(true);
    setError(null);
    try {
      const res = await fetch(SAMPLE_DATASET_URL);
      if (!res.ok) throw new Error("Could not load the sample dataset.");
      const blob = await res.blob();
      accept(new File([blob], SAMPLE_DATASET_NAME, { type: "text/csv" }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingSample(false);
    }
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        accept(e.dataTransfer.files?.[0]);
      }}
      className={`card border-dashed transition-colors ${dragging ? "border-accent bg-accent-soft" : ""}`}
    >
      <div className="flex flex-col items-center text-center px-6 py-10">
        <UploadCloud className={`h-8 w-8 mb-3 ${dragging ? "text-accent" : "text-faint"}`} strokeWidth={1.5} />
        <p className="text-sm font-medium">
          {file ? file.name : "Drop a traffic capture, or browse"}
        </p>
        <p className="text-xs text-muted mt-1">
          {file
            ? `${(file.size / 1024).toFixed(0)} KB`
            : `${formats.map((f) => f.replace(".", "").toUpperCase()).join(" · ")} · up to ${maxMb} MB`}
        </p>
        {!file && (
          <p className="text-[11px] text-faint mt-1.5 max-w-md leading-relaxed">
            Column names are matched regardless of case, spacing, or separators — so
            <span className="font-mono"> flow_duration</span>,
            <span className="font-mono"> Flow Duration</span> and
            <span className="font-mono"> FLOW-DURATION</span> all resolve to the same
            feature. Identifier columns (IPs, ports, timestamps) are ignored.
          </p>
        )}

        <div className="flex items-center gap-3 mt-5">
          <input
            type="file"
            accept={formats.join(",")}
            id="soc-file"
            className="sr-only"
            onChange={(e) => accept(e.target.files?.[0])}
          />
          <label
            htmlFor="soc-file"
            className="px-3.5 py-2 rounded-md border border-line text-sm cursor-pointer hover:bg-elevated transition-colors"
          >
            Browse file
          </label>
          <button
            onClick={onAnalyze}
            disabled={!file || loading}
            className="px-4 py-2 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-40 hover:opacity-90 transition-opacity"
          >
            {loading ? "Scanning…" : "Run scan"}
          </button>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1.5 mt-4">
          <button
            onClick={useSampleDataset}
            disabled={loadingSample || loading}
            className="text-xs text-accent hover:underline disabled:opacity-40"
          >
            {loadingSample ? "Loading sample…" : "Use the sample dataset (6,000 flows)"}
          </button>
          <span className="text-faint text-xs">·</span>
          <a
            href={schemaTemplateUrl()}
            download
            className="text-xs text-muted hover:text-ink hover:underline"
          >
            Download schema template
            {schema ? ` (${schema.feature_count} columns)` : ""}
          </a>
        </div>
      </div>
    </div>
  );
}

export default function SocDashboard() {
  const { chart } = useTheme();
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [selected, setSelected] = useState(null);
  const [page, setPage] = useState(0);
  const [onlyThreats, setOnlyThreats] = useState(false);
  const [schema, setSchema] = useState(null);

  // Best-effort: the dropzone falls back to sensible constants if this fails,
  // so a schema fetch error must not block uploading.
  useEffect(() => {
    let cancelled = false;
    fetchSchema()
      .then((s) => !cancelled && setSchema(s))
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  const rows = useMemo(() => result?.rows || [], [result]);
  const filtered = useMemo(
    () => (onlyThreats ? rows.filter((r) => r.is_attack) : rows),
    [rows, onlyThreats]
  );
  const pageCount = Math.max(1, Math.ceil(filtered.length / ROWS_PER_PAGE));
  const visible = filtered.slice(page * ROWS_PER_PAGE, (page + 1) * ROWS_PER_PAGE);

  const summary = result?.summary;
  const threatTone = summary ? THREAT_TONE[summary.threat_level] ?? "safe" : "neutral";

  const breakdown = useMemo(
    () =>
      Object.entries(result?.class_breakdown || {})
        .filter(([, v]) => v > 0)
        .map(([name, count]) => ({ name, count, benign: name === "BENIGN" })),
    [result]
  );

  const shapData = useMemo(
    () => (result?.global_shap || []).slice(0, 10).map((g) => ({ name: g.feature, value: g.mean_abs_shap })).reverse(),
    [result]
  );

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setPage(0);
    try {
      setResult(await postAnalyze(file));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const exportFlagged = () => {
    const flagged = rows.filter((r) => r.is_attack);
    const header = ["flow_id", "predicted_class", "confidence", "top_features"];
    const lines = flagged.map((r) =>
      [
        r.flow_id,
        r.prediction,
        r.confidence,
        (r.top_features || []).map((f) => `${f.feature}=${f.value}`).join("; "),
      ]
        .map((v) => `"${String(v).replace(/"/g, '""')}"`)
        .join(",")
    );
    const blob = new Blob([[header.join(","), ...lines].join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sentinelops-flagged-${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Detect"
        title="Network Traffic Detection"
        description="Upload a flow export — every connection is classified and explained against the production model."
      />

      <Dropzone
        file={file}
        setFile={setFile}
        setError={setError}
        loading={loading}
        onAnalyze={analyze}
        schema={schema}
      />
      {error && (
        <div className="flex items-start gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger-soft text-danger text-xs">
          <FileWarning className="h-4 w-4 shrink-0 mt-0.5" />
          {error}
        </div>
      )}

      {result && (
        <>
          {/* Summary strip */}
          <div className="card p-5">
            <div className="flex flex-wrap items-center gap-8">
              <div className="flex items-center gap-3 pr-6 border-r border-line">
                <ShieldAlert className={`h-7 w-7 ${THREAT_TEXT[summary.threat_level] ?? "text-safe"}`} strokeWidth={1.75} />
                <div>
                  <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-faint">Threat level</p>
                  <StatusBadge tone={threatTone}>{summary.threat_level}</StatusBadge>
                </div>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-10 gap-y-3 flex-1">
                <MetricCard label="Connections analyzed" value={fmt(summary.total_connections)} dense />
                <MetricCard label="Threats detected" value={fmt(summary.total_attacks)} tone="danger" dense />
                <MetricCard label="Attack rate" value={`${(summary.attack_rate * 100).toFixed(1)}%`} dense />
              </div>
              <p className="text-[11px] text-faint font-mono ml-auto shrink-0">{summary.filename}</p>
            </div>

            <div className="mt-4 pt-4 border-t border-line">
              <SchemaMatch summary={summary} />
            </div>

            {(summary.ground_truth_available || summary.truncated) && (
              <div className="mt-4 space-y-1.5 text-xs text-muted">
                {summary.ground_truth_available && summary.ground_truth_accuracy != null && (
                  <p>
                    File included ground-truth labels — measured accuracy{" "}
                    <span className="font-mono text-ink">
                      {(summary.ground_truth_accuracy * 100).toFixed(2)}%
                    </span>
                    {/* Labels that do not resolve to the taxonomy are excluded, so
                        the denominator is shown whenever it is not the whole file. */}
                    {summary.ground_truth_scored > 0 &&
                      summary.ground_truth_scored < summary.total_connections && (
                        <span className="text-warn">
                          {" "}
                          (over {fmt(summary.ground_truth_scored)} of{" "}
                          {fmt(summary.total_connections)} rows — the rest carried
                          labels outside the taxonomy)
                        </span>
                      )}
                  </p>
                )}
                {/* Imputed columns are reported by SchemaMatch above, alongside
                    the coverage figure that gives them their meaning. */}
                {summary.truncated && (
                  <p>
                    Showing first {fmt(summary.rows_returned)} of {fmt(summary.total_connections)} rows.
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Analytical area */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div className="space-y-5">
              <ChartCard title="Attack class distribution" description="Predicted class across every analyzed connection">
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={breakdown} margin={{ left: -20, right: 4, top: 4 }}>
                    <CartesianGrid strokeDasharray="2 4" stroke={chart.grid} vertical={false} />
                    <XAxis dataKey="name" tick={{ fill: chart.axis, fontSize: 10 }} angle={-30} textAnchor="end" height={54} tickLine={false} axisLine={{ stroke: chart.grid }} />
                    <YAxis tick={{ fill: chart.axis, fontSize: 10 }} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={tooltipStyle(chart)} cursor={{ fill: chart.grid, opacity: 0.4 }} formatter={(v) => [fmt(v), "flows"]} />
                    <Bar dataKey="count" radius={[2, 2, 0, 0]} minPointSize={2}>
                      {breakdown.map((d) => (
                        <Cell key={d.name} fill={d.benign ? chart.safe : chart.danger} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>

              <ChartCard title="Confidence distribution" description="Model confidence across all predictions">
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={result.confidence_histogram} margin={{ left: -20, right: 4, top: 4 }}>
                    <CartesianGrid strokeDasharray="2 4" stroke={chart.grid} vertical={false} />
                    <XAxis dataKey="bucket" tick={{ fill: chart.axis, fontSize: 10 }} tickLine={false} axisLine={{ stroke: chart.grid }} height={54} angle={-30} textAnchor="end" />
                    <YAxis tick={{ fill: chart.axis, fontSize: 10 }} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={tooltipStyle(chart)} cursor={{ fill: chart.grid, opacity: 0.4 }} formatter={(v) => [fmt(v), "flows"]} />
                    <Bar dataKey="count" fill={chart.accent} radius={[2, 2, 0, 0]} minPointSize={2} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>

            <ChartCard
              title="Most influential features"
              description={`Mean |SHAP| across ${fmt(summary.rows_explained)} explained flows`}
            >
              <ResponsiveContainer width="100%" height={424}>
                <BarChart data={shapData} layout="vertical" margin={{ left: 0, right: 12 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke={chart.grid} horizontal={false} />
                  <XAxis type="number" tick={{ fill: chart.axis, fontSize: 10 }} tickLine={false} axisLine={false} />
                  <YAxis type="category" dataKey="name" width={150} tick={{ fill: chart.axis, fontSize: 9 }} tickLine={false} axisLine={false} interval={0} />
                  <Tooltip contentStyle={tooltipStyle(chart)} cursor={{ fill: chart.grid, opacity: 0.4 }} formatter={(v) => [v.toFixed(4), "mean |SHAP|"]} />
                  <Bar dataKey="value" fill={chart.accent} radius={[0, 2, 2, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {/* Connection log */}
          <Panel
            title="Connection log"
            description={`${fmt(filtered.length)} shown · select a row to open the investigation panel`}
            bodyClassName="p-0"
            action={
              <div className="flex items-center gap-2">
                <button
                  onClick={() => { setOnlyThreats((v) => !v); setPage(0); }}
                  className={`px-3 py-1.5 rounded-md text-xs border transition-colors ${
                    onlyThreats
                      ? "border-danger text-danger bg-danger-soft"
                      : "border-line text-muted hover:text-ink hover:bg-elevated"
                  }`}
                >
                  Threats only
                </button>
                <button
                  onClick={exportFlagged}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-line text-xs text-muted hover:text-ink hover:bg-elevated transition-colors"
                >
                  <Download className="h-3.5 w-3.5" />
                  Export
                </button>
              </div>
            }
          >
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-line">
                    <Th>Flow</Th>
                    <Th>Predicted class</Th>
                    <Th right>Confidence</Th>
                    <Th>Status</Th>
                    <Th>Leading factor</Th>
                    <Th />
                  </tr>
                </thead>
                <tbody>
                  {visible.map((row) => (
                    <tr
                      key={row.flow_id}
                      onClick={() => setSelected(row)}
                      className={`border-b border-line last:border-0 cursor-pointer transition-colors ${
                        row.is_attack ? "bg-danger-soft/40 hover:bg-danger-soft" : "hover:bg-elevated"
                      }`}
                    >
                      <td className="py-2.5 px-4 font-mono text-xs text-faint tabular-nums">{row.flow_id}</td>
                      <td className="py-2.5 px-4">
                        <span className={row.is_attack ? "text-danger font-medium" : "text-ink"}>{row.prediction}</span>
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono text-xs tabular-nums text-muted">
                        {(row.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-4">
                        <span className="inline-flex items-center gap-1.5 text-xs">
                          <span className={`h-1.5 w-1.5 rounded-full ${row.is_attack ? "bg-danger" : "bg-safe"}`} />
                          <span className={row.is_attack ? "text-danger" : "text-safe"}>
                            {row.is_attack ? "Threat" : "Benign"}
                          </span>
                        </span>
                      </td>
                      <td className="py-2.5 px-4 text-xs text-muted truncate max-w-xs">
                        {row.top_features?.[0]?.feature ?? "—"}
                      </td>
                      <td className="py-2.5 px-4 text-right">
                        <ChevronRight className="h-3.5 w-3.5 text-faint" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {pageCount > 1 && (
              <div className="px-5 py-3 border-t border-line flex items-center justify-between text-xs text-muted">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="flex items-center gap-1 px-2 py-1 rounded-md disabled:opacity-30 hover:bg-elevated"
                >
                  <ChevronLeft className="h-3.5 w-3.5" /> Previous
                </button>
                <span className="font-mono tabular-nums">
                  {page + 1} / {pageCount}
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                  disabled={page >= pageCount - 1}
                  className="flex items-center gap-1 px-2 py-1 rounded-md disabled:opacity-30 hover:bg-elevated"
                >
                  Next <ChevronRight className="h-3.5 w-3.5" />
                </button>
              </div>
            )}
          </Panel>
        </>
      )}

      {!result && !loading && (
        <EmptyState
          icon={ShieldAlert}
          title="No scan yet"
          description="Drop a CSV above to see the threat summary, class breakdown, and the connection log."
        />
      )}

      <FlowDetailDrawer
        flow={selected}
        eyebrow={selected ? `Flow ${selected.flow_id}` : "Flow"}
        onClose={() => setSelected(null)}
      />
    </div>
  );
}
