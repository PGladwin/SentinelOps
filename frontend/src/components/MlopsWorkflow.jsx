import React from "react";
import {
  AlertTriangle, Check, CircleDashed, Database, GitBranch, Layers,
  Radar, Rocket, ScrollText, ShieldCheck, SlidersHorizontal,
} from "lucide-react";

/**
 * The MLOps loop, drawn as the closed cycle it actually is.
 *
 * Every other panel on this page reports one slice of the lifecycle -- runs,
 * registry, gate decisions, DVC stages, drift. Read separately they look like
 * five unrelated dashboards. This shows them as one circuit, with each stage
 * carrying its own live state, so the question "where is this model right now
 * and what moves it forward" has a visual answer.
 *
 * Nothing here is decorative: every status and count is read from the exported
 * pipeline state. A stage with no data reports "not run" rather than a
 * plausible-looking default.
 */

const TONE = {
  done: {
    ring: "border-safe/40 bg-safe-soft",
    icon: "text-safe",
    label: "text-safe",
  },
  active: {
    ring: "border-accent/40 bg-accent-soft",
    icon: "text-accent",
    label: "text-accent",
  },
  alert: {
    ring: "border-danger/40 bg-danger-soft",
    icon: "text-danger",
    label: "text-danger",
  },
  warn: {
    ring: "border-warn/40 bg-warn-soft",
    icon: "text-warn",
    label: "text-warn",
  },
  idle: {
    ring: "border-line bg-elevated",
    icon: "text-faint",
    label: "text-faint",
  },
};

/**
 * Build the eight lifecycle stages from exported state.
 *
 * Kept as a pure function of the state document so the stage list, its
 * statuses, and its counts are all traceable to something the pipeline
 * actually wrote.
 */
function buildStages({ pipeline, runs, governance, registry, drift, apiHealthy }) {
  const stageByName = Object.fromEntries((pipeline?.stages || []).map((s) => [s.name, s]));
  const tracked = (name) => Boolean(stageByName[name]?.locked);

  const promotions = governance?.promotions ?? 0;
  const rejections = governance?.rejections ?? 0;
  const decisions = governance?.total_decisions ?? promotions + rejections;

  const versions = registry?.versions?.length ?? 0;
  const champion = registry?.champion_version;

  const driftKnown = drift?.available !== false && drift?.drift_detected != null;
  const driftDetected = Boolean(drift?.drift_detected);

  return [
    {
      key: "data",
      icon: Database,
      title: "Ingest & clean",
      detail: tracked("prepare_data") ? "CIC-IDS2017 · labels normalized" : "Not run",
      status: tracked("prepare_data") ? "done" : "idle",
      badge: "DVC",
    },
    {
      key: "preprocess",
      icon: SlidersHorizontal,
      title: "Preprocess",
      detail: tracked("preprocess") ? "Split · correlation filter · scale" : "Not run",
      status: tracked("preprocess") ? "done" : "idle",
      badge: "DVC",
    },
    {
      key: "train",
      icon: Layers,
      title: "Train & track",
      detail: runs?.length ? `${runs.length} run${runs.length === 1 ? "" : "s"} logged` : "No runs logged",
      status: runs?.length ? "done" : "idle",
      badge: "MLflow",
    },
    {
      key: "gate",
      icon: ShieldCheck,
      title: "Governance gate",
      detail: decisions
        ? `${promotions} promoted · ${rejections} rejected`
        : "No decisions recorded",
      // A gate that has rejected challengers is the gate working, not a
      // problem -- so rejections never colour this stage as an alert.
      status: decisions ? "done" : "idle",
      badge: "Champion/challenger",
    },
    {
      key: "registry",
      icon: ScrollText,
      title: "Registry",
      detail: versions
        ? `v${champion ?? "?"} of ${versions} in Production`
        : "No versions registered",
      status: versions ? "done" : "idle",
      badge: "MLflow",
    },
    {
      key: "deploy",
      icon: Rocket,
      title: "Serve",
      detail: apiHealthy ? "Champion loaded · answering requests" : "API not reachable",
      status: apiHealthy ? "active" : "alert",
      badge: "FastAPI",
    },
    {
      key: "monitor",
      icon: Radar,
      title: "Monitor drift",
      detail: driftKnown
        ? driftDetected
          ? `Drift on ${drift.n_drifted_features}/${drift.n_features} features`
          : `Stable · ${drift.n_drifted_features}/${drift.n_features} drifted`
        : "No drift check recorded",
      status: driftKnown ? (driftDetected ? "alert" : "done") : "idle",
      badge: "Evidently",
    },
    {
      key: "retrain",
      icon: GitBranch,
      title: "Retrain trigger",
      detail: driftDetected
        ? "Drift breached — retraining warranted"
        : driftKnown
          ? "Armed · no trigger"
          : "Awaiting a drift check",
      status: driftDetected ? "warn" : "idle",
      badge: "GitHub Actions",
    },
  ];
}

function StageCard({ stage, index }) {
  const tone = TONE[stage.status] || TONE.idle;
  const Icon = stage.icon;

  return (
    <li className="relative">
      <div className={`h-full rounded-lg border p-3.5 ${tone.ring}`}>
        <div className="flex items-start justify-between gap-2">
          <Icon className={`h-4 w-4 shrink-0 ${tone.icon}`} strokeWidth={1.75} />
          <span className="text-[9px] font-mono text-faint tabular-nums">
            {String(index + 1).padStart(2, "0")}
          </span>
        </div>

        <p className="text-[13px] font-medium mt-2 leading-tight">{stage.title}</p>
        <p className="text-[11px] text-muted mt-1 leading-snug">{stage.detail}</p>

        <div className="flex items-center gap-1.5 mt-2.5 pt-2.5 border-t border-line/70">
          {stage.status === "done" && <Check className={`h-3 w-3 ${tone.icon}`} />}
          {stage.status === "alert" && <AlertTriangle className={`h-3 w-3 ${tone.icon}`} />}
          {stage.status === "warn" && <AlertTriangle className={`h-3 w-3 ${tone.icon}`} />}
          {stage.status === "idle" && <CircleDashed className={`h-3 w-3 ${tone.icon}`} />}
          {stage.status === "active" && (
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full rounded-full bg-accent opacity-60 animate-ping" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
            </span>
          )}
          <span className="text-[10px] font-medium uppercase tracking-[0.08em] text-faint truncate">
            {stage.badge}
          </span>
        </div>
      </div>
    </li>
  );
}

export default function MlopsWorkflow({ pipeline, runs, governance, registry, drift, apiHealthy }) {
  const stages = buildStages({ pipeline, runs, governance, registry, drift, apiHealthy });
  const driftDetected = Boolean(drift?.drift_detected);

  return (
    <section className="card">
      <div className="px-5 py-3.5 border-b border-line">
        <h2 className="text-[13px] font-medium">Lifecycle workflow</h2>
        <p className="text-[11px] text-muted mt-0.5 leading-relaxed">
          The full loop, from raw capture to a monitored model and back. Each stage shows its
          own state, read from the last pipeline run.
        </p>
      </div>

      <div className="p-5">
        {/* A fixed grid rather than flex-wrap: with eight stages, wrapping left
            the last one alone on its own row stretched to full width, which read
            as a separate conclusion instead of the eighth step of a cycle. Four
            columns give two even rows that scan as a loop. */}
        <ol className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2.5">
          {stages.map((stage, i) => (
            <StageCard key={stage.key} stage={stage} index={i} />
          ))}
        </ol>

        {/* The return edge is what makes this a lifecycle rather than a
            checklist, so it is drawn explicitly rather than implied. */}
        <div className="mt-4 flex items-start gap-2.5 rounded-md border border-line bg-elevated px-3.5 py-3">
          <GitBranch className="h-3.5 w-3.5 text-faint shrink-0 mt-0.5 -scale-x-100" />
          <p className="text-[11px] text-muted leading-relaxed">
            <span className="text-ink font-medium">Stage 08 closes back onto stage 03.</span>{" "}
            {driftDetected ? (
              <>
                Drift has breached its threshold, so retraining is warranted. A challenger
                still has to clear the governance gate at stage 04 — drift means the traffic
                changed, not that a new model is better.
              </>
            ) : (
              <>
                When drift breaches its threshold the scheduled workflow retrains, and the
                challenger re-enters at the governance gate. A new model reaches production
                only by clearing stage 04, never by being newer.
              </>
            )}
          </p>
        </div>
      </div>
    </section>
  );
}
