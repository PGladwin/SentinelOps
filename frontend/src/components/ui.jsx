import React from "react";

/** Small uppercase eyebrow + title + description, used atop every page and panel. */
export function SectionHeader({ eyebrow, title, description, action, size = "md" }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div className="max-w-2xl">
        {eyebrow && (
          <p className="text-[10px] font-medium uppercase tracking-[0.12em] text-faint mb-1">
            {eyebrow}
          </p>
        )}
        <h1 className={size === "lg" ? "text-xl font-semibold tracking-tight" : "text-base font-semibold tracking-tight"}>
          {title}
        </h1>
        {description && <p className="text-xs text-muted mt-1 leading-relaxed">{description}</p>}
      </div>
      {action}
    </div>
  );
}

/** Bordered panel with a header row — the base unit of the whole console. */
export function Panel({ title, description, action, children, className = "", bodyClassName = "p-5" }) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <div className="px-5 py-3.5 border-b border-line flex flex-wrap items-start justify-between gap-3">
          <div className="max-w-3xl">
            {title && <h2 className="text-[13px] font-medium">{title}</h2>}
            {description && <p className="text-[11px] text-muted mt-0.5 leading-relaxed">{description}</p>}
          </div>
          {action}
        </div>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}

const TONE_DOT = {
  safe: "bg-safe",
  danger: "bg-danger",
  warn: "bg-warn",
  neutral: "bg-faint",
  accent: "bg-accent",
};

const TONE_TEXT = {
  safe: "text-safe",
  danger: "text-danger",
  warn: "text-warn",
  neutral: "text-muted",
  accent: "text-accent",
};

const TONE_BG = {
  safe: "border-safe/30 bg-safe-soft",
  danger: "border-danger/30 bg-danger-soft",
  warn: "border-warn/30 bg-warn-soft",
  neutral: "border-line bg-elevated",
  accent: "border-accent/30 bg-accent-soft",
};

/** Small dot + label used for live/connection-style status. */
export function StatusIndicator({ tone = "neutral", label, pulse = false }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted">
      <span className="relative flex h-1.5 w-1.5">
        {pulse && (
          <span className={`absolute inline-flex h-full w-full rounded-full ${TONE_DOT[tone]} opacity-60 animate-ping`} />
        )}
        <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${TONE_DOT[tone]}`} />
      </span>
      {label}
    </span>
  );
}

/** Filled pill for discrete states (PROMOTED / REJECTED / BREACH / etc). */
export function StatusBadge({ tone = "neutral", children }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded border text-[11px] font-medium tracking-wide ${TONE_BG[tone]} ${TONE_TEXT[tone]}`}
    >
      {children}
    </span>
  );
}

/** Label-over-value metric block — the atomic unit of every summary strip. */
export function MetricCard({ label, value, tone = "ink", sub, dense = false }) {
  const toneClass =
    tone === "danger" ? "text-danger" : tone === "safe" ? "text-safe" : tone === "warn" ? "text-warn" : "text-ink";
  return (
    <div>
      <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-faint">{label}</p>
      <p className={`font-semibold tabular-nums mt-1 ${dense ? "text-lg" : "text-2xl"} ${toneClass}`}>{value}</p>
      {sub && <p className="text-[11px] text-faint mt-0.5">{sub}</p>}
    </div>
  );
}

/** Centered placeholder for a panel with nothing to show yet. */
export function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-14 px-6">
      {Icon && <Icon className="h-7 w-7 text-faint mb-3" strokeWidth={1.5} />}
      <p className="text-sm font-medium text-ink">{title}</p>
      {description && <p className="text-xs text-muted mt-1 max-w-sm leading-relaxed">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

/** Table header cell shared by every data table in the console. */
export function Th({ children, right = false }) {
  return (
    <th
      className={`${right ? "text-right" : "text-left"} font-medium py-2 px-4 text-[10px] uppercase tracking-[0.08em] text-faint`}
    >
      {children}
    </th>
  );
}

/** Wraps a chart with a compact heading — keeps chart framing identical across pages. */
export function ChartCard({ title, description, action, children }) {
  return (
    <div className="card p-5">
      <div className="flex items-start justify-between gap-3 mb-4">
        <div>
          <h2 className="text-[13px] font-medium">{title}</h2>
          {description && <p className="text-[11px] text-faint mt-0.5">{description}</p>}
        </div>
        {action}
      </div>
      {children}
    </div>
  );
}

/** One DVC stage node with a connecting line to the next stage. */
export function PipelineStage({ name, cmd, outs, locked, last = false, branch = false }) {
  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center pt-0.5">
        <span
          className={`h-2.5 w-2.5 rounded-full shrink-0 border-2 ${
            locked ? "bg-safe border-safe" : "bg-transparent border-line"
          }`}
        />
        {!last && <span className={`w-px flex-1 mt-1 ${branch ? "bg-line" : "bg-line"}`} style={{ minHeight: 28 }} />}
      </div>
      <div className="min-w-0 pb-6">
        <div className="flex flex-wrap items-baseline gap-x-3">
          <span className="font-mono text-[13px] text-ink">{name}</span>
          <span className={`text-[10px] uppercase tracking-wide ${locked ? "text-safe" : "text-faint"}`}>
            {locked ? "tracked" : "pending"}
          </span>
        </div>
        <p className="text-[11px] text-muted font-mono mt-0.5 truncate">{cmd}</p>
        {outs?.length > 0 && <p className="text-[11px] text-faint mt-0.5">→ {outs.join(", ")}</p>}
      </div>
    </div>
  );
}
