import React from "react";
import { X } from "lucide-react";

/**
 * Shared SHAP presentation.
 *
 * The batch scan and the live feed investigate the same object -- a classified
 * flow carrying top-k signed contributions -- so they read it the same way.
 * Keeping one implementation matters beyond deduplication: the sign convention
 * below is subtle, and two copies would eventually disagree about which
 * direction means "more dangerous".
 */

/** One-sentence rationale an analyst can act on without reading SHAP values. */
function rationale(flow) {
  if (!flow?.top_features?.length) return null;
  const raising = flow.top_features.filter((f) => f.increases_threat);
  const drivers = (raising.length ? raising : flow.top_features)
    .slice(0, 3)
    .map((f) => `${f.feature} (${f.value})`);
  return `Classified ${flow.prediction} at ${(flow.confidence * 100).toFixed(1)}% confidence, driven by ${drivers.join(", ")}.`;
}

/**
 * Full class distribution for one verdict.
 *
 * Confidence alone hides the shape of a decision: 0.55 BENIGN with 0.44 DDoS
 * behind it is a different situation from 0.55 spread evenly over seven
 * classes, and only the second is genuinely uncertain. Rendered only when the
 * source supplies probabilities.
 */
function Distribution({ probabilities, prediction, isAttack }) {
  const ranked = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .filter(([, p], i) => i < 4 || p >= 0.001);

  return (
    <div className="space-y-2">
      {ranked.map(([cls, prob]) => {
        const isTop = cls === prediction;
        return (
          <div key={cls}>
            <div className="flex justify-between text-[11px] mb-1">
              <span className={isTop ? "text-ink font-medium" : "text-muted"}>{cls}</span>
              <span className={`font-mono tabular-nums ${isTop ? "text-ink" : "text-faint"}`}>
                {(prob * 100).toFixed(2)}%
              </span>
            </div>
            <div className="h-1 rounded-full bg-elevated overflow-hidden">
              <div
                className={`h-full rounded-full ${
                  isTop ? (isAttack ? "bg-danger" : "bg-safe") : "bg-line"
                }`}
                // A floor keeps a near-zero class visible as a hairline rather
                // than vanishing, so the reader can tell it was considered.
                style={{ width: `${Math.max(prob * 100, 0.5)}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** Signed contribution bars, diverging from a centre baseline. */
export function Contributions({ flow }) {
  const data = flow?.top_features || [];
  if (!data.length) return <p className="text-xs text-faint">No attribution available.</p>;
  const max = Math.max(...data.map((f) => Math.abs(f.shap_value)), 1e-4);

  return (
    <div className="space-y-3">
      {data.map((f) => {
        const width = (Math.abs(f.shap_value) / max) * 50;
        return (
          <div key={f.feature}>
            <div className="flex justify-between items-baseline gap-3 mb-1">
              <span className="text-xs text-ink truncate">{f.feature}</span>
              <span className={`text-xs font-mono tabular-nums shrink-0 ${f.increases_threat ? "text-danger" : "text-safe"}`}>
                {f.shap_value > 0 ? "+" : ""}{f.shap_value.toFixed(4)}
              </span>
            </div>
            <div className="relative h-1.5 rounded-full bg-elevated">
              <div className="absolute left-1/2 -top-0.5 -bottom-0.5 w-px bg-line" />
              <div
                className={`absolute top-0 bottom-0 rounded-full ${f.increases_threat ? "bg-danger left-1/2" : "bg-safe"}`}
                style={f.increases_threat ? { width: `${width}%` } : { width: `${width}%`, right: "50%" }}
              />
            </div>
            <p className="text-[11px] text-faint font-mono mt-1">value {f.value}</p>
          </div>
        );
      })}
      <p className="text-[11px] text-faint leading-relaxed pt-1 border-t border-line">
        Right of centre raises threat, left lowers it. Direction accounts for the predicted
        class, so on a benign verdict a positive value lowers threat.
      </p>
    </div>
  );
}

/**
 * Right-hand investigation drawer for a single classified flow.
 *
 * `eyebrow` labels the flow however the calling view identifies it (a row
 * number in the batch scan, a timestamp and socket in the live feed), and
 * `meta` renders any additional key/value context above the rationale.
 */
export function FlowDetailDrawer({ flow, onClose, eyebrow = "Flow", meta = [] }) {
  if (!flow) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <aside className="relative w-full max-w-sm h-full bg-surface border-l border-line overflow-y-auto animate-drawer-in">
        <div className="sticky top-0 bg-surface border-b border-line px-5 py-4 flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-faint truncate">{eyebrow}</p>
            <div className="flex items-center gap-2 mt-1">
              <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${flow.is_attack ? "bg-danger" : "bg-safe"}`} />
              <h3 className="text-lg font-semibold leading-none">{flow.prediction}</h3>
            </div>
            <p className="text-xs text-muted font-mono tabular-nums mt-1">
              {(flow.confidence * 100).toFixed(2)}% confidence
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="p-1 rounded-md text-muted hover:text-ink hover:bg-elevated shrink-0"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-5 space-y-6">
          {meta.length > 0 && (
            <dl className="space-y-2">
              {meta.map(({ label, value, tone }) => (
                <div key={label} className="flex items-baseline justify-between gap-4 text-xs">
                  <dt className="text-faint shrink-0">{label}</dt>
                  <dd
                    className={`font-mono tabular-nums text-right truncate ${
                      tone === "danger" ? "text-danger" : tone === "safe" ? "text-safe" : "text-ink"
                    }`}
                  >
                    {value}
                  </dd>
                </div>
              ))}
            </dl>
          )}

          <p className="text-xs text-muted leading-relaxed">{rationale(flow)}</p>

          {flow.probabilities && (
            <div>
              <h4 className="text-[10px] font-medium uppercase tracking-[0.1em] text-faint mb-3">
                Class probabilities
              </h4>
              <Distribution
                probabilities={flow.probabilities}
                prediction={flow.prediction}
                isAttack={flow.is_attack}
              />
            </div>
          )}

          <div>
            <h4 className="text-[10px] font-medium uppercase tracking-[0.1em] text-faint mb-3">
              Feature contributions
            </h4>
            <Contributions flow={flow} />
          </div>
        </div>
      </aside>
    </div>
  );
}
