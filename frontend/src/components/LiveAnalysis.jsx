import React, { useEffect, useState } from "react";
import { Activity, Check, Search, X as XIcon } from "lucide-react";
import { postPredict } from "../api";
import { useTheme } from "../theme-context";
import { EmptyState, MetricCard, Panel, SectionHeader, StatusBadge } from "./ui";

export default function LiveAnalysis({ demoSamples }) {
  const { chart } = useTheme();
  const [selectedId, setSelectedId] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // demoSamples arrive asynchronously; seed the selection once they land.
  useEffect(() => {
    if (!selectedId && demoSamples.length) setSelectedId(demoSamples[0].id);
  }, [demoSamples, selectedId]);

  const sample = demoSamples.find((s) => s.id === selectedId) || demoSamples[0];

  const analyze = async () => {
    if (!sample) return;
    setAnalyzing(true);
    setError(null);
    try {
      setResult(await postPredict(sample.features));
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const probabilities = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  const raisingFeatures = result?.explanation?.filter((f) => f.increases_threat) || [];
  const loweringFeatures = result?.explanation?.filter((f) => !f.increases_threat) || [];

  const sampleIndex = demoSamples.findIndex((s) => s.id === selectedId);
  const matchesGroundTruth = result && sample ? result.prediction === sample.label : null;

  return (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Explain"
        title="Single-Flow Scorer"
        description="Score one flow in real time against the same /predict endpoint a SIEM or EDR integration would call, then see why."
      />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Selector + verdict */}
        <div className="lg:col-span-4 space-y-5">
          <Panel title="Traffic sample" description="Reference flows held out from training — the label is withheld until you analyze.">
            <div className="space-y-4">
              <select
                id="sample"
                value={selectedId}
                onChange={(e) => {
                  setSelectedId(e.target.value);
                  setResult(null);
                  setError(null);
                }}
                className="w-full bg-canvas border border-line rounded-md px-3 py-2 text-sm text-ink focus:border-accent outline-none"
              >
                {demoSamples.map((s, i) => (
                  <option key={s.id} value={s.id}>
                    Traffic sample {i + 1}
                  </option>
                ))}
              </select>

              <button
                onClick={analyze}
                disabled={analyzing || !sample}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-accent text-white text-sm font-medium disabled:opacity-40 hover:opacity-90 transition-opacity"
              >
                <Search className="h-3.5 w-3.5" />
                {analyzing ? "Analyzing…" : "Analyze flow"}
              </button>

              {error && <p className="text-xs text-danger">{error}</p>}
            </div>
          </Panel>

          {sample && (
            <Panel title="Input features">
              <dl className="space-y-1.5 text-xs max-h-64 overflow-y-auto pr-1">
                {Object.entries(sample.features).slice(0, 12).map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-3">
                    <dt className="text-muted truncate">{k}</dt>
                    <dd className="font-mono tabular-nums text-ink shrink-0">
                      {typeof v === "number" ? v.toLocaleString() : v}
                    </dd>
                  </div>
                ))}
              </dl>
              <p className="text-[11px] text-faint mt-3 pt-3 border-t border-line">
                Showing 12 of {Object.keys(sample.features).length}
              </p>
            </Panel>
          )}

          {result && sample && (
            <Panel title="Verdict">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <StatusBadge tone={result.is_attack ? "danger" : "safe"}>
                    {result.prediction}
                  </StatusBadge>
                  <p className="text-[11px] text-faint mt-1.5">
                    {result.is_attack ? "Malicious traffic" : "Benign traffic"}
                  </p>
                </div>
                <MetricCard label="Confidence" value={`${(result.confidence * 100).toFixed(1)}%`} dense />
              </div>

              <div className="mt-4 pt-4 border-t border-line space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-faint">Known label (sample {sampleIndex + 1})</span>
                  <span className={sample.label === "BENIGN" ? "text-safe" : "text-danger"}>{sample.label}</span>
                </div>
                <div className={`flex items-center gap-1.5 text-xs ${matchesGroundTruth ? "text-safe" : "text-danger"}`}>
                  {matchesGroundTruth ? <Check className="h-3.5 w-3.5" /> : <XIcon className="h-3.5 w-3.5" />}
                  {matchesGroundTruth ? "Matches known label" : "Does not match known label"}
                </div>
                <p className="text-[11px] text-muted leading-relaxed pt-1">{sample.description}</p>
              </div>
            </Panel>
          )}
        </div>

        {/* Explanation */}
        <div className="lg:col-span-8">
          {!result ? (
            <div className="card h-full min-h-[420px]">
              <EmptyState
                icon={Activity}
                title="No analysis yet"
                description="Select a flow on the left and run the analysis to see the verdict and its explanation."
              />
            </div>
          ) : (
            <div className="space-y-5">
              <Panel title="Class probabilities">
                <div className="space-y-2.5">
                  {probabilities.map(([cls, prob]) => {
                    const isTop = cls === result.prediction;
                    return (
                      <div key={cls}>
                        <div className="flex justify-between text-xs mb-1">
                          <span className={isTop ? "text-ink font-medium" : "text-muted"}>{cls}</span>
                          <span className={`font-mono tabular-nums ${isTop ? "text-ink" : "text-faint"}`}>
                            {(prob * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="h-1 rounded-full bg-elevated overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{
                              width: `${Math.max(prob * 100, 0.5)}%`,
                              backgroundColor: isTop
                                ? result.is_attack ? chart.danger : chart.safe
                                : chart.grid,
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Panel>

              <Panel
                title="Why this verdict"
                description={`Feature contributions toward the predicted class, ${result.prediction}`}
              >
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-5">
                  <div className="rounded-md border border-danger/25 bg-danger-soft px-3.5 py-3">
                    <p className="text-[11px] font-medium text-danger">Pushed toward threat</p>
                    <p className="text-[11px] text-muted mt-1 leading-relaxed">
                      {raisingFeatures.length
                        ? `${raisingFeatures.length} feature(s) increased the model's estimated threat probability.`
                        : "No feature pushed this prediction toward a higher threat."}
                    </p>
                  </div>
                  <div className="rounded-md border border-safe/25 bg-safe-soft px-3.5 py-3">
                    <p className="text-[11px] font-medium text-safe">Pushed toward benign</p>
                    <p className="text-[11px] text-muted mt-1 leading-relaxed">
                      {loweringFeatures.length
                        ? `${loweringFeatures.length} feature(s) pushed the prediction toward benign.`
                        : "No feature pushed this prediction toward benign."}
                    </p>
                  </div>
                </div>

                <div className="space-y-3">
                  {result.explanation.map((item) => {
                    const raises = item.increases_threat ?? item.shap_value > 0;
                    const max = Math.max(
                      ...result.explanation.map((e) => Math.abs(e.shap_value)),
                      1e-4
                    );
                    const width = (Math.abs(item.shap_value) / max) * 100;
                    return (
                      <div key={item.feature} className="grid grid-cols-12 gap-3 items-center">
                        <span className="col-span-4 text-xs text-ink truncate">{item.feature}</span>
                        <span className="col-span-2 text-xs font-mono tabular-nums text-faint text-right">
                          {item.value}
                        </span>
                        <div className="col-span-4 h-1.5 rounded-full bg-elevated overflow-hidden">
                          <div
                            className={`h-full rounded-full ${raises ? "bg-danger" : "bg-safe"}`}
                            style={{ width: `${width}%` }}
                          />
                        </div>
                        <span className={`col-span-2 text-xs font-mono tabular-nums text-right ${raises ? "text-danger" : "text-safe"}`}>
                          {item.shap_value > 0 ? "+" : ""}{item.shap_value.toFixed(3)}
                        </span>
                      </div>
                    );
                  })}
                </div>

                <p className="text-[11px] text-faint mt-4 pt-4 border-t border-line leading-relaxed">
                  Red raises threat, green lowers it. Direction accounts for the predicted
                  class, so on a benign verdict a positive SHAP value lowers threat.
                </p>
              </Panel>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
