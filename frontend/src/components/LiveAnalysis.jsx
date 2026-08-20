import React, { useState } from "react";
import { Shield, ShieldAlert, ShieldCheck, Zap, AlertTriangle, ArrowRight, CheckCircle2, TrendingUp, TrendingDown, Info, Play } from "lucide-react";
import { postPredict } from "../api";

export default function LiveAnalysis({ demoSamples, onNewPrediction }) {
  const [selectedSampleId, setSelectedSampleId] = useState(demoSamples[0]?.id || "");
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const selectedSample = demoSamples.find((s) => s.id === selectedSampleId) || demoSamples[0];

  const handleAnalyze = async () => {
    if (!selectedSample) return;
    setAnalyzing(true);
    setError(null);
    try {
      const pred = await postPredict(selectedSample.features);
      setResult(pred);
      if (onNewPrediction) {
        onNewPrediction({
          ...pred,
          sampleId: selectedSample.id,
          sampleLabel: selectedSample.label,
          timestamp: new Date().toLocaleTimeString(),
        });
      }
    } catch (err) {
      setError(err.message || "Failed to analyze traffic flow.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white flex items-center gap-2.5">
            <Shield className="h-6 w-6 text-cyan-400" />
            <span>Live Flow Analysis & SHAP Explainability</span>
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
            Select authentic CIC-IDS2017 network traffic and run real-time inference with TreeExplainer attribution.
          </p>
        </div>
      </div>

      {/* Traffic Selection & Action Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Sample Selector & Feature Snapshot */}
        <div className="lg:col-span-5 space-y-4">
          <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 shadow-lg space-y-4">
            <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider">
              1. Select Demo Traffic Sample
            </label>

            {/* Dropdown */}
            <select
              value={selectedSampleId}
              onChange={(e) => {
                setSelectedSampleId(e.target.value);
                setResult(null);
                setError(null);
              }}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white font-medium focus:ring-2 focus:ring-cyan-500 focus:border-transparent outline-none"
            >
              {demoSamples.map((sample) => (
                <option key={sample.id} value={sample.id}>
                  [{sample.label}] {sample.id} — {sample.description.substring(0, 45)}...
                </option>
              ))}
            </select>

            {/* Selected Sample Meta */}
            {selectedSample && (
              <div className="p-3.5 rounded-lg bg-slate-950/80 border border-slate-800/80 space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">Dataset Ground Truth:</span>
                  <span className={`font-bold px-2 py-0.5 rounded text-[11px] ${
                    selectedSample.label === "BENIGN"
                      ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                      : "bg-rose-950 text-rose-400 border border-rose-800"
                  }`}>
                    {selectedSample.label}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block mb-0.5">Description:</span>
                  <span className="text-slate-300 italic">{selectedSample.description}</span>
                </div>
                <div className="flex items-center justify-between pt-1 border-t border-slate-800 text-[11px]">
                  <span className="text-slate-500">Feature Dimensions:</span>
                  <span className="text-cyan-400 font-mono font-bold">40 Top Features</span>
                </div>
              </div>
            )}

            {/* Action Button */}
            <button
              onClick={handleAnalyze}
              disabled={analyzing || !selectedSample}
              className="w-full flex items-center justify-center space-x-2 py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/20 disabled:opacity-50 transition-all cursor-pointer"
            >
              {analyzing ? (
                <>
                  <div className="h-4 w-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin"></div>
                  <span>Evaluating Flow & Computing SHAP...</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-current" />
                  <span>ANALYZE TRAFFIC FLOW</span>
                </>
              )}
            </button>

            {error && (
              <div className="p-3 rounded-lg bg-rose-950/80 border border-rose-800 text-rose-300 text-xs flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Sample Features Preview */}
          {selectedSample && (
            <div className="rounded-xl bg-slate-900/60 border border-slate-800 p-4 space-y-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
                Sample Top Features Snapshot
              </span>
              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono max-h-48 overflow-y-auto pr-1">
                {Object.entries(selectedSample.features).slice(0, 10).map(([k, v]) => (
                  <div key={k} className="p-2 rounded bg-slate-950/60 border border-slate-800/50">
                    <span className="text-slate-400 block truncate" title={k}>{k}</span>
                    <span className="text-cyan-400 font-bold">{typeof v === 'number' ? v.toFixed(3) : v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Prediction Verdict & SHAP Explanation */}
        <div className="lg:col-span-7 space-y-4">
          {!result && !analyzing && (
            <div className="rounded-xl bg-slate-900/40 border border-dashed border-slate-800 p-12 text-center flex flex-col items-center justify-center min-h-[380px]">
              <div className="p-4 rounded-full bg-slate-900 text-slate-600 mb-3 border border-slate-800">
                <Shield className="h-10 w-10" />
              </div>
              <h3 className="text-base font-bold text-slate-300">Ready for Live Analysis</h3>
              <p className="text-xs text-slate-500 max-w-sm mt-1">
                Select a network flow from the left and click "ANALYZE TRAFFIC FLOW" to run the Top-40 XGBoost Champion and inspect its SHAP explanation.
              </p>
            </div>
          )}

          {analyzing && (
            <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-12 text-center flex flex-col items-center justify-center min-h-[380px] space-y-4">
              <div className="relative">
                <div className="h-14 w-14 rounded-full border-4 border-cyan-500/20 border-t-cyan-400 animate-spin"></div>
                <Shield className="h-6 w-6 text-cyan-400 absolute inset-0 m-auto" />
              </div>
              <div>
                <h4 className="text-sm font-bold text-white">Evaluating 40 Feature Dimensions</h4>
                <p className="text-xs text-slate-400 mt-1">Generating multi-class probability scores & TreeExplainer attribution...</p>
              </div>
            </div>
          )}

          {result && !analyzing && (
            <div className="space-y-4">
              {/* Verdict Banner Card */}
              <div className={`rounded-xl border p-5 shadow-xl transition-all ${
                result.is_attack
                  ? "bg-gradient-to-r from-rose-950/60 via-slate-900 to-slate-900 border-rose-500/50 shadow-rose-950/30"
                  : "bg-gradient-to-r from-emerald-950/60 via-slate-900 to-slate-900 border-emerald-500/50 shadow-emerald-950/30"
              }`}>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div className="flex items-center space-x-3.5">
                    <div className={`p-3 rounded-xl border ${
                      result.is_attack
                        ? "bg-rose-500/20 text-rose-400 border-rose-500/30 shadow-lg shadow-rose-500/20"
                        : "bg-emerald-500/20 text-emerald-400 border-emerald-500/30 shadow-lg shadow-emerald-500/20"
                    }`}>
                      {result.is_attack ? <ShieldAlert className="h-8 w-8" /> : <ShieldCheck className="h-8 w-8" />}
                    </div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className={`text-xs font-black tracking-wider uppercase px-2 py-0.5 rounded ${
                          result.is_attack ? "bg-rose-900/80 text-rose-300 border border-rose-700" : "bg-emerald-900/80 text-emerald-300 border border-emerald-700"
                        }`}>
                          {result.is_attack ? "THREAT DETECTED" : "BENIGN TRAFFIC"}
                        </span>
                        <span className="text-xs text-slate-400 font-mono">
                          {(result.confidence * 100).toFixed(2)}% Confidence
                        </span>
                      </div>
                      <h2 className="text-2xl font-black text-white mt-1 tracking-tight">
                        {result.prediction}
                      </h2>
                    </div>
                  </div>

                  <div className="sm:text-right border-t sm:border-t-0 pt-2 sm:pt-0 border-slate-800 text-xs text-slate-400 font-mono">
                    <p className="text-slate-300 font-semibold">{result.model}</p>
                    <p className="text-[11px] text-slate-500 mt-0.5">{result.feature_count} features evaluated</p>
                  </div>
                </div>
              </div>

              {/* Class Probability Distribution */}
              <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-4 space-y-3 shadow-lg">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                    Multiclass Probability Distribution
                  </span>
                  <span className="text-[11px] text-slate-500 font-mono">8 Target Classes</span>
                </div>

                <div className="space-y-2 text-xs font-mono">
                  {Object.entries(result.probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, prob]) => {
                      const pct = (prob * 100).toFixed(2);
                      const isSelected = cls === result.prediction;
                      return (
                        <div key={cls} className="space-y-1">
                          <div className="flex justify-between text-[11px]">
                            <span className={isSelected ? "font-bold text-cyan-300" : "text-slate-400"}>
                              {cls} {isSelected && "★"}
                            </span>
                            <span className={isSelected ? "font-bold text-cyan-300" : "text-slate-500"}>
                              {pct}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                isSelected
                                  ? result.is_attack ? "bg-rose-500" : "bg-emerald-400"
                                  : "bg-slate-700"
                              }`}
                              style={{ width: `${Math.max(prob * 100, 1)}%` }}
                            ></div>
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>

              {/* SHAP Explanation Section: "WHY WAS THIS PREDICTED?" */}
              <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 space-y-4 shadow-lg">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Info className="h-4 w-4 text-cyan-400" />
                      <span>Why was this predicted? (SHAP Attribution)</span>
                    </h3>
                    <p className="text-[11px] text-slate-400 mt-0.5">
                      Top feature contributions from the XGBoost TreeExplainer for <span className="text-cyan-300 font-bold">{result.prediction}</span>
                    </p>
                  </div>
                </div>

                <div className="space-y-2.5">
                  {result.explanation.map((item, idx) => {
                    const isPositive = item.shap_value > 0;
                    return (
                      <div
                        key={idx}
                        className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-2 hover:border-slate-700 transition-all text-xs"
                      >
                        <div className="space-y-0.5 min-w-[200px]">
                          <div className="flex items-center space-x-1.5">
                            <span className="font-semibold text-white">{item.feature}</span>
                          </div>
                          <div className="text-[11px] text-slate-400 font-mono">
                            Actual Value: <span className="text-cyan-400 font-bold">{item.value}</span>
                          </div>
                        </div>

                        <div className="flex items-center space-x-3 sm:justify-end">
                          <div className="text-right font-mono">
                            <span className={`font-bold ${isPositive ? "text-rose-400" : "text-emerald-400"}`}>
                              {isPositive ? `+${item.shap_value.toFixed(4)}` : item.shap_value.toFixed(4)}
                            </span>
                          </div>

                          <div className={`px-2 py-0.5 rounded text-[10px] font-bold flex items-center gap-1 ${
                            isPositive
                              ? "bg-rose-950/80 text-rose-300 border border-rose-800/80"
                              : "bg-emerald-950/80 text-emerald-300 border border-emerald-800/80"
                          }`}>
                            {isPositive ? (
                              <>
                                <TrendingUp className="h-3 w-3" />
                                <span>Increases Threat Risk</span>
                              </>
                            ) : (
                              <>
                                <TrendingDown className="h-3 w-3" />
                                <span>Decreases Threat Risk</span>
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
