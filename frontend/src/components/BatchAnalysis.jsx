import React, { useState } from "react";
import { FileSpreadsheet, Upload, Download, AlertTriangle, ShieldCheck, ShieldAlert, CheckCircle2, RefreshCw } from "lucide-react";
import { postBatchPredict } from "../api";

export default function BatchAnalysis({ demoSamples }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a CSV file first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await postBatchPredict(file);
      setResults(data);
    } catch (err) {
      setError(err.message || "Failed to process batch predictions.");
    } finally {
      setLoading(false);
    }
  };

  // Helper to download a ready-to-use sample CSV derived from the real demo samples
  const handleDownloadSampleCsv = () => {
    if (!demoSamples || demoSamples.length === 0) return;
    const headers = Object.keys(demoSamples[0].features);
    const rows = demoSamples.map((s) => headers.map((h) => s.features[h]).join(","));
    const csvContent = [headers.join(","), ...rows].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "sentinelops_sample_batch_flows.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white flex items-center gap-2.5">
            <FileSpreadsheet className="h-6 w-6 text-cyan-400" />
            <span>Batch Traffic Flow Evaluation</span>
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
            Upload multi-flow CSV datasets to execute batch threat classification with class distribution analytics.
          </p>
        </div>

        <button
          onClick={handleDownloadSampleCsv}
          className="flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-400 text-xs font-semibold border border-slate-700 transition-all w-fit"
        >
          <Download className="h-4 w-4" />
          <span>Download Sample CSV ({demoSamples.length} flows)</span>
        </button>
      </div>

      {/* Upload Box */}
      <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-6 shadow-lg space-y-4">
        <div className="border-2 border-dashed border-slate-700 hover:border-cyan-500/50 rounded-xl p-6 text-center cursor-pointer transition-all bg-slate-950/40">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="hidden"
            id="csv-file-input"
          />
          <label htmlFor="csv-file-input" className="cursor-pointer flex flex-col items-center justify-center space-y-2">
            <div className="p-3 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <Upload className="h-6 w-6" />
            </div>
            <div>
              <p className="text-sm font-semibold text-white">
                {file ? file.name : "Click to browse or drop CSV file here"}
              </p>
              <p className="text-xs text-slate-400 mt-0.5">
                File must contain the 40 required Top-40 feature columns
              </p>
            </div>
          </label>
        </div>

        <div className="flex items-center justify-end gap-3">
          {file && (
            <span className="text-xs text-slate-400 font-mono">
              Selected: <strong className="text-slate-200">{file.name}</strong> ({(file.size / 1024).toFixed(1)} KB)
            </span>
          )}
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/25 disabled:opacity-50 transition-all cursor-pointer"
          >
            {loading ? (
              <>
                <div className="h-4 w-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin"></div>
                <span>Evaluating Batch...</span>
              </>
            ) : (
              <>
                <Upload className="h-4 w-4" />
                <span>Run Batch Classification</span>
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="p-3 rounded-lg bg-rose-950/80 border border-rose-800 text-rose-300 text-xs flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Batch Results Summary */}
      {results && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5">
            {Object.entries(results.prediction_counts).map(([cls, count]) => {
              const isBenign = cls === "BENIGN";
              return (
                <div
                  key={cls}
                  className={`p-3 rounded-lg border text-center ${
                    count > 0
                      ? isBenign
                        ? "bg-emerald-950/40 border-emerald-800/80 text-emerald-400"
                        : "bg-rose-950/40 border-rose-800/80 text-rose-400"
                      : "bg-slate-900/60 border-slate-800 text-slate-500"
                  }`}
                >
                  <p className="text-[10px] font-bold uppercase truncate" title={cls}>{cls}</p>
                  <p className="text-lg font-black mt-1 font-mono">{count}</p>
                </div>
              );
            })}
          </div>

          {/* Predictions Table */}
          <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 shadow-lg space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white">Flow-by-Flow Classification Log</h3>
                <p className="text-xs text-slate-400 mt-0.5">Total flows evaluated: {results.total_samples}</p>
              </div>
            </div>

            <div className="overflow-x-auto max-h-96 overflow-y-auto">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-slate-800 text-slate-400 font-mono uppercase bg-slate-950/60 sticky top-0">
                  <tr>
                    <th className="py-2.5 px-3">Flow Index</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3">Predicted Class</th>
                    <th className="py-2.5 px-3">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {results.predictions.map((row) => (
                    <tr key={row.flow_id} className="hover:bg-slate-800/30 transition-colors">
                      <td className="py-2.5 px-3 text-slate-400 font-semibold">Flow #{row.flow_id}</td>
                      <td className="py-2.5 px-3">
                        {row.is_attack ? (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-rose-950 text-rose-400 border border-rose-800/60">
                            THREAT
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-800/60">
                            BENIGN
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-3 font-bold text-white">{row.prediction}</td>
                      <td className="py-2.5 px-3 text-cyan-400 font-bold">{(row.confidence * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
