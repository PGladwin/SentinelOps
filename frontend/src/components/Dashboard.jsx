import React from "react";
import { Shield, ShieldAlert, ShieldCheck, Cpu, Database, Activity, ArrowRight, Zap, RefreshCw } from "lucide-react";

export default function Dashboard({ modelInfo, stats, recentPredictions, setActiveTab, onRefreshHealth, apiStatus }) {
  const attackRate = stats.total > 0 ? ((stats.attacks / stats.total) * 100).toFixed(1) : "0.0";

  return (
    <div className="space-y-6">
      {/* Top Welcome Banner */}
      <div className="relative overflow-hidden rounded-xl bg-gradient-to-r from-slate-900 via-slate-900/90 to-cyan-950/40 border border-slate-800 p-6 shadow-xl">
        <div className="absolute right-0 top-0 bottom-0 w-1/3 bg-radial from-cyan-500/10 to-transparent pointer-events-none"></div>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 relative z-10">
          <div>
            <div className="inline-flex items-center space-x-2 px-2.5 py-1 rounded-full bg-cyan-950/80 border border-cyan-800/60 text-cyan-400 text-xs font-semibold mb-2">
              <Zap className="h-3 w-3" />
              <span>PHASE 2 MODEL VERIFICATION ACTIVE</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              SentinelOps Network Security Center
            </h1>
            <p className="text-slate-400 text-sm mt-1 max-w-2xl">
              Real-time cyber threat detection engine powered by the XGBoost Champion trained on genuine CIC-IDS2017 flow data with SHAP TreeExplainer explainability.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setActiveTab("live")}
              className="flex items-center space-x-2 px-4 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/25 transition-all"
            >
              <Activity className="h-4 w-4" />
              <span>Launch Live Analysis</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Primary Status Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Model Status Card */}
        <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 shadow-lg relative overflow-hidden group hover:border-cyan-500/40 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Model Status</span>
            <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <Cpu className="h-5 w-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-lg font-bold text-white flex items-center gap-2">
              <span>XGBoost Champion</span>
              <span className="h-2 w-2 rounded-full bg-emerald-400"></span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Top-40 Multiclass Tree Model
            </p>
          </div>
        </div>

        {/* Features Representation Card */}
        <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 shadow-lg relative overflow-hidden group hover:border-cyan-500/40 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Features</span>
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <Database className="h-5 w-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-white tracking-tight">40</div>
            <p className="text-xs text-slate-400 mt-1">
              Leakage-free training-ranked subset
            </p>
          </div>
        </div>

        {/* API Backend Card */}
        <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 shadow-lg relative overflow-hidden group hover:border-cyan-500/40 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">API Status</span>
            <div className={`p-2 rounded-lg ${apiStatus === "connected" ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-rose-500/10 text-rose-400 border border-rose-500/20"}`}>
              <Activity className="h-5 w-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-lg font-bold text-white flex items-center gap-2">
              <span className={apiStatus === "connected" ? "text-emerald-400" : "text-rose-400"}>
                {apiStatus === "connected" ? "Connected" : "Offline"}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              FastAPI backend on port 8000
            </p>
          </div>
        </div>

        {/* Attack Rate / Session Activity */}
        <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 shadow-lg relative overflow-hidden group hover:border-cyan-500/40 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">Attack Ratio</span>
            <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20">
              <ShieldAlert className="h-5 w-5" />
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold text-white tracking-tight">{attackRate}%</div>
            <p className="text-xs text-slate-400 mt-1">
              {stats.attacks} threats / {stats.total} total flows
            </p>
          </div>
        </div>
      </div>

      {/* Session Traffic Statistics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-xl bg-slate-900/60 border border-slate-800/80 p-4 flex items-center space-x-4">
          <div className="p-3 rounded-lg bg-cyan-950 border border-cyan-800 text-cyan-400">
            <Activity className="h-6 w-6" />
          </div>
          <div>
            <p className="text-xs font-medium text-slate-400 uppercase">Total Analyzed</p>
            <p className="text-xl font-bold text-white">{stats.total} flows</p>
          </div>
        </div>

        <div className="rounded-xl bg-slate-900/60 border border-slate-800/80 p-4 flex items-center space-x-4">
          <div className="p-3 rounded-lg bg-rose-950 border border-rose-800 text-rose-400">
            <ShieldAlert className="h-6 w-6" />
          </div>
          <div>
            <p className="text-xs font-medium text-slate-400 uppercase">Threats Detected</p>
            <p className="text-xl font-bold text-rose-400">{stats.attacks} attacks</p>
          </div>
        </div>

        <div className="rounded-xl bg-slate-900/60 border border-slate-800/80 p-4 flex items-center space-x-4">
          <div className="p-3 rounded-lg bg-emerald-950 border border-emerald-800 text-emerald-400">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <div>
            <p className="text-xs font-medium text-slate-400 uppercase">Benign Traffic</p>
            <p className="text-xl font-bold text-emerald-400">{stats.benign} safe</p>
          </div>
        </div>
      </div>

      {/* Recent Predictions Feed Table */}
      <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Activity className="h-4 w-4 text-cyan-400" />
              <span>Recent Inference Log</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">Live session predictions generated by the local API service</p>
          </div>
          <button
            onClick={() => setActiveTab("live")}
            className="text-xs text-cyan-400 hover:text-cyan-300 font-semibold flex items-center gap-1 transition-colors"
          >
            <span>Analyze new flow</span>
            <ArrowRight className="h-3 w-3" />
          </button>
        </div>

        {recentPredictions.length === 0 ? (
          <div className="text-center py-10 border border-dashed border-slate-800 rounded-lg">
            <Shield className="h-8 w-8 text-slate-600 mx-auto mb-2" />
            <p className="text-sm text-slate-400 font-medium">No live predictions in current session</p>
            <p className="text-xs text-slate-500 mt-1">Navigate to Live Analysis or Batch Analysis to evaluate network traffic.</p>
            <button
              onClick={() => setActiveTab("live")}
              className="mt-4 px-3.5 py-1.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 text-xs font-semibold transition-all"
            >
              Test Demo Traffic
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-800 text-slate-400 font-mono uppercase bg-slate-950/40">
                <tr>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Flow ID / Source</th>
                  <th className="py-2.5 px-3">Verdict</th>
                  <th className="py-2.5 px-3">Predicted Class</th>
                  <th className="py-2.5 px-3">Confidence</th>
                  <th className="py-2.5 px-3">Top SHAP Feature</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {recentPredictions.map((pred, i) => (
                  <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                    <td className="py-2.5 px-3 text-slate-400">{pred.timestamp}</td>
                    <td className="py-2.5 px-3 text-slate-300 font-semibold">{pred.sampleId || `flow_${i + 1}`}</td>
                    <td className="py-2.5 px-3">
                      {pred.is_attack ? (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-rose-950 text-rose-400 border border-rose-800/60">
                          THREAT
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-800/60">
                          BENIGN
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-white font-bold">{pred.prediction}</td>
                    <td className="py-2.5 px-3 text-cyan-400 font-bold">{(pred.confidence * 100).toFixed(2)}%</td>
                    <td className="py-2.5 px-3 text-slate-400">
                      {pred.explanation && pred.explanation[0] ? pred.explanation[0].feature : "N/A"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
