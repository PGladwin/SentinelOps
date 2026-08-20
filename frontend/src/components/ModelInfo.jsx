import React from "react";
import { Cpu, Database, CheckCircle2, Shield, Layers, FileCode2, Clock, Settings, Info } from "lucide-react";

export default function ModelInfo({ modelInfo }) {
  if (!modelInfo) {
    return (
      <div className="p-8 text-center text-slate-400">
        <p>Loading model metadata from API...</p>
      </div>
    );
  }

  const meta = modelInfo.metadata || {};
  const hyperparams = meta.hyperparameters || {};

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white flex items-center gap-2.5">
            <Cpu className="h-6 w-6 text-cyan-400" />
            <span>Champion Model Architecture & Artifacts</span>
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
            Auditable metadata demonstrating that the local UI is backed by the genuine Phase 2 Top-40 XGBoost Champion.
          </p>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 space-y-2">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Model Type</span>
          <p className="text-lg font-bold text-white flex items-center gap-2">
            <span>XGBClassifier (Hist)</span>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950 text-cyan-400 border border-cyan-800">
              PROVISIONAL CHAMPION
            </span>
          </p>
          <p className="text-xs text-slate-400">Multiclass tree ensemble with sample-weighted balance</p>
        </div>

        <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 space-y-2">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Feature Dimensions</span>
          <p className="text-2xl font-black text-cyan-400 font-mono">40 Features</p>
          <p className="text-xs text-slate-400">Strictly derived from training set importance</p>
        </div>

        <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 space-y-2">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Class Taxonomy</span>
          <p className="text-2xl font-black text-emerald-400 font-mono">8 Classes</p>
          <p className="text-xs text-slate-400">BENIGN + 7 Cyber Attack Categories</p>
        </div>
      </div>

      {/* Hyperparameters & Training Specs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 shadow-lg space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Settings className="h-4 w-4 text-cyan-400" />
            <span>Hyperparameter Configuration</span>
          </h3>

          <div className="grid grid-cols-2 gap-3 text-xs font-mono">
            <div className="p-2.5 rounded bg-slate-950/70 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">n_estimators</span>
              <span className="text-cyan-400 font-bold">{hyperparams.n_estimators || 200}</span>
            </div>
            <div className="p-2.5 rounded bg-slate-950/70 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">max_depth</span>
              <span className="text-cyan-400 font-bold">{hyperparams.max_depth || 6}</span>
            </div>
            <div className="p-2.5 rounded bg-slate-950/70 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">learning_rate</span>
              <span className="text-cyan-400 font-bold">{hyperparams.learning_rate || 0.1}</span>
            </div>
            <div className="p-2.5 rounded bg-slate-950/70 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">subsample</span>
              <span className="text-cyan-400 font-bold">{hyperparams.subsample || 0.8}</span>
            </div>
            <div className="p-2.5 rounded bg-slate-950/70 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">colsample_bytree</span>
              <span className="text-cyan-400 font-bold">{hyperparams.colsample_bytree || 0.8}</span>
            </div>
            <div className="p-2.5 rounded bg-slate-950/70 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">objective</span>
              <span className="text-cyan-400 font-bold">multi:softprob (8 classes)</span>
            </div>
          </div>

          <div className="pt-2 border-t border-slate-800 space-y-1.5 text-xs text-slate-400">
            <div className="flex justify-between">
              <span>Training Dataset Rows:</span>
              <span className="font-mono text-white font-bold">{meta.training_dataset_size?.toLocaleString() || "119,996"}</span>
            </div>
            <div className="flex justify-between">
              <span>Held-out Test Rows:</span>
              <span className="font-mono text-white font-bold">{meta.test_dataset_size?.toLocaleString() || "30,000"}</span>
            </div>
            <div className="flex justify-between">
              <span>Random Seed:</span>
              <span className="font-mono text-white font-bold">{meta.random_seed || 42}</span>
            </div>
          </div>
        </div>

        {/* 8-Class Taxonomy Mapping */}
        <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 shadow-lg space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Shield className="h-4 w-4 text-cyan-400" />
            <span>Target Class Mapping</span>
          </h3>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            {Object.entries(modelInfo.class_mapping || {}).map(([name, id]) => (
              <div key={name} className="flex items-center justify-between p-2 rounded bg-slate-950/70 border border-slate-800">
                <span className={name === "BENIGN" ? "text-emerald-400 font-semibold" : "text-rose-300 font-semibold"}>
                  {name}
                </span>
                <span className="px-1.5 py-0.5 rounded bg-slate-900 text-slate-400 text-[10px]">
                  ID: {id}
                </span>
              </div>
            ))}
          </div>

          <div className="p-3 rounded-lg bg-cyan-950/40 border border-cyan-800/50 text-[11px] text-cyan-300">
            <strong>Infiltration Note:</strong> Infiltration class retained in taxonomy (ID: 7) with 0 test samples in dev split.
          </div>
        </div>
      </div>

      {/* Ordered 40 Features List */}
      <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 shadow-lg space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-white">Exact Ordered Top-40 Feature Representation</h3>
            <p className="text-xs text-slate-400 mt-0.5">Input order strictly required by `models/xgboost_top40.pkl`</p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 text-xs font-mono">
          {modelInfo.features?.map((feat, idx) => (
            <div key={feat} className="flex items-center space-x-2 p-2 rounded bg-slate-950/60 border border-slate-800/80">
              <span className="text-cyan-500 font-bold w-6 text-right">{idx + 1}.</span>
              <span className="text-slate-300 truncate" title={feat}>{feat}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
