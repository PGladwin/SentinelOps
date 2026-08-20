import React from "react";
import { Shield, Activity, Database, FileSpreadsheet, Server, Cpu } from "lucide-react";

export default function Navbar({ activeTab, setActiveTab, apiStatus }) {
  const tabs = [
    { id: "dashboard", label: "Dashboard", icon: Activity },
    { id: "live", label: "Live Analysis", icon: Shield },
    { id: "batch", label: "Batch Analysis", icon: FileSpreadsheet },
    { id: "model", label: "Model Info", icon: Cpu },
  ];

  return (
    <header className="border-b border-slate-800 bg-[#0c121e]/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab("dashboard")}>
            <div className="h-10 w-10 rounded-lg bg-gradient-to-tr from-cyan-600 to-blue-500 flex items-center justify-center shadow-lg shadow-cyan-500/20 ring-1 ring-cyan-400/30">
              <Shield className="h-6 w-6 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg tracking-wider text-white">SENTINEL<span className="text-cyan-400">OPS</span></span>
                <span className="text-[10px] font-semibold tracking-wider bg-cyan-950 text-cyan-400 border border-cyan-800/60 px-1.5 py-0.5 rounded">CHAMPION</span>
              </div>
              <p className="text-[11px] text-slate-400 -mt-0.5">AI-Powered Network Intrusion Detection</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex space-x-1 sm:space-x-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-sm"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent"
                  }`}
                >
                  <Icon className={`h-4 w-4 ${isActive ? "text-cyan-400" : "text-slate-400"}`} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          {/* API Health Indicator */}
          <div className="flex items-center space-x-2.5">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-900/90 border border-slate-800 text-xs">
              <span className={`h-2 w-2 rounded-full ${apiStatus === "connected" ? "bg-emerald-400 animate-pulse" : "bg-rose-500"}`}></span>
              <span className="text-slate-300 font-mono">
                API: {apiStatus === "connected" ? "ONLINE (8000)" : "OFFLINE"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
