import React, { useState, useEffect } from "react";
import Navbar from "./components/Navbar";
import Dashboard from "./components/Dashboard";
import LiveAnalysis from "./components/LiveAnalysis";
import BatchAnalysis from "./components/BatchAnalysis";
import ModelInfo from "./components/ModelInfo";
import { fetchHealth, fetchModelInfo, fetchDemoSamples } from "./api";
import { AlertCircle, RefreshCw } from "lucide-react";

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [apiStatus, setApiStatus] = useState("checking");
  const [modelInfo, setModelInfo] = useState(null);
  const [demoSamples, setDemoSamples] = useState([]);
  const [errorMessage, setErrorMessage] = useState(null);

  // Session Statistics
  const [stats, setStats] = useState({
    total: 0,
    attacks: 0,
    benign: 0,
  });

  // Recent Predictions
  const [recentPredictions, setRecentPredictions] = useState([]);

  const initData = async () => {
    setApiStatus("checking");
    setErrorMessage(null);
    try {
      const health = await fetchHealth();
      if (health.status === "healthy") {
        setApiStatus("connected");
      } else {
        setApiStatus("disconnected");
      }

      const info = await fetchModelInfo();
      setModelInfo(info);

      const samples = await fetchDemoSamples();
      setDemoSamples(samples);
    } catch (err) {
      setApiStatus("disconnected");
      setErrorMessage("Could not connect to FastAPI server at http://localhost:8000. Ensure the backend is running.");
    }
  };

  useEffect(() => {
    initData();
    const interval = setInterval(async () => {
      try {
        const health = await fetchHealth();
        if (health.status === "healthy") setApiStatus("connected");
      } catch (err) {
        setApiStatus("disconnected");
      }
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleNewPrediction = (pred) => {
    setRecentPredictions((prev) => [pred, ...prev.slice(0, 9)]);
    setStats((prev) => ({
      total: prev.total + 1,
      attacks: pred.is_attack ? prev.attacks + 1 : prev.attacks,
      benign: !pred.is_attack ? prev.benign + 1 : prev.benign,
    }));
  };

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        apiStatus={apiStatus}
      />

      {errorMessage && apiStatus === "disconnected" && (
        <div className="bg-rose-950/90 border-b border-rose-800 text-rose-200 px-4 py-2.5 text-xs flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-4 w-4 text-rose-400 shrink-0" />
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={initData}
            className="flex items-center space-x-1 underline font-semibold hover:text-white"
          >
            <RefreshCw className="h-3 w-3" />
            <span>Retry Connection</span>
          </button>
        </div>
      )}

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === "dashboard" && (
          <Dashboard
            modelInfo={modelInfo}
            stats={stats}
            recentPredictions={recentPredictions}
            setActiveTab={setActiveTab}
            onRefreshHealth={initData}
            apiStatus={apiStatus}
          />
        )}

        {activeTab === "live" && (
          <LiveAnalysis
            demoSamples={demoSamples}
            onNewPrediction={handleNewPrediction}
          />
        )}

        {activeTab === "batch" && (
          <BatchAnalysis
            demoSamples={demoSamples}
          />
        )}

        {activeTab === "model" && (
          <ModelInfo
            modelInfo={modelInfo}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/60 py-4 text-center text-xs text-slate-500 font-mono">
        <p>SentinelOps MLOps Security Platform • Model: XGBoost Champion (Top-40) • Leakage-Free Pipeline</p>
      </footer>
    </div>
  );
}
