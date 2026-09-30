import React, { useEffect, useState } from "react";
import Navbar from "./components/Navbar";
import SocDashboard from "./components/SocDashboard";
import LiveFeed from "./components/LiveFeed";
import MlopsPanel from "./components/MlopsPanel";
import { fetchHealth, fetchModelInfo } from "./api";

// Three views, each answering a different question: what is happening now,
// what is in this file, and how did this model get here.
//
// Live leads because it is the view that shows the system doing its job. A
// fourth "Explain" tab scored one pre-baked sample at a time; both Live and
// Detect now open a full SHAP investigation on any flow, so it was a longer
// route to something already one click away.
const TABS = ["live", "detect", "mlops"];

export default function App() {
  const [activeTab, setActiveTab] = useState("live");
  const [apiStatus, setApiStatus] = useState("checking");
  const [modelInfo, setModelInfo] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const initData = async () => {
    setApiStatus("checking");
    setErrorMessage(null);
    try {
      const health = await fetchHealth();
      setApiStatus(health.status === "healthy" ? "connected" : "disconnected");
      setModelInfo(await fetchModelInfo());
    } catch (err) {
      setApiStatus("disconnected");
      setErrorMessage(err.message);
    }
  };

  useEffect(() => {
    initData();
    const interval = setInterval(async () => {
      try {
        const health = await fetchHealth();
        setApiStatus(health.status === "healthy" ? "connected" : "disconnected");
      } catch {
        setApiStatus("disconnected");
      }
    }, 20000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-canvas text-ink flex flex-col">
      <Navbar
        tabs={TABS}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        apiStatus={apiStatus}
        modelInfo={modelInfo}
      />

      {apiStatus === "disconnected" && (
        <div className="border-b border-line bg-danger-soft px-5 py-2.5 text-xs text-danger flex flex-wrap items-center justify-between gap-2">
          <span>{errorMessage || "Backend unreachable."}</span>
          <button onClick={initData} className="underline font-medium hover:no-underline">
            Retry
          </button>
        </div>
      )}

      <main key={activeTab} className="flex-1 w-full max-w-[1400px] mx-auto px-5 sm:px-8 py-8 animate-tab-in">
        {activeTab === "live" && <LiveFeed />}
        {activeTab === "detect" && <SocDashboard />}
        {activeTab === "mlops" && <MlopsPanel modelInfo={modelInfo} />}
      </main>

      <footer className="border-t border-line py-4 px-5 sm:px-8">
        <div className="max-w-[1400px] mx-auto flex flex-wrap items-center justify-between gap-2 text-xs text-faint">
          <span>SentinelOps — network intrusion detection</span>
          {modelInfo && (
            <span className="font-mono">
              {modelInfo.model_name} · {modelInfo.feature_count} features · 8 classes
            </span>
          )}
        </div>
      </footer>
    </div>
  );
}
