import React from "react";
import { Moon, ShieldCheck, Sun } from "lucide-react";
import { useTheme } from "../theme-context";
import { StatusIndicator } from "./ui";

const LABELS = {
  live: "Live",
  detect: "Detect",
  mlops: "MLOps",
};

function apiTone(status) {
  if (status === "connected") return "safe";
  if (status === "checking") return "warn";
  return "danger";
}

function apiLabel(status) {
  if (status === "connected") return "API online";
  if (status === "checking") return "Connecting";
  return "API offline";
}

export default function Navbar({ tabs, activeTab, setActiveTab, apiStatus, modelInfo }) {
  const { theme, toggle } = useTheme();

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-surface/95 backdrop-blur">
      <div className="max-w-[1400px] mx-auto px-5 sm:px-8">
        <div className="h-14 flex items-center justify-between gap-6">
          <div className="flex items-center gap-3 shrink-0">
            <ShieldCheck className="h-5 w-5 text-accent" strokeWidth={2} aria-hidden="true" />
            <div className="flex items-baseline gap-2">
              <span className="font-semibold tracking-tight text-[15px] leading-none">SentinelOps</span>
              <span className="hidden sm:inline text-[10px] font-medium uppercase tracking-[0.1em] text-faint leading-none">
                Network Security / MLOps
              </span>
            </div>
          </div>

          <nav className="flex items-center gap-1" aria-label="Primary">
            {tabs.map((tab) => {
              const isActive = activeTab === tab;
              return (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  aria-current={isActive ? "page" : undefined}
                  className={`relative px-3 py-1.5 text-[13px] font-medium transition-colors ${
                    isActive ? "text-ink" : "text-muted hover:text-ink"
                  }`}
                >
                  {LABELS[tab]}
                  <span
                    className={`absolute left-2 right-2 -bottom-[1px] h-[2px] rounded-full transition-opacity ${
                      isActive ? "bg-accent opacity-100" : "opacity-0"
                    }`}
                  />
                </button>
              );
            })}
          </nav>

          <div className="flex items-center gap-4 shrink-0">
            {modelInfo && (
              <span className="hidden md:inline-flex items-center gap-1.5 text-xs text-muted font-mono">
                <span className="h-1.5 w-1.5 rounded-full bg-safe shrink-0" aria-hidden="true" />
                {modelInfo.model_type} <span className="text-faint">production</span>
              </span>
            )}
            <StatusIndicator tone={apiTone(apiStatus)} label={apiLabel(apiStatus)} pulse={apiStatus === "checking"} />
            <button
              onClick={toggle}
              className="p-1.5 rounded-md text-muted hover:text-ink hover:bg-elevated transition-colors"
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
