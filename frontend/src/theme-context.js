import { createContext, useContext } from "react";

/**
 * Chart colours, resolved per theme.
 *
 * Recharts writes these into SVG presentation attributes, where CSS custom
 * properties are unreliable across browsers — so concrete hex values are
 * handed over rather than `var(--…)`.
 */
export const CHART_TOKENS = {
  dark: {
    accent: "#4da3ff",
    danger: "#ff6b63",
    warn: "#f0b429",
    safe: "#3ecf8e",
    grid: "#222a36",
    axis: "#6a7482",
    tooltipBg: "#11151c",
    tooltipLine: "#222a36",
    tooltipInk: "#e7edf5",
  },
  light: {
    accent: "#0b63ce",
    danger: "#c02a2a",
    warn: "#a56708",
    safe: "#157347",
    grid: "#e3e7ec",
    axis: "#8b95a4",
    tooltipBg: "#ffffff",
    tooltipLine: "#e3e7ec",
    tooltipInk: "#131820",
  },
};

export const STORAGE_KEY = "sentinelops-theme";

export const ThemeContext = createContext({
  theme: "dark",
  toggle: () => {},
  chart: CHART_TOKENS.dark,
});

export const useTheme = () => useContext(ThemeContext);

/** Shared Recharts tooltip styling, so every chart reads identically. */
export function tooltipStyle(chart) {
  return {
    background: chart.tooltipBg,
    border: `1px solid ${chart.tooltipLine}`,
    borderRadius: 6,
    fontSize: 12,
    color: chart.tooltipInk,
    boxShadow: "none",
  };
}
