import React, { useCallback, useEffect, useMemo, useState } from "react";
import { CHART_TOKENS, STORAGE_KEY, ThemeContext } from "./theme-context";

/**
 * Resolve the initial theme without waiting for React.
 *
 * index.html runs the same logic in an inline script before first paint, so
 * this only has to agree with what is already on <html> — reading the class
 * back avoids a flash if the two ever diverge.
 */
function initialTheme() {
  if (typeof document === "undefined") return "dark";
  if (document.documentElement.classList.contains("dark")) return "dark";
  if (document.documentElement.classList.contains("light")) return "light";
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "dark" || stored === "light") return stored;
  } catch {
    /* private mode / blocked storage: fall through to the media query */
  }
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(initialTheme);

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("dark", theme === "dark");
    root.classList.toggle("light", theme === "light");
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      /* preference simply will not persist */
    }
  }, [theme]);

  const toggle = useCallback(() => setTheme((t) => (t === "dark" ? "light" : "dark")), []);

  const value = useMemo(() => ({ theme, toggle, chart: CHART_TOKENS[theme] }), [theme, toggle]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}
