/**
 * SentinelOps API Client
 *
 * Base URL comes from VITE_API_BASE_URL so the same build can point at a
 * local uvicorn or a deployed service. It was previously hardcoded to
 * localhost:8000, which made the bundle undeployable.
 */

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

/** Public repo backing the CI/CD panel; overridable per deployment. */
const GITHUB_REPO = import.meta.env.VITE_GITHUB_REPO || "prithish47/Mlops-proj";

export { API_BASE_URL, GITHUB_REPO };

/**
 * Surface the API's own error text rather than a generic status message.
 * FastAPI puts the useful part (missing columns, row caps, why the model
 * failed to load) in `detail`, and hiding it makes the UI undebuggable.
 */
async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new Error(
      `Cannot reach the SentinelOps API at ${API_BASE_URL}. Is the backend running?`
    );
  }

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body; keep the status line */
    }
    throw new Error(detail);
  }

  return res.json();
}

// ---------------------------------------------------------------------------
// Core service
// ---------------------------------------------------------------------------

export const fetchHealth = () => request("/health");
export const fetchModelInfo = () => request("/model-info");
export const fetchDemoSamples = () => request("/demo-samples");

export const postPredict = (features) =>
  request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });

/** Full SOC analysis of a raw traffic CSV. */
export const postAnalyze = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return request("/analyze", { method: "POST", body: formData });
};

// ---------------------------------------------------------------------------
// MLOps control panel
// ---------------------------------------------------------------------------

export const fetchMlopsState = () => request("/mlops/state");
export const fetchMlopsRuns = () => request("/mlops/runs");
export const fetchMlopsRegistry = () => request("/mlops/registry");
export const fetchMlopsGovernance = () => request("/mlops/governance");
export const fetchMlopsMetrics = () => request("/mlops/metrics");
export const fetchMlopsPipeline = () => request("/mlops/pipeline");
export const fetchMlopsDrift = () => request("/mlops/drift");

export const driftReportUrl = () => `${API_BASE_URL}/mlops/drift/report`;

/**
 * GitHub Actions history, queried directly rather than proxied.
 *
 * Public repos allow unauthenticated reads (60/hr), so this panel shows the
 * genuinely current CI state instead of a snapshot exported at pipeline time.
 * Returns null when the repo has no workflow history or the rate limit is hit,
 * so the panel can say so rather than render an empty table.
 */
export async function fetchGithubRuns(limit = 10) {
  const url = `https://api.github.com/repos/${GITHUB_REPO}/actions/runs?per_page=${limit}`;
  try {
    const res = await fetch(url, { headers: { Accept: "application/vnd.github+json" } });
    if (!res.ok) {
      // 404 from the Actions API means the repository is private:
      // unauthenticated reads cannot see it. A public repo with no workflows
      // returns 200 and an empty list instead.
      const reason =
        res.status === 404
          ? `No public Actions history for ${GITHUB_REPO}. The repository is private, or CI has not been set up yet.`
          : res.status === 403
            ? "GitHub API rate limit reached (60 requests/hour for unauthenticated reads)."
            : `GitHub API returned ${res.status}.`;
      return { available: false, reason, repo: GITHUB_REPO, runs: [] };
    }
    const data = await res.json();
    return {
      available: true,
      repo: GITHUB_REPO,
      runs: (data.workflow_runs || []).map((r) => ({
        id: r.id,
        name: r.name,
        number: r.run_number,
        event: r.event,
        status: r.status,
        conclusion: r.conclusion,
        branch: r.head_branch,
        created_at: r.created_at,
        url: r.html_url,
      })),
    };
  } catch {
    return { available: false, reason: "GitHub API unreachable", runs: [] };
  }
}
