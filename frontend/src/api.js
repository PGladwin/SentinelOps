/**
 * SentinelOps API Client
 */

const API_BASE_URL = "http://localhost:8000";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function fetchModelInfo() {
  const res = await fetch(`${API_BASE_URL}/model-info`);
  if (!res.ok) throw new Error("Failed to fetch model info");
  return res.json();
}

export async function fetchDemoSamples() {
  const res = await fetch(`${API_BASE_URL}/demo-samples`);
  if (!res.ok) throw new Error("Failed to fetch demo samples");
  return res.json();
}

export async function postPredict(features) {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Prediction request failed");
  }
  return res.json();
}

export async function postBatchPredict(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE_URL}/batch-predict`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Batch prediction failed");
  }
  return res.json();
}
