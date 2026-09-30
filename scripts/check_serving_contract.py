"""
SentinelOps - Serving Contract Check
====================================
Asserts that the committed Champion artifacts are internally consistent and
loadable, before anything is deployed.

The failure this exists to catch: training promotes a new Champion and writes
``champion.pkl``, ``champion_features.json`` and ``champion_preprocessor.json``
as a set. If any one of them is stale -- a partial commit, a merge that took
one side of a conflict, a rerun that crashed between writes -- every unit test
still passes, because they exercise the code rather than the artifacts. The
mismatch surfaces at container startup instead, as a 503 in production.

Checks:
  1. Every expected artifact is present.
  2. The model's own feature count matches champion_features.json.
  3. The preprocessing bundle's feature order matches, exactly and in order.
  4. The class encoding covers every class the model can emit.
  5. A real flow scores end to end, producing a valid probability distribution.

Exit code 0 on success, 1 on any violation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"

REQUIRED_ARTIFACTS = [
    "champion.pkl",
    "champion_features.json",
    "champion_metadata.json",
    "champion_preprocessor.json",
    "demo_samples.json",
]

failures: list[str] = []
checks = 0


def check(condition: bool, description: str, detail: str = "") -> bool:
    """Record one assertion and print its result."""
    global checks
    checks += 1
    if condition:
        print(f"  PASS  {description}")
        return True
    print(f"  FAIL  {description}")
    if detail:
        print(f"        {detail}")
    failures.append(description)
    return False


def main() -> int:
    print(f"Serving contract check against {MODELS_DIR}\n")

    # -- 1. Artifacts present -------------------------------------------
    print("Artifacts")
    for name in REQUIRED_ARTIFACTS:
        if not check((MODELS_DIR / name).exists(), f"{name} present"):
            # Nothing downstream can run without the full set.
            print("\nAborting: required artifacts are missing.")
            return 1

    # -- 2. Model and feature list agree --------------------------------
    print("\nFeature contract")
    from src.data.inference_prep import load_inference_bundle
    from src.models.train import load_champion_model

    model, features, metadata = load_champion_model(MODELS_DIR)
    check(len(features) > 0, "champion_features.json is non-empty")

    model_n = getattr(model, "n_features_in_", None)
    check(
        model_n is None or model_n == len(features),
        "model input width matches the feature list",
        f"model expects {model_n}, feature list has {len(features)}",
    )

    bundle = load_inference_bundle(MODELS_DIR / "champion_preprocessor.json")
    check(
        bundle["feature_names"] == features,
        "preprocessor bundle matches the feature list, in order",
        "Regenerate models/champion_preprocessor.json -- the bundle and the "
        "Champion disagree, so served predictions would use mismatched scaling.",
    )
    check(
        all(f in bundle["fill_values"] for f in features),
        "every feature has a training-median fill value",
    )

    # -- 3. Class encoding ----------------------------------------------
    print("\nClass taxonomy")
    encoding = metadata.get("class_mapping", {})
    check(bool(encoding), "champion metadata carries a class mapping")

    if encoding:
        n_model_classes = getattr(model, "n_classes_", len(encoding))
        check(
            len(encoding) >= n_model_classes,
            "class mapping covers every class the model can emit",
            f"model emits {n_model_classes} classes, mapping has {len(encoding)}",
        )
        check(
            len(set(encoding.values())) == len(encoding),
            "class encoding indices are unique",
        )

    # -- 4. End-to-end scoring ------------------------------------------
    print("\nInference")
    with open(MODELS_DIR / "demo_samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not check(bool(samples), "demo sample bank is non-empty"):
        return 1

    from api.model_service import ModelService

    service = ModelService(models_dir=MODELS_DIR)
    sample = samples[0]
    result = service.predict_single(sample["features"])

    probabilities = np.array(list(result["probabilities"].values()))
    check(
        abs(probabilities.sum() - 1.0) < 1e-4,
        "probabilities form a valid distribution",
        f"sum = {probabilities.sum():.6f}",
    )
    check(
        result["prediction"] in encoding,
        "predicted label belongs to the taxonomy",
        f"got {result['prediction']!r}",
    )
    check(len(result["explanation"]) > 0, "SHAP explanation is produced")

    # -- Summary ---------------------------------------------------------
    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("\nFailed:")
        for name in failures:
            print(f"  - {name}")
        return 1

    print(f"\nChampion '{metadata.get('model_name')}' is consistent and servable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
