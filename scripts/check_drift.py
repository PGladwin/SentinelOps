"""
SentinelOps - Drift Check
==========================
Compares a traffic batch against the training reference distribution and writes
reports/drift_summary.json, which the API serves at GET /mlops/drift.

Two modes:

  --batch <file.csv|.parquet>
      Score a real batch. Raw CSVs are put through the same preprocessing the
      Champion uses, so the comparison happens in the representation the model
      actually sees.

  --simulate
      Build a deliberately shifted batch from held-out TEST rows by resampling
      the class mix (the reference is ~85% BENIGN; the simulated batch is
      ~50/50 benign/attack). Every row is genuine CIC-IDS2017 traffic -- only
      the proportions are constructed, which is exactly what a real attack
      campaign looks like from a sensor's point of view.

  --simulate --no-shift
      Control: a batch drawn from test rows WITHOUT resampling. This should
      report no dataset drift, and is what makes the positive result meaningful.

Usage:
    python scripts/check_drift.py --simulate
    python scripts/check_drift.py --simulate --no-shift
    python scripts/check_drift.py --batch samples/demo_traffic_mixed.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("sentinelops.drift")


def build_simulated_batch(
    test_path: Path,
    n_rows: int,
    attack_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    """
    Resample held-out test rows to a target benign/attack mix.

    The rows are real; only their proportions are constructed.
    """
    df = pd.read_parquet(test_path)
    if "label_class" not in df.columns:
        raise ValueError(f"{test_path} has no label_class column.")

    benign = df[df["label_class"] == "BENIGN"]
    attack = df[df["label_class"] != "BENIGN"]

    n_attack = min(len(attack), int(round(n_rows * attack_fraction)))
    n_benign = min(len(benign), n_rows - n_attack)

    batch = pd.concat([
        benign.sample(n=n_benign, random_state=random_seed),
        attack.sample(n=n_attack, random_state=random_seed),
    ]).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    logger.info(
        f"Simulated batch: {len(batch):,} rows "
        f"({n_benign:,} benign / {n_attack:,} attack = {n_attack / max(1, len(batch)):.0%} attack)"
    )
    return batch


def build_control_batch(test_path: Path, n_rows: int, random_seed: int) -> pd.DataFrame:
    """Draw an unmodified sample of test rows -- the in-distribution control."""
    df = pd.read_parquet(test_path)
    batch = df.sample(n=min(n_rows, len(df)), random_state=random_seed).reset_index(drop=True)
    logger.info(f"Control batch: {len(batch):,} rows, class mix untouched")
    return batch


def load_external_batch(path: Path, bundle: dict[str, Any]) -> pd.DataFrame:
    """
    Load a real traffic file and put it into the Champion's scaled space.

    The reference is stored scaled, so an unscaled batch would register as
    drift on every single feature -- an artifact of units, not a real shift.
    """
    from src.data.inference_prep import normalize_columns, prepare_with_report

    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path, low_memory=False)

    raw = normalize_columns(raw)
    X, report = prepare_with_report(raw, bundle, already_normalized=True, allow_missing=True)
    if report["n_imputed_columns"]:
        logger.warning(
            f"{report['n_imputed_columns']} feature(s) imputed from training medians: "
            f"{report['imputed_columns']}"
        )
    return pd.DataFrame(X, columns=bundle["feature_names"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a traffic batch against the drift reference")
    parser.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    parser.add_argument("--batch", help="CSV or Parquet file of traffic to score")
    parser.add_argument("--simulate", action="store_true", help="Build a batch from held-out test rows")
    parser.add_argument("--no-shift", action="store_true", help="With --simulate: control batch, no resampling")
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--attack-fraction", type=float, default=0.50)
    parser.add_argument("--html", action="store_true", help="Also render an Evidently HTML report")
    parser.add_argument("--no-save", action="store_true", help="Print only; do not write drift_summary.json")
    args = parser.parse_args()

    import yaml
    with open(args.params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    from src.mlops.drift import detect_drift, load_reference, save_drift_summary

    seed = params.get("general", {}).get("random_seed", 42)
    drift_cfg = params.get("drift", {})
    processed_dir = PROJECT_ROOT / params["paths"]["processed_dir"]
    reference_dir = PROJECT_ROOT / params["paths"]["reference_dir"]
    reports_dir = PROJECT_ROOT / params["paths"]["reports_dir"]

    reference = load_reference(reference_dir)
    if reference is None:
        logger.error("No reference distribution. Run `dvc repro drift_reference` first.")
        return 1

    if args.batch:
        from src.data.inference_prep import load_inference_bundle
        bundle = load_inference_bundle(PROJECT_ROOT / params["paths"]["models_dir"] / "champion_preprocessor.json")
        current = load_external_batch(Path(args.batch), bundle)
        source = args.batch
    elif args.simulate:
        test_path = processed_dir / "test.parquet"
        if args.no_shift:
            current = build_control_batch(test_path, args.rows, seed)
            source = "held-out test sample (control, no resampling)"
        else:
            current = build_simulated_batch(test_path, args.rows, args.attack_fraction, seed)
            source = f"held-out test rows resampled to {args.attack_fraction:.0%} attack"
    else:
        parser.error("Provide --batch <file> or --simulate")

    feature_names = json.loads((processed_dir / "feature_names.json").read_text(encoding="utf-8"))

    summary = detect_drift(
        reference=reference,
        current=current,
        features=feature_names,
        stattest=drift_cfg.get("stattest", "auto"),
        drift_share_threshold=float(drift_cfg.get("drift_share_threshold", 0.20)),
        pvalue_threshold=float(drift_cfg.get("pvalue_threshold", 0.05)),
        distance_threshold=float(drift_cfg.get("distance_threshold", 0.10)),
    )
    summary["current_source"] = source

    if args.html:
        try:
            from src.mlops.drift import render_evidently_report
            path = render_evidently_report(reference, current, feature_names, reports_dir)
            summary["html_report"] = str(path.relative_to(PROJECT_ROOT)) if path else None
        except Exception as e:
            logger.warning(f"Evidently HTML report unavailable: {e}")
            summary["html_report"] = None

    if not args.no_save:
        save_drift_summary(summary, reports_dir)

    # ---- Console summary ------------------------------------------
    print("\n" + "=" * 74)
    print("DRIFT REPORT")
    print("=" * 74)
    print(f"  Source            : {source}")
    print(f"  Reference rows    : {summary['reference_rows']:,}")
    print(f"  Current rows      : {summary['current_rows']:,}")
    print(f"  Test              : {summary['stattest']}")
    print(f"  Drifted features  : {summary['n_drifted_features']} / {summary['n_features']} "
          f"({summary['drift_share']:.1%})")
    print(f"  Share threshold   : {summary['drift_share_threshold']:.0%}")
    print(f"  VERDICT           : {'DRIFT DETECTED' if summary['drift_detected'] else 'no dataset drift'}")
    if summary["top_drifted_features"]:
        print("\n  Most-shifted features:")
        for item in summary["top_drifted_features"][:10]:
            print(f"    {item['feature']:<30} {item['test']:<12} score={item['score']:.4f} "
                  f"(threshold {item['threshold']})")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
