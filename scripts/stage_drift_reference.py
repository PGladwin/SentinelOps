"""
SentinelOps - DVC Stage: Drift Reference
=========================================
Snapshots the training distribution into data/reference/ as the baseline every
future traffic batch is compared against.

Fills the directory params.yaml:paths.reference_dir has always pointed at but
which was empty, leaving drift detection with nothing to compare to.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("sentinelops.stage.drift_reference")


def main() -> int:
    parser = argparse.ArgumentParser(description="DVC stage: build the drift reference snapshot")
    parser.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    parser.add_argument("--max-rows", type=int, default=50000)
    args = parser.parse_args()

    import yaml
    with open(args.params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    from src.mlops.drift import build_reference

    processed_dir = PROJECT_ROOT / params["paths"]["processed_dir"]
    reference_dir = PROJECT_ROOT / params["paths"]["reference_dir"]
    train_path = processed_dir / "train.parquet"
    features_path = processed_dir / "feature_names.json"

    if not train_path.exists():
        logger.error(f"{train_path} not found. Run `dvc repro preprocess` first.")
        return 1

    with open(features_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    build_reference(
        train_path=train_path,
        feature_names=feature_names,
        output_dir=reference_dir,
        max_rows=args.max_rows,
        random_seed=params.get("general", {}).get("random_seed", 42),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
