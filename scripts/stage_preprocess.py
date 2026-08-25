"""
SentinelOps - DVC Stage: Preprocess
====================================
Second stage of the tracked pipeline.

Consumes data/interim/clean.parquet and produces the model-ready splits and
every preprocessing artifact the serving layer depends on.

All fitting is confined to the training split: NaN fill values, the
correlation filter, and the RobustScaler are each derived from X_train only
and then applied to X_test.

Outputs (data/processed/):
    train.parquet, test.parquet, scaler.pkl, feature_names.json,
    class_encoding.json, inverse_encoding.json, fill_values.json,
    dropped_correlated_features.json, preprocessing_metadata.json
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("sentinelops.stage.preprocess")


def main() -> int:
    parser = argparse.ArgumentParser(description="DVC stage: preprocess cleaned data")
    parser.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    args = parser.parse_args()

    import pandas as pd
    import yaml

    with open(args.params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    from src.data.preprocess import print_preprocessing_report, run_preprocessing

    interim_dir = PROJECT_ROOT / params["paths"].get("interim_dir", "data/interim")
    clean_path = interim_dir / "clean.parquet"
    if not clean_path.exists():
        logger.error(
            f"{clean_path} not found. Run scripts/stage_prepare_data.py first "
            f"(or `dvc repro prepare_data`)."
        )
        return 1

    t0 = time.time()
    logger.info(f"Loading cleaned dataset: {clean_path}")
    clean_df = pd.read_parquet(clean_path)
    logger.info(f"Loaded {len(clean_df):,} rows x {clean_df.shape[1]} cols")

    # dev_sample_size lives in params.yaml rather than a CLI flag so DVC can
    # see it: a flag would be invisible to change detection and `dvc repro`
    # would wrongly consider the stage up to date after switching sample size.
    dev_sample = params["preprocessing"].get("dev_sample_size")
    logger.info(
        f"dev_sample_size={dev_sample!r} "
        f"({'stratified development sample' if dev_sample else 'FULL dataset'})"
    )

    artifacts = run_preprocessing(clean_df, params, use_dev_sample=dev_sample is not None)
    del clean_df

    print_preprocessing_report(artifacts)
    logger.info(f"STAGE COMPLETE in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
