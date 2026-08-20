"""
SentinelOps - Phase 1 Pipeline Runner
========================================
Runs the complete Phase 1 data pipeline:
  1. Ingest all Parquet files
  2. Clean and normalize labels
  3. Preprocess (sample → split → scale → save)

Usage:
    python run_phase1.py                  # Uses dev sample (150k rows)
    python run_phase1.py --full           # Uses full dataset
    python run_phase1.py --params custom_params.yaml

Output files:
    data/processed/train.parquet
    data/processed/test.parquet
    data/processed/scaler.pkl
    data/processed/feature_names.json
    data/processed/class_encoding.json
    data/processed/inverse_encoding.json
    data/processed/fill_values.json
    data/processed/dropped_correlated_features.json
    data/processed/preprocessing_metadata.json
    reports/phase1_ingestion_report.json
    reports/phase1_cleaning_report.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configure logging before any imports
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("sentinelops.phase1")


def load_params(params_path: str) -> dict:
    """Load params.yaml using PyYAML."""
    try:
        import yaml
    except ImportError:
        logger.error(
            "PyYAML not installed. Install with: pip install pyyaml"
        )
        sys.exit(1)

    path = Path(params_path)
    if not path.exists():
        logger.error(f"params.yaml not found: {path.resolve()}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    logger.info(f"Loaded params from: {path.resolve()}")
    return params


def main():
    parser = argparse.ArgumentParser(
        description="SentinelOps Phase 1: Data Ingestion, Cleaning, Preprocessing"
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Path to params.yaml (default: params.yaml)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full dataset instead of development sample",
    )
    parser.add_argument(
        "--no-corr-filter",
        action="store_true",
        help="Skip correlation-based feature filtering",
    )
    args = parser.parse_args()

    start_time = time.time()

    # ----------------------------------------------------------------
    # Load parameters
    # ----------------------------------------------------------------
    params = load_params(args.params)

    if args.full:
        logger.info("--full flag set: will use FULL dataset (not dev sample)")
        # Override dev_sample_size to None
        params["preprocessing"]["dev_sample_size"] = None
    else:
        dev_size = params["preprocessing"].get("dev_sample_size")
        if dev_size:
            logger.info(f"Using development sample: {dev_size:,} rows")
        else:
            logger.info("dev_sample_size=null in params.yaml → using full dataset")

    if args.no_corr_filter:
        logger.info("--no-corr-filter: skipping correlation filtering")
        params["preprocessing"]["correlation_threshold"] = None

    # ----------------------------------------------------------------
    # Ensure reports directory exists
    # ----------------------------------------------------------------
    reports_dir = Path(params["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: Ingestion
    # ----------------------------------------------------------------
    logger.info("")
    logger.info("STEP 1: DATA INGESTION")
    logger.info("-" * 40)

    from src.data.ingest import load_all_parquet, print_ingestion_report

    t0 = time.time()
    raw_df, ingestion_report = load_all_parquet(params)
    t1 = time.time()

    print_ingestion_report(ingestion_report)

    # Save ingestion report
    ingestion_report_path = reports_dir / "phase1_ingestion_report.json"
    with open(ingestion_report_path, "w", encoding="utf-8") as f:
        # Convert non-serializable label keys (may contain unicode)
        json.dump(ingestion_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Ingestion report saved: {ingestion_report_path}")
    logger.info(f"Ingestion time: {t1 - t0:.1f}s")

    # ----------------------------------------------------------------
    # Step 2: Cleaning
    # ----------------------------------------------------------------
    logger.info("")
    logger.info("STEP 2: DATA CLEANING")
    logger.info("-" * 40)

    from src.data.clean import clean_dataframe, print_cleaning_report

    t0 = time.time()
    clean_df, cleaning_stats = clean_dataframe(raw_df, params)
    t1 = time.time()

    print_cleaning_report(cleaning_stats)

    # Save cleaning report
    cleaning_report_path = reports_dir / "phase1_cleaning_report.json"

    # Convert any non-serializable items
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, (int, float, bool, str, type(None))):
            return obj
        else:
            return str(obj)

    with open(cleaning_report_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(cleaning_stats), f, indent=2, ensure_ascii=False)
    logger.info(f"Cleaning report saved: {cleaning_report_path}")
    logger.info(f"Cleaning time: {t1 - t0:.1f}s")

    # Free raw_df memory
    del raw_df

    # ----------------------------------------------------------------
    # Step 3: Preprocessing
    # ----------------------------------------------------------------
    logger.info("")
    logger.info("STEP 3: PREPROCESSING")
    logger.info("-" * 40)

    from src.data.preprocess import run_preprocessing, print_preprocessing_report

    use_dev = not args.full

    t0 = time.time()
    artifacts = run_preprocessing(clean_df, params, use_dev_sample=use_dev)
    t1 = time.time()

    print_preprocessing_report(artifacts)
    logger.info(f"Preprocessing time: {t1 - t0:.1f}s")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    total_time = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time          : {total_time:.1f}s")
    logger.info(f"Train samples       : {artifacts['metadata']['n_train']:,}")
    logger.info(f"Test samples        : {artifacts['metadata']['n_test']:,}")
    logger.info(f"Feature count       : {len(artifacts['feature_names'])}")
    logger.info(f"Classes             : {len(artifacts['class_encoding'])}")
    logger.info("")
    logger.info("Ready for Phase 2: Feature Engineering + Model Training")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
