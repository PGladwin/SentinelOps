"""
SentinelOps - DVC Stage: Ingest + Clean
========================================
First stage of the tracked pipeline.

Reads the raw CIC-IDS2017 Parquet files, normalizes labels to the canonical
8-class taxonomy, removes duplicates and constant columns, and writes a single
cleaned dataset to data/interim/clean.parquet.

Split out from run_phase1.py so `dvc repro` can re-run preprocessing (a
different correlation threshold or scaler) without repeating ingestion of
2.3M rows, which is the expensive half.

Outputs:
    data/interim/clean.parquet
    reports/phase1_ingestion_report.json
    reports/phase1_cleaning_report.json
"""

import argparse
import json
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
logger = logging.getLogger("sentinelops.stage.prepare")


def make_serializable(obj):
    """Coerce report values into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    return str(obj)


def main() -> int:
    parser = argparse.ArgumentParser(description="DVC stage: ingest and clean CIC-IDS2017")
    parser.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    args = parser.parse_args()

    import yaml
    with open(args.params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    from src.data.clean import clean_dataframe, print_cleaning_report
    from src.data.ingest import load_all_parquet, print_ingestion_report

    reports_dir = PROJECT_ROOT / params["paths"]["reports_dir"]
    interim_dir = PROJECT_ROOT / params["paths"].get("interim_dir", "data/interim")
    reports_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    raw_df, ingestion_report = load_all_parquet(params)
    print_ingestion_report(ingestion_report)

    with open(reports_dir / "phase1_ingestion_report.json", "w", encoding="utf-8") as f:
        json.dump(ingestion_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Ingestion complete in {time.time() - t0:.1f}s")

    t1 = time.time()
    clean_df, cleaning_stats = clean_dataframe(raw_df, params)
    print_cleaning_report(cleaning_stats)

    # Release the raw frame before writing; on the full corpus both together
    # are the pipeline's memory high-water mark.
    del raw_df

    with open(reports_dir / "phase1_cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(make_serializable(cleaning_stats), f, indent=2, ensure_ascii=False)
    logger.info(f"Cleaning complete in {time.time() - t1:.1f}s")

    out_path = interim_dir / "clean.parquet"
    clean_df.to_parquet(out_path, index=False, compression="snappy")
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"Wrote {out_path} ({len(clean_df):,} rows x {clean_df.shape[1]} cols, {size_mb:.0f} MB)")

    logger.info(f"STAGE COMPLETE in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
