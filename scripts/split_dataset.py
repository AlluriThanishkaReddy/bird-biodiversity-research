"""
Standalone CLI entry point for BirdCLEF 2024 leakage-safe dataset splitting.

Executes recording-level grouped stratified splitting:
- 70% Train / 15% Validation / 15% Test
- Groups by original audio recording ID (XC ID) to prevent cross-split leakage
- Guarantees species representation in train, validation, and test sets
- Produces data/manifests/split_manifest.csv and reports/results/split_summary.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splitter import LeakageSafeSplitter
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("cli_split_dataset")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run leakage-safe recording-level grouped stratified dataset splitting."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Override path to dataset root.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Target training set proportion (default: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Target validation set proportion (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Target test set proportion (default: 0.15).",
    )
    parser.add_argument(
        "--manifest-output",
        type=str,
        default=None,
        help="Custom destination for split_manifest.csv.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Custom destination for split_summary.json.",
    )
    return parser.parse_args()


def print_cli_summary(summary: dict):
    """Print executive splitting audit summary to stdout."""
    actual = summary.get("actual_splits", {})
    leak = summary.get("leakage_verification", {})
    cov = summary.get("species_coverage", {})
    rare = summary.get("rare_species_audit", {}).get("species_under_10_samples", {})

    print("\n" + "=" * 80)
    print("      BIRDCLEF 2024 LEAKAGE-SAFE DATASET SPLIT REPORT      ")
    print("=" * 80)
    print(f"Random Seed:                   {summary.get('random_seed')}")
    print(f"Total Rows / Unique Audio:     {summary.get('total_metadata_rows'):,} / {summary.get('total_unique_recordings'):,}")
    print(f"Target Proportions:            70.0% Train | 15.0% Val | 15.0% Test")
    print("-" * 80)

    print("ACTUAL SPLIT ALLOCATIONS:")
    for s_name in ["train", "val", "test"]:
        s_data = actual.get(s_name, {})
        print(
            f"  {s_name.upper():<6}: {s_data.get('count', 0):>6,} rows "
            f"({s_data.get('percentage', 0):>5.2f}%) | "
            f"Unique Audio: {s_data.get('unique_recordings', 0):>5,} | "
            f"Species: {s_data.get('unique_species', 0):>3}/182"
        )

    print("-" * 80)
    print("LEAKAGE & REPRODUCIBILITY VERIFICATION:")
    print(f"  Every Recording Assigned:     {leak.get('every_recording_assigned')}")
    print(f"  Recording ID Leakage Found:   {leak.get('recording_id_leakage_detected')} (Count: {leak.get('leaking_recordings_count')})")
    print(f"  Shared Multi-Species Audio:   {leak.get('shared_xc_recordings_total')} XC recordings")
    print(f"  Shared Audio Leaking Across:  {leak.get('shared_xc_recordings_leaking')} (All pairs kept strictly in same split)")
    print(f"  All 182 Species in All Splits:{cov.get('all_species_in_all_splits')}")

    print("-" * 80)
    print("EXTREME FEW-SHOT SPECIES AUDIT (N <= 10):")
    for sp, data in rare.items():
        print(f"  {sp:<10} Total={data['total']:>2} | Train={data['train']:>2} | Val={data['val']:>2} | Test={data['test']:>2}")

    print("=" * 80 + "\n")


def main():
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    cfg = load_config(config_path)

    splitter = LeakageSafeSplitter(
        config=cfg,
        dataset_root=args.dataset_root,
        random_seed=args.random_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # 1. Generate split
    splitter.create_split()

    # 2. Save manifest and summary
    manifest_path = splitter.save_split_manifest(args.manifest_output)
    summary_path = splitter.save_split_summary(args.summary_output)

    # 3. Print CLI summary
    print_cli_summary(splitter.summary_report)

    logger.info("Dataset splitting pipeline completed successfully.")
    logger.info(f"Split Manifest: {manifest_path}")
    logger.info(f"Split Summary:  {summary_path}")


if __name__ == "__main__":
    main()
