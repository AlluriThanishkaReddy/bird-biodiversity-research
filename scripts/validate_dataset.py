"""
Standalone CLI entry point for BirdCLEF 2024 dataset validation.

Executes all 13 dataset integrity checks:
1. Dataset directory structure
2. train_metadata.csv schema & completeness
3. Number of recordings
4. Number of unique species
5. Class distribution
6. Missing metadata
7. Duplicate filenames
8. Physical existence of referenced audio files
9. Audio decoding
10. Sampling-rate distribution
11. Duration distribution
12. Number of channels
13. Corrupted/unreadable audio

Outputs:
- reports/results/dataset_summary.json
- reports/results/dataset_quality.csv
- reports/figures/class_distribution.png
- reports/figures/duration_distribution.png
- reports/figures/sample_rate_distribution.png
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validator import DatasetValidator
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("cli_validate_dataset")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run comprehensive BirdCLEF 2024 dataset validation."
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
        help="Override path to dataset root (defaults to data.dataset_root in config).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of concurrent worker threads for audio inspection.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Custom destination for dataset_summary.json.",
    )
    parser.add_argument(
        "--quality-output",
        type=str,
        default=None,
        help="Custom destination for dataset_quality.csv.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=None,
        help="Custom destination directory for generated figures.",
    )
    return parser.parse_args()


def print_cli_summary(validator: DatasetValidator):
    """Print an executive research summary to stdout."""
    summary = validator.summary_report
    checks = summary.get("validation_checks", {})

    print("\n" + "=" * 80)
    print("      BIRDCLEF 2024 DATASET AUDIT & INTEGRITY REPORT      ")
    print("=" * 80)
    print(f"Dataset Root:           {summary.get('dataset_root')}")
    print(f"Overall Status:         {summary.get('overall_status')}")
    print(f"Audit Timestamp:        {summary.get('timestamp')}")
    print("-" * 80)

    print("CHECK RESULTS (13 CRITICAL VERIFICATIONS):")
    for check_name, res in checks.items():
        status = res.get("status", "UNKNOWN")
        icon = "[PASS]" if status == "PASS" else ("[WARN]" if status == "WARN" else "[FAIL]")
        clean_name = check_name.replace("_", " ").title()
        print(f"  {icon:<8} {clean_name}")

    print("-" * 80)
    # Core stats
    meta_check = checks.get("3_number_of_recordings", {})
    species_check = checks.get("4_unique_species", {})
    class_check = checks.get("5_class_distribution", {})
    sr_check = checks.get("10_sampling_rate_distribution", {})
    dur_check = checks.get("11_duration_distribution", {})
    ch_check = checks.get("12_number_of_channels", {})
    dup_check = checks.get("7_duplicate_filenames", {})
    corrupt_check = checks.get("13_corrupted_unreadable_audio", {})
    missing_meta = checks.get("6_missing_metadata", {})

    print("KEY DATASET METRICS:")
    print(f"  Total Recordings:             {meta_check.get('metadata_recordings_count', 'N/A'):,}")
    print(f"  Files Physically on Disk:     {meta_check.get('disk_recordings_count', 'N/A'):,}")
    print(f"  Missing Audio Files:          {meta_check.get('missing_from_disk_count', 0)}")
    print(f"  Unique Avian Species:         {species_check.get('unique_species_count', 'N/A')}")
    print(f"  Class Imbalance Ratio:        {class_check.get('imbalance_ratio', 'N/A')}x (min={class_check.get('min_recordings_per_class')}, max={class_check.get('max_recordings_per_class')})")
    print(f"  Classes <= 10 Recordings:     {class_check.get('classes_under_10_samples_count', 0)} species (extreme few-shot)")
    print(f"  Classes <= 20 Recordings:     {class_check.get('classes_under_20_samples_count', 0)} species")
    
    dur_stats = dur_check.get("summary_statistics", {})
    print(f"  Total Audio Content:          {dur_stats.get('total_audio_hours', 'N/A')} Hours")
    print(f"  Duration (Min / Median / Max): {dur_stats.get('min_seconds', 'N/A')}s / {dur_stats.get('median_seconds', 'N/A')}s / {dur_stats.get('max_seconds', 'N/A')}s")
    print(f"  Clips < Baseline 5.0s:        {dur_check.get('clips_shorter_than_baseline_5s', 0):,} ({dur_check.get('clips_shorter_than_baseline_percentage', 0)}%)")
    
    print(f"  Channels:                     Mono: {ch_check.get('mono_count', 0):,} | Stereo: {ch_check.get('stereo_count', 0):,}")
    print(f"  Baseline 32 kHz Files:        {sr_check.get('files_matching_baseline_sr', 0):,} ({100 - sr_check.get('resampling_percentage', 0):.1f}%)")
    print(f"  Resampling Required:          {sr_check.get('files_requiring_resampling', 0):,} ({sr_check.get('resampling_percentage', 0):.1f}%)")
    print(f"  Corrupted / Unreadable:       {corrupt_check.get('corrupted_files_count', 0)}")
    print(f"  Missing Lat/Lon Metadata:     {missing_meta.get('latitude_missing', 0):,} recordings ({missing_meta.get('latitude_missing', 0) / meta_check.get('metadata_recordings_count', 1) * 100:.2f}%)")
    print(f"  Shared Audio Basenames (XC):  {dup_check.get('shared_audio_basenames_count', 0)} IDs shared across species (Leakage Risk!)")
    print("=" * 80 + "\n")


def main():
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    cfg = load_config(config_path)

    validator = DatasetValidator(
        config=cfg,
        dataset_root=args.dataset_root,
        num_workers=args.num_workers,
    )

    # 1. Run all 13 validation checks
    validator.validate_all()

    # 2. Save JSON summary and Quality CSV
    summary_path = validator.save_summary_report(args.summary_output)
    quality_path = validator.save_quality_table(args.quality_output)

    # 3. Generate visualization plots
    plots = validator.generate_plots(args.figures_dir)

    # 4. Print CLI summary
    print_cli_summary(validator)

    logger.info("Validation pipeline completed successfully.")
    logger.info(f"Summary JSON:   {summary_path}")
    logger.info(f"Quality CSV:    {quality_path}")
    for k, p in plots.items():
        logger.info(f"Figure ({k}): {p}")


if __name__ == "__main__":
    main()
