"""
Dataset validation module for BirdCLEF 2024 bioacoustic research.

Performs comprehensive validation of the external dataset including:
1. Directory structure and expected files
2. train_metadata.csv integrity and schema
3. Total number of recordings
4. Unique species count and taxonomic alignment
5. Class distribution and imbalance analysis
6. Missing metadata quantification
7. Duplicate filenames and shared audio identifiers
8. Physical existence of referenced audio files
9. Audio decoding verification
10. Sampling-rate distribution
11. Duration distribution and windowing implications
12. Channel count distribution (mono vs multi-channel)
13. Corrupted or unreadable audio detection
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.utils.config import ConfigDict, find_default_config_path, load_config
from src.utils.logger import get_logger

logger = get_logger("dataset_validator")


class DatasetValidator:
    """
    Comprehensive validator for the BirdCLEF 2024 dataset.
    
    Adheres strictly to research governance:
    - Raw dataset is treated as immutable and read-only.
    - No audio conversion, clip generation, or data modification.
    - All statistics are computed directly from the actual dataset files.
    """

    EXPECTED_METADATA_COLUMNS = [
        "primary_label",
        "secondary_labels",
        "type",
        "latitude",
        "longitude",
        "scientific_name",
        "common_name",
        "author",
        "license",
        "rating",
        "url",
        "filename",
    ]

    EXPECTED_DATASET_ITEMS = [
        ("train_audio", "dir"),
        ("train_metadata.csv", "file"),
        ("test_soundscapes", "dir"),
        ("eBird_Taxonomy_v2021.csv", "file"),
        ("sample_submission.csv", "file"),
    ]

    def __init__(
        self,
        config: Optional[ConfigDict] = None,
        dataset_root: Optional[str | Path] = None,
        num_workers: int = 16,
    ):
        """
        Initialize the validator with configuration settings.

        Args:
            config: Optional ConfigDict loaded from configs/config.yaml.
            dataset_root: Optional override for the dataset root path.
            num_workers: Number of threads for concurrent audio inspection.
        """
        if config is None:
            self.config = load_config(find_default_config_path())
        else:
            self.config = config

        if dataset_root is not None:
            self.dataset_root = Path(dataset_root).resolve()
        else:
            self.dataset_root = Path(self.config.data.dataset_root).resolve()

        self.num_workers = max(1, num_workers)
        self.train_audio_dir = self.dataset_root / self.config.data.get("train_audio_dir", "train_audio")
        self.train_metadata_path = self.dataset_root / self.config.data.get("train_metadata", "train_metadata.csv")

        # Project output directories
        project_root = find_default_config_path().parent.parent
        self.results_dir = project_root / self.config.outputs.get("results_dir", "reports/results")
        self.figures_dir = project_root / self.config.outputs.get("figures_dir", "reports/figures")

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Baseline audio settings for comparison
        self.baseline_sr = int(self.config.audio.sample_rate)
        self.baseline_duration = float(self.config.audio.clip_duration)

        # Internal state
        self.metadata_df: Optional[pd.DataFrame] = None
        self.quality_df: Optional[pd.DataFrame] = None
        self.summary_report: Dict[str, Any] = {}
        self.check_statuses: Dict[str, Dict[str, Any]] = {}

    def check_directory_structure(self) -> Dict[str, Any]:
        """
        Check 1: Validate dataset directory structure and core items.
        """
        logger.info(f"Validating directory structure at: {self.dataset_root}")
        if not self.dataset_root.is_dir():
            result = {
                "status": "FAIL",
                "message": f"Dataset root directory does not exist: {self.dataset_root}",
                "items_found": {},
            }
            self.check_statuses["1_directory_structure"] = result
            return result

        items_status = {}
        all_passed = True
        for name, item_type in self.EXPECTED_DATASET_ITEMS:
            p = self.dataset_root / name
            if item_type == "dir":
                exists = p.is_dir()
            else:
                exists = p.is_file()
            items_status[name] = {
                "type": item_type,
                "exists": exists,
                "path": str(p),
            }
            if not exists:
                all_passed = False

        # Additional directories (optional/supplemental soundscapes)
        unlabeled_dir = self.dataset_root / "unlabeled_soundscapes"
        items_status["unlabeled_soundscapes"] = {
            "type": "dir",
            "exists": unlabeled_dir.is_dir(),
            "path": str(unlabeled_dir),
            "optional": True,
        }

        # Check train_audio species folders
        species_folders = []
        if self.train_audio_dir.is_dir():
            species_folders = [
                d.name for d in self.train_audio_dir.iterdir() if d.is_dir()
            ]

        result = {
            "status": "PASS" if all_passed else "FAIL",
            "dataset_root": str(self.dataset_root),
            "expected_items": items_status,
            "species_folders_count": len(species_folders),
        }
        self.check_statuses["1_directory_structure"] = result
        return result

    def check_train_metadata(self) -> pd.DataFrame:
        """
        Check 2: Validate train_metadata.csv readability, encoding, and schema.
        """
        logger.info(f"Validating train_metadata.csv at: {self.train_metadata_path}")
        if not self.train_metadata_path.is_file():
            raise FileNotFoundError(f"train_metadata.csv not found at: {self.train_metadata_path}")

        try:
            df = pd.read_csv(self.train_metadata_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(self.train_metadata_path, encoding="latin1")

        missing_cols = [c for c in self.EXPECTED_METADATA_COLUMNS if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in self.EXPECTED_METADATA_COLUMNS]

        status = "PASS" if len(missing_cols) == 0 else "FAIL"
        result = {
            "status": status,
            "path": str(self.train_metadata_path),
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "columns": list(df.columns),
            "missing_expected_columns": missing_cols,
            "extra_columns": extra_cols,
        }
        self.check_statuses["2_train_metadata"] = result
        self.metadata_df = df
        return df

    def check_number_of_recordings(self) -> Dict[str, Any]:
        """
        Check 3: Validate number of recordings in metadata vs on disk.
        """
        if self.metadata_df is None:
            self.check_train_metadata()
        df = self.metadata_df

        metadata_count = int(len(df))

        # Count actual audio files on disk in train_audio
        disk_ogg_files: List[str] = []
        if self.train_audio_dir.is_dir():
            for root, _, files in os.walk(self.train_audio_dir):
                for f in files:
                    if f.lower().endswith(".ogg"):
                        rel_path = os.path.relpath(os.path.join(root, f), self.train_audio_dir)
                        # Normalize to forward slashes
                        disk_ogg_files.append(rel_path.replace("\\", "/"))

        disk_count = len(disk_ogg_files)
        metadata_filenames = set(df["filename"].str.replace("\\", "/"))
        disk_filenames_set = set(disk_ogg_files)

        orphaned_on_disk = list(disk_filenames_set - metadata_filenames)
        missing_from_disk = list(metadata_filenames - disk_filenames_set)

        status = "PASS" if (metadata_count == disk_count and len(missing_from_disk) == 0) else "FAIL"
        result = {
            "status": status,
            "metadata_recordings_count": metadata_count,
            "disk_recordings_count": disk_count,
            "counts_match": metadata_count == disk_count,
            "orphaned_on_disk_count": len(orphaned_on_disk),
            "missing_from_disk_count": len(missing_from_disk),
            "orphaned_samples": orphaned_on_disk[:5],
            "missing_samples": missing_from_disk[:5],
        }
        self.check_statuses["3_number_of_recordings"] = result
        return result

    def check_unique_species(self) -> Dict[str, Any]:
        """
        Check 4: Validate unique species count and alignment with folders and sample submission.
        """
        if self.metadata_df is None:
            self.check_train_metadata()
        df = self.metadata_df

        unique_primary = sorted(df["primary_label"].unique().tolist())
        num_species = len(unique_primary)

        # Folders on disk
        species_folders = []
        if self.train_audio_dir.is_dir():
            species_folders = sorted([
                d.name for d in self.train_audio_dir.iterdir() if d.is_dir()
            ])

        folders_match = (unique_primary == species_folders)

        # Check sample submission if available
        sample_sub_path = self.dataset_root / "sample_submission.csv"
        sample_sub_species = []
        sub_match = None
        if sample_sub_path.is_file():
            sub_df = pd.read_csv(sample_sub_path, nrows=1)
            sample_sub_species = sorted([c for c in sub_df.columns if c != "row_id"])
            sub_match = (unique_primary == sample_sub_species)

        status = "PASS" if (folders_match and (sub_match is None or sub_match)) else "FAIL"
        result = {
            "status": status,
            "unique_species_count": num_species,
            "expected_species_count": 182,
            "species_folders_count": len(species_folders),
            "species_match_folders": folders_match,
            "sample_submission_species_count": len(sample_sub_species),
            "species_match_sample_submission": sub_match,
            "species_list_sample": unique_primary[:10],
        }
        self.check_statuses["4_unique_species"] = result
        return result

    def check_class_distribution(self) -> Dict[str, Any]:
        """
        Check 5: Analyze class distribution, imbalance ratio, and few-shot species.
        """
        if self.metadata_df is None:
            self.check_train_metadata()
        df = self.metadata_df

        counts = df["primary_label"].value_counts()
        min_count = int(counts.min())
        max_count = int(counts.max())
        mean_count = float(counts.mean())
        median_count = float(counts.median())
        std_count = float(counts.std())
        imbalance_ratio = float(max_count / min_count) if min_count > 0 else float("inf")

        # Species with severe class imbalance
        under_10 = counts[counts <= 10].to_dict()
        under_20 = counts[counts <= 20].to_dict()
        under_50 = counts[counts <= 50].to_dict()

        # Secondary label multi-label analysis
        def parse_sec_labels(val: Any) -> List[str]:
            if pd.isna(val) or not val:
                return []
            if isinstance(val, str):
                val_str = val.strip()
                if val_str in ("[]", ""):
                    return []
                try:
                    import ast
                    parsed = ast.literal_eval(val_str)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    # Fallback string stripping
                    cleaned = val_str.strip("[]").replace("'", "").replace('"', "")
                    return [x.strip() for x in cleaned.split(",") if x.strip()]
            return []

        sec_labels_series = df["secondary_labels"].apply(parse_sec_labels)
        sec_lens = sec_labels_series.apply(len)
        multi_label_stats = {
            "clips_with_0_secondary": int((sec_lens == 0).sum()),
            "clips_with_1_secondary": int((sec_lens == 1).sum()),
            "clips_with_2_secondary": int((sec_lens == 2).sum()),
            "clips_with_3_or_more_secondary": int((sec_lens >= 3).sum()),
            "max_secondary_labels_in_single_clip": int(sec_lens.max()),
        }

        result = {
            "status": "PASS",
            "total_classes": len(counts),
            "min_recordings_per_class": min_count,
            "max_recordings_per_class": max_count,
            "mean_recordings_per_class": round(mean_count, 2),
            "median_recordings_per_class": median_count,
            "std_recordings_per_class": round(std_count, 2),
            "imbalance_ratio": round(imbalance_ratio, 2),
            "classes_under_10_samples_count": len(under_10),
            "classes_under_20_samples_count": len(under_20),
            "classes_under_50_samples_count": len(under_50),
            "rarest_species": under_10,
            "top_10_species": counts.head(10).to_dict(),
            "multi_label_statistics": multi_label_stats,
        }
        self.check_statuses["5_class_distribution"] = result
        return result

    def check_missing_metadata(self) -> Dict[str, Any]:
        """
        Check 6: Identify missing or invalid metadata attributes.
        """
        if self.metadata_df is None:
            self.check_train_metadata()
        df = self.metadata_df

        missing_per_col = {}
        total = len(df)
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            missing_per_col[col] = {
                "missing_count": null_count,
                "missing_percentage": round((null_count / total) * 100.0, 4),
            }

        # Check specifically for latitude and longitude
        lat_missing = int(df["latitude"].isna().sum())
        lon_missing = int(df["longitude"].isna().sum())

        status = "WARN" if (lat_missing > 0 or lon_missing > 0) else "PASS"
        result = {
            "status": status,
            "total_rows": total,
            "missing_per_column": missing_per_col,
            "latitude_missing": lat_missing,
            "longitude_missing": lon_missing,
            "rating_min": float(df["rating"].min()),
            "rating_max": float(df["rating"].max()),
            "rating_mean": round(float(df["rating"].mean()), 2),
            "zero_rating_count": int((df["rating"] == 0.0).sum()),
        }
        self.check_statuses["6_missing_metadata"] = result
        return result

    def check_duplicate_filenames(self) -> Dict[str, Any]:
        """
        Check 7: Detect duplicate relative filenames and cross-species identical audio identifiers.
        """
        if self.metadata_df is None:
            self.check_train_metadata()
        df = self.metadata_df

        exact_duplicate_filenames = int(df["filename"].duplicated().sum())

        # Check duplicate base filenames (e.g. XC12345.ogg appearing in multiple species folders)
        base_names = [os.path.basename(f) for f in df["filename"]]
        from collections import Counter
        base_counts = Counter(base_names)
        shared_basenames = {k: v for k, v in base_counts.items() if v > 1}

        shared_details: List[Dict[str, Any]] = []
        if shared_basenames:
            for base, cnt in shared_basenames.items():
                matched_rows = df[df["filename"].str.endswith(base)]
                shared_details.append({
                    "basename": base,
                    "count": cnt,
                    "species_assigned": matched_rows["primary_label"].tolist(),
                    "relative_paths": matched_rows["filename"].tolist(),
                    "authors": matched_rows["author"].tolist(),
                    "urls": matched_rows["url"].tolist(),
                })

        status = "PASS" if exact_duplicate_filenames == 0 else "FAIL"
        warning = None
        if shared_basenames:
            warning = (
                f"Detected {len(shared_basenames)} audio recordings shared across multiple species directories. "
                "These share identical recording source IDs and must be grouped by recording ID during split creation "
                "to prevent cross-validation data leakage."
            )

        result = {
            "status": status,
            "exact_duplicate_filename_rows": exact_duplicate_filenames,
            "unique_basenames_count": len(base_counts),
            "shared_audio_basenames_count": len(shared_basenames),
            "shared_recordings": shared_details,
            "leakage_risk_warning": warning,
        }
        self.check_statuses["7_duplicate_filenames"] = result
        return result

    def check_physical_audio_files(self) -> Dict[str, Any]:
        """
        Check 8: Verify physical existence of all audio files referenced in train_metadata.csv.
        """
        if self.metadata_df is None:
            self.check_train_metadata()
        df = self.metadata_df

        missing_files = []
        for idx, row in df.iterrows():
            rel_path = row["filename"]
            full_path = self.train_audio_dir / rel_path
            if not full_path.is_file():
                missing_files.append({
                    "row_index": idx,
                    "filename": rel_path,
                    "expected_path": str(full_path),
                })

        status = "PASS" if len(missing_files) == 0 else "FAIL"
        result = {
            "status": status,
            "total_referenced_files": len(df),
            "missing_physical_files_count": len(missing_files),
            "missing_files_list": missing_files[:20],
        }
        self.check_statuses["8_physical_existence"] = result
        return result

    def _inspect_single_audio_file(self, row_tuple: Tuple[int, pd.Series]) -> Dict[str, Any]:
        """Worker function for single audio file inspection."""
        idx, row = row_tuple
        rel_path = row["filename"]
        primary_label = row["primary_label"]
        rating = row.get("rating", None)

        # Check missing metadata in this row
        missing_fields = []
        for col in self.EXPECTED_METADATA_COLUMNS:
            if col in row and pd.isna(row[col]):
                missing_fields.append(col)

        full_path = self.train_audio_dir / rel_path
        if not full_path.is_file():
            return {
                "filename": rel_path,
                "primary_label": primary_label,
                "file_exists": False,
                "file_size_bytes": 0,
                "is_readable": False,
                "can_decode": False,
                "sample_rate": None,
                "channels": None,
                "duration_seconds": None,
                "frames": None,
                "format": None,
                "subtype": None,
                "has_missing_metadata": len(missing_fields) > 0,
                "missing_fields": ",".join(missing_fields),
                "rating": rating,
                "quality_status": "MISSING_FILE",
                "error_message": "Audio file not found on disk",
            }

        size_bytes = full_path.stat().st_size
        if size_bytes == 0:
            return {
                "filename": rel_path,
                "primary_label": primary_label,
                "file_exists": True,
                "file_size_bytes": 0,
                "is_readable": False,
                "can_decode": False,
                "sample_rate": None,
                "channels": None,
                "duration_seconds": 0.0,
                "frames": 0,
                "format": None,
                "subtype": None,
                "has_missing_metadata": len(missing_fields) > 0,
                "missing_fields": ",".join(missing_fields),
                "rating": rating,
                "quality_status": "CORRUPTED_ZERO_BYTE",
                "error_message": "Zero-byte empty file",
            }

        try:
            info = sf.info(str(full_path))
            # Test decoding sample frames
            test_frames = min(1024, max(1, info.frames))
            data, sr = sf.read(str(full_path), frames=test_frames)

            # Check validity
            if len(data) == 0 and info.frames > 0:
                raise ValueError("Decoder returned empty audio array despite non-zero frames in header")

            if np.isnan(data).any() or np.isinf(data).any():
                raise ValueError("Decoded audio samples contain NaN or Inf values")

            if info.duration <= 0.0 or info.frames <= 0:
                return {
                    "filename": rel_path,
                    "primary_label": primary_label,
                    "file_exists": True,
                    "file_size_bytes": size_bytes,
                    "is_readable": True,
                    "can_decode": True,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "duration_seconds": float(info.duration),
                    "frames": int(info.frames),
                    "format": info.format,
                    "subtype": info.subtype,
                    "has_missing_metadata": len(missing_fields) > 0,
                    "missing_fields": ",".join(missing_fields),
                    "rating": rating,
                    "quality_status": "ZERO_DURATION",
                    "error_message": "Audio file has zero or negative duration",
                }

            status = "VALID"
            if len(missing_fields) > 0:
                status = "VALID_WITH_METADATA_GAPS"

            return {
                "filename": rel_path,
                "primary_label": primary_label,
                "file_exists": True,
                "file_size_bytes": size_bytes,
                "is_readable": True,
                "can_decode": True,
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "duration_seconds": round(float(info.duration), 4),
                "frames": int(info.frames),
                "format": str(info.format),
                "subtype": str(info.subtype),
                "has_missing_metadata": len(missing_fields) > 0,
                "missing_fields": ",".join(missing_fields),
                "rating": rating,
                "quality_status": status,
                "error_message": "",
            }

        except Exception as e:
            return {
                "filename": rel_path,
                "primary_label": primary_label,
                "file_exists": True,
                "file_size_bytes": size_bytes,
                "is_readable": False,
                "can_decode": False,
                "sample_rate": None,
                "channels": None,
                "duration_seconds": None,
                "frames": None,
                "format": None,
                "subtype": None,
                "has_missing_metadata": len(missing_fields) > 0,
                "missing_fields": ",".join(missing_fields),
                "rating": rating,
                "quality_status": "CORRUPTED_DECODE_ERROR",
                "error_message": str(e),
            }

    def check_audio_properties_and_integrity(self) -> pd.DataFrame:
        """
        Executes Checks 9, 10, 11, 12, and 13 concurrently across all audio recordings.
        
        Checks:
        9. Audio decoding
        10. Sampling-rate distribution
        11. Duration distribution
        12. Number of channels
        13. Corrupted/unreadable audio
        """
        if self.metadata_df is None:
            self.check_train_metadata()
        df = self.metadata_df

        logger.info(
            f"Auditing audio integrity for all {len(df)} files using {self.num_workers} threads..."
        )

        rows = list(df.iterrows())
        results = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for item in tqdm(
                executor.map(self._inspect_single_audio_file, rows),
                total=len(rows),
                desc="Auditing audio files",
            ):
                results.append(item)

        quality_df = pd.DataFrame(results)
        self.quality_df = quality_df

        # --- Check 9: Audio Decoding ---
        decodable_count = int(quality_df["can_decode"].sum())
        undecodable_count = int(len(quality_df) - decodable_count)
        self.check_statuses["9_audio_decoding"] = {
            "status": "PASS" if undecodable_count == 0 else "FAIL",
            "total_files": len(quality_df),
            "successfully_decoded": decodable_count,
            "failed_decoding": undecodable_count,
        }

        # --- Check 10: Sampling-Rate Distribution ---
        sr_counts = quality_df["sample_rate"].dropna().astype(int).value_counts().to_dict()
        sr_dist = {
            str(sr): {
                "count": count,
                "percentage": round((count / len(quality_df)) * 100.0, 2),
                "matches_baseline": (sr == self.baseline_sr),
            }
            for sr, count in sr_counts.items()
        }
        resample_needed_count = sum(
            c for sr, c in sr_counts.items() if sr != self.baseline_sr
        )
        self.check_statuses["10_sampling_rate_distribution"] = {
            "status": "PASS",
            "baseline_sample_rate": self.baseline_sr,
            "distribution": sr_dist,
            "files_matching_baseline_sr": sr_counts.get(self.baseline_sr, 0),
            "files_requiring_resampling": resample_needed_count,
            "resampling_percentage": round((resample_needed_count / len(quality_df)) * 100.0, 2),
        }

        # --- Check 11: Duration Distribution ---
        valid_durations = quality_df["duration_seconds"].dropna()
        dur_stats = {
            "min_seconds": float(valid_durations.min()),
            "max_seconds": float(valid_durations.max()),
            "mean_seconds": round(float(valid_durations.mean()), 2),
            "median_seconds": round(float(valid_durations.median()), 2),
            "std_seconds": round(float(valid_durations.std()), 2),
            "p05_seconds": round(float(valid_durations.quantile(0.05)), 2),
            "p25_seconds": round(float(valid_durations.quantile(0.25)), 2),
            "p50_seconds": round(float(valid_durations.quantile(0.50)), 2),
            "p75_seconds": round(float(valid_durations.quantile(0.75)), 2),
            "p95_seconds": round(float(valid_durations.quantile(0.95)), 2),
            "p99_seconds": round(float(valid_durations.quantile(0.99)), 2),
            "total_audio_hours": round(float(valid_durations.sum()) / 3600.0, 2),
        }
        under_5s = int((valid_durations < self.baseline_duration).sum())
        under_1s = int((valid_durations < 1.0).sum())
        over_60s = int((valid_durations > 60.0).sum())
        over_120s = int((valid_durations > 120.0).sum())
        over_300s = int((valid_durations > 300.0).sum())

        self.check_statuses["11_duration_distribution"] = {
            "status": "PASS",
            "baseline_window_seconds": self.baseline_duration,
            "summary_statistics": dur_stats,
            "clips_shorter_than_baseline_5s": under_5s,
            "clips_shorter_than_baseline_percentage": round((under_5s / len(quality_df)) * 100.0, 2),
            "clips_shorter_than_1s": under_1s,
            "clips_longer_than_60s": over_60s,
            "clips_longer_than_120s": over_120s,
            "clips_longer_than_300s": over_300s,
        }

        # --- Check 12: Number of Channels ---
        ch_counts = quality_df["channels"].dropna().astype(int).value_counts().to_dict()
        ch_dist = {
            str(ch): {
                "count": count,
                "percentage": round((count / len(quality_df)) * 100.0, 2),
            }
            for ch, count in ch_counts.items()
        }
        stereo_count = ch_counts.get(2, 0)
        self.check_statuses["12_number_of_channels"] = {
            "status": "PASS",
            "distribution": ch_dist,
            "mono_count": ch_counts.get(1, 0),
            "stereo_count": stereo_count,
            "downmix_required_count": stereo_count,
        }

        # --- Check 13: Corrupted/Unreadable Audio ---
        corrupted_df = quality_df[~quality_df["is_readable"] | ~quality_df["can_decode"]]
        corrupted_list = corrupted_df[["filename", "quality_status", "error_message"]].to_dict(
            orient="records"
        )
        self.check_statuses["13_corrupted_unreadable_audio"] = {
            "status": "PASS" if len(corrupted_list) == 0 else "FAIL",
            "corrupted_files_count": len(corrupted_list),
            "corrupted_files": corrupted_list,
        }

        return quality_df

    def validate_all(self) -> Dict[str, Any]:
        """
        Run the complete suite of all 13 validation checks.
        """
        logger.info("Executing comprehensive BirdCLEF 2024 dataset validation...")
        self.check_directory_structure()
        self.check_train_metadata()
        self.check_number_of_recordings()
        self.check_unique_species()
        self.check_class_distribution()
        self.check_missing_metadata()
        self.check_duplicate_filenames()
        self.check_physical_audio_files()
        self.check_audio_properties_and_integrity()

        overall_status = "PASS"
        for name, res in self.check_statuses.items():
            if res.get("status") == "FAIL":
                overall_status = "FAIL"
                break
            elif res.get("status") == "WARN" and overall_status != "FAIL":
                overall_status = "WARN"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "dataset_root": str(self.dataset_root),
            "baseline_sample_rate": self.baseline_sr,
            "baseline_clip_duration": self.baseline_duration,
            "validation_checks": self.check_statuses,
        }
        self.summary_report = summary
        return summary

    def save_summary_report(self, output_path: Optional[str | Path] = None) -> Path:
        """
        Save dataset_summary.json to reports/results/dataset_summary.json.
        """
        if not self.summary_report:
            self.validate_all()

        if output_path is None:
            target_path = self.results_dir / "dataset_summary.json"
        else:
            target_path = Path(output_path).resolve()

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.summary_report, f, indent=2)

        logger.info(f"Saved dataset summary report to: {target_path}")
        return target_path

    def save_quality_table(self, output_path: Optional[str | Path] = None) -> Path:
        """
        Save dataset_quality.csv to reports/results/dataset_quality.csv.
        """
        if self.quality_df is None:
            self.check_audio_properties_and_integrity()

        if output_path is None:
            target_path = self.results_dir / "dataset_quality.csv"
        else:
            target_path = Path(output_path).resolve()

        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.quality_df.to_csv(target_path, index=False)
        logger.info(f"Saved dataset quality table to: {target_path}")
        return target_path

    def generate_plots(self, figures_dir: Optional[str | Path] = None) -> Dict[str, Path]:
        """
        Generate research-grade publication figures:
        1. reports/figures/class_distribution.png
        2. reports/figures/duration_distribution.png
        3. reports/figures/sample_rate_distribution.png
        """
        if self.metadata_df is None or self.quality_df is None:
            self.validate_all()

        fig_dir = Path(figures_dir) if figures_dir else self.figures_dir
        fig_dir.mkdir(parents=True, exist_ok=True)

        generated_plots = {}

        # -------------------------------------------------------------
        # Figure 1: Class Distribution
        # -------------------------------------------------------------
        class_counts = self.metadata_df["primary_label"].value_counts()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        # 1a. Sorted frequency distribution across all 182 species
        ax = axes[0, 0]
        ax.plot(range(len(class_counts)), class_counts.values, color="#1f77b4", lw=2)
        ax.fill_between(range(len(class_counts)), class_counts.values, color="#1f77b4", alpha=0.2)
        ax.axhline(class_counts.median(), color="green", linestyle="--", label=f"Median ({int(class_counts.median())})")
        ax.axhline(class_counts.mean(), color="orange", linestyle="--", label=f"Mean ({class_counts.mean():.1f})")
        ax.set_title("Species Recording Frequency Curve (All 182 Classes)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Species Rank (Sorted by Abundance)", fontsize=11)
        ax.set_ylabel("Number of Audio Recordings", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True)

        # 1b. Histogram of recordings per class
        ax = axes[0, 1]
        ax.hist(class_counts.values, bins=25, color="#2ca02c", edgecolor="black", alpha=0.75)
        ax.set_title("Histogram of Class Abundance", fontsize=13, fontweight="bold")
        ax.set_xlabel("Recordings per Species", fontsize=11)
        ax.set_ylabel("Number of Species", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.text(
            0.65, 0.85,
            f"Total Species: {len(class_counts)}\nMin: {class_counts.min()}\nMax: {class_counts.max()}\nImbalance: {class_counts.max()/class_counts.min():.1f}x",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            fontsize=10,
        )

        # 1c. Top 15 most frequent species
        ax = axes[1, 0]
        top15 = class_counts.head(15)
        ax.barh(range(len(top15)), top15.values, color="#d62728", edgecolor="black", alpha=0.8)
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_title("Top 15 Most Abundant Species (Capped at 500)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Recordings", fontsize=11)
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)

        # 1d. Bottom 15 rarest species (few-shot classes)
        ax = axes[1, 1]
        bottom15 = class_counts.tail(15)
        ax.barh(range(len(bottom15)), bottom15.values, color="#9467bd", edgecolor="black", alpha=0.8)
        ax.set_yticks(range(len(bottom15)))
        ax.set_yticklabels(bottom15.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_title("Bottom 15 Rarest Species (Few-Shot Extreme)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Recordings", fontsize=11)
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)

        fig.suptitle(
            "BirdCLEF 2024 Dataset: Class Distribution & Imbalance Audit",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        class_plot_path = fig_dir / "class_distribution.png"
        fig.savefig(class_plot_path, bbox_inches="tight")
        plt.close(fig)
        generated_plots["class_distribution"] = class_plot_path
        logger.info(f"Generated class distribution plot: {class_plot_path}")

        # -------------------------------------------------------------
        # Figure 2: Duration Distribution
        # -------------------------------------------------------------
        durations = self.quality_df["duration_seconds"].dropna()
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=300, gridspec_kw={"height_ratios": [3, 2]})
        plt.subplots_adjust(hspace=0.3)

        # 2a. Duration histogram with canonical window markers (truncated at 120s for readability)
        ax = axes[0]
        clipped_durations = np.clip(durations, 0, 120)
        n, bins, patches = ax.hist(
            clipped_durations,
            bins=60,
            color="#3498db",
            edgecolor="black",
            alpha=0.75,
            density=False,
        )
        ax.axvline(
            self.baseline_duration,
            color="red",
            linestyle="-",
            lw=2.5,
            label=f"Baseline Window ({self.baseline_duration:.1f} s)",
        )
        ax.axvline(
            durations.median(),
            color="green",
            linestyle="--",
            lw=2,
            label=f"Median ({durations.median():.1f} s)",
        )
        ax.axvline(
            durations.mean(),
            color="darkorange",
            linestyle="--",
            lw=2,
            label=f"Mean ({durations.mean():.1f} s)",
        )

        short_clips_cnt = int((durations < self.baseline_duration).sum())
        short_clips_pct = (short_clips_cnt / len(durations)) * 100.0

        ax.set_title(
            "Audio Duration Distribution (Zoomed to 0-120 s)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Recording Duration (seconds)", fontsize=11)
        ax.set_ylabel("Number of Recordings", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=10)

        ax.annotate(
            f"Clips < {self.baseline_duration}s: {short_clips_cnt} ({short_clips_pct:.2f}%)\nRequire zero-padding",
            xy=(self.baseline_duration, max(n) * 0.7),
            xytext=(self.baseline_duration + 8, max(n) * 0.8),
            arrowprops=dict(facecolor="red", shrink=0.05, width=1.5, headwidth=8),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffebee", edgecolor="red", alpha=0.9),
            fontsize=10,
        )

        # 2b. Log-scale duration boxplot showing extreme long tails
        ax = axes[1]
        bp = ax.boxplot(
            durations,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="#a8dadc", color="#1d3557"),
            whiskerprops=dict(color="#1d3557", lw=1.5),
            medianprops=dict(color="#e63946", lw=2),
            flierprops=dict(marker="o", color="#457b9d", alpha=0.3, markersize=3),
        )
        ax.set_xscale("log")
        ax.set_title("Full Audio Duration Distribution (Log Scale)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Duration in Seconds (Logarithmic Scale)", fontsize=11)
        ax.set_yticks([])
        ax.grid(True, which="both", linestyle=":", alpha=0.6)

        total_hours = durations.sum() / 3600.0
        ax.text(
            0.02, 0.75,
            f"Total Duration: {total_hours:.1f} Hours ({total_hours/24:.1f} Days)\n"
            f"Min: {durations.min():.2f} s | Max: {durations.max():.2f} s ({durations.max()/60:.1f} min)\n"
            f"P25: {durations.quantile(0.25):.1f} s | P75: {durations.quantile(0.75):.1f} s | P99: {durations.quantile(0.99):.1f} s",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
            fontsize=10,
        )

        fig.suptitle(
            "BirdCLEF 2024 Dataset: Audio Duration & Temporal Windowing Audit",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        dur_plot_path = fig_dir / "duration_distribution.png"
        fig.savefig(dur_plot_path, bbox_inches="tight")
        plt.close(fig)
        generated_plots["duration_distribution"] = dur_plot_path
        logger.info(f"Generated duration distribution plot: {dur_plot_path}")

        # -------------------------------------------------------------
        # Figure 3: Sample Rate Distribution
        # -------------------------------------------------------------
        sr_data = self.quality_df["sample_rate"].dropna().astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

        sr_labels = [f"{sr} Hz" for sr in sr_data.index]
        colors = ["#2ca02c" if sr == self.baseline_sr else "#3498db" for sr in sr_data.index]

        bars = ax.bar(sr_labels, sr_data.values, color=colors, edgecolor="black", width=0.55, alpha=0.85)

        for bar, count in zip(bars, sr_data.values):
            pct = (count / len(self.quality_df)) * 100.0
            ax.annotate(
                f"{count:,}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title("Sampling Rate Distribution Across BirdCLEF 2024 Audio Recordings", fontsize=14, fontweight="bold")
        ax.set_xlabel("Audio Sampling Rate", fontsize=12)
        ax.set_ylabel("Number of Audio Recordings", fontsize=12)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

        # Highlight baseline
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ca02c", edgecolor="black", label=f"Canonical Baseline Rate ({self.baseline_sr} Hz)"),
            Patch(facecolor="#3498db", edgecolor="black", label="Requires Resampling to 32 kHz"),
        ]
        ax.legend(handles=legend_elements, frameon=True, fontsize=11, loc="upper right")

        # Annotation box
        resample_count = sum(c for sr, c in sr_data.items() if sr != self.baseline_sr)
        resample_pct = (resample_count / len(self.quality_df)) * 100.0
        ax.text(
            0.03, 0.78,
            f"Canonical Baseline: {self.baseline_sr} Hz\n"
            f"Native 32 kHz: {sr_data.get(self.baseline_sr, 0):,} ({100-resample_pct:.1f}%)\n"
            f"Requiring Resampling: {resample_count:,} ({resample_pct:.1f}%)",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9c4", edgecolor="#fbc02d", alpha=0.9),
            fontsize=11,
        )

        fig_sr_path = fig_dir / "sample_rate_distribution.png"
        fig.savefig(fig_sr_path, bbox_inches="tight")
        plt.close(fig)
        generated_plots["sample_rate_distribution"] = fig_sr_path
        logger.info(f"Generated sample rate distribution plot: {fig_sr_path}")

        return generated_plots
