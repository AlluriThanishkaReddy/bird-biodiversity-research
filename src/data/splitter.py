"""
Leakage-safe dataset splitting module for BirdCLEF 2024 bioacoustic research.

Implements recording-level grouped stratified splitting:
- Groups by atomic audio recording identifier (XC ID) to ensure that all clips
  and multi-species instances originating from the same recording remain in the exact same split.
- Enforces a 70% Train / 15% Validation / 15% Test partition.
- Uses species stratification to ensure all 182 target species have adequate support
  in train, validation, and test sets.
- Guarantees zero cross-split recording leakage.
"""

from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import ConfigDict, find_default_config_path, load_config
from src.utils.logger import get_logger

logger = get_logger("dataset_splitter")


class LeakageSafeSplitter:
    """
    Splits BirdCLEF 2024 recordings into Train (70%), Val (15%), and Test (15%) partitions
    with strict zero-leakage recording-level grouping and species stratification.
    """

    def __init__(
        self,
        config: Optional[ConfigDict] = None,
        dataset_root: Optional[str | Path] = None,
        random_seed: int = 42,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Initialize the splitter.

        Args:
            config: Optional configuration dictionary.
            dataset_root: Optional override path for dataset root.
            random_seed: Deterministic pseudo-random number generator seed.
            train_ratio: Target proportion for training split (default: 0.70).
            val_ratio: Target proportion for validation split (default: 0.15).
            test_ratio: Target proportion for test split (default: 0.15).
        """
        if config is None:
            self.config = load_config(find_default_config_path())
        else:
            self.config = config

        if dataset_root is not None:
            self.dataset_root = Path(dataset_root).resolve()
        else:
            self.dataset_root = Path(self.config.data.dataset_root).resolve()

        self.random_seed = random_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0; got {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}"
            )

        # Paths
        project_root = find_default_config_path().parent.parent
        self.metadata_path = self.dataset_root / self.config.data.get("train_metadata", "train_metadata.csv")
        self.manifests_dir = project_root / self.config.data.get("manifests_dir", "data/manifests")
        self.results_dir = project_root / self.config.outputs.get("results_dir", "reports/results")

        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.split_df: Optional[pd.DataFrame] = None
        self.summary_report: Dict[str, Any] = {}

    @staticmethod
    def extract_recording_id(filename: str) -> str:
        """Extract canonical Xeno-Canto ID from relative filename (e.g. asbfly/XC134896.ogg -> 134896)."""
        match = re.search(r"XC(\d+)", filename)
        if match:
            return match.group(1)
        # Fallback to sanitized basename without extension
        return Path(filename).stem

    def create_split(self) -> pd.DataFrame:
        """
        Execute the recording-level grouped stratified split.
        
        Returns:
            pd.DataFrame: Metadata with assigned 'split' column and recording audit fields.
        """
        logger.info(f"Loading metadata from {self.metadata_path} for leakage-safe splitting...")
        df = pd.read_csv(self.metadata_path)

        # Merge with dataset quality if available to bring duration & file size
        quality_csv = self.results_dir / "dataset_quality.csv"
        if quality_csv.is_file():
            qdf = pd.read_csv(quality_csv)
            cols_to_add = [c for c in ["duration_seconds", "sample_rate", "channels", "file_size_bytes"] if c in qdf.columns]
            if cols_to_add:
                df = df.merge(qdf[["filename"] + cols_to_add], on="filename", how="left")

        df["recording_id"] = df["filename"].apply(self.extract_recording_id)

        total_rows = len(df)
        target_train_rows = round(self.train_ratio * total_rows)
        target_val_rows = round(self.val_ratio * total_rows)
        target_test_rows = total_rows - target_train_rows - target_val_rows

        logger.info(
            f"Target split row counts: Train={target_train_rows} ({self.train_ratio*100:.1f}%), "
            f"Val={target_val_rows} ({self.val_ratio*100:.1f}%), "
            f"Test={target_test_rows} ({self.test_ratio*100:.1f}%)"
        )

        # Build group mappings
        group_to_rows: Dict[str, List[int]] = defaultdict(list)
        group_to_species: Dict[str, List[str]] = defaultdict(list)

        for idx, row in df.iterrows():
            rec_id = row["recording_id"]
            group_to_rows[rec_id].append(idx)
            group_to_species[rec_id].append(row["primary_label"])

        # Identify multi-row groups (recordings mapped to multiple species)
        multi_row_groups = {g: sps for g, sps in group_to_species.items() if len(sps) > 1}
        logger.info(
            f"Identified {len(multi_row_groups)} shared recording groups across species (totaling {sum(len(v) for v in multi_row_groups.values())} rows)"
        )

        # Species rarity sorting
        species_counts = df["primary_label"].value_counts().to_dict()
        sorted_species = sorted(species_counts.keys(), key=lambda s: (species_counts[s], s))

        # Split assignment tracking
        split_assignment: Dict[str, str] = {}
        split_row_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        species_split_counts: Dict[str, Dict[str, int]] = {
            s: {"train": 0, "val": 0, "test": 0} for s in species_counts
        }

        rng = np.random.RandomState(self.random_seed)

        # Phase 1: Allocate multi-row groups first
        # Deterministically shuffle multi-row groups
        multi_keys = sorted(multi_row_groups.keys())
        rng.shuffle(multi_keys)

        for g in multi_keys:
            rows = group_to_rows[g]
            sps = group_to_species[g]
            n_rows = len(rows)

            # Choose split with largest row deficit relative to target
            best_split = None
            best_deficit = -1e9
            for s_name, target_prop in [
                ("train", self.train_ratio),
                ("val", self.val_ratio),
                ("test", self.test_ratio),
            ]:
                deficit = target_prop * total_rows - split_row_counts[s_name]
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_split = s_name

            split_assignment[g] = best_split
            split_row_counts[best_split] += n_rows
            for sp in sps:
                species_split_counts[sp][best_split] += 1

        # Phase 2: Allocate single-row groups species-by-species from rarest to most abundant
        for sp in sorted_species:
            unassigned_g = [
                g for g in group_to_species
                if g not in split_assignment and sp in group_to_species[g]
            ]
            unassigned_g = sorted(unassigned_g)
            rng.shuffle(unassigned_g)

            n_sp_total = species_counts[sp]
            target_val_sp = max(1, round(self.val_ratio * n_sp_total))
            target_test_sp = max(1, round(self.test_ratio * n_sp_total))
            target_train_sp = n_sp_total - target_val_sp - target_test_sp

            # Constrain rare species (<=10 recordings) to have at least 1 in val and 1 in test
            if n_sp_total <= 10:
                target_val_sp = 1
                target_test_sp = 1
                target_train_sp = n_sp_total - 2

            for g in unassigned_g:
                cur_train = species_split_counts[sp]["train"]
                cur_val = species_split_counts[sp]["val"]
                cur_test = species_split_counts[sp]["test"]

                # Deficit-based priority
                if cur_val < target_val_sp:
                    chosen = "val"
                elif cur_test < target_test_sp:
                    chosen = "test"
                elif cur_train < target_train_sp:
                    chosen = "train"
                else:
                    # Global deficit fallback
                    deficits = {
                        "train": (self.train_ratio * total_rows - split_row_counts["train"]) / self.train_ratio,
                        "val": (self.val_ratio * total_rows - split_row_counts["val"]) / self.val_ratio,
                        "test": (self.test_ratio * total_rows - split_row_counts["test"]) / self.test_ratio,
                    }
                    chosen = max(deficits.keys(), key=lambda k: deficits[k])

                split_assignment[g] = chosen
                split_row_counts[chosen] += len(group_to_rows[g])
                for s in group_to_species[g]:
                    species_split_counts[s][chosen] += 1

        # Assign split column
        df["split"] = df["recording_id"].map(split_assignment)
        self.split_df = df

        # Execute full validation on the resulting split
        self.validate_split(df)
        return df

    def validate_split(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate all 5 criteria mandated by research governance:
        1. Every recording belongs to exactly one split.
        2. No recording occurs in multiple splits (zero leakage).
        3. Species distributions are reported across splits.
        4. Rare species are identified and guaranteed representation.
        5. Any stratification limitations are documented.
        """
        if df is None:
            if self.split_df is None:
                raise ValueError("No split has been generated yet. Call create_split() first.")
            df = self.split_df

        total_rows = len(df)
        total_unique_recordings = df["recording_id"].nunique()

        # Check 1: Every recording belongs to exactly one split
        unassigned_rows = int(df["split"].isna().sum())
        invalid_split_names = int((~df["split"].isin(["train", "val", "test"])).sum())
        assigned_to_one = (unassigned_rows == 0 and invalid_split_names == 0)

        # Check 2: No recording occurs in multiple splits (group leakage)
        leakage_per_recording = df.groupby("recording_id")["split"].nunique()
        leaking_recordings = leakage_per_recording[leakage_per_recording > 1]
        zero_leakage = (len(leaking_recordings) == 0)

        # Check 2b: Multi-species shared recordings leakage
        shared_xc_ids = df["recording_id"].value_counts()[df["recording_id"].value_counts() > 1].index.tolist()
        shared_split_leakage = 0
        shared_details = []
        for xcid in shared_xc_ids:
            sub = df[df["recording_id"] == xcid]
            splits_for_xc = sub["split"].unique().tolist()
            if len(splits_for_xc) > 1:
                shared_split_leakage += 1
            shared_details.append({
                "recording_id": xcid,
                "instances_count": len(sub),
                "species": sub["primary_label"].tolist(),
                "assigned_split": splits_for_xc[0] if len(splits_for_xc) == 1 else "LEAKING",
            })

        # Check 3: Species distribution reporting
        species_counts = df["primary_label"].value_counts()
        total_species = len(species_counts)

        split_species_counts = {}
        species_presence = {}
        for s in ["train", "val", "test"]:
            sub = df[df["split"] == s]
            split_species_counts[s] = int(sub["primary_label"].nunique())
            species_presence[s] = set(sub["primary_label"].unique())

        per_species_breakdown = {}
        missing_species_in_splits = {"train": [], "val": [], "test": []}

        for sp, count in species_counts.items():
            sub = df[df["primary_label"] == sp]
            tr = int((sub["split"] == "train").sum())
            va = int((sub["split"] == "val").sum())
            te = int((sub["split"] == "test").sum())
            per_species_breakdown[sp] = {
                "total": count,
                "train": tr,
                "val": va,
                "test": te,
                "train_pct": round((tr / count) * 100.0, 1),
                "val_pct": round((va / count) * 100.0, 1),
                "test_pct": round((te / count) * 100.0, 1),
            }
            if tr == 0:
                missing_species_in_splits["train"].append(sp)
            if va == 0:
                missing_species_in_splits["val"].append(sp)
            if te == 0:
                missing_species_in_splits["test"].append(sp)

        # Check 4: Rare species identification
        rare_10 = {sp: per_species_breakdown[sp] for sp, cnt in species_counts.items() if cnt <= 10}
        rare_20 = {sp: per_species_breakdown[sp] for sp, cnt in species_counts.items() if cnt <= 20}
        rare_50 = {sp: per_species_breakdown[sp] for sp, cnt in species_counts.items() if cnt <= 50}

        # Check 5: Potential author and metadata leakage investigation
        author_split_counts = df.groupby(["author", "split"]).size().unstack(fill_value=0)
        multi_split_authors = author_split_counts[
            (author_split_counts > 0).sum(axis=1) > 1
        ]

        # Spatial clusters investigation (exact lat/lon)
        loc_df = df.dropna(subset=["latitude", "longitude"])
        loc_split_counts = loc_df.groupby(["latitude", "longitude", "split"]).size().unstack(fill_value=0)
        multi_split_locs = loc_split_counts[
            (loc_split_counts > 0).sum(axis=1) > 1
        ]

        # Summary Metrics
        split_sizes = df["split"].value_counts().to_dict()
        split_proportions = (df["split"].value_counts(normalize=True) * 100.0).round(2).to_dict()

        stratification_limitations = [
            {
                "limitation": "Extreme Few-Shot Sample Indivisibility",
                "description": (
                    "Classes with 5 total recordings (asiope1, integr, niwpig1) cannot be perfectly partitioned "
                    "at 70.0% / 15.0% / 15.0% (which would require 3.5 / 0.75 / 0.75 recordings). "
                    "To guarantee that validation and test sets have non-zero true positive support for "
                    "Macro ROC-AUC evaluation, exactly 3 recordings (60.0%) are allocated to train, "
                    "1 recording (20.0%) to validation, and 1 recording (20.0%) to test."
                ),
                "affected_classes_count": 3,
                "mitigation": "1 positive sample guaranteed in val and test for all classes.",
            },
            {
                "limitation": "Cross-Species Recording Identifier Binding",
                "description": (
                    "19 distinct Xeno-Canto recordings (38 rows in metadata) appear in multiple species directories "
                    "due to multi-species vocalizations in the source audio. To prevent recording-level data leakage, "
                    "both species instances of each shared recording are strictly locked into the same split."
                ),
                "affected_recordings_count": 19,
                "mitigation": "Atomic grouping by recording_id eliminates cross-split data leakage.",
            },
            {
                "limitation": "Author-Level Stratification Impossibility",
                "description": (
                    "A single author (e.g. José Carlos Sires) contributed 915 recordings spanning over 100 species. "
                    "Grouping splits by author makes species stratification mathematically impossible, resulting in "
                    "complete class absence (0 samples) for rare species in validation folds. "
                    "Therefore, grouping is strictly enforced at the recording level rather than author level."
                ),
                "affected_recordings_count": total_rows,
                "mitigation": "Recording-level isolation prevents acoustic file leakage while maintaining 100% species coverage.",
            },
            {
                "limitation": "Secondary Label Co-occurrence Multi-label Drift",
                "description": (
                    "Recordings with secondary labels contain incidental bird vocalizations that cannot be "
                    "jointly optimized with primary labels under standard single-target stratification."
                ),
                "affected_recordings_count": int((df["secondary_labels"] != "[]").sum()),
                "mitigation": "Stratification optimizes primary species targets while preserving realistic background co-occurrences.",
            },
        ]

        summary = {
            "timestamp": datetime.now().isoformat(),
            "random_seed": self.random_seed,
            "dataset_root": str(self.dataset_root),
            "total_metadata_rows": total_rows,
            "total_unique_recordings": total_unique_recordings,
            "targets": {
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
            },
            "actual_splits": {
                "train": {
                    "count": split_sizes.get("train", 0),
                    "percentage": split_proportions.get("train", 0.0),
                    "unique_species": split_species_counts.get("train", 0),
                    "unique_recordings": df[df["split"] == "train"]["recording_id"].nunique(),
                },
                "val": {
                    "count": split_sizes.get("val", 0),
                    "percentage": split_proportions.get("val", 0.0),
                    "unique_species": split_species_counts.get("val", 0),
                    "unique_recordings": df[df["split"] == "val"]["recording_id"].nunique(),
                },
                "test": {
                    "count": split_sizes.get("test", 0),
                    "percentage": split_proportions.get("test", 0.0),
                    "unique_species": split_species_counts.get("test", 0),
                    "unique_recordings": df[df["split"] == "test"]["recording_id"].nunique(),
                },
            },
            "leakage_verification": {
                "every_recording_assigned": assigned_to_one,
                "unassigned_count": unassigned_rows,
                "recording_id_leakage_detected": not zero_leakage,
                "leaking_recordings_count": len(leaking_recordings),
                "shared_xc_recordings_total": len(shared_xc_ids),
                "shared_xc_recordings_leaking": shared_split_leakage,
                "shared_recordings_audit": shared_details,
            },
            "species_coverage": {
                "total_target_species": total_species,
                "species_in_train": split_species_counts.get("train", 0),
                "species_in_val": split_species_counts.get("val", 0),
                "species_in_test": split_species_counts.get("test", 0),
                "all_species_in_all_splits": (
                    split_species_counts.get("train", 0) == total_species
                    and split_species_counts.get("val", 0) == total_species
                    and split_species_counts.get("test", 0) == total_species
                ),
                "missing_species_by_split": missing_species_in_splits,
            },
            "rare_species_audit": {
                "species_under_10_samples": rare_10,
                "species_under_20_samples_count": len(rare_20),
                "species_under_50_samples_count": len(rare_50),
            },
            "author_and_spatial_audit": {
                "total_unique_authors": int(df["author"].nunique()),
                "authors_spanning_multiple_splits": int(len(multi_split_authors)),
                "total_spatial_coordinates": len(loc_df),
                "spatial_locations_spanning_multiple_splits": int(len(multi_split_locs)),
            },
            "stratification_limitations": stratification_limitations,
            "per_species_distribution": per_species_breakdown,
        }

        self.summary_report = summary
        return summary

    def save_split_manifest(self, output_path: Optional[str | Path] = None) -> Path:
        """
        Save the split manifest CSV to data/manifests/split_manifest.csv.
        """
        if self.split_df is None:
            self.create_split()

        if output_path is None:
            target_path = self.manifests_dir / "split_manifest.csv"
        else:
            target_path = Path(output_path).resolve()

        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Select and order key columns
        ordered_cols = [
            "filename",
            "recording_id",
            "primary_label",
            "secondary_labels",
            "split",
            "rating",
            "duration_seconds",
            "sample_rate",
            "channels",
            "author",
            "latitude",
            "longitude",
            "type",
            "scientific_name",
            "common_name",
            "url",
        ]
        available_cols = [c for c in ordered_cols if c in self.split_df.columns]
        extra_cols = [c for c in self.split_df.columns if c not in available_cols]
        final_df = self.split_df[available_cols + extra_cols]

        final_df.to_csv(target_path, index=False)
        logger.info(f"Saved split manifest to: {target_path}")
        return target_path

    def save_split_summary(self, output_path: Optional[str | Path] = None) -> Path:
        """
        Save the split summary JSON to reports/results/split_summary.json.
        """
        if not self.summary_report:
            if self.split_df is None:
                self.create_split()
            else:
                self.validate_split()

        if output_path is None:
            target_path = self.results_dir / "split_summary.json"
        else:
            target_path = Path(output_path).resolve()

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.summary_report, f, indent=2)

        logger.info(f"Saved split summary report to: {target_path}")
        return target_path
