"""
Unit tests for the leakage-safe dataset splitting module.
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.data.splitter import LeakageSafeSplitter
from src.utils.config import find_default_config_path, load_config


class TestLeakageSafeSplitter(unittest.TestCase):
    """Test suite for LeakageSafeSplitter logic, zero-leakage constraints, and stratification."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = find_default_config_path()
        cls.config = load_config(cls.config_path)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_dataset_root = Path(self.temp_dir.name)

        # Create synthetic metadata with:
        # - 3 species: species_a (20 samples), species_b (10 samples), species_c (5 samples)
        # - Shared XC ID between species_a and species_b (multi-species recording)
        rows = []
        # species_a (20 samples: XC100 to XC119)
        for i in range(100, 120):
            rows.append({
                "primary_label": "species_a",
                "secondary_labels": "[]",
                "type": "song",
                "latitude": 10.0,
                "longitude": 76.0,
                "scientific_name": "Species A",
                "common_name": "Common A",
                "author": f"Author_{i%3}",
                "license": "CC",
                "rating": 4.0,
                "url": f"https://xeno-canto.org/{i}",
                "filename": f"species_a/XC{i}.ogg",
            })

        # species_b (10 samples: XC200 to XC208, and duplicate XC100 from species_a!)
        rows.append({
            "primary_label": "species_b",
            "secondary_labels": "[]",
            "type": "call",
            "latitude": 10.0,
            "longitude": 76.0,
            "scientific_name": "Species B",
            "common_name": "Common B",
            "author": "Author_0",
            "license": "CC",
            "rating": 4.0,
            "url": "https://xeno-canto.org/100",
            "filename": "species_b/XC100.ogg",  # duplicate XC ID!
        })
        for i in range(200, 209):
            rows.append({
                "primary_label": "species_b",
                "secondary_labels": "[]",
                "type": "song",
                "latitude": 11.0,
                "longitude": 77.0,
                "scientific_name": "Species B",
                "common_name": "Common B",
                "author": f"Author_{i%4}",
                "license": "CC",
                "rating": 3.5,
                "url": f"https://xeno-canto.org/{i}",
                "filename": f"species_b/XC{i}.ogg",
            })

        # species_c (5 samples: XC300 to XC304, extreme few-shot)
        for i in range(300, 305):
            rows.append({
                "primary_label": "species_c",
                "secondary_labels": "[]",
                "type": "call",
                "latitude": 12.0,
                "longitude": 78.0,
                "scientific_name": "Species C",
                "common_name": "Common C",
                "author": f"Author_{i}",
                "license": "CC",
                "rating": 5.0,
                "url": f"https://xeno-canto.org/{i}",
                "filename": f"species_c/XC{i}.ogg",
            })

        self.mock_meta_path = self.mock_dataset_root / "train_metadata.csv"
        pd.DataFrame(rows).to_csv(self.mock_meta_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_recording_id(self):
        """Test extraction of XC ID from various file path formats."""
        self.assertEqual(LeakageSafeSplitter.extract_recording_id("asbfly/XC134896.ogg"), "134896")
        self.assertEqual(LeakageSafeSplitter.extract_recording_id("subfolder/deep/XC9999.ogg"), "9999")
        self.assertEqual(LeakageSafeSplitter.extract_recording_id("test_clip.ogg"), "test_clip")

    def test_zero_leakage_and_shared_xc_handling(self):
        """Test that shared XC ID XC100 is allocated to the EXACT same split across both species."""
        splitter = LeakageSafeSplitter(
            config=self.config,
            dataset_root=self.mock_dataset_root,
            random_seed=42,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        df = splitter.create_split()

        # Verify XC100 instances
        xc100_rows = df[df["recording_id"] == "100"]
        self.assertEqual(len(xc100_rows), 2)
        self.assertEqual(
            xc100_rows["split"].nunique(), 1,
            "All instances of XC100 must be in the exact same split!"
        )

        # Check total leakage
        summary = splitter.summary_report
        self.assertFalse(summary["leakage_verification"]["recording_id_leakage_detected"])
        self.assertEqual(summary["leakage_verification"]["leaking_recordings_count"], 0)

    def test_rare_species_coverage(self):
        """Test that extreme few-shot class species_c (5 samples) has representation in train, val, and test."""
        splitter = LeakageSafeSplitter(
            config=self.config,
            dataset_root=self.mock_dataset_root,
            random_seed=42,
        )
        df = splitter.create_split()

        sp_c_splits = df[df["primary_label"] == "species_c"]["split"].value_counts().to_dict()
        self.assertGreaterEqual(sp_c_splits.get("train", 0), 1)
        self.assertGreaterEqual(sp_c_splits.get("val", 0), 1)
        self.assertGreaterEqual(sp_c_splits.get("test", 0), 1)

    def test_all_recordings_assigned_to_exactly_one_split(self):
        """Test that every recording belongs to exactly one split."""
        splitter = LeakageSafeSplitter(
            config=self.config,
            dataset_root=self.mock_dataset_root,
            random_seed=42,
        )
        df = splitter.create_split()

        self.assertEqual(df["split"].isna().sum(), 0)
        self.assertTrue(df["split"].isin(["train", "val", "test"]).all())
        self.assertEqual(len(df), 35)

    def test_manifest_and_summary_saving(self):
        """Test saving manifest and summary files to temporary paths."""
        splitter = LeakageSafeSplitter(
            config=self.config,
            dataset_root=self.mock_dataset_root,
            random_seed=42,
        )
        splitter.create_split()

        out_manifest = self.mock_dataset_root / "test_split_manifest.csv"
        out_summary = self.mock_dataset_root / "test_split_summary.json"

        p_man = splitter.save_split_manifest(out_manifest)
        p_sum = splitter.save_split_summary(out_summary)

        self.assertTrue(p_man.is_file())
        self.assertTrue(p_sum.is_file())
        self.assertGreater(p_man.stat().st_size, 0)
        self.assertGreater(p_sum.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
