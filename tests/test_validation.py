"""
Unit tests for the BirdCLEF 2024 dataset validation module.
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import soundfile as sf

from src.data.validator import DatasetValidator
from src.utils.config import ConfigDict, find_default_config_path, load_config


class TestDatasetValidator(unittest.TestCase):
    """Test suite for DatasetValidator logic, mock data auditing, and configuration loading."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = find_default_config_path()
        cls.config = load_config(cls.config_path)

    def setUp(self):
        # Create a lightweight temporary synthetic dataset directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_dataset_root = Path(self.temp_dir.name)

        # Create expected directory structure
        self.mock_train_audio = self.mock_dataset_root / "train_audio"
        self.mock_train_audio.mkdir(parents=True)
        (self.mock_dataset_root / "test_soundscapes").mkdir(parents=True)

        # Create dummy taxonomy and sample submission
        (self.mock_dataset_root / "eBird_Taxonomy_v2021.csv").write_text("species_code,scientific_name\n", encoding="utf-8")
        (self.mock_dataset_root / "sample_submission.csv").write_text("row_id,species_a,species_b\n1,0.5,0.5\n", encoding="utf-8")

        # Create two species subdirectories
        (self.mock_train_audio / "species_a").mkdir()
        (self.mock_train_audio / "species_b").mkdir()

        # Create 3 valid small audio files (OGG format)
        self.sr = 32000
        # 1-second audio
        audio1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, self.sr)).astype(np.float32)
        sf.write(str(self.mock_train_audio / "species_a" / "XC001.ogg"), audio1, self.sr, format="OGG")

        # 6-second audio (stereo)
        audio2 = np.column_stack([
            np.sin(2 * np.pi * 880 * np.linspace(0, 6, 6 * self.sr)),
            np.sin(2 * np.pi * 440 * np.linspace(0, 6, 6 * self.sr)),
        ]).astype(np.float32)
        sf.write(str(self.mock_train_audio / "species_a" / "XC002.ogg"), audio2, 44100, format="OGG")

        # 4-second audio in species_b
        audio3 = np.sin(2 * np.pi * 600 * np.linspace(0, 4, 4 * self.sr)).astype(np.float32)
        sf.write(str(self.mock_train_audio / "species_b" / "XC003.ogg"), audio3, self.sr, format="OGG")

        # Create metadata DataFrame
        self.metadata = pd.DataFrame([
            {
                "primary_label": "species_a",
                "secondary_labels": "[]",
                "type": "song",
                "latitude": 10.5,
                "longitude": 76.5,
                "scientific_name": "Species A",
                "common_name": "Common A",
                "author": "Author 1",
                "license": "CC",
                "rating": 4.5,
                "url": "https://xeno-canto.org/1",
                "filename": "species_a/XC001.ogg",
            },
            {
                "primary_label": "species_a",
                "secondary_labels": "['species_b']",
                "type": "call",
                "latitude": np.nan,  # intentionally missing latitude
                "longitude": np.nan,
                "scientific_name": "Species A",
                "common_name": "Common A",
                "author": "Author 2",
                "license": "CC",
                "rating": 3.0,
                "url": "https://xeno-canto.org/2",
                "filename": "species_a/XC002.ogg",
            },
            {
                "primary_label": "species_b",
                "secondary_labels": "[]",
                "type": "song",
                "latitude": 11.0,
                "longitude": 77.0,
                "scientific_name": "Species B",
                "common_name": "Common B",
                "author": "Author 3",
                "license": "CC",
                "rating": 5.0,
                "url": "https://xeno-canto.org/3",
                "filename": "species_b/XC003.ogg",
            },
        ])
        self.metadata.to_csv(self.mock_dataset_root / "train_metadata.csv", index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_directory_structure_check(self):
        """Test directory structure verification on mock dataset."""
        validator = DatasetValidator(config=self.config, dataset_root=self.mock_dataset_root, num_workers=2)
        res = validator.check_directory_structure()
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["species_folders_count"], 2)

    def test_train_metadata_check(self):
        """Test metadata schema verification and row reading."""
        validator = DatasetValidator(config=self.config, dataset_root=self.mock_dataset_root, num_workers=2)
        df = validator.check_train_metadata()
        self.assertEqual(len(df), 3)
        self.assertIn("primary_label", df.columns)
        self.assertEqual(validator.check_statuses["2_train_metadata"]["status"], "PASS")

    def test_missing_metadata_detection(self):
        """Test detection of missing latitude/longitude."""
        validator = DatasetValidator(config=self.config, dataset_root=self.mock_dataset_root, num_workers=2)
        res = validator.check_missing_metadata()
        self.assertEqual(res["status"], "WARN")
        self.assertEqual(res["latitude_missing"], 1)
        self.assertEqual(res["longitude_missing"], 1)

    def test_duplicate_basename_detection(self):
        """Test duplicate audio basename detection when same audio is mapped to multiple classes."""
        # Add a duplicate basename row referencing species_b with same basename
        dup_row = pd.DataFrame([{
            "primary_label": "species_b",
            "secondary_labels": "[]",
            "type": "song",
            "latitude": 11.2,
            "longitude": 77.2,
            "scientific_name": "Species B",
            "common_name": "Common B",
            "author": "Author 1",
            "license": "CC",
            "rating": 4.5,
            "url": "https://xeno-canto.org/1",
            "filename": "species_b/XC001.ogg",
        }])
        augmented_meta = pd.concat([self.metadata, dup_row], ignore_index=True)
        augmented_meta.to_csv(self.mock_dataset_root / "train_metadata.csv", index=False)

        validator = DatasetValidator(config=self.config, dataset_root=self.mock_dataset_root, num_workers=2)
        validator.check_train_metadata()
        res = validator.check_duplicate_filenames()
        self.assertEqual(res["shared_audio_basenames_count"], 1)
        self.assertIsNotNone(res["leakage_risk_warning"])

    def test_audio_properties_and_integrity(self):
        """Test audio properties extraction: decoding, duration, sample rate, and channels."""
        validator = DatasetValidator(config=self.config, dataset_root=self.mock_dataset_root, num_workers=2)
        validator.check_train_metadata()
        qdf = validator.check_audio_properties_and_integrity()

        self.assertEqual(len(qdf), 3)
        self.assertTrue(all(qdf["can_decode"]))
        self.assertTrue(all(qdf["is_readable"]))

        # Channels check: 2 mono (1), 1 stereo (2)
        ch_check = validator.check_statuses["12_number_of_channels"]
        self.assertEqual(ch_check["mono_count"], 2)
        self.assertEqual(ch_check["stereo_count"], 1)

        # Sampling rate check: 2 at 32000, 1 at 44100
        sr_check = validator.check_statuses["10_sampling_rate_distribution"]
        self.assertEqual(sr_check["files_matching_baseline_sr"], 2)
        self.assertEqual(sr_check["files_requiring_resampling"], 1)

    def test_corrupted_file_detection(self):
        """Test that zero-byte or corrupt files are properly caught without crashing."""
        # Create a zero-byte file
        corrupt_path = self.mock_train_audio / "species_a" / "XC_corrupt.ogg"
        corrupt_path.touch()

        corrupt_row = pd.DataFrame([{
            "primary_label": "species_a",
            "secondary_labels": "[]",
            "type": "call",
            "latitude": 10.0,
            "longitude": 76.0,
            "scientific_name": "Species A",
            "common_name": "Common A",
            "author": "Author Corrupt",
            "license": "CC",
            "rating": 1.0,
            "url": "https://xeno-canto.org/999",
            "filename": "species_a/XC_corrupt.ogg",
        }])
        augmented_meta = pd.concat([self.metadata, corrupt_row], ignore_index=True)
        augmented_meta.to_csv(self.mock_dataset_root / "train_metadata.csv", index=False)

        validator = DatasetValidator(config=self.config, dataset_root=self.mock_dataset_root, num_workers=2)
        validator.check_train_metadata()
        qdf = validator.check_audio_properties_and_integrity()

        corrupt_entries = qdf[~qdf["is_readable"] | ~qdf["can_decode"]]
        self.assertEqual(len(corrupt_entries), 1)
        self.assertEqual(corrupt_entries.iloc[0]["quality_status"], "CORRUPTED_ZERO_BYTE")


if __name__ == "__main__":
    unittest.main()
