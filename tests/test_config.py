"""
Unit tests for configuration loading, reproducibility settings, and project architecture.
"""

from pathlib import Path
import random
import unittest

from src.utils.config import find_default_config_path, load_config
from src.utils.seed import seed_everything


class TestProjectConfiguration(unittest.TestCase):
    """Test suite for research configuration and directory structure."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = find_default_config_path()
        cls.config = load_config(cls.config_path)
        cls.project_root = cls.config_path.parent.parent

    def test_config_file_exists(self):
        """Verify that configs/config.yaml exists and is accessible."""
        self.assertTrue(self.config_path.is_file(), f"Config file not found at {self.config_path}")

    def test_config_loading_structure(self):
        """Verify config contains all top-level required sections."""
        required_sections = ["project", "reproducibility", "data", "audio", "noise_simulation", "model", "training", "evaluation", "outputs"]
        for section in required_sections:
            self.assertIn(section, self.config, f"Missing section '{section}' in config.yaml")

    def test_dataset_root_configuration(self):
        """Verify dataset root points to C:/Datasets/birdclef-2024 using forward slashes."""
        dataset_root = self.config.data.dataset_root
        self.assertEqual(dataset_root, "C:/Datasets/birdclef-2024")
        self.assertNotIn("\\", dataset_root, "dataset_root should use forward slashes")

    def test_reproducibility_parameters(self):
        """Verify all initial research reproducibility settings match exact baseline specifications."""
        # Random seed = 42
        self.assertEqual(
            self.config.reproducibility.random_seed,
            42,
            "Baseline random seed must be 42",
        )

        # Audio parameters
        audio = self.config.audio
        self.assertEqual(audio.sample_rate, 32000, "Baseline sample rate must be 32000 Hz")
        self.assertEqual(float(audio.clip_duration), 5.0, "Baseline clip duration must be 5.0 seconds")
        self.assertEqual(audio.n_fft, 2048, "Baseline n_fft must be 2048")
        self.assertEqual(audio.hop_length, 512, "Baseline hop_length must be 512")
        self.assertEqual(audio.n_mels, 128, "Baseline n_mels must be 128")
        self.assertEqual(audio.fmin, 50, "Baseline fmin must be 50 Hz")
        self.assertEqual(audio.fmax, 16000, "Baseline fmax must be 16000 Hz")

    def test_fixed_research_question(self):
        """Verify fixed research question is documented in config."""
        expected_rq = (
            "How does realistic environmental noise affect bird species classification, "
            "and to what extent can noise-aware training improve the robustness and "
            "reliability of acoustic biodiversity monitoring?"
        )
        self.assertEqual(self.config.project.research_question, expected_rq)

    def test_seed_everything_determinism(self):
        """Verify seed_everything produces deterministic pseudo-random sequences."""
        seed_everything(42)
        seq1 = [random.random() for _ in range(5)]

        seed_everything(42)
        seq2 = [random.random() for _ in range(5)]

        self.assertEqual(seq1, seq2, "Random sequence should be identical given seed 42")

        # NumPy determinism if installed
        try:
            import numpy as np
            seed_everything(42)
            np_seq1 = np.random.rand(5).tolist()
            seed_everything(42)
            np_seq2 = np.random.rand(5).tolist()
            self.assertEqual(np_seq1, np_seq2, "NumPy random sequence should be identical given seed 42")
        except ImportError:
            pass

    def test_project_directories_exist(self):
        """Verify all mandated directories exist within the project repository."""
        required_dirs = [
            "src",
            "configs",
            "notebooks",
            "scripts",
            "experiments",
            "reports",
            "reports/figures",
            "reports/results",
            "docs",
            "tests",
            "models",
            "app",
            "data",
            "data/raw",
            "data/processed",
            "data/manifests",
        ]
        for rel_dir in required_dirs:
            target_dir = self.project_root / rel_dir
            self.assertTrue(
                target_dir.is_dir(),
                f"Required directory '{rel_dir}' does not exist at {target_dir}",
            )


if __name__ == "__main__":
    unittest.main()
