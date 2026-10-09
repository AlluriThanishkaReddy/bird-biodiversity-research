"""
PyTorch Dataset for BirdCLEF clean-audio CNN experiments.

The dataset uses the existing leakage-safe split_manifest.csv.
No new random train/validation/test split is created here.
"""

from pathlib import Path

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.audio.processor import AudioProcessor, AudioConfig


class BirdCLEFDataset(Dataset):
    """
    BirdCLEF dataset producing log-Mel spectrogram tensors.

    Each item contains:
        spectrogram: Tensor [1, n_mels, time_frames]
        label: integer species index
    """

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        split: str,
        label_to_index: Dict[str, int],
        audio_config: AudioConfig | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.label_to_index = label_to_index

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        if self.split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be one of: train, val, test"
            )

        self.manifest = pd.read_csv(
            self.manifest_path
        )

        required_columns = {
            "filename",
            "primary_label",
            "split",
        }

        missing_columns = (
            required_columns
            - set(self.manifest.columns)
        )

        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        self.manifest = self.manifest[
            self.manifest["split"] == self.split
        ].reset_index(drop=True)

        if len(self.manifest) == 0:
            raise ValueError(
                f"No recordings found for split '{self.split}'"
            )

        self.audio_processor = AudioProcessor(
            audio_config
        )

    def __len__(self) -> int:
        """Return number of recordings in this split."""
        return len(self.manifest)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load one recording and convert it to a CNN-ready tensor.
        """

        row = self.manifest.iloc[index]

        relative_path = Path(
            str(row["filename"])
        )

        audio_path = (
            self.dataset_root
            / "train_audio"
            / relative_path
        )

        _, log_mel = self.audio_processor.process_file(
            audio_path
        )

        # Add channel dimension:
        # [mel, time] -> [1, mel, time]
        spectrogram = np.expand_dims(
            log_mel,
            axis=0,
        )

        spectrogram = torch.from_numpy(
            spectrogram
        ).float()

        label_name = str(
            row["primary_label"]
        )

        if label_name not in self.label_to_index:
            raise KeyError(
                f"Unknown species label: {label_name}"
            )

        label = torch.tensor(
            self.label_to_index[label_name],
            dtype=torch.long,
        )

        return spectrogram, label
