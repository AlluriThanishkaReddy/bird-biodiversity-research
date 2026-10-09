"""
Label mapping and DataLoader construction for BirdCLEF.

The label mapping is created once from the complete manifest so that
train, validation, and test use exactly the same species-to-index mapping.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.audio.processor import AudioConfig
from src.datasets.bird_dataset import BirdCLEFDataset


def build_label_mapping(
    manifest_path: str | Path,
) -> Dict[str, int]:
    """
    Build a deterministic species-to-index mapping.

    Species names are sorted alphabetically so the mapping is reproducible.
    """

    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)

    if "primary_label" not in manifest.columns:
        raise ValueError(
            "Manifest must contain 'primary_label'."
        )

    species = sorted(
        manifest["primary_label"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(species) == 0:
        raise ValueError(
            "No species labels found in manifest."
        )

    return {
        label: index
        for index, label in enumerate(species)
    }


def build_dataloaders(
    manifest_path: str | Path,
    dataset_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    audio_config: AudioConfig | None = None,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    Dict[str, int],
]:
    """
    Build train, validation, and test DataLoaders.
    """

    label_to_index = build_label_mapping(
        manifest_path
    )

    train_dataset = BirdCLEFDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split="train",
        label_to_index=label_to_index,
        audio_config=audio_config,
    )

    val_dataset = BirdCLEFDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split="val",
        label_to_index=label_to_index,
        audio_config=audio_config,
    )

    test_dataset = BirdCLEFDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split="test",
        label_to_index=label_to_index,
        audio_config=audio_config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        label_to_index,
    )
