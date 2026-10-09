from pathlib import Path

import torch

from src.audio.processor import AudioConfig
from src.datasets.bird_dataset import BirdCLEFDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "split_manifest.csv"
)
DATASET_ROOT = Path(
    "C:/Datasets/birdclef-2024"
)


def test_dataset_can_read_train_split():
    labels = ["asbfly"]

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    dataset = BirdCLEFDataset(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
        split="train",
        label_to_index=label_to_index,
        audio_config=AudioConfig(),
    )

    # The first row of the current manifest is asbfly.
    assert len(dataset) > 0


def test_dataset_output_shape():
    label_to_index = {
        "asbfly": 0,
    }

    dataset = BirdCLEFDataset(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
        split="train",
        label_to_index=label_to_index,
        audio_config=AudioConfig(),
    )

    spectrogram, label = dataset[0]

    expected_shape = (
        1,
        128,
        AudioConfig().time_frames,
    )

    assert spectrogram.shape == expected_shape
    assert label.dtype == torch.long


def test_dataset_output_is_finite():
    label_to_index = {
        "asbfly": 0,
    }

    dataset = BirdCLEFDataset(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
        split="train",
        label_to_index=label_to_index,
        audio_config=AudioConfig(),
    )

    spectrogram, _ = dataset[0]

    assert torch.isfinite(
        spectrogram
    ).all()
