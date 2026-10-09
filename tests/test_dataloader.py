from pathlib import Path

from src.data.dataloader import (
    build_label_mapping,
    build_dataloaders,
)


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


def test_label_mapping_contains_182_species():
    mapping = build_label_mapping(
        MANIFEST_PATH
    )

    assert len(mapping) == 182
    assert len(set(mapping.values())) == 182


def test_label_mapping_is_deterministic():
    mapping_1 = build_label_mapping(
        MANIFEST_PATH
    )

    mapping_2 = build_label_mapping(
        MANIFEST_PATH
    )

    assert mapping_1 == mapping_2


def test_dataloaders_can_produce_batch():
    (
        train_loader,
        val_loader,
        test_loader,
        label_to_index,
    ) = build_dataloaders(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
        batch_size=2,
        num_workers=0,
    )

    assert len(label_to_index) == 182

    train_x, train_y = next(
        iter(train_loader)
    )

    assert train_x.ndim == 4
    assert train_x.shape[0] == 2
    assert train_x.shape[1] == 1
    assert train_x.shape[2] == 128
    assert train_y.shape[0] == 2
