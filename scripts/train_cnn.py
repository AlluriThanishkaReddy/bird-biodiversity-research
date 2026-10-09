"""
Train the clean BirdCLEF CNN baseline.

Research condition:
    Clean training audio -> clean validation audio

No environmental noise augmentation is applied in this baseline.
"""

import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Make the project root importable when this file is executed directly.
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch
import torch.nn as nn

from src.audio.processor import AudioConfig
from src.data.dataloader import build_dataloaders
from src.models.cnn import BirdCNN
from src.training.trainer import (
    evaluate,
    save_checkpoint,
    train_one_epoch,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "split_manifest.csv"
)

DATASET_ROOT = Path(
    "C:/Datasets/birdclef-2024"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "results"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

SEED = 42

BATCH_SIZE = 32
NUM_WORKERS = 0

NUM_CLASSES = 182

EPOCHS = 30

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

MIN_LR = 1e-6

GRADIENT_CLIP = 1.0

WARMUP_EPOCHS = 3


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def get_device() -> torch.device:
    """Select GPU when available, otherwise CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ---------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create the cosine learning-rate schedule."""

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(
            1,
            EPOCHS - WARMUP_EPOCHS,
        ),
        eta_min=MIN_LR,
    )


# ---------------------------------------------------------------------
# Main training procedure
# ---------------------------------------------------------------------

def main() -> None:
    """Run the clean CNN baseline experiment."""

    set_seed(SEED)

    device = get_device()

    print("=" * 70)
    print("BirdCLEF Clean CNN Baseline")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print("=" * 70)

    # -------------------------------------------------------------
    # Data
    # -------------------------------------------------------------

    audio_config = AudioConfig()

    (
        train_loader,
        val_loader,
        test_loader,
        label_to_index,
    ) = build_dataloaders(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        audio_config=audio_config,
    )

    print(
        f"Train samples: {len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: {len(val_loader.dataset)}"
    )

    print(
        f"Test samples: {len(test_loader.dataset)}"
    )

    print(
        f"Number of labels: {len(label_to_index)}"
    )

    if len(label_to_index) != NUM_CLASSES:
        raise ValueError(
            "Unexpected number of species: "
            f"{len(label_to_index)} "
            f"(expected {NUM_CLASSES})"
        )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = BirdCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    # -------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------

    # This is a single-label multiclass problem.
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_scheduler(
        optimizer
    )

    # -------------------------------------------------------------
    # Training history
    # -------------------------------------------------------------

    history = []

    best_validation_loss = float("inf")
    best_epoch = -1

    best_checkpoint_path = (
        MODEL_DIR
        / "clean_cnn_best.pt"
    )

    # -------------------------------------------------------------
    # Epoch loop
    # -------------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        # Warmup learning rate.
        if epoch <= WARMUP_EPOCHS:

            warmup_lr = (
                LEARNING_RATE
                * epoch
                / WARMUP_EPOCHS
            )

            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = warmup_lr

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip=GRADIENT_CLIP,
        )

        validation_loss, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train loss: {train_loss:.6f}"
        )

        print(
            f"Validation loss: {validation_loss:.6f}"
        )

        print(
            f"Learning rate: {current_lr:.8f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                path=best_checkpoint_path,
            )

            print(
                "Saved new best checkpoint."
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(
                    validation_loss
                ),
                "learning_rate": float(
                    current_lr
                ),
            }
        )

        if epoch > WARMUP_EPOCHS:
            scheduler.step()

    # -----------------------------------------------------------------
    # Save training history
    # -----------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        OUTPUT_DIR
        / "clean_cnn_training_history.json"
    )

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Save experiment metadata
    # -----------------------------------------------------------------

    metadata = {
        "experiment": "clean_cnn_baseline",
        "research_condition": (
            "clean_training_clean_validation"
        ),
        "seed": SEED,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "warmup_epochs": WARMUP_EPOCHS,
        "loss": "CrossEntropyLoss",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "best_epoch": best_epoch,
        "best_validation_loss": (
            float(best_validation_loss)
        ),
        "audio": {
            "sample_rate": audio_config.sample_rate,
            "clip_duration": audio_config.clip_duration,
            "n_fft": audio_config.n_fft,
            "hop_length": audio_config.hop_length,
            "n_mels": audio_config.n_mels,
            "fmin": audio_config.fmin,
            "fmax": audio_config.fmax,
            "normalization": (
                audio_config.normalization
            ),
        },
    }

    metadata_path = (
        OUTPUT_DIR
        / "clean_cnn_experiment.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("Training complete.")
    print("=" * 70)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        "Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Checkpoint: {best_checkpoint_path}"
    )

    print(
        f"History: {history_path}"
    )

    print(
        f"Metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()
