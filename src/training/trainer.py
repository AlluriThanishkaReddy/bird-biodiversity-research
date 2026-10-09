"""
Training utilities for the clean BirdCLEF CNN baseline.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> float:
    """
    Train the model for one epoch.

    Returns:
        Average training loss.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets,
        )

        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip,
            )

        optimizer.step()

        batch_size = inputs.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Training loader contains no samples."
        )

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate the model.

    Returns:
        loss,
        true labels,
        predicted labels.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(
            outputs,
            targets,
        )

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        batch_size = inputs.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        all_targets.append(
            targets.cpu().numpy()
        )

        all_predictions.append(
            predictions.cpu().numpy()
        )

    if total_samples == 0:
        raise ValueError(
            "Evaluation loader contains no samples."
        )

    average_loss = (
        total_loss / total_samples
    )

    targets = np.concatenate(
        all_targets
    )

    predictions = np.concatenate(
        all_predictions
    )

    return (
        average_loss,
        targets,
        predictions,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    path: str | Path,
) -> None:
    """
    Save a training checkpoint.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
        },
        path,
    )
