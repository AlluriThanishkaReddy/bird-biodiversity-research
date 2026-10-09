import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import BirdCNN
from src.training.trainer import (
    evaluate,
    train_one_epoch,
)


def _small_loader():
    x = torch.randn(
        4,
        1,
        128,
        313,
    )

    y = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    dataset = TensorDataset(
        x,
        y,
    )

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )


def test_train_one_epoch_returns_finite_loss():
    model = BirdCNN(
        num_classes=4
    )

    loader = _small_loader()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    criterion = nn.CrossEntropyLoss()

    loss = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_evaluate_returns_predictions():
    model = BirdCNN(
        num_classes=4
    )

    loader = _small_loader()

    criterion = nn.CrossEntropyLoss()

    loss, targets, predictions = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert np.isfinite(loss)
    assert len(targets) == 4
    assert len(predictions) == 4
