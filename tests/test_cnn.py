import torch

from src.models.cnn import BirdCNN


def test_cnn_output_shape():
    model = BirdCNN(num_classes=182)

    x = torch.randn(2, 1, 128, 313)

    output = model(x)

    assert output.shape == (2, 182)


def test_cnn_output_is_finite():
    model = BirdCNN(num_classes=182)

    x = torch.randn(2, 1, 128, 313)

    output = model(x)

    assert torch.isfinite(output).all()


def test_cnn_accepts_single_example():
    model = BirdCNN(num_classes=182)

    x = torch.randn(1, 1, 128, 313)

    output = model(x)

    assert output.shape == (1, 182)
