"""
Convolutional neural network baseline for bird species classification.

This model provides the clean-audio deep-learning baseline before
environmental noise augmentation is introduced.
"""

import torch
import torch.nn as nn


class BirdCNN(nn.Module):
    """
    CNN baseline for Mel-spectrogram bird classification.

    Expected input:
        [batch, 1, mel_bins, time_frames]

    Output:
        [batch, num_classes]
    """

    def __init__(self, num_classes: int = 182):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the CNN."""
        x = self.features(x)
        return self.classifier(x)
