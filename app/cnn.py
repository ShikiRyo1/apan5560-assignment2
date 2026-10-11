"""The exact two-convolution architecture specified in Assignment 2."""

import torch
from torch import nn

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
IMAGE_SIZE = 64


class AssignmentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss expects these unnormalized logits, not softmax values.
        return self.classifier(self.features(images))
