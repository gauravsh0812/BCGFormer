"""
3D-CNN Model for Hyperspectral Image Classification
Reference: Li et al., "Deep Learning for Hyperspectral Image Classification:
           An Overview", IEEE TGRS 2019.
"""

import torch
import torch.nn as nn


class CNN3D(nn.Module):
    def __init__(self, num_bands: int, num_classes: int, window_size: int = 5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 32, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv3d(32, 64, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # Dynamically compute flattened size
        dummy     = torch.zeros(1, 1, num_bands, window_size, window_size)
        flat_size = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, labels=None):
        # x: B x C x H x W  →  B x 1 x C x H x W
        x = x.unsqueeze(1)
        x = self.features(x)
        logits = self.classifier(x)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}