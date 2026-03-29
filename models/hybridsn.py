"""
HybridSN Model
Reference: Roy et al., "HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy
           for Hyperspectral Image Classification", IEEE GRSL 2020.
"""

import torch
import torch.nn as nn


class HybridSN(nn.Module):
    def __init__(self, num_bands: int, num_classes: int, window_size: int = 5):
        super().__init__()

        # ── 3D Convolutional Blocks ──────────────────────────────────────
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # ── 2D Convolutional Block ───────────────────────────────────────
        # After 3D conv, reshape: (B, 32*num_bands, H, W) → 2D conv input
        self.conv2d = nn.Sequential(
            nn.Conv2d(32 * num_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── Classifier ──────────────────────────────────────────────────
        flat_size = 64 * window_size * window_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, labels=None):
        # x: B x C x H x W  →  B x 1 x C x H x W  (add 3D channel dim)
        x = x.unsqueeze(1)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        # Reshape for 2D conv: B x (32*C) x H x W
        B, C, D, H, W = x.shape
        x = x.reshape(B, C * D, H, W)
        x = self.conv2d(x)
        logits = self.classifier(x)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}