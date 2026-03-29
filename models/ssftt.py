"""
SSFTT — Spectral-Spatial Feature Tokenization Transformer
Reference: Sun et al., "Spectral-Spatial Feature Tokenization Transformer
           for Hyperspectral Image Classification", IEEE TGRS 2022.
           https://github.com/zgr6010/HSI_SSFTT

Key design:
- 3D conv + 2D conv shallow feature extractor
- Gaussian-weighted feature tokenizer
- Standard Transformer encoder
- CLS token classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# SHALLOW FEATURE EXTRACTOR (3D + 2D CNN)
# ─────────────────────────────────────────────
class ShallowFeatureExtractor(nn.Module):
    """
    Extracts low-level spectral-spatial features using
    a 3D conv followed by a 2D conv, as in the original paper.
    """
    def __init__(self, num_bands: int, out_channels: int = 64):
        super().__init__()

        # 3D conv: captures joint spectral-spatial features
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(8),
            nn.GELU(),
        )

        # After 3D conv: B x 8 x C x H x W → reshape → B x (8*C) x H x W
        in_2d = 8 * num_bands

        # 2D conv: further spatial feature refinement
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_2d, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.out_channels = out_channels

    def forward(self, x):
        # x: B x C x H x W
        B, C, H, W = x.shape
        x = x.unsqueeze(1)              # B x 1 x C x H x W
        x = self.conv3d(x)              # B x 8 x C x H x W
        x = x.reshape(B, -1, H, W)     # B x (8*C) x H x W
        x = self.conv2d(x)              # B x out_channels x H x W
        return x


# ─────────────────────────────────────────────
# GAUSSIAN-WEIGHTED FEATURE TOKENIZER
# ─────────────────────────────────────────────
class GaussianTokenizer(nn.Module):
    """
    Converts shallow CNN features into transformer tokens
    using a Gaussian-weighted linear projection.
    Gaussian weights emphasize central/important features.
    """
    def __init__(self, in_channels: int, num_tokens: int, embed_dim: int):
        super().__init__()
        self.num_tokens = num_tokens

        # Learnable Gaussian-initialized projection weights
        self.proj = nn.Linear(in_channels, num_tokens * embed_dim)
        self.embed_dim = embed_dim

        # Initialize with Gaussian distribution (key design choice)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: B x C x H x W
        B, C, H, W = x.shape
        # Pool spatial dims to get per-pixel feature vector
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W) x C
        x = self.proj(x)                                   # (B*H*W) x (T*E)
        x = x.reshape(B * H * W, self.num_tokens, self.embed_dim)
        return x, B, H, W


# ─────────────────────────────────────────────
# TRANSFORMER ENCODER BLOCK
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm      = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x           = x + attn_out
        x           = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# SSFTT
# ─────────────────────────────────────────────
class SSFTT(nn.Module):
    def __init__(
        self,
        num_bands:      int,
        num_classes:    int,
        window_size:    int   = 5,
        cnn_channels:   int   = 64,
        num_tokens:     int   = 16,
        embed_dim:      int   = 64,
        num_heads:      int   = 4,
        depth:          int   = 4,
        dropout:        float = 0.1,
    ):
        super().__init__()

        # Stage 1 — Shallow feature extraction
        self.extractor = ShallowFeatureExtractor(num_bands, cnn_channels)

        # Stage 2 — Gaussian tokenizer
        self.tokenizer = GaussianTokenizer(cnn_channels, num_tokens, embed_dim)

        # CLS token + positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        # Stage 3 — Transformer encoder
        self.encoder = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Stage 4 — Classifier
        # CLS token per pixel → reshape over spatial window → classify
        flat_dim = embed_dim * window_size * window_size

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, labels=None):
        # x: B x C x H x W
        B, C, H, W = x.shape

        # Shallow features
        feats = self.extractor(x)                          # B x cnn_ch x H x W

        # Tokenize
        tokens, B, H, W = self.tokenizer(feats)           # (B*H*W) x T x E

        # Prepend CLS token
        BHW = B * H * W
        cls = self.cls_token.expand(BHW, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)           # (B*H*W) x (T+1) x E
        tokens = self.pos_drop(tokens + self.pos_embed)

        # Transformer encoder
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        # CLS token extraction
        cls_out = tokens[:, 0]                             # (B*H*W) x E

        # Reshape over spatial window and flatten
        cls_out = cls_out.reshape(B, H * W, -1)           # B x (H*W) x E
        cls_out = cls_out.reshape(B, -1)                  # B x (H*W*E)

        logits = self.classifier(cls_out)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}