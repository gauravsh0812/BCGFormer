"""
HiT — Hyperspectral Image Transformer Classification Network
Reference: Yang et al., "Hyperspectral Image Transformer Classification Networks",
           IEEE Transactions on Geoscience and Remote Sensing, 2022.
           https://www.fst.um.edu.mo/personal/wp-content/uploads/2022/06/HiT.pdf

Key design:
- Spectral-Adaptive 3D Convolution Projection Module (SA-3DCP)
  Produces local spatial-spectral tokens using spectral-adaptive kernels
- ConV-Permutator (CVP)
  Replaces MLP in transformer with depthwise conv for local spatial context
- Standard multi-head self-attention for global spectral dependencies
- Lightweight hybrid CNN-Transformer design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# SPECTRAL-ADAPTIVE 3D CONVOLUTION PROJECTION
# ─────────────────────────────────────────────
class SpectralAdaptive3DConv(nn.Module):
    """
    Spectral-Adaptive 3D Convolution Projection Module (SA-3DCP).
    Uses grouped 3D convolutions with spectral-adaptive kernel sizes
    to capture both local spectral correlations and spatial context.
    Two parallel branches with different spectral kernel sizes are fused.
    """
    def __init__(self, num_bands: int, embed_dim: int):
        super().__init__()

        mid_dim = embed_dim // 2

        # Branch 1: narrow spectral kernel (fine-grained spectral features)
        self.branch1 = nn.Sequential(
            nn.Conv3d(1, mid_dim // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(mid_dim // 2),
            nn.GELU(),
        )

        # Branch 2: wider spectral kernel (coarse spectral features)
        self.branch2 = nn.Sequential(
            nn.Conv3d(1, mid_dim // 2, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(mid_dim // 2),
            nn.GELU(),
        )

        # Fusion: merge branches and project to embed_dim
        # After 3D conv: B x mid_dim x C x H x W → reshape → B x (mid_dim*C) x H x W
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_dim * num_bands, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.num_bands = num_bands

    def forward(self, x):
        # x: B x C x H x W
        B, C, H, W = x.shape
        x3d = x.unsqueeze(1)                               # B x 1 x C x H x W

        b1  = self.branch1(x3d)                            # B x mid//2 x C x H x W
        b2  = self.branch2(x3d)                            # B x mid//2 x C x H x W
        out = torch.cat([b1, b2], dim=1)                   # B x mid x C x H x W

        # Reshape for 2D fusion
        out = out.reshape(B, -1, H, W)                     # B x (mid*C) x H x W
        out = self.fusion(out)                             # B x embed_dim x H x W
        return out


# ─────────────────────────────────────────────
# CONV-PERMUTATOR (CVP)
# ─────────────────────────────────────────────
class ConvPermutator(nn.Module):
    """
    ConV-Permutator (CVP) — replaces the standard MLP in transformer.
    Uses depthwise separable convolutions to capture local spatial context
    that standard MLP layers miss.
    Input treated as sequence: B x N x C → reshape → apply conv → reshape back.
    """
    def __init__(self, embed_dim: int, spatial_size: int, dropout: float = 0.1):
        super().__init__()
        self.spatial_size = spatial_size

        # Depthwise conv over spatial tokens
        self.dw_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Pointwise projection
        self.pw_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: B x N x C
        B, N, C = x.shape

        # Depthwise conv over token sequence
        x_dw = x.transpose(1, 2)                           # B x C x N
        x_dw = self.dw_conv(x_dw).transpose(1, 2)         # B x N x C

        # Pointwise MLP
        x_pw = self.pw_conv(x)                             # B x N x C

        return x_dw + x_pw                                 # fuse both paths


# ─────────────────────────────────────────────
# HIT TRANSFORMER BLOCK
# ─────────────────────────────────────────────
class HiTBlock(nn.Module):
    """
    HiT Transformer block:
    LayerNorm → MHSA → residual → LayerNorm → CVP → residual
    Replaces standard MLP with ConV-Permutator.
    """
    def __init__(
        self,
        embed_dim:    int,
        num_heads:    int,
        spatial_size: int,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cvp   = ConvPermutator(embed_dim, spatial_size, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        x_norm      = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x           = x + self.drop(attn_out)

        # ConV-Permutator instead of MLP
        x = x + self.drop(self.cvp(self.norm2(x)))
        return x


# ─────────────────────────────────────────────
# HIT — HYPERSPECTRAL IMAGE TRANSFORMER
# ─────────────────────────────────────────────
class HiT(nn.Module):
    def __init__(
        self,
        num_bands:   int,
        num_classes: int,
        window_size: int   = 5,
        embed_dim:   int   = 64,
        num_heads:   int   = 4,
        depth:       int   = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()

        spatial_size = window_size * window_size

        # Stage 1 — Spectral-Adaptive 3D Convolution Projection
        self.sa3dcp = SpectralAdaptive3DConv(num_bands, embed_dim)

        # Stage 2 — Token sequence preparation
        # CLS token + positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, spatial_size + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        # Stage 3 — Stacked HiT blocks
        self.blocks = nn.ModuleList([
            HiTBlock(embed_dim, num_heads, spatial_size + 1, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Stage 4 — Classifier using CLS token
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, labels=None):
        # x: B x C x H x W
        B, C, H, W = x.shape

        # Stage 1: spectral-adaptive 3D projection
        feats = self.sa3dcp(x)                             # B x embed_dim x H x W

        # Flatten spatial → token sequence
        feats = feats.flatten(2).transpose(1, 2)           # B x (H*W) x embed_dim

        # Prepend CLS token
        cls   = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feats], dim=1)            # B x (H*W+1) x embed_dim
        tokens = self.pos_drop(tokens + self.pos_embed)

        # HiT blocks (MHSA + ConV-Permutator)
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # CLS token as global representation
        cls_out = tokens[:, 0]                             # B x embed_dim

        logits = self.classifier(cls_out)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}