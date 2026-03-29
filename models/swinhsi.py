"""
Swin-HSI — Spectral Swin Transformer for Hyperspectral Image Classification
Reference: Liu et al., "Spectral Swin Transformer Network for Hyperspectral
           Image Classification", Remote Sensing 2023.
           https://github.com/MinatoRyu007/Swin-HSI

Key design:
- PCA-based spectral dimensionality reduction
- Single-stage Swin Transformer (simplified from 4-stage to 1-stage)
- Shifted window multi-head self-attention (SW-MSA)
- Window-based local attention with cyclic shift
- CLS token classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# WINDOW PARTITION & REVERSE
# ─────────────────────────────────────────────
def window_partition(x, window_size):
    """Partition tokens into non-overlapping windows."""
    B, N, C = x.shape
    # Treat sequence as 1D windows over spectral tokens
    num_windows = N // window_size
    x = x[:, :num_windows * window_size, :].reshape(B, num_windows, window_size, C)
    windows = x.reshape(B * num_windows, window_size, C)
    return windows, num_windows


def window_reverse(windows, num_windows, B, N, C):
    """Reverse window partition."""
    x = windows.reshape(B, num_windows, -1, C)
    x = x.reshape(B, num_windows * windows.shape[1], C)
    return x


# ─────────────────────────────────────────────
# WINDOW ATTENTION
# ─────────────────────────────────────────────
class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA)."""
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads  = num_heads
        self.head_dim   = dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Relative position bias table
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        coords  = torch.arange(N, device=x.device)
        rel_idx = coords[:, None] - coords[None, :]
        rel_idx = rel_idx + self.window_size - 1
        rel_idx = rel_idx.clamp(0, 2 * self.window_size - 2)
        bias    = self.rel_pos_bias[rel_idx].permute(2, 0, 1).unsqueeze(0)
        attn    = attn + bias

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ─────────────────────────────────────────────
# SWIN TRANSFORMER BLOCK
# ─────────────────────────────────────────────
class SwinBlock(nn.Module):
    """
    Swin Transformer block with alternating W-MSA and SW-MSA.
    shift=True → shifted window attention (SW-MSA)
    shift=False → regular window attention (W-MSA)
    """
    def __init__(
        self,
        dim:         int,
        num_heads:   int,
        window_size: int   = 4,
        shift:       bool  = False,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.shift       = shift
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, window_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, N, C = x.shape

        # Cyclic shift for SW-MSA
        if self.shift:
            shift_size = self.window_size // 2
            x_shifted  = torch.roll(x, shifts=-shift_size, dims=1)
        else:
            x_shifted = x

        # Window partition → attention → reverse
        windows, num_windows = window_partition(x_shifted, self.window_size)
        attn_out = self.attn(self.norm1(windows))
        x_attn   = window_reverse(attn_out, num_windows, B, N, C)

        # Reverse cyclic shift
        if self.shift:
            x_attn = torch.roll(x_attn, shifts=shift_size, dims=1)

        # Pad/crop back to original length if needed
        if x_attn.shape[1] < N:
            x_attn = F.pad(x_attn, (0, 0, 0, N - x_attn.shape[1]))
        elif x_attn.shape[1] > N:
            x_attn = x_attn[:, :N, :]

        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# SPECTRAL DIMENSIONALITY REDUCTION (PCA-free)
# ─────────────────────────────────────────────
class SpectralProjection(nn.Module):
    """
    Learnable spectral projection to reduce bands.
    Replaces PCA from original paper for end-to-end training.
    """
    def __init__(self, num_bands: int, projected_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(num_bands, projected_dim, kernel_size=1),
            nn.BatchNorm1d(projected_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: B x C x H x W → treat spectral as channels
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C, 1)
        x = self.proj(x).squeeze(-1)                       # (B*H*W) x projected_dim
        return x, B, H, W


# ─────────────────────────────────────────────
# SWIN-HSI
# ─────────────────────────────────────────────
class SwinHSI(nn.Module):
    def __init__(
        self,
        num_bands:      int,
        num_classes:    int,
        window_size:    int   = 5,
        projected_dim:  int   = 64,
        embed_dim:      int   = 64,
        num_heads:      int   = 4,
        depth:          int   = 4,
        swin_window:    int   = 4,
        dropout:        float = 0.1,
    ):
        super().__init__()

        # Stage 1 — Spectral projection (replaces PCA)
        self.spectral_proj = SpectralProjection(num_bands, projected_dim)

        # Stage 2 — Token embedding
        self.token_embed = nn.Linear(projected_dim, embed_dim)

        # CLS token + positional encoding
        # Sequence length = H * W spatial positions per pixel
        seq_len = window_size * window_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        # Stage 3 — Single-stage Swin Transformer
        # Alternating W-MSA and SW-MSA blocks
        self.swin_blocks = nn.ModuleList([
            SwinBlock(
                dim         = embed_dim,
                num_heads   = num_heads,
                window_size = swin_window,
                shift       = (i % 2 == 1),   # alternate shift
                dropout     = dropout,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Stage 4 — Classifier over spatial window
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

        # Spectral projection per pixel
        proj, B, H, W = self.spectral_proj(x)             # (B*H*W) x projected_dim

        # Embed and reshape to spatial sequence
        proj = self.token_embed(proj)                      # (B*H*W) x embed_dim
        proj = proj.reshape(B, H * W, -1)                 # B x (H*W) x embed_dim

        # Prepend CLS token
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, proj], dim=1)             # B x (H*W+1) x embed_dim
        tokens = self.pos_drop(tokens + self.pos_embed)

        # Swin Transformer blocks
        for block in self.swin_blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Extract spatial tokens (exclude CLS), flatten
        spatial = tokens[:, 1:, :]                        # B x (H*W) x embed_dim
        cls_out = spatial.reshape(B, -1)                  # B x (H*W*embed_dim)

        logits = self.classifier(cls_out)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}