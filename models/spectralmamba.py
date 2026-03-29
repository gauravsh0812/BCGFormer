"""
SpectralMamba — Efficient Mamba for Hyperspectral Image Classification
Reference: Yao et al., "SpectralMamba: Efficient Mamba for Hyperspectral
           Image Classification", arXiv 2404.08489, 2024.

Key design:
- Dynamical mask (CNN-based) for spatial-spectral encoding
- Piece-wise Sequence Scanning (PSS) to reduce sequence length
- Gated State Space Module (GSSM) — simplified SSM without full Mamba deps
- Spectral-focused, lightweight design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
# SIMPLIFIED STATE SPACE MODULE (SSM)
# ─────────────────────────────────────────────
class SelectiveSSM(nn.Module):
    """
    Simplified selective state space model (S6).
    Input-dependent A, B, C parameters — the core Mamba mechanism.
    Implemented without mamba_ssm package dependency.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projection
        self.in_proj  = nn.Linear(d_model, d_model * 2)   # x and z branches

        # Conv for local context (causal conv)
        self.conv1d = nn.Conv1d(
            d_model, d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model)

        # Fixed A matrix (log parameterized)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: B x L x D
        B, L, D = x.shape

        # Split into x and z (gating) branches
        xz  = self.in_proj(x)
        x_b, z = xz.chunk(2, dim=-1)          # B x L x D each

        # Causal conv
        x_b = x_b.transpose(1, 2)             # B x D x L
        x_b = self.conv1d(x_b)[..., :L]       # causal: trim padding
        x_b = x_b.transpose(1, 2)             # B x L x D
        x_b = F.silu(x_b)

        # SSM
        A   = -torch.exp(self.A_log.float())   # D x d_state
        ssm_params = self.x_proj(x_b)          # B x L x (2*d_state + D)
        dt, B_ssm, C_ssm = torch.split(
            ssm_params,
            [D, self.d_state, self.d_state],
            dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))      # B x L x D

        # Discretize A
        dA = torch.exp(dt.unsqueeze(-1) * A)   # B x L x D x d_state

        # Simplified SSM scan (parallel approximation)
        dB_x = dt.unsqueeze(-1) * B_ssm.unsqueeze(2) * x_b.unsqueeze(-1)
        # dB_x: B x L x D x d_state

        # Cumulative sum approximation of recurrence
        h = dB_x.cumsum(dim=1)                 # B x L x D x d_state
        y = (h * C_ssm.unsqueeze(2)).sum(-1)   # B x L x D
        y = y + self.D * x_b

        # Gate with z
        y = y * F.silu(z)
        return self.out_proj(y)


# ─────────────────────────────────────────────
# GATED STATE SPACE MODULE (GSSM)
# ─────────────────────────────────────────────
class GSSM(nn.Module):
    """
    Gated State Space Module — wraps SSM with layer norm and residual.
    The core building block of SpectralMamba.
    """
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm  = SelectiveSSM(d_model, d_state)
        self.drop = nn.Dropout(dropout)

        # Feed-forward after SSM
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.drop(self.ssm(self.norm(x)))
        x = x + self.ff(x)
        return x


# ─────────────────────────────────────────────
# DYNAMICAL MASK (Spatial-Spectral Encoder)
# ─────────────────────────────────────────────
class DynamicalMask(nn.Module):
    """
    CNN-based dynamical mask to encode spatial regularity
    and spectral peculiarity simultaneously.
    Attenuates spectral variability before SSM processing.
    """
    def __init__(self, num_bands: int, embed_dim: int):
        super().__init__()
        self.mask_conv = nn.Sequential(
            nn.Conv2d(num_bands, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.BatchNorm2d(embed_dim),
            nn.Sigmoid(),                       # mask values in [0,1]
        )
        self.feat_conv = nn.Sequential(
            nn.Conv2d(num_bands, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: B x C x H x W
        mask  = self.mask_conv(x)    # B x embed_dim x H x W
        feats = self.feat_conv(x)    # B x embed_dim x H x W
        return feats * mask          # element-wise gating


# ─────────────────────────────────────────────
# PIECE-WISE SEQUENCE SCANNING (PSS)
# ─────────────────────────────────────────────
class PiecewiseScanning(nn.Module):
    """
    Reduces sequence length by grouping spatial positions
    into pieces and aggregating within each piece.
    Transfers ~continuous spectrum into squeezed sequences.
    """
    def __init__(self, piece_size: int = 2):
        super().__init__()
        self.piece_size = piece_size

    def forward(self, x):
        # x: B x L x D
        B, L, D = x.shape
        # Pad if needed
        pad = (self.piece_size - L % self.piece_size) % self.piece_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        # Reshape and mean-pool within pieces
        x = x.reshape(B, -1, self.piece_size, D)
        x = x.mean(dim=2)             # B x (L//piece_size) x D
        return x


# ─────────────────────────────────────────────
# SPECTRALMAMBA
# ─────────────────────────────────────────────
class SpectralMamba(nn.Module):
    def __init__(
        self,
        num_bands:   int,
        num_classes: int,
        window_size: int   = 5,
        embed_dim:   int   = 64,
        d_state:     int   = 16,
        depth:       int   = 4,
        piece_size:  int   = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # Stage 1 — Dynamical mask encoding
        self.dyn_mask = DynamicalMask(num_bands, embed_dim)

        # Stage 2 — Piece-wise sequence scanning
        self.pss = PiecewiseScanning(piece_size)

        # Positional encoding
        seq_len        = (window_size * window_size + piece_size - 1) // piece_size
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 4, embed_dim))  # +buffer
        self.pos_drop  = nn.Dropout(dropout)

        # Stage 3 — Stacked GSSM blocks
        self.gssm_blocks = nn.ModuleList([
            GSSM(embed_dim, d_state, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Stage 4 — Classifier
        # Use adaptive pooling to get fixed-size representation
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, labels=None):
        # x: B x C x H x W
        B, C, H, W = x.shape

        # Dynamical mask
        feats = self.dyn_mask(x)                           # B x embed_dim x H x W

        # Flatten spatial → sequence
        feats = feats.flatten(2).transpose(1, 2)           # B x (H*W) x embed_dim

        # Piece-wise scanning (sequence compression)
        feats = self.pss(feats)                            # B x L' x embed_dim

        # Positional encoding (trim/pad to match)
        L = feats.shape[1]
        pos = self.pos_embed[:, :L, :]
        feats = self.pos_drop(feats + pos)

        # GSSM blocks
        for block in self.gssm_blocks:
            feats = block(feats)

        feats = self.norm(feats)                           # B x L' x embed_dim

        # Global average pooling over sequence
        cls_out = feats.transpose(1, 2)                    # B x embed_dim x L'
        cls_out = self.pool(cls_out).squeeze(-1)           # B x embed_dim

        logits = self.classifier(cls_out)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}