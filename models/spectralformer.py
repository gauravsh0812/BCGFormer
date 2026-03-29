"""
SpectralFormer Model
Reference: Hong et al., "SpectralFormer: Rethinking Hyperspectral Image
           Classification with Transformers", IEEE TGRS 2022.
           https://github.com/danfenghong/IEEE_TGRS_SpectralFormer

Key design:
- Groupwise spectral embeddings (neighboring band groups)
- Cross-layer Adaptive Fusion (CAF) skip connections
- Patch-wise (spatial-spectral) version implemented here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# GROUPWISE SPECTRAL EMBEDDING
# ─────────────────────────────────────────────
class GroupwiseSpectralEmbedding(nn.Module):
    """
    Splits bands into overlapping groups and projects each group
    into a token embedding. Captures local spectral continuity.
    """
    def __init__(self, num_bands: int, embed_dim: int, group_size: int = 3):
        super().__init__()
        self.group_size = group_size
        self.num_groups = num_bands - group_size + 1   # overlapping groups
        self.proj = nn.Linear(group_size, embed_dim)

    def forward(self, x):
        # x: B x C x H x W — flatten spatial, treat bands as sequence
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W) x C

        # Build overlapping spectral groups
        groups = []
        for i in range(self.num_groups):
            groups.append(x[:, i:i + self.group_size])   # each: (B*H*W) x group_size
        x = torch.stack(groups, dim=1)                   # (B*H*W) x num_groups x group_size
        x = self.proj(x)                                 # (B*H*W) x num_groups x embed_dim
        return x, B, H, W


# ─────────────────────────────────────────────
# TRANSFORMER ENCODER BLOCK
# ─────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
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
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# CROSS-LAYER ADAPTIVE FUSION (CAF)
# ─────────────────────────────────────────────
class CrossLayerAdaptiveFusion(nn.Module):
    """
    Learns to fuse 'soft' residuals from a shallow layer
    into a deeper layer — the key contribution of SpectralFormer.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))   # learnable fusion weight

    def forward(self, deep, shallow):
        return deep + torch.sigmoid(self.alpha) * shallow


# ─────────────────────────────────────────────
# SPECTRALFORMER
# ─────────────────────────────────────────────
class SpectralFormer(nn.Module):
    def __init__(
        self,
        num_bands:   int,
        num_classes: int,
        window_size: int   = 5,
        embed_dim:   int   = 64,
        num_heads:   int   = 4,
        depth:       int   = 4,
        group_size:  int   = 3,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # Spectral tokenizer
        self.embedding = GroupwiseSpectralEmbedding(num_bands, embed_dim, group_size)
        num_tokens     = num_bands - group_size + 1

        # CLS token + positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        # Transformer encoder stack
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        # Cross-layer adaptive fusion (connect layer 0 → layer depth//2)
        self.caf = CrossLayerAdaptiveFusion(embed_dim)
        self.caf_from = 0
        self.caf_to   = depth // 2

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Spatial aggregation over window
        # After CLS extraction: project to classifier
        self.spatial_pool = nn.AdaptiveAvgPool1d(1)   # pool over spatial positions
        flat_dim = embed_dim * window_size * window_size

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Weight init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, labels=None):
        # x: B x C x H x W
        tokens, B, H, W = self.embedding(x)   # (B*H*W) x num_tokens x embed_dim

        # Prepend CLS token
        BHW = B * H * W
        cls = self.cls_token.expand(BHW, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)           # (B*H*W) x (T+1) x E
        tokens = self.pos_drop(tokens + self.pos_embed)

        # Transformer layers with CAF
        shallow = None
        for i, layer in enumerate(self.layers):
            tokens = layer(tokens)
            if i == self.caf_from:
                shallow = tokens                            # store shallow features
            if i == self.caf_to and shallow is not None:
                tokens = self.caf(tokens, shallow)         # adaptive fusion

        tokens = self.norm(tokens)

        # Extract CLS token as pixel representation
        cls_out = tokens[:, 0]                             # (B*H*W) x E

        # Reshape back to spatial grid and flatten
        cls_out = cls_out.reshape(B, H * W, -1)           # B x (H*W) x E
        cls_out = cls_out.reshape(B, -1)                  # B x (H*W*E)

        logits = self.classifier(cls_out)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}