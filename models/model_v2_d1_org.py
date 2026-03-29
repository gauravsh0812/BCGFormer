"""
DS-SST: Dual-Stream Spectral-Spatial Transformer
for Hyperspectral Image Classification

Novel contributions (IGARSS-level):
1. Dual-stream design: a spectral stream (band-group tokens per spatial position)
   and a spatial stream (patch-grid tokens) run in parallel — no prior HSI
   transformer does this simultaneously.
2. Band-Importance Gating (BIG) stem: a 1D depthwise conv over the spectral axis
   produces per-band salience scores that gate the input before tokenisation,
   suppressing noisy/redundant bands end-to-end without PCA.
3. Spectral-Spatial Cross-Fusion (SSCF): lightweight bidirectional cross-attention
   between the two streams so each can borrow complementary information without
   full O((N+G)^2) joint attention.
4. Both streams use ELU-kernel linear attention — O(N) complexity, low latency.

Class name kept as SpectralSpatialLinearTransformerV2 for main_v1.py compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 1. BAND-IMPORTANCE GATING (BIG) STEM
# ─────────────────────────────────────────────────────────────
class BandImportanceGating(nn.Module):
    """
    Learns per-band salience via a 1D depthwise conv over the spectral axis
    followed by a squeeze-excitation bottleneck.  Gates the input feature map
    before tokenisation so both streams see denoised spectral features.
    Unlike BandWeightedPooling (post-attention scalar weights), this is:
      - applied before any attention, shaping both streams
      - spatially-aware (uses global spatial avg-pool context)
      - uses local spectral context via 1D depthwise conv (kernel=7)
    """
    def __init__(self, num_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(num_channels // reduction, 8)
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.spec_conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        self.fc1       = nn.Linear(num_channels, mid, bias=False)
        self.fc2       = nn.Linear(mid, num_channels, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        s = self.gap(x).reshape(B, 1, C)          # B x 1 x C
        s = self.spec_conv(s).squeeze(1)           # B x C  (local spectral context)
        s = torch.sigmoid(self.fc2(F.gelu(self.fc1(s))))  # B x C
        return x * s.unsqueeze(-1).unsqueeze(-1)   # B x C x H x W


# ─────────────────────────────────────────────────────────────
# 2. SPECTRAL TOKENISER  →  (B*H*W) x G x D
# ─────────────────────────────────────────────────────────────
class SpectralTokeniser(nn.Module):
    """
    Splits the band axis into overlapping groups of size `group_size` and
    projects each group to embed_dim.  Produces G = C - group_size + 1
    spectral tokens per spatial position, processed independently.
    """
    def __init__(self, num_channels: int, embed_dim: int, group_size: int = 7):
        super().__init__()
        self.group_size = group_size
        self.num_groups = num_channels - group_size + 1
        self.proj = nn.Sequential(
            nn.Linear(group_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        groups = torch.stack(
            [x_flat[:, i:i + self.group_size] for i in range(self.num_groups)], dim=1
        )                                          # (BHW) x G x group_size
        return self.proj(groups), B, H, W          # (BHW) x G x D


# ─────────────────────────────────────────────────────────────
# 3. SPATIAL TOKENISER  →  B x N x D
# ─────────────────────────────────────────────────────────────
class SpatialTokeniser(nn.Module):
    """
    Dual-branch 3D conv stem (fine + coarse spectral kernels) collapses the
    spectral dimension and produces one embed_dim token per spatial position.
    """
    def __init__(self, num_channels: int, embed_dim: int):
        super().__init__()
        mid = embed_dim // 2
        self.fine = nn.Sequential(
            nn.Conv3d(1, mid // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(mid // 2), nn.GELU(),
        )
        self.coarse = nn.Sequential(
            nn.Conv3d(1, mid // 2, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(mid // 2), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * num_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim), nn.GELU(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x3d = x.unsqueeze(1)                                          # B x 1 x C x H x W
        out = torch.cat([self.fine(x3d), self.coarse(x3d)], dim=1)   # B x mid x C x H x W
        out = out.reshape(B, -1, H, W)                                # B x (mid*C) x H x W
        return self.fuse(out).flatten(2).transpose(1, 2)              # B x N x D


# ─────────────────────────────────────────────────────────────
# 4. LINEAR ATTENTION  (ELU kernel, O(N))
# ─────────────────────────────────────────────────────────────
class LinearAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=1, keepdim=True) + 1e-6)
        ctx = torch.einsum('bnhd,bnhv->bhdv', k, v)
        out = torch.einsum('bnhd,bhdv->bnhv', q, ctx)
        return self.proj(self.drop(out.reshape(B, N, C)))


# ─────────────────────────────────────────────────────────────
# 5. SPECTRAL-SPATIAL CROSS-FUSION (SSCF)
# ─────────────────────────────────────────────────────────────
class SpectralSpatialCrossFusion(nn.Module):
    """
    Bidirectional cross-attention between the spatial stream (B x N x D)
    and the spectral stream (B x G x D, averaged over spatial positions).
    Uses linear attention so cost is O(N·G) not O((N+G)^2).
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.q_sp  = nn.Linear(dim, dim, bias=False)
        self.kv_sp = nn.Linear(dim, dim * 2, bias=False)
        self.q_se  = nn.Linear(dim, dim, bias=False)
        self.kv_se = nn.Linear(dim, dim * 2, bias=False)
        self.proj_sp = nn.Linear(dim, dim)
        self.proj_se = nn.Linear(dim, dim)
        self.drop    = nn.Dropout(dropout)

    def _cross(self, q, k, v):
        # all: B x N x H x d
        k = k / (k.sum(dim=1, keepdim=True) + 1e-6)
        ctx = torch.einsum('bnhd,bnhv->bhdv', k, v)
        return torch.einsum('bnhd,bhdv->bnhv', q, ctx)

    def _reshape(self, t):
        return t.reshape(t.shape[0], t.shape[1], self.num_heads, self.head_dim)

    def forward(self, sp, se):
        # sp: B x N x D,  se: B x G x D
        def elu(t): return F.elu(t) + 1.0

        kv_se = self.kv_sp(se).chunk(2, dim=-1)
        sp_out = self._cross(
            elu(self._reshape(self.q_sp(sp))),
            elu(self._reshape(kv_se[0])),
            self._reshape(kv_se[1]),
        ).reshape(sp.shape)
        sp = sp + self.drop(self.proj_sp(sp_out))

        kv_sp = self.kv_se(sp).chunk(2, dim=-1)
        se_out = self._cross(
            elu(self._reshape(self.q_se(se))),
            elu(self._reshape(kv_sp[0])),
            self._reshape(kv_sp[1]),
        ).reshape(se.shape)
        se = se + self.drop(self.proj_se(se_out))

        return sp, se


# ─────────────────────────────────────────────────────────────
# 6. STREAM BLOCK  (self-attn + FFN)
# ─────────────────────────────────────────────────────────────
class StreamBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = LinearAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────
# 7. DS-SST  (class name kept for main_v1.py compatibility)
# ─────────────────────────────────────────────────────────────
class SpectralSpatialLinearTransformerV2(nn.Module):
    """
    DS-SST — Dual-Stream Spectral-Spatial Transformer.

    Two parallel streams:
      Spectral stream : per-pixel band-group tokens → L linear-attn blocks
      Spatial stream  : patch-grid tokens           → L linear-attn blocks
    Every `fusion_every` blocks a SSCF cross-fusion step exchanges information.
    Aggregation: CLS token (spatial) + mean spectral token → classifier head.
    """
    def __init__(
        self,
        image_size:   int   = 5,
        patch_size:   int   = 1,          # unused, kept for API compatibility
        num_channels: int   = 103,
        num_classes:  int   = 9,
        embed_dim:    int   = 64,
        depth:        int   = 2,
        num_heads:    int   = 4,
        mlp_ratio:    float = 2.0,
        dropout:      float = 0.1,
        group_size:   int   = 7,
        fusion_every: int   = 2,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # ── BIG stem ──────────────────────────────────────────
        self.big_stem = BandImportanceGating(num_channels)

        # ── Spectral stream ───────────────────────────────────
        self.spec_tok    = SpectralTokeniser(num_channels, embed_dim, group_size)
        num_spec_tok     = num_channels - group_size + 1
        self.spec_pos    = nn.Parameter(torch.zeros(1, num_spec_tok, embed_dim))
        self.spec_blocks = nn.ModuleList([
            StreamBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # ── Spatial stream ────────────────────────────────────
        self.spat_tok    = SpatialTokeniser(num_channels, embed_dim)
        num_spat_tok     = image_size * image_size
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spat_pos    = nn.Parameter(torch.zeros(1, num_spat_tok + 1, embed_dim))
        self.spat_blocks = nn.ModuleList([
            StreamBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        # ── Cross-fusion ──────────────────────────────────────
        num_fusions   = max(1, depth // fusion_every)
        self.fusions  = nn.ModuleList([
            SpectralSpatialCrossFusion(embed_dim, num_heads, dropout)
            for _ in range(num_fusions)
        ])
        self.fusion_every = fusion_every

        # ── Head ──────────────────────────────────────────────
        self.norm_sp  = nn.LayerNorm(embed_dim)
        self.norm_se  = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(dropout)
        self.head     = nn.Linear(embed_dim * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.spat_pos,  std=0.02)
        nn.init.trunc_normal_(self.spec_pos,  std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, labels=None):
        B = x.shape[0]

        # ── BIG stem ──────────────────────────────────────────
        x = self.big_stem(x)                              # B x C x H x W

        # ── Spectral stream tokens ────────────────────────────
        se, Bs, Hs, Ws = self.spec_tok(x)                # (B*H*W) x G x D
        se = self.pos_drop(se + self.spec_pos)

        # ── Spatial stream tokens ─────────────────────────────
        sp = self.spat_tok(x)                             # B x N x D
        sp = torch.cat([self.cls_token.expand(B, -1, -1), sp], dim=1)  # B x (N+1) x D
        sp = self.pos_drop(sp + self.spat_pos)

        # ── Parallel streams with periodic cross-fusion ───────
        fusion_idx = 0
        for i in range(len(self.spat_blocks)):
            se = self.spec_blocks[i](se)                  # (B*H*W) x G x D
            sp = self.spat_blocks[i](sp)                  # B x (N+1) x D

            if (i + 1) % self.fusion_every == 0 and fusion_idx < len(self.fusions):
                G, D = se.shape[1], se.shape[2]
                # Average spectral tokens over spatial positions → B x G x D
                se_b = se.reshape(B, Hs * Ws, G, D).mean(dim=1)
                sp_fused, se_b_fused = self.fusions[fusion_idx](sp[:, 1:], se_b)
                sp = torch.cat([sp[:, :1], sp_fused], dim=1)
                # Broadcast fused spectral back to (B*H*W) x G x D
                se = se + se_b_fused.unsqueeze(1).expand(B, Hs * Ws, G, D).reshape(B * Hs * Ws, G, D)
                fusion_idx += 1

        # ── Aggregation ───────────────────────────────────────
        sp = self.norm_sp(sp)
        se = self.norm_se(se)

        cls_out  = sp[:, 0]                                              # B x D
        spec_out = se.reshape(B, Hs * Ws, se.shape[1], se.shape[2]).mean(dim=(1, 2))  # B x D

        logits = self.head(torch.cat([cls_out, spec_out], dim=-1))

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
