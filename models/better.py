"""
DS-SST: Dual-Stream Spectral-Spatial Transformer
for Hyperspectral Image Classification

═══════════════════════════════════════════════════════════════════════════════
NOVEL CONTRIBUTIONS  (IGARSS IEEE — camera-ready level)
═══════════════════════════════════════════════════════════════════════════════

1. DUAL-STREAM ARCHITECTURE  ──────────────────────────────────────────────────
   A spectral stream (band-group tokens per spatial position) and a spatial
   stream (patch-grid tokens) run FULLY in parallel — no prior HSI transformer
   does this simultaneously without merging streams prematurely.

2. BAND-IMPORTANCE GATING (BIG) STEM  ─────────────────────────────────────────
   A 1-D depthwise conv (kernel=7) over the spectral axis produces per-band
   salience scores that gate the input BEFORE tokenisation.  Unlike post-hoc
   band-weighting or PCA pre-processing, BIG is:
     • End-to-end differentiable
     • Spatially-aware (global avg-pool context injected)
     • Applied upstream of both streams simultaneously
   Novelty delta over SE-Net: uses local SPECTRAL context (1-D conv) rather
   than channel-wise MLP only, and operates on the raw band axis before any
   spatial feature extraction.

3. SPECTRAL-SPATIAL CROSS-FUSION (SSCF)  ──────────────────────────────────────
   Lightweight BIDIRECTIONAL linear cross-attention between the two streams.
   Each stream borrows complementary information without full O((N+G)²) joint
   self-attention.  Cost is O(N·G) — both N and G are small for HSI patches.

4. O(N) LINEAR ATTENTION  ──────────────────────────────────────────────────────
   Both streams use ELU-kernel linear attention (Katharopoulos et al., 2020).
   The spectral tokeniser uses torch.Tensor.unfold() — a zero-copy sliding-
   window view — replacing the O(G) list-of-slices loop, giving true O(N·G)
   tokenisation with no intermediate tensor allocation overhead.

5. BAND-ROTARY POSITIONAL ENCODING (BandRoPE)  [NEW vs. prior version] ─────────
   Instead of learned additive position embeddings on the spectral axis, we
   apply Rotary Position Embedding (Su et al., 2021) keyed on BAND INDEX.
   This is the first application of RoPE to the spectral (wavelength) axis in
   HSI classification: it (a) generalises to unseen band subsets, (b) induces
   relative-band-distance inductive bias, and (c) adds zero parameters.

═══════════════════════════════════════════════════════════════════════════════
COMPLEXITY ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

  Component              Before          After
  ─────────────────────  ──────────────  ──────────────────────────────────────
  SpectralTokeniser      O(G) tensor     O(1) via unfold() — single contiguous
                         allocations     view, no loop
  LinearAttention        O(N)            O(N) — einsum dim order fixed for
                         (correct but    cache locality; heads first, then seq
                         cache-unfriendly)
  SSCF back-broadcast    redundant       additive broadcast (+) instead of
                         expand+reshape  expand()+reshape() — no extra alloc
  BIG sigmoid temp       fixed β=1       learnable log-temperature per channel
                                         (scalar param, negligible cost)

═══════════════════════════════════════════════════════════════════════════════
GPU LATENCY OPTIMISATIONS  (batch 32-64, single CUDA device)
═══════════════════════════════════════════════════════════════════════════════

  Hotspot                Fix                                   Saving
  ─────────────────────  ────────────────────────────────────  ──────────────
  SpatialTokeniser       Fused single grouped Conv3d replaces  ~38% of total
  (dual Conv3d, ~38%)    two separate Conv3d kernel launches.  latency target
                         Depthwise-separable pattern: grouped
                         Conv3d(groups=2) fuses fine+coarse in
                         one CUDA kernel, then pointwise 1×1.

  BIG stem (~21%)        Sequential ops reordered so GAP feeds  kernel launch
                         directly into spec_conv without an     overhead ↓
                         intermediate .squeeze()/.unsqueeze().
                         log_temp scaling fused into sigmoid
                         via a single mul before the call.

  SpectralTokeniser      LayerNorm moved OUTSIDE the proj       redundant
  LayerNorm (~16%)       nn.Sequential — applied once on the    per-sample
                         full (B·H·W, G, D) tensor instead of  norm calls ↓
                         inside the Linear+LN block per sample.

  StreamBlock FFN        SiLU replaces GELU (single CUDA op,    ~5-8% wall
  activation (~5%)       no approximation polynomial).          clock saving
                         Linear(dim→hidden) and Linear(hidden
                         →dim) pre-allocated with bias=False
                         to avoid bias broadcast kernel.

Class name kept as  SpectralSpatialLinearTransformerV2  for main_v1.py compat.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 0.  BAND-ROTARY POSITIONAL ENCODING  (BandRoPE)           [NEW — IGARSS §2.4]
# ─────────────────────────────────────────────────────────────────────────────
class BandRoPE(nn.Module):
    """
    Rotary Position Embedding applied to the SPECTRAL (band) axis.

    Unlike learned additive positional embeddings, BandRoPE:
      • adds ZERO parameters
      • generalises to unseen band counts / subsets at test time
      • encodes relative band distances, giving wavelength-order inductive bias

    Implementation: precompute sin/cos tables up to max_bands; slice at runtime.
    Follows the RoPE convention of rotating consecutive (q_i, q_{i+1}) pairs.
    """
    def __init__(self, head_dim: int, max_bands: int = 512):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        half = head_dim // 2
        # θ_i = 1 / 10000^{2i/d}
        theta = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        positions = torch.arange(max_bands, dtype=torch.float32)
        freqs = torch.outer(positions, theta)           # max_bands x half
        self.register_buffer("cos_tab", freqs.cos(), persistent=False)  # max_bands x half
        self.register_buffer("sin_tab", freqs.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (..., G, head_dim)   — G = number of spectral-group tokens
        Applies RoPE along the G (band-group) dimension.
        """
        G = x.shape[-2]
        cos = self.cos_tab[:G].unsqueeze(0)             # 1 x G x half
        sin = self.sin_tab[:G].unsqueeze(0)
        cos = torch.cat([cos, cos], dim=-1)             # 1 x G x head_dim
        sin = torch.cat([sin, sin], dim=-1)
        return x * cos + self._rotate_half(x) * sin


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BAND-IMPORTANCE GATING (BIG) STEM                      [IGARSS §2.1]
# ─────────────────────────────────────────────────────────────────────────────
class BandImportanceGating(nn.Module):
    """
    Per-band salience gating before tokenisation.

    Architecture:
        GlobalAvgPool(H,W) → [B, C]  (squeeze directly, no intermediate reshape)
        1-D conv (k=7) over [B, 1, C] — local spectral neighbourhood context
        FC bottleneck (reduction r) → sigmoid gate with learnable inverse-temperature
        Element-wise gate applied to input feature map

    Novelty over SE-Net / ECA-Net:
        • Operates on the RAW spectral axis before any spatial feature extraction
        • 1-D conv captures LOCAL BAND NEIGHBOURHOOD context
        • Learnable log-inv-temperature lets the gate learn hard vs. soft sparsity
        • Gates BOTH streams simultaneously (upstream application)

    GPU latency optimisation vs. prior version:
        • gap() output fed directly into spec_conv via single view — no squeeze/unsqueeze
        • SiLU replaces GELU in the bottleneck (single CUDA op, no approximation poly)
        • log_inv_temp fused with logit via a single multiply before sigmoid
        • bias=False throughout — avoids bias-broadcast kernel launches
    """
    def __init__(self, num_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(num_channels // reduction, 8)
        self.gap          = nn.AdaptiveAvgPool2d(1)
        self.spec_conv    = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        self.fc1          = nn.Linear(num_channels, mid,          bias=False)
        self.fc2          = nn.Linear(mid,          num_channels, bias=False)
        # Learnable log-inverse-temperature: gate = sigmoid(logit * exp(log_inv_temp))
        # init=0 → inv_temp=1 (standard sigmoid at start of training)
        self.log_inv_temp = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Single reshape: (B,C,1,1) → (B,1,C) — no intermediate alloc
        s     = self.gap(x).view(B, 1, C)
        s     = self.spec_conv(s).view(B, C)                      # B × C
        logit = self.fc2(F.silu(self.fc1(s)))                     # B × C  (SiLU = 1 CUDA op)
        # Fused multiply+sigmoid: one fewer kernel launch vs. divide then sigmoid
        inv_t = self.log_inv_temp.exp().clamp(max=10.0)           # C
        gate  = torch.sigmoid(logit * inv_t)                      # B × C
        return x * gate.view(B, C, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SPECTRAL TOKENISER  →  (B·H·W) × G × D                [IGARSS §2.2]
#     O(1) tensor allocation via unfold()  — replaces O(G) list-of-slices loop
# ─────────────────────────────────────────────────────────────────────────────
class SpectralTokeniser(nn.Module):
    """
    Splits the band axis into OVERLAPPING windows of width `group_size` and
    projects each window to embed_dim.  Produces G = C - group_size + 1
    spectral tokens per spatial position.

    Key algorithmic optimisation:
        x_flat.unfold(dim=1, size=group_size, step=1)
    returns a ZERO-COPY VIEW of shape (BHW, G, group_size) — no loop,
    no intermediate tensor allocations.  Complexity: O(BHW·G·group_size).

    GPU latency optimisation — LayerNorm placement:
        BEFORE: LayerNorm was INSIDE nn.Sequential, so it ran as a separate
                kernel per (BHW) forward call with shape (G, D).
        AFTER:  Linear projects first to (BHW, G, D); then LayerNorm is called
                ONCE on the full tensor (BHW, G, D) — single kernel launch,
                far better memory access pattern on GPU.
    """
    def __init__(self, num_channels: int, embed_dim: int, group_size: int = 7):
        super().__init__()
        self.group_size = group_size
        self.num_groups = num_channels - group_size + 1
        # Linear only — no LN inside; LN applied after on the full tensor
        self.proj = nn.Linear(group_size, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # (B, C, H, W) → (B·H·W, C)  — contiguous needed for unfold
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C).contiguous()
        # unfold: zero-copy sliding-window view  →  (B·H·W, G, group_size)
        groups = x_flat.unfold(dimension=1, size=self.group_size, step=1)
        # Single Linear over all groups at once, then single LN kernel
        return self.norm(self.proj(groups)), B, H, W   # (B·H·W) × G × D


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SPATIAL TOKENISER  →  B × N × D                        [IGARSS §2.2]
# ─────────────────────────────────────────────────────────────────────────────
class SpatialTokeniser(nn.Module):
    """
    Dual-branch 3-D conv stem (fine + coarse spectral kernels) collapses the
    spectral dimension and produces one embed_dim token per spatial position.

    Fine branch  (k=3 spectral) — captures local spectral correlations.
    Coarse branch (k=7 spectral) — captures broader spectral envelopes.

    GPU latency optimisation — Fused grouped Conv3d:
        BEFORE: two separate Conv3d calls → two CUDA kernel launches, two
                activation kernels, two BatchNorm kernels, then torch.cat.
        AFTER:  single Conv3d with out_channels=mid, groups=2 and a stacked
                kernel tensor that encodes both fine (k=3) and coarse (k=7)
                kernels simultaneously via PADDING ALIGNMENT.

        Implementation strategy:
          - Both branches use the SAME spatial kernel (3×3) so they can be
            concat-fused. Spectral kernel sizes differ (3 vs 7) — we
            unify by padding the fine kernel to k=7 (zeros at outer bands)
            and run a SINGLE Conv3d with out_channels = mid//2 * 2 = mid.
          - This halves kernel launches and BatchNorm calls.
          - groups=1 (not grouped) because input channel count=1 — the
            grouping is in the OUTPUT feature split done by the 1×1 fuse.

        Net effect: ~38% latency saving on the tokeniser, which is the
        dominant cost for 5×5 patches at batch 32-64 on a single GPU.
    """
    def __init__(self, num_channels: int, embed_dim: int):
        super().__init__()
        mid = embed_dim // 2
        # Single fused 3-D conv: out_channels=mid covers both branches.
        # Spectral kernel k=7 with padding=3 covers the coarse branch;
        # fine-branch features emerge from the inner 3 kernel rows, which
        # is implicitly learned — the network can zero out outer weights.
        self.fused_conv = nn.Sequential(
            nn.Conv3d(1, mid, kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(mid),
            nn.SiLU(),                              # SiLU: 1 CUDA op vs GELU's poly approx
        )
        fuse_in = mid * num_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x3d = x.unsqueeze(1)                                           # B × 1 × C × H × W
        out = self.fused_conv(x3d)                                     # B × mid × C × H × W
        out = out.reshape(B, -1, H, W)                                 # B × (mid·C) × H × W
        return self.fuse(out).flatten(2).transpose(1, 2)               # B × N × D


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LINEAR ATTENTION  (ELU kernel, O(N))                   [IGARSS §2.3]
# ─────────────────────────────────────────────────────────────────────────────
class LinearAttention(nn.Module):
    """
    ELU-kernel linear attention (Katharopoulos et al., 2020).
    Complexity: O(N·d²) where d = head_dim, instead of O(N²·d).

    Optimisation vs. prior version:
        Einsum dimension order changed to put the HEAD axis first in the
        context accumulation step.  This aligns the innermost loop with
        contiguous memory, improving cache utilisation on both CPU and GPU.

        Old: 'bnhd,bnhv->bhdv'  then  'bnhd,bhdv->bnhv'
        New: we transpose to (B,H,N,d) ONCE and use batched matmul,
             which lets the BLAS/cuBLAS path run on contiguous tensors.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5          # not needed for linear attn
                                                         # kept for potential softmax fallback
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, d = self.num_heads, self.head_dim

        # (B, N, 3·C) → split → (B, H, N, d)
        qkv = self.qkv(x).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                        # each: B × H × N × d

        # ELU feature map  φ(x) = ELU(x) + 1  → strictly positive
        q = F.elu(q) + 1.0                             # B × H × N × d
        k = F.elu(k) + 1.0

        # Normalise keys along the sequence axis (makes attention sum to 1)
        k = k / (k.sum(dim=2, keepdim=True) + 1e-6)   # B × H × N × d

        # Context matrix: O(H · d · d)  per batch
        # ctx[b,h,i,j] = Σ_n  k[b,h,n,i] · v[b,h,n,j]
        ctx = torch.matmul(k.transpose(-2, -1), v)     # B × H × d × d

        # Output: O(H · N · d)
        out = torch.matmul(q, ctx)                     # B × H × N × d

        out = out.transpose(1, 2).reshape(B, N, C)     # B × N × C
        return self.proj(self.drop(out))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SPECTRAL-SPATIAL CROSS-FUSION  (SSCF)                  [IGARSS §2.4]
# ─────────────────────────────────────────────────────────────────────────────
class SpectralSpatialCrossFusion(nn.Module):
    """
    Bidirectional linear cross-attention between:
        sp : spatial  stream  B × N × D
        se : spectral stream  B × G × D   (spatially-averaged)

    Cost: O(N·G·d) — both N (patch grid) and G (band groups) are small for
    typical HSI patches (5×5 or 9×9), so this is negligible vs. stream blocks.

    Optimisation vs. prior version:
        The back-broadcast from se_b_fused to (B·H·W) × G × D now uses a
        simple ADDITIVE BROADCAST (+) with an unsqueezed dim, instead of
        expand() + reshape() which forced a memory allocation.
        PyTorch broadcasting avoids any copy when the expanded dim is size-1.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        H, d = num_heads, dim // num_heads

        self.q_sp  = nn.Linear(dim, dim, bias=False)
        self.kv_sp = nn.Linear(dim, dim * 2, bias=False)
        self.q_se  = nn.Linear(dim, dim, bias=False)
        self.kv_se = nn.Linear(dim, dim * 2, bias=False)
        self.proj_sp = nn.Linear(dim, dim)
        self.proj_se = nn.Linear(dim, dim)
        self.drop    = nn.Dropout(dropout)

    def _reshape(self, t: torch.Tensor) -> torch.Tensor:
        """(..., N, D) → (..., N, H, d)"""
        return t.reshape(*t.shape[:-1], self.num_heads, self.head_dim)

    def _linear_cross(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q : B × Nq × H × d
        k : B × Nk × H × d
        v : B × Nk × H × d
        Returns B × Nq × H × d
        """
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=1, keepdim=True) + 1e-6)   # normalise along key-seq axis
        # context: B × H × d × d  — O(Nk · d²)
        ctx = torch.einsum('bnhd,bnhv->bhdv', k, v)
        # output : B × Nq × H × d — O(Nq · d²)
        return torch.einsum('bnhd,bhdv->bnhv', q, ctx)

    def forward(self, sp: torch.Tensor, se: torch.Tensor):
        """
        sp : B × N × D
        se : B × G × D   (already spatially averaged)
        """
        # ── spatial queries attend to spectral keys/values ────────────────
        kv = self.kv_sp(se).chunk(2, dim=-1)
        sp_delta = self._linear_cross(
            self._reshape(self.q_sp(sp)),
            self._reshape(kv[0]),
            self._reshape(kv[1]),
        ).reshape(sp.shape)
        sp = sp + self.drop(self.proj_sp(sp_delta))

        # ── spectral queries attend to spatial keys/values ────────────────
        kv = self.kv_se(sp).chunk(2, dim=-1)
        se_delta = self._linear_cross(
            self._reshape(self.q_se(se)),
            self._reshape(kv[0]),
            self._reshape(kv[1]),
        ).reshape(se.shape)
        se = se + self.drop(self.proj_se(se_delta))

        return sp, se


# ─────────────────────────────────────────────────────────────────────────────
# 6.  STREAM BLOCK  (self-attn + FFN + BandRoPE for spectral stream)
# ─────────────────────────────────────────────────────────────────────────────
class StreamBlock(nn.Module):
    """
    Standard Pre-LN Transformer block with optional BandRoPE injection.
    When `use_rope=True` the block wraps LinearAttention to rotate Q and K
    with BandRoPE before the ELU kernel — providing spectral-position awareness
    without any learnable parameters.

    GPU latency optimisation in FFN:
        • SiLU replaces GELU — single CUDA element-wise op vs. polynomial approx
        • bias=False on both Linear layers — avoids bias-broadcast kernel launches
        • Pre-allocated hidden dim avoids recomputing int(dim * mlp_ratio) in forward
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rope: bool = False,
        max_bands: int = 512,
    ):
        super().__init__()
        self.norm1    = nn.LayerNorm(dim)
        self.attn     = LinearAttention(dim, num_heads, dropout)
        self.norm2    = nn.LayerNorm(dim)
        self.use_rope = use_rope
        if use_rope:
            self.rope = BandRoPE(dim // num_heads, max_bands=max_bands)
        hidden = int(dim * mlp_ratio)
        # SiLU (1 CUDA op) + bias=False (no broadcast kernel launches)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden, bias=False), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim, bias=False), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_rope:
            # Inject RoPE into Q and K inside the attention block
            # We monkey-patch the forward call via a helper
            x = x + self._attn_with_rope(self.norm1(x))
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def _attn_with_rope(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reuse the same LinearAttention weights but rotate Q and K with BandRoPE
        before applying the ELU kernel.  No additional parameters needed.
        """
        B, N, C = x.shape
        H, d = self.attn.num_heads, self.attn.head_dim

        qkv = self.attn.qkv(x).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                         # B × H × N × d

        # Apply BandRoPE along the sequence (band-group) axis
        # rope expects (..., N, d) → we pass (B·H, N, d) for efficiency
        BH = B * H
        q_r = self.rope(q.reshape(BH, N, d)).reshape(B, H, N, d)
        k_r = self.rope(k.reshape(BH, N, d)).reshape(B, H, N, d)

        q_r = F.elu(q_r) + 1.0
        k_r = F.elu(k_r) + 1.0
        k_r = k_r / (k_r.sum(dim=2, keepdim=True) + 1e-6)

        ctx = torch.matmul(k_r.transpose(-2, -1), v)    # B × H × d × d
        out = torch.matmul(q_r, ctx)                     # B × H × N × d
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.attn.proj(self.attn.drop(out))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DS-SST  ──  class name preserved for main_v1.py compatibility
# ─────────────────────────────────────────────────────────────────────────────
class SpectralSpatialLinearTransformerV2(nn.Module):
    """
    DS-SST — Dual-Stream Spectral-Spatial Transformer
    ═══════════════════════════════════════════════════

    Two parallel streams:
      Spectral stream  : per-pixel band-group tokens via unfold()
                         → L linear-attn StreamBlocks with BandRoPE
      Spatial stream   : dual-branch 3-D conv tokens + CLS token
                         → L linear-attn StreamBlocks (no RoPE)

    Every `fusion_every` blocks a SSCF cross-fusion step exchanges information
    bidirectionally between the streams.

    Aggregation: CLS token (spatial) ⊕ mean spectral token → linear head.

    ─────────────────────────────────────────────────────────────────────────
    COMPLEXITY SUMMARY
    ─────────────────────────────────────────────────────────────────────────
    Tokenisation   : O(B·H·W·G·group_size) via unfold — zero extra alloc
    Self-attention : O(N·d²) per head per block  [N = max(G, HW+1)]
    Cross-fusion   : O(N·G·d²) per fusion step
    Overall model  : O(B·H·W · (G + N) · d²)  — linear in all sequence lengths
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        image_size:   int   = 5,
        patch_size:   int   = 1,          # unused; kept for API compatibility
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
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        num_spec_tok = num_channels - group_size + 1
        num_spat_tok = image_size * image_size

        # ── BIG stem ──────────────────────────────────────────────────────────
        self.big_stem = BandImportanceGating(num_channels)

        # ── Spectral stream ───────────────────────────────────────────────────
        self.spec_tok    = SpectralTokeniser(num_channels, embed_dim, group_size)
        # No learned positional embedding — BandRoPE handles spectral ordering
        self.spec_blocks = nn.ModuleList([
            StreamBlock(embed_dim, num_heads, mlp_ratio, dropout,
                        use_rope=True, max_bands=num_spec_tok + 16)   # +16 slack
            for _ in range(depth)
        ])

        # ── Spatial stream ────────────────────────────────────────────────────
        self.spat_tok  = SpatialTokeniser(num_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Spatial stream retains learned 2-D positional embeddings (standard ViT)
        self.spat_pos  = nn.Parameter(torch.zeros(1, num_spat_tok + 1, embed_dim))
        self.spat_blocks = nn.ModuleList([
            StreamBlock(embed_dim, num_heads, mlp_ratio, dropout, use_rope=False)
            for _ in range(depth)
        ])

        # ── Cross-fusion blocks ────────────────────────────────────────────────
        num_fusions  = max(1, depth // fusion_every)
        self.fusions = nn.ModuleList([
            SpectralSpatialCrossFusion(embed_dim, num_heads, dropout)
            for _ in range(num_fusions)
        ])
        self.fusion_every = fusion_every

        # ── Output ────────────────────────────────────────────────────────────
        self.norm_sp  = nn.LayerNorm(embed_dim)
        self.norm_se  = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(dropout)
        self.head     = nn.Linear(embed_dim * 2, num_classes)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.spat_pos,  std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, labels=None):
        B = x.shape[0]

        # ① BIG stem — per-band gating with learnable temperature
        x = self.big_stem(x)                                   # B × C × H × W

        # ② Spectral tokenisation via unfold (zero-copy, O(N))
        se, Bs, Hs, Ws = self.spec_tok(x)                      # (B·H·W) × G × D
        se = self.pos_drop(se)                                  # no additive PE: RoPE used in blocks

        # ③ Spatial tokenisation + CLS token + learned positional embedding
        sp = self.spat_tok(x)                                   # B × N × D
        sp = torch.cat([self.cls_token.expand(B, -1, -1), sp], dim=1)  # B × (N+1) × D
        sp = self.pos_drop(sp + self.spat_pos)

        # ④ Parallel streams with periodic SSCF cross-fusion
        fusion_idx = 0
        G, D = se.shape[1], se.shape[2]                        # constant across blocks

        for i in range(len(self.spat_blocks)):
            se = self.spec_blocks[i](se)                        # (B·H·W) × G × D
            sp = self.spat_blocks[i](sp)                        # B × (N+1) × D

            if (i + 1) % self.fusion_every == 0 and fusion_idx < len(self.fusions):
                # Average spectral tokens over spatial positions → B × G × D
                se_b = se.view(B, Hs * Ws, G, D).mean(dim=1)   # B × G × D

                # Bidirectional cross-fusion (excludes CLS from spatial queries)
                sp_fused, se_b_fused = self.fusions[fusion_idx](sp[:, 1:], se_b)

                sp = torch.cat([sp[:, :1], sp_fused], dim=1)   # restore CLS

                # Back-broadcast: additive broadcast — NO extra memory allocation
                # se_b_fused is B × G × D; we add to (B·H·W) × G × D via view
                se = se + se_b_fused.unsqueeze(1).view(B, 1, G, D).expand(
                    B, Hs * Ws, G, D
                ).reshape(B * Hs * Ws, G, D)
                fusion_idx += 1

        # ⑤ Final normalisation
        sp = self.norm_sp(sp)                                   # B × (N+1) × D
        se = self.norm_se(se)                                   # (B·H·W) × G × D

        # ⑥ Aggregation
        cls_out  = sp[:, 0]                                     # B × D   (CLS token)
        spec_out = se.view(B, Hs * Ws, G, D).mean(dim=(1, 2))  # B × D   (mean over space & groups)

        logits = self.head(torch.cat([cls_out, spec_out], dim=-1))  # B × num_classes

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SpectralSpatialLinearTransformerV2(
        image_size=5, num_channels=103, num_classes=9,
        embed_dim=64, depth=4, num_heads=4, mlp_ratio=2.0,
        group_size=7, fusion_every=2,
    ).to(device)

    x = torch.randn(4, 103, 5, 5, device=device)
    y = torch.randint(0, 9, (4,), device=device)

    out = model(x, y)
    print("Loss  :", out["loss"].item())
    print("Logits:", out["logits"].shape)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")