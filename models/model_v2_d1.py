import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralSpatialLinearTransformerV2(nn.Module):
    """
    CNN-Transformer HSI classifier with two novel HSI-specific components:
      1. Band-Contextual Gating (BCG): Conv1d spectral neighborhood context
         + learnable temperature sharpening — extends SE-Net for HSI
      2. Linear attention (ELU kernel): O(N) complexity replacing softmax
    """
    def __init__(
        self,
        image_size:    int   = 5,
        num_channels:  int   = 103,
        num_classes:   int   = 9,
        embed_dim:     int   = 64,
        depth:         int   = 3,
        num_heads:     int   = 4,
        mlp_ratio:     float = 2.0,
        dropout:       float = 0.05,
        group_size:    int   = 7,      # kept for pipeline compat
        fusion_every:  int   = 999,    # kept for pipeline compat
        stem_channels: int   = 8,      # kept for pipeline compat
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        # ── CNN stem (unchanged — MobileNet-style, fast) ──────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1,
                      groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

        # ── Novel component 1: Band-Contextual Gating (BCG) ──────────────
        # Conv1d captures spectral neighborhood correlations (physically
        # meaningful: adjacent HSI bands are correlated by material response)
        # Learnable temperature sharpens gate toward hard selection
        self.spec_context = nn.Conv1d(
            1, 1, kernel_size=7, padding=3, bias=False
        )
        self.spec_fc1   = nn.Linear(embed_dim, embed_dim // 4, bias=False)
        self.spec_fc2   = nn.Linear(embed_dim // 4, embed_dim, bias=False)
        self.log_temp   = nn.Parameter(torch.zeros(embed_dim))

        # ── Positional encoding ───────────────────────────────────────────
        self.pos = nn.Parameter(
            torch.zeros(1, image_size * image_size, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        # ── Transformer blocks (linear attention) ────────────────────────
        self.blocks = nn.ModuleList([
            self._make_block(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # ── Head ──────────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    # ── block factory ─────────────────────────────────────────────────────
    def _make_block(self, dim, heads, mlp_ratio, dropout):
        hidden = int(dim * mlp_ratio)
        return nn.ModuleDict({
            "norm1": nn.LayerNorm(dim),
            "qkv":   nn.Linear(dim, dim * 3, bias=False),
            "proj":  nn.Linear(dim, dim),
            "drop":  nn.Dropout(dropout),
            "norm2": nn.LayerNorm(dim),
            "mlp":   nn.Sequential(
                nn.Linear(dim, hidden, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, dim, bias=False),
                nn.Dropout(dropout),
            ),
        })

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Novel component 2: linear attention (ELU kernel) ──────────────────
    def _linear_attn(self, x, block):
        B, N, C = x.shape
        H, d = self.num_heads, C // self.num_heads

        qkv = block["qkv"](block["norm1"](x))
        qkv = qkv.reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                    # B × H × N × d

        # ELU feature map — replaces softmax, O(N) complexity
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=2, keepdim=True) + 1e-6)

        ctx = torch.matmul(k.transpose(-2, -1), v) # B × H × d × d
        out = torch.matmul(q, ctx)                 # B × H × N × d
        out = out.transpose(1, 2).reshape(B, N, C)
        return block["proj"](block["drop"](out))

    def forward(self, x, labels=None):
        B = x.shape[0]

        # 1. CNN spatial feature extraction
        x = self.stem(x)                           # B × D × H × W

        # 2. Band-Contextual Gating (BCG) — Novel component 1
        #    global average → spectral neighborhood context → sharpened gate
        s    = x.mean(dim=[2, 3])                  # B × D
        s    = self.spec_context(
                   s.unsqueeze(1)).squeeze(1)      # B × D  (band context)
        logit = self.spec_fc2(
                    F.silu(self.spec_fc1(s)))      # B × D
        temp  = self.log_temp.exp().clamp(max=10.0)
        gate  = torch.sigmoid(logit * temp)        # B × D
        x     = x * gate.unsqueeze(-1).unsqueeze(-1)

        # 3. Tokenise + positional encoding
        x = x.flatten(2).transpose(1, 2)          # B × N × D
        x = self.pos_drop(x + self.pos)

        # 4. Linear attention transformer blocks — Novel component 2
        for blk in self.blocks:
            x = x + self._linear_attn(x, blk)
            x = x + blk["mlp"](blk["norm2"](x))

        # 5. Classify
        logits = self.head(self.norm(x.mean(dim=1)))

        if labels is not None:
            return {"loss": F.cross_entropy(logits, labels),
                    "logits": logits}
        return {"logits": logits}