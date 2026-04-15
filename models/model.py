import torch
import torch.nn as nn
import torch.nn.functional as F


class BandRoPE(nn.Module):
    def __init__(self, head_dim, max_len=512):
        super().__init__()
        assert head_dim % 2 == 0
        half  = head_dim // 2
        theta = 1.0 / (10000 ** (torch.arange(0, half).float() / half))
        pos   = torch.arange(max_len).float()
        freqs = torch.outer(pos, theta)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x, N):
        # x: B × N × D
        cos = torch.cat([self.cos[:N]] * 2, dim=-1).unsqueeze(0)
        sin = torch.cat([self.sin[:N]] * 2, dim=-1).unsqueeze(0)
        t1, t2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-t2, t1], dim=-1) * sin


class SpectralSpatialLinearTransformerV2(nn.Module):
    """
    CNN-Transformer HSI classifier.

    Novel contributions:
      1. Band-Contextual Gating (BCG): Conv1d spectral neighborhood context
         + learnable temperature sharpening — extends SE-Net for HSI
      2. Spectral summary token: BCG output injected as a learned token
         that attends jointly with spatial tokens (spectral-spatial bridge)
      3. Linear attention (ELU kernel) + single-pass RoPE positional encoding
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
        group_size:    int   = 7,      # pipeline compat
        fusion_every:  int   = 999,    # pipeline compat
        stem_channels: int   = 8,      # pipeline compat
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.num_spat  = image_size * image_size

        # ── CNN stem ──────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1,
                      groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

        # ── Novel 1: Band-Contextual Gating (BCG) ────────────────────────
        self.spec_context = nn.Conv1d(1, 1, kernel_size=7,
                                      padding=3, bias=False)
        self.spec_fc1  = nn.Linear(embed_dim, embed_dim // 4, bias=False)
        self.spec_fc2  = nn.Linear(embed_dim // 4, embed_dim, bias=False)
        self.log_temp  = nn.Parameter(torch.zeros(embed_dim))

        # ── Novel 2: Spectral summary token ──────────────────────────────
        self.spec_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Positional encoding over [spec_token | spatial tokens] ────────
        self.pos      = nn.Parameter(
            torch.zeros(1, 1 + self.num_spat, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # ── Novel 3: Single-pass RoPE (applied once before blocks) ────────
        self.rope = BandRoPE(embed_dim,
                             max_len=1 + self.num_spat + 16)

        # ── Transformer blocks (linear attention) ────────────────────────
        self.blocks = nn.ModuleList([
            self._make_block(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # ── Head ──────────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────
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
        nn.init.trunc_normal_(self.pos,        std=0.02)
        nn.init.trunc_normal_(self.spec_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Linear attention (ELU kernel, O(N)) ───────────────────────────────
    def _linear_attn(self, x, block):
        B, N, C = x.shape
        H, d = self.num_heads, C // self.num_heads

        qkv = block["qkv"](block["norm1"](x))
        qkv = qkv.reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                    # B × H × N × d

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=2, keepdim=True) + 1e-6)

        ctx = torch.matmul(k.transpose(-2, -1), v) # B × H × d × d
        out = torch.matmul(q, ctx)                 # B × H × N × d
        out = out.transpose(1, 2).reshape(B, N, C)
        return block["proj"](block["drop"](out))

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x, labels=None):
        B = x.shape[0]

        # 1. CNN spatial feature extraction
        x = self.stem(x)                            # B × D × H × W

        # 2. Band-Contextual Gating (BCG)
        s     = x.mean(dim=[2, 3])                  # B × D
        s     = self.spec_context(s.unsqueeze(1)).squeeze(1)
        logit = self.spec_fc2(F.silu(self.spec_fc1(s)))
        temp  = self.log_temp.exp().clamp(max=10.0)
        gate  = torch.sigmoid(logit * temp)         # B × D
        x     = x * gate.unsqueeze(-1).unsqueeze(-1)

        # 3. Spectral summary token (BCG-gated mean → learnable token)
        spec_summary = x.mean(dim=[2, 3])           # B × D
        spec_tok     = (self.spec_token.expand(B, -1, -1)
                        + spec_summary.unsqueeze(1)) # B × 1 × D

        # 4. Spatial tokens
        x_spat  = x.flatten(2).transpose(1, 2)     # B × N × D

        # 5. Concatenate [spec_token | spatial] + learned position
        tokens  = torch.cat([spec_tok, x_spat], dim=1)  # B × (1+N) × D
        tokens  = self.pos_drop(tokens + self.pos)

        # 6. Single-pass RoPE — applied once, not per block
        tokens  = self.rope(tokens, tokens.shape[1])

        # 7. Linear attention transformer blocks
        for blk in self.blocks:
            tokens = tokens + self._linear_attn(tokens, blk)
            tokens = tokens + blk["mlp"](blk["norm2"](tokens))

        tokens = self.norm(tokens)

        # 8. Spectral token + spatial mean → addition → classify
        spec_out = tokens[:, 0]                     # B × D
        spat_out = tokens[:, 1:].mean(dim=1)        # B × D
        features = spec_out + spat_out              # B × D (for visualization)
        logits   = self.head(features)

        if labels is not None:
            return {"loss": F.cross_entropy(logits, labels),
                    "logits": logits, "features": features}
        return {"logits": logits, "features": features}