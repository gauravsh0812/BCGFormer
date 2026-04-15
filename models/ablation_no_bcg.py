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
        cos = torch.cat([self.cos[:N]] * 2, dim=-1).unsqueeze(0)
        sin = torch.cat([self.sin[:N]] * 2, dim=-1).unsqueeze(0)
        t1, t2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-t2, t1], dim=-1) * sin


class SpectralSpatialLinearTransformerV2_NoBCG(nn.Module):
    """Ablation: WITHOUT Band-Contextual Gating (BCG)"""
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
        group_size:    int   = 7,
        fusion_every:  int   = 999,
        stem_channels: int   = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.num_spat  = image_size * image_size

        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1,
                      groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

        # BCG components removed

        self.spec_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.num_spat, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.rope = BandRoPE(embed_dim, max_len=1 + self.num_spat + 16)

        self.blocks = nn.ModuleList([
            self._make_block(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

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
        nn.init.trunc_normal_(self.spec_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _linear_attn(self, x, block):
        B, N, C = x.shape
        H, d = self.num_heads, C // self.num_heads

        qkv = block["qkv"](block["norm1"](x))
        qkv = qkv.reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        k = k / (k.sum(dim=2, keepdim=True) + 1e-6)

        ctx = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, ctx)
        out = out.transpose(1, 2).reshape(B, N, C)
        return block["proj"](block["drop"](out))

    def forward(self, x, labels=None):
        B = x.shape[0]

        x = self.stem(x)  # No BCG gating applied

        spec_summary = x.mean(dim=[2, 3])
        spec_tok = self.spec_token.expand(B, -1, -1) + spec_summary.unsqueeze(1)

        x_spat = x.flatten(2).transpose(1, 2)
        tokens = torch.cat([spec_tok, x_spat], dim=1)
        tokens = self.pos_drop(tokens + self.pos)
        tokens = self.rope(tokens, tokens.shape[1])

        for blk in self.blocks:
            tokens = tokens + self._linear_attn(tokens, blk)
            tokens = tokens + blk["mlp"](blk["norm2"](tokens))

        tokens = self.norm(tokens)
        spec_out = tokens[:, 0]
        spat_out = tokens[:, 1:].mean(dim=1)
        logits = self.head(spec_out + spat_out)

        if labels is not None:
            return {"loss": F.cross_entropy(logits, labels), "logits": logits}
        return {"logits": logits}
