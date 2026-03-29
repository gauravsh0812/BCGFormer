import torch
import torch.nn as nn


class LightweightSpectralStem(nn.Module):
    """
    Lightweight depthwise 3D conv stem to replace raw Conv2d patch embed.
    Captures local spectral-spatial correlations like HiT's SA-3DCP but cheaper.
    Two parallel branches with different spectral kernel sizes, fused via 1x1 Conv2d.
    """
    def __init__(self, num_channels: int, embed_dim: int):
        super().__init__()
        mid = embed_dim // 2

        # Fine-grained spectral branch
        self.branch1 = nn.Sequential(
            nn.Conv3d(1, mid // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(mid // 2),
            nn.GELU(),
        )
        # Coarse spectral branch
        self.branch2 = nn.Sequential(
            nn.Conv3d(1, mid // 2, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(mid // 2),
            nn.GELU(),
        )
        # Fuse branches → embed_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(mid * num_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.num_channels = num_channels

    def forward(self, x):
        # x: B x C x H x W
        B, C, H, W = x.shape
        x3d = x.unsqueeze(1)                        # B x 1 x C x H x W
        b1 = self.branch1(x3d)                      # B x mid//2 x C x H x W
        b2 = self.branch2(x3d)                      # B x mid//2 x C x H x W
        out = torch.cat([b1, b2], dim=1)            # B x mid x C x H x W
        out = out.reshape(B, -1, H, W)              # B x (mid*C) x H x W
        return self.fusion(out)                     # B x embed_dim x H x W


class SpectralSpatialLinearAttention(nn.Module):
    """
    O(N) linear attention with spectral gating applied on attended features.
    Fix: gate now uses the attended output (not raw x) for meaningful gating.
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        # Gate on attended output, not raw x
        self.spectral_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        k = k.softmax(dim=1)
        context = torch.einsum('bnhd,bnhv->bhdv', k, v)
        out = torch.einsum('bnhd,bhdv->bnhv', q, context)
        out = out.reshape(B, N, C)
        out = self.attn_drop(out)

        # Gate using attended output instead of raw x
        gate = self.spectral_gate(out)
        out = out * gate

        return self.proj(out)


class SpectralSpatialViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpectralSpatialLinearAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpectralSpatialLinearTransformerV2(nn.Module):
    """
    SSLT v2 — Improved Spectral-Spatial Linear Transformer.

    Changes over v1:
    - LightweightSpectralStem replaces raw Conv2d patch embed (3D spectral-spatial features)
    - CLS token replaces BandWeightedPooling (more stable global aggregation)
    - Spectral gate now applied on attended output, not raw input
    - Dropout added throughout (attention, MLP, positional)
    - pos_embed initialized with trunc_normal instead of zeros
    - GlobalAttentionBlock removed (O(N^2), ablation showed minimal benefit, hurts latency)
    - O(N) linear attention blocks preserved for lowest latency
    """
    def __init__(
        self,
        image_size: int = 5,
        patch_size: int = 1,
        num_channels: int = 103,
        num_classes: int = 9,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # 3D spectral-spatial stem (no striding, always outputs image_size x image_size)
        self.patch_embed = LightweightSpectralStem(num_channels, embed_dim)

        num_patches = image_size * image_size

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SpectralSpatialViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, labels=None):
        B = x.shape[0]

        # 3D spectral-spatial stem → tokens
        x = self.patch_embed(x)                     # B x embed_dim x H x W
        x = x.flatten(2).transpose(1, 2)            # B x N x embed_dim

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)              # B x (N+1) x embed_dim
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        logits = self.head(x[:, 0])                 # CLS token

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
