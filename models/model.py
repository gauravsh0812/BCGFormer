import torch
import torch.nn as nn


class BandWeightedPooling(nn.Module):
    """Learnable spectral-band weighting for global token aggregation."""
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        w = torch.softmax(self.weights, dim=0)
        return (x * w).sum(dim=1)


class SpectralSpatialLinearAttention(nn.Module):
    """Linear attention with explicit spectral gating. O(N) complexity."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
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
        
        gate = self.spectral_gate(x)
        out = out * gate
        
        return self.proj(out)


class GlobalAttentionBlock(nn.Module):
    """Single full-attention block to restore long-range interactions."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x)
        out, _ = self.attn(h, h, h)
        return x + out


class SpectralSpatialViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpectralSpatialLinearAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpectralSpatialLinearTransformer(nn.Module):
    """Spectral-Spatial Linear Transformer for Hyperspectral Image Classification"""
    def __init__(
        self,
        image_size=5,
        patch_size=1,
        num_channels=103,
        num_classes=9,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        
        self.patch_embed = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches_h = (image_size - patch_size) // patch_size + 1
        num_patches_w = (image_size - patch_size) // patch_size + 1
        num_patches = num_patches_h * num_patches_w
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        self.blocks = nn.ModuleList([
            SpectralSpatialViTBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.global_block = GlobalAttentionBlock(embed_dim, num_heads)
        self.spectral_pool = BandWeightedPooling(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, labels=None):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.global_block(x)
        x = self.spectral_pool(x)
        x = self.norm(x)
        logits = self.head(x)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
