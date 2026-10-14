"""A small, explicit ViT for 32x32 images. No pretrained weights."""
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        self.heads, self.head_dim, self.dropout = heads, dim // heads, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        # Image classification uses bidirectional, not causal, attention.
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        x = x.transpose(1, 2).reshape(batch, tokens, dim)
        return self.out_dropout(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, dropout)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SmallViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, dim=192, depth=6, heads=3,
                 mlp_ratio=4, dropout=0.1, num_classes=10):
        super().__init__()
        if patch_size <= 0 or image_size % patch_size:
            raise ValueError('patch_size must divide image_size')
        if depth < 1 or heads < 1 or dim < 1 or dim % heads:
            raise ValueError('invalid transformer dimensions')
        self.image_size = image_size
        # This convolution is a shared linear map over non-overlapping patches.
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        tokens = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, tokens + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*(Block(dim, heads, mlp_ratio, dropout) for _ in range(depth)))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f'expected [batch, 3, {self.image_size}, {self.image_size}]')
        x = self.patch_embedding(images).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.dropout(x + self.position_embedding)
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])
