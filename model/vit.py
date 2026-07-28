"""Vision Transformer models for CIFAR-sized images."""
from __future__ import annotations

import torch
from torch import nn


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, dim: int) -> None:
        super().__init__()
        patch_dim = 3 * patch_size**2
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.norm_in = nn.LayerNorm(patch_dim)
        self.projection = nn.Linear(patch_dim, dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unfold(x).transpose(1, 2)
        return self.norm_out(self.projection(self.norm_in(x)))


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        batch_size, num_tokens, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            tensor.reshape(batch_size, num_tokens, self.heads, -1).transpose(1, 2)
            for tensor in qkv
        )
        attention = self.dropout(self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale))
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).reshape(batch_size, num_tokens, -1)
        return self.to_out(output)


class _Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _Attention(dim, heads, dim_head, dropout),
                        _FeedForward(dim, mlp_dim, dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention, feed_forward in self.layers:
            x = attention(x) + x
            x = feed_forward(x) + x
        return x


class ViTSmall(nn.Module):
    """ViT-small classifier configured for 32x32 CIFAR images."""

    image_size = 32
    patch_size = 4

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        dim = 512
        num_patches = (self.image_size // self.patch_size) ** 2
        self.patch_embedding = _PatchEmbedding(self.patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.embedding_dropout = nn.Dropout(0.1)
        self.transformer = _Transformer(
            dim=dim,
            depth=6,
            heads=8,
            dim_head=64,
            mlp_dim=512,
            dropout=0.1,
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(
                "ViT-small expects input shape [B, 3, 32, 32] for CIFAR images; "
                f"received {tuple(x.shape)}."
            )
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.embedding_dropout(x + self.pos_embedding)
        x = self.transformer(x)
        return self.head(x[:, 0])


def vit_small(num_classes: int = 10) -> nn.Module:
    """Create a randomly initialized ViT-small classifier for CIFAR images."""
    return ViTSmall(num_classes=num_classes)


__all__ = ["ViTSmall", "vit_small"]
