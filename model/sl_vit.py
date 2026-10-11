"""CIFAR SL-ViT from https://github.com/aanna0701/SPT_LSA_ViT.

Architecture follows revision 54bd796acd28fa11deac21a97b14b154379b34e7.
Native PyTorch patchification replaces einops; parameter names match upstream.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class PatchShifting(nn.Module):
    """Concatenate the image and four zero-padded diagonal half-patch shifts."""

    def __init__(self, patch_size: int = 4) -> None:
        super().__init__()
        self.shift = patch_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.shift
        padded = F.pad(x, (s, s, s, s))
        return torch.cat((x, padded[:, :, :-2*s, :-2*s],
                          padded[:, :, :-2*s, 2*s:], padded[:, :, 2*s:, :-2*s],
                          padded[:, :, 2*s:, 2*s:]), dim=1)


class _Patchify(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        # Upstream flattens patch rows, patch columns, then channels (not CHW).
        return x.reshape(b, c, h//p, p, w//p, p).permute(0, 2, 4, 3, 5, 1).reshape(b, (h//p)*(w//p), p*p*c)


class ShiftedPatchTokenization(nn.Module):
    def __init__(self, dim: int = 192, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_shifting = PatchShifting(patch_size)
        patch_dim = 3 * 5 * patch_size**2
        self.merging = nn.Sequential(_Patchify(patch_size), nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merging(self.patch_shifting(x))


class LocalitySelfAttention(nn.Module):
    def __init__(self, dim: int = 192, heads: int = 12, num_tokens: int = 65) -> None:
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, 3*dim, bias=False)
        # Upstream initializes QKV once here and again in the model-wide pass.
        _init_weights(self.to_qkv)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.0))
        self.scale = nn.Parameter(torch.full((heads,), (dim//heads)**-0.5))
        self.register_buffer("mask", torch.eye(num_tokens, dtype=torch.bool), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, dim = x.shape
        q, k, v = (t.reshape(b, n, self.heads, -1).transpose(1, 2) for t in self.to_qkv(x).chunk(3, dim=-1))
        scores = (q @ k.transpose(-1, -2)) * self.scale[None, :, None, None]
        scores = scores.masked_fill(self.mask, torch.finfo(scores.dtype).min)
        x = scores.softmax(dim=-1) @ v
        return self.to_out(x.transpose(1, 2).reshape(b, n, dim))


class _PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class _FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Dropout(0.0),
                                 nn.Linear(2*dim, dim), nn.Dropout(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = (keep + torch.rand((x.shape[0], 1, 1), dtype=x.dtype, device=x.device)).floor()
        return x / keep * mask


class _Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, stochastic_depth: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([_PreNorm(dim, LocalitySelfAttention(dim, heads)), _PreNorm(dim, _FeedForward(dim))])
            for _ in range(depth)
        ])
        # The author uses a constant rate in every block, not a depth-wise ramp.
        self.drop_path = _DropPath(stochastic_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention, feed_forward in self.layers:
            x = x + self.drop_path(attention(x))
            x = x + self.drop_path(feed_forward(x))
        return x


class SLViT(nn.Module):
    """Official 32x32 SL-ViT: patch 4, width 192, depth 9, 12 heads, MLP ratio 2."""

    image_size = 32
    patch_size = 4

    def __init__(self, num_classes: int = 10, stochastic_depth: float = 0.1) -> None:
        super().__init__()
        if num_classes <= 0 or not 0 <= stochastic_depth < 1:
            raise ValueError("num_classes must be positive and stochastic_depth must be in [0, 1).")
        dim = 192
        self.num_classes = num_classes
        self.to_patch_embedding = ShiftedPatchTokenization(dim, self.patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, 65, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(0.0)
        self.transformer = _Transformer(dim, 9, 12, stochastic_depth)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.apply(_init_weights)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError(f"SL-ViT expects [B, 3, 32, 32], received {tuple(images.shape)}.")
        x = self.to_patch_embedding(images)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.dropout(x + self.pos_embedding)
        return self.mlp_head(self.transformer(x)[:, 0])


def sl_vit(num_classes: int = 10, stochastic_depth: float = 0.1) -> nn.Module:
    return SLViT(num_classes=num_classes, stochastic_depth=stochastic_depth)


__all__ = ["SLViT", "sl_vit"]
