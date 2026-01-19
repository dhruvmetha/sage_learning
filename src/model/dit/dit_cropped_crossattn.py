from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Return (B, dim) sinusoidal embeddings for normalized timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, patch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.patch = patch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class ContextEncoder(nn.Module):
    def __init__(self, in_ch: int, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, dim, 3, 1, 1), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv(x)
        return feats.flatten(2).transpose(1, 2)


class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.time_proj = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor, context: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]

        h = self.norm2(x)
        x = x + self.cross_attn(h, context, context, need_weights=False)[0]

        h = self.norm3(x)
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = x + self.mlp(h)
        return x


class DiTCroppedCrossAttn(nn.Module):
    """
    Cross-attention DiT for cropped target prediction.

    Target (noisy) is tokenized from a small crop, while context comes from a
    larger local crop encoded into tokens for cross-attention.
    """

    def __init__(
        self,
        crop_size: int = 32,
        context_size: int = 64,
        patch: int = 4,
        in_ch: int = 1,
        context_channels: int = 5,
        out_ch: int = 1,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        if crop_size % patch != 0:
            raise ValueError("crop_size must be divisible by patch.")
        if context_size % 8 != 0:
            raise ValueError("context_size must be divisible by 8 for context encoder.")

        self.crop_size = crop_size
        self.context_size = context_size

        self.patch_embed = PatchEmbed(in_ch, patch, dim)
        num_patches = (crop_size // patch) ** 2
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)

        self.context_encoder = ContextEncoder(context_channels, dim)
        num_ctx_tokens = (context_size // 8) ** 2
        self.ctx_pos_emb = nn.Parameter(torch.randn(1, num_ctx_tokens, dim) * 0.02)

        self.time_mlp = TimeEmbedding(dim)

        self.blocks = nn.ModuleList(
            [CrossAttnBlock(dim, heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.unpatch = nn.ConvTranspose2d(dim, out_ch, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tok = self.patch_embed(x) + self.pos_emb

        ctx = self.context_encoder(context)
        ctx = ctx + self.ctx_pos_emb

        t_emb = self.time_mlp(sinusoidal_embedding(t, tok.size(-1)))

        for block in self.blocks:
            tok = block(tok, ctx, t_emb)

        tok = self.norm(tok)
        h = w = int(math.sqrt(tok.size(1)))
        tok = tok.transpose(1, 2).reshape(B, -1, h, w)
        return self.unpatch(tok)
