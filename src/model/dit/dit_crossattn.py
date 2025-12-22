"""
DiT with Cross-Attention Context Conditioning (Full Resolution)

This variant uses cross-attention instead of AdaLN for context conditioning:
- Input: (B, in_ch, img_size, img_size) - concatenated [context, coord_grid, noisy_target]
- Output: (B, out_ch, img_size, img_size) - denoised prediction

The context channels are extracted and encoded separately to produce spatial
features for cross-attention, while the noisy target goes through the transformer.
Time conditioning uses AdaLN-style modulation.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import sinusoidal_embedding, TimeEmbedding, PatchEmbed


class SpatialContextEncoderFull(nn.Module):
    """
    CNN encoder that preserves spatial structure for cross-attention.

    Processes context channels and produces spatial features for cross-attention.

    Args:
        in_channels: Number of context channels (default 5)
        img_size: Spatial size of input (default 64)
        embed_dim: Output embedding dimension
        downsample_factor: How much to downsample spatially (default 4)
    """

    def __init__(
        self,
        in_channels: int = 5,
        img_size: int = 64,
        embed_dim: int = 256,
        downsample_factor: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.img_size = img_size
        self.downsample_factor = downsample_factor

        # CNN to process context
        # For img_size=64, factor=4: 64 -> 32 -> 16
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # /2
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # /4
            nn.SiLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=1, padding=1),
        )

        self.output_size = img_size // downsample_factor

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, in_channels, img_size, img_size)

        Returns:
            Spatial features as sequence: (B, N_ctx, embed_dim)
        """
        features = self.conv(context)  # (B, embed_dim, H', W')
        return features.flatten(2).transpose(1, 2)  # (B, N_ctx, embed_dim)


class TransformerBlockCrossAttnFull(nn.Module):
    """
    Transformer block with self-attention, cross-attention, and MLP.

    Time conditioning is applied via scale/shift modulation.
    """

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: int = 4):
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        # Cross-attention to context
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        # MLP
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

        # Time modulation (scale and shift)
        self.time_proj = nn.Linear(dim, dim * 2)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tokens (B, N, D)
            context: Context features (B, N_ctx, D)
            t_emb: Time embedding (B, D)
        """
        # Self-attention
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h)[0]

        # Cross-attention to context
        h = self.norm2(x)
        x = x + self.cross_attn(h, context, context)[0]

        # MLP with time modulation
        h = self.norm3(x)
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = x + self.mlp(h)

        return x


class DiTCrossAttn(nn.Module):
    """
    DiT with cross-attention context conditioning (full resolution).

    Unlike the standard DiT which concatenates context with noisy input,
    this version:
    1. Extracts context channels and encodes them separately
    2. Uses cross-attention to inject context information
    3. Only processes the noisy target through the main transformer

    Args:
        img_size: Image size (default 64)
        patch: Patch size for tokenization (default 4)
        in_ch: Total input channels (context + coord_grid + noisy_target)
        context_channels: Number of context channels to extract (default 5)
        dim: Model dimension (default 256)
        depth: Number of transformer blocks (default 8)
        heads: Number of attention heads (default 8)
        out_ch: Output channels (default 1)
    """

    def __init__(
        self,
        img_size: int = 64,
        patch: int = 4,
        in_ch: int = 8,  # 5 context + 2 coord_grid + 1 noisy
        context_channels: int = 5,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        out_ch: int = 1,
    ) -> None:
        super().__init__()

        assert img_size % patch == 0, "Image size must be divisible by patch size."

        self.img_size = img_size
        self.patch = patch
        self.context_channels = context_channels
        # Channels for noisy input: total - context = coord_grid + noisy_target
        self.noisy_channels = in_ch - context_channels

        # Patch embedding for non-context channels (coord_grid + noisy_target)
        self.patch_embed = PatchEmbed(self.noisy_channels, patch, dim)
        num_patches = (img_size // patch) ** 2

        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)

        # Time embedding
        self.time_mlp = TimeEmbedding(dim)

        # Context encoder (produces spatial features for cross-attention)
        self.context_encoder = SpatialContextEncoderFull(
            in_channels=context_channels,
            img_size=img_size,
            embed_dim=dim,
            downsample_factor=4,  # 64 -> 16x16 = 256 context tokens
        )

        # Positional embeddings for context tokens
        ctx_tokens = (img_size // 4) ** 2
        self.ctx_pos_emb = nn.Parameter(torch.randn(1, ctx_tokens, dim) * 0.02)

        # Transformer stack with cross-attention
        self.blocks = nn.ModuleList([
            TransformerBlockCrossAttnFull(dim, heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.unpatch = nn.ConvTranspose2d(dim, out_ch, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, in_ch, img_size, img_size)
               First context_channels are context, rest is coord_grid + noisy_target
            t: Timesteps (B,)

        Returns:
            Prediction (B, out_ch, img_size, img_size)
        """
        B = x.size(0)

        # Split input into context and noisy parts
        context = x[:, :self.context_channels]  # (B, 5, H, W)
        noisy = x[:, self.context_channels:]     # (B, 3, H, W) - coord_grid + noisy

        # Tokenize noisy input
        tok = self.patch_embed(noisy) + self.pos_emb  # (B, N, D)

        # Time embedding
        t_emb = self.time_mlp(sinusoidal_embedding(t, tok.size(-1)))  # (B, D)

        # Context features for cross-attention
        ctx = self.context_encoder(context) + self.ctx_pos_emb  # (B, N_ctx, D)

        # Transformer with cross-attention
        for blk in self.blocks:
            tok = blk(tok, ctx, t_emb)

        tok = self.norm(tok)

        # Un-patchify
        H = W = int(math.sqrt(tok.size(1)))
        tok = tok.transpose(1, 2).reshape(B, -1, H, W)
        return self.unpatch(tok)
