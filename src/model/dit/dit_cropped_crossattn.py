"""
DiT with Cropped Output and Cross-Attention Context Conditioning

This variant uses cross-attention instead of AdaLN for context conditioning:
- Noisy input: (B, 1, crop_size, crop_size) - small crop around target center
- Context: (B, context_channels, context_size, context_size) - full resolution scene
- Output: (B, 1, crop_size, crop_size) - denoised prediction

The context is processed by a CNN encoder to produce spatial features,
which are then used as keys/values in cross-attention layers.
Time conditioning still uses AdaLN.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import sinusoidal_embedding, TimeEmbedding, PatchEmbed


class SpatialContextEncoder(nn.Module):
    """
    CNN encoder that preserves spatial structure for cross-attention.

    Unlike the pooling-based encoder, this produces a sequence of spatial
    features that can be used as keys/values in cross-attention.

    Args:
        in_channels: Number of context channels (default 5)
        context_size: Spatial size of context input (default 64)
        embed_dim: Output embedding dimension
        downsample_factor: How much to downsample spatially (default 8: 64->8)
    """

    def __init__(
        self,
        in_channels: int = 5,
        context_size: int = 64,
        embed_dim: int = 256,
        downsample_factor: int = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.context_size = context_size
        self.downsample_factor = downsample_factor

        # Progressive downsampling CNN
        # 64 -> 32 -> 16 -> 8 (with downsample_factor=8)
        layers = []
        ch = in_channels
        out_channels = [32, 64, 128]
        current_size = context_size

        for out_ch in out_channels:
            if current_size > context_size // downsample_factor:
                layers.extend([
                    nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.SiLU(),
                ])
                current_size //= 2
                ch = out_ch

        # Final conv to embed_dim (no stride)
        layers.append(nn.Conv2d(ch, embed_dim, kernel_size=3, stride=1, padding=1))

        self.conv = nn.Sequential(*layers)

        # Spatial size after encoding
        self.output_size = context_size // downsample_factor

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, in_channels, context_size, context_size)

        Returns:
            Spatial features as sequence: (B, N_ctx, embed_dim)
            where N_ctx = (context_size // downsample_factor)^2
        """
        features = self.conv(context)  # (B, embed_dim, H', W')
        B, C, H, W = features.shape
        # Flatten spatial dims to sequence
        return features.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)


class TransformerBlockCrossAttn(nn.Module):
    """
    Transformer block with self-attention, cross-attention to context, and MLP.

    Uses standard LayerNorm (not AdaLN) for the attention layers,
    but can optionally use AdaLN for time conditioning on the MLP.

    Architecture:
        x -> LN -> Self-Attn -> + -> LN -> Cross-Attn(Q=x, KV=ctx) -> + -> LN -> MLP -> +
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

        # Time modulation for MLP (scale and shift after norm3)
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

        Returns:
            Output tokens (B, N, D)
        """
        # Self-attention
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h)[0]

        # Cross-attention to context
        h = self.norm2(x)
        x = x + self.cross_attn(h, context, context)[0]

        # MLP with time modulation
        h = self.norm3(x)
        # Apply time-dependent scale and shift
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = x + self.mlp(h)

        return x


class DiTCroppedCrossAttn(nn.Module):
    """
    DiT variant with cropped output and cross-attention context conditioning.

    Architecture:
    - Context (B, 5, 64, 64) -> SpatialContextEncoder -> (B, N_ctx, D)
    - Noisy input (B, 1, crop_size, crop_size) -> PatchEmbed -> (B, N, D)
    - Time -> TimeEmbedding -> (B, D)
    - Each block: Self-Attn -> Cross-Attn(Q=tokens, KV=context) -> MLP

    Args:
        crop_size: Size of cropped input/output (default 24)
        context_size: Size of full context input (default 64)
        patch: Patch size for tokenization (default 4)
        in_ch: Input channels for noisy input (default 1)
        context_channels: Number of context channels (default 5)
        dim: Model dimension (default 256)
        depth: Number of transformer blocks (default 8)
        heads: Number of attention heads (default 8)
        out_ch: Output channels (default 1)
    """

    def __init__(
        self,
        crop_size: int = 24,
        context_size: int = 64,
        patch: int = 4,
        in_ch: int = 1,
        context_channels: int = 5,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        out_ch: int = 1,
    ) -> None:
        super().__init__()

        assert crop_size % patch == 0, "Crop size must be divisible by patch size."

        self.crop_size = crop_size
        self.context_size = context_size
        self.context_channels = context_channels

        # Patch embedding for noisy input only
        self.patch_embed = PatchEmbed(in_ch, patch, dim)
        num_patches = (crop_size // patch) ** 2  # e.g., 36 for 24x24 with patch=4

        # Learned positional embeddings for noisy input tokens
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)

        # Time embedding
        self.time_mlp = TimeEmbedding(dim)

        # Context encoder (produces spatial features for cross-attention)
        self.context_encoder = SpatialContextEncoder(
            in_channels=context_channels,
            context_size=context_size,
            embed_dim=dim,
            downsample_factor=8,  # 64 -> 8x8 = 64 context tokens
        )

        # Learned positional embeddings for context tokens
        ctx_tokens = (context_size // 8) ** 2  # 64 tokens for 64x64 with factor=8
        self.ctx_pos_emb = nn.Parameter(torch.randn(1, ctx_tokens, dim) * 0.02)

        # Transformer stack with cross-attention
        self.blocks = nn.ModuleList([
            TransformerBlockCrossAttn(dim, heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.unpatch = nn.ConvTranspose2d(dim, out_ch, kernel_size=patch, stride=patch)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with separate noisy input and context.

        Args:
            x_noisy: Noisy/interpolated target, shape (B, in_ch, crop_size, crop_size)
            t: Timesteps, shape (B,)
            context: Full context channels, shape (B, context_channels, context_size, context_size)

        Returns:
            Prediction (noise or velocity), shape (B, out_ch, crop_size, crop_size)
        """
        B = x_noisy.size(0)

        # Tokenize noisy input
        tok = self.patch_embed(x_noisy) + self.pos_emb  # (B, N, D)

        # Time embedding
        t_emb = self.time_mlp(sinusoidal_embedding(t, tok.size(-1)))  # (B, D)

        # Context features for cross-attention
        ctx = self.context_encoder(context) + self.ctx_pos_emb  # (B, N_ctx, D)

        # Transformer with cross-attention
        for blk in self.blocks:
            tok = blk(tok, ctx, t_emb)

        tok = self.norm(tok)

        # Un-patchify back to image
        H = W = int(math.sqrt(tok.size(1)))
        tok = tok.transpose(1, 2).reshape(B, -1, H, W)
        return self.unpatch(tok)
