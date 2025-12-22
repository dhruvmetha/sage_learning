"""
DiT with Cropped Output and Separate Context Conditioning

This variant separates the noisy input from context:
- Noisy input: (B, 1, crop_size, crop_size) - small crop around target center
- Context: (B, context_channels, context_size, context_size) - full resolution scene
- Output: (B, 1, crop_size, crop_size) - denoised prediction

The context is processed by a CNN encoder and injected via AdaLN conditioning,
while the transformer only operates on the small cropped region.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import sinusoidal_embedding, AdaLN, TimeEmbedding, PatchEmbed, TransformerBlockAdaLN


class ContextEncoder(nn.Module):
    """
    CNN encoder for full-resolution context channels.

    Processes the full scene context (64x64) and produces a conditioning
    embedding for AdaLN injection into the transformer.

    Args:
        in_channels: Number of context channels (default 5)
        context_size: Spatial size of context input (default 64)
        embed_dim: Output embedding dimension
        zero_init: Initialize final projection to near-zero
    """

    def __init__(
        self,
        in_channels: int = 5,
        context_size: int = 64,
        embed_dim: int = 256,
        zero_init: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.context_size = context_size

        # Progressive downsampling CNN
        # 64 -> 32 -> 16 -> 8 -> 4 -> global pool
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 64->32
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32->16
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16->8
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8->4
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),  # 4->1
            nn.Flatten(),  # (B, 256)
        )

        self.proj = nn.Linear(256, embed_dim)

        # Zero-init for smooth training start
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, in_channels, context_size, context_size)

        Returns:
            Conditioning embedding (B, embed_dim)
        """
        features = self.conv(context)
        return self.proj(features)


class DiTCropped(nn.Module):
    """
    DiT variant with cropped output and separate context conditioning.

    Architecture:
    - Noisy input (B, 1, crop_size, crop_size) -> PatchEmbed -> Transformer -> Output
    - Context (B, 5, 64, 64) -> ContextEncoder -> cond_emb
    - Time -> TimeEmbedding -> t_emb
    - cond_emb = t_emb + context_emb -> AdaLN parameters

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

        # Learned positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)

        # Time embedding
        self.time_mlp = TimeEmbedding(dim)

        # Context encoder (processes full 64x64 context)
        self.context_encoder = ContextEncoder(
            in_channels=context_channels,
            context_size=context_size,
            embed_dim=dim,
            zero_init=True,
        )

        # Transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlockAdaLN(dim, heads) for _ in range(depth)
        ])

        # AdaLN projections
        self.ada_proj = nn.ModuleList([])
        for _ in range(depth):
            proj = nn.Linear(dim, dim * 4)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.ada_proj.append(proj)

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

        # Context embedding
        context_emb = self.context_encoder(context)  # (B, D)

        # Combined conditioning
        cond_emb = t_emb + context_emb

        # Transformer with AdaLN
        for blk, proj in zip(self.blocks, self.ada_proj):
            gammas_betas = proj(cond_emb)
            g1, b1, g2, b2 = gammas_betas.chunk(4, dim=-1)
            tok = blk(tok, g1, b1, g2, b2)

        tok = self.norm(tok)

        # Un-patchify back to image
        H = W = int(math.sqrt(tok.size(1)))
        tok = tok.transpose(1, 2).reshape(B, -1, H, W)
        return self.unpatch(tok)
