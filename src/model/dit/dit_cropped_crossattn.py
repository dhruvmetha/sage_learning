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

from .dit import sinusoidal_embedding, TimeEmbedding, PatchEmbed, AdaLN


def _token_centers(crop_size: int, context_size: int, patch: int, downsample_factor: int):
    """
    Compute physical pixel-center coordinates for query and KV tokens.

    Returns:
        q_centers: (N_q, 2) center pixel coords of each query token in context-frame.
        kv_centers: (N_kv, 2) center pixel coords of each KV token in context-frame.
    """
    q_grid = crop_size // patch
    kv_grid = context_size // downsample_factor
    offset = (context_size - crop_size) // 2  # crop starts at this pixel in context

    q_centers = torch.zeros(q_grid * q_grid, 2)
    for qi in range(q_grid):
        for qj in range(q_grid):
            q_centers[qi * q_grid + qj, 0] = offset + qi * patch + patch / 2.0
            q_centers[qi * q_grid + qj, 1] = offset + qj * patch + patch / 2.0

    kv_centers = torch.zeros(kv_grid * kv_grid, 2)
    for ki in range(kv_grid):
        for kj in range(kv_grid):
            kv_centers[ki * kv_grid + kj, 0] = ki * downsample_factor + downsample_factor / 2.0
            kv_centers[ki * kv_grid + kj, 1] = kj * downsample_factor + downsample_factor / 2.0

    return q_centers, kv_centers


class CrossAttnWithPosBias(nn.Module):
    """
    Multi-head cross-attention with a learnable relative position bias.

    The bias is added to the attention logits before softmax and is initialized
    as a 2D Gaussian peaked at the spatially-correct (Q, KV) partner pair
    (using physical pixel-center coordinates). This gives the model a strong
    "attend to your own world location" prior at init without forbidding it
    from learning a different pattern.

    Args:
        dim: Embedding dimension.
        heads: Number of attention heads.
        q_centers: (N_q, 2) pixel-center coords of Q tokens (context-frame).
        kv_centers: (N_kv, 2) pixel-center coords of KV tokens (context-frame).
        use_pos_bias: If False, behaves as standard cross-attn (no learnable bias).
        init_amp: Peak amplitude of the Gaussian bias at init.
        init_sigma: Spread of the Gaussian (in pixels).
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        q_centers: torch.Tensor,
        kv_centers: torch.Tensor,
        use_pos_bias: bool = True,
        init_amp: float = 2.0,
        init_sigma: float = 8.0,
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        if use_pos_bias:
            # Pairwise pixel distance between every Q and KV center
            diff = q_centers.unsqueeze(1) - kv_centers.unsqueeze(0)  # (N_q, N_kv, 2)
            dist2 = (diff ** 2).sum(dim=-1)                          # (N_q, N_kv)
            bias = init_amp * torch.exp(-dist2 / (2.0 * init_sigma ** 2))
            # Per-head learnable bias
            self.pos_bias = nn.Parameter(
                bias.unsqueeze(0).expand(heads, -1, -1).contiguous()
            )
        else:
            self.register_parameter("pos_bias", None)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, N_q, _ = q.shape
        _, N_kv, _ = kv.shape

        Q = self.q_proj(q).reshape(B, N_q, self.heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).reshape(B, N_kv, self.heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).reshape(B, N_kv, self.heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N_q, N_kv)
        if self.pos_bias is not None:
            scores = scores + self.pos_bias.unsqueeze(0)  # broadcast over batch

        attn = F.softmax(scores, dim=-1)
        out = attn @ V  # (B, H, N_q, head_dim)
        out = out.transpose(1, 2).reshape(B, N_q, self.dim)
        return self.out_proj(out)


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
    Transformer block with AdaLN-Zero on all three sub-layers
    (self-attn, cross-attn, MLP) and a learnable relative-position bias
    on the cross-attention.

    Architecture:
        x -> AdaLN(g1,b1) -> Self-Attn       -> +
        x -> AdaLN(g2,b2) -> Cross-Attn(pos) -> +
        x -> AdaLN(g3,b3) -> MLP             -> +

    Time conditioning is broadcast to every sub-layer (matching the DiT paper).
    Cross-attn AdaLN gamma starts at `cross_attn_init_scale` (instead of zero)
    so cross-attn contributes a nonzero gradient signal from step 1 and avoids
    the slow ramp-up of pure AdaLN-Zero.

    Args:
        dim: Embedding dimension.
        heads: Number of attention heads.
        q_centers: (N_q, 2) Q-token pixel-center coords (passed to cross-attn).
        kv_centers: (N_kv, 2) KV-token pixel-center coords.
        mlp_ratio: MLP hidden expansion factor.
        use_pos_bias: Enable learnable relative position bias on cross-attn.
        pos_bias_init_amp / pos_bias_init_sigma: Gaussian shape at init.
        cross_attn_init_scale: Initial gamma value for the cross-attn AdaLN.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        q_centers: torch.Tensor,
        kv_centers: torch.Tensor,
        mlp_ratio: int = 4,
        use_pos_bias: bool = True,
        pos_bias_init_amp: float = 2.0,
        pos_bias_init_sigma: float = 8.0,
        cross_attn_init_scale: float = 0.01,
    ):
        super().__init__()
        self.dim = dim

        # AdaLN for each of the three sub-layers
        self.adaln1 = AdaLN(dim)  # self-attn
        self.adaln2 = AdaLN(dim)  # cross-attn
        self.adaln3 = AdaLN(dim)  # mlp

        # Single Linear: t_emb -> (g1, b1, g2, b2, g3, b3)
        self.ada_proj = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.ada_proj.weight)
        nn.init.zeros_(self.ada_proj.bias)
        # Bias g2 (cross-attn gamma) to small-nonzero so cross-attn isn't dead at init.
        # Layout: [g1 | b1 | g2 | b2 | g3 | b3], each of size `dim`.
        with torch.no_grad():
            self.ada_proj.bias[2 * dim : 3 * dim].fill_(cross_attn_init_scale)

        # Sub-layers
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = CrossAttnWithPosBias(
            dim=dim,
            heads=heads,
            q_centers=q_centers,
            kv_centers=kv_centers,
            use_pos_bias=use_pos_bias,
            init_amp=pos_bias_init_amp,
            init_sigma=pos_bias_init_sigma,
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Per-block AdaLN params for all three sub-layers
        gb = self.ada_proj(t_emb)
        g1, b1, g2, b2, g3, b3 = gb.chunk(6, dim=-1)

        # Self-attn
        h = self.adaln1(x, g1, b1)
        x = x + self.self_attn(h, h, h)[0]

        # Cross-attn with learnable position bias
        h = self.adaln2(x, g2, b2)
        x = x + self.cross_attn(h, context)

        # MLP
        h = self.adaln3(x, g3, b3)
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
        ctx_downsample_factor: int = 8,
        use_pos_bias: bool = True,
        pos_bias_init_amp: float = 2.0,
        pos_bias_init_sigma: float = 8.0,
        cross_attn_init_scale: float = 0.01,
    ) -> None:
        super().__init__()

        assert crop_size % patch == 0, "Crop size must be divisible by patch size."
        assert context_size % ctx_downsample_factor == 0, \
            "Context size must be divisible by ctx_downsample_factor."

        self.crop_size = crop_size
        self.context_size = context_size
        self.context_channels = context_channels

        # Patch embedding for noisy input only
        self.patch_embed = PatchEmbed(in_ch, patch, dim)
        num_patches = (crop_size // patch) ** 2

        # Learned positional embeddings for noisy input tokens
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)

        # Time embedding
        self.time_mlp = TimeEmbedding(dim)

        # Context encoder (produces spatial features for cross-attention)
        self.context_encoder = SpatialContextEncoder(
            in_channels=context_channels,
            context_size=context_size,
            embed_dim=dim,
            downsample_factor=ctx_downsample_factor,
        )

        # Learned positional embeddings for context tokens
        ctx_tokens = (context_size // ctx_downsample_factor) ** 2
        self.ctx_pos_emb = nn.Parameter(torch.randn(1, ctx_tokens, dim) * 0.02)

        # Precompute pixel-center coords of Q and KV tokens in the context frame.
        # Used to initialize the cross-attn position bias so each Q starts with a
        # strong prior toward the KV cell covering its same world location.
        q_centers, kv_centers = _token_centers(
            crop_size=crop_size,
            context_size=context_size,
            patch=patch,
            downsample_factor=ctx_downsample_factor,
        )

        # Transformer stack with cross-attention
        self.blocks = nn.ModuleList([
            TransformerBlockCrossAttn(
                dim=dim,
                heads=heads,
                q_centers=q_centers,
                kv_centers=kv_centers,
                use_pos_bias=use_pos_bias,
                pos_bias_init_amp=pos_bias_init_amp,
                pos_bias_init_sigma=pos_bias_init_sigma,
                cross_attn_init_scale=cross_attn_init_scale,
            )
            for _ in range(depth)
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
