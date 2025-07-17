from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------
# Helper: sinusoidal timestep embedding
# ---------------------------------------------------------------

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Return (B, dim) sinusoidal embeddings of integer timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


# ---------------------------------------------------------------
# Adaptive LayerNorm‑Zero
# ---------------------------------------------------------------

class AdaLN(nn.Module):
    """LayerNorm whose (gamma, beta) come from outside (per‑sample)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # x : (B, N, D); gamma/beta : (B, D)
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return gamma.unsqueeze(1) * x_norm + beta.unsqueeze(1)


# ---------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------

class TimeEmbedding(nn.Module):
    """Two‑layer MLP mapping timestep sinusoid → model dim."""

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
    """Convert image to (B, N, dim) patch tokens via strided conv."""

    def __init__(self, in_ch: int, patch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.patch = patch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, dim, H/patch, W/patch)
        return x.flatten(2).transpose(1, 2)  # (B, N, dim)


class TransformerBlockAdaLN(nn.Module):
    """ViT block with two AdaLN‑Zero pre‑norms."""

    def __init__(self, dim: int, heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.adaln1 = AdaLN(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.adaln2 = AdaLN(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        g1: torch.Tensor,
        b1: torch.Tensor,
        g2: torch.Tensor,
        b2: torch.Tensor,
    ) -> torch.Tensor:
        # Self‑attention block
        h = self.adaln1(x, g1, b1)
        x = x + self.attn(h, h, h)[0]
        # MLP block
        h = self.adaln2(x, g2, b2)
        x = x + self.mlp(h)
        return x


# ---------------------------------------------------------------
# DiT with AdaLN‑Zero
# ---------------------------------------------------------------

class RelationalDiT(nn.Module):
    """Minimal AdaLN‑equipped Diffusion Transformer for square RGB images."""

    def __init__(
        self,
        img_size: int = 32,
        patch: int = 4,
        in_ch: int = 3,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        out_ch: int = 2,
    ) -> None:
        super().__init__()
        assert img_size % patch == 0, "Image size must be divisible by patch size."
        
        self.robot_embed = PatchEmbed(1, img_size, dim)
        self.goal_embed = PatchEmbed(1, img_size, dim)
        self.movable_embed = PatchEmbed(1, patch, dim)
        
        self.patch_embed = PatchEmbed(in_ch, patch, dim)
        num_patches = (img_size // patch) ** 2

        # Learned positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)

        # Time embedding →(B, dim)
        self.time_mlp = TimeEmbedding(dim)

        # Transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlockAdaLN(dim, heads) for _ in range(depth)
        ])

        # Each block gets its own Linear that maps t‑emb → γβγβ (4*D)
        self.ada_proj = nn.ModuleList([])
        for _ in range(depth):
            proj = nn.Linear(dim, dim * 4)
            nn.init.zeros_(proj.weight)  # AdaLN‑Zero init
            nn.init.zeros_(proj.bias)
            self.ada_proj.append(proj)

        self.norm = nn.LayerNorm(dim)  # final LN (non‑adaptive)
        self.unpatch = nn.ConvTranspose2d(dim, out_ch, kernel_size=patch, stride=patch)

    # -----------------------------------------------------------
    # Forward
    # -----------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict ε from noisy image and timestep."""
        B = x.size(0)

        # Tokenise & add positions
        tok = self.patch_embed(x) + self.pos_emb  # (B, N, D)

        # Time embedding
        t_emb = self.time_mlp(sinusoidal_embedding(t, tok.size(-1)))  # (B, D)

        # Transformer with AdaLN parameters
        for blk, proj in zip(self.blocks, self.ada_proj):
            gammas_betas = proj(t_emb)  # (B, 4*D)
            g1, b1, g2, b2 = gammas_betas.chunk(4, dim=-1)
            tok = blk(tok, g1, b1, g2, b2)

        tok = self.norm(tok)

        # Un‑patchify back to image
        H = W = int(math.sqrt(tok.size(1)))
        tok = tok.transpose(1, 2).reshape(B, -1, H, W)
        return self.unpatch(tok)