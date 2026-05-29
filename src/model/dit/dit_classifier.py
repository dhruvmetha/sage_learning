"""
DiT-based classifier for primitive feasibility prediction.

Reuses the patch embedding and transformer blocks from DiT,
removes diffusion-specific parts (time embedding, AdaLN noise conditioning),
adds a classification head that outputs 60x10 logits.
"""

import math
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Image → patch tokens via Conv2d projection."""
    def __init__(self, in_ch: int, patch: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm (no AdaLN)."""
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


class DiTClassifier(nn.Module):
    """DiT backbone adapted for primitive feasibility classification.

    Input: (B, in_ch, H, W) scene masks
    Output: (B, 60, 10) logits per primitive

    Architecture:
        Patch embedding → positional embedding → N transformer blocks
        → global average pool → MLP → 600 logits → reshape to (60, 10)
    """

    def __init__(
        self,
        img_size: int = 64,
        patch: int = 4,
        in_channels: int = 5,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        hidden_dim: int = 512,
    ):
        super().__init__()
        assert img_size % patch == 0
        num_patches = (img_size // patch) ** 2  # 256 for 64/4

        self.patch_embed = PatchEmbed(in_channels, patch, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 600),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Patch embed + positional embedding
        tok = self.patch_embed(x) + self.pos_emb  # (B, N, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        tok = torch.cat([cls, tok], dim=1)  # (B, N+1, D)

        # Transformer blocks
        for blk in self.blocks:
            tok = blk(tok)

        tok = self.norm(tok)

        # CLS token output → classification head
        cls_out = tok[:, 0]  # (B, D)
        logits = self.head(cls_out)  # (B, 600)

        return logits.view(B, 60, 10)
