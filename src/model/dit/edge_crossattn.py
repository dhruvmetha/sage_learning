"""HACMan-faithful per-EDGE critic (our 60-edge "point cloud").

Each of the 60 edges becomes a token = (local scene feature gathered at its contact pixel) +
(positional id of its contact location). Edge tokens cross-attend to the scene feature map (global
context, like HACMan's PointNet++ global/skip path) and self-attend among themselves (point-transformer
local-among-points), then a shared per-edge head emits depth logits. Output (B, 60, num_depths).

This replaces the global CLS readout (one vector -> 60 scores) that caused the 96.6%-wrong-edge failure:
now every edge reasons for itself from its own local feature + a global look at the scene.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, in_ch, patch, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # B, N, D


class SelfBlock(nn.Module):
    def __init__(self, dim, heads, mlp=4.0, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

    def forward(self, x):
        h = self.n1(x); x = x + self.attn(h, h, h)[0]
        x = x + self.mlp(self.n2(x)); return x


class CrossBlock(nn.Module):
    """edge tokens: cross-attend to scene, then self-attend among edges, then MLP."""
    def __init__(self, dim, heads, mlp=4.0, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.cross = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim); self.slf = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

    def forward(self, e, scene):
        h = self.n1(e); e = e + self.cross(h, scene, scene)[0]
        h = self.n2(e); e = e + self.slf(h, h, h)[0]
        e = e + self.mlp(self.n3(e)); return e


class EdgeCrossAttn(nn.Module):
    def __init__(self, img_size=64, patch=4, in_channels=5, dim=192, scene_depth=4, edge_depth=4,
                 heads=6, num_depths=5, num_edges=60, dropout=0.0):
        super().__init__()
        self.dim = dim; self.num_depths = num_depths; self.S = img_size; self.grid = img_size // patch
        npatch = self.grid ** 2
        self.patch = PatchEmbed(in_channels, patch, dim)
        self.scene_pos = nn.Parameter(torch.randn(1, npatch, dim) * 0.02)
        self.scene_blocks = nn.ModuleList([SelfBlock(dim, heads, drop=dropout) for _ in range(scene_depth)])
        self.scene_norm = nn.LayerNorm(dim)
        self.edge_pos = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))  # positional id of contact (x,y)
        self.local_proj = nn.Linear(dim, dim)
        self.edge_blocks = nn.ModuleList([CrossBlock(dim, heads, drop=dropout) for _ in range(edge_depth)])
        self.edge_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_depths))

    def forward(self, x, contact_px):
        """x: (B,5,H,W);  contact_px: (B,60,2) pixel coords (px=x/col, py=y/row) in the HxW frame."""
        B = x.size(0)
        tok = self.patch(x) + self.scene_pos                       # B, Np, D
        for blk in self.scene_blocks:
            tok = blk(tok)
        tok = self.scene_norm(tok)                                 # scene tokens
        fmap = tok.transpose(1, 2).reshape(B, self.dim, self.grid, self.grid)  # B,D,16,16
        grid = (contact_px / self.S) * 2 - 1                       # B,60,2 in [-1,1], (x,y)
        gathered = F.grid_sample(fmap, grid.unsqueeze(1), align_corners=False, mode="bilinear", padding_mode="border")
        gathered = gathered.squeeze(2).transpose(1, 2)             # B,60,D  (local feature per edge)
        e = self.local_proj(gathered) + self.edge_pos(grid)        # B,60,D  (+ positional id)
        for blk in self.edge_blocks:
            e = blk(e, tok)
        e = self.edge_norm(e)
        return self.head(e)                                        # B,60,num_depths
