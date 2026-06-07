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
                 heads=6, num_depths=5, num_edges=60, dropout=0.0,
                 use_zoom=False, zoom_size=128, zoom_patch=4, zoom_depth=2, use_local=True):
        super().__init__()
        self.dim = dim; self.num_depths = num_depths; self.S = img_size; self.grid = img_size // patch
        npatch = self.grid ** 2
        self.patch = PatchEmbed(in_channels, patch, dim)
        self.scene_pos = nn.Parameter(torch.randn(1, npatch, dim) * 0.02)
        self.scene_blocks = nn.ModuleList([SelfBlock(dim, heads, drop=dropout) for _ in range(scene_depth)])
        self.scene_norm = nn.LayerNorm(dim)
        self.edge_pos = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))  # positional id of contact (x,y)
        # ABLATION: use_local=False drops the per-edge LOCAL gather entirely -> edge token = positional-id
        # (coordinate) + cross-attention to the scene only (the most HACMan-faithful "point = coord + context",
        # no rasterized gather -> no aliasing). local_proj is created ONLY when use_local, so a no-gather ckpt
        # has no local_proj keys and eval can auto-detect the variant.
        self.use_local = use_local
        if use_local:
            self.local_proj = nn.Linear(dim, dim)
        self.edge_blocks = nn.ModuleList([CrossBlock(dim, heads, drop=dropout) for _ in range(edge_depth)])
        self.edge_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_depths))
        # OPTIONAL dual-crop: a second light stem over a tight zoom crop, from which the per-edge LOCAL
        # feature is gathered (de-aliased). OFF by default -> forward is byte-identical to the single-crop
        # model, so E2/E4 reproduce exactly. Context (cross-attn to the wide scene tokens) is unchanged;
        # the positional id stays in the WIDE frame (the shared coordinate that glues zoom-local to wide-context).
        self.use_zoom = use_zoom; self.zoom_size = zoom_size
        if use_zoom:
            self.zgrid = zoom_size // zoom_patch
            self.zoom_patch = PatchEmbed(in_channels, zoom_patch, dim)
            self.zoom_pos = nn.Parameter(torch.randn(1, self.zgrid ** 2, dim) * 0.02)
            self.zoom_blocks = nn.ModuleList([SelfBlock(dim, heads, drop=dropout) for _ in range(zoom_depth)])
            self.zoom_norm = nn.LayerNorm(dim)

    def forward(self, x, contact_px, x_zoom=None, contact_px_zoom=None):
        """x: (B,5,H,W); contact_px: (B,60,2) px in the HxW (wide) frame.
        Dual-crop (use_zoom): x_zoom (B,5,Z,Z) tight object crop + contact_px_zoom (B,60,2) px in the ZZ frame —
        the LOCAL feature is gathered from the zoom map; positional id + context still use the wide frame."""
        B = x.size(0)
        tok = self.patch(x) + self.scene_pos                       # B, Np, D
        for blk in self.scene_blocks:
            tok = blk(tok)
        tok = self.scene_norm(tok)                                 # scene tokens (context)
        grid = (contact_px / self.S) * 2 - 1                       # B,60,2 in [-1,1] (WIDE frame; the shared pos)
        if self.use_local:
            if self.use_zoom:
                zt = self.zoom_patch(x_zoom) + self.zoom_pos       # zoom stem
                for blk in self.zoom_blocks:
                    zt = blk(zt)
                zt = self.zoom_norm(zt)
                src = zt.transpose(1, 2).reshape(B, self.dim, self.zgrid, self.zgrid)
                gg = (contact_px_zoom / self.zoom_size) * 2 - 1    # gather in the ZOOM frame
            else:
                src = tok.transpose(1, 2).reshape(B, self.dim, self.grid, self.grid)
                gg = grid                                          # gather in the WIDE frame (original)
            gathered = F.grid_sample(src, gg.unsqueeze(1), align_corners=False, mode="bilinear", padding_mode="border")
            gathered = gathered.squeeze(2).transpose(1, 2)         # B,60,D  (local feature per edge)
            e = self.local_proj(gathered) + self.edge_pos(grid)    # B,60,D  (local + positional id)
        else:
            e = self.edge_pos(grid)                                # NO-GATHER: positional id (coordinate) only
        for blk in self.edge_blocks:
            e = blk(e, tok)                                        # cross-attend WIDE scene + self-attend edges
        e = self.edge_norm(e)
        return self.head(e)                                        # B,60,num_depths
