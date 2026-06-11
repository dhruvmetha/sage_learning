"""HACMan-faithful per-EDGE critic (our 60-edge "point cloud").

Each of the 60 edges becomes a token = (local scene feature gathered at its contact pixel) +
(positional id of its contact location). Edge tokens cross-attend to the scene feature map (global
context, like HACMan's PointNet++ global/skip path) and self-attend among themselves (point-transformer
local-among-points), then a shared per-edge head emits depth logits. Output (B, 60, num_depths).

This replaces the global CLS readout (one vector -> 60 scores) that caused the 96.6%-wrong-edge failure:
now every edge reasons for itself from its own local feature + a global look at the scene.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_encode(coords, L):
    """coords (B,N,2) in [-1,1] -> (B,N,4L) NeRF-style sin/cos over L geometric freq bands.
    High bands make nearby coords map to very different features (fixes MLP spectral bias)."""
    freqs = (2.0 ** torch.arange(L, device=coords.device, dtype=coords.dtype)) * math.pi  # (L,)
    a = coords.unsqueeze(-1) * freqs                       # (B,N,2,L)
    return torch.cat([a.sin(), a.cos()], dim=-1).flatten(-2)  # (B,N,4L)


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
    """edge tokens: cross-attend to scene, then (optionally) self-attend among edges, then MLP.

    self_attn=False (H2 ablation) -> edges are scored INDEPENDENTLY given the scene: no inter-edge
    information flow anywhere in the network (the module is not even constructed). [USER] hypothesis:
    under sparse/masked labels, independent edges should hold up better (no co-adaptation channel)."""
    def __init__(self, dim, heads, mlp=4.0, drop=0.0, self_attn=True):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.cross = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.self_attn = self_attn
        if self_attn:
            self.n2 = nn.LayerNorm(dim); self.slf = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim))

    def forward(self, e, scene):
        h = self.n1(e); e = e + self.cross(h, scene, scene)[0]
        if self.self_attn:
            h = self.n2(e); e = e + self.slf(h, h, h)[0]
        e = e + self.mlp(self.n3(e)); return e


class EdgeCrossAttn(nn.Module):
    def __init__(self, img_size=64, patch=4, in_channels=5, dim=192, scene_depth=4, edge_depth=4,
                 heads=6, num_depths=5, num_edges=60, dropout=0.0,
                 use_zoom=False, zoom_size=128, zoom_patch=4, zoom_depth=2, use_local=True,
                 pos_fourier=False, fourier_L=8, use_edge_embed=False,
                 fine_stem=False, fine_stride=2, edge_self_attn=True,
                 budget_cond=False, max_budget=3, value_bins=0):
        super().__init__()
        self.dim = dim; self.num_depths = num_depths; self.S = img_size; self.grid = img_size // patch
        self.num_edges = num_edges
        # BUDGET-CONDITIONED HORIZON-Q (horizon_q_build_journal.md): the same map answers Q(s,a,H) for a
        # remaining push budget H. budget_cond=True adds an H embedding to every edge token (UVFA/Decision-
        # Transformer-style conditioning). value_bins>0 switches the per-(edge,depth) head from a single
        # sigmoid logit to a HL-Gauss classification over `value_bins` bins of [0,1] (Stop-Regressing 2403.03950
        # — classification value heads beat regression). Both default OFF -> forward is byte-identical to E2/E4.
        self.budget_cond = budget_cond; self.value_bins = value_bins; self.max_budget = max_budget
        npatch = self.grid ** 2
        self.patch = PatchEmbed(in_channels, patch, dim)
        self.scene_pos = nn.Parameter(torch.randn(1, npatch, dim) * 0.02)
        self.scene_blocks = nn.ModuleList([SelfBlock(dim, heads, drop=dropout) for _ in range(scene_depth)])
        self.scene_norm = nn.LayerNorm(dim)
        # SHARP-ID (lit: NeRF/Tancik Fourier features + ViT-style per-element embedding) — fixes the
        # raw-coord MLP spectral bias that can't separate nearby edges, and the per-edge embedding gives
        # each of the 60 edges a guaranteed-distinct identity (fixes the 4 coincident corners).
        self.pos_fourier = pos_fourier; self.fourier_L = fourier_L
        pos_in = 4 * fourier_L if pos_fourier else 2          # [sin,cos] x 2 axes x L bands
        self.edge_pos = nn.Sequential(nn.Linear(pos_in, dim), nn.GELU(), nn.Linear(dim, dim))
        self.use_edge_embed = use_edge_embed
        if use_edge_embed:
            self.edge_embed = nn.Embedding(num_edges, dim)
        if budget_cond:
            self.budget_embed = nn.Embedding(max_budget + 1, dim)   # H in {0..max_budget}; index by remaining budget
        # ABLATION: use_local=False drops the per-edge LOCAL gather entirely -> edge token = positional-id
        # (coordinate) + cross-attention to the scene only (the most HACMan-faithful "point = coord + context",
        # no rasterized gather -> no aliasing). local_proj is created ONLY when use_local, so a no-gather ckpt
        # has no local_proj keys and eval can auto-detect the variant.
        self.use_local = use_local
        if use_local:
            self.local_proj = nn.Linear(dim, dim)
        # DE-ALIASED gather (lit: aliasing agent's specific fix). Gather the per-edge local feature from a
        # FINE stride-2 conv map (img_size/2 = 32x32) instead of the coarse 16x16 patch map. A sharper map
        # means the bilinear sample at the contact pixel mixes in less of the neighbouring edges' content,
        # so two nearby edges get more distinct local features. Cheap: one conv, no extra self-attn tokens.
        self.fine_stem = fine_stem
        if fine_stem:
            self.fine_conv = nn.Conv2d(in_channels, dim, fine_stride, fine_stride)
            self.fine_grid = img_size // fine_stride
        self.edge_blocks = nn.ModuleList([CrossBlock(dim, heads, drop=dropout, self_attn=edge_self_attn)
                                          for _ in range(edge_depth)])
        self.edge_norm = nn.LayerNorm(dim)
        head_out = num_depths * value_bins if value_bins > 0 else num_depths
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, head_out))
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

    def forward(self, x, contact_px, x_zoom=None, contact_px_zoom=None, H=None):
        """x: (B,5,H,W); contact_px: (B,60,2) px in the HxW (wide) frame. H: (B,) long remaining-budget (budget_cond).
        Dual-crop (use_zoom): x_zoom (B,5,Z,Z) tight object crop + contact_px_zoom (B,60,2) px in the ZZ frame —
        the LOCAL feature is gathered from the zoom map; positional id + context still use the wide frame."""
        B = x.size(0)
        tok = self.patch(x) + self.scene_pos                       # B, Np, D
        for blk in self.scene_blocks:
            tok = blk(tok)
        tok = self.scene_norm(tok)                                 # scene tokens (context)
        grid = (contact_px / self.S) * 2 - 1                       # B,60,2 in [-1,1] (WIDE frame; the shared pos)
        # positional id: Fourier(coord) if sharp, else raw coord; + per-edge embedding if enabled
        pf = fourier_encode(grid, self.fourier_L) if self.pos_fourier else grid
        pos = self.edge_pos(pf)                                    # B,60,D
        if self.use_edge_embed:
            pos = pos + self.edge_embed.weight.unsqueeze(0)        # +(1,60,D) -> per-edge identity
        if self.use_local:
            if self.use_zoom:
                zt = self.zoom_patch(x_zoom) + self.zoom_pos       # zoom stem
                for blk in self.zoom_blocks:
                    zt = blk(zt)
                zt = self.zoom_norm(zt)
                src = zt.transpose(1, 2).reshape(B, self.dim, self.zgrid, self.zgrid)
                gg = (contact_px_zoom / self.zoom_size) * 2 - 1    # gather in the ZOOM frame
            elif self.fine_stem:
                src = self.fine_conv(x)                            # B,D,32,32 (de-aliased fine map)
                gg = grid                                          # same [-1,1] WIDE coords, sharper sampling
            else:
                src = tok.transpose(1, 2).reshape(B, self.dim, self.grid, self.grid)
                gg = grid                                          # gather in the WIDE frame (original)
            gathered = F.grid_sample(src, gg.unsqueeze(1), align_corners=False, mode="bilinear", padding_mode="border")
            gathered = gathered.squeeze(2).transpose(1, 2)         # B,60,D  (local feature per edge)
            e = self.local_proj(gathered) + pos                    # B,60,D  (local + positional id)
        else:
            e = pos                                                # NO-GATHER: positional id only
        if self.budget_cond and H is not None:
            e = e + self.budget_embed(H).unsqueeze(1)              # (B,1,D) remaining-budget id, broadcast over 60 edges
        for blk in self.edge_blocks:
            e = blk(e, tok)                                        # cross-attend WIDE scene + self-attend edges
        e = self.edge_norm(e)
        out = self.head(e)                                         # B,60,(num_depths | num_depths*value_bins)
        if self.value_bins > 0:
            out = out.view(B, self.num_edges, self.num_depths, self.value_bins)  # B,60,nd,bins (HL-Gauss logits)
        return out                                                 # B,60,nd  or  B,60,nd,bins
