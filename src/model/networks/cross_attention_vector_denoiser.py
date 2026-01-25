import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SpatialContextEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 256, token_grid: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, hidden_dim, 3, 2, 1), nn.GroupNorm(8, hidden_dim), nn.SiLU(),
            nn.AdaptiveAvgPool2d((token_grid, token_grid)),
        )
        self.token_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x)
        tokens = feats.flatten(2).transpose(1, 2)
        return self.token_norm(tokens)


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_ctx = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        inner_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(x).unsqueeze(1)
        k = self.norm_ctx(context_tokens)
        attn_out, _ = self.attn(q, k, k, need_weights=False)
        x = x + attn_out.squeeze(1)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class CrossAttentionVectorDenoiserBackbone(nn.Module):
    def __init__(
        self,
        vector_dim: int = 3,
        image_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        token_grid: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.encoder = SpatialContextEncoder(image_channels, hidden_dim, token_grid)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(vector_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(num_layers)
            ]
        )
        self.final_layer = nn.Linear(hidden_dim, vector_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        context_tokens = self.encoder(images)
        t_emb = self.time_mlp(t)
        x = self.input_proj(x) + t_emb
        for layer in self.layers:
            x = layer(x, context_tokens)
        return self.final_layer(x)
