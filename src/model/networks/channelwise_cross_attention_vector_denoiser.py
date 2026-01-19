import torch
import torch.nn as nn

from .cross_attention_vector_denoiser import SinusoidalPositionEmbeddings, CrossAttentionBlock


class ChannelwiseContextEncoder(nn.Module):
    def __init__(self, image_channels: int = 3, hidden_dim: int = 256, token_grid: int = 8):
        super().__init__()
        self.image_channels = image_channels
        self.token_grid = token_grid

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, hidden_dim, 3, 2, 1), nn.GroupNorm(8, hidden_dim), nn.SiLU(),
            nn.AdaptiveAvgPool2d((token_grid, token_grid)),
        )
        self.channel_embed = nn.Parameter(torch.zeros(image_channels, hidden_dim))
        self.token_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.image_channels:
            raise ValueError(f"Expected {self.image_channels} channels, got {c}.")

        x = x.view(b * c, 1, h, w)
        feats = self.encoder(x)
        tokens = feats.flatten(2).transpose(1, 2)
        tokens = self.token_norm(tokens)

        tokens = tokens.view(b, c, -1, tokens.shape[-1])
        tokens = tokens + self.channel_embed.view(1, c, 1, -1)
        tokens = tokens.view(b, c * tokens.shape[2], tokens.shape[-1])
        return tokens


class ChannelwiseCrossAttentionVectorDenoiserBackbone(nn.Module):
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
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.encoder = ChannelwiseContextEncoder(image_channels, hidden_dim, token_grid)
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
