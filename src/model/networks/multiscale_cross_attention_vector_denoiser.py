import torch
import torch.nn as nn

from .cross_attention_vector_denoiser import SinusoidalPositionEmbeddings, CrossAttentionBlock


class MultiScaleContextEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 256, token_grids=(16, 8, 4)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1), nn.GroupNorm(8, 256), nn.SiLU(),
        )

        self.token_grids = tuple(token_grids)
        num_scales = len(self.token_grids)
        if num_scales < 1 or num_scales > 3:
            raise ValueError("token_grids must have 1 to 3 entries.")

        feature_channels = [64, 128, 256]
        feature_channels = feature_channels[-num_scales:]
        self.proj_layers = nn.ModuleList(
            [nn.Conv2d(ch, hidden_dim, 1) for ch in feature_channels]
        )
        self.scale_embed = nn.Parameter(torch.zeros(num_scales, hidden_dim))
        self.token_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        features = [f1, f2, f3]

        num_scales = len(self.token_grids)
        features = features[-num_scales:]

        tokens_list = []
        for idx, (feat, grid, proj) in enumerate(zip(features, self.token_grids, self.proj_layers)):
            pooled = nn.functional.adaptive_avg_pool2d(feat, (grid, grid))
            proj_feat = proj(pooled)
            tokens = proj_feat.flatten(2).transpose(1, 2)
            tokens = tokens + self.scale_embed[idx].view(1, 1, -1)
            tokens = self.token_norm(tokens)
            tokens_list.append(tokens)
        return torch.cat(tokens_list, dim=1)


class MultiScaleCrossAttentionVectorDenoiserBackbone(nn.Module):
    def __init__(
        self,
        vector_dim: int = 3,
        image_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        token_grids=(16, 8, 4),
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.encoder = MultiScaleContextEncoder(image_channels, hidden_dim, token_grids)
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
