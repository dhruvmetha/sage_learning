from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ConvGNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleCoordContextEncoder(nn.Module):
    def __init__(self, image_channels: int = 5, hidden_dim: int = 256):
        super().__init__()
        in_channels = image_channels + 2
        self.stem = ConvGNBlock(in_channels, 64, stride=1)
        self.stage1 = ConvGNBlock(64, 96, stride=2)
        self.stage2 = ConvGNBlock(96, 128, stride=2)
        self.stage3 = ConvGNBlock(128, 192, stride=2)
        self.stage4 = ConvGNBlock(192, 256, stride=2)

        self.proj_layers = nn.ModuleList(
            [
                nn.Conv2d(128, hidden_dim, kernel_size=1),
                nn.Conv2d(192, hidden_dim, kernel_size=1),
                nn.Conv2d(256, hidden_dim, kernel_size=1),
            ]
        )
        self.scale_embeds = nn.Parameter(torch.zeros(3, 1, hidden_dim))
        self.pos_embeds = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, 16 * 16, hidden_dim)),
                nn.Parameter(torch.zeros(1, 8 * 8, hidden_dim)),
                nn.Parameter(torch.zeros(1, 4 * 4, hidden_dim)),
            ]
        )
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.scale_embeds, std=0.02)
        for pos in self.pos_embeds:
            nn.init.normal_(pos, std=0.02)

    @staticmethod
    def _coord_channels(
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ys, xs = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([xs, ys], dim=0)
        return coords.unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = x.shape
        coords = self._coord_channels(batch, height, width, x.device, x.dtype)
        x = torch.cat([x, coords], dim=1)

        x = self.stem(x)
        x = self.stage1(x)
        feat16 = self.stage2(x)
        feat8 = self.stage3(feat16)
        feat4 = self.stage4(feat8)

        feats = [feat16, feat8, feat4]
        token_chunks = []
        for idx, (feat, proj, pos_embed) in enumerate(zip(feats, self.proj_layers, self.pos_embeds)):
            proj_feat = proj(feat)
            tokens = proj_feat.flatten(2).transpose(1, 2)
            tokens = tokens + pos_embed + self.scale_embeds[idx]
            token_chunks.append(self.token_norm(tokens))

        context_tokens = torch.cat(token_chunks, dim=1)
        global_ctx = self.global_norm(context_tokens.mean(dim=1))
        return context_tokens, global_ctx


class HypothesisDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, use_self_attn: bool = True):
        super().__init__()
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.self_norm = nn.LayerNorm(hidden_dim)
            self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_norm_q = nn.LayerNorm(hidden_dim)
        self.cross_norm_ctx = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        inner_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, queries: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        if self.use_self_attn:
            q = self.self_norm(queries)
            self_out, _ = self.self_attn(q, q, q, need_weights=False)
            queries = queries + self_out

        q = self.cross_norm_q(queries)
        ctx = self.cross_norm_ctx(context_tokens)
        cross_out, _ = self.cross_attn(q, ctx, ctx, need_weights=False)
        queries = queries + cross_out
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries


class MultiScaleHypothesisPosePredictor(nn.Module):
    def __init__(
        self,
        image_channels: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_hypotheses: int = 4,
        mlp_ratio: float = 4.0,
        use_self_attn: bool = True,
        per_slot_heads: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.encoder = MultiScaleCoordContextEncoder(image_channels=image_channels, hidden_dim=hidden_dim)
        self.global_query_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_embed = nn.Parameter(torch.zeros(num_hypotheses, hidden_dim))
        self.layers = nn.ModuleList(
            [
                HypothesisDecoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_self_attn=use_self_attn,
                )
                for _ in range(num_layers)
            ]
        )
        self.per_slot_heads = per_slot_heads
        if per_slot_heads:
            self.pose_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, 3),
                    )
                    for _ in range(num_hypotheses)
                ]
            )
            self.logit_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, 1),
                    )
                    for _ in range(num_hypotheses)
                ]
            )
        else:
            self.pose_norm = nn.LayerNorm(hidden_dim)
            self.pose_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )
            self.logit_norm = nn.LayerNorm(hidden_dim)
            self.logit_head = nn.Linear(hidden_dim, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.query_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context_tokens, global_ctx = self.encoder(images)
        global_bias = self.global_query_proj(global_ctx).unsqueeze(1)
        queries = self.query_embed.unsqueeze(0).expand(images.shape[0], -1, -1)
        queries = queries + global_bias
        for layer in self.layers:
            queries = layer(queries, context_tokens)

        if self.per_slot_heads:
            pose_preds = torch.stack(
                [head(queries[:, i]) for i, head in enumerate(self.pose_heads)],
                dim=1,
            )
            logits = torch.stack(
                [head(queries[:, i]).squeeze(-1) for i, head in enumerate(self.logit_heads)],
                dim=1,
            )
        else:
            pose_preds = self.pose_head(self.pose_norm(queries))
            logits = self.logit_head(self.logit_norm(queries)).squeeze(-1)
        return pose_preds, logits


__all__ = [
    "ConvGNBlock",
    "MultiScaleCoordContextEncoder",
    "HypothesisDecoderBlock",
    "MultiScaleHypothesisPosePredictor",
]
