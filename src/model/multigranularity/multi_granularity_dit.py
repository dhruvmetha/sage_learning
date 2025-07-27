from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import existing components from DiT
from src.model.dit.dit import sinusoidal_embedding, TimeEmbedding, AdaLN, TransformerBlockAdaLN


class MultiGranularityPatchEmbed(nn.Module):
    """Multi-granularity patch embedding for different entity types."""
    
    def __init__(self, img_size: int = 64, dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.dim = dim
        
        # Global embedders (64x64 → 1 token each)
        self.robot_embed = nn.Conv2d(1, dim, kernel_size=64, stride=64)
        self.goal_embed = nn.Conv2d(1, dim, kernel_size=64, stride=64)
        
        # Fine-grained embedders (4x4 patches → 256 tokens each)
        self.movable_embed = nn.Conv2d(1, dim, kernel_size=4, stride=4)
        self.static_embed = nn.Conv2d(1, dim, kernel_size=4, stride=4)
        
        # Object mask embedder for continuous mode (64x64 → 1 token)
        self.object_mask_embed = nn.Conv2d(1, dim, kernel_size=64, stride=64)
        
        # Calculate number of patches for each type
        self.num_robot_tokens = 1
        self.num_goal_tokens = 1
        self.num_movable_tokens = (img_size // 4) ** 2  # 256 tokens
        self.num_static_tokens = (img_size // 4) ** 2   # 256 tokens
        self.total_tokens = (self.num_robot_tokens + self.num_goal_tokens + 
                           self.num_movable_tokens + self.num_static_tokens)  # 514 tokens
        
    def forward(self, robot_mask: torch.Tensor, goal_mask: torch.Tensor,
                movable_mask: torch.Tensor, static_mask: torch.Tensor, 
                object_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            robot_mask: (B, 1, 64, 64)
            goal_mask: (B, 1, 64, 64)  
            movable_mask: (B, 1, 64, 64)
            static_mask: (B, 1, 64, 64)
            object_mask: (B, 1, 64, 64) - Optional, for continuous mode
            
        Returns:
            tokens: (B, 514/515, dim) - concatenated token sequence
        """
        B = robot_mask.size(0)
        
        # Global embeddings (1 token each)
        robot_tokens = self.robot_embed(robot_mask).flatten(2).transpose(1, 2)    # (B, 1, dim)
        goal_tokens = self.goal_embed(goal_mask).flatten(2).transpose(1, 2)       # (B, 1, dim)
        
        # Fine-grained embeddings (256 tokens each)
        movable_tokens = self.movable_embed(movable_mask).flatten(2).transpose(1, 2)  # (B, 256, dim)
        static_tokens = self.static_embed(static_mask).flatten(2).transpose(1, 2)     # (B, 256, dim)
        
        # Build token sequence
        token_list = [robot_tokens, goal_tokens, movable_tokens, static_tokens]
        
        # Add object mask token for continuous mode
        if object_mask is not None:
            object_mask_tokens = self.object_mask_embed(object_mask).flatten(2).transpose(1, 2)  # (B, 1, dim)
            token_list.append(object_mask_tokens)
        
        # Concatenate all tokens
        tokens = torch.cat(token_list, dim=1)
        return tokens  # (B, 514, dim) for discrete or (B, 515, dim) for continuous


class MultiGranularityPositionalEncoding(nn.Module):
    """Type-aware positional encoding for multi-granularity tokens."""
    
    def __init__(self, total_tokens: int, dim: int, img_size: int = 64):
        super().__init__()
        self.total_tokens = total_tokens
        self.dim = dim
        self.img_size = img_size
        
        # Type embeddings
        self.type_embedding = nn.Embedding(5, dim)  # robot=0, goal=1, movable=2, static=3, object_mask=4
        
        # Spatial positional embeddings for fine-grained tokens
        patch_size = 4
        num_spatial_patches = (img_size // patch_size) ** 2
        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, num_spatial_patches, dim) * 0.02)
        
        # Global position embeddings (for robot/goal tokens)
        self.global_pos_embedding = nn.Parameter(torch.randn(1, 2, dim) * 0.02)  # 2 global tokens
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, 514/515, dim) - variable length for discrete/continuous
            
        Returns:
            tokens_with_pos: (B, 514/515, dim) - tokens with positional encoding added
        """
        B, N, D = tokens.shape
        device = tokens.device
        
        # Create type IDs for each token position based on sequence length
        type_ids = torch.zeros(N, dtype=torch.long, device=device)
        type_ids[0] = 0      # robot
        type_ids[1] = 1      # goal  
        type_ids[2:258] = 2  # movable (256 tokens)
        type_ids[258:514] = 3 # static (256 tokens)
        
        # If continuous mode (515 tokens), add object_mask type
        if N == 515:
            type_ids[514] = 4  # object_mask (1 token)
        
        # Add type embeddings
        type_emb = self.type_embedding(type_ids).unsqueeze(0).expand(B, -1, -1)  # (B, 514, dim)
        tokens = tokens + type_emb
        
        # Add spatial positional embeddings
        # Robot and goal get global positions
        tokens[:, 0:2] = tokens[:, 0:2] + self.global_pos_embedding.expand(B, -1, -1)
        
        # Movable and static get spatial positions (repeated for both)
        spatial_pos = self.spatial_pos_embedding.expand(B, -1, -1)  # (B, 256, dim)
        tokens[:, 2:258] = tokens[:, 2:258] + spatial_pos      # movable
        tokens[:, 258:514] = tokens[:, 258:514] + spatial_pos  # static
        
        # Object mask gets global position (if present in continuous mode)
        if N == 515:
            # Use the first position of global pos embedding for object_mask
            tokens[:, 514:515] = tokens[:, 514:515] + self.global_pos_embedding[:, 0:1].expand(B, -1, -1)
        
        return tokens


class MultiGranularityDiT(nn.Module):
    """Multi-granularity Diffusion Transformer for robotics manipulation."""
    
    def __init__(
        self,
        img_size: int = 64,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        robot_patch_size: int = 64,
        goal_patch_size: int = 64,
        object_patch_size: int = 4,
    ) -> None:
        super().__init__()
        
        self.img_size = img_size
        self.dim = dim
        
        # Multi-granularity patch embedding
        self.patch_embed = MultiGranularityPatchEmbed(img_size, dim)
        
        # Positional encoding
        self.pos_encoding = MultiGranularityPositionalEncoding(
            self.patch_embed.total_tokens, dim, img_size
        )
        
        # Time embedding
        self.time_mlp = TimeEmbedding(dim)
        
        # Transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlockAdaLN(dim, heads) for _ in range(depth)
        ])
        
        # AdaLN projection layers (one per block)
        self.ada_proj = nn.ModuleList([])
        for _ in range(depth):
            proj = nn.Linear(dim, dim * 4)
            nn.init.zeros_(proj.weight)  # AdaLN-Zero init
            nn.init.zeros_(proj.bias)
            self.ada_proj.append(proj)
        
        self.norm = nn.LayerNorm(dim)
        
        # Prediction head for movable object selection
        # Predict which of the 256 movable patches to interact with
        self.prediction_head = nn.Linear(dim, 1)  # Per-patch prediction
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise for movable object selection.
        
        Args:
            x: (B, 5, 64, 64) - [robot, goal, movable, static, target_movable]
            t: (B,) - timestep
            
        Returns:
            noise_pred: (B, 1, 64, 64) - predicted noise for target movable mask
        """
        B = x.size(0)
        
        # Split input channels based on number of channels
        robot_mask = x[:, 0:1]      # (B, 1, 64, 64)
        goal_mask = x[:, 1:2]       # (B, 1, 64, 64)  
        movable_mask = x[:, 2:3]    # (B, 1, 64, 64)
        static_mask = x[:, 3:4]     # (B, 1, 64, 64)
        
        if x.shape[1] == 5:
            # Discrete mode: [robot, goal, movable, static, noisy_target]
            # Note: x[:, 4:5] is the noisy target that we're trying to denoise
            tokens = self.patch_embed(robot_mask, goal_mask, movable_mask, static_mask)  # (B, 514, dim)
        elif x.shape[1] == 6:
            # Continuous mode: [robot, goal, movable, static, object_mask, noisy_target]  
            object_mask = x[:, 4:5]  # (B, 1, 64, 64)
            # Note: x[:, 5:6] is the noisy target that we're trying to denoise
            tokens = self.patch_embed(robot_mask, goal_mask, movable_mask, static_mask, object_mask)  # (B, 515, dim)
        else:
            raise ValueError(f"Expected 5 or 6 input channels, got {x.shape[1]}")
        
        # Add positional encoding
        tokens = self.pos_encoding(tokens)  # (B, 514/515, dim)
        
        # Time embedding
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.dim))  # (B, dim)
        
        # Transformer blocks with AdaLN conditioning
        for blk, proj in zip(self.blocks, self.ada_proj):
            gammas_betas = proj(t_emb)  # (B, 4*dim)
            g1, b1, g2, b2 = gammas_betas.chunk(4, dim=-1)
            tokens = blk(tokens, g1, b1, g2, b2)
        
        tokens = self.norm(tokens)
        
        # Extract movable tokens for prediction (tokens 2:258)
        movable_tokens = tokens[:, 2:258]  # (B, 256, dim)
        
        # Predict per-patch values
        patch_predictions = self.prediction_head(movable_tokens)  # (B, 256, 1)
        patch_predictions = patch_predictions.squeeze(-1)  # (B, 256)
        
        # Reshape back to spatial dimensions (16x16 patches)
        H = W = int(math.sqrt(movable_tokens.size(1)))  # 16
        patch_predictions = patch_predictions.view(B, H, W)  # (B, 16, 16)
        
        # Upsample to original image size (64x64)
        noise_pred = F.interpolate(
            patch_predictions.unsqueeze(1),  # (B, 1, 16, 16)
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )  # (B, 1, 64, 64)
        
        return noise_pred