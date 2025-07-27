from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import existing components from DiT
from src.model.dit.dit import sinusoidal_embedding, TimeEmbedding, TransformerBlockAdaLN


class GoalPatchEmbed(nn.Module):
    """Multi-granularity patch embedding for goal generation."""
    
    def __init__(self, img_size: int = 64, dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.dim = dim
        
        # Global embedders (64x64 → 1 token each)
        self.robot_embed = nn.Conv2d(1, dim, kernel_size=64, stride=64)
        self.goal_embed = nn.Conv2d(1, dim, kernel_size=64, stride=64)
        self.object_mask_embed = nn.Conv2d(1, dim, kernel_size=64, stride=64)  # Selected object
        
        # Fine-grained embedders (4x4 patches → 256 tokens each)
        self.movable_embed = nn.Conv2d(1, dim, kernel_size=4, stride=4)
        self.static_embed = nn.Conv2d(1, dim, kernel_size=4, stride=4)
        self.noisy_target_embed = nn.Conv2d(1, dim, kernel_size=4, stride=4)  # ADDED: noisy target
        
        # Calculate number of patches for each type
        self.num_robot_tokens = 1
        self.num_goal_tokens = 1
        self.num_object_mask_tokens = 1
        self.num_movable_tokens = (img_size // 4) ** 2  # 256 tokens
        self.num_static_tokens = (img_size // 4) ** 2   # 256 tokens
        self.num_noisy_target_tokens = (img_size // 4) ** 2  # 256 tokens
        self.total_tokens = 771  # Fixed for goal generation with noisy target
        
    def forward(self, robot_mask: torch.Tensor, goal_mask: torch.Tensor,
                movable_mask: torch.Tensor, static_mask: torch.Tensor, 
                object_mask: torch.Tensor, noisy_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            robot_mask: (B, 1, 64, 64)
            goal_mask: (B, 1, 64, 64)  
            movable_mask: (B, 1, 64, 64)
            static_mask: (B, 1, 64, 64)
            object_mask: (B, 1, 64, 64) - Selected object from previous stage
            noisy_target: (B, 1, 64, 64) - The noisy goal mask to denoise
            
        Returns:
            tokens: (B, 771, dim) - concatenated token sequence
        """
        # Global embeddings (1 token each)
        robot_tokens = self.robot_embed(robot_mask).flatten(2).transpose(1, 2)    # (B, 1, dim)
        goal_tokens = self.goal_embed(goal_mask).flatten(2).transpose(1, 2)       # (B, 1, dim)
        object_mask_tokens = self.object_mask_embed(object_mask).flatten(2).transpose(1, 2)  # (B, 1, dim)
        
        # Fine-grained embeddings (256 tokens each)
        movable_tokens = self.movable_embed(movable_mask).flatten(2).transpose(1, 2)  # (B, 256, dim)
        static_tokens = self.static_embed(static_mask).flatten(2).transpose(1, 2)     # (B, 256, dim)
        noisy_target_tokens = self.noisy_target_embed(noisy_target).flatten(2).transpose(1, 2)  # (B, 256, dim)
        
        # Concatenate all tokens: [robot, goal, movable, static, object_mask, noisy_target]
        tokens = torch.cat([robot_tokens, goal_tokens, movable_tokens, static_tokens, object_mask_tokens, noisy_target_tokens], dim=1)
        return tokens  # (B, 771, dim)


class GoalPositionalEncoding(nn.Module):
    """Type-aware positional encoding for goal generation model."""
    
    def __init__(self, dim: int, img_size: int = 64):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        
        # Type embeddings: robot=0, goal=1, movable=2, static=3, object_mask=4, noisy_target=5
        self.type_embedding = nn.Embedding(6, dim)
        
        # Spatial positional embeddings for fine-grained tokens
        patch_size = 4
        num_spatial_patches = (img_size // patch_size) ** 2  # 256
        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, num_spatial_patches, dim) * 0.02)
        
        # Global position embeddings (for robot/goal/object_mask tokens)
        self.global_pos_embedding = nn.Parameter(torch.randn(1, 3, dim) * 0.02)
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, 771, dim)
            
        Returns:
            tokens_with_pos: (B, 771, dim) - tokens with positional encoding added
        """
        B, N, D = tokens.shape
        device = tokens.device
        
        # Create type IDs for each token position (fixed sequence)
        type_ids = torch.zeros(N, dtype=torch.long, device=device)
        type_ids[0] = 0      # robot
        type_ids[1] = 1      # goal  
        type_ids[2:258] = 2  # movable (256 tokens)
        type_ids[258:514] = 3 # static (256 tokens)
        type_ids[514] = 4    # object_mask (1 token)
        type_ids[515:771] = 5 # noisy_target (256 tokens)
        
        # Add type embeddings
        type_emb = self.type_embedding(type_ids).unsqueeze(0).expand(B, -1, -1)  # (B, 771, dim)
        tokens = tokens + type_emb
        
        # Add spatial positional embeddings
        # Robot, goal, and object_mask get global positions
        tokens[:, 0:1] = tokens[:, 0:1] + self.global_pos_embedding[:, 0:1].expand(B, -1, -1)  # robot
        tokens[:, 1:2] = tokens[:, 1:2] + self.global_pos_embedding[:, 1:2].expand(B, -1, -1)  # goal
        tokens[:, 514:515] = tokens[:, 514:515] + self.global_pos_embedding[:, 2:3].expand(B, -1, -1)  # object_mask
        
        # Movable, static, and noisy_target get spatial positions (same spatial grid)
        spatial_pos = self.spatial_pos_embedding.expand(B, -1, -1)  # (B, 256, dim)
        tokens[:, 2:258] = tokens[:, 2:258] + spatial_pos      # movable
        tokens[:, 258:514] = tokens[:, 258:514] + spatial_pos  # static
        tokens[:, 515:771] = tokens[:, 515:771] + spatial_pos  # noisy_target
        
        return tokens


class MultiGranularityGoalDiT(nn.Module):
    """Multi-granularity Diffusion Transformer for goal generation."""
    
    def __init__(
        self,
        img_size: int = 64,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
    ) -> None:
        super().__init__()
        
        self.img_size = img_size
        self.dim = dim
        
        # Multi-granularity patch embedding
        self.patch_embed = GoalPatchEmbed(img_size, dim)
        
        # Positional encoding
        self.pos_encoding = GoalPositionalEncoding(dim, img_size)
        
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
        
        # Output layer: ConvTranspose2d like standard DiT (unpatchify)
        patch_size = 4
        self.unpatch = nn.ConvTranspose2d(dim, 1, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise for goal generation - following standard DiT approach.
        
        Args:
            x: (B, 6, 64, 64) - [robot, goal, movable, static, object_mask, noisy_goal_mask]
            t: (B,) - timestep
            
        Returns:
            noise_pred: (B, 1, 64, 64) - predicted noise for goal mask
        """
        B = x.size(0)
        
        # Split input channels (fixed 6-channel input)
        robot_mask = x[:, 0:1]      # (B, 1, 64, 64)
        goal_mask = x[:, 1:2]       # (B, 1, 64, 64)  
        movable_mask = x[:, 2:3]    # (B, 1, 64, 64)
        static_mask = x[:, 3:4]     # (B, 1, 64, 64)
        object_mask = x[:, 4:5]     # (B, 1, 64, 64) - Selected object from Stage 1
        noisy_target = x[:, 5:6]    # (B, 1, 64, 64) - The noisy goal mask to denoise
        
        # Multi-granularity patch embedding (now includes noisy target)
        tokens = self.patch_embed(robot_mask, goal_mask, movable_mask, static_mask, object_mask, noisy_target)  # (B, 771, dim)
        
        # Add positional encoding
        tokens = self.pos_encoding(tokens)  # (B, 771, dim)
        
        # Time embedding
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.dim))  # (B, dim)
        
        # Transformer blocks with AdaLN conditioning
        for blk, proj in zip(self.blocks, self.ada_proj):
            gammas_betas = proj(t_emb)  # (B, 4*dim)
            g1, b1, g2, b2 = gammas_betas.chunk(4, dim=-1)
            tokens = blk(tokens, g1, b1, g2, b2)
        
        tokens = self.norm(tokens)
        
        # Extract noisy target tokens for prediction (like standard DiT)
        noisy_target_tokens = tokens[:, 515:771]  # (B, 256, dim) - last 256 tokens
        
        # Un-patchify: reshape tokens back to spatial grid and convert to image
        H = W = int(math.sqrt(noisy_target_tokens.size(1)))  # 16 (from 256 patches)
        noisy_target_tokens = noisy_target_tokens.transpose(1, 2).reshape(B, -1, H, W)  # (B, dim, 16, 16)
        
        # ConvTranspose2d to convert back to image space (like standard DiT)
        noise_pred = self.unpatch(noisy_target_tokens)  # (B, 1, 64, 64)
        
        return noise_pred