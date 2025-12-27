import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class EnvironmentEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.GroupNorm(8, 256), nn.SiLU(),
            
            # --- THE FIX ---
            # Instead of LazyLinear, we use Adaptive Pooling to a 4x4 grid.
            # This ensures the output is ALWAYS (256 * 4 * 4) = 4096, 
            # regardless of input image size.
            # It preserves spatial data (unlike 1x1 pooling) but fixes the crash.
            nn.AdaptiveAvgPool2d((4, 4)), 
            
            nn.Flatten(),
            
            # 256 channels * 4 * 4 grid = 4096 input features
            nn.Linear(4096, out_dim), 
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class FiLMBlock(nn.Module):
    """
    Kept the name 'FiLMBlock' for compatibility, but internally 
    this is now a Concatenation Block for stronger signal.
    """
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, condition):
        # Concatenation forces the network to look at the condition
        combined = torch.cat([x, condition], dim=-1)
        return x + self.mlp(combined)

class VectorDenoiserBackbone(nn.Module):
    def __init__(self, vector_dim=3, image_channels=3, hidden_dim=256, cond_dim=512, num_layers=6):
        super().__init__()
        self.encoder = EnvironmentEncoder(image_channels, cond_dim)
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_proj = nn.Linear(vector_dim, hidden_dim)
        
        self.layers = nn.ModuleList([FiLMBlock(hidden_dim, cond_dim) for _ in range(num_layers)])
        
        self.final_layer = nn.Linear(hidden_dim, vector_dim)

    def forward(self, x, t, images):
        cond_vec = self.encoder(images)
        t_emb = self.time_mlp(t)
        x = self.input_proj(x) + t_emb
        for layer in self.layers:
            x = layer(x, cond_vec)
        return self.final_layer(x)