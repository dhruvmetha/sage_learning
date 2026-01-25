import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class TrainingState:
    x_t: torch.Tensor      # The input to the network (Noisy)
    t: torch.Tensor        # The timestep
    target: torch.Tensor   # The ground truth (Velocity or Noise)

class BasePath(ABC):
    @abstractmethod
    def compute_loss_samples(self, x_0: torch.Tensor, x_1: torch.Tensor) -> TrainingState:
        pass

class BaseSampler(ABC):
    @abstractmethod
    def sample(self, model, x_init, num_steps, show_progress=False, device='cpu'):
        pass

# -----------------------------------------------------------------------------
# FLOW MATCHING
# -----------------------------------------------------------------------------

class FlowMatchingPath(BasePath):
    def __init__(self, sigma_min: float = 1e-5):
        self.sigma_min = sigma_min

    def compute_loss_samples(self, x_0: torch.Tensor, x_1: torch.Tensor) -> TrainingState:
        """
        Conditional Flow Matching with linear interpolation path.
        
        Path: psi_t(x) = (1-t) * x_0 + t * x_1
        Velocity: v_t = d(psi_t)/dt = x_1 - x_0
        
        The network learns to predict v_t given x_t and t.
        """
        B = x_0.shape[0]
        device = x_0.device
        t = torch.rand((B,), device=device)
        t_expand = t.view(B, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        # Correct velocity for linear interpolation CFM
        target_v = x_1 - x_0
        return TrainingState(x_t=x_t, t=t, target=target_v)

class ODESampler(BaseSampler):
    def __init__(self, method: str = 'euler'):
        self.method = method

    @torch.no_grad()
    def sample(self, model, x_init, num_steps, show_progress=False, device='cpu'):
        B = x_init.shape[0]
        x = x_init
        dt = 1.0 / num_steps
        iterator = range(num_steps)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Sampling")
        for i in iterator:
            t_val = i / num_steps
            t = torch.full((B,), t_val, device=device)
            v = model(x, t)
            x = x + v * dt
        return x

# -----------------------------------------------------------------------------
# DIFFUSION
# -----------------------------------------------------------------------------

class DiffusionPath(BasePath):
    def __init__(self, timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def compute_loss_samples(self, x_0: torch.Tensor, x_1: torch.Tensor) -> TrainingState:
        B = x_1.shape[0]
        device = x_1.device
        t = torch.randint(0, self.timesteps, (B,), device=device)
        alpha_bar = self.alphas_cumprod.to(device)[t].view(B, 1)
        noise = x_0 
        x_t = torch.sqrt(alpha_bar) * x_1 + torch.sqrt(1 - alpha_bar) * noise
        return TrainingState(x_t=x_t, t=t, target=noise)

class DDPMSampler(BaseSampler):
    def __init__(self, timesteps: int = 1000):
        self.timesteps = timesteps
        self.betas = torch.linspace(0.0001, 0.02, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    @torch.no_grad()
    def sample(self, model, x_init, num_steps, show_progress=False, device='cpu'):
        x = x_init
        iterator = reversed(range(self.timesteps))
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=self.timesteps)
        for t_idx in iterator:
            t = torch.full((x.shape[0],), t_idx, device=device)
            pred_noise = model(x, t)
            alpha = self.alphas[t_idx].to(device)
            alpha_bar = self.alphas_cumprod[t_idx].to(device)
            beta = self.betas[t_idx].to(device)
            if t_idx > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            coeff = (1 - alpha) / (torch.sqrt(1 - alpha_bar))
            x = (1 / torch.sqrt(alpha)) * (x - coeff * pred_noise) + torch.sqrt(beta) * noise
        return x