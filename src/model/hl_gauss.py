"""HL-Gauss value head (Imani & White; "Stop Regressing" arXiv:2403.03950).

Regress a scalar value in [vmin, vmax] as a CLASSIFICATION over `num_bins` bins with a Gaussian-smoothed
soft target, instead of MSE on a raw scalar. Classification value heads beat regression for RL/value
learning. Used by the budget-conditioned horizon-Q (gamma-discounted targets in [0,1]; see
horizon_q_build_journal.md). Value at inference = E[bin] under softmax.
"""
import torch
import torch.nn.functional as F


class HLGauss:
    def __init__(self, num_bins: int = 51, vmin: float = 0.0, vmax: float = 1.0, sigma_ratio: float = 0.75):
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        edges = torch.linspace(vmin, vmax, num_bins + 1)
        self.bin_edges = edges                                  # (num_bins+1,)
        self.bin_centers = 0.5 * (edges[:-1] + edges[1:])       # (num_bins,)
        self.sigma = sigma_ratio * (vmax - vmin) / num_bins

    def to(self, device):
        self.bin_edges = self.bin_edges.to(device)
        self.bin_centers = self.bin_centers.to(device)
        return self

    def target(self, y: torch.Tensor) -> torch.Tensor:
        """y: (...) scalar in [vmin,vmax] -> (..., num_bins) soft histogram via Gaussian CDF differences."""
        edges = self.bin_edges.to(y.device)                     # (num_bins+1,)
        y = y.clamp(self.vmin, self.vmax).unsqueeze(-1)         # (...,1)
        cdf = 0.5 * (1.0 + torch.erf((edges - y) / (self.sigma * (2.0 ** 0.5))))  # (..., num_bins+1)
        probs = cdf[..., 1:] - cdf[..., :-1]                    # (..., num_bins)
        return probs / probs.sum(-1, keepdim=True).clamp_min(1e-8)

    def loss(self, logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """logits: (..., num_bins); y: (...); mask: (...) -> mean masked cross-entropy to the soft target."""
        tgt = self.target(y)                                    # (..., num_bins)
        logp = F.log_softmax(logits, dim=-1)
        ce = -(tgt * logp).sum(-1)                              # (...)
        if mask is not None:
            mask = mask.float()
            return (ce * mask).sum() / mask.sum().clamp_min(1.0)
        return ce.mean()

    def value(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (..., num_bins) -> (...) expected value E[bin] in [vmin,vmax]."""
        p = F.softmax(logits, dim=-1)
        return (p * self.bin_centers.to(logits.device)).sum(-1)
