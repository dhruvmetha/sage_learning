"""Sampler implementations for generative models.

Samplers define the reverse process (how to generate samples from noise).

Available:
- FBODESampler: Facebook's flow_matching library (requires: pip install flow-matching torchdyn)
- HFDiffusionSampler: HuggingFace's diffusers library (requires: pip install diffusers)
"""

from ..base import BaseSampler

__all__ = ["BaseSampler"]

# Facebook's flow_matching library sampler
try:
    from .fb_ode_sampler import FBODESampler
    __all__.append("FBODESampler")
except ImportError:
    pass

# HuggingFace diffusers library sampler
try:
    from .hf_diffusion_sampler import HFDiffusionSampler
    __all__.append("HFDiffusionSampler")
except ImportError:
    pass
