"""Path implementations for generative models.

Paths define the forward process (how to add noise/interpolate).

Available:
- FBFlowMatchingPath: Facebook's flow_matching library (requires: pip install flow-matching)
- HFDiffusionPath: HuggingFace's diffusers library (requires: pip install diffusers)
"""

from ..base import BasePath

__all__ = ["BasePath"]

# Facebook's flow_matching library path
try:
    from .fb_flow_matching_path import FBFlowMatchingPath
    __all__.append("FBFlowMatchingPath")
except ImportError:
    pass

# HuggingFace diffusers library path
try:
    from .hf_diffusion_path import HFDiffusionPath
    __all__.append("HFDiffusionPath")
except ImportError:
    pass
