# Model module exports

# Base classes
from .base import BasePath, PathSample, BaseSampler

# Unified generative module (supports both Flow Matching and Diffusion)
from .generative_module import GenerativeModule

__all__ = [
    # Base
    "BasePath",
    "PathSample",
    "BaseSampler",
    # Main module
    "GenerativeModule",
]

# Optional: Facebook flow matching
try:
    from .paths import FBFlowMatchingPath
    from .samplers import FBODESampler
    __all__.extend(["FBFlowMatchingPath", "FBODESampler"])
except ImportError:
    pass

# Optional: HuggingFace diffusion
try:
    from .paths import HFDiffusionPath
    from .samplers import HFDiffusionSampler
    __all__.extend(["HFDiffusionPath", "HFDiffusionSampler"])
except ImportError:
    pass
