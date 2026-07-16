"""NanoUSD agentic Gaussian real-to-sim development oracle."""

from .core import Bounds, Collider, Joint, SceneNode, Workspace
from .mesh_completion import fit_mesh_pbr_completion
from .rlvr import EpisodeRequirements, RealToSimEpisode, RewardSnapshot

__all__ = [
    "Bounds",
    "Collider",
    "EpisodeRequirements",
    "fit_mesh_pbr_completion",
    "Joint",
    "RealToSimEpisode",
    "RewardSnapshot",
    "SceneNode",
    "Workspace",
]
__version__ = "0.1.0"
