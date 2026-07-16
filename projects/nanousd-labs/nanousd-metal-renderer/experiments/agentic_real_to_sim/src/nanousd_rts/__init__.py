"""NanoUSD agentic Gaussian real-to-sim development oracle."""

from .core import Bounds, Collider, Joint, SceneNode, Workspace
from .mesh_completion import fit_mesh_pbr_completion
from .learned_materials import generate_material_bundle
from .material_preview import write_material_comparison
from .rlvr import EpisodeRequirements, RealToSimEpisode, RewardSnapshot

__all__ = [
    "Bounds",
    "Collider",
    "EpisodeRequirements",
    "fit_mesh_pbr_completion",
    "generate_material_bundle",
    "write_material_comparison",
    "Joint",
    "RealToSimEpisode",
    "RewardSnapshot",
    "SceneNode",
    "Workspace",
]
__version__ = "0.1.0"
