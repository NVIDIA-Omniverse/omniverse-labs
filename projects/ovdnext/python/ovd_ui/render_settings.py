# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated rendering options, independent of Qt's legacy preferences."""
from dataclasses import asdict, dataclass, fields
from enum import IntEnum
import math

from .palette import parse_color


class DisplayMode(IntEnum):
    NONE = 0
    ALL = 1
    SELECTED = 2
    UNSELECTED = 3


class PlaneMode(IntEnum):
    AUTO = 0
    NONE = 1
    BASIS = 2
    SCENE = 3


class FaceCulling(IntEnum):
    NONE = 0
    NORMAL = 1
    INVERSE = 2


RENDERING_SCHEMA = 2


@dataclass(frozen=True)
class RenderingSettings:
    anti_aliasing: bool = False
    face_culling: int = FaceCulling.NONE
    geometry_scale: float = 1.
    planes: int = PlaneMode.AUTO
    plane_scale: float = 1.  # OVUI enhancement: finite patch extent multiplier.
    particle_size: float = 5.
    show_particles: bool = True
    draw_orbit_center: bool = False
    sleeping_objects: int = DisplayMode.ALL
    trigger_shapes: int = DisplayMode.ALL
    wireframe: int = DisplayMode.NONE
    background_mode: str = "workspace"
    background_color: str = "#CCCCCC"

    def __post_init__(self):
        for key, enum in (("face_culling", FaceCulling), ("planes", PlaneMode),
                          ("sleeping_objects", DisplayMode), ("trigger_shapes", DisplayMode),
                          ("wireframe", DisplayMode)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, IntEnum)):
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be a supported mode")
            enum(value)
        for key, low, high in (("geometry_scale", 1e-6, 1e6), ("plane_scale", .01, 1000.),
                               ("particle_size", 0., 10.)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high or not math.isfinite(value):
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be between {low:g} and {high:g}")
        if any(not isinstance(getattr(self, key), bool)
               for key in ("anti_aliasing", "show_particles", "draw_orbit_center")):
            raise ValueError("Anti-aliasing, particle visibility and orbit center must be boolean")
        if self.background_mode not in ("workspace", "reference", "custom"):
            raise ValueError("Background must be Workspace, Reference gray, or Custom")
        parse_color(self.background_color)

    def background(self, workspace=None):
        if self.background_mode == "reference":
            return (.8, .8, .8, 1.)
        if self.background_mode == "custom":
            return (*parse_color(self.background_color)[:3], 1.)
        return workspace

    def to_dict(self):
        return asdict(self)

    def to_preferences(self):
        # Unlike the old implicit default, newly saved Normal is intentional.
        return {**self.to_dict(), "schema": RENDERING_SCHEMA}

    @classmethod
    def from_dict(cls, value):
        """Decode settings without applying persisted-preference migrations."""
        if not isinstance(value, dict):
            raise ValueError("Rendering preferences must be an object")
        keys = {field.name for field in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in keys})

    @classmethod
    def from_preferences(cls, value):
        """Decode saved preferences, migrating only the old implicit Normal."""
        if not isinstance(value, dict):
            raise ValueError("Rendering preferences must be an object")
        culling = value.get("face_culling")
        if ("schema" not in value and isinstance(culling, int) and not isinstance(culling, bool)
                and culling == FaceCulling.NORMAL):
            # Before v0.2.10 every rendering save wrote Normal, even unchanged.
            # Only unmarked legacy preferences migrate; future markers survive.
            value = {key: item for key, item in value.items() if key != "face_culling"}
        return cls.from_dict(value)


def display_mask(mode, selected):
    """Vector selection gate shared by drawing and object-ID picking."""
    if mode == DisplayMode.NONE:
        return False
    if mode == DisplayMode.ALL:
        return True
    return selected if mode == DisplayMode.SELECTED else ~selected
