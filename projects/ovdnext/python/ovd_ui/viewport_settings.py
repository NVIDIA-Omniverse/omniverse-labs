# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viewport choices, independent of capture-owned interaction IDs.

Transparency amount is the UI fraction (0 opaque, 1 invisible), the inverse of
the original Qt transparencyAmount opacity. No legacy preference store is read.
"""
from dataclasses import asdict, dataclass, fields, replace
import math
import numpy as np

from .interaction import TransparencyMode
from .render_settings import DisplayMode


PREFERENCE_KEY = "viewport_display"
MODE_FIELDS = {1: "bounds", 2: "actor_axes", 4: "linear_velocity", 16: "joints",
               32: "contacts", 256: "center_of_mass"}
TOGGLE_FIELDS = {64: "scene_queries"}
SCALE_FIELDS = {2: "axes_scale", 4: "linear_velocity_scale", 8: "angular_velocity_scale",
                32: "contact_scale", 256: "com_scale"}
OWNED_MASK = sum((*MODE_FIELDS, *TOGGLE_FIELDS))
GIZMO_MIN = .0001
FLOAT32_MAX = float.fromhex("0x1.fffffep+127")


def gizmo_value(value):
    """Preserve the existing control minimum, reject invalid native scalars."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not -FLOAT32_MAX <= value <= FLOAT32_MAX or not math.isfinite(value)):
        raise ValueError("Gizmo scale must be a finite number within the native float range")
    return max(GIZMO_MIN, float(value))


@dataclass(frozen=True)
class ViewportDisplaySettings:
    bounds: int = DisplayMode.NONE
    actor_axes: int = DisplayMode.NONE
    linear_velocity: int = DisplayMode.NONE
    joints: int = DisplayMode.NONE
    contacts: int = DisplayMode.NONE
    scene_queries: bool = False
    transparency_mode: int = TransparencyMode.NONE
    transparency_amount: float = .5
    gizmo_scale: float = 1.
    center_of_mass: int = DisplayMode.NONE
    axes_scale: float = 1.
    linear_velocity_scale: float = 1.
    angular_velocity_scale: float = 1.
    contact_scale: float = 1.
    com_scale: float = 1.
    master_scale: float = 1.

    def __post_init__(self):
        for name in (*MODE_FIELDS.values(), "transparency_mode"):
            value = getattr(self, name)
            enum = TransparencyMode if name == "transparency_mode" else DisplayMode
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name.replace('_', ' ').capitalize()} must be a supported mode")
            enum(value)
        for name in TOGGLE_FIELDS.values():
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name.replace('_', ' ').capitalize()} must be boolean")
        amount = self.transparency_amount
        if (isinstance(amount, bool) or not isinstance(amount, (int, float))
                or not 0 <= amount <= 1):
            raise ValueError("Transparency amount must be between 0 and 1")
        for name in ("master_scale", "gizmo_scale", *SCALE_FIELDS.values()):
            if gizmo_value(getattr(self, name)) != getattr(self, name):
                raise ValueError(f"{name} must be at least {GIZMO_MIN:g}")
        with np.errstate(over="ignore"):
            products=np.asarray([self.gizmo_scale,*self.scales.values()],dtype="f4")*np.float32(self.master_scale)
        if not np.isfinite(products).all():
            raise ValueError("Combined master and family scale exceeds the native float range")

    @classmethod
    def from_preferences(cls, preferences):
        """Malformed blocks use existing defaults without rewriting any key."""
        value = preferences.get(PREFERENCE_KEY, {})
        try:
            if not isinstance(value, dict):
                raise ValueError("Viewport preferences must be an object")
            names = {field.name for field in fields(cls)}
            values = {key: item for key, item in value.items() if key in names}
            for name in ("contacts", "joints"):  # Older preferences used on/off toggles.
                if isinstance(values.get(name), bool):
                    values[name] = int(values[name])
            for name in SCALE_FIELDS.values():
                values.setdefault(name, values.get("gizmo_scale", 1.))
            return cls(**values)
        except (ValueError, TypeError, OverflowError):
            return cls()

    def to_dict(self):
        return asdict(self)

    @property
    def modes(self):
        return {bit: DisplayMode(getattr(self, name)) for bit, name in MODE_FIELDS.items()}

    @property
    def scales(self):
        return {bit: getattr(self, name) for bit, name in SCALE_FIELDS.items()}

    @property
    def mask(self):
        return sum(bit for bit, name in (*MODE_FIELDS.items(), *TOGGLE_FIELDS.items())
                   if getattr(self, name))

    def visual_reset(self, gizmo_scale):
        # Original reset omits the joints visibility selector. Other extensions
        # are outside this model and remain owned by the application's session.
        return replace(self, bounds=DisplayMode.NONE, actor_axes=DisplayMode.NONE,
                       linear_velocity=DisplayMode.NONE, contacts=DisplayMode.NONE, scene_queries=False,
                       transparency_mode=TransparencyMode.NONE, transparency_amount=.5,
                       gizmo_scale=gizmo_scale, master_scale=1.,
                       **{name: gizmo_scale for name in SCALE_FIELDS.values()})
