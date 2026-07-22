# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construction and reuse policy for OVRTX sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import color_presentation
from . import ovrtx_scene_composition
from .render_requests import RenderRequest


COMPOSED_SCENE = "composed_scene"
RUNTIME_UPDATE = "runtime_update"
CameraPoseSource = Literal["composed_scene", "runtime_update"]

# OVRTX 0.3 requires session replacement for output resize. OVRTX 0.4 is
# expected to make resize reusable only with a verified live render-product
# resize plus camera projection path and composition identity that separates
# those changes from session-breaking scene changes. Changing this flag alone
# is invalid.
SAME_SESSION_OUTPUT_RESIZE_SUPPORTED = False


def _dimension(value: int) -> int:
    return max(1, int(value))


def _render_var_from_request(request: RenderRequest) -> str:
    """Resolve the render var the presentation mode classified for the request.

    The request already carries the classified ``color_presentation``
    diagnostics (``render_var`` is ``LdrColor`` for the LDR passthrough / any
    fail-closed path and ``HdrColor`` for scene-linear). Fall back to the LDR
    default when the mapping is missing or has no ``render_var`` so pre-task
    and direct-USD requests keep their existing session identity.
    """

    presentation = getattr(request, "color_presentation", None) or {}
    getter = getattr(presentation, "get", None)
    value = (
        getter("render_var", color_presentation.RENDER_VAR_LDR_COLOR)
        if callable(getter)
        else color_presentation.RENDER_VAR_LDR_COLOR
    )
    return str(value or color_presentation.RENDER_VAR_LDR_COLOR)


@dataclass(frozen=True)
class OvrtxSessionSpec:
    ovrtx_scene_composition: ovrtx_scene_composition.OvrtxSceneComposition
    sensor_paths: tuple[str, ...]
    width: int
    height: int
    camera_prim_path: str
    camera_pose_source: CameraPoseSource
    #: Resolved color-presentation render var (``LdrColor`` RGBA8 vs
    #: ``HdrColor`` RGBA16F) the session reads back. It joins session identity
    #: (render-quality-color-controls task02-02): a presentation-mode change
    #: that resolves to a different render var re-keys the session through the
    #: ordinary ``reuse_decision`` path (no ad-hoc teardown). The default keeps
    #: pre-task specs (LDR passthrough) unchanged.
    render_var: str = color_presentation.RENDER_VAR_LDR_COLOR

    def __post_init__(self) -> None:
        if not isinstance(self.ovrtx_scene_composition, ovrtx_scene_composition.OvrtxSceneComposition):
            raise TypeError("OVRTX scene composition must be an OvrtxSceneComposition")
        if not isinstance(self.camera_prim_path, str):
            raise TypeError("camera prim path must be a string")
        if self.camera_pose_source not in {COMPOSED_SCENE, RUNTIME_UPDATE}:
            raise ValueError(f"invalid camera pose source: {self.camera_pose_source}")
        object.__setattr__(
            self,
            "render_var",
            str(self.render_var or color_presentation.RENDER_VAR_LDR_COLOR),
        )
        object.__setattr__(
            self,
            "sensor_paths",
            ovrtx_scene_composition.normalize_sensor_paths(self.sensor_paths),
        )
        object.__setattr__(self, "width", _dimension(self.width))
        object.__setattr__(self, "height", _dimension(self.height))


@dataclass(frozen=True)
class OvrtxSessionReuseDecision:
    reuse: bool
    reason: str


def build_spec(request: RenderRequest) -> OvrtxSessionSpec:
    """Compose and freeze the exact desired OVRTX session input."""

    sensor_paths = ovrtx_scene_composition.normalize_sensor_paths(request.sensor_paths)
    width = _dimension(request.width)
    height = _dimension(request.height)
    render_var = _render_var_from_request(request)
    composition = ovrtx_scene_composition.compose(
        source_scene_path=request.input_usd_path,
        camera_prim_path=request.camera_prim_path,
        sensor_paths=sensor_paths,
        width=width,
        height=height,
        camera_projection=request.camera_projection,
        material_scene_layer=request.material_scene_layer,
        light_scene_layer=request.light_scene_layer,
        generate_scene_presentation=request.current_scene_generation,
        scene_camera_matrix=getattr(request, "scene_camera_matrix", None),
        camera_value_route_classes=tuple(
            getattr(request, "camera_value_route_classes", ()) or ()
        ),
        rtpt_quality=getattr(request, "rtpt_quality", None),
        rtpt_value_route=bool(getattr(request, "rtpt_value_route", False)),
        dlss_enabled=bool(getattr(request, "dlss_enabled", True)),
        render_vars=(render_var,),
    )
    return OvrtxSessionSpec(
        ovrtx_scene_composition=composition,
        sensor_paths=sensor_paths,
        width=width,
        height=height,
        camera_prim_path=str(request.camera_prim_path or ""),
        camera_pose_source=(
            RUNTIME_UPDATE if request.camera_matrix is not None else COMPOSED_SCENE
        ),
        render_var=render_var,
    )


def reuse_decision(
    current: OvrtxSessionSpec,
    desired: OvrtxSessionSpec,
) -> OvrtxSessionReuseDecision:
    """Evaluate the concrete OVRTX 0.3 session reuse policy in priority order."""

    if (
        not SAME_SESSION_OUTPUT_RESIZE_SUPPORTED
        and (current.width, current.height) != (desired.width, desired.height)
    ):
        return OvrtxSessionReuseDecision(False, "output_shape_changed")
    if current.camera_prim_path != desired.camera_prim_path:
        return OvrtxSessionReuseDecision(False, "camera_prim_changed")
    if current.render_var != desired.render_var:
        # The selected output is authored into the render product, so check it
        # before the resulting composition digest and preserve the useful
        # presentation-specific replacement reason.
        return OvrtxSessionReuseDecision(False, "render_var_changed")
    if (
        current.ovrtx_scene_composition.composed_scene_path != desired.ovrtx_scene_composition.composed_scene_path
        or current.ovrtx_scene_composition.digest != desired.ovrtx_scene_composition.digest
    ):
        return OvrtxSessionReuseDecision(False, "scene_composition_changed")
    if current.sensor_paths != desired.sensor_paths:
        return OvrtxSessionReuseDecision(False, "declared_sensors_changed")
    if (
        current.camera_pose_source == RUNTIME_UPDATE
        and desired.camera_pose_source == COMPOSED_SCENE
    ):
        return OvrtxSessionReuseDecision(False, "camera_pose_override_removed")
    return OvrtxSessionReuseDecision(True, "same_session")


__all__ = [
    "COMPOSED_SCENE",
    "RUNTIME_UPDATE",
    "SAME_SESSION_OUTPUT_RESIZE_SUPPORTED",
    "OvrtxSessionReuseDecision",
    "OvrtxSessionSpec",
    "build_spec",
    "reuse_decision",
]
