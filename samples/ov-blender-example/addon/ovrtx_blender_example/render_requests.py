# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render requests and keys derived from Blender state."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Any, Mapping

from . import usd_paths as usd_paths
from .bundled_runtime import DEFAULT_OVRTX_NATIVE_CLIENT_MODULE
from .properties import DEFAULT_RENDER_PRODUCT_PATH


CAMERA_MATRIX_DIGITS = 7
DOF_FSTOP_SCALE = 100.0
CAMERA_CONTROLS_USD = "usd_camera"
CAMERA_CONTROLS_BLENDER_VIEW = "blender_view"
ENV_FIXED_VIEWPORT_RESOLUTION = "OV_BLENDER_EXAMPLE_FIXED_VIEWPORT_RESOLUTION"
LIVE_AUTHORING_CAMERA_PATH = "/World/OVRTXCamera"
OVRTX_SCENE_COMPOSITION_ROUTE = "ovrtx_scene_composition"
RUNTIME_PROJECTION_UNPROVEN = "same_session_runtime_unproven"
PERSPECTIVE_USER_VIEW = "perspective_user_view"
ORTHOGRAPHIC_USER_VIEW = "orthographic_user_view"
ACTIVE_CAMERA_VIEW = "active_camera_view"
CAMERA_PROJECTION_RUNTIME_CANDIDATES = (
    "projection",
    "focalLength",
    "horizontalAperture",
    "verticalAperture",
    "horizontalApertureOffset",
    "verticalApertureOffset",
    "fStop",
    "focusDistance",
)
CAMERA_PROJECTION_SESSION_IDENTITY_FIELDS = (
    "viewport_region_size",
    "render_product_resolution",
)
CAMERA_PROJECTION_COMPOSED_USD_ONLY_FIELDS = ("clippingRange",)


@dataclass(frozen=True)
class MaterialPresentationLayer:
    target_path: str
    layer_body: str
    authored_properties: tuple[tuple[str, str], ...]
    digest_content: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class RenderRequest:
    """Add-on-owned output of Blender render-signal translation."""

    input_usd_path: str = ""
    #: True when ``input_usd_path`` is a generation of the current Blender
    #: scene rather than an exact-stage validation input.
    current_scene_generation: bool = False
    sensor_paths: tuple[str, ...] = (DEFAULT_RENDER_PRODUCT_PATH,)
    selected_sensor_paths: tuple[str, ...] = (DEFAULT_RENDER_PRODUCT_PATH,)
    width: int = 64
    height: int = 64
    min_samples: int = 1
    max_samples: int = 128
    camera_prim_path: str = ""
    camera_matrix: tuple[tuple[float, ...], ...] | None = None
    camera_projection: Any | None = None
    #: Active scene camera's world transform in USD row convention, authored
    #: into the composed scene for final renders (COMPOSED_SCENE pose source).
    #: None on viewport requests, where the pose is a runtime update instead.
    scene_camera_matrix: tuple[tuple[float, ...], ...] | None = None
    #: Camera value probe classes currently routed as live value updates
    #: (blender-live-render task04-05). Attributes owned by these classes
    #: are excluded from the OVRTX scene composition digest so honored (and
    #: not-yet-probed) camera value edits stay out of session identity;
    #: unhonored classes are absent, folding their values back into the
    #: digest so ``reuse_decision`` forces a replacement. Empty (the
    #: default, used by F12 and direct requests) keeps every composed
    #: camera value in the digest — the pre-task behavior.
    camera_value_route_classes: tuple[str, ...] = ()
    #: Provenance of the authored scene generation materialized into
    #: ``input_usd_path`` (blender-live-render task05-04): ADR 0014's
    #: content digest and the generation number, stamped by
    #: ``engine._final_render_request_with_authored_input`` on the live
    #: final-render route and recorded by the render-result artifact so
    #: viewport/F12 parity is checkable artifact-to-artifact. Never part
    #: of session or composition identity; the defaults keep direct
    #: (env-override) and viewport requests untouched.
    authored_generation_digest: str = ""
    authored_generation: int | None = None
    worker_command: str = ""
    native_client_module: str = DEFAULT_OVRTX_NATIVE_CLIENT_MODULE
    timeline_controls_enabled: bool = False
    timeline_playing: bool = False
    timeline_frame: int = 1
    timeline_start: int = 1
    timeline_end: int = 1
    simulation_reset_token: int = 0
    material_scene_layer: MaterialPresentationLayer | None = None
    light_scene_layer: MaterialPresentationLayer | None = None
    color_presentation: Mapping[str, Any] = field(default_factory=dict)
    #: RTPT render-quality values keyed by ``RTPT_RENDER_SETTINGS`` property
    #: name (render-quality-color-controls task01-03). Authored onto the
    #: generated ``RenderProduct`` on every composition — viewport and F12
    #: alike — so the artist's values are deterministic session state.
    #: Empty (the default) authors the documented runtime defaults.
    rtpt_quality: Mapping[str, Any] = field(default_factory=dict)
    #: When True the RTPT quality attributes are excluded from the OVRTX scene
    #: composition digest so a live quality change (applied as a runtime
    #: attribute write on the render thread, render-quality-color-controls
    #: task01-04) does not change session identity — ``reuse_decision`` keeps
    #: the running session instead of replacing it. Set on the live viewport
    #: route; the default (False, used by F12 and direct requests) folds the
    #: values into the digest like task01-03, so a re-keyed session composes
    #: the new opinions.
    rtpt_value_route: bool = False
    #: DLSS Super-Resolution toggle (default True = worker default, DLSS on).
    #: A real-GPU A/B proved this RealTimePathTracing worker build always runs
    #: DLSS and exposes no full off; the only honored knob is
    #: ``/rtx/post/dlss/execMode``, and — unlike the rtpt family — it is honored
    #: as ``omni:rtx:post:dlss:execMode`` on the generated ``RenderProduct`` at
    #: SESSION creation, so it joins the composition digest and a change re-keys
    #: the session with no worker restart. False authors the Performance-preset
    #: execMode value (the strongest exposed change); True leaves the engine
    #: default. Also written to the worker-startup config for fresh workers.
    dlss_enabled: bool = True
    blender_signal: Mapping[str, Any] = field(default_factory=dict)

    @property
    def render_product_path(self) -> str:
        selected = tuple(path for path in self.selected_sensor_paths if path)
        declared = tuple(path for path in self.sensor_paths if path)
        return (selected or declared or (DEFAULT_RENDER_PRODUCT_PATH,))[0]


@dataclass(frozen=True)
class CameraProjectionState:
    source: str
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float
    projection: str = "perspective"
    horizontal_aperture_offset: float = 0.0
    vertical_aperture_offset: float = 0.0
    clipping_range: tuple[float, float] | None = None
    f_stop: float = 0.0
    focus_distance: float = 0.0
    viewport_region: tuple[int, int] | None = None
    render_size: tuple[int, int] = (0, 0)
    route: str = OVRTX_SCENE_COMPOSITION_ROUTE
    runtime_status: str = RUNTIME_PROJECTION_UNPROVEN
    dof_fstop_scale: float = DOF_FSTOP_SCALE
    sensor_fit: str = ""
    lens_shift: tuple[float, float] | None = None

    def usd_attributes(self) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            "projection": self.projection,
            "focalLength": self.focal_length,
            "horizontalAperture": self.horizontal_aperture,
            "verticalAperture": self.vertical_aperture,
            "horizontalApertureOffset": self.horizontal_aperture_offset,
            "verticalApertureOffset": self.vertical_aperture_offset,
            "fStop": self.f_stop if self.f_stop > 0.0 and self.focus_distance > 0.0 else 0.0,
        }
        if self.clipping_range is not None:
            attributes["clippingRange"] = self.clipping_range
        if self.f_stop > 0.0 and self.focus_distance > 0.0:
            attributes["focusDistance"] = self.focus_distance
        return attributes

    def identity_key(self) -> tuple[Any, ...]:
        return (
            self.source,
            self.projection,
            self.focal_length,
            self.horizontal_aperture,
            self.vertical_aperture,
            self.horizontal_aperture_offset,
            self.vertical_aperture_offset,
            self.clipping_range,
            self.f_stop,
            self.focus_distance,
            self.viewport_region,
            self.render_size,
            self.route,
            self.runtime_status,
            self.sensor_fit,
            self.lens_shift,
        )

    def to_diagnostics(self) -> dict[str, Any]:
        attributes = self.usd_attributes()
        viewport_region = (
            {
                "width": self.viewport_region[0],
                "height": self.viewport_region[1],
            }
            if self.viewport_region is not None
            else {}
        )
        return {
            "available": True,
            "source": self.source,
            "route": self.route,
            "runtime_write_status": self.runtime_status,
            "same_session_runtime_supported": False,
            "attributes": attributes,
            "candidate_runtime_attributes": list(CAMERA_PROJECTION_RUNTIME_CANDIDATES),
            "composed_usd_only_attributes": list(CAMERA_PROJECTION_COMPOSED_USD_ONLY_FIELDS),
            "session_identity_fields": list(CAMERA_PROJECTION_SESSION_IDENTITY_FIELDS),
            "viewport_region": viewport_region,
            "render_size": {
                "width": self.render_size[0],
                "height": self.render_size[1],
            },
            "render_aspect": (
                self.render_size[0] / self.render_size[1]
                if self.render_size[0] > 0 and self.render_size[1] > 0
                else 0.0
            ),
            "dof_fstop_scale": self.dof_fstop_scale,
            "sensor_fit": self.sensor_fit,
            "lens_shift": (
                {
                    "x": self.lens_shift[0],
                    "y": self.lens_shift[1],
                    "mapping_status": "candidate_mapping_unproven",
                }
                if self.lens_shift is not None
                else {}
            ),
        }


@dataclass(frozen=True)
class _Camera:
    width: int
    height: int
    camera_prim_path: str
    camera_matrix: tuple[tuple[float, ...], ...] | None
    camera_projection: CameraProjectionState | None
    camera_controls_mode: str


def tick(
    request: Any,
    *,
    now_ns: int | None = None,
) -> Any:
    from .runtime_scheduler import RuntimeTickRequest

    return RuntimeTickRequest(
        input_usd_path=request.input_usd_path,
        now_ns=now_ns,
        timeline_controls_enabled=request.timeline_controls_enabled,
        timeline_playing=request.timeline_playing,
        timeline_frame=request.timeline_frame,
        timeline_start=request.timeline_start,
        timeline_end=request.timeline_end,
        simulation_reset_token=request.simulation_reset_token,
    )


def camera(
    *,
    base_width: int,
    base_height: int,
    camera_prim_path: str,
    sync_viewport_camera: bool,
    context: Any | None,
    scene: Any | None = None,
) -> _Camera:
    known_camera_path = usd_paths.known_usd_path(camera_prim_path)
    camera_matrix = (
        camera_matrix_from_context(context)
        if sync_viewport_camera and known_camera_path
        else None
    )
    width = int(base_width)
    height = int(base_height)
    if camera_matrix is None:
        camera_projection = (
            camera_projection_from_scene_camera(scene, None, width, height)
            if context is None and known_camera_path
            else None
        )
        return _Camera(
            width=width,
            height=height,
            camera_prim_path=camera_prim_path,
            camera_matrix=None,
            camera_projection=camera_projection,
            camera_controls_mode=CAMERA_CONTROLS_USD,
        )

    if not os.environ.get(ENV_FIXED_VIEWPORT_RESOLUTION):
        width, height = synced_resolution(width, height, context)
    return _Camera(
        width=width,
        height=height,
        camera_prim_path=camera_prim_path,
        camera_matrix=camera_matrix,
        camera_projection=camera_projection_from_context(context, width, height),
        camera_controls_mode=CAMERA_CONTROLS_BLENDER_VIEW,
    )


def resolution_from_scene(scene: Any) -> tuple[int, int]:
    render = scene.render
    scale = render.resolution_percentage / 100.0
    return (
        clamp_dimension(int(render.resolution_x * scale)),
        clamp_dimension(int(render.resolution_y * scale)),
    )


def perspective_viewport_context(context: Any | None) -> tuple[Any, Any] | None:
    if context is None:
        return None
    region = getattr(context, "region", None)
    region_data = getattr(context, "region_data", None)
    if region is None or region_data is None:
        return None
    if str(getattr(region_data, "view_perspective", "")) not in {"PERSP", "ORTHO"}:
        return None
    return region, region_data


def synced_resolution(
    base_width: int,
    base_height: int,
    context: Any | None,
) -> tuple[int, int]:
    viewport_context = perspective_viewport_context(context)
    if viewport_context is None:
        return base_width, base_height
    region, _region_data = viewport_context
    try:
        region_width = int(getattr(region, "width"))
        region_height = int(getattr(region, "height"))
    except Exception:
        return base_width, base_height
    if region_width <= 0 or region_height <= 0:
        return base_width, base_height

    target_pixels = max(1, int(base_width) * int(base_height))
    divisor = math.gcd(region_width, region_height)
    ratio_width = max(1, region_width // max(1, divisor))
    ratio_height = max(1, region_height // max(1, divisor))
    ratio_pixels = ratio_width * ratio_height
    if ratio_pixels <= target_pixels:
        scale = max(1, int(math.sqrt(target_pixels / ratio_pixels)))
        return clamp_dimension(ratio_width * scale), clamp_dimension(ratio_height * scale)

    aspect = region_width / region_height
    height = max(1, int(round(math.sqrt(target_pixels / aspect))))
    width = max(1, int(round(height * aspect)))
    return clamp_dimension(width), clamp_dimension(height)


def camera_projection_from_context(
    context: Any | None,
    width: int,
    height: int,
) -> CameraProjectionState | None:
    if context is None or width <= 0 or height <= 0:
        return None
    viewport_context = projection_viewport_context(context)
    if viewport_context is None:
        return None
    region, region_data = viewport_context
    view_perspective = str(getattr(region_data, "view_perspective", ""))
    if view_perspective == "CAMERA":
        return camera_projection_from_scene_camera(getattr(context, "scene", None), region, width, height)
    if view_perspective == "PERSP":
        return perspective_projection_from_window_matrix(
            region,
            region_data,
            width,
            height,
        )
    if view_perspective == "ORTHO":
        return orthographic_projection_from_window_matrix(
            region,
            region_data,
            width,
            height,
        )
    return None


def projection_viewport_context(context: Any | None) -> tuple[Any, Any] | None:
    if context is None:
        return None
    region = getattr(context, "region", None)
    region_data = getattr(context, "region_data", None)
    if region is None or region_data is None:
        return None
    if str(getattr(region_data, "view_perspective", "")) not in {"PERSP", "CAMERA", "ORTHO"}:
        return None
    return region, region_data


def perspective_projection_from_window_matrix(
    region: Any,
    region_data: Any,
    width: int,
    height: int,
) -> CameraProjectionState | None:
    terms = projection_window_terms(region_data)
    if terms is None:
        return None
    window_matrix, window_x_scale, window_y_scale, horizontal_offset_term, vertical_offset_term = terms
    focal_length = 28.0
    horizontal_aperture = 2.0 * focal_length / abs(window_x_scale)
    vertical_aperture = 2.0 * focal_length / abs(window_y_scale)
    horizontal_aperture_offset = focal_length * horizontal_offset_term / window_x_scale
    vertical_aperture_offset = focal_length * vertical_offset_term / window_y_scale
    values = (
        focal_length,
        horizontal_aperture,
        vertical_aperture,
        horizontal_aperture_offset,
        vertical_aperture_offset,
    )
    if not all(math.isfinite(value) for value in values):
        return None
    if horizontal_aperture <= 0.0 or vertical_aperture <= 0.0:
        return None
    return CameraProjectionState(
        source=PERSPECTIVE_USER_VIEW,
        focal_length=round(focal_length, 9),
        horizontal_aperture=round(horizontal_aperture, 9),
        vertical_aperture=round(vertical_aperture, 9),
        projection="perspective",
        horizontal_aperture_offset=round(horizontal_aperture_offset, 9),
        vertical_aperture_offset=round(vertical_aperture_offset, 9),
        clipping_range=clipping_range_from_window_matrix(window_matrix),
        viewport_region=region_size(region),
        render_size=(int(width), int(height)),
    )


def orthographic_projection_from_window_matrix(
    region: Any,
    region_data: Any,
    width: int,
    height: int,
) -> CameraProjectionState | None:
    terms = projection_window_terms(region_data)
    if terms is None:
        return None
    _window_matrix, window_x_scale, window_y_scale, horizontal_offset_term, vertical_offset_term = terms
    focal_length = 28.0
    horizontal_aperture = 20.0 / abs(window_x_scale)
    vertical_aperture = 20.0 / abs(window_y_scale)
    horizontal_aperture_offset = -10.0 * horizontal_offset_term / window_x_scale
    vertical_aperture_offset = -10.0 * vertical_offset_term / window_y_scale
    values = (
        focal_length,
        horizontal_aperture,
        vertical_aperture,
        horizontal_aperture_offset,
        vertical_aperture_offset,
    )
    if not all(math.isfinite(value) for value in values):
        return None
    if horizontal_aperture <= 0.0 or vertical_aperture <= 0.0:
        return None
    return CameraProjectionState(
        source=ORTHOGRAPHIC_USER_VIEW,
        focal_length=round(focal_length, 9),
        horizontal_aperture=round(horizontal_aperture, 9),
        vertical_aperture=round(vertical_aperture, 9),
        projection="orthographic",
        horizontal_aperture_offset=round(horizontal_aperture_offset, 9),
        vertical_aperture_offset=round(vertical_aperture_offset, 9),
        clipping_range=None,
        viewport_region=region_size(region),
        render_size=(int(width), int(height)),
    )


def projection_window_terms(region_data: Any) -> tuple[Any, float, float, float, float] | None:
    window_matrix = getattr(region_data, "window_matrix", None)
    if window_matrix is None:
        return None
    try:
        window_x_scale = float(window_matrix[0][0])
        window_y_scale = float(window_matrix[1][1])
        horizontal_offset_term = float(window_matrix[0][2])
        vertical_offset_term = float(window_matrix[1][2])
    except Exception:
        return None
    if (
        not math.isfinite(window_x_scale)
        or not math.isfinite(window_y_scale)
        or abs(window_x_scale) < 1.0e-9
        or abs(window_y_scale) < 1.0e-9
    ):
        return None
    return (window_matrix, window_x_scale, window_y_scale, horizontal_offset_term, vertical_offset_term)


def camera_projection_from_scene_camera(
    scene: Any,
    region: Any,
    width: int,
    height: int,
) -> CameraProjectionState | None:
    camera_object = getattr(scene, "camera", None)
    data = getattr(camera_object, "data", None)
    if data is None:
        return None
    if str(getattr(data, "type", "PERSP")) == "ORTHO":
        return orthographic_projection_from_scene_camera(scene, region, width, height)
    try:
        focal_length = float(getattr(data, "lens"))
        sensor_width = float(getattr(data, "sensor_width"))
        sensor_height = float(getattr(data, "sensor_height", sensor_width))
    except Exception:
        return None
    if not all(math.isfinite(value) and value > 0.0 for value in (focal_length, sensor_width, sensor_height)):
        return None
    sensor_fit = str(getattr(data, "sensor_fit", "AUTO") or "AUTO")
    if sensor_fit == "VERTICAL" or (sensor_fit == "AUTO" and int(height) > int(width)):
        vertical_aperture = sensor_height if sensor_fit == "VERTICAL" else sensor_width
        horizontal_aperture = vertical_aperture * float(width) / float(height)
    else:
        horizontal_aperture = sensor_width
        vertical_aperture = horizontal_aperture * float(height) / float(width)
    clipping_range = camera_data_clipping_range(data)
    f_stop, focus_distance = camera_data_dof(data, camera_object)
    lens_shift = lens_shift_from_camera_data(data)
    return CameraProjectionState(
        source=ACTIVE_CAMERA_VIEW,
        focal_length=round(focal_length, 9),
        horizontal_aperture=round(horizontal_aperture, 9),
        vertical_aperture=round(vertical_aperture, 9),
        projection="perspective",
        horizontal_aperture_offset=0.0,
        vertical_aperture_offset=0.0,
        clipping_range=clipping_range,
        f_stop=round(f_stop * DOF_FSTOP_SCALE, 9) if f_stop > 0.0 and focus_distance > 0.0 else 0.0,
        focus_distance=round(focus_distance, 9) if f_stop > 0.0 and focus_distance > 0.0 else 0.0,
        viewport_region=region_size(region),
        render_size=(int(width), int(height)),
        sensor_fit=sensor_fit,
        lens_shift=lens_shift,
    )


def orthographic_projection_from_scene_camera(
    scene: Any | None,
    region: Any | None,
    width: int,
    height: int,
) -> CameraProjectionState | None:
    if scene is None or width <= 0 or height <= 0:
        return None
    camera_object = getattr(scene, "camera", None)
    data = getattr(camera_object, "data", None)
    if data is None or str(getattr(data, "type", "PERSP")) != "ORTHO":
        return None
    frame = orthographic_frame_from_scene_camera(data, scene)
    if frame is None:
        return None
    try:
        focal_length = float(getattr(data, "lens", 50.0) or 50.0)
    except Exception:
        focal_length = 50.0
    if not math.isfinite(focal_length) or focal_length <= 0.0:
        focal_length = 50.0
    frame_width, frame_height, frame_center_x, frame_center_y = frame
    horizontal_aperture = frame_width * 10.0
    vertical_aperture = frame_height * 10.0
    clipping_range = camera_data_clipping_range(data)
    f_stop, focus_distance = camera_data_dof(data, camera_object)
    lens_shift = lens_shift_from_camera_data(data)
    return CameraProjectionState(
        source=ACTIVE_CAMERA_VIEW,
        focal_length=round(focal_length, 9),
        horizontal_aperture=round(horizontal_aperture, 9),
        vertical_aperture=round(vertical_aperture, 9),
        projection="orthographic",
        horizontal_aperture_offset=round(frame_center_x * 10.0, 9),
        vertical_aperture_offset=round(frame_center_y * 10.0, 9),
        clipping_range=clipping_range,
        f_stop=round(f_stop * DOF_FSTOP_SCALE, 9) if f_stop > 0.0 and focus_distance > 0.0 else 0.0,
        focus_distance=round(focus_distance, 9) if f_stop > 0.0 and focus_distance > 0.0 else 0.0,
        viewport_region=region_size(region) if region is not None else None,
        render_size=(int(width), int(height)),
        sensor_fit=str(getattr(data, "sensor_fit", "ORTHOGRAPHIC") or "ORTHOGRAPHIC"),
        lens_shift=lens_shift,
    )


def orthographic_frame_from_scene_camera(data: Any, scene: Any) -> tuple[float, float, float, float] | None:
    view_frame = getattr(data, "view_frame", None)
    if not callable(view_frame):
        return None
    try:
        frame = tuple(view_frame(scene=scene))
        xs = [float(vertex.x) for vertex in frame]
        ys = [float(vertex.y) for vertex in frame]
    except Exception:
        return None
    if not xs or not ys:
        return None
    frame_width = max(xs) - min(xs)
    frame_height = max(ys) - min(ys)
    if not all(math.isfinite(value) and value > 0.0 for value in (frame_width, frame_height)):
        return None
    frame_center_x = (max(xs) + min(xs)) * 0.5
    frame_center_y = (max(ys) + min(ys)) * 0.5
    if not all(math.isfinite(value) for value in (frame_center_x, frame_center_y)):
        return None
    return (frame_width, frame_height, frame_center_x, frame_center_y)


def orthographic_frame_size_from_scene_camera(data: Any, scene: Any) -> tuple[float, float] | None:
    frame = orthographic_frame_from_scene_camera(data, scene)
    if frame is None:
        return None
    return (frame[0], frame[1])


def camera_projection_usd_attributes(projection: Any) -> dict[str, Any]:
    if projection is None:
        return {}
    attributes = getattr(projection, "usd_attributes", None)
    if callable(attributes):
        return dict(attributes())
    return {}


def camera_projection_key(projection: Any) -> Any:
    if projection is None:
        return None
    key = getattr(projection, "identity_key", None)
    if callable(key):
        return key()
    return projection


def camera_projection_diagnostics(projection: Any) -> dict[str, Any]:
    if projection is None:
        return {"available": False}
    diagnostics = getattr(projection, "to_diagnostics", None)
    if callable(diagnostics):
        return dict(diagnostics())
    return {"available": False}


def clipping_range_from_window_matrix(window_matrix: Any) -> tuple[float, float] | None:
    try:
        z_scale = float(window_matrix[2][2])
        z_offset = float(window_matrix[2][3])
    except Exception:
        return None
    near_denominator = z_scale - 1.0
    far_denominator = z_scale + 1.0
    if abs(near_denominator) < 1.0e-9 or abs(far_denominator) < 1.0e-9:
        return None
    near = z_offset / near_denominator
    far = z_offset / far_denominator
    if not all(math.isfinite(value) for value in (near, far)):
        return None
    near, far = sorted((max(1.0e-4, near), max(1.0, far)))
    return (round(near, 9), round(far, 9))


def camera_data_clipping_range(data: Any) -> tuple[float, float] | None:
    try:
        near = float(getattr(data, "clip_start"))
        far = float(getattr(data, "clip_end"))
    except Exception:
        return None
    if not all(math.isfinite(value) and value > 0.0 for value in (near, far)):
        return None
    near, far = sorted((near, far))
    return (round(near, 9), round(far, 9))


def camera_data_dof(data: Any, camera_object: Any) -> tuple[float, float]:
    dof = getattr(data, "dof", None)
    if dof is None or not bool(getattr(dof, "use_dof", False)):
        return (0.0, 0.0)
    try:
        f_stop = float(getattr(dof, "aperture_fstop", 0.0) or 0.0)
    except Exception:
        f_stop = 0.0
    focus_object = getattr(dof, "focus_object", None)
    if focus_object is not None:
        try:
            focus_distance = float(
                (
                    focus_object.matrix_world.translation
                    - camera_object.matrix_world.translation
                ).length
            )
        except Exception:
            focus_distance = 0.0
    else:
        try:
            focus_distance = float(getattr(dof, "focus_distance", 0.0) or 0.0)
        except Exception:
            focus_distance = 0.0
    if not all(math.isfinite(value) and value > 0.0 for value in (f_stop, focus_distance)):
        return (0.0, 0.0)
    return (f_stop, focus_distance)


def lens_shift_from_camera_data(data: Any) -> tuple[float, float] | None:
    try:
        shift_x = round(float(getattr(data, "shift_x", 0.0) or 0.0), 9)
        shift_y = round(float(getattr(data, "shift_y", 0.0) or 0.0), 9)
    except Exception:
        return None
    if not all(math.isfinite(value) for value in (shift_x, shift_y)):
        return None
    return (shift_x, shift_y)


def region_size(region: Any) -> tuple[int, int] | None:
    try:
        width = int(getattr(region, "width"))
        height = int(getattr(region, "height"))
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def camera_matrix_from_context(context: Any | None) -> tuple[tuple[float, ...], ...] | None:
    if context is None:
        return None
    region_data = getattr(context, "region_data", None)
    if str(getattr(region_data, "view_perspective", "")) not in {"PERSP", "CAMERA", "ORTHO"}:
        return None
    view_matrix = getattr(region_data, "view_matrix", None)
    if view_matrix is None:
        return None
    try:
        camera_world = view_matrix.inverted()
    except Exception:
        return None
    return matrix_to_usd_rows(camera_world)


def matrix_to_usd_rows(matrix: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(round(float(matrix[column][row]), 9) for column in range(4))
        for row in range(4)
    )


def scene_camera_world_matrix(scene: Any | None) -> tuple[tuple[float, ...], ...] | None:
    """Return the active scene camera's world transform in USD row convention."""

    camera_object = getattr(scene, "camera", None)
    matrix = getattr(camera_object, "matrix_world", None)
    if matrix is None:
        return None
    try:
        return matrix_to_usd_rows(matrix)
    except Exception:
        return None


def scene_camera_pose_delta(context: Any | None) -> float | None:
    if context is None:
        return None
    region_data = getattr(context, "region_data", None)
    view_matrix = getattr(region_data, "view_matrix", None)
    scene = getattr(context, "scene", None)
    scene_camera = getattr(scene, "camera", None)
    camera_matrix = getattr(scene_camera, "matrix_world", None)
    if view_matrix is None or camera_matrix is None:
        return None
    try:
        view_world = view_matrix.inverted()
        return round(
            max(
                abs(float(view_world[row][column]) - float(camera_matrix[row][column]))
                for row in range(4)
                for column in range(4)
            ),
            9,
        )
    except Exception:
        return None


def stable_camera_matrix(
    matrix: tuple[tuple[float, ...], ...] | None,
) -> tuple[tuple[float, ...], ...] | None:
    if matrix is None:
        return None
    rows: list[tuple[float, ...]] = []
    for row in matrix:
        values: list[float] = []
        for value in row:
            rounded = round(float(value), CAMERA_MATRIX_DIGITS)
            values.append(0.0 if rounded == 0.0 else rounded)
        rows.append(tuple(values))
    return tuple(rows)


def reset_reason(
    *,
    composition_changed: bool = False,
    camera_changed: bool = False,
    snapshot_changed: bool = False,
    value_edit: bool = False,
) -> str:
    if composition_changed:
        return "composition_changed"
    if camera_changed:
        return "camera_changed"
    if snapshot_changed:
        return "snapshot_changed"
    if value_edit:
        # Pure value-update batch (task04-06): an applied live value edit
        # reset refinement without any camera/composition/snapshot change.
        return "value_edit"
    return ""


def viewport_sampling_due(completed_samples: int, max_samples: int) -> bool:
    """Whether a viewport may acquire another sample (``0`` is unbounded)."""

    return int(max_samples) == 0 or int(completed_samples) < int(max_samples)


def clamp_dimension(value: int) -> int:
    return max(1, min(16384, int(value)))
