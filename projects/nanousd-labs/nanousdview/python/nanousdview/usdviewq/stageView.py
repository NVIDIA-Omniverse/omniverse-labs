# SPDX-License-Identifier: Apache-2.0
"""
nanousdview/usdviewq/stageView.py — OVRTX-backed viewport widget.

REPLACEMENT for Pixar's usdview stageView.py (vendored at
_stageView_pixar_orig.py for reference). The original is a 2,444-line
QGLWidget that drives UsdImagingGL.Engine through Hydra. This is a
QWidget that drives an OVRTX renderer implementation directly. No Hydra.
No UsdImagingGL. No GLSLProgram, GLSL HUD,
Outline/Reticles/Mask Prim2D classes — all that's in the original
file if anyone ever needs to crib from it.

Public API surface mirrors the original closely enough that the
vendored appController.py works unmodified.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from .qt import QtCore, QtGui, QtWidgets

from pxr import Gf, Sdf, Usd, UsdGeom, Tf

# Reuse Pixar's freeCamera (renderer-agnostic — orbit math + Gf.Camera).
from .freeCamera import FreeCamera

from .common import (
    RenderModes, ColorCorrectionModes, IncludedPurposes, ShadedRenderModes,
    PickModes,
)

from .gizmo import (
    TransformGizmo, AppState as GizmoAppState, RigidTransform,
    Mode as GizmoMode, make_pick_ray, decompose_trs, write_xform_to_prim,
)

try:
    from nanousdview._backend import (
        OvrtxViewportRenderer,
        VIEW_RENDER_RASTER,
        VIEW_RENDER_SHADOW,
        VIEW_RENDER_RT,
        get_backend,
    )
    _RENDERER_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    _RENDERER_AVAILABLE = False
    OvrtxViewportRenderer = None
    VIEW_RENDER_RT = 2; VIEW_RENDER_RASTER = 0; VIEW_RENDER_SHADOW = 1
    _RENDERER_IMPORT_ERROR = _e


# Rendering mode display names (must match what RenderModes exposes).
# Approximate ACES filmic tonemap (Krzysztof Narkowicz) — single-RGB curve
# applied after gamma decode. Used to bring vulkan + opengl renders into
# the same character as ovrtx/Kit, which run a real photographic exposure
# pipeline before tone-map. Already-clipped highlights stay at 255 — full
# parity needs HDR-float renderer output.
_ACES_A, _ACES_B, _ACES_C, _ACES_D, _ACES_E = 2.51, 0.03, 2.43, 0.59, 0.14
# Pre-tonemap exposure scale matching Kit's photographic-exposure formula
# (responsivity * filmIso / (100 * shutter * fNumber²)) at default values
# — but inverted/biased upward since the input is already LDR-clamped
# (we can't recover dynamic range, only compress what's there). Empirical
# 0.85 is what brings vulkan output into ovrtx-like brightness on the
# chess scene without crushing midtones.
_TONEMAP_PRE_EXP = 0.85


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * np.power(x, 1.0 / 2.4) - a)


def _aces_tonemap_full(rgba: np.ndarray) -> np.ndarray:
    """Reference per-pixel ACES tonemap (full float32 numpy pass).

    Kept as the source of truth for the cached LUT and the equivalence
    self-test; NOT called per frame (see _apply_aces_tonemap). Every op here
    is element-wise on uint8 RGBA with no spatial or cross-channel coupling,
    which is what makes the LUT below exact.
    """
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    lin = _srgb_to_linear(rgb) * _TONEMAP_PRE_EXP
    aces = (lin * (_ACES_A * lin + _ACES_B)) / (lin * (_ACES_C * lin + _ACES_D) + _ACES_E)
    aces = np.clip(aces, 0.0, 1.0)
    out_rgb = (_linear_to_srgb(aces) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    out = np.empty_like(rgba)
    out[..., :3] = out_rgb
    out[..., 3] = rgba[..., 3]
    return np.ascontiguousarray(out)


def _build_aces_lut() -> np.ndarray:
    """256-entry uint8 LUT == _aces_tonemap_full evaluated at every input.

    Because the tonemap is a per-channel 1D function on uint8 input, the full
    image pass equals a table lookup. The LUT is computed by running the
    reference function itself on a 0..255 ramp, so it is bit-identical by
    construction regardless of numpy scalar-promotion rules.
    """
    ramp = np.zeros((1, 256, 4), dtype=np.uint8)
    ramp[0, :, 0] = ramp[0, :, 1] = ramp[0, :, 2] = np.arange(256, dtype=np.uint8)
    ramp[0, :, 3] = 255
    return np.ascontiguousarray(_aces_tonemap_full(ramp)[0, :, 0])


_ACES_LUT = _build_aces_lut()


def _apply_aces_tonemap(rgba: np.ndarray) -> np.ndarray:
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[-1] != 4:
        return rgba
    # The tonemap is a per-channel 1D curve on uint8, so a cached 256-entry LUT
    # is bit-identical to the full float32 pass at a fraction of the cost (the
    # dominant viewer stall once 1a landed). Gathering all 4 channels in one
    # contiguous pass then restoring the (untonemapped) alpha is ~2x faster
    # than a strided RGB-only gather. See VULKAN_RT_ORBIT_PERF_PLAN.md (1b).
    out = _ACES_LUT[rgba]
    out[..., 3] = rgba[..., 3]
    return np.ascontiguousarray(out)


_RENDER_MODE_TO_VIEW = {
    RenderModes.SMOOTH_SHADED: VIEW_RENDER_RT,
    RenderModes.WIREFRAME: VIEW_RENDER_RASTER,
    RenderModes.FLAT_SHADED: VIEW_RENDER_RASTER,
    RenderModes.WIREFRAME_ON_SURFACE: VIEW_RENDER_RASTER,
    RenderModes.POINTS: VIEW_RENDER_RASTER,
    RenderModes.GEOM_ONLY: VIEW_RENDER_RASTER,
    RenderModes.GEOM_FLAT: VIEW_RENDER_RASTER,
    RenderModes.GEOM_SMOOTH: VIEW_RENDER_RASTER,
    RenderModes.HIDDEN_SURFACE_WIREFRAME: VIEW_RENDER_RASTER,
}


def _default_view_render_mode():
    try:
        backend = get_backend()
    except Exception:
        return VIEW_RENDER_RASTER
    if backend in ("metal", "ovrtx"):
        return VIEW_RENDER_RT
    return VIEW_RENDER_RASTER


def _supported_view_render_modes():
    try:
        backend = get_backend()
        if backend == "opengl":
            return [VIEW_RENDER_RASTER]
        if backend == "ovrtx":
            return [VIEW_RENDER_RT]
    except Exception:
        return [VIEW_RENDER_RASTER]
    return [VIEW_RENDER_RT, VIEW_RENDER_SHADOW, VIEW_RENDER_RASTER]


class StageView(QtWidgets.QWidget):
    """OVRTX-backed viewport. Drop-in for Pixar's QGLWidget
    StageView, scoped to the API surface appController.py actually uses."""

    # ---- Signals ----
    signalPrimSelected = QtCore.Signal(object, int, object, int, str, str)
    signalPrimRollover = QtCore.Signal(object, int, object, int)
    signalMouseDrag = QtCore.Signal()
    signalErrorMessage = QtCore.Signal(str)
    signalSwitchedToFreeCam = QtCore.Signal()
    signalFrustumChanged = QtCore.Signal()

    # Camera mode constants (used by appController menu wiring).
    DefaultDataModel = type("DefaultDataModel", (), {})

    def __init__(self, parent=None, dataModel=None, makeTimer=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(320, 240)

        self._dataModel = dataModel or StageView.DefaultDataModel()
        self._makeTimer = makeTimer
        self._allowAsync = False
        self._bboxstandin = False
        self._rolloverPicking = False
        self._isFirstImage = True
        self._lastComputedGfCamera = None
        self._lastAspectRatio = 1.0
        self._reportedContextError = False

        # Viewport image we display (kept as numpy + QImage).
        self._frame_pixels: np.ndarray | None = None
        self._frame_qimage: QtGui.QImage | None = None
        self._frame_dirty = True
        self._stage_loaded_path: str | None = None
        self._authored_light_cache_key = None
        self._authored_light_cache_value = False

        # Renderer instance (lazy: created on first paint when we know the size)
        self._renderer: "OvrtxViewportRenderer | None" = None
        # Default to the viewer's parity path: Smooth Shaded should match the
        # raster-backed OpenGL/Storm-style viewport. NOTE: on Vulkan the View
        # menu's RT selection is currently force-mapped back to raster in
        # _renderModeForViewSetting (live RT/raster viewport parity is still WIP),
        # so interactive RT is NOT reachable there; explicit RT comes only from
        # the NANOUSD_VIEW_RENDER_MODE / --render-mode override below (used by
        # capture scripts for apples-to-apples comparisons).
        self._render_mode = _default_view_render_mode()
        # Allow CLI/env override so capture scripts can produce
        # apples-to-apples comparisons across modes (e.g. vulkan-rt vs
        # vulkan-shadow vs vulkan-raster).
        _mode_override = (os.environ.get("NANOUSD_VIEW_RENDER_MODE") or "").lower()
        _has_mode_override = False
        if _mode_override in ("rt", "raytrace", "raytraced"):
            self._render_mode = VIEW_RENDER_RT
            _has_mode_override = True
        elif _mode_override in ("shadow", "raster_shadow", "raster+shadow"):
            self._render_mode = VIEW_RENDER_SHADOW
            _has_mode_override = True
        elif _mode_override in ("raster", "raster_only"):
            self._render_mode = VIEW_RENDER_RASTER
            _has_mode_override = True
        if self._render_mode not in _supported_view_render_modes():
            self._render_mode = _supported_view_render_modes()[0]
        if not _has_mode_override:
            view_render_mode = self._renderModeForViewSetting(
                self._viewSettingsRenderMode())
            if view_render_mode is not None:
                self._render_mode = view_render_mode
        self._last_view_settings_render_mode = self._viewSettingsRenderMode()
        self._render_error: str | None = None

        self._explicit_camera = self._parse_explicit_camera(
            os.environ.get("NANOUSD_VIEW_CAMERA"))

        # Camera state — start with FreeCamera (orbit). WASD mode toggled with F.
        self._fly_mode = False
        self._fly_pos = np.array([5.0, 3.0, 5.0], dtype=np.float64)
        self._fly_yaw = 0.0
        self._fly_pitch = 0.0
        self._fly_speed = 1.0  # m/s; rescaled per scene diagonal

        # Quaternion-based orbit orientation in navigation-local axes.
        # Navigation local +Y is the stage-authored up axis; for a Y-up
        # stage the local frame is identity, while for a Z-up stage local
        # +Y maps to world +Z.
        self._orbit_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._arcball_start: np.ndarray | None = None

        # Mouse state for orbit + fly look.
        self._mouse_down = None
        self._last_mouse_pos = QtCore.QPoint()
        self._wasd_keys: set[int] = set()

        # Tick timer for WASD update + redraw.
        self._tick = QtCore.QTimer(self)
        self._tick.setInterval(16)  # ~60 FPS
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()

        # HUD slots — appController writes directly to these.
        self.fpsHUDInfo = {}
        self.fpsHUDKeys = []
        self.upperHUDInfo = {}
        self.HUDStatKeys = []
        self.camerasWithGuides = []
        # Visibility flag for the on-viewport HUD (mode/FPS overlay). Toggled
        # by the View menu's "Show HUD" action and by the H key.
        self._show_hud = True

        # Transform gizmo (translate/rotate/scale manipulator). Drawn over
        # the rendered image with QPainter, anchored at the selected prim's
        # world origin. 1/2/3 switch mode; G toggles visibility.
        # Currently disabled by default — flip _show_gizmo and uncomment the
        # G/1/2/3 keybindings in keyPressEvent to re-enable.
        self._gizmo = TransformGizmo()
        self._gizmo_xform = RigidTransform()
        self._gizmo_app_state = GizmoAppState()
        self._gizmo_dragging = False
        self._show_gizmo = False

        # Hook data-model signals if available.
        try:
            self._dataModel.viewSettings.signalVisibleSettingChanged.connect(
                self._visibleSettingChanged)
            self._dataModel.viewSettings.signalFreeCameraSettingChanged.connect(
                self._onFreeCameraSettingChanged)
            self._dataModel.viewSettings.signalAutoComputeClippingPlanesChanged.connect(
                self._onAutoComputeClippingChanged)
            self._dataModel.signalStageReplaced.connect(self._stageReplaced)
            self._dataModel.selection.signalPrimSelectionChanged.connect(
                self._primSelectionChanged)
        except Exception:
            pass

        # Seed a FreeCamera so view-settings dialog works.
        try:
            self._dataModel.viewSettings.freeCamera = self._createNewFreeCamera(
                self._dataModel.viewSettings, self._stage_is_z_up())
        except Exception:
            self._dataModel.viewSettings = type("_VS", (), {})()
            self._dataModel.viewSettings.freeCamera = None

    def _viewSettingsRenderMode(self):
        try:
            return self._dataModel.viewSettings.renderMode
        except Exception:
            return None

    def _renderModeForViewSetting(self, render_mode):
        view_render_mode = _RENDER_MODE_TO_VIEW.get(render_mode)
        if view_render_mode is None:
            return None
        try:
            if get_backend() == "vulkan" and view_render_mode == VIEW_RENDER_RT:
                # Live Vulkan RT viewport parity is still WIP, so an interactive
                # RT pick from the View menu is mapped to raster. Warn once so
                # the downgrade is not silent; the NANOUSD_VIEW_RENDER_MODE / CLI
                # override path bypasses this resolver and still gets true RT.
                if not getattr(self, "_warned_vulkan_rt_downgrade", False):
                    self._warned_vulkan_rt_downgrade = True
                    sys.stderr.write(
                        "nanousdview: Vulkan RT view mode is mapped to raster in "
                        "the viewport (live RT parity WIP); set "
                        "NANOUSD_VIEW_RENDER_MODE=rt for an RT capture.\n"
                    )
                view_render_mode = VIEW_RENDER_RASTER
        except Exception:
            pass
        supported_modes = _supported_view_render_modes()
        if view_render_mode in supported_modes:
            return view_render_mode
        return supported_modes[0] if supported_modes else None

    def _visibleSettingChanged(self):
        render_mode = self._viewSettingsRenderMode()
        if render_mode != self._last_view_settings_render_mode:
            self._last_view_settings_render_mode = render_mode
            view_render_mode = self._renderModeForViewSetting(render_mode)
            if view_render_mode is not None and view_render_mode != self._render_mode:
                self._render_mode = view_render_mode
                self._frame_dirty = True
        self.update()

    # ---- Stage / scene loading ----------------------------------
    @staticmethod
    def _parse_explicit_camera(value):
        """Parse NANOUSD_VIEW_CAMERA=eye3,target3[,up3[,fov]]."""
        if not value:
            return None
        try:
            vals = [float(x.strip()) for x in value.split(",") if x.strip()]
        except Exception:
            return None
        if len(vals) not in (6, 9, 10):
            return None
        eye = np.array(vals[0:3], dtype=np.float64)
        target = np.array(vals[3:6], dtype=np.float64)
        up = np.array(vals[6:9], dtype=np.float64) if len(vals) >= 9 else None
        fov = float(vals[9]) if len(vals) == 10 else None
        return eye, target, up, fov

    def _stage_up_vector(self):
        try:
            stage = getattr(self._dataModel, "stage", None)
            up_axis = ""
            if stage:
                try:
                    up_axis = str(UsdGeom.GetStageUpAxis(stage) or "").upper()
                except Exception:
                    up_axis = str(stage.GetMetadata("upAxis") or "").upper()
            if up_axis == "Z":
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        except Exception:
            pass
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def _stage_is_z_up(self):
        return bool(np.dot(self._stage_up_vector(),
                           np.array([0.0, 0.0, 1.0])) > 0.5)

    @staticmethod
    def _normalize_vec(v, fallback):
        v = np.asarray(v, dtype=np.float64)
        n = float(np.linalg.norm(v))
        if n <= 1e-12:
            return np.asarray(fallback, dtype=np.float64)
        return v / n

    def _nav_basis(self):
        """Return local-navigation axes as world-space columns.

        Local axes match the historical Y-up controls: +X right, +Y up,
        -Z forward. The basis remaps only the authored up direction, so
        Y-up stages keep exact old behavior and Z-up stages navigate with
        world +Z vertical.
        """
        up = self._normalize_vec(self._stage_up_vector(), (0.0, 1.0, 0.0))
        preferred_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(preferred_right, up))) > 0.95:
            preferred_right = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = preferred_right - up * float(np.dot(preferred_right, up))
        right = self._normalize_vec(right, (1.0, 0.0, 0.0))
        forward = self._normalize_vec(np.cross(up, right), (0.0, 0.0, -1.0))
        back = -forward
        return np.column_stack((right, up, back))

    def _nav_to_world(self, v):
        return self._nav_basis() @ np.asarray(v, dtype=np.float64)

    def _world_to_nav(self, v):
        return self._nav_basis().T @ np.asarray(v, dtype=np.float64)

    @staticmethod
    def _local_forward_from_yaw_pitch(yaw, pitch):
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        return np.array([sy * cp, sp, -cy * cp], dtype=np.float64)

    @staticmethod
    def _yaw_pitch_from_local_forward(forward):
        forward = StageView._normalize_vec(forward, (0.0, 0.0, -1.0))
        yaw = math.atan2(forward[0], -forward[2])
        pitch = math.atan2(forward[1], math.hypot(forward[0], forward[2]))
        return yaw, pitch

    def _set_yaw_pitch_from_world_forward(self, forward):
        local_forward = self._world_to_nav(forward)
        self._fly_yaw, self._fly_pitch = self._yaw_pitch_from_local_forward(
            local_forward)

    def _fly_basis_from_yaw_pitch(self):
        local_forward = self._local_forward_from_yaw_pitch(
            self._fly_yaw, self._fly_pitch)
        local_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        local_right = self._normalize_vec(
            np.cross(local_forward, local_up), (1.0, 0.0, 0.0))
        local_camera_up = self._normalize_vec(
            np.cross(local_right, local_forward), (0.0, 1.0, 0.0))
        return (
            self._nav_to_world(local_right),
            self._nav_to_world(local_camera_up),
            self._nav_to_world(local_forward),
        )

    def _view_basis(self, eye, target, up=None):
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = self._normalize_vec(
            self._stage_up_vector() if up is None else up,
            self._stage_up_vector())
        f_vec = self._normalize_vec(target - eye,
                                    self._nav_to_world((0.0, 0.0, -1.0)))
        s = np.cross(f_vec, up)
        if float(np.linalg.norm(s)) <= 1e-12:
            s = self._nav_basis()[:, 0]
        s = self._normalize_vec(s, self._nav_basis()[:, 0])
        u = self._normalize_vec(np.cross(s, f_vec), up)
        return s, u, f_vec

    def _stage_has_authored_lights(self, stage) -> bool:
        try:
            root = stage.GetRootLayer()
            key = getattr(root, "identifier", None) or getattr(root, "realPath", None)
        except Exception:
            key = None
        if key is not None and key == self._authored_light_cache_key:
            return bool(self._authored_light_cache_value)
        light_types = {
            "DomeLight", "DistantLight", "RectLight", "SphereLight",
            "CylinderLight", "DiskLight", "GeometryLight", "PortalLight",
        }
        found = False
        try:
            for prim in stage.Traverse():
                type_name = str(prim.GetTypeName() or "")
                if type_name in light_types or type_name.endswith("Light"):
                    found = True
                    break
        except Exception:
            found = False
        self._authored_light_cache_key = key
        self._authored_light_cache_value = found
        return found

    def _syncDefaultLightingIntoRenderer(self, stage) -> None:
        if self._renderer is None or not hasattr(self._renderer, "set_default_lighting"):
            return
        view_settings = getattr(self._dataModel, "viewSettings", None)
        has_authored_lights = self._stage_has_authored_lights(stage)
        use_fallback = not has_authored_lights
        camera_light = use_fallback and bool(
            getattr(view_settings, "ambientLightOnly", True))
        dome_default = bool(getattr(view_settings, "domeLightEnabled", False))
        env_dome = (
            os.environ.get("NANOUSD_VIEW_FALLBACK_DOME")
            or os.environ.get("NUVIEW_OVRTX_FALLBACK_DOME")
        )
        if env_dome is None:
            dome_default = True
        else:
            dome_default = env_dome.lower() not in (
                "0", "false", "off", "no")
        dome_light = use_fallback and dome_default
        self._renderer.set_default_lighting(
            camera_light=camera_light,
            dome_light=dome_light,
            stage_up="Z" if self._stage_is_z_up() else "Y",
        )

    def _loadStageIntoRenderer(self):
        """Hand the current stage to the OVRTX runtime stage."""
        if not _RENDERER_AVAILABLE or self._renderer is None:
            return
        try:
            stage = getattr(self._dataModel, "stage", None)
            if stage is None:
                return
            self._syncDefaultLightingIntoRenderer(stage)
            root = stage.GetRootLayer()
            usd_path = None
            if root is not None:
                usd_path = getattr(root, "identifier", None) \
                    or getattr(root, "realPath", None)
            if not usd_path or usd_path == self._stage_loaded_path:
                return

            if hasattr(self._renderer, "update_from_usd_time"):
                self._renderer.update_from_usd_time(self._current_time_value())
            self._renderer.load_stage(usd_path)
            self._stage_loaded_path = usd_path
            self._frame_dirty = True
            # Only auto-frame on the very first load. Subsequent reloads
            # (frame changes, attribute edits) preserve the camera so the
            # animation/edit is actually visible.
            if not getattr(self, "_auto_framed_once", False):
                if os.environ.get("NUSD_VIEW_SKIP_INITIAL_FRAME", "").lower() in (
                    "1", "true", "yes", "on"
                ):
                    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    center_env = os.environ.get("NUSD_VIEW_SCENE_CENTER", "")
                    try:
                        vals = [float(x.strip()) for x in center_env.split(",") if x.strip()]
                        if len(vals) == 3:
                            center = np.array(vals, dtype=np.float64)
                    except Exception:
                        pass
                    try:
                        diag = max(float(os.environ.get("NUSD_VIEW_SCENE_DIAG", "10000")), 0.05)
                    except ValueError:
                        diag = 10000.0
                    self._scene_center = center
                    self._scene_diag = diag
                    self._frame_distance = diag * 0.9
                else:
                    self._auto_frame()
                self._auto_framed_once = True
        except Exception as e:  # noqa: BLE001
            self._render_error = f"load_usd failed: {e}"

    def _maybe_load_default_envmap(self):
        """No-op retained for appController compatibility.

        Environment and lighting are now authored through USD/OVRTX layers,
        not backend-specific imperative renderer hooks.
        """

    def _auto_frame(self):
        """Position the camera to look at the scene from a fixed iso angle.
        Bounds come from the USD stage so the viewport stays on the OVRTX API
        instead of asking a backend-specific renderer for scene internals."""
        mn = mx = None
        try:
            if mn is None or mx is None:
                from pxr import UsdGeom
                stage = self._dataModel.stage
                cache = UsdGeom.BBoxCache(0.0, ["default"])
                box = cache.ComputeWorldBound(stage.GetPseudoRoot())
                r = box.GetRange() if hasattr(box, "GetRange") else None
                if r is None or r.IsEmpty():
                    return
                mn, mx = r.GetMin(), r.GetMax()
            cx, cy, cz = (mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5
            ex, ey, ez = mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]
            diag = max((ex * ex + ey * ey + ez * ez) ** 0.5, 0.05)
            self._scene_center = np.array([cx, cy, cz], dtype=np.float64)
            self._scene_diag = diag
            # FOV-fit: project the bbox extent onto the view-aligned plane
            # and solve for distance from the FOV. Empirically tighter
            # framing than Kit's bounding-sphere formula (radius=diag*0.45,
            # dist=radius/tan(fov/2)) — Kit's puts the camera ~25% too far
            # back for thin/wide scenes (chess: 0.7×0.17×0.7 — bounding
            # sphere is 5× larger than the silhouette). View direction is
            # the iso (0.7, 0.5, 0.7) angle below.
            fov_v = math.radians(60.0)
            aspect = max(self.width() / max(self.height(), 1), 1e-6)
            half_h = max(ex, ey, ez) * 0.5
            half_w = half_h * aspect / max(aspect, 1.0)  # squarer for tall scenes
            tan_v_half = math.tan(fov_v * 0.5)
            tan_h_half = math.tan(math.atan(tan_v_half * aspect))
            d_v = half_h / max(tan_v_half, 1e-6)
            d_h = half_w / max(tan_h_half, 1e-6)
            fit_dist = max(d_v, d_h) * 0.85  # tight crop — get up close to the subject
            # Hold onto fit_dist so orbit mode can pick it up too.
            self._frame_distance = fit_dist
            # Iso direction in navigation-local axes: +X right, +Y stage-up,
            # +Z backwards from the view direction. This preserves the old
            # Y-up framing and makes Z-up stages stand upright.
            iso = self._nav_to_world((0.7, 0.3, 0.7))
            iso = self._normalize_vec(iso, (0.7, 0.3, 0.7))
            self._fly_pos = self._scene_center + iso * fit_dist
            self._set_yaw_pitch_from_world_forward(
                self._scene_center - self._fly_pos)
            self._fly_speed = max(diag / 8.0, 0.1)  # ~8 sec to traverse scene
            # Seed orbit quaternion from the framed look angle so trackball
            # starts where _auto_frame placed it.
            self._sync_orbit_from_yaw_pitch()
        except Exception:
            self._scene_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            self._scene_diag = 10.0

    def _drawSelectionHighlight(self, painter, eye, target, up=None):
        """Draw a 2D wireframe AABB outlining each selected prim's world
        bounding box, projected through the current view/projection."""
        if os.environ.get("NUVIEW_NO_SEL_HIGHLIGHT") == "1":
            return
        try:
            sel = self._dataModel.selection.getLCDPrims()
        except Exception:
            return
        if not sel:
            return
        if os.environ.get("NUVIEW_DEBUG_SEL") == "1":
            try:
                paths = [p.GetPath() for p in sel]
                print(f"[sel-highlight] sel={paths}", file=sys.stderr)
            except Exception:
                pass
        try:
            from pxr import UsdGeom
        except Exception:
            return

        w, h = self.width(), self.height()
        # Build view + projection matrices from eye/target/stage-up.
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        s, u, f_vec = self._view_basis(eye, target, up)
        # View matrix (col-major) for column-vector convention:
        view = np.eye(4, dtype=np.float64)
        view[0, 0:3] = s
        view[1, 0:3] = u
        view[2, 0:3] = -f_vec
        view[0, 3] = -float(np.dot(s, eye))
        view[1, 3] = -float(np.dot(u, eye))
        view[2, 3] = float(np.dot(f_vec, eye))
        fov_rad = math.radians(60.0)
        aspect = max(w / max(h, 1), 1e-6)
        f_proj = 1.0 / math.tan(fov_rad * 0.5)
        near = max(self._scene_diag * 0.001, 0.001)
        far = max(self._scene_diag * 100.0, 1000.0)
        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = f_proj / aspect
        proj[1, 1] = f_proj
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0
        vp = proj @ view

        pen = QtGui.QPen(QtGui.QColor(255, 220, 60), 2)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        # Reuse one BBoxCache across paints — the previous code allocated
        # a fresh cache + walked the entire selected-prim subtree on every
        # paint event, which was the dominant cost during orbit on
        # Kitchen_set (~250 ms / paint on a 1788-mesh subtree).
        if not hasattr(self, "_selectionBBoxCache"):
            self._selectionBBoxCache = UsdGeom.BBoxCache(0.0, ["default"])
        cache = self._selectionBBoxCache
        for prim in sel:
            try:
                box = cache.ComputeWorldBound(prim)
                r = box.GetRange() if hasattr(box, "GetRange") else None
                if r is None or r.IsEmpty(): continue
                mn, mx = r.GetMin(), r.GetMax()
                corners = np.array([
                    [mn[0], mn[1], mn[2], 1], [mx[0], mn[1], mn[2], 1],
                    [mn[0], mx[1], mn[2], 1], [mx[0], mx[1], mn[2], 1],
                    [mn[0], mn[1], mx[2], 1], [mx[0], mn[1], mx[2], 1],
                    [mn[0], mx[1], mx[2], 1], [mx[0], mx[1], mx[2], 1],
                ], dtype=np.float64)
                clip = corners @ vp.T
                # Skip prims fully behind near-plane.
                if np.all(clip[:, 3] <= 0): continue
                ww = np.where(np.abs(clip[:, 3]) < 1e-9, 1.0, clip[:, 3])
                ndc = clip[:, :3] / ww[:, None]
                screen = np.empty((8, 2), dtype=np.float64)
                screen[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * w
                screen[:, 1] = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * h
                # Draw 12 AABB edges.
                edges = [(0,1),(2,3),(4,5),(6,7),
                         (0,2),(1,3),(4,6),(5,7),
                         (0,4),(1,5),(2,6),(3,7)]
                for a, b in edges:
                    if clip[a, 3] <= 0 and clip[b, 3] <= 0: continue
                    painter.drawLine(int(screen[a, 0]), int(screen[a, 1]),
                                     int(screen[b, 0]), int(screen[b, 1]))
            except Exception:
                continue

    def _drawAxisGizmo(self, painter, eye, target, up=None):
        """Bottom-left XYZ axis gizmo. Computes the same view rotation
        the renderer uses, strokes 3 colored axes (X red, Y green, Z blue)
        and labels them."""
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        s, u, f_vec = self._view_basis(eye, target, up)
        # 3x3 view rotation only; we want the screen-space direction of
        # each world axis, no perspective.
        Rv = np.array([s, u, -f_vec], dtype=np.float64)  # 3x3

        cx, cy = 60, self.height() - 60  # gizmo center pixel
        L = 36                           # axis length in pixels
        axes = [
            (np.array([1.0, 0.0, 0.0]), QtGui.QColor(220, 70, 70),  "X"),
            (np.array([0.0, 1.0, 0.0]), QtGui.QColor(80, 200, 80),  "Y"),
            (np.array([0.0, 0.0, 1.0]), QtGui.QColor(80, 130, 230), "Z"),
        ]
        # Sort by depth so axes pointing away render first.
        projected = []
        for v, color, name in axes:
            d = Rv @ v
            projected.append((float(d[2]), d, color, name))
        projected.sort(key=lambda t: t[0])  # most-into-screen first

        font = painter.font(); font.setPointSize(9); font.setBold(True)
        painter.setFont(font)
        for _z, d, color, name in projected:
            # X+ => right, Y+ => up (Qt y inverted), Z+ => out-of-screen.
            ex = cx + d[0] * L
            ey = cy - d[1] * L
            painter.setPen(QtGui.QPen(color, 2))
            painter.drawLine(int(cx), int(cy), int(ex), int(ey))
            # Cap label slightly past the line tip.
            lx = cx + d[0] * (L + 8)
            ly = cy - d[1] * (L + 8)
            painter.drawText(int(lx) - 4, int(ly) + 4, name)

    def _drawTransformGizmo(self, painter, eye, target):
        """Draw the T/R/S transform manipulator anchored at the lone selected
        prim's world origin. Hidden when nothing or many prims are selected,
        or when toggled off via G. During an active drag, the gizmo's own
        state owns the transform; otherwise we re-seed from USD each frame."""
        if not self._show_gizmo:
            return
        prim = self._selected_single_prim()
        if prim is None:
            return
        if not self._gizmo_dragging:
            self._seed_gizmo_from_prim(prim)
        # Cursor pixel for hover-test (no drag).
        mx = self._last_mouse_pos.x() if self._last_mouse_pos else 0
        my = self._last_mouse_pos.y() if self._last_mouse_pos else 0
        vp = self._build_vp(eye, target)
        st = self._populate_gizmo_app_state(mx, my, mouse_left=False, eye=eye, vp_col=vp)
        # passive=True so we don't fabricate a release-edge from the
        # transient mouse_left=False state in a non-drag paint frame.
        self._gizmo.update(st, passive=True)
        self._gizmo.tick(self._gizmo_xform)
        self._gizmo.draw(painter, vp, max(self.width(), 1), max(self.height(), 1))

    def _frame_selected_prim(self):
        """Pixar 'F' shortcut: frame camera on the selected prim. Falls
        back to the whole scene if nothing's selected or bounds are empty."""
        try:
            from pxr import UsdGeom
        except Exception:
            self._auto_frame(); return
        try:
            sel = self._dataModel.selection.getLCDPrims() \
                if hasattr(self._dataModel, "selection") else []
        except Exception:
            sel = []
        mn = mx = None
        for p in sel:
            try:
                cache = UsdGeom.BBoxCache(0.0, ["default"])
                box = cache.ComputeWorldBound(p)
                r = box.GetRange() if hasattr(box, "GetRange") else None
                if r is None or r.IsEmpty(): continue
                mmn, mmx = r.GetMin(), r.GetMax()
                pmn = np.array([mmn[0], mmn[1], mmn[2]], dtype=np.float64)
                pmx = np.array([mmx[0], mmx[1], mmx[2]], dtype=np.float64)
                mn = pmn if mn is None else np.minimum(mn, pmn)
                mx = pmx if mx is None else np.maximum(mx, pmx)
            except Exception:
                continue
        if mn is None or mx is None:
            self._auto_frame(); return
        cx, cy, cz = (mn + mx) * 0.5
        ext = mx - mn
        diag = max(float(np.linalg.norm(ext)), 0.05)
        self._scene_center = np.array([cx, cy, cz], dtype=np.float64)
        self._scene_diag = diag
        # Keep the current viewing angle — Pixar's F doesn't rotate, just
        # moves+zooms onto the selection. Use the same FOV-fit math as
        # _auto_frame so flat/thin selections frame tightly instead of
        # being lost in the bounding-sphere padding.
        fov_v = math.radians(60.0)
        aspect = max(self.width() / max(self.height(), 1), 1e-6)
        half_h = float(max(ext[0], ext[1], ext[2])) * 0.5
        tan_v_half = math.tan(fov_v * 0.5)
        fit_dist = (half_h / max(tan_v_half, 1e-6)) * 1.15
        self._frame_distance = fit_dist
        try:
            eye, target, _up = self._camera_eye_target_up()
            forward = self._normalize_vec(
                np.asarray(target, dtype=np.float64) -
                np.asarray(eye, dtype=np.float64),
                self._nav_to_world((0.0, 0.0, -1.0)))
        except Exception:
            iso = self._nav_to_world((0.7, 0.5, 0.7))
            forward = -self._normalize_vec(iso, (0.7, 0.5, 0.7))
        self._fly_pos = self._scene_center - forward * fit_dist
        self._set_yaw_pitch_from_world_forward(forward)
        self._sync_orbit_from_yaw_pitch()
        self._fly_speed = max(diag / 8.0, 0.05)

    def _ensure_renderer(self):
        if not _RENDERER_AVAILABLE or self._renderer is not None:
            return
        w = max(self.width(), 320)
        h = max(self.height(), 240)
        try:
            self._renderer = OvrtxViewportRenderer(width=w, height=h)
        except Exception as e:  # noqa: BLE001
            self._render_error = f"OVRTX renderer init: {e}"
            self._renderer = None
            return
        self._loadStageIntoRenderer()
        self._registerStageNoticeListener()

    def _stageReplaced(self):
        self._stage_loaded_path = None
        self._authored_light_cache_key = None
        self._authored_light_cache_value = False
        self._auto_framed_once = False  # Re-frame on the new stage.
        try:
            self._dataModel.viewSettings.freeCamera = self._createNewFreeCamera(
                self._dataModel.viewSettings, self._stage_is_z_up())
        except Exception:
            pass
        # Drop the cached BBox cache — paths from the old stage are stale.
        if hasattr(self, "_selectionBBoxCache"):
            try: self._selectionBBoxCache.Clear()
            except Exception: pass
            del self._selectionBBoxCache
        self._loadStageIntoRenderer()
        self._registerStageNoticeListener()
        self.update()

    def _registerStageNoticeListener(self):
        """Listen for Tf.Notice.ObjectsChanged on the current stage so that
        variant edits, attribute changes, and other resync events kick the
        renderer into reloading. Phase 5: keeps the rendered scene in lock
        step with the Python stage view."""
        try:
            if getattr(self, "_notice_handle", None) is not None:
                try: self._notice_handle.Revoke()
                except Exception: pass
                self._notice_handle = None
            stage = getattr(self._dataModel, "stage", None)
            if stage is None:
                return
            self._notice_handle = Tf.Notice.Register(
                Usd.Notice.ObjectsChanged, self._onObjectsChanged, stage)
        except Exception:
            self._notice_handle = None

    def _onObjectsChanged(self, notice, sender):
        # Force a full reload — cheapest correct path. The shared stage handle
        # means we only re-walk the (already-composed) stage, not re-parse.
        try:
            self._stage_loaded_path = None
            self._authored_light_cache_key = None
            self._authored_light_cache_value = False
            self._loadStageIntoRenderer()
            self._frame_dirty = True
            self.update()
        except Exception as e:  # noqa: BLE001
            self._render_error = f"objectsChanged reload: {e}"

    def _primSelectionChanged(self, *a, **kw):
        self.update()

    def _onFreeCameraSettingChanged(self, *a, **kw):
        self.update()

    def _onAutoComputeClippingChanged(self, *a, **kw):
        self.update()

    # ---- Camera helpers ----------------------------------------
    def _createNewFreeCamera(self, viewSettings, isZUp=True):
        """appController calls this to (re)create the free camera."""
        try:
            return FreeCamera(
                isZUp=isZUp,
                fov=getattr(viewSettings, "freeCameraFOV", 60.0) or 60.0,
                aspectRatio=self._lastAspectRatio,
                overrideNear=None, overrideFar=None,
            )
        except Exception:
            return None

    def _activeScenePrimCamera(self):
        """Return the active UsdGeom.Camera prim if one is set, else None."""
        try:
            cam = getattr(self._dataModel.viewSettings, "cameraPrim", None)
            if cam is None: return None
            if hasattr(cam, "IsValid") and not cam.IsValid(): return None
            if hasattr(cam, "IsActive") and not cam.IsActive(): return None
            return cam
        except Exception:
            return None

    def _scene_camera_fov_degrees(self, cam_prim):
        """Return the vertical FOV for an authored camera, if available."""
        if cam_prim is None:
            return None
        try:
            time_value = self._current_time_value()
            fl_attr = cam_prim.GetAttribute("focalLength")
            ha_attr = cam_prim.GetAttribute("horizontalAperture")
            va_attr = cam_prim.GetAttribute("verticalAperture")
            fl = float(fl_attr.Get(time_value) or 0.0) \
                if fl_attr and fl_attr.IsValid() else 0.0
            ha = float(ha_attr.Get(time_value) or 0.0) \
                if ha_attr and ha_attr.IsValid() else 0.0
            va = float(va_attr.Get(time_value) or 0.0) \
                if va_attr and va_attr.IsValid() else 0.0
            if fl <= 1e-6:
                return None
            if va > 1e-6:
                return math.degrees(2.0 * math.atan(va / (2.0 * fl)))
            if ha > 1e-6:
                aspect = max(self.width() / max(self.height(), 1), 1e-6)
                return math.degrees(
                    2.0 * math.atan((ha / aspect) / (2.0 * fl)))
        except Exception:
            return None
        return None

    def _detach_scene_camera_for_navigation(self):
        """Switch scene-camera viewing to the free camera before user nav."""
        if self._explicit_camera is not None:
            return
        cam_prim = self._activeScenePrimCamera()
        if cam_prim is None:
            return

        eye, target, _up = self._camera_eye_target_up()
        eye = np.asarray(eye, dtype=np.float64)
        forward = np.asarray(target, dtype=np.float64) - eye
        authored_target_distance = float(np.linalg.norm(forward))
        if authored_target_distance <= 1e-9:
            forward = self._nav_to_world((0.0, 0.0, -1.0))
        else:
            forward = forward / authored_target_distance

        scene_diag = float(getattr(self, "_scene_diag", 10.0) or 10.0)
        fallback_distance = max(float(getattr(
            self, "_frame_distance", scene_diag * 0.5) or scene_diag * 0.5),
            0.001)
        previous_center = np.asarray(getattr(
            self, "_scene_center", eye + forward * fallback_distance),
            dtype=np.float64)
        distance = float(np.dot(previous_center - eye, forward))
        if distance <= max(scene_diag * 0.01, 0.1):
            distance = fallback_distance
        orbit_target = eye + forward * max(distance, 0.001)

        self._fly_pos = eye.copy()
        self._scene_center = orbit_target
        self._frame_distance = max(distance, 0.001)
        self._set_yaw_pitch_from_world_forward(forward)
        self._sync_orbit_from_yaw_pitch()

        fov = self._scene_camera_fov_degrees(cam_prim)
        if fov is not None:
            try:
                self._dataModel.viewSettings.freeCameraFOV = fov
            except Exception:
                pass
        try:
            self._dataModel.viewSettings.cameraPrim = None
        except Exception:
            pass
        self.signalFrustumChanged.emit()

    def _current_time_value(self) -> float:
        try:
            cf = getattr(self._dataModel, "currentFrame", None)
            if cf is None:
                return 0.0
            if hasattr(cf, "IsDefault") and cf.IsDefault():
                return 0.0
            value = cf.GetValue() if hasattr(cf, "GetValue") else cf
            value = float(value)
            return value if math.isfinite(value) else 0.0
        except Exception:
            return 0.0

    # ---- Quaternion + arcball helpers --------------------------
    @staticmethod
    def _qmul(a, b):
        """Hamilton product of two (w,x,y,z) quaternions."""
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dtype=np.float64)

    @staticmethod
    def _qrotate(q, v):
        """Rotate 3-vector v by quaternion q = (w,x,y,z)."""
        w, x, y, z = q
        # v' = q * (0,v) * q_conj — expanded form.
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        # t = 2 * cross(q.xyz, v)
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        # v' = v + w*t + cross(q.xyz, t)
        rx = vx + w * tx + (y * tz - z * ty)
        ry = vy + w * ty + (z * tx - x * tz)
        rz = vz + w * tz + (x * ty - y * tx)
        return np.array([rx, ry, rz], dtype=np.float64)

    @staticmethod
    def _quat_from_yaw_pitch(yaw, pitch):
        """Build orbit quaternion matching the _fly_* yaw/pitch convention:
        forward = (sin(yaw)*cos(pitch), sin(pitch), -cos(yaw)*cos(pitch)).
        Yaw rotates around -Y (so positive yaw turns the look toward +X),
        pitch rotates around +X.
        """
        cy = math.cos(yaw * 0.5);  sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
        qy = np.array([cy, 0.0, -sy, 0.0], dtype=np.float64)  # yaw around -Y
        qx = np.array([cp, sp, 0.0, 0.0], dtype=np.float64)   # pitch around +X
        return StageView._qmul(qy, qx)

    def _arcball_project(self, x, y):
        """Map pixel (x, y) onto the virtual unit sphere centered on the
        viewport. Outside r=1 falls onto a hyperbolic sheet (Holroyd's
        improvement on Shoemake) so the rotation keeps responding past
        the edge instead of clamping."""
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        radius = 0.5 * min(w, h)
        nx = (x - 0.5 * w) / radius
        ny = (0.5 * h - y) / radius   # flip Y so screen-up == sphere +Y
        r2 = nx * nx + ny * ny
        if r2 <= 1.0:
            nz = math.sqrt(1.0 - r2)
        else:
            inv = 1.0 / math.sqrt(r2)
            nx *= inv
            ny *= inv
            nz = 0.0
        v = np.array([nx, ny, nz], dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _arcball_step(self, p_old, p_new):
        """Rotation taking sphere point p_old to p_new (Shoemake)."""
        d = float(np.dot(p_old, p_new))
        d = max(-1.0, min(1.0, d))
        if d > 0.99999:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        axis = np.cross(p_old, p_new)
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        axis /= n
        angle = math.acos(d)
        s = math.sin(angle * 0.5)
        c = math.cos(angle * 0.5)
        # 2x rotation feels more responsive (matches most DCC trackballs).
        s2 = math.sin(angle)
        c2 = math.cos(angle)
        return np.array([c2, axis[0] * s2, axis[1] * s2, axis[2] * s2], dtype=np.float64)

    def _orbit_basis(self):
        """Return (right, up, forward) world-space unit vectors for the
        current orbit camera orientation. forward points from eye → target."""
        local_forward = self._qrotate(
            self._orbit_quat, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        local_up = self._qrotate(
            self._orbit_quat, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        local_right = self._qrotate(
            self._orbit_quat, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        return (
            self._nav_to_world(local_right),
            self._nav_to_world(local_up),
            self._nav_to_world(local_forward),
        )

    def _sync_orbit_from_yaw_pitch(self):
        """Refresh _orbit_quat from current _fly_yaw/_fly_pitch — call this
        after _auto_frame / _frame_selected_prim so the trackball starts at
        the framed orientation."""
        self._orbit_quat = self._quat_from_yaw_pitch(self._fly_yaw, self._fly_pitch)

    def _camera_eye_target_up(self):
        """Compute eye/target/up for the current camera mode."""
        if self._explicit_camera is not None:
            eye, target, up, _fov = self._explicit_camera
            if up is None:
                up = self._stage_up_vector()
            return eye.copy(), target.copy(), up.copy()

        # USD scene camera takes precedence — Pixar's "Set As Active
        # Camera" right-click action sets viewSettings.cameraPrim, and
        # the viewport switches to that camera's authored transform.
        cam_prim = self._activeScenePrimCamera()
        if cam_prim is not None:
            try:
                from pxr import UsdGeom
                xf = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(
                    self._current_time_value())
                M = np.asarray(xf, dtype=np.float64).reshape(4, 4)
                # Pixar UsdGeom.Camera: looks along -Z in local space,
                # +Y up. Row-major: row 3 is translation, rows 0-2 are
                # axes (X right, Y up, -Z forward).
                eye = M[3, :3].copy()
                fwd = -M[2, :3]
                up  =  M[1, :3]
                target = eye + fwd
                # Normalize-ish to avoid degenerate near/far.
                if np.linalg.norm(fwd) < 1e-9:
                    target = eye + self._nav_to_world((0.0, 0.0, -1.0))
                return eye, target, up
            except Exception:
                pass
        if self._fly_mode:
            _right, up, forward = self._fly_basis_from_yaw_pitch()
            target = self._fly_pos + forward
            return self._fly_pos.copy(), target, up
        # Orbit: quaternion-driven trackball. The orbit_quat rotates the
        # default frame (forward=-Z, up=+Y); eye sits `distance` opposite
        # the forward vector from scene_center.
        if not hasattr(self, "_scene_center"):
            self._scene_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            self._scene_diag = 10.0
        _, up, forward = self._orbit_basis()
        # Prefer the FOV-fit distance computed by _auto_frame / _frameSelection
        # — falls back to scene_diag*0.9 if neither has run (no scene yet).
        distance = float(getattr(self, "_frame_distance", self._scene_diag * 0.9))
        eye = self._scene_center - forward * distance
        target = self._scene_center.copy()
        return eye, target, up

    # ---- Gizmo helpers -----------------------------------------
    def _build_vp(self, eye, target):
        """Construct the column-vector view-projection matrix matching
        `_drawSelectionHighlight`'s convention. Returns 4x4 numpy array."""
        w, h = max(self.width(), 1), max(self.height(), 1)
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        s, u, f_vec = self._view_basis(eye, target)
        view = np.eye(4, dtype=np.float64)
        view[0, 0:3] = s
        view[1, 0:3] = u
        view[2, 0:3] = -f_vec
        view[0, 3] = -float(np.dot(s, eye))
        view[1, 3] = -float(np.dot(u, eye))
        view[2, 3] = float(np.dot(f_vec, eye))
        fov_rad = math.radians(60.0)
        aspect = max(w / max(h, 1), 1e-6)
        f_proj = 1.0 / math.tan(fov_rad * 0.5)
        near = max(self._scene_diag * 0.001, 0.001)
        far = max(self._scene_diag * 100.0, 1000.0)
        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = f_proj / aspect
        proj[1, 1] = f_proj
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj @ view

    def _selected_single_prim(self):
        """Return the lone selected prim, or None for 0/multi selection."""
        try:
            sel = self._dataModel.selection.getLCDPrims() \
                if hasattr(self._dataModel, "selection") else []
        except Exception:
            return None
        return sel[0] if len(sel) == 1 else None

    def _seed_gizmo_from_prim(self, prim):
        """Snapshot the prim's world transform into the gizmo state. Called
        when not actively dragging — during drag the gizmo owns the transform."""
        try:
            M_row = np.asarray(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0),
                dtype=np.float64).reshape(4, 4)
        except Exception:
            M_row = np.eye(4, dtype=np.float64)
        # USD returns row-vector matrices (translation at row 3). gizmo.py
        # uses column-vector convention (translation at column 3).
        M_col = M_row.T
        pos, q, scl = decompose_trs(M_col)
        self._gizmo_xform.position[:] = pos
        self._gizmo_xform.orientation[:] = q
        self._gizmo_xform.scale[:] = scl

    def _populate_gizmo_app_state(self, mx, my, mouse_left, eye, vp_col):
        """Build the per-frame AppState the gizmo consumes (mouse, ray, cam)."""
        w, h = max(self.width(), 1), max(self.height(), 1)
        ro, rd = make_pick_ray(float(mx), float(my), vp_col, w, h)
        s = self._gizmo_app_state
        s.viewport_size = (w, h)
        s.ray_origin[:] = ro
        s.ray_direction[:] = rd
        s.cam_eye[:] = eye
        s.cam_yfov = math.radians(60.0)
        s.mouse_left = bool(mouse_left)
        s.screenspace_scale = 80.0
        return s

    def _commit_gizmo_to_prim(self, prim):
        """Author the gizmo's world transform back onto `prim` as a canonical
        Translate/Orient/Scale stack. Accounts for the parent's world transform
        so a non-root prim still moves predictably."""
        try:
            from pxr import UsdGeom as _UG
            parent = prim.GetParent()
            parent_world_col = np.eye(4, dtype=np.float64)
            if parent and parent.IsValid() and str(parent.GetPath()) != "/":
                try:
                    pw_row = np.asarray(
                        _UG.Xformable(parent).ComputeLocalToWorldTransform(0.0),
                        dtype=np.float64).reshape(4, 4)
                    parent_world_col = pw_row.T
                except Exception:
                    pass
            world_col = self._gizmo_xform.matrix()
            local_col = np.linalg.inv(parent_world_col) @ world_col
            pos, q, scl = decompose_trs(local_col)
            write_xform_to_prim(prim, pos, q, scl)
            # Force the renderer to re-read the stage on next frame.
            self._stage_loaded_path = None
            self._authored_light_cache_key = None
            self._authored_light_cache_value = False
            self._loadStageIntoRenderer()
            self._frame_dirty = True
        except Exception as e:  # noqa: BLE001
            print(f"[stageView] gizmo commit failed: {e}",
                  file=sys.stderr, flush=True)

    # ---- Painting ----------------------------------------------
    def paintEvent(self, ev):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(40, 40, 50))

        self._ensure_renderer()

        if not _RENDERER_AVAILABLE:
            painter.setPen(QtGui.QColor(255, 200, 200))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter,
                             "ovrtx renderer not available\n"
                             f"{getattr(globals(), '_RENDERER_IMPORT_ERROR', '?')}")
            return

        if self._renderer is None:
            painter.setPen(QtGui.QColor(255, 200, 200))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter,
                             self._render_error or "Renderer not ready")
            return

        # Resize renderer if our widget changed size.
        # Render at *device* pixels (logical * devicePixelRatio): on a retina
        # display the widget is logical NxM but its drawable surface is
        # 2N x 2M; rendering at the smaller logical size and blitting a
        # logical-pixels-treated-as-device QImage covers only the upper-left
        # quadrant and leaves the selection-highlight + axis-gizmo overlays
        # floating over stale background. Render at device size and tag the
        # QImage with DPR below so painter scales it back to logical for blit.
        try:
            dpr = float(self.devicePixelRatio() or 1.0)
        except Exception:
            dpr = 1.0
        w = max(int(round(self.width() * dpr)), 1)
        h = max(int(round(self.height() * dpr)), 1)
        try:
            if not hasattr(self, "_rw") or self._rw != w or self._rh != h:
                self._renderer.set_size(w, h)
                self._rw, self._rh = w, h
                self._frame_dirty = True
        except Exception:
            pass

        # Push camera state via OVRTX attribute writes on the viewport camera.
        eye, target, up = self._camera_eye_target_up()
        # If a USD-authored camera is active, mirror its focalLength /
        # horizontalAperture / clippingRange so the renderer frames the
        # scene the same way ovrtx (and any other Hydra-driven viewer)
        # would. Otherwise fall back to the orbit defaults.
        fov_degrees = float(getattr(
            self._dataModel.viewSettings, "freeCameraFOV", 60.0) or 60.0)
        near_clip = max(self._scene_diag * 0.001, 0.001)
        far_clip = max(self._scene_diag * 100.0, 1000.0)
        focal_length = None
        horizontal_aperture = None
        vertical_aperture = None
        cam_prim = self._activeScenePrimCamera()
        if self._explicit_camera is not None and self._explicit_camera[3] is not None:
            fov_degrees = float(self._explicit_camera[3])
        elif cam_prim is not None:
            try:
                time_value = self._current_time_value()
                fl_attr = cam_prim.GetAttribute("focalLength")
                ha_attr = cam_prim.GetAttribute("horizontalAperture")
                va_attr = cam_prim.GetAttribute("verticalAperture")
                fl = float(fl_attr.Get(time_value) or 0.0) \
                    if fl_attr and fl_attr.IsValid() else 0.0
                ha = float(ha_attr.Get(time_value) or 0.0) \
                    if ha_attr and ha_attr.IsValid() else 0.0
                va = float(va_attr.Get(time_value) or 0.0) \
                    if va_attr and va_attr.IsValid() else 0.0
                if fl > 1e-6:
                    if ha > 1e-6 and va > 1e-6:
                        focal_length = fl
                        horizontal_aperture = ha
                        vertical_aperture = va
                        fov_degrees = math.degrees(
                            2.0 * math.atan(va / (2.0 * fl)))
                    elif ha > 1e-6:
                        aspect = max(w / max(h, 1), 1e-6)
                        fov_degrees = math.degrees(
                            2.0 * math.atan((ha / aspect) / (2.0 * fl)))
                    elif va > 1e-6:
                        fov_degrees = math.degrees(
                            2.0 * math.atan(va / (2.0 * fl)))
                cr_attr = cam_prim.GetAttribute("clippingRange")
                if cr_attr and cr_attr.IsValid():
                    cr = cr_attr.Get(time_value)
                    if cr is not None and len(cr) >= 2:
                        n, f = float(cr[0]), float(cr[1])
                        if n > 0.0 and f > n:
                            near_clip, far_clip = n, f
            except Exception:
                pass
        try:
            self._renderer.set_camera(
                eye=tuple(float(x) for x in eye),
                target=tuple(float(x) for x in target),
                up=tuple(float(x) for x in up),
                fov_degrees=fov_degrees,
                near_clip=near_clip,
                far_clip=far_clip,
                focal_length=focal_length,
                horizontal_aperture=horizontal_aperture,
                vertical_aperture=vertical_aperture,
            )
        except Exception as e:  # noqa: BLE001
            self._render_error = f"set_camera: {e}"

        # Render + readback through OVRTX RenderProduct outputs.
        try:
            _t0 = time.monotonic()
            pixels = self._renderer.render_ldr(delta_time=0.0, render_mode=self._render_mode)
            _ms = (time.monotonic() - _t0) * 1000.0
            try:
                _fps = (1000.0 / _ms) if _ms > 0 else 0.0
                self.fpsHUDInfo["Render"] = f"{_ms:.2f} ms ({_fps:.1f} FPS)"
            except Exception: pass
            # pixels is (h, w, 4) uint8 RGBA from the renderer; build QImage.
            ah, aw, ac = pixels.shape
            # Optional photographic tone-map for renderers that don't ship
            # one. Kit / ovrtx apply ACES + photographic exposure
            # (responsivity * filmIso / (100 * shutter * fNumber²),
            # default ~0.000882 multiplier) so HDR scenes don't blow out;
            # vulkan + opengl currently apply intensity linearly with no
            # exposure stage and clip highlights to white. Apply ACES on
            # the LDR uint8 here to compress the range so captures match
            # ovrtx's character. Already-clipped highlights stay at 255 —
            # the renderers need a true HDR float output for full parity.
            # Toggle with NUVIEW_TONEMAP={none,aces}; default disabled
            # (auto-on only for backends without their own tonemap).
            mode = os.environ.get("NUVIEW_TONEMAP", "auto").lower()
            auto_on = (mode == "auto"
                       and getattr(self._renderer, "backend", "") != "ovrtx")
            if mode == "aces" or auto_on:
                pixels = _apply_aces_tonemap(pixels)
                self._frame_pixels = pixels
            else:
                self._frame_pixels = pixels
            qimg = QtGui.QImage(pixels.data, aw, ah, aw * 4,
                                QtGui.QImage.Format_RGBA8888)
            # Tag the QImage with the widget's device pixel ratio so painter
            # treats its raw pixels as device-pixels backing a logical area
            # of (aw/dpr, ah/dpr). The renderer was sized to logical*dpr in
            # the resize block above, so this lands a device-perfect blit
            # that covers the full widget on retina (without DPR tagging,
            # the QImage's raw pixels are interpreted as logical and only
            # the upper-left quadrant gets covered on a 2x display).
            try:
                qimg.setDevicePixelRatio(float(self.devicePixelRatio() or 1.0))
            except Exception:
                pass
            self._frame_qimage = qimg
            # Renderer was already sized to (logical*dpr) above so the QImage
            # logical extent matches the widget size and we can blit directly.
            # The previous `.scaled(... SmoothTransformation)` was a CPU
            # bilinear resample of the full framebuffer every paint — capped
            # Kitchen_set orbit to ~4 fps even though the GPU was rendering
            # in <3 ms. Fall back to scaled+fast only on genuine size
            # mismatch (transient during resize before the renderer catches up).
            if aw == self._rw and ah == self._rh:
                painter.drawImage(0, 0, qimg)
            else:
                painter.drawImage(0, 0, qimg.scaled(self.size(),
                                                    QtCore.Qt.IgnoreAspectRatio,
                                                    QtCore.Qt.FastTransformation))
            self._frame_dirty = False
            self._isFirstImage = False
            self._render_error = None

            # Selection highlight — draw a 2D wireframe AABB for each
            # selected prim's world bounds. RT pipeline doesn't support
            # wireframe overlays, so we project the 8 corners through the
            # current view+projection and stroke an outline on top of the
            # rendered image.
            try:
                self._drawSelectionHighlight(painter, eye, target, up)
            except Exception: pass
            try:
                self._drawAxisGizmo(painter, eye, target, up)
            except Exception: pass
            try:
                self._drawTransformGizmo(painter, eye, target)
            except Exception: pass
        except Exception as e:  # noqa: BLE001
            # First-paint readback can fail before the renderer has produced
            # a frame. Stay silent — the next paint typically succeeds.
            # Persist the error only if it has changed, for debugging via
            # NUVIEW_RENDER_ERROR_TRACE=1.
            self._render_error = f"render: {e}"
            if os.environ.get("NUVIEW_RENDER_ERROR_TRACE"):
                painter.setPen(QtGui.QColor(255, 200, 200))
                painter.drawText(self.rect(), QtCore.Qt.AlignCenter,
                                 self._render_error)

        # HUD — top-left: render mode + camera state. usdview-style upper
        # stats (Prims/Verts/Faces) when populated by appController, plus
        # bottom-right fps stats (Render/Playback) when present.
        f = painter.font(); f.setPointSize(10); painter.setFont(f)
        mode_name = {VIEW_RENDER_RT: "RT", VIEW_RENDER_SHADOW: "SHADOW",
                     VIEW_RENDER_RASTER: "RASTER"}.get(
            self._render_mode, "?")
        cam_mode = "FLY (WASD)" if self._fly_mode else "ORBIT"
        try:
            backend_label = {"vulkan": "Vulkan RT", "opengl": "OpenGL",
                             "metal":  "Metal RT",
                             "ovrtx":  "OVRTX"}.get(get_backend(), "?")
        except Exception:
            backend_label = "?"
        if not self._show_hud:
            return
        hud_lines = [
            f"nanousdview · {backend_label} · {mode_name} · {cam_mode}",
            f"speed: {self._fly_speed:.2f} m/s",
        ]
        try:
            keys = list(self.HUDStatKeys) if self.HUDStatKeys else \
                   sorted(self.upperHUDInfo.keys())
            for k in keys:
                v = self.upperHUDInfo.get(k)
                if v is not None:
                    hud_lines.append(f"{k}: {v}")
        except Exception: pass

        def _draw_hud(rect, align, text):
            # Drop-shadow for legibility on bright/dark backgrounds.
            painter.setPen(QtGui.QColor(0, 0, 0, 200))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                painter.drawText(rect.translated(dx, dy), align, text)
            painter.setPen(QtGui.QColor(255, 230, 100))
            painter.drawText(rect, align, text)

        _draw_hud(QtCore.QRect(8, 6, 500, 200),
                  QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft,
                  "\n".join(hud_lines))

        try:
            fps_lines = []
            for k in self.fpsHUDKeys or list(self.fpsHUDInfo.keys()):
                v = self.fpsHUDInfo.get(k)
                if v is not None:
                    fps_lines.append(f"{k}: {v}")
            if fps_lines:
                rect = QtCore.QRect(self.width() - 308, self.height() - 60,
                                    300, 50)
                _draw_hud(rect,
                          QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight,
                          "\n".join(fps_lines))
        except Exception: pass

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._lastAspectRatio = max(self.width() / max(self.height(), 1), 1e-6)
        self._frame_dirty = True
        self.update()

    # ---- Tick (WASD camera update) -----------------------------
    def _on_tick(self):
        # WASD is always-on: in fly mode it walks the eye (FPS-style),
        # in orbit mode it pans the orbit target along camera-aligned axes
        # (so the scene stays centered as you fly through it).
        if self._wasd_keys:
            sprint = 3.0 if QtCore.Qt.Key_Shift in self._wasd_keys else 1.0
            step = self._fly_speed * 0.016 * sprint
            if self._fly_mode:
                right, up, forward = self._fly_basis_from_yaw_pitch()
            else:
                right, up, forward = self._orbit_basis()
            mv = np.zeros(3)
            if QtCore.Qt.Key_W in self._wasd_keys: mv += forward * step
            if QtCore.Qt.Key_S in self._wasd_keys: mv -= forward * step
            if QtCore.Qt.Key_A in self._wasd_keys: mv -= right * step
            if QtCore.Qt.Key_D in self._wasd_keys: mv += right * step
            if QtCore.Qt.Key_E in self._wasd_keys or \
               QtCore.Qt.Key_Space in self._wasd_keys: mv += up * step
            if QtCore.Qt.Key_Q in self._wasd_keys or \
               QtCore.Qt.Key_Control in self._wasd_keys: mv -= up * step
            if np.any(mv != 0):
                if self._fly_mode:
                    self._fly_pos += mv
                else:
                    self._scene_center += mv
                self.update()
        elif self._frame_dirty:
            self.update()

    # ---- Keyboard ----------------------------------------------
    def _clear_motion_keys(self):
        if self._wasd_keys:
            self._wasd_keys.clear()
            self.update()

    def _is_motion_key(self, key):
        return key in (
            QtCore.Qt.Key_W, QtCore.Qt.Key_A, QtCore.Qt.Key_S, QtCore.Qt.Key_D,
            QtCore.Qt.Key_Q, QtCore.Qt.Key_E, QtCore.Qt.Key_Space,
            QtCore.Qt.Key_Shift, QtCore.Qt.Key_Control,
        )

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == QtCore.Qt.Key_F:
            mods = ev.modifiers()
            # Shift+F: toggle fly mode. Plain F: frame the selected prim
            # (Pixar usdview convention) — falls back to scene if no sel.
            if mods & QtCore.Qt.ShiftModifier:
                self._detach_scene_camera_for_navigation()
                if not self._fly_mode:
                    # Orbit → fly: copy eye from the trackball, seed
                    # _fly_yaw/_fly_pitch from the orbit-quat forward vector
                    # so first-person look starts pointed at scene_center.
                    eye, _, _ = self._camera_eye_target_up()
                    self._fly_pos = eye.copy()
                    _, _, fwd = self._orbit_basis()
                    self._set_yaw_pitch_from_world_forward(fwd)
                    self._fly_mode = True
                else:
                    # Fly → orbit: place scene_center half a diag in front
                    # of the eye, then seed the trackball quaternion from
                    # the FPS yaw/pitch so the camera doesn't snap.
                    _right, _up, forward = self._fly_basis_from_yaw_pitch()
                    self._scene_center = self._fly_pos + forward * (
                        getattr(self, "_scene_diag", 10.0) * 0.5)
                    self._sync_orbit_from_yaw_pitch()
                    self._fly_mode = False
            else:
                self._frame_selected_prim()
            self.update()
            return
        if k == QtCore.Qt.Key_T:
            cycle = _supported_view_render_modes()
            i = cycle.index(self._render_mode) if \
                self._render_mode in cycle else 0
            self._render_mode = cycle[(i + 1) % len(cycle)]
            resync = getattr(self, "_resync_render_mode_menu", None)
            if resync: resync()
            self.update()
            return
        if k == QtCore.Qt.Key_H:
            self._show_hud = not self._show_hud
            menu_action = getattr(self, "_hud_menu_action", None)
            if menu_action is not None:
                menu_action.blockSignals(True)
                menu_action.setChecked(self._show_hud)
                menu_action.blockSignals(False)
            self.update()
            return
        # Gizmo keybindings disabled — re-enable along with _show_gizmo=True
        # in __init__ to bring the transform manipulator back.
        # if k == QtCore.Qt.Key_G:
        #     self._show_gizmo = not self._show_gizmo
        #     self.update()
        #     return
        # if k == QtCore.Qt.Key_1:
        #     self._gizmo.set_mode(GizmoMode.TRANSLATE)
        #     self.update()
        #     return
        # if k == QtCore.Qt.Key_2:
        #     self._gizmo.set_mode(GizmoMode.ROTATE)
        #     self.update()
        #     return
        # if k == QtCore.Qt.Key_3:
        #     self._gizmo.set_mode(GizmoMode.SCALE)
        #     self.update()
        #     return
        if k in (QtCore.Qt.Key_Escape,):
            return
        # WASD / QE / Space / Ctrl-down work in BOTH modes — fly walks the
        # eye, orbit pans the orbit target. Restrict to the keys we react
        # to so plain typing in any text field still flows to super().
        if self._is_motion_key(k):
            if ev.isAutoRepeat():
                ev.accept()
                return
            if k not in (QtCore.Qt.Key_Shift, QtCore.Qt.Key_Control):
                self._detach_scene_camera_for_navigation()
            self._wasd_keys.add(k)
            if ev.modifiers() & QtCore.Qt.ShiftModifier:
                self._wasd_keys.add(QtCore.Qt.Key_Shift)
            if ev.modifiers() & QtCore.Qt.ControlModifier:
                self._wasd_keys.add(QtCore.Qt.Key_Control)
            ev.accept()
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        k = ev.key()
        if ev.isAutoRepeat() and self._is_motion_key(k):
            ev.accept()
            return
        self._wasd_keys.discard(k)
        if not (ev.modifiers() & QtCore.Qt.ShiftModifier):
            self._wasd_keys.discard(QtCore.Qt.Key_Shift)
        if not (ev.modifiers() & QtCore.Qt.ControlModifier):
            self._wasd_keys.discard(QtCore.Qt.Key_Control)
        if self._is_motion_key(k):
            ev.accept()
            return
        super().keyReleaseEvent(ev)

    def focusOutEvent(self, ev):
        self._clear_motion_keys()
        super().focusOutEvent(ev)

    def hideEvent(self, ev):
        self._clear_motion_keys()
        super().hideEvent(ev)

    def event(self, ev):
        if ev.type() in (
            QtCore.QEvent.Type.WindowDeactivate,
            QtCore.QEvent.Type.ApplicationDeactivate,
        ):
            self._clear_motion_keys()
        return super().event(ev)

    # ---- Mouse -------------------------------------------------
    def mousePressEvent(self, ev):
        self._mouse_down = ev.button()
        self._last_mouse_pos = ev.pos()
        # Set focus so subsequent key presses (WASD, F, T) reach this widget
        # and not the tree / property editor.
        self.setFocus(QtCore.Qt.MouseFocusReason)
        # Gizmo gets first crack at LMB: if the cursor hits a handle, the
        # gizmo claims the drag and we skip arcball / camera input.
        if (ev.button() == QtCore.Qt.LeftButton
                and self._show_gizmo
                and self._selected_single_prim() is not None):
            try:
                prim = self._selected_single_prim()
                self._seed_gizmo_from_prim(prim)
                eye, target, _ = self._camera_eye_target_up()
                vp = self._build_vp(eye, target)
                st = self._populate_gizmo_app_state(
                    ev.pos().x(), ev.pos().y(), mouse_left=True,
                    eye=eye, vp_col=vp)
                self._gizmo.update(st, passive=False)
                self._gizmo.tick(self._gizmo_xform)
                if self._gizmo.is_dragging:
                    self._gizmo_dragging = True
                    self.update()
                    ev.accept()
                    return
            except Exception as e:  # noqa: BLE001
                if os.environ.get("NUVIEW_INPUT_TRACE"):
                    print(f"[stageView] gizmo pick error: {e}",
                          file=sys.stderr, flush=True)
        # Arm the arcball: in orbit mode, LMB-down records the start sphere
        # point so subsequent move events can compute incremental rotations.
        if ev.button() in (
                QtCore.Qt.LeftButton,
                QtCore.Qt.MiddleButton,
                QtCore.Qt.RightButton):
            self._detach_scene_camera_for_navigation()
        if not self._fly_mode and ev.button() == QtCore.Qt.LeftButton:
            self._arcball_start = self._arcball_project(ev.pos().x(), ev.pos().y())
        if os.environ.get("NUVIEW_INPUT_TRACE"):
            print(f"[stageView] mousePress button={ev.button()} fly={self._fly_mode}",
                  file=sys.stderr, flush=True)
        ev.accept()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._gizmo_dragging:
            try:
                eye, target, _ = self._camera_eye_target_up()
                vp = self._build_vp(eye, target)
                st = self._populate_gizmo_app_state(
                    ev.pos().x(), ev.pos().y(), mouse_left=False,
                    eye=eye, vp_col=vp)
                self._gizmo.update(st, passive=False)
                self._gizmo.tick(self._gizmo_xform)
                prim = self._selected_single_prim()
                if prim is not None:
                    self._commit_gizmo_to_prim(prim)
            finally:
                self._gizmo_dragging = False
                self._mouse_down = None
                self._arcball_start = None
                self.update()
            ev.accept()
            return
        self._mouse_down = None
        self._arcball_start = None
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._gizmo_dragging:
            self._last_mouse_pos = ev.pos()
            try:
                eye, target, _ = self._camera_eye_target_up()
                vp = self._build_vp(eye, target)
                st = self._populate_gizmo_app_state(
                    ev.pos().x(), ev.pos().y(), mouse_left=True,
                    eye=eye, vp_col=vp)
                self._gizmo.update(st, passive=False)
                self._gizmo.tick(self._gizmo_xform)
            except Exception as e:  # noqa: BLE001
                if os.environ.get("NUVIEW_INPUT_TRACE"):
                    print(f"[stageView] gizmo drag error: {e}",
                          file=sys.stderr, flush=True)
            self.update()
            ev.accept()
            return
        if self._mouse_down is None:
            return super().mouseMoveEvent(ev)
        dx = ev.pos().x() - self._last_mouse_pos.x()
        dy = ev.pos().y() - self._last_mouse_pos.y()
        self._last_mouse_pos = ev.pos()
        if self._fly_mode and self._mouse_down == QtCore.Qt.RightButton:
            # Mouse-look in fly mode (RMB) — yaw/pitch is the right model
            # here (FPS controls), no gimbal-lock concerns since we clamp
            # pitch to ~89°.
            self._fly_yaw -= dx * 0.005
            self._fly_pitch -= dy * 0.005
            self._fly_pitch = max(-1.55, min(1.55, self._fly_pitch))
            self.signalFrustumChanged.emit()
            self.update()
        elif not self._fly_mode and self._mouse_down == QtCore.Qt.LeftButton:
            # Turntable orbit keeps a stable world-up horizon. The previous
            # free arcball allowed roll, which made the up vector feel wrong
            # for viewport navigation.
            self._fly_yaw -= dx * 0.005
            self._fly_pitch -= dy * 0.005
            self._fly_pitch = max(-1.45, min(1.45, self._fly_pitch))
            self._sync_orbit_from_yaw_pitch()
            self.signalFrustumChanged.emit()
            self.update()
        elif not self._fly_mode and self._mouse_down == QtCore.Qt.MiddleButton:
            # Pan (MMB drag) along the camera-aligned right/up basis derived
            # from the orbit quaternion (so panning still feels right after
            # an over-the-top trackball flip).
            right, up, _ = self._orbit_basis()
            scale = getattr(self, "_scene_diag", 10.0) / 500.0
            self._scene_center += -right * dx * scale + up * dy * scale
            self.update()
        super().mouseMoveEvent(ev)

    def wheelEvent(self, ev):
        self._detach_scene_camera_for_navigation()
        delta = ev.angleDelta().y()
        if self._fly_mode:
            # Adjust fly speed.
            factor = 1.2 if delta > 0 else 1.0 / 1.2
            self._fly_speed = max(0.001, self._fly_speed * factor)
        else:
            # Zoom toward/away.
            factor = 0.9 if delta > 0 else 1.0 / 0.9
            distance = float(getattr(
                self, "_frame_distance", getattr(self, "_scene_diag", 10.0) * 0.9))
            self._frame_distance = max(0.001, distance * factor)
            self.signalFrustumChanged.emit()
        self.update()
        super().wheelEvent(ev)

    # ---- appController-facing API ------------------------------
    # Signals already declared above. Methods below cover everything
    # appController.py grep'd for.

    def closeRenderer(self):
        if self._renderer is not None:
            try: self._renderer.close()
            except Exception: pass
            self._renderer = None

    def updateView(self, resetCam=False, forceComputeBBox=False,
                   frameFit=1.1, *args, **kwargs):
        # Pixar's _frameSelection calls updateView(True, True). Treat
        # (resetCam, forceComputeBBox) as "frame the current selection
        # (or whole scene if no selection)".
        if resetCam or forceComputeBBox:
            try: self._frame_selected_prim()
            except Exception: pass
        self._pushTimeToRenderer()
        self._frame_dirty = True
        self.update()

    def updateForPlayback(self, *args, **kwargs):
        self.updateView()

    def _pushTimeToRenderer(self):
        """Push the data model's currentFrame to the renderer and trigger
        a fresh scene load so xformOp.timeSamples / etc. evaluate at the
        new time. Called from updateView (frame change) and updateForPlayback.
        Cheap no-op when the renderer has no OVRTX time-update hook."""
        if self._renderer is None:
            return
        if not hasattr(self._renderer, "update_from_usd_time"):
            return
        try:
            t = self._current_time_value()
            # Track last pushed time — only reload when it actually changes.
            if getattr(self, "_last_pushed_time", None) == t:
                return
            self._renderer.update_from_usd_time(t)
            self._last_pushed_time = t
            # Force a fresh scene load so the new time takes effect.
            self._stage_loaded_path = None
            self._loadStageIntoRenderer()
        except Exception as e:
            self._render_error = f"frame change: {e}"

    def updateBboxPurposes(self, *a, **kw):
        pass

    def updateSelection(self, *a, **kw):
        self._frame_dirty = True
        self.update()

    def copyViewState(self):
        return {
            "fly_pos": self._fly_pos.copy(),
            "fly_yaw": self._fly_yaw,
            "fly_pitch": self._fly_pitch,
            "fly_mode": self._fly_mode,
            "fly_speed": self._fly_speed,
            "scene_center": getattr(self, "_scene_center", np.zeros(3)).copy(),
            "scene_diag": getattr(self, "_scene_diag", 10.0),
        }

    def restoreViewState(self, state):
        if not state: return
        self._fly_pos = state.get("fly_pos", self._fly_pos).copy() \
            if hasattr(state.get("fly_pos", None), "copy") else self._fly_pos
        self._fly_yaw = state.get("fly_yaw", 0.0)
        self._fly_pitch = state.get("fly_pitch", 0.0)
        self._fly_mode = state.get("fly_mode", False)
        self._fly_speed = state.get("fly_speed", 1.0)
        self._scene_center = state.get("scene_center", np.zeros(3))
        self._scene_diag = state.get("scene_diag", 10.0)
        self.update()

    # ---- Renderer-plugin API (reflects the active backend so the UI
    # ---- header / dropdown matches what's actually rendering) -------
    def _backend_label(self):
        try:
            from nanousdview._backend import get_backend
            return {"vulkan": ("nanousd-vulkan-rt", "nanousd Vulkan RT", "Vulkan"),
                    "opengl": ("nanousd-opengl",    "nanousd OpenGL",    "OpenGL"),
                    "metal":  ("nanousd-metal-rt",  "nanousd Metal RT",  "Metal"),
                   }.get(get_backend(), ("nanousd", "nanousd", "?"))
        except Exception:
            return ("nanousd", "nanousd", "?")
    def GetRendererPlugins(self): return [self._backend_label()[0]]
    def GetRendererDisplayName(self, _id): return self._backend_label()[1]
    def GetRendererHgiDisplayName(self): return self._backend_label()[2]
    def GetCurrentRendererId(self): return self._backend_label()[0]
    def SetRendererPlugin(self, _id): return True
    def GetRendererAovs(self):
        return ["color"]

    def SetRendererAov(self, aov):
        return str(aov or "color") == "color"
    def SupportsCustomRendererAovs(self): return False
    def GetRendererSettingsList(self): return []
    def GetRendererSetting(self, _name): return None
    def SetRendererSetting(self, _name, _value): pass
    def GetRendererCommands(self): return []
    def InvokeRendererCommand(self, _cmd): pass
    def IsPauseRendererSupported(self): return False
    def IsStopRendererSupported(self): return False
    def IsRendererConverged(self): return True
    def SetRendererPaused(self, _b): pass
    def SetRendererStopped(self, _b): pass
    def PollForAsynchronousUpdates(self): return False
    def GetActiveRenderPassPrimPath(self): return ""
    def GetActiveRenderSettingsPrimPath(self): return ""
    def SetActiveRenderPassPrim(self, _path): pass
    def SetActiveRenderSettingsPrim(self, _path): pass

    @property
    def rendererAovName(self): return "color"

    @property
    def rendererDisplayName(self): return self._backend_label()[1]

    @property
    def allowAsync(self): return self._allowAsync
    @allowAsync.setter
    def allowAsync(self, v): self._allowAsync = bool(v)

    @property
    def bboxstandin(self): return self._bboxstandin
    @bboxstandin.setter
    def bboxstandin(self, v): self._bboxstandin = bool(v)

    @property
    def rolloverPicking(self): return self._rolloverPicking
    @rolloverPicking.setter
    def rolloverPicking(self, v): self._rolloverPicking = bool(v)

    def GetPhysicalWindowSize(self):
        return self.size().width(), self.size().height()

    # Pixar's QGLWidget had grabFrameBuffer; QWidget has grab() — alias.
    # appController passes cropToAspectRatio=False; we ignore it because
    # the QImage we return already reflects the renderer's aspect.
    def grabFrameBuffer(self, cropToAspectRatio=False):
        if self._frame_qimage is not None:
            return self._frame_qimage.copy()
        return self.grab().toImage()
