# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OVRTX backend selection for nanousdview.

nanousdview uses the OVRTX Python API as its only renderer-facing surface.
The ``vulkan``, ``opengl``, and ``metal`` choices select nanousd-backed
``ovrtx`` implementations; ``ovrtx`` selects NVIDIA's runtime directly.
"""

from __future__ import annotations

import importlib
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


_AVAILABLE = ("vulkan", "opengl", "metal", "ovrtx")
_NANOUSD_BACKENDS = ("vulkan", "opengl", "metal")

VIEW_RENDER_RASTER = 0
VIEW_RENDER_SHADOW = 1
VIEW_RENDER_RT = 2
_VIEW_RENDER_MODE_TOKENS = {
    VIEW_RENDER_RASTER: "raster",
    VIEW_RENDER_SHADOW: "shadow",
    VIEW_RENDER_RT: "rt",
}

VIEW_CAMERA_PATH = "/NanousdView/Camera"
VIEW_RENDER_PRODUCT_PATH = "/Render/Camera"


def _normalize3(value: tuple[float, float, float], fallback: tuple[float, float, float]) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64)
    n = float(np.linalg.norm(vec))
    if n <= 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return vec / n


def _distant_light_transform(
    direction: tuple[float, float, float],
    up_hint: tuple[float, float, float],
) -> str:
    normal = _normalize3(direction, (0.0, 0.0, -1.0))
    z_axis = -normal
    up = _normalize3(up_hint, (0.0, 1.0, 0.0))
    if abs(float(np.dot(up, z_axis))) > 0.95:
        up = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    x_axis = np.cross(up, z_axis)
    x_axis = _normalize3(tuple(float(x) for x in x_axis), (1.0, 0.0, 0.0))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = _normalize3(tuple(float(x) for x in y_axis), tuple(float(x) for x in up))
    rows = (
        (x_axis[0], x_axis[1], x_axis[2], 0.0),
        (y_axis[0], y_axis[1], y_axis[2], 0.0),
        (z_axis[0], z_axis[1], z_axis[2], 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    return "(" + ", ".join(
        "(" + ", ".join(f"{float(v):.9g}" for v in row) + ")"
        for row in rows
    ) + ")"


def _fallback_studio_light_layer(
    *,
    stage_up: str,
    camera_light: bool,
    dome_light: bool,
    backend: str,
    render_mode: int,
) -> str:
    if not camera_light and not dome_light:
        return ""
    z_up = str(stage_up).upper() == "Z"
    up_hint = (0.0, 0.0, 1.0) if z_up else (0.0, 1.0, 0.0)
    if z_up:
        key_dir = (-0.45, -0.35, -0.82)
        fill_dir = (0.70, -0.15, -0.35)
        rim_dir = (0.25, 0.85, -0.45)
    else:
        key_dir = (-0.45, -0.82, -0.35)
        fill_dir = (0.70, -0.35, 0.15)
        rim_dir = (0.25, -0.45, 0.85)

    if backend == "ovrtx":
        dome_color = (0.78, 0.82, 0.90)
        dome_intensity = 360
        key_intensity = 520
        fill_color = (0.62, 0.72, 1.0)
        fill_intensity = 90
        rim_intensity = 120
    elif backend == "opengl":
        dome_color = (0.84, 0.85, 0.88)
        dome_intensity = 390
        key_intensity = 900
        fill_color = (0.78, 0.82, 0.96)
        fill_intensity = 360
        rim_intensity = 900
    elif render_mode == VIEW_RENDER_RASTER:
        dome_color = (0.82, 0.84, 0.88)
        dome_intensity = 350
        key_intensity = 520
        fill_color = (0.72, 0.78, 0.96)
        fill_intensity = 190
        rim_intensity = 520
    else:
        dome_color = (0.82, 0.84, 0.88)
        dome_intensity = 330
        key_intensity = 430
        fill_color = (0.70, 0.76, 0.95)
        fill_intensity = 130
        rim_intensity = 300

    lights = []
    if dome_light:
        lights.append(f"""
    def DomeLight "DefaultDomeLight" {{
        color3f inputs:color = ({dome_color[0]}, {dome_color[1]}, {dome_color[2]})
        float inputs:intensity = {dome_intensity}
    }}
""")
    if camera_light:
        lights.append(f"""
    def DistantLight "DefaultKeyLight" {{
        color3f inputs:color = (1.0, 0.94, 0.84)
        float inputs:intensity = {key_intensity}
        float inputs:angle = 6
        matrix4d xformOp:transform = {_distant_light_transform(key_dir, up_hint)}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def DistantLight "DefaultFillLight" {{
        color3f inputs:color = ({fill_color[0]}, {fill_color[1]}, {fill_color[2]})
        float inputs:intensity = {fill_intensity}
        float inputs:angle = 16
        matrix4d xformOp:transform = {_distant_light_transform(fill_dir, up_hint)}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def DistantLight "DefaultRimLight" {{
        color3f inputs:color = (1.0, 0.96, 0.88)
        float inputs:intensity = {rim_intensity}
        float inputs:angle = 10
        matrix4d xformOp:transform = {_distant_light_transform(rim_dir, up_hint)}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
""")
        if backend == "ovrtx":
            lights.append("""
    def SphereLight "DefaultCameraLight" {
        color3f inputs:color = (1.0, 0.96, 0.90)
        float inputs:intensity = 12000
        float inputs:radius = 0.6
    }
""")
    return "".join(lights)


def _platform_default() -> str:
    return "metal" if platform.system() == "Darwin" else "vulkan"


def get_backend() -> str:
    """Return the selected viewer backend name."""
    name = (
        os.environ.get("NANOUSD_VIEW_BACKEND")
        or os.environ.get("NANOUSD_OVRTX_BACKEND")
        or ""
    ).strip().lower()
    if not name:
        name = _platform_default()
    if name not in _AVAILABLE:
        raise ValueError(f"Unknown renderer backend {name!r}. Pick one of {_AVAILABLE}.")
    return name


def _workspace_from_here() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _prepend_path(path: Path) -> None:
    s = str(path)
    if not path.is_dir():
        return
    sys.path[:] = [p for p in sys.path if p != s]
    sys.path.insert(0, s)


def _remove_path(path: Path) -> None:
    s = str(path)
    sys.path[:] = [p for p in sys.path if p != s]


def _nanousd_python_paths(workspace: Path) -> tuple[Path, ...]:
    return (
        workspace / "nanousd-vulkan-renderer" / "python",
        workspace / "nanousd-opengl-renderer" / "python",
        workspace / "nanousd-metal-renderer" / "python",
    )


def _evict_ovrtx_modules() -> None:
    for key in list(sys.modules):
        if key == "ovrtx" or key.startswith("ovrtx."):
            del sys.modules[key]


def configure_backend(backend: Optional[str] = None, workspace: Optional[Path] = None) -> str:
    """Configure import paths and environment for the selected OVRTX backend."""
    name = (backend or get_backend()).strip().lower()
    if name not in _AVAILABLE:
        raise ValueError(f"Unknown renderer backend {name!r}. Pick one of {_AVAILABLE}.")

    workspace = workspace or _workspace_from_here()
    os.environ["NANOUSD_VIEW_BACKEND"] = name

    if name in _NANOUSD_BACKENDS:
        os.environ["NANOUSD_OVRTX_BACKEND"] = name
        backend_dirs = {
            "vulkan": (),
            "opengl": ("nanousd-opengl-renderer",),
            "metal": ("nanousd-metal-renderer",),
        }[name]
        for dirname in backend_dirs:
            _prepend_path(workspace / dirname / "python")
        # The nanousd OVRTX facade lives with the Vulkan renderer and is the
        # stable API front door for every nanousd renderer implementation.
        _prepend_path(workspace / "nanousd-vulkan-renderer" / "python")
    else:
        os.environ.pop("NANOUSD_OVRTX_BACKEND", None)
        for path in _nanousd_python_paths(workspace):
            _remove_path(path)

    return name


def _import_ovrtx():
    backend = configure_backend()
    saved_pxr = {k: v for k, v in sys.modules.items() if k == "pxr" or k.startswith("pxr.")}
    for key in saved_pxr:
        del sys.modules[key]
    try:
        _evict_ovrtx_modules()
        module = importlib.import_module("ovrtx")
    finally:
        sys.modules.update(saved_pxr)

    version = str(getattr(module, "__version__", ""))
    is_nanousd = version.endswith("-nanousd") or version.endswith("+nanousd")
    if backend in _NANOUSD_BACKENDS and not is_nanousd:
        raise ImportError(
            f"Backend {backend!r} requires the nanousd ovrtx facade. "
            "Ensure nanousd-vulkan-renderer/python is ahead of the official ovrtx wheel."
        )
    if backend == "ovrtx" and is_nanousd:
        raise ImportError(
            "Backend 'ovrtx' selected the nanousd ovrtx facade instead of NVIDIA OVRTX. "
            "Remove nanousd-vulkan-renderer/python from PYTHONPATH for this backend."
        )
    return module


def validate_renderer() -> None:
    """Import the selected OVRTX implementation and construct a Renderer."""
    if get_backend() == "metal" and platform.system() != "Darwin":
        raise ImportError("Metal backend is only supported on macOS.")
    ovrtx = _import_ovrtx()
    renderer = ovrtx.Renderer(ovrtx.RendererConfig())
    del renderer


def _camera_matrix(eye: Any, target: Any, up: Any) -> np.ndarray:
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    up_v = np.asarray(up, dtype=np.float64)
    forward = target_v - eye_v
    fn = float(np.linalg.norm(forward))
    if fn <= 1e-12:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        forward /= fn
    up_n = float(np.linalg.norm(up_v))
    if up_n <= 1e-12:
        up_v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        up_v /= up_n
    right = np.cross(forward, up_v)
    rn = float(np.linalg.norm(right))
    if rn <= 1e-12:
        up_v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up_v)
        rn = float(np.linalg.norm(right))
    if rn <= 1e-12:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right /= rn
    camera_up = np.cross(right, forward)
    un = float(np.linalg.norm(camera_up))
    if un > 1e-12:
        camera_up /= un

    mat = np.eye(4, dtype=np.float64)
    mat[0] = [right[0], right[1], right[2], 0.0]
    mat[1] = [camera_up[0], camera_up[1], camera_up[2], 0.0]
    mat[2] = [-forward[0], -forward[1], -forward[2], 0.0]
    mat[3] = [eye_v[0], eye_v[1], eye_v[2], 1.0]
    return np.ascontiguousarray(mat)


def _camera_lens_arrays(
    fov_degrees: float,
    aspect: float,
    near_clip: float,
    far_clip: float,
    focal_length: Optional[float] = None,
    horizontal_aperture: Optional[float] = None,
    vertical_aperture: Optional[float] = None,
):
    if (
        focal_length is not None
        and horizontal_aperture is not None
        and vertical_aperture is not None
        and focal_length > 1e-6
        and horizontal_aperture > 1e-6
        and vertical_aperture > 1e-6
    ):
        focal = float(focal_length)
        horizontal = float(horizontal_aperture)
        vertical = float(vertical_aperture)
    else:
        vertical = 20.955
        fov = math.radians(max(float(fov_degrees), 1.0))
        focal = vertical / max(2.0 * math.tan(fov * 0.5), 1e-6)
        horizontal = vertical * max(float(aspect), 1e-6)
    return (
        np.asarray([focal], dtype=np.float32),
        np.asarray([horizontal], dtype=np.float32),
        np.asarray([vertical], dtype=np.float32),
        np.asarray([[near_clip, far_clip]], dtype=np.float32),
    )


def _normalize3(value: Any, fallback: tuple[float, float, float]) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(vec))
    if n <= 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return vec / n


def _view_projection_matrix(
    eye: Any,
    target: Any,
    up: Any,
    fov_degrees: float,
    width: int,
    height: int,
    near_clip: float,
    far_clip: float,
) -> np.ndarray:
    """Forward row-major view-projection matrix for deferred visibility cull."""
    eye_v = np.asarray(eye, dtype=np.float32).reshape(3)
    target_v = np.asarray(target, dtype=np.float32).reshape(3)
    forward = _normalize3(target_v - eye_v, (0.0, 0.0, -1.0))
    up_v = _normalize3(up, (0.0, 1.0, 0.0))
    right = np.cross(forward, up_v)
    if float(np.linalg.norm(right)) <= 1e-6:
        up_v = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        right = np.cross(forward, up_v)
    right = _normalize3(right, (1.0, 0.0, 0.0))
    camera_up = _normalize3(np.cross(right, forward), (0.0, 1.0, 0.0))

    view = np.zeros((4, 4), dtype=np.float32)
    view[0, 0:3] = right
    view[0, 3] = -float(np.dot(right, eye_v))
    view[1, 0:3] = camera_up
    view[1, 3] = -float(np.dot(camera_up, eye_v))
    view[2, 0:3] = -forward
    view[2, 3] = float(np.dot(forward, eye_v))
    view[3, 3] = 1.0

    fov = math.radians(float(fov_degrees))
    tan_half = math.tan(fov * 0.5)
    if not np.isfinite(tan_half) or tan_half <= 1e-8:
        tan_half = math.tan(math.radians(60.0) * 0.5)
    aspect = max(float(width), 1.0) / max(float(height), 1.0)
    near = max(float(near_clip), 1e-6)
    far = max(float(far_clip), near + 1.0)

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 1.0 / (aspect * tan_half)
    proj[1, 1] = -1.0 / tan_half
    proj[2, 2] = far / (near - far)
    proj[2, 3] = -(far * near) / (far - near)
    proj[3, 2] = -1.0
    return np.ascontiguousarray((proj @ view).reshape(16))


class OvrtxViewportRenderer:
    """Small viewport helper that drives only ``ovrtx.Renderer`` APIs."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        enable_rt: Optional[bool] = None,
        enable_materials: Optional[bool] = None,
    ):
        self._ovrtx = _import_ovrtx()
        self._backend = get_backend()
        self._width = int(width)
        self._height = int(height)
        config_kwargs = {}
        if self._backend in _NANOUSD_BACKENDS:
            if enable_rt is not None:
                config_kwargs["enable_rt"] = bool(enable_rt)
            if enable_materials is not None:
                config_kwargs["enable_materials"] = bool(enable_materials)
        if self._backend == "ovrtx":
            config_kwargs.update(
                keep_system_alive=False,
                selection_outline_enabled=False,
            )
            log_path = os.environ.get("NUVIEW_OVRTX_LOG")
            if log_path:
                config_kwargs["log_file_path"] = log_path
                config_kwargs["log_level"] = os.environ.get("NUVIEW_OVRTX_LOG_LEVEL", "warn")
        self._renderer = self._ovrtx.Renderer(
            self._ovrtx.RendererConfig(**config_kwargs)
        )
        self._usd_path: Optional[str] = None
        self._time = 0.0
        self._camera_dirty = True
        self._last_camera = None
        self._last_lens = None
        self._render_mode = VIEW_RENDER_RT if self._backend != "opengl" else VIEW_RENDER_RASTER
        self._deferred_visible_extracted = False
        self._fallback_camera_light = False
        self._fallback_dome_light = False
        self._fallback_stage_up = "Y"

    @property
    def backend(self) -> str:
        return self._backend

    def close(self) -> None:
        renderer = self._renderer
        self._renderer = None
        if renderer is not None and hasattr(renderer, "reset_stage"):
            try:
                renderer.reset_stage()
            except Exception:
                pass

    def load_stage(self, usd_path: str) -> None:
        usd_path = str(Path(usd_path).resolve())
        if usd_path == self._usd_path:
            return
        self._usd_path = usd_path
        self._rebuild_stage()
        self._camera_dirty = True
        self._deferred_visible_extracted = False

    def set_size(self, width: int, height: int) -> None:
        width = max(int(width), 1)
        height = max(int(height), 1)
        if width == self._width and height == self._height:
            return
        self._width = width
        self._height = height
        if self._backend == "ovrtx" and self._usd_path:
            last_camera = self._last_camera
            last_lens = self._last_lens
            self._rebuild_stage()
            if last_camera is not None and last_lens is not None:
                eye, target, up = last_camera
                (
                    fov_degrees, near_clip, far_clip, focal_length,
                    horizontal_aperture, vertical_aperture,
                ) = last_lens
                self.set_camera(
                    eye, target, up, fov_degrees, near_clip, far_clip,
                    focal_length=focal_length,
                    horizontal_aperture=horizontal_aperture,
                    vertical_aperture=vertical_aperture,
                )
            else:
                self._camera_dirty = True
        else:
            self._write_resolution()
            self._camera_dirty = True

    def set_render_mode(self, mode: int) -> None:
        mode = int(mode)
        if mode == self._render_mode:
            return
        self._render_mode = mode
        if (
            self._backend != "ovrtx"
            and self._usd_path
            and (self._fallback_camera_light or self._fallback_dome_light)
        ):
            self._rebuild_stage()
            return
        self._write_render_mode()

    def set_default_lighting(
        self,
        camera_light: bool = False,
        dome_light: bool = False,
        stage_up: str = "Y",
    ) -> None:
        camera_light = bool(camera_light)
        dome_light = bool(dome_light)
        stage_up = "Z" if str(stage_up).upper() == "Z" else "Y"
        if (
            camera_light == self._fallback_camera_light
            and dome_light == self._fallback_dome_light
            and stage_up == self._fallback_stage_up
        ):
            return
        self._fallback_camera_light = camera_light
        self._fallback_dome_light = dome_light
        self._fallback_stage_up = stage_up
        if self._usd_path:
            self._rebuild_stage()

    def update_from_usd_time(self, usd_time: float) -> None:
        self._time = float(usd_time)
        if self._usd_path:
            self._renderer.update_from_usd_time(float(usd_time))

    def set_camera(
        self,
        eye: Any,
        target: Any,
        up: Any,
        fov_degrees: float,
        near_clip: float,
        far_clip: float,
        focal_length: Optional[float] = None,
        horizontal_aperture: Optional[float] = None,
        vertical_aperture: Optional[float] = None,
    ) -> None:
        aspect = self._width / max(self._height, 1)
        mat = _camera_matrix(eye, target, up)
        self._renderer.write_attribute(
            prim_paths=[VIEW_CAMERA_PATH],
            attribute_name="omni:xform",
            tensor=mat.reshape(1, 4, 4),
            semantic=self._ovrtx.Semantic.XFORM_MAT4x4,
            prim_mode=self._ovrtx.PrimMode.CREATE_NEW,
        )
        if self._backend == "ovrtx" and self._fallback_camera_light:
            try:
                self._renderer.write_attribute(
                    prim_paths=["/NanousdView/DefaultCameraLight"],
                    attribute_name="omni:xform",
                    tensor=mat.reshape(1, 4, 4),
                    semantic=self._ovrtx.Semantic.XFORM_MAT4x4,
                    prim_mode=self._ovrtx.PrimMode.CREATE_NEW,
                )
            except Exception:
                pass
        focal, horizontal, vertical, clipping = _camera_lens_arrays(
            fov_degrees, aspect, near_clip, far_clip,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
        )
        for name, value in (
            ("focalLength", focal),
            ("horizontalAperture", horizontal),
            ("verticalAperture", vertical),
            ("clippingRange", clipping),
        ):
            try:
                self._renderer.write_attribute([VIEW_CAMERA_PATH], name, value)
            except Exception:
                pass
        self._last_camera = (tuple(float(x) for x in eye), tuple(float(x) for x in target), tuple(float(x) for x in up))
        self._last_lens = (
            float(fov_degrees), float(near_clip), float(far_clip),
            None if focal_length is None else float(focal_length),
            None if horizontal_aperture is None else float(horizontal_aperture),
            None if vertical_aperture is None else float(vertical_aperture),
        )
        self._camera_dirty = False

    def _maybe_extract_visible_deferred(self) -> None:
        if self._deferred_visible_extracted:
            return
        if os.environ.get("NUSD_LAZY_MESH", "").strip() in ("", "0"):
            return
        if os.environ.get("NUSD_VIEW_VISIBLE_EXTRACT", "1").strip().lower() in (
            "0", "false", "off", "no",
        ):
            return
        try:
            spec = getattr(self._renderer, "_render_product_specs", {}).get(
                VIEW_RENDER_PRODUCT_PATH
            )
            if spec is not None and hasattr(self._renderer, "_ensure_native_stage_for_product"):
                self._renderer._ensure_native_stage_for_product(spec)
            nu = getattr(self._renderer, "_nu", None)
            if nu is None and hasattr(self._renderer, "_ensure_renderer"):
                nu = self._renderer._ensure_renderer()
            if nu is None:
                return
            if self._last_camera is None or self._last_lens is None:
                if hasattr(nu, "extract_deferred"):
                    nu.extract_deferred()
                    self._deferred_visible_extracted = True
                return
            if not hasattr(nu, "extract_deferred_visible"):
                if hasattr(nu, "extract_deferred"):
                    nu.extract_deferred()
                    self._deferred_visible_extracted = True
                return
            eye, target, up = self._last_camera
            fov_degrees, near_clip, far_clip = self._last_lens[:3]
            vp = _view_projection_matrix(
                eye, target, up, fov_degrees,
                self._width, self._height, near_clip, far_clip,
            )
            nu.extract_deferred_visible([vp])
            self._deferred_visible_extracted = True
        except Exception:
            self._deferred_visible_extracted = True
            raise

    def render_ldr(self, delta_time: float = 0.0, render_mode: Optional[int] = None) -> np.ndarray:
        if self._backend == "ovrtx":
            render_mode = VIEW_RENDER_RT
        else:
            mode_override = os.environ.get("NANOUSD_VIEW_RENDER_MODE", "").strip().lower()
            if mode_override in ("raster", "raster_only"):
                render_mode = VIEW_RENDER_RASTER
            elif mode_override in ("shadow", "raster_shadow", "raster+shadow"):
                render_mode = VIEW_RENDER_SHADOW
            elif mode_override in ("rt", "raytrace", "raytraced"):
                render_mode = VIEW_RENDER_RT
        if render_mode is not None:
            self.set_render_mode(render_mode)
        self._maybe_extract_visible_deferred()
        default_timeout_ns = "60000000000" if self._backend == "ovrtx" else "5000000000"
        timeout_ns = int(os.environ.get("NANOUSD_VIEW_STEP_TIMEOUT_NS", default_timeout_ns))
        pending = self._renderer.step_async({VIEW_RENDER_PRODUCT_PATH}, float(delta_time)).wait(timeout_ns=timeout_ns)
        if pending is None:
            raise TimeoutError("OVRTX step did not complete before wait timeout")
        products = pending.fetch(timeout_ns=timeout_ns)
        if products is None:
            raise TimeoutError("OVRTX step did not produce render products before fetch timeout")
        try:
            product = products[VIEW_RENDER_PRODUCT_PATH]
            frame = product.frames[-1]
            render_var = frame.render_vars["LdrColor"]
            with render_var.map(device=self._ovrtx.Device.CPU) as mapped:
                source = mapped if hasattr(mapped, "__dlpack__") else mapped.tensor
                pixels = np.from_dlpack(source)
                return np.ascontiguousarray(np.array(pixels, copy=True))
        finally:
            destroy = getattr(products, "destroy", None)
            if callable(destroy):
                destroy()

    def _push_render_layer(self, include_scene: bool) -> None:
        sublayers = ""
        if include_scene and self._usd_path:
            layer_path = self._usd_path.replace("\\", "/")
            sublayers = f"""
(
    subLayers = [
        @{layer_path}@
    ]
)
"""
        mode_token = _VIEW_RENDER_MODE_TOKENS.get(self._render_mode, "rt")
        if self._backend == "ovrtx":
            mode = os.environ.get("NUVIEW_OVRTX_RENDER_MODE", "rt2").strip()
            mode_key = "".join(ch for ch in mode.lower() if ch.isalnum())
            if mode_key in ("pt", "pathtrace", "pathtracing"):
                render_mode = "PathTracing"
                default_spp = "16"
            elif mode_key in ("minimal",):
                render_mode = "Minimal"
                default_spp = "1"
            elif mode_key in ("legacyrt", "raytraced", "raytracedlighting"):
                render_mode = "RaytracedLighting"
                default_spp = "1"
            else:
                render_mode = "RealTimePathTracing"
                default_spp = "1"
            spp = max(int(os.environ.get("NUVIEW_OVRTX_SPP", default_spp)), 1)
            denoise = os.environ.get(
                "NUVIEW_OVRTX_DENOISE",
                "1" if render_mode == "PathTracing" else "0",
            ).lower() not in ("0", "false", "off", "no")
            max_samples = max(self._width * self._height * spp, 1)
            if render_mode == "PathTracing":
                render_mode_settings = f"""        token omni:rtx:rendermode = "{render_mode}"
        bool omni:rtx:pt:denoising:enabled = {"true" if denoise else "false"}
        uint omni:rtx:pt:samplesPerPixel = {spp}
        int omni:rtx:pt:maxSamplesPerLaunch = {max_samples}
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]"""
            elif render_mode == "Minimal":
                render_mode_settings = f"""        token omni:rtx:rendermode = "{render_mode}"
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]"""
            else:
                render_mode_settings = f"""        token omni:rtx:rendermode = "{render_mode}"
        token omni:rtx:post:aa:op = "dlss"
        token omni:rtx:post:dlss:execMode = "quality"
        token omni:rtx:directLighting:domeLight:denoisingTechnique = "ReLAX"
        token omni:rtx:directLighting:sampledLighting:denoisingTechnique = "NRD ReLAX"
        bool omni:rtx:rtpt:fireflyFilter:enabled = true
        int omni:rtx:rtpt:spp = {spp}
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]"""
        else:
            render_mode_settings = (
                f'        custom token nanousd:renderMode = "{mode_token}"'
            )
        fallback_lights = _fallback_studio_light_layer(
            stage_up=self._fallback_stage_up,
            camera_light=self._fallback_camera_light,
            dome_light=self._fallback_dome_light,
            backend=self._backend,
            render_mode=self._render_mode,
        )
        layer = f"""#usda 1.0
{sublayers}

def "NanousdView" {{
    def Camera "Camera" {{
        float focalLength = 18.1476
        float horizontalAperture = 27.94
        float verticalAperture = 20.955
        float2 clippingRange = (0.001, 10000)
    }}
{fallback_lights}
}}

def "Render" {{
    def RenderProduct "Camera" {{
        int2 resolution = ({self._width}, {self._height})
{render_mode_settings}
        rel camera = <{VIEW_CAMERA_PATH}>
        rel orderedVars = [<LdrColor>]

        def RenderVar "LdrColor" {{
            string sourceName = "LdrColor"
        }}
    }}
}}
"""
        self._renderer.open_usd_from_string(layer)

    def _rebuild_stage(self) -> None:
        self._deferred_visible_extracted = False
        self._push_render_layer(include_scene=True)
        if self._time:
            self._renderer.update_from_usd_time(float(self._time))

    def _write_resolution(self) -> None:
        value = np.asarray([[self._width, self._height]], dtype=np.int32)
        try:
            self._renderer.write_attribute([VIEW_RENDER_PRODUCT_PATH], "resolution", value)
        except Exception:
            if self._usd_path:
                self._rebuild_stage()

    def _write_render_mode(self) -> None:
        if self._backend == "ovrtx":
            if self._usd_path:
                self._rebuild_stage()
            return
        token = _VIEW_RENDER_MODE_TOKENS.get(self._render_mode, "rt")
        try:
            self._renderer.write_attribute([VIEW_RENDER_PRODUCT_PATH], "nanousd:renderMode", np.asarray([token]))
        except Exception:
            if self._usd_path:
                self._rebuild_stage()


__all__ = [
    "VIEW_RENDER_RASTER",
    "VIEW_RENDER_SHADOW",
    "VIEW_RENDER_RT",
    "VIEW_RENDER_PRODUCT_PATH",
    "OvrtxViewportRenderer",
    "configure_backend",
    "get_backend",
    "validate_renderer",
]
