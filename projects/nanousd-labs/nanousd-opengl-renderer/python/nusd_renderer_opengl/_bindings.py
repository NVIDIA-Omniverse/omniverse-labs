# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ctypes bindings for the nanousd OpenGL renderer dynamic library."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import numpy as np
from pathlib import Path

# Render mode constants
NU_RENDER_RASTER = 0
NU_RENDER_SHADOW = 1
NU_RENDER_RT = 2

# Pixel format constants
NU_PIXEL_RGBA8 = 0
NU_PIXEL_RGBAF32 = 1
NU_PIXEL_BGRA8 = 2  # raw swapchain order (returned by nu_fetch_pixels_cuda)

# Error codes
NU_OK = 0
NU_ERROR = -1
NU_ERROR_NO_RT = -2
NU_ERROR_NO_SCENE = -3
NU_ERROR_BAD_ID = -4

NU_EXPOSURE_HAS_FSTOP = 1 << 0
NU_EXPOSURE_HAS_RESPONSIVITY = 1 << 1
NU_EXPOSURE_HAS_TIME = 1 << 2
NU_EXPOSURE_HAS_WHITE_POINT = 1 << 3
NU_EXPOSURE_HAS_TONEMAP_FNUM = 1 << 4
NU_EXPOSURE_HAS_TONEMAP_CM2 = 1 << 5
NU_EXPOSURE_HAS_AUTO_EXPOSURE = 1 << 6


def _find_library():
    """Find libnusd_renderer_opengl (.dylib on macOS, .so elsewhere)."""
    here = Path(__file__).parent
    suffixes = (".dylib", ".so") if sys.platform == "darwin" else (".so", ".dylib")
    base_dirs = [here, here.parent.parent / "build", here.parent.parent / "build" / "Release"]
    candidates = [d / f"libnusd_renderer_opengl{s}" for d in base_dirs for s in suffixes]

    env_path = os.environ.get("NUSD_RENDERER_LIB")
    if env_path:
        candidates.insert(0, Path(env_path))

    for p in candidates:
        if p.exists():
            return str(p)

    # Try system library search
    found = ctypes.util.find_library("nusd_renderer_opengl")
    if found:
        return found

    raise RuntimeError(
        "libnusd_renderer_opengl not found. Set NUSD_RENDERER_LIB or build the library."
    )


# C struct definitions
class NuRendererConfig(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("gpu_index", ctypes.c_int),
        ("enable_rt", ctypes.c_int),
        ("enable_materials", ctypes.c_int),
        ("visible", ctypes.c_int),
    ]


class NuMeshDesc(ctypes.Structure):
    _fields_ = [
        ("positions", ctypes.POINTER(ctypes.c_float)),
        ("normals", ctypes.POINTER(ctypes.c_float)),
        ("colors", ctypes.POINTER(ctypes.c_float)),
        ("texcoords", ctypes.POINTER(ctypes.c_float)),
        ("indices", ctypes.POINTER(ctypes.c_uint32)),
        ("nvertices", ctypes.c_int),
        ("nindices", ctypes.c_int),
        ("transform", ctypes.POINTER(ctypes.c_float)),
        ("display_color", ctypes.c_float * 3),
        ("material_index", ctypes.c_int),
        ("name", ctypes.c_char_p),
    ]


class NuCameraDesc(ctypes.Structure):
    _fields_ = [
        ("eye", ctypes.c_float * 3),
        ("target", ctypes.c_float * 3),
        ("fov_degrees", ctypes.c_float),
        ("near_clip", ctypes.c_float),
        ("far_clip", ctypes.c_float),
    ]


class NuExposureDesc(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("exposure_f_stop", ctypes.c_float),
        ("exposure_responsivity", ctypes.c_float),
        ("exposure_time", ctypes.c_float),
        ("white_point_scale", ctypes.c_float),
        ("tonemap_f_number", ctypes.c_float),
        ("tonemap_cm2_factor", ctypes.c_float),
        ("auto_exposure_enabled", ctypes.c_int),
    ]


class NuPhaseTimings(ctypes.Structure):
    """perf/vk-instrumentation: per-phase GPU timings in milliseconds.

    Mirrors NuPhaseTimings in nusd_renderer.h. Populated via
    NuRenderer.get_phase_timings_ms(). All fields are 0.0 until the
    corresponding phase has run at least once."""
    _fields_ = [
        ("rt_dispatch_ms", ctypes.c_float),
        ("pixel_readback_ms", ctypes.c_float),
        ("blas_build_ms", ctypes.c_float),
        ("tlas_build_ms", ctypes.c_float),
        ("curve_blas_build_ms", ctypes.c_float),
        ("staging_upload_segs_ms", ctypes.c_float),
        ("staging_upload_aabbs_ms", ctypes.c_float),
        ("staging_upload_colors_ms", ctypes.c_float),
    ]


class _Lib:
    """Lazy-loaded shared library wrapper."""

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            path = _find_library()
            lib = ctypes.CDLL(path)
            cls._setup_functions(lib)
            cls._instance = lib
        return cls._instance

    @classmethod
    def _setup_functions(cls, lib):
        """Declare C function signatures."""
        lib.nu_renderer_create.argtypes = [ctypes.POINTER(NuRendererConfig)]
        lib.nu_renderer_create.restype = ctypes.c_void_p

        lib.nu_renderer_destroy.argtypes = [ctypes.c_void_p]
        lib.nu_renderer_destroy.restype = None

        lib.nu_add_mesh.argtypes = [ctypes.c_void_p, ctypes.POINTER(NuMeshDesc)]
        lib.nu_add_mesh.restype = ctypes.c_int

        lib.nu_add_mesh_instance.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.nu_add_mesh_instance.restype = ctypes.c_int

        lib.nu_remove_mesh.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.nu_remove_mesh.restype = ctypes.c_int

        lib.nu_clear_scene.argtypes = [ctypes.c_void_p]
        lib.nu_clear_scene.restype = None

        lib.nu_load_usd.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nu_load_usd.restype = ctypes.c_int

        if hasattr(lib, "nu_get_scene_bounds"):
            lib.nu_get_scene_bounds.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float * 3),
                ctypes.POINTER(ctypes.c_float * 3),
            ]
            lib.nu_get_scene_bounds.restype = ctypes.c_int

        # nu_load_usd_from_handle(NuRenderer*, NanousdStage stage, const char* label)
        # Optional — present only on builds that have it (renderer w/ Phase 5).
        if hasattr(lib, "nu_load_usd_from_handle"):
            lib.nu_load_usd_from_handle.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
            ]
            lib.nu_load_usd_from_handle.restype = ctypes.c_int

        if hasattr(lib, "nu_load_usd_from_handle_with_dir"):
            lib.nu_load_usd_from_handle_with_dir.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_char_p, ctypes.c_char_p,
            ]
            lib.nu_load_usd_from_handle_with_dir.restype = ctypes.c_int

        if hasattr(lib, "nu_get_camera"):
            lib.nu_get_camera.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(NuCameraDesc),
            ]
            lib.nu_get_camera.restype = ctypes.c_int

        if hasattr(lib, "nu_set_camera_explicit"):
            lib.nu_set_camera_explicit.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
            ]
            lib.nu_set_camera_explicit.restype = ctypes.c_int

        if hasattr(lib, "nu_set_camera_explicit_window"):
            lib.nu_set_camera_explicit_window.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
            ]
            lib.nu_set_camera_explicit_window.restype = ctypes.c_int

        if hasattr(lib, "nu_set_exposure"):
            lib.nu_set_exposure.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(NuExposureDesc),
            ]
            lib.nu_set_exposure.restype = ctypes.c_int

        if hasattr(lib, "nu_get_mesh_name"):
            lib.nu_get_mesh_name.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.c_char_p, ctypes.c_int,
            ]
            lib.nu_get_mesh_name.restype = ctypes.c_int
        if hasattr(lib, "nu_get_mesh_transform"):
            lib.nu_get_mesh_transform.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.nu_get_mesh_transform.restype = ctypes.c_int

        lib.nu_build_accel.argtypes = [ctypes.c_void_p]
        lib.nu_build_accel.restype = ctypes.c_int

        # nu_set_current_time(NuRenderer*, double time) — used by animation.
        if hasattr(lib, "nu_set_current_time"):
            lib.nu_set_current_time.argtypes = [ctypes.c_void_p, ctypes.c_double]
            lib.nu_set_current_time.restype = ctypes.c_int

        lib.nu_set_transforms.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        lib.nu_set_transforms.restype = ctypes.c_int

        lib.nu_set_colors.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        lib.nu_set_colors.restype = ctypes.c_int

        lib.nu_set_visibility.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        lib.nu_set_visibility.restype = ctypes.c_int

        lib.nu_set_camera.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(NuCameraDesc),
        ]
        lib.nu_set_camera.restype = ctypes.c_int

        lib.nu_render.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.nu_render.restype = ctypes.c_int

        lib.nu_fetch_pixels.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.nu_fetch_pixels.restype = ctypes.c_int

        # CUDA interop variant — returns an fd the caller imports into CUDA.
        try:
            lib.nu_fetch_pixels_cuda.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.nu_fetch_pixels_cuda.restype = ctypes.c_int
        except AttributeError:
            pass

        # Async double-buffered render+fetch (optional).
        try:
            lib.nu_render_async.argtypes = [ctypes.c_void_p]
            lib.nu_render_async.restype = ctypes.c_int
            lib.nu_fetch_async.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.nu_fetch_async.restype = ctypes.c_int
        except AttributeError:
            pass

        lib.nu_map_pixels_gpu.argtypes = [ctypes.c_void_p]
        lib.nu_map_pixels_gpu.restype = ctypes.c_void_p

        lib.nu_unmap_pixels_gpu.argtypes = [ctypes.c_void_p]
        lib.nu_unmap_pixels_gpu.restype = None

        lib.nu_save_ppm.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nu_save_ppm.restype = ctypes.c_int

        lib.nu_set_render_size.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.nu_set_render_size.restype = ctypes.c_int

        lib.nu_rt_available.argtypes = [ctypes.c_void_p]
        lib.nu_rt_available.restype = ctypes.c_int

        lib.nu_get_render_size.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.nu_get_render_size.restype = None

        lib.nu_load_environment.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nu_load_environment.restype = ctypes.c_int
        try:
            lib.nu_load_environment_intensity.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]
            lib.nu_load_environment_intensity.restype = ctypes.c_int
        except AttributeError:
            pass  # older lib without the intensity overload

        lib.nu_get_mesh_count.argtypes = [ctypes.c_void_p]
        lib.nu_get_mesh_count.restype = ctypes.c_int

        lib.nu_get_gpu_memory_used.argtypes = [ctypes.c_void_p]
        lib.nu_get_gpu_memory_used.restype = ctypes.c_uint64

        lib.nu_get_cmd_cache_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.nu_get_cmd_cache_stats.restype = None

        lib.nu_get_last_error.argtypes = [ctypes.c_void_p]
        lib.nu_get_last_error.restype = ctypes.c_char_p

        # perf/vk-instrumentation: per-phase GPU timings.
        # Optional — may be missing on older libnusd_renderer_opengl.so builds.
        try:
            lib.nu_get_phase_timings_ms.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(NuPhaseTimings),
            ]
            lib.nu_get_phase_timings_ms.restype = ctypes.c_int
        except AttributeError:
            pass

        # Tiled multi-camera rendering
        lib.nu_render_tiled.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.nu_render_tiled.restype = ctypes.c_int

        lib.nu_fetch_pixels_tiled.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.nu_fetch_pixels_tiled.restype = ctypes.c_int

        lib.nu_set_fast_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.nu_set_fast_mode.restype = None

        # Optional data-type outputs (depth, segmentation, normals).
        # These may not be present in older builds of libnusd_renderer_opengl.so.
        try:
            lib.nu_set_depth_enabled.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_set_depth_enabled.restype = None
            lib.nu_fetch_depth_tiled.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            lib.nu_fetch_depth_tiled.restype = ctypes.c_int
        except AttributeError:
            pass

        try:
            lib.nu_set_segmentation_enabled.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_set_segmentation_enabled.restype = None
            lib.nu_fetch_segmentation_tiled.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            lib.nu_fetch_segmentation_tiled.restype = ctypes.c_int
        except AttributeError:
            pass

        try:
            lib.nu_set_normals_enabled.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_set_normals_enabled.restype = None
            lib.nu_fetch_normals_tiled.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            lib.nu_fetch_normals_tiled.restype = ctypes.c_int
        except AttributeError:
            pass

        # Raycast (LiDAR/radar)
        lib.nu_cast_rays.argtypes = [
            ctypes.c_void_p,                      # renderer
            ctypes.POINTER(ctypes.c_float),       # ray_origins
            ctypes.POINTER(ctypes.c_float),       # ray_directions
            ctypes.c_int,                         # num_rays
            ctypes.c_float,                       # max_distance
            ctypes.POINTER(ctypes.c_float),       # out_distances
            ctypes.POINTER(ctypes.c_float),       # out_normals
            ctypes.POINTER(ctypes.c_float),       # out_hit_positions
        ]
        lib.nu_cast_rays.restype = ctypes.c_int

        lib.nu_set_per_env_layout.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.nu_set_per_env_layout.restype = None

        # Optional (older builds may not ship nu_set_tiled_srgb).
        try:
            lib.nu_set_tiled_srgb.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_set_tiled_srgb.restype = None
        except AttributeError:
            pass

        # Async raycast (optional — may not be in older builds)
        try:
            lib.nu_cast_rays_async.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float,
            ]
            lib.nu_cast_rays_async.restype = ctypes.c_int
            lib.nu_cast_rays_wait.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ]
            lib.nu_cast_rays_wait.restype = ctypes.c_int
        except AttributeError:
            pass

        # Raycast GPU interop
        lib.nu_raycast_get_interop_info.argtypes = [
            ctypes.c_void_p,                      # renderer
            ctypes.c_int,                         # num_rays
            ctypes.c_void_p,                      # NuRaycastInteropInfo*
        ]
        lib.nu_raycast_get_interop_info.restype = ctypes.c_int

        lib.nu_cast_rays_gpu.argtypes = [
            ctypes.c_void_p,                      # renderer
            ctypes.c_int,                         # num_rays
            ctypes.c_float,                       # max_distance
        ]
        lib.nu_cast_rays_gpu.restype = ctypes.c_int

        lib.nu_cast_rays_wait_fence.argtypes = [ctypes.c_void_p]
        lib.nu_cast_rays_wait_fence.restype = ctypes.c_int

        # CUDA-Vulkan interop
        lib.nu_interop_available.argtypes = [ctypes.c_void_p]
        lib.nu_interop_available.restype = ctypes.c_int

        lib.nu_wait_tiled_complete.argtypes = [ctypes.c_void_p]
        lib.nu_wait_tiled_complete.restype = ctypes.c_int

        lib.nu_wait_previous_tiled_complete.argtypes = [ctypes.c_void_p]
        lib.nu_wait_previous_tiled_complete.restype = ctypes.c_int

        lib.nu_get_interop_read_idx.argtypes = [ctypes.c_void_p]
        lib.nu_get_interop_read_idx.restype = ctypes.c_int

        lib.nu_get_interop_prev_idx.argtypes = [ctypes.c_void_p]
        lib.nu_get_interop_prev_idx.restype = ctypes.c_int

        lib.nu_set_skip_staging.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.nu_set_skip_staging.restype = None

        lib.nu_get_cuda_interop_info.argtypes = [
            ctypes.c_void_p,   # renderer
            ctypes.c_void_p,   # NuCudaInteropInfo* out
            ctypes.c_int,      # num_cameras
            ctypes.c_int,      # tile_w
            ctypes.c_int,      # tile_h
        ]
        lib.nu_get_cuda_interop_info.restype = ctypes.c_int

        lib.nu_map_tiled_pixels_raw.argtypes = [
            ctypes.c_void_p,                    # renderer
            ctypes.c_int,                       # num_cameras
            ctypes.c_int,                       # tile_w
            ctypes.c_int,                       # tile_h
            ctypes.POINTER(ctypes.c_int),       # out_w
            ctypes.POINTER(ctypes.c_int),       # out_h
        ]
        lib.nu_map_tiled_pixels_raw.restype = ctypes.c_void_p

        # Async-readback slot variants (optional — older builds may not ship them).
        try:
            lib.nu_map_tiled_pixels_raw_slot.argtypes = [
                ctypes.c_void_p,                    # renderer
                ctypes.c_int,                       # num_cameras
                ctypes.c_int,                       # tile_w
                ctypes.c_int,                       # tile_h
                ctypes.c_int,                       # slot
                ctypes.POINTER(ctypes.c_int),       # out_w
                ctypes.POINTER(ctypes.c_int),       # out_h
            ]
            lib.nu_map_tiled_pixels_raw_slot.restype = ctypes.c_void_p
            lib.nu_get_last_tiled_slot.argtypes = [ctypes.c_void_p]
            lib.nu_get_last_tiled_slot.restype = ctypes.c_int
        except AttributeError:
            pass


class NuRenderer:
    """Python wrapper around libnusd_renderer."""

    def __init__(self, width=1920, height=1080, enable_rt=True, enable_materials=False, visible=False):
        lib = _Lib.get()
        config = NuRendererConfig()
        config.width = width
        config.height = height
        config.gpu_index = 0
        config.enable_rt = 1 if enable_rt else 0
        config.enable_materials = 1 if enable_materials else 0
        config.visible = 1 if visible else 0

        self._handle = lib.nu_renderer_create(ctypes.byref(config))
        if not self._handle:
            raise RuntimeError("nu_renderer_create failed")
        self._lib = lib
        self._width = width
        self._height = height

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.nu_renderer_destroy(self._handle)
            self._handle = None

    def add_mesh(
        self,
        positions: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        texcoords: np.ndarray | None = None,
        transform: np.ndarray | None = None,
        display_color: tuple[float, float, float] = (0.7, 0.7, 0.7),
        name: str | None = None,
    ) -> int:
        positions = np.ascontiguousarray(positions, dtype=np.float32).ravel()
        indices = np.ascontiguousarray(indices, dtype=np.uint32).ravel()

        desc = NuMeshDesc()
        desc.positions = positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        desc.indices = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        desc.nvertices = len(positions) // 3
        desc.nindices = len(indices)

        if normals is not None:
            normals = np.ascontiguousarray(normals, dtype=np.float32).ravel()
            desc.normals = normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if colors is not None:
            colors = np.ascontiguousarray(colors, dtype=np.float32).ravel()
            desc.colors = colors.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if texcoords is not None:
            texcoords = np.ascontiguousarray(texcoords, dtype=np.float32).ravel()
            desc.texcoords = texcoords.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if transform is not None:
            transform = np.ascontiguousarray(transform, dtype=np.float32).ravel()
            desc.transform = transform.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        desc.display_color[0] = display_color[0]
        desc.display_color[1] = display_color[1]
        desc.display_color[2] = display_color[2]

        if name:
            desc.name = name.encode("utf-8")

        mesh_id = self._lib.nu_add_mesh(self._handle, ctypes.byref(desc))
        if mesh_id < 0:
            raise RuntimeError(f"nu_add_mesh failed: {self.last_error}")
        return mesh_id

    def add_mesh_instance(
        self,
        prototype_mesh_id: int,
        transform: np.ndarray | None = None,
    ) -> int:
        """Create an instance sharing an existing mesh's geometry and BLAS.

        The instance gets its own transform but references the prototype's
        vertex/index data, avoiding duplicate BLAS builds. Ideal for instanced
        environments (e.g. IsaacLab) where many shapes share the same mesh.

        Args:
            prototype_mesh_id: mesh_id of the prototype to instance.
            transform: (4, 4) float32 row-major world transform, or None for identity.

        Returns:
            New mesh_id for the instance.
        """
        if transform is not None:
            transform = np.ascontiguousarray(transform, dtype=np.float32).ravel()
            assert len(transform) == 16
            xform_ptr = transform.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            xform_ptr = None

        mesh_id = self._lib.nu_add_mesh_instance(
            self._handle, prototype_mesh_id, xform_ptr
        )
        if mesh_id < 0:
            raise RuntimeError(f"nu_add_mesh_instance failed: {self.last_error}")
        return mesh_id

    def remove_mesh(self, mesh_id: int):
        res = self._lib.nu_remove_mesh(self._handle, mesh_id)
        if res != NU_OK:
            raise RuntimeError(f"nu_remove_mesh failed: {self.last_error}")

    def clear_scene(self):
        self._lib.nu_clear_scene(self._handle)

    def load_usd(self, path: str) -> int:
        n = self._lib.nu_load_usd(self._handle, path.encode("utf-8"))
        if n < 0:
            raise RuntimeError(f"nu_load_usd failed: {self.last_error}")
        return n

    def get_scene_bounds(self):
        """Return ((min_x, min_y, min_z), (max_x, max_y, max_z)) or None."""
        if not hasattr(self._lib, "nu_get_scene_bounds"): return None
        bmin = (ctypes.c_float * 3)()
        bmax = (ctypes.c_float * 3)()
        if self._lib.nu_get_scene_bounds(
            self._handle, ctypes.byref(bmin), ctypes.byref(bmax)
        ) != 0:
            return None
        return ((bmin[0], bmin[1], bmin[2]), (bmax[0], bmax[1], bmax[2]))

    def load_usd_from_handle(self, stage_handle, label: str = "") -> int:
        """Load a scene from an already-open NanousdStage handle. The
        renderer borrows the stage — the caller (typically the pxr_compat
        shim) keeps ownership and must keep it alive while the renderer
        holds it.

        `stage_handle` may be a ctypes pointer, a ctypes-pointer-typed
        value, or a raw int representing the void* address.

        Phase 5: lets one parsed nanousd stage drive both the Python UI
        (hierarchy / properties / variants) and the renderer's scene,
        avoiding a double-parse and a state-skew between two copies of the
        same file."""
        if not hasattr(self._lib, "nu_load_usd_from_handle"):
            raise RuntimeError(
                "nu_load_usd_from_handle is unavailable in this libnusd_renderer build")
        # Coerce any of (ctypes pointer | int address | None) into c_void_p.
        if stage_handle is None:
            raise RuntimeError("stage_handle is None")
        if isinstance(stage_handle, int):
            ptr = ctypes.c_void_p(stage_handle)
        else:
            ptr = ctypes.cast(stage_handle, ctypes.c_void_p)
        n = self._lib.nu_load_usd_from_handle(
            self._handle, ptr, (label or "").encode("utf-8"),
        )
        if n < 0:
            raise RuntimeError(f"nu_load_usd_from_handle failed: {self.last_error}")
        return n

    def load_environment(self, hdr_path: str, intensity: float | None = None):
        """Load an HDR environment. If intensity is None, uses auto-exposure
        (legacy behavior). If intensity > 0, applies it as a sky multiplier
        (matches ovrtx's `dome.color = light.color * intensity` convention).
        """
        if intensity is not None:
            try:
                fn = self._lib.nu_load_environment_intensity
            except AttributeError:
                raise RuntimeError("nu_load_environment_intensity unavailable in this libnusd_renderer build")
            res = fn(self._handle, hdr_path.encode("utf-8"), float(intensity))
            if res != NU_OK:
                raise RuntimeError(f"nu_load_environment_intensity failed: {self.last_error}")
            return
        # Legacy auto-exposure path
        res = self._lib.nu_load_environment(self._handle, hdr_path.encode("utf-8"))
        if res != NU_OK:
            raise RuntimeError(f"nu_load_environment failed: {self.last_error}")

    def build_accel(self):
        """Build or rebuild the RT acceleration structure (BLAS + TLAS).

        Called automatically on first render(), but can be called explicitly
        after adding meshes to ensure the TLAS is built before rendering.
        """
        res = self._lib.nu_build_accel(self._handle)
        if res != NU_OK:
            raise RuntimeError(f"nu_build_accel failed: {self.last_error}")

    def set_current_time(self, time: float) -> bool:
        """Set the time at which subsequent scene loads evaluate USD xforms.

        Pass float('nan') for authored default time. Returns True on success.
        Triggers re-sampling of xformOp:translate.timeSamples (and friends)
        when the next nu_load_usd / nu_load_usd_from_handle runs.
        """
        if not hasattr(self._lib, "nu_set_current_time"):
            return False
        return self._lib.nu_set_current_time(self._handle, float(time)) == NU_OK

    def set_camera(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        fov_degrees: float = 60.0,
        near_clip: float = 0.1,
        far_clip: float = 10000.0,
    ):
        desc = NuCameraDesc()
        desc.eye[0], desc.eye[1], desc.eye[2] = eye
        desc.target[0], desc.target[1], desc.target[2] = target
        desc.fov_degrees = fov_degrees
        desc.near_clip = near_clip
        desc.far_clip = far_clip
        res = self._lib.nu_set_camera(self._handle, 0, ctypes.byref(desc))
        if res != NU_OK:
            raise RuntimeError(f"nu_set_camera failed: {self.last_error}")

    def set_camera_explicit(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        up: tuple[float, float, float],
        fov_degrees: float = 60.0,
        near_clip: float = 0.1,
        far_clip: float = 10000.0,
    ):
        """Set a camera with an explicit up vector, matching Vulkan's API."""
        if not hasattr(self._lib, "nu_set_camera_explicit"):
            raise RuntimeError(
                "nu_set_camera_explicit not available in this libnusd_renderer_opengl.so build."
            )
        eye_a = (ctypes.c_float * 3)(*eye)
        target_a = (ctypes.c_float * 3)(*target)
        up_a = (ctypes.c_float * 3)(*up)
        res = self._lib.nu_set_camera_explicit(
            self._handle, eye_a, target_a, up_a,
            ctypes.c_float(float(fov_degrees)),
            ctypes.c_float(float(near_clip)),
            ctypes.c_float(float(far_clip)),
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_set_camera_explicit failed: {self.last_error}")

    def set_camera_explicit_window(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        up: tuple[float, float, float],
        fov_degrees: float = 60.0,
        near_clip: float = 0.1,
        far_clip: float = 10000.0,
        projection_shift: tuple[float, float] = (0.0, 0.0),
    ):
        """Set an explicit camera with an OVRTX/USD lens-shifted projection."""
        if not hasattr(self._lib, "nu_set_camera_explicit_window"):
            self.set_camera_explicit(eye, target, up, fov_degrees, near_clip, far_clip)
            return
        eye_a = (ctypes.c_float * 3)(*eye)
        target_a = (ctypes.c_float * 3)(*target)
        up_a = (ctypes.c_float * 3)(*up)
        res = self._lib.nu_set_camera_explicit_window(
            self._handle,
            eye_a,
            target_a,
            up_a,
            ctypes.c_float(float(fov_degrees)),
            ctypes.c_float(float(near_clip)),
            ctypes.c_float(float(far_clip)),
            ctypes.c_float(float(projection_shift[0])),
            ctypes.c_float(float(projection_shift[1])),
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_set_camera_explicit_window failed: {self.last_error}")

    def set_exposure(
        self,
        *,
        f_stop: float | None = None,
        responsivity: float | None = None,
        time: float | None = None,
        white_point_scale: float | None = None,
        tonemap_f_number: float | None = None,
        tonemap_cm2_factor: float | None = None,
        auto_exposure_enabled: bool | None = None,
    ):
        """Set OVRTX-style camera exposure and tonemap metadata."""
        if not hasattr(self._lib, "nu_set_exposure"):
            raise RuntimeError("nu_set_exposure not available in this libnusd_renderer_opengl.so build.")
        desc = NuExposureDesc()
        if f_stop is not None:
            desc.flags |= NU_EXPOSURE_HAS_FSTOP
            desc.exposure_f_stop = float(f_stop)
        if responsivity is not None:
            desc.flags |= NU_EXPOSURE_HAS_RESPONSIVITY
            desc.exposure_responsivity = float(responsivity)
        if time is not None:
            desc.flags |= NU_EXPOSURE_HAS_TIME
            desc.exposure_time = float(time)
        if white_point_scale is not None:
            desc.flags |= NU_EXPOSURE_HAS_WHITE_POINT
            desc.white_point_scale = float(white_point_scale)
        if tonemap_f_number is not None:
            desc.flags |= NU_EXPOSURE_HAS_TONEMAP_FNUM
            desc.tonemap_f_number = float(tonemap_f_number)
        if tonemap_cm2_factor is not None:
            desc.flags |= NU_EXPOSURE_HAS_TONEMAP_CM2
            desc.tonemap_cm2_factor = float(tonemap_cm2_factor)
        if auto_exposure_enabled is not None:
            desc.flags |= NU_EXPOSURE_HAS_AUTO_EXPOSURE
            desc.auto_exposure_enabled = 1 if auto_exposure_enabled else 0
        res = self._lib.nu_set_exposure(self._handle, ctypes.byref(desc))
        if res != NU_OK:
            raise RuntimeError(f"nu_set_exposure failed: {self.last_error}")

    def set_transforms(self, mesh_ids, transforms: np.ndarray):
        """``mesh_ids`` may be a Python list or a numpy int32 array; the latter
        avoids per-element ctypes conversion for large live-update batches."""
        if isinstance(mesh_ids, np.ndarray):
            ids_arr = np.ascontiguousarray(mesh_ids, dtype=np.int32)
            ids_ptr = ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            n = ids_arr.shape[0]
        else:
            ids_arr = (ctypes.c_int * len(mesh_ids))(*mesh_ids)
            ids_ptr = ids_arr
            n = len(mesh_ids)
        transforms = np.ascontiguousarray(transforms, dtype=np.float32).ravel()
        res = self._lib.nu_set_transforms(
            self._handle,
            ids_ptr,
            transforms.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_set_transforms failed: {self.last_error}")

    def set_colors(self, mesh_ids, colors: np.ndarray):
        if isinstance(mesh_ids, np.ndarray):
            ids_arr = np.ascontiguousarray(mesh_ids, dtype=np.int32)
            ids_ptr = ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            n = ids_arr.shape[0]
        else:
            ids_arr = (ctypes.c_int * len(mesh_ids))(*mesh_ids)
            ids_ptr = ids_arr
            n = len(mesh_ids)
        colors = np.ascontiguousarray(colors, dtype=np.float32).ravel()
        res = self._lib.nu_set_colors(
            self._handle,
            ids_ptr,
            colors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_set_colors failed: {self.last_error}")

    def set_visibility(self, mesh_ids, visible):
        if isinstance(mesh_ids, np.ndarray):
            ids_arr = np.ascontiguousarray(mesh_ids, dtype=np.int32)
            ids_ptr = ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            n = ids_arr.shape[0]
        else:
            ids_arr = (ctypes.c_int * len(mesh_ids))(*mesh_ids)
            ids_ptr = ids_arr
            n = len(mesh_ids)
        vis_arr = np.ascontiguousarray(np.asarray(visible, dtype=np.int32).reshape(n))
        res = self._lib.nu_set_visibility(
            self._handle,
            ids_ptr,
            vis_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n,
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_set_visibility failed: {self.last_error}")

    def render(self, mode: int = NU_RENDER_RT) -> int:
        res = self._lib.nu_render(self._handle, 0, mode)
        if res != NU_OK:
            raise RuntimeError(f"nu_render failed ({res}): {self.last_error}")
        return res

    def fetch_pixels(self) -> np.ndarray:
        buf = np.zeros((self._height, self._width, 4), dtype=np.uint8)
        res = self._lib.nu_fetch_pixels(
            self._handle,
            buf.ctypes.data_as(ctypes.c_void_p),
            NU_PIXEL_RGBA8,
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_fetch_pixels failed: {self.last_error}")
        return buf

    def fetch_pixels_cuda(self) -> dict:
        """Copy the rendered pixels into a CUDA-importable Vulkan buffer.

        Returns a dict with:
            mem_fd:    POSIX fd (cached by the renderer; do NOT close)
            size:      allocation size in bytes
            width/height: image dimensions
            format:    NU_PIXEL_BGRA8 (raw swapchain order)

        The caller should import the fd ONCE via cuImportExternalMemory +
        cuExternalMemoryGetMappedBuffer, then re-use the imported memory across
        frames.
        """
        fn = getattr(self._lib, "nu_fetch_pixels_cuda", None)
        if fn is None:
            raise RuntimeError("nu_fetch_pixels_cuda not present")
        fd = ctypes.c_int(0)
        size = ctypes.c_uint64(0)
        w = ctypes.c_int(0)
        h = ctypes.c_int(0)
        fmt = ctypes.c_int(0)
        res = fn(self._handle, ctypes.byref(fd), ctypes.byref(size),
                 ctypes.byref(w), ctypes.byref(h), ctypes.byref(fmt))
        if res != NU_OK:
            raise RuntimeError(f"nu_fetch_pixels_cuda failed: {self.last_error}")
        return {"mem_fd": fd.value, "size": size.value,
                "width": w.value, "height": h.value, "format": fmt.value}

    # ---- Async double-buffered render+fetch ----------------------------------

    def render_async(self) -> int:
        """Submit a frame for async rendering — returns immediately."""
        res = self._lib.nu_render_async(self._handle)
        if res != NU_OK:
            raise RuntimeError(f"nu_render_async failed ({res}): {self.last_error}")
        return res

    def fetch_async(self, out_pixels: "np.ndarray | None" = None) -> "np.ndarray":
        """Fetch the previous async-rendered frame's pixels."""
        if out_pixels is None:
            out_pixels = np.zeros((self._height, self._width, 4), dtype=np.uint8)
        res = self._lib.nu_fetch_async(
            self._handle,
            out_pixels.ctypes.data_as(ctypes.c_void_p),
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_fetch_async failed ({res}): {self.last_error}")
        return out_pixels

    def render_tiled(
        self,
        vp_inv_matrices: np.ndarray,
        num_cameras: int,
        tile_w: int,
        tile_h: int,
        mode: int = NU_RENDER_RT,
    ) -> int:
        """Render multiple cameras in a single GPU dispatch.

        Args:
            vp_inv_matrices: (num_cameras, 32) float32 — pairs of (view_inv[16], proj_inv[16]).
            num_cameras: Number of cameras.
            tile_w: Per-camera width in pixels.
            tile_h: Per-camera height in pixels.
            mode: Render mode (NU_RENDER_RT only).

        Returns:
            NU_OK on success.
        """
        vp = np.ascontiguousarray(vp_inv_matrices, dtype=np.float32).ravel()
        res = self._lib.nu_render_tiled(
            self._handle,
            vp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_cameras,
            tile_w,
            tile_h,
            mode,
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_render_tiled failed ({res}): {self.last_error}")
        return res

    def fetch_pixels_tiled(
        self, num_cameras: int, tile_w: int, tile_h: int
    ) -> np.ndarray:
        """Read back tiled render output as (num_cameras, tile_h, tile_w, 4) uint8.

        Args:
            num_cameras: Number of cameras rendered.
            tile_w: Per-camera width.
            tile_h: Per-camera height.

        Returns:
            numpy array of shape (num_cameras, tile_h, tile_w, 4), dtype uint8.
        """
        buf = np.zeros((num_cameras, tile_h, tile_w, 4), dtype=np.uint8)
        res = self._lib.nu_fetch_pixels_tiled(
            self._handle,
            buf.ctypes.data_as(ctypes.c_void_p),
            num_cameras,
            tile_w,
            tile_h,
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_fetch_pixels_tiled failed: {self.last_error}")
        return buf

    def fetch_depth_tiled(
        self, num_cameras: int, tile_w: int, tile_h: int
    ) -> np.ndarray:
        """Read back tiled depth output as (num_cameras, tile_h, tile_w) float32.

        Values: positive = ray hit distance (world units), -1.0 = miss/sky.
        Requires set_depth_enabled(True) before rendering.

        Args:
            num_cameras: Number of cameras rendered.
            tile_w: Per-camera width.
            tile_h: Per-camera height.

        Returns:
            numpy array of shape (num_cameras, tile_h, tile_w), dtype float32.
        """
        buf = np.empty((num_cameras, tile_h, tile_w), dtype=np.float32)
        res = self._lib.nu_fetch_depth_tiled(
            self._handle,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_cameras,
            tile_w,
            tile_h,
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_fetch_depth_tiled failed: {self.last_error}")
        return buf

    def fetch_segmentation_tiled(
        self, num_cameras: int, tile_w: int, tile_h: int
    ) -> np.ndarray:
        """Read back tiled segmentation output as (num_cameras, tile_h, tile_w) uint32.

        Values: 0 = miss/sky/ground, mesh_index+1 = geometry hit.
        Requires set_segmentation_enabled(True) before rendering.

        Args:
            num_cameras: Number of cameras rendered.
            tile_w: Per-camera width.
            tile_h: Per-camera height.

        Returns:
            numpy array of shape (num_cameras, tile_h, tile_w), dtype uint32.
        """
        buf = np.empty((num_cameras, tile_h, tile_w), dtype=np.uint32)
        res = self._lib.nu_fetch_segmentation_tiled(
            self._handle,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            num_cameras,
            tile_w,
            tile_h,
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_fetch_segmentation_tiled failed: {self.last_error}")
        return buf

    def fetch_normals_tiled(
        self, num_cameras: int, tile_w: int, tile_h: int
    ) -> np.ndarray:
        """Read back tiled normals output as (num_cameras, tile_h, tile_w, 3) float32.

        Values: (0,0,0) = miss/sky, (0,1,0) = ground, otherwise surface normal.
        Requires set_normals_enabled(True) before rendering.

        Args:
            num_cameras: Number of cameras rendered.
            tile_w: Per-camera width.
            tile_h: Per-camera height.

        Returns:
            numpy array of shape (num_cameras, tile_h, tile_w, 3), dtype float32.
        """
        buf = np.empty((num_cameras, tile_h, tile_w, 3), dtype=np.float32)
        res = self._lib.nu_fetch_normals_tiled(
            self._handle,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_cameras,
            tile_w,
            tile_h,
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_fetch_normals_tiled failed: {self.last_error}")
        return buf

    def fetch_tiled_raw(
        self, num_cameras: int, tile_w: int, tile_h: int
    ) -> np.ndarray:
        """Map tiled staging buffer as (total_h, total_w, 4) uint8 without de-tiling.

        Returns the raw tiled image — no per-camera extraction. Tiles are packed
        in a ceil(sqrt(N)) column grid. This avoids the O(num_cameras * tile_h)
        memcpy de-tile loop, which dominates at high camera counts.

        Args:
            num_cameras: Number of cameras rendered.
            tile_w: Per-camera width.
            tile_h: Per-camera height.

        Returns:
            numpy array of shape (total_h, total_w, 4), dtype uint8.
            This is a *view* over the staging buffer — valid until next render_tiled().
        """
        out_w = ctypes.c_int(0)
        out_h = ctypes.c_int(0)
        ptr = self._lib.nu_map_tiled_pixels_raw(
            self._handle, num_cameras, tile_w, tile_h,
            ctypes.byref(out_w), ctypes.byref(out_h),
        )
        if not ptr:
            raise RuntimeError(f"nu_map_tiled_pixels_raw failed: {self.last_error}")
        total_w = out_w.value
        total_h = out_h.value
        # Wrap the staging pointer as a numpy array (zero-copy view)
        ArrayType = ctypes.c_uint8 * (total_w * total_h * 4)
        arr = np.ctypeslib.as_array(ArrayType.from_address(ptr))
        return arr.reshape((total_h, total_w, 4))

    def fetch_tiled_raw_slot(
        self, num_cameras: int, tile_w: int, tile_h: int, slot: int
    ) -> np.ndarray:
        """Map a specific double-buffer slot (0 or 1) of the tiled staging.

        Async-readback companion to fetch_tiled_raw. The caller records which
        slot its pending render wrote to (via get_last_tiled_slot, immediately
        after render_tiled) and fetches that slot on the NEXT frame — so the
        GPU-side copy overlaps with CPU work on the previous frame.

        Returns a zero-copy view valid until the slot is reused (normally two
        renders later, once the ping-pong lands back on the same slot).
        """
        out_w = ctypes.c_int(0)
        out_h = ctypes.c_int(0)
        ptr = self._lib.nu_map_tiled_pixels_raw_slot(
            self._handle, num_cameras, tile_w, tile_h, slot,
            ctypes.byref(out_w), ctypes.byref(out_h),
        )
        if not ptr:
            raise RuntimeError(
                f"nu_map_tiled_pixels_raw_slot({slot}) failed: {self.last_error}"
            )
        total_w = out_w.value
        total_h = out_h.value
        ArrayType = ctypes.c_uint8 * (total_w * total_h * 4)
        arr = np.ctypeslib.as_array(ArrayType.from_address(ptr))
        return arr.reshape((total_h, total_w, 4))

    def get_last_tiled_slot(self) -> int:
        """Return the slot (0 or 1) that the most recent render_tiled wrote to.

        Stash this after render_tiled and pass to fetch_tiled_raw_slot on the
        next frame to pick up the async-ready pixels.
        """
        return int(self._lib.nu_get_last_tiled_slot(self._handle))

    def wait_tiled_complete(self):
        """Wait for the most recent render_tiled() to complete on the GPU.

        After this, the interop buffer (exported via get_cuda_interop_info)
        contains valid pixel data and can be safely read by CUDA.
        Cheaper than fetch_tiled_raw — waits on fence only, no staging map.
        """
        res = self._lib.nu_wait_tiled_complete(self._handle)
        if res != NU_OK:
            raise RuntimeError(f"nu_wait_tiled_complete failed: {self.last_error}")

    def wait_previous_tiled_complete(self):
        """Wait for the PREVIOUS frame's render to complete on the GPU.

        For double-buffered overlap: submit frame N with render_tiled(),
        then call this to wait on frame N-1 (typically instant since N-1
        was submitted before N). Returns without error if no previous frame.
        """
        res = self._lib.nu_wait_previous_tiled_complete(self._handle)
        # Don't error on "no previous frame" — first frame has no previous

    def get_interop_prev_idx(self) -> int:
        """Return which interop buffer (0 or 1) has the PREVIOUS frame's data.

        For double-buffered overlap: read this buffer while the current
        frame renders. The first call after init returns an uninitialized buffer.
        """
        return self._lib.nu_get_interop_prev_idx(self._handle)

    @property
    def interop_available(self) -> bool:
        """Whether CUDA-Vulkan interop (zero-copy) is available."""
        return bool(self._lib.nu_interop_available(self._handle))

    def set_skip_staging(self, skip: bool):
        """Skip CPU staging buffer copy when CUDA interop is the only consumer.

        Saves one full-image GPU copy per frame. When enabled,
        fetch_pixels_tiled / fetch_tiled_raw will not work.
        """
        self._lib.nu_set_skip_staging(self._handle, 1 if skip else 0)

    def get_cuda_interop_info(
        self, num_cameras: int, tile_w: int, tile_h: int
    ) -> dict:
        """Get CUDA interop handles for zero-copy access to the tiled render target.

        Returns a dict with:
            mem_fds:     List of two opaque POSIX fds for double-buffered interop buffers
            mem_size:    Allocation size in bytes (same for both buffers)
            image_w/h:   Total tiled image dimensions
            tile_w/h:    Per-camera tile dimensions
            num_cameras: Number of cameras in the grid
            sem_fd:      Opaque POSIX fd for timeline semaphore
            sem_value:   Current timeline value (incremented per render)

        The fds are one-time-use — caller must import them into CUDA or close them.
        """

        class _InteropInfo(ctypes.Structure):
            _fields_ = [
                ("mem_fd", ctypes.c_int * 2),
                ("mem_size", ctypes.c_uint64),
                ("image_w", ctypes.c_uint32),
                ("image_h", ctypes.c_uint32),
                ("tile_w", ctypes.c_uint32),
                ("tile_h", ctypes.c_uint32),
                ("num_cameras", ctypes.c_int),
                ("sem_fd", ctypes.c_int),
                ("sem_value", ctypes.c_uint64),
            ]

        info = _InteropInfo()
        res = self._lib.nu_get_cuda_interop_info(
            self._handle, ctypes.byref(info), num_cameras, tile_w, tile_h
        )
        if res != NU_OK:
            raise RuntimeError(
                f"nu_get_cuda_interop_info failed: {self.last_error}"
            )
        return {
            "mem_fds": [info.mem_fd[0], info.mem_fd[1]],
            "mem_size": info.mem_size,
            "image_w": info.image_w,
            "image_h": info.image_h,
            "tile_w": info.tile_w,
            "tile_h": info.tile_h,
            "num_cameras": info.num_cameras,
            "sem_fd": info.sem_fd,
            "sem_value": info.sem_value,
        }

    def get_interop_read_idx(self) -> int:
        """Return which interop buffer (0 or 1) contains completed frame data.

        Call after render_tiled() to know which double-buffered interop buffer
        CUDA should read from. The other buffer is being written by Vulkan.
        """
        return self._lib.nu_get_interop_read_idx(self._handle)

    def set_fast_mode(self, fast: bool):
        """Enable fast mode for RL sensors: skip shadow rays and use simple
        diffuse lighting. Roughly halves RT dispatch time at the cost of
        visual quality. fast=True enables, fast=False disables (default)."""
        self._lib.nu_set_fast_mode(self._handle, 1 if fast else 0)

    def set_depth_enabled(self, enable: bool):
        """Enable depth output from tiled ray tracing.
        When enabled, the closest-hit and miss shaders write ray T distances
        to a float32 buffer, retrievable via fetch_depth_tiled().
        enable=True enables, enable=False disables (default)."""
        self._lib.nu_set_depth_enabled(self._handle, 1 if enable else 0)

    def set_segmentation_enabled(self, enable: bool):
        """Enable semantic instance segmentation output from tiled ray tracing.
        When enabled, the closest-hit shader writes mesh_index+1 to a uint32
        buffer (0 = miss/sky/ground), retrievable via fetch_segmentation_tiled().
        enable=True enables, enable=False disables (default)."""
        self._lib.nu_set_segmentation_enabled(self._handle, 1 if enable else 0)

    def set_normals_enabled(self, enable: bool):
        """Enable surface normals output from tiled ray tracing.
        When enabled, shaders write (x,y,z) normal vectors to a float32 buffer
        (3 floats per pixel), retrievable via fetch_normals_tiled().
        enable=True enables, enable=False disables (default)."""
        self._lib.nu_set_normals_enabled(self._handle, 1 if enable else 0)

    def cast_rays(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        num_rays: int,
        max_distance: float,
        out_distances: np.ndarray,
        out_normals: np.ndarray,
        out_hit_positions: np.ndarray,
    ):
        """Cast rays against the scene using hardware ray tracing.

        Args:
            origins: [num_rays, 3] float32 — ray origins in world space.
            directions: [num_rays, 3] float32 — normalized ray directions.
            num_rays: Number of rays.
            max_distance: Maximum ray travel distance.
            out_distances: [num_rays] float32 — pre-allocated output distances.
            out_normals: [num_rays, 3] float32 — pre-allocated output normals.
            out_hit_positions: [num_rays, 3] float32 — pre-allocated output hit positions.

        Distances of -1.0 indicate a miss (no intersection within max_distance).
        """
        origins = np.ascontiguousarray(origins, dtype=np.float32).ravel()
        directions = np.ascontiguousarray(directions, dtype=np.float32).ravel()

        res = self._lib.nu_cast_rays(
            self._handle,
            origins.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            directions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_rays,
            max_distance,
            out_distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_hit_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_cast_rays failed ({res}): {self.last_error}")

    def set_per_env_layout(self, enable: bool):
        """Enable per-env SSBO layout: raygen shader writes pixels in
        [env, H, W, 4] contiguous layout instead of 2D tiled layout.
        Eliminates CUDA de-tiling kernel. Requires direct write (skip_staging).
        enable=True on, False off (default)."""
        self._lib.nu_set_per_env_layout(self._handle, 1 if enable else 0)

    def set_tiled_srgb(self, enable: bool):
        """Apply the sRGB transfer function inside the tiled raygen shader.

        When enabled, tiled RGBA8 output is already gamma-encoded, so the
        Python side can skip its CPU LUT pass (~16 ms per 4-camera tick at
        1920x1200). The flag is a no-op on builds without nu_set_tiled_srgb.
        """
        fn = getattr(self._lib, "nu_set_tiled_srgb", None)
        if fn is None:
            return
        fn(self._handle, 1 if enable else 0)

    def cast_rays_async(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        num_rays: int,
        max_distance: float,
    ):
        """Submit async ray cast — returns immediately, GPU works in background.

        Call cast_rays_wait() later to read results. Allows overlapping
        raycast with camera rendering or CPU work.
        """
        origins = np.ascontiguousarray(origins, dtype=np.float32).ravel()
        directions = np.ascontiguousarray(directions, dtype=np.float32).ravel()

        res = self._lib.nu_cast_rays_async(
            self._handle,
            origins.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            directions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_rays,
            max_distance,
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_cast_rays_async failed ({res}): {self.last_error}")

    def cast_rays_wait(
        self,
        out_distances: np.ndarray,
        out_normals: np.ndarray,
        out_hit_positions: np.ndarray,
    ):
        """Wait for async ray cast and read results into pre-allocated arrays."""
        res = self._lib.nu_cast_rays_wait(
            self._handle,
            out_distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_hit_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_cast_rays_wait failed ({res}): {self.last_error}")

    def raycast_get_interop_info(self, num_rays: int) -> dict:
        """Get CUDA-importable FDs for raycast device-local buffers.

        Creates exportable Vulkan buffers and returns opaque file descriptors
        that CUDA can import via cuImportExternalMemory.

        Returns dict with keys:
            input_fd, input_size, output_fd, output_size, max_rays

        Buffer layouts:
            input:  [origins: N*3 floats][directions: N*3 floats]
            output: [distances: N floats][normals: N*3 floats][hit_positions: N*3 floats]
        """

        class NuRaycastInteropInfo(ctypes.Structure):
            _fields_ = [
                ("input_fd", ctypes.c_int),
                ("input_size", ctypes.c_uint64),
                ("output_fd", ctypes.c_int),
                ("output_size", ctypes.c_uint64),
                ("max_rays", ctypes.c_uint32),
            ]

        info = NuRaycastInteropInfo()
        res = self._lib.nu_raycast_get_interop_info(
            self._handle, num_rays, ctypes.byref(info)
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_raycast_get_interop_info failed ({res}): {self.last_error}")
        return {
            "input_fd": info.input_fd,
            "input_size": info.input_size,
            "output_fd": info.output_fd,
            "output_size": info.output_size,
            "max_rays": info.max_rays,
        }

    def cast_rays_gpu(self, num_rays: int, max_distance: float):
        """Dispatch raycast compute on data already in device-local buffer.

        The caller must have written origins+directions into the CUDA-imported
        input buffer before calling this. Call cast_rays_wait_fence() to wait
        for completion (output stays in device-local buffer for CUDA to read).
        """
        res = self._lib.nu_cast_rays_gpu(self._handle, num_rays, max_distance)
        if res != NU_OK:
            raise RuntimeError(f"nu_cast_rays_gpu failed ({res}): {self.last_error}")

    def cast_rays_wait_fence(self):
        """Wait for async raycast to finish (fence only, no staging readback).

        Use this with the GPU interop path where CUDA reads the output buffer
        directly. For the CPU path, use cast_rays_wait() instead.
        """
        res = self._lib.nu_cast_rays_wait_fence(self._handle)
        if res != NU_OK:
            raise RuntimeError(f"nu_cast_rays_wait_fence failed ({res}): {self.last_error}")

    def save_ppm(self, path: str):
        res = self._lib.nu_save_ppm(self._handle, path.encode("utf-8"))
        if res != NU_OK:
            raise RuntimeError(f"nu_save_ppm failed: {self.last_error}")

    def set_render_size(self, width: int, height: int):
        self._width = width
        self._height = height
        res = self._lib.nu_set_render_size(self._handle, width, height)
        if res != NU_OK:
            raise RuntimeError(f"nu_set_render_size failed: {self.last_error}")

    @property
    def rt_available(self) -> bool:
        return bool(self._lib.nu_rt_available(self._handle))

    @property
    def mesh_count(self) -> int:
        """Number of active meshes (including instances) in the scene."""
        return self._lib.nu_get_mesh_count(self._handle)

    def get_mesh_name(self, mesh_id: int) -> str:
        """Return the debug name/path for a renderer mesh, if available."""
        fn = getattr(self._lib, "nu_get_mesh_name", None)
        if fn is None:
            return ""
        n = fn(self._handle, int(mesh_id), None, 0)
        if n < 0:
            return ""
        buf = ctypes.create_string_buffer(n + 1)
        got = fn(self._handle, int(mesh_id), buf, n + 1)
        if got < 0:
            return ""
        return buf.value.decode("utf-8", errors="replace")

    def get_mesh_transform(self, mesh_id: int) -> np.ndarray:
        """Return the renderer-canonical row-major column-vector transform."""
        fn = getattr(self._lib, "nu_get_mesh_transform", None)
        if fn is None:
            raise RuntimeError("nu_get_mesh_transform not available in this libnusd_renderer_opengl build.")
        out = np.zeros(16, dtype=np.float32)
        res = fn(self._handle, int(mesh_id), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if res != NU_OK:
            raise RuntimeError(f"nu_get_mesh_transform failed: {self.last_error}")
        return out.reshape(4, 4)

    @property
    def gpu_memory_used(self) -> int:
        return self._lib.nu_get_gpu_memory_used(self._handle)

    def cmd_cache_stats(self) -> dict:
        """Counts of pre-recorded cmd-buffer cache replays vs full re-records.

        Returned dict has keys: rt_replays, rt_records, tiled_replays, tiled_records.
        For a static scene with a fixed camera/grid, replays should dominate after
        the first few frames.
        """
        rt_r = ctypes.c_uint64(0)
        rt_n = ctypes.c_uint64(0)
        ti_r = ctypes.c_uint64(0)
        ti_n = ctypes.c_uint64(0)
        self._lib.nu_get_cmd_cache_stats(
            self._handle,
            ctypes.byref(rt_r), ctypes.byref(rt_n),
            ctypes.byref(ti_r), ctypes.byref(ti_n),
        )
        return {
            "rt_replays":    rt_r.value,
            "rt_records":    rt_n.value,
            "tiled_replays": ti_r.value,
            "tiled_records": ti_n.value,
        }

    @property
    def last_error(self) -> str:
        err = self._lib.nu_get_last_error(self._handle)
        return err.decode("utf-8") if err else ""

    def get_phase_timings_ms(self) -> "dict[str, float] | None":
        """Return per-phase GPU timings in milliseconds, or None if
        timestamps are unsupported on this device.

        perf/vk-instrumentation: companion to the VK_EXT_debug_utils
        labels emitted around the same regions. Use these for benchmark
        breakdowns; nsys's vulkan_marker_sum report shows the same
        regions as CPU-side ranges."""
        if not hasattr(self._lib, "nu_get_phase_timings_ms"):
            return None
        t = NuPhaseTimings()
        rc = self._lib.nu_get_phase_timings_ms(self._handle, ctypes.byref(t))
        if rc != 0:
            return None
        return {f: getattr(t, f) for f, _ in NuPhaseTimings._fields_}
