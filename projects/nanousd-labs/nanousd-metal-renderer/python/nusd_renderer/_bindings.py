# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ctypes bindings for libnusd_renderer (Metal backend, macOS)."""

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
NU_PIXEL_BGRA8 = 2

# Gaussian constants
NU_GS_PROXY_ICOSAHEDRON = 0
NU_GS_PROXY_AABB = 1
NU_GS_COLOR_LINEAR = 0
NU_GS_COLOR_SRGB = 1
NU_GS_CAMERA_PINHOLE = 0
NU_GS_CAMERA_FISHEYE = 1
NU_GS_CAMERA_EQUIRECT = 2

# Error codes
NU_OK = 0
NU_ERROR = -1
NU_ERROR_NO_RT = -2
NU_ERROR_NO_SCENE = -3
NU_ERROR_BAD_ID = -4
NU_ERROR_UNSUPPORTED = -5

# Backend capability ABI
NU_BACKEND_INFO_VERSION = 1

NU_CAP_RENDER_RASTER = 1 << 0
NU_CAP_RENDER_RAYTRACE = 1 << 1
NU_CAP_RENDER_SHADOW = 1 << 2
NU_CAP_RENDER_TILED_RAYTRACE = 1 << 3
NU_CAP_AOV_COLOR = 1 << 4
NU_CAP_AOV_DEPTH = 1 << 5
NU_CAP_AOV_NORMAL = 1 << 6
NU_CAP_AOV_PRIM_ID = 1 << 7
NU_CAP_LOAD_USD_FILE = 1 << 8
NU_CAP_LOAD_USD_HANDLE = 1 << 9
NU_CAP_LOAD_USD_HANDLE_WITH_DIR = 1 << 10
NU_CAP_SCENE_BOUNDS = 1 << 11
NU_CAP_MESH_PATHS = 1 << 12
NU_CAP_CURVES = 1 << 13
NU_CAP_CAMERAS = 1 << 14
NU_CAP_ANALYTIC_LIGHTS = 1 << 15
NU_CAP_DOME_LIGHT = 1 << 16
NU_CAP_MATERIALS = 1 << 17
NU_CAP_TEXTURES = 1 << 18
NU_CAP_MATERIALX = 1 << 19
NU_CAP_SET_TRANSFORMS = 1 << 20
NU_CAP_SET_COLORS = 1 << 21
NU_CAP_SET_VISIBILITY = 1 << 22
NU_CAP_SET_CURRENT_TIME = 1 << 23
NU_CAP_TIMINGS = 1 << 24
NU_CAP_GPU_MEMORY = 1 << 25
NU_CAP_COMMAND_CACHE_STATS = 1 << 26
NU_CAP_CUDA_INTEROP = 1 << 27
NU_CAP_VISIBLE_WINDOW = 1 << 28
NU_CAP_ASYNC_RENDER = 1 << 29
NU_CAP_MESHLETS = 1 << 30
NU_CAP_GEOMETRY_CACHE = 1 << 31

NU_MESHLET_STATS_VERSION = 1

_CAPABILITY_FIELDS = {
    "render_raster": NU_CAP_RENDER_RASTER,
    "render_raytrace": NU_CAP_RENDER_RAYTRACE,
    "render_shadow": NU_CAP_RENDER_SHADOW,
    "render_tiled_raytrace": NU_CAP_RENDER_TILED_RAYTRACE,
    "aov_color": NU_CAP_AOV_COLOR,
    "aov_depth": NU_CAP_AOV_DEPTH,
    "aov_normal": NU_CAP_AOV_NORMAL,
    "aov_prim_id": NU_CAP_AOV_PRIM_ID,
    "load_usd_file": NU_CAP_LOAD_USD_FILE,
    "load_usd_handle": NU_CAP_LOAD_USD_HANDLE,
    "load_usd_handle_with_dir": NU_CAP_LOAD_USD_HANDLE_WITH_DIR,
    "scene_bounds": NU_CAP_SCENE_BOUNDS,
    "mesh_paths": NU_CAP_MESH_PATHS,
    "curves": NU_CAP_CURVES,
    "cameras": NU_CAP_CAMERAS,
    "analytic_lights": NU_CAP_ANALYTIC_LIGHTS,
    "dome_light": NU_CAP_DOME_LIGHT,
    "materials": NU_CAP_MATERIALS,
    "textures": NU_CAP_TEXTURES,
    "materialx": NU_CAP_MATERIALX,
    "set_transforms": NU_CAP_SET_TRANSFORMS,
    "set_colors": NU_CAP_SET_COLORS,
    "set_visibility": NU_CAP_SET_VISIBILITY,
    "set_current_time": NU_CAP_SET_CURRENT_TIME,
    "timings": NU_CAP_TIMINGS,
    "gpu_memory": NU_CAP_GPU_MEMORY,
    "command_cache_stats": NU_CAP_COMMAND_CACHE_STATS,
    "cuda_interop": NU_CAP_CUDA_INTEROP,
    "visible_window": NU_CAP_VISIBLE_WINDOW,
    "async_render": NU_CAP_ASYNC_RENDER,
    "meshlets": NU_CAP_MESHLETS,
    "geometry_cache": NU_CAP_GEOMETRY_CACHE,
}


def _find_library():
    """Find libnusd_renderer dynamic library (.dylib on macOS, .so elsewhere)."""
    here = Path(__file__).parent
    # macOS uses .dylib; keep .so as fallback so the same module works for
    # mixed-platform installs that ship a Linux build alongside.
    suffixes = (".dylib", ".so") if sys.platform == "darwin" else (".so", ".dylib")
    base_dirs = [here, here.parent.parent / "build", here.parent.parent / "build" / "Release"]
    candidates = [d / f"libnusd_renderer{s}" for d in base_dirs for s in suffixes]

    env_path = os.environ.get("NUSD_RENDERER_LIB")
    if env_path:
        candidates.insert(0, Path(env_path))

    for p in candidates:
        if p.exists():
            return str(p)

    found = ctypes.util.find_library("nusd_renderer")
    if found:
        return found

    raise RuntimeError(
        "libnusd_renderer not found. Set NUSD_RENDERER_LIB or build the library."
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


NU_EXPOSURE_HAS_FSTOP = 1 << 0
NU_EXPOSURE_HAS_RESPONSIVITY = 1 << 1
NU_EXPOSURE_HAS_TIME = 1 << 2
NU_EXPOSURE_HAS_WHITE_POINT = 1 << 3
NU_EXPOSURE_HAS_TONEMAP_FNUM = 1 << 4
NU_EXPOSURE_HAS_TONEMAP_CM2 = 1 << 5
NU_EXPOSURE_HAS_AUTO_EXPOSURE = 1 << 6


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


class NuGsDesc(ctypes.Structure):
    _fields_ = [
        ("positions", ctypes.POINTER(ctypes.c_float)),
        ("scales", ctypes.POINTER(ctypes.c_float)),
        ("orientations", ctypes.POINTER(ctypes.c_float)),
        ("opacities", ctypes.POINTER(ctypes.c_float)),
        ("sh_coefficients", ctypes.POINTER(ctypes.c_float)),
        ("sh_degree", ctypes.c_int),
        ("particle_count", ctypes.c_int),
        ("prim_xform", ctypes.POINTER(ctypes.c_float)),
    ]


class NuBackendInfo(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint64),
        ("backend_name", ctypes.c_char * 32),
        ("backend_version", ctypes.c_char * 32),
        ("renderer_name", ctypes.c_char * 64),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class NuMeshletStats(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("active_meshlets", ctypes.c_uint32),
        ("active_meshlet_indices", ctypes.c_uint32),
        ("active_meshes_with_meshlets", ctypes.c_uint32),
        ("total_meshlets", ctypes.c_uint32),
        ("total_meshlet_indices", ctypes.c_uint32),
        ("meshlet_raster_enabled", ctypes.c_uint32),
        ("max_vertices", ctypes.c_uint32),
        ("max_triangles", ctypes.c_uint32),
        ("cpu_index_bytes", ctypes.c_uint64),
        ("gpu_index_bytes", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 8),
    ]


class NuPhaseTimings(ctypes.Structure):
    _fields_ = [
        ("rt_dispatch_ms", ctypes.c_float),
        ("pixel_readback_ms", ctypes.c_float),
        ("blas_build_ms", ctypes.c_float),
        ("tlas_build_ms", ctypes.c_float),
        ("curve_blas_build_ms", ctypes.c_float),
        ("staging_upload_segs_ms", ctypes.c_float),
        ("staging_upload_aabbs_ms", ctypes.c_float),
        ("staging_upload_colors_ms", ctypes.c_float),
        ("trace_rays_tiled_ms", ctypes.c_float),
        ("deferred_compute_ms", ctypes.c_float),
    ]


def _decode_c_string(value) -> str:
    raw = value if isinstance(value, bytes) else bytes(value)
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _backend_info_to_dict(info: NuBackendInfo) -> dict:
    capabilities = int(info.capabilities)
    result = {
        "version": int(info.version),
        "struct_size": int(info.struct_size),
        "backend_name": _decode_c_string(info.backend_name),
        "backend_version": _decode_c_string(info.backend_version),
        "renderer_name": _decode_c_string(info.renderer_name),
        "capabilities": capabilities,
    }
    result.update(
        {
            name: bool(capabilities & flag)
            for name, flag in _CAPABILITY_FIELDS.items()
        }
    )
    return result


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
        try:
            lib.nu_get_backend_info.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(NuBackendInfo),
            ]
            lib.nu_get_backend_info.restype = ctypes.c_int
        except AttributeError as exc:
            raise RuntimeError(
                "libnusd_renderer does not export nu_get_backend_info; "
                "rebuild the Metal renderer shared library."
            ) from exc

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

        if hasattr(lib, "nu_save_geometry_cache"):
            lib.nu_save_geometry_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.nu_save_geometry_cache.restype = ctypes.c_int
        if hasattr(lib, "nu_load_geometry_cache"):
            lib.nu_load_geometry_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.nu_load_geometry_cache.restype = ctypes.c_int

        if hasattr(lib, "nu_load_usd_from_handle"):
            lib.nu_load_usd_from_handle.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_char_p,
            ]
            lib.nu_load_usd_from_handle.restype = ctypes.c_int

        if hasattr(lib, "nu_load_usd_from_handle_with_dir"):
            lib.nu_load_usd_from_handle_with_dir.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
            ]
            lib.nu_load_usd_from_handle_with_dir.restype = ctypes.c_int

        if hasattr(lib, "nu_extract_deferred"):
            lib.nu_extract_deferred.argtypes = [ctypes.c_void_p]
            lib.nu_extract_deferred.restype = ctypes.c_int

        if hasattr(lib, "nu_extract_deferred_visible"):
            lib.nu_extract_deferred_visible.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.nu_extract_deferred_visible.restype = ctypes.c_int

        if hasattr(lib, "nu_extract_deferred_batched"):
            lib.nu_extract_deferred_batched.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_extract_deferred_batched.restype = ctypes.c_int

        if hasattr(lib, "nu_get_scene_bounds"):
            lib.nu_get_scene_bounds.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float * 3),
                ctypes.POINTER(ctypes.c_float * 3),
            ]
            lib.nu_get_scene_bounds.restype = ctypes.c_int

        lib.nu_build_accel.argtypes = [ctypes.c_void_p]
        lib.nu_build_accel.restype = ctypes.c_int

        if hasattr(lib, "nu_finalize_scene"):
            lib.nu_finalize_scene.argtypes = [ctypes.c_void_p]
            lib.nu_finalize_scene.restype = ctypes.c_int

        # Gaussian splat RT path (optional on older builds).
        try:
            lib.nu_gs_set_particles.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(NuGsDesc),
            ]
            lib.nu_gs_set_particles.restype = ctypes.c_int
            lib.nu_gs_clear_particles.argtypes = [ctypes.c_void_p]
            lib.nu_gs_clear_particles.restype = ctypes.c_int
            lib.nu_gs_set_proxy.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_gs_set_proxy.restype = ctypes.c_int
            lib.nu_gs_set_color_space.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_gs_set_color_space.restype = ctypes.c_int
            lib.nu_gs_set_camera_model.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_gs_set_camera_model.restype = ctypes.c_int
            lib.nu_gs_set_k.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_gs_set_k.restype = ctypes.c_int
            lib.nu_gs_set_max_passes.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_gs_set_max_passes.restype = ctypes.c_int
            lib.nu_gs_set_min_transmittance.argtypes = [
                ctypes.c_void_p,
                ctypes.c_float,
            ]
            lib.nu_gs_set_min_transmittance.restype = ctypes.c_int
            lib.nu_gs_set_iso_opacity_threshold.argtypes = [
                ctypes.c_void_p,
                ctypes.c_float,
            ]
            lib.nu_gs_set_iso_opacity_threshold.restype = ctypes.c_int
            lib.nu_gs_available.argtypes = [ctypes.c_void_p]
            lib.nu_gs_available.restype = ctypes.c_int
            lib.nu_gs_particle_count.argtypes = [ctypes.c_void_p]
            lib.nu_gs_particle_count.restype = ctypes.c_int
            lib.nu_gs_render.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_gs_render.restype = ctypes.c_int
            lib.nu_gs_fetch_depth.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.nu_gs_fetch_depth.restype = ctypes.c_int
            lib.nu_gs_fetch_normal.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.nu_gs_fetch_normal.restype = ctypes.c_int
        except AttributeError:
            pass

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

        if hasattr(lib, "nu_set_mesh_color"):
            lib.nu_set_mesh_color.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.nu_set_mesh_color.restype = ctypes.c_int

        lib.nu_set_visibility.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        lib.nu_set_visibility.restype = ctypes.c_int

        if hasattr(lib, "nu_set_instance_masks"):
            lib.nu_set_instance_masks.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_int,
            ]
            lib.nu_set_instance_masks.restype = ctypes.c_int

        if hasattr(lib, "nu_set_env_partition"):
            lib.nu_set_env_partition.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.nu_set_env_partition.restype = ctypes.c_int

        lib.nu_set_camera.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(NuCameraDesc),
        ]
        lib.nu_set_camera.restype = ctypes.c_int

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

        lib.nu_render.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.nu_render.restype = ctypes.c_int

        lib.nu_fetch_pixels.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.nu_fetch_pixels.restype = ctypes.c_int

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

        if hasattr(lib, "nu_load_environment_intensity"):
            lib.nu_load_environment_intensity.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_float,
            ]
            lib.nu_load_environment_intensity.restype = ctypes.c_int

        if hasattr(lib, "nu_set_dome_color"):
            lib.nu_set_dome_color.argtypes = [
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
            ]
            lib.nu_set_dome_color.restype = ctypes.c_int

        if hasattr(lib, "nu_load_texture"):
            lib.nu_load_texture.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.nu_load_texture.restype = ctypes.c_int

        if hasattr(lib, "nu_set_mesh_texture"):
            lib.nu_set_mesh_texture.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.nu_set_mesh_texture.restype = ctypes.c_int

        lib.nu_get_mesh_count.argtypes = [ctypes.c_void_p]
        lib.nu_get_mesh_count.restype = ctypes.c_int

        if hasattr(lib, "nu_get_mesh_name"):
            lib.nu_get_mesh_name.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            lib.nu_get_mesh_name.restype = ctypes.c_int

        if hasattr(lib, "nu_get_mesh_transform"):
            lib.nu_get_mesh_transform.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.nu_get_mesh_transform.restype = ctypes.c_int

        if hasattr(lib, "nu_get_curve_segment_count"):
            lib.nu_get_curve_segment_count.argtypes = [ctypes.c_void_p]
            lib.nu_get_curve_segment_count.restype = ctypes.c_int

        lib.nu_get_gpu_memory_used.argtypes = [ctypes.c_void_p]
        lib.nu_get_gpu_memory_used.restype = ctypes.c_uint64

        try:
            lib.nu_get_meshlet_stats.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(NuMeshletStats),
            ]
            lib.nu_get_meshlet_stats.restype = ctypes.c_int
        except AttributeError:
            pass

        if hasattr(lib, "nu_get_cmd_cache_stats"):
            lib.nu_get_cmd_cache_stats.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            lib.nu_get_cmd_cache_stats.restype = None

        if hasattr(lib, "nu_get_phase_timings_ms"):
            lib.nu_get_phase_timings_ms.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(NuPhaseTimings),
            ]
            lib.nu_get_phase_timings_ms.restype = ctypes.c_int

        lib.nu_get_last_error.argtypes = [ctypes.c_void_p]
        lib.nu_get_last_error.restype = ctypes.c_char_p

        # Async double-buffered render+fetch (single-cam) — wraps the
        # tiled path internally so steady-state cost drops to roughly
        # max(GPU work, readback memcpy) instead of GPU+readback.
        try:
            lib.nu_render_async.argtypes = [ctypes.c_void_p]
            lib.nu_render_async.restype = ctypes.c_int
            lib.nu_fetch_async.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            lib.nu_fetch_async.restype = ctypes.c_int
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
        # These may not be present in older builds of libnusd_renderer.so.
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

        if hasattr(lib, "nu_set_deferred_shade"):
            lib.nu_set_deferred_shade.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_set_deferred_shade.restype = None
        if hasattr(lib, "nu_set_deferred_debug_mode"):
            lib.nu_set_deferred_debug_mode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            lib.nu_set_deferred_debug_mode.restype = None

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

    def __init__(
        self,
        width=1920,
        height=1080,
        enable_rt=True,
        enable_materials=False,
        visible=False,
    ):
        lib = _Lib.get()
        config = NuRendererConfig()
        config.width = width
        config.height = height
        config.gpu_index = 0
        config.enable_rt = 1 if enable_rt else 0
        config.enable_materials = 1 if enable_materials else 0
        # Metal is always offscreen today; accept visible= for parity with
        # other backend wrappers.
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

    def get_backend_info(self) -> dict:
        """Return backend metadata and capability booleans."""
        info = NuBackendInfo()
        res = self._lib.nu_get_backend_info(self._handle, ctypes.byref(info))
        if res != NU_OK:
            raise RuntimeError(f"nu_get_backend_info failed: {self.last_error}")
        return _backend_info_to_dict(info)

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

    def save_geometry_cache(self, path: str):
        if not hasattr(self._lib, "nu_save_geometry_cache"):
            raise RuntimeError("nu_save_geometry_cache is not exported by this build")
        res = self._lib.nu_save_geometry_cache(self._handle, path.encode("utf-8"))
        if res != NU_OK:
            raise RuntimeError(f"nu_save_geometry_cache failed ({res}): {self.last_error}")

    def load_geometry_cache(self, path: str) -> int:
        if not hasattr(self._lib, "nu_load_geometry_cache"):
            raise RuntimeError("nu_load_geometry_cache is not exported by this build")
        n = self._lib.nu_load_geometry_cache(self._handle, path.encode("utf-8"))
        if n < 0:
            raise RuntimeError(f"nu_load_geometry_cache failed ({n}): {self.last_error}")
        return n

    def extract_deferred(self):
        if not hasattr(self._lib, "nu_extract_deferred"):
            raise RuntimeError("nu_extract_deferred is not exported by this build")
        res = self._lib.nu_extract_deferred(self._handle)
        if res != NU_OK:
            raise RuntimeError(f"nu_extract_deferred failed: {self.last_error}")
        return res

    def extract_deferred_visible(self, vp_matrices) -> int:
        if not hasattr(self._lib, "nu_extract_deferred_visible"):
            raise RuntimeError("nu_extract_deferred_visible is not exported by this build")
        flat = []
        for m in vp_matrices:
            flat.extend(float(x) for x in m)
        if len(flat) % 16 != 0:
            raise ValueError("vp_matrices must contain 16 floats per camera")
        num_cameras = len(flat) // 16
        arr = (ctypes.c_float * len(flat))(*flat) if flat else None
        ptr = ctypes.cast(arr, ctypes.POINTER(ctypes.c_float)) if arr is not None else None
        res = self._lib.nu_extract_deferred_visible(self._handle, ptr, num_cameras)
        if res != NU_OK:
            raise RuntimeError(f"nu_extract_deferred_visible failed: {self.last_error}")
        return res

    def extract_deferred_batched(self, num_batches: int):
        if not hasattr(self._lib, "nu_extract_deferred_batched"):
            raise RuntimeError("nu_extract_deferred_batched is not exported by this build")
        res = self._lib.nu_extract_deferred_batched(self._handle, int(num_batches))
        if res != NU_OK:
            raise RuntimeError(f"nu_extract_deferred_batched failed: {self.last_error}")
        return res

    def load_usd_from_handle(self, stage_handle, label: str = "") -> int:
        """Load a scene from an already-open NanousdStage handle.

        The native ABI takes the borrowed stage handle plus a label/path for
        diagnostics and sidecar material path derivation.
        """
        if stage_handle is None:
            raise RuntimeError("stage_handle is None")
        if isinstance(stage_handle, int):
            ptr = ctypes.c_void_p(stage_handle)
        else:
            ptr = ctypes.cast(stage_handle, ctypes.c_void_p)
        scene_dir = os.path.dirname(os.fspath(label)) if label else ""
        if scene_dir and hasattr(self._lib, "nu_load_usd_from_handle_with_dir"):
            n = self._lib.nu_load_usd_from_handle_with_dir(
                self._handle,
                ptr,
                (label or "").encode("utf-8"),
                scene_dir.encode("utf-8"),
            )
        elif hasattr(self._lib, "nu_load_usd_from_handle"):
            n = self._lib.nu_load_usd_from_handle(
                self._handle, ptr, (label or "").encode("utf-8")
            )
        else:
            raise RuntimeError(
                "nu_load_usd_from_handle not available in this libnusd_renderer build"
            )
        if n < 0:
            raise RuntimeError(f"nu_load_usd_from_handle failed: {self.last_error}")
        return n

    def load_usd_from_handle_with_dir(
        self, stage_handle, label: str = "", scene_dir: str | None = None
    ) -> int:
        """Load a borrowed NanousdStage handle with an explicit scene directory."""
        if stage_handle is None:
            raise RuntimeError("stage_handle is None")
        if isinstance(stage_handle, int):
            ptr = ctypes.c_void_p(stage_handle)
        else:
            ptr = ctypes.cast(stage_handle, ctypes.c_void_p)
        if scene_dir is None and label:
            scene_dir = os.path.dirname(os.fspath(label))
        if scene_dir and hasattr(self._lib, "nu_load_usd_from_handle_with_dir"):
            n = self._lib.nu_load_usd_from_handle_with_dir(
                self._handle,
                ptr,
                (label or "").encode("utf-8"),
                os.fspath(scene_dir).encode("utf-8"),
            )
        elif hasattr(self._lib, "nu_load_usd_from_handle"):
            n = self._lib.nu_load_usd_from_handle(
                self._handle, ptr, (label or "").encode("utf-8")
            )
        else:
            raise RuntimeError(
                "nu_load_usd_from_handle not available in this libnusd_renderer build"
            )
        if n < 0:
            raise RuntimeError(
                f"nu_load_usd_from_handle_with_dir failed: {self.last_error}"
            )
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

    def load_environment(self, hdr_path: str, intensity: float | None = None):
        """Load an HDR equirectangular env map for image-based lighting."""
        if intensity is not None and hasattr(self._lib, "nu_load_environment_intensity"):
            res = self._lib.nu_load_environment_intensity(
                self._handle, hdr_path.encode("utf-8"), float(intensity)
            )
        else:
            res = self._lib.nu_load_environment(self._handle, hdr_path.encode("utf-8"))
        if res != NU_OK:
            raise RuntimeError(f"nu_load_environment failed: {self.last_error}")

    def set_dome_color(self, color, intensity: float = 1.0):
        if not hasattr(self._lib, "nu_set_dome_color"):
            raise RuntimeError("nu_set_dome_color is not exported by this build")
        c = np.ascontiguousarray(color, dtype=np.float32).ravel()
        if c.size != 3:
            raise ValueError("color must contain 3 floats")
        res = self._lib.nu_set_dome_color(
            self._handle,
            float(c[0]),
            float(c[1]),
            float(c[2]),
            float(intensity),
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_set_dome_color failed: {self.last_error}")

    def load_texture(self, pixels: np.ndarray) -> int:
        if not hasattr(self._lib, "nu_load_texture"):
            raise RuntimeError("nu_load_texture is not exported by this build")
        arr = np.ascontiguousarray(pixels, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise ValueError("pixels must have shape (height, width, 4)")
        idx = self._lib.nu_load_texture(
            self._handle,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(arr.shape[1]),
            int(arr.shape[0]),
        )
        if idx < 0:
            raise RuntimeError(f"nu_load_texture failed: {self.last_error}")
        return idx

    def set_mesh_texture(self, mesh_id: int, tex_index: int):
        if not hasattr(self._lib, "nu_set_mesh_texture"):
            raise RuntimeError("nu_set_mesh_texture is not exported by this build")
        res = self._lib.nu_set_mesh_texture(self._handle, int(mesh_id), int(tex_index))
        if res != NU_OK:
            raise RuntimeError(f"nu_set_mesh_texture failed: {self.last_error}")

    def build_accel(self):
        """Build or rebuild the RT acceleration structure (BLAS + TLAS).

        Called automatically on first render(), but can be called explicitly
        after adding meshes to ensure the TLAS is built before rendering.
        """
        res = self._lib.nu_build_accel(self._handle)
        if res != NU_OK:
            raise RuntimeError(f"nu_build_accel failed: {self.last_error}")

    def finalize_scene(self):
        if not hasattr(self._lib, "nu_finalize_scene"):
            raise RuntimeError("nu_finalize_scene is not exported by this build")
        res = self._lib.nu_finalize_scene(self._handle)
        if res != NU_OK:
            raise RuntimeError(f"nu_finalize_scene failed: {self.last_error}")

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
        fov_degrees: float = 45.0,
        near_clip: float = 0.01,
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
        fov_degrees: float = 45.0,
        near_clip: float = 0.01,
        far_clip: float = 10000.0,
    ):
        """Set a camera with arbitrary up vector (e.g. Z-up for IsaacLab)."""
        if not hasattr(self._lib, "nu_set_camera_explicit"):
            raise RuntimeError(
                "nu_set_camera_explicit not available in this libnusd_renderer.so build."
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
            raise RuntimeError("nu_set_exposure not available in this libnusd_renderer.dylib build.")
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
        """``mesh_ids`` may be a Python list or a numpy int32 array."""
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

    def set_mesh_color(self, mesh_id: int, color):
        if not hasattr(self._lib, "nu_set_mesh_color"):
            raise RuntimeError("nu_set_mesh_color is not exported by this build")
        c = np.ascontiguousarray(color, dtype=np.float32).ravel()
        if c.size != 3:
            raise ValueError("color must contain 3 floats")
        res = self._lib.nu_set_mesh_color(
            self._handle,
            int(mesh_id),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_set_mesh_color failed: {self.last_error}")

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

    def set_instance_masks(self, masks):
        if not hasattr(self._lib, "nu_set_instance_masks"):
            raise RuntimeError("nu_set_instance_masks is not exported by this build")
        if masks is None:
            res = self._lib.nu_set_instance_masks(self._handle, None, 0)
        else:
            arr = np.ascontiguousarray(masks, dtype=np.uint8).ravel()
            res = self._lib.nu_set_instance_masks(
                self._handle,
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                int(arr.size),
            )
        if res != NU_OK:
            raise RuntimeError(f"nu_set_instance_masks failed: {self.last_error}")

    def set_env_partition(self, mesh_to_env, num_envs: int):
        if not hasattr(self._lib, "nu_set_env_partition"):
            raise RuntimeError("nu_set_env_partition is not exported by this build")
        if mesh_to_env is None:
            res = self._lib.nu_set_env_partition(self._handle, None, 0, 0)
        else:
            arr = np.ascontiguousarray(mesh_to_env, dtype=np.int32).ravel()
            res = self._lib.nu_set_env_partition(
                self._handle,
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                int(arr.size),
                int(num_envs),
            )
        if res != NU_OK:
            raise RuntimeError(f"nu_set_env_partition failed: {self.last_error}")

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

    # ---- Gaussian splat RT ---------------------------------------------

    def gs_set_particles(
        self,
        positions: np.ndarray,
        scales: np.ndarray,
        orientations: np.ndarray,
        opacities: np.ndarray,
        sh_coefficients: np.ndarray,
        sh_degree: int,
        prim_xform: np.ndarray | None = None,
    ):
        """Upload 3D Gaussian splats for the RT Gaussian path."""
        if not hasattr(self._lib, "nu_gs_set_particles"):
            raise RuntimeError("Gaussian API not exported by this build")

        positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        scales = np.ascontiguousarray(scales, dtype=np.float32).reshape(-1, 3)
        orientations = np.ascontiguousarray(orientations, dtype=np.float32).reshape(-1, 4)
        opacities = np.ascontiguousarray(opacities, dtype=np.float32).reshape(-1)
        sh_coefficients = np.ascontiguousarray(sh_coefficients, dtype=np.float32)

        n = positions.shape[0]
        sh_per = (int(sh_degree) + 1) ** 2
        sh_coefficients = sh_coefficients.reshape(n, sh_per, 3)
        if scales.shape[0] != n or orientations.shape[0] != n or opacities.shape[0] != n:
            raise ValueError("Gaussian arrays must all have the same particle count")

        desc = NuGsDesc()
        desc.positions = positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        desc.scales = scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        desc.orientations = orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        desc.opacities = opacities.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        desc.sh_coefficients = sh_coefficients.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        desc.sh_degree = int(sh_degree)
        desc.particle_count = int(n)

        xform = None
        if prim_xform is not None:
            xform = np.ascontiguousarray(prim_xform, dtype=np.float32).reshape(16)
            desc.prim_xform = xform.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        res = self._lib.nu_gs_set_particles(self._handle, ctypes.byref(desc))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_particles failed ({res}): {self.last_error}")

    def gs_clear_particles(self):
        res = self._lib.nu_gs_clear_particles(self._handle)
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_clear_particles failed ({res}): {self.last_error}")

    def gs_set_proxy(self, proxy: int):
        res = self._lib.nu_gs_set_proxy(self._handle, int(proxy))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_proxy failed ({res}): {self.last_error}")

    def gs_set_k(self, k: int):
        res = self._lib.nu_gs_set_k(self._handle, int(k))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_k failed ({res}): {self.last_error}")

    def gs_set_max_passes(self, max_passes: int):
        res = self._lib.nu_gs_set_max_passes(self._handle, int(max_passes))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_max_passes failed ({res}): {self.last_error}")

    def gs_set_color_space(self, color_space: int):
        res = self._lib.nu_gs_set_color_space(self._handle, int(color_space))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_color_space failed ({res}): {self.last_error}")

    def gs_set_camera_model(self, camera_model: int):
        res = self._lib.nu_gs_set_camera_model(self._handle, int(camera_model))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_camera_model failed ({res}): {self.last_error}")

    def gs_set_min_transmittance(self, eps: float):
        res = self._lib.nu_gs_set_min_transmittance(self._handle, float(eps))
        if res != NU_OK:
            raise RuntimeError(
                f"nu_gs_set_min_transmittance failed ({res}): {self.last_error}"
            )

    def gs_set_iso_opacity_threshold(self, iso: float):
        res = self._lib.nu_gs_set_iso_opacity_threshold(
            self._handle,
            float(iso),
        )
        if res != NU_OK:
            raise RuntimeError(
                f"nu_gs_set_iso_opacity_threshold failed ({res}): {self.last_error}"
            )

    def gs_render(self):
        res = self._lib.nu_gs_render(self._handle, 0)
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_render failed ({res}): {self.last_error}")

    def gs_fetch_depth(self) -> np.ndarray:
        buf = np.empty((self._height, self._width), dtype=np.float32)
        res = self._lib.nu_gs_fetch_depth(
            self._handle,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_fetch_depth failed ({res}): {self.last_error}")
        return buf

    def gs_fetch_normal(self) -> np.ndarray:
        buf = np.empty((self._height, self._width, 3), dtype=np.float32)
        res = self._lib.nu_gs_fetch_normal(
            self._handle,
            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_fetch_normal failed ({res}): {self.last_error}")
        return buf

    @property
    def gs_available(self) -> bool:
        return bool(self._lib.nu_gs_available(self._handle))

    @property
    def gs_particle_count(self) -> int:
        return int(self._lib.nu_gs_particle_count(self._handle))

    # ---- Async double-buffered render+fetch (single-cam) ---------------

    def render_async(self) -> int:
        """Submit a frame for async rendering — returns immediately.

        First call has no previous frame, so fetch_async returns zeros;
        steady-state pairs run as: render_async submits frame N to GPU,
        fetch_async returns frame N-1's pixels. The GPU work and CPU
        readback overlap.
        """
        if not hasattr(self._lib, "nu_render_async"):
            raise RuntimeError("nu_render_async not exported by this build")
        res = self._lib.nu_render_async(self._handle)
        if res != NU_OK:
            raise RuntimeError(f"nu_render_async failed ({res}): {self.last_error}")
        return res

    def fetch_async(self, out_pixels: "np.ndarray | None" = None) -> "np.ndarray":
        """Fetch the previous async-rendered frame's pixels.

        Allocates an (h, w, 4) uint8 RGBA buffer if out_pixels is None.
        Returns zeros on the very first call (no previous frame).
        """
        if not hasattr(self._lib, "nu_fetch_async"):
            raise RuntimeError("nu_fetch_async not exported by this build")
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
        """Return the renderer-canonical row-major world transform."""
        fn = getattr(self._lib, "nu_get_mesh_transform", None)
        if fn is None:
            raise RuntimeError("nu_get_mesh_transform is not exported by this build")
        out = np.zeros(16, dtype=np.float32)
        res = fn(
            self._handle,
            int(mesh_id),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_get_mesh_transform failed: {self.last_error}")
        return out.reshape(4, 4)

    @property
    def curve_segment_count(self) -> int:
        """Number of extracted BasisCurves segments in the scene."""
        fn = getattr(self._lib, "nu_get_curve_segment_count", None)
        return int(fn(self._handle)) if fn is not None else 0

    @property
    def gpu_memory_used(self) -> int:
        return self._lib.nu_get_gpu_memory_used(self._handle)

    def meshlet_stats(self) -> dict:
        fn = getattr(self._lib, "nu_get_meshlet_stats", None)
        if fn is None:
            return {
                "version": 0,
                "active_meshlets": 0,
                "total_meshlets": 0,
                "meshlet_raster_enabled": False,
            }
        stats = NuMeshletStats()
        res = fn(self._handle, ctypes.byref(stats))
        if res != NU_OK:
            raise RuntimeError(f"nu_get_meshlet_stats failed: {self.last_error}")
        return {
            "version": int(stats.version),
            "struct_size": int(stats.struct_size),
            "active_meshlets": int(stats.active_meshlets),
            "active_meshlet_indices": int(stats.active_meshlet_indices),
            "active_meshes_with_meshlets": int(stats.active_meshes_with_meshlets),
            "total_meshlets": int(stats.total_meshlets),
            "total_meshlet_indices": int(stats.total_meshlet_indices),
            "meshlet_raster_enabled": bool(stats.meshlet_raster_enabled),
            "max_vertices": int(stats.max_vertices),
            "max_triangles": int(stats.max_triangles),
            "cpu_index_bytes": int(stats.cpu_index_bytes),
            "gpu_index_bytes": int(stats.gpu_index_bytes),
        }

    def get_cmd_cache_stats(self) -> dict:
        if not hasattr(self._lib, "nu_get_cmd_cache_stats"):
            return {
                "rt_replays": 0,
                "rt_records": 0,
                "tiled_replays": 0,
                "tiled_records": 0,
            }
        rt_replays = ctypes.c_uint64()
        rt_records = ctypes.c_uint64()
        tiled_replays = ctypes.c_uint64()
        tiled_records = ctypes.c_uint64()
        self._lib.nu_get_cmd_cache_stats(
            self._handle,
            ctypes.byref(rt_replays),
            ctypes.byref(rt_records),
            ctypes.byref(tiled_replays),
            ctypes.byref(tiled_records),
        )
        return {
            "rt_replays": int(rt_replays.value),
            "rt_records": int(rt_records.value),
            "tiled_replays": int(tiled_replays.value),
            "tiled_records": int(tiled_records.value),
        }

    def get_phase_timings_ms(self) -> dict:
        if not hasattr(self._lib, "nu_get_phase_timings_ms"):
            return {}
        timings = NuPhaseTimings()
        res = self._lib.nu_get_phase_timings_ms(self._handle, ctypes.byref(timings))
        if res != NU_OK:
            raise RuntimeError(f"nu_get_phase_timings_ms failed: {self.last_error}")
        return {name: float(getattr(timings, name)) for name, _ in NuPhaseTimings._fields_}

    @property
    def last_error(self) -> str:
        err = self._lib.nu_get_last_error(self._handle)
        return err.decode("utf-8") if err else ""
