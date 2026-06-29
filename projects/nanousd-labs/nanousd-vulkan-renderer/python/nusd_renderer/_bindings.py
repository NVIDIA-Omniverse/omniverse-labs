# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ctypes bindings for libnusd_renderer.so"""

import ctypes
import ctypes.util
import os
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

# VK3DGRT — Gaussian splat scene constants. See docs/plans/VK3DGRT_PLAN.md.
NU_GS_PROXY_ICOSAHEDRON = 0
NU_GS_PROXY_AABB = 1

NU_GS_COLOR_LINEAR = 0
NU_GS_COLOR_SRGB = 1

NU_GS_CAMERA_PINHOLE = 0
NU_GS_CAMERA_FISHEYE = 1
NU_GS_CAMERA_EQUIRECT = 2


def _find_library():
    """Find libnusd_renderer.so."""
    # Check alongside this file
    here = Path(__file__).parent
    candidates = [
        here / "libnusd_renderer.so",
        here.parent.parent / "build" / "libnusd_renderer.so",
        here.parent.parent / "build" / "Release" / "libnusd_renderer.so",
    ]
    # Check LD_LIBRARY_PATH and standard locations
    env_path = os.environ.get("NUSD_RENDERER_LIB")
    if env_path:
        candidates.insert(0, Path(env_path))

    for p in candidates:
        if p.exists():
            return str(p)

    # Try system library search
    found = ctypes.util.find_library("nusd_renderer")
    if found:
        return found

    raise RuntimeError(
        "libnusd_renderer.so not found. Set NUSD_RENDERER_LIB or build the library."
    )


# C struct definitions
class NuRendererConfig(ctypes.Structure):
    # MUST mirror the C NuRendererConfig in nusd_renderer.h field-for-field.
    # The trailing `visible` field defaults to 0 (headless) on the C side;
    # if Python's struct is shorter than C's, the renderer reads past the
    # end of the allocation, sees uninitialized memory, and may open a
    # GLFW window beside the embedding Qt UI.
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


class NuGsDesc(ctypes.Structure):
    """VK3DGRT particle descriptor — mirrors NuGsDesc in nusd_renderer.h.

    Per-particle arrays are length particle_count. SH coefficients are
    laid out as [particle_count, (sh_degree+1)**2, 3]. prim_xform is a
    row-major 4x4; pass NULL for identity."""
    _fields_ = [
        ("positions",       ctypes.POINTER(ctypes.c_float)),
        ("scales",          ctypes.POINTER(ctypes.c_float)),
        ("orientations",    ctypes.POINTER(ctypes.c_float)),
        ("opacities",       ctypes.POINTER(ctypes.c_float)),
        ("sh_coefficients", ctypes.POINTER(ctypes.c_float)),
        ("sh_degree",       ctypes.c_int),
        ("particle_count",  ctypes.c_int),
        ("prim_xform",      ctypes.POINTER(ctypes.c_float)),
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
        # Phase C.4 mechanism hunt — populated by gpu_wait_tiled_complete
        # after the tiled-RT readback fence signals.
        ("trace_rays_tiled_ms", ctypes.c_float),
        ("deferred_compute_ms", ctypes.c_float),
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

        lib.nu_set_mesh_color.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.nu_set_mesh_color.restype = ctypes.c_int

        lib.nu_set_instance_masks.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
        ]
        lib.nu_set_instance_masks.restype = ctypes.c_int

        lib.nu_set_env_partition.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.nu_set_env_partition.restype = ctypes.c_int

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

        # nu_load_usd_from_handle_with_dir(NuRenderer*, stage, label, scene_dir)
        if hasattr(lib, "nu_load_usd_from_handle_with_dir"):
            lib.nu_load_usd_from_handle_with_dir.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_char_p, ctypes.c_char_p,
            ]
            lib.nu_load_usd_from_handle_with_dir.restype = ctypes.c_int

        # nu_get_camera(NuRenderer*, int cam_id, NuCameraDesc* out_desc)
        if hasattr(lib, "nu_get_camera"):
            lib.nu_get_camera.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(NuCameraDesc),
            ]
            lib.nu_get_camera.restype = ctypes.c_int

        # nu_get_mesh_name(NuRenderer*, int mesh_id, char* out_buf, int buf_cap)
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
        if hasattr(lib, "nu_get_mesh_material_index"):
            lib.nu_get_mesh_material_index.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
            ]
            lib.nu_get_mesh_material_index.restype = ctypes.c_int
        if hasattr(lib, "nu_get_material_name"):
            lib.nu_get_material_name.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.c_char_p, ctypes.c_int,
            ]
            lib.nu_get_material_name.restype = ctypes.c_int

        lib.nu_build_accel.argtypes = [ctypes.c_void_p]
        lib.nu_build_accel.restype = ctypes.c_int

        # nu_extract_deferred(NuRenderer*) — materialize geometry for a
        # lazy-loaded scene. No-op if no lazy load is pending.
        if hasattr(lib, "nu_extract_deferred"):
            lib.nu_extract_deferred.argtypes = [ctypes.c_void_p]
            lib.nu_extract_deferred.restype = ctypes.c_int

        # nu_extract_deferred_visible(NuRenderer*, const float* vp_matrices,
        # int num_cameras) — frustum-cull variant. Tier 3 step 4.
        if hasattr(lib, "nu_extract_deferred_visible"):
            lib.nu_extract_deferred_visible.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.nu_extract_deferred_visible.restype = ctypes.c_int

        # nu_extract_deferred_batched(NuRenderer*, int num_batches)
        # — batched eager extract. Tier 3 step 5.
        if hasattr(lib, "nu_extract_deferred_batched"):
            lib.nu_extract_deferred_batched.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
            ]
            lib.nu_extract_deferred_batched.restype = ctypes.c_int

        # nu_finalize_scene(NuRenderer*) — drop CPU vertex/index mirror after
        # GPU buffers + BLAS/TLAS are built. After calling, mutation APIs
        # (add_mesh, set_mesh_color, remove_mesh, load_texture) fail fast.
        if hasattr(lib, "nu_finalize_scene"):
            lib.nu_finalize_scene.argtypes = [ctypes.c_void_p]
            lib.nu_finalize_scene.restype = ctypes.c_int

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
        # Optional — may be missing on older libnusd_renderer.so builds.
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

        # nu_set_dome_color: scene's flat DomeLight color (RGB + intensity).
        # May not exist in older builds, so probe defensively.
        try:
            lib.nu_set_dome_color.argtypes = [
                ctypes.c_void_p,
                ctypes.c_float, ctypes.c_float,
                ctypes.c_float, ctypes.c_float,
            ]
            lib.nu_set_dome_color.restype = ctypes.c_int
        except AttributeError:
            pass

        # nu_load_texture / nu_set_mesh_texture (Phase B textures).
        try:
            lib.nu_load_texture.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int, ctypes.c_int,
            ]
            lib.nu_load_texture.restype = ctypes.c_int
        except AttributeError:
            pass
        try:
            lib.nu_set_mesh_texture.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int,
            ]
            lib.nu_set_mesh_texture.restype = ctypes.c_int
        except AttributeError:
            pass

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

        # Phase B deferred-shading toggle. May not exist in older builds.
        try:
            lib.nu_set_deferred_shade.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_set_deferred_shade.restype = None
        except AttributeError:
            pass

        # Phase C.2 deferred debug-mode selector. May not exist in older builds.
        try:
            lib.nu_set_deferred_debug_mode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            lib.nu_set_deferred_debug_mode.restype = None
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
        if hasattr(lib, "nu_cast_rays_with_ids"):
            lib.nu_cast_rays_with_ids.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.nu_cast_rays_with_ids.restype = ctypes.c_int

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

        # Hybrid C — inverted memory ownership.
        # Older libraries don't ship these symbols, so guard with hasattr.
        if hasattr(lib, "nu_set_external_output_buffers"):
            lib.nu_set_external_output_buffers.argtypes = [
                ctypes.c_void_p,                # renderer
                ctypes.POINTER(ctypes.c_int),   # mem_fds[2]
                ctypes.c_uint64,                # mem_size_each
                ctypes.c_int,                   # num_cameras
                ctypes.c_int,                   # tile_w
                ctypes.c_int,                   # tile_h
            ]
            lib.nu_set_external_output_buffers.restype = ctypes.c_int

        if hasattr(lib, "nu_get_external_timeline_semaphore_fd"):
            lib.nu_get_external_timeline_semaphore_fd.argtypes = [
                ctypes.c_void_p,                       # renderer
                ctypes.POINTER(ctypes.c_int),          # out_sem_fd
                ctypes.POINTER(ctypes.c_uint64),       # out_sem_value
            ]
            lib.nu_get_external_timeline_semaphore_fd.restype = ctypes.c_int

        # PR 2: GPU-driven TLAS instance translation
        # Optional — older libraries don't ship these symbols.
        if hasattr(lib, "nu_get_transforms_interop_info"):
            lib.nu_get_transforms_interop_info.argtypes = [
                ctypes.c_void_p,   # renderer
                ctypes.c_int,      # count
                ctypes.c_void_p,   # NuTransformsInteropInfo* out
            ]
            lib.nu_get_transforms_interop_info.restype = ctypes.c_int

            lib.nu_set_transform_layout.argtypes = [
                ctypes.c_void_p,   # renderer
                ctypes.POINTER(ctypes.c_int),  # mesh_ids
                ctypes.c_int,      # count
            ]
            lib.nu_set_transform_layout.restype = ctypes.c_int

            lib.nu_translate_instances_gpu.argtypes = [ctypes.c_void_p]
            lib.nu_translate_instances_gpu.restype = ctypes.c_int

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

        # VK3DGRT — Gaussian splat scene API. Optional (older builds don't
        # ship these symbols). All gated on hasattr so the wrapper works
        # against pre-VK3DGRT libnusd_renderer.so builds.
        if hasattr(lib, "nu_gs_set_particles"):
            lib.nu_gs_set_particles.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(NuGsDesc),
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
                ctypes.c_void_p, ctypes.c_float,
            ]
            lib.nu_gs_set_min_transmittance.restype = ctypes.c_int

            lib.nu_gs_set_iso_opacity_threshold.argtypes = [
                ctypes.c_void_p, ctypes.c_float,
            ]
            lib.nu_gs_set_iso_opacity_threshold.restype = ctypes.c_int

            lib.nu_gs_available.argtypes = [ctypes.c_void_p]
            lib.nu_gs_available.restype = ctypes.c_int

            lib.nu_gs_particle_count.argtypes = [ctypes.c_void_p]
            lib.nu_gs_particle_count.restype = ctypes.c_int

            lib.nu_gs_render.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.nu_gs_render.restype = ctypes.c_int

        if hasattr(lib, "nu_gs_fetch_depth"):
            lib.nu_gs_fetch_depth.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ]
            lib.nu_gs_fetch_depth.restype = ctypes.c_int

        if hasattr(lib, "nu_gs_fetch_normal"):
            lib.nu_gs_fetch_normal.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ]
            lib.nu_gs_fetch_normal.restype = ctypes.c_int


class NuRenderer:
    """Python wrapper around libnusd_renderer."""

    def __init__(self, width=1920, height=1080, enable_rt=True, enable_materials=False,
                 visible=False):
        lib = _Lib.get()
        config = NuRendererConfig()
        config.width = width
        config.height = height
        config.gpu_index = 0
        config.enable_rt = 1 if enable_rt else 0
        config.enable_materials = 1 if enable_materials else 0
        # Default headless — Qt-embedded use never wants the renderer to
        # also pop its own GLFW window. Standalone-viewer paths can pass
        # visible=True if they want the swapchain.
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

    def set_mesh_color(self, mesh_id: int, color):
        """Override the display color of an existing mesh slot. Useful after
        :meth:`add_mesh_instance` to give per-instance colors that differ
        from the prototype.

        Args:
            mesh_id: int, mesh_id from :meth:`add_mesh` / :meth:`add_mesh_instance`.
            color: 3-tuple/list/array of floats in linear RGB [0,1].
        """
        c = np.ascontiguousarray(color, dtype=np.float32).ravel()
        if c.size != 3:
            raise ValueError(f"color must have 3 components, got {c.size}")
        c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        res = self._lib.nu_set_mesh_color(self._handle, mesh_id, c_ptr)
        if res != NU_OK:
            raise RuntimeError(f"nu_set_mesh_color failed: {self.last_error}")

    def set_env_partition(self, mesh_to_env, num_envs):
        """Phase C: declare a per-mesh env partition for per-tile env isolation.

        Args:
            mesh_to_env: 1-D int32 array, mesh_to_env[i] = env_idx (or -1 for
                static globals visible to all rays).
            num_envs: total env count. 0 disables partitioning.
        """
        if num_envs <= 0:
            res = self._lib.nu_set_env_partition(self._handle, None, 0, 0)
        else:
            arr = np.ascontiguousarray(mesh_to_env, dtype=np.int32).ravel()
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            res = self._lib.nu_set_env_partition(self._handle, ptr, len(arr), num_envs)
        if res != NU_OK:
            raise RuntimeError(f"nu_set_env_partition failed: {self.last_error}")

    def set_instance_masks(self, masks):
        """Set TLAS instance.mask bytes for all meshes [0..len(masks)).

        Used for Phase A per-tile env isolation. Each per-tile ray uses a
        cullMask that's AND-ed with each instance's mask byte; only matching
        instances are eligible for the trace.

        Args:
            masks: 1-D array-like of uint8 (or castable). length should equal
                self.nmeshes (or the renderer will clamp).
        """
        m = np.ascontiguousarray(masks, dtype=np.uint8).ravel()
        ptr = m.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        res = self._lib.nu_set_instance_masks(self._handle, ptr, len(m))
        if res != NU_OK:
            raise RuntimeError(f"nu_set_instance_masks failed: {self.last_error}")

    def remove_mesh(self, mesh_id: int):
        res = self._lib.nu_remove_mesh(self._handle, mesh_id)
        if res != NU_OK:
            raise RuntimeError(f"nu_remove_mesh failed: {self.last_error}")

    def clear_scene(self):
        self._lib.nu_clear_scene(self._handle)

    # ---- VK3DGRT (Gaussian splat scene) ----

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
        """Upload particles for the VK3DGRT splat path. Replaces any prior
        particle set. See docs/plans/VK3DGRT_PLAN.md §1 for the input contract
        (matches AOUSD ParticleField3DGaussianSplat 26.03).

        Args:
            positions:       (N, 3) float32 — object-space centers.
            scales:          (N, 3) float32 — linear sigma (post-exp from PLY log-σ).
            orientations:    (N, 4) float32 — wxyz quaternions, normalized.
            opacities:       (N,)   float32 — in [0,1], post-sigmoid.
            sh_coefficients: (N, (sh_degree+1)**2, 3) float32 — DC at index 0.
            sh_degree:       int 0..3.
            prim_xform:      (4, 4) float32 row-major; None = identity.
        """
        if not hasattr(self._lib, "nu_gs_set_particles"):
            raise RuntimeError("libnusd_renderer.so was built without VK3DGRT support")

        positions = np.ascontiguousarray(positions, dtype=np.float32).ravel()
        scales = np.ascontiguousarray(scales, dtype=np.float32).ravel()
        orientations = np.ascontiguousarray(orientations, dtype=np.float32).ravel()
        opacities = np.ascontiguousarray(opacities, dtype=np.float32).ravel()
        sh_coefficients = np.ascontiguousarray(sh_coefficients, dtype=np.float32).ravel()

        N = positions.size // 3
        sh_per = (sh_degree + 1) ** 2
        if scales.size != N * 3:
            raise ValueError(f"scales must be (N, 3), got {scales.size} for N={N}")
        if orientations.size != N * 4:
            raise ValueError(f"orientations must be (N, 4), got {orientations.size} for N={N}")
        if opacities.size != N:
            raise ValueError(f"opacities must be (N,), got {opacities.size} for N={N}")
        if sh_coefficients.size != N * sh_per * 3:
            raise ValueError(
                f"sh_coefficients must be (N, {sh_per}, 3), "
                f"got {sh_coefficients.size} for N={N}, sh_degree={sh_degree}"
            )

        desc = NuGsDesc()
        desc.positions = positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        desc.scales = scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        desc.orientations = orientations.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        desc.opacities = opacities.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        desc.sh_coefficients = sh_coefficients.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        desc.sh_degree = int(sh_degree)
        desc.particle_count = int(N)

        if prim_xform is not None:
            xform = np.ascontiguousarray(prim_xform, dtype=np.float32).ravel()
            if xform.size != 16:
                raise ValueError(f"prim_xform must be (4, 4), got size {xform.size}")
            desc.prim_xform = xform.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self._gs_xform_keepalive = xform  # keep alive across the C call
        else:
            desc.prim_xform = None

        # Hold refs across the C call so the descriptor's pointers stay valid.
        self._gs_keepalive = (
            positions, scales, orientations, opacities, sh_coefficients,
        )
        res = self._lib.nu_gs_set_particles(self._handle, ctypes.byref(desc))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_particles failed: {self.last_error}")

    def gs_clear_particles(self):
        if not hasattr(self._lib, "nu_gs_clear_particles"): return
        res = self._lib.nu_gs_clear_particles(self._handle)
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_clear_particles failed: {self.last_error}")
        self._gs_keepalive = None
        self._gs_xform_keepalive = None

    def gs_set_proxy(self, kind: int):
        """kind: NU_GS_PROXY_ICOSAHEDRON or NU_GS_PROXY_AABB."""
        res = self._lib.nu_gs_set_proxy(self._handle, int(kind))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_proxy failed: {self.last_error}")

    def gs_set_color_space(self, cs: int):
        """cs: NU_GS_COLOR_LINEAR or NU_GS_COLOR_SRGB."""
        res = self._lib.nu_gs_set_color_space(self._handle, int(cs))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_color_space failed: {self.last_error}")

    def gs_set_camera_model(self, cm: int):
        """cm: NU_GS_CAMERA_PINHOLE / FISHEYE / EQUIRECT."""
        res = self._lib.nu_gs_set_camera_model(self._handle, int(cm))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_camera_model failed: {self.last_error}")

    def gs_set_k(self, k: int):
        """K-buffer size: 8, 16, or 32."""
        res = self._lib.nu_gs_set_k(self._handle, int(k))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_k failed: {self.last_error}")

    def gs_set_max_passes(self, max_passes: int):
        res = self._lib.nu_gs_set_max_passes(self._handle, int(max_passes))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_max_passes failed: {self.last_error}")

    def gs_set_min_transmittance(self, eps: float):
        res = self._lib.nu_gs_set_min_transmittance(self._handle, float(eps))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_min_transmittance failed: {self.last_error}")

    def gs_set_iso_opacity_threshold(self, iso: float):
        res = self._lib.nu_gs_set_iso_opacity_threshold(self._handle, float(iso))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_set_iso_opacity_threshold failed: {self.last_error}")

    @property
    def gs_available(self) -> bool:
        if not hasattr(self._lib, "nu_gs_available"): return False
        return bool(self._lib.nu_gs_available(self._handle))

    @property
    def gs_particle_count(self) -> int:
        if not hasattr(self._lib, "nu_gs_particle_count"): return 0
        return int(self._lib.nu_gs_particle_count(self._handle))

    def gs_render(self, cam_id: int = 0):
        """Trace the splat scene and render into the renderer's RT image.
        Pixels are retrievable via fetch_pixels() after this call.
        Iso-opacity surface depth is also written; fetch via
        gs_fetch_depth()."""
        res = self._lib.nu_gs_render(self._handle, int(cam_id))
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_render failed: {self.last_error}")

    def gs_fetch_depth(self) -> np.ndarray:
        """Return per-pixel iso-opacity depth as a (height, width) float32
        numpy array. -1.0 means the ray never accumulated enough opacity
        to cross the threshold (sky / thin scene)."""
        if not hasattr(self._lib, "nu_gs_fetch_depth"):
            raise RuntimeError(
                "nu_gs_fetch_depth not available in this libnusd_renderer.so build."
            )
        out = np.empty((self._height, self._width), dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        res = self._lib.nu_gs_fetch_depth(self._handle, ptr)
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_fetch_depth failed: {self.last_error}")
        return out

    def gs_fetch_normal(self) -> np.ndarray:
        """Return per-pixel iso-opacity surface normal as a (height, width, 3)
        float32 numpy array (world space). (0, 0, 0) on miss / no iso
        crossing. Plan §7."""
        if not hasattr(self._lib, "nu_gs_fetch_normal"):
            raise RuntimeError(
                "nu_gs_fetch_normal not available in this libnusd_renderer.so build."
            )
        out = np.empty((self._height, self._width, 3), dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        res = self._lib.nu_gs_fetch_normal(self._handle, ptr)
        if res != NU_OK:
            raise RuntimeError(f"nu_gs_fetch_normal failed: {self.last_error}")
        return out

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

    def extract_deferred(self):
        """Materialize geometry for a lazy-loaded scene (Tier 3).

        When ``NUSD_LAZY_MESH=1`` is set during ``load_usd`` /
        ``load_usd_from_handle``, the renderer holds a metadata-only
        scene and renders fail with "no geometry to render". This
        method re-runs the eager scene load on the pinned stage,
        re-attaches the resulting scene through the normal path, and
        leaves the renderer with usable geometry.

        Idempotent: no-op if no lazy load is pending. Silently does
        nothing on builds older than the symbol's introduction.
        """
        if not hasattr(self._lib, "nu_extract_deferred"):
            return
        res = self._lib.nu_extract_deferred(self._handle)
        if res != NU_OK:
            raise RuntimeError(f"nu_extract_deferred failed: {self.last_error}")

    def extract_deferred_visible(self, vp_matrices):
        """Materialize only meshes visible to at least one camera frustum.

        ``vp_matrices`` is an iterable of row-major 4x4 view-projection
        matrices (16 floats each). The renderer extracts 6 frustum planes
        per camera via Gribb–Hartmann, tests each lazy-walk-time world
        AABB against the union, and re-runs ``scene_load_from_stage`` with
        a prim-index filter. If no AABBs were snapshotted at lazy attach
        (NUSD_LAZY_AABB=0 was set, or the load was eager), falls back to
        the unfiltered ``extract_deferred`` path.

        Returns nothing. Tier 3 step 4 — see
        ``docs/plans/TIER_3_LAZY_MESH.md``.
        """
        if not hasattr(self._lib, "nu_extract_deferred_visible"):
            return self.extract_deferred()
        # Flatten + ensure C-contiguous float32.
        flat = []
        n = 0
        for vp in vp_matrices:
            if len(vp) != 16:
                raise ValueError("each VP matrix must be 16 floats")
            flat.extend(float(x) for x in vp)
            n += 1
        if n == 0:
            return self.extract_deferred()
        arr = (ctypes.c_float * (n * 16))(*flat)
        res = self._lib.nu_extract_deferred_visible(self._handle, arr, n)
        if res != NU_OK:
            raise RuntimeError(
                f"nu_extract_deferred_visible failed: {self.last_error}")

    def extract_deferred_batched(self, num_batches: int):
        """Materialize geometry in N slices to cap peak RSS.

        Splits the lazy stage's prims into ``num_batches`` round-robin
        slices, runs ``scene_load_from_stage_filtered`` + attach per
        slice, with each slice's arena released before the next starts.
        Intermediate slices skip ``NUSD_AUTO_FINALIZE`` so cpu_vertices
        can accumulate; the last slice handles finalize.

        ``num_batches`` ≤ 1 falls back to the single-shot
        ``extract_deferred`` path. Tier 3 step 5 — see
        ``docs/plans/TIER_3_LAZY_MESH.md``.
        """
        if not hasattr(self._lib, "nu_extract_deferred_batched"):
            return self.extract_deferred()
        res = self._lib.nu_extract_deferred_batched(self._handle, int(num_batches))
        if res != NU_OK:
            raise RuntimeError(
                f"nu_extract_deferred_batched failed: {self.last_error}")

    def finalize_scene(self):
        """Drop the renderer's CPU vertex/index mirror after GPU upload.

        Call after load_usd/load_usd_from_handle completes and before
        entering the render loop. Frees the multi-GB cpu_vertices/cpu_indices
        host buffer that exists only as an upload-staging source. Mutation
        APIs (add_mesh, set_mesh_color, remove_mesh, load_texture) will
        fail after this — to mutate again, call clear_scene() and reload.

        Silently no-ops on builds older than the symbol's introduction.
        """
        if not hasattr(self._lib, "nu_finalize_scene"):
            return
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
            raise RuntimeError("nu_set_exposure not available in this libnusd_renderer.so build.")
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
        is dramatically cheaper (no per-element Python→ctypes conversion at
        200K-instance scale). ``transforms`` must be a contiguous float32
        array of shape (N, 16) or (N*16,)."""
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

    # ---- Hybrid C: inverted memory ownership ------------------------------------

    def set_external_output_buffers(
        self,
        mem_fds: list[int],
        mem_size_each: int,
        num_cameras: int,
        tile_w: int,
        tile_h: int,
    ) -> None:
        """Provide caller-allocated CUDA memory as the tiled output buffer (Hybrid C).

        See ``nu_set_external_output_buffers`` in nusd_renderer.h for the full
        contract. ``mem_fds`` is a list of two POSIX fds returned by
        cuMemExportToShareableHandle; the renderer takes ownership on success
        (vkAllocateMemory consumes them — caller must NOT close on success).

        On failure the caller still owns the fds and should close them.
        """
        if not hasattr(self._lib, "nu_set_external_output_buffers"):
            raise RuntimeError(
                "nu_set_external_output_buffers not present in this libnusd_renderer build"
            )
        if len(mem_fds) != 2:
            raise ValueError("mem_fds must be a list of two ints")

        fds = (ctypes.c_int * 2)(int(mem_fds[0]), int(mem_fds[1]))
        res = self._lib.nu_set_external_output_buffers(
            self._handle, fds, ctypes.c_uint64(int(mem_size_each)),
            int(num_cameras), int(tile_w), int(tile_h),
        )
        if res != NU_OK:
            raise RuntimeError(
                f"nu_set_external_output_buffers failed: {self.last_error}"
            )

    def get_external_timeline_semaphore_fd(self) -> dict:
        """Export the renderer's timeline semaphore for caller-side cuImportExternalSemaphore.

        Returns dict with ``sem_fd`` (one-time-use POSIX fd) and ``sem_value``
        (current timeline value). Companion to set_external_output_buffers.
        """
        if not hasattr(self._lib, "nu_get_external_timeline_semaphore_fd"):
            raise RuntimeError(
                "nu_get_external_timeline_semaphore_fd not present in this libnusd_renderer build"
            )
        sem_fd = ctypes.c_int(-1)
        sem_value = ctypes.c_uint64(0)
        res = self._lib.nu_get_external_timeline_semaphore_fd(
            self._handle, ctypes.byref(sem_fd), ctypes.byref(sem_value),
        )
        if res != NU_OK:
            raise RuntimeError(
                f"nu_get_external_timeline_semaphore_fd failed: {self.last_error}"
            )
        return {"sem_fd": sem_fd.value, "sem_value": sem_value.value}

    # ---- PR 2: GPU-driven TLAS instance translation -----------------------------

    def get_transforms_interop_info(self, count: int) -> dict:
        """Allocate the per-shape transforms buffer and return CUDA-importable info.

        The renderer creates an exportable Vulkan storage buffer sized for
        ``count`` row-major 4x4 transforms (count * 64 bytes). The caller imports
        the returned ``mem_fd`` via ``cuImportExternalMemory`` +
        ``cuExternalMemoryGetMappedBuffer`` and wraps the resulting CUDA device
        pointer as a warp array so kernels can write directly into the buffer.

        Args:
            count: Number of 4x4 transforms (typically ``n_valid``).

        Returns:
            dict with keys:
                mem_fd: POSIX fd (single-use; consume via cuImportExternalMemory).
                mem_size: Logical size in bytes (count * 64).
                count: Number of transforms.
        """
        if not hasattr(self._lib, "nu_get_transforms_interop_info"):
            raise RuntimeError(
                "nu_get_transforms_interop_info not present in this libnusd_renderer build"
            )

        class _Info(ctypes.Structure):
            _fields_ = [
                ("mem_fd", ctypes.c_int),
                ("mem_size", ctypes.c_uint64),
                ("count", ctypes.c_int),
            ]

        info = _Info()
        res = self._lib.nu_get_transforms_interop_info(
            self._handle, int(count), ctypes.byref(info)
        )
        if res != NU_OK:
            raise RuntimeError(
                f"nu_get_transforms_interop_info failed: {self.last_error}"
            )
        return {"mem_fd": info.mem_fd, "mem_size": info.mem_size, "count": info.count}

    def set_transform_layout(self, mesh_ids):
        """Upload the (gid -> tlas_instance_idx) lookup table for the GPU compute path.

        ``mesh_ids[i]`` is the renderer mesh_id (returned by
        :meth:`add_mesh` / :meth:`add_mesh_instance`) of the shape that warp
        thread ``i`` writes. The renderer inverts ``instance_custom`` internally
        to obtain the TLAS-slot index for each gid. ``mesh_ids[i] < 0`` skips
        that slot.

        Args:
            mesh_ids: Sequence of mesh_ids; numpy int32 array preferred for
                large counts (avoids per-element ctypes conversion).
        """
        if not hasattr(self._lib, "nu_set_transform_layout"):
            raise RuntimeError(
                "nu_set_transform_layout not present in this libnusd_renderer build"
            )
        if isinstance(mesh_ids, np.ndarray):
            ids_arr = np.ascontiguousarray(mesh_ids, dtype=np.int32)
            ids_ptr = ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            n = ids_arr.shape[0]
        else:
            ids_arr = (ctypes.c_int * len(mesh_ids))(*mesh_ids)
            ids_ptr = ids_arr
            n = len(mesh_ids)
        res = self._lib.nu_set_transform_layout(self._handle, ids_ptr, n)
        if res != NU_OK:
            raise RuntimeError(
                f"nu_set_transform_layout failed: {self.last_error}"
            )

    def translate_instances_gpu(self):
        """Arm the next render_tiled() to take the GPU-driven TLAS update path.

        Replaces :meth:`set_transforms` for the GPU-driven path. The caller must
        have:
            1. Allocated the transforms buffer via :meth:`get_transforms_interop_info`.
            2. Imported it into CUDA and configured a warp kernel to write into it.
            3. Uploaded the layout via :meth:`set_transform_layout`.
            4. Invoked the warp kernel to populate the buffer.
            5. Synchronized the CUDA stream (e.g. ``wp.synchronize_device()``)
               before this call — there is no automatic CUDA→Vulkan semaphore wait.

        On the next :meth:`render_tiled` the renderer dispatches the compute
        shader (which writes ``VkAccelerationStructureInstanceKHR`` records)
        and the TLAS update inline, no host data involved.
        """
        if not hasattr(self._lib, "nu_translate_instances_gpu"):
            raise RuntimeError(
                "nu_translate_instances_gpu not present in this libnusd_renderer build"
            )
        res = self._lib.nu_translate_instances_gpu(self._handle)
        if res != NU_OK:
            raise RuntimeError(
                f"nu_translate_instances_gpu failed: {self.last_error}"
            )

    def set_fast_mode(self, fast: bool):
        """Enable fast mode for RL sensors: skip shadow rays and use simple
        diffuse lighting. Roughly halves RT dispatch time at the cost of
        visual quality. fast=True enables, fast=False disables (default)."""
        self._lib.nu_set_fast_mode(self._handle, 1 if fast else 0)

    def load_texture(self, pixels: np.ndarray, width: int, height: int) -> int:
        """Upload a single RGBA8 texture and return its index (>=0).

        The renderer copies the pixel buffer; the caller may free / reuse
        the source array after the call. The texture is treated as sRGB
        color data by the renderer (hardware sRGB → linear at sample).
        Use the returned index with :meth:`set_mesh_texture` to bind the
        texture to a specific mesh.

        Args:
            pixels: 1-D or N-D ``uint8`` ndarray with ``width * height * 4``
                elements (RGBA8 layout).
            width: Texture width in pixels.
            height: Texture height in pixels.

        Returns:
            int: New texture index (``>= 0``), or raises on failure.
        """
        if not hasattr(self._lib, "nu_load_texture"):
            raise RuntimeError("nu_load_texture unavailable in this libnusd_renderer build")
        arr = np.ascontiguousarray(pixels, dtype=np.uint8).reshape(-1)
        expected = width * height * 4
        if arr.size != expected:
            raise ValueError(
                f"load_texture: expected {expected} bytes ({width}x{height}x4), got {arr.size}"
            )
        idx = self._lib.nu_load_texture(
            self._handle,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(width), int(height),
        )
        if idx < 0:
            raise RuntimeError(f"nu_load_texture failed: {self.last_error}")
        return int(idx)

    def set_mesh_texture(self, mesh_id: int, tex_index: int):
        """Bind a previously-loaded texture index to a mesh.

        Args:
            mesh_id: Mesh ID returned by :meth:`add_mesh` /
                :meth:`add_mesh_instance`.
            tex_index: Texture index returned by :meth:`load_texture`. Pass
                ``-1`` to clear (revert to flat per-mesh color).
        """
        if not hasattr(self._lib, "nu_set_mesh_texture"):
            return
        res = self._lib.nu_set_mesh_texture(self._handle, int(mesh_id), int(tex_index))
        if res != NU_OK:
            raise RuntimeError(f"nu_set_mesh_texture failed: {self.last_error}")

    def set_dome_color(self, r: float, g: float, b: float, intensity: float = 1.0):
        """Set the scene's flat (no-IBL) DomeLight color used by the
        fast_mode rmiss sky and rchit hemispheric ambient.

        Defaults inside the renderer are near-white at intensity 1.0
        (matches Newton's 0xEEEEEE flat clear color). For RL training where
        the scene authors a USD DomeLight, the IsaacLab adapter should map
        that light's `inputs:color` and `inputs:intensity` through this
        setter. The renderer applies no additional auto-exposure to the
        fast_mode path, so callers should pre-normalize over-bright
        intensities (e.g. 2000) into [0,1] (e.g. divide by 2000).

        Args:
            r: Linear-RGB red in [0,1].
            g: Linear-RGB green in [0,1].
            b: Linear-RGB blue in [0,1].
            intensity: Scalar multiplier applied to (r,g,b) inside the
                shader. Defaults to 1.0.
        """
        if not hasattr(self._lib, "nu_set_dome_color"):
            return
        res = self._lib.nu_set_dome_color(
            self._handle,
            ctypes.c_float(float(r)),
            ctypes.c_float(float(g)),
            ctypes.c_float(float(b)),
            ctypes.c_float(float(intensity)),
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_set_dome_color failed: {self.last_error}")

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

    def set_deferred_shade(self, enable: bool):
        """Phase B deferred-shading toggle.

        When enabled, the closest-hit/miss shaders write a per-pixel G-buffer
        (binding 17) inside the fast_mode path, and a follow-on compute
        dispatch reads it to produce the final pixels. Phase B writes
        flat-shaded per-mesh display color only (a "color-ID" debug viz).
        Phase C/D will add full materials, IBL, and lighting.
        Also auto-enabled by NU_DEFERRED_SHADE=1 at renderer creation time."""
        if hasattr(self._lib, "nu_set_deferred_shade"):
            self._lib.nu_set_deferred_shade(self._handle, 1 if enable else 0)

    def set_deferred_debug_mode(self, mode: int):
        """Phase C.2 deferred-shading debug visualization selector.

        Modes:
          0 = base color (Phase C.1 textured albedo, byte-identical default)
          1 = world-space shading normal as RGB (post normal-map TBN)
          2 = PBR-packed: metallic.r, roughness.g, ao.b
          3 = full lit (placeholder; falls back to mode 0 until Phase C.4)
        Only meaningful when set_deferred_shade(True) is also active."""
        if hasattr(self._lib, "nu_set_deferred_debug_mode"):
            self._lib.nu_set_deferred_debug_mode(self._handle, int(mode) & 0xFFFFFFFF)

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

    def cast_rays_with_ids(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        num_rays: int,
        max_distance: float,
        out_distances: np.ndarray,
        out_mesh_ids: np.ndarray,
        out_normals: np.ndarray,
        out_hit_positions: np.ndarray,
    ):
        """Cast rays and return renderer mesh IDs in addition to hit data."""
        fn = getattr(self._lib, "nu_cast_rays_with_ids", None)
        if fn is None:
            raise RuntimeError("nu_cast_rays_with_ids not available")
        origins = np.ascontiguousarray(origins, dtype=np.float32).ravel()
        directions = np.ascontiguousarray(directions, dtype=np.float32).ravel()
        res = fn(
            self._handle,
            origins.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            directions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_rays,
            max_distance,
            out_distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_mesh_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_hit_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if res != NU_OK:
            raise RuntimeError(f"nu_cast_rays_with_ids failed ({res}): {self.last_error}")

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
            raise RuntimeError("nu_get_mesh_transform not available in this libnusd_renderer build.")
        out = np.zeros(16, dtype=np.float32)
        res = fn(self._handle, int(mesh_id), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if res != NU_OK:
            raise RuntimeError(f"nu_get_mesh_transform failed: {self.last_error}")
        return out.reshape(4, 4)

    def get_mesh_material_index(self, mesh_id: int) -> int:
        """Return the material index for a renderer mesh, or -1 if none."""
        fn = getattr(self._lib, "nu_get_mesh_material_index", None)
        if fn is None:
            return -1
        return int(fn(self._handle, int(mesh_id)))

    def get_material_name(self, material_index: int) -> str:
        """Return the debug name/path for a material, if available."""
        fn = getattr(self._lib, "nu_get_material_name", None)
        if fn is None or material_index < 0:
            return ""
        n = fn(self._handle, int(material_index), None, 0)
        if n < 0:
            return ""
        buf = ctypes.create_string_buffer(n + 1)
        got = fn(self._handle, int(material_index), buf, n + 1)
        if got < 0:
            return ""
        return buf.value.decode("utf-8", errors="replace")

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
