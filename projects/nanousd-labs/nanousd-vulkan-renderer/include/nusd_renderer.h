// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * nusd_renderer.h — Headless Vulkan renderer with hardware ray tracing.
 *
 * Renders scenes built from mesh arrays or loaded from USD files (via nanousd).
 * Designed as an open-source alternative to the ovrtx rendering API,
 * targeting IsaacLab/Newton as a render backend.
 *
 * All functions are thread-safe per NuRenderer instance.
 */

#ifndef NUSD_RENDERER_H
#define NUSD_RENDERER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Opaque handle ---- */

typedef struct NuRenderer NuRenderer;

/* ---- Enums ---- */

typedef enum {
    NU_RENDER_RASTER   = 0,   /* Three-point lit raster */
    NU_RENDER_SHADOW   = 1,   /* Raster + ray query shadows */
    NU_RENDER_RT       = 2,   /* Full hardware ray tracing */
} NuRenderMode;

typedef enum {
    NU_OK              =  0,
    NU_ERROR           = -1,
    NU_ERROR_NO_RT     = -2,  /* Hardware RT not available */
    NU_ERROR_NO_SCENE  = -3,  /* No meshes loaded */
    NU_ERROR_BAD_ID    = -4,  /* Invalid mesh_id or cam_id */
    NU_ERROR_UNSUPPORTED = -5, /* Entry point is intentionally unsupported */
} NuResult;

/* ---- Backend capability ABI ---- */

#define NU_BACKEND_INFO_VERSION 1u

#define NU_CAP_RENDER_RASTER              (UINT64_C(1) << 0)
#define NU_CAP_RENDER_RAYTRACE            (UINT64_C(1) << 1)
#define NU_CAP_RENDER_SHADOW              (UINT64_C(1) << 2)
#define NU_CAP_RENDER_TILED_RAYTRACE      (UINT64_C(1) << 3)
#define NU_CAP_AOV_COLOR                  (UINT64_C(1) << 4)
#define NU_CAP_AOV_DEPTH                  (UINT64_C(1) << 5)
#define NU_CAP_AOV_NORMAL                 (UINT64_C(1) << 6)
#define NU_CAP_AOV_PRIM_ID                (UINT64_C(1) << 7)
#define NU_CAP_LOAD_USD_FILE              (UINT64_C(1) << 8)
#define NU_CAP_LOAD_USD_HANDLE            (UINT64_C(1) << 9)
#define NU_CAP_LOAD_USD_HANDLE_WITH_DIR   (UINT64_C(1) << 10)
#define NU_CAP_SCENE_BOUNDS               (UINT64_C(1) << 11)
#define NU_CAP_MESH_PATHS                 (UINT64_C(1) << 12)
#define NU_CAP_CURVES                     (UINT64_C(1) << 13)
#define NU_CAP_CAMERAS                    (UINT64_C(1) << 14)
#define NU_CAP_ANALYTIC_LIGHTS            (UINT64_C(1) << 15)
#define NU_CAP_DOME_LIGHT                 (UINT64_C(1) << 16)
#define NU_CAP_MATERIALS                  (UINT64_C(1) << 17)
#define NU_CAP_TEXTURES                   (UINT64_C(1) << 18)
#define NU_CAP_MATERIALX                  (UINT64_C(1) << 19)
#define NU_CAP_SET_TRANSFORMS             (UINT64_C(1) << 20)
#define NU_CAP_SET_COLORS                 (UINT64_C(1) << 21)
#define NU_CAP_SET_VISIBILITY             (UINT64_C(1) << 22)
#define NU_CAP_SET_CURRENT_TIME           (UINT64_C(1) << 23)
#define NU_CAP_TIMINGS                    (UINT64_C(1) << 24)
#define NU_CAP_GPU_MEMORY                 (UINT64_C(1) << 25)
#define NU_CAP_COMMAND_CACHE_STATS        (UINT64_C(1) << 26)
#define NU_CAP_CUDA_INTEROP               (UINT64_C(1) << 27)
#define NU_CAP_VISIBLE_WINDOW             (UINT64_C(1) << 28)
#define NU_CAP_ASYNC_RENDER               (UINT64_C(1) << 29)
#define NU_CAP_MESHLETS                   (UINT64_C(1) << 30)
#define NU_CAP_GEOMETRY_CACHE             (UINT64_C(1) << 31)

typedef struct {
    uint32_t version;
    uint32_t struct_size;
    uint64_t capabilities;
    char backend_name[32];
    char backend_version[32];
    char renderer_name[64];
    uint64_t reserved[8];
} NuBackendInfo;

typedef enum {
    NU_PIXEL_RGBA8     = 0,   /* 4 bytes per pixel, uint8 */
    NU_PIXEL_RGBAF32   = 1,   /* 16 bytes per pixel, float32 */
    NU_PIXEL_BGRA8     = 2,   /* 4 bytes per pixel, uint8 — raw swapchain order
                               * (used by nu_fetch_pixels_cuda and
                               * nu_fetch_pixels(NU_PIXEL_BGRA8); no CPU
                               * BGRA→RGBA swizzle: pure memcpy out of the
                               * device-cached staging buffer) */
} NuPixelFormat;

/* ---- Descriptors ---- */

typedef struct {
    int      width;           /* Render target width (default: 1920) */
    int      height;          /* Render target height (default: 1080) */
    int      gpu_index;       /* CUDA/Vulkan device index (default: 0) */
    int      enable_rt;       /* 1 = build acceleration structures (default: 1) */
    int      enable_materials;/* 1 = enable MaterialX PBR pipeline (default: 0) */
    int      visible;         /* 1 = open a visible window + present to swapchain
                               *     (interactive viewer mode); 0 = hidden +
                               *     headless (default, for offscreen/compute use). */
} NuRendererConfig;

typedef struct {
    /* Vertex data — flat arrays */
    const float*    positions;    /* float[nvertices * 3] — required */
    const float*    normals;      /* float[nvertices * 3] — optional, NULL = auto-compute */
    const float*    colors;       /* float[nvertices * 3] — optional per-vertex RGB */
    const float*    texcoords;    /* float[nvertices * 2] — optional UVs */
    const uint32_t* indices;      /* uint32_t[nindices]   — required, triangles */
    int             nvertices;
    int             nindices;

    /* Initial transform (row-major 4x4 float, NULL = identity) */
    const float*    transform;

    /* Fallback display color if no per-vertex colors (RGB 0-1) */
    float           display_color[3];

    /* Material index into the renderer's material collection, -1 = none */
    int             material_index;

    /* Name for debugging (optional, copied internally) */
    const char*     name;
} NuMeshDesc;

typedef struct {
    float eye[3];
    float target[3];
    float fov_degrees;         /* Vertical FOV (default: 60) */
    float near_clip;           /* Default: 0.1 */
    float far_clip;            /* Default: 10000.0 */
} NuCameraDesc;

enum {
    NU_EXPOSURE_HAS_FSTOP          = 1u << 0,
    NU_EXPOSURE_HAS_RESPONSIVITY  = 1u << 1,
    NU_EXPOSURE_HAS_TIME          = 1u << 2,
    NU_EXPOSURE_HAS_WHITE_POINT   = 1u << 3,
    NU_EXPOSURE_HAS_TONEMAP_FNUM  = 1u << 4,
    NU_EXPOSURE_HAS_TONEMAP_CM2   = 1u << 5,
    NU_EXPOSURE_HAS_AUTO_EXPOSURE = 1u << 6,
};

typedef struct {
    uint32_t flags;               /* NU_EXPOSURE_HAS_* bitmask */
    float    exposure_f_stop;     /* OmniRtxCameraExposureAPI exposure:fStop */
    float    exposure_responsivity;
    float    exposure_time;       /* shutter duration in seconds */
    float    white_point_scale;   /* omni:rtx:autoExposure:whitePointScale */
    float    tonemap_f_number;    /* rtx/omni:rtx:post:tonemap:fNumber */
    float    tonemap_cm2_factor;  /* rtx/omni:rtx:post:tonemap:cm2Factor */
    int      auto_exposure_enabled;
} NuExposureDesc;

/* ---- Lifecycle ---- */

/* Query backend metadata and supported capabilities. Pass r == NULL for
 * conservative static metadata; hardware/runtime capabilities are added
 * when a renderer instance is provided. */
NuResult     nu_get_backend_info(NuRenderer* r, NuBackendInfo* out_info);

/* Create a headless renderer. Pass NULL config for defaults. */
NuRenderer*  nu_renderer_create(const NuRendererConfig* config);

/* Destroy renderer and free all GPU resources. */
void         nu_renderer_destroy(NuRenderer* r);

/* ---- Scene building ---- */

/* Add a mesh to the scene. Returns mesh_id >= 0, or NuResult < 0 on error. */
int          nu_add_mesh(NuRenderer* r, const NuMeshDesc* desc);

/* Create an instance of an existing mesh, sharing its geometry and BLAS.
 * The instance gets its own transform but references the prototype's vertex/index
 * data, avoiding duplicate BLAS builds. Ideal for instanced environments (e.g.
 * IsaacLab) where many shapes share the same underlying mesh.
 * Returns new mesh_id >= 0, or NuResult < 0 on error. */
int          nu_add_mesh_instance(NuRenderer* r, int prototype_mesh_id,
                                  const float transform[16]);

/* Override the display color of an existing mesh slot. Use after
 * nu_add_mesh_instance() to give per-instance colors that differ from the
 * prototype. Color is RGB in linear space, [0,1]. */
NuResult     nu_set_mesh_color(NuRenderer* r, int mesh_id, const float color[3]);

/* Set TLAS instance.mask bytes for a contiguous range of meshes [0..count).
 * Used for per-tile env isolation: each ray's cullMask is AND-ed with
 * instance.mask, so an instance with mask=0x01 only renders for rays whose
 * cullMask includes bit 0. Default mask is 0xFF (visible to all rays).
 * Pass NULL or count=0 to clear all per-instance masks back to 0xFF.
 * IsaacLab Vulkan renderer packs env_idx into 8 buckets via mask = 1 << (env_idx % 8). */
NuResult     nu_set_instance_masks(NuRenderer* r, const unsigned char* masks, int count);

/* Phase C — Per-env TLAS partition. Declares a logical N-way partition of
 * the loaded shapes into envs (e.g. for IsaacLab tile-camera training). The
 * renderer stores the partition; if Phase C support is fully implemented at
 * the gpu_vulkan layer, future render calls will build and bind one TLAS
 * per env (true per-tile isolation, like newton's bvh_shapes_group_roots).
 *
 * Currently the partition is stored but the gpu_vulkan layer falls back to
 * the single-TLAS + 8-bucket cullMask path (Phase A). Future work: switch
 * to per-env TLAS array indexed by cam_env_idx in the raygen shader.
 *
 * mesh_to_env: int[count] mapping from mesh_id (returned by add_mesh /
 *              add_mesh_instance) to env_idx in [0, num_envs).
 * count:       must equal nmeshes; pass mesh_to_env[i] = -1 for static
 *              globals (visible to all rays).
 * num_envs:    total env count. 0 disables partitioning. */
NuResult     nu_set_env_partition(NuRenderer* r,
                                  const int* mesh_to_env,
                                  int count,
                                  int num_envs);

/* Remove a previously added mesh. */
NuResult     nu_remove_mesh(NuRenderer* r, int mesh_id);

/* Remove all meshes and reset the scene. */
void         nu_clear_scene(NuRenderer* r);

/* Convenience: load all meshes from a USD file (USDA or USDC).
 * Returns number of meshes loaded, or NuResult < 0 on error. */
int          nu_load_usd(NuRenderer* r, const char* path);

/* Load a scene from an already-open NanousdStage handle.
 * The renderer borrows the stage — the caller retains ownership and is
 * responsible for keeping it alive until nu_clear_scene / nu_renderer_destroy
 * runs (or another nu_load_usd* call replaces the scene).
 * `stage_label` is a short identifier used in diagnostics (typically the
 * source file path). Returns number of meshes loaded, or NuResult < 0. */
int          nu_load_usd_from_handle(NuRenderer* r,
                                     void* stage,
                                     const char* stage_label);

/* Same as nu_load_usd_from_handle but lets the caller hint the directory
 * that should be scanned for sidecar .mtlx files. Without this, materials
 * loading sees only the directory derived from `stage_label`'s path; when
 * the stage was opened by a caller that doesn't pass a meaningful
 * stage_label (e.g. a bare in-memory stage), the side-loader finds no
 * sidecar materials. NULL/empty `scene_dir` falls back to the stage_label
 * derivation (same behavior as nu_load_usd_from_handle). */
int          nu_load_usd_from_handle_with_dir(NuRenderer* r,
                                              void*       stage,
                                              const char* stage_label,
                                              const char* scene_dir);

/* Rebuild acceleration structures after adding/removing meshes.
 * Called automatically by nu_render() if scene is dirty,
 * but can be called explicitly to control timing. */
NuResult     nu_build_accel(NuRenderer* r);

/* Materialize geometry for a lazy-loaded scene (Tier 3, see
 * docs/plans/TIER_3_LAZY_MESH.md). When NUSD_LAZY_MESH=1 is set during
 * nu_load_usd / nu_load_usd_from_handle, scene_load returns a
 * metadata-only scene and nu_attach_scene parks the stage handle
 * instead of extracting geometry. Render calls fail-fast with "no
 * geometry to render" until this function runs.
 *
 * nu_extract_deferred re-runs scene_load_from_stage on the pinned
 * stage with the lazy flag cleared, then re-attaches the resulting
 * eager scene through the standard nu_attach_scene path. After
 * success, the renderer holds a normal eager-loaded scene and
 * rendering works.
 *
 * Idempotent: returns NU_OK without doing anything if there is no
 * lazy load pending. Safe to call regardless of whether the most
 * recent load was lazy. */
NuResult     nu_extract_deferred(NuRenderer* r);

/* Materialize geometry for ONLY the meshes whose world-space AABB
 * intersects at least one of the supplied camera frusta (Tier 3 step 4
 * frustum cull). Same contract as nu_extract_deferred, but the renderer
 * walks the lazy scene's world AABBs against the union of frusta first
 * and re-runs scene_load_from_stage with a prim-index filter.
 *
 * vp_matrices: float[num_cameras * 16] — row-major 4x4 view-projection
 *              matrix per camera (FORWARD, not inverse). The 6 frustum
 *              planes are extracted from each VP via the standard
 *              Gribb–Hartmann row-add/row-sub trick.
 * num_cameras: number of cameras to union; 0 = no cull, extract everything
 *              (matches nu_extract_deferred behaviour).
 *
 * If the most recent load was eager OR no AABBs were computed during the
 * lazy walk (NUSD_LAZY_AABB=0), this falls back to extracting everything
 * — frustum cull only kicks in when both lazy + AABB walk ran.
 *
 * Diagnostic: a one-line `[scene_load] filter active: K/N prims wanted`
 * message reports the cull rate at extract time. */
NuResult     nu_extract_deferred_visible(NuRenderer* r,
                                         const float* vp_matrices,
                                         int num_cameras);

/* Materialize geometry in N batches over disjoint prim-index slices
 * (Tier 3 step 5 — batched eager extraction). Each slice goes through
 * scene_load_from_stage_filtered → nu_attach_scene, then the slice's
 * arena is released before the next slice starts. Caps peak RSS at
 * approximately (nanousd_stage + arena_per_slice + accumulated_cpu_vertices).
 *
 * num_batches: number of slices to split prims into (e.g. 8).
 *              1 or 0 = same as nu_extract_deferred (no batching).
 *
 * Slice assignment: round-robin by prim index (`i % num_batches`).
 * Spreads spatial clustering across slices so each batch has a roughly
 * uniform footprint.
 *
 * Materials/lights/dome-IBL re-upload on each batch (wasteful but
 * functionally correct — same stage produces the same data each time).
 * NUSD_AUTO_FINALIZE only fires on the LAST batch; intermediate batches
 * keep cpu_vertices alive so the next batch's nu_add_mesh appends.
 *
 * Returns NU_OK on success, NU_ERROR on any batch failure. */
NuResult     nu_extract_deferred_batched(NuRenderer* r, int num_batches);

/* Finalize the scene: ensure all GPU buffers + BLAS/TLAS are built, then
 * drop the renderer-side CPU mirror (vertex/index arrays). After this
 * returns NU_OK the renderer holds geometry exclusively on the GPU, and
 * subsequent mutation APIs (nu_add_mesh, nu_set_mesh_color, nu_remove_mesh,
 * nu_load_texture) will fail fast — to mutate the scene again, call
 * nu_clear_scene and reload.
 *
 * Typical use: call once after nu_load_usd / nu_load_usd_from_handle
 * completes and before entering the render loop. On DSX-scale scenes this
 * frees the multi-GB cpu_vertices/cpu_indices mirror that exists only as
 * an upload-staging source for rebuild_gpu_buffers().
 *
 * Returns NU_OK on success. Returns NU_ERROR if buffer/accel build fails
 * (CPU mirror is preserved in that case so a retry is possible). */
NuResult     nu_finalize_scene(NuRenderer* r);

/* Set the time at which subsequent scene loads / xform reads evaluate
 * USD attributes. Pass NaN (or any non-finite value) to use the
 * authored default time. The renderer caches this as a double — the
 * scene loader passes it to nanousd_get_local_transform() so
 * xformOp:translate.timeSamples and friends resolve at the right frame.
 * After calling this, trigger a fresh nu_load_usd_from_handle() (or
 * nu_load_usd) for the change to take effect.
 *
 * Returns NU_OK on success, NU_ERROR_BAD_ARG if r is NULL. */
NuResult     nu_set_current_time(NuRenderer* r, double time);

/* ---- Gaussian splat scene (VK3DGRT) ---- */
/* See docs/plans/VK3DGRT_PLAN.md (parent repo). 3D Gaussian Ray Tracing
 * path operating in parallel to the mesh scene. Inputs match the AOUSD
 * UsdVol.ParticleField3DGaussianSplat schema (USD 26.03). RT-only. */

typedef enum {
    NU_GS_PROXY_ICOSAHEDRON = 0, /* 20-tri unit BLAS, hardware ray-tri (default) */
    NU_GS_PROXY_AABB        = 1, /* unit AABB + procedural intersection */
} NuGsProxyKind;

typedef enum {
    NU_GS_COLOR_LINEAR      = 0, /* default — matches training convention */
    NU_GS_COLOR_SRGB        = 1, /* SH coefficients in nonlinear sRGB */
} NuGsColorSpace;

typedef enum {
    NU_GS_CAMERA_PINHOLE    = 0, /* default */
    NU_GS_CAMERA_FISHEYE    = 1,
    NU_GS_CAMERA_EQUIRECT   = 2,
} NuGsCameraModel;

typedef struct {
    /* Per-particle arrays — all length particle_count unless noted. */
    const float* positions;        /* particle_count * 3 — object space */
    const float* scales;           /* particle_count * 3 — linear sigma */
    const float* orientations;     /* particle_count * 4 — wxyz quaternion, normalized */
    const float* opacities;        /* particle_count * 1 — [0,1], post-sigmoid */
    const float* sh_coefficients;  /* particle_count * (sh_degree+1)^2 * 3 */
    int          sh_degree;        /* 0..3 (DC at index 0) */
    int          particle_count;
    /* Optional row-major 4x4 prim transform; NULL = identity. */
    const float* prim_xform;
} NuGsDesc;

/* Upload (or replace) all gaussian particles. Triggers TLAS rebuild on next render. */
NuResult     nu_gs_set_particles(NuRenderer* r, const NuGsDesc* desc);

/* Remove all gaussian particles. */
NuResult     nu_gs_clear_particles(NuRenderer* r);

/* Choose particle proxy: ICOSAHEDRON (HW ray-tri, default) or AABB (procedural rint). */
NuResult     nu_gs_set_proxy(NuRenderer* r, NuGsProxyKind kind);

/* Color space of SH coefficients. Default LINEAR. */
NuResult     nu_gs_set_color_space(NuRenderer* r, NuGsColorSpace cs);

/* Camera model used by the VK3DGRT raygen. Default PINHOLE. */
NuResult     nu_gs_set_camera_model(NuRenderer* r, NuGsCameraModel cm);

/* K-buffer size (8, 16, or 32). Default 16. Re-trigger pipeline rebuild. */
NuResult     nu_gs_set_k(NuRenderer* r, int k);

/* Maximum re-trace passes per pixel. Default 16. */
NuResult     nu_gs_set_max_passes(NuRenderer* r, int max_passes);

/* Minimum transmittance to continue tracing (early-out). Default 0.03. */
NuResult     nu_gs_set_min_transmittance(NuRenderer* r, float eps);

/* Iso-opacity threshold for the surface depth/normal AOV. Default 0.5. */
NuResult     nu_gs_set_iso_opacity_threshold(NuRenderer* r, float iso);

/* Query whether VK3DGRT is available (extensions present + at least one
 * particle uploaded). Returns 1 if ready to render, 0 otherwise. */
int          nu_gs_available(NuRenderer* r);

/* Query particle count currently uploaded. */
int          nu_gs_particle_count(NuRenderer* r);

/* Trace the splat scene from camera `cam_id` into the renderer's RT image.
 * Lazily builds the splat RT pipeline + SBT on first call. Pixels are
 * retrievable via nu_fetch_pixels after this call. Iso-opacity depth
 * is also written to a side buffer; fetch via nu_gs_fetch_depth.
 * Returns NU_OK on success; NU_ERROR_NO_SCENE if no particles uploaded;
 * NU_ERROR_NO_RT if the RT pipeline can't be built. */
NuResult     nu_gs_render(NuRenderer* r, int cam_id);

/* Read back the iso-opacity surface depth for the most recent
 * nu_gs_render. `out_depth` must be width*height float32s.
 * Per-pixel value: t at which accumulated opacity first crosses
 * the configured iso_opacity_threshold (default 0.5); -1 if it
 * never crosses (sky / cumulative opacity insufficient). Plan §7. */
NuResult     nu_gs_fetch_depth(NuRenderer* r, float* out_depth);

/* Read back the iso-opacity surface normal for the most recent
 * nu_gs_render. `out_normal` must be width*height*3 float32s
 * (xyz per pixel, world space). (0, 0, 0) on miss. Plan §7. */
NuResult     nu_gs_fetch_normal(NuRenderer* r, float* out_normal);

/* ---- Attribute updates ---- */

/* Set world transforms for meshes. mat4x4s is count * 16 floats (row-major). */
NuResult     nu_set_transforms(NuRenderer* r,
                               const int* mesh_ids, const float* mat4x4s, int count);

/* Set display colors for meshes. rgb is count * 3 floats. */
NuResult     nu_set_colors(NuRenderer* r,
                           const int* mesh_ids, const float* rgb, int count);

/* Set visibility for meshes. visible is count ints (0=hidden, 1=visible). */
NuResult     nu_set_visibility(NuRenderer* r,
                               const int* mesh_ids, const int* visible, int count);

/* ---- Camera ---- */

/* Set camera parameters. cam_id 0 is always available. */
NuResult     nu_set_camera(NuRenderer* r, int cam_id, const NuCameraDesc* desc);

/* Set a fully explicit camera with arbitrary up vector. Use for Z-up scenes
 * (e.g. IsaacLab/Newton) where the default Y-up spherical camera doesn't
 * apply. eye/target/up are world-space float[3]. */
NuResult     nu_set_camera_explicit(NuRenderer* r,
                                    const float eye[3],
                                    const float target[3],
                                    const float up[3],
                                    float fov_degrees,
                                    float near_clip,
                                    float far_clip);

/* Set an explicit camera with an off-axis projection. projection_shift_x/y are
 * normalized aperture offsets in NDC units; 0,0 is the symmetric projection.
 * This is used for USD horizontal/verticalApertureOffset lens shift. */
NuResult     nu_set_camera_explicit_window(NuRenderer* r,
                                           const float eye[3],
                                           const float target[3],
                                           const float up[3],
                                           float fov_degrees,
                                           float near_clip,
                                           float far_clip,
                                           float projection_shift_x,
                                           float projection_shift_y);

/* Read back the renderer's current camera state into out_desc. Useful for
 * picking up the auto-framed camera produced by nu_load_usd before applying
 * user-driven mouse / scroll deltas. */
NuResult     nu_get_camera(NuRenderer* r, int cam_id, NuCameraDesc* out_desc);

/* Set OVRTX/Omniverse-style camera exposure and post-tonemap hints. The
 * renderer maps these to a relative scale on its existing ACES tone path:
 * no authored fields means no visual change, while f-stop/time/responsivity
 * follow the OVRTX photographic exposure relation. */
NuResult     nu_set_exposure(NuRenderer* r, const NuExposureDesc* desc);

/* ---- Rendering ---- */

/* Render a frame. Blocks until complete.
 * Returns NU_OK on success. */
NuResult     nu_render(NuRenderer* r, int cam_id, NuRenderMode mode);

/* Read back rendered pixels to CPU buffer.
 * out_pixels must be pre-allocated (width * height * bytes_per_pixel).
 * Call after nu_render(). */
NuResult     nu_fetch_pixels(NuRenderer* r, void* out_pixels, NuPixelFormat format);

/* CUDA-Vulkan interop variant of nu_fetch_pixels: copy the rendered pixels
 * into a device-local exportable Vulkan buffer and return a POSIX fd that
 * the caller imports into CUDA via cuImportExternalMemory + cuExternalMemoryGetMappedBuffer.
 * The data layout is BGRA8 (raw swapchain — no CPU swizzle pass).
 *
 * The returned fd is owned by the renderer and closed by nu_renderer_destroy.
 * Subsequent calls return the SAME fd, so the caller can import once and
 * reuse the imported memory across frames (avoiding per-frame fd / kernel-side
 * cuImportExternalMemory cost).
 *
 * out_mem_fd:    POSIX fd (cached; do NOT close)
 * out_size:      allocation size in bytes (use this for cuExternalMemoryGetMappedBuffer)
 * out_width:     pixel width
 * out_height:    pixel height
 * out_format:    pixel format (currently always NU_PIXEL_BGRA8)
 *
 * Call after nu_render(). Returns NU_OK on success. */
NuResult     nu_fetch_pixels_cuda(NuRenderer* r,
                                  int* out_mem_fd,
                                  uint64_t* out_size,
                                  int* out_width,
                                  int* out_height,
                                  int* out_format);

/* Map rendered pixels as a CUDA device pointer (zero-copy Vulkan-CUDA interop).
 * Returns CUDA device pointer, or NULL if interop unavailable.
 * Pointer is valid until nu_unmap_pixels_gpu() or next nu_render(). */
void*        nu_map_pixels_gpu(NuRenderer* r);

/* Unmap previously mapped GPU pixels. */
void         nu_unmap_pixels_gpu(NuRenderer* r);

/* ---- Async double-buffered render+fetch (single camera) ----
 *
 * The pair (nu_render_async, nu_fetch_async) implements a producer/consumer
 * pipeline: while the GPU traces frame N, the CPU memcpys frame N-1 from the
 * other slot. Steady-state cost drops from sync render+fetch (~28 ms at
 * 1280x720, curve scenes) to roughly max(GPU work, readback memcpy).
 *
 * Internally these wrap the existing tiled path with num_cameras=1, so they
 * inherit the double-buffered staging buffers, fences, and command-buffer
 * recycling that the tiled async path already uses. Output is RGBA8 (no BGRA
 * swizzle, unlike nu_fetch_pixels) and lives in r->camera-derived inverse
 * matrices supplied by nu_set_camera.
 *
 * Usage (target: 1 ms/frame steady state for small GPU work):
 *
 *   for (i = 0; i < N; i++) {
 *       nu_render_async(r);                  // submit frame N to GPU
 *       nu_fetch_async(r, host_buf);         // returns frame N-1's pixels
 *   }
 *
 * On the FIRST call, fetch_async has no previous frame to return — the
 * destination buffer is zero-filled and NU_OK is returned. The first valid
 * pixels come back on the second call. */
NuResult     nu_render_async(NuRenderer* r);

/* Fetch the previous async-rendered frame's pixels into a caller-allocated
 * buffer. The buffer must be width*height*4 bytes (RGBA8). Returns NU_OK
 * on success, including the first-frame zero-fill case. */
NuResult     nu_fetch_async(NuRenderer* r, void* out_pixels);

/* ---- Tiled multi-camera rendering ---- */

/* Render multiple cameras in a single GPU dispatch using tiled ray tracing.
 * Each camera's output occupies one tile_w x tile_h region in a grid layout.
 *
 * vp_inv_matrices: num_cameras * 32 floats — pairs of (view_inverse[16], proj_inverse[16])
 *                  Row-major 4x4 matrices. Same layout as GpuRtPushConstants.
 * num_cameras:     Number of cameras (tiles) to render.
 * tile_w, tile_h:  Per-camera resolution (pixels).
 * mode:            NU_RENDER_RT only (raster tiling not supported).
 *
 * Tiles are laid out in a ceil(sqrt(N)) x ceil(N/cols) grid.
 * Returns NU_OK on success. */
NuResult     nu_render_tiled(NuRenderer* r,
                             const float* vp_inv_matrices, int num_cameras,
                             int tile_w, int tile_h, NuRenderMode mode);

/* Read back tiled render output to CPU buffer.
 * out_pixels must hold num_cameras * tile_w * tile_h * 4 bytes (RGBA8).
 * Output is de-tiled: camera 0 at offset 0, camera 1 at tile_w*tile_h*4, etc.
 * Call after nu_render_tiled(). */
NuResult     nu_fetch_pixels_tiled(NuRenderer* r, void* out_pixels,
                                   int num_cameras, int tile_w, int tile_h);

/* Read back tiled depth output to CPU buffer.
 * out_depth must hold num_cameras * tile_w * tile_h floats.
 * Values: positive = ray hit distance (world units), -1.0 = miss/sky.
 * Output is de-tiled: camera 0 at offset 0, camera 1 at tile_w*tile_h, etc.
 * Requires nu_set_depth_enabled(r, 1) before rendering.
 * Call after nu_render_tiled(). */
NuResult     nu_fetch_depth_tiled(NuRenderer* r, float* out_depth,
                                   int num_cameras, int tile_w, int tile_h);

/* Read back tiled segmentation output as de-tiled per-camera uint32 arrays.
 * out_ids must hold num_cameras * tile_w * tile_h uint32s.
 * Values: 0 = miss/sky/ground, mesh_index+1 = geometry hit.
 * Output is de-tiled: camera 0 at offset 0, etc.
 * Requires nu_set_segmentation_enabled(r, 1) before rendering.
 * Call after nu_render_tiled(). */
NuResult     nu_fetch_segmentation_tiled(NuRenderer* r, uint32_t* out_ids,
                                          int num_cameras, int tile_w, int tile_h);

/* Read back tiled normals output as de-tiled per-camera float arrays.
 * out_normals must hold num_cameras * tile_w * tile_h * 3 floats (x,y,z per pixel).
 * Values: (0,0,0) = miss/sky, (0,1,0) = ground, otherwise surface normal.
 * Output is de-tiled: camera 0 at offset 0, etc.
 * Requires nu_set_normals_enabled(r, 1) before rendering.
 * Call after nu_render_tiled(). */
NuResult     nu_fetch_normals_tiled(NuRenderer* r, float* out_normals,
                                     int num_cameras, int tile_w, int tile_h);

/* Map tiled staging buffer directly without de-tiling.
 * Returns a pointer to the raw tiled image (total_w * total_h * 4 bytes, RGBA8).
 * Tiles are packed in a ceil(sqrt(N)) column grid — tile (cam) is at
 *   row = cam / num_cols, col = cam % num_cols.
 * out_total_w/out_total_h receive the tiled image dimensions.
 * Pointer is valid until the next nu_render_tiled() call.
 * Returns NULL on failure. */
const void*  nu_map_tiled_pixels_raw(NuRenderer* r,
                                     int num_cameras, int tile_w, int tile_h,
                                     int* out_total_w, int* out_total_h);

/* Async-readback variant: map a specific slot (0 or 1) of the double-buffered
 * tiled staging, independent of the current write_idx. The caller tracks which
 * slot holds its pending render (via nu_get_last_tiled_slot after render_tiled)
 * and reads from that slot on the NEXT frame, so the GPU copy overlaps with
 * CPU-side work. Returns NULL if the slot has no in-flight submission. */
const void*  nu_map_tiled_pixels_raw_slot(NuRenderer* r,
                                          int num_cameras, int tile_w, int tile_h,
                                          int slot,
                                          int* out_total_w, int* out_total_h);

/* Return the double-buffer slot (0 or 1) that the most recent nu_render_tiled
 * wrote to. Call immediately after render to stash the slot for a subsequent
 * async fetch via nu_map_tiled_pixels_raw_slot. Returns -1 on error. */
int          nu_get_last_tiled_slot(NuRenderer* r);

/* ---- Ray casting (LiDAR/radar sensor simulation) ---- */

/* Cast rays against the scene using hardware ray tracing (compute shader +
 * GL_EXT_ray_query).  This traces arbitrary rays — not tied to screen space
 * or camera parameters — making it suitable for LiDAR/radar simulation.
 *
 * ray_origins:       num_rays * 3 floats — world-space ray origin positions
 * ray_directions:    num_rays * 3 floats — normalized ray directions
 * num_rays:          number of rays to cast
 * max_distance:      maximum ray travel distance (meters)
 * out_distances:     num_rays floats — hit distance per ray (-1.0 = miss)
 * out_normals:       num_rays * 3 floats — approximate surface normal at hit
 * out_hit_positions: num_rays * 3 floats — world-space hit position
 *
 * All output arrays must be pre-allocated by the caller.
 * Requires RT to be available and the scene to have been built.
 * Returns NU_OK on success. */
NuResult     nu_cast_rays(NuRenderer* r,
                          const float* ray_origins,
                          const float* ray_directions,
                          int num_rays,
                          float max_distance,
                          float* out_distances,
                          float* out_normals,
                          float* out_hit_positions);

/* Debug raycast variant that also returns renderer mesh_id per ray
 * (-1 on miss). Output arrays must be pre-allocated by the caller. */
NuResult     nu_cast_rays_with_ids(NuRenderer* r,
                                   const float* ray_origins,
                                   const float* ray_directions,
                                   int num_rays,
                                   float max_distance,
                                   float* out_distances,
                                   int* out_mesh_ids,
                                   float* out_normals,
                                   float* out_hit_positions);

/* Async raycast: submit dispatch without blocking.
 * Call nu_cast_rays_wait() later to read results.
 * This allows overlapping raycast GPU work with CPU or other GPU work. */
NuResult     nu_cast_rays_async(NuRenderer* r,
                                const float* ray_origins,
                                const float* ray_directions,
                                int num_rays,
                                float max_distance);

/* Async raycast: wait for pending dispatch and read results.
 * If no dispatch is pending, returns NU_OK immediately. */
NuResult     nu_cast_rays_wait(NuRenderer* r,
                               float* out_distances,
                               float* out_normals,
                               float* out_hit_positions);

/* Save the last rendered frame as a PPM image file. */
NuResult     nu_save_ppm(NuRenderer* r, const char* path);

/* ---- Configuration ---- */

/* Resize the render target. Takes effect on next nu_render(). */
NuResult     nu_set_render_size(NuRenderer* r, int width, int height);

/* Query whether hardware ray tracing is available. */
int          nu_rt_available(NuRenderer* r);

/* Query current render target dimensions. */
void         nu_get_render_size(NuRenderer* r, int* width, int* height);

/* Return the underlying GLFWwindow* (cast to void* to avoid leaking the
 * GLFW header into the public API). Valid only when the renderer was
 * created with NuRendererConfig.visible == 1. NULL otherwise. The
 * caller can use this handle to wire up GLFW input callbacks
 * (glfwSetKeyCallback, glfwSetCursorPosCallback, etc.) and to drive the
 * event loop via glfwPollEvents / glfwWindowShouldClose. */
void*        nu_get_window(NuRenderer* r);

/* ---- Environment ---- */

/* Load an HDR environment map for image-based lighting. */
NuResult     nu_load_environment(NuRenderer* r, const char* hdr_path);

/* Variant with explicit DomeLight intensity multiplier (matches ovrtx's
 * `dome.color = light.color * intensity`). Negative intensity selects
 * the auto-exposure path (same as plain nu_load_environment). */
NuResult     nu_load_environment_intensity(NuRenderer* r, const char* hdr_path, float intensity);

/* Upload a single RGBA8 texture (linear or sRGB, depending on use) and
 * append it to the renderer's runtime texture array. Returns the new
 * texture index (>= 0) suitable for nu_set_mesh_texture(), or NU_ERROR
 * (-1) on failure.
 *
 * pixels:  width * height * 4 bytes, RGBA8. Treated as sRGB color data
 *          (hardware sRGB → linear at sample), matching the existing
 *          UsdPreviewSurface diffuse-texture pipeline.
 * width, height: texture dimensions (must be > 0).
 *
 * The texture is owned by the renderer until nu_clear_scene() or
 * nu_renderer_destroy(); the caller may free `pixels` after the call.
 *
 * Internally this calls gpu_upload_materials(NULL, 0, textures, ntextures)
 * with the accumulated texture array. For the IsaacLab fast-path (no USD
 * material parser), this is the only texture-upload entry point.
 *
 * To bind a uploaded texture to a mesh, call nu_set_mesh_texture(). Without
 * an explicit bind, every mesh's tex_index defaults to the sentinel
 * 0xFFFFFFFF and the rchit fast_mode path skips texture sampling
 * (byte-identical for un-textured meshes). */
int          nu_load_texture(NuRenderer* r, const uint8_t* pixels,
                             int width, int height);

/* Bind a previously-loaded texture index to a mesh by its mesh_id (returned
 * from nu_add_mesh / nu_add_mesh_instance). Pass tex_index < 0 to clear
 * (revert to flat per-mesh color). The mesh must have been created with
 * non-NULL `texcoords` for the rchit UV read to produce the right value.
 * Returns NU_OK on success. */
NuResult     nu_set_mesh_texture(NuRenderer* r, int mesh_id, int tex_index);

/* Set the scene's flat (no-IBL) DomeLight color used by the fast_mode path.
 * In fast_mode, the rmiss shader emits `domeColor.rgb * domeColor.a` as the
 * sky, and the rchit hemispheric ambient is modulated by `domeColor.rgb`.
 * Defaults to white at intensity 1.0 (matches Newton's 0xEEEEEE clear color
 * within ~17/255). For RL training where the scene authors a USD DomeLight,
 * the IsaacLab adapter should map the light's color/intensity through this
 * setter so the Vulkan sky background tracks the scene's actual lighting.
 *
 * r/g/b are linear RGB in [0,1] (typical USD DomeLight `inputs:color`).
 * intensity is a scalar multiplier (typical USD DomeLight `inputs:intensity`
 * already pre-normalized to [0,1] by the caller — this entry point does not
 * apply any sqrt/auto-exposure compression).
 *
 * Takes effect on the next frame; cheap (writes a vec4 into the SceneData
 * SSBO header). Returns NU_OK on success. */
NuResult     nu_set_dome_color(NuRenderer* r, float r_, float g, float b,
                               float intensity);

/* ---- CUDA Interop (zero-copy GPU sharing) ---- */

/* Query whether CUDA-Vulkan interop is available.
 * Requires VK_KHR_external_memory_fd + VK_KHR_external_semaphore_fd. */
int          nu_interop_available(NuRenderer* r);

/* Wait for the most recent nu_render_tiled() to complete on the GPU.
 * After this returns, the interop buffer (exported via nu_get_cuda_interop_info)
 * contains the tiled render result and can be safely read by CUDA.
 * This is cheaper than nu_map_tiled_pixels_raw — it waits on the Vulkan fence
 * but does NOT map or copy any staging data. */
NuResult     nu_wait_tiled_complete(NuRenderer* r);

/* Interop info for importing the tiled image into CUDA.
 * Double-buffered: mem_fd[0] and mem_fd[1] are two separate GPU buffers.
 * Vulkan writes to one while CUDA reads from the other. */
typedef struct {
    int      mem_fd[2];       /* Opaque fds for the two interop buffer VkDeviceMemory objects */
    uint64_t mem_size;        /* Allocation size in bytes (same for both buffers) */
    uint32_t image_w;         /* Tiled image width (total_w) */
    uint32_t image_h;         /* Tiled image height (total_h) */
    uint32_t tile_w;          /* Per-camera tile width */
    uint32_t tile_h;          /* Per-camera tile height */
    int      num_cameras;     /* Number of cameras in the grid */
    int      sem_fd;          /* Opaque fd for timeline semaphore */
    uint64_t sem_value;       /* Current timeline value (incremented per render) */
} NuCudaInteropInfo;

/* Get CUDA interop handles for zero-copy access to the tiled render target.
 * The mem_fds and sem_fd are one-time-use POSIX fds — caller must import them
 * into CUDA (cuImportExternalMemory / cuImportExternalSemaphore) or close them.
 * Call after nu_render_tiled(). Returns NU_OK on success.
 *
 * Returns NU_ERROR if the renderer is currently using caller-owned external
 * output buffers (set via nu_set_external_output_buffers) — in that mode the
 * caller already owns the memory and there is nothing to export. */
NuResult     nu_get_cuda_interop_info(NuRenderer* r, NuCudaInteropInfo* out,
                                       int num_cameras, int tile_w, int tile_h);

/* Hybrid C — Inverted memory ownership for the tiled output buffer.
 *
 * Default interop has Vulkan allocate the tiled output buffer and CUDA import
 * it via cuImportExternalMemory + cuExternalMemoryGetMappedBuffer. That import
 * is suspected of triggering a slow-mode in cudaStreamSynchronize when the
 * trainer's CUDA context holds active imports of Vulkan-owned memory.
 *
 * This entry point inverts the direction: the caller (trainer) allocates the
 * output buffer via the CUDA VMM API — cuMemCreate(POSIX_FILE_DESCRIPTOR) +
 * cuMemAddressReserve + cuMemMap + cuMemSetAccess + cuMemExportToShareableHandle —
 * and passes the resulting fds in. The renderer creates VkBuffer objects with
 * VK_KHR_external_memory_fd, imports each fd as VkDeviceMemory via
 * VkImportMemoryFdInfoKHR, and writes its tiled output directly into the
 * caller-owned memory. The caller reads pixels via its own CUdeviceptr — no
 * cuImportExternalMemory needed.
 *
 * The renderer takes ownership of the fds on success (consumed by
 * vkAllocateMemory). On failure, the caller still owns the fds and must close
 * them.
 *
 * mem_fds:        two POSIX fds, one per double-buffered slot (slot 0 / slot 1).
 * mem_size_each:  size of EACH buffer in bytes. Must be >= num_cameras *
 *                 tile_w * tile_h * 4 AND >= Vulkan req.size (which the caller
 *                 can satisfy by rounding the logical size up to the maximum
 *                 of CUDA VMM granularity and 64 KiB).
 * num_cameras:    number of cameras in the tiled grid (matches subsequent
 *                 nu_render_tiled calls).
 * tile_w, tile_h: per-camera tile resolution.
 *
 * After this call, nu_get_cuda_interop_info returns NU_ERROR — there is
 * nothing for the renderer to export. The timeline semaphore (still
 * Vulkan-owned) is exported separately via nu_get_external_timeline_semaphore_fd. */
NuResult     nu_set_external_output_buffers(NuRenderer* r,
                                            int mem_fds[2],
                                            uint64_t mem_size_each,
                                            int num_cameras,
                                            int tile_w,
                                            int tile_h);

/* Companion to nu_set_external_output_buffers: export the renderer's timeline
 * semaphore as a POSIX fd so the caller can import it via cuImportExternalSemaphore
 * and issue cuWaitExternalSemaphoresAsync on its CUDA stream. The current value
 * is whatever has been signalled so far; subsequent renders increment it.
 *
 * out_sem_fd:    one-time-use POSIX fd (consumed by cuImportExternalSemaphore
 *                or closed by the caller).
 * out_sem_value: current timeline value — pass-through of gpu_get_interop_timeline_value.
 *
 * Returns NU_OK on success, NU_ERROR otherwise. */
NuResult     nu_get_external_timeline_semaphore_fd(NuRenderer* r,
                                                   int* out_sem_fd,
                                                   uint64_t* out_sem_value);

/* Return the interop buffer index (0 or 1) for the CURRENT frame.
 * This is the buffer written by the most recent render_tiled(). */
int          nu_get_interop_read_idx(NuRenderer* r);

/* Return the interop buffer index (0 or 1) for the PREVIOUS frame.
 * For double-buffered overlap: read this while the current frame renders. */
int          nu_get_interop_prev_idx(NuRenderer* r);

/* Wait for the PREVIOUS frame's tiled render to complete.
 * For double-buffered overlap: submit frame N, then call this to wait on
 * frame N-1 (typically instant since N-1 was submitted before N).
 * Returns NU_OK on success, NU_ERROR if no previous frame. */
NuResult     nu_wait_previous_tiled_complete(NuRenderer* r);

/* Skip CPU staging buffer copy when CUDA interop is active.
 * Saves one full-image GPU copy per frame (~80MB at 2048 envs).
 * When enabled, nu_fetch_pixels_tiled / nu_map_tiled_pixels_raw will not work. */
void         nu_set_skip_staging(NuRenderer* r, int skip);

/* Enable fast mode for RL sensors: skip shadow rays and use simple diffuse
 * lighting. Roughly halves RT dispatch time at the cost of visual quality.
 * fast=1 enables, fast=0 disables (default). */
void         nu_set_fast_mode(NuRenderer* r, int fast);

/* Enable per-env SSBO layout: when direct write is active, the raygen shader
 * writes pixels in [env, H, W] contiguous layout instead of 2D tiled layout.
 * This eliminates the need for a CUDA de-tiling kernel. enable=1 on, 0 off. */
void         nu_set_per_env_layout(NuRenderer* r, int enable);

/* Apply the sRGB transfer function in the tiled raygen shader. When enabled,
 * tiled RGBA8 output is already gamma-encoded, so Python callers can skip
 * their CPU LUT pass (~16 ms per 4-camera tick at 1920x1200). enable=1 on. */
void         nu_set_tiled_srgb(NuRenderer* r, int enable);

/* Enable depth output from tiled ray tracing.
 * When enabled, the closest-hit and miss shaders write ray T distances
 * to a float32 SSBO, retrievable via nu_fetch_depth_tiled().
 * enable=1 enables, enable=0 disables (default). */
void         nu_set_depth_enabled(NuRenderer* r, int enable);

/* Enable semantic instance segmentation output from tiled ray tracing.
 * When enabled, the closest-hit shader writes mesh_index+1 to a uint32 SSBO
 * (0 = miss/sky/ground). Retrievable via nu_fetch_segmentation_tiled().
 * enable=1 enables, enable=0 disables (default). */
void         nu_set_segmentation_enabled(NuRenderer* r, int enable);

/* Enable surface normals output from tiled ray tracing.
 * When enabled, the closest-hit and miss shaders write (x,y,z) normal vectors
 * to a float32 SSBO (3 floats per pixel). Retrievable via nu_fetch_normals_tiled().
 * enable=1 enables, enable=0 disables (default). */
void         nu_set_normals_enabled(NuRenderer* r, int enable);

/* Phase B deferred-shading toggle. When enabled, the rchit/rmiss write a
 * per-pixel G-buffer (binding 17) inside the fast_mode path, and a
 * follow-on compute dispatch reads it to produce the final pixels. Phase B
 * is plumbing-only: the compute pass writes flat-shaded per-mesh display
 * color (a "color-ID" debug viz). Phase C/D will add full materials, IBL,
 * and lighting to the compute pass. enable=1 on, 0 off (default).
 *
 * Also auto-enabled by NU_DEFERRED_SHADE=1 at renderer creation time. */
void         nu_set_deferred_shade(NuRenderer* r, int enable);

/* Phase C.2 deferred-shading debug-mode selector. Drives the compute
 * shader's debug visualization output:
 *   0 = base color (Phase C.1 textured albedo, byte-identical default)
 *   1 = world-space shading normal as RGB (post normal-map TBN)
 *   2 = PBR-packed: metallic.r, roughness.g, ao.b
 *   3 = full lit (placeholder; falls back to mode 0 until Phase C.4)
 * Only meaningful when nu_set_deferred_shade(1) is also active. */
void         nu_set_deferred_debug_mode(NuRenderer* r, uint32_t mode);

/* ---- PR 2: GPU-driven TLAS instance translation ----
 *
 * The "host loop + numpy() sync" path of nu_set_transforms is the dominant
 * cost at high instance counts. PR 2 moves it onto the GPU:
 *
 *   1. nu_get_transforms_interop_info() returns a POSIX fd for an
 *      exportable Vulkan storage buffer sized for `count` row-major 4x4
 *      transforms (count * 64 bytes). The Python side imports this via
 *      cuImportExternalMemory and wraps it as a wp.array so warp kernels
 *      write directly into the Vulkan-owned device memory.
 *
 *   2. nu_set_transform_layout() uploads a (gid → tlas_instance_idx)
 *      lookup table that the compute shader uses to direct each thread's
 *      transform write into the right VkAccelerationStructureInstanceKHR
 *      slot. Called once at scene init.
 *
 *   3. nu_translate_instances_gpu() is called every frame in place of
 *      nu_set_transforms — it dispatches the compute shader (writes
 *      transforms into instance_buf) and records the TLAS update + build
 *      in the same command stream. No host data, no .numpy() sync.
 *
 * Visibility changes are NOT propagated by the GPU path — bytes 48..63
 * of each instance record (customIndex/mask/sbtOffset/flags + AS-ref)
 * are preserved from scene init. If the caller needs to toggle visibility,
 * they must fall back to nu_set_visibility + nu_set_transforms (CPU path). */

typedef struct {
    int      mem_fd;       /* one-time-use POSIX fd; caller imports via cuImportExternalMemory */
    uint64_t mem_size;     /* meaningful payload size (count * 64) for cuExternalMemoryGetMappedBuffer */
    int      count;        /* number of 4x4 transforms the buffer holds */
} NuTransformsInteropInfo;

/* Allocate (or resize) the per-shape transforms buffer and return its fd.
 * `count` is the number of row-major 4x4 transforms (n_valid).
 * The fd is single-use; the caller must either import it via
 * cuImportExternalMemory or close it. Returns NU_OK on success. */
NuResult nu_get_transforms_interop_info(NuRenderer* r, int count,
                                        NuTransformsInteropInfo* out);

/* Upload the (gid → tlas_instance_idx) lookup table that the compute
 * shader uses to translate the dense (n_valid) warp output into the
 * sparse-by-mesh-id VkAccelerationStructureInstanceKHR slots of the
 * renderer's instance buffer. mesh_ids[i] is the renderer mesh_id (as
 * returned by nu_add_mesh / nu_add_mesh_instance) of the shape that
 * warp thread `i` writes; the renderer translates these into TLAS-
 * instance indices internally.
 *
 * mesh_ids[i] < 0 → that gid is skipped (used for invalid shapes).
 *
 * Returns NU_OK on success. Call once at scene init (after nu_load_usd
 * / nu_add_mesh* and before the first nu_translate_instances_gpu call). */
NuResult nu_set_transform_layout(NuRenderer* r, const int* mesh_ids, int count);

/* Dispatch the compute shader to translate the warp-written transforms
 * into VkAccelerationStructureInstanceKHR records, then record the TLAS
 * update + build in the same Vulkan command stream. Replaces the host
 * loop in nu_set_transforms.
 *
 * The caller MUST ensure all warp writes into the imported transforms
 * buffer have completed (e.g. via wp.synchronize_device()) before
 * calling this — there is no GPU-side semaphore wiring yet.
 *
 * Returns NU_OK on success. */
NuResult nu_translate_instances_gpu(NuRenderer* r);

/* ---- Raycast GPU Interop (CUDA device pointers for LiDAR) ---- */

/* Info struct for importing raycast buffers into CUDA.
 * Layout (input):  [origins: N*3 floats] [directions: N*3 floats]
 * Layout (output): [distances: N floats] [normals: N*3 floats] [hit_positions: N*3 floats] */
typedef struct {
    int      input_fd;         /* Opaque fd for input VkDeviceMemory (origins+directions) */
    uint64_t input_size;       /* Allocation size in bytes */
    int      output_fd;        /* Opaque fd for output VkDeviceMemory (distances+normals+hits) */
    uint64_t output_size;      /* Allocation size in bytes */
    uint32_t max_rays;         /* Buffer capacity in rays */
} NuRaycastInteropInfo;

/* Ensure raycast buffers exist with at least `num_rays` capacity, create them
 * with external-memory export enabled, and return opaque fds for CUDA import.
 * Call once; the fds are valid until the renderer is destroyed or buffers grow.
 * Returns NU_OK on success. */
NuResult     nu_raycast_get_interop_info(NuRenderer* r, int num_rays,
                                          NuRaycastInteropInfo* out);

/* Dispatch raycast compute shader on data already in the device-local input buffer.
 * The caller (CUDA/Warp) must have written origins+directions into the imported
 * input buffer before calling this. Skips staging upload entirely.
 * This is the async submit; call nu_cast_rays_wait() to read results.
 * Output is written to the device-local output buffer (also CUDA-importable). */
NuResult     nu_cast_rays_gpu(NuRenderer* r, int num_rays, float max_distance);

/* Wait for async raycast fence only (no staging readback).
 * Used by the GPU interop path where CUDA reads from the output buffer directly. */
NuResult     nu_cast_rays_wait_fence(NuRenderer* r);

/* ---- Info ---- */

/* Query the number of active meshes (including instances) in the scene. */
int          nu_get_mesh_count(NuRenderer* r);

/* Query the world-space bounds computed at scene_load time.
 * Writes 3 floats each into out_min and out_max.
 * Returns NU_OK on success; outputs are left untouched if no scene loaded. */
NuResult     nu_get_scene_bounds(NuRenderer* r, float out_min[3], float out_max[3]);

/* Copy the name (typically a USD prim path like "/World/foo/bar") of mesh
 * `mesh_id` into `out_buf`, NUL-terminated, up to `buf_cap` bytes including
 * the terminator. Returns the number of bytes written (excluding NUL), or
 * a negative NuResult on error. If the mesh has no name, writes "" and
 * returns 0. If buf_cap is 0, returns the required size (excluding NUL). */
int          nu_get_mesh_name(NuRenderer* r, int mesh_id,
                              char* out_buf, int buf_cap);

/* Copy the renderer-canonical row-major column-vector world transform
 * (translation in [3], [7], [11]) for `mesh_id` into out_mat4x4[16].
 * Returns NU_OK on success. */
NuResult     nu_get_mesh_transform(NuRenderer* r, int mesh_id,
                                   float out_mat4x4[16]);

/* Query the material index bound to a mesh, or a negative NuResult on error.
 * Returns -1 for meshes without a material binding. */
int          nu_get_mesh_material_index(NuRenderer* r, int mesh_id);

/* Copy the material debug path/name into out_buf. Same return convention as
 * nu_get_mesh_name. */
int          nu_get_material_name(NuRenderer* r, int material_index,
                                  char* out_buf, int buf_cap);

/* Phase 11.A: query the total number of curve segments extracted from
 * the loaded scene's BasisCurves prims. Returns 0 if no curves loaded. */
int          nu_get_curve_segment_count(NuRenderer* r);

/* Query GPU memory usage in bytes. */
uint64_t     nu_get_gpu_memory_used(NuRenderer* r);

/* Cmd-buffer cache introspection (perf/cmdbuf-cache).
 * Returns counts of replays vs. full re-records since renderer creation.
 * For a fully static scene with a fixed camera, replays should dominate
 * after the first frame. */
void         nu_get_cmd_cache_stats(NuRenderer* r,
                                    uint64_t* out_rt_replays,
                                    uint64_t* out_rt_records,
                                    uint64_t* out_tiled_replays,
                                    uint64_t* out_tiled_records);

/* Get human-readable error string for the last failed call. */
const char*  nu_get_last_error(NuRenderer* r);

/* ---- GPU phase timings (perf/vk-instrumentation) ----
 *
 * Timestamps captured via VkQueryPool around the major Vulkan submissions.
 * Wrapped in matching VK_EXT_debug_utils labels so nsys's
 * vulkan_gpu_marker_sum report mirrors the same regions.
 *
 * All values are in milliseconds. A field is 0.0 when the phase has not
 * run yet (e.g. blas_build_ms before nu_build_accel/nu_load_usd, or
 * rt_dispatch_ms before nu_render). One-shot phases (BLAS/TLAS/staging
 * uploads) record once at scene-build time; per-frame phases (RT dispatch,
 * pixel readback) are refreshed every nu_render + nu_fetch_pixels pair. */
typedef struct {
    float rt_dispatch_ms;             /* vkCmdTraceRaysKHR for single-camera RT */
    float pixel_readback_ms;          /* image -> staging copy in nu_fetch_pixels */
    float blas_build_ms;              /* triangle BLAS build (sum across batches) */
    float tlas_build_ms;              /* TLAS instance upload + build */
    float curve_blas_build_ms;        /* AABB BLAS build for BasisCurves */
    float staging_upload_segs_ms;     /* curve segment SSBO staging copy */
    float staging_upload_aabbs_ms;    /* curve AABB buffer staging copy */
    float staging_upload_colors_ms;   /* curve color SSBO staging copy */
    /* Phase C.4 mechanism hunt — per-dispatch GPU times in the tiled path.
     * Resolved inside gpu_wait_tiled_complete() after fence signal. */
    float trace_rays_tiled_ms;        /* vkCmdTraceRaysKHR for tiled multi-cam RT */
    float deferred_compute_ms;        /* deferred-shading vkCmdDispatch GPU time */
} NuPhaseTimings;

/* Copy the most recent phase timings into *out.
 * Returns NU_OK on success, NU_ERROR if timestamps are not supported on
 * this GPU (timestampComputeAndGraphics == false). Pass NULL to probe
 * for support without copying.
 *
 * Companion to VK_EXT_debug_utils labels - both are emitted from the same
 * sites so nsys's vulkan_gpu_marker_sum report matches these values. */
NuResult     nu_get_phase_timings_ms(NuRenderer* r, NuPhaseTimings* out);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_RENDERER_H */
