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

/* Same as nu_load_usd_from_handle but lets the caller hint a directory
 * for sidecar .mtlx lookup. NULL/empty `scene_dir` falls back to the
 * stage_label-derived directory (same behavior as nu_load_usd_from_handle).
 * NOTE: opengl backend currently stubs handle-based loading; this entry
 * point is provided for API parity and returns NU_ERROR until handle
 * loading is implemented. */
int          nu_load_usd_from_handle_with_dir(NuRenderer* r,
                                              void*       stage,
                                              const char* stage_label,
                                              const char* scene_dir);

/* Rebuild acceleration structures after adding/removing meshes.
 * Called automatically by nu_render() if scene is dirty,
 * but can be called explicitly to control timing. */
NuResult     nu_build_accel(NuRenderer* r);

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
 * (e.g. IsaacLab/Newton) where the default Y-up camera doesn't apply.
 * eye/target/up are world-space float[3]. */
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
 * Call after nu_render_tiled(). Returns NU_OK on success. */
NuResult     nu_get_cuda_interop_info(NuRenderer* r, NuCudaInteropInfo* out,
                                       int num_cameras, int tile_w, int tile_h);

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
 * `mesh_id` into `out_buf`, NUL-terminated. Returns the number of bytes
 * written (excluding NUL), or a negative NuResult on error. If buf_cap is
 * 0, returns the required size (excluding NUL). */
int          nu_get_mesh_name(NuRenderer* r, int mesh_id,
                              char* out_buf, int buf_cap);

/* Copy the renderer-canonical row-major column-vector world transform
 * (translation in [3], [7], [11]) for `mesh_id` into out_mat4x4[16].
 * Returns NU_OK on success. */
NuResult     nu_get_mesh_transform(NuRenderer* r, int mesh_id,
                                   float out_mat4x4[16]);

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
