// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * gpu_vulkan.c — Vulkan implementation of the gpu.h RHI
 *
 * Minimal but complete Vulkan backend for rendering indexed triangle meshes
 * with push constants and ray query shadows. Uses volk for function loading.
 */

#define VOLK_IMPLEMENTATION
#include <volk.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include "gpu.h"
#include "dlss.h"
#include "font8x16.h"
#include "stb_image.h"
#include <math.h>

#define MAX_FRAMES_IN_FLIGHT 2
#define PUSH_CONSTANT_SIZE      192  /* MVP+model+color+eye + ibl_params + tone_params */
#define PUSH_CONSTANT_SIZE_DLSS 224  /* + 64-byte prev_mvp (for motion vectors) */

/* Optional RTX Mega Geometry support needs very recent Vulkan headers. Keep the
 * normal raster/RT build working when the installed SDK predates these NV
 * extension declarations. */
#if defined(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME) && \
    defined(VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME) && \
    defined(VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME)
#define NUSD_HAS_VK_NV_MEGA_GEOMETRY 1
#else
#define NUSD_HAS_VK_NV_MEGA_GEOMETRY 0
#endif

/* ---- Internal structs ---- */

struct GpuBuffer_s {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    uint64_t       size;
};

struct GpuPipeline_s {
    VkPipeline       pipeline;
    VkPipelineLayout layout;    /* layout this pipeline was created with */
};

struct Gpu {
    /* Core */
    VkInstance       instance;
    VkSurfaceKHR     surface;
    VkPhysicalDevice physical_device;
    VkDevice         device;
    VkQueue          graphics_queue;
    uint32_t         queue_family;
    int              headless;  /* 1 = skip present (no compositor round-trip) */
    int              surfaceless;  /* 1 = no VkSurface/swapchain; render to owned images */

    /* Swapchain */
    VkSwapchainKHR   swapchain;
    VkFormat         swapchain_format;
    VkExtent2D       swapchain_extent;
    VkImage*         swapchain_images;
    VkImageView*     swapchain_views;
    VkDeviceMemory*  swapchain_memories;  /* non-NULL only for surfaceless images */
    uint32_t         image_count;
    uint32_t         current_image;

    /* Depth */
    VkImage          depth_image;
    VkDeviceMemory   depth_memory;
    VkImageView      depth_view;
    VkFormat         depth_format;

    /* Render pass */
    VkRenderPass     render_pass;

    /* Framebuffers */
    VkFramebuffer*   framebuffers;

    /* Commands */
    VkCommandPool    command_pool;
    VkCommandBuffer* command_buffers;
    VkCommandBuffer  current_cmd;

    /* Sync */
    VkSemaphore      image_available[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore      render_finished[MAX_FRAMES_IN_FLIGHT];
    VkFence          in_flight[MAX_FRAMES_IN_FLIGHT];
    uint32_t         frame_index;

    /* Shared pipeline layout */
    VkPipelineLayout pipeline_layout;

    /* Driver pipeline cache: persisted to disk so subsequent runs skip
     * SPIR-V → ISA compilation for already-seen pipelines. Path:
     * $XDG_CACHE_HOME/nusd_renderer/pipeline_cache.bin (or
     * $HOME/.cache/nusd_renderer/pipeline_cache.bin if XDG unset). */
    VkPipelineCache  pipeline_cache;
    char             pipeline_cache_path[1024];

    /* On-disk BLAS cache sidecar path (<usd>.nzblas), set by
     * gpu_set_blas_cache_path before gpu_build_rt_scene. Empty = disabled. */
    char             blas_cache_path[1024];

    /* Window size */
    int width;
    int height;

    /* ---- Ray tracing ---- */
    int rt_available;
    int rt_built;

    /* VK_NV_ray_tracing_invocation_reorder (Shader Execution Reordering, SER).
     * 1 if the extension is supported AND the device feature
     * `rayTracingInvocationReorder` is VK_TRUE. Reorder hint property
     * captured for diagnostic logging only — actual reorder behaviour is
     * driven by the SER-enabled rgen variant.
     *
     * Used by ensure_tiled_pipeline (renderer.c) to pick between the
     * regular and SER-enabled tiled raygen SPV. */
    int ser_available;
    int ser_reorder_hint;  /* VkRayTracingInvocationReorderModeNV cast to int */
    int barycentric_available;  /* VK_KHR_fragment_shader_barycentric enabled */

    /* RT properties */
    uint32_t rt_handle_size;
    uint32_t rt_handle_alignment;

    /* Per-mesh BLAS array */
    VkAccelerationStructureKHR* blas_list;
    VkBuffer                    blas_pool_buf;
    VkDeviceMemory              blas_pool_mem;
    VkBuffer                    blas_extra_pool_buf;
    VkDeviceMemory              blas_extra_pool_mem;
    uint32_t                    blas_count;

    /* TLAS */
    VkAccelerationStructureKHR tlas;
    VkBuffer                   tlas_buf;
    VkDeviceMemory             tlas_mem;
    VkBuffer                   instance_buf;
    VkDeviceMemory             instance_mem;

    /* Storage image (RT render target) */
    VkImage                    rt_image;
    VkDeviceMemory             rt_image_mem;
    VkImageView                rt_image_view;
    uint32_t                   rt_image_w, rt_image_h;

    /* RT pipeline */
    VkPipeline                 rt_pipeline;
    VkPipelineLayout           rt_pipeline_layout;

    /* Shader binding table */
    VkBuffer                   sbt_buf;
    VkDeviceMemory             sbt_mem;
    VkStridedDeviceAddressRegionKHR sbt_rgen;
    VkStridedDeviceAddressRegionKHR sbt_miss;
    VkStridedDeviceAddressRegionKHR sbt_hit;
    VkStridedDeviceAddressRegionKHR sbt_call;

    /* RT descriptors */
    VkDescriptorPool           rt_desc_pool;
    VkDescriptorSetLayout      rt_desc_layout;
    VkDescriptorSet            rt_desc_set;

    /* Scene data SSBO (vertex + index buffer addresses + colors) */
    VkBuffer                   scene_data_buf;
    VkDeviceMemory             scene_data_mem;
    uint32_t                   scene_data_nmeshes;
    int                        scene_up_axis;

    /* Phase A deferred-shading G-buffer (binding 17). 32 B per pixel.
     * For the simple RT pipeline a real device-local SSBO sized to
     * width*height is allocated; rchit (hit) + rmiss (primary miss)
     * write 8 dwords/pixel. Phase B's compute pass reads these. */
    VkBuffer                   gbuffer_buf;
    VkDeviceMemory             gbuffer_mem;
    VkBuffer                   gbuffer_staging_buf;
    VkDeviceMemory             gbuffer_staging_mem;
    void*                      gbuffer_staging_mapped;
    uint32_t                   gbuffer_w;
    uint32_t                   gbuffer_h;
    VkDeviceSize               gbuffer_size;

    /* Phase B: per-launch tiled G-buffer (binding 17 on the tiled set).
     * Sized to (tiled_image_w * tiled_image_h * 32 B); allocated lazily
     * the first time r->deferred_shade_enabled is observed. Phase A bound
     * binding 17 to `scene_data_buf` as a stub on the tiled descriptor
     * sets — gpu_tiled_ensure_gbuffer() rebinds to this real buffer. */
    VkBuffer                   tiled_gbuffer_buf;
    VkDeviceMemory             tiled_gbuffer_mem;
    uint32_t                   tiled_gbuffer_w;
    uint32_t                   tiled_gbuffer_h;
    VkDeviceSize               tiled_gbuffer_size;

    /* Phase B: deferred-shading compute pipeline. Owns its own descriptor
     * set layout / pool / sets (one per parity, mirroring tiled_rt_desc_set
     * + _b). The sets bind 2/9/17 — the rest of the bindings (textures,
     * IBL, lights) are not needed for the flat-shaded skeleton. */
    VkPipeline                 deferred_pipeline;
    VkPipelineLayout           deferred_pipeline_layout;
    VkDescriptorSetLayout      deferred_desc_layout;
    VkDescriptorPool           deferred_desc_pool;
    VkDescriptorSet            deferred_desc_set;
    VkDescriptorSet            deferred_desc_set_b;
    int                        deferred_built;

    /* Shadow descriptors (TLAS for ray query in fragment shader) */
    VkDescriptorPool           shadow_desc_pool;
    VkDescriptorSetLayout      shadow_desc_layout;
    VkDescriptorSet            shadow_desc_set;
    VkPipelineLayout           shadow_pipeline_layout;

    /* Current pipeline layout (for push constants) */
    VkPipelineLayout           current_layout;

    /* ---- DLSS offscreen rendering ---- */
    int dlss_active;

    /* Offscreen render targets (render resolution) */
    VkImage          dlss_color;
    VkDeviceMemory   dlss_color_mem;
    VkImageView      dlss_color_view;

    VkImage          dlss_mv;           /* RG16F motion vectors */
    VkDeviceMemory   dlss_mv_mem;
    VkImageView      dlss_mv_view;

    VkImage          dlss_depth;        /* R32F shader-readable depth */
    VkDeviceMemory   dlss_depth_mem;
    VkImageView      dlss_depth_view;

    VkImage          dlss_depth_att;    /* D32F depth attachment for render pass */
    VkDeviceMemory   dlss_depth_att_mem;
    VkImageView      dlss_depth_att_view;

    VkImage          dlss_output;       /* RGBA8 DLSS output (display resolution) */
    VkDeviceMemory   dlss_output_mem;
    VkImageView      dlss_output_view;

    /* Offscreen render pass: color(RGBA8) + MV(RG16F) + depth(D32F) */
    VkRenderPass     dlss_render_pass;
    VkFramebuffer    dlss_framebuffer;

    /* DLSS pipeline layout (208-byte push constants) */
    VkPipelineLayout dlss_pipeline_layout;

    /* DLSS RT pipeline + descriptors (5 bindings) */
    VkPipeline                 dlss_rt_pipeline;
    VkPipelineLayout           dlss_rt_pipeline_layout;
    VkDescriptorPool           dlss_rt_desc_pool;
    VkDescriptorSetLayout      dlss_rt_desc_layout;
    VkDescriptorSet            dlss_rt_desc_set;
    VkBuffer                   dlss_sbt_buf;
    VkDeviceMemory             dlss_sbt_mem;
    VkStridedDeviceAddressRegionKHR dlss_sbt_rgen;
    VkStridedDeviceAddressRegionKHR dlss_sbt_miss;
    VkStridedDeviceAddressRegionKHR dlss_sbt_hit;
    VkStridedDeviceAddressRegionKHR dlss_sbt_call;

    /* DLSS RT storage images at render resolution */
    VkImage          dlss_rt_color;
    VkDeviceMemory   dlss_rt_color_mem;
    VkImageView      dlss_rt_color_view;
    VkImage          dlss_rt_depth;
    VkDeviceMemory   dlss_rt_depth_mem;
    VkImageView      dlss_rt_depth_view;
    VkImage          dlss_rt_mv;
    VkDeviceMemory   dlss_rt_mv_mem;
    VkImageView      dlss_rt_mv_view;

    uint32_t         dlss_render_w, dlss_render_h;
    int              dlss_quality_mode;

    /* DLSS context (opaque, from dlss.h) */
    void*            dlss_ctx;        /* DlssContext* */

    /* ---- Text overlay ---- */
    VkImage          overlay_font_image;
    VkDeviceMemory   overlay_font_memory;
    VkImageView      overlay_font_view;
    VkSampler        overlay_font_sampler;
    VkDescriptorSetLayout overlay_desc_layout;
    VkDescriptorPool      overlay_desc_pool;
    VkDescriptorSet       overlay_desc_set;
    VkPipelineLayout      overlay_pipe_layout;
    VkPipeline            overlay_pipeline;
    VkRenderPass          overlay_render_pass;
    VkFramebuffer*        overlay_framebuffers;

    /* Dynamic vertex buffer for text quads (host-visible) */
    VkBuffer         overlay_vb;
    VkDeviceMemory   overlay_vb_mem;
    void*            overlay_vb_mapped;       /* persistent map */
    #define OVERLAY_MAX_VERTICES (16384)       /* 2730 chars max */
    float*           overlay_cpu_verts;        /* CPU staging (10 floats/vert) */
    uint32_t         overlay_vert_count;
    int              overlay_inited;

    /* Memory tracking */
    uint64_t         total_allocated_gpu_mem;

    /* ---- Material system ---- */
    int mat_uploaded;

    /* Material SSBO */
    VkBuffer                   mat_ssbo_buf;
    VkDeviceMemory             mat_ssbo_mem;
    int                        mat_count;
    /* 1 iff mat_count==1 because gpu_upload_materials was given 0 real
     * materials and synthesized a zeroed placeholder for descriptor
     * wiring. The shader's `hasMaterials` gate must read 0 in that case
     * so it falls back to per-mesh displayColor instead of reading
     * materials[0] (zeroed → black baseColor + 0 opacity → transmission). */
    int                        mat_only_placeholder;

    /* Textures */
    VkImage*                   mat_images;
    VkDeviceMemory*            mat_image_mems;
    VkImageView*               mat_image_views;
    VkSampler                  mat_sampler;
    int                        mat_tex_count;

    /* Material descriptors */
    VkDescriptorPool           mat_desc_pool;
    VkDescriptorSetLayout      mat_desc_layout;
    VkDescriptorSet            mat_desc_set;
    VkPipelineLayout           mat_pipeline_layout;

    /* ---- Scene lights (RectLight + DistantLight) — binding 13 of tiled RT set ---- */
    VkBuffer                   light_ssbo_buf;
    VkDeviceMemory             light_ssbo_mem;
    int                        light_count;

    /* Real material colors sampled from authored Ptex maps. Packed RGBA8,
     * three uints per triangulated mesh primitive, indexed by MeshData offset. */
    VkBuffer                   rt_tri_color_ssbo_buf;
    VkDeviceMemory             rt_tri_color_ssbo_mem;
    uint32_t                   rt_tri_color_count;

    /* ---- Phase 11.A: BasisCurves data ----
     * curve_seg_ssbo_buf : per-segment data (32 B SceneCurveSegment) — SSBO
     *                      consumed by intersection + closest-hit shaders,
     *                      indexed by gl_PrimitiveID.
     * curve_color_ssbo_buf : per-segment color (12 B float3) — SSBO,
     *                      indexed by gl_PrimitiveID.
     * curve_aabb_buf     : VkAabbPositionsKHR (24 B) — BLAS build input.
     * Phase 12.2 will replace `curve_color_ssbo` with a material_id +
     * materials[] LUT once the shader-side material lookup is in place. */
    VkBuffer                   curve_seg_ssbo_buf;
    VkDeviceMemory             curve_seg_ssbo_mem;
    VkBuffer                   curve_color_ssbo_buf;
    VkDeviceMemory             curve_color_ssbo_mem;
    VkBuffer                   curve_aabb_buf;
    VkDeviceMemory             curve_aabb_mem;
    int                        curve_seg_count;

    /* Curve BLAS: single VK_GEOMETRY_TYPE_AABBS_KHR acceleration structure
     * built over all SceneCurveSegment AABBs. Phase 12.1's "single flat
     * BLAS" strategy applied at the smallest scale (single curve prim or
     * tens of segments) — same code shape will scale to 10M+ at minimal
     * change. */
    VkAccelerationStructureKHR curve_blas;
    VkBuffer                   curve_blas_buf;
    VkDeviceMemory             curve_blas_mem;

    /* ---- Phase 12.x: GPU-side curve AABB generation ----
     * A compute pipeline that derives per-segment AABBs from the
     * curve segment SSBO. Built lazily on first use; lives in the gpu
     * struct so we don't re-create it on every BLAS rebuild. The
     * descriptor set is updated each call (segment + AABB buffers
     * change when seg_count changes). */
    VkPipeline                 curve_aabb_gen_pipeline;
    VkPipelineLayout           curve_aabb_gen_pl_layout;
    VkDescriptorSetLayout      curve_aabb_gen_ds_layout;
    VkDescriptorPool           curve_aabb_gen_ds_pool;
    VkDescriptorSet            curve_aabb_gen_ds_set;
    int                        curve_aabb_gen_built;

    /* GPU timestamp pool for AABB-gen dispatch perf measurement.
     * 2 timestamps: before and after the dispatch. Period (ns/tick) is
     * cached in curve_aabb_gen_ts_period so the host can compute
     * dispatch-only ms. */
    VkQueryPool                curve_aabb_gen_ts_pool;
    double                     curve_aabb_gen_ts_period;
    double                     curve_aabb_gen_last_ms;

    /* ---- IBL (environment map + BRDF LUT) ---- */
    VkImage          env_image;
    VkDeviceMemory   env_image_mem;
    VkImageView      env_image_view;
    VkSampler        env_sampler;
    int              env_mip_levels;
    /* USD-authored DomeLight intensity, applied as a separate multiplier
     * in the rmiss shader so the sky saturates like ovrtx without
     * over-brightening surfaces (rchit) which use env_scale-normalized
     * irradiance. Default 1.0; set by gpu_load_environment_intensity. */
    float            env_intensity;
    /* Flat DomeLight color (RGB) + intensity (A) for the fast_mode rmiss
     * sky and rchit hemispheric ambient. Pushed into the SceneData SSBO
     * header at offset 16. Default near-white at intensity 1.0. */
    float            dome_color[4];
    float            tone_exposure_scale;
    float            tone_sky_scale;
    float            tone_white_point;
    uint32_t         tone_flags;
    /* Per-mesh texture index (uint32, 0xFFFFFFFF = none) staged for the
     * next gpu_build_rt_scene SSBO upload + patched in-place when the SSBO
     * is already live. Caller writes via gpu_set_mesh_texture(). */
    uint32_t*        mesh_tex_indices;
    uint32_t         mesh_tex_indices_count;
    uint32_t         mesh_tex_indices_capacity;

    VkImage          brdf_lut_image;
    VkDeviceMemory   brdf_lut_mem;
    VkImageView      brdf_lut_view;
    VkSampler        brdf_lut_sampler;

    VkImage          irr_image;
    VkDeviceMemory   irr_image_mem;
    VkImageView      irr_image_view;
    VkSampler        irr_sampler;

    int              ibl_loaded;

    /* Environment background */
    VkPipeline       env_bg_pipeline;

    /* TLAS rebuild data (persisted from initial build for transform updates) */
    uint32_t         tlas_instance_count;
    VkDeviceAddress* blas_addresses;  /* [blas_count] device addresses */
    uint32_t*        mesh_to_blas;    /* [tlas_instance_count] maps instance → BLAS index */
    uint32_t*        instance_custom; /* [tlas_instance_count] custom index (mesh id in SSBO) */
    uint8_t*         instance_mask;   /* [nmeshes] Phase A: per-mesh TLAS mask byte; NULL = all 0xFF */
    uint32_t         instance_mask_count; /* count of entries valid in instance_mask */
    VkDeviceSize     tlas_size;       /* tlas accel struct size (for rebuild) */

    /* Phase C: per-env TLAS array. When num_envs > 0, gpu_build_rt_scene
     * builds (in addition to the legacy single TLAS in `tlas` above)
     * `num_envs` per-env TLASes, each containing only that env's
     * instances. Bound as a TLAS-array descriptor (with UAB) on the
     * partitioned-pipeline binding; the tiled raygen indexes by
     * cam_env_idx. The legacy `tlas` field stays valid as a fallback
     * for non-partitioned (visualizer / raster / single-cam RT) paths.
     *
     * mesh_to_env  : [tlas_instance_count] env_idx per TLAS instance,
     *                or -1 for a static global (visible to every env).
     * env_inst_idx : [num_envs+1] prefix-sum offsets into a stable
     *                instance order; instances for env e occupy
     *                env_inst_idx[e] .. env_inst_idx[e+1] - 1.
     * inst_to_partial : [tlas_instance_count] index into the partitioned
     *                instance order (so we can reuse the same warp
     *                transforms array as the legacy path).
     * tlas_arr     : [num_envs] one VkAccelerationStructure per env.
     * tlas_arr_buf : [num_envs] backing storage buffer per env.
     * tlas_arr_mem : single backing memory allocation; per-env offsets.
     * tlas_arr_inst_buf / mem : single big instance buffer holding all
     *                envs' instances laid out per env_inst_idx[]. Per-env
     *                TLAS build references its slice via deviceAddress +
     *                env_inst_idx[e] * sizeof(instance). */
    int              num_envs;
    int*             mesh_to_env;
    int              mesh_to_env_count;  /* length of mesh_to_env (== nmeshes) */
    uint32_t*        env_inst_idx;     /* [num_envs+1] prefix sum */
    uint32_t*        inst_to_partial;  /* [tlas_instance_count] */
    VkAccelerationStructureKHR* tlas_arr;
    VkBuffer*        tlas_arr_buf;
    VkDeviceMemory   tlas_arr_mem;
    VkDeviceSize*    tlas_arr_buf_offset;
    VkBuffer         tlas_arr_inst_buf;
    VkDeviceMemory   tlas_arr_inst_mem;
    void*            tlas_arr_inst_mapped;
    VkDeviceSize     tlas_arr_inst_size;
    VkBuffer         tlas_arr_scratch_buf;
    VkDeviceMemory   tlas_arr_scratch_mem;
    VkDeviceSize     tlas_arr_scratch_size;
    uint32_t         tlas_arr_total_inst;  /* total instances across all envs */
    int              tlas_arr_built;
    VkDeviceSize     min_as_scratch_align; /* minAccelerationStructureScratchOffsetAlignment */
    VkDeviceSize     tlas_scratch_size;       /* build scratch size */
    VkDeviceSize     tlas_update_scratch_size; /* update scratch size (may differ) */

    /* Persistent buffers for TLAS updates (reused across frames) */
    VkBuffer         tlas_update_scratch_buf;
    VkDeviceMemory   tlas_update_scratch_mem;
    VkBuffer         tlas_update_staging_buf;
    VkDeviceMemory   tlas_update_staging_mem;
    void*            tlas_update_staging_mapped;  /* persistent map */
    VkDeviceSize     tlas_update_staging_size;

    /* Persistent readback staging buffer (reused across frames) */
    VkBuffer         readback_buf;
    VkDeviceMemory   readback_mem;
    void*            readback_mapped;    /* persistent map */
    uint32_t         readback_w, readback_h;
    int              readback_coherent;  /* 1 = coherent (no invalidation needed) */

    /* Single-camera CUDA interop pixels buffer (exportable, device-local,
     * BGRA8). Mirrors readback_buf but uses VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
     * so CUDA can import via cuImportExternalMemory and read with cuMemcpy. */
    VkBuffer         pixels_interop_buf;
    VkDeviceMemory   pixels_interop_mem;
    VkDeviceSize     pixels_interop_alloc_size;   /* req.size from vkGetBufferMemoryRequirements */
    VkDeviceSize     pixels_interop_logical_size; /* w * h * 4 */
    uint32_t         pixels_interop_w, pixels_interop_h;
    int              pixels_interop_fd;           /* cached fd; -1 = not yet exported */

    /* ---- Tiled multi-camera RT ---- */
    VkImage          tiled_image;
    VkDeviceMemory   tiled_image_mem;
    VkImageView      tiled_image_view;
    uint32_t         tiled_image_w, tiled_image_h;

    VkBuffer         tiled_camera_buf;       /* Camera SSBO: N * (view_inv + proj_inv) */
    VkDeviceMemory   tiled_camera_mem;
    void*            tiled_camera_mapped;     /* persistent map */
    int              tiled_camera_count;

    VkPipeline       tiled_rt_pipeline;
    VkPipelineLayout tiled_rt_pipeline_layout;
    VkDescriptorPool tiled_rt_desc_pool;
    VkDescriptorSetLayout tiled_rt_desc_layout;
    VkDescriptorSet  tiled_rt_desc_set;       /* primary descriptor set */
    VkDescriptorSet  tiled_rt_desc_set_b;     /* secondary set: binding 9 → interop_buf[1] */
    VkBuffer         tiled_sbt_buf;
    VkDeviceMemory   tiled_sbt_mem;
    VkStridedDeviceAddressRegionKHR tiled_sbt_rgen;
    VkStridedDeviceAddressRegionKHR tiled_sbt_miss;
    VkStridedDeviceAddressRegionKHR tiled_sbt_hit;
    VkStridedDeviceAddressRegionKHR tiled_sbt_call;
    int              tiled_rt_built;

    /* Double-buffered tiled readback staging (async overlap) */
    VkBuffer         tiled_readback_buf[2];
    VkDeviceMemory   tiled_readback_mem[2];
    void*            tiled_readback_mapped[2];
    VkFence          tiled_readback_fence[2];
    VkCommandBuffer  tiled_readback_cmd[2];   /* kept alive until fence signals */
    int              tiled_readback_submitted[2]; /* 1 = fence in flight */
    int              tiled_readback_write_idx;    /* next slot to write (flips 0↔1) */
    uint32_t         tiled_readback_w, tiled_readback_h;
    int              tiled_readback_coherent; /* 1 = coherent (no invalidation needed) */

    /* CUDA-Vulkan interop (Phase 2: zero-copy, double-buffered) */
    int              interop_available;        /* 1 = external memory + semaphore extensions enabled */
    VkDeviceSize     tiled_image_alloc_size;   /* allocation size (image, for reference) */
    VkBuffer         interop_buf[2];           /* device-local linear buffers (exportable, double-buffered) */
    VkDeviceMemory   interop_buf_mem[2];       /* exportable device memory for the linear buffers */
    VkDeviceSize     interop_buf_size;         /* size of each linear buffer (both same size) */
    int              interop_write_idx;        /* next interop buffer to write (flips 0↔1) */
    int              external_buffers_active;  /* Hybrid C: 1 = interop_buf_mem imported from
                                                * caller-owned CUDA VMM allocation; tiled_create_interop_buffer
                                                * skips export-allocation; gpu_export_tiled_image_fd refuses */
    VkSemaphore      interop_timeline_sem;     /* exportable timeline semaphore */
    uint64_t         interop_timeline_value;   /* monotonically increasing per render */
    int              skip_staging_copy;        /* 1 = skip CPU staging copy (CUDA-only mode) */
    int              direct_write_active;     /* 1 = raygen shader writes SSBO instead of imageStore */

    /* Depth output (float32 per pixel, written by CHL/miss shaders) */
    VkBuffer         tiled_depth_buf;         /* device-local SSBO for depth output */
    VkDeviceMemory   tiled_depth_mem;
    VkBuffer         tiled_depth_staging;     /* host-visible staging for CPU readback */
    VkDeviceMemory   tiled_depth_staging_mem;
    void*            tiled_depth_staging_mapped;
    int              tiled_depth_enabled;     /* 1 = depth output active */
    uint32_t         tiled_depth_w;           /* dims the depth buffer was sized for */
    uint32_t         tiled_depth_h;

    /* Segmentation output (uint32 per pixel, written by CHL/miss shaders) */
    VkBuffer         tiled_seg_buf;           /* device-local SSBO for segmentation IDs */
    VkDeviceMemory   tiled_seg_mem;
    VkBuffer         tiled_seg_staging;       /* host-visible staging for CPU readback */
    VkDeviceMemory   tiled_seg_staging_mem;
    void*            tiled_seg_staging_mapped;
    int              tiled_seg_enabled;       /* 1 = segmentation output active */
    uint32_t         tiled_seg_w;             /* dims the segmentation buffer was sized for */
    uint32_t         tiled_seg_h;

    /* Normals output (float32×3 per pixel, written by CHL/miss shaders) */
    VkBuffer         tiled_norm_buf;          /* device-local SSBO for normals */
    VkDeviceMemory   tiled_norm_mem;
    VkBuffer         tiled_norm_staging;      /* host-visible staging for CPU readback */
    VkDeviceMemory   tiled_norm_staging_mem;
    void*            tiled_norm_staging_mapped;
    int              tiled_norm_enabled;      /* 1 = normals output active */
    uint32_t         tiled_norm_w;            /* dims the normals buffer was sized for */
    uint32_t         tiled_norm_h;

    /* ---- Raycast compute pipeline (LiDAR/radar) ---- */
    VkPipeline                 raycast_pipeline;
    VkPipelineLayout           raycast_pipeline_layout;
    VkDescriptorPool           raycast_desc_pool;
    VkDescriptorSetLayout      raycast_desc_layout;
    VkDescriptorSet            raycast_desc_set;
    VkBuffer                   raycast_input_buf;     /* device-local SSBO: origins + directions */
    VkDeviceMemory             raycast_input_mem;
    VkBuffer                   raycast_output_buf;    /* device-local SSBO: distances + meshIds + normals + hitPos */
    VkDeviceMemory             raycast_output_mem;
    VkBuffer                   raycast_staging_in;    /* host-visible staging for upload */
    VkDeviceMemory             raycast_staging_in_mem;
    void*                      raycast_staging_in_mapped;
    VkBuffer                   raycast_staging_out;   /* host-visible staging for readback */
    VkDeviceMemory             raycast_staging_out_mem;
    void*                      raycast_staging_out_mapped;
    int                        raycast_built;
    uint32_t                   raycast_max_rays;      /* current buffer capacity */

    /* ---- Async raycast state ---- */
    VkCommandBuffer            raycast_async_cmd;     /* pending async command buffer */
    VkFence                    raycast_async_fence;    /* signaled when async dispatch completes */
    int                        raycast_async_pending;  /* 1 = async dispatch in flight */
    uint32_t                   raycast_async_num_rays; /* rays in pending dispatch */

    /* ---- Raycast CUDA interop ---- */
    int                        raycast_exportable;     /* 1 = buffers created with export flag */
    uint64_t                   raycast_input_alloc_size;
    uint64_t                   raycast_output_alloc_size;

    /* ---- GPU phase timings (perf/vk-instrumentation) ---- */
    int              debug_utils_enabled;
    VkDebugUtilsMessengerEXT debug_messenger;  /* NUSD_VK_VALIDATION=1 */
    int              cluster_as_available;  /* RTX Mega Geometry feature flags */
    int              ptlas_available;
    int              lss_available;
    /* Phase 0 cluster-AS (NUSD_RT_CLUSTER): backing buffers for opt-in cluster
     * BLASes, kept alive until gpu_destroy_rt_scene. */
    VkBuffer*        cluster_as_bufs;
    VkDeviceMemory*  cluster_as_mems;
    int              cluster_as_count;
    int              cluster_as_cap;
    int              timestamps_supported;
    float            timestamp_period_ns;
    VkQueryPool      timestamp_pool;
    uint64_t         phase_timing_ns[GPU_PHASE_COUNT];
    uint8_t          phase_pending[GPU_PHASE_COUNT];

    /* ---- Pre-recorded command buffer cache (perf/cmdbuf-cache) ---- */
    int                        rt_cmd_cache_dirty;
    int                        rt_cmd_cache_valid_count;
    int*                       rt_cmd_cache_valid;
    GpuRtPushConstants*        rt_cmd_cache_pc;
    int                        rt_cmd_replay_active;
    int                        tiled_cmd_replay_active;
    int                        tiled_cmd_replay_idx;
    int                        tiled_cmd_has_tlas_update;
    VkCommandBuffer            tiled_cached_cmd[2];
    int                        tiled_cmd_cache_valid[2];
    GpuRtTiledPushConstants    tiled_cmd_cache_pc[2];
    int                        tiled_cmd_cache_direct_write[2];
    int                        tiled_cmd_cache_dirty;
    uint64_t                   rt_cmd_cache_replays;
    uint64_t                   rt_cmd_cache_records;
    uint64_t                   tiled_cmd_cache_replays;
    uint64_t                   tiled_cmd_cache_records;

    /* ---- perf/mem-pool: GPU memory sub-allocator pools ---- */
    void*            resident_pool;
    void*            transient_pool;
    void*            staging_pool;

    /* ---- GPU-driven TLAS instance translation (PR 2) ----
     *
     * The warp transform kernel writes a dense (n_valid, 16) row-major
     * matrix array into `tlas_xforms_buf`, an exportable storage buffer
     * imported into CUDA so warp can write it directly. A compute
     * dispatch then reads each thread's 4x4 and copies the upper 3x4 (12
     * floats) into `instance_buf` at the slot pointed to by
     * `tlas_indices_buf` — leaving bytes 48..63 of each
     * VkAccelerationStructureInstanceKHR record untouched (those were
     * authored by the initial host-side gpu_update_tlas() at scene init
     * and stay valid).
     *
     * Lifecycle: created lazily on first nu_get_transforms_interop_info()
     * call; destroyed in gpu_shutdown(). Each fd export is caller-owned:
     * CUDA consumes the fd on cuImportExternalMemory, or the caller closes
     * it if import is not attempted. */
    VkBuffer                   tlas_xforms_buf;       /* exportable, device-local, n_valid*64 bytes */
    VkDeviceMemory             tlas_xforms_mem;
    VkDeviceSize               tlas_xforms_size;      /* allocated size (req.size) */
    VkDeviceSize               tlas_xforms_logical;   /* n_valid*64 (the meaningful payload) */
    int                        tlas_xforms_count;     /* n_valid the buffer was sized for */
    int                        tlas_xforms_fd;        /* legacy; kept -1 for ABI-private struct stability */

    VkBuffer                   tlas_indices_buf;      /* device-local int[count]; gid → tlas_idx */
    VkDeviceMemory             tlas_indices_mem;
    int                        tlas_indices_count;    /* current upload size */

    VkPipeline                 tlas_translate_pipeline;
    VkPipelineLayout           tlas_translate_pl_layout;
    VkDescriptorSetLayout      tlas_translate_ds_layout;
    VkDescriptorPool           tlas_translate_ds_pool;
    VkDescriptorSet            tlas_translate_ds_set;
    int                        tlas_translate_built;
    int                        tlas_translate_ds_dirty;  /* 1 = needs vkUpdateDescriptorSets before next dispatch */

    /* Phase C: partitioned-translate pipeline. Same shader pattern as
     * tlas_translate but with a 4th binding (inst_to_partial) and writes
     * into gpu->tlas_arr_inst_buf. */
    VkPipeline                 tlas_translate_p_pipeline;
    VkPipelineLayout           tlas_translate_p_pl_layout;
    VkDescriptorSetLayout      tlas_translate_p_ds_layout;
    VkDescriptorPool           tlas_translate_p_ds_pool;
    VkDescriptorSet            tlas_translate_p_ds_set;
    int                        tlas_translate_p_built;
    int                        tlas_translate_p_ds_dirty;
    VkBuffer                   inst_to_partial_buf;
    VkDeviceMemory             inst_to_partial_mem;
    uint32_t                   inst_to_partial_count;

    /* ---- VK3DGRT (3D Gaussian Ray Tracing) ----
     *
     * Splat-scene state, parallel to the mesh BLAS/TLAS path. See
     * docs/plans/VK3DGRT_PLAN.md and src/gs_scene.c. Phase 2/3.
     *
     * Lifecycle:
     *   gpu_gs_upload_particles(...) → fills SSBOs from NuGsScene host arrays.
     *   gpu_gs_build_accel(...)      → builds unit BLAS once per proxy kind,
     *                                   dispatches gs_as_build to fill the
     *                                   TLAS instance buffer, builds TLAS.
     *   gpu_gs_render_tile(...)      → traces rays via the gs RT pipeline
     *                                   (Phase 3, K-buffer SoA SSBO).
     *   gpu_gs_destroy(...)          → tears down all gs resources. */
    int              gs_built;            /* 1 iff TLAS built and ready */
    int              gs_pipeline_built;   /* 1 iff RT pipeline ready */
    int              gs_proxy_kind;       /* 0=icosa, 1=AABB; matches NuGsProxyKind */
    uint32_t         gs_particle_count;
    int              gs_sh_degree;        /* 0..3 */

    /* Unit BLAS (one per proxy kind, lazily built and reused). */
    VkAccelerationStructureKHR gs_blas_icosa;
    VkBuffer                   gs_blas_icosa_buf;
    VkDeviceMemory             gs_blas_icosa_mem;
    VkDeviceAddress            gs_blas_icosa_addr;
    VkBuffer                   gs_icosa_vb;       /* 12 verts × float3 */
    VkDeviceMemory             gs_icosa_vb_mem;
    VkBuffer                   gs_icosa_ib;       /* 60 indices */
    VkDeviceMemory             gs_icosa_ib_mem;

    VkAccelerationStructureKHR gs_blas_aabb;
    VkBuffer                   gs_blas_aabb_buf;
    VkDeviceMemory             gs_blas_aabb_mem;
    VkDeviceAddress            gs_blas_aabb_addr;
    VkBuffer                   gs_aabb_buf;       /* one VkAabbPositionsKHR */
    VkDeviceMemory             gs_aabb_buf_mem;

    /* Particle SSBOs (filled by host from NuGsScene). */
    VkBuffer                   gs_pos_buf;        /* float3 × N */
    VkDeviceMemory             gs_pos_mem;
    VkBuffer                   gs_scale_buf;      /* float3 × N */
    VkDeviceMemory             gs_scale_mem;
    VkBuffer                   gs_quat_buf;       /* float4 × N (wxyz) */
    VkDeviceMemory             gs_quat_mem;
    VkBuffer                   gs_opa_buf;        /* float × N */
    VkDeviceMemory             gs_opa_mem;
    VkBuffer                   gs_kerS_buf;       /* float × N */
    VkDeviceMemory             gs_kerS_mem;
    VkBuffer                   gs_sh_buf;         /* float × N × (sh_degree+1)^2 × 3 */
    VkDeviceMemory             gs_sh_mem;
    uint32_t                   gs_particle_capacity; /* current SSBO sizing */

    /* TLAS + GPU-built instance buffer. */
    VkAccelerationStructureKHR gs_tlas;
    VkBuffer                   gs_tlas_buf;
    VkDeviceMemory             gs_tlas_mem;
    VkDeviceSize               gs_tlas_size;
    VkBuffer                   gs_tlas_scratch_buf;
    VkDeviceMemory             gs_tlas_scratch_mem;
    VkDeviceSize               gs_tlas_scratch_size;
    VkBuffer                   gs_inst_buf;       /* VkAccelerationStructureInstanceKHR × N */
    VkDeviceMemory             gs_inst_mem;
    VkDeviceSize               gs_inst_capacity;  /* current size in bytes */

    /* gs_as_build compute pipeline (TLAS-instance generator). */
    VkPipeline                 gs_as_pipeline;
    VkPipelineLayout           gs_as_pl_layout;
    VkDescriptorSetLayout      gs_as_ds_layout;
    VkDescriptorPool           gs_as_ds_pool;
    VkDescriptorSet            gs_as_ds;
    int                        gs_as_built;

    /* Phase 3: RT pipeline state — populated by gpu_gs_build_pipeline. */
    VkPipeline                 gs_rt_pipeline;
    VkPipelineLayout           gs_rt_pl_layout;
    VkDescriptorSetLayout      gs_rt_ds_layout;
    VkDescriptorPool           gs_rt_ds_pool;
    VkDescriptorSet            gs_rt_ds;
    VkBuffer                   gs_sbt_buf;
    VkDeviceMemory             gs_sbt_mem;
    VkStridedDeviceAddressRegionKHR gs_sbt_rgen;
    VkStridedDeviceAddressRegionKHR gs_sbt_miss;
    VkStridedDeviceAddressRegionKHR gs_sbt_hit;
    VkStridedDeviceAddressRegionKHR gs_sbt_call;

    /* Per-pixel SoA K-buffer (plan §6 D3). Sized W * H * K bytes per
     * channel; lazily reallocated when tile size or K changes. The
     * shader hardcodes K = GS_KBUF_K (see src/shaders/slang/gs_common.h.slang). */
    VkBuffer                   gs_kbuf_id_buf;
    VkDeviceMemory             gs_kbuf_id_mem;
    VkBuffer                   gs_kbuf_dist_buf;
    VkDeviceMemory             gs_kbuf_dist_mem;
    uint32_t                   gs_kbuf_w, gs_kbuf_h, gs_kbuf_k;

    /* Iso-opacity surface depth (plan §7): per-pixel float32, gives the
     * t-value where accumulated opacity first crosses
     * `iso_opacity_threshold` (default 0.5). -1 = miss / never crossed. */
    VkBuffer                   gs_depth_buf;
    VkDeviceMemory             gs_depth_mem;
    uint32_t                   gs_depth_w, gs_depth_h;

    /* Iso-opacity surface normal (plan §7): per-pixel float32 × 3.
     * Analytic gradient of accumulated opacity at the iso crossing,
     * normalized. (0,0,0) = miss (no iso crossing). */
    VkBuffer                   gs_normal_buf;
    VkDeviceMemory             gs_normal_mem;
    uint32_t                   gs_normal_w, gs_normal_h;
};

/* ---- Helpers ---- */

/* Wrapper around vkAllocateMemory that tracks total allocation size */
static VkResult gpu_alloc_memory(Gpu* gpu, const VkMemoryAllocateInfo* ai,
                                 VkDeviceMemory* out)
{
    VkResult r = vkAllocateMemory(gpu->device, ai, NULL, out);
    if (r == VK_SUCCESS)
        gpu->total_allocated_gpu_mem += ai->allocationSize;
    return r;
}

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits,
                                 VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }

    fprintf(stderr, "gpu_vulkan: failed to find suitable memory type\n");
    return 0;
}

/* Find best memory type for CPU readback: prefer HOST_CACHED for fast reads.
 * Falls back to HOST_VISIBLE | HOST_COHERENT if cached not available.
 * Sets *is_coherent to indicate whether explicit invalidation is needed. */
static uint32_t find_readback_memory_type(VkPhysicalDevice phys, uint32_t type_bits,
                                           int* is_coherent)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);

    /* Best: HOST_VISIBLE | HOST_CACHED | HOST_COHERENT (no invalidation needed) */
    VkMemoryPropertyFlags ideal = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & ideal) == ideal) {
            *is_coherent = 1;
            return i;
        }
    }

    /* Good: HOST_VISIBLE | HOST_CACHED (needs invalidation before read) */
    VkMemoryPropertyFlags cached = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & cached) == cached) {
            *is_coherent = 0;
            return i;
        }
    }

    /* Fallback: HOST_VISIBLE | HOST_COHERENT (slow uncached reads) */
    *is_coherent = 1;
    return find_memory_type(phys, type_bits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

static VkShaderModule create_shader_module(VkDevice device,
                                           const uint32_t* code,
                                           uint32_t size)
{
    VkShaderModuleCreateInfo ci = {0};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = size;
    ci.pCode    = code;

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, NULL, &module) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create shader module\n");
    }
    return module;
}

/* ---- VkPipelineCache: persisted to disk so RT/graphics/compute pipelines
 * skip SPIR-V → ISA recompilation on every fresh process start. The blob
 * is keyed on vendorID + deviceID + pipelineCacheUUID; a header mismatch
 * (driver upgrade, different GPU, etc.) just means we discard and start
 * empty. Bytes after the 32-byte header are opaque driver-private data.
 *
 * Layout: $XDG_CACHE_HOME/nusd_renderer/pipeline_cache.bin
 *         (falls back to $HOME/.cache/nusd_renderer/pipeline_cache.bin)
 */

static void pc_resolve_path(char* out, size_t out_size)
{
    const char* xdg = getenv("XDG_CACHE_HOME");
    const char* home = getenv("HOME");
    if (xdg && xdg[0]) {
        snprintf(out, out_size, "%s/nusd_renderer/pipeline_cache.bin", xdg);
    } else if (home && home[0]) {
        snprintf(out, out_size, "%s/.cache/nusd_renderer/pipeline_cache.bin", home);
    } else {
        /* Last resort */
        snprintf(out, out_size, "/tmp/nusd_renderer_pipeline_cache.bin");
    }
}

/* mkdir -p semantics: ensure the parent directory of `path` exists.
 * Walks the path, creating each segment with mode 0700. Returns 1 on
 * success (or already exists), 0 on failure. */
static int pc_mkdir_parents(const char* path)
{
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    /* Strip filename — just want the directory part. */
    char* slash = strrchr(tmp, '/');
    if (!slash) return 1; /* no directory component */
    *slash = '\0';
    if (tmp[0] == '\0') return 1;

    /* Walk components and mkdir each. */
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
                fprintf(stderr,
                    "gpu_vulkan: pipeline cache: mkdir %s failed (%s)\n",
                    tmp, strerror(errno));
                return 0;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
        fprintf(stderr,
            "gpu_vulkan: pipeline cache: mkdir %s failed (%s)\n",
            tmp, strerror(errno));
        return 0;
    }
    return 1;
}

/* Validate a cache blob's 32-byte header against the current physical
 * device's vendorID / deviceID / pipelineCacheUUID. Returns 1 if the
 * blob is safe to feed to vkCreatePipelineCache, 0 otherwise. */
static int pc_validate_header(Gpu* gpu, const uint8_t* data, size_t size)
{
    if (size < 32) return 0;

    /* The header is described in the spec as a packed layout of:
     *   uint32 headerSize (=32)
     *   uint32 headerVersion (=1, VK_PIPELINE_CACHE_HEADER_VERSION_ONE)
     *   uint32 vendorID
     *   uint32 deviceID
     *   uint8  pipelineCacheUUID[16]
     * We can't just memcpy a VkPipelineCacheHeaderVersionOne over it
     * because the headerVersion enum's underlying width is impl-defined
     * — pull each field out by offset. */
    uint32_t header_size, header_version, vendor_id, device_id;
    memcpy(&header_size,    data + 0,  4);
    memcpy(&header_version, data + 4,  4);
    memcpy(&vendor_id,      data + 8,  4);
    memcpy(&device_id,      data + 12, 4);
    const uint8_t* uuid = data + 16;

    if (header_size != 32) {
        fprintf(stderr,
            "gpu_vulkan: pipeline cache: header_size=%u, expected 32 — discarding\n",
            header_size);
        return 0;
    }
    if (header_version != VK_PIPELINE_CACHE_HEADER_VERSION_ONE) {
        fprintf(stderr,
            "gpu_vulkan: pipeline cache: header_version=%u, expected %u — discarding\n",
            header_version, (uint32_t)VK_PIPELINE_CACHE_HEADER_VERSION_ONE);
        return 0;
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu->physical_device, &props);

    if (vendor_id != props.vendorID) {
        fprintf(stderr,
            "gpu_vulkan: pipeline cache: vendorID 0x%x != device 0x%x — discarding\n",
            vendor_id, props.vendorID);
        return 0;
    }
    if (device_id != props.deviceID) {
        fprintf(stderr,
            "gpu_vulkan: pipeline cache: deviceID 0x%x != device 0x%x — discarding\n",
            device_id, props.deviceID);
        return 0;
    }
    if (memcmp(uuid, props.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
        fprintf(stderr,
            "gpu_vulkan: pipeline cache: pipelineCacheUUID mismatch (driver upgrade?) — discarding\n");
        return 0;
    }
    return 1;
}

/* Create the cache, populating it from disk if a valid blob exists.
 * Always succeeds in producing a usable cache handle: on any error the
 * cache is created empty and pipelines will simply compile from scratch. */
static void pc_create(Gpu* gpu)
{
    pc_resolve_path(gpu->pipeline_cache_path,
                    sizeof(gpu->pipeline_cache_path));

    uint8_t* blob = NULL;
    size_t   blob_size = 0;

    FILE* f = fopen(gpu->pipeline_cache_path, "rb");
    if (f) {
        if (fseek(f, 0, SEEK_END) == 0) {
            long sz = ftell(f);
            if (sz > 0 && sz < (long)(256 * 1024 * 1024)) { /* sanity cap 256 MB */
                fseek(f, 0, SEEK_SET);
                blob = (uint8_t*)malloc((size_t)sz);
                if (blob) {
                    size_t got = fread(blob, 1, (size_t)sz, f);
                    if (got == (size_t)sz) {
                        blob_size = (size_t)sz;
                    } else {
                        free(blob);
                        blob = NULL;
                    }
                }
            }
        }
        fclose(f);
    }

    int use_blob = (blob && pc_validate_header(gpu, blob, blob_size));

    VkPipelineCacheCreateInfo ci = {0};
    ci.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    ci.initialDataSize = use_blob ? blob_size : 0;
    ci.pInitialData    = use_blob ? blob      : NULL;

    VkResult r = vkCreatePipelineCache(gpu->device, &ci, NULL,
                                       &gpu->pipeline_cache);
    if (r != VK_SUCCESS) {
        fprintf(stderr,
            "gpu_vulkan: pipeline cache: vkCreatePipelineCache failed (%d), running without cache\n",
            (int)r);
        gpu->pipeline_cache = VK_NULL_HANDLE;
    } else {
        if (use_blob) {
            fprintf(stderr,
                "gpu_vulkan: pipeline cache: loaded %zu bytes from %s\n",
                blob_size, gpu->pipeline_cache_path);
        } else {
            fprintf(stderr,
                "gpu_vulkan: pipeline cache: starting empty (path=%s)\n",
                gpu->pipeline_cache_path);
        }
    }

    free(blob);
}

/* Persist the current cache contents atomically (write to .tmp, rename).
 * Best-effort: any failure logs and returns. */
static void pc_save(Gpu* gpu)
{
    if (!gpu->pipeline_cache) return;
    if (!gpu->pipeline_cache_path[0]) return;

    size_t size = 0;
    if (vkGetPipelineCacheData(gpu->device, gpu->pipeline_cache,
                               &size, NULL) != VK_SUCCESS || size == 0) {
        return;
    }

    uint8_t* data = (uint8_t*)malloc(size);
    if (!data) return;

    if (vkGetPipelineCacheData(gpu->device, gpu->pipeline_cache,
                               &size, data) != VK_SUCCESS) {
        free(data);
        return;
    }

    if (!pc_mkdir_parents(gpu->pipeline_cache_path)) {
        free(data);
        return;
    }

    char tmp_path[1100];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp",
             gpu->pipeline_cache_path);

    FILE* f = fopen(tmp_path, "wb");
    if (!f) {
        fprintf(stderr,
            "gpu_vulkan: pipeline cache: open %s for write failed (%s)\n",
            tmp_path, strerror(errno));
        free(data);
        return;
    }
    size_t put = fwrite(data, 1, size, f);
    int    closed = (fclose(f) == 0);
    if (put != size || !closed) {
        fprintf(stderr,
            "gpu_vulkan: pipeline cache: short write (%zu of %zu) to %s\n",
            put, size, tmp_path);
        unlink(tmp_path);
        free(data);
        return;
    }

    if (rename(tmp_path, gpu->pipeline_cache_path) != 0) {
        fprintf(stderr,
            "gpu_vulkan: pipeline cache: rename %s → %s failed (%s)\n",
            tmp_path, gpu->pipeline_cache_path, strerror(errno));
        unlink(tmp_path);
    } else {
        fprintf(stderr,
            "gpu_vulkan: pipeline cache: saved %zu bytes to %s\n",
            size, gpu->pipeline_cache_path);
    }

    free(data);
}

/* ============================================================ *
 * perf/mem-pool: GPU memory sub-allocator
 *
 * Backs `gpu->resident_pool`, `gpu->transient_pool`, and
 * `gpu->staging_pool` with one pre-allocated VkDeviceMemory each.
 * Buffers are sub-allocated via a linear bump cursor; the resident
 * pool grows monotonically across a build, while the transient and
 * staging pools are reset (cursor=0) at the boundaries of build_accel.
 *
 * Why this matters: nsys on tera (12.3 M curves) shows ~1028 ms in
 * vkAllocateMemory + ~386 ms in vkFreeMemory across 23 calls — pure
 * driver allocator overhead. With the pool, those 23 calls collapse
 * to ~3 amortised allocations, saving ~1.0 s of build_ms.
 *
 * Sub-allocations don't have their own VkDeviceMemory — the caller's
 * `*out_mem` slot gets VK_NULL_HANDLE, so existing destroy paths
 * (vkDestroyBuffer + vkFreeMemory(NULL)) stay correct (vkFreeMemory
 * on a null handle is spec-defined as a no-op).
 *
 * memoryTypeBits compatibility: each sub-alloc asserts that the
 * caller's memoryTypeBits & (1u<<pool->mem_type_idx). On NVIDIA
 * discrete GPUs every DEVICE_LOCAL request meets this; we degrade
 * to a per-buffer vkAllocateMemory if it ever fails.
 * ============================================================ */

typedef struct GpuMemPool {
    VkDevice         device;
    VkDeviceMemory   memory;       /* VK_NULL_HANDLE = pool not yet created */
    VkDeviceSize     capacity;     /* full pool size */
    VkDeviceSize     cursor;       /* next free byte */
    uint32_t         mem_type_idx;
    VkMemoryPropertyFlags props;
    int              with_device_address; /* 1 = pool was allocated with
                                             VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT */
    void*            persistent_map;     /* HOST_VISIBLE pools only — whole-pool
                                             persistent map (vkMapMemory once at
                                             pool creation). Sub-allocs return a
                                             pointer = persistent_map + offset.
                                             Avoids the spec restriction that
                                             forbids overlapping maps on the
                                             same VkDeviceMemory. */
    /* fallback fail-safe: if sub-allocation fails, the caller is told
     * to take the slow path (per-buffer vkAllocateMemory). */
} GpuMemPool;

/* Create a memory pool of `size` bytes with the requested properties.
 * `with_device_address` adds VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT —
 * required for buffers that will be queried via
 * vkGetBufferDeviceAddress (BLAS pool, scratch, AABB buf, TLAS, scene
 * data...). It's harmless on non-address buffers. */
static GpuMemPool* gpu_pool_create(VkDevice device,
                                   VkPhysicalDevice phys,
                                   VkDeviceSize size,
                                   VkMemoryPropertyFlags props,
                                   int with_device_address)
{
    GpuMemPool* p = (GpuMemPool*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->device              = device;
    p->capacity            = size;
    p->cursor              = 0;
    p->props               = props;
    p->with_device_address = with_device_address;

    /* Find a memoryTypeIndex matching props. We don't know the
     * memoryTypeBits filter until a real buffer arrives, so pick one
     * that's compatible with "all bits set" (i.e. any heap that
     * matches `props`). */
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    p->mem_type_idx = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((mem_props.memoryTypes[i].propertyFlags & props) == props) {
            p->mem_type_idx = i;
            break;
        }
    }
    if (p->mem_type_idx == UINT32_MAX) {
        fprintf(stderr, "gpu_pool_create: no memory type matches props=0x%x\n",
                (unsigned)props);
        free(p);
        return NULL;
    }

    VkMemoryAllocateFlagsInfo flags_info = {0};
    flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags_info.flags = with_device_address ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;

    VkMemoryAllocateInfo ai = {0};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext           = with_device_address ? &flags_info : NULL;
    ai.allocationSize  = size;
    ai.memoryTypeIndex = p->mem_type_idx;

    if (vkAllocateMemory(device, &ai, NULL, &p->memory) != VK_SUCCESS) {
        fprintf(stderr,
                "gpu_pool_create: vkAllocateMemory FAILED size=%zu MB props=0x%x\n",
                (size_t)(size / (1024 * 1024)), (unsigned)props);
        free(p);
        return NULL;
    }

    /* HOST_VISIBLE pools: persistent-map the entire range so sub-allocs
     * just hand out offset pointers. The spec forbids overlapping
     * vkMapMemory calls on the same VkDeviceMemory, so per-suballoc
     * mapping won't work for a shared staging pool. */
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        if (vkMapMemory(device, p->memory, 0, VK_WHOLE_SIZE, 0, &p->persistent_map)
            != VK_SUCCESS) {
            fprintf(stderr, "gpu_pool_create: vkMapMemory failed for staging pool\n");
            vkFreeMemory(device, p->memory, NULL);
            free(p);
            return NULL;
        }
    }

    fprintf(stderr,
            "gpu_pool_create: %zu MB pool created (props=0x%x, type=%u, dev_addr=%d)\n",
            (size_t)(size / (1024 * 1024)),
            (unsigned)props, p->mem_type_idx, with_device_address);
    return p;
}

/* Reset the bump cursor to 0. All previously sub-allocated buffers
 * are now invalid; caller must vkDestroyBuffer them BEFORE calling
 * this (the buffer handle is still bound to memory in the pool's
 * range). Used at build_accel boundaries for transient/staging. */
static void gpu_pool_reset(GpuMemPool* p)
{
    if (p) p->cursor = 0;
}

/* Free the underlying VkDeviceMemory + the pool struct. */
static void gpu_pool_destroy(GpuMemPool* p)
{
    if (!p) return;
    if (p->memory) {
        if (p->persistent_map) vkUnmapMemory(p->device, p->memory);
        vkFreeMemory(p->device, p->memory, NULL);
    }
    free(p);
}

/* Sub-allocate `bytes` from the pool, respecting `alignment`. Returns
 * the offset; on success the caller binds an existing VkBuffer with
 * vkBindBufferMemory(buf, p->memory, offset). On failure (out of
 * pool space, or memoryTypeBits mismatch) returns UINT64_MAX so the
 * caller can fall back to a private vkAllocateMemory. */
static VkDeviceSize gpu_pool_suballoc(GpuMemPool* p,
                                       VkDeviceSize bytes,
                                       VkDeviceSize alignment,
                                       uint32_t memory_type_bits)
{
    if (!p) return (VkDeviceSize)-1;
    /* memoryTypeBits compatibility — pool's type must be acceptable. */
    if (!(memory_type_bits & (1u << p->mem_type_idx))) {
        fprintf(stderr,
                "gpu_pool_suballoc: pool type %u not in memoryTypeBits 0x%x — fallback\n",
                p->mem_type_idx, memory_type_bits);
        return (VkDeviceSize)-1;
    }
    if (alignment == 0) alignment = 1;
    VkDeviceSize aligned = (p->cursor + alignment - 1) & ~(alignment - 1);
    if (aligned + bytes > p->capacity) {
        fprintf(stderr,
                "gpu_pool_suballoc: pool exhausted (need %zu B, have %zu B free) — fallback\n",
                (size_t)bytes, (size_t)(p->capacity - aligned));
        return (VkDeviceSize)-1;
    }
    p->cursor = aligned + bytes;
    return aligned;
}

/* High-level helper: create a buffer and bind it into the pool. Returns 1 on
 * success, 0 on failure (caller should fall back to private alloc).
 *
 * On success:
 *   *out_buf is the new VkBuffer handle (caller must vkDestroyBuffer)
 *   *out_mem is set to VK_NULL_HANDLE so existing
 *            `if (mem) vkFreeMemory(...)` becomes a correct no-op.
 *   *bytes_consumed is the (aligned) sub-alloc size — caller adds this to
 *            gpu->total_allocated_gpu_mem to keep gpu_memory_used accounting
 *            in line with the pre-pool path.
 */
static int gpu_pool_alloc_buffer(GpuMemPool* p,
                                  VkDeviceSize size,
                                  VkBufferUsageFlags usage,
                                  VkBuffer* out_buf,
                                  VkDeviceMemory* out_mem,
                                  VkDeviceSize* bytes_consumed)
{
    *out_buf = VK_NULL_HANDLE;
    *out_mem = VK_NULL_HANDLE;
    if (bytes_consumed) *bytes_consumed = 0;
    if (!p) return 0;

    VkBufferCreateInfo bci = {0};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(p->device, &bci, NULL, out_buf) != VK_SUCCESS) {
        return 0;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(p->device, *out_buf, &req);

    VkDeviceSize offset = gpu_pool_suballoc(p, req.size, req.alignment,
                                             req.memoryTypeBits);
    if (offset == (VkDeviceSize)-1) {
        vkDestroyBuffer(p->device, *out_buf, NULL);
        *out_buf = VK_NULL_HANDLE;
        return 0;
    }

    if (vkBindBufferMemory(p->device, *out_buf, p->memory, offset) != VK_SUCCESS) {
        /* Roll back the bump cursor — this sub-alloc never made it to
         * the device-side bookkeeping. */
        p->cursor = offset;
        vkDestroyBuffer(p->device, *out_buf, NULL);
        *out_buf = VK_NULL_HANDLE;
        return 0;
    }

    /* *out_mem stays VK_NULL_HANDLE — sub-allocations share the pool's
     * memory, and vkFreeMemory(NULL) is a spec-defined no-op so the
     * dozens of existing `if (mem) vkFreeMemory(...)` call sites become
     * correct no-ops with no further edits. */
    if (bytes_consumed) *bytes_consumed = req.size;
    return 1;
}
/* ============================================================ */

/* ---- Swapchain / depth / framebuffer creation (shared by init and resize) ---- */

static void create_swapchain(Gpu* gpu)
{
    int surface_diag = getenv("NUSD_VK_SURFACE_DIAG") != NULL;
    /* Query surface capabilities */
    VkSurfaceCapabilitiesKHR caps;
    if (surface_diag) fprintf(stderr, "gpu_vulkan: surface diag: capabilities\n");
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu->physical_device,
                                              gpu->surface, &caps);

    /* Choose format: prefer B8G8R8A8_SRGB */
    uint32_t fmt_count = 0;
    if (surface_diag) fprintf(stderr, "gpu_vulkan: surface diag: formats count\n");
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu->physical_device, gpu->surface,
                                         &fmt_count, NULL);
    if (fmt_count == 0) {
        fprintf(stderr, "gpu_vulkan: surface reports no formats\n");
        return;
    }
    VkSurfaceFormatKHR* formats = malloc(fmt_count * sizeof(VkSurfaceFormatKHR));
    if (!formats) return;
    if (surface_diag) fprintf(stderr, "gpu_vulkan: surface diag: formats fill %u\n", fmt_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu->physical_device, gpu->surface,
                                         &fmt_count, formats);

    VkSurfaceFormatKHR chosen_fmt = formats[0];
    for (uint32_t i = 0; i < fmt_count; i++) {
        if (formats[i].format == VK_FORMAT_B8G8R8A8_SRGB &&
            formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen_fmt = formats[i];
            break;
        }
    }
    free(formats);

    gpu->swapchain_format = chosen_fmt.format;

    /* Choose present mode: prefer MAILBOX, fallback FIFO */
    uint32_t pm_count = 0;
    if (surface_diag) fprintf(stderr, "gpu_vulkan: surface diag: present modes count\n");
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu->physical_device,
                                              gpu->surface, &pm_count, NULL);

    VkPresentModeKHR chosen_mode = VK_PRESENT_MODE_FIFO_KHR;
    int want_immediate = (getenv("NUSD_VSYNC_OFF") != NULL);
    if (pm_count > 0) {
        VkPresentModeKHR* modes = malloc(pm_count * sizeof(VkPresentModeKHR));
        if (modes) {
            if (surface_diag) fprintf(stderr, "gpu_vulkan: surface diag: present modes fill %u\n", pm_count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(gpu->physical_device,
                                                      gpu->surface, &pm_count, modes);
            for (uint32_t i = 0; i < pm_count; i++) {
                if (want_immediate && modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                    chosen_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                    break;
                }
                if (!want_immediate && modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
                    chosen_mode = VK_PRESENT_MODE_MAILBOX_KHR;
                    break;
                }
            }
            free(modes);
        }
    }

    /* Choose extent */
    if (caps.currentExtent.width != UINT32_MAX) {
        gpu->swapchain_extent = caps.currentExtent;
    } else {
        gpu->swapchain_extent.width  = (uint32_t)gpu->width;
        gpu->swapchain_extent.height = (uint32_t)gpu->height;
        if (gpu->swapchain_extent.width < caps.minImageExtent.width)
            gpu->swapchain_extent.width = caps.minImageExtent.width;
        if (gpu->swapchain_extent.width > caps.maxImageExtent.width)
            gpu->swapchain_extent.width = caps.maxImageExtent.width;
        if (gpu->swapchain_extent.height < caps.minImageExtent.height)
            gpu->swapchain_extent.height = caps.minImageExtent.height;
        if (gpu->swapchain_extent.height > caps.maxImageExtent.height)
            gpu->swapchain_extent.height = caps.maxImageExtent.height;
    }

    /* Image count: min+1, cap at max */
    uint32_t img_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && img_count > caps.maxImageCount)
        img_count = caps.maxImageCount;
    if (img_count > 3)
        img_count = 3;

    VkSwapchainCreateInfoKHR sci = {0};
    sci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface          = gpu->surface;
    sci.minImageCount    = img_count;
    sci.imageFormat      = chosen_fmt.format;
    sci.imageColorSpace  = chosen_fmt.colorSpace;
    sci.imageExtent      = gpu->swapchain_extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode      = chosen_mode;
    sci.clipped          = VK_TRUE;
    sci.oldSwapchain     = VK_NULL_HANDLE;

    if (surface_diag) fprintf(stderr, "gpu_vulkan: surface diag: create swapchain %ux%u images=%u\n",
                              gpu->swapchain_extent.width,
                              gpu->swapchain_extent.height, img_count);
    if (vkCreateSwapchainKHR(gpu->device, &sci, NULL,
                             &gpu->swapchain) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create swapchain\n");
        return;
    }

    /* Get swapchain images */
    if (surface_diag) fprintf(stderr, "gpu_vulkan: surface diag: swapchain images count\n");
    vkGetSwapchainImagesKHR(gpu->device, gpu->swapchain,
                            &gpu->image_count, NULL);
    gpu->swapchain_images = malloc(gpu->image_count * sizeof(VkImage));
    if (surface_diag) fprintf(stderr, "gpu_vulkan: surface diag: swapchain images fill %u\n", gpu->image_count);
    vkGetSwapchainImagesKHR(gpu->device, gpu->swapchain,
                            &gpu->image_count, gpu->swapchain_images);

    /* Create image views */
    gpu->swapchain_views = malloc(gpu->image_count * sizeof(VkImageView));
    for (uint32_t i = 0; i < gpu->image_count; i++) {
        VkImageViewCreateInfo vci = {0};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = gpu->swapchain_images[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = gpu->swapchain_format;
        vci.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        vci.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        vci.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        vci.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        vci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel   = 0;
        vci.subresourceRange.levelCount     = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount     = 1;

        if (vkCreateImageView(gpu->device, &vci, NULL,
                              &gpu->swapchain_views[i]) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to create swapchain image view\n");
        }
    }
}

static void create_surfaceless_images(Gpu* gpu)
{
    gpu->swapchain_format = VK_FORMAT_B8G8R8A8_SRGB;
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(gpu->physical_device,
                                        gpu->swapchain_format, &props);
    VkFormatFeatureFlags need =
        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
        VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if ((props.optimalTilingFeatures & need) != need)
        gpu->swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;

    gpu->swapchain_extent.width = gpu->width > 0 ? (uint32_t)gpu->width : 1u;
    gpu->swapchain_extent.height = gpu->height > 0 ? (uint32_t)gpu->height : 1u;
    gpu->image_count = MAX_FRAMES_IN_FLIGHT;
    gpu->swapchain_images =
        (VkImage*)calloc(gpu->image_count, sizeof(VkImage));
    gpu->swapchain_views =
        (VkImageView*)calloc(gpu->image_count, sizeof(VkImageView));
    gpu->swapchain_memories =
        (VkDeviceMemory*)calloc(gpu->image_count, sizeof(VkDeviceMemory));
    if (!gpu->swapchain_images || !gpu->swapchain_views ||
        !gpu->swapchain_memories) {
        fprintf(stderr, "gpu_vulkan: failed to allocate surfaceless image tables\n");
        return;
    }

    for (uint32_t i = 0; i < gpu->image_count; i++) {
        VkImageCreateInfo ici = {0};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = gpu->swapchain_format;
        ici.extent.width = gpu->swapchain_extent.width;
        ici.extent.height = gpu->swapchain_extent.height;
        ici.extent.depth = 1;
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(gpu->device, &ici, NULL,
                          &gpu->swapchain_images[i]) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to create surfaceless image %u\n", i);
            gpu->image_count = i;
            return;
        }

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(gpu->device,
                                     gpu->swapchain_images[i], &req);
        VkMemoryAllocateInfo ai = {0};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory_type(
            gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (gpu_alloc_memory(gpu, &ai, &gpu->swapchain_memories[i]) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to allocate surfaceless image memory %u\n", i);
            gpu->image_count = i;
            return;
        }
        vkBindImageMemory(gpu->device, gpu->swapchain_images[i],
                          gpu->swapchain_memories[i], 0);

        VkImageViewCreateInfo vci = {0};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = gpu->swapchain_images[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = gpu->swapchain_format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        if (vkCreateImageView(gpu->device, &vci, NULL,
                              &gpu->swapchain_views[i]) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to create surfaceless image view %u\n", i);
            gpu->image_count = i;
            return;
        }
    }

    fprintf(stderr, "gpu_vulkan: surfaceless target %ux%u images=%u\n",
            gpu->swapchain_extent.width, gpu->swapchain_extent.height,
            gpu->image_count);
}

static void create_depth_buffer(Gpu* gpu)
{
    gpu->depth_format = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo ici = {0};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = gpu->depth_format;
    ici.extent.width  = gpu->swapchain_extent.width;
    ici.extent.height = gpu->swapchain_extent.height;
    ici.extent.depth  = 1;
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;  /* for DLSS depth copy */
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(gpu->device, &ici, NULL,
                      &gpu->depth_image) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create depth image\n");
        return;
    }

    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(gpu->device, gpu->depth_image, &mem_req);

    VkMemoryAllocateInfo ai = {0};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = mem_req.size;
    ai.memoryTypeIndex = find_memory_type(gpu->physical_device,
                                          mem_req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (gpu_alloc_memory(gpu, &ai, &gpu->depth_memory) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to allocate depth memory\n");
        vkDestroyImage(gpu->device, gpu->depth_image, NULL);
        gpu->depth_image = VK_NULL_HANDLE;
        return;
    }

    vkBindImageMemory(gpu->device, gpu->depth_image, gpu->depth_memory, 0);

    VkImageViewCreateInfo vci = {0};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = gpu->depth_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = gpu->depth_format;
    vci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    vci.subresourceRange.baseMipLevel   = 0;
    vci.subresourceRange.levelCount     = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount     = 1;

    if (vkCreateImageView(gpu->device, &vci, NULL,
                          &gpu->depth_view) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create depth image view\n");
    }
}

static void create_framebuffers(Gpu* gpu)
{
    gpu->framebuffers = malloc(gpu->image_count * sizeof(VkFramebuffer));

    for (uint32_t i = 0; i < gpu->image_count; i++) {
        VkImageView attachments[2] = {
            gpu->swapchain_views[i],
            gpu->depth_view,
        };

        VkFramebufferCreateInfo fci = {0};
        fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass      = gpu->render_pass;
        fci.attachmentCount = 2;
        fci.pAttachments    = attachments;
        fci.width           = gpu->swapchain_extent.width;
        fci.height          = gpu->swapchain_extent.height;
        fci.layers          = 1;

        if (vkCreateFramebuffer(gpu->device, &fci, NULL,
                                &gpu->framebuffers[i]) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to create framebuffer %u\n", i);
        }
    }
}

static void destroy_swapchain_resources(Gpu* gpu)
{
    if (gpu->framebuffers) {
        for (uint32_t i = 0; i < gpu->image_count; i++) {
            vkDestroyFramebuffer(gpu->device, gpu->framebuffers[i], NULL);
        }
        free(gpu->framebuffers);
        gpu->framebuffers = NULL;
    }

    vkDestroyImageView(gpu->device, gpu->depth_view, NULL);
    vkDestroyImage(gpu->device, gpu->depth_image, NULL);
    vkFreeMemory(gpu->device, gpu->depth_memory, NULL);

    for (uint32_t i = 0; i < gpu->image_count; i++) {
        if (gpu->swapchain_views)
            vkDestroyImageView(gpu->device, gpu->swapchain_views[i], NULL);
        if (gpu->surfaceless && gpu->swapchain_images)
            vkDestroyImage(gpu->device, gpu->swapchain_images[i], NULL);
        if (gpu->surfaceless && gpu->swapchain_memories)
            vkFreeMemory(gpu->device, gpu->swapchain_memories[i], NULL);
    }
    free(gpu->swapchain_views);
    free(gpu->swapchain_images);
    free(gpu->swapchain_memories);
    gpu->swapchain_views  = NULL;
    gpu->swapchain_images = NULL;
    gpu->swapchain_memories = NULL;

    if (gpu->swapchain != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(gpu->device, gpu->swapchain, NULL);
    gpu->swapchain = VK_NULL_HANDLE;
    gpu->image_count = 0;
}

/* ---- Public API ---- */

/* perf/mem-pool: forward declarations for the pool helpers used by
 * gpu_init's eager pool pre-allocation. The full pool implementation
 * (GpuMemPool struct, gpu_pool_*, rt_create_buffer_pooled, etc.) lives
 * just above gpu_build_rt_scene further down so it can sit next to
 * the call sites it serves. */
enum {
    RT_POOL_NONE = 0,
    RT_POOL_RESIDENT,
    RT_POOL_TRANSIENT,
    RT_POOL_STAGING,
};
static GpuMemPool* rt_ensure_pool(Gpu* gpu, int pool_tag);
#if NUSD_HAS_VK_NV_MEGA_GEOMETRY
static void nu_cluster_build_selftest(Gpu* gpu);  /* Phase 0 cluster-AS GPU build */
static VkDeviceAddress nu_build_cluster_blas_for_mesh(
        Gpu* gpu, VkDeviceAddress vaddr, VkDeviceAddress iaddr,
        uint32_t nverts, uint32_t ntris, uint32_t vstride,
        VkBuffer keep_buf[2], VkDeviceMemory keep_mem[2]);
#endif
static VkCommandBuffer rt_begin_cmd(Gpu* gpu);
static void rt_end_cmd(Gpu* gpu, VkCommandBuffer cmd);

/* Debug-utils callback for NUSD_VK_VALIDATION=1. Prints warning+error
 * validation messages to stderr so cluster-AS (and other) mistakes surface as
 * real VUIDs instead of an opaque VK_ERROR_DEVICE_LOST. */
static VKAPI_ATTR VkBool32 VKAPI_CALL nu_vk_debug_cb(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT types,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void* user)
{
    (void)types; (void)user;
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        fprintf(stderr, "[vk-validation] %s\n",
                (data && data->pMessage) ? data->pMessage : "(no message)");
    return VK_FALSE;
}

Gpu* gpu_init(void* glfw_window, int width, int height)
{
    Gpu* gpu = calloc(1, sizeof(Gpu));
    if (!gpu) return NULL;
    int true_headless = glfw_window == NULL;

    gpu->width  = width;
    gpu->height = height;
    gpu->headless = true_headless;
    gpu->surfaceless = true_headless;
    gpu->pixels_interop_fd = -1;  /* not yet exported */
    gpu->tlas_xforms_fd    = -1;  /* PR 2: not yet exported */
    gpu->env_intensity = 1.0f;    /* default no-op; set by load_environment_intensity */
    gpu->scene_up_axis = 1;       /* USD default: Y-up */
    gpu->tone_exposure_scale = 1.0f;
    gpu->tone_sky_scale = 1.0f;
    gpu->tone_white_point = 1.0f;
    gpu->tone_flags = 0u;
    /* Default dome color: near-white (matches Newton's 0xEEEEEE clear).
     * Override via gpu_set_dome_color(). */
    gpu->dome_color[0] = 0.93f;
    gpu->dome_color[1] = 0.93f;
    gpu->dome_color[2] = 0.93f;
    gpu->dome_color[3] = 1.0f;

    /* --- Volk init (loads global Vulkan functions) --- */
    if (volkInitialize() != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to initialize volk\n");
        free(gpu);
        return NULL;
    }

    /* --- Instance --- */
    {
        VkApplicationInfo app_info = {0};
        app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName   = "nanousd-vulkan-renderer";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName        = "gpu_vulkan";
        app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion         = VK_API_VERSION_1_2;

        uint32_t surface_ext_count = 0;
        const char** surface_exts = NULL;
        const char* headless_exts[2];
        if (true_headless) {
            headless_exts[surface_ext_count++] = VK_KHR_SURFACE_EXTENSION_NAME;
            surface_exts = headless_exts;
        } else {
            surface_exts = glfwGetRequiredInstanceExtensions(&surface_ext_count);
            if (!surface_exts || surface_ext_count == 0) {
                fprintf(stderr, "gpu_vulkan: GLFW returned no Vulkan surface extensions\n");
                free(gpu);
                return NULL;
            }
        }

        /* Merge GLFW extensions with any DLSS-required instance extensions. */
        const char* dlss_inst_exts[16];
        uint32_t dlss_inst_ext_count = dlss_get_instance_extensions(dlss_inst_exts, 16);

        /* perf/vk-instrumentation: probe for VK_EXT_debug_utils so we can
         * emit nsys-visible region labels (vulkan_gpu_marker_sum). The
         * extension is present on every modern desktop driver but absent
         * on some headless / VK1.0 stacks, so we feature-detect rather
         * than hard-require it. */
        int has_debug_utils = 0;
        {
            uint32_t avail_count = 0;
            vkEnumerateInstanceExtensionProperties(NULL, &avail_count, NULL);
            if (avail_count > 0) {
                VkExtensionProperties* avail =
                    malloc(avail_count * sizeof(VkExtensionProperties));
                vkEnumerateInstanceExtensionProperties(NULL, &avail_count, avail);
                for (uint32_t i = 0; i < avail_count; i++) {
                    if (strcmp(avail[i].extensionName,
                               VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
                        has_debug_utils = 1;
                        break;
                    }
                }
                free(avail);
            }
        }

        uint32_t total_inst_ext_count = surface_ext_count + dlss_inst_ext_count
                                        + (has_debug_utils ? 1 : 0);
        const char** all_inst_exts = (const char**)malloc(total_inst_ext_count * sizeof(const char*));
        for (uint32_t i = 0; i < surface_ext_count; i++)
            all_inst_exts[i] = surface_exts[i];
        for (uint32_t i = 0; i < dlss_inst_ext_count; i++)
            all_inst_exts[surface_ext_count + i] = dlss_inst_exts[i];
        if (has_debug_utils) {
            all_inst_exts[surface_ext_count + dlss_inst_ext_count] =
                VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        }

        int want_validation = (getenv("NUSD_VK_VALIDATION") != NULL);
        const char* val_layers[] = { "VK_LAYER_KHRONOS_validation" };

        VkInstanceCreateInfo ici = {0};
        ici.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo        = &app_info;
        ici.enabledExtensionCount   = total_inst_ext_count;
        ici.ppEnabledExtensionNames = all_inst_exts;
        if (want_validation) {
            ici.enabledLayerCount   = 1;
            ici.ppEnabledLayerNames = val_layers;
        }

        if (vkCreateInstance(&ici, NULL, &gpu->instance) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to create instance\n");
            free(all_inst_exts);
            free(gpu);
            return NULL;
        }

        free(all_inst_exts);
        volkLoadInstance(gpu->instance);
        gpu->debug_utils_enabled = has_debug_utils;
        if (want_validation && has_debug_utils) {
            VkDebugUtilsMessengerCreateInfoEXT dm = {0};
            dm.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            dm.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                               | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dm.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                           | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                           | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dm.pfnUserCallback = nu_vk_debug_cb;
            if (vkCreateDebugUtilsMessengerEXT(gpu->instance, &dm, NULL,
                                               &gpu->debug_messenger) == VK_SUCCESS)
                fprintf(stderr, "gpu_vulkan: NUSD_VK_VALIDATION — validation layer + debug messenger active\n");
            else
                fprintf(stderr, "gpu_vulkan: NUSD_VK_VALIDATION — messenger create failed\n");
        }
        if (has_debug_utils) {
            fprintf(stderr,
                    "gpu_vulkan: VK_EXT_debug_utils enabled "
                    "(nsys: --trace=vulkan,vulkan-annotations populates "
                    "vulkan_marker_sum CPU-side; for GPU-side timings use "
                    "nu_get_phase_timings_ms())\n");
        }
    }

    /* --- Surface --- */
    if (glfw_window) {
        if (glfwCreateWindowSurface(gpu->instance, (GLFWwindow*)glfw_window,
                                    NULL, &gpu->surface) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to create window surface\n");
            vkDestroyInstance(gpu->instance, NULL);
            free(gpu);
            return NULL;
        }
    } else {
        fprintf(stderr, "gpu_vulkan: using surfaceless headless images\n");
    }

    /* --- Physical device --- */
    {
        uint32_t dev_count = 0;
        vkEnumeratePhysicalDevices(gpu->instance, &dev_count, NULL);
        if (dev_count == 0) {
            fprintf(stderr, "gpu_vulkan: no Vulkan-capable GPUs found\n");
            vkDestroySurfaceKHR(gpu->instance, gpu->surface, NULL);
            vkDestroyInstance(gpu->instance, NULL);
            free(gpu);
            return NULL;
        }

        VkPhysicalDevice* devices = malloc(dev_count * sizeof(VkPhysicalDevice));
        vkEnumeratePhysicalDevices(gpu->instance, &dev_count, devices);

        /* Prefer discrete GPU */
        gpu->physical_device = devices[0];
        for (uint32_t i = 0; i < dev_count; i++) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[i], &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                gpu->physical_device = devices[i];
                break;
            }
        }
        free(devices);
    }

    /* --- Queue family --- */
    {
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(gpu->physical_device,
                                                  &qf_count, NULL);
        VkQueueFamilyProperties* qf_props =
            malloc(qf_count * sizeof(VkQueueFamilyProperties));
        vkGetPhysicalDeviceQueueFamilyProperties(gpu->physical_device,
                                                  &qf_count, qf_props);

        gpu->queue_family = UINT32_MAX;
        for (uint32_t i = 0; i < qf_count; i++) {
            if ((qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0)
                continue;
            if (true_headless) {
                gpu->queue_family = i;
                break;
            }
            VkBool32 present_support = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(gpu->physical_device, i,
                                                  gpu->surface,
                                                  &present_support);
            if (present_support) {
                gpu->queue_family = i;
                break;
            }
        }
        free(qf_props);

        if (gpu->queue_family == UINT32_MAX) {
            fprintf(stderr, "gpu_vulkan: no suitable queue family\n");
            vkDestroySurfaceKHR(gpu->instance, gpu->surface, NULL);
            vkDestroyInstance(gpu->instance, NULL);
            free(gpu);
            return NULL;
        }
    }

    /* --- Logical device (with RT extensions if available) --- */
    {
        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci = {0};
        qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = gpu->queue_family;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &priority;

        /* Check for RT extension support */
        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(gpu->physical_device, NULL, &ext_count, NULL);
        VkExtensionProperties* avail_exts = malloc(ext_count * sizeof(VkExtensionProperties));
        vkEnumerateDeviceExtensionProperties(gpu->physical_device, NULL, &ext_count, avail_exts);

        int has_accel = 0, has_rt_pipe = 0, has_deferred = 0, has_ray_query = 0;
        int has_ext_mem_fd = 0, has_ext_sem_fd = 0;
        int has_ser_ext = 0;
        int has_barycentric_ext = 0;  /* VK_KHR_fragment_shader_barycentric (raster Ptex interp) */
        int has_cluster_as = 0, has_ptlas = 0, has_lss = 0;  /* RTX Mega Geometry */
        int cluster_as_enable = 0, ptlas_enable = 0, lss_enable = 0;
        for (uint32_t i = 0; i < ext_count; i++) {
            if (strcmp(avail_exts[i].extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0) has_accel = 1;
            if (strcmp(avail_exts[i].extensionName, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) == 0) has_rt_pipe = 1;
            if (strcmp(avail_exts[i].extensionName, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) == 0) has_deferred = 1;
            if (strcmp(avail_exts[i].extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME) == 0) has_ray_query = 1;
            if (strcmp(avail_exts[i].extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) == 0) has_ext_mem_fd = 1;
            if (strcmp(avail_exts[i].extensionName, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) == 0) has_ext_sem_fd = 1;
            if (strcmp(avail_exts[i].extensionName, VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME) == 0) has_ser_ext = 1;
            if (strcmp(avail_exts[i].extensionName, VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME) == 0) has_barycentric_ext = 1;
#if NUSD_HAS_VK_NV_MEGA_GEOMETRY
            if (strcmp(avail_exts[i].extensionName, VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0) has_cluster_as = 1;
            if (strcmp(avail_exts[i].extensionName, VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0) has_ptlas = 1;
            if (strcmp(avail_exts[i].extensionName, VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME) == 0) has_lss = 1;
#endif
        }
        free(avail_exts);

        gpu->rt_available = (has_accel && has_rt_pipe && has_deferred && has_ray_query) ? 1 : 0;
        gpu->interop_available = (has_ext_mem_fd && has_ext_sem_fd) ? 1 : 0;

        /* SER (Shader Execution Reordering) probe. The extension exists only
         * on Ada (SM 8.9) and newer NVIDIA GPUs; check both the extension
         * and the `rayTracingInvocationReorder` feature flag, since some
         * driver/HW combinations advertise the extension without the
         * feature actually being enabled. Property struct captures the
         * driver's preferred reorder hint mode (NONE / REORDER) for logging. */
        gpu->ser_available = 0;
        gpu->ser_reorder_hint = 0;
        if (has_ser_ext && gpu->rt_available) {
            VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV ser_feat = {0};
            ser_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV;

            VkPhysicalDeviceFeatures2 feat2 = {0};
            feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            feat2.pNext = &ser_feat;
            vkGetPhysicalDeviceFeatures2(gpu->physical_device, &feat2);
            if (ser_feat.rayTracingInvocationReorder == VK_TRUE) {
                gpu->ser_available = 1;

                VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV ser_props = {0};
                ser_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV;
                VkPhysicalDeviceProperties2 props2 = {0};
                props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                props2.pNext = &ser_props;
                vkGetPhysicalDeviceProperties2(gpu->physical_device, &props2);
                gpu->ser_reorder_hint =
                    (int)ser_props.rayTracingInvocationReorderReorderingHint;
            }
        }

        /* Fragment-shader barycentrics probe — lets the raster mesh.frag
         * interpolate the three Ptex triangle-corner colors per pixel (matching
         * RT's barycentric blend) instead of flat-averaging them. Optional:
         * when unavailable the renderer keeps the averaging frag variant. */
        gpu->barycentric_available = 0;
        if (has_barycentric_ext) {
            VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary_feat = {0};
            bary_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR;
            VkPhysicalDeviceFeatures2 feat2 = {0};
            feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            feat2.pNext = &bary_feat;
            vkGetPhysicalDeviceFeatures2(gpu->physical_device, &feat2);
            if (bary_feat.fragmentShaderBarycentric == VK_TRUE)
                gpu->barycentric_available = 1;
        }

        /* RTX Mega Geometry probe (cluster AS / PTLAS / LSS). These are RT
         * features, so gate on rt_available. Confirm the device advertises each
         * feature (the extension may be present without the feature) and read
         * the device-queried cluster limits — maxTrianglesPerCluster /
         * maxVerticesPerCluster / maxPartitionCount are inputs the cluster-AS
         * build path depends on, so log them at init. */
#if NUSD_HAS_VK_NV_MEGA_GEOMETRY
        if (gpu->rt_available && (has_cluster_as || has_ptlas || has_lss)) {
            VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clas_feat = {0};
            clas_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV;
            VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV ptlas_feat = {0};
            ptlas_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV;
            VkPhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV lss_feat = {0};
            lss_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_LINEAR_SWEPT_SPHERES_FEATURES_NV;
            void* fc = NULL;
            if (has_cluster_as) { clas_feat.pNext = fc; fc = &clas_feat; }
            if (has_ptlas)      { ptlas_feat.pNext = fc; fc = &ptlas_feat; }
            if (has_lss)        { lss_feat.pNext = fc; fc = &lss_feat; }
            VkPhysicalDeviceFeatures2 mf2 = {0};
            mf2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            mf2.pNext = fc;
            vkGetPhysicalDeviceFeatures2(gpu->physical_device, &mf2);
            cluster_as_enable = (has_cluster_as && clas_feat.clusterAccelerationStructure) ? 1 : 0;
            ptlas_enable      = (has_ptlas && ptlas_feat.partitionedAccelerationStructure) ? 1 : 0;
            lss_enable        = (has_lss && lss_feat.linearSweptSpheres) ? 1 : 0;
            gpu->cluster_as_available = cluster_as_enable;
            gpu->ptlas_available      = ptlas_enable;
            gpu->lss_available        = lss_enable;

            VkPhysicalDeviceClusterAccelerationStructurePropertiesNV clas_props = {0};
            clas_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV;
            VkPhysicalDevicePartitionedAccelerationStructurePropertiesNV ptlas_props = {0};
            ptlas_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_PROPERTIES_NV;
            void* pc = NULL;
            if (cluster_as_enable) { clas_props.pNext = pc; pc = &clas_props; }
            if (ptlas_enable)      { ptlas_props.pNext = pc; pc = &ptlas_props; }
            if (pc) {
                VkPhysicalDeviceProperties2 mp2 = {0};
                mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                mp2.pNext = pc;
                vkGetPhysicalDeviceProperties2(gpu->physical_device, &mp2);
            }
            fprintf(stderr,
                    "gpu_vulkan: RTX Mega Geometry — clusterAS=%d ptlas=%d lss=%d | "
                    "maxTrisPerCluster=%u maxVertsPerCluster=%u maxPartitions=%u\n",
                    cluster_as_enable, ptlas_enable, lss_enable,
                    clas_props.maxTrianglesPerCluster, clas_props.maxVerticesPerCluster,
                    ptlas_props.maxPartitionCount);
        }
#endif

        const char* dev_exts_base[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
        const char* dev_exts_rt[] = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_RAY_QUERY_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        };
        /* CUDA interop extensions (external memory + semaphore fd export) */
        const char* dev_exts_interop[] = {
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        };

        /* Query DLSS-required device extensions and merge. */
        const char* dlss_dev_exts[16];
        uint32_t dlss_dev_ext_count = dlss_get_device_extensions(
            gpu->instance, gpu->physical_device, dlss_dev_exts, 16);

        uint32_t base_count;
        const char** base_exts;
        if (gpu->rt_available) {
            base_exts = dev_exts_rt;
            base_count = 5;
        } else {
            base_exts = dev_exts_base;
            base_count = 1;
        }

        uint32_t interop_count = gpu->interop_available ? 2 : 0;
        uint32_t ser_count     = gpu->ser_available ? 1 : 0;
        uint32_t bary_count    = gpu->barycentric_available ? 1 : 0;
#if NUSD_HAS_VK_NV_MEGA_GEOMETRY
        uint32_t mega_count    = (cluster_as_enable ? 1u : 0u) + (ptlas_enable ? 1u : 0u) + (lss_enable ? 1u : 0u);
#else
        uint32_t mega_count    = 0;
#endif
        uint32_t dev_ext_count = base_count + interop_count + ser_count + bary_count + mega_count + dlss_dev_ext_count;
        const char** dev_exts = (const char**)malloc(dev_ext_count * sizeof(const char*));
        uint32_t eoff = 0;
        for (uint32_t i = 0; i < base_count; i++)    dev_exts[eoff++] = base_exts[i];
        for (uint32_t i = 0; i < interop_count; i++) dev_exts[eoff++] = dev_exts_interop[i];
        if (ser_count)         dev_exts[eoff++] = VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME;
        if (bary_count)        dev_exts[eoff++] = VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME;
#if NUSD_HAS_VK_NV_MEGA_GEOMETRY
        if (cluster_as_enable) dev_exts[eoff++] = VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME;
        if (ptlas_enable)      dev_exts[eoff++] = VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME;
        if (lss_enable)        dev_exts[eoff++] = VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME;
#endif
        for (uint32_t i = 0; i < dlss_dev_ext_count; i++) dev_exts[eoff++] = dlss_dev_exts[i];

        /* Features chain: Vulkan 1.2 features + RT features */
        VkPhysicalDeviceVulkan12Features vk12_features = {0};
        vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk12_features.bufferDeviceAddress = VK_TRUE;
        vk12_features.scalarBlockLayout   = VK_TRUE;
        vk12_features.timelineSemaphore   = VK_TRUE;  /* needed for CUDA interop sync */
        vk12_features.hostQueryReset      = VK_TRUE;  /* perf/vk-instrumentation: vkResetQueryPool from host */
        /* Phase C: per-env TLAS array. With 4096 envs, the static
         * maxPerStageDescriptorAccelerationStructures (16 on Blackwell) is
         * too small; UAB lifts it into the millions. */
        vk12_features.descriptorIndexing                            = VK_TRUE;
        vk12_features.runtimeDescriptorArray                        = VK_TRUE;
        vk12_features.shaderSampledImageArrayNonUniformIndexing     = VK_TRUE;
        vk12_features.descriptorBindingPartiallyBound               = VK_TRUE;
        vk12_features.descriptorBindingVariableDescriptorCount      = VK_TRUE;
        vk12_features.descriptorBindingUpdateUnusedWhilePending     = VK_TRUE;
        vk12_features.descriptorBindingSampledImageUpdateAfterBind  = VK_TRUE;
        vk12_features.descriptorBindingStorageImageUpdateAfterBind  = VK_TRUE;
        vk12_features.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
        vk12_features.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;

        VkPhysicalDeviceVulkan11Features vk11_features = {0};
        vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vk11_features.pNext = &vk12_features;

        /* Fragment-shader barycentrics (raster Ptex per-pixel interp). Insert
         * between vk11 and vk12 so it coexists with the RT feature tail that the
         * block below appends onto vk12_features.pNext. */
        VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary_features = {0};
        if (gpu->barycentric_available) {
            bary_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR;
            bary_features.fragmentShaderBarycentric = VK_TRUE;
            bary_features.pNext = vk11_features.pNext;  /* &vk12_features */
            vk11_features.pNext = &bary_features;
        }

        void* features_chain = &vk11_features;

        VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features = {0};
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_features = {0};
        VkPhysicalDeviceRayQueryFeaturesKHR rq_features = {0};
        VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV ser_features = {0};
#if NUSD_HAS_VK_NV_MEGA_GEOMETRY
        VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clas_dev_feat = {0};
        VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV ptlas_dev_feat = {0};
        VkPhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV lss_dev_feat = {0};
#endif
        if (gpu->rt_available) {
            as_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
            as_features.accelerationStructure = VK_TRUE;
            /* Phase C: needed to bind a 4096-entry TLAS array via descriptor indexing. */
            as_features.descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE;

            rt_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
            rt_features.rayTracingPipeline = VK_TRUE;

            rq_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
            rq_features.rayQuery = VK_TRUE;

            vk12_features.pNext = &as_features;
            as_features.pNext = &rt_features;
            rt_features.pNext = &rq_features;

            /* SER (VK_NV_ray_tracing_invocation_reorder) — chain only when
             * the extension is enabled on the device, otherwise the driver
             * may reject the unknown sType. */
            if (gpu->ser_available) {
                ser_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV;
                ser_features.rayTracingInvocationReorder = VK_TRUE;
                rq_features.pNext = &ser_features;
            }

            /* RTX Mega Geometry: chain only the enabled features onto the tail
             * of the RT feature chain. Append in order; `tail` tracks the last
             * pNext slot (ser_features if SER is on, else rq_features). */
#if NUSD_HAS_VK_NV_MEGA_GEOMETRY
            void** tail = gpu->ser_available ? &ser_features.pNext : &rq_features.pNext;
            if (cluster_as_enable) {
                clas_dev_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV;
                clas_dev_feat.clusterAccelerationStructure = VK_TRUE;
                *tail = &clas_dev_feat; tail = &clas_dev_feat.pNext;
            }
            if (ptlas_enable) {
                ptlas_dev_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV;
                ptlas_dev_feat.partitionedAccelerationStructure = VK_TRUE;
                *tail = &ptlas_dev_feat; tail = &ptlas_dev_feat.pNext;
            }
            if (lss_enable) {
                lss_dev_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_LINEAR_SWEPT_SPHERES_FEATURES_NV;
                lss_dev_feat.linearSweptSpheres = VK_TRUE;
                lss_dev_feat.spheres = VK_TRUE;
                *tail = &lss_dev_feat; tail = &lss_dev_feat.pNext;
            }
#endif
        }

        VkPhysicalDeviceFeatures2 features2 = {0};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = features_chain;
        /* Required for anisotropic texture filtering on material samplers. */
        features2.features.samplerAnisotropy = VK_TRUE;
        /* Required by the raster BasisCurves tube pipeline. */
        features2.features.tessellationShader = VK_TRUE;

        VkDeviceCreateInfo dci = {0};
        dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.pNext                   = &features2;
        dci.queueCreateInfoCount    = 1;
        dci.pQueueCreateInfos       = &qci;
        dci.enabledExtensionCount   = dev_ext_count;
        dci.ppEnabledExtensionNames = dev_exts;

        if (vkCreateDevice(gpu->physical_device, &dci, NULL,
                           &gpu->device) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to create logical device\n");
            free(dev_exts);
            vkDestroySurfaceKHR(gpu->instance, gpu->surface, NULL);
            vkDestroyInstance(gpu->instance, NULL);
            free(gpu);
            return NULL;
        }

        free(dev_exts);

        vkGetDeviceQueue(gpu->device, gpu->queue_family, 0,
                         &gpu->graphics_queue);

        volkLoadDevice(gpu->device);

        /* Query RT properties */
        if (gpu->rt_available) {
            /* Query RT pipeline properties + AS properties (Phase C scratch align) */
            VkPhysicalDeviceAccelerationStructurePropertiesKHR as_props = {0};
            as_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
            VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props = {0};
            rt_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
            rt_props.pNext = &as_props;
            VkPhysicalDeviceProperties2 props2 = {0};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &rt_props;
            vkGetPhysicalDeviceProperties2(gpu->physical_device, &props2);

            gpu->rt_handle_size      = rt_props.shaderGroupHandleSize;
            gpu->rt_handle_alignment = rt_props.shaderGroupBaseAlignment;
            gpu->min_as_scratch_align = as_props.minAccelerationStructureScratchOffsetAlignment;
            if (gpu->min_as_scratch_align == 0) gpu->min_as_scratch_align = 256;

            fprintf(stderr, "gpu_vulkan: RT available (handle_size=%u, alignment=%u, scratch_align=%llu)\n",
                    gpu->rt_handle_size, gpu->rt_handle_alignment,
                    (unsigned long long)gpu->min_as_scratch_align);
        } else {
            fprintf(stderr, "gpu_vulkan: RT extensions not available, using raster only\n");
        }

        /* Phase 0 cluster-AS build-sizes probe (env NUSD_RT_CLUSTER_PROBE).
         * volk 1.4.304 predates the cluster entry points, so load them manually
         * via vkGetDeviceProcAddr. Querying CLAS + Cluster-BLAS build sizes for a
         * representative 256-tri/256-vert cluster validates function loading +
         * input-struct construction on this stack before the full GPU build is
         * wired. Pure host query — no allocation, cannot affect rendering. */
#if NUSD_HAS_VK_NV_MEGA_GEOMETRY
        if (cluster_as_enable && getenv("NUSD_RT_CLUSTER_PROBE")) {
            PFN_vkGetClusterAccelerationStructureBuildSizesNV pGetClusterSizes =
                (PFN_vkGetClusterAccelerationStructureBuildSizesNV)
                vkGetDeviceProcAddr(gpu->device, "vkGetClusterAccelerationStructureBuildSizesNV");
            PFN_vkCmdBuildClusterAccelerationStructureIndirectNV pCmdBuildCluster =
                (PFN_vkCmdBuildClusterAccelerationStructureIndirectNV)
                vkGetDeviceProcAddr(gpu->device, "vkCmdBuildClusterAccelerationStructureIndirectNV");
            fprintf(stderr, "gpu_vulkan: cluster-AS fns loaded: getSizes=%p cmdBuild=%p\n",
                    (void*)pGetClusterSizes, (void*)pCmdBuildCluster);
            if (pGetClusterSizes) {
                VkClusterAccelerationStructureTriangleClusterInputNV tri = {0};
                tri.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
                tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                tri.maxClusterUniqueGeometryCount = 1;
                tri.maxClusterTriangleCount = 256;
                tri.maxClusterVertexCount = 256;
                tri.maxTotalTriangleCount = 256;
                tri.maxTotalVertexCount = 256;
                VkClusterAccelerationStructureInputInfoNV cin = {0};
                cin.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
                cin.maxAccelerationStructureCount = 1;
                cin.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
                cin.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
                cin.opInput.pTriangleClusters = &tri;
                VkAccelerationStructureBuildSizesInfoKHR sz = {0};
                sz.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
                pGetClusterSizes(gpu->device, &cin, &sz);
                fprintf(stderr,
                        "gpu_vulkan: CLAS build sizes — accelStruct=%llu scratch=%llu\n",
                        (unsigned long long)sz.accelerationStructureSize,
                        (unsigned long long)sz.buildScratchSize);

                VkClusterAccelerationStructureClustersBottomLevelInputNV bl = {0};
                bl.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
                bl.maxTotalClusterCount = 1;
                bl.maxClusterCountPerAccelerationStructure = 1;
                VkClusterAccelerationStructureInputInfoNV bin = {0};
                bin.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
                bin.maxAccelerationStructureCount = 1;
                bin.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
                bin.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
                bin.opInput.pClustersBottomLevel = &bl;
                VkAccelerationStructureBuildSizesInfoKHR bsz = {0};
                bsz.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
                pGetClusterSizes(gpu->device, &bin, &bsz);
                fprintf(stderr,
                        "gpu_vulkan: ClusterBLAS build sizes — accelStruct=%llu scratch=%llu\n",
                        (unsigned long long)bsz.accelerationStructureSize,
                        (unsigned long long)bsz.buildScratchSize);
            }
        }
#endif

        if (gpu->interop_available) {
            fprintf(stderr, "gpu_vulkan: CUDA interop available (external_memory_fd + external_semaphore_fd)\n");
        }

        /* Log SER availability after device-create so failures during
         * vkCreateDevice surface as device-create errors instead of being
         * masked by a misleading "available" log line. */
        if (gpu->ser_available) {
            const char* hint_str =
                (gpu->ser_reorder_hint == VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_NV)
                    ? "REORDER" : "NONE";
            fprintf(stderr,
                    "gpu_vulkan: SER (VK_NV_ray_tracing_invocation_reorder) available "
                    "(reorder_hint=%s)\n", hint_str);
        } else {
            fprintf(stderr,
                    "gpu_vulkan: SER (VK_NV_ray_tracing_invocation_reorder) unavailable\n");
        }
        fprintf(stderr,
                "gpu_vulkan: fragment-shader barycentrics %s (raster Ptex %s)\n",
                gpu->barycentric_available ? "available" : "unavailable",
                gpu->barycentric_available ? "per-pixel interpolated"
                                           : "flat triangle-averaged");

        /* perf/vk-instrumentation: timestamp pool for per-phase GPU timing.
         * timestampPeriod is ns/tick; timestampComputeAndGraphics == TRUE
         * means vkCmdWriteTimestamp works on graphics+compute queue
         * families (i.e. our graphics_queue). One pool of GPU_PHASE_COUNT*2
         * slots: index = phase*2 + (begin?0:1). */
        {
            VkPhysicalDeviceProperties dprops;
            vkGetPhysicalDeviceProperties(gpu->physical_device, &dprops);
            gpu->timestamp_period_ns = dprops.limits.timestampPeriod;
            gpu->timestamps_supported =
                (dprops.limits.timestampComputeAndGraphics == VK_TRUE &&
                 gpu->timestamp_period_ns > 0.0f) ? 1 : 0;
            if (gpu->timestamps_supported) {
                VkQueryPoolCreateInfo qpci = {0};
                qpci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
                qpci.queryCount = (uint32_t)GPU_PHASE_COUNT * 2u;
                if (vkCreateQueryPool(gpu->device, &qpci, NULL,
                                       &gpu->timestamp_pool) != VK_SUCCESS) {
                    fprintf(stderr,
                        "gpu_vulkan: vkCreateQueryPool failed; timestamps off\n");
                    gpu->timestamps_supported = 0;
                    gpu->timestamp_pool = VK_NULL_HANDLE;
                } else {
                    /* Reset all slots so the first read after a begin/end
                     * pair returns valid data. Requires hostQueryReset. */
                    vkResetQueryPool(gpu->device, gpu->timestamp_pool,
                                     0, qpci.queryCount);
                    fprintf(stderr,
                        "gpu_vulkan: timestamp queries enabled "
                        "(period=%.2f ns/tick, %u slots)\n",
                        (double)gpu->timestamp_period_ns,
                        qpci.queryCount);
                }
            } else {
                fprintf(stderr,
                    "gpu_vulkan: timestamp queries unsupported "
                    "(timestampComputeAndGraphics=%d period=%.2f)\n",
                    (int)dprops.limits.timestampComputeAndGraphics,
                    (double)gpu->timestamp_period_ns);
            }
        }
    }

    /* --- Pipeline cache (must come before any vkCreate*Pipelines* call) --- */
    pc_create(gpu);

    /* --- Swapchain --- */
    if (gpu->surfaceless)
        create_surfaceless_images(gpu);
    else
        create_swapchain(gpu);
    if ((!gpu->surfaceless && gpu->swapchain == VK_NULL_HANDLE) ||
        gpu->image_count == 0) {
        fprintf(stderr, "gpu_vulkan: color target initialization failed\n");
        vkDestroyDevice(gpu->device, NULL);
        if (gpu->surface != VK_NULL_HANDLE)
            vkDestroySurfaceKHR(gpu->instance, gpu->surface, NULL);
        vkDestroyInstance(gpu->instance, NULL);
        free(gpu);
        return NULL;
    }

    /* --- Render pass --- */
    {
        VkAttachmentDescription attachments[2] = {0};

        /* Color attachment */
        attachments[0].format         = gpu->swapchain_format;
        attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout    = gpu->surfaceless
            ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
            : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        /* Depth attachment */
        attachments[1].format         = VK_FORMAT_D32_SFLOAT;
        attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference color_ref = {0};
        color_ref.attachment = 0;
        color_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_ref = {0};
        depth_ref.attachment = 1;
        depth_ref.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {0};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 1;
        subpass.pColorAttachments       = &color_ref;
        subpass.pDepthStencilAttachment = &depth_ref;

        VkSubpassDependency dep = {0};
        dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass    = 0;
        dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = 0;
        dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpci = {0};
        rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpci.attachmentCount = 2;
        rpci.pAttachments    = attachments;
        rpci.subpassCount    = 1;
        rpci.pSubpasses      = &subpass;
        rpci.dependencyCount = 1;
        rpci.pDependencies   = &dep;

        if (vkCreateRenderPass(gpu->device, &rpci, NULL,
                               &gpu->render_pass) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to create render pass\n");
            destroy_swapchain_resources(gpu);
            vkDestroyDevice(gpu->device, NULL);
            if (gpu->surface != VK_NULL_HANDLE)
                vkDestroySurfaceKHR(gpu->instance, gpu->surface, NULL);
            vkDestroyInstance(gpu->instance, NULL);
            free(gpu);
            return NULL;
        }
    }

    /* --- Depth buffer --- */
    create_depth_buffer(gpu);

    /* --- Framebuffers --- */
    create_framebuffers(gpu);

    /* --- Command pool + buffers --- */
    {
        VkCommandPoolCreateInfo cpci = {0};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = gpu->queue_family;

        if (vkCreateCommandPool(gpu->device, &cpci, NULL,
                                &gpu->command_pool) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to create command pool\n");
            destroy_swapchain_resources(gpu);
            vkDestroyRenderPass(gpu->device, gpu->render_pass, NULL);
            vkDestroyDevice(gpu->device, NULL);
            vkDestroySurfaceKHR(gpu->instance, gpu->surface, NULL);
            vkDestroyInstance(gpu->instance, NULL);
            free(gpu);
            return NULL;
        }

        gpu->command_buffers =
            malloc(gpu->image_count * sizeof(VkCommandBuffer));

        VkCommandBufferAllocateInfo cbai = {0};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = gpu->command_pool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = gpu->image_count;

        if (vkAllocateCommandBuffers(gpu->device, &cbai,
                                     gpu->command_buffers) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to allocate command buffers\n");
        }

        /* Allocate cmd-cache shadow arrays sized to image_count. */
        gpu->rt_cmd_cache_valid_count = (int)gpu->image_count;
        gpu->rt_cmd_cache_valid = (int*)calloc(gpu->image_count, sizeof(int));
        gpu->rt_cmd_cache_pc    = (GpuRtPushConstants*)calloc(gpu->image_count,
                                                              sizeof(GpuRtPushConstants));
    }

    /* --- Sync objects --- */
    {
        VkSemaphoreCreateInfo sci = {0};
        sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fci = {0};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkCreateSemaphore(gpu->device, &sci, NULL,
                              &gpu->image_available[i]);
            vkCreateSemaphore(gpu->device, &sci, NULL,
                              &gpu->render_finished[i]);
            vkCreateFence(gpu->device, &fci, NULL, &gpu->in_flight[i]);
        }
    }

    /* --- Pipeline layout (shared) --- */
    {
        VkPushConstantRange pcr = {0};
        pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                         VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                         VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
                         VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr.offset     = 0;
        pcr.size       = PUSH_CONSTANT_SIZE;

        VkPipelineLayoutCreateInfo plci = {0};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;

        if (vkCreatePipelineLayout(gpu->device, &plci, NULL,
                                   &gpu->pipeline_layout) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to create pipeline layout\n");
            for (int j = 0; j < MAX_FRAMES_IN_FLIGHT; j++) {
                vkDestroySemaphore(gpu->device, gpu->image_available[j], NULL);
                vkDestroySemaphore(gpu->device, gpu->render_finished[j], NULL);
                vkDestroyFence(gpu->device, gpu->in_flight[j], NULL);
            }
            free(gpu->command_buffers);
            vkDestroyCommandPool(gpu->device, gpu->command_pool, NULL);
            destroy_swapchain_resources(gpu);
            vkDestroyRenderPass(gpu->device, gpu->render_pass, NULL);
            vkDestroyDevice(gpu->device, NULL);
            vkDestroySurfaceKHR(gpu->instance, gpu->surface, NULL);
            vkDestroyInstance(gpu->instance, NULL);
            free(gpu);
            return NULL;
        }
    }

    gpu->frame_index = 0;
    gpu->current_cmd = VK_NULL_HANDLE;

    /* perf/mem-pool: optionally create the three RT memory pools at init time.
     *
     * The HOST_VISIBLE staging pool's vkAllocateMemory + vkMapMemory takes
     * ~600 ms for 2 GB (the kernel commits + pins all pages eagerly when
     * we map). DEVICE_LOCAL pools are much faster (~60 ms each, just VA
     * reservation; physical pages are committed lazily on bind).
     *
     * Doing this inside `gpu_build_rt_scene` (the timed `build_accel`
     * call) would add ~700 ms of one-shot init to every bench run. By
     * moving pool create into `gpu_init` (called from NuRenderer
     * construction, before the bench timer starts) the cost is amortised
     * across all subsequent build_accel and render calls.
     *
     * DSX raster/geometry benchmarks should not pay the staging map cost or
     * reserve RT VA. Leave pools lazy by default; set NUSD_VK_EAGER_RT_POOLS=1
     * to keep the old benchmark behaviour and move the first-create cost into
     * renderer construction. */
    const char* eager_rt_pools = getenv("NUSD_VK_EAGER_RT_POOLS");
    if (gpu->rt_available && eager_rt_pools && eager_rt_pools[0] != '\0' &&
        eager_rt_pools[0] != '0') {
        rt_ensure_pool(gpu, RT_POOL_RESIDENT);
        rt_ensure_pool(gpu, RT_POOL_TRANSIENT);
        rt_ensure_pool(gpu, RT_POOL_STAGING);
    }

    if (gpu->rt_available && getenv("NUSD_RT_CLUSTER_SELFTEST")) {
#if NUSD_HAS_VK_NV_MEGA_GEOMETRY
        nu_cluster_build_selftest(gpu);
#else
        fprintf(stderr,
                "gpu_vulkan: NUSD_RT_CLUSTER_SELFTEST ignored; Vulkan headers "
                "do not expose RTX Mega Geometry\n");
#endif
    }

    return gpu;
}

/* Forward decls for tiled_destroy_* (definitions are further down). */
static void tiled_destroy_depth(Gpu* gpu);
static void tiled_destroy_segmentation(Gpu* gpu);
static void tiled_destroy_normals(Gpu* gpu);

/* Forward declaration: curve resource cleanup (Phase 11.5.3). Definition
 * lives near gpu_upload_curve_data / gpu_build_curve_blas where the
 * matching upload/build code lives. */
static void curve_destroy(Gpu* gpu);

void gpu_shutdown(Gpu* gpu)
{
    if (!gpu) return;

    vkDeviceWaitIdle(gpu->device);

    /* Shut down DLSS before destroying Vulkan objects */
    if (gpu->dlss_active)
        gpu_dlss_shutdown(gpu);

    /* Phase B: tear down deferred-shading compute pipeline before
     * gpu_destroy_rt_scene(), since that frees scene_data_buf which the
     * descriptor sets reference. Idempotent. */
    gpu_destroy_deferred_pipeline(gpu);

    gpu_destroy_rt_scene(gpu);
    gpu_destroy_materials(gpu);
    if (gpu->light_ssbo_buf) {
        vkDestroyBuffer(gpu->device, gpu->light_ssbo_buf, NULL);
        vkFreeMemory(gpu->device, gpu->light_ssbo_mem, NULL);
        gpu->light_ssbo_buf = VK_NULL_HANDLE;
        gpu->light_ssbo_mem = VK_NULL_HANDLE;
        gpu->light_count = 0;
    }

    /* VK3DGRT: tear down splat BLAS/TLAS, SSBOs, RT pipeline. Idempotent. */
    gpu_gs_destroy(gpu);

    /* Phase 11.5.3: belt-and-suspenders — gpu_destroy_rt_scene early-
     * exits when rt_built==0, so a renderer that uploaded curve data
     * but never built the RT scene would still leak. curve_destroy
     * is idempotent. */
    curve_destroy(gpu);

    gpu_overlay_shutdown(gpu);

    /* Persist the pipeline cache to disk before destroying it. Best-
     * effort: failure is logged but does not affect shutdown. */
    pc_save(gpu);
    if (gpu->pipeline_cache) {
        vkDestroyPipelineCache(gpu->device, gpu->pipeline_cache, NULL);
        gpu->pipeline_cache = VK_NULL_HANDLE;
    }

    vkDestroyPipelineLayout(gpu->device, gpu->pipeline_layout, NULL);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(gpu->device, gpu->image_available[i], NULL);
        vkDestroySemaphore(gpu->device, gpu->render_finished[i], NULL);
        vkDestroyFence(gpu->device, gpu->in_flight[i], NULL);
    }

    /* Free cached tiled cmd buffers (perf/cmdbuf-cache).  Must happen
     * BEFORE vkDestroyCommandPool implicitly frees them, otherwise the
     * handles are dangling. */
    for (int i = 0; i < 2; i++) {
        if (gpu->tiled_cached_cmd[i]) {
            vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1,
                                 &gpu->tiled_cached_cmd[i]);
            gpu->tiled_cached_cmd[i] = VK_NULL_HANDLE;
        }
        gpu->tiled_cmd_cache_valid[i] = 0;
    }

    free(gpu->rt_cmd_cache_valid);
    free(gpu->rt_cmd_cache_pc);
    gpu->rt_cmd_cache_valid = NULL;
    gpu->rt_cmd_cache_pc    = NULL;
    gpu->rt_cmd_cache_valid_count = 0;

    free(gpu->command_buffers);
    vkDestroyCommandPool(gpu->device, gpu->command_pool, NULL);

    /* Persistent readback staging buffer */
    if (gpu->readback_buf) {
        if (gpu->readback_mapped)
            vkUnmapMemory(gpu->device, gpu->readback_mem);
        vkDestroyBuffer(gpu->device, gpu->readback_buf, NULL);
        vkFreeMemory(gpu->device, gpu->readback_mem, NULL);
    }

    /* Single-camera CUDA interop pixels buffer */
    if (gpu->pixels_interop_fd >= 0) {
        close(gpu->pixels_interop_fd);
        gpu->pixels_interop_fd = -1;
    }
    if (gpu->pixels_interop_buf) {
        vkDestroyBuffer(gpu->device, gpu->pixels_interop_buf, NULL);
        gpu->pixels_interop_buf = VK_NULL_HANDLE;
    }
    if (gpu->pixels_interop_mem) {
        vkFreeMemory(gpu->device, gpu->pixels_interop_mem, NULL);
        gpu->pixels_interop_mem = VK_NULL_HANDLE;
    }

    /* Tiled buffers — must be released before vkDestroyDevice */
    tiled_destroy_depth(gpu);
    tiled_destroy_segmentation(gpu);
    tiled_destroy_normals(gpu);

    /* GPU-driven TLAS instance translation (PR 2): destroy compute
     * pipeline + descriptor pool + the imported transforms / indices SSBOs. */
    if (gpu->tlas_translate_pipeline) {
        vkDestroyPipeline(gpu->device, gpu->tlas_translate_pipeline, NULL);
        gpu->tlas_translate_pipeline = VK_NULL_HANDLE;
    }
    if (gpu->tlas_translate_pl_layout) {
        vkDestroyPipelineLayout(gpu->device, gpu->tlas_translate_pl_layout, NULL);
        gpu->tlas_translate_pl_layout = VK_NULL_HANDLE;
    }
    if (gpu->tlas_translate_ds_pool) {
        vkDestroyDescriptorPool(gpu->device, gpu->tlas_translate_ds_pool, NULL);
        gpu->tlas_translate_ds_pool = VK_NULL_HANDLE;
        gpu->tlas_translate_ds_set  = VK_NULL_HANDLE;
    }
    if (gpu->tlas_translate_ds_layout) {
        vkDestroyDescriptorSetLayout(gpu->device, gpu->tlas_translate_ds_layout, NULL);
        gpu->tlas_translate_ds_layout = VK_NULL_HANDLE;
    }
    gpu->tlas_translate_built = 0;

    gpu->tlas_xforms_fd = -1;  /* transform fds are caller-owned one-shot exports */
    if (gpu->tlas_xforms_buf) {
        vkDestroyBuffer(gpu->device, gpu->tlas_xforms_buf, NULL);
        gpu->tlas_xforms_buf = VK_NULL_HANDLE;
    }
    if (gpu->tlas_xforms_mem) {
        vkFreeMemory(gpu->device, gpu->tlas_xforms_mem, NULL);
        gpu->tlas_xforms_mem = VK_NULL_HANDLE;
    }
    if (gpu->tlas_indices_buf) {
        vkDestroyBuffer(gpu->device, gpu->tlas_indices_buf, NULL);
        gpu->tlas_indices_buf = VK_NULL_HANDLE;
    }
    if (gpu->tlas_indices_mem) {
        vkFreeMemory(gpu->device, gpu->tlas_indices_mem, NULL);
        gpu->tlas_indices_mem = VK_NULL_HANDLE;
    }

    /* perf/vk-instrumentation: destroy timestamp pool */
    if (gpu->timestamp_pool) {
        vkDestroyQueryPool(gpu->device, gpu->timestamp_pool, NULL);
        gpu->timestamp_pool = VK_NULL_HANDLE;
    }

    /* perf/mem-pool: destroy pre-allocated memory pools (must come before
     * vkDestroyDevice; gpu_destroy_rt_scene already destroyed any
     * VkBuffer handles bound into the pools). */
    if (gpu->resident_pool)  { gpu_pool_destroy((GpuMemPool*)gpu->resident_pool);  gpu->resident_pool  = NULL; }
    if (gpu->transient_pool) { gpu_pool_destroy((GpuMemPool*)gpu->transient_pool); gpu->transient_pool = NULL; }
    if (gpu->staging_pool)   { gpu_pool_destroy((GpuMemPool*)gpu->staging_pool);   gpu->staging_pool   = NULL; }

    destroy_swapchain_resources(gpu);

    vkDestroyRenderPass(gpu->device, gpu->render_pass, NULL);
    vkDestroyDevice(gpu->device, NULL);
    if (gpu->surface != VK_NULL_HANDLE)
        vkDestroySurfaceKHR(gpu->instance, gpu->surface, NULL);
    if (gpu->debug_messenger)
        vkDestroyDebugUtilsMessengerEXT(gpu->instance, gpu->debug_messenger, NULL);
    vkDestroyInstance(gpu->instance, NULL);

    free(gpu);
}

void gpu_set_headless(Gpu* gpu, int headless)
{
    if (gpu) gpu->headless = headless;
}

/* ---- GPU phase timings (perf/vk-instrumentation) ----
 *
 * Wraps Vulkan command buffer regions with both VK_EXT_debug_utils labels
 * (so nsys's vulkan_gpu_marker_sum populates) and timestamp queries (so
 * the host can query per-phase ms via nu_get_phase_timings_ms).
 *
 * The reset is per-phase (just the 2 slots) so phases written in
 * different command buffers don't trample each other.
 */

int gpu_debug_utils_enabled(Gpu* gpu) { return gpu ? gpu->debug_utils_enabled : 0; }
int gpu_timestamps_supported(Gpu* gpu) { return gpu ? gpu->timestamps_supported : 0; }

void gpu_phase_begin(Gpu* gpu, void* cmd_in, GpuPhase phase, const char* label)
{
    if (!gpu || !cmd_in) return;
    if ((unsigned)phase >= (unsigned)GPU_PHASE_COUNT) return;
    VkCommandBuffer cmd = (VkCommandBuffer)cmd_in;

    if (gpu->debug_utils_enabled && label) {
        VkDebugUtilsLabelEXT lbl = {0};
        lbl.sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        lbl.pLabelName = label;
        /* Distinct color per phase makes nsys timeline easier to read. */
        const float colors[GPU_PHASE_COUNT][4] = {
            { 0.95f, 0.30f, 0.30f, 1.0f }, /* RT_DISPATCH       red */
            { 0.30f, 0.95f, 0.30f, 1.0f }, /* PIXEL_READBACK    green */
            { 0.30f, 0.55f, 0.95f, 1.0f }, /* BLAS_BUILD        blue */
            { 0.95f, 0.85f, 0.20f, 1.0f }, /* TLAS_BUILD        gold */
            { 0.65f, 0.40f, 0.95f, 1.0f }, /* CURVE_BLAS_BUILD  purple */
            { 0.95f, 0.55f, 0.20f, 1.0f }, /* UPLOAD_SEGS       orange */
            { 0.20f, 0.85f, 0.85f, 1.0f }, /* UPLOAD_AABBS      cyan */
            { 0.85f, 0.40f, 0.65f, 1.0f }, /* UPLOAD_COLORS     pink */
            { 0.95f, 0.20f, 0.55f, 1.0f }, /* TRACE_RAYS_TILED  magenta */
            { 0.20f, 0.55f, 0.95f, 1.0f }, /* DEFERRED_COMPUTE  azure */
        };
        memcpy(lbl.color, colors[(int)phase], sizeof(lbl.color));
        vkCmdBeginDebugUtilsLabelEXT(cmd, &lbl);
    }

    if (gpu->timestamps_supported) {
        uint32_t slot = (uint32_t)phase * 2u;
        vkCmdResetQueryPool(cmd, gpu->timestamp_pool, slot, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            gpu->timestamp_pool, slot);
        gpu->phase_pending[(int)phase] = 1;
    }
}

void gpu_phase_end(Gpu* gpu, void* cmd_in, GpuPhase phase)
{
    if (!gpu || !cmd_in) return;
    if ((unsigned)phase >= (unsigned)GPU_PHASE_COUNT) return;
    VkCommandBuffer cmd = (VkCommandBuffer)cmd_in;

    if (gpu->timestamps_supported && gpu->phase_pending[(int)phase]) {
        uint32_t slot = (uint32_t)phase * 2u + 1u;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            gpu->timestamp_pool, slot);
    }

    if (gpu->debug_utils_enabled) {
        vkCmdEndDebugUtilsLabelEXT(cmd);
    }
}

void gpu_phase_resolve(Gpu* gpu, GpuPhase phase)
{
    if (!gpu) return;
    if (!gpu->timestamps_supported) return;
    if ((unsigned)phase >= (unsigned)GPU_PHASE_COUNT) return;
    if (!gpu->phase_pending[(int)phase]) return;

    uint32_t slot = (uint32_t)phase * 2u;
    uint64_t ts[2] = {0, 0};
    /* No WAIT_BIT: caller is responsible for guaranteeing the writes have
     * landed (vkDeviceWaitIdle / fence wait / vkQueueWaitIdle). Using
     * WAIT_BIT here would block in lib code, which we don't want. */
    VkResult r = vkGetQueryPoolResults(gpu->device, gpu->timestamp_pool,
                                        slot, 2,
                                        sizeof(ts), ts, sizeof(uint64_t),
                                        VK_QUERY_RESULT_64_BIT);
    if (r == VK_SUCCESS && ts[1] >= ts[0]) {
        uint64_t ticks = ts[1] - ts[0];
        gpu->phase_timing_ns[(int)phase] =
            (uint64_t)((double)ticks * (double)gpu->timestamp_period_ns);
    }
    gpu->phase_pending[(int)phase] = 0;
}

void gpu_phase_resolve_all(Gpu* gpu)
{
    if (!gpu) return;
    for (int p = 0; p < (int)GPU_PHASE_COUNT; p++) {
        gpu_phase_resolve(gpu, (GpuPhase)p);
    }
}

float gpu_phase_get_ms(Gpu* gpu, GpuPhase phase)
{
    if (!gpu) return 0.0f;
    if ((unsigned)phase >= (unsigned)GPU_PHASE_COUNT) return 0.0f;
    return (float)((double)gpu->phase_timing_ns[(int)phase] / 1.0e6);
}

void gpu_resize(Gpu* gpu, int width, int height)
{
    if (!gpu) return;

    gpu->width  = width;
    gpu->height = height;

    vkDeviceWaitIdle(gpu->device);

    /* Resize destroys swapchain images / depth image; any cached cmd
     * buffer that referenced those resources is now stale. */
    gpu_invalidate_rt_cmd_cache(gpu);
    gpu_invalidate_tiled_cmd_cache(gpu);

    /* Destroy overlay framebuffers before swapchain recreation */
    if (gpu->overlay_inited) {
        for (uint32_t i = 0; i < gpu->image_count; i++)
            vkDestroyFramebuffer(gpu->device, gpu->overlay_framebuffers[i], NULL);
        free(gpu->overlay_framebuffers);
        gpu->overlay_framebuffers = NULL;
    }

    destroy_swapchain_resources(gpu);
    if (gpu->surfaceless)
        create_surfaceless_images(gpu);
    else
        create_swapchain(gpu);
    create_depth_buffer(gpu);
    create_framebuffers(gpu);

    /* Recreate overlay framebuffers with new swapchain */
    if (gpu->overlay_inited) {
        gpu->overlay_framebuffers = (VkFramebuffer*)calloc(gpu->image_count, sizeof(VkFramebuffer));
        for (uint32_t i = 0; i < gpu->image_count; i++) {
            VkFramebufferCreateInfo fbci = {0};
            fbci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fbci.renderPass      = gpu->overlay_render_pass;
            fbci.attachmentCount = 1;
            fbci.pAttachments    = &gpu->swapchain_views[i];
            fbci.width           = gpu->swapchain_extent.width;
            fbci.height          = gpu->swapchain_extent.height;
            fbci.layers          = 1;
            vkCreateFramebuffer(gpu->device, &fbci, NULL, &gpu->overlay_framebuffers[i]);
        }
    }
}

/* ---- Buffer ---- */

GpuBuffer gpu_create_buffer(Gpu* gpu, const GpuBufferDesc* desc)
{
    GpuBuffer buf = calloc(1, sizeof(struct GpuBuffer_s));
    if (!buf) return NULL;
    buf->size = desc->size;

    VkBufferUsageFlags usage = 0;
    if (desc->usage == GPU_BUFFER_VERTEX)
        usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    else
        usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    /* Add RT usage flags if ray tracing is available */
    if (gpu->rt_available) {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
              |  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
              |  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }

    if (desc->data != NULL) {
        /* Device-local buffer with staging upload */
        usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        /* Create device-local buffer first */
        VkBufferCreateInfo bci = {0};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = desc->size;
        bci.usage       = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(gpu->device, &bci, NULL, &buf->buffer) != VK_SUCCESS) {
            fprintf(stderr, "gpu_create_buffer: vkCreateBuffer failed (size=%zu)\n",
                    (size_t)desc->size);
            free(buf);
            return NULL;
        }

        VkMemoryRequirements mem_req;
        vkGetBufferMemoryRequirements(gpu->device, buf->buffer, &mem_req);

        VkMemoryAllocateFlagsInfo alloc_flags = {0};
        alloc_flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        if (gpu->rt_available)
            alloc_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo ai = {0};
        ai.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.pNext          = gpu->rt_available ? &alloc_flags : NULL;
        ai.allocationSize = mem_req.size;
        ai.memoryTypeIndex = find_memory_type(
            gpu->physical_device, mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (gpu_alloc_memory(gpu, &ai, &buf->memory) != VK_SUCCESS) {
            fprintf(stderr, "gpu_create_buffer: device memory alloc failed (size=%zu)\n",
                    (size_t)mem_req.size);
            vkDestroyBuffer(gpu->device, buf->buffer, NULL);
            free(buf);
            return NULL;
        }
        vkBindBufferMemory(gpu->device, buf->buffer, buf->memory, 0);

        /* Chunked staging upload — host-visible heaps are often limited (~256 MB).
         * Upload in STAGING_CHUNK_SIZE pieces to avoid failing large allocations. */
        #define STAGING_CHUNK_SIZE (256ULL * 1024 * 1024)  /* 256 MB */

        VkDeviceSize remaining = desc->size;
        VkDeviceSize offset = 0;
        const uint8_t* src = (const uint8_t*)desc->data;

        while (remaining > 0) {
            VkDeviceSize chunk = remaining < STAGING_CHUNK_SIZE ? remaining : STAGING_CHUNK_SIZE;

            VkBuffer staging_buf;
            VkDeviceMemory staging_mem;

            VkBufferCreateInfo staging_bci = {0};
            staging_bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            staging_bci.size        = chunk;
            staging_bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            staging_bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateBuffer(gpu->device, &staging_bci, NULL, &staging_buf) != VK_SUCCESS) {
                fprintf(stderr, "gpu_create_buffer: staging buffer create failed (chunk=%zu)\n",
                        (size_t)chunk);
                vkDestroyBuffer(gpu->device, buf->buffer, NULL);
                vkFreeMemory(gpu->device, buf->memory, NULL);
                free(buf);
                return NULL;
            }

            VkMemoryRequirements staging_req;
            vkGetBufferMemoryRequirements(gpu->device, staging_buf, &staging_req);

            VkMemoryAllocateInfo staging_ai = {0};
            staging_ai.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            staging_ai.allocationSize = staging_req.size;
            staging_ai.memoryTypeIndex = find_memory_type(
                gpu->physical_device, staging_req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            if (vkAllocateMemory(gpu->device, &staging_ai, NULL, &staging_mem) != VK_SUCCESS) {
                fprintf(stderr, "gpu_create_buffer: staging alloc failed (chunk=%zu)\n",
                        (size_t)chunk);
                vkDestroyBuffer(gpu->device, staging_buf, NULL);
                vkDestroyBuffer(gpu->device, buf->buffer, NULL);
                vkFreeMemory(gpu->device, buf->memory, NULL);
                free(buf);
                return NULL;
            }
            vkBindBufferMemory(gpu->device, staging_buf, staging_mem, 0);

            void* mapped;
            vkMapMemory(gpu->device, staging_mem, 0, chunk, 0, &mapped);
            memcpy(mapped, src + offset, (size_t)chunk);
            vkUnmapMemory(gpu->device, staging_mem);

            /* Copy chunk via one-shot command buffer */
            VkCommandBufferAllocateInfo cbai = {0};
            cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cbai.commandPool        = gpu->command_pool;
            cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbai.commandBufferCount = 1;

            VkCommandBuffer cmd;
            vkAllocateCommandBuffers(gpu->device, &cbai, &cmd);

            VkCommandBufferBeginInfo begin_info = {0};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &begin_info);

            VkBufferCopy copy_region = {0};
            copy_region.srcOffset = 0;
            copy_region.dstOffset = offset;
            copy_region.size      = chunk;
            vkCmdCopyBuffer(cmd, staging_buf, buf->buffer, 1, &copy_region);

            vkEndCommandBuffer(cmd);

            VkSubmitInfo submit = {0};
            submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers    = &cmd;

            vkQueueSubmit(gpu->graphics_queue, 1, &submit, VK_NULL_HANDLE);
            vkQueueWaitIdle(gpu->graphics_queue);

            vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &cmd);
            vkDestroyBuffer(gpu->device, staging_buf, NULL);
            vkFreeMemory(gpu->device, staging_mem, NULL);

            offset    += chunk;
            remaining -= chunk;
        }
        #undef STAGING_CHUNK_SIZE
    } else {
        /* Host-visible buffer (no initial data) */
        VkBufferCreateInfo bci = {0};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = desc->size;
        bci.usage       = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCreateBuffer(gpu->device, &bci, NULL, &buf->buffer);

        VkMemoryRequirements mem_req;
        vkGetBufferMemoryRequirements(gpu->device, buf->buffer, &mem_req);

        VkMemoryAllocateInfo ai = {0};
        ai.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mem_req.size;
        ai.memoryTypeIndex = find_memory_type(
            gpu->physical_device, mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        gpu_alloc_memory(gpu, &ai, &buf->memory);
        vkBindBufferMemory(gpu->device, buf->buffer, buf->memory, 0);
    }

    return buf;
}

void gpu_destroy_buffer(Gpu* gpu, GpuBuffer buf)
{
    if (!gpu || !buf) return;
    vkDestroyBuffer(gpu->device, buf->buffer, NULL);
    vkFreeMemory(gpu->device, buf->memory, NULL);
    free(buf);
}

/* ---- Pipeline ---- */

static VkFormat gpu_vertex_format_to_vk(GpuVertexFormat f)
{
    switch (f) {
        case GPU_FORMAT_FLOAT3: return VK_FORMAT_R32G32B32_SFLOAT;
        case GPU_FORMAT_FLOAT2: return VK_FORMAT_R32G32_SFLOAT;
        case GPU_FORMAT_UINT:   return VK_FORMAT_R32_UINT;
        case GPU_FORMAT_FLOAT:  return VK_FORMAT_R32_SFLOAT;
        case GPU_FORMAT_FLOAT4: return VK_FORMAT_R32G32B32A32_SFLOAT;
        default:                return VK_FORMAT_R32G32B32_SFLOAT;
    }
}

GpuPipeline gpu_create_pipeline(Gpu* gpu, const GpuPipelineDesc* desc)
{
    GpuPipeline pipe = calloc(1, sizeof(struct GpuPipeline_s));
    if (!pipe) return NULL;

    VkShaderModule vert_mod = create_shader_module(gpu->device,
                                                   desc->vert_spv,
                                                   desc->vert_size);
    VkShaderModule frag_mod = create_shader_module(gpu->device,
                                                   desc->frag_spv,
                                                   desc->frag_size);
    int has_tess = desc->tesc_spv && desc->tese_spv &&
                   desc->tesc_size > 0 && desc->tese_size > 0;
    int half_tess = (desc->tesc_spv || desc->tese_spv ||
                     desc->tesc_size || desc->tese_size) && !has_tess;
    if (half_tess || !vert_mod || !frag_mod) {
        fprintf(stderr, "gpu_vulkan: invalid graphics shader set\n");
        if (vert_mod) vkDestroyShaderModule(gpu->device, vert_mod, NULL);
        if (frag_mod) vkDestroyShaderModule(gpu->device, frag_mod, NULL);
        free(pipe);
        return NULL;
    }
    VkShaderModule tesc_mod = VK_NULL_HANDLE;
    VkShaderModule tese_mod = VK_NULL_HANDLE;
    if (has_tess) {
        tesc_mod = create_shader_module(gpu->device,
                                        desc->tesc_spv,
                                        desc->tesc_size);
        tese_mod = create_shader_module(gpu->device,
                                        desc->tese_spv,
                                        desc->tese_size);
        if (!tesc_mod || !tese_mod) {
            fprintf(stderr, "gpu_vulkan: failed to create tessellation shader modules\n");
            if (tesc_mod) vkDestroyShaderModule(gpu->device, tesc_mod, NULL);
            if (tese_mod) vkDestroyShaderModule(gpu->device, tese_mod, NULL);
            vkDestroyShaderModule(gpu->device, vert_mod, NULL);
            vkDestroyShaderModule(gpu->device, frag_mod, NULL);
            free(pipe);
            return NULL;
        }
    }

    /* Shader stages */
    VkPipelineShaderStageCreateInfo stages[4] = {0};
    uint32_t stage_count = 0;
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_mod;
    stages[0].pName  = "main";
    stage_count++;

    if (has_tess) {
        stages[stage_count].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[stage_count].stage  = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        stages[stage_count].module = tesc_mod;
        stages[stage_count].pName  = "main";
        stage_count++;

        stages[stage_count].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[stage_count].stage  = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        stages[stage_count].module = tese_mod;
        stages[stage_count].pName  = "main";
        stage_count++;
    }

    stages[stage_count].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[stage_count].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[stage_count].module = frag_mod;
    stages[stage_count].pName  = "main";
    stage_count++;

    /* Vertex input. Binding 0 is always the per-vertex stream; binding 1 is an
     * optional per-instance stream (instanced raster PI draw). */
    int has_instance_binding =
        (desc->instance_stride > 0 && desc->n_instance_attribs > 0);
    VkVertexInputBindingDescription bindings[2] = {0};
    bindings[0].binding   = 0;
    bindings[0].stride    = desc->vertex_stride;
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    uint32_t nbindings = 1;
    if (has_instance_binding) {
        bindings[1].binding   = 1;
        bindings[1].stride    = desc->instance_stride;
        bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
        nbindings = 2;
    }

    uint32_t total_attribs = desc->nattribs +
        (has_instance_binding ? desc->n_instance_attribs : 0u);
    VkVertexInputAttributeDescription* attrs = NULL;
    if (total_attribs > 0) {
        attrs = malloc(total_attribs * sizeof(VkVertexInputAttributeDescription));
        uint32_t a = 0;
        for (uint32_t i = 0; i < desc->nattribs; i++, a++) {
            attrs[a].location = desc->attribs[i].location;
            attrs[a].binding  = 0;
            attrs[a].offset   = desc->attribs[i].offset;
            attrs[a].format   = gpu_vertex_format_to_vk(desc->attribs[i].format);
        }
        if (has_instance_binding) {
            for (uint32_t i = 0; i < desc->n_instance_attribs; i++, a++) {
                attrs[a].location = desc->instance_attribs[i].location;
                attrs[a].binding  = 1;
                attrs[a].offset   = desc->instance_attribs[i].offset;
                attrs[a].format   = gpu_vertex_format_to_vk(desc->instance_attribs[i].format);
            }
        }
    }

    VkPipelineVertexInputStateCreateInfo vertex_input = {0};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount   = nbindings;
    vertex_input.pVertexBindingDescriptions      = bindings;
    vertex_input.vertexAttributeDescriptionCount = total_attribs;
    vertex_input.pVertexAttributeDescriptions    = attrs;

    /* Input assembly */
    VkPipelineInputAssemblyStateCreateInfo input_asm = {0};
    input_asm.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_asm.topology = has_tess
        ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST
        : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_asm.primitiveRestartEnable = VK_FALSE;

    VkPipelineTessellationStateCreateInfo tess = {0};
    tess.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    tess.patchControlPoints = desc->patch_control_points
        ? desc->patch_control_points
        : 4u;

    /* Viewport / scissor (dynamic) */
    VkPipelineViewportStateCreateInfo viewport_state = {0};
    viewport_state.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount  = 1;

    /* Rasterization */
    VkPipelineRasterizationStateCreateInfo raster = {0};
    raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode    = VK_CULL_MODE_NONE;
    raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth   = 1.0f;

    /* Multisampling */
    VkPipelineMultisampleStateCreateInfo multisample = {0};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    /* Depth stencil */
    VkPipelineDepthStencilStateCreateInfo depth = {0};
    depth.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable  = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    /* Color blending (opaque) */
    VkPipelineColorBlendAttachmentState blend_att = {0};
    blend_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                               VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT |
                               VK_COLOR_COMPONENT_A_BIT;
    blend_att.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo blend = {0};
    blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments    = &blend_att;

    /* Dynamic state */
    VkDynamicState dyn_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dynamic = {0};
    dynamic.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates    = dyn_states;

    /* Create pipeline */
    VkGraphicsPipelineCreateInfo pci = {0};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = stage_count;
    pci.pStages             = stages;
    pci.pVertexInputState   = &vertex_input;
    pci.pInputAssemblyState = &input_asm;
    pci.pTessellationState  = has_tess ? &tess : NULL;
    pci.pViewportState      = &viewport_state;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState   = &multisample;
    pci.pDepthStencilState  = &depth;
    pci.pColorBlendState    = &blend;
    pci.pDynamicState       = &dynamic;
    /* Pipelines that need to sample materials use mat_pipeline_layout
     * (push-constants + descriptor set 0 = mat_desc_layout). */
    VkPipelineLayout chosen_layout = (desc->use_mat_layout && gpu->mat_pipeline_layout)
        ? gpu->mat_pipeline_layout
        : gpu->pipeline_layout;
    pci.layout              = chosen_layout;
    pci.renderPass          = gpu->render_pass;
    pci.subpass             = 0;

    if (vkCreateGraphicsPipelines(gpu->device, gpu->pipeline_cache, 1, &pci, NULL,
                                  &pipe->pipeline) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create graphics pipeline\n");
        free(pipe);
        pipe = NULL;
    } else {
        pipe->layout = chosen_layout;
    }

    free(attrs);
    vkDestroyShaderModule(gpu->device, vert_mod, NULL);
    if (tesc_mod) vkDestroyShaderModule(gpu->device, tesc_mod, NULL);
    if (tese_mod) vkDestroyShaderModule(gpu->device, tese_mod, NULL);
    vkDestroyShaderModule(gpu->device, frag_mod, NULL);

    return pipe;
}

void gpu_destroy_pipeline(Gpu* gpu, GpuPipeline pipe)
{
    if (!gpu || !pipe) return;
    vkDestroyPipeline(gpu->device, pipe->pipeline, NULL);
    free(pipe);
}

GpuPipeline gpu_create_shadow_pipeline(Gpu* gpu, const GpuPipelineDesc* desc)
{
    if (!gpu->shadow_pipeline_layout) {
        fprintf(stderr, "gpu_vulkan: shadow pipeline layout not created (call gpu_build_rt_scene first)\n");
        return NULL;
    }

    GpuPipeline pipe = calloc(1, sizeof(struct GpuPipeline_s));
    if (!pipe) return NULL;

    VkShaderModule vert_mod = create_shader_module(gpu->device,
                                                   desc->vert_spv,
                                                   desc->vert_size);
    VkShaderModule frag_mod = create_shader_module(gpu->device,
                                                   desc->frag_spv,
                                                   desc->frag_size);

    VkPipelineShaderStageCreateInfo stages[2] = {0};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_mod;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_mod;
    stages[1].pName  = "main";

    VkVertexInputBindingDescription binding = {0};
    binding.binding   = 0;
    binding.stride    = desc->vertex_stride;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription* attrs = NULL;
    if (desc->nattribs > 0) {
        attrs = malloc(desc->nattribs * sizeof(VkVertexInputAttributeDescription));
        for (uint32_t i = 0; i < desc->nattribs; i++) {
            attrs[i].location = desc->attribs[i].location;
            attrs[i].binding  = 0;
            attrs[i].offset   = desc->attribs[i].offset;
            attrs[i].format   = VK_FORMAT_R32G32B32_SFLOAT;
        }
    }

    VkPipelineVertexInputStateCreateInfo vertex_input = {0};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount   = 1;
    vertex_input.pVertexBindingDescriptions      = &binding;
    vertex_input.vertexAttributeDescriptionCount = desc->nattribs;
    vertex_input.pVertexAttributeDescriptions    = attrs;

    VkPipelineInputAssemblyStateCreateInfo input_asm = {0};
    input_asm.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_asm.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_state = {0};
    viewport_state.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo raster = {0};
    raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode    = VK_CULL_MODE_NONE;
    raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample = {0};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth = {0};
    depth.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable  = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blend_att = {0};
    blend_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend = {0};
    blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments    = &blend_att;

    VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic = {0};
    dynamic.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates    = dyn_states;

    VkGraphicsPipelineCreateInfo pci = {0};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &vertex_input;
    pci.pInputAssemblyState = &input_asm;
    pci.pViewportState      = &viewport_state;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState   = &multisample;
    pci.pDepthStencilState  = &depth;
    pci.pColorBlendState    = &blend;
    pci.pDynamicState       = &dynamic;
    pci.layout              = gpu->shadow_pipeline_layout;
    pci.renderPass          = gpu->render_pass;
    pci.subpass             = 0;

    if (vkCreateGraphicsPipelines(gpu->device, gpu->pipeline_cache, 1, &pci, NULL,
                                  &pipe->pipeline) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create shadow pipeline\n");
        free(pipe);
        pipe = NULL;
    } else {
        pipe->layout = gpu->shadow_pipeline_layout;
    }

    free(attrs);
    vkDestroyShaderModule(gpu->device, vert_mod, NULL);
    vkDestroyShaderModule(gpu->device, frag_mod, NULL);

    return pipe;
}

void gpu_cmd_bind_shadow(Gpu* gpu)
{
    vkCmdBindDescriptorSets(gpu->current_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            gpu->shadow_pipeline_layout, 0, 1,
                            &gpu->shadow_desc_set, 0, NULL);
}

/* ---- Frame ---- */

int gpu_begin_frame(Gpu* gpu)
{
    uint32_t fi = gpu->frame_index;

    vkWaitForFences(gpu->device, 1, &gpu->in_flight[fi], VK_TRUE, UINT64_MAX);

    if (gpu->headless) {
        /* Headless: no acquire/present round-trip. Cycle the target image with
         * the in-flight frame index so the command buffer
         * (command_buffers[current_image]), its framebuffer, and the fence
         * (in_flight[fi]) all rotate together. Previously hidden-window headless
         * (surfaceless==0) pinned current_image to 0 while frame_index still
         * rotated the fence — so frame N+1 reset/re-recorded command_buffers[0]
         * while frame N was still in flight under a *different* fence
         * (in_flight[fi] for the new frame was already signaled, so the wait was
         * a no-op). On a light scene frame N finished in time and it happened to
         * work; a heavy scene (Isaac warehouse) was still executing, so the
         * reused command buffer was corrupted and the whole frame came back
         * black. Rotating current_image with fi keeps each command buffer paired
         * with the fence that guards it. fi < MAX_FRAMES_IN_FLIGHT <= image_count
         * so this stays in range for both surfaceless and hidden-window modes. */
        gpu->current_image = gpu->image_count
            ? (fi % gpu->image_count) : 0;
    } else {
        VkResult result = vkAcquireNextImageKHR(
            gpu->device, gpu->swapchain, UINT64_MAX,
            gpu->image_available[fi], VK_NULL_HANDLE, &gpu->current_image);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            return 0;
        }
    }

    vkResetFences(gpu->device, 1, &gpu->in_flight[fi]);

    VkCommandBuffer cmd = gpu->command_buffers[gpu->current_image];
    gpu->current_cmd = cmd;

    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &begin_info);

    /* Set viewport and scissor */
    VkViewport viewport = {0};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = (float)gpu->swapchain_extent.width;
    viewport.height   = (float)gpu->swapchain_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {0};
    scissor.extent = gpu->swapchain_extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    /* Begin render pass */
    VkClearValue clear_values[2];
    /* Authored direct-light/no-IBL scenes should clear to black when default
     * lighting is disabled, matching OVRTX. Keep the neutral viewer backdrop
     * for genuinely no-light scenes. */
    float clear_rgb =
        (gpu->light_count > 0 && gpu->env_image_view == VK_NULL_HANDLE)
        ? 0.0f : 0.39f;
    clear_values[0].color.float32[0] = clear_rgb;
    clear_values[0].color.float32[1] = clear_rgb;
    clear_values[0].color.float32[2] = clear_rgb;
    clear_values[0].color.float32[3] = 1.0f;
    clear_values[1].depthStencil.depth   = 1.0f;
    clear_values[1].depthStencil.stencil = 0;

    VkRenderPassBeginInfo rpbi = {0};
    rpbi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass        = gpu->render_pass;
    rpbi.framebuffer       = gpu->framebuffers[gpu->current_image];
    rpbi.renderArea.extent = gpu->swapchain_extent;
    rpbi.clearValueCount   = 2;
    rpbi.pClearValues      = clear_values;

    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    return 1;
}

void gpu_end_frame(Gpu* gpu)
{
    uint32_t fi = gpu->frame_index;
    VkCommandBuffer cmd = gpu->current_cmd;

    vkCmdEndRenderPass(cmd);

    /* Overlay text (after main render pass, before submit) */
    gpu_overlay_flush(gpu);

    VkResult end_res = vkEndCommandBuffer(cmd);
    if (end_res != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: vkEndCommandBuffer failed in gpu_end_frame (%d)\n", end_res);
    }

    /* Submit */
    VkSubmitInfo submit = {0};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;

    if (gpu->headless) {
        /* Headless: no semaphores (no acquire/present) — fence only */
        VkResult submit_res = vkQueueSubmit(gpu->graphics_queue, 1, &submit, gpu->in_flight[fi]);
        if (submit_res != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: vkQueueSubmit failed in gpu_end_frame (%d)\n", submit_res);
        }
    } else {
        VkSemaphore wait_sems[]   = { gpu->image_available[fi] };
        VkSemaphore signal_sems[] = { gpu->render_finished[fi] };
        VkPipelineStageFlags wait_stages[] = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        };
        submit.waitSemaphoreCount   = 1;
        submit.pWaitSemaphores      = wait_sems;
        submit.pWaitDstStageMask    = wait_stages;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores    = signal_sems;

        VkResult submit_res = vkQueueSubmit(gpu->graphics_queue, 1, &submit, gpu->in_flight[fi]);
        if (submit_res != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: vkQueueSubmit failed in gpu_end_frame (%d)\n", submit_res);
        }

        VkPresentInfoKHR present = {0};
        present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores    = signal_sems;
        present.swapchainCount     = 1;
        present.pSwapchains        = &gpu->swapchain;
        present.pImageIndices      = &gpu->current_image;

        VkResult result = vkQueuePresentKHR(gpu->graphics_queue, &present);
        (void)result;
    }

    gpu->frame_index = (gpu->frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
}

/* ---- Draw commands ---- */

void gpu_cmd_bind_pipeline(Gpu* gpu, GpuPipeline pipe)
{
    vkCmdBindPipeline(gpu->current_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      pipe->pipeline);
    gpu->current_layout = pipe->layout;
}

void gpu_cmd_bind_vertex_buffer(Gpu* gpu, GpuBuffer buf)
{
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(gpu->current_cmd, 0, 1, &buf->buffer, &offset);
}

void gpu_cmd_bind_instance_buffer(Gpu* gpu, GpuBuffer buf)
{
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(gpu->current_cmd, 1, 1, &buf->buffer, &offset);
}

void gpu_cmd_bind_index_buffer(Gpu* gpu, GpuBuffer buf)
{
    gpu_cmd_bind_index_buffer_typed(gpu, buf, 0, 32);
}

void gpu_cmd_bind_index_buffer_typed(Gpu* gpu, GpuBuffer buf,
                                     uint64_t offset_bytes,
                                     int index_type_bits)
{
    VkIndexType type = (index_type_bits == 16)
        ? VK_INDEX_TYPE_UINT16
        : VK_INDEX_TYPE_UINT32;
    vkCmdBindIndexBuffer(gpu->current_cmd, buf->buffer,
                         (VkDeviceSize)offset_bytes, type);
}

void gpu_cmd_push_constants(Gpu* gpu, const void* data, uint32_t size)
{
    VkPipelineLayout layout = gpu->current_layout ? gpu->current_layout : gpu->pipeline_layout;
    VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT |
                                VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                                VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
                                VK_SHADER_STAGE_FRAGMENT_BIT;
    if (layout == gpu->mat_pipeline_layout) {
        stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    } else if (layout == gpu->shadow_pipeline_layout) {
        stages = VK_SHADER_STAGE_VERTEX_BIT;
    }
    vkCmdPushConstants(gpu->current_cmd, layout,
                       stages, 0, size, data);
}

void gpu_cmd_draw(Gpu* gpu, uint32_t vertex_count, uint32_t first_vertex)
{
    vkCmdDraw(gpu->current_cmd, vertex_count, 1, first_vertex, 0);
}

void gpu_cmd_draw_indexed(Gpu* gpu, uint32_t index_count,
                          uint32_t first_index, int32_t vertex_offset)
{
    vkCmdDrawIndexed(gpu->current_cmd, index_count, 1, first_index,
                     vertex_offset, 0);
}

void gpu_cmd_draw_indexed_instanced(Gpu* gpu, uint32_t index_count,
                                    uint32_t instance_count,
                                    uint32_t first_index, int32_t vertex_offset,
                                    uint32_t first_instance)
{
    vkCmdDrawIndexed(gpu->current_cmd, index_count, instance_count,
                     first_index, vertex_offset, first_instance);
}

/* ==== Screenshot (readback swapchain to PPM) ==== */

int gpu_screenshot(Gpu* gpu, const char* path)
{
    if (!gpu) return 0;

    uint32_t w = gpu->swapchain_extent.width;
    uint32_t h = gpu->swapchain_extent.height;

    /* Wait for device idle so the last frame is fully rendered */
    vkDeviceWaitIdle(gpu->device);

    /* Create a host-visible buffer for readback */
    VkDeviceSize buf_size = (VkDeviceSize)w * h * 4;
    VkBuffer staging_buf;
    VkDeviceMemory staging_mem;

    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = buf_size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    if (vkCreateBuffer(gpu->device, &bci, NULL, &staging_buf) != VK_SUCCESS)
        return 0;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gpu->device, staging_buf, &req);

    VkMemoryAllocateInfo mai = {0};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(gpu->device, &mai, NULL, &staging_mem) != VK_SUCCESS) {
        vkDestroyBuffer(gpu->device, staging_buf, NULL);
        return 0;
    }
    vkBindBufferMemory(gpu->device, staging_buf, staging_mem, 0);

    /* Record a one-shot command buffer to copy swapchain image to staging buffer */
    VkCommandBufferAllocateInfo cai = {0};
    cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool        = gpu->command_pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(gpu->device, &cai, &cmd);

    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkImageLayout color_ready_layout = gpu->surfaceless
        ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    /* Transition color target to TRANSFER_SRC. */
    VkImageMemoryBarrier to_src = {0};
    to_src.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_src.srcAccessMask                   = VK_ACCESS_MEMORY_READ_BIT;
    to_src.dstAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;
    to_src.oldLayout                       = color_ready_layout;
    to_src.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_src.image                           = gpu->swapchain_images[gpu->current_image];
    to_src.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    to_src.subresourceRange.levelCount     = 1;
    to_src.subresourceRange.layerCount     = 1;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 0, NULL, 1, &to_src);

    /* Copy image to buffer */
    VkBufferImageCopy region = {0};
    region.bufferRowLength   = w;
    region.bufferImageHeight = h;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width  = w;
    region.imageExtent.height = h;
    region.imageExtent.depth  = 1;

    vkCmdCopyImageToBuffer(cmd,
        gpu->swapchain_images[gpu->current_image], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        staging_buf, 1, &region);

    /* Transition back: TRANSFER_SRC → PRESENT_SRC */
    VkImageMemoryBarrier to_present = to_src;
    to_present.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_present.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    to_present.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_present.newLayout     = color_ready_layout;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 0, NULL, 0, NULL, 1, &to_present);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit = {0};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;
    vkQueueSubmit(gpu->graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(gpu->graphics_queue);

    vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &cmd);

    /* Map and write PPM */
    void* mapped;
    vkMapMemory(gpu->device, staging_mem, 0, buf_size, 0, &mapped);

    FILE* f = fopen(path, "wb");
    if (f) {
        fprintf(f, "P6\n%u %u\n255\n", w, h);
        const uint8_t* pixels = (const uint8_t*)mapped;
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                /* Swapchain is B8G8R8A8, need to write R,G,B */
                uint32_t idx = (y * w + x) * 4;
                uint8_t rgb[3] = { pixels[idx + 2], pixels[idx + 1], pixels[idx + 0] };
                fwrite(rgb, 1, 3, f);
            }
        }
        fclose(f);
        fprintf(stderr, "gpu_vulkan: screenshot saved to %s (%ux%u)\n", path, w, h);
    }

    vkUnmapMemory(gpu->device, staging_mem);
    vkDestroyBuffer(gpu->device, staging_buf, NULL);
    vkFreeMemory(gpu->device, staging_mem, NULL);

    /* Phase A G-buffer debug dump — gated on NU_PHASE_A_GBUFFER_DUMP=1.
     * Copies the device-local G-buffer to the host-visible staging buffer
     * and prints stats: hit-rate, distinct instances, hit_t range, plus the
     * first 4 hit pixels. Proves rchit + rmiss writes actually happened
     * (visual byte-identity proves only that the existing color path is
     * untouched). */
    const char* dump_env = getenv("NU_PHASE_A_GBUFFER_DUMP");
    if (dump_env && dump_env[0] >= '1' && gpu->gbuffer_buf && gpu->gbuffer_staging_buf) {
        VkCommandBufferAllocateInfo cai2 = {0};
        cai2.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai2.commandPool        = gpu->command_pool;
        cai2.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai2.commandBufferCount = 1;
        VkCommandBuffer cmd2;
        vkAllocateCommandBuffers(gpu->device, &cai2, &cmd2);

        VkCommandBufferBeginInfo begin2 = {0};
        begin2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin2.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd2, &begin2);
        VkBufferCopy copy = {0, 0, gpu->gbuffer_size};
        vkCmdCopyBuffer(cmd2, gpu->gbuffer_buf, gpu->gbuffer_staging_buf, 1, &copy);
        vkEndCommandBuffer(cmd2);

        VkSubmitInfo submit2 = {0};
        submit2.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit2.commandBufferCount = 1;
        submit2.pCommandBuffers    = &cmd2;
        vkQueueSubmit(gpu->graphics_queue, 1, &submit2, VK_NULL_HANDLE);
        vkQueueWaitIdle(gpu->graphics_queue);
        vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &cmd2);

        if (gpu->gbuffer_staging_mapped) {
            uint32_t total_pix = gpu->gbuffer_w * gpu->gbuffer_h;
            const uint32_t* p = (const uint32_t*)gpu->gbuffer_staging_mapped;
            uint32_t hits = 0;
            uint32_t distinct[1024] = {0};
            uint32_t distinct_cnt = 0;
            float min_t = 1e30f, max_t = -1e30f;
            uint32_t first_hits_logged = 0;
            for (uint32_t i = 0; i < total_pix; i++) {
                const uint32_t* e = p + i * 8;  /* 32 B / 4 = 8 dwords */
                uint32_t inst_and_flags = e[0];
                if (inst_and_flags & 0x80000000u) {
                    hits++;
                    uint32_t inst = inst_and_flags & 0x00FFFFFFu;
                    /* tiny set tracker (linear, 1024 cap) */
                    int found = 0;
                    for (uint32_t j = 0; j < distinct_cnt && j < 1024; j++) {
                        if (distinct[j] == inst) { found = 1; break; }
                    }
                    if (!found && distinct_cnt < 1024) distinct[distinct_cnt++] = inst;

                    float ht = *(const float*)&e[3];
                    if (ht < min_t) min_t = ht;
                    if (ht > max_t) max_t = ht;

                    if (first_hits_logged < 4) {
                        float nx = *(const float*)&e[4];
                        float ny = *(const float*)&e[5];
                        float nz = *(const float*)&e[6];
                        fprintf(stderr,
                                "gbuf[pix=%u]: hit=1 inst=%u prim=%u bary=0x%08x t=%.3f n=(%.3f,%.3f,%.3f)\n",
                                i, inst, e[1], e[2], ht, nx, ny, nz);
                        first_hits_logged++;
                    }
                }
            }
            fprintf(stderr,
                    "gbuf stats: %u/%u pixels hit (%.1f%%), distinct_inst=%u (cap 1024), hit_t=[%.3f, %.3f]\n",
                    hits, total_pix, 100.0 * (double)hits / (double)total_pix,
                    distinct_cnt, (double)min_t, (double)max_t);

            /* Optional: dump a tiny normal-map PPM. NU_PHASE_A_GBUFFER_DUMP=2
             * (or anything > '1') triggers it. */
            if (dump_env[0] >= '2') {
                char npath[1024];
                snprintf(npath, sizeof(npath), "%s.gbnormal.ppm", path);
                FILE* nf = fopen(npath, "wb");
                if (nf) {
                    fprintf(nf, "P6\n%u %u\n255\n", gpu->gbuffer_w, gpu->gbuffer_h);
                    for (uint32_t i = 0; i < total_pix; i++) {
                        const uint32_t* e = p + i * 8;
                        uint32_t inst_and_flags = e[0];
                        uint8_t rgb[3] = {0, 0, 0};
                        if (inst_and_flags & 0x80000000u) {
                            float nx = *(const float*)&e[4];
                            float ny = *(const float*)&e[5];
                            float nz = *(const float*)&e[6];
                            rgb[0] = (uint8_t)(127.5f * (nx + 1.0f));
                            rgb[1] = (uint8_t)(127.5f * (ny + 1.0f));
                            rgb[2] = (uint8_t)(127.5f * (nz + 1.0f));
                        }
                        fwrite(rgb, 1, 3, nf);
                    }
                    fclose(nf);
                    fprintf(stderr, "gbuf normal-map PPM saved to %s\n", npath);
                }
            }
        }
    }

    return f != NULL;
}

int gpu_readback_pixels(Gpu* gpu, uint8_t* out_pixels, uint32_t width, uint32_t height,
                        int swizzle_to_rgba)
{
    if (!gpu || !out_pixels) return 0;

    uint32_t w = gpu->swapchain_extent.width;
    uint32_t h = gpu->swapchain_extent.height;
    if (width != w || height != h) return 0;

    /* perf/no-waitidle: wait only on THIS frame's render fence instead of
     * the whole device. gpu_end_frame_rt signals in_flight[fi] then bumps
     * frame_index, so the just-submitted fence lives at the previous slot. */
    uint32_t prev_fi = (gpu->frame_index + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT;
    vkWaitForFences(gpu->device, 1, &gpu->in_flight[prev_fi], VK_TRUE, UINT64_MAX);

    /* perf/vk-instrumentation: the per-frame fence wait above guarantees that
     * the most recent gpu_cmd_trace_rays' timestamps have landed in the pool.
     * Resolve RT_dispatch here so the host has fresh per-frame numbers. */
    gpu_phase_resolve(gpu, GPU_PHASE_RT_DISPATCH);

    /* Lazily allocate (or reallocate on resize) a persistent staging buffer */
    VkDeviceSize buf_size = (VkDeviceSize)w * h * 4;
    if (!gpu->readback_buf || gpu->readback_w != w || gpu->readback_h != h) {
        /* Free old buffer if resized */
        if (gpu->readback_buf) {
            if (gpu->readback_mapped)
                vkUnmapMemory(gpu->device, gpu->readback_mem);
            vkDestroyBuffer(gpu->device, gpu->readback_buf, NULL);
            vkFreeMemory(gpu->device, gpu->readback_mem, NULL);
            gpu->readback_buf = VK_NULL_HANDLE;
            gpu->readback_mapped = NULL;
        }

        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = buf_size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->readback_buf) != VK_SUCCESS)
            return 0;

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->readback_buf, &req);

        /* perf/host-cached: prefer HOST_CACHED memory for ~10x faster CPU reads.
         * Discrete GPUs commonly expose HOST_VISIBLE | HOST_COHERENT as
         * write-combined memory (~140 MB/s reads), which dominates steady-state
         * for full-frame readback. find_readback_memory_type tries
         * CACHED+COHERENT first, then CACHED-only, falling back to COHERENT. */
        int is_coherent = 0;
        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_readback_memory_type(gpu->physical_device,
                                                        req.memoryTypeBits,
                                                        &is_coherent);

        if (vkAllocateMemory(gpu->device, &mai, NULL, &gpu->readback_mem) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, gpu->readback_buf, NULL);
            gpu->readback_buf = VK_NULL_HANDLE;
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->readback_buf, gpu->readback_mem, 0);

        /* Persistent map — kept until shutdown or resize */
        vkMapMemory(gpu->device, gpu->readback_mem, 0, buf_size, 0, &gpu->readback_mapped);

        gpu->readback_w = w;
        gpu->readback_h = h;
        gpu->readback_coherent = is_coherent;
    }

    /* One-shot command buffer to copy swapchain image to staging buffer */
    VkCommandBufferAllocateInfo cai = {0};
    cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool        = gpu->command_pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(gpu->device, &cai, &cmd);

    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkImageLayout color_ready_layout = gpu->surfaceless
        ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    /* Transition color target to TRANSFER_SRC. */
    VkImageMemoryBarrier to_src = {0};
    to_src.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_src.srcAccessMask                   = VK_ACCESS_MEMORY_READ_BIT;
    to_src.dstAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;
    to_src.oldLayout                       = color_ready_layout;
    to_src.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_src.image                           = gpu->swapchain_images[gpu->current_image];
    to_src.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    to_src.subresourceRange.levelCount     = 1;
    to_src.subresourceRange.layerCount     = 1;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 0, NULL, 1, &to_src);

    VkBufferImageCopy region = {0};
    region.bufferRowLength   = w;
    region.bufferImageHeight = h;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width  = w;
    region.imageExtent.height = h;
    region.imageExtent.depth  = 1;

    /* perf/vk-instrumentation: bracket the image->staging copy. */
    gpu_phase_begin(gpu, cmd, GPU_PHASE_PIXEL_READBACK, "pixel_readback");
    vkCmdCopyImageToBuffer(cmd,
        gpu->swapchain_images[gpu->current_image], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        gpu->readback_buf, 1, &region);
    gpu_phase_end(gpu, cmd, GPU_PHASE_PIXEL_READBACK);

    /* Transition back: TRANSFER_SRC → PRESENT_SRC */
    VkImageMemoryBarrier to_present = to_src;
    to_present.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_present.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    to_present.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_present.newLayout     = color_ready_layout;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 0, NULL, 0, NULL, 1, &to_present);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit = {0};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;
    vkQueueSubmit(gpu->graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(gpu->graphics_queue);

    /* perf/vk-instrumentation: pixel_readback timestamps are now landed. */
    gpu_phase_resolve(gpu, GPU_PHASE_PIXEL_READBACK);

    vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &cmd);

    /* perf/host-cached: if staging is HOST_CACHED but not HOST_COHERENT, the
     * CPU's view of the staging buffer may still be stale. Invalidate the
     * mapped range so the cache picks up the GPU's writes. */
    if (!gpu->readback_coherent) {
        VkMappedMemoryRange range = {0};
        range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = gpu->readback_mem;
        range.offset = 0;
        range.size   = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(gpu->device, 1, &range);
    }

    if (swizzle_to_rgba) {
        /* Swizzle BGRA → RGBA into caller's buffer (32-bit integer ops for speed).
         * On HOST_CACHED staging this is bandwidth-bound and small (~0.07 ms at
         * 1280x720); on HOST_COHERENT (write-combined) it dominates the readback. */
        const uint32_t* src32 = (const uint32_t*)gpu->readback_mapped;
        uint32_t* dst32 = (uint32_t*)out_pixels;
        uint32_t npixels = w * h;
        for (uint32_t i = 0; i < npixels; i++) {
            uint32_t bgra = src32[i];
            /* BGRA: B=byte0, G=byte1, R=byte2, A=byte3 (little-endian) */
            /* RGBA: R=byte0, G=byte1, B=byte2, A=byte3 */
            uint32_t r = (bgra >> 16) & 0xFF;   /* R from byte2 */
            uint32_t g = (bgra >>  8) & 0xFF;   /* G from byte1 */
            uint32_t b = (bgra      ) & 0xFF;   /* B from byte0 */
            uint32_t a = (bgra >> 24) & 0xFF;   /* A from byte3 */
            dst32[i] = r | (g << 8) | (b << 16) | (a << 24);
        }
    } else {
        /* perf/no-swizzle: caller asked for raw BGRA8 — single bulk copy
         * straight out of the device-cached staging buffer. No per-pixel work. */
        memcpy(out_pixels, gpu->readback_mapped, (size_t)w * h * 4);
    }

    return 1;
}

/* CUDA-Vulkan interop variant of gpu_readback_pixels. Copies the swapchain
 * image into a device-local exportable buffer that CUDA can import via
 * cuImportExternalMemory. Returns 1 on success, 0 on failure.
 *
 * On success, the buffer's contents are BGRA8 (raw swapchain format — no
 * BGRA→RGBA swizzle, since that's a CPU op which defeats the point of this
 * path). The fd is cached on the Gpu and closed by gpu_destroy. */
int gpu_readback_pixels_cuda(Gpu* gpu, uint32_t width, uint32_t height,
                              int* out_fd, uint64_t* out_size,
                              uint64_t* out_logical_size)
{
    if (!gpu || !gpu->interop_available) return 0;

    uint32_t w = gpu->swapchain_extent.width;
    uint32_t h = gpu->swapchain_extent.height;
    if (width != w || height != h) return 0;

    VkDeviceSize logical_size = (VkDeviceSize)w * h * 4;

    /* Lazily allocate (or reallocate on resize) an exportable device-local
     * buffer. The associated fd is exported lazily on first call. */
    if (!gpu->pixels_interop_buf || gpu->pixels_interop_w != w ||
        gpu->pixels_interop_h != h) {
        /* Free old (and any cached fd) on resize */
        if (gpu->pixels_interop_fd >= 0) {
            close(gpu->pixels_interop_fd);
            gpu->pixels_interop_fd = -1;
        }
        if (gpu->pixels_interop_buf) {
            vkDestroyBuffer(gpu->device, gpu->pixels_interop_buf, NULL);
            gpu->pixels_interop_buf = VK_NULL_HANDLE;
        }
        if (gpu->pixels_interop_mem) {
            vkFreeMemory(gpu->device, gpu->pixels_interop_mem, NULL);
            gpu->pixels_interop_mem = VK_NULL_HANDLE;
        }

        VkExternalMemoryBufferCreateInfo ext_buf_ci = {0};
        ext_buf_ci.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ext_buf_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.pNext = &ext_buf_ci;
        bci.size  = logical_size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        if (vkCreateBuffer(gpu->device, &bci, NULL,
                           &gpu->pixels_interop_buf) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: pixels_interop vkCreateBuffer failed\n");
            return 0;
        }

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->pixels_interop_buf, &req);

        VkExportMemoryAllocateInfo export_mai = {0};
        export_mai.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        export_mai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext           = &export_mai;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (gpu_alloc_memory(gpu, &mai, &gpu->pixels_interop_mem) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: pixels_interop vkAllocateMemory failed\n");
            vkDestroyBuffer(gpu->device, gpu->pixels_interop_buf, NULL);
            gpu->pixels_interop_buf = VK_NULL_HANDLE;
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->pixels_interop_buf,
                           gpu->pixels_interop_mem, 0);

        gpu->pixels_interop_alloc_size   = req.size;
        gpu->pixels_interop_logical_size = logical_size;
        gpu->pixels_interop_w = w;
        gpu->pixels_interop_h = h;

        fprintf(stderr,
                "gpu_vulkan: pixels_interop buffer created (%ux%u, %.1f MB alloc, %.1f MB logical)\n",
                w, h, (double)req.size / (1024.0 * 1024.0),
                (double)logical_size / (1024.0 * 1024.0));
    }

    /* One-shot command buffer to copy swapchain image → exportable buffer */
    VkCommandBufferAllocateInfo cai = {0};
    cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool        = gpu->command_pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(gpu->device, &cai, &cmd);

    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkImageLayout color_ready_layout = gpu->surfaceless
        ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    /* Transition color target to TRANSFER_SRC. */
    VkImageMemoryBarrier to_src = {0};
    to_src.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_src.srcAccessMask                   = VK_ACCESS_MEMORY_READ_BIT;
    to_src.dstAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;
    to_src.oldLayout                       = color_ready_layout;
    to_src.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_src.image                           = gpu->swapchain_images[gpu->current_image];
    to_src.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    to_src.subresourceRange.levelCount     = 1;
    to_src.subresourceRange.layerCount     = 1;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 0, NULL, 1, &to_src);

    VkBufferImageCopy region = {0};
    region.bufferRowLength   = w;
    region.bufferImageHeight = h;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width  = w;
    region.imageExtent.height = h;
    region.imageExtent.depth  = 1;

    vkCmdCopyImageToBuffer(cmd,
        gpu->swapchain_images[gpu->current_image], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        gpu->pixels_interop_buf, 1, &region);

    /* Transition back: TRANSFER_SRC → PRESENT_SRC */
    VkImageMemoryBarrier to_present = to_src;
    to_present.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    to_present.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    to_present.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    to_present.newLayout     = color_ready_layout;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 0, NULL, 0, NULL, 1, &to_present);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit = {0};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;
    vkQueueSubmit(gpu->graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(gpu->graphics_queue);

    vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &cmd);

    /* Export the fd lazily; cache it for re-use across frames so we don't
     * pay the kernel cost of vkGetMemoryFdKHR + cuImportExternalMemory on
     * every frame. The fd is closed in gpu_destroy. */
    if (gpu->pixels_interop_fd < 0) {
        VkMemoryGetFdInfoKHR fd_info = {0};
        fd_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fd_info.memory     = gpu->pixels_interop_mem;
        fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        int fd = -1;
        if (vkGetMemoryFdKHR(gpu->device, &fd_info, &fd) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: vkGetMemoryFdKHR(pixels_interop) failed\n");
            return 0;
        }
        gpu->pixels_interop_fd = fd;
        fprintf(stderr,
                "gpu_vulkan: pixels_interop fd exported (%d) for one-time CUDA import\n",
                fd);
    }

    if (out_fd)            *out_fd            = gpu->pixels_interop_fd;
    if (out_size)          *out_size          = (uint64_t)gpu->pixels_interop_alloc_size;
    if (out_logical_size)  *out_logical_size  = (uint64_t)gpu->pixels_interop_logical_size;
    return 1;
}

/* ==== Ray Tracing Implementation ==== */

int gpu_rt_available(Gpu* gpu) { return gpu ? gpu->rt_available : 0; }
int gpu_rt_built(Gpu* gpu) { return gpu ? gpu->rt_built : 0; }
int gpu_ser_available(Gpu* gpu) { return gpu ? gpu->ser_available : 0; }
int gpu_barycentric_available(Gpu* gpu) { return gpu ? gpu->barycentric_available : 0; }

/* perf/mem-pool: RT_POOL_* enum + rt_ensure_pool forward declaration
 * are at the top of the file (just before gpu_init) so gpu_init can
 * eagerly create the pools at construction time, outside the timed
 * build_accel path.
 *
 * RESIDENT keeps the buffer alive until gpu_destroy_rt_scene;
 * TRANSIENT lives only inside one build_accel call (resets cursor at
 * boundary); STAGING is a HOST_VISIBLE upload buffer (resets at
 * boundary too); NONE forces a private vkAllocateMemory (used for
 * exportable buffers that need VkExportMemoryAllocateInfo). */

/* Pool-aware buffer create. Tries the requested pool first; falls back
 * to a private vkAllocateMemory if the pool is exhausted or rejects
 * the memoryTypeBits. Always sets VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
 * (matches the legacy rt_create_buffer behaviour). The resident /
 * transient pools are created with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
 * so vkGetBufferDeviceAddress works.
 *
 * Returns 1 on success.
 *
 * On pool success: *out_mem == VK_NULL_HANDLE (vkFreeMemory(NULL) is a
 * no-op, so existing destroy paths stay correct).
 * On fallback success: *out_mem holds a private VkDeviceMemory the
 * caller must vkFreeMemory at destroy time. */
static int rt_create_buffer_pooled(Gpu* gpu, VkDeviceSize size,
                                    VkBufferUsageFlags usage,
                                    VkMemoryPropertyFlags mem_props,
                                    int pool_tag,
                                    VkBuffer* out_buf, VkDeviceMemory* out_mem)
{
    *out_buf = VK_NULL_HANDLE;
    *out_mem = VK_NULL_HANDLE;

    VkBufferUsageFlags full_usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    /* Try pool first. */
    if (pool_tag != RT_POOL_NONE) {
        GpuMemPool* pool = rt_ensure_pool(gpu, pool_tag);
        if (pool) {
            VkDeviceSize bytes = 0;
            if (gpu_pool_alloc_buffer(pool, size, full_usage,
                                       out_buf, out_mem, &bytes)) {
                /* perf/mem-pool: track sub-alloc bytes the SAME WAY the
                 * pre-pool code did:
                 *   - Pre-pool, every rt_create_buffer call used
                 *     gpu_alloc_memory which incremented
                 *     total_allocated_gpu_mem by ai->allocationSize.
                 *   - The seg/color/AABB sites used direct vkAllocateMemory
                 *     (intentionally untracked — see comment in
                 *     gpu_upload_curve_data). These now go through the
                 *     pool path with a different helper that we'll keep
                 *     untracked too: see rt_create_buffer_pooled_untracked
                 *     below.
                 *
                 * Net result: gpu_mem_mb stays roughly equal to the v2
                 * baseline (3734 MB on tera). The pool's own
                 * VkDeviceMemory allocs are NOT counted (gpu_pool_create
                 * uses raw vkAllocateMemory by design — they're VA
                 * reservations and physical pages are committed lazily
                 * as buffers are bound, so the cumulative bytes added
                 * here approximate the committed working set). */
                gpu->total_allocated_gpu_mem += bytes;
                return 1;
            }
            /* fall through to private alloc on pool failure */
        }
    }

    /* Private allocation fallback (matches old rt_create_buffer behavior). */
    VkBufferCreateInfo bci = {0};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = full_usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(gpu->device, &bci, NULL, out_buf) != VK_SUCCESS)
        return 0;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gpu->device, *out_buf, &req);

    VkMemoryAllocateFlagsInfo flags_info = {0};
    flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo ai = {0};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext           = &flags_info;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits, mem_props);

    if (gpu_alloc_memory(gpu, &ai, out_mem) != VK_SUCCESS) {
        vkDestroyBuffer(gpu->device, *out_buf, NULL);
        *out_buf = VK_NULL_HANDLE;
        return 0;
    }
    vkBindBufferMemory(gpu->device, *out_buf, *out_mem, 0);
    return 1;
}

/* Legacy entry point (call sites that haven't been migrated). */
static int rt_create_buffer(Gpu* gpu, VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags mem_props,
                            VkBuffer* out_buf, VkDeviceMemory* out_mem)
{
    return rt_create_buffer_pooled(gpu, size, usage, mem_props,
                                   RT_POOL_NONE, out_buf, out_mem);
}

/* Pool-aware buffer create that does NOT add the bound size to
 * total_allocated_gpu_mem. Used for the curve seg/color/AABB SSBOs
 * (which were direct vkAllocateMemory in v2 — explicitly untracked
 * per the gpu_upload_curve_data comment) so the reported gpu_mem_mb
 * stays comparable to the pre-pool baseline. */
static int rt_create_buffer_pooled_untracked(Gpu* gpu, VkDeviceSize size,
                                              VkBufferUsageFlags usage,
                                              VkMemoryPropertyFlags mem_props,
                                              int pool_tag,
                                              VkBuffer* out_buf,
                                              VkDeviceMemory* out_mem)
{
    /* Snapshot + restore the tracked counter around the pooled call so we
     * subtract whatever the tracked path added (and the fallback path,
     * via gpu_alloc_memory, also adds — but those bytes would have been
     * untracked in the pre-pool world too). */
    uint64_t before = gpu->total_allocated_gpu_mem;
    int ok = rt_create_buffer_pooled(gpu, size, usage, mem_props, pool_tag,
                                      out_buf, out_mem);
    gpu->total_allocated_gpu_mem = before;
    return ok;
}

/* Lazily allocate the requested pool. Sizes are tuned to comfortably
 * cover the tera fixture (12.3 M curves, ~12 M segments — produces
 * a 2.5 GB BLAS scratch, the largest single allocation) while staying
 * well under the 24 GB device-local heap on RTX 5090.
 *
 * Pool sizes (chosen with empirical headroom from tera bench output):
 *   resident   : 4 GB  — segs (1.1 GB) + colors (50 MB) + AABBs (790 MB)
 *                + curve BLAS storage (~1.2 GB) + scene_data + TLAS
 *                + persistent TLAS update scratch + instance_buf.
 *                tera total: ~3.2 GB. 4 GB has small but adequate room.
 *   transient  : 4 GB  — curve BLAS scratch (2.5 GB on tera) is the
 *                dominant user; TLAS scratch + mesh BLAS scratch are
 *                much smaller. Reset between builds.
 *   staging    : 2 GB  — staging upload max(seg, color) = 1.1 GB on tera
 *                + small instance/SSBO uploads. Reset between builds.
 *
 * NOTE: pools are virtual address space — physical pages are committed
 * lazily by the kernel/driver as buffers are bound. Total VA budget
 * for the three pools is 10 GB; on tera the actual committed footprint
 * ends up around 4-5 GB, matching the v2 baseline gpu_mem_mb.
 */
/* Read a "pool size in GB" env var. Falls back to default_gb. Caps at 16 GB
 * to avoid accidentally requesting more than the heap. */
static VkDeviceSize rt_pool_size_env(const char* env_name, double default_gb)
{
    const char* e = getenv(env_name);
    double gb = default_gb;
    if (e && e[0]) {
        char* endp = NULL;
        double parsed = strtod(e, &endp);
        if (endp != e && parsed > 0.0 && parsed <= 16.0) gb = parsed;
    }
    /* Multiply via VkDeviceSize (uint64_t) to dodge float precision near GB boundaries. */
    return (VkDeviceSize)(gb * 1024.0 * 1024.0 * 1024.0 + 0.5);
}

static GpuMemPool* rt_ensure_pool(Gpu* gpu, int pool_tag)
{
    if (pool_tag == RT_POOL_RESIDENT) {
        if (!gpu->resident_pool) {
            /* DSX-with-dedup needs ~1.4 GB; DSX-without-dedup needs ~9.5 GB.
             * Default 4 GB matches the historical sizing; override via env. */
            VkDeviceSize size = rt_pool_size_env("NUSD_RESIDENT_POOL_GB", 4.0);
            gpu->resident_pool = gpu_pool_create(
                gpu->device, gpu->physical_device, size,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                /*with_device_address=*/1);
        }
        return (GpuMemPool*)gpu->resident_pool;
    }
    if (pool_tag == RT_POOL_TRANSIENT) {
        if (!gpu->transient_pool) {
            /* BLAS scratch (~2.5 GB on curve-heavy fixtures). Resettable
             * after each build; override via env to size for the scene's
             * actual scratch budget. */
            VkDeviceSize size = rt_pool_size_env("NUSD_TRANSIENT_POOL_GB", 4.0);
            gpu->transient_pool = gpu_pool_create(
                gpu->device, gpu->physical_device, size,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                /*with_device_address=*/1);
        }
        return (GpuMemPool*)gpu->transient_pool;
    }
    if (pool_tag == RT_POOL_STAGING) {
        if (!gpu->staging_pool) {
            /* Host-visible upload buffer. Persistently mapped, so its
             * virtual size IS in process RSS — keep it minimal for the
             * upload chunk size you actually use (256 MB chunks). */
            VkDeviceSize size = rt_pool_size_env("NUSD_STAGING_POOL_GB", 2.0);
            gpu->staging_pool = gpu_pool_create(
                gpu->device, gpu->physical_device, size,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                /*with_device_address=*/0);
        }
        return (GpuMemPool*)gpu->staging_pool;
    }
    return NULL;
}

/* Reset transient + staging pools at the boundary of build_accel
 * (mark/reset for buffers that don't need to outlive this call).
 * Caller must have already destroyed any VkBuffer handles bound into
 * these pools. Resident pool is NOT reset — its buffers persist
 * across builds. */
static void rt_pool_reset_transient(Gpu* gpu)
{
    if (gpu->transient_pool) gpu_pool_reset((GpuMemPool*)gpu->transient_pool);
    if (gpu->staging_pool)   gpu_pool_reset((GpuMemPool*)gpu->staging_pool);
}

/* perf/mem-pool: HOST_VISIBLE staging buffer creation, pool-aware.
 *
 * On success returns 1 and sets:
 *   *out_buf      = VkBuffer (caller vkDestroyBuffer)
 *   *out_mem      = VK_NULL_HANDLE on pool path; private VkDeviceMemory on
 *                   fallback (caller vkFreeMemory iff not VK_NULL_HANDLE)
 *   *out_mapped   = host pointer for memcpy (the pool is persistent-mapped
 *                   from creation, so this is just persistent_map + offset).
 *
 * Staging is HOST_VISIBLE | HOST_COHERENT, TRANSFER_SRC only. */
static int rt_create_staging_buffer(Gpu* gpu, VkDeviceSize size,
                                     VkBuffer* out_buf,
                                     VkDeviceMemory* out_mem,
                                     void** out_mapped)
{
    *out_buf    = VK_NULL_HANDLE;
    *out_mem    = VK_NULL_HANDLE;
    *out_mapped = NULL;

    GpuMemPool* pool = rt_ensure_pool(gpu, RT_POOL_STAGING);
    if (pool && pool->persistent_map) {
        VkBufferCreateInfo bci = {0};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = size;
        bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buf = VK_NULL_HANDLE;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &buf) == VK_SUCCESS) {
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(gpu->device, buf, &req);
            VkDeviceSize offset = gpu_pool_suballoc(pool, req.size, req.alignment,
                                                     req.memoryTypeBits);
            if (offset != (VkDeviceSize)-1 &&
                vkBindBufferMemory(gpu->device, buf, pool->memory, offset) == VK_SUCCESS) {
                /* The pool is persistent-mapped at creation; just compute
                 * the sub-region pointer. No vkMapMemory call here. */
                *out_buf    = buf;
                *out_mem    = VK_NULL_HANDLE;
                *out_mapped = (uint8_t*)pool->persistent_map + offset;
                return 1;
            }
            vkDestroyBuffer(gpu->device, buf, NULL);
        }
    }

    /* Fallback: private vkAllocateMemory. */
    VkBufferCreateInfo bci = {0};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(gpu->device, &bci, NULL, out_buf) != VK_SUCCESS)
        return 0;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gpu->device, *out_buf, &req);
    VkMemoryAllocateInfo ai = {0};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(gpu->device, &ai, NULL, out_mem) != VK_SUCCESS) {
        vkDestroyBuffer(gpu->device, *out_buf, NULL);
        *out_buf = VK_NULL_HANDLE;
        return 0;
    }
    vkBindBufferMemory(gpu->device, *out_buf, *out_mem, 0);
    if (vkMapMemory(gpu->device, *out_mem, 0, size, 0, out_mapped) != VK_SUCCESS) {
        vkFreeMemory(gpu->device, *out_mem, NULL);
        vkDestroyBuffer(gpu->device, *out_buf, NULL);
        *out_buf = VK_NULL_HANDLE;
        *out_mem = VK_NULL_HANDLE;
        return 0;
    }
    return 1;
}

/* perf/mem-pool: tear down a staging buffer made by rt_create_staging_buffer.
 * Handles both the pool path (mem == VK_NULL_HANDLE) and the fallback. */
static void rt_destroy_staging_buffer(Gpu* gpu, VkBuffer buf, VkDeviceMemory mem,
                                       void* mapped)
{
    (void)gpu; (void)mapped;
    if (mem) {
        /* Fallback: caller's mapped pointer was set by vkMapMemory on this
         * private allocation — unmap+free as a normal private allocation. */
        vkUnmapMemory(gpu->device, mem);
        vkFreeMemory(gpu->device, mem, NULL);
    }
    /* Pool-path: nothing to unmap (persistent map lives on the pool); just
     * destroy the buffer handle. The pool's cursor is reclaimed by
     * rt_pool_reset_transient(). */
    if (buf) vkDestroyBuffer(gpu->device, buf, NULL);
}

static VkDeviceAddress rt_buf_addr(Gpu* gpu, VkBuffer buf)
{
    VkBufferDeviceAddressInfo info = {0};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buf;
    return vkGetBufferDeviceAddress(gpu->device, &info);
}

/* Build a single-cluster CLAS for one mesh and wrap it in a Cluster BLAS.
 * Inputs are the mesh's existing vertex (R32G32B32_SFLOAT @ vstride) and uint32
 * index device addresses. Requires nverts/ntris <= the device per-cluster caps
 * (256/256 on Blackwell): one cluster covers the whole mesh, so the mesh's own
 * indices double as cluster-local indices. On success returns the Cluster BLAS
 * device address (usable directly as a TLAS accelerationStructureReference) and
 * hands back the two AS backing buffers the caller must keep alive until scene
 * teardown. Returns 0 on failure. Frees scratch + transient input buffers. */
#if NUSD_HAS_VK_NV_MEGA_GEOMETRY
static VkDeviceAddress nu_build_cluster_blas_for_mesh(
        Gpu* gpu, VkDeviceAddress vaddr, VkDeviceAddress iaddr,
        uint32_t nverts, uint32_t ntris, uint32_t vstride,
        VkBuffer keep_buf[2], VkDeviceMemory keep_mem[2])
{
    keep_buf[0]=keep_buf[1]=VK_NULL_HANDLE; keep_mem[0]=keep_mem[1]=VK_NULL_HANDLE;
    PFN_vkGetClusterAccelerationStructureBuildSizesNV pSizes =
        (PFN_vkGetClusterAccelerationStructureBuildSizesNV)
        vkGetDeviceProcAddr(gpu->device, "vkGetClusterAccelerationStructureBuildSizesNV");
    PFN_vkCmdBuildClusterAccelerationStructureIndirectNV pBuild =
        (PFN_vkCmdBuildClusterAccelerationStructureIndirectNV)
        vkGetDeviceProcAddr(gpu->device, "vkCmdBuildClusterAccelerationStructureIndirectNV");
    if (!pSizes || !pBuild || nverts == 0 || ntris == 0) return 0;

    const VkBufferUsageFlags U_IN = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    const VkBufferUsageFlags U_AS = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    const VkBufferUsageFlags U_SCR = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkMemoryPropertyFlags M_HOST = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags M_DEV = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    void* mp = NULL; uint32_t one = 1;

    /* ---- Phase 1: triangle cluster (CLAS) ---- */
    VkClusterAccelerationStructureTriangleClusterInputNV tri = {0};
    tri.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    tri.maxClusterUniqueGeometryCount = 1;
    tri.maxClusterTriangleCount = ntris;
    tri.maxClusterVertexCount = nverts;
    tri.maxTotalTriangleCount = ntris;
    tri.maxTotalVertexCount = nverts;
    VkClusterAccelerationStructureInputInfoNV cin = {0};
    cin.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
    cin.maxAccelerationStructureCount = 1;
    cin.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    cin.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    cin.opInput.pTriangleClusters = &tri;
    VkAccelerationStructureBuildSizesInfoKHR csz = {0};
    csz.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pSizes(gpu->device, &cin, &csz);

    VkBuffer clasDst=0, scr=0, clasAddr=0, srcInfo=0, cnt=0;
    VkDeviceMemory clasDstM=0, scrM=0, clasAddrM=0, srcInfoM=0, cntM=0;
    rt_create_buffer(gpu, csz.accelerationStructureSize, U_AS, M_DEV, &clasDst, &clasDstM);
    rt_create_buffer(gpu, csz.buildScratchSize?csz.buildScratchSize:4, U_SCR, M_DEV, &scr, &scrM);
    rt_create_buffer(gpu, sizeof(VkDeviceAddress), U_SCR, M_HOST, &clasAddr, &clasAddrM);
    rt_create_buffer(gpu, sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV), U_IN, M_HOST, &srcInfo, &srcInfoM);
    rt_create_buffer(gpu, sizeof(uint32_t), U_IN, M_HOST, &cnt, &cntM);

    VkClusterAccelerationStructureBuildTriangleClusterInfoNV ti = {0};
    ti.clusterID = 0; ti.triangleCount = ntris; ti.vertexCount = nverts;
    ti.indexType = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV;
    ti.indexBufferStride = (uint16_t)sizeof(uint32_t);
    ti.vertexBufferStride = (uint16_t)vstride;
    ti.indexBuffer = iaddr; ti.vertexBuffer = vaddr;
    vkMapMemory(gpu->device, srcInfoM,0,sizeof(ti),0,&mp); memcpy(mp,&ti,sizeof(ti)); vkUnmapMemory(gpu->device, srcInfoM);
    vkMapMemory(gpu->device, cntM,0,sizeof(one),0,&mp); memcpy(mp,&one,sizeof(one)); vkUnmapMemory(gpu->device, cntM);

    VkClusterAccelerationStructureCommandsInfoNV ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
    ci.input = cin;
    ci.dstImplicitData = rt_buf_addr(gpu, clasDst);
    ci.scratchData = rt_buf_addr(gpu, scr);
    ci.dstAddressesArray.deviceAddress = rt_buf_addr(gpu, clasAddr);
    ci.dstAddressesArray.stride = sizeof(VkDeviceAddress);
    ci.dstAddressesArray.size = sizeof(VkDeviceAddress);
    ci.srcInfosArray.deviceAddress = rt_buf_addr(gpu, srcInfo);
    ci.srcInfosArray.stride = sizeof(ti); ci.srcInfosArray.size = sizeof(ti);
    ci.srcInfosCount = rt_buf_addr(gpu, cnt);
    VkCommandBuffer cmd = rt_begin_cmd(gpu); pBuild(cmd,&ci); rt_end_cmd(gpu,cmd);

    /* ---- Phase 2: cluster bottom-level (Cluster BLAS) ---- */
    VkClusterAccelerationStructureClustersBottomLevelInputNV bl = {0};
    bl.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
    bl.maxTotalClusterCount = 1; bl.maxClusterCountPerAccelerationStructure = 1;
    VkClusterAccelerationStructureInputInfoNV bin = {0};
    bin.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
    bin.maxAccelerationStructureCount = 1;
    bin.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    bin.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    bin.opInput.pClustersBottomLevel = &bl;
    VkAccelerationStructureBuildSizesInfoKHR bsz = {0};
    bsz.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pSizes(gpu->device, &bin, &bsz);

    VkBuffer blasDst=0, bscr=0, blasAddr=0, blSrc=0, blCnt=0;
    VkDeviceMemory blasDstM=0, bscrM=0, blasAddrM=0, blSrcM=0, blCntM=0;
    rt_create_buffer(gpu, bsz.accelerationStructureSize, U_AS, M_DEV, &blasDst, &blasDstM);
    rt_create_buffer(gpu, bsz.buildScratchSize?bsz.buildScratchSize:4, U_SCR, M_DEV, &bscr, &bscrM);
    rt_create_buffer(gpu, sizeof(VkDeviceAddress), U_SCR, M_HOST, &blasAddr, &blasAddrM);
    rt_create_buffer(gpu, sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV), U_IN, M_HOST, &blSrc, &blSrcM);
    rt_create_buffer(gpu, sizeof(uint32_t), U_IN, M_HOST, &blCnt, &blCntM);

    VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV bi = {0};
    bi.clusterReferencesCount = 1; bi.clusterReferencesStride = sizeof(VkDeviceAddress);
    bi.clusterReferences = rt_buf_addr(gpu, clasAddr);
    vkMapMemory(gpu->device, blSrcM,0,sizeof(bi),0,&mp); memcpy(mp,&bi,sizeof(bi)); vkUnmapMemory(gpu->device, blSrcM);
    vkMapMemory(gpu->device, blCntM,0,sizeof(one),0,&mp); memcpy(mp,&one,sizeof(one)); vkUnmapMemory(gpu->device, blCntM);

    VkClusterAccelerationStructureCommandsInfoNV bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
    bci.input = bin;
    bci.dstImplicitData = rt_buf_addr(gpu, blasDst);
    bci.scratchData = rt_buf_addr(gpu, bscr);
    bci.dstAddressesArray.deviceAddress = rt_buf_addr(gpu, blasAddr);
    bci.dstAddressesArray.stride = sizeof(VkDeviceAddress);
    bci.dstAddressesArray.size = sizeof(VkDeviceAddress);
    bci.srcInfosArray.deviceAddress = rt_buf_addr(gpu, blSrc);
    bci.srcInfosArray.stride = sizeof(bi); bci.srcInfosArray.size = sizeof(bi);
    bci.srcInfosCount = rt_buf_addr(gpu, blCnt);
    cmd = rt_begin_cmd(gpu); pBuild(cmd,&bci); rt_end_cmd(gpu,cmd);

    VkDeviceAddress blas_addr = 0;
    vkMapMemory(gpu->device, blasAddrM,0,sizeof(blas_addr),0,&mp); memcpy(&blas_addr,mp,sizeof(blas_addr)); vkUnmapMemory(gpu->device, blasAddrM);

    VkBuffer tmpB[] = {scr,clasAddr,srcInfo,cnt,bscr,blasAddr,blSrc,blCnt};
    VkDeviceMemory tmpM[] = {scrM,clasAddrM,srcInfoM,cntM,bscrM,blasAddrM,blSrcM,blCntM};
    for (int i=0;i<8;i++){ if(tmpB[i]) vkDestroyBuffer(gpu->device,tmpB[i],NULL); if(tmpM[i]) vkFreeMemory(gpu->device,tmpM[i],NULL); }
    keep_buf[0]=clasDst; keep_mem[0]=clasDstM;
    keep_buf[1]=blasDst; keep_mem[1]=blasDstM;
    return blas_addr;
}

/* Phase 0 self-test (env NUSD_RT_CLUSTER_SELFTEST): build a CLAS for one
 * triangle, wrap it in a Cluster BLAS, read back both device addresses. Proves
 * the GPU cluster-AS build path end-to-end (sans trace). Fully isolated — owns
 * its geometry, no scene/render-path interaction. Two separate submissions with
 * a queue-idle between, so phase 2 sees phase 1's output without an in-cmd
 * barrier. Run with NUSD_VK_VALIDATION=1 to surface mistakes as VUIDs. */
static void nu_cluster_build_selftest(Gpu* gpu)
{
    PFN_vkGetClusterAccelerationStructureBuildSizesNV pSizes =
        (PFN_vkGetClusterAccelerationStructureBuildSizesNV)
        vkGetDeviceProcAddr(gpu->device, "vkGetClusterAccelerationStructureBuildSizesNV");
    PFN_vkCmdBuildClusterAccelerationStructureIndirectNV pBuild =
        (PFN_vkCmdBuildClusterAccelerationStructureIndirectNV)
        vkGetDeviceProcAddr(gpu->device, "vkCmdBuildClusterAccelerationStructureIndirectNV");
    if (!pSizes || !pBuild) { fprintf(stderr, "cluster-selftest: entry points missing\n"); return; }

    const VkBufferUsageFlags U_IN =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    const VkBufferUsageFlags U_AS =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    const VkBufferUsageFlags U_SCRATCH = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkMemoryPropertyFlags M_HOST =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags M_DEV = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    void* mp = NULL;
    uint32_t one = 1;

    /* one triangle */
    float verts[9] = { 0,0,0, 1,0,0, 0,1,0 };
    uint32_t idx[3] = { 0,1,2 };
    VkBuffer vbuf=0, ibuf=0; VkDeviceMemory vmem=0, imem=0;
    rt_create_buffer(gpu, sizeof(verts), U_IN, M_HOST, &vbuf, &vmem);
    rt_create_buffer(gpu, sizeof(idx),   U_IN, M_HOST, &ibuf, &imem);
    vkMapMemory(gpu->device, vmem, 0, sizeof(verts), 0, &mp); memcpy(mp, verts, sizeof(verts)); vkUnmapMemory(gpu->device, vmem);
    vkMapMemory(gpu->device, imem, 0, sizeof(idx), 0, &mp); memcpy(mp, idx, sizeof(idx)); vkUnmapMemory(gpu->device, imem);

    /* ---- Phase 1: triangle cluster (CLAS) ---- */
    VkClusterAccelerationStructureTriangleClusterInputNV tri = {0};
    tri.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
    tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    tri.maxClusterUniqueGeometryCount = 1;
    tri.maxClusterTriangleCount = 1;
    tri.maxClusterVertexCount = 3;
    tri.maxTotalTriangleCount = 1;
    tri.maxTotalVertexCount = 3;
    VkClusterAccelerationStructureInputInfoNV cin = {0};
    cin.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
    cin.maxAccelerationStructureCount = 1;
    cin.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    cin.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    cin.opInput.pTriangleClusters = &tri;
    VkAccelerationStructureBuildSizesInfoKHR csz = {0};
    csz.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pSizes(gpu->device, &cin, &csz);

    VkBuffer clasDst=0, scr=0, clasAddr=0, srcInfo=0, cnt=0;
    VkDeviceMemory clasDstM=0, scrM=0, clasAddrM=0, srcInfoM=0, cntM=0;
    rt_create_buffer(gpu, csz.accelerationStructureSize, U_AS, M_DEV, &clasDst, &clasDstM);
    rt_create_buffer(gpu, csz.buildScratchSize ? csz.buildScratchSize : 4, U_SCRATCH, M_DEV, &scr, &scrM);
    rt_create_buffer(gpu, sizeof(VkDeviceAddress), U_SCRATCH, M_HOST, &clasAddr, &clasAddrM);
    rt_create_buffer(gpu, sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV), U_IN, M_HOST, &srcInfo, &srcInfoM);
    rt_create_buffer(gpu, sizeof(uint32_t), U_IN, M_HOST, &cnt, &cntM);

    VkClusterAccelerationStructureBuildTriangleClusterInfoNV ti = {0};
    ti.clusterID = 0;
    ti.triangleCount = 1;
    ti.vertexCount = 3;
    ti.indexType = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV;
    ti.indexBufferStride = (uint16_t)sizeof(uint32_t);
    ti.vertexBufferStride = (uint16_t)(3 * sizeof(float));
    ti.indexBuffer  = rt_buf_addr(gpu, ibuf);
    ti.vertexBuffer = rt_buf_addr(gpu, vbuf);
    vkMapMemory(gpu->device, srcInfoM, 0, sizeof(ti), 0, &mp); memcpy(mp, &ti, sizeof(ti)); vkUnmapMemory(gpu->device, srcInfoM);
    vkMapMemory(gpu->device, cntM, 0, sizeof(one), 0, &mp); memcpy(mp, &one, sizeof(one)); vkUnmapMemory(gpu->device, cntM);

    VkClusterAccelerationStructureCommandsInfoNV ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
    ci.input = cin;
    ci.dstImplicitData = rt_buf_addr(gpu, clasDst);
    ci.scratchData     = rt_buf_addr(gpu, scr);
    ci.dstAddressesArray.deviceAddress = rt_buf_addr(gpu, clasAddr);
    ci.dstAddressesArray.stride = sizeof(VkDeviceAddress);
    ci.dstAddressesArray.size   = sizeof(VkDeviceAddress);
    ci.srcInfosArray.deviceAddress = rt_buf_addr(gpu, srcInfo);
    ci.srcInfosArray.stride = sizeof(ti);
    ci.srcInfosArray.size   = sizeof(ti);
    ci.srcInfosCount = rt_buf_addr(gpu, cnt);

    VkCommandBuffer cmd = rt_begin_cmd(gpu);
    pBuild(cmd, &ci);
    rt_end_cmd(gpu, cmd);

    VkDeviceAddress clas_addr = 0;
    vkMapMemory(gpu->device, clasAddrM, 0, sizeof(clas_addr), 0, &mp); memcpy(&clas_addr, mp, sizeof(clas_addr)); vkUnmapMemory(gpu->device, clasAddrM);
    fprintf(stderr, "cluster-selftest: Phase1 CLAS addr=0x%llx (size=%llu)\n",
            (unsigned long long)clas_addr, (unsigned long long)csz.accelerationStructureSize);

    /* ---- Phase 2: cluster bottom-level (Cluster BLAS) ---- */
    VkClusterAccelerationStructureClustersBottomLevelInputNV bl = {0};
    bl.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
    bl.maxTotalClusterCount = 1;
    bl.maxClusterCountPerAccelerationStructure = 1;
    VkClusterAccelerationStructureInputInfoNV bin = {0};
    bin.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
    bin.maxAccelerationStructureCount = 1;
    bin.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    bin.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    bin.opInput.pClustersBottomLevel = &bl;
    VkAccelerationStructureBuildSizesInfoKHR bsz = {0};
    bsz.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pSizes(gpu->device, &bin, &bsz);

    VkBuffer blasDst=0, bscr=0, blasAddr=0, blSrc=0, blCnt=0;
    VkDeviceMemory blasDstM=0, bscrM=0, blasAddrM=0, blSrcM=0, blCntM=0;
    rt_create_buffer(gpu, bsz.accelerationStructureSize, U_AS, M_DEV, &blasDst, &blasDstM);
    rt_create_buffer(gpu, bsz.buildScratchSize ? bsz.buildScratchSize : 4, U_SCRATCH, M_DEV, &bscr, &bscrM);
    rt_create_buffer(gpu, sizeof(VkDeviceAddress), U_SCRATCH, M_HOST, &blasAddr, &blasAddrM);
    rt_create_buffer(gpu, sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV), U_IN, M_HOST, &blSrc, &blSrcM);
    rt_create_buffer(gpu, sizeof(uint32_t), U_IN, M_HOST, &blCnt, &blCntM);

    VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV bi = {0};
    bi.clusterReferencesCount  = 1;
    bi.clusterReferencesStride = sizeof(VkDeviceAddress);
    bi.clusterReferences       = rt_buf_addr(gpu, clasAddr);  /* CLAS handle buffer from phase 1 */
    vkMapMemory(gpu->device, blSrcM, 0, sizeof(bi), 0, &mp); memcpy(mp, &bi, sizeof(bi)); vkUnmapMemory(gpu->device, blSrcM);
    vkMapMemory(gpu->device, blCntM, 0, sizeof(one), 0, &mp); memcpy(mp, &one, sizeof(one)); vkUnmapMemory(gpu->device, blCntM);

    VkClusterAccelerationStructureCommandsInfoNV bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
    bci.input = bin;
    bci.dstImplicitData = rt_buf_addr(gpu, blasDst);
    bci.scratchData     = rt_buf_addr(gpu, bscr);
    bci.dstAddressesArray.deviceAddress = rt_buf_addr(gpu, blasAddr);
    bci.dstAddressesArray.stride = sizeof(VkDeviceAddress);
    bci.dstAddressesArray.size   = sizeof(VkDeviceAddress);
    bci.srcInfosArray.deviceAddress = rt_buf_addr(gpu, blSrc);
    bci.srcInfosArray.stride = sizeof(bi);
    bci.srcInfosArray.size   = sizeof(bi);
    bci.srcInfosCount = rt_buf_addr(gpu, blCnt);

    cmd = rt_begin_cmd(gpu);
    pBuild(cmd, &bci);
    rt_end_cmd(gpu, cmd);

    VkDeviceAddress blas_addr = 0;
    vkMapMemory(gpu->device, blasAddrM, 0, sizeof(blas_addr), 0, &mp); memcpy(&blas_addr, mp, sizeof(blas_addr)); vkUnmapMemory(gpu->device, blasAddrM);
    fprintf(stderr, "cluster-selftest: Phase2 Cluster BLAS addr=0x%llx %s\n",
            (unsigned long long)blas_addr, blas_addr ? "OK" : "(NULL!)");

    VkBuffer bufs[] = {vbuf,ibuf,clasDst,scr,clasAddr,srcInfo,cnt,blasDst,bscr,blasAddr,blSrc,blCnt};
    VkDeviceMemory mems[] = {vmem,imem,clasDstM,scrM,clasAddrM,srcInfoM,cntM,blasDstM,bscrM,blasAddrM,blSrcM,blCntM};
    for (int i = 0; i < 12; i++) {
        if (bufs[i]) vkDestroyBuffer(gpu->device, bufs[i], NULL);
        if (mems[i]) vkFreeMemory(gpu->device, mems[i], NULL);
    }
    fprintf(stderr, "cluster-selftest: DONE clas=0x%llx blas=0x%llx\n",
            (unsigned long long)clas_addr, (unsigned long long)blas_addr);
}

#endif

/* Query the device-local heap size */
static VkDeviceSize rt_device_local_heap_size(Gpu* gpu)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(gpu->physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            return mem_props.memoryHeaps[i].size;
    }
    return 0;
}

/* Helper: one-shot command buffer */
static VkCommandBuffer rt_begin_cmd(Gpu* gpu)
{
    VkCommandBufferAllocateInfo ai = {0};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = gpu->command_pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(gpu->device, &ai, &cmd);

    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

static void rt_end_cmd(Gpu* gpu, VkCommandBuffer cmd)
{
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si = {0};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(gpu->graphics_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(gpu->graphics_queue);
    vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &cmd);
}

static void rt_create_storage_image(Gpu* gpu)
{
    uint32_t w = gpu->swapchain_extent.width;
    uint32_t h = gpu->swapchain_extent.height;

    /* Destroy old if exists */
    if (gpu->rt_image_view) vkDestroyImageView(gpu->device, gpu->rt_image_view, NULL);
    if (gpu->rt_image)      vkDestroyImage(gpu->device, gpu->rt_image, NULL);
    if (gpu->rt_image_mem)  vkFreeMemory(gpu->device, gpu->rt_image_mem, NULL);

    VkImageCreateInfo ici = {0};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent.width  = w;
    ici.extent.height = h;
    ici.extent.depth  = 1;
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkCreateImage(gpu->device, &ici, NULL, &gpu->rt_image);

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(gpu->device, gpu->rt_image, &req);

    VkMemoryAllocateInfo ai = {0};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    gpu_alloc_memory(gpu, &ai, &gpu->rt_image_mem);
    vkBindImageMemory(gpu->device, gpu->rt_image, gpu->rt_image_mem, 0);

    VkImageViewCreateInfo vci = {0};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = gpu->rt_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel   = 0;
    vci.subresourceRange.levelCount     = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount     = 1;

    vkCreateImageView(gpu->device, &vci, NULL, &gpu->rt_image_view);

    /* Transition to GENERAL */
    VkCommandBuffer cmd = rt_begin_cmd(gpu);

    VkImageMemoryBarrier barrier = {0};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout                       = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                           = gpu->rt_image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;
    barrier.srcAccessMask                   = 0;
    barrier.dstAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0, 0, NULL, 0, NULL, 1, &barrier);

    rt_end_cmd(gpu, cmd);

    gpu->rt_image_w = w;
    gpu->rt_image_h = h;
}

/* Phase A deferred-shading: ensure the G-buffer SSBO at binding 17 is sized
 * for the current swapchain (one GBufferEntry per pixel, 32 B each). The
 * staging buffer is host-visible for read-back during validation. */
static int gpu_ensure_gbuffer(Gpu* gpu)
{
    uint32_t w = gpu->swapchain_extent.width;
    uint32_t h = gpu->swapchain_extent.height;
    if (w == 0 || h == 0) { w = 1; h = 1; }

    if (gpu->gbuffer_buf && gpu->gbuffer_w == w && gpu->gbuffer_h == h)
        return 1;

    /* Tear down any old allocation. */
    if (gpu->gbuffer_staging_mapped) {
        vkUnmapMemory(gpu->device, gpu->gbuffer_staging_mem);
        gpu->gbuffer_staging_mapped = NULL;
    }
    if (gpu->gbuffer_staging_buf) {
        vkDestroyBuffer(gpu->device, gpu->gbuffer_staging_buf, NULL);
        vkFreeMemory(gpu->device, gpu->gbuffer_staging_mem, NULL);
        gpu->gbuffer_staging_buf = VK_NULL_HANDLE;
        gpu->gbuffer_staging_mem = VK_NULL_HANDLE;
    }
    if (gpu->gbuffer_buf) {
        vkDestroyBuffer(gpu->device, gpu->gbuffer_buf, NULL);
        vkFreeMemory(gpu->device, gpu->gbuffer_mem, NULL);
        gpu->gbuffer_buf = VK_NULL_HANDLE;
        gpu->gbuffer_mem = VK_NULL_HANDLE;
    }

    VkDeviceSize buf_size = (VkDeviceSize)w * h * 32;  /* 8 dwords/pixel */
    gpu->gbuffer_size = buf_size;
    gpu->gbuffer_w = w;
    gpu->gbuffer_h = h;

    /* Device-local SSBO (rchit + rmiss writes here). */
    {
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = buf_size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->gbuffer_buf) != VK_SUCCESS)
            return 0;

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->gbuffer_buf, &req);

        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (gpu_alloc_memory(gpu, &mai, &gpu->gbuffer_mem) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, gpu->gbuffer_buf, NULL);
            gpu->gbuffer_buf = VK_NULL_HANDLE;
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->gbuffer_buf, gpu->gbuffer_mem, 0);
    }

    /* Host-visible staging for CPU read-back during Phase A validation. */
    {
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = buf_size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->gbuffer_staging_buf) != VK_SUCCESS) {
            return 1;  /* device buffer is fine; staging is optional */
        }

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->gbuffer_staging_buf, &req);

        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (gpu_alloc_memory(gpu, &mai, &gpu->gbuffer_staging_mem) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, gpu->gbuffer_staging_buf, NULL);
            gpu->gbuffer_staging_buf = VK_NULL_HANDLE;
            return 1;
        }
        vkBindBufferMemory(gpu->device, gpu->gbuffer_staging_buf,
                           gpu->gbuffer_staging_mem, 0);
        vkMapMemory(gpu->device, gpu->gbuffer_staging_mem, 0, buf_size, 0,
                    &gpu->gbuffer_staging_mapped);
    }

    fprintf(stderr, "gpu_vulkan: Phase A G-buffer created (%ux%u, %zu bytes)\n",
            w, h, (size_t)buf_size);
    return 1;
}

/* ====================================================================
 * BLAS serialization cache (Phase A — write side)
 *
 * Opt-in via NUSD_BLAS_CACHE_WRITE=1. After the BLAS pool is built and
 * compacted, each unique BLAS is serialized to host memory with
 * vkCmdCopyAccelerationStructureToMemoryKHR and streamed into a
 * `<usd>.nzblas` sidecar. A warm start (Phase B) will deserialize these
 * instead of rebuilding — skipping the tens-of-seconds BLAS build.
 *
 * Records are content-keyed (geo_hash) so the file is robust to the
 * USD-parse mesh-order nondeterminism observed across cold loads.
 *
 * SERIALIZE is a read-only snapshot of each BLAS, so enabling the write
 * leaves the rendered frame bit-identical.
 * ==================================================================== */

void gpu_set_blas_cache_path(Gpu* gpu, const char* usd_path)
{
    if (!gpu) return;
    if (usd_path && usd_path[0]) {
        snprintf(gpu->blas_cache_path, sizeof(gpu->blas_cache_path),
                 "%s.nzblas", usd_path);
    } else {
        gpu->blas_cache_path[0] = '\0';
    }
}

#define NZBLAS_HEADER_SIZE    64u
#define NZBLAS_RECORD_SIZE    40u
#define NZBLAS_FORMAT_VERSION 3u
#define NZBLAS_ALIGN          256u  /* AS-to-memory dst.deviceAddress alignment */

static uint64_t rt_hash_string64(const char* s)
{
    uint64_t h = 1469598103934665603ULL;
    if (!s || !s[0]) return 0;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 1099511628211ULL;
    }
    return h ? h : 0x9e3779b97f4a7c15ULL;
}

static uint64_t rt_hash_mesh_cache_name64(const char* s)
{
    if (!s) return 0;
    /* OpenUSD's implicit prototype numbering is load-order dependent. Strip
     * only the volatile leading token and keep the stable prototype contents. */
    if (strncmp(s, "/__Prototype_", 13) == 0) {
        const char* p = s + 13;
        while (*p >= '0' && *p <= '9') p++;
        if (*p == '/') s = p;
    }
    return rt_hash_string64(s);
}

/* Serialize every built BLAS into `gpu->blas_cache_path`. proto_meshes[0..
 * n_protos) are the mesh indices owning a built BLAS (already truncated if
 * the build hit its time limit); prim_counts is indexed by BLAS slot.
 * Best-effort: any failure logs and returns, never aborting the RT build. */
static void blas_cache_write(Gpu* gpu, const GpuRtMeshDesc* meshes,
                             const uint32_t* mesh_to_blas,
                             const uint32_t* proto_meshes, uint32_t n_protos,
                             const uint32_t* prim_counts)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Packed per-BLAS arrays, in proto_meshes (= record) order. */
    VkAccelerationStructureKHR* hs = (VkAccelerationStructureKHR*)
        malloc((size_t)n_protos * sizeof(*hs));
    uint64_t* ser_sizes = (uint64_t*)calloc(n_protos, sizeof(uint64_t));
    uint64_t* stg_off   = (uint64_t*)malloc((size_t)n_protos * sizeof(uint64_t));
    if (!hs || !ser_sizes || !stg_off) {
        fprintf(stderr, "gpu_vulkan: BLAS cache: scratch alloc failed — skipping\n");
        free(hs); free(ser_sizes); free(stg_off);
        return;
    }
    for (uint32_t j = 0; j < n_protos; j++)
        hs[j] = gpu->blas_list[mesh_to_blas[proto_meshes[j]]];

    /* ---- 1. Query each BLAS's serialized size. ---- */
    VkQueryPool qp = VK_NULL_HANDLE;
    {
        VkQueryPoolCreateInfo qci = {0};
        qci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qci.queryType  = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR;
        qci.queryCount = n_protos;
        if (vkCreateQueryPool(gpu->device, &qci, NULL, &qp) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: BLAS cache: query pool create failed — skipping\n");
            free(hs); free(ser_sizes); free(stg_off);
            return;
        }
    }
    vkResetQueryPool(gpu->device, qp, 0, n_protos);
    {
        VkCommandBuffer cmd = rt_begin_cmd(gpu);
        vkCmdWriteAccelerationStructuresPropertiesKHR(cmd, n_protos, hs,
            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR, qp, 0);
        rt_end_cmd(gpu, cmd);
    }
    if (vkGetQueryPoolResults(gpu->device, qp, 0, n_protos,
            (size_t)n_protos * sizeof(uint64_t), ser_sizes, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: BLAS cache: serialization-size query failed — skipping\n");
        vkDestroyQueryPool(gpu->device, qp, NULL);
        free(hs); free(ser_sizes); free(stg_off);
        return;
    }
    vkDestroyQueryPool(gpu->device, qp, NULL);

    uint64_t total_ser = 0, largest = 0;
    for (uint32_t j = 0; j < n_protos; j++) {
        total_ser += ser_sizes[j];
        if (ser_sizes[j] > largest) largest = ser_sizes[j];
    }
    if (total_ser == 0 || largest == 0) {
        fprintf(stderr, "gpu_vulkan: BLAS cache: zero serialized size — skipping\n");
        free(hs); free(ser_sizes); free(stg_off);
        return;
    }

    /* ---- 2. Host-visible, device-addressable streaming staging buffer. ---- */
    uint64_t largest_aligned =
        (largest + NZBLAS_ALIGN - 1) & ~(uint64_t)(NZBLAS_ALIGN - 1);
    uint64_t staging_cap = 256ULL * 1024 * 1024;
    if (largest_aligned > staging_cap) staging_cap = largest_aligned;

    VkBuffer       stg_buf = VK_NULL_HANDLE;
    VkDeviceMemory stg_mem = VK_NULL_HANDLE;
    void*          stg_map = NULL;
    {
        VkBufferCreateInfo bci = {0};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = staging_cap;
        bci.usage       = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &stg_buf) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: BLAS cache: staging buffer create failed — skipping\n");
            free(hs); free(ser_sizes); free(stg_off);
            return;
        }
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, stg_buf, &req);
        VkMemoryAllocateFlagsInfo fi = {0};
        fi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai = {0};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.pNext           = &fi;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(gpu->device, &ai, NULL, &stg_mem) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: BLAS cache: staging alloc failed (%.0f MB) — skipping\n",
                    (double)staging_cap / (1024.0*1024.0));
            vkDestroyBuffer(gpu->device, stg_buf, NULL);
            free(hs); free(ser_sizes); free(stg_off);
            return;
        }
        vkBindBufferMemory(gpu->device, stg_buf, stg_mem, 0);
        if (vkMapMemory(gpu->device, stg_mem, 0, staging_cap, 0, &stg_map) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: BLAS cache: staging map failed — skipping\n");
            vkFreeMemory(gpu->device, stg_mem, NULL);
            vkDestroyBuffer(gpu->device, stg_buf, NULL);
            free(hs); free(ser_sizes); free(stg_off);
            return;
        }
    }
    VkDeviceAddress stg_addr = rt_buf_addr(gpu, stg_buf);
    if ((stg_addr & (NZBLAS_ALIGN - 1)) != 0) {
        /* dst.deviceAddress must be 256-aligned. A dedicated allocation is
         * page-aligned in practice; guard rather than risk a VU error. */
        fprintf(stderr, "gpu_vulkan: BLAS cache: staging addr not 256-aligned — skipping\n");
        vkUnmapMemory(gpu->device, stg_mem);
        vkFreeMemory(gpu->device, stg_mem, NULL);
        vkDestroyBuffer(gpu->device, stg_buf, NULL);
        free(hs); free(ser_sizes); free(stg_off);
        return;
    }

    /* ---- 3. Open the temp sidecar, write the 64-byte header + records. ---- */
    char tmp_path[1100];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", gpu->blas_cache_path);
    pc_mkdir_parents(gpu->blas_cache_path);

    int ok = 1;
    FILE* f = fopen(tmp_path, "wb");
    if (!f) {
        fprintf(stderr, "gpu_vulkan: BLAS cache: open %s failed (%s) — skipping\n",
                tmp_path, strerror(errno));
        ok = 0;
    }

    uint64_t blobs_off = NZBLAS_HEADER_SIZE + (uint64_t)n_protos * NZBLAS_RECORD_SIZE;
    if (ok) {
        /* Source-USD identity for coarse whole-file invalidation. */
        uint64_t src_mtime = 0, src_size = 0;
        {
            char usd[1024];
            snprintf(usd, sizeof(usd), "%s", gpu->blas_cache_path);
            size_t ul = strlen(usd);
            if (ul > 7) usd[ul - 7] = '\0';  /* strip ".nzblas" */
            struct stat st;
            if (stat(usd, &st) == 0) {
                src_mtime = (uint64_t)st.st_mtime;
                src_size  = (uint64_t)st.st_size;
            }
        }
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpu->physical_device, &props);

        uint8_t hdr[NZBLAS_HEADER_SIZE];
        memset(hdr, 0, sizeof(hdr));
        memcpy(hdr + 0,  "NUSDBLAS", 8);
        uint32_t fmt = NZBLAS_FORMAT_VERSION;
        memcpy(hdr + 8,  &fmt, 4);
        memcpy(hdr + 12, &n_protos, 4);
        memcpy(hdr + 16, &src_mtime, 8);
        memcpy(hdr + 24, &src_size, 8);
        memcpy(hdr + 32, &props.vendorID, 4);
        memcpy(hdr + 36, &props.deviceID, 4);
        memcpy(hdr + 40, props.pipelineCacheUUID, 16);
        memcpy(hdr + 56, &total_ser, 8);
        if (fwrite(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) ok = 0;

        /* Per-BLAS records — content-keyed, blob offsets precomputed. */
        uint8_t* recs = ok ? (uint8_t*)calloc(n_protos, NZBLAS_RECORD_SIZE) : NULL;
        if (ok && !recs) ok = 0;
        if (ok) {
            uint64_t off = blobs_off;
            for (uint32_t j = 0; j < n_protos; j++) {
                uint32_t m = proto_meshes[j];
                uint32_t b = mesh_to_blas[m];
                uint8_t*  r = recs + (size_t)j * NZBLAS_RECORD_SIZE;
                uint64_t gh = meshes[m].geo_hash;
                uint64_t ss = ser_sizes[j];
                uint32_t pc = prim_counts[b];
                uint32_t vc = meshes[m].vertex_count;
                uint64_t nh = rt_hash_mesh_cache_name64(meshes[m].debug_name);
                memcpy(r + 0,  &gh, 8);
                memcpy(r + 8,  &off, 8);
                memcpy(r + 16, &ss, 8);
                memcpy(r + 24, &pc, 4);
                memcpy(r + 28, &vc, 4);
                memcpy(r + 32, &nh, 8);
                off += ss;
            }
            if (fwrite(recs, NZBLAS_RECORD_SIZE, n_protos, f) != n_protos) ok = 0;
        }
        free(recs);
    }

    /* ---- 4. Stream: serialize a batch → wait → append blobs → repeat. ---- */
    uint32_t compatible = 0;
    if (ok) {
        const uint32_t CMD_BATCH = 1024;
        uint32_t j = 0;
        while (ok && j < n_protos) {
            uint32_t fs  = j;
            uint64_t off = 0;
            while (j < n_protos && (j - fs) < CMD_BATCH) {
                uint64_t a = (ser_sizes[j] + NZBLAS_ALIGN - 1) & ~(uint64_t)(NZBLAS_ALIGN - 1);
                if (off + a > staging_cap) break;
                stg_off[j] = off;
                off += a;
                j++;
            }
            if (j == fs) { ok = 0; break; }  /* staging_cap covers the largest BLAS */

            VkCommandBuffer cmd = rt_begin_cmd(gpu);
            for (uint32_t k = fs; k < j; k++) {
                VkCopyAccelerationStructureToMemoryInfoKHR ci = {0};
                ci.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR;
                ci.src   = hs[k];
                ci.dst.deviceAddress = stg_addr + stg_off[k];
                ci.mode  = VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR;
                vkCmdCopyAccelerationStructureToMemoryKHR(cmd, &ci);
            }
            rt_end_cmd(gpu, cmd);  /* submits + waits for completion */

            for (uint32_t k = fs; k < j; k++) {
                const uint8_t* blob = (const uint8_t*)stg_map + stg_off[k];
                /* Verification floor: the just-written blob must report as
                 * deserializable on this device. */
                VkAccelerationStructureVersionInfoKHR vi = {0};
                vi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_VERSION_INFO_KHR;
                vi.pVersionData = blob;
                VkAccelerationStructureCompatibilityKHR compat =
                    VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
                vkGetDeviceAccelerationStructureCompatibilityKHR(gpu->device, &vi, &compat);
                if (compat == VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR)
                    compatible++;
                if (fwrite(blob, 1, ser_sizes[k], f) != ser_sizes[k]) { ok = 0; break; }
            }
        }
    }

    /* ---- 5. Close + atomic rename. ---- */
    if (f && fclose(f) != 0) ok = 0;
    if (ok) {
        if (rename(tmp_path, gpu->blas_cache_path) != 0) {
            fprintf(stderr, "gpu_vulkan: BLAS cache: rename failed (%s)\n", strerror(errno));
            unlink(tmp_path);
            ok = 0;
        }
    } else {
        unlink(tmp_path);
    }

    /* ---- 6. Teardown + measurement log (Phase A deliverable). ---- */
    vkUnmapMemory(gpu->device, stg_mem);
    vkFreeMemory(gpu->device, stg_mem, NULL);
    vkDestroyBuffer(gpu->device, stg_buf, NULL);
    free(hs); free(ser_sizes); free(stg_off);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (ok) {
        fprintf(stderr,
            "gpu_vulkan: BLAS cache WRITE: %u BLAS, %.1f MB serialized "
            "(avg %.1f KB/BLAS), %u/%u device-COMPATIBLE, %.2fs -> %s\n",
            n_protos, (double)total_ser / (1024.0*1024.0),
            (double)total_ser / 1024.0 / (double)n_protos,
            compatible, n_protos, secs, gpu->blas_cache_path);
    } else {
        fprintf(stderr,
            "gpu_vulkan: BLAS cache WRITE failed after %.2fs — no sidecar written\n",
            secs);
    }
}

/* ====================================================================
 * BLAS serialization cache (Phase B — deserialize / warm-load build-skip)
 *
 * Opt-in via NUSD_BLAS_CACHE=1. Attempts to populate the entire BLAS set
 * from the `<usd>.nzblas` sidecar (`vkCmdCopyMemoryToAccelerationStructureKHR`,
 * MODE_DESERIALIZE) instead of rebuilding it. On full success the caller
 * skips the BLAS build + compaction entirely. Returns 0 — caller builds
 * normally, gpu->blas_* untouched — on a disabled cache, a missing / stale /
 * wrong-device file, any TLAS-needed BLAS whose triangle hash is absent, or
 * any device-incompatible blob (a build fallback for any miss).
 * ==================================================================== */
static int blas_cache_try_load(Gpu* gpu, const GpuRtMeshDesc* meshes,
                               uint32_t nmeshes, const uint32_t* mesh_to_blas,
                               uint32_t nblas,
                               uint8_t** out_blas_included,
                               uint32_t* out_nblas_included)
{
    const char* bc = getenv("NUSD_BLAS_CACHE");
    if (!bc || !bc[0] || bc[0] == '0') return 0;
    if (!gpu->blas_cache_path[0] || nblas == 0) return 0;
    if (!out_blas_included || !out_nblas_included) return 0;
    *out_blas_included = NULL;
    *out_nblas_included = 0;

    /* All function-scope state up front so the goto-fail cleanup never
     * jumps over a live declaration. */
    int ret = 0;
    FILE* f = NULL;
    uint8_t*  recs          = NULL;
    uint64_t* blas_geo_hash = NULL;
    uint64_t* blas_name_hash = NULL;
    uint32_t* blas_prim_count = NULL;
    uint32_t* blas_vertex_count = NULL;
    int*      match         = NULL;
    uint8_t*  needed        = NULL;
    uint8_t*  included      = NULL;
    uint32_t* load_blas     = NULL;
    uint64_t* deser_size    = NULL;
    uint64_t* blob_off      = NULL;
    uint64_t* ser_size      = NULL;
    uint64_t* pool_off      = NULL;
    uint64_t* stg_slot      = NULL;
    int32_t*  htab          = NULL;
    int32_t*  name_htab     = NULL;
    VkAccelerationStructureKHR* blas_list = NULL;
    VkBuffer       pool_buf = VK_NULL_HANDLE;
    VkDeviceMemory pool_mem = VK_NULL_HANDLE;
    VkBuffer       stg_buf  = VK_NULL_HANDLE;
    VkDeviceMemory stg_mem  = VK_NULL_HANDLE;
    void*          stg_map  = NULL;
    uint32_t rec_count = 0, hcap = 1, needed_count = 0, load_count = 0;
    uint64_t pool_bytes = 0, staging_cap = 256ULL * 1024 * 1024, largest = 0;
    VkDeviceAddress stg_addr = 0;
    struct timespec t0, t1;
    uint8_t hdr[NZBLAS_HEADER_SIZE];

    clock_gettime(CLOCK_MONOTONIC, &t0);

    f = fopen(gpu->blas_cache_path, "rb");
    if (!f) return 0;  /* no sidecar — silent; caller builds + may write one */

    /* ---- 1. Header validation. ---- */
    if (fread(hdr, 1, sizeof(hdr), f) != sizeof(hdr) ||
        memcmp(hdr, "NUSDBLAS", 8) != 0) {
        fclose(f); return 0;
    }
    {
        uint32_t fmt = 0;
        memcpy(&fmt, hdr + 8, 4);
        memcpy(&rec_count, hdr + 12, 4);
        if (fmt != NZBLAS_FORMAT_VERSION || rec_count == 0) {
            fprintf(stderr, "gpu_vulkan: BLAS cache: format v%u unusable — rebuilding\n", fmt);
            fclose(f); return 0;
        }
        uint64_t hdr_mtime = 0, hdr_size = 0;
        uint32_t hdr_vendor = 0, hdr_device = 0;
        memcpy(&hdr_mtime,  hdr + 16, 8);
        memcpy(&hdr_size,   hdr + 24, 8);
        memcpy(&hdr_vendor, hdr + 32, 4);
        memcpy(&hdr_device, hdr + 36, 4);
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpu->physical_device, &props);
        if (hdr_vendor != props.vendorID || hdr_device != props.deviceID ||
            memcmp(hdr + 40, props.pipelineCacheUUID, 16) != 0) {
            fprintf(stderr, "gpu_vulkan: BLAS cache: device/driver mismatch — rebuilding\n");
            fclose(f); return 0;
        }
        char usd[1024];
        snprintf(usd, sizeof(usd), "%s", gpu->blas_cache_path);
        size_t ul = strlen(usd);
        if (ul > 7) usd[ul - 7] = '\0';   /* strip ".nzblas" */
        struct stat st;
        if (stat(usd, &st) != 0 ||
            (uint64_t)st.st_mtime != hdr_mtime ||
            (uint64_t)st.st_size  != hdr_size) {
            fprintf(stderr, "gpu_vulkan: BLAS cache: source USD changed — rebuilding\n");
            fclose(f); return 0;
        }
    }

    /* ---- 2. Records. ---- */
    recs = (uint8_t*)malloc((size_t)rec_count * NZBLAS_RECORD_SIZE);
    if (!recs || fread(recs, NZBLAS_RECORD_SIZE, rec_count, f) != rec_count) goto fail;

    /* ---- 3. Match every TLAS-needed BLAS to a record by BLAS content hash.
     * Hidden BLASes are never referenced by the TLAS, so a cold cache that
     * wrote only visible/included BLASes remains valid for warm load. */
    blas_geo_hash = (uint64_t*)calloc(nblas, sizeof(uint64_t));
    blas_name_hash = (uint64_t*)calloc(nblas, sizeof(uint64_t));
    blas_prim_count = (uint32_t*)calloc(nblas, sizeof(uint32_t));
    blas_vertex_count = (uint32_t*)calloc(nblas, sizeof(uint32_t));
    match         = (int*)malloc((size_t)nblas * sizeof(int));
    needed        = (uint8_t*)calloc(nblas, sizeof(uint8_t));
    included      = (uint8_t*)calloc(nblas, sizeof(uint8_t));
    if (!blas_geo_hash || !blas_name_hash || !blas_prim_count ||
        !blas_vertex_count || !match || !needed || !included) goto fail;
    for (uint32_t b = 0; b < nblas; b++) match[b] = -1;
    for (uint32_t m = 0; m < nmeshes; m++) {
        if ((uint32_t)meshes[m].prototype_idx == m) {
            uint32_t b = mesh_to_blas[m];
            blas_geo_hash[b] = meshes[m].geo_hash;
            blas_name_hash[b] = rt_hash_mesh_cache_name64(meshes[m].debug_name);
            blas_prim_count[b] = meshes[m].index_count / 3u;
            blas_vertex_count[b] = meshes[m].vertex_count;
        }
        if (meshes[m].visible) {
            uint32_t b = mesh_to_blas[m];
            if (!needed[b]) {
                needed[b] = 1;
                needed_count++;
            }
        }
    }
    if (needed_count == 0) goto fail;
    while (hcap < rec_count * 2u) hcap <<= 1;
    htab = (int32_t*)malloc((size_t)hcap * sizeof(int32_t));
    name_htab = (int32_t*)malloc((size_t)hcap * sizeof(int32_t));
    if (!htab || !name_htab) goto fail;
    for (uint32_t i = 0; i < hcap; i++) {
        htab[i] = -1;
        name_htab[i] = -1;
    }
    for (uint32_t r = 0; r < rec_count; r++) {
        uint64_t gh, nh;
        memcpy(&gh, recs + (size_t)r * NZBLAS_RECORD_SIZE, 8);
        uint32_t s = (uint32_t)gh & (hcap - 1);
        while (htab[s] >= 0) s = (s + 1) & (hcap - 1);
        htab[s] = (int32_t)r;
        memcpy(&nh, recs + (size_t)r * NZBLAS_RECORD_SIZE + 32, 8);
        if (nh) {
            s = (uint32_t)nh & (hcap - 1);
            while (name_htab[s] >= 0) s = (s + 1) & (hcap - 1);
            name_htab[s] = (int32_t)r;
        }
    }
    {
        uint32_t unmatched = 0, name_fallback = 0;
        uint32_t diag_printed = 0;
        const char* diag_env = getenv("NUSD_BLAS_CACHE_DIAG");
        int diag = (diag_env && diag_env[0] && diag_env[0] != '0');
        for (uint32_t b = 0; b < nblas; b++) {
            if (!needed[b]) continue;
            uint64_t gh = blas_geo_hash[b];
            uint64_t nh = blas_name_hash[b];
            int found = -1;
            if (gh != 0) {
                uint32_t s = (uint32_t)gh & (hcap - 1);
                while (htab[s] >= 0) {
                    const uint8_t* rr = recs + (size_t)htab[s] * NZBLAS_RECORD_SIZE;
                    uint64_t rh;
                    uint32_t pc = 0, vc = 0;
                    memcpy(&rh, rr, 8);
                    memcpy(&pc, rr + 24, 4);
                    memcpy(&vc, rr + 28, 4);
                    if (rh == gh && pc == blas_prim_count[b] &&
                        vc == blas_vertex_count[b]) {
                        found = htab[s];
                        break;
                    }
                    s = (s + 1) & (hcap - 1);
                }
            }
            if (found < 0 && nh != 0) {
                uint32_t s = (uint32_t)nh & (hcap - 1);
                while (name_htab[s] >= 0) {
                    const uint8_t* rr = recs + (size_t)name_htab[s] * NZBLAS_RECORD_SIZE;
                    uint64_t rn = 0;
                    uint32_t pc = 0, vc = 0;
                    memcpy(&rn, rr + 32, 8);
                    memcpy(&pc, rr + 24, 4);
                    memcpy(&vc, rr + 28, 4);
                    if (rn == nh && pc == blas_prim_count[b] &&
                        vc == blas_vertex_count[b]) {
                        found = name_htab[s];
                        name_fallback++;
                        break;
                    }
                    s = (s + 1) & (hcap - 1);
                }
            }
            match[b] = found;
            if (found < 0) {
                unmatched++;
                if (diag && diag_printed < 32) {
                    uint32_t proto_mesh = UINT32_MAX;
                    const char* name = "";
                    for (uint32_t m = 0; m < nmeshes; m++) {
                        if ((uint32_t)meshes[m].prototype_idx == m &&
                            mesh_to_blas[m] == b) {
                            proto_mesh = m;
                            name = meshes[m].debug_name ? meshes[m].debug_name : "";
                            break;
                        }
                    }
                    fprintf(stderr,
                            "gpu_vulkan: BLAS cache miss diag: b=%u proto_mesh=%u gh=%016llx nh=%016llx pc=%u vc=%u name=%s\n",
                            b, proto_mesh,
                            (unsigned long long)gh,
                            (unsigned long long)nh,
                            blas_prim_count[b], blas_vertex_count[b],
                            name);
                    diag_printed++;
                }
            }
        }
        if (unmatched > 0) {
            fprintf(stderr, "gpu_vulkan: BLAS cache: %u/%u TLAS-needed BLAS not in cache - building misses\n",
                    unmatched, needed_count);
        }
        if (name_fallback > 0) {
            fprintf(stderr,
                    "gpu_vulkan: BLAS cache: %u TLAS-needed BLAS matched by stable mesh path fallback\n",
                    name_fallback);
        }
    }
    load_blas = (uint32_t*)malloc((size_t)needed_count * sizeof(uint32_t));
    if (!load_blas) goto fail;
    for (uint32_t b = 0; b < nblas; b++) {
        if (!needed[b] || match[b] < 0) continue;
        load_blas[load_count++] = b;
        included[b] = 1;
    }
    if (load_count == 0) goto fail;
    if (load_count < needed_count) {
        fprintf(stderr,
                "gpu_vulkan: BLAS cache PARTIAL: %u/%u TLAS-needed BLAS will load from cache\n",
                load_count, needed_count);
    }

    /* ---- 4. Per-BLAS deserialized size + device-compat check. ---- */
    deser_size = (uint64_t*)malloc((size_t)nblas * sizeof(uint64_t));
    blob_off   = (uint64_t*)malloc((size_t)nblas * sizeof(uint64_t));
    ser_size   = (uint64_t*)malloc((size_t)nblas * sizeof(uint64_t));
    if (!deser_size || !blob_off || !ser_size) goto fail;
    for (uint32_t i = 0; i < load_count; i++) {
        uint32_t b = load_blas[i];
        const uint8_t* r = recs + (size_t)match[b] * NZBLAS_RECORD_SIZE;
        uint8_t bh[56];   /* serialized-AS header: 2*UUID + 3*u64 */
        memcpy(&blob_off[b], r + 8,  8);
        memcpy(&ser_size[b], r + 16, 8);
        if (ser_size[b] < sizeof(bh) ||
            fseek(f, (long)blob_off[b], SEEK_SET) != 0 ||
            fread(bh, 1, sizeof(bh), f) != sizeof(bh)) {
            fprintf(stderr, "gpu_vulkan: BLAS cache: truncated blob — rebuilding\n");
            goto fail;
        }
        VkAccelerationStructureVersionInfoKHR vi = {0};
        vi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_VERSION_INFO_KHR;
        vi.pVersionData = bh;
        VkAccelerationStructureCompatibilityKHR compat =
            VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
        vkGetDeviceAccelerationStructureCompatibilityKHR(gpu->device, &vi, &compat);
        if (compat != VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR) {
            fprintf(stderr, "gpu_vulkan: BLAS cache: incompatible blob — rebuilding\n");
            goto fail;
        }
        memcpy(&deser_size[b], bh + 40, 8);   /* deserializedSize */
        if (deser_size[b] == 0) goto fail;
    }

    /* ---- 5. Allocate the deserialize pool + AS handles. ---- */
    pool_off = (uint64_t*)malloc((size_t)nblas * sizeof(uint64_t));
    if (!pool_off) goto fail;
    for (uint32_t i = 0; i < load_count; i++) {
        uint32_t b = load_blas[i];
        pool_off[b] = pool_bytes;
        pool_bytes += (deser_size[b] + NZBLAS_ALIGN - 1) & ~(uint64_t)(NZBLAS_ALIGN - 1);
    }
    if (!rt_create_buffer(gpu, pool_bytes,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &pool_buf, &pool_mem)) {
        fprintf(stderr, "gpu_vulkan: BLAS cache: pool alloc (%.0f MB) failed — rebuilding\n",
                (double)pool_bytes / (1024.0*1024.0));
        goto fail;
    }
    blas_list = (VkAccelerationStructureKHR*)calloc(nblas, sizeof(*blas_list));
    if (!blas_list) goto fail;
    for (uint32_t i = 0; i < load_count; i++) {
        uint32_t b = load_blas[i];
        VkAccelerationStructureCreateInfoKHR ci = {0};
        ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = pool_buf;
        ci.offset = pool_off[b];
        ci.size   = deser_size[b];
        ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if (vkCreateAccelerationStructureKHR(gpu->device, &ci, NULL, &blas_list[b]) != VK_SUCCESS)
            goto fail;
    }

    /* ---- 6. Host-visible, device-addressable streaming staging buffer. ---- */
    for (uint32_t i = 0; i < load_count; i++) {
        uint32_t b = load_blas[i];
        uint64_t a = (ser_size[b] + NZBLAS_ALIGN - 1) & ~(uint64_t)(NZBLAS_ALIGN - 1);
        if (a > largest) largest = a;
    }
    if (largest > staging_cap) staging_cap = largest;
    {
        VkBufferCreateInfo bci = {0};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = staging_cap;
        bci.usage       = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &stg_buf) != VK_SUCCESS) goto fail;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, stg_buf, &req);
        VkMemoryAllocateFlagsInfo fi = {0};
        fi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai = {0};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.pNext           = &fi;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(gpu->device, &ai, NULL, &stg_mem) != VK_SUCCESS) goto fail;
        vkBindBufferMemory(gpu->device, stg_buf, stg_mem, 0);
        if (vkMapMemory(gpu->device, stg_mem, 0, staging_cap, 0, &stg_map) != VK_SUCCESS) goto fail;
    }
    stg_addr = rt_buf_addr(gpu, stg_buf);
    if ((stg_addr & (NZBLAS_ALIGN - 1)) != 0) {
        fprintf(stderr, "gpu_vulkan: BLAS cache: staging addr not 256-aligned — rebuilding\n");
        goto fail;
    }

    /* ---- 7. Stream: fill staging from disk -> deserialize batch -> repeat. ---- */
    stg_slot = (uint64_t*)malloc((size_t)nblas * sizeof(uint64_t));
    if (!stg_slot) goto fail;
    {
        const uint32_t CMD_BATCH = 1024;
        uint32_t j = 0;
        while (j < load_count) {
            uint32_t fs  = j;
            uint64_t off = 0;
            while (j < load_count && (j - fs) < CMD_BATCH) {
                uint32_t b = load_blas[j];
                uint64_t a = (ser_size[b] + NZBLAS_ALIGN - 1) & ~(uint64_t)(NZBLAS_ALIGN - 1);
                if (off + a > staging_cap) break;
                stg_slot[b] = off;
                off += a;
                j++;
            }
            if (j == fs) goto fail;   /* staging_cap covers the largest blob */
            for (uint32_t k = fs; k < j; k++) {
                uint32_t b = load_blas[k];
                if (fseek(f, (long)blob_off[b], SEEK_SET) != 0 ||
                    fread((uint8_t*)stg_map + stg_slot[b], 1, ser_size[b], f) != ser_size[b])
                    goto fail;
            }
            VkCommandBuffer cmd = rt_begin_cmd(gpu);
            for (uint32_t k = fs; k < j; k++) {
                uint32_t b = load_blas[k];
                VkCopyMemoryToAccelerationStructureInfoKHR ci = {0};
                ci.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR;
                ci.src.deviceAddress = stg_addr + stg_slot[b];
                ci.dst  = blas_list[b];
                ci.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR;
                vkCmdCopyMemoryToAccelerationStructureKHR(cmd, &ci);
            }
            rt_end_cmd(gpu, cmd);   /* submits + waits for completion */
        }
    }

    /* ---- 8. Success — hand the pool + AS handles to the Gpu. ---- */
    gpu->blas_list     = blas_list;
    gpu->blas_count    = nblas;
    gpu->blas_pool_buf = pool_buf;
    gpu->blas_pool_mem = pool_mem;
    *out_blas_included = included;
    *out_nblas_included = load_count;
    blas_list = NULL;            /* owned by gpu now — cleanup must not destroy */
    pool_buf  = VK_NULL_HANDLE;
    pool_mem  = VK_NULL_HANDLE;
    included  = NULL;            /* owned by caller now */
    ret = 1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    {
        double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        fprintf(stderr,
            "gpu_vulkan: BLAS cache LOAD: %u/%u TLAS-needed BLAS deserialized from cache "
            "(%.1f MB, %.2fs) - BLAS build + compaction skipped\n",
            load_count, needed_count, (double)pool_bytes / (1024.0*1024.0), secs);
    }

fail:
    if (stg_map) vkUnmapMemory(gpu->device, stg_mem);
    if (stg_mem) vkFreeMemory(gpu->device, stg_mem, NULL);
    if (stg_buf) vkDestroyBuffer(gpu->device, stg_buf, NULL);
    if (blas_list) {
        for (uint32_t b = 0; b < nblas; b++)
            if (blas_list[b])
                vkDestroyAccelerationStructureKHR(gpu->device, blas_list[b], NULL);
        free(blas_list);
    }
    if (pool_buf) vkDestroyBuffer(gpu->device, pool_buf, NULL);
    if (pool_mem) vkFreeMemory(gpu->device, pool_mem, NULL);
    free(recs); free(blas_geo_hash); free(blas_name_hash);
    free(blas_prim_count); free(blas_vertex_count); free(match);
    free(needed); free(included); free(load_blas);
    free(deser_size); free(blob_off); free(ser_size);
    free(pool_off); free(stg_slot); free(htab); free(name_htab);
    if (f) fclose(f);
    return ret;
}

typedef struct {
    uint32_t     blas;
    uint32_t     proto_mesh;
    uint32_t     source_id;
    uint32_t     visible_instances;
    uint32_t     triangles;
    VkDeviceSize size;
    const char*  name;
} RtBlasCandidate;

typedef struct {
    VkDeviceSize pool_budget;
    VkDeviceSize attempted_pool_size;
    VkDeviceSize scratch_size;
    VkDeviceSize post_build_reserve;
    const char*  failed_allocation;
    VkDeviceSize failed_allocation_size;
    int          build_started;
    int          strict_no_drop;
} RtOmitManifestDiag;

static int rt_blas_candidate_small_first(const void* av, const void* bv)
{
    const RtBlasCandidate* a = (const RtBlasCandidate*)av;
    const RtBlasCandidate* b = (const RtBlasCandidate*)bv;
    if (a->size < b->size) return -1;
    if (a->size > b->size) return 1;
    if (a->visible_instances > b->visible_instances) return -1;
    if (a->visible_instances < b->visible_instances) return 1;
    return (a->blas > b->blas) - (a->blas < b->blas);
}

static int rt_blas_candidate_large_first(const void* av, const void* bv)
{
    const RtBlasCandidate* a = (const RtBlasCandidate*)av;
    const RtBlasCandidate* b = (const RtBlasCandidate*)bv;
    if (a->size > b->size) return -1;
    if (a->size < b->size) return 1;
    if (a->visible_instances > b->visible_instances) return -1;
    if (a->visible_instances < b->visible_instances) return 1;
    return (a->blas > b->blas) - (a->blas < b->blas);
}

static int rt_env_u32(const char* name, uint32_t def, uint32_t max_value)
{
    const char* s = getenv(name);
    if (!s || !s[0]) return def;
    char* end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (end == s || v > (unsigned long)max_value) return def;
    return (uint32_t)v;
}

static int rt_env_enabled_default_off(const char* name)
{
    const char* s = getenv(name);
    return (s && s[0] && s[0] != '0');
}

static uint64_t rt_align_u64(uint64_t v, uint64_t align)
{
    return (v + align - 1u) & ~(align - 1u);
}

static void rt_include_blas(uint32_t b, VkDeviceSize blas_size,
                            uint8_t* blas_included, uint64_t* blas_offsets,
                            uint64_t* pool_size, uint32_t* nblas_included)
{
    uint64_t aligned_sz = rt_align_u64((uint64_t)blas_size, 256u);
    blas_offsets[b] = *pool_size;
    blas_included[b] = 1;
    *pool_size += aligned_sz;
    (*nblas_included)++;
}

static void rt_dump_excluded_blas_diag(const RtBlasCandidate* large_order,
                                       uint32_t nblas,
                                       const uint8_t* blas_included,
                                       uint32_t limit)
{
    if (limit == 0) return;
    uint32_t printed = 0;
    for (uint32_t i = 0; i < nblas && printed < limit; i++) {
        const RtBlasCandidate* c = &large_order[i];
        if (blas_included[c->blas]) continue;
        if (c->visible_instances == 0) continue;
        fprintf(stderr,
                "gpu_vulkan: excluded BLAS diag: b=%u size=%.1f MB tris=%u visible_instances=%u source_id=%u name=%s\n",
                c->blas, (double)c->size / (1024.0 * 1024.0),
                c->triangles, c->visible_instances,
                c->source_id, c->name ? c->name : "");
        printed++;
    }
}

static void rt_json_string(FILE* f, const char* s)
{
    fputc('"', f);
    if (s) {
        for (; *s; s++) {
            unsigned char c = (unsigned char)*s;
            if (c == '"' || c == '\\') {
                fputc('\\', f);
                fputc(c, f);
            } else if (c >= 0x20) {
                fputc(c, f);
            }
        }
    }
    fputc('"', f);
}

static void rt_write_omitted_blas_manifest(const char* reason,
                                           const RtBlasCandidate* large_order,
                                           uint32_t nblas,
                                           const uint8_t* blas_included,
                                           const GpuRtMeshDesc* meshes,
                                           uint32_t nblas_included,
                                           uint32_t total_visible_blas,
                                           const RtOmitManifestDiag* diag)
{
    const char* path = getenv("NUSD_RT_OMIT_MANIFEST");
    if (!path || !path[0]) return;

    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "gpu_vulkan: failed to write omitted BLAS manifest %s: %s\n",
                path, strerror(errno));
        return;
    }
    uint64_t visible_bytes = 0;
    uint64_t omitted_visible_bytes = 0;
    uint64_t visible_instances = 0;
    uint64_t omitted_visible_instances = 0;
    for (uint32_t i = 0; i < nblas; i++) {
        const RtBlasCandidate* c = &large_order[i];
        if (c->visible_instances == 0) continue;
        visible_instances += (uint64_t)c->visible_instances;
        visible_bytes += (uint64_t)c->size;
        if (!blas_included[c->blas]) {
            omitted_visible_instances += (uint64_t)c->visible_instances;
            omitted_visible_bytes += (uint64_t)c->size;
        }
    }
    uint32_t top_visible_limit = 64;
    {
        const char* s = getenv("NUSD_RT_OMIT_TOP_VISIBLE");
        if (s && s[0]) {
            char* end = NULL;
            unsigned long v = strtoul(s, &end, 10);
            if (end != s && v <= 1024UL)
                top_visible_limit = (uint32_t)v;
        }
    }
    int build_started = diag ? diag->build_started : 1;
    uint32_t built_blas = build_started ? nblas_included : 0;
    fprintf(f,
            "{\n"
            "  \"manifest_version\": 3,\n"
            "  \"reason\": ");
    rt_json_string(f, reason ? reason : "omitted");
    fprintf(f,
            ",\n"
            "  \"built_blas\": %u,\n"
            "  \"selected_blas\": %u,\n"
            "  \"total_blas\": %u,\n"
            "  \"total_visible_blas\": %u,\n"
            "  \"total_visible_instances\": %llu,\n"
            "  \"omitted_visible_instances\": %llu,\n"
            "  \"visible_blas_precompact_bytes\": %llu,\n"
            "  \"omitted_visible_precompact_bytes\": %llu,\n"
            "  \"allocation_context\": {\n"
            "    \"build_started\": %s,\n"
            "    \"strict_no_drop\": %s,\n"
            "    \"pool_budget_bytes\": %llu,\n"
            "    \"attempted_pool_bytes\": %llu,\n"
            "    \"scratch_bytes\": %llu,\n"
            "    \"post_build_reserve_bytes\": %llu,\n"
            "    \"failed_allocation\": ",
            built_blas, nblas_included, nblas, total_visible_blas,
            (unsigned long long)visible_instances,
            (unsigned long long)omitted_visible_instances,
            (unsigned long long)visible_bytes,
            (unsigned long long)omitted_visible_bytes,
            build_started ? "true" : "false",
            (diag && diag->strict_no_drop) ? "true" : "false",
            (unsigned long long)(diag ? diag->pool_budget : 0),
            (unsigned long long)(diag ? diag->attempted_pool_size : 0),
            (unsigned long long)(diag ? diag->scratch_size : 0),
            (unsigned long long)(diag ? diag->post_build_reserve : 0));
    rt_json_string(f, (diag && diag->failed_allocation) ? diag->failed_allocation : "");
    fprintf(f,
            ",\n"
            "    \"failed_allocation_bytes\": %llu\n"
            "  },\n"
            "  \"omitted\": [\n",
            (unsigned long long)(diag ? diag->failed_allocation_size : 0));

    int first = 1;
    for (uint32_t i = 0; i < nblas; i++) {
        const RtBlasCandidate* c = &large_order[i];
        if (c->visible_instances == 0 || blas_included[c->blas]) continue;
        const GpuRtMeshDesc* md = &meshes[c->proto_mesh];
        if (!first) fprintf(f, ",\n");
        first = 0;
        fprintf(f,
                "    {\"blas\": %u, \"proto_mesh\": %u, \"source_id\": %u, "
                "\"visible_instances\": %u, \"triangles\": %u, "
                "\"size_bytes\": %llu, \"material_index\": %d, "
                "\"bounds_min\": [%.9g, %.9g, %.9g], "
                "\"bounds_max\": [%.9g, %.9g, %.9g], "
                "\"name\": ",
                c->blas, c->proto_mesh, c->source_id,
                c->visible_instances, c->triangles,
                (unsigned long long)c->size, md->material_index,
                md->bounds_min[0], md->bounds_min[1], md->bounds_min[2],
                md->bounds_max[0], md->bounds_max[1], md->bounds_max[2]);
        rt_json_string(f, c->name ? c->name : "");
        fprintf(f, "}");
    }
    fprintf(f, "\n  ],\n"
               "  \"largest_visible\": [\n");
    first = 1;
    uint32_t written = 0;
    for (uint32_t i = 0; i < nblas && written < top_visible_limit; i++) {
        const RtBlasCandidate* c = &large_order[i];
        if (c->visible_instances == 0) continue;
        const GpuRtMeshDesc* md = &meshes[c->proto_mesh];
        if (!first) fprintf(f, ",\n");
        first = 0;
        fprintf(f,
                "    {\"blas\": %u, \"proto_mesh\": %u, \"source_id\": %u, "
                "\"included\": %s, \"visible_instances\": %u, "
                "\"triangles\": %u, \"size_bytes\": %llu, "
                "\"material_index\": %d, "
                "\"bounds_min\": [%.9g, %.9g, %.9g], "
                "\"bounds_max\": [%.9g, %.9g, %.9g], "
                "\"name\": ",
                c->blas, c->proto_mesh, c->source_id,
                blas_included[c->blas] ? "true" : "false",
                c->visible_instances, c->triangles,
                (unsigned long long)c->size, md->material_index,
                md->bounds_min[0], md->bounds_min[1], md->bounds_min[2],
                md->bounds_max[0], md->bounds_max[1], md->bounds_max[2]);
        rt_json_string(f, c->name ? c->name : "");
        fprintf(f, "}");
        written++;
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
}

int gpu_build_rt_scene(Gpu* gpu,
                       const GpuRtMeshDesc* meshes, uint32_t nmeshes,
                       const uint32_t* rgen_spv, uint32_t rgen_size,
                       const uint32_t* miss_spv, uint32_t miss_size,
                       const uint32_t* chit_spv, uint32_t chit_size,
                       const uint32_t* rint_spv, uint32_t rint_size,
                       const uint32_t* curve_chit_spv, uint32_t curve_chit_size)
{
    if (!gpu || !gpu->rt_available) return 0;

    /* Building/rebuilding the RT scene tears down BLAS/TLAS/pipeline/SBT
     * and recreates the descriptor set; any cached cmd buffer is stale. */
    gpu_invalidate_rt_cmd_cache(gpu);
    gpu_invalidate_tiled_cmd_cache(gpu);

    /* Phase 11.A.2.5: curves can render without any meshes. has_curves
     * gates every curve-specific code branch below. */
    int has_curves = (rint_spv && curve_chit_spv && rint_size > 0 && curve_chit_size > 0
                      && gpu->curve_blas != VK_NULL_HANDLE
                      && gpu->curve_seg_count > 0);
    if (nmeshes == 0 && !has_curves) return 0;

    /* All meshes share the same underlying buffer; grab the base addresses once.
     * Curve-only scenes (nmeshes==0) skip this entirely. */
    VkDeviceAddress vb_addr = (nmeshes > 0) ? rt_buf_addr(gpu, meshes[0].vertex_buf->buffer) : 0;
    VkDeviceAddress ib_addr = (nmeshes > 0) ? rt_buf_addr(gpu, meshes[0].index_buf->buffer) : 0;

    /* ---- Instancing-aware BLAS build ---- */
    /* Only build BLASes for unique prototypes (prototype_idx == own index).
     * Instances share their prototype's BLAS via TLAS instance transforms. */

    /* Build mesh → BLAS mapping.  mesh_to_blas[m] = BLAS slot for mesh m. */
    uint32_t* mesh_to_blas = (uint32_t*)malloc(nmeshes * sizeof(uint32_t));
    uint32_t nblas = 0;
    for (uint32_t m = 0; m < nmeshes; m++) {
        if ((uint32_t)meshes[m].prototype_idx == m) {
            mesh_to_blas[m] = nblas++;
        } else {
            mesh_to_blas[m] = mesh_to_blas[meshes[m].prototype_idx];
        }
    }

    fprintf(stderr, "gpu_vulkan: RT instancing: %u total meshes → %u unique BLAS\n", nmeshes, nblas);

    /* blas_included / nblas_included are hoisted here (rather than declared
     * at the pool-build site below) so the Phase-B cache path can populate
     * them and jump straight to build_tlas_phase. */
    uint8_t* blas_included = NULL;
    uint8_t* blas_preloaded = NULL;
    uint32_t nblas_included = 0;
    uint32_t nblas_preloaded = 0;

    /* Phase B: try to deserialize the whole BLAS set from the .nzblas cache.
     * Full success skips the BLAS build + compaction; any miss / staleness /
     * device mismatch falls through to a normal build (gpu->blas_* untouched). */
    if (blas_cache_try_load(gpu, meshes, nmeshes, mesh_to_blas, nblas,
                            &blas_included, &nblas_included)) {
        blas_preloaded = (uint8_t*)malloc(nblas);
        if (!blas_preloaded) {
            free(mesh_to_blas);
            free(blas_included);
            gpu_destroy_rt_scene(gpu);
            return 0;
        }
        memcpy(blas_preloaded, blas_included, nblas);
        nblas_preloaded = nblas_included;
    }

    /* Pre-flight: query BLAS sizes for unique prototypes only */
    uint32_t* prim_counts = (uint32_t*)malloc(nblas * sizeof(uint32_t));
    VkDeviceSize* blas_sizes = (VkDeviceSize*)malloc(nblas * sizeof(VkDeviceSize));
    VkDeviceSize max_scratch_size = 0;
    uint64_t total_blas_bytes = 0;
    uint32_t total_triangles  = 0;

    for (uint32_t m = 0; m < nmeshes; m++) {
        if ((uint32_t)meshes[m].prototype_idx != m) continue;  /* skip instances */
        uint32_t b = mesh_to_blas[m];
        const GpuRtMeshDesc* md = &meshes[m];
        VkDeviceAddress mesh_vb = vb_addr + (VkDeviceAddress)md->vertex_offset * md->vertex_stride;
        VkDeviceAddress mesh_ib = ib_addr + (VkDeviceAddress)md->index_offset * sizeof(uint32_t);

        VkAccelerationStructureGeometryTrianglesDataKHR triangles = {0};
        triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = mesh_vb;
        triangles.vertexStride             = md->vertex_stride;
        triangles.maxVertex                = md->vertex_count - 1;
        triangles.indexType                = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress  = mesh_ib;

        VkAccelerationStructureGeometryKHR geom = {0};
        geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geom.geometry.triangles = triangles;

        VkAccelerationStructureBuildGeometryInfoKHR build_info = {0};
        build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                   VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
        build_info.geometryCount = 1;
        build_info.pGeometries   = &geom;

        prim_counts[b] = md->index_count / 3;

        VkAccelerationStructureBuildSizesInfoKHR sizes = {0};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkGetAccelerationStructureBuildSizesKHR(gpu->device,
                                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                &build_info, &prim_counts[b], &sizes);

        blas_sizes[b] = sizes.accelerationStructureSize;
        if (sizes.buildScratchSize > max_scratch_size)
            max_scratch_size = sizes.buildScratchSize;
        total_blas_bytes += sizes.accelerationStructureSize;
        total_triangles  += prim_counts[b];
    }

    /* Budget check: use actual tracked VRAM allocation (includes VB, IB, materials,
     * framebuffers, swapchain, etc.) for accurate remaining-memory calculation. */
    VkDeviceSize heap_size = rt_device_local_heap_size(gpu);
    VkDeviceSize already_used = gpu->total_allocated_gpu_mem;
    uint64_t estimated_compact = (uint64_t)(total_blas_bytes * 0.35);

    fprintf(stderr, "gpu_vulkan: RT budget: VRAM %.0f MB, already used %.0f MB, BLAS %.0f MB (pre), ~%.0f MB (est. compact), scratch %.0f MB\n",
            (double)heap_size / (1024.0*1024.0), (double)already_used / (1024.0*1024.0),
            (double)total_blas_bytes / (1024.0*1024.0), (double)estimated_compact / (1024.0*1024.0),
            (double)max_scratch_size / (1024.0*1024.0));

    /* ---- Pooled BLAS build ----
     * ONE allocation for all BLAS (pool buffer), batched builds, optional pooled compaction.
     * Total vkAllocateMemory: 2 (pool + scratch) or 3 (+ compact pool).
     * Total vkQueueSubmit: O(nblas / BATCH_SIZE).
     *
     * If the full pool exceeds VRAM budget, build a visible subset.  The old
     * ascending-size policy maximized instance count per byte, but on DSX it
     * omitted a handful of huge terrain/road/site BLASes that dominate the
     * image.  The hybrid order below reserves part of the budget for the
     * largest visible BLASes, then fills the remainder with small meshes. */
    #define BLAS_BATCH_SIZE 32
    #define BLAS_ALIGN 256  /* AS offset alignment requirement */

    RtBlasCandidate* small_order = (RtBlasCandidate*)calloc(nblas, sizeof(RtBlasCandidate));
    RtBlasCandidate* large_order = (RtBlasCandidate*)calloc(nblas, sizeof(RtBlasCandidate));
    if (!small_order || !large_order) {
        free(small_order); free(large_order);
        free(prim_counts); free(blas_sizes); free(mesh_to_blas);
        return 0;
    }

    for (uint32_t m = 0; m < nmeshes; m++) {
        if ((uint32_t)meshes[m].prototype_idx != m) continue;
        uint32_t b = mesh_to_blas[m];
        small_order[b].blas      = b;
        small_order[b].proto_mesh = m;
        small_order[b].source_id = meshes[m].source_id;
        small_order[b].triangles = prim_counts[b];
        small_order[b].size      = blas_sizes[b];
        small_order[b].name      = meshes[m].debug_name;
    }
    for (uint32_t m = 0; m < nmeshes; m++) {
        if (meshes[m].visible)
            small_order[mesh_to_blas[m]].visible_instances++;
    }
    memcpy(large_order, small_order, nblas * sizeof(RtBlasCandidate));
    qsort(small_order, nblas, sizeof(RtBlasCandidate), rt_blas_candidate_small_first);
    qsort(large_order, nblas, sizeof(RtBlasCandidate), rt_blas_candidate_large_first);
    uint32_t visible_blas_count = 0;
    uint32_t visible_instance_count = 0;
    for (uint32_t b = 0; b < nblas; b++) {
        if (small_order[b].visible_instances > 0) {
            visible_blas_count++;
            visible_instance_count += small_order[b].visible_instances;
        }
    }

    /* Budget for the pool: total VRAM minus what's already allocated, minus scratch
     * that must be co-resident, minus 256 MB safety margin for driver overhead and
     * hidden allocations not tracked by our allocator. */
    VkDeviceSize safety_margin = 256ULL * 1024 * 1024;
    VkDeviceSize pool_budget = (already_used + max_scratch_size + safety_margin < heap_size)
        ? heap_size - already_used - max_scratch_size - safety_margin : 0;

    /* Compute pool layout: walk BLASes in sorted order, include those that fit.
     * If the pool allocation fails (VRAM fragmentation), halve the budget and retry.
     * This binary-searches for the largest pool the driver can actually allocate. */
    if (!blas_included)
        blas_included = (uint8_t*)calloc(nblas, sizeof(uint8_t));
    uint64_t* blas_offsets = (uint64_t*)calloc(nblas, sizeof(uint64_t));
    if (!blas_included || !blas_offsets) {
        free(small_order); free(large_order);
        free(blas_included); free(blas_preloaded); free(blas_offsets);
        free(prim_counts); free(blas_sizes); free(mesh_to_blas);
        return 0;
    }
    uint64_t pool_size = 0;
    nblas_included = nblas_preloaded;

    VkBuffer pool_buf = VK_NULL_HANDLE;
    VkDeviceMemory pool_mem = VK_NULL_HANDLE;
    VkBuffer scratch_buf = VK_NULL_HANDLE;
    VkDeviceMemory scratch_mem = VK_NULL_HANDLE;
    VkDeviceSize post_build_reserve = 256ULL * 1024ULL * 1024ULL;
    {
        const char* reserve_env = getenv("NUSD_RT_POST_BUILD_RESERVE_MB");
        if (reserve_env && reserve_env[0]) {
            char* end = NULL;
            unsigned long mb = strtoul(reserve_env, &end, 10);
            if (end != reserve_env && mb <= 4096UL)
                post_build_reserve = (VkDeviceSize)mb * 1024ULL * 1024ULL;
        }
    }
    int visual_priority = rt_env_enabled_default_off("NUSD_RT_VISUAL_PRIORITY");
    int require_all_visible =
        rt_env_enabled_default_off("NUSD_RT_REQUIRE_ALL_VISIBLE") ||
        rt_env_enabled_default_off("NUSD_RT_CAMERA_RESIDENCY");
    uint32_t large_budget_pct = rt_env_u32("NUSD_RT_LARGE_BLAS_BUDGET_PCT", 35u, 95u);
    uint32_t excluded_diag_limit = rt_env_u32("NUSD_RT_EXCLUDED_DIAG", 0u, 200u);

    for (int attempt = 0; attempt < 48; attempt++) {
        /* Reset layout */
        if (blas_preloaded)
            memcpy(blas_included, blas_preloaded, nblas);
        else
            memset(blas_included, 0, nblas * sizeof(uint8_t));
        memset(blas_offsets, 0, nblas * sizeof(uint64_t));
        pool_size = 0;
        nblas_included = nblas_preloaded;

        uint64_t priority_pool_size = 0;
        uint64_t priority_budget = ((uint64_t)pool_budget * (uint64_t)large_budget_pct) / 100u;
        if (visual_priority && large_budget_pct > 0) {
            for (uint32_t i = 0; i < nblas; i++) {
                const RtBlasCandidate* c = &large_order[i];
                if (c->visible_instances == 0) continue;
                if (blas_included[c->blas]) continue;
                uint64_t aligned_sz = rt_align_u64((uint64_t)c->size, BLAS_ALIGN);
                if (pool_size + aligned_sz > pool_budget) continue;
                if (priority_pool_size + aligned_sz > priority_budget && priority_pool_size > 0)
                    continue;
                rt_include_blas(c->blas, c->size, blas_included, blas_offsets,
                                &pool_size, &nblas_included);
                priority_pool_size += aligned_sz;
            }
        }

        for (uint32_t i = 0; i < nblas; i++) {
            const RtBlasCandidate* c = &small_order[i];
            if (c->visible_instances == 0) continue;
            if (blas_included[c->blas]) continue;
            uint64_t aligned_sz = rt_align_u64((uint64_t)c->size, BLAS_ALIGN);
            if (pool_size + aligned_sz > pool_budget && pool_size > 0)
                break;
            if (pool_size + aligned_sz > pool_budget)
                continue;
            rt_include_blas(c->blas, c->size, blas_included, blas_offsets,
                            &pool_size, &nblas_included);
        }

        if (pool_size == 0 && nblas_included >= visible_blas_count && !has_curves) {
            fprintf(stderr,
                    "gpu_vulkan: BLAS cache satisfied all %u visible BLASes; skipping BLAS build\n",
                    visible_blas_count);
            free(small_order); small_order = NULL;
            free(large_order); large_order = NULL;
            free(blas_offsets); blas_offsets = NULL;
            free(prim_counts); prim_counts = NULL;
            free(blas_sizes); blas_sizes = NULL;
            goto build_tlas_phase;
        }

        if (nblas_included == 0 && !has_curves) {
            fprintf(stderr, "gpu_vulkan: no BLASes fit within VRAM budget (%.0f MB) — skipping RT\n",
                    (double)pool_budget / (1024.0*1024.0));
            rt_write_omitted_blas_manifest("no_visible_blas_fit",
                                           large_order, nblas, blas_included,
                                           meshes, nblas_included,
                                           visible_blas_count,
                                           &(RtOmitManifestDiag){
                                               .pool_budget = pool_budget,
                                               .attempted_pool_size = pool_size,
                                               .scratch_size = max_scratch_size,
                                               .post_build_reserve = post_build_reserve,
                                               .failed_allocation = "visible_blas_budget",
                                               .failed_allocation_size = 0,
                                               .build_started = 0,
                                               .strict_no_drop = require_all_visible,
                                           });
            free(small_order); free(large_order); free(blas_included); free(blas_preloaded); free(blas_offsets);
            free(prim_counts); free(blas_sizes); free(mesh_to_blas);
            return 0;
        }
        /* Phase 11.A.2.5: nblas_included == 0 + has_curves means a curve-only
         * scene; fall through to TLAS construction with just the curve BLAS
         * instance. Skip the mesh BLAS pool alloc (size == 0). */
        if (nblas_included == 0 && has_curves) {
            pool_buf = VK_NULL_HANDLE;
            pool_mem = VK_NULL_HANDLE;
            pool_size = 0;
            break;  /* exit the do-while retry loop */
        }

        if (nblas_included < nblas) {
            uint32_t omitted_visible = 0;
            uint32_t omitted_instances = 0;
            for (uint32_t i = 0; i < nblas; i++) {
                const RtBlasCandidate* c = &large_order[i];
                if (c->visible_instances == 0 || blas_included[c->blas]) continue;
                omitted_visible++;
                omitted_instances += c->visible_instances;
            }
            fprintf(stderr,
                    "gpu_vulkan: partial RT: building %u/%u BLASes (%.1f MB pool, large_priority=%s %u%%, omitted_visible=%u, omitted_instances=%u)\n",
                    nblas_included, nblas, (double)pool_size / (1024.0*1024.0),
                    visual_priority ? "on" : "off", large_budget_pct,
                    omitted_visible, omitted_instances);
        }
        if (require_all_visible && nblas_included < visible_blas_count) {
            fprintf(stderr,
                    "gpu_vulkan: refusing partial RT build: selected %u/%u visible BLASes "
                    "(%u visible instances); set NUSD_RT_REQUIRE_ALL_VISIBLE=0 only for "
                    "diagnostics that tolerate missing geometry\n",
                    nblas_included, visible_blas_count, visible_instance_count);
            rt_write_omitted_blas_manifest("visible_blas_budget_exceeded",
                                           large_order, nblas, blas_included,
                                           meshes, nblas_included,
                                           visible_blas_count,
                                           &(RtOmitManifestDiag){
                                               .pool_budget = pool_budget,
                                               .attempted_pool_size = pool_size,
                                               .scratch_size = max_scratch_size,
                                               .post_build_reserve = post_build_reserve,
                                               .failed_allocation = "visible_blas_budget",
                                               .failed_allocation_size = 0,
                                               .build_started = 0,
                                               .strict_no_drop = require_all_visible,
                                           });
            free(small_order); free(large_order); free(blas_included); free(blas_preloaded); free(blas_offsets);
            free(prim_counts); free(blas_sizes); free(mesh_to_blas);
            return 0;
        }

        /* Try to allocate the pool buffer.
         *
         * perf/mem-pool: NOT routed through the resident pool because the
         * compaction path destroys this BLAS pool mid-build and reallocates
         * a smaller compact_buf — sub-allocating from a bump pool would
         * leave this range stranded (cursor doesn't go backward). On
         * curve-only scenes this is small (often zero) so a private alloc
         * is fine; on mesh-heavy scenes we'd want a separate BLAS-pool
         * slab eventually. */
        if (rt_create_buffer(gpu, pool_size,
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &pool_buf, &pool_mem)) {
            /* Allocate ONE shared scratch buffer before accepting this BLAS
             * pool size.  DSX can allocate a 9+ GB BLAS pool and then fail a
             * tiny required scratch allocation; that is not a render-fatal
             * condition, it means the BLAS pool must be trimmed further. */
            if (rt_create_buffer_pooled(gpu, max_scratch_size,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         RT_POOL_NONE,
                         &scratch_buf, &scratch_mem)) {
                if (post_build_reserve == 0) {
                    break;  /* success: BLAS pool + scratch coexist */
                }

                /* Leave explicit headroom for the allocations that happen
                 * after BLAS build succeeds: TLAS storage/scratch, storage
                 * image, G-buffer, scene-data SSBO, descriptors/raycast, and
                 * driver bookkeeping. DSX can find the first allocatable BLAS
                 * pool, build it, then fail a tiny scene_data SSBO. Treat
                 * that as a too-large BLAS pool and keep trimming. */
                VkBuffer reserve_buf = VK_NULL_HANDLE;
                VkDeviceMemory reserve_mem = VK_NULL_HANDLE;
                if (rt_create_buffer_pooled_untracked(gpu, post_build_reserve,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             RT_POOL_NONE,
                             &reserve_buf, &reserve_mem)) {
                    vkDestroyBuffer(gpu->device, reserve_buf, NULL);
                    if (reserve_mem) vkFreeMemory(gpu->device, reserve_mem, NULL);
                    break;  /* success: BLAS pool + scratch + headroom coexist */
                }

                fprintf(stderr,
                        "gpu_vulkan: post-build reserve alloc failed (%.1f MB) with %.1f MB BLAS pool, reducing BLAS budget\n",
                        (double)post_build_reserve / (1024.0*1024.0),
                        (double)pool_size / (1024.0*1024.0));
                vkDestroyBuffer(gpu->device, scratch_buf, NULL);
                if (scratch_mem) vkFreeMemory(gpu->device, scratch_mem, NULL);
                scratch_buf = VK_NULL_HANDLE;
                scratch_mem = VK_NULL_HANDLE;
                vkDestroyBuffer(gpu->device, pool_buf, NULL);
                if (pool_mem) vkFreeMemory(gpu->device, pool_mem, NULL);
                pool_buf = VK_NULL_HANDLE;
                pool_mem = VK_NULL_HANDLE;

                if (require_all_visible) {
                    fprintf(stderr,
                            "gpu_vulkan: refusing to trim BLAS pool after post-build reserve failure "
                            "because all selected visible geometry is required\n");
                    rt_write_omitted_blas_manifest("post_build_reserve_allocation_failed",
                                                   large_order, nblas, blas_included,
                                                   meshes, nblas_included,
                                                   visible_blas_count,
                                                   &(RtOmitManifestDiag){
                                                       .pool_budget = pool_budget,
                                                       .attempted_pool_size = pool_size,
                                                       .scratch_size = max_scratch_size,
                                                       .post_build_reserve = post_build_reserve,
                                                       .failed_allocation = "post_build_reserve",
                                                       .failed_allocation_size = post_build_reserve,
                                                       .build_started = 0,
                                                       .strict_no_drop = require_all_visible,
                                                   });
                    free(small_order); free(large_order); free(blas_included); free(blas_preloaded); free(blas_offsets);
                    free(prim_counts); free(blas_sizes); free(mesh_to_blas);
                    return 0;
                }

                VkDeviceSize step = pool_size / 20;  /* capped 5% of rejected alloc */
                const VkDeviceSize min_step = 64ULL * 1024ULL * 1024ULL;
                const VkDeviceSize max_step = 256ULL * 1024ULL * 1024ULL;
                if (step < min_step) step = min_step;
                if (step > max_step) step = max_step;
                VkDeviceSize new_budget =
                    (pool_size > step) ? (VkDeviceSize)(pool_size - step)
                                       : (VkDeviceSize)(pool_budget / 2);
                fprintf(stderr, "gpu_vulkan: retrying with budget %.0f MB\n",
                        (double)new_budget / (1024.0*1024.0));
                pool_budget = new_budget;
                if (pool_budget < 16ULL * 1024 * 1024) {
                    fprintf(stderr, "gpu_vulkan: cannot reserve BLAS pool plus post-build headroom — skipping RT\n");
                    free(small_order); free(large_order); free(blas_included); free(blas_preloaded); free(blas_offsets);
                    free(prim_counts); free(blas_sizes); free(mesh_to_blas);
                    return 0;
                }
                continue;
            }

            fprintf(stderr,
                    "gpu_vulkan: BLAS scratch alloc failed (%.1f MB) with %.1f MB pool, reducing BLAS budget\n",
                    (double)max_scratch_size / (1024.0*1024.0),
                    (double)pool_size / (1024.0*1024.0));
            vkDestroyBuffer(gpu->device, pool_buf, NULL);
            if (pool_mem) vkFreeMemory(gpu->device, pool_mem, NULL);
            pool_buf = VK_NULL_HANDLE;
            pool_mem = VK_NULL_HANDLE;

            if (require_all_visible) {
                fprintf(stderr,
                        "gpu_vulkan: refusing to trim BLAS pool after scratch allocation failure "
                        "because all selected visible geometry is required\n");
                rt_write_omitted_blas_manifest("scratch_allocation_failed",
                                               large_order, nblas, blas_included,
                                               meshes, nblas_included,
                                               visible_blas_count,
                                               &(RtOmitManifestDiag){
                                                   .pool_budget = pool_budget,
                                                   .attempted_pool_size = pool_size,
                                                   .scratch_size = max_scratch_size,
                                                   .post_build_reserve = post_build_reserve,
                                                   .failed_allocation = "blas_scratch",
                                                   .failed_allocation_size = max_scratch_size,
                                                   .build_started = 0,
                                                   .strict_no_drop = require_all_visible,
                                               });
                free(small_order); free(large_order); free(blas_included); free(blas_preloaded); free(blas_offsets);
                free(prim_counts); free(blas_sizes); free(mesh_to_blas);
                return 0;
            }

            VkDeviceSize step = pool_size / 20;  /* capped 5% of rejected alloc */
            const VkDeviceSize min_step = 64ULL * 1024ULL * 1024ULL;
            const VkDeviceSize max_step = 256ULL * 1024ULL * 1024ULL;
            if (step < min_step) step = min_step;
            if (step > max_step) step = max_step;
            VkDeviceSize new_budget =
                (pool_size > step) ? (VkDeviceSize)(pool_size - step)
                                   : (VkDeviceSize)(pool_budget / 2);
            fprintf(stderr, "gpu_vulkan: retrying with budget %.0f MB\n",
                    (double)new_budget / (1024.0*1024.0));
            pool_budget = new_budget;
            if (pool_budget < 16ULL * 1024 * 1024) {
                fprintf(stderr, "gpu_vulkan: cannot reserve BLAS pool plus scratch — skipping RT\n");
                free(small_order); free(large_order); free(blas_included); free(blas_preloaded); free(blas_offsets);
                free(prim_counts); free(blas_sizes); free(mesh_to_blas);
                return 0;
            }
            continue;
        }

        /* Alloc failed.  Do not immediately halve the pool: DSX sits right
         * on a driver/fragmentation cliff where a 10.58 GB BLAS pool can
         * fail while a 10.57 GB pool succeeds.  A single 50% drop excludes
         * thousands of otherwise renderable BLASes from the TLAS, producing
         * visible missing geometry.  Trim in small steps first, then fall
         * back harder if the driver still cannot provide a contiguous block. */
        VkDeviceSize step = pool_size / 20;  /* capped 5% of rejected alloc */
        const VkDeviceSize min_step = 64ULL * 1024ULL * 1024ULL;
        const VkDeviceSize max_step = 256ULL * 1024ULL * 1024ULL;
        if (step < min_step) step = min_step;
        if (step > max_step) step = max_step;
        VkDeviceSize new_budget =
            (pool_size > step) ? (VkDeviceSize)(pool_size - step)
                               : (VkDeviceSize)(pool_budget / 2);
        fprintf(stderr, "gpu_vulkan: BLAS pool alloc failed (%.1f MB), retrying with budget %.0f MB\n",
                (double)pool_size / (1024.0*1024.0), (double)new_budget / (1024.0*1024.0));
        if (require_all_visible) {
            fprintf(stderr,
                    "gpu_vulkan: refusing to trim BLAS pool after allocation failure "
                    "because all selected visible geometry is required\n");
            rt_write_omitted_blas_manifest("blas_pool_allocation_failed",
                                           large_order, nblas, blas_included,
                                           meshes, nblas_included,
                                           visible_blas_count,
                                           &(RtOmitManifestDiag){
                                               .pool_budget = pool_budget,
                                               .attempted_pool_size = pool_size,
                                               .scratch_size = max_scratch_size,
                                               .post_build_reserve = post_build_reserve,
                                               .failed_allocation = "blas_pool",
                                               .failed_allocation_size = pool_size,
                                               .build_started = 0,
                                               .strict_no_drop = require_all_visible,
                                           });
            free(small_order); free(large_order); free(blas_included); free(blas_preloaded); free(blas_offsets);
            free(prim_counts); free(blas_sizes); free(mesh_to_blas);
            return 0;
        }
        pool_budget = new_budget;
        if (pool_budget < 16ULL * 1024 * 1024) {  /* 16 MB minimum */
            fprintf(stderr, "gpu_vulkan: cannot allocate any BLAS pool — skipping RT\n");
            free(small_order); free(large_order); free(blas_included); free(blas_preloaded); free(blas_offsets);
            free(prim_counts); free(blas_sizes); free(mesh_to_blas);
            return 0;
        }
    }

    if ((pool_buf == VK_NULL_HANDLE || scratch_buf == VK_NULL_HANDLE) && !has_curves) {
        fprintf(stderr, "gpu_vulkan: BLAS pool alloc failed after retries — skipping RT\n");
        free(small_order); free(large_order); free(blas_included); free(blas_preloaded); free(blas_offsets);
        free(prim_counts); free(blas_sizes); free(mesh_to_blas);
        return 0;
    }
    /* Phase 11.A.2.5: pool_buf == NULL is legitimate when has_curves and
     * nblas_included == 0. Skip the entire mesh-AS-handle / build / scratch
     * / compaction block by short-circuiting to the post-mesh cleanup. */
    if (nblas_included == 0 && has_curves) {
        gpu->blas_count = 0;
        gpu->blas_list  = NULL;
        free(small_order); small_order = NULL;
        free(large_order); large_order = NULL;
        free(blas_offsets); blas_offsets = NULL;
        free(prim_counts); prim_counts = NULL;
        free(blas_sizes); blas_sizes = NULL;
        /* mesh_to_blas + blas_included kept until the late free() at
         * line ~3181 (the for loops that read them iterate nmeshes==0
         * times so they're not actually dereferenced). */
        goto build_tlas_phase;  /* skip mesh BLAS build + compaction */
    }

    /* Create AS handles at offsets within the pool buffer (included BLASes only) */
    gpu->blas_count = nblas;
    if (!gpu->blas_list)
        gpu->blas_list  = (VkAccelerationStructureKHR*)calloc(nblas, sizeof(VkAccelerationStructureKHR));
    if (!gpu->blas_list) {
        free(small_order); free(large_order); free(blas_included); free(blas_preloaded); free(blas_offsets);
        free(prim_counts); free(blas_sizes); free(mesh_to_blas);
        return 0;
    }
    for (uint32_t b = 0; b < nblas; b++) {
        if (!blas_included[b]) continue;
        if (blas_preloaded && blas_preloaded[b]) continue;
        VkAccelerationStructureCreateInfoKHR as_ci = {0};
        as_ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        as_ci.buffer = pool_buf;
        as_ci.offset = blas_offsets[b];
        as_ci.size   = blas_sizes[b];
        as_ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        vkCreateAccelerationStructureKHR(gpu->device, &as_ci, NULL, &gpu->blas_list[b]);
    }

    VkDeviceAddress scratch_addr = rt_buf_addr(gpu, scratch_buf);

    /* Collect unique prototype mesh indices (only included BLASes) */
    uint32_t* proto_meshes = (uint32_t*)malloc(nblas * sizeof(uint32_t));
    uint32_t n_protos = 0;
    for (uint32_t m = 0; m < nmeshes; m++) {
        uint32_t b = mesh_to_blas[m];
        if ((uint32_t)meshes[m].prototype_idx == m && blas_included[b] &&
            !(blas_preloaded && blas_preloaded[b]))
            proto_meshes[n_protos++] = m;
    }
    if (n_protos == 0) {
        free(small_order); small_order = NULL;
        free(large_order); large_order = NULL;
        free(proto_meshes); proto_meshes = NULL;
        free(blas_offsets); blas_offsets = NULL;
        free(prim_counts); prim_counts = NULL;
        free(blas_sizes); blas_sizes = NULL;
        goto build_tlas_phase;
    }

    /* Single query pool for compaction sizes — only need slots for included BLASes.
     * Query slot index = BLAS index b (sparse: unincluded slots are unused). */
    uint32_t batch_cap = n_protos < BLAS_BATCH_SIZE ? n_protos : BLAS_BATCH_SIZE;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    {
        VkQueryPoolCreateInfo qp_ci = {0};
        qp_ci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qp_ci.queryType   = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        qp_ci.queryCount  = nblas;  /* sparse: index by BLAS index */
        vkCreateQueryPool(gpu->device, &qp_ci, NULL, &query_pool);
        vkResetQueryPool(gpu->device, query_pool, 0, nblas);
    }

    /* Phase 1: Build all BLAS + write compaction queries in batched command buffers.
     * Use small batches (32) to avoid GPU stalls under memory pressure.
     * Large command buffers with 100+ serialized BLAS builds can hang on
     * memory-constrained GPUs because the driver can't reclaim resources
     * between builds within a single submission. */
    struct timespec blas_start_time;
    clock_gettime(CLOCK_MONOTONIC, &blas_start_time);
    uint32_t blas_time_limit_sec = require_all_visible ? 0u : 120u;
    {
        const char* limit_env = getenv("NUSD_RT_BLAS_TIME_LIMIT_SEC");
        if (limit_env && limit_env[0]) {
            char* end = NULL;
            unsigned long v = strtoul(limit_env, &end, 10);
            if (end != limit_env && v <= 86400UL)
                blas_time_limit_sec = (uint32_t)v;
        }
    }

    /* perf/vk-instrumentation: bracket the entire BLAS build with a single
     * timestamped + nsys-labelled region. Begin lives in the first batch's
     * cmd buffer, end in the last. The timestamp pool is device-scoped so
     * this works across multiple submissions. */
    int blas_label_open = 0;

    for (uint32_t batch_start = 0; batch_start < n_protos; batch_start += batch_cap) {
        uint32_t batch_end = batch_start + batch_cap;
        if (batch_end > n_protos) batch_end = n_protos;
        uint32_t batch_count = batch_end - batch_start;
        int is_first_batch = (batch_start == 0);
        int is_last_batch  = (batch_end == n_protos);

        VkCommandBuffer cmd = rt_begin_cmd(gpu);

        if (is_first_batch) {
            gpu_phase_begin(gpu, cmd, GPU_PHASE_BLAS_BUILD, "BLAS_build");
            blas_label_open = 1;
        }

        for (uint32_t i = 0; i < batch_count; i++) {
            uint32_t m = proto_meshes[batch_start + i];
            uint32_t b = mesh_to_blas[m];
            const GpuRtMeshDesc* md = &meshes[m];
            VkDeviceAddress mesh_vb = vb_addr + (VkDeviceAddress)md->vertex_offset * md->vertex_stride;
            VkDeviceAddress mesh_ib = ib_addr + (VkDeviceAddress)md->index_offset * sizeof(uint32_t);

            VkAccelerationStructureGeometryTrianglesDataKHR triangles = {0};
            triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
            triangles.vertexData.deviceAddress = mesh_vb;
            triangles.vertexStride             = md->vertex_stride;
            triangles.maxVertex                = md->vertex_count - 1;
            triangles.indexType                = VK_INDEX_TYPE_UINT32;
            triangles.indexData.deviceAddress  = mesh_ib;

            VkAccelerationStructureGeometryKHR geom = {0};
            geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geom.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
            geom.geometry.triangles = triangles;

            VkAccelerationStructureBuildGeometryInfoKHR build_info = {0};
            build_info.sType                     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            build_info.type                      = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            build_info.flags                     = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                                   VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
            build_info.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            build_info.geometryCount             = 1;
            build_info.pGeometries               = &geom;
            build_info.dstAccelerationStructure  = gpu->blas_list[b];
            build_info.scratchData.deviceAddress = scratch_addr;

            VkAccelerationStructureBuildRangeInfoKHR range = {0};
            range.primitiveCount = prim_counts[b];
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &pRange);

            /* Barrier: build must complete before query + next scratch reuse */
            VkMemoryBarrier mb = {0};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                               VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                 VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                 0, 1, &mb, 0, NULL, 0, NULL);

            /* Write compaction size query inline (after barrier ensures build is done) */
            vkCmdWriteAccelerationStructuresPropertiesKHR(cmd, 1, &gpu->blas_list[b],
                VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, query_pool, b);
        }

        /* perf/vk-instrumentation: pre-check time-limit so we can close the
         * region in this cmd buffer if it'll be the final one. */
        int time_limited_break = 0;
        if (!is_last_batch && blas_time_limit_sec > 0) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed_pre = (now.tv_sec - blas_start_time.tv_sec) +
                                 (now.tv_nsec - blas_start_time.tv_nsec) * 1e-9;
            if (elapsed_pre > blas_time_limit_sec) time_limited_break = 1;
        }
        if ((is_last_batch || time_limited_break) && blas_label_open) {
            gpu_phase_end(gpu, cmd, GPU_PHASE_BLAS_BUILD);
            blas_label_open = 0;
        }

        rt_end_cmd(gpu, cmd);  /* Single submit for entire batch */

        if (n_protos > batch_cap) {
            fprintf(stderr, "gpu_vulkan: BLAS build progress: %u/%u\n",
                    batch_start + batch_count, n_protos);
        }

        if (time_limited_break) {
            uint32_t built = batch_start + batch_count;
            fprintf(stderr, "gpu_vulkan: BLAS time limit (%us) reached after %u/%u — stopping\n",
                    blas_time_limit_sec, built, n_protos);
            /* Mark remaining BLASes as not included */
            for (uint32_t j = built; j < n_protos; j++) {
                uint32_t mm = proto_meshes[j];
                uint32_t bb = mesh_to_blas[mm];
                if (gpu->blas_list[bb]) {
                    vkDestroyAccelerationStructureKHR(gpu->device, gpu->blas_list[bb], NULL);
                    gpu->blas_list[bb] = VK_NULL_HANDLE;
                }
                blas_included[bb] = 0;
                nblas_included = nblas_preloaded + built;
            }
            n_protos = built;
            break;
        }
    }
    /* perf/vk-instrumentation: timestamps land after rt_end_cmd's queue
     * wait. Resolve into phase_timing_ns. No-op when timestamps are off
     * or when n_protos == 0 (label was never opened). */
    if (gpu->timestamps_supported) gpu_phase_resolve(gpu, GPU_PHASE_BLAS_BUILD);

    /* Phase 2: Read compaction sizes — only for included BLASes.
     * Non-included BLASes never wrote queries, so reading their slots with
     * VK_QUERY_RESULT_WAIT_BIT would block forever. */
    VkDeviceSize* compact_sizes = (VkDeviceSize*)calloc(nblas, sizeof(VkDeviceSize));
    for (uint32_t b = 0; b < nblas; b++) {
        if (!blas_included[b]) continue;
        if (blas_preloaded && blas_preloaded[b]) continue;
        vkGetQueryPoolResults(gpu->device, query_pool, b, 1,
            sizeof(VkDeviceSize), &compact_sizes[b], sizeof(VkDeviceSize),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    }

    /* Compute compact pool layout (included BLASes only) */
    uint64_t* compact_offsets = (uint64_t*)calloc(nblas, sizeof(uint64_t));
    uint64_t compact_pool_size = 0;
    for (uint32_t b = 0; b < nblas; b++) {
        if (!blas_included[b]) continue;
        if (blas_preloaded && blas_preloaded[b]) continue;
        compact_offsets[b] = compact_pool_size;
        VkDeviceSize sz = compact_sizes[b] > 0 ? compact_sizes[b] : blas_sizes[b];
        compact_pool_size += (sz + BLAS_ALIGN - 1) & ~(uint64_t)(BLAS_ALIGN - 1);
    }

    /* Only compact if we save at least 10% */
    int do_compact = compact_pool_size < (uint64_t)(pool_size * 0.9);

    if (do_compact) {
        VkBuffer compact_buf = VK_NULL_HANDLE;
        VkDeviceMemory compact_mem = VK_NULL_HANDLE;
        if (rt_create_buffer(gpu, compact_pool_size,
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &compact_buf, &compact_mem)) {

            /* Create compact AS handles and copy (included BLASes only) */
            VkAccelerationStructureKHR* compact_as = (VkAccelerationStructureKHR*)calloc(nblas, sizeof(VkAccelerationStructureKHR));
            for (uint32_t b = 0; b < nblas; b++) {
                if (!blas_included[b]) continue;
                if (blas_preloaded && blas_preloaded[b]) continue;
                VkDeviceSize sz = compact_sizes[b] > 0 ? compact_sizes[b] : blas_sizes[b];
                VkAccelerationStructureCreateInfoKHR ci = {0};
                ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
                ci.buffer = compact_buf;
                ci.offset = compact_offsets[b];
                ci.size   = sz;
                ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                vkCreateAccelerationStructureKHR(gpu->device, &ci, NULL, &compact_as[b]);
            }

            /* Copy all compactions in batched command buffers (included only) */
            for (uint32_t batch_start = 0; batch_start < nblas; batch_start += batch_cap) {
                uint32_t batch_end = batch_start + batch_cap;
                if (batch_end > nblas) batch_end = nblas;

                VkCommandBuffer cmd = rt_begin_cmd(gpu);
                for (uint32_t b = batch_start; b < batch_end; b++) {
                    if (!blas_included[b]) continue;
                    if (blas_preloaded && blas_preloaded[b]) continue;
                    VkCopyAccelerationStructureInfoKHR copy_info = {0};
                    copy_info.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
                    copy_info.src   = gpu->blas_list[b];
                    copy_info.dst   = compact_as[b];
                    copy_info.mode  = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
                    vkCmdCopyAccelerationStructureKHR(cmd, &copy_info);
                }
                rt_end_cmd(gpu, cmd);
            }

            /* Destroy old AS handles and pre-compaction pool */
            for (uint32_t b = 0; b < nblas; b++) {
                if (blas_preloaded && blas_preloaded[b]) continue;
                if (gpu->blas_list[b])
                    vkDestroyAccelerationStructureKHR(gpu->device, gpu->blas_list[b], NULL);
            }
            vkDestroyBuffer(gpu->device, pool_buf, NULL);
            vkFreeMemory(gpu->device, pool_mem, NULL);

            /* Swap to compacted */
            for (uint32_t b = 0; b < nblas; b++) {
                if (compact_as[b])
                    gpu->blas_list[b] = compact_as[b];
            }
            free(compact_as);
            if (blas_preloaded) {
                gpu->blas_extra_pool_buf = compact_buf;
                gpu->blas_extra_pool_mem = compact_mem;
            } else {
                gpu->blas_pool_buf = compact_buf;
                gpu->blas_pool_mem = compact_mem;
            }

            fprintf(stderr, "gpu_vulkan: %u BLAS compacted (%.1f MB → %.1f MB, saved %.0f%%)\n",
                    n_protos, (double)pool_size / (1024.0*1024.0),
                    (double)compact_pool_size / (1024.0*1024.0),
                    100.0 * (1.0 - (double)compact_pool_size / (double)pool_size));
        } else {
            /* Compact alloc failed, keep pre-compaction pool */
            if (blas_preloaded) {
                gpu->blas_extra_pool_buf = pool_buf;
                gpu->blas_extra_pool_mem = pool_mem;
            } else {
                gpu->blas_pool_buf = pool_buf;
                gpu->blas_pool_mem = pool_mem;
            }
        }
    } else {
        /* Compaction not worth it, keep pre-compaction pool */
        if (blas_preloaded) {
            gpu->blas_extra_pool_buf = pool_buf;
            gpu->blas_extra_pool_mem = pool_mem;
        } else {
            gpu->blas_pool_buf = pool_buf;
            gpu->blas_pool_mem = pool_mem;
        }
    }

    /* BLAS build+compact wall time — directly comparable to the Phase-B
     * "BLAS cache LOAD" self-timing (blas_start_time is captured just
     * before the batched build loop). */
    {
        struct timespec bt;
        clock_gettime(CLOCK_MONOTONIC, &bt);
        double build_secs = (bt.tv_sec - blas_start_time.tv_sec) +
                            (bt.tv_nsec - blas_start_time.tv_nsec) * 1e-9;
        fprintf(stderr, "gpu_vulkan: BLAS build+compact: %u BLAS in %.2fs\n",
                n_protos, build_secs);
    }

    /* Phase A: opt-in BLAS serialization cache write. Best-effort and
     * read-only w.r.t. the BLAS — the render is bit-identical whether or
     * not this runs. proto_meshes[0..n_protos) are exactly the BLAS that
     * were built (post time-limit truncation), all still live here. */
    {
        const char* bcw = getenv("NUSD_BLAS_CACHE_WRITE");
        if (bcw && bcw[0] && bcw[0] != '0' &&
            gpu->blas_cache_path[0] && n_protos > 0) {
            blas_cache_write(gpu, meshes, mesh_to_blas,
                             proto_meshes, n_protos, prim_counts);
        }
    }

    free(compact_sizes);
    free(compact_offsets);
    free(blas_offsets);
    if (nblas_included < visible_blas_count)
        rt_write_omitted_blas_manifest("visible_blas_omitted",
                                       large_order, nblas, blas_included,
                                       meshes, nblas_included,
                                       visible_blas_count,
                                       &(RtOmitManifestDiag){
                                           .pool_budget = pool_budget,
                                           .attempted_pool_size = pool_size,
                                           .scratch_size = max_scratch_size,
                                           .post_build_reserve = post_build_reserve,
                                           .failed_allocation = NULL,
                                           .failed_allocation_size = 0,
                                           .build_started = 1,
                                           .strict_no_drop = require_all_visible,
                                       });
    if (excluded_diag_limit && nblas_included < nblas)
        rt_dump_excluded_blas_diag(large_order, nblas, blas_included, excluded_diag_limit);
    free(small_order);
    free(large_order);
    free(proto_meshes);
    if (query_pool) vkDestroyQueryPool(gpu->device, query_pool, NULL);

    vkDestroyBuffer(gpu->device, scratch_buf, NULL);
    vkFreeMemory(gpu->device, scratch_mem, NULL);

    free(prim_counts);
    free(blas_sizes);

    #undef BLAS_BATCH_SIZE
    #undef BLAS_ALIGN

    fprintf(stderr, "gpu_vulkan: %u/%u BLAS built (%u triangles, %.1f MB, scratch %.1f MB)\n",
            nblas_included, nblas, total_triangles, (double)total_blas_bytes / (1024.0*1024.0),
            (double)max_scratch_size / (1024.0*1024.0));

build_tlas_phase:
    /* ---- TLAS: one instance per mesh with a built BLAS + (Phase 11.A.2.5)
     * one curve instance pointing at the AABB BLAS with sbtRecordOffset=1. */
    {
        /* Count instances with built BLASes */
        uint32_t tlas_instance_count = 0;
        for (uint32_t m = 0; m < nmeshes; m++) {
            if (meshes[m].visible && blas_included[mesh_to_blas[m]])
                tlas_instance_count++;
        }
        if (has_curves) tlas_instance_count++;

        VkDeviceSize inst_buf_size = tlas_instance_count * sizeof(VkAccelerationStructureInstanceKHR);
        VkAccelerationStructureInstanceKHR* instances =
            (VkAccelerationStructureInstanceKHR*)calloc(tlas_instance_count, sizeof(VkAccelerationStructureInstanceKHR));

        uint32_t inst_idx = 0;
        for (uint32_t m = 0; m < nmeshes; m++) {
            uint32_t b = mesh_to_blas[m];
            if (!meshes[m].visible) continue;
            if (!blas_included[b]) continue;

            VkAccelerationStructureDeviceAddressInfoKHR addr_info = {0};
            addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            addr_info.accelerationStructure = gpu->blas_list[b];
            VkDeviceAddress blas_ref = vkGetAccelerationStructureDeviceAddressKHR(gpu->device, &addr_info);

            /* World transform from descriptor (3x4 row-major VkTransformMatrixKHR format) */
            memcpy(&instances[inst_idx].transform, meshes[m].transform, sizeof(float) * 12);
            instances[inst_idx].instanceCustomIndex                    = m;  /* maps to SceneData SSBO */
            /* Phase A/C: per-mesh env mask if set (1 << (env_idx & 7)), else 0xFF.
             * The GPU-driven TLAS-update path (tlas_translate.comp.glsl) does
             * NOT re-write the mask byte — only transforms — so the value
             * authored here at build time is what's used at trace time. */
            {
                uint8_t env_mask = 0xFF;
                if (gpu->instance_mask && m < gpu->instance_mask_count) {
                    env_mask = gpu->instance_mask[m];
                }
                instances[inst_idx].mask = env_mask;
            }
            instances[inst_idx].instanceShaderBindingTableRecordOffset = 0;
            instances[inst_idx].flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            instances[inst_idx].accelerationStructureReference         = blas_ref;
            inst_idx++;
        }
        if (has_curves) {
            VkAccelerationStructureDeviceAddressInfoKHR addr_info = {0};
            addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            addr_info.accelerationStructure = gpu->curve_blas;
            VkDeviceAddress curve_blas_ref = vkGetAccelerationStructureDeviceAddressKHR(gpu->device, &addr_info);

            /* Identity transform — segments are world-space (Phase 11.A.2.5 prep). */
            float identity[12] = { 1,0,0,0,  0,1,0,0,  0,0,1,0 };
            memcpy(&instances[inst_idx].transform, identity, sizeof(identity));
            instances[inst_idx].instanceCustomIndex                    = 0xFFFFFFu;  /* sentinel: not a mesh */
            instances[inst_idx].mask                                   = 0xFF;
            instances[inst_idx].instanceShaderBindingTableRecordOffset = 1;  /* picks curve hit-group (offset 1 within hit region) */
            instances[inst_idx].flags                                  = 0;  /* AABB geometry — no triangle-face culling */
            instances[inst_idx].accelerationStructureReference         = curve_blas_ref;
            inst_idx++;
        }

        /* Upload instances via staging buffer.
         * perf/mem-pool: route through staging pool (HOST_VISIBLE,
         * persistent-mapped). Pool is reset at next build_accel start. */
        VkBuffer staging_buf = VK_NULL_HANDLE;
        VkDeviceMemory staging_mem = VK_NULL_HANDLE;
        void* staging_mapped = NULL;
        if (!rt_create_staging_buffer(gpu, inst_buf_size, &staging_buf,
                                       &staging_mem, &staging_mapped)) {
            fprintf(stderr, "gpu_vulkan: TLAS instance staging alloc failed\n");
            free(instances);
            free(mesh_to_blas); free(blas_included); free(blas_preloaded);
            gpu->rt_built = 1;
            gpu_destroy_rt_scene(gpu);
            return 0;
        }
        memcpy(staging_mapped, instances, inst_buf_size);
        /* `instances` is kept alive across the full TLAS build so Phase C
         * (gpu_build_partitioned_tlases below) can copy real transforms
         * from it. Freed after the partitioned build (or fall-through). */

        /* Instance data is small on mesh-heavy scenes; keep it private so a
         * large BLAS pool does not have to coexist with the default 4 GB
         * resident pool reservation. */
        if (!rt_create_buffer_pooled(gpu, inst_buf_size,
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                         | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                         | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         RT_POOL_NONE,
                         &gpu->instance_buf, &gpu->instance_mem)) {
            fprintf(stderr, "gpu_vulkan: instance buffer alloc failed (GPU OOM) — skipping RT\n");
            rt_destroy_staging_buffer(gpu, staging_buf, staging_mem, staging_mapped);
            free(mesh_to_blas); free(blas_included); free(blas_preloaded);
            gpu->rt_built = 1;
            gpu_destroy_rt_scene(gpu);
            return 0;
        }

        /* Single command buffer: copy instances + barrier + build TLAS */
        VkCommandBuffer cmd = rt_begin_cmd(gpu);

        /* perf/vk-instrumentation: bracket instance copy + TLAS build. */
        gpu_phase_begin(gpu, cmd, GPU_PHASE_TLAS_BUILD, "TLAS_build");

        VkBufferCopy copy = {0};
        copy.size = inst_buf_size;
        vkCmdCopyBuffer(cmd, staging_buf, gpu->instance_buf, 1, &copy);

        VkMemoryBarrier mb = {0};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &mb, 0, NULL, 0, NULL);

        /* Build TLAS (same command buffer as instance copy) */
        VkAccelerationStructureGeometryInstancesDataKHR instances_data = {0};
        instances_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        instances_data.data.deviceAddress = rt_buf_addr(gpu, gpu->instance_buf);

        VkAccelerationStructureGeometryKHR tlas_geom = {0};
        tlas_geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        tlas_geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlas_geom.geometry.instances = instances_data;

        VkAccelerationStructureBuildGeometryInfoKHR tlas_build = {0};
        tlas_build.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        tlas_build.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tlas_build.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                 | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        tlas_build.geometryCount = 1;
        tlas_build.pGeometries   = &tlas_geom;

        VkAccelerationStructureBuildSizesInfoKHR tlas_sizes = {0};
        tlas_sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkGetAccelerationStructureBuildSizesKHR(gpu->device,
                                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                &tlas_build, &tlas_instance_count, &tlas_sizes);

        /* TLAS storage is referenced per-frame but is small relative to the
         * BLAS pool; private allocation avoids reserving the 4 GB resident
         * pool on DSX-sized mesh scenes. */
        if (!rt_create_buffer_pooled(gpu, tlas_sizes.accelerationStructureSize,
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         RT_POOL_NONE,
                         &gpu->tlas_buf, &gpu->tlas_mem)) {
            fprintf(stderr, "gpu_vulkan: TLAS buffer alloc failed (GPU OOM) — skipping RT\n");
            rt_end_cmd(gpu, cmd);
            rt_destroy_staging_buffer(gpu, staging_buf, staging_mem, staging_mapped);
            free(mesh_to_blas); free(blas_included); free(blas_preloaded);
            gpu->rt_built = 1;
            gpu_destroy_rt_scene(gpu);
            return 0;
        }

        VkAccelerationStructureCreateInfoKHR tlas_ci = {0};
        tlas_ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        tlas_ci.buffer = gpu->tlas_buf;
        tlas_ci.size   = tlas_sizes.accelerationStructureSize;
        tlas_ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        vkCreateAccelerationStructureKHR(gpu->device, &tlas_ci, NULL, &gpu->tlas);

        /* TLAS build scratch is consumed inside this submit only and is small
         * for mesh TLASes, so keep it private for the same reason as mesh BLAS
         * scratch above. */
        VkBuffer tlas_scratch_buf;
        VkDeviceMemory tlas_scratch_mem;
        if (!rt_create_buffer_pooled(gpu, tlas_sizes.buildScratchSize,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         RT_POOL_NONE,
                         &tlas_scratch_buf, &tlas_scratch_mem)) {
            fprintf(stderr, "gpu_vulkan: TLAS scratch alloc failed (GPU OOM) — skipping RT\n");
            rt_end_cmd(gpu, cmd);
            rt_destroy_staging_buffer(gpu, staging_buf, staging_mem, staging_mapped);
            free(mesh_to_blas); free(blas_included); free(blas_preloaded);
            gpu->rt_built = 1;
            gpu_destroy_rt_scene(gpu);
            return 0;
        }

        tlas_build.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        tlas_build.dstAccelerationStructure  = gpu->tlas;
        tlas_build.scratchData.deviceAddress = rt_buf_addr(gpu, tlas_scratch_buf);

        VkAccelerationStructureBuildRangeInfoKHR tlas_range = {0};
        tlas_range.primitiveCount = tlas_instance_count;
        const VkAccelerationStructureBuildRangeInfoKHR* pTlasRange = &tlas_range;

        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlas_build, &pTlasRange);
        gpu_phase_end(gpu, cmd, GPU_PHASE_TLAS_BUILD);

        rt_end_cmd(gpu, cmd);
        if (gpu->timestamps_supported) gpu_phase_resolve(gpu, GPU_PHASE_TLAS_BUILD);

        rt_destroy_staging_buffer(gpu, staging_buf, staging_mem, staging_mapped);
        vkDestroyBuffer(gpu->device, tlas_scratch_buf, NULL);
        if (tlas_scratch_mem) vkFreeMemory(gpu->device, tlas_scratch_mem, NULL);

        fprintf(stderr, "gpu_vulkan: TLAS built (%u instances, %u/%u unique BLAS)\n",
                tlas_instance_count, nblas_included, nblas);

        /* Persist data needed for TLAS-only rebuilds (transform updates) */
        gpu->tlas_instance_count      = tlas_instance_count;
        gpu->tlas_size                = tlas_sizes.accelerationStructureSize;
        gpu->tlas_scratch_size        = tlas_sizes.buildScratchSize;
        gpu->tlas_update_scratch_size = tlas_sizes.updateScratchSize;

        /* Pre-allocate persistent scratch buffer for TLAS updates.
         * Use the larger of build/update scratch to handle both modes. */
        {
            VkDeviceSize scratch_sz = tlas_sizes.updateScratchSize;
            if (scratch_sz < tlas_sizes.buildScratchSize)
                scratch_sz = tlas_sizes.buildScratchSize;
            rt_create_buffer_pooled(gpu, scratch_sz,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             RT_POOL_NONE,
                             &gpu->tlas_update_scratch_buf,
                             &gpu->tlas_update_scratch_mem);
        }

        /* Pre-allocate persistent staging buffer for instance uploads.
         * perf/mem-pool: HOST_VISIBLE persistent across frames; the
         * staging pool is reset by gpu_destroy_rt_scene only after the
         * VkBuffer is already destroyed, so this is safe in the staging
         * pool. The pool stays persistently mapped, so we just compute
         * the sub-region pointer (no per-buffer vkMapMemory). */
        {
            VkDeviceSize staging_sz = tlas_instance_count * sizeof(VkAccelerationStructureInstanceKHR);
            VkBuffer buf = VK_NULL_HANDLE;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            void* mapped = NULL;
            if (rt_create_staging_buffer(gpu, staging_sz, &buf, &mem, &mapped)) {
                gpu->tlas_update_staging_buf    = buf;
                gpu->tlas_update_staging_mem    = mem; /* VK_NULL_HANDLE on pool path */
                gpu->tlas_update_staging_mapped = mapped;
                gpu->tlas_update_staging_size   = staging_sz;
            }
        }

        /* Store BLAS device addresses */
        gpu->blas_addresses = (VkDeviceAddress*)calloc(gpu->blas_count, sizeof(VkDeviceAddress));
        for (uint32_t b = 0; b < gpu->blas_count; b++) {
            if (gpu->blas_list[b]) {
                VkAccelerationStructureDeviceAddressInfoKHR ai = {0};
                ai.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
                ai.accelerationStructure = gpu->blas_list[b];
                gpu->blas_addresses[b] = vkGetAccelerationStructureDeviceAddressKHR(gpu->device, &ai);
            }
        }

        /* Phase 0 (opt-in NUSD_RT_CLUSTER): replace small prototypes' classic
         * BLAS reference with a cluster (RTX Mega Geometry) Cluster BLAS so the
         * existing TLAS + trace exercises the cluster path. Gated to meshes that
         * fit a single cluster (<= device per-cluster caps); larger meshes need
         * meshletization (a later phase) and stay classic. The classic BLAS is
         * still built but goes unreferenced for these protos. Backing buffers
         * are kept on gpu->cluster_as_* and freed in gpu_destroy_rt_scene. */
#if NUSD_HAS_VK_NV_MEGA_GEOMETRY
        if (gpu->cluster_as_available && getenv("NUSD_RT_CLUSTER")) {
            int n_cluster = 0;
            for (uint32_t m = 0; m < nmeshes; m++) {
                if ((uint32_t)meshes[m].prototype_idx != m) continue;  /* prototypes only */
                uint32_t b = mesh_to_blas[m];
                if (b >= gpu->blas_count) continue;
                uint32_t nv = meshes[m].vertex_count;
                uint32_t nt = meshes[m].index_count / 3;
                if (nv == 0 || nt == 0 || nv > 256 || nt > 256) continue;
                VkDeviceAddress va = vb_addr + (VkDeviceAddress)meshes[m].vertex_offset * meshes[m].vertex_stride;
                VkDeviceAddress ia = ib_addr + (VkDeviceAddress)meshes[m].index_offset * sizeof(uint32_t);
                VkBuffer kb[2]; VkDeviceMemory km[2];
                VkDeviceAddress cl = nu_build_cluster_blas_for_mesh(
                    gpu, va, ia, nv, nt, meshes[m].vertex_stride, kb, km);
                if (!cl) continue;
                gpu->blas_addresses[b] = cl;
                for (int k = 0; k < 2; k++) {
                    if (gpu->cluster_as_count >= gpu->cluster_as_cap) {
                        int nc = gpu->cluster_as_cap ? gpu->cluster_as_cap * 2 : 16;
                        gpu->cluster_as_bufs = (VkBuffer*)realloc(gpu->cluster_as_bufs, (size_t)nc * sizeof(VkBuffer));
                        gpu->cluster_as_mems = (VkDeviceMemory*)realloc(gpu->cluster_as_mems, (size_t)nc * sizeof(VkDeviceMemory));
                        gpu->cluster_as_cap = nc;
                    }
                    gpu->cluster_as_bufs[gpu->cluster_as_count] = kb[k];
                    gpu->cluster_as_mems[gpu->cluster_as_count] = km[k];
                    gpu->cluster_as_count++;
                }
                n_cluster++;
            }
            fprintf(stderr, "gpu_vulkan: NUSD_RT_CLUSTER — %d prototype BLAS(es) now cluster BLAS\n", n_cluster);
        }
#endif

        /* Store instance→BLAS mapping and custom indices */
        gpu->mesh_to_blas    = (uint32_t*)calloc(tlas_instance_count, sizeof(uint32_t));
        gpu->instance_custom = (uint32_t*)calloc(tlas_instance_count, sizeof(uint32_t));
        {
            uint32_t ii = 0;
            for (uint32_t m = 0; m < nmeshes; m++) {
                if (!meshes[m].visible) continue;
                if (!blas_included[mesh_to_blas[m]]) continue;
                gpu->mesh_to_blas[ii]    = mesh_to_blas[m];
                gpu->instance_custom[ii] = m;
                ii++;
            }
        }

        /* Phase C: build per-env TLAS array now that legacy TLAS data is
         * in place (blas_addresses + mesh_to_blas + instance_custom set
         * above) and `instances` is still alive — its transforms are the
         * source of truth for the partitioned per-env TLASes. */
        if (gpu->num_envs > 0 && gpu->mesh_to_env) {
            if (!gpu_build_partitioned_tlases(gpu, instances, tlas_instance_count)) {
                fprintf(stderr, "gpu_vulkan: partitioned TLAS build failed — "
                        "falling back to legacy single-TLAS path\n");
                /* Don't fail the whole RT build — partitioned path is optional. */
            }
        }
        free(instances);
    }

    free(mesh_to_blas);
    free(blas_included);
    free(blas_preloaded);
    mesh_to_blas = NULL;

    /* ---- Storage image ---- */
    rt_create_storage_image(gpu);

    /* ---- Phase A G-buffer (binding 17) ---- */
    gpu_ensure_gbuffer(gpu);

    /* ---- Scene data SSBO (header + per-mesh vertex/index addresses + color + per-instance material) ---- */
    {
        /* Layout matches SceneData in raytrace.rchit.glsl:
         *   uint     vertexStride;    (floats per vertex)
         *   uint     hasMaterials;
         *   float    envMipLevels;    (0 = no IBL, >0 = IBL mip count)
         *   float    envIntensity;    (USD DomeLight intensity for rmiss sky)
         *   vec4     domeColor;       (rgb + intensity, fast_mode flat sky/ambient)
         *   uint     upAxis;          (0=X, 1=Y, 2=Z)
         *   uint[3]  pad;
         *   MeshData meshes[];        (two uint64_t + vec4 color + uint tex_index +
         *                              uint source_id + uint material_id +
         *                              uint ptex_color_offset)
         */
        uint32_t vertex_stride_floats = (nmeshes > 0)
            ? meshes[0].vertex_stride / 4
            : 12;
        uint32_t has_materials = (gpu->mat_uploaded && gpu->mat_count > 0
                                  && !gpu->mat_only_placeholder) ? 1 : 0;
        float env_mip_levels = gpu->env_image_view ? (float)gpu->env_mip_levels : 0.0f;
        float env_intensity = gpu->env_image_view ? gpu->env_intensity : 1.0f;

        /* 16-byte tag block + 16-byte vec4 + 16-byte up-axis block = 48-byte header. */
        VkDeviceSize header_size = 4 * sizeof(uint32_t) + 4 * sizeof(float) +
                                   4 * sizeof(uint32_t);
        /* Per-mesh: 2 x uint64 (addrs) + vec4 color + uint tex_index +
         * uint source_id + uint material_id + uint ptex_color_offset
         * = 16 + 16 + 16 = 48 bytes. The 12-byte trailing scalar block keeps the 16-byte
         * alignment that scalar layout would otherwise enforce on the next
         * struct (irrelevant here — flexible array is the last member — but
         * we keep it natural for consumers that mmap the SSBO). */
        VkDeviceSize per_mesh_size = 2 * sizeof(uint64_t) + 4 * sizeof(float) + 4 * sizeof(uint32_t);
        VkDeviceSize ssbo_size   = header_size + nmeshes * per_mesh_size;

        uint8_t* buf = (uint8_t*)calloc(1, ssbo_size);  /* zero-init: tex_index pad starts at 0 */
        /* Header */
        memcpy(buf + 0, &vertex_stride_floats, sizeof(uint32_t));
        memcpy(buf + 4, &has_materials, sizeof(uint32_t));
        memcpy(buf + 8, &env_mip_levels, sizeof(float));
        memcpy(buf + 12, &env_intensity, sizeof(float));
        memcpy(buf + 16, gpu->dome_color, 4 * sizeof(float));
        uint32_t up_axis = (gpu->scene_up_axis >= 0 && gpu->scene_up_axis <= 2)
            ? (uint32_t)gpu->scene_up_axis : 1u;
        memcpy(buf + 32, &up_axis, sizeof(uint32_t));
        /* Per-mesh data */
        for (uint32_t m = 0; m < nmeshes; m++) {
            const GpuRtMeshDesc* md = &meshes[m];
            uint8_t* entry = buf + header_size + m * per_mesh_size;
            uint64_t va = (uint64_t)(vb_addr + (VkDeviceAddress)md->vertex_offset * md->vertex_stride);
            uint64_t ia = (uint64_t)(ib_addr + (VkDeviceAddress)md->index_offset * sizeof(uint32_t));
            memcpy(entry + 0,  &va, sizeof(uint64_t));
            memcpy(entry + 8,  &ia, sizeof(uint64_t));
            float color[4] = { md->color[0], md->color[1], md->color[2], 1.0f };
            memcpy(entry + 16, color, 4 * sizeof(float));
            uint32_t tex_index = (m < gpu->mesh_tex_indices_count
                                  && gpu->mesh_tex_indices)
                                  ? gpu->mesh_tex_indices[m] : 0xFFFFFFFFu;
            memcpy(entry + 32, &tex_index, sizeof(uint32_t));
            memcpy(entry + 36, &md->source_id, sizeof(uint32_t));
            uint32_t material_id = (md->material_index >= 0 &&
                                    md->material_index < (int32_t)gpu->mat_count)
                ? (uint32_t)md->material_index
                : 0xFFFFFFFFu;  /* unbound: shader keeps per-mesh displayColor */
            memcpy(entry + 40, &material_id, sizeof(uint32_t));
            uint32_t ptex_color_offset =
                (gpu->rt_tri_color_ssbo_buf &&
                 md->ptex_color_offset < gpu->rt_tri_color_count)
                ? md->ptex_color_offset : 0xFFFFFFFFu;
            memcpy(entry + 44, &ptex_color_offset, sizeof(uint32_t));
        }

        /* perf/mem-pool: staging via the staging pool (HOST_VISIBLE,
         * persistent-mapped). */
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory staging_mem = VK_NULL_HANDLE;
        void* staging_mapped = NULL;
        if (!rt_create_staging_buffer(gpu, ssbo_size, &staging, &staging_mem,
                                       &staging_mapped)) {
            fprintf(stderr, "gpu_vulkan: scene data staging alloc failed\n");
            free(buf);
            gpu->rt_built = 1;
            gpu_destroy_rt_scene(gpu);
            return 0;
        }
        memcpy(staging_mapped, buf, ssbo_size);
        free(buf);

        /* scene_data_buf is referenced per-frame from the RT descriptor set
         * but is small enough to allocate privately on mesh-heavy scenes. */
        if (!rt_create_buffer_pooled(gpu, ssbo_size,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         RT_POOL_NONE,
                         &gpu->scene_data_buf, &gpu->scene_data_mem)) {
            fprintf(stderr, "gpu_vulkan: scene data SSBO alloc failed (GPU OOM) — skipping RT\n");
            rt_destroy_staging_buffer(gpu, staging, staging_mem, staging_mapped);
            gpu->rt_built = 1;
            gpu_destroy_rt_scene(gpu);
            return 0;
        }

        VkCommandBuffer cmd = rt_begin_cmd(gpu);
        VkBufferCopy copy = {0};
        copy.size = ssbo_size;
        vkCmdCopyBuffer(cmd, staging, gpu->scene_data_buf, 1, &copy);
        rt_end_cmd(gpu, cmd);

        rt_destroy_staging_buffer(gpu, staging, staging_mem, staging_mapped);
    }

    gpu->scene_data_nmeshes = nmeshes;

    /* ---- RT descriptors ---- */
    {
        int has_mat_ssbo = (gpu->mat_uploaded && gpu->mat_count > 0) ? 1 : 0;
        int has_textures = (gpu->mat_uploaded && gpu->mat_tex_count > 0) ? 1 : 0;
        int has_ibl = (gpu->env_image_view != VK_NULL_HANDLE) ? 1 : 0;
        int rt_tex_count = has_textures ? gpu->mat_tex_count : 0;
        int num_bindings = 3 + (has_mat_ssbo ? 1 : 0) + (has_textures ? 1 : 0)
                         + (has_ibl ? 3 : 0);  /* env, brdfLUT, irradiance */

        /* Phase 11.A.2.5: bumped from [14] to [16] to accommodate
         * curve segments (binding 14) + curve colors (binding 15).
         * Curve bindings are appended only when has_curves.
         * Phase A deferred-shading: bumped to [17] for the G-buffer SSBO
         * at binding 17 (rchit + rmiss write, compute will read in Phase B).
         * Real Ptex colors use binding 18. */
        VkDescriptorSetLayoutBinding bindings[19] = {0};
        /* Binding 0: TLAS */
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;

        /* Binding 1: Storage image */
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

        /* Binding 2: Scene data SSBO (closest-hit reads mesh data, miss reads envMipLevels) */
        bindings[2].binding         = 2;
        bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;

        int bind_idx = 3;
        /* Binding 3: Material SSBO (if materials uploaded) */
        if (has_mat_ssbo) {
            bindings[bind_idx].binding         = 3;
            bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[bind_idx].descriptorCount = 1;
            bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bind_idx++;
        }

        /* Binding 4: Texture array (if textures uploaded) */
        if (has_textures) {
            bindings[bind_idx].binding         = 4;
            bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[bind_idx].descriptorCount = (uint32_t)rt_tex_count;
            bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bind_idx++;
        }

        /* Bindings 5-7: IBL (env map, BRDF LUT, irradiance map) */
        if (has_ibl) {
            for (int b = 5; b <= 7; b++) {
                bindings[bind_idx].binding         = (uint32_t)b;
                bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                bindings[bind_idx].descriptorCount = 1;
                bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
                bind_idx++;
            }
        }

        /* Binding 10: Depth output SSBO (written by closest-hit and miss shaders).
         * Single-camera pipeline always sets depth_enabled=0, so the shader never writes. */
        bindings[bind_idx].binding         = 10;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        bind_idx++;

        /* Binding 11: Segmentation output SSBO (written by closest-hit and miss shaders).
         * Single-camera pipeline always sets segmentation_enabled=0. */
        bindings[bind_idx].binding         = 11;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        bind_idx++;

        /* Binding 12: Normals output SSBO. Single-camera pipeline always sets normals_enabled=0. */
        bindings[bind_idx].binding         = 12;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        bind_idx++;

        /* Binding 13: Scene lights SSBO (read by closest-hit and miss) */
        bindings[bind_idx].binding         = 13;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        bind_idx++;

        /* Bindings 14, 15: Curve segments + per-segment colors (Phase 11.A.2.5).
         * Read by the procedural intersection shader and the curve closest-hit. */
        if (has_curves) {
            bindings[bind_idx].binding         = 14;
            bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[bind_idx].descriptorCount = 1;
            bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bind_idx++;

            bindings[bind_idx].binding         = 15;
            bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[bind_idx].descriptorCount = 1;
            bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bind_idx++;
        }

        /* Phase A deferred-shading: binding 17 — G-buffer SSBO. Written by
         * rchit (on hit) and rmiss (on primary miss). Compute consumer
         * lands in Phase B. The buffer is sized for the current launch
         * (width * height * 32 B), allocated in gpu_ensure_gbuffer below. */
        bindings[bind_idx].binding         = 17;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        bind_idx++;

        /* Binding 18: packed RGBA8 real material colors sampled from authored Ptex maps. */
        bindings[bind_idx].binding         = 18;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        bind_idx++;

        VkDescriptorSetLayoutCreateInfo layout_ci = {0};
        layout_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_ci.bindingCount = (uint32_t)bind_idx;
        layout_ci.pBindings    = bindings;
        vkCreateDescriptorSetLayout(gpu->device, &layout_ci, NULL, &gpu->rt_desc_layout);

        int ibl_sampler_count = has_ibl ? 3 : 0;
        int num_pool_sizes = 3;
        /* Storage buffers: scene_data + (mat) + depth + seg + normals + lights
         * + (curve_seg + curve_color when has_curves) + gbuffer (Phase A)
         * + Ptex triangle-corner colors = 5 baseline + has_mat + 2*has_curves
         * + 1 (gbuffer) + 1 (Ptex) */
        uint32_t storage_buf_count = (uint32_t)((has_mat_ssbo ? 2 : 1) + 4 + (has_curves ? 2 : 0) + 2);
        VkDescriptorPoolSize pool_sizes[6] = {
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, storage_buf_count },
        };
        if (has_textures || has_ibl) {
            pool_sizes[num_pool_sizes].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            pool_sizes[num_pool_sizes].descriptorCount = (uint32_t)(rt_tex_count + ibl_sampler_count);
            num_pool_sizes++;
        }

        VkDescriptorPoolCreateInfo pool_ci = {0};
        pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_ci.maxSets       = 1;
        pool_ci.poolSizeCount = (uint32_t)num_pool_sizes;
        pool_ci.pPoolSizes    = pool_sizes;
        vkCreateDescriptorPool(gpu->device, &pool_ci, NULL, &gpu->rt_desc_pool);

        VkDescriptorSetAllocateInfo alloc_info = {0};
        alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool     = gpu->rt_desc_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts        = &gpu->rt_desc_layout;
        vkAllocateDescriptorSets(gpu->device, &alloc_info, &gpu->rt_desc_set);

        /* Write descriptors */
        VkWriteDescriptorSetAccelerationStructureKHR as_write = {0};
        as_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        as_write.accelerationStructureCount = 1;
        as_write.pAccelerationStructures    = &gpu->tlas;

        VkDescriptorImageInfo img_info = {0};
        img_info.imageView   = gpu->rt_image_view;
        img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo buf_info = {0};
        buf_info.buffer = gpu->scene_data_buf;
        buf_info.offset = 0;
        buf_info.range  = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo mat_buf_info = {0};
        if (has_mat_ssbo) {
            mat_buf_info.buffer = gpu->mat_ssbo_buf;
            mat_buf_info.offset = 0;
            mat_buf_info.range  = VK_WHOLE_SIZE;
        }

        /* Phase 11.A.2.5: bumped from [14] to [16] for curve bindings 14, 15.
         * Phase A deferred-shading: bumped to [17] for G-buffer at binding 17.
         * Real Ptex colors use binding 18. */
        VkWriteDescriptorSet writes[18] = {0};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].pNext           = &as_write;
        writes[0].dstSet          = gpu->rt_desc_set;
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = gpu->rt_desc_set;
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo      = &img_info;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = gpu->rt_desc_set;
        writes[2].dstBinding      = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo     = &buf_info;

        int num_writes = 3;
        if (has_mat_ssbo) {
            writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[num_writes].dstSet          = gpu->rt_desc_set;
            writes[num_writes].dstBinding      = 3;
            writes[num_writes].descriptorCount = 1;
            writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[num_writes].pBufferInfo     = &mat_buf_info;
            num_writes++;
        }

        /* Texture array descriptors */
        VkDescriptorImageInfo* tex_infos = NULL;
        if (has_textures) {
            tex_infos = (VkDescriptorImageInfo*)calloc(
                (size_t)rt_tex_count, sizeof(VkDescriptorImageInfo));
            for (int t = 0; t < rt_tex_count; t++) {
                tex_infos[t].sampler     = gpu->mat_sampler;
                tex_infos[t].imageView   = gpu->mat_image_views[t]
                    ? gpu->mat_image_views[t] : gpu->mat_image_views[0];
                tex_infos[t].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[num_writes].dstSet          = gpu->rt_desc_set;
            writes[num_writes].dstBinding      = 4;
            writes[num_writes].descriptorCount = (uint32_t)rt_tex_count;
            writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[num_writes].pImageInfo      = tex_infos;
            num_writes++;
        }

        /* IBL descriptor writes: bindings 5 (env map), 6 (BRDF LUT), 7 (irradiance) */
        VkDescriptorImageInfo ibl_infos[3] = {0};
        if (has_ibl) {
            ibl_infos[0].sampler     = gpu->env_sampler;
            ibl_infos[0].imageView   = gpu->env_image_view;
            ibl_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            ibl_infos[1].sampler     = gpu->env_sampler;
            ibl_infos[1].imageView   = gpu->brdf_lut_view;
            ibl_infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            ibl_infos[2].sampler     = gpu->irr_sampler ? gpu->irr_sampler : gpu->env_sampler;
            ibl_infos[2].imageView   = gpu->irr_image_view ? gpu->irr_image_view : gpu->env_image_view;
            ibl_infos[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            for (int b = 0; b < 3; b++) {
                writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[num_writes].dstSet          = gpu->rt_desc_set;
                writes[num_writes].dstBinding      = (uint32_t)(5 + b);
                writes[num_writes].descriptorCount = 1;
                writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[num_writes].pImageInfo      = &ibl_infos[b];
                num_writes++;
            }
        }

        /* Bindings 10-11: Depth + Segmentation output SSBOs (dummy — disabled for single-camera) */
        VkDescriptorBufferInfo sensor_dummy_info = {0};
        sensor_dummy_info.buffer = gpu->scene_data_buf;  /* dummy buffer */
        sensor_dummy_info.offset = 0;
        sensor_dummy_info.range  = VK_WHOLE_SIZE;
        /* Binding 10: depth */
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->rt_desc_set;
        writes[num_writes].dstBinding      = 10;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &sensor_dummy_info;
        num_writes++;
        /* Binding 11: segmentation */
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->rt_desc_set;
        writes[num_writes].dstBinding      = 11;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &sensor_dummy_info;
        num_writes++;
        /* Binding 12: normals */
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->rt_desc_set;
        writes[num_writes].dstBinding      = 12;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &sensor_dummy_info;
        num_writes++;

        /* Binding 13: scene lights SSBO. Procedural add_mesh scenes may not
         * have passed through the USD light upload path, but the closest-hit
         * shader always declares this binding. Give it a real zero-light
         * header instead of falling back to unrelated scene/camera buffers. */
        if (!gpu->light_ssbo_buf)
            gpu_upload_lights(gpu, NULL, 0);
        VkDescriptorBufferInfo lights_info = {0};
        lights_info.buffer = gpu->light_ssbo_buf ? gpu->light_ssbo_buf : gpu->scene_data_buf;
        lights_info.offset = 0;
        lights_info.range  = VK_WHOLE_SIZE;
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->rt_desc_set;
        writes[num_writes].dstBinding      = 13;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &lights_info;
        num_writes++;

        /* Phase 11.A.2.5: bindings 14, 15 — curve segments + colors. */
        VkDescriptorBufferInfo curve_seg_info = {0};
        VkDescriptorBufferInfo curve_col_info = {0};
        if (has_curves) {
            curve_seg_info.buffer = gpu->curve_seg_ssbo_buf;
            curve_seg_info.offset = 0;
            curve_seg_info.range  = VK_WHOLE_SIZE;
            writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[num_writes].dstSet          = gpu->rt_desc_set;
            writes[num_writes].dstBinding      = 14;
            writes[num_writes].descriptorCount = 1;
            writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[num_writes].pBufferInfo     = &curve_seg_info;
            num_writes++;

            curve_col_info.buffer = gpu->curve_color_ssbo_buf;
            curve_col_info.offset = 0;
            curve_col_info.range  = VK_WHOLE_SIZE;
            writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[num_writes].dstSet          = gpu->rt_desc_set;
            writes[num_writes].dstBinding      = 15;
            writes[num_writes].descriptorCount = 1;
            writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[num_writes].pBufferInfo     = &curve_col_info;
            num_writes++;
        }

        /* Phase A deferred-shading: binding 17 — G-buffer SSBO. Falls back to
         * scene_data_buf if gbuffer alloc failed (defensive — should not
         * happen on H100 / typical IsaacLab/showcase swapchain sizes). */
        VkDescriptorBufferInfo gbuf_info = {0};
        gbuf_info.buffer = gpu->gbuffer_buf ? gpu->gbuffer_buf : gpu->scene_data_buf;
        gbuf_info.offset = 0;
        gbuf_info.range  = VK_WHOLE_SIZE;
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->rt_desc_set;
        writes[num_writes].dstBinding      = 17;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &gbuf_info;
        num_writes++;

        VkDescriptorBufferInfo ptex_tri_info = {0};
        ptex_tri_info.buffer = gpu->rt_tri_color_ssbo_buf
            ? gpu->rt_tri_color_ssbo_buf : gpu->scene_data_buf;
        ptex_tri_info.offset = 0;
        ptex_tri_info.range  = VK_WHOLE_SIZE;
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->rt_desc_set;
        writes[num_writes].dstBinding      = 18;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &ptex_tri_info;
        num_writes++;

        vkUpdateDescriptorSets(gpu->device, (uint32_t)num_writes, writes, 0, NULL);
        free(tex_infos);
    }

    /* ---- Shadow descriptors (TLAS for ray query in fragment shader) ---- */
    {
        VkDescriptorSetLayoutBinding binding = {0};
        binding.binding         = 0;
        binding.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        binding.descriptorCount = 1;
        binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layout_ci = {0};
        layout_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_ci.bindingCount = 1;
        layout_ci.pBindings    = &binding;
        vkCreateDescriptorSetLayout(gpu->device, &layout_ci, NULL, &gpu->shadow_desc_layout);

        VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 };
        VkDescriptorPoolCreateInfo pool_ci = {0};
        pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_ci.maxSets       = 1;
        pool_ci.poolSizeCount = 1;
        pool_ci.pPoolSizes    = &pool_size;
        vkCreateDescriptorPool(gpu->device, &pool_ci, NULL, &gpu->shadow_desc_pool);

        VkDescriptorSetAllocateInfo alloc_info = {0};
        alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool     = gpu->shadow_desc_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts        = &gpu->shadow_desc_layout;
        vkAllocateDescriptorSets(gpu->device, &alloc_info, &gpu->shadow_desc_set);

        VkWriteDescriptorSetAccelerationStructureKHR as_write = {0};
        as_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        as_write.accelerationStructureCount = 1;
        as_write.pAccelerationStructures    = &gpu->tlas;

        VkWriteDescriptorSet write = {0};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.pNext           = &as_write;
        write.dstSet          = gpu->shadow_desc_set;
        write.dstBinding      = 0;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        vkUpdateDescriptorSets(gpu->device, 1, &write, 0, NULL);

        /* Mirror the same TLAS into the textured-raster materials
         * descriptor set (binding 5) so mesh.frag can do inline
         * ray-queried shadows on textured scenes (NU_RENDER_SHADOW
         * mode). mat_desc_set is created during gpu_upload_materials
         * which happens before the TLAS build, so by the time we get
         * here it should already be allocated. Skip silently if it
         * isn't (procedural-light scenes without uploaded materials). */
        if (gpu->mat_desc_set) {
            VkWriteDescriptorSet mat_write = {0};
            mat_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            mat_write.pNext           = &as_write;
            mat_write.dstSet          = gpu->mat_desc_set;
            mat_write.dstBinding      = 5;
            mat_write.descriptorCount = 1;
            mat_write.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            vkUpdateDescriptorSets(gpu->device, 1, &mat_write, 0, NULL);
        }

        VkPushConstantRange pcr = {0};
        pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pcr.offset     = 0;
        pcr.size       = PUSH_CONSTANT_SIZE;

        VkPipelineLayoutCreateInfo plci = {0};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &gpu->shadow_desc_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        vkCreatePipelineLayout(gpu->device, &plci, NULL, &gpu->shadow_pipeline_layout);

        fprintf(stderr, "gpu_vulkan: shadow descriptors created (ray query TLAS)\n");
    }

    /* ---- RT pipeline ---- */
    {
        VkPushConstantRange pcr = {0};
        pcr.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        pcr.offset     = 0;
        /* Use the larger DLSS struct size so the same layout works for both paths */
        pcr.size       = sizeof(GpuRtPushConstantsDlss) > sizeof(GpuRtPushConstants)
                       ? sizeof(GpuRtPushConstantsDlss) : sizeof(GpuRtPushConstants);

        VkPipelineLayoutCreateInfo plci = {0};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &gpu->rt_desc_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        vkCreatePipelineLayout(gpu->device, &plci, NULL, &gpu->rt_pipeline_layout);

        VkShaderModule rgen_mod = create_shader_module(gpu->device, rgen_spv, rgen_size);
        VkShaderModule miss_mod = create_shader_module(gpu->device, miss_spv, miss_size);
        VkShaderModule chit_mod = create_shader_module(gpu->device, chit_spv, chit_size);
        /* Phase 11.A.2.5: optional curve shader modules. */
        VkShaderModule rint_mod = has_curves
            ? create_shader_module(gpu->device, rint_spv, rint_size)
            : VK_NULL_HANDLE;
        VkShaderModule curve_chit_mod = has_curves
            ? create_shader_module(gpu->device, curve_chit_spv, curve_chit_size)
            : VK_NULL_HANDLE;

        /* Shader stages: 3 baseline (rgen, miss, chit) + 2 with curves (rint, curve_chit). */
        VkPipelineShaderStageCreateInfo stages[5] = {0};
        uint32_t n_stages = 0;
        stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[n_stages].stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        stages[n_stages].module = rgen_mod;
        stages[n_stages].pName  = "main";
        uint32_t rgen_idx = n_stages++;

        stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[n_stages].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
        stages[n_stages].module = miss_mod;
        stages[n_stages].pName  = "main";
        uint32_t miss_idx = n_stages++;

        stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[n_stages].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        stages[n_stages].module = chit_mod;
        stages[n_stages].pName  = "main";
        uint32_t chit_idx = n_stages++;

        uint32_t rint_idx = 0, curve_chit_idx = 0;
        if (has_curves) {
            stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[n_stages].stage  = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
            stages[n_stages].module = rint_mod;
            stages[n_stages].pName  = "main";
            rint_idx = n_stages++;

            stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[n_stages].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            stages[n_stages].module = curve_chit_mod;
            stages[n_stages].pName  = "main";
            curve_chit_idx = n_stages++;
        }

        /* Shader groups: 3 baseline + 1 with curves (procedural hit-group). */
        VkRayTracingShaderGroupCreateInfoKHR groups[4] = {0};
        uint32_t n_groups = 0;
        /* Group 0: raygen */
        groups[n_groups].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        groups[n_groups].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[n_groups].generalShader      = rgen_idx;
        groups[n_groups].closestHitShader   = VK_SHADER_UNUSED_KHR;
        groups[n_groups].anyHitShader       = VK_SHADER_UNUSED_KHR;
        groups[n_groups].intersectionShader = VK_SHADER_UNUSED_KHR;
        n_groups++;
        /* Group 1: miss */
        groups[n_groups].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        groups[n_groups].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[n_groups].generalShader      = miss_idx;
        groups[n_groups].closestHitShader   = VK_SHADER_UNUSED_KHR;
        groups[n_groups].anyHitShader       = VK_SHADER_UNUSED_KHR;
        groups[n_groups].intersectionShader = VK_SHADER_UNUSED_KHR;
        n_groups++;
        /* Group 2: triangle hit-group (mesh) */
        groups[n_groups].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        groups[n_groups].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        groups[n_groups].generalShader      = VK_SHADER_UNUSED_KHR;
        groups[n_groups].closestHitShader   = chit_idx;
        groups[n_groups].anyHitShader       = VK_SHADER_UNUSED_KHR;
        groups[n_groups].intersectionShader = VK_SHADER_UNUSED_KHR;
        n_groups++;
        /* Group 3 (curves only): procedural hit-group for AABB curve segments.
         * Curve TLAS instance picks this group via instanceShaderBindingTableRecordOffset=1. */
        if (has_curves) {
            groups[n_groups].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            groups[n_groups].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
            groups[n_groups].generalShader      = VK_SHADER_UNUSED_KHR;
            groups[n_groups].closestHitShader   = curve_chit_idx;
            groups[n_groups].anyHitShader       = VK_SHADER_UNUSED_KHR;
            groups[n_groups].intersectionShader = rint_idx;
            n_groups++;
        }

        VkRayTracingPipelineCreateInfoKHR rt_ci = {0};
        rt_ci.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        rt_ci.stageCount                   = n_stages;
        rt_ci.pStages                      = stages;
        rt_ci.groupCount                   = n_groups;
        rt_ci.pGroups                      = groups;
        rt_ci.maxPipelineRayRecursionDepth = 3;
        rt_ci.layout                       = gpu->rt_pipeline_layout;

        struct timespec _rt_pipe_t0, _rt_pipe_t1;
        clock_gettime(CLOCK_MONOTONIC, &_rt_pipe_t0);
        VkResult _rt_pipe_res = vkCreateRayTracingPipelinesKHR(
            gpu->device, VK_NULL_HANDLE, gpu->pipeline_cache,
            1, &rt_ci, NULL, &gpu->rt_pipeline);
        clock_gettime(CLOCK_MONOTONIC, &_rt_pipe_t1);
        double _rt_pipe_ms =
            (_rt_pipe_t1.tv_sec  - _rt_pipe_t0.tv_sec)  * 1000.0 +
            (_rt_pipe_t1.tv_nsec - _rt_pipe_t0.tv_nsec) / 1e6;
        fprintf(stderr,
            "gpu_vulkan: create RT pipeline took %.1f ms (cache=%s)\n",
            _rt_pipe_ms, gpu->pipeline_cache ? "on" : "off");
        if (_rt_pipe_res != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to create RT pipeline\n");
            vkDestroyShaderModule(gpu->device, rgen_mod, NULL);
            vkDestroyShaderModule(gpu->device, miss_mod, NULL);
            vkDestroyShaderModule(gpu->device, chit_mod, NULL);
            if (rint_mod)       vkDestroyShaderModule(gpu->device, rint_mod, NULL);
            if (curve_chit_mod) vkDestroyShaderModule(gpu->device, curve_chit_mod, NULL);
            /* Clean up everything allocated so far in this function */
            gpu->rt_built = 1;  /* allow gpu_destroy_rt_scene to clean up */
            gpu_destroy_rt_scene(gpu);
            return 0;
        }

        vkDestroyShaderModule(gpu->device, rgen_mod, NULL);
        vkDestroyShaderModule(gpu->device, miss_mod, NULL);
        vkDestroyShaderModule(gpu->device, chit_mod, NULL);
        if (rint_mod)       vkDestroyShaderModule(gpu->device, rint_mod, NULL);
        if (curve_chit_mod) vkDestroyShaderModule(gpu->device, curve_chit_mod, NULL);
    }

    /* ---- Shader Binding Table ---- */
    {
        uint32_t handle_size   = gpu->rt_handle_size;
        uint32_t align         = gpu->rt_handle_alignment;
        uint32_t handle_stride = (handle_size + (align - 1)) & ~(align - 1);
        /* Phase 11.A.2.5: SBT layout = rgen | miss | hit-groups[1 or 2].
         * Groups 2 = triangle, Group 3 = curve procedural (when has_curves). */
        uint32_t n_hit_groups = 1u + (has_curves ? 1u : 0u);
        uint32_t n_total      = 2u + n_hit_groups;          /* rgen + miss + hits */
        uint32_t sbt_size     = handle_stride * n_total;

        /* Get shader group handles */
        uint8_t* handles = malloc((size_t)handle_size * n_total);
        vkGetRayTracingShaderGroupHandlesKHR(gpu->device, gpu->rt_pipeline, 0, n_total,
                                   handle_size * n_total, handles);

        /* Create SBT buffer */
        if (!rt_create_buffer(gpu, sbt_size,
                         VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &gpu->sbt_buf, &gpu->sbt_mem)) {
            fprintf(stderr, "gpu_vulkan: SBT buffer alloc failed (GPU OOM) — skipping RT\n");
            free(handles);
            gpu->rt_built = 1;
            gpu_destroy_rt_scene(gpu);
            return 0;
        }

        /* Map and write handles at aligned offsets */
        void* mapped;
        vkMapMemory(gpu->device, gpu->sbt_mem, 0, sbt_size, 0, &mapped);
        memset(mapped, 0, sbt_size);
        for (uint32_t g = 0; g < n_total; g++) {
            memcpy((uint8_t*)mapped + handle_stride * g,
                   handles + handle_size * g, handle_size);
        }
        vkUnmapMemory(gpu->device, gpu->sbt_mem);
        free(handles);

        VkDeviceAddress sbt_addr = rt_buf_addr(gpu, gpu->sbt_buf);

        gpu->sbt_rgen.deviceAddress = sbt_addr + handle_stride * 0;
        gpu->sbt_rgen.stride        = handle_stride;
        gpu->sbt_rgen.size          = handle_stride;

        gpu->sbt_miss.deviceAddress = sbt_addr + handle_stride * 1;
        gpu->sbt_miss.stride        = handle_stride;
        gpu->sbt_miss.size          = handle_stride;

        /* Hit-group region spans both triangle (offset 0) and curve (offset 1)
         * groups; the per-instance shaderBindingTableRecordOffset selects. */
        gpu->sbt_hit.deviceAddress  = sbt_addr + handle_stride * 2;
        gpu->sbt_hit.stride         = handle_stride;
        gpu->sbt_hit.size           = handle_stride * n_hit_groups;

        memset(&gpu->sbt_call, 0, sizeof(gpu->sbt_call));
    }

    gpu->rt_built = 1;
    fprintf(stderr, "gpu_vulkan: RT scene built successfully\n");
    return 1;
}

int gpu_update_tlas(Gpu* gpu, const float* transforms, const uint8_t* visibility, uint32_t nmeshes)
{
    if (!gpu || !gpu->rt_built || !gpu->tlas_instance_count) return 0;
    if (!gpu->blas_addresses || !gpu->mesh_to_blas || !gpu->instance_custom) return 0;

    uint32_t ninst = gpu->tlas_instance_count;
    VkDeviceSize inst_buf_size = ninst * sizeof(VkAccelerationStructureInstanceKHR);

    /* Write instance data directly into persistent mapped staging buffer */
    VkAccelerationStructureInstanceKHR* instances;
    VkBuffer staging_buf;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    int staging_is_persistent = 0;

    if (gpu->tlas_update_staging_mapped && gpu->tlas_update_staging_size >= inst_buf_size) {
        /* Fast path: use persistent staging buffer (no alloc/free) */
        instances = (VkAccelerationStructureInstanceKHR*)gpu->tlas_update_staging_mapped;
        staging_buf = gpu->tlas_update_staging_buf;
        staging_is_persistent = 1;
    } else {
        /* Fallback: allocate temporary staging (size mismatch) */
        VkBufferCreateInfo bci = {0};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = inst_buf_size;
        bci.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &staging_buf) != VK_SUCCESS)
            return 0;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, staging_buf, &req);
        VkMemoryAllocateInfo ai = {0};
        ai.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(gpu->device, &ai, NULL, &staging_mem) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, staging_buf, NULL);
            return 0;
        }
        vkBindBufferMemory(gpu->device, staging_buf, staging_mem, 0);
        void* mapped;
        vkMapMemory(gpu->device, staging_mem, 0, inst_buf_size, 0, &mapped);
        instances = (VkAccelerationStructureInstanceKHR*)mapped;
    }

    /* Populate instance transforms */
    for (uint32_t i = 0; i < ninst; i++) {
        uint32_t mesh_idx = gpu->instance_custom[i];
        uint32_t blas_idx = gpu->mesh_to_blas[i];

        if (mesh_idx < nmeshes) {
            memcpy(&instances[i].transform, &transforms[mesh_idx * 12], sizeof(float) * 12);
        } else {
            float id[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
            memcpy(&instances[i].transform, id, sizeof(float) * 12);
        }

        instances[i].instanceCustomIndex                    = mesh_idx;
        {
            uint8_t hidden = (visibility && mesh_idx < nmeshes && !visibility[mesh_idx]);
            uint8_t env_mask = 0xFF;
            if (gpu->instance_mask && mesh_idx < gpu->instance_mask_count) {
                env_mask = gpu->instance_mask[mesh_idx];
            }
            instances[i].mask = hidden ? 0x00 : env_mask;
        }
        instances[i].instanceShaderBindingTableRecordOffset = 0;
        instances[i].flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instances[i].accelerationStructureReference         = gpu->blas_addresses[blas_idx];
    }

    if (!staging_is_persistent)
        vkUnmapMemory(gpu->device, staging_mem);

    /* Use persistent scratch buffer if available */
    VkBuffer scratch_buf;
    VkDeviceMemory scratch_mem_tmp = VK_NULL_HANDLE;
    int scratch_is_persistent = 0;

    if (gpu->tlas_update_scratch_buf) {
        scratch_buf = gpu->tlas_update_scratch_buf;
        scratch_is_persistent = 1;
    } else {
        if (!rt_create_buffer(gpu, gpu->tlas_update_scratch_size,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &scratch_buf, &scratch_mem_tmp)) {
            if (!staging_is_persistent) {
                vkDestroyBuffer(gpu->device, staging_buf, NULL);
                vkFreeMemory(gpu->device, staging_mem, NULL);
            }
            return 0;
        }
    }

    /* Record: copy instances + barrier + update TLAS (in-place) */
    VkCommandBuffer cmd = rt_begin_cmd(gpu);

    VkBufferCopy copy = {0};
    copy.size = inst_buf_size;
    vkCmdCopyBuffer(cmd, staging_buf, gpu->instance_buf, 1, &copy);

    VkMemoryBarrier mb = {0};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &mb, 0, NULL, 0, NULL);

    VkAccelerationStructureGeometryInstancesDataKHR instances_data = {0};
    instances_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances_data.data.deviceAddress = rt_buf_addr(gpu, gpu->instance_buf);

    VkAccelerationStructureGeometryKHR tlas_geom = {0};
    tlas_geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlas_geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlas_geom.geometry.instances = instances_data;

    VkAccelerationStructureBuildGeometryInfoKHR tlas_build = {0};
    tlas_build.sType                      = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    tlas_build.type                       = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlas_build.flags                      = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                          | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    tlas_build.mode                       = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    tlas_build.srcAccelerationStructure   = gpu->tlas;  /* in-place update */
    tlas_build.dstAccelerationStructure   = gpu->tlas;
    tlas_build.geometryCount              = 1;
    tlas_build.pGeometries                = &tlas_geom;
    tlas_build.scratchData.deviceAddress  = rt_buf_addr(gpu, scratch_buf);

    VkAccelerationStructureBuildRangeInfoKHR tlas_range = {0};
    tlas_range.primitiveCount = ninst;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &tlas_range;

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlas_build, &pRange);

    rt_end_cmd(gpu, cmd);

    /* Clean up only temporary allocations */
    if (!staging_is_persistent) {
        vkDestroyBuffer(gpu->device, staging_buf, NULL);
        vkFreeMemory(gpu->device, staging_mem, NULL);
    }
    if (!scratch_is_persistent) {
        vkDestroyBuffer(gpu->device, scratch_buf, NULL);
        vkFreeMemory(gpu->device, scratch_mem_tmp, NULL);
    }

    return 1;
}

/* Inline TLAS update: record instance upload + TLAS build into the current
 * command buffer (gpu->current_cmd) instead of a separate submission.
 * Eliminates a full vkQueueWaitIdle stall per frame when transforms change. */
int gpu_update_tlas_inline(Gpu* gpu, const float* transforms, const uint8_t* visibility, uint32_t nmeshes)
{
    if (!gpu || !gpu->rt_built || !gpu->tlas_instance_count) return 0;
    if (!gpu->current_cmd) return 0;  /* must be called after gpu_begin_frame_*() */
    if (!gpu->blas_addresses || !gpu->mesh_to_blas || !gpu->instance_custom) return 0;

    /* Inline TLAS update bakes vkCmd* into the current cmd buffer; that
     * buffer cannot be replayed verbatim on the next frame (it would
     * re-issue the TLAS build).  Invalidate both caches so the next frame
     * re-records, and so the current-frame recording will not be stored.
     *
     * Setting tiled_cmd_has_tlas_update tells gpu_end_frame_tiled_rt() to
     * skip stashing the in-progress recording as a cache slot. */
    gpu_invalidate_rt_cmd_cache(gpu);

    /* Tiled deferred-begin: gpu_begin_frame_tiled_rt may have set
     * current_cmd to a cached buffer that is NOT in the recording state
     * (because we'd planned to replay).  If so, we now need to put it
     * into recording state because we have new commands to record. */
    int parity = gpu->tiled_cmd_replay_idx & 1;
    if (gpu->current_cmd == gpu->tiled_cached_cmd[parity] &&
        gpu->tiled_cmd_cache_valid[parity]) {
        gpu->tiled_cmd_cache_valid[parity] = 0;  /* will be re-recorded */
        VkCommandBuffer ccmd = gpu->tiled_cached_cmd[parity];
        vkResetCommandBuffer(ccmd, 0);
        VkCommandBufferBeginInfo begin = {0};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        if (vkBeginCommandBuffer(ccmd, &begin) != VK_SUCCESS) {
            return 0;
        }
    }

    gpu_invalidate_tiled_cmd_cache(gpu);
    gpu->tiled_cmd_has_tlas_update = 1;

    VkCommandBuffer cmd = gpu->current_cmd;
    uint32_t ninst = gpu->tlas_instance_count;
    VkDeviceSize inst_buf_size = ninst * sizeof(VkAccelerationStructureInstanceKHR);

    /* Write instance data into persistent mapped staging buffer */
    VkAccelerationStructureInstanceKHR* instances;
    VkBuffer staging_buf;

    if (gpu->tlas_update_staging_mapped && gpu->tlas_update_staging_size >= inst_buf_size) {
        instances = (VkAccelerationStructureInstanceKHR*)gpu->tlas_update_staging_mapped;
        staging_buf = gpu->tlas_update_staging_buf;
    } else {
        return 0;  /* No persistent staging — caller should fall back to gpu_update_tlas() */
    }

    /* Populate instance transforms */
    for (uint32_t i = 0; i < ninst; i++) {
        uint32_t mesh_idx = gpu->instance_custom[i];
        uint32_t blas_idx = gpu->mesh_to_blas[i];

        if (mesh_idx < nmeshes) {
            memcpy(&instances[i].transform, &transforms[mesh_idx * 12], sizeof(float) * 12);
        } else {
            float id[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
            memcpy(&instances[i].transform, id, sizeof(float) * 12);
        }

        instances[i].instanceCustomIndex                    = mesh_idx;
        {
            uint8_t hidden = (visibility && mesh_idx < nmeshes && !visibility[mesh_idx]);
            uint8_t env_mask = 0xFF;
            if (gpu->instance_mask && mesh_idx < gpu->instance_mask_count) {
                env_mask = gpu->instance_mask[mesh_idx];
            }
            instances[i].mask = hidden ? 0x00 : env_mask;
        }
        instances[i].instanceShaderBindingTableRecordOffset = 0;
        instances[i].flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instances[i].accelerationStructureReference         = gpu->blas_addresses[blas_idx];
    }

    /* Copy staging → instance buffer */
    VkBufferCopy copy = {0};
    copy.size = inst_buf_size;
    vkCmdCopyBuffer(cmd, staging_buf, gpu->instance_buf, 1, &copy);

    /* Barrier: transfer write → AS build read */
    VkMemoryBarrier mb = {0};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &mb, 0, NULL, 0, NULL);

    /* Build TLAS (in-place update) */
    VkAccelerationStructureGeometryInstancesDataKHR instances_data = {0};
    instances_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances_data.data.deviceAddress = rt_buf_addr(gpu, gpu->instance_buf);

    VkAccelerationStructureGeometryKHR tlas_geom = {0};
    tlas_geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlas_geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlas_geom.geometry.instances = instances_data;

    VkBuffer scratch_buf = gpu->tlas_update_scratch_buf;
    if (!scratch_buf) return 0;  /* Need persistent scratch */

    VkAccelerationStructureBuildGeometryInfoKHR tlas_build = {0};
    tlas_build.sType                      = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    tlas_build.type                       = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlas_build.flags                      = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                          | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    tlas_build.mode                       = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    tlas_build.srcAccelerationStructure   = gpu->tlas;
    tlas_build.dstAccelerationStructure   = gpu->tlas;
    tlas_build.geometryCount              = 1;
    tlas_build.pGeometries                = &tlas_geom;
    tlas_build.scratchData.deviceAddress  = rt_buf_addr(gpu, scratch_buf);

    VkAccelerationStructureBuildRangeInfoKHR tlas_range = {0};
    tlas_range.primitiveCount = ninst;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &tlas_range;

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlas_build, &pRange);

    /* Barrier: AS build → RT shader read */
    VkMemoryBarrier mb2 = {0};
    mb2.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb2.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    mb2.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0, 1, &mb2, 0, NULL, 0, NULL);

    return 1;
}

static void tiled_destroy_readback(Gpu* gpu); /* forward decl */
static void tiled_destroy_interop(Gpu* gpu);  /* forward decl */
static void tiled_destroy_depth(Gpu* gpu);    /* forward decl */
static int  tiled_ensure_depth(Gpu* gpu, uint32_t total_w, uint32_t total_h); /* forward decl */
static void tiled_destroy_segmentation(Gpu* gpu);    /* forward decl */
static int  tiled_ensure_segmentation(Gpu* gpu, uint32_t total_w, uint32_t total_h); /* forward decl */
static void tiled_destroy_normals(Gpu* gpu);         /* forward decl */
static int  tiled_ensure_normals(Gpu* gpu, uint32_t total_w, uint32_t total_h); /* forward decl */

void gpu_set_instance_masks(Gpu* gpu, const uint8_t* masks, uint32_t count)
{
    if (!gpu) return;
    if (!masks || count == 0) {
        free(gpu->instance_mask);
        gpu->instance_mask = NULL;
        gpu->instance_mask_count = 0;
        return;
    }
    if (count != gpu->instance_mask_count) {
        free(gpu->instance_mask);
        gpu->instance_mask = (uint8_t*)malloc(count);
        if (!gpu->instance_mask) {
            gpu->instance_mask_count = 0;
            return;
        }
        gpu->instance_mask_count = count;
    }
    memcpy(gpu->instance_mask, masks, count);
}

/* Phase C: declare an N-way env partition. The next gpu_build_rt_scene
 * (or standalone gpu_build_partitioned_tlases) will build N per-env
 * TLASes alongside the legacy single TLAS. */
void gpu_set_env_partition(Gpu* gpu, const int* mesh_to_env, int count, int num_envs)
{
    if (!gpu) return;
    free(gpu->mesh_to_env); gpu->mesh_to_env = NULL;
    gpu->mesh_to_env_count = 0;
    gpu->num_envs = 0;

    if (!mesh_to_env || count <= 0 || num_envs <= 0) return;

    gpu->mesh_to_env = (int*)malloc(sizeof(int) * (size_t)count);
    if (!gpu->mesh_to_env) return;
    memcpy(gpu->mesh_to_env, mesh_to_env, sizeof(int) * (size_t)count);
    gpu->mesh_to_env_count = count;
    gpu->num_envs = num_envs;
}

/* Forward decls used below */
static void gpu_destroy_partitioned_tlases(Gpu* gpu);

/* Phase C: build per-env TLAS array. Walks the just-built legacy
 * instances[] array and re-groups into per-env contiguous slices.
 * Each env's slice is built into its own VkAccelerationStructureKHR.
 * Returns 1 on success, 0 on failure. */
int gpu_build_partitioned_tlases(Gpu* gpu,
                                  const void* legacy_instances_void,
                                  uint32_t legacy_count)
{
    if (!gpu || !gpu->rt_available) return 0;
    if (gpu->num_envs <= 0 || !gpu->mesh_to_env) return 0;
    if (!gpu->blas_addresses || !gpu->mesh_to_blas || !gpu->instance_custom) {
        fprintf(stderr, "gpu_vulkan: build_partitioned_tlases — legacy TLAS data missing\n");
        return 0;
    }
    if (!legacy_instances_void || legacy_count == 0) {
        fprintf(stderr, "gpu_vulkan: build_partitioned_tlases — legacy instances missing\n");
        return 0;
    }
    const VkAccelerationStructureInstanceKHR* legacy_instances =
        (const VkAccelerationStructureInstanceKHR*)legacy_instances_void;
    /* Tear down any prior partitioned TLAS state first. */
    gpu_destroy_partitioned_tlases(gpu);

    int num_envs = gpu->num_envs;
    int nmeshes  = gpu->mesh_to_env_count;
    uint32_t legacy_inst_count = legacy_count;

    /* Step 1: compute env_inst_idx[] — for each env count its instances.
     * An env's instance set = (meshes with mesh_to_env[m] == e) plus
     * (meshes with mesh_to_env[m] == -1, i.e. shared globals).
     * We must also drop meshes whose BLAS was excluded from the legacy
     * build (instance_custom[i] is the legacy-side mesh id; we walk those). */

    uint32_t* env_count = (uint32_t*)calloc((size_t)num_envs + 1, sizeof(uint32_t));
    if (!env_count) return 0;

    /* Walk legacy TLAS instances (size = legacy_inst_count). For each one,
     * its custom index is the renderer mesh id; consult mesh_to_env. */
    for (uint32_t i = 0; i < legacy_inst_count; i++) {
        uint32_t m = gpu->instance_custom[i];
        if (m == 0xFFFFFFu) continue;  /* curve sentinel — skip from per-env TLASes */
        int env = (int)m < nmeshes ? gpu->mesh_to_env[m] : -1;
        if (env < 0) {
            /* shared/global — count once per env */
            for (int e = 0; e < num_envs; e++) env_count[e]++;
        } else if (env < num_envs) {
            env_count[env]++;
        }
    }

    /* Prefix sum into env_inst_idx */
    uint32_t* env_inst_idx = (uint32_t*)malloc(sizeof(uint32_t) * ((size_t)num_envs + 1));
    if (!env_inst_idx) { free(env_count); return 0; }
    uint32_t cum = 0;
    for (int e = 0; e < num_envs; e++) {
        env_inst_idx[e] = cum;
        cum += env_count[e];
    }
    env_inst_idx[num_envs] = cum;
    uint32_t total_inst = cum;
    free(env_count);

    if (total_inst == 0) {
        /* Nothing to build — partition is empty. Not an error. */
        free(env_inst_idx);
        return 1;
    }

    gpu->env_inst_idx = env_inst_idx;
    gpu->tlas_arr_total_inst = total_inst;

    /* Step 2: build the partitioned instance buffer (CPU-side) and the
     * inst_to_partial mapping (legacy_idx → partitioned_idx).
     * Walk per env, then within each env walk legacy instances and pick
     * matching ones. O(num_envs * legacy_inst_count) but cheap. */

    uint32_t* inst_to_partial = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)legacy_inst_count);
    if (!inst_to_partial) { free(env_inst_idx); gpu->env_inst_idx = NULL; return 0; }
    for (uint32_t i = 0; i < legacy_inst_count; i++) inst_to_partial[i] = 0xFFFFFFFFu;

    VkAccelerationStructureInstanceKHR* instances =
        (VkAccelerationStructureInstanceKHR*)calloc((size_t)total_inst, sizeof(VkAccelerationStructureInstanceKHR));
    if (!instances) { free(inst_to_partial); free(env_inst_idx); gpu->env_inst_idx = NULL; return 0; }

    uint32_t* per_env_count = (uint32_t*)calloc((size_t)num_envs, sizeof(uint32_t));
    if (!per_env_count) { free(instances); free(inst_to_partial); free(env_inst_idx); gpu->env_inst_idx = NULL; return 0; }

    for (uint32_t i = 0; i < legacy_inst_count; i++) {
        uint32_t m = gpu->instance_custom[i];
        if (m == 0xFFFFFFu) continue;
        int env = (int)m < nmeshes ? gpu->mesh_to_env[m] : -1;

        /* Copy the legacy instance entry (transform + metadata) verbatim
         * — that's the source of truth (built upstream from
         * meshes[m].transform with proper instance.mask, BLAS-ref, etc.). */
        if (env < 0) {
            for (int e = 0; e < num_envs; e++) {
                uint32_t pidx = env_inst_idx[e] + per_env_count[e];
                instances[pidx] = legacy_instances[i];
                per_env_count[e]++;
            }
            /* Map legacy → first env's partitioned slot. Shared globals
             * only get refitted via env 0's slot in the GPU-driven scatter;
             * since identical entries already populate every env's TLAS,
             * the env-0 transform copies are correct for all of them. */
            inst_to_partial[i] = env_inst_idx[0] + (per_env_count[0] - 1);
        } else if (env < num_envs) {
            uint32_t pidx = env_inst_idx[env] + per_env_count[env];
            instances[pidx] = legacy_instances[i];
            per_env_count[env]++;
            inst_to_partial[i] = pidx;
        }
    }
    free(per_env_count);

    gpu->inst_to_partial = inst_to_partial;

    /* Step 3: create the device-local instance buffer and upload */
    VkDeviceSize inst_buf_size = (VkDeviceSize)total_inst * sizeof(VkAccelerationStructureInstanceKHR);

    VkBuffer staging_buf = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    void* staging_mapped = NULL;
    if (!rt_create_staging_buffer(gpu, inst_buf_size, &staging_buf, &staging_mem, &staging_mapped)) {
        fprintf(stderr, "gpu_vulkan: partitioned TLAS — staging alloc failed\n");
        free(instances);
        gpu_destroy_partitioned_tlases(gpu);
        return 0;
    }
    memcpy(staging_mapped, instances, inst_buf_size);
    free(instances);

    if (!rt_create_buffer_pooled(gpu, inst_buf_size,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                     | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                     | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     RT_POOL_RESIDENT,
                     &gpu->tlas_arr_inst_buf, &gpu->tlas_arr_inst_mem)) {
        fprintf(stderr, "gpu_vulkan: partitioned TLAS — instance buf alloc failed\n");
        rt_destroy_staging_buffer(gpu, staging_buf, staging_mem, staging_mapped);
        gpu_destroy_partitioned_tlases(gpu);
        return 0;
    }
    gpu->tlas_arr_inst_size = inst_buf_size;

    /* Step 4: query each env's TLAS size + scratch, sub-allocate, build */
    gpu->tlas_arr     = (VkAccelerationStructureKHR*)calloc((size_t)num_envs, sizeof(VkAccelerationStructureKHR));
    gpu->tlas_arr_buf = (VkBuffer*)calloc((size_t)num_envs, sizeof(VkBuffer));
    gpu->tlas_arr_buf_offset = (VkDeviceSize*)calloc((size_t)num_envs, sizeof(VkDeviceSize));
    if (!gpu->tlas_arr || !gpu->tlas_arr_buf || !gpu->tlas_arr_buf_offset) {
        fprintf(stderr, "gpu_vulkan: partitioned TLAS — array alloc failed\n");
        rt_destroy_staging_buffer(gpu, staging_buf, staging_mem, staging_mapped);
        gpu_destroy_partitioned_tlases(gpu);
        return 0;
    }

    /* Get instance buf device address */
    VkDeviceAddress inst_addr = rt_buf_addr(gpu, gpu->tlas_arr_inst_buf);

    /* Pre-compute build infos and sizes per env */
    VkAccelerationStructureBuildSizesInfoKHR* sizes_info =
        (VkAccelerationStructureBuildSizesInfoKHR*)calloc((size_t)num_envs, sizeof(VkAccelerationStructureBuildSizesInfoKHR));
    VkAccelerationStructureBuildGeometryInfoKHR* build_infos =
        (VkAccelerationStructureBuildGeometryInfoKHR*)calloc((size_t)num_envs, sizeof(VkAccelerationStructureBuildGeometryInfoKHR));
    VkAccelerationStructureGeometryKHR* geoms =
        (VkAccelerationStructureGeometryKHR*)calloc((size_t)num_envs, sizeof(VkAccelerationStructureGeometryKHR));
    VkAccelerationStructureBuildRangeInfoKHR* ranges =
        (VkAccelerationStructureBuildRangeInfoKHR*)calloc((size_t)num_envs, sizeof(VkAccelerationStructureBuildRangeInfoKHR));
    const VkAccelerationStructureBuildRangeInfoKHR** range_ptrs =
        (const VkAccelerationStructureBuildRangeInfoKHR**)calloc((size_t)num_envs, sizeof(void*));
    if (!sizes_info || !build_infos || !geoms || !ranges || !range_ptrs) {
        fprintf(stderr, "gpu_vulkan: partitioned TLAS — build info alloc failed\n");
        free(sizes_info); free(build_infos); free(geoms); free(ranges); free(range_ptrs);
        rt_destroy_staging_buffer(gpu, staging_buf, staging_mem, staging_mapped);
        gpu_destroy_partitioned_tlases(gpu);
        return 0;
    }

    VkDeviceSize total_tlas_size = 0;
    VkDeviceSize total_scratch_size = 0;
    VkDeviceSize tlas_align = 256;          /* VkAccelerationStructure size alignment */
    VkDeviceSize scratch_align = gpu->min_as_scratch_align ? gpu->min_as_scratch_align : 256;

    for (int e = 0; e < num_envs; e++) {
        uint32_t env_count_e = env_inst_idx[e+1] - env_inst_idx[e];

        sizes_info[e].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

        geoms[e].sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geoms[e].geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geoms[e].geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geoms[e].geometry.instances.data.deviceAddress =
            inst_addr + (VkDeviceSize)env_inst_idx[e] * sizeof(VkAccelerationStructureInstanceKHR);

        build_infos[e].sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build_infos[e].type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        build_infos[e].flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                     | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        build_infos[e].mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_infos[e].geometryCount = 1;
        build_infos[e].pGeometries   = &geoms[e];

        vkGetAccelerationStructureBuildSizesKHR(gpu->device,
                                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                &build_infos[e], &env_count_e, &sizes_info[e]);

        gpu->tlas_arr_buf_offset[e] = total_tlas_size;
        VkDeviceSize sz = sizes_info[e].accelerationStructureSize;
        sz = (sz + tlas_align - 1) & ~(tlas_align - 1);
        total_tlas_size += sz;
        VkDeviceSize ssz = sizes_info[e].buildScratchSize;
        ssz = (ssz + scratch_align - 1) & ~(scratch_align - 1);
        total_scratch_size += ssz;

        ranges[e].primitiveCount = env_count_e;
        range_ptrs[e] = &ranges[e];
    }

    /* Allocate the big TLAS storage buffer */
    {
        VkBuffer big_tlas_buf = VK_NULL_HANDLE;
        VkDeviceMemory big_tlas_mem = VK_NULL_HANDLE;
        if (!rt_create_buffer_pooled(gpu, total_tlas_size,
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         RT_POOL_RESIDENT,
                         &big_tlas_buf, &big_tlas_mem)) {
            fprintf(stderr, "gpu_vulkan: partitioned TLAS — storage alloc failed (%llu MB)\n",
                    (unsigned long long)(total_tlas_size / (1024*1024)));
            free(sizes_info); free(build_infos); free(geoms); free(ranges); free(range_ptrs);
            rt_destroy_staging_buffer(gpu, staging_buf, staging_mem, staging_mapped);
            gpu_destroy_partitioned_tlases(gpu);
            return 0;
        }
        /* Each env's tlas_arr_buf points into this single backing — store
         * the same handle, vary offset. */
        for (int e = 0; e < num_envs; e++) {
            gpu->tlas_arr_buf[e] = big_tlas_buf;  /* shared handle */
        }
        gpu->tlas_arr_mem = big_tlas_mem;
        /* Stash the actual buffer in slot 0 for cleanup; use a sentinel by
         * keeping all entries equal — destroy walks unique values. */
    }

    /* Allocate the scratch buffer */
    if (!rt_create_buffer_pooled(gpu, total_scratch_size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     RT_POOL_TRANSIENT,
                     &gpu->tlas_arr_scratch_buf, &gpu->tlas_arr_scratch_mem)) {
        fprintf(stderr, "gpu_vulkan: partitioned TLAS — scratch alloc failed\n");
        free(sizes_info); free(build_infos); free(geoms); free(ranges); free(range_ptrs);
        rt_destroy_staging_buffer(gpu, staging_buf, staging_mem, staging_mapped);
        gpu_destroy_partitioned_tlases(gpu);
        return 0;
    }
    gpu->tlas_arr_scratch_size = total_scratch_size;
    VkDeviceAddress scratch_addr = rt_buf_addr(gpu, gpu->tlas_arr_scratch_buf);

    /* Create each VkAccelerationStructureKHR at its offset, hook scratch */
    VkDeviceSize tlas_off = 0;
    VkDeviceSize scratch_off = 0;
    for (int e = 0; e < num_envs; e++) {
        VkAccelerationStructureCreateInfoKHR ci = {0};
        ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = gpu->tlas_arr_buf[0];
        ci.offset = tlas_off;
        ci.size   = sizes_info[e].accelerationStructureSize;
        ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        if (vkCreateAccelerationStructureKHR(gpu->device, &ci, NULL, &gpu->tlas_arr[e]) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: partitioned TLAS — vkCreateAccelerationStructureKHR failed env=%d\n", e);
            free(sizes_info); free(build_infos); free(geoms); free(ranges); free(range_ptrs);
            rt_destroy_staging_buffer(gpu, staging_buf, staging_mem, staging_mapped);
            gpu_destroy_partitioned_tlases(gpu);
            return 0;
        }
        gpu->tlas_arr_buf_offset[e] = tlas_off;
        VkDeviceSize sz = sizes_info[e].accelerationStructureSize;
        sz = (sz + tlas_align - 1) & ~(tlas_align - 1);
        tlas_off += sz;

        build_infos[e].dstAccelerationStructure  = gpu->tlas_arr[e];
        build_infos[e].scratchData.deviceAddress = scratch_addr + scratch_off;
        VkDeviceSize ssz = sizes_info[e].buildScratchSize;
        ssz = (ssz + scratch_align - 1) & ~(scratch_align - 1);
        scratch_off += ssz;
    }

    /* Step 5: record copy + barrier + batched build */
    VkCommandBuffer cmd = rt_begin_cmd(gpu);
    VkBufferCopy copy = {0};
    copy.size = inst_buf_size;
    vkCmdCopyBuffer(cmd, staging_buf, gpu->tlas_arr_inst_buf, 1, &copy);

    VkMemoryBarrier mb = {0};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &mb, 0, NULL, 0, NULL);

    /* Batched build — one call covers all envs */
    vkCmdBuildAccelerationStructuresKHR(cmd, (uint32_t)num_envs, build_infos, range_ptrs);

    rt_end_cmd(gpu, cmd);

    rt_destroy_staging_buffer(gpu, staging_buf, staging_mem, staging_mapped);
    free(sizes_info); free(build_infos); free(geoms); free(ranges); free(range_ptrs);

    gpu->tlas_arr_built = 1;

    /* Upload inst_to_partial[] as a GPU storage buffer so the
     * partitioned-translate compute shader can read it. */
    {
        VkDeviceSize idx_size = (VkDeviceSize)legacy_inst_count * sizeof(uint32_t);
        if (gpu->inst_to_partial_buf && gpu->inst_to_partial_count != legacy_inst_count) {
            vkDestroyBuffer(gpu->device, gpu->inst_to_partial_buf, NULL);
            vkFreeMemory(gpu->device, gpu->inst_to_partial_mem, NULL);
            gpu->inst_to_partial_buf = VK_NULL_HANDLE;
            gpu->inst_to_partial_mem = VK_NULL_HANDLE;
        }
        if (!gpu->inst_to_partial_buf) {
            VkBufferCreateInfo bci = {0};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size  = idx_size;
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                      | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            vkCreateBuffer(gpu->device, &bci, NULL, &gpu->inst_to_partial_buf);
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(gpu->device, gpu->inst_to_partial_buf, &req);
            VkMemoryAllocateInfo ai = {0};
            ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize  = req.size;
            ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            gpu_alloc_memory(gpu, &ai, &gpu->inst_to_partial_mem);
            vkBindBufferMemory(gpu->device, gpu->inst_to_partial_buf, gpu->inst_to_partial_mem, 0);
            gpu->inst_to_partial_count = legacy_inst_count;
        }
        /* Upload via host-visible staging */
        VkBuffer st_buf = VK_NULL_HANDLE;
        VkDeviceMemory st_mem = VK_NULL_HANDLE;
        VkBufferCreateInfo sbci = {0};
        sbci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        sbci.size  = idx_size;
        sbci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        vkCreateBuffer(gpu->device, &sbci, NULL, &st_buf);
        VkMemoryRequirements sreq;
        vkGetBufferMemoryRequirements(gpu->device, st_buf, &sreq);
        VkMemoryAllocateInfo sai = {0};
        sai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        sai.allocationSize  = sreq.size;
        sai.memoryTypeIndex = find_memory_type(gpu->physical_device, sreq.memoryTypeBits,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        gpu_alloc_memory(gpu, &sai, &st_mem);
        vkBindBufferMemory(gpu->device, st_buf, st_mem, 0);
        void* mapped = NULL;
        vkMapMemory(gpu->device, st_mem, 0, idx_size, 0, &mapped);
        memcpy(mapped, gpu->inst_to_partial, idx_size);
        vkUnmapMemory(gpu->device, st_mem);

        VkCommandBuffer cmd_up = rt_begin_cmd(gpu);
        VkBufferCopy cp = {0}; cp.size = idx_size;
        vkCmdCopyBuffer(cmd_up, st_buf, gpu->inst_to_partial_buf, 1, &cp);
        rt_end_cmd(gpu, cmd_up);

        vkDestroyBuffer(gpu->device, st_buf, NULL);
        vkFreeMemory(gpu->device, st_mem, NULL);
        gpu->tlas_translate_p_ds_dirty = 1;
    }

    fprintf(stderr, "gpu_vulkan: partitioned TLAS array built (%d envs, %u total instances, %llu MB tlas, %llu MB scratch)\n",
            num_envs, total_inst,
            (unsigned long long)(total_tlas_size / (1024*1024)),
            (unsigned long long)(total_scratch_size / (1024*1024)));
    return 1;
}

static void gpu_destroy_partitioned_tlases(Gpu* gpu)
{
    if (!gpu) return;
    if (gpu->tlas_arr) {
        for (int e = 0; e < gpu->num_envs; e++) {
            if (gpu->tlas_arr[e])
                vkDestroyAccelerationStructureKHR(gpu->device, gpu->tlas_arr[e], NULL);
        }
        free(gpu->tlas_arr);
        gpu->tlas_arr = NULL;
    }
    if (gpu->tlas_arr_buf) {
        /* All entries share one VkBuffer (sub-allocated by offset). Destroy once. */
        VkBuffer shared = gpu->tlas_arr_buf[0];
        if (shared) vkDestroyBuffer(gpu->device, shared, NULL);
        free(gpu->tlas_arr_buf);
        gpu->tlas_arr_buf = NULL;
    }
    if (gpu->tlas_arr_mem) {
        /* Pool memory — pool reset will reclaim. Don't vkFreeMemory directly
         * if it came from the pool. We track mem only for the private-alloc
         * fallback path; the pool path returns mem == VK_NULL_HANDLE.
         * rt_create_buffer_pooled returns mem!=NULL only on private-alloc
         * fallback, in which case we own it. */
        vkFreeMemory(gpu->device, gpu->tlas_arr_mem, NULL);
        gpu->tlas_arr_mem = VK_NULL_HANDLE;
    }
    free(gpu->tlas_arr_buf_offset);
    gpu->tlas_arr_buf_offset = NULL;

    if (gpu->tlas_arr_inst_buf) {
        vkDestroyBuffer(gpu->device, gpu->tlas_arr_inst_buf, NULL);
        gpu->tlas_arr_inst_buf = VK_NULL_HANDLE;
    }
    if (gpu->tlas_arr_inst_mem) {
        vkFreeMemory(gpu->device, gpu->tlas_arr_inst_mem, NULL);
        gpu->tlas_arr_inst_mem = VK_NULL_HANDLE;
    }
    gpu->tlas_arr_inst_mapped = NULL;
    gpu->tlas_arr_inst_size = 0;

    if (gpu->tlas_arr_scratch_buf) {
        vkDestroyBuffer(gpu->device, gpu->tlas_arr_scratch_buf, NULL);
        gpu->tlas_arr_scratch_buf = VK_NULL_HANDLE;
    }
    if (gpu->tlas_arr_scratch_mem) {
        vkFreeMemory(gpu->device, gpu->tlas_arr_scratch_mem, NULL);
        gpu->tlas_arr_scratch_mem = VK_NULL_HANDLE;
    }
    gpu->tlas_arr_scratch_size = 0;

    free(gpu->env_inst_idx);    gpu->env_inst_idx = NULL;
    free(gpu->inst_to_partial); gpu->inst_to_partial = NULL;
    gpu->tlas_arr_total_inst = 0;
    gpu->tlas_arr_built = 0;

    /* Phase C: tear down the inst_to_partial GPU buffer + invalidate
     * partitioned-translate descriptor (rebuilt on next dispatch). */
    if (gpu->inst_to_partial_buf) {
        vkDestroyBuffer(gpu->device, gpu->inst_to_partial_buf, NULL);
        gpu->inst_to_partial_buf = VK_NULL_HANDLE;
    }
    if (gpu->inst_to_partial_mem) {
        vkFreeMemory(gpu->device, gpu->inst_to_partial_mem, NULL);
        gpu->inst_to_partial_mem = VK_NULL_HANDLE;
    }
    gpu->inst_to_partial_count = 0;
    gpu->tlas_translate_p_ds_dirty = 1;
}

int gpu_update_scene_colors(Gpu* gpu, const float* colors, uint32_t nmeshes)
{
    if (!gpu || !colors || !gpu->scene_data_buf) return 0;
    if (nmeshes != gpu->scene_data_nmeshes) return 0;

    /* Scene SSBO layout: 48-byte header (4 scalars + vec4 dome_color +
     * upAxis/pad), then
     * per-mesh entries of 48 bytes each (2 x uint64 addresses + vec4 color
     * + uint tex_index + 12-byte pad).  Color starts at offset 16 within
     * each entry; tex_index at offset 32.  Use vkCmdUpdateBuffer for small
     * updates (limit: 65536 bytes). */
    uint32_t header_size = 48;
    uint32_t entry_size = 48;  /* 2*8 (addrs) + 4*4 (color) + 4*4 (tex_index + pad) */
    uint32_t color_offset_in_entry = 16;

    /* Build a contiguous buffer of color updates.
     * vkCmdUpdateBuffer can only update contiguous regions, so we update the
     * entire per-mesh array and just keep the existing address fields. */
    VkDeviceSize update_size = (VkDeviceSize)nmeshes * entry_size;

    /* For large scenes, fall back to staging buffer */
    if (update_size > 65536) {
        /* Staging path for > 2048 meshes */
        VkBuffer staging;
        VkDeviceMemory staging_mem;
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = (VkDeviceSize)nmeshes * 16;  /* just the color vec4s */
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &staging) != VK_SUCCESS) return 0;

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, staging, &req);
        VkMemoryAllocateInfo ai = {0};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (gpu_alloc_memory(gpu, &ai, &staging_mem) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, staging, NULL);
            return 0;
        }
        vkBindBufferMemory(gpu->device, staging, staging_mem, 0);

        void* mapped;
        vkMapMemory(gpu->device, staging_mem, 0, bci.size, 0, &mapped);
        float* dst = (float*)mapped;
        for (uint32_t m = 0; m < nmeshes; m++) {
            dst[m * 4 + 0] = colors[m * 3 + 0];
            dst[m * 4 + 1] = colors[m * 3 + 1];
            dst[m * 4 + 2] = colors[m * 3 + 2];
            dst[m * 4 + 3] = 1.0f;
        }
        vkUnmapMemory(gpu->device, staging_mem);

        VkCommandBuffer cmd = rt_begin_cmd(gpu);
        VkBufferCopy* regions = (VkBufferCopy*)malloc(nmeshes * sizeof(VkBufferCopy));
        for (uint32_t m = 0; m < nmeshes; m++) {
            regions[m].srcOffset = (VkDeviceSize)m * 16;
            regions[m].dstOffset = (VkDeviceSize)header_size + m * entry_size + color_offset_in_entry;
            regions[m].size = 16;
        }
        vkCmdCopyBuffer(cmd, staging, gpu->scene_data_buf, nmeshes, regions);
        free(regions);
        rt_end_cmd(gpu, cmd);

        vkDestroyBuffer(gpu->device, staging, NULL);
        vkFreeMemory(gpu->device, staging_mem, NULL);
    } else {
        /* Fast path: use vkCmdUpdateBuffer for each mesh's color (small scenes) */
        VkCommandBuffer cmd = rt_begin_cmd(gpu);
        for (uint32_t m = 0; m < nmeshes; m++) {
            float color[4] = { colors[m * 3 + 0], colors[m * 3 + 1], colors[m * 3 + 2], 1.0f };
            VkDeviceSize offset = (VkDeviceSize)header_size + m * entry_size + color_offset_in_entry;
            vkCmdUpdateBuffer(cmd, gpu->scene_data_buf, offset, 16, color);
        }
        rt_end_cmd(gpu, cmd);
    }

    return 1;
}

int gpu_set_dome_color(Gpu* gpu, float r, float g, float b, float intensity)
{
    if (!gpu) return 0;
    /* Stash on the Gpu so the next gpu_build_rt_scene call uses the new
     * value. Negative intensity falls back to 0 — caller's responsibility
     * to pre-normalize USD-authored intensity to [0,1] for fast_mode. */
    gpu->dome_color[0] = (r > 0.0f) ? r : 0.0f;
    gpu->dome_color[1] = (g > 0.0f) ? g : 0.0f;
    gpu->dome_color[2] = (b > 0.0f) ? b : 0.0f;
    gpu->dome_color[3] = (intensity > 0.0f) ? intensity : 0.0f;

    /* If the SSBO is already live, patch the header in-place. The vec4
     * domeColor lives at offset 16 (4 ints + vec4). vkCmdUpdateBuffer is
     * limited to 65536 bytes; 16 bytes is tiny. */
    if (gpu->scene_data_buf) {
        VkCommandBuffer cmd = rt_begin_cmd(gpu);
        vkCmdUpdateBuffer(cmd, gpu->scene_data_buf, 16, 16, gpu->dome_color);
        rt_end_cmd(gpu, cmd);
        /* The cached RT/tiled command buffers reference scene_data_buf via
         * descriptor sets, not by content — but invalidate to be safe in
         * case any path encodes the dome color as part of replay state. */
        gpu_invalidate_rt_cmd_cache(gpu);
        gpu_invalidate_tiled_cmd_cache(gpu);
    }
    return 1;
}

int gpu_set_scene_up_axis(Gpu* gpu, int up_axis)
{
    if (!gpu) return 0;
    if (up_axis < 0 || up_axis > 2) up_axis = 1;
    gpu->scene_up_axis = up_axis;

    if (gpu->scene_data_buf) {
        uint32_t axis_u = (uint32_t)up_axis;
        VkCommandBuffer cmd = rt_begin_cmd(gpu);
        vkCmdUpdateBuffer(cmd, gpu->scene_data_buf, 32, 4, &axis_u);
        rt_end_cmd(gpu, cmd);
        gpu_invalidate_rt_cmd_cache(gpu);
        gpu_invalidate_tiled_cmd_cache(gpu);
    }
    return 1;
}

int gpu_set_mesh_texture(Gpu* gpu, uint32_t mesh_id, uint32_t tex_index)
{
    if (!gpu) return 0;

    /* Grow the staging array up to mesh_id+1 entries, init new slots to
     * the sentinel 0xFFFFFFFF ("no texture"). */
    uint32_t need = mesh_id + 1;
    if (need > gpu->mesh_tex_indices_capacity) {
        uint32_t new_cap = gpu->mesh_tex_indices_capacity ? gpu->mesh_tex_indices_capacity : 64;
        while (new_cap < need) new_cap *= 2;
        uint32_t* p = (uint32_t*)realloc(gpu->mesh_tex_indices, new_cap * sizeof(uint32_t));
        if (!p) return 0;
        for (uint32_t k = gpu->mesh_tex_indices_capacity; k < new_cap; k++) {
            p[k] = 0xFFFFFFFFu;
        }
        gpu->mesh_tex_indices = p;
        gpu->mesh_tex_indices_capacity = new_cap;
    }
    if (need > gpu->mesh_tex_indices_count) {
        gpu->mesh_tex_indices_count = need;
    }
    gpu->mesh_tex_indices[mesh_id] = tex_index;

    /* If the SSBO is already live AND the slot exists in the upload, patch
     * the per-mesh tex_index field in place. The entry layout is:
     *   offset 0..15  : 2 x uint64 vertex/index addrs
     *   offset 16..31 : vec4 color
     *   offset 32..35 : uint tex_index   ← we write this
     *   offset 36..47 : pad (untouched)
     * Per-entry size is 48; SceneData header is 48. */
    if (gpu->scene_data_buf && mesh_id < gpu->scene_data_nmeshes) {
        VkDeviceSize entry_off = 48 + (VkDeviceSize)mesh_id * 48 + 32;
        VkCommandBuffer cmd = rt_begin_cmd(gpu);
        vkCmdUpdateBuffer(cmd, gpu->scene_data_buf, entry_off, 4, &tex_index);
        rt_end_cmd(gpu, cmd);
        /* No descriptor changes — same SSBO; cmd cache replay is fine. */
    }
    return 1;
}

static void update_material_ptex_descriptor(Gpu* gpu)
{
    if (!gpu || !gpu->mat_desc_set || !gpu->mat_desc_pool || !gpu->mat_desc_layout)
        return;

    VkDescriptorBufferInfo ptex_info = {0};
    if (gpu->rt_tri_color_ssbo_buf) {
        ptex_info.buffer = gpu->rt_tri_color_ssbo_buf;
    } else if (gpu->mat_ssbo_buf) {
        ptex_info.buffer = gpu->mat_ssbo_buf;
    } else if (gpu->light_ssbo_buf) {
        ptex_info.buffer = gpu->light_ssbo_buf;
    } else {
        return;
    }
    ptex_info.offset = 0;
    ptex_info.range  = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write = {0};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = gpu->mat_desc_set;
    write.dstBinding      = 7;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo     = &ptex_info;
    vkUpdateDescriptorSets(gpu->device, 1, &write, 0, NULL);
}

static void rt_triangle_colors_destroy(Gpu* gpu)
{
    if (!gpu) return;
    if (gpu->rt_tri_color_ssbo_buf) {
        vkDestroyBuffer(gpu->device, gpu->rt_tri_color_ssbo_buf, NULL);
        if (gpu->rt_tri_color_ssbo_mem)
            vkFreeMemory(gpu->device, gpu->rt_tri_color_ssbo_mem, NULL);
        gpu->rt_tri_color_ssbo_buf = VK_NULL_HANDLE;
        gpu->rt_tri_color_ssbo_mem = VK_NULL_HANDLE;
        gpu->rt_tri_color_count = 0;
    }
    update_material_ptex_descriptor(gpu);
}

void gpu_destroy_rt_scene(Gpu* gpu)
{
    if (!gpu) return;
    if (!gpu->rt_built) {
        rt_triangle_colors_destroy(gpu);
        return;
    }
    vkDeviceWaitIdle(gpu->device);

    /* Phase 0 cluster-AS backing buffers (NUSD_RT_CLUSTER) — device is idle. */
    for (int i = 0; i < gpu->cluster_as_count; i++) {
        if (gpu->cluster_as_bufs[i]) vkDestroyBuffer(gpu->device, gpu->cluster_as_bufs[i], NULL);
        if (gpu->cluster_as_mems[i]) vkFreeMemory(gpu->device, gpu->cluster_as_mems[i], NULL);
    }
    free(gpu->cluster_as_bufs); gpu->cluster_as_bufs = NULL;
    free(gpu->cluster_as_mems); gpu->cluster_as_mems = NULL;
    gpu->cluster_as_count = 0; gpu->cluster_as_cap = 0;

    /* Tearing down the RT scene frees pipelines, descriptor pools, SBT, etc.;
     * any cached cmd buffer that referenced these is now dangling. */
    gpu_invalidate_rt_cmd_cache(gpu);
    gpu_invalidate_tiled_cmd_cache(gpu);

    /* Shadow descriptors */
    if (gpu->shadow_pipeline_layout) { vkDestroyPipelineLayout(gpu->device, gpu->shadow_pipeline_layout, NULL); gpu->shadow_pipeline_layout = VK_NULL_HANDLE; }
    if (gpu->shadow_desc_pool)       { vkDestroyDescriptorPool(gpu->device, gpu->shadow_desc_pool, NULL); gpu->shadow_desc_pool = VK_NULL_HANDLE; }
    if (gpu->shadow_desc_layout)     { vkDestroyDescriptorSetLayout(gpu->device, gpu->shadow_desc_layout, NULL); gpu->shadow_desc_layout = VK_NULL_HANDLE; }

    if (gpu->rt_pipeline)        { vkDestroyPipeline(gpu->device, gpu->rt_pipeline, NULL); gpu->rt_pipeline = VK_NULL_HANDLE; }
    if (gpu->rt_pipeline_layout) { vkDestroyPipelineLayout(gpu->device, gpu->rt_pipeline_layout, NULL); gpu->rt_pipeline_layout = VK_NULL_HANDLE; }
    if (gpu->sbt_buf)            { vkDestroyBuffer(gpu->device, gpu->sbt_buf, NULL); vkFreeMemory(gpu->device, gpu->sbt_mem, NULL); gpu->sbt_buf = VK_NULL_HANDLE; gpu->sbt_mem = VK_NULL_HANDLE; }
    if (gpu->rt_desc_pool)       { vkDestroyDescriptorPool(gpu->device, gpu->rt_desc_pool, NULL); gpu->rt_desc_pool = VK_NULL_HANDLE; }
    if (gpu->rt_desc_layout)     { vkDestroyDescriptorSetLayout(gpu->device, gpu->rt_desc_layout, NULL); gpu->rt_desc_layout = VK_NULL_HANDLE; }
    if (gpu->scene_data_buf)     { vkDestroyBuffer(gpu->device, gpu->scene_data_buf, NULL); vkFreeMemory(gpu->device, gpu->scene_data_mem, NULL); gpu->scene_data_buf = VK_NULL_HANDLE; gpu->scene_data_mem = VK_NULL_HANDLE; }
    rt_triangle_colors_destroy(gpu);
    if (gpu->rt_image_view)      { vkDestroyImageView(gpu->device, gpu->rt_image_view, NULL); gpu->rt_image_view = VK_NULL_HANDLE; }
    if (gpu->rt_image)           { vkDestroyImage(gpu->device, gpu->rt_image, NULL); gpu->rt_image = VK_NULL_HANDLE; }
    if (gpu->rt_image_mem)       { vkFreeMemory(gpu->device, gpu->rt_image_mem, NULL); gpu->rt_image_mem = VK_NULL_HANDLE; }
    /* Phase A G-buffer */
    if (gpu->gbuffer_staging_mapped) { vkUnmapMemory(gpu->device, gpu->gbuffer_staging_mem); gpu->gbuffer_staging_mapped = NULL; }
    if (gpu->gbuffer_staging_buf) { vkDestroyBuffer(gpu->device, gpu->gbuffer_staging_buf, NULL); vkFreeMemory(gpu->device, gpu->gbuffer_staging_mem, NULL); gpu->gbuffer_staging_buf = VK_NULL_HANDLE; gpu->gbuffer_staging_mem = VK_NULL_HANDLE; }
    if (gpu->gbuffer_buf)        { vkDestroyBuffer(gpu->device, gpu->gbuffer_buf, NULL); vkFreeMemory(gpu->device, gpu->gbuffer_mem, NULL); gpu->gbuffer_buf = VK_NULL_HANDLE; gpu->gbuffer_mem = VK_NULL_HANDLE; }
    gpu->gbuffer_w = 0; gpu->gbuffer_h = 0; gpu->gbuffer_size = 0;
    /* Phase B tiled G-buffer */
    if (gpu->tiled_gbuffer_buf) { vkDestroyBuffer(gpu->device, gpu->tiled_gbuffer_buf, NULL); vkFreeMemory(gpu->device, gpu->tiled_gbuffer_mem, NULL); gpu->tiled_gbuffer_buf = VK_NULL_HANDLE; gpu->tiled_gbuffer_mem = VK_NULL_HANDLE; }
    gpu->tiled_gbuffer_w = 0; gpu->tiled_gbuffer_h = 0; gpu->tiled_gbuffer_size = 0;
    if (gpu->tlas)               { vkDestroyAccelerationStructureKHR(gpu->device, gpu->tlas, NULL); gpu->tlas = VK_NULL_HANDLE; }
    if (gpu->tlas_buf)           { vkDestroyBuffer(gpu->device, gpu->tlas_buf, NULL); vkFreeMemory(gpu->device, gpu->tlas_mem, NULL); gpu->tlas_buf = VK_NULL_HANDLE; gpu->tlas_mem = VK_NULL_HANDLE; }
    if (gpu->instance_buf)       { vkDestroyBuffer(gpu->device, gpu->instance_buf, NULL); vkFreeMemory(gpu->device, gpu->instance_mem, NULL); gpu->instance_buf = VK_NULL_HANDLE; gpu->instance_mem = VK_NULL_HANDLE; }
    for (uint32_t i = 0; i < gpu->blas_count; i++) {
        if (gpu->blas_list && gpu->blas_list[i])
            vkDestroyAccelerationStructureKHR(gpu->device, gpu->blas_list[i], NULL);
    }
    free(gpu->blas_list); gpu->blas_list = NULL;
    if (gpu->blas_pool_buf) { vkDestroyBuffer(gpu->device, gpu->blas_pool_buf, NULL); gpu->blas_pool_buf = VK_NULL_HANDLE; }
    if (gpu->blas_pool_mem) { vkFreeMemory(gpu->device, gpu->blas_pool_mem, NULL); gpu->blas_pool_mem = VK_NULL_HANDLE; }
    if (gpu->blas_extra_pool_buf) { vkDestroyBuffer(gpu->device, gpu->blas_extra_pool_buf, NULL); gpu->blas_extra_pool_buf = VK_NULL_HANDLE; }
    if (gpu->blas_extra_pool_mem) { vkFreeMemory(gpu->device, gpu->blas_extra_pool_mem, NULL); gpu->blas_extra_pool_mem = VK_NULL_HANDLE; }
    gpu->blas_count = 0;

    /* Persistent TLAS update buffers */
    if (gpu->tlas_update_scratch_buf) { vkDestroyBuffer(gpu->device, gpu->tlas_update_scratch_buf, NULL); vkFreeMemory(gpu->device, gpu->tlas_update_scratch_mem, NULL); gpu->tlas_update_scratch_buf = VK_NULL_HANDLE; gpu->tlas_update_scratch_mem = VK_NULL_HANDLE; }
    if (gpu->tlas_update_staging_buf) {
        /* perf/mem-pool: when mem == VK_NULL_HANDLE the buffer was
         * sub-allocated from the staging pool — vkUnmapMemory on the
         * pool's persistent map is the wrong action and would orphan
         * the pool. The pool will be reset (cursor=0) at the end of
         * gpu_destroy_rt_scene; until then the buffer's range is
         * still backed. We just destroy the VkBuffer handle. */
        if (gpu->tlas_update_staging_mem) {
            if (gpu->tlas_update_staging_mapped)
                vkUnmapMemory(gpu->device, gpu->tlas_update_staging_mem);
            vkFreeMemory(gpu->device, gpu->tlas_update_staging_mem, NULL);
        }
        vkDestroyBuffer(gpu->device, gpu->tlas_update_staging_buf, NULL);
        gpu->tlas_update_staging_buf    = VK_NULL_HANDLE;
        gpu->tlas_update_staging_mem    = VK_NULL_HANDLE;
        gpu->tlas_update_staging_mapped = NULL;
        gpu->tlas_update_staging_size   = 0;
    }

    free(gpu->blas_addresses);  gpu->blas_addresses  = NULL;
    free(gpu->mesh_to_blas);    gpu->mesh_to_blas    = NULL;
    free(gpu->instance_custom); gpu->instance_custom = NULL;
    free(gpu->instance_mask);   gpu->instance_mask   = NULL;
    gpu->instance_mask_count = 0;
    gpu->tlas_instance_count = 0;

    /* Phase C: tear down partitioned TLAS array. mesh_to_env / num_envs
     * persist (set via gpu_set_env_partition by the renderer); the next
     * gpu_build_rt_scene will rebuild the array. */
    gpu_destroy_partitioned_tlases(gpu);
    gpu->rt_desc_set = VK_NULL_HANDLE;
    gpu->rt_image_w = 0;
    gpu->rt_image_h = 0;

    /* perf/mem-pool: every VkBuffer that was bound into the resident /
     * transient / staging pools has now been destroyed (or will be by
     * curve_destroy below). Reset the cursors so the next build_accel
     * starts at offset 0. We also reset the resident pool because the
     * new build's resident buffers (segs, colors, AABBs, BLAS pool,
     * TLAS, scene_data...) all get fresh allocations. */
    if (gpu->resident_pool)  gpu_pool_reset((GpuMemPool*)gpu->resident_pool);
    if (gpu->transient_pool) gpu_pool_reset((GpuMemPool*)gpu->transient_pool);
    if (gpu->staging_pool)   gpu_pool_reset((GpuMemPool*)gpu->staging_pool);

    /* Phase 11.5.3: curve resources are part of the RT scene; free
     * them here so scene reloads don't leave stale BLAS handles
     * pointing at freed memory once 11.A.2.5 wires them into TLAS. */
    curve_destroy(gpu);

    /* Tiled RT interop + readback staging (double-buffered) + depth + segmentation + normals */
    tiled_destroy_interop(gpu);
    tiled_destroy_readback(gpu);
    tiled_destroy_depth(gpu);
    tiled_destroy_segmentation(gpu);
    tiled_destroy_normals(gpu);
    if (gpu->tiled_rt_pipeline)        { vkDestroyPipeline(gpu->device, gpu->tiled_rt_pipeline, NULL); gpu->tiled_rt_pipeline = VK_NULL_HANDLE; }
    if (gpu->tiled_rt_pipeline_layout) { vkDestroyPipelineLayout(gpu->device, gpu->tiled_rt_pipeline_layout, NULL); gpu->tiled_rt_pipeline_layout = VK_NULL_HANDLE; }
    if (gpu->tiled_sbt_buf)            { vkDestroyBuffer(gpu->device, gpu->tiled_sbt_buf, NULL); vkFreeMemory(gpu->device, gpu->tiled_sbt_mem, NULL); gpu->tiled_sbt_buf = VK_NULL_HANDLE; gpu->tiled_sbt_mem = VK_NULL_HANDLE; }
    if (gpu->tiled_rt_desc_pool)       { vkDestroyDescriptorPool(gpu->device, gpu->tiled_rt_desc_pool, NULL); gpu->tiled_rt_desc_pool = VK_NULL_HANDLE; }
    if (gpu->tiled_rt_desc_layout)     { vkDestroyDescriptorSetLayout(gpu->device, gpu->tiled_rt_desc_layout, NULL); gpu->tiled_rt_desc_layout = VK_NULL_HANDLE; }
    gpu->tiled_rt_desc_set = VK_NULL_HANDLE;
    gpu->tiled_rt_desc_set_b = VK_NULL_HANDLE;
    gpu->tiled_rt_built = 0;
    memset(&gpu->tiled_sbt_rgen, 0, sizeof(gpu->tiled_sbt_rgen));
    memset(&gpu->tiled_sbt_miss, 0, sizeof(gpu->tiled_sbt_miss));
    memset(&gpu->tiled_sbt_hit,  0, sizeof(gpu->tiled_sbt_hit));
    memset(&gpu->tiled_sbt_call, 0, sizeof(gpu->tiled_sbt_call));

    gpu->rt_built = 0;
}

/* ---- perf/cmdbuf-cache: external invalidation -------------------------- */

void gpu_invalidate_rt_cmd_cache(Gpu* gpu)
{
    if (!gpu) return;
    gpu->rt_cmd_cache_dirty = 1;
    if (gpu->rt_cmd_cache_valid) {
        for (int i = 0; i < gpu->rt_cmd_cache_valid_count; i++)
            gpu->rt_cmd_cache_valid[i] = 0;
    }
}

void gpu_invalidate_tiled_cmd_cache(Gpu* gpu)
{
    if (!gpu) return;
    gpu->tiled_cmd_cache_dirty = 1;
    gpu->tiled_cmd_cache_valid[0] = 0;
    gpu->tiled_cmd_cache_valid[1] = 0;
}

void gpu_get_cmd_cache_stats(Gpu* gpu,
                             uint64_t* out_rt_replays,
                             uint64_t* out_rt_records,
                             uint64_t* out_tiled_replays,
                             uint64_t* out_tiled_records)
{
    if (!gpu) {
        if (out_rt_replays)    *out_rt_replays    = 0;
        if (out_rt_records)    *out_rt_records    = 0;
        if (out_tiled_replays) *out_tiled_replays = 0;
        if (out_tiled_records) *out_tiled_records = 0;
        return;
    }
    if (out_rt_replays)    *out_rt_replays    = gpu->rt_cmd_cache_replays;
    if (out_rt_records)    *out_rt_records    = gpu->rt_cmd_cache_records;
    if (out_tiled_replays) *out_tiled_replays = gpu->tiled_cmd_cache_replays;
    if (out_tiled_records) *out_tiled_records = gpu->tiled_cmd_cache_records;
}

int gpu_begin_frame_rt(Gpu* gpu)
{
    uint32_t fi = gpu->frame_index;
    vkWaitForFences(gpu->device, 1, &gpu->in_flight[fi], VK_TRUE, UINT64_MAX);

    if (gpu->headless) {
        gpu->current_image = gpu->surfaceless && gpu->image_count
            ? (fi % gpu->image_count) : 0;
    } else {
        VkResult result = vkAcquireNextImageKHR(
            gpu->device, gpu->swapchain, UINT64_MAX,
            gpu->image_available[fi], VK_NULL_HANDLE, &gpu->current_image);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) return 0;
    }

    vkResetFences(gpu->device, 1, &gpu->in_flight[fi]);

    VkCommandBuffer cmd = gpu->command_buffers[gpu->current_image];
    gpu->current_cmd = cmd;

    /* Resize check happens BEFORE we decide whether to begin a new
     * recording or replay a cached one — a resize tears down the storage
     * image and invalidates every cached cmd buffer. */
    if (gpu->rt_image_w != gpu->swapchain_extent.width ||
        gpu->rt_image_h != gpu->swapchain_extent.height) {
        rt_create_storage_image(gpu);

        /* Update descriptor for new storage image */
        VkDescriptorImageInfo img_info = {0};
        img_info.imageView   = gpu->rt_image_view;
        img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet write = {0};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = gpu->rt_desc_set;
        write.dstBinding      = 1;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo      = &img_info;
        vkUpdateDescriptorSets(gpu->device, 1, &write, 0, NULL);

        gpu_invalidate_rt_cmd_cache(gpu);
    }

    /* If the cache is dirty (camera moved, scene rebuilt, etc.), clear it
     * so we re-record this frame.  This is independent of the per-image
     * validity bit — once we've successfully re-recorded all images, the
     * dirty flag is cleared in gpu_end_frame_rt(). */
    if (gpu->rt_cmd_cache_dirty && gpu->rt_cmd_cache_valid) {
        for (int i = 0; i < gpu->rt_cmd_cache_valid_count; i++)
            gpu->rt_cmd_cache_valid[i] = 0;
    }

    /* Defer Reset+Begin until gpu_cmd_trace_rays(): only then do we know
     * the push-constant payload and can decide whether to replay. */
    gpu->rt_cmd_replay_active = 0;

    return 1;
}

/* Comparison helper — bytewise equality of push constants.  The struct
 * is plain POD with float/uint32 fields so memcmp is safe. */
static int rt_push_consts_equal(const GpuRtPushConstants* a,
                                const GpuRtPushConstants* b)
{
    return memcmp(a, b, sizeof(GpuRtPushConstants)) == 0;
}

void gpu_cmd_trace_rays(Gpu* gpu, const GpuRtPushConstants* pc)
{
    /* Clay-viz opt-in (NUSD_CLAY_VIZ=1): set bit 30 of tone_flags so the
     * rchit can override material shading for diagnostics. Routed through a
     * local copy so the cmd-cache comparison + push both see it. */
    GpuRtPushConstants pc_eff = *pc;
    if (rt_env_enabled_default_off("NUSD_CLAY_VIZ"))
        pc_eff.tone_flags |= 0x40000000u;
    if (rt_env_enabled_default_off("NUSD_RT_SKIP_SECONDARY_VISIBILITY"))
        pc_eff.tone_flags |= 0x20000000u;
    if (rt_env_enabled_default_off("NUSD_RT_SKIP_AO_VISIBILITY"))
        pc_eff.tone_flags |= 0x10000000u;
    if (rt_env_enabled_default_off("NUSD_RT_SKIP_DIRECT_SHADOWS"))
        pc_eff.tone_flags |= 0x08000000u;
    if (rt_env_enabled_default_off("NUSD_RT_RECT_SHARED_SHADOWS"))
        pc_eff.tone_flags |= 0x04000000u;
    pc = &pc_eff;

    int img = (int)gpu->current_image;
    int can_replay =
        (gpu->rt_cmd_cache_valid != NULL) &&
        (img < gpu->rt_cmd_cache_valid_count) &&
        (gpu->rt_cmd_cache_valid[img]) &&
        (rt_push_consts_equal(pc, &gpu->rt_cmd_cache_pc[img]));

    if (can_replay) {
        /* Cache hit: skip recording entirely.  gpu_end_frame_rt() will
         * see rt_cmd_replay_active==1 and submit the existing buffer. */
        gpu->rt_cmd_replay_active = 1;
        return;
    }

    /* Cache miss: record from scratch. */
    VkCommandBuffer cmd = gpu->current_cmd;
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &begin_info);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, gpu->rt_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            gpu->rt_pipeline_layout, 0, 1, &gpu->rt_desc_set, 0, NULL);
    vkCmdPushConstants(cmd, gpu->rt_pipeline_layout,
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                       0, sizeof(GpuRtPushConstants), pc);

    /* perf/vk-instrumentation: label + timestamp the trace dispatch.
     * Resolved at the start of the next gpu_readback_pixels (after the
     * per-frame fence wait) so the host has settled measurements. */
    gpu_phase_begin(gpu, cmd, GPU_PHASE_RT_DISPATCH, "RT_dispatch");
    vkCmdTraceRaysKHR(cmd,
                         &gpu->sbt_rgen, &gpu->sbt_miss, &gpu->sbt_hit, &gpu->sbt_call,
                         gpu->swapchain_extent.width, gpu->swapchain_extent.height, 1);
    gpu_phase_end(gpu, cmd, GPU_PHASE_RT_DISPATCH);

    /* Stash the pc that this recording corresponds to. */
    if (gpu->rt_cmd_cache_pc && img < gpu->rt_cmd_cache_valid_count)
        gpu->rt_cmd_cache_pc[img] = *pc;
}

void gpu_end_frame_rt(Gpu* gpu)
{
    uint32_t fi = gpu->frame_index;
    VkCommandBuffer cmd = gpu->current_cmd;
    int img = (int)gpu->current_image;

    /* Cache-replay fast path: we never reset/began the cmd buffer in
     * this frame, so jump straight to submission. */
    if (gpu->rt_cmd_replay_active) {
        gpu->rt_cmd_cache_replays++;

        VkSubmitInfo submit = {0};
        submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers    = &cmd;

        if (gpu->headless) {
            vkQueueSubmit(gpu->graphics_queue, 1, &submit, gpu->in_flight[fi]);
        } else {
            VkSemaphore wait_sems[]   = { gpu->image_available[fi] };
            VkSemaphore signal_sems[] = { gpu->render_finished[fi] };
            VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR };
            submit.waitSemaphoreCount   = 1;
            submit.pWaitSemaphores      = wait_sems;
            submit.pWaitDstStageMask    = wait_stages;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores    = signal_sems;

            vkQueueSubmit(gpu->graphics_queue, 1, &submit, gpu->in_flight[fi]);

            VkPresentInfoKHR present = {0};
            present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            present.waitSemaphoreCount = 1;
            present.pWaitSemaphores    = signal_sems;
            present.swapchainCount     = 1;
            present.pSwapchains        = &gpu->swapchain;
            present.pImageIndices      = &gpu->current_image;

            vkQueuePresentKHR(gpu->graphics_queue, &present);
        }
        gpu->frame_index = (gpu->frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
        return;
    }

    /* Cache miss: record the rest of the buffer normally, then submit.
     * Once submitted, mark this image's cache slot valid so subsequent
     * frames can replay (provided no external invalidation). */

    /* Barrier: RT writes → transfer read */
    VkImageMemoryBarrier barriers[2] = {0};

    /* Storage image: GENERAL → TRANSFER_SRC */
    barriers[0].sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[0].srcAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;
    barriers[0].dstAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].oldLayout                       = VK_IMAGE_LAYOUT_GENERAL;
    barriers[0].newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image                           = gpu->rt_image;
    barriers[0].subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barriers[0].subresourceRange.levelCount     = 1;
    barriers[0].subresourceRange.layerCount     = 1;

    /* Swapchain image: UNDEFINED → TRANSFER_DST */
    barriers[1].sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[1].srcAccessMask                   = 0;
    barriers[1].dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[1].newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].image                           = gpu->swapchain_images[gpu->current_image];
    barriers[1].subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barriers[1].subresourceRange.levelCount     = 1;
    barriers[1].subresourceRange.layerCount     = 1;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 2, barriers);

    /* Copy storage image → swapchain image */
    VkImageCopy region = {0};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.extent.width  = gpu->swapchain_extent.width;
    region.extent.height = gpu->swapchain_extent.height;
    region.extent.depth  = 1;

    /* Use blit for format conversion (storage is RGBA8_UNORM, swapchain may be BGRA8_SRGB) */
    VkImageBlit blit = {0};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[1].x = (int32_t)gpu->swapchain_extent.width;
    blit.srcOffsets[1].y = (int32_t)gpu->swapchain_extent.height;
    blit.srcOffsets[1].z = 1;
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[1].x = (int32_t)gpu->swapchain_extent.width;
    blit.dstOffsets[1].y = (int32_t)gpu->swapchain_extent.height;
    blit.dstOffsets[1].z = 1;

    vkCmdBlitImage(cmd,
                   gpu->rt_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   gpu->swapchain_images[gpu->current_image], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &blit, VK_FILTER_NEAREST);

    /* Transition storage image back to GENERAL for next frame */
    VkImageMemoryBarrier restore[2] = {0};
    restore[0].sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    restore[0].srcAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;
    restore[0].dstAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;
    restore[0].oldLayout                       = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    restore[0].newLayout                       = VK_IMAGE_LAYOUT_GENERAL;
    restore[0].srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    restore[0].dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    restore[0].image                           = gpu->rt_image;
    restore[0].subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    restore[0].subresourceRange.levelCount     = 1;
    restore[0].subresourceRange.layerCount     = 1;

    /* Color target: TRANSFER_DST → readable final layout. */
    restore[1].sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    restore[1].srcAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;
    restore[1].dstAccessMask                   = 0;
    restore[1].oldLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    restore[1].newLayout                       = gpu->surfaceless
        ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    restore[1].srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    restore[1].dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    restore[1].image                           = gpu->swapchain_images[gpu->current_image];
    restore[1].subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    restore[1].subresourceRange.levelCount     = 1;
    restore[1].subresourceRange.layerCount     = 1;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, NULL, 0, NULL, 2, restore);

    /* Overlay text (after blit to swapchain, before submit).
     * Overlay flush is a no-op when overlay_vert_count == 0; if any
     * caller did add overlay text, we conservatively avoid caching this
     * frame because the next frame's overlay verts may differ. */
    int overlay_active = gpu->overlay_inited && gpu->overlay_vert_count != 0;
    gpu_overlay_flush(gpu);

    vkEndCommandBuffer(cmd);

    /* Mark cache valid: this recording is replayable until invalidated.
     * Skip if overlay was active (overlay verts change per call) or if the
     * cache shadow array hasn't been allocated.  We clear rt_cmd_cache_dirty
     * here because a successful recording supersedes any external
     * invalidation that happened earlier in this frame (e.g. resize). */
    if (!overlay_active && gpu->rt_cmd_cache_valid &&
        img < gpu->rt_cmd_cache_valid_count) {
        gpu->rt_cmd_cache_valid[img] = 1;
        gpu->rt_cmd_cache_dirty = 0;
    } else if (gpu->rt_cmd_cache_valid &&
               img < gpu->rt_cmd_cache_valid_count) {
        gpu->rt_cmd_cache_valid[img] = 0;
    }
    gpu->rt_cmd_cache_records++;

    /* Submit */
    VkSubmitInfo submit = {0};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;

    if (gpu->headless) {
        vkQueueSubmit(gpu->graphics_queue, 1, &submit, gpu->in_flight[fi]);
    } else {
        VkSemaphore wait_sems[]   = { gpu->image_available[fi] };
        VkSemaphore signal_sems[] = { gpu->render_finished[fi] };
        VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR };
        submit.waitSemaphoreCount   = 1;
        submit.pWaitSemaphores      = wait_sems;
        submit.pWaitDstStageMask    = wait_stages;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores    = signal_sems;

        vkQueueSubmit(gpu->graphics_queue, 1, &submit, gpu->in_flight[fi]);

        VkPresentInfoKHR present = {0};
        present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores    = signal_sems;
        present.swapchainCount     = 1;
        present.pSwapchains        = &gpu->swapchain;
        present.pImageIndices      = &gpu->current_image;

        vkQueuePresentKHR(gpu->graphics_queue, &present);
    }

    gpu->frame_index = (gpu->frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
}

/* ========================================================================== */
/* DLSS — offscreen targets, MRT render pass, frame functions                 */
/* ========================================================================== */

/* Helper: create a device-local image + view at given format and dimensions. */
static void dlss_create_image(Gpu* gpu, uint32_t w, uint32_t h,
                              VkFormat format, VkImageUsageFlags usage,
                              VkImageAspectFlags aspect,
                              VkImage* out_image, VkDeviceMemory* out_mem,
                              VkImageView* out_view)
{
    VkImageCreateInfo ici = {0};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = format;
    ici.extent.width  = w;
    ici.extent.height = h;
    ici.extent.depth  = 1;
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = usage;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkCreateImage(gpu->device, &ici, NULL, out_image);

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(gpu->device, *out_image, &req);

    VkMemoryAllocateInfo ai = {0};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    gpu_alloc_memory(gpu, &ai, out_mem);
    vkBindImageMemory(gpu->device, *out_image, *out_mem, 0);

    VkImageViewCreateInfo vci = {0};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = *out_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = format;
    vci.subresourceRange.aspectMask     = aspect;
    vci.subresourceRange.baseMipLevel   = 0;
    vci.subresourceRange.levelCount     = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount     = 1;

    vkCreateImageView(gpu->device, &vci, NULL, out_view);
}

/* Helper: destroy image + mem + view if they exist. */
static void dlss_destroy_image(Gpu* gpu, VkImage* img, VkDeviceMemory* mem, VkImageView* view)
{
    if (*view) { vkDestroyImageView(gpu->device, *view, NULL); *view = VK_NULL_HANDLE; }
    if (*img)  { vkDestroyImage(gpu->device, *img, NULL);      *img  = VK_NULL_HANDLE; }
    if (*mem)  { vkFreeMemory(gpu->device, *mem, NULL);         *mem  = VK_NULL_HANDLE; }
}

/* Transition an image layout using a one-shot command buffer. */
static void dlss_transition_image(Gpu* gpu, VkImage image,
                                  VkImageLayout old_layout, VkImageLayout new_layout,
                                  VkAccessFlags src_access, VkAccessFlags dst_access,
                                  VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                                  VkImageAspectFlags aspect)
{
    VkCommandBuffer cmd = rt_begin_cmd(gpu);

    VkImageMemoryBarrier barrier = {0};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                       = old_layout;
    barrier.newLayout                       = new_layout;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                           = image;
    barrier.subresourceRange.aspectMask     = aspect;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;
    barrier.srcAccessMask                   = src_access;
    barrier.dstAccessMask                   = dst_access;

    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &barrier);
    rt_end_cmd(gpu, cmd);
}

static void dlss_create_offscreen_targets(Gpu* gpu)
{
    uint32_t rw = gpu->dlss_render_w;
    uint32_t rh = gpu->dlss_render_h;
    uint32_t dw = gpu->swapchain_extent.width;
    uint32_t dh = gpu->swapchain_extent.height;

    /* Color target (render res) — RGBA8, used as color attachment + DLSS input */
    dlss_create_image(gpu, rw, rh,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        &gpu->dlss_color, &gpu->dlss_color_mem, &gpu->dlss_color_view);

    /* Motion vector target (render res) — RG16F */
    dlss_create_image(gpu, rw, rh,
        VK_FORMAT_R16G16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        &gpu->dlss_mv, &gpu->dlss_mv_mem, &gpu->dlss_mv_view);

    /* Depth attachment (render res) — D32F, for depth testing in offscreen render pass.
     * SAMPLED_BIT allows DLSS to read the depth directly (no copy to R32F needed). */
    dlss_create_image(gpu, rw, rh,
        VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        &gpu->dlss_depth_att, &gpu->dlss_depth_att_mem, &gpu->dlss_depth_att_view);

    /* Shader-readable depth (render res) — R32F, copy target from D32F */
    dlss_create_image(gpu, rw, rh,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        &gpu->dlss_depth, &gpu->dlss_depth_mem, &gpu->dlss_depth_view);

    /* DLSS output (display res) — RGBA8 */
    dlss_create_image(gpu, dw, dh,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        &gpu->dlss_output, &gpu->dlss_output_mem, &gpu->dlss_output_view);

    /* Transition images to usable layouts */
    dlss_transition_image(gpu, gpu->dlss_color,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    dlss_transition_image(gpu, gpu->dlss_mv,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    dlss_transition_image(gpu, gpu->dlss_depth,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    dlss_transition_image(gpu, gpu->dlss_output,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    /* RT storage images at render resolution (for DLSS RT path) */
    if (gpu->rt_available) {
        dlss_create_image(gpu, rw, rh,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            &gpu->dlss_rt_color, &gpu->dlss_rt_color_mem, &gpu->dlss_rt_color_view);

        dlss_create_image(gpu, rw, rh,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            &gpu->dlss_rt_depth, &gpu->dlss_rt_depth_mem, &gpu->dlss_rt_depth_view);

        dlss_create_image(gpu, rw, rh,
            VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            &gpu->dlss_rt_mv, &gpu->dlss_rt_mv_mem, &gpu->dlss_rt_mv_view);

        /* Transition RT images to GENERAL for shader writes */
        dlss_transition_image(gpu, gpu->dlss_rt_color,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_IMAGE_ASPECT_COLOR_BIT);

        dlss_transition_image(gpu, gpu->dlss_rt_depth,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_IMAGE_ASPECT_COLOR_BIT);

        dlss_transition_image(gpu, gpu->dlss_rt_mv,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

static void dlss_destroy_offscreen_targets(Gpu* gpu)
{
    dlss_destroy_image(gpu, &gpu->dlss_color, &gpu->dlss_color_mem, &gpu->dlss_color_view);
    dlss_destroy_image(gpu, &gpu->dlss_mv, &gpu->dlss_mv_mem, &gpu->dlss_mv_view);
    dlss_destroy_image(gpu, &gpu->dlss_depth, &gpu->dlss_depth_mem, &gpu->dlss_depth_view);
    dlss_destroy_image(gpu, &gpu->dlss_depth_att, &gpu->dlss_depth_att_mem, &gpu->dlss_depth_att_view);
    dlss_destroy_image(gpu, &gpu->dlss_output, &gpu->dlss_output_mem, &gpu->dlss_output_view);
    dlss_destroy_image(gpu, &gpu->dlss_rt_color, &gpu->dlss_rt_color_mem, &gpu->dlss_rt_color_view);
    dlss_destroy_image(gpu, &gpu->dlss_rt_depth, &gpu->dlss_rt_depth_mem, &gpu->dlss_rt_depth_view);
    dlss_destroy_image(gpu, &gpu->dlss_rt_mv, &gpu->dlss_rt_mv_mem, &gpu->dlss_rt_mv_view);
}

/* Create MRT render pass: color (RGBA8) + motion vectors (RG16F) + depth (D32F). */
static void dlss_create_render_pass(Gpu* gpu)
{
    VkAttachmentDescription attachments[3] = {0};

    /* 0: Color */
    attachments[0].format         = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].finalLayout    = VK_IMAGE_LAYOUT_GENERAL;

    /* 1: Motion vectors */
    attachments[1].format         = VK_FORMAT_R16G16_SFLOAT;
    attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[1].finalLayout    = VK_IMAGE_LAYOUT_GENERAL;

    /* 2: Depth */
    attachments[2].format         = VK_FORMAT_D32_SFLOAT;
    attachments[2].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[2].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[2].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[2].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_refs[2];
    color_refs[0].attachment = 0;
    color_refs[0].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_refs[1].attachment = 1;
    color_refs[1].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_ref = {0};
    depth_ref.attachment = 2;
    depth_ref.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {0};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 2;
    subpass.pColorAttachments       = color_refs;
    subpass.pDepthStencilAttachment = &depth_ref;

    VkSubpassDependency dep = {0};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci = {0};
    rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 3;
    rpci.pAttachments    = attachments;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies   = &dep;

    if (vkCreateRenderPass(gpu->device, &rpci, NULL, &gpu->dlss_render_pass) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create DLSS render pass\n");
    }
}

static void dlss_create_framebuffer(Gpu* gpu)
{
    if (gpu->dlss_framebuffer) {
        vkDestroyFramebuffer(gpu->device, gpu->dlss_framebuffer, NULL);
        gpu->dlss_framebuffer = VK_NULL_HANDLE;
    }

    VkImageView att[3] = {
        gpu->dlss_color_view,
        gpu->dlss_mv_view,
        gpu->dlss_depth_att_view,
    };

    VkFramebufferCreateInfo fci = {0};
    fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fci.renderPass      = gpu->dlss_render_pass;
    fci.attachmentCount = 3;
    fci.pAttachments    = att;
    fci.width           = gpu->dlss_render_w;
    fci.height          = gpu->dlss_render_h;
    fci.layers          = 1;

    vkCreateFramebuffer(gpu->device, &fci, NULL, &gpu->dlss_framebuffer);
}

/* ---- Pipeline layout for DLSS (208-byte push constants) ---- */

static void dlss_create_pipeline_layout(Gpu* gpu)
{
    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset     = 0;
    pcr.size       = PUSH_CONSTANT_SIZE_DLSS;

    VkPipelineLayoutCreateInfo plci = {0};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;

    vkCreatePipelineLayout(gpu->device, &plci, NULL, &gpu->dlss_pipeline_layout);
}

/* ---- Public DLSS API ---- */

int gpu_dlss_available(Gpu* gpu)
{
    if (!gpu) return 0;
    return dlss_sr_available((DlssContext*)gpu->dlss_ctx);
}

int gpu_dlss_init(Gpu* gpu, int quality_mode)
{
    if (!gpu) return 0;

    /* Initialize DLSS context if not already done */
    if (!gpu->dlss_ctx) {
        DlssInitInfo info = {0};
        info.vk_instance        = (void*)gpu->instance;
        info.vk_physical_device = (void*)gpu->physical_device;
        info.vk_device          = (void*)gpu->device;
        info.queue_family       = gpu->queue_family;
        info.vk_queue           = (void*)gpu->graphics_queue;

        gpu->dlss_ctx = dlss_init(&info);
        if (!gpu->dlss_ctx) {
            fprintf(stderr, "gpu_vulkan: DLSS initialization failed\n");
            return 0;
        }
    }

    /* Query optimal render resolution */
    DlssOptimalSettings settings = {0};
    settings.display_width  = gpu->swapchain_extent.width;
    settings.display_height = gpu->swapchain_extent.height;
    settings.quality        = (DlssQualityMode)quality_mode;

    if (!dlss_get_optimal_settings((DlssContext*)gpu->dlss_ctx, &settings)) {
        fprintf(stderr, "gpu_vulkan: DLSS optimal settings query failed\n");
        return 0;
    }

    gpu->dlss_render_w    = settings.render_width;
    gpu->dlss_render_h    = settings.render_height;
    gpu->dlss_quality_mode = quality_mode;

    /* Create offscreen resources */
    dlss_create_offscreen_targets(gpu);
    dlss_create_render_pass(gpu);
    dlss_create_framebuffer(gpu);
    dlss_create_pipeline_layout(gpu);

    /* Create the DLSS feature (needs a recording command buffer) */
    VkCommandBuffer cmd = rt_begin_cmd(gpu);
    int ok = dlss_create_feature((DlssContext*)gpu->dlss_ctx, &settings, (void*)cmd);
    rt_end_cmd(gpu, cmd);

    if (!ok) {
        fprintf(stderr, "gpu_vulkan: DLSS feature creation failed\n");
        dlss_destroy_offscreen_targets(gpu);
        return 0;
    }

    gpu->dlss_active = 1;
    fprintf(stderr, "gpu_vulkan: DLSS enabled (%s, %ux%u -> %ux%u)\n",
            dlss_quality_names[quality_mode],
            gpu->dlss_render_w, gpu->dlss_render_h,
            gpu->swapchain_extent.width, gpu->swapchain_extent.height);
    return 1;
}

void gpu_dlss_shutdown(Gpu* gpu)
{
    if (!gpu || !gpu->dlss_active) return;

    vkDeviceWaitIdle(gpu->device);

    dlss_release_feature((DlssContext*)gpu->dlss_ctx);

    dlss_destroy_offscreen_targets(gpu);

    if (gpu->dlss_framebuffer) {
        vkDestroyFramebuffer(gpu->device, gpu->dlss_framebuffer, NULL);
        gpu->dlss_framebuffer = VK_NULL_HANDLE;
    }
    if (gpu->dlss_render_pass) {
        vkDestroyRenderPass(gpu->device, gpu->dlss_render_pass, NULL);
        gpu->dlss_render_pass = VK_NULL_HANDLE;
    }
    if (gpu->dlss_pipeline_layout) {
        vkDestroyPipelineLayout(gpu->device, gpu->dlss_pipeline_layout, NULL);
        gpu->dlss_pipeline_layout = VK_NULL_HANDLE;
    }

    /* Destroy DLSS RT pipeline resources */
    if (gpu->dlss_rt_pipeline) {
        vkDestroyPipeline(gpu->device, gpu->dlss_rt_pipeline, NULL);
        gpu->dlss_rt_pipeline = VK_NULL_HANDLE;
    }
    if (gpu->dlss_rt_pipeline_layout) {
        vkDestroyPipelineLayout(gpu->device, gpu->dlss_rt_pipeline_layout, NULL);
        gpu->dlss_rt_pipeline_layout = VK_NULL_HANDLE;
    }
    if (gpu->dlss_rt_desc_pool) {
        vkDestroyDescriptorPool(gpu->device, gpu->dlss_rt_desc_pool, NULL);
        gpu->dlss_rt_desc_pool = VK_NULL_HANDLE;
    }
    if (gpu->dlss_rt_desc_layout) {
        vkDestroyDescriptorSetLayout(gpu->device, gpu->dlss_rt_desc_layout, NULL);
        gpu->dlss_rt_desc_layout = VK_NULL_HANDLE;
    }
    if (gpu->dlss_sbt_buf) {
        vkDestroyBuffer(gpu->device, gpu->dlss_sbt_buf, NULL);
        vkFreeMemory(gpu->device, gpu->dlss_sbt_mem, NULL);
        gpu->dlss_sbt_buf = VK_NULL_HANDLE;
    }

    gpu->dlss_active = 0;
}

void gpu_dlss_set_quality(Gpu* gpu, int quality_mode)
{
    if (!gpu || !gpu->dlss_active) return;
    gpu_dlss_shutdown(gpu);
    gpu_dlss_init(gpu, quality_mode);
}

void gpu_get_render_extent(Gpu* gpu, uint32_t* w, uint32_t* h)
{
    if (gpu && gpu->dlss_active) {
        *w = gpu->dlss_render_w;
        *h = gpu->dlss_render_h;
    } else if (gpu) {
        *w = gpu->swapchain_extent.width;
        *h = gpu->swapchain_extent.height;
    }
}

/* ---- DLSS raster frame functions ---- */

int gpu_begin_frame_dlss(Gpu* gpu)
{
    uint32_t fi = gpu->frame_index;
    vkWaitForFences(gpu->device, 1, &gpu->in_flight[fi], VK_TRUE, UINT64_MAX);

    VkResult result = vkAcquireNextImageKHR(
        gpu->device, gpu->swapchain, UINT64_MAX,
        gpu->image_available[fi], VK_NULL_HANDLE, &gpu->current_image);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) return 0;

    vkResetFences(gpu->device, 1, &gpu->in_flight[fi]);

    VkCommandBuffer cmd = gpu->command_buffers[gpu->current_image];
    gpu->current_cmd = cmd;
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &begin_info);

    /* Set viewport and scissor to RENDER resolution */
    VkViewport viewport = {0};
    viewport.width    = (float)gpu->dlss_render_w;
    viewport.height   = (float)gpu->dlss_render_h;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = {0};
    scissor.extent.width  = gpu->dlss_render_w;
    scissor.extent.height = gpu->dlss_render_h;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    /* Begin offscreen MRT render pass */
    VkClearValue clear_values[3];
    clear_values[0].color.float32[0] = 0.72f;  /* sky horizon */
    clear_values[0].color.float32[1] = 0.72f;
    clear_values[0].color.float32[2] = 0.70f;
    clear_values[0].color.float32[3] = 1.0f;
    clear_values[1].color.float32[0] = 0.0f;  /* MV = zero for background */
    clear_values[1].color.float32[1] = 0.0f;
    clear_values[1].color.float32[2] = 0.0f;
    clear_values[1].color.float32[3] = 0.0f;
    clear_values[2].depthStencil.depth   = 1.0f;
    clear_values[2].depthStencil.stencil = 0;

    VkRenderPassBeginInfo rpbi = {0};
    rpbi.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass        = gpu->dlss_render_pass;
    rpbi.framebuffer       = gpu->dlss_framebuffer;
    rpbi.renderArea.extent.width  = gpu->dlss_render_w;
    rpbi.renderArea.extent.height = gpu->dlss_render_h;
    rpbi.clearValueCount   = 3;
    rpbi.pClearValues      = clear_values;

    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    return 1;
}

void gpu_end_frame_dlss(Gpu* gpu, float jitter_x, float jitter_y,
                        float dt_ms, int reset)
{
    uint32_t fi = gpu->frame_index;
    VkCommandBuffer cmd = gpu->current_cmd;

    vkCmdEndRenderPass(cmd);

    /* Transition depth attachment from DEPTH_STENCIL_ATTACHMENT to SHADER_READ
     * so DLSS can sample it directly (no D32F→R32F copy needed). */
    {
        VkImageMemoryBarrier barrier = {0};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barrier.newLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = gpu->dlss_depth_att;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask       = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, NULL, 0, NULL, 1, &barrier);
    }

    /* Run DLSS evaluation (inserts compute work into command buffer). */
    DlssEvalParams params = {0};
    params.color_image    = (void*)gpu->dlss_color;
    params.color_view     = (void*)gpu->dlss_color_view;
    params.depth_image    = (void*)gpu->dlss_depth_att;
    params.depth_view     = (void*)gpu->dlss_depth_att_view;
    params.depth_is_d32   = 1;
    params.mv_image       = (void*)gpu->dlss_mv;
    params.mv_view        = (void*)gpu->dlss_mv_view;
    params.output_image   = (void*)gpu->dlss_output;
    params.output_view    = (void*)gpu->dlss_output_view;
    params.cmd            = (void*)cmd;
    params.render_width   = gpu->dlss_render_w;
    params.render_height  = gpu->dlss_render_h;
    params.display_width  = gpu->swapchain_extent.width;
    params.display_height = gpu->swapchain_extent.height;
    params.jitter_x       = jitter_x;
    params.jitter_y       = jitter_y;
    params.delta_time_ms  = dt_ms;
    params.reset          = reset;

    int eval_ok = dlss_evaluate((DlssContext*)gpu->dlss_ctx, &params);
    (void)eval_ok;

    /* Blit DLSS output to swapchain */
    {
        VkImageMemoryBarrier barriers[2] = {0};

        /* DLSS output -> TRANSFER_SRC */
        barriers[0].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image               = gpu->dlss_output;
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.layerCount = 1;
        barriers[0].srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[0].dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;

        /* Swapchain -> TRANSFER_DST */
        barriers[1].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[1].oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[1].newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].image               = gpu->swapchain_images[gpu->current_image];
        barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[1].subresourceRange.levelCount = 1;
        barriers[1].subresourceRange.layerCount = 1;
        barriers[1].srcAccessMask       = 0;
        barriers[1].dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 0, NULL, 2, barriers);

        VkImageBlit blit = {0};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1].x = (int32_t)gpu->swapchain_extent.width;
        blit.srcOffsets[1].y = (int32_t)gpu->swapchain_extent.height;
        blit.srcOffsets[1].z = 1;
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1].x = (int32_t)gpu->swapchain_extent.width;
        blit.dstOffsets[1].y = (int32_t)gpu->swapchain_extent.height;
        blit.dstOffsets[1].z = 1;

        vkCmdBlitImage(cmd,
            gpu->dlss_output, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            gpu->swapchain_images[gpu->current_image], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_NEAREST);

        /* Restore layouts for next frame */
        VkImageMemoryBarrier restore[2] = {0};

        restore[0].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        restore[0].oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        restore[0].newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        restore[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[0].image               = gpu->dlss_output;
        restore[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        restore[0].subresourceRange.levelCount = 1;
        restore[0].subresourceRange.layerCount = 1;
        restore[0].srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
        restore[0].dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;

        restore[1].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        restore[1].oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        restore[1].newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        restore[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[1].image               = gpu->swapchain_images[gpu->current_image];
        restore[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        restore[1].subresourceRange.levelCount = 1;
        restore[1].subresourceRange.layerCount = 1;
        restore[1].srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        restore[1].dstAccessMask       = 0;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, NULL, 0, NULL, 2, restore);
    }

    /* Transition color + MV back to COLOR_ATTACHMENT for next frame */
    {
        VkImageMemoryBarrier barriers[2] = {0};

        barriers[0].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image               = gpu->dlss_color;
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.layerCount = 1;
        barriers[0].srcAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        barriers[0].dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        barriers[1].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[1].oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barriers[1].newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].image               = gpu->dlss_mv;
        barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[1].subresourceRange.levelCount = 1;
        barriers[1].subresourceRange.layerCount = 1;
        barriers[1].srcAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        barriers[1].dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, NULL, 0, NULL, 2, barriers);
    }

    /* Overlay text (after DLSS blit, before submit) */
    gpu_overlay_flush(gpu);

    vkEndCommandBuffer(cmd);

    /* Submit */
    VkSemaphore wait_sems[]   = { gpu->image_available[fi] };
    VkSemaphore signal_sems[] = { gpu->render_finished[fi] };
    VkPipelineStageFlags wait_stages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
    };

    VkSubmitInfo submit = {0};
    submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount   = 1;
    submit.pWaitSemaphores      = wait_sems;
    submit.pWaitDstStageMask    = wait_stages;
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = signal_sems;

    vkQueueSubmit(gpu->graphics_queue, 1, &submit, gpu->in_flight[fi]);

    /* Present (skip in headless mode) */
    if (!gpu->headless) {
        VkPresentInfoKHR present = {0};
        present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores    = signal_sems;
        present.swapchainCount     = 1;
        present.pSwapchains        = &gpu->swapchain;
        present.pImageIndices      = &gpu->current_image;

        vkQueuePresentKHR(gpu->graphics_queue, &present);
    }

    gpu->frame_index = (gpu->frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
}

/* END gpu_end_frame_dlss */

/* ---- DLSS RT frame functions ---- */

int gpu_begin_frame_rt_dlss(Gpu* gpu)
{
    uint32_t fi = gpu->frame_index;

    vkWaitForFences(gpu->device, 1, &gpu->in_flight[fi], VK_TRUE, UINT64_MAX);

    VkResult result = vkAcquireNextImageKHR(
        gpu->device, gpu->swapchain, UINT64_MAX,
        gpu->image_available[fi], VK_NULL_HANDLE, &gpu->current_image);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) return 0;

    vkResetFences(gpu->device, 1, &gpu->in_flight[fi]);

    VkCommandBuffer cmd = gpu->command_buffers[gpu->current_image];
    gpu->current_cmd = cmd;
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &begin_info);

    return 1;
}

void gpu_cmd_trace_rays_dlss(Gpu* gpu, const GpuRtPushConstantsDlss* pc)
{
    VkCommandBuffer cmd = gpu->current_cmd;

    /* Use the DLSS RT pipeline if available, otherwise fall back to standard */
    VkPipeline pipeline = gpu->dlss_rt_pipeline ? gpu->dlss_rt_pipeline : gpu->rt_pipeline;
    VkPipelineLayout layout = gpu->dlss_rt_pipeline_layout ? gpu->dlss_rt_pipeline_layout : gpu->rt_pipeline_layout;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

    VkDescriptorSet ds = gpu->dlss_rt_desc_set ? gpu->dlss_rt_desc_set : gpu->rt_desc_set;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            layout, 0, 1, &ds, 0, NULL);

    if (!gpu->dlss_rt_pipeline) {
        /* Fallback: regular pipeline uses GpuRtPushConstants layout.
         * Copy view_inv/proj_inv + ground_y/scene_scale to the regular layout. */
        GpuRtPushConstants regular_pc;
        memset(&regular_pc, 0, sizeof(regular_pc));
        memcpy(regular_pc.view_inv, pc->view_inv, sizeof(float) * 16);
        memcpy(regular_pc.proj_inv, pc->proj_inv, sizeof(float) * 16);
        regular_pc.ground_y    = pc->ground_y;
        regular_pc.scene_scale = pc->scene_scale;
        regular_pc.fast_mode              = 0;
        regular_pc.depth_enabled          = 0;
        regular_pc.segmentation_enabled   = 0;
        regular_pc.normals_enabled        = 0;
        regular_pc.deferred_shade_enabled = 0;
        regular_pc.tone_exposure_scale    = 1.0f;
        regular_pc.tone_sky_scale         = 1.0f;
        regular_pc.tone_white_point       = 1.0f;
        regular_pc.rt_ibl_fill_scale      = 1.0f;
        vkCmdPushConstants(cmd, layout,
                           VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                           0, (uint32_t)sizeof(GpuRtPushConstants), &regular_pc);
    } else {
        vkCmdPushConstants(cmd, layout,
                           VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                           0, (uint32_t)sizeof(GpuRtPushConstantsDlss), pc);
    }

    VkStridedDeviceAddressRegionKHR* rgen = gpu->dlss_sbt_buf ? &gpu->dlss_sbt_rgen : &gpu->sbt_rgen;
    VkStridedDeviceAddressRegionKHR* miss = gpu->dlss_sbt_buf ? &gpu->dlss_sbt_miss : &gpu->sbt_miss;
    VkStridedDeviceAddressRegionKHR* hit  = gpu->dlss_sbt_buf ? &gpu->dlss_sbt_hit  : &gpu->sbt_hit;
    VkStridedDeviceAddressRegionKHR* call = gpu->dlss_sbt_buf ? &gpu->dlss_sbt_call : &gpu->sbt_call;

    /* Trace at render resolution */
    vkCmdTraceRaysKHR(cmd, rgen, miss, hit, call,
                      gpu->dlss_render_w, gpu->dlss_render_h, 1);
}

void gpu_end_frame_rt_dlss(Gpu* gpu, float jitter_x, float jitter_y,
                           float dt_ms, int reset)
{
    uint32_t fi = gpu->frame_index;
    VkCommandBuffer cmd = gpu->current_cmd;

    /* The RT shaders wrote to dlss_rt_color, dlss_rt_depth, dlss_rt_mv
     * in GENERAL layout. Use these directly as DLSS input. */



    /* Run DLSS evaluation */
    DlssEvalParams params = {0};
    params.color_image    = (void*)gpu->dlss_rt_color;
    params.color_view     = (void*)gpu->dlss_rt_color_view;
    params.depth_image    = (void*)gpu->dlss_rt_depth;
    params.depth_view     = (void*)gpu->dlss_rt_depth_view;
    params.mv_image       = (void*)gpu->dlss_rt_mv;
    params.mv_view        = (void*)gpu->dlss_rt_mv_view;
    params.output_image   = (void*)gpu->dlss_output;
    params.output_view    = (void*)gpu->dlss_output_view;
    params.cmd            = (void*)cmd;
    params.render_width   = gpu->dlss_render_w;
    params.render_height  = gpu->dlss_render_h;
    params.display_width  = gpu->swapchain_extent.width;
    params.display_height = gpu->swapchain_extent.height;
    params.jitter_x       = jitter_x;
    params.jitter_y       = jitter_y;
    params.delta_time_ms  = dt_ms;
    params.reset          = reset;

    dlss_evaluate((DlssContext*)gpu->dlss_ctx, &params);

    /* Blit DLSS output to swapchain (same as raster DLSS path) */
    {
        VkImageMemoryBarrier barriers[2] = {0};

        barriers[0].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image               = gpu->dlss_output;
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.layerCount = 1;
        barriers[0].srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[0].dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;

        barriers[1].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[1].oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[1].newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[1].image               = gpu->swapchain_images[gpu->current_image];
        barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[1].subresourceRange.levelCount = 1;
        barriers[1].subresourceRange.layerCount = 1;
        barriers[1].srcAccessMask       = 0;
        barriers[1].dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 0, NULL, 2, barriers);

        VkImageBlit blit = {0};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1].x = (int32_t)gpu->swapchain_extent.width;
        blit.srcOffsets[1].y = (int32_t)gpu->swapchain_extent.height;
        blit.srcOffsets[1].z = 1;
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1].x = (int32_t)gpu->swapchain_extent.width;
        blit.dstOffsets[1].y = (int32_t)gpu->swapchain_extent.height;
        blit.dstOffsets[1].z = 1;

        vkCmdBlitImage(cmd,
            gpu->dlss_output, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            gpu->swapchain_images[gpu->current_image], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_NEAREST);

        /* Restore layouts */
        VkImageMemoryBarrier restore[2] = {0};

        restore[0].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        restore[0].oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        restore[0].newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        restore[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[0].image               = gpu->dlss_output;
        restore[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        restore[0].subresourceRange.levelCount = 1;
        restore[0].subresourceRange.layerCount = 1;
        restore[0].srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
        restore[0].dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;

        restore[1].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        restore[1].oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        restore[1].newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        restore[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[1].image               = gpu->swapchain_images[gpu->current_image];
        restore[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        restore[1].subresourceRange.levelCount = 1;
        restore[1].subresourceRange.layerCount = 1;
        restore[1].srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        restore[1].dstAccessMask       = 0;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, NULL, 0, NULL, 2, restore);
    }

    /* Overlay text (after DLSS RT blit, before submit) */
    gpu_overlay_flush(gpu);

    vkEndCommandBuffer(cmd);

    /* Submit */
    VkSemaphore wait_sems[]   = { gpu->image_available[fi] };
    VkSemaphore signal_sems[] = { gpu->render_finished[fi] };
    VkPipelineStageFlags wait_stages[] = {
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
    };

    VkSubmitInfo submit = {0};
    submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount   = 1;
    submit.pWaitSemaphores      = wait_sems;
    submit.pWaitDstStageMask    = wait_stages;
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = signal_sems;

    vkQueueSubmit(gpu->graphics_queue, 1, &submit, gpu->in_flight[fi]);

    /* Present (skip in headless mode) */
    if (!gpu->headless) {
        VkPresentInfoKHR present = {0};
        present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores    = signal_sems;
        present.swapchainCount     = 1;
        present.pSwapchains        = &gpu->swapchain;
        present.pImageIndices      = &gpu->current_image;

        vkQueuePresentKHR(gpu->graphics_queue, &present);
    }

    gpu->frame_index = (gpu->frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
}

/* ---- DLSS pipeline creation (MRT: 2 color attachments) ---- */

GpuPipeline gpu_create_dlss_pipeline(Gpu* gpu, const GpuPipelineDesc* desc)
{
    GpuPipeline pipe = calloc(1, sizeof(struct GpuPipeline_s));
    if (!pipe) return NULL;

    VkShaderModule vert_mod = create_shader_module(gpu->device, desc->vert_spv, desc->vert_size);
    VkShaderModule frag_mod = create_shader_module(gpu->device, desc->frag_spv, desc->frag_size);

    VkPipelineShaderStageCreateInfo stages[2] = {0};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_mod;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_mod;
    stages[1].pName  = "main";

    VkVertexInputBindingDescription binding = {0};
    binding.stride    = desc->vertex_stride;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription* attrs = NULL;
    if (desc->nattribs > 0) {
        attrs = malloc(desc->nattribs * sizeof(VkVertexInputAttributeDescription));
        for (uint32_t i = 0; i < desc->nattribs; i++) {
            attrs[i].location = desc->attribs[i].location;
            attrs[i].binding  = 0;
            attrs[i].offset   = desc->attribs[i].offset;
            attrs[i].format   = VK_FORMAT_R32G32B32_SFLOAT;
        }
    }

    VkPipelineVertexInputStateCreateInfo vertex_input = {0};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount   = 1;
    vertex_input.pVertexBindingDescriptions      = &binding;
    vertex_input.vertexAttributeDescriptionCount = desc->nattribs;
    vertex_input.pVertexAttributeDescriptions    = attrs;

    VkPipelineInputAssemblyStateCreateInfo input_asm = {0};
    input_asm.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_asm.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_state = {0};
    viewport_state.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo raster = {0};
    raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode    = VK_CULL_MODE_NONE;
    raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample = {0};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth = {0};
    depth.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable  = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    /* Two color blend attachments (color + motion vectors) */
    VkPipelineColorBlendAttachmentState blend_att[2] = {0};
    blend_att[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blend_att[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;

    VkPipelineColorBlendStateCreateInfo blend = {0};
    blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 2;
    blend.pAttachments    = blend_att;

    VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic = {0};
    dynamic.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates    = dyn_states;

    VkGraphicsPipelineCreateInfo pci = {0};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &vertex_input;
    pci.pInputAssemblyState = &input_asm;
    pci.pViewportState      = &viewport_state;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState   = &multisample;
    pci.pDepthStencilState  = &depth;
    pci.pColorBlendState    = &blend;
    pci.pDynamicState       = &dynamic;
    pci.layout              = gpu->dlss_pipeline_layout;
    pci.renderPass          = gpu->dlss_render_pass;
    pci.subpass             = 0;

    if (vkCreateGraphicsPipelines(gpu->device, gpu->pipeline_cache, 1, &pci, NULL,
                                  &pipe->pipeline) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create DLSS pipeline\n");
    }
    pipe->layout = gpu->dlss_pipeline_layout;

    vkDestroyShaderModule(gpu->device, vert_mod, NULL);
    vkDestroyShaderModule(gpu->device, frag_mod, NULL);
    free(attrs);

    return pipe;
}

GpuPipeline gpu_create_dlss_shadow_pipeline(Gpu* gpu, const GpuPipelineDesc* desc)
{
    /* Same as dlss_pipeline but with shadow descriptor set layout. */
    /* For now, create the same pipeline — shadow TLAS binding is separate. */
    return gpu_create_dlss_pipeline(gpu, desc);
}

int gpu_build_rt_scene_dlss(Gpu* gpu,
                            const GpuRtMeshDesc* meshes, uint32_t nmeshes,
                            const uint32_t* rgen_spv, uint32_t rgen_size,
                            const uint32_t* miss_spv, uint32_t miss_size,
                            const uint32_t* chit_spv, uint32_t chit_size)
{
    (void)meshes; (void)nmeshes;
    (void)rgen_spv; (void)rgen_size;
    (void)miss_spv; (void)miss_size;
    (void)chit_spv; (void)chit_size;
    return 0;
}

/* ========================================================================
 * Text overlay rendering
 * ======================================================================== */

int gpu_overlay_init(Gpu* gpu)
{
    /* ---- Font texture (128x128 R8, arranged as 16x8 grid of 8x16 chars) ---- */
    uint8_t pixels[128 * 128];
    memset(pixels, 0, sizeof(pixels));
    for (int ch = 0; ch < FONT_NUM_CHARS; ch++) {
        int col = ch % 16;
        int row = ch / 16;
        for (int y = 0; y < FONT_CHAR_H; y++) {
            uint8_t bits = font8x16_data[ch][y];
            for (int x = 0; x < FONT_CHAR_W; x++) {
                if (bits & (1 << (7 - x)))
                    pixels[(row * FONT_CHAR_H + y) * 128 + col * FONT_CHAR_W + x] = 255;
            }
        }
    }

    /* Create image */
    VkImageCreateInfo ici = {0};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R8_UNORM;
    ici.extent        = (VkExtent3D){ 128, 128, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkCreateImage(gpu->device, &ici, NULL, &gpu->overlay_font_image);

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(gpu->device, gpu->overlay_font_image, &req);
    VkMemoryAllocateInfo ai = {0};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize   = req.size;
    ai.memoryTypeIndex  = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    gpu_alloc_memory(gpu, &ai, &gpu->overlay_font_memory);
    vkBindImageMemory(gpu->device, gpu->overlay_font_image, gpu->overlay_font_memory, 0);

    /* Staging buffer for upload */
    VkBuffer staging;
    VkDeviceMemory staging_mem;
    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = 128 * 128;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(gpu->device, &bci, NULL, &staging);

    VkMemoryRequirements breq;
    vkGetBufferMemoryRequirements(gpu->device, staging, &breq);
    VkMemoryAllocateInfo bai = {0};
    bai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bai.allocationSize   = breq.size;
    bai.memoryTypeIndex  = find_memory_type(gpu->physical_device, breq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(gpu->device, &bai, NULL, &staging_mem);
    vkBindBufferMemory(gpu->device, staging, staging_mem, 0);

    void* mapped;
    vkMapMemory(gpu->device, staging_mem, 0, 128 * 128, 0, &mapped);
    memcpy(mapped, pixels, 128 * 128);
    vkUnmapMemory(gpu->device, staging_mem);

    /* One-shot command buffer for upload */
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cba = {0};
    cba.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool        = gpu->command_pool;
    cba.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    vkAllocateCommandBuffers(gpu->device, &cba, &cmd);

    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    /* Transition to TRANSFER_DST */
    VkImageMemoryBarrier bar = {0};
    bar.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
    bar.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    bar.image                           = gpu->overlay_font_image;
    bar.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    bar.subresourceRange.levelCount     = 1;
    bar.subresourceRange.layerCount     = 1;
    bar.dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 1, &bar);

    VkBufferImageCopy region = {0};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = (VkExtent3D){ 128, 128, 1 };
    vkCmdCopyBufferToImage(cmd, staging, gpu->overlay_font_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    /* Transition to SHADER_READ */
    bar.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, NULL, 0, NULL, 1, &bar);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = {0};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(gpu->graphics_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(gpu->graphics_queue);

    vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &cmd);
    vkDestroyBuffer(gpu->device, staging, NULL);
    vkFreeMemory(gpu->device, staging_mem, NULL);

    /* ---- Image view ---- */
    VkImageViewCreateInfo vci = {0};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = gpu->overlay_font_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = VK_FORMAT_R8_UNORM;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.layerCount = 1;
    vkCreateImageView(gpu->device, &vci, NULL, &gpu->overlay_font_view);

    /* ---- Sampler ---- */
    VkSamplerCreateInfo sci = {0};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(gpu->device, &sci, NULL, &gpu->overlay_font_sampler);

    /* ---- Descriptor set layout (binding 0 = combined image sampler) ---- */
    VkDescriptorSetLayoutBinding binding = {0};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslci = {0};
    dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount  = 1;
    dslci.pBindings     = &binding;
    vkCreateDescriptorSetLayout(gpu->device, &dslci, NULL, &gpu->overlay_desc_layout);

    /* ---- Descriptor pool + set ---- */
    VkDescriptorPoolSize pool_size = {
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
    VkDescriptorPoolCreateInfo dpci = {0};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    vkCreateDescriptorPool(gpu->device, &dpci, NULL, &gpu->overlay_desc_pool);

    VkDescriptorSetAllocateInfo dsai = {0};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = gpu->overlay_desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &gpu->overlay_desc_layout;
    vkAllocateDescriptorSets(gpu->device, &dsai, &gpu->overlay_desc_set);

    VkDescriptorImageInfo dii = {0};
    dii.sampler     = gpu->overlay_font_sampler;
    dii.imageView   = gpu->overlay_font_view;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet wds = {0};
    wds.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet          = gpu->overlay_desc_set;
    wds.dstBinding      = 0;
    wds.descriptorCount = 1;
    wds.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wds.pImageInfo      = &dii;
    vkUpdateDescriptorSets(gpu->device, 1, &wds, 0, NULL);

    /* ---- Pipeline layout (push constant = 2 floats for screen size) ---- */
    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcr.size       = 8; /* vec2 screen_size */

    VkPipelineLayoutCreateInfo plci = {0};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &gpu->overlay_desc_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    vkCreatePipelineLayout(gpu->device, &plci, NULL, &gpu->overlay_pipe_layout);

    /* ---- Overlay render pass ---- */
    VkAttachmentDescription att = {0};
    att.format         = gpu->swapchain_format;
    att.samples        = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
    att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription sub = {0};
    sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments    = &ref;

    VkSubpassDependency dep = {0};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

    VkRenderPassCreateInfo rpci = {0};
    rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1;
    rpci.pAttachments    = &att;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &sub;
    rpci.dependencyCount = 1;
    rpci.pDependencies   = &dep;
    vkCreateRenderPass(gpu->device, &rpci, NULL, &gpu->overlay_render_pass);

    /* ---- Overlay framebuffers (one per swapchain image) ---- */
    gpu->overlay_framebuffers = (VkFramebuffer*)calloc(gpu->image_count, sizeof(VkFramebuffer));
    for (uint32_t i = 0; i < gpu->image_count; i++) {
        VkFramebufferCreateInfo fbci = {0};
        fbci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbci.renderPass      = gpu->overlay_render_pass;
        fbci.attachmentCount = 1;
        fbci.pAttachments    = &gpu->swapchain_views[i];
        fbci.width           = gpu->swapchain_extent.width;
        fbci.height          = gpu->swapchain_extent.height;
        fbci.layers          = 1;
        vkCreateFramebuffer(gpu->device, &fbci, NULL, &gpu->overlay_framebuffers[i]);
    }

    /* ---- Graphics pipeline ---- */
    {
        /* Load shaders */
        char vert_path[512], frag_path[512];
        snprintf(vert_path, sizeof(vert_path), "%s/overlay.vert.spv", SHADER_DIR);
        snprintf(frag_path, sizeof(frag_path), "%s/overlay.frag.spv", SHADER_DIR);

        FILE* f = fopen(vert_path, "rb");
        if (!f) { fprintf(stderr, "gpu_vulkan: can't open %s\n", vert_path); return 0; }
        fseek(f, 0, SEEK_END);
        long vsz = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint32_t* vcode = (uint32_t*)malloc(vsz);
        fread(vcode, 1, vsz, f);
        fclose(f);

        f = fopen(frag_path, "rb");
        if (!f) { fprintf(stderr, "gpu_vulkan: can't open %s\n", frag_path); free(vcode); return 0; }
        fseek(f, 0, SEEK_END);
        long fsz = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint32_t* fcode = (uint32_t*)malloc(fsz);
        fread(fcode, 1, fsz, f);
        fclose(f);

        VkShaderModule vert_mod = create_shader_module(gpu->device, vcode, (uint32_t)vsz);
        VkShaderModule frag_mod = create_shader_module(gpu->device, fcode, (uint32_t)fsz);
        free(vcode);
        free(fcode);

        VkPipelineShaderStageCreateInfo stages[2] = {0};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert_mod;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag_mod;
        stages[1].pName  = "main";

        /* Vertex input: pos(2f) + uv(2f) + color(4f) = 32 bytes */
        VkVertexInputBindingDescription vbind = {0};
        vbind.binding   = 0;
        vbind.stride    = 32;
        vbind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription vattrs[3] = {0};
        vattrs[0].location = 0;
        vattrs[0].format   = VK_FORMAT_R32G32_SFLOAT;
        vattrs[0].offset   = 0;
        vattrs[1].location = 1;
        vattrs[1].format   = VK_FORMAT_R32G32_SFLOAT;
        vattrs[1].offset   = 8;
        vattrs[2].location = 2;
        vattrs[2].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        vattrs[2].offset   = 16;

        VkPipelineVertexInputStateCreateInfo vertex_input = {0};
        vertex_input.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input.vertexBindingDescriptionCount   = 1;
        vertex_input.pVertexBindingDescriptions      = &vbind;
        vertex_input.vertexAttributeDescriptionCount = 3;
        vertex_input.pVertexAttributeDescriptions    = vattrs;

        VkPipelineInputAssemblyStateCreateInfo ia = {0};
        ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp = {0};
        vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rast = {0};
        rast.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rast.polygonMode = VK_POLYGON_MODE_FILL;
        rast.cullMode    = VK_CULL_MODE_NONE;
        rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rast.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms = {0};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds = {0};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_FALSE;
        ds.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState blend_att = {0};
        blend_att.blendEnable         = VK_TRUE;
        blend_att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend_att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_att.colorBlendOp        = VK_BLEND_OP_ADD;
        blend_att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend_att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_att.alphaBlendOp        = VK_BLEND_OP_ADD;
        blend_att.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo blend = {0};
        blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend.attachmentCount = 1;
        blend.pAttachments    = &blend_att;

        VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn = {0};
        dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates    = dyn_states;

        VkGraphicsPipelineCreateInfo gpci = {0};
        gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpci.stageCount          = 2;
        gpci.pStages             = stages;
        gpci.pVertexInputState   = &vertex_input;
        gpci.pInputAssemblyState = &ia;
        gpci.pViewportState      = &vp;
        gpci.pRasterizationState = &rast;
        gpci.pMultisampleState   = &ms;
        gpci.pDepthStencilState  = &ds;
        gpci.pColorBlendState    = &blend;
        gpci.pDynamicState       = &dyn;
        gpci.layout              = gpu->overlay_pipe_layout;
        gpci.renderPass          = gpu->overlay_render_pass;
        gpci.subpass             = 0;

        vkCreateGraphicsPipelines(gpu->device, gpu->pipeline_cache, 1, &gpci, NULL, &gpu->overlay_pipeline);
        vkDestroyShaderModule(gpu->device, vert_mod, NULL);
        vkDestroyShaderModule(gpu->device, frag_mod, NULL);
    }

    /* ---- Dynamic vertex buffer (host-visible, persistently mapped) ---- */
    {
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = OVERLAY_MAX_VERTICES * 32; /* 32 bytes/vertex */
        bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vkCreateBuffer(gpu->device, &bci, NULL, &gpu->overlay_vb);

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->overlay_vb, &req);
        VkMemoryAllocateInfo ai = {0};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize   = req.size;
        ai.memoryTypeIndex  = find_memory_type(gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        gpu_alloc_memory(gpu, &ai, &gpu->overlay_vb_mem);
        vkBindBufferMemory(gpu->device, gpu->overlay_vb, gpu->overlay_vb_mem, 0);
        vkMapMemory(gpu->device, gpu->overlay_vb_mem, 0, bci.size, 0, &gpu->overlay_vb_mapped);
    }

    gpu->overlay_cpu_verts = (float*)malloc(OVERLAY_MAX_VERTICES * 8 * sizeof(float));
    gpu->overlay_vert_count = 0;
    gpu->overlay_inited = 1;

    fprintf(stderr, "gpu_vulkan: overlay initialized\n");
    return 1;
}

void gpu_overlay_shutdown(Gpu* gpu)
{
    if (!gpu->overlay_inited) return;
    vkDeviceWaitIdle(gpu->device);

    free(gpu->overlay_cpu_verts);

    if (gpu->overlay_vb_mapped)
        vkUnmapMemory(gpu->device, gpu->overlay_vb_mem);
    vkDestroyBuffer(gpu->device, gpu->overlay_vb, NULL);
    vkFreeMemory(gpu->device, gpu->overlay_vb_mem, NULL);

    vkDestroyPipeline(gpu->device, gpu->overlay_pipeline, NULL);
    vkDestroyPipelineLayout(gpu->device, gpu->overlay_pipe_layout, NULL);
    vkDestroyRenderPass(gpu->device, gpu->overlay_render_pass, NULL);

    for (uint32_t i = 0; i < gpu->image_count; i++)
        vkDestroyFramebuffer(gpu->device, gpu->overlay_framebuffers[i], NULL);
    free(gpu->overlay_framebuffers);

    vkDestroyDescriptorPool(gpu->device, gpu->overlay_desc_pool, NULL);
    vkDestroyDescriptorSetLayout(gpu->device, gpu->overlay_desc_layout, NULL);

    vkDestroySampler(gpu->device, gpu->overlay_font_sampler, NULL);
    vkDestroyImageView(gpu->device, gpu->overlay_font_view, NULL);
    vkDestroyImage(gpu->device, gpu->overlay_font_image, NULL);
    vkFreeMemory(gpu->device, gpu->overlay_font_memory, NULL);

    gpu->overlay_inited = 0;
}

void gpu_overlay_text(Gpu* gpu, float x, float y, float scale,
                      float r, float g, float b, float a,
                      const char* text)
{
    if (!gpu->overlay_inited || !text) return;

    float char_w = FONT_CHAR_W * scale;
    float char_h = FONT_CHAR_H * scale;
    float cx = x;

    for (const char* p = text; *p; p++) {
        if (*p == '\n') {
            cx = x;
            y += char_h;
            continue;
        }
        int ch = (unsigned char)*p;
        if (ch < FONT_FIRST_CHAR || ch >= FONT_FIRST_CHAR + FONT_NUM_CHARS)
            ch = FONT_FIRST_CHAR; /* space for unprintable */

        int idx = ch - FONT_FIRST_CHAR;
        int col = idx % 16;
        int row = idx / 16;

        /* UV coords in the 128x128 atlas */
        float u0 = (col * FONT_CHAR_W) / 128.0f;
        float v0 = (row * FONT_CHAR_H) / 128.0f;
        float u1 = u0 + FONT_CHAR_W / 128.0f;
        float v1 = v0 + FONT_CHAR_H / 128.0f;

        if (gpu->overlay_vert_count + 6 > OVERLAY_MAX_VERTICES)
            break;

        /* 6 vertices per char (2 triangles), 8 floats per vertex: x,y,u,v,r,g,b,a */
        float* v = gpu->overlay_cpu_verts + gpu->overlay_vert_count * 8;

        /* Triangle 1: top-left, top-right, bottom-left */
        v[0] = cx;          v[1] = y;          v[2] = u0; v[3] = v0; v[4] = r; v[5] = g; v[6] = b; v[7] = a;
        v += 8;
        v[0] = cx + char_w; v[1] = y;          v[2] = u1; v[3] = v0; v[4] = r; v[5] = g; v[6] = b; v[7] = a;
        v += 8;
        v[0] = cx;          v[1] = y + char_h; v[2] = u0; v[3] = v1; v[4] = r; v[5] = g; v[6] = b; v[7] = a;
        v += 8;

        /* Triangle 2: top-right, bottom-right, bottom-left */
        v[0] = cx + char_w; v[1] = y;          v[2] = u1; v[3] = v0; v[4] = r; v[5] = g; v[6] = b; v[7] = a;
        v += 8;
        v[0] = cx + char_w; v[1] = y + char_h; v[2] = u1; v[3] = v1; v[4] = r; v[5] = g; v[6] = b; v[7] = a;
        v += 8;
        v[0] = cx;          v[1] = y + char_h; v[2] = u0; v[3] = v1; v[4] = r; v[5] = g; v[6] = b; v[7] = a;

        gpu->overlay_vert_count += 6;
        cx += char_w;
    }
}

void gpu_overlay_rect(Gpu* gpu, float x, float y, float w, float h,
                      float r, float g, float b, float a)
{
    if (!gpu->overlay_inited) return;
    if (gpu->overlay_vert_count + 6 > OVERLAY_MAX_VERTICES) return;

    /* 6 vertices, UV = (-1,-1) signals solid fill to the fragment shader */
    float* v = gpu->overlay_cpu_verts + gpu->overlay_vert_count * 8;

    v[0] = x;     v[1] = y;     v[2] = -1; v[3] = -1; v[4] = r; v[5] = g; v[6] = b; v[7] = a; v += 8;
    v[0] = x + w; v[1] = y;     v[2] = -1; v[3] = -1; v[4] = r; v[5] = g; v[6] = b; v[7] = a; v += 8;
    v[0] = x;     v[1] = y + h; v[2] = -1; v[3] = -1; v[4] = r; v[5] = g; v[6] = b; v[7] = a; v += 8;

    v[0] = x + w; v[1] = y;     v[2] = -1; v[3] = -1; v[4] = r; v[5] = g; v[6] = b; v[7] = a; v += 8;
    v[0] = x + w; v[1] = y + h; v[2] = -1; v[3] = -1; v[4] = r; v[5] = g; v[6] = b; v[7] = a; v += 8;
    v[0] = x;     v[1] = y + h; v[2] = -1; v[3] = -1; v[4] = r; v[5] = g; v[6] = b; v[7] = a;

    gpu->overlay_vert_count += 6;
}

void gpu_overlay_flush(Gpu* gpu)
{
    if (!gpu->overlay_inited || gpu->overlay_vert_count == 0) return;

    VkCommandBuffer cmd = gpu->current_cmd;

    /* Copy CPU vertex data to mapped GPU buffer */
    memcpy(gpu->overlay_vb_mapped, gpu->overlay_cpu_verts,
           gpu->overlay_vert_count * 8 * sizeof(float));

    /* Transition swapchain from PRESENT_SRC to COLOR_ATTACHMENT for overlay */
    {
        VkImageMemoryBarrier barrier = {0};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = gpu->swapchain_images[gpu->current_image];
        barrier.subresourceRange    = (VkImageSubresourceRange){
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                      VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, NULL, 0, NULL, 1, &barrier);
    }

    /* Begin overlay render pass (initialLayout = COLOR_ATTACHMENT_OPTIMAL) */
    VkRenderPassBeginInfo rpbi = {0};
    rpbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass      = gpu->overlay_render_pass;
    rpbi.framebuffer     = gpu->overlay_framebuffers[gpu->current_image];
    rpbi.renderArea.extent = gpu->swapchain_extent;
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    /* Set viewport and scissor */
    VkViewport vp = {0};
    vp.width  = (float)gpu->swapchain_extent.width;
    vp.height = (float)gpu->swapchain_extent.height;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D sc = {0};
    sc.extent = gpu->swapchain_extent;
    vkCmdSetScissor(cmd, 0, 1, &sc);

    /* Bind pipeline, descriptors, vertex buffer */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gpu->overlay_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            gpu->overlay_pipe_layout, 0, 1, &gpu->overlay_desc_set,
                            0, NULL);

    float screen_size[2] = { vp.width, vp.height };
    vkCmdPushConstants(cmd, gpu->overlay_pipe_layout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, 8, screen_size);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &gpu->overlay_vb, &offset);

    vkCmdDraw(cmd, gpu->overlay_vert_count, 1, 0, 0);

    vkCmdEndRenderPass(cmd);

    /* Reset for next frame */
    gpu->overlay_vert_count = 0;
}

/* ---- Raycast compute pipeline (LiDAR/radar) ---- */

static void raycast_destroy_buffers(Gpu* gpu)
{
    /* Device-local buffers */
    if (gpu->raycast_input_buf) {
        vkDestroyBuffer(gpu->device, gpu->raycast_input_buf, NULL);
        vkFreeMemory(gpu->device, gpu->raycast_input_mem, NULL);
        gpu->raycast_input_buf = VK_NULL_HANDLE;
    }
    if (gpu->raycast_output_buf) {
        vkDestroyBuffer(gpu->device, gpu->raycast_output_buf, NULL);
        vkFreeMemory(gpu->device, gpu->raycast_output_mem, NULL);
        gpu->raycast_output_buf = VK_NULL_HANDLE;
    }
    /* Staging buffers */
    if (gpu->raycast_staging_in) {
        if (gpu->raycast_staging_in_mapped)
            vkUnmapMemory(gpu->device, gpu->raycast_staging_in_mem);
        vkDestroyBuffer(gpu->device, gpu->raycast_staging_in, NULL);
        vkFreeMemory(gpu->device, gpu->raycast_staging_in_mem, NULL);
        gpu->raycast_staging_in = VK_NULL_HANDLE;
        gpu->raycast_staging_in_mapped = NULL;
    }
    if (gpu->raycast_staging_out) {
        if (gpu->raycast_staging_out_mapped)
            vkUnmapMemory(gpu->device, gpu->raycast_staging_out_mem);
        vkDestroyBuffer(gpu->device, gpu->raycast_staging_out, NULL);
        vkFreeMemory(gpu->device, gpu->raycast_staging_out_mem, NULL);
        gpu->raycast_staging_out = VK_NULL_HANDLE;
        gpu->raycast_staging_out_mapped = NULL;
    }
    gpu->raycast_max_rays = 0;
}

static int raycast_ensure_buffers(Gpu* gpu, uint32_t num_rays)
{
    if (num_rays <= gpu->raycast_max_rays) return 1;

    /* Free old buffers */
    vkDeviceWaitIdle(gpu->device);
    raycast_destroy_buffers(gpu);

    /* Round up to power of 2 for amortized allocation */
    uint32_t cap = 1;
    while (cap < num_rays) cap <<= 1;

    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    VkMemoryRequirements req;
    VkMemoryAllocateInfo mai = {0};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

    VkDeviceSize input_size  = (VkDeviceSize)cap * 6 * sizeof(float);
    VkDeviceSize output_size = (VkDeviceSize)cap * 8 * sizeof(float);

    /* ---- Device-local input SSBO (GPU reads ray data from VRAM) ---- */
    bci.size  = input_size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->raycast_input_buf) != VK_SUCCESS)
        return 0;
    vkGetBufferMemoryRequirements(gpu->device, gpu->raycast_input_buf, &req);
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (gpu_alloc_memory(gpu, &mai, &gpu->raycast_input_mem) != VK_SUCCESS) return 0;
    vkBindBufferMemory(gpu->device, gpu->raycast_input_buf, gpu->raycast_input_mem, 0);

    /* ---- Device-local output SSBO (GPU writes results to VRAM) ---- */
    bci.size  = output_size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->raycast_output_buf) != VK_SUCCESS)
        return 0;
    vkGetBufferMemoryRequirements(gpu->device, gpu->raycast_output_buf, &req);
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (gpu_alloc_memory(gpu, &mai, &gpu->raycast_output_mem) != VK_SUCCESS) return 0;
    vkBindBufferMemory(gpu->device, gpu->raycast_output_buf, gpu->raycast_output_mem, 0);

    /* ---- Host-visible staging buffer for upload ---- */
    bci.size  = input_size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->raycast_staging_in) != VK_SUCCESS)
        return 0;
    vkGetBufferMemoryRequirements(gpu->device, gpu->raycast_staging_in, &req);
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (gpu_alloc_memory(gpu, &mai, &gpu->raycast_staging_in_mem) != VK_SUCCESS) return 0;
    vkBindBufferMemory(gpu->device, gpu->raycast_staging_in, gpu->raycast_staging_in_mem, 0);
    vkMapMemory(gpu->device, gpu->raycast_staging_in_mem, 0, bci.size, 0, &gpu->raycast_staging_in_mapped);

    /* ---- Host-visible staging buffer for readback ---- */
    bci.size  = output_size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->raycast_staging_out) != VK_SUCCESS)
        return 0;
    vkGetBufferMemoryRequirements(gpu->device, gpu->raycast_staging_out, &req);
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (gpu_alloc_memory(gpu, &mai, &gpu->raycast_staging_out_mem) != VK_SUCCESS) return 0;
    vkBindBufferMemory(gpu->device, gpu->raycast_staging_out, gpu->raycast_staging_out_mem, 0);
    vkMapMemory(gpu->device, gpu->raycast_staging_out_mem, 0, bci.size, 0, &gpu->raycast_staging_out_mapped);

    gpu->raycast_max_rays = cap;

    /* Update descriptor set with new device-local buffer handles */
    if (gpu->raycast_desc_set) {
        VkDescriptorBufferInfo in_info = {0};
        in_info.buffer = gpu->raycast_input_buf;
        in_info.offset = 0;
        in_info.range  = input_size;

        VkDescriptorBufferInfo out_info = {0};
        out_info.buffer = gpu->raycast_output_buf;
        out_info.offset = 0;
        out_info.range  = output_size;

        VkWriteDescriptorSet writes[2] = {{0}, {0}};

        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = gpu->raycast_desc_set;
        writes[0].dstBinding      = 1;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo     = &in_info;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = gpu->raycast_desc_set;
        writes[1].dstBinding      = 2;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo     = &out_info;

        vkUpdateDescriptorSets(gpu->device, 2, writes, 0, NULL);
    }

    return 1;
}

int gpu_build_raycast_pipeline(Gpu* gpu,
                                const uint32_t* comp_spv, uint32_t comp_size)
{
    if (!gpu->rt_available || !gpu->tlas) {
        fprintf(stderr, "gpu_vulkan: raycast requires RT + built TLAS\n");
        return 0;
    }
    if (!gpu->scene_data_buf) {
        fprintf(stderr, "gpu_vulkan: raycast requires scene_data_buf (vertex/index addresses)\n");
        return 0;
    }

    /* Descriptor set layout: binding 0 = TLAS, 1 = input SSBO, 2 = output SSBO,
     * 3 = scene data SSBO (per-mesh vertex/index buffer device addresses —
     * lets the shader fetch the real triangle normal for grazing + Lambertian). */
    VkDescriptorSetLayoutBinding bindings[4] = {{0}};

    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding         = 2;
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[3].binding         = 3;
    bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
    dsl_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 4;
    dsl_ci.pBindings    = bindings;

    if (vkCreateDescriptorSetLayout(gpu->device, &dsl_ci, NULL,
                                     &gpu->raycast_desc_layout) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create raycast descriptor set layout\n");
        return 0;
    }

    /* Descriptor pool */
    VkDescriptorPoolSize pool_sizes[2] = {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
    };
    VkDescriptorPoolCreateInfo pool_ci = {0};
    pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets       = 1;
    pool_ci.poolSizeCount = 2;
    pool_ci.pPoolSizes    = pool_sizes;

    if (vkCreateDescriptorPool(gpu->device, &pool_ci, NULL,
                                &gpu->raycast_desc_pool) != VK_SUCCESS)
        return 0;

    /* Allocate descriptor set */
    VkDescriptorSetAllocateInfo ds_ai = {0};
    ds_ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_ai.descriptorPool     = gpu->raycast_desc_pool;
    ds_ai.descriptorSetCount = 1;
    ds_ai.pSetLayouts        = &gpu->raycast_desc_layout;

    if (vkAllocateDescriptorSets(gpu->device, &ds_ai,
                                  &gpu->raycast_desc_set) != VK_SUCCESS)
        return 0;

    /* Write TLAS descriptor (binding 0) */
    VkWriteDescriptorSetAccelerationStructureKHR as_info = {0};
    as_info.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as_info.accelerationStructureCount = 1;
    as_info.pAccelerationStructures    = &gpu->tlas;

    VkWriteDescriptorSet initial_writes[2] = {{0}, {0}};

    initial_writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    initial_writes[0].pNext           = &as_info;
    initial_writes[0].dstSet          = gpu->raycast_desc_set;
    initial_writes[0].dstBinding      = 0;
    initial_writes[0].descriptorCount = 1;
    initial_writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    /* Binding 3: scene data SSBO (vertex+index buffer device addresses per mesh).
     * Raycast compute shader fetches this to compute the real triangle normal. */
    VkDescriptorBufferInfo scene_info = {0};
    scene_info.buffer = gpu->scene_data_buf;
    scene_info.offset = 0;
    scene_info.range  = VK_WHOLE_SIZE;

    initial_writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    initial_writes[1].dstSet          = gpu->raycast_desc_set;
    initial_writes[1].dstBinding      = 3;
    initial_writes[1].descriptorCount = 1;
    initial_writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    initial_writes[1].pBufferInfo     = &scene_info;

    uint32_t nw = gpu->scene_data_buf ? 2 : 1;
    vkUpdateDescriptorSets(gpu->device, nw, initial_writes, 0, NULL);

    /* Push constant range */
    VkPushConstantRange pc_range = {0};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset     = 0;
    pc_range.size       = sizeof(GpuRaycastPushConstants);

    /* Pipeline layout */
    VkPipelineLayoutCreateInfo pl_ci = {0};
    pl_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount         = 1;
    pl_ci.pSetLayouts            = &gpu->raycast_desc_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges    = &pc_range;

    if (vkCreatePipelineLayout(gpu->device, &pl_ci, NULL,
                                &gpu->raycast_pipeline_layout) != VK_SUCCESS)
        return 0;

    /* Compute pipeline */
    VkShaderModule comp_mod = create_shader_module(gpu->device, comp_spv, comp_size);
    if (!comp_mod) return 0;

    VkPipelineShaderStageCreateInfo stage = {0};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = comp_mod;
    stage.pName  = "main";

    VkComputePipelineCreateInfo pipe_ci = {0};
    pipe_ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_ci.stage  = stage;
    pipe_ci.layout = gpu->raycast_pipeline_layout;

    VkResult res = vkCreateComputePipelines(gpu->device, gpu->pipeline_cache, 1,
                                             &pipe_ci, NULL,
                                             &gpu->raycast_pipeline);
    vkDestroyShaderModule(gpu->device, comp_mod, NULL);

    if (res != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create raycast compute pipeline\n");
        return 0;
    }

    gpu->raycast_built = 1;

    /* Create fence for async ray dispatch */
    VkFenceCreateInfo fence_ci = {0};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_ci.flags = 0;  /* Start unsignaled — first submit must not see a signaled fence */
    if (vkCreateFence(gpu->device, &fence_ci, NULL, &gpu->raycast_async_fence) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create raycast async fence\n");
        return 0;
    }
    gpu->raycast_async_pending = 0;
    gpu->raycast_async_cmd = VK_NULL_HANDLE;

    fprintf(stderr, "gpu_vulkan: raycast compute pipeline created\n");
    return 1;
}

void gpu_destroy_raycast_pipeline(Gpu* gpu)
{
    if (!gpu) return;
    vkDeviceWaitIdle(gpu->device);

    /* Clean up async state */
    if (gpu->raycast_async_cmd) {
        vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &gpu->raycast_async_cmd);
        gpu->raycast_async_cmd = VK_NULL_HANDLE;
    }
    if (gpu->raycast_async_fence) {
        vkDestroyFence(gpu->device, gpu->raycast_async_fence, NULL);
        gpu->raycast_async_fence = VK_NULL_HANDLE;
    }
    gpu->raycast_async_pending = 0;

    raycast_destroy_buffers(gpu);

    if (gpu->raycast_pipeline) {
        vkDestroyPipeline(gpu->device, gpu->raycast_pipeline, NULL);
        gpu->raycast_pipeline = VK_NULL_HANDLE;
    }
    if (gpu->raycast_pipeline_layout) {
        vkDestroyPipelineLayout(gpu->device, gpu->raycast_pipeline_layout, NULL);
        gpu->raycast_pipeline_layout = VK_NULL_HANDLE;
    }
    if (gpu->raycast_desc_pool) {
        vkDestroyDescriptorPool(gpu->device, gpu->raycast_desc_pool, NULL);
        gpu->raycast_desc_pool = VK_NULL_HANDLE;
        gpu->raycast_desc_set  = VK_NULL_HANDLE;
    }
    if (gpu->raycast_desc_layout) {
        vkDestroyDescriptorSetLayout(gpu->device, gpu->raycast_desc_layout, NULL);
        gpu->raycast_desc_layout = VK_NULL_HANDLE;
    }
    gpu->raycast_built = 0;
}

/* ----------------------------------------------------------------------
 * Phase B deferred-shading compute pipeline
 *
 * Bindings (must match deferred_shade.comp.glsl):
 *   2  storage buffer  COMPUTE  scene_data_buf  (per-mesh display color)
 *   9  storage buffer  COMPUTE  interop_buf[N]  (final pixel output)
 *   17 storage buffer  COMPUTE  tiled_gbuffer   (rchit/rmiss-written)
 *
 * Two descriptor sets ('A' and 'B') mirror tiled_rt_desc_set + _b — set A
 * binds binding 9 to interop_buf[0], set B to interop_buf[1]. The dispatch
 * helper picks the parity matching the RT path that just ran. */
int gpu_build_deferred_pipeline(Gpu* gpu,
                                 const uint32_t* comp_spv, uint32_t comp_size)
{
    if (!gpu || !comp_spv || comp_size == 0) return 0;
    if (gpu->deferred_built) return 1;

    /* Phase C.1: also need to bind material SSBO (3) and textures[] (4)
     * for diffuse-color sampling. Mirror the tiled RT pipeline's
     * conditional pattern (gpu_vulkan.c:13325-13367) — bindings 3/4 are
     * only declared in the layout when materials/textures are uploaded;
     * the SPIR-V's reference is statically gated by scene.hasMaterials
     * inside main() so unbound declarations don't cause runtime access.
     *
     * Phase C.3: also bind IBL (5/6/7) and the camera SSBO (8) when an
     * env map is loaded. The shader gates fetches on
     * `scene.envMipLevels > 0.0` so the unbound case is fine for the
     * SPIR-V reference; we just need the *bound* case to be wired up.
     *
     * Phase C.4: bind the TLAS at binding 0 (inline ray-query shadow tests
     * inside the direct-light evaluators) and the LightsBuffer at binding
     * 13 (scene-light direct contribution). The TLAS is bound to gpu->tlas,
     * mirroring the rchit's binding 0 exactly; the lights buffer falls back
     * to tiled_camera_buf when no lights are uploaded — same fallback the
     * RT pipeline uses (gpu_vulkan.c:13878). */
    int has_mat_ssbo = (gpu->mat_uploaded && gpu->mat_count > 0) ? 1 : 0;
    int has_textures = (gpu->mat_uploaded && gpu->mat_tex_count > 0) ? 1 : 0;
    int has_ibl      = (gpu->env_image_view != VK_NULL_HANDLE) ? 1 : 0;
    int rt_tex_count = has_textures ? gpu->mat_tex_count : 0;

    /* Descriptor set layout:
     *   0: TLAS                         (Phase C.4 — shadow ray queries)
     *   1: storage image (fallback when useDirectOut == 0)
     *   2: SceneData SSBO (per-mesh display color, vertex/index addrs)
     *   3: MaterialBuffer SSBO (Phase C.1 — only when has_mat_ssbo)
     *   4: textures[] sampler array (Phase C.1 — only when has_textures)
     *   5: envMap (Phase C.3 — only when has_ibl)
     *   6: brdfLUT (Phase C.3 — only when has_ibl; declared, not sampled)
     *   7: irrMap (Phase C.3 — only when has_ibl)
     *   8: cameras[] (Phase C.3 — view/proj inverses for ray-dir recompute)
     *   9: DirectOutput SSBO (final pixel writes)
     *  13: LightsBuffer SSBO            (Phase C.4 — scene lights)
     *  17: G-buffer SSBO (rchit/rmiss writes consumed by compute) */
    VkDescriptorSetLayoutBinding bindings[12] = {{0}};
    int nb = 0;

    /* Binding 0: TLAS — Phase C.4. */
    bindings[nb].binding         = 0;
    bindings[nb].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[nb].descriptorCount = 1;
    bindings[nb].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    nb++;

    bindings[nb].binding         = 1;
    bindings[nb].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[nb].descriptorCount = 1;
    bindings[nb].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    nb++;

    bindings[nb].binding         = 2;
    bindings[nb].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[nb].descriptorCount = 1;
    bindings[nb].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    nb++;

    if (has_mat_ssbo) {
        bindings[nb].binding         = 3;
        bindings[nb].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[nb].descriptorCount = 1;
        bindings[nb].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        nb++;
    }
    if (has_textures) {
        bindings[nb].binding         = 4;
        bindings[nb].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[nb].descriptorCount = (uint32_t)rt_tex_count;
        bindings[nb].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        nb++;
    }

    /* Phase C.3: bindings 5/6/7 (envMap / brdfLUT / irrMap). Mirror the
     * RT pipeline's IBL block at gpu_vulkan.c:5299-5308. */
    if (has_ibl) {
        for (int b = 5; b <= 7; b++) {
            bindings[nb].binding         = (uint32_t)b;
            bindings[nb].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[nb].descriptorCount = 1;
            bindings[nb].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            nb++;
        }
    }

    /* Phase C.3: binding 8 (cameras[] — needed by mode 3 for V = -ray_dir
     * reconstruction). Bind unconditionally — the storage buffer handle
     * is allocated by gpu_tiled_init for every tiled render. */
    bindings[nb].binding         = 8;
    bindings[nb].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[nb].descriptorCount = 1;
    bindings[nb].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    nb++;

    bindings[nb].binding         = 9;
    bindings[nb].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[nb].descriptorCount = 1;
    bindings[nb].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    nb++;

    /* Binding 13: LightsBuffer — Phase C.4. Always bound (falls back to
     * tiled_camera_buf when no lights are uploaded — the shader gates the
     * loop on `nlights > 0` so the fallback case never reads past the
     * 16-byte header). */
    bindings[nb].binding         = 13;
    bindings[nb].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[nb].descriptorCount = 1;
    bindings[nb].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    nb++;

    bindings[nb].binding         = 17;
    bindings[nb].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[nb].descriptorCount = 1;
    bindings[nb].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    nb++;

    VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
    dsl_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = (uint32_t)nb;
    dsl_ci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(gpu->device, &dsl_ci, NULL,
                                     &gpu->deferred_desc_layout) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: deferred desc layout create failed\n");
        return 0;
    }

    /* Pool: 2 sets x (1 TLAS + 1 storage image + N SSBOs + maybe rt_tex_count
     * samplers + maybe 3 IBL samplers).
     * SSBO count per set = 5 (scene + cameras + direct + lights + gbuf)
     *                    + (has_mat_ssbo ? 1 : 0). Phase C.4 bumps from 4
     *                    to 5 with the LightsBuffer (binding 13).
     * Phase C.3 adds binding 8 (cameras) — always present — and bindings
     * 5/6/7 (3 combined image samplers, only when has_ibl).
     * Phase C.4 adds binding 0 (TLAS) — one acceleration-structure descriptor
     *                    per set (2 total). */
    int ssbo_per_set = 5 + (has_mat_ssbo ? 1 : 0);
    int ibl_sampler_count = has_ibl ? 3 : 0;
    int sampler_per_set = rt_tex_count + ibl_sampler_count;
    VkDescriptorPoolSize pool_sizes[4] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)(ssbo_per_set * 2) },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, (uint32_t)(sampler_per_set * 2) },
    };
    int n_pool_sizes = (sampler_per_set > 0) ? 4 : 3;
    VkDescriptorPoolCreateInfo pool_ci = {0};
    pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets       = 2;
    pool_ci.poolSizeCount = (uint32_t)n_pool_sizes;
    pool_ci.pPoolSizes    = pool_sizes;
    if (vkCreateDescriptorPool(gpu->device, &pool_ci, NULL,
                                &gpu->deferred_desc_pool) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(gpu->device, gpu->deferred_desc_layout, NULL);
        gpu->deferred_desc_layout = VK_NULL_HANDLE;
        return 0;
    }

    VkDescriptorSetLayout layouts[2] = { gpu->deferred_desc_layout,
                                         gpu->deferred_desc_layout };
    VkDescriptorSet sets[2] = {0};
    VkDescriptorSetAllocateInfo dsai = {0};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = gpu->deferred_desc_pool;
    dsai.descriptorSetCount = 2;
    dsai.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(gpu->device, &dsai, sets) != VK_SUCCESS) {
        vkDestroyDescriptorPool(gpu->device, gpu->deferred_desc_pool, NULL);
        vkDestroyDescriptorSetLayout(gpu->device, gpu->deferred_desc_layout, NULL);
        gpu->deferred_desc_pool   = VK_NULL_HANDLE;
        gpu->deferred_desc_layout = VK_NULL_HANDLE;
        return 0;
    }
    gpu->deferred_desc_set   = sets[0];
    gpu->deferred_desc_set_b = sets[1];

    /* Pipeline layout: same push-constant range as tiled RT. */
    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(GpuRtTiledPushConstants);

    VkPipelineLayoutCreateInfo pl_ci = {0};
    pl_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount         = 1;
    pl_ci.pSetLayouts            = &gpu->deferred_desc_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(gpu->device, &pl_ci, NULL,
                                &gpu->deferred_pipeline_layout) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: deferred pipeline layout create failed\n");
        return 0;
    }

    VkShaderModule mod = create_shader_module(gpu->device, comp_spv, comp_size);
    if (!mod) return 0;

    VkPipelineShaderStageCreateInfo stage = {0};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName  = "main";

    VkComputePipelineCreateInfo pipe_ci = {0};
    pipe_ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_ci.stage  = stage;
    pipe_ci.layout = gpu->deferred_pipeline_layout;

    VkResult res = vkCreateComputePipelines(gpu->device, gpu->pipeline_cache, 1,
                                             &pipe_ci, NULL,
                                             &gpu->deferred_pipeline);
    vkDestroyShaderModule(gpu->device, mod, NULL);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: deferred compute pipeline create failed (vk=%d)\n", (int)res);
        return 0;
    }

    gpu->deferred_built = 1;
    fprintf(stderr, "gpu_vulkan: Phase B deferred-shading pipeline built\n");
    return 1;
}

void gpu_destroy_deferred_pipeline(Gpu* gpu)
{
    if (!gpu) return;
    vkDeviceWaitIdle(gpu->device);
    if (gpu->deferred_pipeline) {
        vkDestroyPipeline(gpu->device, gpu->deferred_pipeline, NULL);
        gpu->deferred_pipeline = VK_NULL_HANDLE;
    }
    if (gpu->deferred_pipeline_layout) {
        vkDestroyPipelineLayout(gpu->device, gpu->deferred_pipeline_layout, NULL);
        gpu->deferred_pipeline_layout = VK_NULL_HANDLE;
    }
    if (gpu->deferred_desc_pool) {
        vkDestroyDescriptorPool(gpu->device, gpu->deferred_desc_pool, NULL);
        gpu->deferred_desc_pool   = VK_NULL_HANDLE;
        gpu->deferred_desc_set    = VK_NULL_HANDLE;
        gpu->deferred_desc_set_b  = VK_NULL_HANDLE;
    }
    if (gpu->deferred_desc_layout) {
        vkDestroyDescriptorSetLayout(gpu->device, gpu->deferred_desc_layout, NULL);
        gpu->deferred_desc_layout = VK_NULL_HANDLE;
    }
    gpu->deferred_built = 0;
}

int gpu_deferred_pipeline_built(Gpu* gpu)
{
    return (gpu && gpu->deferred_built) ? 1 : 0;
}

/* Refresh the deferred descriptor sets' bindings 2 / 3 / 4 / 9 / 17 to match
 * the RT pipeline's current resources. Caller invokes this after the tiled
 * G-buffer / interop buffers have been (re-)allocated.
 *
 * Set A gets binding 9 → interop_buf[0]; set B → interop_buf[1]. Both sets
 * share bindings 2 (scene_data_buf), 3 (mat_ssbo_buf when uploaded),
 * 4 (textures[] sampler array when uploaded), and 17 (tiled_gbuffer_buf
 * if present, else scene_data_buf as a stub — same fallback the RT pipeline
 * uses). Phase C.1 added 3 + 4 for diffuse-color sampling in the compute
 * shader. */
static void gpu_descriptor_writes_for_deferred(Gpu* gpu)
{
    if (!gpu || !gpu->deferred_built) return;
    if (!gpu->deferred_desc_set) return;

    VkBuffer scene = gpu->scene_data_buf;
    VkBuffer gbuf  = gpu->tiled_gbuffer_buf ? gpu->tiled_gbuffer_buf : scene;

    int has_mat_ssbo = (gpu->mat_uploaded && gpu->mat_count > 0) ? 1 : 0;
    int has_textures = (gpu->mat_uploaded && gpu->mat_tex_count > 0) ? 1 : 0;
    int has_ibl      = (gpu->env_image_view != VK_NULL_HANDLE) ? 1 : 0;
    int rt_tex_count = has_textures ? gpu->mat_tex_count : 0;

    /* Storage image is shared across parities — same target the rgen
     * imageStore-path writes. Only meaningful when useDirectOut == 0. */
    VkDescriptorImageInfo img_info = {0};
    img_info.imageView   = gpu->tiled_image_view;
    img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    /* Texture array (binding 4) — same source the tiled RT pipeline uses
     * (gpu_vulkan.c:13585-13599). Built once and reused for both parities. */
    VkDescriptorImageInfo* tex_infos = NULL;
    if (has_textures) {
        tex_infos = (VkDescriptorImageInfo*)calloc(
            (size_t)rt_tex_count, sizeof(VkDescriptorImageInfo));
        for (int t = 0; t < rt_tex_count; t++) {
            tex_infos[t].sampler     = gpu->mat_sampler;
            tex_infos[t].imageView   = gpu->mat_image_views[t]
                ? gpu->mat_image_views[t] : gpu->mat_image_views[0];
            tex_infos[t].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }

    /* IBL infos (bindings 5/6/7) — mirror gpu_vulkan.c:5481-5493. */
    VkDescriptorImageInfo ibl_infos[3] = {0};
    if (has_ibl) {
        ibl_infos[0].sampler     = gpu->env_sampler;
        ibl_infos[0].imageView   = gpu->env_image_view;
        ibl_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        ibl_infos[1].sampler     = gpu->env_sampler;
        ibl_infos[1].imageView   = gpu->brdf_lut_view;
        ibl_infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        ibl_infos[2].sampler     = gpu->irr_sampler ? gpu->irr_sampler : gpu->env_sampler;
        ibl_infos[2].imageView   = gpu->irr_image_view ? gpu->irr_image_view
                                                       : gpu->env_image_view;
        ibl_infos[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    for (int parity = 0; parity < 2; parity++) {
        VkDescriptorSet ds = (parity == 0) ? gpu->deferred_desc_set
                                           : gpu->deferred_desc_set_b;
        if (!ds) continue;

        /* Pick the interop buffer matching the parity. If interop is
         * inactive (single-cam / showcase), fall back to scene so the
         * descriptor write is still legal — the dispatch is gated upstream
         * on direct-write mode anyway. */
        VkBuffer out = (parity == 0)
            ? (gpu->interop_buf[0] ? gpu->interop_buf[0] : scene)
            : (gpu->interop_buf[1] ? gpu->interop_buf[1] : scene);

        VkDescriptorBufferInfo bi_scene = {0};
        bi_scene.buffer = scene; bi_scene.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo bi_mat = {0};
        if (has_mat_ssbo) { bi_mat.buffer = gpu->mat_ssbo_buf; bi_mat.range = VK_WHOLE_SIZE; }
        VkDescriptorBufferInfo bi_out = {0};
        bi_out.buffer = out; bi_out.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo bi_gbuf = {0};
        bi_gbuf.buffer = gbuf; bi_gbuf.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo bi_cam = {0};
        bi_cam.buffer = gpu->tiled_camera_buf ? gpu->tiled_camera_buf : scene;
        bi_cam.range  = VK_WHOLE_SIZE;
        /* Phase C.4: lights buffer + TLAS. Ensure a real zero-light header
         * for procedural/no-light scenes; camera buffers do not contain a
         * valid sceneLights.nlights field. */
        if (!gpu->light_ssbo_buf)
            gpu_upload_lights(gpu, NULL, 0);
        VkDescriptorBufferInfo bi_lights = {0};
        bi_lights.buffer = gpu->light_ssbo_buf ? gpu->light_ssbo_buf
                                               : (gpu->tiled_camera_buf
                                                  ? gpu->tiled_camera_buf : scene);
        bi_lights.range = VK_WHOLE_SIZE;

        /* Phase C.4: w[] sized to 12 (was 10) — bindings 0/13 added. */
        VkWriteDescriptorSet w[12] = {0};
        int nw = 0;

        /* Binding 0: TLAS — Phase C.4. Skip the write if the scene has no
         * TLAS yet (e.g. pipeline built before scene loaded — the
         * descriptor-cache invalidation in renderer.c rebuild_accel will
         * tear down the pipeline once the TLAS is rebuilt; until then,
         * leave the descriptor unwritten and gate the dispatch upstream). */
        VkWriteDescriptorSetAccelerationStructureKHR as_write = {0};
        if (gpu->tlas != VK_NULL_HANDLE) {
            as_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            as_write.accelerationStructureCount = 1;
            as_write.pAccelerationStructures    = &gpu->tlas;
            w[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[nw].pNext           = &as_write;
            w[nw].dstSet          = ds;
            w[nw].dstBinding      = 0;
            w[nw].descriptorCount = 1;
            w[nw].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            nw++;
        }

        /* Binding 1: storage image. */
        if (gpu->tiled_image_view) {
            w[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[nw].dstSet          = ds;
            w[nw].dstBinding      = 1;
            w[nw].descriptorCount = 1;
            w[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[nw].pImageInfo      = &img_info;
            nw++;
        }

        /* Binding 2: SceneData. */
        w[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[nw].dstSet          = ds;
        w[nw].dstBinding      = 2;
        w[nw].descriptorCount = 1;
        w[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[nw].pBufferInfo     = &bi_scene;
        nw++;

        /* Binding 3: MaterialBuffer (Phase C.1) — only when uploaded. */
        if (has_mat_ssbo) {
            w[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[nw].dstSet          = ds;
            w[nw].dstBinding      = 3;
            w[nw].descriptorCount = 1;
            w[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[nw].pBufferInfo     = &bi_mat;
            nw++;
        }

        /* Binding 4: textures[] sampler array (Phase C.1) — only when uploaded. */
        if (has_textures) {
            w[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[nw].dstSet          = ds;
            w[nw].dstBinding      = 4;
            w[nw].descriptorCount = (uint32_t)rt_tex_count;
            w[nw].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[nw].pImageInfo      = tex_infos;
            nw++;
        }

        /* Bindings 5/6/7: IBL (Phase C.3) — only when env loaded. The
         * shader gates fetches on `scene.envMipLevels > 0.0` so the
         * unbound case also stays safe at runtime. */
        if (has_ibl) {
            for (int b = 0; b < 3; b++) {
                w[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[nw].dstSet          = ds;
                w[nw].dstBinding      = (uint32_t)(5 + b);
                w[nw].descriptorCount = 1;
                w[nw].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[nw].pImageInfo      = &ibl_infos[b];
                nw++;
            }
        }

        /* Binding 8: cameras[] (Phase C.3) — used by mode 3 for ray-dir
         * recompute. Always bound (the shader only dereferences in
         * mode 3). */
        w[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[nw].dstSet          = ds;
        w[nw].dstBinding      = 8;
        w[nw].descriptorCount = 1;
        w[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[nw].pBufferInfo     = &bi_cam;
        nw++;

        /* Binding 9: DirectOutput. */
        w[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[nw].dstSet          = ds;
        w[nw].dstBinding      = 9;
        w[nw].descriptorCount = 1;
        w[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[nw].pBufferInfo     = &bi_out;
        nw++;

        /* Binding 13: scene lights — Phase C.4. */
        w[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[nw].dstSet          = ds;
        w[nw].dstBinding      = 13;
        w[nw].descriptorCount = 1;
        w[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[nw].pBufferInfo     = &bi_lights;
        nw++;

        /* Binding 17: G-buffer. */
        w[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[nw].dstSet          = ds;
        w[nw].dstBinding      = 17;
        w[nw].descriptorCount = 1;
        w[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[nw].pBufferInfo     = &bi_gbuf;
        nw++;

        vkUpdateDescriptorSets(gpu->device, (uint32_t)nw, w, 0, NULL);
    }
    free(tex_infos);
}

void gpu_cmd_deferred_shade(Gpu* gpu, const GpuRtTiledPushConstants* pc)
{
    if (!gpu || !pc || !gpu->deferred_built) return;

    /* If the RT path took the cached-cmd-buffer fast path, the entire
     * frame's commands replay from a previously-recorded buffer — the
     * dispatch we'd record here would be appended to a CLOSED cmd buffer.
     * The caller MUST invalidate the cache when r->deferred_shade_enabled
     * toggles, so on the very next frame the path falls into the recording
     * branch and includes our dispatch. Bail safely on replay. */
    if (gpu->tiled_cmd_replay_active) return;

    VkCommandBuffer cmd = gpu->current_cmd;
    if (cmd == VK_NULL_HANDLE) return;

    /* Refresh descriptor writes — cheap and idempotent; ensures binding
     * 0 / 9 / 13 / 17 always point at the most-recent TLAS + interop +
     * lights + gbuffer allocations. */
    gpu_descriptor_writes_for_deferred(gpu);

    /* Barrier 1: protect the G-buffer (RT writes -> COMPUTE reads). */
    VkBufferMemoryBarrier gbuf_bar = {0};
    gbuf_bar.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    gbuf_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    gbuf_bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    gbuf_bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    gbuf_bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    gbuf_bar.buffer        = gpu->tiled_gbuffer_buf ? gpu->tiled_gbuffer_buf
                                                    : gpu->scene_data_buf;
    gbuf_bar.size          = VK_WHOLE_SIZE;

    /* Barrier 2: write-after-write on binding 9 (rgen WRITE -> compute WRITE).
     * Apply to whichever interop buf the active descriptor set will write to. */
    int parity_b = (gpu->direct_write_active && gpu->interop_write_idx == 1) ? 1 : 0;
    VkBuffer out_buf = parity_b ? gpu->interop_buf[1] : gpu->interop_buf[0];

    VkBufferMemoryBarrier out_bar = {0};
    out_bar.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    out_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    out_bar.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    out_bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    out_bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    out_bar.buffer        = out_buf ? out_buf : gpu->scene_data_buf;
    out_bar.size          = VK_WHOLE_SIZE;

    VkBufferMemoryBarrier bars[2] = { gbuf_bar, out_bar };
    int nbars = out_buf ? 2 : 1;

    /* Image barrier for the storage image (rgen WRITE -> compute WRITE).
     * Layout stays GENERAL — same on both sides. */
    VkImageMemoryBarrier img_bar = {0};
    img_bar.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    img_bar.srcAccessMask               = VK_ACCESS_SHADER_WRITE_BIT;
    img_bar.dstAccessMask               = VK_ACCESS_SHADER_WRITE_BIT;
    img_bar.oldLayout                   = VK_IMAGE_LAYOUT_GENERAL;
    img_bar.newLayout                   = VK_IMAGE_LAYOUT_GENERAL;
    img_bar.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    img_bar.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
    img_bar.image                       = gpu->tiled_image;
    img_bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    img_bar.subresourceRange.levelCount = 1;
    img_bar.subresourceRange.layerCount = 1;
    int nimg = gpu->tiled_image ? 1 : 0;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, nbars, bars, nimg, &img_bar);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gpu->deferred_pipeline);

    VkDescriptorSet ds = parity_b ? gpu->deferred_desc_set_b : gpu->deferred_desc_set;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            gpu->deferred_pipeline_layout, 0, 1, &ds, 0, NULL);

    vkCmdPushConstants(cmd, gpu->deferred_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(GpuRtTiledPushConstants), pc);

    uint32_t gx = (gpu->tiled_image_w + 15u) / 16u;
    uint32_t gy = (gpu->tiled_image_h + 15u) / 16u;
    /* Phase C.4 mechanism hunt: wrap the deferred compute dispatch so we
     * can compare its GPU cost against the trace-rays GPU time saved. */
    gpu_phase_begin(gpu, cmd, GPU_PHASE_DEFERRED_COMPUTE, "deferred_compute");
    vkCmdDispatch(cmd, gx, gy, 1);
    gpu_phase_end(gpu, cmd, GPU_PHASE_DEFERRED_COMPUTE);

    /* Post-dispatch barrier — make compute image/SSBO writes visible to the
     * subsequent transfer (vkCmdCopyImageToBuffer for staging) and to the
     * RT pipeline of the next frame (which may consume binding 9 readbacks). */
    VkBufferMemoryBarrier post_out_bar = out_bar;
    post_out_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    post_out_bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    VkImageMemoryBarrier post_img_bar = img_bar;
    post_img_bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    post_img_bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0, 0, NULL, out_buf ? 1 : 0, &post_out_bar, nimg, &post_img_bar);
}

int gpu_cast_rays(Gpu* gpu,
                  const float* origins, const float* directions,
                  int num_rays, float max_distance,
                  float* out_distances, int* out_mesh_ids, float* out_normals,
                  float* out_hit_positions)
{
    if (!gpu->raycast_built) {
        fprintf(stderr, "gpu_vulkan: raycast pipeline not built\n");
        return 0;
    }
    if (num_rays <= 0) return 1;

    /* Ensure buffers are large enough */
    if (!raycast_ensure_buffers(gpu, (uint32_t)num_rays)) return 0;

    VkDeviceSize input_bytes  = (VkDeviceSize)num_rays * 6 * sizeof(float);
    VkDeviceSize output_bytes = (VkDeviceSize)num_rays * 8 * sizeof(float);

    /* Upload ray data to staging buffer: split layout — origins then directions */
    {
        float* mapped = (float*)gpu->raycast_staging_in_mapped;
        memcpy(mapped, origins, (size_t)num_rays * 3 * sizeof(float));
        memcpy(mapped + num_rays * 3, directions, (size_t)num_rays * 3 * sizeof(float));
    }

    /* Record command buffer: staging upload → compute → staging readback */
    VkCommandBufferAllocateInfo cmd_ai = {0};
    cmd_ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_ai.commandPool        = gpu->command_pool;
    cmd_ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(gpu->device, &cmd_ai, &cmd);

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);

    /* 1. Copy staging_in → device-local input */
    VkBufferCopy copy_region = {0};
    copy_region.size = input_bytes;
    vkCmdCopyBuffer(cmd, gpu->raycast_staging_in, gpu->raycast_input_buf, 1, &copy_region);

    /* Barrier: transfer complete → compute can read */
    VkBufferMemoryBarrier transfer_to_compute = {0};
    transfer_to_compute.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    transfer_to_compute.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    transfer_to_compute.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    transfer_to_compute.buffer              = gpu->raycast_input_buf;
    transfer_to_compute.offset              = 0;
    transfer_to_compute.size                = input_bytes;
    transfer_to_compute.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    transfer_to_compute.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 1, &transfer_to_compute, 0, NULL);

    /* 2. Bind compute pipeline and dispatch */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gpu->raycast_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            gpu->raycast_pipeline_layout, 0, 1,
                            &gpu->raycast_desc_set, 0, NULL);

    GpuRaycastPushConstants pc;
    pc.num_rays     = (uint32_t)num_rays;
    pc.max_distance = max_distance;
    vkCmdPushConstants(cmd, gpu->raycast_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t groups = ((uint32_t)num_rays + 255) / 256;
    vkCmdDispatch(cmd, groups, 1, 1);

    /* Barrier: compute write complete → transfer can read */
    VkBufferMemoryBarrier compute_to_transfer = {0};
    compute_to_transfer.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    compute_to_transfer.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    compute_to_transfer.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    compute_to_transfer.buffer              = gpu->raycast_output_buf;
    compute_to_transfer.offset              = 0;
    compute_to_transfer.size                = output_bytes;
    compute_to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    compute_to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 1, &compute_to_transfer, 0, NULL);

    /* 3. Copy device-local output → staging_out */
    copy_region.size = output_bytes;
    vkCmdCopyBuffer(cmd, gpu->raycast_output_buf, gpu->raycast_staging_out, 1, &copy_region);

    vkEndCommandBuffer(cmd);

    /* Submit and wait */
    VkSubmitInfo submit = {0};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;
    vkQueueSubmit(gpu->graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(gpu->graphics_queue);
    vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &cmd);

    /* Read back results from staging buffer: split layout */
    {
        const float* mapped = (const float*)gpu->raycast_staging_out_mapped;
        memcpy(out_distances,     mapped, (size_t)num_rays * sizeof(float));
        if (out_mesh_ids) {
            const float* ids = mapped + num_rays;
            for (int i = 0; i < num_rays; i++)
                out_mesh_ids[i] = ids[i] >= 0.0f ? (int)(ids[i] + 0.5f) : -1;
        }
        memcpy(out_normals,       mapped + num_rays * 2, (size_t)num_rays * 3 * sizeof(float));
        memcpy(out_hit_positions, mapped + num_rays * 2 + num_rays * 3, (size_t)num_rays * 3 * sizeof(float));
    }

    return 1;
}

/* ---- Async raycast: submit without blocking ---- */
int gpu_cast_rays_async(Gpu* gpu,
                        const float* origins, const float* directions,
                        int num_rays, float max_distance)
{
    if (!gpu->raycast_built) {
        fprintf(stderr, "gpu_vulkan: raycast pipeline not built\n");
        return 0;
    }
    if (num_rays <= 0) return 1;

    /* Wait for any previous async dispatch to finish first */
    if (gpu->raycast_async_pending) {
        vkWaitForFences(gpu->device, 1, &gpu->raycast_async_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(gpu->device, 1, &gpu->raycast_async_fence);
        if (gpu->raycast_async_cmd) {
            vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &gpu->raycast_async_cmd);
            gpu->raycast_async_cmd = VK_NULL_HANDLE;
        }
        gpu->raycast_async_pending = 0;
    }

    if (!raycast_ensure_buffers(gpu, (uint32_t)num_rays)) return 0;

    VkDeviceSize input_bytes  = (VkDeviceSize)num_rays * 6 * sizeof(float);
    VkDeviceSize output_bytes = (VkDeviceSize)num_rays * 8 * sizeof(float);

    /* Upload ray data to staging buffer */
    {
        float* mapped = (float*)gpu->raycast_staging_in_mapped;
        memcpy(mapped, origins, (size_t)num_rays * 3 * sizeof(float));
        memcpy(mapped + num_rays * 3, directions, (size_t)num_rays * 3 * sizeof(float));
    }

    /* Allocate command buffer */
    VkCommandBufferAllocateInfo cmd_ai = {0};
    cmd_ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_ai.commandPool        = gpu->command_pool;
    cmd_ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(gpu->device, &cmd_ai, &cmd);

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);

    /* 1. Copy staging_in → device-local input */
    VkBufferCopy copy_region = {0};
    copy_region.size = input_bytes;
    vkCmdCopyBuffer(cmd, gpu->raycast_staging_in, gpu->raycast_input_buf, 1, &copy_region);

    /* Barrier: transfer → compute */
    VkBufferMemoryBarrier transfer_to_compute = {0};
    transfer_to_compute.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    transfer_to_compute.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    transfer_to_compute.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    transfer_to_compute.buffer              = gpu->raycast_input_buf;
    transfer_to_compute.offset              = 0;
    transfer_to_compute.size                = input_bytes;
    transfer_to_compute.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    transfer_to_compute.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 1, &transfer_to_compute, 0, NULL);

    /* 2. Dispatch compute */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gpu->raycast_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            gpu->raycast_pipeline_layout, 0, 1,
                            &gpu->raycast_desc_set, 0, NULL);

    GpuRaycastPushConstants pc;
    pc.num_rays     = (uint32_t)num_rays;
    pc.max_distance = max_distance;
    vkCmdPushConstants(cmd, gpu->raycast_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t groups = ((uint32_t)num_rays + 255) / 256;
    vkCmdDispatch(cmd, groups, 1, 1);

    /* Barrier: compute → transfer */
    VkBufferMemoryBarrier compute_to_transfer = {0};
    compute_to_transfer.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    compute_to_transfer.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    compute_to_transfer.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    compute_to_transfer.buffer              = gpu->raycast_output_buf;
    compute_to_transfer.offset              = 0;
    compute_to_transfer.size                = output_bytes;
    compute_to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    compute_to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 1, &compute_to_transfer, 0, NULL);

    /* 3. Copy device-local output → staging_out */
    copy_region.size = output_bytes;
    vkCmdCopyBuffer(cmd, gpu->raycast_output_buf, gpu->raycast_staging_out, 1, &copy_region);

    vkEndCommandBuffer(cmd);

    /* Submit with fence — returns immediately */
    vkResetFences(gpu->device, 1, &gpu->raycast_async_fence);
    VkSubmitInfo submit = {0};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;
    vkQueueSubmit(gpu->graphics_queue, 1, &submit, gpu->raycast_async_fence);

    gpu->raycast_async_cmd      = cmd;
    gpu->raycast_async_num_rays = (uint32_t)num_rays;
    gpu->raycast_async_pending  = 1;

    return 1;
}

int gpu_cast_rays_wait(Gpu* gpu,
                       float* out_distances, float* out_normals,
                       float* out_hit_positions)
{
    if (!gpu->raycast_async_pending) return 1;  /* nothing pending */

    vkWaitForFences(gpu->device, 1, &gpu->raycast_async_fence, VK_TRUE, UINT64_MAX);

    uint32_t num_rays = gpu->raycast_async_num_rays;

    /* Read back from staging buffer */
    {
        const float* mapped = (const float*)gpu->raycast_staging_out_mapped;
        memcpy(out_distances,     mapped, (size_t)num_rays * sizeof(float));
        memcpy(out_normals,       mapped + num_rays * 2, (size_t)num_rays * 3 * sizeof(float));
        memcpy(out_hit_positions, mapped + num_rays * 2 + num_rays * 3, (size_t)num_rays * 3 * sizeof(float));
    }

    /* Cleanup */
    if (gpu->raycast_async_cmd) {
        vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &gpu->raycast_async_cmd);
        gpu->raycast_async_cmd = VK_NULL_HANDLE;
    }
    gpu->raycast_async_pending = 0;

    return 1;
}

/* ---- Raycast CUDA interop: exportable buffers + GPU-only dispatch ---- */

static int raycast_ensure_buffers_exportable(Gpu* gpu, uint32_t num_rays)
{
    if (num_rays <= gpu->raycast_max_rays && gpu->raycast_exportable) return 1;

    vkDeviceWaitIdle(gpu->device);
    raycast_destroy_buffers(gpu);

    uint32_t cap = 1;
    while (cap < num_rays) cap <<= 1;

    VkDeviceSize input_size  = (VkDeviceSize)cap * 6 * sizeof(float);
    VkDeviceSize output_size = (VkDeviceSize)cap * 8 * sizeof(float);

    VkMemoryRequirements req;

    /* ---- Device-local input SSBO with external memory export ---- */
    {
        VkExternalMemoryBufferCreateInfo ext_buf_ci = {0};
        ext_buf_ci.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ext_buf_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.pNext = &ext_buf_ci;
        bci.size  = input_size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->raycast_input_buf) != VK_SUCCESS)
            return 0;

        vkGetBufferMemoryRequirements(gpu->device, gpu->raycast_input_buf, &req);

        VkExportMemoryAllocateInfo export_mai = {0};
        export_mai.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        export_mai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext           = &export_mai;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (gpu_alloc_memory(gpu, &mai, &gpu->raycast_input_mem) != VK_SUCCESS) return 0;
        vkBindBufferMemory(gpu->device, gpu->raycast_input_buf, gpu->raycast_input_mem, 0);
        gpu->raycast_input_alloc_size = req.size;
    }

    /* ---- Device-local output SSBO with external memory export ---- */
    {
        VkExternalMemoryBufferCreateInfo ext_buf_ci = {0};
        ext_buf_ci.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ext_buf_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.pNext = &ext_buf_ci;
        bci.size  = output_size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->raycast_output_buf) != VK_SUCCESS)
            return 0;

        vkGetBufferMemoryRequirements(gpu->device, gpu->raycast_output_buf, &req);

        VkExportMemoryAllocateInfo export_mai = {0};
        export_mai.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        export_mai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext           = &export_mai;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (gpu_alloc_memory(gpu, &mai, &gpu->raycast_output_mem) != VK_SUCCESS) return 0;
        vkBindBufferMemory(gpu->device, gpu->raycast_output_buf, gpu->raycast_output_mem, 0);
        gpu->raycast_output_alloc_size = req.size;
    }

    /* ---- Staging buffers (still needed for CPU fallback path) ---- */
    {
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        VkMemoryAllocateInfo mai = {0};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

        bci.size  = input_size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->raycast_staging_in) != VK_SUCCESS) return 0;
        vkGetBufferMemoryRequirements(gpu->device, gpu->raycast_staging_in, &req);
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (gpu_alloc_memory(gpu, &mai, &gpu->raycast_staging_in_mem) != VK_SUCCESS) return 0;
        vkBindBufferMemory(gpu->device, gpu->raycast_staging_in, gpu->raycast_staging_in_mem, 0);
        vkMapMemory(gpu->device, gpu->raycast_staging_in_mem, 0, bci.size, 0, &gpu->raycast_staging_in_mapped);

        bci.size  = output_size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->raycast_staging_out) != VK_SUCCESS) return 0;
        vkGetBufferMemoryRequirements(gpu->device, gpu->raycast_staging_out, &req);
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (gpu_alloc_memory(gpu, &mai, &gpu->raycast_staging_out_mem) != VK_SUCCESS) return 0;
        vkBindBufferMemory(gpu->device, gpu->raycast_staging_out, gpu->raycast_staging_out_mem, 0);
        vkMapMemory(gpu->device, gpu->raycast_staging_out_mem, 0, bci.size, 0, &gpu->raycast_staging_out_mapped);
    }

    gpu->raycast_max_rays = cap;
    gpu->raycast_exportable = 1;

    /* Update descriptor set */
    if (gpu->raycast_desc_set) {
        VkDescriptorBufferInfo in_info  = {0};
        in_info.buffer = gpu->raycast_input_buf;
        in_info.range  = input_size;
        VkDescriptorBufferInfo out_info = {0};
        out_info.buffer = gpu->raycast_output_buf;
        out_info.range  = output_size;
        VkWriteDescriptorSet writes[2] = {{0}, {0}};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = gpu->raycast_desc_set;
        writes[0].dstBinding = 1;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &in_info;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = gpu->raycast_desc_set;
        writes[1].dstBinding = 2;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &out_info;
        vkUpdateDescriptorSets(gpu->device, 2, writes, 0, NULL);
    }

    return 1;
}

int gpu_raycast_get_interop_fds(Gpu* gpu, int num_rays,
                                 int* out_input_fd, uint64_t* out_input_size,
                                 int* out_output_fd, uint64_t* out_output_size,
                                 uint32_t* out_max_rays)
{
    if (!gpu->interop_available) return 0;
    if (!raycast_ensure_buffers_exportable(gpu, (uint32_t)num_rays)) return 0;

    /* Export input buffer memory fd */
    VkMemoryGetFdInfoKHR fd_info = {0};
    fd_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    fd_info.memory = gpu->raycast_input_mem;
    int in_fd = -1;
    if (vkGetMemoryFdKHR(gpu->device, &fd_info, &in_fd) != VK_SUCCESS)
        return 0;

    fd_info.memory = gpu->raycast_output_mem;
    int out_fd = -1;
    if (vkGetMemoryFdKHR(gpu->device, &fd_info, &out_fd) != VK_SUCCESS) {
        close(in_fd);
        return 0;
    }

    *out_input_fd    = in_fd;
    *out_input_size  = gpu->raycast_input_alloc_size;
    *out_output_fd   = out_fd;
    *out_output_size = gpu->raycast_output_alloc_size;
    *out_max_rays    = gpu->raycast_max_rays;
    return 1;
}

/* Dispatch raycast compute without staging upload — data already in device-local
 * buffer via CUDA external memory write. */
int gpu_cast_rays_gpu(Gpu* gpu, int num_rays, float max_distance)
{
    if (!gpu->raycast_built) return 0;
    if (num_rays <= 0) return 1;
    if ((uint32_t)num_rays > gpu->raycast_max_rays) return 0;

    /* Wait for any previous async dispatch */
    if (gpu->raycast_async_pending) {
        vkWaitForFences(gpu->device, 1, &gpu->raycast_async_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(gpu->device, 1, &gpu->raycast_async_fence);
        if (gpu->raycast_async_cmd) {
            vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &gpu->raycast_async_cmd);
            gpu->raycast_async_cmd = VK_NULL_HANDLE;
        }
        gpu->raycast_async_pending = 0;
    }

    VkDeviceSize output_bytes = (VkDeviceSize)num_rays * 8 * sizeof(float);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cmd_ai = {0};
    cmd_ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_ai.commandPool        = gpu->command_pool;
    cmd_ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(gpu->device, &cmd_ai, &cmd);

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);

    /* No staging copy — CUDA already wrote to the device-local input buffer.
     * Just need a memory barrier to ensure CUDA writes are visible to Vulkan. */
    VkMemoryBarrier external_barrier = {0};
    external_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    external_barrier.srcAccessMask = 0;  /* external (CUDA) writes */
    external_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &external_barrier, 0, NULL, 0, NULL);

    /* 2. Dispatch compute */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gpu->raycast_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        gpu->raycast_pipeline_layout, 0, 1, &gpu->raycast_desc_set, 0, NULL);

    struct { int n; float max_d; } push = { num_rays, max_distance };
    vkCmdPushConstants(cmd, gpu->raycast_pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t groups = ((uint32_t)num_rays + 63) / 64;
    vkCmdDispatch(cmd, groups, 1, 1);

    /* 3. Barrier: compute output ready for external (CUDA) read */
    VkMemoryBarrier output_barrier = {0};
    output_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    output_barrier.dstAccessMask = 0;  /* external (CUDA) will read */
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 1, &output_barrier, 0, NULL, 0, NULL);

    vkEndCommandBuffer(cmd);

    /* Submit with fence */
    VkSubmitInfo si = {0};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(gpu->graphics_queue, 1, &si, gpu->raycast_async_fence);

    gpu->raycast_async_cmd = cmd;
    gpu->raycast_async_pending = 1;
    gpu->raycast_async_num_rays = (uint32_t)num_rays;

    return 1;
}

/* Wait for async raycast fence only (no staging readback).
 * Used by the GPU interop path where CUDA reads from the output buffer directly. */
int gpu_cast_rays_wait_fence(Gpu* gpu)
{
    if (!gpu->raycast_async_pending) return 1;

    vkWaitForFences(gpu->device, 1, &gpu->raycast_async_fence, VK_TRUE, UINT64_MAX);

    if (gpu->raycast_async_cmd) {
        vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &gpu->raycast_async_cmd);
        gpu->raycast_async_cmd = VK_NULL_HANDLE;
    }
    gpu->raycast_async_pending = 0;

    return 1;
}

uint64_t gpu_get_allocated_memory(Gpu* gpu)
{
    return gpu->total_allocated_gpu_mem;
}

uint64_t gpu_get_heap_size(Gpu* gpu)
{
    return (uint64_t)rt_device_local_heap_size(gpu);
}

/* ---- Material system ---- */

static VkCommandBuffer begin_single_command(Gpu* gpu)
{
    VkCommandBufferAllocateInfo ai = {0};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = gpu->command_pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(gpu->device, &ai, &cmd);

    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

static void end_single_command(Gpu* gpu, VkCommandBuffer cmd)
{
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si = {0};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(gpu->graphics_queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(gpu->graphics_queue);
    vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &cmd);
}

/* Phase-tagged variant: when `phase < GPU_PHASE_COUNT`, the staging->device
 * copy is wrapped in a debug-utils label + timestamp pair, then resolved
 * once end_single_command's queue-idle returns. Pass GPU_PHASE_COUNT for
 * untagged uploads. */
static void create_buffer_with_data_phase(Gpu* gpu, VkBufferUsageFlags usage,
                                          const void* data, VkDeviceSize size,
                                          VkBuffer* out_buf, VkDeviceMemory* out_mem,
                                          GpuPhase phase, const char* label)
{
    /* Staging buffer */
    VkBuffer staging;
    VkDeviceMemory staging_mem;

    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    vkCreateBuffer(gpu->device, &bci, NULL, &staging);
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gpu->device, staging, &req);

    VkMemoryAllocateInfo mai = {0};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(gpu->device, &mai, NULL, &staging_mem);
    vkBindBufferMemory(gpu->device, staging, staging_mem, 0);

    void* mapped;
    vkMapMemory(gpu->device, staging_mem, 0, size, 0, &mapped);
    memcpy(mapped, data, (size_t)size);
    vkUnmapMemory(gpu->device, staging_mem);

    /* Device-local buffer */
    bci.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vkCreateBuffer(gpu->device, &bci, NULL, out_buf);
    vkGetBufferMemoryRequirements(gpu->device, *out_buf, &req);

    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(gpu->device, &mai, NULL, out_mem);
    vkBindBufferMemory(gpu->device, *out_buf, *out_mem, 0);

    /* Copy (optionally bracketed with phase label + timestamps). */
    VkCommandBuffer cmd2 = begin_single_command(gpu);
    int label_open = 0;
    if (phase < GPU_PHASE_COUNT) {
        gpu_phase_begin(gpu, cmd2, phase, label ? label : "staging_upload");
        label_open = 1;
    }
    VkBufferCopy region = {0};
    region.size = size;
    vkCmdCopyBuffer(cmd2, staging, *out_buf, 1, &region);
    if (label_open) {
        gpu_phase_end(gpu, cmd2, phase);
    }
    end_single_command(gpu, cmd2);
    if (label_open && gpu->timestamps_supported) {
        gpu_phase_resolve(gpu, phase);
    }

    vkDestroyBuffer(gpu->device, staging, NULL);
    vkFreeMemory(gpu->device, staging_mem, NULL);
}

static void create_buffer_with_data(Gpu* gpu, VkBufferUsageFlags usage,
                                    const void* data, VkDeviceSize size,
                                    VkBuffer* out_buf, VkDeviceMemory* out_mem)
{
    create_buffer_with_data_phase(gpu, usage, data, size, out_buf, out_mem,
                                   GPU_PHASE_COUNT, NULL);
}

/* Profiling (iter 4): split gpu_upload_materials' per-texture cost into staging
 * alloc+memcpy, image alloc, and command submit+WaitIdle+mipgen — to target the
 * right batching/pooling fix (197x per-texture allocs vs 197x WaitIdle).
 * Output-neutral. */
static double g_up_staging_ms = 0.0, g_up_image_ms = 0.0, g_up_submit_ms = 0.0;
static double ctex_now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* Reusable staging buffer for material texture uploads (iter 5): avoids 197x
 * vkCreateBuffer/vkAllocateMemory/free (~1.1s). Reused serially — each texture
 * does an end_single_command WaitIdle before the next reuses it, so the buffer
 * is idle on reuse. Grows (rounded to 16MB) to the largest texture; freed by
 * gpu_upload_staging_pool_free() after the upload loop. */
static VkBuffer       g_stg_buf    = VK_NULL_HANDLE;
static VkDeviceMemory g_stg_mem    = VK_NULL_HANDLE;
static VkDeviceSize   g_stg_cap    = 0;
static void*          g_stg_mapped = NULL;

static void staging_pool_ensure(Gpu* gpu, VkDeviceSize size) {
    if (g_stg_buf != VK_NULL_HANDLE && size <= g_stg_cap) return;
    VkDeviceSize chunk = (VkDeviceSize)16 << 20;
    VkDeviceSize want = ((size + chunk - 1) / chunk) * chunk;
    if (g_stg_mapped) { vkUnmapMemory(gpu->device, g_stg_mem); g_stg_mapped = NULL; }
    if (g_stg_buf != VK_NULL_HANDLE) vkDestroyBuffer(gpu->device, g_stg_buf, NULL);
    if (g_stg_mem != VK_NULL_HANDLE) vkFreeMemory(gpu->device, g_stg_mem, NULL);
    g_stg_buf = VK_NULL_HANDLE; g_stg_mem = VK_NULL_HANDLE; g_stg_cap = 0;

    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = want;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(gpu->device, &bci, NULL, &g_stg_buf) != VK_SUCCESS) {
        g_stg_buf = VK_NULL_HANDLE; return;
    }
    VkMemoryRequirements sreq;
    vkGetBufferMemoryRequirements(gpu->device, g_stg_buf, &sreq);
    VkMemoryAllocateInfo smai = {0};
    smai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    smai.allocationSize  = sreq.size;
    smai.memoryTypeIndex = find_memory_type(gpu->physical_device, sreq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(gpu->device, &smai, NULL, &g_stg_mem) != VK_SUCCESS) {
        vkDestroyBuffer(gpu->device, g_stg_buf, NULL);
        g_stg_buf = VK_NULL_HANDLE; g_stg_mem = VK_NULL_HANDLE; return;
    }
    vkBindBufferMemory(gpu->device, g_stg_buf, g_stg_mem, 0);
    vkMapMemory(gpu->device, g_stg_mem, 0, VK_WHOLE_SIZE, 0, &g_stg_mapped);
    g_stg_cap = want;
}

static void gpu_upload_staging_pool_free(Gpu* gpu) {
    if (g_stg_mapped) { vkUnmapMemory(gpu->device, g_stg_mem); g_stg_mapped = NULL; }
    if (g_stg_buf != VK_NULL_HANDLE) vkDestroyBuffer(gpu->device, g_stg_buf, NULL);
    if (g_stg_mem != VK_NULL_HANDLE) vkFreeMemory(gpu->device, g_stg_mem, NULL);
    g_stg_buf = VK_NULL_HANDLE; g_stg_mem = VK_NULL_HANDLE; g_stg_cap = 0;
}

static void create_texture_image(Gpu* gpu, const GpuTextureData* tex,
                                 VkImage* out_image, VkDeviceMemory* out_mem,
                                 VkImageView* out_view)
{
    VkDeviceSize size = (VkDeviceSize)tex->width * tex->height * 4;

    /* Full mip chain so the ray-cone LOD path (roughness / minification)
     * has band-limited levels instead of aliasing mip 0. Matches ovrtx. */
    int mips = 1;
    {
        int mw = tex->width, mh = tex->height;
        while (mw > 1 || mh > 1) { mw /= 2; mh /= 2; mips++; }
    }
    VkFormat fmt = tex->is_srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;

    /* Staging buffer — pooled (one reusable host-visible buffer across all
     * textures; safe because each texture's WaitIdle completes before reuse). */
    double _t_stg = ctex_now_ms();
    staging_pool_ensure(gpu, size);
    if (g_stg_mapped) memcpy(g_stg_mapped, tex->pixels, (size_t)size);
    g_up_staging_ms += ctex_now_ms() - _t_stg;

    /* req / mai are reused by the image allocation below. */
    VkMemoryRequirements req;
    VkMemoryAllocateInfo mai = {0};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;

    /* Image */
    double _t_img = ctex_now_ms();
    VkImageCreateInfo ici = {0};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = fmt;
    ici.extent.width  = (uint32_t)tex->width;
    ici.extent.height = (uint32_t)tex->height;
    ici.extent.depth  = 1;
    ici.mipLevels     = (uint32_t)mips;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT
                      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                      | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(gpu->device, &ici, NULL, out_image);

    vkGetImageMemoryRequirements(gpu->device, *out_image, &req);
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(gpu->device, &mai, NULL, out_mem);
    vkBindImageMemory(gpu->device, *out_image, *out_mem, 0);
    g_up_image_ms += ctex_now_ms() - _t_img;

    double _t_sub = ctex_now_ms();
    VkCommandBuffer cmd2 = begin_single_command(gpu);

    /* Transition all mips UNDEFINED → TRANSFER_DST. */
    VkImageMemoryBarrier barrier = {0};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                           = *out_image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = (uint32_t)mips;
    barrier.subresourceRange.layerCount     = 1;
    barrier.srcAccessMask                   = 0;
    barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd2,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier);

    /* Upload mip 0. */
    VkBufferImageCopy region = {0};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel   = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width  = (uint32_t)tex->width;
    region.imageExtent.height = (uint32_t)tex->height;
    region.imageExtent.depth  = 1;
    vkCmdCopyBufferToImage(cmd2, g_stg_buf, *out_image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    /* Generate remaining mips via linear blits (box filter). sRGB formats
     * support linear blits on any Vulkan 1.0 conformant device. */
    int mw = tex->width, mh = tex->height;
    for (int m = 1; m < mips; m++) {
        /* src level: TRANSFER_DST → TRANSFER_SRC */
        barrier.subresourceRange.baseMipLevel = (uint32_t)(m - 1);
        barrier.subresourceRange.levelCount   = 1;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd2,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 0, NULL, 1, &barrier);

        int next_w = mw > 1 ? mw / 2 : 1;
        int next_h = mh > 1 ? mh / 2 : 1;

        VkImageBlit blit = {0};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel   = (uint32_t)(m - 1);
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1].x = mw;
        blit.srcOffsets[1].y = mh;
        blit.srcOffsets[1].z = 1;
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel   = (uint32_t)m;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1].x = next_w;
        blit.dstOffsets[1].y = next_h;
        blit.dstOffsets[1].z = 1;
        vkCmdBlitImage(cmd2, *out_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       *out_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_LINEAR);

        /* src level (m-1): TRANSFER_SRC → SHADER_READ */
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd2,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, NULL, 0, NULL, 1, &barrier);

        mw = next_w;
        mh = next_h;
    }

    /* Last mip: TRANSFER_DST → SHADER_READ. */
    barrier.subresourceRange.baseMipLevel = (uint32_t)(mips - 1);
    barrier.subresourceRange.levelCount   = 1;
    barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd2,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier);

    end_single_command(gpu, cmd2);
    g_up_submit_ms += ctex_now_ms() - _t_sub;

    /* Image view — all mip levels. */
    VkImageViewCreateInfo vci = {0};
    vci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image                           = *out_image;
    vci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    vci.format                          = fmt;
    vci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel   = 0;
    vci.subresourceRange.levelCount     = (uint32_t)mips;
    vci.subresourceRange.layerCount     = 1;
    vkCreateImageView(gpu->device, &vci, NULL, out_view);
}

int gpu_upload_materials(Gpu* gpu,
                         const GpuMaterialParams* materials, int nmaterials,
                         const GpuTextureData* textures, int ntextures)
{
    if (!gpu) return 0;

    /* Allow nmaterials == 0 — create a 1-material placeholder SSBO so the
     * descriptor set + env image bindings can still be wired. This lets
     * dome-light-only USDAs (e.g. the chess wrapper that breaks variant
     * material binding through references) install IBL without first
     * having any scene materials. The shader doesn't read materials[0]
     * for material-less hits since hasMaterials==0 short-circuits in
     * raytrace.rchit.glsl. */
    GpuMaterialParams placeholder_mat = {0};
    GpuMaterialParams fallback_mat = {0};
    GpuMaterialParams* appended_mats = NULL;
    int only_placeholder = 0;
    if (nmaterials == 0) {
        /* Safe placeholder: white base_color * displayColor stays as
         * displayColor; opacity=1 avoids spurious transmission;
         * use_vertex_color=1 tells the raster shader to fall back to
         * fragColor (per-mesh displayColor). The RT shader gates on
         * scene.hasMaterials==0 separately (set via mat_only_placeholder
         * in the SceneData SSBO) so it never reads this slot. */
        placeholder_mat.base_color[0] = 1.0f;
        placeholder_mat.base_color[1] = 1.0f;
        placeholder_mat.base_color[2] = 1.0f;
        placeholder_mat.base_color[3] = 1.0f;
        placeholder_mat.opacity       = 1.0f;
        placeholder_mat.ior           = 1.5f;
        placeholder_mat.roughness     = 0.5f;
        placeholder_mat.occlusion     = 1.0f;
        placeholder_mat.use_vertex_color = 1;
        for (int t = 0; t < 8; t++) placeholder_mat.tex_indices[t] = -1;
        placeholder_mat.tex_subsurface_weight = -1;
        placeholder_mat.tex_transmission_weight = -1;
        materials = &placeholder_mat;
        nmaterials = 1;
        ntextures = 0;
        only_placeholder = 1;
    } else {
        /* Extra slot for meshes with no resolved material binding. Using
         * displayColor keeps authored per-mesh color instead of borrowing
         * material 0, which is arbitrary in composed USD scenes. */
        fallback_mat.base_color[0] = 1.0f;
        fallback_mat.base_color[1] = 1.0f;
        fallback_mat.base_color[2] = 1.0f;
        fallback_mat.base_color[3] = 1.0f;
        fallback_mat.opacity       = 1.0f;
        fallback_mat.ior           = 1.5f;
        fallback_mat.roughness     = 0.55f;
        fallback_mat.occlusion     = 1.0f;
        fallback_mat.use_vertex_color = 1;
        fallback_mat.normal_tex_scale[0] = 2.0f;
        fallback_mat.normal_tex_scale[1] = 2.0f;
        fallback_mat.normal_tex_scale[2] = 2.0f;
        fallback_mat.normal_tex_scale[3] = 1.0f;
        fallback_mat.normal_tex_bias[0] = -1.0f;
        fallback_mat.normal_tex_bias[1] = -1.0f;
        fallback_mat.normal_tex_bias[2] = -1.0f;
        fallback_mat.normal_tex_bias[3] = 0.0f;
        for (int t = 0; t < 8; t++) fallback_mat.tex_indices[t] = -1;
        fallback_mat.tex_subsurface_weight = -1;
        fallback_mat.tex_transmission_weight = -1;

        appended_mats = (GpuMaterialParams*)malloc(
            (size_t)(nmaterials + 1) * sizeof(GpuMaterialParams));
        if (!appended_mats) return 0;
        memcpy(appended_mats, materials,
               (size_t)nmaterials * sizeof(GpuMaterialParams));
        appended_mats[nmaterials] = fallback_mat;
        materials = appended_mats;
        nmaterials += 1;
    }

    gpu->mat_count = nmaterials;
    gpu->mat_only_placeholder = only_placeholder;
    gpu->mat_tex_count = ntextures;

    /* Upload material SSBO */
    VkDeviceSize ssbo_size = (VkDeviceSize)nmaterials * sizeof(GpuMaterialParams);
    create_buffer_with_data(gpu, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            materials, ssbo_size,
                            &gpu->mat_ssbo_buf, &gpu->mat_ssbo_mem);
    free(appended_mats);

    fprintf(stderr, "gpu_vulkan: uploaded material SSBO (%d materials, %zu bytes%s)\n",
            nmaterials, (size_t)ssbo_size,
            only_placeholder ? "" : ", last slot is unbound fallback");

    /* Create sampler: 16x anisotropy + full mip chain. These are common
     * high-quality defaults and match public ovrtx output closely. */
    {
        VkSamplerCreateInfo sci = {0};
        sci.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter        = VK_FILTER_LINEAR;
        sci.minFilter        = VK_FILTER_LINEAR;
        sci.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.anisotropyEnable = VK_TRUE;
        sci.maxAnisotropy    = 16.0f;
        sci.minLod           = 0.0f;
        sci.maxLod           = VK_LOD_CLAMP_NONE;
        vkCreateSampler(gpu->device, &sci, NULL, &gpu->mat_sampler);
    }

    /* Upload textures */
    int actual_tex = (ntextures > 0) ? ntextures : 1; /* Need at least 1 for descriptor */
    gpu->mat_images     = calloc(actual_tex, sizeof(VkImage));
    gpu->mat_image_mems = calloc(actual_tex, sizeof(VkDeviceMemory));
    gpu->mat_image_views = calloc(actual_tex, sizeof(VkImageView));

    for (int i = 0; i < ntextures; i++) {
        if (textures[i].pixels && textures[i].width > 0 && textures[i].height > 0) {
            create_texture_image(gpu, &textures[i],
                &gpu->mat_images[i], &gpu->mat_image_mems[i],
                &gpu->mat_image_views[i]);
        }
    }

    /* Create a 1x1 white dummy texture for unused slots */
    if (ntextures == 0) {
        unsigned char white[] = {255, 255, 255, 255};
        GpuTextureData dummy = { white, 1, 1 };
        create_texture_image(gpu, &dummy,
            &gpu->mat_images[0], &gpu->mat_image_mems[0],
            &gpu->mat_image_views[0]);
        actual_tex = 1;
    }

    fprintf(stderr, "gpu_vulkan: UPLOAD-SPLIT staging(alloc+memcpy)=%.1f ms | "
            "image_alloc=%.1f ms | submit+wait+mipgen=%.1f ms (%d textures)\n",
            g_up_staging_ms, g_up_image_ms, g_up_submit_ms, ntextures);
    gpu_upload_staging_pool_free(gpu);  /* release the reusable upload buffer */

    /* Create descriptor set layout */
    {
        VkDescriptorSetLayoutBinding bindings[8] = {0};

        /* Binding 0: Material SSBO */
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        /* Binding 1: Texture array */
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = (uint32_t)actual_tex;
        bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        /* Binding 2: Environment map (IBL specular) */
        bindings[2].binding         = 2;
        bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        /* Binding 3: BRDF integration LUT (IBL) */
        bindings[3].binding         = 3;
        bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        /* Binding 4: SH irradiance map (IBL diffuse) */
        bindings[4].binding         = 4;
        bindings[4].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        /* Binding 5: TLAS for inline ray-queried shadows in textured raster.
         * Optional — bound when scene has built an acceleration structure.
         * mesh.frag gates ray queries on pc.ibl_params.w to avoid invalid
         * dereference when only the procedural-light path is active. */
        bindings[5].binding         = 5;
        bindings[5].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        /* Binding 6: authored scene lights for textured raster. The RT
         * pipelines use the same SSBO layout at binding 13. */
        bindings[6].binding         = 6;
        bindings[6].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[6].descriptorCount = 1;
        bindings[6].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        /* Binding 7: packed per-triangle-corner Ptex colors, shared with the
         * RT path. Raster uses the per-draw offset carried in pc.color.w. */
        bindings[7].binding         = 7;
        bindings[7].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[7].descriptorCount = 1;
        bindings[7].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo dslci = {0};
        dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 8;
        dslci.pBindings    = bindings;
        vkCreateDescriptorSetLayout(gpu->device, &dslci, NULL, &gpu->mat_desc_layout);
    }

    /* Create pipeline layout */
    {
        VkPushConstantRange pc_range = {0};
        pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc_range.offset     = 0;
        pc_range.size       = PUSH_CONSTANT_SIZE;

        VkPipelineLayoutCreateInfo plci = {0};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &gpu->mat_desc_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pc_range;
        vkCreatePipelineLayout(gpu->device, &plci, NULL, &gpu->mat_pipeline_layout);
    }

    /* Create descriptor pool (+ slots for IBL env/BRDF LUT/irradiance, plus
     * one TLAS for inline ray-queried shadows on the textured-raster
     * pipeline). */
    {
        VkDescriptorPoolSize pool_sizes[3] = {0};
        pool_sizes[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[0].descriptorCount = 3;
        pool_sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        pool_sizes[1].descriptorCount = (uint32_t)actual_tex + 3; /* +3: env, brdf LUT, irradiance */
        pool_sizes[2].type            = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        pool_sizes[2].descriptorCount = 1;

        VkDescriptorPoolCreateInfo dpci = {0};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1;
        dpci.poolSizeCount = 3;
        dpci.pPoolSizes    = pool_sizes;
        vkCreateDescriptorPool(gpu->device, &dpci, NULL, &gpu->mat_desc_pool);
    }

    /* Allocate descriptor set */
    {
        VkDescriptorSetAllocateInfo dsai = {0};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = gpu->mat_desc_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &gpu->mat_desc_layout;
        vkAllocateDescriptorSets(gpu->device, &dsai, &gpu->mat_desc_set);
    }

    /* Update descriptor set */
    {
        /* SSBO */
        VkDescriptorBufferInfo ssbo_info = {0};
        ssbo_info.buffer = gpu->mat_ssbo_buf;
        ssbo_info.range  = ssbo_size;

        /* Textures */
        VkDescriptorImageInfo* img_infos = calloc(actual_tex,
            sizeof(VkDescriptorImageInfo));
        for (int i = 0; i < actual_tex; i++) {
            img_infos[i].sampler     = gpu->mat_sampler;
            img_infos[i].imageView   = gpu->mat_image_views[i]
                ? gpu->mat_image_views[i] : gpu->mat_image_views[0];
            img_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        /* Placeholder for IBL bindings (use first material texture as dummy) */
        VkDescriptorImageInfo ibl_placeholder = {0};
        ibl_placeholder.sampler     = gpu->mat_sampler;
        ibl_placeholder.imageView   = gpu->mat_image_views[0];
        ibl_placeholder.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        if (!gpu->light_ssbo_buf) {
            unsigned char zero_light[16 + sizeof(GpuLight)] = {0};
            create_buffer_with_data(gpu, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    zero_light, sizeof(zero_light),
                                    &gpu->light_ssbo_buf, &gpu->light_ssbo_mem);
            gpu->light_count = 0;
        }

        VkDescriptorBufferInfo lights_info = {0};
        lights_info.buffer = gpu->light_ssbo_buf;
        lights_info.offset = 0;
        lights_info.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet writes[6] = {0};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = gpu->mat_desc_set;
        writes[0].dstBinding      = 0;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo     = &ssbo_info;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = gpu->mat_desc_set;
        writes[1].dstBinding      = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = (uint32_t)actual_tex;
        writes[1].pImageInfo      = img_infos;

        /* Binding 2: env map (placeholder until IBL loaded) */
        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = gpu->mat_desc_set;
        writes[2].dstBinding      = 2;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo      = &ibl_placeholder;

        /* Binding 3: BRDF LUT (placeholder until IBL loaded) */
        writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet          = gpu->mat_desc_set;
        writes[3].dstBinding      = 3;
        writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].descriptorCount = 1;
        writes[3].pImageInfo      = &ibl_placeholder;

        /* Binding 6: authored scene lights. gpu_upload_lights() refreshes
         * this descriptor when the real scene light buffer is uploaded. */
        writes[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet          = gpu->mat_desc_set;
        writes[4].dstBinding      = 6;
        writes[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].descriptorCount = 1;
        writes[4].pBufferInfo     = &lights_info;

        /* Binding 7: Ptex triangle colors. Use the material SSBO as a valid
         * placeholder until gpu_upload_rt_triangle_colors installs the real
         * packed color buffer. Meshes without Ptex carry the 0xFFFFFFFF offset
         * sentinel, so the shader never reads this placeholder. */
        writes[5].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet          = gpu->mat_desc_set;
        writes[5].dstBinding      = 7;
        writes[5].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].descriptorCount = 1;
        writes[5].pBufferInfo     = &ssbo_info;

        vkUpdateDescriptorSets(gpu->device, 6, writes, 0, NULL);
        free(img_infos);
    }

    gpu->mat_uploaded = 1;
    update_material_ptex_descriptor(gpu);

    /* If the TLAS already exists (gpu_upload_materials called after
     * gpu_build_accel — uncommon path), wire it into binding 5 now.
     * Common path: TLAS-build site does this when shadow_desc_set is
     * written. */
    if (gpu->tlas) {
        VkWriteDescriptorSetAccelerationStructureKHR as_write = {0};
        as_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        as_write.accelerationStructureCount = 1;
        as_write.pAccelerationStructures    = &gpu->tlas;
        VkWriteDescriptorSet mat_write = {0};
        mat_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        mat_write.pNext           = &as_write;
        mat_write.dstSet          = gpu->mat_desc_set;
        mat_write.dstBinding      = 5;
        mat_write.descriptorCount = 1;
        mat_write.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        vkUpdateDescriptorSets(gpu->device, 1, &mat_write, 0, NULL);
    }

    fprintf(stderr, "gpu_vulkan: material descriptors ready (%d textures)\n",
            ntextures);
    return 1;
}

/* Phase 11.A: upload per-segment data (segments + colors) for
 * BasisCurves. Caller passes flat arrays of length seg_count.
 *
 * Three device-local buffers are created:
 *   - curve_seg_ssbo_buf   : SceneCurveSegment[seg_count]    (SSBO)
 *   - curve_color_ssbo_buf : RGBA8[seg_count]                (SSBO)
 *   - curve_aabb_buf       : VkAabbPositionsKHR[seg_count]   (AS build
 *                                                             + STORAGE,
 *                                                             written by
 *                                                             compute pass)
 *
 * Returns 1 on success. seg_count == 0 frees previous uploads only.
 *
 * Phase 12.x: the AABB buffer is allocated empty — the data is filled
 * by gpu_build_curve_aabbs_compute(), dispatched in the same command
 * buffer as the AS build. Saves the ~825 MB host→device upload that
 * 24 B × 34 M segs needed on the tera fixture.
 *
 * Note: this just lays down the data; gpu_build_curve_blas (next step)
 * is what actually constructs the acceleration structure. Splitting
 * upload from build keeps the BLAS rebuild path cheap when only
 * non-topology data (e.g. colors) changes. */

/* Phase 11.5.3: free all curve resources (BLAS + 3 SSBOs/buffers).
 * Idempotent. Safe on a never-uploaded gpu. Called from both
 * gpu_destroy_rt_scene (scene reload path) and gpu_shutdown
 * (process exit path) — neither freed curves prior to this. */
static void curve_destroy(Gpu* gpu)
{
    if (!gpu) return;
    if (gpu->curve_blas) {
        vkDestroyAccelerationStructureKHR(gpu->device, gpu->curve_blas, NULL);
        gpu->curve_blas = VK_NULL_HANDLE;
    }
    if (gpu->curve_blas_buf) {
        vkDestroyBuffer(gpu->device, gpu->curve_blas_buf, NULL);
        vkFreeMemory(gpu->device, gpu->curve_blas_mem, NULL);
        gpu->curve_blas_buf = VK_NULL_HANDLE;
        gpu->curve_blas_mem = VK_NULL_HANDLE;
    }
    if (gpu->curve_seg_ssbo_buf) {
        vkDestroyBuffer(gpu->device, gpu->curve_seg_ssbo_buf, NULL);
        vkFreeMemory(gpu->device, gpu->curve_seg_ssbo_mem, NULL);
        gpu->curve_seg_ssbo_buf = VK_NULL_HANDLE;
        gpu->curve_seg_ssbo_mem = VK_NULL_HANDLE;
    }
    if (gpu->curve_color_ssbo_buf) {
        vkDestroyBuffer(gpu->device, gpu->curve_color_ssbo_buf, NULL);
        vkFreeMemory(gpu->device, gpu->curve_color_ssbo_mem, NULL);
        gpu->curve_color_ssbo_buf = VK_NULL_HANDLE;
        gpu->curve_color_ssbo_mem = VK_NULL_HANDLE;
    }
    if (gpu->curve_aabb_buf) {
        vkDestroyBuffer(gpu->device, gpu->curve_aabb_buf, NULL);
        vkFreeMemory(gpu->device, gpu->curve_aabb_mem, NULL);
        gpu->curve_aabb_buf = VK_NULL_HANDLE;
        gpu->curve_aabb_mem = VK_NULL_HANDLE;
    }
    /* Phase 12.x: tear down the AABB-gen compute pipeline alongside
     * the curve data so future scene reloads start from a clean slate.
     * The pipeline is rebuilt lazily on next BLAS build. */
    if (gpu->curve_aabb_gen_pipeline) {
        vkDestroyPipeline(gpu->device, gpu->curve_aabb_gen_pipeline, NULL);
        gpu->curve_aabb_gen_pipeline = VK_NULL_HANDLE;
    }
    if (gpu->curve_aabb_gen_pl_layout) {
        vkDestroyPipelineLayout(gpu->device, gpu->curve_aabb_gen_pl_layout, NULL);
        gpu->curve_aabb_gen_pl_layout = VK_NULL_HANDLE;
    }
    if (gpu->curve_aabb_gen_ds_pool) {
        vkDestroyDescriptorPool(gpu->device, gpu->curve_aabb_gen_ds_pool, NULL);
        gpu->curve_aabb_gen_ds_pool = VK_NULL_HANDLE;
        gpu->curve_aabb_gen_ds_set  = VK_NULL_HANDLE;
    }
    if (gpu->curve_aabb_gen_ds_layout) {
        vkDestroyDescriptorSetLayout(gpu->device, gpu->curve_aabb_gen_ds_layout, NULL);
        gpu->curve_aabb_gen_ds_layout = VK_NULL_HANDLE;
    }
    if (gpu->curve_aabb_gen_ts_pool) {
        vkDestroyQueryPool(gpu->device, gpu->curve_aabb_gen_ts_pool, NULL);
        gpu->curve_aabb_gen_ts_pool = VK_NULL_HANDLE;
    }
    gpu->curve_aabb_gen_built = 0;
    gpu->curve_seg_count = 0;
}

int gpu_upload_rt_triangle_colors(Gpu* gpu,
                                  const uint32_t* colors,
                                  uint32_t count)
{
    if (!gpu) return 0;

    rt_triangle_colors_destroy(gpu);
    if (!colors || count == 0) return 1;

    VkDeviceSize bytes = (VkDeviceSize)count * sizeof(uint32_t);
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    void* staging_mapped = NULL;
    if (!rt_create_staging_buffer(gpu, bytes, &staging, &staging_mem,
                                   &staging_mapped)) {
        fprintf(stderr, "gpu_vulkan: Ptex triangle color staging alloc failed\n");
        return 0;
    }
    memcpy(staging_mapped, colors, (size_t)bytes);

    if (!rt_create_buffer_pooled(gpu, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_NONE,
            &gpu->rt_tri_color_ssbo_buf,
            &gpu->rt_tri_color_ssbo_mem)) {
        fprintf(stderr, "gpu_vulkan: Ptex triangle color SSBO alloc failed (%zu B)\n",
                (size_t)bytes);
        rt_destroy_staging_buffer(gpu, staging, staging_mem, staging_mapped);
        return 0;
    }

    VkCommandBuffer cmd = rt_begin_cmd(gpu);
    VkBufferCopy copy = {0};
    copy.size = bytes;
    vkCmdCopyBuffer(cmd, staging, gpu->rt_tri_color_ssbo_buf, 1, &copy);
    rt_end_cmd(gpu, cmd);

    rt_destroy_staging_buffer(gpu, staging, staging_mem, staging_mapped);
    gpu->rt_tri_color_count = count;
    update_material_ptex_descriptor(gpu);
    fprintf(stderr,
            "gpu_vulkan: uploaded %u real Ptex triangle-corner colors "
            "(%u triangles, %.1f MB)\n",
            count, count / 3u, (double)bytes / (1024.0 * 1024.0));
    return 1;
}

int gpu_upload_curve_data(Gpu* gpu,
                          const void* segments, /* SceneCurveSegment[] */
                          const float* colors,  /* float[seg_count*3]  */
                          int seg_count)
{
    if (!gpu) return 0;

    /* Free any previous upload. */
    if (gpu->curve_seg_ssbo_buf) {
        vkDestroyBuffer(gpu->device, gpu->curve_seg_ssbo_buf, NULL);
        vkFreeMemory(gpu->device, gpu->curve_seg_ssbo_mem, NULL);
        gpu->curve_seg_ssbo_buf = VK_NULL_HANDLE;
        gpu->curve_seg_ssbo_mem = VK_NULL_HANDLE;
    }
    if (gpu->curve_color_ssbo_buf) {
        vkDestroyBuffer(gpu->device, gpu->curve_color_ssbo_buf, NULL);
        vkFreeMemory(gpu->device, gpu->curve_color_ssbo_mem, NULL);
        gpu->curve_color_ssbo_buf = VK_NULL_HANDLE;
        gpu->curve_color_ssbo_mem = VK_NULL_HANDLE;
    }
    if (gpu->curve_aabb_buf) {
        vkDestroyBuffer(gpu->device, gpu->curve_aabb_buf, NULL);
        vkFreeMemory(gpu->device, gpu->curve_aabb_mem, NULL);
        gpu->curve_aabb_buf = VK_NULL_HANDLE;
        gpu->curve_aabb_mem = VK_NULL_HANDLE;
    }
    gpu->curve_seg_count = 0;

    if (seg_count <= 0) return 1;
    if (!segments || !colors) return 0;

    /* perf/blas-fast (staging-reuse): the previous implementation called
     * create_buffer_with_data_phase twice — once each for segments and
     * colors. Each call allocated + freed its own ~1 GB-class staging
     * memory. On RTX 5090 / PCIe Gen5 with ReBAR=246 MB (too small for
     * the segment buffer to bypass staging), each pass pays ~325 ms
     * vkAllocateMemory + ~221 ms vkFreeMemory on top of the actual
     * DMA — that's ~1100 ms of host-allocator overhead for the two
     * uploads. Sharing a single staging buffer across both passes
     * eliminates one alloc + one free (~650 ms + ~440 ms = ~1090 ms).
     *
     * The AABB buffer (allocated below this block) is GPU-derived by
     * curve_aabb_gen_dispatch and never sees the staging path, so it's
     * not part of this reuse loop.
     *
     * Phase tagging is preserved: each vkCmdCopyBuffer is bracketed
     * with gpu_phase_begin/end + gpu_phase_resolve so nsys and
     * nu_get_phase_timings_ms continue to attribute the time to
     * staging_upload_segs / staging_upload_colors as before.
     */

    /* Per-segment data SSBO: 32 B/segment. */
    VkDeviceSize seg_bytes = (VkDeviceSize)seg_count * 32;  /* SceneCurveSegment */
    VkDeviceSize color_bytes = (VkDeviceSize)seg_count * sizeof(uint32_t);

    /* ---- 1. Pre-pack the std430 colour buffer (pure host work). ----
     *
     * Per-segment color SSBO: pack to RGBA8 (4 B/segment, 4× smaller than
     * the previous vec4 std430 layout). On a 34 M-segment "tera" scene
     * this drops the color buffer from ~525 MB to ~131 MB. The shader
     * unpacks via unpackUnorm4x8(); see raytrace_curve.rchit.glsl.
     *
     * Byte order: byte 0 (LSB) = R, byte 1 = G, byte 2 = B, byte 3 = A.
     * This matches GLSL's unpackUnorm4x8 which puts byte 0 in .x.
     * Quantization uses round-to-nearest (×255 + 0.5) so values like
     * 0.2 land on 51 instead of 50. */
    uint32_t* color_packed = (uint32_t*)malloc((size_t)seg_count * sizeof(uint32_t));
    if (!color_packed) return 0;
    for (int i = 0; i < seg_count; i++) {
        float r = colors[i*3 + 0];
        float g = colors[i*3 + 1];
        float b = colors[i*3 + 2];
        if (r < 0.0f) r = 0.0f; else if (r > 1.0f) r = 1.0f;
        if (g < 0.0f) g = 0.0f; else if (g > 1.0f) g = 1.0f;
        if (b < 0.0f) b = 0.0f; else if (b > 1.0f) b = 1.0f;
        uint32_t ri = (uint32_t)(r * 255.0f + 0.5f);
        uint32_t gi = (uint32_t)(g * 255.0f + 0.5f);
        uint32_t bi = (uint32_t)(b * 255.0f + 0.5f);
        color_packed[i] = ri | (gi << 8) | (bi << 16) | (0xFFu << 24);
    }

    /* ---- 2. Allocate the two device-local destination buffers. ----
     *
     * perf/mem-pool: route through the resident pool, untracked variant
     * (the v2 path used direct vkAllocateMemory specifically to keep
     * these out of gpu_memory_used; we preserve that). */
    {
        /* Segments dst */
        if (!rt_create_buffer_pooled_untracked(gpu, seg_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                RT_POOL_RESIDENT,
                &gpu->curve_seg_ssbo_buf, &gpu->curve_seg_ssbo_mem)) {
            free(color_packed); return 0;
        }
    }
    {
        /* Colors dst */
        if (!rt_create_buffer_pooled_untracked(gpu, color_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                RT_POOL_RESIDENT,
                &gpu->curve_color_ssbo_buf, &gpu->curve_color_ssbo_mem)) {
            vkDestroyBuffer(gpu->device, gpu->curve_seg_ssbo_buf, NULL);
            if (gpu->curve_seg_ssbo_mem) vkFreeMemory(gpu->device, gpu->curve_seg_ssbo_mem, NULL);
            gpu->curve_seg_ssbo_buf = VK_NULL_HANDLE;
            gpu->curve_seg_ssbo_mem = VK_NULL_HANDLE;
            free(color_packed); return 0;
        }
    }

    /* ---- 3. One reusable staging buffer of max(seg, color) bytes. ----
     *
     * perf/mem-pool: route through the staging pool (HOST_VISIBLE,
     * persistent-mapped). The pool's bump cursor gets reset at the
     * end of build_accel via rt_pool_reset_transient(), so a 1 GB
     * staging buffer here doesn't permanently hold device memory. */
    VkDeviceSize stage_bytes = (seg_bytes > color_bytes) ? seg_bytes : color_bytes;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    void* staging_mapped = NULL;
    if (!rt_create_staging_buffer(gpu, stage_bytes, &staging, &staging_mem,
                                   &staging_mapped)) {
        vkDestroyBuffer(gpu->device, gpu->curve_seg_ssbo_buf, NULL);
        vkDestroyBuffer(gpu->device, gpu->curve_color_ssbo_buf, NULL);
        if (gpu->curve_seg_ssbo_mem)   vkFreeMemory(gpu->device, gpu->curve_seg_ssbo_mem, NULL);
        if (gpu->curve_color_ssbo_mem) vkFreeMemory(gpu->device, gpu->curve_color_ssbo_mem, NULL);
        gpu->curve_seg_ssbo_buf = VK_NULL_HANDLE;
        gpu->curve_color_ssbo_buf = VK_NULL_HANDLE;
        gpu->curve_seg_ssbo_mem = VK_NULL_HANDLE;
        gpu->curve_color_ssbo_mem = VK_NULL_HANDLE;
        free(color_packed); return 0;
    }

    /* ---- 4. Two sequential staging-mediated copies via persistent map. ----
     *
     * Each pass:
     *   - memcpy host data into the persistently-mapped staging memory
     *   - record vkCmdCopyBuffer into a one-shot command buffer
     *     (bracketed with gpu_phase_begin/end so nsys + the timing API
     *      attribute the GPU time to staging_upload_segs /
     *      staging_upload_colors as in the create_buffer_with_data_phase
     *      version this replaces)
     *   - submit + waitIdle (the wait makes the staging memory available
     *     for the next pass's memcpy and lets us resolve the timestamps)
     */
    struct {
        const void*    src;
        VkDeviceSize   bytes;
        VkBuffer       dst;
        GpuPhase       phase;
        const char*    label;
    } passes[2] = {
        { segments,     seg_bytes,   gpu->curve_seg_ssbo_buf,
          GPU_PHASE_STAGING_UPLOAD_SEGS,    "staging_upload_segs"   },
        { color_packed, color_bytes, gpu->curve_color_ssbo_buf,
          GPU_PHASE_STAGING_UPLOAD_COLORS,  "staging_upload_colors" },
    };

    for (int p = 0; p < 2; p++) {
        memcpy(staging_mapped, passes[p].src, (size_t)passes[p].bytes);

        VkCommandBuffer cmd = begin_single_command(gpu);
        gpu_phase_begin(gpu, cmd, passes[p].phase, passes[p].label);

        VkBufferCopy region = {0};
        region.size = passes[p].bytes;
        vkCmdCopyBuffer(cmd, staging, passes[p].dst, 1, &region);

        gpu_phase_end(gpu, cmd, passes[p].phase);
        end_single_command(gpu, cmd);
        if (gpu->timestamps_supported)
            gpu_phase_resolve(gpu, passes[p].phase);
    }

    /* ---- 5. Tear down the shared staging buffer. ----
     *
     * perf/mem-pool: when staging came from the staging pool the bytes
     * stay reserved until the next pool reset (end of build_accel) — we
     * just destroy the VkBuffer handle. */
    rt_destroy_staging_buffer(gpu, staging, staging_mem, staging_mapped);
    free(color_packed);

    /* AABB buffer: 24 B/AABB; allocated empty as a device-local SSBO +
     * AS-build input. The AABB-gen compute pass (dispatched in
     * gpu_build_curve_blas) writes the contents from the segment SSBO
     * before the AS build reads them.
     *
     * perf/blas-scratch: PRIVATE alloc (was RT_POOL_RESIDENT). The
     * AABB buffer is consumed exactly once (BLAS build); freeing it
     * after `vkCmdBuildAccelerationStructuresKHR` saves 24 B/seg
     * (~5.1 GB on zetta) of long-lived VRAM. The bump-allocated
     * resident pool can't actually release a buffer mid-scene — the
     * pool memory stays committed until the next pool reset (end of
     * scene). Going private means `vkFreeMemory` in
     * gpu_build_curve_blas truly returns the bytes to the driver. */
    VkDeviceSize aabb_bytes = (VkDeviceSize)seg_count * 24;  /* VkAabbPositionsKHR */
    if (!rt_create_buffer_pooled_untracked(gpu, aabb_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_NONE,  /* private alloc — see comment above */
            &gpu->curve_aabb_buf, &gpu->curve_aabb_mem)) {
        fprintf(stderr, "gpu_vulkan: curve AABB private alloc failed (%zu B)\n",
                (size_t)aabb_bytes);
        return 0;
    }

    gpu->curve_seg_count = seg_count;
    fprintf(stderr, "gpu_vulkan: uploaded %d curve segments (%zu B segs + %zu B colors); AABB buffer (%zu B) allocated for GPU compute\n",
            seg_count, (size_t)seg_bytes, (size_t)color_bytes, (size_t)aabb_bytes);
    return 1;
}

/* Phase 12.x: build (lazily) the AABB-gen compute pipeline + descriptor
 * set objects, then refresh the descriptor set to point at the current
 * segment SSBO and AABB buffer. Returns 1 on success.
 *
 * The pipeline objects are re-used across BLAS rebuilds; the descriptor
 * set is rewritten each call because the underlying buffer handles
 * change when seg_count changes. */
static int curve_aabb_gen_ensure_pipeline(Gpu* gpu)
{
    if (gpu->curve_aabb_gen_built) {
        /* Pipeline already exists — just refresh the descriptor set
         * (cheap; two writes). */
        VkDescriptorBufferInfo seg_info = {0};
        seg_info.buffer = gpu->curve_seg_ssbo_buf;
        seg_info.offset = 0;
        seg_info.range  = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo aabb_info = {0};
        aabb_info.buffer = gpu->curve_aabb_buf;
        aabb_info.offset = 0;
        aabb_info.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet ws[2] = {{0}, {0}};
        ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[0].dstSet = gpu->curve_aabb_gen_ds_set;
        ws[0].dstBinding = 0;
        ws[0].descriptorCount = 1;
        ws[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[0].pBufferInfo = &seg_info;
        ws[1] = ws[0];
        ws[1].dstBinding = 1;
        ws[1].pBufferInfo = &aabb_info;
        vkUpdateDescriptorSets(gpu->device, 2, ws, 0, NULL);
        return 1;
    }

    /* ---- Descriptor set layout: 2 SSBOs (seg in, aabb out) ---- */
    VkDescriptorSetLayoutBinding bindings[2] = {{0}, {0}};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1] = bindings[0];
    bindings[1].binding = 1;

    VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
    dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 2;
    dsl_ci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(gpu->device, &dsl_ci, NULL,
                                    &gpu->curve_aabb_gen_ds_layout) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: curve_aabb_gen DS layout failed\n");
        return 0;
    }

    /* ---- Descriptor pool + set ---- */
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pool_ci = {0};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(gpu->device, &pool_ci, NULL,
                               &gpu->curve_aabb_gen_ds_pool) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: curve_aabb_gen DS pool failed\n");
        return 0;
    }

    VkDescriptorSetAllocateInfo dsa = {0};
    dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = gpu->curve_aabb_gen_ds_pool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &gpu->curve_aabb_gen_ds_layout;
    if (vkAllocateDescriptorSets(gpu->device, &dsa,
                                 &gpu->curve_aabb_gen_ds_set) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: curve_aabb_gen DS alloc failed\n");
        return 0;
    }

    /* Initial descriptor writes — same logic as the cached path. */
    {
        VkDescriptorBufferInfo seg_info = {0};
        seg_info.buffer = gpu->curve_seg_ssbo_buf;
        seg_info.offset = 0;
        seg_info.range  = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo aabb_info = {0};
        aabb_info.buffer = gpu->curve_aabb_buf;
        aabb_info.offset = 0;
        aabb_info.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet ws[2] = {{0}, {0}};
        ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[0].dstSet = gpu->curve_aabb_gen_ds_set;
        ws[0].dstBinding = 0;
        ws[0].descriptorCount = 1;
        ws[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[0].pBufferInfo = &seg_info;
        ws[1] = ws[0];
        ws[1].dstBinding = 1;
        ws[1].pBufferInfo = &aabb_info;
        vkUpdateDescriptorSets(gpu->device, 2, ws, 0, NULL);
    }

    /* ---- Pipeline layout (DS + push constant: uint num_segments) ---- */
    VkPushConstantRange pc = {0};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pl_ci = {0};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = &gpu->curve_aabb_gen_ds_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(gpu->device, &pl_ci, NULL,
                               &gpu->curve_aabb_gen_pl_layout) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: curve_aabb_gen pipeline layout failed\n");
        return 0;
    }

    /* ---- Load SPV + create compute pipeline ---- */
    char shader_path[512];
    snprintf(shader_path, sizeof(shader_path), "%s/curve_aabb_gen.comp.spv", SHADER_DIR);
    FILE* f = fopen(shader_path, "rb");
    if (!f) {
        fprintf(stderr, "gpu_vulkan: can't open %s\n", shader_path);
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* code = (uint32_t*)malloc((size_t)sz);
    if (!code) { fclose(f); return 0; }
    if (fread(code, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(code);
        fprintf(stderr, "gpu_vulkan: short read on %s\n", shader_path);
        return 0;
    }
    fclose(f);

    VkShaderModule mod = create_shader_module(gpu->device, code, (uint32_t)sz);
    free(code);
    if (!mod) {
        fprintf(stderr, "gpu_vulkan: curve_aabb_gen create_shader_module failed\n");
        return 0;
    }

    VkComputePipelineCreateInfo cp = {0};
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = mod;
    cp.stage.pName = "main";
    cp.layout = gpu->curve_aabb_gen_pl_layout;
    VkResult res = vkCreateComputePipelines(gpu->device, VK_NULL_HANDLE,
                                             1, &cp, NULL,
                                             &gpu->curve_aabb_gen_pipeline);
    vkDestroyShaderModule(gpu->device, mod, NULL);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: curve_aabb_gen vkCreateComputePipelines failed (%d)\n", res);
        return 0;
    }

    gpu->curve_aabb_gen_built = 1;
    return 1;
}

/* Phase 12.x: dispatch the AABB-gen compute shader inside an existing
 * command buffer. After this call you must issue a memory barrier
 * (compute write → AS-build read) before the BLAS build. */
/* Workgroup size for the AABB-gen compute shader. MUST match
 * `local_size_x = NUSD_AABB_GEN_LOCAL_X` in
 * src/shaders/curve_aabb_gen.comp.glsl. Defined as a CMake-overridable
 * macro so we can A/B 64 vs 128 vs 256 without forking the shader. */
#ifndef NUSD_AABB_GEN_LOCAL_X
#define NUSD_AABB_GEN_LOCAL_X 64u
#endif

static void curve_aabb_gen_dispatch(Gpu* gpu, VkCommandBuffer cmd)
{
    uint32_t n = (uint32_t)gpu->curve_seg_count;

    /* Lazily create a 2-slot timestamp pool the first time we dispatch.
     * Used to capture the GPU-side AABB-gen duration (ms) for tuning
     * the workgroup size. The period is cached so we can convert ticks
     * to ms cheaply on each readback. */
    if (gpu->curve_aabb_gen_ts_pool == VK_NULL_HANDLE) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpu->physical_device, &props);
        gpu->curve_aabb_gen_ts_period = (double)props.limits.timestampPeriod;

        VkQueryPoolCreateInfo qpci = {0};
        qpci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        if (vkCreateQueryPool(gpu->device, &qpci, NULL,
                              &gpu->curve_aabb_gen_ts_pool) != VK_SUCCESS) {
            gpu->curve_aabb_gen_ts_pool = VK_NULL_HANDLE;
        }
    }

    if (gpu->curve_aabb_gen_ts_pool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, gpu->curve_aabb_gen_ts_pool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            gpu->curve_aabb_gen_ts_pool, 0);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      gpu->curve_aabb_gen_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            gpu->curve_aabb_gen_pl_layout,
                            0, 1, &gpu->curve_aabb_gen_ds_set, 0, NULL);
    vkCmdPushConstants(cmd, gpu->curve_aabb_gen_pl_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(uint32_t), &n);
    /* Workgroup size matches local_size_x = NUSD_AABB_GEN_LOCAL_X in
     * curve_aabb_gen.comp.glsl. Ceiling division so the tail thread
     * can early-out via the bounds check. */
    uint32_t groups = (n + NUSD_AABB_GEN_LOCAL_X - 1u) / NUSD_AABB_GEN_LOCAL_X;
    vkCmdDispatch(cmd, groups, 1, 1);

    if (gpu->curve_aabb_gen_ts_pool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            gpu->curve_aabb_gen_ts_pool, 1);
    }
}

/* Read back the timestamps captured around the most recent
 * curve_aabb_gen_dispatch + queue submit. Caches the result in
 * gpu->curve_aabb_gen_last_ms and prints it via stderr for the bench
 * scraper. Caller must have already waited for the queue to be idle. */
static void curve_aabb_gen_read_ts(Gpu* gpu)
{
    if (gpu->curve_aabb_gen_ts_pool == VK_NULL_HANDLE) return;
    uint64_t ts[2] = {0, 0};
    VkResult r = vkGetQueryPoolResults(gpu->device,
                                       gpu->curve_aabb_gen_ts_pool,
                                       0, 2,
                                       sizeof(ts), ts, sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT
                                       | VK_QUERY_RESULT_WAIT_BIT);
    if (r != VK_SUCCESS) return;
    uint64_t ticks = (ts[1] >= ts[0]) ? (ts[1] - ts[0]) : 0u;
    double ns = (double)ticks * gpu->curve_aabb_gen_ts_period;
    gpu->curve_aabb_gen_last_ms = ns / 1.0e6;
    fprintf(stderr,
            "gpu_vulkan: aabb_gen_dispatch_ms = %.3f (local_size_x=%u, %d segs)\n",
            gpu->curve_aabb_gen_last_ms,
            (unsigned)NUSD_AABB_GEN_LOCAL_X,
            gpu->curve_seg_count);
}

/* Phase 11.A.2.3: build a VK_GEOMETRY_TYPE_AABBS_KHR BLAS over the
 * curve segment AABBs uploaded by gpu_upload_curve_data. Single
 * acceleration structure containing all segments — Strategy A from
 * BIG_PLAN Phase 12.1 (best traversal quality, simplest plumbing).
 *
 * Caller must have already uploaded data via gpu_upload_curve_data.
 * No-op if there are no curve segments. Returns 1 on success.
 *
 * Phase 12.x: the per-segment AABBs are GENERATED on-GPU by a compute
 * dispatch (curve_aabb_gen.comp.spv) issued in the same command buffer
 * as the AS build, with a memory barrier between them. The host never
 * uploads AABB data — saves ~825 MB host→device transfer on tera
 * fixtures (~80 ms on PCIe gen 5).
 *
 * The BLAS is independent of the mesh BLAS pool (gpu->blas_list) — this
 * keeps the existing 1284-LOC gpu_build_rt_scene path unchanged. The
 * curve BLAS gets wired into the TLAS as an additional instance in a
 * subsequent step. */
int gpu_build_curve_blas(Gpu* gpu)
{
    if (!gpu) return 0;
    if (gpu->curve_seg_count <= 0 || !gpu->curve_aabb_buf) return 1;  /* nothing to build */

    /* Free any previous BLAS build. */
    if (gpu->curve_blas) {
        vkDestroyAccelerationStructureKHR(gpu->device, gpu->curve_blas, NULL);
        gpu->curve_blas = VK_NULL_HANDLE;
    }
    if (gpu->curve_blas_buf) {
        vkDestroyBuffer(gpu->device, gpu->curve_blas_buf, NULL);
        vkFreeMemory(gpu->device, gpu->curve_blas_mem, NULL);
        gpu->curve_blas_buf = VK_NULL_HANDLE;
        gpu->curve_blas_mem = VK_NULL_HANDLE;
    }

    /* Phase 12.x: ensure the AABB-gen compute pipeline is ready and
     * the descriptor set points at the current buffers. Note: this
     * descriptor refresh is required when seg_count changes (the
     * underlying VkBuffer handles are recreated in
     * gpu_upload_curve_data). */
    if (!curve_aabb_gen_ensure_pipeline(gpu)) {
        fprintf(stderr, "gpu_vulkan: curve_aabb_gen pipeline init failed\n");
        return 0;
    }

    VkDeviceAddress aabb_addr = rt_buf_addr(gpu, gpu->curve_aabb_buf);

    VkAccelerationStructureGeometryAabbsDataKHR aabbs = {0};
    aabbs.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbs.data.deviceAddress = aabb_addr;
    aabbs.stride = 24;  /* sizeof(VkAabbPositionsKHR) */

    VkAccelerationStructureGeometryKHR geom = {0};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geom.flags        = 0;  /* not VK_GEOMETRY_OPAQUE_BIT_KHR — intersection
                             * shader is mandatory for AABB geometry */
    geom.geometry.aabbs = aabbs;

    /* perf/blas-scratch: env-controlled BLAS build flags for the curve
     * BLAS. Default is FAST_TRACE (matches main). Set
     * NUSD_CURVE_BLAS_FLAGS=fast_build to switch to FAST_BUILD (~50%
     * less scratch on AABB BLAS), or =fast_build_lowmem to stack
     * VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR. */
    /* Default: FAST_TRACE + LOW_MEMORY. LOW_MEMORY cuts the AS size
     * ~50% (37 → 19 B/seg on RTX 5090) for +0.08 ms steady on tera —
     * net 613 MB saved at tera, ~4 GB at zetta. */
    VkBuildAccelerationStructureFlagsKHR blas_flags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
        | VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;
    const char* flags_env = getenv("NUSD_CURVE_BLAS_FLAGS");
    const char* flags_label = "fast_trace_lowmem";
    if (flags_env) {
        if (strcmp(flags_env, "fast_trace") == 0) {
            /* Pure FAST_TRACE — opts out of LOW_MEMORY's ~10 % steady-state
             * cost. Trades ~613 MB of VRAM at tera (37 vs 19 B/seg AS) for
             * the throughput. Documented as the perf-priority alternative
             * to the LOW_MEMORY default in 451ddfa's commit message; the
             * parser was missing this case before. */
            blas_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            flags_label = "fast_trace";
        } else if (strcmp(flags_env, "fast_build") == 0) {
            blas_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
            flags_label = "fast_build";
        } else if (strcmp(flags_env, "fast_build_lowmem") == 0) {
            blas_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR
                       | VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;
            flags_label = "fast_build_lowmem";
        } else if (strcmp(flags_env, "fast_trace_lowmem") == 0) {
            blas_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                       | VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;
            flags_label = "fast_trace_lowmem";
        }
    }
    fprintf(stderr, "gpu_vulkan: BLAS_FLAGS=%s (curve BLAS, %d segs)\n",
            flags_label, (int)gpu->curve_seg_count);

    VkAccelerationStructureBuildGeometryInfoKHR build_info = {0};
    build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags         = blas_flags;
    build_info.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries   = &geom;

    uint32_t prim_count = (uint32_t)gpu->curve_seg_count;
    VkAccelerationStructureBuildSizesInfoKHR sizes = {0};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(gpu->device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &prim_count, &sizes);
    /* perf/blas-scratch: log raw scratch + AS bytes so each variant
     * self-identifies in the bench output. */
    fprintf(stderr, "gpu_vulkan: curve BLAS sizes — AS=%.1f MB scratch=%.1f MB (%.1f B/seg AS, %.1f B/seg scratch)\n",
            (double)sizes.accelerationStructureSize / (1024.0*1024.0),
            (double)sizes.buildScratchSize / (1024.0*1024.0),
            (double)sizes.accelerationStructureSize / (double)prim_count,
            (double)sizes.buildScratchSize / (double)prim_count);

    /* perf/mem-pool: curve BLAS storage stays for the life of the
     * scene → resident pool. */
    if (!rt_create_buffer_pooled(gpu, sizes.accelerationStructureSize,
                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          RT_POOL_RESIDENT,
                          &gpu->curve_blas_buf, &gpu->curve_blas_mem)) {
        fprintf(stderr, "gpu_vulkan: curve BLAS storage alloc failed (%.1f KB)\n",
                (double)sizes.accelerationStructureSize / 1024.0);
        return 0;
    }

    VkAccelerationStructureCreateInfoKHR as_ci = {0};
    as_ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    as_ci.buffer = gpu->curve_blas_buf;
    as_ci.offset = 0;
    as_ci.size   = sizes.accelerationStructureSize;
    as_ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR(gpu->device, &as_ci, NULL, &gpu->curve_blas) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: vkCreateAccelerationStructureKHR (curve BLAS) failed\n");
        vkDestroyBuffer(gpu->device, gpu->curve_blas_buf, NULL);
        if (gpu->curve_blas_mem) vkFreeMemory(gpu->device, gpu->curve_blas_mem, NULL);
        gpu->curve_blas_buf = VK_NULL_HANDLE;
        gpu->curve_blas_mem = VK_NULL_HANDLE;
        return 0;
    }

    /* Scratch buffer (one-shot, freed after build).
     * perf/mem-pool: transient — bump-and-reset at end of build_accel. */
    VkBuffer scratch_buf = VK_NULL_HANDLE;
    VkDeviceMemory scratch_mem = VK_NULL_HANDLE;
    if (!rt_create_buffer_pooled(gpu, sizes.buildScratchSize,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          RT_POOL_TRANSIENT,
                          &scratch_buf, &scratch_mem)) {
        fprintf(stderr, "gpu_vulkan: curve BLAS scratch alloc failed\n");
        vkDestroyAccelerationStructureKHR(gpu->device, gpu->curve_blas, NULL);
        vkDestroyBuffer(gpu->device, gpu->curve_blas_buf, NULL);
        if (gpu->curve_blas_mem) vkFreeMemory(gpu->device, gpu->curve_blas_mem, NULL);
        gpu->curve_blas = VK_NULL_HANDLE;
        gpu->curve_blas_buf = VK_NULL_HANDLE;
        gpu->curve_blas_mem = VK_NULL_HANDLE;
        return 0;
    }

    build_info.dstAccelerationStructure  = gpu->curve_blas;
    build_info.scratchData.deviceAddress = rt_buf_addr(gpu, scratch_buf);

    VkAccelerationStructureBuildRangeInfoKHR range = {0};
    range.primitiveCount = prim_count;
    const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range;

    /* Phase 12.x: dispatch AABB-gen, barrier, then AS build — all in
     * the same command buffer so the AS build sees the compute
     * writes via the device-local memory barrier. */
    VkCommandBuffer cmd = rt_begin_cmd(gpu);
    /* perf/vk-instrumentation: bracket the curve BLAS build. */
    gpu_phase_begin(gpu, cmd, GPU_PHASE_CURVE_BLAS_BUILD, "curve_BLAS_build");

    curve_aabb_gen_dispatch(gpu, cmd);

    /* Compute → AS-build memory barrier. We use a buffer-memory
     * barrier rather than a global one so validation can scope the
     * sync to the AABB buffer; both buffer and global achieve
     * correctness here. */
    VkBufferMemoryBarrier mb = {0};
    mb.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb.buffer        = gpu->curve_aabb_buf;
    mb.offset        = 0;
    mb.size          = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 0, NULL, 1, &mb, 0, NULL);

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &p_range);
    gpu_phase_end(gpu, cmd, GPU_PHASE_CURVE_BLAS_BUILD);
    rt_end_cmd(gpu, cmd);
    if (gpu->timestamps_supported) gpu_phase_resolve(gpu, GPU_PHASE_CURVE_BLAS_BUILD);

    /* rt_end_cmd does vkQueueWaitIdle, so timestamp results are now
     * available. Read them out for the workgroup-tuning bench. */
    curve_aabb_gen_read_ts(gpu);

    vkDestroyBuffer(gpu->device, scratch_buf, NULL);
    vkFreeMemory(gpu->device, scratch_mem, NULL);

    /* perf/blas-scratch: free the AABB input buffer now that the BLAS
     * build has consumed it. The BLAS retains its own internal copy of
     * primitive bounds; the AABB buffer (24 B/seg, ~5 GB on zetta) is
     * never read again after this point. The previous code left the
     * AABB buffer alive for the life of the scene because it was bound
     * into RT_POOL_RESIDENT — that pool is never reset, so the bytes
     * leaked until close.
     *
     * rt_end_cmd above does vkQueueWaitIdle, so the build has fully
     * completed (and the device read the AABBs) before we destroy. */
    if (gpu->curve_aabb_buf) {
        vkDestroyBuffer(gpu->device, gpu->curve_aabb_buf, NULL);
        if (gpu->curve_aabb_mem)
            vkFreeMemory(gpu->device, gpu->curve_aabb_mem, NULL);
        gpu->curve_aabb_buf = VK_NULL_HANDLE;
        gpu->curve_aabb_mem = VK_NULL_HANDLE;
    }

    fprintf(stderr, "gpu_vulkan: built curve BLAS (%d AABBs via GPU compute, %.1f KB AS, %.1f KB scratch) [AABB buf freed]\n",
            (int)prim_count,
            (double)sizes.accelerationStructureSize / 1024.0,
            (double)sizes.buildScratchSize / 1024.0);
    return 1;
}

int gpu_upload_lights(Gpu* gpu, const GpuLight* lights, int nlights)
{
    if (!gpu) return 0;

    /* Free any previous upload */
    if (gpu->light_ssbo_buf) {
        vkDestroyBuffer(gpu->device, gpu->light_ssbo_buf, NULL);
        vkFreeMemory(gpu->device, gpu->light_ssbo_mem, NULL);
        gpu->light_ssbo_buf = VK_NULL_HANDLE;
        gpu->light_ssbo_mem = VK_NULL_HANDLE;
    }

    /* The shader-side layout starts with a 16-byte header (int nlights + pad)
     * followed by GpuLight[]. Always allocate at least one element so the
     * SSBO is non-empty even when the scene has no lights — the descriptor
     * binding requires a valid buffer. */
    int n_payload = (nlights > 0) ? nlights : 1;
    VkDeviceSize header_bytes = 16;
    VkDeviceSize body_bytes   = (VkDeviceSize)n_payload * sizeof(GpuLight);
    VkDeviceSize total_bytes  = header_bytes + body_bytes;

    void* upload = calloc(1, (size_t)total_bytes);
    if (!upload) return 0;
    int* hdr = (int*)upload;
    hdr[0] = nlights;
    if (nlights > 0) {
        memcpy((char*)upload + header_bytes, lights, (size_t)body_bytes);
    }

    create_buffer_with_data(gpu, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            upload, total_bytes,
                            &gpu->light_ssbo_buf, &gpu->light_ssbo_mem);
    free(upload);

    gpu->light_count = nlights;
    fprintf(stderr, "gpu_vulkan: uploaded light SSBO (%d lights, %zu bytes)\n",
            nlights, (size_t)total_bytes);

    if (gpu->mat_desc_set) {
        VkDescriptorBufferInfo lights_info = {0};
        lights_info.buffer = gpu->light_ssbo_buf;
        lights_info.offset = 0;
        lights_info.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet write = {0};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = gpu->mat_desc_set;
        write.dstBinding      = 6;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo     = &lights_info;
        vkUpdateDescriptorSets(gpu->device, 1, &write, 0, NULL);
    }

    /* The deferred-shading compute pipeline reads the same lights at binding 13
     * (deferred_shade.comp). This destroy+recreate gives light_ssbo_buf a new
     * handle; if the deferred descriptor sets were already written (built before
     * this upload), their binding 13 now dangles at the freed buffer and the
     * compute reads a stale/zero nlights — i.e. re-uploading lights after the
     * deferred pipeline exists (scene reload, animated lights) silently drops the
     * deferred SphereLight/RectLight/DistantLight contribution. Rebind binding 13
     * on both sets, mirroring the binding-6 refresh above. (When the deferred set
     * is built *after* the upload — the common first-render path — it already
     * picks up the current buffer via gpu_descriptor_writes_for_deferred, so this
     * is the re-upload safety net.) Load-time only — sets are not in flight. */
    VkDescriptorSet defsets[2] = { gpu->deferred_desc_set, gpu->deferred_desc_set_b };
    for (int i = 0; i < 2; i++) {
        if (!defsets[i]) continue;
        VkDescriptorBufferInfo li = {0};
        li.buffer = gpu->light_ssbo_buf;
        li.offset = 0;
        li.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w = {0};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = defsets[i];
        w.dstBinding      = 13;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo     = &li;
        vkUpdateDescriptorSets(gpu->device, 1, &w, 0, NULL);
    }
    return 1;
}

GpuPipeline gpu_create_material_pipeline(Gpu* gpu, const GpuPipelineDesc* desc)
{
    if (!gpu || !gpu->mat_uploaded) return NULL;

    GpuPipeline pipe = calloc(1, sizeof(struct GpuPipeline_s));
    if (!pipe) return NULL;

    VkShaderModule vert_mod = create_shader_module(gpu->device,
                                                   desc->vert_spv,
                                                   desc->vert_size);
    VkShaderModule frag_mod = create_shader_module(gpu->device,
                                                   desc->frag_spv,
                                                   desc->frag_size);

    if (!vert_mod || !frag_mod) {
        fprintf(stderr, "gpu_vulkan: failed to create material shader modules\n");
        if (vert_mod) vkDestroyShaderModule(gpu->device, vert_mod, NULL);
        if (frag_mod) vkDestroyShaderModule(gpu->device, frag_mod, NULL);
        free(pipe);
        return NULL;
    }

    VkPipelineShaderStageCreateInfo stages[2] = {0};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_mod;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_mod;
    stages[1].pName  = "main";

    VkVertexInputBindingDescription binding = {0};
    binding.binding   = 0;
    binding.stride    = desc->vertex_stride;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription* attrs = NULL;
    if (desc->nattribs > 0) {
        attrs = malloc(desc->nattribs * sizeof(VkVertexInputAttributeDescription));
        for (uint32_t i = 0; i < desc->nattribs; i++) {
            attrs[i].location = desc->attribs[i].location;
            attrs[i].binding  = 0;
            attrs[i].offset   = desc->attribs[i].offset;
            switch (desc->attribs[i].format) {
                case GPU_FORMAT_FLOAT3: attrs[i].format = VK_FORMAT_R32G32B32_SFLOAT; break;
                case GPU_FORMAT_FLOAT2: attrs[i].format = VK_FORMAT_R32G32_SFLOAT;    break;
                case GPU_FORMAT_UINT:   attrs[i].format = VK_FORMAT_R32_UINT;         break;
                case GPU_FORMAT_FLOAT:  attrs[i].format = VK_FORMAT_R32_SFLOAT;      break;
                default:                attrs[i].format = VK_FORMAT_R32G32B32_SFLOAT;  break;
            }
        }
    }

    VkPipelineVertexInputStateCreateInfo vi = {0};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &binding;
    vi.vertexAttributeDescriptionCount = desc->nattribs;
    vi.pVertexAttributeDescriptions    = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia = {0};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vps = {0};
    vps.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo raster = {0};
    raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode    = VK_CULL_MODE_NONE;
    raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms = {0};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds = {0};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    /* Alpha blending */
    VkPipelineColorBlendAttachmentState blend_att = {0};
    blend_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blend_att.blendEnable         = VK_TRUE;
    blend_att.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend_att.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend_att.colorBlendOp        = VK_BLEND_OP_ADD;
    blend_att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend_att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blend_att.alphaBlendOp        = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo blend = {0};
    blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments    = &blend_att;

    VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic = {0};
    dynamic.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates    = dyn_states;

    VkGraphicsPipelineCreateInfo pci = {0};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState      = &vps;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState   = &ms;
    pci.pDepthStencilState  = &ds;
    pci.pColorBlendState    = &blend;
    pci.pDynamicState       = &dynamic;
    pci.layout              = gpu->mat_pipeline_layout;
    pci.renderPass          = gpu->render_pass;
    pci.subpass             = 0;

    if (vkCreateGraphicsPipelines(gpu->device, gpu->pipeline_cache, 1, &pci, NULL,
                                  &pipe->pipeline) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create material pipeline\n");
        free(pipe);
        pipe = NULL;
    } else {
        pipe->layout = gpu->mat_pipeline_layout;
    }

    free(attrs);
    vkDestroyShaderModule(gpu->device, vert_mod, NULL);
    vkDestroyShaderModule(gpu->device, frag_mod, NULL);
    return pipe;
}

void gpu_cmd_bind_materials(Gpu* gpu)
{
    if (!gpu || !gpu->mat_uploaded) return;
    vkCmdBindDescriptorSets(gpu->current_cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        gpu->mat_pipeline_layout,
        0, 1, &gpu->mat_desc_set, 0, NULL);
    gpu->current_layout = gpu->mat_pipeline_layout;
}

/* ---- Environment background pipeline ---- */

extern int env_bg_compile_shaders(uint32_t** vert_spv, uint32_t* vert_size,
                                   uint32_t** frag_spv, uint32_t* frag_size);

int gpu_create_env_bg_pipeline(Gpu* gpu)
{
    if (!gpu || !gpu->mat_pipeline_layout) return 0;

    uint32_t *vert_spv = NULL, *frag_spv = NULL;
    uint32_t  vert_size = 0, frag_size = 0;
    if (!env_bg_compile_shaders(&vert_spv, &vert_size, &frag_spv, &frag_size))
        return 0;

    VkShaderModule vert_mod = VK_NULL_HANDLE, frag_mod = VK_NULL_HANDLE;
    {
        VkShaderModuleCreateInfo ci = {0};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = vert_size; ci.pCode = vert_spv;
        vkCreateShaderModule(gpu->device, &ci, NULL, &vert_mod);
        ci.codeSize = frag_size; ci.pCode = frag_spv;
        vkCreateShaderModule(gpu->device, &ci, NULL, &frag_mod);
    }
    free(vert_spv); free(frag_spv);
    if (!vert_mod || !frag_mod) { return 0; }

    VkPipelineShaderStageCreateInfo stages[2] = {0};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_mod;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_mod;
    stages[1].pName  = "main";

    /* No vertex input — fullscreen triangle from gl_VertexIndex */
    VkPipelineVertexInputStateCreateInfo vi = {0};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia = {0};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vps = {0};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;  vps.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster = {0};
    raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode    = VK_CULL_MODE_NONE;
    raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms = {0};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    /* Draw at the far plane after scene geometry: pass only cleared pixels. */
    VkPipelineDepthStencilStateCreateInfo ds = {0};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blend_att = {0};
    blend_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend = {0};
    blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments    = &blend_att;

    VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic = {0};
    dynamic.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates    = dyn_states;

    VkGraphicsPipelineCreateInfo pci = {0};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState      = &vps;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState   = &ms;
    pci.pDepthStencilState  = &ds;
    pci.pColorBlendState    = &blend;
    pci.pDynamicState       = &dynamic;
    pci.layout              = gpu->mat_pipeline_layout;
    pci.renderPass          = gpu->render_pass;
    pci.subpass             = 0;

    VkResult res = vkCreateGraphicsPipelines(gpu->device, gpu->pipeline_cache, 1, &pci, NULL,
                                              &gpu->env_bg_pipeline);
    vkDestroyShaderModule(gpu->device, vert_mod, NULL);
    vkDestroyShaderModule(gpu->device, frag_mod, NULL);

    if (res != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create env bg pipeline\n");
        return 0;
    }
    fprintf(stderr, "gpu_vulkan: env background pipeline created\n");
    return 1;
}

void gpu_draw_env_background(Gpu* gpu, const float* view_inv, const float* proj_inv)
{
    if (!gpu || !gpu->env_bg_pipeline || !gpu->ibl_loaded || !gpu->mat_uploaded) return;
    if (getenv("NUSD_DISABLE_ENV_BG")) return;

    vkCmdBindPipeline(gpu->current_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      gpu->env_bg_pipeline);
    gpu->current_layout = gpu->mat_pipeline_layout;
    vkCmdBindDescriptorSets(gpu->current_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            gpu->mat_pipeline_layout, 0, 1, &gpu->mat_desc_set, 0, NULL);

    VkViewport viewport = {0};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = (float)gpu->swapchain_extent.width;
    viewport.height   = (float)gpu->swapchain_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(gpu->current_cmd, 0, 1, &viewport);

    VkRect2D scissor = {0};
    scissor.extent = gpu->swapchain_extent;
    vkCmdSetScissor(gpu->current_cmd, 0, 1, &scissor);

    /* Pass view_inv in mvp slot (offset 0), proj_inv in model slot (offset 64). */
    float pc[48] = {0};
    memcpy(pc + 0,  view_inv, 16 * sizeof(float));  /* mvp = view_inv */
    memcpy(pc + 16, proj_inv, 16 * sizeof(float));   /* model = proj_inv */
    pc[40] = gpu->ibl_loaded ? 1.0f : 0.0f;
    pc[41] = (float)gpu->env_mip_levels;
    pc[42] = gpu->env_intensity;
    pc[44] = gpu->tone_exposure_scale;
    pc[45] = gpu->tone_sky_scale;
    pc[46] = gpu->tone_white_point;
    /* Up axis in tone_flags bits 24-25 so the env_bg equirect lookup matches
     * the RT rmiss / mesh.frag up-axis convention (sky parity on Y-up scenes). */
    uint32_t tone_flags = gpu->tone_flags |
        (((uint32_t)gpu->scene_up_axis & 3u) << 24);
    memcpy(pc + 47, &tone_flags, sizeof(tone_flags));
    vkCmdPushConstants(gpu->current_cmd, gpu->mat_pipeline_layout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, PUSH_CONSTANT_SIZE, pc);

    vkCmdDraw(gpu->current_cmd, 3, 1, 0, 0);
}

void gpu_destroy_materials(Gpu* gpu)
{
    if (!gpu || !gpu->mat_uploaded) return;

    vkDeviceWaitIdle(gpu->device);

    if (gpu->mat_desc_pool) {
        vkDestroyDescriptorPool(gpu->device, gpu->mat_desc_pool, NULL);
        gpu->mat_desc_pool = VK_NULL_HANDLE;
        gpu->mat_desc_set = VK_NULL_HANDLE;
    }
    if (gpu->mat_desc_layout) {
        vkDestroyDescriptorSetLayout(gpu->device, gpu->mat_desc_layout, NULL);
        gpu->mat_desc_layout = VK_NULL_HANDLE;
    }
    if (gpu->mat_pipeline_layout) {
        vkDestroyPipelineLayout(gpu->device, gpu->mat_pipeline_layout, NULL);
        gpu->mat_pipeline_layout = VK_NULL_HANDLE;
    }

    int actual_tex = (gpu->mat_tex_count > 0) ? gpu->mat_tex_count : 1;
    for (int i = 0; i < actual_tex; i++) {
        if (gpu->mat_image_views && gpu->mat_image_views[i])
            vkDestroyImageView(gpu->device, gpu->mat_image_views[i], NULL);
        if (gpu->mat_images && gpu->mat_images[i])
            vkDestroyImage(gpu->device, gpu->mat_images[i], NULL);
        if (gpu->mat_image_mems && gpu->mat_image_mems[i])
            vkFreeMemory(gpu->device, gpu->mat_image_mems[i], NULL);
    }
    free(gpu->mat_images);
    free(gpu->mat_image_mems);
    free(gpu->mat_image_views);
    gpu->mat_images     = NULL;
    gpu->mat_image_mems = NULL;
    gpu->mat_image_views = NULL;

    if (gpu->mat_sampler) {
        vkDestroySampler(gpu->device, gpu->mat_sampler, NULL);
        gpu->mat_sampler = VK_NULL_HANDLE;
    }
    if (gpu->mat_ssbo_buf) {
        vkDestroyBuffer(gpu->device, gpu->mat_ssbo_buf, NULL);
        vkFreeMemory(gpu->device, gpu->mat_ssbo_mem, NULL);
        gpu->mat_ssbo_buf = VK_NULL_HANDLE;
    }

    gpu->mat_uploaded = 0;
}

/* ---- IBL: Environment map + BRDF LUT ---- */

static void generate_brdf_lut(float* out, int size)
{
    for (int y = 0; y < size; y++) {
        float roughness = (float)(y + 0.5f) / (float)size;
        float a = roughness * roughness;
        float a2 = a * a;

        for (int x = 0; x < size; x++) {
            float NdotV = (float)(x + 0.5f) / (float)size;
            if (NdotV < 0.001f) NdotV = 0.001f;

            float V[3] = { sqrtf(1.0f - NdotV * NdotV), 0.0f, NdotV };
            float scale = 0.0f, bias = 0.0f;
            const int N_SAMPLES = 256;

            for (int i = 0; i < N_SAMPLES; i++) {
                float xi1 = (float)i / (float)N_SAMPLES;
                unsigned int bits = (unsigned int)i;
                bits = (bits << 16u) | (bits >> 16u);
                bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
                bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
                bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
                bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
                float xi2 = (float)bits * 2.3283064365386963e-10f;

                float phi = 2.0f * 3.14159265f * xi1;
                float cosTheta = sqrtf((1.0f - xi2) / (1.0f + (a2 - 1.0f) * xi2));
                float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

                float H[3] = { cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta };
                float VdotH = V[0]*H[0] + V[1]*H[1] + V[2]*H[2];
                float L[3] = { 2.0f*VdotH*H[0] - V[0],
                                2.0f*VdotH*H[1] - V[1],
                                2.0f*VdotH*H[2] - V[2] };
                float NdotL = L[2];

                if (NdotL > 0.0f) {
                    if (VdotH < 0.0f) VdotH = 0.0f;
                    float k = a / 2.0f;
                    float G_V = NdotV / (NdotV * (1.0f - k) + k);
                    float G_L = NdotL / (NdotL * (1.0f - k) + k);
                    float G = G_V * G_L;
                    float G_Vis = (G * VdotH) / (H[2] * NdotV);
                    float Fc = powf(1.0f - VdotH, 5.0f);
                    scale += (1.0f - Fc) * G_Vis;
                    bias += Fc * G_Vis;
                }
            }

            scale /= (float)N_SAMPLES;
            bias /= (float)N_SAMPLES;
            int idx = (y * size + x) * 2;
            out[idx + 0] = scale;
            out[idx + 1] = bias;
        }
    }
}

/* ---- SH Irradiance: project env map onto order-3 spherical harmonics,
 *      then evaluate into a lat-long irradiance map. ---- */

static void sh_project_environment(const float* rgb_data, int w, int h,
                                   float sh_coeffs[9][3])
{
    /* Zero out coefficients */
    memset(sh_coeffs, 0, 9 * 3 * sizeof(float));

    const float PI = 3.14159265358979323846f;
    float weight_sum = 0.0f;

    for (int y = 0; y < h; y++) {
        float theta = PI * ((float)y + 0.5f) / (float)h;
        float sin_theta = sinf(theta);
        float cos_theta = cosf(theta);
        /* solid angle for this row */
        float da = (2.0f * PI / (float)w) * (PI / (float)h) * sin_theta;

        for (int x = 0; x < w; x++) {
            float phi = 2.0f * PI * ((float)x + 0.5f) / (float)w;

            /* Direction vector in the same dome convention as the shaders:
             * +Z is the lat-long north pole and +Y is zero-longitude forward.
             * Keep the longitude offset in the inverse mapping so SH
             * irradiance and GGX prefiltering agree with visible sky. */
            const float dome_offset = 340.0f / 360.0f;
            float phi_aligned = phi - (0.5f + dome_offset) * 2.0f * PI;
            float dx = sin_theta * sinf(phi_aligned);
            float dy = sin_theta * cosf(phi_aligned);
            float dz = cos_theta;

            int idx = (y * w + x) * 3;
            float r = rgb_data[idx + 0] * da;
            float g = rgb_data[idx + 1] * da;
            float b = rgb_data[idx + 2] * da;

            /* SH basis functions (real, orthonormal) */
            float Y[9];
            Y[0] = 0.282095f;                              /* Y_00 */
            Y[1] = 0.488603f * dy;                          /* Y_1,0 */
            Y[2] = 0.488603f * dz;                          /* Y_1,1 */
            Y[3] = 0.488603f * dx;                          /* Y_1,-1 */
            Y[4] = 1.092548f * dx * dy;                     /* Y_2,-1 */
            Y[5] = 1.092548f * dy * dz;                     /* Y_2,0... */
            Y[6] = 0.315392f * (3.0f*dz*dz - 1.0f);        /* Y_2,0 */
            Y[7] = 1.092548f * dx * dz;                     /* Y_2,1 */
            Y[8] = 0.546274f * (dx*dx - dy*dy);             /* Y_2,2 */
            /* Note: convention varies; the standard here matches MaterialX */

            for (int c = 0; c < 9; c++) {
                sh_coeffs[c][0] += r * Y[c];
                sh_coeffs[c][1] += g * Y[c];
                sh_coeffs[c][2] += b * Y[c];
            }
            weight_sum += da;
        }
    }
    (void)weight_sum;
}

static void sh_render_irradiance(const float sh_coeffs[9][3],
                                 float* rgba_out, int w, int h)
{
    /* Evaluate the cosine-convolved SH irradiance at each direction.
     * Ramamoorthi & Hanrahan (2001) convolution constants. */
    const float PI = 3.14159265358979323846f;
    const float c1 = 0.429043f, c2 = 0.511664f;
    const float c3 = 0.743125f, c4 = 0.886227f, c5 = 0.247708f;

    for (int y = 0; y < h; y++) {
        float theta = PI * ((float)y + 0.5f) / (float)h;
        float sin_theta = sinf(theta);
        float cos_theta = cosf(theta);

        for (int x = 0; x < w; x++) {
            float phi = 2.0f * PI * ((float)x + 0.5f) / (float)w;
            const float dome_offset = 340.0f / 360.0f;
            float phi_aligned = phi - (0.5f + dome_offset) * 2.0f * PI;
            float nx = sin_theta * sinf(phi_aligned);
            float ny = sin_theta * cosf(phi_aligned);
            float nz = cos_theta;

            int idx = (y * w + x) * 4;
            for (int ch = 0; ch < 3; ch++) {
                const float* L = NULL;
                float Lvals[9];
                for (int i = 0; i < 9; i++) Lvals[i] = sh_coeffs[i][ch];
                L = Lvals;

                float irr = c4 * L[0]
                          + 2.0f * c2 * (L[1]*ny + L[2]*nz + L[3]*nx)
                          + 2.0f * c1 * (L[4]*nx*ny + L[5]*ny*nz + L[7]*nx*nz)
                          + c3 * L[6] * (nz*nz) - c5 * L[6]
                          + c1 * L[8] * (nx*nx - ny*ny);

                if (irr < 0.0f) irr = 0.0f;
                rgba_out[idx + ch] = irr;
            }
            rgba_out[idx + 3] = 1.0f;
        }
    }
}

/* ---- GGX-prefiltered glossy mip chain (Karis 2014 split-sum) -----------
 *
 * Replaces the box-filter mip chain `vkCmdBlitImage` produces with a
 * properly GGX-pre-integrated specular environment. Each output mip targets
 * roughness = mip / (mipCount - 1); per-texel cost is `samples` GGX
 * importance-sampled lookups into the source HDR.
 *
 * Assumes the V = N = R simplification used by Karis-style real-time
 * environment prefiltering. The
 * Smith G term is folded into the runtime split-sum BRDF LUT, so the
 * prefilter weight is just `NdotL` (Karis's "no-Fresnel" form). */

/* Hammersley low-discrepancy sequence: pair of (u, v) in [0,1)². */
static void hammersley(uint32_t i, uint32_t N, float* u, float* v)
{
    uint32_t bits = i;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    *u = (float)i / (float)N;
    *v = (float)bits * 2.3283064365386963e-10f;  /* 1 / 2^32 */
}

/* Sample H from an isotropic GGX distribution given alpha and a target
 * normal N. Returns H in world space. Karis 2014 form. */
static void importance_sample_ggx(float u, float v, float alpha,
                                  const float N[3], float H_out[3])
{
    const float PI = 3.14159265358979323846f;
    float alpha2 = alpha * alpha;
    float phi = 2.0f * PI * u;
    float cos_theta = sqrtf((1.0f - v) / (1.0f + (alpha2 - 1.0f) * v));
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    /* H in tangent space (Z = N) */
    float Hx = sin_theta * cosf(phi);
    float Hy = sin_theta * sinf(phi);
    float Hz = cos_theta;

    /* Build orthonormal basis around N (Pixar branchless ONB). */
    float sgn = N[2] >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (sgn + N[2]);
    float b = N[0] * N[1] * a;
    float Tx = 1.0f + sgn * N[0] * N[0] * a;
    float Ty = sgn * b;
    float Tz = -sgn * N[0];
    float Bx = b;
    float By = sgn + N[1] * N[1] * a;
    float Bz = -N[1];

    H_out[0] = Tx * Hx + Bx * Hy + N[0] * Hz;
    H_out[1] = Ty * Hx + By * Hy + N[1] * Hz;
    H_out[2] = Tz * Hx + Bz * Hy + N[2] * Hz;
}

/* Bilinear lookup in a float32 RGB lat-long environment. */
static void env_sample_bilinear(const float* env_rgb, int env_w, int env_h,
                                const float dir[3], float out_rgb[3])
{
    const float PI = 3.14159265358979323846f;
    /* Direction → lat-long UV (matches dirToEquirect in raytrace.rchit.glsl).
     * OVRTX/Hydra DomeLight convention for these assets: +Z is the north
     * pole and +Y is zero-longitude forward. */
    const float dome_offset = 340.0f / 360.0f;
    float u = atan2f(dir[0], dir[1]) * (1.0f / (2.0f * PI)) + 0.5f + dome_offset;
    u = u - floorf(u);
    float v = 1.0f - (asinf(fmaxf(-1.0f, fminf(1.0f, dir[2]))) * (1.0f / PI) + 0.5f);
    /* Bilinear in pixel space, repeat U, clamp V. */
    float fx = u * (float)env_w - 0.5f;
    float fy = v * (float)env_h - 0.5f;
    if (fy < 0.0f) fy = 0.0f;
    if (fy > (float)(env_h - 1)) fy = (float)(env_h - 1);
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    float wx = fx - (float)x0;
    float wy = fy - (float)y0;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    /* Wrap U */
    x0 = ((x0 % env_w) + env_w) % env_w;
    x1 = ((x1 % env_w) + env_w) % env_w;
    if (y1 > env_h - 1) y1 = env_h - 1;
    const float* p00 = &env_rgb[(y0 * env_w + x0) * 3];
    const float* p10 = &env_rgb[(y0 * env_w + x1) * 3];
    const float* p01 = &env_rgb[(y1 * env_w + x0) * 3];
    const float* p11 = &env_rgb[(y1 * env_w + x1) * 3];
    for (int c = 0; c < 3; c++) {
        float a = p00[c] * (1 - wx) + p10[c] * wx;
        float b = p01[c] * (1 - wx) + p11[c] * wx;
        out_rgb[c] = a * (1 - wy) + b * wy;
    }
}

/* Convert lat-long UV → unit direction vector (inverse of dirToEquirect). */
static void latlong_uv_to_dir(float u, float v, float dir_out[3])
{
    const float PI = 3.14159265358979323846f;
    const float dome_offset = 340.0f / 360.0f;
    float phi = (u - 0.5f - dome_offset) * 2.0f * PI;
    float theta = (0.5f - v) * PI;
    dir_out[0] = sinf(phi) * cosf(theta);
    dir_out[1] = cosf(phi) * cosf(theta);
    dir_out[2] = sinf(theta);
}

/* Pre-filter a single mip level of the GGX-glossy env. mip_w/mip_h are the
 * output dimensions; src_rgb is the float32 RGB source HDR (env_w × env_h);
 * roughness selects the GGX kernel. Output is written to mip_rgba (RGBA32F).
 * sample_count: 256–1024 typical. */
static void ggx_prefilter_mip(const float* src_rgb, int env_w, int env_h,
                              float* mip_rgba, int mip_w, int mip_h,
                              float roughness, int sample_count)
{
    const float alpha = roughness * roughness;
    /* mip 0 = mirror reflection: copy source resampled, no kernel. Saves
     * ~mip0_size * sample_count ops; matters because mip 0 is the largest. */
    if (roughness < 1e-3f) {
        for (int y = 0; y < mip_h; y++) {
            float v = ((float)y + 0.5f) / (float)mip_h;
            for (int x = 0; x < mip_w; x++) {
                float u = ((float)x + 0.5f) / (float)mip_w;
                float dir[3];
                latlong_uv_to_dir(u, v, dir);
                float rgb[3];
                env_sample_bilinear(src_rgb, env_w, env_h, dir, rgb);
                int o = (y * mip_w + x) * 4;
                mip_rgba[o + 0] = rgb[0];
                mip_rgba[o + 1] = rgb[1];
                mip_rgba[o + 2] = rgb[2];
                mip_rgba[o + 3] = 1.0f;
            }
        }
        return;
    }

    #pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int y = 0; y < mip_h; y++) {
        for (int x = 0; x < mip_w; x++) {
            float u = ((float)x + 0.5f) / (float)mip_w;
            float v = ((float)y + 0.5f) / (float)mip_h;
            float N[3];
            latlong_uv_to_dir(u, v, N);
            /* Karis simplification: V = N = R. */
            const float* V = N;

            float acc[3] = {0, 0, 0};
            float total_weight = 0.0f;
            for (int i = 0; i < sample_count; i++) {
                float xi_u, xi_v;
                hammersley((uint32_t)i, (uint32_t)sample_count, &xi_u, &xi_v);
                float H[3];
                importance_sample_ggx(xi_u, xi_v, alpha, N, H);
                /* L = reflect(-V, H) = 2(V·H)H − V */
                float VdotH = V[0]*H[0] + V[1]*H[1] + V[2]*H[2];
                float L[3] = {
                    2.0f * VdotH * H[0] - V[0],
                    2.0f * VdotH * H[1] - V[1],
                    2.0f * VdotH * H[2] - V[2],
                };
                float NdotL = N[0]*L[0] + N[1]*L[1] + N[2]*L[2];
                if (NdotL > 0.0f) {
                    float Li[3];
                    env_sample_bilinear(src_rgb, env_w, env_h, L, Li);
                    /* Solid-angle Jacobian for the lat-long parameter-
                     * isation (matches ovrtx's `computeSolidAngleLatLong`
                     * idea — same algorithm, our own implementation).
                     * The lat-long stretches near the poles, so a uniform
                     * sample in (φ, θ) over-represents pixels near the
                     * top/bottom of the texture; multiplying the weight
                     * by sin(θ_L) corrects for the area density. Here the
                     * dome north pole is +Z, so sin(theta_L)=sqrt(1-L.z²). */
                    float sinThetaL = sqrtf(fmaxf(0.0f, 1.0f - L[2] * L[2]));
                    float w = NdotL * sinThetaL;
                    acc[0] += Li[0] * w;
                    acc[1] += Li[1] * w;
                    acc[2] += Li[2] * w;
                    total_weight += w;
                }
            }
            int o = (y * mip_w + x) * 4;
            if (total_weight > 0.0f) {
                mip_rgba[o + 0] = acc[0] / total_weight;
                mip_rgba[o + 1] = acc[1] / total_weight;
                mip_rgba[o + 2] = acc[2] / total_weight;
            } else {
                mip_rgba[o + 0] = mip_rgba[o + 1] = mip_rgba[o + 2] = 0.0f;
            }
            mip_rgba[o + 3] = 1.0f;
        }
    }
}

/* Float32 → IEEE 754 half-precision float16 conversion */
static uint16_t float_to_half(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0)  return (uint16_t)sign;               /* underflow → ±0 */
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);    /* overflow  → ±inf */
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* Helper: create a Vulkan image from float data (RGBA16F) with mip chain */
static void create_hdr_texture(Gpu* gpu, const float* rgba_data, int w, int h,
                                int gen_mips,
                                VkImage* out_image, VkDeviceMemory* out_mem,
                                VkImageView* out_view, int* out_mip_levels)
{
    int mips = 1;
    if (gen_mips) {
        int mw = w, mh = h;
        while (mw > 1 || mh > 1) { mw /= 2; mh /= 2; mips++; }
    }
    *out_mip_levels = mips;

    /* Convert float32 RGBA → float16 RGBA for R16G16B16A16_SFLOAT upload */
    size_t npixels = (size_t)w * h;
    VkDeviceSize size = (VkDeviceSize)npixels * 4 * sizeof(uint16_t);
    uint16_t* half_data = (uint16_t*)malloc(npixels * 4 * sizeof(uint16_t));
    if (!half_data) return;
    for (size_t i = 0; i < npixels * 4; i++)
        half_data[i] = float_to_half(rgba_data[i]);

    /* Staging buffer */
    VkBuffer staging;
    VkDeviceMemory staging_mem;
    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(gpu->device, &bci, NULL, &staging);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gpu->device, staging, &req);
    VkMemoryAllocateInfo mai = {0};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(gpu->device, &mai, NULL, &staging_mem);
    vkBindBufferMemory(gpu->device, staging, staging_mem, 0);

    void* mapped;
    vkMapMemory(gpu->device, staging_mem, 0, size, 0, &mapped);
    memcpy(mapped, half_data, (size_t)size);
    vkUnmapMemory(gpu->device, staging_mem);
    free(half_data);

    /* Image */
    VkImageCreateInfo ici = {0};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
    ici.extent.width  = (uint32_t)w;
    ici.extent.height = (uint32_t)h;
    ici.extent.depth  = 1;
    ici.mipLevels     = (uint32_t)mips;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                      | (gen_mips ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(gpu->device, &ici, NULL, out_image);

    vkGetImageMemoryRequirements(gpu->device, *out_image, &req);
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    gpu_alloc_memory(gpu, &mai, out_mem);
    vkBindImageMemory(gpu->device, *out_image, *out_mem, 0);

    VkCommandBuffer cmd = begin_single_command(gpu);

    /* Transition mip 0 to transfer dst */
    VkImageMemoryBarrier barrier = {0};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = *out_image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = (uint32_t)mips;
    barrier.subresourceRange.layerCount     = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier);

    /* Copy buffer to mip 0 */
    VkBufferImageCopy region = {0};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width  = (uint32_t)w;
    region.imageExtent.height = (uint32_t)h;
    region.imageExtent.depth  = 1;
    vkCmdCopyBufferToImage(cmd, staging, *out_image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    /* Generate mip chain via vkCmdBlitImage */
    if (gen_mips) {
        int mw = w, mh = h;
        for (int m = 1; m < mips; m++) {
            /* Transition previous level to transfer src */
            barrier.subresourceRange.baseMipLevel = (uint32_t)(m - 1);
            barrier.subresourceRange.levelCount   = 1;
            barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, NULL, 0, NULL, 1, &barrier);

            int next_w = mw > 1 ? mw / 2 : 1;
            int next_h = mh > 1 ? mh / 2 : 1;

            VkImageBlit blit = {0};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel   = (uint32_t)(m - 1);
            blit.srcSubresource.layerCount = 1;
            blit.srcOffsets[1].x = mw;
            blit.srcOffsets[1].y = mh;
            blit.srcOffsets[1].z = 1;
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel   = (uint32_t)m;
            blit.dstSubresource.layerCount = 1;
            blit.dstOffsets[1].x = next_w;
            blit.dstOffsets[1].y = next_h;
            blit.dstOffsets[1].z = 1;
            vkCmdBlitImage(cmd, *out_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           *out_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &blit, VK_FILTER_LINEAR);

            /* Transition src level to shader read */
            barrier.subresourceRange.baseMipLevel = (uint32_t)(m - 1);
            barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, NULL, 0, NULL, 1, &barrier);

            mw = next_w;
            mh = next_h;
        }
    }

    /* Transition last mip to shader read */
    barrier.subresourceRange.baseMipLevel = gen_mips ? (uint32_t)(mips - 1) : 0;
    barrier.subresourceRange.levelCount   = 1;
    barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier);

    end_single_command(gpu, cmd);

    vkDestroyBuffer(gpu->device, staging, NULL);
    vkFreeMemory(gpu->device, staging_mem, NULL);

    /* Image view */
    VkImageViewCreateInfo vci = {0};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = *out_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = VK_FORMAT_R16G16B16A16_SFLOAT;
    vci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount     = (uint32_t)mips;
    vci.subresourceRange.layerCount     = 1;
    vkCreateImageView(gpu->device, &vci, NULL, out_view);
}

/* Create a Vulkan image from a source HDR with a GGX-pre-filtered glossy
 * mip chain. Each mip is integrated against the GGX kernel for
 * roughness = mip / (mipCount - 1). Mirrors create_hdr_texture's upload
 * path but uploads each mip explicitly rather than blitting from mip 0. */
static void create_hdr_texture_ggx_prefiltered(
    Gpu* gpu, const float* src_rgb, int w, int h,
    VkImage* out_image, VkDeviceMemory* out_mem,
    VkImageView* out_view, int* out_mip_levels, int sample_count)
{
    int mips = 1;
    {
        int mw = w, mh = h;
        while (mw > 1 || mh > 1) { mw /= 2; mh /= 2; mips++; }
    }
    *out_mip_levels = mips;

    /* Total bytes for all mips packed (RGBA16F = 8 bytes/texel). */
    size_t total_bytes = 0;
    int mip_w[16], mip_h[16];
    size_t mip_offset[16];
    int mw = w, mh = h;
    for (int m = 0; m < mips; m++) {
        mip_w[m] = mw;
        mip_h[m] = mh;
        mip_offset[m] = total_bytes;
        total_bytes += (size_t)mw * (size_t)mh * 8;  /* RGBA16F */
        mw = mw > 1 ? mw / 2 : 1;
        mh = mh > 1 ? mh / 2 : 1;
    }

    uint16_t* staging_data = (uint16_t*)malloc(total_bytes);
    if (!staging_data) return;

    /* Per-mip GGX prefilter into a temporary float32 RGBA buffer, then
     * convert to half-float for the staging buffer. */
    float* tmp_rgba = NULL;
    size_t tmp_capacity = 0;
    fprintf(stderr, "gpu_vulkan: GGX-prefilter env mip chain (%dx%d, %d mips, %d samples/texel)...\n",
            w, h, mips, sample_count);
    double t0 = (double)clock() / (double)CLOCKS_PER_SEC;
    for (int m = 0; m < mips; m++) {
        size_t mip_pixels = (size_t)mip_w[m] * (size_t)mip_h[m];
        if (tmp_capacity < mip_pixels * 4) {
            free(tmp_rgba);
            tmp_rgba = (float*)malloc(mip_pixels * 4 * sizeof(float));
            tmp_capacity = mip_pixels * 4;
        }
        if (!tmp_rgba) { free(staging_data); return; }
        float roughness = (mips > 1) ? (float)m / (float)(mips - 1) : 0.0f;
        /* Smaller sample count for tiny mips — they have <samples output texels
         * anyway and high roughness GGX kernels are wide so fewer samples
         * still integrate well. */
        int adj_samples = sample_count;
        if (mip_w[m] <= 16) adj_samples = sample_count / 4;
        else if (mip_w[m] <= 64) adj_samples = sample_count / 2;
        ggx_prefilter_mip(src_rgb, w, h, tmp_rgba, mip_w[m], mip_h[m],
                          roughness, adj_samples);

        /* Convert to half-float into the packed staging buffer. */
        uint16_t* dst = staging_data + mip_offset[m] / 2;
        for (size_t i = 0; i < mip_pixels * 4; i++)
            dst[i] = float_to_half(tmp_rgba[i]);
    }
    free(tmp_rgba);
    double t1 = (double)clock() / (double)CLOCKS_PER_SEC;
    fprintf(stderr, "gpu_vulkan: GGX prefilter done in %.2f s\n", t1 - t0);

    /* Staging buffer (host-visible, source for image copy). */
    VkBuffer staging;
    VkDeviceMemory staging_mem;
    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = total_bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(gpu->device, &bci, NULL, &staging);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gpu->device, staging, &req);
    VkMemoryAllocateInfo mai = {0};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(gpu->device, &mai, NULL, &staging_mem);
    vkBindBufferMemory(gpu->device, staging, staging_mem, 0);

    void* mapped;
    vkMapMemory(gpu->device, staging_mem, 0, total_bytes, 0, &mapped);
    memcpy(mapped, staging_data, total_bytes);
    vkUnmapMemory(gpu->device, staging_mem);
    free(staging_data);

    /* Image. */
    VkImageCreateInfo ici = {0};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
    ici.extent.width  = (uint32_t)w;
    ici.extent.height = (uint32_t)h;
    ici.extent.depth  = 1;
    ici.mipLevels     = (uint32_t)mips;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(gpu->device, &ici, NULL, out_image);

    vkGetImageMemoryRequirements(gpu->device, *out_image, &req);
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    gpu_alloc_memory(gpu, &mai, out_mem);
    vkBindImageMemory(gpu->device, *out_image, *out_mem, 0);

    VkCommandBuffer cmd = begin_single_command(gpu);

    /* Transition all mips to TRANSFER_DST. */
    VkImageMemoryBarrier barrier = {0};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = *out_image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = (uint32_t)mips;
    barrier.subresourceRange.layerCount     = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier);

    /* Copy all mips in one call. */
    VkBufferImageCopy regions[16] = {0};
    for (int m = 0; m < mips; m++) {
        regions[m].bufferOffset = (VkDeviceSize)mip_offset[m];
        regions[m].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regions[m].imageSubresource.mipLevel   = (uint32_t)m;
        regions[m].imageSubresource.layerCount = 1;
        regions[m].imageExtent.width  = (uint32_t)mip_w[m];
        regions[m].imageExtent.height = (uint32_t)mip_h[m];
        regions[m].imageExtent.depth  = 1;
    }
    vkCmdCopyBufferToImage(cmd, staging, *out_image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)mips, regions);

    /* Transition all mips to SHADER_READ_ONLY. */
    barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier);

    end_single_command(gpu, cmd);

    vkDestroyBuffer(gpu->device, staging, NULL);
    vkFreeMemory(gpu->device, staging_mem, NULL);

    VkImageViewCreateInfo vci = {0};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = *out_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = VK_FORMAT_R16G16B16A16_SFLOAT;
    vci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount     = (uint32_t)mips;
    vci.subresourceRange.layerCount     = 1;
    vkCreateImageView(gpu->device, &vci, NULL, out_view);
}

/* Convenience: legacy auto-exposure path. Negative intensity selects the
 * SH-luminance-based auto-exposure tuning, positive intensity overrides
 * with a USD-authored scale (matches ovrtx's `dome.color = color * intensity`
 * convention). */
int gpu_load_environment(Gpu* gpu, const char* hdr_path)
{
    return gpu_load_environment_intensity(gpu, hdr_path, -1.0f);
}

static int env_max_width(void)
{
    const char* e = getenv("NUSD_ENV_MAX_WIDTH");
    if (!e || !e[0]) return 1024;
    char* end = NULL;
    long v = strtol(e, &end, 10);
    if (end == e || v < 256) return 1024;
    if (v > 8192) v = 8192;
    return (int)v;
}

static int env_prefilter_sample_count(int width, float intensity)
{
    const char* e = getenv("NUSD_ENV_PREFILTER_SAMPLES");
    if (e && e[0]) {
        char* end = NULL;
        long v = strtol(e, &end, 10);
        if (end != e) {
            if (v < 16) v = 16;
            if (v > 512) v = 512;
            return (int)v;
        }
    }

    /* The visible-dome fallback path feeds display-referred PNGs through the
     * same prefilter as HDR EXRs. They are low-frequency and large, so 64
     * samples avoids a multi-second startup tax without visible impact. */
    if (intensity < -1.0f)
        return 64;
    if (width >= 1024)
        return 128;
    return 256;
}

static float* downsample_env_rgb(const float* src, int sw, int sh,
                                 int dw, int dh)
{
    if (!src || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return NULL;
    float* dst = (float*)malloc((size_t)dw * (size_t)dh * 3u * sizeof(float));
    if (!dst) return NULL;

    const float sx = (float)sw / (float)dw;
    const float sy = (float)sh / (float)dh;
    for (int y = 0; y < dh; ++y) {
        float fy = ((float)y + 0.5f) * sy - 0.5f;
        int y0 = (int)floorf(fy);
        float ty = fy - (float)y0;
        if (y0 < 0) { y0 = 0; ty = 0.0f; }
        int y1 = y0 + 1;
        if (y1 >= sh) y1 = sh - 1;
        for (int x = 0; x < dw; ++x) {
            float fx = ((float)x + 0.5f) * sx - 0.5f;
            int x0 = (int)floorf(fx);
            float tx = fx - (float)x0;
            if (x0 < 0) { x0 = 0; tx = 0.0f; }
            int x1 = x0 + 1;
            if (x1 >= sw) x1 = sw - 1;

            const float* p00 = src + ((size_t)y0 * sw + x0) * 3u;
            const float* p10 = src + ((size_t)y0 * sw + x1) * 3u;
            const float* p01 = src + ((size_t)y1 * sw + x0) * 3u;
            const float* p11 = src + ((size_t)y1 * sw + x1) * 3u;
            float* out = dst + ((size_t)y * dw + x) * 3u;
            for (int c = 0; c < 3; ++c) {
                float a = p00[c] + (p10[c] - p00[c]) * tx;
                float b = p01[c] + (p11[c] - p01[c]) * tx;
                out[c] = a + (b - a) * ty;
            }
        }
    }
    return dst;
}

static int gpu_load_environment_rgb_intensity(Gpu* gpu,
                                              const float* rgb_data,
                                              int w, int h,
                                              float intensity,
                                              const float tint[3],
                                              const char* label)
{
    if (!gpu || !rgb_data || w <= 0 || h <= 0 || !gpu->mat_uploaded) return 0;

    float tint_rgb[3] = {1.0f, 1.0f, 1.0f};
    if (tint) {
        tint_rgb[0] = fmaxf(tint[0], 0.0f);
        tint_rgb[1] = fmaxf(tint[1], 0.0f);
        tint_rgb[2] = fmaxf(tint[2], 0.0f);
    }

    size_t npix = (size_t)w * (size_t)h;
    float* tinted_rgb = (float*)malloc(npix * 3 * sizeof(float));
    if (!tinted_rgb) return 0;
    for (size_t i = 0; i < npix; i++) {
        tinted_rgb[i*3 + 0] = rgb_data[i*3 + 0] * tint_rgb[0];
        tinted_rgb[i*3 + 1] = rgb_data[i*3 + 1] * tint_rgb[1];
        tinted_rgb[i*3 + 2] = rgb_data[i*3 + 2] * tint_rgb[2];
    }

    fprintf(stderr,
            "gpu_vulkan: loading %s environment %dx%d "
            "(tint=%.3f,%.3f,%.3f intensity=%.3f)\n",
            label ? label : "RGB", w, h,
            tint_rgb[0], tint_rgb[1], tint_rgb[2], intensity);

    /* ---- Project environment onto SH3 before converting to RGBA ---- */
    float sh_coeffs[9][3];
    sh_project_environment(tinted_rgb, w, h, sh_coeffs);

    /* ---- Generate SH irradiance map (256x128) ---- */
    const int irr_w = 256, irr_h = 128;
    float* irr_rgba = (float*)malloc((size_t)irr_w * irr_h * 4 * sizeof(float));
    if (irr_rgba) {
        sh_render_irradiance(sh_coeffs, irr_rgba, irr_w, irr_h);
    }

    /* env_scale always uses auto-exposure (matches today's behavior for
     * surface lighting / IBL irradiance — output luminance lands in our
     * fixed-exposure ACES range). USD-authored intensity is tracked
     * separately as gpu->env_intensity and applied to the sky lookup in
     * raytrace.rmiss.glsl, matching ovrtx's intensity=1000 → saturated
     * sky behaviour without over-brightening surfaces.
     *
     * Negative intensities below -1 are a viewer-internal marker used by
     * nanousdview's default HDR: abs(intensity) is the visible-sky
     * multiplier, and the sign tells the closest-hit shader to add the
     * same default direct key/fill that the ovrtx adapter authors for
     * otherwise unlit scenes. The public one-argument nu_load_environment
     * still passes -1 and keeps the legacy sky-intensity=1 behavior. */
    float avg_lum = 0.2126f * sh_coeffs[0][0] + 0.7152f * sh_coeffs[0][1]
                  + 0.0722f * sh_coeffs[0][2];
    float avg_irr = 0.886227f * avg_lum;  /* c4 * L[0] luminance */
    float env_scale = (avg_irr > 0.001f) ? 3.14159f / avg_irr : 1.0f;
    if (env_scale > 20.0f) env_scale = 20.0f;
    gpu->env_intensity = (intensity < -1.0f || intensity > 0.0f) ? intensity : 1.0f;
    fprintf(stderr, "gpu_vulkan: HDR auto-exposure: avg_irr=%.3f, scale=%.2f, "
            "sky-intensity=%.3f\n",
            avg_irr, env_scale, gpu->env_intensity);

    /* Scale irradiance map */
    if (irr_rgba) {
        for (int i = 0; i < irr_w * irr_h; i++) {
            irr_rgba[i*4+0] *= env_scale;
            irr_rgba[i*4+1] *= env_scale;
            irr_rgba[i*4+2] *= env_scale;
        }
    }

    /* Apply auto-exposure scale to the source HDR before pre-filtering so
     * the integrated radiance in each mip is correctly scaled — same
     * convention as the irradiance map above. */
    float* scaled_rgb = (float*)malloc(npix * 3 * sizeof(float));
    if (!scaled_rgb) {
        free(tinted_rgb);
        free(irr_rgba);
        return 0;
    }
    for (size_t i = 0; i < npix * 3; i++)
        scaled_rgb[i] = tinted_rgb[i] * env_scale;
    free(tinted_rgb);

    /* GGX-prefilter the env into a glossy mip chain. Each mip targets
     * roughness = mip / (mipCount - 1); shader samples
     * `prefilteredColor = textureLod(envMap, R, roughness * (mips-1))`,
     * following the standard split-sum IBL prefiltering model. Replaces
     * the box-filter mips that vkCmdBlitImage produced — those preserved
     * bright sun pixels at every mip and over-bright metallic surfaces. */
    int mips = 0;
    int prefilter_samples = env_prefilter_sample_count(w, intensity);
    create_hdr_texture_ggx_prefiltered(gpu, scaled_rgb, w, h,
                                       &gpu->env_image, &gpu->env_image_mem,
                                       &gpu->env_image_view, &mips,
                                       prefilter_samples);
    free(scaled_rgb);
    gpu->env_mip_levels = mips;

    /* Env map sampler (linear mip, repeat horizontally, clamp vertically) */
    {
        VkSamplerCreateInfo sci = {0};
        sci.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter     = VK_FILTER_LINEAR;
        sci.minFilter     = VK_FILTER_LINEAR;
        sci.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod        = (float)mips;
        vkCreateSampler(gpu->device, &sci, NULL, &gpu->env_sampler);
    }

    /* Upload irradiance map (no mips needed — already low-freq) */
    if (irr_rgba) {
        int irr_mips = 0;
        create_hdr_texture(gpu, irr_rgba, irr_w, irr_h, 0,
                           &gpu->irr_image, &gpu->irr_image_mem,
                           &gpu->irr_image_view, &irr_mips);
        free(irr_rgba);

        VkSamplerCreateInfo sci = {0};
        sci.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter     = VK_FILTER_LINEAR;
        sci.minFilter     = VK_FILTER_LINEAR;
        sci.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod        = 0.0f;
        vkCreateSampler(gpu->device, &sci, NULL, &gpu->irr_sampler);
        fprintf(stderr, "gpu_vulkan: SH irradiance map generated (%dx%d)\n", irr_w, irr_h);
    }

    /* Generate BRDF integration LUT (128x128, RG float → RGBA16F) */
    const int lut_size = 128;
    float* lut_rg = (float*)malloc((size_t)lut_size * lut_size * 2 * sizeof(float));
    if (!lut_rg) return 0;
    generate_brdf_lut(lut_rg, lut_size);

    /* Convert RG to RGBA for Vulkan (RGBA16F) */
    float* lut_rgba = (float*)malloc((size_t)lut_size * lut_size * 4 * sizeof(float));
    for (int i = 0; i < lut_size * lut_size; i++) {
        lut_rgba[i*4+0] = lut_rg[i*2+0];
        lut_rgba[i*4+1] = lut_rg[i*2+1];
        lut_rgba[i*4+2] = 0.0f;
        lut_rgba[i*4+3] = 1.0f;
    }
    free(lut_rg);

    int lut_mips = 0;
    create_hdr_texture(gpu, lut_rgba, lut_size, lut_size, 0,
                       &gpu->brdf_lut_image, &gpu->brdf_lut_mem,
                       &gpu->brdf_lut_view, &lut_mips);
    free(lut_rgba);

    /* BRDF LUT sampler (linear, clamp) */
    {
        VkSamplerCreateInfo sci = {0};
        sci.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter     = VK_FILTER_LINEAR;
        sci.minFilter     = VK_FILTER_LINEAR;
        sci.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod        = 0.0f;
        vkCreateSampler(gpu->device, &sci, NULL, &gpu->brdf_lut_sampler);
    }

    /* Update descriptor set bindings 2, 3, and 4 */
    VkDescriptorImageInfo env_info = {0};
    env_info.sampler     = gpu->env_sampler;
    env_info.imageView   = gpu->env_image_view;
    env_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo lut_info = {0};
    lut_info.sampler     = gpu->brdf_lut_sampler;
    lut_info.imageView   = gpu->brdf_lut_view;
    lut_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo irr_info = {0};
    irr_info.sampler     = gpu->irr_sampler ? gpu->irr_sampler : gpu->env_sampler;
    irr_info.imageView   = gpu->irr_image_view ? gpu->irr_image_view : gpu->env_image_view;
    irr_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet writes[3] = {0};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = gpu->mat_desc_set;
    writes[0].dstBinding      = 2;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo      = &env_info;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = gpu->mat_desc_set;
    writes[1].dstBinding      = 3;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo      = &lut_info;

    writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet          = gpu->mat_desc_set;
    writes[2].dstBinding      = 4;
    writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].descriptorCount = 1;
    writes[2].pImageInfo      = &irr_info;

    vkUpdateDescriptorSets(gpu->device, 3, writes, 0, NULL);

    gpu->ibl_loaded = 1;
    if (!gpu->env_bg_pipeline) {
        gpu_create_env_bg_pipeline(gpu);
    }
    fprintf(stderr, "gpu_vulkan: IBL loaded (%d mip levels, 128x128 BRDF LUT, SH irradiance)\n", mips);
    return 1;
}

int gpu_load_environment_tinted_intensity(Gpu* gpu, const char* hdr_path,
                                          float intensity,
                                          const float tint[3])
{
    if (!gpu || !hdr_path || !gpu->mat_uploaded) return 0;

    int w, h, channels;
    float* hdr_data = stbi_loadf(hdr_path, &w, &h, &channels, 3);
    if (!hdr_data) {
        fprintf(stderr, "gpu_vulkan: failed to load HDR: %s\n", hdr_path);
        return 0;
    }
    int hdr_data_owned_by_malloc = 0;

    int max_w = env_max_width();
    if (w > max_w) {
        int down_w = max_w;
        int down_h = (int)floorf((float)h * ((float)down_w / (float)w) + 0.5f);
        if (down_h < 1) down_h = 1;
        float* down = downsample_env_rgb(hdr_data, w, h, down_w, down_h);
        if (!down) {
            stbi_image_free(hdr_data);
            return 0;
        }
        fprintf(stderr, "gpu_vulkan: downsampled environment %dx%d -> %dx%d\n",
                w, h, down_w, down_h);
        stbi_image_free(hdr_data);
        hdr_data = down;
        w = down_w;
        h = down_h;
        hdr_data_owned_by_malloc = 1;
    }

    int ok = gpu_load_environment_rgb_intensity(gpu, hdr_data, w, h,
                                                intensity, tint, hdr_path);
    if (hdr_data_owned_by_malloc) free(hdr_data);
    else stbi_image_free(hdr_data);
    return ok;
}

int gpu_load_environment_intensity(Gpu* gpu, const char* hdr_path, float intensity)
{
    return gpu_load_environment_tinted_intensity(gpu, hdr_path, intensity, NULL);
}

int gpu_load_flat_environment(Gpu* gpu, const float color[3], float intensity)
{
    if (!gpu || !gpu->mat_uploaded) return 0;

    const int w = 16, h = 8;
    float rgb[16 * 8 * 3];
    float c[3] = {1.0f, 1.0f, 1.0f};
    if (color) {
        c[0] = fmaxf(color[0], 0.0f);
        c[1] = fmaxf(color[1], 0.0f);
        c[2] = fmaxf(color[2], 0.0f);
    }
    for (int i = 0; i < w * h; i++) {
        rgb[i*3 + 0] = c[0];
        rgb[i*3 + 1] = c[1];
        rgb[i*3 + 2] = c[2];
    }
    return gpu_load_environment_rgb_intensity(gpu, rgb, w, h,
                                              intensity, NULL,
                                              "flat DomeLight");
}

void gpu_destroy_environment(Gpu* gpu)
{
    if (!gpu) return;
    vkDeviceWaitIdle(gpu->device);

    if (gpu->env_image_view)    vkDestroyImageView(gpu->device, gpu->env_image_view, NULL);
    if (gpu->env_image)         vkDestroyImage(gpu->device, gpu->env_image, NULL);
    if (gpu->env_image_mem)     vkFreeMemory(gpu->device, gpu->env_image_mem, NULL);
    if (gpu->env_sampler)       vkDestroySampler(gpu->device, gpu->env_sampler, NULL);
    gpu->env_image_view = VK_NULL_HANDLE;
    gpu->env_image      = VK_NULL_HANDLE;
    gpu->env_image_mem  = VK_NULL_HANDLE;
    gpu->env_sampler    = VK_NULL_HANDLE;

    if (gpu->brdf_lut_view)     vkDestroyImageView(gpu->device, gpu->brdf_lut_view, NULL);
    if (gpu->brdf_lut_image)    vkDestroyImage(gpu->device, gpu->brdf_lut_image, NULL);
    if (gpu->brdf_lut_mem)      vkFreeMemory(gpu->device, gpu->brdf_lut_mem, NULL);
    if (gpu->brdf_lut_sampler)  vkDestroySampler(gpu->device, gpu->brdf_lut_sampler, NULL);
    gpu->brdf_lut_view    = VK_NULL_HANDLE;
    gpu->brdf_lut_image   = VK_NULL_HANDLE;
    gpu->brdf_lut_mem     = VK_NULL_HANDLE;
    gpu->brdf_lut_sampler = VK_NULL_HANDLE;

    if (gpu->irr_image_view)    vkDestroyImageView(gpu->device, gpu->irr_image_view, NULL);
    if (gpu->irr_image)         vkDestroyImage(gpu->device, gpu->irr_image, NULL);
    if (gpu->irr_image_mem)     vkFreeMemory(gpu->device, gpu->irr_image_mem, NULL);
    if (gpu->irr_sampler)       vkDestroySampler(gpu->device, gpu->irr_sampler, NULL);
    gpu->irr_image_view = VK_NULL_HANDLE;
    gpu->irr_image      = VK_NULL_HANDLE;
    gpu->irr_image_mem  = VK_NULL_HANDLE;
    gpu->irr_sampler    = VK_NULL_HANDLE;

    if (gpu->env_bg_pipeline) {
        vkDestroyPipeline(gpu->device, gpu->env_bg_pipeline, NULL);
        gpu->env_bg_pipeline = VK_NULL_HANDLE;
    }

    gpu->env_mip_levels = 0;
    gpu->env_intensity  = 1.0f;
    gpu->ibl_loaded     = 0;
}

int gpu_get_env_mip_levels(Gpu* gpu)
{
    return gpu ? gpu->env_mip_levels : 0;
}

int gpu_get_ibl_loaded(Gpu* gpu)
{
    return (gpu && gpu->env_image_view) ? 1 : 0;
}

float gpu_get_env_intensity(Gpu* gpu)
{
    return (gpu && gpu->env_image_view) ? gpu->env_intensity : 1.0f;
}

void gpu_set_tone_mapping(Gpu* gpu, float exposure_scale, float sky_scale,
                          float white_point_scale, uint32_t flags)
{
    if (!gpu) return;
    if (!(exposure_scale > 0.0f) || !isfinite(exposure_scale)) exposure_scale = 1.0f;
    if (!(sky_scale > 0.0f) || !isfinite(sky_scale)) sky_scale = exposure_scale;
    if (!(white_point_scale > 0.0f) || !isfinite(white_point_scale)) white_point_scale = 1.0f;
    gpu->tone_exposure_scale = exposure_scale;
    gpu->tone_sky_scale = sky_scale;
    gpu->tone_white_point = white_point_scale;
    gpu->tone_flags = flags;
}

int gpu_materials_uploaded(const Gpu* gpu)
{
    return gpu ? gpu->mat_uploaded : 0;
}

/* ==== Tiled Multi-Camera RT ==== */

static void tiled_destroy_image(Gpu* gpu)
{
    if (gpu->tiled_image_view) vkDestroyImageView(gpu->device, gpu->tiled_image_view, NULL);
    if (gpu->tiled_image)      vkDestroyImage(gpu->device, gpu->tiled_image, NULL);
    if (gpu->tiled_image_mem)  vkFreeMemory(gpu->device, gpu->tiled_image_mem, NULL);
    gpu->tiled_image_view = VK_NULL_HANDLE;
    gpu->tiled_image      = VK_NULL_HANDLE;
    gpu->tiled_image_mem  = VK_NULL_HANDLE;
    gpu->tiled_image_alloc_size = 0;
    gpu->tiled_image_w = 0;
    gpu->tiled_image_h = 0;
}

static void tiled_destroy_interop(Gpu* gpu)
{
    for (int i = 0; i < 2; i++) {
        if (gpu->interop_buf[i]) {
            vkDestroyBuffer(gpu->device, gpu->interop_buf[i], NULL);
            gpu->interop_buf[i] = VK_NULL_HANDLE;
        }
        if (gpu->interop_buf_mem[i]) {
            vkFreeMemory(gpu->device, gpu->interop_buf_mem[i], NULL);
            gpu->interop_buf_mem[i] = VK_NULL_HANDLE;
        }
    }
    gpu->interop_buf_size = 0;
    gpu->interop_write_idx = 0;
    gpu->external_buffers_active = 0;
    if (gpu->interop_timeline_sem)
        vkDestroySemaphore(gpu->device, gpu->interop_timeline_sem, NULL);
    gpu->interop_timeline_sem   = VK_NULL_HANDLE;
    gpu->interop_timeline_value = 0;
}

static int tiled_create_storage_image(Gpu* gpu, uint32_t w, uint32_t h)
{
    /* Validate against GPU limits before creating the image */
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu->physical_device, &props);
    uint32_t max_dim = props.limits.maxImageDimension2D;
    if (w > max_dim || h > max_dim) {
        fprintf(stderr, "gpu_vulkan: tiled image %ux%u exceeds maxImageDimension2D (%u)\n",
                w, h, max_dim);
        return 0;
    }

    /* The cached tiled cmd buffer references gpu->tiled_image directly via
     * barrier/copy commands; replacing the image invalidates the recording. */
    gpu_invalidate_tiled_cmd_cache(gpu);

    tiled_destroy_image(gpu);

    /* When CUDA interop is available, make the image memory exportable so
     * CUDA can import the same GPU allocation via cudaImportExternalMemory. */
    VkExternalMemoryImageCreateInfo ext_mem_ici = {0};
    ext_mem_ici.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext_mem_ici.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkImageCreateInfo ici = {0};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent.width  = w;
    ici.extent.height = h;
    ici.extent.depth  = 1;
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (gpu->interop_available)
        ici.pNext = &ext_mem_ici;

    if (vkCreateImage(gpu->device, &ici, NULL, &gpu->tiled_image) != VK_SUCCESS)
        return 0;

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(gpu->device, gpu->tiled_image, &req);

    /* For CUDA interop: chain export + dedicated allocation.
     * Many NVIDIA drivers require a dedicated allocation for external images. */
    VkMemoryDedicatedAllocateInfo dedicated_ai = {0};
    dedicated_ai.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated_ai.image = gpu->tiled_image;

    VkExportMemoryAllocateInfo export_mai = {0};
    export_mai.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_mai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    export_mai.pNext       = &dedicated_ai;

    VkMemoryAllocateInfo ai = {0};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (gpu->interop_available)
        ai.pNext = &export_mai;

    if (gpu_alloc_memory(gpu, &ai, &gpu->tiled_image_mem) != VK_SUCCESS) {
        vkDestroyImage(gpu->device, gpu->tiled_image, NULL);
        gpu->tiled_image = VK_NULL_HANDLE;
        return 0;
    }
    gpu->tiled_image_alloc_size = req.size;
    vkBindImageMemory(gpu->device, gpu->tiled_image, gpu->tiled_image_mem, 0);

    VkImageViewCreateInfo vci = {0};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = gpu->tiled_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.baseMipLevel   = 0;
    vci.subresourceRange.levelCount     = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount     = 1;
    vkCreateImageView(gpu->device, &vci, NULL, &gpu->tiled_image_view);

    /* Transition to GENERAL */
    VkCommandBuffer cmd = rt_begin_cmd(gpu);
    VkImageMemoryBarrier barrier = {0};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout                       = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                           = gpu->tiled_image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount     = 1;
    barrier.subresourceRange.layerCount     = 1;
    barrier.srcAccessMask                   = 0;
    barrier.dstAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, NULL, 0, NULL, 1, &barrier);
    rt_end_cmd(gpu, cmd);

    gpu->tiled_image_w = w;
    gpu->tiled_image_h = h;

    fprintf(stderr, "gpu_vulkan: tiled storage image created (%ux%u)\n", w, h);
    return 1;
}

static int tiled_create_interop_semaphore(Gpu* gpu)
{
    if (!gpu->interop_available) return 1;  /* not an error — just not available */
    if (gpu->interop_timeline_sem) return 1;  /* already created */

    VkSemaphoreTypeCreateInfo type_info = {0};
    type_info.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    type_info.initialValue  = 0;

    VkExportSemaphoreCreateInfo export_info = {0};
    export_info.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    export_info.pNext       = &type_info;

    VkSemaphoreCreateInfo sci = {0};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &export_info;

    if (vkCreateSemaphore(gpu->device, &sci, NULL, &gpu->interop_timeline_sem) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: failed to create interop timeline semaphore\n");
        gpu->interop_available = 0;
        return 0;
    }
    gpu->interop_timeline_value = 0;
    fprintf(stderr, "gpu_vulkan: interop timeline semaphore created\n");
    return 1;
}

/* Create two exportable device-local linear buffers matching the tiled image size.
 * Double-buffered: render writes to buf[write_idx], CUDA reads from buf[1-write_idx].
 * This lets CUDA read the previous frame while Vulkan renders the current frame. */
static int tiled_create_interop_buffer(Gpu* gpu)
{
    if (!gpu->interop_available) return 1;
    /* Hybrid C: caller owns the memory (cuMemCreate'd, imported via
     * gpu_set_external_output_buffers). Skip the export-side allocation. */
    if (gpu->external_buffers_active) return 1;

    VkDeviceSize buf_size = (VkDeviceSize)gpu->tiled_image_w * gpu->tiled_image_h * 4;
    if (gpu->interop_buf[0] && gpu->interop_buf_size == buf_size)
        return 1;  /* already the right size */

    /* Cached tiled cmd buffer references gpu->interop_buf[0/1] directly. */
    gpu_invalidate_tiled_cmd_cache(gpu);

    /* Destroy old */
    for (int i = 0; i < 2; i++) {
        if (gpu->interop_buf[i]) {
            vkDestroyBuffer(gpu->device, gpu->interop_buf[i], NULL);
            gpu->interop_buf[i] = VK_NULL_HANDLE;
        }
        if (gpu->interop_buf_mem[i]) {
            vkFreeMemory(gpu->device, gpu->interop_buf_mem[i], NULL);
            gpu->interop_buf_mem[i] = VK_NULL_HANDLE;
        }
    }

    for (int i = 0; i < 2; i++) {
        VkExternalMemoryBufferCreateInfo ext_buf_ci = {0};
        ext_buf_ci.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ext_buf_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.pNext = &ext_buf_ci;
        bci.size  = buf_size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->interop_buf[i]) != VK_SUCCESS)
            return 0;

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->interop_buf[i], &req);

        VkExportMemoryAllocateInfo export_mai = {0};
        export_mai.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        export_mai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkMemoryAllocateInfo ai = {0};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.pNext           = &export_mai;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (gpu_alloc_memory(gpu, &ai, &gpu->interop_buf_mem[i]) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, gpu->interop_buf[i], NULL);
            gpu->interop_buf[i] = VK_NULL_HANDLE;
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->interop_buf[i], gpu->interop_buf_mem[i], 0);

        if (i == 0)
            gpu->interop_buf_size = req.size;  /* both buffers have the same size */
    }
    gpu->interop_write_idx = 0;

    fprintf(stderr, "gpu_vulkan: interop double-buffer created (%ux%u, 2x %.1f MB alloc, %.1f MB logical each)\n",
            gpu->tiled_image_w, gpu->tiled_image_h,
            (double)gpu->interop_buf_size / (1024.0 * 1024.0),
            (double)buf_size / (1024.0 * 1024.0));
    return 1;
}

/* Phase B: allocate / resize the per-launch tiled G-buffer at binding 17 on
 * BOTH tiled descriptor sets (set + set_b). 32 bytes/pixel.
 *
 * The Phase A code path bound binding 17 to `scene_data_buf` as a stub. We
 * must rebind to a real allocation BEFORE flipping the rchit/rmiss
 * deferred_shade_enabled gate, otherwise rchit writes corrupt scene_data_buf.
 *
 * Returns 1 on success (real buffer bound), 0 on failure (stub still bound;
 * caller must NOT enable deferred shading). */
int gpu_tiled_ensure_gbuffer(Gpu* gpu)
{
    if (!gpu) return 0;
    if (gpu->tiled_image_w == 0 || gpu->tiled_image_h == 0) return 0;

    uint32_t w = gpu->tiled_image_w;
    uint32_t h = gpu->tiled_image_h;
    if (gpu->tiled_gbuffer_buf &&
        gpu->tiled_gbuffer_w == w && gpu->tiled_gbuffer_h == h)
        return 1;

    /* Cache replays reference the descriptor set bound to binding 17, so
     * any rebind requires invalidating the cached cmd buffer. */
    gpu_invalidate_tiled_cmd_cache(gpu);

    /* Tear down old. */
    if (gpu->tiled_gbuffer_buf) {
        vkDestroyBuffer(gpu->device, gpu->tiled_gbuffer_buf, NULL);
        vkFreeMemory(gpu->device, gpu->tiled_gbuffer_mem, NULL);
        gpu->tiled_gbuffer_buf = VK_NULL_HANDLE;
        gpu->tiled_gbuffer_mem = VK_NULL_HANDLE;
    }

    VkDeviceSize buf_size = (VkDeviceSize)w * h * 32;
    gpu->tiled_gbuffer_size = buf_size;
    gpu->tiled_gbuffer_w    = w;
    gpu->tiled_gbuffer_h    = h;

    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = buf_size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->tiled_gbuffer_buf) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tiled G-buffer create failed (%zu bytes)\n",
                (size_t)buf_size);
        gpu->tiled_gbuffer_buf  = VK_NULL_HANDLE;
        gpu->tiled_gbuffer_w    = 0;
        gpu->tiled_gbuffer_h    = 0;
        gpu->tiled_gbuffer_size = 0;
        return 0;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gpu->device, gpu->tiled_gbuffer_buf, &req);

    VkMemoryAllocateInfo mai = {0};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (gpu_alloc_memory(gpu, &mai, &gpu->tiled_gbuffer_mem) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tiled G-buffer alloc failed (%zu bytes)\n",
                (size_t)buf_size);
        vkDestroyBuffer(gpu->device, gpu->tiled_gbuffer_buf, NULL);
        gpu->tiled_gbuffer_buf  = VK_NULL_HANDLE;
        gpu->tiled_gbuffer_w    = 0;
        gpu->tiled_gbuffer_h    = 0;
        gpu->tiled_gbuffer_size = 0;
        return 0;
    }
    vkBindBufferMemory(gpu->device, gpu->tiled_gbuffer_buf, gpu->tiled_gbuffer_mem, 0);

    fprintf(stderr, "gpu_vulkan: tiled G-buffer created (%ux%u, %.1f MB)\n",
            w, h, (double)buf_size / (1024.0 * 1024.0));

    /* Rebind binding 17 on BOTH tiled descriptor sets. The compute set's
     * binding 17 is rebound separately in gpu_descriptor_writes_for_deferred(). */
    if (gpu->tiled_rt_desc_set) {
        VkDescriptorBufferInfo bi = {0};
        bi.buffer = gpu->tiled_gbuffer_buf;
        bi.offset = 0;
        bi.range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet wr = {0};
        wr.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr.dstSet          = gpu->tiled_rt_desc_set;
        wr.dstBinding      = 17;
        wr.descriptorCount = 1;
        wr.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr.pBufferInfo     = &bi;
        vkUpdateDescriptorSets(gpu->device, 1, &wr, 0, NULL);

        if (gpu->tiled_rt_desc_set_b) {
            wr.dstSet = gpu->tiled_rt_desc_set_b;
            vkUpdateDescriptorSets(gpu->device, 1, &wr, 0, NULL);
        }
    }
    return 1;
}

void gpu_destroy_tiled_rt_pipeline(Gpu* gpu)
{
    if (!gpu) return;
    vkDeviceWaitIdle(gpu->device);
    gpu_invalidate_tiled_cmd_cache(gpu);

    if (gpu->tiled_rt_pipeline) {
        vkDestroyPipeline(gpu->device, gpu->tiled_rt_pipeline, NULL);
        gpu->tiled_rt_pipeline = VK_NULL_HANDLE;
    }
    if (gpu->tiled_rt_pipeline_layout) {
        vkDestroyPipelineLayout(gpu->device, gpu->tiled_rt_pipeline_layout, NULL);
        gpu->tiled_rt_pipeline_layout = VK_NULL_HANDLE;
    }
    if (gpu->tiled_sbt_buf) {
        vkDestroyBuffer(gpu->device, gpu->tiled_sbt_buf, NULL);
        vkFreeMemory(gpu->device, gpu->tiled_sbt_mem, NULL);
        gpu->tiled_sbt_buf = VK_NULL_HANDLE;
        gpu->tiled_sbt_mem = VK_NULL_HANDLE;
    }
    if (gpu->tiled_rt_desc_pool) {
        vkDestroyDescriptorPool(gpu->device, gpu->tiled_rt_desc_pool, NULL);
        gpu->tiled_rt_desc_pool = VK_NULL_HANDLE;
    }
    if (gpu->tiled_rt_desc_layout) {
        vkDestroyDescriptorSetLayout(gpu->device, gpu->tiled_rt_desc_layout, NULL);
        gpu->tiled_rt_desc_layout = VK_NULL_HANDLE;
    }
    gpu->tiled_rt_desc_set = VK_NULL_HANDLE;
    gpu->tiled_rt_desc_set_b = VK_NULL_HANDLE;
    memset(&gpu->tiled_sbt_rgen, 0, sizeof(gpu->tiled_sbt_rgen));
    memset(&gpu->tiled_sbt_miss, 0, sizeof(gpu->tiled_sbt_miss));
    memset(&gpu->tiled_sbt_hit,  0, sizeof(gpu->tiled_sbt_hit));
    memset(&gpu->tiled_sbt_call, 0, sizeof(gpu->tiled_sbt_call));
    gpu->tiled_rt_built = 0;
}

int gpu_tiled_rt_pipeline_built(Gpu* gpu)
{
    return (gpu && gpu->tiled_rt_built) ? 1 : 0;
}

int gpu_tiled_init(Gpu* gpu, uint32_t total_w, uint32_t total_h, int num_cameras)
{
    if (!gpu) return 0;

    int image_changed = 0;
    int camera_changed = 0;
    int image_resize = (gpu->tiled_image_w != total_w || gpu->tiled_image_h != total_h);
    int camera_resize = (gpu->tiled_camera_count != num_cameras || !gpu->tiled_camera_buf);

    if (gpu->tiled_rt_built && (image_resize || camera_resize)) {
        gpu_destroy_tiled_rt_pipeline(gpu);
    }

    /* Create/resize tiled storage image if needed */
    if (image_resize) {
        if (!tiled_create_storage_image(gpu, total_w, total_h))
            return 0;
        image_changed = 1;
    }

    /* Create interop resources (once or on resize) */
    tiled_create_interop_semaphore(gpu);
    tiled_create_interop_buffer(gpu);

    /* Create sensor output buffers */
    tiled_ensure_depth(gpu, total_w, total_h);
    tiled_ensure_segmentation(gpu, total_w, total_h);
    tiled_ensure_normals(gpu, total_w, total_h);

    /* Create/resize camera SSBO: num_cameras * 2 mat4 = num_cameras * 128 bytes */
    VkDeviceSize cam_size = (VkDeviceSize)num_cameras * 32 * sizeof(float);

    if (camera_resize) {
        /* Free old */
        if (gpu->tiled_camera_buf) {
            if (gpu->tiled_camera_mapped)
                vkUnmapMemory(gpu->device, gpu->tiled_camera_mem);
            vkDestroyBuffer(gpu->device, gpu->tiled_camera_buf, NULL);
            vkFreeMemory(gpu->device, gpu->tiled_camera_mem, NULL);
            gpu->tiled_camera_buf    = VK_NULL_HANDLE;
            gpu->tiled_camera_mapped = NULL;
        }

        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = cam_size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->tiled_camera_buf) != VK_SUCCESS)
            return 0;

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->tiled_camera_buf, &req);

        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (gpu_alloc_memory(gpu, &mai, &gpu->tiled_camera_mem) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, gpu->tiled_camera_buf, NULL);
            gpu->tiled_camera_buf = VK_NULL_HANDLE;
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->tiled_camera_buf, gpu->tiled_camera_mem, 0);
        vkMapMemory(gpu->device, gpu->tiled_camera_mem, 0, cam_size, 0, &gpu->tiled_camera_mapped);

        gpu->tiled_camera_count = num_cameras;
        camera_changed = 1;
        fprintf(stderr, "gpu_vulkan: tiled camera SSBO created (%d cameras, %zu bytes)\n",
                num_cameras, (size_t)cam_size);
    }

    /* If descriptor set exists and resources changed, update the relevant bindings */
    if (gpu->tiled_rt_desc_set && (image_changed || camera_changed)) {
        VkWriteDescriptorSet writes[7] = {0};
        int nw = 0;

        VkDescriptorImageInfo img_info = {0};
        if (image_changed) {
            img_info.imageView   = gpu->tiled_image_view;
            img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            writes[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[nw].dstSet          = gpu->tiled_rt_desc_set;
            writes[nw].dstBinding      = 1;
            writes[nw].descriptorCount = 1;
            writes[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[nw].pImageInfo      = &img_info;
            nw++;
        }

        VkDescriptorBufferInfo cam_info = {0};
        if (camera_changed) {
            cam_info.buffer = gpu->tiled_camera_buf;
            cam_info.offset = 0;
            cam_info.range  = VK_WHOLE_SIZE;
            writes[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[nw].dstSet          = gpu->tiled_rt_desc_set;
            writes[nw].dstBinding      = 8;
            writes[nw].descriptorCount = 1;
            writes[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[nw].pBufferInfo     = &cam_info;
            nw++;
        }

        /* Update binding 10 (depth SSBO) when depth buffer was (re)created */
        VkDescriptorBufferInfo depth_info = {0};
        if (image_changed && gpu->tiled_depth_buf) {
            depth_info.buffer = gpu->tiled_depth_buf;
            depth_info.range  = VK_WHOLE_SIZE;
            writes[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[nw].dstSet          = gpu->tiled_rt_desc_set;
            writes[nw].dstBinding      = 10;
            writes[nw].descriptorCount = 1;
            writes[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[nw].pBufferInfo     = &depth_info;
            nw++;
        }

        /* Update binding 11 (segmentation SSBO) when seg buffer was (re)created */
        VkDescriptorBufferInfo seg_info = {0};
        if (image_changed && gpu->tiled_seg_buf) {
            seg_info.buffer = gpu->tiled_seg_buf;
            seg_info.range  = VK_WHOLE_SIZE;
            writes[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[nw].dstSet          = gpu->tiled_rt_desc_set;
            writes[nw].dstBinding      = 11;
            writes[nw].descriptorCount = 1;
            writes[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[nw].pBufferInfo     = &seg_info;
            nw++;
        }

        /* Update binding 12 (normals SSBO) when norm buffer was (re)created */
        VkDescriptorBufferInfo norm_info = {0};
        if (image_changed && gpu->tiled_norm_buf) {
            norm_info.buffer = gpu->tiled_norm_buf;
            norm_info.range  = VK_WHOLE_SIZE;
            writes[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[nw].dstSet          = gpu->tiled_rt_desc_set;
            writes[nw].dstBinding      = 12;
            writes[nw].descriptorCount = 1;
            writes[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[nw].pBufferInfo     = &norm_info;
            nw++;
        }

        /* Update binding 9 (direct output SSBO) when interop buffer changed.
         * Set A → buf[0], also need a separate write for set B → buf[1]. */
        VkDescriptorBufferInfo direct_info_a = {0};
        VkDescriptorBufferInfo direct_info_b = {0};
        int binding9_write_idx = -1;
        if (image_changed && gpu->interop_buf[0]) {
            direct_info_a.buffer = gpu->interop_buf[0];
            direct_info_a.range  = VK_WHOLE_SIZE;
            writes[nw].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[nw].dstSet          = gpu->tiled_rt_desc_set;
            writes[nw].dstBinding      = 9;
            writes[nw].descriptorCount = 1;
            writes[nw].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[nw].pBufferInfo     = &direct_info_a;
            binding9_write_idx = nw;
            nw++;
        }

        if (nw > 0) {
            vkUpdateDescriptorSets(gpu->device, (uint32_t)nw, writes, 0, NULL);
            /* Also update set B with the same changes */
            for (int w = 0; w < nw; w++)
                writes[w].dstSet = gpu->tiled_rt_desc_set_b;
            if (binding9_write_idx >= 0) {
                direct_info_b.buffer = gpu->interop_buf[1] ? gpu->interop_buf[1] : gpu->tiled_camera_buf;
                direct_info_b.range  = VK_WHOLE_SIZE;
                writes[binding9_write_idx].pBufferInfo = &direct_info_b;
            }
            vkUpdateDescriptorSets(gpu->device, (uint32_t)nw, writes, 0, NULL);
            fprintf(stderr, "gpu_vulkan: tiled descriptors updated (image=%d, camera=%d)\n",
                    image_changed, camera_changed);
        }
    }

    return 1;
}

int gpu_tiled_upload_cameras(Gpu* gpu, const float* data, int num_cameras)
{
    if (!gpu || !gpu->tiled_camera_mapped) return 0;
    if (num_cameras > gpu->tiled_camera_count) return 0;

    memcpy(gpu->tiled_camera_mapped, data, (size_t)num_cameras * 32 * sizeof(float));
    return 1;
}

int gpu_build_tiled_rt_pipeline(Gpu* gpu,
                                 const uint32_t* rgen_spv, uint32_t rgen_size,
                                 const uint32_t* miss_spv, uint32_t miss_size,
                                 const uint32_t* chit_spv, uint32_t chit_size,
                                 const uint32_t* rint_spv, uint32_t rint_size,
                                 const uint32_t* curve_chit_spv, uint32_t curve_chit_size)
{
    if (!gpu || !gpu->rt_available) return 0;

    /* Phase 11.A.2.5b: optional curves in tiled IsaacLab pipeline.
     * Same gating as gpu_build_rt_scene — gpu->curve_blas / curve_seg_count
     * must be populated by gpu_upload_curve_data + gpu_build_curve_blas
     * before this is called. NULL/0 keeps the legacy 3-group pipeline. */
    int has_curves = (rint_spv && curve_chit_spv && rint_size > 0 && curve_chit_size > 0
                      && gpu->curve_blas != VK_NULL_HANDLE
                      && gpu->curve_seg_count > 0);

    /* ---- Descriptor set layout ---- */
    /* Same as regular RT but with additional binding 8 for camera SSBO */
    {
        int has_mat_ssbo = (gpu->mat_uploaded && gpu->mat_count > 0) ? 1 : 0;
        int has_textures = (gpu->mat_uploaded && gpu->mat_tex_count > 0) ? 1 : 0;
        int has_ibl = (gpu->env_image_view != VK_NULL_HANDLE) ? 1 : 0;
        int rt_tex_count = has_textures ? gpu->mat_tex_count : 0;
        /* bindings: 0=TLAS, 1=image, 2=scene, [3=mat], [4=tex], [5-7=ibl], 8=cam, 9=direct, 10=depth, 11=seg, 12=norm, 13=lights */
        int num_bindings = 3 + (has_mat_ssbo ? 1 : 0) + (has_textures ? 1 : 0)
                         + (has_ibl ? 3 : 0) + 1 + 1 + 1 + 1 + 1 + 1; /* cam+direct+depth+seg+norm+lights */
        (void)num_bindings;

        VkDescriptorSetLayoutBinding bindings[19] = {0};
        /* Binding 0: TLAS */
        bindings[0].binding         = 0;
        bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;

        /* Binding 1: Storage image (tiled) */
        bindings[1].binding         = 1;
        bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

        /* Binding 2: Scene data SSBO */
        bindings[2].binding         = 2;
        bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;

        int bind_idx = 3;
        if (has_mat_ssbo) {
            bindings[bind_idx].binding         = 3;
            bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[bind_idx].descriptorCount = 1;
            bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bind_idx++;
        }
        if (has_textures) {
            bindings[bind_idx].binding         = 4;
            bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[bind_idx].descriptorCount = (uint32_t)rt_tex_count;
            bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bind_idx++;
        }
        if (has_ibl) {
            for (int b = 5; b <= 7; b++) {
                bindings[bind_idx].binding         = (uint32_t)b;
                bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                bindings[bind_idx].descriptorCount = 1;
                bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
                bind_idx++;
            }
        }

        /* Binding 8: Camera SSBO (ray gen only) */
        bindings[bind_idx].binding         = 8;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        bind_idx++;

        /* Binding 9: Direct output buffer (ray gen only, for zero-copy interop) */
        bindings[bind_idx].binding         = 9;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        bind_idx++;

        /* Binding 10: Depth output SSBO (written by raygen, closest-hit, and miss) */
        bindings[bind_idx].binding         = 10;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        bind_idx++;

        /* Binding 11: Segmentation output SSBO (written by raygen, closest-hit, and miss) */
        bindings[bind_idx].binding         = 11;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        bind_idx++;

        /* Binding 12: Normals output SSBO (written by closest-hit, and miss) */
        bindings[bind_idx].binding         = 12;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        bind_idx++;

        /* Binding 13: Scene lights SSBO (read by closest-hit and miss) */
        bindings[bind_idx].binding         = 13;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        bind_idx++;

        /* Bindings 14, 15: Curve segments + per-segment colors (Phase 11.A.2.5b).
         * Same shape as the simple gpu_build_rt_scene path. */
        if (has_curves) {
            bindings[bind_idx].binding         = 14;
            bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[bind_idx].descriptorCount = 1;
            bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bind_idx++;

            bindings[bind_idx].binding         = 15;
            bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[bind_idx].descriptorCount = 1;
            bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            bind_idx++;
        }

        /* Phase C: Binding 16 — per-env TLAS array. Always declared, even
         * when no partition is set (num_envs=0 → 1-element array bound
         * to the legacy single TLAS as fallback so the SPIR-V reference
         * is satisfied). When num_envs>0, descriptorCount=num_envs and
         * each slot holds tlas_arr[e]. Uses descriptor-indexing features
         * (PARTIALLY_BOUND + UPDATE_AFTER_BIND) for resilience to scene
         * rebuilds. Stage = ray-gen only — closest-hit / miss don't need
         * the array for now. */
        int phase_c_tlas_idx = bind_idx;  /* save for the binding-flags array */
        uint32_t tlas_arr_count = (gpu->num_envs > 0) ? (uint32_t)gpu->num_envs : 1;
        bindings[bind_idx].binding         = 16;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[bind_idx].descriptorCount = tlas_arr_count;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        bind_idx++;

        /* Phase A deferred-shading: binding 17 — G-buffer SSBO. The shared
         * rchit + rmiss declare this binding, so the tiled descriptor set
         * must include it even though the tiled pipeline doesn't yet
         * consume the data (Phase B will). A 1 KB scratch SSBO is enough
         * to satisfy the validator; the rchit/rmiss writes for tiled
         * launches go into the first 32 entries (innocuously — nothing
         * reads them yet). Phase B will resize this to width*height*32 B. */
        bindings[bind_idx].binding         = 17;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        bind_idx++;

        /* Binding 18: packed RGBA8 real material colors sampled from authored Ptex maps. */
        bindings[bind_idx].binding         = 18;
        bindings[bind_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bind_idx].descriptorCount = 1;
        bindings[bind_idx].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        bind_idx++;

        /* Per-binding flags: only binding 16 needs PARTIALLY_BOUND +
         * UPDATE_AFTER_BIND; everything else is a normal descriptor.
         * Phase A: bumped to [18] for binding 17 (G-buffer stub).
         * Real Ptex colors use binding 18. */
        VkDescriptorBindingFlags bind_flags[19] = {0};
        bind_flags[phase_c_tlas_idx] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
                                     | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo bind_flags_ci = {0};
        bind_flags_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        bind_flags_ci.bindingCount  = (uint32_t)bind_idx;
        bind_flags_ci.pBindingFlags = bind_flags;

        VkDescriptorSetLayoutCreateInfo layout_ci = {0};
        layout_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_ci.pNext        = &bind_flags_ci;
        layout_ci.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layout_ci.bindingCount = (uint32_t)bind_idx;
        layout_ci.pBindings    = bindings;
        vkCreateDescriptorSetLayout(gpu->device, &layout_ci, NULL, &gpu->tiled_rt_desc_layout);

        /* Descriptor pool */
        /* Pool counts are 2x because we allocate two descriptor sets (A and B)
         * for double-buffered SSBO direct write: set A has binding 9 → buf[0],
         * set B has binding 9 → buf[1]. All other bindings are identical.
         * Phase C: AS pool size is 2 (binding 0) + 2*tlas_arr_count (binding 16). */
        int ibl_sampler_count = has_ibl ? 3 : 0;
        int ssbo_per_set = 1 + (has_mat_ssbo ? 1 : 0) + 1 + 1 + 1 + 1 + 1 + 1
                         + (has_curves ? 2 : 0) + 2; /* scene+mat+cam+direct+depth+seg+norm+lights+curves+gbuffer+ptex */
        VkDescriptorPoolSize pool_sizes[4] = {
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2 + 2 * tlas_arr_count },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)(ssbo_per_set * 2) },
        };
        int num_pool_sizes = 3;
        if (has_textures || has_ibl) {
            pool_sizes[num_pool_sizes].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            pool_sizes[num_pool_sizes].descriptorCount = (uint32_t)((rt_tex_count + ibl_sampler_count) * 2);
            num_pool_sizes++;
        }

        VkDescriptorPoolCreateInfo pool_ci = {0};
        pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_ci.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool_ci.maxSets       = 2;
        pool_ci.poolSizeCount = (uint32_t)num_pool_sizes;
        pool_ci.pPoolSizes    = pool_sizes;
        vkCreateDescriptorPool(gpu->device, &pool_ci, NULL, &gpu->tiled_rt_desc_pool);

        VkDescriptorSetLayout layouts[2] = { gpu->tiled_rt_desc_layout, gpu->tiled_rt_desc_layout };
        VkDescriptorSet       sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VkDescriptorSetAllocateInfo alloc_info = {0};
        alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool     = gpu->tiled_rt_desc_pool;
        alloc_info.descriptorSetCount = 2;
        alloc_info.pSetLayouts        = layouts;
        vkAllocateDescriptorSets(gpu->device, &alloc_info, sets);
        gpu->tiled_rt_desc_set   = sets[0];
        gpu->tiled_rt_desc_set_b = sets[1];

        /* Write descriptors */
        VkWriteDescriptorSetAccelerationStructureKHR as_write = {0};
        as_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        as_write.accelerationStructureCount = 1;
        as_write.pAccelerationStructures    = &gpu->tlas;

        VkDescriptorImageInfo img_info = {0};
        img_info.imageView   = gpu->tiled_image_view;
        img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo scene_buf_info = {0};
        scene_buf_info.buffer = gpu->scene_data_buf;
        scene_buf_info.offset = 0;
        scene_buf_info.range  = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo cam_buf_info = {0};
        cam_buf_info.buffer = gpu->tiled_camera_buf;
        cam_buf_info.offset = 0;
        cam_buf_info.range  = VK_WHOLE_SIZE;

        /* Phase C: bumped from [16] to [17] for binding 16 (TLAS array).
         * Phase A: bumped to [18] for binding 17 (G-buffer SSBO).
         * Real Ptex colors use binding 18. */
        VkWriteDescriptorSet writes[19] = {0};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].pNext           = &as_write;
        writes[0].dstSet          = gpu->tiled_rt_desc_set;
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = gpu->tiled_rt_desc_set;
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo      = &img_info;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = gpu->tiled_rt_desc_set;
        writes[2].dstBinding      = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo     = &scene_buf_info;

        int num_writes = 3;

        VkDescriptorBufferInfo mat_buf_info = {0};
        if (has_mat_ssbo) {
            mat_buf_info.buffer = gpu->mat_ssbo_buf;
            mat_buf_info.offset = 0;
            mat_buf_info.range  = VK_WHOLE_SIZE;
            writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
            writes[num_writes].dstBinding      = 3;
            writes[num_writes].descriptorCount = 1;
            writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[num_writes].pBufferInfo     = &mat_buf_info;
            num_writes++;
        }

        VkDescriptorImageInfo* tex_infos = NULL;
        if (has_textures) {
            tex_infos = (VkDescriptorImageInfo*)calloc(
                (size_t)rt_tex_count, sizeof(VkDescriptorImageInfo));
            for (int t = 0; t < rt_tex_count; t++) {
                tex_infos[t].sampler     = gpu->mat_sampler;
                tex_infos[t].imageView   = gpu->mat_image_views[t]
                    ? gpu->mat_image_views[t] : gpu->mat_image_views[0];
                tex_infos[t].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
            writes[num_writes].dstBinding      = 4;
            writes[num_writes].descriptorCount = (uint32_t)rt_tex_count;
            writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[num_writes].pImageInfo      = tex_infos;
            num_writes++;
        }

        VkDescriptorImageInfo ibl_infos[3] = {0};
        if (has_ibl) {
            ibl_infos[0].sampler     = gpu->env_sampler;
            ibl_infos[0].imageView   = gpu->env_image_view;
            ibl_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ibl_infos[1].sampler     = gpu->env_sampler;
            ibl_infos[1].imageView   = gpu->brdf_lut_view;
            ibl_infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ibl_infos[2].sampler     = gpu->irr_sampler ? gpu->irr_sampler : gpu->env_sampler;
            ibl_infos[2].imageView   = gpu->irr_image_view ? gpu->irr_image_view : gpu->env_image_view;
            ibl_infos[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            for (int b = 0; b < 3; b++) {
                writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
                writes[num_writes].dstBinding      = (uint32_t)(5 + b);
                writes[num_writes].descriptorCount = 1;
                writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[num_writes].pImageInfo      = &ibl_infos[b];
                num_writes++;
            }
        }

        /* Camera SSBO */
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
        writes[num_writes].dstBinding      = 8;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &cam_buf_info;
        num_writes++;

        /* Direct output SSBO (binding 9) — set A → interop_buf[0], set B → interop_buf[1].
         * Falls back to camera SSBO if interop not available (shader won't access it). */
        VkDescriptorBufferInfo direct_buf_info_a = {0};
        direct_buf_info_a.buffer = gpu->interop_buf[0] ? gpu->interop_buf[0] : gpu->tiled_camera_buf;
        direct_buf_info_a.range  = VK_WHOLE_SIZE;
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
        writes[num_writes].dstBinding      = 9;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &direct_buf_info_a;
        int binding9_write_idx_initial = num_writes;
        num_writes++;

        /* Depth output SSBO (binding 10) — shared between sets A and B.
         * Falls back to camera SSBO if depth buffer not yet created. */
        VkDescriptorBufferInfo depth_buf_info = {0};
        depth_buf_info.buffer = gpu->tiled_depth_buf ? gpu->tiled_depth_buf : gpu->tiled_camera_buf;
        depth_buf_info.range  = VK_WHOLE_SIZE;
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
        writes[num_writes].dstBinding      = 10;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &depth_buf_info;
        num_writes++;

        /* Segmentation output SSBO (binding 11) — shared between sets A and B. */
        VkDescriptorBufferInfo seg_buf_info = {0};
        seg_buf_info.buffer = gpu->tiled_seg_buf ? gpu->tiled_seg_buf : gpu->tiled_camera_buf;
        seg_buf_info.range  = VK_WHOLE_SIZE;
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
        writes[num_writes].dstBinding      = 11;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &seg_buf_info;
        num_writes++;

        /* Normals output SSBO (binding 12) — shared between sets A and B. */
        VkDescriptorBufferInfo norm_buf_info = {0};
        norm_buf_info.buffer = gpu->tiled_norm_buf ? gpu->tiled_norm_buf : gpu->tiled_camera_buf;
        norm_buf_info.range  = VK_WHOLE_SIZE;
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
        writes[num_writes].dstBinding      = 12;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &norm_buf_info;
        num_writes++;

        /* Scene lights SSBO (binding 13) — shared between sets A and B.
         * Procedural add_mesh scenes do not upload lights through the USD
         * path, so create a valid nlights=0 header before binding. */
        if (!gpu->light_ssbo_buf)
            gpu_upload_lights(gpu, NULL, 0);
        VkDescriptorBufferInfo lights_buf_info = {0};
        lights_buf_info.buffer = gpu->light_ssbo_buf ? gpu->light_ssbo_buf : gpu->tiled_camera_buf;
        lights_buf_info.range  = VK_WHOLE_SIZE;
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
        writes[num_writes].dstBinding      = 13;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &lights_buf_info;
        num_writes++;

        /* Phase 11.A.2.5b: bindings 14, 15 — curve segments + colors. */
        VkDescriptorBufferInfo curve_seg_info = {0};
        VkDescriptorBufferInfo curve_col_info = {0};
        if (has_curves) {
            curve_seg_info.buffer = gpu->curve_seg_ssbo_buf;
            curve_seg_info.range  = VK_WHOLE_SIZE;
            writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
            writes[num_writes].dstBinding      = 14;
            writes[num_writes].descriptorCount = 1;
            writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[num_writes].pBufferInfo     = &curve_seg_info;
            num_writes++;

            curve_col_info.buffer = gpu->curve_color_ssbo_buf;
            curve_col_info.range  = VK_WHOLE_SIZE;
            writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
            writes[num_writes].dstBinding      = 15;
            writes[num_writes].descriptorCount = 1;
            writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[num_writes].pBufferInfo     = &curve_col_info;
            num_writes++;
        }

        /* Phase A deferred-shading: stub for binding 17 (G-buffer SSBO).
         * The tiled pipeline doesn't yet consume the data; binding scene_data_buf
         * (already host-allocated for this scene build) satisfies the SPIR-V
         * reference. Phase B will swap this for a per-launch device-local
         * buffer sized to (tile_w * tile_h * num_cameras * 32) bytes. */
        VkDescriptorBufferInfo gbuf_stub_info = {0};
        gbuf_stub_info.buffer = gpu->scene_data_buf;
        gbuf_stub_info.offset = 0;
        gbuf_stub_info.range  = VK_WHOLE_SIZE;
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
        writes[num_writes].dstBinding      = 17;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &gbuf_stub_info;
        num_writes++;

        VkDescriptorBufferInfo ptex_tri_info = {0};
        ptex_tri_info.buffer = gpu->rt_tri_color_ssbo_buf
            ? gpu->rt_tri_color_ssbo_buf : gpu->scene_data_buf;
        ptex_tri_info.offset = 0;
        ptex_tri_info.range  = VK_WHOLE_SIZE;
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
        writes[num_writes].dstBinding      = 18;
        writes[num_writes].descriptorCount = 1;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[num_writes].pBufferInfo     = &ptex_tri_info;
        num_writes++;

        /* Phase C: write TLAS array (binding 16). When num_envs > 0 use
         * gpu->tlas_arr; otherwise fall back to a 1-element array of the
         * legacy TLAS so the SPIR-V reference is satisfied even when the
         * partition is off. */
        VkAccelerationStructureKHR* tlas_array_handles = NULL;
        VkWriteDescriptorSetAccelerationStructureKHR tlas_arr_as_write = {0};
        tlas_arr_as_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        tlas_arr_as_write.accelerationStructureCount = tlas_arr_count;
        if (gpu->num_envs > 0 && gpu->tlas_arr) {
            tlas_arr_as_write.pAccelerationStructures = gpu->tlas_arr;
        } else {
            /* Single-slot fallback bound to legacy tlas. */
            tlas_array_handles = (VkAccelerationStructureKHR*)calloc(1, sizeof(VkAccelerationStructureKHR));
            tlas_array_handles[0] = gpu->tlas;
            tlas_arr_as_write.pAccelerationStructures = tlas_array_handles;
        }
        int phase_c_write_idx = num_writes;
        writes[num_writes].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[num_writes].pNext           = &tlas_arr_as_write;
        writes[num_writes].dstSet          = gpu->tiled_rt_desc_set;
        writes[num_writes].dstBinding      = 16;
        writes[num_writes].dstArrayElement = 0;
        writes[num_writes].descriptorCount = tlas_arr_count;
        writes[num_writes].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        num_writes++;

        vkUpdateDescriptorSets(gpu->device, (uint32_t)num_writes, writes, 0, NULL);

        /* Write set B: identical to set A except binding 9 → interop_buf[1]. */
        for (int w = 0; w < num_writes; w++)
            writes[w].dstSet = gpu->tiled_rt_desc_set_b;
        /* Fix binding 9 to point to buf[1] */
        VkDescriptorBufferInfo direct_buf_info_b = {0};
        direct_buf_info_b.buffer = gpu->interop_buf[1] ? gpu->interop_buf[1] : gpu->tiled_camera_buf;
        direct_buf_info_b.range  = VK_WHOLE_SIZE;
        writes[binding9_write_idx_initial].pBufferInfo = &direct_buf_info_b;
        /* Set B reuses the same TLAS-array — re-attach the pNext pointer. */
        writes[phase_c_write_idx].pNext = &tlas_arr_as_write;

        vkUpdateDescriptorSets(gpu->device, (uint32_t)num_writes, writes, 0, NULL);
        free(tlas_array_handles);
        free(tex_infos);
    }

    /* ---- Pipeline layout ---- */
    {
        VkPushConstantRange pcr = {0};
        pcr.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        pcr.offset     = 0;
        pcr.size       = sizeof(GpuRtTiledPushConstants);

        VkPipelineLayoutCreateInfo plci = {0};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &gpu->tiled_rt_desc_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        vkCreatePipelineLayout(gpu->device, &plci, NULL, &gpu->tiled_rt_pipeline_layout);
    }

    /* ---- RT pipeline ---- */
    {
        VkShaderModule rgen_mod = create_shader_module(gpu->device, rgen_spv, rgen_size);
        VkShaderModule miss_mod = create_shader_module(gpu->device, miss_spv, miss_size);
        VkShaderModule chit_mod = create_shader_module(gpu->device, chit_spv, chit_size);
        /* Phase 11.A.2.5b: optional curve shaders. */
        VkShaderModule rint_mod = has_curves
            ? create_shader_module(gpu->device, rint_spv, rint_size) : VK_NULL_HANDLE;
        VkShaderModule curve_chit_mod = has_curves
            ? create_shader_module(gpu->device, curve_chit_spv, curve_chit_size) : VK_NULL_HANDLE;

        VkPipelineShaderStageCreateInfo stages[5] = {0};
        uint32_t n_stages = 0;
        stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[n_stages].stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        stages[n_stages].module = rgen_mod;
        stages[n_stages].pName  = "main";
        uint32_t rgen_idx = n_stages++;
        stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[n_stages].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
        stages[n_stages].module = miss_mod;
        stages[n_stages].pName  = "main";
        uint32_t miss_idx = n_stages++;
        stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[n_stages].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        stages[n_stages].module = chit_mod;
        stages[n_stages].pName  = "main";
        uint32_t chit_idx = n_stages++;
        uint32_t rint_idx = 0, curve_chit_idx = 0;
        if (has_curves) {
            stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[n_stages].stage  = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
            stages[n_stages].module = rint_mod;
            stages[n_stages].pName  = "main";
            rint_idx = n_stages++;
            stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[n_stages].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            stages[n_stages].module = curve_chit_mod;
            stages[n_stages].pName  = "main";
            curve_chit_idx = n_stages++;
        }

        VkRayTracingShaderGroupCreateInfoKHR groups[4] = {0};
        uint32_t n_groups = 0;
        groups[n_groups].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        groups[n_groups].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[n_groups].generalShader      = rgen_idx;
        groups[n_groups].closestHitShader   = VK_SHADER_UNUSED_KHR;
        groups[n_groups].anyHitShader       = VK_SHADER_UNUSED_KHR;
        groups[n_groups].intersectionShader = VK_SHADER_UNUSED_KHR;
        n_groups++;
        groups[n_groups].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        groups[n_groups].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[n_groups].generalShader      = miss_idx;
        groups[n_groups].closestHitShader   = VK_SHADER_UNUSED_KHR;
        groups[n_groups].anyHitShader       = VK_SHADER_UNUSED_KHR;
        groups[n_groups].intersectionShader = VK_SHADER_UNUSED_KHR;
        n_groups++;
        groups[n_groups].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        groups[n_groups].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        groups[n_groups].generalShader      = VK_SHADER_UNUSED_KHR;
        groups[n_groups].closestHitShader   = chit_idx;
        groups[n_groups].anyHitShader       = VK_SHADER_UNUSED_KHR;
        groups[n_groups].intersectionShader = VK_SHADER_UNUSED_KHR;
        n_groups++;
        if (has_curves) {
            groups[n_groups].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            groups[n_groups].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
            groups[n_groups].generalShader      = VK_SHADER_UNUSED_KHR;
            groups[n_groups].closestHitShader   = curve_chit_idx;
            groups[n_groups].anyHitShader       = VK_SHADER_UNUSED_KHR;
            groups[n_groups].intersectionShader = rint_idx;
            n_groups++;
        }

        VkRayTracingPipelineCreateInfoKHR rt_ci = {0};
        rt_ci.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        rt_ci.stageCount                   = n_stages;
        rt_ci.pStages                      = stages;
        rt_ci.groupCount                   = n_groups;
        rt_ci.pGroups                      = groups;
        rt_ci.maxPipelineRayRecursionDepth = 3;
        rt_ci.layout                       = gpu->tiled_rt_pipeline_layout;

        struct timespec _tiled_rt_t0, _tiled_rt_t1;
        clock_gettime(CLOCK_MONOTONIC, &_tiled_rt_t0);
        VkResult _tiled_rt_res = vkCreateRayTracingPipelinesKHR(
            gpu->device, VK_NULL_HANDLE, gpu->pipeline_cache,
            1, &rt_ci, NULL, &gpu->tiled_rt_pipeline);
        clock_gettime(CLOCK_MONOTONIC, &_tiled_rt_t1);
        double _tiled_rt_ms =
            (_tiled_rt_t1.tv_sec  - _tiled_rt_t0.tv_sec)  * 1000.0 +
            (_tiled_rt_t1.tv_nsec - _tiled_rt_t0.tv_nsec) / 1e6;
        fprintf(stderr,
            "gpu_vulkan: create tiled RT pipeline took %.1f ms (cache=%s)\n",
            _tiled_rt_ms, gpu->pipeline_cache ? "on" : "off");
        if (_tiled_rt_res != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: failed to create tiled RT pipeline\n");
            vkDestroyShaderModule(gpu->device, rgen_mod, NULL);
            vkDestroyShaderModule(gpu->device, miss_mod, NULL);
            vkDestroyShaderModule(gpu->device, chit_mod, NULL);
            if (rint_mod)       vkDestroyShaderModule(gpu->device, rint_mod, NULL);
            if (curve_chit_mod) vkDestroyShaderModule(gpu->device, curve_chit_mod, NULL);
            return 0;
        }

        vkDestroyShaderModule(gpu->device, rgen_mod, NULL);
        vkDestroyShaderModule(gpu->device, miss_mod, NULL);
        vkDestroyShaderModule(gpu->device, chit_mod, NULL);
        if (rint_mod)       vkDestroyShaderModule(gpu->device, rint_mod, NULL);
        if (curve_chit_mod) vkDestroyShaderModule(gpu->device, curve_chit_mod, NULL);
    }

    /* ---- Shader Binding Table ---- */
    {
        uint32_t handle_size   = gpu->rt_handle_size;
        uint32_t align         = gpu->rt_handle_alignment;
        uint32_t handle_stride = (handle_size + (align - 1)) & ~(align - 1);
        /* Phase 11.A.2.5b: 4 entries when has_curves (rgen | miss | mesh-hit | curve-hit). */
        uint32_t n_hit_groups = 1u + (has_curves ? 1u : 0u);
        uint32_t n_total      = 2u + n_hit_groups;
        uint32_t sbt_size     = handle_stride * n_total;

        uint8_t* handles = malloc((size_t)handle_size * n_total);
        vkGetRayTracingShaderGroupHandlesKHR(gpu->device, gpu->tiled_rt_pipeline, 0, n_total,
                                             handle_size * n_total, handles);

        if (!rt_create_buffer(gpu, sbt_size,
                         VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &gpu->tiled_sbt_buf, &gpu->tiled_sbt_mem)) {
            fprintf(stderr, "gpu_vulkan: tiled SBT alloc failed\n");
            free(handles);
            return 0;
        }

        void* mapped;
        vkMapMemory(gpu->device, gpu->tiled_sbt_mem, 0, sbt_size, 0, &mapped);
        memset(mapped, 0, sbt_size);
        for (uint32_t g = 0; g < n_total; g++) {
            memcpy((uint8_t*)mapped + handle_stride * g,
                   handles + handle_size * g, handle_size);
        }
        vkUnmapMemory(gpu->device, gpu->tiled_sbt_mem);
        free(handles);

        VkDeviceAddress sbt_addr = rt_buf_addr(gpu, gpu->tiled_sbt_buf);

        gpu->tiled_sbt_rgen.deviceAddress = sbt_addr + handle_stride * 0;
        gpu->tiled_sbt_rgen.stride        = handle_stride;
        gpu->tiled_sbt_rgen.size          = handle_stride;

        gpu->tiled_sbt_miss.deviceAddress = sbt_addr + handle_stride * 1;
        gpu->tiled_sbt_miss.stride        = handle_stride;
        gpu->tiled_sbt_miss.size          = handle_stride;

        /* Hit-group region spans both triangle (offset 0) and curve (offset 1) groups. */
        gpu->tiled_sbt_hit.deviceAddress  = sbt_addr + handle_stride * 2;
        gpu->tiled_sbt_hit.stride         = handle_stride;
        gpu->tiled_sbt_hit.size           = handle_stride * n_hit_groups;

        memset(&gpu->tiled_sbt_call, 0, sizeof(gpu->tiled_sbt_call));
    }

    gpu->tiled_rt_built = 1;
    fprintf(stderr, "gpu_vulkan: tiled RT pipeline built successfully\n");
    return 1;
}

int gpu_begin_frame_tiled_rt(Gpu* gpu)
{
    if (!gpu || !gpu->tiled_rt_built) return 0;

    /* Pick the cache slot for this frame.  Both readback_write_idx and
     * interop_write_idx flip together starting at 0, so they always agree.
     * Use interop_write_idx as the canonical parity. */
    int parity = gpu->interop_write_idx & 1;

    /* If the cache is externally dirty, drop both slots once. */
    if (gpu->tiled_cmd_cache_dirty) {
        gpu->tiled_cmd_cache_valid[0] = 0;
        gpu->tiled_cmd_cache_valid[1] = 0;
        gpu->tiled_cmd_cache_dirty    = 0;
    }

    /* Reset replay state for this frame.  gpu_cmd_trace_rays_tiled() will
     * upgrade it to a replay if pc matches. */
    gpu->tiled_cmd_replay_active = 0;
    gpu->tiled_cmd_replay_idx    = parity;
    gpu->tiled_cmd_has_tlas_update = 0;

    /* Lazily allocate the persistent cmd buffer for this parity slot.
     * It uses SIMULTANEOUS_USE_BIT so the previous slot's submission can
     * still be in flight via tiled_readback_fence. */
    if (gpu->tiled_cached_cmd[parity] == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cai = {0};
        cai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool        = gpu->command_pool;
        cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(gpu->device, &cai,
                                     &gpu->tiled_cached_cmd[parity]) != VK_SUCCESS) {
            return 0;
        }
    }

    VkCommandBuffer cmd = gpu->tiled_cached_cmd[parity];

    /* Wait on the previous submission of this parity slot's fence
     * unconditionally.  Even on the deferred-replay path, the cmd buffer
     * may be in pending state — replay re-submission is allowed under
     * SIMULTANEOUS_USE_BIT, but if the path later falls through to
     * vkResetCommandBuffer (pc differs, or gpu_update_tlas_inline runs),
     * Vulkan VUID-vkResetCommandBuffer-commandBuffer-00045 forbids
     * resetting a pending buffer.  A signaled fence-wait is essentially
     * free, so do it here once for both paths. */
    if (gpu->tiled_readback_submitted[parity] && gpu->tiled_readback_fence[parity]) {
        vkWaitForFences(gpu->device, 1, &gpu->tiled_readback_fence[parity],
                        VK_TRUE, UINT64_MAX);
        vkResetFences(gpu->device, 1, &gpu->tiled_readback_fence[parity]);
        gpu->tiled_readback_submitted[parity] = 0;
        gpu->tiled_readback_cmd[parity]       = VK_NULL_HANDLE;
    }

    /* If cache slot is valid, defer Begin: gpu_cmd_trace_rays_tiled() will
     * decide whether the pc matches and we can replay verbatim. */
    if (gpu->tiled_cmd_cache_valid[parity]) {
        gpu->current_cmd = cmd;  /* used only for cache-key bookkeeping; no recording */
        return 1;
    }

    /* Cache miss: Reset+Begin (safe now that the fence has signaled). */
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
        return 0;
    }
    gpu->current_cmd = cmd;
    return 1;
}

void gpu_abort_frame_tiled_rt(Gpu* gpu)
{
    if (!gpu || !gpu->current_cmd) return;
    /* If we're aborting after a replay decision, no recording was begun;
     * just clear state. */
    if (gpu->tiled_cmd_replay_active) {
        gpu->tiled_cmd_replay_active = 0;
        gpu->current_cmd = VK_NULL_HANDLE;
        return;
    }
    /* Otherwise the buffer is in the recording state.  End it (so it
     * can be reset+rebegun cleanly later) but keep the persistent
     * allocation — the cache slot was invalidated by whoever caused
     * the abort. */
    vkEndCommandBuffer(gpu->current_cmd);
    int parity = gpu->tiled_cmd_replay_idx & 1;
    gpu->tiled_cmd_cache_valid[parity] = 0;
    gpu->current_cmd = VK_NULL_HANDLE;
}

/* Comparison helper for tiled push constants. */
static int tiled_push_consts_equal(const GpuRtTiledPushConstants* a,
                                   const GpuRtTiledPushConstants* b)
{
    return memcmp(a, b, sizeof(GpuRtTiledPushConstants)) == 0;
}

void gpu_cmd_trace_rays_tiled(Gpu* gpu, const GpuRtTiledPushConstants* pc)
{
    int parity = gpu->tiled_cmd_replay_idx & 1;

    /* If gpu_begin_frame_tiled_rt() found a valid cache slot, decide here
     * whether the new pc matches the stored one.  Match → mark replay
     * active and skip recording entirely.  Mismatch → fall through to
     * recording (the cmd buffer is still in its previous "ended" state
     * and needs to be reset+begun before we can record into it). */
    if (gpu->tiled_cmd_cache_valid[parity] &&
        gpu->tiled_cmd_cache_direct_write[parity] == gpu->direct_write_active &&
        tiled_push_consts_equal(pc, &gpu->tiled_cmd_cache_pc[parity])) {
        gpu->tiled_cmd_replay_active = 1;
        return;
    }

    /* Cache miss path: if begin_frame deferred the Begin call (because the
     * slot looked valid at the time), perform it now. */
    if (gpu->tiled_cmd_cache_valid[parity]) {
        gpu->tiled_cmd_cache_valid[parity] = 0;
        VkCommandBuffer cmd = gpu->tiled_cached_cmd[parity];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo begin = {0};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        vkBeginCommandBuffer(cmd, &begin);
        gpu->current_cmd = cmd;
    }

    VkCommandBuffer cmd = gpu->current_cmd;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, gpu->tiled_rt_pipeline);
    /* In direct-write mode, pick the descriptor set for the current interop write buffer.
     * Set A (desc_set) has binding 9 → buf[0], set B (desc_set_b) → buf[1]. */
    VkDescriptorSet active_set = (gpu->direct_write_active && gpu->interop_write_idx == 1)
                                 ? gpu->tiled_rt_desc_set_b : gpu->tiled_rt_desc_set;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            gpu->tiled_rt_pipeline_layout, 0, 1, &active_set, 0, NULL);
    vkCmdPushConstants(cmd, gpu->tiled_rt_pipeline_layout,
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                       0, sizeof(GpuRtTiledPushConstants), pc);

    /* NU_TILE_RES probe: when the rgen launches at SCALED tile dimensions
     * but the caller-allocated direct-write buffer is sized at the UNSCALED
     * (default 64x64) per-env stride, the rgen only fills the first
     * (scaled_tile_w * scaled_tile_h) bytes of each per-env slot — the
     * remaining bytes per env retain whatever was in the buffer before
     * (possibly 0xAB pre-fill, possibly stale data). Pre-clear the entire
     * direct-write buffer to opaque black (0xFF000000) before each rgen
     * dispatch so the unrendered region is well-defined.
     *
     * Cost: one vkCmdFillBuffer at GPU memory bandwidth (~0.1 ms for 64 MB
     * on RTX 5090). The fill is recorded into the persistent cmd buffer
     * cache, so it's a one-time cost per cache miss.
     *
     * Gating:
     *   - Only when direct_write_active (rgen writes to SSBO, not image).
     *   - Only when external_buffers_active (caller-allocated buffer; the
     *     legacy renderer-allocated path sizes the interop buf to exactly
     *     scaled dims, so no padding region exists).
     *   - Only when interop_buf_size > scaled coverage (the rgen does not
     *     fill the whole buffer). At default NU_TILE_RES the rgen fills
     *     the whole buffer, so this guard skips the fill — keeps
     *     default-path bytes byte-identical to pre-fix. */
    VkDeviceSize rgen_coverage = (VkDeviceSize)gpu->tiled_image_w
                               * gpu->tiled_image_h * 4u;
    if (gpu->direct_write_active && gpu->external_buffers_active &&
        gpu->interop_buf[gpu->interop_write_idx] &&
        gpu->interop_buf_size > rgen_coverage) {
        VkBuffer fill_buf = gpu->interop_buf[gpu->interop_write_idx];

        /* Barrier: ensure prior frame's CUDA reads (or any prior shader
         * writes) are complete before we overwrite. dstStage = TRANSFER
         * for the upcoming vkCmdFillBuffer. */
        VkBufferMemoryBarrier pre_fill = {0};
        pre_fill.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        pre_fill.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        pre_fill.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        pre_fill.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre_fill.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre_fill.buffer        = fill_buf;
        pre_fill.size          = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 1, &pre_fill, 0, NULL);

        /* Fill with opaque black: 0xFF000000 in little-endian byte order
         * matches RGBA8 (R=0,G=0,B=0,A=0xFF). */
        vkCmdFillBuffer(cmd, fill_buf, 0, VK_WHOLE_SIZE, 0xFF000000u);

        /* Barrier: transfer write -> shader write/read for the rgen. */
        VkBufferMemoryBarrier post_fill = {0};
        post_fill.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        post_fill.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        post_fill.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        post_fill.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        post_fill.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        post_fill.buffer        = fill_buf;
        post_fill.size          = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 0, NULL, 1, &post_fill, 0, NULL);
    }

    /* Dispatch at full tiled resolution.
     *
     * Phase C.4 mechanism hunt: wrap with GPU_PHASE_TRACE_RAYS_TILED so we
     * can discriminate the 4.13x speedup mechanism. Note: when the cmd-
     * buffer cache replays this dispatch, the recorded vkCmdWriteTimestamp
     * pair re-executes — the GPU writes new tick values — but
     * phase_pending is only set on cache-miss frames (where this code path
     * runs), so resolve only succeeds after a non-cached recording. The
     * resolved value still reflects per-frame GPU time once captured. */
    gpu_phase_begin(gpu, cmd, GPU_PHASE_TRACE_RAYS_TILED, "trace_rays_tiled");
    vkCmdTraceRaysKHR(cmd,
                      &gpu->tiled_sbt_rgen, &gpu->tiled_sbt_miss,
                      &gpu->tiled_sbt_hit, &gpu->tiled_sbt_call,
                      gpu->tiled_image_w, gpu->tiled_image_h, 1);
    gpu_phase_end(gpu, cmd, GPU_PHASE_TRACE_RAYS_TILED);

    /* Stash pc + state for cache key.  end_frame finalises validity. */
    gpu->tiled_cmd_cache_pc[parity]           = *pc;
    gpu->tiled_cmd_cache_direct_write[parity] = gpu->direct_write_active;
}

/* ---- Double-buffered async tiled readback ----
 *
 * Two staging buffers in round-robin.  gpu_end_frame_tiled_rt() submits the
 * RT dispatch + image-to-buffer copy with a VkFence and returns IMMEDIATELY
 * (non-blocking).  gpu_map_tiled_staging() waits on that fence before
 * returning the mapped pointer.
 *
 * Safety invariants:
 *   - A staging slot is never written by the GPU while the CPU is reading it.
 *   - Before reusing a slot we wait on its fence (from 2 frames ago at most).
 *   - Command buffers are kept alive until their fence signals.
 */

static void tiled_destroy_readback(Gpu* gpu)
{
    vkDeviceWaitIdle(gpu->device);
    for (int i = 0; i < 2; i++) {
        if (gpu->tiled_readback_cmd[i]) {
            vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1,
                                 &gpu->tiled_readback_cmd[i]);
            gpu->tiled_readback_cmd[i] = VK_NULL_HANDLE;
        }
        if (gpu->tiled_readback_fence[i]) {
            vkDestroyFence(gpu->device, gpu->tiled_readback_fence[i], NULL);
            gpu->tiled_readback_fence[i] = VK_NULL_HANDLE;
        }
        if (gpu->tiled_readback_buf[i]) {
            if (gpu->tiled_readback_mapped[i])
                vkUnmapMemory(gpu->device, gpu->tiled_readback_mem[i]);
            vkDestroyBuffer(gpu->device, gpu->tiled_readback_buf[i], NULL);
            vkFreeMemory(gpu->device, gpu->tiled_readback_mem[i], NULL);
            gpu->tiled_readback_buf[i]    = VK_NULL_HANDLE;
            gpu->tiled_readback_mem[i]    = VK_NULL_HANDLE;
            gpu->tiled_readback_mapped[i] = NULL;
        }
        gpu->tiled_readback_submitted[i] = 0;
    }
    gpu->tiled_readback_write_idx = 0;
    gpu->tiled_readback_w = 0;
    gpu->tiled_readback_h = 0;
}

static int tiled_ensure_readback(Gpu* gpu, uint32_t total_w, uint32_t total_h)
{
    /* Already allocated at the right size? */
    if (gpu->tiled_readback_buf[0] &&
        gpu->tiled_readback_w == total_w && gpu->tiled_readback_h == total_h)
        return 1;

    /* Cached tiled cmd buffer references tiled_readback_buf[0/1] in
     * vkCmdCopyImageToBuffer; recreating invalidates it. */
    gpu_invalidate_tiled_cmd_cache(gpu);

    /* Tear down old buffers (waits for any in-flight work) */
    tiled_destroy_readback(gpu);

    VkDeviceSize buf_size = (VkDeviceSize)total_w * total_h * 4;
    int is_coherent = 1;

    for (int i = 0; i < 2; i++) {
        /* Staging buffer */
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = buf_size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->tiled_readback_buf[i]) != VK_SUCCESS) {
            gpu->tiled_readback_buf[i] = VK_NULL_HANDLE;
            return 0;
        }

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->tiled_readback_buf[i], &req);

        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_readback_memory_type(gpu->physical_device,
            req.memoryTypeBits, &is_coherent);
        if (gpu_alloc_memory(gpu, &mai, &gpu->tiled_readback_mem[i]) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, gpu->tiled_readback_buf[i], NULL);
            gpu->tiled_readback_buf[i] = VK_NULL_HANDLE;
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->tiled_readback_buf[i],
                           gpu->tiled_readback_mem[i], 0);
        vkMapMemory(gpu->device, gpu->tiled_readback_mem[i], 0, buf_size, 0,
                    &gpu->tiled_readback_mapped[i]);

        /* Fence (created unsignaled) */
        VkFenceCreateInfo fci = {0};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(gpu->device, &fci, NULL, &gpu->tiled_readback_fence[i]) != VK_SUCCESS)
            return 0;
    }

    gpu->tiled_readback_w = total_w;
    gpu->tiled_readback_h = total_h;
    gpu->tiled_readback_coherent = is_coherent;
    gpu->tiled_readback_write_idx = 0;

    fprintf(stderr, "gpu_vulkan: tiled double-buffer readback created (%ux%u, %s)\n",
            total_w, total_h, is_coherent ? "coherent" : "cached");
    return 1;
}

/* ---- Depth output buffers (device SSBO + host staging) ---- */

static void tiled_destroy_depth(Gpu* gpu)
{
    if (gpu->tiled_depth_staging) {
        if (gpu->tiled_depth_staging_mapped)
            vkUnmapMemory(gpu->device, gpu->tiled_depth_staging_mem);
        vkDestroyBuffer(gpu->device, gpu->tiled_depth_staging, NULL);
        vkFreeMemory(gpu->device, gpu->tiled_depth_staging_mem, NULL);
        gpu->tiled_depth_staging        = VK_NULL_HANDLE;
        gpu->tiled_depth_staging_mem    = VK_NULL_HANDLE;
        gpu->tiled_depth_staging_mapped = NULL;
    }
    if (gpu->tiled_depth_buf) {
        vkDestroyBuffer(gpu->device, gpu->tiled_depth_buf, NULL);
        vkFreeMemory(gpu->device, gpu->tiled_depth_mem, NULL);
        gpu->tiled_depth_buf = VK_NULL_HANDLE;
        gpu->tiled_depth_mem = VK_NULL_HANDLE;
    }
    gpu->tiled_depth_enabled = 0;
    gpu->tiled_depth_w = 0;
    gpu->tiled_depth_h = 0;
}

static int tiled_ensure_depth(Gpu* gpu, uint32_t total_w, uint32_t total_h)
{
    /* Already allocated at the right size? */
    if (gpu->tiled_depth_buf && gpu->tiled_depth_enabled &&
        gpu->tiled_depth_w == total_w && gpu->tiled_depth_h == total_h)
        return 1;

    /* Cached tiled cmd buffer references the depth/staging buffers in
     * barriers and vkCmdCopyBuffer commands; recreating the buffer
     * invalidates the recording. */
    gpu_invalidate_tiled_cmd_cache(gpu);

    tiled_destroy_depth(gpu);

    VkDeviceSize buf_size = (VkDeviceSize)total_w * total_h * sizeof(float);

    /* Device-local SSBO for shader writes */
    {
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = buf_size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->tiled_depth_buf) != VK_SUCCESS)
            return 0;

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->tiled_depth_buf, &req);

        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (gpu_alloc_memory(gpu, &mai, &gpu->tiled_depth_mem) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, gpu->tiled_depth_buf, NULL);
            gpu->tiled_depth_buf = VK_NULL_HANDLE;
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->tiled_depth_buf, gpu->tiled_depth_mem, 0);
    }

    /* Host-visible staging for CPU readback */
    {
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = buf_size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->tiled_depth_staging) != VK_SUCCESS) {
            tiled_destroy_depth(gpu);
            return 0;
        }

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->tiled_depth_staging, &req);

        int is_coherent = 1;
        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_readback_memory_type(gpu->physical_device,
            req.memoryTypeBits, &is_coherent);
        if (gpu_alloc_memory(gpu, &mai, &gpu->tiled_depth_staging_mem) != VK_SUCCESS) {
            tiled_destroy_depth(gpu);
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->tiled_depth_staging,
                           gpu->tiled_depth_staging_mem, 0);
        vkMapMemory(gpu->device, gpu->tiled_depth_staging_mem, 0, buf_size, 0,
                    &gpu->tiled_depth_staging_mapped);
    }

    gpu->tiled_depth_enabled = 1;
    gpu->tiled_depth_w = total_w;
    gpu->tiled_depth_h = total_h;
    fprintf(stderr, "gpu_vulkan: tiled depth buffer created (%ux%u, %zu bytes)\n",
            total_w, total_h, (size_t)buf_size);
    return 1;
}

/* ---- Segmentation output buffers (device SSBO + host staging) ---- */

static void tiled_destroy_segmentation(Gpu* gpu)
{
    if (gpu->tiled_seg_staging) {
        if (gpu->tiled_seg_staging_mapped)
            vkUnmapMemory(gpu->device, gpu->tiled_seg_staging_mem);
        vkDestroyBuffer(gpu->device, gpu->tiled_seg_staging, NULL);
        vkFreeMemory(gpu->device, gpu->tiled_seg_staging_mem, NULL);
        gpu->tiled_seg_staging        = VK_NULL_HANDLE;
        gpu->tiled_seg_staging_mem    = VK_NULL_HANDLE;
        gpu->tiled_seg_staging_mapped = NULL;
    }
    if (gpu->tiled_seg_buf) {
        vkDestroyBuffer(gpu->device, gpu->tiled_seg_buf, NULL);
        vkFreeMemory(gpu->device, gpu->tiled_seg_mem, NULL);
        gpu->tiled_seg_buf = VK_NULL_HANDLE;
        gpu->tiled_seg_mem = VK_NULL_HANDLE;
    }
    gpu->tiled_seg_enabled = 0;
    gpu->tiled_seg_w = 0;
    gpu->tiled_seg_h = 0;
}

static int tiled_ensure_segmentation(Gpu* gpu, uint32_t total_w, uint32_t total_h)
{
    if (gpu->tiled_seg_buf && gpu->tiled_seg_enabled &&
        gpu->tiled_seg_w == total_w && gpu->tiled_seg_h == total_h)
        return 1;

    gpu_invalidate_tiled_cmd_cache(gpu);

    tiled_destroy_segmentation(gpu);

    VkDeviceSize buf_size = (VkDeviceSize)total_w * total_h * sizeof(uint32_t);

    /* Device-local SSBO for shader writes */
    {
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = buf_size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->tiled_seg_buf) != VK_SUCCESS)
            return 0;

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->tiled_seg_buf, &req);

        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (gpu_alloc_memory(gpu, &mai, &gpu->tiled_seg_mem) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, gpu->tiled_seg_buf, NULL);
            gpu->tiled_seg_buf = VK_NULL_HANDLE;
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->tiled_seg_buf, gpu->tiled_seg_mem, 0);
    }

    /* Host-visible staging for CPU readback */
    {
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = buf_size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->tiled_seg_staging) != VK_SUCCESS) {
            tiled_destroy_segmentation(gpu);
            return 0;
        }

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->tiled_seg_staging, &req);

        int is_coherent = 1;
        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_readback_memory_type(gpu->physical_device,
            req.memoryTypeBits, &is_coherent);
        if (gpu_alloc_memory(gpu, &mai, &gpu->tiled_seg_staging_mem) != VK_SUCCESS) {
            tiled_destroy_segmentation(gpu);
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->tiled_seg_staging,
                           gpu->tiled_seg_staging_mem, 0);
        vkMapMemory(gpu->device, gpu->tiled_seg_staging_mem, 0, buf_size, 0,
                    &gpu->tiled_seg_staging_mapped);
    }

    gpu->tiled_seg_enabled = 1;
    gpu->tiled_seg_w = total_w;
    gpu->tiled_seg_h = total_h;
    fprintf(stderr, "gpu_vulkan: tiled segmentation buffer created (%ux%u, %zu bytes)\n",
            total_w, total_h, (size_t)buf_size);
    return 1;
}

/* ---- Normals output buffers (device SSBO + host staging) ---- */

static void tiled_destroy_normals(Gpu* gpu)
{
    if (gpu->tiled_norm_staging) {
        if (gpu->tiled_norm_staging_mapped)
            vkUnmapMemory(gpu->device, gpu->tiled_norm_staging_mem);
        vkDestroyBuffer(gpu->device, gpu->tiled_norm_staging, NULL);
        vkFreeMemory(gpu->device, gpu->tiled_norm_staging_mem, NULL);
        gpu->tiled_norm_staging        = VK_NULL_HANDLE;
        gpu->tiled_norm_staging_mem    = VK_NULL_HANDLE;
        gpu->tiled_norm_staging_mapped = NULL;
    }
    if (gpu->tiled_norm_buf) {
        vkDestroyBuffer(gpu->device, gpu->tiled_norm_buf, NULL);
        vkFreeMemory(gpu->device, gpu->tiled_norm_mem, NULL);
        gpu->tiled_norm_buf = VK_NULL_HANDLE;
        gpu->tiled_norm_mem = VK_NULL_HANDLE;
    }
    gpu->tiled_norm_enabled = 0;
    gpu->tiled_norm_w = 0;
    gpu->tiled_norm_h = 0;
}

static int tiled_ensure_normals(Gpu* gpu, uint32_t total_w, uint32_t total_h)
{
    if (gpu->tiled_norm_buf && gpu->tiled_norm_enabled &&
        gpu->tiled_norm_w == total_w && gpu->tiled_norm_h == total_h)
        return 1;

    gpu_invalidate_tiled_cmd_cache(gpu);

    tiled_destroy_normals(gpu);

    /* 3 floats per pixel for (x, y, z) */
    VkDeviceSize buf_size = (VkDeviceSize)total_w * total_h * 3 * sizeof(float);

    {
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = buf_size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->tiled_norm_buf) != VK_SUCCESS)
            return 0;

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->tiled_norm_buf, &req);

        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (gpu_alloc_memory(gpu, &mai, &gpu->tiled_norm_mem) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, gpu->tiled_norm_buf, NULL);
            gpu->tiled_norm_buf = VK_NULL_HANDLE;
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->tiled_norm_buf, gpu->tiled_norm_mem, 0);
    }

    {
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = buf_size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->tiled_norm_staging) != VK_SUCCESS) {
            tiled_destroy_normals(gpu);
            return 0;
        }

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->tiled_norm_staging, &req);

        int is_coherent = 1;
        VkMemoryAllocateInfo mai = {0};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_readback_memory_type(gpu->physical_device,
            req.memoryTypeBits, &is_coherent);
        if (gpu_alloc_memory(gpu, &mai, &gpu->tiled_norm_staging_mem) != VK_SUCCESS) {
            tiled_destroy_normals(gpu);
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->tiled_norm_staging,
                           gpu->tiled_norm_staging_mem, 0);
        vkMapMemory(gpu->device, gpu->tiled_norm_staging_mem, 0, buf_size, 0,
                    &gpu->tiled_norm_staging_mapped);
    }

    gpu->tiled_norm_enabled = 1;
    gpu->tiled_norm_w = total_w;
    gpu->tiled_norm_h = total_h;
    fprintf(stderr, "gpu_vulkan: tiled normals buffer created (%ux%u, %zu bytes)\n",
            total_w, total_h, (size_t)buf_size);
    return 1;
}

void gpu_end_frame_tiled_rt(Gpu* gpu)
{
    VkCommandBuffer cmd = gpu->current_cmd;
    uint32_t total_w = gpu->tiled_image_w;
    uint32_t total_h = gpu->tiled_image_h;
    int parity = gpu->tiled_cmd_replay_idx & 1;

    /* Fast path: CUDA-only mode — skip staging buffer entirely.
     * Only copy tiled image → interop buffer, no CPU staging, no HOST barrier. */
    int interop_only = (gpu->skip_staging_copy && gpu->interop_buf[0]);

    if (!interop_only) {
        /* Ensure double-buffered staging exists at the right size */
        if (!tiled_ensure_readback(gpu, total_w, total_h))
            goto submit_only;
    } else {
        /* In interop-only mode, ensure fences exist for synchronization.
         * Staging buffers are not needed, but gpu_wait_tiled_complete() uses fences. */
        for (int i = 0; i < 2; i++) {
            if (!gpu->tiled_readback_fence[i]) {
                VkFenceCreateInfo fci = {0};
                fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                if (vkCreateFence(gpu->device, &fci, NULL, &gpu->tiled_readback_fence[i]) != VK_SUCCESS)
                    goto submit_only;
            }
        }
    }

    int idx = gpu->tiled_readback_write_idx;

    /* Wait for this slot's previous submission (from 2 frames ago) before
     * reusing the staging buffer.  If the fence was already waited on by
     * gpu_map_tiled_staging(), this returns immediately. */
    if (gpu->tiled_readback_submitted[idx]) {
        vkWaitForFences(gpu->device, 1, &gpu->tiled_readback_fence[idx],
                        VK_TRUE, UINT64_MAX);
        vkResetFences(gpu->device, 1, &gpu->tiled_readback_fence[idx]);
        /* Free the old command buffer that was kept alive for this slot,
         * but only if it isn't one of our persistent cached buffers. */
        if (gpu->tiled_readback_cmd[idx] &&
            gpu->tiled_readback_cmd[idx] != gpu->tiled_cached_cmd[0] &&
            gpu->tiled_readback_cmd[idx] != gpu->tiled_cached_cmd[1]) {
            vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1,
                                 &gpu->tiled_readback_cmd[idx]);
        }
        gpu->tiled_readback_cmd[idx] = VK_NULL_HANDLE;
        gpu->tiled_readback_submitted[idx] = 0;
    }

    /* --- Cache replay fast path -------------------------------------- */
    if (gpu->tiled_cmd_replay_active) {
        gpu->tiled_cmd_cache_replays++;

        /* The cmd buffer was never reset/begun this frame.  Just submit
         * the cached recording and signal the fence/timeline semaphore
         * the same way the recording path does. */
        VkCommandBuffer ccmd = gpu->tiled_cached_cmd[parity];

        uint64_t signal_value = 0;
        VkTimelineSemaphoreSubmitInfo timeline_si = {0};
        timeline_si.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;

        VkSubmitInfo si = {0};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &ccmd;

        if (gpu->interop_timeline_sem) {
            gpu->interop_timeline_value++;
            signal_value = gpu->interop_timeline_value;
            timeline_si.signalSemaphoreValueCount = 1;
            timeline_si.pSignalSemaphoreValues    = &signal_value;
            si.pNext                = &timeline_si;
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores    = &gpu->interop_timeline_sem;
        }

        vkQueueSubmit(gpu->graphics_queue, 1, &si, gpu->tiled_readback_fence[idx]);

        gpu->tiled_readback_cmd[idx]       = ccmd;
        gpu->tiled_readback_submitted[idx] = 1;
        gpu->tiled_readback_write_idx      = 1 - idx;
        gpu->interop_write_idx             = 1 - gpu->interop_write_idx;
        gpu->tiled_cmd_replay_active = 0;
        return;
    }

    /* Direct write mode: raygen shader writes to SSBO instead of imageStore.
     * No image→buffer copy needed. Only a buffer memory barrier for the SSBO. */
    int direct_write = gpu->direct_write_active;
    int interop_idx = gpu->interop_write_idx;

    if (direct_write) {
        /* Barrier: RT shader SSBO writes → bottom of pipe (for fence signaling) */
        VkMemoryBarrier mem_barrier = {0};
        mem_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mem_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mem_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 1, &mem_barrier, 0, NULL, 0, NULL);
    } else {
        /* Barrier: RT writes → transfer read */
        VkImageMemoryBarrier barrier = {0};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout                       = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout                       = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = gpu->tiled_image;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.layerCount     = 1;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 0, NULL, 1, &barrier);

        if (!interop_only) {
            /* Copy tiled image → staging[idx] (for CPU readback) */
            VkBufferImageCopy region = {0};
            region.bufferRowLength   = total_w;
            region.bufferImageHeight = total_h;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent.width  = total_w;
            region.imageExtent.height = total_h;
            region.imageExtent.depth  = 1;

            vkCmdCopyImageToBuffer(cmd,
                gpu->tiled_image, VK_IMAGE_LAYOUT_GENERAL,
                gpu->tiled_readback_buf[idx], 1, &region);
        }

        /* Copy tiled image → interop linear buffer (for CUDA zero-copy).
         * Double-buffered: write to interop_buf[interop_write_idx], CUDA reads
         * from interop_buf[1 - interop_write_idx] (the previous frame). */
        if (gpu->interop_buf[interop_idx]) {
            VkBufferImageCopy region = {0};
            region.bufferRowLength   = total_w;
            region.bufferImageHeight = total_h;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent.width  = total_w;
            region.imageExtent.height = total_h;
            region.imageExtent.depth  = 1;

            vkCmdCopyImageToBuffer(cmd,
                gpu->tiled_image, VK_IMAGE_LAYOUT_GENERAL,
                gpu->interop_buf[interop_idx], 1, &region);
        }

        if (!interop_only) {
            /* Barrier: transfer write → host read (makes staging visible to CPU) */
            VkMemoryBarrier mem_barrier = {0};
            mem_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mem_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mem_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                0, 1, &mem_barrier, 0, NULL, 0, NULL);
        }
    }

    /* Depth SSBO → staging copy (if depth output enabled) */
    if (gpu->tiled_depth_enabled && gpu->tiled_depth_buf && gpu->tiled_depth_staging) {
        VkDeviceSize depth_size = (VkDeviceSize)total_w * total_h * sizeof(float);

        /* Barrier: RT shader writes to depth SSBO → transfer read */
        VkBufferMemoryBarrier depth_barrier = {0};
        depth_barrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        depth_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        depth_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        depth_barrier.buffer        = gpu->tiled_depth_buf;
        depth_barrier.size          = depth_size;
        depth_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        depth_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 1, &depth_barrier, 0, NULL);

        /* Copy depth SSBO → staging */
        VkBufferCopy copy_region = {0};
        copy_region.size = depth_size;
        vkCmdCopyBuffer(cmd, gpu->tiled_depth_buf, gpu->tiled_depth_staging, 1, &copy_region);

        /* Barrier: transfer write → host read */
        VkBufferMemoryBarrier staging_barrier = {0};
        staging_barrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        staging_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        staging_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        staging_barrier.buffer        = gpu->tiled_depth_staging;
        staging_barrier.size          = depth_size;
        staging_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        staging_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, NULL, 1, &staging_barrier, 0, NULL);
    }

    /* Segmentation SSBO → staging copy (if segmentation output enabled) */
    if (gpu->tiled_seg_enabled && gpu->tiled_seg_buf && gpu->tiled_seg_staging) {
        VkDeviceSize seg_size = (VkDeviceSize)total_w * total_h * sizeof(uint32_t);

        VkBufferMemoryBarrier seg_barrier = {0};
        seg_barrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        seg_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        seg_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        seg_barrier.buffer        = gpu->tiled_seg_buf;
        seg_barrier.size          = seg_size;
        seg_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        seg_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 1, &seg_barrier, 0, NULL);

        VkBufferCopy copy_region = {0};
        copy_region.size = seg_size;
        vkCmdCopyBuffer(cmd, gpu->tiled_seg_buf, gpu->tiled_seg_staging, 1, &copy_region);

        VkBufferMemoryBarrier seg_staging_barrier = {0};
        seg_staging_barrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        seg_staging_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        seg_staging_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        seg_staging_barrier.buffer        = gpu->tiled_seg_staging;
        seg_staging_barrier.size          = seg_size;
        seg_staging_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        seg_staging_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, NULL, 1, &seg_staging_barrier, 0, NULL);
    }

    /* Normals SSBO → staging copy (if normals output enabled) */
    if (gpu->tiled_norm_enabled && gpu->tiled_norm_buf && gpu->tiled_norm_staging) {
        VkDeviceSize norm_size = (VkDeviceSize)total_w * total_h * 3 * sizeof(float);

        VkBufferMemoryBarrier norm_barrier = {0};
        norm_barrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        norm_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        norm_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        norm_barrier.buffer        = gpu->tiled_norm_buf;
        norm_barrier.size          = norm_size;
        norm_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        norm_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 1, &norm_barrier, 0, NULL);

        VkBufferCopy copy_region = {0};
        copy_region.size = norm_size;
        vkCmdCopyBuffer(cmd, gpu->tiled_norm_buf, gpu->tiled_norm_staging, 1, &copy_region);

        VkBufferMemoryBarrier norm_staging_barrier = {0};
        norm_staging_barrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        norm_staging_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        norm_staging_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        norm_staging_barrier.buffer        = gpu->tiled_norm_staging;
        norm_staging_barrier.size          = norm_size;
        norm_staging_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        norm_staging_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, NULL, 1, &norm_staging_barrier, 0, NULL);
    }

    vkEndCommandBuffer(cmd);
    gpu->tiled_cmd_cache_records++;

    /* Submit with fence — NON-BLOCKING.
     * If interop is available, also signal the timeline semaphore so CUDA
     * can wait on it before reading the tiled image. */
    {
        uint64_t signal_value = 0;
        VkTimelineSemaphoreSubmitInfo timeline_si = {0};
        timeline_si.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;

        VkSubmitInfo si = {0};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;

        if (gpu->interop_timeline_sem) {
            gpu->interop_timeline_value++;
            signal_value = gpu->interop_timeline_value;
            timeline_si.signalSemaphoreValueCount = 1;
            timeline_si.pSignalSemaphoreValues    = &signal_value;
            si.pNext                  = &timeline_si;
            si.signalSemaphoreCount   = 1;
            si.pSignalSemaphores      = &gpu->interop_timeline_sem;
        }

        vkQueueSubmit(gpu->graphics_queue, 1, &si, gpu->tiled_readback_fence[idx]);
    }

    /* Mark cache valid IFF this recording is replayable: it lives in our
     * persistent buffer for this parity AND no inline TLAS update was
     * recorded.  pc + direct_write were stashed by gpu_cmd_trace_rays_tiled.
     *
     * Note: invalidations triggered during this frame's recording (e.g.
     * tiled_ensure_readback creating buffers) set tiled_cmd_cache_dirty
     * but the recording itself is still consistent because the recording
     * happened AFTER those creations.  Clear dirty so the next frame
     * doesn't unnecessarily re-record. */
    int is_persistent = (cmd == gpu->tiled_cached_cmd[parity]);
    if (is_persistent && !gpu->tiled_cmd_has_tlas_update) {
        gpu->tiled_cmd_cache_valid[parity] = 1;
        gpu->tiled_cmd_cache_dirty         = 0;
    } else {
        gpu->tiled_cmd_cache_valid[parity] = 0;
    }

    /* Keep cmd alive until fence signals; record which slot is in flight */
    gpu->tiled_readback_cmd[idx]       = cmd;
    gpu->tiled_readback_submitted[idx] = 1;
    gpu->tiled_readback_write_idx      = 1 - idx;  /* flip for next frame */
    gpu->interop_write_idx             = 1 - interop_idx;  /* flip interop buffer */
    return;

submit_only:
    {
        /* If the cached cmd buffer is still in executable state from a
         * previous recording (deferred-Begin was set up by begin_frame
         * but no recording has happened this frame yet), there is no
         * vkEndCommandBuffer needed.  Detect that by checking whether
         * the buffer is the cached one AND the slot is still valid. */
        int is_cached_executable = gpu->tiled_cmd_cache_valid[parity]
                                 && cmd == gpu->tiled_cached_cmd[parity];
        if (!is_cached_executable)
            vkEndCommandBuffer(cmd);
        gpu->tiled_cmd_cache_records++;

        VkSubmitInfo si = {0};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;
        vkQueueSubmit(gpu->graphics_queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(gpu->graphics_queue);
    }
    /* submit_only is a fallback that never gets cached — even though the
     * buffer might be persistent, the slot's validity is left at 0. */
    if (cmd != gpu->tiled_cached_cmd[0] && cmd != gpu->tiled_cached_cmd[1]) {
        vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &cmd);
    }
    gpu->tiled_cmd_cache_valid[parity] = 0;
}

/* ---- CUDA Interop exports ---- */

int gpu_interop_available(Gpu* gpu)
{
    return gpu ? gpu->interop_available : 0;
}

void gpu_set_skip_staging_copy(Gpu* gpu, int skip)
{
    if (!gpu) return;
    int prev_skip = gpu->skip_staging_copy;
    int prev_direct = gpu->direct_write_active;
    gpu->skip_staging_copy = skip;
    /* Auto-enable direct write when interop buffer exists and staging is skipped.
     * This makes the raygen shader write to SSBO instead of imageStore,
     * eliminating the vkCmdCopyImageToBuffer step (~0.3ms at 2048 envs). */
    gpu->direct_write_active = (skip && gpu->interop_buf[0]) ? 1 : 0;
    if (gpu->direct_write_active)
        fprintf(stderr, "gpu_vulkan: direct buffer write enabled (SSBO binding 9)\n");

    /* Either toggle changes which barriers/copies the tiled cmd buffer
     * records, so any cached buffer is stale. */
    if (prev_skip != skip || prev_direct != gpu->direct_write_active)
        gpu_invalidate_tiled_cmd_cache(gpu);
}

void gpu_set_direct_write(Gpu* gpu, int enable)
{
    if (!gpu) return;
    if (gpu->direct_write_active != enable)
        gpu_invalidate_tiled_cmd_cache(gpu);
    gpu->direct_write_active = enable;
}

int gpu_is_direct_write(Gpu* gpu)
{
    return gpu ? gpu->direct_write_active : 0;
}

int gpu_export_tiled_image_fd(Gpu* gpu, int buf_idx)
{
    if (!gpu || !gpu->interop_available) return -1;
    if (buf_idx < 0 || buf_idx > 1) return -1;
    /* Hybrid C: memory is owned by the caller (imported), not exported. */
    if (gpu->external_buffers_active) return -1;
    if (!gpu->interop_buf_mem[buf_idx]) return -1;

    /* Export the linear interop buffer's memory (NOT the tiled image memory,
     * which uses VK_IMAGE_TILING_OPTIMAL and can't be read linearly by CUDA). */
    VkMemoryGetFdInfoKHR fd_info = {0};
    fd_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fd_info.memory     = gpu->interop_buf_mem[buf_idx];
    fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = -1;
    if (vkGetMemoryFdKHR(gpu->device, &fd_info, &fd) != VK_SUCCESS)
        return -1;
    return fd;
}

int gpu_set_external_output_buffers(Gpu* gpu,
                                    int mem_fds[2],
                                    uint64_t mem_size_each,
                                    uint32_t total_w,
                                    uint32_t total_h)
{
    if (!gpu) return 0;
    if (!gpu->interop_available) {
        fprintf(stderr, "gpu_vulkan: gpu_set_external_output_buffers: interop not available\n");
        return 0;
    }
    if (mem_fds[0] < 0 || mem_fds[1] < 0) {
        fprintf(stderr, "gpu_vulkan: gpu_set_external_output_buffers: invalid fds (%d, %d)\n",
                mem_fds[0], mem_fds[1]);
        return 0;
    }

    /* Stall until any pending GPU work referencing the existing
     * interop_buf[0/1] (which we are about to destroy) is fully drained.
     *
     * Why: this function is called once during inverted-interop init,
     * AFTER `nu_render_tiled` has already submitted a non-direct-write
     * frame whose cmd buffer's descriptor set still references the
     * internal interop_buf[0/1].  Without UPDATE_AFTER_BIND on binding 9,
     * `vkDestroyBuffer` + `vkUpdateDescriptorSets` while that submission
     * is still in flight is undefined behavior:
     *   - At NU_TILE_RES=64 the new VkBuffer happens to inherit the old
     *     slot's address, so subsequent shader writes still land somewhere
     *     valid, masking the bug.
     *   - At NU_TILE_RES != 64 the allocator gives different addresses
     *     and shader writes through the stale descriptor go nowhere
     *     (silent loss; rgen pixels never reach the imported buffer).
     *
     * vkDeviceWaitIdle runs ONCE at init, so the perf cost is irrelevant. */
    vkDeviceWaitIdle(gpu->device);

    VkDeviceSize logical = (VkDeviceSize)total_w * total_h * 4;
    if ((VkDeviceSize)mem_size_each < logical) {
        fprintf(stderr, "gpu_vulkan: external buffer too small (%llu < %llu logical)\n",
                (unsigned long long)mem_size_each, (unsigned long long)logical);
        return 0;
    }

    /* Tiled storage image must match dimensions before we wire descriptor binding 9. */
    if (gpu->tiled_image_w != total_w || gpu->tiled_image_h != total_h) {
        if (!tiled_create_storage_image(gpu, total_w, total_h)) {
            fprintf(stderr, "gpu_vulkan: tiled_create_storage_image failed for %ux%u\n",
                    total_w, total_h);
            return 0;
        }
    }
    tiled_create_interop_semaphore(gpu);

    /* Drop any existing interop buffers (allocated or imported). */
    gpu_invalidate_tiled_cmd_cache(gpu);
    for (int i = 0; i < 2; i++) {
        if (gpu->interop_buf[i]) {
            vkDestroyBuffer(gpu->device, gpu->interop_buf[i], NULL);
            gpu->interop_buf[i] = VK_NULL_HANDLE;
        }
        if (gpu->interop_buf_mem[i]) {
            vkFreeMemory(gpu->device, gpu->interop_buf_mem[i], NULL);
            gpu->interop_buf_mem[i] = VK_NULL_HANDLE;
        }
    }

    /* Create VkBuffer + import VkDeviceMemory for each fd. fds are consumed by
     * vkAllocateMemory on success — caller must NOT close them on success. */
    for (int i = 0; i < 2; i++) {
        VkExternalMemoryBufferCreateInfo ext_buf_ci = {0};
        ext_buf_ci.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ext_buf_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.pNext = &ext_buf_ci;
        bci.size  = (VkDeviceSize)mem_size_each;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->interop_buf[i]) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: gpu_set_external_output_buffers: vkCreateBuffer[%d] failed\n", i);
            goto fail;
        }

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->interop_buf[i], &req);
        if (req.size > (VkDeviceSize)mem_size_each) {
            fprintf(stderr,
                    "gpu_vulkan: external buffer mem_size_each=%llu < req.size=%llu (slot %d) — caller must round up to Vulkan req.alignment\n",
                    (unsigned long long)mem_size_each, (unsigned long long)req.size, i);
            goto fail;
        }

        /* VkMemoryDedicatedAllocateInfo: NVIDIA's CUDA-Vulkan interop docs
         * recommend dedicated allocations when importing CUDA-allocated memory
         * into a Vulkan buffer. Without it, shader accesses through the
         * descriptor table can land in the wrong physical page even though
         * the buffer's bound memory looks correct (vkCmdFillBuffer still
         * works, but shader writes go nowhere). */
        VkMemoryDedicatedAllocateInfo dedicated = {0};
        dedicated.sType  = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicated.buffer = gpu->interop_buf[i];

        VkImportMemoryFdInfoKHR import_info = {0};
        import_info.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        import_info.fd         = mem_fds[i];
        import_info.pNext      = &dedicated;

        VkMemoryAllocateInfo ai = {0};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.pNext           = &import_info;
        ai.allocationSize  = (VkDeviceSize)mem_size_each;
        ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(gpu->device, &ai, NULL, &gpu->interop_buf_mem[i]) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: gpu_set_external_output_buffers: vkAllocateMemory(import) slot %d failed\n", i);
            goto fail;
        }
        gpu->total_allocated_gpu_mem += (VkDeviceSize)mem_size_each;
        vkBindBufferMemory(gpu->device, gpu->interop_buf[i], gpu->interop_buf_mem[i], 0);
    }

    gpu->interop_buf_size        = (VkDeviceSize)mem_size_each;
    gpu->interop_write_idx       = 0;
    gpu->external_buffers_active = 1;

    /* Update binding 9 in any existing descriptor sets. New descriptor sets created
     * later pick up the imported buffers via the standard write path in gpu_tiled_init. */
    if (gpu->tiled_rt_desc_set) {
        VkDescriptorBufferInfo direct_info_a = {0};
        direct_info_a.buffer = gpu->interop_buf[0];
        direct_info_a.offset = 0;
        direct_info_a.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w_a = {0};
        w_a.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w_a.dstSet          = gpu->tiled_rt_desc_set;
        w_a.dstBinding      = 9;
        w_a.descriptorCount = 1;
        w_a.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w_a.pBufferInfo     = &direct_info_a;
        vkUpdateDescriptorSets(gpu->device, 1, &w_a, 0, NULL);
    }
    if (gpu->tiled_rt_desc_set_b) {
        VkDescriptorBufferInfo direct_info_b = {0};
        direct_info_b.buffer = gpu->interop_buf[1];
        direct_info_b.offset = 0;
        direct_info_b.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w_b = {0};
        w_b.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w_b.dstSet          = gpu->tiled_rt_desc_set_b;
        w_b.dstBinding      = 9;
        w_b.descriptorCount = 1;
        w_b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w_b.pBufferInfo     = &direct_info_b;
        vkUpdateDescriptorSets(gpu->device, 1, &w_b, 0, NULL);
    }

    fprintf(stderr,
            "gpu_vulkan: external output buffers imported (%ux%u, 2x %.1f MB, fds %d/%d)\n",
            total_w, total_h, (double)mem_size_each / (1024.0 * 1024.0),
            mem_fds[0], mem_fds[1]);
    return 1;

fail:
    for (int i = 0; i < 2; i++) {
        if (gpu->interop_buf[i]) {
            vkDestroyBuffer(gpu->device, gpu->interop_buf[i], NULL);
            gpu->interop_buf[i] = VK_NULL_HANDLE;
        }
        if (gpu->interop_buf_mem[i]) {
            vkFreeMemory(gpu->device, gpu->interop_buf_mem[i], NULL);
            gpu->interop_buf_mem[i] = VK_NULL_HANDLE;
        }
    }
    gpu->interop_buf_size        = 0;
    gpu->external_buffers_active = 0;
    return 0;
}

int gpu_get_interop_read_idx(Gpu* gpu)
{
    if (!gpu) return 0;
    /* The read buffer is the one NOT currently being written.
     * After gpu_end_frame_tiled_rt() flips interop_write_idx, the previous
     * frame's buffer is at (interop_write_idx ^ 1) = the old write_idx. */
    return 1 - gpu->interop_write_idx;
}

int gpu_export_timeline_semaphore_fd(Gpu* gpu)
{
    if (!gpu || !gpu->interop_timeline_sem) return -1;

    VkSemaphoreGetFdInfoKHR fd_info = {0};
    fd_info.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    fd_info.semaphore  = gpu->interop_timeline_sem;
    fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = -1;
    if (vkGetSemaphoreFdKHR(gpu->device, &fd_info, &fd) != VK_SUCCESS)
        return -1;
    return fd;
}

uint64_t gpu_get_tiled_image_alloc_size(Gpu* gpu)
{
    /* Return the interop buffer size (linear, CUDA-importable), not the
     * tiled image alloc size (optimal tiling, not directly usable by CUDA). */
    return gpu ? (uint64_t)gpu->interop_buf_size : 0;
}

uint64_t gpu_get_interop_timeline_value(Gpu* gpu)
{
    return gpu ? gpu->interop_timeline_value : 0;
}

int gpu_readback_tiled_pixels(Gpu* gpu, uint8_t* out_rgba8,
                               uint32_t total_w, uint32_t total_h)
{
    if (!gpu || !out_rgba8) return 0;
    if (gpu->tiled_readback_w != total_w || gpu->tiled_readback_h != total_h) return 0;

    /* The most recently submitted slot is at (write_idx ^ 1) */
    int read_idx = 1 - gpu->tiled_readback_write_idx;
    if (!gpu->tiled_readback_submitted[read_idx]) return 0;
    if (!gpu->tiled_readback_mapped[read_idx])    return 0;

    /* Wait for the GPU copy to complete (may already be signaled) */
    vkWaitForFences(gpu->device, 1, &gpu->tiled_readback_fence[read_idx],
                    VK_TRUE, UINT64_MAX);

    /* If non-coherent memory, invalidate CPU caches to see GPU writes */
    if (!gpu->tiled_readback_coherent) {
        VkMappedMemoryRange range = {0};
        range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = gpu->tiled_readback_mem[read_idx];
        range.offset = 0;
        range.size   = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(gpu->device, 1, &range);
    }

    VkDeviceSize buf_size = (VkDeviceSize)total_w * total_h * 4;
    memcpy(out_rgba8, gpu->tiled_readback_mapped[read_idx], (size_t)buf_size);

    return 1;
}

int gpu_readback_tiled_depth(Gpu* gpu, float* out_depth,
                              uint32_t total_w, uint32_t total_h)
{
    if (!gpu || !out_depth) return 0;
    if (!gpu->tiled_depth_enabled || !gpu->tiled_depth_staging_mapped) return 0;
    if (gpu->tiled_readback_w != total_w || gpu->tiled_readback_h != total_h) return 0;

    /* Wait on the same fence as color readback — depth copy is in the same cmd buffer */
    int read_idx = 1 - gpu->tiled_readback_write_idx;
    if (!gpu->tiled_readback_submitted[read_idx]) return 0;

    vkWaitForFences(gpu->device, 1, &gpu->tiled_readback_fence[read_idx],
                    VK_TRUE, UINT64_MAX);

    VkDeviceSize buf_size = (VkDeviceSize)total_w * total_h * sizeof(float);
    memcpy(out_depth, gpu->tiled_depth_staging_mapped, (size_t)buf_size);

    return 1;
}

int gpu_readback_tiled_segmentation(Gpu* gpu, uint32_t* out_ids,
                                      uint32_t total_w, uint32_t total_h)
{
    if (!gpu || !out_ids) return 0;
    if (!gpu->tiled_seg_enabled || !gpu->tiled_seg_staging_mapped) return 0;
    if (gpu->tiled_readback_w != total_w || gpu->tiled_readback_h != total_h) return 0;

    int read_idx = 1 - gpu->tiled_readback_write_idx;
    if (!gpu->tiled_readback_submitted[read_idx]) return 0;

    vkWaitForFences(gpu->device, 1, &gpu->tiled_readback_fence[read_idx],
                    VK_TRUE, UINT64_MAX);

    VkDeviceSize buf_size = (VkDeviceSize)total_w * total_h * sizeof(uint32_t);
    memcpy(out_ids, gpu->tiled_seg_staging_mapped, (size_t)buf_size);

    return 1;
}

int gpu_readback_tiled_normals(Gpu* gpu, float* out_normals,
                                uint32_t total_w, uint32_t total_h)
{
    if (!gpu || !out_normals) return 0;
    if (!gpu->tiled_norm_enabled || !gpu->tiled_norm_staging_mapped) return 0;
    if (gpu->tiled_readback_w != total_w || gpu->tiled_readback_h != total_h) return 0;

    int read_idx = 1 - gpu->tiled_readback_write_idx;
    if (!gpu->tiled_readback_submitted[read_idx]) return 0;

    vkWaitForFences(gpu->device, 1, &gpu->tiled_readback_fence[read_idx],
                    VK_TRUE, UINT64_MAX);

    VkDeviceSize buf_size = (VkDeviceSize)total_w * total_h * 3 * sizeof(float);
    memcpy(out_normals, gpu->tiled_norm_staging_mapped, (size_t)buf_size);

    return 1;
}

int gpu_wait_tiled_complete(Gpu* gpu)
{
    if (!gpu) return 0;
    /* Wait on the most recently submitted readback fence (current frame).
     * Both the staging copy and interop buffer copy are in the same command
     * buffer, so when this fence signals, the interop buffer is also ready. */
    int read_idx = 1 - gpu->tiled_readback_write_idx;
    if (!gpu->tiled_readback_submitted[read_idx]) return 0;
    vkWaitForFences(gpu->device, 1, &gpu->tiled_readback_fence[read_idx],
                    VK_TRUE, UINT64_MAX);
    /* Phase C.4 mechanism hunt: resolve tiled-RT GPU phase timestamps now
     * that the fence has signalled. No-ops when timestamps unsupported or
     * no pending writes on a phase. Mirrors the resolve points in
     * gpu_readback_pixels (line ~3465) and gpu_build_rt_scene. */
    if (gpu->timestamps_supported) {
        gpu_phase_resolve(gpu, GPU_PHASE_TRACE_RAYS_TILED);
        gpu_phase_resolve(gpu, GPU_PHASE_DEFERRED_COMPUTE);
    }
    return 1;
}

int gpu_wait_previous_tiled_complete(Gpu* gpu)
{
    if (!gpu) return 0;
    /* Wait on the PREVIOUS frame's fence (for double-buffered overlap).
     * After render N, tiled_readback_write_idx points to the slot for frame N+1.
     * The current slot (1-write_idx) has frame N's fence.
     * The write_idx slot has frame N-1's fence — that's what we want.
     * This fence typically signals faster since frame N-1 was submitted earlier. */
    int prev_idx = gpu->tiled_readback_write_idx;  /* the "other" slot = N-1 */
    if (!gpu->tiled_readback_submitted[prev_idx]) return 0;
    vkWaitForFences(gpu->device, 1, &gpu->tiled_readback_fence[prev_idx],
                    VK_TRUE, UINT64_MAX);
    return 1;
}

int gpu_get_interop_prev_idx(Gpu* gpu)
{
    if (!gpu) return 0;
    /* Return the interop buffer that was written in the PREVIOUS frame.
     * After render N flips interop_write_idx, the current write_idx points
     * to the slot for frame N+1. The value of interop_write_idx IS the slot
     * that was written in frame N-1 (since it was flipped from N-1's write). */
    return gpu->interop_write_idx;
}

const uint8_t* gpu_map_tiled_staging(Gpu* gpu, uint32_t total_w, uint32_t total_h)
{
    if (!gpu) return NULL;
    /* The most recently submitted slot is at (write_idx ^ 1) */
    return gpu_map_tiled_staging_slot(gpu, total_w, total_h,
                                      1 - gpu->tiled_readback_write_idx);
}

const uint8_t* gpu_map_tiled_staging_slot(Gpu* gpu, uint32_t total_w,
                                          uint32_t total_h, int slot)
{
    if (!gpu) return NULL;
    if (slot != 0 && slot != 1) return NULL;
    if (gpu->tiled_readback_w != total_w || gpu->tiled_readback_h != total_h) return NULL;
    if (!gpu->tiled_readback_submitted[slot]) return NULL;
    if (!gpu->tiled_readback_mapped[slot])    return NULL;

    /* Wait for the GPU copy to complete (may already be signaled) */
    vkWaitForFences(gpu->device, 1, &gpu->tiled_readback_fence[slot],
                    VK_TRUE, UINT64_MAX);

    /* If non-coherent memory, invalidate CPU caches to see GPU writes */
    if (!gpu->tiled_readback_coherent) {
        VkMappedMemoryRange range = {0};
        range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = gpu->tiled_readback_mem[slot];
        range.offset = 0;
        range.size   = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(gpu->device, 1, &range);
    }

    return (const uint8_t*)gpu->tiled_readback_mapped[slot];
}

int gpu_get_last_tiled_slot(Gpu* gpu)
{
    if (!gpu) return -1;
    /* After render_tiled submits to slot X it flips write_idx to 1-X, so the
     * slot that was just written is (1 - write_idx). */
    return 1 - gpu->tiled_readback_write_idx;
}

/* ============================================================================
 * PR 2: GPU-driven TLAS instance translation
 *
 * Eliminates the host-side `wp.array.numpy()` sync + per-instance memcpy
 * loop in nu_set_transforms / gpu_update_tlas_inline. Warp writes per-shape
 * 4x4 transforms directly into a CUDA-imported Vulkan storage buffer, then
 * a compute dispatch reads each transform and overwrites the upper-3x4
 * portion of the matching VkAccelerationStructureInstanceKHR record in
 * gpu->instance_buf — preserving the customIndex/mask/sbtOffset/flags/
 * AS-reference bytes that the initial host-side gpu_update_tlas() wrote.
 *
 * The shared transforms buffer is created via VK_EXTERNAL_MEMORY_HANDLE_
 * TYPE_OPAQUE_FD_BIT so CUDA can import it via cuImportExternalMemory.
 * The mapping (gid → tlas_instance_idx) is uploaded once at scene init
 * via gpu_set_transform_layout(); the dispatch is recorded inline into
 * the current frame's command buffer via gpu_translate_instances_inline().
 * ============================================================================ */

/* Lazily allocate the exportable transforms buffer + the tlas-indices SSBO
 * sized for `count` warp threads, and ensure the compute pipeline / DS exist.
 * Returns 1 on success. */
static int tlas_translate_ensure_pipeline(Gpu* gpu)
{
    if (gpu->tlas_translate_built) return 1;

    /* Descriptor set layout: 3 SSBOs (transforms in, instance buf out, tlas indices in) */
    VkDescriptorSetLayoutBinding bindings[3] = {{0}, {0}, {0}};
    for (int i = 0; i < 3; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
    dsl_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 3;
    dsl_ci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(gpu->device, &dsl_ci, NULL,
                                    &gpu->tlas_translate_ds_layout) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tlas_translate DS layout failed\n");
        return 0;
    }

    /* Descriptor pool + set */
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo pool_ci = {0};
    pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets       = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes    = &ps;
    if (vkCreateDescriptorPool(gpu->device, &pool_ci, NULL,
                               &gpu->tlas_translate_ds_pool) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tlas_translate DS pool failed\n");
        return 0;
    }

    VkDescriptorSetAllocateInfo dsa = {0};
    dsa.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool     = gpu->tlas_translate_ds_pool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts        = &gpu->tlas_translate_ds_layout;
    if (vkAllocateDescriptorSets(gpu->device, &dsa,
                                 &gpu->tlas_translate_ds_set) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tlas_translate DS alloc failed\n");
        return 0;
    }

    /* Pipeline layout (DS + push constant: uint count) */
    VkPushConstantRange pc = {0};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset     = 0;
    pc.size       = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pl_ci = {0};
    pl_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount         = 1;
    pl_ci.pSetLayouts            = &gpu->tlas_translate_ds_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges    = &pc;
    if (vkCreatePipelineLayout(gpu->device, &pl_ci, NULL,
                               &gpu->tlas_translate_pl_layout) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tlas_translate pipeline layout failed\n");
        return 0;
    }

    /* Load SPV + create compute pipeline */
    char shader_path[512];
    snprintf(shader_path, sizeof(shader_path), "%s/tlas_translate.comp.spv", SHADER_DIR);
    FILE* f = fopen(shader_path, "rb");
    if (!f) {
        fprintf(stderr, "gpu_vulkan: can't open %s\n", shader_path);
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* code = (uint32_t*)malloc((size_t)sz);
    if (!code) { fclose(f); return 0; }
    if (fread(code, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(code);
        fprintf(stderr, "gpu_vulkan: short read on %s\n", shader_path);
        return 0;
    }
    fclose(f);

    VkShaderModule mod = create_shader_module(gpu->device, code, (uint32_t)sz);
    free(code);
    if (!mod) {
        fprintf(stderr, "gpu_vulkan: tlas_translate create_shader_module failed\n");
        return 0;
    }

    VkComputePipelineCreateInfo cp = {0};
    cp.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = mod;
    cp.stage.pName  = "main";
    cp.layout       = gpu->tlas_translate_pl_layout;
    VkResult res = vkCreateComputePipelines(gpu->device, gpu->pipeline_cache,
                                             1, &cp, NULL,
                                             &gpu->tlas_translate_pipeline);
    vkDestroyShaderModule(gpu->device, mod, NULL);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tlas_translate vkCreateComputePipelines failed (%d)\n", res);
        return 0;
    }

    gpu->tlas_translate_built     = 1;
    gpu->tlas_translate_ds_dirty  = 1;  /* needs writes once buffers exist */
    return 1;
}

/* Refresh the descriptor set bindings.  Called whenever the underlying
 * buffer handles change (xforms buf grows; instance_buf rebuilt by accel
 * rebuild; indices buf grows). */
static void tlas_translate_update_descriptors(Gpu* gpu)
{
    if (!gpu->tlas_translate_built)        return;
    if (!gpu->tlas_xforms_buf)              return;
    if (!gpu->instance_buf)                 return;
    if (!gpu->tlas_indices_buf)             return;

    VkDescriptorBufferInfo bi[3] = {{0}, {0}, {0}};
    bi[0].buffer = gpu->tlas_xforms_buf;
    bi[0].offset = 0;
    bi[0].range  = VK_WHOLE_SIZE;
    bi[1].buffer = gpu->instance_buf;
    bi[1].offset = 0;
    bi[1].range  = VK_WHOLE_SIZE;
    bi[2].buffer = gpu->tlas_indices_buf;
    bi[2].offset = 0;
    bi[2].range  = VK_WHOLE_SIZE;

    VkWriteDescriptorSet ws[3] = {{0}, {0}, {0}};
    for (int i = 0; i < 3; i++) {
        ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet          = gpu->tlas_translate_ds_set;
        ws[i].dstBinding      = (uint32_t)i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[i].pBufferInfo     = &bi[i];
    }
    vkUpdateDescriptorSets(gpu->device, 3, ws, 0, NULL);

    gpu->tlas_translate_ds_dirty = 0;
}

/* Phase C: partitioned-translate compute pipeline. Mirrors
 * tlas_translate_ensure_pipeline but with 4 storage-buffer bindings:
 * 0=Transforms, 1=PartitionedInstanceBuf, 2=TlasIndices, 3=InstToPartial. */
static int tlas_translate_partitioned_ensure_pipeline(Gpu* gpu)
{
    if (gpu->tlas_translate_p_built) return 1;

    VkDescriptorSetLayoutBinding bindings[4] = {{0}};
    for (int i = 0; i < 4; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
    dsl_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 4;
    dsl_ci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(gpu->device, &dsl_ci, NULL,
                                    &gpu->tlas_translate_p_ds_layout) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tlas_translate_p DS layout failed\n");
        return 0;
    }

    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo pool_ci = {0};
    pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets       = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes    = &ps;
    if (vkCreateDescriptorPool(gpu->device, &pool_ci, NULL,
                               &gpu->tlas_translate_p_ds_pool) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tlas_translate_p DS pool failed\n");
        return 0;
    }

    VkDescriptorSetAllocateInfo dsa = {0};
    dsa.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool     = gpu->tlas_translate_p_ds_pool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts        = &gpu->tlas_translate_p_ds_layout;
    if (vkAllocateDescriptorSets(gpu->device, &dsa,
                                 &gpu->tlas_translate_p_ds_set) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tlas_translate_p DS alloc failed\n");
        return 0;
    }

    VkPushConstantRange pc = {0};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset     = 0;
    pc.size       = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pl_ci = {0};
    pl_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount         = 1;
    pl_ci.pSetLayouts            = &gpu->tlas_translate_p_ds_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges    = &pc;
    if (vkCreatePipelineLayout(gpu->device, &pl_ci, NULL,
                               &gpu->tlas_translate_p_pl_layout) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tlas_translate_p pipeline layout failed\n");
        return 0;
    }

    char shader_path[512];
    snprintf(shader_path, sizeof(shader_path), "%s/tlas_translate_partitioned.comp.spv", SHADER_DIR);
    FILE* f = fopen(shader_path, "rb");
    if (!f) { fprintf(stderr, "gpu_vulkan: can't open %s\n", shader_path); return 0; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* code = (uint32_t*)malloc((size_t)sz);
    if (!code) { fclose(f); return 0; }
    if (fread(code, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(code); return 0;
    }
    fclose(f);
    VkShaderModule mod = create_shader_module(gpu->device, code, (uint32_t)sz);
    free(code);
    if (!mod) return 0;

    VkComputePipelineCreateInfo cp = {0};
    cp.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = mod;
    cp.stage.pName  = "main";
    cp.layout       = gpu->tlas_translate_p_pl_layout;
    VkResult res = vkCreateComputePipelines(gpu->device, gpu->pipeline_cache,
                                             1, &cp, NULL,
                                             &gpu->tlas_translate_p_pipeline);
    vkDestroyShaderModule(gpu->device, mod, NULL);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tlas_translate_p pipeline create failed (%d)\n", res);
        return 0;
    }

    gpu->tlas_translate_p_built    = 1;
    gpu->tlas_translate_p_ds_dirty = 1;
    return 1;
}

static void tlas_translate_partitioned_update_descriptors(Gpu* gpu)
{
    if (!gpu->tlas_translate_p_built)  return;
    if (!gpu->tlas_xforms_buf)          return;
    if (!gpu->tlas_arr_inst_buf)        return;
    if (!gpu->tlas_indices_buf)         return;
    if (!gpu->inst_to_partial_buf)      return;

    VkDescriptorBufferInfo bi[4] = {{0}};
    bi[0].buffer = gpu->tlas_xforms_buf;     bi[0].range = VK_WHOLE_SIZE;
    bi[1].buffer = gpu->tlas_arr_inst_buf;   bi[1].range = VK_WHOLE_SIZE;
    bi[2].buffer = gpu->tlas_indices_buf;    bi[2].range = VK_WHOLE_SIZE;
    bi[3].buffer = gpu->inst_to_partial_buf; bi[3].range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet ws[4] = {{0}};
    for (int i = 0; i < 4; i++) {
        ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet          = gpu->tlas_translate_p_ds_set;
        ws[i].dstBinding      = (uint32_t)i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[i].pBufferInfo     = &bi[i];
    }
    vkUpdateDescriptorSets(gpu->device, 4, ws, 0, NULL);
    gpu->tlas_translate_p_ds_dirty = 0;
}

/* Public: ensure the exportable per-shape transforms buffer exists with at
 * least `count * 64` bytes.  Returns 1 on success. The Vulkan buffer is
 * reused while `count` is unchanged; each gpu_export_tlas_xforms_fd() call
 * returns a fresh caller-owned one-shot fd for the same allocation.
 *
 * Recreating the buffer (e.g. count grew) invalidates the pipeline DS,
 * which is re-written by the next dispatch / set_layout call. */
int gpu_create_tlas_xforms_buffer(Gpu* gpu, int count)
{
    if (!gpu || count <= 0) return 0;
    if (!gpu->interop_available) {
        fprintf(stderr, "gpu_vulkan: tlas_xforms requires CUDA interop (external memory) "
                        "extensions, which were not enabled at device create.\n");
        return 0;
    }

    VkDeviceSize logical = (VkDeviceSize)count * 16ULL * sizeof(float);  /* count * 64 */

    if (gpu->tlas_xforms_buf && gpu->tlas_xforms_count == count)
        return 1;  /* already sized */

    /* Free any previous allocation. Any exported fd was caller-owned and
     * consumed/closed by the interop client. */
    gpu->tlas_xforms_fd = -1;
    if (gpu->tlas_xforms_buf) {
        vkDestroyBuffer(gpu->device, gpu->tlas_xforms_buf, NULL);
        gpu->tlas_xforms_buf = VK_NULL_HANDLE;
    }
    if (gpu->tlas_xforms_mem) {
        vkFreeMemory(gpu->device, gpu->tlas_xforms_mem, NULL);
        gpu->tlas_xforms_mem = VK_NULL_HANDLE;
    }

    VkExternalMemoryBufferCreateInfo ext_buf_ci = {0};
    ext_buf_ci.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext_buf_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext = &ext_buf_ci;
    bci.size  = logical;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
              | VK_BUFFER_USAGE_TRANSFER_DST_BIT
              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->tlas_xforms_buf) != VK_SUCCESS) {
        fprintf(stderr, "gpu_vulkan: tlas_xforms vkCreateBuffer failed\n");
        return 0;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gpu->device, gpu->tlas_xforms_buf, &req);

    VkExportMemoryAllocateInfo export_mai = {0};
    export_mai.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_mai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo ai = {0};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext           = &export_mai;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (gpu_alloc_memory(gpu, &ai, &gpu->tlas_xforms_mem) != VK_SUCCESS) {
        vkDestroyBuffer(gpu->device, gpu->tlas_xforms_buf, NULL);
        gpu->tlas_xforms_buf = VK_NULL_HANDLE;
        fprintf(stderr, "gpu_vulkan: tlas_xforms vkAllocateMemory failed\n");
        return 0;
    }
    vkBindBufferMemory(gpu->device, gpu->tlas_xforms_buf, gpu->tlas_xforms_mem, 0);

    gpu->tlas_xforms_size    = req.size;
    gpu->tlas_xforms_logical = logical;
    gpu->tlas_xforms_count   = count;
    gpu->tlas_translate_ds_dirty = 1;

    fprintf(stderr, "gpu_vulkan: tlas_xforms buffer created (%d transforms, %.2f MB)\n",
            count, (double)logical / (1024.0 * 1024.0));
    return 1;
}

/* Public: export the transforms buffer's VkDeviceMemory as a POSIX fd.
 * Each call returns a fresh one-shot fd. The caller must either pass it to
 * cuImportExternalMemory (which consumes it) or close it. */
int gpu_export_tlas_xforms_fd(Gpu* gpu)
{
    if (!gpu || !gpu->tlas_xforms_mem) return -1;

    VkMemoryGetFdInfoKHR fd_info = {0};
    fd_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fd_info.memory     = gpu->tlas_xforms_mem;
    fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = -1;
    if (vkGetMemoryFdKHR(gpu->device, &fd_info, &fd) != VK_SUCCESS)
        return -1;
    return fd;
}

uint64_t gpu_get_tlas_xforms_size(Gpu* gpu)
{
    /* Return the LOGICAL size (count*64) which is what CUDA needs for
     * cuExternalMemoryGetMappedBuffer.  The VkDeviceMemory allocation
     * may be larger (req.size), but the meaningful payload is `logical`. */
    return gpu ? (uint64_t)gpu->tlas_xforms_logical : 0;
}

/* Public: upload the (gid → tlas_instance_idx) map. Called once at scene
 * init by Python after the renderer-mesh-id → tlas-idx inversion is
 * complete.  Returns 1 on success. */
int gpu_set_transform_layout(Gpu* gpu, const int* tlas_indices, int count)
{
    if (!gpu || !tlas_indices || count <= 0) return 0;

    VkDeviceSize size = (VkDeviceSize)count * sizeof(int);

    /* Recreate the indices buffer if size changed. */
    if (gpu->tlas_indices_buf && gpu->tlas_indices_count != count) {
        vkDestroyBuffer(gpu->device, gpu->tlas_indices_buf, NULL);
        vkFreeMemory(gpu->device, gpu->tlas_indices_mem, NULL);
        gpu->tlas_indices_buf = VK_NULL_HANDLE;
        gpu->tlas_indices_mem = VK_NULL_HANDLE;
    }

    if (!gpu->tlas_indices_buf) {
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &gpu->tlas_indices_buf) != VK_SUCCESS) {
            fprintf(stderr, "gpu_vulkan: tlas_indices vkCreateBuffer failed\n");
            return 0;
        }
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, gpu->tlas_indices_buf, &req);

        VkMemoryAllocateInfo ai = {0};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (gpu_alloc_memory(gpu, &ai, &gpu->tlas_indices_mem) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, gpu->tlas_indices_buf, NULL);
            gpu->tlas_indices_buf = VK_NULL_HANDLE;
            return 0;
        }
        vkBindBufferMemory(gpu->device, gpu->tlas_indices_buf, gpu->tlas_indices_mem, 0);
        gpu->tlas_indices_count = count;
        gpu->tlas_translate_ds_dirty = 1;
    }

    /* Upload via host-visible staging buffer (one-shot). */
    VkBuffer staging_buf = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    {
        VkBufferCreateInfo bci = {0};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(gpu->device, &bci, NULL, &staging_buf) != VK_SUCCESS) return 0;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(gpu->device, staging_buf, &req);
        VkMemoryAllocateInfo ai = {0};
        ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = find_memory_type(gpu->physical_device, req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                              | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (gpu_alloc_memory(gpu, &ai, &staging_mem) != VK_SUCCESS) {
            vkDestroyBuffer(gpu->device, staging_buf, NULL);
            return 0;
        }
        vkBindBufferMemory(gpu->device, staging_buf, staging_mem, 0);
        void* mapped = NULL;
        vkMapMemory(gpu->device, staging_mem, 0, size, 0, &mapped);
        memcpy(mapped, tlas_indices, size);
        vkUnmapMemory(gpu->device, staging_mem);
    }

    /* Copy staging → device-local */
    VkCommandBuffer cmd = rt_begin_cmd(gpu);
    VkBufferCopy copy = {0};
    copy.size = size;
    vkCmdCopyBuffer(cmd, staging_buf, gpu->tlas_indices_buf, 1, &copy);
    rt_end_cmd(gpu, cmd);  /* synchronous submit + wait */

    vkDestroyBuffer(gpu->device, staging_buf, NULL);
    vkFreeMemory(gpu->device, staging_mem, NULL);

    /* Refresh DS now that buffers exist. */
    if (tlas_translate_ensure_pipeline(gpu)) {
        if (gpu->tlas_translate_ds_dirty)
            tlas_translate_update_descriptors(gpu);
    }
    return 1;
}

/* Public: record the compute dispatch + barrier into the CURRENT command
 * buffer (gpu->current_cmd) and signal the cmd-buffer cache that the
 * current frame must not be cached for replay (since it now embeds an
 * inline TLAS update with fresh transforms).
 *
 * The caller is responsible for ensuring the warp kernel that wrote into
 * the imported transforms buffer has completed before this dispatch is
 * submitted. With a single CUDA stream + Vulkan queue this happens via
 * Python `wp.synchronize_device()` before nu_translate_instances_gpu().
 *
 * Returns 1 on success, 0 if the renderer is not in a state where the
 * dispatch can be recorded (e.g. no current cmd, no TLAS, missing buffers). */
int gpu_translate_instances_inline(Gpu* gpu, int count)
{
    if (!gpu || count <= 0)                          return 0;
    if (!gpu->rt_built || !gpu->tlas_instance_count) return 0;
    if (!gpu->current_cmd)                           return 0;
    if (!gpu->instance_buf)                          return 0;
    if (!gpu->tlas_xforms_buf)                       return 0;
    if (!gpu->tlas_indices_buf)                      return 0;

    if (!tlas_translate_ensure_pipeline(gpu)) return 0;
    if (gpu->tlas_translate_ds_dirty)
        tlas_translate_update_descriptors(gpu);

    /* DEBUG: opt-in env var to skip the compute dispatch (reuses last
     * frame's instance_buf contents). Useful for isolating compute-vs-
     * AS-build issues during development. */
    int skip_compute = (getenv("NUSD_TLAS_SKIP_COMPUTE") != NULL);

    /* Same cmd-cache discipline as gpu_update_tlas_inline: invalidate so
     * the recording in progress does not get stashed for replay. */
    gpu_invalidate_rt_cmd_cache(gpu);

    int parity = gpu->tiled_cmd_replay_idx & 1;
    if (gpu->current_cmd == gpu->tiled_cached_cmd[parity] &&
        gpu->tiled_cmd_cache_valid[parity]) {
        gpu->tiled_cmd_cache_valid[parity] = 0;
        VkCommandBuffer ccmd = gpu->tiled_cached_cmd[parity];
        vkResetCommandBuffer(ccmd, 0);
        VkCommandBufferBeginInfo begin = {0};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        if (vkBeginCommandBuffer(ccmd, &begin) != VK_SUCCESS) return 0;
    }
    gpu_invalidate_tiled_cmd_cache(gpu);
    gpu->tiled_cmd_has_tlas_update = 1;

    VkCommandBuffer cmd = gpu->current_cmd;

    if (!skip_compute) {
        /* Bind compute pipeline + DS, push constants, dispatch. */
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          gpu->tlas_translate_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                gpu->tlas_translate_pl_layout,
                                0, 1, &gpu->tlas_translate_ds_set, 0, NULL);
        uint32_t n = (uint32_t)count;
        vkCmdPushConstants(cmd, gpu->tlas_translate_pl_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &n);
        /* local_size_x = NUSD_TLAS_TRANSLATE_LOCAL_X (default 64) */
        uint32_t groups = (n + 63u) / 64u;
        vkCmdDispatch(cmd, groups, 1, 1);

        /* Phase C: dispatch the partitioned-translate compute too if a
         * per-env TLAS array is built. Writes the same transforms into
         * gpu->tlas_arr_inst_buf at partitioned-order slots. Both
         * dispatches' writes are flushed by the single compute→AS_BUILD
         * barrier below. */
        if (gpu->tlas_arr_built && gpu->tlas_arr_inst_buf) {
            if (tlas_translate_partitioned_ensure_pipeline(gpu)) {
                if (gpu->tlas_translate_p_ds_dirty)
                    tlas_translate_partitioned_update_descriptors(gpu);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  gpu->tlas_translate_p_pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        gpu->tlas_translate_p_pl_layout,
                                        0, 1, &gpu->tlas_translate_p_ds_set, 0, NULL);
                vkCmdPushConstants(cmd, gpu->tlas_translate_p_pl_layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &n);
                vkCmdDispatch(cmd, groups, 1, 1);
            }
        }

        /* Compute write → AS-build read barrier. NVIDIA's AS-build reads
         * the instance buffer via deviceAddress (bypassing the descriptor
         * binding), so the barrier needs broad coverage. Use a global
         * memory barrier with ALL_COMMANDS source/dest to ensure all
         * compute-stage writes are flushed before the AS-build reads.
         * This is more conservative than the strict COMPUTE→AS_BUILD
         * barrier but works around what appears to be a sync issue with
         * how some drivers handle compute-write → AS_BUILD-read on
         * STORAGE_BUFFER + ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
         * dual-use buffers. */
        VkMemoryBarrier mb_compute = {0};
        mb_compute.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb_compute.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT
                                 | VK_ACCESS_MEMORY_WRITE_BIT;
        mb_compute.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
                                 | VK_ACCESS_SHADER_READ_BIT
                                 | VK_ACCESS_MEMORY_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, &mb_compute, 0, NULL, 0, NULL);
    } else {
        fprintf(stderr, "[tlas_translate] SKIP_COMPUTE active — using existing instance_buf contents\n");
    }

    /* Now record the TLAS update + build, identical to gpu_update_tlas_inline
     * minus the host-side instance-record write (the compute shader did it).
     * The persistent scratch buffer is still required. */
    VkBuffer scratch_buf = gpu->tlas_update_scratch_buf;
    if (!scratch_buf) return 0;

    /* Phase C optimization: when the partitioned TLAS array is active, the
     * tile rgen shader traces against tlas_arr[cam_env_idx] and never
     * touches the legacy single TLAS — so skip the legacy refit (saves
     * ~1 ms / call at 4096 envs). The legacy TLAS still gets refitted for
     * non-partitioned consumers (visualizer, single-cam RT, raster) when
     * NUSD_LEGACY_TLAS_REFIT_ALWAYS=1, and always when num_envs == 0. */
    int partition_active = (gpu->tlas_arr_built && gpu->num_envs > 0);
    int force_legacy_refit = (getenv("NUSD_LEGACY_TLAS_REFIT_ALWAYS") != NULL);

    if (!partition_active || force_legacy_refit) {
        VkAccelerationStructureGeometryInstancesDataKHR instances_data = {0};
        instances_data.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        instances_data.data.deviceAddress = rt_buf_addr(gpu, gpu->instance_buf);

        VkAccelerationStructureGeometryKHR tlas_geom = {0};
        tlas_geom.sType                = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        tlas_geom.geometryType         = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlas_geom.geometry.instances   = instances_data;

        VkAccelerationStructureBuildGeometryInfoKHR tlas_build = {0};
        tlas_build.sType                      = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        tlas_build.type                       = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tlas_build.flags                      = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                              | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        tlas_build.mode                       = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        tlas_build.srcAccelerationStructure   = gpu->tlas;
        tlas_build.dstAccelerationStructure   = gpu->tlas;
        tlas_build.geometryCount              = 1;
        tlas_build.pGeometries                = &tlas_geom;
        tlas_build.scratchData.deviceAddress  = rt_buf_addr(gpu, scratch_buf);

        VkAccelerationStructureBuildRangeInfoKHR tlas_range = {0};
        tlas_range.primitiveCount = gpu->tlas_instance_count;
        const VkAccelerationStructureBuildRangeInfoKHR* pRange = &tlas_range;

        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlas_build, &pRange);
    }

    /* Phase C: refit all per-env TLASes after the legacy refit. Each env
     * uses the same shared scratch buffer (`gpu->tlas_arr_scratch_buf`),
     * sub-allocated by offset like at build time. */
    if (gpu->tlas_arr_built && gpu->tlas_arr && gpu->tlas_arr_scratch_buf
        && gpu->num_envs > 0 && gpu->env_inst_idx) {
        int num_envs = gpu->num_envs;
        VkAccelerationStructureBuildSizesInfoKHR* sz_info =
            (VkAccelerationStructureBuildSizesInfoKHR*)calloc((size_t)num_envs, sizeof(*sz_info));
        VkAccelerationStructureBuildGeometryInfoKHR* binfos =
            (VkAccelerationStructureBuildGeometryInfoKHR*)calloc((size_t)num_envs, sizeof(*binfos));
        VkAccelerationStructureGeometryKHR* gms =
            (VkAccelerationStructureGeometryKHR*)calloc((size_t)num_envs, sizeof(*gms));
        VkAccelerationStructureBuildRangeInfoKHR* rngs =
            (VkAccelerationStructureBuildRangeInfoKHR*)calloc((size_t)num_envs, sizeof(*rngs));
        const VkAccelerationStructureBuildRangeInfoKHR** rng_ptrs =
            (const VkAccelerationStructureBuildRangeInfoKHR**)calloc((size_t)num_envs, sizeof(void*));
        if (sz_info && binfos && gms && rngs && rng_ptrs) {
            VkDeviceAddress inst_addr   = rt_buf_addr(gpu, gpu->tlas_arr_inst_buf);
            VkDeviceAddress scratch_addr= rt_buf_addr(gpu, gpu->tlas_arr_scratch_buf);
            VkDeviceSize scratch_align  = gpu->min_as_scratch_align ? gpu->min_as_scratch_align : 256;
            VkDeviceSize scratch_off    = 0;

            for (int e = 0; e < num_envs; e++) {
                uint32_t env_count_e = gpu->env_inst_idx[e+1] - gpu->env_inst_idx[e];

                sz_info[e].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

                gms[e].sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                gms[e].geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
                gms[e].geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
                gms[e].geometry.instances.data.deviceAddress =
                    inst_addr + (VkDeviceSize)gpu->env_inst_idx[e] * sizeof(VkAccelerationStructureInstanceKHR);

                binfos[e].sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
                binfos[e].type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
                binfos[e].flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                        | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
                binfos[e].mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
                binfos[e].srcAccelerationStructure = gpu->tlas_arr[e];
                binfos[e].dstAccelerationStructure = gpu->tlas_arr[e];
                binfos[e].geometryCount = 1;
                binfos[e].pGeometries   = &gms[e];

                vkGetAccelerationStructureBuildSizesKHR(gpu->device,
                                        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                        &binfos[e], &env_count_e, &sz_info[e]);

                binfos[e].scratchData.deviceAddress = scratch_addr + scratch_off;
                /* Use buildScratchSize for the slot allocation (it's the
                 * larger of build/update on every implementation, so the
                 * existing scratch buffer is always sized for it). */
                VkDeviceSize ssz = sz_info[e].buildScratchSize;
                ssz = (ssz + scratch_align - 1) & ~(scratch_align - 1);
                scratch_off += ssz;

                rngs[e].primitiveCount = env_count_e;
                rng_ptrs[e] = &rngs[e];
            }

            vkCmdBuildAccelerationStructuresKHR(cmd, (uint32_t)num_envs, binfos, rng_ptrs);
        }
        free(sz_info); free(binfos); free(gms); free(rngs); free(rng_ptrs);
    }

    /* Barrier: AS-build write → ray-trace read (matches gpu_cmd_trace_rays_tiled prologue). */
    VkMemoryBarrier mb_after = {0};
    mb_after.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb_after.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    mb_after.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0, 1, &mb_after, 0, NULL, 0, NULL);

    return 1;
}

/* Build (mesh_id → tlas_instance_idx) by inverting gpu->instance_custom[].
 * Slots not present in the TLAS are written as -1. Returns 1 on success. */
int gpu_get_mesh_to_tlas_idx(Gpu* gpu, int* out_mesh_to_tlas_idx, int nmeshes)
{
    if (!gpu || !out_mesh_to_tlas_idx || nmeshes <= 0) return 0;
    if (!gpu->instance_custom || gpu->tlas_instance_count == 0) return 0;

    for (int i = 0; i < nmeshes; i++) out_mesh_to_tlas_idx[i] = -1;
    for (uint32_t i = 0; i < gpu->tlas_instance_count; i++) {
        uint32_t mid = gpu->instance_custom[i];
        if ((int)mid < nmeshes)
            out_mesh_to_tlas_idx[mid] = (int)i;
    }
    return 1;
}

/* ============================================================================
 * VK3DGRT — 3D Gaussian Ray Tracing GPU integration.
 *
 * Splat-scene state lives parallel to the mesh BLAS/TLAS path; everything is
 * gated behind the `gs_*` Gpu fields. See docs/plans/VK3DGRT_PLAN.md and
 * src/gs_scene.c for the host-side architecture.
 *
 * Phase 2 is partially landed:
 *   - Slang shaders (gs_3dgrt.{rgen,rahit,rint,rmiss}, gs_as_build.comp)
 *     compile to SPIR-V via cmake/SlangShaders.cmake.
 *   - Public C API + opaque state are wired (this section).
 *   - Particle-SSBO upload + unit-BLAS + TLAS-via-compute build are still
 *     TODO — the function bodies below mirror the existing patterns in
 *     gpu_build_rt_scene / rt_create_buffer_pooled / rt_buf_addr but stop
 *     short of recording the BLAS/TLAS build commands. They will fill in
 *     incrementally (see VK3DGRT_PLAN.md §5/§6 for shape).
 *
 * Phase 3 (RT pipeline + per-tile SoA K-buffer) is not yet wired.
 * ============================================================================ */

void gpu_gs_destroy(Gpu* gpu)
{
    if (!gpu) return;

    /* RT pipeline state (Phase 3). */
    if (gpu->gs_kbuf_id_buf)    { vkDestroyBuffer(gpu->device, gpu->gs_kbuf_id_buf, NULL);    gpu->gs_kbuf_id_buf = VK_NULL_HANDLE; }
    if (gpu->gs_kbuf_id_mem)    { vkFreeMemory(gpu->device, gpu->gs_kbuf_id_mem, NULL);        gpu->gs_kbuf_id_mem = VK_NULL_HANDLE; }
    if (gpu->gs_kbuf_dist_buf)  { vkDestroyBuffer(gpu->device, gpu->gs_kbuf_dist_buf, NULL);  gpu->gs_kbuf_dist_buf = VK_NULL_HANDLE; }
    if (gpu->gs_kbuf_dist_mem)  { vkFreeMemory(gpu->device, gpu->gs_kbuf_dist_mem, NULL);      gpu->gs_kbuf_dist_mem = VK_NULL_HANDLE; }
    gpu->gs_kbuf_w = gpu->gs_kbuf_h = gpu->gs_kbuf_k = 0;
    if (gpu->gs_depth_buf)      { vkDestroyBuffer(gpu->device, gpu->gs_depth_buf, NULL);      gpu->gs_depth_buf = VK_NULL_HANDLE; }
    if (gpu->gs_depth_mem)      { vkFreeMemory(gpu->device, gpu->gs_depth_mem, NULL);          gpu->gs_depth_mem = VK_NULL_HANDLE; }
    gpu->gs_depth_w = gpu->gs_depth_h = 0;
    if (gpu->gs_normal_buf)     { vkDestroyBuffer(gpu->device, gpu->gs_normal_buf, NULL);     gpu->gs_normal_buf = VK_NULL_HANDLE; }
    if (gpu->gs_normal_mem)     { vkFreeMemory(gpu->device, gpu->gs_normal_mem, NULL);         gpu->gs_normal_mem = VK_NULL_HANDLE; }
    gpu->gs_normal_w = gpu->gs_normal_h = 0;
    if (gpu->gs_sbt_buf)        { vkDestroyBuffer(gpu->device, gpu->gs_sbt_buf, NULL);        gpu->gs_sbt_buf = VK_NULL_HANDLE; }
    if (gpu->gs_sbt_mem)        { vkFreeMemory(gpu->device, gpu->gs_sbt_mem, NULL);            gpu->gs_sbt_mem = VK_NULL_HANDLE; }
    if (gpu->gs_rt_pipeline)    { vkDestroyPipeline(gpu->device, gpu->gs_rt_pipeline, NULL);   gpu->gs_rt_pipeline = VK_NULL_HANDLE; }
    if (gpu->gs_rt_pl_layout)   { vkDestroyPipelineLayout(gpu->device, gpu->gs_rt_pl_layout, NULL); gpu->gs_rt_pl_layout = VK_NULL_HANDLE; }
    if (gpu->gs_rt_ds_layout)   { vkDestroyDescriptorSetLayout(gpu->device, gpu->gs_rt_ds_layout, NULL); gpu->gs_rt_ds_layout = VK_NULL_HANDLE; }
    if (gpu->gs_rt_ds_pool)     { vkDestroyDescriptorPool(gpu->device, gpu->gs_rt_ds_pool, NULL); gpu->gs_rt_ds_pool = VK_NULL_HANDLE; }
    gpu->gs_rt_ds = VK_NULL_HANDLE;
    gpu->gs_pipeline_built = 0;

    /* Compute pipeline (gs_as_build). */
    if (gpu->gs_as_pipeline)  { vkDestroyPipeline(gpu->device, gpu->gs_as_pipeline, NULL);   gpu->gs_as_pipeline = VK_NULL_HANDLE; }
    if (gpu->gs_as_pl_layout) { vkDestroyPipelineLayout(gpu->device, gpu->gs_as_pl_layout, NULL); gpu->gs_as_pl_layout = VK_NULL_HANDLE; }
    if (gpu->gs_as_ds_layout) { vkDestroyDescriptorSetLayout(gpu->device, gpu->gs_as_ds_layout, NULL); gpu->gs_as_ds_layout = VK_NULL_HANDLE; }
    if (gpu->gs_as_ds_pool)   { vkDestroyDescriptorPool(gpu->device, gpu->gs_as_ds_pool, NULL);    gpu->gs_as_ds_pool = VK_NULL_HANDLE; }
    gpu->gs_as_ds = VK_NULL_HANDLE;
    gpu->gs_as_built = 0;

    /* TLAS + instance buffer. */
    if (gpu->gs_tlas)              { vkDestroyAccelerationStructureKHR(gpu->device, gpu->gs_tlas, NULL); gpu->gs_tlas = VK_NULL_HANDLE; }
    if (gpu->gs_tlas_buf)          { vkDestroyBuffer(gpu->device, gpu->gs_tlas_buf, NULL);  gpu->gs_tlas_buf = VK_NULL_HANDLE; }
    if (gpu->gs_tlas_mem)          { vkFreeMemory(gpu->device, gpu->gs_tlas_mem, NULL);      gpu->gs_tlas_mem = VK_NULL_HANDLE; }
    if (gpu->gs_tlas_scratch_buf)  { vkDestroyBuffer(gpu->device, gpu->gs_tlas_scratch_buf, NULL); gpu->gs_tlas_scratch_buf = VK_NULL_HANDLE; }
    if (gpu->gs_tlas_scratch_mem)  { vkFreeMemory(gpu->device, gpu->gs_tlas_scratch_mem, NULL);     gpu->gs_tlas_scratch_mem = VK_NULL_HANDLE; }
    if (gpu->gs_inst_buf)          { vkDestroyBuffer(gpu->device, gpu->gs_inst_buf, NULL);  gpu->gs_inst_buf = VK_NULL_HANDLE; }
    if (gpu->gs_inst_mem)          { vkFreeMemory(gpu->device, gpu->gs_inst_mem, NULL);      gpu->gs_inst_mem = VK_NULL_HANDLE; }
    gpu->gs_tlas_size = 0;
    gpu->gs_tlas_scratch_size = 0;
    gpu->gs_inst_capacity = 0;

    /* Unit BLASes (icosahedron + AABB). */
    if (gpu->gs_blas_icosa)     { vkDestroyAccelerationStructureKHR(gpu->device, gpu->gs_blas_icosa, NULL); gpu->gs_blas_icosa = VK_NULL_HANDLE; }
    if (gpu->gs_blas_icosa_buf) { vkDestroyBuffer(gpu->device, gpu->gs_blas_icosa_buf, NULL); gpu->gs_blas_icosa_buf = VK_NULL_HANDLE; }
    if (gpu->gs_blas_icosa_mem) { vkFreeMemory(gpu->device, gpu->gs_blas_icosa_mem, NULL);     gpu->gs_blas_icosa_mem = VK_NULL_HANDLE; }
    if (gpu->gs_icosa_vb)       { vkDestroyBuffer(gpu->device, gpu->gs_icosa_vb, NULL);     gpu->gs_icosa_vb = VK_NULL_HANDLE; }
    if (gpu->gs_icosa_vb_mem)   { vkFreeMemory(gpu->device, gpu->gs_icosa_vb_mem, NULL);     gpu->gs_icosa_vb_mem = VK_NULL_HANDLE; }
    if (gpu->gs_icosa_ib)       { vkDestroyBuffer(gpu->device, gpu->gs_icosa_ib, NULL);     gpu->gs_icosa_ib = VK_NULL_HANDLE; }
    if (gpu->gs_icosa_ib_mem)   { vkFreeMemory(gpu->device, gpu->gs_icosa_ib_mem, NULL);     gpu->gs_icosa_ib_mem = VK_NULL_HANDLE; }
    gpu->gs_blas_icosa_addr = 0;

    if (gpu->gs_blas_aabb)      { vkDestroyAccelerationStructureKHR(gpu->device, gpu->gs_blas_aabb, NULL); gpu->gs_blas_aabb = VK_NULL_HANDLE; }
    if (gpu->gs_blas_aabb_buf)  { vkDestroyBuffer(gpu->device, gpu->gs_blas_aabb_buf, NULL); gpu->gs_blas_aabb_buf = VK_NULL_HANDLE; }
    if (gpu->gs_blas_aabb_mem)  { vkFreeMemory(gpu->device, gpu->gs_blas_aabb_mem, NULL);     gpu->gs_blas_aabb_mem = VK_NULL_HANDLE; }
    if (gpu->gs_aabb_buf)       { vkDestroyBuffer(gpu->device, gpu->gs_aabb_buf, NULL);     gpu->gs_aabb_buf = VK_NULL_HANDLE; }
    if (gpu->gs_aabb_buf_mem)   { vkFreeMemory(gpu->device, gpu->gs_aabb_buf_mem, NULL);     gpu->gs_aabb_buf_mem = VK_NULL_HANDLE; }
    gpu->gs_blas_aabb_addr = 0;

    /* Particle SSBOs. */
    #define _GS_DROP(buf, mem)                                                  \
        do { if (gpu->buf) { vkDestroyBuffer(gpu->device, gpu->buf, NULL); gpu->buf = VK_NULL_HANDLE; } \
             if (gpu->mem) { vkFreeMemory(gpu->device, gpu->mem, NULL); gpu->mem = VK_NULL_HANDLE; } } while (0)
    _GS_DROP(gs_pos_buf,   gs_pos_mem);
    _GS_DROP(gs_scale_buf, gs_scale_mem);
    _GS_DROP(gs_quat_buf,  gs_quat_mem);
    _GS_DROP(gs_opa_buf,   gs_opa_mem);
    _GS_DROP(gs_kerS_buf,  gs_kerS_mem);
    _GS_DROP(gs_sh_buf,    gs_sh_mem);
    #undef _GS_DROP
    gpu->gs_particle_capacity = 0;
    gpu->gs_particle_count = 0;
    gpu->gs_built = 0;
}

/* Resize-or-create one of the splat SSBOs (resident pool, untracked to
 * match the curve-data pattern). Frees the prior buffer if size differs. */
static int gs_ensure_ssbo(Gpu* gpu, VkBuffer* buf, VkDeviceMemory* mem,
                          VkDeviceSize bytes)
{
    if (*buf) {
        vkDestroyBuffer(gpu->device, *buf, NULL);
        *buf = VK_NULL_HANDLE;
    }
    if (*mem) {
        vkFreeMemory(gpu->device, *mem, NULL);
        *mem = VK_NULL_HANDLE;
    }
    if (bytes == 0) return 1;

    return rt_create_buffer_pooled(gpu, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        RT_POOL_RESIDENT,
        buf, mem);
}

int gpu_gs_upload_particles(Gpu* gpu,
                            const float* positions,
                            const float* scales,
                            const float* orientations,
                            const float* opacities,
                            const float* kernel_scales,
                            const float* sh_coefficients,
                            uint32_t particle_count,
                            int sh_degree,
                            int proxy_kind)
{
    if (!gpu) return 0;
    if (particle_count == 0) {
        gpu->gs_particle_count = 0;
        gpu->gs_built = 0;
        return 1;
    }
    if (!positions || !scales || !orientations || !opacities ||
        !kernel_scales || !sh_coefficients) return 0;
    if (sh_degree < 0 || sh_degree > 3) return 0;

    const uint32_t N      = particle_count;
    const uint32_t sh_per = (uint32_t)((sh_degree + 1) * (sh_degree + 1));

    const VkDeviceSize pos_b   = (VkDeviceSize)N * 3 * sizeof(float);
    const VkDeviceSize scale_b = (VkDeviceSize)N * 3 * sizeof(float);
    const VkDeviceSize quat_b  = (VkDeviceSize)N * 4 * sizeof(float);
    const VkDeviceSize opa_b   = (VkDeviceSize)N * 1 * sizeof(float);
    const VkDeviceSize kerS_b  = (VkDeviceSize)N * 1 * sizeof(float);
    const VkDeviceSize sh_b    = (VkDeviceSize)N * sh_per * 3 * sizeof(float);

    if (!gs_ensure_ssbo(gpu, &gpu->gs_pos_buf,   &gpu->gs_pos_mem,   pos_b))   goto fail;
    if (!gs_ensure_ssbo(gpu, &gpu->gs_scale_buf, &gpu->gs_scale_mem, scale_b)) goto fail;
    if (!gs_ensure_ssbo(gpu, &gpu->gs_quat_buf,  &gpu->gs_quat_mem,  quat_b))  goto fail;
    if (!gs_ensure_ssbo(gpu, &gpu->gs_opa_buf,   &gpu->gs_opa_mem,   opa_b))   goto fail;
    if (!gs_ensure_ssbo(gpu, &gpu->gs_kerS_buf,  &gpu->gs_kerS_mem,  kerS_b))  goto fail;
    if (!gs_ensure_ssbo(gpu, &gpu->gs_sh_buf,    &gpu->gs_sh_mem,    sh_b))    goto fail;

    /* Shared staging buffer sized to the largest SSBO. */
    VkDeviceSize stage_b = pos_b;
    if (scale_b > stage_b) stage_b = scale_b;
    if (quat_b  > stage_b) stage_b = quat_b;
    if (opa_b   > stage_b) stage_b = opa_b;
    if (kerS_b  > stage_b) stage_b = kerS_b;
    if (sh_b    > stage_b) stage_b = sh_b;

    VkBuffer       staging        = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem    = VK_NULL_HANDLE;
    void*          staging_mapped = NULL;
    if (!rt_create_staging_buffer(gpu, stage_b, &staging, &staging_mem, &staging_mapped))
        goto fail;

    struct {
        const void*  src;
        VkDeviceSize bytes;
        VkBuffer     dst;
    } passes[6] = {
        { positions,       pos_b,   gpu->gs_pos_buf   },
        { scales,          scale_b, gpu->gs_scale_buf },
        { orientations,    quat_b,  gpu->gs_quat_buf  },
        { opacities,       opa_b,   gpu->gs_opa_buf   },
        { kernel_scales,   kerS_b,  gpu->gs_kerS_buf  },
        { sh_coefficients, sh_b,    gpu->gs_sh_buf    },
    };

    for (int p = 0; p < 6; p++) {
        memcpy(staging_mapped, passes[p].src, (size_t)passes[p].bytes);
        VkCommandBuffer cmd = begin_single_command(gpu);
        VkBufferCopy region = {0};
        region.size = passes[p].bytes;
        vkCmdCopyBuffer(cmd, staging, passes[p].dst, 1, &region);
        end_single_command(gpu, cmd);
    }

    rt_destroy_staging_buffer(gpu, staging, staging_mem, staging_mapped);

    gpu->gs_particle_count    = N;
    gpu->gs_particle_capacity = N;
    gpu->gs_sh_degree         = sh_degree;
    gpu->gs_proxy_kind        = proxy_kind;
    return 1;

fail:
    fprintf(stderr, "gpu_gs_upload_particles: SSBO alloc/upload failed for N=%u\n", N);
    return 0;
}

/* Unit icosahedron BLAS — vertices normalized so max extent fits [-1,1]^3
 * (matching gs_common.h.slang's GS_ICOSA_VRT_SCALE compensation).
 * Faces match the reference vk_gaussian_splatting/shaders/particle_as_build
 * winding so the per-instance scale/quaternion math behaves identically. */
static const float kGsIcosaVerts[12 * 3] = {
    -0.618034f,  1.000000f,  0.000000f,
     0.618034f,  1.000000f,  0.000000f,
     0.000000f,  0.618034f, -1.000000f,
    -1.000000f,  0.000000f, -0.618034f,
    -1.000000f,  0.000000f,  0.618034f,
     0.000000f,  0.618034f,  1.000000f,
     1.000000f,  0.000000f,  0.618034f,
     0.000000f, -0.618034f,  1.000000f,
    -0.618034f, -1.000000f,  0.000000f,
     0.000000f, -0.618034f, -1.000000f,
     1.000000f,  0.000000f, -0.618034f,
     0.618034f, -1.000000f,  0.000000f,
};
static const uint32_t kGsIcosaIndices[60] = {
    0,  1,  2,   0,  2,  3,   0,  3,  4,   0,  4,  5,   0,  5,  1,
    6,  1,  5,   6,  5,  7,   6,  7, 11,   6, 11, 10,   6, 10,  1,
    8,  4,  3,   8,  3,  9,   8,  9, 11,   8, 11,  7,   8,  7,  4,
    9,  3,  2,   9,  2, 10,   9, 10, 11,   5,  4,  7,   1, 10,  2,
};

/* Lazily build the unit icosahedron BLAS. Idempotent: subsequent calls
 * return immediately if gs_blas_icosa is already built. Returns 1 on
 * success, 0 on failure (caller bails on the splat path). */
static int gs_build_unit_icosa_blas(Gpu* gpu)
{
    if (gpu->gs_blas_icosa) return 1;

    const VkDeviceSize vb_bytes = sizeof(kGsIcosaVerts);
    const VkDeviceSize ib_bytes = sizeof(kGsIcosaIndices);

    /* Vertex + index buffers — device-local, one-time staging upload. */
    if (!rt_create_buffer_pooled(gpu, vb_bytes,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_RESIDENT,
            &gpu->gs_icosa_vb, &gpu->gs_icosa_vb_mem)) {
        return 0;
    }
    if (!rt_create_buffer_pooled(gpu, ib_bytes,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_RESIDENT,
            &gpu->gs_icosa_ib, &gpu->gs_icosa_ib_mem)) {
        return 0;
    }

    /* Staging upload (shared buffer, sized to vb). */
    VkBuffer       stg = VK_NULL_HANDLE;
    VkDeviceMemory stg_mem = VK_NULL_HANDLE;
    void*          stg_map = NULL;
    if (!rt_create_staging_buffer(gpu, vb_bytes > ib_bytes ? vb_bytes : ib_bytes,
                                   &stg, &stg_mem, &stg_map))
        return 0;

    /* Vertex copy. */
    memcpy(stg_map, kGsIcosaVerts, vb_bytes);
    {
        VkCommandBuffer cmd = begin_single_command(gpu);
        VkBufferCopy r = { .size = vb_bytes };
        vkCmdCopyBuffer(cmd, stg, gpu->gs_icosa_vb, 1, &r);
        end_single_command(gpu, cmd);
    }
    /* Index copy. */
    memcpy(stg_map, kGsIcosaIndices, ib_bytes);
    {
        VkCommandBuffer cmd = begin_single_command(gpu);
        VkBufferCopy r = { .size = ib_bytes };
        vkCmdCopyBuffer(cmd, stg, gpu->gs_icosa_ib, 1, &r);
        end_single_command(gpu, cmd);
    }
    rt_destroy_staging_buffer(gpu, stg, stg_mem, stg_map);

    /* Geometry descriptor. NOT marked OPAQUE — VK3DGRT uses any-hit for
     * the K-buffer insertion-sort, and the OPAQUE flag lets the
     * implementation skip any-hit invocations entirely (per Vulkan
     * spec). Closest-hit still fires regardless, which is why the
     * Pragmatist gate (rchit-only shading) worked but the K-buffer
     * algorithm silently got zero rahit fires. */
    VkAccelerationStructureGeometryKHR geom = {0};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags        = 0;
    geom.geometry.triangles.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geom.geometry.triangles.vertexFormat  = VK_FORMAT_R32G32B32_SFLOAT;
    geom.geometry.triangles.vertexData.deviceAddress = rt_buf_addr(gpu, gpu->gs_icosa_vb);
    geom.geometry.triangles.vertexStride  = 3 * sizeof(float);
    geom.geometry.triangles.maxVertex     = 11;
    geom.geometry.triangles.indexType     = VK_INDEX_TYPE_UINT32;
    geom.geometry.triangles.indexData.deviceAddress = rt_buf_addr(gpu, gpu->gs_icosa_ib);

    VkAccelerationStructureBuildGeometryInfoKHR build = {0};
    build.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = 1;
    build.pGeometries   = &geom;

    const uint32_t prim_count = 20;
    VkAccelerationStructureBuildSizesInfoKHR sizes = {0};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(gpu->device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build, &prim_count, &sizes);

    /* AS storage buffer. */
    if (!rt_create_buffer_pooled(gpu, sizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_RESIDENT,
            &gpu->gs_blas_icosa_buf, &gpu->gs_blas_icosa_mem)) {
        return 0;
    }
    VkAccelerationStructureCreateInfoKHR as_ci = {0};
    as_ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    as_ci.buffer = gpu->gs_blas_icosa_buf;
    as_ci.offset = 0;
    as_ci.size   = sizes.accelerationStructureSize;
    as_ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR(gpu->device, &as_ci, NULL,
                                          &gpu->gs_blas_icosa) != VK_SUCCESS) {
        return 0;
    }

    /* Scratch buffer (single-use; freed at end of this function). */
    VkBuffer       scratch_buf = VK_NULL_HANDLE;
    VkDeviceMemory scratch_mem = VK_NULL_HANDLE;
    if (!rt_create_buffer_pooled(gpu, sizes.buildScratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_TRANSIENT,
            &scratch_buf, &scratch_mem)) {
        return 0;
    }

    build.dstAccelerationStructure  = gpu->gs_blas_icosa;
    build.scratchData.deviceAddress = rt_buf_addr(gpu, scratch_buf);

    VkAccelerationStructureBuildRangeInfoKHR range = {0};
    range.primitiveCount = prim_count;
    const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range;

    VkCommandBuffer cmd = begin_single_command(gpu);
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &p_range);
    end_single_command(gpu, cmd);

    /* Cache the BLAS device address for instance buffer construction. */
    VkAccelerationStructureDeviceAddressInfoKHR addr_info = {0};
    addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addr_info.accelerationStructure = gpu->gs_blas_icosa;
    gpu->gs_blas_icosa_addr = vkGetAccelerationStructureDeviceAddressKHR(
        gpu->device, &addr_info);

    /* Scratch is in the transient pool — destroying the buffer handle
     * leaks no memory; the pool's bump cursor is reset elsewhere. */
    if (scratch_buf) vkDestroyBuffer(gpu->device, scratch_buf, NULL);
    if (scratch_mem) vkFreeMemory(gpu->device, scratch_mem, NULL);

    fprintf(stderr, "gpu_vulkan: VK3DGRT unit icosahedron BLAS built "
                    "(20 tris, %llu B).\n",
            (unsigned long long)sizes.accelerationStructureSize);
    return 1;
}

/* Lazily build the unit AABB BLAS for NU_GS_PROXY_AABB. One procedural
 * primitive at [-1, +1]^3 in object space; the gs_as_build compute shader
 * applies per-instance scale `scl·kerS·1.0` so the AABB in world space
 * exactly bounds the 3-σ Gaussian cube. The AABB-proxy hit group has the
 * gs_3dgrt.rint intersection shader; rahit then runs as usual.
 * Idempotent. */
static int gs_build_unit_aabb_blas(Gpu* gpu)
{
    if (gpu->gs_blas_aabb) return 1;

    /* VkAabbPositionsKHR = float minX,minY,minZ,maxX,maxY,maxZ (24 B). */
    const float aabb_positions[6] = {
        -1.0f, -1.0f, -1.0f,
        +1.0f, +1.0f, +1.0f,
    };
    const VkDeviceSize aabb_bytes = sizeof(aabb_positions);

    if (!rt_create_buffer_pooled(gpu, aabb_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_RESIDENT,
            &gpu->gs_aabb_buf, &gpu->gs_aabb_buf_mem)) {
        return 0;
    }

    /* Staging upload. */
    VkBuffer       stg = VK_NULL_HANDLE;
    VkDeviceMemory stg_mem = VK_NULL_HANDLE;
    void*          stg_map = NULL;
    if (!rt_create_staging_buffer(gpu, aabb_bytes, &stg, &stg_mem, &stg_map))
        return 0;
    memcpy(stg_map, aabb_positions, aabb_bytes);
    {
        VkCommandBuffer cmd = begin_single_command(gpu);
        VkBufferCopy r = { .size = aabb_bytes };
        vkCmdCopyBuffer(cmd, stg, gpu->gs_aabb_buf, 1, &r);
        end_single_command(gpu, cmd);
    }
    rt_destroy_staging_buffer(gpu, stg, stg_mem, stg_map);

    /* Geometry descriptor — AABBs (NOT marked OPAQUE so any-hit fires
     * for the K-buffer insertion-sort, just like the icosa path). */
    VkAccelerationStructureGeometryAabbsDataKHR aabbs_data = {0};
    aabbs_data.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbs_data.data.deviceAddress = rt_buf_addr(gpu, gpu->gs_aabb_buf);
    aabbs_data.stride = 24;  /* sizeof(VkAabbPositionsKHR) */

    VkAccelerationStructureGeometryKHR geom = {0};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geom.flags        = 0;
    geom.geometry.aabbs = aabbs_data;

    VkAccelerationStructureBuildGeometryInfoKHR build = {0};
    build.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = 1;
    build.pGeometries   = &geom;

    const uint32_t prim_count = 1;
    VkAccelerationStructureBuildSizesInfoKHR sizes = {0};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(gpu->device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build, &prim_count, &sizes);

    if (!rt_create_buffer_pooled(gpu, sizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_RESIDENT,
            &gpu->gs_blas_aabb_buf, &gpu->gs_blas_aabb_mem)) {
        return 0;
    }

    VkAccelerationStructureCreateInfoKHR as_ci = {0};
    as_ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    as_ci.buffer = gpu->gs_blas_aabb_buf;
    as_ci.offset = 0;
    as_ci.size   = sizes.accelerationStructureSize;
    as_ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR(gpu->device, &as_ci, NULL,
                                          &gpu->gs_blas_aabb) != VK_SUCCESS) {
        return 0;
    }

    VkBuffer       scratch_buf = VK_NULL_HANDLE;
    VkDeviceMemory scratch_mem = VK_NULL_HANDLE;
    if (!rt_create_buffer_pooled(gpu, sizes.buildScratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_TRANSIENT,
            &scratch_buf, &scratch_mem)) {
        return 0;
    }

    build.dstAccelerationStructure  = gpu->gs_blas_aabb;
    build.scratchData.deviceAddress = rt_buf_addr(gpu, scratch_buf);

    VkAccelerationStructureBuildRangeInfoKHR range = { .primitiveCount = prim_count };
    const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range;

    VkCommandBuffer cmd = begin_single_command(gpu);
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &p_range);
    end_single_command(gpu, cmd);

    VkAccelerationStructureDeviceAddressInfoKHR addr_info = {0};
    addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addr_info.accelerationStructure = gpu->gs_blas_aabb;
    gpu->gs_blas_aabb_addr = vkGetAccelerationStructureDeviceAddressKHR(
        gpu->device, &addr_info);

    if (scratch_buf) vkDestroyBuffer(gpu->device, scratch_buf, NULL);
    if (scratch_mem) vkFreeMemory(gpu->device, scratch_mem, NULL);

    fprintf(stderr, "gpu_vulkan: VK3DGRT unit AABB BLAS built (%llu B).\n",
            (unsigned long long)sizes.accelerationStructureSize);
    return 1;
}

/* Push constant layout for gs_as_build.comp.slang. Must match
 * GsAsBuildPC in src/shaders/slang/gs_common.h.slang. Slang's
 * -matrix-layout-column-major flag means we ship a column-major mat4
 * here; the C-side prim_xform is row-major 4x4, so we transpose at
 * push time. */
typedef struct {
    uint64_t blas_address;
    uint64_t instance_address;
    uint32_t particle_count;
    uint32_t proxy_kind;
    uint32_t kernel_degree;
    uint32_t adaptive_clamping;
    float    prim_xform_col_major[16];  /* 64 B, 16-byte aligned */
} GsAsBuildPushConstants;

/* Lazily build the gs_as_build compute pipeline. Idempotent. */
static int gs_ensure_as_build_pipeline(Gpu* gpu)
{
    if (gpu->gs_as_built) return 1;

    /* Descriptor set layout: 6 storage buffers (5 in + 1 out). */
    VkDescriptorSetLayoutBinding bindings[6] = {{0}};
    for (int i = 0; i < 6; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
    dsl_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 6;
    dsl_ci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(gpu->device, &dsl_ci, NULL,
                                     &gpu->gs_as_ds_layout) != VK_SUCCESS) {
        fprintf(stderr, "gs_as_build: DS layout failed\n");
        return 0;
    }

    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
    VkDescriptorPoolCreateInfo dp_ci = {0};
    dp_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets       = 1;
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes    = &ps;
    if (vkCreateDescriptorPool(gpu->device, &dp_ci, NULL,
                                &gpu->gs_as_ds_pool) != VK_SUCCESS) {
        fprintf(stderr, "gs_as_build: DS pool failed\n");
        return 0;
    }

    VkDescriptorSetAllocateInfo dsa = {0};
    dsa.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool     = gpu->gs_as_ds_pool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts        = &gpu->gs_as_ds_layout;
    if (vkAllocateDescriptorSets(gpu->device, &dsa, &gpu->gs_as_ds) != VK_SUCCESS) {
        fprintf(stderr, "gs_as_build: DS alloc failed\n");
        return 0;
    }

    /* Pipeline layout (DS + push-constant block). */
    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(GsAsBuildPushConstants);

    VkPipelineLayoutCreateInfo pl_ci = {0};
    pl_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount         = 1;
    pl_ci.pSetLayouts            = &gpu->gs_as_ds_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(gpu->device, &pl_ci, NULL,
                                &gpu->gs_as_pl_layout) != VK_SUCCESS) {
        fprintf(stderr, "gs_as_build: pipeline layout failed\n");
        return 0;
    }

    /* Load SPIR-V (slang compiler emits .compute.spv suffix). */
    char path[512];
    snprintf(path, sizeof(path), "%s/slang/gs_as_build.compute.spv", SHADER_DIR);
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "gs_as_build: can't open %s\n", path);
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* code = (uint32_t*)malloc((size_t)sz);
    if (!code || fread(code, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(code);
        fprintf(stderr, "gs_as_build: shader read failed\n");
        return 0;
    }
    fclose(f);

    VkShaderModule mod = create_shader_module(gpu->device, code, (uint32_t)sz);
    free(code);
    if (!mod) return 0;

    VkComputePipelineCreateInfo cp = {0};
    cp.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = mod;
    cp.stage.pName  = "main";
    cp.layout       = gpu->gs_as_pl_layout;
    VkResult r = vkCreateComputePipelines(gpu->device, gpu->pipeline_cache,
                                           1, &cp, NULL, &gpu->gs_as_pipeline);
    vkDestroyShaderModule(gpu->device, mod, NULL);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "gs_as_build: vkCreateComputePipelines failed (%d)\n", r);
        return 0;
    }

    gpu->gs_as_built = 1;
    return 1;
}

/* Refresh the 6 storage-buffer descriptor writes for the gs_as_build set.
 * Called after any of the SSBO buffers is reallocated. */
static void gs_update_as_build_descriptors(Gpu* gpu)
{
    if (!gpu->gs_as_built) return;

    VkBuffer bufs[6] = {
        gpu->gs_pos_buf, gpu->gs_scale_buf, gpu->gs_quat_buf,
        gpu->gs_opa_buf, gpu->gs_kerS_buf,  gpu->gs_inst_buf,
    };
    for (int i = 0; i < 6; i++) {
        if (!bufs[i]) return; /* not ready */
    }

    VkDescriptorBufferInfo bi[6] = {{0}};
    for (int i = 0; i < 6; i++) {
        bi[i].buffer = bufs[i];
        bi[i].offset = 0;
        bi[i].range  = VK_WHOLE_SIZE;
    }
    VkWriteDescriptorSet ws[6] = {{0}};
    for (int i = 0; i < 6; i++) {
        ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet          = gpu->gs_as_ds;
        ws[i].dstBinding      = (uint32_t)i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[i].pBufferInfo     = &bi[i];
    }
    vkUpdateDescriptorSets(gpu->device, 6, ws, 0, NULL);
}

/* Allocate-or-resize gs_inst_buf to hold N VkAccelerationStructureInstanceKHR
 * (64 B each), with INSTANCES (AS-build read), SHADER_DEVICE_ADDRESS (so
 * we can pass the device address to the compute shader as the output
 * pointer), and STORAGE_BUFFER (compute writes via descriptor set). */
static int gs_ensure_inst_buf(Gpu* gpu, uint32_t N)
{
    const VkDeviceSize bytes = (VkDeviceSize)N * 64;
    if (gpu->gs_inst_buf && gpu->gs_inst_capacity >= bytes) return 1;

    if (gpu->gs_inst_buf) {
        vkDestroyBuffer(gpu->device, gpu->gs_inst_buf, NULL);
        gpu->gs_inst_buf = VK_NULL_HANDLE;
    }
    if (gpu->gs_inst_mem) {
        vkFreeMemory(gpu->device, gpu->gs_inst_mem, NULL);
        gpu->gs_inst_mem = VK_NULL_HANDLE;
    }

    VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    if (!rt_create_buffer_pooled(gpu, bytes, usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_RESIDENT,
            &gpu->gs_inst_buf, &gpu->gs_inst_mem)) {
        return 0;
    }
    gpu->gs_inst_capacity = bytes;
    return 1;
}

/* Build gs_tlas over the GPU-built instance buffer. Single-AS, mode=BUILD,
 * flags=PREFER_FAST_TRACE. Allocates/grows scratch + as-storage on demand. */
static int gs_build_tlas(Gpu* gpu, uint32_t N)
{
    /* Drop prior AS handle (re-creating each build is fine; TLAS storage
     * is tiny relative to the BLAS pool). */
    if (gpu->gs_tlas) {
        vkDestroyAccelerationStructureKHR(gpu->device, gpu->gs_tlas, NULL);
        gpu->gs_tlas = VK_NULL_HANDLE;
    }

    VkAccelerationStructureGeometryInstancesDataKHR inst_data = {0};
    inst_data.sType            = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    inst_data.arrayOfPointers  = VK_FALSE;
    inst_data.data.deviceAddress = rt_buf_addr(gpu, gpu->gs_inst_buf);

    VkAccelerationStructureGeometryKHR geom = {0};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.geometry.instances = inst_data;

    VkAccelerationStructureBuildGeometryInfoKHR build = {0};
    build.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = 1;
    build.pGeometries   = &geom;

    VkAccelerationStructureBuildSizesInfoKHR sizes = {0};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(gpu->device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build, &N, &sizes);

    /* AS storage. */
    if (!gpu->gs_tlas_buf || gpu->gs_tlas_size < sizes.accelerationStructureSize) {
        if (gpu->gs_tlas_buf) { vkDestroyBuffer(gpu->device, gpu->gs_tlas_buf, NULL); gpu->gs_tlas_buf = VK_NULL_HANDLE; }
        if (gpu->gs_tlas_mem) { vkFreeMemory(gpu->device, gpu->gs_tlas_mem, NULL);     gpu->gs_tlas_mem = VK_NULL_HANDLE; }
        if (!rt_create_buffer_pooled(gpu, sizes.accelerationStructureSize,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                RT_POOL_RESIDENT,
                &gpu->gs_tlas_buf, &gpu->gs_tlas_mem)) {
            return 0;
        }
        gpu->gs_tlas_size = sizes.accelerationStructureSize;
    }

    VkAccelerationStructureCreateInfoKHR as_ci = {0};
    as_ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    as_ci.buffer = gpu->gs_tlas_buf;
    as_ci.offset = 0;
    as_ci.size   = sizes.accelerationStructureSize;
    as_ci.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR(gpu->device, &as_ci, NULL,
                                          &gpu->gs_tlas) != VK_SUCCESS) {
        return 0;
    }

    /* Scratch storage. */
    if (!gpu->gs_tlas_scratch_buf ||
        gpu->gs_tlas_scratch_size < sizes.buildScratchSize) {
        if (gpu->gs_tlas_scratch_buf) { vkDestroyBuffer(gpu->device, gpu->gs_tlas_scratch_buf, NULL); gpu->gs_tlas_scratch_buf = VK_NULL_HANDLE; }
        if (gpu->gs_tlas_scratch_mem) { vkFreeMemory(gpu->device, gpu->gs_tlas_scratch_mem, NULL); gpu->gs_tlas_scratch_mem = VK_NULL_HANDLE; }
        if (!rt_create_buffer_pooled(gpu, sizes.buildScratchSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                RT_POOL_RESIDENT,
                &gpu->gs_tlas_scratch_buf, &gpu->gs_tlas_scratch_mem)) {
            return 0;
        }
        gpu->gs_tlas_scratch_size = sizes.buildScratchSize;
    }

    build.dstAccelerationStructure  = gpu->gs_tlas;
    build.scratchData.deviceAddress = rt_buf_addr(gpu, gpu->gs_tlas_scratch_buf);

    VkAccelerationStructureBuildRangeInfoKHR range = { .primitiveCount = N };
    const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range;

    /* Build in the same one-shot cmd buffer as the compute dispatch
     * (caller passed the cmd already), but for now we use a separate
     * one-shot — the compute -> build sync uses an explicit barrier. */
    VkCommandBuffer cmd = begin_single_command(gpu);
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &p_range);
    end_single_command(gpu, cmd);
    return 1;
}

int gpu_gs_build_accel(Gpu* gpu, const float prim_xform[16])
{
    if (!gpu) return 0;
    if (gpu->gs_particle_count == 0) {
        gpu->gs_built = 0;
        return 0;
    }

    /* 1. Unit BLAS for the active proxy kind. */
    if (gpu->gs_proxy_kind == 0) {            /* NU_GS_PROXY_ICOSAHEDRON */
        if (!gs_build_unit_icosa_blas(gpu)) {
            fprintf(stderr, "gpu_gs_build_accel: unit icosa BLAS failed\n");
            return 0;
        }
    } else {                                   /* NU_GS_PROXY_AABB */
        if (!gs_build_unit_aabb_blas(gpu)) {
            fprintf(stderr, "gpu_gs_build_accel: unit AABB BLAS failed\n");
            return 0;
        }
    }

    /* 2. Allocate (or grow) the TLAS instance buffer. */
    if (!gs_ensure_inst_buf(gpu, gpu->gs_particle_count)) {
        fprintf(stderr, "gpu_gs_build_accel: instance buffer alloc failed\n");
        return 0;
    }

    /* 3. Build the gs_as_build compute pipeline (lazy) + refresh DS. */
    if (!gs_ensure_as_build_pipeline(gpu)) {
        fprintf(stderr, "gpu_gs_build_accel: compute pipeline build failed\n");
        return 0;
    }
    gs_update_as_build_descriptors(gpu);

    /* 4. Dispatch gs_as_build to fill the instance buffer.
     *
     * Push constants: blas_address, instance_address, particle_count,
     *                 proxy_kind, kernel_degree (=2), adaptive_clamping (=1),
     *                 prim_xform (column-major).
     * Workgroup size = 256 (matches [numthreads(256,1,1)] in the slang). */
    GsAsBuildPushConstants pc = {0};
    pc.blas_address      = (gpu->gs_proxy_kind == 0)
                              ? gpu->gs_blas_icosa_addr
                              : gpu->gs_blas_aabb_addr;
    pc.instance_address  = rt_buf_addr(gpu, gpu->gs_inst_buf);
    pc.particle_count    = gpu->gs_particle_count;
    pc.proxy_kind        = (uint32_t)gpu->gs_proxy_kind;
    pc.kernel_degree     = 2;
    pc.adaptive_clamping = 1;

    /* Transpose row-major C → column-major shader. */
    if (prim_xform) {
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                pc.prim_xform_col_major[c * 4 + r] = prim_xform[r * 4 + c];
    } else {
        pc.prim_xform_col_major[0]  = 1.0f;
        pc.prim_xform_col_major[5]  = 1.0f;
        pc.prim_xform_col_major[10] = 1.0f;
        pc.prim_xform_col_major[15] = 1.0f;
    }

    {
        VkCommandBuffer cmd = begin_single_command(gpu);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          gpu->gs_as_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                gpu->gs_as_pl_layout, 0, 1,
                                &gpu->gs_as_ds, 0, NULL);
        vkCmdPushConstants(cmd, gpu->gs_as_pl_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(pc), &pc);
        const uint32_t groups = (gpu->gs_particle_count + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);

        /* Compute → AS_BUILD barrier on gs_inst_buf. */
        VkBufferMemoryBarrier mb = {0};
        mb.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mb.buffer = gpu->gs_inst_buf;
        mb.offset = 0;
        mb.size   = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            0, 0, NULL, 1, &mb, 0, NULL);
        end_single_command(gpu, cmd);
    }

    /* 5. Build the TLAS over the just-filled instance buffer. */
    if (!gs_build_tlas(gpu, gpu->gs_particle_count)) {
        fprintf(stderr, "gpu_gs_build_accel: TLAS build failed\n");
        return 0;
    }

    gpu->gs_built = 1;
    fprintf(stderr,
            "gpu_vulkan: VK3DGRT TLAS built (N=%u, %llu instance bytes).\n",
            gpu->gs_particle_count,
            (unsigned long long)gpu->gs_inst_capacity);
    return 1;
}

/* GsRtPC — must match gs_common.h.slang. Slang column-major matrices
 * lay out as 16 floats per matrix where matrix[col][row] = floats[col*4+row].
 * The tail of the struct is 8 uint/float scalars = 32 B; total = 160 B. */
typedef struct {
    float    view_inverse[16];   /* column-major */
    float    proj_inverse[16];   /* column-major */
    uint32_t particle_count;
    uint32_t sh_degree;
    uint32_t K;
    uint32_t max_passes;
    uint32_t camera_model;
    uint32_t color_space;
    float    min_transmittance;
    float    iso_opacity_threshold;
} GsRtPushConstants;

int gpu_gs_build_pipeline(Gpu* gpu,
                          const uint32_t* rgen_spv,  uint32_t rgen_sz,
                          const uint32_t* rahit_spv, uint32_t rahit_sz,
                          const uint32_t* rchit_spv, uint32_t rchit_sz,
                          const uint32_t* rint_spv,  uint32_t rint_sz,
                          const uint32_t* rmiss_spv, uint32_t rmiss_sz)
{
    if (!gpu || !rgen_spv || !rmiss_spv || !rahit_spv) return 0;
    if (gpu->gs_pipeline_built) return 1;

    /* gpu_gs_build_pipeline runs once. The first-time provisioning of
     * rt_image happens in gpu_gs_render_tile so that a resize between
     * calls is handled. (See the resize check there.) */
    if (!gpu->rt_image) {
        rt_create_storage_image(gpu);
        if (!gpu->rt_image) {
            fprintf(stderr, "gs_rt: rt_image alloc failed\n");
            return 0;
        }
    }

    /* Descriptor set layout:
     *   0    = TLAS                                  (rgen)
     *   1    = output storage image                  (rgen)
     *   2..7 = particle SSBOs (pos/scl/ori/opa/kerS/sh)  (rgen + rahit)
     *   8    = K-buffer id   (RWStructuredBuffer<uint>)  (rgen + rahit)
     *   9    = K-buffer dist (RWStructuredBuffer<float>) (rgen + rahit)
     *   10   = iso-opacity depth   (RWStructuredBuffer<float>) (rgen)
     *   11   = iso-opacity normal  (RWStructuredBuffer<float>) (rgen)
     *
     * K-buffer is per-tile SoA, sized W * H * K (plan §6 D3). rahit
     * insertion-sorts (t, instance_id) per-pixel; rgen walks the
     * sorted K-buffer front-to-back, α-composites, and writes the
     * iso-opacity surface t + analytic normal to bindings 10/11
     * (plan §7). Normal storage is RGB f32 × W·H (no padding). */
    VkDescriptorSetLayoutBinding bindings[12] = {{0}};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    for (int i = 2; i < 12; i++) {
        bindings[i].binding         = (uint32_t)i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR
                                    | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
                                    | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                                    | VK_SHADER_STAGE_INTERSECTION_BIT_KHR
                                    | VK_SHADER_STAGE_MISS_BIT_KHR;
    }

    VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
    dsl_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 12;
    dsl_ci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(gpu->device, &dsl_ci, NULL,
                                     &gpu->gs_rt_ds_layout) != VK_SUCCESS) {
        fprintf(stderr, "gs_rt: DS layout failed\n"); return 0;
    }

    VkDescriptorPoolSize ps[3] = {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             10 },
    };
    VkDescriptorPoolCreateInfo dp_ci = {0};
    dp_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets       = 1;
    dp_ci.poolSizeCount = 3;
    dp_ci.pPoolSizes    = ps;
    if (vkCreateDescriptorPool(gpu->device, &dp_ci, NULL,
                                &gpu->gs_rt_ds_pool) != VK_SUCCESS) {
        fprintf(stderr, "gs_rt: DS pool failed\n"); return 0;
    }
    VkDescriptorSetAllocateInfo dsa = {0};
    dsa.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool     = gpu->gs_rt_ds_pool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts        = &gpu->gs_rt_ds_layout;
    if (vkAllocateDescriptorSets(gpu->device, &dsa, &gpu->gs_rt_ds) != VK_SUCCESS) {
        fprintf(stderr, "gs_rt: DS alloc failed\n"); return 0;
    }

    /* Pipeline layout (DS + push constants). */
    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR
                   | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                   | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
                   | VK_SHADER_STAGE_INTERSECTION_BIT_KHR
                   | VK_SHADER_STAGE_MISS_BIT_KHR;
    pcr.offset     = 0;
    pcr.size       = sizeof(GsRtPushConstants);

    VkPipelineLayoutCreateInfo pl_ci = {0};
    pl_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount         = 1;
    pl_ci.pSetLayouts            = &gpu->gs_rt_ds_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(gpu->device, &pl_ci, NULL,
                                &gpu->gs_rt_pl_layout) != VK_SUCCESS) {
        fprintf(stderr, "gs_rt: pipeline layout failed\n"); return 0;
    }

    /* Stages: rgen, rmiss, rahit, [rchit], [rint for AABB]. */
    const int has_rint  = (gpu->gs_proxy_kind != 0) && rint_spv  && rint_sz  > 0;
    const int has_rchit = rchit_spv && rchit_sz > 0;
    VkShaderModule mod_rgen  = create_shader_module(gpu->device, rgen_spv,  rgen_sz);
    VkShaderModule mod_rmiss = create_shader_module(gpu->device, rmiss_spv, rmiss_sz);
    VkShaderModule mod_rahit = create_shader_module(gpu->device, rahit_spv, rahit_sz);
    VkShaderModule mod_rchit = has_rchit
        ? create_shader_module(gpu->device, rchit_spv, rchit_sz)
        : VK_NULL_HANDLE;
    VkShaderModule mod_rint  = has_rint
        ? create_shader_module(gpu->device, rint_spv, rint_sz)
        : VK_NULL_HANDLE;
    if (!mod_rgen || !mod_rmiss || !mod_rahit ||
        (has_rchit && !mod_rchit) || (has_rint && !mod_rint)) {
        fprintf(stderr, "gs_rt: shader module creation failed\n");
        return 0;
    }

    VkPipelineShaderStageCreateInfo stages[5] = {{0}};
    uint32_t n_stages = 0;
    stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[n_stages].stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[n_stages].module = mod_rgen;
    stages[n_stages].pName  = "main";
    const uint32_t s_rgen = n_stages++;

    stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[n_stages].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[n_stages].module = mod_rmiss;
    stages[n_stages].pName  = "main";
    const uint32_t s_rmiss = n_stages++;

    stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[n_stages].stage  = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[n_stages].module = mod_rahit;
    stages[n_stages].pName  = "main";
    const uint32_t s_rahit = n_stages++;

    uint32_t s_rchit = VK_SHADER_UNUSED_KHR;
    if (has_rchit) {
        stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[n_stages].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        stages[n_stages].module = mod_rchit;
        stages[n_stages].pName  = "main";
        s_rchit = n_stages++;
    }

    uint32_t s_rint = VK_SHADER_UNUSED_KHR;
    if (has_rint) {
        stages[n_stages].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[n_stages].stage  = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
        stages[n_stages].module = mod_rint;
        stages[n_stages].pName  = "main";
        s_rint = n_stages++;
    }

    /* Groups: rgen | rmiss | hit_group (triangle or procedural). */
    VkRayTracingShaderGroupCreateInfoKHR groups[3] = {{0}};
    groups[0].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader      = s_rgen;
    groups[0].closestHitShader   = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader       = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[1].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader      = s_rmiss;
    groups[1].closestHitShader   = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader       = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[2].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type               = has_rint
                                     ? VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
                                     : VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].generalShader      = VK_SHADER_UNUSED_KHR;
    groups[2].closestHitShader   = s_rchit;     /* VK_SHADER_UNUSED_KHR if no rchit */
    groups[2].anyHitShader       = s_rahit;
    groups[2].intersectionShader = has_rint ? s_rint : VK_SHADER_UNUSED_KHR;

    VkRayTracingPipelineCreateInfoKHR rt_ci = {0};
    rt_ci.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rt_ci.stageCount                   = n_stages;
    rt_ci.pStages                      = stages;
    rt_ci.groupCount                   = 3;
    rt_ci.pGroups                      = groups;
    rt_ci.maxPipelineRayRecursionDepth = 1;
    rt_ci.layout                       = gpu->gs_rt_pl_layout;

    VkResult r = vkCreateRayTracingPipelinesKHR(gpu->device,
        VK_NULL_HANDLE, gpu->pipeline_cache, 1, &rt_ci, NULL,
        &gpu->gs_rt_pipeline);

    vkDestroyShaderModule(gpu->device, mod_rgen,  NULL);
    vkDestroyShaderModule(gpu->device, mod_rmiss, NULL);
    vkDestroyShaderModule(gpu->device, mod_rahit, NULL);
    if (mod_rchit) vkDestroyShaderModule(gpu->device, mod_rchit, NULL);
    if (mod_rint)  vkDestroyShaderModule(gpu->device, mod_rint,  NULL);

    if (r != VK_SUCCESS) {
        fprintf(stderr, "gs_rt: vkCreateRayTracingPipelinesKHR failed (%d)\n", r);
        return 0;
    }

    /* SBT: 3 entries (rgen | miss | hit_group). */
    {
        const uint32_t handle_size = gpu->rt_handle_size;
        const uint32_t align       = gpu->rt_handle_alignment;
        const uint32_t stride      = (handle_size + align - 1) & ~(align - 1);
        const uint32_t n_total     = 3;
        const VkDeviceSize sbt_size = (VkDeviceSize)stride * n_total;

        uint8_t* handles = (uint8_t*)malloc((size_t)handle_size * n_total);
        vkGetRayTracingShaderGroupHandlesKHR(gpu->device, gpu->gs_rt_pipeline,
            0, n_total, (size_t)handle_size * n_total, handles);

        if (!rt_create_buffer_pooled(gpu, sbt_size,
                VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                RT_POOL_NONE,  /* host-visible */
                &gpu->gs_sbt_buf, &gpu->gs_sbt_mem)) {
            free(handles);
            fprintf(stderr, "gs_rt: SBT alloc failed\n");
            return 0;
        }

        void* mapped = NULL;
        vkMapMemory(gpu->device, gpu->gs_sbt_mem, 0, sbt_size, 0, &mapped);
        memset(mapped, 0, (size_t)sbt_size);
        for (uint32_t g = 0; g < n_total; g++) {
            memcpy((uint8_t*)mapped + (VkDeviceSize)stride * g,
                   handles + (size_t)handle_size * g, handle_size);
        }
        vkUnmapMemory(gpu->device, gpu->gs_sbt_mem);
        free(handles);

        const VkDeviceAddress sbt_addr = rt_buf_addr(gpu, gpu->gs_sbt_buf);
        gpu->gs_sbt_rgen.deviceAddress = sbt_addr + (VkDeviceSize)stride * 0;
        gpu->gs_sbt_rgen.stride        = stride;
        gpu->gs_sbt_rgen.size          = stride;
        gpu->gs_sbt_miss.deviceAddress = sbt_addr + (VkDeviceSize)stride * 1;
        gpu->gs_sbt_miss.stride        = stride;
        gpu->gs_sbt_miss.size          = stride;
        gpu->gs_sbt_hit.deviceAddress  = sbt_addr + (VkDeviceSize)stride * 2;
        gpu->gs_sbt_hit.stride         = stride;
        gpu->gs_sbt_hit.size           = stride;
        memset(&gpu->gs_sbt_call, 0, sizeof(gpu->gs_sbt_call));
    }

    gpu->gs_pipeline_built = 1;
    fprintf(stderr, "gpu_vulkan: VK3DGRT RT pipeline built (proxy=%d, has_rchit=%d, has_rint=%d)\n",
            gpu->gs_proxy_kind, has_rchit, has_rint);
    return 1;
}

/* Allocate (or grow) the per-tile K-buffer (id + dist SoA). Must match
 * GS_KBUF_K in src/shaders/slang/gs_common.h.slang. Plan §6 default. */
#define GS_KBUF_K 18u

/* Allocate (or grow) the iso-opacity depth output (R32F per pixel). */
static int gs_ensure_depth(Gpu* gpu, uint32_t w, uint32_t h)
{
    if (gpu->gs_depth_buf && gpu->gs_depth_w == w && gpu->gs_depth_h == h)
        return 1;
    if (gpu->gs_depth_buf) { vkDestroyBuffer(gpu->device, gpu->gs_depth_buf, NULL); gpu->gs_depth_buf = VK_NULL_HANDLE; }
    if (gpu->gs_depth_mem) { vkFreeMemory(gpu->device, gpu->gs_depth_mem, NULL);     gpu->gs_depth_mem = VK_NULL_HANDLE; }

    const VkDeviceSize bytes = (VkDeviceSize)w * h * 4;
    if (!rt_create_buffer_pooled(gpu, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_RESIDENT,
            &gpu->gs_depth_buf, &gpu->gs_depth_mem)) {
        return 0;
    }
    gpu->gs_depth_w = w;
    gpu->gs_depth_h = h;
    return 1;
}

/* Allocate (or grow) the iso-opacity normal output (3 × R32F per pixel). */
static int gs_ensure_normal(Gpu* gpu, uint32_t w, uint32_t h)
{
    if (gpu->gs_normal_buf && gpu->gs_normal_w == w && gpu->gs_normal_h == h)
        return 1;
    if (gpu->gs_normal_buf) { vkDestroyBuffer(gpu->device, gpu->gs_normal_buf, NULL); gpu->gs_normal_buf = VK_NULL_HANDLE; }
    if (gpu->gs_normal_mem) { vkFreeMemory(gpu->device, gpu->gs_normal_mem, NULL);     gpu->gs_normal_mem = VK_NULL_HANDLE; }

    const VkDeviceSize bytes = (VkDeviceSize)w * h * 3 * 4;
    if (!rt_create_buffer_pooled(gpu, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_RESIDENT,
            &gpu->gs_normal_buf, &gpu->gs_normal_mem)) {
        return 0;
    }
    gpu->gs_normal_w = w;
    gpu->gs_normal_h = h;
    return 1;
}

static int gs_ensure_kbuf(Gpu* gpu, uint32_t w, uint32_t h, uint32_t k)
{
    if (gpu->gs_kbuf_id_buf &&
        gpu->gs_kbuf_w == w && gpu->gs_kbuf_h == h && gpu->gs_kbuf_k == k) {
        return 1;
    }

    if (gpu->gs_kbuf_id_buf)   { vkDestroyBuffer(gpu->device, gpu->gs_kbuf_id_buf, NULL);   gpu->gs_kbuf_id_buf = VK_NULL_HANDLE; }
    if (gpu->gs_kbuf_id_mem)   { vkFreeMemory(gpu->device, gpu->gs_kbuf_id_mem, NULL);       gpu->gs_kbuf_id_mem = VK_NULL_HANDLE; }
    if (gpu->gs_kbuf_dist_buf) { vkDestroyBuffer(gpu->device, gpu->gs_kbuf_dist_buf, NULL); gpu->gs_kbuf_dist_buf = VK_NULL_HANDLE; }
    if (gpu->gs_kbuf_dist_mem) { vkFreeMemory(gpu->device, gpu->gs_kbuf_dist_mem, NULL);     gpu->gs_kbuf_dist_mem = VK_NULL_HANDLE; }

    const VkDeviceSize bytes = (VkDeviceSize)w * h * k * 4;
    if (!rt_create_buffer_pooled(gpu, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_RESIDENT,
            &gpu->gs_kbuf_id_buf, &gpu->gs_kbuf_id_mem)) {
        return 0;
    }
    if (!rt_create_buffer_pooled(gpu, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            RT_POOL_RESIDENT,
            &gpu->gs_kbuf_dist_buf, &gpu->gs_kbuf_dist_mem)) {
        return 0;
    }
    gpu->gs_kbuf_w = w;
    gpu->gs_kbuf_h = h;
    gpu->gs_kbuf_k = k;
    return 1;
}

/* Refresh the splat RT descriptor set with the current TLAS, output image,
 * per-particle SSBOs, and K-buffer SSBOs. Called per-render because the
 * underlying buffer handles may grow. */
static void gs_update_rt_descriptors(Gpu* gpu)
{
    if (!gpu->gs_pipeline_built || !gpu->gs_tlas) return;

    VkBuffer ssbos[6] = {
        gpu->gs_pos_buf, gpu->gs_scale_buf, gpu->gs_quat_buf,
        gpu->gs_opa_buf, gpu->gs_kerS_buf,  gpu->gs_sh_buf,
    };
    for (int i = 0; i < 6; i++) {
        if (!ssbos[i]) return;  /* particles not yet uploaded */
    }
    if (!gpu->gs_kbuf_id_buf || !gpu->gs_kbuf_dist_buf) return;
    if (!gpu->gs_depth_buf || !gpu->gs_normal_buf) return;

    VkWriteDescriptorSetAccelerationStructureKHR ws_tlas = {0};
    ws_tlas.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    ws_tlas.accelerationStructureCount = 1;
    ws_tlas.pAccelerationStructures    = &gpu->gs_tlas;

    VkDescriptorImageInfo img_info = {0};
    img_info.imageView   = gpu->rt_image_view;
    img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo bi[10] = {{0}};
    for (int i = 0; i < 6; i++) {
        bi[i].buffer = ssbos[i];
        bi[i].offset = 0;
        bi[i].range  = VK_WHOLE_SIZE;
    }
    bi[6].buffer = gpu->gs_kbuf_id_buf;
    bi[6].offset = 0;
    bi[6].range  = VK_WHOLE_SIZE;
    bi[7].buffer = gpu->gs_kbuf_dist_buf;
    bi[7].offset = 0;
    bi[7].range  = VK_WHOLE_SIZE;
    bi[8].buffer = gpu->gs_depth_buf;
    bi[8].offset = 0;
    bi[8].range  = VK_WHOLE_SIZE;
    bi[9].buffer = gpu->gs_normal_buf;
    bi[9].offset = 0;
    bi[9].range  = VK_WHOLE_SIZE;

    VkWriteDescriptorSet ws[12] = {{0}};
    ws[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[0].pNext           = &ws_tlas;
    ws[0].dstSet          = gpu->gs_rt_ds;
    ws[0].dstBinding      = 0;
    ws[0].descriptorCount = 1;
    ws[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    ws[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[1].dstSet          = gpu->gs_rt_ds;
    ws[1].dstBinding      = 1;
    ws[1].descriptorCount = 1;
    ws[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[1].pImageInfo      = &img_info;

    for (int i = 0; i < 10; i++) {
        ws[2 + i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[2 + i].dstSet          = gpu->gs_rt_ds;
        ws[2 + i].dstBinding      = (uint32_t)(2 + i);
        ws[2 + i].descriptorCount = 1;
        ws[2 + i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[2 + i].pBufferInfo     = &bi[i];
    }

    vkUpdateDescriptorSets(gpu->device, 12, ws, 0, NULL);
}

int gpu_gs_fetch_depth(Gpu* gpu, float* out_depth, uint32_t w, uint32_t h)
{
    if (!gpu || !out_depth || !gpu->gs_depth_buf) return 0;
    if (w != gpu->gs_depth_w || h != gpu->gs_depth_h) return 0;

    const VkDeviceSize bytes = (VkDeviceSize)w * h * 4;

    /* Wait for any in-flight render; the splat path uses begin/end_single_command
     * which already vkQueueWaitIdle's, but a top-level wait is cheap insurance. */
    vkDeviceWaitIdle(gpu->device);

    VkBuffer       stg = VK_NULL_HANDLE;
    VkDeviceMemory stg_mem = VK_NULL_HANDLE;
    void*          stg_map = NULL;
    if (!rt_create_staging_buffer(gpu, bytes, &stg, &stg_mem, &stg_map))
        return 0;

    VkCommandBuffer cmd = begin_single_command(gpu);
    VkBufferCopy region = { .size = bytes };
    vkCmdCopyBuffer(cmd, gpu->gs_depth_buf, stg, 1, &region);
    end_single_command(gpu, cmd);

    memcpy(out_depth, stg_map, (size_t)bytes);
    rt_destroy_staging_buffer(gpu, stg, stg_mem, stg_map);
    return 1;
}

int gpu_gs_fetch_normal(Gpu* gpu, float* out_normal, uint32_t w, uint32_t h)
{
    if (!gpu || !out_normal || !gpu->gs_normal_buf) return 0;
    if (w != gpu->gs_normal_w || h != gpu->gs_normal_h) return 0;

    const VkDeviceSize bytes = (VkDeviceSize)w * h * 3 * 4;

    vkDeviceWaitIdle(gpu->device);

    VkBuffer       stg = VK_NULL_HANDLE;
    VkDeviceMemory stg_mem = VK_NULL_HANDLE;
    void*          stg_map = NULL;
    if (!rt_create_staging_buffer(gpu, bytes, &stg, &stg_mem, &stg_map))
        return 0;

    VkCommandBuffer cmd = begin_single_command(gpu);
    VkBufferCopy region = { .size = bytes };
    vkCmdCopyBuffer(cmd, gpu->gs_normal_buf, stg, 1, &region);
    end_single_command(gpu, cmd);

    memcpy(out_normal, stg_map, (size_t)bytes);
    rt_destroy_staging_buffer(gpu, stg, stg_mem, stg_map);
    return 1;
}

int gpu_gs_render_tile(Gpu* gpu,
                       const float view_inverse[16],
                       const float proj_inverse[16],
                       int K, int max_passes,
                       float min_transmittance,
                       int color_space,
                       int camera_model,
                       int tile_w, int tile_h)
{
    if (!gpu) return 0;
    if (!gpu->gs_built || !gpu->gs_pipeline_built) {
        fprintf(stderr, "gpu_gs_render_tile: gs not built (built=%d, pipe=%d)\n",
                gpu->gs_built, gpu->gs_pipeline_built);
        return 0;
    }
    /* Recreate rt_image when the swapchain has been resized. Without
     * this, the trace dispatch writes outside the bound storage and the
     * subsequent blit reads garbage outside the actually-rendered
     * region. Mirrors the mesh-RT resize check at gpu_vulkan.c:~6564. */
    if (!gpu->rt_image
        || gpu->rt_image_w != gpu->swapchain_extent.width
        || gpu->rt_image_h != gpu->swapchain_extent.height) {
        rt_create_storage_image(gpu);
        if (!gpu->rt_image) {
            fprintf(stderr, "gpu_gs_render_tile: rt_image alloc failed\n");
            return 0;
        }
    }

    /* Push the current camera + render config. */
    GsRtPushConstants pc = {0};
    /* Slang column-major: transpose row-major C input. */
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) {
        pc.view_inverse[c * 4 + r] = view_inverse[r * 4 + c];
        pc.proj_inverse[c * 4 + r] = proj_inverse[r * 4 + c];
    }
    pc.particle_count        = gpu->gs_particle_count;
    pc.sh_degree             = (uint32_t)gpu->gs_sh_degree;
    pc.K                     = (uint32_t)(K > 0 ? K : 16);
    pc.max_passes            = (uint32_t)(max_passes > 0 ? max_passes : 16);
    pc.camera_model          = (uint32_t)camera_model;
    pc.color_space           = (uint32_t)color_space;
    pc.min_transmittance     = (min_transmittance >= 0.0f) ? min_transmittance : 0.03f;
    pc.iso_opacity_threshold = 0.5f;

    /* Ensure the per-tile SoA K-buffer + iso-opacity AOV buffers exist. */
    if (!gs_ensure_kbuf(gpu, (uint32_t)tile_w, (uint32_t)tile_h, GS_KBUF_K)) {
        fprintf(stderr, "gpu_gs_render_tile: K-buffer alloc failed\n");
        return 0;
    }
    if (!gs_ensure_depth(gpu, (uint32_t)tile_w, (uint32_t)tile_h)) {
        fprintf(stderr, "gpu_gs_render_tile: depth buffer alloc failed\n");
        return 0;
    }
    if (!gs_ensure_normal(gpu, (uint32_t)tile_w, (uint32_t)tile_h)) {
        fprintf(stderr, "gpu_gs_render_tile: normal buffer alloc failed\n");
        return 0;
    }

    gs_update_rt_descriptors(gpu);

    VkCommandBuffer cmd = begin_single_command(gpu);

    /* Transition output image UNDEFINED → GENERAL for storage write. */
    VkImageMemoryBarrier mb = {0};
    mb.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    mb.oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
    mb.newLayout        = VK_IMAGE_LAYOUT_GENERAL;
    mb.srcAccessMask    = 0;
    mb.dstAccessMask    = VK_ACCESS_SHADER_WRITE_BIT;
    mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb.image            = gpu->rt_image;
    mb.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    mb.subresourceRange.baseMipLevel   = 0;
    mb.subresourceRange.levelCount     = 1;
    mb.subresourceRange.baseArrayLayer = 0;
    mb.subresourceRange.layerCount     = 1;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, NULL, 0, NULL, 1, &mb);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                      gpu->gs_rt_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            gpu->gs_rt_pl_layout, 0, 1, &gpu->gs_rt_ds, 0, NULL);
    vkCmdPushConstants(cmd, gpu->gs_rt_pl_layout,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR
      | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
      | VK_SHADER_STAGE_ANY_HIT_BIT_KHR
      | VK_SHADER_STAGE_INTERSECTION_BIT_KHR
      | VK_SHADER_STAGE_MISS_BIT_KHR,
        0, sizeof(pc), &pc);

    vkCmdTraceRaysKHR(cmd,
        &gpu->gs_sbt_rgen, &gpu->gs_sbt_miss, &gpu->gs_sbt_hit, &gpu->gs_sbt_call,
        (uint32_t)tile_w, (uint32_t)tile_h, 1);

    /* Blit rt_image → current swapchain image so nu_fetch_pixels (which
     * reads the swapchain) sees the splat render. Mirrors gpu_end_frame_rt
     * line ~6700+. Single-camera; tile_w/tile_h must match swapchain
     * extent for now. */
    {
        VkImageMemoryBarrier b[2] = {{0}};
        b[0].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[0].srcAccessMask    = VK_ACCESS_SHADER_WRITE_BIT;
        b[0].dstAccessMask    = VK_ACCESS_TRANSFER_READ_BIT;
        b[0].oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
        b[0].newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[0].image            = gpu->rt_image;
        b[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b[0].subresourceRange.levelCount = 1;
        b[0].subresourceRange.layerCount = 1;

        b[1].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[1].srcAccessMask    = VK_ACCESS_MEMORY_READ_BIT;
        b[1].dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
        b[1].oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED;     /* contents unused */
        b[1].newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[1].image            = gpu->swapchain_images[gpu->current_image];
        b[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b[1].subresourceRange.levelCount = 1;
        b[1].subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 0, NULL, 2, b);

        VkImageBlit blit = {0};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1].x = (int32_t)gpu->swapchain_extent.width;
        blit.srcOffsets[1].y = (int32_t)gpu->swapchain_extent.height;
        blit.srcOffsets[1].z = 1;
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1].x = (int32_t)gpu->swapchain_extent.width;
        blit.dstOffsets[1].y = (int32_t)gpu->swapchain_extent.height;
        blit.dstOffsets[1].z = 1;

        vkCmdBlitImage(cmd,
            gpu->rt_image,                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            gpu->swapchain_images[gpu->current_image],  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_NEAREST);

        /* Restore: rt_image → GENERAL (next render), swapchain → PRESENT_SRC. */
        VkImageMemoryBarrier r2[2] = {{0}};
        r2[0]                      = b[0];
        r2[0].srcAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
        r2[0].dstAccessMask        = VK_ACCESS_SHADER_WRITE_BIT;
        r2[0].oldLayout            = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        r2[0].newLayout            = VK_IMAGE_LAYOUT_GENERAL;
        r2[1]                      = b[1];
        r2[1].srcAccessMask        = VK_ACCESS_TRANSFER_WRITE_BIT;
        r2[1].dstAccessMask        = 0;
        r2[1].oldLayout            = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        r2[1].newLayout            = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, NULL, 0, NULL, 2, r2);
    }

    end_single_command(gpu, cmd);
    return 1;
}
