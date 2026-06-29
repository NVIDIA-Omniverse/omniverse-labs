// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_GPU_H
#define NUSD_GPU_H

/*
 * gpu.h — Render Hardware Interface (RHI)
 *
 * Thin abstraction over Vulkan (Linux/Windows) and Metal (macOS).
 * All handles are opaque. Platform-specific implementation is selected
 * at compile time:
 *   - gpu_vulkan.c  (Vulkan)
 *   - gpu_metal.m   (Metal, future)
 *
 * Pattern: The Forge / sokol_gfx style — minimal, no hidden state.
 */

#include <stdint.h>
#include <stddef.h>

#ifndef MAX_MTLX_PROC_NODES
#define MAX_MTLX_PROC_NODES 64
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Opaque handles ---- */

typedef struct Gpu          Gpu;
typedef struct GpuBuffer_s* GpuBuffer;
typedef struct GpuPipeline_s* GpuPipeline;

/* ---- Enums ---- */

typedef enum {
    GPU_BUFFER_VERTEX = 0,
    GPU_BUFFER_INDEX  = 1,
} GpuBufferUsage;

typedef enum {
    GPU_FORMAT_FLOAT3 = 0,  /* VK_FORMAT_R32G32B32_SFLOAT / MTLVertexFormatFloat3 */
    GPU_FORMAT_FLOAT2 = 1,  /* VK_FORMAT_R32G32_SFLOAT */
    GPU_FORMAT_UINT   = 2,  /* VK_FORMAT_R32_UINT */
    GPU_FORMAT_FLOAT  = 3,  /* VK_FORMAT_R32_SFLOAT */
} GpuVertexFormat;

/* ---- Descriptors ---- */

typedef struct {
    GpuBufferUsage usage;
    uint64_t       size;
    const void*    data;     /* NULL = allocate only, don't upload */
} GpuBufferDesc;

typedef struct {
    uint32_t        location;
    uint32_t        offset;
    GpuVertexFormat format;
} GpuVertexAttrib;

typedef struct {
    const uint32_t*     vert_spv;
    uint32_t            vert_size;    /* bytes */
    const uint32_t*     frag_spv;
    uint32_t            frag_size;    /* bytes */
    uint32_t            push_constant_size;
    uint32_t            vertex_stride;
    const GpuVertexAttrib* attribs;
    uint32_t            nattribs;
} GpuPipelineDesc;

/* Push constant layout: MVP + model + color + eye + matID + instancing = 176 bytes */
typedef struct {
    float mvp[16];        /* view-projection * model (for position transform) */
    float model[16];      /* world transform (for normal/position to world space) */
    float color[4];       /* .w > 0.5 → use this color; else use vertex color */
    float eye_pos[4];     /* camera world position (xyz), .w unused */
    int   material_index; /* -1 = no material; else index into materials[] */
    uint32_t instanced;   /* 1 = fetch per-instance raster data from buffer(2) */
    uint32_t instance_base;
    uint32_t _pad[1];     /* keep 16-byte alignment for the next param block */
} GpuMeshPushConstants;

typedef struct {
    float mvp[16];
    float model[16];
    float color[4];
    int   material_index;
    int   _pad[3];
} GpuRasterInstanceData;

/* RT push constants: inverse view + inverse projection + scene info = 156 bytes */
typedef struct {
    float view_inv[16];
    float proj_inv[16];
    float ground_y;       /* ground plane Y coordinate (scene bounds min Y) */
    float scene_scale;    /* scene diagonal for ground checker scale */
    uint32_t fast_mode;   /* 1 = skip shadow rays + PBR for RL sensors */
    uint32_t depth_enabled; /* 1 = write depth to binding 10 SSBO */
    uint32_t segmentation_enabled; /* 1 = write instance IDs to binding 11 SSBO */
    uint32_t normals_enabled;  /* 1 = write normals to binding 12 SSBO */
    uint32_t curve_fast;       /* 1 = skip curve AO/self-shadow secondary rays
                                * (mesh shadows unaffected) — for dense curve
                                * scenes where 6 secondary rays/hit through a
                                * huge curve BVH would stall the dispatch */
    float    tone_exposure_scale; /* runtime exposure multiplier (NUSD_EXPOSURE_SCALE,
                                   * default 1.0); calibration knob, single-camera path */
    float    jitter_x;      /* sub-pixel camera jitter in pixel units */
    float    jitter_y;
} GpuRtPushConstants;

/* ---- DLSS types ---- */

/* Push constant layout for DLSS raster: MVP + model + color + eye + prev_mvp = 224 bytes */
typedef struct {
    float mvp[16];        /* current jittered MVP */
    float model[16];      /* world transform (for normals) */
    float color[4];       /* .w > 0.5 → override color */
    float eye_pos[4];     /* camera world position (xyz), .w unused */
    float prev_mvp[16];   /* previous frame's unjittered MVP (for motion vectors) */
} GpuMeshPushConstantsDlss;

/* RT push constants for DLSS: + prev VP + jitter + scene info = 208 bytes */
typedef struct {
    float view_inv[16];   /* current frame inverse view */
    float proj_inv[16];   /* current frame inverse projection (jittered) */
    float prev_vp[16];    /* previous frame's unjittered view-projection */
    float jitter[2];      /* pixel-space jitter offsets */
    float ground_y;       /* ground plane Y coordinate */
    float scene_scale;    /* scene diagonal for ground checker scale */
} GpuRtPushConstantsDlss;

/* ---- Lifecycle ---- */

Gpu*  gpu_init(void* glfw_window, int width, int height);
void  gpu_shutdown(Gpu* gpu);
void  gpu_resize(Gpu* gpu, int width, int height);

/* Enable headless mode: skip vkQueuePresentKHR in end_frame calls.
 * Swapchain images are still used as render targets but not displayed.
 * Call after gpu_init(), before first frame. */
void  gpu_set_headless(Gpu* gpu, int headless);

/* ---- Resources ---- */

GpuBuffer   gpu_create_buffer(Gpu* gpu, const GpuBufferDesc* desc);
GpuBuffer   gpu_create_private_buffer(Gpu* gpu, GpuBufferUsage usage,
                                       uint64_t size);
void        gpu_destroy_buffer(Gpu* gpu, GpuBuffer buf);
void*       gpu_buffer_contents(GpuBuffer buf);
int         gpu_update_buffer(Gpu* gpu, GpuBuffer buf, const void* data,
                              uint64_t size, uint64_t offset);
int         gpu_copy_buffer(Gpu* gpu, GpuBuffer src, uint64_t src_offset,
                            GpuBuffer dst, uint64_t dst_offset,
                            uint64_t size);

GpuPipeline gpu_create_pipeline(Gpu* gpu, const GpuPipelineDesc* desc);
GpuPipeline gpu_create_shadow_pipeline(Gpu* gpu, const GpuPipelineDesc* desc);
void        gpu_destroy_pipeline(Gpu* gpu, GpuPipeline pipe);

/* ---- Frame ---- */

/* Returns 1 if frame started OK, 0 if swapchain needs resize */
int   gpu_begin_frame(Gpu* gpu);
void  gpu_end_frame(Gpu* gpu);

/* ---- Draw commands (valid between begin_frame / end_frame) ---- */

void  gpu_cmd_bind_pipeline(Gpu* gpu, GpuPipeline pipe);
void  gpu_cmd_bind_shadow(Gpu* gpu);   /* bind TLAS descriptor for ray query shadows */
void  gpu_cmd_bind_vertex_buffer(Gpu* gpu, GpuBuffer buf);
void  gpu_cmd_bind_instance_buffer(Gpu* gpu, GpuBuffer buf);
void  gpu_cmd_bind_index_buffer(Gpu* gpu, GpuBuffer buf);
void  gpu_cmd_push_constants(Gpu* gpu, const void* data, uint32_t size);
void  gpu_cmd_draw(Gpu* gpu, uint32_t vertex_count, uint32_t first_vertex);
void  gpu_cmd_draw_indexed(Gpu* gpu, uint32_t index_count,
                           uint32_t first_index, int32_t vertex_offset);
void  gpu_cmd_draw_indexed_instanced(Gpu* gpu, uint32_t index_count,
                                     uint32_t first_index, int32_t vertex_offset,
                                     uint32_t instance_count, uint32_t first_instance);
int   gpu_cmd_draw_curves(Gpu* gpu, const float* vp, const float* eye);

/* ---- Materials ---- */

/* Texture image data for GPU upload */
typedef struct {
    const unsigned char* pixels;  /* RGBA8 */
    int width;
    int height;
    int is_srgb;  /* 1 = color data (sRGB), 0 = linear data (roughness, normal, etc.) */
} GpuTextureData;

typedef struct {
    int   op;
    int   type;
    int   in0;
    int   in1;
    float value[4];
    int   in2;
    int   in3;
    int   _pad0;
    int   _pad1;
} GpuMtlxProcNode;

/* Material parameters (matches MaterialParams in material.h, GPU-uploadable).
 * Phase 7c trailing block keeps the older head-of-struct binary-compatible
 * with anything that only reads UsdPreviewSurface fields. */
typedef struct {
    float base_color[4];
    float emissive_color[4];
    float metallic;
    float roughness;
    float opacity;
    float ior;
    float occlusion;
    float clearcoat;
    float clearcoat_roughness;
    float normal_scale;
    int   tex_indices[8];
    int   use_vertex_color;
    float udim_scale_u;
    float udim_scale_v;
    float opacity_threshold;     /* UPS alpha-cutout threshold; 0 = disabled */
    /* Phase 7c: Standard Surface */
    float subsurface_color[4];
    float subsurface_radius[4];
    float transmission_color[4];
    float subsurface_weight;
    float subsurface_scale;
    float transmission_weight;
    float transmission_ior;
    int   tex_subsurface_weight;
    int   tex_transmission_weight;
    int   sss_color_authored;    /* 1 = constant authored, 0 = follow baseColor */
    int   use_specular_workflow; /* 1 iff UsdPreviewSurface useSpecularWorkflow=1 */
    float specular_color[4];     /* rgb (linear), w unused */
    float normal_tex_scale[4];   /* TEX_NORMAL UsdUVTexture scale */
    float normal_tex_bias[4];    /* TEX_NORMAL UsdUVTexture bias */
    float uv_transform[4];       /* xy texture scale, zw texture offset */
    float roughness_tex_transform[4]; /* x scale, y bias for sampled roughness */
    int   v_flip;                /* 1 = sample texture V as 1-v for MDL */
    int   _pad_v_flip[3];
    float base_weight;           /* Standard Surface base input */
    float specular_weight;       /* Standard Surface specular input */
    float sheen_weight;          /* Standard Surface sheen input */
    float sheen_roughness;       /* Standard Surface sheen_roughness input */
    float sheen_color[4];        /* Standard Surface sheen_color input */
    float thin_film_thickness;   /* nanometers; 0 disables */
    float thin_film_ior;
    float specular_anisotropy;
    int   standard_surface_lobes;/* 1 iff Standard Surface extras are authored */
    int   procedural_kind;       /* 0 none, 1 MaterialX marble3d graph */
    int   procedural_base_color; /* 1 iff base_color is graph-driven */
    int   procedural_subsurface_color; /* 1 iff subsurface_color is graph-driven */
    int   procedural_octaves;
    float procedural_color1[4];
    float procedural_color2[4];
    float procedural_params[4];  /* scale1, scale2, power, noise_amp */
    int   procedural_node_count;
    int   procedural_base_color_output;
    int   procedural_subsurface_color_output;
    int   procedural_roughness_output;
    int   procedural_normal_output;
    int   procedural_graph_flags;
    int   procedural_graph_pad0;
    int   procedural_graph_pad1;
    GpuMtlxProcNode procedural_nodes[MAX_MTLX_PROC_NODES];
} GpuMaterialParams;

/* Upload material data (SSBO + textures) to GPU.
 * Returns 1 on success. Must be called before creating material pipelines. */
int  gpu_upload_materials(Gpu* gpu,
                          const GpuMaterialParams* materials, int nmaterials,
                          const GpuTextureData* textures, int ntextures);

/* GPU-side scene light layout — matches std430 packing in
 * raytrace.rchit.glsl. Each vec3 packs with the following scalar so the
 * struct stays 16-byte aligned without explicit padding fields. */
#ifndef GPU_MAX_SCENE_LIGHTS
#define GPU_MAX_SCENE_LIGHTS 64
#endif
typedef struct {
    float position[3];   /* RectLight: rect center; DistantLight: unused */
    float intensity;     /* inputs:intensity * 2^exposure */
    float normal[3];     /* emit direction (unit vector) */
    int   kind;          /* 0 = rect, 1 = distant */
    float u_axis[3];     /* RectLight: world half-extent along light +X */
    int   normalize;     /* inputs:normalize: 1 = power, 0 = radiance */
    float v_axis[3];     /* RectLight: world half-extent along light +Y */
    float angle_deg;     /* DistantLight cone half-angle */
    float color[3];      /* linear RGB tint */
    float _pad;
} GpuLight;

/* Upload scene light buffer to GPU (binding 13 of tiled RT descriptor set).
 * Call after gpu_upload_materials() and before gpu_build_tiled_rt_pipeline(). */
int  gpu_upload_lights(Gpu* gpu, const GpuLight* lights, int nlights);

/* Phase 11.A: upload BasisCurves segment data + AABBs + per-segment colors.
 * - segments  : SceneCurveSegment[seg_count]    (32 B/seg)
 * - aabbs     : SceneCurveAabb[seg_count]       (24 B/AABB; min/max float3)
 * - colors    : float[seg_count * 3]            (RGB; padded to vec4 internally)
 * Pass seg_count == 0 to free the existing upload. Three Shared MTLBuffers
 * are created (one per input array). The AABB buffer is sized to match
 * Metal's MTLAxisAlignedBoundingBox stride (24 B). The two SSBO-equivalents
 * are read by the curve intersection function (intersection in Metal terms)
 * and the curve hit branch in the RT compute kernel.
 *
 * Does NOT build the BLAS; call gpu_build_curve_blas separately. */
int  gpu_upload_curve_data(Gpu* gpu,
                           const void* segments,
                           const void* aabbs,
                           const float* colors,
                           int seg_count);

/* Phase 11.A.2.3: build a single bounding-box-geometry BLAS
 * (MTLAccelerationStructureBoundingBoxGeometryDescriptor) over the AABBs
 * uploaded by gpu_upload_curve_data. No-op if no curve data is uploaded.
 * Returns 1 on success.
 *
 * The BLAS is independent of the mesh BLAS pool. Wiring it into the TLAS
 * as an additional instance with intersectionFunctionTableOffset set
 * happens later when the IFT plumbing lands. */
int  gpu_build_curve_blas(Gpu* gpu);
int  gpu_curve_blas_built(Gpu* gpu);

/* Create a pipeline for material rendering (uses material descriptor set).
 * Same vertex format as regular pipeline but with UV and material ID. */
GpuPipeline gpu_create_material_pipeline(Gpu* gpu, const GpuPipelineDesc* desc);

/* Bind the material descriptor set (SSBO + textures) for drawing. */
void gpu_cmd_bind_materials(Gpu* gpu);

/* Destroy material GPU resources. */
void gpu_destroy_materials(Gpu* gpu);

/* ---- Environment / IBL ---- */

/* Load an HDR environment map and generate BRDF integration LUT.
 * Returns 1 on success. Must be called after gpu_upload_materials()
 * so that IBL textures are added to the material descriptor set. */
int   gpu_load_environment(Gpu* gpu, const char* hdr_path);

/* Stamp the USD-authored DomeLight intensity onto the SceneHeader.
 * 1.0 = no scaling (default). Higher values brighten the sky path
 * via sqrt-soft-compression in shade_miss. */
void  gpu_set_env_intensity(Gpu* gpu, float intensity);
void  gpu_set_dome_color(Gpu* gpu, float r, float g, float b, float intensity);

/* Destroy IBL resources (env map + BRDF LUT). */
void  gpu_destroy_environment(Gpu* gpu);

/* Query number of environment map mip levels (0 = no IBL loaded). */
int   gpu_get_env_mip_levels(Gpu* gpu);

/* Create a pipeline for drawing the environment map as background. */
int   gpu_create_env_bg_pipeline(Gpu* gpu);

/* Draw the environment map as fullscreen background.
 * view_inv and proj_inv are 4x4 row-major inverse matrices. */
void  gpu_draw_env_background(Gpu* gpu, const float* view_inv, const float* proj_inv);

/* SSAO post-process (Metal): darken the rendered color in contacts using the
 * stored depth buffer. Call after the raster frame ends. params (8 floats) =
 * { near, far, world_radius, strength, bias, power, 1/width, 1/height }. */
void  gpu_ssao_composite(Gpu* gpu, const float* params);

/* ---- Screenshot / Readback ---- */

/* Read back the last presented swapchain image and write to a PPM file.
 * Must be called after gpu_end_frame(). Returns 1 on success. */
int   gpu_screenshot(Gpu* gpu, const char* path);

/* Read back the last presented swapchain image into a caller-provided buffer.
 * out_rgba8 must hold width*height*4 bytes. Swapchain BGRA is swizzled to RGBA.
 * Must be called after gpu_end_frame(). Returns 1 on success. */
int   gpu_readback_pixels(Gpu* gpu, uint8_t* out_rgba8, uint32_t width, uint32_t height);
int   gpu_readback_pixels_f32(Gpu* gpu, float* out_rgba, uint32_t width, uint32_t height);

/* ---- Ray tracing ---- */

/* Returns 1 if hardware ray tracing is available on this GPU. */
int   gpu_rt_available(Gpu* gpu);

/* Returns 1 if the RT scene (TLAS + BLASes) was successfully built. */
int   gpu_rt_built(Gpu* gpu);

/* Per-mesh descriptor for RT acceleration structure build.
 * Unique prototypes get their own BLAS; instances reference their prototype's BLAS.
 * All composed into one TLAS with per-instance world transforms. */
typedef struct {
    GpuBuffer  vertex_buf;
    uint32_t   vertex_count;
    uint32_t   vertex_stride;   /* bytes per vertex (e.g. 36) */
    GpuBuffer  index_buf;
    uint32_t   index_count;
    uint32_t   vertex_offset;   /* first vertex in the shared buffer */
    uint32_t   index_offset;    /* first index in the shared buffer */
    int32_t    prototype_idx;   /* index of prototype mesh; == own index if unique */
    float      transform[12];   /* 3x4 row-major world transform (VkTransformMatrixKHR) */
    float      color[3];        /* per-mesh display color (for RT shading) */
    int32_t    material_index;  /* index into materials[] SSBO; -1 = no material */
    uint8_t    mask;            /* TLAS instance mask; 0 hides from primary rays */
} GpuRtMeshDesc;

/* Build one BLAS per mesh + TLAS + RT pipeline.
 * Returns 1 on success. */
int   gpu_build_rt_scene(Gpu* gpu,
                         const GpuRtMeshDesc* meshes, uint32_t nmeshes,
                         const uint32_t* rgen_spv, uint32_t rgen_size,
                         const uint32_t* miss_spv, uint32_t miss_size,
                         const uint32_t* chit_spv, uint32_t chit_size);
void  gpu_destroy_rt_scene(Gpu* gpu);

/* 1 if a prior gpu_build_rt_scene left a reusable BLAS set cached
 * (camera-invariant geometry). Used to gate gpu_rebuild_rt_tlas. */
int   gpu_rt_can_reuse_blas(Gpu* gpu);

/* Rebuild ONLY the TLAS + instance/scene-data buffers, reusing the cached
 * per-mesh BLAS from the previous gpu_build_rt_scene. Valid only when the
 * base-mesh geometry is unchanged (culled-proxy: only the visible PI-clone
 * tail and the camera vary frame-to-frame). nmeshes is the new total
 * descriptor count (base + visible clones [+ curve]). The recomputed unique
 * BLAS count must match the cached count, else returns 0 so the caller can
 * fall back to a full gpu_build_rt_scene. Returns 1 on success. */
int   gpu_rebuild_rt_tlas(Gpu* gpu, const GpuRtMeshDesc* meshes, uint32_t nmeshes);

/* Update TLAS instance transforms without rebuilding BLASes.
 * transforms is [nmeshes][12] row-major 3x4 (VkTransformMatrixKHR format).
 * nmeshes must match the count from the original gpu_build_rt_scene() call.
 * masks (optional, may be NULL) is [nmeshes] uint8 — 0 hides.
 * Returns 1 on success. */
int   gpu_update_tlas(Gpu* gpu, const float* transforms, const uint8_t* masks, uint32_t nmeshes);

/* Update per-mesh display colors in the scene data SSBO without rebuilding
 * BLASes or TLAS.  colors is [nmeshes][3] float RGB.
 * Returns 1 on success. */
int   gpu_update_scene_colors(Gpu* gpu, const float* colors, uint32_t nmeshes);

/* Inline TLAS update: upload instances + record TLAS build into the CURRENT
 * command buffer (gpu->current_cmd), adding a barrier before the trace dispatch.
 * Call between gpu_begin_frame_tiled_rt() and gpu_cmd_trace_rays_tiled().
 * masks (optional, may be NULL) is [nmeshes] uint8 — 0 hides.
 * Returns 1 on success, 0 on error (falls back to gpu_update_tlas). */
int   gpu_update_tlas_inline(Gpu* gpu, const float* transforms, const uint8_t* masks, uint32_t nmeshes);

/* Abort an in-progress tiled frame (discard the command buffer). */
void  gpu_abort_frame_tiled_rt(Gpu* gpu);

/* RT frame (alternative to raster begin_frame/end_frame) */
int   gpu_begin_frame_rt(Gpu* gpu);
void  gpu_cmd_trace_rays(Gpu* gpu, const GpuRtPushConstants* pc);
void  gpu_end_frame_rt(Gpu* gpu);

/* ---- Gaussian splat RT ---- */

int   gpu_gs_upload_particles(Gpu* gpu,
                              const float* positions,
                              const float* scales,
                              const float* orientations,
                              const float* opacities,
                              const float* kernel_scales,
                              const float* sh_coefficients,
                              uint32_t particle_count,
                              uint32_t sh_degree,
                              const float prim_xform[16]);
int   gpu_gs_build_accel(Gpu* gpu);
int   gpu_gs_render(Gpu* gpu,
                    const float vp_inv[32],
                    uint32_t sh_degree,
                    uint32_t k,
                    uint32_t max_passes,
                    float min_transmittance,
                    float iso_opacity_threshold,
                    uint32_t color_space);
int   gpu_gs_available(Gpu* gpu);
int   gpu_gs_particle_count(Gpu* gpu);
int   gpu_gs_fetch_depth(Gpu* gpu, float* out_depth,
                         uint32_t width, uint32_t height);
int   gpu_gs_fetch_normal(Gpu* gpu, float* out_normal,
                          uint32_t width, uint32_t height);
void  gpu_gs_destroy(Gpu* gpu);

/* ---- Tiled multi-camera RT ---- */

/* Push constants for tiled RT.
 * Layout matches GpuRtPushConstants (148 bytes) so that the shared miss/hit
 * shaders can read ground_y and scene_scale at the same offsets (128, 132).
 * The first 16 bytes overlap with view_inv[0..3] but are reinterpreted as
 * tile parameters by the tiled raygen shader via floatBitsToUint(). */
typedef struct {
    uint32_t tile_w;        /* offset 0:  per-camera width in pixels */
    uint32_t tile_h;        /* offset 4:  per-camera height in pixels */
    uint32_t num_cols;      /* offset 8:  number of tile columns in the grid */
    uint32_t num_cameras;   /* offset 12: total number of cameras */
    float    _pad[28];      /* offset 16-127: padding to align ground_y */
    float    ground_y;      /* offset 128: ground plane Y (read by miss shader) */
    float    scene_scale;   /* offset 132: scene diagonal (read by miss shader) */
    uint32_t fast_mode;     /* offset 136: 1 = skip shadow rays for RL sensors */
    uint32_t depth_enabled; /* offset 140: 1 = write depth to binding 10 SSBO */
    uint32_t segmentation_enabled; /* offset 144: 1 = write IDs to binding 11 SSBO */
    uint32_t normals_enabled;     /* offset 148: 1 = write normals to binding 12 SSBO */
    uint32_t curve_fast;          /* offset 152: 1 = skip curve secondary rays */
} GpuRtTiledPushConstants;

/* Create/resize the tiled storage image and camera SSBO.
 * total_w x total_h is the full tiled image size.
 * Returns 1 on success. */
int   gpu_tiled_init(Gpu* gpu, uint32_t total_w, uint32_t total_h,
                     int num_cameras);

/* Upload camera inverse VP matrices to the camera SSBO.
 * data is num_cameras * 32 floats: pairs of (view_inv[16], proj_inv[16]).
 * Returns 1 on success. */
int   gpu_tiled_upload_cameras(Gpu* gpu, const float* data, int num_cameras);

/* Build the tiled RT pipeline (separate from single-camera RT pipeline).
 * Uses a different ray gen shader with camera SSBO + tile indexing.
 * Returns 1 on success. */
int   gpu_build_tiled_rt_pipeline(Gpu* gpu,
                                   const uint32_t* rgen_spv, uint32_t rgen_size,
                                   const uint32_t* miss_spv, uint32_t miss_size,
                                   const uint32_t* chit_spv, uint32_t chit_size);

/* Tiled RT frame: trace rays at full tiled resolution. */
int   gpu_begin_frame_tiled_rt(Gpu* gpu);
void  gpu_cmd_trace_rays_tiled(Gpu* gpu, const GpuRtTiledPushConstants* pc);
void  gpu_end_frame_tiled_rt(Gpu* gpu);

/* Read back the tiled storage image into CPU buffer.
 * out_rgba8 must hold total_w * total_h * 4 bytes.
 * Returns 1 on success. */
int   gpu_readback_tiled_pixels(Gpu* gpu, uint8_t* out_rgba8,
                                 uint32_t total_w, uint32_t total_h);

/* Read back tiled depth buffer into CPU float array.
 * out_depth must hold total_w * total_h floats (ray T distances).
 * Values: positive = hit distance, -1.0 = miss/sky.
 * Returns 1 on success. */
int   gpu_readback_tiled_depth(Gpu* gpu, float* out_depth,
                                uint32_t total_w, uint32_t total_h);

/* Read back segmentation IDs from the tiled RT pipeline.
 * Output is uint32_t[total_w * total_h]. Values: mesh_id+1 = hit, 0 = miss/sky.
 * Returns 1 on success. */
int   gpu_readback_tiled_segmentation(Gpu* gpu, uint32_t* out_ids,
                                       uint32_t total_w, uint32_t total_h);

/* Read back normals from the tiled RT pipeline.
 * Output is float[total_w * total_h * 3]. Layout: (x,y,z) per pixel.
 * (0,0,0) for sky miss, (0,1,0) for ground.
 * Returns 1 on success. */
int   gpu_readback_tiled_normals(Gpu* gpu, float* out_normals,
                                  uint32_t total_w, uint32_t total_h);

/* Copy tiled image to staging and return a pointer to the mapped staging buffer.
 * The pointer is valid until the next call to any gpu_readback/map function.
 * total_w/total_h must match the tiled image dimensions.
 * Returns NULL on failure. */
const uint8_t* gpu_map_tiled_staging(Gpu* gpu, uint32_t total_w, uint32_t total_h);

/* Map a specific slot of the double-buffered tiled staging. slot must be 0 or 1.
 * Used by async readback: caller tracks which slot holds its pending render and
 * reads from it on the next frame, overlapping the GPU wait with CPU work.
 * Returns NULL if the slot has no in-flight submission or dims mismatch. */
const uint8_t* gpu_map_tiled_staging_slot(Gpu* gpu, uint32_t total_w,
                                          uint32_t total_h, int slot);

/* Return the slot index (0 or 1) that the most recent render_tiled wrote to.
 * Call immediately after render_tiled to record the pending-read slot. */
int gpu_get_last_tiled_slot(Gpu* gpu);

/* Wait for the most recent tiled render to complete on the GPU.
 * After this, the interop buffer (exported fd) contains valid pixel data.
 * Cheaper than gpu_map_tiled_staging — no staging pointer setup. */
int gpu_wait_tiled_complete(Gpu* gpu);

/* Wait for the PREVIOUS frame's tiled render to complete.
 * Used for double-buffered overlap: submit frame N, then wait for frame N-1
 * (which is usually already done). Returns 0 if no previous frame exists. */
int gpu_wait_previous_tiled_complete(Gpu* gpu);

/* Return the interop buffer index containing the PREVIOUS frame's data.
 * For double-buffered overlap: read this buffer while the current frame renders. */
int gpu_get_interop_prev_idx(Gpu* gpu);

/* ---- CUDA Interop ---- */

/* Returns 1 if CUDA-Vulkan interop is available (external memory fd + semaphore fd). */
int   gpu_interop_available(Gpu* gpu);

/* Export a tiled interop buffer's VkDeviceMemory as a POSIX file descriptor.
 * buf_idx: 0 or 1 (double-buffered interop buffers).
 * The fd is a one-time-use handle — caller must close it or import it.
 * Returns fd >= 0 on success, -1 on failure. */
int   gpu_export_tiled_image_fd(Gpu* gpu, int buf_idx);

/* Return the interop buffer index that contains the most recently completed
 * frame's data (the "read" buffer — the one NOT currently being written). */
int   gpu_get_interop_read_idx(Gpu* gpu);

/* Export the interop timeline semaphore as a POSIX file descriptor.
 * Returns fd >= 0 on success, -1 on failure. */
int   gpu_export_timeline_semaphore_fd(Gpu* gpu);

/* Get the tiled image allocation size in bytes (needed for CUDA import). */
uint64_t gpu_get_tiled_image_alloc_size(Gpu* gpu);

/* Get the current timeline semaphore value (incremented each render). */
uint64_t gpu_get_interop_timeline_value(Gpu* gpu);

/* Enable/disable CPU staging copy.  When skip=1 and interop is active,
 * the staging buffer allocation and image→staging copy are skipped entirely,
 * saving one full image copy worth of GPU bandwidth per frame. */
void gpu_set_skip_staging_copy(Gpu* gpu, int skip);

/* Enable/disable direct buffer write mode.  When enabled, the raygen shader
 * writes pixels directly to interop_buf[0] via SSBO (binding 9), skipping
 * the imageStore + vkCmdCopyImageToBuffer steps entirely. */
void gpu_set_direct_write(Gpu* gpu, int enable);
int  gpu_is_direct_write(Gpu* gpu);

/* ---- DLSS ---- */

/* Returns 1 if DLSS is available on this GPU.
 * Only valid after gpu_init() + gpu_build_rt_scene(). */
int   gpu_dlss_available(Gpu* gpu);

/* Initialize DLSS at the given quality mode.
 * Creates offscreen render targets and DLSS feature.
 * Returns 1 on success. */
int   gpu_dlss_init(Gpu* gpu, int quality_mode);

/* Shut down DLSS and free offscreen resources. */
void  gpu_dlss_shutdown(Gpu* gpu);

/* Change DLSS quality mode. Recreates the feature at new render resolution. */
void  gpu_dlss_set_quality(Gpu* gpu, int quality_mode);

/* Query the current render resolution (differs from display when DLSS active). */
void  gpu_get_render_extent(Gpu* gpu, uint32_t* w, uint32_t* h);

/* Metal-only direct texture accessors. Return id<MTLTexture> as void* (cast
 * via __bridge in the Metal backend). NULL on Vulkan + before first frame
 * on Metal. Owned by Gpu — caller must not release. Used by external
 * Metal-aware consumers (nanousd-metal-viewer's gizmo overlay) that want
 * to bind the renderer's color/depth targets directly without paying the
 * gpu_screenshot CPU round-trip. */
void* gpu_get_metal_color_texture(Gpu* gpu);
void* gpu_get_metal_depth_texture(Gpu* gpu);

/* Raster + DLSS frame: renders to offscreen targets at render resolution.
 * Returns 1 if frame started, 0 if swapchain needs resize. */
int   gpu_begin_frame_dlss(Gpu* gpu);
void  gpu_end_frame_dlss(Gpu* gpu, float jitter_x, float jitter_y,
                          float dt_ms, int reset);

/* RT + DLSS frame: traces rays to offscreen targets at render resolution. */
int   gpu_begin_frame_rt_dlss(Gpu* gpu);
void  gpu_cmd_trace_rays_dlss(Gpu* gpu, const GpuRtPushConstantsDlss* pc);
void  gpu_end_frame_rt_dlss(Gpu* gpu, float jitter_x, float jitter_y,
                             float dt_ms, int reset);

/* Build RT scene with DLSS variant shaders. */
int   gpu_build_rt_scene_dlss(Gpu* gpu,
                               const GpuRtMeshDesc* meshes, uint32_t nmeshes,
                               const uint32_t* rgen_spv, uint32_t rgen_size,
                               const uint32_t* miss_spv, uint32_t miss_size,
                               const uint32_t* chit_spv, uint32_t chit_size);

/* Create pipeline for DLSS raster (MRT: color + motion vectors). */
GpuPipeline gpu_create_dlss_pipeline(Gpu* gpu, const GpuPipelineDesc* desc);
GpuPipeline gpu_create_dlss_shadow_pipeline(Gpu* gpu, const GpuPipelineDesc* desc);

/* ---- Text overlay ---- */

/* Initialize overlay rendering (font texture, pipeline, etc.).
 * Called once after gpu_init(). Returns 1 on success. */
int   gpu_overlay_init(Gpu* gpu);

/* Destroy overlay resources. */
void  gpu_overlay_shutdown(Gpu* gpu);

/* Queue a text string for rendering at (x, y) in screen pixels (top-left origin).
 * Color is RGBA [0-1]. Scale multiplies the base font size. */
void  gpu_overlay_text(Gpu* gpu, float x, float y, float scale,
                       float r, float g, float b, float a,
                       const char* text);

/* Queue a filled rectangle for the overlay (e.g. background panel).
 * Position and size in screen pixels (top-left origin). Color is RGBA [0-1]. */
void  gpu_overlay_rect(Gpu* gpu, float x, float y, float w, float h,
                       float r, float g, float b, float a);

/* Render all queued text and clear the queue.
 * Called internally by gpu_end_frame* functions. */
void  gpu_overlay_flush(Gpu* gpu);

/* ---- Raycast compute (LiDAR/radar) ---- */

/* Push constants for raycast compute shader */
typedef struct {
    uint32_t num_rays;
    float    max_distance;
} GpuRaycastPushConstants;

/* Build the raycast compute pipeline.  Uses a compute shader with
 * GL_EXT_ray_query to trace arbitrary rays against the TLAS.
 * comp_spv/comp_size: compiled SPIR-V for the compute shader.
 * Returns 1 on success. */
int   gpu_build_raycast_pipeline(Gpu* gpu,
                                  const uint32_t* comp_spv, uint32_t comp_size);

/* Destroy raycast pipeline and associated resources. */
void  gpu_destroy_raycast_pipeline(Gpu* gpu);

/* Cast rays against the scene TLAS.
 * origins:     [num_rays * 3] float — world-space ray origins
 * directions:  [num_rays * 3] float — normalized ray directions
 * num_rays:    number of rays
 * max_distance: maximum ray distance
 * out_distances:     [num_rays] float — hit distance (-1 = miss)
 * out_normals:       [num_rays * 3] float — approximate surface normal
 * out_hit_positions: [num_rays * 3] float — world-space hit position
 * Returns 1 on success. */
int   gpu_cast_rays(Gpu* gpu,
                    const float* origins, const float* directions,
                    int num_rays, float max_distance,
                    float* out_distances, float* out_normals,
                    float* out_hit_positions);

/* Async raycast: submit dispatch without blocking. */
int   gpu_cast_rays_async(Gpu* gpu,
                          const float* origins, const float* directions,
                          int num_rays, float max_distance);

/* Async raycast: wait for pending dispatch and read results. */
int   gpu_cast_rays_wait(Gpu* gpu,
                         float* out_distances, float* out_normals,
                         float* out_hit_positions);

/* Raycast CUDA interop: create exportable buffers and return opaque fds. */
int   gpu_raycast_get_interop_fds(Gpu* gpu, int num_rays,
                                   int* out_input_fd, uint64_t* out_input_size,
                                   int* out_output_fd, uint64_t* out_output_size,
                                   uint32_t* out_max_rays);

/* GPU-only raycast dispatch: data already in device-local buffer via CUDA. */
int   gpu_cast_rays_gpu(Gpu* gpu, int num_rays, float max_distance);

/* Wait for async raycast fence only (no staging readback). For GPU interop path. */
int   gpu_cast_rays_wait_fence(Gpu* gpu);

/* Query GPU memory allocated by the viewer (bytes). */
uint64_t gpu_get_allocated_memory(Gpu* gpu);

/* Query total device-local heap size (bytes). */
uint64_t gpu_get_heap_size(Gpu* gpu);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_GPU_H */
