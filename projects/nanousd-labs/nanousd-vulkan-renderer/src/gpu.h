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
    GPU_FORMAT_FLOAT4 = 4,  /* VK_FORMAT_R32G32B32A32_SFLOAT (instance mat columns) */
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
    const uint32_t*     tesc_spv;
    uint32_t            tesc_size;    /* bytes */
    const uint32_t*     tese_spv;
    uint32_t            tese_size;    /* bytes */
    const uint32_t*     frag_spv;
    uint32_t            frag_size;    /* bytes */
    uint32_t            push_constant_size;
    uint32_t            vertex_stride;
    uint32_t            patch_control_points;
    const GpuVertexAttrib* attribs;
    uint32_t            nattribs;
    /* When non-zero, gpu_create_pipeline uses gpu->mat_pipeline_layout
     * (push-constants + mat_desc_layout @ set 0) instead of the bare
     * gpu->pipeline_layout. Used by the textured-raster pipeline so the
     * fragment shader can sample the material UBO + texture array. */
    int                 use_mat_layout;
    /* Optional second vertex binding (binding 1) at VK_VERTEX_INPUT_RATE_INSTANCE.
     * Used by the instanced raster pipeline to feed each PointInstancer placement
     * its world matrix as four vec4 columns. Zero stride = no instance binding. */
    uint32_t            instance_stride;
    const GpuVertexAttrib* instance_attribs;
    uint32_t            n_instance_attribs;
} GpuPipelineDesc;

/* Push constant layout:
 * MVP + model + color + eye + ibl_params + tone_params = 192 bytes. */
typedef struct {
    float mvp[16];     /* view-projection * model (for position transform) */
    float model[16];   /* world transform (for normal/position to world space) */
    float color[4];    /* .w > 0.5 → use this color; else use vertex color */
    float eye_pos[4];  /* camera world position (xyz); .w = per-draw raster
                        * material id as int bits (floatBitsToInt), clamped to
                        * >=0 like the legacy per-vertex matID. Lets the textured
                        * frag take material per-instance instead of per-vertex,
                        * so content-hash dedup can share geometry across
                        * materials. */
    /* IBL gating + parameters consumed by mesh.frag.glsl. The raster shader
     * historically only had a procedural sky/ground hemisphere; with these
     * fields it samples the same env+irr+brdf textures bound at set 0
     * bindings 2/3/4 as the RT path, closing the parity gap on Kitchen_set's
     * dimly-ambient ceiling underside (#44).
     *   x = has_ibl    (1.0 if env loaded, else 0.0 → procedural fallback)
     *   y = mip_levels (env prefilter mip count, used for roughness LOD)
     *   z = intensity  (USD DomeLight intensity multiplier; matches rmiss)
     *   w = unused (reserved)
     */
    float ibl_params[4];
    /* OVRTX-style exposure/tonemap scale.
     *   x = surface exposure multiplier (1.0 = legacy calibrated output)
     *   y = visible-sky exposure multiplier
     *   z = white point scale (reserved for future auto-exposure)
     *   w = flags as floatBitsToUint-compatible storage (currently unused)
     */
    float tone_params[4];
} GpuMeshPushConstants;

/* RT push constants: inverse view + inverse projection + scene info +
 * exposure/tone metadata = 176 bytes. */
typedef struct {
    float view_inv[16];
    float proj_inv[16];
    float ground_y;       /* historical name: Z-up ground plane height */
    float scene_scale;    /* scene diagonal for ground checker scale */
    uint32_t fast_mode;   /* 1 = skip shadow rays + PBR for RL sensors */
    uint32_t depth_enabled; /* 1 = write depth to binding 10 SSBO */
    uint32_t segmentation_enabled; /* 1 = write instance IDs to binding 11 SSBO */
    uint32_t normals_enabled;  /* 1 = write normals to binding 12 SSBO */
    /* When 1, rchit/rmiss write the G-buffer SSBO at binding 17. Single-camera
     * RT leaves this off unless NUSD_RT_GBUFFER=1 opts into diagnostics; tiled
     * RT enables it only after allocating the tiled G-buffer used by the
     * deferred compute pass. */
    uint32_t deferred_shade_enabled; /* offset 152 */
    float    tone_exposure_scale;    /* offset 156: 1.0 = legacy calibrated output */
    float    tone_sky_scale;         /* offset 160: visible sky/background scale */
    float    tone_white_point;       /* offset 164: reserved for auto-exposure */
    uint32_t tone_flags;             /* offset 168 */
    float    rt_ibl_fill_scale;      /* offset 172: IBL multiplier when explicit lights exist */
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
    float ground_y;       /* historical name: Z-up ground plane height */
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
void        gpu_destroy_buffer(Gpu* gpu, GpuBuffer buf);

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
void  gpu_cmd_bind_index_buffer(Gpu* gpu, GpuBuffer buf);
void  gpu_cmd_bind_index_buffer_typed(Gpu* gpu, GpuBuffer buf,
                                      uint64_t offset_bytes,
                                      int index_type_bits);
void  gpu_cmd_push_constants(Gpu* gpu, const void* data, uint32_t size);
void  gpu_cmd_draw(Gpu* gpu, uint32_t vertex_count, uint32_t first_vertex);
void  gpu_cmd_draw_indexed(Gpu* gpu, uint32_t index_count,
                           uint32_t first_index, int32_t vertex_offset);
/* Bind a per-instance vertex buffer at binding 1 (instance-rate attributes). */
void  gpu_cmd_bind_instance_buffer(Gpu* gpu, GpuBuffer buf);
/* Instanced indexed draw — instance_count placements starting at first_instance
 * (so instance-rate attributes are fetched from buf[first_instance + i]). */
void  gpu_cmd_draw_indexed_instanced(Gpu* gpu, uint32_t index_count,
                                     uint32_t instance_count,
                                     uint32_t first_index, int32_t vertex_offset,
                                     uint32_t first_instance);

/* ---- Materials ---- */

/* Texture image data for GPU upload */
typedef struct {
    const unsigned char* pixels;  /* RGBA8 */
    int width;
    int height;
    int is_srgb;  /* 1 = color data (sRGB), 0 = linear data (roughness, normal, etc.) */
} GpuTextureData;

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
    float opacity_threshold;  /* UPS alpha-cutout threshold; 0 = disabled */
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
    int   sss_color_authored;     /* 1 iff loader wrote subsurface_color */
    int   use_specular_workflow;  /* 1 iff useSpecularWorkflow=1 (UPS) */
    float specular_color[4];      /* rgb (linear), w unused */
    /* TEX_NORMAL UsdUVTexture inputs:scale and inputs:bias.
     * Defaults (2,2,2,1) / (-1,-1,-1,0) reproduce the prior implicit
     * `nm = nm * 2 - 1` so unauthored scenes are bit-identical. */
    float normal_tex_scale[4];
    float normal_tex_bias[4];
    float mdl_uv_transform[4];     /* xy scale + zw bias for Isaac MDL UVs */
    int   v_flip;                 /* 1 = sample texture V as 1-v for MDL */
    float roughness_tex_scale;     /* sampled roughness = tex.g * scale + bias */
    float roughness_tex_bias;      /* Isaac MDL RoughnessMin/Max remap */
    int   _pad_v_flip;
} GpuMaterialParams;

/* Struct-size discipline.
 *
 * The std430 SSBO `MaterialBuffer` in src/shaders/raytrace.rchit.glsl
 * uses a `MaterialData` struct that MUST be byte-for-byte the same
 * size as GpuMaterialParams here. If they ever drift, `materials[i]`
 * in the shader reads from the wrong offset for any i >= 1 — that
 * was the chess-set King-Black-as-white-marble bug fixed in commit
 * 540d525. Bake the expected size as a constant and assert it on
 * every build so adding a new field forces a matching shader-side
 * update.
 *
 * If you bump this number, also update `struct MaterialData` in
 * src/shaders/raytrace.rchit.glsl and re-clear the pipeline cache
 * (`rm -rf ~/.cache/nusd_renderer/`).
 */
#define NUSD_GPU_MATERIAL_PARAMS_SIZE 272
#ifdef __cplusplus
static_assert(sizeof(GpuMaterialParams) == NUSD_GPU_MATERIAL_PARAMS_SIZE,
              "GpuMaterialParams size drift — see comment above and "
              "update raytrace.rchit.glsl::MaterialData to match.");
#else
_Static_assert(sizeof(GpuMaterialParams) == NUSD_GPU_MATERIAL_PARAMS_SIZE,
               "GpuMaterialParams size drift — see comment above and "
               "update raytrace.rchit.glsl::MaterialData to match.");
#endif

/* Upload material data (SSBO + textures) to GPU.
 * Returns 1 on success. Must be called before creating material pipelines. */
int  gpu_upload_materials(Gpu* gpu,
                          const GpuMaterialParams* materials, int nmaterials,
                          const GpuTextureData* textures, int ntextures);

/* GPU-side scene light layout — matches std430 packing in
 * raytrace.rchit.glsl. Each vec3 packs with the following scalar so the
 * struct stays 16-byte aligned without explicit padding fields. */
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

/* Phase 11.A: upload BasisCurves segment data + per-segment colors.
 * - segments  : SceneCurveSegment[seg_count]    (32 B/seg)
 * - colors    : float[seg_count * 3]            (RGB; packed to RGBA8 internally)
 * Pass seg_count == 0 to free the existing upload. Three device-local
 * buffers are created: a segment SSBO, a color SSBO, and an AABB
 * device buffer. The AABB buffer is allocated empty — its contents
 * are computed by a compute shader at gpu_build_curve_blas time
 * (Phase 12.x: GPU AABB-gen replaces the 825 MB host→device upload).
 * The SSBOs use VK_BUFFER_USAGE_STORAGE_BUFFER_BIT and are read by
 * the curve intersection / closest-hit shaders.
 *
 * Does NOT build the BLAS; call gpu_build_curve_blas separately
 * (Phase 11.A.2 step 3). */
int  gpu_upload_curve_data(Gpu* gpu,
                           const void* segments,
                           const float* colors,
                           int seg_count);

/* Phase 11.A.2.3: build a single VK_GEOMETRY_TYPE_AABBS_KHR BLAS over
 * the curve segments. No-op if no curve data is uploaded. Returns 1
 * on success.
 *
 * Phase 12.x: GPU-side AABB generation. This function dispatches a
 * compute shader that reads the segment SSBO and writes the AABB
 * device buffer, then issues a single memory barrier and submits the
 * AS build in the same command buffer. Eliminates the ~825 MB
 * host→device AABB upload on tera fixtures.
 *
 * The BLAS is independent of the mesh BLAS pool (gpu->blas_list).
 * Wiring it into the TLAS as an additional instance happens in a
 * subsequent step (Phase 11.A.2.5). */
int  gpu_build_curve_blas(Gpu* gpu);

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

/* Variant that takes an explicit intensity multiplier (matches ovrtx's
 * `dome.color = light.color * intensity * exp2(exposure)` convention).
 * Skips the auto-exposure normalisation and instead multiplies the HDR
 * by `intensity` directly. Use when the source intensity is known
 * (USD-authored DomeLight). */
int   gpu_load_environment_intensity(Gpu* gpu, const char* hdr_path, float intensity);
int   gpu_load_environment_tinted_intensity(Gpu* gpu, const char* hdr_path,
                                            float intensity, const float tint[3]);
int   gpu_load_flat_environment(Gpu* gpu, const float color[3], float intensity);

/* Destroy IBL resources (env map + BRDF LUT). */
void  gpu_destroy_environment(Gpu* gpu);

/* Query number of environment map mip levels (0 = no IBL loaded). */
int   gpu_get_env_mip_levels(Gpu* gpu);

/* Query whether IBL is currently loaded (1 = env_image_view != NULL). */
int   gpu_get_ibl_loaded(Gpu* gpu);

/* Query the USD-DomeLight intensity multiplier currently applied (1.0
 * default; up to thousands when DomeLight authors a high intensity).
 * Used by raster mesh.frag.glsl to scale the IBL ambient contribution
 * to match the rmiss sky multiplier. */
float gpu_get_env_intensity(Gpu* gpu);

/* Store OVRTX-style tone parameters for GPU-owned passes such as the
 * raster environment background. RT/raster mesh passes receive the same
 * values through their push constants from renderer.c. */
void  gpu_set_tone_mapping(Gpu* gpu, float exposure_scale, float sky_scale,
                           float white_point_scale, uint32_t flags);

/* Query whether the material descriptor set has been uploaded (placeholder
 * or real). Callers (e.g. nu_load_environment) need this because IBL setup
 * binds into the same descriptor set and refuses to run before it exists. */
int   gpu_materials_uploaded(const Gpu* gpu);

/* Create a pipeline for drawing the environment map as background. */
int   gpu_create_env_bg_pipeline(Gpu* gpu);

/* Draw the environment map as fullscreen background.
 * view_inv and proj_inv are 4x4 row-major inverse matrices. */
void  gpu_draw_env_background(Gpu* gpu, const float* view_inv, const float* proj_inv);

/* ---- Screenshot / Readback ---- */

/* Read back the last presented swapchain image and write to a PPM file.
 * Must be called after gpu_end_frame(). Returns 1 on success. */
int   gpu_screenshot(Gpu* gpu, const char* path);

/* Read back the last presented swapchain image into a caller-provided buffer.
 * out_pixels must hold width*height*4 bytes.
 * If swizzle_to_rgba=1, the BGRA8 swapchain is swizzled to RGBA8 via a
 * per-pixel 32-bit shuffle. If 0, the raw BGRA8 bytes are memcpy'd verbatim —
 * caller is responsible for any channel reorder.
 * Must be called after gpu_end_frame(). Returns 1 on success. */
int   gpu_readback_pixels(Gpu* gpu, uint8_t* out_pixels, uint32_t width, uint32_t height,
                          int swizzle_to_rgba);

/* CUDA-Vulkan interop variant: copy the last swapchain image into a
 * device-local exportable buffer (BGRA8, no CPU swizzle), and return the
 * cached POSIX fd for cuImportExternalMemory. The fd is owned by the Gpu
 * and closed by gpu_destroy — callers MUST NOT close it.
 *
 * Returns 1 on success, 0 if interop unavailable / size mismatch.
 *
 * out_fd:           POSIX fd (cached; do NOT close)
 * out_size:         allocation size in bytes (req.size, may be padded)
 * out_logical_size: meaningful payload size = width*height*4 */
int   gpu_readback_pixels_cuda(Gpu* gpu, uint32_t width, uint32_t height,
                                int* out_fd, uint64_t* out_size,
                                uint64_t* out_logical_size);

/* ---- Ray tracing ---- */

/* Returns 1 if hardware ray tracing is available on this GPU. */
int   gpu_rt_available(Gpu* gpu);

/* Returns 1 if the RT scene (TLAS + BLASes) was successfully built. */
int   gpu_rt_built(Gpu* gpu);

/* Returns 1 if Shader Execution Reordering (VK_NV_ray_tracing_invocation_reorder)
 * is supported and enabled on this device. The tiled-RT pipeline build path
 * uses this to choose between the SER and non-SER raygen SPVs. */
int   gpu_ser_available(Gpu* gpu);

/* 1 if VK_KHR_fragment_shader_barycentric is enabled — the textured/instanced
 * raster pipelines then use the mesh_bary.frag variant that interpolates the
 * three Ptex triangle-corner colors per pixel instead of flat-averaging. */
int   gpu_barycentric_available(Gpu* gpu);

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
    uint8_t    visible;         /* 0 = build BLAS for sharing, omit from TLAS */
    uint32_t   source_id;       /* caller's mesh id before RT descriptor compaction */
    int32_t    material_index;  /* per-instance material id; prototypes may differ */
    const char* debug_name;     /* optional caller-owned mesh path for diagnostics */
    float      bounds_min[3];   /* world-space AABB for diagnostics/residency manifests */
    float      bounds_max[3];
    float      transform[12];   /* 3x4 row-major world transform (VkTransformMatrixKHR) */
    float      color[3];        /* per-mesh display color (for RT shading) */
    uint64_t   geo_hash;        /* FNV-1a-64 triangle BLAS hash — positions +
                                   indices only. 0 when BLAS cache is inactive. */
    uint32_t   ptex_color_offset; /* offset into packed per-triangle Ptex color SSBO,
                                     0xFFFFFFFF = none */
} GpuRtMeshDesc;

/* Build one BLAS per mesh + TLAS + RT pipeline.
 *
 * Phase 11.A.2.5: optional curve shaders. When `rint_spv` and
 * `curve_chit_spv` are both non-NULL AND `gpu->curve_blas` /
 * `gpu->curve_seg_count` were populated by gpu_upload_curve_data +
 * gpu_build_curve_blas before this call, a curve procedural hit-group
 * gets added to the pipeline (4 stages / 4 groups / SBT-size 4) and a
 * curve TLAS instance appended with instanceShaderBindingTableRecord
 * Offset=1.
 *
 * Curve-only scenes (nmeshes==0 + curves present) are supported.
 *
 * Pass NULL/0 for the curve params to keep the legacy 3-group
 * pipeline byte-identical. Returns 1 on success. */
int   gpu_build_rt_scene(Gpu* gpu,
                         const GpuRtMeshDesc* meshes, uint32_t nmeshes,
                         const uint32_t* rgen_spv, uint32_t rgen_size,
                         const uint32_t* miss_spv, uint32_t miss_size,
                         const uint32_t* chit_spv, uint32_t chit_size,
                         const uint32_t* rint_spv, uint32_t rint_size,
                         const uint32_t* curve_chit_spv, uint32_t curve_chit_size);
void  gpu_destroy_rt_scene(Gpu* gpu);

/* Upload packed RGBA8 real material colors sampled from authored Ptex maps.
 * Three uints per mesh triangle, indexed from closest-hit by
 * MeshData.ptex_color_offset + gl_PrimitiveID*3 + corner. */
int   gpu_upload_rt_triangle_colors(Gpu* gpu,
                                    const uint32_t* colors,
                                    uint32_t count);

/* Set the source USD path used to derive the on-disk BLAS cache sidecar
 * (`<usd_path>.nzblas`). Call before gpu_build_rt_scene. Pass NULL or "" to
 * disable the BLAS cache for this build (e.g. programmatic / handle scenes).
 * The cache write itself is opt-in via the NUSD_BLAS_CACHE_WRITE env var. */
void  gpu_set_blas_cache_path(Gpu* gpu, const char* usd_path);

/* Update TLAS instance transforms without rebuilding BLASes.
 * transforms is [nmeshes][12] row-major 3x4 (VkTransformMatrixKHR format).
 * nmeshes must match the count from the original gpu_build_rt_scene() call.
 * visibility (optional, may be NULL) is [nmeshes] uint8 — 0 hides, nonzero shows.
 * Returns 1 on success. */
int   gpu_update_tlas(Gpu* gpu, const float* transforms, const uint8_t* visibility, uint32_t nmeshes);

/* Update per-mesh display colors in the scene data SSBO without rebuilding
 * BLASes or TLAS.  colors is [nmeshes][3] float RGB.
 * Returns 1 on success. */
int   gpu_update_scene_colors(Gpu* gpu, const float* colors, uint32_t nmeshes);

/* Set the authored scene up-axis (0=X, 1=Y, 2=Z) consumed by RT shaders
 * for environment lookup, hemisphere fill, and height-based material terms.
 * Stashed on the Gpu and uploaded into the SceneData SSBO header. */
int   gpu_set_scene_up_axis(Gpu* gpu, int up_axis);

/* Set the flat DomeLight color (RGB + intensity) consumed by the fast_mode
 * rmiss shader (sky background) and rchit hemispheric ambient. The values
 * are stashed on the Gpu and uploaded into the SceneData SSBO header at
 * offset 16 (matching `vec4 domeColor;` in the shaders). When the SSBO is
 * already built, the header is patched in-place via vkCmdUpdateBuffer; when
 * not yet built, the values are stored and consumed by the next
 * gpu_build_rt_scene call. Returns 1 on success. */
int   gpu_set_dome_color(Gpu* gpu, float r, float g, float b, float intensity);

/* Set per-mesh tex_index for the fast_mode rchit texture sample.
 * tex_index = 0xFFFFFFFF means "no texture, use flat color" (rchit fast
 * path takes the early return). When the SSBO is already built, the
 * per-mesh entry is patched in-place via vkCmdUpdateBuffer; when not
 * yet built, the value is stashed in the per-mesh array and consumed
 * by the next gpu_build_rt_scene SSBO upload. Returns 1 on success. */
int   gpu_set_mesh_texture(Gpu* gpu, uint32_t mesh_id, uint32_t tex_index);

/* Inline TLAS update: upload instances + record TLAS build into the CURRENT
 * command buffer (gpu->current_cmd), adding a barrier before the trace dispatch.
 * Call between gpu_begin_frame_tiled_rt() and gpu_cmd_trace_rays_tiled().
 * visibility (optional, may be NULL) is [nmeshes] uint8 — 0 hides, nonzero shows.
 * Returns 1 on success, 0 on error (falls back to gpu_update_tlas). */
int   gpu_update_tlas_inline(Gpu* gpu, const float* transforms, const uint8_t* visibility, uint32_t nmeshes);

/* Phase A — per-tile env isolation. Set the per-mesh TLAS instance.mask byte
 * read by gpu_update_tlas / gpu_update_tlas_inline. masks is [nmeshes] uint8.
 * If never called (or count = 0), TLAS-update fills mask = 0xFF (visible to all
 * rays). When set, the mask is AND-ed with the visibility byte at fill time
 * (visibility=0 still hides the instance). */
void  gpu_set_instance_masks(Gpu* gpu, const uint8_t* masks, uint32_t count);

/* Phase C — declare an N-way env partition. Stores mesh_to_env[count] +
 * num_envs on the GPU state. The next gpu_build_rt_scene call (or
 * standalone gpu_build_partitioned_tlases) will build one TLAS per env in
 * addition to the legacy single TLAS. count = 0 / num_envs = 0 clears the
 * partition. Backwards-compatible: when no partition is set, the existing
 * single-TLAS path runs unchanged. */
void  gpu_set_env_partition(Gpu* gpu, const int* mesh_to_env, int count, int num_envs);

/* Build the per-env TLAS array (Phase C). Called from gpu_build_rt_scene
 * with the legacy `instances` array still alive — transforms and
 * metadata are copied from it into the per-env partitioned layout.
 * Requires gpu_set_env_partition() to have been called.
 * `legacy_instances` must be NULL or a pointer to the just-built legacy
 * VkAccelerationStructureInstanceKHR array of length `legacy_count`.
 * Returns 1 on success, 0 on failure. */
struct VkAccelerationStructureInstanceKHR;
int   gpu_build_partitioned_tlases(Gpu* gpu,
                                    const void* legacy_instances,
                                    uint32_t legacy_count);

/* ---- PR 2: GPU-driven TLAS instance translation ----
 *
 * Allocate (or resize) an exportable storage buffer of size `count * 64`
 * bytes that holds (count) row-major 4x4 transforms. The warp transform
 * kernel writes into the CUDA-imported view of this buffer; the compute
 * shader (tlas_translate.comp) then copies the upper 3x4 from each into
 * the corresponding VkAccelerationStructureInstanceKHR slot of
 * gpu->instance_buf — leaving the per-instance metadata bytes 48..63
 * (customIndex/mask/sbtOffset/flags + AS-reference) untouched.
 *
 * Returns 1 on success. */
int      gpu_create_tlas_xforms_buffer(Gpu* gpu, int count);

/* Export the transforms buffer's VkDeviceMemory as a POSIX fd for CUDA.
 * Each call returns a fresh single-use fd consumed by cuImportExternalMemory,
 * or closed by the caller if import is not attempted. Returns -1 on error. */
int      gpu_export_tlas_xforms_fd(Gpu* gpu);

/* Logical size in bytes (count*64) of the transforms buffer.  Use this
 * for cuExternalMemoryGetMappedBuffer's `size` field. */
uint64_t gpu_get_tlas_xforms_size(Gpu* gpu);

/* Upload the (gid → tlas_instance_idx) lookup table that the compute
 * shader uses to direct each thread's transform write into the right
 * VkAccelerationStructureInstanceKHR slot in instance_buf. tlas_indices
 * must be `count` int32s. tlas_indices[gid] < 0 ⇒ skip slot for that gid.
 *
 * Synchronous: blocks while the device-local indices buffer is uploaded. */
int      gpu_set_transform_layout(Gpu* gpu, const int* tlas_indices, int count);

/* Record the compute dispatch + barrier + inline TLAS update build into
 * the CURRENT frame's command buffer.  Replaces gpu_update_tlas_inline()
 * for the GPU-driven path: the host side never reads the transforms,
 * never builds VkAccelerationStructureInstanceKHR records, never copies
 * into instance_buf — the compute shader does it on the GPU.
 *
 * The caller is responsible for ensuring all warp writes into the imported
 * transforms buffer have completed before this dispatch is submitted (in
 * the current single-stream / single-queue setup, a `wp.synchronize_device()`
 * before this call is sufficient).
 *
 * Returns 1 on success, 0 if pre-conditions fail (no current cmd, no
 * scratch, or buffers missing). */
int      gpu_translate_instances_inline(Gpu* gpu, int count);

/* Read (mesh_id → tlas_instance_idx) inversion data so renderer.c can
 * translate the caller-supplied dense mesh-id list into the TLAS-slot
 * indices the compute shader needs. Returns the array length (always
 * gpu->tlas_instance_count) and writes the inverse-of-instance_custom
 * mapping into out_mesh_to_tlas_idx, sized to nmeshes (caller-provided).
 * Slots not bound to any TLAS instance are written as -1. Returns 0 on
 * error. */
int      gpu_get_mesh_to_tlas_idx(Gpu* gpu, int* out_mesh_to_tlas_idx, int nmeshes);

/* Abort an in-progress tiled frame (discard the command buffer). */
void  gpu_abort_frame_tiled_rt(Gpu* gpu);

/* RT frame (alternative to raster begin_frame/end_frame) */
int   gpu_begin_frame_rt(Gpu* gpu);
void  gpu_cmd_trace_rays(Gpu* gpu, const GpuRtPushConstants* pc);
void  gpu_end_frame_rt(Gpu* gpu);

/* Pre-recorded command-buffer cache (perf/cmdbuf-cache).
 * For static scenes with the same render parameters, the per-frame
 * vkBeginCommandBuffer + vkCmd* sequence + vkEndCommandBuffer can be
 * replayed across frames.  Resize, accel rebuild, descriptor changes,
 * camera change, etc. invalidate the cache.  Callers must invalidate
 * whenever they mutate state that affects either pipeline bindings,
 * descriptor contents, push constants, or barrier resources. */
void  gpu_invalidate_rt_cmd_cache(Gpu* gpu);
void  gpu_invalidate_tiled_cmd_cache(Gpu* gpu);

/* Test/debug introspection: how many cached cmd-buffer replays vs full
 * re-records have occurred since gpu_create.  Useful for verifying that
 * caching actually engages on the steady-state path. */
void  gpu_get_cmd_cache_stats(Gpu* gpu,
                              uint64_t* out_rt_replays,
                              uint64_t* out_rt_records,
                              uint64_t* out_tiled_replays,
                              uint64_t* out_tiled_records);

/* ---- Tiled multi-camera RT ---- */

/* Push constants for tiled RT.
 * Layout matches GpuRtPushConstants so that the shared miss/hit
 * shaders can read ground_y and scene_scale at the same offsets (128, 132).
 * The first 16 bytes overlap with view_inv[0..3] but are reinterpreted as
 * tile parameters by the tiled raygen shader via floatBitsToUint(). */
typedef struct {
    uint32_t tile_w;        /* offset 0:  per-camera width in pixels */
    uint32_t tile_h;        /* offset 4:  per-camera height in pixels */
    uint32_t num_cols;      /* offset 8:  number of tile columns in the grid */
    uint32_t num_cameras;   /* offset 12: total number of cameras */
    float    _pad[28];      /* offset 16-127: padding to align ground_y */
    float    ground_y;      /* offset 128: Z-up ground height (read by miss shader) */
    float    scene_scale;   /* offset 132: scene diagonal (read by miss shader) */
    uint32_t fast_mode;     /* offset 136: 1 = skip shadow rays for RL sensors */
    uint32_t depth_enabled; /* offset 140: 1 = write depth to binding 10 SSBO */
    uint32_t segmentation_enabled; /* offset 144: 1 = write IDs to binding 11 SSBO */
    uint32_t normals_enabled;     /* offset 148: 1 = write normals to binding 12 SSBO */
    /* Phase B deferred-shading: when 1, rchit/rmiss write the G-buffer SSBO
     * (binding 17) inside the fast_mode path, and the host runs a follow-up
     * compute dispatch that reads the G-buffer and produces flat-shaded
     * pixels into binding 9. Default 0 leaves the legacy RT color path. */
    uint32_t deferred_shade_enabled; /* offset 152 */
    float    tone_exposure_scale;    /* offset 156 */
    float    tone_sky_scale;         /* offset 160 */
    float    tone_white_point;       /* offset 164 */
    uint32_t tone_flags;             /* offset 168 */
    float    rt_ibl_fill_scale;      /* offset 172 */
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
                                   const uint32_t* chit_spv, uint32_t chit_size,
                                   const uint32_t* rint_spv, uint32_t rint_size,
                                   const uint32_t* curve_chit_spv, uint32_t curve_chit_size);
void  gpu_destroy_tiled_rt_pipeline(Gpu* gpu);
int   gpu_tiled_rt_pipeline_built(Gpu* gpu);

/* Tiled RT frame: trace rays at full tiled resolution. */
int   gpu_begin_frame_tiled_rt(Gpu* gpu);
void  gpu_cmd_trace_rays_tiled(Gpu* gpu, const GpuRtTiledPushConstants* pc);
void  gpu_end_frame_tiled_rt(Gpu* gpu);

/* ---- Phase B deferred-shading compute pass ----
 *
 * Build/destroy the deferred-shade compute pipeline. Bindings:
 *   2  (storage buffer, READ): SceneData (per-mesh display color)
 *   9  (storage buffer, WRITE): DirectOutput (overwrites the rgen's pixels)
 *   17 (storage buffer, READ): G-buffer populated by rchit + rmiss
 *
 * The pipeline shares a push-constant range with the tiled RT pipeline
 * (GpuRtTiledPushConstants), so the same `pc` value pushed before
 * vkCmdTraceRaysKHR is also pushed before vkCmdDispatch. */
int   gpu_build_deferred_pipeline(Gpu* gpu,
                                   const uint32_t* comp_spv, uint32_t comp_size);
void  gpu_destroy_deferred_pipeline(Gpu* gpu);
int   gpu_deferred_pipeline_built(Gpu* gpu);

/* Allocate (or resize) the per-launch tiled G-buffer at binding 17 and
 * rebind both tiled descriptor sets (set + set_b) to point at it. Falls
 * back to the legacy `scene_data_buf` stub on allocation failure (no
 * deferred shading possible). Safe to call repeatedly; a no-op when the
 * buffer is already sized to (tiled_image_w * tiled_image_h * 32 B). */
int   gpu_tiled_ensure_gbuffer(Gpu* gpu);

/* Record the deferred-shade compute dispatch into the current tiled
 * command buffer. Must be called between gpu_cmd_trace_rays_tiled() and
 * gpu_end_frame_tiled_rt(). When the cached cmd-buffer fast path is
 * replaying, this is a no-op (the cached cmd already holds the dispatch). */
void  gpu_cmd_deferred_shade(Gpu* gpu, const GpuRtTiledPushConstants* pc);

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
 * Returns fd >= 0 on success, -1 on failure (also -1 when external_buffers_active). */
int   gpu_export_tiled_image_fd(Gpu* gpu, int buf_idx);

/* Hybrid C — Inverted memory ownership.
 * Caller (trainer) allocates the tiled output buffer via the CUDA VMM API
 * (cuMemCreate + cuMemExportToShareableHandle, POSIX_FILE_DESCRIPTOR) and passes
 * the resulting fds. The renderer creates VkBuffers with VK_KHR_external_memory_fd,
 * imports each fd as VkDeviceMemory, and writes its tiled output directly into
 * the caller-owned memory. The caller skips cuImportExternalMemory entirely.
 *
 * fds are consumed by vkAllocateMemory on success — caller must NOT close them.
 * On failure, caller still owns the fds and must close them.
 *
 * mem_size_each: bytes per buffer (each of the two double-buffered slots).
 *                Must be >= total_w * total_h * 4 AND >= Vulkan req.size for the
 *                buffer (which the caller can satisfy by rounding the logical
 *                size up to the larger of CUDA VMM granularity and 64 KiB).
 *
 * Returns 1 on success, 0 on failure. */
int   gpu_set_external_output_buffers(Gpu* gpu,
                                      int mem_fds[2],
                                      uint64_t mem_size_each,
                                      uint32_t total_w,
                                      uint32_t total_h);

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
                    float* out_distances, int* out_mesh_ids, float* out_normals,
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

/* ---- GPU phase timings (perf/vk-instrumentation) ----
 *
 * Each phase has a fixed slot index in a single VkQueryPool. Begin
 * records vkCmdWriteTimestamp at TOP_OF_PIPE; end records at
 * BOTTOM_OF_PIPE; the difference (in nanoseconds, scaled by
 * timestampPeriod) lands in gpu->phase_timing_ns[phase].
 *
 * The same call also emits VK_EXT_debug_utils labels when the extension
 * is enabled, so nsys's vulkan_gpu_marker_sum report sees identical
 * region boundaries.
 *
 * Both ops are no-ops when:
 *   - cmd is VK_NULL_HANDLE (defensive),
 *   - timestamps aren't supported (gpu->timestamps_supported == 0), AND
 *   - debug-utils is unavailable (gpu->debug_utils_enabled == 0).
 *
 * The slot-reset logic is bundled into gpu_phase_begin so callers don't
 * need to remember vkCmdResetQueryPool. */
typedef enum {
    GPU_PHASE_RT_DISPATCH        = 0,
    GPU_PHASE_PIXEL_READBACK     = 1,
    GPU_PHASE_BLAS_BUILD         = 2,
    GPU_PHASE_TLAS_BUILD         = 3,
    GPU_PHASE_CURVE_BLAS_BUILD   = 4,
    GPU_PHASE_STAGING_UPLOAD_SEGS    = 5,
    GPU_PHASE_STAGING_UPLOAD_AABBS   = 6,
    GPU_PHASE_STAGING_UPLOAD_COLORS  = 7,
    /* Phase C.4 mechanism hunt — per-dispatch GPU timestamps wrapping the
     * tiled-RT vkCmdTraceRaysKHR and the deferred-shading vkCmdDispatch.
     * Used to discriminate between (a) the rgen short-circuit that writes
     * a G-buffer and skips inline PBR — which would shrink TRACE_RAYS_TILED
     * GPU time — and (b) async-overlap from new sync boundaries — which
     * leaves trace-rays GPU time unchanged but moves the host's fence wait
     * to a previous frame. See docs/reports/dex_perf_2026-05-06/. */
    GPU_PHASE_TRACE_RAYS_TILED   = 8,
    GPU_PHASE_DEFERRED_COMPUTE   = 9,
    GPU_PHASE_COUNT              = 10,
} GpuPhase;

/* Begin/end a phase region inside `cmd`. `label` is shown in nsys debug
 * markers - keep it short and human-readable. */
void  gpu_phase_begin(Gpu* gpu, void* cmd /* VkCommandBuffer */,
                      GpuPhase phase, const char* label);
void  gpu_phase_end(Gpu* gpu, void* cmd /* VkCommandBuffer */, GpuPhase phase);

/* Block until the queue is idle and resolve all pending timestamp queries
 * into gpu->phase_timing_ns[]. The "all" form is used at scene-build
 * time after the renderer has issued vkQueueWaitIdle anyway. */
void  gpu_phase_resolve_all(Gpu* gpu);

/* Resolve only the slots for a specific phase (used for RT_dispatch +
 * pixel_readback after gpu_readback_pixels' vkDeviceWaitIdle). */
void  gpu_phase_resolve(Gpu* gpu, GpuPhase phase);

/* Read most-recent timing in milliseconds. 0.0 if phase never recorded. */
float gpu_phase_get_ms(Gpu* gpu, GpuPhase phase);

/* 1 if VK_EXT_debug_utils is enabled on the current instance. */
int   gpu_debug_utils_enabled(Gpu* gpu);

/* 1 if device-side timestamp queries are supported on graphics queue. */
int   gpu_timestamps_supported(Gpu* gpu);

/* ---- VK3DGRT (3D Gaussian Ray Tracing) ----
 *
 * Splat-scene GPU integration, parallel to the mesh BLAS/TLAS path. See
 * docs/plans/VK3DGRT_PLAN.md and src/gs_scene.c. Phase 2/3.
 *
 * The host-side particle arrays + config knobs live in NuGsScene (CPU);
 * these functions push them onto the GPU and run the 3DGRT trace.
 * RT-only path; no raster fallback. */

/* Upload (or replace) particle SSBOs from host arrays. Allocates / resizes
 * the per-particle buffers on demand. Triggers a TLAS rebuild on the next
 * gpu_gs_build_accel call. Returns 1 on success.
 *
 * positions:     N * 3 float
 * scales:        N * 3 float (linear sigma)
 * orientations:  N * 4 float (wxyz)
 * opacities:     N * 1 float ([0,1])
 * kernel_scales: N * 1 float (precomputed by host)
 * sh_coefficients: N * (sh_degree+1)^2 * 3 float
 * proxy_kind:    0 = icosahedron BLAS, 1 = AABB BLAS
 * prim_xform:    optional row-major 4x4 (NULL = identity) */
int  gpu_gs_upload_particles(Gpu* gpu,
                             const float* positions,
                             const float* scales,
                             const float* orientations,
                             const float* opacities,
                             const float* kernel_scales,
                             const float* sh_coefficients,
                             uint32_t particle_count,
                             int sh_degree,
                             int proxy_kind);

/* Build (or rebuild) the splat acceleration structures: lazily creates
 * the unit BLAS for the active proxy kind, dispatches the gs_as_build
 * compute shader to fill the TLAS instance buffer, then builds the TLAS.
 * Must be called after gpu_gs_upload_particles. Returns 1 on success. */
int  gpu_gs_build_accel(Gpu* gpu, const float prim_xform[16]);

/* Build (or rebuild) the splat RT pipeline (rgen/rahit/rchit/rint/rmiss).
 * SBT layout depends on proxy_kind (icosa = no rint group, AABB = rint
 * group). The rchit stage is optional (NULL/0 to omit) — Phase 3's full
 * K-buffer algorithm has no closest-hit (rgen integrates over the per-pixel
 * SoA K-buffer that any-hit fills). For the Pragmatist visual gate we
 * pass an rchit that paints by InstanceID. Returns 1 on success. */
int  gpu_gs_build_pipeline(Gpu* gpu,
                           const uint32_t* rgen_spv,  uint32_t rgen_sz,
                           const uint32_t* rahit_spv, uint32_t rahit_sz,
                           const uint32_t* rchit_spv, uint32_t rchit_sz,
                           const uint32_t* rint_spv,  uint32_t rint_sz,
                           const uint32_t* rmiss_spv, uint32_t rmiss_sz);

/* Trace the splat scene into an output image. Phase 3 wires the per-tile
 * SoA K-buffer in here; Phase 2 just produces the launch-id gradient
 * stub from the rgen shader for verification. Returns 1 on success. */
int  gpu_gs_render_tile(Gpu* gpu,
                        const float view_inverse[16],
                        const float proj_inverse[16],
                        int K, int max_passes,
                        float min_transmittance,
                        int color_space,
                        int camera_model,
                        int tile_w, int tile_h);

/* Read back the iso-opacity depth buffer. `out_depth` is W*H float32s
 * (one per pixel). Per-pixel value is the t at which accumulated
 * opacity first crosses iso_opacity_threshold; -1 if it never crosses
 * (sky pixel or transmittance never dropped below 1 - iso). Plan §7.
 * Returns 1 on success. */
int  gpu_gs_fetch_depth(Gpu* gpu, float* out_depth, uint32_t w, uint32_t h);

/* Read back the iso-opacity surface normal. `out_normal` is W*H*3
 * float32s (xyz per pixel). (0, 0, 0) on miss / no iso crossing.
 * Plan §7. Returns 1 on success. */
int  gpu_gs_fetch_normal(Gpu* gpu, float* out_normal, uint32_t w, uint32_t h);

/* Tear down all VK3DGRT resources. Idempotent. Called from gpu_shutdown
 * and from explicit nu_gs_clear_particles when proxy_kind changes. */
void gpu_gs_destroy(Gpu* gpu);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_GPU_H */
