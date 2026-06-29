// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_GPU_H
#define NUSD_GPU_H

/*
 * gpu.h — Render Hardware Interface (RHI)
 *
 * Thin abstraction over OpenGL ES 3.2 for the OpenGL renderer.
 * Same interface as the Vulkan viewer's gpu.h, with RT/DLSS
 * functions compiled as no-ops.
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
    GPU_FORMAT_FLOAT3 = 0,
    GPU_FORMAT_FLOAT2 = 1,
    GPU_FORMAT_UINT   = 2,
    GPU_FORMAT_FLOAT1 = 3,
    GPU_FORMAT_SNORM16X4 = 4,
    GPU_FORMAT_UNORM8X4  = 5,
} GpuVertexFormat;

/* ---- Descriptors ---- */

typedef struct {
    GpuBufferUsage usage;
    uint64_t       size;
    const void*    data;
} GpuBufferDesc;

typedef struct {
    uint32_t        location;
    uint32_t        offset;
    GpuVertexFormat format;
} GpuVertexAttrib;

typedef struct {
    const char*         vert_glsl;        /* GLES: GLSL source string */
    const char*         frag_glsl;        /* GLES: GLSL source string */
    /* Optional tessellation stages. Both must be set or both NULL. When
     * set, the pipeline draws GL_PATCHES with `patch_vertices` CVs per
     * patch (used for BasisCurves tube rendering). */
    const char*         tcs_glsl;
    const char*         tes_glsl;
    uint32_t            patch_vertices;   /* 0 = non-tess pipeline */
    const uint32_t*     vert_spv;         /* ignored in GLES */
    uint32_t            vert_size;
    const uint32_t*     frag_spv;         /* ignored in GLES */
    uint32_t            frag_size;
    uint32_t            push_constant_size;
    uint32_t            vertex_stride;
    const GpuVertexAttrib* attribs;
    uint32_t            nattribs;
} GpuPipelineDesc;

/* Push constant layout: MVP + model + color + eye_pos */
typedef struct {
    float mvp[16];
    float model[16];
    float color[4];
    float eye_pos[3];
    float _pad_eye;
} GpuMeshPushConstants;

/* ---- Lifecycle ---- */

Gpu*  gpu_init(void* glfw_window, int width, int height);
void  gpu_shutdown(Gpu* gpu);
void  gpu_resize(Gpu* gpu, int width, int height);

/* ---- Resources ---- */

GpuBuffer   gpu_create_buffer(Gpu* gpu, const GpuBufferDesc* desc);
void        gpu_destroy_buffer(Gpu* gpu, GpuBuffer buf);

GpuPipeline gpu_create_pipeline(Gpu* gpu, const GpuPipelineDesc* desc);
void        gpu_destroy_pipeline(Gpu* gpu, GpuPipeline pipe);

/* ---- Frame ---- */

int   gpu_begin_frame(Gpu* gpu);
void  gpu_end_frame(Gpu* gpu);

/* ---- Draw commands ---- */

void  gpu_cmd_bind_pipeline(Gpu* gpu, GpuPipeline pipe);
void  gpu_cmd_bind_vertex_buffer(Gpu* gpu, GpuBuffer buf);
void  gpu_cmd_bind_index_buffer(Gpu* gpu, GpuBuffer buf);
void  gpu_cmd_push_constants(Gpu* gpu, const void* data, uint32_t size);
void  gpu_cmd_draw(Gpu* gpu, uint32_t vertex_count, uint32_t first_vertex);
void  gpu_cmd_draw_indexed(Gpu* gpu, uint32_t index_count,
                           uint32_t first_index, int32_t vertex_offset);
void  gpu_cmd_draw_indexed_typed(Gpu* gpu, uint32_t index_count,
                                 uint64_t index_byte_offset,
                                 int32_t vertex_offset,
                                 int index_type_bits);

/* Compact-PointInstancer instancing. Upload all per-instance world matrices
 * (16 floats each, column-vector row-major — see scene_instance_transform_to_model16)
 * once; then draw a prototype's geometry instance_count times, reading the matrix
 * slice starting at first_instance (GLES 3.2 has no baseInstance, so the slice is
 * selected by the instance attribute pointer offset). */
void  gpu_upload_instance_transforms(Gpu* gpu, const float* matrices16, uint32_t count);
void  gpu_cmd_draw_instanced(Gpu* gpu, uint32_t index_count, uint32_t first_index,
                             uint32_t instance_count, uint32_t first_instance);
#define GPU_MAX_SHADOW_LIGHTS 2

int   gpu_shadow_begin(Gpu* gpu, int slot, int size, const float light_vp[16],
                       int light_index);
void  gpu_shadow_end(Gpu* gpu);

/* ---- Materials ---- */

typedef struct {
    const unsigned char* pixels;
    int width;
    int height;
    int is_srgb;            /* 1 = upload as GL_SRGB8_ALPHA8 (color); 0 = GL_RGBA8 (data) */
} GpuTextureData;

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
    int   v_flip;             /* 1 → sample at (u, 1-v); MDL-authored materials */
    /* UsdPreviewSurface opacityThreshold. > 0 = alpha-cutout (discard
     * pixels where opacity*opacity_tex < threshold); 0 = alpha-blend
     * (no discard, output blended alpha). Mirrors the vulkan path. */
    float opacity_threshold;
    float _pad_a, _pad_b, _pad_c; /* align following vec4 in std140 */
    float mdl_uv_transform[4];    /* xy texture scale + zw bias; zero means identity */
    float transmission_color[4];
    float transmission_weight;
    float transmission_ior;
    float _pad_d, _pad_e;         /* std140 vec4 */
    int   use_specular_workflow;
    float _pad_f, _pad_g, _pad_h;
    float specular_color[4];
    float roughness_tex_scale;    /* sampled roughness = tex.g * scale + bias */
    float roughness_tex_bias;     /* Isaac MDL RoughnessMin/Max remap */
    float _pad_i, _pad_j;         /* std140 align trailing block */
} GpuMaterialParams;

/* Per-mesh data uploaded to one big UBO, indexed via glBindBufferRange.
 * std140-friendly: mat4s are 64-byte aligned, vec4 is 16-byte aligned.
 * Total = 64 + 64 + 16 + 16 = 160 bytes; pad in the UBO to 256-byte stride
 * to match GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT on macOS. */
typedef struct {
    float mvp[16];      /* 64 B */
    float model[16];    /* 64 B */
    float color[4];     /* 16 B — .w > 0.5 = override vertex color */
    uint32_t ptex[4];    /* x = packed Ptex color offset, 0xFFFFFFFF = none */
} GpuMeshData;

/* Allocate the mesh-data UBO sized for `nmeshes` slices. Stored on Gpu
 * so subsequent map/bind calls find it. Returns 1 on success. */
int  gpu_alloc_mesh_buffer(Gpu* gpu, int nmeshes);
/* Map the entire mesh-data UBO for write. Returns the base of a CPU-
 * visible region; caller writes one GpuMeshData per slot at
 * `(unsigned char*)base + slot * stride`. Use INVALIDATE_BUFFER +
 * UNSYNCHRONIZED so writes don't stall on in-flight GPU reads. */
void* gpu_begin_mesh_writes(Gpu* gpu);
/* Per-mesh stride (bytes) for `gpu_begin_mesh_writes` slot pointer math. */
int  gpu_mesh_stride(Gpu* gpu);
void gpu_end_mesh_writes(Gpu* gpu);
/* Bind one mesh's slice of the mesh UBO at binding=1 for the next draw. */
void gpu_cmd_bind_mesh_data(Gpu* gpu, int mesh_index);
/* Set the per-frame eye-position uniform (shared across all meshes). */
void gpu_cmd_set_eye_pos(Gpu* gpu, const float eye[3]);

/* Set the per-frame u_view + u_proj plain uniforms on the current
 * pipeline (Storm-style separate-matrix path used by the curve TES).
 * Matrices are column-major, 16 floats. No-op if the pipeline lacks
 * either uniform. */
void gpu_cmd_set_view_proj(Gpu* gpu, const float view16[16],
                           const float proj16[16]);

/* Set the curve TES's u_basis_id uniform: 0=bezier, 1=bspline,
 * 2=catmullRom, 3=linear. No-op if the current pipeline doesn't
 * declare u_basis_id. */
void gpu_cmd_set_basis_id(Gpu* gpu, int basis_id);

/* OVRTX-style exposure/tonemap scale applied by built-in shaders.
 * Values of 1,1,1,0 preserve legacy output. */
void gpu_set_tone_mapping(Gpu* gpu, float exposure_scale, float sky_scale,
                          float white_point_scale, uint32_t flags);
void gpu_set_fallback_lighting(Gpu* gpu, int enabled);

int  gpu_upload_materials(Gpu* gpu,
                          const GpuMaterialParams* materials, int nmaterials,
                          const GpuTextureData* textures, int ntextures);
int  gpu_upload_ptex_triangle_colors(Gpu* gpu,
                                      const uint32_t* colors, uint32_t count);

#define GPU_MAX_SCENE_LIGHTS 32

typedef struct {
    float position[3];
    float intensity;
    float normal[3];
    int   kind;          /* 0 = RectLight, 1 = DistantLight, 2 = SphereLight */
    float u_axis[3];
    int   normalize;
    float v_axis[3];
    float angle_deg;
    float color[3];
    float _pad;
} GpuLight;

int  gpu_upload_lights(Gpu* gpu, const GpuLight* lights, int nlights);
void gpu_set_authored_light_count(Gpu* gpu, int nlights);

GpuPipeline gpu_create_material_pipeline(Gpu* gpu, const GpuPipelineDesc* desc);
void gpu_cmd_begin_material_pass(Gpu* gpu);
void gpu_cmd_bind_material(Gpu* gpu, int material_index);
void gpu_cmd_bind_materials(Gpu* gpu);
void gpu_destroy_materials(Gpu* gpu);

/* ---- Debug ---- */

void  gpu_set_debug_mode(Gpu* gpu, int mode);

/* ---- Screenshot ---- */

int   gpu_screenshot(Gpu* gpu, const char* path);

/* ---- Text overlay ---- */

int   gpu_overlay_init(Gpu* gpu);
void  gpu_overlay_shutdown(Gpu* gpu);
void  gpu_overlay_text(Gpu* gpu, float x, float y, float scale,
                       float r, float g, float b, float a,
                       const char* text);
void  gpu_overlay_rect(Gpu* gpu, float x, float y, float w, float h,
                       float r, float g, float b, float a);
void  gpu_overlay_flush(Gpu* gpu);

/* ---- Environment (IBL) ---- */

int  gpu_load_environment(Gpu* gpu, const char* hdr_path);
int  gpu_load_environment_intensity(Gpu* gpu, const char* hdr_path, float intensity);
int  gpu_load_environment_tinted_intensity(Gpu* gpu, const char* hdr_path,
                                           float intensity, const float tint[3]);
int  gpu_load_flat_environment(Gpu* gpu, const float color[3], float intensity);
void gpu_set_environment_intensity(Gpu* gpu, float intensity);
void gpu_draw_env_background(Gpu* gpu, const float view_inv[16], const float proj_inv[16]);
void gpu_destroy_environment(Gpu* gpu);

/* ---- Diagnostics ---- */

uint64_t gpu_get_allocated_memory(Gpu* gpu);

/* ---- RT/DLSS stubs (no-ops for GLES) ---- */

static inline int gpu_rt_available(Gpu* gpu) { (void)gpu; return 0; }
static inline void gpu_destroy_rt_scene(Gpu* gpu) { (void)gpu; }
static inline int gpu_dlss_available(Gpu* gpu) { (void)gpu; return 0; }
static inline void gpu_dlss_shutdown(Gpu* gpu) { (void)gpu; }
static inline GpuPipeline gpu_create_shadow_pipeline(Gpu* gpu, const GpuPipelineDesc* desc)
    { (void)gpu; (void)desc; return NULL; }
static inline void gpu_cmd_bind_shadow(Gpu* gpu) { (void)gpu; }
static inline void gpu_get_render_extent(Gpu* gpu, uint32_t* w, uint32_t* h)
    { (void)gpu; (void)w; (void)h; }

#ifdef __cplusplus
}
#endif

#endif /* NUSD_GPU_H */
