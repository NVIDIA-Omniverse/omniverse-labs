// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * gpu_metal.mm — Metal implementation of the gpu.h RHI.
 *
 * Phase 1: skeleton. Real MTLDevice + MTLCommandQueue created; resource
 * functions (gpu_create_buffer / gpu_destroy_buffer) work with real
 * MTLBuffer backing; all rendering, ray tracing, materials, IBL, and
 * interop functions are stubs that return safe defaults so the existing
 * renderer.c proceeds without crashing.
 *
 * Subsequent phases fill in:
 *   Phase 2 — raster pipeline + mesh / overlay shader execution
 *   Phase 3 — gpu_screenshot / gpu_readback_pixels (real)
 *   Phase 4 — hardware ray tracing (BLAS / TLAS + intersector kernel)
 *   Phase 5 — tiled multi-camera RT + depth/seg/normal output
 *   Phase 6 — raycast (LiDAR) compute pipeline
 *   Phase 7 — MaterialX MSL backend + IBL
 *
 * CUDA-Vulkan zero-copy interop is permanently stubbed (no CUDA on macOS).
 * DLSS is permanently stubbed (NVIDIA-only; MetalFX would be a future
 * separate feature).
 *
 * Compiled as Objective-C++ with ARC enabled (-fobjc-arc). The opaque Gpu /
 * GpuBuffer_s / GpuPipeline_s structs hold strong id<MTL...> references via
 * ARC, so destruction is via `delete` (C++ destructor releases the ids).
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "gpu.h"
#include "gs_scene.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cstddef>
#include <ctime>

static_assert(sizeof(GpuLight) == 80,
              "GpuLight must match raytrace.metal packed_float3 layout");
static_assert(sizeof(GpuMaterialParams) == 3504,
              "GpuMaterialParams must match Metal shader material stride");
static_assert(offsetof(GpuMaterialParams, uv_transform) == 240,
              "GpuMaterialParams uv_transform offset mismatch");
static_assert(offsetof(GpuMaterialParams, v_flip) == 272,
              "GpuMaterialParams v_flip offset mismatch");
static_assert(offsetof(GpuMaterialParams, procedural_nodes) == 432,
              "GpuMaterialParams procedural_nodes offset mismatch");

static int env_flag_enabled(const char* name)
{
    const char* e = getenv(name);
    if (!e || !e[0]) return 0;
    return strcmp(e, "0") != 0 &&
           strcmp(e, "false") != 0 &&
           strcmp(e, "FALSE") != 0 &&
           strcmp(e, "off") != 0 &&
           strcmp(e, "OFF") != 0 &&
           strcmp(e, "no") != 0 &&
           strcmp(e, "NO") != 0;
}

/* View-transform opt-in (NUSD_VIEW_TRANSFORM=1): bit 1 of the SceneHeader
 * has_materials word selects AgX over the default ACES tonemap in raytrace.metal.
 * Default off, so other consumers (e.g. IsaacLab) keep the existing ACES look. */
static uint32_t view_transform_bit(void)
{
    return env_flag_enabled("NUSD_VIEW_TRANSFORM") ? 2u : 0u;
}

static int env_int_clamped(const char* name, int fallback, int min_value, int max_value)
{
    const char* e = getenv(name);
    if (!e || !e[0]) return fallback;
    char* end = NULL;
    long v = strtol(e, &end, 10);
    if (end == e) return fallback;
    if (v < min_value) v = min_value;
    if (v > max_value) v = max_value;
    return (int)v;
}

/* stb_image — for HDR (Phase 7 IBL). When materials are enabled,
 * material.cpp provides the implementation; only define it here when
 * materials are off so we don't link duplicate symbols. */
#ifndef NUSD_ENABLE_MATERIALS
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#endif
#include "stb_image.h"

/* ---- Internal types ---- */

struct GpuBuffer_s {
    id<MTLBuffer> buffer;
    uint64_t      size;
    int           private_storage;
};

struct GpuPipeline_s {
    id<MTLRenderPipelineState> render;
    id<MTLDepthStencilState>   depth_stencil;
    int                        is_shadow;
};

typedef struct {
    float center[3];
    float opacity;
    float axis0[3];
    float inv_s0;
    float axis1[3];
    float inv_s1;
    float axis2[3];
    float inv_s2;
    float kernel2;
    float _pad[3];
} GpuGsParticle;

typedef struct {
    float min[3];
    float max[3];
} GpuGsAabb;

static const float GS_ICOSA_VRT_SCALE = 1.0704662767622f;
static const float GS_RT_SOFT_MAX_SIGMA_FRACTION = 0.00125f;
static const uint32_t GS_RT_SOFT_MAX_SIGMA_MIN_COUNT = 4096u;

static const float GS_ICOSA_VERTS[12][3] = {
    {-0.618034f,  1.0f,       0.0f},
    { 0.618034f,  1.0f,       0.0f},
    { 0.0f,       0.618034f, -1.0f},
    {-1.0f,       0.0f,      -0.618034f},
    {-1.0f,       0.0f,       0.618034f},
    { 0.0f,       0.618034f,  1.0f},
    { 1.0f,       0.0f,       0.618034f},
    { 0.0f,      -0.618034f,  1.0f},
    {-0.618034f, -1.0f,       0.0f},
    { 0.0f,      -0.618034f, -1.0f},
    { 1.0f,       0.0f,      -0.618034f},
    { 0.618034f, -1.0f,       0.0f},
};

static const uint32_t GS_ICOSA_INDS[60] = {
    0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 1,
    6, 1, 5, 6, 5, 7, 6, 7, 11, 6, 11, 10, 6, 10, 1,
    8, 4, 3, 8, 3, 9, 8, 9, 11, 8, 11, 7, 8, 7, 4,
    9, 3, 2, 9, 2, 10, 9, 10, 11, 5, 4, 7, 1, 10, 2
};

typedef struct {
    float    view_inv[16];
    float    proj_inv[16];
    uint32_t particle_count;
    uint32_t sh_degree;
    uint32_t k;
    uint32_t max_passes;
    float    min_transmittance;
    float    iso_opacity_threshold;
    uint32_t color_space;
    uint32_t _pad0;
} GpuGsPushConstants;

struct Gpu {
    /* Core */
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;

    /* Compiled at gpu_init from `src/shaders/...metal` — holds named functions
     * (vertex_mesh, fragment_mesh, fragment_mesh_shadow, etc.) that
     * gpu_create_pipeline picks up by hard-coded name. */
    id<MTLLibrary>      shader_library;

    /* Render target dimensions */
    int width;
    int height;
    int headless;

    /* Offscreen render targets (recreated on size change) */
    id<MTLTexture>      color_target;     /* RGBA8 unorm — raster + RT output target */

    /* Reduced-resolution orbit path (NUSD_ORBIT_RENDER_SCALE, default 1.0 =
     * off). When scale < 1 the single-camera RT traces into color_target_lo
     * (rw x rh) and an MPS bilinear pass upscales it into color_target (full
     * res) before readback — fewer rays + smaller readback during motion. The
     * RT kernel reads its resolution from the output texture's dims, so a
     * smaller bind + grid is all it takes. Viewport/single-camera only; the
     * tiled path uses its own lifecycle and is unaffected. */
    id<MTLTexture>      color_target_lo;
    float               render_scale;     /* 1.0 = full res (disabled) */
    int                 rw, rh;           /* reduced render dims when scaled */
    MPSImageBilinearScale* upscaler;      /* lazy; lo -> full bilinear */
    id<MTLTexture>      depth_target;     /* Depth32 — z-buffer (also sampled by SSAO) */
    id<MTLTexture>      post_target;      /* RGBA8 — SSAO composite output */
    MTLPixelFormat      color_format;
    MTLPixelFormat      depth_format;

    /* SSAO post-process. ssao_active = the last frame ran the SSAO pass, so
     * gpu_screenshot reads post_target instead of color_target. */
    id<MTLRenderPipelineState> ssao_pipeline;
    int                        ssao_active;

    /* Shared-storage buffer used to read color_target back to the CPU. Sized
     * for `width * height * 4` bytes; reallocated in gpu_resize. */
    id<MTLBuffer>       readback_buffer;

    /* Async single-camera readback (NUSD_VIEWPORT_ASYNC): a ping-pong twin of
     * readback_buffer. In async mode the trace + color->readback blit are
     * committed WITHOUT waiting, and gpu_readback_pixels returns the previous
     * (already-complete) frame, so the CPU never blocks on the GPU (1-frame
     * latency). Viewport-only; the tiled path uses its own lifecycle and is
     * unaffected. Default off (deterministic full-res sync path unchanged). */
    id<MTLBuffer>       readback_buffer2;
    int                 async_readback;
    id<MTLCommandBuffer> rb_cmd[2];   /* command buffer that writes rb[i] */
    int                 rb_valid[2];  /* rb[i] holds a submitted frame */
    int                 rb_last;      /* index most recently written (0/1) */

    /* Per-frame (set between gpu_begin_frame and gpu_end_frame, or the RT
     * equivalents). Only one of in_frame / in_frame_rt is set at a time. */
    id<MTLCommandBuffer>           current_cmd;
    id<MTLRenderCommandEncoder>    current_render_enc;
    id<MTLComputeCommandEncoder>   current_compute_enc;
    GpuPipeline_s*                 current_pipeline;
    GpuBuffer                      current_vertex_buffer;
    GpuBuffer                      current_index_buffer;
    int                            in_frame;
    int                            in_frame_rt;
    int                            has_frame;       /* 1 = at least one frame complete */

    /* Capability bits */
    int rt_available;        /* MTLDevice.supportsRaytracing */
    int rt_built;            /* set when gpu_build_rt_scene succeeds */

    /* Phase 4 RT — persistent (compiled once, reused across scene rebuilds) */
    id<MTLLibrary>                rt_library;
    id<MTLComputePipelineState>   rt_pipeline;

    /* Phase 4 RT — per-scene (rebuilt by gpu_build_rt_scene, freed by
     * gpu_destroy_rt_scene). NSMutableArray holds strong refs to BLAS objects;
     * passed verbatim as TLAS instancedAccelerationStructures. */
    NSMutableArray<id<MTLAccelerationStructure>>* blas_list;
    uint32_t                          blas_count;
    id<MTLAccelerationStructure>      tlas;
    id<MTLBuffer>                     instance_buf;     /* Shared */
    id<MTLBuffer>                     scene_data_buf;   /* Shared */
    id<MTLBuffer>                     light_buf;        /* Shared: 16-byte header + GpuLight[] */
    int                               light_count;
    id<MTLBuffer>                     tlas_scratch;     /* Private */
    id<MTLBuffer>                     rt_vertex_buf;    /* shared with renderer.c */
    id<MTLBuffer>                     rt_index_buf;
    uint32_t                          scene_nmeshes;
    uint32_t*                         mesh_to_blas;     /* size = scene_nmeshes */

    /* Phase 11.A — BasisCurves data + AABB BLAS. Uploaded by
     * gpu_upload_curve_data, BLAS built by gpu_build_curve_blas. Wired
     * into the unified TLAS as one extra instance with mask = 0xFE (so
     * shadow rays at mask 0x01 skip it; mesh instances stay at 0xFF and
     * mask AND 0x01 ≠ 0). The intersection function table dispatches
     * AABB hits into curve_isect. */
    id<MTLBuffer>                     curve_seg_buf;    /* Shared, 32 B/seg */
    id<MTLBuffer>                     curve_color_buf;  /* Shared, 16 B/seg (vec4 padded) */
    id<MTLBuffer>                     curve_aabb_buf;   /* Shared, 24 B/AABB (matches MTLAxisAlignedBoundingBox) */
    id<MTLAccelerationStructure>      curve_blas;
    uint32_t                          curve_seg_count;
    id<MTLRenderPipelineState>        curve_raster_pipeline;
    id<MTLDepthStencilState>          curve_raster_depth;

    /* Gaussian splat RT path. Particles are packed as world-space ellipsoid
     * data and accelerated with one unit-icosahedron BLAS plus one TLAS
     * instance per particle, matching vk_gaussian_splatting's default
     * front-surface ordering. */
    id<MTLLibrary>                    gs_library;
    id<MTLComputePipelineState>       gs_pipeline;
    id<MTLIntersectionFunctionTable>  gs_ift;
    id<MTLBuffer>                     gs_particle_buf;  /* Shared GpuGsParticle[] */
    id<MTLBuffer>                     gs_sh_buf;        /* Shared float3 SH coeffs */
    id<MTLBuffer>                     gs_aabb_buf;      /* Shared GpuGsAabb[] */
    id<MTLBuffer>                     gs_ico_vertex_buf;
    id<MTLBuffer>                     gs_ico_index_buf;
    id<MTLBuffer>                     gs_instance_buf;  /* Shared MTLAccelerationStructureInstanceDescriptor[] */
    id<MTLBuffer>                     gs_tlas_scratch;  /* Private */
    id<MTLAccelerationStructure>      gs_blas;
    id<MTLAccelerationStructure>      gs_tlas;
    uint32_t                          gs_particle_count;
    uint32_t                          gs_sh_degree;
    id<MTLBuffer>                     gs_depth_buf;     /* Shared width*height float */
    id<MTLBuffer>                     gs_normal_buf;    /* Shared width*height float3 */
    uint32_t                          gs_output_w;
    uint32_t                          gs_output_h;

    /* Lazily compiled compute kernel that fills curve_aabb_buf from
     * curve_seg_buf on the GPU. Replaces the host-side AABB upload
     * (Vulkan port's Phase 12.x) when run; cuts ~24 B × seg-count
     * worth of CPU→GPU traffic at scene load. */
    id<MTLComputePipelineState>       curve_aabb_pipeline;

    /* Phase 11.A — Intersection Function Tables, one per RT compute
     * pipeline. Both tables hold curve_isect at slot 0; the IFT also
     * holds curve_seg_buf at MSL [[buffer(11)]] for the intersection
     * function to read. Created on demand in ensure_curve_ift /
     * ensure_tiled_curve_ift; lifetime tied to the parent pipeline. */
    id<MTLIntersectionFunctionTable>  curve_ift;
    id<MTLIntersectionFunctionTable>  tiled_curve_ift;
    /* Tiny dummy buffer used as a placeholder for curve_seg / curve_color
     * kernel bindings when no curves are present in the scene. Lets us
     * keep one kernel + one pipeline that works with or without curves. */
    id<MTLBuffer>                     curve_dummy_buf;

    /* Bookkeeping for gpu_get_allocated_memory */
    uint64_t allocated_bytes;

    /* Tiled RT bookkeeping (set by gpu_tiled_init in Phase 5).
     * MVP scope: single-buffered output (tiled_last_slot == 0 always);
     * the slot API is honored trivially. Phase 5b can promote to
     * double-buffered tiled_color_buf[2] for CPU/GPU overlap. */
    uint32_t tiled_total_w;
    uint32_t tiled_total_h;
    int      tiled_num_cameras;
    int      tiled_last_slot;

    /* Phase 5 — persistent (compiled once, allocated lazily on first
     * gpu_tiled_init, resized when total_w/total_h/num_cameras change). */
    id<MTLComputePipelineState> tiled_rt_pipeline;
    id<MTLBuffer>               tiled_camera_buf;     /* Shared: num_cams * 32 floats */
    id<MTLBuffer>               tiled_color_buf;      /* Shared: total_w*total_h*4 B (uchar4) */
    id<MTLBuffer>               tiled_depth_buf;      /* Shared: total_w*total_h*4 B (float) */
    id<MTLBuffer>               tiled_seg_buf;        /* Shared: total_w*total_h*4 B (uint) */
    id<MTLBuffer>               tiled_normals_buf;    /* Shared: total_w*total_h*12 B (float3) */
    int                         tiled_camera_capacity;
    id<MTLCommandBuffer>        tiled_inflight_cmd;   /* commit'd by end_frame_tiled_rt; mapped/awaited later */
    int                         in_frame_rt_tiled;

    /* Direct-write / staging-skip flags (Phase 5+) */
    int  direct_write_enabled;
    int  skip_staging;

    /* Environment / IBL (Phase 7).
     * env_texture: equirectangular HDR env map, RGBA16Float, mipmapped
     *   (mip 0 = full res, deeper mips drive specular IBL roughness blur).
     * irr_texture: SH-rendered diffuse irradiance map, 256x128 RGBA16Float
     *   (cosine-convolved environment — Ramamoorthi & Hanrahan 2001).
     * brdf_lut:    BRDF integration LUT, 128x128 RG16Float (split-sum
     *   approximation: scale + bias for Schlick Fresnel × G_Vis).
     * env_mip_levels: 0 == "no IBL loaded" (kernel falls back to
     *   synthetic 3-point lighting). */
    int             env_mip_levels;
    float           env_intensity;   /* DomeLight inputs:intensity (1.0 default) */
    float           dome_color[3];   /* Flat fallback DomeLight color */
    float           dome_intensity;  /* <=0 means use procedural fallback */
    id<MTLTexture>  env_texture;
    id<MTLTexture>  irr_texture;
    id<MTLTexture>  brdf_lut;

    /* Environment background — fullscreen-triangle pass that paints the
     * GGX-prefiltered HDR onto background pixels (z=0.9999 NDC, behind
     * any actual geometry). Without this the raster viewport shows
     * gpu_begin_frame's clear color (sky-blue) where the authored
     * DomeLight should be visible. Built lazily on first
     * gpu_create_env_bg_pipeline call after the shader_library exists.
     * Mirrors nanousd-opengl-renderer/src/gpu_opengles.c:env_bg_program. */
    id<MTLRenderPipelineState>    env_bg_pipeline;

    /* Phase 7b — materials + textures. Bound at fixed slots in the RT
     * kernels: material_buf at buffer(16), RT texture array at texture(4)
     * up to MATERIAL_TEXTURE_SLOT_COUNT entries. The shader's MaterialParams
     * layout matches GpuMaterialParams (gpu.h); the shader indexes
     * mat_textures[mat.tex_indices[slot]] when slot value >= 0. */
    id<MTLBuffer>                          material_buf;     /* Shared, GpuMaterialParams[] */
    NSMutableArray<id<MTLTexture>>*        material_textures;
    id<MTLArgumentEncoder>                 material_arg_encoder;
    id<MTLBuffer>                          material_arg_buffer; /* RT argument-buffer texture table */
    id<MTLTexture>                         material_dummy_tex;  /* 1x1 placeholder for unbound slots */
    id<MTLSamplerState>                    material_sampler;
    uint32_t                               material_count;
    uint32_t                               texture_count;
    int                                    has_materials;
    /* 1 iff material_count==1 because gpu_upload_materials was given 0
     * real materials and synthesized a placeholder for descriptor wiring.
     * The shader's `has_materials` SceneHeader gate must read 0 in that
     * case so rchit/raster fall back to per-mesh displayColor instead of
     * reading the placeholder slot for hits that need real material data. */
    int                                    mat_only_placeholder;
};

/* ---- Helpers ---- */

static int gpu_color_target_is_f16(Gpu* gpu)
{
    return gpu && gpu->color_format == MTLPixelFormatRGBA16Float;
}

static NSUInteger gpu_color_bytes_per_pixel(Gpu* gpu)
{
    return gpu_color_target_is_f16(gpu) ? 8u : 4u;
}

static float half_to_float_mtl(uint16_t h)
{
    const int sign = (h & 0x8000) ? -1 : 1;
    const int exp = (h >> 10) & 0x1f;
    const int mant = h & 0x03ff;
    if (exp == 0) {
        if (mant == 0) return sign < 0 ? -0.0f : 0.0f;
        return sign * ldexpf((float)mant, -24);
    }
    if (exp == 31) return sign > 0 ? 1.0f : -1.0f;
    return sign * ldexpf(1.0f + (float)mant / 1024.0f, exp - 15);
}

static float srgb_u8_to_linear(uint8_t v)
{
    float s = (float)v / 255.0f;
    return (s <= 0.04045f) ? (s / 12.92f) : powf((s + 0.055f) / 1.055f, 2.4f);
}

static uint8_t unorm_u8(float v)
{
    if (!(v > 0.0f)) return 0;
    if (v >= 1.0f) return 255;
    return (uint8_t)(v * 255.0f + 0.5f);
}

static uint8_t linear_to_srgb_u8(float v)
{
    if (!(v > 0.0f)) return 0;
    if (v >= 1.0f) return 255;
    float s = (v <= 0.0031308f)
            ? (12.92f * v)
            : (1.055f * powf(v, 1.0f / 2.4f) - 0.055f);
    return unorm_u8(s);
}

static void readback_storage_to_rgba8(Gpu* gpu, const void* src_void,
                                      uint8_t* dst, size_t npx)
{
    if (!gpu_color_target_is_f16(gpu)) {
        memcpy(dst, src_void, npx * 4);
        return;
    }
    const uint16_t* src = (const uint16_t*)src_void;
    for (size_t i = 0; i < npx; i++) {
        dst[i * 4 + 0] = linear_to_srgb_u8(half_to_float_mtl(src[i * 4 + 0]));
        dst[i * 4 + 1] = linear_to_srgb_u8(half_to_float_mtl(src[i * 4 + 1]));
        dst[i * 4 + 2] = linear_to_srgb_u8(half_to_float_mtl(src[i * 4 + 2]));
        dst[i * 4 + 3] = unorm_u8(half_to_float_mtl(src[i * 4 + 3]));
    }
}

static void readback_storage_to_rgba32f(Gpu* gpu, const void* src_void,
                                        float* dst, size_t npx)
{
    if (gpu_color_target_is_f16(gpu)) {
        const uint16_t* src = (const uint16_t*)src_void;
        for (size_t i = 0; i < npx * 4; i++) dst[i] = half_to_float_mtl(src[i]);
        return;
    }
    const uint8_t* src = (const uint8_t*)src_void;
    for (size_t i = 0; i < npx; i++) {
        dst[i * 4 + 0] = srgb_u8_to_linear(src[i * 4 + 0]);
        dst[i * 4 + 1] = srgb_u8_to_linear(src[i * 4 + 1]);
        dst[i * 4 + 2] = srgb_u8_to_linear(src[i * 4 + 2]);
        dst[i * 4 + 3] = (float)src[i * 4 + 3] / 255.0f;
    }
}

static FILE* open_ppm_for_write(const char* path, uint32_t w, uint32_t h)
{
    FILE* f = fopen(path, "wb");
    if (!f) return NULL;
    fprintf(f, "P6\n%u %u\n255\n", w, h);
    return f;
}

/* Read entire file into an NSString. Returns nil on failure. */
static NSString* read_text_file(const char* path)
{
    NSError* err = nil;
    NSString* contents = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path]
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
    if (!contents) {
        fprintf(stderr, "gpu_metal: failed to read %s: %s\n",
                path, err ? [[err localizedDescription] UTF8String] : "(unknown)");
        return nil;
    }
    return contents;
}

/* Compile a single MSL source file into an MTLLibrary. */
static id<MTLLibrary> compile_msl_file(id<MTLDevice> device, const char* path)
{
    NSString* src = read_text_file(path);
    if (!src) return nil;

    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.fastMathEnabled = YES;
    if (@available(macOS 13.0, *)) {
        opts.languageVersion = MTLLanguageVersion3_0;
    }

    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "gpu_metal: MSL compile failed for %s: %s\n",
                path, err ? [[err localizedDescription] UTF8String] : "(unknown)");
        return nil;
    }
    return lib;
}

/* (Re)create offscreen color + depth render targets at the current size. */
static void recreate_render_targets(Gpu* gpu)
{
    if (gpu->width <= 0 || gpu->height <= 0) return;

    MTLTextureDescriptor* cd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:gpu->color_format
                                     width:(NSUInteger)gpu->width
                                    height:(NSUInteger)gpu->height
                                 mipmapped:NO];
    /* ShaderWrite is needed for the RT compute kernel's outputImage write;
     * ShaderRead lets gpu_screenshot blit the texture into readback_buffer. */
    cd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    cd.storageMode = MTLStorageModePrivate;
    gpu->color_target = [gpu->device newTextureWithDescriptor:cd];

    /* Reduced-res orbit target (NUSD_ORBIT_RENDER_SCALE). Read here so init +
     * resize keep it sized. scale >= 1.0 (or unset) disables it entirely — the
     * full-res deterministic path is byte-for-byte unchanged. */
    gpu->color_target_lo = nil;
    gpu->render_scale = 1.0f;
    gpu->rw = gpu->width;
    gpu->rh = gpu->height;
    {
        const char* rs = getenv("NUSD_ORBIT_RENDER_SCALE");
        float s = rs ? (float)atof(rs) : 1.0f;
        if (s >= 0.25f && s < 1.0f) {
            gpu->render_scale = s;
            gpu->rw = (int)(gpu->width  * s + 0.5f); if (gpu->rw < 1) gpu->rw = 1;
            gpu->rh = (int)(gpu->height * s + 0.5f); if (gpu->rh < 1) gpu->rh = 1;
            MTLTextureDescriptor* ld = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:gpu->color_format
                                             width:(NSUInteger)gpu->rw
                                            height:(NSUInteger)gpu->rh
                                         mipmapped:NO];
            ld.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            ld.storageMode = MTLStorageModePrivate;
            gpu->color_target_lo = [gpu->device newTextureWithDescriptor:ld];
        }
    }

    MTLTextureDescriptor* dd = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:gpu->depth_format
                                     width:(NSUInteger)gpu->width
                                    height:(NSUInteger)gpu->height
                                 mipmapped:NO];
    /* ShaderRead so the SSAO post-pass can sample the z-buffer. */
    dd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    dd.storageMode = MTLStorageModePrivate;
    gpu->depth_target = [gpu->device newTextureWithDescriptor:dd];

    /* SSAO composite output (only allocated lazily by the pass would be nicer,
     * but size tracking is simplest here). */
    MTLTextureDescriptor* pd2 = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:gpu->color_format
                                     width:(NSUInteger)gpu->width
                                    height:(NSUInteger)gpu->height
                                 mipmapped:NO];
    pd2.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    pd2.storageMode = MTLStorageModePrivate;
    gpu->post_target = [gpu->device newTextureWithDescriptor:pd2];

    /* Readback buffer sized for color target. Shared storage on Apple Silicon
     * means CPU sees the buffer with no copy; on Intel Macs it's still
     * allocated in shared memory. Counted toward allocated_bytes so that
     * gpu_get_allocated_memory reports nonzero immediately after gpu_init —
     * matches the Vulkan port (which allocates readback_buf at init time). */
    if (gpu->readback_buffer && gpu->allocated_bytes >= [gpu->readback_buffer length]) {
        gpu->allocated_bytes -= [gpu->readback_buffer length];
    }
    NSUInteger row_bytes = (NSUInteger)gpu->width * gpu_color_bytes_per_pixel(gpu);
    NSUInteger total = row_bytes * (NSUInteger)gpu->height;
    gpu->readback_buffer = [gpu->device newBufferWithLength:total
                                                    options:MTLResourceStorageModeShared];
    gpu->allocated_bytes += total;

    /* Twin readback buffer for the async viewport path (NUSD_VIEWPORT_ASYNC).
     * Allocated here so init + resize keep both buffers correctly sized. Any
     * in-flight async cmd referencing the old buffers keeps them alive via its
     * own retain, so reallocating mid-stream is safe; invalidate the pipeline
     * state so the next readback re-syncs against the new buffers. */
    if (gpu->readback_buffer2 && gpu->allocated_bytes >= [gpu->readback_buffer2 length]) {
        gpu->allocated_bytes -= [gpu->readback_buffer2 length];
    }
    gpu->readback_buffer2 = [gpu->device newBufferWithLength:total
                                                     options:MTLResourceStorageModeShared];
    gpu->allocated_bytes += total;
    gpu->rb_valid[0] = gpu->rb_valid[1] = 0;
    gpu->rb_cmd[0] = gpu->rb_cmd[1] = nil;
    gpu->rb_last = 0;
    {
        const char* a = getenv("NUSD_VIEWPORT_ASYNC");
        gpu->async_readback = (a && a[0] && a[0] != '0') ? 1 : 0;
    }
}

/* ---- Lifecycle ---- */

Gpu* gpu_init(void* /*glfw_window*/, int width, int height)
{
    /* `new Gpu()` value-initializes — POD fields are zeroed, ARC id fields are nil.
     * Don't memset over the struct: that would clobber ARC's strong references. */
    Gpu* gpu = new Gpu();
    gpu->device = MTLCreateSystemDefaultDevice();
    if (!gpu->device) {
        fprintf(stderr, "gpu_metal: MTLCreateSystemDefaultDevice() returned nil\n");
        delete gpu;
        return nullptr;
    }
    gpu->queue = [gpu->device newCommandQueue];
    if (!gpu->queue) {
        fprintf(stderr, "gpu_metal: newCommandQueue failed\n");
        delete gpu;
        return nullptr;
    }
    gpu->width  = width;
    gpu->height = height;
    gpu->headless = 0;
    /* Default: sRGB color target matching the Vulkan swapchain format
     * (`VK_FORMAT_B8G8R8A8_SRGB`). Opt-in high-quality single-camera paths can
     * request RGBA16Float via NUSD_COLOR_TARGET_F16=1, preserving display-linear
     * LdrColor until readback/viewport upload instead of quantizing on store. */
    const char* f16 = getenv("NUSD_COLOR_TARGET_F16");
    gpu->color_format = (f16 && f16[0] &&
                         !(f16[0] == '0' || f16[0] == 'f' || f16[0] == 'F' ||
                           f16[0] == 'n' || f16[0] == 'N'))
                      ? MTLPixelFormatRGBA16Float
                      : MTLPixelFormatRGBA8Unorm_sRGB;
    gpu->depth_format = MTLPixelFormatDepth32Float;

    if (@available(macOS 11.0, *)) {
        gpu->rt_available = [gpu->device supportsRaytracing] ? 1 : 0;
    } else {
        gpu->rt_available = 0;
    }

    /* Compile mesh shaders. `SHADER_DIR` is set by CMake; we look for
     * mesh.metal there (CMake's `renderer_shaders` target stages it). */
    char mesh_path[1024];
    snprintf(mesh_path, sizeof(mesh_path), "%s/mesh.metal", SHADER_DIR);
    gpu->shader_library = compile_msl_file(gpu->device, mesh_path);
    if (!gpu->shader_library) {
        fprintf(stderr, "gpu_metal: WARNING — mesh.metal failed to compile; raster pipeline will be unavailable\n");
        /* Continue without library — gpu_create_pipeline returns NULL; callers
         * (renderer.c) handle the fallback. */
    }

    recreate_render_targets(gpu);

    fprintf(stderr, "gpu_metal: device=%s rt=%s shaders=%s\n",
            [[gpu->device name] UTF8String],
            gpu->rt_available ? "yes" : "no",
            gpu->shader_library ? "ok" : "missing");

    return gpu;
}

void gpu_shutdown(Gpu* gpu)
{
    if (!gpu) return;
    /* ARC releases the id fields when the struct is destroyed. */
    delete gpu;
}

void gpu_resize(Gpu* gpu, int width, int height)
{
    if (!gpu) return;
    if (width == gpu->width && height == gpu->height) return;
    gpu->width = width;
    gpu->height = height;
    recreate_render_targets(gpu);
}

void gpu_set_headless(Gpu* gpu, int headless)
{
    if (!gpu) return;
    gpu->headless = headless;
}

/* ---- Resources ---- */

GpuBuffer gpu_create_buffer(Gpu* gpu, const GpuBufferDesc* desc)
{
    if (!gpu || !desc || desc->size == 0) return nullptr;

    MTLResourceOptions opts = MTLResourceStorageModeShared;
    /* On Apple Silicon, Shared storage is GPU-accessible at no copy cost.
     * On Intel Macs, Managed would be more efficient for static data, but
     * Shared still works. Phase 2 may switch to Managed for vertex/index
     * buffers when running on Intel. */

    id<MTLBuffer> buf = nil;
    if (desc->data) {
        buf = [gpu->device newBufferWithBytes:desc->data
                                       length:(NSUInteger)desc->size
                                      options:opts];
    } else {
        buf = [gpu->device newBufferWithLength:(NSUInteger)desc->size
                                       options:opts];
    }
    if (!buf) return nullptr;

    GpuBuffer_s* gb = new GpuBuffer_s();
    gb->buffer = buf;
    gb->size = desc->size;
    gb->private_storage = 0;
    gpu->allocated_bytes += desc->size;
    return gb;
}

GpuBuffer gpu_create_private_buffer(Gpu* gpu, GpuBufferUsage /*usage*/,
                                    uint64_t size)
{
    if (!gpu || size == 0) return nullptr;

    id<MTLBuffer> buf = [gpu->device newBufferWithLength:(NSUInteger)size
                                                 options:MTLResourceStorageModePrivate];
    if (!buf) return nullptr;

    GpuBuffer_s* gb = new GpuBuffer_s();
    gb->buffer = buf;
    gb->size = size;
    gb->private_storage = 1;
    gpu->allocated_bytes += size;
    return gb;
}

void gpu_destroy_buffer(Gpu* gpu, GpuBuffer buf)
{
    if (!gpu || !buf) return;
    if (gpu->allocated_bytes >= buf->size)
        gpu->allocated_bytes -= buf->size;
    delete buf;  /* ARC releases the MTLBuffer */
}

void* gpu_buffer_contents(GpuBuffer buf)
{
    if (!buf || buf->private_storage) return nullptr;
    return [buf->buffer contents];
}

int gpu_update_buffer(Gpu* gpu, GpuBuffer buf, const void* data,
                      uint64_t size, uint64_t offset)
{
    if (!gpu || !buf || !data) return 0;
    if (offset > buf->size || size > buf->size - offset) return 0;
    if (buf->private_storage) {
        id<MTLBuffer> stage = [gpu->device newBufferWithBytes:data
                                                       length:(NSUInteger)size
                                                      options:MTLResourceStorageModeShared];
        if (!stage) return 0;
        id<MTLCommandBuffer> cmd = [gpu->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:stage
                sourceOffset:0
                    toBuffer:buf->buffer
           destinationOffset:(NSUInteger)offset
                        size:(NSUInteger)size];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return cmd.status == MTLCommandBufferStatusCompleted;
    }
    memcpy((uint8_t*)[buf->buffer contents] + offset, data, (size_t)size);
    return 1;
}

int gpu_copy_buffer(Gpu* gpu, GpuBuffer src, uint64_t src_offset,
                    GpuBuffer dst, uint64_t dst_offset, uint64_t size)
{
    if (!gpu || !src || !dst) return 0;
    if (src_offset > src->size || size > src->size - src_offset) return 0;
    if (dst_offset > dst->size || size > dst->size - dst_offset) return 0;
    if (size == 0) return 1;

    id<MTLCommandBuffer> cmd = [gpu->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:src->buffer
            sourceOffset:(NSUInteger)src_offset
                toBuffer:dst->buffer
       destinationOffset:(NSUInteger)dst_offset
                    size:(NSUInteger)size];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted;
}

static MTLVertexFormat mtl_vertex_format(GpuVertexFormat format)
{
    switch (format) {
        case GPU_FORMAT_FLOAT3: return MTLVertexFormatFloat3;
        case GPU_FORMAT_FLOAT2: return MTLVertexFormatFloat2;
        case GPU_FORMAT_UINT:   return MTLVertexFormatUInt;
        case GPU_FORMAT_FLOAT:  return MTLVertexFormatFloat;
        default:                return MTLVertexFormatInvalid;
    }
}

/* Build an MTLRenderPipelineState for a vertex+fragment function pair.
 * Returns nil on failure. */
static GpuPipeline_s* build_render_pipeline(Gpu* gpu,
                                            NSString* vertex_fn_name,
                                            NSString* fragment_fn_name,
                                            const GpuPipelineDesc* desc)
{
    if (!gpu->shader_library) return nullptr;

    id<MTLFunction> vfn = [gpu->shader_library newFunctionWithName:vertex_fn_name];
    id<MTLFunction> ffn = [gpu->shader_library newFunctionWithName:fragment_fn_name];
    if (!vfn || !ffn) {
        fprintf(stderr, "gpu_metal: missing function (vertex=%s fragment=%s)\n",
                [vertex_fn_name UTF8String], [fragment_fn_name UTF8String]);
        return nullptr;
    }

    MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction   = vfn;
    pd.fragmentFunction = ffn;
    pd.colorAttachments[0].pixelFormat = gpu->color_format;
    pd.depthAttachmentPixelFormat      = gpu->depth_format;

    /* Vertex descriptor mirrors the RHI pipeline desc. renderer.c uses either
     * the material 48-byte layout or the compact 36-byte geometry layout. */
    MTLVertexDescriptor* vd = [MTLVertexDescriptor new];
    uint32_t stride = (desc && desc->vertex_stride) ? desc->vertex_stride : 48u;
    if (desc && desc->attribs && desc->nattribs > 0) {
        for (uint32_t i = 0; i < desc->nattribs; i++) {
            uint32_t loc = desc->attribs[i].location;
            if (loc >= 31) continue;
            vd.attributes[loc].format = mtl_vertex_format(desc->attribs[i].format);
            vd.attributes[loc].offset = desc->attribs[i].offset;
            vd.attributes[loc].bufferIndex = 0;
        }
    } else {
        vd.attributes[0].format = MTLVertexFormatFloat3;
        vd.attributes[0].offset = 0;
        vd.attributes[0].bufferIndex = 0;
        vd.attributes[1].format = MTLVertexFormatFloat3;
        vd.attributes[1].offset = 12;
        vd.attributes[1].bufferIndex = 0;
        vd.attributes[2].format = MTLVertexFormatFloat2;
        vd.attributes[2].offset = 36;
        vd.attributes[2].bufferIndex = 0;
    }
    vd.layouts[0].stride       = stride;
    vd.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    pd.vertexDescriptor = vd;

    NSError* err = nil;
    id<MTLRenderPipelineState> pso =
        [gpu->device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!pso) {
        fprintf(stderr, "gpu_metal: pipeline build failed (%s/%s): %s\n",
                [vertex_fn_name UTF8String], [fragment_fn_name UTF8String],
                err ? [[err localizedDescription] UTF8String] : "(unknown)");
        return nullptr;
    }

    MTLDepthStencilDescriptor* dsd = [MTLDepthStencilDescriptor new];
    dsd.depthCompareFunction = MTLCompareFunctionLess;
    dsd.depthWriteEnabled    = YES;
    id<MTLDepthStencilState> dss = [gpu->device newDepthStencilStateWithDescriptor:dsd];

    GpuPipeline_s* p = new GpuPipeline_s();
    p->render        = pso;
    p->depth_stencil = dss;
    p->is_shadow     = 0;
    return p;
}

GpuPipeline gpu_create_pipeline(Gpu* gpu, const GpuPipelineDesc* desc)
{
    if (!gpu) return nullptr;
    const char* geom_env = getenv("NUSD_GEOM_RASTER_PIPE");
    int geom_only = geom_env && geom_env[0] && geom_env[0] != '0';
    return build_render_pipeline(gpu, @"vertex_mesh",
                                 geom_only ? @"fragment_mesh_geom" : @"fragment_mesh",
                                 desc);
}

GpuPipeline gpu_create_shadow_pipeline(Gpu* gpu, const GpuPipelineDesc* desc)
{
    if (!gpu) return nullptr;
    GpuPipeline_s* p = build_render_pipeline(gpu, @"vertex_mesh", @"fragment_mesh_shadow", desc);
    if (p) p->is_shadow = 1;
    return p;
}

void gpu_destroy_pipeline(Gpu* /*gpu*/, GpuPipeline pipe)
{
    if (pipe) delete pipe;
}

/* ---- Frame ---- */

int gpu_begin_frame(Gpu* gpu)
{
    if (!gpu || !gpu->color_target) return 0;
    if (gpu->in_frame) return 0;

    gpu->current_cmd = [gpu->queue commandBuffer];

    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture     = gpu->color_target;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    /* Clear to a soft sky-ish blue — matches sky palette in the fragment
     * shader and the Vulkan raster path. */
    rpd.colorAttachments[0].clearColor  = MTLClearColorMake(0.50, 0.58, 0.75, 1.0);

    rpd.depthAttachment.texture     = gpu->depth_target;
    rpd.depthAttachment.loadAction  = MTLLoadActionClear;
    /* Store the z-buffer so the SSAO post-pass can sample it. */
    rpd.depthAttachment.storeAction = MTLStoreActionStore;
    rpd.depthAttachment.clearDepth  = 1.0;
    gpu->ssao_active = 0;

    gpu->current_render_enc = [gpu->current_cmd renderCommandEncoderWithDescriptor:rpd];
    [gpu->current_render_enc setFrontFacingWinding:MTLWindingCounterClockwise];
    [gpu->current_render_enc setCullMode:MTLCullModeNone];

    gpu->in_frame = 1;
    gpu->current_pipeline      = nullptr;
    gpu->current_vertex_buffer = nullptr;
    gpu->current_index_buffer  = nullptr;
    return 1;
}

void gpu_end_frame(Gpu* gpu)
{
    if (!gpu || !gpu->in_frame) return;
    [gpu->current_render_enc endEncoding];
    gpu->current_render_enc = nil;

    [gpu->current_cmd commit];
    [gpu->current_cmd waitUntilCompleted];
    gpu->current_cmd = nil;

    gpu->in_frame  = 0;
    gpu->has_frame = 1;
}

/* Forward decls — defined below alongside the RT material binding. */
static id<MTLTexture>      ensure_material_dummy_tex(Gpu* gpu);
static id<MTLSamplerState> ensure_material_sampler(Gpu* gpu);

void gpu_cmd_bind_pipeline(Gpu* gpu, GpuPipeline pipe)
{
    if (!gpu || !gpu->in_frame || !pipe || !pipe->render) return;
    [gpu->current_render_enc setRenderPipelineState:pipe->render];
    if (pipe->depth_stencil)
        [gpu->current_render_enc setDepthStencilState:pipe->depth_stencil];
    gpu->current_pipeline = pipe;

    /* Bind material params SSBO + texture array for the raster fragment
     * shader. mesh.metal:fragment_mesh expects:
     *   buffer(2) → GpuMaterialParams[]
     *   buffer(3) → RasterFragUniform { has_materials, has_ibl, env_mip_levels, env_intensity }
     *   texture(0..63) → texture array (TEX_DIFFUSE etc. lookup)
     *   texture(64) → env_texture (GGX-prefiltered HDR)
     *   texture(65) → irr_texture (SH irradiance)
     *   sampler(0)
     * Bind defensively: a scene without materials still needs buffer(3)
     * with has_materials=0 so the shader's gate branch falls through. */
    ensure_material_dummy_tex(gpu);
    ensure_material_sampler(gpu);
    id<MTLBuffer> mat_buf = gpu->material_buf ? gpu->material_buf : gpu->curve_dummy_buf;
    if (mat_buf) {
        [gpu->current_render_enc setFragmentBuffer:mat_buf offset:0 atIndex:2];
    }
    /* Placeholder-only counts as "no materials" for shader gating: the
     * shader must fall back to per-mesh displayColor (use_vertex_color
     * branch) instead of reading the synthetic placeholder slot. */
    struct {
        uint32_t has_materials;
        uint32_t has_ibl;
        float    env_mip_levels;
        float    env_intensity;
        uint32_t debug_mode;
        /* mesh.metal uses uint3 padding; Metal aligns that member to a
         * 16-byte boundary, so reflection requires a 48-byte argument even
         * though the live fields above occupy only 20 bytes. */
        uint32_t _pad0;
        uint32_t _pad1;
        uint32_t _pad2;
        uint32_t light_count;  /* mesh.metal RasterFragUniform.light_count (offset 32) */
        uint32_t _pad4;
        uint32_t _pad5;
        uint32_t _pad6;
        float    dome_color[4];
    } frag_u = {
        (gpu->has_materials && !gpu->mat_only_placeholder) ? 1u : 0u,
        gpu->env_texture ? 1u : 0u,
        (float)(gpu->env_mip_levels > 0 ? gpu->env_mip_levels : 1),
        gpu->env_intensity > 0.0f ? gpu->env_intensity : 1.0f,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        {
            gpu->dome_color[0],
            gpu->dome_color[1],
            gpu->dome_color[2],
            gpu->dome_intensity > 0.0f ? gpu->dome_intensity : 0.0f,
        },
    };
    if (const char* dbg = getenv("NUSD_RASTER_DEBUG")) {
        if (!strcmp(dbg, "basecolor") || !strcmp(dbg, "albedo")) frag_u.debug_mode = 1u;
        else if (!strcmp(dbg, "metallic")) frag_u.debug_mode = 2u;
        else if (!strcmp(dbg, "roughness")) frag_u.debug_mode = 3u;
        else if (!strcmp(dbg, "normal")) frag_u.debug_mode = 4u;
        else if (!strcmp(dbg, "diffuse_ibl")) frag_u.debug_mode = 5u;
        else if (!strcmp(dbg, "specular_ibl")) frag_u.debug_mode = 6u;
        else if (!strcmp(dbg, "shadow_contact")) frag_u.debug_mode = 7u;
        else if (!strcmp(dbg, "shadow_spec")) frag_u.debug_mode = 8u;
        else if (!strcmp(dbg, "shadow_key")) frag_u.debug_mode = 9u;
        else if (!strcmp(dbg, "shadow_factor") || !strcmp(dbg, "shadow")) frag_u.debug_mode = 10u;
    }
    /* Authored scene-light count: mesh.metal gates the many-light/no-IBL
     * procedural-ambient attenuation (Isaac warehouse) on light_count > 8. */
    frag_u.light_count = (uint32_t)(gpu->light_count > 0 ? gpu->light_count : 0);
    [gpu->current_render_enc setFragmentBytes:&frag_u length:sizeof(frag_u) atIndex:3];

    /* Raster fragment binds the texture array at slot 0 — distinct from
     * the RT compute path which uses slot 4. The shader's
     * `array<texture2d<float>, 64>` matches `texture(0..63)`. Same
     * slot count as RT. */
    id<MTLTexture> tex_array[64];
    NSUInteger nbound = (NSUInteger)gpu->texture_count;
    if (nbound > 64) nbound = 64;
    for (NSUInteger i = 0; i < nbound; i++) tex_array[i] = gpu->material_textures[i];
    for (NSUInteger i = nbound; i < 64; i++) tex_array[i] = gpu->material_dummy_tex;
    [gpu->current_render_enc setFragmentTextures:tex_array
                                       withRange:NSMakeRange(0, 64)];
    [gpu->current_render_enc setFragmentSamplerState:gpu->material_sampler atIndex:0];

    /* IBL textures — slot 64 (env, GGX-prefiltered), 65 (SH irradiance),
     * 66 (BRDF integration LUT). Fall back to the material-dummy texture
     * when the dome HDR hasn't been loaded; the shader's has_ibl gate
     * skips sampling those slots. The BRDF LUT is what unlocks parity
     * with RT's split-sum specular (raytrace.metal:sample_specular_ibl)
     * + Kulla-Conty multi-scatter — without it the raster path under-
     * shoots dielectric specular and the diffuse-IBL warmth dominates
     * (visible as the brown chess-board floor on usd-wg/assets MTLX). */
    id<MTLTexture> env_t  = gpu->env_texture ? gpu->env_texture : gpu->material_dummy_tex;
    id<MTLTexture> irr_t  = gpu->irr_texture ? gpu->irr_texture : gpu->material_dummy_tex;
    id<MTLTexture> brdf_t = gpu->brdf_lut    ? gpu->brdf_lut    : gpu->material_dummy_tex;
    [gpu->current_render_enc setFragmentTexture:env_t  atIndex:64];
    [gpu->current_render_enc setFragmentTexture:irr_t  atIndex:65];
    [gpu->current_render_enc setFragmentTexture:brdf_t atIndex:66];
}

void gpu_cmd_bind_shadow(Gpu* gpu)
{
    if (!gpu || !gpu->in_frame || !gpu->current_render_enc || !gpu->tlas) return;
    [gpu->current_render_enc setFragmentAccelerationStructure:gpu->tlas atBufferIndex:4];
    for (id<MTLAccelerationStructure> blas in gpu->blas_list) {
        [gpu->current_render_enc useResource:blas
                                       usage:MTLResourceUsageRead
                                      stages:MTLRenderStageFragment];
    }
    if (gpu->curve_blas) {
        [gpu->current_render_enc useResource:gpu->curve_blas
                                       usage:MTLResourceUsageRead
                                      stages:MTLRenderStageFragment];
    }
}

void gpu_cmd_bind_vertex_buffer(Gpu* gpu, GpuBuffer buf)
{
    if (!gpu || !gpu->in_frame || !buf) return;
    [gpu->current_render_enc setVertexBuffer:buf->buffer offset:0 atIndex:0];
    gpu->current_vertex_buffer = buf;
}

void gpu_cmd_bind_instance_buffer(Gpu* gpu, GpuBuffer buf)
{
    if (!gpu || !gpu->in_frame || !buf) return;
    [gpu->current_render_enc setVertexBuffer:buf->buffer offset:0 atIndex:2];
}

void gpu_cmd_bind_index_buffer(Gpu* gpu, GpuBuffer buf)
{
    if (!gpu || !gpu->in_frame || !buf) return;
    /* Metal binds the index buffer at draw time, not via the encoder. Stash it. */
    gpu->current_index_buffer = buf;
}

void gpu_cmd_push_constants(Gpu* gpu, const void* data, uint32_t size)
{
    if (!gpu || !gpu->in_frame || !data || size == 0) return;
    /* Metal limit is 4 KB for setBytes; renderer.c push constants are ≤224 B. */
    [gpu->current_render_enc setVertexBytes:data length:size atIndex:1];
    [gpu->current_render_enc setFragmentBytes:data length:size atIndex:1];
}

void gpu_cmd_draw(Gpu* gpu, uint32_t vertex_count, uint32_t first_vertex)
{
    if (!gpu || !gpu->in_frame) return;
    [gpu->current_render_enc drawPrimitives:MTLPrimitiveTypeTriangle
                                vertexStart:first_vertex
                                vertexCount:vertex_count];
}

void gpu_cmd_draw_indexed(Gpu* gpu, uint32_t index_count,
                          uint32_t first_index, int32_t vertex_offset)
{
    gpu_cmd_draw_indexed_instanced(gpu, index_count, first_index,
                                   vertex_offset, 1u, 0u);
}

void gpu_cmd_draw_indexed_instanced(Gpu* gpu, uint32_t index_count,
                                    uint32_t first_index, int32_t vertex_offset,
                                    uint32_t instance_count, uint32_t first_instance)
{
    if (!gpu || !gpu->in_frame || !gpu->current_index_buffer ||
        index_count == 0 || instance_count == 0) return;
    /* renderer.c passes vertex_offset = mesh.vertex_offset as a vertex index;
     * Metal applies this via baseVertex. first_index is also a vertex-count
     * index (4 bytes per uint32 element). */
    [gpu->current_render_enc
        drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                   indexCount:index_count
                    indexType:MTLIndexTypeUInt32
                  indexBuffer:gpu->current_index_buffer->buffer
            indexBufferOffset:(NSUInteger)first_index * sizeof(uint32_t)
                instanceCount:instance_count
                   baseVertex:vertex_offset
                 baseInstance:first_instance];
}

static float env_float_clamped(const char* name, float fallback,
                               float min_value, float max_value)
{
    const char* e = getenv(name);
    if (!e || !e[0]) return fallback;
    char* end = NULL;
    double v = strtod(e, &end);
    if (end == e) return fallback;
    if (v < min_value) v = min_value;
    if (v > max_value) v = max_value;
    return (float)v;
}

static int ensure_curve_raster_pipeline(Gpu* gpu)
{
    if (!gpu) return 0;
    if (gpu->curve_raster_pipeline && gpu->curve_raster_depth) return 1;
    if (!gpu->shader_library) return 0;

    id<MTLFunction> vfn = [gpu->shader_library newFunctionWithName:@"curve_raster_vs"];
    id<MTLFunction> ffn = [gpu->shader_library newFunctionWithName:@"curve_raster_fs"];
    if (!vfn || !ffn) {
        fprintf(stderr,
                "gpu_metal: curve raster pipeline missing function "
                "(curve_raster_vs=%s, curve_raster_fs=%s)\n",
                vfn ? "ok" : "MISSING", ffn ? "ok" : "MISSING");
        return 0;
    }

    MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction   = vfn;
    pd.fragmentFunction = ffn;
    pd.colorAttachments[0].pixelFormat = gpu->color_format;
    pd.depthAttachmentPixelFormat      = gpu->depth_format;

    NSError* err = nil;
    id<MTLRenderPipelineState> pso =
        [gpu->device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!pso) {
        fprintf(stderr, "gpu_metal: curve raster pipeline build failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "(unknown)");
        return 0;
    }

    MTLDepthStencilDescriptor* dsd = [MTLDepthStencilDescriptor new];
    dsd.depthCompareFunction = MTLCompareFunctionLess;
    dsd.depthWriteEnabled    = YES;

    gpu->curve_raster_pipeline = pso;
    gpu->curve_raster_depth =
        [gpu->device newDepthStencilStateWithDescriptor:dsd];
    return gpu->curve_raster_depth != nil;
}

int gpu_cmd_draw_curves(Gpu* gpu, const float* vp, const float* eye)
{
    if (!gpu || !gpu->in_frame || !gpu->current_render_enc) return 0;
    if (gpu->curve_seg_count == 0 || !gpu->curve_seg_buf || !gpu->curve_color_buf)
        return 1;
    if (!vp || !eye) return 0;
    if (!ensure_curve_raster_pipeline(gpu)) return 0;

    struct CurveRasterUniformCPU {
        float vp[16];
        float eye_pos[4];
        float viewport_size[2];
        float min_pixel_radius;
        float max_pixel_radius;
        float dome_color[4];
    } u;
    memcpy(u.vp, vp, sizeof(u.vp));
    u.eye_pos[0] = eye[0];
    u.eye_pos[1] = eye[1];
    u.eye_pos[2] = eye[2];
    u.eye_pos[3] = 0.0f;
    u.viewport_size[0] = (float)gpu->width;
    u.viewport_size[1] = (float)gpu->height;
    u.min_pixel_radius =
        env_float_clamped("NUSD_RASTER_CURVE_MIN_PX", 1.25f, 0.0f, 16.0f);
    u.max_pixel_radius =
        env_float_clamped("NUSD_RASTER_CURVE_MAX_PX", 8.0f, 0.5f, 64.0f);
    u.dome_color[0] = gpu->dome_color[0];
    u.dome_color[1] = gpu->dome_color[1];
    u.dome_color[2] = gpu->dome_color[2];
    u.dome_color[3] = gpu->dome_intensity > 0.0f ? gpu->dome_intensity : 0.0f;

    [gpu->current_render_enc setRenderPipelineState:gpu->curve_raster_pipeline];
    [gpu->current_render_enc setDepthStencilState:gpu->curve_raster_depth];
    [gpu->current_render_enc setVertexBuffer:gpu->curve_seg_buf
                                      offset:0
                                     atIndex:0];
    [gpu->current_render_enc setVertexBuffer:gpu->curve_color_buf
                                      offset:0
                                     atIndex:1];
    [gpu->current_render_enc setVertexBytes:&u length:sizeof(u) atIndex:2];
    [gpu->current_render_enc setFragmentBytes:&u length:sizeof(u) atIndex:2];
    [gpu->current_render_enc drawPrimitives:MTLPrimitiveTypeTriangle
                                vertexStart:0
                                vertexCount:6
                              instanceCount:(NSUInteger)gpu->curve_seg_count];
    return 1;
}

/* ---- Materials ---- */

/* Maximum textures bound to the RT kernel as a single texture-array
 * slot range. Picked so the array fits in one Metal argument-buffer
 * binding range without needing actual argument buffers. Material
 * tex_indices[] entries are interpreted as indices into this array;
 * values >= MATERIAL_TEXTURE_SLOT_COUNT or < 0 mean "no texture in
 * this slot — use the material parameter directly". */
static const NSUInteger MATERIAL_TEXTURE_SLOT_BASE  = 4;
static const NSUInteger MATERIAL_TEXTURE_SLOT_COUNT = 192; /* Isaac full warehouse needs 135 unique textures. */

/* 1x1 white texture used to fill unbound texture-array slots. */
static id<MTLTexture> ensure_material_dummy_tex(Gpu* gpu)
{
    if (gpu->material_dummy_tex) return gpu->material_dummy_tex;
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:1 height:1 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    gpu->material_dummy_tex = [gpu->device newTextureWithDescriptor:td];
    uint8_t white[4] = { 255, 255, 255, 255 };
    [gpu->material_dummy_tex replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                               mipmapLevel:0
                                 withBytes:white
                               bytesPerRow:4];
    return gpu->material_dummy_tex;
}

static id<MTLSamplerState> ensure_material_sampler(Gpu* gpu)
{
    if (gpu->material_sampler) return gpu->material_sampler;
    MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.mipFilter = MTLSamplerMipFilterLinear;
    sd.sAddressMode = MTLSamplerAddressModeRepeat;
    sd.tAddressMode = MTLSamplerAddressModeRepeat;
    sd.maxAnisotropy = 4;
    gpu->material_sampler = [gpu->device newSamplerStateWithDescriptor:sd];
    return gpu->material_sampler;
}

int gpu_upload_materials(Gpu* gpu,
                         const GpuMaterialParams* materials, int nmaterials,
                         const GpuTextureData* textures, int ntextures)
{
    if (!gpu) return 0;

    /* Free previous upload (ARC drops textures on reassign). */
    gpu->material_buf      = nil;
    gpu->material_textures = nil;
    gpu->material_arg_buffer = nil;
    gpu->material_count    = 0;
    gpu->texture_count     = 0;
    gpu->has_materials     = 0;
    gpu->mat_only_placeholder = 0;

    /* Synthesize a placeholder material when callers pass 0 real materials
     * so the descriptor wiring stays valid (e.g. gpu_load_environment uses
     * the same descriptor pool). The shader gates real material reads
     * separately on scene.has_materials (set to 0 below for placeholder),
     * so this slot only matters for the raster path's per-mesh
     * displayColor fallback (use_vertex_color=1). */
    GpuMaterialParams placeholder_mat;
    memset(&placeholder_mat, 0, sizeof(placeholder_mat));
    int only_placeholder = 0;
    if (nmaterials <= 0 || !materials) {
        placeholder_mat.base_color[0] = 1.0f;
        placeholder_mat.base_color[1] = 1.0f;
        placeholder_mat.base_color[2] = 1.0f;
        placeholder_mat.base_color[3] = 1.0f;
        placeholder_mat.opacity       = 1.0f;
        placeholder_mat.ior           = 1.5f;
        placeholder_mat.roughness     = 0.5f;
        placeholder_mat.occlusion     = 1.0f;
        placeholder_mat.normal_tex_scale[0] = 2.0f;
        placeholder_mat.normal_tex_scale[1] = 2.0f;
        placeholder_mat.normal_tex_scale[2] = 2.0f;
        placeholder_mat.normal_tex_scale[3] = 1.0f;
        placeholder_mat.normal_tex_bias[0]  = -1.0f;
        placeholder_mat.normal_tex_bias[1]  = -1.0f;
        placeholder_mat.normal_tex_bias[2]  = -1.0f;
        placeholder_mat.normal_tex_bias[3]  =  0.0f;
        placeholder_mat.uv_transform[0] = 1.0f;
        placeholder_mat.uv_transform[1] = 1.0f;
        placeholder_mat.uv_transform[2] = 0.0f;
        placeholder_mat.uv_transform[3] = 0.0f;
        placeholder_mat.roughness_tex_transform[0] = 1.0f;
        placeholder_mat.roughness_tex_transform[1] = 0.0f;
        placeholder_mat.roughness_tex_transform[2] = 0.0f;
        placeholder_mat.roughness_tex_transform[3] = 0.0f;
        placeholder_mat.v_flip              = 0;
        placeholder_mat.base_weight          = 1.0f;
        placeholder_mat.specular_weight      = 1.0f;
        placeholder_mat.sheen_color[0]       = 1.0f;
        placeholder_mat.sheen_color[1]       = 1.0f;
        placeholder_mat.sheen_color[2]       = 1.0f;
        placeholder_mat.sheen_color[3]       = 1.0f;
        placeholder_mat.sheen_roughness      = 0.3f;
        placeholder_mat.thin_film_ior        = 1.5f;
        placeholder_mat.use_vertex_color = 1;
        for (int t = 0; t < 8; t++) placeholder_mat.tex_indices[t] = -1;
        placeholder_mat.tex_subsurface_weight = -1;
        placeholder_mat.tex_transmission_weight = -1;
        materials = &placeholder_mat;
        nmaterials = 1;
        ntextures = 0;
        only_placeholder = 1;
    }
    gpu->mat_only_placeholder = only_placeholder;

    /* Materials SSBO. GpuMaterialParams matches the C struct in gpu.h;
     * the MSL mirrors use the same 16-byte-aligned stride. */
    NSUInteger mat_bytes = (NSUInteger)nmaterials * sizeof(GpuMaterialParams);
    gpu->material_buf = [gpu->device newBufferWithBytes:materials
                                                 length:mat_bytes
                                                options:MTLResourceStorageModeShared];
    if (!gpu->material_buf) {
        fprintf(stderr, "gpu_metal: material SSBO alloc failed (%d materials)\n", nmaterials);
        return 0;
    }
    gpu->material_count = (uint32_t)nmaterials;

    /* Texture upload. RGBA8 only — UDIM atlases and HDR textures aren't
     * yet handled. is_srgb selects the appropriate pixel format so
     * Metal hardware does the sRGB→linear gamma decode on sample. */
    if (ntextures > 0 && textures) {
        if (ntextures > (int)MATERIAL_TEXTURE_SLOT_COUNT) {
            fprintf(stderr, "gpu_metal: WARNING — capping textures from %d to %lu (raise MATERIAL_TEXTURE_SLOT_COUNT to lift)\n",
                    ntextures, (unsigned long)MATERIAL_TEXTURE_SLOT_COUNT);
            ntextures = (int)MATERIAL_TEXTURE_SLOT_COUNT;
        }
        gpu->material_textures = [[NSMutableArray alloc] initWithCapacity:ntextures];

        NSUInteger total_bytes = 0;
        /* Pooled staging buffer (port of vulkan 88aaf32): reuse one Shared
         * MTLBuffer for every texture's CPU->GPU blit instead of allocating a
         * fresh newBufferWithBytes per texture. Each texture's waitUntilCompleted
         * finishes before the next reuses the buffer, so reuse is safe. Grown in
         * 16 MB chunks to the largest texture; released by ARC at scope end.
         * NUSD_STAGING_POOL=0 forces the per-texture path for A/B measurement.
         * On Apple-Silicon unified memory newBufferWithBytes is a near-fused
         * alloc+copy, so the win is smaller here than Vulkan's. */
        bool use_stage_pool = true;
        if (const char* e = getenv("NUSD_STAGING_POOL"))
            use_stage_pool = !(e[0] == '0');
        id<MTLBuffer> stage_pool = nil;
        NSUInteger    stage_cap  = 0;
        struct timespec _ut0; clock_gettime(CLOCK_MONOTONIC, &_ut0);
        for (int i = 0; i < ntextures; i++) {
            const GpuTextureData* td = &textures[i];
            if (!td->pixels || td->width <= 0 || td->height <= 0) {
                /* Skip — push the dummy in its place so indices line up. */
                [gpu->material_textures addObject:ensure_material_dummy_tex(gpu)];
                continue;
            }

            MTLPixelFormat fmt = td->is_srgb
                ? MTLPixelFormatRGBA8Unorm_sRGB
                : MTLPixelFormatRGBA8Unorm;

            /* Mipmap level count: 1 + floor(log2(max(w, h))) so we can
             * use trilinear filtering at distance for pipe + cable runs. */
            NSUInteger mips = 1;
            for (int m = (td->width > td->height ? td->width : td->height); m > 1; m >>= 1) mips++;

            MTLTextureDescriptor* desc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:fmt
                                             width:(NSUInteger)td->width
                                            height:(NSUInteger)td->height
                                         mipmapped:(mips > 1 ? YES : NO)];
            desc.mipmapLevelCount = mips;
            desc.usage = MTLTextureUsageShaderRead | (mips > 1 ? MTLTextureUsageRenderTarget : 0);
            desc.storageMode = MTLStorageModePrivate;

            id<MTLTexture> tex = [gpu->device newTextureWithDescriptor:desc];
            if (!tex) {
                fprintf(stderr, "gpu_metal: texture alloc failed (idx=%d, %dx%d)\n",
                        i, td->width, td->height);
                [gpu->material_textures addObject:ensure_material_dummy_tex(gpu)];
                continue;
            }

            /* Stage RGBA8 into a Shared MTLBuffer + blit-copy into the
             * Private MTLTexture's mip 0. Generate the mipchain via
             * blit::generateMipmaps. */
            NSUInteger row_bytes = (NSUInteger)td->width * 4;
            NSUInteger size = row_bytes * (NSUInteger)td->height;
            id<MTLBuffer> stage;
            if (use_stage_pool) {
                if (!stage_pool || size > stage_cap) {
                    NSUInteger chunk = (NSUInteger)16 << 20;
                    NSUInteger want = ((size + chunk - 1) / chunk) * chunk;
                    stage_pool = [gpu->device newBufferWithLength:want
                                                         options:MTLResourceStorageModeShared];
                    stage_cap = stage_pool ? want : 0;
                }
                if (stage_pool) memcpy(stage_pool.contents, td->pixels, (size_t)size);
                stage = stage_pool;
            } else {
                stage = [gpu->device newBufferWithBytes:td->pixels
                                                 length:size
                                                options:MTLResourceStorageModeShared];
            }
            id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromBuffer:stage
                    sourceOffset:0
               sourceBytesPerRow:row_bytes
             sourceBytesPerImage:size
                      sourceSize:MTLSizeMake((NSUInteger)td->width, (NSUInteger)td->height, 1)
                       toTexture:tex
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
            if (mips > 1) {
                [blit generateMipmapsForTexture:tex];
            }
            [blit endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            [gpu->material_textures addObject:tex];
            total_bytes += size;
        }

        stage_pool = nil;  /* release the reusable staging buffer (ARC) */
        struct timespec _ut1; clock_gettime(CLOCK_MONOTONIC, &_ut1);
        double _upload_ms = (_ut1.tv_sec - _ut0.tv_sec) * 1000.0 +
                            (_ut1.tv_nsec - _ut0.tv_nsec) / 1.0e6;
        gpu->texture_count = (uint32_t)ntextures;
        fprintf(stderr, "gpu_metal: uploaded %d materials + %d textures (%.1f MB) "
                "in %.1f ms [staging_pool=%d]\n",
                nmaterials, ntextures, (double)total_bytes / (1024.0 * 1024.0),
                _upload_ms, use_stage_pool ? 1 : 0);
    } else {
        fprintf(stderr, "gpu_metal: uploaded %d materials (no textures)\n", nmaterials);
    }

    /* Make sure the dummy + sampler exist so the kernel always has
     * a valid binding to fill empty array slots. */
    ensure_material_dummy_tex(gpu);
    ensure_material_sampler(gpu);

    gpu->has_materials = 1;

    /* Patch SceneHeader.has_materials in place if a scene is already
     * built (Shared storage = visible on next dispatch). The kernel's
     * material branch is gated on this flag. Placeholder-only counts as
     * "no materials" so per-mesh displayColor still drives shading. */
    if (gpu->scene_data_buf) {
        uint32_t flag = (gpu->mat_only_placeholder ? 0u : 1u) | view_transform_bit();
        memcpy((uint8_t*)[gpu->scene_data_buf contents] + 4, &flag, 4);
    }
    return 1;
}

static id<MTLBuffer> ensure_light_buffer(Gpu* gpu)
{
    if (!gpu) return nil;
    if (!gpu->light_buf) {
        gpu_upload_lights(gpu, NULL, 0);
    }
    return gpu->light_buf;
}

int gpu_upload_lights(Gpu* gpu, const GpuLight* lights, int nlights)
{
    if (!gpu) return 0;
    if (nlights < 0) nlights = 0;
    int n = nlights;
    if (n > GPU_MAX_SCENE_LIGHTS) {
        fprintf(stderr,
                "gpu_metal: truncating %d scene lights to %d\n",
                nlights, GPU_MAX_SCENE_LIGHTS);
        n = GPU_MAX_SCENE_LIGHTS;
    }

    const NSUInteger header_bytes = 16;
    int payload_count = (n > 0) ? n : 1;
    NSUInteger total_bytes = header_bytes + (NSUInteger)payload_count * sizeof(GpuLight);
    id<MTLBuffer> buf = [gpu->device newBufferWithLength:total_bytes
                                                  options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "gpu_metal: light buffer alloc failed (%d lights)\n", n);
        return 0;
    }

    uint8_t* p = (uint8_t*)[buf contents];
    memset(p, 0, total_bytes);
    memcpy(p, &n, sizeof(int));
    if (lights && n > 0) {
        memcpy(p + header_bytes, lights, (size_t)n * sizeof(GpuLight));
    }

    gpu->light_buf = buf;
    gpu->light_count = n;
    return 1;
}

static int ensure_material_argument_buffer(Gpu* gpu, NSString* kernel_name)
{
    if (!gpu || !gpu->rt_library) return 0;
    ensure_material_dummy_tex(gpu);

    if (!gpu->material_arg_encoder) {
        id<MTLFunction> fn = [gpu->rt_library newFunctionWithName:kernel_name];
        if (!fn) {
            fprintf(stderr, "gpu_metal: %s not found for material argument encoder\n",
                    [kernel_name UTF8String]);
            return 0;
        }
        gpu->material_arg_encoder = [fn newArgumentEncoderWithBufferIndex:18];
        if (!gpu->material_arg_encoder) {
            fprintf(stderr, "gpu_metal: material argument encoder unavailable for %s\n",
                    [kernel_name UTF8String]);
            return 0;
        }
    }

    if (!gpu->material_arg_buffer) {
        gpu->material_arg_buffer =
            [gpu->device newBufferWithLength:gpu->material_arg_encoder.encodedLength
                                      options:MTLResourceStorageModeShared];
        if (!gpu->material_arg_buffer) {
            fprintf(stderr, "gpu_metal: material argument buffer alloc failed (%lu bytes)\n",
                    (unsigned long)gpu->material_arg_encoder.encodedLength);
            return 0;
        }

        [gpu->material_arg_encoder setArgumentBuffer:gpu->material_arg_buffer
                                              offset:0];
        NSUInteger nbound = (NSUInteger)gpu->texture_count;
        if (nbound > MATERIAL_TEXTURE_SLOT_COUNT) nbound = MATERIAL_TEXTURE_SLOT_COUNT;
        for (NSUInteger i = 0; i < MATERIAL_TEXTURE_SLOT_COUNT; i++) {
            id<MTLTexture> tex = (i < nbound && gpu->material_textures)
                ? gpu->material_textures[i]
                : gpu->material_dummy_tex;
            [gpu->material_arg_encoder setTexture:tex atIndex:i];
        }
    }
    return 1;
}

static void bind_material_argument_buffer(Gpu* gpu, NSString* kernel_name)
{
    if (!gpu || !gpu->current_compute_enc) return;
    ensure_material_dummy_tex(gpu);
    ensure_material_sampler(gpu);
    if (!ensure_material_argument_buffer(gpu, kernel_name)) return;

    [gpu->current_compute_enc setBuffer:gpu->material_arg_buffer offset:0 atIndex:18];
    [gpu->current_compute_enc setSamplerState:gpu->material_sampler atIndex:0];

    NSUInteger nbound = (NSUInteger)gpu->texture_count;
    if (nbound > MATERIAL_TEXTURE_SLOT_COUNT) nbound = MATERIAL_TEXTURE_SLOT_COUNT;
    for (NSUInteger i = 0; i < nbound; i++) {
        [gpu->current_compute_enc useResource:gpu->material_textures[i]
                                        usage:MTLResourceUsageRead];
    }
}

/* Phase 11.A — curve uploads. Three Shared MTLBuffers, one per input array.
 * Idempotent: passing seg_count == 0 frees previous uploads and returns
 * success. The AABB layout matches MTLAxisAlignedBoundingBox (24 B: min
 * float3, max float3), so we can hand the same buffer directly to the
 * bounding-box BLAS descriptor below. */
int gpu_upload_curve_data(Gpu* gpu,
                          const void* segments,
                          const void* aabbs,
                          const float* colors,
                          int seg_count)
{
    if (!gpu) return 0;

    /* Free previous upload — ARC releases the old MTLBuffers when we
     * reassign. The associated BLAS is invalidated; clear it too so the
     * next gpu_build_curve_blas rebuilds from fresh data. */
    gpu->curve_seg_buf   = nil;
    gpu->curve_color_buf = nil;
    gpu->curve_aabb_buf  = nil;
    gpu->curve_blas      = nil;
    gpu->curve_seg_count = 0;

    if (seg_count <= 0) return 1;
    if (!segments || !aabbs || !colors) return 0;

    /* Per-segment data (32 B each: vec3 p0, float r0, vec3 p1, float r1). */
    NSUInteger seg_bytes = (NSUInteger)seg_count * 32;
    gpu->curve_seg_buf = [gpu->device newBufferWithBytes:segments
                                                  length:seg_bytes
                                                 options:MTLResourceStorageModeShared];

    /* Per-segment color, padded to vec4 for std430-style layout
     * (parity with the Vulkan port's curve_color SSBO). */
    NSUInteger color_bytes = (NSUInteger)seg_count * 4 * sizeof(float);
    float* color_padded = (float*)malloc(color_bytes);
    if (!color_padded) return 0;
    for (int i = 0; i < seg_count; i++) {
        color_padded[i*4 + 0] = colors[i*3 + 0];
        color_padded[i*4 + 1] = colors[i*3 + 1];
        color_padded[i*4 + 2] = colors[i*3 + 2];
        color_padded[i*4 + 3] = 1.0f;
    }
    gpu->curve_color_buf = [gpu->device newBufferWithBytes:color_padded
                                                    length:color_bytes
                                                   options:MTLResourceStorageModeShared];
    free(color_padded);

    /* AABB buffer for the BLAS. SceneCurveAabb is `float min[3]; float max[3]`
     * = 24 bytes, identical to MTLAxisAlignedBoundingBox, so we copy bytes
     * verbatim and set boundingBoxStride = 24 in the BLAS descriptor. */
    NSUInteger aabb_bytes = (NSUInteger)seg_count * 24;
    gpu->curve_aabb_buf = [gpu->device newBufferWithBytes:aabbs
                                                   length:aabb_bytes
                                                  options:MTLResourceStorageModeShared];

    if (!gpu->curve_seg_buf || !gpu->curve_color_buf || !gpu->curve_aabb_buf) {
        fprintf(stderr, "gpu_metal: curve buffer alloc failed (%d segments)\n", seg_count);
        gpu->curve_seg_buf = gpu->curve_color_buf = gpu->curve_aabb_buf = nil;
        return 0;
    }

    gpu->curve_seg_count = (uint32_t)seg_count;
    fprintf(stderr, "gpu_metal: uploaded %d curve segments (%lu B segs + %lu B colors + %lu B AABBs)\n",
            seg_count, (unsigned long)seg_bytes,
            (unsigned long)color_bytes, (unsigned long)aabb_bytes);
    return 1;
}

/* Phase 11.A — build a single bounding-box-geometry BLAS over the AABBs
 * uploaded by gpu_upload_curve_data. Independent of gpu_build_rt_scene —
 * this is a free-standing acceleration structure that the TLAS-side
 * IFT plumbing will reference as one additional instance. */
/* Lazy-compile the curve_aabb_gen MSL kernel and cache it on the Gpu.
 * Returns 1 on success, 0 on shader compile / function lookup failure. */
static int ensure_curve_aabb_pipeline(Gpu* gpu)
{
    if (gpu->curve_aabb_pipeline) return 1;

    char path[1024];
    snprintf(path, sizeof(path), "%s/curve_aabb_gen.metal", SHADER_DIR);
    id<MTLLibrary> lib = compile_msl_file(gpu->device, path);
    if (!lib) {
        fprintf(stderr, "gpu_metal: curve_aabb_gen.metal compile failed\n");
        return 0;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"curve_aabb_gen"];
    if (!fn) {
        fprintf(stderr, "gpu_metal: curve_aabb_gen kernel function not found\n");
        return 0;
    }
    NSError* err = nil;
    gpu->curve_aabb_pipeline =
        [gpu->device newComputePipelineStateWithFunction:fn error:&err];
    if (!gpu->curve_aabb_pipeline) {
        fprintf(stderr, "gpu_metal: curve_aabb_gen pipeline build failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "(unknown)");
        return 0;
    }
    return 1;
}

int gpu_build_curve_blas(Gpu* gpu)
{
    if (!gpu || !gpu->rt_available) return 0;
    if (gpu->curve_seg_count == 0 || !gpu->curve_aabb_buf) return 1;  /* nothing to build */

    /* Phase 12.x — recompute AABBs on the GPU from the segment SSBO,
     * overwriting whatever the host uploaded at gpu_upload_curve_data
     * time. The values match either way (closed-form constant-radius
     * cylinder bounds), but running this on the GPU lets the host stop
     * computing them in scene.c when we're ready to drop that path. */
    if (ensure_curve_aabb_pipeline(gpu)) {
        id<MTLCommandBuffer>        cb_aabb  = [gpu->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc_aabb = [cb_aabb computeCommandEncoder];
        [enc_aabb setComputePipelineState:gpu->curve_aabb_pipeline];
        [enc_aabb setBuffer:gpu->curve_seg_buf  offset:0 atIndex:0];
        [enc_aabb setBuffer:gpu->curve_aabb_buf offset:0 atIndex:1];
        uint32_t n = gpu->curve_seg_count;
        [enc_aabb setBytes:&n length:sizeof(n) atIndex:2];
        NSUInteger threadgroup_size = 64;
        NSUInteger grid_x = ((NSUInteger)n + threadgroup_size - 1)
                          / threadgroup_size * threadgroup_size;
        [enc_aabb dispatchThreads:MTLSizeMake(grid_x, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
        [enc_aabb endEncoding];
        [cb_aabb commit];
        [cb_aabb waitUntilCompleted];
    }

    /* Free any previous BLAS build (ARC releases on reassign). */
    gpu->curve_blas = nil;

    MTLAccelerationStructureBoundingBoxGeometryDescriptor* gd =
        [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
    gd.boundingBoxBuffer       = gpu->curve_aabb_buf;
    gd.boundingBoxBufferOffset = 0;
    gd.boundingBoxCount        = (NSUInteger)gpu->curve_seg_count;
    gd.boundingBoxStride       = 24;  /* sizeof(MTLAxisAlignedBoundingBox) */
    /* Per-instance intersection function table offset is set on the TLAS
     * instance side, not here; per-geometry offset stays at 0. The Metal
     * intersector dispatches into the IFT via that instance offset. */
    gd.intersectionFunctionTableOffset = 0;
    gd.opaque = NO;  /* AABB geometry is never "opaque" — intersection
                     * function decides whether to accept hits. */

    MTLPrimitiveAccelerationStructureDescriptor* desc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    desc.geometryDescriptors = @[gd];

    MTLAccelerationStructureSizes sizes =
        [gpu->device accelerationStructureSizesWithDescriptor:desc];

    gpu->curve_blas =
        [gpu->device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    if (!gpu->curve_blas) {
        fprintf(stderr, "gpu_metal: curve BLAS alloc failed (%.1f KB)\n",
                (double)sizes.accelerationStructureSize / 1024.0);
        return 0;
    }

    id<MTLBuffer> scratch =
        [gpu->device newBufferWithLength:sizes.buildScratchBufferSize
                                 options:MTLResourceStorageModePrivate];
    if (!scratch) {
        fprintf(stderr, "gpu_metal: curve BLAS scratch alloc failed (%.1f KB)\n",
                (double)sizes.buildScratchBufferSize / 1024.0);
        gpu->curve_blas = nil;
        return 0;
    }

    {
        id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> enc =
            [cb accelerationStructureCommandEncoder];
        [enc buildAccelerationStructure:gpu->curve_blas
                             descriptor:desc
                          scratchBuffer:scratch
                    scratchBufferOffset:0];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    fprintf(stderr, "gpu_metal: curve BLAS built (%u AABBs, %.1f KB AS, %.1f KB scratch)\n",
            gpu->curve_seg_count,
            (double)sizes.accelerationStructureSize / 1024.0,
            (double)sizes.buildScratchBufferSize / 1024.0);
    return 1;
}

int gpu_curve_blas_built(Gpu* gpu)
{
    return gpu && gpu->curve_blas != nil;
}

/* ---- Gaussian splat RT -------------------------------------------------- */

static inline float gs_clampf(float v, float lo, float hi)
{
    return fminf(fmaxf(v, lo), hi);
}

static inline void gs_transform_point(const float m[16], const float p[3], float out[3])
{
    /* USD row-vector convention: translation is m[12..14]. */
    out[0] = p[0] * m[0] + p[1] * m[4] + p[2] * m[8]  + m[12];
    out[1] = p[0] * m[1] + p[1] * m[5] + p[2] * m[9]  + m[13];
    out[2] = p[0] * m[2] + p[1] * m[6] + p[2] * m[10] + m[14];
}

static inline void gs_transform_vector(const float m[16], const float v[3], float out[3])
{
    out[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8];
    out[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9];
    out[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10];
}

static inline float gs_normalize3(float v[3])
{
    float len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len < 1e-12f) {
        v[0] = 1.0f; v[1] = 0.0f; v[2] = 0.0f;
        return 1.0f;
    }
    float inv = 1.0f / len;
    v[0] *= inv; v[1] *= inv; v[2] *= inv;
    return len;
}

static inline void gs_quat_axes_wxyz(const float q_in[4],
                                     float axis0[3], float axis1[3], float axis2[3])
{
    float w = q_in[0], x = q_in[1], y = q_in[2], z = q_in[3];
    float len = sqrtf(w*w + x*x + y*y + z*z);
    if (len < 1e-12f) {
        w = 1.0f; x = y = z = 0.0f;
    } else {
        float inv = 1.0f / len;
        w *= inv; x *= inv; y *= inv; z *= inv;
    }

    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    axis0[0] = 1.0f - 2.0f * (yy + zz);
    axis0[1] = 2.0f * (xy - wz);
    axis0[2] = 2.0f * (xz + wy);

    axis1[0] = 2.0f * (xy + wz);
    axis1[1] = 1.0f - 2.0f * (xx + zz);
    axis1[2] = 2.0f * (yz - wx);

    axis2[0] = 2.0f * (xz - wy);
    axis2[1] = 2.0f * (yz + wx);
    axis2[2] = 1.0f - 2.0f * (xx + yy);
}

static inline void gs_write_identity_instance(MTLAccelerationStructureInstanceDescriptor* inst)
{
    memset(inst, 0, sizeof(*inst));
    MTLPackedFloat4x3* m = &inst->transformationMatrix;
    m->columns[0].x = 1; m->columns[0].y = 0; m->columns[0].z = 0;
    m->columns[1].x = 0; m->columns[1].y = 1; m->columns[1].z = 0;
    m->columns[2].x = 0; m->columns[2].y = 0; m->columns[2].z = 1;
    m->columns[3].x = 0; m->columns[3].y = 0; m->columns[3].z = 0;
    inst->mask = 0xFFu;
    inst->intersectionFunctionTableOffset = 0;
    inst->accelerationStructureIndex = 0;
}

static inline void gs_write_particle_instance(MTLAccelerationStructureInstanceDescriptor* inst,
                                              const float center[3],
                                              const float axis0[3], float extent0,
                                              const float axis1[3], float extent1,
                                              const float axis2[3], float extent2)
{
    memset(inst, 0, sizeof(*inst));
    MTLPackedFloat4x3* m = &inst->transformationMatrix;
    m->columns[0].x = axis0[0] * extent0;
    m->columns[0].y = axis0[1] * extent0;
    m->columns[0].z = axis0[2] * extent0;
    m->columns[1].x = axis1[0] * extent1;
    m->columns[1].y = axis1[1] * extent1;
    m->columns[1].z = axis1[2] * extent1;
    m->columns[2].x = axis2[0] * extent2;
    m->columns[2].y = axis2[1] * extent2;
    m->columns[2].z = axis2[2] * extent2;
    m->columns[3].x = center[0];
    m->columns[3].y = center[1];
    m->columns[3].z = center[2];
    inst->mask = 0xFFu;
    inst->intersectionFunctionTableOffset = 0;
    inst->accelerationStructureIndex = 0;
}

void gpu_gs_destroy(Gpu* gpu)
{
    if (!gpu) return;
    gpu->gs_particle_buf = nil;
    gpu->gs_sh_buf       = nil;
    gpu->gs_aabb_buf     = nil;
    gpu->gs_ico_vertex_buf = nil;
    gpu->gs_ico_index_buf  = nil;
    gpu->gs_instance_buf = nil;
    gpu->gs_tlas_scratch = nil;
    gpu->gs_blas         = nil;
    gpu->gs_tlas         = nil;
    gpu->gs_particle_count = 0;
    gpu->gs_sh_degree = 0;
    gpu->gs_depth_buf = nil;
    gpu->gs_normal_buf = nil;
    gpu->gs_output_w = gpu->gs_output_h = 0;
}

int gpu_gs_available(Gpu* gpu)
{
    return (gpu && gpu->rt_available) ? 1 : 0;
}

int gpu_gs_particle_count(Gpu* gpu)
{
    return gpu ? (int)gpu->gs_particle_count : 0;
}

int gpu_gs_upload_particles(Gpu* gpu,
                            const float* positions,
                            const float* scales,
                            const float* orientations,
                            const float* opacities,
                            const float* kernel_scales,
                            const float* sh_coefficients,
                            uint32_t particle_count,
                            uint32_t sh_degree,
                            const float prim_xform[16])
{
    if (!gpu || !gpu->rt_available) return 0;
    gpu_gs_destroy(gpu);
    if (particle_count == 0) return 1;
    if (!positions || !scales || !orientations || !opacities ||
        !kernel_scales || !sh_coefficients || sh_degree > 3) {
        return 0;
    }

    NSUInteger particle_bytes = (NSUInteger)particle_count * sizeof(GpuGsParticle);
    NSUInteger inst_bytes     = (NSUInteger)particle_count * sizeof(MTLAccelerationStructureInstanceDescriptor);
    GpuGsParticle* packed = (GpuGsParticle*)malloc(particle_bytes);
    MTLAccelerationStructureInstanceDescriptor* insts =
        (MTLAccelerationStructureInstanceDescriptor*)malloc(inst_bytes);
    if (!packed || !insts) {
        free(packed);
        free(insts);
        return 0;
    }

    float identity[16] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };
    const float* xf = prim_xform ? prim_xform : identity;

    float bounds_min[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float bounds_max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (uint32_t i = 0; i < particle_count; i++) {
        float center[3];
        gs_transform_point(xf, &positions[i * 3u], center);
        for (int c = 0; c < 3; c++) {
            if (center[c] < bounds_min[c]) bounds_min[c] = center[c];
            if (center[c] > bounds_max[c]) bounds_max[c] = center[c];
        }
    }
    float dx = bounds_max[0] - bounds_min[0];
    float dy = bounds_max[1] - bounds_min[1];
    float dz = bounds_max[2] - bounds_min[2];
    float scene_diag = sqrtf(dx * dx + dy * dy + dz * dz);
    float soft_max_sigma = scene_diag * GS_RT_SOFT_MAX_SIGMA_FRACTION;
    int use_soft_sigma_opacity = (particle_count >= GS_RT_SOFT_MAX_SIGMA_MIN_COUNT &&
                                  soft_max_sigma > 1e-6f);

    for (uint32_t i = 0; i < particle_count; i++) {
        const float* pos = &positions[i * 3u];
        const float* scl = &scales[i * 3u];
        const float* q   = &orientations[i * 4u];

        float center[3];
        gs_transform_point(xf, pos, center);

        float ax0_obj[3], ax1_obj[3], ax2_obj[3];
        gs_quat_axes_wxyz(q, ax0_obj, ax1_obj, ax2_obj);

        float ax0[3], ax1[3], ax2[3];
        gs_transform_vector(xf, ax0_obj, ax0);
        gs_transform_vector(xf, ax1_obj, ax1);
        gs_transform_vector(xf, ax2_obj, ax2);
        float xform_s0 = gs_normalize3(ax0);
        float xform_s1 = gs_normalize3(ax1);
        float xform_s2 = gs_normalize3(ax2);

        float sigma0 = fmaxf(fabsf(scl[0]) * xform_s0, 1e-6f);
        float sigma1 = fmaxf(fabsf(scl[1]) * xform_s1, 1e-6f);
        float sigma2 = fmaxf(fabsf(scl[2]) * xform_s2, 1e-6f);
        float kernel = fmaxf(kernel_scales[i], 0.1f);
        float opacity = gs_clampf(opacities[i], 0.0f, 0.99f);
        if (use_soft_sigma_opacity) {
            float max_sigma = fmaxf(sigma0, fmaxf(sigma1, sigma2));
            if (max_sigma > soft_max_sigma) {
                float f = soft_max_sigma / max_sigma;
                opacity *= f * f;
                kernel = fmaxf(gs_compute_kernel_scale(opacity, NU_GS_DEFAULT_KERNEL_DEGREE), 0.1f);
            }
        }

        GpuGsParticle* p = &packed[i];
        memset(p, 0, sizeof(*p));
        p->center[0] = center[0];
        p->center[1] = center[1];
        p->center[2] = center[2];
        p->opacity = gs_clampf(opacity, 0.0f, 0.99f);
        memcpy(p->axis0, ax0, sizeof(p->axis0));
        memcpy(p->axis1, ax1, sizeof(p->axis1));
        memcpy(p->axis2, ax2, sizeof(p->axis2));
        p->inv_s0 = 1.0f / sigma0;
        p->inv_s1 = 1.0f / sigma1;
        p->inv_s2 = 1.0f / sigma2;
        p->kernel2 = kernel * kernel;

        float e0 = sigma0 * kernel * GS_ICOSA_VRT_SCALE;
        float e1 = sigma1 * kernel * GS_ICOSA_VRT_SCALE;
        float e2 = sigma2 * kernel * GS_ICOSA_VRT_SCALE;
        gs_write_particle_instance(&insts[i], center, ax0, e0, ax1, e1, ax2, e2);
    }

    int sh_per = (int)((sh_degree + 1u) * (sh_degree + 1u));
    NSUInteger sh_bytes = (NSUInteger)particle_count * (NSUInteger)sh_per * 3u * sizeof(float);

    gpu->gs_particle_buf = [gpu->device newBufferWithBytes:packed
                                                    length:particle_bytes
                                                   options:MTLResourceStorageModeShared];
    gpu->gs_instance_buf = [gpu->device newBufferWithBytes:insts
                                                    length:inst_bytes
                                                   options:MTLResourceStorageModeShared];
    gpu->gs_sh_buf = [gpu->device newBufferWithBytes:sh_coefficients
                                              length:sh_bytes
                                             options:MTLResourceStorageModeShared];

    free(packed);
    free(insts);

    if (!gpu->gs_particle_buf || !gpu->gs_instance_buf || !gpu->gs_sh_buf) {
        fprintf(stderr, "gpu_metal: Gaussian buffer alloc failed (%u particles)\n", particle_count);
        gpu_gs_destroy(gpu);
        return 0;
    }

    gpu->gs_particle_count = particle_count;
    gpu->gs_sh_degree = sh_degree;
    fprintf(stderr, "gpu_metal: uploaded %u Gaussian splats (%.1f MB particles + %.1f MB SH)\n",
            particle_count,
            (double)particle_bytes / (1024.0 * 1024.0),
            (double)sh_bytes / (1024.0 * 1024.0));
    return 1;
}

int gpu_gs_build_accel(Gpu* gpu)
{
    if (!gpu || !gpu->rt_available || !gpu->gs_instance_buf || gpu->gs_particle_count == 0)
        return 0;

    gpu->gs_blas = nil;
    gpu->gs_tlas = nil;
    gpu->gs_tlas_scratch = nil;

    gpu->gs_ico_vertex_buf = [gpu->device newBufferWithBytes:GS_ICOSA_VERTS
                                                      length:sizeof(GS_ICOSA_VERTS)
                                                     options:MTLResourceStorageModeShared];
    gpu->gs_ico_index_buf = [gpu->device newBufferWithBytes:GS_ICOSA_INDS
                                                     length:sizeof(GS_ICOSA_INDS)
                                                    options:MTLResourceStorageModeShared];
    if (!gpu->gs_ico_vertex_buf || !gpu->gs_ico_index_buf) {
        fprintf(stderr, "gpu_metal: Gaussian icosahedron buffer alloc failed\n");
        gpu->gs_ico_vertex_buf = nil;
        gpu->gs_ico_index_buf = nil;
        return 0;
    }

    MTLAccelerationStructureTriangleGeometryDescriptor* gd =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    gd.vertexBuffer       = gpu->gs_ico_vertex_buf;
    gd.vertexBufferOffset = 0;
    gd.vertexStride       = 3 * sizeof(float);
    gd.vertexFormat       = MTLAttributeFormatFloat3;
    gd.indexBuffer        = gpu->gs_ico_index_buf;
    gd.indexBufferOffset  = 0;
    gd.indexType          = MTLIndexTypeUInt32;
    gd.triangleCount      = 20;
    gd.opaque             = YES;

    MTLPrimitiveAccelerationStructureDescriptor* pd =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    pd.geometryDescriptors = @[gd];

    MTLAccelerationStructureSizes bs =
        [gpu->device accelerationStructureSizesWithDescriptor:pd];
    gpu->gs_blas = [gpu->device newAccelerationStructureWithSize:bs.accelerationStructureSize];
    if (!gpu->gs_blas) {
        fprintf(stderr, "gpu_metal: Gaussian BLAS alloc failed\n");
        return 0;
    }
    id<MTLBuffer> blas_scratch =
        [gpu->device newBufferWithLength:bs.buildScratchBufferSize
                                 options:MTLResourceStorageModePrivate];
    if (!blas_scratch) {
        fprintf(stderr, "gpu_metal: Gaussian BLAS scratch alloc failed\n");
        gpu->gs_blas = nil;
        return 0;
    }

    {
        id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> enc =
            [cb accelerationStructureCommandEncoder];
        [enc buildAccelerationStructure:gpu->gs_blas
                             descriptor:pd
                          scratchBuffer:blas_scratch
                    scratchBufferOffset:0];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    MTLInstanceAccelerationStructureDescriptor* td =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    td.instanceDescriptorBuffer       = gpu->gs_instance_buf;
    td.instanceDescriptorBufferOffset = 0;
    td.instanceDescriptorStride       = sizeof(MTLAccelerationStructureInstanceDescriptor);
    td.instanceCount                  = (NSUInteger)gpu->gs_particle_count;
    td.instancedAccelerationStructures = @[gpu->gs_blas];

    MTLAccelerationStructureSizes ts =
        [gpu->device accelerationStructureSizesWithDescriptor:td];
    gpu->gs_tlas = [gpu->device newAccelerationStructureWithSize:ts.accelerationStructureSize];
    if (!gpu->gs_tlas) {
        fprintf(stderr, "gpu_metal: Gaussian TLAS alloc failed\n");
        gpu->gs_blas = nil;
        gpu->gs_instance_buf = nil;
        return 0;
    }
    gpu->gs_tlas_scratch =
        [gpu->device newBufferWithLength:ts.buildScratchBufferSize
                                 options:MTLResourceStorageModePrivate];
    if (!gpu->gs_tlas_scratch) {
        fprintf(stderr, "gpu_metal: Gaussian TLAS scratch alloc failed\n");
        gpu->gs_blas = nil;
        gpu->gs_tlas = nil;
        gpu->gs_instance_buf = nil;
        return 0;
    }

    {
        id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> enc =
            [cb accelerationStructureCommandEncoder];
        [enc buildAccelerationStructure:gpu->gs_tlas
                             descriptor:td
                          scratchBuffer:gpu->gs_tlas_scratch
                    scratchBufferOffset:0];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    fprintf(stderr, "gpu_metal: Gaussian BLAS/TLAS built (%u icosa instances, %.1f MB AS)\n",
            gpu->gs_particle_count,
            (double)(bs.accelerationStructureSize + ts.accelerationStructureSize) / (1024.0 * 1024.0));
    return 1;
}

static int ensure_gs_pipeline(Gpu* gpu)
{
    if (gpu->gs_pipeline && gpu->gs_ift) return 1;

    if (!gpu->gs_library) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/gaussian_rt.metal", SHADER_DIR);
        gpu->gs_library = compile_msl_file(gpu->device, path);
        if (!gpu->gs_library) {
            fprintf(stderr, "gpu_metal: gaussian_rt.metal compile failed\n");
            return 0;
        }
    }

    id<MTLFunction> kernel_fn = [gpu->gs_library newFunctionWithName:@"gs_rt_render"];
    if (!kernel_fn) {
        fprintf(stderr, "gpu_metal: gs_rt_render kernel function not found\n");
        return 0;
    }
    id<MTLFunction> isect_fn = [gpu->gs_library newFunctionWithName:@"gs_isect"];
    if (!isect_fn) {
        fprintf(stderr, "gpu_metal: gs_isect intersection function not found\n");
        return 0;
    }

    NSError* err = nil;
    if (!gpu->gs_pipeline) {
        MTLLinkedFunctions* lf = [MTLLinkedFunctions linkedFunctions];
        lf.functions = @[isect_fn];

        MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = kernel_fn;
        desc.linkedFunctions = lf;
        desc.supportIndirectCommandBuffers = NO;

        gpu->gs_pipeline = [gpu->device newComputePipelineStateWithDescriptor:desc
                                                                      options:MTLPipelineOptionNone
                                                                   reflection:nil
                                                                        error:&err];
        if (!gpu->gs_pipeline) {
            fprintf(stderr, "gpu_metal: Gaussian RT pipeline build failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
    }

    if (!gpu->gs_ift) {
        MTLIntersectionFunctionTableDescriptor* iftDesc =
            [MTLIntersectionFunctionTableDescriptor intersectionFunctionTableDescriptor];
        iftDesc.functionCount = 1;
        gpu->gs_ift = [gpu->gs_pipeline newIntersectionFunctionTableWithDescriptor:iftDesc];
        if (!gpu->gs_ift) {
            fprintf(stderr, "gpu_metal: Gaussian IFT allocation failed\n");
            return 0;
        }
        id<MTLFunctionHandle> handle = [gpu->gs_pipeline functionHandleWithFunction:isect_fn];
        if (!handle) {
            fprintf(stderr, "gpu_metal: Gaussian IFT function handle failed\n");
            gpu->gs_ift = nil;
            return 0;
        }
        [gpu->gs_ift setFunction:handle atIndex:0];
    }
    return 1;
}

static int ensure_gs_outputs(Gpu* gpu)
{
    if (!gpu) return 0;
    uint32_t w = (uint32_t)gpu->width;
    uint32_t h = (uint32_t)gpu->height;
    if (gpu->gs_depth_buf && gpu->gs_normal_buf &&
        gpu->gs_output_w == w && gpu->gs_output_h == h) {
        return 1;
    }

    NSUInteger pixels = (NSUInteger)w * (NSUInteger)h;
    gpu->gs_depth_buf = [gpu->device newBufferWithLength:pixels * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    gpu->gs_normal_buf = [gpu->device newBufferWithLength:pixels * 3u * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    if (!gpu->gs_depth_buf || !gpu->gs_normal_buf) {
        fprintf(stderr, "gpu_metal: Gaussian output buffer alloc failed\n");
        gpu->gs_depth_buf = nil;
        gpu->gs_normal_buf = nil;
        gpu->gs_output_w = gpu->gs_output_h = 0;
        return 0;
    }
    gpu->gs_output_w = w;
    gpu->gs_output_h = h;
    return 1;
}

int gpu_gs_render(Gpu* gpu,
                  const float vp_inv[32],
                  uint32_t sh_degree,
                  uint32_t k,
                  uint32_t max_passes,
                  float min_transmittance,
                  float iso_opacity_threshold,
                  uint32_t color_space)
{
    if (!gpu || !vp_inv || !gpu->rt_available || !gpu->gs_tlas ||
        !gpu->gs_particle_buf || !gpu->gs_sh_buf || !gpu->color_target) {
        return 0;
    }
    if (!ensure_gs_pipeline(gpu) || !ensure_gs_outputs(gpu)) return 0;

    GpuGsPushConstants pc;
    memset(&pc, 0, sizeof(pc));
    memcpy(pc.view_inv, vp_inv, 16 * sizeof(float));
    memcpy(pc.proj_inv, vp_inv + 16, 16 * sizeof(float));
    pc.particle_count = gpu->gs_particle_count;
    pc.sh_degree = sh_degree;
    pc.k = k;
    pc.max_passes = max_passes;
    pc.min_transmittance = min_transmittance;
    pc.iso_opacity_threshold = iso_opacity_threshold;
    pc.color_space = color_space;

    id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:gpu->gs_pipeline];
    [enc setTexture:gpu->color_target atIndex:0];
    [enc setAccelerationStructure:gpu->gs_tlas atBufferIndex:0];
    [enc setIntersectionFunctionTable:gpu->gs_ift atBufferIndex:6];
    [gpu->gs_ift setBuffer:gpu->gs_particle_buf offset:0 atIndex:6];

    alignas(16) uint8_t padded[160] = {0};
    static_assert(sizeof(GpuGsPushConstants) <= sizeof(padded),
                  "GpuGsPushConstants exceeded 160 B");
    memcpy(padded, &pc, sizeof(pc));
    [enc setBytes:padded length:sizeof(padded) atIndex:1];
    [enc setBuffer:gpu->gs_particle_buf offset:0 atIndex:2];
    [enc setBuffer:gpu->gs_sh_buf       offset:0 atIndex:3];
    [enc setBuffer:gpu->gs_depth_buf    offset:0 atIndex:4];
    [enc setBuffer:gpu->gs_normal_buf   offset:0 atIndex:5];
    [enc useResource:gpu->gs_blas usage:MTLResourceUsageRead];

    NSUInteger w = (NSUInteger)gpu->width;
    NSUInteger h = (NSUInteger)gpu->height;
    NSUInteger tew = gpu->gs_pipeline.threadExecutionWidth;
    NSUInteger maxT = gpu->gs_pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger ty = MAX((NSUInteger)1, maxT / tew);
    if (ty > h) ty = h;
    [enc dispatchThreads:MTLSizeMake(w, h, 1)
   threadsPerThreadgroup:MTLSizeMake(tew, ty, 1)];
    [enc endEncoding];

    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] == MTLCommandBufferStatusError) {
        NSError* err = [cb error];
        fprintf(stderr, "gpu_metal: Gaussian render command buffer failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "(unknown)");
        return 0;
    }
    gpu->has_frame = 1;
    return 1;
}

int gpu_gs_fetch_depth(Gpu* gpu, float* out_depth, uint32_t width, uint32_t height)
{
    if (!gpu || !out_depth || !gpu->gs_depth_buf ||
        width != gpu->gs_output_w || height != gpu->gs_output_h) {
        return 0;
    }
    memcpy(out_depth, [gpu->gs_depth_buf contents],
           (size_t)width * (size_t)height * sizeof(float));
    return 1;
}

int gpu_gs_fetch_normal(Gpu* gpu, float* out_normal, uint32_t width, uint32_t height)
{
    if (!gpu || !out_normal || !gpu->gs_normal_buf ||
        width != gpu->gs_output_w || height != gpu->gs_output_h) {
        return 0;
    }
    memcpy(out_normal, [gpu->gs_normal_buf contents],
           (size_t)width * (size_t)height * 3u * sizeof(float));
    return 1;
}

GpuPipeline gpu_create_material_pipeline(Gpu* gpu, const GpuPipelineDesc* desc)
{
    return gpu_create_pipeline(gpu, desc);
}

void gpu_cmd_bind_materials(Gpu* /*gpu*/) {}

void gpu_destroy_materials(Gpu* /*gpu*/) {}

/* ---- Environment / IBL ---- */

/* ---- Phase 7 IBL math helpers (verbatim ports of Vulkan side) ---- */

/* IEEE 754 half-precision conversion. RGBA16Float upload path stages
 * float32 source through this. Used for env / irradiance / BRDF LUT. */
static uint16_t float_to_half(float f)
{
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* Pack RGBA float32 → RGBA float16 in-place. dst stride = 8 bytes. */
static void rgba32f_to_rgba16f(const float* src, uint16_t* dst, size_t n_pixels)
{
    for (size_t i = 0; i < n_pixels; i++) {
        dst[i*4 + 0] = float_to_half(src[i*4 + 0]);
        dst[i*4 + 1] = float_to_half(src[i*4 + 1]);
        dst[i*4 + 2] = float_to_half(src[i*4 + 2]);
        dst[i*4 + 3] = float_to_half(src[i*4 + 3]);
    }
}

/* Project an equirectangular env map onto order-3 spherical harmonics.
 * 9 coefficients × 3 channels (RGB). Standard MaterialX-compatible
 * orthonormal basis. */
static void sh_project_environment(const float* rgb_data, int w, int h,
                                   float sh_coeffs[9][3])
{
    memset(sh_coeffs, 0, 9 * 3 * sizeof(float));
    const float PI = 3.14159265358979323846f;

    for (int y = 0; y < h; y++) {
        float theta = PI * ((float)y + 0.5f) / (float)h;
        float sin_theta = sinf(theta);
        float cos_theta = cosf(theta);
        float da = (2.0f * PI / (float)w) * (PI / (float)h) * sin_theta;

        for (int x = 0; x < w; x++) {
            float phi = 2.0f * PI * ((float)x + 0.5f) / (float)w;
            float dx = sin_theta * sinf(phi);
            float dy = cos_theta;
            float dz = sin_theta * cosf(phi);

            int idx = (y * w + x) * 3;
            float r = rgb_data[idx + 0] * da;
            float g = rgb_data[idx + 1] * da;
            float b = rgb_data[idx + 2] * da;

            float Y[9];
            Y[0] = 0.282095f;
            Y[1] = 0.488603f * dy;
            Y[2] = 0.488603f * dz;
            Y[3] = 0.488603f * dx;
            Y[4] = 1.092548f * dx * dy;
            Y[5] = 1.092548f * dy * dz;
            Y[6] = 0.315392f * (3.0f * dz * dz - 1.0f);
            Y[7] = 1.092548f * dx * dz;
            Y[8] = 0.546274f * (dx * dx - dy * dy);

            for (int c = 0; c < 9; c++) {
                sh_coeffs[c][0] += r * Y[c];
                sh_coeffs[c][1] += g * Y[c];
                sh_coeffs[c][2] += b * Y[c];
            }
        }
    }
}

/* Evaluate cosine-convolved SH irradiance at every direction in a
 * 256x128 lat-long grid. Output is 4-channel float (alpha=1). */
static void sh_render_irradiance(const float sh_coeffs[9][3],
                                 float* rgba_out, int w, int h)
{
    const float PI = 3.14159265358979323846f;
    const float c1 = 0.429043f, c2 = 0.511664f;
    const float c3 = 0.743125f, c4 = 0.886227f, c5 = 0.247708f;

    for (int y = 0; y < h; y++) {
        float theta = PI * ((float)y + 0.5f) / (float)h;
        float sin_theta = sinf(theta);
        float cos_theta = cosf(theta);

        for (int x = 0; x < w; x++) {
            float phi = 2.0f * PI * ((float)x + 0.5f) / (float)w;
            float nx = sin_theta * sinf(phi);
            float ny = cos_theta;
            float nz = sin_theta * cosf(phi);

            int idx = (y * w + x) * 4;
            for (int ch = 0; ch < 3; ch++) {
                float L[9];
                for (int i = 0; i < 9; i++) L[i] = sh_coeffs[i][ch];
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

/* Generate the BRDF integration LUT (split-sum approximation):
 * RG16Float, x = NdotV, y = roughness. Output: scale + bias for the
 * specular IBL split-sum. */
static void generate_brdf_lut(float* out, int size)
{
    for (int y = 0; y < size; y++) {
        float roughness = ((float)y + 0.5f) / (float)size;
        float a = roughness * roughness;
        float a2 = a * a;
        for (int x = 0; x < size; x++) {
            float NdotV = ((float)x + 0.5f) / (float)size;
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
                float H[3] = { cosf(phi)*sinTheta, sinf(phi)*sinTheta, cosTheta };
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
            bias  /= (float)N_SAMPLES;
            int idx = (y * size + x) * 2;
            out[idx + 0] = scale;
            out[idx + 1] = bias;
        }
    }
}

/* ---- GGX-prefiltered glossy env mip chain ----------------------------
 *
 * Replaces the box-filter mip chain that [blit generateMipmapsForTexture:]
 * produces. The split-sum specular IBL in raytrace.metal:200
 * (sample_specular_ibl) reads `env_tex` at LOD = roughness*(mips-1); a
 * box-filter chain leaves bright sun pixels hot at every mip, so high-
 * roughness samples dump the entire sun's radiance into the lobe and
 * produce white-hot blobs on chrome / copper / gold. A real GGX prefilter
 * (Karis 2014: Hammersley + importance-sample H, weight by NdotL × sin θ
 * for lat-long Jacobian) integrates the kernel correctly so each output
 * mip already represents the BRDF-weighted radiance at its target
 * roughness. Port from Vulkan e1cc791 + aded83e.
 * ---------------------------------------------------------------------- */
static void prefilter_hammersley(uint32_t i, uint32_t N, float* u, float* v)
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

/* Sample H from an isotropic GGX(α) distribution given basis N. */
static void prefilter_importance_sample_ggx(float u, float v, float alpha,
                                            const float N[3], float H_out[3])
{
    const float PI = 3.14159265358979323846f;
    float alpha2    = alpha * alpha;
    float phi       = 2.0f * PI * u;
    float cos_theta = sqrtf((1.0f - v) / (1.0f + (alpha2 - 1.0f) * v));
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

    float Hx = sin_theta * cosf(phi);
    float Hy = sin_theta * sinf(phi);
    float Hz = cos_theta;

    /* Pixar branchless ONB around N. */
    float sgn = N[2] >= 0.0f ? 1.0f : -1.0f;
    float a   = -1.0f / (sgn + N[2]);
    float b   = N[0] * N[1] * a;
    float Tx  = 1.0f + sgn * N[0] * N[0] * a;
    float Ty  = sgn * b;
    float Tz  = -sgn * N[0];
    float Bx  = b;
    float By  = sgn + N[1] * N[1] * a;
    float Bz  = -N[1];

    H_out[0] = Tx * Hx + Bx * Hy + N[0] * Hz;
    H_out[1] = Ty * Hx + By * Hy + N[1] * Hz;
    H_out[2] = Tz * Hx + Bz * Hy + N[2] * Hz;
}

/* Bilinear lookup in a float32 RGBA lat-long environment. Matches
 * dir_to_equirect_uv() in raytrace.metal: u = atan2(dx, dz)/2π + 0.5,
 * v = asin(dy)/π + 0.5. Wrap U, clamp V. */
static void prefilter_env_sample_bilinear_rgba(const float* env_rgba,
                                               int env_w, int env_h,
                                               const float dir[3],
                                               float out_rgb[3])
{
    const float PI = 3.14159265358979323846f;
    float u  = atan2f(dir[0], dir[2]) * (1.0f / (2.0f * PI)) + 0.5f;
    float dy = fmaxf(-1.0f, fminf(1.0f, dir[1]));
    float v  = asinf(dy) * (1.0f / PI) + 0.5f;
    float fx = u * (float)env_w - 0.5f;
    float fy = v * (float)env_h - 0.5f;
    if (fy < 0.0f) fy = 0.0f;
    if (fy > (float)(env_h - 1)) fy = (float)(env_h - 1);
    int x0   = (int)floorf(fx);
    int y0   = (int)floorf(fy);
    float wx = fx - (float)x0;
    float wy = fy - (float)y0;
    int x1   = x0 + 1;
    int y1   = y0 + 1;
    x0 = ((x0 % env_w) + env_w) % env_w;
    x1 = ((x1 % env_w) + env_w) % env_w;
    if (y1 > env_h - 1) y1 = env_h - 1;
    const float* p00 = &env_rgba[(y0 * env_w + x0) * 4];
    const float* p10 = &env_rgba[(y0 * env_w + x1) * 4];
    const float* p01 = &env_rgba[(y1 * env_w + x0) * 4];
    const float* p11 = &env_rgba[(y1 * env_w + x1) * 4];
    for (int c = 0; c < 3; c++) {
        float a = p00[c] * (1.0f - wx) + p10[c] * wx;
        float b = p01[c] * (1.0f - wx) + p11[c] * wx;
        out_rgb[c] = a * (1.0f - wy) + b * wy;
    }
}

/* Lat-long UV → unit direction. Matches the inverse of
 * dir_to_equirect_uv() in raytrace.metal. */
static void prefilter_latlong_uv_to_dir(float u, float v, float dir_out[3])
{
    const float PI = 3.14159265358979323846f;
    float phi   = (u - 0.5f) * 2.0f * PI;
    float theta = (v - 0.5f) * PI;
    dir_out[0] = sinf(phi) * cosf(theta);
    dir_out[1] = sinf(theta);
    dir_out[2] = cosf(phi) * cosf(theta);
}

/* GGX-prefilter a single output mip (mip_w × mip_h, RGBA32F) sampling
 * the source RGBA32F lat-long env. roughness ∈ [0,1]; mirror copy at
 * roughness=0 to skip the kernel for mip 0. Parallelized via
 * dispatch_apply across rows. */
static void prefilter_ggx_mip(const float* src_rgba, int env_w, int env_h,
                              float* mip_rgba, int mip_w, int mip_h,
                              float roughness, int sample_count)
{
    const float alpha = roughness * roughness;

    if (roughness < 1e-3f) {
        dispatch_apply((size_t)mip_h, dispatch_get_global_queue(0, 0), ^(size_t y) {
            float vy = ((float)y + 0.5f) / (float)mip_h;
            for (int x = 0; x < mip_w; x++) {
                float ux = ((float)x + 0.5f) / (float)mip_w;
                float dir[3], rgb[3];
                prefilter_latlong_uv_to_dir(ux, vy, dir);
                prefilter_env_sample_bilinear_rgba(src_rgba, env_w, env_h, dir, rgb);
                int o = ((int)y * mip_w + x) * 4;
                mip_rgba[o + 0] = rgb[0];
                mip_rgba[o + 1] = rgb[1];
                mip_rgba[o + 2] = rgb[2];
                mip_rgba[o + 3] = 1.0f;
            }
        });
        return;
    }

    dispatch_apply((size_t)mip_h, dispatch_get_global_queue(0, 0), ^(size_t y) {
        for (int x = 0; x < mip_w; x++) {
            float ux = ((float)x + 0.5f) / (float)mip_w;
            float vy = ((float)y + 0.5f) / (float)mip_h;
            float N[3];
            prefilter_latlong_uv_to_dir(ux, vy, N);
            const float* V = N;  /* Karis: V = N = R */

            float acc[3] = {0, 0, 0};
            float total_weight = 0.0f;
            for (int i = 0; i < sample_count; i++) {
                float xi_u, xi_v;
                prefilter_hammersley((uint32_t)i, (uint32_t)sample_count, &xi_u, &xi_v);
                float H[3];
                prefilter_importance_sample_ggx(xi_u, xi_v, alpha, N, H);
                float VdotH = V[0]*H[0] + V[1]*H[1] + V[2]*H[2];
                float L[3] = {
                    2.0f * VdotH * H[0] - V[0],
                    2.0f * VdotH * H[1] - V[1],
                    2.0f * VdotH * H[2] - V[2],
                };
                float NdotL = N[0]*L[0] + N[1]*L[1] + N[2]*L[2];
                if (NdotL > 0.0f) {
                    float Li[3];
                    prefilter_env_sample_bilinear_rgba(src_rgba, env_w, env_h, L, Li);
                    /* Lat-long Jacobian: sin(theta_L) = sqrt(1 − L.y²).
                     * Without it, polar pixels are over-counted because
                     * the lat-long parameterisation stretches near the
                     * poles. Port from Vulkan aded83e. */
                    float sin_theta_L = sqrtf(fmaxf(0.0f, 1.0f - L[1] * L[1]));
                    float w = NdotL * sin_theta_L;
                    acc[0] += Li[0] * w;
                    acc[1] += Li[1] * w;
                    acc[2] += Li[2] * w;
                    total_weight += w;
                }
            }
            int o = ((int)y * mip_w + x) * 4;
            if (total_weight > 0.0f) {
                mip_rgba[o + 0] = acc[0] / total_weight;
                mip_rgba[o + 1] = acc[1] / total_weight;
                mip_rgba[o + 2] = acc[2] / total_weight;
            } else {
                mip_rgba[o + 0] = mip_rgba[o + 1] = mip_rgba[o + 2] = 0.0f;
            }
            mip_rgba[o + 3] = 1.0f;
        }
    });
}

/* Drop-in replacement for create_hdr_texture_metal that GGX-prefilters
 * the mip chain at upload time. Returns nil on failure. */
static id<MTLTexture> create_hdr_texture_metal_ggx_prefiltered(
    Gpu* gpu, const float* src_rgba, int w, int h, int sample_count,
    int* out_mip_levels)
{
    NSUInteger mip_count = 1;
    {
        int m = w > h ? w : h;
        while (m > 1) { mip_count++; m >>= 1; }
    }

    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                     width:(NSUInteger)w
                                    height:(NSUInteger)h
                                 mipmapped:YES];
    td.mipmapLevelCount = mip_count;
    td.usage            = MTLTextureUsageShaderRead;
    td.storageMode      = MTLStorageModePrivate;
    id<MTLTexture> tex  = [gpu->device newTextureWithDescriptor:td];
    if (!tex) return nil;

    int mip_w[16] = {0}, mip_h[16] = {0};
    {
        int mw = w, mh = h;
        for (NSUInteger m = 0; m < mip_count; m++) {
            mip_w[m] = mw; mip_h[m] = mh;
            mw = mw > 1 ? mw / 2 : 1;
            mh = mh > 1 ? mh / 2 : 1;
        }
    }

    fprintf(stderr,
            "gpu_metal: GGX-prefilter env mip chain (%dx%d, %d mips, %d samples/texel)...\n",
            w, h, (int)mip_count, sample_count);
    double t0 = (double)clock() / (double)CLOCKS_PER_SEC;

    /* Per-mip prefilter into a temp RGBA32F buffer, half-pack into
     * an MTLBuffer staging area, blit into the texture. */
    float* tmp = NULL;
    size_t tmp_cap = 0;
    for (NSUInteger m = 0; m < mip_count; m++) {
        size_t pixels = (size_t)mip_w[m] * (size_t)mip_h[m];
        if (tmp_cap < pixels * 4) {
            free(tmp);
            tmp = (float*)malloc(pixels * 4 * sizeof(float));
            tmp_cap = pixels * 4;
        }
        if (!tmp) return nil;
        float roughness = (mip_count > 1) ? (float)m / (float)(mip_count - 1) : 0.0f;
        int adj = sample_count;
        if (mip_w[m] <= 16)      adj = sample_count / 4;
        else if (mip_w[m] <= 64) adj = sample_count / 2;
        prefilter_ggx_mip(src_rgba, w, h, tmp, mip_w[m], mip_h[m], roughness, adj);

        size_t row_bytes = (size_t)mip_w[m] * 8;
        size_t total     = row_bytes * (size_t)mip_h[m];
        id<MTLBuffer> staging = [gpu->device newBufferWithLength:total
                                                         options:MTLResourceStorageModeShared];
        if (!staging) { free(tmp); return nil; }
        rgba32f_to_rgba16f(tmp, (uint16_t*)[staging contents], pixels);

        id<MTLCommandBuffer> cb     = [gpu->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:staging
                sourceOffset:0
           sourceBytesPerRow:row_bytes
         sourceBytesPerImage:total
                  sourceSize:MTLSizeMake((NSUInteger)mip_w[m], (NSUInteger)mip_h[m], 1)
                   toTexture:tex
            destinationSlice:0
            destinationLevel:m
           destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    free(tmp);

    double t1 = (double)clock() / (double)CLOCKS_PER_SEC;
    fprintf(stderr, "gpu_metal: GGX prefilter done in %.2f s\n", t1 - t0);

    if (out_mip_levels) *out_mip_levels = (int)mip_count;
    return tex;
}

static float* resize_env_rgba_bilinear(const float* src, int sw, int sh, int dw, int dh)
{
    if (!src || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return NULL;
    float* dst = (float*)malloc((size_t)dw * (size_t)dh * 4u * sizeof(float));
    if (!dst) return NULL;
    const float sx_scale = (float)sw / (float)dw;
    const float sy_scale = (float)sh / (float)dh;
    for (int y = 0; y < dh; y++) {
        float sy = ((float)y + 0.5f) * sy_scale - 0.5f;
        if (sy < 0.0f) sy = 0.0f;
        if (sy > (float)(sh - 1)) sy = (float)(sh - 1);
        int y0 = (int)floorf(sy);
        int y1 = y0 + 1 < sh ? y0 + 1 : y0;
        float fy = sy - (float)y0;
        for (int x = 0; x < dw; x++) {
            float sx = ((float)x + 0.5f) * sx_scale - 0.5f;
            if (sx < 0.0f) sx = 0.0f;
            if (sx > (float)(sw - 1)) sx = (float)(sw - 1);
            int x0 = (int)floorf(sx);
            int x1 = x0 + 1 < sw ? x0 + 1 : x0;
            float fx = sx - (float)x0;
            const float* p00 = src + ((size_t)y0 * (size_t)sw + (size_t)x0) * 4u;
            const float* p10 = src + ((size_t)y0 * (size_t)sw + (size_t)x1) * 4u;
            const float* p01 = src + ((size_t)y1 * (size_t)sw + (size_t)x0) * 4u;
            const float* p11 = src + ((size_t)y1 * (size_t)sw + (size_t)x1) * 4u;
            float* o = dst + ((size_t)y * (size_t)dw + (size_t)x) * 4u;
            for (int c = 0; c < 4; c++) {
                float a = p00[c] + (p10[c] - p00[c]) * fx;
                float b = p01[c] + (p11[c] - p01[c]) * fx;
                o[c] = a + (b - a) * fy;
            }
        }
    }
    return dst;
}

/* Upload a float32 RGBA buffer as an MTLTexture(RGBA16Float). When
 * generate_mips is non-zero, allocates a full mip chain and runs
 * blit::generateMipmaps; otherwise creates a single mip. */
static id<MTLTexture> create_hdr_texture_metal(Gpu* gpu,
                                               const float* rgba_f32,
                                               int w, int h,
                                               int generate_mips,
                                               int* out_mip_levels)
{
    NSUInteger mip_count = 1;
    if (generate_mips) {
        int m = w > h ? w : h;
        while (m > 1) { mip_count++; m >>= 1; }
    }

    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                     width:(NSUInteger)w
                                    height:(NSUInteger)h
                                 mipmapped:(generate_mips ? YES : NO)];
    td.mipmapLevelCount = mip_count;
    /* RenderTarget needed because [blit generateMipmapsForTexture:] uses
     * an internal blit that writes through a render attachment. */
    td.usage = MTLTextureUsageShaderRead | (generate_mips ? MTLTextureUsageRenderTarget : 0);
    td.storageMode = MTLStorageModePrivate;

    id<MTLTexture> tex = [gpu->device newTextureWithDescriptor:td];
    if (!tex) return nil;

    /* Pack float32 → float16 on CPU and upload mip 0 via a staging buffer. */
    size_t pixel_count = (size_t)w * (size_t)h;
    NSUInteger row_bytes = (NSUInteger)w * 8;     /* RGBA16F = 8 B/pixel */
    NSUInteger total = row_bytes * (NSUInteger)h;
    id<MTLBuffer> staging = [gpu->device newBufferWithLength:total
                                                     options:MTLResourceStorageModeShared];
    if (!staging) return nil;
    rgba32f_to_rgba16f(rgba_f32, (uint16_t*)[staging contents], pixel_count);

    id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging
            sourceOffset:0
       sourceBytesPerRow:row_bytes
     sourceBytesPerImage:total
              sourceSize:MTLSizeMake((NSUInteger)w, (NSUInteger)h, 1)
               toTexture:tex
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];
    if (generate_mips) {
        [blit generateMipmapsForTexture:tex];
    }
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (out_mip_levels) *out_mip_levels = (int)mip_count;
    return tex;
}

/* Public hook so the loader (or harness) can stamp the USD-authored
 * DomeLight intensity onto the SceneHeader before the next render.
 * Patches the live scene_data_buf if one exists; otherwise the value
 * is consumed at the next gpu_build_rt_scene write (port from Vulkan
 * 43bbb19 / a376dc3 — Metal flavor). */
void gpu_set_env_intensity(Gpu* gpu, float intensity)
{
    if (!gpu) return;
    if (intensity < 0.0f) intensity = 0.0f;
    gpu->env_intensity = intensity;
    if (gpu->scene_data_buf) {
        float v = (intensity > 0.0f) ? intensity : 1.0f;
        memcpy((uint8_t*)[gpu->scene_data_buf contents] + 12, &v, 4);
    }
}

void gpu_set_dome_color(Gpu* gpu, float r, float g, float b, float intensity)
{
    if (!gpu) return;
    if (intensity < 0.0f) intensity = 0.0f;
    gpu->dome_color[0] = r < 0.0f ? 0.0f : r;
    gpu->dome_color[1] = g < 0.0f ? 0.0f : g;
    gpu->dome_color[2] = b < 0.0f ? 0.0f : b;
    gpu->dome_intensity = intensity;
    if (gpu->scene_data_buf) {
        float dome[4] = {
            gpu->dome_color[0],
            gpu->dome_color[1],
            gpu->dome_color[2],
            gpu->dome_intensity,
        };
        memcpy((uint8_t*)[gpu->scene_data_buf contents] + 16, dome, sizeof(dome));
    }
}

int gpu_load_environment(Gpu* gpu, const char* hdr_path)
{
    if (!gpu || !hdr_path) return 0;

    /* Default the env intensity to 1.0 if not yet set (loader will
     * override later when a USD DomeLight authors a value). */
    if (gpu->env_intensity <= 0.0f) gpu->env_intensity = 1.0f;

    int w, h, channels;
    float* hdr_data = stbi_loadf(hdr_path, &w, &h, &channels, 3);
    if (!hdr_data) {
        fprintf(stderr, "gpu_metal: stbi_loadf failed for %s\n", hdr_path);
        return 0;
    }
    fprintf(stderr, "gpu_metal: loading HDR environment %dx%d from %s\n",
            w, h, hdr_path);

    /* SH3 projection → diffuse irradiance + auto-exposure scale.
     * Target: white Lambertian surface lit from above produces ~1.0
     * after the shader's diffuseIBL = irradiance * (1/PI). */
    float sh_coeffs[9][3];
    sh_project_environment(hdr_data, w, h, sh_coeffs);

    float avg_lum = 0.2126f * sh_coeffs[0][0]
                  + 0.7152f * sh_coeffs[0][1]
                  + 0.0722f * sh_coeffs[0][2];
    float avg_irr = 0.886227f * avg_lum;
    float env_scale = (avg_irr > 0.001f) ? (3.14159f / avg_irr) : 1.0f;
    if (env_scale > 20.0f) env_scale = 20.0f;
    fprintf(stderr, "gpu_metal: HDR auto-exposure: avg_irr=%.3f, scale=%.2f\n",
            avg_irr, env_scale);

    /* Apply exposure scale to env data (RGB → RGBA, A=1). */
    float* env_rgba = (float*)malloc((size_t)w * h * 4 * sizeof(float));
    if (!env_rgba) { stbi_image_free(hdr_data); return 0; }
    for (int i = 0; i < w * h; i++) {
        env_rgba[i*4 + 0] = hdr_data[i*3 + 0] * env_scale;
        env_rgba[i*4 + 1] = hdr_data[i*3 + 1] * env_scale;
        env_rgba[i*4 + 2] = hdr_data[i*3 + 2] * env_scale;
        env_rgba[i*4 + 3] = 1.0f;
    }
    stbi_image_free(hdr_data);

    int max_dim = env_int_clamped("NUSD_ENV_MAX_DIM", 0, 0, 32768);
    if (max_dim > 0 && (w > max_dim || h > max_dim)) {
        float scale = (float)max_dim / (float)(w > h ? w : h);
        int nw = (int)floorf((float)w * scale + 0.5f);
        int nh = (int)floorf((float)h * scale + 0.5f);
        if (nw < 1) nw = 1;
        if (nh < 1) nh = 1;
        float* small = resize_env_rgba_bilinear(env_rgba, w, h, nw, nh);
        if (small) {
            fprintf(stderr,
                    "gpu_metal: downsampled HDR environment %dx%d -> %dx%d (NUSD_ENV_MAX_DIM=%d)\n",
                    w, h, nw, nh, max_dim);
            free(env_rgba);
            env_rgba = small;
            w = nw;
            h = nh;
        }
    }

    int env_mips = 0;
    /* GGX-prefiltered glossy mip chain (Karis 2014) — replaces the box-
     * filter that [blit generateMipmapsForTexture:] produced. 256 samples
     * per texel matches Vulkan; bottom mips drop to 64-128 internally for
     * the kernel-wide-but-output-tiny levels. */
    int samples = env_int_clamped("NUSD_GGX_PREFILTER_SAMPLES", 256, 1, 1024);
    if (env_flag_enabled("NUSD_FAST_IBL")) {
        fprintf(stderr,
                "gpu_metal: NUSD_FAST_IBL=1 — using Metal box-filter env mip chain (%dx%d)\n",
                w, h);
        gpu->env_texture = create_hdr_texture_metal(gpu, env_rgba, w, h, 1, &env_mips);
    } else {
        gpu->env_texture = create_hdr_texture_metal_ggx_prefiltered(
            gpu, env_rgba, w, h, samples, &env_mips);
    }
    free(env_rgba);
    if (!gpu->env_texture) {
        fprintf(stderr, "gpu_metal: env texture upload failed\n");
        return 0;
    }
    gpu->env_mip_levels = env_mips;

    /* If a scene is already built, patch SceneHeader.env_mip_levels in
     * place — Shared storage means CPU writes are visible to the GPU on
     * the next dispatch. Without this, the kernel's IBL branch stays
     * dead because env_mip_levels was sampled at scene-build time. */
    if (gpu->scene_data_buf) {
        float env_mip_f = (float)env_mips;
        memcpy((uint8_t*)[gpu->scene_data_buf contents] + 8, &env_mip_f, 4);
    }

    /* Diffuse irradiance map (256x128). Scale it by env_scale too so
     * shader does NOT need to know the exposure factor. */
    const int irr_w = 256, irr_h = 128;
    float* irr_rgba = (float*)malloc((size_t)irr_w * irr_h * 4 * sizeof(float));
    if (irr_rgba) {
        sh_render_irradiance(sh_coeffs, irr_rgba, irr_w, irr_h);
        for (int i = 0; i < irr_w * irr_h; i++) {
            irr_rgba[i*4 + 0] *= env_scale;
            irr_rgba[i*4 + 1] *= env_scale;
            irr_rgba[i*4 + 2] *= env_scale;
        }
        gpu->irr_texture = create_hdr_texture_metal(gpu, irr_rgba, irr_w, irr_h, 0, NULL);
        free(irr_rgba);
        if (!gpu->irr_texture) {
            fprintf(stderr, "gpu_metal: irradiance texture upload failed\n");
        } else {
            fprintf(stderr, "gpu_metal: SH irradiance map generated (%dx%d)\n",
                    irr_w, irr_h);
        }
    }

    /* BRDF integration LUT (128x128 RG16Float). Pack RG into RGBA32F →
     * RGBA16F upload, then alias the same texture as RG via the
     * shader's sampler (kernel reads .xy). Cheap on Apple Silicon. */
    const int lut_size = 128;
    float* lut_rg = (float*)malloc((size_t)lut_size * lut_size * 2 * sizeof(float));
    if (lut_rg) {
        generate_brdf_lut(lut_rg, lut_size);

        /* Convert RG → RGBA float32, then create a texture as RGBA16F.
         * (RG16F is supported but doing RGBA16F keeps one upload path.) */
        float* lut_rgba = (float*)malloc((size_t)lut_size * lut_size * 4 * sizeof(float));
        if (lut_rgba) {
            for (int i = 0; i < lut_size * lut_size; i++) {
                lut_rgba[i*4 + 0] = lut_rg[i*2 + 0];
                lut_rgba[i*4 + 1] = lut_rg[i*2 + 1];
                lut_rgba[i*4 + 2] = 0.0f;
                lut_rgba[i*4 + 3] = 1.0f;
            }
            gpu->brdf_lut = create_hdr_texture_metal(gpu, lut_rgba, lut_size, lut_size, 0, NULL);
            free(lut_rgba);
            if (!gpu->brdf_lut) {
                fprintf(stderr, "gpu_metal: BRDF LUT upload failed\n");
            } else {
                fprintf(stderr, "gpu_metal: BRDF integration LUT generated (%dx%d)\n",
                        lut_size, lut_size);
            }
        }
        free(lut_rg);
    }

    return 1;
}

void gpu_destroy_environment(Gpu* gpu)
{
    if (!gpu) return;
    gpu->env_texture     = nil;
    gpu->irr_texture     = nil;
    gpu->brdf_lut        = nil;
    gpu->env_mip_levels  = 0;
}

int gpu_get_env_mip_levels(Gpu* gpu) { return gpu ? gpu->env_mip_levels : 0; }

/* Build the environment-background pipeline lazily — first call needs the
 * shader_library populated (mesh.metal compiled). Re-entrant: returns 1 if
 * the pipeline already exists. Returns 0 on failure (renderer will skip the
 * env_bg draw and fall back to the begin_frame clear color). */
int gpu_create_env_bg_pipeline(Gpu* gpu)
{
    if (!gpu) return 0;
    if (gpu->env_bg_pipeline) return 1;
    if (!gpu->shader_library) return 0;

    id<MTLFunction> vfn = [gpu->shader_library newFunctionWithName:@"vertex_env_bg"];
    id<MTLFunction> ffn = [gpu->shader_library newFunctionWithName:@"fragment_env_bg"];
    if (!vfn || !ffn) {
        fprintf(stderr,
                "gpu_metal: env_bg pipeline missing function (vertex_env_bg=%s, "
                "fragment_env_bg=%s)\n",
                vfn ? "ok" : "MISSING", ffn ? "ok" : "MISSING");
        return 0;
    }

    MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction   = vfn;
    pd.fragmentFunction = ffn;
    pd.colorAttachments[0].pixelFormat = gpu->color_format;
    pd.depthAttachmentPixelFormat      = gpu->depth_format;
    /* No vertex descriptor — the shader synthesizes positions from
     * [[vertex_id]] alone, no vertex buffer required. */

    NSError* err = nil;
    id<MTLRenderPipelineState> pso =
        [gpu->device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!pso) {
        fprintf(stderr, "gpu_metal: env_bg pipeline build failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "(unknown)");
        return 0;
    }
    gpu->env_bg_pipeline = pso;
    return 1;
}

/* Paint the equirect HDR onto the current render target as a fullscreen
 * triangle. No-ops cleanly if the env_bg pipeline isn't built or the env
 * texture isn't loaded — caller can drive it unconditionally. Must be
 * called between gpu_begin_frame and gpu_end_frame so the active render
 * encoder is the one writing to the color target.
 *
 * view_inv / proj_inv match camera_get_view_inverse / camera_get_proj_inverse —
 * row-major mat4. The shader's `vec * mat` order does the equivalent
 * transpose so MSL column-major storage produces the same effective math
 * as the GLES port's `u_*Inv * vec` path. */
void gpu_draw_env_background(Gpu* gpu, const float* view_inv, const float* proj_inv)
{
    if (!gpu || !gpu->in_frame || !gpu->current_render_enc) return;
    if (!gpu->env_bg_pipeline || !gpu->env_texture) return;
    if (gpu->env_mip_levels <= 0) return;
    if (!view_inv || !proj_inv) return;

    [gpu->current_render_enc setRenderPipelineState:gpu->env_bg_pipeline];
    /* Skip depth-stencil state binding — the env_bg draw doesn't depend on
     * depth-write state, and the prior pipeline's depth-stencil state stays
     * in effect. With z=0.9999 NDC and depth-less compare, mesh draws
     * occlude env-bg correctly without us touching the state. */

    /* Push constants: { mat4 view_inv; mat4 proj_inv; vec4 env_params; } at
     * fragment buffer(1). env_params.x carries DomeLight intensity so the
     * raster background uses the same direct-dome exposure as RT misses. */
    float pc[36] = {0};
    memcpy(pc + 0,  view_inv, 16 * sizeof(float));
    memcpy(pc + 16, proj_inv, 16 * sizeof(float));
    pc[32] = gpu->env_intensity > 0.0f ? gpu->env_intensity : 1.0f;
    [gpu->current_render_enc setFragmentBytes:pc length:sizeof(pc) atIndex:1];

    /* Env texture at fragment texture(0). Mip 0 is sampled in the shader
     * (forced via level(0.0)), but the texture binding still needs the
     * full mip chain bound. */
    [gpu->current_render_enc setFragmentTexture:gpu->env_texture atIndex:0];

    /* Draw 3 vertices: one fullscreen triangle. No vertex buffer needed —
     * vertex_env_bg derives positions from [[vertex_id]]. */
    [gpu->current_render_enc drawPrimitives:MTLPrimitiveTypeTriangle
                                vertexStart:0
                                vertexCount:3];
}

/* Build the SSAO post-process pipeline lazily (needs shader_library). */
static int gpu_create_ssao_pipeline(Gpu* gpu)
{
    if (!gpu) return 0;
    if (gpu->ssao_pipeline) return 1;
    if (!gpu->shader_library) return 0;
    id<MTLFunction> vfn = [gpu->shader_library newFunctionWithName:@"vertex_ssao"];
    id<MTLFunction> ffn = [gpu->shader_library newFunctionWithName:@"fragment_ssao"];
    if (!vfn || !ffn) {
        fprintf(stderr, "gpu_metal: ssao pipeline missing function\n");
        return 0;
    }
    MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction   = vfn;
    pd.fragmentFunction = ffn;
    pd.colorAttachments[0].pixelFormat = gpu->color_format;
    NSError* err = nil;
    id<MTLRenderPipelineState> pso =
        [gpu->device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!pso) {
        fprintf(stderr, "gpu_metal: ssao pipeline build failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "(unknown)");
        return 0;
    }
    gpu->ssao_pipeline = pso;
    return 1;
}

/* Run the SSAO post-process: read color_target + depth_target, write the
 * darkened result to post_target. Call AFTER gpu_end_frame (color + depth
 * stored). params = { near, far, radius, strength, bias, power, 1/w, 1/h }.
 * Sets ssao_active so gpu_screenshot reads post_target. */
void gpu_ssao_composite(Gpu* gpu, const float* params)
{
    if (!gpu || !gpu->color_target || !gpu->depth_target || !gpu->post_target)
        return;
    if (!gpu_create_ssao_pipeline(gpu)) return;

    id<MTLCommandBuffer> cmd = [gpu->queue commandBuffer];
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture     = gpu->post_target;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionDontCare;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> enc =
        [cmd renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:gpu->ssao_pipeline];
    [enc setFragmentBytes:params length:(8 * sizeof(float)) atIndex:0];
    [enc setFragmentTexture:gpu->color_target atIndex:0];
    [enc setFragmentTexture:gpu->depth_target atIndex:1];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    gpu->ssao_active = 1;
}

/* ---- Direct texture accessors (Metal-only, public API) ---- */

void* gpu_get_metal_color_texture(Gpu* gpu)
{
    if (!gpu || !gpu->color_target) return nullptr;
    /* __bridge — caller is on the same ARC boundary; we hold the strong
     * reference inside Gpu, the consumer just borrows the id<MTLTexture>. */
    return (__bridge void*)gpu->color_target;
}

void* gpu_get_metal_depth_texture(Gpu* gpu)
{
    if (!gpu || !gpu->depth_target) return nullptr;
    return (__bridge void*)gpu->depth_target;
}

/* ---- Screenshot / Readback ---- */

/* Blit color_target into the shared-storage readback_buffer and wait. After
 * this returns, [readback_buffer contents] points to the active target storage
 * (RGBA8 sRGB or RGBA16Float display-linear, top-down rows). */
static int blit_color_to_readback(Gpu* gpu)
{
    if (!gpu || !gpu->color_target || !gpu->readback_buffer) return 0;

    /* If the SSAO post-pass ran this frame, its darkened result is in
     * post_target — read that back instead of the raw color. */
    id<MTLTexture> src = (gpu->ssao_active && gpu->post_target)
                       ? gpu->post_target : gpu->color_target;

    id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];

    NSUInteger row_bytes = (NSUInteger)gpu->width * gpu_color_bytes_per_pixel(gpu);
    [blit copyFromTexture:src
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake((NSUInteger)gpu->width, (NSUInteger)gpu->height, 1)
                 toBuffer:gpu->readback_buffer
        destinationOffset:0
   destinationBytesPerRow:row_bytes
 destinationBytesPerImage:row_bytes * (NSUInteger)gpu->height];

    /* No synchronizeResource: — `readback_buffer` is MTLStorageModeShared,
     * for which Metal API Validation rejects the call (it's only valid for
     * Managed). Apple Silicon makes the buffer CPU-coherent without it. */
    [blit endEncoding];

    [cb commit];
    [cb waitUntilCompleted];
    return 1;
}

int gpu_readback_pixels(Gpu* gpu, uint8_t* out_rgba8, uint32_t width, uint32_t height)
{
    if (!gpu || !out_rgba8) return 0;
    if ((int)width != gpu->width || (int)height != gpu->height) {
        fprintf(stderr, "gpu_metal: readback size mismatch (got %ux%u, target %dx%d)\n",
                width, height, gpu->width, gpu->height);
        return 0;
    }
    if (!gpu->has_frame) {
        memset(out_rgba8, 0, (size_t)width * (size_t)height * 4);
        return 1;
    }

    /* Async viewport path: the blit was already folded into the render cmd by
     * gpu_end_frame_rt. Return the PREVIOUS frame (the buffer that is NOT the
     * most-recently-submitted one) — it has had a full frame to complete, so
     * the waitUntilCompleted on its cmd is effectively free. The very first
     * frame has no previous, so we drain the just-submitted one (one-time sync).
     * 1-frame latency; on the deterministic backend a static frame is identical
     * to its predecessor, so a still camera shows no artifact. */
    if (gpu->async_readback && gpu->readback_buffer2) {
        int prev = 1 - gpu->rb_last;
        int idx = gpu->rb_valid[prev] ? prev : gpu->rb_last;
        id<MTLCommandBuffer> cmd = gpu->rb_cmd[idx];
        id<MTLBuffer> buf = idx ? gpu->readback_buffer2 : gpu->readback_buffer;
        if (cmd) [cmd waitUntilCompleted];
        if (buf) readback_storage_to_rgba8(gpu, [buf contents], out_rgba8,
                                           (size_t)width * (size_t)height);
        else memset(out_rgba8, 0, (size_t)width * (size_t)height * 4);
        return 1;
    }

    if (!blit_color_to_readback(gpu)) return 0;

    readback_storage_to_rgba8(gpu, [gpu->readback_buffer contents], out_rgba8,
                              (size_t)width * (size_t)height);
    return 1;
}

int gpu_readback_pixels_f32(Gpu* gpu, float* out_rgba, uint32_t width, uint32_t height)
{
    if (!gpu || !out_rgba) return 0;
    if ((int)width != gpu->width || (int)height != gpu->height) {
        fprintf(stderr, "gpu_metal: readback_f32 size mismatch (got %ux%u, target %dx%d)\n",
                width, height, gpu->width, gpu->height);
        return 0;
    }
    if (!gpu->has_frame) {
        memset(out_rgba, 0, (size_t)width * (size_t)height * 4 * sizeof(float));
        return 1;
    }

    if (gpu->async_readback && gpu->readback_buffer2) {
        int prev = 1 - gpu->rb_last;
        int idx = gpu->rb_valid[prev] ? prev : gpu->rb_last;
        id<MTLCommandBuffer> cmd = gpu->rb_cmd[idx];
        id<MTLBuffer> buf = idx ? gpu->readback_buffer2 : gpu->readback_buffer;
        if (cmd) [cmd waitUntilCompleted];
        if (buf) readback_storage_to_rgba32f(gpu, [buf contents], out_rgba,
                                             (size_t)width * (size_t)height);
        else memset(out_rgba, 0, (size_t)width * (size_t)height * 4 * sizeof(float));
        return 1;
    }

    if (!blit_color_to_readback(gpu)) return 0;

    readback_storage_to_rgba32f(gpu, [gpu->readback_buffer contents], out_rgba,
                                (size_t)width * (size_t)height);
    return 1;
}

int gpu_screenshot(Gpu* gpu, const char* path)
{
    if (!gpu || !path) return 0;
    if (!gpu->has_frame) {
        fprintf(stderr, "gpu_metal: gpu_screenshot called with no frame rendered\n");
        return 0;
    }
    FILE* f = open_ppm_for_write(path, (uint32_t)gpu->width, (uint32_t)gpu->height);
    if (!f) return 0;

    const size_t rgba_bytes = (size_t)gpu->width * (size_t)gpu->height * 4;
    uint8_t* rgba = (uint8_t*)malloc(rgba_bytes);
    if (!rgba) { fclose(f); return 0; }
    if (!gpu_readback_pixels(gpu, rgba, (uint32_t)gpu->width, (uint32_t)gpu->height)) {
        free(rgba);
        fclose(f);
        return 0;
    }

    const size_t out_row_bytes = (size_t)gpu->width * 3;
    uint8_t* row = (uint8_t*)malloc(out_row_bytes);
    if (!row) { free(rgba); fclose(f); return 0; }

    for (int y = 0; y < gpu->height; y++) {
        const uint8_t* in_row = rgba + (size_t)y * gpu->width * 4;
        for (int x = 0; x < gpu->width; x++) {
            row[x * 3 + 0] = in_row[x * 4 + 0];
            row[x * 3 + 1] = in_row[x * 4 + 1];
            row[x * 3 + 2] = in_row[x * 4 + 2];
        }
        fwrite(row, 1, out_row_bytes, f);
    }
    free(row);
    free(rgba);
    fclose(f);
    return 1;
}

/* ---- Ray tracing ---- */

int gpu_rt_available(Gpu* gpu) { return gpu ? gpu->rt_available : 0; }
int gpu_rt_built(Gpu* gpu)     { return gpu ? gpu->rt_built     : 0; }

/* Layout of the SceneData buffer. Mirrors the SceneHeader / MeshData structs in
 * src/shaders/raytrace.metal (and conceptually mirrors the SceneData std430
 * layout in raytrace.rchit.glsl, except we store offsets rather than 64-bit
 * device addresses since Metal kernels receive the shared vertex/index buffers
 * via direct setBuffer rather than buffer_reference). */
static const NSUInteger SCENE_HEADER_BYTES   = 32;
static const NSUInteger SCENE_PER_MESH_BYTES = 32;

/* Build a compute pipeline state for the named function with curve_isect
 * linked in (Phase 11.A). Linked-function plumbing lets the kernel's
 * intersector dispatch into curve_isect when it hits AABB geometry. We
 * always link the curve function — even for curve-free scenes — so a
 * single pipeline serves both shapes. The runtime cost is one extra
 * function-handle slot in the pipeline; no perf impact when the IFT is
 * empty. */
static id<MTLComputePipelineState> build_rt_compute_pipeline(
    Gpu* gpu, NSString* kernel_name, NSError** out_err)
{
    id<MTLFunction> kernel_fn = [gpu->rt_library newFunctionWithName:kernel_name];
    if (!kernel_fn) {
        if (out_err) {
            NSDictionary* ui = @{ NSLocalizedDescriptionKey:
                [NSString stringWithFormat:@"%@ not found in raytrace.metal", kernel_name] };
            *out_err = [NSError errorWithDomain:@"gpu_metal" code:1 userInfo:ui];
        }
        return nil;
    }

    id<MTLFunction> curve_fn = [gpu->rt_library newFunctionWithName:@"curve_isect"];
    if (!curve_fn) {
        if (out_err) {
            NSDictionary* ui = @{ NSLocalizedDescriptionKey:
                @"curve_isect not found in raytrace.metal" };
            *out_err = [NSError errorWithDomain:@"gpu_metal" code:1 userInfo:ui];
        }
        return nil;
    }

    MTLLinkedFunctions* lf = [MTLLinkedFunctions linkedFunctions];
    lf.functions = @[curve_fn];

    MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = kernel_fn;
    desc.linkedFunctions = lf;
    /* Allow indirect command buffers to bypass the IFT — required when
     * any linked function is present, even if the IFT is bound at a
     * fixed slot in the kernel. */
    desc.supportIndirectCommandBuffers = NO;

    return [gpu->device newComputePipelineStateWithDescriptor:desc
                                                      options:MTLPipelineOptionNone
                                                   reflection:nil
                                                        error:out_err];
}

/* Build a 1-slot intersection function table for the given pipeline,
 * holding curve_isect at slot 0. The function's [[buffer(11)]] param
 * (curve segments) is bound on the IFT itself so the intersection
 * function sees it; the kernel sees its own copy at buffer(14)/15. */
static id<MTLIntersectionFunctionTable> build_curve_ift(
    Gpu* gpu, id<MTLComputePipelineState> pipeline)
{
    if (!pipeline) return nil;

    MTLIntersectionFunctionTableDescriptor* iftDesc =
        [MTLIntersectionFunctionTableDescriptor intersectionFunctionTableDescriptor];
    iftDesc.functionCount = 1;

    id<MTLIntersectionFunctionTable> ift =
        [pipeline newIntersectionFunctionTableWithDescriptor:iftDesc];
    if (!ift) return nil;

    id<MTLFunction> curve_fn = [gpu->rt_library newFunctionWithName:@"curve_isect"];
    id<MTLFunctionHandle> handle = [pipeline functionHandleWithFunction:curve_fn];
    if (!handle) return nil;
    [ift setFunction:handle atIndex:0];
    return ift;
}

/* Set up gpu->curve_dummy_buf used as a placeholder when curves aren't
 * present. Must be ≥ MSL's stride for a single CurveSegment (32 bytes —
 * the struct uses packed_float3 so the GPU stride matches the host's
 * 32 B/segment upload) so Metal API Validation accepts the binding for
 * `device const CurveSegment*` even though the kernel never reads through
 * it (no AABB hits occur in a curves-free scene). 256 B leaves headroom
 * for any other slot whose stride is ≤ 256. */
static void ensure_curve_dummy_buf(Gpu* gpu)
{
    if (gpu->curve_dummy_buf) return;
    uint8_t z[256] = {0};
    gpu->curve_dummy_buf = [gpu->device newBufferWithBytes:z
                                                    length:sizeof(z)
                                                   options:MTLResourceStorageModeShared];
}

/* (Re)point the IFT's [[buffer(11)]] slot at the live curve segment
 * buffer (or the dummy when curves are absent). Called from the bind
 * helpers — cheap on Apple Silicon, just patches the IFT descriptor. */
static void update_curve_ift_buffer(
    Gpu* gpu, id<MTLIntersectionFunctionTable> ift)
{
    if (!ift) return;
    id<MTLBuffer> seg = gpu->curve_seg_buf ? gpu->curve_seg_buf : gpu->curve_dummy_buf;
    [ift setBuffer:seg offset:0 atIndex:11];
}

/* Compile raytrace.metal and create the rt_render compute pipeline once.
 * Persistent across scene rebuilds — gpu_destroy_rt_scene keeps these. */
static int ensure_rt_pipeline(Gpu* gpu)
{
    if (gpu->rt_pipeline && gpu->curve_ift) return 1;

    if (!gpu->rt_library) {
        char rt_path[1024];
        snprintf(rt_path, sizeof(rt_path), "%s/raytrace.metal", SHADER_DIR);
        gpu->rt_library = compile_msl_file(gpu->device, rt_path);
        if (!gpu->rt_library) {
            fprintf(stderr, "gpu_metal: raytrace.metal compile failed\n");
            return 0;
        }
    }

    if (!gpu->rt_pipeline) {
        NSError* err = nil;
        gpu->rt_pipeline = build_rt_compute_pipeline(gpu, @"rt_render", &err);
        if (!gpu->rt_pipeline) {
            fprintf(stderr, "gpu_metal: RT pipeline build failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
    }

    if (!gpu->curve_ift) {
        gpu->curve_ift = build_curve_ift(gpu, gpu->rt_pipeline);
        if (!gpu->curve_ift) {
            fprintf(stderr, "gpu_metal: curve IFT build failed (rt_pipeline)\n");
            return 0;
        }
    }
    ensure_curve_dummy_buf(gpu);
    update_curve_ift_buffer(gpu, gpu->curve_ift);
    return 1;
}

/* Convert renderer.c's VkTransformMatrixKHR-format 3x4 (row-major; vk[r*4+c] is
 * element (r,c)) into Metal's MTLPackedFloat4x3 (4 packed columns of 3 floats;
 * mt.columns[c][r] is element (r,c)). The two layouts have the same number of
 * elements but different memory order, so this is a transpose-into-different-
 * stride. Silent breakage here puts meshes at wrong positions. */
static inline void vk3x4_to_mtl4x3(const float vk[12], MTLPackedFloat4x3* mt)
{
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 3; r++) {
            mt->columns[c][r] = vk[r * 4 + c];
        }
    }
}

/* Write per-mesh entry color into the scene_data_buf at offset 16 within the
 * mesh slot. Shared storage means the GPU sees the new value on the next
 * encoder dispatch (no flush needed on Apple Silicon). */
static inline void write_scene_color(uint8_t* p, uint32_t m, const float color[3])
{
    uint8_t* e = p + SCENE_HEADER_BYTES + (NSUInteger)m * SCENE_PER_MESH_BYTES;
    float c[4] = { color[0], color[1], color[2], 1.0f };
    memcpy(e + 16, c, sizeof(c));
}

void gpu_destroy_rt_scene(Gpu* gpu)
{
    if (!gpu) return;
    /* Keep rt_library / rt_pipeline alive — they're persistent. */
    gpu->blas_list      = nil;
    gpu->blas_count     = 0;
    gpu->tlas           = nil;
    gpu->instance_buf   = nil;
    gpu->scene_data_buf = nil;
    gpu->tlas_scratch   = nil;
    gpu->rt_vertex_buf  = nil;
    gpu->rt_index_buf   = nil;
    gpu->scene_nmeshes  = 0;
    free(gpu->mesh_to_blas);
    gpu->mesh_to_blas   = nullptr;
    gpu->rt_built       = 0;

    /* Phase 11.A — curve state has independent lifetime from the mesh
     * scene. nu_render rebuilds the mesh TLAS via destroy + build, but
     * curves should persist across that (curves were uploaded BEFORE
     * the TLAS rebuild so the TLAS could include the curve instance).
     * curve_seg_buf / curve_color_buf / curve_aabb_buf / curve_blas are
     * cleared in gpu_upload_curve_data when re-uploaded, and in
     * nu_clear_scene's per-buffer free path. */
}

int gpu_build_rt_scene(Gpu* gpu,
                       const GpuRtMeshDesc* meshes, uint32_t nmeshes,
                       const uint32_t* /*rgen_spv*/, uint32_t /*rgen_size*/,
                       const uint32_t* /*miss_spv*/, uint32_t /*miss_size*/,
                       const uint32_t* /*chit_spv*/, uint32_t /*chit_size*/)
{
    if (!gpu || !gpu->rt_available) return 0;
    /* nmeshes == 0 is valid for a curves-only scene (Phase 11.A). The
     * TLAS still needs a curve instance pointing at gpu->curve_blas;
     * if neither meshes nor curves are present, there's no scene to
     * build. */
    if (nmeshes == 0 && !gpu->curve_blas) return 0;
    if (nmeshes > 0 && !meshes) return 0;
    if (!ensure_rt_pipeline(gpu)) return 0;

    /* Renderer.c packs all meshes into one shared vertex_buf + index_buf;
     * remember them so the RT kernel can index into them directly. With
     * nmeshes==0 (curves-only scene), the kernel doesn't read the mesh
     * vertex/index buffers — but it still needs *something* bound at
     * those slots, so we'll fall back to dummy buffers in begin_frame_rt. */
    gpu->rt_vertex_buf = (nmeshes > 0) ? meshes[0].vertex_buf->buffer : nil;
    gpu->rt_index_buf  = (nmeshes > 0) ? meshes[0].index_buf->buffer  : nil;

    /* Mesh → BLAS mapping (only unique prototypes get a BLAS).
     * Assumes prototypes appear in `meshes[]` BEFORE any of their instances
     * (i.e. prototype_idx < m whenever it isn't self-referential). Renderer.c
     * sets prototype_idx = own index for every mesh, so the order is trivially
     * correct; the Vulkan reference makes the same assumption. */
    gpu->scene_nmeshes = nmeshes;
    free(gpu->mesh_to_blas);
    gpu->mesh_to_blas = (nmeshes > 0)
        ? (uint32_t*)malloc((size_t)nmeshes * sizeof(uint32_t))
        : nullptr;
    uint32_t nblas = 0;
    for (uint32_t m = 0; m < nmeshes; m++) {
        int proto = meshes[m].prototype_idx;
        if (proto < 0 || (uint32_t)proto == m) {
            gpu->mesh_to_blas[m] = nblas++;
        } else {
            gpu->mesh_to_blas[m] = gpu->mesh_to_blas[proto];
        }
    }
    gpu->blas_count = nblas;

    /* Phase 1: build BLAS descriptors and query sizes */
    NSMutableArray<MTLPrimitiveAccelerationStructureDescriptor*>* descs =
        [[NSMutableArray alloc] initWithCapacity:nblas];
    for (uint32_t i = 0; i < nblas; i++)
        [descs addObject:(id)[NSNull null]];

    NSMutableArray<id<MTLAccelerationStructure>>* blasArr =
        [[NSMutableArray alloc] initWithCapacity:nblas];
    for (uint32_t i = 0; i < nblas; i++)
        [blasArr addObject:(id)[NSNull null]];

    NSUInteger max_scratch = 0;
    NSUInteger total_blas_bytes = 0;

    for (uint32_t m = 0; m < nmeshes; m++) {
        int proto = meshes[m].prototype_idx;
        if (proto >= 0 && (uint32_t)proto != m) continue;
        uint32_t b = gpu->mesh_to_blas[m];
        const GpuRtMeshDesc* md = &meshes[m];

        MTLAccelerationStructureTriangleGeometryDescriptor* gd =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        gd.vertexBuffer       = gpu->rt_vertex_buf;
        gd.vertexBufferOffset = (NSUInteger)md->vertex_offset * (NSUInteger)md->vertex_stride;
        gd.vertexStride       = (NSUInteger)md->vertex_stride;
        gd.vertexFormat       = MTLAttributeFormatFloat3;
        gd.indexBuffer        = gpu->rt_index_buf;
        gd.indexBufferOffset  = (NSUInteger)md->index_offset * sizeof(uint32_t);
        gd.indexType          = MTLIndexTypeUInt32;
        gd.triangleCount      = (NSUInteger)md->index_count / 3;
        gd.opaque             = YES;

        MTLPrimitiveAccelerationStructureDescriptor* desc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        desc.geometryDescriptors = @[gd];
        descs[b] = desc;

        MTLAccelerationStructureSizes sizes =
            [gpu->device accelerationStructureSizesWithDescriptor:desc];
        if (sizes.buildScratchBufferSize > max_scratch) max_scratch = sizes.buildScratchBufferSize;
        total_blas_bytes += sizes.accelerationStructureSize;

        id<MTLAccelerationStructure> blas =
            [gpu->device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        if (!blas) {
            fprintf(stderr, "gpu_metal: BLAS alloc failed (mesh=%u, %.1f MB)\n",
                    m, (double)sizes.accelerationStructureSize / (1024.0 * 1024.0));
            gpu_destroy_rt_scene(gpu);
            return 0;
        }
        blasArr[b] = blas;
    }

    /* Phase 2: build all BLASes in one cmdbuf with one shared scratch.
     * Curves-only scene (nblas == 0): skip the mesh BLAS build entirely. */
    if (nblas > 0) {
        id<MTLBuffer> blas_scratch =
            [gpu->device newBufferWithLength:max_scratch
                                     options:MTLResourceStorageModePrivate];
        if (!blas_scratch) {
            fprintf(stderr, "gpu_metal: BLAS scratch alloc failed (%.1f MB)\n",
                    (double)max_scratch / (1024.0 * 1024.0));
            gpu_destroy_rt_scene(gpu);
            return 0;
        }

        id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> enc =
            [cb accelerationStructureCommandEncoder];
        for (uint32_t b = 0; b < nblas; b++) {
            [enc buildAccelerationStructure:blasArr[b]
                                 descriptor:descs[b]
                              scratchBuffer:blas_scratch
                        scratchBufferOffset:0];
            /* Phase 4 MVP defers BLAS compaction. The compaction step would
             * write the compacted size to a buffer with `writeCompactedSize`,
             * read it back, and call `copyAndCompactAccelerationStructure`
             * — saves ~30-40% memory but adds a build/read/build dance. */
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    /* Phase 11.A — append the curve BLAS to blasArr (if curves are
     * present) so the unified TLAS can dispatch into AABB geometry too.
     * The curve instance gets mask 0xFE (excluded from shadow rays at
     * mask 0x01) and intersectionFunctionTableOffset 0 (slot 0 of the
     * IFT == curve_isect). */
    uint32_t curve_inst_idx = (uint32_t)-1;
    if (gpu->curve_blas) {
        curve_inst_idx = (uint32_t)[blasArr count];
        [blasArr addObject:gpu->curve_blas];
    }
    gpu->blas_list = blasArr;

    uint32_t total_instances = nmeshes + (gpu->curve_blas ? 1u : 0u);

    /* Phase 3: instance buffer (Shared, persistent — refit re-uses it) */
    NSUInteger inst_bytes = (NSUInteger)total_instances * sizeof(MTLAccelerationStructureInstanceDescriptor);
    gpu->instance_buf = [gpu->device newBufferWithLength:inst_bytes
                                                 options:MTLResourceStorageModeShared];
    if (!gpu->instance_buf) {
        fprintf(stderr, "gpu_metal: instance buffer alloc failed\n");
        gpu_destroy_rt_scene(gpu);
        return 0;
    }
    {
        MTLAccelerationStructureInstanceDescriptor* inst =
            (MTLAccelerationStructureInstanceDescriptor*)[gpu->instance_buf contents];
        memset(inst, 0, inst_bytes);
        for (uint32_t m = 0; m < nmeshes; m++) {
            vk3x4_to_mtl4x3(meshes[m].transform, &inst[m].transformationMatrix);
            inst[m].options = MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
            inst[m].mask                            = meshes[m].mask ? meshes[m].mask : 0u;
            inst[m].intersectionFunctionTableOffset = 0;
            inst[m].accelerationStructureIndex      = gpu->mesh_to_blas[m];
        }
        if (gpu->curve_blas) {
            /* Identity transform — segments are extracted in world space.
             * MTLPackedFloat4x3 is 4 columns of MTLPackedFloat3; assign
             * field-by-field since the inline-initializer-list ctor isn't
             * available on the struct in macOS-13's MTLAccelerationStructureTypes.h. */
            uint32_t ci = nmeshes;
            MTLPackedFloat4x3* m = &inst[ci].transformationMatrix;
            m->columns[0].x = 1; m->columns[0].y = 0; m->columns[0].z = 0;
            m->columns[1].x = 0; m->columns[1].y = 1; m->columns[1].z = 0;
            m->columns[2].x = 0; m->columns[2].y = 0; m->columns[2].z = 1;
            m->columns[3].x = 0; m->columns[3].y = 0; m->columns[3].z = 0;
            inst[ci].options = 0;  /* AABB geometry; triangle-culling flag inapplicable */
            inst[ci].mask                            = 0xFEu;  /* skip shadow rays (mask 0x01) */
            inst[ci].intersectionFunctionTableOffset = 0;       /* curve_isect at IFT[0] */
            inst[ci].accelerationStructureIndex      = curve_inst_idx;
        }
    }

    /* Phase 4: TLAS build (refit-allowed) */
    MTLInstanceAccelerationStructureDescriptor* td =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    td.instanceDescriptorBuffer       = gpu->instance_buf;
    td.instanceDescriptorBufferOffset = 0;
    td.instanceDescriptorStride       = sizeof(MTLAccelerationStructureInstanceDescriptor);
    td.instanceCount                  = total_instances;
    td.instancedAccelerationStructures = blasArr;
    td.usage                          = MTLAccelerationStructureUsageRefit;

    MTLAccelerationStructureSizes ts =
        [gpu->device accelerationStructureSizesWithDescriptor:td];
    fprintf(stderr,
            "gpu_metal: TLAS sizing — instances=%u  AS_size=%.2f GiB  "
            "buildScratch=%.2f GiB  maxBufferLength=%.2f GiB  "
            "recommendedMaxWorkingSet=%.2f GiB\n",
            (unsigned)total_instances,
            (double)ts.accelerationStructureSize / (1024.0*1024.0*1024.0),
            (double)ts.buildScratchBufferSize / (1024.0*1024.0*1024.0),
            (double)gpu->device.maxBufferLength / (1024.0*1024.0*1024.0),
            (double)gpu->device.recommendedMaxWorkingSetSize / (1024.0*1024.0*1024.0));
    gpu->tlas = [gpu->device newAccelerationStructureWithSize:ts.accelerationStructureSize];
    if (!gpu->tlas) {
        fprintf(stderr, "gpu_metal: TLAS alloc failed\n");
        gpu_destroy_rt_scene(gpu);
        return 0;
    }

    NSUInteger tlas_scratch_sz = ts.buildScratchBufferSize;
    if (ts.refitScratchBufferSize > tlas_scratch_sz) tlas_scratch_sz = ts.refitScratchBufferSize;
    gpu->tlas_scratch = [gpu->device newBufferWithLength:tlas_scratch_sz
                                                 options:MTLResourceStorageModePrivate];
    if (!gpu->tlas_scratch) {
        fprintf(stderr, "gpu_metal: TLAS scratch alloc failed\n");
        gpu_destroy_rt_scene(gpu);
        return 0;
    }

    {
        id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> enc =
            [cb accelerationStructureCommandEncoder];
        [enc buildAccelerationStructure:gpu->tlas
                             descriptor:td
                          scratchBuffer:gpu->tlas_scratch
                    scratchBufferOffset:0];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    /* Phase 5: SceneData buffer. With nmeshes==0 the meshes[] tail is
     * empty, but we still bind the same buffer at offset SCENE_HEADER_BYTES
     * for the kernel's MeshData* param — Metal API Validation requires
     * offset < length, so allocate at least one stride past the header. */
    NSUInteger sd_size = SCENE_HEADER_BYTES
                       + ((NSUInteger)(nmeshes > 0 ? nmeshes : 1)) * SCENE_PER_MESH_BYTES;
    gpu->scene_data_buf = [gpu->device newBufferWithLength:sd_size
                                                   options:MTLResourceStorageModeShared];
    if (!gpu->scene_data_buf) {
        fprintf(stderr, "gpu_metal: scene data buffer alloc failed\n");
        gpu_destroy_rt_scene(gpu);
        return 0;
    }
    {
        uint8_t* p = (uint8_t*)[gpu->scene_data_buf contents];
        /* renderer.c packs vertices at 48 B (12 floats) regardless of mesh
         * count; default to that when nmeshes == 0 (curves-only scene). */
        uint32_t vstride_floats = (nmeshes > 0) ? (meshes[0].vertex_stride / 4) : 12u;
        /* Placeholder-only counts as "no materials" for shader gating —
         * see comment on mat_only_placeholder. */
        uint32_t has_mat = ((gpu->has_materials && !gpu->mat_only_placeholder) ? 1u : 0u)
                         | view_transform_bit();
        float    env_mip = (float)gpu->env_mip_levels;
        float    env_int = (gpu->env_intensity > 0.0f) ? gpu->env_intensity : 1.0f;
        float    dome[4] = {
            gpu->dome_color[0],
            gpu->dome_color[1],
            gpu->dome_color[2],
            gpu->dome_intensity,
        };
        memcpy(p +  0, &vstride_floats, 4);
        memcpy(p +  4, &has_mat,        4);
        memcpy(p +  8, &env_mip,        4);
        memcpy(p + 12, &env_int,        4);
        memcpy(p + 16, dome, sizeof(dome));
        for (uint32_t m = 0; m < nmeshes; m++) {
            uint8_t* e = p + SCENE_HEADER_BYTES + (NSUInteger)m * SCENE_PER_MESH_BYTES;
            uint32_t v_off = meshes[m].vertex_offset;
            uint32_t i_off = meshes[m].index_offset;
            int32_t  mat_idx = meshes[m].material_index;
            uint32_t z = 0;
            memcpy(e +  0, &v_off,   4);
            memcpy(e +  4, &i_off,   4);
            memcpy(e +  8, &mat_idx, 4);  /* MeshData.material_index (was _pad0) */
            memcpy(e + 12, &z,       4);
            write_scene_color(p, m, meshes[m].color);
        }
    }

    fprintf(stderr, "gpu_metal: RT scene built (%u meshes, %u BLAS, %.1f MB BLAS)\n",
            nmeshes, nblas, (double)total_blas_bytes / (1024.0 * 1024.0));

    gpu->rt_built = 1;
    return 1;
}

int gpu_rt_can_reuse_blas(Gpu* gpu)
{
    return gpu && gpu->rt_available && gpu->rt_built
        && gpu->blas_list != nil && gpu->blas_count > 0;
}

/* Rebuild the TLAS + instance/scene-data buffers while REUSING the cached
 * per-mesh BLAS. Camera-invariant geometry (the base meshes) is unchanged;
 * only the visible PI-clone tail + camera vary, so the ~140 s BLAS build is
 * skipped and we pay only the cheap (0.14 GiB) TLAS rebuild per camera. */
int gpu_rebuild_rt_tlas(Gpu* gpu, const GpuRtMeshDesc* meshes, uint32_t nmeshes)
{
    if (!gpu_rt_can_reuse_blas(gpu)) return 0;
    if (nmeshes == 0 && !gpu->curve_blas) return 0;
    if (nmeshes > 0 && !meshes) return 0;

    /* Recompute mesh→BLAS. Base meshes are byte-identical across cameras, so
     * the self-prototype walk yields the SAME blas indices as the cached
     * build; PI clones reference a base proto (< rt_base_count) and never
     * introduce a new BLAS. If the unique count drifts from the cache, the
     * geometry actually changed — bail so the caller does a full rebuild. */
    uint32_t* m2b = (uint32_t*)malloc((size_t)(nmeshes ? nmeshes : 1) * sizeof(uint32_t));
    if (!m2b) return 0;
    uint32_t nblas = 0;
    for (uint32_t m = 0; m < nmeshes; m++) {
        int proto = meshes[m].prototype_idx;
        if (proto < 0 || (uint32_t)proto == m) m2b[m] = nblas++;
        else                                   m2b[m] = m2b[proto];
    }
    if (nblas != gpu->blas_count) {   /* base geometry changed — can't reuse */
        free(m2b);
        return 0;
    }

    free(gpu->mesh_to_blas);
    gpu->mesh_to_blas  = m2b;
    gpu->scene_nmeshes = nmeshes;
    gpu->rt_vertex_buf = (nmeshes > 0) ? meshes[0].vertex_buf->buffer : gpu->rt_vertex_buf;
    gpu->rt_index_buf  = (nmeshes > 0) ? meshes[0].index_buf->buffer  : gpu->rt_index_buf;

    /* Re-sync the curve slot: the culled curve BLAS is rebuilt per camera
     * (rebuild_curve_blas_culled), so blas_list's tail must point at the
     * CURRENT gpu->curve_blas. Trim to the cached base BLAS, then append the
     * curve BLAS if present. */
    while ((uint32_t)[gpu->blas_list count] > gpu->blas_count)
        [gpu->blas_list removeLastObject];
    if (gpu->curve_blas)
        [gpu->blas_list addObject:gpu->curve_blas];

    /* gpu->blas_list now holds [base BLAS x blas_count] (+ curve_blas at the
     * tail when curves are present). curve_inst_idx == blas_count, exactly as
     * the full build wrote it. */
    uint32_t curve_inst_idx = gpu->curve_blas ? gpu->blas_count : (uint32_t)-1;
    uint32_t total_instances = nmeshes + (gpu->curve_blas ? 1u : 0u);

    /* Instance buffer (size varies per camera → realloc each call). */
    NSUInteger inst_bytes = (NSUInteger)total_instances * sizeof(MTLAccelerationStructureInstanceDescriptor);
    gpu->instance_buf = [gpu->device newBufferWithLength:inst_bytes
                                                 options:MTLResourceStorageModeShared];
    if (!gpu->instance_buf) {
        fprintf(stderr, "gpu_metal: (reuse) instance buffer alloc failed\n");
        return 0;
    }
    {
        MTLAccelerationStructureInstanceDescriptor* inst =
            (MTLAccelerationStructureInstanceDescriptor*)[gpu->instance_buf contents];
        memset(inst, 0, inst_bytes);
        for (uint32_t m = 0; m < nmeshes; m++) {
            vk3x4_to_mtl4x3(meshes[m].transform, &inst[m].transformationMatrix);
            inst[m].options = MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
            inst[m].mask                            = meshes[m].mask ? meshes[m].mask : 0u;
            inst[m].intersectionFunctionTableOffset = 0;
            inst[m].accelerationStructureIndex      = gpu->mesh_to_blas[m];
        }
        if (gpu->curve_blas) {
            uint32_t ci = nmeshes;
            MTLPackedFloat4x3* m = &inst[ci].transformationMatrix;
            m->columns[0].x = 1; m->columns[0].y = 0; m->columns[0].z = 0;
            m->columns[1].x = 0; m->columns[1].y = 1; m->columns[1].z = 0;
            m->columns[2].x = 0; m->columns[2].y = 0; m->columns[2].z = 1;
            m->columns[3].x = 0; m->columns[3].y = 0; m->columns[3].z = 0;
            inst[ci].options = 0;
            inst[ci].mask                            = 0xFEu;
            inst[ci].intersectionFunctionTableOffset = 0;
            inst[ci].accelerationStructureIndex      = curve_inst_idx;
        }
    }

    /* TLAS (refit-allowed) — reuses the cached BLAS array verbatim. */
    MTLInstanceAccelerationStructureDescriptor* td =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    td.instanceDescriptorBuffer       = gpu->instance_buf;
    td.instanceDescriptorBufferOffset = 0;
    td.instanceDescriptorStride       = sizeof(MTLAccelerationStructureInstanceDescriptor);
    td.instanceCount                  = total_instances;
    td.instancedAccelerationStructures = gpu->blas_list;
    td.usage                          = MTLAccelerationStructureUsageRefit;

    MTLAccelerationStructureSizes ts =
        [gpu->device accelerationStructureSizesWithDescriptor:td];
    gpu->tlas = [gpu->device newAccelerationStructureWithSize:ts.accelerationStructureSize];
    if (!gpu->tlas) {
        fprintf(stderr, "gpu_metal: (reuse) TLAS alloc failed\n");
        return 0;
    }
    NSUInteger tlas_scratch_sz = ts.buildScratchBufferSize;
    if (ts.refitScratchBufferSize > tlas_scratch_sz) tlas_scratch_sz = ts.refitScratchBufferSize;
    gpu->tlas_scratch = [gpu->device newBufferWithLength:tlas_scratch_sz
                                                 options:MTLResourceStorageModePrivate];
    if (!gpu->tlas_scratch) {
        fprintf(stderr, "gpu_metal: (reuse) TLAS scratch alloc failed\n");
        return 0;
    }
    {
        id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> enc =
            [cb accelerationStructureCommandEncoder];
        [enc buildAccelerationStructure:gpu->tlas
                             descriptor:td
                          scratchBuffer:gpu->tlas_scratch
                    scratchBufferOffset:0];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    /* Per-mesh scene data (colors / material idx vary per visible set). */
    NSUInteger sd_size = SCENE_HEADER_BYTES
                       + ((NSUInteger)(nmeshes > 0 ? nmeshes : 1)) * SCENE_PER_MESH_BYTES;
    gpu->scene_data_buf = [gpu->device newBufferWithLength:sd_size
                                                   options:MTLResourceStorageModeShared];
    if (!gpu->scene_data_buf) {
        fprintf(stderr, "gpu_metal: (reuse) scene data buffer alloc failed\n");
        return 0;
    }
    {
        uint8_t* p = (uint8_t*)[gpu->scene_data_buf contents];
        uint32_t vstride_floats = (nmeshes > 0) ? (meshes[0].vertex_stride / 4) : 12u;
        uint32_t has_mat = (gpu->has_materials && !gpu->mat_only_placeholder) ? 1u : 0u;
        float    env_mip = (float)gpu->env_mip_levels;
        float    env_int = (gpu->env_intensity > 0.0f) ? gpu->env_intensity : 1.0f;
        float    dome[4] = { gpu->dome_color[0], gpu->dome_color[1],
                             gpu->dome_color[2], gpu->dome_intensity };
        memcpy(p +  0, &vstride_floats, 4);
        memcpy(p +  4, &has_mat,        4);
        memcpy(p +  8, &env_mip,        4);
        memcpy(p + 12, &env_int,        4);
        memcpy(p + 16, dome, sizeof(dome));
        for (uint32_t m = 0; m < nmeshes; m++) {
            uint8_t* e = p + SCENE_HEADER_BYTES + (NSUInteger)m * SCENE_PER_MESH_BYTES;
            uint32_t v_off = meshes[m].vertex_offset;
            uint32_t i_off = meshes[m].index_offset;
            int32_t  mat_idx = meshes[m].material_index;
            uint32_t z = 0;
            memcpy(e +  0, &v_off,   4);
            memcpy(e +  4, &i_off,   4);
            memcpy(e +  8, &mat_idx, 4);
            memcpy(e + 12, &z,       4);
            write_scene_color(p, m, meshes[m].color);
        }
    }

    if (getenv("NUSD_LOAD_TIMING") || getenv("NUSD_CULL_DIAG"))
        fprintf(stderr, "gpu_metal: RT TLAS rebuilt (reuse BLAS) — %u instances, "
                "%u cached BLAS\n", total_instances, gpu->blas_count);

    gpu->rt_built = 1;
    return 1;
}

int gpu_update_tlas(Gpu* gpu, const float* transforms, const uint8_t* masks,
                    uint32_t nmeshes)
{
    if (!gpu || !gpu->rt_built || !transforms || nmeshes != gpu->scene_nmeshes) return 0;

    uint32_t total_instances = nmeshes + (gpu->curve_blas ? 1u : 0u);

    /* Re-write the instance buffer (Shared — CPU writes are visible to GPU on
     * the next encoder dispatch). */
    MTLAccelerationStructureInstanceDescriptor* inst =
        (MTLAccelerationStructureInstanceDescriptor*)[gpu->instance_buf contents];
    for (uint32_t m = 0; m < nmeshes; m++) {
        vk3x4_to_mtl4x3(&transforms[m * 12], &inst[m].transformationMatrix);
        inst[m].mask = masks ? masks[m] : 0xFFu;
        inst[m].options = MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
        inst[m].intersectionFunctionTableOffset = 0;
        inst[m].accelerationStructureIndex      = gpu->mesh_to_blas[m];
    }
    /* Curve instance preserved — its position past the mesh entries is
     * untouched by the mesh-only transform update path. */

    /* Refit the TLAS in place (much cheaper than rebuild). */
    MTLInstanceAccelerationStructureDescriptor* td =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    td.instanceDescriptorBuffer       = gpu->instance_buf;
    td.instanceDescriptorBufferOffset = 0;
    td.instanceDescriptorStride       = sizeof(MTLAccelerationStructureInstanceDescriptor);
    td.instanceCount                  = total_instances;
    td.instancedAccelerationStructures = gpu->blas_list;
    td.usage                          = MTLAccelerationStructureUsageRefit;

    id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> enc = [cb accelerationStructureCommandEncoder];
    [enc refitAccelerationStructure:gpu->tlas
                         descriptor:td
                        destination:gpu->tlas
                      scratchBuffer:gpu->tlas_scratch
                scratchBufferOffset:0];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return 1;
}

int gpu_update_scene_colors(Gpu* gpu, const float* colors, uint32_t nmeshes)
{
    if (!gpu || !gpu->rt_built || !colors || !gpu->scene_data_buf) return 0;
    if (nmeshes != gpu->scene_nmeshes) return 0;

    uint8_t* p = (uint8_t*)[gpu->scene_data_buf contents];
    for (uint32_t m = 0; m < nmeshes; m++) {
        write_scene_color(p, m, &colors[m * 3]);
    }
    return 1;
}

/* Forward decl — defined in the tiled section below. Used by the inline
 * refit path to re-bind compute state on a fresh encoder. */
static void tiled_bind_compute_resources(Gpu* gpu);

int gpu_update_tlas_inline(Gpu* gpu, const float* transforms, const uint8_t* masks,
                           uint32_t nmeshes)
{
    if (!gpu || !gpu->rt_built || !transforms || nmeshes != gpu->scene_nmeshes) return 0;

    /* Outside a tiled frame: same effect as gpu_update_tlas. The single-camera
     * RT path has its own commit/wait, so a synchronous refit here is correct. */
    if (!gpu->in_frame_rt_tiled) {
        return gpu_update_tlas(gpu, transforms, masks, nmeshes);
    }

    /* Inside a tiled frame: end the compute encoder, refit on an AS encoder
     * (recorded into the same in-flight cmdbuf — no commit / no wait), then
     * reopen the compute encoder and re-bind everything. Encoder-bound state
     * (PSO, buffers, useResource hints) does NOT survive the encoder swap, so
     * tiled_bind_compute_resources must be called again. */
    MTLAccelerationStructureInstanceDescriptor* inst =
        (MTLAccelerationStructureInstanceDescriptor*)[gpu->instance_buf contents];
    for (uint32_t m = 0; m < nmeshes; m++) {
        vk3x4_to_mtl4x3(&transforms[m * 12], &inst[m].transformationMatrix);
        inst[m].mask = masks ? masks[m] : 0xFFu;
        inst[m].options = MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
        inst[m].intersectionFunctionTableOffset = 0;
        inst[m].accelerationStructureIndex      = gpu->mesh_to_blas[m];
    }

    [gpu->current_compute_enc endEncoding];
    gpu->current_compute_enc = nil;

    uint32_t total_instances = nmeshes + (gpu->curve_blas ? 1u : 0u);

    MTLInstanceAccelerationStructureDescriptor* td =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    td.instanceDescriptorBuffer       = gpu->instance_buf;
    td.instanceDescriptorBufferOffset = 0;
    td.instanceDescriptorStride       = sizeof(MTLAccelerationStructureInstanceDescriptor);
    td.instanceCount                  = total_instances;
    td.instancedAccelerationStructures = gpu->blas_list;
    td.usage                          = MTLAccelerationStructureUsageRefit;

    id<MTLAccelerationStructureCommandEncoder> as_enc =
        [gpu->current_cmd accelerationStructureCommandEncoder];
    [as_enc refitAccelerationStructure:gpu->tlas
                            descriptor:td
                           destination:gpu->tlas
                         scratchBuffer:gpu->tlas_scratch
                   scratchBufferOffset:0];
    [as_enc endEncoding];

    gpu->current_compute_enc = [gpu->current_cmd computeCommandEncoder];
    tiled_bind_compute_resources(gpu);
    return 1;
}

int gpu_begin_frame_rt(Gpu* gpu)
{
    if (!gpu || !gpu->rt_built || !gpu->rt_pipeline || !gpu->color_target) return 0;
    if (gpu->in_frame || gpu->in_frame_rt) return 0;

    gpu->current_cmd = [gpu->queue commandBuffer];
    gpu->current_compute_enc = [gpu->current_cmd computeCommandEncoder];
    [gpu->current_compute_enc setComputePipelineState:gpu->rt_pipeline];

    /* Static bindings — the per-frame push constants are set in cmd_trace_rays.
     * Reduced-res orbit: trace into the smaller color_target_lo; gpu_end_frame_rt
     * upscales it into color_target. The kernel reads W/H from this texture's
     * dims, so binding the smaller target is all that's needed. */
    id<MTLTexture> rt_out = (gpu->render_scale < 1.0f && gpu->color_target_lo)
                          ? gpu->color_target_lo : gpu->color_target;
    [gpu->current_compute_enc setTexture:rt_out atIndex:0];
    [gpu->current_compute_enc setAccelerationStructure:gpu->tlas atBufferIndex:0];
    /* buffer(1) = push constants, set in trace_rays via setBytes */
    [gpu->current_compute_enc setBuffer:gpu->scene_data_buf offset:0 atIndex:2];
    /* MeshData[] starts at SCENE_HEADER_BYTES into the same buffer. */
    [gpu->current_compute_enc setBuffer:gpu->scene_data_buf offset:SCENE_HEADER_BYTES atIndex:3];
    /* Curves-only scene: no mesh vertex/index buffers — bind dummy so the
     * kernel param has a valid (but never-read) backing. The kernel only
     * reads verts/idxs in the triangle-hit branch, which is unreachable
     * when there are no triangle BLASes in the TLAS. */
    id<MTLBuffer> verts_b = gpu->rt_vertex_buf ? gpu->rt_vertex_buf : gpu->curve_dummy_buf;
    id<MTLBuffer> idx_b   = gpu->rt_index_buf  ? gpu->rt_index_buf  : gpu->curve_dummy_buf;
    [gpu->current_compute_enc setBuffer:verts_b offset:0 atIndex:4];
    [gpu->current_compute_enc setBuffer:idx_b   offset:0 atIndex:5];

    /* Phase 11.A — IFT for curve_isect, plus curve segment + color buffers
     * for the kernel hit-shading branch. When no curves are loaded the
     * IFT is still bound (its single slot is already populated by
     * ensure_rt_pipeline) and the seg/color buffers fall back to a
     * 1-byte dummy (kernel never reads through them in that case). */
    [gpu->current_compute_enc setIntersectionFunctionTable:gpu->curve_ift atBufferIndex:11];
    id<MTLBuffer> seg_b   = gpu->curve_seg_buf   ? gpu->curve_seg_buf   : gpu->curve_dummy_buf;
    id<MTLBuffer> color_b = gpu->curve_color_buf ? gpu->curve_color_buf : gpu->curve_dummy_buf;
    [gpu->current_compute_enc setBuffer:seg_b   offset:0 atIndex:14];
    [gpu->current_compute_enc setBuffer:color_b offset:0 atIndex:15];
    if (gpu->curve_blas) {
        [gpu->current_compute_enc useResource:gpu->curve_blas usage:MTLResourceUsageRead];
    }

    /* Phase 7 IBL — bind env / irradiance / BRDF-LUT textures. When no
     * environment is loaded, fall back to the color_target itself as a
     * placeholder texture (any RGBA8 / RGBA16F texture satisfies the
     * binding; the kernel won't read it because env_mip_levels == 0). */
    id<MTLTexture> env_t  = gpu->env_texture ? gpu->env_texture : gpu->color_target;
    id<MTLTexture> irr_t  = gpu->irr_texture ? gpu->irr_texture : gpu->color_target;
    id<MTLTexture> brdf_t = gpu->brdf_lut    ? gpu->brdf_lut    : gpu->color_target;
    [gpu->current_compute_enc setTexture:env_t  atIndex:1];
    [gpu->current_compute_enc setTexture:irr_t  atIndex:2];
    [gpu->current_compute_enc setTexture:brdf_t atIndex:3];

    /* Phase 7b — materials SSBO + RT argument-buffer texture table.
     * Always bind so the kernel params have valid backing; the kernel
     * only samples when scene->has_materials > 0 in shade_hit. */
    ensure_material_dummy_tex(gpu);
    ensure_material_sampler(gpu);
    id<MTLBuffer> mat_buf = gpu->material_buf ? gpu->material_buf : gpu->curve_dummy_buf;
    [gpu->current_compute_enc setBuffer:mat_buf offset:0 atIndex:16];
    id<MTLBuffer> light_buf = ensure_light_buffer(gpu);
    [gpu->current_compute_enc setBuffer:light_buf offset:0  atIndex:13];
    [gpu->current_compute_enc setBuffer:light_buf offset:16 atIndex:17];
    bind_material_argument_buffer(gpu, @"rt_render");

    /* The TLAS internally references the BLASes; mark them resident so they
     * stay valid for the dispatch. (The TLAS is also marked implicitly.) */
    for (id<MTLAccelerationStructure> blas in gpu->blas_list) {
        [gpu->current_compute_enc useResource:blas usage:MTLResourceUsageRead];
    }

    gpu->in_frame_rt = 1;
    return 1;
}

void gpu_cmd_trace_rays(Gpu* gpu, const GpuRtPushConstants* pc)
{
    if (!gpu || !gpu->in_frame_rt || !pc) return;

    /* MSL aligns the `PushConstants` struct (which contains a `float4x4`) to
     * 16 bytes. Metal API Validation rejects setBytes with a shorter C-side
     * length, so pad to the next 16-byte boundary. */
    alignas(16) uint8_t padded[176] = {0};
    static_assert(sizeof(*pc) <= sizeof(padded), "GpuRtPushConstants exceeded 176 B");
    memcpy(padded, pc, sizeof(*pc));
    [gpu->current_compute_enc setBytes:padded length:sizeof(padded) atIndex:1];

    /* Reduced-res orbit: dispatch over the smaller render dims (matches the
     * bound color_target_lo); full res otherwise. */
    int scaled = (gpu->render_scale < 1.0f && gpu->color_target_lo);
    NSUInteger w = (NSUInteger)(scaled ? gpu->rw : gpu->width);
    NSUInteger h = (NSUInteger)(scaled ? gpu->rh : gpu->height);

    NSUInteger tew = gpu->rt_pipeline.threadExecutionWidth;
    NSUInteger maxT = gpu->rt_pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger ty = MAX((NSUInteger)1, maxT / tew);
    if (ty > h) ty = h;
    MTLSize tg  = MTLSizeMake(tew, ty, 1);
    MTLSize grid = MTLSizeMake(w, h, 1);
    [gpu->current_compute_enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

void gpu_end_frame_rt(Gpu* gpu)
{
    if (!gpu || !gpu->in_frame_rt) return;
    [gpu->current_compute_enc endEncoding];
    gpu->current_compute_enc = nil;

    /* Reduced-res orbit: upscale the low-res trace into color_target (full res),
     * encoded into this same command buffer, before any readback reads it. */
    if (gpu->render_scale < 1.0f && gpu->color_target_lo) {
        if (!gpu->upscaler)
            gpu->upscaler = [[MPSImageBilinearScale alloc] initWithDevice:gpu->device];
        [gpu->upscaler encodeToCommandBuffer:gpu->current_cmd
                               sourceTexture:gpu->color_target_lo
                          destinationTexture:gpu->color_target];
    }

    /* Async viewport path: fold the color->readback blit into THIS command
     * buffer and commit without waiting. gpu_readback_pixels then returns the
     * previous frame's (already-complete) buffer, so the CPU overlaps the GPU
     * by one frame instead of stalling on commit+readback. */
    if (gpu->async_readback && gpu->readback_buffer && gpu->readback_buffer2) {
        int w = 1 - gpu->rb_last;
        id<MTLBuffer> dst = w ? gpu->readback_buffer2 : gpu->readback_buffer;
        id<MTLTexture> src = (gpu->ssao_active && gpu->post_target)
                           ? gpu->post_target : gpu->color_target;
        id<MTLBlitCommandEncoder> blit = [gpu->current_cmd blitCommandEncoder];
        NSUInteger row = (NSUInteger)gpu->width * gpu_color_bytes_per_pixel(gpu);
        [blit copyFromTexture:src
                  sourceSlice:0 sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake((NSUInteger)gpu->width, (NSUInteger)gpu->height, 1)
                     toBuffer:dst destinationOffset:0
       destinationBytesPerRow:row
     destinationBytesPerImage:row * (NSUInteger)gpu->height];
        [blit endEncoding];
        [gpu->current_cmd commit];   /* no waitUntilCompleted — pipelined */
        gpu->rb_cmd[w] = gpu->current_cmd;
        gpu->rb_valid[w] = 1;
        gpu->rb_last = w;
        gpu->current_cmd = nil;
        gpu->in_frame_rt = 0;
        gpu->has_frame   = 1;
        return;
    }

    [gpu->current_cmd commit];
    [gpu->current_cmd waitUntilCompleted];
    gpu->current_cmd = nil;
    gpu->in_frame_rt = 0;
    gpu->has_frame   = 1;
}

/* ---- Tiled multi-camera RT (Phase 5) ---- */

/* Compile raytrace.metal (already done by ensure_rt_pipeline if single-camera RT
 * was used) and create the rt_render_tiled compute pipeline. Persistent across
 * scene rebuilds. */
static int ensure_tiled_rt_pipeline(Gpu* gpu)
{
    if (gpu->tiled_rt_pipeline && gpu->tiled_curve_ift) return 1;

    if (!gpu->rt_library) {
        char rt_path[1024];
        snprintf(rt_path, sizeof(rt_path), "%s/raytrace.metal", SHADER_DIR);
        gpu->rt_library = compile_msl_file(gpu->device, rt_path);
        if (!gpu->rt_library) {
            fprintf(stderr, "gpu_metal: raytrace.metal compile failed\n");
            return 0;
        }
    }

    if (!gpu->tiled_rt_pipeline) {
        NSError* err = nil;
        gpu->tiled_rt_pipeline = build_rt_compute_pipeline(gpu, @"rt_render_tiled", &err);
        if (!gpu->tiled_rt_pipeline) {
            fprintf(stderr, "gpu_metal: tiled RT pipeline build failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "(unknown)");
            return 0;
        }
    }

    if (!gpu->tiled_curve_ift) {
        gpu->tiled_curve_ift = build_curve_ift(gpu, gpu->tiled_rt_pipeline);
        if (!gpu->tiled_curve_ift) {
            fprintf(stderr, "gpu_metal: curve IFT build failed (tiled_rt_pipeline)\n");
            return 0;
        }
    }
    ensure_curve_dummy_buf(gpu);
    update_curve_ift_buffer(gpu, gpu->tiled_curve_ift);
    return 1;
}

/* Allocate/resize all four tiled output buffers (color + depth + seg + normals)
 * to fit total_w*total_h pixels. Buffers are MTLStorageModeShared so the kernel
 * can write and CPU can read [contents] directly with no blit. */
static int tiled_ensure_output_buffers(Gpu* gpu, uint32_t total_w, uint32_t total_h)
{
    if (gpu->tiled_total_w == total_w && gpu->tiled_total_h == total_h
        && gpu->tiled_color_buf && gpu->tiled_depth_buf
        && gpu->tiled_seg_buf   && gpu->tiled_normals_buf) {
        return 1;
    }

    NSUInteger n = (NSUInteger)total_w * (NSUInteger)total_h;
    NSUInteger color_bytes   = n * 4;          /* uchar4 */
    NSUInteger depth_bytes   = n * 4;          /* float */
    NSUInteger seg_bytes     = n * 4;          /* uint */
    NSUInteger normals_bytes = n * 12;         /* float3 */

    gpu->tiled_color_buf   = [gpu->device newBufferWithLength:color_bytes
                                                      options:MTLResourceStorageModeShared];
    gpu->tiled_depth_buf   = [gpu->device newBufferWithLength:depth_bytes
                                                      options:MTLResourceStorageModeShared];
    gpu->tiled_seg_buf     = [gpu->device newBufferWithLength:seg_bytes
                                                      options:MTLResourceStorageModeShared];
    gpu->tiled_normals_buf = [gpu->device newBufferWithLength:normals_bytes
                                                      options:MTLResourceStorageModeShared];
    if (!gpu->tiled_color_buf || !gpu->tiled_depth_buf
        || !gpu->tiled_seg_buf || !gpu->tiled_normals_buf) {
        fprintf(stderr, "gpu_metal: tiled output buffer alloc failed (%ux%u)\n",
                total_w, total_h);
        return 0;
    }

    gpu->tiled_total_w = total_w;
    gpu->tiled_total_h = total_h;
    return 1;
}

/* Bind every per-frame resource the rt_render_tiled kernel needs onto the
 * current compute encoder, including useResource for each BLAS. Called on a
 * fresh encoder both in begin_frame_tiled_rt and after the inline TLAS refit
 * (Metal binds are per-encoder — switching to an AS encoder and back loses all
 * setBuffer / setAccelerationStructure / useResource state). */
static void tiled_bind_compute_resources(Gpu* gpu)
{
    [gpu->current_compute_enc setComputePipelineState:gpu->tiled_rt_pipeline];
    [gpu->current_compute_enc setAccelerationStructure:gpu->tlas atBufferIndex:0];
    /* buffer(1) — TiledPushConstants — set in cmd_trace_rays_tiled via setBytes */
    [gpu->current_compute_enc setBuffer:gpu->scene_data_buf  offset:0                  atIndex:2];
    [gpu->current_compute_enc setBuffer:gpu->scene_data_buf  offset:SCENE_HEADER_BYTES atIndex:3];
    {
        id<MTLBuffer> verts_b = gpu->rt_vertex_buf ? gpu->rt_vertex_buf : gpu->curve_dummy_buf;
        id<MTLBuffer> idx_b   = gpu->rt_index_buf  ? gpu->rt_index_buf  : gpu->curve_dummy_buf;
        [gpu->current_compute_enc setBuffer:verts_b offset:0 atIndex:4];
        [gpu->current_compute_enc setBuffer:idx_b   offset:0 atIndex:5];
    }
    [gpu->current_compute_enc setBuffer:gpu->tiled_camera_buf  offset:0 atIndex:6];
    [gpu->current_compute_enc setBuffer:gpu->tiled_color_buf   offset:0 atIndex:7];
    [gpu->current_compute_enc setBuffer:gpu->tiled_depth_buf   offset:0 atIndex:8];
    [gpu->current_compute_enc setBuffer:gpu->tiled_seg_buf     offset:0 atIndex:9];
    [gpu->current_compute_enc setBuffer:gpu->tiled_normals_buf offset:0 atIndex:10];

    /* Phase 11.A — same curve plumbing as the single-camera path. */
    [gpu->current_compute_enc setIntersectionFunctionTable:gpu->tiled_curve_ift atBufferIndex:11];
    id<MTLBuffer> seg_b   = gpu->curve_seg_buf   ? gpu->curve_seg_buf   : gpu->curve_dummy_buf;
    id<MTLBuffer> color_b = gpu->curve_color_buf ? gpu->curve_color_buf : gpu->curve_dummy_buf;
    [gpu->current_compute_enc setBuffer:seg_b   offset:0 atIndex:14];
    [gpu->current_compute_enc setBuffer:color_b offset:0 atIndex:15];
    if (gpu->curve_blas) {
        [gpu->current_compute_enc useResource:gpu->curve_blas usage:MTLResourceUsageRead];
    }

    /* Phase 7 IBL — env / irradiance / BRDF-LUT bindings. Fall back to
     * any non-null texture (color_target works as a placeholder) when
     * no env is loaded. */
    id<MTLTexture> env_t  = gpu->env_texture ? gpu->env_texture : gpu->color_target;
    id<MTLTexture> irr_t  = gpu->irr_texture ? gpu->irr_texture : gpu->color_target;
    id<MTLTexture> brdf_t = gpu->brdf_lut    ? gpu->brdf_lut    : gpu->color_target;
    [gpu->current_compute_enc setTexture:env_t  atIndex:1];
    [gpu->current_compute_enc setTexture:irr_t  atIndex:2];
    [gpu->current_compute_enc setTexture:brdf_t atIndex:3];

    /* Phase 7b — materials SSBO + RT argument-buffer texture table.
     * Same setup as gpu_begin_frame_rt; the tiled kernel reads the same
     * MaterialParams binding through the same kernel signature. */
    ensure_material_dummy_tex(gpu);
    ensure_material_sampler(gpu);
    id<MTLBuffer> mat_buf = gpu->material_buf ? gpu->material_buf : gpu->curve_dummy_buf;
    [gpu->current_compute_enc setBuffer:mat_buf offset:0 atIndex:16];
    id<MTLBuffer> light_buf = ensure_light_buffer(gpu);
    [gpu->current_compute_enc setBuffer:light_buf offset:0  atIndex:13];
    [gpu->current_compute_enc setBuffer:light_buf offset:16 atIndex:17];
    bind_material_argument_buffer(gpu, @"rt_render_tiled");

    /* The TLAS internally references the BLASes; mark them resident so they
     * stay valid for the dispatch. (Same caveat as the single-camera path.) */
    for (id<MTLAccelerationStructure> blas in gpu->blas_list) {
        [gpu->current_compute_enc useResource:blas usage:MTLResourceUsageRead];
    }
}

int gpu_tiled_init(Gpu* gpu, uint32_t total_w, uint32_t total_h, int num_cameras)
{
    if (!gpu || total_w == 0 || total_h == 0 || num_cameras <= 0) return 0;

    if (!tiled_ensure_output_buffers(gpu, total_w, total_h)) return 0;

    /* Camera SSBO: num_cameras * 32 floats = num_cameras * 128 bytes. */
    if (gpu->tiled_camera_capacity < num_cameras || !gpu->tiled_camera_buf) {
        NSUInteger cam_bytes = (NSUInteger)num_cameras * 32u * sizeof(float);
        gpu->tiled_camera_buf = [gpu->device newBufferWithLength:cam_bytes
                                                         options:MTLResourceStorageModeShared];
        if (!gpu->tiled_camera_buf) {
            fprintf(stderr, "gpu_metal: tiled camera SSBO alloc failed (%d cameras)\n",
                    num_cameras);
            return 0;
        }
        gpu->tiled_camera_capacity = num_cameras;
    }

    gpu->tiled_num_cameras = num_cameras;
    return 1;
}

int gpu_tiled_upload_cameras(Gpu* gpu, const float* data, int num_cameras)
{
    if (!gpu || !data || !gpu->tiled_camera_buf || num_cameras <= 0) return 0;
    if (num_cameras > gpu->tiled_camera_capacity) return 0;
    memcpy([gpu->tiled_camera_buf contents], data,
           (size_t)num_cameras * 32u * sizeof(float));
    return 1;
}

int gpu_build_tiled_rt_pipeline(Gpu* gpu,
                                const uint32_t* /*rgen_spv*/, uint32_t /*rgen_size*/,
                                const uint32_t* /*miss_spv*/, uint32_t /*miss_size*/,
                                const uint32_t* /*chit_spv*/, uint32_t /*chit_size*/)
{
    if (!gpu || !gpu->rt_available) return 0;
    return ensure_tiled_rt_pipeline(gpu);
}

int gpu_begin_frame_tiled_rt(Gpu* gpu)
{
    if (!gpu || !gpu->rt_built || !gpu->tiled_rt_pipeline) return 0;
    if (!gpu->tiled_color_buf || !gpu->tiled_camera_buf) return 0;
    if (gpu->in_frame || gpu->in_frame_rt || gpu->in_frame_rt_tiled) return 0;

    /* Single-buffered MVP — wait for the previous frame's GPU work to finish
     * before reusing the same output buffers. waitUntilCompleted on a
     * complete cmdbuf is effectively free. */
    if (gpu->tiled_inflight_cmd) {
        [gpu->tiled_inflight_cmd waitUntilCompleted];
        gpu->tiled_inflight_cmd = nil;
    }

    gpu->current_cmd = [gpu->queue commandBuffer];
    gpu->current_compute_enc = [gpu->current_cmd computeCommandEncoder];
    tiled_bind_compute_resources(gpu);
    gpu->in_frame_rt_tiled = 1;
    return 1;
}

void gpu_cmd_trace_rays_tiled(Gpu* gpu, const GpuRtTiledPushConstants* pc)
{
    if (!gpu || !gpu->in_frame_rt_tiled || !pc) return;

    [gpu->current_compute_enc setBytes:pc length:sizeof(*pc) atIndex:1];

    NSUInteger total_w = (NSUInteger)pc->tile_w * (NSUInteger)pc->num_cols;
    NSUInteger num_rows = ((NSUInteger)pc->num_cameras + (NSUInteger)pc->num_cols - 1)
                          / (NSUInteger)pc->num_cols;
    NSUInteger total_h = (NSUInteger)pc->tile_h * num_rows;

    NSUInteger tew  = gpu->tiled_rt_pipeline.threadExecutionWidth;
    NSUInteger maxT = gpu->tiled_rt_pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger ty = MAX((NSUInteger)1, maxT / tew);
    if (ty > total_h) ty = total_h;
    MTLSize tg   = MTLSizeMake(tew, ty, 1);
    MTLSize grid = MTLSizeMake(total_w, total_h, 1);
    [gpu->current_compute_enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

void gpu_end_frame_tiled_rt(Gpu* gpu)
{
    if (!gpu || !gpu->in_frame_rt_tiled) return;
    [gpu->current_compute_enc endEncoding];
    gpu->current_compute_enc = nil;

    /* Commit but DO NOT wait — readback paths (gpu_map_tiled_staging /
     * gpu_wait_tiled_complete) handle the wait, which lets callers overlap
     * CPU work with GPU. Single-buffer MVP: only one in-flight cmdbuf. */
    gpu->tiled_inflight_cmd = gpu->current_cmd;
    gpu->tiled_last_slot    = 0;
    [gpu->current_cmd commit];
    gpu->current_cmd = nil;
    gpu->in_frame_rt_tiled = 0;
    gpu->has_frame = 1;
}

void gpu_abort_frame_tiled_rt(Gpu* gpu)
{
    if (!gpu || !gpu->in_frame_rt_tiled) return;
    [gpu->current_compute_enc endEncoding];
    gpu->current_compute_enc = nil;
    /* Drop the cmdbuf without committing — Metal's deferred-encoding model
     * means the recorded work simply never executes. */
    gpu->current_cmd = nil;
    gpu->in_frame_rt_tiled = 0;
}

/* Wait for the most recent tiled cmdbuf to finish. Idempotent. */
static void tiled_wait_inflight(Gpu* gpu)
{
    if (gpu && gpu->tiled_inflight_cmd) {
        [gpu->tiled_inflight_cmd waitUntilCompleted];
        gpu->tiled_inflight_cmd = nil;
    }
}

int gpu_readback_tiled_pixels(Gpu* gpu, uint8_t* out_rgba8, uint32_t total_w, uint32_t total_h)
{
    if (!gpu || !out_rgba8 || !gpu->tiled_color_buf) return 0;
    if (total_w != gpu->tiled_total_w || total_h != gpu->tiled_total_h) return 0;
    tiled_wait_inflight(gpu);
    memcpy(out_rgba8, [gpu->tiled_color_buf contents],
           (size_t)total_w * (size_t)total_h * 4);
    return 1;
}

int gpu_readback_tiled_depth(Gpu* gpu, float* out_depth, uint32_t total_w, uint32_t total_h)
{
    if (!gpu || !out_depth || !gpu->tiled_depth_buf) return 0;
    if (total_w != gpu->tiled_total_w || total_h != gpu->tiled_total_h) return 0;
    tiled_wait_inflight(gpu);
    memcpy(out_depth, [gpu->tiled_depth_buf contents],
           (size_t)total_w * (size_t)total_h * sizeof(float));
    return 1;
}

int gpu_readback_tiled_segmentation(Gpu* gpu, uint32_t* out_ids, uint32_t total_w, uint32_t total_h)
{
    if (!gpu || !out_ids || !gpu->tiled_seg_buf) return 0;
    if (total_w != gpu->tiled_total_w || total_h != gpu->tiled_total_h) return 0;
    tiled_wait_inflight(gpu);
    memcpy(out_ids, [gpu->tiled_seg_buf contents],
           (size_t)total_w * (size_t)total_h * sizeof(uint32_t));
    return 1;
}

int gpu_readback_tiled_normals(Gpu* gpu, float* out_normals, uint32_t total_w, uint32_t total_h)
{
    if (!gpu || !out_normals || !gpu->tiled_normals_buf) return 0;
    if (total_w != gpu->tiled_total_w || total_h != gpu->tiled_total_h) return 0;
    tiled_wait_inflight(gpu);
    memcpy(out_normals, [gpu->tiled_normals_buf contents],
           (size_t)total_w * (size_t)total_h * 3u * sizeof(float));
    return 1;
}

const uint8_t* gpu_map_tiled_staging(Gpu* gpu, uint32_t total_w, uint32_t total_h)
{
    if (!gpu || !gpu->tiled_color_buf) return nullptr;
    if (total_w != gpu->tiled_total_w || total_h != gpu->tiled_total_h) return nullptr;
    tiled_wait_inflight(gpu);
    return (const uint8_t*)[gpu->tiled_color_buf contents];
}

const uint8_t* gpu_map_tiled_staging_slot(Gpu* gpu, uint32_t total_w,
                                          uint32_t total_h, int slot)
{
    /* Single-buffered MVP: only slot 0 is valid. The double-buffered API
     * shape is preserved so renderer.c's overlap path still type-checks. */
    if (slot != 0) return nullptr;
    return gpu_map_tiled_staging(gpu, total_w, total_h);
}

int gpu_get_last_tiled_slot(Gpu* gpu) { return gpu ? gpu->tiled_last_slot : -1; }

int gpu_wait_tiled_complete(Gpu* gpu)
{
    if (!gpu) return 0;
    tiled_wait_inflight(gpu);
    return 1;
}

int gpu_wait_previous_tiled_complete(Gpu* /*gpu*/) { return 0; }
int gpu_get_interop_prev_idx(Gpu* /*gpu*/)         { return -1; }

/* ---- CUDA Interop (permanently stubbed on macOS) ---- */

int      gpu_interop_available(Gpu* /*gpu*/)               { return 0; }
int      gpu_export_tiled_image_fd(Gpu* /*gpu*/, int /*idx*/)  { return -1; }
int      gpu_get_interop_read_idx(Gpu* /*gpu*/)            { return 0; }
int      gpu_export_timeline_semaphore_fd(Gpu* /*gpu*/)    { return -1; }
uint64_t gpu_get_tiled_image_alloc_size(Gpu* /*gpu*/)      { return 0; }
uint64_t gpu_get_interop_timeline_value(Gpu* /*gpu*/)      { return 0; }
void     gpu_set_skip_staging_copy(Gpu* gpu, int skip)     { if (gpu) gpu->skip_staging = skip; }
void     gpu_set_direct_write(Gpu* gpu, int enable)        { if (gpu) gpu->direct_write_enabled = enable; }
int      gpu_is_direct_write(Gpu* gpu)                     { return gpu ? gpu->direct_write_enabled : 0; }

/* ---- DLSS (permanently stubbed on macOS) ---- */

int  gpu_dlss_available(Gpu* /*gpu*/) { return 0; }
int  gpu_dlss_init(Gpu* /*gpu*/, int /*quality_mode*/) { return 0; }
void gpu_dlss_shutdown(Gpu* /*gpu*/) {}
void gpu_dlss_set_quality(Gpu* /*gpu*/, int /*quality_mode*/) {}

void gpu_get_render_extent(Gpu* gpu, uint32_t* w, uint32_t* h)
{
    if (!gpu) { if (w) *w = 0; if (h) *h = 0; return; }
    if (w) *w = (uint32_t)gpu->width;
    if (h) *h = (uint32_t)gpu->height;
}

int  gpu_begin_frame_dlss(Gpu* /*gpu*/) { return 0; }
void gpu_end_frame_dlss(Gpu* /*gpu*/, float, float, float, int) {}

int  gpu_begin_frame_rt_dlss(Gpu* /*gpu*/) { return 0; }
void gpu_cmd_trace_rays_dlss(Gpu* /*gpu*/, const GpuRtPushConstantsDlss* /*pc*/) {}
void gpu_end_frame_rt_dlss(Gpu* /*gpu*/, float, float, float, int) {}

int gpu_build_rt_scene_dlss(Gpu* /*gpu*/,
                            const GpuRtMeshDesc* /*meshes*/, uint32_t /*nmeshes*/,
                            const uint32_t* /*rgen_spv*/, uint32_t /*rgen_size*/,
                            const uint32_t* /*miss_spv*/, uint32_t /*miss_size*/,
                            const uint32_t* /*chit_spv*/, uint32_t /*chit_size*/) { return 0; }

GpuPipeline gpu_create_dlss_pipeline(Gpu* gpu, const GpuPipelineDesc* desc)
{ return gpu_create_pipeline(gpu, desc); }

GpuPipeline gpu_create_dlss_shadow_pipeline(Gpu* gpu, const GpuPipelineDesc* desc)
{ return gpu_create_shadow_pipeline(gpu, desc); }

/* ---- Text overlay ---- */

int  gpu_overlay_init(Gpu* /*gpu*/) { return 1; }
void gpu_overlay_shutdown(Gpu* /*gpu*/) {}
void gpu_overlay_text(Gpu* /*gpu*/, float, float, float,
                      float, float, float, float, const char* /*text*/) {}
void gpu_overlay_rect(Gpu* /*gpu*/, float, float, float, float,
                      float, float, float, float) {}
void gpu_overlay_flush(Gpu* /*gpu*/) {}

/* ---- Raycast compute (LiDAR/radar) ---- */

int gpu_build_raycast_pipeline(Gpu* /*gpu*/, const uint32_t* /*comp_spv*/, uint32_t /*comp_size*/)
{ return 0; }

void gpu_destroy_raycast_pipeline(Gpu* /*gpu*/) {}

int gpu_cast_rays(Gpu* /*gpu*/,
                  const float* /*origins*/, const float* /*directions*/,
                  int num_rays, float /*max_distance*/,
                  float* out_distances, float* out_normals,
                  float* out_hit_positions)
{
    if (out_distances)     for (int i = 0; i < num_rays; i++) out_distances[i] = -1.0f;
    if (out_normals)       memset(out_normals, 0, (size_t)num_rays * 3 * sizeof(float));
    if (out_hit_positions) memset(out_hit_positions, 0, (size_t)num_rays * 3 * sizeof(float));
    return 1;
}

int gpu_cast_rays_async(Gpu* /*gpu*/, const float*, const float*, int, float) { return 1; }

int gpu_cast_rays_wait(Gpu* /*gpu*/, float* out_distances, float* out_normals,
                       float* out_hit_positions)
{
    /* No state — just zero outputs if provided. */
    if (out_distances)     out_distances[0] = -1.0f;
    if (out_normals)       memset(out_normals, 0, 3 * sizeof(float));
    if (out_hit_positions) memset(out_hit_positions, 0, 3 * sizeof(float));
    return 1;
}

int gpu_raycast_get_interop_fds(Gpu* /*gpu*/, int /*num_rays*/,
                                int* out_input_fd, uint64_t* out_input_size,
                                int* out_output_fd, uint64_t* out_output_size,
                                uint32_t* out_max_rays)
{
    if (out_input_fd)    *out_input_fd  = -1;
    if (out_input_size)  *out_input_size  = 0;
    if (out_output_fd)   *out_output_fd = -1;
    if (out_output_size) *out_output_size = 0;
    if (out_max_rays)    *out_max_rays = 0;
    return 0;
}

int gpu_cast_rays_gpu(Gpu* /*gpu*/, int /*num_rays*/, float /*max_distance*/) { return 0; }
int gpu_cast_rays_wait_fence(Gpu* /*gpu*/) { return 1; }

uint64_t gpu_get_allocated_memory(Gpu* gpu) { return gpu ? gpu->allocated_bytes : 0; }

uint64_t gpu_get_heap_size(Gpu* gpu)
{
    if (!gpu) return 0;
    /* recommendedMaxWorkingSetSize is the best proxy on Apple Silicon. */
    if (@available(macOS 10.12, *)) {
        return (uint64_t)[gpu->device recommendedMaxWorkingSetSize];
    }
    return 0;
}
