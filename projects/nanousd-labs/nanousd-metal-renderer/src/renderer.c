// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * renderer.c — Implementation of nusd_renderer.h public API.
 *
 * Metal port: fully headless via offscreen MTLTexture; no window required.
 * GLFW is dropped entirely (the Vulkan version used a hidden window for
 * VkSurfaceKHR creation, which Metal doesn't need).
 */

#include "nusd_renderer.h"
#include "gpu.h"
#include "scene.h"
#include "gs_scene.h"
#include "camera.h"
#include "material.h"

#include <nanousd/nanousdapi.h>
#include <meshoptimizer.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>  /* close() for fd cleanup on error */

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* ---- Internal mesh record ---- */

typedef struct {
    int          active;        /* 0 = slot is free */
    int          visible;
    float        transform[16]; /* row-major 4x4 */
    float        color[3];
    int          nvertices;
    int          nindices;
    uint32_t     vertex_offset; /* offset in merged buffer */
    uint32_t     index_offset;
    uint32_t     meshlet_offset;
    uint32_t     meshlet_count;
    uint32_t     meshlet_index_offset;
    uint32_t     meshlet_index_count;
    int          prototype_idx; /* for instancing */
    int          material_index; /* -1 = none */
    unsigned char env_mask;     /* TLAS instance mask, 0 hides from RT */
    char*        name;
} RendererMesh;

typedef struct {
    uint32_t     prototype_idx;
    uint32_t     index_offset;
    uint32_t     index_count;
    float        center[3];
    float        radius;
    float        cone_axis[3];
    float        cone_cutoff;
} RendererMeshlet;

typedef struct {
    uint64_t hash;
    int      hits;
    int      slot;
} GeoHashEntry;

typedef struct {
    GeoHashEntry* entries;
    uint32_t      capacity;
    uint32_t      count;
    uint64_t      total_calls;
    uint64_t      dup_calls;
    uint64_t      total_bytes;
} GeoHashCensus;

/* ---- Phase 11.A: Curve registry ----
 * Flat segment + AABB lists across all BasisCurves prims in the loaded
 * scene. A single AABB BLAS is built over all of these (Strategy A from
 * RENDERER_BIG_PLAN.md Phase 12.1 — one flat BLAS over all segments,
 * best traversal quality at scale). Per-segment color carries the
 * curve's display_color for now; Phase 12.2 adds material_id indirection. */
typedef struct {
    SceneCurveSegment* segments;     /* float[nseg * 8]  — 32 B/segment   */
    SceneCurveAabb*    aabbs;        /* float[nseg * 6]  — VkAabbPositionsKHR */
    float*             colors;       /* float[nseg * 3]  — display_color  */
    int                nseg;
} RendererCurves;

/* Compact PointInstancer batch in renderer space. proto_renderer_idx is
 * the index into r->meshes (the RendererMesh holding the prototype's
 * vertex/index buffer offsets + BLAS). transforms borrows the slab from
 * the Scene's arena — both lifetimes overlap (scene lives as long as the
 * renderer keeps usd_scene non-NULL). */
typedef struct {
    int                            proto_renderer_idx;
    uint32_t                       count;
    const SceneInstanceTransform*  transforms;
    int                            source_prim_idx;
    int                            material_or_binding_id;
    /* Prototype bounding sphere (object space), captured at attach from the
     * scene proto mesh bounds — RendererMesh carries no bounds, so the cull
     * needs it here. */
    float                          proto_center[3];
    float                          proto_radius;
    /* Cached world-space AABB over all of this batch's instance spheres,
     * computed once (cull_valid) for hierarchical draw-time frustum culling:
     * a whole batch outside the frustum is skipped without touching its
     * (up to ~570k) instances. */
    float                          cull_min[3];
    float                          cull_max[3];
    int                            cull_valid;
} RendererPiBatch;

/* ---- Internal state ---- */

struct NuRenderer {
    /* Configuration */
    int          width;
    int          height;
    int          enable_rt;
    int          enable_materials;

    /* Current evaluation time (USD time code) — propagated to scene
     * loading so xformOp:translate.timeSamples / etc. resolve at the
     * right frame. NaN means "use default time". Set via
     * nu_set_current_time(); read by scene.c::compute_world_xform via
     * scene_set_load_time(). Cf. Vulkan port commit 18d341d. */
    double       current_time;

    /* GPU state */
    Gpu*         gpu;

    /* Mesh registry */
    RendererMesh* meshes;
    int          nmeshes;
    int          mesh_capacity;

    /* Curve registry (Phase 11.A) — populated by nu_load_usd, uploaded
     * to GPU at nu_build_accel time. */
    RendererCurves curves;
    int            curves_dirty;

    /* Gaussian splat scene (RT-only, independent of mesh/curve scene). */
    NuGsScene*     gs;
    int            gs_dirty;

    /* GPU buffers */
    GpuBuffer    vertex_buffer;
    GpuBuffer    index_buffer;
    GpuBuffer    meshlet_index_buffer;
    GpuBuffer    instance_buffer;
    uint32_t     instance_buffer_capacity;
    uint32_t*    raster_batch_counts;
    uint32_t*    raster_batch_firsts;
    uint32_t*    raster_batch_cursors;
    uint32_t*    raster_batch_prototypes;
    uint32_t     raster_batch_capacity;
    uint32_t     raster_batch_visible_count;
    uint32_t     raster_batch_draw_count;
    int          raster_instance_cache_valid;
    float        raster_instance_cache_vp[16];
    uint32_t     total_vertices;
    uint32_t     total_indices;

    /* Compact PointInstancer batches (Section 8B). pi_transforms is the
     * renderer-owned copy of the scene's compact transform slab — copying
     * is required because scene_free runs early when materials are
     * disabled (see nu_attach_scene), which would dangle borrowed
     * pointers. RendererPiBatch.transforms references entries in
     * pi_transforms_owned. raster path issues one instanced draw per
     * batch; RT path materialises a transient flat GpuRtMeshDesc[] at
     * TLAS-build time and frees it after the build. */
    RendererPiBatch*        pi_batches;
    int                     npi_batches;
    SceneInstanceTransform* pi_transforms_owned;
    uint64_t                npi_transforms_total;

    /* Merged buffer data (CPU side, rebuilt on scene change).
     * Material path: 12 floats/vertex: pos(3) normal(3) pad(3) uv(2) matID.
     * Geometry path: 9 floats/vertex: pos(3) normal(3) uv(2) matID.
     * The geometry layout keeps raster shader inputs defined while avoiding
     * unused material padding on DSX geometry-only loads. */
    float*       cpu_vertices;
    uint32_t*    cpu_indices;
    uint32_t     vertex_stride_floats;
    uint32_t     cpu_vertex_capacity;
    uint32_t     cpu_index_capacity;
    GeoHashCensus geo_census;

    /* Optional meshoptimizer meshlet data. The meshlet index stream is a
     * conventional uint32 index buffer flattened from meshlet micro-indices so
     * the current Metal raster pipeline can consume it without mesh shaders. */
    RendererMeshlet* meshlets;
    uint32_t*    cpu_meshlet_indices;
    uint32_t     meshlet_count;
    uint32_t     meshlet_capacity;
    uint32_t     total_meshlet_indices;
    uint32_t     cpu_meshlet_index_capacity;
    int          meshlets_enabled;
    int          meshlet_raster_enabled;
    int          meshlet_raster_replaces_index_buffer;
    int          preserve_instance_draws;
    int          release_cpu_staging_after_upload;
    int          loaded_from_geometry_cache;
    int          cpu_staging_released;

    /* Pipelines */
    GpuPipeline  raster_pipeline;
    GpuPipeline  shadow_pipeline;

    /* Camera */
    Camera       camera;
    int          camera_valid;
    NuExposureDesc exposure_desc;

    /* Scene dirty flag */
    int          scene_dirty;    /* 1 = need to rebuild GPU buffers + accel */
    int          accel_dirty;    /* 1 = need to rebuild BLAS/TLAS */
    int          rt_blas_valid;  /* 1 = cached BLAS reusable (only camera/cull changed) */
    int          tlas_dirty;     /* 1 = transforms changed, rebuild TLAS only */
    int          colors_dirty;   /* 1 = display colors changed, update scene SSBO */

    /* Scene loaded from USD (optional) */
    Scene*       usd_scene;
    int          has_cached_bounds;
    float        cached_bounds_min[3];
    float        cached_bounds_max[3];

    /* Tier 3 lazy load. When NUSD_LAZY_MESH=1, nu_attach_scene parks the
     * metadata-only Scene here; nu_extract_deferred reruns eager extraction
     * on the same stage and swaps in the real scene. */
    int          lazy_pending;

    /* Error string */
    char         last_error[512];

    /* Readback buffer */
    uint8_t*     readback_pixels; /* RGBA8, width * height * 4 */
    int          has_frame;       /* 1 = pixels valid after render */

    /* Tiled RT state */
    int          tiled_pipeline_built; /* 1 = tiled RT pipeline ready */
    int          fast_mode;            /* 1 = skip shadow rays for RL sensors */
    int          per_env_layout;      /* 1 = SSBO writes in [env,H,W] not tiled */
    int          tiled_srgb;           /* 1 = apply sRGB gamma in raygen shader */
    int          depth_enabled;       /* 1 = write depth to SSBO during RT */
    int          segmentation_enabled; /* 1 = write instance IDs to SSBO during RT */
    int          normals_enabled;      /* 1 = write surface normals to SSBO during RT */

    /* Raycast compute pipeline state */
    int          raycast_pipeline_built; /* 1 = raycast compute pipeline ready */

    /* Async render+fetch state (nu_render_async/nu_fetch_async).
     * Tracks a 2-deep history of tiled-staging slot indices so that fetch on
     * frame N can pick up the slot written by frame N-1's render. The Metal
     * tiled backend is currently single-buffered (gpu_metal.mm "Phase 5b" note
     * in the gpu struct) so both slots resolve to 0 today; the wrapper still
     * honors the API contract (first call zero-fills, valid pixels by 2nd). */
    int          async_curr_slot;   /* slot written by the most recent render_async */
    int          async_prev_slot;   /* slot written by the render before that      */
    int          async_count;       /* clamps at 2: # of render_async calls so far */

    NuPhaseTimings timings;
};

/* ---- Helpers ---- */

static void set_error(NuRenderer* r, const char* msg)
{
    snprintf(r->last_error, sizeof(r->last_error), "%s", msg);
}

static double wall_seconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

static float elapsed_ms(double start)
{
    return (float)((wall_seconds() - start) * 1000.0);
}

static void write_fixed_string(char* dst, size_t dst_size, const char* src)
{
    if (!dst || dst_size == 0) return;
    snprintf(dst, dst_size, "%s", src ? src : "");
}

static uint64_t static_backend_capabilities(void)
{
    uint64_t caps =
        NU_CAP_RENDER_RASTER |
        NU_CAP_AOV_COLOR |
        NU_CAP_LOAD_USD_FILE |
        NU_CAP_LOAD_USD_HANDLE |
        NU_CAP_LOAD_USD_HANDLE_WITH_DIR |
        NU_CAP_SCENE_BOUNDS |
        NU_CAP_MESH_PATHS |
        NU_CAP_CURVES |
        NU_CAP_CAMERAS |
        NU_CAP_DOME_LIGHT |
        NU_CAP_SET_TRANSFORMS |
        NU_CAP_SET_COLORS |
        NU_CAP_SET_VISIBILITY |
        NU_CAP_SET_CURRENT_TIME |
        NU_CAP_TIMINGS |
        NU_CAP_GPU_MEMORY |
        NU_CAP_MESHLETS |
        NU_CAP_GEOMETRY_CACHE;

    if (materials_backend_available()) {
        caps |= NU_CAP_MATERIALS |
                NU_CAP_TEXTURES |
                NU_CAP_MATERIALX;
    }

    return caps;
}

NuResult nu_get_backend_info(NuRenderer* r, NuBackendInfo* out_info)
{
    if (!out_info) return NU_ERROR;

    memset(out_info, 0, sizeof(*out_info));
    out_info->version = NU_BACKEND_INFO_VERSION;
    out_info->struct_size = (uint32_t)sizeof(*out_info);
    out_info->capabilities = static_backend_capabilities();
    write_fixed_string(out_info->backend_name,
                       sizeof(out_info->backend_name), "Metal");
    write_fixed_string(out_info->backend_version,
                       sizeof(out_info->backend_version), "Metal 3");
    write_fixed_string(out_info->renderer_name,
                       sizeof(out_info->renderer_name),
                       "nanousd-metal-renderer");

    if (!r) return NU_OK;

    if (r->enable_rt && gpu_rt_available(r->gpu)) {
        out_info->capabilities |= NU_CAP_RENDER_RAYTRACE |
                                  NU_CAP_RENDER_TILED_RAYTRACE |
                                  NU_CAP_AOV_DEPTH |
                                  NU_CAP_AOV_NORMAL |
                                  NU_CAP_AOV_PRIM_ID;
        if (r->shadow_pipeline) {
            out_info->capabilities |= NU_CAP_RENDER_SHADOW;
        }
    }

    return NU_OK;
}

static void identity_matrix(float m[16])
{
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static int geo_census_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("NUSD_HASH_CENSUS");
        cached = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return cached;
}

static int geo_dedup_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        /* Default ON: content-hash dedup collapses byte-identical meshes (the
         * warehouse's repeated racks/pallets/boxes) onto one shared geometry
         * range + BLAS, regardless of material — material is applied per-instance
         * downstream, so per-instance transform/color/material are preserved and
         * the render is bit-identical to the per-mesh path (verified md5-stable
         * on the warehouse for both RT and raster). On the full warehouse this
         * drops RT BLAS 3469 -> 138 and GPU memory 32.5 -> 3.6 MB.
         * NUSD_HASH_DEDUP=0 forces the per-mesh path for A/B measurement. */
        const char* e = getenv("NUSD_HASH_DEDUP");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached;
}

static void geo_census_reset(GeoHashCensus* census)
{
    if (!census) return;
    free(census->entries);
    census->entries = NULL;
    census->capacity = 0;
    census->count = 0;
    census->total_calls = 0;
    census->dup_calls = 0;
    census->total_bytes = 0;
}

static void geo_census_init(GeoHashCensus* census, uint32_t initial_slots)
{
    if (!census || census->entries) return;
    uint32_t cap = 1024;
    while (cap < initial_slots * 2u) cap <<= 1;
    census->entries = (GeoHashEntry*)calloc(cap, sizeof(GeoHashEntry));
    census->capacity = census->entries ? cap : 0;
}

static void geo_census_grow(GeoHashCensus* census)
{
    if (!census) return;
    if (census->count * 2u < census->capacity) return;
    uint32_t new_cap = census->capacity ? census->capacity * 2u : 1024u;
    GeoHashEntry* ne = (GeoHashEntry*)calloc(new_cap, sizeof(GeoHashEntry));
    if (!ne) return;
    for (uint32_t i = 0; i < census->capacity; i++) {
        if (!census->entries[i].hash) continue;
        uint32_t mask = new_cap - 1u;
        uint32_t s = (uint32_t)census->entries[i].hash & mask;
        while (ne[s].hash) s = (s + 1u) & mask;
        ne[s] = census->entries[i];
    }
    free(census->entries);
    census->entries = ne;
    census->capacity = new_cap;
}

static inline uint64_t fnv1a_64_update(uint64_t h, const void* data, size_t n)
{
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t geo_hash_compute(const NuMeshDesc* desc, size_t* out_bytes)
{
    if (!desc || desc->nvertices <= 0 || desc->nindices <= 0) {
        if (out_bytes) *out_bytes = 0;
        return 0;
    }

    uint64_t h = 1469598103934665603ULL;
    int nv = desc->nvertices;
    int ni = desc->nindices;
    h = fnv1a_64_update(h, &nv, sizeof(nv));
    h = fnv1a_64_update(h, &ni, sizeof(ni));
    uint8_t present = (desc->normals ? 1u : 0u) | (desc->texcoords ? 2u : 0u);
    h = fnv1a_64_update(h, &present, sizeof(present));

    size_t pos_bytes = (size_t)nv * 3u * sizeof(float);
    size_t idx_bytes = (size_t)ni * sizeof(uint32_t);
    size_t nrm_bytes = desc->normals ? (size_t)nv * 3u * sizeof(float) : 0u;
    size_t uv_bytes = desc->texcoords ? (size_t)nv * 2u * sizeof(float) : 0u;
    if (desc->positions) h = fnv1a_64_update(h, desc->positions, pos_bytes);
    if (desc->indices) h = fnv1a_64_update(h, desc->indices, idx_bytes);
    if (desc->normals) h = fnv1a_64_update(h, desc->normals, nrm_bytes);
    if (desc->texcoords) h = fnv1a_64_update(h, desc->texcoords, uv_bytes);
    if (h == 0) h = 0xdeadbeefcafebabeULL;

    if (out_bytes) *out_bytes = pos_bytes + idx_bytes + nrm_bytes + uv_bytes;
    return h;
}

static int geo_dedup_lookup(NuRenderer* r, const NuMeshDesc* desc)
{
    if (!r || !r->geo_census.entries) return -1;
    GeoHashCensus* census = &r->geo_census;
    size_t bytes = 0;
    uint64_t h = geo_hash_compute(desc, &bytes);
    if (!h) return -1;

    uint32_t mask = census->capacity - 1u;
    uint32_t s = (uint32_t)h & mask;
    while (census->entries[s].hash) {
        if (census->entries[s].hash == h && census->entries[s].slot >= 0) {
            census->entries[s].hits++;
            census->dup_calls++;
            census->total_calls++;
            census->total_bytes += bytes;
            return census->entries[s].slot;
        }
        s = (s + 1u) & mask;
    }
    return -1;
}

static void geo_dedup_insert(NuRenderer* r, const NuMeshDesc* desc, int slot)
{
    if (!r) return;
    if (!geo_dedup_enabled() && !geo_census_enabled()) return;
    GeoHashCensus* census = &r->geo_census;
    if (!census->entries) geo_census_init(census, 4096);
    if (!census->entries) return;

    size_t bytes = 0;
    uint64_t h = geo_hash_compute(desc, &bytes);
    if (!h) return;
    census->total_calls++;
    census->total_bytes += bytes;
    geo_census_grow(census);

    uint32_t mask = census->capacity - 1u;
    uint32_t s = (uint32_t)h & mask;
    while (census->entries[s].hash) {
        if (census->entries[s].hash == h) {
            census->entries[s].hits++;
            census->dup_calls++;
            if (census->entries[s].slot < 0 && slot >= 0)
                census->entries[s].slot = slot;
            return;
        }
        s = (s + 1u) & mask;
    }

    census->entries[s].hash = h;
    census->entries[s].hits = 1;
    census->entries[s].slot = slot;
    census->count++;
}

static void geo_census_record(NuRenderer* r, const NuMeshDesc* desc)
{
    if (!geo_census_enabled() || !desc) return;
    geo_dedup_insert(r, desc, -1);
}

static void geo_census_dump_and_reset(NuRenderer* r)
{
    if (!r) return;
    GeoHashCensus* census = &r->geo_census;
    if ((!geo_census_enabled() && !geo_dedup_enabled()) ||
        census->total_calls == 0) {
        return;
    }

    /* Dedup is on by default, so reset the census every load — but only print
     * the histogram when explicitly profiling (NUSD_HASH_CENSUS), otherwise the
     * default path would spam stderr on every scene load. */
    if (!geo_census_enabled()) {
        geo_census_reset(census);
        return;
    }

    uint64_t reuse_1 = 0, reuse_2 = 0, reuse_3_10 = 0, reuse_11p = 0;
    int max_hits = 0;
    for (uint32_t i = 0; i < census->capacity; i++) {
        int h = census->entries[i].hits;
        if (h == 0) continue;
        if (h > max_hits) max_hits = h;
        if (h == 1) reuse_1++;
        else if (h == 2) reuse_2++;
        else if (h <= 10) reuse_3_10++;
        else reuse_11p++;
    }
    double dup_pct = 100.0 * (double)census->dup_calls /
                     (double)census->total_calls;
    fprintf(stderr,
            "geo_hash_census: %llu nu_add_mesh calls -> %u unique geometries "
            "(%llu dup hits, %.1f%% dup rate, %.1f GB hashed)\n"
            "geo_hash_census: reuse histogram - singletons=%llu, 2x=%llu, "
            "3-10x=%llu, 11+=%llu, max_hits=%d\n",
            (unsigned long long)census->total_calls,
            census->count,
            (unsigned long long)census->dup_calls,
            dup_pct,
            (double)census->total_bytes / (1024.0 * 1024.0 * 1024.0),
            (unsigned long long)reuse_1,
            (unsigned long long)reuse_2,
            (unsigned long long)reuse_3_10,
            (unsigned long long)reuse_11p,
            max_hits);
    geo_census_reset(census);
}

/* Convert a USD row-vector row-major 4x4 (translation at src[12..14]) into
 * VkTransformMatrixKHR's 3x4 column-vector row-major layout
 * (translation at dst[3], dst[7], dst[11]). This is a transpose of the
 * 4x4 with the last column dropped. */
static void usd4x4_to_vk3x4(const float src[16], float dst[12])
{
    dst[0]  = src[0];  dst[1]  = src[4];  dst[2]  = src[8];   dst[3]  = src[12];
    dst[4]  = src[1];  dst[5]  = src[5];  dst[6]  = src[9];   dst[7]  = src[13];
    dst[8]  = src[2];  dst[9]  = src[6];  dst[10] = src[10];  dst[11] = src[14];
}

/* Convert SceneInstanceTransform (packed 12-float column-major affine —
 * [c0.xyz | c1.xyz | c2.xyz | c3.xyz], translation in slab.m[9..11]) into
 * VkTransformMatrixKHR's 3x4 row-major layout. Used to feed compact PI
 * batches into gpu_build_rt_scene without an intermediate 4x4. */
static void pi_xform12_to_vk3x4(const float src[12], float dst[12])
{
    dst[0]  = src[0]; dst[1]  = src[3]; dst[2]  = src[6];  dst[3]  = src[9];
    dst[4]  = src[1]; dst[5]  = src[4]; dst[6]  = src[7];  dst[7]  = src[10];
    dst[8]  = src[2]; dst[9]  = src[5]; dst[10] = src[8];  dst[11] = src[11];
}

static void transpose4x4(const float src[16], float dst[16])
{
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            dst[row * 4 + col] = src[col * 4 + row];
        }
    }
}

static void compute_view_inverse(const Camera* cam, float vi[16])
{
    float v[16];
    camera_get_view(cam, v);

    memset(vi, 0, sizeof(float) * 16);
    vi[0] = v[0]; vi[1] = v[4]; vi[2]  = v[8];
    vi[4] = v[1]; vi[5] = v[5]; vi[6]  = v[9];
    vi[8] = v[2]; vi[9] = v[6]; vi[10] = v[10];
    vi[3]  = -(vi[0]*v[3]  + vi[1]*v[7]  + vi[2]*v[11]);
    vi[7]  = -(vi[4]*v[3]  + vi[5]*v[7]  + vi[6]*v[11]);
    vi[11] = -(vi[8]*v[3]  + vi[9]*v[7]  + vi[10]*v[11]);
    vi[15] = 1.0f;
}

static void compute_proj_inverse(const Camera* cam, float pi[16])
{
    float p[16];
    camera_get_proj(cam, p);
    memset(pi, 0, sizeof(float) * 16);
    pi[0]  = 1.0f / p[0];
    pi[5]  = 1.0f / p[5];
    pi[3]  = p[2] / p[0];
    pi[7]  = p[6] / p[5];
    pi[11] = 1.0f / p[14];
    pi[14] = -1.0f;
    pi[15] = p[10] / p[14];
}

/* Pack the renderer's current camera into the 32-float [view_inv | proj_inv]
 * layout that the tiled raygen shader's camera SSBO expects. */
static void compute_camera_inverses(const Camera* cam, float vp_inv[32])
{
    compute_view_inverse(cam, vp_inv);
    compute_proj_inverse(cam, vp_inv + 16);
}

static int ensure_mesh_capacity(NuRenderer* r, int needed)
{
    if (needed <= r->mesh_capacity) return 1;
    int new_cap = r->mesh_capacity ? r->mesh_capacity * 2 : 64;
    while (new_cap < needed) new_cap *= 2;
    RendererMesh* new_meshes = (RendererMesh*)realloc(r->meshes,
        (size_t)new_cap * sizeof(RendererMesh));
    if (!new_meshes) return 0;
    memset(&new_meshes[r->mesh_capacity], 0,
           (size_t)(new_cap - r->mesh_capacity) * sizeof(RendererMesh));
    r->meshes = new_meshes;
    r->mesh_capacity = new_cap;
    return 1;
}

static int append_mesh_slot(NuRenderer* r)
{
    int slot = r->nmeshes;
    if (!ensure_mesh_capacity(r, slot + 1)) return -1;
    r->nmeshes = slot + 1;
    return slot;
}

static int reserve_cpu_geometry(NuRenderer* r, uint64_t vertices, uint64_t indices)
{
    if (!r) return 0;
    if (vertices > UINT32_MAX || indices > UINT32_MAX) return 0;
    uint32_t stride = r->vertex_stride_floats ? r->vertex_stride_floats : 12u;

    if ((uint32_t)vertices > r->cpu_vertex_capacity) {
        float* new_vertices = (float*)realloc(
            r->cpu_vertices, (size_t)vertices * stride * sizeof(float));
        if (!new_vertices && vertices > 0) return 0;
        r->cpu_vertices = new_vertices;
        r->cpu_vertex_capacity = (uint32_t)vertices;
    }

    if ((uint32_t)indices > r->cpu_index_capacity) {
        uint32_t* new_indices = (uint32_t*)realloc(
            r->cpu_indices, (size_t)indices * sizeof(uint32_t));
        if (!new_indices && indices > 0) return 0;
        r->cpu_indices = new_indices;
        r->cpu_index_capacity = (uint32_t)indices;
    }

    return 1;
}

static int reserve_cpu_vertices_only(NuRenderer* r, uint64_t vertices)
{
    if (!r) return 0;
    if (vertices > UINT32_MAX) return 0;
    uint32_t stride = r->vertex_stride_floats ? r->vertex_stride_floats : 12u;
    if ((uint32_t)vertices <= r->cpu_vertex_capacity) return 1;

    float* new_vertices = (float*)realloc(
        r->cpu_vertices, (size_t)vertices * stride * sizeof(float));
    if (!new_vertices && vertices > 0) return 0;
    r->cpu_vertices = new_vertices;
    r->cpu_vertex_capacity = (uint32_t)vertices;
    return 1;
}

static int reserve_cpu_indices_only(NuRenderer* r, uint64_t indices)
{
    if (!r) return 0;
    if (indices > UINT32_MAX) return 0;
    if ((uint32_t)indices <= r->cpu_index_capacity) return 1;

    uint32_t* new_indices = (uint32_t*)realloc(
        r->cpu_indices, (size_t)indices * sizeof(uint32_t));
    if (!new_indices && indices > 0) return 0;
    r->cpu_indices = new_indices;
    r->cpu_index_capacity = (uint32_t)indices;
    return 1;
}

static int reserve_meshlet_data(NuRenderer* r, uint64_t meshlets, uint64_t indices)
{
    if (!r) return 0;
    if (meshlets > UINT32_MAX || indices > UINT32_MAX) return 0;

    if ((uint32_t)meshlets > r->meshlet_capacity) {
        uint32_t new_cap = r->meshlet_capacity ? r->meshlet_capacity * 2u : 256u;
        while (new_cap < (uint32_t)meshlets) {
            if (new_cap > UINT32_MAX / 2u) {
                new_cap = (uint32_t)meshlets;
                break;
            }
            new_cap *= 2u;
        }
        RendererMeshlet* new_meshlets = (RendererMeshlet*)realloc(
            r->meshlets, (size_t)new_cap * sizeof(RendererMeshlet));
        if (!new_meshlets && meshlets > 0) return 0;
        memset(new_meshlets + r->meshlet_capacity, 0,
               (size_t)(new_cap - r->meshlet_capacity) * sizeof(RendererMeshlet));
        r->meshlets = new_meshlets;
        r->meshlet_capacity = new_cap;
    }

    if ((uint32_t)indices > r->cpu_meshlet_index_capacity) {
        uint32_t new_cap = r->cpu_meshlet_index_capacity ? r->cpu_meshlet_index_capacity * 2u : 4096u;
        while (new_cap < (uint32_t)indices) {
            if (new_cap > UINT32_MAX / 2u) {
                new_cap = (uint32_t)indices;
                break;
            }
            new_cap *= 2u;
        }
        uint32_t* new_indices = (uint32_t*)realloc(
            r->cpu_meshlet_indices, (size_t)new_cap * sizeof(uint32_t));
        if (!new_indices && indices > 0) return 0;
        r->cpu_meshlet_indices = new_indices;
        r->cpu_meshlet_index_capacity = new_cap;
    }

    return 1;
}

static int env_flag_enabled(const char* name)
{
    const char* e = getenv(name);
    return e && e[0] && e[0] != '0';
}

static int env_flag_requested(const char* name)
{
    const char* e = getenv(name);
    if (!e || !e[0]) return 0;
    if (e[0] == '0' || !strcmp(e, "false") || !strcmp(e, "off") ||
        !strcmp(e, "no")) {
        return 0;
    }
    return 1;
}

static int no_cull_all_geometry_requested(void)
{
    return env_flag_requested("NUSD_NO_CULL_ALL_GEOMETRY") ||
           env_flag_requested("NUSD_ALL_GEOMETRY_NO_CULL");
}

static int env_int_limit(const char* name, int fallback)
{
    const char* e = getenv(name);
    if (!e || !e[0]) return fallback;
    char* end = NULL;
    long v = strtol(e, &end, 10);
    if (end == e || v < 0 || v > INT_MAX) return fallback;
    return (int)v;
}

static uint64_t env_u64_limit(const char* name, uint64_t fallback)
{
    const char* e = getenv(name);
    if (!e || !e[0]) return fallback;
    char* end = NULL;
    unsigned long long v = strtoull(e, &end, 10);
    if (end == e) return fallback;
    return (uint64_t)v;
}

static int path_ends_with_ci(const char* path, const char* suffix)
{
    if (!path || !suffix) return 0;
    size_t plen = strlen(path);
    size_t slen = strlen(suffix);
    if (slen > plen) return 0;
    path += plen - slen;
    for (size_t i = 0; i < slen; i++) {
        unsigned char a = (unsigned char)path[i];
        unsigned char b = (unsigned char)suffix[i];
        if (tolower(a) != tolower(b)) return 0;
    }
    return 1;
}

static void copy_loaded_environment_path(char* dst, size_t dst_size,
                                         const char* path)
{
    if (!dst || dst_size == 0) return;
    if (!path) path = "";
    snprintf(dst, dst_size, "%s", path);
}

static int gpu_load_environment_with_fallbacks(Gpu* gpu, const char* path,
                                               char* loaded_path,
                                               size_t loaded_path_size)
{
    if (!gpu || !path || !path[0]) return 0;
    if (gpu_load_environment(gpu, path)) {
        copy_loaded_environment_path(loaded_path, loaded_path_size, path);
        return 1;
    }

    if (!path_ends_with_ci(path, ".exr")) return 0;

    size_t base_len = strlen(path) - 4;
    const char* suffixes[] = {
        "VIS.png",
        ".png",
        "VIS.jpg",
        ".jpg",
    };
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        char candidate[PATH_MAX];
        size_t suffix_len = strlen(suffixes[i]);
        if (base_len + suffix_len >= sizeof(candidate))
            continue;
        memcpy(candidate, path, base_len);
        memcpy(candidate + base_len, suffixes[i], suffix_len + 1);
        if (access(candidate, R_OK) != 0)
            continue;
        if (gpu_load_environment(gpu, candidate)) {
            copy_loaded_environment_path(loaded_path, loaded_path_size,
                                         candidate);
            return 1;
        }
    }

    return 0;
}

static void release_cpu_staging(NuRenderer* r)
{
    if (!r || r->cpu_staging_released) return;

    free(r->cpu_vertices); r->cpu_vertices = NULL;
    free(r->cpu_indices); r->cpu_indices = NULL;
    free(r->cpu_meshlet_indices); r->cpu_meshlet_indices = NULL;
    free(r->meshlets); r->meshlets = NULL;
    r->cpu_vertex_capacity = 0;
    r->cpu_index_capacity = 0;
    r->cpu_meshlet_index_capacity = 0;
    r->meshlet_capacity = 0;
    r->cpu_staging_released = 1;
}

static int can_release_cpu_staging_after_upload(const NuRenderer* r)
{
    if (!r || !r->release_cpu_staging_after_upload) return 0;
    if (r->enable_rt) return 0;
    if (!r->vertex_buffer) return 0;
    if (!r->index_buffer && !r->meshlet_index_buffer) return 0;
    return 1;
}

static void release_curve_cpu_staging(NuRenderer* r)
{
    if (!r || r->curves_dirty) return;
    if (!r->curves.segments && !r->curves.aabbs && !r->curves.colors) return;

    int nseg = r->curves.nseg;
    free(r->curves.segments); r->curves.segments = NULL;
    free(r->curves.aabbs);    r->curves.aabbs    = NULL;
    free(r->curves.colors);   r->curves.colors   = NULL;
    fprintf(stderr,
            "nusd_renderer: released CPU curve staging for %d segments "
            "(NUSD_KEEP_CPU_CURVES=1 keeps it)\n",
            nseg);
}

static int can_release_curve_cpu_staging_after_upload(const NuRenderer* r)
{
    if (!r || r->curves_dirty || r->curves.nseg <= 0) return 0;
    if (env_flag_requested("NUSD_KEEP_CPU_CURVES")) return 0;
    /* Under the culled-proxy RT path the curve BLAS is rebuilt from the CPU
     * arrays every camera (rebuild_curve_blas_culled), so they must persist. */
    if (env_flag_requested("NUSD_NO_CULL_ALL_GEOMETRY") ||
        env_flag_requested("NUSD_RT_CULL")) return 0;
    return 1;
}

static void invalidate_raster_instance_cache(NuRenderer* r)
{
    if (!r) return;
    r->raster_instance_cache_valid = 0;
    r->raster_batch_visible_count = 0;
    r->raster_batch_draw_count = 0;
}

static unsigned char mesh_rt_mask(const RendererMesh* m)
{
    if (!m || !m->active || !m->visible) return 0u;
    return m->env_mask ? m->env_mask : 0xFFu;
}

static void fill_rt_instance_masks(const NuRenderer* r, uint8_t* masks)
{
    if (!r || !masks) return;
    for (int m = 0; m < r->nmeshes; m++)
        masks[m] = mesh_rt_mask(&r->meshes[m]);
}

static int ensure_raster_batch_scratch(NuRenderer* r, uint32_t count)
{
    if (!r) return 0;
    if (count == 0) count = 1;
    if (count <= r->raster_batch_capacity &&
        r->raster_batch_counts &&
        r->raster_batch_firsts &&
        r->raster_batch_cursors &&
        r->raster_batch_prototypes) {
        return 1;
    }

    uint32_t new_cap = r->raster_batch_capacity ? r->raster_batch_capacity : 1024u;
    while (new_cap < count) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = count;
            break;
        }
        new_cap *= 2u;
    }

    uint32_t* counts = (uint32_t*)realloc(r->raster_batch_counts,
                                          (size_t)new_cap * sizeof(uint32_t));
    if (!counts) return 0;
    r->raster_batch_counts = counts;

    uint32_t* firsts = (uint32_t*)realloc(r->raster_batch_firsts,
                                          (size_t)new_cap * sizeof(uint32_t));
    if (!firsts) return 0;
    r->raster_batch_firsts = firsts;

    uint32_t* cursors = (uint32_t*)realloc(r->raster_batch_cursors,
                                           (size_t)new_cap * sizeof(uint32_t));
    if (!cursors) return 0;
    r->raster_batch_cursors = cursors;

    uint32_t* prototypes = (uint32_t*)realloc(r->raster_batch_prototypes,
                                              (size_t)new_cap * sizeof(uint32_t));
    if (!prototypes) return 0;
    r->raster_batch_prototypes = prototypes;

    r->raster_batch_capacity = new_cap;
    invalidate_raster_instance_cache(r);
    return 1;
}

static int ensure_instance_buffer(NuRenderer* r, uint32_t count)
{
    if (!r) return 0;
    if (count == 0) count = 1;
    if (count <= r->instance_buffer_capacity && r->instance_buffer) return 1;

    uint32_t new_cap = r->instance_buffer_capacity ? r->instance_buffer_capacity : 1024u;
    while (new_cap < count) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = count;
            break;
        }
        new_cap *= 2u;
    }
    /* For large counts (Moana no-cull: ~30M PI placements) the geometric
     * doubling wastes up to ~1 GiB on a 160 B/instance buffer (33.5M vs 30.2M
     * ≈ 0.8 GiB). Above 4M instances, allocate exactly — the buffer is already
     * multi-GiB and re-grow churn is not the bottleneck there. */
    if (count > (4u << 20) && new_cap > count)
        new_cap = count;

    if (r->instance_buffer) {
        gpu_destroy_buffer(r->gpu, r->instance_buffer);
        r->instance_buffer = NULL;
        r->instance_buffer_capacity = 0;
        invalidate_raster_instance_cache(r);
    }

    GpuBufferDesc desc;
    desc.usage = GPU_BUFFER_VERTEX;
    desc.size = (uint64_t)new_cap * sizeof(GpuRasterInstanceData);
    desc.data = NULL;
    r->instance_buffer = gpu_create_buffer(r->gpu, &desc);
    if (!r->instance_buffer) return 0;
    r->instance_buffer_capacity = new_cap;
    return 1;
}

static void fill_raster_instance_data(const RendererMesh* m,
                                      const float vp[16],
                                      GpuRasterInstanceData* out)
{
    float model_rm[16];
    memcpy(model_rm, m->transform, sizeof(model_rm));

    float model[16];
    for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
            model[row * 4 + col] = model_rm[col * 4 + row];

    float mvp[16];
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++)
                sum += vp[row * 4 + k] * model[k * 4 + col];
            mvp[row * 4 + col] = sum;
        }
    }

    memset(out, 0, sizeof(*out));
    memcpy(out->mvp, mvp, sizeof(out->mvp));
    /* Shader world-space normal/position math expects the USD row-vector
     * model, while MVP uses the transposed column-vector convention. */
    memcpy(out->model, model_rm, sizeof(out->model));
    out->color[0] = m->color[0];
    out->color[1] = m->color[1];
    out->color[2] = m->color[2];
    out->color[3] = 1.0f;
    out->material_index = m->material_index;
}

static int renderer_mesh_root_proto(const NuRenderer* r, int mesh_index)
{
    const RendererMesh* m = &r->meshes[mesh_index];
    int proto = m->prototype_idx;
    if (proto < 0 || proto >= r->nmeshes || !r->meshes[proto].active)
        return mesh_index;
    return proto;
}

static int meshlet_ranges_cover_active_meshes(const NuRenderer* r)
{
    if (!r || r->enable_rt) return 0;
    if (r->total_meshlet_indices == 0) return 0;

    for (int i = 0; i < r->nmeshes; i++) {
        const RendererMesh* m = &r->meshes[i];
        if (!m->active || m->nindices == 0) continue;
        int proto = renderer_mesh_root_proto(r, i);
        const RendererMesh* p = &r->meshes[proto];
        if (p->meshlet_index_count == 0) return 0;
    }
    return 1;
}

static int meshlet_raster_covers_active_meshes(const NuRenderer* r)
{
    return (r && r->meshlet_raster_enabled) ?
        meshlet_ranges_cover_active_meshes(r) : 0;
}

static void extract_frustum_planes(const float vp[16], float planes[6][4]);
static int aabb_in_frustum(const float planes[6][4],
                           const float mn[3], const float mx[3]);

/* World-space bounding sphere of one compact PI instance (12-float
 * column-major affine `m`) applied to the prototype sphere (center `c`,
 * radius `rad`). Returns center in wc[3] and scaled radius. */
static inline float pi_instance_world_sphere(const float* m, const float c[3],
                                             float rad, float wc[3])
{
    wc[0] = m[0]*c[0] + m[3]*c[1] + m[6]*c[2] + m[9];
    wc[1] = m[1]*c[0] + m[4]*c[1] + m[7]*c[2] + m[10];
    wc[2] = m[2]*c[0] + m[5]*c[1] + m[8]*c[2] + m[11];
    float s0 = sqrtf(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
    float s1 = sqrtf(m[3]*m[3] + m[4]*m[4] + m[5]*m[5]);
    float s2 = sqrtf(m[6]*m[6] + m[7]*m[7] + m[8]*m[8]);
    float sc = s0 > s1 ? (s0 > s2 ? s0 : s2) : (s1 > s2 ? s1 : s2);
    return rad * sc;
}

/* Sphere fully outside the frustum? (plane convention: pl·c + pl[3] < 0 = behind) */
static inline int sphere_outside_frustum(const float planes[6][4],
                                         const float c[3], float r)
{
    for (int i = 0; i < 6; i++) {
        const float* pl = planes[i];
        if (pl[0]*c[0] + pl[1]*c[1] + pl[2]*c[2] + pl[3] < -r) return 1;
    }
    return 0;
}

/* Compute (once) the batch's world AABB over all its instance spheres, for
 * hierarchical pre-cull. O(count) but only the first time. */
static void pi_batch_ensure_bounds(NuRenderer* r, RendererPiBatch* pb)
{
    (void)r;
    if (pb->cull_valid) return;
    pb->cull_valid = 1;
    pb->cull_min[0] = pb->cull_min[1] = pb->cull_min[2] = FLT_MAX;
    pb->cull_max[0] = pb->cull_max[1] = pb->cull_max[2] = -FLT_MAX;
    for (uint32_t k = 0; k < pb->count; k++) {
        float wc[3];
        float wr = pi_instance_world_sphere(pb->transforms[k].m,
                                            pb->proto_center, pb->proto_radius, wc);
        for (int a = 0; a < 3; a++) {
            if (wc[a] - wr < pb->cull_min[a]) pb->cull_min[a] = wc[a] - wr;
            if (wc[a] + wr > pb->cull_max[a]) pb->cull_max[a] = wc[a] + wr;
        }
    }
}

static int render_raster_instanced_batches(NuRenderer* r,
                                           const float vp[16],
                                           const float eye[3],
                                           int* bound_index_stream)
{
    if (!r || r->nmeshes <= 0) return 1;

    uint32_t n = (uint32_t)r->nmeshes;
    if (!ensure_raster_batch_scratch(r, n)) {
        set_error(r, "Failed to allocate instanced draw batches");
        return 0;
    }

    uint32_t* counts = r->raster_batch_counts;
    uint32_t* firsts = r->raster_batch_firsts;
    uint32_t* cursors = r->raster_batch_cursors;
    uint32_t* prototypes = r->raster_batch_prototypes;
    GpuRasterInstanceData* instances =
        (GpuRasterInstanceData*)gpu_buffer_contents(r->instance_buffer);
    int cache_hit = (r->raster_instance_cache_valid &&
                     memcmp(r->raster_instance_cache_vp, vp, 16 * sizeof(float)) == 0 &&
                     instances != NULL);
    uint32_t visible_count = r->raster_batch_visible_count;
    uint32_t draw_count = r->raster_batch_draw_count;

    if (!cache_hit) {
        memset(counts, 0, (size_t)n * sizeof(uint32_t));
        memset(cursors, 0, (size_t)n * sizeof(uint32_t));

        /* Draw-time hierarchical frustum cull for compact PI batches: a whole
         * batch outside the frustum is skipped via its cached AABB; survivors
         * are tested per-instance (bounding sphere vs frustum). Optional
         * screen-angular cull (env NUSD_CULL_MIN_ANGULAR = radius/distance
         * threshold) drops sub-pixel instances to bound wide views. This is
         * what lets the 202M-instance Moana fit a bounded instance buffer
         * instead of a single 32 GiB allocation. */
        float planes[6][4];
        extract_frustum_planes(vp, planes);
        const char* ang_env = getenv("NUSD_CULL_MIN_ANGULAR");
        float min_angular = ang_env ? (float)atof(ang_env) : 0.0f;
        long long pi_culled_batches = 0, pi_culled_instances = 0;

        visible_count = 0;
        for (int i = 0; i < r->nmeshes; i++) {
            RendererMesh* m = &r->meshes[i];
            if (!m->active || !m->visible || m->nindices == 0) continue;
            int proto = renderer_mesh_root_proto(r, i);
            counts[proto]++;
            visible_count++;
        }

        /* Compact PointInstancer batches (Section 8B): each batch contributes
         * `count` extra raster instances to its prototype's draw call. The
         * prototype RendererMesh itself stays invisible (is_proto_only sets
         * rm->visible=0 at attach time); the PI draws are the only thing
         * that lights it up. */
        for (int b = 0; b < r->npi_batches; b++) {
            RendererPiBatch* pb = &r->pi_batches[b];
            int proto = pb->proto_renderer_idx;
            if (proto < 0 || proto >= r->nmeshes) continue;
            RendererMesh* pm = &r->meshes[proto];
            if (!pm->active || pm->nindices == 0) continue;
            pi_batch_ensure_bounds(r, pb);
            if (!aabb_in_frustum(planes, pb->cull_min, pb->cull_max)) {
                pi_culled_batches++;
                pi_culled_instances += (long long)pb->count;
                continue;
            }
            uint32_t vis = 0;
            for (uint32_t k = 0; k < pb->count; k++) {
                float wc[3];
                float wr = pi_instance_world_sphere(pb->transforms[k].m,
                                                    pb->proto_center, pb->proto_radius, wc);
                if (sphere_outside_frustum(planes, wc, wr)) continue;
                if (min_angular > 0.0f) {
                    float dx = wc[0]-eye[0], dy = wc[1]-eye[1], dz = wc[2]-eye[2];
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (dist > 1e-4f && wr/dist < min_angular) continue;
                }
                vis++;
            }
            counts[proto] += vis;
            visible_count += vis;
            pi_culled_instances += (long long)(pb->count - vis);
        }
        if (getenv("NUSD_CULL_DIAG")) {
            fprintf(stderr,
                    "nusd_renderer: draw-cull — visible=%u  culled_batches=%lld  "
                    "culled_instances=%lld  (min_angular=%.5f)\n",
                    visible_count, pi_culled_batches, pi_culled_instances,
                    min_angular);
        }

        if (visible_count == 0) {
            invalidate_raster_instance_cache(r);
            return 1;
        }

        uint32_t running = 0;
        draw_count = 0;
        for (int i = 0; i < r->nmeshes; i++) {
            if (!counts[i]) continue;
            firsts[i] = running;
            prototypes[draw_count++] = (uint32_t)i;
            running += counts[i];
        }

        if (!ensure_instance_buffer(r, visible_count)) {
            set_error(r, "Failed to create raster instance buffer");
            return 0;
        }
        instances = (GpuRasterInstanceData*)gpu_buffer_contents(r->instance_buffer);
        if (!instances) {
            invalidate_raster_instance_cache(r);
            set_error(r, "Raster instance buffer is not CPU-visible");
            return 0;
        }

        for (int i = 0; i < r->nmeshes; i++) {
            RendererMesh* m = &r->meshes[i];
            if (!m->active || !m->visible || m->nindices == 0) continue;
            int proto = renderer_mesh_root_proto(r, i);
            uint32_t dst = firsts[proto] + cursors[proto]++;
            fill_raster_instance_data(m, vp, &instances[dst]);
        }

        /* Compact PI fill: synthesise a transient RendererMesh wrapper per
         * instance so fill_raster_instance_data can reuse the same shader-
         * facing layout (MVP + model + color). This avoids forking the GPU
         * push-constant / instance-data contract for compact PI; the only
         * difference vs. the per-clone path is where the transform lives. */
        for (int b = 0; b < r->npi_batches; b++) {
            const RendererPiBatch* pb = &r->pi_batches[b];
            int proto = pb->proto_renderer_idx;
            if (proto < 0 || proto >= r->nmeshes) continue;
            RendererMesh* pm = &r->meshes[proto];
            if (!pm->active || pm->nindices == 0) continue;
            /* Same hierarchical cull as the count pass — must match exactly so
             * each proto fills precisely counts[proto] slots. */
            if (pb->cull_valid &&
                !aabb_in_frustum(planes, pb->cull_min, pb->cull_max)) continue;
            for (uint32_t k = 0; k < pb->count; k++) {
                float wc[3];
                float wr = pi_instance_world_sphere(pb->transforms[k].m,
                                                    pb->proto_center, pb->proto_radius, wc);
                if (sphere_outside_frustum(planes, wc, wr)) continue;
                if (min_angular > 0.0f) {
                    float dx = wc[0]-eye[0], dy = wc[1]-eye[1], dz = wc[2]-eye[2];
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (dist > 1e-4f && wr/dist < min_angular) continue;
                }
                RendererMesh tmp = *pm;
                /* Unpack 12-float column-major affine into the 4x4 USD row
                 * matrix the shader expects (column-major in m->transform[16]
                 * — rows 3,7,11 are the dropped (0,0,0); row 15 = 1). */
                const float* sm = pb->transforms[k].m;
                tmp.transform[ 0] = sm[ 0]; tmp.transform[ 1] = sm[ 1]; tmp.transform[ 2] = sm[ 2]; tmp.transform[ 3] = 0.0f;
                tmp.transform[ 4] = sm[ 3]; tmp.transform[ 5] = sm[ 4]; tmp.transform[ 6] = sm[ 5]; tmp.transform[ 7] = 0.0f;
                tmp.transform[ 8] = sm[ 6]; tmp.transform[ 9] = sm[ 7]; tmp.transform[10] = sm[ 8]; tmp.transform[11] = 0.0f;
                tmp.transform[12] = sm[ 9]; tmp.transform[13] = sm[10]; tmp.transform[14] = sm[11]; tmp.transform[15] = 1.0f;
                uint32_t dst = firsts[proto] + cursors[proto]++;
                fill_raster_instance_data(&tmp, vp, &instances[dst]);
            }
        }

        r->raster_batch_visible_count = visible_count;
        r->raster_batch_draw_count = draw_count;
        memcpy(r->raster_instance_cache_vp, vp, 16 * sizeof(float));
        r->raster_instance_cache_valid = 1;
    } else if (visible_count == 0) {
        return 1;
    }

    gpu_cmd_bind_instance_buffer(r->gpu, r->instance_buffer);

    GpuMeshPushConstants pc;
    memset(&pc, 0, sizeof(pc));
    memcpy(pc.eye_pos, eye, 3 * sizeof(float));
    pc.eye_pos[3] = 0.0f;
    pc.material_index = -1;
    pc.instanced = 1u;
    pc.instance_base = 0u;
    gpu_cmd_push_constants(r->gpu, &pc, sizeof(pc));

    for (uint32_t bi = 0; bi < draw_count; bi++) {
        int i = (int)prototypes[bi];
        uint32_t instance_count = counts[i];
        if (!instance_count) continue;

        RendererMesh* m = &r->meshes[i];
        if (!m->active || m->nindices == 0) continue;

        int want_meshlet_stream = (r->meshlet_raster_enabled &&
                                   r->meshlet_index_buffer &&
                                   m->meshlet_index_count > 0);
        GpuBuffer desired_index_buffer = want_meshlet_stream
            ? r->meshlet_index_buffer : r->index_buffer;
        if (!desired_index_buffer) continue;
        if (want_meshlet_stream != *bound_index_stream) {
            gpu_cmd_bind_index_buffer(r->gpu, desired_index_buffer);
            *bound_index_stream = want_meshlet_stream;
        }
        gpu_cmd_draw_indexed_instanced(r->gpu,
            want_meshlet_stream ? m->meshlet_index_count : (uint32_t)m->nindices,
            want_meshlet_stream ? m->meshlet_index_offset : m->index_offset,
            (int32_t)m->vertex_offset,
            instance_count,
            firsts[i]);
    }

    return 1;
}

#define NUSD_MESHLET_MAX_VERTICES 64u
#define NUSD_MESHLET_MAX_TRIANGLES 124u

static int build_meshlets_for_prototype(NuRenderer* r,
                                        RendererMesh* m,
                                        const uint32_t* indices,
                                        size_t index_count,
                                        const float* vertices,
                                        size_t vertex_count,
                                        size_t vertex_stride)
{
    if (!r || !m || !indices || !vertices) return 0;
    if (!r->meshlets_enabled) return 1;
    if (index_count == 0 || vertex_count == 0) return 1;

    size_t max_meshlets = meshopt_buildMeshletsBound(
        index_count, NUSD_MESHLET_MAX_VERTICES, NUSD_MESHLET_MAX_TRIANGLES);
    if (max_meshlets == 0) return 1;

    struct meshopt_Meshlet* tmp_meshlets = (struct meshopt_Meshlet*)malloc(
        max_meshlets * sizeof(struct meshopt_Meshlet));
    unsigned int* tmp_vertices = (unsigned int*)malloc(
        max_meshlets * NUSD_MESHLET_MAX_VERTICES * sizeof(unsigned int));
    unsigned char* tmp_triangles = (unsigned char*)malloc(
        max_meshlets * NUSD_MESHLET_MAX_TRIANGLES * 3u * sizeof(unsigned char));
    if (!tmp_meshlets || !tmp_vertices || !tmp_triangles) {
        free(tmp_meshlets);
        free(tmp_vertices);
        free(tmp_triangles);
        return 0;
    }

    size_t meshlet_count = meshopt_buildMeshlets(
        tmp_meshlets,
        tmp_vertices,
        tmp_triangles,
        indices,
        index_count,
        vertices,
        vertex_count,
        vertex_stride,
        NUSD_MESHLET_MAX_VERTICES,
        NUSD_MESHLET_MAX_TRIANGLES,
        0.0f);

    if (meshlet_count > UINT32_MAX ||
        meshlet_count > UINT32_MAX - r->meshlet_count ||
        index_count > UINT32_MAX - r->total_meshlet_indices) {
        free(tmp_meshlets);
        free(tmp_vertices);
        free(tmp_triangles);
        return 0;
    }

    uint32_t first_meshlet = r->meshlet_count;
    uint32_t first_index = r->total_meshlet_indices;
    if (!reserve_meshlet_data(r,
            (uint64_t)r->meshlet_count + (uint64_t)meshlet_count,
            (uint64_t)r->total_meshlet_indices + (uint64_t)index_count)) {
        free(tmp_meshlets);
        free(tmp_vertices);
        free(tmp_triangles);
        return 0;
    }

    m->meshlet_offset = first_meshlet;
    m->meshlet_count = (uint32_t)meshlet_count;
    m->meshlet_index_offset = first_index;

    for (size_t mi = 0; mi < meshlet_count; mi++) {
        const struct meshopt_Meshlet* src = &tmp_meshlets[mi];
        unsigned int* ml_vertices = tmp_vertices + src->vertex_offset;
        unsigned char* ml_triangles = tmp_triangles + src->triangle_offset;

        meshopt_optimizeMeshlet(ml_vertices, ml_triangles,
                                src->triangle_count, src->vertex_count);

        struct meshopt_Bounds bounds = meshopt_computeMeshletBounds(
            ml_vertices,
            ml_triangles,
            src->triangle_count,
            vertices,
            vertex_count,
            vertex_stride);

        RendererMeshlet* dst = &r->meshlets[r->meshlet_count++];
        dst->prototype_idx = m->prototype_idx;
        dst->index_offset = r->total_meshlet_indices;
        dst->index_count = src->triangle_count * 3u;
        dst->center[0] = bounds.center[0];
        dst->center[1] = bounds.center[1];
        dst->center[2] = bounds.center[2];
        dst->radius = bounds.radius;
        dst->cone_axis[0] = bounds.cone_axis[0];
        dst->cone_axis[1] = bounds.cone_axis[1];
        dst->cone_axis[2] = bounds.cone_axis[2];
        dst->cone_cutoff = bounds.cone_cutoff;

        for (unsigned int t = 0; t < src->triangle_count; t++) {
            unsigned int tri = t * 3u;
            if (r->total_meshlet_indices > UINT32_MAX - 3u) {
                free(tmp_meshlets);
                free(tmp_vertices);
                free(tmp_triangles);
                return 0;
            }
            r->cpu_meshlet_indices[r->total_meshlet_indices++] =
                ml_vertices[ml_triangles[tri + 0u]];
            r->cpu_meshlet_indices[r->total_meshlet_indices++] =
                ml_vertices[ml_triangles[tri + 1u]];
            r->cpu_meshlet_indices[r->total_meshlet_indices++] =
                ml_vertices[ml_triangles[tri + 2u]];
        }
    }

    m->meshlet_index_count = r->total_meshlet_indices - first_index;

    free(tmp_meshlets);
    free(tmp_vertices);
    free(tmp_triangles);
    return 1;
}

#define NUSD_GEOMETRY_CACHE_MAGIC "NUSDGEO1"
#define NUSD_GEOMETRY_CACHE_VERSION 1u
#define NUSD_GEOMETRY_CACHE_FLAG_MESHLETS 1u
#define NUSD_GEOMETRY_CACHE_FLAG_MESHLET_INDEX_ONLY 2u
#define NUSD_GEOMETRY_CACHE_FLAG_OMIT_NAMES 4u

typedef struct {
    char     magic[8];
    uint32_t version;
    uint32_t header_size;
    uint32_t vertex_stride_floats;
    uint32_t flags;
    uint32_t nmeshes;
    uint32_t total_vertices;
    uint32_t total_indices;
    uint32_t meshlet_count;
    uint32_t total_meshlet_indices;
    float    bounds_min[3];
    float    bounds_max[3];
    uint64_t vertices_bytes;
    uint64_t indices_bytes;
    uint64_t meshlets_bytes;
    uint64_t meshlet_indices_bytes;
    uint64_t mesh_records_bytes;
    uint64_t reserved[8];
} NuGeometryCacheHeader;

typedef struct {
    uint32_t active;
    uint32_t visible;
    float    transform[16];
    float    color[3];
    uint32_t nvertices;
    uint32_t nindices;
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t meshlet_offset;
    uint32_t meshlet_count;
    uint32_t meshlet_index_offset;
    uint32_t meshlet_index_count;
    int32_t  prototype_idx;
    int32_t  material_index;
    uint32_t name_len;
    uint32_t reserved[7];
} NuGeometryCacheMeshRecord;

typedef struct {
    uint32_t prototype_idx;
    uint32_t index_offset;
    uint32_t index_count;
    float    center[3];
    float    radius;
    float    cone_axis[3];
    float    cone_cutoff;
} NuGeometryCacheMeshletRecord;

static int write_exact(FILE* f, const void* data, size_t bytes)
{
    return bytes == 0 || fwrite(data, 1, bytes, f) == bytes;
}

static int read_exact(FILE* f, void* data, size_t bytes)
{
    return bytes == 0 || fread(data, 1, bytes, f) == bytes;
}

static int skip_exact(FILE* f, uint64_t bytes)
{
    if (bytes == 0) return 1;
    if (bytes > (uint64_t)LLONG_MAX) return 0;
    return fseeko(f, (off_t)bytes, SEEK_CUR) == 0;
}

static void renderer_cache_bounds(NuRenderer* r,
                                  const float bounds_min[3],
                                  const float bounds_max[3]);
static void renderer_auto_frame_bounds(NuRenderer* r,
                                       const float bounds_min[3],
                                       const float bounds_max[3]);

static int seek_abs_u64(FILE* f, uint64_t offset)
{
    if (offset > (uint64_t)LLONG_MAX) return 0;
    return fseeko(f, (off_t)offset, SEEK_SET) == 0;
}

static int add_u64_checked(uint64_t a, uint64_t b, uint64_t* out)
{
    if (UINT64_MAX - a < b) return 0;
    *out = a + b;
    return 1;
}

static int geometry_cache_offsets(const NuGeometryCacheHeader* h,
                                  uint64_t* vertices_offset,
                                  uint64_t* indices_offset,
                                  uint64_t* meshlets_offset,
                                  uint64_t* meshlet_indices_offset,
                                  uint64_t* mesh_records_offset)
{
    uint64_t off = h->header_size;
    *vertices_offset = off;
    if (!add_u64_checked(off, h->vertices_bytes, &off)) return 0;
    *indices_offset = off;
    if (!add_u64_checked(off, h->indices_bytes, &off)) return 0;
    *meshlets_offset = off;
    if (!add_u64_checked(off, h->meshlets_bytes, &off)) return 0;
    *meshlet_indices_offset = off;
    if (!add_u64_checked(off, h->meshlet_indices_bytes, &off)) return 0;
    *mesh_records_offset = off;
    return 1;
}

static size_t geometry_cache_upload_chunk_size(void)
{
    const char* e = getenv("NUSD_CACHE_UPLOAD_CHUNK_MB");
    unsigned long mb = e && e[0] ? strtoul(e, NULL, 10) : 256ul;
    if (mb < 1ul) mb = 1ul;
    if (mb > 512ul) mb = 512ul;
    return (size_t)mb * 1024u * 1024u;
}

static int upload_file_region_to_gpu_buffer(NuRenderer* r,
                                            FILE* f,
                                            uint64_t file_offset,
                                            GpuBuffer dst,
                                            uint64_t dst_offset,
                                            uint64_t bytes)
{
    if (bytes == 0) return 1;
    if (!r || !f || !dst) return 0;
    if (!seek_abs_u64(f, file_offset)) return 0;

    size_t chunk_size = geometry_cache_upload_chunk_size();
    GpuBufferDesc stage_desc;
    stage_desc.usage = GPU_BUFFER_VERTEX;
    stage_desc.size = chunk_size;
    stage_desc.data = NULL;
    GpuBuffer stage = gpu_create_buffer(r->gpu, &stage_desc);
    uint8_t* chunk = (uint8_t*)gpu_buffer_contents(stage);
    while ((!stage || !chunk) && chunk_size > 1024u * 1024u) {
        if (stage) gpu_destroy_buffer(r->gpu, stage);
        chunk_size /= 2u;
        stage_desc.size = chunk_size;
        stage = gpu_create_buffer(r->gpu, &stage_desc);
        chunk = (uint8_t*)gpu_buffer_contents(stage);
    }
    if (!stage || !chunk) {
        if (stage) gpu_destroy_buffer(r->gpu, stage);
        return 0;
    }

    uint64_t uploaded = 0;
    while (uploaded < bytes) {
        uint64_t remaining = bytes - uploaded;
        size_t n = remaining < (uint64_t)chunk_size ? (size_t)remaining : chunk_size;
        if (!read_exact(f, chunk, n) ||
            !gpu_copy_buffer(r->gpu, stage, 0, dst, dst_offset + uploaded, n)) {
            gpu_destroy_buffer(r->gpu, stage);
            return 0;
        }
        uploaded += n;
    }

    gpu_destroy_buffer(r->gpu, stage);
    return 1;
}

static int synthesize_meshlet_fallbacks_from_cache_indices(FILE* f,
                                                          const NuGeometryCacheHeader* h,
                                                          NuRenderer* r)
{
    if (!f || !h || !r) return 0;
    if (h->vertices_bytes > (uint64_t)LLONG_MAX - h->header_size)
        return 0;

    uint64_t index_base = (uint64_t)h->header_size + h->vertices_bytes;
    for (int i = 0; i < r->nmeshes; i++) {
        RendererMesh* m = &r->meshes[i];
        if (!m->active || m->prototype_idx != i || m->nindices <= 0 ||
            m->meshlet_index_count > 0)
            continue;

        uint32_t nidx = (uint32_t)m->nindices;
        if (m->index_offset > h->total_indices ||
            nidx > h->total_indices - m->index_offset ||
            (uint64_t)m->index_offset > UINT64_MAX / sizeof(uint32_t))
            return 0;

        uint64_t byte_offset = index_base + (uint64_t)m->index_offset * sizeof(uint32_t);
        if (byte_offset > (uint64_t)LLONG_MAX) return 0;
        if (!reserve_meshlet_data(r, r->meshlet_count,
                (uint64_t)r->total_meshlet_indices + (uint64_t)nidx))
            return 0;

        uint32_t dst = r->total_meshlet_indices;
        if (fseeko(f, (off_t)byte_offset, SEEK_SET) != 0 ||
            !read_exact(f, r->cpu_meshlet_indices + dst, (size_t)nidx * sizeof(uint32_t)))
            return 0;

        m->meshlet_index_offset = dst;
        m->meshlet_index_count = nidx;
        r->total_meshlet_indices += nidx;
    }

    for (int i = 0; i < r->nmeshes; i++) {
        RendererMesh* m = &r->meshes[i];
        int proto = m->prototype_idx;
        if (!m->active || proto == i ||
            proto < 0 || proto >= r->nmeshes || !r->meshes[proto].active)
            continue;
        m->meshlet_offset = r->meshes[proto].meshlet_offset;
        m->meshlet_count = r->meshes[proto].meshlet_count;
        m->meshlet_index_offset = r->meshes[proto].meshlet_index_offset;
        m->meshlet_index_count = r->meshes[proto].meshlet_index_count;
    }

    return 1;
}

static int count_meshlet_fallback_indices(const NuGeometryCacheHeader* h,
                                          const NuRenderer* r,
                                          uint32_t* out_extra_indices)
{
    uint64_t extra = 0;

    for (int i = 0; i < r->nmeshes; i++) {
        const RendererMesh* m = &r->meshes[i];
        if (!m->active || m->prototype_idx != i || m->nindices <= 0 ||
            m->meshlet_index_count > 0)
            continue;

        uint32_t nidx = (uint32_t)m->nindices;
        if (m->index_offset > h->total_indices ||
            nidx > h->total_indices - m->index_offset)
            return 0;
        extra += nidx;
        if (extra > UINT32_MAX ||
            extra > (uint64_t)UINT32_MAX - h->total_meshlet_indices)
            return 0;
    }

    *out_extra_indices = (uint32_t)extra;
    return 1;
}

static void propagate_meshlet_fallback_ranges_to_instances(NuRenderer* r)
{
    for (int i = 0; i < r->nmeshes; i++) {
        RendererMesh* m = &r->meshes[i];
        int proto = m->prototype_idx;
        if (!m->active || proto == i ||
            proto < 0 || proto >= r->nmeshes || !r->meshes[proto].active)
            continue;
        m->meshlet_offset = r->meshes[proto].meshlet_offset;
        m->meshlet_count = r->meshes[proto].meshlet_count;
        m->meshlet_index_offset = r->meshes[proto].meshlet_index_offset;
        m->meshlet_index_count = r->meshes[proto].meshlet_index_count;
    }
}

static int bake_meshlet_fallback_indices_from_cpu(NuRenderer* r)
{
    if (!r || !r->cpu_indices) return 0;

    for (int i = 0; i < r->nmeshes; i++) {
        RendererMesh* m = &r->meshes[i];
        if (!m->active || m->prototype_idx != i || m->nindices <= 0 ||
            m->meshlet_index_count > 0)
            continue;

        uint32_t nidx = (uint32_t)m->nindices;
        if (m->index_offset > r->total_indices ||
            nidx > r->total_indices - m->index_offset ||
            (uint64_t)r->total_meshlet_indices + (uint64_t)nidx > UINT32_MAX)
            return 0;
        if (!reserve_meshlet_data(r, r->meshlet_count,
                (uint64_t)r->total_meshlet_indices + (uint64_t)nidx))
            return 0;

        uint32_t dst = r->total_meshlet_indices;
        memcpy(r->cpu_meshlet_indices + dst,
               r->cpu_indices + m->index_offset,
               (size_t)nidx * sizeof(uint32_t));
        m->meshlet_index_offset = dst;
        m->meshlet_index_count = nidx;
        r->total_meshlet_indices += nidx;
    }

    propagate_meshlet_fallback_ranges_to_instances(r);
    return meshlet_ranges_cover_active_meshes(r);
}

static int append_meshlet_fallbacks_to_gpu_indices(FILE* f,
                                                   const NuGeometryCacheHeader* h,
                                                   NuRenderer* r)
{
    uint64_t vertices_offset, indices_offset, meshlets_offset;
    uint64_t meshlet_indices_offset, mesh_records_offset;
    (void)vertices_offset;
    (void)meshlets_offset;
    (void)meshlet_indices_offset;
    (void)mesh_records_offset;
    if (!geometry_cache_offsets(h, &vertices_offset, &indices_offset,
                                &meshlets_offset, &meshlet_indices_offset,
                                &mesh_records_offset))
        return 0;

    for (int i = 0; i < r->nmeshes; i++) {
        RendererMesh* m = &r->meshes[i];
        if (!m->active || m->prototype_idx != i || m->nindices <= 0 ||
            m->meshlet_index_count > 0)
            continue;

        uint32_t nidx = (uint32_t)m->nindices;
        if (m->index_offset > h->total_indices ||
            nidx > h->total_indices - m->index_offset ||
            (uint64_t)m->index_offset > UINT64_MAX / sizeof(uint32_t))
            return 0;

        uint64_t byte_offset = indices_offset + (uint64_t)m->index_offset * sizeof(uint32_t);

        uint32_t dst = r->total_meshlet_indices;
        if ((uint64_t)dst + nidx > UINT32_MAX ||
            !upload_file_region_to_gpu_buffer(
                r, f, byte_offset, r->meshlet_index_buffer,
                (uint64_t)dst * sizeof(uint32_t),
                (uint64_t)nidx * sizeof(uint32_t)))
            return 0;

        m->meshlet_index_offset = dst;
        m->meshlet_index_count = nidx;
        r->total_meshlet_indices += nidx;
    }

    propagate_meshlet_fallback_ranges_to_instances(r);
    return 1;
}

static int load_geometry_cache_direct_meshlet_gpu(NuRenderer* r,
                                                  FILE* f,
                                                  const NuGeometryCacheHeader* h)
{
    uint64_t vertices_offset, indices_offset, meshlets_offset;
    uint64_t meshlet_indices_offset, mesh_records_offset;
    if (!geometry_cache_offsets(h, &vertices_offset, &indices_offset,
                                &meshlets_offset, &meshlet_indices_offset,
                                &mesh_records_offset)) {
        set_error(r, "Geometry cache offsets are invalid");
        return -1;
    }
    (void)indices_offset;
    (void)meshlets_offset;

    nu_clear_scene(r);

    if (!ensure_mesh_capacity(r, (int)h->nmeshes)) {
        nu_clear_scene(r);
        set_error(r, "Failed to allocate geometry cache mesh records");
        return -1;
    }

    r->vertex_buffer = gpu_create_private_buffer(
        r->gpu, GPU_BUFFER_VERTEX, h->vertices_bytes);
    if (!r->vertex_buffer ||
        !upload_file_region_to_gpu_buffer(
            r, f, vertices_offset, r->vertex_buffer, 0, h->vertices_bytes)) {
        nu_clear_scene(r);
        set_error(r, "Failed to stream geometry cache vertices into GPU buffer");
        return -1;
    }

    /* Raster draws from per-mesh meshlet index ranges plus the flat meshlet
     * index stream. The per-meshlet bounds table is kept for cache fallback
     * compatibility but is not consumed on this direct warm path. */
    r->meshlet_count = h->meshlet_count;
    r->total_meshlet_indices = h->total_meshlet_indices;

    if (!seek_abs_u64(f, mesh_records_offset)) {
        nu_clear_scene(r);
        set_error(r, "Failed to seek geometry cache mesh records");
        return -1;
    }

    r->nmeshes = (int)h->nmeshes;
    int active_count = 0;
    uint64_t records_left = h->mesh_records_bytes;
    for (uint32_t i = 0; i < h->nmeshes; i++) {
        NuGeometryCacheMeshRecord rec;
        if (records_left < sizeof(rec)) {
            nu_clear_scene(r);
            set_error(r, "Geometry cache mesh record table is invalid");
            return -1;
        }
        if (!read_exact(f, &rec, sizeof(rec))) {
            nu_clear_scene(r);
            set_error(r, "Failed to read geometry cache mesh records");
            return -1;
        }
        records_left -= sizeof(rec);
        if (rec.nvertices > (uint32_t)INT_MAX ||
            rec.nindices > (uint32_t)INT_MAX ||
            rec.vertex_offset > h->total_vertices ||
            rec.nvertices > h->total_vertices - rec.vertex_offset ||
            rec.index_offset > h->total_indices ||
            rec.nindices > h->total_indices - rec.index_offset ||
            rec.meshlet_offset > h->meshlet_count ||
            rec.meshlet_count > h->meshlet_count - rec.meshlet_offset ||
            rec.meshlet_index_offset > h->total_meshlet_indices ||
            rec.meshlet_index_count > h->total_meshlet_indices - rec.meshlet_index_offset ||
            rec.name_len > records_left ||
            rec.prototype_idx >= (int32_t)h->nmeshes) {
            nu_clear_scene(r);
            set_error(r, "Geometry cache mesh record range is invalid");
            return -1;
        }

        RendererMesh* dst = &r->meshes[i];
        memset(dst, 0, sizeof(*dst));
        dst->active = rec.active ? 1 : 0;
        dst->visible = rec.visible ? 1 : 0;
        dst->env_mask = 0xFFu;
        memcpy(dst->transform, rec.transform, sizeof(dst->transform));
        memcpy(dst->color, rec.color, sizeof(dst->color));
        dst->nvertices = (int)rec.nvertices;
        dst->nindices = (int)rec.nindices;
        dst->vertex_offset = rec.vertex_offset;
        dst->index_offset = rec.index_offset;
        dst->meshlet_offset = rec.meshlet_offset;
        dst->meshlet_count = rec.meshlet_count;
        dst->meshlet_index_offset = rec.meshlet_index_offset;
        dst->meshlet_index_count = rec.meshlet_index_count;
        dst->prototype_idx = rec.prototype_idx;
        dst->material_index = rec.material_index;
        if (rec.name_len > 0) {
            dst->name = (char*)malloc((size_t)rec.name_len + 1u);
            if (!dst->name || !read_exact(f, dst->name, rec.name_len)) {
                nu_clear_scene(r);
                set_error(r, "Failed to read geometry cache mesh name");
                return -1;
            }
            dst->name[rec.name_len] = '\0';
        }
        records_left -= rec.name_len;
        if (dst->active) active_count++;
    }
    if (records_left != 0) {
        nu_clear_scene(r);
        set_error(r, "Geometry cache mesh record table has trailing data");
        return -1;
    }

    uint32_t fallback_indices = 0;
    if (!count_meshlet_fallback_indices(h, r, &fallback_indices)) {
        nu_clear_scene(r);
        set_error(r, "Geometry cache fallback meshlet ranges are invalid");
        return -1;
    }
    if ((h->flags & NUSD_GEOMETRY_CACHE_FLAG_MESHLET_INDEX_ONLY) &&
        fallback_indices > 0) {
        nu_clear_scene(r);
        set_error(r, "Geometry cache omitted original indices but still needs meshlet fallbacks");
        return -1;
    }

    uint64_t final_meshlet_indices =
        (uint64_t)h->total_meshlet_indices + (uint64_t)fallback_indices;
    if (final_meshlet_indices == 0 || final_meshlet_indices > UINT32_MAX ||
        final_meshlet_indices > UINT64_MAX / sizeof(uint32_t)) {
        nu_clear_scene(r);
        set_error(r, "Geometry cache meshlet index count is invalid");
        return -1;
    }

    r->meshlet_index_buffer = gpu_create_private_buffer(
        r->gpu, GPU_BUFFER_INDEX, final_meshlet_indices * sizeof(uint32_t));
    if (!r->meshlet_index_buffer ||
        !upload_file_region_to_gpu_buffer(
            r, f, meshlet_indices_offset, r->meshlet_index_buffer,
            0, h->meshlet_indices_bytes)) {
        nu_clear_scene(r);
        set_error(r, "Failed to stream geometry cache meshlet indices into GPU buffer");
        return -1;
    }

    r->total_meshlet_indices = h->total_meshlet_indices;
    if (fallback_indices > 0 &&
        !append_meshlet_fallbacks_to_gpu_indices(f, h, r)) {
        nu_clear_scene(r);
        set_error(r, "Failed to synthesize geometry cache meshlet fallbacks");
        return -1;
    }

    if (!meshlet_raster_covers_active_meshes(r)) {
        nu_clear_scene(r);
        set_error(r, "Geometry cache meshlets do not cover all active meshes");
        return -1;
    }

    renderer_cache_bounds(r, h->bounds_min, h->bounds_max);
    if (active_count > 0)
        renderer_auto_frame_bounds(r, h->bounds_min, h->bounds_max);

    r->total_vertices = h->total_vertices;
    r->total_indices = h->total_indices;
    r->meshlet_raster_replaces_index_buffer = 1;
    r->meshlets_enabled = r->meshlets_enabled || ((h->flags & NUSD_GEOMETRY_CACHE_FLAG_MESHLETS) != 0);
    r->loaded_from_geometry_cache = 1;
    r->cpu_staging_released = 1;
    r->scene_dirty = 0;
    r->accel_dirty = 1;
    r->rt_blas_valid = 0;
    r->has_frame = 0;
    return active_count;
}

static int renderer_geometry_bounds(NuRenderer* r, float out_min[3], float out_max[3])
{
    if (!r || !out_min || !out_max) return 0;

    if (r->usd_scene) {
        memcpy(out_min, r->usd_scene->bounds_min, 3 * sizeof(float));
        memcpy(out_max, r->usd_scene->bounds_max, 3 * sizeof(float));
        return 1;
    }
    if (r->has_cached_bounds) {
        memcpy(out_min, r->cached_bounds_min, 3 * sizeof(float));
        memcpy(out_max, r->cached_bounds_max, 3 * sizeof(float));
        return 1;
    }
    if (!r->cpu_vertices || r->total_vertices == 0) return 0;

    out_min[0] = out_min[1] = out_min[2] = FLT_MAX;
    out_max[0] = out_max[1] = out_max[2] = -FLT_MAX;
    uint32_t stride = r->vertex_stride_floats ? r->vertex_stride_floats : 12u;
    int found = 0;

    for (int mi = 0; mi < r->nmeshes; mi++) {
        RendererMesh* m = &r->meshes[mi];
        if (!m->active || m->nvertices <= 0) continue;
        if (m->vertex_offset > r->total_vertices ||
            (uint32_t)m->nvertices > r->total_vertices - m->vertex_offset)
            continue;

        for (int vi = 0; vi < m->nvertices; vi++) {
            const float* v = r->cpu_vertices + ((size_t)m->vertex_offset + (size_t)vi) * stride;
            const float* xf = m->transform;
            float x = v[0] * xf[0] + v[1] * xf[4] + v[2] * xf[8]  + xf[12];
            float y = v[0] * xf[1] + v[1] * xf[5] + v[2] * xf[9]  + xf[13];
            float z = v[0] * xf[2] + v[1] * xf[6] + v[2] * xf[10] + xf[14];
            if (x < out_min[0]) out_min[0] = x;
            if (y < out_min[1]) out_min[1] = y;
            if (z < out_min[2]) out_min[2] = z;
            if (x > out_max[0]) out_max[0] = x;
            if (y > out_max[1]) out_max[1] = y;
            if (z > out_max[2]) out_max[2] = z;
            found = 1;
        }
    }

    return found;
}

static void renderer_cache_bounds(NuRenderer* r, const float bounds_min[3], const float bounds_max[3])
{
    if (!r || !bounds_min || !bounds_max) return;
    memcpy(r->cached_bounds_min, bounds_min, 3 * sizeof(float));
    memcpy(r->cached_bounds_max, bounds_max, 3 * sizeof(float));
    r->has_cached_bounds = 1;
}

static void renderer_auto_frame_bounds(NuRenderer* r,
                                       const float bounds_min[3],
                                       const float bounds_max[3])
{
    if (!r || !bounds_min || !bounds_max) return;

    float cx = (bounds_min[0] + bounds_max[0]) * 0.5f;
    float cy = (bounds_min[1] + bounds_max[1]) * 0.5f;
    float cz = (bounds_min[2] + bounds_max[2]) * 0.5f;

    float dx = bounds_max[0] - bounds_min[0];
    float dy = bounds_max[1] - bounds_min[1];
    float dz = bounds_max[2] - bounds_min[2];
    if (dx < 1e-3f) dx = 1.0f;
    if (dy < 1e-3f) dy = 1.0f;
    if (dz < 1e-3f) dz = 1.0f;
    float diag = sqrtf(dx * dx + dy * dy + dz * dz);

    float dirx, diry, dirz;
    if (dx >= dy && dx >= dz) {
        dirx = 0.4f; diry = 0.5f; dirz = 1.0f;
    } else if (dz >= dx && dz >= dy) {
        dirx = 1.0f; diry = 0.5f; dirz = 0.4f;
    } else {
        dirx = 1.0f; diry = 0.3f; dirz = 1.0f;
    }
    float dirlen = sqrtf(dirx * dirx + diry * diry + dirz * dirz);
    dirx /= dirlen; diry /= dirlen; dirz /= dirlen;

    const float HALF_FOV_TAN = 0.41421f;
    const float FRAME_SLACK  = 1.4f;
    float radius = diag * 0.5f;
    float dist = (radius / HALF_FOV_TAN) * FRAME_SLACK;

    float eye[3] = {
        cx + dirx * dist,
        cy + diry * dist,
        cz + dirz * dist,
    };
    float target[3] = { cx, cy, cz };

    NuCameraDesc cam;
    memcpy(cam.eye, eye, sizeof(eye));
    memcpy(cam.target, target, sizeof(target));
    cam.fov_degrees = 45.0f;
    cam.near_clip = diag * 0.001f;
    cam.far_clip = diag * 100.0f;
    nu_set_camera(r, 0, &cam);
}

typedef struct {
    const char* name;
    int         had_value;
    char        value[32];
} EnvOverride;

static void env_set_default(EnvOverride* e, const char* name, const char* value)
{
    if (!e || !name || !value) return;
    e->name = name;
    const char* prev = getenv(name);
    e->had_value = prev != NULL;
    e->value[0] = '\0';
    if (prev) {
        size_t n = strlen(prev);
        if (n >= sizeof(e->value)) n = sizeof(e->value) - 1;
        memcpy(e->value, prev, n);
        e->value[n] = '\0';
        return;
    }
    setenv(name, value, 1);
}

static void env_restore(const EnvOverride* e)
{
    if (!e || !e->name) return;
    if (e->had_value) setenv(e->name, e->value, 1);
    else              unsetenv(e->name);
}

static int lazy_mesh_requested(void)
{
    const char* e = getenv("NUSD_LAZY_MESH");
    return e && e[0] && e[0] != '0';
}

static uint32_t* load_shader_file(const char* path, uint32_t* out_size)
{
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t* data = (uint32_t*)malloc((size_t)sz);
    if (data) {
        if (fread(data, 1, (size_t)sz, f) != (size_t)sz) {
            free(data);
            fclose(f);
            return NULL;
        }
    }
    fclose(f);
    *out_size = (uint32_t)sz;
    return data;
}

/* Rebuild merged vertex/index buffers and upload to GPU */
static int rebuild_gpu_buffers(NuRenderer* r)
{
    if (r->cpu_staging_released) {
        set_error(r, "Cannot rebuild GPU buffers after CPU staging was released");
        return 0;
    }

    /* Count total vertices and indices, assign offsets — skip instances.
     * Instances share their prototype's vertex/index data and inherited the
     * prototype's offsets at nu_add_mesh_instance() time; reassigning them
     * here would invalidate that aliasing and inflate total_{v,i} past the
     * actual cpu_{vertices,indices} length. */
    uint32_t total_v = 0, total_i = 0;
    for (int i = 0; i < r->nmeshes; i++) {
        if (!r->meshes[i].active) continue;
        if (r->meshes[i].prototype_idx != i) continue;  /* instance — keep inherited offsets */
        r->meshes[i].vertex_offset = total_v;
        r->meshes[i].index_offset = total_i;
        total_v += (uint32_t)r->meshes[i].nvertices;
        total_i += (uint32_t)r->meshes[i].nindices;
    }
    /* Now propagate updated prototype offsets to all instances so they match
     * the rebuilt buffer layout. */
    for (int i = 0; i < r->nmeshes; i++) {
        if (!r->meshes[i].active) continue;
        int proto = r->meshes[i].prototype_idx;
        if (proto == i) continue;
        if (proto >= 0 && proto < r->nmeshes && r->meshes[proto].active) {
            r->meshes[i].vertex_offset = r->meshes[proto].vertex_offset;
            r->meshes[i].index_offset  = r->meshes[proto].index_offset;
        }
    }

    if (total_v == 0 || total_i == 0) {
        r->total_vertices = 0;
        r->total_indices = 0;
        r->meshlet_raster_replaces_index_buffer = 0;
        return 1; /* empty scene is valid */
    }

    /* Destroy old GPU buffers */
    if (r->vertex_buffer) { gpu_destroy_buffer(r->gpu, r->vertex_buffer); r->vertex_buffer = NULL; }
    if (r->index_buffer)  { gpu_destroy_buffer(r->gpu, r->index_buffer);  r->index_buffer = NULL; }
    if (r->meshlet_index_buffer) {
        gpu_destroy_buffer(r->gpu, r->meshlet_index_buffer);
        r->meshlet_index_buffer = NULL;
    }

    /* Upload merged vertex buffer */
    if (r->cpu_vertices) {
        uint32_t stride = r->vertex_stride_floats ? r->vertex_stride_floats : 12u;
        GpuBufferDesc vb_desc;
        vb_desc.usage = GPU_BUFFER_VERTEX;
        vb_desc.size  = (uint64_t)total_v * stride * sizeof(float);
        vb_desc.data  = r->cpu_vertices;
        r->vertex_buffer = gpu_create_buffer(r->gpu, &vb_desc);
        if (!r->vertex_buffer) {
            set_error(r, "Failed to create vertex buffer");
            return 0;
        }
    }

    r->meshlet_raster_replaces_index_buffer = meshlet_raster_covers_active_meshes(r);

    /* Upload merged index buffer. Meshlet raster can fully replace this buffer
     * for geometry-only raster loads, avoiding a duplicate multi-GB index
     * allocation on full DSX. */
    if (r->cpu_indices && !r->meshlet_raster_replaces_index_buffer) {
        GpuBufferDesc ib_desc;
        ib_desc.usage = GPU_BUFFER_INDEX;
        ib_desc.size  = (uint64_t)total_i * sizeof(uint32_t);
        ib_desc.data  = r->cpu_indices;
        r->index_buffer = gpu_create_buffer(r->gpu, &ib_desc);
        if (!r->index_buffer) {
            set_error(r, "Failed to create index buffer");
            return 0;
        }
    }

    if (r->meshlet_raster_enabled && r->cpu_meshlet_indices && r->total_meshlet_indices > 0) {
        GpuBufferDesc mib_desc;
        mib_desc.usage = GPU_BUFFER_INDEX;
        mib_desc.size  = (uint64_t)r->total_meshlet_indices * sizeof(uint32_t);
        mib_desc.data  = r->cpu_meshlet_indices;
        r->meshlet_index_buffer = gpu_create_buffer(r->gpu, &mib_desc);
        if (!r->meshlet_index_buffer) {
            set_error(r, "Failed to create meshlet index buffer");
            return 0;
        }
    }

    r->total_vertices = total_v;
    r->total_indices = total_i;
    r->scene_dirty = 0;
    r->accel_dirty = 1;
    r->rt_blas_valid = 0;
    if (can_release_cpu_staging_after_upload(r))
        release_cpu_staging(r);

    return 1;
}

/* Build RT acceleration structures from current mesh data */
/* RT culled-proxy mode: build the TLAS from only the camera-visible PI set
 * (the full Moana 202M-instance TLAS is 45.85 GiB — over the device limit).
 * Active when the no-cull-all-geometry load is in effect or forced. */
static int rt_cull_active(void)
{
    const char* a = getenv("NUSD_NO_CULL_ALL_GEOMETRY");
    const char* b = getenv("NUSD_RT_CULL");
    if (b && b[0] && b[0] != '0') return 1;
    if (a && a[0] && a[0] != '0') return 1;
    return 0;
}

/* Skip curve AO/self-shadow secondary rays (6 per curve hit, each traversing
 * the whole curve BVH). For dense-curve scenes (Moana: 25.5M segments) those
 * secondary rays stall the dispatch indefinitely. Auto-on above a threshold;
 * NUSD_CURVE_FAST={0,1} forces it off/on. Mesh shadows are unaffected. */
static uint32_t renderer_curve_fast(const NuRenderer* r)
{
    const char* e = getenv("NUSD_CURVE_FAST");
    if (e && e[0]) return (e[0] != '0') ? 1u : 0u;
    return (r->curves.nseg > 1000000) ? 1u : 0u;
}

/* Curve culled proxy: a single BLAS over all ~25.5M Moana curve AABBs is
 * pathological for ray traversal (thin overlapping boxes → enormous candidate
 * lists per ray; the dispatch never finishes). Mirror the mesh culled-proxy:
 * keep only camera-visible curve segments, build a small curve BLAS from that
 * subset, every frame. Returns 1 on success (gpu->curve_blas valid or curves
 * empty), 0 on failure. CPU curve arrays must be retained (see
 * can_release_curve_cpu_staging_after_upload). */
static int rebuild_curve_blas_culled(NuRenderer* r, const float planes[6][4],
                                     const float eye[3], float min_angular)
{
    if (r->curves.nseg <= 0 || !r->curves.segments || !r->curves.aabbs)
        return 1;  /* nothing to do (or CPU staging gone — leave existing BLAS) */

    int nseg = r->curves.nseg;
    const SceneCurveAabb* ab = r->curves.aabbs;

    /* Count pass. */
    int kept = 0;
    for (int i = 0; i < nseg; i++) {
        float mn[3] = { ab[i].min_x, ab[i].min_y, ab[i].min_z };
        float mx[3] = { ab[i].max_x, ab[i].max_y, ab[i].max_z };
        if (!aabb_in_frustum(planes, mn, mx)) continue;
        float c[3] = { 0.5f*(mn[0]+mx[0]), 0.5f*(mn[1]+mx[1]), 0.5f*(mn[2]+mx[2]) };
        float dx = mx[0]-mn[0], dy = mx[1]-mn[1], dz = mx[2]-mn[2];
        float rad = 0.5f * sqrtf(dx*dx + dy*dy + dz*dz);
        if (sphere_outside_frustum(planes, c, rad)) continue;
        if (min_angular > 0.0f) {
            float ex = c[0]-eye[0], ey = c[1]-eye[1], ez = c[2]-eye[2];
            float dist = sqrtf(ex*ex + ey*ey + ez*ez);
            if (dist > 1e-4f && rad/dist < min_angular) continue;
        }
        kept++;
    }

    if (getenv("NUSD_CULL_DIAG"))
        fprintf(stderr, "nusd_renderer: curve cull — %d of %d segments visible\n",
                kept, nseg);

    if (kept == 0) {
        /* No visible curves this camera — drop the curve BLAS so the TLAS
         * carries no curve instance (seg_count<=0 nils curve_blas). */
        gpu_upload_curve_data(r->gpu, NULL, NULL, NULL, 0);
        return 1;
    }

    SceneCurveSegment* fs = (SceneCurveSegment*)malloc((size_t)kept * sizeof(SceneCurveSegment));
    SceneCurveAabb*    fa = (SceneCurveAabb*)   malloc((size_t)kept * sizeof(SceneCurveAabb));
    float*             fc = (float*)            malloc((size_t)kept * 3 * sizeof(float));
    if (!fs || !fa || !fc) { free(fs); free(fa); free(fc);
        set_error(r, "curve cull alloc failed"); return 0; }

    int w = 0;
    for (int i = 0; i < nseg && w < kept; i++) {
        float mn[3] = { ab[i].min_x, ab[i].min_y, ab[i].min_z };
        float mx[3] = { ab[i].max_x, ab[i].max_y, ab[i].max_z };
        if (!aabb_in_frustum(planes, mn, mx)) continue;
        float c[3] = { 0.5f*(mn[0]+mx[0]), 0.5f*(mn[1]+mx[1]), 0.5f*(mn[2]+mx[2]) };
        float dx = mx[0]-mn[0], dy = mx[1]-mn[1], dz = mx[2]-mn[2];
        float rad = 0.5f * sqrtf(dx*dx + dy*dy + dz*dz);
        if (sphere_outside_frustum(planes, c, rad)) continue;
        if (min_angular > 0.0f) {
            float ex = c[0]-eye[0], ey = c[1]-eye[1], ez = c[2]-eye[2];
            float dist = sqrtf(ex*ex + ey*ey + ez*ez);
            if (dist > 1e-4f && rad/dist < min_angular) continue;
        }
        fs[w] = r->curves.segments[i];
        fa[w] = r->curves.aabbs[i];
        fc[w*3+0] = r->curves.colors[i*3+0];
        fc[w*3+1] = r->curves.colors[i*3+1];
        fc[w*3+2] = r->curves.colors[i*3+2];
        w++;
    }

    int ok = gpu_upload_curve_data(r->gpu, fs, fa, fc, w) && gpu_build_curve_blas(r->gpu);
    free(fs); free(fa); free(fc);
    if (!ok) { set_error(r, "culled curve BLAS build failed"); return 0; }
    return 1;
}

static int rebuild_accel(NuRenderer* r)
{
    if (!r->enable_rt || !gpu_rt_available(r->gpu)) return 1;
    /* Empty scene (no meshes AND no curves) — nothing to accelerate. */
    if (r->total_indices == 0 && r->curves.nseg == 0) return 1;
    double t0 = wall_seconds();

    /* Load RT shaders */
    char rgen_path[1024], rmiss_path[1024], rchit_path[1024];
    snprintf(rgen_path, sizeof(rgen_path), "%s/raytrace.rgen.spv", SHADER_DIR);
    snprintf(rmiss_path, sizeof(rmiss_path), "%s/raytrace.rmiss.spv", SHADER_DIR);
    snprintf(rchit_path, sizeof(rchit_path), "%s/raytrace.rchit.spv", SHADER_DIR);

    uint32_t rgen_sz = 0, rmiss_sz = 0, rchit_sz = 0;
    uint32_t* rgen_spv  = load_shader_file(rgen_path, &rgen_sz);
    uint32_t* rmiss_spv = load_shader_file(rmiss_path, &rmiss_sz);
    uint32_t* rchit_spv = load_shader_file(rchit_path, &rchit_sz);

    if (!rgen_spv || !rmiss_spv || !rchit_spv) {
        free(rgen_spv); free(rmiss_spv); free(rchit_spv);
        set_error(r, "Failed to load RT shaders");
        return 0;
    }

    /* Build RT mesh descriptors */
    int rt_base_count = 0;
    for (int i = 0; i < r->nmeshes; i++)
        if (r->meshes[i].active && r->meshes[i].nindices > 0)
            rt_base_count++;

    /* Compact PI fanout: each batch contributes `count` extra TLAS
     * instances pointing at the batch's prototype BLAS. The temporary
     * GpuRtMeshDesc array peaks at (~96 B * (rt_base + pi_total)) and is
     * freed after gpu_build_rt_scene returns; the steady-state scene
     * representation stays compact. */
    /* Culled-proxy setup: when active, only camera-visible PI instances enter
     * the TLAS (same frustum + screen-angular test as the raster draw cull). */
    int   rt_cull = rt_cull_active();
    float rt_planes[6][4];
    float rt_eye[3] = {0,0,0};
    float rt_min_angular = 0.0f;
    if (rt_cull) {
        float rt_vp[16];
        camera_get_vp(&r->camera, rt_vp);
        camera_get_eye(&r->camera, rt_eye);
        extract_frustum_planes(rt_vp, rt_planes);
        const char* ang = getenv("NUSD_CULL_MIN_ANGULAR");
        rt_min_angular = ang ? (float)atof(ang) : 0.0f;
    }

    /* Curve culled proxy (per camera). Rebuilds gpu->curve_blas from only the
     * visible curve segments BEFORE the TLAS build below picks it up. Skipped
     * when not culling — there the one-time full curve upload (nu_render) owns
     * the curve BLAS. */
    if (rt_cull && r->curves.nseg > 0) {
        if (!rebuild_curve_blas_culled(r, rt_planes, rt_eye, rt_min_angular))
            return 0;
    }

    uint64_t rt_pi_total = 0;
    for (int b = 0; b < r->npi_batches; b++) {
        RendererPiBatch* pb = &r->pi_batches[b];
        int proto = pb->proto_renderer_idx;
        if (proto < 0 || proto >= r->nmeshes) continue;
        if (!r->meshes[proto].active || r->meshes[proto].nindices == 0) continue;
        if (!rt_cull) { rt_pi_total += pb->count; continue; }
        pi_batch_ensure_bounds(r, pb);
        if (!aabb_in_frustum(rt_planes, pb->cull_min, pb->cull_max)) continue;
        for (uint32_t k = 0; k < pb->count; k++) {
            float wc[3];
            float wr = pi_instance_world_sphere(pb->transforms[k].m,
                                                pb->proto_center, pb->proto_radius, wc);
            if (sphere_outside_frustum(rt_planes, wc, wr)) continue;
            if (rt_min_angular > 0.0f) {
                float dx = wc[0]-rt_eye[0], dy = wc[1]-rt_eye[1], dz = wc[2]-rt_eye[2];
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist > 1e-4f && wr/dist < rt_min_angular) continue;
            }
            rt_pi_total++;
        }
    }
    uint64_t rt_count_total = (uint64_t)rt_base_count + rt_pi_total;
    if (rt_count_total > (uint64_t)INT32_MAX) {
        set_error(r, "RT scene exceeds INT32 mesh count (PI fanout too large)");
        free(rgen_spv); free(rmiss_spv); free(rchit_spv);
        return 0;
    }
    int rt_count = (int)rt_count_total;

    GpuRtMeshDesc* rt_meshes = (GpuRtMeshDesc*)calloc(
        (size_t)rt_count, sizeof(GpuRtMeshDesc));
    if (!rt_meshes) {
        set_error(r, "Failed to allocate RT mesh descriptors");
        free(rgen_spv); free(rmiss_spv); free(rchit_spv);
        return 0;
    }

    /* Map original mesh-table index → renumbered RT index (active+indexed only).
     * Required so we can translate each instance's prototype_idx (an index into
     * r->meshes) into the RT descriptor index gpu_build_rt_scene expects. */
    int* mesh_to_ri = (int*)malloc((size_t)r->nmeshes * sizeof(int));
    for (int i = 0; i < r->nmeshes; i++) mesh_to_ri[i] = -1;
    int ri = 0;
    for (int i = 0; i < r->nmeshes; i++) {
        if (r->meshes[i].active && r->meshes[i].nindices > 0) {
            mesh_to_ri[i] = ri++;
        }
    }

    ri = 0;
    for (int i = 0; i < r->nmeshes; i++) {
        RendererMesh* m = &r->meshes[i];
        if (!m->active || m->nindices == 0) continue;

        rt_meshes[ri].vertex_buf    = r->vertex_buffer;
        rt_meshes[ri].vertex_count  = (uint32_t)m->nvertices;
        rt_meshes[ri].vertex_stride = (r->vertex_stride_floats ? r->vertex_stride_floats : 12u) * sizeof(float);
        rt_meshes[ri].index_buf     = r->index_buffer;
        rt_meshes[ri].index_count   = (uint32_t)m->nindices;
        rt_meshes[ri].vertex_offset = m->vertex_offset;
        rt_meshes[ri].index_offset  = m->index_offset;

        /* Resolve prototype_idx through the active-only renumbering. Falls
         * back to self-prototype if the recorded prototype isn't active in
         * this build (e.g. instance whose prototype was removed). */
        int proto = m->prototype_idx;
        int proto_ri = (proto >= 0 && proto < r->nmeshes) ? mesh_to_ri[proto] : -1;
        rt_meshes[ri].prototype_idx = (proto_ri >= 0) ? proto_ri : ri;

        /* World transform: USD row-vector 4x4 → VkTransformMatrixKHR 3x4
         * (column-vector row-major; translation in last column). */
        usd4x4_to_vk3x4(m->transform, rt_meshes[ri].transform);

        rt_meshes[ri].color[0] = m->color[0];
        rt_meshes[ri].color[1] = m->color[1];
        rt_meshes[ri].color[2] = m->color[2];
        rt_meshes[ri].material_index = m->material_index;
        rt_meshes[ri].mask = mesh_rt_mask(m);

        ri++;
    }

    /* Append PI clones. Each clone shares the prototype's vertex/index
     * buffer offsets and BLAS — only the transform + color/material vary
     * per instance. mask comes from the prototype RendererMesh so RT
     * visibility honors the same env_mask the proto would. */
    for (int b = 0; b < r->npi_batches; b++) {
        const RendererPiBatch* pb = &r->pi_batches[b];
        int proto = pb->proto_renderer_idx;
        if (proto < 0 || proto >= r->nmeshes) continue;
        RendererMesh* pm = &r->meshes[proto];
        if (!pm->active || pm->nindices == 0) continue;
        int proto_ri = mesh_to_ri[proto];
        if (proto_ri < 0) continue;

        unsigned char pi_mask = mesh_rt_mask(pm);
        if (rt_cull && pb->cull_valid &&
            !aabb_in_frustum(rt_planes, pb->cull_min, pb->cull_max)) continue;
        for (uint32_t k = 0; k < pb->count; k++) {
            if (rt_cull) {
                float wc[3];
                float wr = pi_instance_world_sphere(pb->transforms[k].m,
                                                    pb->proto_center, pb->proto_radius, wc);
                if (sphere_outside_frustum(rt_planes, wc, wr)) continue;
                if (rt_min_angular > 0.0f) {
                    float dx = wc[0]-rt_eye[0], dy = wc[1]-rt_eye[1], dz = wc[2]-rt_eye[2];
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (dist > 1e-4f && wr/dist < rt_min_angular) continue;
                }
            }
            if (ri >= rt_count) break;   /* safety: never exceed the count pass */
            rt_meshes[ri].vertex_buf    = r->vertex_buffer;
            rt_meshes[ri].vertex_count  = (uint32_t)pm->nvertices;
            rt_meshes[ri].vertex_stride = (r->vertex_stride_floats
                                           ? r->vertex_stride_floats
                                           : 12u) * sizeof(float);
            rt_meshes[ri].index_buf     = r->index_buffer;
            rt_meshes[ri].index_count   = (uint32_t)pm->nindices;
            rt_meshes[ri].vertex_offset = pm->vertex_offset;
            rt_meshes[ri].index_offset  = pm->index_offset;
            rt_meshes[ri].prototype_idx = proto_ri;
            pi_xform12_to_vk3x4(pb->transforms[k].m, rt_meshes[ri].transform);
            rt_meshes[ri].color[0] = pm->color[0];
            rt_meshes[ri].color[1] = pm->color[1];
            rt_meshes[ri].color[2] = pm->color[2];
            rt_meshes[ri].material_index = pm->material_index;
            rt_meshes[ri].mask = pi_mask;
            ri++;
        }
    }
    free(mesh_to_ri);

    if (rt_cull && getenv("NUSD_CULL_DIAG"))
        fprintf(stderr, "nusd_renderer: RT culled-proxy TLAS — %d instances "
                "(of %llu base + %llu PI)\n", ri, (unsigned long long)rt_base_count,
                (unsigned long long)rt_pi_total);

    /* BLAS reuse: when only the camera/cull set changed (geometry untouched),
     * keep the cached ~30k BLAS and rebuild just the TLAS — turns a ~140 s
     * per-camera rebuild into a few seconds. Falls back to a full build if
     * the cached BLAS aren't reusable (geometry/curve change, or the unique
     * BLAS count drifted). */
    int ok = 0;
    if (r->rt_blas_valid && gpu_rt_can_reuse_blas(r->gpu)) {
        ok = gpu_rebuild_rt_tlas(r->gpu, rt_meshes, (uint32_t)ri);
        if (!ok) {
            /* reuse rejected (e.g. count drift) — fall back to full rebuild */
            gpu_destroy_rt_scene(r->gpu);
            r->rt_blas_valid = 0;
        }
    }
    if (!ok) {
        gpu_destroy_rt_scene(r->gpu);
        ok = gpu_build_rt_scene(r->gpu,
            rt_meshes, (uint32_t)ri,
            rgen_spv, rgen_sz,
            rmiss_spv, rmiss_sz,
            rchit_spv, rchit_sz);
        if (ok) r->rt_blas_valid = 1;
    }

    free(rt_meshes);
    free(rgen_spv);
    free(rmiss_spv);
    free(rchit_spv);

    if (!ok) {
        set_error(r, "Failed to build RT acceleration structures");
        return 0;
    }

    r->accel_dirty = 0;
    r->tiled_pipeline_built = 0;    /* tiled descriptor set references old TLAS/scene */
    r->raycast_pipeline_built = 0;  /* raycast descriptor set references old TLAS */
    r->timings.blas_build_ms = elapsed_ms(t0);
    r->timings.tlas_build_ms = 0.0f; /* Metal builds BLAS/TLAS in one helper today. */
    if (getenv("NUSD_LOAD_TIMING"))
        fprintf(stderr, "nusd_renderer: PROFILE BLAS/TLAS build = %.0f ms\n",
                r->timings.blas_build_ms);
    return 1;
}

/* ---- Lifecycle ---- */

NuRenderer* nu_renderer_create(const NuRendererConfig* config)
{
    NuRenderer* r = (NuRenderer*)calloc(1, sizeof(NuRenderer));
    if (!r) return NULL;

    /* Apply config (or defaults) */
    r->width  = config && config->width  > 0 ? config->width  : 1920;
    r->height = config && config->height > 0 ? config->height : 1080;
    r->enable_rt = config ? config->enable_rt : 1;
    r->enable_materials = config ? config->enable_materials : 0;
    r->vertex_stride_floats = r->enable_materials ? 12u : 9u;
    r->meshlet_raster_enabled = env_flag_enabled("NUSD_MESHLET_RASTER");
    r->meshlets_enabled = r->meshlet_raster_enabled || env_flag_enabled("NUSD_MESHLETS");
    r->preserve_instance_draws = !env_flag_enabled("NUSD_FLATTEN_INSTANCE_DRAWS");
    r->release_cpu_staging_after_upload = !env_flag_enabled("NUSD_KEEP_CPU_STAGING");
    r->current_time = (double)NAN;  /* default: use authored default time */

    /* Init GPU (Metal: no window/surface needed) */
    r->gpu = gpu_init(NULL, r->width, r->height);
    if (!r->gpu) {
        set_error(r, "gpu_init failed");
        free(r);
        return NULL;
    }

    /* Headless: skip acquire/present, render to fixed swapchain image */
    gpu_set_headless(r->gpu, 1);

    gpu_overlay_init(r->gpu);

    r->gs = gs_scene_create();
    if (!r->gs) {
        set_error(r, "gs_scene_create failed");
        gpu_shutdown(r->gpu);
        free(r);
        return NULL;
    }

    /* Init camera at a default position */
    {
        float bmin[3] = {-1, -1, -1};
        float bmax[3] = { 1,  1,  1};
        camera_init(&r->camera, bmin, bmax,
                    (float)r->width / (float)r->height);
    }

    /* Create raster pipeline */
    char vert_path[1024], frag_path[1024];
    snprintf(vert_path, sizeof(vert_path), "%s/mesh.vert.spv", SHADER_DIR);
    snprintf(frag_path, sizeof(frag_path), "%s/mesh.frag.spv", SHADER_DIR);

    uint32_t vert_sz = 0, frag_sz = 0;
    uint32_t* vert_spv = load_shader_file(vert_path, &vert_sz);
    uint32_t* frag_spv = load_shader_file(frag_path, &frag_sz);

    if (vert_spv && frag_spv) {
        uint32_t uv_offset = (r->vertex_stride_floats >= 12u) ? 9u : 6u;
        GpuVertexAttrib attribs[3] = {
            { .location = 0, .offset = 0,  .format = GPU_FORMAT_FLOAT3 },
            { .location = 1, .offset = 12, .format = GPU_FORMAT_FLOAT3 },
            { .location = 2, .offset = uv_offset * sizeof(float), .format = GPU_FORMAT_FLOAT2 },
        };

        GpuPipelineDesc pipe_desc;
        memset(&pipe_desc, 0, sizeof(pipe_desc));
        pipe_desc.vert_spv           = vert_spv;
        pipe_desc.vert_size          = vert_sz;
        pipe_desc.frag_spv           = frag_spv;
        pipe_desc.frag_size          = frag_sz;
        pipe_desc.push_constant_size = (uint32_t)sizeof(GpuMeshPushConstants);
        pipe_desc.vertex_stride      = r->vertex_stride_floats * sizeof(float);
        pipe_desc.attribs            = attribs;
        pipe_desc.nattribs           = 3;

        r->raster_pipeline = gpu_create_pipeline(r->gpu, &pipe_desc);

        /* Shadow pipeline */
        if (gpu_rt_available(r->gpu)) {
            char shadow_path[1024];
            snprintf(shadow_path, sizeof(shadow_path),
                     "%s/mesh_shadow.frag.spv", SHADER_DIR);
            uint32_t shadow_sz = 0;
            uint32_t* shadow_spv = load_shader_file(shadow_path, &shadow_sz);
            if (shadow_spv) {
                GpuPipelineDesc sdesc = pipe_desc;
                sdesc.frag_spv  = shadow_spv;
                sdesc.frag_size = shadow_sz;
                r->shadow_pipeline = gpu_create_shadow_pipeline(r->gpu, &sdesc);
                free(shadow_spv);
            }
        }

        /* Environment-background pipeline (fullscreen triangle that paints
         * the GGX-prefiltered HDR onto background pixels). Best-effort:
         * silently no-ops at draw time on backends where it returns 0
         * (e.g. when the shader_library isn't fully populated yet). */
        gpu_create_env_bg_pipeline(r->gpu);
    }
    free(vert_spv);
    free(frag_spv);

    /* Allocate readback buffer */
    r->readback_pixels = (uint8_t*)malloc((size_t)r->width * r->height * 4);

    fprintf(stderr, "nusd_renderer: created (%dx%d, RT=%s)\n",
            r->width, r->height,
            gpu_rt_available(r->gpu) ? "yes" : "no");

    return r;
}

void nu_renderer_destroy(NuRenderer* r)
{
    if (!r) return;

    free(r->readback_pixels);
    free(r->cpu_vertices);
    free(r->cpu_indices);
    free(r->cpu_meshlet_indices);
    free(r->meshlets);
    free(r->raster_batch_counts);
    free(r->raster_batch_firsts);
    free(r->raster_batch_cursors);
    free(r->raster_batch_prototypes);

    free(r->pi_batches);
    r->pi_batches = NULL;
    r->npi_batches = 0;
    free(r->pi_transforms_owned);
    r->pi_transforms_owned = NULL;
    r->npi_transforms_total = 0;

    for (int i = 0; i < r->nmeshes; i++)
        free(r->meshes[i].name);
    free(r->meshes);

    free(r->curves.segments);
    free(r->curves.aabbs);
    free(r->curves.colors);
    gs_scene_destroy(r->gs);

    if (r->usd_scene) scene_free(r->usd_scene);

    if (r->raster_pipeline) gpu_destroy_pipeline(r->gpu, r->raster_pipeline);
    if (r->shadow_pipeline) gpu_destroy_pipeline(r->gpu, r->shadow_pipeline);
    if (r->vertex_buffer) gpu_destroy_buffer(r->gpu, r->vertex_buffer);
    if (r->index_buffer) gpu_destroy_buffer(r->gpu, r->index_buffer);
    if (r->meshlet_index_buffer) gpu_destroy_buffer(r->gpu, r->meshlet_index_buffer);
    if (r->instance_buffer) gpu_destroy_buffer(r->gpu, r->instance_buffer);

    gpu_gs_destroy(r->gpu);
    gpu_destroy_rt_scene(r->gpu);
    gpu_shutdown(r->gpu);

    geo_census_reset(&r->geo_census);
    free(r);
}

/* ---- Scene building ---- */

int nu_add_mesh(NuRenderer* r, const NuMeshDesc* desc)
{
    if (!r || !desc || !desc->positions || !desc->indices)
        return NU_ERROR;
    if (r->cpu_staging_released) {
        set_error(r, "Cannot add mesh after CPU staging was released");
        return NU_ERROR_UNSUPPORTED;
    }
    if (desc->nvertices <= 0 || desc->nindices <= 0)
        return NU_ERROR;

    if (geo_dedup_enabled()) {
        int proto_slot = geo_dedup_lookup(r, desc);
        /* Share geometry across differing materials. Material is applied
         * per-instance downstream — RT reads MeshData.material_index and raster
         * reads RasterInstanceData.material_index (both filled from this mesh's
         * own material_index at line below / fill_raster_instance_data). The
         * per-vertex matID slot in the shared buffer is never read by either
         * shader, so collapsing byte-identical geometry with distinct materials
         * onto one prototype/BLAS is bit-identical to the per-mesh path. */
        if (proto_slot >= 0 && proto_slot < r->nmeshes &&
            r->meshes[proto_slot].active) {
            int slot = append_mesh_slot(r);
            if (slot < 0) return NU_ERROR;
            RendererMesh* p = &r->meshes[proto_slot];
            RendererMesh* m = &r->meshes[slot];
            memset(m, 0, sizeof(RendererMesh));
            m->active = 1;
            m->visible = 1;
            m->env_mask = 0xFFu;
            m->nvertices = p->nvertices;
            m->nindices = p->nindices;
            m->vertex_offset = p->vertex_offset;
            m->index_offset = p->index_offset;
            m->meshlet_offset = p->meshlet_offset;
            m->meshlet_count = p->meshlet_count;
            m->meshlet_index_offset = p->meshlet_index_offset;
            m->meshlet_index_count = p->meshlet_index_count;
            m->prototype_idx = p->prototype_idx;
            m->material_index = desc->material_index;
            if (desc->transform) {
                memcpy(m->transform, desc->transform, 16 * sizeof(float));
            } else {
                identity_matrix(m->transform);
            }
            if (desc->display_color[0] != 0.0f ||
                desc->display_color[1] != 0.0f ||
                desc->display_color[2] != 0.0f) {
                m->color[0] = desc->display_color[0];
                m->color[1] = desc->display_color[1];
                m->color[2] = desc->display_color[2];
            } else {
                m->color[0] = m->color[1] = m->color[2] = 0.7f;
            }
            if (desc->name) {
                m->name = strdup(desc->name);
            }
            r->scene_dirty = 1;
            invalidate_raster_instance_cache(r);
            return slot;
        }
    }

    geo_census_record(r, desc);

    /* Large USD loads append hundreds of thousands of meshes; scanning for
     * inactive holes per add makes attach O(N^2). nu_clear_scene resets
     * nmeshes, so append-only is the right load path. */
    int slot = append_mesh_slot(r);
    if (slot < 0) return NU_ERROR;

    RendererMesh* m = &r->meshes[slot];
    memset(m, 0, sizeof(RendererMesh));
    m->active = 1;
    m->visible = 1;
    m->env_mask = 0xFFu;
    m->nvertices = desc->nvertices;
    m->nindices = desc->nindices;
    m->prototype_idx = slot;
    m->material_index = desc->material_index;

    if (desc->transform) {
        memcpy(m->transform, desc->transform, 16 * sizeof(float));
    } else {
        identity_matrix(m->transform);
    }

    /* Use explicit color if any channel is non-zero; otherwise default to grey.
     * The old per-channel check (> 0 ? color : 0.7) corrupted colors with zero
     * channels — e.g. pure red (1,0,0) became (1, 0.7, 0.7). */
    if (desc->display_color[0] != 0.0f || desc->display_color[1] != 0.0f || desc->display_color[2] != 0.0f) {
        m->color[0] = desc->display_color[0];
        m->color[1] = desc->display_color[1];
        m->color[2] = desc->display_color[2];
    } else {
        m->color[0] = 0.7f;
        m->color[1] = 0.7f;
        m->color[2] = 0.7f;
    }

    if (desc->name) {
        m->name = strdup(desc->name);
    }

    /* Append vertex data. The material path keeps the historical
     * 12-float layout (pos, normal, pad, uv, matID); the geometry-only path
     * uses 9 floats (pos, normal, uv, matID) so raster inputs remain defined
     * without paying for unused material padding. */
    {
        uint32_t old_total = r->total_vertices;
        uint64_t new_total64 = (uint64_t)old_total + (uint64_t)desc->nvertices;
        if (new_total64 > UINT32_MAX) return NU_ERROR;
        uint32_t new_total = (uint32_t)new_total64;
        if (!reserve_cpu_geometry(r, new_total, r->cpu_index_capacity))
            return NU_ERROR;

        uint32_t stride = r->vertex_stride_floats ? r->vertex_stride_floats : 12u;
        uint32_t uv_offset = (stride >= 12u) ? 9u : 6u;
        uint32_t mat_offset = (stride >= 12u) ? 11u : 8u;
        float* dst = r->cpu_vertices + (size_t)old_total * stride;
        uint32_t mat_id_u32 = (desc->material_index >= 0)
            ? (uint32_t)desc->material_index : 0u;
        float mat_id_as_float;
        memcpy(&mat_id_as_float, &mat_id_u32, sizeof(float));

        for (int i = 0; i < desc->nvertices; i++) {
            float* v = dst + (size_t)i * stride;
            v[0] = desc->positions[i * 3 + 0];
            v[1] = desc->positions[i * 3 + 1];
            v[2] = desc->positions[i * 3 + 2];
            if (desc->normals) {
                v[3] = desc->normals[i * 3 + 0];
                v[4] = desc->normals[i * 3 + 1];
                v[5] = desc->normals[i * 3 + 2];
            } else {
                v[3] = 0.0f; v[4] = 1.0f; v[5] = 0.0f;
            }
            if (stride >= 12u) {
                v[6] = 0.0f; v[7] = 0.0f; v[8] = 0.0f;
            }
            if (desc->texcoords) {
                v[uv_offset + 0] = desc->texcoords[i * 2 + 0];
                v[uv_offset + 1] = 1.0f - desc->texcoords[i * 2 + 1];
            } else {
                v[uv_offset + 0] = 0.0f;
                v[uv_offset + 1] = 0.0f;
            }
            v[mat_offset] = mat_id_as_float;
        }

        m->vertex_offset = old_total;
        r->total_vertices = new_total;
    }

    /* Append index data to cpu buffer */
    {
        uint32_t old_total = r->total_indices;
        uint64_t new_total64 = (uint64_t)old_total + (uint64_t)desc->nindices;
        if (new_total64 > UINT32_MAX) return NU_ERROR;
        uint32_t new_total = (uint32_t)new_total64;
        if (!reserve_cpu_geometry(r, r->cpu_vertex_capacity, new_total))
            return NU_ERROR;

        memcpy(r->cpu_indices + old_total, desc->indices,
               (size_t)desc->nindices * sizeof(uint32_t));

        /* Lossless local reorder. meshopt requires triangle-list indices
         * within [0, nvertices); malformed DSX prototypes must skip the perf
         * path instead of handing invalid input to the optimizer. */
        int meshopt_safe = (desc->nvertices > 0 &&
                            desc->nindices >= 3 &&
                            desc->nindices % 3 == 0);
        if (meshopt_safe) {
            const uint32_t* idx = r->cpu_indices + old_total;
            uint32_t nv = (uint32_t)desc->nvertices;
            for (int j = 0; j < desc->nindices; j++) {
                if (idx[j] >= nv) {
                    meshopt_safe = 0;
                    break;
                }
            }
        }
        if (meshopt_safe) {
            meshopt_optimizeVertexCache(r->cpu_indices + old_total,
                                        r->cpu_indices + old_total,
                                        (size_t)desc->nindices,
                                        (size_t)desc->nvertices);
        }

        m->index_offset = old_total;
        r->total_indices = new_total;

        if (meshopt_safe && r->meshlets_enabled) {
            uint32_t stride = r->vertex_stride_floats ? r->vertex_stride_floats : 12u;
            const float* vertices = r->cpu_vertices + (size_t)m->vertex_offset * stride;
            if (!build_meshlets_for_prototype(r, m,
                    r->cpu_indices + old_total,
                    (size_t)desc->nindices,
                    vertices,
                    (size_t)desc->nvertices,
                    (size_t)stride * sizeof(float))) {
                set_error(r, "Failed to build meshlets");
                return NU_ERROR;
            }
        }
    }

    r->scene_dirty = 1;
    invalidate_raster_instance_cache(r);

    if (geo_dedup_enabled()) {
        geo_dedup_insert(r, desc, slot);
    }

    return slot;
}

int nu_add_mesh_instance(NuRenderer* r, int prototype_mesh_id, const float transform[16])
{
    if (!r) return NU_ERROR;
    if (r->cpu_staging_released) {
        set_error(r, "Cannot add mesh instance after CPU staging was released");
        return NU_ERROR_UNSUPPORTED;
    }
    if (prototype_mesh_id < 0 || prototype_mesh_id >= r->nmeshes)
        return NU_ERROR_BAD_ID;
    if (!r->meshes[prototype_mesh_id].active)
        return NU_ERROR_BAD_ID;

    /* Resolve to the root prototype (in case prototype_mesh_id is itself an instance) */
    RendererMesh proto_copy = r->meshes[prototype_mesh_id];
    int root_proto = proto_copy.prototype_idx;

    int slot = append_mesh_slot(r);
    if (slot < 0) return NU_ERROR;

    RendererMesh* m = &r->meshes[slot];
    memset(m, 0, sizeof(RendererMesh));
    m->active = 1;
    m->visible = 1;
    m->env_mask = proto_copy.env_mask ? proto_copy.env_mask : 0xFFu;

    /* Share geometry from the prototype — no vertex/index data copy */
    m->nvertices    = proto_copy.nvertices;
    m->nindices     = proto_copy.nindices;
    m->vertex_offset = proto_copy.vertex_offset;
    m->index_offset  = proto_copy.index_offset;
    m->meshlet_offset = proto_copy.meshlet_offset;
    m->meshlet_count = proto_copy.meshlet_count;
    m->meshlet_index_offset = proto_copy.meshlet_index_offset;
    m->meshlet_index_count = proto_copy.meshlet_index_count;
    m->prototype_idx = root_proto;
    m->material_index = proto_copy.material_index;

    /* Copy display color from prototype */
    m->color[0] = proto_copy.color[0];
    m->color[1] = proto_copy.color[1];
    m->color[2] = proto_copy.color[2];

    /* Set instance transform */
    if (transform) {
        memcpy(m->transform, transform, 16 * sizeof(float));
    } else {
        identity_matrix(m->transform);
    }

    /* Mark scene dirty — needs TLAS rebuild but NOT new BLAS */
    r->scene_dirty = 1;
    invalidate_raster_instance_cache(r);

    return slot;
}

NuResult nu_remove_mesh(NuRenderer* r, int mesh_id)
{
    if (!r || mesh_id < 0 || mesh_id >= r->nmeshes)
        return NU_ERROR_BAD_ID;
    if (r->cpu_staging_released) {
        set_error(r, "Cannot remove mesh after CPU staging was released");
        return NU_ERROR_UNSUPPORTED;
    }
    if (!r->meshes[mesh_id].active)
        return NU_ERROR_BAD_ID;

    free(r->meshes[mesh_id].name);
    r->meshes[mesh_id].active = 0;
    r->scene_dirty = 1;
    r->accel_dirty = 1;  /* BLAS/TLAS must be rebuilt without this mesh */
    r->rt_blas_valid = 0;
    invalidate_raster_instance_cache(r);

    return NU_OK;
}

void nu_clear_scene(NuRenderer* r)
{
    if (!r) return;

    geo_census_reset(&r->geo_census);

    for (int i = 0; i < r->nmeshes; i++) {
        free(r->meshes[i].name);
        r->meshes[i].active = 0;
    }
    r->nmeshes = 0;

    free(r->cpu_vertices); r->cpu_vertices = NULL;
    free(r->cpu_indices);  r->cpu_indices = NULL;
    free(r->cpu_meshlet_indices); r->cpu_meshlet_indices = NULL;
    free(r->meshlets); r->meshlets = NULL;
    r->cpu_vertex_capacity = 0;
    r->cpu_index_capacity = 0;
    r->cpu_meshlet_index_capacity = 0;
    r->meshlet_capacity = 0;
    r->total_vertices = 0;
    r->total_indices = 0;
    r->total_meshlet_indices = 0;
    r->meshlet_count = 0;
    r->meshlet_raster_replaces_index_buffer = 0;
    r->loaded_from_geometry_cache = 0;
    r->cpu_staging_released = 0;
    r->has_cached_bounds = 0;

    if (r->vertex_buffer) { gpu_destroy_buffer(r->gpu, r->vertex_buffer); r->vertex_buffer = NULL; }
    if (r->index_buffer)  { gpu_destroy_buffer(r->gpu, r->index_buffer);  r->index_buffer = NULL; }
    if (r->meshlet_index_buffer) {
        gpu_destroy_buffer(r->gpu, r->meshlet_index_buffer);
        r->meshlet_index_buffer = NULL;
    }
    if (r->instance_buffer) {
        gpu_destroy_buffer(r->gpu, r->instance_buffer);
        r->instance_buffer = NULL;
        r->instance_buffer_capacity = 0;
    }
    invalidate_raster_instance_cache(r);

    free(r->curves.segments); r->curves.segments = NULL;
    free(r->curves.aabbs);    r->curves.aabbs    = NULL;
    free(r->curves.colors);   r->curves.colors   = NULL;
    r->curves.nseg = 0;
    r->curves_dirty = 0;

    free(r->pi_batches);
    r->pi_batches = NULL;
    r->npi_batches = 0;
    free(r->pi_transforms_owned);
    r->pi_transforms_owned = NULL;
    r->npi_transforms_total = 0;

    if (r->gs) gs_scene_clear_particles(r->gs);
    r->gs_dirty = 0;

    gpu_gs_destroy(r->gpu);
    gpu_destroy_rt_scene(r->gpu);
    gpu_upload_lights(r->gpu, NULL, 0);

    r->scene_dirty = 0;
    r->accel_dirty = 0;
    r->has_frame = 0;
    memset(&r->timings, 0, sizeof(r->timings));

    r->lazy_pending = 0;
    if (r->usd_scene) { scene_free(r->usd_scene); r->usd_scene = NULL; }
}

NuResult nu_save_geometry_cache(NuRenderer* r, const char* path)
{
    if (!r || !path) return NU_ERROR;
    if (r->enable_materials) {
        set_error(r, "Geometry cache does not serialize materials yet");
        return NU_ERROR_UNSUPPORTED;
    }
    if (r->cpu_staging_released) {
        set_error(r, "Cannot save geometry cache after CPU staging was released");
        return NU_ERROR_UNSUPPORTED;
    }
    if (!r->cpu_vertices || !r->cpu_indices ||
        r->total_vertices == 0 || r->total_indices == 0) {
        set_error(r, "No geometry to save");
        return NU_ERROR_NO_SCENE;
    }

    FILE* f = fopen(path, "wb");
    if (!f) {
        set_error(r, "Failed to open geometry cache for writing");
        return NU_ERROR;
    }

    uint32_t stride = r->vertex_stride_floats ? r->vertex_stride_floats : 12u;
    int omit_original_indices = 0;
    int omit_names = env_flag_enabled("NUSD_GEOMETRY_CACHE_OMIT_NAMES");
    if (env_flag_enabled("NUSD_MESHLET_CACHE_OMIT_ORIGINAL_INDICES")) {
        if (!r->meshlet_count || !r->cpu_meshlet_indices) {
            set_error(r, "Cannot omit original indices without meshlet data");
            return NU_ERROR_UNSUPPORTED;
        }
        if (!bake_meshlet_fallback_indices_from_cpu(r)) {
            set_error(r, "Cannot omit original indices because meshlets do not cover all active meshes");
            return NU_ERROR_UNSUPPORTED;
        }
        omit_original_indices = 1;
    }

    uint64_t vertices_bytes = (uint64_t)r->total_vertices * stride * sizeof(float);
    uint64_t indices_bytes = omit_original_indices
        ? 0u
        : (uint64_t)r->total_indices * sizeof(uint32_t);
    uint64_t meshlets_bytes = (uint64_t)r->meshlet_count * sizeof(NuGeometryCacheMeshletRecord);
    uint64_t meshlet_indices_bytes = (uint64_t)r->total_meshlet_indices * sizeof(uint32_t);
    uint64_t mesh_records_bytes = 0;

    for (int i = 0; i < r->nmeshes; i++) {
        size_t name_len = (!omit_names && r->meshes[i].name) ? strlen(r->meshes[i].name) : 0;
        mesh_records_bytes += sizeof(NuGeometryCacheMeshRecord) + (uint64_t)name_len;
    }

    NuGeometryCacheHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, NUSD_GEOMETRY_CACHE_MAGIC, sizeof(h.magic));
    h.version = NUSD_GEOMETRY_CACHE_VERSION;
    h.header_size = (uint32_t)sizeof(h);
    h.vertex_stride_floats = stride;
    h.flags = (r->meshlet_count > 0) ? NUSD_GEOMETRY_CACHE_FLAG_MESHLETS : 0u;
    if (omit_original_indices)
        h.flags |= NUSD_GEOMETRY_CACHE_FLAG_MESHLET_INDEX_ONLY;
    if (omit_names)
        h.flags |= NUSD_GEOMETRY_CACHE_FLAG_OMIT_NAMES;
    h.nmeshes = (uint32_t)r->nmeshes;
    h.total_vertices = r->total_vertices;
    h.total_indices = r->total_indices;
    h.meshlet_count = r->meshlet_count;
    h.total_meshlet_indices = r->total_meshlet_indices;
    h.vertices_bytes = vertices_bytes;
    h.indices_bytes = indices_bytes;
    h.meshlets_bytes = meshlets_bytes;
    h.meshlet_indices_bytes = meshlet_indices_bytes;
    h.mesh_records_bytes = mesh_records_bytes;
    if (!renderer_geometry_bounds(r, h.bounds_min, h.bounds_max)) {
        h.bounds_min[0] = h.bounds_min[1] = h.bounds_min[2] = -1.0f;
        h.bounds_max[0] = h.bounds_max[1] = h.bounds_max[2] =  1.0f;
    }

    int ok = 1;
    ok = ok && write_exact(f, &h, sizeof(h));
    ok = ok && write_exact(f, r->cpu_vertices, (size_t)vertices_bytes);
    if (!omit_original_indices)
        ok = ok && write_exact(f, r->cpu_indices, (size_t)indices_bytes);

    for (uint32_t i = 0; ok && i < r->meshlet_count; i++) {
        RendererMeshlet* src = &r->meshlets[i];
        NuGeometryCacheMeshletRecord rec;
        memset(&rec, 0, sizeof(rec));
        rec.prototype_idx = src->prototype_idx;
        rec.index_offset = src->index_offset;
        rec.index_count = src->index_count;
        memcpy(rec.center, src->center, 3 * sizeof(float));
        rec.radius = src->radius;
        memcpy(rec.cone_axis, src->cone_axis, 3 * sizeof(float));
        rec.cone_cutoff = src->cone_cutoff;
        ok = ok && write_exact(f, &rec, sizeof(rec));
    }

    ok = ok && write_exact(f, r->cpu_meshlet_indices, (size_t)meshlet_indices_bytes);

    for (int i = 0; ok && i < r->nmeshes; i++) {
        RendererMesh* src = &r->meshes[i];
        NuGeometryCacheMeshRecord rec;
        memset(&rec, 0, sizeof(rec));
        rec.active = (uint32_t)src->active;
        rec.visible = (uint32_t)src->visible;
        memcpy(rec.transform, src->transform, sizeof(rec.transform));
        memcpy(rec.color, src->color, sizeof(rec.color));
        rec.nvertices = (uint32_t)src->nvertices;
        rec.nindices = (uint32_t)src->nindices;
        rec.vertex_offset = src->vertex_offset;
        rec.index_offset = src->index_offset;
        rec.meshlet_offset = src->meshlet_offset;
        rec.meshlet_count = src->meshlet_count;
        rec.meshlet_index_offset = src->meshlet_index_offset;
        rec.meshlet_index_count = src->meshlet_index_count;
        rec.prototype_idx = src->prototype_idx;
        rec.material_index = src->material_index;
        rec.name_len = (!omit_names && src->name) ? (uint32_t)strlen(src->name) : 0u;
        ok = ok && write_exact(f, &rec, sizeof(rec));
        ok = ok && write_exact(f, src->name, rec.name_len);
    }

    if (fclose(f) != 0) ok = 0;
    if (!ok) {
        set_error(r, "Failed to write geometry cache");
        return NU_ERROR;
    }

    return NU_OK;
}

int nu_load_geometry_cache(NuRenderer* r, const char* path)
{
    if (!r || !path) return NU_ERROR;
    if (r->enable_materials) {
        set_error(r, "Geometry cache does not serialize materials yet");
        return NU_ERROR_UNSUPPORTED;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        set_error(r, "Failed to open geometry cache");
        return NU_ERROR;
    }

    NuGeometryCacheHeader h;
    if (!read_exact(f, &h, sizeof(h))) {
        fclose(f);
        set_error(r, "Failed to read geometry cache header");
        return NU_ERROR;
    }

    if (memcmp(h.magic, NUSD_GEOMETRY_CACHE_MAGIC, sizeof(h.magic)) != 0 ||
        h.version != NUSD_GEOMETRY_CACHE_VERSION ||
        h.header_size != sizeof(h)) {
        fclose(f);
        set_error(r, "Invalid geometry cache header");
        return NU_ERROR;
    }

    uint32_t stride = r->vertex_stride_floats ? r->vertex_stride_floats : 12u;
    if (h.vertex_stride_floats != stride) {
        fclose(f);
        set_error(r, "Geometry cache vertex layout does not match renderer");
        return NU_ERROR_UNSUPPORTED;
    }
    int meshlet_index_only = (h.flags & NUSD_GEOMETRY_CACHE_FLAG_MESHLET_INDEX_ONLY) != 0;
    if (h.nmeshes > (uint32_t)INT_MAX ||
        (h.flags & ~(NUSD_GEOMETRY_CACHE_FLAG_MESHLETS |
                     NUSD_GEOMETRY_CACHE_FLAG_MESHLET_INDEX_ONLY |
                     NUSD_GEOMETRY_CACHE_FLAG_OMIT_NAMES)) != 0 ||
        h.vertices_bytes != (uint64_t)h.total_vertices * stride * sizeof(float) ||
        h.indices_bytes != (meshlet_index_only
            ? 0u
            : (uint64_t)h.total_indices * sizeof(uint32_t)) ||
        h.meshlets_bytes != (uint64_t)h.meshlet_count * sizeof(NuGeometryCacheMeshletRecord) ||
        h.meshlet_indices_bytes != (uint64_t)h.total_meshlet_indices * sizeof(uint32_t) ||
        h.mesh_records_bytes < (uint64_t)h.nmeshes * sizeof(NuGeometryCacheMeshRecord) ||
        (meshlet_index_only && ((h.flags & NUSD_GEOMETRY_CACHE_FLAG_MESHLETS) == 0 ||
                                h.total_meshlet_indices == 0)) ||
        ((h.flags & NUSD_GEOMETRY_CACHE_FLAG_MESHLETS) == 0 &&
            (h.meshlet_count != 0 || h.total_meshlet_indices != 0))) {
        fclose(f);
        set_error(r, "Geometry cache sizes are invalid");
        return NU_ERROR;
    }

    int skip_original_indices = (r->meshlet_raster_enabled &&
                                 !r->enable_rt &&
                                 (h.flags & NUSD_GEOMETRY_CACHE_FLAG_MESHLETS) &&
                                 h.total_meshlet_indices > 0);
    if (meshlet_index_only && !skip_original_indices) {
        fclose(f);
        set_error(r, "Geometry cache requires meshlet raster to omit original indices");
        return NU_ERROR_UNSUPPORTED;
    }
    int direct_gpu_upload = (skip_original_indices &&
                             r->release_cpu_staging_after_upload &&
                             !env_flag_enabled("NUSD_DISABLE_DIRECT_CACHE_GPU_UPLOAD"));

    if (direct_gpu_upload) {
        int active_count = load_geometry_cache_direct_meshlet_gpu(r, f, &h);
        if (fclose(f) != 0 && active_count >= 0) {
            nu_clear_scene(r);
            set_error(r, "Failed to close geometry cache");
            return NU_ERROR;
        }
        return active_count >= 0 ? active_count : NU_ERROR;
    }

    nu_clear_scene(r);

    if (!ensure_mesh_capacity(r, (int)h.nmeshes) ||
        !reserve_cpu_vertices_only(r, h.total_vertices) ||
        (!skip_original_indices && !reserve_cpu_indices_only(r, h.total_indices)) ||
        !reserve_meshlet_data(r, h.meshlet_count, h.total_meshlet_indices)) {
        fclose(f);
        nu_clear_scene(r);
        set_error(r, "Failed to allocate geometry cache");
        return NU_ERROR;
    }

    if (!read_exact(f, r->cpu_vertices, (size_t)h.vertices_bytes) ||
        (skip_original_indices
            ? !skip_exact(f, h.indices_bytes)
            : !read_exact(f, r->cpu_indices, (size_t)h.indices_bytes))) {
        fclose(f);
        nu_clear_scene(r);
        set_error(r, "Failed to read geometry cache buffers");
        return NU_ERROR;
    }
    r->total_vertices = h.total_vertices;
    r->total_indices = h.total_indices;

    for (uint32_t i = 0; i < h.meshlet_count; i++) {
        NuGeometryCacheMeshletRecord rec;
        if (!read_exact(f, &rec, sizeof(rec))) {
            fclose(f);
            nu_clear_scene(r);
            set_error(r, "Failed to read geometry cache meshlets");
            return NU_ERROR;
        }
        RendererMeshlet* dst = &r->meshlets[i];
        if (rec.index_offset > h.total_meshlet_indices ||
            rec.index_count > h.total_meshlet_indices - rec.index_offset ||
            rec.prototype_idx >= h.nmeshes) {
            fclose(f);
            nu_clear_scene(r);
            set_error(r, "Geometry cache meshlet range is invalid");
            return NU_ERROR;
        }
        dst->prototype_idx = rec.prototype_idx;
        dst->index_offset = rec.index_offset;
        dst->index_count = rec.index_count;
        memcpy(dst->center, rec.center, 3 * sizeof(float));
        dst->radius = rec.radius;
        memcpy(dst->cone_axis, rec.cone_axis, 3 * sizeof(float));
        dst->cone_cutoff = rec.cone_cutoff;
    }
    r->meshlet_count = h.meshlet_count;
    r->total_meshlet_indices = h.total_meshlet_indices;
    if (!read_exact(f, r->cpu_meshlet_indices, (size_t)h.meshlet_indices_bytes)) {
        fclose(f);
        nu_clear_scene(r);
        set_error(r, "Failed to read geometry cache meshlet indices");
        return NU_ERROR;
    }

    r->nmeshes = (int)h.nmeshes;
    int active_count = 0;
    uint64_t records_left = h.mesh_records_bytes;
    for (uint32_t i = 0; i < h.nmeshes; i++) {
        NuGeometryCacheMeshRecord rec;
        if (records_left < sizeof(rec)) {
            fclose(f);
            nu_clear_scene(r);
            set_error(r, "Geometry cache mesh record table is invalid");
            return NU_ERROR;
        }
        if (!read_exact(f, &rec, sizeof(rec))) {
            fclose(f);
            nu_clear_scene(r);
            set_error(r, "Failed to read geometry cache mesh records");
            return NU_ERROR;
        }
        records_left -= sizeof(rec);
        if (rec.nvertices > (uint32_t)INT_MAX ||
            rec.nindices > (uint32_t)INT_MAX ||
            rec.vertex_offset > h.total_vertices ||
            rec.nvertices > h.total_vertices - rec.vertex_offset ||
            rec.index_offset > h.total_indices ||
            rec.nindices > h.total_indices - rec.index_offset ||
            rec.meshlet_offset > h.meshlet_count ||
            rec.meshlet_count > h.meshlet_count - rec.meshlet_offset ||
            rec.meshlet_index_offset > h.total_meshlet_indices ||
            rec.meshlet_index_count > h.total_meshlet_indices - rec.meshlet_index_offset ||
            rec.name_len > records_left ||
            rec.prototype_idx >= (int32_t)h.nmeshes) {
            fclose(f);
            nu_clear_scene(r);
            set_error(r, "Geometry cache mesh record range is invalid");
            return NU_ERROR;
        }
        RendererMesh* dst = &r->meshes[i];
        memset(dst, 0, sizeof(*dst));
        dst->active = rec.active ? 1 : 0;
        dst->visible = rec.visible ? 1 : 0;
        dst->env_mask = 0xFFu;
        memcpy(dst->transform, rec.transform, sizeof(dst->transform));
        memcpy(dst->color, rec.color, sizeof(dst->color));
        dst->nvertices = (int)rec.nvertices;
        dst->nindices = (int)rec.nindices;
        dst->vertex_offset = rec.vertex_offset;
        dst->index_offset = rec.index_offset;
        dst->meshlet_offset = rec.meshlet_offset;
        dst->meshlet_count = rec.meshlet_count;
        dst->meshlet_index_offset = rec.meshlet_index_offset;
        dst->meshlet_index_count = rec.meshlet_index_count;
        dst->prototype_idx = rec.prototype_idx;
        dst->material_index = rec.material_index;
        if (rec.name_len > 0) {
            dst->name = (char*)malloc((size_t)rec.name_len + 1u);
            if (!dst->name || !read_exact(f, dst->name, rec.name_len)) {
                fclose(f);
                nu_clear_scene(r);
                set_error(r, "Failed to read geometry cache mesh name");
                return NU_ERROR;
            }
            dst->name[rec.name_len] = '\0';
        }
        records_left -= rec.name_len;
        if (dst->active) active_count++;
    }
    if (records_left != 0) {
        fclose(f);
        nu_clear_scene(r);
        set_error(r, "Geometry cache mesh record table has trailing data");
        return NU_ERROR;
    }

    if (meshlet_index_only && !meshlet_raster_covers_active_meshes(r)) {
        fclose(f);
        nu_clear_scene(r);
        set_error(r, "Geometry cache omitted original indices but meshlets do not cover all active meshes");
        return NU_ERROR_UNSUPPORTED;
    }

    if (skip_original_indices && !meshlet_raster_covers_active_meshes(r)) {
        if (!synthesize_meshlet_fallbacks_from_cache_indices(f, &h, r) ||
            !meshlet_raster_covers_active_meshes(r)) {
            fclose(f);
            nu_clear_scene(r);
            set_error(r, "Geometry cache meshlets do not cover all active meshes");
            return NU_ERROR_UNSUPPORTED;
        }
    }

    if (fclose(f) != 0) {
        nu_clear_scene(r);
        set_error(r, "Failed to close geometry cache");
        return NU_ERROR;
    }

    renderer_cache_bounds(r, h.bounds_min, h.bounds_max);
    if (active_count > 0)
        renderer_auto_frame_bounds(r, h.bounds_min, h.bounds_max);
    r->meshlets_enabled = r->meshlets_enabled || ((h.flags & NUSD_GEOMETRY_CACHE_FLAG_MESHLETS) != 0);
    r->loaded_from_geometry_cache = 1;
    r->cpu_staging_released = 0;
    r->scene_dirty = 1;
    r->accel_dirty = 1;
    r->rt_blas_valid = 0;
    r->has_frame = 0;

    return active_count;
}

/* Internal: hand a pre-built Scene to the renderer. Takes ownership of the
 * Scene struct and drives the GPU upload + mesh registration + auto-framing.
 * Both nu_load_usd (file path) and nu_load_usd_from_handle (borrowed stage)
 * funnel into this function. `label` is used purely for the final log line. */
static int nu_attach_scene(NuRenderer* r, Scene* scene, const char* label)
{
    r->loaded_from_geometry_cache = 0;
    r->cpu_staging_released = 0;
    r->usd_scene = scene;
    renderer_cache_bounds(r, scene->bounds_min, scene->bounds_max);

    if (scene && scene->nmeshes > 0 &&
        scene->meshes[0].positions == NULL &&
        scene->meshes[0].lazy_prim_idx >= 0) {
        r->lazy_pending = 1;
        fprintf(stderr,
                "nusd_renderer: lazy scene attached (%d metadata meshes) — "
                "call extract_deferred() to materialize geometry.\n",
                scene->nmeshes);
        return scene->nmeshes;
    }

    /* Upload materials to GPU before BLAS build (RT descriptor set
     * references mat_ssbo_buf + texture array, which must exist). */
    MaterialCollection* mc = (MaterialCollection*)scene->materials;
    if (mc && mc->nmaterials > 0) {
        GpuMaterialParams* gpu_mats = (GpuMaterialParams*)calloc(
            (size_t)mc->nmaterials, sizeof(GpuMaterialParams));
        for (int i = 0; i < mc->nmaterials; i++) {
            const MaterialParams* mp = &mc->materials[i].params;
            GpuMaterialParams* gm = &gpu_mats[i];
            memcpy(gm->base_color,     mp->base_color,     4 * sizeof(float));
            memcpy(gm->emissive_color, mp->emissive_color, 4 * sizeof(float));
            gm->metallic            = mp->metallic;
            gm->roughness           = mp->roughness;
            gm->opacity             = mp->opacity;
            gm->ior                 = mp->ior;
            gm->occlusion           = mp->occlusion;
            gm->clearcoat           = mp->clearcoat;
            gm->clearcoat_roughness = mp->clearcoat_roughness;
            gm->normal_scale        = mp->normal_scale;
            memcpy(gm->tex_indices, mp->tex_indices, 8 * sizeof(int));
            gm->use_vertex_color = mp->use_vertex_color;
            gm->udim_scale_u = (mp->udim_scale_u > 0.0f) ? mp->udim_scale_u : 1.0f;
            gm->udim_scale_v = (mp->udim_scale_v > 0.0f) ? mp->udim_scale_v : 1.0f;
            gm->opacity_threshold = mp->opacity_threshold;
            /* Phase 7c trailing block — Standard Surface SSS + transmission.
             * Forwarding these was missing on the Metal port (port from
             * Vulkan 343d937). Without it, transmission_weight stays at
             * 0 on the GPU even when the .mtlx authored 1, so e.g. the
             * chess Pawn-tops never enter the transmission code path
             * and render as opaque white spheres. */
            memcpy(gm->subsurface_color,   mp->subsurface_color,   4 * sizeof(float));
            memcpy(gm->subsurface_radius,  mp->subsurface_radius,  4 * sizeof(float));
            memcpy(gm->transmission_color, mp->transmission_color, 4 * sizeof(float));
            gm->subsurface_weight       = mp->subsurface_weight;
            gm->subsurface_scale        = mp->subsurface_scale;
            gm->transmission_weight     = mp->transmission_weight;
            gm->transmission_ior        = mp->transmission_ior;
            gm->tex_subsurface_weight   = mp->tex_subsurface_weight;
            gm->tex_transmission_weight = mp->tex_transmission_weight;
            gm->sss_color_authored      = mp->sss_color_authored;
            gm->use_specular_workflow   = mp->use_specular_workflow;
            memcpy(gm->specular_color, mp->specular_color, 4 * sizeof(float));
            memcpy(gm->normal_tex_scale, mp->normal_tex_scale, 4 * sizeof(float));
            memcpy(gm->normal_tex_bias,  mp->normal_tex_bias,  4 * sizeof(float));
            memcpy(gm->uv_transform, mp->uv_transform, 4 * sizeof(float));
            memcpy(gm->roughness_tex_transform, mp->roughness_tex_transform, 4 * sizeof(float));
            gm->v_flip                 = mp->v_flip;
            gm->base_weight             = mp->base_weight;
            gm->specular_weight         = mp->specular_weight;
            gm->sheen_weight            = mp->sheen_weight;
            gm->sheen_roughness         = mp->sheen_roughness;
            memcpy(gm->sheen_color, mp->sheen_color, 4 * sizeof(float));
            gm->thin_film_thickness     = mp->thin_film_thickness;
            gm->thin_film_ior           = mp->thin_film_ior;
            gm->specular_anisotropy     = mp->specular_anisotropy;
            gm->standard_surface_lobes  = mp->standard_surface_lobes;
            gm->procedural_kind              = mp->procedural_kind;
            gm->procedural_base_color        = mp->procedural_base_color;
            gm->procedural_subsurface_color  = mp->procedural_subsurface_color;
            gm->procedural_octaves           = mp->procedural_octaves;
            memcpy(gm->procedural_color1, mp->procedural_color1, 4 * sizeof(float));
            memcpy(gm->procedural_color2, mp->procedural_color2, 4 * sizeof(float));
            memcpy(gm->procedural_params, mp->procedural_params, 4 * sizeof(float));
            gm->procedural_node_count = mp->procedural_node_count;
            gm->procedural_base_color_output = mp->procedural_base_color_output;
            gm->procedural_subsurface_color_output = mp->procedural_subsurface_color_output;
            gm->procedural_roughness_output = mp->procedural_roughness_output;
            gm->procedural_normal_output = mp->procedural_normal_output;
            gm->procedural_graph_flags = mp->procedural_graph_flags;
            gm->procedural_graph_pad0 = mp->procedural_graph_pad0;
            gm->procedural_graph_pad1 = mp->procedural_graph_pad1;
            memcpy(gm->procedural_nodes, mp->procedural_nodes,
                   sizeof(gm->procedural_nodes));
        }

        GpuTextureData* gpu_texs = NULL;
        if (mc->ntextures > 0) {
            gpu_texs = (GpuTextureData*)calloc(
                (size_t)mc->ntextures, sizeof(GpuTextureData));
            for (int i = 0; i < mc->ntextures; i++) {
                gpu_texs[i].pixels  = mc->textures[i].pixels;
                gpu_texs[i].width   = mc->textures[i].width;
                gpu_texs[i].height  = mc->textures[i].height;
                /* Per-texture sRGB decided by material loader based on which
                 * slots reference this texture (color vs data). Replaces the
                 * old unconditional `is_srgb = 1` that corrupted normal /
                 * roughness / metallic / AO maps with hardware sRGB->linear
                 * sampling on linear-encoded data. */
                gpu_texs[i].is_srgb = mc->textures[i].is_srgb;
            }
        }

        gpu_upload_materials(r->gpu,
                             gpu_mats, mc->nmaterials,
                             gpu_texs, mc->ntextures);

        free(gpu_mats);
        free(gpu_texs);
    }

    /* Upload scene lights to GPU SSBO (binding 13). Always upload — even a
     * zero-light scene needs a valid SSBO with nlights=0 so the closest-hit
     * shader doesn't read garbage from the fallback buffer binding. */
    if (scene->nlights > 0) {
        GpuLight* gpu_lights = (GpuLight*)calloc(
            (size_t)scene->nlights, sizeof(GpuLight));
        for (int i = 0; i < scene->nlights; i++) {
            const SceneLight* sl = &scene->lights[i];
            GpuLight* gl = &gpu_lights[i];
            memcpy(gl->position, sl->position, 3 * sizeof(float));
            memcpy(gl->normal,   sl->normal,   3 * sizeof(float));
            memcpy(gl->u_axis,   sl->u_axis,   3 * sizeof(float));
            memcpy(gl->v_axis,   sl->v_axis,   3 * sizeof(float));
            memcpy(gl->color,    sl->color,    3 * sizeof(float));
            gl->intensity = sl->intensity;
            gl->kind      = sl->kind;
            gl->normalize = sl->normalize;
            gl->angle_deg = sl->angle_deg;
        }
        gpu_upload_lights(r->gpu, gpu_lights, scene->nlights);
        free(gpu_lights);
        fprintf(stderr, "nusd_renderer: uploaded %d lights to GPU\n", scene->nlights);
    } else {
        gpu_upload_lights(r->gpu, NULL, 0);
    }

    /* Add each mesh from the scene. Preserve scene-loader instancing:
     * prototype geometry is uploaded once, and instance records become
     * renderer instances that share offsets/BLAS. */
    int loaded = 0;
    int* scene_to_renderer = (int*)malloc((size_t)scene->nmeshes * sizeof(int));
    if (!scene_to_renderer) {
        set_error(r, "Failed to allocate scene-to-renderer mesh map");
        return NU_ERROR;
    }
    for (int i = 0; i < scene->nmeshes; i++) scene_to_renderer[i] = -1;

    uint64_t reserve_vertices = 0;
    uint64_t reserve_indices = 0;
    for (int i = 0; i < scene->nmeshes; i++) {
        SceneMesh* sm = &scene->meshes[i];
        if (sm->nvertices <= 0 || sm->nindices <= 0) continue;
        if (sm->prototype_idx >= 0 && sm->prototype_idx != i &&
            sm->prototype_idx < scene->nmeshes) {
            continue;
        }
        reserve_vertices += (uint64_t)sm->nvertices;
        reserve_indices += (uint64_t)sm->nindices;
    }
    {
        uint64_t default_max_vertices =
            no_cull_all_geometry_requested() ? 0ULL : 50000000ULL;
        uint64_t default_max_indices =
            no_cull_all_geometry_requested() ? 0ULL : 250000000ULL;
        uint64_t max_vertices =
            env_u64_limit("NUSD_MAX_GEOM_VERTICES", default_max_vertices);
        uint64_t max_indices =
            env_u64_limit("NUSD_MAX_GEOM_INDICES", default_max_indices);
        if ((max_vertices > 0 && reserve_vertices > max_vertices) ||
            (max_indices > 0 && reserve_indices > max_indices)) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "Scene exceeds renderer geometry budget "
                     "(vertices=%llu limit=%llu, indices=%llu limit=%llu)",
                     (unsigned long long)reserve_vertices,
                     (unsigned long long)max_vertices,
                     (unsigned long long)reserve_indices,
                     (unsigned long long)max_indices);
            free(scene_to_renderer);
            set_error(r, msg);
            return NU_ERROR;
        }
    }
    if (!reserve_cpu_geometry(r, reserve_vertices, reserve_indices)) {
        free(scene_to_renderer);
        set_error(r, "Failed to reserve CPU geometry buffers");
        return NU_ERROR;
    }

    /* Geometry-accounting: every scene mesh record must end up in exactly one
     * bucket so we can prove no renderable geometry is silently dropped.
     * empty   = no geometry authored (nvertices/nindices <= 0)
     * failed  = nu_add_mesh / instance returned < 0 (capacity/overflow/alloc)
     * proto   = added but is_proto_only (renders via its instances, not itself)
     * visible = added and directly rendered (== loaded). */
    int skip_empty = 0, fail_add = 0, proto_only_added = 0, instanced_added = 0;
    for (int i = 0; i < scene->nmeshes; i++) {
        SceneMesh* sm = &scene->meshes[i];
        if (sm->nvertices <= 0 || sm->nindices <= 0) { skip_empty++; continue; }

        float xform[16];
        for (int j = 0; j < 16; j++)
            xform[j] = (float)sm->world_xform[j];

        int mesh_id = NU_ERROR;
        int via_instance = 0;
        if (sm->prototype_idx >= 0 && sm->prototype_idx != i &&
            sm->prototype_idx < scene->nmeshes &&
            scene_to_renderer[sm->prototype_idx] >= 0) {
            mesh_id = nu_add_mesh_instance(r, scene_to_renderer[sm->prototype_idx], xform);
            via_instance = 1;
        } else {
            NuMeshDesc desc;
            memset(&desc, 0, sizeof(desc));
            desc.positions      = sm->positions;
            desc.normals        = sm->normals;
            desc.colors         = sm->colors;
            desc.texcoords      = sm->texcoords;
            desc.indices        = sm->indices;
            desc.nvertices      = sm->nvertices;
            desc.nindices       = sm->nindices;
            desc.material_index = sm->material_index;
            desc.transform      = xform;

            if (sm->has_display_color) {
                desc.display_color[0] = sm->display_color[0];
                desc.display_color[1] = sm->display_color[1];
                desc.display_color[2] = sm->display_color[2];
            } else {
                desc.display_color[0] = 0.7f;
                desc.display_color[1] = 0.7f;
                desc.display_color[2] = 0.7f;
            }

            if (sm->path[0]) desc.name = sm->path;

            mesh_id = nu_add_mesh(r, &desc);
        }

        if (mesh_id >= 0) {
            scene_to_renderer[i] = mesh_id;
            RendererMesh* rm = &r->meshes[mesh_id];
            if (sm->has_display_color) {
                rm->color[0] = sm->display_color[0];
                rm->color[1] = sm->display_color[1];
                rm->color[2] = sm->display_color[2];
            }
            if (sm->path[0]) {
                char* copy = strdup(sm->path);
                if (copy) {
                    free(rm->name);
                    rm->name = copy;
                }
            }
            if (via_instance) instanced_added++;
            if (sm->is_proto_only) {
                rm->visible = 0;
                proto_only_added++;
            } else {
                loaded++;
            }
        } else {
            /* mesh_id < 0: nu_add_mesh / instance rejected this record. This is
             * the only path that loses renderable geometry, so make it loud and
             * dump the first few offenders with their authored sizes. */
            fail_add++;
            if (fail_add <= 8) {
                fprintf(stderr,
                        "nusd_renderer: WARNING dropped mesh '%s' (idx %d, "
                        "nverts=%d nindices=%d): %s\n",
                        sm->path[0] ? sm->path : "<unnamed>", i,
                        sm->nvertices, sm->nindices, nu_get_last_error(r));
            }
        }
    }
    if (fail_add > 0) {
        fprintf(stderr,
                "nusd_renderer: ERROR %d scene mesh records were DROPPED "
                "(not rendered) — geometry is incomplete\n", fail_add);
    }
    if (fail_add > 0 || getenv("NUSD_LOAD_TIMING")) {
        fprintf(stderr,
                "nusd_renderer: geometry accounting — scene_records=%d, empty=%d, "
                "dropped=%d, proto_only(via instances)=%d, instanced=%d, visible=%d "
                "[sum=%d]\n",
                scene->nmeshes, skip_empty, fail_add, proto_only_added,
                instanced_added, loaded,
                skip_empty + fail_add + proto_only_added + loaded);
    }

    /* ---- Compact PointInstancer batches (Section 8B) ----
     * Borrow the transform slab from scene->pi_transforms; translate the
     * scene-mesh prototype index through scene_to_renderer. Batches whose
     * prototype failed to load (scene_to_renderer[idx] < 0) are dropped
     * with a single warning so the rest of the scene still renders. */
    if (scene->npi_batches > 0 && scene->npi_transforms > 0) {
        r->pi_batches = (RendererPiBatch*)calloc(
            (size_t)scene->npi_batches, sizeof(RendererPiBatch));
        if (!r->pi_batches) {
            free(scene_to_renderer);
            set_error(r, "Failed to allocate renderer PI batches");
            return NU_ERROR;
        }
        /* Renderer-owned deep copy of the compact transform slab. Required
         * because scene_free runs immediately when materials are disabled
         * (below) which would dangle scene->pi_transforms pointers. The
         * copy is 48 B/instance = 133 MiB at Moana scale — orders of
         * magnitude smaller than the per-clone SceneMesh+RendererMesh
         * record cost (~2.6 GiB) the §8B compact path replaces. */
        r->pi_transforms_owned = (SceneInstanceTransform*)malloc(
            (size_t)scene->npi_transforms * sizeof(SceneInstanceTransform));
        if (!r->pi_transforms_owned) {
            free(r->pi_batches);
            r->pi_batches = NULL;
            free(scene_to_renderer);
            set_error(r, "Failed to allocate renderer PI transform slab");
            return NU_ERROR;
        }
        memcpy(r->pi_transforms_owned, scene->pi_transforms,
               (size_t)scene->npi_transforms * sizeof(SceneInstanceTransform));

        int kept = 0;
        int dropped = 0;
        uint64_t total_transforms = 0;
        for (int b = 0; b < scene->npi_batches; b++) {
            const SceneInstanceBatch* sb = &scene->pi_batches[b];
            if (sb->prototype_mesh_idx < 0 ||
                sb->prototype_mesh_idx >= scene->nmeshes) {
                dropped++;
                continue;
            }
            int proto_ri = scene_to_renderer[sb->prototype_mesh_idx];
            if (proto_ri < 0) {
                dropped++;
                continue;
            }
            if ((uint64_t)sb->transform_offset + sb->transform_count >
                scene->npi_transforms) {
                dropped++;
                continue;
            }
            RendererPiBatch* rb = &r->pi_batches[kept++];
            rb->proto_renderer_idx = proto_ri;
            rb->count              = sb->transform_count;
            rb->transforms         = &r->pi_transforms_owned[sb->transform_offset];
            rb->source_prim_idx    = sb->source_prim_idx;
            rb->material_or_binding_id = sb->material_or_binding_id;
            {
                const SceneMesh* spm = &scene->meshes[sb->prototype_mesh_idx];
                float mn[3], mx[3];
                /* Stored bounds can be left uninitialized (FLT_MAX/-FLT_MAX)
                 * for proto-only meshes under NUSD_GEOMETRY_ONLY — which would
                 * make proto_radius ~inf and frustum-cull every batch. Fall
                 * back to computing the AABB from the proto positions. */
                int degenerate = !(spm->bounds_max[0] >= spm->bounds_min[0] &&
                                   spm->bounds_max[1] >= spm->bounds_min[1] &&
                                   spm->bounds_max[2] >= spm->bounds_min[2]) ||
                                 spm->bounds_min[0] == FLT_MAX;
                if (degenerate && spm->positions && spm->nvertices > 0) {
                    mn[0]=mn[1]=mn[2]=FLT_MAX; mx[0]=mx[1]=mx[2]=-FLT_MAX;
                    for (int v = 0; v < spm->nvertices; v++) {
                        const float* p = &spm->positions[v*3];
                        for (int a = 0; a < 3; a++) {
                            if (p[a] < mn[a]) mn[a] = p[a];
                            if (p[a] > mx[a]) mx[a] = p[a];
                        }
                    }
                } else {
                    for (int a = 0; a < 3; a++) { mn[a]=spm->bounds_min[a]; mx[a]=spm->bounds_max[a]; }
                }
                for (int a = 0; a < 3; a++) rb->proto_center[a] = 0.5f * (mn[a] + mx[a]);
                float ex = mx[0]-mn[0], ey = mx[1]-mn[1], ez = mx[2]-mn[2];
                rb->proto_radius = 0.5f * sqrtf(ex*ex + ey*ey + ez*ez);
            }
            total_transforms += sb->transform_count;
        }
        r->npi_batches = kept;
        r->npi_transforms_total = total_transforms;
        if (dropped > 0) {
            fprintf(stderr,
                    "nusd_renderer: PI batch translation dropped %d/%d "
                    "batches (prototype mesh not loaded)\n",
                    dropped, scene->npi_batches);
        }
        fprintf(stderr,
                "nusd_renderer: compact PI: %d batches, %llu transforms "
                "(%llu B, renderer-owned)\n",
                r->npi_batches,
                (unsigned long long)r->npi_transforms_total,
                (unsigned long long)(scene->npi_transforms *
                                     sizeof(SceneInstanceTransform)));
    }

    free(scene_to_renderer);

    /* ---- Phase 11.A: extract curve segments + AABBs ----
     * Flat layout across all BasisCurves prims (Strategy A from
     * RENDERER_BIG_PLAN.md Phase 12.1). Per-segment color carries each
     * curve's display_color for now; Phase 12.2 will replace the color
     * array with a material_id SSBO + materials LUT. */
    if (scene->ncurves > 0) {
        int total_segs = 0;
        int skipped_curve_prims = 0;
        int skipped_curve_segments = 0;
        int default_curve_segments =
            no_cull_all_geometry_requested() ? 0 : 8000000;
        int max_curve_segments =
            env_int_limit("NUSD_MAX_CURVE_SEGMENTS", default_curve_segments);
        for (int i = 0; i < scene->ncurves; i++) {
            int n = scene_curve_count_segments(&scene->curves[i]);
            if (n <= 0) continue;
            if (max_curve_segments > 0 &&
                total_segs > max_curve_segments - n) {
                skipped_curve_prims++;
                skipped_curve_segments += n;
                continue;
            }
            total_segs += n;
        }
        if (skipped_curve_prims > 0) {
            fprintf(stderr,
                    "nusd_renderer: curve segment cap kept %d segments, "
                    "skipped %d segments from %d BasisCurves "
                    "(NUSD_MAX_CURVE_SEGMENTS=%d; 0 disables cap)\n",
                    total_segs, skipped_curve_segments, skipped_curve_prims,
                    max_curve_segments);
            if (env_flag_requested("NUSD_FAIL_ON_LIMIT")) {
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "curve segment budget exceeded "
                         "(kept=%d skipped=%d limit=%d; raise "
                         "NUSD_MAX_CURVE_SEGMENTS or set it to 0)",
                         total_segs, skipped_curve_segments,
                         max_curve_segments);
                nu_clear_scene(r);
                set_error(r, msg);
                return NU_ERROR;
            }
        }

        if (total_segs > 0) {
            r->curves.segments = (SceneCurveSegment*)malloc((size_t)total_segs * sizeof(SceneCurveSegment));
            r->curves.aabbs    = (SceneCurveAabb*)   malloc((size_t)total_segs * sizeof(SceneCurveAabb));
            r->curves.colors   = (float*)            malloc((size_t)total_segs * 3 * sizeof(float));
            if (!r->curves.segments || !r->curves.aabbs || !r->curves.colors) {
                free(r->curves.segments); r->curves.segments = NULL;
                free(r->curves.aabbs);    r->curves.aabbs    = NULL;
                free(r->curves.colors);   r->curves.colors   = NULL;
                fprintf(stderr, "nusd_renderer: failed to allocate curve segment buffers (%d segments)\n", total_segs);
            } else {
                int written = 0;
                for (int i = 0; i < scene->ncurves; i++) {
                    const SceneCurve* c = &scene->curves[i];
                    int n_this = scene_curve_count_segments(c);
                    if (n_this == 0) continue;
                    if (written > total_segs - n_this) continue;
                    int got = scene_curve_to_segments_colored(c,
                        &r->curves.segments[written],
                        &r->curves.aabbs[written],
                        &r->curves.colors[written * 3]);
                    written += got;
                }
                r->curves.nseg  = written;
                r->curves_dirty = 1;
                fprintf(stderr, "nusd_renderer: extracted %d curve segments from %d BasisCurves\n",
                        written, scene->ncurves);
            }
        }
    }

    if (no_cull_all_geometry_requested() ||
        env_flag_requested("NUSD_MEMORY_LEDGER")) {
        uint32_t stride = r->vertex_stride_floats ? r->vertex_stride_floats : 12u;
        uint64_t mesh_vertex_bytes =
            reserve_vertices * (uint64_t)stride * sizeof(float);
        uint64_t mesh_index_bytes = reserve_indices * sizeof(uint32_t);
        uint64_t curve_cpu_bytes =
            (uint64_t)r->curves.nseg *
            (sizeof(SceneCurveSegment) + sizeof(SceneCurveAabb) +
             3u * sizeof(float));
        uint64_t curve_gpu_bytes =
            (uint64_t)r->curves.nseg *
            (sizeof(SceneCurveSegment) + sizeof(SceneCurveAabb) +
             4u * sizeof(float));
        uint64_t scene_mesh_bytes =
            (uint64_t)scene->nmeshes * sizeof(SceneMesh);
        uint64_t renderer_mesh_bytes =
            (uint64_t)r->nmeshes * sizeof(RendererMesh);
        fprintf(stderr,
                "nusd_renderer: all-geometry memory ledger\n"
                "  scene meshes=%d renderer meshes=%d loaded=%d\n"
                "  unique upload vertices=%llu indices=%llu stride=%u\n"
                "  mesh buffers: vertex=%llu index=%llu cpu+gpu=%llu bytes\n"
                "  curve segments=%d cpu=%llu gpu=%llu bytes\n"
                "  records: SceneMesh=%llu RendererMesh=%llu bytes\n"
                "  mode: no_view_cull=%d geom_caps_default=%s curve_cap_default=%s\n",
                scene->nmeshes, r->nmeshes, loaded,
                (unsigned long long)reserve_vertices,
                (unsigned long long)reserve_indices,
                stride,
                (unsigned long long)mesh_vertex_bytes,
                (unsigned long long)mesh_index_bytes,
                (unsigned long long)(mesh_vertex_bytes + mesh_index_bytes) * 2ull,
                r->curves.nseg,
                (unsigned long long)curve_cpu_bytes,
                (unsigned long long)curve_gpu_bytes,
                (unsigned long long)scene_mesh_bytes,
                (unsigned long long)renderer_mesh_bytes,
                no_cull_all_geometry_requested(),
                no_cull_all_geometry_requested() ? "off" : "on",
                no_cull_all_geometry_requested() ? "off" : "on");
    }

    /* Auto-frame camera on scene bounds. The helper keeps geometry-cache
     * loads framed exactly like USD loads. */
    if (loaded > 0)
        renderer_auto_frame_bounds(r, scene->bounds_min, scene->bounds_max);

    /* Auto-load DomeLight HDR if the USD authored one. The path is
     * already-resolved by scene.c against the USD's parent dir.
     * gpu_load_environment is idempotent — silently no-ops on bad
     * paths. Also stamps env_intensity so high-intensity domes
     * (warehouse / sphere rigs at 1000) compress correctly in the
     * shader. Port from Vulkan ade51f3 + 43bbb19. */
    int force_flat_dome = env_flag_requested("NUSD_DISABLE_SCENE_DOME_HDR") ||
                          env_flag_requested("NUSD_FLAT_DOME_ONLY");
    if (force_flat_dome && scene->dome_hdr_path[0]) {
        gpu_set_dome_color(r->gpu,
                           scene->dome_color[0],
                           scene->dome_color[1],
                           scene->dome_color[2],
                           scene->dome_intensity);
        gpu_set_env_intensity(r->gpu, scene->dome_intensity);
        fprintf(stderr,
                "nusd_renderer: scene DomeLight HDR skipped; flat dome-light color=(%.3f %.3f %.3f) intensity=%.3f\n",
                scene->dome_color[0],
                scene->dome_color[1],
                scene->dome_color[2],
                scene->dome_intensity);
    } else if (scene->has_dome_light && !scene->dome_hdr_path[0]) {
        gpu_set_dome_color(r->gpu,
                           scene->dome_color[0],
                           scene->dome_color[1],
                           scene->dome_color[2],
                           scene->dome_intensity);
        gpu_set_env_intensity(r->gpu, scene->dome_intensity);
        fprintf(stderr,
                "nusd_renderer: flat dome-light color=(%.3f %.3f %.3f) intensity=%.3f\n",
                scene->dome_color[0],
                scene->dome_color[1],
                scene->dome_color[2],
                scene->dome_intensity);
    } else if (scene->dome_hdr_path[0]) {
        char loaded_env_path[PATH_MAX];
        loaded_env_path[0] = '\0';
        if (gpu_load_environment_with_fallbacks(r->gpu,
                                                scene->dome_hdr_path,
                                                loaded_env_path,
                                                sizeof(loaded_env_path))) {
            gpu_set_env_intensity(r->gpu, scene->dome_intensity);
            if (strcmp(loaded_env_path, scene->dome_hdr_path) == 0) {
                fprintf(stderr,
                        "nusd_renderer: dome-light IBL -> %s (intensity=%.3f)\n",
                        scene->dome_hdr_path, scene->dome_intensity);
            } else {
                fprintf(stderr,
                        "nusd_renderer: dome-light IBL -> %s "
                        "(fallback for %s, intensity=%.3f)\n",
                        loaded_env_path, scene->dome_hdr_path,
                        scene->dome_intensity);
            }
        } else {
            fprintf(stderr,
                    "nusd_renderer: dome-light load failed for %s - "
                    "falling back to procedural sky\n",
                    scene->dome_hdr_path);
        }
    }

    fprintf(stderr, "nusd_renderer: loaded %d meshes from %s\n",
            loaded, label ? label : "<stage>");
    geo_census_dump_and_reset(r);
    if (!r->enable_materials) {
        scene_free(scene);
        r->usd_scene = NULL;
    }
    return loaded;
}

int nu_load_usd(NuRenderer* r, const char* path)
{
    if (!r || !path) return NU_ERROR;

    nu_clear_scene(r);
    scene_set_load_time(r->current_time);
    scene_set_load_materials(r->enable_materials);

    EnvOverride prefetch_env = {0};
    EnvOverride reserve_env = {0};
    int lazy_open_policy = lazy_mesh_requested();
    if (lazy_open_policy) {
        /* DSX-scale composed opens are dominated by PathPool contention and
         * reserve rehashing during nanousd prefetch. These defaults keep the
         * lazy load-to-return path fast while preserving explicit user envs. */
        env_set_default(&prefetch_env, "NANOUSD_PREFETCH_THREADS", "1");
        env_set_default(&reserve_env, "NANOUSD_PATHPOOL_RESERVE", "0");
    }

    Scene* scene = scene_load(path);
    if (lazy_open_policy) {
        env_restore(&reserve_env);
        env_restore(&prefetch_env);
    }
    if (!scene) {
        const char* serr = scene_last_error();
        set_error(r, serr ? serr : "scene_load failed");
        return NU_ERROR;
    }
    return nu_attach_scene(r, scene, path);
}

int nu_load_usd_from_handle(NuRenderer* r, void* stage_handle,
                            const char* stage_label)
{
    return nu_load_usd_from_handle_with_dir(r, stage_handle, stage_label, NULL);
}

int nu_load_usd_from_handle_with_dir(NuRenderer* r, void* stage_handle,
                                     const char* stage_label,
                                     const char* scene_dir)
{
    if (!r || !stage_handle) return NU_ERROR;

    nu_clear_scene(r);
    scene_set_load_time(r->current_time);
    scene_set_load_materials(r->enable_materials);

    /* `stage_label` is what scene.c's filepath-based scene_dir derivation
     * uses (via dirname). When the caller hints scene_dir explicitly, we
     * synthesise a path that dirname() will resolve correctly: a synthetic
     * file under scene_dir, e.g. "/path/to/scene/<borrowed-stage>". */
    char synthetic_path[1024];
    const char* label = (stage_label && *stage_label) ? stage_label : "<borrowed-stage>";
    if (scene_dir && *scene_dir) {
        size_t dlen = strlen(scene_dir);
        size_t llen = strlen(label);
        if (dlen + 1 + llen + 1 < sizeof(synthetic_path)) {
            memcpy(synthetic_path, scene_dir, dlen);
            synthetic_path[dlen] = '/';
            memcpy(synthetic_path + dlen + 1, label, llen);
            synthetic_path[dlen + 1 + llen] = '\0';
            label = synthetic_path;
        }
    }

    Scene* scene = scene_load_from_stage(stage_handle, label);
    if (!scene) {
        const char* serr = scene_last_error();
        set_error(r, serr ? serr : "scene_load_from_stage failed");
        return NU_ERROR;
    }
    return nu_attach_scene(r, scene, label);
}

NuResult nu_extract_deferred(NuRenderer* r)
{
    if (!r) return NU_ERROR;
    if (!r->lazy_pending) return NU_OK;
    Scene* lazy_scene = r->usd_scene;
    if (!lazy_scene || !lazy_scene->_stage) {
        set_error(r, "nu_extract_deferred: no lazy stage pinned");
        r->lazy_pending = 0;
        return NU_ERROR;
    }

    char prev_env[16] = {0};
    const char* prev = getenv("NUSD_LAZY_MESH");
    int had_prev = prev != NULL;
    if (had_prev) {
        size_t n = strlen(prev);
        if (n >= sizeof(prev_env)) n = sizeof(prev_env) - 1;
        memcpy(prev_env, prev, n);
        prev_env[n] = '\0';
    }
    setenv("NUSD_LAZY_MESH", "0", 1);

    void* stage = lazy_scene->_stage;
    int owns_stage = lazy_scene->_owns_stage;
    fprintf(stderr,
            "nusd_renderer: nu_extract_deferred — running eager scene_load_from_stage "
            "on pinned lazy stage (owns_stage=%d).\n",
            owns_stage);

    scene_set_load_time(r->current_time);
    scene_set_load_materials(r->enable_materials);
    Scene* eager = scene_load_from_stage(stage, "<lazy-extract>");
    if (!eager) {
        const char* serr = scene_last_error();
        set_error(r, serr ? serr
                          : "nu_extract_deferred: eager scene_load_from_stage failed");
        if (had_prev) setenv("NUSD_LAZY_MESH", prev_env, 1);
        else          unsetenv("NUSD_LAZY_MESH");
        return NU_ERROR;
    }

    eager->_owns_stage = owns_stage;
    lazy_scene->_owns_stage = 0;
    scene_free(lazy_scene);
    r->usd_scene = NULL;
    r->lazy_pending = 0;

    int rc = nu_attach_scene(r, eager, "<lazy-extract>");

    if (had_prev) setenv("NUSD_LAZY_MESH", prev_env, 1);
    else          unsetenv("NUSD_LAZY_MESH");

    return rc >= 0 ? NU_OK : NU_ERROR;
}

static void extract_frustum_planes(const float vp[16], float planes[6][4])
{
    #define ROW(r,c) vp[(r) * 4 + (c)]
    float p[6][4];
    for (int c = 0; c < 4; c++) p[0][c] = ROW(3,c) + ROW(0,c);
    for (int c = 0; c < 4; c++) p[1][c] = ROW(3,c) - ROW(0,c);
    for (int c = 0; c < 4; c++) p[2][c] = ROW(3,c) + ROW(1,c);
    for (int c = 0; c < 4; c++) p[3][c] = ROW(3,c) - ROW(1,c);
    for (int c = 0; c < 4; c++) p[4][c] = ROW(3,c) + ROW(2,c);
    for (int c = 0; c < 4; c++) p[5][c] = ROW(3,c) - ROW(2,c);
    #undef ROW

    for (int i = 0; i < 6; i++) {
        float nx = p[i][0], ny = p[i][1], nz = p[i][2], d = p[i][3];
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        float inv_len = len > 0.0f ? 1.0f / len : 1.0f;
        planes[i][0] = nx * inv_len;
        planes[i][1] = ny * inv_len;
        planes[i][2] = nz * inv_len;
        planes[i][3] = d  * inv_len;
    }
}

static int aabb_in_frustum(const float planes[6][4],
                           const float mn[3], const float mx[3])
{
    if (mn[0] == -FLT_MAX || mx[0] == FLT_MAX)
        return 1;
    for (int i = 0; i < 6; i++) {
        const float* pl = planes[i];
        float px = (pl[0] >= 0.0f) ? mx[0] : mn[0];
        float py = (pl[1] >= 0.0f) ? mx[1] : mn[1];
        float pz = (pl[2] >= 0.0f) ? mx[2] : mn[2];
        if (pl[0] * px + pl[1] * py + pl[2] * pz + pl[3] < 0.0f)
            return 0;
    }
    return 1;
}

NuResult nu_extract_deferred_visible(NuRenderer* r,
                                     const float* vp_matrices,
                                     int num_cameras)
{
    if (!r) return NU_ERROR;
    if (!r->lazy_pending) return NU_OK;
    Scene* lazy_scene = r->usd_scene;
    if (!lazy_scene || !lazy_scene->_stage) {
        set_error(r, "nu_extract_deferred_visible: no lazy stage pinned");
        r->lazy_pending = 0;
        return NU_ERROR;
    }

    if (no_cull_all_geometry_requested()) {
        fprintf(stderr,
                "nusd_renderer: nu_extract_deferred_visible — "
                "NUSD_NO_CULL_ALL_GEOMETRY=1, extracting all geometry.\n");
        return nu_extract_deferred(r);
    }

    if (!vp_matrices || num_cameras <= 0 || lazy_scene->nmeshes <= 0) {
        fprintf(stderr,
                "nusd_renderer: nu_extract_deferred_visible — no cull "
                "(cameras=%d, lazy_records=%d); falling back to extract-all.\n",
                num_cameras, lazy_scene->nmeshes);
        return nu_extract_deferred(r);
    }

    int nprims = nanousd_nprims((NanousdStage)lazy_scene->_stage);
    if (nprims <= 0) {
        return nu_extract_deferred(r);
    }

    float (*planes)[6][4] = (float (*)[6][4])
        malloc((size_t)num_cameras * 6u * 4u * sizeof(float));
    unsigned char* wanted = (unsigned char*)calloc((size_t)nprims, 1);
    if (!planes || !wanted) {
        free(planes);
        free(wanted);
        set_error(r, "nu_extract_deferred_visible: allocation failed");
        return NU_ERROR;
    }
    for (int c = 0; c < num_cameras; c++)
        extract_frustum_planes(&vp_matrices[c * 16], planes[c]);

    const char** visible_instance_child_paths = NULL;
    int n_visible_instance_child_paths = 0;
    if (lazy_scene->nmeshes > 0) {
        visible_instance_child_paths =
            (const char**)calloc((size_t)lazy_scene->nmeshes,
                                 sizeof(const char*));
    }

    int n_visible = 0;
    for (int i = 0; i < lazy_scene->nmeshes; i++) {
        SceneMesh* sm = &lazy_scene->meshes[i];
        int pidx = sm->lazy_prim_idx;
        if (pidx < 0 || pidx >= nprims) continue;
        for (int c = 0; c < num_cameras; c++) {
            if (aabb_in_frustum(planes[c], sm->bounds_min, sm->bounds_max)) {
                if (!wanted[pidx]) {
                    wanted[pidx] = 1;
                    n_visible++;
                }
                if (visible_instance_child_paths && sm->path[0])
                    visible_instance_child_paths[n_visible_instance_child_paths++] = sm->path;
                break;
            }
        }
    }

    fprintf(stderr,
            "nusd_renderer: nu_extract_deferred_visible — %d/%d lazy records "
            "visible across %d cameras (%d instance children); running "
            "filtered eager extract.\n",
            n_visible, lazy_scene->nmeshes, num_cameras,
            n_visible_instance_child_paths);

    scene_set_point_instance_frusta((const float*)planes, num_cameras);
    scene_set_visible_instance_child_paths(
        visible_instance_child_paths, n_visible_instance_child_paths);
    free(visible_instance_child_paths);
    visible_instance_child_paths = NULL;
    free(planes);
    planes = NULL;

    char prev_env[16] = {0};
    const char* prev = getenv("NUSD_LAZY_MESH");
    int had_prev = prev != NULL;
    if (had_prev) {
        size_t n = strlen(prev);
        if (n >= sizeof(prev_env)) n = sizeof(prev_env) - 1;
        memcpy(prev_env, prev, n);
        prev_env[n] = '\0';
    }
    setenv("NUSD_LAZY_MESH", "0", 1);

    void* stage = lazy_scene->_stage;
    int owns_stage = lazy_scene->_owns_stage;
    scene_set_load_time(r->current_time);
    scene_set_load_materials(r->enable_materials);
    Scene* eager = scene_load_from_stage_filtered(
        stage, "<lazy-extract-visible>", wanted, nprims);
    scene_set_point_instance_frusta(NULL, 0);
    scene_set_visible_instance_child_paths(NULL, 0);
    free(wanted);
    if (!eager) {
        const char* serr = scene_last_error();
        set_error(r, serr ? serr
                          : "nu_extract_deferred_visible: filtered eager load failed");
        if (had_prev) setenv("NUSD_LAZY_MESH", prev_env, 1);
        else          unsetenv("NUSD_LAZY_MESH");
        return NU_ERROR;
    }

    eager->_owns_stage = owns_stage;
    lazy_scene->_owns_stage = 0;
    scene_free(lazy_scene);
    r->usd_scene = NULL;
    r->lazy_pending = 0;

    int rc = nu_attach_scene(r, eager, "<lazy-extract-visible>");

    if (had_prev) setenv("NUSD_LAZY_MESH", prev_env, 1);
    else          unsetenv("NUSD_LAZY_MESH");

    return rc >= 0 ? NU_OK : NU_ERROR;
}

NuResult nu_extract_deferred_batched(NuRenderer* r, int num_batches)
{
    (void)num_batches;
    return nu_extract_deferred(r);
}

NuResult nu_set_current_time(NuRenderer* r, double time)
{
    if (!r) return NU_ERROR;
    r->current_time = time;
    return NU_OK;
}

NuResult nu_build_accel(NuRenderer* r)
{
    if (!r) return NU_ERROR;

    if (r->scene_dirty) {
        if (!rebuild_gpu_buffers(r))
            return NU_ERROR;
    }

    /* Phase 11.A.2: upload curve data + build the AABB BLAS before the mesh
     * accel rebuild, so rebuild_accel can wire the curve BLAS into the TLAS.
     * Skipped under rt_cull: a single BLAS over all curves doesn't traverse
     * (Moana = 25.5M segs) — rebuild_accel rebuilds a culled curve BLAS per
     * camera instead, from the retained CPU arrays. */
    if (r->curves_dirty && r->curves.nseg > 0 && !rt_cull_active()) {
        double t_upload = wall_seconds();
        if (!gpu_upload_curve_data(r->gpu,
                                   r->curves.segments,
                                   r->curves.aabbs,
                                   r->curves.colors,
                                   r->curves.nseg)) {
            set_error(r, "gpu_upload_curve_data failed");
            return NU_ERROR;
        }
        float upload_ms = elapsed_ms(t_upload);
        r->timings.staging_upload_segs_ms = upload_ms / 3.0f;
        r->timings.staging_upload_aabbs_ms = upload_ms / 3.0f;
        r->timings.staging_upload_colors_ms = upload_ms / 3.0f;
        double t_curve = wall_seconds();
        if (!gpu_build_curve_blas(r->gpu)) {
            set_error(r, "gpu_build_curve_blas failed");
            return NU_ERROR;
        }
        r->timings.curve_blas_build_ms = elapsed_ms(t_curve);
        r->curves_dirty = 0;
        r->accel_dirty = 1;
        r->rt_blas_valid = 0;  /* curve BLAS must be (re)wired into blas_list */
        if (can_release_curve_cpu_staging_after_upload(r))
            release_curve_cpu_staging(r);
    }

    if (r->accel_dirty) {
        if (!rebuild_accel(r))
            return NU_ERROR;
    }

    return NU_OK;
}

NuResult nu_finalize_scene(NuRenderer* r)
{
    if (!r) return NU_ERROR;
    if (r->cpu_staging_released) return NU_OK;

    if (nu_build_accel(r) != NU_OK) {
        set_error(r, "nu_finalize_scene: nu_build_accel failed");
        return NU_ERROR;
    }

    release_cpu_staging(r);
    return NU_OK;
}

/* ---- Gaussian splat scene ---- */

NuResult nu_gs_set_particles(NuRenderer* r, const NuGsDesc* desc)
{
    if (!r || !desc || !r->gs) return NU_ERROR;
    NuResult res = gs_scene_set_particles(r->gs, desc);
    if (res == NU_OK) {
        r->gs_dirty = 1;
        r->has_frame = 0;
    }
    return res;
}

NuResult nu_gs_clear_particles(NuRenderer* r)
{
    if (!r || !r->gs) return NU_ERROR;
    NuResult res = gs_scene_clear_particles(r->gs);
    gpu_gs_destroy(r->gpu);
    r->gs_dirty = 0;
    r->has_frame = 0;
    return res;
}

NuResult nu_gs_set_proxy(NuRenderer* r, NuGsProxyKind kind)
{
    if (!r || !r->gs) return NU_ERROR;
    return gs_scene_set_proxy(r->gs, kind);
}

NuResult nu_gs_set_color_space(NuRenderer* r, NuGsColorSpace cs)
{
    if (!r || !r->gs) return NU_ERROR;
    return gs_scene_set_color_space(r->gs, cs);
}

NuResult nu_gs_set_camera_model(NuRenderer* r, NuGsCameraModel cm)
{
    if (!r || !r->gs) return NU_ERROR;
    return gs_scene_set_camera_model(r->gs, cm);
}

NuResult nu_gs_set_k(NuRenderer* r, int k)
{
    if (!r || !r->gs) return NU_ERROR;
    return gs_scene_set_k(r->gs, k);
}

NuResult nu_gs_set_max_passes(NuRenderer* r, int max_passes)
{
    if (!r || !r->gs) return NU_ERROR;
    return gs_scene_set_max_passes(r->gs, max_passes);
}

NuResult nu_gs_set_min_transmittance(NuRenderer* r, float eps)
{
    if (!r || !r->gs) return NU_ERROR;
    return gs_scene_set_min_transmittance(r->gs, eps);
}

NuResult nu_gs_set_iso_opacity_threshold(NuRenderer* r, float iso)
{
    if (!r || !r->gs) return NU_ERROR;
    return gs_scene_set_iso_opacity_threshold(r->gs, iso);
}

int nu_gs_available(NuRenderer* r)
{
    if (!r || !r->gs) return 0;
    return (gs_scene_particle_count(r->gs) > 0 && gpu_gs_available(r->gpu)) ? 1 : 0;
}

int nu_gs_particle_count(NuRenderer* r)
{
    return (r && r->gs) ? gs_scene_particle_count(r->gs) : 0;
}

NuResult nu_gs_render(NuRenderer* r, int cam_id)
{
    if (!r || !r->gs) return NU_ERROR;
    if (cam_id != 0) return NU_ERROR_BAD_ID;
    if (!r->enable_rt || !gpu_rt_available(r->gpu)) {
        set_error(r, "Gaussian RT not available");
        return NU_ERROR_NO_RT;
    }
    if (gs_scene_particle_count(r->gs) <= 0) {
        set_error(r, "No Gaussian particles to render");
        return NU_ERROR_NO_SCENE;
    }

    if (r->gs_dirty) {
        if (!gpu_gs_upload_particles(r->gpu,
                                     gs_scene_positions(r->gs),
                                     gs_scene_scales(r->gs),
                                     gs_scene_orientations(r->gs),
                                     gs_scene_opacities(r->gs),
                                     gs_scene_kernel_scales(r->gs),
                                     gs_scene_sh_coefficients(r->gs),
                                     (uint32_t)gs_scene_particle_count(r->gs),
                                     (uint32_t)gs_scene_sh_degree(r->gs),
                                     gs_scene_prim_xform(r->gs))) {
            set_error(r, "gpu_gs_upload_particles failed");
            return NU_ERROR;
        }
        if (!gpu_gs_build_accel(r->gpu)) {
            set_error(r, "gpu_gs_build_accel failed");
            return NU_ERROR_NO_RT;
        }
        r->gs_dirty = 0;
    }

    float vp_inv[32];
    compute_camera_inverses(&r->camera, vp_inv);

    if (!gpu_gs_render(r->gpu, vp_inv,
                       (uint32_t)gs_scene_sh_degree(r->gs),
                       (uint32_t)gs_scene_k(r->gs),
                       (uint32_t)gs_scene_max_passes(r->gs),
                       gs_scene_min_transmittance(r->gs),
                       gs_scene_iso_opacity_threshold(r->gs),
                       (uint32_t)gs_scene_color_space(r->gs))) {
        set_error(r, "gpu_gs_render failed");
        return NU_ERROR;
    }

    r->has_frame = 1;
    return NU_OK;
}

NuResult nu_gs_fetch_depth(NuRenderer* r, float* out_depth)
{
    if (!r || !out_depth) return NU_ERROR;
    if (!gpu_gs_fetch_depth(r->gpu, out_depth, (uint32_t)r->width, (uint32_t)r->height)) {
        set_error(r, "gpu_gs_fetch_depth failed");
        return NU_ERROR;
    }
    return NU_OK;
}

NuResult nu_gs_fetch_normal(NuRenderer* r, float* out_normal)
{
    if (!r || !out_normal) return NU_ERROR;
    if (!gpu_gs_fetch_normal(r->gpu, out_normal, (uint32_t)r->width, (uint32_t)r->height)) {
        set_error(r, "gpu_gs_fetch_normal failed");
        return NU_ERROR;
    }
    return NU_OK;
}

/* ---- Attribute updates ---- */

NuResult nu_set_transforms(NuRenderer* r,
                           const int* mesh_ids, const float* mat4x4s, int count)
{
    if (!r || !mesh_ids || !mat4x4s || count <= 0) return NU_ERROR;

    for (int i = 0; i < count; i++) {
        int id = mesh_ids[i];
        if (id < 0 || id >= r->nmeshes || !r->meshes[id].active)
            return NU_ERROR_BAD_ID;
    }

    for (int i = 0; i < count; i++) {
        int id = mesh_ids[i];
        transpose4x4(&mat4x4s[i * 16], r->meshes[id].transform);
    }

    /* Mark TLAS dirty so the next RT render rebuilds instance transforms.
     * BLASes are unchanged — only the TLAS needs updating. */
    r->tlas_dirty = 1;
    invalidate_raster_instance_cache(r);
    return NU_OK;
}

NuResult nu_set_colors(NuRenderer* r,
                       const int* mesh_ids, const float* rgb, int count)
{
    if (!r || !mesh_ids || !rgb || count <= 0) return NU_ERROR;

    for (int i = 0; i < count; i++) {
        int id = mesh_ids[i];
        if (id < 0 || id >= r->nmeshes || !r->meshes[id].active)
            return NU_ERROR_BAD_ID;
    }

    for (int i = 0; i < count; i++) {
        int id = mesh_ids[i];
        r->meshes[id].color[0] = rgb[i * 3 + 0];
        r->meshes[id].color[1] = rgb[i * 3 + 1];
        r->meshes[id].color[2] = rgb[i * 3 + 2];
    }

    r->colors_dirty = 1;
    invalidate_raster_instance_cache(r);
    return NU_OK;
}

NuResult nu_set_mesh_color(NuRenderer* r, int mesh_id, const float color[3])
{
    if (!r || !color) return NU_ERROR;
    return nu_set_colors(r, &mesh_id, color, 1);
}

NuResult nu_set_visibility(NuRenderer* r,
                           const int* mesh_ids, const int* visible, int count)
{
    if (!r || !mesh_ids || !visible || count <= 0) return NU_ERROR;

    for (int i = 0; i < count; i++) {
        int id = mesh_ids[i];
        if (id < 0 || id >= r->nmeshes || !r->meshes[id].active)
            return NU_ERROR_BAD_ID;
    }

    for (int i = 0; i < count; i++) {
        int id = mesh_ids[i];
        r->meshes[id].visible = visible[i] ? 1 : 0;
    }

    /* Visibility changes require TLAS rebuild to exclude/include instances */
    r->tlas_dirty = 1;
    invalidate_raster_instance_cache(r);
    return NU_OK;
}

NuResult nu_set_instance_masks(NuRenderer* r, const unsigned char* masks, int count)
{
    if (!r) return NU_ERROR;
    if (count < 0) return NU_ERROR;

    if (!masks || count == 0) {
        for (int i = 0; i < r->nmeshes; i++)
            r->meshes[i].env_mask = 0xFFu;
        r->tlas_dirty = 1;
        return NU_OK;
    }

    int n = count < r->nmeshes ? count : r->nmeshes;
    for (int i = 0; i < n; i++)
        r->meshes[i].env_mask = masks[i];
    r->tlas_dirty = 1;
    return NU_OK;
}

NuResult nu_set_env_partition(NuRenderer* r,
                              const int* mesh_to_env,
                              int count,
                              int num_envs)
{
    (void)mesh_to_env;
    (void)count;
    (void)num_envs;
    if (!r) return NU_ERROR;
    set_error(r, "nu_set_env_partition is not supported by the Metal backend yet");
    return NU_ERROR_UNSUPPORTED;
}

/* ---- Camera ---- */

NuResult nu_set_camera(NuRenderer* r, int cam_id, const NuCameraDesc* desc)
{
    if (!r || !desc || cam_id != 0) return NU_ERROR_BAD_ID;

    camera_set_eye_target(&r->camera, desc->eye, desc->target);

    if (desc->fov_degrees > 0)
        r->camera.fov_y = desc->fov_degrees * 3.14159265f / 180.0f;
    if (desc->near_clip > 0)
        r->camera.near_z = desc->near_clip;
    if (desc->far_clip > 0)
        r->camera.far_z = desc->far_clip;

    r->camera_valid = 1;
    return NU_OK;
}

/* Set a fully explicit camera with arbitrary up vector. Bypasses the
 * spherical (Y-up yaw/pitch) reconstruction so callers with Z-up scenes
 * (e.g. Newton/IsaacLab) can render with the correct vertical axis. */
NuResult nu_set_camera_explicit(NuRenderer* r,
                                const float eye[3],
                                const float target[3],
                                const float up[3],
                                float fov_degrees,
                                float near_clip,
                                float far_clip)
{
    if (!r || !eye || !target || !up) return NU_ERROR;
    float fov_rad = (fov_degrees > 0 ? fov_degrees : 45.0f) * 3.14159265f / 180.0f;
    float nz = near_clip > 0 ? near_clip : 0.01f;
    float fz = far_clip > 0 ? far_clip : 10000.0f;
    camera_set_explicit_view(&r->camera, eye, target, up, fov_rad, nz, fz);
    r->camera_valid = 1;
    return NU_OK;
}

NuResult nu_set_camera_explicit_window(NuRenderer* r,
                                       const float eye[3],
                                       const float target[3],
                                       const float up[3],
                                       float fov_degrees,
                                       float near_clip,
                                       float far_clip,
                                       float projection_shift_x,
                                       float projection_shift_y)
{
    NuResult rc = nu_set_camera_explicit(r, eye, target, up,
                                         fov_degrees, near_clip, far_clip);
    if (rc != NU_OK) return rc;
    camera_set_projection_shift(&r->camera, projection_shift_x, projection_shift_y);
    return NU_OK;
}

NuResult nu_set_exposure(NuRenderer* r, const NuExposureDesc* desc)
{
    if (!r) return NU_ERROR;
    if (desc) {
        r->exposure_desc = *desc;
    } else {
        memset(&r->exposure_desc, 0, sizeof(r->exposure_desc));
    }
    return NU_OK;
}

NuResult nu_get_camera(NuRenderer* r, int cam_id, NuCameraDesc* out_desc)
{
    if (!r || !out_desc || cam_id != 0) return NU_ERROR_BAD_ID;

    float eye[3];
    camera_get_eye(&r->camera, eye);
    out_desc->eye[0]    = eye[0];
    out_desc->eye[1]    = eye[1];
    out_desc->eye[2]    = eye[2];
    out_desc->target[0] = r->camera.target[0];
    out_desc->target[1] = r->camera.target[1];
    out_desc->target[2] = r->camera.target[2];
    out_desc->fov_degrees = r->camera.fov_y * 180.0f / 3.14159265f;
    out_desc->near_clip   = r->camera.near_z;
    out_desc->far_clip    = r->camera.far_z;
    return NU_OK;
}

/* ---- Rendering ---- */

NuResult nu_render(NuRenderer* r, int cam_id, NuRenderMode mode)
{
    if (!r) return NU_ERROR;
    if (cam_id != 0) return NU_ERROR_BAD_ID;

    /* Rebuild GPU buffers if scene changed */
    if (r->scene_dirty) {
        if (!rebuild_gpu_buffers(r)) return NU_ERROR;
    }

    /* "Geometry" means meshes OR curves — Phase 11.A added curves as a
     * standalone primitive class; raster now draws curves as camera-facing
     * ribbons while RT keeps using the procedural tube hit path. */
    int has_meshes = (r->total_indices > 0);
    int has_curves = (r->curves.nseg > 0);
    if (!has_meshes && !has_curves) {
        set_error(r, "No geometry to render");
        return NU_ERROR_NO_SCENE;
    }

    /* Phase 11.A: upload curves before either raster curve draw or RT accel
     * rebuild. Only RT/shadow need the curve BLAS; raster consumes the same
     * segment/color buffers directly. */
    int need_curve_blas =
        has_curves && (mode == NU_RENDER_RT || mode == NU_RENDER_SHADOW) &&
        !gpu_curve_blas_built(r->gpu);
    if (has_curves && (r->curves_dirty || need_curve_blas) && !rt_cull_active()) {
        if (r->curves_dirty) {
            if (!gpu_upload_curve_data(r->gpu, r->curves.segments, r->curves.aabbs,
                                       r->curves.colors, r->curves.nseg)) {
                set_error(r, "gpu_upload_curve_data failed");
                return NU_ERROR;
            }
        }
        if (mode == NU_RENDER_RT || mode == NU_RENDER_SHADOW) {
            if (!gpu_build_curve_blas(r->gpu)) {
                set_error(r, "gpu_build_curve_blas failed");
                return NU_ERROR;
            }
            /* Curve presence changed the TLAS instance set — force a rebuild. */
            r->accel_dirty = 1;
            r->rt_blas_valid = 0;  /* curve BLAS must be (re)wired into blas_list */
        }
        r->curves_dirty = 0;
        if (can_release_curve_cpu_staging_after_upload(r))
            release_curve_cpu_staging(r);
    }

    /* Culled-proxy TLAS: when the no-cull-all-geometry path is active (Moana
     * scale — the full 202M-instance TLAS doesn't fit, 45.85 GiB), the RT
     * accel is built from only the camera-visible set, so it must rebuild
     * every frame as the camera moves. */
    if ((mode == NU_RENDER_RT || mode == NU_RENDER_SHADOW) && rt_cull_active())
        r->accel_dirty = 1;

    /* Rebuild accel structures if needed */
    if (r->accel_dirty && (mode == NU_RENDER_RT || mode == NU_RENDER_SHADOW)) {
        if (!rebuild_accel(r)) return NU_ERROR;
    }

    /* Get camera matrices */
    float view[16], proj[16], vp[16];
    camera_get_view(&r->camera, view);
    camera_get_proj(&r->camera, proj);
    camera_get_vp(&r->camera, vp);

    float eye[3];
    camera_get_eye(&r->camera, eye);

    if (mode == NU_RENDER_RT) {
        if (!gpu_rt_available(r->gpu) || !gpu_rt_built(r->gpu)) {
            set_error(r, "RT not available or not built");
            return NU_ERROR_NO_RT;
        }
        double t_rt = wall_seconds();

        /* Update scene SSBO colors if they changed */
        if (r->colors_dirty) {
            float* cols = (float*)malloc(r->nmeshes * 3 * sizeof(float));
            if (cols) {
                for (int m = 0; m < r->nmeshes; m++) {
                    cols[m * 3 + 0] = r->meshes[m].color[0];
                    cols[m * 3 + 1] = r->meshes[m].color[1];
                    cols[m * 3 + 2] = r->meshes[m].color[2];
                }
                gpu_update_scene_colors(r->gpu, cols, (uint32_t)r->nmeshes);
                free(cols);
            }
            r->colors_dirty = 0;
        }

        /* Rebuild TLAS if transforms changed */
        if (r->tlas_dirty) {
            /* USD row-vector 4x4 → VkTransformMatrixKHR 3x4
             * (column-vector row-major; translation in last column). */
            float* xforms_3x4 = (float*)malloc(r->nmeshes * 12 * sizeof(float));
            if (xforms_3x4) {
                for (int m = 0; m < r->nmeshes; m++) {
                    usd4x4_to_vk3x4(r->meshes[m].transform, &xforms_3x4[m * 12]);
                }
                uint8_t* masks = (uint8_t*)malloc(r->nmeshes);
                if (masks)
                    fill_rt_instance_masks(r, masks);
                gpu_update_tlas(r->gpu, xforms_3x4, masks, (uint32_t)r->nmeshes);
                free(masks);
                free(xforms_3x4);
            }
            r->tlas_dirty = 0;
        }

        if (!gpu_begin_frame_rt(r->gpu)) return NU_ERROR;

        /* Compute inverse matrices for RT push constants. */
        GpuRtPushConstants pc;
        compute_view_inverse(&r->camera, pc.view_inv);
        compute_proj_inverse(&r->camera, pc.proj_inv);
        pc.ground_y = -1.0f;    /* scene bounds min Y */
        pc.scene_scale = 10.0f; /* scene diagonal */
        pc.fast_mode = (uint32_t)r->fast_mode;
        pc.depth_enabled = 0;   /* no depth output for single-camera */
        pc.segmentation_enabled = 0;   /* no segmentation for single-camera */
        pc.normals_enabled = 0;        /* no normals for single-camera */
        pc.curve_fast = renderer_curve_fast(r);
        /* Runtime exposure knob (matches the Vulkan backend's NUSD_EXPOSURE_SCALE);
         * default 1.0 leaves the tonemap unchanged. */
        {
            const char *es = getenv("NUSD_EXPOSURE_SCALE");
            float s = es ? (float)atof(es) : 1.0f;
            pc.tone_exposure_scale = (s > 0.0f) ? s : 1.0f;
        }
        pc.jitter_x = 0.0f;
        pc.jitter_y = 0.0f;
        {
            const char* jf = getenv("NUSD_RT_JITTER_FRAME");
            if (jf && jf[0]) {
                uint32_t frame = (uint32_t)strtoul(jf, NULL, 10) + 1u;
                pc.jitter_x = halton(frame, 2u) - 0.5f;
                pc.jitter_y = halton(frame, 3u) - 0.5f;
            }
            const char* jx = getenv("NUSD_RT_JITTER_X");
            const char* jy = getenv("NUSD_RT_JITTER_Y");
            if (jx && jx[0]) pc.jitter_x = (float)atof(jx);
            if (jy && jy[0]) pc.jitter_y = (float)atof(jy);
        }

        gpu_cmd_trace_rays(r->gpu, &pc);
        gpu_end_frame_rt(r->gpu);
        r->timings.rt_dispatch_ms = elapsed_ms(t_rt);

    } else {
        /* Raster path */
        if (!gpu_begin_frame(r->gpu)) return NU_ERROR;

        GpuPipeline pipe = (mode == NU_RENDER_SHADOW && r->shadow_pipeline)
                            ? r->shadow_pipeline
                            : r->raster_pipeline;
        if (!pipe) {
            set_error(r, "Pipeline not available");
            return NU_ERROR;
        }

        gpu_cmd_bind_pipeline(r->gpu, pipe);
        if (mode == NU_RENDER_SHADOW)
        gpu_cmd_bind_shadow(r->gpu);
        gpu_cmd_bind_vertex_buffer(r->gpu, r->vertex_buffer);
        if (r->index_buffer)
            gpu_cmd_bind_index_buffer(r->gpu, r->index_buffer);
        if (!ensure_instance_buffer(r, 1u)) {
            set_error(r, "Failed to create raster instance buffer");
            return NU_ERROR;
        }
        gpu_cmd_bind_instance_buffer(r->gpu, r->instance_buffer);
        int bound_index_stream = r->index_buffer ? 0 : -1; /* -1 = none, 0 = mesh, 1 = meshlet */

        if (r->preserve_instance_draws) {
            if (!render_raster_instanced_batches(r, vp, eye, &bound_index_stream))
                return NU_ERROR;
        } else {
        for (int i = 0; i < r->nmeshes; i++) {
            RendererMesh* m = &r->meshes[i];
            if (!m->active || !m->visible || m->nindices == 0) continue;

            /* Build MVP. Convention:
             *   - view & proj from camera_get_*: stored row-major but
             *     mathematically column-vector (translate at indices
             *     3, 7, 11). Confirmed by the standard f/aspect, -f, …
             *     proj layout.
             *   - m->transform from USD: row-major row-vector convention
             *     (translate in BOTTOM ROW — indices 12, 13, 14).
             *
             * RT works around this by feeding m->transform through
             * vk3x4_to_mtl4x3 which transposes into Metal's column-major
             * 4x3 instance-transform layout. Raster historically did
             * NOT transpose, so vp * model mixed conventions and the
             * translate component dropped to (0,0,0,1) — every mesh
             * stacked on the origin. Render looked like one giant blob.
             *
             * Fix: transpose model from row-vector to column-vector
             * convention before the matmul, matching view/proj. */
            float model_rm[16];
            memcpy(model_rm, m->transform, sizeof(model_rm));
            float model[16];
            for (int row = 0; row < 4; row++)
                for (int col = 0; col < 4; col++)
                    model[row * 4 + col] = model_rm[col * 4 + row];


            float mvp[16];
            /* mvp = vp * model (both column-vector convention now) */
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 4; col++) {
                    float sum = 0;
                    for (int k = 0; k < 4; k++)
                        sum += vp[row * 4 + k] * model[k * 4 + col];
                    mvp[row * 4 + col] = sum;
                }
            }

            GpuMeshPushConstants pc;
            memcpy(pc.mvp, mvp, sizeof(pc.mvp));
            /* Shader's per-fragment world-space normal needs the row-vector
             * model (translate in bottom row); mvp uses the transposed
             * column-vector form. Keep both representations on pc. */
            memcpy(pc.model, model_rm, sizeof(pc.model));
            pc.material_index = m->material_index;
            pc.instanced = 0u;
            pc.instance_base = 0u;
            pc._pad[0] = 0u;
            pc.color[0] = m->color[0];
            pc.color[1] = m->color[1];
            pc.color[2] = m->color[2];
            pc.color[3] = 1.0f; /* use this color */
            memcpy(pc.eye_pos, eye, 3 * sizeof(float));
            pc.eye_pos[3] = 0.0f;

            gpu_cmd_push_constants(r->gpu, &pc, sizeof(pc));
            int want_meshlet_stream = (r->meshlet_raster_enabled &&
                                       r->meshlet_index_buffer &&
                                       m->meshlet_index_count > 0);
            GpuBuffer desired_index_buffer = want_meshlet_stream
                ? r->meshlet_index_buffer : r->index_buffer;
            if (!desired_index_buffer) continue;
            if (want_meshlet_stream != bound_index_stream) {
                gpu_cmd_bind_index_buffer(r->gpu, desired_index_buffer);
                bound_index_stream = want_meshlet_stream;
            }
            gpu_cmd_draw_indexed(r->gpu,
                want_meshlet_stream ? m->meshlet_index_count : (uint32_t)m->nindices,
                want_meshlet_stream ? m->meshlet_index_offset : m->index_offset,
                (int32_t)m->vertex_offset);
        }
        }

        if (has_curves) {
            if (!gpu_cmd_draw_curves(r->gpu, vp, eye)) {
                set_error(r, "gpu_cmd_draw_curves failed");
                return NU_ERROR;
            }
        }

        /* Environment-background fullscreen pass — paints the GGX-prefiltered
         * HDR onto pixels not occluded by mesh draws. Same view/proj inverse
         * the RT path computes above; without this, the raster viewport
         * shows gpu_begin_frame's clear color instead of the authored dome.
         * No-op when env isn't loaded (gpu_draw_env_background gates on
         * env_mip_levels > 0 and env_bg_pipeline). */
        {
            float vi[16];
            float pi[16];
            compute_view_inverse(&r->camera, vi);
            compute_proj_inverse(&r->camera, pi);

            gpu_draw_env_background(r->gpu, vi, pi);
        }

        gpu_end_frame(r->gpu);

        /* Optional SSAO post-process (NUSD_SSAO=1; tunable via
         * NUSD_SSAO_RADIUS / _STRENGTH / _POWER). Grounds the geometry by
         * darkening contacts using the stored depth buffer. */
        const char* ssao = getenv("NUSD_SSAO");
        if (ssao && ssao[0] && ssao[0] != '0') {
            float radius   = getenv("NUSD_SSAO_RADIUS")  ? (float)atof(getenv("NUSD_SSAO_RADIUS"))  : 1.5f;
            float strength = getenv("NUSD_SSAO_STRENGTH")? (float)atof(getenv("NUSD_SSAO_STRENGTH")): 1.4f;
            float power    = getenv("NUSD_SSAO_POWER")   ? (float)atof(getenv("NUSD_SSAO_POWER"))   : 1.6f;
            float params[8] = {
                r->camera.near_z, r->camera.far_z, radius, strength,
                0.02f, power,
                1.0f / (float)r->width, 1.0f / (float)r->height,
            };
            gpu_ssao_composite(r->gpu, params);
        }
    }

    r->has_frame = 1;
    return NU_OK;
}

/* Metal-only direct texture accessors — return id<MTLTexture> as void*. Used
 * by nanousd-metal-viewer to skip the CPU round-trip in nu_fetch_pixels and
 * to bind the renderer's depth target into the gizmo overlay's fragment
 * shader for occlusion-aware compositing. */
void* nu_get_metal_color_texture(NuRenderer* r)
{
    if (!r) return NULL;
    return gpu_get_metal_color_texture(r->gpu);
}

void* nu_get_metal_depth_texture(NuRenderer* r)
{
    if (!r) return NULL;
    return gpu_get_metal_depth_texture(r->gpu);
}

NuResult nu_fetch_pixels(NuRenderer* r, void* out_pixels, NuPixelFormat format)
{
    double t0 = wall_seconds();
    if (!r || !out_pixels) return NU_ERROR;
    if (!r->has_frame) {
        set_error(r, "No frame rendered yet");
        return NU_ERROR;
    }

    if (format == NU_PIXEL_RGBA8) {
        /* Direct readback: Metal color target is RGBA8 sRGB, so the staging
         * buffer is already RGBA — pure memcpy out of shared storage. */
        if (!gpu_readback_pixels(r->gpu, (uint8_t*)out_pixels,
                                 (uint32_t)r->width, (uint32_t)r->height)) {
            set_error(r, "gpu_readback_pixels failed");
            return NU_ERROR;
        }
    } else if (format == NU_PIXEL_RGBAF32) {
        if (!gpu_readback_pixels_f32(r->gpu, (float*)out_pixels,
                                     (uint32_t)r->width, (uint32_t)r->height)) {
            set_error(r, "gpu_readback_pixels_f32 failed");
            return NU_ERROR;
        }
    } else if (format == NU_PIXEL_BGRA8) {
        /* API-parity with the Vulkan port. On Metal the color target is
         * native RGBA, so this format pays for an R↔B swizzle (the inverse
         * of Vulkan's case where BGRA8 was the zero-cost path). Provided
         * for callers that hand pixels straight to a BGRA-expecting sink
         * (CoreVideo, VideoToolbox, swapchains, etc.). */
        if (!gpu_readback_pixels(r->gpu, (uint8_t*)out_pixels,
                                 (uint32_t)r->width, (uint32_t)r->height)) {
            set_error(r, "gpu_readback_pixels failed");
            return NU_ERROR;
        }
        uint8_t* p = (uint8_t*)out_pixels;
        size_t npx = (size_t)r->width * (size_t)r->height;
        for (size_t i = 0; i < npx; i++) {
            uint8_t tmp = p[i * 4 + 0];
            p[i * 4 + 0] = p[i * 4 + 2];
            p[i * 4 + 2] = tmp;
        }
    } else {
        set_error(r, "Unsupported pixel format");
        return NU_ERROR_UNSUPPORTED;
    }

    r->timings.pixel_readback_ms = elapsed_ms(t0);
    return NU_OK;
}

NuResult nu_fetch_pixels_cuda(NuRenderer* r,
                              int* out_fd,
                              uint64_t* out_size,
                              int* out_width,
                              int* out_height,
                              int* out_format)
{
    if (out_fd) *out_fd = -1;
    if (out_size) *out_size = 0;
    if (out_width) *out_width = 0;
    if (out_height) *out_height = 0;
    if (out_format) *out_format = NU_PIXEL_RGBA8;
    if (r) set_error(r, "nu_fetch_pixels_cuda is not supported by the Metal backend");
    return NU_ERROR_UNSUPPORTED;
}

/* ---- Tiled multi-camera rendering ---- */

static int ensure_tiled_pipeline(NuRenderer* r)
{
    if (r->tiled_pipeline_built) return 1;
    if (!gpu_rt_available(r->gpu) || !gpu_rt_built(r->gpu)) return 0;

    /* Load tiled ray gen shader + reuse standard miss/chit */
    char rgen_path[1024], rmiss_path[1024], rchit_path[1024];
    snprintf(rgen_path, sizeof(rgen_path), "%s/raytrace_tiled.rgen.spv", SHADER_DIR);
    snprintf(rmiss_path, sizeof(rmiss_path), "%s/raytrace.rmiss.spv", SHADER_DIR);
    snprintf(rchit_path, sizeof(rchit_path), "%s/raytrace.rchit.spv", SHADER_DIR);

    uint32_t rgen_sz = 0, rmiss_sz = 0, rchit_sz = 0;
    uint32_t* rgen_spv  = load_shader_file(rgen_path, &rgen_sz);
    uint32_t* rmiss_spv = load_shader_file(rmiss_path, &rmiss_sz);
    uint32_t* rchit_spv = load_shader_file(rchit_path, &rchit_sz);

    if (!rgen_spv || !rmiss_spv || !rchit_spv) {
        free(rgen_spv); free(rmiss_spv); free(rchit_spv);
        set_error(r, "Failed to load tiled RT shaders");
        return 0;
    }

    int ok = gpu_build_tiled_rt_pipeline(r->gpu,
        rgen_spv, rgen_sz, rmiss_spv, rmiss_sz, rchit_spv, rchit_sz);

    free(rgen_spv);
    free(rmiss_spv);
    free(rchit_spv);

    if (!ok) {
        set_error(r, "Failed to build tiled RT pipeline");
        return 0;
    }

    r->tiled_pipeline_built = 1;
    return 1;
}

NuResult nu_render_tiled(NuRenderer* r,
                         const float* vp_inv_matrices, int num_cameras,
                         int tile_w, int tile_h, NuRenderMode mode)
{
    if (!r || !vp_inv_matrices || num_cameras <= 0) return NU_ERROR;
    if (mode != NU_RENDER_RT) {
        set_error(r, "Tiled rendering only supports NU_RENDER_RT mode");
        return NU_ERROR_UNSUPPORTED;
    }

    /* Rebuild GPU buffers if scene changed */
    if (r->scene_dirty) {
        if (!rebuild_gpu_buffers(r)) return NU_ERROR;
    }

    if (r->total_indices == 0) {
        set_error(r, "No geometry to render");
        return NU_ERROR_NO_SCENE;
    }

    /* Rebuild accel structures if needed */
    if (r->accel_dirty) {
        if (!rebuild_accel(r)) return NU_ERROR;
    }

    /* Update scene SSBO colors if they changed */
    if (r->colors_dirty) {
        float* cols = (float*)malloc(r->nmeshes * 3 * sizeof(float));
        if (cols) {
            for (int m = 0; m < r->nmeshes; m++) {
                cols[m * 3 + 0] = r->meshes[m].color[0];
                cols[m * 3 + 1] = r->meshes[m].color[1];
                cols[m * 3 + 2] = r->meshes[m].color[2];
            }
            gpu_update_scene_colors(r->gpu, cols, (uint32_t)r->nmeshes);
            free(cols);
        }
        r->colors_dirty = 0;
    }

    /* Prepare TLAS update data if transforms changed */
    float* xforms_3x4 = NULL;
    uint8_t* masks = NULL;
    int tlas_needs_update = r->tlas_dirty;
    if (tlas_needs_update) {
        xforms_3x4 = (float*)malloc(r->nmeshes * 12 * sizeof(float));
        masks = (uint8_t*)malloc(r->nmeshes);
        if (xforms_3x4) {
            /* USD row-vector 4x4 → VkTransformMatrixKHR 3x4
             * (translation at dst[3]/[7]/[11]) — same conversion the
             * single-camera RT path does. A flat memcpy here would silently
             * drop the row-3 translation. */
            for (int m = 0; m < r->nmeshes; m++) {
                usd4x4_to_vk3x4(r->meshes[m].transform, &xforms_3x4[m * 12]);
            }
        }
        if (masks)
            fill_rt_instance_masks(r, masks);
        r->tlas_dirty = 0;
    }

    /* Compute tile grid layout */
    int num_cols = (int)ceilf(sqrtf((float)num_cameras));
    int num_rows = (num_cameras + num_cols - 1) / num_cols;
    uint32_t total_w = (uint32_t)(num_cols * tile_w);
    uint32_t total_h = (uint32_t)(num_rows * tile_h);

    /* Initialize tiled resources (storage image + camera SSBO) */
    if (!gpu_tiled_init(r->gpu, total_w, total_h, num_cameras)) {
        set_error(r, "Failed to initialize tiled RT resources");
        free(xforms_3x4); free(masks);
        return NU_ERROR;
    }

    /* Upload camera matrices */
    if (!gpu_tiled_upload_cameras(r->gpu, vp_inv_matrices, num_cameras)) {
        set_error(r, "Failed to upload camera matrices");
        free(xforms_3x4); free(masks);
        return NU_ERROR;
    }

    /* Lazily build tiled RT pipeline */
    if (!ensure_tiled_pipeline(r)) { free(xforms_3x4); free(masks); return NU_ERROR; }

    /* Render — open command buffer */
    double t_tiled = wall_seconds();
    if (!gpu_begin_frame_tiled_rt(r->gpu)) {
        set_error(r, "gpu_begin_frame_tiled_rt failed");
        free(xforms_3x4);
        free(masks);
        return NU_ERROR;
    }

    /* TLAS update: try inline (same command buffer) to avoid a separate
     * vkQueueSubmit + vkQueueWaitIdle stall. Falls back to synchronous. */
    if (xforms_3x4) {
        if (!gpu_update_tlas_inline(r->gpu, xforms_3x4, masks, (uint32_t)r->nmeshes)) {
            /* Inline failed (no persistent staging/scratch) — must do it
             * synchronously. Abort the current frame, do the TLAS update,
             * then reopen a new frame command buffer. */
            gpu_abort_frame_tiled_rt(r->gpu);
            gpu_update_tlas(r->gpu, xforms_3x4, masks, (uint32_t)r->nmeshes);

            if (!gpu_begin_frame_tiled_rt(r->gpu)) {
                set_error(r, "gpu_begin_frame_tiled_rt failed after TLAS update");
                free(xforms_3x4);
                free(masks);
                return NU_ERROR;
            }
        }
        free(xforms_3x4);
        free(masks);
    }

    GpuRtTiledPushConstants pc;
    memset(&pc, 0, sizeof(pc));
    pc.tile_w      = (uint32_t)tile_w;
    pc.tile_h      = (uint32_t)tile_h;
    pc.num_cols    = (uint32_t)num_cols;
    pc.num_cameras = (uint32_t)num_cameras;
    pc.ground_y    = -1.0f;
    pc.scene_scale = 10.0f;
    pc.fast_mode              = (uint32_t)r->fast_mode;
    pc.depth_enabled          = (uint32_t)r->depth_enabled;
    pc.segmentation_enabled   = (uint32_t)r->segmentation_enabled;
    pc.normals_enabled        = (uint32_t)r->normals_enabled;
    pc.curve_fast             = renderer_curve_fast(r);

    /* Signal the raygen shader to write directly to SSBO instead of imageStore.
     * _pad[0] at offset 16 maps to viewInverse[1][0] read via floatBitsToUint. */
    if (gpu_is_direct_write(r->gpu)) {
        uint32_t use_direct = 1;
        memcpy(&pc._pad[0], &use_direct, sizeof(uint32_t));
        /* _pad[1] at offset 20 maps to viewInverse[1][1]: per-env layout flag.
         * When set, SSBO writes use [env, H, W] layout instead of tiled,
         * so CUDA can read directly without a de-tiling kernel. */
        if (r->per_env_layout) {
            uint32_t use_per_env = 1;
            memcpy(&pc._pad[1], &use_per_env, sizeof(uint32_t));
        }
    }
    /* _pad[2] at offset 24 maps to viewInverse[1][2]: sRGB encode flag.
     * When set, the raygen shader applies the sRGB transfer function before
     * UNORM quantize, so callers can skip a CPU LUT pass. */
    if (r->tiled_srgb) {
        uint32_t use_srgb = 1;
        memcpy(&pc._pad[2], &use_srgb, sizeof(uint32_t));
    }

    gpu_cmd_trace_rays_tiled(r->gpu, &pc);
    gpu_end_frame_tiled_rt(r->gpu);
    r->timings.trace_rays_tiled_ms = elapsed_ms(t_tiled);

    r->has_frame = 1;
    return NU_OK;
}

NuResult nu_fetch_pixels_tiled(NuRenderer* r, void* out_pixels,
                                int num_cameras, int tile_w, int tile_h)
{
    if (!r || !out_pixels) return NU_ERROR;
    if (!r->has_frame) {
        set_error(r, "No frame rendered yet");
        return NU_ERROR;
    }

    int num_cols = (int)ceilf(sqrtf((float)num_cameras));
    int num_rows = (num_cameras + num_cols - 1) / num_cols;
    uint32_t total_w = (uint32_t)(num_cols * tile_w);
    uint32_t total_h = (uint32_t)(num_rows * tile_h);

    /* Map staging buffer directly — avoids intermediate malloc + memcpy */
    const uint8_t* tiled_buf = gpu_map_tiled_staging(r->gpu, total_w, total_h);
    if (!tiled_buf) {
        set_error(r, "gpu_map_tiled_staging failed");
        return NU_ERROR;
    }

    /* De-tile: extract each camera's tile into contiguous output.
     * Output layout: camera 0 (tile_w * tile_h * 4 bytes), camera 1, ... */
    uint8_t* out = (uint8_t*)out_pixels;
    size_t tile_bytes = (size_t)tile_w * 4;  /* bytes per row of one tile */
    size_t tile_total = (size_t)tile_w * tile_h * 4;

    for (int cam = 0; cam < num_cameras; cam++) {
        int col = cam % num_cols;
        int row = cam / num_cols;

        uint8_t* dst = out + cam * tile_total;
        for (int y = 0; y < tile_h; y++) {
            const uint8_t* src = tiled_buf
                + (size_t)(row * tile_h + y) * total_w * 4
                + (size_t)col * tile_w * 4;
            memcpy(dst + y * tile_bytes, src, tile_bytes);
        }
    }

    return NU_OK;
}

NuResult nu_fetch_depth_tiled(NuRenderer* r, float* out_depth,
                               int num_cameras, int tile_w, int tile_h)
{
    if (!r || !out_depth) return NU_ERROR;
    if (!r->has_frame) {
        set_error(r, "No frame rendered yet");
        return NU_ERROR;
    }
    if (!r->depth_enabled) {
        set_error(r, "Depth output not enabled (call nu_set_depth_enabled first)");
        return NU_ERROR;
    }

    int num_cols = (int)ceilf(sqrtf((float)num_cameras));
    int num_rows = (num_cameras + num_cols - 1) / num_cols;
    uint32_t total_w = (uint32_t)(num_cols * tile_w);
    uint32_t total_h = (uint32_t)(num_rows * tile_h);

    if (!gpu_readback_tiled_depth(r->gpu, out_depth, total_w, total_h)) {
        set_error(r, "gpu_readback_tiled_depth failed");
        return NU_ERROR;
    }

    /* De-tile depth: extract each camera's tile into contiguous output.
     * Input is tiled grid, output is camera 0 (tile_w * tile_h floats), camera 1, ... */
    /* We need to de-tile in place or use a temp buffer. The readback wrote to out_depth
     * which now holds the raw tiled depth. De-tile into a temp buffer, then copy back. */
    size_t tile_floats = (size_t)tile_w * tile_h;
    size_t total_floats = (size_t)total_w * total_h;
    float* temp = (float*)malloc(total_floats * sizeof(float));
    if (!temp) return NU_ERROR;
    memcpy(temp, out_depth, total_floats * sizeof(float));

    for (int cam = 0; cam < num_cameras; cam++) {
        int col = cam % num_cols;
        int row = cam / num_cols;

        float* dst = out_depth + cam * tile_floats;
        for (int y = 0; y < tile_h; y++) {
            const float* src = temp
                + (size_t)(row * tile_h + y) * total_w
                + (size_t)col * tile_w;
            memcpy(dst + y * tile_w, src, (size_t)tile_w * sizeof(float));
        }
    }

    free(temp);
    return NU_OK;
}

NuResult nu_fetch_segmentation_tiled(NuRenderer* r, uint32_t* out_ids,
                                      int num_cameras, int tile_w, int tile_h)
{
    if (!r || !out_ids) return NU_ERROR;
    if (!r->has_frame) {
        set_error(r, "No frame rendered yet");
        return NU_ERROR;
    }
    if (!r->segmentation_enabled) {
        set_error(r, "Segmentation not enabled (call nu_set_segmentation_enabled first)");
        return NU_ERROR;
    }

    int num_cols = (int)ceilf(sqrtf((float)num_cameras));
    int num_rows = (num_cameras + num_cols - 1) / num_cols;
    uint32_t total_w = (uint32_t)(num_cols * tile_w);
    uint32_t total_h = (uint32_t)(num_rows * tile_h);

    if (!gpu_readback_tiled_segmentation(r->gpu, out_ids, total_w, total_h)) {
        set_error(r, "gpu_readback_tiled_segmentation failed");
        return NU_ERROR;
    }

    /* De-tile segmentation IDs: same layout as depth de-tiling */
    size_t tile_uints = (size_t)tile_w * tile_h;
    size_t total_uints = (size_t)total_w * total_h;
    uint32_t* temp = (uint32_t*)malloc(total_uints * sizeof(uint32_t));
    if (!temp) return NU_ERROR;
    memcpy(temp, out_ids, total_uints * sizeof(uint32_t));

    for (int cam = 0; cam < num_cameras; cam++) {
        int col = cam % num_cols;
        int row = cam / num_cols;

        uint32_t* dst = out_ids + cam * tile_uints;
        for (int y = 0; y < tile_h; y++) {
            const uint32_t* src = temp
                + (size_t)(row * tile_h + y) * total_w
                + (size_t)col * tile_w;
            memcpy(dst + y * tile_w, src, (size_t)tile_w * sizeof(uint32_t));
        }
    }

    free(temp);
    return NU_OK;
}

const void* nu_map_tiled_pixels_raw(NuRenderer* r,
                                     int num_cameras, int tile_w, int tile_h,
                                     int* out_total_w, int* out_total_h)
{
    if (!r) return NULL;
    if (!r->has_frame) {
        set_error(r, "No frame rendered yet");
        return NULL;
    }

    int num_cols = (int)ceilf(sqrtf((float)num_cameras));
    int num_rows = (num_cameras + num_cols - 1) / num_cols;
    uint32_t total_w = (uint32_t)(num_cols * tile_w);
    uint32_t total_h = (uint32_t)(num_rows * tile_h);

    const uint8_t* ptr = gpu_map_tiled_staging(r->gpu, total_w, total_h);
    if (!ptr) {
        set_error(r, "gpu_map_tiled_staging failed");
        return NULL;
    }

    if (out_total_w) *out_total_w = (int)total_w;
    if (out_total_h) *out_total_h = (int)total_h;
    return ptr;
}

const void* nu_map_tiled_pixels_raw_slot(NuRenderer* r,
                                          int num_cameras, int tile_w, int tile_h,
                                          int slot,
                                          int* out_total_w, int* out_total_h)
{
    if (!r) return NULL;
    int num_cols = (int)ceilf(sqrtf((float)num_cameras));
    int num_rows = (num_cameras + num_cols - 1) / num_cols;
    uint32_t total_w = (uint32_t)(num_cols * tile_w);
    uint32_t total_h = (uint32_t)(num_rows * tile_h);

    const uint8_t* ptr = gpu_map_tiled_staging_slot(r->gpu, total_w, total_h, slot);
    if (!ptr) {
        set_error(r, "gpu_map_tiled_staging_slot failed");
        return NULL;
    }

    if (out_total_w) *out_total_w = (int)total_w;
    if (out_total_h) *out_total_h = (int)total_h;
    return ptr;
}

int nu_get_last_tiled_slot(NuRenderer* r)
{
    if (!r) return -1;
    return gpu_get_last_tiled_slot(r->gpu);
}

NuResult nu_render_async(NuRenderer* r)
{
    if (!r) return NU_ERROR;

    /* Pack camera inverses for the tiled raygen shader's camera SSBO
     * (which expects num_cameras pairs of view_inv[16] + proj_inv[16]). */
    float vp_inv[32];
    compute_camera_inverses(&r->camera, vp_inv);

    /* Reuse the existing tiled path with num_cameras=1. This inherits the
     * tiled staging + cmdbuf-recycling machinery; on Metal today the backend
     * is single-buffered (gpu_metal.mm Phase 5b note) so the contract is
     * preserved without extra GPU/CPU overlap. */
    NuResult res = nu_render_tiled(r, vp_inv, /*num_cameras=*/1,
                                   r->width, r->height, NU_RENDER_RT);
    if (res != NU_OK) return res;

    /* Slot ping-pong: the just-submitted slot becomes async_curr_slot. The
     * slot from the prior render becomes async_prev_slot — that's the slot
     * the *next* fetch_async should read.
     *
     * Why next, not now? Because at this exact moment the GPU is still
     * working on async_curr_slot, but async_prev_slot's fence (from one
     * render ago) is essentially guaranteed to have signaled. The NEXT
     * fetch_async — paired with the NEXT render_async — picks up
     * async_prev_slot's pixels with minimal wait. */
    int slot = nu_get_last_tiled_slot(r);
    if (slot < 0) {
        set_error(r, "nu_get_last_tiled_slot returned -1");
        return NU_ERROR;
    }

    if (r->async_count >= 1) {
        r->async_prev_slot = r->async_curr_slot;
    }
    r->async_curr_slot = slot;
    if (r->async_count < 2) r->async_count++;
    return NU_OK;
}

NuResult nu_fetch_async(NuRenderer* r, void* out_pixels)
{
    if (!r || !out_pixels) return NU_ERROR;

    size_t bytes = (size_t)r->width * (size_t)r->height * 4;

    /* Pipe is still ramping up — we have <2 renders submitted, so there
     * isn't a "previous" frame to hand back. Zero-fill and succeed. The
     * first valid pixels arrive on the second fetch_async (after at least
     * 2 render_async calls). This keeps fetch a pure consumer with no GPU
     * sync on the very first frame. */
    if (r->async_count < 2) {
        memset(out_pixels, 0, bytes);
        return NU_OK;
    }

    /* Steady state: read the slot from the PREVIOUS render_async. The
     * underlying gpu_map_tiled_staging_slot() waits on that slot's fence;
     * typically it has long since signaled because the previous frame's
     * GPU work overlapped with this frame's submit + memcpy. */
    int slot = r->async_prev_slot;
    int total_w = 0, total_h = 0;
    const void* ptr = nu_map_tiled_pixels_raw_slot(r, /*num_cameras=*/1,
                                                   r->width, r->height, slot,
                                                   &total_w, &total_h);
    if (!ptr) {
        /* Slot wasn't actually submitted (shouldn't happen given count>=2,
         * but guard against it). Zero-fill so callers don't get stale data. */
        memset(out_pixels, 0, bytes);
        return NU_OK;
    }

    /* num_cameras=1 → total_w/total_h equal width/height, no de-tiling. */
    memcpy(out_pixels, ptr, bytes);
    return NU_OK;
}

NuResult nu_wait_tiled_complete(NuRenderer* r)
{
    if (!r) return NU_ERROR;
    if (!r->has_frame) {
        set_error(r, "No frame rendered yet");
        return NU_ERROR;
    }
    if (!gpu_wait_tiled_complete(r->gpu)) {
        set_error(r, "No tiled readback in flight");
        return NU_ERROR;
    }
    return NU_OK;
}

NuResult nu_wait_previous_tiled_complete(NuRenderer* r)
{
    if (!r) return NU_ERROR;
    if (!gpu_wait_previous_tiled_complete(r->gpu)) {
        set_error(r, "Previous tiled readback wait is not supported on Metal");
        return NU_ERROR_UNSUPPORTED;
    }
    return NU_OK;
}

int nu_interop_available(NuRenderer* r)
{
    if (!r) return 0;
    return gpu_interop_available(r->gpu);
}

void nu_set_skip_staging(NuRenderer* r, int skip)
{
    if (r) gpu_set_skip_staging_copy(r->gpu, skip);
}

void nu_set_fast_mode(NuRenderer* r, int fast)
{
    if (r) r->fast_mode = fast;
}

void nu_set_per_env_layout(NuRenderer* r, int enable)
{
    if (r) r->per_env_layout = enable;
}

void nu_set_tiled_srgb(NuRenderer* r, int enable)
{
    if (r) r->tiled_srgb = enable;
}

void nu_set_depth_enabled(NuRenderer* r, int enable)
{
    if (r) r->depth_enabled = enable;
}

void nu_set_segmentation_enabled(NuRenderer* r, int enable)
{
    if (r) r->segmentation_enabled = enable;
}

void nu_set_normals_enabled(NuRenderer* r, int enable)
{
    if (r) r->normals_enabled = enable;
}

void nu_set_deferred_shade(NuRenderer* r, int enable)
{
    (void)r;
    (void)enable;
}

void nu_set_deferred_debug_mode(NuRenderer* r, uint32_t mode)
{
    (void)r;
    (void)mode;
}

NuResult nu_fetch_normals_tiled(NuRenderer* r, float* out_normals,
                                 int num_cameras, int tile_w, int tile_h)
{
    if (!r || !out_normals) return NU_ERROR;
    if (!r->has_frame) {
        set_error(r, "No frame rendered yet");
        return NU_ERROR;
    }
    if (!r->normals_enabled) {
        set_error(r, "Normals not enabled (call nu_set_normals_enabled first)");
        return NU_ERROR;
    }

    int num_cols = (int)ceilf(sqrtf((float)num_cameras));
    int num_rows = (num_cameras + num_cols - 1) / num_cols;
    uint32_t total_w = (uint32_t)(num_cols * tile_w);
    uint32_t total_h = (uint32_t)(num_rows * tile_h);

    if (!gpu_readback_tiled_normals(r->gpu, out_normals, total_w, total_h)) {
        set_error(r, "gpu_readback_tiled_normals failed");
        return NU_ERROR;
    }

    /* De-tile normals: 3 floats per pixel */
    size_t tile_floats = (size_t)tile_w * tile_h * 3;
    size_t total_floats = (size_t)total_w * total_h * 3;
    float* temp = (float*)malloc(total_floats * sizeof(float));
    if (!temp) return NU_ERROR;
    memcpy(temp, out_normals, total_floats * sizeof(float));

    for (int cam = 0; cam < num_cameras; cam++) {
        int col = cam % num_cols;
        int row = cam / num_cols;

        float* dst = out_normals + cam * tile_floats;
        for (int y = 0; y < tile_h; y++) {
            const float* src = temp
                + ((size_t)(row * tile_h + y) * total_w
                   + (size_t)col * tile_w) * 3;
            memcpy(dst + y * tile_w * 3, src, (size_t)tile_w * 3 * sizeof(float));
        }
    }

    free(temp);
    return NU_OK;
}

NuResult nu_get_cuda_interop_info(NuRenderer* r, NuCudaInteropInfo* out,
                                   int num_cameras, int tile_w, int tile_h)
{
    if (!r || !out) return NU_ERROR;
    if (!gpu_interop_available(r->gpu)) {
        set_error(r, "CUDA interop not available");
        return NU_ERROR_UNSUPPORTED;
    }

    int num_cols = (int)ceilf(sqrtf((float)num_cameras));
    int num_rows = (num_cameras + num_cols - 1) / num_cols;

    /* Export both double-buffered interop buffers */
    for (int i = 0; i < 2; i++) {
        out->mem_fd[i] = gpu_export_tiled_image_fd(r->gpu, i);
        if (out->mem_fd[i] < 0) {
            for (int j = 0; j < i; j++) {
                close(out->mem_fd[j]);
                out->mem_fd[j] = -1;
            }
            set_error(r, "Failed to export tiled image memory fd");
            return NU_ERROR;
        }
    }

    out->mem_size    = gpu_get_tiled_image_alloc_size(r->gpu);
    out->image_w     = (uint32_t)(num_cols * tile_w);
    out->image_h     = (uint32_t)(num_rows * tile_h);
    out->tile_w      = (uint32_t)tile_w;
    out->tile_h      = (uint32_t)tile_h;
    out->num_cameras = num_cameras;

    out->sem_fd = gpu_export_timeline_semaphore_fd(r->gpu);
    if (out->sem_fd < 0) {
        for (int i = 0; i < 2; i++) {
            close(out->mem_fd[i]);
            out->mem_fd[i] = -1;
        }
        set_error(r, "Failed to export timeline semaphore fd");
        return NU_ERROR;
    }

    out->sem_value = gpu_get_interop_timeline_value(r->gpu);
    return NU_OK;
}

NuResult nu_set_external_output_buffers(NuRenderer* r,
                                        int mem_fds[2],
                                        uint64_t mem_size_each,
                                        int num_cameras,
                                        int tile_w,
                                        int tile_h)
{
    (void)mem_fds;
    (void)mem_size_each;
    (void)num_cameras;
    (void)tile_w;
    (void)tile_h;
    if (!r) return NU_ERROR;
    set_error(r, "nu_set_external_output_buffers is not supported by the Metal backend");
    return NU_ERROR_UNSUPPORTED;
}

NuResult nu_get_external_timeline_semaphore_fd(NuRenderer* r,
                                               int* out_sem_fd,
                                               uint64_t* out_sem_value)
{
    if (out_sem_fd) *out_sem_fd = -1;
    if (out_sem_value) *out_sem_value = 0;
    if (!r) return NU_ERROR;
    set_error(r, "nu_get_external_timeline_semaphore_fd is not supported by the Metal backend");
    return NU_ERROR_UNSUPPORTED;
}

NuResult nu_get_transforms_interop_info(NuRenderer* r, int count,
                                        NuTransformsInteropInfo* out)
{
    (void)count;
    if (out) {
        out->mem_fd = -1;
        out->mem_size = 0;
        out->count = 0;
    }
    if (!r) return NU_ERROR;
    set_error(r, "nu_get_transforms_interop_info is not supported by the Metal backend");
    return NU_ERROR_UNSUPPORTED;
}

NuResult nu_set_transform_layout(NuRenderer* r, const int* mesh_ids, int count)
{
    (void)mesh_ids;
    (void)count;
    if (!r) return NU_ERROR;
    set_error(r, "nu_set_transform_layout is not supported by the Metal backend");
    return NU_ERROR_UNSUPPORTED;
}

NuResult nu_translate_instances_gpu(NuRenderer* r)
{
    if (!r) return NU_ERROR;
    set_error(r, "nu_translate_instances_gpu is not supported by the Metal backend");
    return NU_ERROR_UNSUPPORTED;
}

int nu_get_interop_read_idx(NuRenderer* r)
{
    if (!r) return 0;
    return gpu_get_interop_read_idx(r->gpu);
}

int nu_get_interop_prev_idx(NuRenderer* r)
{
    if (!r) return 0;
    return gpu_get_interop_prev_idx(r->gpu);
}

void* nu_map_pixels_gpu(NuRenderer* r)
{
    /* Legacy stub */
    (void)r;
    return NULL;
}

void nu_unmap_pixels_gpu(NuRenderer* r)
{
    (void)r;
}

/* ---- Raycast (LiDAR/radar) ---- */

static int ensure_raycast_pipeline(NuRenderer* r)
{
    if (r->raycast_pipeline_built) return 1;

    /* Need RT + built TLAS */
    if (!r->enable_rt || !gpu_rt_available(r->gpu) || !gpu_rt_built(r->gpu))
        return 0;

    /* Load compute shader */
    char comp_path[1024];
    snprintf(comp_path, sizeof(comp_path), "%s/raycast.comp.spv", SHADER_DIR);

    uint32_t comp_sz = 0;
    uint32_t* comp_spv = load_shader_file(comp_path, &comp_sz);
    if (!comp_spv) {
        set_error(r, "Failed to load raycast compute shader");
        return 0;
    }

    gpu_destroy_raycast_pipeline(r->gpu);
    int ok = gpu_build_raycast_pipeline(r->gpu, comp_spv, comp_sz);
    free(comp_spv);

    if (!ok) {
        set_error(r, "Failed to build raycast compute pipeline");
        return 0;
    }

    r->raycast_pipeline_built = 1;
    return 1;
}

NuResult nu_cast_rays(NuRenderer* r,
                      const float* ray_origins,
                      const float* ray_directions,
                      int num_rays,
                      float max_distance,
                      float* out_distances,
                      float* out_normals,
                      float* out_hit_positions)
{
    if (!r) return NU_ERROR;
    if (num_rays <= 0) return NU_OK;
    if (!ray_origins || !ray_directions || !out_distances ||
        !out_normals || !out_hit_positions) {
        set_error(r, "NULL pointer argument");
        return NU_ERROR;
    }

    /* Rebuild GPU buffers if scene changed */
    if (r->scene_dirty) {
        if (!rebuild_gpu_buffers(r)) return NU_ERROR;
    }

    if (r->total_indices == 0) {
        set_error(r, "No geometry to cast rays against");
        return NU_ERROR_NO_SCENE;
    }

    /* Rebuild accel structures if needed */
    if (r->accel_dirty) {
        if (!rebuild_accel(r)) return NU_ERROR;
    }

    /* Rebuild TLAS if transforms changed */
    if (r->tlas_dirty) {
        /* USD row-vector 4x4 → VkTransformMatrixKHR 3x4 — see usd4x4_to_vk3x4.
         * Earlier revisions did a flat 12-float copy, which silently dropped
         * the row-3 translation. Same fix as the tiled-RT and single-camera
         * paths above. */
        float* xforms_3x4 = (float*)malloc(r->nmeshes * 12 * sizeof(float));
        if (xforms_3x4) {
            for (int m = 0; m < r->nmeshes; m++) {
                usd4x4_to_vk3x4(r->meshes[m].transform, &xforms_3x4[m * 12]);
            }
            uint8_t* masks = (uint8_t*)malloc(r->nmeshes);
            if (masks)
                fill_rt_instance_masks(r, masks);
            gpu_update_tlas(r->gpu, xforms_3x4, masks, (uint32_t)r->nmeshes);
            free(masks);
            free(xforms_3x4);
        }
        r->tlas_dirty = 0;
    }

    /* Ensure raycast compute pipeline is built */
    if (!ensure_raycast_pipeline(r)) {
        set_error(r, "Raycast pipeline not available (RT required)");
        return NU_ERROR_NO_RT;
    }

    /* Cast rays */
    if (!gpu_cast_rays(r->gpu, ray_origins, ray_directions,
                       num_rays, max_distance,
                       out_distances, out_normals, out_hit_positions)) {
        set_error(r, "gpu_cast_rays failed");
        return NU_ERROR;
    }

    return NU_OK;
}

NuResult nu_cast_rays_async(NuRenderer* r,
                            const float* ray_origins,
                            const float* ray_directions,
                            int num_rays,
                            float max_distance)
{
    if (!r) return NU_ERROR;
    if (num_rays <= 0) return NU_OK;
    if (!ray_origins || !ray_directions) {
        set_error(r, "NULL pointer argument");
        return NU_ERROR;
    }

    if (r->scene_dirty) {
        if (!rebuild_gpu_buffers(r)) return NU_ERROR;
    }
    if (r->total_indices == 0) {
        set_error(r, "No geometry to cast rays against");
        return NU_ERROR_NO_SCENE;
    }
    if (r->accel_dirty) {
        if (!rebuild_accel(r)) return NU_ERROR;
    }
    if (r->tlas_dirty) {
        /* USD row-vector 4x4 → VkTransformMatrixKHR 3x4 — see usd4x4_to_vk3x4.
         * Earlier revisions did a flat 12-float copy, which silently dropped
         * the row-3 translation. Same fix as the tiled-RT and single-camera
         * paths above. */
        float* xforms_3x4 = (float*)malloc(r->nmeshes * 12 * sizeof(float));
        if (xforms_3x4) {
            for (int m = 0; m < r->nmeshes; m++) {
                usd4x4_to_vk3x4(r->meshes[m].transform, &xforms_3x4[m * 12]);
            }
            uint8_t* masks = (uint8_t*)malloc(r->nmeshes);
            if (masks)
                fill_rt_instance_masks(r, masks);
            gpu_update_tlas(r->gpu, xforms_3x4, masks, (uint32_t)r->nmeshes);
            free(masks);
            free(xforms_3x4);
        }
        r->tlas_dirty = 0;
    }
    if (!ensure_raycast_pipeline(r)) {
        set_error(r, "Raycast pipeline not available (RT required)");
        return NU_ERROR_NO_RT;
    }

    if (!gpu_cast_rays_async(r->gpu, ray_origins, ray_directions,
                             num_rays, max_distance)) {
        set_error(r, "gpu_cast_rays_async failed");
        return NU_ERROR;
    }
    return NU_OK;
}

NuResult nu_cast_rays_wait(NuRenderer* r,
                           float* out_distances,
                           float* out_normals,
                           float* out_hit_positions)
{
    if (!r) return NU_ERROR;
    if (!out_distances || !out_normals || !out_hit_positions) {
        set_error(r, "NULL output pointer");
        return NU_ERROR;
    }

    if (!gpu_cast_rays_wait(r->gpu, out_distances, out_normals, out_hit_positions)) {
        set_error(r, "gpu_cast_rays_wait failed");
        return NU_ERROR;
    }
    return NU_OK;
}

NuResult nu_raycast_get_interop_info(NuRenderer* r, int num_rays,
                                      NuRaycastInteropInfo* out)
{
    if (!r || !out || num_rays <= 0) return NU_ERROR;

    if (r->scene_dirty) {
        if (!rebuild_gpu_buffers(r)) return NU_ERROR;
    }
    if (r->accel_dirty) {
        if (!rebuild_accel(r)) return NU_ERROR;
    }
    if (!ensure_raycast_pipeline(r)) {
        set_error(r, "raycast pipeline setup failed");
        return NU_ERROR;
    }

    if (!gpu_raycast_get_interop_fds(r->gpu, num_rays,
            &out->input_fd, &out->input_size,
            &out->output_fd, &out->output_size,
            &out->max_rays)) {
        set_error(r, "gpu_raycast_get_interop_fds failed");
        return NU_ERROR;
    }
    return NU_OK;
}

NuResult nu_cast_rays_wait_fence(NuRenderer* r)
{
    if (!r) return NU_ERROR;
    if (!gpu_cast_rays_wait_fence(r->gpu)) {
        set_error(r, "gpu_cast_rays_wait_fence failed");
        return NU_ERROR;
    }
    return NU_OK;
}

NuResult nu_cast_rays_gpu(NuRenderer* r, int num_rays, float max_distance)
{
    if (!r) return NU_ERROR;
    if (num_rays <= 0) return NU_OK;

    if (r->scene_dirty) {
        if (!rebuild_gpu_buffers(r)) return NU_ERROR;
    }
    if (r->accel_dirty) {
        if (!rebuild_accel(r)) return NU_ERROR;
    }
    if (!ensure_raycast_pipeline(r)) {
        set_error(r, "raycast pipeline setup failed");
        return NU_ERROR;
    }

    if (!gpu_cast_rays_gpu(r->gpu, num_rays, max_distance)) {
        set_error(r, "gpu_cast_rays_gpu failed");
        return NU_ERROR;
    }
    return NU_OK;
}

NuResult nu_save_ppm(NuRenderer* r, const char* path)
{
    if (!r || !path) return NU_ERROR;
    if (!r->has_frame) {
        set_error(r, "No frame rendered yet");
        return NU_ERROR;
    }

    if (!gpu_screenshot(r->gpu, path)) {
        set_error(r, "gpu_screenshot failed");
        return NU_ERROR;
    }

    return NU_OK;
}

/* ---- Configuration ---- */

NuResult nu_set_render_size(NuRenderer* r, int width, int height)
{
    if (!r || width <= 0 || height <= 0) return NU_ERROR;

    r->width = width;
    r->height = height;
    gpu_resize(r->gpu, width, height);
    r->camera.aspect = (float)width / (float)height;

    free(r->readback_pixels);
    r->readback_pixels = (uint8_t*)malloc((size_t)width * height * 4);

    /* Reset async slot tracking — staging buffers will be reallocated on
     * the next nu_render_tiled() / nu_render_async() call. */
    r->async_count = 0;
    r->async_curr_slot = 0;
    r->async_prev_slot = 0;

    return NU_OK;
}

int nu_rt_available(NuRenderer* r)
{
    return r ? gpu_rt_available(r->gpu) : 0;
}

void nu_get_render_size(NuRenderer* r, int* width, int* height)
{
    if (!r) return;
    if (width)  *width  = r->width;
    if (height) *height = r->height;
}

void* nu_get_window(NuRenderer* r)
{
    (void)r;
    return NULL;
}

/* ---- Environment ---- */

NuResult nu_load_environment(NuRenderer* r, const char* hdr_path)
{
    return nu_load_environment_intensity(r, hdr_path, -1.0f);
}

NuResult nu_load_environment_intensity(NuRenderer* r, const char* hdr_path,
                                       float intensity)
{
    if (!r || !hdr_path) return NU_ERROR;
    if (intensity >= 0.0f)
        gpu_set_env_intensity(r->gpu, intensity);

    if (gpu_load_environment_with_fallbacks(r->gpu, hdr_path, NULL, 0))
        return NU_OK;

    set_error(r, "Failed to load environment map");
    return NU_ERROR;
}

int nu_load_texture(NuRenderer* r, const uint8_t* pixels, int width, int height)
{
    (void)pixels;
    (void)width;
    (void)height;
    if (r) set_error(r, "nu_load_texture is not supported by the Metal backend yet");
    return NU_ERROR_UNSUPPORTED;
}

NuResult nu_set_mesh_texture(NuRenderer* r, int mesh_id, int tex_index)
{
    (void)mesh_id;
    (void)tex_index;
    if (!r) return NU_ERROR;
    set_error(r, "nu_set_mesh_texture is not supported by the Metal backend yet");
    return NU_ERROR_UNSUPPORTED;
}

NuResult nu_set_dome_color(NuRenderer* r, float r_, float g, float b,
                           float intensity)
{
    if (!r) return NU_ERROR;
    gpu_set_dome_color(r->gpu, r_, g, b, intensity);
    return NU_OK;
}

/* ---- Info ---- */

int nu_get_mesh_count(NuRenderer* r)
{
    if (!r) return 0;
    int count = 0;
    for (int i = 0; i < r->nmeshes; i++) {
        if (r->meshes[i].active) count++;
    }
    return count;
}

NuResult nu_get_scene_bounds(NuRenderer* r, float out_min[3], float out_max[3])
{
    if (!r || !out_min || !out_max) return NU_ERROR;
    Scene* scene = r->usd_scene;
    if (scene) {
        out_min[0] = scene->bounds_min[0];
        out_min[1] = scene->bounds_min[1];
        out_min[2] = scene->bounds_min[2];
        out_max[0] = scene->bounds_max[0];
        out_max[1] = scene->bounds_max[1];
        out_max[2] = scene->bounds_max[2];
        return NU_OK;
    }
    if (!r->has_cached_bounds) return NU_ERROR;
    memcpy(out_min, r->cached_bounds_min, 3 * sizeof(float));
    memcpy(out_max, r->cached_bounds_max, 3 * sizeof(float));
    return NU_OK;
}

int nu_get_mesh_name(NuRenderer* r, int mesh_id, char* out_buf, int buf_cap)
{
    if (!r) return NU_ERROR;
    if (mesh_id < 0 || mesh_id >= r->nmeshes) return NU_ERROR_BAD_ID;
    if (!r->meshes[mesh_id].active) return NU_ERROR_BAD_ID;

    const char* name = r->meshes[mesh_id].name ? r->meshes[mesh_id].name : "";
    int len = (int)strlen(name);

    if (buf_cap <= 0) return len;          /* size query */
    if (!out_buf)     return NU_ERROR;

    int copy = (len < buf_cap - 1) ? len : (buf_cap - 1);
    memcpy(out_buf, name, (size_t)copy);
    out_buf[copy] = '\0';
    return copy;
}

NuResult nu_get_mesh_transform(NuRenderer* r, int mesh_id, float out_mat4x4[16])
{
    if (!r || !out_mat4x4) return NU_ERROR;
    if (mesh_id < 0 || mesh_id >= r->nmeshes) return NU_ERROR_BAD_ID;
    if (!r->meshes[mesh_id].active) return NU_ERROR_BAD_ID;
    transpose4x4(r->meshes[mesh_id].transform, out_mat4x4);
    return NU_OK;
}

int nu_get_curve_segment_count(NuRenderer* r)
{
    return r ? r->curves.nseg : 0;
}

uint64_t nu_get_gpu_memory_used(NuRenderer* r)
{
    return r ? gpu_get_allocated_memory(r->gpu) : 0;
}

NuResult nu_get_meshlet_stats(NuRenderer* r, NuMeshletStats* out_stats)
{
    if (!r || !out_stats) return NU_ERROR;

    memset(out_stats, 0, sizeof(*out_stats));
    out_stats->version = NU_MESHLET_STATS_VERSION;
    out_stats->struct_size = (uint32_t)sizeof(*out_stats);
    out_stats->total_meshlets = r->meshlet_count;
    out_stats->total_meshlet_indices = r->total_meshlet_indices;
    out_stats->meshlet_raster_enabled = (uint32_t)r->meshlet_raster_enabled;
    out_stats->max_vertices = NUSD_MESHLET_MAX_VERTICES;
    out_stats->max_triangles = NUSD_MESHLET_MAX_TRIANGLES;
    uint64_t meshlet_index_bytes = (uint64_t)r->total_meshlet_indices * sizeof(uint32_t);
    out_stats->cpu_index_bytes = r->cpu_meshlet_indices ? meshlet_index_bytes : 0;
    out_stats->gpu_index_bytes = (r->meshlet_index_buffer && r->meshlet_raster_enabled)
        ? meshlet_index_bytes : 0;

    for (int i = 0; i < r->nmeshes; i++) {
        RendererMesh* m = &r->meshes[i];
        if (!m->active || m->prototype_idx != i || m->meshlet_count == 0) continue;
        out_stats->active_meshes_with_meshlets++;
        out_stats->active_meshlets += m->meshlet_count;
        out_stats->active_meshlet_indices += m->meshlet_index_count;
    }

    return NU_OK;
}

void nu_get_cmd_cache_stats(NuRenderer* r,
                            uint64_t* out_rt_replays,
                            uint64_t* out_rt_records,
                            uint64_t* out_tiled_replays,
                            uint64_t* out_tiled_records)
{
    (void)r;
    if (out_rt_replays) *out_rt_replays = 0;
    if (out_rt_records) *out_rt_records = 0;
    if (out_tiled_replays) *out_tiled_replays = 0;
    if (out_tiled_records) *out_tiled_records = 0;
}

NuResult nu_get_phase_timings_ms(NuRenderer* r, NuPhaseTimings* out)
{
    if (!r) return NU_ERROR;
    if (!out) return NU_OK;
    *out = r->timings;
    return NU_OK;
}

const char* nu_get_last_error(NuRenderer* r)
{
    return r ? r->last_error : "NULL renderer";
}
