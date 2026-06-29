// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * renderer.c — Implementation of nusd_renderer.h public API.
 *
 * Uses a hidden GLFW window to satisfy Vulkan surface requirements (v1).
 * Internally wraps gpu.h for Vulkan rendering and scene.h for USD loading.
 * All rendering happens offscreen; pixels are read back via staging buffer.
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "nusd_renderer.h"
#include "gpu.h"
#include "scene.h"
#include "camera.h"
#include "material.h"
#include "gs_scene.h"
#include <nanousd/nanousdapi.h>  /* NanousdStage + nanousd_close — needed
                                  * by nu_clear_scene's lazy_stage_pending
                                  * cleanup path. */
#include <meshoptimizer.h>

#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>  /* close() for fd cleanup on error */
#include <float.h>
#include <errno.h>
#include <limits.h>
#ifdef __GLIBC__
#include <malloc.h>  /* malloc_trim() — glibc extension, lets nu_finalize_scene
                     * return freed CPU mirror pages to the kernel so VmRSS
                     * reflects the drop without waiting for arena reclaim. */
#endif

/* ---- NU_TILE_RES probe (BVH-side ray-count lever) ----
 *
 * Opt-in env var that scales every (tile_w, tile_h) value flowing through
 * the tiled-render API by a fixed factor relative to the IsaacLab default
 * of 64. Setting NU_TILE_RES=32 halves each tile dimension (4x ray cut);
 * setting NU_TILE_RES=16 quarters them (16x). Default (unset OR =64) is a
 * no-op — bytes through the renderer are byte-identical to the pre-probe
 * baseline.
 *
 * The factor is captured once on the first scaling call (steady state =
 * one int compare) and applied UNIFORMLY at every C-API entry point that
 * takes (tile_w, tile_h): nu_render_tiled, nu_fetch_*_tiled,
 * nu_map_tiled_pixels_raw{,_slot}, nu_get_cuda_interop_info,
 * nu_set_external_output_buffers. That uniformity keeps buffer geometry
 * consistent — the renderer writes scaled tiles, fetchers read scaled
 * tiles, interop info reports scaled image_w/image_h/tile_w/tile_h.
 *
 * Caller-side note: callers that pre-allocate output buffers from their
 * OWN (unscaled) tile_w/tile_h (e.g. the IsaacLab adapter's warp arrays
 * at 64x64) will see a layout MISMATCH — the renderer writes scaled
 * tiles, packed contiguously, not at the caller's assumed stride. The
 * caller must either (a) read the SCALED dims back from
 * nu_get_cuda_interop_info and reshape accordingly, or (b) read the env
 * var itself and pass scaled dims into the API in the first place. This
 * probe is a "does the policy tolerate degraded vision?" question, not
 * a correctness guarantee — that's why it's an env-var opt-in. */
static int g_tile_res_scale_x100 = -1;  /* -1 = uninitialized; 100 = no-op */
static int g_glfw_refcount = 0;

static int acquire_glfw(void)
{
    if (g_glfw_refcount == 0) {
        if (!glfwInit()) {
            return 0;
        }
    }
    g_glfw_refcount++;
    return 1;
}

static int dome_visible_fallback_path(const char* hdr_path,
                                      char* out,
                                      size_t out_size)
{
    if (!hdr_path || !out || out_size == 0) return 0;
    const char* dot = strrchr(hdr_path, '.');
    if (!dot || strcasecmp(dot, ".exr") != 0) return 0;
    size_t stem_len = (size_t)(dot - hdr_path);
    const char suffix[] = "VIS.png";
    if (stem_len + sizeof(suffix) > out_size) return 0;
    memcpy(out, hdr_path, stem_len);
    memcpy(out + stem_len, suffix, sizeof(suffix));
    return access(out, R_OK) == 0;
}

/* The Disney Moana package ships an EXR dome path plus a separate
 * islandsunVIS.png preview. The EXR can be unavailable in lightweight
 * checkouts; the VIS PNG is already display-referred LDR, so it needs the
 * same internal visible-sky scale used by nanousdview's default viewer HDR
 * instead of the authored DomeLight intensity of 1.0. */
static float visible_dome_fallback_intensity(void)
{
    const char* e = getenv("NUSD_VISIBLE_DOME_FALLBACK_INTENSITY");
    if (e && e[0]) {
        char* end = NULL;
        float v = strtof(e, &end);
        if (end != e && isfinite(v) && fabsf(v) > 1.0f)
            return v < 0.0f ? v : -v;
    }
    return -1500.0f;
}

static void release_glfw(void)
{
    if (g_glfw_refcount <= 0) {
        return;
    }
    g_glfw_refcount--;
    if (g_glfw_refcount == 0) {
        glfwTerminate();
    }
}

static int effective_tile(int t)
{
    if (g_tile_res_scale_x100 < 0) {
        const char* s = getenv("NU_TILE_RES");
        int n = (s && *s) ? atoi(s) : 64;
        if (n > 0 && n != 64) {
            g_tile_res_scale_x100 = (n * 100) / 64;
            fprintf(stderr,
                "[nu_renderer] NU_TILE_RES=%d -> effective tile scale %d/100 "
                "(default tile_w=tile_h=64 will become %d)\n",
                n, g_tile_res_scale_x100, n);
        } else {
            g_tile_res_scale_x100 = 100;  /* no-op */
        }
    }
    if (g_tile_res_scale_x100 == 100) return t;
    int s = (t * g_tile_res_scale_x100) / 100;
    return s > 0 ? s : 1;  /* clamp to >=1 to keep dispatches valid */
}

/* ---- NU_FRAME_SKIP probe (RT-trace skip lever) ----
 *
 * Opt-in env var that renders only every Nth call to nu_render_tiled and
 * returns the cached prior tiled output on intermediate calls without
 * re-tracing rays. Default (unset OR =1) is a no-op — every frame renders
 * and bytes through the renderer are byte-identical to the pre-probe
 * baseline.
 *
 * Mechanism: a static counter is bumped on each render call. When
 * (counter % N) != 0 AND a prior frame has been rendered (r->has_frame),
 * we return NU_OK without touching any GPU state, leaving the slot
 * ping-pong (interop_write_idx, tiled_readback_write_idx) and dirty bits
 * (scene_dirty, accel_dirty, tlas_dirty, colors_dirty) untouched. All
 * downstream readers compute their slot as (1 - write_idx), so they
 * receive the most-recently-rendered frame again. Dirty bits accumulate
 * and are processed on the next non-skipped frame ("latest transforms
 * win"). The very first call always renders (has_frame guard).
 *
 * This is a "does the policy tolerate stale visuals?" probe — the scene
 * barely changes between IsaacLab physics steps, so N=2 should look
 * indistinguishable from N=1 while halving vkCmdTraceRaysKHR cost. */
static int g_frame_skip_n     = -1;  /* -1 = uninitialized; 1 = no-op */
static unsigned long g_frame_count = 0;

static int frame_skip_active(void)
{
    if (g_frame_skip_n < 0) {
        const char* s = getenv("NU_FRAME_SKIP");
        int n = (s && *s) ? atoi(s) : 1;
        if (n > 1) {
            g_frame_skip_n = n;
            fprintf(stderr,
                "[nu_renderer] NU_FRAME_SKIP=%d -> rendering every %dth frame "
                "(skipped frames return cached prior tiled output)\n", n, n);
        } else {
            g_frame_skip_n = 1;  /* no-op */
        }
    }
    return g_frame_skip_n;
}

/* OVRTX/Omniverse-style exposure metadata. OVRTX's public wrapper copies
 * camera exposure attrs to its camera and tonemap attrs to its render product.
 * These helpers map the same authored data to a relative multiplier on this
 * renderer's existing calibrated ACES passes, so un-authored scenes keep their
 * current look exactly. */
static int parse_float_env2(const char* name_a, const char* name_b, float* out)
{
    const char* s = getenv(name_a);
    if ((!s || !*s) && name_b) s = getenv(name_b);
    if (!s || !*s) return 0;
    char* end = NULL;
    float v = strtof(s, &end);
    if (end == s || !isfinite(v)) return 0;
    *out = v;
    return 1;
}

static int parse_bool_env2(const char* name_a, const char* name_b, int* out)
{
    const char* s = getenv(name_a);
    if ((!s || !*s) && name_b) s = getenv(name_b);
    if (!s || !*s) return 0;
    if (strcmp(s, "1") == 0 || strcasecmp(s, "true") == 0 ||
        strcasecmp(s, "yes") == 0 || strcasecmp(s, "on") == 0) {
        *out = 1;
        return 1;
    }
    if (strcmp(s, "0") == 0 || strcasecmp(s, "false") == 0 ||
        strcasecmp(s, "no") == 0 || strcasecmp(s, "off") == 0) {
        *out = 0;
        return 1;
    }
    return 0;
}

static int parse_float_attr_line(const char* text, const char* name, float* out)
{
    size_t name_len = strlen(name);
    const char* p = text;
    while ((p = strstr(p, name)) != NULL) {
        const char* line_end = strchr(p, '\n');
        if (!line_end) line_end = p + strlen(p);
        const char* q = p + name_len;
        while (q < line_end && *q != '=') q++;
        if (q < line_end && *q == '=') {
            char* end = NULL;
            float v = strtof(q + 1, &end);
            if (end != q + 1 && isfinite(v)) {
                *out = v;
                return 1;
            }
        }
        p += name_len;
    }
    return 0;
}

static int parse_bool_attr_line(const char* text, const char* name, int* out)
{
    size_t name_len = strlen(name);
    const char* p = text;
    while ((p = strstr(p, name)) != NULL) {
        const char* line_end = strchr(p, '\n');
        if (!line_end) line_end = p + strlen(p);
        const char* q = p + name_len;
        while (q < line_end && *q != '=') q++;
        if (q < line_end && *q == '=') {
            q++;
            while (q < line_end && (*q == ' ' || *q == '\t')) q++;
            if (q < line_end) {
                if (*q == '1' || strncasecmp(q, "true", 4) == 0) {
                    *out = 1;
                    return 1;
                }
                if (*q == '0' || strncasecmp(q, "false", 5) == 0) {
                    *out = 0;
                    return 1;
                }
            }
        }
        p += name_len;
    }
    return 0;
}

static char* read_usd_scan_text(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    const size_t max_bytes = 8u * 1024u * 1024u;
    char* text = (char*)malloc(max_bytes + 1u);
    if (!text) {
        fclose(f);
        return NULL;
    }
    size_t n = fread(text, 1u, max_bytes, f);
    fclose(f);
    text[n] = '\0';
    return text;
}

static char* renderer_strdup(const char* s);

static void exposure_desc_merge_text(const char* text, NuExposureDesc* desc)
{
    if (!text || !desc) return;
    if (!(desc->flags & NU_EXPOSURE_HAS_FSTOP) &&
        parse_float_attr_line(text, "exposure:fStop", &desc->exposure_f_stop))
        desc->flags |= NU_EXPOSURE_HAS_FSTOP;
    if (!(desc->flags & NU_EXPOSURE_HAS_RESPONSIVITY) &&
        parse_float_attr_line(text, "exposure:responsivity", &desc->exposure_responsivity))
        desc->flags |= NU_EXPOSURE_HAS_RESPONSIVITY;
    if (!(desc->flags & NU_EXPOSURE_HAS_TIME) &&
        parse_float_attr_line(text, "exposure:time", &desc->exposure_time))
        desc->flags |= NU_EXPOSURE_HAS_TIME;
    if (!(desc->flags & NU_EXPOSURE_HAS_AUTO_EXPOSURE) &&
        parse_bool_attr_line(text, "omni:rtx:autoExposure:enabled", &desc->auto_exposure_enabled))
        desc->flags |= NU_EXPOSURE_HAS_AUTO_EXPOSURE;
    if (!(desc->flags & NU_EXPOSURE_HAS_WHITE_POINT) &&
        parse_float_attr_line(text, "omni:rtx:autoExposure:whitePointScale", &desc->white_point_scale))
        desc->flags |= NU_EXPOSURE_HAS_WHITE_POINT;
    if (!(desc->flags & NU_EXPOSURE_HAS_TONEMAP_FNUM) &&
        (parse_float_attr_line(text, "rtx:post:tonemap:fNumber", &desc->tonemap_f_number) ||
         parse_float_attr_line(text, "omni:rtx:post:tonemap:fNumber", &desc->tonemap_f_number)))
        desc->flags |= NU_EXPOSURE_HAS_TONEMAP_FNUM;
    if (!(desc->flags & NU_EXPOSURE_HAS_TONEMAP_CM2) &&
        (parse_float_attr_line(text, "rtx:post:tonemap:cm2Factor", &desc->tonemap_cm2_factor) ||
         parse_float_attr_line(text, "omni:rtx:post:tonemap:cm2Factor", &desc->tonemap_cm2_factor)))
        desc->flags |= NU_EXPOSURE_HAS_TONEMAP_CM2;
}

static int exposure_desc_complete(const NuExposureDesc* desc)
{
    const uint32_t want = NU_EXPOSURE_HAS_FSTOP |
                          NU_EXPOSURE_HAS_RESPONSIVITY |
                          NU_EXPOSURE_HAS_TIME |
                          NU_EXPOSURE_HAS_AUTO_EXPOSURE |
                          NU_EXPOSURE_HAS_WHITE_POINT |
                          NU_EXPOSURE_HAS_TONEMAP_FNUM |
                          NU_EXPOSURE_HAS_TONEMAP_CM2;
    return desc && ((desc->flags & want) == want);
}

static int usd_asset_ref_ext_ok(const char* s, size_t n)
{
    if (!s || n < 4) return 0;
    const char* end = s + n;
    const char* p = end;
    while (p > s && p[-1] != '/' && p[-1] != '\\') p--;
    const char* dot = NULL;
    for (const char* q = p; q < end; q++)
        if (*q == '.') dot = q;
    if (!dot) return 0;
    size_t ext_n = (size_t)(end - dot);
    if (ext_n == 4 && strncasecmp(dot, ".usd", 4) == 0) return 1;
    if (ext_n == 5 &&
        (strncasecmp(dot, ".usda", 5) == 0 ||
         strncasecmp(dot, ".usdc", 5) == 0)) return 1;
    return 0;
}

static char* path_dirname_dup(const char* path)
{
    if (!path) return NULL;
    const char* slash = strrchr(path, '/');
    if (!slash) return renderer_strdup(".");
    size_t n = (slash == path) ? 1u : (size_t)(slash - path);
    char* out = (char*)malloc(n + 1u);
    if (!out) return NULL;
    memcpy(out, path, n);
    out[n] = '\0';
    return out;
}

static char* resolve_usd_asset_ref(const char* owner_path, const char* ref,
                                   size_t ref_len)
{
    if (!owner_path || !ref || ref_len == 0) return NULL;
    char* raw = (char*)malloc(ref_len + 1u);
    if (!raw) return NULL;
    memcpy(raw, ref, ref_len);
    raw[ref_len] = '\0';
    const char* r = raw;
    if (strncmp(r, "file:", 5) == 0) r += 5;
    if (strstr(r, "://")) {
        free(raw);
        return NULL;
    }

    char* joined = NULL;
    if (r[0] == '/') {
        joined = renderer_strdup(r);
    } else {
        char* dir = path_dirname_dup(owner_path);
        if (dir) {
            size_t dn = strlen(dir), rn = strlen(r);
            joined = (char*)malloc(dn + 1u + rn + 1u);
            if (joined)
                snprintf(joined, dn + 1u + rn + 1u, "%s/%s", dir, r);
            free(dir);
        }
    }
    free(raw);
    if (!joined) return NULL;

    FILE* f = fopen(joined, "rb");
    if (!f) {
        free(joined);
        return NULL;
    }
    fclose(f);

    char realbuf[PATH_MAX];
    if (realpath(joined, realbuf)) {
        free(joined);
        return renderer_strdup(realbuf);
    }
    return joined;
}

static int path_in_list(char** paths, int count, const char* path)
{
    for (int i = 0; i < count; i++)
        if (paths[i] && strcmp(paths[i], path) == 0)
            return 1;
    return 0;
}

static void enqueue_usd_asset_refs(const char* owner_path, const char* text,
                                   char** paths, int* count, int max_count)
{
    if (!owner_path || !text || !paths || !count) return;
    const char* p = text;
    while (*count < max_count && (p = strchr(p, '@')) != NULL) {
        const char* start = p + 1;
        const char* end = strchr(start, '@');
        if (!end) break;
        size_t n = (size_t)(end - start);
        if (usd_asset_ref_ext_ok(start, n)) {
            char* resolved = resolve_usd_asset_ref(owner_path, start, n);
            if (resolved) {
                if (!path_in_list(paths, *count, resolved)) {
                    paths[*count] = resolved;
                    (*count)++;
                } else {
                    free(resolved);
                }
            }
        }
        p = end + 1;
    }
}

static void exposure_desc_from_usd_path(const char* path, NuExposureDesc* desc)
{
    memset(desc, 0, sizeof(*desc));
    enum { MAX_EXPOSURE_SCAN_FILES = 96 };
    char* paths[MAX_EXPOSURE_SCAN_FILES] = {0};
    int path_count = 0;
    if (path && *path) {
        char realbuf[PATH_MAX];
        if (realpath(path, realbuf))
            paths[path_count++] = renderer_strdup(realbuf);
        else
            paths[path_count++] = renderer_strdup(path);
    }
    for (int i = 0; i < path_count; i++) {
        if (!paths[i]) continue;
        char* text = read_usd_scan_text(paths[i]);
        if (!text) continue;
        exposure_desc_merge_text(text, desc);
        if (!exposure_desc_complete(desc))
            enqueue_usd_asset_refs(paths[i], text, paths, &path_count,
                                   MAX_EXPOSURE_SCAN_FILES);
        free(text);
        if (exposure_desc_complete(desc)) break;
    }
    for (int i = 0; i < path_count; i++) free(paths[i]);

    if (parse_float_env2("NUSD_EXPOSURE_FSTOP", "NUSD_OVRTX_EXPOSURE_FSTOP", &desc->exposure_f_stop))
        desc->flags |= NU_EXPOSURE_HAS_FSTOP;
    if (parse_float_env2("NUSD_EXPOSURE_RESPONSIVITY", "NUSD_OVRTX_EXPOSURE_RESPONSIVITY", &desc->exposure_responsivity))
        desc->flags |= NU_EXPOSURE_HAS_RESPONSIVITY;
    if (parse_float_env2("NUSD_EXPOSURE_TIME", "NUSD_OVRTX_EXPOSURE_TIME", &desc->exposure_time))
        desc->flags |= NU_EXPOSURE_HAS_TIME;
    if (parse_float_env2("NUSD_WHITE_POINT_SCALE", "NUSD_OVRTX_WHITE_POINT_SCALE", &desc->white_point_scale))
        desc->flags |= NU_EXPOSURE_HAS_WHITE_POINT;
    if (parse_float_env2("NUSD_TONEMAP_FNUMBER", "NUSD_OVRTX_TONEMAP_FNUMBER", &desc->tonemap_f_number))
        desc->flags |= NU_EXPOSURE_HAS_TONEMAP_FNUM;
    if (parse_float_env2("NUSD_TONEMAP_CM2_FACTOR", "NUSD_OVRTX_TONEMAP_CM2_FACTOR", &desc->tonemap_cm2_factor))
        desc->flags |= NU_EXPOSURE_HAS_TONEMAP_CM2;
    if (parse_bool_env2("NUSD_AUTO_EXPOSURE_ENABLED", "NUSD_OVRTX_AUTO_EXPOSURE_ENABLED", &desc->auto_exposure_enabled))
        desc->flags |= NU_EXPOSURE_HAS_AUTO_EXPOSURE;
}

static float exposure_clamp_scale(float v)
{
    if (!(v > 0.0f) || !isfinite(v)) return 1.0f;
    if (v < 1.0e-4f) return 1.0e-4f;
    if (v > 1.0e4f) return 1.0e4f;
    return v;
}

static char* renderer_strdup(const char* s)
{
    if (!s) return NULL;
    size_t n = strlen(s) + 1u;
    char* out = (char*)malloc(n);
    if (out) memcpy(out, s, n);
    return out;
}

/* ---- Internal mesh record ---- */

typedef struct {
    int          active;        /* 0 = slot is free */
    int          visible;
    uint8_t      env_mask;      /* TLAS instance.mask byte; 0xFF = visible to all rays.
                                   Per-tile env isolation (Phase A) packs env-bucket bits here. */
    float        transform[16]; /* row-major 4x4 */
    float        color[3];
    int          nvertices;
    int          nindices;
    uint32_t     vertex_offset; /* offset in merged buffer */
    uint32_t     index_offset;
    int          prototype_idx; /* for instancing */
    int          material_index; /* -1 = none */
    char*        name;
    uint64_t     geo_hash;      /* FNV-1a-64 triangle BLAS hash
                                   (positions + indices). 0 when unused. */
    uint32_t*    ptex_tri_colors;       /* packed RGBA8, one color per triangle corner */
    uint32_t     ptex_tri_color_count;
    int          ptex_tri_colors_owned;
    uint32_t     ptex_color_offset;     /* offset in RT triangle color SSBO */
    int          bounds_valid;
    float        local_bounds_min[3];
    float        local_bounds_max[3];
    float        bounds_min[3];
    float        bounds_max[3];
} RendererMesh;

#define RENDERER_PTEX_COLOR_NONE 0xFFFFFFFFu

static void renderer_mesh_clear_ptex(RendererMesh* m)
{
    if (!m) return;
    if (m->ptex_tri_colors_owned)
        free(m->ptex_tri_colors);
    m->ptex_tri_colors = NULL;
    m->ptex_tri_color_count = 0;
    m->ptex_tri_colors_owned = 0;
    m->ptex_color_offset = RENDERER_PTEX_COLOR_NONE;
}

static void renderer_mesh_copy_ptex(RendererMesh* m, const SceneMesh* sm)
{
    if (!m || !sm || !sm->ptex_tri_colors || sm->ptex_tri_color_count <= 0)
        return;
    if (sm->ptex_tri_color_count != m->nindices)
        return;

    uint32_t count = (uint32_t)sm->ptex_tri_color_count;
    uint32_t* colors = (uint32_t*)malloc((size_t)count * sizeof(uint32_t));
    if (!colors) return;
    memcpy(colors, sm->ptex_tri_colors, (size_t)count * sizeof(uint32_t));

    renderer_mesh_clear_ptex(m);
    m->ptex_tri_colors = colors;
    m->ptex_tri_color_count = count;
    m->ptex_tri_colors_owned = 1;
    m->ptex_color_offset = RENDERER_PTEX_COLOR_NONE;
}

static int renderer_color_is_debug_placeholder(float r, float g, float b)
{
    int hi = (r > 0.95f) + (g > 0.95f) + (b > 0.95f);
    int lo = (r < 0.05f) + (g < 0.05f) + (b < 0.05f);
    return hi >= 1 && hi <= 2 && hi + lo == 3;
}

static int env_enabled_default_off(const char* name)
{
    const char* s = getenv(name);
    return (s && s[0] && s[0] != '0');
}

static float env_float_or(const char* name, float def)
{
    const char* s = getenv(name);
    if (!s || !s[0]) return def;
    char* end = NULL;
    float v = strtof(s, &end);
    return (end != s && isfinite(v)) ? v : def;
}

static int env_int_or(const char* name, int def)
{
    const char* s = getenv(name);
    if (!s || !s[0]) return def;
    char* end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s) return def;
    if (v < INT_MIN) return INT_MIN;
    if (v > INT_MAX) return INT_MAX;
    return (int)v;
}

static int rt_camera_residency_enabled(void)
{
    return env_enabled_default_off("NUSD_RT_CAMERA_RESIDENCY");
}

static int rt_camera_residency_allow_drop(void)
{
    return env_enabled_default_off("NUSD_RT_CAMERA_RESIDENCY_ALLOW_DROP");
}

static void bounds_reset(float mn[3], float mx[3])
{
    mn[0] = mn[1] = mn[2] =  FLT_MAX;
    mx[0] = mx[1] = mx[2] = -FLT_MAX;
}

static void transform_point4x4(const float m[16], const float p[3], float out[3])
{
    out[0] = m[0] * p[0] + m[1] * p[1] + m[2]  * p[2] + m[3];
    out[1] = m[4] * p[0] + m[5] * p[1] + m[6]  * p[2] + m[7];
    out[2] = m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11];
}

static void mesh_compute_world_bounds(RendererMesh* m)
{
    if (!m || !m->bounds_valid) return;
    bounds_reset(m->bounds_min, m->bounds_max);
    const float* mn = m->local_bounds_min;
    const float* mx = m->local_bounds_max;
    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                float p[3] = {
                    ix ? mx[0] : mn[0],
                    iy ? mx[1] : mn[1],
                    iz ? mx[2] : mn[2],
                };
                float wp[3];
                transform_point4x4(m->transform, p, wp);
                for (int k = 0; k < 3; k++) {
                    if (wp[k] < m->bounds_min[k]) m->bounds_min[k] = wp[k];
                    if (wp[k] > m->bounds_max[k]) m->bounds_max[k] = wp[k];
                }
            }
        }
    }
}

static void mesh_compute_local_bounds(RendererMesh* m, const float* positions, int nvertices)
{
    if (!m || !positions || nvertices <= 0) {
        if (m) m->bounds_valid = 0;
        return;
    }
    bounds_reset(m->local_bounds_min, m->local_bounds_max);
    for (int i = 0; i < nvertices; i++) {
        const float* p = positions + i * 3;
        for (int k = 0; k < 3; k++) {
            if (p[k] < m->local_bounds_min[k]) m->local_bounds_min[k] = p[k];
            if (p[k] > m->local_bounds_max[k]) m->local_bounds_max[k] = p[k];
        }
    }
    m->bounds_valid = 1;
    mesh_compute_world_bounds(m);
}

/* ---- Phase 11.A: Curve registry ----
 * Flat segment list across all BasisCurves prims in the loaded scene.
 * A single AABB BLAS is built over all of these (Strategy A from
 * RENDERER_BIG_PLAN.md Phase 12.1 — one flat BLAS over all segments,
 * best traversal quality at scale). Per-segment color carries the
 * curve's display_color for now; Phase 12.2 adds material_id indirection.
 *
 * Phase 12.x: per-segment AABBs are GPU-generated at BLAS-build time
 * (gpu_build_curve_blas runs a compute pass that reads the segment
 * SSBO and writes the AABB device buffer). No host AABB array exists.
 * On a 34M-segment "tera" fixture this saves ~825 MB of host→device
 * upload bandwidth (~80 ms on PCIe gen 5). */
typedef struct {
    SceneCurveSegment* segments;     /* float[nseg * 8]  — 32 B/segment   */
    float*             colors;       /* float[nseg * 3]  — display_color  */
    int                nseg;
} RendererCurves;

typedef struct {
    uint32_t curve_id;
    uint64_t index_byte_offset;
    uint32_t npatches;
    int32_t  vertex_offset;
    int      index_type_bits;
    float    model[16];
    float    color[3];
    float    bounds_min[3];
    float    bounds_max[3];
    int      basis_id;
    int      material_index;
} RendererRasterCurve;

typedef struct {
    RendererRasterCurve* draws;
    int                  ndraws;
    GpuBuffer            vertex_buffer;
    GpuBuffer            index_buffer;
    uint32_t             total_vertices;
    uint32_t             total_indices;
    uint64_t             total_index_bytes;
} RendererRasterCurves;

typedef struct {
    float    position[3];
    float    width;
    uint32_t color_rgba8;
} RendererCurveVertex;

/* ---- Internal state ---- */

struct NuRenderer {
    /* Configuration */
    int          width;
    int          height;
    int          enable_rt;
    int          enable_materials;

    /* Source USD path of the currently loaded scene — set by nu_load_usd,
     * empty for programmatic / handle-loaded scenes. Used to derive the
     * on-disk BLAS cache sidecar path. */
    char         usd_path[1024];

    /* Current evaluation time (USD time code) — propagated to scene
     * loading so xformOp:translate.timeSamples / etc. resolve at the
     * right frame. NaN means "use default time". Set via
     * nu_set_current_time(); read by scene.c::compute_world_xform. */
    double       current_time;

    /* GLFW (hidden window for Vulkan surface) */
    GLFWwindow*  window;
    int          glfw_acquired;

    /* GPU state */
    Gpu*         gpu;

    /* Mesh registry */
    RendererMesh* meshes;
    int          nmeshes;
    /* Phase 2 (NUSD_RENDER_PI_BATCHES): renderer-owned copy of the scene's
     * compact nested-PI batches; prototype_mesh_idx is renderer-space here.
     * The RT build expands these into instanced TLAS entries. */
    SceneInstanceBatch*     pi_batches;
    int                     npi_batches;
    SceneInstanceTransform* pi_transforms;
    uint64_t                npi_transforms;
    int          mesh_capacity;
    char**       material_paths;
    int          material_path_count;
    int          suppress_next_meshopt;
    int          suppress_next_geo_dedup;

    /* Phase C: per-env TLAS partition (set via nu_set_env_partition).
     * mesh_to_env[i] = env_idx for mesh slot i, or -1 for static globals.
     * num_envs = 0 disables partitioning (single-TLAS legacy path). */
    int*         mesh_to_env;
    int          num_envs;

    /* Curve registry (Phase 11.A) — populated by nu_load_usd, uploaded
     * to GPU at nu_build_accel time. */
    RendererCurves curves;
    RendererRasterCurves raster_curves;
    int            curves_dirty;

    /* GPU buffers */
    GpuBuffer    vertex_buffer;
    GpuBuffer    index_buffer;
    uint32_t     total_vertices;
    uint32_t     total_indices;

    /* Merged buffer data (CPU side, rebuilt on scene change).
     * Material path: 12 floats/vertex: pos(3) normal(3) pad(3) uv(2) matID.
     * Geometry path: 9 floats/vertex: pos(3) normal(3) uv(2) matID.
     * The geometry layout keeps raster shader inputs defined while avoiding
     * the unused 3-float material padding on DSX geometry-only loads. */
    float*       cpu_vertices;
    uint32_t*    cpu_indices;
    uint32_t     vertex_stride_floats;
    uint32_t     cpu_vertex_capacity;
    uint32_t     cpu_index_capacity;

    /* Runtime texture array — populated by nu_load_texture, committed
     * via gpu_upload_materials before the next rt scene build. Each entry
     * holds the RGBA8 bytes + size for a single texture. The renderer
     * owns the pixel buffer (caller may free its source after the call). */
    struct {
        unsigned char* pixels;
        int width;
        int height;
    }*           runtime_textures;
    int          runtime_textures_count;
    int          runtime_textures_capacity;
    int          runtime_textures_committed;  /* 1 = uploaded via gpu_upload_materials */

    /* Pipelines */
    GpuPipeline  raster_pipeline;
    GpuPipeline  raster_pipeline_textured;  /* sampler-bound variant; created after gpu_upload_materials */
    int          raster_textured_pipeline_attempted;
    GpuPipeline  raster_pipeline_instanced; /* instance-rate variant for compact PI batches */
    int          raster_instanced_pipeline_attempted;
    GpuBuffer    pi_instance_buffer;        /* per-instance world matrices (16 floats each) */
    uint64_t     pi_instance_count;         /* number of matrices in pi_instance_buffer */
    GpuPipeline  curve_pipeline;
    GpuPipeline  shadow_pipeline;

    /* Camera */
    Camera       camera;
    int          camera_valid;

    /* OVRTX-style exposure/tonemap state. `tone_*` values are relative
     * multipliers consumed by the shaders; `exposure_desc` keeps the
     * authored/API values for callers that set all fields explicitly. */
    NuExposureDesc exposure_desc;
    float        tone_exposure_scale;
    float        tone_sky_scale;
    float        tone_white_point;
    uint32_t     tone_flags;
    float        rt_ibl_fill_scale;

    /* Scene dirty flag */
    int          scene_dirty;    /* 1 = need to rebuild GPU buffers + accel */
    int          accel_dirty;    /* 1 = need to rebuild BLAS/TLAS */
    int          tlas_dirty;     /* 1 = transforms changed, rebuild TLAS only */
    int          colors_dirty;   /* 1 = display colors changed, update scene SSBO */

    /* 1 = nu_finalize_scene has dropped the CPU-side vertex/index mirror.
     * After this flag is set, the renderer's r->cpu_vertices/r->cpu_indices
     * are NULL and any mutation API (nu_add_mesh, nu_set_mesh_color, ...)
     * that would re-trigger rebuild_gpu_buffers must FAIL fast: the rebuild
     * path reads from the (now-freed) mirror. Callers that need a mutable
     * scene must load fresh via nu_clear_scene + nu_load_usd. */
    int          scene_finalized;

    /* Scene loaded from USD (optional). Successful USD attaches copy the
     * bounds below and release the heavy scene arena immediately; this
     * pointer is retained only while attach is in progress or after a
     * partial/error path so cleanup still has ownership. */
    Scene*       usd_scene;
    float        scene_bounds_min[3];
    float        scene_bounds_max[3];
    int          scene_bounds_valid;
    int          scene_up_axis;       /* 0=X, 1=Y, 2=Z; default camera up */
    int          fallback_lights_active; /* OVRTX-style defaults for unlit scenes */
    int          authored_light_count;

    /* Tier 3 lazy load (see docs/plans/TIER_3_LAZY_MESH.md). Set by
     * nu_attach_scene when scene_load returned a metadata-only scene
     * (NUSD_LAZY_MESH=1). nu_extract_deferred consumes these to do a
     * second scene_load_from_stage with the lazy env-var cleared, then
     * re-attaches the eager scene through the normal path. */
    int          lazy_pending;
    void*        lazy_stage_pending;     /* NanousdStage, kept alive across attach */
    int          lazy_owns_stage;        /* mirrors scene->_owns_stage at lazy attach */

    /* Tier 3 step 4: per-mesh world AABB snapshot taken at lazy-attach
     * time. nu_extract_deferred_visible walks this against caller-supplied
     * camera frusta, builds a prim-index bitmap, and passes it to
     * scene_load_from_stage_filtered. Allocated on the heap so it survives
     * the scene_free at attach end. Freed when the lazy state clears
     * (nu_extract_deferred*, nu_clear_scene, nu_renderer_destroy). */
    struct {
        int   lazy_prim_idx;
        float bounds_min[3];
        float bounds_max[3];
    } *lazy_mesh_aabbs;
    int          lazy_mesh_aabb_count;
    int          lazy_nprims_snapshot;   /* nanousd_nprims(stage) at lazy attach;
                                            sizes the wanted_prims bitmap. */

    /* Error string */
    char         last_error[512];

    /* Readback buffer */
    uint8_t*     readback_pixels; /* RGBA8, width * height * 4 */
    int          has_frame;       /* 1 = pixels valid after render */

    /* Tiled RT state */
    int          tiled_pipeline_built; /* 1 = tiled RT pipeline ready */
    int          fast_mode;            /* 1 = skip shadow rays for RL sensors */
    /* Flat DomeLight color (RGB) + intensity (A) for the fast_mode rmiss
     * sky and rchit hemispheric ambient. Defaults to white at intensity 1.0.
     * Set via nu_set_dome_color(); pushed into the SceneData SSBO header. */
    float        dome_color[4];
    int          dome_color_dirty;     /* 1 = SSBO header re-upload pending */
    int          per_env_layout;      /* 1 = SSBO writes in [env,H,W] not tiled */
    int          tiled_srgb;           /* 1 = apply sRGB gamma in raygen shader */
    int          depth_enabled;       /* 1 = write depth to SSBO during RT */
    int          segmentation_enabled; /* 1 = write instance IDs to SSBO during RT */
    int          normals_enabled;      /* 1 = write surface normals to SSBO during RT */

    /* Phase B deferred-shading: when 1, the rchit/rmiss write a per-pixel
     * G-buffer (binding 17) and a follow-on compute dispatch reads it to
     * produce the final pixels. Phase B is plumbing-only — output is a
     * "color-ID" debug viz (one flat color per mesh). Default 0 keeps the
     * legacy path byte-for-byte. */
    int          deferred_shade_enabled;
    int          deferred_pipeline_built; /* 1 = compute pipeline ready */
    /* Phase C.2 — runtime debug visualization selector for the deferred
     * compute shader. Decoded at offset 28 of the push constant block (the
     * unused _pad[3] slot, viewInverse[1][3] in shader land). 0 = base
     * color (C.1 path, byte-identical), 1 = world-space shading normal RGB,
     * 2 = pbr packed (metallic.r, roughness.g, ao.b), 3 = full-lit
     * (placeholder; falls back to mode 0 until C.4). */
    uint32_t     deferred_debug_mode;

    /* Diagnostic-only: when > 0, the deferred compute shader writes the LINEAR
     * pre-tonemap radiance (scaled by this gain, clamped to [0,1], no ACES /
     * exposure / sRGB) instead of the tonemapped color. Set via
     * NU_DEFERRED_LINEAR_PROBE. Used to read true radiance for light-unit
     * calibration against ovrtx's linear HdrColor AOV; 0 = normal tonemap. */
    float        deferred_linear_probe_gain;

    /* Raycast compute pipeline state */
    int          raycast_pipeline_built; /* 1 = raycast compute pipeline ready */

    /* Async render+fetch state (nu_render_async/nu_fetch_async).
     * Tracks a 2-deep history of tiled-staging slot indices so that fetch on
     * frame N can pick up the slot written by frame N-1's render. */
    int          async_curr_slot;   /* slot written by the most recent render_async */
    int          async_prev_slot;   /* slot written by the render before that      */
    int          async_count;       /* clamps at 2: # of render_async calls so far */

    /* PR 2: GPU-driven TLAS instance translation. When use_gpu_transforms
     * is set, nu_render_tiled records gpu_translate_instances_inline()
     * (compute dispatch + TLAS build) instead of gpu_update_tlas_inline()
     * (host loop + transfer + TLAS build). */
    int          use_gpu_transforms;        /* 1 = GPU path active for next render */
    int          gpu_transform_count;       /* dispatch size (n_valid) */

    /* VK3DGRT — Gaussian splat scene state (parallel to mesh path).
     * Owns particle CPU arrays + config knobs; GPU side lives in gpu_vulkan.c
     * (P2/P3, not yet implemented). gs_dirty marks "particle data changed,
     * rebuild splat BLAS/TLAS on next render". gs_pipeline_built tracks
     * whether the splat RT pipeline is current for the active config (K /
     * camera model / proxy). See docs/plans/VK3DGRT_PLAN.md. */
    NuGsScene*   gs;
    int          gs_dirty;
    int          gs_pipeline_built;
};

static int bounds_finite_valid(const float mn[3], const float mx[3])
{
    for (int k = 0; k < 3; k++) {
        if (!isfinite(mn[k]) || !isfinite(mx[k]) || mn[k] > mx[k])
            return 0;
    }
    return 1;
}

static void renderer_rt_scene_metrics(const NuRenderer* r,
                                      float* out_ground_y,
                                      float* out_scene_scale)
{
    float mn[3], mx[3];
    int have_bounds = 0;

    if (out_ground_y) *out_ground_y = -1.0f;
    if (out_scene_scale) *out_scene_scale = 10.0f;
    if (!r) return;

    if (r->scene_bounds_valid &&
        bounds_finite_valid(r->scene_bounds_min, r->scene_bounds_max)) {
        memcpy(mn, r->scene_bounds_min, sizeof(mn));
        memcpy(mx, r->scene_bounds_max, sizeof(mx));
        have_bounds = 1;
    } else {
        bounds_reset(mn, mx);
        for (int i = 0; i < r->nmeshes; i++) {
            const RendererMesh* m = &r->meshes[i];
            if (!m->active || !m->bounds_valid ||
                !bounds_finite_valid(m->bounds_min, m->bounds_max))
                continue;
            for (int k = 0; k < 3; k++) {
                if (m->bounds_min[k] < mn[k]) mn[k] = m->bounds_min[k];
                if (m->bounds_max[k] > mx[k]) mx[k] = m->bounds_max[k];
            }
            have_bounds = 1;
        }
    }

    if (!have_bounds) return;

    float dx = mx[0] - mn[0];
    float dy = mx[1] - mn[1];
    float dz = mx[2] - mn[2];
    float diag = sqrtf(dx * dx + dy * dy + dz * dz);
    if (!isfinite(diag) || diag < 1.0e-3f) diag = 1.0f;

    if (out_ground_y) *out_ground_y = mn[2];
    if (out_scene_scale) *out_scene_scale = diag;
}

static int mesh_bounds_intersects_camera(const NuRenderer* r, const RendererMesh* m)
{
    if (!r || !m || !m->bounds_valid) return 1;

    float view[16];
    camera_get_view(&r->camera, view);

    float center[3] = {
        0.5f * (m->bounds_min[0] + m->bounds_max[0]),
        0.5f * (m->bounds_min[1] + m->bounds_max[1]),
        0.5f * (m->bounds_min[2] + m->bounds_max[2]),
    };
    float dx = 0.5f * (m->bounds_max[0] - m->bounds_min[0]);
    float dy = 0.5f * (m->bounds_max[1] - m->bounds_min[1]);
    float dz = 0.5f * (m->bounds_max[2] - m->bounds_min[2]);
    float radius = sqrtf(dx * dx + dy * dy + dz * dz);

    float vc[3];
    transform_point4x4(view, center, vc);
    float depth = -vc[2];  /* camera looks down -Z */

    float margin = env_float_or("NUSD_RT_RESIDENCY_MARGIN", 0.35f);
    if (margin < 0.0f) margin = 0.0f;
    float tan_base_y = tanf(r->camera.fov_y * 0.5f);
    float tan_y = tan_base_y * (1.0f + margin + fabsf(r->camera.projection_shift_y));
    float tan_x = tan_base_y * r->camera.aspect * (1.0f + margin + fabsf(r->camera.projection_shift_x));

    float near_z = r->camera.near_z;
    float far_z = r->camera.far_z;
    if (depth + radius < near_z) return 0;
    if (depth - radius > far_z) return 0;

    float d_for_extent = fmaxf(depth + radius, near_z);
    if (fabsf(vc[0]) - radius > d_for_extent * tan_x) return 0;
    if (fabsf(vc[1]) - radius > d_for_extent * tan_y) return 0;

    float max_dist = env_float_or("NUSD_RT_RESIDENCY_MAX_DISTANCE", 0.0f);
    if (max_dist > 0.0f && depth - radius > max_dist)
        return 0;
    return 1;
}

static void write_rt_residency_manifest(const NuRenderer* r,
                                        uint32_t selected_visible,
                                        uint32_t excluded_visible)
{
    const char* path = getenv("NUSD_RT_RESIDENCY_MANIFEST");
    if (!path || !path[0]) return;

    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "nusd_renderer: failed to write residency manifest %s: %s\n",
                path, strerror(errno));
        return;
    }
    float eye[3];
    camera_get_eye(&r->camera, eye);
    fprintf(f,
            "{\n"
            "  \"mode\": \"camera_residency\",\n"
            "  \"width\": %d,\n"
            "  \"height\": %d,\n"
            "  \"fov_degrees\": %.9g,\n"
            "  \"near\": %.9g,\n"
            "  \"far\": %.9g,\n"
            "  \"eye\": [%.9g, %.9g, %.9g],\n"
            "  \"target\": [%.9g, %.9g, %.9g],\n"
            "  \"selected_visible\": %u,\n"
            "  \"excluded_visible\": %u,\n"
            "  \"allow_drop\": %s,\n"
            "  \"excluded\": [\n",
            r->width, r->height,
            r->camera.fov_y * 180.0f / 3.14159265f,
            r->camera.near_z, r->camera.far_z,
            eye[0], eye[1], eye[2],
            r->camera.target[0], r->camera.target[1], r->camera.target[2],
            selected_visible, excluded_visible,
            rt_camera_residency_allow_drop() ? "true" : "false");

    int first = 1;
    for (int i = 0; i < r->nmeshes; i++) {
        const RendererMesh* m = &r->meshes[i];
        if (!m->active || !m->visible || m->nindices == 0) continue;
        if (mesh_bounds_intersects_camera(r, m)) continue;
        if (!first) fprintf(f, ",\n");
        first = 0;
        fprintf(f,
                "    {\"mesh_id\": %d, \"prototype_idx\": %d, "
                "\"material_index\": %d, \"vertices\": %d, \"indices\": %d, "
                "\"bounds_min\": [%.9g, %.9g, %.9g], "
                "\"bounds_max\": [%.9g, %.9g, %.9g], "
                "\"name\": \"",
                i, m->prototype_idx, m->material_index,
                m->nvertices, m->nindices,
                m->bounds_min[0], m->bounds_min[1], m->bounds_min[2],
                m->bounds_max[0], m->bounds_max[1], m->bounds_max[2]);
        const char* s = m->name ? m->name : "";
        for (; *s; s++) {
            if (*s == '"' || *s == '\\') fputc('\\', f);
            if ((unsigned char)*s >= 0x20) fputc(*s, f);
        }
        fprintf(f, "\"}");
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
}

static void renderer_free_material_paths(NuRenderer* r)
{
    if (!r) return;
    for (int i = 0; i < r->material_path_count; i++)
        free(r->material_paths[i]);
    free(r->material_paths);
    r->material_paths = NULL;
    r->material_path_count = 0;
}

static void renderer_apply_exposure_desc(NuRenderer* r, const NuExposureDesc* desc)
{
    if (!r) return;

    if (desc) {
        r->exposure_desc = *desc;
    } else {
        memset(&r->exposure_desc, 0, sizeof(r->exposure_desc));
    }

    const NuExposureDesc* d = &r->exposure_desc;
    float scale = 1.0f;
    const uint32_t photo_flags = NU_EXPOSURE_HAS_FSTOP |
                                 NU_EXPOSURE_HAS_RESPONSIVITY |
                                 NU_EXPOSURE_HAS_TIME |
                                 NU_EXPOSURE_HAS_TONEMAP_FNUM |
                                 NU_EXPOSURE_HAS_TONEMAP_CM2;

    if ((d->flags & photo_flags) != 0u) {
        /* The existing shader tonemap was calibrated against the old Vulkan
         * DSX look, which was darker than the fixed OVRTX wrapper camera.
         * Keep DSX's authored photographic tuple as the relative reference,
         * then apply the measured OVRTX parity gain by default. Un-authored
         * scenes still keep scale 1.0.
         */
        const float ref_responsivity = 1.1026709f;
        const float ref_time = 0.02f;       /* shutter = 1/50 */
        const float ref_f_number = 21.0f;
        const float ref_cm2_factor = 0.5f;
        const float ovrtx_parity_gain = 3.0f;
        const float ref_photo = ref_responsivity * ref_time /
                                (ref_f_number * ref_f_number) *
                                ref_cm2_factor;

        float f_number = ref_f_number;
        if ((d->flags & NU_EXPOSURE_HAS_FSTOP) && d->exposure_f_stop > 0.0f) {
            f_number = d->exposure_f_stop;
        } else if ((d->flags & NU_EXPOSURE_HAS_TONEMAP_FNUM) &&
                   d->tonemap_f_number > 0.0f) {
            f_number = d->tonemap_f_number;
        }

        float responsivity = ((d->flags & NU_EXPOSURE_HAS_RESPONSIVITY) &&
                              d->exposure_responsivity > 0.0f)
                           ? d->exposure_responsivity : ref_responsivity;
        float exposure_time = ((d->flags & NU_EXPOSURE_HAS_TIME) &&
                               d->exposure_time > 0.0f)
                            ? d->exposure_time : ref_time;
        float cm2_factor = ((d->flags & NU_EXPOSURE_HAS_TONEMAP_CM2) &&
                            d->tonemap_cm2_factor > 0.0f)
                         ? d->tonemap_cm2_factor : ref_cm2_factor;
        float photo = responsivity * exposure_time /
                      (f_number * f_number) * cm2_factor;
        scale = (photo / ref_photo) * ovrtx_parity_gain;

        if ((d->flags & NU_EXPOSURE_HAS_AUTO_EXPOSURE) &&
            d->auto_exposure_enabled &&
            (d->flags & NU_EXPOSURE_HAS_WHITE_POINT) &&
            d->white_point_scale > 0.0f) {
            scale /= d->white_point_scale;
        }
    }

    float direct_scale = 1.0f;
    if (parse_float_env2("NUSD_EXPOSURE_SCALE", "NUSD_OVRTX_EXPOSURE_SCALE",
                         &direct_scale)) {
        scale *= direct_scale;
    }

    r->tone_exposure_scale = exposure_clamp_scale(scale);
    r->tone_sky_scale = r->tone_exposure_scale;
    r->tone_white_point = ((d->flags & NU_EXPOSURE_HAS_WHITE_POINT) &&
                           d->white_point_scale > 0.0f)
                        ? d->white_point_scale : 1.0f;
    r->tone_flags = d->flags;

    if (r->gpu) {
        gpu_set_tone_mapping(r->gpu, r->tone_exposure_scale,
                             r->tone_sky_scale, r->tone_white_point,
                             r->tone_flags);
    }
}

/* Forward declaration: defined further down for the async render path,
 * also used by nu_gs_render. */
static void compute_camera_inverses(const Camera* cam, float vp_inv[32]);
static void read_smaps_rollup_kb(long* rss, long* anon, long* file_kb);

/* NUSD_RSS_TRACE=1 prints a one-line RSS attribution at each tagged
 * point in nu_attach_scene. Used to pin which allocation drives the
 * eager-pass peak — the inline-mirror-drop result (-0.95 GB) made
 * clear the peak isn't where the streaming-upload model predicted.
 * Cheap (one /proc/self/smaps_rollup read); default off. */
static void rss_trace(const char* tag)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char* e = getenv("NUSD_RSS_TRACE");
        enabled = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    if (!enabled) return;
    long rss = 0, anon = 0, fkb = 0;
    read_smaps_rollup_kb(&rss, &anon, &fkb);
    fprintf(stderr, "[rss_trace] %-32s RSS=%6ld MB Anon=%6ld MB File=%5ld MB\n",
            tag ? tag : "<?>", rss / 1024, anon / 1024, fkb / 1024);
}

/* ---- Helpers ---- */

static void set_error(NuRenderer* r, const char* msg)
{
    snprintf(r->last_error, sizeof(r->last_error), "%s", msg);
}

/* Forward declaration — body lives just after nu_finalize_scene because
 * it depends on the scene_finalized flag introduced for that API. */
static int check_scene_mutable(NuRenderer* r, const char* api);

static void identity_matrix(float m[16])
{
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

/* ---- Geometry hash census (NUSD_HASH_CENSUS=1) ----
 *
 * Measure the content-dedup ceiling without changing behavior. On every
 * nu_add_mesh call, FNV-1a hash the (nvertices, nindices, positions,
 * indices, normals) tuple and record unique vs duplicate hits. Dumped
 * via nu_finalize_scene's log. Cheap (~hundreds of MB/s on FNV-1a),
 * one-shot, never affects rendering.
 *
 * If unique << total on a representative scene, content-hash dedup is
 * worth implementing as a fast path in nu_add_mesh. If unique ≈ total,
 * skip dedup and prioritize streaming-upload / vertex compression. */

typedef struct {
    uint64_t hash;
    int      hits;    /* >=2 means a dup was found */
    int      slot;    /* renderer mesh slot of the prototype (first occurrence) */
} GeoHashEntry;

typedef struct {
    GeoHashEntry* entries;     /* open addressing, power-of-two capacity */
    uint32_t      capacity;    /* slot count; 0 = disabled */
    uint32_t      count;       /* unique hashes seen */
    uint64_t      total_calls; /* nu_add_mesh fast path passes */
    uint64_t      dup_calls;   /* hash already in table */
    uint64_t      hash_collisions; /* same 64-bit hash, different stream/material */
    uint64_t      total_bytes; /* sum of (positions+indices+normals) bytes hashed */
} GeoHashCensus;

static GeoHashCensus g_geo_census = { NULL, 0, 0, 0, 0, 0, 0 };

static int geo_census_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("NUSD_HASH_CENSUS");
        cached = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return cached;
}

/* NUSD_HASH_DEDUP=1 turns the census table into an active dedup cache:
 * nu_add_mesh checks the hash table and, on a confirmed hit, redirects
 * to the geometry-sharing fast path (skips the multi-MB vertex/index
 * memcpy into r->cpu_vertices). Implies census semantics. */
static int geo_dedup_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        /* Default ON: material-independent content-hash dedup is bit-identical
         * to the per-mesh path (RT reads per-slot MeshData.material_id; raster
         * reads per-draw push-constant material id) and collapses byte-identical
         * geometry onto one shared BLAS, which is load-enabling on large scenes
         * (DSX per-mesh BLAS would be tens of GB). NUSD_HASH_DEDUP=0 forces the
         * per-mesh path for A/B measurement. */
        const char* e = getenv("NUSD_HASH_DEDUP");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached;
}

static void geo_census_init(uint32_t initial_slots)
{
    if (g_geo_census.entries) return;
    uint32_t cap = 1024;
    while (cap < initial_slots * 2u) cap <<= 1;
    g_geo_census.entries  = (GeoHashEntry*)calloc(cap, sizeof(GeoHashEntry));
    g_geo_census.capacity = g_geo_census.entries ? cap : 0;
}

static void geo_census_grow(void)
{
    if (g_geo_census.count * 2u < g_geo_census.capacity) return;
    uint32_t new_cap = g_geo_census.capacity ? g_geo_census.capacity * 2u : 1024u;
    GeoHashEntry* ne = (GeoHashEntry*)calloc(new_cap, sizeof(GeoHashEntry));
    if (!ne) return;
    for (uint32_t i = 0; i < g_geo_census.capacity; i++) {
        if (!g_geo_census.entries[i].hash) continue;
        uint32_t mask = new_cap - 1u;
        uint32_t s = (uint32_t)(g_geo_census.entries[i].hash) & mask;
        while (ne[s].hash) s = (s + 1) & mask;
        ne[s] = g_geo_census.entries[i];
    }
    free(g_geo_census.entries);
    g_geo_census.entries = ne;
    g_geo_census.capacity = new_cap;
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

/* Compute the geometry hash for a NuMeshDesc. Returns 0 if the desc is
 * empty (caller should skip the dedup path). Hash domain:
 * (nvertices, nindices, positions, indices, normals?, texcoords?). All
 * data baked into the renderer's vertex stream (renderer.c:961-983) is
 * in the key — texcoords matter because they're at v[uv_offset]; sharing
 * meshes that differ only in texcoords would corrupt textured materials.
 * material_index, display_color, and transform are NOT in the hash:
 * they're stored per-instance on the RendererMesh, not in vertex data.
 * material_index IS checked separately in the caller as a fast guard
 * (it's the most common reason for not-quite-identical meshes). */
static uint64_t geo_hash_compute(const NuMeshDesc* desc, size_t* out_bytes)
{
    if (!desc || desc->nvertices <= 0 || desc->nindices <= 0) {
        if (out_bytes) *out_bytes = 0;
        return 0;
    }
    uint64_t h = 1469598103934665603ULL;   /* FNV-1a offset basis */
    int nv = desc->nvertices;
    int ni = desc->nindices;
    h = fnv1a_64_update(h, &nv, sizeof(nv));
    h = fnv1a_64_update(h, &ni, sizeof(ni));
    /* Domain marker bits: differentiate "no normals" from "all-zero normals"
     * etc., so meshes that ONLY differ in which optional streams are
     * present don't collide. */
    uint8_t present = (desc->normals ? 1 : 0) | (desc->texcoords ? 2 : 0);
    h = fnv1a_64_update(h, &present, sizeof(present));
    size_t pos_bytes = (size_t)nv * 3u * sizeof(float);
    size_t idx_bytes = (size_t)ni * sizeof(int);
    size_t nrm_bytes = desc->normals   ? (size_t)nv * 3u * sizeof(float) : 0u;
    size_t uv_bytes  = desc->texcoords ? (size_t)nv * 2u * sizeof(float) : 0u;
    if (desc->positions) h = fnv1a_64_update(h, desc->positions, pos_bytes);
    if (desc->indices)   h = fnv1a_64_update(h, desc->indices,   idx_bytes);
    if (desc->normals)   h = fnv1a_64_update(h, desc->normals,   nrm_bytes);
    if (desc->texcoords) h = fnv1a_64_update(h, desc->texcoords, uv_bytes);
    if (h == 0) h = 0xdeadbeefcafebabeULL;  /* reserve 0 = empty slot */
    if (out_bytes) *out_bytes = pos_bytes + idx_bytes + nrm_bytes + uv_bytes;
    return h;
}

/* BLAS cache key. A Vulkan triangle BLAS only consumes vertex positions and
 * indices, not normals, UVs, display color, or material data. Keep this hash
 * narrower than geo_hash_compute so harmless shading-stream differences do
 * not invalidate a serialized acceleration-structure cache. */
static uint64_t blas_hash_compute(const NuMeshDesc* desc)
{
    if (!desc || !desc->positions || !desc->indices ||
        desc->nvertices <= 0 || desc->nindices <= 0)
        return 0;
    uint64_t h = 1469598103934665603ULL;
    int nv = desc->nvertices;
    int ni = desc->nindices;
    h = fnv1a_64_update(h, &nv, sizeof(nv));
    h = fnv1a_64_update(h, &ni, sizeof(ni));
    h = fnv1a_64_update(h, desc->positions, (size_t)nv * 3u * sizeof(float));
    h = fnv1a_64_update(h, desc->indices, (size_t)ni * sizeof(int));
    if (h == 0) h = 0xfeedfacecafebeefULL;
    return h;
}

static int geo_desc_matches_renderer_mesh(const NuRenderer* r,
                                          int slot,
                                          const NuMeshDesc* desc)
{
    if (!r || !desc || slot < 0 || slot >= r->nmeshes) return 0;
    const RendererMesh* m = &r->meshes[slot];
    if (!m->active) return 0;
    if (m->nvertices != desc->nvertices || m->nindices != desc->nindices)
        return 0;
    /* Geometry-only predicate: material is intentionally NOT compared. Dedup
     * collapses byte-identical geometry onto one shared vertex range + BLAS
     * regardless of material, which is then applied per-instance downstream
     * (RT MeshData.material_id, raster push-constant material id). The
     * per-vertex matID slot in the prototype's vertex data is dead for both
     * shaders, so a cross-material share is bit-identical. */
    if (!r->cpu_vertices || !r->cpu_indices)
        return 0;

    uint32_t stride = r->vertex_stride_floats ? r->vertex_stride_floats : 12u;
    uint32_t uv_offset = (stride >= 12u) ? 9u : 6u;
    const float* vbase = r->cpu_vertices + (size_t)m->vertex_offset * stride;
    for (int i = 0; i < desc->nvertices; i++) {
        const float* v = vbase + (size_t)i * stride;
        const float* p = desc->positions + (size_t)i * 3u;
        if (v[0] != p[0] || v[1] != p[1] || v[2] != p[2])
            return 0;

        if (desc->normals) {
            const float* n = desc->normals + (size_t)i * 3u;
            if (v[3] != n[0] || v[4] != n[1] || v[5] != n[2])
                return 0;
        } else if (v[3] != 0.0f || v[4] != 1.0f || v[5] != 0.0f) {
            return 0;
        }

        if (desc->texcoords) {
            const float* uv = desc->texcoords + (size_t)i * 2u;
            if (v[uv_offset + 0] != uv[0] ||
                v[uv_offset + 1] != 1.0f - uv[1])
                return 0;
        } else if (v[uv_offset + 0] != 0.0f ||
                   v[uv_offset + 1] != 0.0f) {
            return 0;
        }
    }

    const uint32_t* ibase = r->cpu_indices + m->index_offset;
    for (int i = 0; i < desc->nindices; i++) {
        if (ibase[i] != (uint32_t)desc->indices[i])
            return 0;
    }
    return 1;
}

/* True when the on-disk BLAS cache is active and each mesh's geo_hash
 * should be computed + retained (it keys the .nzblas records). Gated so a
 * plain load pays no extra hashing cost. NUSD_BLAS_CACHE_WRITE arms the
 * Phase-A serialize/write path; NUSD_BLAS_CACHE is reserved for the
 * Phase-B deserialize path. */
static int geo_hash_wanted(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char* w = getenv("NUSD_BLAS_CACHE_WRITE");
        const char* rd = getenv("NUSD_BLAS_CACHE");
        cached = ((w  && w[0]  && w[0]  != '0') ||
                  (rd && rd[0] && rd[0] != '0')) ? 1 : 0;
    }
    return cached;
}

/* If `desc` exactly matches an entry already in the census table, return
 * the prototype slot recorded there. Same-hash entries must still be
 * verified against the renderer's retained CPU stream; a 64-bit hash alone
 * is only a bucket key, not correctness proof.
 * Caller guarantees geo_dedup_enabled(). */
static int geo_dedup_lookup(const NuRenderer* r, const NuMeshDesc* desc)
{
    if (!g_geo_census.entries) return -1;
    size_t bytes = 0;
    uint64_t h = geo_hash_compute(desc, &bytes);
    if (!h) return -1;
    uint32_t mask = g_geo_census.capacity - 1u;
    uint32_t s = (uint32_t)h & mask;
    while (g_geo_census.entries[s].hash) {
        if (g_geo_census.entries[s].hash == h && g_geo_census.entries[s].slot >= 0) {
            int slot = g_geo_census.entries[s].slot;
            if (geo_desc_matches_renderer_mesh(r, slot, desc)) {
                g_geo_census.entries[s].hits++;
                g_geo_census.dup_calls++;
                g_geo_census.total_calls++;
                g_geo_census.total_bytes += bytes;
                return slot;
            }
            g_geo_census.hash_collisions++;
        }
        s = (s + 1) & mask;
    }
    return -1;
}

/* Insert (or update) the hash entry for `desc`, recording `slot` as the
 * prototype so future content-identical meshes redirect to it. Called on
 * the slow path after a successful nu_add_mesh slot allocation. */
static void geo_dedup_insert(const NuRenderer* r, const NuMeshDesc* desc, int slot)
{
    if (!geo_dedup_enabled() && !geo_census_enabled()) return;
    if (!g_geo_census.entries) geo_census_init(4096);
    if (!g_geo_census.entries) return;
    size_t bytes = 0;
    uint64_t h = geo_hash_compute(desc, &bytes);
    if (!h) return;
    g_geo_census.total_calls++;
    g_geo_census.total_bytes += bytes;
    geo_census_grow();
    uint32_t mask = g_geo_census.capacity - 1u;
    uint32_t s = (uint32_t)h & mask;
    while (g_geo_census.entries[s].hash) {
        if (g_geo_census.entries[s].hash == h) {
            int entry_slot = g_geo_census.entries[s].slot;
            int same_entry = (slot < 0 || entry_slot < 0);
            if (!same_entry && r)
                same_entry = geo_desc_matches_renderer_mesh(r, entry_slot, desc);
            if (same_entry) {
                g_geo_census.entries[s].hits++;
                g_geo_census.dup_calls++;
                /* Only set slot once (first occurrence is the prototype). */
                if (g_geo_census.entries[s].slot < 0 && slot >= 0)
                    g_geo_census.entries[s].slot = slot;
                return;
            }
            g_geo_census.hash_collisions++;
        }
        s = (s + 1) & mask;
    }
    g_geo_census.entries[s].hash = h;
    g_geo_census.entries[s].hits = 1;
    g_geo_census.entries[s].slot = slot;
    g_geo_census.count++;
}

/* Census-only path used when NUSD_HASH_DEDUP is OFF: hash + record
 * but never insert a real slot. Kept as a thin wrapper so the existing
 * geo_census_record call site at the top of nu_add_mesh stays valid. */
static void geo_census_record(const NuMeshDesc* desc)
{
    if (!geo_census_enabled() || !desc) return;
    /* Use insert with slot=-1 so dedup mode never trips on a census-only
     * pass; entry stays in "observed, no prototype" state. */
    geo_dedup_insert(NULL, desc, -1);
}

static void geo_census_dump_and_reset(void)
{
    /* Dump if EITHER census or dedup was active this run — both populate
     * g_geo_census; dedup also wants to know the resulting hit-rate. */
    if ((!geo_census_enabled() && !geo_dedup_enabled())
        || g_geo_census.total_calls == 0) return;
    /* Top-N reuse histogram: count how many hashes have 1, 2, 3-10, 11+ hits. */
    uint64_t reuse_1 = 0, reuse_2 = 0, reuse_3_10 = 0, reuse_11p = 0;
    int      max_hits = 0;
    for (uint32_t i = 0; i < g_geo_census.capacity; i++) {
        int h = g_geo_census.entries[i].hits;
        if (h == 0) continue;
        if (h > max_hits) max_hits = h;
        if (h == 1)      reuse_1++;
        else if (h == 2) reuse_2++;
        else if (h <= 10) reuse_3_10++;
        else              reuse_11p++;
    }
    double dup_pct = 100.0 * (double)g_geo_census.dup_calls / (double)g_geo_census.total_calls;
    fprintf(stderr,
            "geo_hash_census: %llu nu_add_mesh calls -> %u unique geometries "
            "(%llu dup hits, %.1f%% dup rate, %llu same-hash nonmatches, "
            "%.1f GB hashed)\n"
            "geo_hash_census: reuse histogram — singletons=%llu, 2x=%llu, 3-10x=%llu, 11+=%llu, max_hits=%d\n",
            (unsigned long long)g_geo_census.total_calls,
            g_geo_census.count,
            (unsigned long long)g_geo_census.dup_calls,
            dup_pct,
            (unsigned long long)g_geo_census.hash_collisions,
            (double)g_geo_census.total_bytes / (1024.0 * 1024.0 * 1024.0),
            (unsigned long long)reuse_1,
            (unsigned long long)reuse_2,
            (unsigned long long)reuse_3_10,
            (unsigned long long)reuse_11p,
            max_hits);
    free(g_geo_census.entries);
    g_geo_census.entries  = NULL;
    g_geo_census.capacity = 0;
    g_geo_census.count    = 0;
    g_geo_census.total_calls = 0;
    g_geo_census.dup_calls   = 0;
    g_geo_census.hash_collisions = 0;
    g_geo_census.total_bytes = 0;
}

/* Convert a USD column-vector row-major 4x4 (translation at src[3], src[7],
 * src[11]) into VkTransformMatrixKHR's 3x4 row-major layout (also translation
 * at dst[3], dst[7], dst[11]). Both formats are row-major column-vector — the
 * mapping is just a copy of the first 12 elements (dropping the last
 * `[0,0,0,1]` row). The previous implementation transposed under the
 * assumption that the input was *row-vector* USD; in this codebase every
 * Python producer (`_transform7_to_mat4`, `_compute_transforms_kernel`,
 * `_batch_transform7_to_mat4`) writes column-vector row-major, so the
 * transpose silently zeroed every instance's translation (it picked up the
 * `src[12..14]` last-row entries, which are 0 in a proper rigid transform). */
static void usd4x4_to_vk3x4(const float src[16], float dst[12])
{
    memcpy(dst, src, 12 * sizeof(float));
}

/* PR 2: returns 1 if the GPU-driven transform path is armed AND ready
 * (count was uploaded via nu_set_transform_layout). Used by nu_render_tiled
 * to choose between the GPU dispatch and the host-side fallback. */
static int gpu_transforms_ready(NuRenderer* r)
{
    if (!r->use_gpu_transforms) return 0;
    return r->gpu_transform_count > 0;
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

static void renderer_usd_xform_to_model(const double in[16], float out[16])
{
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            out[row * 4 + col] = (float)in[col * 4 + row];
        }
    }
}

static uint32_t renderer_pack_rgba8(float r, float g, float b, float a)
{
    int ri = (int)lrintf(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f);
    int gi = (int)lrintf(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f);
    int bi = (int)lrintf(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f);
    int ai = (int)lrintf(fminf(fmaxf(a, 0.0f), 1.0f) * 255.0f);
    return (uint32_t)ri |
           ((uint32_t)gi << 8) |
           ((uint32_t)bi << 16) |
           ((uint32_t)ai << 24);
}

static uint64_t renderer_align_u64(uint64_t v, uint64_t align)
{
    return (v + align - 1u) & ~(align - 1u);
}

static void renderer_usd_xform_point(const double m[16],
                                     const float p[3],
                                     float out[3])
{
    double px = (double)p[0], py = (double)p[1], pz = (double)p[2];
    out[0] = (float)(m[0]*px + m[4]*py + m[8]*pz  + m[12]);
    out[1] = (float)(m[1]*px + m[5]*py + m[9]*pz  + m[13]);
    out[2] = (float)(m[2]*px + m[6]*py + m[10]*pz + m[14]);
}

static void renderer_transform_aabb_usd(const double m[16],
                                        const float obj_min[3],
                                        const float obj_max[3],
                                        float world_min[3],
                                        float world_max[3])
{
    world_min[0] = world_min[1] = world_min[2] =  FLT_MAX;
    world_max[0] = world_max[1] = world_max[2] = -FLT_MAX;
    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                float p[3] = {
                    ix ? obj_max[0] : obj_min[0],
                    iy ? obj_max[1] : obj_min[1],
                    iz ? obj_max[2] : obj_min[2],
                };
                float wp[3];
                renderer_usd_xform_point(m, p, wp);
                for (int k = 0; k < 3; k++) {
                    if (wp[k] < world_min[k]) world_min[k] = wp[k];
                    if (wp[k] > world_max[k]) world_max[k] = wp[k];
                }
            }
        }
    }
}

static void renderer_curve_chunk_bounds_world(const SceneCurve* c,
                                             uint32_t patch_start,
                                             uint32_t patch_count,
                                             float world_min[3],
                                             float world_max[3])
{
    float obj_min[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float obj_max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    uint32_t begin = patch_start * 4u;
    uint32_t end = begin + patch_count * 4u;
    for (uint32_t j = begin; j < end; j++) {
        uint32_t idx = c->patch_indices[j];
        if (idx >= (uint32_t)c->nv) continue;
        const float* p = &c->cvs[idx * 3u];
        float radius = c->widths ? c->widths[idx] * 0.5f : 0.025f;
        for (int k = 0; k < 3; k++) {
            float lo = p[k] - radius;
            float hi = p[k] + radius;
            if (lo < obj_min[k]) obj_min[k] = lo;
            if (hi > obj_max[k]) obj_max[k] = hi;
        }
    }
    if (obj_min[0] == FLT_MAX) {
        memcpy(world_min, c->bounds_min, sizeof(float) * 3);
        memcpy(world_max, c->bounds_max, sizeof(float) * 3);
        return;
    }
    renderer_transform_aabb_usd(c->world_xform, obj_min, obj_max,
                                world_min, world_max);
}

static int renderer_aabb_outside_vp(const float vp[16],
                                    const float bmin[3],
                                    const float bmax[3])
{
    int left = 1, right = 1, bottom = 1, top = 1, nearp = 1, farp = 1;
    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                float x = ix ? bmax[0] : bmin[0];
                float y = iy ? bmax[1] : bmin[1];
                float z = iz ? bmax[2] : bmin[2];
                float cx = vp[0]*x  + vp[1]*y  + vp[2]*z  + vp[3];
                float cy = vp[4]*x  + vp[5]*y  + vp[6]*z  + vp[7];
                float cz = vp[8]*x  + vp[9]*y  + vp[10]*z + vp[11];
                float cw = vp[12]*x + vp[13]*y + vp[14]*z + vp[15];
                if (cw <= 0.0f) return 0;
                left   &= (cx < -cw);
                right  &= (cx >  cw);
                bottom &= (cy < -cw);
                top    &= (cy >  cw);
                nearp  &= (cz < -cw);
                farp   &= (cz >  cw);
            }
        }
    }
    return left || right || bottom || top || nearp || farp;
}

static void renderer_raster_curves_clear(NuRenderer* r)
{
    if (!r) return;
    if (r->raster_curves.vertex_buffer) {
        gpu_destroy_buffer(r->gpu, r->raster_curves.vertex_buffer);
        r->raster_curves.vertex_buffer = NULL;
    }
    if (r->raster_curves.index_buffer) {
        gpu_destroy_buffer(r->gpu, r->raster_curves.index_buffer);
        r->raster_curves.index_buffer = NULL;
    }
    free(r->raster_curves.draws);
    r->raster_curves.draws = NULL;
    r->raster_curves.ndraws = 0;
    r->raster_curves.total_vertices = 0;
    r->raster_curves.total_indices = 0;
    r->raster_curves.total_index_bytes = 0;
}

static int renderer_curve_select_uniform_color(const MaterialCollection* mc,
                                               const SceneCurve* c,
                                               float out_color[3])
{
    out_color[0] = 0.34f;
    out_color[1] = 0.27f;
    out_color[2] = 0.18f;
    if (!c) return 0;

    if (mc && c->material_index >= 0 && c->material_index < mc->nmaterials) {
        const SceneMaterial* smat = &mc->materials[c->material_index];
        const MaterialParams* mp = &smat->params;
        float mr = fminf(fmaxf(mp->base_color[0], 0.0f), 1.0f);
        float mg = fminf(fmaxf(mp->base_color[1], 0.0f), 1.0f);
        float mb = fminf(fmaxf(mp->base_color[2], 0.0f), 1.0f);
        if (smat->has_ptex_average_color ||
            (mp->use_vertex_color == 0 &&
             !renderer_color_is_debug_placeholder(mr, mg, mb))) {
            out_color[0] = mr;
            out_color[1] = mg;
            out_color[2] = mb;
            return 1;
        }
    }

    if (c->has_display_color &&
        !renderer_color_is_debug_placeholder(c->display_color[0],
                                             c->display_color[1],
                                             c->display_color[2])) {
        out_color[0] = fminf(fmaxf(c->display_color[0], 0.0f), 1.0f);
        out_color[1] = fminf(fmaxf(c->display_color[1], 0.0f), 1.0f);
        out_color[2] = fminf(fmaxf(c->display_color[2], 0.0f), 1.0f);
        return 0;
    }

    return 0;
}

static GpuPipeline renderer_create_curve_pipeline(NuRenderer* r)
{
    char vert_path[1024], tesc_path[1024], tese_path[1024], frag_path[1024];
    snprintf(vert_path, sizeof(vert_path), "%s/curve.vert.spv", SHADER_DIR);
    snprintf(tesc_path, sizeof(tesc_path), "%s/curve.tesc.spv", SHADER_DIR);
    snprintf(tese_path, sizeof(tese_path), "%s/curve.tese.spv", SHADER_DIR);
    snprintf(frag_path, sizeof(frag_path), "%s/curve.frag.spv", SHADER_DIR);

    uint32_t vsz = 0, tcsz = 0, tesz = 0, fsz = 0;
    uint32_t* vspv = load_shader_file(vert_path, &vsz);
    uint32_t* tcspv = load_shader_file(tesc_path, &tcsz);
    uint32_t* tespv = load_shader_file(tese_path, &tesz);
    uint32_t* fspv = load_shader_file(frag_path, &fsz);
    if (!vspv || !tcspv || !tespv || !fspv) {
        free(vspv);
        free(tcspv);
        free(tespv);
        free(fspv);
        return NULL;
    }

    GpuVertexAttrib attribs[3] = {
        { .location = 0, .offset = 0,  .format = GPU_FORMAT_FLOAT3 },
        { .location = 1, .offset = 12, .format = GPU_FORMAT_FLOAT },
        { .location = 2, .offset = 16, .format = GPU_FORMAT_UINT },
    };
    GpuPipelineDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.vert_spv = vspv;
    desc.vert_size = vsz;
    desc.tesc_spv = tcspv;
    desc.tesc_size = tcsz;
    desc.tese_spv = tespv;
    desc.tese_size = tesz;
    desc.frag_spv = fspv;
    desc.frag_size = fsz;
    desc.push_constant_size = (uint32_t)sizeof(GpuMeshPushConstants);
    desc.vertex_stride = (uint32_t)sizeof(RendererCurveVertex);
    desc.patch_control_points = 4u;
    desc.attribs = attribs;
    desc.nattribs = 3u;

    GpuPipeline pipe = gpu_create_pipeline(r->gpu, &desc);
    free(vspv);
    free(tcspv);
    free(tespv);
    free(fspv);
    return pipe;
}

static int renderer_build_raster_curves(NuRenderer* r,
                                        const Scene* scene,
                                        const MaterialCollection* mc)
{
    renderer_raster_curves_clear(r);
    if (!r || !scene || scene->ncurves <= 0) return 1;

    int patch_chunk_i = env_int_or("NUSD_RASTER_CURVE_PATCH_CHUNK", 8192);
    if (patch_chunk_i < 256) patch_chunk_i = 256;
    if (patch_chunk_i > 65536) patch_chunk_i = 65536;
    uint32_t patch_chunk = (uint32_t)patch_chunk_i;

    uint64_t total_vertices = 0;
    uint64_t total_indices = 0;
    uint64_t total_patches = 0;
    int ndraws = 0;
    for (int i = 0; i < scene->ncurves; i++) {
        const SceneCurve* c = &scene->curves[i];
        if (!c->cvs || c->nv <= 0 || !c->patch_indices || c->npatches <= 0)
            continue;
        total_vertices += (uint64_t)c->nv;
        total_indices += (uint64_t)c->npatches * 4u;
        total_patches += (uint64_t)c->npatches;
        ndraws += (c->npatches + (int)patch_chunk - 1) / (int)patch_chunk;
    }
    if (ndraws == 0) return 1;
    if (total_vertices > UINT32_MAX || total_indices > UINT32_MAX) {
        set_error(r, "Raster curve buffers exceed 32-bit index limits");
        return 0;
    }

    RendererRasterCurve* draws = (RendererRasterCurve*)calloc(
        (size_t)ndraws, sizeof(RendererRasterCurve));
    RendererCurveVertex* vb = (RendererCurveVertex*)malloc(
        (size_t)total_vertices * sizeof(RendererCurveVertex));
    uint8_t* ib = (uint8_t*)malloc((size_t)total_indices * sizeof(uint32_t));
    if (!draws || !vb || !ib) {
        free(draws);
        free(vb);
        free(ib);
        set_error(r, "Failed to allocate raster curve upload buffers");
        return 0;
    }

    uint32_t voff = 0;
    uint64_t ibyte = 0;
    int chunks16 = 0;
    int chunks32 = 0;
    int draw_i = 0;
    for (int i = 0; i < scene->ncurves; i++) {
        const SceneCurve* c = &scene->curves[i];
        if (!c->cvs || c->nv <= 0 || !c->patch_indices || c->npatches <= 0)
            continue;

        float uniform_color[3];
        int material_color = renderer_curve_select_uniform_color(mc, c, uniform_color);
        int use_authored_colors =
            !material_color && c->colors && c->has_display_color &&
            !renderer_color_is_debug_placeholder(c->display_color[0],
                                                 c->display_color[1],
                                                 c->display_color[2]);

        for (int v = 0; v < c->nv; v++) {
            RendererCurveVertex* dst = &vb[(size_t)voff + (size_t)v];
            dst->position[0] = c->cvs[v*3 + 0];
            dst->position[1] = c->cvs[v*3 + 1];
            dst->position[2] = c->cvs[v*3 + 2];
            dst->width = c->widths ? c->widths[v] : 0.05f;
            if (use_authored_colors) {
                dst->color_rgba8 = renderer_pack_rgba8(
                    c->colors[v*3 + 0],
                    c->colors[v*3 + 1],
                    c->colors[v*3 + 2],
                    1.0f);
            } else {
                dst->color_rgba8 = renderer_pack_rgba8(
                    uniform_color[0], uniform_color[1], uniform_color[2], 1.0f);
            }
        }

        for (uint32_t patch_start = 0;
             patch_start < (uint32_t)c->npatches;
             patch_start += patch_chunk) {
            uint32_t chunk_n = (uint32_t)c->npatches - patch_start;
            if (chunk_n > patch_chunk) chunk_n = patch_chunk;
            uint32_t index_start = patch_start * 4u;
            uint32_t index_count = chunk_n * 4u;
            uint32_t min_idx = UINT32_MAX;
            uint32_t max_idx = 0;
            for (uint32_t j = 0; j < index_count; j++) {
                uint32_t idx = c->patch_indices[index_start + j];
                if (idx < min_idx) min_idx = idx;
                if (idx > max_idx) max_idx = idx;
            }
            if (min_idx == UINT32_MAX) min_idx = max_idx = 0;
            uint64_t base_vertex64 = (uint64_t)voff + (uint64_t)min_idx;
            int use_u16 =
                max_idx >= min_idx &&
                (uint64_t)max_idx - (uint64_t)min_idx <= UINT16_MAX &&
                base_vertex64 <= INT32_MAX;
            int index_bits = use_u16 ? 16 : 32;
            uint64_t index_align = use_u16 ? 2u : 4u;
            ibyte = renderer_align_u64(ibyte, index_align);

            RendererRasterCurve* d = &draws[draw_i++];
            d->curve_id = (uint32_t)i;
            d->index_byte_offset = ibyte;
            d->npatches = chunk_n;
            d->vertex_offset = use_u16 ? (int32_t)base_vertex64 : 0;
            d->index_type_bits = index_bits;
            d->basis_id = c->type_is_cubic ? (int)c->basis : (int)CURVE_BASIS_LINEAR;
            d->material_index = c->material_index;
            memcpy(d->color, uniform_color, sizeof(d->color));
            renderer_usd_xform_to_model(c->world_xform, d->model);
            renderer_curve_chunk_bounds_world(c, patch_start, chunk_n,
                                              d->bounds_min, d->bounds_max);
            if (use_u16) {
                uint16_t* dst = (uint16_t*)(void*)(ib + ibyte);
                for (uint32_t j = 0; j < index_count; j++)
                    dst[j] = (uint16_t)(c->patch_indices[index_start + j] - min_idx);
                ibyte += (uint64_t)index_count * sizeof(uint16_t);
                chunks16++;
            } else {
                uint32_t* dst = (uint32_t*)(void*)(ib + ibyte);
                for (uint32_t j = 0; j < index_count; j++)
                    dst[j] = c->patch_indices[index_start + j] + voff;
                ibyte += (uint64_t)index_count * sizeof(uint32_t);
                chunks32++;
            }
        }

        voff += (uint32_t)c->nv;
    }

    GpuBufferDesc vbd;
    vbd.usage = GPU_BUFFER_VERTEX;
    vbd.size = total_vertices * sizeof(RendererCurveVertex);
    vbd.data = vb;
    GpuBufferDesc ibd;
    ibd.usage = GPU_BUFFER_INDEX;
    ibd.size = ibyte;
    ibd.data = ib;
    r->raster_curves.vertex_buffer = gpu_create_buffer(r->gpu, &vbd);
    r->raster_curves.index_buffer = gpu_create_buffer(r->gpu, &ibd);
    free(vb);
    free(ib);

    if (!r->raster_curves.vertex_buffer || !r->raster_curves.index_buffer) {
        free(draws);
        renderer_raster_curves_clear(r);
        set_error(r, "Failed to create raster curve GPU buffers");
        return 0;
    }

    r->raster_curves.draws = draws;
    r->raster_curves.ndraws = draw_i;
    r->raster_curves.total_vertices = (uint32_t)total_vertices;
    r->raster_curves.total_indices = (uint32_t)total_indices;
    r->raster_curves.total_index_bytes = ibd.size;
    fprintf(stderr,
            "nusd_renderer: raster curves ready — %d draw chunks, %u CVs, "
            "%llu patches, %d-patch chunks, %.1f MiB VBO, %.1f MiB IBO "
            "(%d u16 chunks / %d u32 chunks)\n",
            r->raster_curves.ndraws,
            r->raster_curves.total_vertices,
            (unsigned long long)total_patches,
            patch_chunk_i,
            (double)vbd.size / (1024.0 * 1024.0),
            (double)ibd.size / (1024.0 * 1024.0),
            chunks16, chunks32);
    return 1;
}

/* Rebuild merged vertex/index buffers and upload to GPU */
static int rebuild_gpu_buffers(NuRenderer* r)
{
    /* Defense-in-depth: after nu_finalize_scene drops the CPU mirror, this
     * function has nothing to upload from. The mutation-API guards
     * (check_scene_mutable) should prevent us getting here, but if some
     * other code path flips scene_dirty=1, refuse to nuke the existing
     * GPU buffers — they're still correct from the pre-finalize build. */
    if (r->scene_finalized) {
        if (r->vertex_buffer && r->index_buffer) {
            r->scene_dirty = 0;
            return 1;
        }
        set_error(r, "rebuild_gpu_buffers: scene finalized but GPU buffers missing");
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
        return 1; /* empty scene is valid */
    }

    /* Destroy old GPU buffers */
    if (r->vertex_buffer) { gpu_destroy_buffer(r->gpu, r->vertex_buffer); r->vertex_buffer = NULL; }
    if (r->index_buffer)  { gpu_destroy_buffer(r->gpu, r->index_buffer);  r->index_buffer = NULL; }

    /* Peak-RSS optimization: when the caller will finalize (drop CPU
     * mirror) anyway, free the CPU side of each buffer immediately after
     * gpu_create_buffer returns (synchronous — vkQueueWaitIdle at the
     * end of staging). That removes the ~6 + 3 GB host overlap on DSX
     * while the second-half GPU upload runs. Gated on
     * NUSD_AUTO_FINALIZE so paths that legitimately need a second
     * rebuild_gpu_buffers (re-dirty after build_accel) keep their
     * source data.
     *
     * Final RSS is unchanged — nu_finalize_scene's tail free becomes a
     * free(NULL) no-op when this path runs. Diagnostic log lines from
     * finalize still fire.
     *
     * Cache the env var lookup at function-entry time so a churning
     * scene doesn't pay the getenv cost per rebuild. */
    int drop_cpu_after_upload = 0;
    {
        const char* env = getenv("NUSD_AUTO_FINALIZE");
        if (env && env[0] && env[0] != '0') drop_cpu_after_upload = 1;
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
        if (drop_cpu_after_upload) {
            free(r->cpu_vertices);
            r->cpu_vertices = NULL;
            r->cpu_vertex_capacity = 0;
        }
    }

    /* Upload merged index buffer */
    if (r->cpu_indices) {
        GpuBufferDesc ib_desc;
        ib_desc.usage = GPU_BUFFER_INDEX;
        ib_desc.size  = (uint64_t)total_i * sizeof(uint32_t);
        ib_desc.data  = r->cpu_indices;
        r->index_buffer = gpu_create_buffer(r->gpu, &ib_desc);
        if (!r->index_buffer) {
            set_error(r, "Failed to create index buffer");
            return 0;
        }
        if (drop_cpu_after_upload) {
            free(r->cpu_indices);
            r->cpu_indices = NULL;
            r->cpu_index_capacity = 0;
#ifdef __GLIBC__
            /* Return pages to the kernel so the peak measurement is
             * meaningful before nu_finalize_scene's tail trim runs. */
            malloc_trim(0);
#endif
        }
    }

    r->total_vertices = total_v;
    r->total_indices = total_i;
    r->scene_dirty = 0;
    r->accel_dirty = 1;

    return 1;
}

/* Commit any pending runtime textures via gpu_upload_materials. Called
 * by rebuild_accel() before gpu_build_rt_scene so the texture array is
 * sized correctly for the descriptor set layout. Idempotent — safe to
 * call when there are no pending textures. */
static void commit_runtime_textures(NuRenderer* r)
{
    if (!r || r->runtime_textures_count == 0) return;
    if (r->runtime_textures_committed) return;

    /* Build a GpuTextureData array referencing the renderer-owned pixels. */
    GpuTextureData* tex = (GpuTextureData*)calloc(
        (size_t)r->runtime_textures_count, sizeof(GpuTextureData));
    if (!tex) return;
    for (int i = 0; i < r->runtime_textures_count; i++) {
        tex[i].pixels  = r->runtime_textures[i].pixels;
        tex[i].width   = r->runtime_textures[i].width;
        tex[i].height  = r->runtime_textures[i].height;
        tex[i].is_srgb = 1;  /* IsaacLab DexCube glyphs are sRGB color data */
    }
    /* nmaterials = 0 → placeholder material is synthesized; we only care
     * about the texture array binding (4) which the rchit fast_mode path
     * samples by tex_index. */
    gpu_upload_materials(r->gpu, NULL, 0, tex, r->runtime_textures_count);
    free(tex);
    r->runtime_textures_committed = 1;
}

/* RT culled-proxy mode: build the TLAS from only the camera-visible PI set.
 * The full Moana TLAS is ~202M instances -> ~46 GiB acceleration structure,
 * over the 32 GB device limit -> the build fails. Restricting the TLAS to the
 * camera-visible instances (same frustum + screen-angular test as the raster
 * draw cull) is UE5's Nanite-RT-proxy strategy. Opt-in so the default IsaacLab
 * path is unchanged. */
static int rt_cull_active(void)
{
    const char* a = getenv("NUSD_NO_CULL_ALL_GEOMETRY");
    const char* b = getenv("NUSD_RT_CULL");
    const char* c = getenv("NUSD_ALL_GEOMETRY_NO_CULL");
    if (b && b[0] && b[0] != '0') return 1;
    if (a && a[0] && a[0] != '0') return 1;
    if (c && c[0] && c[0] != '0') return 1;
    return 0;
}

static int compact_pi_batches_active(void)
{
    return env_enabled_default_off("NUSD_RENDER_PI_BATCHES") ||
           env_enabled_default_off("NUSD_PI_COMPACT_ONLY") ||
           env_enabled_default_off("NUSD_NO_CULL_ALL_GEOMETRY") ||
           env_enabled_default_off("NUSD_ALL_GEOMETRY_NO_CULL") ||
           env_enabled_default_off("NUSD_RT_CULL");
}

/* True if the sphere (center c, radius r) is fully outside any frustum plane.
 * Planes carry inward normals (extract_frustum_planes), so p.xyz·c + p.w is the
 * signed distance to the plane; < -r means the whole sphere is on the far side. */
static inline int sphere_outside_frustum(const float planes[6][4],
                                         const float c[3], float r)
{
    for (int i = 0; i < 6; i++) {
        const float* pl = planes[i];
        if (pl[0]*c[0] + pl[1]*c[1] + pl[2]*c[2] + pl[3] < -r) return 1;
    }
    return 0;
}

/* Transform a prototype-local bounding sphere (center c, radius rad) by a
 * 12-float column-major affine m (SceneInstanceTransform) into world space.
 * Writes the world center to wc and returns the world radius (rad * max column
 * scale — a conservative upper bound under non-uniform scale). */
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

/* Prototype-local bounding sphere of a mesh (from its proto-local AABB).
 * Returns 0 when the local bounds are degenerate/uninitialized — the caller
 * must then treat the whole batch as visible rather than risk culling real
 * geometry on a bogus (NaN/inf) radius. */
static inline int mesh_proto_sphere(const RendererMesh* pm, float c[3], float* rad)
{
    const float* mn = pm->local_bounds_min;
    const float* mx = pm->local_bounds_max;
    if (!(mx[0] >= mn[0] && mx[1] >= mn[1] && mx[2] >= mn[2]) || mn[0] >= 1e37f)
        return 0;
    c[0] = 0.5f * (mn[0] + mx[0]);
    c[1] = 0.5f * (mn[1] + mx[1]);
    c[2] = 0.5f * (mn[2] + mx[2]);
    float ex = mx[0]-mn[0], ey = mx[1]-mn[1], ez = mx[2]-mn[2];
    *rad = 0.5f * sqrtf(ex*ex + ey*ey + ez*ez);
    return 1;
}

/* Defined later in the file (after rebuild_accel). */
static void extract_frustum_planes(const float vp[16], float planes[6][4]);

static int upload_rt_ptex_triangle_colors(NuRenderer* r)
{
    if (!r || !r->gpu) return 0;

    uint64_t total = 0;
    uint32_t mesh_count = 0;
    for (int i = 0; i < r->nmeshes; i++) {
        RendererMesh* m = &r->meshes[i];
        m->ptex_color_offset = RENDERER_PTEX_COLOR_NONE;
        if (!m->active || !m->ptex_tri_colors || m->ptex_tri_color_count == 0)
            continue;
        if (m->ptex_tri_color_count != (uint32_t)m->nindices)
            continue;
        total += m->ptex_tri_color_count;
        mesh_count++;
    }

    if (total == 0) {
        gpu_upload_rt_triangle_colors(r->gpu, NULL, 0);
        return 1;
    }
    if (total > UINT32_MAX) {
        fprintf(stderr,
                "nusd_renderer: real Ptex triangle-corner colors exceed 32-bit "
                "index space (%llu colors); disabling Ptex color SSBO\n",
                (unsigned long long)total);
        gpu_upload_rt_triangle_colors(r->gpu, NULL, 0);
        return 1;
    }

    uint32_t* flat = (uint32_t*)malloc((size_t)total * sizeof(uint32_t));
    if (!flat) return 0;

    uint32_t offset = 0;
    for (int i = 0; i < r->nmeshes; i++) {
        RendererMesh* m = &r->meshes[i];
        if (!m->active || !m->ptex_tri_colors || m->ptex_tri_color_count == 0)
            continue;
        if (m->ptex_tri_color_count != (uint32_t)m->nindices)
            continue;
        m->ptex_color_offset = offset;
        memcpy(flat + offset, m->ptex_tri_colors,
               (size_t)m->ptex_tri_color_count * sizeof(uint32_t));
        offset += m->ptex_tri_color_count;
    }

    int ok = gpu_upload_rt_triangle_colors(r->gpu, flat, offset);
    if (ok) {
        fprintf(stderr,
                "nusd_renderer: uploaded real Ptex triangle-corner colors: "
                "%u colors (%u triangles), %u meshes, %.1f MB\n",
                offset, offset / 3u, mesh_count,
                (double)((uint64_t)offset * sizeof(uint32_t)) / (1024.0 * 1024.0));
    }
    free(flat);
    return ok;
}

/* Build RT acceleration structures from current mesh data */
static int rebuild_accel(NuRenderer* r)
{
    if (!r->enable_rt || !gpu_rt_available(r->gpu)) return 1;
    /* Phase 11.A.2.5: scenes with only curves (no triangle meshes) still
     * need the RT pipeline — total_indices alone is insufficient gating. */
    if (r->total_indices == 0 && r->curves.nseg == 0) return 1;

    /* Phase B textures: commit runtime-loaded textures BEFORE
     * gpu_build_rt_scene so its descriptor set layout sees the right
     * mat_tex_count. No-op when no textures were registered. */
    commit_runtime_textures(r);

    /* Load RT shaders */
    char rgen_path[1024], rmiss_path[1024], rchit_path[1024];
    char rint_path[1024], rchit_curve_path[1024];
    snprintf(rgen_path, sizeof(rgen_path), "%s/raytrace.rgen.spv", SHADER_DIR);
    snprintf(rmiss_path, sizeof(rmiss_path), "%s/raytrace.rmiss.spv", SHADER_DIR);
    snprintf(rchit_path, sizeof(rchit_path), "%s/raytrace.rchit.spv", SHADER_DIR);
    snprintf(rint_path,  sizeof(rint_path),  "%s/raytrace_curve.rint.spv",  SHADER_DIR);
    snprintf(rchit_curve_path, sizeof(rchit_curve_path), "%s/raytrace_curve.rchit.spv", SHADER_DIR);

    uint32_t rgen_sz = 0, rmiss_sz = 0, rchit_sz = 0;
    uint32_t rint_sz = 0, rchit_curve_sz = 0;
    uint32_t* rgen_spv  = load_shader_file(rgen_path, &rgen_sz);
    uint32_t* rmiss_spv = load_shader_file(rmiss_path, &rmiss_sz);
    uint32_t* rchit_spv = load_shader_file(rchit_path, &rchit_sz);
    /* Curve shaders are optional — only loaded if curves are present.
     * NULL/0 keeps the legacy 3-group pipeline path. */
    uint32_t* rint_spv       = (r->curves.nseg > 0) ? load_shader_file(rint_path, &rint_sz) : NULL;
    uint32_t* rchit_curve_spv = (r->curves.nseg > 0) ? load_shader_file(rchit_curve_path, &rchit_curve_sz) : NULL;

    if (!rgen_spv || !rmiss_spv || !rchit_spv) {
        free(rgen_spv); free(rmiss_spv); free(rchit_spv);
        free(rint_spv); free(rchit_curve_spv);
        set_error(r, "Failed to load RT shaders");
        return 0;
    }
    if (r->curves.nseg > 0 && (!rint_spv || !rchit_curve_spv)) {
        free(rgen_spv); free(rmiss_spv); free(rchit_spv);
        free(rint_spv); free(rchit_curve_spv);
        set_error(r, "Failed to load curve RT shaders (rint/rchit_curve)");
        return 0;
    }

    /* Build RT mesh descriptors */
    int rt_count = 0;
    for (int i = 0; i < r->nmeshes; i++)
        if (r->meshes[i].active && r->meshes[i].nindices > 0)
            rt_count++;

    /* Culled-proxy setup: when active, only camera-visible PI instances enter
     * the TLAS (same frustum + screen-angular test as the raster draw cull).
     * Count pass here and the fill pass below run identical cull math. */
    int   rt_cull = rt_cull_active();
    float rt_planes[6][4];
    float rt_eye[3] = {0.0f, 0.0f, 0.0f};
    float rt_min_angular = 0.0f;
    if (rt_cull) {
        float rt_vp[16];
        camera_get_vp(&r->camera, rt_vp);
        camera_get_eye(&r->camera, rt_eye);
        extract_frustum_planes(rt_vp, rt_planes);
        const char* ang = getenv("NUSD_CULL_MIN_ANGULAR");
        rt_min_angular = ang ? (float)atof(ang) : 0.0f;
    }

    /* Phase 2: extra descriptor slots for compact nested-PI batch instances. */
    uint64_t n_batch_inst = 0;
    for (int b = 0; b < r->npi_batches; b++) {
        const SceneInstanceBatch* pb = &r->pi_batches[b];
        if (!rt_cull) { n_batch_inst += pb->transform_count; continue; }
        int rp = pb->prototype_mesh_idx;
        if (rp < 0 || rp >= r->nmeshes) continue;
        const RendererMesh* pm = &r->meshes[rp];
        if (pm->nindices == 0) continue;
        float pc[3], prad;
        if (!mesh_proto_sphere(pm, pc, &prad)) {
            n_batch_inst += pb->transform_count;   /* degenerate bounds: keep all */
            continue;
        }
        for (uint32_t k = 0; k < pb->transform_count; k++) {
            const SceneInstanceTransform* t = &r->pi_transforms[pb->transform_offset + k];
            float wc[3];
            float wr = pi_instance_world_sphere(t->m, pc, prad, wc);
            if (sphere_outside_frustum(rt_planes, wc, wr)) continue;
            if (rt_min_angular > 0.0f) {
                float dx = wc[0]-rt_eye[0], dy = wc[1]-rt_eye[1], dz = wc[2]-rt_eye[2];
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist > 1e-4f && wr/dist < rt_min_angular) continue;
            }
            n_batch_inst++;
        }
    }
    int rt_total = rt_count + (int)n_batch_inst;

    GpuRtMeshDesc* rt_meshes = (GpuRtMeshDesc*)calloc(
        (size_t)rt_total, sizeof(GpuRtMeshDesc));

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

    int residency_enabled = rt_camera_residency_enabled();
    uint32_t residency_selected_visible = 0;
    uint32_t residency_excluded_visible = 0;
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
        uint8_t rt_visible = m->visible ? 1u : 0u;
        if (residency_enabled && rt_visible) {
            if (mesh_bounds_intersects_camera(r, m)) {
                residency_selected_visible++;
            } else {
                rt_visible = 0u;
                residency_excluded_visible++;
            }
        }
        rt_meshes[ri].visible = rt_visible;
        rt_meshes[ri].source_id = (uint32_t)i;
        rt_meshes[ri].material_index = m->material_index;
        rt_meshes[ri].debug_name = m->name;
        memcpy(rt_meshes[ri].bounds_min, m->bounds_min, sizeof(rt_meshes[ri].bounds_min));
        memcpy(rt_meshes[ri].bounds_max, m->bounds_max, sizeof(rt_meshes[ri].bounds_max));

        /* World transform: USD row-vector 4x4 → VkTransformMatrixKHR 3x4
         * (column-vector row-major; translation in last column). */
        usd4x4_to_vk3x4(m->transform, rt_meshes[ri].transform);

        rt_meshes[ri].color[0] = m->color[0];
        rt_meshes[ri].color[1] = m->color[1];
        rt_meshes[ri].color[2] = m->color[2];
        rt_meshes[ri].geo_hash = m->geo_hash;
        rt_meshes[ri].ptex_color_offset = m->ptex_color_offset;

        ri++;
    }

    /* Phase 2: expand compact nested-PI batches into instanced RT descriptors.
     * Each instance shares the prototype's geometry + BLAS via prototype_idx.
     * Never-cull bounds so the residency pass can't drop the foliage. */
    for (int b = 0; b < r->npi_batches; b++) {
        const SceneInstanceBatch* pb = &r->pi_batches[b];
        int rp = pb->prototype_mesh_idx;
        if (rp < 0 || rp >= r->nmeshes) continue;
        RendererMesh* pm = &r->meshes[rp];
        if (pm->nindices == 0) continue;
        int proto_ri = mesh_to_ri[rp];
        float pc[3], prad;
        int can_cull = rt_cull ? mesh_proto_sphere(pm, pc, &prad) : 0;
        for (uint32_t k = 0; k < pb->transform_count && ri < rt_total; k++) {
            const SceneInstanceTransform* t = &r->pi_transforms[pb->transform_offset + k];
            if (rt_cull && can_cull) {
                float wc[3];
                float wr = pi_instance_world_sphere(t->m, pc, prad, wc);
                if (sphere_outside_frustum(rt_planes, wc, wr)) continue;
                if (rt_min_angular > 0.0f) {
                    float dx = wc[0]-rt_eye[0], dy = wc[1]-rt_eye[1], dz = wc[2]-rt_eye[2];
                    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (dist > 1e-4f && wr/dist < rt_min_angular) continue;
                }
            }
            rt_meshes[ri].vertex_buf    = r->vertex_buffer;
            rt_meshes[ri].vertex_count  = (uint32_t)pm->nvertices;
            rt_meshes[ri].vertex_stride = (r->vertex_stride_floats ? r->vertex_stride_floats : 12u) * sizeof(float);
            rt_meshes[ri].index_buf     = r->index_buffer;
            rt_meshes[ri].index_count   = (uint32_t)pm->nindices;
            rt_meshes[ri].vertex_offset = pm->vertex_offset;
            rt_meshes[ri].index_offset  = pm->index_offset;
            rt_meshes[ri].prototype_idx = (proto_ri >= 0) ? proto_ri : ri;
            rt_meshes[ri].visible = 1;
            rt_meshes[ri].source_id = 0x40000000u | (uint32_t)b;
            rt_meshes[ri].material_index = pm->material_index;
            rt_meshes[ri].debug_name = pm->name;
            rt_meshes[ri].bounds_min[0] = rt_meshes[ri].bounds_min[1] = rt_meshes[ri].bounds_min[2] = -3.4e38f;
            rt_meshes[ri].bounds_max[0] = rt_meshes[ri].bounds_max[1] = rt_meshes[ri].bounds_max[2] =  3.4e38f;
            scene_instance_transform_to_vk3x4(t, rt_meshes[ri].transform);
            rt_meshes[ri].color[0] = pm->color[0];
            rt_meshes[ri].color[1] = pm->color[1];
            rt_meshes[ri].color[2] = pm->color[2];
            rt_meshes[ri].geo_hash = pm->geo_hash;
            rt_meshes[ri].ptex_color_offset = pm->ptex_color_offset;
            ri++;
        }
    }
    if (r->npi_batches > 0)
        fprintf(stderr, "nusd_renderer: RT — %d PI batches -> %d instanced descriptors\n",
                r->npi_batches, ri - rt_count);

    if (residency_enabled) {
        int residency_allow_drop = rt_camera_residency_allow_drop();
        fprintf(stderr,
                "nusd_renderer: RT camera residency selected %u visible meshes, "
                "excluded %u outside camera influence (margin=%.2f, allow_drop=%d)\n",
                residency_selected_visible, residency_excluded_visible,
                env_float_or("NUSD_RT_RESIDENCY_MARGIN", 0.35f),
                residency_allow_drop);
        write_rt_residency_manifest(r, residency_selected_visible,
                                    residency_excluded_visible);
        if (residency_excluded_visible > 0 && !residency_allow_drop) {
            free(mesh_to_ri);
            free(rt_meshes);
            free(rgen_spv);
            free(rmiss_spv);
            free(rchit_spv);
            free(rint_spv);
            free(rchit_curve_spv);
            set_error(r,
                      "RT camera residency would drop visible geometry; "
                      "set NUSD_RT_CAMERA_RESIDENCY_ALLOW_DROP=1 for diagnostic partial renders");
            return 0;
        }
    }
    free(mesh_to_ri);

    gpu_destroy_rt_scene(r->gpu);
    if (!upload_rt_ptex_triangle_colors(r)) {
        free(rt_meshes);
        free(rgen_spv);
        free(rmiss_spv);
        free(rchit_spv);
        free(rint_spv);
        free(rchit_curve_spv);
        set_error(r, "Failed to upload real Ptex triangle-corner colors");
        return 0;
    }
    {
        int rt_desc_count = rt_cull ? ri : rt_total;
        for (int j = 0; j < rt_desc_count; j++) {
            uint32_t sid = rt_meshes[j].source_id;
            if ((sid & 0x40000000u) != 0u) {
                int b = (int)(sid & 0x3FFFFFFFu);
                if (b >= 0 && b < r->npi_batches) {
                    int rp = r->pi_batches[b].prototype_mesh_idx;
                    if (rp >= 0 && rp < r->nmeshes)
                        rt_meshes[j].ptex_color_offset = r->meshes[rp].ptex_color_offset;
                }
            } else if (sid < (uint32_t)r->nmeshes) {
                rt_meshes[j].ptex_color_offset = r->meshes[sid].ptex_color_offset;
            }
        }
    }

    if (rt_cull && env_enabled_default_off("NUSD_CULL_DIAG")) {
        uint64_t pi_total = 0;
        for (int b = 0; b < r->npi_batches; b++)
            pi_total += r->pi_batches[b].transform_count;
        fprintf(stderr, "nusd_renderer: RT culled-proxy TLAS — %d instances built "
                "(%d base mesh + %llu visible PI of %llu total)\n",
                ri, rt_count, (unsigned long long)n_batch_inst,
                (unsigned long long)pi_total);
    }

    /* Derive the on-disk BLAS cache sidecar from the source USD path
     * (NULL/"" for handle scenes → BLAS cache disabled for this build). */
    gpu_set_blas_cache_path(r->gpu, r->usd_path);

    int ok = gpu_build_rt_scene(r->gpu,
        rt_meshes, (uint32_t)(rt_cull ? ri : rt_total),
        rgen_spv, rgen_sz,
        rmiss_spv, rmiss_sz,
        rchit_spv, rchit_sz,
        rint_spv, rint_sz,
        rchit_curve_spv, rchit_curve_sz);

    free(rt_meshes);
    free(rgen_spv);
    free(rmiss_spv);
    free(rchit_spv);
    free(rint_spv);
    free(rchit_curve_spv);

    if (!ok) {
        set_error(r, "Failed to build RT acceleration structures");
        return 0;
    }

    r->accel_dirty = 0;
    r->tiled_pipeline_built = 0;    /* tiled descriptor set references old TLAS/scene */
    r->raycast_pipeline_built = 0;  /* raycast descriptor set references old TLAS */
    return 1;
}

/* ---- Lifecycle ---- */

static void write_fixed_string(char* dst, size_t dst_size, const char* src)
{
    if (!dst || dst_size == 0) return;
    snprintf(dst, dst_size, "%s", src ? src : "");
}

/* Capabilities this backend supports regardless of the runtime device/config.
 * Hardware/instance-gated capabilities (RT modes, RT-only AOVs, CUDA interop,
 * async render, materials) are layered on in nu_get_backend_info once a
 * renderer instance is available. MESHLETS / GEOMETRY_CACHE are intentionally
 * NOT advertised: this backend caches geometry internally (geo_cache.c) but
 * exposes no public nu_save_geometry_cache / nu_get_meshlet_stats entry point,
 * so the bits stay clear per the capability-ABI contract. */
static uint64_t static_backend_capabilities(void)
{
    return
        NU_CAP_RENDER_RASTER |
        NU_CAP_AOV_COLOR |
        NU_CAP_LOAD_USD_FILE |
        NU_CAP_LOAD_USD_HANDLE |
        NU_CAP_LOAD_USD_HANDLE_WITH_DIR |
        NU_CAP_SCENE_BOUNDS |
        NU_CAP_MESH_PATHS |
        NU_CAP_CURVES |
        NU_CAP_CAMERAS |
        NU_CAP_ANALYTIC_LIGHTS |
        NU_CAP_DOME_LIGHT |
        NU_CAP_SET_TRANSFORMS |
        NU_CAP_SET_COLORS |
        NU_CAP_SET_VISIBILITY |
        NU_CAP_SET_CURRENT_TIME |
        NU_CAP_TIMINGS |
        NU_CAP_GPU_MEMORY |
        NU_CAP_COMMAND_CACHE_STATS |
        NU_CAP_VISIBLE_WINDOW;
}

/* contract-first-ffi: tools/check_renderer_abi.py validates field *text*, not
 * the compiled layout. Pin the real ABI so any change to NuBackendInfo breaks
 * the build here instead of silently shifting bytes under a same-text header. */
_Static_assert(sizeof(NuBackendInfo) == 208,
               "NuBackendInfo layout changed -- re-sync nusd_renderer.h across all backends");

NuResult nu_get_backend_info(NuRenderer* r, NuBackendInfo* out_info)
{
    if (!out_info) return NU_ERROR;

    memset(out_info, 0, sizeof(*out_info));
    out_info->version = NU_BACKEND_INFO_VERSION;
    out_info->struct_size = (uint32_t)sizeof(*out_info);
    out_info->capabilities = static_backend_capabilities();
    write_fixed_string(out_info->backend_name,
                       sizeof(out_info->backend_name), "vulkan");
    write_fixed_string(out_info->backend_version,
                       sizeof(out_info->backend_version), "Vulkan 1.2");
    write_fixed_string(out_info->renderer_name,
                       sizeof(out_info->renderer_name),
                       "nanousd-vulkan-renderer");

    if (!r) return NU_OK;

    /* MaterialX PBR pipeline is only wired when the instance opted in. */
    if (r->enable_materials) {
        out_info->capabilities |= NU_CAP_MATERIALS |
                                  NU_CAP_TEXTURES |
                                  NU_CAP_MATERIALX;
    }

    /* Hardware ray tracing gates the RT render modes, the RT-only AOVs
     * (depth / normal / prim-id come from the tiled raygen path), and the
     * async path (nu_render_async drives nu_render_tiled in NU_RENDER_RT). */
    if (r->enable_rt && gpu_rt_available(r->gpu)) {
        out_info->capabilities |= NU_CAP_RENDER_RAYTRACE |
                                  NU_CAP_RENDER_TILED_RAYTRACE |
                                  NU_CAP_AOV_DEPTH |
                                  NU_CAP_AOV_NORMAL |
                                  NU_CAP_AOV_PRIM_ID |
                                  NU_CAP_ASYNC_RENDER;
        if (r->shadow_pipeline) {
            out_info->capabilities |= NU_CAP_RENDER_SHADOW;
        }
    }

    /* CUDA-Vulkan zero-copy interop requires the external-memory/-semaphore
     * extensions; gpu_interop_available reflects what this device negotiated. */
    if (gpu_interop_available(r->gpu)) {
        out_info->capabilities |= NU_CAP_CUDA_INTEROP;
    }

    return NU_OK;
}

NuRenderer* nu_renderer_create(const NuRendererConfig* config)
{
    NuRenderer* r = (NuRenderer*)calloc(1, sizeof(NuRenderer));
    if (!r) return NULL;

    /* Apply config (or defaults) */
    r->width  = config && config->width  > 0 ? config->width  : 1920;
    r->height = config && config->height > 0 ? config->height : 1080;
    r->enable_rt = config ? config->enable_rt : 1;
    r->enable_materials = config ? config->enable_materials : 0;
    if (env_enabled_default_off("NUSD_ENABLE_MATERIALS"))
        r->enable_materials = 1;
    if (env_enabled_default_off("NUSD_DISABLE_MATERIALS"))
        r->enable_materials = 0;
    r->vertex_stride_floats = r->enable_materials ? 12u : 9u;
    r->current_time = NAN;  /* default: use authored default time */
    r->scene_up_axis = 1;
    /* Default dome color: near-white at full intensity (matches Newton's
     * 0xEEEEEE flat clear when no scene-authored DomeLight is plumbed).
     * Override via nu_set_dome_color(). */
    r->dome_color[0] = 0.93f;
    r->dome_color[1] = 0.93f;
    r->dome_color[2] = 0.93f;
    r->dome_color[3] = 1.0f;
    r->dome_color_dirty = 0;
    r->rt_ibl_fill_scale = env_float_or("NUSD_RT_IBL_FILL_SCALE", 1.0f);
    if (!isfinite(r->rt_ibl_fill_scale) || r->rt_ibl_fill_scale < 0.0f)
        r->rt_ibl_fill_scale = 1.0f;
    int visible = config ? config->visible : 0;
    int true_headless =
        !visible &&
        (!getenv("DISPLAY") || getenv("DISPLAY")[0] == '\0') &&
        (!getenv("WAYLAND_DISPLAY") || getenv("WAYLAND_DISPLAY")[0] == '\0');
    const char* force_headless = getenv("NUSD_TRUE_HEADLESS");
    if (force_headless && force_headless[0] && force_headless[0] != '0')
        true_headless = !visible;

    /* Init GLFW. Window visibility controls whether we present to the
     * swapchain (interactive viewer) or run headless (offscreen RT). */
    if (!true_headless) {
        if (!acquire_glfw()) {
            set_error(r, "glfwInit failed");
            free(r);
            return NULL;
        }
        r->glfw_acquired = 1;

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
        /* RESIZABLE must be GLFW_TRUE even for hidden windows. nanousdview
         * embeds the renderer's offscreen output into a Qt widget and calls
         * nu_set_render_size as the user resizes the host window. With
         * GLFW_RESIZABLE=FALSE on a hidden GLFW window, glfwSetWindowSize
         * is silently a no-op on most platforms, the Vulkan surface caps
         * stay clamped to the original extent, and gpu_resize ends up
         * recreating a swapchain at the OLD size — leaving rt_image
         * (sized from swapchain_extent) clipping the rendered region. */
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        r->window = glfwCreateWindow(r->width, r->height,
                                      "nusd_renderer", NULL, NULL);
        if (!r->window) {
            set_error(r, "glfwCreateWindow failed");
            release_glfw();
            free(r);
            return NULL;
        }
    }

    /* Init GPU */
    r->gpu = gpu_init(r->window, r->width, r->height);
    if (!r->gpu) {
        set_error(r, "gpu_init failed");
        if (r->window) {
            glfwDestroyWindow(r->window);
            r->window = NULL;
        }
        if (r->glfw_acquired) {
            release_glfw();
            r->glfw_acquired = 0;
        }
        free(r);
        return NULL;
    }

    /* Headless skips acquire/present and renders to swapchain image 0. */
    gpu_set_headless(r->gpu, visible ? 0 : 1);
    renderer_apply_exposure_desc(r, NULL);

    gpu_overlay_init(r->gpu);

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
        GpuVertexAttrib attribs[2] = {
            { .location = 0, .offset = 0,  .format = GPU_FORMAT_FLOAT3 },
            { .location = 1, .offset = 12, .format = GPU_FORMAT_FLOAT3 },
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
        pipe_desc.nattribs           = 2;

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
    }
    free(vert_spv);
    free(frag_spv);

    r->curve_pipeline = renderer_create_curve_pipeline(r);
    if (r->curve_pipeline) {
        fprintf(stderr, "nusd_renderer: raster BasisCurves pipeline ready\n");
    }

    /* Allocate readback buffer */
    r->readback_pixels = (uint8_t*)malloc((size_t)r->width * r->height * 4);

    /* VK3DGRT splat-scene substate. Defaults are baked in by gs_scene_create
     * (proxy=icosahedron, color=linear, camera=pinhole, K=16, maxPasses=16,
     * minTransmittance=0.03, isoOpacityThreshold=0.5). Particle count is 0
     * until nu_gs_set_particles is called. */
    r->gs = gs_scene_create();
    r->gs_dirty = 0;
    r->gs_pipeline_built = 0;

    fprintf(stderr, "nusd_renderer: created (%dx%d, RT=%s)\n",
            r->width, r->height,
            gpu_rt_available(r->gpu) ? "yes" : "no");

    /* Phase B: NU_DEFERRED_SHADE=1 opts into the deferred-shading compute
     * path on creation. The Python adapter can also call nu_set_deferred_shade
     * later. Keep the var name short — the IsaacLab launchers set it as a
     * single token in the env block. */
    const char* defer_env = getenv("NU_DEFERRED_SHADE");
    if (defer_env && defer_env[0] == '1') {
        r->deferred_shade_enabled = 1;
        fprintf(stderr, "nusd_renderer: deferred-shade ENABLED via NU_DEFERRED_SHADE=1\n");
    }

    /* Phase C.2/C.3: NU_DEFERRED_DEBUG_MODE picks the compute-shader debug
     * mode (0 = Phase C.1 base color, 1 = normals, 2 = pbr-packed,
     * 3 = full IBL-lit). Same opt-in pattern as NU_DEFERRED_SHADE. */
    const char* mode_env = getenv("NU_DEFERRED_DEBUG_MODE");
    if (mode_env && mode_env[0]) {
        long m = strtol(mode_env, NULL, 10);
        if (m >= 0 && m <= 3) {
            r->deferred_debug_mode = (uint32_t)m;
            fprintf(stderr,
                    "nusd_renderer: deferred-debug mode=%ld via NU_DEFERRED_DEBUG_MODE\n",
                    m);
        }
    }

    /* Diagnostic linear-radiance probe (light-unit calibration vs ovrtx). */
    const char* probe_env = getenv("NU_DEFERRED_LINEAR_PROBE");
    if (probe_env && probe_env[0]) {
        double g = strtod(probe_env, NULL);
        if (g > 0.0) {
            r->deferred_linear_probe_gain = (float)g;
            fprintf(stderr,
                    "nusd_renderer: deferred LINEAR-radiance probe gain=%g via "
                    "NU_DEFERRED_LINEAR_PROBE (writes pre-tonemap radiance)\n", g);
        }
    }

    return r;
}

void nu_renderer_destroy(NuRenderer* r)
{
    if (!r) return;

    free(r->readback_pixels);
    free(r->cpu_vertices);
    free(r->cpu_indices);

    for (int i = 0; i < r->nmeshes; i++) {
        free(r->meshes[i].name);
        renderer_mesh_clear_ptex(&r->meshes[i]);
    }
    free(r->meshes);
    free(r->mesh_to_env);
    renderer_free_material_paths(r);

    /* Phase B runtime texture pixels — owned by the renderer until destroy. */
    for (int i = 0; i < r->runtime_textures_count; i++)
        free(r->runtime_textures[i].pixels);
    free(r->runtime_textures);

    free(r->curves.segments);
    free(r->curves.colors);
    renderer_raster_curves_clear(r);

    if (r->usd_scene) scene_free(r->usd_scene);

    /* VK3DGRT splat-scene state. GPU resources (BLAS/TLAS, SSBOs, pipeline)
     * are torn down by gpu_shutdown a few lines below; this just frees the
     * CPU-side particle arrays. */
    if (r->gs) gs_scene_destroy(r->gs);

    if (r->raster_pipeline) gpu_destroy_pipeline(r->gpu, r->raster_pipeline);
    if (r->raster_pipeline_textured) gpu_destroy_pipeline(r->gpu, r->raster_pipeline_textured);
    if (r->raster_pipeline_instanced) gpu_destroy_pipeline(r->gpu, r->raster_pipeline_instanced);
    if (r->pi_instance_buffer) gpu_destroy_buffer(r->gpu, r->pi_instance_buffer);
    if (r->curve_pipeline) gpu_destroy_pipeline(r->gpu, r->curve_pipeline);
    if (r->shadow_pipeline) gpu_destroy_pipeline(r->gpu, r->shadow_pipeline);
    if (r->vertex_buffer) gpu_destroy_buffer(r->gpu, r->vertex_buffer);
    if (r->index_buffer) gpu_destroy_buffer(r->gpu, r->index_buffer);

    gpu_destroy_rt_scene(r->gpu);
    gpu_shutdown(r->gpu);

    if (r->window) {
        glfwDestroyWindow(r->window);
        r->window = NULL;
    }
    if (r->glfw_acquired) {
        release_glfw();
        r->glfw_acquired = 0;
    }

    free(r);
}

/* ---- Scene building ---- */

int nu_add_mesh(NuRenderer* r, const NuMeshDesc* desc)
{
    if (!r || !desc || !desc->positions || !desc->indices)
        return NU_ERROR;
    if (desc->nvertices <= 0 || desc->nindices <= 0)
        return NU_ERROR;
    if (!check_scene_mutable(r, "nu_add_mesh")) return NU_ERROR;
    int suppress_meshopt = r->suppress_next_meshopt;
    int suppress_geo_dedup = r->suppress_next_geo_dedup;
    r->suppress_next_meshopt = 0;
    r->suppress_next_geo_dedup = 0;

    /* Content-hash dedup fast path: when NUSD_HASH_DEDUP=1, look up the
     * geometry hash. If a prior mesh with byte-identical positions /
     * indices / normals / texcoords is already on the GPU, share its
     * vertex_offset + index_offset + BLAS (via prototype_idx) and skip the
     * multi-MB memcpy — REGARDLESS of material. Material is applied
     * per-instance downstream: the RT path reads the per-slot
     * MeshData.material_id (gpu_build_rt_scene writes one SSBO entry per mesh
     * slot, so a dedup'd slot keeps its own material/tex while sharing
     * geometry), and the raster path takes its material id from the per-draw
     * push constant (GpuMeshPushConstants.eye_pos.w, int bits). The legacy
     * per-vertex matID at v[mat_offset] is dead for both shaders, so a
     * cross-material share is bit-identical. */
    if (!suppress_geo_dedup && geo_dedup_enabled()) {
        int proto_slot = geo_dedup_lookup(r, desc);
        if (proto_slot >= 0 && proto_slot < r->nmeshes &&
            r->meshes[proto_slot].active) {
            int slot = append_mesh_slot(r);
            if (slot < 0) return NU_ERROR;
            RendererMesh* p = &r->meshes[proto_slot];
            RendererMesh* m = &r->meshes[slot];
            memset(m, 0, sizeof(RendererMesh));
            m->active        = 1;
            m->visible       = 1;
            m->env_mask      = 0xFF;
            m->nvertices     = p->nvertices;
            m->nindices      = p->nindices;
            m->vertex_offset = p->vertex_offset;
            m->index_offset  = p->index_offset;
            m->prototype_idx = p->prototype_idx;     /* share BLAS */
            m->material_index = desc->material_index;
            m->geo_hash      = p->geo_hash;          /* same geometry as proto */
            m->ptex_color_offset = RENDERER_PTEX_COLOR_NONE;
            if (desc->display_color[0] != 0.0f ||
                desc->display_color[1] != 0.0f ||
                desc->display_color[2] != 0.0f) {
                m->color[0] = desc->display_color[0];
                m->color[1] = desc->display_color[1];
                m->color[2] = desc->display_color[2];
            } else {
                m->color[0] = m->color[1] = m->color[2] = 0.7f;
            }
            if (desc->transform) {
                memcpy(m->transform, desc->transform, 16 * sizeof(float));
            } else {
                identity_matrix(m->transform);
            }
            m->bounds_valid = p->bounds_valid;
            if (m->bounds_valid) {
                memcpy(m->local_bounds_min, p->local_bounds_min, sizeof(m->local_bounds_min));
                memcpy(m->local_bounds_max, p->local_bounds_max, sizeof(m->local_bounds_max));
                mesh_compute_world_bounds(m);
            }
            if (desc->name) m->name = renderer_strdup(desc->name);
            r->scene_dirty = 1;
            return slot;
        }
    }

    /* Optional measurement only: record geometry hash so we know what
     * fraction of nu_add_mesh calls feed in already-uploaded content.
     * No behavior change; gated by NUSD_HASH_CENSUS=1. (Dedup mode
     * records its own hit/slot via geo_dedup_insert below.) */
    geo_census_record(desc);

    /* Append instead of scanning for inactive holes. Large USD loads add
     * hundreds of thousands of meshes/instances; a per-add free-slot scan
     * makes attach O(N^2). nu_clear_scene resets nmeshes back to zero. */
    int slot = append_mesh_slot(r);
    if (slot < 0) return NU_ERROR;

    RendererMesh* m = &r->meshes[slot];
    memset(m, 0, sizeof(RendererMesh));
    m->active = 1;
    m->visible = 1;
    m->env_mask = 0xFF;  /* default: visible to all rays — Phase A overrides per-instance */
    m->nvertices = desc->nvertices;
    m->nindices = desc->nindices;
    m->prototype_idx = slot;
    m->material_index = desc->material_index;
    m->ptex_color_offset = RENDERER_PTEX_COLOR_NONE;
    /* BLAS-cache content key — only hashed when the cache is armed. */
    m->geo_hash = geo_hash_wanted() ? blas_hash_compute(desc) : 0;

    if (desc->transform) {
        memcpy(m->transform, desc->transform, 16 * sizeof(float));
    } else {
        identity_matrix(m->transform);
    }
    mesh_compute_local_bounds(m, desc->positions, desc->nvertices);

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
        m->name = renderer_strdup(desc->name);
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

        /* Vertex cache optimization on the prototype's local indices.
         * Lossless reorder of the triangle list to reduce post-T&L cache
         * misses on the raster path. Indices stay local (0..nvertices-1);
         * baseVertex applies the offset at draw time.
         *
         * meshopt_optimizeVertexCache has hard preconditions: a triangle-
         * list index buffer (nindices a positive multiple of 3) and every
         * index in [0, nvertices). It indexes a vertex-score table of
         * exactly nvertices entries — an out-of-range index walks that
         * table out of bounds and segfaults. Malformed prototypes do occur
         * at DSX scale (non-triangle counts, stale indices). Validate
         * first; the optimization is purely a perf win, so skipping it for
         * a bad mesh is always safe. */
        {
            int meshopt_safe = (desc->nvertices > 0 &&
                                desc->nindices >= 3 &&
                                desc->nindices % 3 == 0);
            if (meshopt_safe) {
                const uint32_t* idx = r->cpu_indices + old_total;
                uint32_t nv = (uint32_t)desc->nvertices;
                for (int j = 0; j < desc->nindices; j++) {
                    if (idx[j] >= nv) { meshopt_safe = 0; break; }
                }
            }
            if (meshopt_safe && !suppress_meshopt && !geo_dedup_enabled()) {
                meshopt_optimizeVertexCache(r->cpu_indices + old_total,
                                            r->cpu_indices + old_total,
                                            (size_t)desc->nindices,
                                            (size_t)desc->nvertices);
            }
        }

        m->index_offset = old_total;
        r->total_indices = new_total;
    }

    r->scene_dirty = 1;

    /* Register this freshly-uploaded geometry as a dedup prototype so
     * future nu_add_mesh calls with byte-identical content can share it.
     * Only runs when dedup mode is enabled (otherwise the census path
     * already inserted with slot=-1 above and this is a no-op). */
    if (!suppress_geo_dedup && geo_dedup_enabled()) {
        geo_dedup_insert(r, desc, slot);
    }

    return slot;
}

int nu_add_mesh_instance(NuRenderer* r, int prototype_mesh_id, const float transform[16])
{
    if (!r) return NU_ERROR;
    if (prototype_mesh_id < 0 || prototype_mesh_id >= r->nmeshes)
        return NU_ERROR_BAD_ID;
    if (!r->meshes[prototype_mesh_id].active)
        return NU_ERROR_BAD_ID;
    if (!check_scene_mutable(r, "nu_add_mesh_instance")) return NU_ERROR;

    /* Resolve to the root prototype (in case prototype_mesh_id is itself an instance) */
    int root_proto = r->meshes[prototype_mesh_id].prototype_idx;

    int slot = append_mesh_slot(r);
    if (slot < 0) return NU_ERROR;

    /* append_mesh_slot() may realloc r->meshes — derive proto AND m from
     * the (possibly moved) base afterward so we never read a freed slot.
     * At full-DSX scale the array crosses a doubling boundary mid-attach
     * and the realloc moves it; a proto captured earlier would dangle. */
    RendererMesh* proto = &r->meshes[prototype_mesh_id];
    RendererMesh* m = &r->meshes[slot];
    memset(m, 0, sizeof(RendererMesh));
    m->active = 1;
    m->visible = 1;
    m->env_mask = 0xFF;  /* default: visible to all rays — Phase A overrides per-instance */

    /* Share geometry from the prototype — no vertex/index data copy */
    m->nvertices    = proto->nvertices;
    m->nindices     = proto->nindices;
    m->vertex_offset = proto->vertex_offset;
    m->index_offset  = proto->index_offset;
    m->prototype_idx = root_proto;
    m->material_index = proto->material_index;
    m->ptex_tri_colors = proto->ptex_tri_colors;
    m->ptex_tri_color_count = proto->ptex_tri_color_count;
    m->ptex_tri_colors_owned = 0;
    m->ptex_color_offset = RENDERER_PTEX_COLOR_NONE;

    /* Copy display color from prototype */
    m->color[0] = proto->color[0];
    m->color[1] = proto->color[1];
    m->color[2] = proto->color[2];
    if (proto->name) m->name = renderer_strdup(proto->name);

    /* Set instance transform */
    if (transform) {
        memcpy(m->transform, transform, 16 * sizeof(float));
    } else {
        identity_matrix(m->transform);
    }
    m->bounds_valid = proto->bounds_valid;
    if (m->bounds_valid) {
        memcpy(m->local_bounds_min, proto->local_bounds_min, sizeof(m->local_bounds_min));
        memcpy(m->local_bounds_max, proto->local_bounds_max, sizeof(m->local_bounds_max));
        mesh_compute_world_bounds(m);
    }

    /* Mark scene dirty — needs TLAS rebuild but NOT new BLAS */
    r->scene_dirty = 1;

    return slot;
}

NuResult nu_set_env_partition(NuRenderer* r, const int* mesh_to_env, int count, int num_envs)
{
    if (!r) return NU_ERROR;
    /* Free any existing partition. */
    free(r->mesh_to_env);
    r->mesh_to_env = NULL;
    r->num_envs = 0;
    if (num_envs <= 0 || count <= 0 || !mesh_to_env) {
        return NU_OK;
    }
    r->mesh_to_env = (int*)malloc(sizeof(int) * count);
    if (!r->mesh_to_env) return NU_ERROR;
    memcpy(r->mesh_to_env, mesh_to_env, sizeof(int) * count);
    r->num_envs = num_envs;

    /* Phase C: store the partition on the GPU side. The next build_accel
     * call will build the per-env TLAS array. */
    gpu_set_env_partition(r->gpu, mesh_to_env, count, num_envs);

    /* Also mirror into the legacy 8-bucket cullMask path so that until the
     * partitioned shader path is bound, the existing single-TLAS path still
     * isolates per tile (env_idx % 8 → mask bit). With env_spacing=3.0 the
     * 8-bucket aliases are >24 units apart — outside any wrist FOV. */
    {
        unsigned char* masks = (unsigned char*)malloc(count);
        if (masks) {
            for (int i = 0; i < count; i++) {
                int env = mesh_to_env[i];
                masks[i] = (env < 0) ? 0xFF : (unsigned char)(1u << (((unsigned int)env) & 7u));
            }
            gpu_set_instance_masks(r->gpu, masks, (uint32_t)count);
            free(masks);
        }
    }

    /* Mark accel dirty; next render rebuilds with the new mask layout. */
    r->accel_dirty = 1;
    r->tlas_dirty = 1;
    return NU_OK;
}

NuResult nu_set_instance_masks(NuRenderer* r, const unsigned char* masks, int count)
{
    if (!r) return NU_ERROR;
    if (count == 0 || !masks) {
        for (int i = 0; i < r->nmeshes; i++) r->meshes[i].env_mask = 0xFF;
        gpu_set_instance_masks(r->gpu, NULL, 0);
        r->accel_dirty = 1; r->tlas_dirty = 1;
        return NU_OK;
    }
    if (count < 0) return NU_ERROR;
    int n = count > r->nmeshes ? r->nmeshes : count;
    for (int i = 0; i < n; i++) r->meshes[i].env_mask = masks[i];
    /* Push to GPU state; the next TLAS update consumes it. */
    gpu_set_instance_masks(r->gpu, masks, (uint32_t)count);
    r->accel_dirty = 1; r->tlas_dirty = 1;
    return NU_OK;
}


NuResult nu_set_mesh_color(NuRenderer* r, int mesh_id, const float color[3])
{
    if (!r || !color) return NU_ERROR;
    if (mesh_id < 0 || mesh_id >= r->nmeshes) return NU_ERROR_BAD_ID;
    if (!r->meshes[mesh_id].active) return NU_ERROR_BAD_ID;
    if (!check_scene_mutable(r, "nu_set_mesh_color")) return NU_ERROR;
    r->meshes[mesh_id].color[0] = color[0];
    r->meshes[mesh_id].color[1] = color[1];
    r->meshes[mesh_id].color[2] = color[2];
    /* SceneData SSBO holds per-mesh color and is rebuilt when scene_dirty;
     * mark it dirty so the next render picks up the new color. */
    r->scene_dirty = 1;
    return NU_OK;
}

NuResult nu_remove_mesh(NuRenderer* r, int mesh_id)
{
    if (!r || mesh_id < 0 || mesh_id >= r->nmeshes)
        return NU_ERROR_BAD_ID;
    if (!r->meshes[mesh_id].active)
        return NU_ERROR_BAD_ID;
    if (!check_scene_mutable(r, "nu_remove_mesh")) return NU_ERROR;

    free(r->meshes[mesh_id].name);
    r->meshes[mesh_id].name = NULL;
    renderer_mesh_clear_ptex(&r->meshes[mesh_id]);
    r->meshes[mesh_id].active = 0;
    r->scene_dirty = 1;
    r->accel_dirty = 1;  /* BLAS/TLAS must be rebuilt without this mesh */

    return NU_OK;
}

void nu_clear_scene(NuRenderer* r)
{
    if (!r) return;

    for (int i = 0; i < r->nmeshes; i++) {
        free(r->meshes[i].name);
        renderer_mesh_clear_ptex(&r->meshes[i]);
        r->meshes[i].active = 0;
    }
    r->nmeshes = 0;
    free(r->pi_batches);    r->pi_batches = NULL;    r->npi_batches = 0;
    free(r->pi_transforms); r->pi_transforms = NULL; r->npi_transforms = 0;
    if (r->pi_instance_buffer) {
        gpu_destroy_buffer(r->gpu, r->pi_instance_buffer);
        r->pi_instance_buffer = NULL;
        r->pi_instance_count = 0;
    }
    renderer_free_material_paths(r);

    free(r->cpu_vertices); r->cpu_vertices = NULL;
    free(r->cpu_indices);  r->cpu_indices = NULL;
    r->cpu_vertex_capacity = 0;
    r->cpu_index_capacity = 0;
    r->total_vertices = 0;
    r->total_indices = 0;

    if (r->vertex_buffer) { gpu_destroy_buffer(r->gpu, r->vertex_buffer); r->vertex_buffer = NULL; }
    if (r->index_buffer)  { gpu_destroy_buffer(r->gpu, r->index_buffer);  r->index_buffer = NULL; }

    free(r->curves.segments); r->curves.segments = NULL;
    free(r->curves.colors);   r->curves.colors   = NULL;
    r->curves.nseg = 0;
    r->curves_dirty = 0;
    renderer_raster_curves_clear(r);

    gpu_destroy_rt_scene(r->gpu);

    /* VK3DGRT: drop any uploaded splat particles too. nu_clear_scene is the
     * "reset everything" entry point — leaving stale splats from a prior
     * scene would surface as ghost particles after the next nu_load_usd. */
    if (r->gs) {
        gs_scene_clear_particles(r->gs);
        r->gs_dirty = 1;
    }

    r->scene_dirty = 0;
    r->accel_dirty = 0;
    r->has_frame = 0;

    /* Re-arm the mutation API: nu_clear_scene + nu_load_usd is the
     * documented re-load path after nu_finalize_scene. */
    r->scene_finalized = 0;

    /* Drop any pending lazy attachment. If we own the stage, close it
     * here so a clear-without-reload-after-lazy sequence doesn't leak
     * the parsed stage. Borrowed stages (lazy_owns_stage == 0) stay
     * the caller's responsibility — typical pxr_compat shim path. */
    if (r->lazy_pending && r->lazy_owns_stage && r->lazy_stage_pending) {
        nanousd_close((NanousdStage)r->lazy_stage_pending);
    }
    r->lazy_pending = 0;
    r->lazy_stage_pending = NULL;
    r->lazy_owns_stage = 0;
    free(r->lazy_mesh_aabbs);
    r->lazy_mesh_aabbs = NULL;
    r->lazy_mesh_aabb_count = 0;
    r->lazy_nprims_snapshot = 0;

    /* Discard any partial geometry-hash census without printing — clears
     * state so a subsequent load starts fresh. */
    if (g_geo_census.entries) {
        free(g_geo_census.entries);
        g_geo_census.entries  = NULL;
        g_geo_census.capacity = 0;
        g_geo_census.count    = 0;
        g_geo_census.total_calls = 0;
        g_geo_census.dup_calls   = 0;
        g_geo_census.hash_collisions = 0;
        g_geo_census.total_bytes = 0;
    }

    if (r->usd_scene) { scene_free(r->usd_scene); r->usd_scene = NULL; }
    r->scene_bounds_valid = 0;
    r->scene_up_axis = 1;
    gpu_set_scene_up_axis(r->gpu, r->scene_up_axis);
    r->fallback_lights_active = 0;
    r->authored_light_count = 0;
}

/* Internal: hand a pre-built Scene to the renderer. Takes ownership of the
 * Scene struct and drives the GPU upload + acceleration-structure build.
 * Both nu_load_usd (file path) and nu_load_usd_from_handle (borrowed stage)
 * funnel into this function. */
static int nu_attach_scene(NuRenderer* r, Scene* scene)
{
    /* Optional per-stage timing. NUSD_LOAD_TIMING=1 prints durations for
     * mesh/light/material upload, curve extraction, etc. Off by default. */
    int do_timing = (getenv("NUSD_LOAD_TIMING") != NULL);
    struct timespec _ts0, _ts1;
    if (do_timing) clock_gettime(CLOCK_MONOTONIC, &_ts0);

    /* Tier 3 step 3 — lazy-scene early branch (see docs/plans/
     * TIER_3_LAZY_MESH.md). scene_load with NUSD_LAZY_MESH=1 returns
     * metadata-only meshes (positions == NULL); nu_attach_scene parks
     * the stage handle for nu_extract_deferred and skips the per-mesh
     * upload loop. Render calls fail-fast on "no geometry" until
     * extract_deferred runs.
     *
     * Detection: nmeshes > 0 AND the first mesh has no positions AND
     * its lazy_prim_idx points back into the stage. Trip any one and
     * we treat the whole scene as lazy. */
    if (scene && scene->nmeshes > 0
        && scene->meshes[0].positions == NULL
        && scene->meshes[0].lazy_prim_idx >= 0)
    {
        /* Take ownership of the stage so the scene_free below doesn't
         * close it. Defang scene->_owns_stage. */
        r->lazy_stage_pending = scene->_stage;
        r->lazy_owns_stage    = scene->_owns_stage;
        r->lazy_pending       = 1;
        scene->_owns_stage    = 0;
        scene->_stage         = NULL;

        memcpy(r->scene_bounds_min, scene->bounds_min, sizeof(r->scene_bounds_min));
        memcpy(r->scene_bounds_max, scene->bounds_max, sizeof(r->scene_bounds_max));
        r->scene_bounds_valid = 1;
        r->scene_up_axis = (scene->up_axis >= 0 && scene->up_axis <= 2)
                         ? scene->up_axis : 1;
        gpu_set_scene_up_axis(r->gpu, r->scene_up_axis);
        r->fallback_lights_active = scene->has_authored_light ? 0 : 1;
        r->authored_light_count = scene->nlights;

        /* Tier 3 step 4: snapshot the lazy walk's per-mesh world AABBs
         * onto the heap so nu_extract_deferred_visible can frustum-cull
         * after the lazy scene + arena are freed below. Skip meshes
         * whose lazy walk failed to compute a real AABB (sentinel
         * bounds at FLT_MAX): those stay always-visible. */
        free(r->lazy_mesh_aabbs);
        r->lazy_mesh_aabbs = NULL;
        r->lazy_mesh_aabb_count = 0;
        if (scene->nmeshes > 0) {
            r->lazy_mesh_aabbs =
                malloc((size_t)scene->nmeshes * sizeof(*r->lazy_mesh_aabbs));
            if (r->lazy_mesh_aabbs) {
                int dst = 0;
                for (int i = 0; i < scene->nmeshes; i++) {
                    const SceneMesh* m = &scene->meshes[i];
                    if (m->lazy_prim_idx < 0) continue;
                    r->lazy_mesh_aabbs[dst].lazy_prim_idx = m->lazy_prim_idx;
                    memcpy(r->lazy_mesh_aabbs[dst].bounds_min, m->bounds_min, sizeof(float)*3);
                    memcpy(r->lazy_mesh_aabbs[dst].bounds_max, m->bounds_max, sizeof(float)*3);
                    dst++;
                }
                r->lazy_mesh_aabb_count = dst;
            }
        }
        if (r->lazy_stage_pending)
            r->lazy_nprims_snapshot = nanousd_nprims((NanousdStage)r->lazy_stage_pending);

        int nm = scene->nmeshes;
        if (r->usd_scene == scene) r->usd_scene = NULL;
        scene_free(scene);

        fprintf(stderr,
                "nusd_renderer: lazy scene attached (%d metadata meshes, "
                "%d AABBs snapshotted, %d prims in stage) — "
                "call nu_extract_deferred() or nu_extract_deferred_visible() "
                "to materialize geometry.\n",
                nm, r->lazy_mesh_aabb_count, r->lazy_nprims_snapshot);
        return nm;
    }

    r->usd_scene = scene;

    /* Upload materials to GPU before BLAS build (RT descriptor set
     * references mat_ssbo_buf + texture array, which must exist). */
    MaterialCollection* mc = (MaterialCollection*)scene->materials;
    int has_real_materials = (mc && mc->nmaterials > 0);
    renderer_free_material_paths(r);
    if (has_real_materials) {
        r->material_paths = (char**)calloc((size_t)mc->nmaterials, sizeof(char*));
        if (r->material_paths) {
            r->material_path_count = mc->nmaterials;
            for (int i = 0; i < mc->nmaterials; i++) {
                const char* path = mc->materials[i].prim_path[0]
                    ? mc->materials[i].prim_path
                    : mc->materials[i].name;
                r->material_paths[i] = renderer_strdup(path ? path : "");
            }
        }
    }
    if (has_real_materials) {
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
            /* Phase 7c trailing block — forward both lobes.
             *
             * The shader's SSS lobe in raytrace.rchit.glsl now treats a
             * (1,1,1) subsurface_color as "use baseColor as the body
             * tint" (see the eval*Light functions) — that's the
             * Standard-Surface intent when the asset wires
             * subsurface_color through a nodegraph the side-loader
             * can't sample (chess King authors `subsurface_color`
             * connected to base_color). With that fallback it's safe
             * to ship subsurface_* up to the GPU. */
            memcpy(gm->subsurface_color,    mp->subsurface_color,    4 * sizeof(float));
            memcpy(gm->subsurface_radius,   mp->subsurface_radius,   4 * sizeof(float));
            memcpy(gm->transmission_color,  mp->transmission_color,  4 * sizeof(float));
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
            memcpy(gm->mdl_uv_transform, mp->mdl_uv_transform, 4 * sizeof(float));
            gm->v_flip = mp->v_flip;
            gm->roughness_tex_scale = mp->roughness_tex_scale;
            gm->roughness_tex_bias = mp->roughness_tex_bias;
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

        struct timespec _ut0, _ut1;  /* load-time instrumentation */
        clock_gettime(CLOCK_MONOTONIC, &_ut0);
        gpu_upload_materials(r->gpu,
                             gpu_mats, mc->nmaterials,
                             gpu_texs, mc->ntextures);
        clock_gettime(CLOCK_MONOTONIC, &_ut1);
        fprintf(stderr,
                "material: LOAD-SPLIT gpu_upload_materials=%.1f ms "
                "(%d textures -> GPU)\n",
                (_ut1.tv_sec - _ut0.tv_sec) * 1000.0 +
                    (_ut1.tv_nsec - _ut0.tv_nsec) / 1.0e6,
                mc->ntextures);

        free(gpu_mats);
        free(gpu_texs);

    } else {
        /* No materials — still create the descriptor set with a single
         * dummy 1×1 texture so env IBL bindings (slots 5/6/7 share the
         * same descriptor pool) can be wired. Without this,
         * gpu_load_environment refuses to run on material-less scenes
         * (Standard Surface defaults, primitive smoke tests, dome-only
         * wrapper USDAs). gpu_upload_materials synthesizes a placeholder
         * material with use_vertex_color=1 + base_color=white so the
         * raster fragment shader's per-mesh fragColor (USD displayColor)
         * survives unscathed. */
        gpu_upload_materials(r->gpu, NULL, 0, NULL, 0);
    }

    /* Materials uploaded (real or placeholder) → mat_pipeline_layout
     * exists. Create the textured-raster pipeline now (4 vertex attribs:
     * pos/normal/uv/matID, layout includes mat_desc_layout @ set 0).
     * The non-textured raster pipeline created in nu_renderer_create
     * stays as a build-time fallback only — at draw time we always
     * select the textured pipeline when it exists, including for
     * displayColor-only scenes (Kitchen_set, test_cube), which need the
     * placeholder material's use_vertex_color=1 to fall back to
     * fragColor instead of reading garbage from an unbound descriptor
     * set. */
    if (!r->raster_textured_pipeline_attempted) {
        r->raster_textured_pipeline_attempted = 1;
        char vert_path[1024], frag_path[1024];
        snprintf(vert_path, sizeof(vert_path), "%s/mesh.vert.spv", SHADER_DIR);
        snprintf(frag_path, sizeof(frag_path), "%s/%s", SHADER_DIR,
                 gpu_barycentric_available(r->gpu) ? "mesh_bary.frag.spv"
                                                   : "mesh.frag.spv");
        uint32_t vsz = 0, fsz = 0;
        uint32_t* vspv = load_shader_file(vert_path, &vsz);
        uint32_t* fspv = load_shader_file(frag_path, &fsz);
        if (vspv && fspv) {
            uint32_t uv_offset = (r->vertex_stride_floats >= 12u) ? 36u : 24u;
            uint32_t mat_offset = (r->vertex_stride_floats >= 12u) ? 44u : 32u;
            GpuVertexAttrib attribs[4] = {
                { .location = 0, .offset = 0,  .format = GPU_FORMAT_FLOAT3 }, /* pos */
                { .location = 1, .offset = 12, .format = GPU_FORMAT_FLOAT3 }, /* normal */
                { .location = 2, .offset = uv_offset, .format = GPU_FORMAT_FLOAT2 }, /* uv */
                /* matID is packed as uint32 bits via memcpy in
                 * nu_add_mesh; reading it as VK_FORMAT_R32_SFLOAT
                 * reinterprets those bytes as a denormal float
                 * (mat_id=2 → 2.8e-45 → int(2.8e-45+0.5) = 0),
                 * collapsing every raster mesh onto materials[0]. Use
                 * R32_UINT and declare `inMatID` as `uint` in the
                 * vertex shader. */
                { .location = 3, .offset = mat_offset, .format = GPU_FORMAT_UINT }, /* matID */
            };
            GpuPipelineDesc pd;
            memset(&pd, 0, sizeof(pd));
            pd.vert_spv = vspv; pd.vert_size = vsz;
            pd.frag_spv = fspv; pd.frag_size = fsz;
            pd.push_constant_size = (uint32_t)sizeof(GpuMeshPushConstants);
            pd.vertex_stride = r->vertex_stride_floats * sizeof(float);
            pd.attribs = attribs;
            pd.nattribs = 4;
            pd.use_mat_layout = 1;
            r->raster_pipeline_textured = gpu_create_pipeline(r->gpu, &pd);
            if (r->raster_pipeline_textured) {
                int nmat = (mc) ? mc->nmaterials : 0;
                int ntex = (mc) ? mc->ntextures : 0;
                fprintf(stderr,
                    "nusd_renderer: textured-raster pipeline ready "
                    "(%d materials%s, %d textures)\n",
                    nmat, (nmat == 0) ? " (placeholder)" : "", ntex);
            }
        }
        free(vspv); free(fspv);
    }

    /* Instanced raster pipeline: identical fragment shading to the textured
     * pipeline, but a per-instance world matrix arrives via instance-rate
     * vertex attributes (binding 1, locations 4..7) instead of pc.model. Used
     * to draw the compact PointInstancer batches so raster matches RT coverage
     * (Moana ocean/beach/foliage are all instanceable → compact PI). */
    if (!r->raster_instanced_pipeline_attempted) {
        r->raster_instanced_pipeline_attempted = 1;
        char vert_path[1024], frag_path[1024];
        snprintf(vert_path, sizeof(vert_path), "%s/mesh_instanced.vert.spv", SHADER_DIR);
        snprintf(frag_path, sizeof(frag_path), "%s/%s", SHADER_DIR,
                 gpu_barycentric_available(r->gpu) ? "mesh_bary.frag.spv"
                                                   : "mesh.frag.spv");
        uint32_t vsz = 0, fsz = 0;
        uint32_t* vspv = load_shader_file(vert_path, &vsz);
        uint32_t* fspv = load_shader_file(frag_path, &fsz);
        if (vspv && fspv) {
            uint32_t uv_offset = (r->vertex_stride_floats >= 12u) ? 36u : 24u;
            uint32_t mat_offset = (r->vertex_stride_floats >= 12u) ? 44u : 32u;
            GpuVertexAttrib attribs[4] = {
                { .location = 0, .offset = 0,  .format = GPU_FORMAT_FLOAT3 }, /* pos */
                { .location = 1, .offset = 12, .format = GPU_FORMAT_FLOAT3 }, /* normal */
                { .location = 2, .offset = uv_offset, .format = GPU_FORMAT_FLOAT2 }, /* uv */
                { .location = 3, .offset = mat_offset, .format = GPU_FORMAT_UINT }, /* matID */
            };
            /* Per-instance world matrix = 4 vec4 columns (16 floats, stride 64). */
            GpuVertexAttrib inst_attribs[4] = {
                { .location = 4, .offset = 0,  .format = GPU_FORMAT_FLOAT4 },
                { .location = 5, .offset = 16, .format = GPU_FORMAT_FLOAT4 },
                { .location = 6, .offset = 32, .format = GPU_FORMAT_FLOAT4 },
                { .location = 7, .offset = 48, .format = GPU_FORMAT_FLOAT4 },
            };
            GpuPipelineDesc pd;
            memset(&pd, 0, sizeof(pd));
            pd.vert_spv = vspv; pd.vert_size = vsz;
            pd.frag_spv = fspv; pd.frag_size = fsz;
            pd.push_constant_size = (uint32_t)sizeof(GpuMeshPushConstants);
            pd.vertex_stride = r->vertex_stride_floats * sizeof(float);
            pd.attribs = attribs;
            pd.nattribs = 4;
            pd.use_mat_layout = 1;
            pd.instance_stride = 16 * sizeof(float);
            pd.instance_attribs = inst_attribs;
            pd.n_instance_attribs = 4;
            r->raster_pipeline_instanced = gpu_create_pipeline(r->gpu, &pd);
            if (r->raster_pipeline_instanced)
                fprintf(stderr,
                    "nusd_renderer: instanced-raster pipeline ready\n");
            else
                fprintf(stderr,
                    "nusd_renderer: instanced-raster pipeline FAILED to create "
                    "(compact PI batches will not draw in raster)\n");
        } else {
            fprintf(stderr,
                "nusd_renderer: mesh_instanced.vert.spv/mesh.frag.spv not found "
                "— compact PI batches will not draw in raster\n");
        }
        free(vspv); free(fspv);
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

    rss_trace("attach: scene loaded, pre-add_mesh loop");

    /* Add each mesh from the scene. Preserve scene-loader instancing here:
     * prototype geometry is uploaded once via nu_add_mesh(), and scene-mesh
     * instances become renderer instances via nu_add_mesh_instance(). */
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
    if (!reserve_cpu_geometry(r, reserve_vertices, reserve_indices)) {
        free(scene_to_renderer);
        set_error(r, "Failed to reserve CPU geometry buffers");
        return NU_ERROR;
    }

    rss_trace("attach: cpu_geometry reserved");

    for (int i = 0; i < scene->nmeshes; i++) {
        SceneMesh* sm = &scene->meshes[i];
        if (sm->nvertices <= 0 || sm->nindices <= 0) continue;

        /* Convert double[16] world_xform to float[16].
         *
         * sm->world_xform is in nanousd's USD-row-vector convention
         * (translation at [12]/[13]/[14] — same as USD's GfMatrix4d
         * memory layout). The renderer-internal `m->transform` and the
         * downstream usd4x4_to_vk3x4 / raster mvp build expect the
         * column-vector row-major form (translation at [3]/[7]/[11])
         * after commit d4ca8b7 ("USD->Vk transform: stop transposing").
         * That commit fixed IsaacLab's Python producers but didn't
         * convert USD-loaded scene meshes — without the transpose
         * here, every chess piece collapsed to origin and we saw a
         * single mesh stack at the board center. Transpose so the
         * scene-load path matches the now-canonical convention. */
        float xform[16];
        for (int j = 0; j < 16; j++)
            xform[j] = (float)sm->world_xform[j];
        float xform_t[16];
        for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
                xform_t[row * 4 + col] = xform[col * 4 + row];

        int scene_instance = (sm->prototype_idx >= 0 && sm->prototype_idx != i &&
                              sm->prototype_idx < scene->nmeshes &&
                              scene_to_renderer[sm->prototype_idx] >= 0);
        int mesh_id = NU_ERROR;
        if (scene_instance) {
            mesh_id = nu_add_mesh_instance(r, scene_to_renderer[sm->prototype_idx], xform_t);
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
            desc.material_index = has_real_materials ? sm->material_index : 0;
            desc.transform      = xform_t;
            desc.name           = sm->path;

            if (sm->has_display_color) {
                desc.display_color[0] = sm->display_color[0];
                desc.display_color[1] = sm->display_color[1];
                desc.display_color[2] = sm->display_color[2];
            } else {
                desc.display_color[0] = 0.7f;
                desc.display_color[1] = 0.7f;
                desc.display_color[2] = 0.7f;
            }

            if (sm->ptex_tri_colors && sm->ptex_tri_color_count == sm->nindices) {
                r->suppress_next_meshopt = 1;
                r->suppress_next_geo_dedup = 1;
            }
            mesh_id = nu_add_mesh(r, &desc);
        }

        if (mesh_id >= 0) {
            scene_to_renderer[i] = mesh_id;
            RendererMesh* rm = &r->meshes[mesh_id];
            if (!scene_instance)
                renderer_mesh_copy_ptex(rm, sm);
            rm->material_index = has_real_materials ? sm->material_index : 0;
            if (sm->path) {
                free(rm->name);
                rm->name = renderer_strdup(sm->path);
            }
            if (sm->has_display_color) {
                rm->color[0] = sm->display_color[0];
                rm->color[1] = sm->display_color[1];
                rm->color[2] = sm->display_color[2];
            }
            /* PointInstancer prototype subtree geometry is needed as the
             * shared buffer source for instances, but should not draw as a
             * standalone object. */
            if (sm->is_proto_only) {
                rm->visible = 0;
                rm->env_mask = 0;
            } else {
                loaded++;
            }
        }
    }
    /* Compact PointInstancer batches: translate each batch's prototype mesh
     * from scene index to renderer index, then steal the scene-owned batch and
     * transform slabs before scene_free. The move avoids a second 48 B/instance
     * copy, which matters at Moana scale. */
    /* Attach whatever compact batches the scene produced. The scene side
     * compacts native instances / PointInstancers / flat dedup with NO expanded
     * SceneMesh fallback (pi_scene_mesh_clones==0), so these batches are the
     * ONLY representation of that geometry — gating attach on a default-off env
     * flag silently dropped it (empty render). Draw is opt-out via
     * NUSD_RASTER_DISABLE_PI / the RT TLAS path. */
    if (scene->npi_batches > 0 && scene->npi_transforms > 0) {
        free(r->pi_batches);    r->pi_batches = NULL;    r->npi_batches = 0;
        free(r->pi_transforms); r->pi_transforms = NULL; r->npi_transforms = 0;

        SceneInstanceBatch* moved_batches = scene->pi_batches;
        SceneInstanceTransform* moved_xforms = scene->pi_transforms;
        uint64_t moved_xform_count = scene->npi_transforms;
        int kept = 0;
        int dropped = 0;
        for (int b = 0; b < scene->npi_batches; b++) {
            SceneInstanceBatch sb = moved_batches[b];
            int sp = sb.prototype_mesh_idx;
            int rp = (sp >= 0 && sp < scene->nmeshes) ? scene_to_renderer[sp] : -1;
            if (rp < 0 ||
                (uint64_t)sb.transform_offset + sb.transform_count > scene->npi_transforms) {
                dropped++;
                continue;
            }
            sb.prototype_mesh_idx = rp;
            moved_batches[kept++] = sb;
        }

        scene->pi_batches = NULL;
        scene->npi_batches = 0;
        scene->pi_transforms = NULL;
        scene->npi_transforms = 0;

        if (kept > 0) {
            r->pi_batches = moved_batches;
            r->npi_batches = kept;
            r->pi_transforms = moved_xforms;
            r->npi_transforms = moved_xform_count;
            fprintf(stderr,
                    "nusd_renderer: compact PI batches attached — %d batches, "
                    "%llu transforms moved%s%d dropped%s\n",
                    r->npi_batches,
                    (unsigned long long)r->npi_transforms,
                    dropped ? ", " : " (",
                    dropped,
                    dropped ? "" : ")");
        } else {
            free(moved_batches);
            free(moved_xforms);
            if (dropped > 0)
                fprintf(stderr,
                        "nusd_renderer: compact PI batches all dropped (%d) "
                        "because prototype meshes were unavailable\n",
                        dropped);
        }
    }

    free(scene_to_renderer);
    rss_trace("attach: add_mesh loop done");

    /* Build the per-instance world-matrix buffer that the raster path uses to
     * draw the compact PointInstancer batches (RT consumes r->pi_transforms via
     * the TLAS; raster needs the same placements as instance-rate vertex data).
     * Layout per instance: 16 floats = the vk3x4 form (column-vector row-major,
     * translation in 3/7/11 — same layout as a direct mesh's pc.model) padded
     * with a (0,0,0,1) bottom row so the vertex shader reconstructs a mat4. */
    if (r->pi_instance_buffer) {
        gpu_destroy_buffer(r->gpu, r->pi_instance_buffer);
        r->pi_instance_buffer = NULL;
        r->pi_instance_count = 0;
    }
    if (r->npi_batches > 0 && r->pi_transforms && r->npi_transforms > 0) {
        uint64_t n = r->npi_transforms;
        float* inst = (float*)malloc((size_t)n * 16 * sizeof(float));
        if (!inst) {
            fprintf(stderr,
                    "nusd_renderer: failed to allocate raster PI instance buffer "
                    "(%llu instances)\n", (unsigned long long)n);
        } else {
            for (uint64_t i = 0; i < n; i++) {
                float* d = &inst[i * 16];
                scene_instance_transform_to_vk3x4(&r->pi_transforms[i], d);
                d[12] = 0.0f; d[13] = 0.0f; d[14] = 0.0f; d[15] = 1.0f;
            }
            GpuBufferDesc ibd;
            ibd.usage = GPU_BUFFER_VERTEX;
            ibd.size  = (uint64_t)n * 16 * sizeof(float);
            ibd.data  = inst;
            r->pi_instance_buffer = gpu_create_buffer(r->gpu, &ibd);
            free(inst);
            if (r->pi_instance_buffer) {
                r->pi_instance_count = n;
                fprintf(stderr,
                        "nusd_renderer: raster PI instance buffer ready — "
                        "%llu instances (%.1f MiB)\n",
                        (unsigned long long)n,
                        (double)ibd.size / (1024.0 * 1024.0));
            } else {
                fprintf(stderr,
                        "nusd_renderer: failed to create raster PI instance "
                        "buffer GPU buffer (%llu instances)\n",
                        (unsigned long long)n);
            }
        }
    }

    if (!renderer_build_raster_curves(r, scene, mc)) {
        if (r->usd_scene == scene) r->usd_scene = NULL;
        scene_free(scene);
        return NU_ERROR;
    }

    if (do_timing) {
        clock_gettime(CLOCK_MONOTONIC, &_ts1);
        fprintf(stderr, "  [timing] mesh+light+mat upload: %.1f ms\n",
            (double)(_ts1.tv_sec - _ts0.tv_sec) * 1000.0 +
            (double)(_ts1.tv_nsec - _ts0.tv_nsec) * 1e-6);
        _ts0 = _ts1;
    }

    /* ---- Phase 11.A: extract curve segments ----
     * Flat layout across all BasisCurves prims (Strategy A from
     * RENDERER_BIG_PLAN.md Phase 12.1). Per-segment color uses the bound
     * material base color when available, then displayColor, then a warm
     * twig/grass fallback. Phase 12.2 will replace the color array with a
     * material_id SSBO + materials LUT.
     *
     * Phase 12.x: AABBs are no longer authored host-side. The GPU pass
     * in gpu_build_curve_blas computes them from the segment SSBO right
     * before vkCmdBuildAccelerationStructuresKHR. */
    int extract_curve_segments =
        r->enable_rt || env_enabled_default_off("NUSD_RASTER_EXTRACT_CURVE_SEGMENTS");
    if (scene->ncurves > 0 && !extract_curve_segments) {
        fprintf(stderr,
                "nusd_renderer: skipped %d BasisCurves segment extraction "
                "(RT disabled; using Vulkan raster curve pipeline)\n",
                scene->ncurves);
    } else if (scene->ncurves > 0) {
        int total_segs = 0;
        for (int i = 0; i < scene->ncurves; i++)
            total_segs += scene_curve_count_segments(&scene->curves[i]);

        if (total_segs > 0) {
            r->curves.segments = (SceneCurveSegment*)malloc((size_t)total_segs * sizeof(SceneCurveSegment));
            r->curves.colors   = (float*)            malloc((size_t)total_segs * 3 * sizeof(float));
            if (!r->curves.segments || !r->curves.colors) {
                free(r->curves.segments); r->curves.segments = NULL;
                free(r->curves.colors);   r->curves.colors   = NULL;
                fprintf(stderr, "nusd_renderer: failed to allocate curve segment buffers (%d segments)\n", total_segs);
            } else {
                /* Curve segment extraction. Per-curve work is highly
                 * unbalanced (one cubic prim on tera owns ~12 M of the
                 * 34 M segments), so we let each curve in turn use all
                 * cores via the parallel-for inside scene_curve_to_segments.
                 * Going parallel at this outer level instead would cap us
                 * at the dominant prim's wall time.
                 *
                 * The color fill IS parallel (it's trivially data-parallel
                 * over n_this segments). We build per-curve offsets via
                 * prefix sum so writes never overlap.
                 *
                 * Phase 12.x: AABBs are no longer authored host-side;
                 * gpu_build_curve_blas dispatches a compute pass to
                 * generate them from the segment SSBO. */
                int* per_curve_count  = (int*)malloc((size_t)scene->ncurves * sizeof(int));
                int* per_curve_offset = (int*)malloc((size_t)scene->ncurves * sizeof(int));
                if (!per_curve_count || !per_curve_offset) {
                    free(per_curve_count); free(per_curve_offset);
                    free(r->curves.segments); r->curves.segments = NULL;
                    free(r->curves.colors);   r->curves.colors   = NULL;
                    fprintf(stderr, "nusd_renderer: failed to allocate curve segment scaffolding\n");
                } else {
                    int running = 0;
                    for (int i = 0; i < scene->ncurves; i++) {
                        per_curve_count[i]  = scene_curve_count_segments(&scene->curves[i]);
                        per_curve_offset[i] = running;
                        running += per_curve_count[i];
                    }

                    int material_colored_curves = 0;
                    int display_colored_curves = 0;
                    int fallback_colored_curves = 0;
                    int ribbon_curves = 0;
                    long long ribbon_segments = 0;
                    for (int i = 0; i < scene->ncurves; i++) {
                        int n_this = per_curve_count[i];
                        if (n_this == 0) continue;
                        int off = per_curve_offset[i];
                        const SceneCurve* c = &scene->curves[i];
                        scene_curve_to_segments(c,
                            &r->curves.segments[off]);
                        if (c->has_normals) {
                            ribbon_curves++;
                            ribbon_segments += (long long)n_this;
                        }
                        float cr = 0.34f, cg = 0.27f, cb = 0.18f;
                        int color_selected = 0;
                        int colors_written = 0;
                        if (mc && c->material_index >= 0 &&
                            c->material_index < mc->nmaterials) {
                            const SceneMaterial* smat =
                                &mc->materials[c->material_index];
                            const MaterialParams* mp =
                                &smat->params;
                            float mr = fminf(fmaxf(mp->base_color[0], 0.0f), 1.0f);
                            float mg = fminf(fmaxf(mp->base_color[1], 0.0f), 1.0f);
                            float mb = fminf(fmaxf(mp->base_color[2], 0.0f), 1.0f);
                            if (smat->has_ptex_average_color ||
                                (mp->use_vertex_color == 0 &&
                                 !renderer_color_is_debug_placeholder(mr, mg, mb))) {
                                cr = mr; cg = mg; cb = mb;
                                color_selected = 1;
                                material_colored_curves++;
                            }
                        }
                        if (!color_selected) {
                            if (c->has_display_color &&
                                !renderer_color_is_debug_placeholder(c->display_color[0],
                                                                     c->display_color[1],
                                                                     c->display_color[2])) {
                                if (c->colors) {
                                    int n_col = scene_curve_to_segment_colors(
                                        c, &r->curves.colors[off * 3]);
                                    if (n_col == n_this)
                                        colors_written = 1;
                                }
                                if (!colors_written) {
                                    cr = fminf(fmaxf(c->display_color[0], 0.0f), 1.0f);
                                    cg = fminf(fmaxf(c->display_color[1], 0.0f), 1.0f);
                                    cb = fminf(fmaxf(c->display_color[2], 0.0f), 1.0f);
                                }
                                display_colored_curves++;
                            } else {
                                fallback_colored_curves++;
                            }
                        }
                        if (!colors_written) {
                            float* col = &r->curves.colors[off * 3];
                            #ifdef _OPENMP
                            #pragma omp parallel for schedule(static) if(n_this > 4096)
                            #endif
                            for (int s = 0; s < n_this; s++) {
                                col[s*3 + 0] = cr;
                                col[s*3 + 1] = cg;
                                col[s*3 + 2] = cb;
                            }
                        }
                    }

                    int written = running;
                    r->curves.nseg  = written;
                    r->curves_dirty = 1;
                    /* Phase 11.A.2.5: curve presence triggers full RT scene
                     * rebuild so the curve hit-group and TLAS instance get
                     * wired into the pipeline. */
                    r->accel_dirty  = 1;
                    fprintf(stderr,
                            "nusd_renderer: extracted %d curve segments from %d "
                            "BasisCurves (curve colors: material=%d display=%d "
                            "fallback=%d; ribbons=%d curves/%lld segments)\n",
                            written, scene->ncurves, material_colored_curves,
                            display_colored_curves, fallback_colored_curves,
                            ribbon_curves, ribbon_segments);
                    free(per_curve_count);
                    free(per_curve_offset);
                }
            }
            if (do_timing) {
                clock_gettime(CLOCK_MONOTONIC, &_ts1);
                fprintf(stderr, "  [timing] curve segment extraction: %.1f ms\n",
                    (double)(_ts1.tv_sec - _ts0.tv_sec) * 1000.0 +
                    (double)(_ts1.tv_nsec - _ts0.tv_nsec) * 1e-6);
                _ts0 = _ts1;
            }
        }
    }

    /* Auto-frame camera on scene bounds */
    if (loaded > 0 || r->npi_batches > 0) {
        float cx = (scene->bounds_min[0] + scene->bounds_max[0]) * 0.5f;
        float cy = (scene->bounds_min[1] + scene->bounds_max[1]) * 0.5f;
        float cz = (scene->bounds_min[2] + scene->bounds_max[2]) * 0.5f;

        float dx = scene->bounds_max[0] - scene->bounds_min[0];
        float dy = scene->bounds_max[1] - scene->bounds_min[1];
        float dz = scene->bounds_max[2] - scene->bounds_min[2];
        float diag = sqrtf(dx*dx + dy*dy + dz*dz);

        float dist = diag * 1.5f;
        float eye[3] = { cx + dist * 0.5f, cy + dist * 0.3f, cz + dist * 0.5f };
        float target[3] = { cx, cy, cz };

        NuCameraDesc cam;
        memcpy(cam.eye, eye, sizeof(eye));
        memcpy(cam.target, target, sizeof(target));
        cam.fov_degrees = 60.0f;
        cam.near_clip = fmaxf(diag * 0.001f, 0.1f);
        cam.far_clip = fmaxf(diag * 100.0f, 10000.0f);
        nu_set_camera(r, 0, &cam);
    }

    /* Auto-load DomeLight as IBL. A textured DomeLight becomes a tinted HDR
     * environment; a textureless DomeLight becomes a constant-color
     * environment. Both paths produce the same env/irradiance/BRDF resources
     * consumed by raster, RT, deferred, and background passes. */
    if (scene->has_dome_light && scene->dome_hdr_path[0]) {
        /* Surface lighting (rchit) uses env_scale-normalised env (auto-
         * exposure on irradiance) regardless of USD intensity — passing
         * raw intensity directly into surface BRDFs blows out our fixed
         * tone-mapper. The intensity is tracked separately and applied
         * to the visible sky in raytrace.rmiss.glsl, so a USD-authored
         * intensity=1000 saturates the bright sky like ovrtx without
         * making the whole frame white. */
        const char* dome_path = scene->dome_hdr_path;
        char fallback_path[1024];
        int loaded = gpu_load_environment_tinted_intensity(
            r->gpu, dome_path, scene->dome_intensity, scene->dome_color);
        float loaded_intensity = scene->dome_intensity;
        if (!loaded &&
            dome_visible_fallback_path(scene->dome_hdr_path, fallback_path,
                                       sizeof(fallback_path))) {
            float fallback_intensity = visible_dome_fallback_intensity();
            fprintf(stderr,
                    "nusd_renderer: trying visible dome fallback %s "
                    "(internal LDR sky intensity=%.3f)\n",
                    fallback_path, fallback_intensity);
            loaded = gpu_load_environment_tinted_intensity(
                r->gpu, fallback_path, fallback_intensity, scene->dome_color);
            if (loaded) {
                dome_path = fallback_path;
                loaded_intensity = fallback_intensity;
            }
        }
        if (loaded) {
            fprintf(stderr,
                    "nusd_renderer: dome-light IBL → %s (intensity=%.3f → "
                    "rmiss sky multiplier)\n",
                    dome_path, loaded_intensity);
            /* Phase C.3: see comment in nu_load_environment_intensity. If
             * the deferred compute pipeline is already cached, its
             * descriptor-set layout was built without IBL bindings; rebuild
             * on next render. */
            if (r->deferred_pipeline_built) {
                gpu_destroy_deferred_pipeline(r->gpu);
                r->deferred_pipeline_built = 0;
                gpu_invalidate_tiled_cmd_cache(r->gpu);
            }
        } else {
            fprintf(stderr,
                    "nusd_renderer: dome-light load failed for %s — "
                    "falling back to procedural sky\n",
                    scene->dome_hdr_path);
        }
    } else if (scene->has_dome_light) {
        int loaded = gpu_load_flat_environment(r->gpu,
                                               scene->dome_color,
                                               scene->dome_intensity);
        if (loaded) {
            fprintf(stderr,
                    "nusd_renderer: flat DomeLight IBL "
                    "(color=%.3f,%.3f,%.3f intensity=%.3f)\n",
                    scene->dome_color[0],
                    scene->dome_color[1],
                    scene->dome_color[2],
                    scene->dome_intensity);
        } else {
            fprintf(stderr,
                    "nusd_renderer: flat DomeLight IBL creation failed — "
                    "falling back to procedural sky\n");
        }
    }

    memcpy(r->scene_bounds_min, scene->bounds_min, sizeof(r->scene_bounds_min));
    memcpy(r->scene_bounds_max, scene->bounds_max, sizeof(r->scene_bounds_max));
    r->scene_bounds_valid = 1;
    r->scene_up_axis = (scene->up_axis >= 0 && scene->up_axis <= 2)
                     ? scene->up_axis : 1;
    gpu_set_scene_up_axis(r->gpu, r->scene_up_axis);
    r->fallback_lights_active = scene->has_authored_light ? 0 : 1;
    r->authored_light_count = scene->nlights;

    if (!(r->enable_rt && gpu_rt_available(r->gpu))) {
        if (!upload_rt_ptex_triangle_colors(r)) {
            if (r->usd_scene == scene) r->usd_scene = NULL;
            scene_free(scene);
            set_error(r, "Failed to upload real Ptex triangle-corner colors");
            return NU_ERROR;
        }
    }

    if (r->usd_scene == scene) r->usd_scene = NULL;
    rss_trace("attach: pre-scene_free");
    scene_free(scene);
    rss_trace("attach: post-scene_free");

    if (loaded == 0 && r->npi_batches > 0)
        loaded = r->npi_batches;
    fprintf(stderr, "nusd_renderer: loaded %d meshes\n", loaded);

    /* NUSD_AUTO_FINALIZE=1 drops the CPU vertex/index mirror at the end of
     * attach. Off by default — callers that want the memory back call
     * nu_finalize_scene() explicitly. The env-var hook is for benchmarks
     * and the headless DSX test harness that don't go through a viewer. */
    {
        const char* env = getenv("NUSD_AUTO_FINALIZE");
        if (env && env[0] && env[0] != '0') {
            if (nu_finalize_scene(r) != NU_OK) {
                fprintf(stderr,
                        "nusd_renderer: NUSD_AUTO_FINALIZE failed — %s\n",
                        r->last_error);
            }
        }
    }

    return loaded;
}

int nu_load_usd(NuRenderer* r, const char* path)
{
    if (!r || !path) return NU_ERROR;
    nu_clear_scene(r);
    snprintf(r->usd_path, sizeof(r->usd_path), "%s", path);
    {
        NuExposureDesc exp;
        exposure_desc_from_usd_path(path, &exp);
        renderer_apply_exposure_desc(r, &exp);
    }
    scene_set_load_time(r->current_time);
    scene_set_load_materials(r->enable_materials);
    Scene* scene = scene_load(path);
    if (!scene) {
        set_error(r, "scene_load failed");
        return NU_ERROR;
    }
    return nu_attach_scene(r, scene);
}

int nu_load_usd_from_handle(NuRenderer* r, void* stage, const char* stage_label)
{
    return nu_load_usd_from_handle_with_dir(r, stage, stage_label, NULL);
}

int nu_load_usd_from_handle_with_dir(NuRenderer* r, void* stage,
                                     const char* stage_label,
                                     const char* scene_dir)
{
    if (!r || !stage) return NU_ERROR;
    nu_clear_scene(r);
    r->usd_path[0] = '\0';  /* handle-loaded scene has no source file */
    renderer_apply_exposure_desc(r, NULL);
    scene_set_load_time(r->current_time);
    scene_set_load_materials(r->enable_materials);

    /* `scene_dir`, when non-NULL, lets the caller hint the directory
     * the materials_load side-loader should scan for sidecar .mtlx files.
     * scene_load_from_stage derives that via dirname(stage_label), so we
     * synthesise a label whose dirname is exactly scene_dir. The label
     * is otherwise only used in diagnostics. */
    char synthetic_label[1024];
    const char* effective_label = stage_label;
    if (scene_dir && *scene_dir) {
        const char* base = stage_label && *stage_label ? stage_label : "<borrowed-stage>";
        size_t dlen = strlen(scene_dir);
        size_t blen = strlen(base);
        if (dlen + 1 + blen + 1 < sizeof(synthetic_label)) {
            memcpy(synthetic_label, scene_dir, dlen);
            synthetic_label[dlen] = '/';
            memcpy(synthetic_label + dlen + 1, base, blen);
            synthetic_label[dlen + 1 + blen] = '\0';
            effective_label = synthetic_label;
        }
    }

    Scene* scene = scene_load_from_stage(stage, effective_label);
    if (!scene) {
        set_error(r, "scene_load_from_stage failed");
        return NU_ERROR;
    }
    /* scene borrows the stage; the caller (typically Python pxr_compat)
     * is responsible for keeping it alive while the renderer holds it. */
    return nu_attach_scene(r, scene);
}

NuResult nu_extract_deferred(NuRenderer* r)
{
    if (!r) return NU_ERROR;
    if (!r->lazy_pending) return NU_OK;  /* idempotent / no-op for eager loads */
    if (!r->lazy_stage_pending) {
        set_error(r, "nu_extract_deferred: no lazy stage pinned");
        r->lazy_pending = 0;
        return NU_ERROR;
    }

    /* Re-run scene_load_from_stage on the same stage with NUSD_LAZY_MESH
     * temporarily disabled. scene.c's lazy path is gated on the env var,
     * so unsetting it forces the eager extraction. The original lazy
     * scene's metadata was freed in nu_attach_scene; the stage is
     * untouched and ready to drive a second extraction pass. */
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

    void* stage    = r->lazy_stage_pending;
    int owns_stage = r->lazy_owns_stage;

    fprintf(stderr,
            "nusd_renderer: nu_extract_deferred — running eager scene_load_from_stage "
            "on pinned lazy stage (owns_stage=%d).\n", owns_stage);

    const char* extract_label = r->usd_path[0] ? r->usd_path : "<lazy-extract>";
    Scene* eager = scene_load_from_stage(stage, extract_label);
    if (!eager) {
        set_error(r, "nu_extract_deferred: eager scene_load_from_stage failed");
        if (had_prev) setenv("NUSD_LAZY_MESH", prev_env, 1);
        else          unsetenv("NUSD_LAZY_MESH");
        return NU_ERROR;
    }
    /* Hand stage ownership to the eager scene so the normal attach path's
     * scene_free at the end either closes it (owned) or leaves it to the
     * caller (borrowed) — matching the original load's semantics. */
    eager->_owns_stage = owns_stage;

    /* Clear lazy state BEFORE recursing so the inner nu_attach_scene
     * doesn't bounce back into the lazy branch. */
    r->lazy_pending       = 0;
    r->lazy_stage_pending = NULL;
    r->lazy_owns_stage    = 0;
    free(r->lazy_mesh_aabbs);
    r->lazy_mesh_aabbs = NULL;
    r->lazy_mesh_aabb_count = 0;
    r->lazy_nprims_snapshot = 0;

    int rc = nu_attach_scene(r, eager);

    /* Restore prior env-var state. */
    if (had_prev) setenv("NUSD_LAZY_MESH", prev_env, 1);
    else          unsetenv("NUSD_LAZY_MESH");

    return rc >= 0 ? NU_OK : NU_ERROR;
}

/* Extract 6 frustum planes from a row-major 4x4 view-projection matrix
 * using the Gribb–Hartmann sum/diff trick. Output planes are
 * (nx, ny, nz, d) where the plane equation is nx*x + ny*y + nz*z + d = 0
 * and a point is *inside* the frustum iff nx*x + ny*y + nz*z + d >= 0.
 * (Normals point inward.) Planes are normalised. */
static void extract_frustum_planes(const float vp[16], float planes[6][4])
{
    /* Row-major: vp[row*4 + col]. */
    #define ROW(r,c) vp[(r)*4 + (c)]
    /* left   = row3 + row0
     * right  = row3 - row0
     * bottom = row3 + row1
     * top    = row3 - row1
     * near   = row3 + row2
     * far    = row3 - row2 */
    float p[6][4];
    for (int c = 0; c < 4; c++) p[0][c] = ROW(3,c) + ROW(0,c);  /* left */
    for (int c = 0; c < 4; c++) p[1][c] = ROW(3,c) - ROW(0,c);  /* right */
    for (int c = 0; c < 4; c++) p[2][c] = ROW(3,c) + ROW(1,c);  /* bottom */
    for (int c = 0; c < 4; c++) p[3][c] = ROW(3,c) - ROW(1,c);  /* top */
    for (int c = 0; c < 4; c++) p[4][c] = ROW(3,c) + ROW(2,c);  /* near */
    for (int c = 0; c < 4; c++) p[5][c] = ROW(3,c) - ROW(2,c);  /* far */
    #undef ROW
    for (int i = 0; i < 6; i++) {
        float nx = p[i][0], ny = p[i][1], nz = p[i][2], d = p[i][3];
        float inv_len = 1.0f / sqrtf(nx*nx + ny*ny + nz*nz);
        planes[i][0] = nx * inv_len;
        planes[i][1] = ny * inv_len;
        planes[i][2] = nz * inv_len;
        planes[i][3] = d  * inv_len;
    }
}

/* Returns 1 if the AABB (mn,mx) is at least partially inside the frustum
 * defined by 6 planes (normals inward). Uses the p-vertex / n-vertex
 * trick: for each plane, compute the farthest-along-normal AABB corner
 * and test only that point. If even one plane rejects, the AABB is
 * fully outside. */
static int aabb_in_frustum(const float planes[6][4],
                           const float mn[3], const float mx[3])
{
    for (int i = 0; i < 6; i++) {
        const float* pl = planes[i];
        float px = (pl[0] >= 0) ? mx[0] : mn[0];
        float py = (pl[1] >= 0) ? mx[1] : mn[1];
        float pz = (pl[2] >= 0) ? mx[2] : mn[2];
        if (pl[0]*px + pl[1]*py + pl[2]*pz + pl[3] < 0.0f)
            return 0;   /* AABB fully outside this plane */
    }
    return 1;
}

NuResult nu_extract_deferred_visible(NuRenderer* r,
                                     const float* vp_matrices,
                                     int num_cameras)
{
    if (!r) return NU_ERROR;
    if (!r->lazy_pending) return NU_OK;  /* idempotent; no-op for eager loads */
    if (!r->lazy_stage_pending) {
        set_error(r, "nu_extract_deferred_visible: no lazy stage pinned");
        r->lazy_pending = 0;
        return NU_ERROR;
    }

    /* Frustum cull only kicks in when caller supplied cameras AND the lazy
     * walk snapshotted AABBs AND we know nprims. Otherwise fall back to
     * "extract everything" — matches nu_extract_deferred semantics and
     * is the safe default for IsaacLab which calls extract before any
     * camera is registered. */
    int can_cull = (vp_matrices && num_cameras > 0
                    && r->lazy_mesh_aabbs && r->lazy_mesh_aabb_count > 0
                    && r->lazy_nprims_snapshot > 0);
    if (!can_cull) {
        fprintf(stderr,
                "nusd_renderer: nu_extract_deferred_visible — no cull "
                "(cameras=%d, aabbs=%d, nprims=%d); falling back to "
                "extract-all.\n",
                num_cameras, r->lazy_mesh_aabb_count, r->lazy_nprims_snapshot);
        return nu_extract_deferred(r);
    }

    /* Build per-camera frustum planes. */
    float (*planes)[6][4] = (float (*)[6][4])
        malloc((size_t)num_cameras * 6 * 4 * sizeof(float));
    if (!planes) {
        set_error(r, "nu_extract_deferred_visible: alloc planes failed");
        return NU_ERROR;
    }
    for (int c = 0; c < num_cameras; c++)
        extract_frustum_planes(&vp_matrices[c * 16], planes[c]);

    /* Walk the lazy AABB snapshot, mark wanted prim indices in a byte
     * bitmap sized at nprims_snapshot. Index out-of-range entries
     * (defensive) are silently skipped. */
    int nprims = r->lazy_nprims_snapshot;
    unsigned char* wanted = (unsigned char*)calloc((size_t)nprims, 1);
    if (!wanted) {
        free(planes);
        set_error(r, "nu_extract_deferred_visible: alloc bitmap failed");
        return NU_ERROR;
    }
    int n_visible = 0;
    for (int i = 0; i < r->lazy_mesh_aabb_count; i++) {
        int pidx = r->lazy_mesh_aabbs[i].lazy_prim_idx;
        if (pidx < 0 || pidx >= nprims) continue;
        const float* mn = r->lazy_mesh_aabbs[i].bounds_min;
        const float* mx = r->lazy_mesh_aabbs[i].bounds_max;
        /* Sentinel "never cull" bounds = -FLT_MAX..FLT_MAX; aabb_in_frustum
         * returns 1 for those because every test corner is at infinity in
         * the inward direction. So sentinel bounds → always visible. */
        for (int c = 0; c < num_cameras; c++) {
            if (aabb_in_frustum(planes[c], mn, mx)) {
                if (!wanted[pidx]) {
                    wanted[pidx] = 1;
                    n_visible++;
                }
                break;
            }
        }
    }
    free(planes);

    fprintf(stderr,
            "nusd_renderer: nu_extract_deferred_visible — %d/%d lazy meshes "
            "visible across %d cameras; running filtered eager "
            "scene_load_from_stage_filtered.\n",
            n_visible, r->lazy_mesh_aabb_count, num_cameras);

    /* Temporarily clear NUSD_LAZY_MESH so the eager re-load takes the
     * non-lazy path. Mirrors nu_extract_deferred. */
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

    void* stage    = r->lazy_stage_pending;
    int owns_stage = r->lazy_owns_stage;

    const char* extract_label = r->usd_path[0] ? r->usd_path : "<lazy-extract-visible>";
    Scene* eager = scene_load_from_stage_filtered(
        stage, extract_label, wanted, nprims);
    free(wanted);
    if (!eager) {
        set_error(r, "nu_extract_deferred_visible: filtered eager load failed");
        if (had_prev) setenv("NUSD_LAZY_MESH", prev_env, 1);
        else          unsetenv("NUSD_LAZY_MESH");
        return NU_ERROR;
    }
    eager->_owns_stage = owns_stage;

    r->lazy_pending       = 0;
    r->lazy_stage_pending = NULL;
    r->lazy_owns_stage    = 0;
    free(r->lazy_mesh_aabbs);
    r->lazy_mesh_aabbs = NULL;
    r->lazy_mesh_aabb_count = 0;
    r->lazy_nprims_snapshot = 0;

    int rc = nu_attach_scene(r, eager);

    if (had_prev) setenv("NUSD_LAZY_MESH", prev_env, 1);
    else          unsetenv("NUSD_LAZY_MESH");

    return rc >= 0 ? NU_OK : NU_ERROR;
}

NuResult nu_extract_deferred_batched(NuRenderer* r, int num_batches)
{
    if (!r) return NU_ERROR;
    if (!r->lazy_pending) return NU_OK;
    if (!r->lazy_stage_pending) {
        set_error(r, "nu_extract_deferred_batched: no lazy stage pinned");
        r->lazy_pending = 0;
        return NU_ERROR;
    }
    /* Fall back to single-shot for trivial batch counts. */
    if (num_batches <= 1) return nu_extract_deferred(r);

    int nprims = r->lazy_nprims_snapshot;
    if (nprims <= 0) return nu_extract_deferred(r);

    /* Save NUSD_LAZY_MESH and NUSD_AUTO_FINALIZE; we control these for
     * the duration of the batch loop. */
    char saved_lm[16] = {0};
    int had_lm = 0;
    {
        const char* e = getenv("NUSD_LAZY_MESH");
        if (e) {
            had_lm = 1;
            size_t n = strlen(e);
            if (n >= sizeof(saved_lm)) n = sizeof(saved_lm) - 1;
            memcpy(saved_lm, e, n);
            saved_lm[n] = '\0';
        }
    }
    setenv("NUSD_LAZY_MESH", "0", 1);

    char saved_af[16] = {0};
    int had_af = 0;
    {
        const char* e = getenv("NUSD_AUTO_FINALIZE");
        if (e) {
            had_af = 1;
            size_t n = strlen(e);
            if (n >= sizeof(saved_af)) n = sizeof(saved_af) - 1;
            memcpy(saved_af, e, n);
            saved_af[n] = '\0';
        }
    }

    /* Capture stage ownership info, then clear the lazy state so attach
     * doesn't bounce back into the lazy branch on each batch. */
    void* stage    = r->lazy_stage_pending;
    int owns_stage = r->lazy_owns_stage;
    r->lazy_pending       = 0;
    r->lazy_stage_pending = NULL;
    r->lazy_owns_stage    = 0;
    free(r->lazy_mesh_aabbs);
    r->lazy_mesh_aabbs = NULL;
    r->lazy_mesh_aabb_count = 0;
    r->lazy_nprims_snapshot = 0;

    unsigned char* wanted = (unsigned char*)calloc((size_t)nprims, 1);
    if (!wanted) {
        if (had_lm) setenv("NUSD_LAZY_MESH", saved_lm, 1);
        else        unsetenv("NUSD_LAZY_MESH");
        set_error(r, "nu_extract_deferred_batched: alloc bitmap failed");
        return NU_ERROR;
    }

    fprintf(stderr,
            "nusd_renderer: nu_extract_deferred_batched — %d slices over "
            "%d prims (round-robin assignment, owns_stage=%d).\n",
            num_batches, nprims, owns_stage);

    NuResult final_rc = NU_OK;
    for (int b = 0; b < num_batches; b++) {
        /* Round-robin: prim i goes to batch (i % num_batches). */
        memset(wanted, 0, (size_t)nprims);
        int wanted_count = 0;
        for (int i = b; i < nprims; i += num_batches) {
            wanted[i] = 1;
            wanted_count++;
        }

        int is_last = (b == num_batches - 1);
        /* Only the LAST batch gets to auto-finalize; intermediate batches
         * must keep cpu_vertices alive so the next batch can append. */
        if (is_last && had_af) setenv("NUSD_AUTO_FINALIZE", saved_af, 1);
        else                   unsetenv("NUSD_AUTO_FINALIZE");

        char label[64];
        snprintf(label, sizeof(label), "<batched-extract %d/%d>", b + 1, num_batches);

        rss_trace(is_last ? "batched: last batch start" : "batched: batch start");

        Scene* slice = scene_load_from_stage_filtered(stage, label,
                                                       wanted, nprims);
        if (!slice) {
            set_error(r, "nu_extract_deferred_batched: slice load failed");
            final_rc = NU_ERROR;
            break;
        }
        /* Only the LAST batch closes the stage (if owned). Intermediate
         * batches must NOT close — the next batch still needs to read. */
        slice->_owns_stage = is_last ? owns_stage : 0;

        int loaded = nu_attach_scene(r, slice);
        if (loaded < 0) {
            set_error(r, "nu_extract_deferred_batched: slice attach failed");
            final_rc = NU_ERROR;
            break;
        }
        fprintf(stderr,
                "nusd_renderer: batched slice %d/%d done — wanted=%d, loaded=%d\n",
                b + 1, num_batches, wanted_count, loaded);
        rss_trace("batched: batch done");
    }

    free(wanted);

    /* Restore env. */
    if (had_lm) setenv("NUSD_LAZY_MESH", saved_lm, 1);
    else        unsetenv("NUSD_LAZY_MESH");
    if (had_af) setenv("NUSD_AUTO_FINALIZE", saved_af, 1);
    else        unsetenv("NUSD_AUTO_FINALIZE");

    return final_rc;
}

NuResult nu_set_current_time(NuRenderer* r, double time)
{
    if (!r) return NU_ERROR;
    r->current_time = time;
    return NU_OK;
}

/* Internal: scene loader queries the renderer's current evaluation time. */
double nu_get_current_time(NuRenderer* r)
{
    return r ? r->current_time : (double)NAN;
}

static int ensure_curve_blas(NuRenderer* r)
{
    if (!r || !r->curves_dirty || r->curves.nseg <= 0) return 1;
    if (!gpu_upload_curve_data(r->gpu,
                               r->curves.segments,
                               r->curves.colors,
                               r->curves.nseg)) {
        set_error(r, "gpu_upload_curve_data failed");
        return 0;
    }
    if (!gpu_build_curve_blas(r->gpu)) {
        set_error(r, "gpu_build_curve_blas failed");
        return 0;
    }
    r->curves_dirty = 0;
    return 1;
}

NuResult nu_build_accel(NuRenderer* r)
{
    if (!r) return NU_ERROR;

    if (r->scene_dirty) {
        if (!rebuild_gpu_buffers(r))
            return NU_ERROR;
    }

    /* Phase 11.A.2.5: curve data + BLAS must exist BEFORE the RT scene
     * is built — gpu_build_rt_scene reads gpu->curve_blas / curve_seg_count
     * to decide whether to include the curve hit-group + TLAS instance.
     *
     * Phase 12.x: AABBs are not uploaded — gpu_build_curve_blas runs a
     * compute pass that derives them from the segment SSBO before the
     * AS build. Saves ~825 MB host→device transfer on tera fixtures. */
    if (!ensure_curve_blas(r)) return NU_ERROR;

    if (r->accel_dirty) {
        if (!rebuild_accel(r))
            return NU_ERROR;
    }

    /* VK3DGRT splat-scene accel build. Independent of the mesh AS — runs
     * after mesh path so the GPU is idle. Phase 2 GPU side (BLAS/TLAS via
     * compute) lands incrementally; until then gpu_gs_build_accel returns
     * 0 with a TODO log and the splat path stays inactive. The mesh path
     * is unaffected — we deliberately do not propagate this failure. */
    if (r->gs_dirty && r->gs && gs_scene_particle_count(r->gs) > 0) {
        gpu_gs_upload_particles(r->gpu,
            gs_scene_positions(r->gs),
            gs_scene_scales(r->gs),
            gs_scene_orientations(r->gs),
            gs_scene_opacities(r->gs),
            gs_scene_kernel_scales(r->gs),
            gs_scene_sh_coefficients(r->gs),
            (uint32_t)gs_scene_particle_count(r->gs),
            gs_scene_sh_degree(r->gs),
            (int)gs_scene_proxy(r->gs));
        gpu_gs_build_accel(r->gpu, gs_scene_prim_xform(r->gs));
        /* Note: clearing gs_dirty even on partial-implementation failure so
         * we don't re-log every frame. The renderer surfaces gs_built via
         * nu_gs_available, which honors gpu->gs_built (still 0 today). */
        r->gs_dirty = 0;
    }

    return NU_OK;
}

/* RSS attribution helper — pull a handful of /proc/self/smaps_rollup
 * fields (kB) into a single string we can print alongside finalize logs.
 * Returns 0-filled out_kb on parse failure so callers can still log
 * a placeholder. */
static void read_smaps_rollup_kb(long* rss, long* anon, long* file_kb)
{
    *rss = *anon = *file_kb = 0;
    FILE* f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if      (strncmp(line, "Rss:",        4) == 0) sscanf(line + 4,  "%ld", rss);
        else if (strncmp(line, "Anonymous:", 10) == 0) sscanf(line + 10, "%ld", anon);
        else if (strncmp(line, "Pss_File:",   9) == 0) sscanf(line + 9,  "%ld", file_kb);
    }
    fclose(f);
}

NuResult nu_finalize_scene(NuRenderer* r)
{
    if (!r) return NU_ERROR;
    if (r->scene_finalized) return NU_OK;  /* idempotent */

    /* Make sure GPU buffers + BLAS/TLAS are fully built from the current
     * CPU mirror BEFORE we free it. nu_build_accel handles scene_dirty
     * (rebuild_gpu_buffers) and accel_dirty (rebuild_accel). */
    if (nu_build_accel(r) != NU_OK) {
        set_error(r, "nu_finalize_scene: nu_build_accel failed");
        return NU_ERROR;
    }

    /* Compute freed-bytes for the diagnostic log (the vertex stride at
     * finalize time matches whatever the most recent rebuild used; both
     * 9-float and 12-float layouts are uploaded contiguously). */
    size_t bytes_v = (size_t)r->cpu_vertex_capacity
                   * (size_t)(r->vertex_stride_floats ? r->vertex_stride_floats : 12u)
                   * sizeof(float);
    size_t bytes_i = (size_t)r->cpu_index_capacity * sizeof(uint32_t);

    long rss_before_kb = 0, anon_before_kb = 0, file_before_kb = 0;
    read_smaps_rollup_kb(&rss_before_kb, &anon_before_kb, &file_before_kb);

    free(r->cpu_vertices);  r->cpu_vertices = NULL;
    free(r->cpu_indices);   r->cpu_indices  = NULL;
    r->cpu_vertex_capacity = 0;
    r->cpu_index_capacity  = 0;

    /* Return freed pages to the kernel so VmRSS reflects the drop
     * immediately. glibc holds back small free()s in its arena; the
     * malloc_trim hint lets us see the actual win in /proc/self/status. */
#ifdef __GLIBC__
    malloc_trim(0);
#endif

    long rss_after_kb = 0, anon_after_kb = 0, file_after_kb = 0;
    read_smaps_rollup_kb(&rss_after_kb, &anon_after_kb, &file_after_kb);

    r->scene_finalized = 1;

    fprintf(stderr,
            "nusd_renderer: scene finalized — dropped CPU mirror "
            "(%.1f MB vertex + %.1f MB index = %.1f MB total)\n"
            "nusd_renderer: RSS attribution (kB)  before: RSS=%ld Anon=%ld File=%ld\n"
            "nusd_renderer: RSS attribution (kB)   after: RSS=%ld Anon=%ld File=%ld "
            "(freed RSS=%ld, Anon=%ld)\n",
            (double)bytes_v / (1024.0 * 1024.0),
            (double)bytes_i / (1024.0 * 1024.0),
            (double)(bytes_v + bytes_i) / (1024.0 * 1024.0),
            rss_before_kb, anon_before_kb, file_before_kb,
            rss_after_kb,  anon_after_kb,  file_after_kb,
            rss_before_kb - rss_after_kb,
            anon_before_kb - anon_after_kb);

    /* Dump the geometry-hash census if it was enabled this run, so the
     * "content dedup potential" number lives in the same log as the
     * mirror-dropped number. */
    geo_census_dump_and_reset();

    return NU_OK;
}

/* Helper: assert the scene is still mutable. Returns 1 if the caller may
 * proceed; logs an error and returns 0 if the scene has been finalized
 * (CPU mirror dropped → rebuild_gpu_buffers would read freed memory).
 * Callers should propagate NU_ERROR. */
static int check_scene_mutable(NuRenderer* r, const char* api)
{
    if (!r) return 0;
    if (r->scene_finalized) {
        char buf[160];
        snprintf(buf, sizeof(buf),
                 "%s called after nu_finalize_scene — load a fresh scene "
                 "via nu_clear_scene + nu_load_usd to mutate again", api);
        set_error(r, buf);
        fprintf(stderr, "nusd_renderer: %s\n", buf);
        return 0;
    }
    return 1;
}

/* ---- VK3DGRT (Gaussian splat scene) ----
 *
 * Public nu_gs_* surface. Particle data + config knobs are owned by the
 * NuGsScene substate (see src/gs_scene.c); GPU acceleration structures and
 * RT pipeline live in gpu_vulkan.c (Phase 2/3 of VK3DGRT plan, not yet
 * wired). Setters that change the GPU footprint flip r->gs_dirty so the
 * next render rebuilds the splat BLAS/TLAS; setters that change pipeline
 * shape (K, camera model, proxy) also clear r->gs_pipeline_built. */

static int gs_set_error(NuRenderer* r, const char* msg)
{
    if (r) set_error(r, msg);
    return NU_ERROR;
}

NuResult nu_gs_set_particles(NuRenderer* r, const NuGsDesc* desc)
{
    if (!r || !r->gs) return NU_ERROR;
    NuResult rc = gs_scene_set_particles(r->gs, desc);
    if (rc != NU_OK) return gs_set_error(r, "nu_gs_set_particles: invalid desc");
    r->gs_dirty = 1;
    return NU_OK;
}

NuResult nu_gs_clear_particles(NuRenderer* r)
{
    if (!r || !r->gs) return NU_ERROR;
    NuResult rc = gs_scene_clear_particles(r->gs);
    if (rc != NU_OK) return rc;
    r->gs_dirty = 1;
    return NU_OK;
}

NuResult nu_gs_set_proxy(NuRenderer* r, NuGsProxyKind kind)
{
    if (!r || !r->gs) return NU_ERROR;
    NuResult rc = gs_scene_set_proxy(r->gs, kind);
    if (rc != NU_OK) return gs_set_error(r, "nu_gs_set_proxy: invalid kind");
    /* Proxy switches the BLAS geometry (icosahedron vs AABB) and toggles
     * the procedural-rint hit group. Both BLAS and pipeline must rebuild. */
    r->gs_dirty = 1;
    r->gs_pipeline_built = 0;
    return NU_OK;
}

NuResult nu_gs_set_color_space(NuRenderer* r, NuGsColorSpace cs)
{
    if (!r || !r->gs) return NU_ERROR;
    NuResult rc = gs_scene_set_color_space(r->gs, cs);
    if (rc != NU_OK) return gs_set_error(r, "nu_gs_set_color_space: invalid value");
    /* Color-space affects only shader-side per-hit math (sRGB→linear); the
     * shader reads it from a uniform, so neither BLAS nor pipeline rebuild. */
    return NU_OK;
}

NuResult nu_gs_set_camera_model(NuRenderer* r, NuGsCameraModel cm)
{
    if (!r || !r->gs) return NU_ERROR;
    NuResult rc = gs_scene_set_camera_model(r->gs, cm);
    if (rc != NU_OK) return gs_set_error(r, "nu_gs_set_camera_model: invalid value");
    /* Camera model is selected via spec-constant in the rgen shader; needs
     * a pipeline rebuild but no BLAS/TLAS change. */
    r->gs_pipeline_built = 0;
    return NU_OK;
}

NuResult nu_gs_set_k(NuRenderer* r, int k)
{
    if (!r || !r->gs) return NU_ERROR;
    NuResult rc = gs_scene_set_k(r->gs, k);
    if (rc != NU_OK) return gs_set_error(r, "nu_gs_set_k: must be 8, 16, or 32");
    /* K is a spec-constant + sizes the per-tile SoA K-buffer SSBO. Pipeline
     * + buffer must reallocate. */
    r->gs_pipeline_built = 0;
    return NU_OK;
}

NuResult nu_gs_set_max_passes(NuRenderer* r, int max_passes)
{
    if (!r || !r->gs) return NU_ERROR;
    NuResult rc = gs_scene_set_max_passes(r->gs, max_passes);
    if (rc != NU_OK) return gs_set_error(r, "nu_gs_set_max_passes: must be >= 1");
    /* Read by raygen as a push-constant — no rebuild. */
    return NU_OK;
}

NuResult nu_gs_set_min_transmittance(NuRenderer* r, float eps)
{
    if (!r || !r->gs) return NU_ERROR;
    NuResult rc = gs_scene_set_min_transmittance(r->gs, eps);
    if (rc != NU_OK) return gs_set_error(r, "nu_gs_set_min_transmittance: must be in [0,1]");
    return NU_OK;
}

NuResult nu_gs_set_iso_opacity_threshold(NuRenderer* r, float iso)
{
    if (!r || !r->gs) return NU_ERROR;
    NuResult rc = gs_scene_set_iso_opacity_threshold(r->gs, iso);
    if (rc != NU_OK) return gs_set_error(r, "nu_gs_set_iso_opacity_threshold: must be in [0,1]");
    return NU_OK;
}

int nu_gs_available(NuRenderer* r)
{
    /* Phase 1: report ready iff particles are uploaded. The actual GPU
     * extension probe (VK_KHR_acceleration_structure / ray_tracing_pipeline /
     * position_fetch) lands with the gpu_vulkan side in Phase 2 — at that
     * point this also gates on rt-availability. */
    if (!r || !r->gs) return 0;
    if (!gpu_rt_available(r->gpu)) return 0;
    return gs_scene_particle_count(r->gs) > 0 ? 1 : 0;
}

int nu_gs_particle_count(NuRenderer* r)
{
    if (!r || !r->gs) return 0;
    return gs_scene_particle_count(r->gs);
}

/* Lazy-load the Slang RT shader blobs. Returns 1 on success; caller must
 * free() each non-NULL pointer. The K-buffer trace loop integrates in
 * rgen and never invokes closest-hit, so rchit is permitted to be NULL
 * (the gpu_gs_build_pipeline path skips the CLOSEST_HIT stage when so).
 * `load_rint` selects whether to load the AABB-proxy intersection
 * shader — only needed for NU_GS_PROXY_AABB. */
static int gs_load_rt_spvs(uint32_t** rgen,  uint32_t* rgen_sz,
                           uint32_t** rmiss, uint32_t* rmiss_sz,
                           uint32_t** rahit, uint32_t* rahit_sz,
                           uint32_t** rint,  uint32_t* rint_sz,
                           int load_rint)
{
    char p[512];
    snprintf(p, sizeof(p), "%s/slang/gs_3dgrt.raygeneration.spv", SHADER_DIR);
    *rgen = load_shader_file(p, rgen_sz);
    snprintf(p, sizeof(p), "%s/slang/gs_3dgrt.miss.spv", SHADER_DIR);
    *rmiss = load_shader_file(p, rmiss_sz);
    snprintf(p, sizeof(p), "%s/slang/gs_3dgrt.anyhit.spv", SHADER_DIR);
    *rahit = load_shader_file(p, rahit_sz);
    *rint = NULL;  *rint_sz = 0;
    if (load_rint) {
        snprintf(p, sizeof(p), "%s/slang/gs_3dgrt.intersection.spv", SHADER_DIR);
        *rint = load_shader_file(p, rint_sz);
    }
    if (!*rgen || !*rmiss || !*rahit || (load_rint && !*rint)) {
        free(*rgen); free(*rmiss); free(*rahit); free(*rint);
        *rgen = *rmiss = *rahit = *rint = NULL;
        return 0;
    }
    return 1;
}

NuResult nu_gs_render(NuRenderer* r, int cam_id)
{
    (void)cam_id;
    if (!r || !r->gs) return NU_ERROR;
    if (!r->gpu) return NU_ERROR_NO_RT;
    if (gs_scene_particle_count(r->gs) == 0) return NU_ERROR_NO_SCENE;

    /* Ensure gs accel structures are built. Same dirty-flag dance as
     * nu_build_accel — the splat path is independent of the mesh path,
     * so we can do it inline here without touching the mesh state. */
    if (r->gs_dirty) {
        gpu_gs_upload_particles(r->gpu,
            gs_scene_positions(r->gs),
            gs_scene_scales(r->gs),
            gs_scene_orientations(r->gs),
            gs_scene_opacities(r->gs),
            gs_scene_kernel_scales(r->gs),
            gs_scene_sh_coefficients(r->gs),
            (uint32_t)gs_scene_particle_count(r->gs),
            gs_scene_sh_degree(r->gs),
            (int)gs_scene_proxy(r->gs));
        gpu_gs_build_accel(r->gpu, gs_scene_prim_xform(r->gs));
        r->gs_dirty = 0;
    }

    if (!nu_gs_available(r)) {
        set_error(r, "nu_gs_render: gs not available (RT or AS missing)");
        return NU_ERROR;
    }

    /* Lazy build the splat RT pipeline. We always include the closest-hit
     * shader in the pipeline because some drivers / spec interpretations
     * skip any-hit invocations when the hit group has neither a
     * closest-hit nor an intersection shader. The K-buffer trace loop
     * doesn't need rchit's output (rgen integrates directly) but having
     * a no-op rchit in the pipeline keeps any-hit firing reliably. */
    if (!r->gs_pipeline_built) {
        const int load_rint = (gs_scene_proxy(r->gs) == NU_GS_PROXY_AABB);
        uint32_t *rgen=NULL,*rmiss=NULL,*rahit=NULL,*rchit=NULL,*rint=NULL;
        uint32_t  rgen_sz=0,rmiss_sz=0,rahit_sz=0,rchit_sz=0,rint_sz=0;
        char p[512];
        snprintf(p, sizeof(p), "%s/slang/gs_3dgrt.closesthit.spv", SHADER_DIR);
        rchit = load_shader_file(p, &rchit_sz);
        if (!gs_load_rt_spvs(&rgen,&rgen_sz, &rmiss,&rmiss_sz,
                             &rahit,&rahit_sz, &rint,&rint_sz, load_rint)) {
            free(rchit);
            set_error(r, "nu_gs_render: missing Slang RT shaders "
                         "(was the renderer built with -DNU_BUILD_GS_RT=ON?)");
            return NU_ERROR_NO_RT;
        }
        int ok = gpu_gs_build_pipeline(r->gpu,
            rgen,  rgen_sz,
            rahit, rahit_sz,
            rchit, rchit_sz,
            rint,  rint_sz,
            rmiss, rmiss_sz);
        free(rgen); free(rmiss); free(rahit); free(rchit); free(rint);
        if (!ok) {
            set_error(r, "nu_gs_render: gpu_gs_build_pipeline failed");
            return NU_ERROR_NO_RT;
        }
        r->gs_pipeline_built = 1;
    }

    /* Pack the camera. compute_camera_inverses is the static helper the
     * async render path uses; forward-declared above. */
    float vp_inv[32];
    compute_camera_inverses(&r->camera, vp_inv);

    int w, h;
    nu_get_render_size(r, &w, &h);
    int rc = gpu_gs_render_tile(r->gpu, vp_inv, vp_inv + 16,
                                gs_scene_k(r->gs),
                                gs_scene_max_passes(r->gs),
                                gs_scene_min_transmittance(r->gs),
                                (int)gs_scene_color_space(r->gs),
                                (int)gs_scene_camera_model(r->gs),
                                w, h);
    if (!rc) {
        set_error(r, "nu_gs_render: gpu_gs_render_tile failed");
        return NU_ERROR;
    }
    r->has_frame = 1;
    return NU_OK;
}

NuResult nu_gs_fetch_depth(NuRenderer* r, float* out_depth)
{
    if (!r || !out_depth) return NU_ERROR;
    if (!r->has_frame) {
        set_error(r, "nu_gs_fetch_depth: no frame rendered yet");
        return NU_ERROR;
    }
    if (!gpu_gs_fetch_depth(r->gpu, out_depth,
                            (uint32_t)r->width, (uint32_t)r->height)) {
        set_error(r, "nu_gs_fetch_depth: gpu_gs_fetch_depth failed");
        return NU_ERROR;
    }
    return NU_OK;
}

NuResult nu_gs_fetch_normal(NuRenderer* r, float* out_normal)
{
    if (!r || !out_normal) return NU_ERROR;
    if (!r->has_frame) {
        set_error(r, "nu_gs_fetch_normal: no frame rendered yet");
        return NU_ERROR;
    }
    if (!gpu_gs_fetch_normal(r->gpu, out_normal,
                             (uint32_t)r->width, (uint32_t)r->height)) {
        set_error(r, "nu_gs_fetch_normal: gpu_gs_fetch_normal failed");
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
        memcpy(r->meshes[id].transform, &mat4x4s[i * 16], 16 * sizeof(float));
    }

    /* Mark TLAS dirty so the next RT render rebuilds instance transforms.
     * BLASes are unchanged — only the TLAS needs updating. */
    r->use_gpu_transforms = 0;
    r->tlas_dirty = 1;
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
        r->meshes[id].color[0] = rgb[i * 3 + 0];
        r->meshes[id].color[1] = rgb[i * 3 + 1];
        r->meshes[id].color[2] = rgb[i * 3 + 2];
    }

    r->colors_dirty = 1;
    return NU_OK;
}

NuResult nu_set_visibility(NuRenderer* r,
                           const int* mesh_ids, const int* visible, int count)
{
    if (!r || !mesh_ids || !visible || count <= 0) return NU_ERROR;

    for (int i = 0; i < count; i++) {
        int id = mesh_ids[i];
        if (id < 0 || id >= r->nmeshes || !r->meshes[id].active)
            return NU_ERROR_BAD_ID;
        r->meshes[id].visible = visible[i] ? 1 : 0;
    }

    /* Visibility changes require TLAS rebuild to exclude/include instances */
    r->tlas_dirty = 1;
    return NU_OK;
}

/* ---- Camera ---- */

static void renderer_default_up(const NuRenderer* r, float up[3])
{
    up[0] = 0.0f;
    up[1] = 1.0f;
    up[2] = 0.0f;
    if (!r) return;
    if (r->scene_up_axis == 0) {
        up[0] = 1.0f; up[1] = 0.0f; up[2] = 0.0f;
    } else if (r->scene_up_axis == 2) {
        up[0] = 0.0f; up[1] = 0.0f; up[2] = 1.0f;
    }
}

NuResult nu_set_camera(NuRenderer* r, int cam_id, const NuCameraDesc* desc)
{
    if (!r || !desc || cam_id != 0) return NU_ERROR_BAD_ID;

    float up[3];
    renderer_default_up(r, up);
    float fov_rad = (desc->fov_degrees > 0 ? desc->fov_degrees : 60.0f)
                  * 3.14159265f / 180.0f;
    float nz = desc->near_clip > 0 ? desc->near_clip : 0.1f;
    float fz = desc->far_clip > nz ? desc->far_clip : 10000.0f;
    camera_set_explicit_view(&r->camera, desc->eye, desc->target, up,
                             fov_rad, nz, fz);

    r->camera_valid = 1;
    if (rt_camera_residency_enabled())
        r->accel_dirty = 1;
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
    float fov_rad = (fov_degrees > 0 ? fov_degrees : 60.0f) * 3.14159265f / 180.0f;
    float nz = near_clip > 0 ? near_clip : 0.1f;
    float fz = far_clip > 0 ? far_clip : 10000.0f;
    camera_set_explicit_view(&r->camera, eye, target, up, fov_rad, nz, fz);
    r->camera_valid = 1;
    if (rt_camera_residency_enabled())
        r->accel_dirty = 1;
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
    NuResult rc = nu_set_camera_explicit(r, eye, target, up, fov_degrees, near_clip, far_clip);
    if (rc != NU_OK) return rc;
    camera_set_projection_shift(&r->camera, projection_shift_x, projection_shift_y);
    if (rt_camera_residency_enabled())
        r->accel_dirty = 1;
    return NU_OK;
}

NuResult nu_get_camera(NuRenderer* r, int cam_id, NuCameraDesc* out_desc)
{
    if (!r || !out_desc) return NU_ERROR;
    if (cam_id != 0) return NU_ERROR_BAD_ID;

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

NuResult nu_set_exposure(NuRenderer* r, const NuExposureDesc* desc)
{
    if (!r || !desc) return NU_ERROR;
    renderer_apply_exposure_desc(r, desc);
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

    /* Phase 11.A.2.5: a curve-only scene has total_indices==0 but is
     * still renderable via the curve hit-group. */
    if (r->total_indices == 0 && r->curves.nseg == 0 &&
        r->raster_curves.total_indices == 0) {
        set_error(r, "No geometry to render");
        return NU_ERROR_NO_SCENE;
    }

    /* Vulkan does not have a raster curve draw path yet. Keep raster as raster
     * so mesh geometry remains renderable with RT disabled; opt in to the old
     * curve-complete RT substitution when explicitly requested. */
    if (mode != NU_RENDER_RT && r->curves.nseg > 0 &&
        env_enabled_default_off("NUSD_RASTER_CURVES_USE_RT") &&
        r->enable_rt && gpu_rt_available(r->gpu))
        mode = NU_RENDER_RT;

    /* Rebuild accel structures if needed. Authored-light raster can use the
     * same TLAS for inline ray-query shadows, which keeps warehouse rect
     * lights from washing out close material shots. */
    int raster_authored_light_shadows =
        (mode == NU_RENDER_RASTER &&
         r->enable_rt &&
         gpu_rt_available(r->gpu) &&
         r->authored_light_count > 0 &&
         (r->curves.nseg == 0 ||
          env_enabled_default_off("NUSD_RASTER_CURVES_RT_SHADOWS")) &&
         getenv("NUSD_RASTER_DISABLE_RT_SHADOWS") == NULL);
    /* Culled-proxy TLAS is camera-dependent — force a rebuild each frame so a
     * moving camera re-selects the visible PI set (the full Moana TLAS doesn't
     * fit on the GPU; only the visible instances are built). */
    if ((mode == NU_RENDER_RT || mode == NU_RENDER_SHADOW) && rt_cull_active())
        r->accel_dirty = 1;
    if (r->accel_dirty && (mode == NU_RENDER_RT ||
                           mode == NU_RENDER_SHADOW ||
                           raster_authored_light_shadows)) {
        if ((mode == NU_RENDER_RT || mode == NU_RENDER_SHADOW) &&
            !ensure_curve_blas(r)) return NU_ERROR;
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

        int gpu_xforms_ready = gpu_transforms_ready(r);

        /* Rebuild TLAS if transforms changed. The GPU-driven path writes
         * instance records from the imported CUDA/Vulkan transform buffer
         * after gpu_begin_frame_rt(), so the host memcpy path stays off. */
        if (r->tlas_dirty && !gpu_xforms_ready) {
            /* USD row-vector 4x4 → VkTransformMatrixKHR 3x4
             * (column-vector row-major; translation in last column). */
            float* xforms_3x4 = (float*)malloc(r->nmeshes * 12 * sizeof(float));
            if (xforms_3x4) {
                for (int m = 0; m < r->nmeshes; m++) {
                    usd4x4_to_vk3x4(r->meshes[m].transform, &xforms_3x4[m * 12]);
                }
                uint8_t* vis = (uint8_t*)malloc(r->nmeshes);
                if (vis) {
                    for (int m = 0; m < r->nmeshes; m++)
                        vis[m] = r->meshes[m].visible ? 1 : 0;
                }
                gpu_update_tlas(r->gpu, xforms_3x4, vis, (uint32_t)r->nmeshes);
                free(vis);
                free(xforms_3x4);
            }
            r->tlas_dirty = 0;
        } else if (r->tlas_dirty && gpu_xforms_ready) {
            r->tlas_dirty = 0;
        }

        if (!gpu_begin_frame_rt(r->gpu)) return NU_ERROR;

        if (gpu_xforms_ready) {
            if (!gpu_translate_instances_inline(r->gpu, r->gpu_transform_count)) {
                fprintf(stderr,
                        "[nu_renderer] gpu_translate_instances_inline failed; "
                        "single-camera RT is using the previous TLAS this frame.\n");
            }
        }

        /* Compute inverse matrices for RT push constants.
         * The view matrix is orthonormal so transpose = inverse for rotation,
         * and we reconstruct translation. For proj, we invert analytically. */
        GpuRtPushConstants pc;
        memset(&pc, 0, sizeof(pc));

        /* View inverse: transpose rotation, negate translated eye */
        {
            float v[16];
            camera_get_view(&r->camera, v);
            /* Transpose 3x3 rotation block */
            float vi[16];
            memset(vi, 0, sizeof(vi));
            vi[0] = v[0]; vi[1] = v[4]; vi[2]  = v[8];
            vi[4] = v[1]; vi[5] = v[5]; vi[6]  = v[9];
            vi[8] = v[2]; vi[9] = v[6]; vi[10] = v[10];
            /* Translation = -R^T * t */
            vi[3]  = -(vi[0]*v[3]  + vi[1]*v[7]  + vi[2]*v[11]);
            vi[7]  = -(vi[4]*v[3]  + vi[5]*v[7]  + vi[6]*v[11]);
            vi[11] = -(vi[8]*v[3]  + vi[9]*v[7]  + vi[10]*v[11]);
            vi[15] = 1.0f;
            memcpy(pc.view_inv, vi, sizeof(pc.view_inv));
        }

        /* Projection inverse, including off-axis lens shift. */
        {
            float p[16];
            camera_get_proj(&r->camera, p);
            float pi[16];
            memset(pi, 0, sizeof(pi));
            pi[0]  = 1.0f / p[0];
            pi[5]  = 1.0f / p[5];
            pi[3]  = p[2] / p[0];
            pi[7]  = p[6] / p[5];
            pi[11] = 1.0f / p[14];
            pi[14] = -1.0f;
            pi[15] = p[10] / p[14];
            memcpy(pc.proj_inv, pi, sizeof(pc.proj_inv));
        }
        renderer_rt_scene_metrics(r, &pc.ground_y, &pc.scene_scale);
        pc.fast_mode = (uint32_t)r->fast_mode;
        pc.depth_enabled = 0;   /* no depth output for single-camera */
        pc.segmentation_enabled = 0;   /* no segmentation for single-camera */
        pc.normals_enabled = 0;        /* no normals for single-camera */
        /* Single-camera RT binds binding 17 to a real G-buffer, but contact
         * sheet/profile renders do not consume it. Keep the diagnostic writes
         * opt-in so visual renders avoid an otherwise invisible per-pixel SSBO
         * write. */
        pc.deferred_shade_enabled =
            (uint32_t)env_enabled_default_off("NUSD_RT_GBUFFER");
        pc.tone_exposure_scale = r->tone_exposure_scale;
        pc.tone_sky_scale = r->tone_sky_scale;
        pc.tone_white_point = r->tone_white_point;
        pc.tone_flags = r->tone_flags;
        pc.rt_ibl_fill_scale = r->rt_ibl_fill_scale;

        gpu_cmd_trace_rays(r->gpu, &pc);
        gpu_end_frame_rt(r->gpu);

    } else {
        /* Raster path */
        if (!gpu_begin_frame(r->gpu)) return NU_ERROR;

        /* Pipeline pick: textured raster handles RASTER and SHADOW alike —
         * mesh.frag's shadow gate (pc.ibl_params[3]) toggles the inline
         * ray queries. The bare 3-point lit raster_pipeline is the
         * pre-materials fallback. */
        GpuPipeline pipe;
        int textured = 0;
        if (r->raster_pipeline_textured) {
            pipe = r->raster_pipeline_textured;
            textured = 1;
        } else {
            pipe = r->raster_pipeline;
        }
        if (!pipe) {
            set_error(r, "Pipeline not available");
            return NU_ERROR;
        }

        gpu_cmd_bind_pipeline(r->gpu, pipe);
        if (textured) gpu_cmd_bind_materials(r->gpu);

        /* Curves-only scenes (e.g. showcase_grid_*.usdc) skip mesh upload —
         * vertex/index buffers stay NULL. gpu_cmd_bind_vertex_buffer would
         * SIGSEGV on `&buf->buffer` if we pushed a NULL handle. The mesh
         * loop below also short-circuits when nmeshes == 0, so just skip
         * the bind+draw block entirely. */
        if (r->vertex_buffer && r->index_buffer) {
            gpu_cmd_bind_vertex_buffer(r->gpu, r->vertex_buffer);
            gpu_cmd_bind_index_buffer(r->gpu, r->index_buffer);
        }

        for (int i = 0; i < r->nmeshes; i++) {
            RendererMesh* m = &r->meshes[i];
            if (!m->active || !m->visible || m->nindices == 0) continue;

            /* Build MVP */
            float mvp[16];
            float model[16];
            memcpy(model, m->transform, sizeof(model));

            /* mvp = vp * model */
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
            memcpy(pc.model, model, sizeof(pc.model));
            pc.color[0] = m->color[0];
            pc.color[1] = m->color[1];
            pc.color[2] = m->color[2];
            {
                uint32_t ptex_offset = m->ptex_color_offset;
                memcpy(&pc.color[3], &ptex_offset, sizeof(ptex_offset));
            }
            memcpy(pc.eye_pos, eye, 3 * sizeof(float));
            /* Per-instance raster material id packed into eye_pos.w (int bits).
             * Clamp <0 to 0 to reproduce the legacy per-vertex matID exactly
             * (nu_add_mesh stored material_index>=0 ? material_index : 0), so
             * this is a pure per-vertex → per-instance move with no render
             * change — and it frees content-hash dedup to share geometry across
             * materials (the prototype's dead per-vertex matID is ignored). */
            {
                int32_t mat_id = (m->material_index >= 0) ? m->material_index : 0;
                memcpy(&pc.eye_pos[3], &mat_id, sizeof(int32_t));
            }

            /* IBL params: gate the raster shader's IBL branch when an env
             * is loaded; otherwise it falls back to procedural sky/ground.
             * Mirrors how the SceneData SSBO drives the rchit. */
            pc.ibl_params[0] = gpu_get_ibl_loaded(r->gpu) ? 1.0f : 0.0f;
            pc.ibl_params[1] = (float)gpu_get_env_mip_levels(r->gpu);
            pc.ibl_params[2] = gpu_get_env_intensity(r->gpu);
            if (pc.ibl_params[0] == 0.0f && !r->fallback_lights_active) {
                /* Negative markers mean authored-light / no-IBL. Small
                 * fixture sets (GB300 probe) use the authored lights at the
                 * normal no-HDR exposure; many-light warehouse scenes keep the
                 * authored lights but use lower exposure until raster has
                 * proper light selection, shadows, and normalization. */
                pc.ibl_params[2] =
                    (r->authored_light_count > 8) ? -3.0f : -2.0f;
            }
            pc.tone_params[0] = r->tone_exposure_scale;
            pc.tone_params[1] = r->tone_sky_scale;
            pc.tone_params[2] = r->tone_white_point;
            /* Pack the scene up axis into tone_flags bits 24-25 so mesh.frag's
             * dirToEquirect samples the env/irradiance maps with the same
             * up-axis convention as the RT path (color parity on Y-up scenes). */
            uint32_t tone_flags = r->tone_flags |
                (((uint32_t)r->scene_up_axis & 3u) << 24);
            memcpy(&pc.tone_params[3], &tone_flags, sizeof(tone_flags));
            /* Ray-query quality gate for mesh.frag binding 5 (TLAS).
             *   0.0: no TLAS queries
             *   0.5: short contact AO when an accel structure is already built
             *   1.0: contact AO plus the shadow-mode long diffuse/spec queries */
            if (mode == NU_RENDER_SHADOW) {
                pc.ibl_params[3] = 1.0f;
            } else {
                pc.ibl_params[3] = gpu_rt_built(r->gpu) ? 0.5f : 0.0f;
            }

            gpu_cmd_push_constants(r->gpu, &pc, sizeof(pc));
            gpu_cmd_draw_indexed(r->gpu,
                (uint32_t)m->nindices,
                m->index_offset,
                (int32_t)m->vertex_offset);
        }

        /* Compact PointInstancer batches — draw each prototype's geometry once
         * per placement via instance-rate vertex attributes. RT consumes the
         * same r->pi_batches/r->pi_transforms through the TLAS; this is the
         * raster equivalent, so Moana's instanceable ocean/beach/foliage (all
         * authored instanceable=true → compact PI) render in raster instead of
         * showing as black. Opt out with NUSD_RASTER_DISABLE_PI=1 for A/B. */
        if (r->raster_pipeline_instanced &&
            r->pi_instance_buffer &&
            r->npi_batches > 0 &&
            r->vertex_buffer && r->index_buffer &&
            getenv("NUSD_RASTER_DISABLE_PI") == NULL) {
            gpu_cmd_bind_pipeline(r->gpu, r->raster_pipeline_instanced);
            gpu_cmd_bind_materials(r->gpu);
            gpu_cmd_bind_vertex_buffer(r->gpu, r->vertex_buffer);
            gpu_cmd_bind_instance_buffer(r->gpu, r->pi_instance_buffer);
            gpu_cmd_bind_index_buffer(r->gpu, r->index_buffer);

            int   ibl_loaded   = gpu_get_ibl_loaded(r->gpu);
            float ibl_intensity = gpu_get_env_intensity(r->gpu);
            float ibl_mips     = (float)gpu_get_env_mip_levels(r->gpu);
            float shadow_gate  = (mode == NU_RENDER_SHADOW)
                ? 1.0f : (gpu_rt_built(r->gpu) ? 0.5f : 0.0f);

            for (int b = 0; b < r->npi_batches; b++) {
                const SceneInstanceBatch* pb = &r->pi_batches[b];
                int rp = pb->prototype_mesh_idx;
                if (rp < 0 || rp >= r->nmeshes) continue;
                const RendererMesh* pm = &r->meshes[rp];
                if (!pm->active || pm->nindices == 0 || pb->transform_count == 0)
                    continue;
                if ((uint64_t)pb->transform_offset + pb->transform_count >
                    r->pi_instance_count)
                    continue;

                GpuMeshPushConstants pc;
                memset(&pc, 0, sizeof(pc));
                /* The instanced vertex shader reads pc.mvp as the camera VP
                 * (the per-instance model comes from binding 1). */
                memcpy(pc.mvp, vp, sizeof(pc.mvp));
                pc.color[0] = pm->color[0];
                pc.color[1] = pm->color[1];
                pc.color[2] = pm->color[2];
                {
                    uint32_t ptex_offset = pm->ptex_color_offset;
                    memcpy(&pc.color[3], &ptex_offset, sizeof(ptex_offset));
                }
                memcpy(pc.eye_pos, eye, 3 * sizeof(float));
                {
                    int32_t mat_id = (pm->material_index >= 0) ? pm->material_index : 0;
                    memcpy(&pc.eye_pos[3], &mat_id, sizeof(int32_t));
                }
                pc.ibl_params[0] = ibl_loaded ? 1.0f : 0.0f;
                pc.ibl_params[1] = ibl_mips;
                pc.ibl_params[2] = ibl_intensity;
                if (pc.ibl_params[0] == 0.0f && !r->fallback_lights_active) {
                    pc.ibl_params[2] =
                        (r->authored_light_count > 8) ? -3.0f : -2.0f;
                }
                pc.ibl_params[3] = shadow_gate;
                pc.tone_params[0] = r->tone_exposure_scale;
                pc.tone_params[1] = r->tone_sky_scale;
                pc.tone_params[2] = r->tone_white_point;
                {
                    /* up axis in bits 24-25 — see the direct-mesh loop. */
                    uint32_t tone_flags = r->tone_flags |
                        (((uint32_t)r->scene_up_axis & 3u) << 24);
                    memcpy(&pc.tone_params[3], &tone_flags, sizeof(tone_flags));
                }

                gpu_cmd_push_constants(r->gpu, &pc, sizeof(pc));
                gpu_cmd_draw_indexed_instanced(r->gpu,
                    (uint32_t)pm->nindices,
                    pb->transform_count,
                    pm->index_offset,
                    (int32_t)pm->vertex_offset,
                    pb->transform_offset);
            }
        }

        if (r->raster_curves.ndraws > 0 &&
            r->raster_curves.vertex_buffer &&
            r->raster_curves.index_buffer) {
            if (!r->curve_pipeline) {
                fprintf(stderr,
                        "nusd_renderer: raster curves present but curve pipeline is unavailable\n");
            } else {
                gpu_cmd_bind_pipeline(r->gpu, r->curve_pipeline);
                gpu_cmd_bind_vertex_buffer(r->gpu, r->raster_curves.vertex_buffer);

                const char* cull_env = getenv("NUSD_RASTER_CURVE_CULL");
                int cull_chunks = !(cull_env && cull_env[0] == '0');
                for (int c = 0; c < r->raster_curves.ndraws; ) {
                    RendererRasterCurve* curve = &r->raster_curves.draws[c];
                    if (curve->npatches == 0) { c++; continue; }
                    if (cull_chunks &&
                        renderer_aabb_outside_vp(vp, curve->bounds_min, curve->bounds_max)) {
                        c++;
                        continue;
                    }

                    uint32_t run_patches = curve->npatches;
                    int next = c + 1;
                    uint64_t bytes_per_index =
                        (curve->index_type_bits == 16) ? 2u : 4u;
                    while (next < r->raster_curves.ndraws) {
                        RendererRasterCurve* n = &r->raster_curves.draws[next];
                        if (n->curve_id != curve->curve_id) break;
                        if (n->index_type_bits != curve->index_type_bits) break;
                        if (n->vertex_offset != curve->vertex_offset) break;
                        if (n->index_byte_offset !=
                            curve->index_byte_offset +
                            (uint64_t)run_patches * 4u * bytes_per_index) {
                            break;
                        }
                        if (cull_chunks &&
                            renderer_aabb_outside_vp(vp, n->bounds_min, n->bounds_max)) {
                            break;
                        }
                        run_patches += n->npatches;
                        next++;
                    }

                    float mvp[16];
                    for (int row = 0; row < 4; row++) {
                        for (int col = 0; col < 4; col++) {
                            float sum = 0.0f;
                            for (int k = 0; k < 4; k++)
                                sum += vp[row * 4 + k] * curve->model[k * 4 + col];
                            mvp[row * 4 + col] = sum;
                        }
                    }

                    GpuMeshPushConstants pc;
                    memset(&pc, 0, sizeof(pc));
                    memcpy(pc.mvp, mvp, sizeof(pc.mvp));
                    memcpy(pc.model, curve->model, sizeof(pc.model));
                    pc.color[0] = curve->color[0];
                    pc.color[1] = curve->color[1];
                    pc.color[2] = curve->color[2];
                    pc.color[3] = 0.0f; /* use per-CV raster curve colors */
                    memcpy(pc.eye_pos, eye, 3 * sizeof(float));
                    {
                        int32_t basis_id = curve->basis_id;
                        memcpy(&pc.eye_pos[3], &basis_id, sizeof(int32_t));
                    }
                    pc.ibl_params[0] = (float)r->width;
                    pc.ibl_params[1] = (float)r->height;
                    pc.tone_params[0] = r->tone_exposure_scale;
                    pc.tone_params[1] = r->tone_sky_scale;
                    pc.tone_params[2] = r->tone_white_point;
                    {
                        uint32_t tone_flags = r->tone_flags;
                        memcpy(&pc.tone_params[3], &tone_flags, sizeof(tone_flags));
                    }

                    gpu_cmd_push_constants(r->gpu, &pc, sizeof(pc));
                    gpu_cmd_bind_index_buffer_typed(
                        r->gpu,
                        r->raster_curves.index_buffer,
                        curve->index_byte_offset,
                        curve->index_type_bits);
                    gpu_cmd_draw_indexed(r->gpu,
                        run_patches * 4u,
                        0,
                        curve->vertex_offset);
                    c = next;
                }
            }
        }

        if (gpu_get_ibl_loaded(r->gpu)) {
            float view_inv[16];
            float proj_inv[16];
            memset(view_inv, 0, sizeof(view_inv));
            memset(proj_inv, 0, sizeof(proj_inv));

            view_inv[0] = view[0]; view_inv[1] = view[4]; view_inv[2]  = view[8];
            view_inv[4] = view[1]; view_inv[5] = view[5]; view_inv[6]  = view[9];
            view_inv[8] = view[2]; view_inv[9] = view[6]; view_inv[10] = view[10];
            view_inv[3]  = -(view_inv[0]*view[3]  + view_inv[1]*view[7]  + view_inv[2]*view[11]);
            view_inv[7]  = -(view_inv[4]*view[3]  + view_inv[5]*view[7]  + view_inv[6]*view[11]);
            view_inv[11] = -(view_inv[8]*view[3]  + view_inv[9]*view[7]  + view_inv[10]*view[11]);
            view_inv[15] = 1.0f;

            proj_inv[0]  = 1.0f / proj[0];
            proj_inv[5]  = 1.0f / proj[5];
            proj_inv[3]  = proj[2] / proj[0];
            proj_inv[7]  = proj[6] / proj[5];
            proj_inv[11] = 1.0f / proj[14];
            proj_inv[14] = -1.0f;
            proj_inv[15] = proj[10] / proj[14];

            gpu_draw_env_background(r->gpu, view_inv, proj_inv);
        }

        gpu_end_frame(r->gpu);
    }

    r->has_frame = 1;
    return NU_OK;
}

NuResult nu_fetch_pixels(NuRenderer* r, void* out_pixels, NuPixelFormat format)
{
    if (!r || !out_pixels) return NU_ERROR;
    if (!r->has_frame) {
        set_error(r, "No frame rendered yet");
        return NU_ERROR;
    }

    if (format == NU_PIXEL_RGBA8) {
        /* Direct readback: BGRA swapchain → RGBA8 in caller's buffer.
         * No temp file, no PPM encode/decode. */
        if (!gpu_readback_pixels(r->gpu, (uint8_t*)out_pixels,
                                 (uint32_t)r->width, (uint32_t)r->height,
                                 /*swizzle_to_rgba=*/1)) {
            set_error(r, "gpu_readback_pixels failed");
            return NU_ERROR;
        }
    } else if (format == NU_PIXEL_BGRA8) {
        /* Raw swapchain readback: skip the per-pixel BGRA→RGBA swizzle.
         * Caller gets bytes in the swapchain's native B,G,R,A order. */
        if (!gpu_readback_pixels(r->gpu, (uint8_t*)out_pixels,
                                 (uint32_t)r->width, (uint32_t)r->height,
                                 /*swizzle_to_rgba=*/0)) {
            set_error(r, "gpu_readback_pixels failed");
            return NU_ERROR;
        }
    } else {
        set_error(r, "Unsupported pixel format");
        return NU_ERROR;
    }

    return NU_OK;
}

NuResult nu_fetch_pixels_cuda(NuRenderer* r,
                              int* out_mem_fd,
                              uint64_t* out_size,
                              int* out_width,
                              int* out_height,
                              int* out_format)
{
    if (!r) return NU_ERROR;
    if (!r->has_frame) {
        set_error(r, "No frame rendered yet");
        return NU_ERROR;
    }
    if (!gpu_interop_available(r->gpu)) {
        set_error(r, "CUDA interop not available "
                     "(missing VK_KHR_external_memory_fd / external_semaphore_fd)");
        return NU_ERROR;
    }

    int fd = -1;
    uint64_t alloc_size = 0;
    uint64_t logical_size = 0;
    if (!gpu_readback_pixels_cuda(r->gpu,
                                   (uint32_t)r->width, (uint32_t)r->height,
                                   &fd, &alloc_size, &logical_size)) {
        set_error(r, "gpu_readback_pixels_cuda failed");
        return NU_ERROR;
    }

    if (out_mem_fd) *out_mem_fd = fd;
    /* Use the allocation size — that's what cuExternalMemoryGetMappedBuffer
     * needs to be ≤ for. The caller's payload is logical_size = w*h*4. */
    if (out_size)   *out_size   = alloc_size;
    if (out_width)  *out_width  = r->width;
    if (out_height) *out_height = r->height;
    if (out_format) *out_format = (int)NU_PIXEL_BGRA8;
    return NU_OK;
}

/* ---- Tiled multi-camera rendering ---- */

static int ensure_tiled_pipeline(NuRenderer* r)
{
    if (r->tiled_pipeline_built && gpu_tiled_rt_pipeline_built(r->gpu)) return 1;
    r->tiled_pipeline_built = 0;
    if (!gpu_rt_available(r->gpu) || !gpu_rt_built(r->gpu)) return 0;

    /* Load tiled ray gen shader + reuse standard miss/chit.
     *
     * SER (VK_NV_ray_tracing_invocation_reorder): when the device supports
     * SER, prefer the SER-enabled raygen variant which uses hitObjectNV +
     * reorderThreadNV to cluster threads by hit instance id before invoking
     * the closest-hit shader. Win comes from coherent material lookup +
     * texture indexing in the shared rchit, since the scene has only a
     * single CHS (no SBT-table dispatch divergence).
     *
     * The non-SER variant remains the fallback so non-NV / pre-Ada GPUs
     * still run with byte-identical output to the legacy path. */
    char rgen_path[1024], rmiss_path[1024], rchit_path[1024];
    char rint_path[1024], rchit_curve_path[1024];
    int use_ser = gpu_ser_available(r->gpu) && !getenv("NUSD_DISABLE_SER");
    if (use_ser) {
        snprintf(rgen_path, sizeof(rgen_path), "%s/raytrace_tiled_ser.rgen.spv", SHADER_DIR);
    } else {
        snprintf(rgen_path, sizeof(rgen_path), "%s/raytrace_tiled.rgen.spv", SHADER_DIR);
    }
    snprintf(rmiss_path, sizeof(rmiss_path), "%s/raytrace.rmiss.spv", SHADER_DIR);
    snprintf(rchit_path, sizeof(rchit_path), "%s/raytrace.rchit.spv", SHADER_DIR);
    snprintf(rint_path,  sizeof(rint_path),  "%s/raytrace_curve.rint.spv",  SHADER_DIR);
    snprintf(rchit_curve_path, sizeof(rchit_curve_path), "%s/raytrace_curve.rchit.spv", SHADER_DIR);
    fprintf(stderr, "renderer: tiled RT raygen = %s%s\n",
            use_ser ? "raytrace_tiled_ser.rgen.spv" : "raytrace_tiled.rgen.spv",
            use_ser ? " (SER on)" : " (SER off)");

    uint32_t rgen_sz = 0, rmiss_sz = 0, rchit_sz = 0;
    uint32_t rint_sz = 0, rchit_curve_sz = 0;
    uint32_t* rgen_spv  = load_shader_file(rgen_path, &rgen_sz);
    uint32_t* rmiss_spv = load_shader_file(rmiss_path, &rmiss_sz);
    uint32_t* rchit_spv = load_shader_file(rchit_path, &rchit_sz);
    /* Phase 11.A.2.5b: optional curve shaders for tiled IsaacLab path. */
    uint32_t* rint_spv        = (r->curves.nseg > 0) ? load_shader_file(rint_path, &rint_sz) : NULL;
    uint32_t* rchit_curve_spv = (r->curves.nseg > 0) ? load_shader_file(rchit_curve_path, &rchit_curve_sz) : NULL;

    if (!rgen_spv || !rmiss_spv || !rchit_spv) {
        free(rgen_spv); free(rmiss_spv); free(rchit_spv);
        free(rint_spv); free(rchit_curve_spv);
        set_error(r, "Failed to load tiled RT shaders");
        return 0;
    }
    if (r->curves.nseg > 0 && (!rint_spv || !rchit_curve_spv)) {
        free(rgen_spv); free(rmiss_spv); free(rchit_spv);
        free(rint_spv); free(rchit_curve_spv);
        set_error(r, "Failed to load curve RT shaders for tiled pipeline");
        return 0;
    }

    int ok = gpu_build_tiled_rt_pipeline(r->gpu,
        rgen_spv, rgen_sz, rmiss_spv, rmiss_sz, rchit_spv, rchit_sz,
        rint_spv, rint_sz, rchit_curve_spv, rchit_curve_sz);

    free(rgen_spv);
    free(rmiss_spv);
    free(rchit_spv);
    free(rint_spv);
    free(rchit_curve_spv);

    if (!ok) {
        set_error(r, "Failed to build tiled RT pipeline");
        return 0;
    }

    r->tiled_pipeline_built = 1;
    return 1;
}

/* Phase B: lazy-build the deferred-shading compute pipeline. Only invoked
 * when r->deferred_shade_enabled flips from 0->1; subsequent toggles reuse
 * the cached pipeline. */
static int ensure_deferred_pipeline(NuRenderer* r)
{
    if (r->deferred_pipeline_built) return 1;
    if (!gpu_rt_available(r->gpu)) return 0;

    char comp_path[1024];
    snprintf(comp_path, sizeof(comp_path), "%s/deferred_shade.comp.spv", SHADER_DIR);

    uint32_t comp_sz = 0;
    uint32_t* comp_spv = load_shader_file(comp_path, &comp_sz);
    if (!comp_spv) {
        set_error(r, "Failed to load deferred_shade.comp.spv");
        return 0;
    }

    int ok = gpu_build_deferred_pipeline(r->gpu, comp_spv, comp_sz);
    free(comp_spv);

    if (!ok) {
        set_error(r, "Failed to build deferred-shading compute pipeline");
        return 0;
    }
    r->deferred_pipeline_built = 1;
    return 1;
}

NuResult nu_render_tiled(NuRenderer* r,
                         const float* vp_inv_matrices, int num_cameras,
                         int tile_w, int tile_h, NuRenderMode mode)
{
    if (!r || !vp_inv_matrices || num_cameras <= 0) return NU_ERROR;
    if (mode != NU_RENDER_RT) {
        set_error(r, "Tiled rendering only supports NU_RENDER_RT mode");
        return NU_ERROR;
    }

    /* NU_FRAME_SKIP probe: only render every Nth call. On skipped frames
     * return NU_OK immediately without touching any GPU state, so the
     * interop/staging slot ping-pong stays put and downstream readers see
     * the prior rendered frame again. Dirty bits accumulate and are
     * processed on the next non-skipped frame.
     *
     * Guarded by r->has_frame so the very first call always renders; this
     * keeps fetch_*_tiled / nu_fetch_async happy on the first frame, and
     * keeps the tiled cmd-buffer cache primed.
     *
     * Default (unset/=1) is byte-identical to pre-probe behaviour. */
    int skip_n = frame_skip_active();
    if (skip_n > 1 && r->has_frame) {
        unsigned long fc = g_frame_count++;
        if ((fc % (unsigned long)skip_n) != 0) {
            return NU_OK;
        }
    }

    /* Save the caller's UNSCALED tile dims for output-stride push constants
     * (per-env layout uses unscaled stride to match the caller's buffer
     * layout — caller doesn't know about NU_TILE_RES scaling). */
    int tile_w_orig = tile_w;
    int tile_h_orig = tile_h;

    /* NU_TILE_RES probe: scale tile dims uniformly. No-op when env unset/=64. */
    tile_w = effective_tile(tile_w);
    tile_h = effective_tile(tile_h);

    /* Rebuild GPU buffers if scene changed */
    if (r->scene_dirty) {
        if (!rebuild_gpu_buffers(r)) return NU_ERROR;
    }

    /* Phase 11.A.2.5: a curve-only scene has total_indices==0 but is
     * still renderable via the curve hit-group. Match the gate from
     * nu_render so curve-only scenes (e.g. /tmp/grid_tera.usdc) work
     * through the tiled / async paths. */
    if (r->total_indices == 0 && r->curves.nseg == 0) {
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
    uint8_t* vis = NULL;
    int tlas_needs_update = r->tlas_dirty;
    if (tlas_needs_update) {
        xforms_3x4 = (float*)malloc(r->nmeshes * 12 * sizeof(float));
        vis = (uint8_t*)malloc(r->nmeshes);
        if (xforms_3x4) {
            for (int m = 0; m < r->nmeshes; m++) {
                const float* src = r->meshes[m].transform;
                float* dst = &xforms_3x4[m * 12];
                dst[0] = src[0]; dst[1] = src[1]; dst[2]  = src[2];  dst[3]  = src[3];
                dst[4] = src[4]; dst[5] = src[5]; dst[6]  = src[6];  dst[7]  = src[7];
                dst[8] = src[8]; dst[9] = src[9]; dst[10] = src[10]; dst[11] = src[11];
            }
        }
        if (vis) {
            for (int m = 0; m < r->nmeshes; m++)
                vis[m] = r->meshes[m].visible ? 1 : 0;
        }
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
        free(xforms_3x4); free(vis);
        return NU_ERROR;
    }

    /* Upload camera matrices */
    if (!gpu_tiled_upload_cameras(r->gpu, vp_inv_matrices, num_cameras)) {
        set_error(r, "Failed to upload camera matrices");
        free(xforms_3x4); free(vis);
        return NU_ERROR;
    }

    /* Lazily build tiled RT pipeline */
    if (!ensure_tiled_pipeline(r)) { free(xforms_3x4); free(vis); return NU_ERROR; }

    /* Phase B: when deferred shading is on, ensure (a) the per-launch
     * tiled G-buffer is allocated + bound at binding 17 (replaces the
     * Phase A scene_data_buf stub) and (b) the deferred-shading compute
     * pipeline is built. If allocation fails, silently fall back to the
     * legacy color path so we never leave the rchit's gate flipped while
     * the buffer is still a stub. */
    if (r->deferred_shade_enabled) {
        if (!gpu_tiled_ensure_gbuffer(r->gpu)) {
            fprintf(stderr,
                "[nu_renderer] tiled G-buffer alloc failed — disabling deferred-shade for this frame\n");
            r->deferred_shade_enabled = 0;
            gpu_invalidate_tiled_cmd_cache(r->gpu);
        } else if (!ensure_deferred_pipeline(r)) {
            fprintf(stderr,
                "[nu_renderer] deferred-shade pipeline build failed — disabling for this frame\n");
            r->deferred_shade_enabled = 0;
            gpu_invalidate_tiled_cmd_cache(r->gpu);
        }
    }

    /* Render — open command buffer */
    if (!gpu_begin_frame_tiled_rt(r->gpu)) {
        set_error(r, "gpu_begin_frame_tiled_rt failed");
        free(xforms_3x4);
        free(vis);
        return NU_ERROR;
    }

    /* PR 2: GPU-driven TLAS update — record compute dispatch + TLAS
     * build inline, no host-side data needed. xforms_3x4 was not
     * populated on this path (tlas_dirty was cleared by nu_translate_
     * instances_gpu). */
    if (gpu_transforms_ready(r)) {
        if (!gpu_translate_instances_inline(r->gpu, r->gpu_transform_count)) {
            /* Pipeline failed to record — log + fall through to host path
             * if possible. The GPU-driven flag stays armed for next frame. */
            fprintf(stderr,
                    "[nu_renderer] gpu_translate_instances_inline failed; "
                    "rendering with stale TLAS this frame.\n");
        }
    }
    /* TLAS update: try inline (same command buffer) to avoid a separate
     * vkQueueSubmit + vkQueueWaitIdle stall. Falls back to synchronous. */
    if (xforms_3x4) {
        if (!gpu_update_tlas_inline(r->gpu, xforms_3x4, vis, (uint32_t)r->nmeshes)) {
            /* Inline failed (no persistent staging/scratch) — must do it
             * synchronously. Abort the current frame, do the TLAS update,
             * then reopen a new frame command buffer. */
            gpu_abort_frame_tiled_rt(r->gpu);
            gpu_update_tlas(r->gpu, xforms_3x4, vis, (uint32_t)r->nmeshes);

            if (!gpu_begin_frame_tiled_rt(r->gpu)) {
                set_error(r, "gpu_begin_frame_tiled_rt failed after TLAS update");
                free(xforms_3x4);
                free(vis);
                return NU_ERROR;
            }
        }
        free(xforms_3x4);
        free(vis);
    }

    GpuRtTiledPushConstants pc;
    memset(&pc, 0, sizeof(pc));
    pc.tile_w      = (uint32_t)tile_w;
    pc.tile_h      = (uint32_t)tile_h;
    pc.num_cols    = (uint32_t)num_cols;
    pc.num_cameras = (uint32_t)num_cameras;
    renderer_rt_scene_metrics(r, &pc.ground_y, &pc.scene_scale);
    pc.fast_mode              = (uint32_t)r->fast_mode;
    pc.depth_enabled          = (uint32_t)r->depth_enabled;
    pc.segmentation_enabled   = (uint32_t)r->segmentation_enabled;
    pc.normals_enabled        = (uint32_t)r->normals_enabled;
    pc.deferred_shade_enabled = (uint32_t)r->deferred_shade_enabled;
    pc.tone_exposure_scale    = r->tone_exposure_scale;
    pc.tone_sky_scale         = r->tone_sky_scale;
    pc.tone_white_point       = r->tone_white_point;
    pc.tone_flags             = r->tone_flags;
    pc.rt_ibl_fill_scale      = r->rt_ibl_fill_scale;

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
    /* _pad[3] at offset 28 maps to viewInverse[1][3]: Phase C.2 deferred
     * debug-mode selector. The rgen does not consume this slot, so the
     * write is invisible to the RT pipeline; the compute shader decodes it
     * via floatBitsToUint(viewInverse[1][3]). */
    if (r->deferred_debug_mode != 0u) {
        memcpy(&pc._pad[3], &r->deferred_debug_mode, sizeof(uint32_t));
    }

    /* _pad[4..5] at offset 32-39 (viewInverse[2][0..1]): NU_TILE_RES
     * output-stride override. The rgen uses these as the per-env stride for
     * the direct-write SSBO when usePerEnvLayout != 0. When NU_TILE_RES is
     * unset / =64 we set them to tile_w/tile_h (no behavior change).  When
     * NU_TILE_RES != 64 we set them to the UNSCALED dims so the per-env
     * stride matches the caller's buffer layout (the caller pre-allocates
     * at the unscaled dims they passed before effective_tile() scaled them
     * down). The rgen still only fills (tile_w * tile_h) pixels per env
     * slot; the padding region is opaque-black via the host's per-frame
     * vkCmdFillBuffer in gpu_cmd_trace_rays_tiled. */
    {
        const uint32_t out_tile_w = (uint32_t)tile_w_orig;
        const uint32_t out_tile_h = (uint32_t)tile_h_orig;
        memcpy(&pc._pad[4], &out_tile_w, sizeof(uint32_t));
        memcpy(&pc._pad[5], &out_tile_h, sizeof(uint32_t));
    }

    /* _pad[6] at offset 40 (viewInverse[2][2]): diagnostic linear-radiance probe
     * gain. >0 makes the deferred compute shader emit pre-tonemap radiance.
     * The rgen ignores this slot; only deferred_shade.comp consumes it. */
    if (r->deferred_linear_probe_gain > 0.0f) {
        memcpy(&pc._pad[6], &r->deferred_linear_probe_gain, sizeof(float));
    }

    gpu_cmd_trace_rays_tiled(r->gpu, &pc);

    /* Phase B: deferred-shading compute pass. The dispatch reads the
     * G-buffer (binding 17) the rchit/rmiss just populated and overwrites
     * the rgen's pixels in binding 9 with flat-shaded per-mesh colors. The
     * helper is internally guarded against the cached-cmd-buffer fast path
     * (a no-op on replay), so we always invoke it when the flag is on. */
    if (r->deferred_shade_enabled) {
        gpu_cmd_deferred_shade(r->gpu, &pc);
    }

    gpu_end_frame_tiled_rt(r->gpu);

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

    /* NU_TILE_RES probe: scale tile dims uniformly. No-op when env unset/=64. */
    tile_w = effective_tile(tile_w);
    tile_h = effective_tile(tile_h);

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

    /* NU_TILE_RES probe: scale tile dims uniformly. No-op when env unset/=64. */
    tile_w = effective_tile(tile_w);
    tile_h = effective_tile(tile_h);

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

    /* NU_TILE_RES probe: scale tile dims uniformly. No-op when env unset/=64. */
    tile_w = effective_tile(tile_w);
    tile_h = effective_tile(tile_h);

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

    /* NU_TILE_RES probe: scale tile dims uniformly. No-op when env unset/=64. */
    tile_w = effective_tile(tile_w);
    tile_h = effective_tile(tile_h);

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

    /* NU_TILE_RES probe: scale tile dims uniformly. No-op when env unset/=64. */
    tile_w = effective_tile(tile_w);
    tile_h = effective_tile(tile_h);

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
        set_error(r, "No previous tiled readback in flight");
        return NU_ERROR;
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

int nu_load_texture(NuRenderer* r, const uint8_t* pixels, int width, int height)
{
    if (!r || !pixels || width <= 0 || height <= 0) return NU_ERROR;
    if (!check_scene_mutable(r, "nu_load_texture")) return NU_ERROR;

    /* Grow runtime texture array. */
    if (r->runtime_textures_count >= r->runtime_textures_capacity) {
        int new_cap = r->runtime_textures_capacity ? r->runtime_textures_capacity * 2 : 16;
        void* p = realloc(r->runtime_textures, (size_t)new_cap * sizeof(*r->runtime_textures));
        if (!p) return NU_ERROR;
        r->runtime_textures = p;
        r->runtime_textures_capacity = new_cap;
    }

    /* Copy the texture pixels — caller may free `pixels` after the call. */
    size_t nbytes = (size_t)width * (size_t)height * 4;
    unsigned char* copy = (unsigned char*)malloc(nbytes);
    if (!copy) return NU_ERROR;
    memcpy(copy, pixels, nbytes);

    int idx = r->runtime_textures_count++;
    r->runtime_textures[idx].pixels = copy;
    r->runtime_textures[idx].width  = width;
    r->runtime_textures[idx].height = height;
    /* New textures invalidate any prior commit — gpu_upload_materials must
     * be re-run before rendering to honor the updated array size. */
    r->runtime_textures_committed = 0;
    /* Tearing down the RT pipeline forces a rebuild that picks up the
     * new mat_tex_count for descriptor sizing. Cheap if RT scene wasn't
     * built yet. */
    r->scene_dirty = 1;
    r->accel_dirty = 1;
    return idx;
}

NuResult nu_set_mesh_texture(NuRenderer* r, int mesh_id, int tex_index)
{
    if (!r) return NU_ERROR;
    if (mesh_id < 0 || mesh_id >= r->nmeshes) return NU_ERROR_BAD_ID;
    /* Allow tex_index < 0 → revert to flat color (sentinel). */
    uint32_t encoded = (tex_index < 0) ? 0xFFFFFFFFu : (uint32_t)tex_index;
    if (r->gpu) gpu_set_mesh_texture(r->gpu, (uint32_t)mesh_id, encoded);
    return NU_OK;
}

NuResult nu_set_dome_color(NuRenderer* r, float r_, float g, float b,
                           float intensity)
{
    if (!r) return NU_ERROR;
    /* Clamp negatives to zero — over-bright (>1) is allowed but the rmiss
     * shader applies no tone-map in the fast_mode path so callers should
     * keep intensity * color in [0,1] for sane sky output. */
    r->dome_color[0] = (r_  > 0.0f) ? r_  : 0.0f;
    r->dome_color[1] = (g   > 0.0f) ? g   : 0.0f;
    r->dome_color[2] = (b   > 0.0f) ? b   : 0.0f;
    r->dome_color[3] = (intensity > 0.0f) ? intensity : 0.0f;
    r->dome_color_dirty = 1;
    /* Push to gpu_vulkan so the SSBO header is patched (or stashed for the
     * next gpu_build_rt_scene). Cheap (16-byte vkCmdUpdateBuffer). */
    if (r->gpu) {
        gpu_set_dome_color(r->gpu,
                           r->dome_color[0], r->dome_color[1],
                           r->dome_color[2], r->dome_color[3]);
    }
    return NU_OK;
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
    if (!r) return;
    int prev = r->deferred_shade_enabled;
    r->deferred_shade_enabled = enable ? 1 : 0;
    if (prev != r->deferred_shade_enabled) {
        /* Toggling the flag changes which descriptor set bindings the
         * cached tiled cmd buffer references and flips the rchit/rmiss
         * G-buffer write gate; the cached cmd buffer must be rebuilt. */
        gpu_invalidate_tiled_cmd_cache(r->gpu);
    }
}

void nu_set_deferred_debug_mode(NuRenderer* r, uint32_t mode)
{
    if (!r) return;
    uint32_t prev = r->deferred_debug_mode;
    r->deferred_debug_mode = mode;
    if (prev != r->deferred_debug_mode) {
        /* The push-constant payload differs by one uint, so the cached
         * tiled cmd buffer (which baked the previous push constants
         * into vkCmdPushConstants) must be re-recorded. */
        gpu_invalidate_tiled_cmd_cache(r->gpu);
    }
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

    /* NU_TILE_RES probe: scale tile dims uniformly. No-op when env unset/=64. */
    tile_w = effective_tile(tile_w);
    tile_h = effective_tile(tile_h);

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
        return NU_ERROR;
    }

    /* NU_TILE_RES probe: scale tile dims uniformly. The struct's
     * tile_w/tile_h/image_w/image_h reported back to the caller reflect the
     * SCALED values so the CUDA importer learns the actual buffer geometry. */
    tile_w = effective_tile(tile_w);
    tile_h = effective_tile(tile_h);

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
    if (!r || !mem_fds) return NU_ERROR;
    if (!gpu_interop_available(r->gpu)) {
        set_error(r, "CUDA interop not available "
                     "(missing VK_KHR_external_memory_fd / external_semaphore_fd)");
        return NU_ERROR;
    }
    if (num_cameras <= 0 || tile_w <= 0 || tile_h <= 0) {
        set_error(r, "nu_set_external_output_buffers: invalid dimensions");
        return NU_ERROR;
    }

    /* NU_TILE_RES probe: scale tile dims uniformly. The caller-provided
     * mem_size_each must accommodate the SCALED footprint (32x32 fits inside
     * a 64x64-allocated buffer; the larger pre-existing alloc is fine, the
     * renderer just writes a smaller subregion). */
    tile_w = effective_tile(tile_w);
    tile_h = effective_tile(tile_h);

    int num_cols = (int)ceilf(sqrtf((float)num_cameras));
    int num_rows = (num_cameras + num_cols - 1) / num_cols;
    uint32_t total_w = (uint32_t)(num_cols * tile_w);
    uint32_t total_h = (uint32_t)(num_rows * tile_h);

    if (!gpu_set_external_output_buffers(r->gpu, mem_fds, mem_size_each, total_w, total_h)) {
        set_error(r, "gpu_set_external_output_buffers failed");
        return NU_ERROR;
    }
    return NU_OK;
}

NuResult nu_get_external_timeline_semaphore_fd(NuRenderer* r,
                                                int* out_sem_fd,
                                                uint64_t* out_sem_value)
{
    if (!r || !out_sem_fd || !out_sem_value) return NU_ERROR;
    if (!gpu_interop_available(r->gpu)) {
        set_error(r, "CUDA interop not available");
        return NU_ERROR;
    }
    int fd = gpu_export_timeline_semaphore_fd(r->gpu);
    if (fd < 0) {
        set_error(r, "gpu_export_timeline_semaphore_fd failed");
        return NU_ERROR;
    }
    *out_sem_fd    = fd;
    *out_sem_value = gpu_get_interop_timeline_value(r->gpu);
    return NU_OK;
}

/* ---- PR 2: GPU-driven TLAS instance translation ---- */

NuResult nu_get_transforms_interop_info(NuRenderer* r, int count,
                                        NuTransformsInteropInfo* out)
{
    if (!r || !out || count <= 0) return NU_ERROR;
    if (!gpu_interop_available(r->gpu)) {
        set_error(r, "CUDA interop not available");
        return NU_ERROR;
    }

    if (!gpu_create_tlas_xforms_buffer(r->gpu, count)) {
        set_error(r, "gpu_create_tlas_xforms_buffer failed");
        return NU_ERROR;
    }

    int fd = gpu_export_tlas_xforms_fd(r->gpu);
    if (fd < 0) {
        set_error(r, "gpu_export_tlas_xforms_fd failed");
        return NU_ERROR;
    }

    out->mem_fd   = fd;
    out->mem_size = gpu_get_tlas_xforms_size(r->gpu);
    out->count    = count;
    return NU_OK;
}

NuResult nu_set_transform_layout(NuRenderer* r, const int* mesh_ids, int count)
{
    if (!r || !mesh_ids || count <= 0) return NU_ERROR;

    /* Make sure the TLAS is built — instance_custom is only populated
     * after rebuild_accel runs. */
    if (r->scene_dirty) {
        if (!rebuild_gpu_buffers(r)) return NU_ERROR;
    }
    if (r->accel_dirty) {
        if (!rebuild_accel(r)) return NU_ERROR;
    }

    /* Invert (tlas_idx → mesh_id) into (mesh_id → tlas_idx). */
    int* mesh_to_tlas = (int*)malloc((size_t)r->nmeshes * sizeof(int));
    if (!mesh_to_tlas) return NU_ERROR;
    if (!gpu_get_mesh_to_tlas_idx(r->gpu, mesh_to_tlas, r->nmeshes)) {
        set_error(r, "gpu_get_mesh_to_tlas_idx failed");
        free(mesh_to_tlas);
        return NU_ERROR;
    }

    /* Translate caller's mesh_ids[gid] into tlas_idx[gid]. */
    int* tlas_idx = (int*)malloc((size_t)count * sizeof(int));
    if (!tlas_idx) { free(mesh_to_tlas); return NU_ERROR; }
    for (int i = 0; i < count; i++) {
        int mid = mesh_ids[i];
        if (mid < 0 || mid >= r->nmeshes) {
            tlas_idx[i] = -1;
        } else {
            tlas_idx[i] = mesh_to_tlas[mid];
        }
    }
    free(mesh_to_tlas);

    int ok = gpu_set_transform_layout(r->gpu, tlas_idx, count);
    free(tlas_idx);
    if (!ok) {
        set_error(r, "gpu_set_transform_layout failed");
        return NU_ERROR;
    }
    r->gpu_transform_count = count;
    return NU_OK;
}

NuResult nu_translate_instances_gpu(NuRenderer* r)
{
    /* Sentinel call from Python: arms the next nu_render_tiled to take
     * the GPU-driven path (compute dispatch + TLAS build) instead of
     * the host-loop path. Clearing tlas_dirty also disables the host-side
     * memcpy of r->meshes[].transform → xforms_3x4 in nu_render_tiled,
     * since the imported buffer already holds the authoritative data.
     *
     * The actual dispatch is recorded inside nu_render_tiled, after
     * gpu_begin_frame_tiled_rt(). Returning NU_OK here means the flag is
     * armed; the next render_tiled call will consume it. */
    if (!r) return NU_ERROR;
    r->tlas_dirty = 0;
    r->use_gpu_transforms = 1;
    return NU_OK;
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
        float* xforms_3x4 = (float*)malloc(r->nmeshes * 12 * sizeof(float));
        if (xforms_3x4) {
            for (int m = 0; m < r->nmeshes; m++) {
                const float* src = r->meshes[m].transform;
                float* dst = &xforms_3x4[m * 12];
                dst[0] = src[0]; dst[1] = src[1]; dst[2]  = src[2];  dst[3]  = src[3];
                dst[4] = src[4]; dst[5] = src[5]; dst[6]  = src[6];  dst[7]  = src[7];
                dst[8] = src[8]; dst[9] = src[9]; dst[10] = src[10]; dst[11] = src[11];
            }
            uint8_t* vis = (uint8_t*)malloc(r->nmeshes);
            if (vis) {
                for (int m = 0; m < r->nmeshes; m++)
                    vis[m] = r->meshes[m].visible ? 1 : 0;
            }
            gpu_update_tlas(r->gpu, xforms_3x4, vis, (uint32_t)r->nmeshes);
            free(vis);
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
                       out_distances, NULL, out_normals, out_hit_positions)) {
        set_error(r, "gpu_cast_rays failed");
        return NU_ERROR;
    }

    return NU_OK;
}

NuResult nu_cast_rays_with_ids(NuRenderer* r,
                               const float* ray_origins,
                               const float* ray_directions,
                               int num_rays,
                               float max_distance,
                               float* out_distances,
                               int* out_mesh_ids,
                               float* out_normals,
                               float* out_hit_positions)
{
    if (!r) return NU_ERROR;
    if (num_rays <= 0) return NU_OK;
    if (!ray_origins || !ray_directions || !out_distances ||
        !out_mesh_ids || !out_normals || !out_hit_positions) {
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
        float* xforms_3x4 = (float*)malloc(r->nmeshes * 12 * sizeof(float));
        if (xforms_3x4) {
            for (int m = 0; m < r->nmeshes; m++) {
                const float* src = r->meshes[m].transform;
                float* dst = &xforms_3x4[m * 12];
                dst[0] = src[0]; dst[1] = src[1]; dst[2]  = src[2];  dst[3]  = src[3];
                dst[4] = src[4]; dst[5] = src[5]; dst[6]  = src[6];  dst[7]  = src[7];
                dst[8] = src[8]; dst[9] = src[9]; dst[10] = src[10]; dst[11] = src[11];
            }
            uint8_t* vis = (uint8_t*)malloc(r->nmeshes);
            if (vis) {
                for (int m = 0; m < r->nmeshes; m++)
                    vis[m] = r->meshes[m].visible ? 1 : 0;
            }
            gpu_update_tlas(r->gpu, xforms_3x4, vis, (uint32_t)r->nmeshes);
            free(vis);
            free(xforms_3x4);
        }
        r->tlas_dirty = 0;
    }

    if (!ensure_raycast_pipeline(r)) {
        set_error(r, "Raycast pipeline not available (RT required)");
        return NU_ERROR_NO_RT;
    }

    if (!gpu_cast_rays(r->gpu, ray_origins, ray_directions,
                       num_rays, max_distance,
                       out_distances, out_mesh_ids, out_normals,
                       out_hit_positions)) {
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
        float* xforms_3x4 = (float*)malloc(r->nmeshes * 12 * sizeof(float));
        if (xforms_3x4) {
            for (int m = 0; m < r->nmeshes; m++) {
                const float* src = r->meshes[m].transform;
                float* dst = &xforms_3x4[m * 12];
                dst[0] = src[0]; dst[1] = src[1]; dst[2]  = src[2];  dst[3]  = src[3];
                dst[4] = src[4]; dst[5] = src[5]; dst[6]  = src[6];  dst[7]  = src[7];
                dst[8] = src[8]; dst[9] = src[9]; dst[10] = src[10]; dst[11] = src[11];
            }
            uint8_t* vis = (uint8_t*)malloc(r->nmeshes);
            if (vis) {
                for (int m = 0; m < r->nmeshes; m++)
                    vis[m] = r->meshes[m].visible ? 1 : 0;
            }
            gpu_update_tlas(r->gpu, xforms_3x4, vis, (uint32_t)r->nmeshes);
            free(vis);
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

/* ---- Async double-buffered render+fetch ----
 *
 * Wraps nu_render_tiled with num_cameras=1 to ride the existing async infra:
 * 2 staging buffers, 2 fences, 2 command buffers, write_idx ping-pong. The
 * first fetch_async has nothing to read (no previous frame submitted) and
 * returns a zero-filled buffer.
 */

static void compute_camera_inverses(const Camera* cam, float vp_inv[32])
{
    /* Same math as nu_render's RT path — extracted into a helper so the async
     * path can pack a 32-float (view_inv[16] + proj_inv[16]) entry for the
     * tiled raygen shader's camera SSBO. */
    float v[16];
    camera_get_view(cam, v);
    float vi[16];
    memset(vi, 0, sizeof(vi));
    /* View inverse: transpose rotation, negate-translated eye */
    vi[0] = v[0]; vi[1] = v[4]; vi[2]  = v[8];
    vi[4] = v[1]; vi[5] = v[5]; vi[6]  = v[9];
    vi[8] = v[2]; vi[9] = v[6]; vi[10] = v[10];
    vi[3]  = -(vi[0]*v[3]  + vi[1]*v[7]  + vi[2]*v[11]);
    vi[7]  = -(vi[4]*v[3]  + vi[5]*v[7]  + vi[6]*v[11]);
    vi[11] = -(vi[8]*v[3]  + vi[9]*v[7]  + vi[10]*v[11]);
    vi[15] = 1.0f;

    float p[16];
    camera_get_proj(cam, p);
    float pi[16];
    memset(pi, 0, sizeof(pi));
    /* Projection inverse, including off-axis lens shift. */
    pi[0]  = 1.0f / p[0];
    pi[5]  = 1.0f / p[5];
    pi[3]  = p[2] / p[0];
    pi[7]  = p[6] / p[5];
    pi[11] = 1.0f / p[14];
    pi[14] = -1.0f;
    pi[15] = p[10] / p[14];

    memcpy(vp_inv,        vi, sizeof(vi));
    memcpy(vp_inv + 16,   pi, sizeof(pi));
}

NuResult nu_render_async(NuRenderer* r)
{
    if (!r) return NU_ERROR;

    /* Pack camera inverses for the tiled raygen shader's camera SSBO
     * (which expects num_cameras pairs of view_inv[16] + proj_inv[16]). */
    float vp_inv[32];
    compute_camera_inverses(&r->camera, vp_inv);

    /* Reuse the existing tiled path with num_cameras=1. This inherits:
     *   - double-buffered staging + fences + command buffers
     *   - non-blocking submit (vkQueueSubmit + VkFence, no WaitIdle)
     *   - GENERAL-layout tiled image (no per-frame layout transitions on the
     *     image; only barriers on the SSBO/staging copies)
     *   - curve-only support (gate relaxed above) */
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

/* ---- Configuration ---- */

NuResult nu_set_render_size(NuRenderer* r, int width, int height)
{
    if (!r || width <= 0 || height <= 0) return NU_ERROR;

    r->width = width;
    r->height = height;
    /* Resize the underlying GLFW window so its Vulkan surface caps allow
     * the new size — without this, gpu_resize recreates the swapchain at
     * the OLD framebuffer extent (clamped by the surface caps), and
     * gpu_readback_pixels rejects the request because
     * gpu->swapchain_extent never changes. The window stays hidden when
     * config.visible was 0 — only its framebuffer dimensions change. */
    if (r->window) {
        glfwSetWindowSize((GLFWwindow*)r->window, width, height);
        /* GLFW may queue the size change on some platforms; a poll lets
         * the window manager apply it before gpu_resize queries caps. */
        glfwPollEvents();
    }
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
    return r ? (void*)r->window : NULL;
}

/* ---- Environment ---- */

NuResult nu_load_environment(NuRenderer* r, const char* hdr_path)
{
    float intensity = (r && r->fallback_lights_active) ? -1500.0f : -1.0f;
    return nu_load_environment_intensity(r, hdr_path, intensity);
}

NuResult nu_load_environment_intensity(NuRenderer* r, const char* hdr_path,
                                       float intensity)
{
    if (!r || !hdr_path) return NU_ERROR;

    /* gpu_load_environment requires the material descriptor set to exist
     * (it writes the env-map binding into it). Upload a placeholder
     * material set so the descriptor set is alive when callers invoke
     * before nu_load_usd. */
    if (!gpu_materials_uploaded(r->gpu))
        gpu_upload_materials(r->gpu, NULL, 0, NULL, 0);

    if (gpu_load_environment_intensity(r->gpu, hdr_path, intensity)) {
        /* Phase C.3: the deferred-shading compute pipeline's descriptor
         * set layout is built once (gpu_build_deferred_pipeline) and reads
         * `gpu->env_image_view != NULL` to decide whether to declare
         * bindings 5/6/7. If the env loads AFTER the pipeline is already
         * built, the cached layout omits the IBL bindings even though the
         * SPIR-V references envMap/irrMap when scene.envMipLevels > 0.0.
         * Tear down the pipeline so the next ensure_deferred_pipeline()
         * call rebuilds it with the IBL bindings present. The teardown
         * is a no-op when the pipeline hasn't been built yet (typical
         * nu_load_usd → gpu_load_environment ordering during scene load). */
        if (r->deferred_pipeline_built) {
            gpu_destroy_deferred_pipeline(r->gpu);
            r->deferred_pipeline_built = 0;
            gpu_invalidate_tiled_cmd_cache(r->gpu);
        }
        return NU_OK;
    }

    set_error(r, "Failed to load environment map");
    return NU_ERROR;
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

int nu_get_curve_segment_count(NuRenderer* r)
{
    return r ? r->curves.nseg : 0;
}

NuResult nu_get_scene_bounds(NuRenderer* r, float out_min[3], float out_max[3])
{
    if (!r || !out_min || !out_max) return NU_ERROR;
    if (r->scene_bounds_valid) {
        out_min[0] = r->scene_bounds_min[0];
        out_min[1] = r->scene_bounds_min[1];
        out_min[2] = r->scene_bounds_min[2];
        out_max[0] = r->scene_bounds_max[0];
        out_max[1] = r->scene_bounds_max[1];
        out_max[2] = r->scene_bounds_max[2];
        return NU_OK;
    }
    Scene* scene = (Scene*)r->usd_scene;
    if (!scene) return NU_ERROR;
    out_min[0] = scene->bounds_min[0];
    out_min[1] = scene->bounds_min[1];
    out_min[2] = scene->bounds_min[2];
    out_max[0] = scene->bounds_max[0];
    out_max[1] = scene->bounds_max[1];
    out_max[2] = scene->bounds_max[2];
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
    memcpy(out_mat4x4, r->meshes[mesh_id].transform, 16 * sizeof(float));
    return NU_OK;
}

int nu_get_mesh_material_index(NuRenderer* r, int mesh_id)
{
    if (!r) return NU_ERROR;
    if (mesh_id < 0 || mesh_id >= r->nmeshes) return NU_ERROR_BAD_ID;
    if (!r->meshes[mesh_id].active) return NU_ERROR_BAD_ID;
    return r->meshes[mesh_id].material_index;
}

int nu_get_material_name(NuRenderer* r, int material_index, char* out_buf, int buf_cap)
{
    if (!r) return NU_ERROR;
    if (material_index < 0 || material_index >= r->material_path_count)
        return NU_ERROR_BAD_ID;

    const char* name = r->material_paths[material_index] ? r->material_paths[material_index] : "";
    int len = (int)strlen(name);

    if (buf_cap <= 0) return len;
    if (!out_buf) return NU_ERROR;

    int copy = (len < buf_cap - 1) ? len : (buf_cap - 1);
    memcpy(out_buf, name, (size_t)copy);
    out_buf[copy] = '\0';
    return copy;
}

uint64_t nu_get_gpu_memory_used(NuRenderer* r)
{
    return r ? gpu_get_allocated_memory(r->gpu) : 0;
}

void nu_get_cmd_cache_stats(NuRenderer* r,
                             uint64_t* out_rt_replays,
                             uint64_t* out_rt_records,
                             uint64_t* out_tiled_replays,
                             uint64_t* out_tiled_records)
{
    if (!r) {
        if (out_rt_replays)    *out_rt_replays    = 0;
        if (out_rt_records)    *out_rt_records    = 0;
        if (out_tiled_replays) *out_tiled_replays = 0;
        if (out_tiled_records) *out_tiled_records = 0;
        return;
    }
    gpu_get_cmd_cache_stats(r->gpu,
                            out_rt_replays, out_rt_records,
                            out_tiled_replays, out_tiled_records);
}

const char* nu_get_last_error(NuRenderer* r)
{
    return r ? r->last_error : "NULL renderer";
}

/* perf/vk-instrumentation: per-phase GPU timings.
 * Pulls from gpu->phase_timing_ns[] which is populated by gpu_phase_resolve
 * called inside the helper that owns the queue-wait (see gpu_vulkan.c).
 * Returns NU_OK on success; NU_ERROR if r is NULL or timestamps aren't
 * supported on this device. Pass out=NULL to probe support. */
NuResult nu_get_phase_timings_ms(NuRenderer* r, NuPhaseTimings* out)
{
    if (!r) return NU_ERROR;
    if (!gpu_timestamps_supported(r->gpu)) return NU_ERROR;
    if (!out) return NU_OK;  /* probe-only call */

    out->rt_dispatch_ms          = gpu_phase_get_ms(r->gpu, GPU_PHASE_RT_DISPATCH);
    out->pixel_readback_ms       = gpu_phase_get_ms(r->gpu, GPU_PHASE_PIXEL_READBACK);
    out->blas_build_ms           = gpu_phase_get_ms(r->gpu, GPU_PHASE_BLAS_BUILD);
    out->tlas_build_ms           = gpu_phase_get_ms(r->gpu, GPU_PHASE_TLAS_BUILD);
    out->curve_blas_build_ms     = gpu_phase_get_ms(r->gpu, GPU_PHASE_CURVE_BLAS_BUILD);
    out->staging_upload_segs_ms  = gpu_phase_get_ms(r->gpu, GPU_PHASE_STAGING_UPLOAD_SEGS);
    out->staging_upload_aabbs_ms = gpu_phase_get_ms(r->gpu, GPU_PHASE_STAGING_UPLOAD_AABBS);
    out->staging_upload_colors_ms = gpu_phase_get_ms(r->gpu, GPU_PHASE_STAGING_UPLOAD_COLORS);
    /* Phase C.4 mechanism hunt — populated by gpu_wait_tiled_complete after
     * the tiled-RT readback fence signals (only on cache-miss frames). */
    out->trace_rays_tiled_ms     = gpu_phase_get_ms(r->gpu, GPU_PHASE_TRACE_RAYS_TILED);
    out->deferred_compute_ms     = gpu_phase_get_ms(r->gpu, GPU_PHASE_DEFERRED_COMPUTE);
    return NU_OK;
}
