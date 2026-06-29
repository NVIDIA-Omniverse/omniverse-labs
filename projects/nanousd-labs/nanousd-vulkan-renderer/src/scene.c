// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * scene.c — Load USD meshes via the nanousd C API for GPU rendering.
 *
 * Uses zero-copy array access (nanousd_arraydataf/i) where possible.
 * All mesh data is arena-allocated for one-shot teardown.
 */

#include "scene.h"
#include "arena.h"
#include "material.h"
#include "ptex_material.h"
#include "geo_cache.h"
#include <nanousd/nanousdapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>
#include <libgen.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
static double get_time_sec(void) {
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

/* ----------------------------------------------------------------
 * qsort comparator for floats (used by outlier rejection median)
 * ---------------------------------------------------------------- */
static int float_cmp(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

typedef struct {
    char** paths;
    int count;
    int cap;
} PathPrefixList;

static char* scene_strdup(const char* s)
{
    if (!s) return NULL;
    size_t n = strlen(s) + 1u;
    char* out = (char*)malloc(n);
    if (out) memcpy(out, s, n);
    return out;
}

static char* scene_arena_strdup(Arena* arena, const char* s)
{
    if (!arena || !s) return NULL;
    size_t n = strlen(s) + 1u;
    char* out = (char*)arena_alloc(arena, n, 1);
    if (out) memcpy(out, s, n);
    return out;
}

static void path_prefix_list_free(PathPrefixList* list)
{
    if (!list) return;
    for (int i = 0; i < list->count; i++) free(list->paths[i]);
    free(list->paths);
    list->paths = NULL;
    list->count = 0;
    list->cap = 0;
}

static void path_prefix_list_add(PathPrefixList* list, const char* path)
{
    if (!list || !path || !path[0]) return;
    if (list->count >= list->cap) {
        int new_cap = list->cap ? list->cap * 2 : 16;
        char** new_paths = (char**)realloc(list->paths, (size_t)new_cap * sizeof(char*));
        if (!new_paths) return;
        list->paths = new_paths;
        list->cap = new_cap;
    }
    list->paths[list->count] = scene_strdup(path);
    if (list->paths[list->count]) list->count++;
}

static int path_is_under_prefix(const char* path, const char* prefix)
{
    if (!path || !prefix) return 0;
    size_t n = strlen(prefix);
    if (strncmp(path, prefix, n) != 0) return 0;
    return path[n] == '\0' || path[n] == '/';
}

static int path_prefix_list_contains(const PathPrefixList* list, const char* path)
{
    if (!list || !path) return 0;
    for (int i = 0; i < list->count; i++) {
        if (path_is_under_prefix(path, list->paths[i])) return 1;
    }
    return 0;
}

static int scene_contains_ci(const char* text, const char* needle)
{
    if (!text || !needle || !needle[0]) return 0;
    size_t nn = strlen(needle);
    for (const char* p = text; *p; ++p) {
        size_t i = 0;
        for (; i < nn; ++i) {
            unsigned char a = (unsigned char)p[i];
            unsigned char b = (unsigned char)needle[i];
            if (!a) return 0;
            if (a >= 'A' && a <= 'Z') a = (unsigned char)(a - 'A' + 'a');
            if (b >= 'A' && b <= 'Z') b = (unsigned char)(b - 'A' + 'a');
            if (a != b) break;
        }
        if (i == nn) return 1;
    }
    return 0;
}

static int scene_env_is_false(const char* s)
{
    if (!s || !s[0]) return 0;
    return s[0] == '0' || s[0] == 'n' || s[0] == 'N' ||
           s[0] == 'f' || s[0] == 'F';
}

static int scene_env_is_auto(const char* s)
{
    return s && (s[0] == 'a' || s[0] == 'A');
}

static int scene_rect_light_cm_dimensions(const char* filepath,
                                          const char* prim_path,
                                          const Scene* scene,
                                          float width,
                                          float height)
{
    const char* env = getenv("NUSD_RECT_LIGHT_CM_DIMS");
    if (env && env[0]) {
        if (scene_env_is_false(env)) return 0;
        if (!scene_env_is_auto(env)) return 1;
    }

    float max_dim = fmaxf(fabsf(width), fabsf(height));
    if (max_dim < 20.0f) return 0;

    /* Disney Moana authors its island, cameras, and rectangular sun/area
     * lights in the same stage units. The 20k-wide sun_quad is intentionally
     * larger than the island; shrinking it as a centimeter-authored fixture
     * removes the warm key light and leaves the scene lit almost entirely by
     * the cool visible-dome fallback. */
    if (scene_contains_ci(filepath, "moana") ||
        scene_contains_ci(prim_path, "sun_quad"))
        return 0;

    int asset_hint =
        scene_contains_ci(filepath, "isaac") ||
        scene_contains_ci(filepath, "omniverse") ||
        scene_contains_ci(filepath, "warehouse") ||
        scene_contains_ci(prim_path, "SM_Lamp") ||
        scene_contains_ci(prim_path, "RectLight_");
    if (asset_hint) return 1;

    if (scene) {
        float dx = scene->bounds_max[0] - scene->bounds_min[0];
        float dy = scene->bounds_max[1] - scene->bounds_min[1];
        float dz = scene->bounds_max[2] - scene->bounds_min[2];
        float extent = fmaxf(fmaxf(fabsf(dx), fabsf(dy)), fabsf(dz));
        if (extent > 0.0f && max_dim >= 50.0f && max_dim > extent * 2.0f)
            return 1;
    }
    return 0;
}

static int render_helper_geometry_hidden(const char* path)
{
    if (!path) return 0;
    const char* keep = getenv("NUSD_RENDER_HELPER_GEOMETRY");
    if (keep && keep[0] && keep[0] != '0')
        return 0;

    /* Omniverse-authored datasets can include an HDR preview sphere as
     * ordinary geometry. OVRTX treats the environment as lighting/background,
     * not as an opaque primary-ray mesh. In DSX, /HDR_Sphere/Sphere otherwise
     * occludes large parts of the site as a pale dome. */
    size_t n = strlen(path);
    return strstr(path, "/HDR_Sphere/") != NULL ||
           (n >= 11 && strcmp(path + n - 11, "/HDR_Sphere") == 0);
}

static int path_is_native_usd_prototype(const char* path)
{
    if (!path) return 0;
    const char* keep = getenv("NUSD_RENDER_NATIVE_PROTOTYPES");
    if (keep && keep[0] && keep[0] != '0')
        return 0;
    return strncmp(path, "/__Prototype_", 13) == 0;
}

static int prim_is_point_instancer(NanousdPrim prim)
{
    const char* tn = prim ? nanousd_typename(prim) : NULL;
    return tn && !strcmp(tn, "PointInstancer");
}

static void collect_scene_prefixes(NanousdStage stage, int nprims,
                                   PathPrefixList* invisible_prefixes,
                                   PathPrefixList* purpose_hidden_prefixes,
                                   PathPrefixList* point_instancer_prefixes)
{
    for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;
        if (!nanousd_isactive(prim)) {
            nanousd_freeprim(prim);
            continue;
        }

        const char* path_raw = nanousd_path(prim);
        char path_buf[8192];
        const char* path = path_raw;
        if (path_raw) {
            snprintf(path_buf, sizeof(path_buf), "%s", path_raw);
            path = path_buf;
        }
        if (prim_is_point_instancer(prim))
            path_prefix_list_add(point_instancer_prefixes, path);

        int ok = 0;
        const char* vis = nanousd_attrib_token(prim, "visibility", &ok);
        if (ok && vis && !strcmp(vis, "invisible"))
            path_prefix_list_add(invisible_prefixes, path);

        ok = 0;
        const char* purpose = nanousd_attrib_token(prim, "purpose", &ok);
        if (ok && purpose && (!strcmp(purpose, "guide") || !strcmp(purpose, "proxy")))
            path_prefix_list_add(purpose_hidden_prefixes, path);

        nanousd_freeprim(prim);
    }
}

static int prim_hidden_from_render_slow(NanousdPrim prim)
{
    if (!prim) return 1;

    const char* ppath = nanousd_path(prim);
    if (ppath && strstr(ppath, "/Navmesh/") == ppath)
        return 1;
    if (render_helper_geometry_hidden(ppath))
        return 1;

    {
        int ok = 0;
        const char* vis = nanousd_attrib_token(prim, "visibility", &ok);
        if (ok && vis && !strcmp(vis, "invisible"))
            return 1;
    }

    NanousdPrim ancestor = nanousd_parent(prim);
    while (ancestor) {
        int ok = 0;
        const char* vis = nanousd_attrib_token(ancestor, "visibility", &ok);
        if (ok && vis && !strcmp(vis, "invisible")) {
            nanousd_freeprim(ancestor);
            return 1;
        }
        NanousdPrim next = nanousd_parent(ancestor);
        nanousd_freeprim(ancestor);
        ancestor = next;
    }

    {
        int ok = 0;
        const char* purpose = nanousd_attrib_token(prim, "purpose", &ok);
        if (ok && purpose && (!strcmp(purpose, "guide") || !strcmp(purpose, "proxy")))
            return 1;
    }

    ancestor = nanousd_parent(prim);
    while (ancestor) {
        int ok = 0;
        const char* purpose = nanousd_attrib_token(ancestor, "purpose", &ok);
        if (ok && purpose && (!strcmp(purpose, "guide") || !strcmp(purpose, "proxy"))) {
            nanousd_freeprim(ancestor);
            return 1;
        }
        NanousdPrim next = nanousd_parent(ancestor);
        nanousd_freeprim(ancestor);
        ancestor = next;
    }

    return 0;
}

static int prim_hidden_from_render_cached(NanousdPrim prim,
                                          const PathPrefixList* invisible_prefixes,
                                          const PathPrefixList* purpose_hidden_prefixes)
{
    if (!prim) return 1;
    if (!invisible_prefixes) return prim_hidden_from_render_slow(prim);

    const char* ppath = nanousd_path(prim);
    if (ppath && strstr(ppath, "/Navmesh/") == ppath)
        return 1;
    if (render_helper_geometry_hidden(ppath))
        return 1;
    if (path_prefix_list_contains(invisible_prefixes, ppath))
        return 1;
    if (path_prefix_list_contains(purpose_hidden_prefixes, ppath))
        return 1;

    {
        int ok = 0;
        const char* purpose = nanousd_attrib_token(prim, "purpose", &ok);
        if (ok && purpose && (!strcmp(purpose, "guide") || !strcmp(purpose, "proxy")))
            return 1;
    }

    return 0;
}

static int prim_under_point_instancer_slow(NanousdPrim prim)
{
    NanousdPrim ancestor = nanousd_parent(prim);
    while (ancestor) {
        const char* tn = nanousd_typename(ancestor);
        if (tn && !strcmp(tn, "PointInstancer")) {
            nanousd_freeprim(ancestor);
            return 1;
        }
        NanousdPrim next = nanousd_parent(ancestor);
        nanousd_freeprim(ancestor);
        ancestor = next;
    }
    return 0;
}

static int prim_under_point_instancer_cached(NanousdPrim prim,
                                             const PathPrefixList* point_instancer_prefixes)
{
    if (!prim) return 0;
    if (!point_instancer_prefixes || point_instancer_prefixes->count == 0)
        return prim_under_point_instancer_slow(prim);
    return path_prefix_list_contains(point_instancer_prefixes, nanousd_path(prim));
}

/* ----------------------------------------------------------------
 * Identity matrix constant
 * ---------------------------------------------------------------- */
static const double kIdentity[16] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
};

/* Time at which xform attributes are evaluated. NaN means "use authored
 * default time" (same as the renderer's initial state). nu_load_usd*
 * paths in renderer.c call scene_set_load_time() right before invoking
 * the loader, propagating the renderer's nu_set_current_time() value. */
static double s_load_time = NAN;
void scene_set_load_time(double t) { s_load_time = t; }
static int s_load_materials = 1;
void scene_set_load_materials(int enabled) { s_load_materials = enabled ? 1 : 0; }

typedef struct {
    uint64_t hash;
    char*    path;
    double   world[16];
} XformCacheSlot;

typedef struct {
    XformCacheSlot* slots;
    int count;
    int cap;
} XformCache;

static XformCache* g_xform_cache = NULL;

static uint64_t xform_hash_str(const char* s)
{
    uint64_t h = 1469598103934665603ull;
    while (s && *s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ull;
    }
    return h ? h : 1ull;
}

static void xform_cache_free(XformCache* c)
{
    if (!c) return;
    for (int i = 0; i < c->cap; i++) {
        free(c->slots[i].path);
    }
    free(c->slots);
    c->slots = NULL;
    c->count = 0;
    c->cap = 0;
}

static int xform_cache_grow(XformCache* c)
{
    int old_cap = c->cap;
    XformCacheSlot* old_slots = c->slots;
    int new_cap = old_cap ? old_cap * 2 : 1024;
    XformCacheSlot* new_slots =
        (XformCacheSlot*)calloc((size_t)new_cap, sizeof(XformCacheSlot));
    if (!new_slots) return 0;
    c->slots = new_slots;
    c->cap = new_cap;
    c->count = 0;
    for (int i = 0; i < old_cap; i++) {
        XformCacheSlot* s = &old_slots[i];
        if (!s->path) continue;
        uint64_t mask = (uint64_t)(c->cap - 1);
        uint64_t j = s->hash & mask;
        while (c->slots[j].path) j = (j + 1u) & mask;
        c->slots[j] = *s;
        c->count++;
    }
    free(old_slots);
    return 1;
}

static int xform_cache_lookup(const XformCache* c, const char* path,
                              double world[16])
{
    if (!c || !path || !c->slots || c->cap <= 0) return 0;
    uint64_t h = xform_hash_str(path);
    uint64_t mask = (uint64_t)(c->cap - 1);
    uint64_t i = h & mask;
    while (c->slots[i].path) {
        const XformCacheSlot* s = &c->slots[i];
        if (s->hash == h && strcmp(s->path, path) == 0) {
            memcpy(world, s->world, sizeof(double) * 16);
            return 1;
        }
        i = (i + 1u) & mask;
    }
    return 0;
}

static void xform_cache_insert(XformCache* c, const char* path,
                               const double world[16])
{
    if (!c || !path || !path[0]) return;
    if (c->count * 10 >= c->cap * 7) {
        if (!xform_cache_grow(c)) return;
    }
    uint64_t h = xform_hash_str(path);
    uint64_t mask = (uint64_t)(c->cap - 1);
    uint64_t i = h & mask;
    while (c->slots[i].path) {
        XformCacheSlot* s = &c->slots[i];
        if (s->hash == h && strcmp(s->path, path) == 0) {
            memcpy(s->world, world, sizeof(double) * 16);
            return;
        }
        i = (i + 1u) & mask;
    }
    c->slots[i].path = scene_strdup(path);
    if (!c->slots[i].path) return;
    c->slots[i].hash = h;
    memcpy(c->slots[i].world, world, sizeof(double) * 16);
    c->count++;
}

/* ----------------------------------------------------------------
 * Compute world transform by walking the parent chain, caching parent
 * transforms by prim path so large USDs don't re-walk the same Xform stack
 * for every child mesh.
 * Uses the load-time time code (s_load_time) so animated xformOps
 * resolve at the right frame.
 * ---------------------------------------------------------------- */
static void compute_world_xform(NanousdPrim prim, double world[16])
{
    if (!prim) {
        memcpy(world, kIdentity, sizeof(double) * 16);
        return;
    }

    const char* path = nanousd_path(prim);
    if (path && g_xform_cache && xform_cache_lookup(g_xform_cache, path, world))
        return;

    double local[16];
    int reset_stack = 0;
    if (!nanousd_get_local_transform(prim, s_load_time, local, &reset_stack)) {
        memcpy(local, kIdentity, sizeof(double) * 16);
    }

    NanousdPrim parent = nanousd_parent(prim);
    if (reset_stack) {
        memcpy(world, local, sizeof(double) * 16);
        if (parent) nanousd_freeprim(parent);
    } else if (parent) {
        const char* parent_path = nanousd_path(parent);
        if (parent_path && parent_path[0] == '/' && parent_path[1] == '\0') {
            memcpy(world, local, sizeof(double) * 16);
        } else {
            double parent_world[16];
            compute_world_xform(parent, parent_world);
            nanousd_mul_m4d(local, parent_world, world);
        }
        nanousd_freeprim(parent);
    } else {
        memcpy(world, local, sizeof(double) * 16);
    }

    if (path && g_xform_cache) xform_cache_insert(g_xform_cache, path, world);
}

/* ----------------------------------------------------------------
 * Topological pre-walk of world transforms.
 *
 * compute_world_xform() above is recursive + path-keyed-cache. For the
 * lazy walk it's the dominant cost (~107 µs/prim × 121k DSX = 13.7 s)
 * even with cache hits, because every prim calls nanousd_path() +
 * nanousd_parent() (heap alloc) + does string-keyed hash lookups.
 *
 * This pre-walk runs ONCE per scene_load and writes a dense
 * `world_by_idx[i*16]` array. It exploits two facts about nanousd's
 * flat prim list:
 *   1) nanousd_prim(stage, i) iterates in DFS/document order, so any
 *      ancestor of prim i has index < i.
 *   2) Most parents are themselves in the flat list (active prims).
 *
 * Per prim: one nanousd_path() lookup, one strrchr(), one O(1) hash
 * lookup, one nanousd_get_local_transform(), one mat4 multiply. No
 * recursion, no nanousd_parent() handle alloc, no strcmp inside the
 * cache.
 *
 * Fallback: if the immediate parent path is NOT in the flat-list map,
 * we use local as world (treats the missing parent as identity). For
 * common scenes (DSX, IsaacLab) where all ancestors are active, this
 * never triggers; we log a count for diagnostics.
 *
 * Returns NULL on alloc failure — caller falls back to the per-prim
 * compute_world_xform() path. */

typedef struct {
    uint64_t hash;
    int      idx;
    char*    path;   /* owned copy for strcmp on collision */
} PathToIdxSlot;

typedef struct {
    PathToIdxSlot* slots;
    int count;
    int cap;
} PathToIdx;

static void path_to_idx_init(PathToIdx* m, int nprims)
{
    int cap = 1;
    while (cap < nprims * 2 || cap < 1024) cap *= 2;
    m->slots = (PathToIdxSlot*)calloc((size_t)cap, sizeof(PathToIdxSlot));
    m->cap = m->slots ? cap : 0;
    m->count = 0;
}
static void path_to_idx_free(PathToIdx* m)
{
    if (!m->slots) return;
    for (int i = 0; i < m->cap; i++) free(m->slots[i].path);
    free(m->slots);
    m->slots = NULL; m->cap = 0; m->count = 0;
}
static void path_to_idx_insert(PathToIdx* m, const char* path, int idx)
{
    if (!m->slots || !path || !path[0]) return;
    uint64_t h = xform_hash_str(path);
    uint64_t mask = (uint64_t)(m->cap - 1);
    uint64_t i = h & mask;
    while (m->slots[i].path) {
        if (m->slots[i].hash == h && !strcmp(m->slots[i].path, path)) {
            m->slots[i].idx = idx;
            return;
        }
        i = (i + 1u) & mask;
    }
    m->slots[i].path = scene_strdup(path);
    if (!m->slots[i].path) return;
    m->slots[i].hash = h;
    m->slots[i].idx = idx;
    m->count++;
}
static int path_to_idx_lookup_n(const PathToIdx* m,
                                const char* path, int path_len)
{
    if (!m->slots || !path || path_len <= 0) return -1;
    /* Hash a path PREFIX of given length without modifying the source —
     * avoids strncpy into a scratch buffer in the hot path. */
    uint64_t h = 1469598103934665603ull;
    for (int j = 0; j < path_len; j++) {
        h ^= (unsigned char)path[j];
        h *= 1099511628211ull;
    }
    if (!h) h = 1ull;
    uint64_t mask = (uint64_t)(m->cap - 1);
    uint64_t i = h & mask;
    while (m->slots[i].path) {
        if (m->slots[i].hash == h
            && (int)strlen(m->slots[i].path) == path_len
            && memcmp(m->slots[i].path, path, (size_t)path_len) == 0) {
            return m->slots[i].idx;
        }
        i = (i + 1u) & mask;
    }
    return -1;
}

static double* prewalk_world_xforms(void* stage_void, int nprims,
                                    int* out_orphan_count)
{
    if (out_orphan_count) *out_orphan_count = 0;
    if (nprims <= 0) return NULL;

    NanousdStage stage = (NanousdStage)stage_void;
    double* world_by_idx = (double*)malloc(
        (size_t)nprims * 16 * sizeof(double));
    if (!world_by_idx) return NULL;

    PathToIdx m;
    path_to_idx_init(&m, nprims);
    if (!m.slots) { free(world_by_idx); return NULL; }

    int orphans = 0;

    for (int i = 0; i < nprims; i++) {
        double* dst = &world_by_idx[(size_t)i * 16];

        NanousdPrim p = nanousd_prim(stage, i);
        if (!p) {
            memcpy(dst, kIdentity, sizeof(double) * 16);
            continue;
        }
        const char* path = nanousd_path(p);
        if (!path || !path[0]) {
            memcpy(dst, kIdentity, sizeof(double) * 16);
            nanousd_freeprim(p);
            continue;
        }

        /* Register self in the lookup table so descendants find us. */
        path_to_idx_insert(&m, path, i);

        /* Read local transform once. */
        double local[16];
        int reset_stack = 0;
        if (!nanousd_get_local_transform(p, s_load_time, local, &reset_stack))
            memcpy(local, kIdentity, sizeof(double) * 16);

        /* Parent path = strip last '/<name>' component. */
        const char* last_slash = strrchr(path, '/');
        int parent_idx = -1;
        if (last_slash && last_slash != path) {
            int parent_len = (int)(last_slash - path);
            parent_idx = path_to_idx_lookup_n(&m, path, parent_len);
        }

        if (reset_stack) {
            memcpy(dst, local, sizeof(double) * 16);
        } else if (parent_idx >= 0) {
            const double* pw = &world_by_idx[(size_t)parent_idx * 16];
            nanousd_mul_m4d(local, pw, dst);
        } else {
            /* Root, pseudo-root, or parent not in flat list (rare). */
            memcpy(dst, local, sizeof(double) * 16);
            orphans++;
        }

        nanousd_freeprim(p);
    }

    path_to_idx_free(&m);
    if (out_orphan_count) *out_orphan_count = orphans;
    return world_by_idx;
}

/* ----------------------------------------------------------------
 * Count total triangulated indices for a mesh given faceVertexCounts and
 * the actual faceVertexIndices length.
 * Each face with N vertices produces (N-2) triangles = (N-2)*3 indices.
 *
 * USD files in the wild are sometimes malformed: faceVertexCounts claims
 * more face-vertices than faceVertexIndices provides. Treat the first
 * incomplete face as end-of-mesh instead of letting triangulation read past
 * the authored index array; that bad read can poison Vulkan BLAS input.
 * ---------------------------------------------------------------- */
static int count_triangulated_indices(const int* face_counts, int nfaces, int nfvi)
{
    int total = 0;
    int fvi_offset = 0;
    for (int i = 0; i < nfaces; i++) {
        int n = face_counts[i];
        if (n <= 0) continue;
        if (fvi_offset + n > nfvi) break;
        if (n >= 3) {
            total += (n - 2) * 3;
        }
        fvi_offset += n;
    }
    return total;
}

/* ----------------------------------------------------------------
 * Triangulate faces using fan triangulation.
 * fvi = faceVertexIndices, fvc = faceVertexCounts.
 * Writes into out_indices which must be pre-allocated.
 * ---------------------------------------------------------------- */
static void triangulate_faces(const int* fvi, int nfvi,
                              const int* fvc, int nfaces,
                              uint32_t* out_indices,
                              int* out_face_vertex_indices)
{
    int fvi_offset = 0;
    int out_offset = 0;

    for (int f = 0; f < nfaces; f++) {
        int n = fvc[f];
        if (n <= 0) {
            continue;
        }
        if (fvi_offset + n > nfvi) {
            break;
        }
        if (n < 3) {
            fvi_offset += n;
            continue;
        }

        int v0 = fvi[fvi_offset];
        for (int t = 0; t < n - 2; t++) {
            int fv0 = fvi_offset;
            int fv1 = fvi_offset + t + 1;
            int fv2 = fvi_offset + t + 2;
            out_indices[out_offset] = (uint32_t)v0;
            if (out_face_vertex_indices) out_face_vertex_indices[out_offset] = fv0;
            out_offset++;
            out_indices[out_offset] = (uint32_t)fvi[fv1];
            if (out_face_vertex_indices) out_face_vertex_indices[out_offset] = fv1;
            out_offset++;
            out_indices[out_offset] = (uint32_t)fvi[fv2];
            if (out_face_vertex_indices) out_face_vertex_indices[out_offset] = fv2;
            out_offset++;
        }

        fvi_offset += n;
    }
}

static int scene_mesh_assign_ptex_colors(Arena* arena,
                                         SceneMesh* mesh,
                                         const MaterialCollection* materials,
                                         const int* fvc,
                                         int fvc_count,
                                         int fvi_count)
{
    if (!arena || !mesh || !materials) return 0;
    if (mesh->material_index < 0 ||
        mesh->material_index >= materials->nmaterials) {
        return 0;
    }
    const SceneMaterial* mat = &materials->materials[mesh->material_index];
    if (!mat->ptex_color_path[0]) return 0;
    if (!mesh->indices || mesh->nindices <= 0 || (mesh->nindices % 3) != 0)
        return 0;

    int color_count = mesh->nindices;
    uint32_t* colors = (uint32_t*)arena_alloc(
        arena, (size_t)color_count * sizeof(uint32_t), 16);
    if (!colors) return 0;

    int written = nusd_ptex_sample_face_tri_colors(mat->ptex_color_path,
                                                   fvc, fvc_count,
                                                   fvi_count,
                                                   colors,
                                                   color_count);
    if (written != color_count) {
        static int warn_count = 0;
        if (warn_count < 16) {
            fprintf(stderr,
                    "scene_load: Ptex color sampling skipped for %s "
                    "(wrote %d/%d triangle-corner colors from %s)\n",
                    mesh->path ? mesh->path : "<mesh>",
                    written, color_count, mat->ptex_color_path);
            warn_count++;
        }
        return 0;
    }

    mesh->ptex_tri_colors = colors;
    mesh->ptex_tri_color_count = color_count;
    return 1;
}

/* ----------------------------------------------------------------
 * Compute smooth vertex normals from triangle indices + positions.
 * Accumulates face normals (area-weighted) per vertex, then normalizes.
 * ---------------------------------------------------------------- */
static float* compute_smooth_normals(Arena* arena, const float* positions, int nvertices,
                                     const uint32_t* indices, int nindices)
{
    float* normals = (float*)arena_calloc(arena, (size_t)nvertices * 3, sizeof(float));
    if (!normals) return NULL;

    for (int i = 0; i + 2 < nindices; i += 3) {
        uint32_t i0 = indices[i], i1 = indices[i+1], i2 = indices[i+2];
        if (i0 >= (uint32_t)nvertices || i1 >= (uint32_t)nvertices || i2 >= (uint32_t)nvertices)
            continue;

        const float* p0 = &positions[i0 * 3];
        const float* p1 = &positions[i1 * 3];
        const float* p2 = &positions[i2 * 3];

        float e1[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
        float e2[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };

        /* Cross product (area-weighted face normal) */
        float nx = e1[1]*e2[2] - e1[2]*e2[1];
        float ny = e1[2]*e2[0] - e1[0]*e2[2];
        float nz = e1[0]*e2[1] - e1[1]*e2[0];

        /* Accumulate to each vertex of the triangle */
        normals[i0*3+0] += nx; normals[i0*3+1] += ny; normals[i0*3+2] += nz;
        normals[i1*3+0] += nx; normals[i1*3+1] += ny; normals[i1*3+2] += nz;
        normals[i2*3+0] += nx; normals[i2*3+1] += ny; normals[i2*3+2] += nz;
    }

    /* Normalize */
    for (int v = 0; v < nvertices; v++) {
        float* n = &normals[v * 3];
        float len = sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        if (len > 1e-12f) {
            float inv = 1.0f / len;
            n[0] *= inv; n[1] *= inv; n[2] *= inv;
        } else {
            n[0] = 0.0f; n[1] = 1.0f; n[2] = 0.0f;
        }
    }

    return normals;
}

/* ----------------------------------------------------------------
 * Expand a mesh when USD authors face-varying normals.
 *
 * The renderer's vertex layout stores one normal per uploaded vertex. USD
 * commonly stores normals per face-vertex, allowing hard edges on shared
 * point topology. If we ignore those normals and smooth shared points, cubes
 * and CAD-like meshes get black diagonal facets. Split each triangulated
 * face-vertex into its own uploaded vertex so the authored normal survives.
 * ---------------------------------------------------------------- */
static int expand_facevarying_normals(Arena* arena, SceneMesh* mesh,
                                      const float* fv_normals,
                                      int n_fv_normals,
                                      const int* tri_fv_indices)
{
    if (!arena || !mesh || !mesh->positions || !mesh->indices ||
        !fv_normals || !tri_fv_indices || mesh->nindices <= 0)
        return 0;

    int new_nvertices = mesh->nindices;
    float* new_positions =
        (float*)arena_alloc(arena, (size_t)new_nvertices * 3u * sizeof(float), 16);
    float* new_normals =
        (float*)arena_alloc(arena, (size_t)new_nvertices * 3u * sizeof(float), 16);
    uint32_t* new_indices =
        (uint32_t*)arena_alloc(arena, (size_t)mesh->nindices * sizeof(uint32_t), 16);
    if (!new_positions || !new_normals || !new_indices)
        return 0;

    const float* old_positions = mesh->positions;
    int old_nvertices = mesh->nvertices;
    int bad_normal_tris = 0;
    for (int i = 0; i < mesh->nindices; i++) {
        uint32_t src_v = mesh->indices[i];
        int src_fv = tri_fv_indices[i];
        if (src_v >= (uint32_t)old_nvertices || src_fv < 0 || src_fv >= n_fv_normals)
            return 0;
        new_positions[i * 3 + 0] = old_positions[src_v * 3 + 0];
        new_positions[i * 3 + 1] = old_positions[src_v * 3 + 1];
        new_positions[i * 3 + 2] = old_positions[src_v * 3 + 2];
        new_normals[i * 3 + 0] = fv_normals[src_fv * 3 + 0];
        new_normals[i * 3 + 1] = fv_normals[src_fv * 3 + 1];
        new_normals[i * 3 + 2] = fv_normals[src_fv * 3 + 2];
        new_indices[i] = (uint32_t)i;
    }

    for (int i = 0; i + 2 < mesh->nindices; i += 3) {
        const float* n0 = &new_normals[(i + 0) * 3];
        const float* n1 = &new_normals[(i + 1) * 3];
        const float* n2 = &new_normals[(i + 2) * 3];
        float d01 = n0[0]*n1[0] + n0[1]*n1[1] + n0[2]*n1[2];
        float d02 = n0[0]*n2[0] + n0[1]*n2[1] + n0[2]*n2[2];
        float d12 = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2];
        if (d01 < -0.25f || d02 < -0.25f || d12 < -0.25f)
            bad_normal_tris++;
    }

    if (bad_normal_tris > 0) {
        /* Some simple/generated USDs author a normals array with face-varying
         * length but no interpolation token and internally contradictory
         * values. Uploading those normals produces black diagonal facets.
         * Keep the split topology but fall back to geometric flat normals,
         * matching Hydra/OVRTX's robust treatment of these malformed inputs. */
        for (int i = 0; i + 2 < mesh->nindices; i += 3) {
            const float* p0 = &new_positions[(i + 0) * 3];
            const float* p1 = &new_positions[(i + 1) * 3];
            const float* p2 = &new_positions[(i + 2) * 3];
            float e1[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
            float e2[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
            float n[3] = {
                e1[1]*e2[2] - e1[2]*e2[1],
                e1[2]*e2[0] - e1[0]*e2[2],
                e1[0]*e2[1] - e1[1]*e2[0],
            };
            float len = sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
            if (len > 1e-12f) {
                float inv = 1.0f / len;
                n[0] *= inv; n[1] *= inv; n[2] *= inv;
            } else {
                n[0] = 0.0f; n[1] = 1.0f; n[2] = 0.0f;
            }
            for (int j = 0; j < 3; j++) {
                new_normals[(i + j) * 3 + 0] = n[0];
                new_normals[(i + j) * 3 + 1] = n[1];
                new_normals[(i + j) * 3 + 2] = n[2];
            }
        }
    }

    mesh->positions = new_positions;
    mesh->normals = new_normals;
    mesh->indices = new_indices;
    mesh->nvertices = new_nvertices;
    return 1;
}

/* ----------------------------------------------------------------
 * Hash-dedup expansion for meshes with faceVarying primvars (UV seams,
 * hard-edge faceVarying normals). Each face-vertex is keyed on
 * (original vertex, UV, normal), so vertices split only where USD authored
 * a distinct per-face value.
 * ---------------------------------------------------------------- */
static int expand_facevarying_mesh(Arena* arena,
                                   SceneMesh* mesh,
                                   const int* fvi_data, int fvi_count,
                                   const int* fvc_data, int fvc_count,
                                   const float* uvs_fv,
                                   const float* normals_fv)
{
    if (!mesh || !mesh->positions || !fvi_data || fvi_count <= 0)
        return 0;

    const int norig = mesh->nvertices;
    const float* const orig_positions = mesh->positions;
    const float* const orig_normals = mesh->normals;
    const float* const orig_uvs = mesh->texcoords;
    const int has_normals = (orig_normals != NULL) || (normals_fv != NULL);
    const int has_uvs = (orig_uvs != NULL) || (uvs_fv != NULL);

    typedef struct {
        float uv[2];
        float n[3];
        uint32_t new_idx;
        int next;
    } Split;

    Split* splits = (Split*)arena_alloc(arena, (size_t)fvi_count * sizeof(Split), 8);
    int* split_head = (int*)arena_alloc(arena, (size_t)norig * sizeof(int), 4);
    uint32_t* fv_to_new =
        (uint32_t*)arena_alloc(arena, (size_t)fvi_count * sizeof(uint32_t), 4);
    if (!splits || !split_head || !fv_to_new)
        return 0;
    for (int v = 0; v < norig; v++)
        split_head[v] = -1;

    float* new_positions =
        (float*)arena_alloc(arena, (size_t)fvi_count * 3 * sizeof(float), 16);
    float* new_normals = has_normals
        ? (float*)arena_alloc(arena, (size_t)fvi_count * 3 * sizeof(float), 16)
        : NULL;
    float* new_uvs = has_uvs
        ? (float*)arena_alloc(arena, (size_t)fvi_count * 2 * sizeof(float), 16)
        : NULL;
    if (!new_positions || (has_normals && !new_normals) || (has_uvs && !new_uvs))
        return 0;

    int split_count = 0;
    int new_nv = 0;
    for (int j = 0; j < fvi_count; j++) {
        const int v = fvi_data[j];
        if (v < 0 || v >= norig) {
            fv_to_new[j] = 0;
            continue;
        }

        float u0 = 0.0f, u1 = 0.0f;
        if (uvs_fv) {
            u0 = uvs_fv[j * 2 + 0];
            u1 = uvs_fv[j * 2 + 1];
        } else if (orig_uvs) {
            u0 = orig_uvs[v * 2 + 0];
            u1 = orig_uvs[v * 2 + 1];
        }

        float n0 = 0.0f, n1 = 0.0f, n2 = 0.0f;
        if (normals_fv) {
            n0 = normals_fv[j * 3 + 0];
            n1 = normals_fv[j * 3 + 1];
            n2 = normals_fv[j * 3 + 2];
        } else if (orig_normals) {
            n0 = orig_normals[v * 3 + 0];
            n1 = orig_normals[v * 3 + 1];
            n2 = orig_normals[v * 3 + 2];
        }

        int found = -1;
        const float eps = 1e-6f;
        for (int s = split_head[v]; s != -1; s = splits[s].next) {
            const Split* sp = &splits[s];
            if (fabsf(sp->uv[0] - u0) < eps && fabsf(sp->uv[1] - u1) < eps &&
                fabsf(sp->n[0] - n0) < eps && fabsf(sp->n[1] - n1) < eps &&
                fabsf(sp->n[2] - n2) < eps) {
                found = (int)sp->new_idx;
                break;
            }
        }

        if (found >= 0) {
            fv_to_new[j] = (uint32_t)found;
            continue;
        }

        const uint32_t idx = (uint32_t)new_nv++;
        const int s = split_count++;
        splits[s].uv[0] = u0;
        splits[s].uv[1] = u1;
        splits[s].n[0] = n0;
        splits[s].n[1] = n1;
        splits[s].n[2] = n2;
        splits[s].new_idx = idx;
        splits[s].next = split_head[v];
        split_head[v] = s;
        fv_to_new[j] = idx;

        new_positions[idx * 3 + 0] = orig_positions[v * 3 + 0];
        new_positions[idx * 3 + 1] = orig_positions[v * 3 + 1];
        new_positions[idx * 3 + 2] = orig_positions[v * 3 + 2];
        if (new_normals) {
            new_normals[idx * 3 + 0] = n0;
            new_normals[idx * 3 + 1] = n1;
            new_normals[idx * 3 + 2] = n2;
        }
        if (new_uvs) {
            new_uvs[idx * 2 + 0] = u0;
            new_uvs[idx * 2 + 1] = u1;
        }
    }

    const int tri_count = (fvc_data && fvc_count > 0)
        ? count_triangulated_indices(fvc_data, fvc_count, fvi_count)
        : fvi_count;
    uint32_t* new_indices =
        (uint32_t*)arena_alloc(arena, (size_t)tri_count * sizeof(uint32_t), 16);
    if (!new_indices)
        return 0;

    int n_idx = 0;
    if (fvc_data && fvc_count > 0) {
        int fvi_off = 0;
        for (int i = 0; i < fvc_count; i++) {
            const int n = fvc_data[i];
            if (n <= 0)
                continue;
            if (fvi_off + n > fvi_count)
                break;
            if (n < 3) {
                if (n > 0)
                    fvi_off += n;
                continue;
            }
            const uint32_t v0 = fv_to_new[fvi_off];
            for (int t = 0; t < n - 2; t++) {
                new_indices[n_idx++] = v0;
                new_indices[n_idx++] = fv_to_new[fvi_off + t + 1];
                new_indices[n_idx++] = fv_to_new[fvi_off + t + 2];
            }
            fvi_off += n;
        }
    } else {
        const int n = fvi_count - (fvi_count % 3);
        for (int j = 0; j < n; j++)
            new_indices[n_idx++] = fv_to_new[j];
    }

    mesh->positions = new_positions;
    mesh->normals = new_normals;
    mesh->texcoords = new_uvs;
    mesh->indices = new_indices;
    mesh->nvertices = new_nv;
    mesh->nindices = n_idx;
    return 1;
}

static int apply_usdskel_skinning(Arena* arena,
                                  NanousdStage stage,
                                  NanousdPrim prim,
                                  SceneMesh* mesh,
                                  const int* fvi_data,
                                  int fvi_count,
                                  float* fv_normals);

/* ----------------------------------------------------------------
 * Load UV primvars and face-varying normals. When any supported primvar is
 * faceVarying, expand the mesh so UV seams and hard normals survive Vulkan's
 * one-vertex/one-attribute upload layout.
 * ---------------------------------------------------------------- */
static int load_mesh_facevarying_primvars(Arena* arena,
                                          SceneMesh* mesh,
                                          NanousdStage stage,
                                          NanousdPrim prim,
                                          const int* fvi_data, int fvi_count,
                                          const int* fvc_data, int fvc_count)
{
    mesh->texcoords = NULL;
    if (!prim || !fvi_data || fvi_count <= 0)
        return 0;

    const int skip_texcoords = !s_load_materials;

    const char* uv_names[] = {
        "primvars:st", "primvars:st0", "primvars:UVMap", "primvars:uv"
    };
    const char* uv_idx_names[] = {
        "primvars:st:indices", "primvars:st0:indices",
        "primvars:UVMap:indices", "primvars:uv:indices"
    };
    const char* uv_attr = NULL;
    const float* uv_pool = NULL;
    int uv_pool_count = 0;
    const int* uv_indices = NULL;
    int uv_idx_count = 0;
    const char* uv_interp = NULL;

    if (!skip_texcoords) {
        for (int i = 0; i < (int)(sizeof(uv_names) / sizeof(uv_names[0])); i++) {
            int n = 0;
            const float* d = nanousd_arraydataf(prim, uv_names[i], &n);
            if (d && n > 0) {
                uv_pool = d;
                uv_pool_count = n;
                uv_attr = uv_names[i];
                int idx_n = 0;
                uv_indices = nanousd_arraydatai(prim, uv_idx_names[i], &idx_n);
                uv_idx_count = idx_n;
                uv_interp = nanousd_attrib_interpolation(prim, uv_attr);
                break;
            }
        }
    }

    int uv_is_face_varying = 0;
    int uv_is_vertex_rate = 0;
    if (uv_pool && uv_pool_count > 0) {
        if (uv_interp && strcmp(uv_interp, "faceVarying") == 0) {
            uv_is_face_varying = 1;
        } else if (uv_interp &&
                   (strcmp(uv_interp, "vertex") == 0 ||
                    strcmp(uv_interp, "varying") == 0)) {
            uv_is_vertex_rate = (uv_pool_count / 2 == mesh->nvertices);
        } else {
            if (uv_pool_count / 2 == mesh->nvertices)
                uv_is_vertex_rate = 1;
            else if ((uv_indices && uv_idx_count == fvi_count) ||
                     (!uv_indices && uv_pool_count / 2 == fvi_count))
                uv_is_face_varying = 1;
        }
    }

    const char* nrm_attr = NULL;
    const float* nrm_data_raw = NULL;
    int nrm_count_raw = 0;
    int nrm_n = 0;
    const float* nrm_try = nanousd_arraydataf(prim, "normals", &nrm_n);
    if (nrm_try) {
        nrm_attr = "normals";
        nrm_data_raw = nrm_try;
        nrm_count_raw = nrm_n;
    } else {
        nrm_try = nanousd_arraydataf(prim, "primvars:normals", &nrm_n);
        if (nrm_try) {
            nrm_attr = "primvars:normals";
            nrm_data_raw = nrm_try;
            nrm_count_raw = nrm_n;
        }
    }

    const char* nrm_interp = nrm_attr ? nanousd_attrib_interpolation(prim, nrm_attr) : NULL;
    int normals_are_face_varying = 0;
    if (nrm_data_raw && nrm_count_raw / 3 == fvi_count) {
        if (nrm_interp && strcmp(nrm_interp, "faceVarying") == 0)
            normals_are_face_varying = 1;
        else if (!nrm_interp && fvi_count != mesh->nvertices)
            normals_are_face_varying = 1;
    }

    float* uvs_fv = NULL;
    float* normals_fv = NULL;
    if (uv_is_face_varying) {
        uvs_fv =
            (float*)arena_alloc(arena, (size_t)fvi_count * 2 * sizeof(float), 16);
        if (uvs_fv) {
            const int n_uvs = uv_pool_count / 2;
            if (uv_indices && uv_idx_count > 0) {
                const int n_walk = fvi_count < uv_idx_count ? fvi_count : uv_idx_count;
                for (int j = 0; j < n_walk; j++) {
                    const int u = uv_indices[j];
                    if (u < 0 || u >= n_uvs) {
                        uvs_fv[j * 2 + 0] = 0.0f;
                        uvs_fv[j * 2 + 1] = 0.0f;
                    } else {
                        uvs_fv[j * 2 + 0] = uv_pool[u * 2 + 0];
                        uvs_fv[j * 2 + 1] = uv_pool[u * 2 + 1];
                    }
                }
                for (int j = n_walk; j < fvi_count; j++) {
                    uvs_fv[j * 2 + 0] = 0.0f;
                    uvs_fv[j * 2 + 1] = 0.0f;
                }
            } else if (uv_pool_count / 2 == fvi_count) {
                memcpy(uvs_fv, uv_pool, (size_t)fvi_count * 2 * sizeof(float));
            } else {
                uvs_fv = NULL;
                uv_is_face_varying = 0;
            }
        } else {
            uv_is_face_varying = 0;
        }
    }

    if (normals_are_face_varying) {
        normals_fv =
            (float*)arena_alloc(arena, (size_t)fvi_count * 3 * sizeof(float), 16);
        if (normals_fv)
            memcpy(normals_fv, nrm_data_raw, (size_t)fvi_count * 3 * sizeof(float));
        else
            normals_are_face_varying = 0;
    }

    apply_usdskel_skinning(arena, stage, prim, mesh,
                           fvi_data, fvi_count, normals_fv);

    if (uv_is_face_varying || normals_are_face_varying) {
        if (!skip_texcoords && uv_is_vertex_rate && uv_pool) {
            mesh->texcoords =
                (float*)arena_alloc(arena, (size_t)uv_pool_count * sizeof(float), 16);
            if (mesh->texcoords)
                memcpy(mesh->texcoords, uv_pool, (size_t)uv_pool_count * sizeof(float));
        }
        return expand_facevarying_mesh(arena, mesh, fvi_data, fvi_count,
                                       fvc_data, fvc_count, uvs_fv, normals_fv);
    }

    if (!skip_texcoords && uv_is_vertex_rate && uv_pool) {
        mesh->texcoords =
            (float*)arena_alloc(arena, (size_t)uv_pool_count * sizeof(float), 16);
        if (mesh->texcoords)
            memcpy(mesh->texcoords, uv_pool, (size_t)uv_pool_count * sizeof(float));
    }
    return 0;
}

/* ----------------------------------------------------------------
 * Transform a point by a 4x4 row-major matrix (for bounds).
 * ---------------------------------------------------------------- */
static void xform_point(const double m[16], const float p[3], float out[3])
{
    double px = (double)p[0], py = (double)p[1], pz = (double)p[2];
    out[0] = (float)(m[0]*px + m[4]*py + m[8]*pz  + m[12]);
    out[1] = (float)(m[1]*px + m[5]*py + m[9]*pz  + m[13]);
    out[2] = (float)(m[2]*px + m[6]*py + m[10]*pz + m[14]);
}

static int invert_affine_m4d_rowvec(const double m[16], double out[16])
{
    double a00 = m[0], a01 = m[1], a02 = m[2];
    double a10 = m[4], a11 = m[5], a12 = m[6];
    double a20 = m[8], a21 = m[9], a22 = m[10];

    double c00 =  a11 * a22 - a12 * a21;
    double c01 = -(a10 * a22 - a12 * a20);
    double c02 =  a10 * a21 - a11 * a20;
    double c10 = -(a01 * a22 - a02 * a21);
    double c11 =  a00 * a22 - a02 * a20;
    double c12 = -(a00 * a21 - a01 * a20);
    double c20 =  a01 * a12 - a02 * a11;
    double c21 = -(a00 * a12 - a02 * a10);
    double c22 =  a00 * a11 - a01 * a10;
    double det = a00 * c00 + a01 * c01 + a02 * c02;
    if (fabs(det) < 1e-12)
        return 0;

    double inv_det = 1.0 / det;
    out[0] = c00 * inv_det; out[1] = c10 * inv_det; out[2]  = c20 * inv_det; out[3]  = 0.0;
    out[4] = c01 * inv_det; out[5] = c11 * inv_det; out[6]  = c21 * inv_det; out[7]  = 0.0;
    out[8] = c02 * inv_det; out[9] = c12 * inv_det; out[10] = c22 * inv_det; out[11] = 0.0;

    double tx = m[12], ty = m[13], tz = m[14];
    out[12] = -(tx * out[0] + ty * out[4] + tz * out[8]);
    out[13] = -(tx * out[1] + ty * out[5] + tz * out[9]);
    out[14] = -(tx * out[2] + ty * out[6] + tz * out[10]);
    out[15] = 1.0;
    return 1;
}

typedef struct {
    char** items;
    int    count;
} TokenArray;

static void token_array_free(TokenArray* a)
{
    if (!a) return;
    for (int i = 0; i < a->count; i++)
        free(a->items[i]);
    free(a->items);
    a->items = NULL;
    a->count = 0;
}

static int token_array_read(NanousdPrim prim, const char* name, TokenArray* out)
{
    if (!prim || !name || !out) return 0;
    memset(out, 0, sizeof(*out));

    int use_token_reader = 1;
    int n = nanousd_attribarraytokens_len(prim, name);
    if (n < 0) {
        use_token_reader = 0;
        n = nanousd_attribarrays_len(prim, name);
    }
    if (n <= 0) return 0;

    out->items = (char**)calloc((size_t)n, sizeof(char*));
    if (!out->items) return 0;
    out->count = n;

    for (int i = 0; i < n; i++) {
        const char* s = use_token_reader
            ? nanousd_attribarraytokens(prim, name, i)
            : nanousd_attribarrays(prim, name, i);
        out->items[i] = scene_strdup(s ? s : "");
        if (!out->items[i]) {
            token_array_free(out);
            return 0;
        }
    }
    return 1;
}

static int token_array_find(const TokenArray* a, const char* needle)
{
    if (!a || !needle) return -1;
    for (int i = 0; i < a->count; i++) {
        if (a->items[i] && strcmp(a->items[i], needle) == 0)
            return i;
    }
    return -1;
}

static int find_ancestor_reltarget(NanousdPrim prim,
                                   const char* rel,
                                   char* out,
                                   size_t out_size)
{
    if (!prim || !rel || !out || out_size == 0) return 0;
    out[0] = '\0';

    NanousdPrim cur = prim;
    int own_cur = 0;
    while (cur) {
        if (nanousd_nreltargets(cur, rel) > 0) {
            const char* target = nanousd_reltarget(cur, rel, 0);
            if (target && target[0]) {
                snprintf(out, out_size, "%s", target);
                if (own_cur) nanousd_freeprim(cur);
                return 1;
            }
        }
        NanousdPrim parent = nanousd_parent(cur);
        if (own_cur) nanousd_freeprim(cur);
        cur = parent;
        own_cur = 1;
    }
    return 0;
}

static int joint_parent_index(const TokenArray* joints, int joint_index)
{
    if (!joints || joint_index < 0 || joint_index >= joints->count)
        return -1;
    const char* joint = joints->items[joint_index];
    if (!joint) return -1;
    const char* slash = strrchr(joint, '/');
    if (!slash) return -1;

    char parent[2048];
    size_t n = (size_t)(slash - joint);
    if (n == 0 || n >= sizeof(parent)) return -1;
    memcpy(parent, joint, n);
    parent[n] = '\0';
    return token_array_find(joints, parent);
}

static void make_trs_matrix(const float* t, const float* q_xyzw,
                            const float* s, double out[16])
{
    double q[4] = {
        (double)q_xyzw[3],
        (double)q_xyzw[0],
        (double)q_xyzw[1],
        (double)q_xyzw[2],
    };
    double q_len = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (q_len > 1e-12) {
        double inv = 1.0 / q_len;
        q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
    } else {
        q[0] = 1.0; q[1] = q[2] = q[3] = 0.0;
    }

    double rot[16];
    nanousd_quat_to_matrix(q, rot);

    /* OpenUSD's UsdSkelMakeTransform writes row-vector matrices. The
     * nanousd quaternion helper returns the opposite-handed 3x3, so
     * transpose the rotation block before applying row scales. */
    out[0] = rot[0] * s[0]; out[1] = rot[4] * s[0]; out[2]  = rot[8]  * s[0]; out[3]  = 0.0;
    out[4] = rot[1] * s[1]; out[5] = rot[5] * s[1]; out[6]  = rot[9]  * s[1]; out[7]  = 0.0;
    out[8] = rot[2] * s[2]; out[9] = rot[6] * s[2]; out[10] = rot[10] * s[2]; out[11] = 0.0;
    out[12] = t[0]; out[13] = t[1]; out[14] = t[2]; out[15] = 1.0;
}

static int read_usdskel_anim_arrayf(NanousdPrim anim, const char* name,
                                    int expected_count, float* out)
{
    if (!anim || !name || expected_count <= 0 || !out)
        return 0;
    if (!isnan(s_load_time) && nanousd_hassamples(anim, name)) {
        int n = nanousd_samplearrayf(anim, name, s_load_time,
                                     out, expected_count);
        if (n == expected_count)
            return 1;
    }
    return nanousd_attribarrayf(anim, name, out, expected_count) ==
           expected_count;
}

static void xform_dir(const double m[16], const float v[3], double out[3])
{
    double x = (double)v[0], y = (double)v[1], z = (double)v[2];
    out[0] = m[0]*x + m[4]*y + m[8]*z;
    out[1] = m[1]*x + m[5]*y + m[9]*z;
    out[2] = m[2]*x + m[6]*y + m[10]*z;
}

static void normalize_dir3(double v[3])
{
    double len = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len > 1e-12) {
        double inv = 1.0 / len;
        v[0] *= inv; v[1] *= inv; v[2] *= inv;
    }
}

static int skin_joint_index_for_vertex(int vertex_index,
                                       int influence_count,
                                       const int* joint_indices,
                                       int mesh_joint_count,
                                       const int* mesh_to_skel,
                                       int influence_slot)
{
    int local_joint = joint_indices[vertex_index * influence_count + influence_slot];
    if (local_joint < 0) return -1;
    if (mesh_joint_count > 0) {
        if (local_joint >= mesh_joint_count) return -1;
        return mesh_to_skel[local_joint];
    }
    return local_joint;
}

/* Normal matrix = inverse-transpose of the affine's upper-3x3, laid out so that
 * xform_dir(out, n) computes the row-vector normal transform n*(A^-1)^T. Normals
 * are covectors: a non-uniform-scale or shear skin matrix skews them when applied
 * directly (the old behavior), so the inverse-transpose is required. For a rigid
 * joint (rotation, or rotation+uniform-scale) it equals the rotation up to a
 * scale that normalize_dir3() removes, so rigid skinning stays bit-identical.
 * On a singular joint matrix we fall back to the affine itself (no worse than
 * before; normalize_dir3 still cleans up). */
static int build_normal_matrix(const double affine[16], double out[16])
{
    double inv[16];
    if (!invert_affine_m4d_rowvec(affine, inv)) {
        memcpy(out, affine, 16u * sizeof(double));
        return 0;
    }
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            out[r * 4 + c] = inv[c * 4 + r];   /* transpose of the inverse */
    return 1;
}

static int apply_usdskel_skinning(Arena* arena,
                                  NanousdStage stage,
                                  NanousdPrim prim,
                                  SceneMesh* mesh,
                                  const int* fvi_data,
                                  int fvi_count,
                                  float* fv_normals)
{
    (void)arena;
    if (!stage || !prim || !mesh || !mesh->positions || mesh->nvertices <= 0)
        return 0;

    char skel_path[2048];
    if (!find_ancestor_reltarget(prim, "skel:skeleton",
                                 skel_path, sizeof(skel_path)))
        return 0;

    NanousdPrim skel = nanousd_primpath(stage, skel_path);
    if (!skel) return 0;

    TokenArray skel_joints;
    if (!token_array_read(skel, "joints", &skel_joints)) {
        nanousd_freeprim(skel);
        return 0;
    }
    const int njoints = skel_joints.count;

    const int bind_len = njoints * 16;
    double* bind = (double*)malloc((size_t)bind_len * sizeof(double));
    double* rest = (double*)malloc((size_t)bind_len * sizeof(double));
    double* local = (double*)malloc((size_t)bind_len * sizeof(double));
    double* skel_xforms = (double*)malloc((size_t)bind_len * sizeof(double));
    double* skin_xforms = (double*)malloc((size_t)bind_len * sizeof(double));
    double* skin_normal_xforms = (double*)malloc((size_t)bind_len * sizeof(double));
    if (!bind || !rest || !local || !skel_xforms || !skin_xforms || !skin_normal_xforms)
        goto fail;

    if (nanousd_attribarrayd(skel, "bindTransforms", bind, bind_len) != bind_len)
        goto fail;
    if (nanousd_attribarrayd(skel, "restTransforms", rest, bind_len) != bind_len)
        goto fail;
    memcpy(local, rest, (size_t)bind_len * sizeof(double));

    char anim_path[2048];
    if (nanousd_nreltargets(skel, "skel:animationSource") > 0) {
        const char* target = nanousd_reltarget(skel, "skel:animationSource", 0);
        snprintf(anim_path, sizeof(anim_path), "%s", target ? target : "");
    } else {
        anim_path[0] = '\0';
    }

    if (anim_path[0]) {
        NanousdPrim anim = nanousd_primpath(stage, anim_path);
        TokenArray anim_joints;
        memset(&anim_joints, 0, sizeof(anim_joints));
        if (anim && token_array_read(anim, "joints", &anim_joints)) {
            const int anim_count = anim_joints.count;
            float* trans = (float*)malloc((size_t)anim_count * 3u * sizeof(float));
            float* rots  = (float*)malloc((size_t)anim_count * 4u * sizeof(float));
            float* scale = (float*)malloc((size_t)anim_count * 3u * sizeof(float));
            if (trans && rots && scale &&
                read_usdskel_anim_arrayf(anim, "translations",
                                         anim_count * 3, trans) &&
                read_usdskel_anim_arrayf(anim, "rotations",
                                         anim_count * 4, rots) &&
                read_usdskel_anim_arrayf(anim, "scales",
                                         anim_count * 3, scale)) {
                for (int i = 0; i < anim_count; i++) {
                    int skel_idx = token_array_find(&skel_joints, anim_joints.items[i]);
                    if (skel_idx >= 0 && skel_idx < njoints) {
                        make_trs_matrix(&trans[i * 3], &rots[i * 4],
                                        &scale[i * 3],
                                        &local[skel_idx * 16]);
                    }
                }
            }
            free(trans);
            free(rots);
            free(scale);
            token_array_free(&anim_joints);
        }
        if (anim) nanousd_freeprim(anim);
    }

    for (int i = 0; i < njoints; i++) {
        int parent = joint_parent_index(&skel_joints, i);
        if (parent >= 0 && parent < i) {
            nanousd_mul_m4d(&local[i * 16],
                            &skel_xforms[parent * 16],
                            &skel_xforms[i * 16]);
        } else {
            memcpy(&skel_xforms[i * 16], &local[i * 16],
                   16u * sizeof(double));
        }

        double inv_bind[16];
        if (!invert_affine_m4d_rowvec(&bind[i * 16], inv_bind))
            goto fail;
        nanousd_mul_m4d(inv_bind, &skel_xforms[i * 16],
                        &skin_xforms[i * 16]);
        /* Normals use the inverse-transpose of the same joint skin matrix. */
        build_normal_matrix(&skin_xforms[i * 16], &skin_normal_xforms[i * 16]);
    }

    TokenArray mesh_joints;
    memset(&mesh_joints, 0, sizeof(mesh_joints));
    int* mesh_to_skel = NULL;
    int mesh_joint_count = 0;
    if (token_array_read(prim, "skel:joints", &mesh_joints)) {
        mesh_joint_count = mesh_joints.count;
        mesh_to_skel = (int*)malloc((size_t)mesh_joint_count * sizeof(int));
        if (!mesh_to_skel) goto fail_mesh_joints;
        for (int i = 0; i < mesh_joint_count; i++)
            mesh_to_skel[i] = token_array_find(&skel_joints, mesh_joints.items[i]);
    }

    int ji_len = nanousd_attribarraylen(prim, "primvars:skel:jointIndices");
    int jw_len = nanousd_attribarraylen(prim, "primvars:skel:jointWeights");
    int influence_count = mesh->nvertices > 0 ? ji_len / mesh->nvertices : 0;
    if (ji_len <= 0 || jw_len <= 0 || ji_len != jw_len ||
        influence_count <= 0 || influence_count * mesh->nvertices != ji_len)
        goto fail_mesh_joints;

    int* joint_indices = (int*)malloc((size_t)ji_len * sizeof(int));
    float* joint_weights = (float*)malloc((size_t)jw_len * sizeof(float));
    if (!joint_indices || !joint_weights) {
        free(joint_indices);
        free(joint_weights);
        goto fail_mesh_joints;
    }
    if (nanousd_attribarrayi(prim, "primvars:skel:jointIndices",
                             joint_indices, ji_len) != ji_len ||
        nanousd_attribarrayf(prim, "primvars:skel:jointWeights",
                             joint_weights, jw_len) != jw_len) {
        free(joint_indices);
        free(joint_weights);
        goto fail_mesh_joints;
    }

    double geom_bind[16];
    if (!nanousd_attribm4d(prim, "primvars:skel:geomBindTransform", geom_bind))
        memcpy(geom_bind, kIdentity, sizeof(geom_bind));

    double mesh_world[16], mesh_world_inv[16], skel_world[16], skel_to_mesh[16];
    compute_world_xform(prim, mesh_world);
    compute_world_xform(skel, skel_world);
    if (invert_affine_m4d_rowvec(mesh_world, mesh_world_inv))
        nanousd_mul_m4d(skel_world, mesh_world_inv, skel_to_mesh);
    else
        memcpy(skel_to_mesh, kIdentity, sizeof(skel_to_mesh));

    /* Inverse-transpose counterparts for the normal path (geomBindTransform and
     * the skel->mesh change-of-basis can both carry scale). */
    double geom_bind_nm[16], skel_to_mesh_nm[16];
    build_normal_matrix(geom_bind, geom_bind_nm);
    build_normal_matrix(skel_to_mesh, skel_to_mesh_nm);

    for (int v = 0; v < mesh->nvertices; v++) {
        float p0[3] = {
            mesh->positions[v * 3 + 0],
            mesh->positions[v * 3 + 1],
            mesh->positions[v * 3 + 2],
        };
        float p_skel[3];
        xform_point(geom_bind, p0, p_skel);

        double accum[3] = {0.0, 0.0, 0.0};
        double wsum = 0.0;
        for (int k = 0; k < influence_count; k++) {
            int skel_idx = skin_joint_index_for_vertex(
                v, influence_count, joint_indices,
                mesh_joint_count, mesh_to_skel, k);
            if (skel_idx < 0 || skel_idx >= njoints) continue;
            float w = joint_weights[v * influence_count + k];
            if (w == 0.0f) continue;
            float tp[3];
            xform_point(&skin_xforms[skel_idx * 16], p_skel, tp);
            accum[0] += (double)w * tp[0];
            accum[1] += (double)w * tp[1];
            accum[2] += (double)w * tp[2];
            wsum += (double)w;
        }
        if (wsum > 0.0) {
            float p_mesh[3];
            float p_accum[3] = {
                (float)accum[0], (float)accum[1], (float)accum[2],
            };
            xform_point(skel_to_mesh, p_accum, p_mesh);
            mesh->positions[v * 3 + 0] = p_mesh[0];
            mesh->positions[v * 3 + 1] = p_mesh[1];
            mesh->positions[v * 3 + 2] = p_mesh[2];
        }
    }

    if (mesh->normals) {
        for (int v = 0; v < mesh->nvertices; v++) {
            double n_skel[3], accum[3] = {0.0, 0.0, 0.0};
            float n0[3] = {
                mesh->normals[v * 3 + 0],
                mesh->normals[v * 3 + 1],
                mesh->normals[v * 3 + 2],
            };
            xform_dir(geom_bind_nm, n0, n_skel);
            for (int k = 0; k < influence_count; k++) {
                int skel_idx = skin_joint_index_for_vertex(
                    v, influence_count, joint_indices,
                    mesh_joint_count, mesh_to_skel, k);
                if (skel_idx < 0 || skel_idx >= njoints) continue;
                float w = joint_weights[v * influence_count + k];
                double tn[3];
                xform_dir(&skin_normal_xforms[skel_idx * 16],
                          (float[3]){(float)n_skel[0], (float)n_skel[1], (float)n_skel[2]},
                          tn);
                accum[0] += (double)w * tn[0];
                accum[1] += (double)w * tn[1];
                accum[2] += (double)w * tn[2];
            }
            double n_mesh[3];
            xform_dir(skel_to_mesh_nm,
                      (float[3]){(float)accum[0], (float)accum[1], (float)accum[2]},
                      n_mesh);
            normalize_dir3(n_mesh);
            mesh->normals[v * 3 + 0] = (float)n_mesh[0];
            mesh->normals[v * 3 + 1] = (float)n_mesh[1];
            mesh->normals[v * 3 + 2] = (float)n_mesh[2];
        }
    }

    if (fv_normals && fvi_data && fvi_count > 0) {
        for (int j = 0; j < fvi_count; j++) {
            int v = fvi_data[j];
            if (v < 0 || v >= mesh->nvertices) continue;
            double n_skel[3], accum[3] = {0.0, 0.0, 0.0};
            float n0[3] = {
                fv_normals[j * 3 + 0],
                fv_normals[j * 3 + 1],
                fv_normals[j * 3 + 2],
            };
            xform_dir(geom_bind_nm, n0, n_skel);
            for (int k = 0; k < influence_count; k++) {
                int skel_idx = skin_joint_index_for_vertex(
                    v, influence_count, joint_indices,
                    mesh_joint_count, mesh_to_skel, k);
                if (skel_idx < 0 || skel_idx >= njoints) continue;
                float w = joint_weights[v * influence_count + k];
                double tn[3];
                xform_dir(&skin_normal_xforms[skel_idx * 16],
                          (float[3]){(float)n_skel[0], (float)n_skel[1], (float)n_skel[2]},
                          tn);
                accum[0] += (double)w * tn[0];
                accum[1] += (double)w * tn[1];
                accum[2] += (double)w * tn[2];
            }
            double n_mesh[3];
            xform_dir(skel_to_mesh_nm,
                      (float[3]){(float)accum[0], (float)accum[1], (float)accum[2]},
                      n_mesh);
            normalize_dir3(n_mesh);
            fv_normals[j * 3 + 0] = (float)n_mesh[0];
            fv_normals[j * 3 + 1] = (float)n_mesh[1];
            fv_normals[j * 3 + 2] = (float)n_mesh[2];
        }
    }

    if (getenv("NUSD_SKEL_DIAG")) {
        const char* path = nanousd_path(prim);
        fprintf(stderr,
                "[skel_diag] skinned mesh=%s skeleton=%s joints=%d influences=%d\n",
                path ? path : "?", skel_path, njoints, influence_count);
    }

    free(joint_indices);
    free(joint_weights);
    free(mesh_to_skel);
    token_array_free(&mesh_joints);
    token_array_free(&skel_joints);
    free(bind);
    free(rest);
    free(local);
    free(skel_xforms);
    free(skin_xforms);
    free(skin_normal_xforms);
    nanousd_freeprim(skel);
    return 1;

fail_mesh_joints:
    free(mesh_to_skel);
    token_array_free(&mesh_joints);
fail:
    token_array_free(&skel_joints);
    free(bind);
    free(rest);
    free(local);
    free(skel_xforms);
    free(skin_xforms);
    free(skin_normal_xforms);
    nanousd_freeprim(skel);
    return 0;
}

static void bounds_init(float mn[3], float mx[3])
{
    mn[0] = mn[1] = mn[2] = FLT_MAX;
    mx[0] = mx[1] = mx[2] = -FLT_MAX;
}

static void scene_expand_bounds(Scene* scene, const float mn[3], const float mx[3])
{
    if (!scene) return;
    for (int k = 0; k < 3; k++) {
        if (mn[k] < scene->bounds_min[k]) scene->bounds_min[k] = mn[k];
        if (mx[k] > scene->bounds_max[k]) scene->bounds_max[k] = mx[k];
    }
}

static int bounds_is_usable(const float mn[3], const float mx[3])
{
    return isfinite(mn[0]) && isfinite(mn[1]) && isfinite(mn[2]) &&
           isfinite(mx[0]) && isfinite(mx[1]) && isfinite(mx[2]) &&
           mx[0] >= mn[0] && mx[1] >= mn[1] && mx[2] >= mn[2] &&
           mn[0] < FLT_MAX * 0.5f && mx[0] > -FLT_MAX * 0.5f;
}

static void world_bounds_from_local(const double xform[16],
                                    const float local_min[3],
                                    const float local_max[3],
                                    float out_min[3],
                                    float out_max[3])
{
    bounds_init(out_min, out_max);
    for (int xi = 0; xi < 2; xi++) {
        for (int yi = 0; yi < 2; yi++) {
            for (int zi = 0; zi < 2; zi++) {
                float p[3] = {
                    xi ? local_max[0] : local_min[0],
                    yi ? local_max[1] : local_min[1],
                    zi ? local_max[2] : local_min[2],
                };
                float wp[3];
                xform_point(xform, p, wp);
                for (int k = 0; k < 3; k++) {
                    if (wp[k] < out_min[k]) out_min[k] = wp[k];
                    if (wp[k] > out_max[k]) out_max[k] = wp[k];
                }
            }
        }
    }
}

static int mesh_local_bounds(const SceneMesh* mesh, float mn[3], float mx[3])
{
    if (!mesh) return 0;
    memcpy(mn, mesh->local_bounds_min, 3 * sizeof(float));
    memcpy(mx, mesh->local_bounds_max, 3 * sizeof(float));
    if (bounds_is_usable(mn, mx)) return 1;
    bounds_init(mn, mx);
    if (!mesh->positions || mesh->nvertices <= 0) return 0;
    for (int v = 0; v < mesh->nvertices; v++) {
        const float* p = &mesh->positions[v * 3];
        for (int k = 0; k < 3; k++) {
            if (p[k] < mn[k]) mn[k] = p[k];
            if (p[k] > mx[k]) mx[k] = p[k];
        }
    }
    return bounds_is_usable(mn, mx);
}

static void mesh_compute_local_bounds(SceneMesh* mesh)
{
    if (!mesh) return;
    bounds_init(mesh->local_bounds_min, mesh->local_bounds_max);
    if (!mesh->positions || mesh->nvertices <= 0) return;
    for (int v = 0; v < mesh->nvertices; v++) {
        const float* p = &mesh->positions[v * 3];
        for (int k = 0; k < 3; k++) {
            if (p[k] < mesh->local_bounds_min[k]) mesh->local_bounds_min[k] = p[k];
            if (p[k] > mesh->local_bounds_max[k]) mesh->local_bounds_max[k] = p[k];
        }
    }
}

static void mesh_compute_world_bounds_from_local(SceneMesh* mesh, Scene* scene)
{
    if (!mesh) return;
    bounds_init(mesh->bounds_min, mesh->bounds_max);

    const float* mn = mesh->local_bounds_min;
    const float* mx = mesh->local_bounds_max;
    for (int xi = 0; xi < 2; xi++) {
        for (int yi = 0; yi < 2; yi++) {
            for (int zi = 0; zi < 2; zi++) {
                float p[3] = {
                    xi ? mx[0] : mn[0],
                    yi ? mx[1] : mn[1],
                    zi ? mx[2] : mn[2],
                };
                float wp[3];
                xform_point(mesh->world_xform, p, wp);
                for (int k = 0; k < 3; k++) {
                    if (wp[k] < mesh->bounds_min[k]) mesh->bounds_min[k] = wp[k];
                    if (wp[k] > mesh->bounds_max[k]) mesh->bounds_max[k] = wp[k];
                }
            }
        }
    }

    scene_expand_bounds(scene, mesh->bounds_min, mesh->bounds_max);
}

/* Prototype mesh lookup: maps a prim path string to its SceneMesh index.
 *
 * Open-addressing hash table with linear probing. The previous
 * implementation was a flat array with strcmp() linear scan, which
 * became O(ninstances * proto_count) on large scenes (e.g. Kitchen_set:
 * ~1.8K instances; vulkan-warp warehouse: 3472 meshes / 1761 materials).
 * The hash table makes lookup O(1) amortised for the same memory layout.
 */
typedef struct {
    char     path[512];   /* path[0] == '\0' marks an empty slot */
    int      mesh_idx;
    uint64_t hash;        /* cached FNV-1a so probing skips strcmp */
} ProtoMeshSlot;

typedef struct {
    ProtoMeshSlot* slots;
    int            cap;   /* always a power of two; 0 until first insert */
    int            count;
} ProtoMeshHash;

static uint64_t proto_fnv1a64(const char* s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (; *s; ++s) { h ^= (unsigned char)*s; h *= 0x100000001b3ULL; }
    return h;
}

/* FNV-1a over a raw byte buffer — early-geometry-dedup content keying. */
static uint64_t geo_fnv1a64_bytes(uint64_t h, const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ULL; }
    return h;
}

/* Minimal open-addressing uint64->int map for early geometry dedup. Key 0 is
 * reserved for "empty"; the rare geometry that hashes to 0 is simply treated
 * as unique (correctness preserved, one missed dedup). */
typedef struct { uint64_t* keys; int* vals; int cap; int count; } GeoDedupHash;
static void geo_dedup_init(GeoDedupHash* m, int expect) {
    int cap = 1024;
    while (cap < (expect > 0 ? expect : 1) * 2) cap <<= 1;
    m->keys = (uint64_t*)calloc((size_t)cap, sizeof(uint64_t));
    m->vals = (int*)malloc((size_t)cap * sizeof(int));
    m->cap = (m->keys && m->vals) ? cap : 0;
    m->count = 0;
    if (!m->cap) { free(m->keys); free(m->vals); m->keys = NULL; m->vals = NULL; }
}
static void geo_dedup_free(GeoDedupHash* m) {
    free(m->keys); free(m->vals);
    m->keys = NULL; m->vals = NULL; m->cap = 0; m->count = 0;
}
static int geo_dedup_find(const GeoDedupHash* m, uint64_t key) {
    if (!m->cap || !key) return -1;
    uint64_t mask = (uint64_t)m->cap - 1;
    for (uint64_t i = key & mask; ; i = (i + 1) & mask) {
        if (m->keys[i] == 0) return -1;
        if (m->keys[i] == key) return m->vals[i];
    }
}
static void geo_dedup_insert(GeoDedupHash* m, uint64_t key, int val) {
    if (!m->cap || !key) return;
    if (m->count * 4 >= m->cap * 3) return; /* keep load factor <0.75 */
    uint64_t mask = (uint64_t)m->cap - 1;
    for (uint64_t i = key & mask; ; i = (i + 1) & mask) {
        if (m->keys[i] == 0) { m->keys[i] = key; m->vals[i] = val; m->count++; return; }
        if (m->keys[i] == key) return;
    }
}

static void proto_hash_grow(ProtoMeshHash* t) {
    int old_cap = t->cap;
    ProtoMeshSlot* old_slots = t->slots;
    int new_cap = old_cap ? old_cap * 2 : 256;
    t->slots = (ProtoMeshSlot*)calloc((size_t)new_cap, sizeof(ProtoMeshSlot));
    t->cap = new_cap;
    t->count = 0;
    if (!old_slots) return;
    for (int i = 0; i < old_cap; ++i) {
        if (old_slots[i].path[0]) {
            uint32_t mask = (uint32_t)(new_cap - 1);
            uint32_t j = (uint32_t)old_slots[i].hash & mask;
            while (t->slots[j].path[0]) j = (j + 1) & mask;
            t->slots[j] = old_slots[i];
            t->count++;
        }
    }
    free(old_slots);
}

static void proto_hash_insert(ProtoMeshHash* t, const char* key, int mesh_idx) {
    if (t->count * 10 >= t->cap * 7) proto_hash_grow(t);
    uint64_t h = proto_fnv1a64(key);
    uint32_t mask = (uint32_t)(t->cap - 1);
    uint32_t i = (uint32_t)h & mask;
    while (t->slots[i].path[0]) {
        if (t->slots[i].hash == h && strcmp(t->slots[i].path, key) == 0)
            return;  /* duplicate; keep first */
        i = (i + 1) & mask;
    }
    snprintf(t->slots[i].path, sizeof(t->slots[i].path), "%s", key);
    t->slots[i].mesh_idx = mesh_idx;
    t->slots[i].hash = h;
    t->count++;
}

static int proto_hash_lookup(const ProtoMeshHash* t, const char* key) {
    if (!t->cap) return -1;
    uint64_t h = proto_fnv1a64(key);
    uint32_t mask = (uint32_t)(t->cap - 1);
    uint32_t i = (uint32_t)h & mask;
    for (int step = 0; step < t->cap; ++step) {
        const ProtoMeshSlot* s = &t->slots[i];
        if (s->path[0] == '\0') return -1;
        if (s->hash == h && strcmp(s->path, key) == 0) return s->mesh_idx;
        i = (i + 1) & mask;
    }
    return -1;
}

static void proto_hash_free(ProtoMeshHash* t) {
    free(t->slots);
    t->slots = NULL; t->cap = 0; t->count = 0;
}

/* ================================================================
 * Phase 11.A — BasisCurves loader (port of nanousd-opengl-renderer's scene.c
 * helpers; algorithm/conventions match Hydra Storm's basisCurves.glslfx
 * + basisCurvesComputations.cpp:187-393).
 * ================================================================ */

/* Snapshot a nanousd float array attrib into the arena. The zero-copy
 * pointer returned by nanousd_arraydataf may be invalidated by a later
 * call against the same prim, so always copy. */
static const float* curves_snapshot_arrayf(NanousdPrim prim, const char* name,
                                           Arena* arena, int* out_count)
{
    int n = 0;
    const float* d = nanousd_arraydataf(prim, name, &n);
    if (!d || n <= 0) {
        n = nanousd_attribarraylen(prim, name);
        if (n <= 0) { *out_count = 0; return NULL; }
        float* tmp = (float*)arena_alloc(arena, (size_t)n * sizeof(float), 16);
        if (!tmp) { *out_count = 0; return NULL; }
        if (nanousd_attribarrayf(prim, name, tmp, n) <= 0) {
            *out_count = 0; return NULL;
        }
        *out_count = n;
        return tmp;
    }
    float* tmp = (float*)arena_alloc(arena, (size_t)n * sizeof(float), 16);
    if (!tmp) { *out_count = 0; return NULL; }
    memcpy(tmp, d, (size_t)n * sizeof(float));
    *out_count = n;
    return tmp;
}

static const int* curves_snapshot_arrayi(NanousdPrim prim, const char* name,
                                         Arena* arena, int* out_count)
{
    int n = 0;
    const int* d = nanousd_arraydatai(prim, name, &n);
    if (!d || n <= 0) {
        n = nanousd_attribarraylen(prim, name);
        if (n <= 0) { *out_count = 0; return NULL; }
        int* tmp = (int*)arena_alloc(arena, (size_t)n * sizeof(int), 16);
        if (!tmp) { *out_count = 0; return NULL; }
        if (nanousd_attribarrayi(prim, name, tmp, n) <= 0) {
            *out_count = 0; return NULL;
        }
        *out_count = n;
        return tmp;
    }
    int* tmp = (int*)arena_alloc(arena, (size_t)n * sizeof(int), 16);
    if (!tmp) { *out_count = 0; return NULL; }
    memcpy(tmp, d, (size_t)n * sizeof(int));
    *out_count = n;
    return tmp;
}

/* ----------------------------------------------------------------
 * UsdGeomSphere / UsdGeomCylinder: generate triangle meshes CPU-side
 * and push into the existing scene->meshes[] pipeline. This matches the
 * OpenGL renderer and gives Vulkan raster, shadow, and RT one shared
 * representation. Defaults follow UsdGeom: sphere radius=1.0, cylinder
 * radius=1.0 height=2.0 axis="Z".
 * ---------------------------------------------------------------- */

static void gen_uv_sphere(Arena* arena, double radius, int n_lat, int n_lon,
                          float** out_pos, float** out_nrm, uint32_t** out_idx,
                          int* out_nv, int* out_ni)
{
    int nv = n_lat * n_lon;
    int nq = (n_lat - 1) * n_lon;
    int ni = nq * 6;
    float* pos = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    float* nrm = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    uint32_t* idx = (uint32_t*)arena_alloc(arena, (size_t)ni * sizeof(uint32_t), 4);
    if (!pos || !nrm || !idx) {
        *out_pos = NULL; *out_nrm = NULL; *out_idx = NULL;
        *out_nv = *out_ni = 0;
        return;
    }

    int v = 0;
    for (int la = 0; la < n_lat; la++) {
        double th = (double)la / (double)(n_lat - 1) * 3.14159265358979323846;
        double sth = sin(th), cth = cos(th);
        for (int lo = 0; lo < n_lon; lo++) {
            double ph = (double)lo / (double)n_lon * 2.0 * 3.14159265358979323846;
            double sph = sin(ph), cph = cos(ph);
            float nx = (float)(sth * cph), ny = (float)cth, nz = (float)(sth * sph);
            pos[v*3+0] = (float)(radius * nx);
            pos[v*3+1] = (float)(radius * ny);
            pos[v*3+2] = (float)(radius * nz);
            nrm[v*3+0] = nx; nrm[v*3+1] = ny; nrm[v*3+2] = nz;
            v++;
        }
    }

    int q = 0;
    for (int la = 0; la < n_lat - 1; la++) {
        for (int lo = 0; lo < n_lon; lo++) {
            int lo1 = (lo + 1) % n_lon;
            int v00 = la * n_lon + lo;
            int v01 = la * n_lon + lo1;
            int v10 = (la + 1) * n_lon + lo;
            int v11 = (la + 1) * n_lon + lo1;
            idx[q*6+0] = (uint32_t)v00; idx[q*6+1] = (uint32_t)v10; idx[q*6+2] = (uint32_t)v11;
            idx[q*6+3] = (uint32_t)v00; idx[q*6+4] = (uint32_t)v11; idx[q*6+5] = (uint32_t)v01;
            q++;
        }
    }

    *out_pos = pos; *out_nrm = nrm; *out_idx = idx;
    *out_nv = nv;   *out_ni = ni;
}

static void gen_cylinder(Arena* arena, double radius, double height,
                         int axis, int n_sides,
                         float** out_pos, float** out_nrm, uint32_t** out_idx,
                         int* out_nv, int* out_ni)
{
    int n_side = n_sides * 2;
    int n_cap  = (n_sides + 1) * 2;
    int nv = n_side + n_cap;
    int ni = n_sides * 12;
    float* pos = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    float* nrm = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    uint32_t* idx = (uint32_t*)arena_alloc(arena, (size_t)ni * sizeof(uint32_t), 4);
    if (!pos || !nrm || !idx) {
        *out_pos = NULL; *out_nrm = NULL; *out_idx = NULL;
        *out_nv = *out_ni = 0;
        return;
    }

    double half_h = height * 0.5;
    int v = 0;
    int a0 = (axis + 1) % 3;
    int a1 = (axis + 2) % 3;
    int a2 = axis;

    for (int s = 0; s < n_sides; s++) {
        double ph = (double)s / (double)n_sides * 2.0 * 3.14159265358979323846;
        float c = (float)cos(ph), si = (float)sin(ph);
        for (int top = 0; top < 2; top++) {
            float n[3] = {0, 0, 0}; n[a0] = c; n[a1] = si;
            float p[3] = {0, 0, 0};
            p[a0] = (float)(radius * c);
            p[a1] = (float)(radius * si);
            p[a2] = top ? (float)half_h : (float)-half_h;
            pos[v*3+0] = p[0]; pos[v*3+1] = p[1]; pos[v*3+2] = p[2];
            nrm[v*3+0] = n[0]; nrm[v*3+1] = n[1]; nrm[v*3+2] = n[2];
            v++;
        }
    }

    for (int top = 0; top < 2; top++) {
        float n[3] = {0, 0, 0}; n[a2] = top ? 1.0f : -1.0f;
        float p[3] = {0, 0, 0}; p[a2] = top ? (float)half_h : (float)-half_h;
        pos[v*3+0] = p[0]; pos[v*3+1] = p[1]; pos[v*3+2] = p[2];
        nrm[v*3+0] = n[0]; nrm[v*3+1] = n[1]; nrm[v*3+2] = n[2];
        v++;
        for (int s = 0; s < n_sides; s++) {
            double ph = (double)s / (double)n_sides * 2.0 * 3.14159265358979323846;
            float c = (float)cos(ph), si = (float)sin(ph);
            float pp[3] = {0, 0, 0};
            pp[a0] = (float)(radius * c);
            pp[a1] = (float)(radius * si);
            pp[a2] = top ? (float)half_h : (float)-half_h;
            pos[v*3+0] = pp[0]; pos[v*3+1] = pp[1]; pos[v*3+2] = pp[2];
            nrm[v*3+0] = n[0]; nrm[v*3+1] = n[1]; nrm[v*3+2] = n[2];
            v++;
        }
    }

    int q = 0;
    for (int s = 0; s < n_sides; s++) {
        int s2 = ((s + 1) % n_sides) * 2;
        int b0 = s*2, t0 = s*2+1, b1 = s2, t1 = s2+1;
        idx[q*6+0] = (uint32_t)b0; idx[q*6+1] = (uint32_t)t0; idx[q*6+2] = (uint32_t)t1;
        idx[q*6+3] = (uint32_t)b0; idx[q*6+4] = (uint32_t)t1; idx[q*6+5] = (uint32_t)b1;
        q++;
    }

    int bot_center = n_side;
    int top_center = n_side + (n_sides + 1);
    for (int s = 0; s < n_sides; s++) {
        int s1 = (s + 1) % n_sides;
        idx[q*6+0] = (uint32_t)bot_center;
        idx[q*6+1] = (uint32_t)(bot_center + 1 + s1);
        idx[q*6+2] = (uint32_t)(bot_center + 1 + s);
        idx[q*6+3] = (uint32_t)top_center;
        idx[q*6+4] = (uint32_t)(top_center + 1 + s);
        idx[q*6+5] = (uint32_t)(top_center + 1 + s1);
        q++;
    }

    *out_pos = pos; *out_nrm = nrm; *out_idx = idx;
    *out_nv = nv;   *out_ni = ni;
}

static void make_scale_xform(double sx, double sy, double sz, double out[16])
{
    memcpy(out, kIdentity, sizeof(double) * 16);
    out[0] = sx;
    out[5] = sy;
    out[10] = sz;
}

/* UsdGeomCapsule: a cylinder body of length `height` plus two true-radius
 * hemisphere caps. Tessellated at TRUE scale (not unit + non-uniform scale,
 * which would flatten the caps into ellipsoids), so the caller passes an
 * identity shape_xform and lets the prim's world transform supply rotation/
 * translation. Axis-generic (matches gen_cylinder's a0/a1/a2 convention).
 * Degenerate-safe: radius<=0 -> empty mesh; height<=0 -> sphere. */
static void gen_capsule(Arena* arena, double radius, double height,
                        int axis, int n_sides, int n_rings,
                        float** out_pos, float** out_nrm, uint32_t** out_idx,
                        int* out_nv, int* out_ni)
{
    *out_pos = NULL; *out_nrm = NULL; *out_idx = NULL; *out_nv = 0; *out_ni = 0;
    if (radius <= 0.0 || n_sides < 3 || n_rings < 1) return;
    if (height < 0.0) height = 0.0;
    const double PI = 3.14159265358979323846;
    double half_h = height * 0.5;
    int a0 = (axis + 1) % 3, a1 = (axis + 2) % 3, a2 = axis;

    int n_lat = 2 * (n_rings + 1);            /* bottom pole..equator, equator..top pole */
    int nv = n_lat * n_sides;
    int ni = (n_lat - 1) * n_sides * 6;
    float* pos = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    float* nrm = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    uint32_t* idx = (uint32_t*)arena_alloc(arena, (size_t)ni * sizeof(uint32_t), 4);
    if (!pos || !nrm || !idx) return;

    for (int r = 0; r < n_lat; r++) {
        double alpha, a2off;
        if (r <= n_rings) {                    /* bottom hemisphere: alpha -pi/2 .. 0 */
            alpha = -PI * 0.5 + (PI * 0.5) * ((double)r / (double)n_rings);
            a2off = -half_h + radius * sin(alpha);
        } else {                               /* top hemisphere: alpha 0 .. +pi/2 */
            alpha = (PI * 0.5) * ((double)(r - (n_rings + 1)) / (double)n_rings);
            a2off = half_h + radius * sin(alpha);
        }
        double rr = radius * cos(alpha);
        double naxial = sin(alpha), nradial = cos(alpha);
        for (int s = 0; s < n_sides; s++) {
            double ph = (double)s / (double)n_sides * 2.0 * PI;
            float c = (float)cos(ph), si = (float)sin(ph);
            int vi = r * n_sides + s;
            float p[3] = {0, 0, 0}, n[3] = {0, 0, 0};
            p[a0] = (float)(rr * c); p[a1] = (float)(rr * si); p[a2] = (float)a2off;
            n[a0] = (float)(nradial * c); n[a1] = (float)(nradial * si); n[a2] = (float)naxial;
            pos[vi*3+0] = p[0]; pos[vi*3+1] = p[1]; pos[vi*3+2] = p[2];
            nrm[vi*3+0] = n[0]; nrm[vi*3+1] = n[1]; nrm[vi*3+2] = n[2];
        }
    }

    int q = 0;
    for (int r = 0; r < n_lat - 1; r++) {
        for (int s = 0; s < n_sides; s++) {
            int s1 = (s + 1) % n_sides;
            uint32_t a = (uint32_t)(r * n_sides + s);
            uint32_t b = (uint32_t)(r * n_sides + s1);
            uint32_t cc = (uint32_t)((r + 1) * n_sides + s);
            uint32_t d = (uint32_t)((r + 1) * n_sides + s1);
            /* outward winding, matching gen_cylinder's side strip */
            idx[q*6+0] = a;  idx[q*6+1] = cc; idx[q*6+2] = d;
            idx[q*6+3] = a;  idx[q*6+4] = d;  idx[q*6+5] = b;
            q++;
        }
    }
    *out_pos = pos; *out_nrm = nrm; *out_idx = idx; *out_nv = nv; *out_ni = ni;
}

/* UsdGeomCube: axis-aligned box of edge `size` (unit here: edge 2, +-1), scaled
 * by the caller's shape_xform. 24 verts (per-face flat normals) + 12 tris. */
static void gen_cube(Arena* arena, double edge,
                     float** out_pos, float** out_nrm, uint32_t** out_idx,
                     int* out_nv, int* out_ni)
{
    *out_pos = NULL; *out_nrm = NULL; *out_idx = NULL; *out_nv = 0; *out_ni = 0;
    double h = edge * 0.5;
    int nv = 24, ni = 36;
    float* pos = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    float* nrm = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    uint32_t* idx = (uint32_t*)arena_alloc(arena, (size_t)ni * sizeof(uint32_t), 4);
    if (!pos || !nrm || !idx) return;
    /* per face: normal axis (0/1/2), sign, and the two in-plane axes */
    static const int faceN[6] = {0, 0, 1, 1, 2, 2};
    static const float faceS[6] = {1, -1, 1, -1, 1, -1};
    int v = 0, q = 0;
    for (int f = 0; f < 6; f++) {
        int na = faceN[f]; float ns = faceS[f];
        int u = (na + 1) % 3, w = (na + 2) % 3;
        float corners[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        uint32_t base = (uint32_t)v;
        for (int c = 0; c < 4; c++) {
            float p[3] = {0, 0, 0}, n[3] = {0, 0, 0};
            p[na] = (float)(ns * h);
            /* wind so that +sign faces use one order, -sign the mirror, for outward CCW */
            float cu = corners[c][0], cv = corners[c][1];
            p[u] = (float)(cu * h * ns); p[w] = (float)(cv * h);
            n[na] = ns;
            pos[v*3+0] = p[0]; pos[v*3+1] = p[1]; pos[v*3+2] = p[2];
            nrm[v*3+0] = n[0]; nrm[v*3+1] = n[1]; nrm[v*3+2] = n[2];
            v++;
        }
        idx[q++] = base; idx[q++] = base+1; idx[q++] = base+2;
        idx[q++] = base; idx[q++] = base+2; idx[q++] = base+3;
    }
    *out_pos = pos; *out_nrm = nrm; *out_idx = idx; *out_nv = nv; *out_ni = ni;
}

/* UsdGeomCone: base ring of `radius` at -height/2, apex at +height/2, plus a
 * base cap. Unit here (radius 1, height 2); caller's shape_xform scales it.
 * Side normals are the true slant; apex is split per side (face normal),
 * mirroring gen_cylinder's flat-cap convention. Axis-generic. */
static void gen_cone(Arena* arena, double radius, double height,
                     int axis, int n_sides,
                     float** out_pos, float** out_nrm, uint32_t** out_idx,
                     int* out_nv, int* out_ni)
{
    *out_pos = NULL; *out_nrm = NULL; *out_idx = NULL; *out_nv = 0; *out_ni = 0;
    if (radius <= 0.0 || n_sides < 3) return;
    const double PI = 3.14159265358979323846;
    double half_h = height * 0.5;
    int a0 = (axis + 1) % 3, a1 = (axis + 2) % 3, a2 = axis;
    /* side: per segment a base vertex + an apex vertex (split apex). + base cap
     * (center + ring). nv = n_sides*2 (side) + 1 + n_sides (cap). */
    int n_side = n_sides * 2;
    int nv = n_side + 1 + n_sides;
    int ni = n_sides * 3 + n_sides * 3;
    float* pos = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    float* nrm = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    uint32_t* idx = (uint32_t*)arena_alloc(arena, (size_t)ni * sizeof(uint32_t), 4);
    if (!pos || !nrm || !idx) return;
    double slant = sqrt(radius * radius + height * height);
    double n_axial = (slant > 0) ? radius / slant : 0.0;   /* apex points along +axis */
    double n_radial = (slant > 0) ? height / slant : 1.0;
    int v = 0;
    for (int s = 0; s < n_sides; s++) {
        double ph = (double)s / (double)n_sides * 2.0 * PI;
        double phm = ((double)s + 0.5) / (double)n_sides * 2.0 * PI;  /* apex normal at segment mid */
        float c = (float)cos(ph), si = (float)sin(ph);
        float cm = (float)cos(phm), sim = (float)sin(phm);
        /* base vertex (slant normal) */
        float pb[3] = {0,0,0}, nb[3] = {0,0,0};
        pb[a0] = (float)(radius * c); pb[a1] = (float)(radius * si); pb[a2] = (float)-half_h;
        nb[a0] = (float)(n_radial * c); nb[a1] = (float)(n_radial * si); nb[a2] = (float)n_axial;
        pos[v*3+0]=pb[0]; pos[v*3+1]=pb[1]; pos[v*3+2]=pb[2];
        nrm[v*3+0]=nb[0]; nrm[v*3+1]=nb[1]; nrm[v*3+2]=nb[2]; v++;
        /* apex vertex (split, normal at segment mid) */
        float pa[3] = {0,0,0}, na_[3] = {0,0,0};
        pa[a2] = (float)half_h;
        na_[a0] = (float)(n_radial * cm); na_[a1] = (float)(n_radial * sim); na_[a2] = (float)n_axial;
        pos[v*3+0]=pa[0]; pos[v*3+1]=pa[1]; pos[v*3+2]=pa[2];
        nrm[v*3+0]=na_[0]; nrm[v*3+1]=na_[1]; nrm[v*3+2]=na_[2]; v++;
    }
    int cap_center = v;
    float pc[3] = {0,0,0}, nc[3] = {0,0,0}; pc[a2] = (float)-half_h; nc[a2] = -1.0f;
    pos[v*3+0]=pc[0]; pos[v*3+1]=pc[1]; pos[v*3+2]=pc[2];
    nrm[v*3+0]=nc[0]; nrm[v*3+1]=nc[1]; nrm[v*3+2]=nc[2]; v++;
    for (int s = 0; s < n_sides; s++) {
        double ph = (double)s / (double)n_sides * 2.0 * PI;
        float c = (float)cos(ph), si = (float)sin(ph);
        float p[3]={0,0,0}; p[a0]=(float)(radius*c); p[a1]=(float)(radius*si); p[a2]=(float)-half_h;
        pos[v*3+0]=p[0]; pos[v*3+1]=p[1]; pos[v*3+2]=p[2];
        nrm[v*3+0]=nc[0]; nrm[v*3+1]=nc[1]; nrm[v*3+2]=nc[2]; v++;
    }
    int q = 0;
    for (int s = 0; s < n_sides; s++) {
        int s1 = (s + 1) % n_sides;
        /* side tri: base[s], base[s1], apex[s] */
        idx[q++] = (uint32_t)(s*2); idx[q++] = (uint32_t)(s1*2); idx[q++] = (uint32_t)(s*2+1);
        /* base cap tri (downward-facing) */
        idx[q++] = (uint32_t)cap_center;
        idx[q++] = (uint32_t)(cap_center + 1 + s);
        idx[q++] = (uint32_t)(cap_center + 1 + s1);
    }
    *out_pos = pos; *out_nrm = nrm; *out_idx = idx; *out_nv = nv; *out_ni = ni;
}

static void implicit_sphere_xform(NanousdPrim prim, double shape_xform[16])
{
    int ok = 0;
    double radius = nanousd_attribd(prim, "radius", &ok);
    if (!ok) radius = 1.0;
    make_scale_xform(radius, radius, radius, shape_xform);
}

static void implicit_cube_xform(NanousdPrim prim, double shape_xform[16])
{
    int ok = 0;
    double size = nanousd_attribd(prim, "size", &ok);
    if (!ok) size = 2.0;
    /* unit cube is edge 2 (+-1); scale by size/2 to get edge `size`. */
    double s = size * 0.5;
    make_scale_xform(s, s, s, shape_xform);
}

static void implicit_cylinder_xform(NanousdPrim prim, int* out_axis,
                                    double shape_xform[16])
{
    int ok = 0;
    double radius = nanousd_attribd(prim, "radius", &ok);
    if (!ok) radius = 1.0;
    ok = 0;
    double height = nanousd_attribd(prim, "height", &ok);
    if (!ok) height = 2.0;

    int axis = 2;
    int axis_ok = 0;
    const char* axis_tok = nanousd_attrib_token(prim, "axis", &axis_ok);
    if (axis_ok && axis_tok) {
        if      (axis_tok[0] == 'X' || axis_tok[0] == 'x') axis = 0;
        else if (axis_tok[0] == 'Y' || axis_tok[0] == 'y') axis = 1;
        else                                                axis = 2;
    }
    if (out_axis) *out_axis = axis;

    double sx = radius, sy = radius, sz = radius;
    if      (axis == 0) sx = height * 0.5;
    else if (axis == 1) sy = height * 0.5;
    else                sz = height * 0.5;
    make_scale_xform(sx, sy, sz, shape_xform);
}

static void finish_implicit_mesh(SceneMesh* mesh, NanousdPrim prim,
                                 Arena* arena,
                                 const double shape_xform[16],
                                 int compute_local_bounds)
{
    mesh->material_index = -1;
    mesh->prototype_idx = 0;
    mesh->has_display_color = 0;
    mesh->display_color[0] = 0.5f;
    mesh->display_color[1] = 0.5f;
    mesh->display_color[2] = 0.5f;
    mesh->colors = NULL;
    mesh->texcoords = NULL;

    int dc_n = 0;
    const float* dc = curves_snapshot_arrayf(prim, "primvars:displayColor", arena, &dc_n);
    if (dc && dc_n >= 3) {
        mesh->display_color[0] = dc[0];
        mesh->display_color[1] = dc[1];
        mesh->display_color[2] = dc[2];
        mesh->has_display_color = 1;
    }

    double prim_world[16];
    compute_world_xform(prim, prim_world);
    if (shape_xform)
        nanousd_mul_m4d(shape_xform, prim_world, mesh->world_xform);
    else
        memcpy(mesh->world_xform, prim_world, sizeof(double) * 16);

    if (compute_local_bounds)
        mesh_compute_local_bounds(mesh);
    mesh_compute_world_bounds_from_local(mesh, NULL);
}

static void copy_implicit_proto_geometry(SceneMesh* mesh, const SceneMesh* proto)
{
    mesh->positions = proto->positions;
    mesh->normals = proto->normals;
    mesh->indices = proto->indices;
    mesh->nvertices = proto->nvertices;
    mesh->nindices = proto->nindices;
    mesh->texcoords = proto->texcoords;
    mesh->colors = proto->colors;
    memcpy(mesh->local_bounds_min, proto->local_bounds_min, sizeof(float) * 3);
    memcpy(mesh->local_bounds_max, proto->local_bounds_max, sizeof(float) * 3);
}

static int finalize_implicit_mesh(SceneMesh* mesh, NanousdPrim prim,
                                  float* obj_pos, float* obj_nrm,
                                  uint32_t* indices, int nv, int ni,
                                  Arena* arena,
                                  const double shape_xform[16])
{
    if (!obj_pos || !obj_nrm || !indices || nv <= 0 || ni <= 0) return 0;
    mesh->positions = obj_pos;
    mesh->normals = obj_nrm;
    mesh->indices = indices;
    mesh->nvertices = nv;
    mesh->nindices = ni;
    mesh->texcoords = NULL;
    mesh->colors = NULL;
    finish_implicit_mesh(mesh, prim, arena, shape_xform, 1);
    return 1;
}

static int load_sphere(NanousdPrim prim, Arena* arena, SceneMesh* mesh,
                       const double shape_xform[16])
{
    float* pos = NULL;
    float* nrm = NULL;
    uint32_t* idx = NULL;
    int nv = 0, ni = 0;
    gen_uv_sphere(arena, 1.0, 24, 48, &pos, &nrm, &idx, &nv, &ni);
    return finalize_implicit_mesh(mesh, prim, pos, nrm, idx, nv, ni, arena,
                                  shape_xform);
}

static int load_sphere_instance(NanousdPrim prim, Arena* arena,
                                SceneMesh* mesh, const SceneMesh* proto,
                                const double shape_xform[16])
{
    if (!proto || !proto->positions || !proto->indices) return 0;
    copy_implicit_proto_geometry(mesh, proto);
    finish_implicit_mesh(mesh, prim, arena, shape_xform, 0);
    return 1;
}

static int load_cylinder(NanousdPrim prim, Arena* arena, SceneMesh* mesh,
                         int axis, const double shape_xform[16])
{
    float* pos = NULL;
    float* nrm = NULL;
    uint32_t* idx = NULL;
    int nv = 0, ni = 0;
    gen_cylinder(arena, 1.0, 2.0, axis, 32, &pos, &nrm, &idx, &nv, &ni);
    return finalize_implicit_mesh(mesh, prim, pos, nrm, idx, nv, ni, arena,
                                  shape_xform);
}

static int load_cylinder_instance(NanousdPrim prim, Arena* arena,
                                  SceneMesh* mesh, const SceneMesh* proto,
                                  const double shape_xform[16])
{
    if (!proto || !proto->positions || !proto->indices) return 0;
    copy_implicit_proto_geometry(mesh, proto);
    finish_implicit_mesh(mesh, prim, arena, shape_xform, 0);
    return 1;
}

static void implicit_capsule_params(NanousdPrim prim, int* out_axis,
                                    double* out_radius, double* out_height)
{
    int ok = 0;
    double radius = nanousd_attribd(prim, "radius", &ok);
    if (!ok) radius = 0.5;
    ok = 0;
    double height = nanousd_attribd(prim, "height", &ok);
    if (!ok) height = 1.0;
    int axis = 2, axis_ok = 0;
    const char* axis_tok = nanousd_attrib_token(prim, "axis", &axis_ok);
    if (axis_ok && axis_tok) {
        if      (axis_tok[0] == 'X' || axis_tok[0] == 'x') axis = 0;
        else if (axis_tok[0] == 'Y' || axis_tok[0] == 'y') axis = 1;
        else                                                axis = 2;
    }
    if (out_axis) *out_axis = axis;
    if (out_radius) *out_radius = radius;
    if (out_height) *out_height = height;
}

static int load_capsule(NanousdPrim prim, Arena* arena, SceneMesh* mesh,
                        int axis, double radius, double height)
{
    float* pos = NULL; float* nrm = NULL; uint32_t* idx = NULL; int nv = 0, ni = 0;
    gen_capsule(arena, radius, height, axis, 24, 8, &pos, &nrm, &idx, &nv, &ni);
    /* True-scale geometry: identity shape_xform so only the prim world
     * transform (rotation/translation) is applied — no scale to flatten caps. */
    return finalize_implicit_mesh(mesh, prim, pos, nrm, idx, nv, ni, arena, kIdentity);
}

static int load_cube(NanousdPrim prim, Arena* arena, SceneMesh* mesh,
                     const double shape_xform[16])
{
    float* pos = NULL; float* nrm = NULL; uint32_t* idx = NULL; int nv = 0, ni = 0;
    gen_cube(arena, 2.0, &pos, &nrm, &idx, &nv, &ni);   /* unit edge 2 (+-1) */
    return finalize_implicit_mesh(mesh, prim, pos, nrm, idx, nv, ni, arena, shape_xform);
}

static int load_cone(NanousdPrim prim, Arena* arena, SceneMesh* mesh,
                     int axis, const double shape_xform[16])
{
    float* pos = NULL; float* nrm = NULL; uint32_t* idx = NULL; int nv = 0, ni = 0;
    gen_cone(arena, 1.0, 2.0, axis, 32, &pos, &nrm, &idx, &nv, &ni);  /* unit cone */
    return finalize_implicit_mesh(mesh, prim, pos, nrm, idx, nv, ni, arena, shape_xform);
}

static SceneCurveBasis curves_parse_basis(const char* s) {
    if (!s) return CURVE_BASIS_BEZIER;
    if (strcmp(s, "bezier") == 0)     return CURVE_BASIS_BEZIER;
    if (strcmp(s, "bspline") == 0)    return CURVE_BASIS_BSPLINE;
    if (strcmp(s, "catmullRom") == 0) return CURVE_BASIS_CATMULLROM;
    return CURVE_BASIS_BEZIER;
}

static SceneCurveWrap curves_parse_wrap(const char* s) {
    if (!s) return CURVE_WRAP_NONPERIODIC;
    if (strcmp(s, "periodic") == 0)    return CURVE_WRAP_PERIODIC;
    if (strcmp(s, "pinned") == 0)      return CURVE_WRAP_PINNED;
    return CURVE_WRAP_NONPERIODIC;
}

/* Raster BasisCurves draw patches.
 *
 * RT consumes SceneCurve as object-space CVs and materializes linear curve
 * segments only when building a curve BLAS. Raster keeps the same compact
 * CV storage and adds this 4-CV patch index stream for the tessellation
 * pipeline. The topology mirrors the OpenGL path and Storm's
 * HdBasisCurves index builders: cubic patches are 4 control points, while
 * linear segments are packed as [v0, v1, v1, v1] so one pipeline can handle
 * both curve types. */
static int curves_raster_count_cubic_patches(int count,
                                             SceneCurveBasis basis,
                                             SceneCurveWrap wrap)
{
    if (count < 2) return 0;
    int vstep = (basis == CURVE_BASIS_BEZIER) ? 3 : 1;
    int periodic = (wrap == CURVE_WRAP_PERIODIC) ? 1 : 0;
    int pinned = (wrap == CURVE_WRAP_PINNED) &&
                 (basis == CURVE_BASIS_BSPLINE ||
                  basis == CURVE_BASIS_CATMULLROM);
    if (count < 4 && !pinned) return 0;
    int num_segs = periodic ? (count / vstep > 1 ? count / vstep : 1)
                            : ((count - 4 > 0 ? (count - 4) / vstep : 0) + 1);
    if (num_segs < 0) num_segs = 0;
    int extra = pinned ? (basis == CURVE_BASIS_BSPLINE ? 4 : 2) : 0;
    return num_segs + extra;
}

static int curves_raster_count_linear_patches(int count,
                                              SceneCurveWrap wrap,
                                              SceneCurveBasis basis)
{
    if (count < 2) return 0;
    int periodic = (wrap == CURVE_WRAP_PERIODIC) ? 1 : 0;
    int pinned = (wrap == CURVE_WRAP_PINNED) ? 1 : 0;
    int skip_endpoints = (basis == CURVE_BASIS_CATMULLROM) && !pinned;
    int seg = count - 1;
    if (skip_endpoints) seg -= 2;
    if (seg < 0) seg = 0;
    if (periodic) seg += 1;
    return seg;
}

static void curves_raster_emit_cubic_patches(int vertex_base,
                                             int count,
                                             SceneCurveBasis basis,
                                             SceneCurveWrap wrap,
                                             uint32_t* idx,
                                             int* patch_cursor)
{
    int vstep = (basis == CURVE_BASIS_BEZIER) ? 3 : 1;
    int periodic = (wrap == CURVE_WRAP_PERIODIC) ? 1 : 0;
    int pinned = (wrap == CURVE_WRAP_PINNED) &&
                 (basis == CURVE_BASIS_BSPLINE ||
                  basis == CURVE_BASIS_CATMULLROM);

    int p = *patch_cursor;
    if (pinned) {
        int v0 = vertex_base;
        int v1 = (count > 1) ? vertex_base + 1 : v0;
        int v2 = (count > 2) ? vertex_base + 2 : v1;
        if (basis == CURVE_BASIS_BSPLINE) {
            idx[p*4+0] = (uint32_t)v0; idx[p*4+1] = (uint32_t)v0;
            idx[p*4+2] = (uint32_t)v0; idx[p*4+3] = (uint32_t)v1; p++;
        }
        idx[p*4+0] = (uint32_t)v0; idx[p*4+1] = (uint32_t)v0;
        idx[p*4+2] = (uint32_t)v1; idx[p*4+3] = (uint32_t)v2; p++;
    }

    int num_segs = periodic ? (count / vstep > 1 ? count / vstep : 1)
                            : ((count - 4 > 0 ? (count - 4) / vstep : 0) + 1);
    if (count < 4 && !pinned) num_segs = 0;
    for (int i = 0; i < num_segs; i++) {
        int offset = i * vstep;
        for (int vv = 0; vv < 4; vv++) {
            int li = periodic
                ? ((offset + vv) % count)
                : ((offset + vv >= count) ? count - 1 : offset + vv);
            idx[p*4+vv] = (uint32_t)(vertex_base + li);
        }
        p++;
    }

    if (pinned) {
        int n = count;
        int vN_3 = vertex_base + (n >= 3 ? n - 3 : 0);
        int vN_2 = vertex_base + (n >= 2 ? n - 2 : 0);
        int vN_1 = vertex_base + (n >= 1 ? n - 1 : 0);
        idx[p*4+0] = (uint32_t)vN_3; idx[p*4+1] = (uint32_t)vN_2;
        idx[p*4+2] = (uint32_t)vN_1; idx[p*4+3] = (uint32_t)vN_1; p++;
        if (basis == CURVE_BASIS_BSPLINE) {
            idx[p*4+0] = (uint32_t)vN_2; idx[p*4+1] = (uint32_t)vN_1;
            idx[p*4+2] = (uint32_t)vN_1; idx[p*4+3] = (uint32_t)vN_1; p++;
        }
    }

    *patch_cursor = p;
}

static void curves_raster_emit_linear_patches(int vertex_base,
                                              int count,
                                              SceneCurveBasis basis,
                                              SceneCurveWrap wrap,
                                              uint32_t* idx,
                                              int* patch_cursor)
{
    int periodic = (wrap == CURVE_WRAP_PERIODIC) ? 1 : 0;
    int pinned = (wrap == CURVE_WRAP_PINNED) ? 1 : 0;
    int skip_endpoints = (basis == CURVE_BASIS_CATMULLROM) && !pinned;
    int p = *patch_cursor;
    int start = skip_endpoints ? 1 : 0;
    int end = skip_endpoints ? count - 2 : count - 1;
    for (int i = start; i < end; i++) {
        int v0 = vertex_base + i;
        int v1 = vertex_base + i + 1;
        idx[p*4+0] = (uint32_t)v0; idx[p*4+1] = (uint32_t)v1;
        idx[p*4+2] = (uint32_t)v1; idx[p*4+3] = (uint32_t)v1; p++;
    }
    if (periodic) {
        int v0 = vertex_base + count - 1;
        int v1 = vertex_base;
        idx[p*4+0] = (uint32_t)v0; idx[p*4+1] = (uint32_t)v1;
        idx[p*4+2] = (uint32_t)v1; idx[p*4+3] = (uint32_t)v1; p++;
    }
    *patch_cursor = p;
}

static void curves_build_raster_patches(Arena* arena,
                                        const int* counts,
                                        int ncounts,
                                        int type_is_cubic,
                                        SceneCurveBasis basis,
                                        SceneCurveWrap wrap,
                                        uint32_t** out_indices,
                                        int* out_npatches)
{
    *out_indices = NULL;
    *out_npatches = 0;
    if (!counts || ncounts <= 0) return;

    int total = 0;
    for (int c = 0; c < ncounts; c++) {
        if (type_is_cubic)
            total += curves_raster_count_cubic_patches(counts[c], basis, wrap);
        else
            total += curves_raster_count_linear_patches(counts[c], wrap, basis);
    }
    if (total <= 0) return;

    uint32_t* idx = (uint32_t*)arena_alloc(arena,
        (size_t)total * 4 * sizeof(uint32_t), 4);
    if (!idx) return;

    int patch_i = 0;
    int vertex_index = 0;
    for (int c = 0; c < ncounts; c++) {
        int count = counts[c];
        if (type_is_cubic)
            curves_raster_emit_cubic_patches(vertex_index, count, basis, wrap,
                                             idx, &patch_i);
        else
            curves_raster_emit_linear_patches(vertex_index, count, basis, wrap,
                                              idx, &patch_i);
        vertex_index += count;
    }

    *out_indices = idx;
    *out_npatches = total;
}

/* Storm-equivalent fan-out of length-{nv|1|ncurves} primvars to per-CV.
 * Length-based heuristic since we don't expose `interpolation` token yet. */
static int curves_fan_f1(Arena* arena,
                         const float* src, int src_count,
                         int nv, int ncurves, const int* counts,
                         float fallback, float** out)
{
    float* dst = (float*)arena_alloc(arena, (size_t)nv * sizeof(float), 16);
    if (!dst) return 0;
    if (src && src_count == nv) {
        memcpy(dst, src, (size_t)nv * sizeof(float));
    } else if (src && src_count == 1) {
        for (int i = 0; i < nv; i++) dst[i] = src[0];
    } else if (src && src_count == ncurves && counts) {
        int v = 0;
        for (int c = 0; c < ncurves; c++)
            for (int k = 0; k < counts[c] && v < nv; k++)
                dst[v++] = src[c];
        while (v < nv) dst[v++] = fallback;
    } else {
        for (int i = 0; i < nv; i++) dst[i] = fallback;
    }
    *out = dst;
    return 1;
}

static int curves_fan_f3(Arena* arena,
                         const float* src, int src_count,
                         int nv, int ncurves, const int* counts,
                         const float fallback[3], float** out)
{
    float* dst = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    if (!dst) return 0;
    int src_vecs = src_count / 3;
    if (src && src_vecs == nv) {
        memcpy(dst, src, (size_t)nv * 3 * sizeof(float));
    } else if (src && src_vecs == 1) {
        for (int i = 0; i < nv; i++) {
            dst[i*3+0] = src[0]; dst[i*3+1] = src[1]; dst[i*3+2] = src[2];
        }
    } else if (src && src_vecs == ncurves && counts) {
        int v = 0;
        for (int c = 0; c < ncurves; c++) {
            for (int k = 0; k < counts[c] && v < nv; k++) {
                dst[v*3+0] = src[c*3+0];
                dst[v*3+1] = src[c*3+1];
                dst[v*3+2] = src[c*3+2];
                v++;
            }
        }
        while (v < nv) {
            dst[v*3+0] = fallback[0]; dst[v*3+1] = fallback[1]; dst[v*3+2] = fallback[2];
            v++;
        }
    } else {
        for (int i = 0; i < nv; i++) {
            dst[i*3+0] = fallback[0]; dst[i*3+1] = fallback[1]; dst[i*3+2] = fallback[2];
        }
    }
    *out = dst;
    return 1;
}

static int curves_f3_authored_topology_supported(int src_count,
                                                 int nv,
                                                 int ncurves)
{
    if (src_count < 3 || (src_count % 3) != 0) return 0;
    int src_vecs = src_count / 3;
    return src_vecs == 1 || src_vecs == nv || src_vecs == ncurves;
}

static void curves_average_f3(const float* values, int nvalues,
                              float out[3])
{
    double sum[3] = {0.0, 0.0, 0.0};
    if (!values || nvalues <= 0) return;
    for (int i = 0; i < nvalues; i++) {
        sum[0] += (double)values[i*3 + 0];
        sum[1] += (double)values[i*3 + 1];
        sum[2] += (double)values[i*3 + 2];
    }
    double inv = 1.0 / (double)nvalues;
    out[0] = (float)(sum[0] * inv);
    out[1] = (float)(sum[1] * inv);
    out[2] = (float)(sum[2] * inv);
}

/* Load one BasisCurves prim into a SceneCurve. Returns 1 on success, 0 on
 * skip/no-data. */
static int load_basiscurves(NanousdStage stage, NanousdPrim prim,
                            Arena* arena, SceneCurve* curve,
                            MaterialCollection* materials)
{
    /* (a) points */
    int pts_count = 0;
    const float* pts = curves_snapshot_arrayf(prim, "points", arena, &pts_count);
    if (!pts || pts_count < 6) return 0;          /* need at least 2 CVs */
    int nv = pts_count / 3;

    /* (b) curveVertexCounts (default to single curve = nv CVs) */
    int cvc_count = 0;
    const int* cvc = curves_snapshot_arrayi(prim, "curveVertexCounts", arena, &cvc_count);
    if (!cvc || cvc_count <= 0) {
        int* tmp = (int*)arena_alloc(arena, sizeof(int), 4);
        if (!tmp) return 0;
        tmp[0] = nv;
        cvc = tmp;
        cvc_count = 1;
    }
    {
        int sum = 0;
        for (int i = 0; i < cvc_count; i++) sum += cvc[i];
        if (sum != nv) {
            fprintf(stderr, "scene_load: BasisCurves sum(curveVertexCounts)=%d != points=%d, skipping\n",
                    sum, nv);
            return 0;
        }
    }

    /* (c) topology tokens. nanousd_attrib_token reuses an internal buffer
     * across calls, so snapshot each token into a stack buffer. */
    char type_buf[32] = {0}, basis_buf[32] = {0}, wrap_buf[32] = {0};
    int ok_t = 0, ok_b = 0, ok_w = 0;
    {
        const char* s = nanousd_attrib_token(prim, "type", &ok_t);
        if (ok_t && s) snprintf(type_buf, sizeof(type_buf), "%s", s);
    }
    {
        const char* s = nanousd_attrib_token(prim, "basis", &ok_b);
        if (ok_b && s) snprintf(basis_buf, sizeof(basis_buf), "%s", s);
    }
    {
        const char* s = nanousd_attrib_token(prim, "wrap", &ok_w);
        if (ok_w && s) snprintf(wrap_buf, sizeof(wrap_buf), "%s", s);
    }
    int type_is_cubic = (!ok_t || !type_buf[0] || strcmp(type_buf, "cubic") == 0) ? 1 : 0;
    SceneCurveBasis basis = type_is_cubic ? curves_parse_basis(ok_b ? basis_buf : NULL)
                                          : CURVE_BASIS_LINEAR;
    SceneCurveWrap wrap = curves_parse_wrap(ok_w ? wrap_buf : NULL);

    /* (d) USD widths are diameters; fan them to per-CV values. */
    int widths_n = 0;
    const float* widths_src = curves_snapshot_arrayf(prim, "widths", arena, &widths_n);
    float* widths = NULL;
    if (!curves_fan_f1(arena, widths_src, widths_n, nv, cvc_count, cvc,
                       0.05f, &widths)) return 0;

    /* (e) primvars:displayColor (fan to per-CV; uniform color also stored
     * as fallback for future material-override paths). */
    int dc_n = 0;
    const float* dc_src = curves_snapshot_arrayf(prim, "primvars:displayColor", arena, &dc_n);
    float* colors = NULL;
    int has_dc = 0;
    int authored_dc = dc_src && dc_n >= 3 &&
                      curves_f3_authored_topology_supported(dc_n, nv, cvc_count);
    float dc_fallback[3] = { 0.7f, 0.7f, 0.7f };
    if (dc_src && dc_n == 3) {
        curve->display_color[0] = dc_src[0];
        curve->display_color[1] = dc_src[1];
        curve->display_color[2] = dc_src[2];
        has_dc = 1;
        dc_fallback[0] = dc_src[0]; dc_fallback[1] = dc_src[1]; dc_fallback[2] = dc_src[2];
    }
    if (!curves_fan_f3(arena, dc_src, dc_n, nv, cvc_count, cvc,
                       dc_fallback, &colors)) return 0;
    if (!has_dc && authored_dc && colors) {
        curves_average_f3(colors, nv, curve->display_color);
        has_dc = 1;
    }

    /* (f) Authored normals turn BasisCurves into oriented ribbons in
     * Storm/HdSt. Keep them per-CV so segment extraction can emit ribbon
     * intersections without expanding the curves into triangles. */
    int normals_n = 0;
    const float* normals_src =
        curves_snapshot_arrayf(prim, "normals", arena, &normals_n);
    float* normals = NULL;
    int has_normals = 0;
    if (normals_src && normals_n >= 3 &&
        curves_f3_authored_topology_supported(normals_n, nv, cvc_count)) {
        float normal_fallback[3] = { 0.0f, 0.0f, 1.0f };
        if (!curves_fan_f3(arena, normals_src, normals_n, nv, cvc_count, cvc,
                           normal_fallback, &normals)) return 0;
        has_normals = 1;
    }

    int st_n = nanousd_attribarraylen(prim, "primvars:st");

    /* (g) world transform. */
    compute_world_xform(prim, curve->world_xform);
    build_normal_matrix(curve->world_xform, curve->normal_xform);

    /* (h) Phase 11.5.2: keep CVs in OBJECT space (was world-baked). The
     * world matrix lives on the per-curve TLAS instance, so dynamic
     * transforms via nu_set_transforms can update curves at the same
     * cost as meshes — re-uploading 10M segments per frame would defeat
     * the whole renderer.
     *
     * Bounds are still computed in world space (consumers need world
     * bounds for camera framing + scene bounds), so each CV is
     * transformed only for the bounds expansion. The CV array stays
     * object-space. */
    float* obj_cvs = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    if (!obj_cvs) return 0;
    memcpy(obj_cvs, pts, (size_t)nv * 3 * sizeof(float));

    curve->bounds_min[0] = curve->bounds_min[1] = curve->bounds_min[2] =  FLT_MAX;
    curve->bounds_max[0] = curve->bounds_max[1] = curve->bounds_max[2] = -FLT_MAX;
    for (int v = 0; v < nv; v++) {
        float wp[3];
        xform_point(curve->world_xform, &obj_cvs[v*3], wp);
        /* Expand bounds by per-CV radius so the AABB encloses the tube. */
        float r = widths[v] * 0.5f;
        for (int k = 0; k < 3; k++) {
            float lo = wp[k] - r, hi = wp[k] + r;
            if (lo < curve->bounds_min[k]) curve->bounds_min[k] = lo;
            if (hi > curve->bounds_max[k]) curve->bounds_max[k] = hi;
        }
    }
    /* Alias world_cvs to obj_cvs for the persist block below — name kept
     * for diff readability; storage is now object-space. */
    float* world_cvs = obj_cvs;

    uint32_t* patch_indices = NULL;
    int npatches = 0;
    curves_build_raster_patches(arena, cvc, cvc_count, type_is_cubic,
                                basis, wrap, &patch_indices, &npatches);

    /* (i) Persist into SceneCurve. RT consumes the CV/topology data below
     * and expands to linear segments only when an acceleration structure is
     * needed. Raster consumes the same CVs plus patch_indices for a compact
     * tessellation draw path. */
    int* cvc_copy = (int*)arena_alloc(arena, (size_t)cvc_count * sizeof(int), 16);
    if (!cvc_copy) return 0;
    memcpy(cvc_copy, cvc, (size_t)cvc_count * sizeof(int));

    curve->cvs                  = world_cvs;
    curve->widths               = widths;
    curve->colors               = colors;
    curve->normals              = normals;
    curve->nv                   = nv;
    curve->ncurves_in_prim      = cvc_count;
    curve->curve_vertex_counts  = cvc_copy;
    curve->patch_indices        = patch_indices;
    curve->npatches             = npatches;
    curve->basis                = basis;
    curve->wrap                 = wrap;
    curve->type_is_cubic        = type_is_cubic;
    curve->has_display_color    = has_dc;
    curve->has_normals          = has_normals;
    curve->has_texcoords        = st_n > 0;
    curve->texcoord_count       = st_n > 0 ? st_n / 2 : 0;
    curve->material_index       = materials
        ? materials_find_binding(materials, stage, prim)
        : -1;
    return 1;
}

/* ----------------------------------------------------------------
 * Per-segment AABB extraction (Phase 11.A — linear curves only).
 *
 * For type=linear: each consecutive CV pair within a sub-curve emits
 * one cylinder/capsule segment, *constant radius* (per Phase 11.5.1
 * wire-format spec). The segment carries `r0` only; varying-radius
 * input (per-CV widths) is rendered as a series of segments each
 * using its leading CV's radius — Storm's pattern. The AABB is the
 * axis-aligned bound of two spheres at the endpoints, both at `r0`,
 * inflated to also enclose the trailing CV's authored radius so a
 * Phase-11.B cone-sphere intersector can use the same AABBs without
 * a rebuild.
 *
 * `mat_flags` is always 0 in Phase 11.A; Phase 12.2 will populate it
 * with `(material_id << 16) | flags`.
 *
 * Cubic curves are loaded with bounds + counts so the rest of the
 * pipeline sees them, but emit zero segments here; Phase 11.B will
 * add cubic AABB extraction (per-patch bounding from CV hull, padded
 * by max width along the patch).
 * ---------------------------------------------------------------- */
/* Phase 11.B: cubic catmullRom curves tessellate into N_CUBIC_SUBSEGS
 * linear sub-segments per patch. 8 sub-segs is a visual sweet spot for
 * the typical S-curves used as flex hoses; below 4 the cubic look is
 * lost, above 16 the segment budget explodes without visible gain.
 *
 * Moana's full native-instance curve replay can expose hundreds of cubic
 * BasisCurves prims. NUSD_CURVE_SUBSEGS lets memory-limited diagnostic
 * renders keep every curve prim visible while reducing the per-patch segment
 * multiplier; default behavior is unchanged.  Storm adaptively tessellates
 * cubic ribbons up to 40 segments by screen-space length; keep the Vulkan
 * cap high enough for diagnostic parity runs while defaulting to 8 for the
 * full Moana memory budget. */
#define N_CUBIC_SUBSEGS 8
#define N_CUBIC_SUBSEGS_MAX 32

static int scene_curve_cubic_subsegs(void)
{
    const char* env = getenv("NUSD_CURVE_SUBSEGS");
    if (!env || !env[0]) return N_CUBIC_SUBSEGS;
    char* end = NULL;
    long v = strtol(env, &end, 10);
    if (end == env) return N_CUBIC_SUBSEGS;
    if (v < 1) v = 1;
    if (v > N_CUBIC_SUBSEGS_MAX) v = N_CUBIC_SUBSEGS_MAX;
    return (int)v;
}

/* Number of patches in a cubic sub-curve under each (basis, wrap) combo.
 * - catmullRom nonperiodic: each patch needs CVs [i, i+1, i+2, i+3] and
 *   spans CV[i+1]..CV[i+2], so N CVs → max(0, N-3) patches.
 * - catmullRom periodic: every CV starts a patch (wraps around) → N patches.
 * - bezier nonperiodic: 4-CV-per-patch, vstep=3 → (N-1)/3 patches.
 * - bspline nonperiodic: each patch needs CVs [i, i+1, i+2, i+3] and spans
 *   the uniform cubic B-spline interval induced by that knot span. */
static int cubic_patches(int n_cvs, SceneCurveBasis basis, SceneCurveWrap wrap)
{
    if (n_cvs < 4) return 0;
    if (basis == CURVE_BASIS_BEZIER) {
        if (wrap == CURVE_WRAP_PERIODIC) return n_cvs / 3;
        return (n_cvs - 1) / 3;
    }
    /* catmullRom / bspline */
    if (wrap == CURVE_WRAP_PERIODIC) return n_cvs;
    return n_cvs - 3;
}

int scene_curve_count_segments(const SceneCurve* curve)
{
    if (!curve) return 0;
    int total = 0;
    if (curve->type_is_cubic) {
        int sub_segs = scene_curve_cubic_subsegs();
        for (int c = 0; c < curve->ncurves_in_prim; c++) {
            int n = curve->curve_vertex_counts[c];
            int p = cubic_patches(n, curve->basis, curve->wrap);
            total += p * sub_segs;
        }
        return total;
    }
    int v = 0;
    for (int c = 0; c < curve->ncurves_in_prim; c++) {
        int n = curve->curve_vertex_counts[c];
        if (n >= 2) total += (n - 1);
        if (curve->wrap == CURVE_WRAP_PERIODIC && n >= 2) total += 1;
        v += n;
    }
    (void)v;
    return total;
}

/* Catmull-Rom evaluation at parameter t∈[0,1] for the patch
 * P[i+1]..P[i+2] using P[i] and P[i+3] as in/out tangent controls.
 * out[3] is filled. */
static inline void catmull_rom_eval(const float* P0, const float* P1,
                                    const float* P2, const float* P3,
                                    float t, float out[3])
{
    float t2 = t * t, t3 = t2 * t;
    /* Standard centripetal-free Catmull-Rom basis. */
    float c0 = -t3 + 2.0f * t2 - t;
    float c1 =  3.0f * t3 - 5.0f * t2 + 2.0f;
    float c2 = -3.0f * t3 + 4.0f * t2 + t;
    float c3 =        t3 -        t2;
    for (int k = 0; k < 3; k++) {
        out[k] = 0.5f * (c0 * P0[k] + c1 * P1[k] + c2 * P2[k] + c3 * P3[k]);
    }
}

static inline float catmull_rom_eval_scalar(float P0, float P1,
                                            float P2, float P3,
                                            float t)
{
    float t2 = t * t, t3 = t2 * t;
    float c0 = -t3 + 2.0f * t2 - t;
    float c1 =  3.0f * t3 - 5.0f * t2 + 2.0f;
    float c2 = -3.0f * t3 + 4.0f * t2 + t;
    float c3 =        t3 -        t2;
    return 0.5f * (c0 * P0 + c1 * P1 + c2 * P2 + c3 * P3);
}

/* Uniform cubic B-spline evaluation for a 4-CV knot span. Unlike Catmull-Rom,
 * this does not interpolate P1/P2; Ironwood's native-instance curves author
 * basis=bspline, so using the interpolating basis makes branches too large and
 * displaced compared with Storm. */
static inline void bspline_basis(float t, float b[4])
{
    float t2 = t * t, t3 = t2 * t;
    b[0] = (-t3 + 3.0f * t2 - 3.0f * t + 1.0f) / 6.0f;
    b[1] = ( 3.0f * t3 - 6.0f * t2          + 4.0f) / 6.0f;
    b[2] = (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) / 6.0f;
    b[3] = t3 / 6.0f;
}

static inline void bspline_eval(const float* P0, const float* P1,
                                const float* P2, const float* P3,
                                float t, float out[3])
{
    float b[4];
    bspline_basis(t, b);
    for (int k = 0; k < 3; k++) {
        out[k] = b[0] * P0[k] + b[1] * P1[k] + b[2] * P2[k] + b[3] * P3[k];
    }
}

static inline float bspline_eval_scalar(float P0, float P1,
                                        float P2, float P3,
                                        float t)
{
    float b[4];
    bspline_basis(t, b);
    return b[0] * P0 + b[1] * P1 + b[2] * P2 + b[3] * P3;
}

/* Bezier evaluation for a 4-CV patch P0..P3. */
static inline void bezier_eval(const float* P0, const float* P1,
                               const float* P2, const float* P3,
                               float t, float out[3])
{
    float u = 1.0f - t;
    float b0 = u * u * u;
    float b1 = 3.0f * u * u * t;
    float b2 = 3.0f * u * t * t;
    float b3 = t * t * t;
    for (int k = 0; k < 3; k++) {
        out[k] = b0 * P0[k] + b1 * P1[k] + b2 * P2[k] + b3 * P3[k];
    }
}

static inline float bezier_eval_scalar(float P0, float P1,
                                       float P2, float P3,
                                       float t)
{
    float u = 1.0f - t;
    float b0 = u * u * u;
    float b1 = 3.0f * u * u * t;
    float b2 = 3.0f * u * t * t;
    float b3 = t * t * t;
    return b0 * P0 + b1 * P1 + b2 * P2 + b3 * P3;
}

static inline float cubic_width_at(const SceneCurve* curve,
                                   const int idx[4],
                                   float t)
{
    float W0 = curve->widths[idx[0]];
    float W1 = curve->widths[idx[1]];
    float W2 = curve->widths[idx[2]];
    float W3 = curve->widths[idx[3]];
    float w;
    if (curve->basis == CURVE_BASIS_BEZIER) {
        w = bezier_eval_scalar(W0, W1, W2, W3, t);
    } else if (curve->basis == CURVE_BASIS_BSPLINE) {
        w = bspline_eval_scalar(W0, W1, W2, W3, t);
    } else {
        w = catmull_rom_eval_scalar(W0, W1, W2, W3, t);
    }
    return (w > 0.0f) ? w : 0.0f;
}

static inline void cubic_f3_at(const float* values,
                               SceneCurveBasis basis,
                               const int idx[4],
                               float t,
                               float out[3])
{
    const float* P0 = &values[idx[0] * 3];
    const float* P1 = &values[idx[1] * 3];
    const float* P2 = &values[idx[2] * 3];
    const float* P3 = &values[idx[3] * 3];
    if (basis == CURVE_BASIS_BEZIER) {
        bezier_eval(P0, P1, P2, P3, t, out);
    } else if (basis == CURVE_BASIS_BSPLINE) {
        bspline_eval(P0, P1, P2, P3, t, out);
    } else {
        catmull_rom_eval(P0, P1, P2, P3, t, out);
    }
}

static inline float curve_sign_not_zero(float v)
{
    return v < 0.0f ? -1.0f : 1.0f;
}

static uint32_t scene_curve_pack_ribbon_normal(const SceneCurve* curve,
                                               const float n_obj[3])
{
    if (!curve || !n_obj) return 0u;

    double n_world_d[3];
    xform_dir(curve->normal_xform, n_obj, n_world_d);
    double len = sqrt(n_world_d[0] * n_world_d[0] +
                      n_world_d[1] * n_world_d[1] +
                      n_world_d[2] * n_world_d[2]);
    if (len <= 1.0e-12) return 0u;

    float x = (float)(n_world_d[0] / len);
    float y = (float)(n_world_d[1] / len);
    float z = (float)(n_world_d[2] / len);
    float l1 = fabsf(x) + fabsf(y) + fabsf(z);
    if (l1 <= 1.0e-12f) return 0u;
    x /= l1;
    y /= l1;
    z /= l1;
    if (z < 0.0f) {
        float old_x = x;
        x = (1.0f - fabsf(y)) * curve_sign_not_zero(old_x);
        y = (1.0f - fabsf(old_x)) * curve_sign_not_zero(y);
    }

    float ux = fminf(fmaxf(x * 0.5f + 0.5f, 0.0f), 1.0f);
    float uy = fminf(fmaxf(y * 0.5f + 0.5f, 0.0f), 1.0f);
    uint32_t qx = (uint32_t)(ux * (float)SCENE_CURVE_SEG_OCT_MASK + 0.5f);
    uint32_t qy = (uint32_t)(uy * (float)SCENE_CURVE_SEG_OCT_MASK + 0.5f);
    return SCENE_CURVE_SEG_FLAG_RIBBON |
           (qx & SCENE_CURVE_SEG_OCT_MASK) |
           ((qy & SCENE_CURVE_SEG_OCT_MASK) << SCENE_CURVE_SEG_OCT_SHIFT_Y);
}

static uint32_t scene_curve_ribbon_flags_cv(const SceneCurve* curve, int cv)
{
    if (!curve || !curve->has_normals || !curve->normals ||
        cv < 0 || cv >= curve->nv) return 0u;
    return scene_curve_pack_ribbon_normal(curve, &curve->normals[cv * 3]);
}

static uint32_t scene_curve_ribbon_flags_cubic(const SceneCurve* curve,
                                               const int idx[4],
                                               float t)
{
    if (!curve || !curve->has_normals || !curve->normals) return 0u;
    float n_obj[3];
    cubic_f3_at(curve->normals, curve->basis, idx, t, n_obj);
    return scene_curve_pack_ribbon_normal(curve, n_obj);
}

/* Phase 11.B: tessellate one cubic patch into N_CUBIC_SUBSEGS linear
 * segments. CV indices are passed in `idx`[0..3] so we can reuse the
 * same loop body for periodic wrapping. Widths are evaluated through the
 * same cubic basis as positions.
 *
 * Phase 12.x: the trailing radius `r_end` no longer matters here — the
 * GPU AABB-gen pass derives the AABB from `(p0, r0, p1)` using the
 * segment's constant radius (matching the cylinder radius used by the
 * intersection shader). */
static int emit_cubic_patch(const SceneCurve* curve,
                            const int idx[4],
                            int sub_segs,
                            SceneCurveSegment* out_segments)
{
    if (sub_segs < 1) sub_segs = 1;
    if (sub_segs > N_CUBIC_SUBSEGS_MAX) sub_segs = N_CUBIC_SUBSEGS_MAX;

    const float* cvs = curve->cvs;
    const float* P0 = &cvs[idx[0] * 3];
    const float* P1 = &cvs[idx[1] * 3];
    const float* P2 = &cvs[idx[2] * 3];
    const float* P3 = &cvs[idx[3] * 3];

    /* Evaluate the curve at N+1 points in object space, transform to
     * world, then emit N segments. */
    float pts_world[N_CUBIC_SUBSEGS_MAX + 1][3];
    for (int s = 0; s <= sub_segs; s++) {
        float t = (float)s / (float)sub_segs;
        float p_obj[3];
        if (curve->basis == CURVE_BASIS_BEZIER) {
            bezier_eval(P0, P1, P2, P3, t, p_obj);
        } else if (curve->basis == CURVE_BASIS_BSPLINE) {
            bspline_eval(P0, P1, P2, P3, t, p_obj);
        } else {
            catmull_rom_eval(P0, P1, P2, P3, t, p_obj);
        }
        xform_point(curve->world_xform, p_obj, pts_world[s]);
    }

    int written = 0;
    for (int s = 0; s < sub_segs; s++) {
        float t0 = (float)s / (float)sub_segs;
        float r0 = 0.5f * cubic_width_at(curve, idx, t0);

        SceneCurveSegment* seg = &out_segments[written];
        const float* a = pts_world[s];
        const float* b = pts_world[s + 1];
        seg->p0[0] = a[0]; seg->p0[1] = a[1]; seg->p0[2] = a[2];
        seg->r0   = r0;
        seg->p1[0] = b[0]; seg->p1[1] = b[1]; seg->p1[2] = b[2];
        seg->mat_flags = scene_curve_ribbon_flags_cubic(curve, idx, t0);
        if ((seg->mat_flags & SCENE_CURVE_SEG_FLAG_RIBBON) != 0u) {
            seg->mat_flags |= SCENE_CURVE_SEG_FLAG_RIBBON_JOIN_PAD;
        }

        written++;
    }
    return written;
}

int scene_curve_to_segments(const SceneCurve* curve,
                            SceneCurveSegment* out_segments)
{
    if (!curve) return 0;
    if (!out_segments) return 0;

    /* Phase 11.B: cubic curves tessellate to linear sub-segments here.
     * The downstream BLAS build sees them as ordinary linear segments.
     *
     * Parallelisation strategy (Phase 13.x perf): on tera, one cubic
     * BasisCurves prim has ~1.5 M patches × 8 sub-segs = 12 M segments —
     * dwarfing the other 5 prims. So we precompute per-sub-curve patch
     * counts + segment offsets serially (a small prefix sum), then the
     * cost-dominant inner patch loop runs in parallel with deterministic
     * write offsets. Output is byte-identical to the serial version.
     *
     * The pragma is guarded with `if(...)`: small curves (<= 1024 patches)
     * run serially to dodge OpenMP fork/join overhead. */
    if (curve->type_is_cubic) {
        int sub_segs = scene_curve_cubic_subsegs();
        int total_patches = 0;
        for (int c = 0; c < curve->ncurves_in_prim; c++) {
            int n = curve->curve_vertex_counts[c];
            total_patches += cubic_patches(n, curve->basis, curve->wrap);
        }
        if (total_patches == 0) return 0;

        int vstep = (curve->basis == CURVE_BASIS_BEZIER) ? 3 : 1;

        /* Precompute (sub-curve start index, v_off, n_patches) so any
         * patch index `gp` ∈ [0, total_patches) can be mapped to its
         * (c, p) without walking the prefix again per patch. */
        int* sc_start  = (int*)malloc((size_t)curve->ncurves_in_prim * sizeof(int));
        int* sc_voff   = (int*)malloc((size_t)curve->ncurves_in_prim * sizeof(int));
        if (!sc_start || !sc_voff) {
            free(sc_start); free(sc_voff);
            /* Fall back to serial path on alloc failure. */
            int written = 0;
            int v_off = 0;
            for (int c = 0; c < curve->ncurves_in_prim; c++) {
                int n = curve->curve_vertex_counts[c];
                int n_patches = cubic_patches(n, curve->basis, curve->wrap);
                for (int p = 0; p < n_patches; p++) {
                    int base = p * vstep;
                    int idx[4];
                    if (curve->wrap == CURVE_WRAP_PERIODIC) {
                        for (int k = 0; k < 4; k++) idx[k] = v_off + ((base + k) % n);
                    } else {
                        idx[0]=v_off+base; idx[1]=v_off+base+1;
                        idx[2]=v_off+base+2; idx[3]=v_off+base+3;
                    }
                    written += emit_cubic_patch(curve, idx, sub_segs,
                        out_segments + written);
                }
                v_off += n;
            }
            return written;
        }
        {
            int v_off = 0;
            int patch_running = 0;
            for (int c = 0; c < curve->ncurves_in_prim; c++) {
                int n = curve->curve_vertex_counts[c];
                int n_patches = cubic_patches(n, curve->basis, curve->wrap);
                sc_start[c] = patch_running;
                sc_voff[c]  = v_off;
                patch_running += n_patches;
                v_off += n;
            }
        }

        /* Binary-search helper: given a global patch index gp, find the
         * sub-curve c such that sc_start[c] <= gp < sc_start[c+1]. */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(total_patches > 1024)
        #endif
        for (int gp = 0; gp < total_patches; gp++) {
            /* Lower-bound search across sc_start. ncurves_in_prim is up
             * to ~1.5 M for tera but per-thread this is hot in cache. */
            int lo = 0, hi = curve->ncurves_in_prim - 1;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (sc_start[mid] <= gp) lo = mid;
                else hi = mid - 1;
            }
            int c = lo;
            int p = gp - sc_start[c];
            int n = curve->curve_vertex_counts[c];
            int v_off = sc_voff[c];
            int base = p * vstep;
            int idx[4];
            if (curve->wrap == CURVE_WRAP_PERIODIC) {
                for (int k = 0; k < 4; k++) idx[k] = v_off + ((base + k) % n);
            } else {
                idx[0] = v_off + base + 0;
                idx[1] = v_off + base + 1;
                idx[2] = v_off + base + 2;
                idx[3] = v_off + base + 3;
            }
            int seg_off = gp * sub_segs;
            emit_cubic_patch(curve, idx, sub_segs,
                out_segments + seg_off);
        }

        free(sc_start);
        free(sc_voff);
        return total_patches * sub_segs;
    }

    /* Phase 11.A.2.5: emit world-space segments (transformed by
     * curve->world_xform during extraction). Single flat BLAS strategy
     * (Phase 12.1) wants all segments in a shared coordinate space; the
     * TLAS instance can then carry an identity transform. CVs remain
     * object-space (Phase 11.5.2) so dynamic-transform consumers can
     * re-extract per-frame against fresh world matrices without
     * touching scene state.
     * Linear path is parallelised the same way as cubic: precompute
     * per-sub-curve segment offsets, then parallel-for over a flat
     * range. The body is cheap (4 mul-adds), and as of phase 12.x AABBs
     * are no longer authored host-side — the GPU compute pass derives
     * them from `(p0, p1, r0)` using the segment's constant radius.
     * Threshold `if(total > 4096)` amortises OpenMP fork/join. */
    int total_segs = 0;
    for (int c = 0; c < curve->ncurves_in_prim; c++) {
        int n = curve->curve_vertex_counts[c];
        if (n < 2) continue;
        total_segs += (n - 1) + (curve->wrap == CURVE_WRAP_PERIODIC ? 1 : 0);
    }
    if (total_segs == 0) return 0;

    int* sc_start = (int*)malloc((size_t)curve->ncurves_in_prim * sizeof(int));
    int* sc_voff  = (int*)malloc((size_t)curve->ncurves_in_prim * sizeof(int));
    if (!sc_start || !sc_voff) {
        free(sc_start); free(sc_voff);
        /* Fallback: serial original. */
        int written = 0;
        int v_off = 0;
        for (int c = 0; c < curve->ncurves_in_prim; c++) {
            int n = curve->curve_vertex_counts[c];
            if (n < 2) { v_off += n; continue; }
            int n_segs = n - 1 + (curve->wrap == CURVE_WRAP_PERIODIC ? 1 : 0);
            for (int s = 0; s < n_segs; s++) {
                int i0 = v_off + s;
                int i1 = (s == n - 1) ? v_off : v_off + s + 1;
                float p0[3], p1[3];
                xform_point(curve->world_xform, &curve->cvs[i0 * 3], p0);
                xform_point(curve->world_xform, &curve->cvs[i1 * 3], p1);
                float r0 = curve->widths[i0] * 0.5f;
                SceneCurveSegment* seg = &out_segments[written];
                seg->p0[0]=p0[0]; seg->p0[1]=p0[1]; seg->p0[2]=p0[2]; seg->r0=r0;
                seg->p1[0]=p1[0]; seg->p1[1]=p1[1]; seg->p1[2]=p1[2];
                seg->mat_flags = scene_curve_ribbon_flags_cv(curve, i0);
                if (n_segs > 1 &&
                    (seg->mat_flags & SCENE_CURVE_SEG_FLAG_RIBBON) != 0u) {
                    seg->mat_flags |= SCENE_CURVE_SEG_FLAG_RIBBON_JOIN_PAD;
                }
                written++;
            }
            v_off += n;
        }
        return written;
    }
    {
        int v_off = 0;
        int seg_running = 0;
        for (int c = 0; c < curve->ncurves_in_prim; c++) {
            int n = curve->curve_vertex_counts[c];
            sc_start[c] = seg_running;
            sc_voff[c]  = v_off;
            if (n >= 2) {
                seg_running += (n - 1) + (curve->wrap == CURVE_WRAP_PERIODIC ? 1 : 0);
            }
            v_off += n;
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(total_segs > 4096)
    #endif
    for (int gs = 0; gs < total_segs; gs++) {
        /* Map global segment index gs back to (c, s) via lower-bound. */
        int lo = 0, hi = curve->ncurves_in_prim - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (sc_start[mid] <= gs) lo = mid;
            else hi = mid - 1;
        }
        int c = lo;
        int s = gs - sc_start[c];
        int n = curve->curve_vertex_counts[c];
        int v_off = sc_voff[c];
        int n_segs = (n - 1) + (curve->wrap == CURVE_WRAP_PERIODIC ? 1 : 0);
        if (n < 2 || s >= n_segs) continue;  /* defensive */

        int i0 = v_off + s;
        int i1 = (s == n - 1) ? v_off : v_off + s + 1;
        float p0[3], p1[3];
        xform_point(curve->world_xform, &curve->cvs[i0 * 3], p0);
        xform_point(curve->world_xform, &curve->cvs[i1 * 3], p1);
        float r0 = curve->widths[i0] * 0.5f;

        SceneCurveSegment* seg = &out_segments[gs];
        seg->p0[0] = p0[0]; seg->p0[1] = p0[1]; seg->p0[2] = p0[2];
        seg->r0   = r0;
        seg->p1[0] = p1[0]; seg->p1[1] = p1[1]; seg->p1[2] = p1[2];
        seg->mat_flags = scene_curve_ribbon_flags_cv(curve, i0);
        if (n_segs > 1 &&
            (seg->mat_flags & SCENE_CURVE_SEG_FLAG_RIBBON) != 0u) {
            seg->mat_flags |= SCENE_CURVE_SEG_FLAG_RIBBON_JOIN_PAD;
        }
    }

    free(sc_start);
    free(sc_voff);
    return total_segs;
}

static inline void write_rgb(float* out, int idx, const float c[3])
{
    out[idx*3 + 0] = c[0];
    out[idx*3 + 1] = c[1];
    out[idx*3 + 2] = c[2];
}

int scene_curve_to_segment_colors(const SceneCurve* curve,
                                  float* out_rgb)
{
    if (!curve || !out_rgb || !curve->colors) return 0;

    if (curve->type_is_cubic) {
        int sub_segs = scene_curve_cubic_subsegs();
        int total_patches = 0;
        for (int c = 0; c < curve->ncurves_in_prim; c++) {
            int n = curve->curve_vertex_counts[c];
            total_patches += cubic_patches(n, curve->basis, curve->wrap);
        }
        if (total_patches == 0) return 0;

        int vstep = (curve->basis == CURVE_BASIS_BEZIER) ? 3 : 1;
        int* sc_start = (int*)malloc((size_t)curve->ncurves_in_prim * sizeof(int));
        int* sc_voff  = (int*)malloc((size_t)curve->ncurves_in_prim * sizeof(int));
        if (!sc_start || !sc_voff) {
            free(sc_start); free(sc_voff);
            int written = 0;
            int v_off = 0;
            for (int c = 0; c < curve->ncurves_in_prim; c++) {
                int n = curve->curve_vertex_counts[c];
                int n_patches = cubic_patches(n, curve->basis, curve->wrap);
                for (int p = 0; p < n_patches; p++) {
                    int base = p * vstep;
                    int idx[4];
                    if (curve->wrap == CURVE_WRAP_PERIODIC) {
                        for (int k = 0; k < 4; k++) idx[k] = v_off + ((base + k) % n);
                    } else {
                        idx[0]=v_off+base; idx[1]=v_off+base+1;
                        idx[2]=v_off+base+2; idx[3]=v_off+base+3;
                    }
                    for (int s = 0; s < sub_segs; s++) {
                        float t0 = (float)s / (float)sub_segs;
                        float color[3];
                        cubic_f3_at(curve->colors, curve->basis, idx, t0, color);
                        write_rgb(out_rgb, written++, color);
                    }
                }
                v_off += n;
            }
            return written;
        }

        {
            int v_off = 0;
            int patch_running = 0;
            for (int c = 0; c < curve->ncurves_in_prim; c++) {
                int n = curve->curve_vertex_counts[c];
                int n_patches = cubic_patches(n, curve->basis, curve->wrap);
                sc_start[c] = patch_running;
                sc_voff[c] = v_off;
                patch_running += n_patches;
                v_off += n;
            }
        }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(total_patches > 1024)
        #endif
        for (int gp = 0; gp < total_patches; gp++) {
            int lo = 0, hi = curve->ncurves_in_prim - 1;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (sc_start[mid] <= gp) lo = mid;
                else hi = mid - 1;
            }
            int c = lo;
            int p = gp - sc_start[c];
            int n = curve->curve_vertex_counts[c];
            int v_off = sc_voff[c];
            int base = p * vstep;
            int idx[4];
            if (curve->wrap == CURVE_WRAP_PERIODIC) {
                for (int k = 0; k < 4; k++) idx[k] = v_off + ((base + k) % n);
            } else {
                idx[0] = v_off + base + 0;
                idx[1] = v_off + base + 1;
                idx[2] = v_off + base + 2;
                idx[3] = v_off + base + 3;
            }
            int seg_off = gp * sub_segs;
            for (int s = 0; s < sub_segs; s++) {
                float t0 = (float)s / (float)sub_segs;
                float color[3];
                cubic_f3_at(curve->colors, curve->basis, idx, t0, color);
                write_rgb(out_rgb, seg_off + s, color);
            }
        }

        free(sc_start);
        free(sc_voff);
        return total_patches * sub_segs;
    }

    int total_segs = 0;
    for (int c = 0; c < curve->ncurves_in_prim; c++) {
        int n = curve->curve_vertex_counts[c];
        if (n < 2) continue;
        total_segs += (n - 1) + (curve->wrap == CURVE_WRAP_PERIODIC ? 1 : 0);
    }
    if (total_segs == 0) return 0;

    int* sc_start = (int*)malloc((size_t)curve->ncurves_in_prim * sizeof(int));
    int* sc_voff  = (int*)malloc((size_t)curve->ncurves_in_prim * sizeof(int));
    if (!sc_start || !sc_voff) {
        free(sc_start); free(sc_voff);
        int written = 0;
        int v_off = 0;
        for (int c = 0; c < curve->ncurves_in_prim; c++) {
            int n = curve->curve_vertex_counts[c];
            if (n < 2) { v_off += n; continue; }
            int n_segs = n - 1 + (curve->wrap == CURVE_WRAP_PERIODIC ? 1 : 0);
            for (int s = 0; s < n_segs; s++) {
                int i0 = v_off + s;
                write_rgb(out_rgb, written++, &curve->colors[i0 * 3]);
            }
            v_off += n;
        }
        return written;
    }

    {
        int v_off = 0;
        int seg_running = 0;
        for (int c = 0; c < curve->ncurves_in_prim; c++) {
            int n = curve->curve_vertex_counts[c];
            sc_start[c] = seg_running;
            sc_voff[c] = v_off;
            if (n >= 2)
                seg_running += (n - 1) + (curve->wrap == CURVE_WRAP_PERIODIC ? 1 : 0);
            v_off += n;
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(total_segs > 4096)
    #endif
    for (int gs = 0; gs < total_segs; gs++) {
        int lo = 0, hi = curve->ncurves_in_prim - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (sc_start[mid] <= gs) lo = mid;
            else hi = mid - 1;
        }
        int c = lo;
        int s = gs - sc_start[c];
        int n = curve->curve_vertex_counts[c];
        int v_off = sc_voff[c];
        int n_segs = (n - 1) + (curve->wrap == CURVE_WRAP_PERIODIC ? 1 : 0);
        if (n < 2 || s >= n_segs) continue;
        int i0 = v_off + s;
        write_rgb(out_rgb, gs, &curve->colors[i0 * 3]);
    }

    free(sc_start);
    free(sc_voff);
    return total_segs;
}

/* Iterate stage and load all BasisCurves into scene->curves.
 *
 * Parallelisation (Phase 13.x perf): per-prim curve loading is trivially
 * data-parallel — each prim parses its own widths/colors/points, fans
 * interpolation, and computes its own world-space bounds. With OpenMP
 * we spawn N threads, give each thread a private arena to allocate
 * into, then splice those private arenas onto the main arena after the
 * parallel region closes. Pointers handed out by per-thread arenas
 * stay valid (block_data is non-relocating).
 *
 * Determinism: each curve writes into a distinct `scene->curves[i]`
 * slot keyed by its position in the (serially built) curve_prim_idx
 * array, so the output array order is identical to the serial version.
 * Bounds reduction is min/max — associative + commutative, so the
 * result is deterministic regardless of merge order.
 *
 * Falls back to a single-arena serial path when _OPENMP isn't
 * defined (the #ifdef guards the parallel region). */
static int scene_load_curves(NanousdStage stage, Scene* scene, Arena* arena)
{
    int nprims = nanousd_nprims(stage);
    MaterialCollection* materials = (MaterialCollection*)scene->materials;

    /* Pass 1: collect indices of all active BasisCurves prims. We need
     * them as a flat array so the parallel for in Pass 2 can iterate by
     * dense index without an atomic counter. */
    int* curve_prim_idx = NULL;
    int  n_curve_prims = 0;
    {
        /* Two-step: first count to size the index array, then fill it. */
        int max_curves = 0;
        for (int i = 0; i < nprims; i++) {
            NanousdPrim p = nanousd_prim(stage, i);
            if (!p) continue;
            if (!nanousd_isactive(p)) { nanousd_freeprim(p); continue; }
            const char* tn = nanousd_typename(p);
            if (tn && !strcmp(tn, "BasisCurves")) max_curves++;
            nanousd_freeprim(p);
        }
        if (max_curves == 0) return 0;

        curve_prim_idx = (int*)arena_alloc(arena, (size_t)max_curves * sizeof(int), 16);
        if (!curve_prim_idx) return -1;
        for (int i = 0; i < nprims; i++) {
            NanousdPrim p = nanousd_prim(stage, i);
            if (!p) continue;
            if (!nanousd_isactive(p)) { nanousd_freeprim(p); continue; }
            const char* tn = nanousd_typename(p);
            if (tn && !strcmp(tn, "BasisCurves")) {
                curve_prim_idx[n_curve_prims++] = i;
            }
            nanousd_freeprim(p);
        }
    }

    scene->curves = (SceneCurve*)arena_calloc(arena, (size_t)n_curve_prims, sizeof(SceneCurve));
    if (!scene->curves) return -1;

    /* Pass 2: load each prim into its slot. Each parallel thread gets a
     * private arena (so arena_alloc isn't contended) plus per-thread
     * bounds + cubic-counter accumulators. After the parallel region we
     * splice the private arenas onto the main arena and merge the
     * scalar reductions serially. */
    int n_cubic_tessellated = 0;
    int n_loaded = 0;

    /* Per-curve "did this load succeed?" flag — needed to compact failed
     * loads out of the curves array after the parallel section. */
    char* loaded_ok = (char*)arena_alloc(arena,
        (size_t)n_curve_prims * sizeof(char), 16);
    if (!loaded_ok) return -1;
    memset(loaded_ok, 0, (size_t)n_curve_prims * sizeof(char));

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads > n_curve_prims) nthreads = n_curve_prims;
    if (nthreads < 1) nthreads = 1;

    /* Per-thread arenas + reduction accumulators. Arenas live on the
     * stack-allocated array but their internal blocks are heap-malloc'd,
     * so ownership transfers cleanly via arena_splice. */
    Arena* tarenas = (Arena*)calloc((size_t)nthreads, sizeof(Arena));
    float (*tmins)[3] = (float (*)[3])malloc((size_t)nthreads * 3 * sizeof(float));
    float (*tmaxs)[3] = (float (*)[3])malloc((size_t)nthreads * 3 * sizeof(float));
    int*   tcubic    = (int*)calloc((size_t)nthreads, sizeof(int));
    int*   tloaded   = (int*)calloc((size_t)nthreads, sizeof(int));
    if (!tarenas || !tmins || !tmaxs || !tcubic || !tloaded) {
        free(tarenas); free(tmins); free(tmaxs); free(tcubic); free(tloaded);
        return -1;
    }
    for (int t = 0; t < nthreads; t++) {
        /* 8 MiB default block — large enough that most curves' fan-out
         * arrays land in a single block per thread, minimising malloc
         * traffic. The arena grows as needed for outsized curves. */
        tarenas[t] = arena_create(8 * 1024 * 1024);
        tmins[t][0] = tmins[t][1] = tmins[t][2] =  FLT_MAX;
        tmaxs[t][0] = tmaxs[t][1] = tmaxs[t][2] = -FLT_MAX;
    }

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
    for (int k = 0; k < n_curve_prims; k++) {
        int tid = omp_get_thread_num();
        Arena* tarena = &tarenas[tid];
        int prim_index = curve_prim_idx[k];
        NanousdPrim p = nanousd_prim(stage, prim_index);
        if (!p) continue;
        SceneCurve* curve = &scene->curves[k];
        if (load_basiscurves(stage, p, tarena, curve, materials)) {
            loaded_ok[k] = 1;
            tloaded[tid]++;
            if (curve->type_is_cubic) tcubic[tid]++;
            for (int j = 0; j < 3; j++) {
                if (curve->bounds_min[j] < tmins[tid][j]) tmins[tid][j] = curve->bounds_min[j];
                if (curve->bounds_max[j] > tmaxs[tid][j]) tmaxs[tid][j] = curve->bounds_max[j];
            }
        }
        nanousd_freeprim(p);
    }

    /* Serial merge: splice arenas, sum counters, reduce bounds. */
    for (int t = 0; t < nthreads; t++) {
        arena_splice(arena, &tarenas[t]);
        n_cubic_tessellated += tcubic[t];
        n_loaded            += tloaded[t];
        for (int j = 0; j < 3; j++) {
            if (tmins[t][j] < scene->bounds_min[j]) scene->bounds_min[j] = tmins[t][j];
            if (tmaxs[t][j] > scene->bounds_max[j]) scene->bounds_max[j] = tmaxs[t][j];
        }
    }
    free(tarenas); free(tmins); free(tmaxs); free(tcubic); free(tloaded);
#else
    /* Serial fallback (no OpenMP). */
    for (int k = 0; k < n_curve_prims; k++) {
        int prim_index = curve_prim_idx[k];
        NanousdPrim p = nanousd_prim(stage, prim_index);
        if (!p) continue;
        SceneCurve* curve = &scene->curves[k];
        if (load_basiscurves(stage, p, arena, curve, materials)) {
            loaded_ok[k] = 1;
            n_loaded++;
            if (curve->type_is_cubic) n_cubic_tessellated++;
            for (int j = 0; j < 3; j++) {
                if (curve->bounds_min[j] < scene->bounds_min[j]) scene->bounds_min[j] = curve->bounds_min[j];
                if (curve->bounds_max[j] > scene->bounds_max[j]) scene->bounds_max[j] = curve->bounds_max[j];
            }
        }
        nanousd_freeprim(p);
    }
#endif

    /* Compact: load_basiscurves can fail (e.g. mismatched curveVertexCounts).
     * The serial version skipped failed loads and densely packed surviving
     * curves. To stay byte-identical we do the same compaction here, in
     * the original prim order. */
    int curve_idx = 0;
    if (n_loaded < n_curve_prims) {
        for (int k = 0; k < n_curve_prims; k++) {
            if (!loaded_ok[k]) continue;
            if (curve_idx != k) {
                scene->curves[curve_idx] = scene->curves[k];
            }
            curve_idx++;
        }
    } else {
        curve_idx = n_loaded;
    }
    scene->ncurves = curve_idx;

    int bound_curves = 0;
    for (int i = 0; i < curve_idx; i++) {
        if (scene->curves[i].material_index >= 0)
            bound_curves++;
    }

    if (n_cubic_tessellated > 0) {
        fprintf(stderr, "scene_load: %d cubic BasisCurves tessellated to %d linear sub-segments per patch (Phase 11.B)\n",
                n_cubic_tessellated, scene_curve_cubic_subsegs());
    }
    if (curve_idx > 0) {
        fprintf(stderr,
                "scene_load: loaded %d BasisCurves prims (%d material-bound)\n",
                curve_idx, bound_curves);
    }
    return curve_idx;
}

/* ----------------------------------------------------------------
 * scene_load
 * ---------------------------------------------------------------- */
Scene* scene_load(const char* filepath)
{
    int do_timing = (getenv("NUSD_LOAD_TIMING") != NULL);

    /* Phase 0 geometry cache (NUSD_GEO_CACHE=1): on a hit, reconstruct the
     * Scene from a meshopt-encoded sidecar and skip USD parse entirely.
     * Material-enabled loads require a material-capable sidecar; geometry-only
     * sidecars are treated as misses so warm/cold visual output stays
     * comparable. See docs/planning/MESHLET_GEOMETRY_CACHE_PLAN.md. */
    int geo_cache = geo_cache_enabled();
    if (geo_cache) {
        double tc0 = get_time_sec();
        Scene* cached = geo_cache_try_load(filepath, s_load_materials);
        if (cached) {
            fprintf(stderr,
                    "[geo_cache] HIT: %s — %d meshes, %d lights, "
                    "%d meshlets, %.1f ms\n",
                    filepath, cached->nmeshes, cached->nlights,
                    cached->nmeshlets, (get_time_sec() - tc0) * 1000.0);
            return cached;
        }
    }

    /* Open the stage and hand it to scene_load_from_stage with ownership.
     * nanousd_open does USDA parse + Compose + Populate eagerly; time it
     * here so the bucket is visible to NUSD_LOAD_TIMING. Previously this
     * cost was hidden because scene_load_from_stage's t0 starts after the
     * open returns (OpenGL/Vulkan timing parity bug). */
    double t_open0 = do_timing ? get_time_sec() : 0.0;
    NanousdStage stage = nanousd_open(filepath);
    if (!nanousd_isvalid(stage)) {
        const char* err = nanousd_error(stage);
        fprintf(stderr, "scene_load: failed to open '%s': %s\n",
                filepath, err ? err : "unknown error");
        if (stage) nanousd_close(stage);
        return NULL;
    }
    if (do_timing) {
        fprintf(stderr, "  [scene_timing] nanousd_open (parse+compose+populate): %.1f ms\n",
                (get_time_sec() - t_open0) * 1000.0);
    }
    Scene* scene = scene_load_from_stage((void*)stage, filepath);
    if (!scene) {
        nanousd_close(stage);
        return NULL;
    }
    /* Transfer ownership: scene_free will close it. */
    scene->_owns_stage = 1;

    /* Phase 0 geometry cache: persist the freshly-parsed geometry-only scene
     * so the next load takes the fast path. Best-effort — a failed write is
     * non-fatal (the Scene is already valid). */
    if (geo_cache) {
        if (geo_cache_write(filepath, scene) == 0)
            fprintf(stderr, "[geo_cache] MISS — wrote cache: %s (%d meshes)\n",
                    filepath, scene->nmeshes);
        else
            fprintf(stderr, "[geo_cache] MISS — not cached "
                    "(curves present or write failed): %s\n",
                    filepath);
    }
    return scene;
}

/* ----------------------------------------------------------------
 * USD GeomSubset material-subset splitting (ported from the Metal
 * backend, commit c36bfa9). A Mesh with face-domain GeomSubset
 * children that bind different materials is split into one submesh
 * per subset (plus a remainder for unbound faces), each with its own
 * material_index. Submeshes share the base mesh's vertex buffers
 * (positions/normals) and differ only in their index partition, so
 * they inherit the base's world_xform/bounds via struct copy. Without
 * this, a multi-material mesh renders with a single material.
 * ---------------------------------------------------------------- */
typedef struct {
    NanousdPrim prim;
    int*        face_indices;   /* authored subset face list (arena copy) */
    int         nfaces;
    int         nindices;       /* triangulated index count for the subset */
    int         material_index;
} GeomSubsetInfo;

/* Map each authored face to its [start,count) range in the mesh's
 * fan-triangulated index buffer (must match triangulate_faces()). */
static int build_mesh_face_index_ranges(Arena* arena, const SceneMesh* mesh,
                                        const int* fvc_data, int fvc_count,
                                        int fvi_count, int** out_starts,
                                        int** out_counts, int* out_nfaces) {
    if (!arena || !mesh || !mesh->indices || mesh->nindices <= 0 ||
        !out_starts || !out_counts || !out_nfaces)
        return 0;
    int nfaces = (fvc_data && fvc_count > 0) ? fvc_count : (mesh->nindices / 3);
    if (nfaces <= 0) return 0;
    int* starts = (int*)arena_alloc(arena, (size_t)nfaces * sizeof(int), 16);
    int* counts = (int*)arena_calloc(arena, (size_t)nfaces, sizeof(int));
    if (!starts || !counts) return 0;
    for (int i = 0; i < nfaces; i++) starts[i] = -1;
    if (fvc_data && fvc_count > 0) {
        int fvi_offset = 0, tri_offset = 0;
        for (int f = 0; f < fvc_count; f++) {
            int n = fvc_data[f];
            if (n <= 0) continue;
            if (fvi_offset + n > fvi_count) break;
            if (n >= 3) {
                int ni = (n - 2) * 3;
                if (tri_offset + ni > mesh->nindices) break;
                starts[f] = tri_offset; counts[f] = ni; tri_offset += ni;
            }
            fvi_offset += n;
        }
    } else {
        for (int f = 0; f < nfaces; f++) { starts[f] = f * 3; counts[f] = 3; }
    }
    *out_starts = starts; *out_counts = counts; *out_nfaces = nfaces;
    return 1;
}

static int count_subset_indices(const int* subset_faces, int subset_face_count,
                                const int* face_counts, int nfaces) {
    int total = 0;
    for (int i = 0; i < subset_face_count; i++) {
        int f = subset_faces[i];
        if (f >= 0 && f < nfaces) total += face_counts[f];
    }
    return total;
}

static int emit_subset_indices(Arena* arena, const SceneMesh* base,
                               const int* face_starts, const int* face_counts,
                               int nfaces, const int* subset_faces,
                               int subset_face_count, uint32_t** out_indices) {
    int nindices = count_subset_indices(subset_faces, subset_face_count,
                                        face_counts, nfaces);
    if (nindices <= 0) return 0;
    uint32_t* indices =
        (uint32_t*)arena_alloc(arena, (size_t)nindices * sizeof(uint32_t), 16);
    if (!indices) return 0;
    int out = 0;
    for (int i = 0; i < subset_face_count; i++) {
        int f = subset_faces[i];
        if (f < 0 || f >= nfaces || face_counts[f] <= 0 || face_starts[f] < 0)
            continue;
        memcpy(&indices[out], &base->indices[face_starts[f]],
               (size_t)face_counts[f] * sizeof(uint32_t));
        out += face_counts[f];
    }
    *out_indices = indices;
    return out;
}

/* Returns how many meshes now occupy scene->meshes[base_mesh_idx..].
 * 1 == not split (leave the base mesh untouched). */
static int split_mesh_by_geom_subsets(Arena* arena, Scene* scene,
                                      NanousdStage stage, NanousdPrim mesh_prim,
                                      int base_mesh_idx, int max_meshes,
                                      const int* fvc_data, int fvc_count,
                                      int fvi_count) {
    if (!arena || !scene || !mesh_prim || base_mesh_idx < 0 ||
        base_mesh_idx >= max_meshes)
        return 1;
    int nch = nanousd_nchildren(mesh_prim);
    if (nch <= 0) return 1;   /* fast path: no children, no subsets */
    SceneMesh base = scene->meshes[base_mesh_idx];
    int* face_starts = NULL; int* face_counts = NULL; int nfaces = 0;
    if (!build_mesh_face_index_ranges(arena, &base, fvc_data, fvc_count,
                                      fvi_count, &face_starts, &face_counts,
                                      &nfaces))
        return 1;
    GeomSubsetInfo* subsets =
        (GeomSubsetInfo*)arena_calloc(arena, (size_t)nch, sizeof(GeomSubsetInfo));
    unsigned char* covered =
        (unsigned char*)arena_calloc(arena, (size_t)nfaces, sizeof(unsigned char));
    if (!subsets || !covered) return 1;
    int nsubsets = 0;
    for (int c = 0; c < nch; c++) {
        NanousdPrim child = nanousd_child(mesh_prim, c);
        if (!child) continue;
        const char* tn = nanousd_typename(child);
        if (!tn || strcmp(tn, "GeomSubset") != 0) { nanousd_freeprim(child); continue; }
        int ok = 0;
        const char* et = nanousd_attrib_token(child, "elementType", &ok);
        if (ok && et && et[0] && strcmp(et, "face") != 0) {
            nanousd_freeprim(child); continue;
        }
        int sfc = 0;
        const int* faces = nanousd_arraydatai(child, "indices", &sfc);
        int* copied = NULL;
        if (!faces || sfc <= 0) {
            int len = nanousd_attribarraylen(child, "indices");
            if (len > 0) {
                copied = (int*)arena_alloc(arena, (size_t)len * sizeof(int), 16);
                if (copied) {
                    nanousd_attribarrayi(child, "indices", copied, len);
                    faces = copied; sfc = len;
                }
            }
        }
        if (!faces || sfc <= 0) { nanousd_freeprim(child); continue; }
        if (!copied) {
            copied = (int*)arena_alloc(arena, (size_t)sfc * sizeof(int), 16);
            if (!copied) { nanousd_freeprim(child); continue; }
            memcpy(copied, faces, (size_t)sfc * sizeof(int));
        }
        int ni = count_subset_indices(copied, sfc, face_counts, nfaces);
        if (ni <= 0) { nanousd_freeprim(child); continue; }
        GeomSubsetInfo* info = &subsets[nsubsets++];
        info->prim = child; info->face_indices = copied;
        info->nfaces = sfc; info->nindices = ni;
        info->material_index = scene->materials
            ? materials_find_binding((MaterialCollection*)scene->materials, stage, child)
            : -1;
        if (info->material_index < 0) info->material_index = base.material_index;
        for (int i = 0; i < sfc; i++) {
            int f = copied[i];
            if (f >= 0 && f < nfaces && face_counts[f] > 0) covered[f] = 1;
        }
    }
    if (nsubsets <= 0) return 1;
    int remainder = 0;
    for (int f = 0; f < nfaces; f++) if (!covered[f]) remainder += face_counts[f];
    int out_count = nsubsets + (remainder > 0 ? 1 : 0);
    if (base_mesh_idx + out_count > max_meshes) {
        for (int s = 0; s < nsubsets; s++) nanousd_freeprim(subsets[s].prim);
        return 1;
    }
    int emitted = 0;
    for (int s = 0; s < nsubsets; s++) {
        uint32_t* idx = NULL;
        int written = emit_subset_indices(arena, &base, face_starts, face_counts,
                                          nfaces, subsets[s].face_indices,
                                          subsets[s].nfaces, &idx);
        if (written <= 0 || !idx) continue;
        SceneMesh* out = &scene->meshes[base_mesh_idx + emitted];
        *out = base;
        out->indices = idx; out->nindices = written;
        out->material_index = subsets[s].material_index;
        out->prototype_idx = base_mesh_idx + emitted;
        out->vertex_offset = 0; out->index_offset = 0;
        emitted++;
    }
    if (remainder > 0) {
        uint32_t* rem =
            (uint32_t*)arena_alloc(arena, (size_t)remainder * sizeof(uint32_t), 16);
        if (rem) {
            int oi = 0;
            for (int f = 0; f < nfaces; f++) {
                if (covered[f] || face_counts[f] <= 0 || face_starts[f] < 0) continue;
                memcpy(&rem[oi], &base.indices[face_starts[f]],
                       (size_t)face_counts[f] * sizeof(uint32_t));
                oi += face_counts[f];
            }
            SceneMesh* out = &scene->meshes[base_mesh_idx + emitted];
            *out = base;
            out->indices = rem; out->nindices = oi;
            out->prototype_idx = base_mesh_idx + emitted;
            out->vertex_offset = 0; out->index_offset = 0;
            emitted++;
        }
    }
    for (int s = 0; s < nsubsets; s++) nanousd_freeprim(subsets[s].prim);
    return emitted > 0 ? emitted : 1;
}

Scene* scene_load_from_stage(void* stage_void, const char* stage_label)
{
    return scene_load_from_stage_filtered(stage_void, stage_label, NULL, 0);
}

/* Grow scene->pi_batches to hold >= `needed` entries (geometric). Returns 0 on
 * allocation failure. (Phase 1 compact PointInstancer model.) */
static int scene_pi_batches_reserve(Scene* scene, int* capacity, int needed)
{
    if (!scene || !capacity || needed <= *capacity) return 1;
    int new_cap = *capacity > 0 ? *capacity : 64;
    while (new_cap < needed) { int n = new_cap * 2; if (n < new_cap) return 0; new_cap = n; }
    if ((size_t)new_cap > ((size_t)-1) / sizeof(SceneInstanceBatch)) return 0;
    SceneInstanceBatch* grown = (SceneInstanceBatch*)realloc(
        scene->pi_batches, (size_t)new_cap * sizeof(SceneInstanceBatch));
    if (!grown) return 0;
    scene->pi_batches = grown;
    *capacity = new_cap;
    return 1;
}

/* Grow scene->pi_transforms by `add` and return a pointer to the first new
 * slot (caller fills it). Returns NULL on overflow/alloc failure. */
static SceneInstanceTransform* scene_pi_transforms_reserve(
        Scene* scene, uint64_t* capacity, uint64_t add)
{
    if (!scene || !capacity || add == 0) return NULL;
    uint64_t needed = scene->npi_transforms + add;
    if (needed < add) return NULL;
    if (needed > *capacity) {
        uint64_t new_cap = *capacity ? *capacity : 4096;
        while (new_cap < needed) { uint64_t n = new_cap * 2; if (n < new_cap) return NULL; new_cap = n; }
        if (new_cap > ((size_t)-1) / sizeof(SceneInstanceTransform)) return NULL;
        SceneInstanceTransform* grown = (SceneInstanceTransform*)realloc(
            scene->pi_transforms, (size_t)new_cap * sizeof(SceneInstanceTransform));
        if (!grown) return NULL;
        scene->pi_transforms = grown;
        *capacity = new_cap;
    }
    SceneInstanceTransform* slot = &scene->pi_transforms[scene->npi_transforms];
    scene->npi_transforms = needed;
    return slot;
}

static void scene_pack_affine12(const double m[16], SceneInstanceTransform* t)
{
    t->m[0] = (float)m[0];   t->m[1] = (float)m[1];   t->m[2] = (float)m[2];
    t->m[3] = (float)m[4];   t->m[4] = (float)m[5];   t->m[5] = (float)m[6];
    t->m[6] = (float)m[8];   t->m[7] = (float)m[9];   t->m[8] = (float)m[10];
    t->m[9] = (float)m[12];  t->m[10] = (float)m[13]; t->m[11] = (float)m[14];
}

static int scene_add_compact_instance_batch(Scene* scene,
                                            uint64_t* xform_cap,
                                            int* batch_cap,
                                            int prototype_mesh_idx,
                                            const SceneMesh* prototype_mesh,
                                            const double world_xform[16],
                                            int source_kind)
{
    if (!scene || !xform_cap || !batch_cap || prototype_mesh_idx < 0 ||
        !prototype_mesh || !world_xform) {
        return 0;
    }
    if (scene->npi_transforms > UINT32_MAX) return 0;
    if (!scene_pi_batches_reserve(scene, batch_cap, scene->npi_batches + 1))
        return 0;

    uint32_t base = (uint32_t)scene->npi_transforms;
    SceneInstanceTransform* t = scene_pi_transforms_reserve(scene, xform_cap, 1);
    if (!t) return 0;
    scene_pack_affine12(world_xform, t);

    SceneInstanceBatch* b = &scene->pi_batches[scene->npi_batches++];
    b->prototype_mesh_idx = prototype_mesh_idx;
    b->transform_offset = base;
    b->transform_count = 1;
    b->source_prim_idx = -1;
    b->material_or_binding_id = -1;
    b->source_kind = source_kind;

    float lmn[3], lmx[3], wmn[3], wmx[3];
    if (mesh_local_bounds(prototype_mesh, lmn, lmx)) {
        world_bounds_from_local(world_xform, lmn, lmx, wmn, wmx);
        scene_expand_bounds(scene, wmn, wmx);
    }
    return 1;
}

/* Nested-PointInstancer record. File scope so the asset-arc seeder can append
 * to the same list the composed walk fills; both are drained after the 2nd pass
 * (the drain re-gets each PI from the MAIN stage by path and decodes it). */
struct NestedPiRec { char path[1024]; double pi_to_inst[16]; };

/* ===================================================================
 * Path B: native-instance asset-arc flat seeding + replay catalog.
 *
 * The composed-child walk (nanousd_child on composed instance prims) is
 * O(N^2) at Moana scale -- per-prim composition cost grows with accumulated
 * stage load. Instead, read each unique prototype's geometry ONCE directly
 * from its referenced asset LAYER via nanousd_traverse_flat (a flat decode,
 * not a composed walk), cache it in a replay catalog, and replay
 * pointer-shared copies for sibling instances. Gated by
 * NUSD_FLAT_NATIVE_INSTANCE_TRAVERSAL (or NUSD_NO_CULL_ALL_GEOMETRY). Mirrors
 * nanousd-metal-renderer's native asset-arc seeding + replay catalog.
 * Requires nanousd "semantic traversal" APIs (traverse_flat, composition_arc,
 * instance_key).
 * =================================================================== */
typedef struct {
    char   relative_path[512];
    int    prototype_idx;
    double relative_xform[16];  /* proto-local: asset_child_world * target_inv */
} NbReplayEntry;
typedef struct {
    char   rel_path[512];   /* PI path relative to the instance root */
    double rel_xform[16];   /* proto-local PI xform: pi_world * proto_root_inv */
} NbPiEntry;
typedef struct {
    char           key[768];
    NbReplayEntry* entries;
    int            count, cap, built, aborted;
    NbPiEntry*     pi_entries;   /* nested PIs under the prototype (drained per instance) */
    int            pi_count, pi_cap;
    uint64_t*      mesh_path_set;
    int            mesh_path_count, mesh_path_cap;
    uint64_t*      xgen_archive_set;
    int            xgen_archive_count, xgen_archive_cap;
} NbReplayCatalog;
typedef struct {
    NbReplayCatalog* items;
    int              count, cap;
} NbReplayCache;
typedef struct {
    NanousdStage* stages;
    int           count, cap;
} NbAssetStages;

static int scene_env_flag(const char* name);
static int scene_env_flag_explicit_false(const char* name);
static int scene_no_cull_all_geometry_active(void);

static int nb_flat_native_replay_active(void) {
    if (scene_env_flag("NUSD_FLAT_NATIVE_INSTANCE_TRAVERSAL") ||
        scene_no_cull_all_geometry_active()) {
        return 1;
    }
    if (scene_env_flag("NUSD_LEGACY_GEOMETRY_DEFAULTS") ||
        scene_env_flag("NUSD_DISABLE_STORM_GEOMETRY_DEFAULTS") ||
        scene_env_flag("NUSD_DISABLE_FLAT_NATIVE_INSTANCE_TRAVERSAL") ||
        scene_env_flag_explicit_false("NUSD_FLAT_NATIVE_INSTANCE_TRAVERSAL")) {
        return 0;
    }
    return 1;
}

static int scene_env_flag(const char* name)
{
    const char* e = getenv(name);
    if (!e || !e[0]) return 0;
    if (e[0] == '0' || !strcmp(e, "false") || !strcmp(e, "off") ||
        !strcmp(e, "no")) {
        return 0;
    }
    return 1;
}

static int scene_env_flag_explicit_false(const char* name)
{
    const char* e = getenv(name);
    if (!e || !e[0]) return 0;
    return e[0] == '0' || !strcmp(e, "false") || !strcmp(e, "off") ||
           !strcmp(e, "no");
}

static int scene_no_cull_all_geometry_active(void)
{
    return scene_env_flag("NUSD_NO_CULL_ALL_GEOMETRY") ||
           scene_env_flag("NUSD_ALL_GEOMETRY_NO_CULL");
}

static int scene_compact_pi_batches_active(void)
{
    if (scene_env_flag("NUSD_RENDER_PI_BATCHES") ||
        scene_env_flag("NUSD_PI_COMPACT_ONLY") ||
        scene_env_flag("NUSD_RT_CULL") ||
        scene_no_cull_all_geometry_active()) {
        return 1;
    }
    if (scene_env_flag("NUSD_LEGACY_GEOMETRY_DEFAULTS") ||
        scene_env_flag("NUSD_DISABLE_STORM_GEOMETRY_DEFAULTS") ||
        scene_env_flag("NUSD_DISABLE_RENDER_PI_BATCHES") ||
        scene_env_flag_explicit_false("NUSD_RENDER_PI_BATCHES")) {
        return 0;
    }
    return 1;
}

/* Storm-style "fast & light" geometry defaults: native instances and
 * content-duplicated flat references are kept compact (prototype + 48-byte
 * transforms) instead of expanded into per-copy SceneMesh rows, and duplicate
 * geometry is skipped before decode. All three are ON by default; the shared
 * NUSD_LEGACY_GEOMETRY_DEFAULTS / NUSD_DISABLE_STORM_GEOMETRY_DEFAULTS escape
 * hatch (or the per-feature NUSD_DISABLE_* / explicit-false) restores the legacy
 * expand path for golden comparison. */
static int scene_storm_geometry_legacy(void)
{
    return scene_env_flag("NUSD_LEGACY_GEOMETRY_DEFAULTS") ||
           scene_env_flag("NUSD_DISABLE_STORM_GEOMETRY_DEFAULTS");
}
/* compact-native instancing: DEFAULT-ON. Extracts each native-instance prototype
 * ONCE and shares it across instances via 48-byte transforms (Storm-style), instead
 * of the legacy per-instance proxy expand-walk (the DSX 48 GB / >30 min blowup).
 * Non-Mesh prototypes (nested PI / implicit shapes) fall back to the legacy walk
 * (B1/B2), so it is geometry-lossless. Validated end-to-end on DSX (980,927 shared
 * placements, renders) and warehouse (no regression). Escape hatch:
 * NUSD_LEGACY_GEOMETRY_DEFAULTS / NUSD_DISABLE_COMPACT_NATIVE_INSTANCES.
 * Known gap: B4 per-instance visibility (invisibleIds) is not yet honored. */
static int scene_compact_native_instances_active(void)
{
    if (scene_env_flag("NUSD_COMPACT_NATIVE_INSTANCES")) return 1;
    if (scene_storm_geometry_legacy() ||
        scene_env_flag("NUSD_DISABLE_COMPACT_NATIVE_INSTANCES") ||
        scene_env_flag_explicit_false("NUSD_COMPACT_NATIVE_INSTANCES")) return 0;
    return 1;
}
static int scene_predecode_dedup_active(void)
{
    return scene_env_flag("NUSD_PREDECODE_DEDUP");
}
/* compact-flat-dedup: DEFAULT-ON. Content-hash keyed (geometry-safe) + the B6
 * per-instance material/UV/displayColor guard at the hit site (diverging
 * duplicates fall through to a full unique decode), so it is correct by default.
 * Escape hatch: NUSD_LEGACY_GEOMETRY_DEFAULTS / NUSD_DISABLE_COMPACT_FLAT_DEDUP.
 * (compact-native + pre-decode stay opt-in until their feature gaps land.) */
static int scene_compact_flat_dedup_active(void)
{
    if (scene_env_flag("NUSD_COMPACT_FLAT_DEDUP")) return 1;
    if (scene_storm_geometry_legacy() ||
        scene_env_flag("NUSD_DISABLE_COMPACT_FLAT_DEDUP") ||
        scene_env_flag_explicit_false("NUSD_COMPACT_FLAT_DEDUP")) return 0;
    return 1;
}

static int nb_chase_arcs_after_direct_active(void)
{
    if (scene_env_flag("NUSD_METAL_PARITY_ARC_SEED") ||
        scene_env_flag("NUSD_DISABLE_NATIVE_ARC_CHASE_AFTER_DIRECT") ||
        scene_env_flag_explicit_false("NUSD_NATIVE_ARC_CHASE_AFTER_DIRECT")) {
        return 0;
    }
    return 1;
}

static int scene_material_arc_path_like(const char* path)
{
    if (!path || !path[0]) return 0;
    return strstr(path, "/materials/") != NULL ||
           strstr(path, "\\materials\\") != NULL ||
           strstr(path, "/Looks") != NULL ||
           strstr(path, "/BaseMaterial") != NULL ||
           strstr(path, "/PtexBaseMaterial") != NULL ||
           strstr(path, "_materials.usd") != NULL ||
           strstr(path, "_materials.usda") != NULL ||
           strstr(path, "materials.usd") != NULL ||
           strstr(path, "materials.usda") != NULL ||
           strstr(path, "material.usd") != NULL ||
           strstr(path, "material.usda") != NULL;
}

static int nb_skip_arc_for_geometry_only(const char* layer_or_asset,
                                         const char* source_or_target)
{
    return scene_material_arc_path_like(layer_or_asset) ||
           scene_material_arc_path_like(source_or_target);
}

static char (*s_nb_toplevel_pi_keys)[512] = NULL;
static int s_nb_toplevel_pi_count = 0;
static int s_nb_toplevel_pi_cap = 0;

static void nb_toplevel_pi_paths_reset(void)
{
    free(s_nb_toplevel_pi_keys);
    s_nb_toplevel_pi_keys = NULL;
    s_nb_toplevel_pi_count = 0;
    s_nb_toplevel_pi_cap = 0;
}

static void nb_pi_skip_key(const char* path, char* out, size_t out_size)
{
    if (out_size) out[0] = '\0';
    if (!path || path[0] != '/' || !out || out_size == 0) return;
    const char* c0 = path + 1;
    const char* s0 = strchr(c0, '/');
    if (!s0) return;
    const char* c1 = s0 + 1;
    const char* s1 = strchr(c1, '/');
    int elen = s1 ? (int)(s1 - c1) : (int)strlen(c1);
    const char* last = strrchr(path, '/');
    last = last ? last + 1 : path;
    snprintf(out, out_size, "%.*s|%s", elen, c1, last);
}

static void nb_toplevel_pi_paths_add(const char* path)
{
    char key[512];
    nb_pi_skip_key(path, key, sizeof(key));
    if (!key[0]) return;
    for (int i = 0; i < s_nb_toplevel_pi_count; i++)
        if (!strcmp(s_nb_toplevel_pi_keys[i], key)) return;
    if (s_nb_toplevel_pi_count >= s_nb_toplevel_pi_cap) {
        int nc = s_nb_toplevel_pi_cap ? s_nb_toplevel_pi_cap * 2 : 64;
        char (*g)[512] = (char (*)[512])realloc(
            s_nb_toplevel_pi_keys, (size_t)nc * sizeof(*s_nb_toplevel_pi_keys));
        if (!g) return;
        s_nb_toplevel_pi_keys = g;
        s_nb_toplevel_pi_cap = nc;
    }
    snprintf(s_nb_toplevel_pi_keys[s_nb_toplevel_pi_count++], 512, "%s", key);
}

static int nb_toplevel_pi_paths_contains(const char* path)
{
    char key[512];
    nb_pi_skip_key(path, key, sizeof(key));
    if (!key[0]) return 0;
    for (int i = 0; i < s_nb_toplevel_pi_count; i++)
        if (!strcmp(s_nb_toplevel_pi_keys[i], key)) return 1;
    return 0;
}

static int nb_path_has_prefix(const char* path, const char* prefix) {
    if (!path || !prefix || !prefix[0]) return 0;
    size_t n = strlen(prefix);
    if (strncmp(path, prefix, n) != 0) return 0;
    return path[n] == '\0' || path[n] == '/';
}

typedef struct {
    char (*paths)[1024];
    int count;
    int cap;
    int used_child_fallback;
} PiProtoTargets;

static void pi_proto_targets_free(PiProtoTargets* t)
{
    if (!t) return;
    free(t->paths);
    t->paths = NULL;
    t->count = 0;
    t->cap = 0;
    t->used_child_fallback = 0;
}

static int pi_proto_targets_add(PiProtoTargets* t, const char* path)
{
    if (!t || !path || path[0] != '/') return 0;
    for (int i = 0; i < t->count; i++)
        if (!strcmp(t->paths[i], path)) return 1;
    if (t->count >= t->cap) {
        int nc = t->cap ? t->cap * 2 : 16;
        char (*g)[1024] = (char (*)[1024])realloc(
            t->paths, (size_t)nc * sizeof(*t->paths));
        if (!g) return 0;
        t->paths = g;
        t->cap = nc;
    }
    snprintf(t->paths[t->count++], sizeof(t->paths[0]), "%s", path);
    return 1;
}

static void pi_proto_targets_add_child(PiProtoTargets* t, NanousdPrim child)
{
    if (!t || !child || !nanousd_isactive(child) || nanousd_isabstract(child))
        return;
    const char* tn = nanousd_typename(child);
    if (tn && (!strcmp(tn, "Shader") || !strcmp(tn, "Material") ||
               !strcmp(tn, "PointInstancer"))) {
        return;
    }
    const char* cp = nanousd_path(child);
    char cbuf[1024];
    if (cp && cp[0]) {
        snprintf(cbuf, sizeof(cbuf), "%s", cp);
        pi_proto_targets_add(t, cbuf);
    }
}

static int pi_proto_targets_collect(NanousdPrim pi, PiProtoTargets* out)
{
    if (!pi || !out) return 0;
    memset(out, 0, sizeof(*out));

    int n_rel = nanousd_nreltargets(pi, "prototypes");
    for (int p = 0; p < n_rel; p++) {
        const char* target = nanousd_reltarget(pi, "prototypes", p);
        pi_proto_targets_add(out, target);
    }
    if (out->count > 0) {
        if (getenv("NUSD_PI_PROTO_DIAG")) {
            const char* pp = nanousd_path(pi);
            fprintf(stderr,
                    "scene_load: PointInstancer '%s': %d prototype rel target(s)\n",
                    pp ? pp : "?", out->count);
        }
        return out->count;
    }

    int n_child = nanousd_nchildren(pi);
    for (int c = 0; c < n_child; c++) {
        NanousdPrim child = nanousd_child(pi, c);
        if (!child) continue;
        const char* name = nanousd_name(child);
        const char* tn = nanousd_typename(child);
        int expanded_scope = 0;
        if (name && !strcmp(name, "Prototypes")) {
            int npc = nanousd_nchildren(child);
            for (int pc = 0; pc < npc; pc++) {
                NanousdPrim proto_child = nanousd_child(child, pc);
                if (proto_child) {
                    pi_proto_targets_add_child(out, proto_child);
                    nanousd_freeprim(proto_child);
                }
            }
            expanded_scope = 1;
        } else if (tn && (!strcmp(tn, "Scope") || !strcmp(tn, "Xform")) &&
                   name && strstr(name, "Prototypes")) {
            int npc = nanousd_nchildren(child);
            for (int pc = 0; pc < npc; pc++) {
                NanousdPrim proto_child = nanousd_child(child, pc);
                if (proto_child) {
                    pi_proto_targets_add_child(out, proto_child);
                    nanousd_freeprim(proto_child);
                }
            }
            expanded_scope = 1;
        }
        if (!expanded_scope)
            pi_proto_targets_add_child(out, child);
        nanousd_freeprim(child);
    }
    if (out->count > 0) {
        out->used_child_fallback = 1;
        if (getenv("NUSD_PI_PROTO_DIAG")) {
            const char* pp = nanousd_path(pi);
            fprintf(stderr,
                    "scene_load: PointInstancer '%s': prototypes rel missing; "
                    "using %d child prototype target(s)\n",
                    pp ? pp : "?", out->count);
        }
    }
    return out->count;
}

static const char* pi_proto_target_composed_path(const char* proto_path_raw,
                                                 const char* pi_full_path,
                                                 char* proto_path_buf,
                                                 size_t proto_path_buf_size)
{
    if (!proto_path_raw) return NULL;
    if (!proto_path_buf || proto_path_buf_size == 0)
        return proto_path_raw;
    proto_path_buf[0] = '\0';
    const char* proto_path = proto_path_raw;
    if (proto_path_raw[0] == '/' && pi_full_path) {
        const char* slash2 = strchr(proto_path_raw + 1, '/');
        size_t first_seg = slash2 ? (size_t)(slash2 - proto_path_raw)
                                  : strlen(proto_path_raw);
        if (first_seg > 1) {
            const char* match = strstr(pi_full_path, proto_path_raw);
            if (!match) {
                char first_seg_buf[256];
                if (first_seg < sizeof(first_seg_buf)) {
                    memcpy(first_seg_buf, proto_path_raw, first_seg);
                    first_seg_buf[first_seg] = '\0';
                    const char* m = strstr(pi_full_path, first_seg_buf);
                    if (m) {
                        char next = m[first_seg];
                        if (next == '\0' || next == '/') {
                            size_t prefix_len = (size_t)(m - pi_full_path);
                            int n = snprintf(proto_path_buf,
                                             proto_path_buf_size,
                                             "%.*s%s",
                                             (int)prefix_len, pi_full_path,
                                             proto_path_raw);
                            if (n > 0 && (size_t)n < proto_path_buf_size) {
                                proto_path = proto_path_buf;
                            }
                        }
                    }
                }
            }
        }
    }
    return proto_path;
}

static int int64_cmp_for_qsort(const void* a, const void* b)
{
    int64_t av = *(const int64_t*)a;
    int64_t bv = *(const int64_t*)b;
    return (av > bv) - (av < bv);
}

static int int64_array_contains_sorted(const int64_t* vals, int n, int64_t v)
{
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int64_t mv = vals[mid];
        if (mv == v) return 1;
        if (mv < v) lo = mid + 1;
        else hi = mid - 1;
    }
    return 0;
}

static unsigned char* pi_visible_mask_from_invisible_ids(NanousdPrim pi,
                                                         int n_instances)
{
    if (!pi || n_instances <= 0) return NULL;
    int n_hidden = nanousd_attribarraylen(pi, "invisibleIds");
    if (n_hidden <= 0) return NULL;

    int64_t* hidden = (int64_t*)malloc((size_t)n_hidden * sizeof(int64_t));
    if (!hidden) return NULL;
    int got_hidden = nanousd_attribarrayi64(pi, "invisibleIds", hidden, n_hidden);
    if (got_hidden <= 0) {
        free(hidden);
        return NULL;
    }
    if (got_hidden < n_hidden) n_hidden = got_hidden;
    qsort(hidden, (size_t)n_hidden, sizeof(int64_t), int64_cmp_for_qsort);

    unsigned char* visible = (unsigned char*)malloc((size_t)n_instances);
    if (!visible) {
        free(hidden);
        return NULL;
    }
    memset(visible, 1, (size_t)n_instances);

    int n_ids = nanousd_attribarraylen(pi, "ids");
    int64_t* ids = NULL;
    if (n_ids > 0) {
        ids = (int64_t*)malloc((size_t)n_ids * sizeof(int64_t));
        if (ids) {
            int got_ids = nanousd_attribarrayi64(pi, "ids", ids, n_ids);
            if (got_ids <= 0) {
                free(ids);
                ids = NULL;
                n_ids = 0;
            } else if (got_ids < n_ids) {
                n_ids = got_ids;
            }
        } else {
            n_ids = 0;
        }
    }

    int hidden_count = 0;
    for (int i = 0; i < n_instances; i++) {
        int64_t id = (ids && i < n_ids) ? ids[i] : (int64_t)i;
        if (int64_array_contains_sorted(hidden, n_hidden, id)) {
            visible[i] = 0;
            hidden_count++;
        }
    }
    if (hidden_count == 0) {
        free(visible);
        visible = NULL;
    } else if (getenv("NUSD_PI_PROTO_DIAG")) {
        const char* pp = nanousd_path(pi);
        fprintf(stderr,
                "scene_load: PointInstancer '%s': skipped %d invisibleIds "
                "instance(s)\n",
                pp ? pp : "?", hidden_count);
    }
    free(ids);
    free(hidden);
    return visible;
}

static void nb_instance_key(NanousdPrim inst_prim, NanousdPrim proto_root,
                            char* out, size_t out_size) {
    out[0] = '\0';
    int kl = nanousd_instance_key(inst_prim, out, out_size);
    const char* pp = proto_root ? nanousd_path(proto_root) : NULL;
    if (kl > 0 && out[0]) {
        if (pp && pp[0]) {
            size_t u = strlen(out);
            if (u < out_size) snprintf(out + u, out_size - u, "|proto=%s", pp);
        }
        return;
    }
    snprintf(out, out_size, "%s", (pp && pp[0]) ? pp : "<no-prototype>");
}

static NbReplayCatalog* nb_cache_find_or_add(NbReplayCache* c, const char* key) {
    if (!c || !key || !key[0]) return NULL;
    for (int i = 0; i < c->count; i++)
        if (strcmp(c->items[i].key, key) == 0) return &c->items[i];
    if (c->count >= c->cap) {
        int nc = c->cap ? c->cap * 2 : 16;
        NbReplayCatalog* g = (NbReplayCatalog*)realloc(c->items, (size_t)nc * sizeof(*g));
        if (!g) return NULL;
        memset(g + c->cap, 0, (size_t)(nc - c->cap) * sizeof(*g));
        c->items = g; c->cap = nc;
    }
    NbReplayCatalog* cat = &c->items[c->count++];
    memset(cat, 0, sizeof(*cat));
    snprintf(cat->key, sizeof(cat->key), "%s", key);
    return cat;
}

static void nb_cache_free(NbReplayCache* c) {
    if (!c) return;
    for (int i = 0; i < c->count; i++) {
        free(c->items[i].entries);
        free(c->items[i].pi_entries);
        free(c->items[i].mesh_path_set);
        free(c->items[i].xgen_archive_set);
    }
    free(c->items); c->items = NULL; c->count = c->cap = 0;
}

static uint64_t nb_rel_hash(const char* rel)
{
    while (rel && *rel == '/') rel++;
    uint64_t h = proto_fnv1a64(rel ? rel : "");
    return h ? h : 0x9e3779b97f4a7c15ull;
}

static int nb_path_set_contains(const uint64_t* slots, int cap, uint64_t h)
{
    if (!slots || cap <= 0) return 0;
    uint32_t mask = (uint32_t)cap - 1u;
    uint32_t i = (uint32_t)h & mask;
    while (slots[i]) {
        if (slots[i] == h) return 1;
        i = (i + 1u) & mask;
    }
    return 0;
}

static int nb_path_set_insert(uint64_t** slots_io, int* count_io, int* cap_io,
                              uint64_t h)
{
    if (!slots_io || !count_io || !cap_io || h == 0) return 0;
    /* Normalize the {slots, cap} pair so callers can't pass an inconsistent
     * NULL-with-nonzero-cap (or vice versa) and trigger a NULL deref in the
     * rehash loop or the final insertion. */
    if (!*slots_io) { *cap_io = 0; *count_io = 0; }
    if (nb_path_set_contains(*slots_io, *cap_io, h)) return 1;
    if (*count_io * 10 >= *cap_io * 7) {
        int old_cap = *cap_io;
        uint64_t* old = *slots_io;
        int new_cap = old_cap ? old_cap * 2 : 1024;
        uint64_t* slots = (uint64_t*)calloc((size_t)new_cap, sizeof(uint64_t));
        if (!slots) return 0;
        uint32_t mask = (uint32_t)new_cap - 1u;
        for (int i = 0; i < old_cap; i++) {
            uint64_t oh = old[i];
            if (!oh) continue;
            uint32_t j = (uint32_t)oh & mask;
            while (slots[j]) j = (j + 1u) & mask;
            slots[j] = oh;
        }
        free(old);
        *slots_io = slots;
        *cap_io = new_cap;
    }
    uint32_t mask = (uint32_t)(*cap_io) - 1u;
    uint32_t i = (uint32_t)h & mask;
    while ((*slots_io)[i]) i = (i + 1u) & mask;
    (*slots_io)[i] = h;
    (*count_io)++;
    return 1;
}

static int nb_catalog_has_mesh_path(const NbReplayCatalog* cat, const char* rel)
{
    if (!cat || !rel) return 0;
    return nb_path_set_contains(cat->mesh_path_set, cat->mesh_path_cap,
                                nb_rel_hash(rel));
}

static void nb_catalog_add_pi(NbReplayCatalog* cat, const char* rel, const double rx[16]) {
    if (!cat) return;
    while (*rel == '/') rel++;
    if (cat->pi_count >= cat->pi_cap) {
        int nc = cat->pi_cap ? cat->pi_cap * 2 : 16;
        NbPiEntry* g = (NbPiEntry*)realloc(cat->pi_entries, (size_t)nc * sizeof(*g));
        if (!g) return;
        cat->pi_entries = g; cat->pi_cap = nc;
    }
    NbPiEntry* e = &cat->pi_entries[cat->pi_count++];
    snprintf(e->rel_path, sizeof(e->rel_path), "%s", rel);
    memcpy(e->rel_xform, rx, sizeof(double) * 16);
}

static void nb_catalog_add(NbReplayCatalog* cat, const char* rel, int proto_idx, const double rx[16]) {
    if (!cat || proto_idx < 0) return;
    while (*rel == '/') rel++;
    uint64_t rel_hash = nb_rel_hash(rel);
    if (nb_path_set_contains(cat->mesh_path_set, cat->mesh_path_cap, rel_hash))
        return;
    if (!nb_path_set_insert(&cat->mesh_path_set, &cat->mesh_path_count,
                            &cat->mesh_path_cap, rel_hash))
        return;
    if (cat->count >= cat->cap) {
        int nc = cat->cap ? cat->cap * 2 : 64;
        NbReplayEntry* g = (NbReplayEntry*)realloc(cat->entries, (size_t)nc * sizeof(*g));
        if (!g) return;
        cat->entries = g; cat->cap = nc;
    }
    NbReplayEntry* e = &cat->entries[cat->count++];
    snprintf(e->relative_path, sizeof(e->relative_path), "%s", rel);
    e->prototype_idx = proto_idx;
    memcpy(e->relative_xform, rx, sizeof(double) * 16);
}

static int nb_stages_add(NbAssetStages* s, NanousdStage st) {
    if (s->count >= s->cap) {
        int nc = s->cap ? s->cap * 2 : 8;
        NanousdStage* g = (NanousdStage*)realloc(s->stages, (size_t)nc * sizeof(*g));
        if (!g) return 0;
        s->stages = g; s->cap = nc;
    }
    s->stages[s->count++] = st;
    return 1;
}
static void nb_stages_close(NbAssetStages* s) {
    for (int i = 0; i < s->count; i++) if (s->stages[i]) nanousd_close(s->stages[i]);
    free(s->stages); s->stages = NULL; s->count = s->cap = 0;
}

static int nb_apply_instance_variants_to_asset_target(NanousdPrim inst_prim,
                                                      NanousdStage asset_stage,
                                                      const char* target_path)
{
    if (!inst_prim || !asset_stage) return 0;
    NanousdPrim target = (target_path && target_path[0])
        ? nanousd_primpath(asset_stage, target_path) : NULL;
    if (!target) target = nanousd_defaultprim(asset_stage);
    if (!target) return 0;

    int changed = 0;
    int nsets = nanousd_nvariantsets(inst_prim);
    for (int i = 0; i < nsets; i++) {
        const char* set_name = nanousd_variantsetname(inst_prim, i);
        if (!set_name || !set_name[0]) continue;
        const char* selected = nanousd_variantselection(inst_prim, set_name);
        if (!selected || !selected[0]) continue;
        if (!nanousd_hasvariantset(target, set_name)) continue;
        if (nanousd_setvariantselection(target, set_name, selected, 0))
            changed = 1;
    }
    nanousd_freeprim(target);
    return changed;
}

/* Read minimal geometry (positions, triangulated indices, smooth normals,
 * displayColor) from an asset-stage Mesh prim. Returns 1 on success. */
static int nb_load_asset_mesh(NanousdPrim p, Arena* arena, SceneMesh* mesh) {
    int pc = 0;
    const float* pts = nanousd_arraydataf(p, "points", &pc);
    if (pts && pc > 0) {
        mesh->nvertices = pc / 3;
        mesh->positions = (float*)arena_alloc(arena, (size_t)pc * sizeof(float), 16);
        if (mesh->positions) memcpy(mesh->positions, pts, (size_t)pc * sizeof(float));
    } else {
        int len = nanousd_attribarraylen(p, "points");
        if (len > 0) {
            mesh->nvertices = len / 3;
            mesh->positions = (float*)arena_alloc(arena, (size_t)len * sizeof(float), 16);
            if (mesh->positions) nanousd_attribarrayf(p, "points", mesh->positions, len);
        }
    }
    if (mesh->nvertices == 0 || !mesh->positions) return 0;

    int fvi_count = 0;
    const int* fvi = nanousd_arraydatai(p, "faceVertexIndices", &fvi_count);
    if (!fvi || fvi_count <= 0) {
        int len = nanousd_attribarraylen(p, "faceVertexIndices");
        if (len <= 0) return 0;
        int* fb = (int*)arena_alloc(arena, (size_t)len * sizeof(int), 16);
        if (!fb) return 0;
        nanousd_attribarrayi(p, "faceVertexIndices", fb, len);
        fvi = fb; fvi_count = len;
    }
    int fvc_count = 0;
    const int* fvc = nanousd_arraydatai(p, "faceVertexCounts", &fvc_count);
    if (!fvc || fvc_count <= 0) {
        int len = nanousd_attribarraylen(p, "faceVertexCounts");
        if (len > 0) {
            int* fb = (int*)arena_alloc(arena, (size_t)len * sizeof(int), 16);
            if (fb) { nanousd_attribarrayi(p, "faceVertexCounts", fb, len); fvc = fb; fvc_count = len; }
        }
    }
    if (fvc && fvc_count > 0) {
        int tri = count_triangulated_indices(fvc, fvc_count, fvi_count);
        mesh->nindices = tri;
        mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)tri * sizeof(uint32_t), 16);
        if (!mesh->indices) return 0;
        triangulate_faces(fvi, fvi_count, fvc, fvc_count, mesh->indices, NULL);
    } else {
        mesh->nindices = fvi_count;
        mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)fvi_count * sizeof(uint32_t), 16);
        if (!mesh->indices) return 0;
        for (int j = 0; j < fvi_count; j++) mesh->indices[j] = (uint32_t)fvi[j];
    }
    if (mesh->nindices <= 0) return 0;

    int nc = 0;
    const float* nd = nanousd_arraydataf(p, "normals", &nc);
    if (nd && nc / 3 == mesh->nvertices) {
        mesh->normals = (float*)arena_alloc(arena, (size_t)nc * sizeof(float), 16);
        if (mesh->normals) memcpy(mesh->normals, nd, (size_t)nc * sizeof(float));
    }
    if (!mesh->normals)
        mesh->normals = compute_smooth_normals(arena, mesh->positions, mesh->nvertices,
                                               mesh->indices, mesh->nindices);

    mesh->has_display_color = 0;
    mesh->display_color[0] = mesh->display_color[1] = mesh->display_color[2] = 0.5f;
    {
        int dc = 0;
        const float* dcd = nanousd_arraydataf(p, "primvars:displayColor", &dc);
        if (dcd && dc >= 3) {
            mesh->display_color[0] = dcd[0];
            mesh->display_color[1] = dcd[1];
            mesh->display_color[2] = dcd[2];
            mesh->has_display_color = 1;
        }
    }
    mesh->material_index = -1;
    return 1;
}

static void nb_identity_m4d(double m[16])
{
    memset(m, 0, sizeof(double) * 16);
    m[0] = m[5] = m[10] = m[15] = 1.0;
}

static int nb_join_path(const char* root, const char* rel,
                        char* out, size_t out_size)
{
    if (!root || !root[0] || !out || out_size == 0) return 0;
    while (rel && *rel == '/') rel++;
    if (rel && rel[0])
        snprintf(out, out_size, "%s/%s", root, rel);
    else
        snprintf(out, out_size, "%s", root);
    return 1;
}

static const char* nb_rel_under(const char* path, const char* root)
{
    if (!path || !root) return NULL;
    size_t n = strlen(root);
    if (strncmp(path, root, n) != 0) return NULL;
    if (path[n] != '\0' && path[n] != '/') return NULL;
    const char* rel = path + n;
    while (*rel == '/') rel++;
    return rel;
}

static int nb_parse_asset_arc_item(const char* item,
                                   char* asset, size_t asset_size,
                                   char* target, size_t target_size)
{
    if (asset_size) asset[0] = '\0';
    if (target_size) target[0] = '\0';
    if (!item || !asset || asset_size == 0 || !target || target_size == 0)
        return 0;
    const char* a0 = strchr(item, '@');
    if (!a0) return 0;
    const char* a1 = strchr(a0 + 1, '@');
    if (!a1 || a1 <= a0 + 1) return 0;
    size_t an = (size_t)(a1 - (a0 + 1));
    if (an >= asset_size) an = asset_size - 1;
    memcpy(asset, a0 + 1, an);
    asset[an] = '\0';

    const char* t0 = strchr(a1 + 1, '<');
    const char* t1 = t0 ? strchr(t0 + 1, '>') : NULL;
    if (t0 && t1 && t1 > t0 + 1) {
        size_t tn = (size_t)(t1 - (t0 + 1));
        if (tn >= target_size) tn = target_size - 1;
        memcpy(target, t0 + 1, tn);
        target[tn] = '\0';
    }
    return asset[0] != '\0';
}

static int nb_seed_asset_stage_recursive(Scene* scene, NanousdStage main_stage,
                                         NanousdPrim inst_prim,
                                         NanousdStage asset_stage,
                                         const char* target_path,
                                         const char* catalog_root_path,
                                         const double inst_world[16],
                                         const char* concrete_root_path,
                                         const double concrete_root_rel[16],
                                         NbReplayCatalog* cat, Arena* arena,
                                         int* mesh_idx, int max_meshes,
                                         uint64_t* total_verts,
                                         uint64_t* total_indices,
                                         struct NestedPiRec** nested,
                                         int* nested_n, int* nested_cap,
                                         uint64_t* pi_xform_cap,
                                         int* pi_batch_cap,
                                         int emit_native_batches,
                                         int depth)
{
    if (!scene || !asset_stage || !catalog_root_path || !inst_world ||
        !concrete_root_path || !concrete_root_rel || !cat || !arena ||
        !mesh_idx || depth > 5) {
        return 0;
    }

    NanousdPrim target = (target_path && target_path[0])
        ? nanousd_primpath(asset_stage, target_path) : NULL;
    if (!target) target = nanousd_defaultprim(asset_stage);
    if (!target) return 0;

    const char* tpr = nanousd_path(target);
    char tpath[1024];
    tpath[0] = '\0';
    if (tpr) snprintf(tpath, sizeof(tpath), "%s", tpr);
    double tworld[16], tinv[16];
    compute_world_xform(target, tworld);
    int have_tinv = invert_affine_m4d_rowvec(tworld, tinv);
    nanousd_freeprim(target);
    if (!have_tinv || !tpath[0]) return 0;
    size_t tlen = strlen(tpath);

    int nflat = nanousd_traverse_flat(asset_stage, NULL, 0);
    if (nflat <= 0) return 0;
    NanousdFlatPrim* flat = (NanousdFlatPrim*)calloc((size_t)nflat, sizeof(NanousdFlatPrim));
    if (!flat) return 0;
    int total = nanousd_traverse_flat(asset_stage, flat, nflat);
    if (total > 0 && total < nflat) nflat = total;

    char pi_pref[64][512];
    int npi_pref = 0;
    int emitted = 0;
    int direct_meshes = 0;
    int target_seen_in_flat = 0;
    for (int i = 0; i < nflat && *mesh_idx < max_meshes; i++) {
        const char* fp = flat[i].path;
        const char* tn = flat[i].type_name;
        if (!fp || !tn || !nb_path_has_prefix(fp, tpath)) continue;
        if (!strcmp(fp, tpath)) target_seen_in_flat = 1;

        const char* rel = fp + tlen;
        while (*rel == '/') rel++;
        char concrete_path[1024];
        if (!nb_join_path(concrete_root_path, rel, concrete_path, sizeof(concrete_path)))
            continue;

        NanousdPrim ap = nanousd_primpath(asset_stage, fp);
        int under_pi = 0;
        for (int z = 0; z < npi_pref; z++) {
            if (nb_path_has_prefix(fp, pi_pref[z])) { under_pi = 1; break; }
        }
        if (!under_pi && !emit_native_batches &&
            scene_compact_pi_batches_active() &&
            strstr(concrete_path, "/instancer/")) {
            under_pi = 1;
        }

        if (strcmp(tn, "PointInstancer") == 0) {
            double pw[16], rel_stage[16], rel_pi[16], pti[16];
            if (!ap) ap = nanousd_primpath(asset_stage, fp);
            if (!ap) continue;
            compute_world_xform(ap, pw);
            nanousd_mul_m4d(pw, tinv, rel_stage);
            nanousd_mul_m4d(rel_stage, concrete_root_rel, rel_pi);
            nanousd_mul_m4d(rel_pi, inst_world, pti);
            int top_level_owned = scene_compact_pi_batches_active() &&
                                  nb_toplevel_pi_paths_contains(concrete_path);
            NanousdPrim pim = top_level_owned ? NULL :
                (main_stage ? nanousd_primpath(main_stage, concrete_path) : NULL);
            if (top_level_owned) {
                emitted++;
            } else if (pim && nested && nested_n && nested_cap &&
                       nanousd_attribarraylen(pim, "protoIndices") > 0) {
                int already_nested = 0;
                for (int dn = 0; dn < *nested_n; dn++) {
                    if (!strcmp((*nested)[dn].path, concrete_path)) {
                        already_nested = 1;
                        break;
                    }
                }
                if (!already_nested) {
                    if (*nested_n >= *nested_cap) {
                        int ncp = *nested_cap ? *nested_cap * 2 : 16;
                        struct NestedPiRec* g = (struct NestedPiRec*)realloc(
                            *nested, (size_t)ncp * sizeof(**nested));
                        if (g) { *nested = g; *nested_cap = ncp; }
                    }
                    if (*nested && *nested_n < *nested_cap) {
                        snprintf((*nested)[*nested_n].path, sizeof((*nested)[0].path),
                                 "%s", concrete_path);
                        memcpy((*nested)[*nested_n].pi_to_inst, pti, sizeof(pti));
                        (*nested_n)++;
                    }
                }
                const char* cat_rel = nb_rel_under(concrete_path, catalog_root_path);
                if (cat_rel) nb_catalog_add_pi(cat, cat_rel, rel_pi);
                emitted++;
            }
            if (pim) nanousd_freeprim(pim);
            if (npi_pref < 64) {
                snprintf(pi_pref[npi_pref], sizeof(pi_pref[0]), "%s", fp);
                npi_pref++;
            }
            if (scene_compact_pi_batches_active() && depth < 5) {
                PiProtoTargets proto_targets;
                pi_proto_targets_collect(ap, &proto_targets);
                if (getenv("NUSD_PI_PROTO_DIAG")) {
                    int npi = nanousd_attribarraylen(ap, "protoIndices");
                    fprintf(stderr,
                            "scene_load: NB seed PI depth=%d asset='%s' "
                            "concrete='%s' instances=%d protos=%d\n",
                            depth, fp, concrete_path, npi,
                            proto_targets.count);
                }
                for (int pt = 0; pt < proto_targets.count && *mesh_idx < max_meshes; pt++) {
                    const char* proto_path = proto_targets.paths[pt];
                    if (!proto_path || proto_path[0] != '/') continue;
                    if (!nb_path_has_prefix(proto_path, tpath)) continue;

                    int has_flat_proto_mesh = 0;
                    for (int q = 0; q < nflat; q++) {
                        const char* qpath = flat[q].path;
                        const char* qtype = flat[q].type_name;
                        if (!qpath || !qtype || strcmp(qtype, "Mesh") != 0)
                            continue;
                        if (nb_path_has_prefix(qpath, proto_path)) {
                            has_flat_proto_mesh = 1;
                            break;
                        }
                    }
                    if (has_flat_proto_mesh) continue;

                    const char* prel = proto_path + tlen;
                    while (*prel == '/') prel++;
                    char concrete_proto_path[1024];
                    if (!nb_join_path(concrete_root_path, prel,
                                      concrete_proto_path,
                                      sizeof(concrete_proto_path))) {
                        continue;
                    }

                    NanousdPrim proto_prim = nanousd_primpath(asset_stage, proto_path);
                    if (!proto_prim) continue;
                    double proto_world[16], proto_to_target[16], proto_root_rel[16];
                    compute_world_xform(proto_prim, proto_world);
                    nanousd_freeprim(proto_prim);
                    nanousd_mul_m4d(proto_world, tinv, proto_to_target);
                    nanousd_mul_m4d(proto_to_target, concrete_root_rel, proto_root_rel);

                    int before_proto_mesh = *mesh_idx;
                    int proto_emitted = nb_seed_asset_stage_recursive(
                        scene, main_stage, inst_prim, asset_stage, proto_path,
                        catalog_root_path, inst_world, concrete_proto_path,
                        proto_root_rel, cat, arena, mesh_idx, max_meshes,
                        total_verts, total_indices, nested, nested_n,
                        nested_cap, pi_xform_cap, pi_batch_cap, 0,
                        depth + 1);
                    emitted += proto_emitted;
                    if (*mesh_idx == before_proto_mesh) {
                        const char* anchor = nanousd_stage_get_root_layer_path(asset_stage);
                        const char* base = strrchr(proto_path, '/');
                        if (anchor && strstr(anchor, "/xgenInstances/") &&
                            base && base[1] &&
                            strstr(concrete_proto_path, "/isMountainB/")) {
                            char rel_asset[512], resolved[2048];
                            snprintf(rel_asset, sizeof(rel_asset),
                                     "archives/%s.usd", base + 1);
                            if (nanousd_resolve_asset_path(anchor, rel_asset,
                                                           resolved,
                                                           sizeof(resolved))) {
                                uint64_t archive_hash = proto_fnv1a64(resolved);
                                if (nb_path_set_contains(cat->xgen_archive_set,
                                                         cat->xgen_archive_cap,
                                                         archive_hash)) {
                                    continue;
                                }
                                if (!nb_path_set_insert(&cat->xgen_archive_set,
                                                        &cat->xgen_archive_count,
                                                        &cat->xgen_archive_cap,
                                                        archive_hash)) {
                                    continue;
                                }
                                NanousdStage arch = nanousd_open(resolved);
                                if (arch && nanousd_isvalid(arch)) {
                                    if (getenv("NUSD_NB_ARC_DIAG"))
                                        fprintf(stderr,
                                                "  NB xgen archive fallback "
                                                "depth=%d proto='%s' asset='%s'\n",
                                                depth, proto_path, resolved);
                                    char arch_target[256];
                                    arch_target[0] = '\0';
                                    if (concrete_proto_path[0] == '/') {
                                        const char* slash2 = strchr(concrete_proto_path + 1, '/');
                                        size_t n = slash2 ? (size_t)(slash2 - concrete_proto_path)
                                                          : strlen(concrete_proto_path);
                                        if (n > 1 && n < sizeof(arch_target)) {
                                            memcpy(arch_target, concrete_proto_path, n);
                                            arch_target[n] = '\0';
                                        }
                                    }
                                    emitted += nb_seed_asset_stage_recursive(
                                        scene, main_stage, inst_prim, arch,
                                        arch_target[0] ? arch_target : NULL,
                                        "/__NUSD_PI_PROTO_NO_CATALOG__",
                                        inst_world, concrete_proto_path,
                                        proto_root_rel, cat, arena, mesh_idx,
                                        max_meshes, total_verts, total_indices,
                                        nested, nested_n, nested_cap,
                                        pi_xform_cap, pi_batch_cap, 0,
                                        depth + 1);
                                }
                                if (arch) nanousd_close(arch);
                            }
                        }
                    }
                }
                pi_proto_targets_free(&proto_targets);
            }
            if (ap) nanousd_freeprim(ap);
            continue;
        }

        if (strcmp(tn, "Mesh") != 0) {
            if (ap) nanousd_freeprim(ap);
            continue;
        }

        int no_catalog_proto_seed =
            catalog_root_path &&
            !strcmp(catalog_root_path, "/__NUSD_PI_PROTO_NO_CATALOG__");
        int load_pi_proto_mesh = under_pi && scene_compact_pi_batches_active() &&
            (no_catalog_proto_seed || strstr(concrete_path, "/isMountainB/"));
        if (under_pi && !load_pi_proto_mesh) {
            if (ap) nanousd_freeprim(ap);
            continue;
        }
        const char* cat_rel = nb_rel_under(concrete_path, catalog_root_path);
        if (!under_pi && cat_rel && nb_catalog_has_mesh_path(cat, cat_rel)) {
            if (ap) nanousd_freeprim(ap);
            continue;
        }
        if (!ap) ap = nanousd_primpath(asset_stage, fp);
        if (!ap) continue;
        if (prim_hidden_from_render_slow(ap)) {
            nanousd_freeprim(ap);
            continue;
        }
        SceneMesh* m = &scene->meshes[*mesh_idx];
        memset(m, 0, sizeof(*m));
        if (!nb_load_asset_mesh(ap, arena, m)) {
            nanousd_freeprim(ap);
            continue;
        }
        if (scene->materials) {
            MaterialCollection* mcoll = (MaterialCollection*)scene->materials;
            int mat_idx = materials_find_binding_by_path(mcoll, concrete_path);
            if (mat_idx >= 0 || !getenv("NUSD_SLOW_INSTANCE_BINDINGS")) {
                m->material_index = mat_idx;
            } else {
                m->material_index = materials_find_binding(mcoll, asset_stage, ap);
            }
        }
        double cw[16], rel_stage[16], rel_full[16];
        compute_world_xform(ap, cw);
        nanousd_freeprim(ap);
        nanousd_mul_m4d(cw, tinv, rel_stage);
        nanousd_mul_m4d(rel_stage, concrete_root_rel, rel_full);
        double world[16];
        nanousd_mul_m4d(rel_full, inst_world, world);
        if (scene_compact_pi_batches_active()) {
            nb_identity_m4d(m->world_xform);
            m->is_proto_only = 1;
        } else {
            memcpy(m->world_xform, world, sizeof(world));
            m->is_proto_only = 0;
        }
        m->path = scene_arena_strdup(arena, concrete_path);
        m->prototype_idx = *mesh_idx;
        m->lazy_prim_idx = -1;
        mesh_compute_local_bounds(m);
        if (scene_compact_pi_batches_active()) {
            if (emit_native_batches && !under_pi) {
                scene_add_compact_instance_batch(
                    scene, pi_xform_cap, pi_batch_cap, *mesh_idx, m, world,
                    SCENE_INSTANCE_SOURCE_NATIVE_INSTANCE);
            }
        } else {
            mesh_compute_world_bounds_from_local(m, scene);
        }
        m->vertex_offset = 0;
        m->index_offset = 0;
        if (!under_pi && cat_rel) nb_catalog_add(cat, cat_rel, *mesh_idx, rel_full);
        if (total_verts) *total_verts += m->nvertices;
        if (total_indices) *total_indices += m->nindices;
        (*mesh_idx)++;
        direct_meshes++;
        emitted++;
    }

    if (direct_meshes > 0 && !nb_chase_arcs_after_direct_active()) {
        if (getenv("NUSD_NB_ARC_DIAG"))
            fprintf(stderr,
                    "  NB recursive direct depth=%d target='%s' meshes=%d "
                    "handled=%d\n",
                    depth, tpath, direct_meshes, emitted);
        free(flat);
        return emitted;
    } else if (direct_meshes > 0 && getenv("NUSD_NB_ARC_DIAG")) {
        fprintf(stderr,
                "  NB recursive direct depth=%d target='%s' meshes=%d "
                "handled=%d; continuing arc chase\n",
                depth, tpath, direct_meshes, emitted);
    }

    /* Storm parity by default: child references/payloads can contain visible
     * XGen/prototype geometry even when the owning asset layer also authored
     * terrain meshes. NUSD_METAL_PARITY_ARC_SEED restores the old shortcut for
     * memory/perf bisects. */
    for (int i = 0; i < nflat && *mesh_idx < max_meshes; i++) {
        const char* fp = flat[i].path;
        const char* tn = flat[i].type_name;
        if (!fp || !tn || !nb_path_has_prefix(fp, tpath)) continue;

        const char* rel = fp + tlen;
        while (*rel == '/') rel++;
        char concrete_path[1024];
        if (!nb_join_path(concrete_root_path, rel, concrete_path, sizeof(concrete_path)))
            continue;

        NanousdPrim ap = nanousd_primpath(asset_stage, fp);
        if (!ap) continue;

        int arc_under_pi = 0;
        for (int z = 0; z < npi_pref; z++) {
            if (nb_path_has_prefix(fp, pi_pref[z])) { arc_under_pi = 1; break; }
        }
        int child_emit_native_batches = emit_native_batches && !arc_under_pi;

        int processed_direct_arcs = 0;
        int narcs = nanousd_ncomposition_arcs(ap);
        for (int a = 0; a < narcs; a++) {
            NanousdCompositionArc arc;
            memset(&arc, 0, sizeof(arc));
            if (!nanousd_composition_arc(ap, a, &arc)) continue;
            if (arc.arc_type != NANOUSD_ARC_REFERENCE &&
                arc.arc_type != NANOUSD_ARC_PAYLOAD) continue;
            if ((arc.flags & NANOUSD_COMPOSITION_ARC_DIRECT) == 0) continue;
            if (!arc.layer_path || !arc.layer_path[0]) continue;
            if (nb_skip_arc_for_geometry_only(arc.layer_path, arc.source_path))
                continue;
            NanousdStage sub = nanousd_open(arc.layer_path);
            if (!sub || !nanousd_isvalid(sub)) {
                if (sub) nanousd_close(sub);
                continue;
            }
            const char* child_target =
                (arc.source_path && arc.source_path[0]) ? arc.source_path : NULL;
            nb_apply_instance_variants_to_asset_target(inst_prim, sub, child_target);

            double owner_world[16], owner_to_target[16], child_root_rel[16];
            compute_world_xform(ap, owner_world);
            nanousd_mul_m4d(owner_world, tinv, owner_to_target);
            nanousd_mul_m4d(owner_to_target, concrete_root_rel, child_root_rel);
            if (getenv("NUSD_NB_ARC_DIAG"))
                fprintf(stderr,
                        "  NB arc depth=%d owner='%s' concrete='%s' "
                        "layer='%s' target='%s' under_pi=%d\n",
                        depth, fp, concrete_path, arc.layer_path,
                        child_target ? child_target : "<default>",
                        arc_under_pi);
            int child_emitted = nb_seed_asset_stage_recursive(
                scene, main_stage, inst_prim, sub, child_target,
                catalog_root_path, inst_world, concrete_path, child_root_rel,
                cat, arena, mesh_idx, max_meshes, total_verts, total_indices,
                nested, nested_n, nested_cap, pi_xform_cap, pi_batch_cap,
                child_emit_native_batches, depth + 1);
            processed_direct_arcs++;
            emitted += child_emitted;
            if (getenv("NUSD_NB_ARC_DIAG"))
                fprintf(stderr,
                        "  NB arc depth=%d owner='%s' emitted=%d mesh_idx=%d\n",
                        depth, fp, child_emitted, *mesh_idx);
            nanousd_close(sub);
        }

        if (processed_direct_arcs == 0) for (int lf = 0; lf < 2; lf++) {
            const char* field = lf == 0 ? "payload" : "references";
            NanousdListOp op = nanousd_prim_listop(ap, field);
            if (!op) continue;
            int nitems = nanousd_listop_nitems(op);
            for (int oi = 0; oi < nitems; oi++) {
                const char* item = nanousd_listop_item(op, oi);
                char asset[1024], target_item[1024], resolved[2048];
                if (!nb_parse_asset_arc_item(item, asset, sizeof(asset),
                                             target_item, sizeof(target_item)))
                    continue;
                if (nb_skip_arc_for_geometry_only(asset, target_item))
                    continue;
                const char* anchor = nanousd_stage_get_root_layer_path(asset_stage);
                if (!anchor || !anchor[0]) continue;
                if (!nanousd_resolve_asset_path(anchor, asset,
                                                resolved, sizeof(resolved)))
                    continue;
                NanousdStage sub = nanousd_open(resolved);
                if (!sub || !nanousd_isvalid(sub)) {
                    if (sub) nanousd_close(sub);
                    continue;
                }
                const char* child_target = target_item[0] ? target_item : NULL;
                nb_apply_instance_variants_to_asset_target(inst_prim, sub, child_target);

                double owner_world[16], owner_to_target[16], child_root_rel[16];
                compute_world_xform(ap, owner_world);
                nanousd_mul_m4d(owner_world, tinv, owner_to_target);
                nanousd_mul_m4d(owner_to_target, concrete_root_rel, child_root_rel);
                if (getenv("NUSD_NB_ARC_DIAG"))
                    fprintf(stderr,
                            "  NB listop depth=%d owner='%s' concrete='%s' "
                            "asset='%s' resolved='%s' target='%s' under_pi=%d\n",
                            depth, fp, concrete_path, asset, resolved,
                            child_target ? child_target : "<default>",
                            arc_under_pi);
                int child_emitted = nb_seed_asset_stage_recursive(
                    scene, main_stage, inst_prim, sub, child_target,
                    catalog_root_path, inst_world, concrete_path, child_root_rel,
                    cat, arena, mesh_idx, max_meshes, total_verts, total_indices,
                    nested, nested_n, nested_cap, pi_xform_cap, pi_batch_cap,
                    child_emit_native_batches, depth + 1);
                emitted += child_emitted;
                if (getenv("NUSD_NB_ARC_DIAG"))
                    fprintf(stderr,
                            "  NB listop depth=%d owner='%s' emitted=%d mesh_idx=%d\n",
                            depth, fp, child_emitted, *mesh_idx);
                nanousd_close(sub);
            }
            nanousd_listop_free(op);
        }
        nanousd_freeprim(ap);
    }

    if (!target_seen_in_flat && *mesh_idx < max_meshes) {
        NanousdPrim ap = nanousd_primpath(asset_stage, tpath);
        if (ap) {
            const char* fp = tpath;
            const char* concrete_path = concrete_root_path;
            int arc_under_pi = !emit_native_batches;
            int child_emit_native_batches = emit_native_batches && !arc_under_pi;

            int processed_direct_arcs = 0;
            int narcs = nanousd_ncomposition_arcs(ap);
            for (int a = 0; a < narcs; a++) {
                NanousdCompositionArc arc;
                memset(&arc, 0, sizeof(arc));
                if (!nanousd_composition_arc(ap, a, &arc)) continue;
                if (arc.arc_type != NANOUSD_ARC_REFERENCE &&
                    arc.arc_type != NANOUSD_ARC_PAYLOAD) continue;
                if ((arc.flags & NANOUSD_COMPOSITION_ARC_DIRECT) == 0) continue;
                if (!arc.layer_path || !arc.layer_path[0]) continue;
                if (nb_skip_arc_for_geometry_only(arc.layer_path, arc.source_path))
                    continue;
                NanousdStage sub = nanousd_open(arc.layer_path);
                if (!sub || !nanousd_isvalid(sub)) {
                    if (sub) nanousd_close(sub);
                    continue;
                }
                const char* child_target =
                    (arc.source_path && arc.source_path[0]) ? arc.source_path : NULL;
                nb_apply_instance_variants_to_asset_target(inst_prim, sub, child_target);

                double owner_world[16], owner_to_target[16], child_root_rel[16];
                compute_world_xform(ap, owner_world);
                nanousd_mul_m4d(owner_world, tinv, owner_to_target);
                nanousd_mul_m4d(owner_to_target, concrete_root_rel, child_root_rel);
                if (getenv("NUSD_NB_ARC_DIAG"))
                    fprintf(stderr,
                            "  NB target-arc depth=%d owner='%s' concrete='%s' "
                            "layer='%s' target='%s' under_pi=%d\n",
                            depth, fp, concrete_path, arc.layer_path,
                            child_target ? child_target : "<default>",
                            arc_under_pi);
                int child_emitted = nb_seed_asset_stage_recursive(
                    scene, main_stage, inst_prim, sub, child_target,
                    catalog_root_path, inst_world, concrete_path, child_root_rel,
                    cat, arena, mesh_idx, max_meshes, total_verts, total_indices,
                    nested, nested_n, nested_cap, pi_xform_cap, pi_batch_cap,
                    child_emit_native_batches, depth + 1);
                processed_direct_arcs++;
                emitted += child_emitted;
                if (getenv("NUSD_NB_ARC_DIAG"))
                    fprintf(stderr,
                            "  NB target-arc depth=%d owner='%s' emitted=%d mesh_idx=%d\n",
                            depth, fp, child_emitted, *mesh_idx);
                nanousd_close(sub);
            }

            if (processed_direct_arcs == 0) for (int lf = 0; lf < 2; lf++) {
                const char* field = lf == 0 ? "payload" : "references";
                NanousdListOp op = nanousd_prim_listop(ap, field);
                if (!op) continue;
                int nitems = nanousd_listop_nitems(op);
                for (int oi = 0; oi < nitems; oi++) {
                    const char* item = nanousd_listop_item(op, oi);
                    char asset[1024], target_item[1024], resolved[2048];
                    if (!nb_parse_asset_arc_item(item, asset, sizeof(asset),
                                                 target_item, sizeof(target_item)))
                        continue;
                    if (nb_skip_arc_for_geometry_only(asset, target_item))
                        continue;
                    const char* anchor = nanousd_stage_get_root_layer_path(asset_stage);
                    if (!anchor || !anchor[0]) continue;
                    if (!nanousd_resolve_asset_path(anchor, asset,
                                                    resolved, sizeof(resolved)))
                        continue;
                    NanousdStage sub = nanousd_open(resolved);
                    if (!sub || !nanousd_isvalid(sub)) {
                        if (sub) nanousd_close(sub);
                        continue;
                    }
                    const char* child_target = target_item[0] ? target_item : NULL;
                    nb_apply_instance_variants_to_asset_target(inst_prim, sub, child_target);

                    double owner_world[16], owner_to_target[16], child_root_rel[16];
                    compute_world_xform(ap, owner_world);
                    nanousd_mul_m4d(owner_world, tinv, owner_to_target);
                    nanousd_mul_m4d(owner_to_target, concrete_root_rel, child_root_rel);
                    if (getenv("NUSD_NB_ARC_DIAG"))
                        fprintf(stderr,
                                "  NB target-listop depth=%d owner='%s' concrete='%s' "
                                "asset='%s' resolved='%s' target='%s' under_pi=%d\n",
                                depth, fp, concrete_path, asset, resolved,
                                child_target ? child_target : "<default>",
                                arc_under_pi);
                    int child_emitted = nb_seed_asset_stage_recursive(
                        scene, main_stage, inst_prim, sub, child_target,
                        catalog_root_path, inst_world, concrete_path, child_root_rel,
                        cat, arena, mesh_idx, max_meshes, total_verts, total_indices,
                        nested, nested_n, nested_cap, pi_xform_cap, pi_batch_cap,
                        child_emit_native_batches, depth + 1);
                    emitted += child_emitted;
                    if (getenv("NUSD_NB_ARC_DIAG"))
                        fprintf(stderr,
                                "  NB target-listop depth=%d owner='%s' emitted=%d mesh_idx=%d\n",
                                depth, fp, child_emitted, *mesh_idx);
                    nanousd_close(sub);
                }
                nanousd_listop_free(op);
            }
            nanousd_freeprim(ap);
        }
    }

    free(flat);
    return emitted;
}

static int nb_seed_top_level_xgen_archive_proto(Scene* scene,
                                                NanousdStage stage,
                                                const char* pi_path,
                                                const char* proto_path,
                                                Arena* arena,
                                                int* mesh_idx,
                                                int max_meshes,
                                                uint64_t* total_verts,
                                                uint64_t* total_indices)
{
    if (!scene || !stage || !pi_path || !proto_path || !arena || !mesh_idx ||
        *mesh_idx >= max_meshes) {
        return 0;
    }
    const char* root_layer = nanousd_stage_get_root_layer_path(stage);
    if (!root_layer || !strstr(root_layer, "/elements/")) {
        return 0;
    }
    const char* instancer_seg = strstr(pi_path, "/instancer");
    if (!instancer_seg) return 0;

    char owner_path[1024];
    size_t owner_len = (size_t)(instancer_seg - pi_path);
    if (owner_len == 0 || owner_len >= sizeof(owner_path)) return 0;
    memcpy(owner_path, pi_path, owner_len);
    owner_path[owner_len] = '\0';
    const char* owner_name = strrchr(owner_path, '/');
    owner_name = owner_name ? owner_name + 1 : owner_path;
    if (!owner_name[0]) return 0;

    char xgen_stage_path[2048];
    xgen_stage_path[0] = '\0';
    NanousdPrim owner = nanousd_primpath(stage, owner_path);
    if (owner) {
        int narcs = nanousd_ncomposition_arcs(owner);
        for (int a = 0; a < narcs && !xgen_stage_path[0]; a++) {
            NanousdCompositionArc arc;
            memset(&arc, 0, sizeof(arc));
            if (!nanousd_composition_arc(owner, a, &arc)) continue;
            if (arc.arc_type != NANOUSD_ARC_REFERENCE &&
                arc.arc_type != NANOUSD_ARC_PAYLOAD) continue;
            if ((arc.flags & NANOUSD_COMPOSITION_ARC_DIRECT) == 0) continue;
            if (arc.layer_path && strstr(arc.layer_path, "/xgenInstances/"))
                snprintf(xgen_stage_path, sizeof(xgen_stage_path), "%s",
                         arc.layer_path);
        }
        for (int lf = 0; lf < 2 && !xgen_stage_path[0]; lf++) {
            const char* field = lf == 0 ? "payload" : "references";
            NanousdListOp op = nanousd_prim_listop(owner, field);
            if (!op) continue;
            int nitems = nanousd_listop_nitems(op);
            for (int oi = 0; oi < nitems && !xgen_stage_path[0]; oi++) {
                const char* item = nanousd_listop_item(op, oi);
                char asset[1024], target_item[1024], resolved[2048];
                if (!nb_parse_asset_arc_item(item, asset, sizeof(asset),
                                             target_item, sizeof(target_item)))
                    continue;
                const char* anchor = nanousd_stage_get_root_layer_path(stage);
                if (!anchor || !anchor[0]) continue;
                if (nanousd_resolve_asset_path(anchor, asset,
                                               resolved, sizeof(resolved)) &&
                    strstr(resolved, "/xgenInstances/")) {
                    snprintf(xgen_stage_path, sizeof(xgen_stage_path), "%s",
                             resolved);
                }
            }
            nanousd_listop_free(op);
        }
        nanousd_freeprim(owner);
    }

    if (!xgen_stage_path[0]) {
        const char* anchor = root_layer;
        if (!anchor || !anchor[0]) return 0;
        char rel_bundle[512];
        snprintf(rel_bundle, sizeof(rel_bundle), "./xgenInstances/%.*s.usd",
                 480, owner_name);
        if (!nanousd_resolve_asset_path(anchor, rel_bundle,
                                        xgen_stage_path,
                                        sizeof(xgen_stage_path))) {
            return 0;
        }
    }

    const char* base = strrchr(proto_path, '/');
    base = base ? base + 1 : proto_path;
    if (!base[0]) return 0;
    char rel_archive[512], archive_path[2048];
    snprintf(rel_archive, sizeof(rel_archive), "archives/%s.usd", base);
    if (!nanousd_resolve_asset_path(xgen_stage_path, rel_archive,
                                    archive_path, sizeof(archive_path))) {
        return 0;
    }

    NanousdStage arch = nanousd_open(archive_path);
    if (!arch || !nanousd_isvalid(arch)) {
        if (arch) nanousd_close(arch);
        return 0;
    }

    NbReplayCatalog dummy_cat;
    memset(&dummy_cat, 0, sizeof(dummy_cat));
    char archive_target[256];
    archive_target[0] = '\0';
    if (proto_path[0] == '/') {
        const char* slash2 = strchr(proto_path + 1, '/');
        size_t n = slash2 ? (size_t)(slash2 - proto_path) : strlen(proto_path);
        if (n > 1 && n < sizeof(archive_target)) {
            memcpy(archive_target, proto_path, n);
            archive_target[n] = '\0';
        }
    }
    int before = *mesh_idx;
    int emitted = nb_seed_asset_stage_recursive(
        scene, stage, NULL, arch,
        archive_target[0] ? archive_target : NULL,
        "/__NUSD_PI_PROTO_NO_CATALOG__", kIdentity, proto_path, kIdentity,
        &dummy_cat, arena, mesh_idx, max_meshes, total_verts, total_indices,
        NULL, NULL, NULL, NULL, NULL, 0, 0);
    free(dummy_cat.entries);
    free(dummy_cat.pi_entries);
    free(dummy_cat.mesh_path_set);
    free(dummy_cat.xgen_archive_set);
    nanousd_close(arch);

    if (getenv("NUSD_PI_PROTO_DIAG") && *mesh_idx > before) {
        fprintf(stderr,
                "scene_load: PointInstancer '%s': seeded unresolved xgen "
                "prototype '%s' from '%s' (%d mesh(es))\n",
                pi_path, proto_path, archive_path, *mesh_idx - before);
    }
    return emitted;
}

static int nb_emit_replay(Scene* scene, int* mesh_idx, int max_meshes,
                          const char* inst_root_path, const double inst_world[16],
                          const NbReplayCatalog* cat, Arena* arena,
                          uint64_t* total_verts, uint64_t* total_indices,
                          struct NestedPiRec** nested, int* nested_n, int* nested_cap,
                          uint64_t* pi_xform_cap, int* pi_batch_cap,
                          int emit_native_batches) {
    int emitted = 0;
    for (int i = 0; i < cat->count; i++) {
        const NbReplayEntry* e = &cat->entries[i];
        if (e->prototype_idx < 0 || e->prototype_idx >= *mesh_idx) continue;
        SceneMesh* pm = &scene->meshes[e->prototype_idx];
        if (!pm->positions || !pm->indices || pm->nvertices <= 0 || pm->nindices <= 0) continue;
        double world[16];
        nanousd_mul_m4d(e->relative_xform, inst_world, world);
        if (scene_compact_pi_batches_active()) {
            if (emit_native_batches) {
                scene_add_compact_instance_batch(
                    scene, pi_xform_cap, pi_batch_cap, e->prototype_idx, pm, world,
                    SCENE_INSTANCE_SOURCE_NATIVE_INSTANCE);
            }
            emitted++;
            continue;
        }
        if (*mesh_idx >= max_meshes) break;
        SceneMesh* m = &scene->meshes[*mesh_idx];
        memset(m, 0, sizeof(*m));
        m->positions = pm->positions; m->normals = pm->normals; m->indices = pm->indices;
        m->nvertices = pm->nvertices; m->nindices = pm->nindices;
        m->colors = pm->colors; m->texcoords = pm->texcoords;
        m->has_display_color = pm->has_display_color;
        m->display_color[0] = pm->display_color[0];
        m->display_color[1] = pm->display_color[1];
        m->display_color[2] = pm->display_color[2];
        m->material_index = pm->material_index;
        m->prototype_idx = e->prototype_idx;
        m->is_proto_only = 0;
        m->lazy_prim_idx = -1;
        memcpy(m->local_bounds_min, pm->local_bounds_min, sizeof(float) * 3);
        memcpy(m->local_bounds_max, pm->local_bounds_max, sizeof(float) * 3);
        memcpy(m->world_xform, world, sizeof(world));
        char pathbuf[1024];
        if (inst_root_path && inst_root_path[0] && e->relative_path[0])
            snprintf(pathbuf, sizeof(pathbuf), "%s/%s", inst_root_path, e->relative_path);
        else if (inst_root_path && inst_root_path[0])
            snprintf(pathbuf, sizeof(pathbuf), "%s", inst_root_path);
        else pathbuf[0] = '\0';
        m->path = scene_arena_strdup(arena, pathbuf);
        mesh_compute_world_bounds_from_local(m, scene);
        m->vertex_offset = 0; m->index_offset = 0;
        if (total_verts) *total_verts += m->nvertices;
        if (total_indices) *total_indices += m->nindices;
        (*mesh_idx)++; emitted++;
    }
    /* Re-record the prototype's nested PIs for THIS sibling instance (each
     * sibling has the same proto-local PI layout but its own inst_world). */
    for (int i = 0; i < cat->pi_count && nested && nested_n && nested_cap; i++) {
        const NbPiEntry* pe = &cat->pi_entries[i];
        double pti[16];
        nanousd_mul_m4d(pe->rel_xform, inst_world, pti);
        char mp[1024];
        if (inst_root_path && inst_root_path[0] && pe->rel_path[0])
            snprintf(mp, sizeof(mp), "%s/%s", inst_root_path, pe->rel_path);
        else if (inst_root_path && inst_root_path[0])
            snprintf(mp, sizeof(mp), "%s", inst_root_path);
        else continue;
        if (*nested_n >= *nested_cap) {
            int ncp = *nested_cap ? *nested_cap * 2 : 16;
            struct NestedPiRec* g = (struct NestedPiRec*)realloc(*nested, (size_t)ncp * sizeof(**nested));
            if (g) { *nested = g; *nested_cap = ncp; }
        }
        if (*nested && *nested_n < *nested_cap) {
            snprintf((*nested)[*nested_n].path, sizeof((*nested)[0].path), "%s", mp);
            memcpy((*nested)[*nested_n].pi_to_inst, pti, sizeof(pti));
            (*nested_n)++; emitted++;
        }
    }
    return emitted;
}

typedef struct {
    char       relative_path[512];
    SceneCurve proto;
    double     relative_xform[16];
} NbCurveEntry;

typedef struct {
    char          key[768];
    NbCurveEntry* entries;
    int           count, cap, built;
    int           skipped_by_policy;
    uint64_t*     path_set;
    int           path_count, path_cap;
} NbCurveCatalog;

typedef struct {
    NbCurveCatalog* items;
    int             count, cap;
} NbCurveCache;

typedef struct {
    SceneCurve* items;
    int         count, cap;
} NbCurveList;

static int nb_native_curves_enabled(void)
{
    const char* e = getenv("NUSD_NATIVE_CURVES");
    if (e && (!strcmp(e, "0") || !strcmp(e, "off") ||
              !strcmp(e, "false") || !strcmp(e, "no"))) {
        return 0;
    }
    return 1;
}

static int nb_native_curves_all(void)
{
    const char* e = getenv("NUSD_NATIVE_CURVES");
    if (!e || !e[0]) return 1;
    return !strcmp(e, "all") || !strcmp(e, "ALL") ||
           !strcmp(e, "1") || !strcmp(e, "on") ||
           !strcmp(e, "true") || !strcmp(e, "yes");
}

static int nb_native_curve_path_allowed(const char* path)
{
    if (nb_native_curves_all()) return 1;
    /* Optional memory-saving mode for Moana-scale previews:
     * NUSD_NATIVE_CURVES=woody/no-needles keeps branch/twig/stem curves while
     * skipping dense needle curves. Default loads everything Storm sees. */
    return !scene_contains_ci(path, "needle");
}

static NbCurveCatalog* nb_curve_cache_find_or_add(NbCurveCache* c,
                                                  const char* key)
{
    if (!c || !key || !key[0]) return NULL;
    for (int i = 0; i < c->count; i++)
        if (!strcmp(c->items[i].key, key)) return &c->items[i];
    if (c->count >= c->cap) {
        int nc = c->cap ? c->cap * 2 : 16;
        NbCurveCatalog* g =
            (NbCurveCatalog*)realloc(c->items, (size_t)nc * sizeof(*g));
        if (!g) return NULL;
        memset(g + c->cap, 0, (size_t)(nc - c->cap) * sizeof(*g));
        c->items = g;
        c->cap = nc;
    }
    NbCurveCatalog* cat = &c->items[c->count++];
    memset(cat, 0, sizeof(*cat));
    snprintf(cat->key, sizeof(cat->key), "%s", key);
    return cat;
}

static void nb_curve_cache_free(NbCurveCache* c)
{
    if (!c) return;
    for (int i = 0; i < c->count; i++) {
        free(c->items[i].entries);
        free(c->items[i].path_set);
    }
    free(c->items);
    c->items = NULL;
    c->count = c->cap = 0;
}

static NbCurveEntry* nb_curve_catalog_add(NbCurveCatalog* cat,
                                          const char* rel,
                                          const SceneCurve* proto,
                                          const double rx[16])
{
    if (!cat || !rel || !proto || !rx) return NULL;
    while (*rel == '/') rel++;
    uint64_t rel_hash = nb_rel_hash(rel);
    if (nb_path_set_contains(cat->path_set, cat->path_cap, rel_hash))
        return NULL;
    if (!nb_path_set_insert(&cat->path_set, &cat->path_count,
                            &cat->path_cap, rel_hash))
        return NULL;
    if (cat->count >= cat->cap) {
        int nc = cat->cap ? cat->cap * 2 : 64;
        NbCurveEntry* g =
            (NbCurveEntry*)realloc(cat->entries, (size_t)nc * sizeof(*g));
        if (!g) return NULL;
        cat->entries = g;
        cat->cap = nc;
    }
    NbCurveEntry* e = &cat->entries[cat->count++];
    memset(e, 0, sizeof(*e));
    snprintf(e->relative_path, sizeof(e->relative_path), "%s", rel);
    e->proto = *proto;
    memcpy(e->relative_xform, rx, sizeof(double) * 16);
    return e;
}

static int nb_curve_list_add(NbCurveList* list, const SceneCurve* curve)
{
    if (!list || !curve) return 0;
    if (list->count >= list->cap) {
        int nc = list->cap ? list->cap * 2 : 64;
        SceneCurve* g =
            (SceneCurve*)realloc(list->items, (size_t)nc * sizeof(*g));
        if (!g) return 0;
        list->items = g;
        list->cap = nc;
    }
    list->items[list->count++] = *curve;
    return 1;
}

static void scene_curve_recompute_bounds(SceneCurve* curve)
{
    if (!curve) return;
    bounds_init(curve->bounds_min, curve->bounds_max);
    for (int v = 0; v < curve->nv; v++) {
        float wp[3];
        xform_point(curve->world_xform, &curve->cvs[v * 3], wp);
        float r = curve->widths ? curve->widths[v] * 0.5f : 0.0f;
        for (int k = 0; k < 3; k++) {
            float lo = wp[k] - r;
            float hi = wp[k] + r;
            if (lo < curve->bounds_min[k]) curve->bounds_min[k] = lo;
            if (hi > curve->bounds_max[k]) curve->bounds_max[k] = hi;
        }
    }
}

static int nb_prim_has_variant_arc(NanousdPrim prim)
{
    if (!prim) return 0;
    int narcs = nanousd_ncomposition_arcs(prim);
    for (int a = 0; a < narcs; a++) {
        NanousdCompositionArc va;
        memset(&va, 0, sizeof(va));
        if (!nanousd_composition_arc(prim, a, &va)) continue;
        if (va.arc_type == NANOUSD_ARC_VARIANT) return 1;
    }
    return 0;
}

static int nb_arc_shadowed_by_variant_arc(NanousdPrim prim,
                                          const NanousdCompositionArc* arc)
{
    if (!prim || !arc || !arc->layer_path || !arc->source_path) return 0;
    int narcs = nanousd_ncomposition_arcs(prim);
    size_t src_len = strlen(arc->source_path);
    for (int a = 0; a < narcs; a++) {
        NanousdCompositionArc va;
        memset(&va, 0, sizeof(va));
        if (!nanousd_composition_arc(prim, a, &va)) continue;
        if (va.arc_type != NANOUSD_ARC_VARIANT) continue;
        if (!va.layer_path || !va.source_path) continue;
        if (strcmp(va.layer_path, arc->layer_path) != 0) continue;
        if (strncmp(va.source_path, arc->source_path, src_len) == 0 &&
            va.source_path[src_len] == '{') {
            return 1;
        }
    }
    return 0;
}

static int nb_seed_curve_asset_stage_recursive(Scene* scene,
                                               NanousdPrim inst_prim,
                                               NanousdStage asset_stage,
                                               const char* target_path,
                                               const char* catalog_root_path,
                                               const char* concrete_root_path,
                                               const double concrete_root_rel[16],
                                               NbCurveCatalog* cat,
                                               Arena* arena,
                                               int depth)
{
    if (!scene || !asset_stage || !catalog_root_path || !concrete_root_path ||
        !concrete_root_rel || !cat || !arena || depth > 5) {
        return 0;
    }

    NanousdPrim target = (target_path && target_path[0])
        ? nanousd_primpath(asset_stage, target_path) : NULL;
    if (!target) target = nanousd_defaultprim(asset_stage);
    if (!target) return 0;

    const char* tpr = nanousd_path(target);
    char tpath[1024];
    tpath[0] = '\0';
    if (tpr) snprintf(tpath, sizeof(tpath), "%s", tpr);
    double tworld[16], tinv[16];
    compute_world_xform(target, tworld);
    int have_tinv = invert_affine_m4d_rowvec(tworld, tinv);
    nanousd_freeprim(target);
    if (!have_tinv || !tpath[0]) return 0;
    size_t tlen = strlen(tpath);

    int nflat = nanousd_traverse_flat(asset_stage, NULL, 0);
    if (nflat <= 0) return 0;
    NanousdFlatPrim* flat =
        (NanousdFlatPrim*)calloc((size_t)nflat, sizeof(NanousdFlatPrim));
    if (!flat) return 0;
    int total = nanousd_traverse_flat(asset_stage, flat, nflat);
    if (total > 0 && total < nflat) nflat = total;

    int emitted = 0;
    char pi_pref[64][512];
    int npi_pref = 0;

    for (int i = 0; i < nflat; i++) {
        const char* fp = flat[i].path;
        const char* tn = flat[i].type_name;
        if (!fp || !tn || !nb_path_has_prefix(fp, tpath)) continue;

        if (!strcmp(tn, "PointInstancer")) {
            if (npi_pref < 64) {
                snprintf(pi_pref[npi_pref], sizeof(pi_pref[0]), "%s", fp);
                npi_pref++;
            }
            continue;
        }
        if (strcmp(tn, "BasisCurves") != 0) continue;

        int under_pi = 0;
        for (int z = 0; z < npi_pref; z++) {
            if (nb_path_has_prefix(fp, pi_pref[z])) { under_pi = 1; break; }
        }
        if (under_pi) continue;

        const char* rel = fp + tlen;
        while (*rel == '/') rel++;
        char concrete_path[1024];
        if (!nb_join_path(concrete_root_path, rel, concrete_path,
                          sizeof(concrete_path)))
            continue;
        const char* cat_rel = nb_rel_under(concrete_path, catalog_root_path);
        if (!cat_rel) continue;
        if (nb_path_set_contains(cat->path_set, cat->path_cap,
                                 nb_rel_hash(cat_rel)))
            continue;
        if (!nb_native_curve_path_allowed(concrete_path)) {
            nb_path_set_insert(&cat->path_set, &cat->path_count,
                               &cat->path_cap, nb_rel_hash(cat_rel));
            cat->skipped_by_policy++;
            continue;
        }

        NanousdPrim ap = nanousd_primpath(asset_stage, fp);
        if (!ap) continue;
        SceneCurve curve;
        memset(&curve, 0, sizeof(curve));
        MaterialCollection* materials = (MaterialCollection*)scene->materials;
        if (!load_basiscurves(asset_stage, ap, arena, &curve, materials)) {
            nanousd_freeprim(ap);
            continue;
        }
        if (materials) {
            int mat_idx = materials_find_binding_by_path(materials,
                                                         concrete_path);
            if (mat_idx >= 0) curve.material_index = mat_idx;
        }
        double cw[16], rel_stage[16], rel_full[16];
        compute_world_xform(ap, cw);
        nanousd_freeprim(ap);
        nanousd_mul_m4d(cw, tinv, rel_stage);
        nanousd_mul_m4d(rel_stage, concrete_root_rel, rel_full);
        if (nb_curve_catalog_add(cat, cat_rel, &curve, rel_full)) {
            emitted++;
        }
    }

    for (int i = 0; i < nflat; i++) {
        const char* fp = flat[i].path;
        const char* tn = flat[i].type_name;
        if (!fp || !tn || !nb_path_has_prefix(fp, tpath)) continue;
        const char* rel = fp + tlen;
        while (*rel == '/') rel++;
        char concrete_path[1024];
        if (!nb_join_path(concrete_root_path, rel, concrete_path,
                          sizeof(concrete_path)))
            continue;

        NanousdPrim ap = nanousd_primpath(asset_stage, fp);
        if (!ap) continue;

        int narcs = nanousd_ncomposition_arcs(ap);
        for (int a = 0; a < narcs; a++) {
            NanousdCompositionArc arc;
            memset(&arc, 0, sizeof(arc));
            if (!nanousd_composition_arc(ap, a, &arc)) continue;
            if (arc.arc_type != NANOUSD_ARC_REFERENCE &&
                arc.arc_type != NANOUSD_ARC_PAYLOAD) continue;
            if ((arc.flags & NANOUSD_COMPOSITION_ARC_DIRECT) == 0) continue;
            if (!arc.layer_path || !arc.layer_path[0]) continue;
            if (nb_arc_shadowed_by_variant_arc(ap, &arc)) continue;
            if (nb_skip_arc_for_geometry_only(arc.layer_path, arc.source_path))
                continue;
            NanousdStage sub = nanousd_open(arc.layer_path);
            if (!sub || !nanousd_isvalid(sub)) {
                if (sub) nanousd_close(sub);
                continue;
            }
            const char* child_target =
                (arc.source_path && arc.source_path[0]) ? arc.source_path : NULL;
            nb_apply_instance_variants_to_asset_target(inst_prim, sub,
                                                       child_target);
            nb_apply_instance_variants_to_asset_target(ap, sub, child_target);

            double owner_world[16], owner_to_target[16], child_root_rel[16];
            compute_world_xform(ap, owner_world);
            nanousd_mul_m4d(owner_world, tinv, owner_to_target);
            nanousd_mul_m4d(owner_to_target, concrete_root_rel,
                            child_root_rel);
            emitted += nb_seed_curve_asset_stage_recursive(
                scene, inst_prim, sub, child_target, catalog_root_path,
                concrete_path, child_root_rel, cat, arena, depth + 1);
            nanousd_close(sub);
        }

        if (nb_prim_has_variant_arc(ap)) {
            nanousd_freeprim(ap);
            continue;
        }

        for (int lf = 0; lf < 2; lf++) {
            const char* field = lf == 0 ? "payload" : "references";
            NanousdListOp op = nanousd_prim_listop(ap, field);
            if (!op) continue;
            int nitems = nanousd_listop_nitems(op);
            for (int oi = 0; oi < nitems; oi++) {
                const char* item = nanousd_listop_item(op, oi);
                char asset[1024], target_item[1024], resolved[2048];
                if (!nb_parse_asset_arc_item(item, asset, sizeof(asset),
                                             target_item, sizeof(target_item)))
                    continue;
                if (nb_skip_arc_for_geometry_only(asset, target_item))
                    continue;
                const char* anchor = nanousd_stage_get_root_layer_path(asset_stage);
                if (!anchor || !anchor[0]) continue;
                if (!nanousd_resolve_asset_path(anchor, asset,
                                                resolved, sizeof(resolved)))
                    continue;
                NanousdStage sub = nanousd_open(resolved);
                if (!sub || !nanousd_isvalid(sub)) {
                    if (sub) nanousd_close(sub);
                    continue;
                }
                const char* child_target = target_item[0] ? target_item : NULL;
                nb_apply_instance_variants_to_asset_target(inst_prim, sub,
                                                           child_target);
                nb_apply_instance_variants_to_asset_target(ap, sub, child_target);

                double owner_world[16], owner_to_target[16], child_root_rel[16];
                compute_world_xform(ap, owner_world);
                nanousd_mul_m4d(owner_world, tinv, owner_to_target);
                nanousd_mul_m4d(owner_to_target, concrete_root_rel,
                                child_root_rel);
                emitted += nb_seed_curve_asset_stage_recursive(
                    scene, inst_prim, sub, child_target, catalog_root_path,
                    concrete_path, child_root_rel, cat, arena, depth + 1);
                nanousd_close(sub);
            }
            nanousd_listop_free(op);
        }
        nanousd_freeprim(ap);
    }

    free(flat);
    return emitted;
}

static int nb_emit_curve_replay(Scene* scene, NbCurveList* out,
                                const char* inst_root_path,
                                const double inst_world[16],
                                const NbCurveCatalog* cat)
{
    if (!scene || !out || !inst_world || !cat) return 0;
    int emitted = 0;
    MaterialCollection* materials = (MaterialCollection*)scene->materials;
    for (int i = 0; i < cat->count; i++) {
        const NbCurveEntry* e = &cat->entries[i];
        SceneCurve curve = e->proto;
        nanousd_mul_m4d(e->relative_xform, inst_world, curve.world_xform);
        build_normal_matrix(curve.world_xform, curve.normal_xform);
        if (materials && inst_root_path && inst_root_path[0]) {
            char pathbuf[1024];
            if (e->relative_path[0])
                snprintf(pathbuf, sizeof(pathbuf), "%s/%s", inst_root_path,
                         e->relative_path);
            else
                snprintf(pathbuf, sizeof(pathbuf), "%s", inst_root_path);
            int mat_idx = materials_find_binding_by_path(materials, pathbuf);
            if (mat_idx >= 0) curve.material_index = mat_idx;
        }
        scene_curve_recompute_bounds(&curve);
        scene_expand_bounds(scene, curve.bounds_min, curve.bounds_max);
        if (!nb_curve_list_add(out, &curve)) break;
        emitted++;
    }
    return emitted;
}

static int scene_load_native_instance_curves(NanousdStage stage,
                                             Scene* scene,
                                             Arena* arena)
{
    if (!nb_native_curves_enabled() || !stage || !scene || !arena) return 0;

    /* Gate: if compact-native's part-(A) scanned every prototype subtree and saw
     * NO BasisCurves, this whole pass would open + flat-traverse (proxy-expanding)
     * hundreds of asset sub-stages to find zero curves — the DSX 7+ min stall.
     * Skip it. Only applies when part-(A) actually ran (compact-native active);
     * otherwise checked_proto_for_curves==0 and behavior is unchanged. */
    if (scene->checked_proto_for_curves && !scene->has_basis_curves) {
        fprintf(stderr, "scene_load: native instance curves — skipped "
                "(prototype scan found no BasisCurves; avoids proxy-expansion walk)\n");
        return 0;
    }

    int nprims = nanousd_nprims(stage);
    NbCurveCache cache = {0, 0, 0};
    NbCurveList replayed = {0, 0, 0};
    int roots = 0;
    int catalogs_built = 0;
    int catalog_entries = 0;
    int emitted_total = 0;
    int skipped_total = 0;

    for (int i = 0; i < nprims; i++) {
        NanousdPrim inst_prim = nanousd_prim(stage, i);
        if (!inst_prim) continue;
        if (!nanousd_isactive(inst_prim) || !nanousd_isinstance(inst_prim)) {
            nanousd_freeprim(inst_prim);
            continue;
        }
        const char* ip = nanousd_path(inst_prim);
        char inst_path[1024];
        inst_path[0] = '\0';
        if (ip) snprintf(inst_path, sizeof(inst_path), "%s", ip);
        if (!inst_path[0]) {
            nanousd_freeprim(inst_prim);
            continue;
        }

        NanousdPrim proto_root = nanousd_prototype(inst_prim);
        char key[768];
        nb_instance_key(inst_prim, proto_root, key, sizeof(key));
        NbCurveCatalog* cat = nb_curve_cache_find_or_add(&cache, key);
        if (!cat) {
            if (proto_root) nanousd_freeprim(proto_root);
            nanousd_freeprim(inst_prim);
            continue;
        }

        if (!cat->built) {
            int narcs = nanousd_ncomposition_arcs(inst_prim);
            int has_direct_payload = 0;
            for (int a = 0; a < narcs; a++) {
                NanousdCompositionArc arc;
                memset(&arc, 0, sizeof(arc));
                if (!nanousd_composition_arc(inst_prim, a, &arc)) continue;
                if (arc.arc_type != NANOUSD_ARC_PAYLOAD) continue;
                if ((arc.flags & NANOUSD_COMPOSITION_ARC_DIRECT) == 0) continue;
                if (!arc.layer_path || !arc.layer_path[0]) continue;
                if (nb_skip_arc_for_geometry_only(arc.layer_path, arc.source_path))
                    continue;
                has_direct_payload = 1;
                break;
            }
            for (int a = 0; a < narcs; a++) {
                NanousdCompositionArc arc;
                memset(&arc, 0, sizeof(arc));
                if (!nanousd_composition_arc(inst_prim, a, &arc)) continue;
                if (arc.arc_type != NANOUSD_ARC_REFERENCE &&
                    arc.arc_type != NANOUSD_ARC_PAYLOAD) continue;
                if (has_direct_payload && arc.arc_type != NANOUSD_ARC_PAYLOAD)
                    continue;
                if ((arc.flags & NANOUSD_COMPOSITION_ARC_DIRECT) == 0) continue;
                if (!arc.layer_path || !arc.layer_path[0]) continue;
                if (nb_skip_arc_for_geometry_only(arc.layer_path, arc.source_path))
                    continue;
                NanousdStage sub = nanousd_open(arc.layer_path);
                if (!sub || !nanousd_isvalid(sub)) {
                    if (sub) nanousd_close(sub);
                    continue;
                }
                const char* target =
                    (arc.source_path && arc.source_path[0]) ? arc.source_path : NULL;
                nb_apply_instance_variants_to_asset_target(inst_prim, sub, target);
                double root_rel[16];
                nb_identity_m4d(root_rel);
                nb_seed_curve_asset_stage_recursive(
                    scene, inst_prim, sub, target, inst_path, inst_path,
                    root_rel, cat, arena, 0);
                nanousd_close(sub);
            }
            cat->built = 1;
            catalogs_built++;
            catalog_entries += cat->count;
            skipped_total += cat->skipped_by_policy;
        }

        double inst_world[16];
        compute_world_xform(inst_prim, inst_world);
        int emitted = nb_emit_curve_replay(scene, &replayed, inst_path,
                                           inst_world, cat);
        if (emitted > 0) {
            roots++;
            emitted_total += emitted;
        }

        if (proto_root) nanousd_freeprim(proto_root);
        nanousd_freeprim(inst_prim);
    }

    if (replayed.count > 0) {
        int old = scene->ncurves;
        SceneCurve* combined = (SceneCurve*)arena_calloc(
            arena, (size_t)(old + replayed.count), sizeof(SceneCurve));
        if (combined) {
            if (old > 0 && scene->curves)
                memcpy(combined, scene->curves, (size_t)old * sizeof(SceneCurve));
            memcpy(combined + old, replayed.items,
                   (size_t)replayed.count * sizeof(SceneCurve));
            scene->curves = combined;
            scene->ncurves = old + replayed.count;
        }
    }

    if (emitted_total > 0 || skipped_total > 0 || getenv("NUSD_NATIVE_CURVES_DIAG")) {
        fprintf(stderr,
                "scene_load: native instance curves — catalogs=%d entries=%d "
                "roots=%d emitted=%d skipped_by_policy=%d mode=%s\n",
                catalogs_built, catalog_entries, roots, emitted_total,
                skipped_total, nb_native_curves_all() ? "all" : "no-needles");
    }

    free(replayed.items);
    nb_curve_cache_free(&cache);
    return emitted_total;
}

/* Returns >0 if it seeded/replayed the instance (caller skips the normal walk),
 * 0 to fall back to the composed-child walk. */
static int nb_seed_native_instance(Scene* scene, NanousdStage main_stage,
                                   NanousdPrim inst_prim, NanousdPrim proto_root,
                                   const char* inst_root_path, const double inst_world[16],
                                   const double proto_root_inv[16], int have_proto_root_inv,
                                   NbReplayCache* cache, Arena* arena,
                                   int* mesh_idx, int max_meshes,
                                   uint64_t* total_verts, uint64_t* total_indices,
                                   struct NestedPiRec** nested, int* nested_n, int* nested_cap,
                                   uint64_t* pi_xform_cap, int* pi_batch_cap,
                                   int emit_native_batches) {
    char key[768];
    nb_instance_key(inst_prim, proto_root, key, sizeof(key));
    NbReplayCatalog* cat = nb_cache_find_or_add(cache, key);
    if (!cat) return 0;
    if (cat->aborted) return 0;  /* contains nested PI: the walk handles it */
    if (cat->built)
        return nb_emit_replay(scene, mesh_idx, max_meshes, inst_root_path,
                              inst_world, cat, arena, total_verts, total_indices,
                              nested, nested_n, nested_cap, pi_xform_cap,
                              pi_batch_cap, emit_native_batches);

    NbAssetStages stages = {0, 0, 0};
    int seeded = 0;
    int pis_recorded = 0;
    int narcs = nanousd_ncomposition_arcs(inst_prim);
    int has_direct_payload = 0;
    for (int a = 0; a < narcs; a++) {
        NanousdCompositionArc arc;
        memset(&arc, 0, sizeof(arc));
        if (!nanousd_composition_arc(inst_prim, a, &arc)) continue;
        if (arc.arc_type != NANOUSD_ARC_PAYLOAD) continue;
        if ((arc.flags & NANOUSD_COMPOSITION_ARC_DIRECT) == 0) continue;
        if (!arc.layer_path || !arc.layer_path[0]) continue;
        if (nb_skip_arc_for_geometry_only(arc.layer_path, arc.source_path))
            continue;
        has_direct_payload = 1;
        break;
    }
    for (int a = 0; a < narcs; a++) {
        NanousdCompositionArc arc;
        memset(&arc, 0, sizeof(arc));
        if (!nanousd_composition_arc(inst_prim, a, &arc)) continue;
        if (arc.arc_type != NANOUSD_ARC_REFERENCE && arc.arc_type != NANOUSD_ARC_PAYLOAD) continue;
        if (has_direct_payload && arc.arc_type != NANOUSD_ARC_PAYLOAD) continue;
        if ((arc.flags & NANOUSD_COMPOSITION_ARC_DIRECT) == 0) continue;
        if (!arc.layer_path || !arc.layer_path[0]) continue;
        if (nb_skip_arc_for_geometry_only(arc.layer_path, arc.source_path))
            continue;
        NanousdStage sub = nanousd_open(arc.layer_path);
        if (!sub || !nanousd_isvalid(sub)) { if (sub) nanousd_close(sub); continue; }
        if (!nb_stages_add(&stages, sub)) { nanousd_close(sub); continue; }
        nb_apply_instance_variants_to_asset_target(inst_prim, sub, arc.source_path);

        NanousdPrim target = (arc.source_path && arc.source_path[0])
                                 ? nanousd_primpath(sub, arc.source_path) : NULL;
        if (!target) target = nanousd_defaultprim(sub);
        if (!target) continue;
        const char* tpr = nanousd_path(target);
        char tpath[1024]; tpath[0] = '\0';
        if (tpr) snprintf(tpath, sizeof(tpath), "%s", tpr);
        double tworld[16], tinv[16];
        compute_world_xform(target, tworld);
        int have_tinv = invert_affine_m4d_rowvec(tworld, tinv);
        nanousd_freeprim(target);
        if (!have_tinv || !tpath[0]) continue;
        size_t tlen = strlen(tpath);

        {
            double root_rel[16];
            nb_identity_m4d(root_rel);
            int rec = nb_seed_asset_stage_recursive(
                scene, main_stage, inst_prim, sub,
                arc.source_path && arc.source_path[0] ? arc.source_path : tpath,
                inst_root_path, inst_world, inst_root_path, root_rel,
                cat, arena, mesh_idx, max_meshes, total_verts, total_indices,
                nested, nested_n, nested_cap, pi_xform_cap, pi_batch_cap,
                emit_native_batches, 0);
            if (rec > 0) {
                seeded += rec;
                continue;
            }
        }

        int nflat = nanousd_traverse_flat(sub, NULL, 0);
        if (getenv("NUSD_NB_ARC_DIAG"))
            fprintf(stderr, "  NB arc type=%d layer='%s' src='%s' -> tpath='%s' nflat=%d\n",
                    arc.arc_type, arc.layer_path,
                    arc.source_path ? arc.source_path : "", tpath, nflat);
        if (nflat <= 0) continue;
        NanousdFlatPrim* flat = (NanousdFlatPrim*)calloc((size_t)nflat, sizeof(NanousdFlatPrim));
        if (!flat) continue;
        int total = nanousd_traverse_flat(sub, flat, nflat);
        if (total > 0 && total < nflat) nflat = total;

        /* Asset paths of PIs found under the target (depth-ordered flat, so a
         * PI precedes its subtree). Meshes under a PI are its prototypes —
         * the drain instances them, so they must NOT be seeded standalone. */
        char pi_pref[32][512];
        int  npi_pref = 0;

        for (int i = 0; i < nflat && *mesh_idx < max_meshes; i++) {
            const char* fp = flat[i].path;
            const char* tn = flat[i].type_name;
            if (!fp || !tn) continue;
            if (!nb_path_has_prefix(fp, tpath)) continue;
            if (strcmp(tn, "PointInstancer") == 0) {
                /* Nested PI: record its MAIN-stage path + pi_to_inst for the
                 * existing drain (which decodes PIs cheaply via array reads, not
                 * the composed mesh walk). Re-get the composed PI from the main
                 * stage so pi_to_inst matches the walk's exactly. No abort -> the
                 * mesh geometry still comes from the flat asset path, so the
                 * cliff is avoided while the PI instances are preserved. */
                const char* prel = fp + tlen;
                while (*prel == '/') prel++;
                char mainpath[1024];
                if (inst_root_path && inst_root_path[0])
                    snprintf(mainpath, sizeof(mainpath), "%s/%s", inst_root_path, prel);
                else
                    snprintf(mainpath, sizeof(mainpath), "%s", fp);
                int top_level_owned = scene_compact_pi_batches_active() &&
                                      nb_toplevel_pi_paths_contains(mainpath);
                NanousdPrim pim = top_level_owned ? NULL :
                    nanousd_primpath(main_stage, mainpath);
                if (top_level_owned) {
                    seeded++;
                } else if (pim && nested && nested_n && nested_cap &&
                           nanousd_attribarraylen(pim, "protoIndices") > 0) {
                    double pw[16], rel_pi[16], pti[16];
                    compute_world_xform(pim, pw);
                    if (have_proto_root_inv) nanousd_mul_m4d(pw, proto_root_inv, rel_pi);
                    else memcpy(rel_pi, pw, sizeof(pw));
                    nanousd_mul_m4d(rel_pi, inst_world, pti);
                    if (*nested_n >= *nested_cap) {
                        int ncp = *nested_cap ? *nested_cap * 2 : 16;
                        struct NestedPiRec* g = (struct NestedPiRec*)realloc(
                            *nested, (size_t)ncp * sizeof(**nested));
                        if (g) { *nested = g; *nested_cap = ncp; }
                    }
                    if (*nested && *nested_n < *nested_cap) {
                        snprintf((*nested)[*nested_n].path, sizeof((*nested)[0].path), "%s", mainpath);
                        memcpy((*nested)[*nested_n].pi_to_inst, pti, sizeof(pti));
                        (*nested_n)++;
                    }
                    /* Store proto-local PI xform so sibling instances of this
                     * prototype re-record their own PIs during replay. */
                    nb_catalog_add_pi(cat, prel, rel_pi);
                    pis_recorded++;
                }
                if (pim) nanousd_freeprim(pim);
                if (npi_pref < 32) {
                    snprintf(pi_pref[npi_pref], sizeof(pi_pref[0]), "%s", fp);
                    npi_pref++;
                }
                continue;  /* PI handed to the drain; keep seeding meshes */
            }
            if (strcmp(tn, "Mesh") != 0) continue;
            /* Skip PI-prototype meshes (under a recorded PI) — the drain
             * instances them; seeding them standalone would double-draw. */
            int under_pi = 0;
            for (int z = 0; z < npi_pref; z++)
                if (nb_path_has_prefix(fp, pi_pref[z])) { under_pi = 1; break; }
            if (under_pi) continue;
            const char* relpath = fp + tlen;
            while (*relpath == '/') relpath++;
            if (nb_catalog_has_mesh_path(cat, relpath)) continue;
            NanousdPrim ap = nanousd_primpath(sub, fp);
            if (!ap) continue;
            SceneMesh* m = &scene->meshes[*mesh_idx];
            memset(m, 0, sizeof(*m));
            if (!nb_load_asset_mesh(ap, arena, m)) { nanousd_freeprim(ap); continue; }
            double cw[16], rel[16], world[16];
            compute_world_xform(ap, cw);
            nanousd_freeprim(ap);
            nanousd_mul_m4d(cw, tinv, rel);
            nanousd_mul_m4d(rel, inst_world, world);
            char pathbuf[1024];
            if (inst_root_path && inst_root_path[0])
                snprintf(pathbuf, sizeof(pathbuf), "%s/%s", inst_root_path, relpath);
            else
                snprintf(pathbuf, sizeof(pathbuf), "%s", fp);
            m->path = scene_arena_strdup(arena, pathbuf);
            m->prototype_idx = *mesh_idx;  /* self = prototype source */
            if (scene_compact_pi_batches_active()) {
                nb_identity_m4d(m->world_xform);
                m->is_proto_only = 1;
            } else {
                memcpy(m->world_xform, world, sizeof(world));
                m->is_proto_only = 0;
            }
            m->lazy_prim_idx = -1;
            mesh_compute_local_bounds(m);
            if (scene_compact_pi_batches_active()) {
                if (emit_native_batches) {
                    scene_add_compact_instance_batch(
                        scene, pi_xform_cap, pi_batch_cap, *mesh_idx, m, world,
                        SCENE_INSTANCE_SOURCE_NATIVE_INSTANCE);
                }
            } else {
                mesh_compute_world_bounds_from_local(m, scene);
            }
            m->vertex_offset = 0; m->index_offset = 0;
            nb_catalog_add(cat, relpath, *mesh_idx, rel);
            if (total_verts) *total_verts += m->nvertices;
            if (total_indices) *total_indices += m->nindices;
            (*mesh_idx)++; seeded++;
        }
        free(flat);
    }
    if (getenv("NUSD_NB_ARC_DIAG"))
        fprintf(stderr, "  NB seed result key='%s' narcs=%d seeded=%d pis=%d\n",
                key, narcs, seeded, pis_recorded);
    nb_stages_close(&stages);
    if (seeded > 0 || pis_recorded > 0) cat->built = 1;
    return seeded + pis_recorded;
}

/* ===================================================================
 * Storm-style compact native instancing (NUSD_COMPACT_NATIVE_INSTANCES).
 *
 * The legacy second pass "expands" each native USD instance by walking its
 * composed child subtree via nanousd_child() and emitting one SceneMesh per
 * instance child. That descent forces nanousd to materialize every instance's
 * composed proxy subtree, which is O(instances x prototype-size) in time AND
 * memory — on DSX (10,429 instances of mesh-heavy prototypes) it blows up to
 * tens of GB and never finishes.
 *
 * Hydra/Storm never does this: it hydrates each PROTOTYPE once and records each
 * instance as a transform. This routine mirrors that: it enumerates the stage
 * prototypes (nanousd_stage_prototype — cheap, no instance-proxy descent),
 * extracts each prototype's meshes ONCE into scene->meshes (is_proto_only), then
 * for each instance reads only nanousd_prototype(inst) + its world transform and
 * appends one compact SceneInstanceBatch per (prototype-mesh x instance). No
 * per-instance geometry, no SceneMesh clones, no nanousd_child on instances.
 *
 * Phase 1 scope: Mesh prototype geometry (points/indices/normals via
 * nb_load_asset_mesh; face-varying UVs/per-instance material overrides are not
 * yet carried — instances inherit the prototype binding). Returns the number of
 * compact instance-mesh placements emitted. */
typedef struct { char path[1024]; int start; int count; int compactable; } CompactProtoRange;

/* Returns the number of FALLBACK instances left in instance_prim_indices (those
 * whose prototype is not compactable — contains nested PointInstancers, implicit
 * shapes, or no Mesh geometry); those are compacted in place to the front and
 * *n_instance_prims_io is updated so the caller runs the legacy expand-walk on
 * exactly them (no geometry loss). 0 = everything compacted. */
static long scene_emit_compact_native_instances(
        void* stage_void, Scene* scene, Arena* arena,
        int* instance_prim_indices, int* n_instance_prims_io,
        int* mesh_idx_io, int max_meshes,
        uint64_t* total_verts, uint64_t* total_indices,
        uint64_t* xform_cap, int* batch_cap, int do_timing)
{
    int n_instance_prims = *n_instance_prims_io;
    NanousdStage stage = (NanousdStage)stage_void;
    double t0 = do_timing ? get_time_sec() : 0.0;
    int nproto = nanousd_stage_nprototypes(stage);
    if (nproto <= 0) return 0;

    /* Part-(A) below DFS-walks every prototype subtree; once it completes it has
     * authoritatively seen every typename, so the native-instance-curve gate can
     * trust has_basis_curves. */
    scene->checked_proto_for_curves = 1;

    CompactProtoRange* ranges =
        (CompactProtoRange*)calloc((size_t)nproto, sizeof(CompactProtoRange));
    if (!ranges) return 0;

    int mesh_idx = *mesh_idx_io;
    long proto_meshes = 0;

    /* (A) Extract each prototype's meshes ONCE (prototype subtree DFS). */
    for (int pi = 0; pi < nproto; pi++) {
        NanousdPrim proto = nanousd_stage_prototype(stage, pi);
        if (!proto) continue;
        const char* pp = nanousd_path(proto);
        snprintf(ranges[pi].path, sizeof(ranges[pi].path), "%s", pp ? pp : "");
        double proto_root_world[16], proto_root_inv[16];
        compute_world_xform(proto, proto_root_world);
        int have_inv = invert_affine_m4d_rowvec(proto_root_world, proto_root_inv);
        ranges[pi].start = mesh_idx;
        /* B1/B2: a prototype is only compactable if its subtree is Mesh-only.
         * Nested PointInstancers, implicit shapes, or skel need the legacy
         * expand-walk (this Mesh-only DFS would silently drop them). */
        int non_compact = 0;

        int cap = 256, sz = 0;
        NanousdPrim* stack = (NanousdPrim*)malloc((size_t)cap * sizeof(NanousdPrim));
        int nc = stack ? nanousd_nchildren(proto) : 0;
        for (int c = nc - 1; c >= 0; c--) {
            NanousdPrim ch = nanousd_child(proto, c);
            if (!ch) continue;
            if (sz >= cap) {
                cap *= 2;
                NanousdPrim* g = (NanousdPrim*)realloc(stack, (size_t)cap * sizeof(NanousdPrim));
                if (!g) { nanousd_freeprim(ch); break; }
                stack = g;
            }
            stack[sz++] = ch;
        }
        while (sz > 0 && mesh_idx < max_meshes) {
            NanousdPrim node = stack[--sz];
            if (!node) continue;
            int ncc = nanousd_nchildren(node);
            for (int c = ncc - 1; c >= 0; c--) {
                NanousdPrim gc = nanousd_child(node, c);
                if (!gc) continue;
                if (sz >= cap) {
                    cap *= 2;
                    NanousdPrim* g = (NanousdPrim*)realloc(stack, (size_t)cap * sizeof(NanousdPrim));
                    if (!g) { nanousd_freeprim(gc); break; }
                    stack = g;
                }
                stack[sz++] = gc;
            }
            const char* tn = nanousd_typename(node);
            if (tn && (!strcmp(tn, "PointInstancer") || !strcmp(tn, "Sphere") ||
                       !strcmp(tn, "Cylinder") || !strcmp(tn, "Capsule") ||
                       !strcmp(tn, "Cube") || !strcmp(tn, "Cone"))) {
                non_compact = 1;   /* prototype not Mesh-only → legacy walk */
            }
            /* Record curve presence for the native-instance-curve gate. This DFS
             * already visits every prototype node, and a proxy is a typename-copy
             * of its prototype, so seeing BasisCurves here is equivalent to (and
             * far cheaper than) the proxy-expanding flat traversal in
             * scene_load_native_instance_curves. */
            if (tn && !strcmp(tn, "BasisCurves")) scene->has_basis_curves = 1;
            if (!tn || strcmp(tn, "Mesh") != 0) { nanousd_freeprim(node); continue; }

            SceneMesh* m = &scene->meshes[mesh_idx];
            memset(m, 0, sizeof(*m));
            if (!nb_load_asset_mesh(node, arena, m)) { nanousd_freeprim(node); continue; }
            const char* mp = nanousd_path(node);
            m->path = mp ? scene_arena_strdup(arena, mp) : NULL;
            m->is_proto_only = 1;
            m->prototype_idx = mesh_idx;            /* self-prototype */
            m->lazy_prim_idx = -1;
            m->material_index = scene->materials
                ? materials_find_binding((MaterialCollection*)scene->materials, stage, node)
                : -1;
            /* Store the PROTOTYPE-RELATIVE transform; instances compose it. */
            double cw[16];
            compute_world_xform(node, cw);
            if (have_inv) nanousd_mul_m4d(cw, proto_root_inv, m->world_xform);
            else memcpy(m->world_xform, cw, sizeof(cw));
            mesh_compute_local_bounds(m);
            *total_verts += (uint64_t)m->nvertices;
            *total_indices += (uint64_t)m->nindices;
            mesh_idx++;
            proto_meshes++;
            nanousd_freeprim(node);
        }
        while (sz > 0) { NanousdPrim n = stack[--sz]; if (n) nanousd_freeprim(n); }
        free(stack);
        if (non_compact) {
            /* Discard the partial Mesh-only extraction; the legacy walk will
             * re-extract the WHOLE prototype (meshes + implicit shapes + nested
             * PIs) for this prototype's instances. */
            proto_meshes -= (long)(mesh_idx - ranges[pi].start);
            mesh_idx = ranges[pi].start;
            ranges[pi].count = 0;
            ranges[pi].compactable = 0;
        } else {
            ranges[pi].count = mesh_idx - ranges[pi].start;
            ranges[pi].compactable = (ranges[pi].count > 0);
        }
        nanousd_freeprim(proto);
    }

    /* (B) For each instance: compact if its prototype is compactable, else keep
     * it (compacted to the front of instance_prim_indices) for the legacy walk. */
    long emitted = 0;
    int fallback_n = 0;
    for (int ii = 0; ii < n_instance_prims; ii++) {
        int prim_index = instance_prim_indices[ii];
        NanousdPrim inst = nanousd_prim(stage, prim_index);
        if (!inst) { instance_prim_indices[fallback_n++] = prim_index; continue; }
        NanousdPrim proto = nanousd_prototype(inst);
        char ppath[1024]; ppath[0] = '\0';
        if (proto) {
            const char* pr = nanousd_path(proto);
            snprintf(ppath, sizeof(ppath), "%s", pr ? pr : "");
            nanousd_freeprim(proto);
        }
        int ri = -1;
        for (int k = 0; k < nproto; k++) {
            if (ppath[0] && strcmp(ranges[k].path, ppath) == 0) { ri = k; break; }
        }
        if (ri < 0 || !ranges[ri].compactable) {
            /* Non-compactable / unknown prototype → legacy walk handles it. */
            instance_prim_indices[fallback_n++] = prim_index;
            nanousd_freeprim(inst);
            continue;
        }

        double inst_world[16];
        compute_world_xform(inst, inst_world);
        for (int m = ranges[ri].start; m < ranges[ri].start + ranges[ri].count; m++) {
            double baked[16];
            nanousd_mul_m4d(scene->meshes[m].world_xform, inst_world, baked);
            if (scene_add_compact_instance_batch(
                    scene, xform_cap, batch_cap, m, &scene->meshes[m], baked,
                    SCENE_INSTANCE_SOURCE_NATIVE_INSTANCE))
                emitted++;
        }
        nanousd_freeprim(inst);
    }

    *mesh_idx_io = mesh_idx;
    *n_instance_prims_io = fallback_n;
    free(ranges);
    fprintf(stderr,
            "scene_load: compact native instancing — %d prototypes, %ld unique "
            "proto meshes, %ld instance-mesh placements; %d instances fell back "
            "to the legacy walk (non-Mesh prototypes)\n",
            nproto, proto_meshes, emitted, fallback_n);
    if (do_timing)
        fprintf(stderr, "  [scene_timing] compact native instancing: %.1f ms\n",
                (get_time_sec() - t0) * 1000.0);
    return fallback_n;
}

/* Position A — pre-decode dedup key from a prim's composition source.
 *
 * Two prims that reference the SAME asset share the (layer_path, source_path) of
 * their Reference/Payload/Variant arcs and differ only in target_path (their
 * placement) — verified on DSX's Revit `Instances/` refs. Hashing those source
 * records yields a key that is identical for content-duplicate references
 * BEFORE any geometry is read, so duplicates can be routed straight to compact
 * batches without decoding points/indices. Variant arcs are included so the same
 * asset under different FamilyType selections does not collide. Returns 0 when
 * the prim has no external source (→ treat as unique, never dedup). */
static uint64_t flat_predecode_key(NanousdPrim prim) {
    int na = nanousd_ncomposition_arcs(prim);
    if (na <= 0) return 0;
    uint64_t h = 0xcbf29ce484222325ULL;
    int used = 0;
    for (int a = 0; a < na; a++) {
        NanousdCompositionArc arc;
        memset(&arc, 0, sizeof(arc));
        arc.struct_size = (int)sizeof(arc);
        if (!nanousd_composition_arc(prim, a, &arc)) continue;
        /* Reference(1) / Payload(2) / Variant(6) — the arcs that pull in shared
         * content. Local/root/inherits/relocate carry the per-duplicate composed
         * path and would break the key, so they are excluded. */
        if (arc.arc_type != 1 && arc.arc_type != 2 && arc.arc_type != 6) continue;
        int at = arc.arc_type;
        h = geo_fnv1a64_bytes(h, &at, sizeof(at));
        if (arc.layer_path)
            h = geo_fnv1a64_bytes(h, arc.layer_path, strlen(arc.layer_path));
        if (arc.source_path)
            h = geo_fnv1a64_bytes(h, arc.source_path, strlen(arc.source_path));
        used++;
    }
    return used ? (h ? h : 1) : 0;
}

Scene* scene_load_from_stage_filtered(void* stage_void,
                                      const char* stage_label,
                                      const unsigned char* wanted_prims,
                                      int nprims_in_bitmap)
{
    NanousdStage stage = (NanousdStage)stage_void;
    if (!nanousd_isvalid(stage)) {
        const char* err = nanousd_error(stage);
        fprintf(stderr, "scene_load_from_stage: invalid stage '%s': %s\n",
                stage_label ? stage_label : "<no label>",
                err ? err : "unknown error");
        return NULL;
    }
    const char* filepath = stage_label ? stage_label : "<stage>";
    double t0 = get_time_sec();
    int do_timing = (getenv("NUSD_LOAD_TIMING") != NULL);
    double t_mark = t0;

    int nprims = nanousd_nprims(stage);

    /* Tier 3 step 4 frustum-cull filter diagnostic. Quietly counts the
     * "wanted" prims so the load log can report cull rate without forcing
     * a second pass through the bitmap on hot paths. */
    if (wanted_prims && nprims_in_bitmap > 0) {
        int wanted_count = 0;
        int upto = nprims_in_bitmap < nprims ? nprims_in_bitmap : nprims;
        for (int i = 0; i < upto; i++) if (wanted_prims[i]) wanted_count++;
        fprintf(stderr,
                "[scene_load] filter active: %d/%d prims wanted "
                "(%.1f%% cull rate)\n",
                wanted_count, upto,
                100.0 * (1.0 - (double)wanted_count / (double)(upto ? upto : 1)));
    }
    if (do_timing) {
        double t_now = get_time_sec();
        fprintf(stderr, "  [scene_timing] nanousd_nprims: %.1f ms (nprims=%d)\n",
            (t_now - t_mark) * 1000.0, nprims);
        t_mark = t_now;
    }

    /* --- Diagnostic: log prim type histogram --- */
    if (getenv("NUSD_SCENE_DIAG")) {
        int n_mesh=0, n_xform=0, n_scope=0, n_pi=0, n_inst=0, n_inactive=0, n_other=0, n_payload_host=0;
        int n_rect=0, n_sphere=0, n_disk=0, n_cyl=0, n_dist=0, n_dome=0;
        for (int di = 0; di < nprims; di++) {
            NanousdPrim p = nanousd_prim(stage, di);
            if (!p) continue;
            if (!nanousd_isactive(p)) { n_inactive++; nanousd_freeprim(p); continue; }
            if (nanousd_isinstance(p)) n_inst++;
            const char* tn = nanousd_typename(p);
            if (!tn) { n_other++; nanousd_freeprim(p); continue; }
            if      (!strcmp(tn, "Mesh"))            n_mesh++;
            else if (!strcmp(tn, "Xform"))           n_xform++;
            else if (!strcmp(tn, "Scope"))           n_scope++;
            else if (!strcmp(tn, "PointInstancer"))  n_pi++;
            else if (!strcmp(tn, "RectLight"))       { n_rect++; n_other++; }
            else if (!strcmp(tn, "SphereLight"))     { n_sphere++; n_other++; }
            else if (!strcmp(tn, "DiskLight"))       { n_disk++; n_other++; }
            else if (!strcmp(tn, "CylinderLight"))   { n_cyl++; n_other++; }
            else if (!strcmp(tn, "DistantLight"))    { n_dist++; n_other++; }
            else if (!strcmp(tn, "DomeLight"))       { n_dome++; n_other++; }
            else                                     n_other++;
            nanousd_freeprim(p);
        }
        fprintf(stderr, "[scene_diag] nprims=%d mesh=%d xform=%d scope=%d pi=%d inst=%d inactive=%d other=%d\n",
                nprims, n_mesh, n_xform, n_scope, n_pi, n_inst, n_inactive, n_other);
        fprintf(stderr, "[scene_diag] lights: rect=%d sphere=%d disk=%d cyl=%d dist=%d dome=%d\n",
                n_rect, n_sphere, n_disk, n_cyl, n_dist, n_dome);
        (void)n_payload_host;
    }

    /* Allocate Scene and Arena. The stage is BORROWED by default —
     * scene_load() flips _owns_stage=1 after we return so it gets closed
     * in scene_free; scene_load_from_stage callers manage stage lifetime
     * themselves. We never close `stage` from inside this function. */
    Scene* scene = (Scene*)calloc(1, sizeof(Scene));
    if (!scene) return NULL;

    Arena* arena = (Arena*)malloc(sizeof(Arena));
    if (!arena) { free(scene); return NULL; }
    /* Scale arena to scene complexity: ~4KB per prim is a reasonable heuristic
     * for mesh-heavy scenes (positions + normals + indices + colors).
     * Minimum 64MB, so small scenes don't under-allocate. */
    size_t arena_size = (size_t)nprims * 4096;
    if (arena_size < 64 * 1024 * 1024) arena_size = 64 * 1024 * 1024;
    *arena = arena_create(arena_size);
    if (!arena->head) { free(arena); free(scene); return NULL; }

    scene->_arena = arena;
    scene->_stage = stage;
    scene->_owns_stage = 0;  /* default: borrowed; scene_load() flips this. */

    XformCache xform_cache = { NULL, 0, 0 };
    g_xform_cache = &xform_cache;

    PathPrefixList invisible_prefixes = { NULL, 0, 0 };
    PathPrefixList purpose_hidden_prefixes = { NULL, 0, 0 };
    PathPrefixList point_instancer_prefixes = { NULL, 0, 0 };
    int prefix_prescan = getenv("NUSD_SCENE_PREFIX_PRESCAN") != NULL;
    if (prefix_prescan)
        collect_scene_prefixes(stage, nprims, &invisible_prefixes, &purpose_hidden_prefixes, &point_instancer_prefixes);

    int lazy_mesh = 0;
    {
        const char* e = getenv("NUSD_LAZY_MESH");
        if (e && e[0] && e[0] != '0') lazy_mesh = 1;
    }

    /* --- Load materials from the stage --- */
    if (s_load_materials && !lazy_mesh) {
        /* Compute scene directory for resolving relative texture paths */
        char dir_buf[1024];
        snprintf(dir_buf, sizeof(dir_buf), "%s", filepath);
        char* scene_dir = dirname(dir_buf);
        scene->materials = (void*)materials_load_filtered(
            stage, scene_dir, wanted_prims, nprims_in_bitmap);
    } else if (do_timing) {
        fprintf(stderr, "  [scene_timing] materials_load: %s\n",
                lazy_mesh ? "deferred for NUSD_LAZY_MESH" : "disabled");
    }
    if (do_timing) {
        double t_now = get_time_sec();
        fprintf(stderr, "  [scene_timing] materials_load: %.1f ms\n",
            (t_now - t_mark) * 1000.0);
        t_mark = t_now;
    }

    /* Over-allocate mesh array. With instancing, the total mesh count can
     * exceed nprims. Small synthetic PointInstancer stress scenes keep the
     * old 16x headroom; huge composed scenes such as DSX would otherwise spend
     * time and memory zeroing millions of unused SceneMesh slots. Override with
     * NUSD_SCENE_MESH_FACTOR when testing pathological expansion cases. */
    int mesh_factor = (nprims > 100000) ? 4 : 16;
    if (scene_compact_pi_batches_active() && mesh_factor < 32)
        mesh_factor = 32;
    const char* mesh_factor_env = getenv("NUSD_SCENE_MESH_FACTOR");
    if (mesh_factor_env && mesh_factor_env[0]) {
        long parsed = strtol(mesh_factor_env, NULL, 10);
        if (parsed >= 1 && parsed <= 64) mesh_factor = (int)parsed;
    }
    int max_meshes = nprims * mesh_factor;
    if (scene_compact_pi_batches_active() && max_meshes < 4096) {
        /* Instance-proxy element stages can report only a tiny composed prim
         * count while Storm-visible XGen prototype payloads contribute hundreds
         * of meshes. Keep a modest default floor so Storm geometry parity does
         * not require NUSD_MAX_SCENE_MESHES on those element roots. */
        max_meshes = 4096;
    }
    const char* max_meshes_env = getenv("NUSD_MAX_SCENE_MESHES");
    if (max_meshes_env && max_meshes_env[0]) {
        char* end = NULL;
        long parsed = strtol(max_meshes_env, &end, 10);
        if (end != max_meshes_env && parsed > 0 && parsed <= INT32_MAX)
            max_meshes = (int)parsed;
    }
    scene->meshes = (SceneMesh*)arena_calloc(arena, (size_t)max_meshes, sizeof(SceneMesh));
    if (!scene->meshes) {
        fprintf(stderr, "scene_load: failed to allocate %d SceneMesh records\n", max_meshes);
        arena_destroy(arena);
        free(arena);
        free(scene);
        return NULL;
    }

    /* Initialize bounds */
    scene->bounds_min[0] = FLT_MAX;
    scene->bounds_min[1] = FLT_MAX;
    scene->bounds_min[2] = FLT_MAX;
    scene->bounds_max[0] = -FLT_MAX;
    scene->bounds_max[1] = -FLT_MAX;
    scene->bounds_max[2] = -FLT_MAX;
    scene->up_axis = 1;

    /* NUSD_LAZY_MESH=1 short-circuits the per-prim attribute extraction
     * passes that dominate load_usd (~58 s of the 102 s DSX load — see
     * docs/plans/TIER_3_LAZY_MESH.md).
     *
     * Step 2 scope (this code): walk the flat prim list and emit
     * METADATA-ONLY SceneMesh records for top-level Mesh / Sphere /
     * Cylinder prims. positions / normals / indices stay NULL;
     * lazy_prim_idx records the prim's index in the stage's flat
     * list so step 3's not-yet-written extract-deferred worker can
     * re-resolve and pull attributes on demand. Instance-prim
     * expansion (the 26 s second pass) and PointInstancer expansion
     * stay non-lazy for now — they require richer state (proto cache,
     * child-walk recursion) and land in step 2.5+.
     *
     * Structural verification only — pixel-level visual gating moves
     * to step 3 once the renderer can actually upload deferred meshes. */
    /* Declared up-front so the lazy-path scene_load diagnostic at
     * `lazy_postprocess` (line ~3510) reads initialised values when
     * `goto lazy_postprocess` skips the eager declarations below. */
    int mesh_idx = 0;
    uint64_t total_verts = 0;
    uint64_t total_indices = 0;
    int instanced_meshes = 0;
    int stage_has_basis_curves = 0;

    /* Tier 3 step 4 follow-up: optional topological world-xform pre-walk.
     * Measured net-negative on DSX (+10.6 s prewalk overhead for −0.5 s
     * savings in the flat mesh pass) because compute_world_xform's
     * path-keyed cache was already hitting on most parent chains.
     * Default OFF. NUSD_LAZY_XFORM_PREWALK=1 opts in for experimentation
     * — kept for scenes with cold-cache xform patterns where it might
     * still win, or for future O(N) topo-sort refinement. */
    double* world_prewalk = NULL;
    int prewalk_orphans = 0;
    int do_prewalk = 0;
    {
        const char* e = getenv("NUSD_LAZY_XFORM_PREWALK");
        if (e && e[0] && e[0] != '0') do_prewalk = 1;
    }
    /* Eager-pass xform prewalk (NUSD_EAGER_XFORM_PREWALK): replace the per-mesh
     * recursive compute_world_xform (measured 39s on DSX — the global path-keyed
     * cache still pays nanousd_path + nanousd_parent heap-alloc + hash per mesh)
     * with one dense O(nprims) topological walk, then an O(1) array read per
     * mesh. FV-agnostic; also removes the global-cache serialization barrier for
     * later parallelization. Default OFF until golden-verified. */
    int eager_prewalk = scene_env_flag("NUSD_EAGER_XFORM_PREWALK");
    if ((lazy_mesh && do_prewalk) || (!lazy_mesh && eager_prewalk)) {
        double t_pw0 = get_time_sec();
        world_prewalk = prewalk_world_xforms(stage, nprims, &prewalk_orphans);
        if (do_timing) {
            fprintf(stderr,
                    "  [scene_timing] xform prewalk: %.1f ms "
                    "(nprims=%d, orphans=%d)\n",
                    (get_time_sec() - t_pw0) * 1000.0,
                    nprims, prewalk_orphans);
        }
    }

    if (lazy_mesh) {
        /* Tier 3 step 4: compute a local AABB per Mesh prim by walking
         * `points` (zero-copy nanousd_arraydataf). Local→world is then
         * folded through the 8-corner transform. Implicit Sphere/Cylinder
         * fall back to "never cull" sentinel bounds (FLT_MAX) so they're
         * always extracted — proper AABBs for implicits cost extra and
         * land in 4.5 alongside instance/PI lazy expansion.
         *
         * **Default is ON** thanks to the `extent` attribute fast path:
         * authored bounds (~all USD-pipeline assets, all of DSX's 107k
         * flat-list Mesh prims) cost two array slots instead of an
         * N×3-float walk. Full-DSX overhead is ~1.6 s + 0.1 GB peak
         * vs step-2 metadata-only (was 3.6 s + 4.9 GB before the fast
         * path). Set NUSD_LAZY_AABB=0 to disable AABBs entirely —
         * extract_deferred_visible silently falls back to extract-all
         * when no AABBs were snapshotted. */
        int lazy_aabb = 1;
        {
            const char* e = getenv("NUSD_LAZY_AABB");
            if (e && e[0] == '0') lazy_aabb = 0;
        }

        /* Tier 3 step 4.5: lazy enumeration of instance-prim children
         * for AABB cull. Costs ~27 s on full DSX (3,025 instances ×
         * ~40 children each × compute_world_xform). Roughly the same
         * order as the eager second pass it would later make cullable,
         * so it only pays off when the camera frustum culls a large
         * fraction of envs. **Default OFF.** Set
         * NUSD_LAZY_INSTANCE_AABB=1 alongside NUSD_LAZY_AABB to opt in. */
        int lazy_inst_aabb = 0;
        {
            const char* e = getenv("NUSD_LAZY_INSTANCE_AABB");
            if (e && e[0] && e[0] != '0') lazy_inst_aabb = 1;
        }
        int lazy_vertex_counts = 0;
        {
            const char* e = getenv("NUSD_LAZY_VERTEX_COUNTS");
            if (e && e[0] && e[0] != '0') lazy_vertex_counts = 1;
        }

        fprintf(stderr,
                "[scene_load] NUSD_LAZY_MESH=1 — metadata-only walk of flat prim list "
                "(Tier 3 step %d; aabb=%d; see docs/plans/TIER_3_LAZY_MESH.md).\n",
                lazy_aabb ? 4 : 2, lazy_aabb);

        int lazy_idx = 0;
        int lazy_n_mesh = 0, lazy_n_sphere = 0, lazy_n_cyl = 0;
        int lazy_n_aabb_walked = 0, lazy_n_aabb_skipped = 0;
        int lazy_n_inst = 0, lazy_n_inst_children = 0;
        for (int i = 0; i < nprims && lazy_idx < max_meshes; i++) {
            NanousdPrim p = nanousd_prim(stage, i);
            if (!p) continue;
            if (!nanousd_isactive(p)) { nanousd_freeprim(p); continue; }
            const char* tn = nanousd_typename(p);
            if (!tn) { nanousd_freeprim(p); continue; }
            if (!strcmp(tn, "BasisCurves")) {
                stage_has_basis_curves = 1;
                nanousd_freeprim(p);
                continue;
            }
            int is_mesh   = !strcmp(tn, "Mesh");
            int is_sphere = !strcmp(tn, "Sphere");
            int is_cyl    = !strcmp(tn, "Cylinder");
            int is_capsule = !strcmp(tn, "Capsule");
            int is_cube = !strcmp(tn, "Cube");
            int is_cone = !strcmp(tn, "Cone");

            /* Tier 3 step 4.5: instance prims aren't Mesh/Sphere/Cyl
             * themselves but their CHILDREN are. The eager second pass
             * walks each instance's descendants and emits one
             * scene_mesh per matched child. For cull purposes we only
             * need the UNION world AABB across all matched children to
             * decide whether the instance is visible — if yes, the
             * eager pass expands it fully later. */
            if (lazy_aabb && lazy_inst_aabb && nanousd_isinstance(p)) {
                /* Walk children with a depth-1 expansion stack mirroring
                 * the eager second pass at scene.c:~2580. */
                float u_min[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
                float u_max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
                int   union_valid = 0;
                int   matched_children = 0;

                NanousdPrim stack_arr[256];
                int stack_size = 0;
                int nc = nanousd_nchildren(p);
                for (int c = nc - 1; c >= 0 && stack_size < 256; c--) {
                    NanousdPrim child = nanousd_child(p, c);
                    if (child) stack_arr[stack_size++] = child;
                }
                while (stack_size > 0) {
                    NanousdPrim child = stack_arr[--stack_size];
                    if (!child) continue;
                    int ncc = nanousd_nchildren(child);
                    for (int c = ncc - 1; c >= 0 && stack_size < 256; c--) {
                        NanousdPrim gc = nanousd_child(child, c);
                        if (gc) stack_arr[stack_size++] = gc;
                    }
                    const char* ctn = nanousd_typename(child);
                    int cmesh   = ctn && !strcmp(ctn, "Mesh");
                    int csphere = ctn && !strcmp(ctn, "Sphere");
                    int ccyl    = ctn && !strcmp(ctn, "Cylinder");
                    if (!cmesh && !csphere && !ccyl) {
                        nanousd_freeprim(child);
                        continue;
                    }
                    matched_children++;

                    /* Local AABB from extent (cheap; fallback to nothing
                     * — we'd rather mark the instance never-cull than
                     * pay the points-walk RSS spike across all 103k
                     * children). */
                    float lmn[3], lmx[3];
                    int have_local = 0;
                    if (cmesh) {
                        int ec = 0;
                        const float* ext = nanousd_arraydataf(child, "extent", &ec);
                        if (ext && ec == 6) {
                            lmn[0]=ext[0]; lmn[1]=ext[1]; lmn[2]=ext[2];
                            lmx[0]=ext[3]; lmx[1]=ext[4]; lmx[2]=ext[5];
                            have_local = 1;
                        }
                    }
                    if (!have_local) {
                        /* Implicit shapes or extent-less Meshes: treat
                         * as never-cull by widening the union to
                         * sentinel. One missing entry forces the whole
                         * instance to always-visible (the cull test
                         * exits early). */
                        u_min[0] = u_min[1] = u_min[2] = -FLT_MAX;
                        u_max[0] = u_max[1] = u_max[2] =  FLT_MAX;
                        union_valid = 1;
                        nanousd_freeprim(child);
                        continue;
                    }

                    /* Transform 8 local corners through the child's
                     * world_xform and accumulate. */
                    double cw[16];
                    compute_world_xform(child, cw);
                    /* compute_world_xform writes a double[16]; we need
                     * float-corners-through-double-matrix. xform_point
                     * already handles the type punning. */
                    SceneMesh tmp;
                    memcpy(tmp.world_xform, cw, sizeof(cw));
                    memcpy(tmp.local_bounds_min, lmn, sizeof(lmn));
                    memcpy(tmp.local_bounds_max, lmx, sizeof(lmx));
                    tmp.nvertices = 0;
                    tmp.positions = NULL;
                    mesh_compute_world_bounds_from_local(&tmp, NULL);
                    for (int k = 0; k < 3; k++) {
                        if (tmp.bounds_min[k] < u_min[k]) u_min[k] = tmp.bounds_min[k];
                        if (tmp.bounds_max[k] > u_max[k]) u_max[k] = tmp.bounds_max[k];
                    }
                    union_valid = 1;
                    nanousd_freeprim(child);
                }

                if (union_valid && lazy_idx < max_meshes) {
                    SceneMesh* m = &scene->meshes[lazy_idx];
                    memset(m, 0, sizeof(*m));
                    /* World xform stored on the instance itself; final
                     * meshes get their own per-child xforms during
                     * eager expansion. The lazy entry is purely a cull
                     * record. */
                    compute_world_xform(p, m->world_xform);
                    memcpy(m->bounds_min, u_min, sizeof(u_min));
                    memcpy(m->bounds_max, u_max, sizeof(u_max));
                    /* Local bounds left at sentinel — they aren't read
                     * for cull (only world bounds_min/max are). */
                    m->local_bounds_min[0]=m->local_bounds_min[1]=m->local_bounds_min[2]=-FLT_MAX;
                    m->local_bounds_max[0]=m->local_bounds_max[1]=m->local_bounds_max[2]= FLT_MAX;
                    m->display_color[0] = m->display_color[1] = m->display_color[2] = 0.7f;
                    m->material_index = -1;
                    m->prototype_idx = lazy_idx;
                    m->lazy_prim_idx = i;
                    lazy_idx++;
                    /* Also expand scene bounds. */
                    scene_expand_bounds(scene, u_min, u_max);
                }
                lazy_n_inst++;
                lazy_n_inst_children += matched_children;
                nanousd_freeprim(p);
                continue;
            }

            if (!is_mesh && !is_sphere && !is_cyl && !is_capsule && !is_cube && !is_cone) {
                nanousd_freeprim(p);
                continue;
            }

            SceneMesh* m = &scene->meshes[lazy_idx];
            m->positions = NULL;
            m->normals = NULL;
            m->colors = NULL;
            m->texcoords = NULL;
            m->indices = NULL;
            m->nvertices = 0;
            m->nindices = 0;   /* filled at extract time */
            if (world_prewalk) {
                memcpy(m->world_xform,
                       &world_prewalk[(size_t)i * 16],
                       sizeof(double) * 16);
            } else {
                compute_world_xform(p, m->world_xform);
            }

            /* Default: never-cull sentinel bounds. */
            m->local_bounds_min[0] = m->local_bounds_min[1] = m->local_bounds_min[2] = -FLT_MAX;
            m->local_bounds_max[0] = m->local_bounds_max[1] = m->local_bounds_max[2] =  FLT_MAX;
            m->bounds_min[0]       = m->bounds_min[1]       = m->bounds_min[2]       = -FLT_MAX;
            m->bounds_max[0]       = m->bounds_max[1]       = m->bounds_max[2]       =  FLT_MAX;

            if (is_mesh) {
                if (lazy_aabb) {
                    /* Prefer the authored `extent` attribute (2 vec3s = 6 floats).
                     * USD assets from any DCC pipeline have this baked in, and
                     * reading it is two array slots vs walking N×3 floats. On
                     * full DSX this drops the AABB walk from ~3.6 s to <0.5 s
                     * and avoids forcing nanousd to materialise per-prim
                     * points data (~4.9 GB peak RSS spike). */
                    int ext_count = 0;
                    const float* ext = nanousd_arraydataf(p, "extent", &ext_count);
                    if (ext && ext_count == 6) {
                        m->local_bounds_min[0] = ext[0];
                        m->local_bounds_min[1] = ext[1];
                        m->local_bounds_min[2] = ext[2];
                        m->local_bounds_max[0] = ext[3];
                        m->local_bounds_max[1] = ext[4];
                        m->local_bounds_max[2] = ext[5];
                        mesh_compute_world_bounds_from_local(m, scene);
                        if (lazy_vertex_counts)
                            m->nvertices = nanousd_attribarraylen(p, "points") / 3;
                        lazy_n_aabb_walked++;
                    } else {
                        /* Fallback: walk points zero-copy. Last resort because
                         * it triggers nanousd's per-prim points materialisation. */
                        int pts_count = 0;
                        const float* pts = nanousd_arraydataf(p, "points", &pts_count);
                        if (pts && pts_count >= 3) {
                            m->nvertices = pts_count / 3;
                            float lmin[3] = { pts[0], pts[1], pts[2] };
                            float lmax[3] = { pts[0], pts[1], pts[2] };
                            int nv = pts_count / 3;
                            for (int v = 1; v < nv; v++) {
                                const float* q = &pts[v * 3];
                                for (int k = 0; k < 3; k++) {
                                    if (q[k] < lmin[k]) lmin[k] = q[k];
                                    if (q[k] > lmax[k]) lmax[k] = q[k];
                                }
                            }
                            memcpy(m->local_bounds_min, lmin, sizeof(lmin));
                            memcpy(m->local_bounds_max, lmax, sizeof(lmax));
                            mesh_compute_world_bounds_from_local(m, scene);
                            lazy_n_aabb_walked++;
                        } else {
                            if (lazy_vertex_counts)
                                m->nvertices = nanousd_attribarraylen(p, "points") / 3;
                            lazy_n_aabb_skipped++;
                        }
                    }
                } else {
                    if (lazy_vertex_counts)
                        m->nvertices = nanousd_attribarraylen(p, "points") / 3;
                    lazy_n_aabb_skipped++;
                }
            }
            m->has_display_color = 0;
            m->display_color[0] = 0.7f;
            m->display_color[1] = 0.7f;
            m->display_color[2] = 0.7f;
            m->material_index = -1;
            m->vertex_offset = 0;
            m->index_offset = 0;
            m->prototype_idx = lazy_idx;   /* self */
            m->is_proto_only = 0;
            m->lazy_prim_idx = i;

            if (is_mesh)        lazy_n_mesh++;
            else if (is_sphere) lazy_n_sphere++;
            else if (is_cyl)    lazy_n_cyl++;
            total_verts += (uint64_t)m->nvertices;
            lazy_idx++;
            nanousd_freeprim(p);
        }
        scene->nmeshes = lazy_idx;
        mesh_idx = lazy_idx;

        if (do_timing) {
            double t_now = get_time_sec();
            fprintf(stderr,
                    "  [scene_timing] flat mesh/implicit pass: %.1f ms "
                    "(LAZY metadata: %d meshes — %d Mesh, %d Sphere, %d Cylinder; "
                    "aabb walked=%d skipped=%d; "
                    "%d instance prims w/ %d matched children -> union AABBs)\n",
                    (t_now - t_mark) * 1000.0,
                    lazy_idx, lazy_n_mesh, lazy_n_sphere, lazy_n_cyl,
                    lazy_n_aabb_walked, lazy_n_aabb_skipped,
                    lazy_n_inst, lazy_n_inst_children);
            t_mark = t_now;
        }
        goto lazy_postprocess;
    }

    /* --- Extract mesh data ---
     * (mesh_idx / total_verts / total_indices / instanced_meshes are now
     * declared above the lazy_mesh check so the lazy_postprocess fallthrough
     * sees zeroed values instead of indeterminate stack noise.) */

    /* Prototype mesh lookup: maps prim path → mesh_idx for instancing */
    ProtoMeshHash proto_mesh_table = { NULL, 0, 0 };

    /* Early geometry dedup (NUSD_EARLY_GEO_DEDUP): extract each distinct raw
     * geometry ONCE; later byte-identical duplicates pointer-share the
     * prototype's triangulated buffers and pay only per-instance world-xform +
     * material binding, skipping the dominant triangulate/smooth-normal/FV
     * compute. Safe for non-instanceable (Revit) geometry because the key is a
     * content hash — baked-per-instance geometry differs and is not merged.
     * Default OFF until golden-verified. */
    int geo_dedup_on = scene_env_flag("NUSD_EARLY_GEO_DEDUP");
    GeoDedupHash geo_dedup_table = { NULL, NULL, 0, 0 };
    long geo_dedup_hits = 0;

    /* Position C — compact flat-mesh dedup (NUSD_COMPACT_FLAT_DEDUP, implies
     * NUSD_EARLY_GEO_DEDUP). The Revit "Instances/" subtree is mostly the SAME
     * asset referenced many times but NOT flagged instanceable, so it lands here
     * as ~104K flat meshes. Instead of pointer-sharing each duplicate into its
     * own SceneMesh (still one fat row + draw per copy), route duplicates into
     * the SAME compact SceneInstanceBatch path used for native instances: the
     * first occurrence becomes an is_proto_only prototype and every copy
     * (including the first) is recorded as a 48-byte transform. Mirrors Storm:
     * one prototype, many instance transforms. */
    int compact_flat_dedup = scene_compact_flat_dedup_active();
    if (compact_flat_dedup) geo_dedup_on = 1;   /* dedup hashing is the prerequisite */
    if (geo_dedup_on) geo_dedup_init(&geo_dedup_table, nprims);
    uint64_t flat_dedup_xcap = scene->npi_transforms;
    int      flat_dedup_bcap = scene->npi_batches;
    long     flat_dedup_compacted = 0;

    /* Position A — pre-decode (composition-source) dedup. Keys each flat mesh on
     * its reference/payload/variant source (flat_predecode_key) BEFORE reading
     * geometry: a duplicate is emitted as a compact batch with ZERO geometry
     * read (vs NUSD_COMPACT_FLAT_DEDUP which reads + content-hashes first). A
     * cheap vertex-count check guards against same-source/different-variant
     * collisions; on mismatch the prim falls through to full decode. */
    int predecode_dedup = scene_predecode_dedup_active();
    GeoDedupHash predecode_table = { NULL, NULL, 0, 0 };
    if (predecode_dedup) geo_dedup_init(&predecode_table, nprims);
    long predecode_hits = 0, predecode_compacted = 0, predecode_sig_miss = 0;

    /* Parallel deferred smooth-normals (NUSD_PARALLEL_NORMALS): move the
     * fallback positions+indices->normals computation off the serial flat-pass
     * critical path into a race-free parallel post-pass. Default OFF until
     * golden-verified. */
    int parallel_normals = scene_env_flag("NUSD_PARALLEL_NORMALS");

    /* Deferred-compute parallelization (NUSD_DEFER_COMPUTE): for "simple" meshes
     * (plain Mesh, fvc-triangulated, no GeomSubset children, no face-varying
     * normals/UVs, no skinning, dedup-miss) keep ALL racy nanousd reads + bind +
     * world-xform + bounds SERIAL in this loop, but defer the race-free
     * triangulation + smooth-normals into a parallel per-thread-arena post-pass
     * (probe showed ~7.8x scaling). Complex meshes stay fully serial (bit-
     * identical by construction). Default OFF until golden-verified. */
    int defer_compute = scene_env_flag("NUSD_DEFER_COMPUTE");
    typedef struct { int mesh_slot; const int* fvi; int fvi_count; const int* fvc; int fvc_count; } DeferredTriWork;
    DeferredTriWork* dwork = NULL;
    int n_deferred = 0;
    if (defer_compute)
        dwork = (DeferredTriWork*)arena_alloc(arena, (size_t)max_meshes * sizeof(DeferredTriWork), 16);
    if (!dwork) defer_compute = 0;  /* alloc fail -> serial path */
    /* Per-clause eligibility rejection counters (non-short-circuited) to
     * diagnose why meshes are/aren't deferred. Printed at the flat-pass timing. */
    long dbg_et = 0, dbg_ok = 0, dbg_rj_notmesh = 0, dbg_rj_proto = 0,
         dbg_rj_fvnorm = 0, dbg_rj_nofvc = 0, dbg_rj_children = 0,
         dbg_rj_uv = 0, dbg_rj_skel = 0;
    long dbg_uv_none = 0, dbg_uv_vert = 0, dbg_uv_fv = 0;  /* UV interp split */
    int implicit_sphere_proto = -1;
    int implicit_cylinder_proto[3] = { -1, -1, -1 };

    /* Collect instance prims for the second pass */
    int n_instance_prims = 0, inst_cap = 0;
    int* instance_prim_indices = NULL;

    /* Tier 3 step 4: filter-aware prim-index test. Returns 1 when the prim
     * at flat-list index `i` should be processed; 0 when it should be
     * skipped (frustum-culled by the caller).
     *
     *   wanted_prims == NULL || nprims_in_bitmap <= 0  → all prims wanted
     *   i out of bitmap range                          → defensively wanted
     *   wanted_prims[i] != 0                           → wanted
     *
     * Mesh AABBs are not enough to safely filter structural prims. Instance
     * roots and PointInstancers perform expansion work for descendant/prototype
     * geometry, so keep those controllers even if they do not have their own
     * wanted bit. Culling their children without an HLOD replacement is a hole.
     */
    #define PRIM_IS_WANTED(I)                                                  \
        (!wanted_prims || nprims_in_bitmap <= 0 ||                             \
         (I) < 0 || (I) >= nprims_in_bitmap || wanted_prims[(I)] != 0)
    #define PRIM_IS_FILTER_CONTROLLER(P)                                       \
        (nanousd_isinstance((P)) || prim_is_point_instancer((P)))
    uint64_t filtered_skips = 0;

    /* M2 load-time profiling: split the flat-pass per-mesh cost into nanousd
     * attribute reads (LZ4 decode — the thread-unsafe "racy" side) vs material
     * binding vs world-xform vs the race-free compute+copy remainder. Always
     * accumulated (clock_gettime is ~free vs the 228s pass); printed under
     * do_timing. Decides the parallelization ceiling (decode- vs compute-bound). */
    double prof_read_s = 0.0, prof_bind_s = 0.0, prof_xform_s = 0.0;
    double prof_pass_t0 = get_time_sec();

    /* First pass: extract meshes from flat prim list + collect instance prims */
    for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;

        int prim_is_inst = nanousd_isinstance(prim);
        int prim_is_active = nanousd_isactive(prim);
        const char* tn = prim_is_active ? nanousd_typename(prim) : NULL;
        int prim_is_pi = tn && !strcmp(tn, "PointInstancer");
        if (tn && !strcmp(tn, "BasisCurves"))
            stage_has_basis_curves = 1;

        if (!PRIM_IS_WANTED(i) && !prim_is_inst && !prim_is_pi) {
            filtered_skips++;
            nanousd_freeprim(prim);
            continue;
        }

        /* Collect instance prims for second pass + record proto→inst mapping */
        if (prim_is_inst) {
            if (n_instance_prims >= inst_cap) {
                inst_cap = inst_cap ? inst_cap * 2 : 64;
                instance_prim_indices = (int*)realloc(instance_prim_indices, (size_t)inst_cap * sizeof(int));
            }
            instance_prim_indices[n_instance_prims++] = i;

        }

        if (!prim_is_active || !tn) {
            nanousd_freeprim(prim);
            continue;
        }

        const char* prim_path_raw = nanousd_path(prim);
        char prim_path_buf[8192];
        const char* prim_path = prim_path_raw;
        if (prim_path_raw) {
            snprintf(prim_path_buf, sizeof(prim_path_buf), "%s", prim_path_raw);
            prim_path = prim_path_buf;
        }
        if (!prefix_prescan) {
            if (prim_is_pi)
                path_prefix_list_add(&point_instancer_prefixes, prim_path);

            int ok = 0;
            const char* vis = nanousd_attrib_token(prim, "visibility", &ok);
            if (ok && vis && !strcmp(vis, "invisible"))
                path_prefix_list_add(&invisible_prefixes, prim_path);

            ok = 0;
            const char* purpose = nanousd_attrib_token(prim, "purpose", &ok);
            if (ok && purpose && (!strcmp(purpose, "guide") || !strcmp(purpose, "proxy")))
                path_prefix_list_add(&purpose_hidden_prefixes, prim_path);
        }

        int is_mesh = !strcmp(tn, "Mesh");
        int is_sphere = !strcmp(tn, "Sphere");
        int is_cylinder = !strcmp(tn, "Cylinder");
        int is_capsule = !strcmp(tn, "Capsule");
        int is_cube = !strcmp(tn, "Cube");
        int is_cone = !strcmp(tn, "Cone");
        if (!is_mesh && !is_sphere && !is_cylinder && !is_capsule && !is_cube && !is_cone) {
            nanousd_freeprim(prim);
            continue;
        }

        if (prim_hidden_from_render_cached(prim, &invisible_prefixes, &purpose_hidden_prefixes)) {
            nanousd_freeprim(prim);
            continue;
        }

        SceneMesh* mesh = &scene->meshes[mesh_idx];
        mesh->path = scene_arena_strdup(arena, prim_path);
        uint64_t pending_geo_hash = 0;  /* early-geo-dedup: set on a miss, recorded at registration */
        uint64_t pending_predecode_key = 0;  /* Position A: set on a pre-decode miss */

        /* USD semantics: meshes under PointInstancer prototypes, and meshes
         * under native USD /__Prototype_N instance-prototype roots, are
         * materialized only through instance expansion. Keep the geometry in
         * scene->meshes for sharing, but do not draw it standalone. */
        mesh->is_proto_only =
            prim_under_point_instancer_cached(prim, &point_instancer_prefixes) ||
            path_is_native_usd_prototype(prim_path);

        /* ---- Position A: pre-decode (composition-source) dedup ----
         * Before reading ANY geometry, key this mesh on its reference source.
         * A duplicate of an already-seen source is emitted as a compact batch
         * (transform only) with zero geometry read; the prototype is converted
         * to is_proto_only on first hit and its own placement recorded too. A
         * cheap vertex-count check guards same-source/different-variant. */
        if (predecode_dedup && is_mesh && !mesh->is_proto_only) {
            uint64_t pk = flat_predecode_key(prim);
            if (pk) {
                int proto = geo_dedup_find(&predecode_table, pk);
                if (proto >= 0 && proto < mesh_idx) {
                    SceneMesh* pm = &scene->meshes[proto];
                    /* attribarraylen for a point3f[] returns the ELEMENT count
                     * (vertices), not the float count — compare directly. */
                    int dup_nverts = nanousd_attribarraylen(prim, "points");
                    if (pm->nvertices > 0 && dup_nverts == pm->nvertices) {
                        double dup_world[16];
                        if (world_prewalk)
                            memcpy(dup_world, &world_prewalk[(size_t)i * 16], 16 * sizeof(double));
                        else
                            compute_world_xform(prim, dup_world);
                        if (!pm->is_proto_only) {
                            pm->is_proto_only = 1;
                            scene_add_compact_instance_batch(
                                scene, &flat_dedup_xcap, &flat_dedup_bcap, proto, pm,
                                pm->world_xform, SCENE_INSTANCE_SOURCE_NATIVE_INSTANCE);
                            predecode_compacted++;
                        }
                        scene_add_compact_instance_batch(
                            scene, &flat_dedup_xcap, &flat_dedup_bcap, proto, pm,
                            dup_world, SCENE_INSTANCE_SOURCE_NATIVE_INSTANCE);
                        predecode_compacted++;
                        predecode_hits++;
                        nanousd_freeprim(prim);
                        continue;            /* skipped the geometry read entirely */
                    }
                    predecode_sig_miss++;    /* same source, different geometry → full decode */
                } else {
                    pending_predecode_key = pk;  /* first occurrence → record at registration */
                }
            }
        }

        if (is_sphere || is_cylinder || is_capsule || is_cube || is_cone) {
            double shape_xform[16];
            int proto_idx = -1;
            int ok = 0;
            if (is_sphere) {
                implicit_sphere_xform(prim, shape_xform);
                proto_idx = implicit_sphere_proto;
                ok = (proto_idx >= 0)
                    ? load_sphere_instance(prim, arena, mesh,
                                           &scene->meshes[proto_idx],
                                           shape_xform)
                    : load_sphere(prim, arena, mesh, shape_xform);
            } else if (is_cylinder) {
                int axis = 2;
                implicit_cylinder_xform(prim, &axis, shape_xform);
                if (axis < 0 || axis > 2) axis = 2;
                proto_idx = implicit_cylinder_proto[axis];
                ok = (proto_idx >= 0)
                    ? load_cylinder_instance(prim, arena, mesh,
                                             &scene->meshes[proto_idx],
                                             shape_xform)
                    : load_cylinder(prim, arena, mesh, axis, shape_xform);
                if (ok && proto_idx < 0)
                    implicit_cylinder_proto[axis] = mesh_idx;
            } else if (is_capsule) {
                /* Capsule: true-scale tessellation, no proto sharing (radius/
                 * height vary per prim, e.g. each Newton robot limb). */
                int axis = 2;
                double radius = 0.5, height = 1.0;
                implicit_capsule_params(prim, &axis, &radius, &height);
                if (axis < 0 || axis > 2) axis = 2;
                ok = load_capsule(prim, arena, mesh, axis, radius, height);
            } else if (is_cube) {
                implicit_cube_xform(prim, shape_xform);
                ok = load_cube(prim, arena, mesh, shape_xform);
            } else {
                /* Cone: radius/height/axis like cylinder; the apex stays a point
                 * under non-uniform scale, so unit + shape_xform is fine. */
                int axis = 2;
                implicit_cylinder_xform(prim, &axis, shape_xform);
                if (axis < 0 || axis > 2) axis = 2;
                ok = load_cone(prim, arena, mesh, axis, shape_xform);
            }
            if (ok) {
                if (proto_idx >= 0) {
                    mesh->prototype_idx = proto_idx;
                } else {
                    mesh->prototype_idx = mesh_idx;
                    if (is_sphere) implicit_sphere_proto = mesh_idx;
                }
                if (prim_path) proto_hash_insert(&proto_mesh_table, prim_path, mesh_idx);
                scene_expand_bounds(scene, mesh->bounds_min, mesh->bounds_max);
                total_verts += mesh->nvertices;
                total_indices += mesh->nindices;
                mesh_idx++;
            }
            nanousd_freeprim(prim);
            continue;
        }

        /* ---- (a) Points ---- */
        int pts_count = 0;
        double _pr0 = get_time_sec();
        const float* pts_data = nanousd_arraydataf(prim, "points", &pts_count);
        prof_read_s += get_time_sec() - _pr0;

        if (pts_data && pts_count > 0) {
            mesh->nvertices = pts_count / 3;
            /* Copy into arena — arraydataf cache may be overwritten by later calls */
            mesh->positions = (float*)arena_alloc(arena, (size_t)pts_count * sizeof(float), 16);
            if (mesh->positions) {
                memcpy(mesh->positions, pts_data, (size_t)pts_count * sizeof(float));
            }
        } else {
            /* Fallback: copy into arena */
            int len = nanousd_attribarraylen(prim, "points");
            if (len > 0) {
                mesh->nvertices = len / 3;
                mesh->positions = (float*)arena_alloc(arena, (size_t)len * sizeof(float), 16);
                if (mesh->positions) {
                    nanousd_attribarrayf(prim, "points", mesh->positions, len);
                }
            }
        }

        if (mesh->nvertices == 0 || !mesh->positions) {
            nanousd_freeprim(prim);
            continue;
        }

        /* ---- (b) Normals ---- */
        /* Try "normals" first, then "primvars:normals" (glTF-origin assets) */
        int normals_count = 0;
        double _pr1 = get_time_sec();
        const float* normals_data = nanousd_arraydataf(prim, "normals", &normals_count);
        prof_read_s += get_time_sec() - _pr1;
        const float* authored_normals_data = normals_data;
        int authored_normals_count = normals_count;
        float* pending_fv_normals = NULL;
        int pending_fv_normals_count = 0;

        if ((!normals_data || normals_count / 3 != mesh->nvertices)) {
            int pv_normals_count = 0;
            double _pr2 = get_time_sec();
            const float* pv_normals_data =
                nanousd_arraydataf(prim, "primvars:normals", &pv_normals_count);
            prof_read_s += get_time_sec() - _pr2;
            if (pv_normals_data && pv_normals_count / 3 == mesh->nvertices) {
                normals_data = pv_normals_data;
                normals_count = pv_normals_count;
            } else if (!authored_normals_data) {
                normals_data = pv_normals_data;
                normals_count = pv_normals_count;
                authored_normals_data = pv_normals_data;
                authored_normals_count = pv_normals_count;
            }
        }

        if (normals_data && normals_count > 0 && normals_count / 3 == mesh->nvertices) {
            mesh->normals = (float*)arena_alloc(arena, (size_t)normals_count * sizeof(float), 16);
            if (mesh->normals)
                memcpy(mesh->normals, normals_data, (size_t)normals_count * sizeof(float));
        } else {
            if (authored_normals_data && authored_normals_count > 0) {
                pending_fv_normals =
                    (float*)arena_alloc(arena, (size_t)authored_normals_count * sizeof(float), 16);
                if (pending_fv_normals) {
                    memcpy(pending_fv_normals, authored_normals_data,
                           (size_t)authored_normals_count * sizeof(float));
                    pending_fv_normals_count = authored_normals_count;
                }
            }
            /* Try fallback copy */
            int len = nanousd_attribarraylen(prim, "normals");
            if (len <= 0 || len / 3 != mesh->nvertices)
                len = nanousd_attribarraylen(prim, "primvars:normals");
            if (len > 0 && len / 3 == mesh->nvertices) {
                mesh->normals = (float*)arena_alloc(arena, (size_t)len * sizeof(float), 16);
                if (mesh->normals) {
                    if (nanousd_attribarrayf(prim, "normals", mesh->normals, len) <= 0)
                        nanousd_attribarrayf(prim, "primvars:normals", mesh->normals, len);
                }
            } else {
                if (!pending_fv_normals && len > 0) {
                    pending_fv_normals =
                        (float*)arena_alloc(arena, (size_t)len * sizeof(float), 16);
                    if (pending_fv_normals) {
                        if (nanousd_attribarrayf(prim, "normals", pending_fv_normals, len) <= 0)
                            nanousd_attribarrayf(prim, "primvars:normals", pending_fv_normals, len);
                        pending_fv_normals_count = len;
                    }
                }
                /* No normals available — leave NULL, viewer will handle */
                mesh->normals = NULL;
            }
        }

        /* ---- (c) Face vertex indices ---- */
        int fvi_count = 0;
        const int* fvi_data = NULL;
        int* fvi_fallback = NULL; /* only allocated if zero-copy unavailable */
        {
            double _pr3 = get_time_sec();
            fvi_data = nanousd_arraydatai(prim, "faceVertexIndices", &fvi_count);
            prof_read_s += get_time_sec() - _pr3;
            if (!fvi_data || fvi_count <= 0) {
                int len = nanousd_attribarraylen(prim, "faceVertexIndices");
                if (len > 0) {
                    fvi_count = len;
                    fvi_fallback = (int*)arena_alloc(arena, (size_t)len * sizeof(int), 16);
                    if (fvi_fallback) nanousd_attribarrayi(prim, "faceVertexIndices", fvi_fallback, len);
                    fvi_data = fvi_fallback;
                }
            }
        }

        if (!fvi_data || fvi_count <= 0) {
            nanousd_freeprim(prim);
            continue;
        }

        /* ---- (d) Face vertex counts ---- */
        int fvc_count = 0;
        const int* fvc_data = NULL;
        int* fvc_fallback = NULL;
        {
            double _pr4 = get_time_sec();
            fvc_data = nanousd_arraydatai(prim, "faceVertexCounts", &fvc_count);
            prof_read_s += get_time_sec() - _pr4;
            if (!fvc_data || fvc_count <= 0) {
                int len = nanousd_attribarraylen(prim, "faceVertexCounts");
                if (len > 0) {
                    fvc_count = len;
                    fvc_fallback = (int*)arena_alloc(arena, (size_t)len * sizeof(int), 16);
                    if (fvc_fallback) nanousd_attribarrayi(prim, "faceVertexCounts", fvc_fallback, len);
                    fvc_data = fvc_fallback;
                }
            }
        }

        /* ---- Early geometry dedup (NUSD_EARLY_GEO_DEDUP) ----
         * Content-hash the raw geometry (positions + raw indices/counts +
         * authored normals). A byte-identical match means a shared referenced
         * asset: pointer-share the prototype's already-triangulated buffers and
         * skip the dominant triangulate/smooth-normal/FV-expand compute, paying
         * only the per-instance world-xform + material binding. Eligible only
         * for plain Mesh prims with no pending face-varying normals and that are
         * not point-instancer/native prototypes (handled by their own paths). */
        if (geo_dedup_on && is_mesh && !pending_fv_normals && !mesh->is_proto_only
            && mesh->positions && mesh->nvertices > 0 && fvi_data && fvi_count > 0) {
            uint64_t gh = 0xcbf29ce484222325ULL;
            gh = geo_fnv1a64_bytes(gh, &mesh->nvertices, sizeof(mesh->nvertices));
            gh = geo_fnv1a64_bytes(gh, mesh->positions,
                                   (size_t)mesh->nvertices * 3 * sizeof(float));
            gh = geo_fnv1a64_bytes(gh, &fvi_count, sizeof(fvi_count));
            gh = geo_fnv1a64_bytes(gh, fvi_data, (size_t)fvi_count * sizeof(int));
            gh = geo_fnv1a64_bytes(gh, &fvc_count, sizeof(fvc_count));
            if (fvc_data && fvc_count > 0)
                gh = geo_fnv1a64_bytes(gh, fvc_data, (size_t)fvc_count * sizeof(int));
            if (mesh->normals)
                gh = geo_fnv1a64_bytes(gh, mesh->normals,
                                       (size_t)mesh->nvertices * 3 * sizeof(float));

            int proto = geo_dedup_find(&geo_dedup_table, gh);
            if (proto >= 0 && proto < mesh_idx && compact_flat_dedup) {
                /* Position C: emit a compact instance batch for this duplicate
                 * instead of a pointer-shared SceneMesh. On the FIRST hit,
                 * convert the prototype to is_proto_only and record its own
                 * placement so it isn't drawn twice. No mesh_idx++ — the
                 * duplicate never becomes a SceneMesh row. */
                SceneMesh* pm = &scene->meshes[proto];
                /* B6 guard: the compact batch carries NO per-instance material/
                 * UV/displayColor (material_or_binding_id=-1 is never read; the
                 * GPU uses the prototype's). Only fuse when this duplicate's
                 * appearance matches the prototype's; otherwise fall through to a
                 * full unique decode so shared-geometry/different-material scenes
                 * (e.g. chess pieces) keep distinct materials. */
                int dup_mat = scene->materials
                    ? materials_find_binding((MaterialCollection*)scene->materials,
                                             stage, prim)
                    : -1;
                int dup_has_uv = (nanousd_attribarraylen(prim, "primvars:st") > 0 ||
                                  nanousd_attribarraylen(prim, "st") > 0);
                int dup_has_dc = (nanousd_attribarraylen(prim, "primvars:displayColor") > 0);
                int appearance_ok =
                    dup_mat == pm->material_index &&
                    dup_has_uv == (pm->texcoords != NULL) &&
                    dup_has_dc == (pm->colors != NULL || pm->has_display_color);
                if (appearance_ok) {
                    double dup_world[16];
                    if (world_prewalk)
                        memcpy(dup_world, &world_prewalk[(size_t)i * 16], 16 * sizeof(double));
                    else
                        compute_world_xform(prim, dup_world);
                    if (!pm->is_proto_only) {
                        pm->is_proto_only = 1;
                        scene_add_compact_instance_batch(
                            scene, &flat_dedup_xcap, &flat_dedup_bcap, proto, pm,
                            pm->world_xform, SCENE_INSTANCE_SOURCE_NATIVE_INSTANCE);
                        flat_dedup_compacted++;
                    }
                    scene_add_compact_instance_batch(
                        scene, &flat_dedup_xcap, &flat_dedup_bcap, proto, pm,
                        dup_world, SCENE_INSTANCE_SOURCE_NATIVE_INSTANCE);
                    flat_dedup_compacted++;
                    geo_dedup_hits++;
                    nanousd_freeprim(prim);
                    continue;
                }
                /* appearance diverged → full unique decode (skip pointer-share) */
            }
            if (proto >= 0 && proto < mesh_idx && !compact_flat_dedup) {
                SceneMesh* pm = &scene->meshes[proto];
                /* Pointer-share the prototype's triangulated geometry (object
                 * space). Per-instance world_xform + binding applied below. */
                mesh->nvertices = pm->nvertices;
                mesh->nindices  = pm->nindices;
                mesh->positions = pm->positions;
                mesh->normals   = pm->normals;
                mesh->texcoords = pm->texcoords;
                mesh->colors    = pm->colors;
                mesh->indices   = pm->indices;
                mesh->has_display_color = pm->has_display_color;
                mesh->display_color[0] = pm->display_color[0];
                mesh->display_color[1] = pm->display_color[1];
                mesh->display_color[2] = pm->display_color[2];
                mesh->prototype_idx = proto;

                double _pbind0 = get_time_sec();
                mesh->material_index = scene->materials
                    ? materials_find_binding((MaterialCollection*)scene->materials,
                                             stage, prim)
                    : -1;
                prof_bind_s += get_time_sec() - _pbind0;
                double _pxform0 = get_time_sec();
                if (world_prewalk)
                    memcpy(mesh->world_xform, &world_prewalk[(size_t)i * 16], 16 * sizeof(double));
                else
                    compute_world_xform(prim, mesh->world_xform);
                prof_xform_s += get_time_sec() - _pxform0;
                mesh_compute_local_bounds(mesh);
                mesh_compute_world_bounds_from_local(mesh, scene);
                if (prim_path)
                    proto_hash_insert(&proto_mesh_table, prim_path, mesh_idx);
                total_verts += mesh->nvertices;
                total_indices += mesh->nindices;
                geo_dedup_hits++;
                mesh_idx++;
                nanousd_freeprim(prim);
                continue;
            }
            /* Miss: this prim becomes a prototype. Record its hash at
             * registration below, but only if it does not split into GeomSubset
             * submeshes (a single shared mesh is required for safe replay). */
            pending_geo_hash = gh ? gh : 1;
        }

        /* ---- Deferred-compute eligibility (NUSD_DEFER_COMPUTE) ----
         * Only "simple" meshes defer their triangulation+smooth-normals: plain
         * Mesh, fvc-triangulated, no GeomSubset/nested children, no face-varying
         * normals, no UVs (skipping FV/UV load is only safe with none), no
         * skinning. Everything else takes today's verbatim serial path. */
        int deferred_this_mesh = 0;
        if (defer_compute) {
            dbg_et++;
            int r_notmesh  = !is_mesh;
            int r_proto    = mesh->is_proto_only ? 1 : 0;
            int r_fvnorm   = pending_fv_normals ? 1 : 0;
            int r_nofvc    = !(fvc_data && fvc_count > 0);
            int r_children = nanousd_nchildren(prim) != 0;
            int r_uv       = (nanousd_attribarraylen(prim, "primvars:st") > 0 ||
                              nanousd_attribarraylen(prim, "st") > 0);
            /* Classify UVs (diagnostic only; still reject all via r_uv so golden
             * stays safe): vertex-rate UVs are deferrable with a serial copy,
             * face-varying need the expansion refactor. */
            if (r_uv) {
                const char* uvname = nanousd_attribarraylen(prim, "primvars:st") > 0
                                     ? "primvars:st" : "st";
                int uvn = nanousd_attribarraylen(prim, uvname);
                const char* itp = nanousd_attrib_interpolation(prim, uvname);
                if (uvn / 2 == mesh->nvertices && (!itp || strcmp(itp, "faceVarying") != 0))
                    dbg_uv_vert++;
                else
                    dbg_uv_fv++;
            } else {
                dbg_uv_none++;
            }
            int r_skel     = nanousd_attribarraylen(prim, "primvars:skel:jointIndices") > 0;
            if (r_notmesh)  dbg_rj_notmesh++;
            if (r_proto)    dbg_rj_proto++;
            if (r_fvnorm)   dbg_rj_fvnorm++;
            if (r_nofvc)    dbg_rj_nofvc++;
            if (r_children) dbg_rj_children++;
            if (r_uv)       dbg_rj_uv++;
            if (r_skel)     dbg_rj_skel++;
            if (!r_notmesh && !r_proto && !r_fvnorm && !r_nofvc && !r_children &&
                !r_uv && !r_skel) {
                deferred_this_mesh = 1;
                dbg_ok++;
            }
        }

        /* ---- (e) Triangulation ---- */
        int* tri_fv_indices = NULL;
        int have_fv_normals = pending_fv_normals &&
                              pending_fv_normals_count / 3 == fvi_count;
        if (!deferred_this_mesh) {
        if (fvc_data && fvc_count > 0) {
            int tri_count = count_triangulated_indices(fvc_data, fvc_count, fvi_count);
            mesh->nindices = tri_count;
            mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)tri_count * sizeof(uint32_t), 16);
            if (have_fv_normals)
                tri_fv_indices = (int*)arena_alloc(arena, (size_t)tri_count * sizeof(int), 16);
            if (mesh->indices) {
                triangulate_faces(fvi_data, fvi_count, fvc_data, fvc_count,
                                  mesh->indices, tri_fv_indices);
            }
        } else {
            /* No faceVertexCounts: assume all triangles */
            mesh->nindices = fvi_count;
            mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)fvi_count * sizeof(uint32_t), 16);
            if (have_fv_normals)
                tri_fv_indices = (int*)arena_alloc(arena, (size_t)fvi_count * sizeof(int), 16);
            if (mesh->indices) {
                for (int j = 0; j < fvi_count; j++) {
                    mesh->indices[j] = (uint32_t)fvi_data[j];
                    if (tri_fv_indices) tri_fv_indices[j] = j;
                }
            }
        }
        }  /* end if (!deferred_this_mesh) triangulation */

        if (!deferred_this_mesh && (!mesh->indices || mesh->nindices == 0)) {
            nanousd_freeprim(prim);
            continue;
        }

        if (!deferred_this_mesh) {
        int expanded_fv_primvars =
            load_mesh_facevarying_primvars(arena, mesh, stage, prim,
                                           fvi_data, fvi_count,
                                           fvc_data, fvc_count);

        if (!expanded_fv_primvars && !mesh->normals && have_fv_normals && tri_fv_indices) {
            expand_facevarying_normals(arena, mesh, pending_fv_normals,
                                       pending_fv_normals_count / 3,
                                       tri_fv_indices);
        }
        }

        /* ---- (e2) Compute smooth normals if none were loaded ----
         * With NUSD_PARALLEL_NORMALS this is deferred to a race-free parallel
         * post-pass over the flat-pass meshes (left NULL here). */
        if (!mesh->normals && mesh->positions && mesh->indices && mesh->nindices > 0
            && !parallel_normals && !deferred_this_mesh) {
            mesh->normals = compute_smooth_normals(arena, mesh->positions, mesh->nvertices,
                                                   mesh->indices, mesh->nindices);
        }

        /* ---- (e3) UV coordinates ---- */
        /* Loaded above with face-varying expansion so UV seams are preserved. */

        /* ---- (e4) Material binding lookup ---- */
        double _pbind0 = get_time_sec();
        mesh->material_index = scene->materials
            ? materials_find_binding((MaterialCollection*)scene->materials, stage, prim)
            : -1;
        prof_bind_s += get_time_sec() - _pbind0;
        if (scene->materials && !deferred_this_mesh) {
            scene_mesh_assign_ptex_colors(arena, mesh,
                                          (const MaterialCollection*)scene->materials,
                                          fvc_data, fvc_count, fvi_count);
        }

        if (getenv("NUSD_BIND_DIAG")) {
            const char* pp = nanousd_path(prim);
            const char* needle = getenv("NUSD_BIND_DIAG");
            const char* mname = "<none>";
            if (mesh->material_index >= 0) {
                MaterialCollection* mcd =
                    (MaterialCollection*)scene->materials;
                if (mcd && mesh->material_index < mcd->nmaterials)
                    mname = mcd->materials[mesh->material_index].prim_path;
            }
            int path_match = pp && (needle[0] == '*' || strstr(pp, needle));
            int mat_match = mname && (needle[0] == '*' || strstr(mname, needle));
            if (path_match || mat_match) {
                fprintf(stderr,
                        "[bind_diag] mesh=%s mat_idx=%d mat=%s nverts=%d uvs=%s\n",
                        pp, mesh->material_index, mname,
                        mesh->nvertices, mesh->texcoords ? "yes" : "NO");
                if (mesh->texcoords) {
                    float umin=1e9f, umax=-1e9f, vmin=1e9f, vmax=-1e9f;
                    for (int v = 0; v < mesh->nvertices; v++) {
                        float u = mesh->texcoords[v*2+0];
                        float vv = mesh->texcoords[v*2+1];
                        if (u<umin) umin=u; if (u>umax) umax=u;
                        if (vv<vmin) vmin=vv; if (vv>vmax) vmax=vv;
                    }
                    fprintf(stderr,
                            "[bind_diag]   uv range: u=[%.3f,%.3f] v=[%.3f,%.3f] first=(%.3f,%.3f)\n",
                            umin, umax, vmin, vmax,
                            mesh->texcoords[0], mesh->texcoords[1]);
                }
                if (mesh->material_index >= 0) {
                    MaterialCollection* mcd2 =
                        (MaterialCollection*)scene->materials;
                    if (mcd2 && mesh->material_index < mcd2->nmaterials) {
                        MaterialParams* p = &mcd2->materials[mesh->material_index].params;
                        fprintf(stderr,
                                "[bind_diag]   tex_indices=[%d,%d,%d,%d,%d,%d,%d] base=(%.2f,%.2f,%.2f)\n",
                                p->tex_indices[0], p->tex_indices[1],
                                p->tex_indices[2], p->tex_indices[3],
                                p->tex_indices[4], p->tex_indices[5],
                                p->tex_indices[6],
                                p->base_color[0], p->base_color[1], p->base_color[2]);
                        if (p->tex_indices[0] >= 0 && p->tex_indices[0] < mcd2->ntextures) {
                            MaterialTexture* t = &mcd2->textures[p->tex_indices[0]];
                            fprintf(stderr,
                                    "[bind_diag]   diff_tex[%d]: %dx%d path=%s\n",
                                    p->tex_indices[0], t->width, t->height,
                                    t->path);
                        }
                    }
                }
            }
        }

        /* ---- (f) displayColor ---- */
        mesh->has_display_color = 0;
        mesh->display_color[0] = 0.5f;
        mesh->display_color[1] = 0.5f;
        mesh->display_color[2] = 0.5f;
        mesh->colors = NULL;

        /* Check for per-vertex displayColor array first */
        int dc_count = 0;
        double _pr5 = get_time_sec();
        const float* dc_data = nanousd_arraydataf(prim, "primvars:displayColor", &dc_count);
        prof_read_s += get_time_sec() - _pr5;
        if (dc_data && dc_count == mesh->nvertices * 3) {
            /* Per-vertex color array — copy to arena for backend safety */
            mesh->colors = (float*)arena_alloc(arena, (size_t)dc_count * sizeof(float), 16);
            if (mesh->colors) {
                memcpy(mesh->colors, dc_data, (size_t)dc_count * sizeof(float));
            }
            mesh->has_display_color = 1;
            /* Also set the uniform fallback from first vertex */
            mesh->display_color[0] = dc_data[0];
            mesh->display_color[1] = dc_data[1];
            mesh->display_color[2] = dc_data[2];
        } else if (dc_data && dc_count == 3) {
            /* Single color stored as array of 1 vec3 */
            mesh->display_color[0] = dc_data[0];
            mesh->display_color[1] = dc_data[1];
            mesh->display_color[2] = dc_data[2];
            mesh->has_display_color = 1;
        } else {
            /* Try scalar vec3 read */
            float color[3];
            if (nanousd_attribv3f(prim, "primvars:displayColor", color)) {
                mesh->display_color[0] = color[0];
                mesh->display_color[1] = color[1];
                mesh->display_color[2] = color[2];
                mesh->has_display_color = 1;
            }
        }

        /* ---- (g) World transform ---- */
        double _pxform0 = get_time_sec();
        if (world_prewalk)
            memcpy(mesh->world_xform, &world_prewalk[(size_t)i * 16], 16 * sizeof(double));
        else
            compute_world_xform(prim, mesh->world_xform);
        prof_xform_s += get_time_sec() - _pxform0;

        if (getenv("NUSD_XFORM_DIAG") && (mesh_idx % 100 == 0 || mesh_idx < 4)) {
            const char* pp = nanousd_path(prim);
            fprintf(stderr, "[xform_diag] mesh %d path=%s T=(%.2f,%.2f,%.2f)\n",
                    mesh_idx, pp ? pp : "?",
                    mesh->world_xform[12], mesh->world_xform[13], mesh->world_xform[14]);
        }

        /* ---- Update bounds (scene + per-mesh) ---- */
        mesh_compute_local_bounds(mesh);
        mesh_compute_world_bounds_from_local(mesh, scene);

        /* ---- Offsets (zeroed, set by viewer later) ---- */
        mesh->vertex_offset = 0;
        mesh->index_offset = 0;

        /* ---- USD GeomSubset material subsets ----
         * Split a multi-material mesh into one submesh per face-domain
         * GeomSubset (each with its own material), else leave it as one
         * mesh. Submeshes share the base's vertex buffers + world_xform +
         * bounds, so scene bounds (expanded just above) are already correct. */
        int emitted = deferred_this_mesh ? 1
            : split_mesh_by_geom_subsets(arena, scene, stage, prim,
                                         mesh_idx, max_meshes,
                                         fvc_data, fvc_count, fvi_count);

        if (emitted > 1) {
            /* Split submeshes are unique (not registered as instance
             * prototypes); they share one vertex buffer, so count verts once. */
            total_verts += mesh->nvertices;
            for (int s = 0; s < emitted; s++)
                total_indices += scene->meshes[mesh_idx + s].nindices;
            mesh_idx += emitted;
        } else {
            /* ---- Instancing: register prototype mesh by path ---- */
            mesh->prototype_idx = mesh_idx;  /* self = unique or first instance */
            /* early-geo-dedup: deferred meshes have NULL indices until the
             * post-pass, so they must NOT enter the dedup prototype table (a
             * later hit would share a NULL buffer). */
            if (pending_geo_hash && !deferred_this_mesh)
                geo_dedup_insert(&geo_dedup_table, pending_geo_hash, mesh_idx);
            /* Position A: register this first-occurrence as the pre-decode
             * prototype for its composition source (deferred meshes excluded —
             * NULL indices until the post-pass). */
            if (pending_predecode_key && !deferred_this_mesh)
                geo_dedup_insert(&predecode_table, pending_predecode_key, mesh_idx);
            {
                const char* mesh_path = nanousd_path(prim);
                if (mesh_path) {
                    proto_hash_insert(&proto_mesh_table, mesh_path, mesh_idx);
                }
            }
            total_verts += mesh->nvertices;
            if (deferred_this_mesh) {
                /* Defer triangulation+smooth-normals to the parallel post-pass.
                 * Arena-copy the raw index/count arrays (insurance vs the
                 * zero-copy decode buffer) and record the work; the post-pass
                 * fills mesh->indices/normals and adds to total_indices. */
                int* fvi_copy = (int*)arena_alloc(arena, (size_t)fvi_count * sizeof(int), 16);
                int* fvc_copy = (int*)arena_alloc(arena, (size_t)fvc_count * sizeof(int), 16);
                if (fvi_copy && fvc_copy) {
                    memcpy(fvi_copy, fvi_data, (size_t)fvi_count * sizeof(int));
                    memcpy(fvc_copy, fvc_data, (size_t)fvc_count * sizeof(int));
                    dwork[n_deferred].mesh_slot = mesh_idx;
                    dwork[n_deferred].fvi = fvi_copy;
                    dwork[n_deferred].fvi_count = fvi_count;
                    dwork[n_deferred].fvc = fvc_copy;
                    dwork[n_deferred].fvc_count = fvc_count;
                    n_deferred++;
                } else {
                    /* alloc fail -> triangulate inline (bit-identical fallback) */
                    int tri = count_triangulated_indices(fvc_data, fvc_count, fvi_count);
                    mesh->nindices = tri;
                    mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)tri * sizeof(uint32_t), 16);
                    if (mesh->indices)
                        triangulate_faces(fvi_data, fvi_count, fvc_data, fvc_count,
                                          mesh->indices, NULL);
                    if (!mesh->normals && mesh->positions && mesh->indices && mesh->nindices > 0)
                        mesh->normals = compute_smooth_normals(arena, mesh->positions,
                                                               mesh->nvertices, mesh->indices,
                                                               mesh->nindices);
                    total_indices += mesh->nindices;
                }
            } else {
                total_indices += mesh->nindices;
            }
            mesh_idx++;
        }

        nanousd_freeprim(prim);
    }

    if (do_timing) {
        double t_now = get_time_sec();
        fprintf(stderr,
                "  [scene_timing] flat mesh/implicit pass: %.1f ms "
                "(meshes=%d, instance_prims=%d)\n",
                (t_now - t_mark) * 1000.0,
                mesh_idx, n_instance_prims);
        fprintf(stderr,
                "  [scene_timing] defer eligibility: defer_compute=%d total=%ld ok=%ld | "
                "reject notmesh=%ld proto=%ld fvnorm=%ld nofvc=%ld children=%ld uv=%ld skel=%ld\n",
                defer_compute, dbg_et, dbg_ok, dbg_rj_notmesh, dbg_rj_proto, dbg_rj_fvnorm,
                dbg_rj_nofvc, dbg_rj_children, dbg_rj_uv, dbg_rj_skel);
        fprintf(stderr,
                "  [scene_timing] defer UV split: none=%ld vertex-rate=%ld face-varying=%ld\n",
                dbg_uv_none, dbg_uv_vert, dbg_uv_fv);
        double prof_total = get_time_sec() - prof_pass_t0;
        double prof_compute = prof_total - prof_read_s - prof_bind_s - prof_xform_s;
        fprintf(stderr,
                "  [scene_timing] flat-pass M2 split: read/decode=%.1f ms | "
                "bind=%.1f ms | xform=%.1f ms | compute+copy=%.1f ms | total=%.1f ms "
                "(decode-frac=%.0f%%)\n",
                prof_read_s * 1000.0, prof_bind_s * 1000.0, prof_xform_s * 1000.0,
                prof_compute * 1000.0, prof_total * 1000.0,
                prof_total > 0 ? 100.0 * prof_read_s / prof_total : 0.0);
        t_mark = t_now;
    }

    if (geo_dedup_on) {
        fprintf(stderr,
                "scene_load: early geo-dedup — %ld instance(s) shared %d unique "
                "prototype(s) across %d flat meshes\n",
                geo_dedup_hits, geo_dedup_table.count, mesh_idx);
        if (compact_flat_dedup)
            fprintf(stderr,
                    "scene_load: compact flat-dedup — %ld duplicate flat meshes "
                    "routed to compact batches (0 extra SceneMesh rows)\n",
                    flat_dedup_compacted);
        geo_dedup_free(&geo_dedup_table);
    }
    if (predecode_dedup) {
        fprintf(stderr,
                "scene_load: pre-decode dedup — %ld duplicate flat meshes routed "
                "to compact batches WITHOUT geometry read (%ld batch entries; "
                "%ld signature fallbacks to full decode)\n",
                predecode_hits, predecode_compacted, predecode_sig_miss);
        geo_dedup_free(&predecode_table);
    }

    /* ---- Deferred-compute post-pass (NUSD_DEFER_COMPUTE) ----
     * Parallel triangulation + smooth-normals for the deferred "simple" meshes
     * into per-thread arenas (race-free; mirrors the curve/normals harness).
     * Runs BEFORE the smooth-normals post-pass so deferred indices exist. Each
     * mesh writes only its OWN slot; per-thread total_indices merged serially. */
    if (defer_compute && n_deferred > 0) {
        double _dc_t0 = get_time_sec();
#ifdef _OPENMP
        int dc_threads = omp_get_max_threads();
        if (dc_threads > n_deferred) dc_threads = n_deferred;
        if (dc_threads < 1) dc_threads = 1;
        Arena* dc_arenas = (Arena*)calloc((size_t)dc_threads, sizeof(Arena));
        uint64_t* dc_idx = (uint64_t*)calloc((size_t)dc_threads, sizeof(uint64_t));
        if (dc_arenas && dc_idx) {
            for (int t = 0; t < dc_threads; t++)
                dc_arenas[t] = arena_create(16 * 1024 * 1024);
            #pragma omp parallel for num_threads(dc_threads) schedule(dynamic, 32)
            for (int k = 0; k < n_deferred; k++) {
                DeferredTriWork* w = &dwork[k];
                SceneMesh* m = &scene->meshes[w->mesh_slot];
                Arena* ta = &dc_arenas[omp_get_thread_num()];
                int tri = count_triangulated_indices(w->fvc, w->fvc_count, w->fvi_count);
                m->nindices = tri;
                m->indices = (uint32_t*)arena_alloc(ta, (size_t)tri * sizeof(uint32_t), 16);
                if (m->indices)
                    triangulate_faces(w->fvi, w->fvi_count, w->fvc, w->fvc_count, m->indices, NULL);
                if (!m->normals && m->positions && m->indices && m->nindices > 0)
                    m->normals = compute_smooth_normals(ta, m->positions, m->nvertices,
                                                        m->indices, m->nindices);
                dc_idx[omp_get_thread_num()] += (uint64_t)m->nindices;
            }
            for (int t = 0; t < dc_threads; t++) {
                arena_splice(arena, &dc_arenas[t]);
                total_indices += dc_idx[t];
            }
        } else {
            for (int k = 0; k < n_deferred; k++) {
                DeferredTriWork* w = &dwork[k];
                SceneMesh* m = &scene->meshes[w->mesh_slot];
                int tri = count_triangulated_indices(w->fvc, w->fvc_count, w->fvi_count);
                m->nindices = tri;
                m->indices = (uint32_t*)arena_alloc(arena, (size_t)tri * sizeof(uint32_t), 16);
                if (m->indices)
                    triangulate_faces(w->fvi, w->fvi_count, w->fvc, w->fvc_count, m->indices, NULL);
                if (!m->normals && m->positions && m->indices && m->nindices > 0)
                    m->normals = compute_smooth_normals(arena, m->positions, m->nvertices,
                                                        m->indices, m->nindices);
                total_indices += (uint64_t)m->nindices;
            }
        }
        free(dc_arenas); free(dc_idx);
#else
        for (int k = 0; k < n_deferred; k++) {
            DeferredTriWork* w = &dwork[k];
            SceneMesh* m = &scene->meshes[w->mesh_slot];
            int tri = count_triangulated_indices(w->fvc, w->fvc_count, w->fvi_count);
            m->nindices = tri;
            m->indices = (uint32_t*)arena_alloc(arena, (size_t)tri * sizeof(uint32_t), 16);
            if (m->indices)
                triangulate_faces(w->fvi, w->fvi_count, w->fvc, w->fvc_count, m->indices, NULL);
            if (!m->normals && m->positions && m->indices && m->nindices > 0)
                m->normals = compute_smooth_normals(arena, m->positions, m->nvertices,
                                                    m->indices, m->nindices);
            total_indices += (uint64_t)m->nindices;
        }
#endif
        if (do_timing)
            fprintf(stderr,
                    "  [scene_timing] deferred-compute post-pass: %.1f ms (%d meshes)\n",
                    (get_time_sec() - _dc_t0) * 1000.0, n_deferred);
    }

    /* ---- Parallel deferred smooth-normals (NUSD_PARALLEL_NORMALS) ----
     * Pure compute (positions+indices -> normals), race-free and deterministic.
     * Sub-pass 1: compute every self-prototype's normals in parallel into
     * per-thread arenas. Sub-pass 2: instances / GeomSubset submeshes
     * (prototype_idx < own index) inherit the prototype's normals pointer —
     * identical to the serial pointer-share, so output stays bit-identical. */
    if (parallel_normals && mesh_idx > 0) {
        double _pn_t0 = get_time_sec();
        long pn_computed = 0;
#ifdef _OPENMP
        int pn_threads = omp_get_max_threads();
        if (pn_threads > mesh_idx) pn_threads = mesh_idx;
        if (pn_threads < 1) pn_threads = 1;
        Arena* pn_arenas = (Arena*)calloc((size_t)pn_threads, sizeof(Arena));
        long*  pn_counts = (long*)calloc((size_t)pn_threads, sizeof(long));
        if (pn_arenas && pn_counts) {
            for (int t = 0; t < pn_threads; t++)
                pn_arenas[t] = arena_create(8 * 1024 * 1024);
            #pragma omp parallel for num_threads(pn_threads) schedule(dynamic, 64)
            for (int i = 0; i < mesh_idx; i++) {
                SceneMesh* m = &scene->meshes[i];
                if (m->normals || !m->positions || !m->indices || m->nindices <= 0)
                    continue;
                if (m->prototype_idx != i) continue;   /* self-prototypes only */
                int tid = omp_get_thread_num();
                m->normals = compute_smooth_normals(&pn_arenas[tid], m->positions,
                                                    m->nvertices, m->indices,
                                                    m->nindices);
                pn_counts[tid]++;
            }
            for (int t = 0; t < pn_threads; t++) {
                arena_splice(arena, &pn_arenas[t]);
                pn_computed += pn_counts[t];
            }
        } else {
            for (int i = 0; i < mesh_idx; i++) {
                SceneMesh* m = &scene->meshes[i];
                if (m->normals || !m->positions || !m->indices || m->nindices <= 0)
                    continue;
                if (m->prototype_idx != i) continue;
                m->normals = compute_smooth_normals(arena, m->positions, m->nvertices,
                                                    m->indices, m->nindices);
                pn_computed++;
            }
        }
        free(pn_arenas); free(pn_counts);
#else
        for (int i = 0; i < mesh_idx; i++) {
            SceneMesh* m = &scene->meshes[i];
            if (m->normals || !m->positions || !m->indices || m->nindices <= 0)
                continue;
            if (m->prototype_idx != i) continue;
            m->normals = compute_smooth_normals(arena, m->positions, m->nvertices,
                                                m->indices, m->nindices);
            pn_computed++;
        }
#endif
        /* Sub-pass 2: dedup instances + GeomSubset submeshes inherit their
         * prototype's normals (prototype_idx is always < own index). */
        for (int i = 0; i < mesh_idx; i++) {
            SceneMesh* m = &scene->meshes[i];
            if (m->normals) continue;
            int p = m->prototype_idx;
            if (p >= 0 && p < i && scene->meshes[p].normals)
                m->normals = scene->meshes[p].normals;
        }
        if (do_timing)
            fprintf(stderr,
                    "  [scene_timing] parallel smooth-normals: %.1f ms (%ld prototypes)\n",
                    (get_time_sec() - _pn_t0) * 1000.0, pn_computed);
    }

    /* ---- Scaling probe (NUSD_SCALING_PROBE) ----
     * Decide whether the flat-pass per-mesh work is CPU- or memory-bandwidth-
     * bound BEFORE investing in the regression-sensitive nanousd thread-safety
     * guards. Over the real unique-prototype geometry, contrast a streaming sum
     * (bandwidth-bound) vs an ALU-heavy kernel (compute-bound), each serial vs
     * N-thread. If the streaming sum scales ~N x, parallelization will pay off;
     * if it stalls (~1.5x) while the ALU kernel scales ~N x, the flat pass is
     * bandwidth-limited and parallelizing it cannot reach parity on this box. */
    if (scene_env_flag("NUSD_SCALING_PROBE") && mesh_idx > 0) {
#ifdef _OPENMP
        int* sp = (int*)malloc((size_t)mesh_idx * sizeof(int));
        int nsp = 0;
        uint64_t total_floats = 0;
        if (sp) {
            for (int i = 0; i < mesh_idx; i++) {
                SceneMesh* m = &scene->meshes[i];
                if (m->prototype_idx == i && m->positions && m->nvertices > 0) {
                    sp[nsp++] = i;
                    total_floats += (uint64_t)m->nvertices * 3;
                }
            }
        }
        if (nsp > 0) {
            int maxT = omp_get_max_threads();
            const int K = 8;
            double gb = (double)total_floats * sizeof(float) / (1024.0*1024.0*1024.0);
            double bw[2] = {0,0}, cp[2] = {0,0};
            for (int pass = 0; pass < 2; pass++) {
                int nt = pass == 0 ? 1 : maxT;
                volatile double sink = 0;
                double t0 = get_time_sec();
                for (int k = 0; k < K; k++) {
                    double acc = 0;
                    #pragma omp parallel for num_threads(nt) schedule(dynamic,16) reduction(+:acc)
                    for (int j = 0; j < nsp; j++) {
                        SceneMesh* m = &scene->meshes[sp[j]];
                        int n = m->nvertices * 3; double s = 0;
                        for (int e = 0; e < n; e++) s += (double)m->positions[e];
                        acc += s;
                    }
                    sink += acc;
                }
                bw[pass] = (get_time_sec() - t0) * 1000.0 / K; (void)sink;
            }
            for (int pass = 0; pass < 2; pass++) {
                int nt = pass == 0 ? 1 : maxT;
                volatile double sink = 0;
                double t0 = get_time_sec();
                for (int k = 0; k < K; k++) {
                    double acc = 0;
                    #pragma omp parallel for num_threads(nt) schedule(dynamic,16) reduction(+:acc)
                    for (int j = 0; j < nsp; j++) {
                        SceneMesh* m = &scene->meshes[sp[j]];
                        int n = m->nvertices * 3; double s = 0;
                        for (int e = 0; e < n; e++) {
                            float v = m->positions[e];
                            for (int r = 0; r < 8; r++) v = sqrtf(v*v + 1.0f);
                            s += (double)v;
                        }
                        acc += s;
                    }
                    sink += acc;
                }
                cp[pass] = (get_time_sec() - t0) * 1000.0 / K; (void)sink;
            }
            fprintf(stderr,
                "[scaling_probe] threads=%d protos=%d data=%.2f GB | "
                "BANDWIDTH(sum): serial=%.1f par=%.1f scale=%.2fx (%.1f GB/s par) | "
                "COMPUTE(sqrtx8): serial=%.1f par=%.1f scale=%.2fx\n",
                maxT, nsp, gb,
                bw[0], bw[1], bw[1] > 0 ? bw[0]/bw[1] : 0.0,
                bw[1] > 0 ? gb/(bw[1]/1000.0) : 0.0,
                cp[0], cp[1], cp[1] > 0 ? cp[0]/cp[1] : 0.0);
        }
        free(sp);
#else
        fprintf(stderr, "[scaling_probe] OpenMP not available\n");
#endif
    }

    /* ---- Second pass: expand instance prims into instanced mesh copies ---- */
    /* Instance mesh children are NOT in the flat prim list — they are only
     * accessible via nanousd_child(). For the first instance of each prototype,
     * we fully load geometry. Subsequent instances share that geometry. */
    if (n_instance_prims > 0) {
        fprintf(stderr, "scene_load: expanding %d instance prims...\n", n_instance_prims);

        /* Nested-PointInstancer accounting (Phase 4 prep): PIs authored inside
         * native-instance prototypes (Moana's dense ground cover) are reachable
         * only via nanousd_child here and are currently dropped (the walk below
         * only materializes Mesh/Sphere/Cylinder children). Count them so the
         * gap is measured; the compact emit (decode -> resolve protos via
         * proto_mesh_table -> compose leaf_world -> pi_batches) is the next step. */
        long long n_nested_pi = 0, n_nested_pi_instances = 0;
        /* Record nested PIs during the walk; resolve + emit after it (a PI's
         * prototype meshes are only in proto_mesh_table once the walk below has
         * materialized them). pi_to_inst maps the PI's proto-local frame to the
         * concrete instance: pi_world_proto * inverse(proto_root_world) *
         * inst_root_world. */
        struct NestedPiRec* nested = NULL;
        int nested_n = 0, nested_cap = 0;

        /* Path B: native asset-arc flat seeding + replay (gated; falls back to
         * the composed walk for instances that contain nested PIs). */
        NbReplayCache nb_cache = {0, 0, 0};
        int nb_active = nb_flat_native_replay_active();
        uint64_t nb_xform_cap = scene->npi_transforms;
        int nb_batch_cap = scene->npi_batches;
        long long nb_seeded_meshes = 0;
        int nb_seeded_roots = 0;

        nb_toplevel_pi_paths_reset();
        if (nb_active && scene_compact_pi_batches_active()) {
            for (int i = 0; i < nprims; i++) {
                if (!PRIM_IS_WANTED(i)) continue;
                NanousdPrim p = nanousd_prim(stage, i);
                if (!p) continue;
                if (prim_is_point_instancer(p)) {
                    const char* pp = nanousd_path(p);
                    if (pp) nb_toplevel_pi_paths_add(pp);
                }
                nanousd_freeprim(p);
            }
        }

        /* Storm-style compact native instancing: extract prototypes once and
         * record each instance as a transform, instead of walking + cloning
         * every instance subtree. Avoids the O(instances x prototype) nanousd
         * instance-proxy composition blowup. Placed AFTER the Path-B/nested-PI
         * locals are initialized so the shared `done_instances` cleanup (which
         * frees nb_cache + drains `nested`, both empty here) is a safe no-op.
         * Kill-switch: unset the flag to fall back to the legacy expand-walk. */
        if (scene_compact_native_instances_active()) {
            int fb = (int)scene_emit_compact_native_instances(
                stage, scene, arena, instance_prim_indices, &n_instance_prims,
                &mesh_idx, max_meshes, &total_verts, &total_indices,
                &nb_xform_cap, &nb_batch_cap, do_timing);
            /* fb instances (non-Mesh prototypes) remain at the front of
             * instance_prim_indices; n_instance_prims is now fb. Run the legacy
             * walk on exactly them so nested PIs / implicit shapes aren't lost.
             * If everything compacted, skip the walk. */
            if (fb <= 0) goto done_instances;
        }

        for (int ii = 0; ii < n_instance_prims; ii++) {
            NanousdPrim inst_prim = nanousd_prim(stage, instance_prim_indices[ii]);
            if (!inst_prim) continue;

            NanousdPrim proto_root = nanousd_prototype(inst_prim);
            if (!proto_root) { nanousd_freeprim(inst_prim); continue; }
            const char* proto_root_path_raw = nanousd_path(proto_root);
            char proto_root_path_buf[8192];
            const char* proto_root_path = proto_root_path_raw;
            if (proto_root_path_raw) {
                snprintf(proto_root_path_buf, sizeof(proto_root_path_buf), "%s", proto_root_path_raw);
                proto_root_path = proto_root_path_buf;
            }
            if (!proto_root_path) { nanousd_freeprim(proto_root); nanousd_freeprim(inst_prim); continue; }

            double proto_root_world[16];
            double proto_root_inv[16];
            double inst_root_world[16];
            compute_world_xform(proto_root, proto_root_world);
            int have_proto_root_inv = invert_affine_m4d_rowvec(proto_root_world, proto_root_inv);
            compute_world_xform(inst_prim, inst_root_world);

            /* Path B: try the asset-arc flat seed/replay first. Returns >0 when
             * it handled the instance (no composed-child walk needed); 0 to
             * fall back (env off, no asset arc, or contains a nested PI). */
            int nb_progress = getenv("NUSD_NB_PROGRESS") != NULL;
            if (nb_active) {
                const char* iprp = nanousd_path(inst_prim);
                char ipbuf[1024]; ipbuf[0] = '\0';
                if (iprp) snprintf(ipbuf, sizeof(ipbuf), "%s", iprp);
                int emit_native_batches =
                    !scene_compact_pi_batches_active() ||
                    !path_prefix_list_contains(&point_instancer_prefixes, ipbuf);
                if (nb_progress)
                    fprintf(stderr, "NB_PROG root %d/%d seed-try '%s'\n",
                            ii, n_instance_prims, ipbuf);
                int before = mesh_idx;
                int handled = nb_seed_native_instance(scene, stage, inst_prim, proto_root,
                                                      ipbuf, inst_root_world,
                                                      proto_root_inv, have_proto_root_inv,
                                                      &nb_cache, arena, &mesh_idx, max_meshes,
                                                      &total_verts, &total_indices,
                                                      &nested, &nested_n, &nested_cap,
                                                      &nb_xform_cap, &nb_batch_cap,
                                                      emit_native_batches);
                if (nb_progress)
                    fprintf(stderr, "NB_PROG root %d/%d handled=%d meshes=%d\n",
                            ii, n_instance_prims, handled, mesh_idx - before);
                if (handled > 0) {
                    nb_seeded_meshes += (mesh_idx - before);
                    nb_seeded_roots++;
                    nanousd_freeprim(proto_root);
                    nanousd_freeprim(inst_prim);
                    continue;
                }
            }
            if (nb_progress)
                fprintf(stderr, "NB_PROG root %d/%d -> WALK fallback\n", ii, n_instance_prims);

            /* Traverse instance children recursively via a GROWABLE stack. A
             * fixed 256 cap silently dropped deep/wide instance subtrees (e.g.
             * Moana isCoral's xgCabbage prototype meshes nested under
             * .../Prototypes/xgCabbage/...), so geometry that should be visible
             * went missing. */
            int stack_cap = 256, stack_size = 0;
            NanousdPrim* stack_arr =
                (NanousdPrim*)malloc((size_t)stack_cap * sizeof(NanousdPrim));

            int nc = nanousd_nchildren(inst_prim);
            for (int c = nc - 1; c >= 0; c--) {
                NanousdPrim child = nanousd_child(inst_prim, c);
                if (!child) continue;
                if (stack_size >= stack_cap) {
                    stack_cap *= 2;
                    stack_arr = (NanousdPrim*)realloc(stack_arr,
                        (size_t)stack_cap * sizeof(NanousdPrim));
                }
                stack_arr[stack_size++] = child;
            }

            while (stack_size > 0) {
                NanousdPrim child = stack_arr[--stack_size];
                if (!child) continue;

                /* Push grandchildren */
                int ncc = nanousd_nchildren(child);
                for (int c = ncc - 1; c >= 0; c--) {
                    NanousdPrim gc = nanousd_child(child, c);
                    if (!gc) continue;
                    if (stack_size >= stack_cap) {
                        stack_cap *= 2;
                        stack_arr = (NanousdPrim*)realloc(stack_arr,
                            (size_t)stack_cap * sizeof(NanousdPrim));
                    }
                    stack_arr[stack_size++] = gc;
                }

                const char* ctn = nanousd_typename(child);
                if (ctn && !strcmp(ctn, "PointInstancer")) {
                    int npi = nanousd_attribarraylen(child, "protoIndices");
                    /* nanousd_path returns a pointer into a reused buffer that the
                     * very next nanousd_path / compute_world_xform call overwrites;
                     * copy it to a stable local immediately (else the recorded path
                     * gets clobbered -> drain re-get NULL -> dropped geometry). */
                    const char* cp_raw = nanousd_path(child);
                    char cp[1024]; cp[0] = '\0';
                    if (cp_raw) snprintf(cp, sizeof(cp), "%s", cp_raw);
                    if (npi > 0 && cp[0]) {
                        n_nested_pi++;
                        n_nested_pi_instances += npi;
                        if (getenv("NUSD_NESTED_PI_DIAG")) {
                            const char* ipr = nanousd_path(inst_prim);
                            fprintf(stderr, "scene_load: nested PI '%s' under instance '%s': "
                                    "%d instances\n", cp, ipr ? ipr : "?", npi);
                        }
                        /* pi_to_inst = pi_world_proto * inverse(proto_root_world) * inst_root_world */
                        double pi_world[16], t1[16], pi_to_inst[16];
                        compute_world_xform(child, pi_world);
                        if (have_proto_root_inv) {
                            nanousd_mul_m4d(pi_world, proto_root_inv, t1);
                            nanousd_mul_m4d(t1, inst_root_world, pi_to_inst);
                        } else {
                            nanousd_mul_m4d(pi_world, inst_root_world, pi_to_inst);
                        }
                        int already_nested = 0;
                        for (int dn = 0; dn < nested_n; dn++) {
                            if (!strcmp(nested[dn].path, cp)) {
                                already_nested = 1;
                                break;
                            }
                        }
                        if (!already_nested) {
                            if (nested_n >= nested_cap) {
                                int nc = nested_cap ? nested_cap * 2 : 16;
                                struct NestedPiRec* g = (struct NestedPiRec*)realloc(
                                    nested, (size_t)nc * sizeof(*nested));
                                if (g) { nested = g; nested_cap = nc; }
                            }
                            if (nested && nested_n < nested_cap) {
                                snprintf(nested[nested_n].path, sizeof(nested[nested_n].path), "%s", cp);
                                memcpy(nested[nested_n].pi_to_inst, pi_to_inst, sizeof(pi_to_inst));
                                nested_n++;
                            }
                        }
                    }
                    nanousd_freeprim(child);
                    continue;
                }
                int child_is_mesh = ctn && !strcmp(ctn, "Mesh");
                int child_is_sphere = ctn && !strcmp(ctn, "Sphere");
                int child_is_cylinder = ctn && !strcmp(ctn, "Cylinder");
                if (!child_is_mesh && !child_is_sphere && !child_is_cylinder) {
                    nanousd_freeprim(child);
                    continue;
                }

                /* Compute prototype-relative key for geometry sharing */
                const char* child_path = nanousd_path(child);
                const char* inst_path = nanousd_path(inst_prim);
                if (!child_path || !inst_path) { nanousd_freeprim(child); continue; }

                size_t inst_len = strlen(inst_path);
                size_t proto_len = strlen(proto_root_path);
                const char* relative = NULL;
                if (strncmp(child_path, inst_path, inst_len) == 0 &&
                    (child_path[inst_len] == '\0' || child_path[inst_len] == '/')) {
                    relative = child_path + inst_len;
                } else if (strncmp(child_path, proto_root_path, proto_len) == 0 &&
                           (child_path[proto_len] == '\0' || child_path[proto_len] == '/')) {
                    relative = child_path + proto_len;
                } else {
                    relative = "";
                }

                char proto_key[8192];
                snprintf(proto_key, sizeof(proto_key), "%s%s", proto_root_path, relative);
                char instance_child_path[8192];
                snprintf(instance_child_path, sizeof(instance_child_path), "%s%s", inst_path, relative);

                /* Look up existing prototype mesh */
                int proto_idx = proto_hash_lookup(&proto_mesh_table, proto_key);

                if (proto_idx >= 0) {
                    /* Bounds check */
                    if (mesh_idx >= max_meshes) {
                        fprintf(stderr, "scene_load: mesh array overflow (%d >= %d), truncating instances\n",
                                mesh_idx, max_meshes);
                        nanousd_freeprim(child);
                        goto done_instances;
                    }
                    /* Share geometry with already-loaded prototype mesh */
                    SceneMesh* proto_m = &scene->meshes[proto_idx];
                    SceneMesh* mesh = &scene->meshes[mesh_idx];
                    mesh->path = scene_arena_strdup(arena, instance_child_path);
                    mesh->positions  = proto_m->positions;
                    mesh->normals    = proto_m->normals;
                    mesh->indices    = proto_m->indices;
                    mesh->nvertices  = proto_m->nvertices;
                    mesh->nindices   = proto_m->nindices;
                    mesh->prototype_idx = proto_idx;
                    mesh->has_display_color = proto_m->has_display_color;
                    mesh->display_color[0] = proto_m->display_color[0];
                    mesh->display_color[1] = proto_m->display_color[1];
                    mesh->display_color[2] = proto_m->display_color[2];
                    mesh->colors = proto_m->colors;
                    mesh->texcoords = proto_m->texcoords;
                    mesh->material_index = proto_m->material_index;
                    mesh->is_proto_only = 0;
                    memcpy(mesh->local_bounds_min, proto_m->local_bounds_min, sizeof(float) * 3);
                    memcpy(mesh->local_bounds_max, proto_m->local_bounds_max, sizeof(float) * 3);

                    {
                        double child_world[16];
                        compute_world_xform(child, child_world);
                        if (have_proto_root_inv) {
                            double child_proto_rel[16];
                            nanousd_mul_m4d(child_world, proto_root_inv, child_proto_rel);
                            nanousd_mul_m4d(child_proto_rel, inst_root_world, mesh->world_xform);
                        } else {
                            memcpy(mesh->world_xform, child_world, sizeof(child_world));
                        }
                    }

                    mesh_compute_world_bounds_from_local(mesh, scene);
                    mesh->vertex_offset = 0;
                    mesh->index_offset = 0;
                    total_verts += mesh->nvertices;
                    total_indices += mesh->nindices;
                    mesh_idx++;
                    instanced_meshes++;
                } else {
                    /* Bounds check */
                    if (mesh_idx >= max_meshes) {
                        fprintf(stderr, "scene_load: mesh array overflow (%d >= %d), truncating instances\n",
                                mesh_idx, max_meshes);
                        nanousd_freeprim(child);
                        goto done_instances;
                    }
                    /* First instance of this prototype mesh — load geometry fully */
                    SceneMesh* mesh = &scene->meshes[mesh_idx];
                    mesh->path = scene_arena_strdup(arena, instance_child_path);

                    if (child_is_sphere || child_is_cylinder) {
                        double shape_xform[16];
                        int proto_idx = -1;
                        int ok = 0;
                        if (child_is_sphere) {
                            implicit_sphere_xform(child, shape_xform);
                            proto_idx = implicit_sphere_proto;
                            ok = (proto_idx >= 0)
                                ? load_sphere_instance(child, arena, mesh,
                                                       &scene->meshes[proto_idx],
                                                       shape_xform)
                                : load_sphere(child, arena, mesh, shape_xform);
                        } else {
                            int axis = 2;
                            implicit_cylinder_xform(child, &axis, shape_xform);
                            if (axis < 0 || axis > 2) axis = 2;
                            proto_idx = implicit_cylinder_proto[axis];
                            ok = (proto_idx >= 0)
                                ? load_cylinder_instance(child, arena, mesh,
                                                         &scene->meshes[proto_idx],
                                                         shape_xform)
                                : load_cylinder(child, arena, mesh, axis,
                                                shape_xform);
                            if (ok && proto_idx < 0)
                                implicit_cylinder_proto[axis] = mesh_idx;
                        }
                        if (ok) {
                            if (proto_idx >= 0) {
                                mesh->prototype_idx = proto_idx;
                            } else {
                                mesh->prototype_idx = mesh_idx;
                                if (child_is_sphere) implicit_sphere_proto = mesh_idx;
                            }
                            scene_expand_bounds(scene, mesh->bounds_min, mesh->bounds_max);
                            mesh->vertex_offset = 0;
                            mesh->index_offset = 0;
                            proto_hash_insert(&proto_mesh_table, proto_key, mesh_idx);
                            total_verts += mesh->nvertices;
                            total_indices += mesh->nindices;
                            mesh_idx++;
                        }
                        nanousd_freeprim(child);
                        continue;
                    }

                    /* Points */
                    int pts_count = 0;
                    const float* pts_data = nanousd_arraydataf(child, "points", &pts_count);
                    if (pts_data && pts_count > 0) {
                        mesh->nvertices = pts_count / 3;
                        mesh->positions = (float*)arena_alloc(arena, (size_t)pts_count * sizeof(float), 16);
                        if (mesh->positions) memcpy(mesh->positions, pts_data, (size_t)pts_count * sizeof(float));
                    } else {
                        int len = nanousd_attribarraylen(child, "points");
                        if (len > 0) {
                            mesh->nvertices = len / 3;
                            mesh->positions = (float*)arena_alloc(arena, (size_t)len * sizeof(float), 16);
                            if (mesh->positions) nanousd_attribarrayf(child, "points", mesh->positions, len);
                        }
                    }
                    if (mesh->nvertices == 0 || !mesh->positions) { nanousd_freeprim(child); continue; }

                    /* Normals */
                    int normals_count = 0;
                    const float* normals_data = nanousd_arraydataf(child, "normals", &normals_count);
                    const float* authored_normals_data = normals_data;
                    int authored_normals_count = normals_count;
                    float* pending_fv_normals = NULL;
                    int pending_fv_normals_count = 0;
                    if (!normals_data || normals_count / 3 != mesh->nvertices) {
                        int pv_normals_count = 0;
                        const float* pv_normals_data =
                            nanousd_arraydataf(child, "primvars:normals", &pv_normals_count);
                        if (pv_normals_data && pv_normals_count / 3 == mesh->nvertices) {
                            normals_data = pv_normals_data;
                            normals_count = pv_normals_count;
                        } else if (!authored_normals_data) {
                            normals_data = pv_normals_data;
                            normals_count = pv_normals_count;
                            authored_normals_data = pv_normals_data;
                            authored_normals_count = pv_normals_count;
                        }
                    }
                    if (normals_data && normals_count > 0 && normals_count / 3 == mesh->nvertices) {
                        mesh->normals = (float*)arena_alloc(arena, (size_t)normals_count * sizeof(float), 16);
                        if (mesh->normals) memcpy(mesh->normals, normals_data, (size_t)normals_count * sizeof(float));
                    } else {
                        if (authored_normals_data && authored_normals_count > 0) {
                            pending_fv_normals =
                                (float*)arena_alloc(arena, (size_t)authored_normals_count * sizeof(float), 16);
                            if (pending_fv_normals) {
                                memcpy(pending_fv_normals, authored_normals_data,
                                       (size_t)authored_normals_count * sizeof(float));
                                pending_fv_normals_count = authored_normals_count;
                            }
                        }
                        mesh->normals = NULL;
                    }

                    /* Face vertex indices + triangulation */
                    int fvi_count = 0;
                    const int* fvi_data = nanousd_arraydatai(child, "faceVertexIndices", &fvi_count);
                    if (!fvi_data || fvi_count <= 0) {
                        int len = nanousd_attribarraylen(child, "faceVertexIndices");
                        if (len > 0) {
                            fvi_count = len;
                            int* fb = (int*)arena_alloc(arena, (size_t)len * sizeof(int), 16);
                            if (fb) nanousd_attribarrayi(child, "faceVertexIndices", fb, len);
                            fvi_data = fb;
                        }
                    }
                    if (!fvi_data || fvi_count <= 0) { nanousd_freeprim(child); continue; }

                    int fvc_count = 0;
                    const int* fvc_data = nanousd_arraydatai(child, "faceVertexCounts", &fvc_count);
                    if (!fvc_data || fvc_count <= 0) {
                        int len = nanousd_attribarraylen(child, "faceVertexCounts");
                        if (len > 0) {
                            fvc_count = len;
                            int* fb = (int*)arena_alloc(arena, (size_t)len * sizeof(int), 16);
                            if (fb) nanousd_attribarrayi(child, "faceVertexCounts", fb, len);
                            fvc_data = fb;
                        }
                    }

                    int* tri_fv_indices = NULL;
                    int have_fv_normals = pending_fv_normals &&
                                          pending_fv_normals_count / 3 == fvi_count;
                    if (fvc_data && fvc_count > 0) {
                        int tri_count = count_triangulated_indices(fvc_data, fvc_count, fvi_count);
                        mesh->nindices = tri_count;
                        mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)tri_count * sizeof(uint32_t), 16);
                        if (have_fv_normals)
                            tri_fv_indices = (int*)arena_alloc(arena, (size_t)tri_count * sizeof(int), 16);
                        if (mesh->indices) {
                            triangulate_faces(fvi_data, fvi_count, fvc_data, fvc_count,
                                              mesh->indices, tri_fv_indices);
                        }
                    } else {
                        mesh->nindices = fvi_count;
                        mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)fvi_count * sizeof(uint32_t), 16);
                        if (have_fv_normals)
                            tri_fv_indices = (int*)arena_alloc(arena, (size_t)fvi_count * sizeof(int), 16);
                        if (mesh->indices) {
                            for (int j = 0; j < fvi_count; j++) {
                                mesh->indices[j] = (uint32_t)fvi_data[j];
                                if (tri_fv_indices) tri_fv_indices[j] = j;
                            }
                        }
                    }
                    if (!mesh->indices || mesh->nindices == 0) { nanousd_freeprim(child); continue; }

                    int expanded_fv_primvars =
                        load_mesh_facevarying_primvars(arena, mesh, stage, child,
                                                       fvi_data, fvi_count,
                                                       fvc_data, fvc_count);

                    if (!expanded_fv_primvars && !mesh->normals && have_fv_normals && tri_fv_indices) {
                        expand_facevarying_normals(arena, mesh, pending_fv_normals,
                                                   pending_fv_normals_count / 3,
                                                   tri_fv_indices);
                    }

                    /* Smooth normals if needed */
                    if (!mesh->normals && mesh->positions && mesh->indices)
                        mesh->normals = compute_smooth_normals(arena, mesh->positions, mesh->nvertices,
                                                               mesh->indices, mesh->nindices);

                    /* UV coordinates are loaded with face-varying expansion above. */

                    /* Material binding lookup */
                    mesh->material_index = scene->materials
                        ? materials_find_binding((MaterialCollection*)scene->materials, stage, child)
                        : -1;

                    /* displayColor */
                    mesh->has_display_color = 0;
                    mesh->display_color[0] = 0.5f; mesh->display_color[1] = 0.5f; mesh->display_color[2] = 0.5f;
                    mesh->colors = NULL;
                    {
                        int dc_count = 0;
                        const float* dc_data = nanousd_arraydataf(child, "primvars:displayColor", &dc_count);
                        if (dc_data && dc_count == mesh->nvertices * 3) {
                            mesh->colors = (float*)arena_alloc(arena, (size_t)dc_count * sizeof(float), 16);
                            if (mesh->colors) memcpy(mesh->colors, dc_data, (size_t)dc_count * sizeof(float));
                            mesh->has_display_color = 1;
                            mesh->display_color[0] = dc_data[0]; mesh->display_color[1] = dc_data[1]; mesh->display_color[2] = dc_data[2];
                        } else if (dc_data && dc_count == 3) {
                            mesh->display_color[0] = dc_data[0]; mesh->display_color[1] = dc_data[1]; mesh->display_color[2] = dc_data[2];
                            mesh->has_display_color = 1;
                        } else {
                            float color[3];
                            if (nanousd_attribv3f(child, "primvars:displayColor", color)) {
                                mesh->display_color[0] = color[0]; mesh->display_color[1] = color[1]; mesh->display_color[2] = color[2];
                                mesh->has_display_color = 1;
                            }
                        }
                    }

                    /* World transform: compose prototype-relative child xform
                     * under the concrete instance root. */
                    {
                        double child_world[16];
                        compute_world_xform(child, child_world);
                        if (have_proto_root_inv) {
                            double child_proto_rel[16];
                            nanousd_mul_m4d(child_world, proto_root_inv, child_proto_rel);
                            nanousd_mul_m4d(child_proto_rel, inst_root_world, mesh->world_xform);
                        } else {
                            memcpy(mesh->world_xform, child_world, sizeof(child_world));
                        }
                    }
                    mesh->prototype_idx = mesh_idx; /* self = first instance */

                    /* Bounds */
                    mesh_compute_local_bounds(mesh);
                    mesh_compute_world_bounds_from_local(mesh, scene);
                    mesh->vertex_offset = 0;
                    mesh->index_offset = 0;

                    /* Register as prototype for sharing */
                    proto_hash_insert(&proto_mesh_table, proto_key, mesh_idx);

                    total_verts += mesh->nvertices;
                    total_indices += mesh->nindices;
                    mesh_idx++;
                }

                nanousd_freeprim(child);
            }
            free(stack_arr);

            nanousd_freeprim(proto_root);
            nanousd_freeprim(inst_prim);
        }
done_instances: ;
        if (nb_active) {
            if (getenv("NUSD_NESTED_PI_DIAG") || nb_seeded_roots > 0)
                fprintf(stderr, "scene_load: Path B asset-arc seed/replay — "
                        "%d roots, %lld meshes (remaining roots fell back to walk)\n",
                        nb_seeded_roots, nb_seeded_meshes);
            nb_cache_free(&nb_cache);
        }
        nb_toplevel_pi_paths_reset();
        /* ---- Drain nested PIs: protos are now in proto_mesh_table; resolve,
         * compose leaf_world = inst_local * pi_to_inst, and emit compact
         * batches. Proto meshes are marked is_proto_only so they aren't drawn
         * standalone (they're instanced via the PI). ---- */
        uint64_t nested_xform_cap = 0;
        int      nested_batch_cap = 0;
        long long nested_emitted = 0;
        for (int ni = 0; ni < nested_n; ni++) {
            NanousdPrim pi = nanousd_primpath(stage, nested[ni].path);
            if (!pi) {
                if (getenv("NUSD_NESTED_PI_DIAG"))
                    fprintf(stderr, "scene_load:   drain re-get NULL: '%s'\n", nested[ni].path);
                continue;
            }
            int n_inst = nanousd_attribarraylen(pi, "protoIndices");
            if (n_inst <= 0) {
                if (getenv("NUSD_NESTED_PI_DIAG"))
                    fprintf(stderr, "scene_load:   drain protoIndices=%d: '%s'\n",
                            n_inst, nested[ni].path);
                nanousd_freeprim(pi); continue;
            }
            int*   pidxs = (int*)malloc((size_t)n_inst * sizeof(int));
            float* pos   = (float*)malloc((size_t)n_inst * 3 * sizeof(float));
            float* ori   = (float*)malloc((size_t)n_inst * 4 * sizeof(float));
            float* scl   = (float*)malloc((size_t)n_inst * 3 * sizeof(float));
            if (!pidxs || !pos || !ori || !scl) {
                free(pidxs); free(pos); free(ori); free(scl); nanousd_freeprim(pi); continue;
            }
            nanousd_attribarrayi(pi, "protoIndices", pidxs, n_inst);
            int np = nanousd_attribarraylen(pi, "positions");
            if (np > 0) nanousd_attribarrayf(pi, "positions", pos, np * 3);
            else memset(pos, 0, (size_t)n_inst * 3 * sizeof(float));
            int no = nanousd_attribarraylen(pi, "orientations");
            if (no > 0) nanousd_attribarrayf(pi, "orientations", ori, no * 4);
            else for (int q = 0; q < n_inst; q++) { ori[q*4]=0; ori[q*4+1]=0; ori[q*4+2]=0; ori[q*4+3]=1; }
            int ns = nanousd_attribarraylen(pi, "scales");
            if (ns > 0) nanousd_attribarrayf(pi, "scales", scl, ns * 3);
            else for (int s = 0; s < n_inst * 3; s++) scl[s] = 1.0f;

            unsigned char* inst_visible = pi_visible_mask_from_invisible_ids(pi, n_inst);
            PiProtoTargets proto_targets;
            pi_proto_targets_collect(pi, &proto_targets);
            int n_protos = proto_targets.count;
            if (n_protos <= 0) {
                if (getenv("NUSD_NESTED_PI_DIAG") || getenv("NUSD_PI_PROTO_DIAG"))
                    fprintf(stderr, "scene_load:   drain '%s': no prototype targets\n",
                            nested[ni].path);
                pi_proto_targets_free(&proto_targets);
                free(inst_visible);
                free(pidxs); free(pos); free(ori); free(scl);
                nanousd_freeprim(pi); continue;
            }
            int** plist  = (int**)calloc((size_t)n_protos, sizeof(int*));
            int*  pcount = (int*)calloc((size_t)n_protos, sizeof(int));
            for (int p = 0; p < n_protos; p++) {
                char tp_buf[1024];
                const char* tp = pi_proto_target_composed_path(
                    proto_targets.paths[p], nested[ni].path,
                    tp_buf, sizeof(tp_buf));
                if (!tp) continue;
                int exact = proto_hash_lookup(&proto_mesh_table, tp);
                if (exact < 0) {
                    for (int sm = 0; sm < mesh_idx; sm++) {
                        const char* sp = scene->meshes[sm].path;
                        if (sp && strcmp(sp, tp) == 0) {
                            exact = sm;
                            break;
                        }
                    }
                }
                if (exact >= 0) {
                    plist[p] = (int*)malloc(sizeof(int)); plist[p][0] = exact; pcount[p] = 1;
                    if (exact < mesh_idx) scene->meshes[exact].is_proto_only = 1;
                    continue;
                }
                size_t tl = strlen(tp);
                int count = 0;
                for (int pm = 0; pm < proto_mesh_table.cap; pm++) {
                    const ProtoMeshSlot* sl = &proto_mesh_table.slots[pm];
                    if (sl->path[0] == '\0') continue;
                    if (strncmp(sl->path, tp, tl) == 0 && (sl->path[tl] == '/' || sl->path[tl] == '\0'))
                        count++;
                }
                if (count > 0) {
                    plist[p] = (int*)malloc((size_t)count * sizeof(int));
                    int wi = 0;
                    for (int pm = 0; pm < proto_mesh_table.cap && wi < count; pm++) {
                        const ProtoMeshSlot* sl = &proto_mesh_table.slots[pm];
                        if (sl->path[0] == '\0') continue;
                        if (strncmp(sl->path, tp, tl) == 0 &&
                            (sl->path[tl] == '/' || sl->path[tl] == '\0')) {
                            plist[p][wi++] = sl->mesh_idx;
                        }
                    }
                    pcount[p] = wi;
                } else {
                    for (int sm = 0; sm < mesh_idx; sm++) {
                        const char* sp = scene->meshes[sm].path;
                        if (!sp) continue;
                        if (strncmp(sp, tp, tl) == 0 &&
                            (sp[tl] == '/' || sp[tl] == '\0')) {
                            count++;
                        }
                    }
                    if (count > 0) {
                        plist[p] = (int*)malloc((size_t)count * sizeof(int));
                        int wi = 0;
                        for (int sm = 0; sm < mesh_idx && wi < count; sm++) {
                            const char* sp = scene->meshes[sm].path;
                            if (!sp) continue;
                            if (strncmp(sp, tp, tl) == 0 &&
                                (sp[tl] == '/' || sp[tl] == '\0')) {
                                plist[p][wi++] = sm;
                            }
                        }
                        pcount[p] = wi;
                    }
                }
                for (int k = 0; k < pcount[p]; k++) {
                    int sm = plist[p][k];
                    if (sm >= 0 && sm < mesh_idx) scene->meshes[sm].is_proto_only = 1;
                }
                if (pcount[p] == 0 && getenv("NUSD_NESTED_PI_DIAG")) {
                    NanousdPrim tpp = nanousd_primpath(stage, tp);
                    fprintf(stderr, "scene_load:   nested-PI proto UNRESOLVED '%s' "
                            "(type=%s instanceable=%d nchild=%d)\n", tp,
                            tpp ? (nanousd_typename(tpp) ? nanousd_typename(tpp) : "?") : "NULL",
                            tpp ? nanousd_isinstanceable(tpp) : -1,
                            tpp ? nanousd_nchildren(tpp) : -1);
                    if (tpp) nanousd_freeprim(tpp);
                }
            }
            for (int pidx = 0; pidx < n_protos; pidx++) {
                if (pcount[pidx] == 0) continue;
                int kept = 0;
                for (int inst = 0; inst < n_inst; inst++)
                    if ((!inst_visible || inst_visible[inst]) && pidxs[inst] == pidx)
                        kept++;
                if (kept == 0) continue;
                uint32_t base = (uint32_t)scene->npi_transforms;
                SceneInstanceTransform* slab =
                    scene_pi_transforms_reserve(scene, &nested_xform_cap, (uint64_t)kept);
                if (!slab) break;
                uint32_t w = 0;
                for (int inst = 0; inst < n_inst && w < (uint32_t)kept; inst++) {
                    if (inst_visible && !inst_visible[inst]) continue;
                    if (pidxs[inst] != pidx) continue;
                    float qi=ori[inst*4],qj=ori[inst*4+1],qk=ori[inst*4+2],qw=ori[inst*4+3];
                    float sx=scl[inst*3],sy=scl[inst*3+1],sz=scl[inst*3+2];
                    double ix[16];
                    double ww=qw*qw,ii=qi*qi,jj=qj*qj,kk=qk*qk,wi=qw*qi,wj=qw*qj,wk=qw*qk,ij=qi*qj,ik=qi*qk,jk=qj*qk;
                    ix[0]=(ww+ii-jj-kk)*sx; ix[1]=2*(ij+wk)*sx; ix[2]=2*(ik-wj)*sx; ix[3]=0;
                    ix[4]=2*(ij-wk)*sy; ix[5]=(ww-ii+jj-kk)*sy; ix[6]=2*(jk+wi)*sy; ix[7]=0;
                    ix[8]=2*(ik+wj)*sz; ix[9]=2*(jk-wi)*sz; ix[10]=(ww-ii-jj+kk)*sz; ix[11]=0;
                    ix[12]=pos[inst*3]; ix[13]=pos[inst*3+1]; ix[14]=pos[inst*3+2]; ix[15]=1;
                    double lw[16];
                    nanousd_mul_m4d(ix, nested[ni].pi_to_inst, lw);
                    SceneInstanceTransform* t = &slab[w++];
                    t->m[0]=(float)lw[0]; t->m[1]=(float)lw[1]; t->m[2]=(float)lw[2];
                    t->m[3]=(float)lw[4]; t->m[4]=(float)lw[5]; t->m[5]=(float)lw[6];
                    t->m[6]=(float)lw[8]; t->m[7]=(float)lw[9]; t->m[8]=(float)lw[10];
                    t->m[9]=(float)lw[12]; t->m[10]=(float)lw[13]; t->m[11]=(float)lw[14];
                }
                for (int sm = 0; sm < pcount[pidx]; sm++) {
                    int src = plist[pidx][sm];
                    if (!scene_pi_batches_reserve(scene, &nested_batch_cap, scene->npi_batches + 1)) break;
                    SceneInstanceBatch* b = &scene->pi_batches[scene->npi_batches++];
                    b->prototype_mesh_idx = src; b->transform_offset = base;
                    b->transform_count = (uint32_t)kept; b->source_prim_idx = -1;
                    b->material_or_binding_id = -1; b->source_kind = SCENE_INSTANCE_SOURCE_POINT_INSTANCER;
                    nested_emitted += kept;
                }
            }
            if (getenv("NUSD_NESTED_PI_DIAG")) {
                long long em = 0; for (int p = 0; p < n_protos; p++) em += pcount[p];
                fprintf(stderr, "scene_load:   drain '%s': n_inst=%d n_protos=%d resolved_protos=%lld\n",
                        nested[ni].path, n_inst, n_protos, em);
            }
            for (int p = 0; p < n_protos; p++) free(plist[p]);
            free(plist); free(pcount);
            pi_proto_targets_free(&proto_targets);
            free(inst_visible);
            free(pidxs); free(pos); free(ori); free(scl);
            nanousd_freeprim(pi);
        }
        free(nested);
        if (n_nested_pi > 0)
            fprintf(stderr,
                    "scene_load: nested PIs: %lld PI(s), %lld instances; compact emitted "
                    "%lld transforms (was 0 = dropped)\n",
                    n_nested_pi, n_nested_pi_instances, nested_emitted);
    }

    free(instance_prim_indices);

    if (do_timing) {
        double t_now = get_time_sec();
        fprintf(stderr,
                "  [scene_timing] instance-prim expansion: %.1f ms "
                "(meshes=%d, instanced=%d)\n",
                (t_now - t_mark) * 1000.0,
                mesh_idx, instanced_meshes);
        t_mark = t_now;
    }

    /* ---- Third pass: materialize PointInstancers ----
     *
     * Legacy/default mode still emits per-instance SceneMesh clones for
     * compatibility. Moana/no-cull mode uses compact-only PI batches so the
     * scene model keeps one prototype mesh plus 48-byte transforms instead of
     * millions of fat SceneMesh rows. */
    uint64_t pi_xform_cap = 0;
    int      pi_batch_cap = 0;
    int      compact_pi_only = scene_compact_pi_batches_active();
    long long pi_scene_mesh_clones = 0;
    for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;
        if (!PRIM_IS_WANTED(i) && !PRIM_IS_FILTER_CONTROLLER(prim)) {
            filtered_skips++;
            nanousd_freeprim(prim);
            continue;
        }
        if (!prim_is_point_instancer(prim)) { nanousd_freeprim(prim); continue; }

        /* Read instance arrays */
        int n_instances = nanousd_attribarraylen(prim, "protoIndices");
        if (n_instances <= 0) { nanousd_freeprim(prim); continue; }

        int* proto_indices = (int*)malloc((size_t)n_instances * sizeof(int));
        float* positions = (float*)malloc((size_t)n_instances * 3 * sizeof(float));
        float* orientations = (float*)malloc((size_t)n_instances * 4 * sizeof(float));
        float* scales = (float*)malloc((size_t)n_instances * 3 * sizeof(float));
        if (!proto_indices || !positions || !orientations || !scales) {
            free(proto_indices); free(positions); free(orientations); free(scales);
            nanousd_freeprim(prim); continue;
        }

        nanousd_attribarrayi(prim, "protoIndices", proto_indices, n_instances);
        int npos = nanousd_attribarraylen(prim, "positions");
        if (npos > 0) nanousd_attribarrayf(prim, "positions", positions, npos * 3);
        else memset(positions, 0, (size_t)n_instances * 3 * sizeof(float));

        int norient = nanousd_attribarraylen(prim, "orientations");
        if (norient > 0) nanousd_attribarrayf(prim, "orientations", orientations, norient * 4);
        else { /* default identity quaternion in nanousd's (x, y, z, w)
                 * layout — not USD's text-form (real, imag) ordering. */
            for (int q = 0; q < n_instances; q++) {
                orientations[q*4+0] = 0.0f; orientations[q*4+1] = 0.0f;
                orientations[q*4+2] = 0.0f; orientations[q*4+3] = 1.0f;
            }
        }

        int nscale = nanousd_attribarraylen(prim, "scales");
        if (nscale > 0) nanousd_attribarrayf(prim, "scales", scales, nscale * 3);
        else { for (int s = 0; s < n_instances * 3; s++) scales[s] = 1.0f; }

        unsigned char* inst_visible =
            pi_visible_mask_from_invisible_ids(prim, n_instances);

        /* Resolve prototype paths. Each prototype may itself be a Mesh
         * (exact-match in proto_mesh_table) OR an Xform/Scope wrapping
         * one OR MORE Mesh descendants (e.g. the chess set's Pawn
         * proto has a Geom_Body and a Geom_Top). We collect ALL
         * matching mesh indices per prototype so the instance loop
         * below can fan out one scene-mesh per (proto-mesh × instance)
         * pair. */
        PiProtoTargets proto_targets;
        pi_proto_targets_collect(prim, &proto_targets);
        int n_protos = proto_targets.count;
        if (n_protos <= 0) {
            if (getenv("NUSD_PI_PROTO_DIAG"))
                fprintf(stderr, "scene_load: PointInstancer '%s': no prototype targets\n",
                        nanousd_path(prim));
            pi_proto_targets_free(&proto_targets);
            free(inst_visible);
            free(proto_indices);
            free(positions);
            free(orientations);
            free(scales);
            nanousd_freeprim(prim);
            continue;
        }
        int** proto_mesh_index_lists = (int**)calloc((size_t)n_protos, sizeof(int*));
        int*  proto_mesh_index_counts = (int*)calloc((size_t)n_protos, sizeof(int));
        const char* _pi_full_path_raw = nanousd_path(prim);
        char _pi_full_path_buf[1024];
        const char* _pi_full_path = _pi_full_path_raw;
        if (_pi_full_path_raw) {
            snprintf(_pi_full_path_buf, sizeof(_pi_full_path_buf), "%s",
                     _pi_full_path_raw);
            _pi_full_path = _pi_full_path_buf;
        }
        for (int p = 0; p < n_protos; p++) {
            const char* proto_path_raw = proto_targets.paths[p];
            if (!proto_path_raw) continue;
            char proto_path_buf[1024];
            const char* proto_path = pi_proto_target_composed_path(
                proto_path_raw, _pi_full_path,
                proto_path_buf, sizeof(proto_path_buf));
            /* Exact match first (proto IS itself a Mesh). */
            int direct = proto_hash_lookup(&proto_mesh_table, proto_path);
            if (direct < 0) {
                for (int sm = 0; sm < mesh_idx; sm++) {
                    const char* sp = scene->meshes[sm].path;
                    if (sp && strcmp(sp, proto_path) == 0) {
                        direct = sm;
                        break;
                    }
                }
            }
            if (direct >= 0) {
                proto_mesh_index_lists[p]  = (int*)malloc(sizeof(int));
                proto_mesh_index_lists[p][0] = direct;
                proto_mesh_index_counts[p] = 1;
                continue;
            }
            /* Otherwise collect every mesh whose path is a descendant of
             * proto_path. Two passes — count, then fill. */
            size_t proto_len = strlen(proto_path);
            int count = 0;
            for (int pm = 0; pm < proto_mesh_table.cap; pm++) {
                const ProtoMeshSlot* s = &proto_mesh_table.slots[pm];
                if (s->path[0] == '\0') continue;
                if (strncmp(s->path, proto_path, proto_len) == 0 &&
                    (s->path[proto_len] == '/' ||
                     s->path[proto_len] == '\0')) {
                    count++;
                }
            }
            if (count == 0) {
                for (int sm = 0; sm < mesh_idx; sm++) {
                    const char* sp = scene->meshes[sm].path;
                    if (!sp) continue;
                    if (strncmp(sp, proto_path, proto_len) == 0 &&
                        (sp[proto_len] == '/' || sp[proto_len] == '\0')) {
                        count++;
                    }
                }
                if (count == 0 && scene_compact_pi_batches_active()) {
                    int before_seed = mesh_idx;
                    nb_seed_top_level_xgen_archive_proto(
                        scene, stage, _pi_full_path, proto_path, arena,
                        &mesh_idx, max_meshes, &total_verts, &total_indices);
                    if (mesh_idx > before_seed) {
                        for (int sm = before_seed; sm < mesh_idx; sm++) {
                            const char* sp = scene->meshes[sm].path;
                            if (!sp) continue;
                            if (strncmp(sp, proto_path, proto_len) == 0 &&
                                (sp[proto_len] == '/' || sp[proto_len] == '\0')) {
                                count++;
                            }
                        }
                    }
                }
                if (count == 0) {
                    if (getenv("NUSD_PI_PROTO_DIAG")) {
                        NanousdPrim pp = nanousd_primpath(stage, proto_path);
                        fprintf(stderr,
                                "scene_load: PointInstancer '%s': unresolved "
                                "prototype '%s' (type=%s nchild=%d)\n",
                                _pi_full_path ? _pi_full_path : "?",
                                proto_path,
                                pp ? (nanousd_typename(pp) ? nanousd_typename(pp) : "?") : "NULL",
                                pp ? nanousd_nchildren(pp) : -1);
                        if (pp) nanousd_freeprim(pp);
                    }
                    continue;
                }
                proto_mesh_index_lists[p]  = (int*)malloc((size_t)count * sizeof(int));
                int wi = 0;
                for (int sm = 0; sm < mesh_idx && wi < count; sm++) {
                    const char* sp = scene->meshes[sm].path;
                    if (!sp) continue;
                    if (strncmp(sp, proto_path, proto_len) == 0 &&
                        (sp[proto_len] == '/' || sp[proto_len] == '\0')) {
                        proto_mesh_index_lists[p][wi++] = sm;
                    }
                }
                proto_mesh_index_counts[p] = wi;
                continue;
            }
            proto_mesh_index_lists[p]  = (int*)malloc((size_t)count * sizeof(int));
            int wi = 0;
            for (int pm = 0; pm < proto_mesh_table.cap; pm++) {
                const ProtoMeshSlot* s = &proto_mesh_table.slots[pm];
                if (s->path[0] == '\0') continue;
                if (strncmp(s->path, proto_path, proto_len) == 0 &&
                    (s->path[proto_len] == '/' ||
                     s->path[proto_len] == '\0')) {
                    proto_mesh_index_lists[p][wi++] = s->mesh_idx;
                }
            }
            proto_mesh_index_counts[p] = count;
        }
        if (getenv("NUSD_PI_PROTO_DIAG")) {
            int resolved_protos = 0;
            int resolved_meshes = 0;
            int max_proto_index = -1;
            for (int p = 0; p < n_protos; p++) {
                if (proto_mesh_index_counts[p] > 0) resolved_protos++;
                resolved_meshes += proto_mesh_index_counts[p];
            }
            for (int inst = 0; inst < n_instances; inst++) {
                if (proto_indices[inst] > max_proto_index)
                    max_proto_index = proto_indices[inst];
            }
            fprintf(stderr,
                    "scene_load: PointInstancer '%s': resolved %d/%d "
                    "prototype target(s), %d mesh binding(s), max protoIndex=%d\n",
                    _pi_full_path ? _pi_full_path : "?",
                    resolved_protos, n_protos, resolved_meshes,
                    max_proto_index);
        }

        /* Instancer world transform */
        double instancer_world[16];
        compute_world_xform(prim, instancer_world);

        /* Metal parity: bucket authored instances by proto index and
         * pre-compose each instance world transform once. The compact path then
         * iterates only the instances that belong to a batch, keeping Moana's
         * top-level PI work O(N * proto-submeshes) instead of O(N * batches). */
        double* inst_worlds = NULL;
        int* pidx_inst_offsets = NULL;
        int* pidx_inst_indices = NULL;
        int have_pi_buckets = 0;
        if (n_protos > 0) {
            inst_worlds = (double*)malloc((size_t)n_instances * 16 * sizeof(double));
            pidx_inst_offsets = (int*)malloc(((size_t)n_protos + 1) * sizeof(int));
            pidx_inst_indices = (int*)malloc((size_t)n_instances * sizeof(int));
            if (inst_worlds && pidx_inst_offsets && pidx_inst_indices) {
                for (int p = 0; p <= n_protos; p++) pidx_inst_offsets[p] = 0;
                for (int inst = 0; inst < n_instances; inst++) {
                    if (inst_visible && !inst_visible[inst]) continue;
                    int pidx = proto_indices[inst];
                    if (pidx >= 0 && pidx < n_protos)
                        pidx_inst_offsets[pidx + 1]++;
                }
                for (int p = 1; p <= n_protos; p++)
                    pidx_inst_offsets[p] += pidx_inst_offsets[p - 1];

                int* cursors = (int*)malloc((size_t)n_protos * sizeof(int));
                if (cursors) {
                    memcpy(cursors, pidx_inst_offsets,
                           (size_t)n_protos * sizeof(int));
                    for (int inst = 0; inst < n_instances; inst++) {
                        if (inst_visible && !inst_visible[inst]) continue;
                        int pidx = proto_indices[inst];
                        if (pidx >= 0 && pidx < n_protos)
                            pidx_inst_indices[cursors[pidx]++] = inst;
                    }
                    free(cursors);

                    for (int inst = 0; inst < n_instances; inst++) {
                        float qi=orientations[inst*4+0], qj=orientations[inst*4+1],
                              qk=orientations[inst*4+2], qw=orientations[inst*4+3];
                        float sx=scales[inst*3+0], sy=scales[inst*3+1],
                              sz=scales[inst*3+2];
                        double ix[16];
                        double ww=qw*qw, ii=qi*qi, jj=qj*qj, kk=qk*qk;
                        double wi=qw*qi, wj=qw*qj, wk=qw*qk;
                        double ij=qi*qj, ik=qi*qk, jk=qj*qk;
                        ix[0]=(ww+ii-jj-kk)*sx; ix[1]=2*(ij+wk)*sx;     ix[2]=2*(ik-wj)*sx;       ix[3]=0;
                        ix[4]=2*(ij-wk)*sy;     ix[5]=(ww-ii+jj-kk)*sy; ix[6]=2*(jk+wi)*sy;       ix[7]=0;
                        ix[8]=2*(ik+wj)*sz;     ix[9]=2*(jk-wi)*sz;     ix[10]=(ww-ii-jj+kk)*sz;  ix[11]=0;
                        ix[12]=positions[inst*3+0]; ix[13]=positions[inst*3+1]; ix[14]=positions[inst*3+2]; ix[15]=1;
                        nanousd_mul_m4d(ix, instancer_world,
                                         &inst_worlds[(size_t)inst * 16]);
                    }
                    have_pi_buckets = 1;
                }
            }
            if (!have_pi_buckets) {
                fprintf(stderr,
                        "scene_load: PointInstancer '%s': bucketed compact "
                        "emit unavailable, falling back to scan path\n",
                        nanousd_path(prim));
            }
        }

        int pi_count = 0;
        if (!compact_pi_only) {
        for (int inst = 0; inst < n_instances; inst++) {
            if (inst_visible && !inst_visible[inst]) continue;
            int pidx = proto_indices[inst];
            if (pidx < 0 || pidx >= n_protos ||
                proto_mesh_index_counts[pidx] == 0) continue;

          /* Build the instance transform once per instance — same
           * across the (typically small) per-proto mesh count.
           *
           * nanousd_attribarrayf returns USD quat values in (x, y, z, w)
           * order — imaginary components first, real last — even though
           * USD's `quath` text representation is written as (real,
           * imag.x, imag.y, imag.z). Reading as (w, x, y, z) here
           * silently turns every PointInstancer's identity quat
           * (1, 0, 0, 0) into a 180° Z-axis rotation, which flips and
           * mirrors every instanced mesh. (Caught while debugging the
           * sibling Metal renderer's OpenChessSet pawns appearing
           * below the board.) */
          float qi_q = orientations[inst*4+0];
          float qj_q = orientations[inst*4+1];
          float qk_q = orientations[inst*4+2];
          float w_q  = orientations[inst*4+3];
          float sx_q = scales[inst*3+0];
          float sy_q = scales[inst*3+1];
          float sz_q = scales[inst*3+2];
          float tx_q = positions[inst*3+0];
          float ty_q = positions[inst*3+1];
          float tz_q = positions[inst*3+2];
          double inst_xform[16];
          {
            double ww=w_q*w_q, ii=qi_q*qi_q, jj=qj_q*qj_q, kk=qk_q*qk_q;
            double wi=w_q*qi_q, wj=w_q*qj_q, wk=w_q*qk_q;
            double ij=qi_q*qj_q, ik=qi_q*qk_q, jk=qj_q*qk_q;
            inst_xform[0]  = (ww+ii-jj-kk)*sx_q; inst_xform[1]  = 2*(ij+wk)*sx_q;     inst_xform[2]  = 2*(ik-wj)*sx_q;     inst_xform[3]  = 0;
            inst_xform[4]  = 2*(ij-wk)*sy_q;      inst_xform[5]  = (ww-ii+jj-kk)*sy_q; inst_xform[6]  = 2*(jk+wi)*sy_q;     inst_xform[7]  = 0;
            inst_xform[8]  = 2*(ik+wj)*sz_q;      inst_xform[9]  = 2*(jk-wi)*sz_q;     inst_xform[10] = (ww-ii-jj+kk)*sz_q; inst_xform[11] = 0;
            inst_xform[12] = tx_q;                inst_xform[13] = ty_q;               inst_xform[14] = tz_q;               inst_xform[15] = 1;
          }
          double inst_world[16];
          nanousd_mul_m4d(inst_xform, instancer_world, inst_world);

         for (int pm_i = 0; pm_i < proto_mesh_index_counts[pidx]; pm_i++) {
            if (mesh_idx >= max_meshes) break;
            int src_mesh_idx = proto_mesh_index_lists[pidx][pm_i];
            SceneMesh* proto_m = &scene->meshes[src_mesh_idx];
            SceneMesh* mesh = &scene->meshes[mesh_idx];
            mesh->path = proto_m->path;

            /* Share geometry */
            mesh->positions  = proto_m->positions;
            mesh->normals    = proto_m->normals;
            mesh->indices    = proto_m->indices;
            mesh->nvertices  = proto_m->nvertices;
            mesh->nindices   = proto_m->nindices;
            mesh->prototype_idx = src_mesh_idx;
            mesh->has_display_color = proto_m->has_display_color;
            mesh->display_color[0] = proto_m->display_color[0];
            mesh->display_color[1] = proto_m->display_color[1];
            mesh->display_color[2] = proto_m->display_color[2];
            mesh->colors = proto_m->colors;
            mesh->texcoords = proto_m->texcoords;
            memcpy(mesh->local_bounds_min, proto_m->local_bounds_min, sizeof(float) * 3);
            memcpy(mesh->local_bounds_max, proto_m->local_bounds_max, sizeof(float) * 3);
            /* Carry the proto's material binding into the expanded
             * instance — without this, all PointInstancer-expanded
             * meshes default to material_index=0 (arena_calloc'd to
             * zero), so chess Pawns inherit the King material instead
             * of M_Pawn_Top_B/W. Surfaced when the transmission lobe
             * was wired and Pawn-tops still rendered as opaque marble. */
            mesh->material_index = proto_m->material_index;

            /* World transform = inst_xform * PI_world (= inst_world).
             * USD semantics: each proto-subtree mesh is materialized at
             * the instance's PI-local position. Most chess-set-style
             * scenes have proto_local = identity, so no extra factor is
             * needed. Renderers that want to honour a non-identity
             * proto-local transform should multiply by
             * proto_world * inverse(PI_world); we punt that to
             * follow-up since nanousd doesn't currently expose a 4x4
             * inverse. */
            memcpy(mesh->world_xform, inst_world, sizeof(double) * 16);

            /* Mark this scene mesh as a proper instance (drawn standalone),
             * even though its source mesh in scene->meshes is_proto_only. */
            mesh->is_proto_only = 0;

            /* Bounds */
            mesh_compute_world_bounds_from_local(mesh, scene);
            mesh->vertex_offset = 0;
            mesh->index_offset = 0;
            total_verts += mesh->nvertices;
            total_indices += mesh->nindices;
            mesh_idx++;
            instanced_meshes++;
            pi_scene_mesh_clones++;
            pi_count++;
          }  /* per-proto-mesh loop */
        }    /* per-instance loop */
        }

        if (pi_count > 0)
            fprintf(stderr, "scene_load: PointInstancer '%s': %d instances (%d protos, %d proto-meshes total)\n",
                    nanousd_path(prim), n_instances, n_protos, pi_count);

        /* Phase 1 (additive): emit the compact pi_transforms/pi_batches for this
         * PI, reusing the resolved protos + instance arrays still in scope. One
         * batch per (proto-index, proto-sub-mesh); its transform slice is the
         * inst_world of every instance with that proto-index — identical to the
         * clones' world_xform above, so Phase 2 can swap the renderer to batches
         * without re-deriving transforms. */
        for (int pidx = 0; pidx < n_protos; pidx++) {
            if (proto_mesh_index_counts[pidx] == 0) continue;
            int inst_lo = have_pi_buckets ? pidx_inst_offsets[pidx] : 0;
            int inst_hi = have_pi_buckets ? pidx_inst_offsets[pidx + 1] : 0;
            int kept = have_pi_buckets ? (inst_hi - inst_lo) : 0;
            if (!have_pi_buckets) {
                for (int inst = 0; inst < n_instances; inst++)
                    if ((!inst_visible || inst_visible[inst]) &&
                        proto_indices[inst] == pidx) kept++;
            }
            if (kept == 0) continue;

            uint32_t base_off = (uint32_t)scene->npi_transforms;
            SceneInstanceTransform* slab =
                scene_pi_transforms_reserve(scene, &pi_xform_cap, (uint64_t)kept);
            if (!slab) break;
            uint32_t w = 0;
            int loop_count = have_pi_buckets ? kept : n_instances;
            for (int li = 0; li < loop_count && w < (uint32_t)kept; li++) {
                int inst = have_pi_buckets ? pidx_inst_indices[inst_lo + li] : li;
                double iw_scan[16];
                const double* iw = NULL;
                if (have_pi_buckets) {
                    iw = &inst_worlds[(size_t)inst * 16];
                } else {
                    if (inst_visible && !inst_visible[inst]) continue;
                    if (proto_indices[inst] != pidx) continue;
                    float qi=orientations[inst*4+0], qj=orientations[inst*4+1],
                          qk=orientations[inst*4+2], qw=orientations[inst*4+3];
                    float sx=scales[inst*3+0], sy=scales[inst*3+1],
                          sz=scales[inst*3+2];
                    double ix[16];
                    double ww=qw*qw, ii=qi*qi, jj=qj*qj, kk=qk*qk;
                    double wi=qw*qi, wj=qw*qj, wk=qw*qk, ij=qi*qj, ik=qi*qk, jk=qj*qk;
                    ix[0]=(ww+ii-jj-kk)*sx; ix[1]=2*(ij+wk)*sx;    ix[2]=2*(ik-wj)*sx;      ix[3]=0;
                    ix[4]=2*(ij-wk)*sy;     ix[5]=(ww-ii+jj-kk)*sy; ix[6]=2*(jk+wi)*sy;      ix[7]=0;
                    ix[8]=2*(ik+wj)*sz;     ix[9]=2*(jk-wi)*sz;     ix[10]=(ww-ii-jj+kk)*sz; ix[11]=0;
                    ix[12]=positions[inst*3+0]; ix[13]=positions[inst*3+1]; ix[14]=positions[inst*3+2]; ix[15]=1;
                    nanousd_mul_m4d(ix, instancer_world, iw_scan);
                    iw = iw_scan;
                }
                SceneInstanceTransform* t = &slab[w++];
                t->m[0]=(float)iw[0];  t->m[1]=(float)iw[1];  t->m[2]=(float)iw[2];
                t->m[3]=(float)iw[4];  t->m[4]=(float)iw[5];  t->m[5]=(float)iw[6];
                t->m[6]=(float)iw[8];  t->m[7]=(float)iw[9];  t->m[8]=(float)iw[10];
                t->m[9]=(float)iw[12]; t->m[10]=(float)iw[13]; t->m[11]=(float)iw[14];
            }

            for (int pm_i = 0; pm_i < proto_mesh_index_counts[pidx]; pm_i++) {
                int src_mesh_idx = proto_mesh_index_lists[pidx][pm_i];
                SceneMesh* proto_m = (src_mesh_idx >= 0 && src_mesh_idx < mesh_idx)
                    ? &scene->meshes[src_mesh_idx] : NULL;
                float local_min[3], local_max[3];
                int have_local_bounds = mesh_local_bounds(proto_m, local_min, local_max);
                for (int li = 0; li < loop_count; li++) {
                    int inst = have_pi_buckets ? pidx_inst_indices[inst_lo + li] : li;
                    double iw_scan[16];
                    const double* iw = NULL;
                    if (have_pi_buckets) {
                        iw = &inst_worlds[(size_t)inst * 16];
                    } else {
                        if (inst_visible && !inst_visible[inst]) continue;
                        if (proto_indices[inst] != pidx) continue;
                        float qi=orientations[inst*4+0], qj=orientations[inst*4+1],
                              qk=orientations[inst*4+2], qw=orientations[inst*4+3];
                        float sx=scales[inst*3+0], sy=scales[inst*3+1],
                              sz=scales[inst*3+2];
                        double ix[16];
                        double ww=qw*qw, ii=qi*qi, jj=qj*qj, kk=qk*qk;
                        double wi=qw*qi, wj=qw*qj, wk=qw*qk, ij=qi*qj, ik=qi*qk, jk=qj*qk;
                        ix[0]=(ww+ii-jj-kk)*sx; ix[1]=2*(ij+wk)*sx;    ix[2]=2*(ik-wj)*sx;      ix[3]=0;
                        ix[4]=2*(ij-wk)*sy;     ix[5]=(ww-ii+jj-kk)*sy; ix[6]=2*(jk+wi)*sy;      ix[7]=0;
                        ix[8]=2*(ik+wj)*sz;     ix[9]=2*(jk-wi)*sz;     ix[10]=(ww-ii-jj+kk)*sz; ix[11]=0;
                        ix[12]=positions[inst*3+0]; ix[13]=positions[inst*3+1]; ix[14]=positions[inst*3+2]; ix[15]=1;
                        nanousd_mul_m4d(ix, instancer_world, iw_scan);
                        iw = iw_scan;
                    }
                    if (have_local_bounds) {
                        float world_min[3], world_max[3];
                        world_bounds_from_local(iw, local_min, local_max,
                                                world_min, world_max);
                        if (bounds_is_usable(world_min, world_max))
                            scene_expand_bounds(scene, world_min, world_max);
                    } else if (proto_m && proto_m->positions) {
                        float world_min[3], world_max[3];
                        bounds_init(world_min, world_max);
                        for (int v = 0; v < proto_m->nvertices; v++) {
                            float wp[3];
                            xform_point(iw, &proto_m->positions[v * 3], wp);
                            for (int a = 0; a < 3; a++) {
                                if (wp[a] < world_min[a]) world_min[a] = wp[a];
                                if (wp[a] > world_max[a]) world_max[a] = wp[a];
                            }
                        }
                        if (bounds_is_usable(world_min, world_max))
                            scene_expand_bounds(scene, world_min, world_max);
                    }
                }
                if (!scene_pi_batches_reserve(scene, &pi_batch_cap, scene->npi_batches + 1))
                    break;
                SceneInstanceBatch* b = &scene->pi_batches[scene->npi_batches++];
                b->prototype_mesh_idx     = src_mesh_idx;
                b->transform_offset       = base_off;
                b->transform_count        = (uint32_t)kept;
                b->source_prim_idx        = i;
                b->material_or_binding_id = -1;
                b->source_kind            = SCENE_INSTANCE_SOURCE_POINT_INSTANCER;
            }
        }

        free(inst_worlds);
        free(pidx_inst_offsets);
        free(pidx_inst_indices);
        pi_proto_targets_free(&proto_targets);
        free(inst_visible);
        free(proto_indices);
        free(positions);
        free(orientations);
        free(scales);
        for (int p = 0; p < n_protos; p++) free(proto_mesh_index_lists[p]);
        free(proto_mesh_index_lists);
        free(proto_mesh_index_counts);
        nanousd_freeprim(prim);
    }

    proto_hash_free(&proto_mesh_table);
    #undef PRIM_IS_FILTER_CONTROLLER
    #undef PRIM_IS_WANTED

    /* Set actual mesh count (over-allocated above using nprims as upper bound) */
    scene->nmeshes = mesh_idx;

    if (scene->nmeshes == 0) {
        fprintf(stderr, "scene_load: no Mesh prims found in '%s' (%d prims total)\n",
                filepath, nprims);
        /* Return empty scene instead of NULL — caller may add meshes via add_mesh() */
    }

    if (do_timing) {
        double t_now = get_time_sec();
        fprintf(stderr,
                "  [scene_timing] point-instancer expansion: %.1f ms "
                "(meshes=%d, instanced=%d)\n",
                (t_now - t_mark) * 1000.0,
                mesh_idx, instanced_meshes);
        t_mark = t_now;
    }

    if (scene->npi_batches > 0 || scene->npi_transforms > 0)
        fprintf(stderr,
                "scene_load: compact PI — %d batches, %llu transforms "
                "(%.1f MiB); pi_scene_mesh_clones=%lld%s\n",
                scene->npi_batches, (unsigned long long)scene->npi_transforms,
                (double)(scene->npi_transforms * sizeof(SceneInstanceTransform)) / (1024.0*1024.0),
                pi_scene_mesh_clones,
                compact_pi_only ? "" : " (legacy mode)");

    /* ---- Extract scene lights (UsdLuxRectLight, UsdLuxDiskLight,
     * UsdLuxDistantLight, UsdLuxSphereLight) ---- */
    {
        int nlights = 0;
        for (int i = 0; i < nprims; i++) {
            NanousdPrim p = nanousd_prim(stage, i);
            if (!p) continue;
            if (nanousd_isactive(p)) {
                const char* tn = nanousd_typename(p);
                if (tn && strstr(tn, "Light") != NULL)
                    scene->has_authored_light = 1;
                if (tn && (!strcmp(tn, "RectLight") ||
                           !strcmp(tn, "DiskLight") ||
                           !strcmp(tn, "CylinderLight") ||
                           !strcmp(tn, "DistantLight") ||
                           !strcmp(tn, "SphereLight"))) {
                    nlights++;
                }
            }
            nanousd_freeprim(p);
        }

        if (nlights > 0) {
            scene->lights = (SceneLight*)arena_calloc(arena, (size_t)nlights, sizeof(SceneLight));
        }
        if (scene->lights) {
            int li = 0;
            for (int i = 0; i < nprims && li < nlights; i++) {
                NanousdPrim p = nanousd_prim(stage, i);
                if (!p) continue;
                if (!nanousd_isactive(p)) { nanousd_freeprim(p); continue; }
                const char* tn = nanousd_typename(p);
                int is_rect = (tn && !strcmp(tn, "RectLight"));
                int is_disk = (tn && !strcmp(tn, "DiskLight"));
                int is_cylinder = (tn && !strcmp(tn, "CylinderLight"));
                int is_dist = (tn && !strcmp(tn, "DistantLight"));
                int is_sphere = (tn && !strcmp(tn, "SphereLight"));
                if (!is_rect && !is_disk && !is_cylinder && !is_dist && !is_sphere) { nanousd_freeprim(p); continue; }

                SceneLight* L = &scene->lights[li];
                int ok;

                L->color[0] = L->color[1] = L->color[2] = 1.0f;
                if (!nanousd_attribv3f(p, "inputs:color", L->color))
                    nanousd_attribv3f(p, "color", L->color);

                float intensity = nanousd_attribf(p, "inputs:intensity", &ok);
                if (!ok) intensity = nanousd_attribf(p, "intensity", &ok);
                if (!ok) intensity = 1.0f;
                float exposure = nanousd_attribf(p, "inputs:exposure", &ok);
                if (!ok) exposure = nanousd_attribf(p, "exposure", &ok);
                if (!ok) exposure = 0.0f;
                L->intensity = intensity * powf(2.0f, exposure);

                int normalize = nanousd_attribb(p, "inputs:normalize", &ok);
                if (!ok) normalize = nanousd_attribb(p, "normalize", &ok);
                if (!ok) normalize = nanousd_attribi(p, "inputs:normalize", &ok);
                if (!ok) normalize = nanousd_attribi(p, "normalize", &ok);
                L->normalize = ok ? normalize : 0;

                double M[16];
                compute_world_xform(p, M);
                L->position[0] = (float)M[12];
                L->position[1] = (float)M[13];
                L->position[2] = (float)M[14];

                if (is_rect || is_disk || is_cylinder) {
                    L->kind = SCENE_LIGHT_RECT;
                    float width = 1.0f, height = 1.0f;
                    const char* light_path = nanousd_path(p);
                    if (is_disk) {
                        float radius = nanousd_attribf(p, "inputs:radius", &ok);
                        if (!ok) radius = nanousd_attribf(p, "radius", &ok);
                        if (!ok) radius = 0.5f;
                        /* Approximate the disk as a square RectLight with
                         * equal emitting area: width * height = pi*r^2. */
                        width = height = sqrtf(3.14159265358979323846f) * radius;
                    } else if (is_cylinder) {
                        float radius = nanousd_attribf(p, "inputs:radius", &ok);
                        if (!ok) radius = nanousd_attribf(p, "radius", &ok);
                        if (!ok) radius = 0.5f;
                        float length = nanousd_attribf(p, "inputs:length", &ok);
                        if (!ok) length = nanousd_attribf(p, "length", &ok);
                        if (!ok) length = 1.0f;
                        /* Approximate CylinderLight with its projected
                         * silhouette in the existing RectLight evaluator.
                         * This preserves authored intensity/color/exposure
                         * and avoids silently dropping the light. */
                        width = length;
                        height = 2.0f * radius;
                    } else {
                        width = nanousd_attribf(p, "inputs:width", &ok);
                        if (!ok) width = nanousd_attribf(p, "width", &ok);
                        if (!ok) width = 1.0f;
                        height = nanousd_attribf(p, "inputs:height", &ok);
                        if (!ok) height = nanousd_attribf(p, "height", &ok);
                        if (!ok) height = 1.0f;
                    }
                    if (scene_rect_light_cm_dimensions(filepath, light_path,
                                                       scene, width, height)) {
                        float authored_width = width;
                        float authored_height = height;
                        width *= 0.01f;
                        height *= 0.01f;
                        if (getenv("NUSD_LIGHT_DIAG")) {
                            fprintf(stderr,
                                    "scene_load: rect light cm dimensions %s "
                                    "%.3fx%.3f -> %.3fx%.3f\n",
                                    light_path ? light_path : "<unknown>",
                                    authored_width, authored_height,
                                    width, height);
                        }
                    }
                    L->u_axis[0] = (float)M[0] * 0.5f * width;
                    L->u_axis[1] = (float)M[1] * 0.5f * width;
                    L->u_axis[2] = (float)M[2] * 0.5f * width;
                    L->v_axis[0] = (float)M[4] * 0.5f * height;
                    L->v_axis[1] = (float)M[5] * 0.5f * height;
                    L->v_axis[2] = (float)M[6] * 0.5f * height;
                    L->normal[0] = -(float)M[8];
                    L->normal[1] = -(float)M[9];
                    L->normal[2] = -(float)M[10];
                    L->angle_deg = 0.0f;
                } else if (is_sphere) {
                    /* SphereLight: emitter at world `position` with `radius`
                     * stored in u_axis[0] (v_axis/normal unused — see scene.h).
                     * The shader uses 1/r² falloff and area-normalises when
                     * inputs:normalize=1. */
                    L->kind = SCENE_LIGHT_SPHERE;
                    float radius = nanousd_attribf(p, "inputs:radius", &ok);
                    if (!ok) radius = nanousd_attribf(p, "radius", &ok);
                    if (!ok) radius = 0.5f;
                    L->u_axis[0] = radius;
                    L->u_axis[1] = L->u_axis[2] = 0.0f;
                    L->v_axis[0] = L->v_axis[1] = L->v_axis[2] = 0.0f;
                    L->normal[0] = L->normal[1] = L->normal[2] = 0.0f;
                    L->angle_deg = 0.0f;
                } else {
                    L->kind = SCENE_LIGHT_DISTANT;
                    L->normal[0] = -(float)M[8];
                    L->normal[1] = -(float)M[9];
                    L->normal[2] = -(float)M[10];
                    L->u_axis[0] = L->u_axis[1] = L->u_axis[2] = 0.0f;
                    L->v_axis[0] = L->v_axis[1] = L->v_axis[2] = 0.0f;
                    float ang = nanousd_attribf(p, "inputs:angle", &ok);
                    if (!ok) ang = nanousd_attribf(p, "angle", &ok);
                    L->angle_deg = ok ? ang : 0.53f;
                }

                {
                    float nx = L->normal[0], ny = L->normal[1], nz = L->normal[2];
                    float len = sqrtf(nx*nx + ny*ny + nz*nz);
                    if (len > 1e-12f) {
                        L->normal[0] = nx / len;
                        L->normal[1] = ny / len;
                        L->normal[2] = nz / len;
                    }
                }

                li++;
                nanousd_freeprim(p);
            }
            scene->nlights = li;
            fprintf(stderr, "scene_load: %d lights (rect/disk+distant+sphere) extracted\n", li);
        }
    }

    /* ---- Extract UsdLuxDomeLight ----
     * Single dome only - first found wins, matching Hydra's pick. The HDR
     * path stays in scene->dome_hdr_path; nu_attach_scene picks it up and
     * calls gpu_load_environment to install it as IBL. Without this the
     * renderer falls back to the procedural sky in raytrace.rchit.glsl. */
    {
        scene->has_dome_light   = 0;
        scene->dome_hdr_path[0] = '\0';
        scene->dome_color[0]    = 1.0f;
        scene->dome_color[1]    = 1.0f;
        scene->dome_color[2]    = 1.0f;
        scene->dome_intensity   = 1.0f;
        scene->dome_rotation_y  = 0.0f;
        for (int i = 0; i < nprims; i++) {
            NanousdPrim p = nanousd_prim(stage, i);
            if (!p) continue;
            if (!nanousd_isactive(p)) { nanousd_freeprim(p); continue; }
            const char* tn = nanousd_typename(p);
            if (!tn || strcmp(tn, "DomeLight") != 0) {
                nanousd_freeprim(p); continue;
            }
            scene->has_authored_light = 1;
            scene->has_dome_light = 1;
            int ok;
            if (!nanousd_attribv3f(p, "inputs:color", scene->dome_color))
                nanousd_attribv3f(p, "color", scene->dome_color);
            const char* asset = nanousd_attribasset(p, "inputs:texture:file", &ok);
            if (ok && asset && asset[0]) {
                /* USD asset paths can be @relative@ or absolute; nanousd
                 * stores them already-resolved when possible. We treat
                 * non-absolute paths as relative to the USD file's parent
                 * directory. */
                const char* resolved = asset;
                char buf[512];
                if (asset[0] != '/') {
                    const char* slash = strrchr(filepath, '/');
                    if (slash) {
                        size_t base_len = (size_t)(slash - filepath + 1);
                        int n = snprintf(buf, sizeof(buf), "%.*s%s",
                                         (int)base_len, filepath, asset);
                        if (n > 0 && (size_t)n < sizeof(buf)) {
                            resolved = buf;
                        }
                    }
                }
                snprintf(scene->dome_hdr_path, sizeof(scene->dome_hdr_path), "%s", resolved);
            }
            float intensity = nanousd_attribf(p, "inputs:intensity", &ok);
            if (!ok) intensity = nanousd_attribf(p, "intensity", &ok);
            if (!ok) intensity = 1.0f;
            float exposure = nanousd_attribf(p, "inputs:exposure", &ok);
            if (!ok) exposure = nanousd_attribf(p, "exposure", &ok);
            if (!ok) exposure = 0.0f;
            scene->dome_intensity = intensity * powf(2.0f, exposure);
            /* DomeLight Y-rotation lives in xformOp:rotateXYZ when authored;
             * we accept the simpler explicit attribute for now and leave
             * full xformOp evaluation to a follow-up. */
            float rot = nanousd_attribf(p, "inputs:rotation", &ok);
            if (ok) scene->dome_rotation_y = rot;
            nanousd_freeprim(p);
            if (scene->dome_hdr_path[0]) {
                fprintf(stderr,
                        "scene_load: DomeLight -> %s "
                        "(color=%.3f,%.3f,%.3f intensity=%.3f)\n",
                        scene->dome_hdr_path,
                        scene->dome_color[0],
                        scene->dome_color[1],
                        scene->dome_color[2],
                        scene->dome_intensity);
            } else {
                fprintf(stderr,
                        "scene_load: DomeLight textureless "
                        "(color=%.3f,%.3f,%.3f intensity=%.3f)\n",
                        scene->dome_color[0],
                        scene->dome_color[1],
                        scene->dome_color[2],
                        scene->dome_intensity);
            }
            break;  /* first dome wins, even without an HDR texture */
        }
    }

    /* ---- upAxis handling ----
     *
     * Match OVRTX/USD composition semantics by leaving geometry and lights in
     * the scene's authored coordinate frame. Older builds baked Z-up scenes
     * into Y-up here; that makes identical eye/target camera inputs point at
     * different world-space content than OVRTX. Keep the old bake available as
     * an explicit compatibility knob for tools that were authored around it. */
    {
        int ok = 0;
        const char* up = nanousd_metadatas(stage, "upAxis", &ok);
        /* upAxis metadata: ok=%d means the value was found in the stage */

        /* Heuristic: if metadata is missing but Z range is < XY range and
         * Z_min ≈ 0, treat as Z-up (common for Omniverse / Isaac Sim assets). */
        int detected_up_axis = 1;
        if (ok && up && (up[0] == 'Z' || up[0] == 'z')) {
            detected_up_axis = 2;
        } else if (ok && up && (up[0] == 'X' || up[0] == 'x')) {
            detected_up_axis = 0;
        }
        int is_z_up = detected_up_axis == 2;
        if (!is_z_up && !ok) {
            float dx = scene->bounds_max[0] - scene->bounds_min[0];
            float dy = scene->bounds_max[1] - scene->bounds_min[1];
            float dz = scene->bounds_max[2] - scene->bounds_min[2];
            float max_xy = dx > dy ? dx : dy;
            if (dz > 1e-3f && dz < max_xy * 0.8f && scene->bounds_min[2] > -dz * 0.1f) {
                fprintf(stderr, "scene_load: heuristic Z-up detection (dz=%.1f < max_xy=%.1f, z_min=%.1f)\n",
                        dz, max_xy, scene->bounds_min[2]);
                is_z_up = 1;
                detected_up_axis = 2;
            }
        }
        scene->up_axis = detected_up_axis;

        const char* bake_env = getenv("NUSD_BAKE_ZUP_TO_YUP");
        int bake_zup_to_yup = bake_env && bake_env[0] && bake_env[0] != '0';
        const char* preserve_env = getenv("NUSD_PRESERVE_UPAXIS");
        if (preserve_env && preserve_env[0] && preserve_env[0] != '0')
            bake_zup_to_yup = 0;

        if (is_z_up && bake_zup_to_yup) {
            /* Z-up → Y-up: rotate -90° around X.
             * In row-vector convention (p' = p * R):
             *   R = [1  0  0  0]   maps (x,y,z) → (x, z, -y)
             *       [0  0 -1  0]
             *       [0  1  0  0]
             *       [0  0  0  1]
             * We post-multiply each world_xform: W' = W * R */
            static const double R[16] = {
                1,  0,  0,  0,
                0,  0, -1,  0,
                0,  1,  0,  0,
                0,  0,  0,  1
            };
            for (int m = 0; m < scene->nmeshes; m++) {
                SceneMesh* mesh = &scene->meshes[m];
                double tmp[16];
                nanousd_mul_m4d(mesh->world_xform, R, tmp);
                memcpy(mesh->world_xform, tmp, sizeof(double) * 16);

                /* Recompute per-mesh bounds in the new coordinate system */
                float omin[3], omax[3];
                memcpy(omin, mesh->bounds_min, sizeof(float) * 3);
                memcpy(omax, mesh->bounds_max, sizeof(float) * 3);
                /* (x, z, -y) of the original corners */
                mesh->bounds_min[0] = omin[0];
                mesh->bounds_min[1] = omin[2];
                mesh->bounds_min[2] = -omax[1];
                mesh->bounds_max[0] = omax[0];
                mesh->bounds_max[1] = omax[2];
                mesh->bounds_max[2] = -omin[1];
            }

            /* Recompute scene bounds */
            scene->bounds_min[0] = FLT_MAX;
            scene->bounds_min[1] = FLT_MAX;
            scene->bounds_min[2] = FLT_MAX;
            scene->bounds_max[0] = -FLT_MAX;
            scene->bounds_max[1] = -FLT_MAX;
            scene->bounds_max[2] = -FLT_MAX;
            for (int m = 0; m < scene->nmeshes; m++) {
                SceneMesh* mesh = &scene->meshes[m];
                for (int k = 0; k < 3; k++) {
                    if (mesh->bounds_min[k] < scene->bounds_min[k])
                        scene->bounds_min[k] = mesh->bounds_min[k];
                    if (mesh->bounds_max[k] > scene->bounds_max[k])
                        scene->bounds_max[k] = mesh->bounds_max[k];
                }
            }

            /* Rotate scene lights into Y-up too: (x, y, z) → (x, z, -y) */
            for (int li = 0; li < scene->nlights; li++) {
                SceneLight* L = &scene->lights[li];
                #define _ZUP_TO_YUP(v) do { float _y = v[1]; v[1] = v[2]; v[2] = -_y; } while (0)
                _ZUP_TO_YUP(L->position);
                _ZUP_TO_YUP(L->normal);
                _ZUP_TO_YUP(L->u_axis);
                _ZUP_TO_YUP(L->v_axis);
                #undef _ZUP_TO_YUP
            }

            fprintf(stderr, "scene_load: applied Z-up → Y-up rotation\n");
            scene->up_axis = 1;
        }
    }

    /* ---- metersPerUnit handling ----
     * Keep authored numeric coordinates by default for OVRTX parity.
     * NUSD_APPLY_METERS_PER_UNIT=1 enables the legacy normalization path.
     */
    {
        int ok = 0;
        double mpu = nanousd_metadatad(stage, "metersPerUnit", &ok);
        const char* units_env = getenv("NUSD_APPLY_METERS_PER_UNIT");
        int apply_units = units_env && units_env[0] && units_env[0] != '0';
        if (apply_units && ok && mpu > 0.0 && mpu != 1.0) {
            for (int m = 0; m < scene->nmeshes; m++) {
                SceneMesh* mesh = &scene->meshes[m];
                for (int i = 0; i < 12; i++) mesh->world_xform[i] *= mpu;
                mesh->world_xform[12] *= mpu;
                mesh->world_xform[13] *= mpu;
                mesh->world_xform[14] *= mpu;
                for (int k = 0; k < 3; k++) {
                    mesh->bounds_min[k] = (float)(mesh->bounds_min[k] * mpu);
                    mesh->bounds_max[k] = (float)(mesh->bounds_max[k] * mpu);
                }
            }
            for (int k = 0; k < 3; k++) {
                scene->bounds_min[k] = (float)(scene->bounds_min[k] * mpu);
                scene->bounds_max[k] = (float)(scene->bounds_max[k] * mpu);
            }
            for (int c = 0; c < scene->ncurves; c++) {
                SceneCurve* cu = &scene->curves[c];
                for (int i = 0; i < 12; i++) cu->world_xform[i] *= mpu;
                cu->world_xform[12] *= mpu;
                cu->world_xform[13] *= mpu;
                cu->world_xform[14] *= mpu;
                for (int v = 0; v < cu->nv; v++)
                    cu->widths[v] = (float)(cu->widths[v] * mpu);
                for (int k = 0; k < 3; k++) {
                    cu->bounds_min[k] = (float)(cu->bounds_min[k] * mpu);
                    cu->bounds_max[k] = (float)(cu->bounds_max[k] * mpu);
                }
            }
            for (int li = 0; li < scene->nlights; li++) {
                SceneLight* L = &scene->lights[li];
                for (int k = 0; k < 3; k++) {
                    L->position[k] = (float)(L->position[k] * mpu);
                    L->u_axis[k]   = (float)(L->u_axis[k] * mpu);
                    L->v_axis[k]   = (float)(L->v_axis[k] * mpu);
                }
            }
            fprintf(stderr, "scene_load: applied metersPerUnit=%.6f scaling\n", mpu);
        }
    }

    /* ---- Robust bounds: reject outlier meshes using cached per-mesh bounds ---- */
    if (scene->nmeshes > 4) {
        /* Compute per-mesh center from cached bounds (no vertex re-traversal) */
        float* cx = (float*)malloc((size_t)scene->nmeshes * 3 * sizeof(float));
        if (cx) {
            for (int m = 0; m < scene->nmeshes; m++) {
                SceneMesh* mesh = &scene->meshes[m];
                cx[m * 3 + 0] = (mesh->bounds_min[0] + mesh->bounds_max[0]) * 0.5f;
                cx[m * 3 + 1] = (mesh->bounds_min[1] + mesh->bounds_max[1]) * 0.5f;
                cx[m * 3 + 2] = (mesh->bounds_min[2] + mesh->bounds_max[2]) * 0.5f;
            }

            /* Median of mesh centers per axis */
            float median[3];
            {
                float* tmp = (float*)malloc((size_t)scene->nmeshes * sizeof(float));
                if (tmp) {
                    for (int k = 0; k < 3; k++) {
                        for (int m = 0; m < scene->nmeshes; m++)
                            tmp[m] = cx[m * 3 + k];
                        qsort(tmp, (size_t)scene->nmeshes, sizeof(float), float_cmp);
                        median[k] = tmp[scene->nmeshes / 2];
                    }
                    free(tmp);
                }
            }

            /* MAD (median absolute deviation) of distances from median */
            float* dists = (float*)malloc((size_t)scene->nmeshes * sizeof(float));
            if (dists) {
                for (int m = 0; m < scene->nmeshes; m++) {
                    float dx = cx[m*3+0]-median[0], dy = cx[m*3+1]-median[1], dz = cx[m*3+2]-median[2];
                    dists[m] = sqrtf(dx*dx + dy*dy + dz*dz);
                }
                /* Find median distance */
                qsort(dists, (size_t)scene->nmeshes, sizeof(float), float_cmp);
                float med_dist = dists[scene->nmeshes / 2];
                /* Threshold: meshes beyond 5x median distance are outliers */
                float threshold = (med_dist > 0.001f) ? med_dist * 5.0f : 1.0f;

                /* Recompute bounds excluding outliers using cached per-mesh bounds */
                float new_min[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
                float new_max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
                int included = 0;
                for (int m = 0; m < scene->nmeshes; m++) {
                    float dx = cx[m*3+0]-median[0], dy = cx[m*3+1]-median[1], dz = cx[m*3+2]-median[2];
                    float d = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (d > threshold) continue;
                    included++;
                    SceneMesh* mesh = &scene->meshes[m];
                    for (int k = 0; k < 3; k++) {
                        if (mesh->bounds_min[k] < new_min[k]) new_min[k] = mesh->bounds_min[k];
                        if (mesh->bounds_max[k] > new_max[k]) new_max[k] = mesh->bounds_max[k];
                    }
                }

                if (included > 0 && included < scene->nmeshes) {
                    float old_diag = sqrtf(
                        (scene->bounds_max[0]-scene->bounds_min[0])*(scene->bounds_max[0]-scene->bounds_min[0]) +
                        (scene->bounds_max[1]-scene->bounds_min[1])*(scene->bounds_max[1]-scene->bounds_min[1]) +
                        (scene->bounds_max[2]-scene->bounds_min[2])*(scene->bounds_max[2]-scene->bounds_min[2]));
                    float new_diag = sqrtf(
                        (new_max[0]-new_min[0])*(new_max[0]-new_min[0]) +
                        (new_max[1]-new_min[1])*(new_max[1]-new_min[1]) +
                        (new_max[2]-new_min[2])*(new_max[2]-new_min[2]));
                    /* Only apply if we'd significantly shrink the bounds */
                    if (new_diag < old_diag * 0.5f) {
                        memcpy(scene->bounds_min, new_min, sizeof(float) * 3);
                        memcpy(scene->bounds_max, new_max, sizeof(float) * 3);
                        fprintf(stderr, "scene_load: ignored %d outlier mesh bounds (scene bounds shrunk %.0fx)\n",
                                scene->nmeshes - included, old_diag / new_diag);
                    }
                }
                free(dists);
            }
            free(cx);
        }
    }

lazy_postprocess:
    if (do_timing) {
        double t_now = get_time_sec();
        fprintf(stderr, "  [scene_timing] scene postprocess: %.1f ms (nmeshes=%d)\n",
            (t_now - t_mark) * 1000.0, scene->nmeshes);
        t_mark = t_now;
    }

    /* ---- Load BasisCurves prims (Phase 11.A) ----
     * The mesh/lazy prim walk above already visits every active flat-list
     * prim and records whether any BasisCurves exist. Reuse that result so
     * mesh-only scenes do not pay a second O(nprims) type scan just to return
     * zero curves. */
    if (stage_has_basis_curves) {
        scene_load_curves(stage, scene, arena);
    } else if (do_timing) {
        fprintf(stderr,
                "scene_load: skipped BasisCurves flat scan "
                "(none seen during prim walk)\n");
    }
    scene_load_native_instance_curves(stage, scene, arena);
    if (do_timing) {
        double t_now = get_time_sec();
        fprintf(stderr, "  [scene_timing] scene_load_curves: %.1f ms (ncurves=%d)\n",
            (t_now - t_mark) * 1000.0, scene->ncurves);
        t_mark = t_now;
    }

    double t1 = get_time_sec();
    if (scene->nmeshes > 0 || scene->ncurves > 0) {
        fprintf(stderr, "scene_load: '%s' loaded in %.1f ms\n"
                        "  %d meshes, %" PRIu64 " vertices, %" PRIu64 " indices, %d curves\n"
                        "  bounds: (%.3f, %.3f, %.3f) - (%.3f, %.3f, %.3f)\n",
                filepath, (t1 - t0) * 1000.0,
                scene->nmeshes, total_verts, total_indices, scene->ncurves,
                scene->bounds_min[0], scene->bounds_min[1], scene->bounds_min[2],
                scene->bounds_max[0], scene->bounds_max[1], scene->bounds_max[2]);
    } else {
        fprintf(stderr, "scene_load: '%s' loaded in %.1f ms (0 meshes, 0 curves, %d prims)\n",
                filepath, (t1 - t0) * 1000.0, nprims);
    }
    if (instanced_meshes > 0) {
        int unique_protos = scene->nmeshes - instanced_meshes;
        fprintf(stderr, "  instancing: %d instances + %d unique = %d total meshes (%.0f%% shared)\n",
                instanced_meshes, unique_protos, scene->nmeshes,
                (double)instanced_meshes / (double)scene->nmeshes * 100.0);
    }

    path_prefix_list_free(&invisible_prefixes);
    path_prefix_list_free(&purpose_hidden_prefixes);
    path_prefix_list_free(&point_instancer_prefixes);
    xform_cache_free(&xform_cache);
    g_xform_cache = NULL;
    free(world_prewalk);

    return scene;
}

/* ----------------------------------------------------------------
 * scene_release_mesh_payloads
 *
 * Drop the per-mesh CPU geometry payload (positions/normals/colors/
 * texcoords/indices) and destroy the geometry arena. The renderer's GPU
 * vertex/index buffers, BLAS/TLAS, materials, and per-mesh scalar fields
 * (material_index, transform, bounds, prototype_idx) are unaffected.
 *
 * Vulkan-specific notes vs OpenGL's port:
 *   - scene->meshes is always arena-allocated here (unlike OpenGL which
 *     calloc's it for big scenes), so we ALWAYS copy meshes to a fresh
 *     heap buffer before destroying the arena.
 *   - scene->lights is also arena-allocated. Lights are consumed at
 *     renderer attach (copied into a GPU SSBO at renderer.c:1338-1357)
 *     and never read from CPU again — so we drop them here and zero
 *     the count.
 *   - We refuse if scene->ncurves > 0 (curves keep CPU data for
 *     procedural intersection AABBs).
 *   - We DO NOT close the nanousd stage in v1: the pxr_compat shim
 *     and the materials collection may still reference stage strings.
 *     A follow-up PR can close the stage once that is audited.
 * ---------------------------------------------------------------- */
int scene_release_mesh_payloads(Scene* scene)
{
    if (!scene) return 0;
    if (scene->ncurves > 0) return 0;
    if (scene->_meshes_heap) return 0;   /* already released */
    if (!scene->_arena)      return 0;   /* nothing to free */

    /* Copy mesh structs to heap; NULL the payload pointers in the copy
     * (the underlying arena will be destroyed below, invalidating the
     * old payload pointers). */
    SceneMesh* heap_meshes = NULL;
    if (scene->nmeshes > 0) {
        heap_meshes = (SceneMesh*)malloc(
            (size_t)scene->nmeshes * sizeof(SceneMesh));
        if (!heap_meshes) return 0;
        memcpy(heap_meshes, scene->meshes,
               (size_t)scene->nmeshes * sizeof(SceneMesh));
        for (int i = 0; i < scene->nmeshes; i++) {
            heap_meshes[i].positions = NULL;
            heap_meshes[i].normals   = NULL;
            heap_meshes[i].colors    = NULL;
            heap_meshes[i].texcoords = NULL;
            heap_meshes[i].indices   = NULL;
            heap_meshes[i].ptex_tri_colors = NULL;
            heap_meshes[i].ptex_tri_color_count = 0;
            heap_meshes[i].path      = NULL;
        }
    }

    /* Lights live in the arena too. Renderer has already consumed them
     * into a GPU SSBO at attach; nothing reads scene->lights past that
     * point. Drop the pointer and zero the count so any stray reader
     * fails fast instead of dereferencing freed memory. */
    scene->lights  = NULL;
    scene->nlights = 0;

    /* Destroy the arena. Invalidates: scene->meshes (old), scene->lights,
     * scene->curves (already gated to 0 above), and every per-mesh
     * positions/normals/colors/texcoords/indices pointer. */
    arena_destroy((Arena*)scene->_arena);
    free(scene->_arena);
    scene->_arena = NULL;

    /* Swap in the heap-allocated mesh array. */
    scene->meshes       = heap_meshes;
    scene->_meshes_heap = 1;
    return 1;
}

/* ----------------------------------------------------------------
 * scene_free
 * ---------------------------------------------------------------- */
void scene_free(Scene* scene)
{
    if (!scene) return;

    /* Free materials (before closing stage — material data may reference stage) */
    if (scene->materials) {
        materials_free((MaterialCollection*)scene->materials);
        scene->materials = NULL;
    }

    /* Close the USD stage (invalidates zero-copy pointers) — only if we
     * own it. Stages handed in via scene_load_from_stage are owned by the
     * caller (e.g. the Python pxr_compat shim) and must not be closed. */
    if (scene->_stage && scene->_owns_stage) {
        nanousd_close((NanousdStage)scene->_stage);
    }
    scene->_stage = NULL;

    /* Destroy the arena (frees all mesh data allocated from it). If
     * scene_release_mesh_payloads already destroyed the arena, scene->meshes
     * was lifted out to the heap and needs free() rather than dangling. */
    if (scene->_arena) {
        arena_destroy((Arena*)scene->_arena);
        free(scene->_arena);
        scene->_arena = NULL;
    }
    if (scene->_meshes_heap && scene->meshes) {
        free(scene->meshes);
    }
    scene->meshes = NULL;
    scene->nmeshes = 0;
    scene->_meshes_heap = 0;

    /* Compact PointInstancer batches (heap, not arena). */
    free(scene->pi_batches);    scene->pi_batches = NULL;    scene->npi_batches = 0;
    free(scene->pi_transforms); scene->pi_transforms = NULL; scene->npi_transforms = 0;

    free(scene);
}
