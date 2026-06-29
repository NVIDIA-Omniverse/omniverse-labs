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
#include "geo_cache.h"
#include <nanousd/nanousdapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>

/* nanousd_isinstance / nanousd_prototype provided by nanousd API */

static unsigned g_scene_load_flags = 0;
static double s_load_time = NAN;
static int s_ref_fallback_depth = 0;

void scene_set_load_flags(unsigned flags)
{
    g_scene_load_flags = flags;
}

void scene_set_load_time(double t)
{
    s_load_time = t;
}

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

static int int64_cmp_for_qsort(const void* a, const void* b)
{
    int64_t ia = *(const int64_t*)a;
    int64_t ib = *(const int64_t*)b;
    return (ia > ib) - (ia < ib);
}

static int int64_array_contains_sorted(const int64_t* values, int count,
                                       int64_t value)
{
    int lo = 0;
    int hi = count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        int64_t mv = values[mid];
        if (mv == value) return 1;
        if (mv < value) lo = mid + 1;
        else hi = mid;
    }
    return 0;
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

static const char* scene_prim_path_copy(NanousdPrim prim, char* buf, size_t buf_size)
{
    if (!buf || buf_size == 0) return NULL;
    buf[0] = '\0';
    const char* path = nanousd_path(prim);
    if (!path) return NULL;
    snprintf(buf, buf_size, "%s", path);
    return buf;
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

static int scene_env_enabled(const char* name)
{
    const char* e = getenv(name);
    return e && e[0] && e[0] != '0';
}

static int scene_parse_asset_arc_item(const char* item,
                                      char* asset,
                                      size_t asset_size,
                                      char* target,
                                      size_t target_size)
{
    if (asset && asset_size > 0) asset[0] = '\0';
    if (target && target_size > 0) target[0] = '\0';
    if (!item || !asset || asset_size == 0) return 0;

    const char* a0 = strchr(item, '@');
    if (!a0) return 0;
    const char* a1 = strchr(a0 + 1, '@');
    if (!a1 || a1 <= a0 + 1) return 0;
    size_t an = (size_t)(a1 - (a0 + 1));
    if (an >= asset_size) an = asset_size - 1;
    memcpy(asset, a0 + 1, an);
    asset[an] = '\0';

    if (target && target_size > 0) {
        const char* t0 = strchr(a1 + 1, '<');
        const char* t1 = t0 ? strchr(t0 + 1, '>') : NULL;
        if (t0 && t1 && t1 > t0 + 1) {
            size_t tn = (size_t)(t1 - (t0 + 1));
            if (tn >= target_size) tn = target_size - 1;
            memcpy(target, t0 + 1, tn);
            target[tn] = '\0';
        }
    }

    return asset[0] != '\0';
}

static int scene_find_listop_asset_arc(NanousdPrim prim,
                                       const char* field,
                                       const char* root_layer,
                                       const char* filepath,
                                       char* out_layer,
                                       size_t out_layer_size,
                                       char* out_source,
                                       size_t out_source_size)
{
    if (!prim || !field || !out_layer || out_layer_size == 0) return 0;

    NanousdListOp op = nanousd_prim_listop(prim, field);
    if (!op) return 0;

    int nitems = nanousd_listop_nitems(op);
    for (int i = 0; i < nitems; i++) {
        const char* item = nanousd_listop_item(op, i);
        char asset[1024];
        char target[1024];
        if (!scene_parse_asset_arc_item(item, asset, sizeof(asset),
                                        target, sizeof(target)))
            continue;

        char resolved[4096];
        resolved[0] = '\0';
        if (!nanousd_resolve_asset_path(root_layer, asset,
                                        resolved, sizeof(resolved))) {
            if (asset[0] != '/') continue;
            snprintf(resolved, sizeof(resolved), "%s", asset);
        }
        if (!resolved[0]) continue;
        if (filepath && strcmp(resolved, filepath) == 0) continue;

        snprintf(out_layer, out_layer_size, "%s", resolved);
        if (out_source && out_source_size > 0 && target[0])
            snprintf(out_source, out_source_size, "%s", target);
        nanousd_listop_free(op);
        return 1;
    }

    nanousd_listop_free(op);
    return 0;
}

static int scene_find_direct_asset_arc(NanousdStage stage,
                                       const char* filepath,
                                       char* out_layer,
                                       size_t out_layer_size,
                                       char* out_source,
                                       size_t out_source_size)
{
    if (!stage || !out_layer || out_layer_size == 0) return 0;
    out_layer[0] = '\0';
    if (out_source && out_source_size > 0) out_source[0] = '\0';

    const char* root_layer = nanousd_stage_get_root_layer_path(stage);
    if (!root_layer || !root_layer[0]) root_layer = filepath;

    int nprims = nanousd_nprims(stage);
    for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;
        int narcs = nanousd_ncomposition_arcs(prim);
        for (int a = 0; a < narcs; a++) {
            NanousdCompositionArc arc;
            memset(&arc, 0, sizeof(arc));
            arc.struct_size = (int)sizeof(arc);
            if (!nanousd_composition_arc(prim, a, &arc)) continue;
            if (arc.arc_type != NANOUSD_ARC_REFERENCE &&
                arc.arc_type != NANOUSD_ARC_PAYLOAD) continue;
            if ((arc.flags & NANOUSD_COMPOSITION_ARC_DIRECT) == 0) continue;
            if (!arc.layer_path || !arc.layer_path[0]) continue;

            char resolved[4096];
            resolved[0] = '\0';
            if (!nanousd_resolve_asset_path(root_layer, arc.layer_path,
                                            resolved, sizeof(resolved))) {
                if (arc.layer_path[0] != '/') continue;
                snprintf(resolved, sizeof(resolved), "%s", arc.layer_path);
            }
            if (!resolved[0]) continue;
            if (filepath && strcmp(resolved, filepath) == 0) continue;

            snprintf(out_layer, out_layer_size, "%s", resolved);
            if (out_source && out_source_size > 0 && arc.source_path)
                snprintf(out_source, out_source_size, "%s", arc.source_path);
            nanousd_freeprim(prim);
            return 1;
        }
        if (scene_find_listop_asset_arc(prim, "references", root_layer, filepath,
                                        out_layer, out_layer_size,
                                        out_source, out_source_size) ||
            scene_find_listop_asset_arc(prim, "payload", root_layer, filepath,
                                        out_layer, out_layer_size,
                                        out_source, out_source_size)) {
            nanousd_freeprim(prim);
            return 1;
        }
        nanousd_freeprim(prim);
    }
    return 0;
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

static int path_is_native_usd_prototype(const char* path)
{
    if (!path) return 0;
    const char* keep = getenv("NUSD_RENDER_NATIVE_PROTOTYPES");
    if (keep && keep[0] && keep[0] != '0')
        return 0;
    return strncmp(path, "/__Prototype_", 13) == 0;
}

static void collect_scene_prefixes(NanousdStage stage, int nprims,
                                   PathPrefixList* invisible_prefixes,
                                   PathPrefixList* point_instancer_prefixes)
{
    for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;
        if (!nanousd_isactive(prim)) {
            nanousd_freeprim(prim);
            continue;
        }

        char path_buf[8192];
        const char* path = scene_prim_path_copy(prim, path_buf, sizeof(path_buf));
        const char* tn = nanousd_typename(prim);
        if (tn && !strcmp(tn, "PointInstancer"))
            path_prefix_list_add(point_instancer_prefixes, path);

        int ok = 0;
        const char* vis = nanousd_attrib_token(prim, "visibility", &ok);
        if (ok && vis && !strcmp(vis, "invisible"))
            path_prefix_list_add(invisible_prefixes, path);

        nanousd_freeprim(prim);
    }
}

static int prim_hidden_from_render_slow(NanousdPrim prim)
{
    if (!prim) return 1;

    const char* ppath = nanousd_path(prim);
    if (ppath && strstr(ppath, "/Navmesh/") == ppath)
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

    return 0;
}

static int prim_hidden_from_render_cached(NanousdPrim prim,
                                          const PathPrefixList* invisible_prefixes)
{
    if (!prim) return 1;
    if (!invisible_prefixes) return prim_hidden_from_render_slow(prim);

    const char* ppath = nanousd_path(prim);
    if (ppath && strstr(ppath, "/Navmesh/") == ppath)
        return 1;
    if (path_prefix_list_contains(invisible_prefixes, ppath))
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

static unsigned char* pi_visible_mask_from_invisible_ids(NanousdPrim pi,
                                                         int n_instances)
{
    if (!pi || n_instances <= 0) return NULL;
    int n_hidden = nanousd_attribarraylen(pi, "invisibleIds");
    if (n_hidden <= 0) return NULL;

    int64_t* hidden = (int64_t*)malloc((size_t)n_hidden * sizeof(int64_t));
    if (!hidden) return NULL;
    int got_hidden = nanousd_attribarrayi64(pi, "invisibleIds",
                                            hidden, n_hidden);
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

/* ----------------------------------------------------------------
 * Identity matrix constant
 * ---------------------------------------------------------------- */
static const double kIdentity[16] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
};

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

static char* xform_strdup(const char* s)
{
    if (!s) return NULL;
    size_t n = strlen(s) + 1u;
    char* out = (char*)malloc(n);
    if (out) memcpy(out, s, n);
    return out;
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
    c->slots[i].path = xform_strdup(path);
    if (!c->slots[i].path) return;
    c->slots[i].hash = h;
    memcpy(c->slots[i].world, world, sizeof(double) * 16);
    c->count++;
}

/* ----------------------------------------------------------------
 * Compute world transform by walking the parent chain.
 * Multiplies local transforms from the prim up to the root.
 * ---------------------------------------------------------------- */
static void compute_world_xform(NanousdPrim prim, double world[16])
{
    if (!prim) {
        memcpy(world, kIdentity, sizeof(double) * 16);
        return;
    }

    char path_buf[8192];
    const char* path = scene_prim_path_copy(prim, path_buf, sizeof(path_buf));
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
        char parent_path_buf[8192];
        const char* parent_path = scene_prim_path_copy(parent, parent_path_buf, sizeof(parent_path_buf));
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
 * Count total triangulated indices for a mesh given faceVertexCounts and
 * the actual faceVertexIndices length.
 * Each face with N vertices produces (N-2) triangles = (N-2)*3 indices.
 * Extra point vertices are valid USD data and do not matter here. If
 * faceVertexCounts over-promises versus faceVertexIndices, count only the
 * complete prefix of faces so later triangulation can recover cleanly.
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
 *
 * Returns the number of indices actually written. If a USD asset
 * declares more faces in faceVertexCounts than there is data in
 * faceVertexIndices (some hand-authored test files do exactly this),
 * we stop at the first face that would read past nfvi rather than
 * fault into adjacent memory and emit garbage indices that later trip
 * the renderer / pickers.
 * ---------------------------------------------------------------- */
static int triangulate_faces(const int* fvi, int nfvi,
                             const int* fvc, int nfaces,
                             uint32_t* out_indices)
{
    int fvi_offset = 0;
    int out_offset = 0;

    for (int f = 0; f < nfaces; f++) {
        int n = fvc[f];
        if (n <= 0)
            continue;
        if (fvi_offset + n > nfvi)
            break;
        if (n < 3) {
            fvi_offset += n;
            continue;
        }

        int v0 = fvi[fvi_offset];
        for (int t = 0; t < n - 2; t++) {
            out_indices[out_offset++] = (uint32_t)v0;
            out_indices[out_offset++] = (uint32_t)fvi[fvi_offset + t + 1];
            out_indices[out_offset++] = (uint32_t)fvi[fvi_offset + t + 2];
        }

        fvi_offset += n;
    }
    return out_offset;
}

typedef struct {
    NanousdPrim prim;
    int*        face_indices;
    int         nfaces;
    int         nindices;
    char        path[512];
} GeomSubsetInfo;

static int build_mesh_face_index_ranges(Arena* arena,
                                        const SceneMesh* mesh,
                                        const int* fvc_data,
                                        int fvc_count,
                                        int fvi_count,
                                        int** out_starts,
                                        int** out_counts,
                                        int* out_nfaces)
{
    if (!arena || !mesh || !mesh->indices || mesh->nindices <= 0 ||
        !out_starts || !out_counts || !out_nfaces)
        return 0;

    int nfaces = (fvc_data && fvc_count > 0) ? fvc_count : mesh->nindices / 3;
    if (nfaces <= 0)
        return 0;

    int* starts = (int*)arena_alloc(arena, (size_t)nfaces * sizeof(int), 16);
    int* counts = (int*)arena_calloc(arena, (size_t)nfaces, sizeof(int));
    if (!starts || !counts)
        return 0;
    for (int i = 0; i < nfaces; i++)
        starts[i] = -1;

    if (fvc_data && fvc_count > 0) {
        int fvi_offset = 0;
        int tri_offset = 0;
        for (int f = 0; f < fvc_count; f++) {
            int n = fvc_data[f];
            if (n <= 0)
                continue;
            if (fvi_offset + n > fvi_count)
                break;
            if (n >= 3) {
                int ni = (n - 2) * 3;
                if (tri_offset + ni > mesh->nindices)
                    break;
                starts[f] = tri_offset;
                counts[f] = ni;
                tri_offset += ni;
            }
            fvi_offset += n;
        }
    } else {
        for (int f = 0; f < nfaces; f++) {
            starts[f] = f * 3;
            counts[f] = 3;
        }
    }

    *out_starts = starts;
    *out_counts = counts;
    *out_nfaces = nfaces;
    return 1;
}

static int count_subset_indices(const int* subset_faces,
                                int subset_face_count,
                                const int* face_counts,
                                int nfaces)
{
    int total = 0;
    for (int i = 0; i < subset_face_count; i++) {
        int f = subset_faces[i];
        if (f >= 0 && f < nfaces)
            total += face_counts[f];
    }
    return total;
}

static int emit_subset_indices(Arena* arena,
                               const SceneMesh* base,
                               const int* face_starts,
                               const int* face_counts,
                               int nfaces,
                               const int* subset_faces,
                               int subset_face_count,
                               uint32_t** out_indices)
{
    int nindices = count_subset_indices(
        subset_faces, subset_face_count, face_counts, nfaces);
    if (nindices <= 0)
        return 0;

    uint32_t* indices =
        (uint32_t*)arena_alloc(arena, (size_t)nindices * sizeof(uint32_t), 16);
    if (!indices)
        return 0;

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

static void mesh_set_heap_path(SceneMesh* mesh, const char* path)
{
    if (!mesh) return;
    mesh->path = scene_strdup(path ? path : "");
}

static int scene_ensure_mesh_capacity(Scene* scene, int* max_meshes, int needed)
{
    if (!scene || !max_meshes || needed <= 0) return 0;
    if (needed <= *max_meshes) return 1;

    int old_cap = *max_meshes;
    int new_cap = old_cap > 0 ? old_cap : 64;
    while (new_cap < needed) {
        if (new_cap > INT_MAX / 2) return 0;
        new_cap *= 2;
    }

    SceneMesh* grown =
        (SceneMesh*)realloc(scene->meshes, (size_t)new_cap * sizeof(SceneMesh));
    if (!grown) return 0;
    memset(grown + old_cap, 0, (size_t)(new_cap - old_cap) * sizeof(SceneMesh));
    scene->meshes = grown;
    *max_meshes = new_cap;
    return 1;
}

static int scene_ensure_curve_capacity(Scene* scene, Arena* arena,
                                       int* max_curves, int needed)
{
    if (!scene || !arena || !max_curves || needed <= 0) return 0;
    if (needed <= *max_curves) return 1;

    int old_cap = *max_curves;
    int new_cap = old_cap > 0 ? old_cap : 64;
    while (new_cap < needed) {
        if (new_cap > INT_MAX / 2) return 0;
        new_cap *= 2;
    }

    SceneCurve* grown =
        (SceneCurve*)arena_calloc(arena, (size_t)new_cap, sizeof(SceneCurve));
    if (!grown) return 0;
    if (scene->curves && old_cap > 0)
        memcpy(grown, scene->curves, (size_t)old_cap * sizeof(SceneCurve));
    scene->curves = grown;
    *max_curves = new_cap;
    return 1;
}

static int split_mesh_by_geom_subsets(Arena* arena,
                                      Scene* scene,
                                      NanousdPrim mesh_prim,
                                      int base_mesh_idx,
                                      int* max_meshes,
                                      const int* fvc_data,
                                      int fvc_count,
                                      int fvi_count)
{
    if (!arena || !scene || !mesh_prim || base_mesh_idx < 0 ||
        !max_meshes || base_mesh_idx >= *max_meshes)
        return 1;

    SceneMesh base = scene->meshes[base_mesh_idx];
    int* face_starts = NULL;
    int* face_counts = NULL;
    int nfaces = 0;
    if (!build_mesh_face_index_ranges(arena, &base, fvc_data, fvc_count,
                                      fvi_count, &face_starts, &face_counts,
                                      &nfaces))
        return 1;

    int nch = nanousd_nchildren(mesh_prim);
    if (nch <= 0)
        return 1;

    GeomSubsetInfo* subsets =
        (GeomSubsetInfo*)arena_calloc(arena, (size_t)nch, sizeof(GeomSubsetInfo));
    unsigned char* covered =
        (unsigned char*)arena_calloc(arena, (size_t)nfaces, sizeof(unsigned char));
    if (!subsets || !covered)
        return 1;

    int nsubsets = 0;
    for (int c = 0; c < nch; c++) {
        NanousdPrim child = nanousd_child(mesh_prim, c);
        if (!child) continue;
        const char* tn = nanousd_typename(child);
        if (!tn || strcmp(tn, "GeomSubset") != 0) {
            nanousd_freeprim(child);
            continue;
        }

        int ok = 0;
        const char* element_type = nanousd_attrib_token(child, "elementType", &ok);
        if (ok && element_type && element_type[0] &&
            strcmp(element_type, "face") != 0) {
            nanousd_freeprim(child);
            continue;
        }

        int subset_face_count = 0;
        const int* authored_faces =
            nanousd_arraydatai(child, "indices", &subset_face_count);
        int* copied_faces = NULL;
        if (!authored_faces || subset_face_count <= 0) {
            int len = nanousd_attribarraylen(child, "indices");
            if (len > 0) {
                copied_faces =
                    (int*)arena_alloc(arena, (size_t)len * sizeof(int), 16);
                if (copied_faces) {
                    nanousd_attribarrayi(child, "indices", copied_faces, len);
                    authored_faces = copied_faces;
                    subset_face_count = len;
                }
            }
        }
        if (!authored_faces || subset_face_count <= 0) {
            nanousd_freeprim(child);
            continue;
        }

        if (!copied_faces) {
            copied_faces =
                (int*)arena_alloc(arena,
                                  (size_t)subset_face_count * sizeof(int), 16);
            if (!copied_faces) {
                nanousd_freeprim(child);
                continue;
            }
            memcpy(copied_faces, authored_faces,
                   (size_t)subset_face_count * sizeof(int));
        }

        int nindices = count_subset_indices(copied_faces, subset_face_count,
                                            face_counts, nfaces);
        if (nindices <= 0) {
            nanousd_freeprim(child);
            continue;
        }

        GeomSubsetInfo* info = &subsets[nsubsets++];
        info->prim = child;
        info->face_indices = copied_faces;
        info->nfaces = subset_face_count;
        info->nindices = nindices;
        const char* cp = nanousd_path(child);
        if (cp && cp[0]) {
            size_t n = strnlen(cp, sizeof(info->path) - 1);
            memcpy(info->path, cp, n);
            info->path[n] = '\0';
        }

        for (int i = 0; i < subset_face_count; i++) {
            int f = copied_faces[i];
            if (f >= 0 && f < nfaces && face_counts[f] > 0)
                covered[f] = 1;
        }
    }

    if (nsubsets <= 0)
        return 1;

    int remainder_nindices = 0;
    for (int f = 0; f < nfaces; f++) {
        if (!covered[f])
            remainder_nindices += face_counts[f];
    }

    int out_count = nsubsets + (remainder_nindices > 0 ? 1 : 0);
    if (!scene_ensure_mesh_capacity(scene, max_meshes,
                                    base_mesh_idx + out_count)) {
        for (int s = 0; s < nsubsets; s++)
            nanousd_freeprim(subsets[s].prim);
        return 1;
    }

    /* USD GeomSubset is a material partition over Mesh faces. OpenGL, like
     * Metal and Vulkan, binds one material per draw record, so we split the
     * mesh into subset SceneMesh records here and let material assignment
     * resolve each subset path later. */
    int emitted = 0;
    for (int s = 0; s < nsubsets; s++) {
        uint32_t* subset_indices = NULL;
        int written = emit_subset_indices(arena, &base, face_starts, face_counts,
                                          nfaces, subsets[s].face_indices,
                                          subsets[s].nfaces, &subset_indices);
        if (written <= 0 || !subset_indices)
            continue;

        SceneMesh* out = &scene->meshes[base_mesh_idx + emitted];
        *out = base;
        out->indices = subset_indices;
        out->nindices = written;
        out->material_index = -1;
        mesh_set_heap_path(out, subsets[s].path);
        out->prototype_idx = base_mesh_idx + emitted;
        out->vertex_offset = 0;
        out->index_offset = 0;
        emitted++;
    }

    if (remainder_nindices > 0) {
        uint32_t* rem_indices =
            (uint32_t*)arena_alloc(arena,
                                   (size_t)remainder_nindices * sizeof(uint32_t),
                                   16);
        int out_i = 0;
        if (rem_indices) {
            for (int f = 0; f < nfaces; f++) {
                if (covered[f] || face_counts[f] <= 0 || face_starts[f] < 0)
                    continue;
                memcpy(&rem_indices[out_i], &base.indices[face_starts[f]],
                       (size_t)face_counts[f] * sizeof(uint32_t));
                out_i += face_counts[f];
            }
            SceneMesh* out = &scene->meshes[base_mesh_idx + emitted];
            *out = base;
            out->indices = rem_indices;
            out->nindices = out_i;
            out->material_index = -1;
            mesh_set_heap_path(out, base.path);
            out->prototype_idx = base_mesh_idx + emitted;
            out->vertex_offset = 0;
            out->index_offset = 0;
            emitted++;
        }
    }

    if (emitted > 0)
        free(base.path);
    for (int s = 0; s < nsubsets; s++)
        nanousd_freeprim(subsets[s].prim);
    return emitted > 0 ? emitted : 1;
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
 * Hash-dedup expansion for meshes with faceVarying primvars (UV seams,
 * hard-edge faceVarying normals). Each face-vertex j is keyed on
 * (orig_vertex = fvi[j], uv at j, normal at j). Smooth verts share a
 * single new slot across all faces; verts on a UV seam or hard edge
 * get a separate slot per distinct primvar value. Mostly-smooth meshes
 * pay ~1.0× growth; only seam vertices duplicate.
 *
 * Storm's pxr/imaging/hd/meshUtil.cpp does the same shape (flatten
 * faceVarying to per-face-vertex) but without dedup; we dedupe so a
 * cube with mostly-smooth shading doesn't pay a 4× vertex count.
 *
 * On entry mesh->positions / nvertices are vertex-rate; mesh->normals
 * and mesh->texcoords are vertex-rate fallbacks (either may be NULL).
 * uvs_fv (when non-NULL, length = fvi_count*2) and normals_fv
 * (length = fvi_count*3) override the vertex-rate fallbacks per
 * face-vertex. On success the function overwrites mesh->positions /
 * normals / texcoords / indices / nvertices / nindices with the
 * expanded set and re-triangulates via fan against fvc_data.
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

    const int          norig          = mesh->nvertices;
    const float* const orig_positions = mesh->positions;
    const float* const orig_normals   = mesh->normals;
    const float* const orig_uvs       = mesh->texcoords;
    const int has_normals = (orig_normals != NULL) || (normals_fv != NULL);
    const int has_uvs     = (orig_uvs     != NULL) || (uvs_fv     != NULL);

    typedef struct {
        float    uv[2];
        float    n[3];
        uint32_t new_idx;
        int      next;
    } Split;

    Split* splits     = (Split*)arena_alloc(arena, (size_t)fvi_count * sizeof(Split), 8);
    int*   split_head = (int*)  arena_alloc(arena, (size_t)norig     * sizeof(int),   4);
    if (!splits || !split_head) return 0;
    for (int v = 0; v < norig; v++) split_head[v] = -1;
    int split_count = 0;

    uint32_t* fv_to_new = (uint32_t*)arena_alloc(arena, (size_t)fvi_count * sizeof(uint32_t), 4);
    if (!fv_to_new) return 0;

    /* Worst case: every face-vertex unique. Real meshes use far fewer;
     * the unused tail is wasted arena bytes, freed wholesale at scene_free. */
    float* new_positions = (float*)arena_alloc(arena, (size_t)fvi_count * 3 * sizeof(float), 16);
    float* new_normals   = has_normals ? (float*)arena_alloc(arena, (size_t)fvi_count * 3 * sizeof(float), 16) : NULL;
    float* new_uvs       = has_uvs     ? (float*)arena_alloc(arena, (size_t)fvi_count * 2 * sizeof(float), 16) : NULL;
    if (!new_positions || (has_normals && !new_normals) || (has_uvs && !new_uvs))
        return 0;

    int new_nv = 0;
    for (int j = 0; j < fvi_count; j++) {
        const int v = fvi_data[j];
        if (v < 0 || v >= norig) {
            fv_to_new[j] = 0;
            continue;
        }

        float u0 = 0.0f, u1 = 0.0f;
        if (uvs_fv)        { u0 = uvs_fv[j*2]; u1 = uvs_fv[j*2+1]; }
        else if (orig_uvs) { u0 = orig_uvs[v*2]; u1 = orig_uvs[v*2+1]; }

        float n0 = 0.0f, n1 = 0.0f, n2 = 0.0f;
        if (normals_fv)        { n0 = normals_fv[j*3]; n1 = normals_fv[j*3+1]; n2 = normals_fv[j*3+2]; }
        else if (orig_normals) { n0 = orig_normals[v*3]; n1 = orig_normals[v*3+1]; n2 = orig_normals[v*3+2]; }

        /* Linear scan through the splits hanging off this orig vertex.
         * S (splits per vertex) is 1 for smooth verts, ≤ face-degree for
         * seam verts; linear search beats hashing for these sizes. */
        int found = -1;
        const float eps = 1e-6f;
        for (int s = split_head[v]; s != -1; s = splits[s].next) {
            const Split* sp = &splits[s];
            if (fabsf(sp->uv[0] - u0) < eps && fabsf(sp->uv[1] - u1) < eps &&
                fabsf(sp->n[0]  - n0) < eps && fabsf(sp->n[1]  - n1) < eps &&
                fabsf(sp->n[2]  - n2) < eps) {
                found = (int)sp->new_idx;
                break;
            }
        }
        if (found >= 0) {
            fv_to_new[j] = (uint32_t)found;
        } else {
            const uint32_t idx = (uint32_t)new_nv++;
            const int s = split_count++;
            splits[s].uv[0]   = u0; splits[s].uv[1]   = u1;
            splits[s].n[0]    = n0; splits[s].n[1]    = n1; splits[s].n[2] = n2;
            splits[s].new_idx = idx;
            splits[s].next    = split_head[v];
            split_head[v]     = s;
            fv_to_new[j]      = idx;

            new_positions[idx*3 + 0] = orig_positions[v*3 + 0];
            new_positions[idx*3 + 1] = orig_positions[v*3 + 1];
            new_positions[idx*3 + 2] = orig_positions[v*3 + 2];
            if (new_normals) {
                new_normals[idx*3 + 0] = n0;
                new_normals[idx*3 + 1] = n1;
                new_normals[idx*3 + 2] = n2;
            }
            if (new_uvs) {
                new_uvs[idx*2 + 0] = u0;
                new_uvs[idx*2 + 1] = u1;
            }
        }
    }

    const int tri_count = (fvc_data && fvc_count > 0)
        ? count_triangulated_indices(fvc_data, fvc_count, fvi_count) : fvi_count;
    uint32_t* new_indices = (uint32_t*)arena_alloc(arena, (size_t)tri_count * sizeof(uint32_t), 16);
    if (!new_indices) return 0;

    int n_idx = 0;
    if (fvc_data && fvc_count > 0) {
        int fvi_off = 0;
        for (int i = 0; i < fvc_count; i++) {
            const int n = fvc_data[i];
            if (n < 3 || fvi_off + n > fvi_count) {
                if (n > 0) fvi_off += n;
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
        for (int j = 0; j < n; j++) new_indices[n_idx++] = fv_to_new[j];
    }

    mesh->positions = new_positions;
    mesh->normals   = new_normals;
    mesh->texcoords = new_uvs;
    mesh->indices   = new_indices;
    mesh->nvertices = new_nv;
    mesh->nindices  = n_idx;
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
 * Load primvars:st (UV) and inspect normals for faceVarying authoring,
 * then either dedup-expand the mesh (when any primvar is faceVarying)
 * or copy vertex-rate UVs straight in. Used by both the prototype-mesh
 * load and the instance-prim child-walker — keeping a single helper
 * means the dedup expander reaches USD-instanced geometry (the bulk of
 * a SimReady warehouse), and there's only one fan-triangulation copy
 * to keep correct.
 * ---------------------------------------------------------------- */
static void load_mesh_facevarying_primvars(Arena* arena,
                                           SceneMesh* mesh,
                                           NanousdStage stage,
                                           NanousdPrim prim,
                                           const int* fvi_data, int fvi_count,
                                           const int* fvc_data, int fvc_count)
{
    mesh->texcoords = NULL;
    if (!prim || !fvi_data || fvi_count <= 0) return;
    const int skip_texcoords =
        (g_scene_load_flags & SCENE_LOAD_SKIP_TEXCOORDS) != 0;
    const int skip_facevarying_normals = skip_texcoords;

    const char* uv_names[]     = { "primvars:st", "primvars:st0",
                                   "primvars:UVMap", "primvars:uv" };
    const char* uv_idx_names[] = { "primvars:st:indices", "primvars:st0:indices",
                                   "primvars:UVMap:indices", "primvars:uv:indices" };
    const char* uv_attr        = NULL;
    const float* uv_pool       = NULL;
    int          uv_pool_count = 0;
    const int*   uv_indices    = NULL;
    int          uv_idx_count  = 0;
    const char*  uv_interp     = NULL;

    if (!skip_texcoords) {
        for (int i = 0; i < (int)(sizeof(uv_names)/sizeof(uv_names[0])); i++) {
            int n = 0;
            const float* d = nanousd_arraydataf(prim, uv_names[i], &n);
            if (d && n > 0) {
                uv_pool       = d;
                uv_pool_count = n;
                uv_attr       = uv_names[i];
                int idx_n = 0;
                uv_indices    = nanousd_arraydatai(prim, uv_idx_names[i], &idx_n);
                uv_idx_count  = idx_n;
                uv_interp     = nanousd_attrib_interpolation(prim, uv_attr);
                break;
            }
        }
    }

    int uv_is_face_varying = 0;
    int uv_is_vertex_rate  = 0;
    if (uv_pool && uv_pool_count > 0) {
        if (uv_interp && strcmp(uv_interp, "faceVarying") == 0) {
            uv_is_face_varying = 1;
        } else if (uv_interp &&
                   (strcmp(uv_interp, "vertex")  == 0 ||
                    strcmp(uv_interp, "varying") == 0)) {
            uv_is_vertex_rate = (uv_pool_count / 2 == mesh->nvertices);
        } else {
            /* Interpolation hint missing — fall back to length match. */
            if (uv_pool_count / 2 == mesh->nvertices) uv_is_vertex_rate = 1;
            else if (uv_indices ? uv_idx_count == fvi_count
                                : uv_pool_count / 2 == fvi_count) uv_is_face_varying = 1;
        }
    }

    const char*  nrm_attr      = NULL;
    const float* nrm_data_raw  = NULL;
    int          nrm_count_raw = 0;
    if (!skip_facevarying_normals) {
        int nrm_n;
        const float* nrm_try = nanousd_arraydataf(prim, "normals", &nrm_n);
        if (nrm_try) { nrm_attr = "normals"; nrm_data_raw = nrm_try; nrm_count_raw = nrm_n; }
        else {
            nrm_try = nanousd_arraydataf(prim, "primvars:normals", &nrm_n);
            if (nrm_try) { nrm_attr = "primvars:normals"; nrm_data_raw = nrm_try; nrm_count_raw = nrm_n; }
        }
    }
    const char* nrm_interp = nrm_attr ? nanousd_attrib_interpolation(prim, nrm_attr) : NULL;
    int normals_are_face_varying = 0;
    if (nrm_data_raw && nrm_count_raw / 3 == fvi_count) {
        if (nrm_interp && strcmp(nrm_interp, "faceVarying") == 0) {
            normals_are_face_varying = 1;
        } else if (!nrm_interp && fvi_count != mesh->nvertices) {
            /* No hint and length distinguishes faceVarying from vertex-rate. */
            normals_are_face_varying = 1;
        }
    }

    /* Build per-face-vertex arrays for whichever primvars are faceVarying. */
    float* uvs_fv     = NULL;
    float* normals_fv = NULL;
    if (uv_is_face_varying) {
        uvs_fv = (float*)arena_alloc(arena, (size_t)fvi_count * 2 * sizeof(float), 16);
        if (uvs_fv) {
            const int n_uvs = uv_pool_count / 2;
            if (uv_indices && uv_idx_count > 0) {
                const int n_walk = fvi_count < uv_idx_count ? fvi_count : uv_idx_count;
                for (int j = 0; j < n_walk; j++) {
                    int u = uv_indices[j];
                    if (u < 0 || u >= n_uvs) {
                        uvs_fv[j*2 + 0] = 0.0f;
                        uvs_fv[j*2 + 1] = 0.0f;
                    } else {
                        uvs_fv[j*2 + 0] = uv_pool[u*2 + 0];
                        uvs_fv[j*2 + 1] = uv_pool[u*2 + 1];
                    }
                }
                for (int j = n_walk; j < fvi_count; j++) {
                    uvs_fv[j*2 + 0] = 0.0f;
                    uvs_fv[j*2 + 1] = 0.0f;
                }
            } else if (uv_pool_count / 2 == fvi_count) {
                memcpy(uvs_fv, uv_pool, (size_t)fvi_count * 2 * sizeof(float));
            } else {
                uvs_fv = NULL;
            }
        }
    }
    if (normals_are_face_varying) {
        normals_fv = (float*)arena_alloc(arena, (size_t)fvi_count * 3 * sizeof(float), 16);
        if (normals_fv) {
            memcpy(normals_fv, nrm_data_raw, (size_t)fvi_count * 3 * sizeof(float));
        }
    }

    apply_usdskel_skinning(arena, stage, prim, mesh,
                           fvi_data, fvi_count, normals_fv);

    if (uv_is_face_varying || normals_are_face_varying) {
        /* Stash vertex-rate UVs (if also present) on mesh->texcoords so
         * the expander can use them as fallback for slots whose
         * faceVarying override is missing. */
        if (!skip_texcoords && uv_is_vertex_rate && uv_pool) {
            mesh->texcoords = (float*)arena_alloc(arena,
                (size_t)uv_pool_count * sizeof(float), 16);
            if (mesh->texcoords)
                memcpy(mesh->texcoords, uv_pool,
                       (size_t)uv_pool_count * sizeof(float));
        }
        (void)expand_facevarying_mesh(arena, mesh,
                                      fvi_data, fvi_count,
                                      fvc_data, fvc_count,
                                      uvs_fv, normals_fv);
    } else if (!skip_texcoords && uv_is_vertex_rate && uv_pool) {
        mesh->texcoords = (float*)arena_alloc(arena,
            (size_t)uv_pool_count * sizeof(float), 16);
        if (mesh->texcoords)
            memcpy(mesh->texcoords, uv_pool,
                   (size_t)uv_pool_count * sizeof(float));
    }
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

    float mn[3] = {
        mesh->local_bounds_min[0],
        mesh->local_bounds_min[1],
        mesh->local_bounds_min[2],
    };
    float mx[3] = {
        mesh->local_bounds_max[0],
        mesh->local_bounds_max[1],
        mesh->local_bounds_max[2],
    };

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

/* ===================================================================
 * Path B support helpers (ported from nanousd-vulkan-renderer/src/scene.c).
 * Bounds + compact-instance reserve/pack used by the asset-arc flat
 * seed/replay + compact PointInstancer loader below.
 * =================================================================== */

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

/* Pack a row-vector row-major 4x4 (translation in m[12..14]) into the 12-float
 * column-major affine SceneInstanceTransform (matches the Vulkan layout). */
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

static int prim_is_point_instancer(NanousdPrim prim)
{
    if (!prim) return 0;
    const char* tn = nanousd_typename(prim);
    return tn && !strcmp(tn, "PointInstancer");
}

/* Prototype mesh lookup: maps a prim path string to its SceneMesh index.
 *
 * Open-addressing hash table with linear probing. The previous
 * implementation was a flat array with strcmp() linear scan, which
 * became O(n_instances * proto_count) on large scenes. Same design
 * as nanousd-vulkan-renderer / nanousd-opengl-renderer (kept in sync —
 * three sister repos shared this anti-pattern, fixed identically).
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

/* ----------------------------------------------------------------
 * BasisCurves: parse a token attrib into our enum. Falls back to the
 * USD-spec defaults when the attribute is unauthored.
 * ---------------------------------------------------------------- */
static SceneCurveBasis parse_curve_basis(const char* s) {
    if (!s) return CURVE_BASIS_BEZIER;
    if (strcmp(s, "bezier") == 0)     return CURVE_BASIS_BEZIER;
    if (strcmp(s, "bspline") == 0)    return CURVE_BASIS_BSPLINE;
    if (strcmp(s, "catmullRom") == 0) return CURVE_BASIS_CATMULLROM;
    return CURVE_BASIS_BEZIER;
}
static SceneCurveWrap parse_curve_wrap(const char* s) {
    if (!s) return CURVE_WRAP_NONPERIODIC;
    if (strcmp(s, "periodic") == 0)    return CURVE_WRAP_PERIODIC;
    if (strcmp(s, "pinned") == 0)      return CURVE_WRAP_PINNED;
    return CURVE_WRAP_NONPERIODIC;
}

/* ----------------------------------------------------------------
 * BasisCurves: build the 4-CV-per-patch index buffer.
 *
 * Cubic path mirrors Storm's _BuildCubicIndexArray
 * (basisCurvesComputations.cpp:187-393): bezier → vStep=3,
 * bspline/catmullRom → vStep=1; periodic wraps with mod; pinned adds
 * ghost segments at start + end (triplicate first/last for bspline,
 * duplicate for catmullRom).
 *
 * Linear path emits adjacent-pair line segments encoded as 4-CV
 * patches with the trailing two CVs duplicated (cv[2]=cv[3]=v1).
 * Storm uses patch_vertices=2 for these, but the linear-basis branch
 * in our TES sets cv[2]/cv[3] weights to zero so the duplication is
 * just a packing convention that lets a single 4-CV pipeline handle
 * everything. Mirrors _BuildLineSegmentIndexArray (cpp:106-184).
 * ---------------------------------------------------------------- */

static int count_cubic_patches(int count, SceneCurveBasis basis,
                               SceneCurveWrap wrap)
{
    if (count < 2) return 0;
    int vStep = (basis == CURVE_BASIS_BEZIER) ? 3 : 1;
    int periodic = (wrap == CURVE_WRAP_PERIODIC) ? 1 : 0;
    int pinned   = (wrap == CURVE_WRAP_PINNED) &&
                   (basis == CURVE_BASIS_BSPLINE ||
                    basis == CURVE_BASIS_CATMULLROM);
    if (count < 4 && !pinned) return 0;
    int numSegs = periodic ? (count / vStep > 1 ? count / vStep : 1)
                            : ((count - 4 > 0 ? (count - 4) / vStep : 0) + 1);
    if (numSegs < 0) numSegs = 0;
    int extra = pinned ? (basis == CURVE_BASIS_BSPLINE ? 4 : 2) : 0;
    return numSegs + extra;
}

static int count_linear_patches(int count, SceneCurveWrap wrap,
                                SceneCurveBasis basis)
{
    if (count < 2) return 0;
    int periodic = (wrap == CURVE_WRAP_PERIODIC) ? 1 : 0;
    int pinned   = (wrap == CURVE_WRAP_PINNED) ? 1 : 0;
    int skip_endpoints = (basis == CURVE_BASIS_CATMULLROM) && !pinned;
    int seg = count - 1;
    if (skip_endpoints) seg -= 2;
    if (seg < 0) seg = 0;
    if (periodic) seg += 1;
    return seg;
}

static void emit_cubic_patches(
    int vertexBase, int count,
    SceneCurveBasis basis, SceneCurveWrap wrap,
    uint32_t* idx, int* patch_cursor)
{
    int vStep = (basis == CURVE_BASIS_BEZIER) ? 3 : 1;
    int periodic = (wrap == CURVE_WRAP_PERIODIC) ? 1 : 0;
    int pinned   = (wrap == CURVE_WRAP_PINNED) &&
                   (basis == CURVE_BASIS_BSPLINE ||
                    basis == CURVE_BASIS_CATMULLROM);

    int p = *patch_cursor;
    if (pinned) {
        int v0 = vertexBase;
        int v1 = (count > 1) ? vertexBase + 1 : v0;
        int v2 = (count > 2) ? vertexBase + 2 : v1;
        if (basis == CURVE_BASIS_BSPLINE) {
            idx[p*4+0]=v0; idx[p*4+1]=v0; idx[p*4+2]=v0; idx[p*4+3]=v1; p++;
        }
        idx[p*4+0]=v0; idx[p*4+1]=v0; idx[p*4+2]=v1; idx[p*4+3]=v2; p++;
    }

    int numSegs = periodic ? (count / vStep > 1 ? count / vStep : 1)
                            : ((count - 4 > 0 ? (count - 4) / vStep : 0) + 1);
    if (count < 4 && !pinned) numSegs = 0;
    for (int i = 0; i < numSegs; i++) {
        int offset = i * vStep;
        for (int vv = 0; vv < 4; vv++) {
            int li = periodic
                       ? ((offset + vv) % count)
                       : ((offset + vv >= count) ? count - 1 : offset + vv);
            idx[p*4+vv] = (uint32_t)(vertexBase + li);
        }
        p++;
    }

    if (pinned) {
        int n = count;
        int vN_3 = vertexBase + (n >= 3 ? n - 3 : 0);
        int vN_2 = vertexBase + (n >= 2 ? n - 2 : 0);
        int vN_1 = vertexBase + (n >= 1 ? n - 1 : 0);
        idx[p*4+0]=vN_3; idx[p*4+1]=vN_2; idx[p*4+2]=vN_1; idx[p*4+3]=vN_1; p++;
        if (basis == CURVE_BASIS_BSPLINE) {
            idx[p*4+0]=vN_2; idx[p*4+1]=vN_1; idx[p*4+2]=vN_1; idx[p*4+3]=vN_1; p++;
        }
    }

    *patch_cursor = p;
}

static void emit_linear_patches(
    int vertexBase, int count,
    SceneCurveBasis basis, SceneCurveWrap wrap,
    uint32_t* idx, int* patch_cursor)
{
    int periodic = (wrap == CURVE_WRAP_PERIODIC) ? 1 : 0;
    int pinned   = (wrap == CURVE_WRAP_PINNED) ? 1 : 0;
    int skip_endpoints = (basis == CURVE_BASIS_CATMULLROM) && !pinned;
    int p = *patch_cursor;
    int start = skip_endpoints ? 1 : 0;
    int end   = skip_endpoints ? count - 2 : count - 1;
    for (int i = start; i < end; i++) {
        int v0 = vertexBase + i;
        int v1 = vertexBase + i + 1;
        idx[p*4+0]=v0; idx[p*4+1]=v1; idx[p*4+2]=v1; idx[p*4+3]=v1; p++;
    }
    if (periodic) {
        int v0 = vertexBase + count - 1;
        int v1 = vertexBase + 0;
        idx[p*4+0]=v0; idx[p*4+1]=v1; idx[p*4+2]=v1; idx[p*4+3]=v1; p++;
    }
    *patch_cursor = p;
}

static void build_curve_patches(
    Arena* arena,
    const int* counts, int ncounts,
    int type_is_cubic,
    SceneCurveBasis basis, SceneCurveWrap wrap,
    uint32_t** out_indices, int* out_npatches)
{
    *out_indices = NULL;
    *out_npatches = 0;

    int total = 0;
    for (int c = 0; c < ncounts; c++) {
        if (type_is_cubic) total += count_cubic_patches(counts[c], basis, wrap);
        else                total += count_linear_patches(counts[c], wrap, basis);
    }
    if (total == 0) return;

    uint32_t* idx = (uint32_t*)arena_alloc(arena,
        (size_t)total * 4 * sizeof(uint32_t), 4);
    if (!idx) return;

    int patch_i = 0;
    int vertexIndex = 0;
    for (int c = 0; c < ncounts; c++) {
        int count = counts[c];
        if (type_is_cubic)
            emit_cubic_patches(vertexIndex, count, basis, wrap, idx, &patch_i);
        else
            emit_linear_patches(vertexIndex, count, basis, wrap, idx, &patch_i);
        vertexIndex += count;
    }
    *out_indices  = idx;
    *out_npatches = total;
}

/* ----------------------------------------------------------------
 * BasisCurves: count the needed "varying" primvars per Storm's
 * HdBasisCurvesTopology::CalculateNeededNumberOfVaryingControlPoints.
 * Storm authors one varying value per curve segment + 1 (or per
 * segment for periodic). We use this to detect "varying"
 * interpolation by source length when the authored interpolation
 * token isn't exposed by nanousd.
 *
 * For non-periodic cubic with vStep:
 *   varying_count = numSegs + 1, where numSegs = (count-4)/vStep + 1
 * For periodic cubic:
 *   varying_count = numSegs = max(count/vStep, 1)
 * For linear (any wrap):
 *   varying_count = count (same as vertex), so the heuristic catches it
 *   via the nv-match branch.
 * ---------------------------------------------------------------- */
static int varying_count_for_curves(
    const int* counts, int ncounts,
    int type_is_cubic, SceneCurveBasis basis, SceneCurveWrap wrap)
{
    if (!type_is_cubic) {
        /* linear — same as vertex; nv-match branch handles it */
        int total = 0;
        for (int c = 0; c < ncounts; c++) total += counts[c];
        return total;
    }
    int vStep = (basis == CURVE_BASIS_BEZIER) ? 3 : 1;
    int periodic = (wrap == CURVE_WRAP_PERIODIC) ? 1 : 0;
    int total = 0;
    for (int c = 0; c < ncounts; c++) {
        int count = counts[c];
        if (count < 4) continue;
        int numSegs = periodic ? (count / vStep > 1 ? count / vStep : 1)
                                : ((count - 4) / vStep + 1);
        total += periodic ? numSegs : (numSegs + 1);
    }
    return total;
}

/* Linearly fan a per-segment-boundary float into per-vertex floats,
 * for varying-interpolation widths. The shader then applies the basis
 * weights to interpolate per-CV; for cubic we want the *intermediate*
 * CVs (between segment boundaries) to evaluate to a linear blend of
 * the boundary widths regardless of the basis basis weights. The
 * cleanest approach for an arbitrary basis: just interpolate on a
 * per-CV linear schedule, which approximates Storm's behavior closely
 * for the common case of constant-or-linear width changes. */
static void fan_varying_to_vertex(
    const float* src, int src_count,
    int* counts, int ncounts,
    int type_is_cubic, SceneCurveBasis basis, SceneCurveWrap wrap,
    float* out, int nv)
{
    int vStep = (type_is_cubic && basis == CURVE_BASIS_BEZIER) ? 3 : 1;
    int periodic = (wrap == CURVE_WRAP_PERIODIC) ? 1 : 0;
    int v = 0;       /* index into per-CV out */
    int s = 0;       /* index into varying source */
    for (int c = 0; c < ncounts && s < src_count; c++) {
        int count = counts[c];
        if (count < 4 && type_is_cubic) {
            for (int i = 0; i < count && v < nv; i++) out[v++] = src[s];
            s += 1;
            continue;
        }
        int numSegs = (!type_is_cubic)
                       ? (count - 1)
                       : (periodic ? (count / vStep > 1 ? count / vStep : 1)
                                    : ((count - 4) / vStep + 1));
        int n_w = periodic ? numSegs : (numSegs + 1);
        if (s + n_w > src_count) n_w = src_count - s;
        if (n_w <= 0) break;
        /* For each CV in this curve, find its position along the curve
         * in segment-space and linearly interpolate src[s..s+n_w]. */
        for (int i = 0; i < count && v < nv; i++) {
            float t;
            if (count <= 1) t = 0.0f;
            else            t = (float)i / (float)(count - 1);
            float seg_pos = t * (float)(periodic ? n_w : (n_w - 1));
            int   k = (int)seg_pos;
            float u = seg_pos - (float)k;
            int   k1 = k + 1;
            if (k  >= n_w) k  = n_w - 1;
            if (k1 >= n_w) k1 = periodic ? 0 : n_w - 1;
            out[v++] = src[s + k] * (1.0f - u) + src[s + k1] * u;
        }
        s += n_w;
    }
    while (v < nv) {
        float fallback = (v > 0) ? out[v-1] : 0.05f;
        out[v++] = fallback;
    }
}

/* ----------------------------------------------------------------
 * BasisCurves: fan a primvar array (widths, colors) out to per-CV
 * (vertex-interpolation) given the source array length and the total
 * CV count. Returns 1 on success.
 *
 * Heuristic: nv-length → as-is, 1-length → broadcast, ncurves-length
 * → broadcast per curve, varying-length → per-segment-boundary fan,
 * otherwise → use fallback. Storm consults the authored interpolation
 * token to disambiguate; we don't have that exposed via nanousd yet
 * so length-based heuristic is the working compromise.
 * ---------------------------------------------------------------- */
static int fan_curve_primvar_f1(
    Arena* arena,
    const float* src, int src_count,
    int nv, int ncurves, const int* counts,
    int type_is_cubic, SceneCurveBasis basis, SceneCurveWrap wrap,
    float fallback,
    float** out)
{
    float* dst = (float*)arena_alloc(arena, (size_t)nv * sizeof(float), 16);
    if (!dst) return 0;
    int varying_n = varying_count_for_curves(counts, ncurves, type_is_cubic,
                                              basis, wrap);
    if (src && src_count == nv) {
        memcpy(dst, src, (size_t)nv * sizeof(float));
    } else if (src && src_count == 1) {
        for (int i = 0; i < nv; i++) dst[i] = src[0];
    } else if (src && src_count == ncurves && counts) {
        int v = 0;
        for (int c = 0; c < ncurves; c++) {
            for (int k = 0; k < counts[c] && v < nv; k++)
                dst[v++] = src[c];
        }
        while (v < nv) dst[v++] = fallback;
    } else if (src && src_count == varying_n) {
        fan_varying_to_vertex(src, src_count, (int*)counts, ncurves,
                              type_is_cubic, basis, wrap, dst, nv);
    } else {
        for (int i = 0; i < nv; i++) dst[i] = fallback;
    }
    *out = dst;
    return 1;
}

/* float3 variant — same heuristic, layout = nv * 3 floats. */
static int fan_curve_primvar_f3(
    Arena* arena,
    const float* src, int src_count,  /* count is total floats, not vec3s */
    int nv, int ncurves, const int* counts,
    const float fallback[3],
    float** out)
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
            dst[v*3+0] = fallback[0];
            dst[v*3+1] = fallback[1];
            dst[v*3+2] = fallback[2];
            v++;
        }
    } else {
        for (int i = 0; i < nv; i++) {
            dst[i*3+0] = fallback[0];
            dst[i*3+1] = fallback[1];
            dst[i*3+2] = fallback[2];
        }
    }
    *out = dst;
    return 1;
}

/* Snapshot a nanousd attribute array into the arena. The zero-copy
 * pointer returned by nanousd_arraydataf may be invalidated by a
 * later call against the same prim, so always copy. */
static const float* snapshot_arrayf(NanousdPrim prim, const char* name,
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

static const int* snapshot_arrayi(NanousdPrim prim, const char* name,
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
 * BasisCurves: load a single prim into a SceneCurve. Returns 1 on
 * success, 0 if the prim has no usable curve data.
 * ---------------------------------------------------------------- */
/* ----------------------------------------------------------------
 * UsdGeomSphere / UsdGeomCylinder: generate triangle meshes CPU-side
 * and push into the existing scene->meshes[] pipeline. Mirrors what
 * Storm's UsdImagingSphereAdapter / CylinderAdapter do — author the
 * implicit shape's polygonal representation at adapter time so the
 * downstream renderer just sees a Mesh. Defaults from
 * ~/OpenUSD/pxr/usd/usdGeom/schema.usda: sphere radius=1.0,
 * cylinder radius=1.0 height=2.0 axis="Z".
 * ---------------------------------------------------------------- */

static void gen_uv_sphere(Arena* arena, double radius, int n_lat, int n_lon,
                          float** out_pos, float** out_nrm, uint32_t** out_idx,
                          int* out_nv, int* out_ni)
{
    /* (n_lat-1) bands of n_lon quads + 2 polar caps. We emit n_lat × n_lon
     * vertices, with poles duplicated per longitude so UV/normal seams
     * don't collapse. */
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
    /* axis: 0=X, 1=Y, 2=Z (USD's UsdGeomCylinder default is "Z"). Side
     * vertices are duplicated per cap to keep flat cap normals; total
     * verts = 2*n_sides (sides) + 2*(n_sides+1) (caps with center). */
    int n_side = n_sides * 2;
    int n_cap  = (n_sides + 1) * 2;
    int nv = n_side + n_cap;
    int ni = n_sides * 6 + n_sides * 6;  /* 2 tris per side quad + 1 fan tri per cap */
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
    /* axis-aligned helpers: (a0, a1, a2) is the local frame where a2 is
     * the cylinder axis. */
    int a0 = (axis + 1) % 3;
    int a1 = (axis + 2) % 3;
    int a2 = axis;
    /* side strip — sides duplicate per ring so cap-normal blending doesn't
     * smooth side normals. */
    for (int s = 0; s < n_sides; s++) {
        double ph = (double)s / (double)n_sides * 2.0 * 3.14159265358979323846;
        float c = (float)cos(ph), si = (float)sin(ph);
        for (int top = 0; top < 2; top++) {
            float n[3] = {0,0,0}; n[a0] = c; n[a1] = si;
            float p[3] = {0,0,0};
            p[a0] = (float)(radius * c);
            p[a1] = (float)(radius * si);
            p[a2] = top ? (float)half_h : (float)-half_h;
            pos[v*3+0]=p[0]; pos[v*3+1]=p[1]; pos[v*3+2]=p[2];
            nrm[v*3+0]=n[0]; nrm[v*3+1]=n[1]; nrm[v*3+2]=n[2];
            v++;
        }
    }
    /* caps */
    for (int top = 0; top < 2; top++) {
        /* center vertex */
        float n[3] = {0,0,0}; n[a2] = top ? 1.0f : -1.0f;
        float p[3] = {0,0,0}; p[a2] = top ? (float)half_h : (float)-half_h;
        pos[v*3+0]=p[0]; pos[v*3+1]=p[1]; pos[v*3+2]=p[2];
        nrm[v*3+0]=n[0]; nrm[v*3+1]=n[1]; nrm[v*3+2]=n[2];
        v++;
        for (int s = 0; s < n_sides; s++) {
            double ph = (double)s / (double)n_sides * 2.0 * 3.14159265358979323846;
            float c = (float)cos(ph), si = (float)sin(ph);
            float pp[3] = {0,0,0};
            pp[a0] = (float)(radius * c);
            pp[a1] = (float)(radius * si);
            pp[a2] = top ? (float)half_h : (float)-half_h;
            pos[v*3+0]=pp[0]; pos[v*3+1]=pp[1]; pos[v*3+2]=pp[2];
            nrm[v*3+0]=n[0]; nrm[v*3+1]=n[1]; nrm[v*3+2]=n[2];
            v++;
        }
    }
    int q = 0;
    /* sides: pair (s_bot=2s, s_top=2s+1) and ((s+1)%n)_bot, _top */
    for (int s = 0; s < n_sides; s++) {
        int s2 = ((s + 1) % n_sides) * 2;
        int b0 = s*2, t0 = s*2+1, b1 = s2, t1 = s2+1;
        idx[q*6+0]=(uint32_t)b0; idx[q*6+1]=(uint32_t)t0; idx[q*6+2]=(uint32_t)t1;
        idx[q*6+3]=(uint32_t)b0; idx[q*6+4]=(uint32_t)t1; idx[q*6+5]=(uint32_t)b1;
        q++;
    }
    /* caps: bottom cap fan (CCW from outside means winding flips between caps) */
    int bot_center = n_side;
    int top_center = n_side + (n_sides + 1);
    for (int s = 0; s < n_sides; s++) {
        int s1 = (s + 1) % n_sides;
        /* bottom: outward normal -axis → wind reversed for CCW from below */
        idx[q*6+0]=(uint32_t)bot_center; idx[q*6+1]=(uint32_t)(bot_center + 1 + s1); idx[q*6+2]=(uint32_t)(bot_center + 1 + s);
        /* top: outward normal +axis → CCW from above */
        idx[q*6+3]=(uint32_t)top_center; idx[q*6+4]=(uint32_t)(top_center + 1 + s); idx[q*6+5]=(uint32_t)(top_center + 1 + s1);
        q++;
    }
    *out_pos = pos; *out_nrm = nrm; *out_idx = idx;
    *out_nv = nv;   *out_ni = ni;
}

/* Populate a SceneMesh slot from a generated implicit-shape mesh.
 * Keep positions object-space like Mesh prims; the renderer applies
 * world_xform, and bounds can be expanded from the local AABB corners. */
static int finalize_implicit_mesh(SceneMesh* mesh, NanousdPrim prim,
                                   float* obj_pos, float* obj_nrm,
                                   uint32_t* indices, int nv, int ni,
                                   Arena* arena)
{
    if (!obj_pos || !obj_nrm || !indices || nv == 0 || ni == 0) return 0;
    mesh->positions = obj_pos;
    mesh->normals   = obj_nrm;
    mesh->indices   = indices;
    mesh->nvertices = nv;
    mesh->nindices  = ni;
    mesh->material_index = -1;
    mesh->prototype_idx  = 0;  /* viewer.c will override */
    /* primvars:displayColor — single value common pattern for implicits */
    int dc_n = 0;
    const float* dc = snapshot_arrayf(prim, "primvars:displayColor", arena, &dc_n);
    if (dc && dc_n >= 3) {
        mesh->display_color[0] = dc[0];
        mesh->display_color[1] = dc[1];
        mesh->display_color[2] = dc[2];
        mesh->has_display_color = 1;
    }
    compute_world_xform(prim, mesh->world_xform);
    mesh_compute_local_bounds(mesh);
    mesh_compute_world_bounds_from_local(mesh, NULL);
    return 1;
}

/* UsdGeomCapsule: cylinder body of length `height` + two true-radius hemisphere
 * caps, tessellated at TRUE scale (a capsule can't be a unit primitive scaled
 * non-uniformly — that flattens the caps into ellipsoids). Axis-generic.
 * Degenerate-safe: radius<=0 -> empty; height<=0 -> sphere. Mirrors the Vulkan
 * renderer's gen_capsule. */
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

    int n_lat = 2 * (n_rings + 1);
    int nv = n_lat * n_sides;
    int ni = (n_lat - 1) * n_sides * 6;
    float* pos = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    float* nrm = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    uint32_t* idx = (uint32_t*)arena_alloc(arena, (size_t)ni * sizeof(uint32_t), 4);
    if (!pos || !nrm || !idx) return;

    for (int r = 0; r < n_lat; r++) {
        double alpha, a2off;
        if (r <= n_rings) {
            alpha = -PI * 0.5 + (PI * 0.5) * ((double)r / (double)n_rings);
            a2off = -half_h + radius * sin(alpha);
        } else {
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
            idx[q*6+0] = a;  idx[q*6+1] = cc; idx[q*6+2] = d;
            idx[q*6+3] = a;  idx[q*6+4] = d;  idx[q*6+5] = b;
            q++;
        }
    }
    *out_pos = pos; *out_nrm = nrm; *out_idx = idx; *out_nv = nv; *out_ni = ni;
}

/* UsdGeomCube: box of edge `size` (+-size/2), 24 verts (per-face flat normals). */
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
            p[u] = (float)(corners[c][0] * h * ns); p[w] = (float)(corners[c][1] * h);
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

/* UsdGeomCone: base ring of `radius` at -height/2, apex at +height/2 + base cap.
 * True slant normals; split apex (per-side face normal). Mirrors Vulkan gen_cone. */
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
    int nv = n_sides * 2 + 1 + n_sides;
    int ni = n_sides * 3 + n_sides * 3;
    float* pos = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    float* nrm = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    uint32_t* idx = (uint32_t*)arena_alloc(arena, (size_t)ni * sizeof(uint32_t), 4);
    if (!pos || !nrm || !idx) return;
    double slant = sqrt(radius * radius + height * height);
    double n_axial = (slant > 0) ? radius / slant : 0.0;
    double n_radial = (slant > 0) ? height / slant : 1.0;
    int v = 0;
    for (int s = 0; s < n_sides; s++) {
        double ph = (double)s / (double)n_sides * 2.0 * PI;
        double phm = ((double)s + 0.5) / (double)n_sides * 2.0 * PI;
        float c = (float)cos(ph), si = (float)sin(ph);
        float cm = (float)cos(phm), sim = (float)sin(phm);
        float pb[3] = {0,0,0}, nb[3] = {0,0,0};
        pb[a0] = (float)(radius * c); pb[a1] = (float)(radius * si); pb[a2] = (float)-half_h;
        nb[a0] = (float)(n_radial * c); nb[a1] = (float)(n_radial * si); nb[a2] = (float)n_axial;
        pos[v*3+0]=pb[0]; pos[v*3+1]=pb[1]; pos[v*3+2]=pb[2];
        nrm[v*3+0]=nb[0]; nrm[v*3+1]=nb[1]; nrm[v*3+2]=nb[2]; v++;
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
        idx[q++] = (uint32_t)(s*2); idx[q++] = (uint32_t)(s1*2); idx[q++] = (uint32_t)(s*2+1);
        idx[q++] = (uint32_t)cap_center;
        idx[q++] = (uint32_t)(cap_center + 1 + s);
        idx[q++] = (uint32_t)(cap_center + 1 + s1);
    }
    *out_pos = pos; *out_nrm = nrm; *out_idx = idx; *out_nv = nv; *out_ni = ni;
}

static int load_sphere(NanousdPrim prim, Arena* arena, SceneMesh* mesh)
{
    int ok = 0;
    double radius = nanousd_attribd(prim, "radius", &ok);
    if (!ok) radius = 1.0;
    float* pos = NULL; float* nrm = NULL; uint32_t* idx = NULL;
    int nv = 0, ni = 0;
    gen_uv_sphere(arena, radius, 24, 48, &pos, &nrm, &idx, &nv, &ni);
    return finalize_implicit_mesh(mesh, prim, pos, nrm, idx, nv, ni, arena);
}

static int load_cylinder(NanousdPrim prim, Arena* arena, SceneMesh* mesh)
{
    int ok = 0;
    double radius = nanousd_attribd(prim, "radius", &ok);
    if (!ok) radius = 1.0;
    ok = 0;
    double height = nanousd_attribd(prim, "height", &ok);
    if (!ok) height = 2.0;
    /* axis: token ("X" / "Y" / "Z"), default "Z" per UsdGeomCylinder. */
    int axis = 2;
    int axis_ok = 0;
    const char* axis_tok = nanousd_attrib_token(prim, "axis", &axis_ok);
    if (axis_ok && axis_tok) {
        if      (axis_tok[0] == 'X' || axis_tok[0] == 'x') axis = 0;
        else if (axis_tok[0] == 'Y' || axis_tok[0] == 'y') axis = 1;
        else                                                axis = 2;
    }
    float* pos = NULL; float* nrm = NULL; uint32_t* idx = NULL;
    int nv = 0, ni = 0;
    gen_cylinder(arena, radius, height, axis, 32, &pos, &nrm, &idx, &nv, &ni);
    return finalize_implicit_mesh(mesh, prim, pos, nrm, idx, nv, ni, arena);
}

static int load_capsule(NanousdPrim prim, Arena* arena, SceneMesh* mesh)
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
    float* pos = NULL; float* nrm = NULL; uint32_t* idx = NULL;
    int nv = 0, ni = 0;
    gen_capsule(arena, radius, height, axis, 24, 8, &pos, &nrm, &idx, &nv, &ni);
    return finalize_implicit_mesh(mesh, prim, pos, nrm, idx, nv, ni, arena);
}

static int load_cube(NanousdPrim prim, Arena* arena, SceneMesh* mesh)
{
    int ok = 0;
    double size = nanousd_attribd(prim, "size", &ok);
    if (!ok) size = 2.0;
    float* pos = NULL; float* nrm = NULL; uint32_t* idx = NULL; int nv = 0, ni = 0;
    gen_cube(arena, size, &pos, &nrm, &idx, &nv, &ni);
    return finalize_implicit_mesh(mesh, prim, pos, nrm, idx, nv, ni, arena);
}

static int load_cone(NanousdPrim prim, Arena* arena, SceneMesh* mesh)
{
    int ok = 0;
    double radius = nanousd_attribd(prim, "radius", &ok);
    if (!ok) radius = 1.0;
    ok = 0;
    double height = nanousd_attribd(prim, "height", &ok);
    if (!ok) height = 2.0;
    int axis = 2, axis_ok = 0;
    const char* axis_tok = nanousd_attrib_token(prim, "axis", &axis_ok);
    if (axis_ok && axis_tok) {
        if      (axis_tok[0] == 'X' || axis_tok[0] == 'x') axis = 0;
        else if (axis_tok[0] == 'Y' || axis_tok[0] == 'y') axis = 1;
        else                                                axis = 2;
    }
    float* pos = NULL; float* nrm = NULL; uint32_t* idx = NULL; int nv = 0, ni = 0;
    gen_cone(arena, radius, height, axis, 32, &pos, &nrm, &idx, &nv, &ni);
    return finalize_implicit_mesh(mesh, prim, pos, nrm, idx, nv, ni, arena);
}

static int load_basiscurves(NanousdPrim prim, Arena* arena, SceneCurve* curve)
{
    /* (a) points */
    int pts_count = 0;
    const float* pts = snapshot_arrayf(prim, "points", arena, &pts_count);
    if (!pts || pts_count < 3) return 0;
    int nv = pts_count / 3;
    if (nv < 2) return 0;

    /* (b) curveVertexCounts */
    int cvc_count = 0;
    const int* cvc = snapshot_arrayi(prim, "curveVertexCounts", arena, &cvc_count);
    if (!cvc || cvc_count <= 0) {
        /* Single curve with all CVs */
        int* tmp = (int*)arena_alloc(arena, sizeof(int), 4);
        if (!tmp) return 0;
        tmp[0] = nv;
        cvc = tmp;
        cvc_count = 1;
    }
    /* Validate sum(cvc) == nv */
    {
        int sum = 0;
        for (int i = 0; i < cvc_count; i++) sum += cvc[i];
        if (sum != nv) {
            fprintf(stderr, "scene_load: BasisCurves curveVertexCounts sum (%d) != points (%d), skipping\n",
                    sum, nv);
            return 0;
        }
    }

    /* (c) topology tokens — defaults per USD spec.
     * `type`/`basis`/`wrap` are USD token-typed attributes; nanousd
     * distinguishes string vs token readers (see nanousdapi.h:506-512). */
    /* nanousd_attrib_token reuses an internal buffer across calls, so
     * snapshot each token into a local stack copy before the next call
     * can clobber it. */
    char type_buf[32] = {0}, basis_buf[32] = {0}, wrap_buf[32] = {0};
    int ok_t = 0, ok_b = 0, ok_w = 0;
    {
        const char* s = nanousd_attrib_token(prim, "type", &ok_t);
        if (ok_t && s) { snprintf(type_buf, sizeof(type_buf), "%s", s); }
    }
    {
        const char* s = nanousd_attrib_token(prim, "basis", &ok_b);
        if (ok_b && s) { snprintf(basis_buf, sizeof(basis_buf), "%s", s); }
    }
    {
        const char* s = nanousd_attrib_token(prim, "wrap", &ok_w);
        if (ok_w && s) { snprintf(wrap_buf, sizeof(wrap_buf), "%s", s); }
    }
    int type_is_cubic = (!ok_t || !type_buf[0] || strcmp(type_buf, "cubic") == 0) ? 1 : 0;
    SceneCurveBasis basis = type_is_cubic ? parse_curve_basis(ok_b ? basis_buf : NULL)
                                          : CURVE_BASIS_LINEAR;
    SceneCurveWrap wrap = parse_curve_wrap(ok_w ? wrap_buf : NULL);

    /* (d) widths — fan to per-CV */
    int widths_n = 0;
    const float* widths_src = snapshot_arrayf(prim, "widths", arena, &widths_n);
    float* widths = NULL;
    if (!fan_curve_primvar_f1(arena, widths_src, widths_n, nv, cvc_count, cvc,
                              type_is_cubic, basis, wrap,
                              0.05f, &widths)) return 0;

    /* (e) displayColor — fan to per-CV (or store as fallback if length 1
     * and the curve has uniform color, used by the override-color path) */
    int dc_n = 0;
    const float* dc_src = snapshot_arrayf(prim, "primvars:displayColor", arena, &dc_n);
    float* colors = NULL;
    int has_dc = 0;
    float dc_fallback[3] = { 0.7f, 0.7f, 0.7f };
    if (dc_src && dc_n == 3) {
        /* Single color — store as override + still fan into per-CV (cheap) */
        curve->display_color[0] = dc_src[0];
        curve->display_color[1] = dc_src[1];
        curve->display_color[2] = dc_src[2];
        has_dc = 1;
        dc_fallback[0] = dc_src[0];
        dc_fallback[1] = dc_src[1];
        dc_fallback[2] = dc_src[2];
    }
    if (!fan_curve_primvar_f3(arena, dc_src, dc_n, nv, cvc_count, cvc,
                              dc_fallback, &colors)) return 0;

    /* (f) world transform — kept on the SceneCurve and forwarded to the
     * shader as mesh.model each frame. CVs themselves stay object-space
     * so subsequent metersPerUnit + Z-up rotation passes can update
     * world_xform once (mirrors the mesh path). The shader uses
     * mesh.model to transform both CV positions AND ring frame vectors
     * — keeping CVs object-space lets the ring-direction rotate
     * correctly into world space. */
    compute_world_xform(prim, curve->world_xform);

    /* (g) Object-space CV copy + world-space bounds (compute via
     * xform_point so scene framing works regardless of authored space). */
    float* obj_cvs = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    if (!obj_cvs) return 0;
    memcpy(obj_cvs, pts, (size_t)nv * 3 * sizeof(float));
    curve->bounds_min[0] = curve->bounds_min[1] = curve->bounds_min[2] =  FLT_MAX;
    curve->bounds_max[0] = curve->bounds_max[1] = curve->bounds_max[2] = -FLT_MAX;
    for (int v = 0; v < nv; v++) {
        float wp[3];
        xform_point(curve->world_xform, &obj_cvs[v*3], wp);
        for (int k = 0; k < 3; k++) {
            if (wp[k] < curve->bounds_min[k]) curve->bounds_min[k] = wp[k];
            if (wp[k] > curve->bounds_max[k]) curve->bounds_max[k] = wp[k];
        }
    }

    /* (h) build patch indices (cubic + linear; periodic + pinned). */
    uint32_t* patch_idx = NULL;
    int npatches = 0;
    build_curve_patches(arena, cvc, cvc_count, type_is_cubic,
                        basis, wrap, &patch_idx, &npatches);

    /* Persist into SceneCurve */
    curve->cvs                 = obj_cvs;
    curve->widths              = widths;
    curve->colors              = colors;
    curve->nv                  = nv;
    curve->ncurves_in_prim     = cvc_count;
    {
        int* cvc_copy = (int*)arena_alloc(arena, (size_t)cvc_count * sizeof(int), 4);
        if (cvc_copy) memcpy(cvc_copy, cvc, (size_t)cvc_count * sizeof(int));
        curve->curve_vertex_counts = cvc_copy;
    }
    curve->patch_indices       = patch_idx;
    curve->npatches            = npatches;
    curve->basis               = basis;
    curve->wrap                = wrap;
    curve->type_is_cubic       = type_is_cubic;
    curve->has_display_color   = has_dc;
    return 1;
}

/* ===================================================================
 * Path B: native-instance asset-arc flat seed/replay + compact PI loader.
 * Ported from nanousd-vulkan-renderer/src/scene.c so OpenGL recovers the
 * deep/nested Moana geometry the composed-child walk drops (and avoids the
 * O(N^2) cliff that walk hits at island scale). Gated identically to Vulkan:
 * active only when scene_compact_pi_batches_active() AND
 * nb_flat_native_replay_active() (see the env helpers below). When the gate
 * is off, the existing composed-child walk + per-instance PI clone passes run
 * unchanged.
 *
 * OpenGL adaptations vs Vulkan:
 *  - mesh->path is heap-owned (scene_free frees every meshes[i].path), so we
 *    use scene_strdup, NOT scene_arena_strdup.
 *  - triangulate_faces() here is 5-arg (no fan-out UV split param).
 *  - no MaterialCollection on Scene: material_index stays -1 (material.c
 *    resolves bindings in a later pass, same as the existing OpenGL passes).
 * =================================================================== */

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

/* Compact PI batches gate. Mirrors the Vulkan helper of the same name: enabled
 * by the no-cull / RT-cull / explicit PI-batch env vars, disabled by the legacy
 * geometry-defaults env vars. */
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
    return 0;
}

/* Flat native-instance asset-arc replay gate. */
static int nb_flat_native_replay_active(void)
{
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
    return 0;
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

/* Nested-PointInstancer record (file scope; both the asset-arc seeder and the
 * composed walk append to the same list, drained after the instance loop). */
struct NestedPiRec { char path[1024]; double pi_to_inst[16]; };

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
    NbPiEntry*     pi_entries;   /* nested PIs under the prototype */
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

/* Top-level PointInstancer path "skip set": PIs that the main-stage PI pass
 * already owns must not also be recorded as nested PIs by the asset seeder. */
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

/* Prototype-target collection for a PointInstancer (rel targets, then child
 * fallback). Mirrors the Vulkan helper. */
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

/* Splice the missing reference prefix onto a PI prototype rel target path so it
 * matches the composed main-stage path. Mirrors the Vulkan helper (and the
 * existing OpenGL PI third-pass prefix-splice). */
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
    /* Fallback when instance_key is unavailable/empty: key on the prototype
     * path (mirrors the Vulkan fallback). */
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
    /* Normalize so callers can't pass an inconsistent {NULL slots, nonzero
     * cap} pair and trigger a NULL deref in the rehash loop. */
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
 * displayColor) from an asset-stage Mesh prim into a SceneMesh. Returns 1 on
 * success. Geometry is arena-allocated. */
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
        mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)tri * sizeof(uint32_t), 16);
        if (!mesh->indices) return 0;
        mesh->nindices = triangulate_faces(fvi, fvi_count, fvc, fvc_count, mesh->indices);
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
    /* OpenGL resolves material bindings in a later pass (material.c). */
    mesh->material_index = -1;
    mesh->visible = 1;
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

/* Seed the geometry of one asset substage (read flat, not composed) into
 * scene->meshes as is_proto_only prototype sub-meshes, record nested PIs for
 * the post-loop drain, and chase child reference/payload arcs. Returns the
 * number of emitted/handled items. Recursive (depth-capped). */
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
        m->path = scene_strdup(concrete_path);
        m->prototype_idx = *mesh_idx;
        m->lazy_prim_idx = -1;
        m->visible = 1;
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

    /* Child references/payloads can contain visible XGen/prototype geometry
     * even when the owning asset layer also authored terrain meshes. */
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
                int child_emitted = nb_seed_asset_stage_recursive(
                    scene, main_stage, inst_prim, sub, child_target,
                    catalog_root_path, inst_world, concrete_path, child_root_rel,
                    cat, arena, mesh_idx, max_meshes, total_verts, total_indices,
                    nested, nested_n, nested_cap, pi_xform_cap, pi_batch_cap,
                    child_emit_native_batches, depth + 1);
                emitted += child_emitted;
                nanousd_close(sub);
            }
            nanousd_listop_free(op);
        }
        nanousd_freeprim(ap);
    }

    if (!target_seen_in_flat && *mesh_idx < max_meshes) {
        NanousdPrim ap = nanousd_primpath(asset_stage, tpath);
        if (ap) {
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
                int child_emitted = nb_seed_asset_stage_recursive(
                    scene, main_stage, inst_prim, sub, child_target,
                    catalog_root_path, inst_world, concrete_path, child_root_rel,
                    cat, arena, mesh_idx, max_meshes, total_verts, total_indices,
                    nested, nested_n, nested_cap, pi_xform_cap, pi_batch_cap,
                    child_emit_native_batches, depth + 1);
                processed_direct_arcs++;
                emitted += child_emitted;
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
                    int child_emitted = nb_seed_asset_stage_recursive(
                        scene, main_stage, inst_prim, sub, child_target,
                        catalog_root_path, inst_world, concrete_path, child_root_rel,
                        cat, arena, mesh_idx, max_meshes, total_verts, total_indices,
                        nested, nested_n, nested_cap, pi_xform_cap, pi_batch_cap,
                        child_emit_native_batches, depth + 1);
                    emitted += child_emitted;
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

/* Replay a built catalog for a sibling instance of the same prototype: under
 * the compact gate this only emits native-instance batches (no SceneMesh
 * clones) and re-records the prototype's nested PIs with this instance's
 * world. */
static int nb_emit_replay(Scene* scene, int* mesh_idx, int max_meshes,
                          const char* inst_root_path, const double inst_world[16],
                          const NbReplayCatalog* cat, Arena* arena,
                          uint64_t* total_verts, uint64_t* total_indices,
                          struct NestedPiRec** nested, int* nested_n, int* nested_cap,
                          uint64_t* pi_xform_cap, int* pi_batch_cap,
                          int emit_native_batches) {
    (void)arena;  /* paths are heap-owned via scene_strdup here, not arena */
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
        m->visible = 1;
        memcpy(m->local_bounds_min, pm->local_bounds_min, sizeof(float) * 3);
        memcpy(m->local_bounds_max, pm->local_bounds_max, sizeof(float) * 3);
        memcpy(m->world_xform, world, sizeof(world));
        char pathbuf[1024];
        if (inst_root_path && inst_root_path[0] && e->relative_path[0])
            snprintf(pathbuf, sizeof(pathbuf), "%s/%s", inst_root_path, e->relative_path);
        else if (inst_root_path && inst_root_path[0])
            snprintf(pathbuf, sizeof(pathbuf), "%s", inst_root_path);
        else pathbuf[0] = '\0';
        m->path = scene_strdup(pathbuf);
        mesh_compute_world_bounds_from_local(m, scene);
        m->vertex_offset = 0; m->index_offset = 0;
        if (total_verts) *total_verts += m->nvertices;
        if (total_indices) *total_indices += m->nindices;
        (*mesh_idx)++; emitted++;
    }
    /* Re-record the prototype's nested PIs for THIS sibling instance. */
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

        char pi_pref[32][512];
        int  npi_pref = 0;

        for (int i = 0; i < nflat && *mesh_idx < max_meshes; i++) {
            const char* fp = flat[i].path;
            const char* tn = flat[i].type_name;
            if (!fp || !tn) continue;
            if (!nb_path_has_prefix(fp, tpath)) continue;
            if (strcmp(tn, "PointInstancer") == 0) {
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
            m->path = scene_strdup(pathbuf);
            m->prototype_idx = *mesh_idx;  /* self = prototype source */
            m->visible = 1;
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

/* ----------------------------------------------------------------
 * scene_load
 * ---------------------------------------------------------------- */
Scene* scene_load(const char* filepath)
{
    double t0 = get_time_sec();
    int do_timing = (getenv("NUSD_SCENE_TIMING") != NULL ||
                     getenv("NUSD_LOAD_TIMING") != NULL);
    double t_mark = t0;

    /* Geometry cache (NUSD_GEO_CACHE=1): a warm hit returns a fully-built
     * Scene from <usd>.nzgeo.gl and skips the USD parse entirely. Disabled
     * in lazy-mesh mode — the two load modes are mutually exclusive. */
    int geo_cache_active;
    {
        const char* lz = getenv("NUSD_LAZY_MESH");
        int lazy = (lz && lz[0] && lz[0] != '0');
        geo_cache_active = geo_cache_enabled() && !lazy;
    }
    if (geo_cache_active) {
        Scene* cached = geo_cache_try_load(filepath);
        if (cached) {
            fprintf(stderr, "[geo_cache] HIT: %s — %d meshes (%.1f ms)\n",
                    filepath, cached->nmeshes, (get_time_sec() - t0) * 1000.0);
            return cached;
        }
        fprintf(stderr, "[geo_cache] MISS: %s — parsing USD\n", filepath);
    }

    /* Open stage. nanousd_open does USDA parse + Compose + Populate eagerly,
     * so this bucket is the real load-time cost; it must NOT be folded into
     * the nanousd_nprims bucket below. */
    NanousdStage stage = nanousd_open(filepath);
    if (!nanousd_isvalid(stage)) {
        const char* err = nanousd_error(stage);
        fprintf(stderr, "scene_load: failed to open '%s': %s\n",
                filepath, err ? err : "unknown error");
        if (stage) nanousd_close(stage);
        return NULL;
    }
    if (do_timing) {
        double t_now = get_time_sec();
        fprintf(stderr, "  [scene_timing] nanousd_open (parse+compose+populate): %.1f ms\n",
                (t_now - t_mark) * 1000.0);
        t_mark = t_now;
    }

    int nprims = nanousd_nprims(stage);
    if (do_timing) {
        double t_now = get_time_sec();
        fprintf(stderr, "  [scene_timing] nanousd_nprims: %.1f ms (nprims=%d)\n",
                (t_now - t_mark) * 1000.0, nprims);
        t_mark = t_now;
    }

    /* Allocate Scene and Arena */
    Scene* scene = (Scene*)calloc(1, sizeof(Scene));
    if (!scene) {
        nanousd_close(stage);
        return NULL;
    }

    Arena* arena = (Arena*)malloc(sizeof(Arena));
    if (!arena) {
        free(scene);
        nanousd_close(stage);
        return NULL;
    }
    /* Scale arena to scene complexity: ~4KB per prim is a reasonable heuristic
     * for mesh-heavy scenes (positions + normals + indices + colors).
     * Minimum 64MB, so small scenes don't under-allocate. */
    size_t arena_size = (size_t)nprims * 4096;
    if (arena_size < 64 * 1024 * 1024) arena_size = 64 * 1024 * 1024;
    *arena = arena_create(arena_size);
    if (!arena->head) {
        free(arena);
        free(scene);
        nanousd_close(stage);
        return NULL;
    }

    scene->_arena = arena;
    scene->_stage = stage;

    for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;
        if (nanousd_isactive(prim)) {
            const char* tn = nanousd_typename(prim);
            if (tn && strstr(tn, "Light") != NULL)
                scene->has_authored_light = 1;
        }
        nanousd_freeprim(prim);
    }

    /* Over-allocate mesh/curve arrays. With instancing, the total mesh count
     * can exceed nprims. Small synthetic PointInstancer stress scenes keep the
     * old 16x headroom; huge composed scenes such as DSX would otherwise spend
     * time and memory zeroing millions of unused SceneMesh slots. Override with
     * NUSD_SCENE_MESH_FACTOR when testing pathological expansion cases. */
    int mesh_factor = (nprims > 100000) ? 4 : (nprims < 1024 ? 64 : 16);
    /* Path B native-instance asset-arc seeding (gated) materializes unique
     * prototype meshes that aren't in the composed prim count, so give it
     * headroom — mirrors the Vulkan renderer's gate-conditional bump. */
    if (scene_compact_pi_batches_active() && mesh_factor < 32)
        mesh_factor = 32;
    const char* mesh_factor_env = getenv("NUSD_SCENE_MESH_FACTOR");
    if (mesh_factor_env && mesh_factor_env[0]) {
        long parsed = strtol(mesh_factor_env, NULL, 10);
        if (parsed >= 1 && parsed <= 64) mesh_factor = (int)parsed;
    }
    int max_meshes = nprims * mesh_factor;
    if (scene_compact_pi_batches_active() && max_meshes < 4096)
        max_meshes = 4096;
    scene->meshes = (SceneMesh*)calloc((size_t)max_meshes, sizeof(SceneMesh));
    if (!scene->meshes) {
        scene_free(scene);
        return NULL;
    }
    scene->_meshes_heap = 1;

    /* With instancing, curve count can exceed nprims (each instance
     * prim expands to multiple curve copies via the second-pass
     * walker). Use the same factor as the mesh array. */
    int max_curves = nprims * mesh_factor;
    scene->curves = (SceneCurve*)arena_calloc(arena, (size_t)max_curves, sizeof(SceneCurve));

    XformCache xform_cache = { NULL, 0, 0 };
    g_xform_cache = &xform_cache;

    PathPrefixList invisible_prefixes = { NULL, 0, 0 };
    PathPrefixList point_instancer_prefixes = { NULL, 0, 0 };
    int prefix_prescan = 0;
    {
        const char* e = getenv("NUSD_SCENE_PREFIX_PRESCAN");
        prefix_prescan = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    if (prefix_prescan) {
        collect_scene_prefixes(stage, nprims, &invisible_prefixes,
                               &point_instancer_prefixes);
        if (do_timing) {
            double t_now = get_time_sec();
            fprintf(stderr,
                    "  [scene_timing] visibility/point-instancer prefix scan: %.1f ms "
                    "(invisible=%d, pointInstancer=%d)\n",
                    (t_now - t_mark) * 1000.0,
                    invisible_prefixes.count,
                    point_instancer_prefixes.count);
            t_mark = t_now;
        }
    }

    /* Initialize bounds */
    scene->bounds_min[0] = FLT_MAX;
    scene->bounds_min[1] = FLT_MAX;
    scene->bounds_min[2] = FLT_MAX;
    scene->bounds_max[0] = -FLT_MAX;
    scene->bounds_max[1] = -FLT_MAX;
    scene->bounds_max[2] = -FLT_MAX;

    uint64_t total_verts = 0;
    uint64_t total_indices = 0;
    /* Declared before the NUSD_LAZY_MESH goto so it is initialized on both
     * the eager and the metadata-only load paths. */
    int instanced_meshes = 0;

    /* NUSD_LAZY_MESH=1 step 2 — mirror Vulkan's metadata walk
     * (see docs/plans/TIER_3_LAZY_MESH.md). Emit SceneMesh records
     * with metadata only (nvertices, world_xform, lazy_prim_idx);
     * positions/normals/indices stay NULL. Instance-prim and PI
     * expansion are skipped in step 2 (deferred to step 2.5).
     *
     * Structural verification only — pixel gating returns when
     * step 3 implements the renderer-side extract-deferred path. */
    {
        const char* e = getenv("NUSD_LAZY_MESH");
        int lazy_mesh = (e && e[0] && e[0] != '0') ? 1 : 0;
        if (lazy_mesh) {
            fprintf(stderr,
                    "[scene_load] NUSD_LAZY_MESH=1 — metadata-only walk of flat prim list "
                    "(Tier 3 step 2; see docs/plans/TIER_3_LAZY_MESH.md).\n");
            int lazy_idx = 0;
            int lazy_n_mesh = 0, lazy_n_sphere = 0, lazy_n_cyl = 0;
            for (int i = 0; i < nprims && lazy_idx < max_meshes; i++) {
                NanousdPrim p = nanousd_prim(stage, i);
                if (!p) continue;
                if (!nanousd_isactive(p)) { nanousd_freeprim(p); continue; }
                const char* tn = nanousd_typename(p);
                if (!tn) { nanousd_freeprim(p); continue; }
                int is_mesh = !strcmp(tn, "Mesh");
                int is_sphere = !strcmp(tn, "Sphere");
                int is_cyl = !strcmp(tn, "Cylinder");
                int is_capsule = !strcmp(tn, "Capsule");
                int is_cube = !strcmp(tn, "Cube");
                int is_cone = !strcmp(tn, "Cone");
                if (!is_mesh && !is_sphere && !is_cyl && !is_capsule && !is_cube && !is_cone) {
                    nanousd_freeprim(p);
                    continue;
                }
                char p_path_buf[8192];
                const char* p_path = scene_prim_path_copy(p, p_path_buf, sizeof(p_path_buf));
                SceneMesh* m = &scene->meshes[lazy_idx];
                m->positions = NULL;
                m->normals = NULL;
                m->colors = NULL;
                m->texcoords = NULL;
                m->indices = NULL;
                m->nvertices = is_mesh ? (nanousd_attribarraylen(p, "points") / 3) : 0;
                m->nindices = 0;
                compute_world_xform(p, m->world_xform);
                m->material_index = -1;
                m->prototype_idx = lazy_idx;
                m->is_proto_only = 0;
                m->path = scene_strdup(p_path);
                m->visible = 1;
                m->has_display_color = 0;
                m->display_color[0] = 0.7f;
                m->display_color[1] = 0.7f;
                m->display_color[2] = 0.7f;
                m->lazy_prim_idx = i;
                if (is_mesh) lazy_n_mesh++;
                else if (is_sphere) lazy_n_sphere++;
                else if (is_cyl) lazy_n_cyl++;
                lazy_idx++;
                nanousd_freeprim(p);
            }
            scene->nmeshes = lazy_idx;
            if (do_timing) {
                double t_now = get_time_sec();
                fprintf(stderr,
                        "  [scene_timing] flat mesh/implicit pass: %.1f ms "
                        "(LAZY metadata: %d meshes — %d Mesh, %d Sphere, %d Cylinder)\n",
                        (t_now - t_mark) * 1000.0,
                        lazy_idx, lazy_n_mesh, lazy_n_sphere, lazy_n_cyl);
                t_mark = t_now;
            }
            goto lazy_postprocess;
        }
    }

    /* --- Extract mesh data --- */
    int mesh_idx = 0;

    /* Prototype mesh lookup: maps prim path → mesh_idx for instancing */
    ProtoMeshHash proto_mesh_table = { NULL, 0, 0 };

    /* Collect instance prims for the second pass */
    int n_instance_prims = 0, inst_cap = 0;
    int* instance_prim_indices = NULL;

    /* First pass: extract meshes from flat prim list + collect instance prims */
    for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;

        /* Collect instance prims for second pass + record proto→inst mapping */
        if (nanousd_isinstance(prim)) {
            if (n_instance_prims >= inst_cap) {
                inst_cap = inst_cap ? inst_cap * 2 : 64;
                instance_prim_indices = (int*)realloc(instance_prim_indices, (size_t)inst_cap * sizeof(int));
            }
            instance_prim_indices[n_instance_prims++] = i;

        }

        const char* tn = nanousd_typename(prim);
        if (!nanousd_isactive(prim) || !tn) {
            nanousd_freeprim(prim);
            continue;
        }

        char prim_path_buf[8192];
        const char* prim_path = scene_prim_path_copy(prim, prim_path_buf, sizeof(prim_path_buf));
        if (!prefix_prescan) {
            if (!strcmp(tn, "PointInstancer"))
                path_prefix_list_add(&point_instancer_prefixes, prim_path);

            int vis_ok = 0;
            const char* vis = nanousd_attrib_token(prim, "visibility", &vis_ok);
            if (vis_ok && vis && !strcmp(vis, "invisible")) {
                path_prefix_list_add(&invisible_prefixes, prim_path);
                nanousd_freeprim(prim);
                continue;
            }
        }

        if ((!strcmp(tn, "Mesh") || !strcmp(tn, "Sphere") ||
             !strcmp(tn, "Cylinder") || !strcmp(tn, "BasisCurves")) &&
            prim_hidden_from_render_cached(prim, &invisible_prefixes)) {
            nanousd_freeprim(prim);
            continue;
        }

        /* BasisCurves: load into scene->curves[] and skip the mesh path. */
        if (strcmp(tn, "BasisCurves") == 0) {
            if (!scene_ensure_curve_capacity(scene, arena, &max_curves,
                                             scene->ncurves + 1)) {
                nanousd_freeprim(prim);
                continue;
            }
            SceneCurve* curve = &scene->curves[scene->ncurves];
            if (load_basiscurves(prim, arena, curve)) {
                /* Fold per-curve bounds into scene bounds. */
                for (int k = 0; k < 3; k++) {
                    if (curve->bounds_min[k] < scene->bounds_min[k])
                        scene->bounds_min[k] = curve->bounds_min[k];
                    if (curve->bounds_max[k] > scene->bounds_max[k])
                        scene->bounds_max[k] = curve->bounds_max[k];
                }
                scene->ncurves++;
            }
            nanousd_freeprim(prim);
            continue;
        }

        /* Sphere / Cylinder / Capsule / Cube / Cone: implicit-shape adapters
         * that generate a SceneMesh in-place (mirrors UsdImagingSphereAdapter). */
        if (strcmp(tn, "Sphere") == 0 || strcmp(tn, "Cylinder") == 0 ||
            strcmp(tn, "Capsule") == 0 || strcmp(tn, "Cube") == 0 ||
            strcmp(tn, "Cone") == 0) {
            if (!scene_ensure_mesh_capacity(scene, &max_meshes, mesh_idx + 1)) {
                nanousd_freeprim(prim);
                continue;
            }
            SceneMesh* mesh = &scene->meshes[mesh_idx];
            int ok = (strcmp(tn, "Sphere") == 0)   ? load_sphere(prim, arena, mesh)
                   : (strcmp(tn, "Capsule") == 0)  ? load_capsule(prim, arena, mesh)
                   : (strcmp(tn, "Cube") == 0)     ? load_cube(prim, arena, mesh)
                   : (strcmp(tn, "Cone") == 0)     ? load_cone(prim, arena, mesh)
                                                   : load_cylinder(prim, arena, mesh);
            if (ok) {
                mesh->path = scene_strdup(prim_path);
                mesh->visible = 1;
                mesh->prototype_idx = mesh_idx;
                mesh->is_proto_only = prim_under_point_instancer_cached(
                    prim, &point_instancer_prefixes) ||
                    path_is_native_usd_prototype(prim_path);
                for (int k = 0; k < 3; k++) {
                    if (mesh->bounds_min[k] < scene->bounds_min[k])
                        scene->bounds_min[k] = mesh->bounds_min[k];
                    if (mesh->bounds_max[k] > scene->bounds_max[k])
                        scene->bounds_max[k] = mesh->bounds_max[k];
                }
                mesh_idx++;
                total_verts   += mesh->nvertices;
                total_indices += mesh->nindices;
            }
            nanousd_freeprim(prim);
            continue;
        }

        if (strcmp(tn, "Mesh") != 0) {
            nanousd_freeprim(prim);
            continue;
        }

        if (!scene_ensure_mesh_capacity(scene, &max_meshes, mesh_idx + 1)) {
            nanousd_freeprim(prim);
            continue;
        }
        SceneMesh* mesh = &scene->meshes[mesh_idx];
        mesh->is_proto_only = prim_under_point_instancer_cached(
            prim, &point_instancer_prefixes) ||
            path_is_native_usd_prototype(prim_path);

        /* ---- (a) Points ---- */
        int pts_count = 0;
        const float* pts_data = nanousd_arraydataf(prim, "points", &pts_count);

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
        const float* normals_data = nanousd_arraydataf(prim, "normals", &normals_count);

        if ((!normals_data || normals_count / 3 != mesh->nvertices)) {
            normals_data = nanousd_arraydataf(prim, "primvars:normals", &normals_count);
        }

        if (normals_data && normals_count > 0 && normals_count / 3 == mesh->nvertices) {
            mesh->normals = (float*)arena_alloc(arena, (size_t)normals_count * sizeof(float), 16);
            if (mesh->normals)
                memcpy(mesh->normals, normals_data, (size_t)normals_count * sizeof(float));
        } else {
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
                /* No normals available — leave NULL, viewer will handle */
                mesh->normals = NULL;
            }
        }

        /* ---- (c) Face vertex indices ---- */
        int fvi_count = 0;
        const int* fvi_data = NULL;
        int* fvi_fallback = NULL; /* only allocated if zero-copy unavailable */
        {
            fvi_data = nanousd_arraydatai(prim, "faceVertexIndices", &fvi_count);
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
            fvc_data = nanousd_arraydatai(prim, "faceVertexCounts", &fvc_count);
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

        /* ---- (e) Triangulation ---- */
        if (fvc_data && fvc_count > 0) {
            int tri_count = count_triangulated_indices(fvc_data, fvc_count, fvi_count);
            mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)tri_count * sizeof(uint32_t), 16);
            if (mesh->indices) {
                /* triangulate_faces() may stop short if faceVertexCounts
                 * over-promises versus faceVertexIndices; trust the
                 * actual write count, not the upper-bound estimate. */
                mesh->nindices = triangulate_faces(fvi_data, fvi_count,
                                                   fvc_data, fvc_count,
                                                   mesh->indices);
            } else {
                mesh->nindices = 0;
            }
        } else {
            /* No faceVertexCounts: assume all triangles */
            mesh->nindices = fvi_count;
            mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)fvi_count * sizeof(uint32_t), 16);
            if (mesh->indices) {
                for (int j = 0; j < fvi_count; j++) {
                    mesh->indices[j] = (uint32_t)fvi_data[j];
                }
            }
        }

        if (!mesh->indices || mesh->nindices == 0) {
            nanousd_freeprim(prim);
            continue;
        }

        /* ---- (e2) Compute smooth normals if none were loaded ---- */
        if (!mesh->normals && mesh->positions && mesh->indices && mesh->nindices > 0) {
            mesh->normals = compute_smooth_normals(arena, mesh->positions, mesh->nvertices,
                                                   mesh->indices, mesh->nindices);
        }

        /* ---- (e3) UV + normal interpolation handling ---- */
        load_mesh_facevarying_primvars(arena, mesh, stage, prim,
                                       fvi_data, fvi_count,
                                       fvc_data, fvc_count);

        /* ---- (e4) Material index (set later by viewer) ---- */
        mesh->material_index = -1;

        /* ---- (f) displayColor ---- */
        mesh->has_display_color = 0;
        mesh->display_color[0] = 0.5f;
        mesh->display_color[1] = 0.5f;
        mesh->display_color[2] = 0.5f;
        mesh->colors = NULL;

        /* Check for per-vertex displayColor array first */
        int dc_count = 0;
        const float* dc_data = nanousd_arraydataf(prim, "primvars:displayColor", &dc_count);
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
        compute_world_xform(prim, mesh->world_xform);

        /* ---- Update bounds (scene + per-mesh) ---- */
        mesh_compute_local_bounds(mesh);
        mesh_compute_world_bounds_from_local(mesh, scene);

        /* ---- Offsets (zeroed, set by viewer later) ---- */
        mesh->vertex_offset = 0;
        mesh->index_offset = 0;

        /* ---- Instancing: register prototype mesh by path ---- */
        mesh->path = scene_strdup(prim_path);
        mesh->visible = 1;
        mesh->prototype_idx = mesh_idx;  /* self = unique or first instance */
        if (prim_path) {
            proto_hash_insert(&proto_mesh_table, prim_path, mesh_idx);
        }

        int emitted_meshes = split_mesh_by_geom_subsets(
            arena, scene, prim, mesh_idx, &max_meshes,
            fvc_data, fvc_count, fvi_count);
        for (int e = 0; e < emitted_meshes; e++) {
            total_verts += scene->meshes[mesh_idx + e].nvertices;
            total_indices += scene->meshes[mesh_idx + e].nindices;
        }
        mesh_idx += emitted_meshes;

        nanousd_freeprim(prim);
    }
    if (do_timing) {
        double t_now = get_time_sec();
        fprintf(stderr,
                "  [scene_timing] flat mesh/implicit pass: %.1f ms "
                "(meshes=%d, curves=%d, instance_prims=%d, invisible=%d, pointInstancer=%d)\n",
                (t_now - t_mark) * 1000.0,
                mesh_idx, scene->ncurves, n_instance_prims,
                invisible_prefixes.count, point_instancer_prefixes.count);
        t_mark = t_now;
    }

    /* ---- Second pass: expand instance prims into instanced mesh copies ---- */
    /* Instance mesh children are NOT in the flat prim list — they are only
     * accessible via nanousd_child(). For the first instance of each prototype,
     * we fully load geometry. Subsequent instances share that geometry. */
    if (n_instance_prims > 0) {
        fprintf(stderr, "scene_load: expanding %d instance prims...\n", n_instance_prims);

        /* Nested-PointInstancer records: PIs authored inside native-instance
         * prototypes (Moana ground cover) are reachable only via the composed
         * walk / asset seed here. The Path-B seeder + composed walk both append
         * to this list; it's drained after the loop into compact pi_batches. */
        struct NestedPiRec* nested = NULL;
        int nested_n = 0, nested_cap = 0;

        /* Path B: native asset-arc flat seeding + replay (gated; falls back to
         * the composed walk for instances Path B can't seed). */
        NbReplayCache nb_cache = {0, 0, 0};
        int nb_active = nb_flat_native_replay_active() &&
                        scene_compact_pi_batches_active();
        uint64_t nb_xform_cap = scene->npi_transforms;
        int nb_batch_cap = scene->npi_batches;
        long long nb_seeded_meshes = 0;
        int nb_seeded_roots = 0;

        nb_toplevel_pi_paths_reset();
        if (nb_active) {
            /* Top-level PI skip set: PIs the main-stage PI pass already owns
             * must not be re-recorded as nested PIs by the asset seeder. */
            for (int i = 0; i < nprims; i++) {
                NanousdPrim p = nanousd_prim(stage, i);
                if (!p) continue;
                if (prim_is_point_instancer(p)) {
                    const char* pp = nanousd_path(p);
                    if (pp) nb_toplevel_pi_paths_add(pp);
                }
                nanousd_freeprim(p);
            }
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
             * fall back (env off, no asset arc, or it couldn't seed). */
            if (nb_active) {
                const char* iprp = nanousd_path(inst_prim);
                char ipbuf[1024]; ipbuf[0] = '\0';
                if (iprp) snprintf(ipbuf, sizeof(ipbuf), "%s", iprp);
                /* If this instance root is itself under a top-level PointInstancer
                 * (the main PI pass owns its placement), seed prototype meshes as
                 * is_proto_only without emitting native batches. Otherwise emit a
                 * native-instance batch per seeded mesh. */
                int emit_native_batches =
                    !path_prefix_list_contains(&point_instancer_prefixes, ipbuf);
                int before = mesh_idx;
                int handled = nb_seed_native_instance(scene, stage, inst_prim, proto_root,
                                                      ipbuf, inst_root_world,
                                                      proto_root_inv, have_proto_root_inv,
                                                      &nb_cache, arena, &mesh_idx, max_meshes,
                                                      &total_verts, &total_indices,
                                                      &nested, &nested_n, &nested_cap,
                                                      &nb_xform_cap, &nb_batch_cap,
                                                      emit_native_batches);
                if (handled > 0) {
                    nb_seeded_meshes += (mesh_idx - before);
                    nb_seeded_roots++;
                    nanousd_freeprim(proto_root);
                    nanousd_freeprim(inst_prim);
                    continue;
                }
            }

            /* Traverse instance children recursively via a stack */
            NanousdPrim stack_arr[256];
            int stack_size = 0;

            int nc = nanousd_nchildren(inst_prim);
            for (int c = nc - 1; c >= 0 && stack_size < 256; c--) {
                NanousdPrim child = nanousd_child(inst_prim, c);
                if (child) stack_arr[stack_size++] = child;
            }

            while (stack_size > 0) {
                NanousdPrim child = stack_arr[--stack_size];
                if (!child) continue;

                /* Push grandchildren */
                int ncc = nanousd_nchildren(child);
                for (int c = ncc - 1; c >= 0 && stack_size < 256; c--) {
                    NanousdPrim gc = nanousd_child(child, c);
                    if (gc) stack_arr[stack_size++] = gc;
                }

                const char* ctn = nanousd_typename(child);
                if (!ctn) {
                    nanousd_freeprim(child);
                    continue;
                }
                /* BasisCurves under an instance prim — load each instance
                 * independently into scene->curves[]. We don't share
                 * geometry across curve instances yet (could mirror
                 * proto_hash_lookup pattern later); given typical curve
                 * counts the per-instance load is fine. */
                if (strcmp(ctn, "BasisCurves") == 0) {
                    if (!scene_ensure_curve_capacity(scene, arena, &max_curves,
                                                     scene->ncurves + 1)) {
                        nanousd_freeprim(child);
                        continue;
                    }
                    SceneCurve* curve = &scene->curves[scene->ncurves];
                    if (load_basiscurves(child, arena, curve)) {
                        for (int k = 0; k < 3; k++) {
                            if (curve->bounds_min[k] < scene->bounds_min[k])
                                scene->bounds_min[k] = curve->bounds_min[k];
                            if (curve->bounds_max[k] > scene->bounds_max[k])
                                scene->bounds_max[k] = curve->bounds_max[k];
                        }
                        scene->ncurves++;
                    }
                    nanousd_freeprim(child);
                    continue;
                }
                /* Nested PointInstancer under this instance prim. The composed
                 * walk (and the asset seeder) only materialize Mesh/BasisCurves;
                 * a PI here would otherwise drop its instanced ground cover.
                 * Record it (gated) so the post-loop drain emits compact
                 * batches. */
                if (nb_active && !strcmp(ctn, "PointInstancer")) {
                    int npi = nanousd_attribarraylen(child, "protoIndices");
                    const char* cp_raw = nanousd_path(child);
                    char cp[1024]; cp[0] = '\0';
                    if (cp_raw) snprintf(cp, sizeof(cp), "%s", cp_raw);
                    if (npi > 0 && cp[0]) {
                        /* pi_to_inst = pi_world_proto * inverse(proto_root_world)
                         *              * inst_root_world */
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
                            if (!strcmp(nested[dn].path, cp)) { already_nested = 1; break; }
                        }
                        if (!already_nested) {
                            if (nested_n >= nested_cap) {
                                int ncp = nested_cap ? nested_cap * 2 : 16;
                                struct NestedPiRec* g = (struct NestedPiRec*)realloc(
                                    nested, (size_t)ncp * sizeof(*nested));
                                if (g) { nested = g; nested_cap = ncp; }
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
                if (strcmp(ctn, "Mesh") != 0) {
                    nanousd_freeprim(child);
                    continue;
                }

                /* Compute prototype-relative key for geometry sharing */
                char child_path_buf[8192];
                const char* child_path = scene_prim_path_copy(child, child_path_buf, sizeof(child_path_buf));
                char inst_path_buf[8192];
                const char* inst_path = scene_prim_path_copy(inst_prim, inst_path_buf, sizeof(inst_path_buf));
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
                    if (!scene_ensure_mesh_capacity(scene, &max_meshes, mesh_idx + 1)) {
                        nanousd_freeprim(child);
                        goto done_instances;
                    }
                    /* Share geometry with already-loaded prototype mesh */
                    SceneMesh* proto_m = &scene->meshes[proto_idx];
                    SceneMesh* mesh = &scene->meshes[mesh_idx];
                    mesh->path = scene_strdup(instance_child_path);
                    mesh->visible = 1;
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
                    mesh->is_proto_only = 0;
                    mesh->material_index = proto_m->material_index;
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
                    if (!scene_ensure_mesh_capacity(scene, &max_meshes, mesh_idx + 1)) {
                        nanousd_freeprim(child);
                        goto done_instances;
                    }
                    /* First instance of this prototype mesh — load geometry fully */
                    SceneMesh* mesh = &scene->meshes[mesh_idx];

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
                    if (!normals_data || normals_count / 3 != mesh->nvertices)
                        normals_data = nanousd_arraydataf(child, "primvars:normals", &normals_count);
                    if (normals_data && normals_count > 0 && normals_count / 3 == mesh->nvertices) {
                        mesh->normals = (float*)arena_alloc(arena, (size_t)normals_count * sizeof(float), 16);
                        if (mesh->normals) memcpy(mesh->normals, normals_data, (size_t)normals_count * sizeof(float));
                    } else {
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

                    if (fvc_data && fvc_count > 0) {
                        int tri_count = count_triangulated_indices(fvc_data, fvc_count, fvi_count);
                        mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)tri_count * sizeof(uint32_t), 16);
                        if (mesh->indices) {
                            mesh->nindices = triangulate_faces(fvi_data, fvi_count,
                                                               fvc_data, fvc_count,
                                                               mesh->indices);
                        } else {
                            mesh->nindices = 0;
                        }
                    } else {
                        mesh->nindices = fvi_count;
                        mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)fvi_count * sizeof(uint32_t), 16);
                        if (mesh->indices) {
                            for (int j = 0; j < fvi_count; j++) mesh->indices[j] = (uint32_t)fvi_data[j];
                        }
                    }
                    if (!mesh->indices || mesh->nindices == 0) { nanousd_freeprim(child); continue; }

                    /* Smooth normals if needed */
                    if (!mesh->normals && mesh->positions && mesh->indices)
                        mesh->normals = compute_smooth_normals(arena, mesh->positions, mesh->nvertices,
                                                               mesh->indices, mesh->nindices);

                    /* UV + normal interpolation — shared helper, so
                     * USD-instanced meshes get the same dedup expansion
                     * the prototype path uses (the bulk of a SimReady
                     * warehouse arrives through this branch). */
                    load_mesh_facevarying_primvars(arena, mesh, stage, child,
                                                   fvi_data, fvi_count,
                                                   fvc_data, fvc_count);

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
                    mesh->path = scene_strdup(instance_child_path);
                    mesh->visible = 1;

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

        /* ---- Drain nested PIs into compact batches: prototypes are now in
         * proto_mesh_table (composed walk) or scene->meshes (Path-B seeds);
         * resolve, compose leaf_world = inst_local * pi_to_inst, emit compact
         * batches. Proto meshes are marked is_proto_only so they aren't drawn
         * standalone (they're instanced via the PI). ---- */
        uint64_t nested_xform_cap = nb_xform_cap;
        int      nested_batch_cap = nb_batch_cap;
        long long nested_emitted = 0;
        long long n_nested_pi = nested_n;
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
                        if (sp && strcmp(sp, tp) == 0) { exact = sm; break; }
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
                    /* Path-B-seeded prototypes aren't in proto_mesh_table; scan
                     * scene->meshes by path prefix. */
                    for (int sm = 0; sm < mesh_idx; sm++) {
                        const char* sp = scene->meshes[sm].path;
                        if (!sp) continue;
                        if (strncmp(sp, tp, tl) == 0 && (sp[tl] == '/' || sp[tl] == '\0'))
                            count++;
                    }
                    if (count > 0) {
                        plist[p] = (int*)malloc((size_t)count * sizeof(int));
                        int wi = 0;
                        for (int sm = 0; sm < mesh_idx && wi < count; sm++) {
                            const char* sp = scene->meshes[sm].path;
                            if (!sp) continue;
                            if (strncmp(sp, tp, tl) == 0 && (sp[tl] == '/' || sp[tl] == '\0'))
                                plist[p][wi++] = sm;
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
                    fprintf(stderr, "scene_load:   nested-PI proto UNRESOLVED '%s'\n", tp);
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
                    double ww=qw*qw,iiq=qi*qi,jj=qj*qj,kk=qk*qk,wi=qw*qi,wj=qw*qj,wk=qw*qk,ij=qi*qj,ik=qi*qk,jk=qj*qk;
                    ix[0]=(ww+iiq-jj-kk)*sx; ix[1]=2*(ij+wk)*sx; ix[2]=2*(ik-wj)*sx; ix[3]=0;
                    ix[4]=2*(ij-wk)*sy; ix[5]=(ww-iiq+jj-kk)*sy; ix[6]=2*(jk+wi)*sy; ix[7]=0;
                    ix[8]=2*(ik+wj)*sz; ix[9]=2*(jk-wi)*sz; ix[10]=(ww-iiq-jj+kk)*sz; ix[11]=0;
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
            for (int p = 0; p < n_protos; p++) free(plist[p]);
            free(plist); free(pcount);
            pi_proto_targets_free(&proto_targets);
            free(inst_visible);
            free(pidxs); free(pos); free(ori); free(scl);
            nanousd_freeprim(pi);
        }
        free(nested);
        if (nb_active && (n_nested_pi > 0 || scene->npi_batches > 0))
            fprintf(stderr,
                    "scene_load: compact PI — %d batches, %llu transforms "
                    "(nested PIs: %lld, emitted %lld transforms)\n",
                    scene->npi_batches, (unsigned long long)scene->npi_transforms,
                    n_nested_pi, nested_emitted);
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

    /* ---- Third pass: expand PointInstancer prims ---- */
    for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;
        /* Detect PointInstancers by typename (matches prim_is_point_instancer
         * used everywhere else here and in the vulkan renderer's third pass)
         * OR by the presence of the PI-specific `protoIndices` attribute.
         * nanousd_isa("PointInstancer") returns 0 for these prims under
         * NUSD_FLAT_NATIVE_INSTANCE_TRAVERSAL (schema inheritance isn't wired
         * up in flat mode), which silently dropped the OpenChessSet pawns —
         * a top-level PI with protoIndices=8 — out of the scene. */
        if (!prim_is_point_instancer(prim) &&
            nanousd_attribarraylen(prim, "protoIndices") <= 0) {
            nanousd_freeprim(prim); continue;
        }

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
        else { /* nanousd returns quaternions as (x, y, z, w). */
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
         * one OR MORE Mesh descendants (e.g. the chess set's Pawn proto
         * has a Geom_Body and a Geom_Top). Collect ALL matching mesh
         * indices per prototype so the instance loop fans out one
         * scene-mesh per (proto-mesh × instance) pair. Mirrors
         * nanousd-vulkan-renderer/src/scene.c — without this fan-out
         * the chess pawns rendered as just their base mesh, missing
         * the body/crown sub-mesh on opengl. */
        int n_protos = nanousd_nreltargets(prim, "prototypes");
        int** proto_mesh_index_lists = (int**)calloc((size_t)n_protos, sizeof(int*));
        int*  proto_mesh_index_counts = (int*)calloc((size_t)n_protos, sizeof(int));
        char _pi_full_path_buf[8192];
        const char* _pi_full_path = scene_prim_path_copy(prim, _pi_full_path_buf, sizeof(_pi_full_path_buf));
        for (int p = 0; p < n_protos; p++) {
            const char* proto_path_raw = nanousd_reltarget(prim, "prototypes", p);
            if (!proto_path_raw) continue;
            /* nanousd's composition doesn't rewrite Sdf path-rels through
             * reference layers. The OpenChessSet wrapper has the
             * PointInstancer at /World/ChessSet/Black/Pawns but its
             * `prototypes` rel target stays as the authored
             * /ChessSet/Black/Pawns/Pawn — proto_mesh_table search
             * misses and the pawns drop out (TLAS goes from full
             * 33 prims to 17). Splice the missing reference prefix on.
             * Same fix landed in nanousd-vulkan-renderer scene.c
             * (commit 599f634). */
            char proto_path_buf[1024];
            const char* proto_path = proto_path_raw;
            if (proto_path_raw[0] == '/' && _pi_full_path) {
                const char* slash2 = strchr(proto_path_raw + 1, '/');
                size_t first_seg = slash2 ? (size_t)(slash2 - proto_path_raw)
                                          : strlen(proto_path_raw);
                if (first_seg > 1) {
                    const char* match = strstr(_pi_full_path, proto_path_raw);
                    if (!match) {
                        char first_seg_buf[256];
                        if (first_seg < sizeof(first_seg_buf)) {
                            memcpy(first_seg_buf, proto_path_raw, first_seg);
                            first_seg_buf[first_seg] = '\0';
                            const char* m = strstr(_pi_full_path, first_seg_buf);
                            if (m) {
                                char next = m[first_seg];
                                if (next == '\0' || next == '/') {
                                    size_t prefix_len = (size_t)(m - _pi_full_path);
                                    int n = snprintf(proto_path_buf,
                                                     sizeof(proto_path_buf),
                                                     "%.*s%s",
                                                     (int)prefix_len, _pi_full_path,
                                                     proto_path_raw);
                                    if (n > 0 && (size_t)n < sizeof(proto_path_buf)) {
                                        proto_path = proto_path_buf;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            /* Exact match first (proto IS itself a Mesh). */
            int direct = proto_hash_lookup(&proto_mesh_table, proto_path);
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
            if (count == 0) continue;
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

        /* Instancer world transform */
        double instancer_world[16];
        compute_world_xform(prim, instancer_world);

        int pi_count = 0;
        int proto_meshes_total = 0;  /* sum of per-prototype mesh counts touched */
        for (int inst = 0; inst < n_instances; inst++) {
            if (inst_visible && !inst_visible[inst]) continue;
            int pidx = proto_indices[inst];
            if (pidx < 0 || pidx >= n_protos ||
                proto_mesh_index_counts[pidx] == 0) continue;

            /* Build the instance transform once per instance — same
             * across the (typically small) per-proto mesh count.
             *
             * nanousd_attribarrayf returns USD quat values in (x, y, z, w)
             * order — imaginary components first, real last. Reading as
             * (w, x, y, z) silently turns every PointInstancer's
             * identity quat (1, 0, 0, 0) into a 180° Z-axis rotation.
             * Match the vulkan/metal renderers' fix. */
            float qi_q = orientations[inst*4+0];
            float qj_q = orientations[inst*4+1];
            float qk_q = orientations[inst*4+2];
            float w_q  = orientations[inst*4+3];
            float sx = scales[inst*3+0];
            float sy = scales[inst*3+1];
            float sz = scales[inst*3+2];
            float tx = positions[inst*3+0];
            float ty = positions[inst*3+1];
            float tz = positions[inst*3+2];

            double inst_xform[16];
            double ww=w_q*w_q, ii=qi_q*qi_q, jj=qj_q*qj_q, kk=qk_q*qk_q;
            double wi=w_q*qi_q, wj=w_q*qj_q, wk=w_q*qk_q;
            double ij=qi_q*qj_q, ik=qi_q*qk_q, jk=qj_q*qk_q;
            inst_xform[0]  = (ww+ii-jj-kk)*sx; inst_xform[1]  = 2*(ij+wk)*sx;     inst_xform[2]  = 2*(ik-wj)*sx;     inst_xform[3]  = 0;
            inst_xform[4]  = 2*(ij-wk)*sy;      inst_xform[5]  = (ww-ii+jj-kk)*sy; inst_xform[6]  = 2*(jk+wi)*sy;     inst_xform[7]  = 0;
            inst_xform[8]  = 2*(ik+wj)*sz;      inst_xform[9]  = 2*(jk-wi)*sz;     inst_xform[10] = (ww-ii-jj+kk)*sz; inst_xform[11] = 0;
            inst_xform[12] = tx;                inst_xform[13] = ty;               inst_xform[14] = tz;               inst_xform[15] = 1;

            double inst_world[16];
            nanousd_mul_m4d(inst_xform, instancer_world, inst_world);

            /* Fan out: emit one scene-mesh per (instance × proto-mesh). */
            for (int pmi = 0; pmi < proto_mesh_index_counts[pidx]; pmi++) {
                if (!scene_ensure_mesh_capacity(scene, &max_meshes, mesh_idx + 1))
                    goto pi_done;
                int src_mesh_idx = proto_mesh_index_lists[pidx][pmi];
                SceneMesh* proto_m = &scene->meshes[src_mesh_idx];
                SceneMesh* mesh = &scene->meshes[mesh_idx];
                {
                    char pi_path_buf[8192];
                    const char* pi_path = scene_prim_path_copy(prim, pi_path_buf, sizeof(pi_path_buf));
                    char generated_path[8192];
                    snprintf(generated_path, sizeof(generated_path), "%s/instance_%d/mesh_%d",
                             pi_path ? pi_path : "/PointInstancer", inst, pmi);
                    mesh->path = scene_strdup(generated_path);
                }
                mesh->visible = 1;

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
                mesh->is_proto_only = 0;
                memcpy(mesh->local_bounds_min, proto_m->local_bounds_min, sizeof(float) * 3);
                memcpy(mesh->local_bounds_max, proto_m->local_bounds_max, sizeof(float) * 3);
                /* Inherit the prototype's material binding — without this
                 * the instanced meshes get whatever calloc landed on
                 * material_index (slot 0 = chessboard marble in the chess
                 * scene), which paints a green-marble band around every
                 * pawn's base. */
                mesh->material_index = proto_m->material_index;

                memcpy(mesh->world_xform, inst_world, sizeof(double) * 16);
                mesh_compute_world_bounds_from_local(mesh, scene);
                mesh->vertex_offset = 0;
                mesh->index_offset = 0;
                total_verts += mesh->nvertices;
                total_indices += mesh->nindices;
                mesh_idx++;
                instanced_meshes++;
                proto_meshes_total++;
            }
            pi_count++;
        }
pi_done:

        if (pi_count > 0)
            fprintf(stderr, "scene_load: PointInstancer '%s': %d instances (%d protos, %d proto-meshes total)\n",
                    nanousd_path(prim), pi_count, n_protos, proto_meshes_total);

        for (int p = 0; p < n_protos; p++) free(proto_mesh_index_lists[p]);
        free(proto_mesh_index_lists);
        free(proto_mesh_index_counts);
        free(inst_visible);
        free(proto_indices);
        free(positions);
        free(orientations);
        free(scales);
        nanousd_freeprim(prim);
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

    proto_hash_free(&proto_mesh_table);

    /* Set actual mesh count (over-allocated above using nprims as upper bound) */
    scene->nmeshes = mesh_idx;

    if (scene->nmeshes == 0 && scene->ncurves == 0) {
        char fallback_layer[4096];
        char fallback_source[1024];
        if (!scene_env_enabled("NUSD_OPENGL_DISABLE_REF_FALLBACK") &&
            s_ref_fallback_depth < 4 &&
            scene_find_direct_asset_arc(stage, filepath,
                                        fallback_layer, sizeof(fallback_layer),
                                        fallback_source, sizeof(fallback_source))) {
            fprintf(stderr,
                    "scene_load: no Mesh or BasisCurves prims in '%s'; "
                    "following direct reference to '%s'%s%s\n",
                    filepath, fallback_layer,
                    fallback_source[0] ? " source " : "",
                    fallback_source[0] ? fallback_source : "");
            g_xform_cache = NULL;
            xform_cache_free(&xform_cache);
            path_prefix_list_free(&invisible_prefixes);
            path_prefix_list_free(&point_instancer_prefixes);
            scene_free(scene);
            s_ref_fallback_depth++;
            Scene* fallback_scene = scene_load(fallback_layer);
            s_ref_fallback_depth--;
            return fallback_scene;
        }

        fprintf(stderr, "scene_load: no Mesh or BasisCurves prims found in '%s' (%d prims total)\n",
                filepath, nprims);
        g_xform_cache = NULL;
        xform_cache_free(&xform_cache);
        path_prefix_list_free(&invisible_prefixes);
        path_prefix_list_free(&point_instancer_prefixes);
        scene_free(scene);
        return NULL;
    }

    /* ---- Extract scene lights (UsdLuxRectLight, DiskLight, CylinderLight,
     *      DistantLight, SphereLight) ----
     *
     * Isaac/Omniverse assets usually author UsdLux fields as inputs:*,
     * while small debug fixtures and older USDs often use legacy bare names
     * such as `float intensity`. Support both so the same wrapper scene lights
     * OVRTX, Vulkan, and OpenGL. */
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

        if (nlights > 0)
            scene->lights = (SceneLight*)arena_calloc(arena, (size_t)nlights,
                                                      sizeof(SceneLight));
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
                if (!is_rect && !is_disk && !is_cylinder && !is_dist && !is_sphere) {
                    nanousd_freeprim(p);
                    continue;
                }

                SceneLight* L = &scene->lights[li];
                int ok = 0;

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
                         * length x diameter silhouette in the existing
                         * RectLight evaluator. */
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
                    float angle = nanousd_attribf(p, "inputs:angle", &ok);
                    if (!ok) angle = nanousd_attribf(p, "angle", &ok);
                    L->angle_deg = ok ? angle : 0.53f;
                }

                {
                    float nx = L->normal[0], ny = L->normal[1], nz = L->normal[2];
                    float len = sqrtf(nx * nx + ny * ny + nz * nz);
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
            fprintf(stderr,
                    "scene_load: %d lights (rect/disk/cylinder+distant+sphere) extracted\n",
                    li);
        }
    }

    /* ---- Extract UsdLuxDomeLight ----
     *
     * Keep this metadata beside direct lights so viewer_create can install
     * the scene-authored HDR as IBL, matching the Vulkan/OVRTX path. First
     * active dome wins even when it is a textureless flat DomeLight. */
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
                nanousd_freeprim(p);
                continue;
            }

            scene->has_authored_light = 1;
            scene->has_dome_light = 1;
            int ok = 0;
            if (!nanousd_attribv3f(p, "inputs:color", scene->dome_color))
                nanousd_attribv3f(p, "color", scene->dome_color);
            const char* asset = nanousd_attribasset(p, "inputs:texture:file", &ok);
            if (ok && asset && asset[0]) {
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
            break;
        }
    }

    /* ---- upAxis handling ----
     *
     * Detect the authored up axis (`upAxis` stage metadata, or shape
     * heuristic when missing), but leave geometry in authored coordinates by
     * default. OVRTX does not bake up-axis conversion into mesh transforms, so
     * camera parity requires the same scene frame here. The old Z-up→Y-up bake
     * remains available through NUSD_BAKE_ZUP_TO_YUP=1. */
    {
        int ok = 0;
        const char* up = nanousd_metadatas(stage, "upAxis", &ok);
        int detected = 1;
        if (ok && up && (up[0] == 'Z' || up[0] == 'z')) {
            detected = 2;
        } else if (ok && up && (up[0] == 'X' || up[0] == 'x')) {
            detected = 0;
        } else if (!ok) {
            float dx = scene->bounds_max[0] - scene->bounds_min[0];
            float dy = scene->bounds_max[1] - scene->bounds_min[1];
            float dz = scene->bounds_max[2] - scene->bounds_min[2];
            float max_xy = dx > dy ? dx : dy;
            if (dz > 1e-3f && dz < max_xy * 0.8f && scene->bounds_min[2] > -dz * 0.5f) {
                fprintf(stderr, "scene_load: heuristic Z-up detection (dz=%.1f < max_xy=%.1f, z_min=%.1f)\n",
                        dz, max_xy, scene->bounds_min[2]);
                detected = 2;
            }
        }

        const char* bake_env = getenv("NUSD_BAKE_ZUP_TO_YUP");
        int bake_zup_to_yup = bake_env && bake_env[0] && bake_env[0] != '0';

        /* Backward-compatible spelling from the old default-bake path. */
        const char* preserve = getenv("NUSD_PRESERVE_UPAXIS");
        if (preserve && *preserve && *preserve != '0')
            bake_zup_to_yup = 0;

        scene->up_axis = detected;
        if (detected == 2 && bake_zup_to_yup) {
            /* Z-up → Y-up: rotate -90° around X.
             * Row-vector convention: p' = p * R, R maps (x,y,z) → (x,z,-y). */
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

                float omin[3], omax[3];
                memcpy(omin, mesh->bounds_min, sizeof(float) * 3);
                memcpy(omax, mesh->bounds_max, sizeof(float) * 3);
                mesh->bounds_min[0] = omin[0];
                mesh->bounds_min[1] = omin[2];
                mesh->bounds_min[2] = -omax[1];
                mesh->bounds_max[0] = omax[0];
                mesh->bounds_max[1] = omax[2];
                mesh->bounds_max[2] = -omin[1];
            }
            /* Curves: same world_xform * R update; bounds rotate the same
             * (x,y,z) → (x,z,-y) component swap. */
            for (int c = 0; c < scene->ncurves; c++) {
                SceneCurve* cu = &scene->curves[c];
                double tmp[16];
                nanousd_mul_m4d(cu->world_xform, R, tmp);
                memcpy(cu->world_xform, tmp, sizeof(double) * 16);

                float omin[3], omax[3];
                memcpy(omin, cu->bounds_min, sizeof(float) * 3);
                memcpy(omax, cu->bounds_max, sizeof(float) * 3);
                cu->bounds_min[0] = omin[0];
                cu->bounds_min[1] = omin[2];
                cu->bounds_min[2] = -omax[1];
                cu->bounds_max[0] = omax[0];
                cu->bounds_max[1] = omax[2];
                cu->bounds_max[2] = -omin[1];
            }
            for (int l = 0; l < scene->nlights; l++) {
                SceneLight* L = &scene->lights[l];
                float p[3] = { L->position[0], L->position[1], L->position[2] };
                float n[3] = { L->normal[0],   L->normal[1],   L->normal[2] };
                float u[3] = { L->u_axis[0],   L->u_axis[1],   L->u_axis[2] };
                float v[3] = { L->v_axis[0],   L->v_axis[1],   L->v_axis[2] };
                L->position[0] = p[0]; L->position[1] = p[2]; L->position[2] = -p[1];
                L->normal[0]   = n[0]; L->normal[1]   = n[2]; L->normal[2]   = -n[1];
                L->u_axis[0]   = u[0]; L->u_axis[1]   = u[2]; L->u_axis[2]   = -u[1];
                L->v_axis[0]   = v[0]; L->v_axis[1]   = v[2]; L->v_axis[2]   = -v[1];
            }
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
            for (int c = 0; c < scene->ncurves; c++) {
                SceneCurve* cu = &scene->curves[c];
                for (int k = 0; k < 3; k++) {
                    if (cu->bounds_min[k] < scene->bounds_min[k])
                        scene->bounds_min[k] = cu->bounds_min[k];
                    if (cu->bounds_max[k] > scene->bounds_max[k])
                        scene->bounds_max[k] = cu->bounds_max[k];
                }
            }
            fprintf(stderr, "scene_load: applied Z-up → Y-up rotation\n");
            scene->up_axis = 1;
        }
    }

    /* ---- metersPerUnit handling ----
     * USD lets each stage declare a coordinate scale: metersPerUnit=0.01
     * means coordinates are authored in centimetres, =1.0 means metres,
     * =0.0254 means inches. OVRTX/USD render in authored numeric coordinates;
     * the metadata does not rescale camera/world coordinates. Preserve that by
     * default. NUSD_APPLY_METERS_PER_UNIT=1 enables the old normalization path
     * for tools that intentionally want metric-space bounds.
     */
    {
        int ok = 0;
        double mpu = nanousd_metadatad(stage, "metersPerUnit", &ok);
        const char* units_env = getenv("NUSD_APPLY_METERS_PER_UNIT");
        int apply_units = units_env && units_env[0] && units_env[0] != '0';
        if (apply_units && ok && mpu > 0.0 && mpu != 1.0) {
            for (int m = 0; m < scene->nmeshes; m++) {
                SceneMesh* mesh = &scene->meshes[m];
                /* Scale the world_xform by mpu — multiply the translation
                 * column and the upper-left 3x3. world_xform is row-major
                 * 4x4: rows [0..2] are basis * scale, row 3 is translation. */
                for (int i = 0; i < 12; i++) mesh->world_xform[i] *= mpu;
                mesh->world_xform[12] *= mpu;
                mesh->world_xform[13] *= mpu;
                mesh->world_xform[14] *= mpu;
                /* Scale per-mesh bounds */
                for (int k = 0; k < 3; k++) {
                    mesh->bounds_min[k] = (float)(mesh->bounds_min[k] * mpu);
                    mesh->bounds_max[k] = (float)(mesh->bounds_max[k] * mpu);
                }
            }
            for (int k = 0; k < 3; k++) {
                scene->bounds_min[k] = (float)(scene->bounds_min[k] * mpu);
                scene->bounds_max[k] = (float)(scene->bounds_max[k] * mpu);
            }
            /* Curves: CVs are object-space (model matrix carries the
             * transform), so scale world_xform like meshes do. Widths
             * are object-space too — scale them; the shader applies
             * model * ringDir which scales by the same mpu. */
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
            for (int l = 0; l < scene->nlights; l++) {
                SceneLight* L = &scene->lights[l];
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
                        fprintf(stderr, "scene_load: rejected %d outlier meshes (bounds shrunk %.0fx)\n",
                                scene->nmeshes - included, old_diag / new_diag);
                    }
                }
                free(dists);
            }
            free(cx);
        }
    }

lazy_postprocess:;
    double t1 = get_time_sec();
    if (do_timing) {
        fprintf(stderr, "  [scene_timing] scene postprocess: %.1f ms\n",
                (t1 - t_mark) * 1000.0);
    }
    fprintf(stderr, "scene_load: '%s' loaded in %.1f ms\n"
                    "  %d meshes, %" PRIu64 " vertices, %" PRIu64 " indices\n"
                    "  bounds: (%.3f, %.3f, %.3f) - (%.3f, %.3f, %.3f)\n",
            filepath, (t1 - t0) * 1000.0,
            scene->nmeshes, total_verts, total_indices,
            scene->bounds_min[0], scene->bounds_min[1], scene->bounds_min[2],
            scene->bounds_max[0], scene->bounds_max[1], scene->bounds_max[2]);
    if (instanced_meshes > 0) {
        int unique_protos = scene->nmeshes - instanced_meshes;
        fprintf(stderr, "  instancing: %d instances + %d unique = %d total meshes (%.0f%% shared)\n",
                instanced_meshes, unique_protos, scene->nmeshes,
                (double)instanced_meshes / (double)scene->nmeshes * 100.0);
    }

    g_xform_cache = NULL;
    xform_cache_free(&xform_cache);
    path_prefix_list_free(&invisible_prefixes);
    path_prefix_list_free(&point_instancer_prefixes);

    /* Geometry cache write: serialize the fully-built Scene to
     * <usd>.nzgeo.gl so the next load skips the USD parse. Best-effort. */
    if (geo_cache_active && scene->nmeshes > 0) {
        if (geo_cache_write(filepath, scene) == 0)
            fprintf(stderr, "[geo_cache] wrote cache: %s (%d meshes)\n",
                    filepath, scene->nmeshes);
    }
    return scene;
}

int scene_release_mesh_payloads(Scene* scene)
{
    if (!scene) return 0;
    if (scene->ncurves > 0) return 0;

    if (scene->_meshes_heap) {
        for (int i = 0; i < scene->nmeshes; i++) {
            scene->meshes[i].positions = NULL;
            scene->meshes[i].normals = NULL;
            scene->meshes[i].colors = NULL;
            scene->meshes[i].texcoords = NULL;
            scene->meshes[i].indices = NULL;
        }

        if (scene->_stage) {
            nanousd_close((NanousdStage)scene->_stage);
            scene->_stage = NULL;
        }

        if (scene->_arena) {
            arena_destroy((Arena*)scene->_arena);
            free(scene->_arena);
            scene->_arena = NULL;
        }

        return 1;
    }

    SceneMesh* meshes = NULL;
    if (scene->nmeshes > 0) {
        meshes = (SceneMesh*)malloc((size_t)scene->nmeshes * sizeof(SceneMesh));
        if (!meshes) return 0;
        memcpy(meshes, scene->meshes, (size_t)scene->nmeshes * sizeof(SceneMesh));
        for (int i = 0; i < scene->nmeshes; i++) {
            meshes[i].positions = NULL;
            meshes[i].normals = NULL;
            meshes[i].colors = NULL;
            meshes[i].texcoords = NULL;
            meshes[i].indices = NULL;
        }
    }

    if (scene->_stage) {
        nanousd_close((NanousdStage)scene->_stage);
        scene->_stage = NULL;
    }

    if (scene->_arena) {
        arena_destroy((Arena*)scene->_arena);
        free(scene->_arena);
        scene->_arena = NULL;
    }

    scene->meshes = meshes;
    scene->_meshes_heap = 1;
    return 1;
}

/* ----------------------------------------------------------------
 * scene_free
 * ---------------------------------------------------------------- */
void scene_free(Scene* scene)
{
    if (!scene) return;

    if (scene->meshes) {
        for (int i = 0; i < scene->nmeshes; i++) {
            free(scene->meshes[i].path);
            scene->meshes[i].path = NULL;
            free(scene->meshes[i].ptex_tri_colors);
            scene->meshes[i].ptex_tri_colors = NULL;
            scene->meshes[i].ptex_tri_color_count = 0;
        }
    }

    /* Compact instance batches/transforms (heap, not arena). */
    free(scene->pi_batches);    scene->pi_batches = NULL;    scene->npi_batches = 0;
    free(scene->pi_transforms); scene->pi_transforms = NULL; scene->npi_transforms = 0;

    /* Close the USD stage (invalidates zero-copy pointers) */
    if (scene->_stage) {
        nanousd_close((NanousdStage)scene->_stage);
        scene->_stage = NULL;
    }

    /* Destroy the arena (frees all mesh data allocated from it) */
    if (scene->_arena) {
        arena_destroy((Arena*)scene->_arena);
        free(scene->_arena);
        scene->_arena = NULL;
    }

    if (scene->_meshes_heap) {
        free(scene->meshes);
        scene->meshes = NULL;
        scene->_meshes_heap = 0;
    }

    free(scene);
}
