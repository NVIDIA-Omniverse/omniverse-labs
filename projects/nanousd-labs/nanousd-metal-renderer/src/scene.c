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
#include "primitive_geom.h"
#include <nanousd/nanousdapi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <libgen.h>
#include <sys/resource.h>

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
 * the loader, propagating the renderer's nu_set_current_time() value.
 * Cf. Vulkan port commit 18d341d. */
static double s_load_time = NAN;
void scene_set_load_time(double t) { s_load_time = t; }

static int s_load_materials = 1;
void scene_set_load_materials(int enabled) { s_load_materials = enabled ? 1 : 0; }

static char s_scene_last_error[512];

static void scene_clear_last_error(void)
{
    s_scene_last_error[0] = '\0';
}

const char* scene_last_error(void)
{
    return s_scene_last_error[0] ? s_scene_last_error : NULL;
}

static void scene_set_last_errorf(const char* fmt, ...)
{
    if (!fmt) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s_scene_last_error, sizeof(s_scene_last_error), fmt, ap);
    va_end(ap);
}

static void scene_fail(Scene* scene, const char* fmt, ...)
{
    if (!fmt) return;
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    snprintf(s_scene_last_error, sizeof(s_scene_last_error), "%s", msg);
    if (scene) {
        scene->load_failed = 1;
        snprintf(scene->load_error, sizeof(scene->load_error), "%s", msg);
    }
    fprintf(stderr, "scene_load: ERROR: %s\n", msg);
}

static char** s_visible_instance_child_paths = NULL;
static int s_visible_instance_child_path_count = 0;

void scene_set_visible_instance_child_paths(const char* const* paths, int count)
{
    for (int i = 0; i < s_visible_instance_child_path_count; i++)
        free(s_visible_instance_child_paths[i]);
    free(s_visible_instance_child_paths);
    s_visible_instance_child_paths = NULL;
    s_visible_instance_child_path_count = 0;

    if (!paths || count <= 0) return;
    s_visible_instance_child_paths =
        (char**)calloc((size_t)count, sizeof(char*));
    if (!s_visible_instance_child_paths) return;

    for (int i = 0; i < count; i++) {
        const char* path = paths[i];
        if (!path || path[0] != '/') continue;
        size_t n = strlen(path);
        char* copy = (char*)malloc(n + 1);
        if (!copy) continue;
        memcpy(copy, path, n + 1);
        s_visible_instance_child_paths[s_visible_instance_child_path_count++] = copy;
    }
}

/* ----------------------------------------------------------------
 * Compute world transform by walking the parent chain.
 * Multiplies local transforms from the prim up to the root.
 * Uses the load-time time code (s_load_time) so animated xformOps
 * resolve at the right frame.
 * ---------------------------------------------------------------- */
static void compute_world_xform(NanousdPrim prim, double world[16])
{
    /* Collect the local transform stack on the C stack.
     * USD hierarchies rarely exceed 32 levels deep. */
    double xforms[64][16];
    NanousdPrim chain[64];
    /* Optional prim-path strings parallel to chain[], for debug logging. */
    const char* path_at[64];
    int depth = 0;

    int dbg = 0;
    {
        const char* env = getenv("NUSD_XFORM_DUMP");
        const char* leaf = nanousd_path(prim);
        if (env && *env && leaf) {
            /* Filter to a substring so we don't drown in 1000+ prim chains. */
            if (strstr(leaf, env) != NULL) dbg = 1;
        }
    }

    /* Get local transform for the mesh prim itself */
    int reset_stack = 0;
    {
        double local[16];
        int ok = nanousd_get_local_transform(prim, s_load_time, local, &reset_stack);
        if (ok) {
            memcpy(xforms[depth], local, sizeof(double) * 16);
        } else {
            memcpy(xforms[depth], kIdentity, sizeof(double) * 16);
        }
        chain[depth] = NULL; /* prim itself is not our allocation */
        path_at[depth] = nanousd_path(prim);
        if (dbg) {
            fprintf(stderr, "[xform] %s ok=%d local=[%.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g]\n",
                    path_at[depth] ? path_at[depth] : "?", ok,
                    xforms[depth][0], xforms[depth][1], xforms[depth][2], xforms[depth][3],
                    xforms[depth][4], xforms[depth][5], xforms[depth][6], xforms[depth][7],
                    xforms[depth][8], xforms[depth][9], xforms[depth][10], xforms[depth][11],
                    xforms[depth][12], xforms[depth][13], xforms[depth][14], xforms[depth][15]);
        }
        depth++;
    }

    /* Walk up the parent chain */
    NanousdPrim cur = reset_stack ? NULL : nanousd_parent(prim);
    while (cur && depth < 64) {
        const char* path = nanousd_path(cur);
        /* Stop at the pseudo-root */
        if (path && path[0] == '/' && path[1] == '\0') {
            nanousd_freeprim(cur);
            break;
        }

        double local[16];
        int ok = nanousd_get_local_transform(cur, s_load_time, local, &reset_stack);
        if (ok) {
            memcpy(xforms[depth], local, sizeof(double) * 16);
        } else {
            memcpy(xforms[depth], kIdentity, sizeof(double) * 16);
        }
        chain[depth] = cur;
        path_at[depth] = path;
        if (dbg) {
            fprintf(stderr, "[xform] %s ok=%d local=[%.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g]\n",
                    path ? path : "?", ok,
                    xforms[depth][0], xforms[depth][1], xforms[depth][2], xforms[depth][3],
                    xforms[depth][4], xforms[depth][5], xforms[depth][6], xforms[depth][7],
                    xforms[depth][8], xforms[depth][9], xforms[depth][10], xforms[depth][11],
                    xforms[depth][12], xforms[depth][13], xforms[depth][14], xforms[depth][15]);
        }
        depth++;

        if (reset_stack) {
            break;
        }

        NanousdPrim parent = nanousd_parent(cur);
        cur = parent;
    }

    /* Multiply from leaf (first) up to root (last).
     * world = xforms[0] * xforms[1] * ... * xforms[depth-1]
     * Row-vector convention: P_world = P_local * leaf * parent * ... * root */
    memcpy(world, xforms[0], sizeof(double) * 16);
    for (int i = 1; i < depth; i++) {
        double tmp[16];
        nanousd_mul_m4d(world, xforms[i], tmp);
        memcpy(world, tmp, sizeof(double) * 16);
    }
    if (dbg) {
        fprintf(stderr, "[xform] WORLD = [%.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g | %.6g %.6g %.6g %.6g]\n",
                world[0], world[1], world[2], world[3],
                world[4], world[5], world[6], world[7],
                world[8], world[9], world[10], world[11],
                world[12], world[13], world[14], world[15]);
    }

    /* Free parent prim handles (skip index 0 which was not allocated) */
    for (int i = 1; i < depth; i++) {
        if (chain[i]) {
            nanousd_freeprim(chain[i]);
        }
    }
}

static int prim_path_depth(const char* path)
{
    if (!path || path[0] != '/') return -1;
    if (path[1] == '\0') return 0;
    int depth = 0;
    for (const char* p = path; *p; p++) {
        if (*p == '/' && p[1] != '\0') depth++;
    }
    return depth;
}

static void compute_world_xform_cached(NanousdPrim prim,
                                       double world[16],
                                       double (*stack)[16],
                                       unsigned char* stack_valid,
                                       int stack_cap)
{
    int reset_stack = 0;
    double local[16];
    int ok = nanousd_get_local_transform(prim, s_load_time, local, &reset_stack);
    if (!ok) memcpy(local, kIdentity, sizeof(local));

    int depth = prim_path_depth(nanousd_path(prim));
    if (reset_stack || depth <= 0 || depth >= stack_cap ||
        !stack || !stack_valid || !stack_valid[depth - 1]) {
        if (!reset_stack && depth > 0 && depth < stack_cap && stack && stack_valid) {
            compute_world_xform(prim, world);
        } else {
            memcpy(world, local, sizeof(local));
        }
    } else {
        nanousd_mul_m4d(local, stack[depth - 1], world);
    }

    if (depth >= 0 && depth < stack_cap && stack && stack_valid) {
        memcpy(stack[depth], world, sizeof(double) * 16);
        stack_valid[depth] = 1;
    }
}

/* ----------------------------------------------------------------
 * Count total triangulated indices for a mesh given faceVertexCounts and
 * the actual faceVertexIndices length.
 * Each face with N vertices produces (N-2) triangles = (N-2)*3 indices.
 *
 * USD files in the wild are sometimes malformed: faceVertexCounts claims
 * more face-vertices than faceVertexIndices provides. Treat the first
 * incomplete face as end-of-mesh instead of reading past the authored index
 * array.
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

typedef struct {
    NanousdPrim prim;
    int*        face_indices;
    int         nfaces;
    int         nindices;
    int         material_index;
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

    int nfaces = 0;
    if (fvc_data && fvc_count > 0) {
        nfaces = fvc_count;
    } else {
        nfaces = mesh->nindices / 3;
    }
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

static void set_mesh_path(SceneMesh* mesh, const char* path)
{
    if (!mesh) return;
    if (path && path[0]) {
        size_t n = strnlen(path, sizeof(mesh->path) - 1);
        memcpy(mesh->path, path, n);
        mesh->path[n] = '\0';
    } else {
        mesh->path[0] = '\0';
    }
}

static int scene_mesh_reserve(Scene* scene, int* capacity, int needed, int limit)
{
    if (!scene || !capacity || needed <= *capacity) return 1;
    if (limit > 0 && needed > limit) return 0;

    int new_cap = *capacity > 0 ? *capacity : 1024;
    while (new_cap < needed) {
        int next = new_cap < 1024 ? 1024 : new_cap * 2;
        if (next < new_cap || (limit > 0 && next > limit)) {
            next = limit;
        }
        if (next <= new_cap) return 0;
        new_cap = next;
    }

    if ((size_t)new_cap > ((size_t)-1) / sizeof(SceneMesh))
        return 0;
    SceneMesh* grown =
        (SceneMesh*)realloc(scene->meshes, (size_t)new_cap * sizeof(SceneMesh));
    if (!grown) return 0;
    if (new_cap > *capacity) {
        memset(grown + *capacity, 0,
               (size_t)(new_cap - *capacity) * sizeof(SceneMesh));
    }
    scene->meshes = grown;
    *capacity = new_cap;
    return 1;
}

/* Grow scene->pi_batches to hold at least `needed` entries. Returns 0 on
 * allocation failure. Geometric growth; small absolute caps are fine
 * because Moana has ~23 PointInstancers × <30 prototypes × few sub-meshes
 * each (~1k batches max in practice). */
static int scene_pi_batches_reserve(Scene* scene, int* capacity, int needed)
{
    if (!scene || !capacity || needed <= *capacity) return 1;
    int new_cap = *capacity > 0 ? *capacity : 64;
    while (new_cap < needed) {
        int next = new_cap * 2;
        if (next < new_cap) return 0;
        new_cap = next;
    }
    if ((size_t)new_cap > ((size_t)-1) / sizeof(SceneInstanceBatch))
        return 0;
    SceneInstanceBatch* grown = (SceneInstanceBatch*)realloc(
        scene->pi_batches, (size_t)new_cap * sizeof(SceneInstanceBatch));
    if (!grown) return 0;
    if (new_cap > *capacity) {
        memset(grown + *capacity, 0,
               (size_t)(new_cap - *capacity) * sizeof(SceneInstanceBatch));
    }
    scene->pi_batches = grown;
    *capacity = new_cap;
    return 1;
}

/* Grow scene->pi_transforms by `add` entries and return a pointer to the
 * first new slot (caller fills it). Returns NULL on allocation failure.
 * Capacity tracked via the in/out *capacity parameter so callers may keep
 * the bump cost amortized. */
static SceneInstanceTransform* scene_pi_transforms_reserve(
        Scene* scene, uint64_t* capacity, uint64_t add)
{
    if (!scene || !capacity) return NULL;
    if (add == 0) return scene->pi_transforms
                       ? &scene->pi_transforms[scene->npi_transforms]
                       : NULL;
    uint64_t needed = scene->npi_transforms + add;
    if (needed < add) return NULL;  /* overflow */
    if (needed > *capacity) {
        uint64_t new_cap = *capacity ? *capacity : 4096;
        while (new_cap < needed) {
            uint64_t next = new_cap * 2;
            if (next < new_cap) return NULL;
            new_cap = next;
        }
        if (new_cap > ((size_t)-1) / sizeof(SceneInstanceTransform))
            return NULL;
        SceneInstanceTransform* grown = (SceneInstanceTransform*)realloc(
            scene->pi_transforms,
            (size_t)new_cap * sizeof(SceneInstanceTransform));
        if (!grown) return NULL;
        scene->pi_transforms = grown;
        *capacity = new_cap;
    }
    SceneInstanceTransform* slot = &scene->pi_transforms[scene->npi_transforms];
    scene->npi_transforms = needed;
    return slot;
}

static long long scene_env_ll_limit(const char* name, long long fallback)
{
    const char* e = getenv(name);
    if (!e || !e[0]) return fallback;
    char* end = NULL;
    long long v = strtoll(e, &end, 10);
    if (end == e || v < 0) return fallback;
    return v;
}

static int scene_env_flag_requested(const char* name)
{
    const char* e = getenv(name);
    if (!e || !e[0]) return 0;
    if (e[0] == '0' || !strcmp(e, "false") || !strcmp(e, "off") ||
        !strcmp(e, "no")) {
        return 0;
    }
    return 1;
}

static long long scene_maxrss_bytes(void)
{
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0)
        return 0;
#ifdef __APPLE__
    return (long long)ru.ru_maxrss;
#else
    return (long long)ru.ru_maxrss * 1024ll;
#endif
}

static int scene_no_cull_all_geometry_requested(void)
{
    return scene_env_flag_requested("NUSD_NO_CULL_ALL_GEOMETRY") ||
           scene_env_flag_requested("NUSD_ALL_GEOMETRY_NO_CULL");
}

static int scene_fail_on_limit(void)
{
    return scene_no_cull_all_geometry_requested() ||
           scene_env_flag_requested("NUSD_FAIL_ON_LIMIT");
}

static long long scene_rss_limit_bytes(void)
{
    long long mib = scene_env_ll_limit("NUSD_MAX_RSS_MIB", 0);
    if (mib > 0) {
        const long long scale = 1024ll * 1024ll;
        if (mib > LLONG_MAX / scale)
            return LLONG_MAX;
        return mib * scale;
    }
    return scene_env_ll_limit("NUSD_MAX_RSS_BYTES", 0);
}

static int scene_check_rss_limit(Scene* scene, const char* phase)
{
    long long limit = scene_rss_limit_bytes();
    if (limit <= 0)
        return 0;
    long long rss = scene_maxrss_bytes();
    if (rss <= 0 || rss <= limit)
        return 0;
    scene_fail(scene,
               "RSS budget exceeded during %s "
               "(maxrss=%.1f MiB, limit=%.1f MiB; "
               "raise NUSD_MAX_RSS_MIB or set it to 0 to disable)",
               phase ? phase : "scene load",
               (double)rss / (1024.0 * 1024.0),
               (double)limit / (1024.0 * 1024.0));
    return 1;
}

static void scene_fail_mesh_record_limit(Scene* scene,
                                         const char* phase,
                                         int needed,
                                         int limit,
                                         const char* path)
{
    scene_fail(scene,
               "mesh record budget exceeded during %s before %s "
               "(needed=%d, limit=%d; raise NUSD_MAX_SCENE_MESHES or "
               "lower instance/PointInstancer expansion)",
               phase ? phase : "scene load",
               (path && path[0]) ? path : "<unknown>",
               needed, limit);
}

static void scene_fail_geometry_budget(Scene* scene,
                                       const char* phase,
                                       const char* path,
                                       long long used_vertices,
                                       long long add_vertices,
                                       long long vertex_limit,
                                       long long used_indices,
                                       long long add_indices,
                                       long long index_limit)
{
    scene_fail(scene,
               "geometry budget exceeded during %s before %s "
               "(vertices=%lld+%lld limit=%lld, indices=%lld+%lld "
               "limit=%lld; raise NUSD_MAX_GEOM_VERTICES/INDICES or set "
               "them to 0)",
               phase ? phase : "mesh extraction",
               (path && path[0]) ? path : "<unknown>",
               used_vertices, add_vertices, vertex_limit,
               used_indices, add_indices, index_limit);
}

static int scene_flat_native_instance_traversal_enabled(void)
{
    const char* e = getenv("NUSD_FLAT_NATIVE_INSTANCE_TRAVERSAL");
    if (e && e[0])
        return scene_env_flag_requested("NUSD_FLAT_NATIVE_INSTANCE_TRAVERSAL");
    return scene_no_cull_all_geometry_requested();
}

static int scene_geometry_only_fast_path(void)
{
    if (scene_env_flag_requested("NUSD_GEOMETRY_ONLY"))
        return 1;
    return scene_no_cull_all_geometry_requested() && !s_load_materials;
}

static int scene_geometry_only_skip_normals(void)
{
    return scene_geometry_only_fast_path() &&
           !scene_env_flag_requested("NUSD_GEOMETRY_ONLY_KEEP_NORMALS");
}

static int scene_geometry_only_skip_display_color(void)
{
    return scene_geometry_only_fast_path() &&
           !scene_env_flag_requested("NUSD_GEOMETRY_ONLY_KEEP_DISPLAY_COLOR");
}

static int scene_native_replay_diag_enabled(void)
{
    return scene_env_flag_requested("NUSD_NATIVE_REPLAY_DIAG");
}

static int scene_native_replay_arc_diag_enabled(void)
{
    return scene_env_flag_requested("NUSD_NATIVE_REPLAY_ARC_DIAG");
}

static int scene_material_arc_path_like(const char* path)
{
    if (!path || !path[0])
        return 0;
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

static int scene_skip_arc_for_geometry_only(const char* layer_or_asset,
                                            const char* source_or_target)
{
    if (!scene_geometry_only_fast_path())
        return 0;
    return scene_material_arc_path_like(layer_or_asset) ||
           scene_material_arc_path_like(source_or_target);
}

static int scene_geometry_budget_allows(long long used_vertices,
                                        long long add_vertices,
                                        long long vertex_limit,
                                        long long used_indices,
                                        long long add_indices,
                                        long long index_limit)
{
    if (vertex_limit > 0 &&
        (add_vertices > vertex_limit ||
         used_vertices > vertex_limit - add_vertices)) {
        return 0;
    }
    if (index_limit > 0 &&
        (add_indices > index_limit ||
         used_indices > index_limit - add_indices)) {
        return 0;
    }
    return 1;
}

static void scene_log_geometry_budget(const char* phase,
                                      const char* path,
                                      long long used_vertices,
                                      long long add_vertices,
                                      long long vertex_limit,
                                      long long used_indices,
                                      long long add_indices,
                                      long long index_limit)
{
    fprintf(stderr,
            "scene_load: geometry budget reached during %s; "
            "truncating before %s "
            "(vertices %lld + %lld / limit %lld, "
            "indices %lld + %lld / limit %lld; "
            "NUSD_MAX_GEOM_VERTICES/INDICES=0 disables)\n",
            phase ? phase : "mesh extraction",
            (path && path[0]) ? path : "<unknown>",
            used_vertices, add_vertices, vertex_limit,
            used_indices, add_indices, index_limit);
}

static int split_mesh_by_geom_subsets(Arena* arena,
                                      Scene* scene,
                                      NanousdStage stage,
                                      NanousdPrim mesh_prim,
                                      int base_mesh_idx,
                                      int* max_meshes,
                                      int mesh_limit,
                                      const int* fvc_data,
                                      int fvc_count,
                                      int fvi_count)
{
    if (!arena || !scene || !mesh_prim || base_mesh_idx < 0 ||
        !max_meshes || base_mesh_idx >= *max_meshes)
        return 1;

    SceneMesh base = scene->meshes[base_mesh_idx];

    /* No children => no GeomSubsets to split: return the base mesh unchanged.
       Checked BEFORE build_mesh_face_index_ranges so meshes without authored
       faceVertexCounts (e.g. UsdGeomPoints-generated box meshes, whose fvc_data
       is absent/invalid) never dereference fvc_data. (ASan: SEGV at scene.c:809.) */
    int nch = nanousd_nchildren(mesh_prim);
    if (nch <= 0)
        return 1;

    int* face_starts = NULL;
    int* face_counts = NULL;
    int nfaces = 0;
    if (!build_mesh_face_index_ranges(arena, &base, fvc_data, fvc_count,
                                      fvi_count, &face_starts, &face_counts,
                                      &nfaces))
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
        info->material_index = materials_find_binding(
            (MaterialCollection*)scene->materials, stage, child);
        if (info->material_index < 0)
            info->material_index = base.material_index;
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
    if (!scene_mesh_reserve(scene, max_meshes, base_mesh_idx + out_count,
                            mesh_limit)) {
        for (int s = 0; s < nsubsets; s++)
            nanousd_freeprim(subsets[s].prim);
        return 1;
    }

    /* USD GeomSubset material binding is a face-domain partition under a
     * Mesh. The GPU backends, including Metal and Vulkan, bind one material
     * per draw record. Splitting here preserves the authored USD material
     * partition while keeping the renderer-side SceneMesh contract simple. */
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
        out->material_index = subsets[s].material_index;
        set_mesh_path(out, subsets[s].path);
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
            out->prototype_idx = base_mesh_idx + emitted;
            out->vertex_offset = 0;
            out->index_offset = 0;
            emitted++;
        }
    }

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

static char* scene_strdup(const char* s)
{
    if (!s) return NULL;
    size_t n = strlen(s);
    char* out = (char*)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
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
    if (!bind || !rest || !local || !skel_xforms || !skin_xforms)
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
            xform_dir(geom_bind, n0, n_skel);
            for (int k = 0; k < influence_count; k++) {
                int skel_idx = skin_joint_index_for_vertex(
                    v, influence_count, joint_indices,
                    mesh_joint_count, mesh_to_skel, k);
                if (skel_idx < 0 || skel_idx >= njoints) continue;
                float w = joint_weights[v * influence_count + k];
                double tn[3];
                xform_dir(&skin_xforms[skel_idx * 16],
                          (float[3]){(float)n_skel[0], (float)n_skel[1], (float)n_skel[2]},
                          tn);
                accum[0] += (double)w * tn[0];
                accum[1] += (double)w * tn[1];
                accum[2] += (double)w * tn[2];
            }
            double n_mesh[3];
            xform_dir(skel_to_mesh,
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
            xform_dir(geom_bind, n0, n_skel);
            for (int k = 0; k < influence_count; k++) {
                int skel_idx = skin_joint_index_for_vertex(
                    v, influence_count, joint_indices,
                    mesh_joint_count, mesh_to_skel, k);
                if (skel_idx < 0 || skel_idx >= njoints) continue;
                float w = joint_weights[v * influence_count + k];
                double tn[3];
                xform_dir(&skin_xforms[skel_idx * 16],
                          (float[3]){(float)n_skel[0], (float)n_skel[1], (float)n_skel[2]},
                          tn);
                accum[0] += (double)w * tn[0];
                accum[1] += (double)w * tn[1];
                accum[2] += (double)w * tn[2];
            }
            double n_mesh[3];
            xform_dir(skel_to_mesh,
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
    nanousd_freeprim(skel);
    return 0;
}

/* ----------------------------------------------------------------
 * Expand a mesh when USD authors face-varying normals.
 *
 * The renderer's vertex layout stores one normal per uploaded vertex. USD
 * commonly stores normals per face-vertex, allowing hard edges on shared
 * point topology. Split each triangulated face-vertex into its own uploaded
 * vertex so the authored normal survives.
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

/* ----------------------------------------------------------------
 * Load UV primvars and face-varying normals. When any supported primvar is
 * faceVarying, expand the mesh so UV seams and hard normals survive Metal's
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

static void bounds_init(float mn[3], float mx[3])
{
    mn[0] = mn[1] = mn[2] = FLT_MAX;
    mx[0] = mx[1] = mx[2] = -FLT_MAX;
}

static void bounds_set_never_cull(float mn[3], float mx[3])
{
    mn[0] = mn[1] = mn[2] = -FLT_MAX;
    mx[0] = mx[1] = mx[2] =  FLT_MAX;
}

static int bounds_is_usable(const float mn[3], const float mx[3])
{
    const float limit = 1.0e20f;
    for (int k = 0; k < 3; k++) {
        if (!isfinite(mn[k]) || !isfinite(mx[k])) return 0;
        if (mn[k] > mx[k]) return 0;
        if (fabsf(mn[k]) > limit || fabsf(mx[k]) > limit) return 0;
    }
    return 1;
}

static void scene_expand_bounds(Scene* scene, const float mn[3], const float mx[3])
{
    if (!scene) return;
    if (!bounds_is_usable(mn, mx)) return;
    for (int k = 0; k < 3; k++) {
        if (mn[k] < scene->bounds_min[k]) scene->bounds_min[k] = mn[k];
        if (mx[k] > scene->bounds_max[k]) scene->bounds_max[k] = mx[k];
    }
}

static void world_bounds_from_local(const double xform[16],
                                    const float local_min[3],
                                    const float local_max[3],
                                    float world_min[3],
                                    float world_max[3])
{
    bounds_init(world_min, world_max);
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
                    if (wp[k] < world_min[k]) world_min[k] = wp[k];
                    if (wp[k] > world_max[k]) world_max[k] = wp[k];
                }
            }
        }
    }
}

static float (*s_pi_frustum_planes)[6][4] = NULL;
static int s_pi_frustum_count = 0;

void scene_set_point_instance_frusta(const float* planes, int num_cameras)
{
    free(s_pi_frustum_planes);
    s_pi_frustum_planes = NULL;
    s_pi_frustum_count = 0;

    if (!planes || num_cameras <= 0) return;
    size_t bytes = (size_t)num_cameras * 6u * 4u * sizeof(float);
    s_pi_frustum_planes = (float (*)[6][4])malloc(bytes);
    if (!s_pi_frustum_planes) return;
    memcpy(s_pi_frustum_planes, planes, bytes);
    s_pi_frustum_count = num_cameras;
}

static int scene_point_instance_cull_active(void)
{
    if (scene_no_cull_all_geometry_requested()) return 0;
    if (!s_pi_frustum_planes || s_pi_frustum_count <= 0) return 0;
    const char* e = getenv("NUSD_PI_FRUSTUM_CULL");
    if (e && e[0] && (e[0] == '0' || !strcmp(e, "false") ||
                      !strcmp(e, "off") || !strcmp(e, "no"))) {
        return 0;
    }
    return 1;
}

static float scene_point_instance_cull_pad(void)
{
    const char* e = getenv("NUSD_PI_CULL_PAD");
    if (!e || !e[0]) return 0.0f;
    char* end = NULL;
    float pad = strtof(e, &end);
    if (end == e || !isfinite(pad) || pad < 0.0f) return 0.0f;
    return pad;
}

static int scene_aabb_in_frustum_planes(const float planes[6][4],
                                        const float mn[3],
                                        const float mx[3])
{
    if (!bounds_is_usable(mn, mx)) return 1;
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

static int scene_aabb_visible_in_any_pi_frustum(const float mn[3],
                                                const float mx[3])
{
    if (!scene_point_instance_cull_active()) return 1;
    if (!bounds_is_usable(mn, mx)) return 1;
    for (int c = 0; c < s_pi_frustum_count; c++) {
        if (scene_aabb_in_frustum_planes(s_pi_frustum_planes[c], mn, mx))
            return 1;
    }
    return 0;
}

static int scene_curve_frustum_cull_active(void)
{
    if (scene_no_cull_all_geometry_requested()) return 0;
    if (!s_pi_frustum_planes || s_pi_frustum_count <= 0) return 0;
    const char* e = getenv("NUSD_CURVE_FRUSTUM_CULL");
    if (e && e[0] && (e[0] == '0' || !strcmp(e, "false") ||
                      !strcmp(e, "off") || !strcmp(e, "no"))) {
        return 0;
    }
    return 1;
}

static int scene_aabb_visible_in_any_frustum_padded(const float mn[3],
                                                    const float mx[3],
                                                    float pad)
{
    if (!s_pi_frustum_planes || s_pi_frustum_count <= 0) return 1;
    if (!bounds_is_usable(mn, mx)) return 1;
    float pmn[3] = { mn[0] - pad, mn[1] - pad, mn[2] - pad };
    float pmx[3] = { mx[0] + pad, mx[1] + pad, mx[2] + pad };
    for (int c = 0; c < s_pi_frustum_count; c++) {
        if (scene_aabb_in_frustum_planes(s_pi_frustum_planes[c], pmn, pmx))
            return 1;
    }
    return 0;
}

static int mesh_local_bounds(const SceneMesh* mesh, float mn[3], float mx[3])
{
    if (!mesh || !mesh->positions || mesh->nvertices <= 0) return 0;
    bounds_init(mn, mx);
    for (int v = 0; v < mesh->nvertices; v++) {
        const float* p = &mesh->positions[v * 3];
        for (int k = 0; k < 3; k++) {
            if (p[k] < mn[k]) mn[k] = p[k];
            if (p[k] > mx[k]) mx[k] = p[k];
        }
    }
    return bounds_is_usable(mn, mx);
}

typedef struct {
    unsigned char state; /* 0 unknown, 1 valid, 2 invalid */
    float mn[3];
    float mx[3];
} PiProtoLocalBounds;

typedef struct {
    char** items;
    int    count;
    int    cap;
} PathPrefixList;

typedef struct {
    NanousdPrim prim;
    NanousdStage stage;
    char        instance_path[512];
    char        relative_path[512];
    int         resolved_in_proto_space;
    int         has_override_world;
    double      override_world[16];
} InstanceChildWork;

typedef struct {
    char   relative_path[512];
    int    prototype_idx;
    double relative_xform[16];
} NativePrototypeReplayEntry;

typedef struct {
    char key[768];
    NativePrototypeReplayEntry* entries;
    int count;
    int cap;
    int built;
    int roots_seen;
} NativePrototypeReplayCatalog;

typedef struct {
    NativePrototypeReplayCatalog* items;
    int count;
    int cap;
} NativePrototypeReplayCache;

typedef struct {
    NanousdStage* stages;
    int count;
    int cap;
} NativeAssetStageList;

static int path_has_prefix(const char* path, const char* prefix)
{
    if (!path || !prefix || !prefix[0]) return 0;
    size_t n = strlen(prefix);
    if (strncmp(path, prefix, n) != 0) return 0;
    return path[n] == '\0' || path[n] == '/';
}

static void append_cstr_bounded(char* dst, size_t dst_size, const char* src)
{
    if (!dst || dst_size == 0 || !src) return;
    size_t used = strlen(dst);
    if (used >= dst_size - 1) return;
    snprintf(dst + used, dst_size - used, "%s", src);
}

static void append_variant_signature(NanousdPrim prim,
                                     const char* label,
                                     char* dst,
                                     size_t dst_size)
{
    if (!prim || !dst || dst_size == 0) return;
    int nsets = nanousd_nvariantsets(prim);
    for (int i = 0; i < nsets; i++) {
        const char* set_name = nanousd_variantsetname(prim, i);
        if (!set_name || !set_name[0]) continue;
        const char* selected = nanousd_variantselection(prim, set_name);
        if (!selected || !selected[0]) continue;
        append_cstr_bounded(dst, dst_size, "|");
        append_cstr_bounded(dst, dst_size, label ? label : "prim");
        append_cstr_bounded(dst, dst_size, ":");
        append_cstr_bounded(dst, dst_size, set_name);
        append_cstr_bounded(dst, dst_size, "=");
        append_cstr_bounded(dst, dst_size, selected);
    }
}

static void native_instance_prototype_key(NanousdPrim inst_prim,
                                          NanousdPrim proto_root,
                                          char* out,
                                          size_t out_size)
{
    if (!out || out_size == 0) return;
    int key_len = nanousd_instance_key(inst_prim, out, out_size);
    if (key_len > 0 && out[0]) {
        if (proto_root) {
            const char* proto_path = nanousd_path(proto_root);
            if (proto_path && proto_path[0]) {
                append_cstr_bounded(out, out_size, "|proto=");
                append_cstr_bounded(out, out_size, proto_path);
            }
        }
        return;
    }
    const char* proto_path = proto_root ? nanousd_path(proto_root) : NULL;
    snprintf(out, out_size, "%s", proto_path && proto_path[0] ? proto_path : "<no-prototype>");
    append_variant_signature(inst_prim, "inst", out, out_size);
    append_variant_signature(proto_root, "proto", out, out_size);
}

static NativePrototypeReplayCatalog*
native_replay_cache_find_or_add(NativePrototypeReplayCache* cache,
                                const char* key,
                                int* out_added)
{
    if (out_added) *out_added = 0;
    if (!cache || !key || !key[0]) return NULL;
    for (int i = 0; i < cache->count; i++) {
        if (strcmp(cache->items[i].key, key) == 0)
            return &cache->items[i];
    }
    if (cache->count >= cache->cap) {
        int new_cap = cache->cap ? cache->cap * 2 : 32;
        NativePrototypeReplayCatalog* grown =
            (NativePrototypeReplayCatalog*)realloc(
                cache->items,
                (size_t)new_cap * sizeof(NativePrototypeReplayCatalog));
        if (!grown) return NULL;
        memset(grown + cache->cap, 0,
               (size_t)(new_cap - cache->cap) * sizeof(NativePrototypeReplayCatalog));
        cache->items = grown;
        cache->cap = new_cap;
    }
    NativePrototypeReplayCatalog* cat = &cache->items[cache->count++];
    memset(cat, 0, sizeof(*cat));
    snprintf(cat->key, sizeof(cat->key), "%s", key);
    if (out_added) *out_added = 1;
    return cat;
}

static int native_replay_catalog_add(NativePrototypeReplayCatalog* cat,
                                     const char* relative_path,
                                     int prototype_idx,
                                     const double relative_xform[16])
{
    if (!cat || prototype_idx < 0 || !relative_xform) return 0;
    const char* rel = relative_path ? relative_path : "";
    while (*rel == '/') rel++;
    for (int i = 0; i < cat->count; i++) {
        if (strcmp(cat->entries[i].relative_path, rel) == 0)
            return 1;
    }
    if (cat->count >= cat->cap) {
        int new_cap = cat->cap ? cat->cap * 2 : 64;
        NativePrototypeReplayEntry* grown =
            (NativePrototypeReplayEntry*)realloc(
                cat->entries,
                (size_t)new_cap * sizeof(NativePrototypeReplayEntry));
        if (!grown) return 0;
        memset(grown + cat->cap, 0,
               (size_t)(new_cap - cat->cap) * sizeof(NativePrototypeReplayEntry));
        cat->entries = grown;
        cat->cap = new_cap;
    }
    NativePrototypeReplayEntry* e = &cat->entries[cat->count++];
    snprintf(e->relative_path, sizeof(e->relative_path), "%s", rel);
    e->prototype_idx = prototype_idx;
    memcpy(e->relative_xform, relative_xform, sizeof(double) * 16);
    return 1;
}

static void native_replay_cache_free(NativePrototypeReplayCache* cache)
{
    if (!cache) return;
    for (int i = 0; i < cache->count; i++)
        free(cache->items[i].entries);
    free(cache->items);
    cache->items = NULL;
    cache->count = 0;
    cache->cap = 0;
}

static int instance_child_work_push(InstanceChildWork** stack,
                                    int* stack_size,
                                    int* stack_cap,
                                    NanousdPrim prim,
                                    const char* instance_path,
                                    int resolved_in_proto_space)
{
    if (!stack || !stack_size || !stack_cap || !prim) return 0;
    if (*stack_size >= *stack_cap) {
        int new_cap = *stack_cap ? *stack_cap * 2 : 4096;
        InstanceChildWork* grown =
            (InstanceChildWork*)realloc(*stack,
                                        (size_t)new_cap * sizeof(InstanceChildWork));
        if (!grown) return 0;
        *stack = grown;
        *stack_cap = new_cap;
    }
    InstanceChildWork* work = &(*stack)[(*stack_size)++];
    memset(work, 0, sizeof(*work));
    work->prim = prim;
    work->resolved_in_proto_space = resolved_in_proto_space;
    if (instance_path && instance_path[0]) {
        snprintf(work->instance_path, sizeof(work->instance_path),
                 "%s", instance_path);
    }
    return 1;
}

static int instance_child_work_push_override(InstanceChildWork** stack,
                                             int* stack_size,
                                             int* stack_cap,
                                             NanousdPrim prim,
                                             NanousdStage prim_stage,
                                             const char* instance_path,
                                             const char* relative_path,
                                             const double world_xform[16])
{
    int before = stack_size ? *stack_size : 0;
    if (!instance_child_work_push(stack, stack_size, stack_cap, prim,
                                  instance_path, 0))
        return 0;
    InstanceChildWork* work = &(*stack)[before];
    work->stage = prim_stage;
    if (relative_path && relative_path[0]) {
        const char* rel = relative_path;
        while (*rel == '/') rel++;
        snprintf(work->relative_path, sizeof(work->relative_path),
                 "%s", rel);
    }
    if (world_xform) {
        work->has_override_world = 1;
        memcpy(work->override_world, world_xform, sizeof(double) * 16);
    }
    return 1;
}

static int native_asset_stage_list_add(NativeAssetStageList* list,
                                       NanousdStage stage)
{
    if (!list || !stage) return 0;
    if (list->count >= list->cap) {
        int new_cap = list->cap ? list->cap * 2 : 8;
        NanousdStage* grown =
            (NanousdStage*)realloc(list->stages,
                                   (size_t)new_cap * sizeof(NanousdStage));
        if (!grown) return 0;
        list->stages = grown;
        list->cap = new_cap;
    }
    list->stages[list->count++] = stage;
    return 1;
}

static void native_asset_stage_list_close(NativeAssetStageList* list)
{
    if (!list) return;
    for (int i = 0; i < list->count; i++) {
        if (list->stages[i])
            nanousd_close(list->stages[i]);
    }
    free(list->stages);
    list->stages = NULL;
    list->count = 0;
    list->cap = 0;
}

static void instance_child_work_stack_free(InstanceChildWork* stack,
                                           int stack_size)
{
    if (!stack) return;
    for (int i = 0; i < stack_size; i++) {
        if (stack[i].prim)
            nanousd_freeprim(stack[i].prim);
    }
    free(stack);
}

static int parse_asset_arc_item(const char* item,
                                char* asset,
                                size_t asset_size,
                                char* target,
                                size_t target_size)
{
    if (!item || item[0] != '@' || !asset || asset_size == 0 ||
        !target || target_size == 0)
        return 0;
    asset[0] = '\0';
    target[0] = '\0';
    const char* asset_end = strchr(item + 1, '@');
    if (!asset_end || asset_end == item + 1)
        return 0;
    size_t asset_len = (size_t)(asset_end - (item + 1));
    if (asset_len >= asset_size)
        asset_len = asset_size - 1;
    memcpy(asset, item + 1, asset_len);
    asset[asset_len] = '\0';

    const char* lt = strchr(asset_end + 1, '<');
    const char* gt = lt ? strchr(lt + 1, '>') : NULL;
    if (lt && gt && gt > lt + 1) {
        size_t target_len = (size_t)(gt - (lt + 1));
        if (target_len >= target_size)
            target_len = target_size - 1;
        memcpy(target, lt + 1, target_len);
        target[target_len] = '\0';
    }
    return 1;
}

static int find_authored_arc_anchor(NanousdStage stage,
                                    const char* composed_path,
                                    const char* field,
                                    char* anchor_layer,
                                    size_t anchor_layer_size)
{
    if (!stage || !composed_path || !field ||
        !anchor_layer || anchor_layer_size == 0)
        return 0;
    anchor_layer[0] = '\0';
    int nlayers = nanousd_stage_n_layers(stage);
    const char* suffix = composed_path;
    while (suffix && suffix[0] == '/') {
        for (int li = 0; li < nlayers; li++) {
            NanousdListOp op =
                nanousd_layer_prim_listop(stage, li, suffix, field);
            if (op) {
                int nitems = nanousd_listop_nitems(op);
                nanousd_listop_free(op);
                if (nitems > 0) {
                    const char* lp = nanousd_stage_layer_path(stage, li);
                    if (lp && lp[0]) {
                        snprintf(anchor_layer, anchor_layer_size, "%s", lp);
                        return 1;
                    }
                }
            }
        }
        const char* next = strchr(suffix + 1, '/');
        suffix = next;
    }
    return 0;
}

static int apply_instance_variants_to_asset_target(NanousdPrim inst_prim,
                                                   NanousdStage asset_stage,
                                                   const char* target_path)
{
    if (!inst_prim || !asset_stage) return 0;
    NanousdPrim target = NULL;
    if (target_path && target_path[0])
        target = nanousd_primpath(asset_stage, target_path);
    if (!target)
        target = nanousd_defaultprim(asset_stage);
    if (!target) return 0;

    int changed = 0;
    int nsets = nanousd_nvariantsets(inst_prim);
    for (int i = 0; i < nsets; i++) {
        const char* set_name = nanousd_variantsetname(inst_prim, i);
        if (!set_name || !set_name[0])
            continue;
        const char* selected = nanousd_variantselection(inst_prim, set_name);
        if (!selected || !selected[0])
            continue;
        if (!nanousd_hasvariantset(target, set_name))
            continue;
        if (nanousd_setvariantselection(target, set_name, selected, 0))
            changed = 1;
    }
    nanousd_freeprim(target);
    return changed;
}

static int seed_asset_stage_meshes(NanousdStage asset_stage,
                                  NanousdPrim inst_prim,
                                  const char* target_path,
                                  const char* inst_root_path,
                                  const double inst_root_world[16],
                                  InstanceChildWork** stack,
                                  int* stack_size,
                                  int* stack_cap,
                                  NativeAssetStageList* owned_stages,
                                  int depth);

/* ---- Nested PointInstancer measurement stub (MEASURE-ONLY) ----
 *
 * A PointInstancer authored *inside* a native-instance asset subtree (e.g.
 * /isBeach/.../xgGroundCover/instancer). The compact Third Pass only handles
 * the ~23 top-level PIs on the main stage; these nested PIs are missed, so
 * their prototype meshes get seeded once as static geometry (piling at the
 * asset-local origin) and the millions of authored placements never happen.
 *
 * Before committing to a data model (flatten vs hierarchical) we must know the
 * TRUE flattened instance count. The authored protoIndices length of a nested
 * PI is NOT the flattened count when its own prototypes contain *another*
 * PointInstancer (cartesian nesting, e.g. xgTreeFill 679 × bonsai-inner
 * 13,025). This stub logs, per nested PI:
 *   - authored instance count (this PI's protoIndices length)
 *   - whether its prototype subtree contains a nested PI (cartesian flag)
 *   - the inner authored total under its prototypes (the cartesian multiplier)
 * and maintains running grand totals so the last NUSD_GEO_DIAG line carries
 * the measured flattened estimate. No geometry is emitted yet. */
static long long s_nested_pi_count = 0;
static long long s_nested_pi_authored_total = 0;   /* Sum outer protoIndices len */
static long long s_nested_pi_cartesian_count = 0;
static long long s_nested_pi_flattened_estimate = 0; /* Sum (outer x inner|1) */

/* Sum authored protoIndices lengths of every PointInstancer found strictly
 * under `proto_prefix` in `asset_stage`. Returns 0 if none (proto subtree is
 * static geometry only). Cheap linear scan of the flat prim list. */
static long long nested_pi_inner_authored_under(NanousdStage asset_stage,
                                                const char* proto_prefix)
{
    if (!asset_stage || !proto_prefix || !proto_prefix[0]) return 0;
    long long inner = 0;
    int nprims = nanousd_nprims(asset_stage);
    for (int i = 0; i < nprims; i++) {
        NanousdPrim p = nanousd_prim(asset_stage, i);
        if (!p) continue;
        const char* pp = nanousd_path(p);
        const char* tn = nanousd_typename(p);
        if (pp && tn && strcmp(tn, "PointInstancer") == 0 &&
            path_has_prefix(pp, proto_prefix) &&
            strcmp(pp, proto_prefix) != 0) {
            int n = nanousd_attribarraylen(p, "protoIndices");
            if (n > 0) inner += n;
        }
        nanousd_freeprim(p);
    }
    return inner;
}

/* ---- Nested PointInstancer compact expansion (hierarchical-decode +
 *      flatten-emit; see docs/moana-no-cull-memory-plan.md Debate #1) ----
 *
 * nested_pi_record() runs during the native-instance asset SEED (per catalog
 * build). It decodes the PI's authored arrays ONCE and stores per-prototype
 * instance-root-LOCAL transforms (local = inst_xform * pi_world_asset *
 * target_inv) in a per-instance-root pending list. After the instance root's
 * child walk has created the prototype meshes (as is_proto_only, so they do
 * not pile at the asset origin), scene_drain_nested_pis() resolves each
 * prototype's concrete path to scene-mesh indices, composes leaf_world =
 * local * inst_root_world, and appends compact SceneInstanceTransform rows +
 * SceneInstanceBatch records — the SAME compact model the top-level Third
 * Pass uses, so no renderer/shader change is needed.
 *
 * Dedup: (a) the 6x intra-build re-seeding of one asset is collapsed by a
 * per-root concrete-path check; (b) PIs the top-level Third Pass already owns
 * (isCoral/isNaupakaA, which are composed-visible) are skipped via
 * s_toplevel_pi_paths. */

typedef struct {
    int                     proto_mesh_idx;   /* resolved at drain (-1 until) */
    char                    proto_concrete[512]; /* inst_root + (proto-target) */
    SceneInstanceTransform* local;            /* [count] inst-root-local 12f */
    uint32_t                count;
} NestedPiProtoGroup;

typedef struct {
    char                pi_concrete[512];      /* dedup + top-level skip key */
    double              inst_root_world[16];
    NestedPiProtoGroup* groups;
    int                 ngroups;
    int                 cartesian;
} NestedPiPending;

static NestedPiPending* s_nested_pending = NULL;
static int s_nested_pending_count = 0;
static int s_nested_pending_cap = 0;

/* Top-level PI concrete paths (owned by the Third Pass) the nested path skips. */
static char (*s_toplevel_pi_paths)[512] = NULL;
static int s_toplevel_pi_count = 0;
static int s_toplevel_pi_cap = 0;

/* Running ledger of nested-PI work actually emitted (for the summary line). */
static long long s_nested_pi_emitted_leaves = 0;
static int       s_nested_pi_emitted_batches = 0;
static int       s_nested_pi_unique = 0;
static int       s_nested_pi_skipped_toplevel = 0;
static int       s_nested_pi_skipped_dup = 0;

static void toplevel_pi_paths_reset(void)
{
    free(s_toplevel_pi_paths);
    s_toplevel_pi_paths = NULL;
    s_toplevel_pi_count = 0;
    s_toplevel_pi_cap = 0;
}

/* Normalize a PI path to an element-scoped skip key "<element>|<piname>",
 * where element = the component right under /island/ and piname = the last
 * path component. This collapses the native-instance MASTER path
 * (/island/isCoral/isCoral/geometry/xgFlutes) and the composed PROXY paths
 * (/island/isCoral/isCoral5/geometry/xgFlutes) the top-level Third Pass
 * actually places, so the nested seed path skips coral/naupaka instead of
 * double-counting them. Element-scoped so a nested isBeach/.../instancer does
 * not collide with isNaupakaA/.../instancer. */
static void pi_skip_key(const char* path, char* out, size_t out_sz)
{
    if (out_sz) out[0] = '\0';
    if (!path || path[0] != '/' || out_sz == 0) return;
    const char* c0 = path + 1;            /* "island" */
    const char* s0 = strchr(c0, '/');
    if (!s0) return;
    const char* c1 = s0 + 1;              /* element */
    const char* s1 = strchr(c1, '/');
    int elen = s1 ? (int)(s1 - c1) : (int)strlen(c1);
    const char* last = strrchr(path, '/');
    last = last ? last + 1 : path;
    snprintf(out, out_sz, "%.*s|%s", elen, c1, last);
}

static void toplevel_pi_paths_add(const char* path)
{
    char key[256];
    pi_skip_key(path, key, sizeof(key));
    if (!key[0]) return;
    for (int i = 0; i < s_toplevel_pi_count; i++)
        if (strcmp(s_toplevel_pi_paths[i], key) == 0) return;  /* dedup */
    if (s_toplevel_pi_count >= s_toplevel_pi_cap) {
        int nc = s_toplevel_pi_cap ? s_toplevel_pi_cap * 2 : 64;
        char (*g)[512] = (char (*)[512])realloc(s_toplevel_pi_paths,
                                                (size_t)nc * 512);
        if (!g) return;
        s_toplevel_pi_paths = g;
        s_toplevel_pi_cap = nc;
    }
    snprintf(s_toplevel_pi_paths[s_toplevel_pi_count++], 512, "%s", key);
}

static int toplevel_pi_paths_contains(const char* path)
{
    char key[256];
    pi_skip_key(path, key, sizeof(key));
    if (!key[0]) return 0;
    for (int i = 0; i < s_toplevel_pi_count; i++)
        if (strcmp(s_toplevel_pi_paths[i], key) == 0) return 1;
    return 0;
}

/* Pack a row-vector affine double[16] into the renderer's 12-float
 * column-major layout (drops the always-(0,0,0,1) 4th row), matching the
 * Third Pass pack at scene.c's slab fill. */
static void affine16_to_12f(const double m[16], float out[12])
{
    out[0]  = (float)m[0];  out[1]  = (float)m[1];  out[2]  = (float)m[2];
    out[3]  = (float)m[4];  out[4]  = (float)m[5];  out[5]  = (float)m[6];
    out[6]  = (float)m[8];  out[7]  = (float)m[9];  out[8]  = (float)m[10];
    out[9]  = (float)m[12]; out[10] = (float)m[13]; out[11] = (float)m[14];
}

/* Inverse: 12-float compact -> row-vector affine double[16]. */
static void compact12_to_affine16(const float f[12], double m[16])
{
    m[0]=f[0];  m[1]=f[1];  m[2]=f[2];  m[3]=0.0;
    m[4]=f[3];  m[5]=f[4];  m[6]=f[5];  m[7]=0.0;
    m[8]=f[6];  m[9]=f[7];  m[10]=f[8]; m[11]=0.0;
    m[12]=f[9]; m[13]=f[10]; m[14]=f[11]; m[15]=1.0;
}

/* Quaternion (x,y,z,w) + scale + translate -> row-vector affine double[16],
 * matching the Third Pass instance-xform build. */
static void pi_trs_to_affine16(const float* q, const float* s, const float* t,
                               double m[16])
{
    double qi=q[0], qj=q[1], qk=q[2], w=q[3];
    double sx=s[0], sy=s[1], sz=s[2];
    double ww=w*w, ii=qi*qi, jj=qj*qj, kk=qk*qk;
    double wi=w*qi, wj=w*qj, wk=w*qk;
    double ij=qi*qj, ik=qi*qk, jk=qj*qk;
    m[0]=(ww+ii-jj-kk)*sx; m[1]=2*(ij+wk)*sx;     m[2]=2*(ik-wj)*sx;     m[3]=0;
    m[4]=2*(ij-wk)*sy;     m[5]=(ww-ii+jj-kk)*sy; m[6]=2*(jk+wi)*sy;     m[7]=0;
    m[8]=2*(ik+wj)*sz;     m[9]=2*(jk-wi)*sz;     m[10]=(ww-ii-jj+kk)*sz; m[11]=0;
    m[12]=t[0];            m[13]=t[1];            m[14]=t[2];            m[15]=1;
}

static void nested_pending_reset(void)
{
    for (int i = 0; i < s_nested_pending_count; i++) {
        for (int g = 0; g < s_nested_pending[i].ngroups; g++)
            free(s_nested_pending[i].groups[g].local);
        free(s_nested_pending[i].groups);
    }
    free(s_nested_pending);
    s_nested_pending = NULL;
    s_nested_pending_count = 0;
    s_nested_pending_cap = 0;
}

static NestedPiPending* nested_pending_find(const char* pi_concrete)
{
    for (int i = 0; i < s_nested_pending_count; i++)
        if (strcmp(s_nested_pending[i].pi_concrete, pi_concrete) == 0)
            return &s_nested_pending[i];
    return NULL;
}

/* Concrete path of an asset-stage prim under a native-instance root:
 *   concrete = inst_root + (asset_path - target).  Returns 0 if asset_path is
 *   not under target. */
static int nested_concrete_path(const char* asset_path, const char* target,
                                const char* inst_root, char* out, size_t out_sz)
{
    if (!asset_path || !target || !inst_root || !out) return 0;
    if (!path_has_prefix(asset_path, target)) return 0;
    const char* rel = asset_path + strlen(target);
    while (*rel == '/') rel++;
    if (rel[0])
        snprintf(out, out_sz, "%s/%s", inst_root, rel);
    else
        snprintf(out, out_sz, "%s", inst_root);
    return 1;
}

static void nested_pi_record(NanousdStage asset_stage,
                             const char* pi_path,
                             const char* actual_target_path,
                             const char* inst_root_path,
                             const double target_inv[16],
                             const double inst_root_world[16])
{
    if (!asset_stage || !pi_path || !actual_target_path || !inst_root_path ||
        !target_inv || !inst_root_world)
        return;

    char pi_concrete[512];
    if (!nested_concrete_path(pi_path, actual_target_path, inst_root_path,
                              pi_concrete, sizeof(pi_concrete)))
        return;

    /* Skip PIs the top-level Third Pass already owns (isCoral/isNaupakaA). */
    if (toplevel_pi_paths_contains(pi_concrete)) {
        s_nested_pi_skipped_toplevel++;
        return;
    }
    /* Per-root dedup of the 6x intra-build re-seed of one asset. */
    if (nested_pending_find(pi_concrete)) {
        s_nested_pi_skipped_dup++;
        return;
    }

    NanousdPrim pi = nanousd_primpath(asset_stage, pi_path);
    if (!pi) return;
    int n_instances = nanousd_attribarraylen(pi, "protoIndices");
    int n_protos    = nanousd_nreltargets(pi, "prototypes");
    if (n_instances <= 0 || n_protos <= 0) { nanousd_freeprim(pi); return; }

    /* Cartesian (nested-nested): a prototype that itself contains a PI. The
     * single Moana case (xgTreeFill) is handled by the deeper seed recursion
     * for now; here we only record the cartesian flag and still place the
     * outer level so the trees populate. (Bounded 3-level flatten is the
     * documented follow-up.) */
    long long inner_total = 0;
    for (int p = 0; p < n_protos; p++) {
        const char* pp = nanousd_reltarget(pi, "prototypes", p);
        if (pp && pp[0]) inner_total += nested_pi_inner_authored_under(asset_stage, pp);
    }
    int cartesian = inner_total > 0;

    int*   proto_indices = (int*)malloc((size_t)n_instances * sizeof(int));
    float* positions     = (float*)malloc((size_t)n_instances * 3 * sizeof(float));
    float* orientations  = (float*)malloc((size_t)n_instances * 4 * sizeof(float));
    float* scales        = (float*)malloc((size_t)n_instances * 3 * sizeof(float));
    if (!proto_indices || !positions || !orientations || !scales) {
        free(proto_indices); free(positions); free(orientations); free(scales);
        nanousd_freeprim(pi); return;
    }
    nanousd_attribarrayi(pi, "protoIndices", proto_indices, n_instances);
    int npos = nanousd_attribarraylen(pi, "positions");
    if (npos > 0) nanousd_attribarrayf(pi, "positions", positions, npos * 3);
    else memset(positions, 0, (size_t)n_instances * 3 * sizeof(float));
    int norient = nanousd_attribarraylen(pi, "orientations");
    if (norient > 0) nanousd_attribarrayf(pi, "orientations", orientations, norient * 4);
    else for (int q = 0; q < n_instances; q++) {
        orientations[q*4+0]=0; orientations[q*4+1]=0;
        orientations[q*4+2]=0; orientations[q*4+3]=1;
    }
    int nscale = nanousd_attribarraylen(pi, "scales");
    if (nscale > 0) nanousd_attribarrayf(pi, "scales", scales, nscale * 3);
    else for (int s = 0; s < n_instances * 3; s++) scales[s] = 1.0f;

    /* pi_world_asset = PI instancer world in the asset stage. */
    double pi_world[16], pi_world_target_inv[16];
    compute_world_xform(pi, pi_world);
    /* fold target_inv once: M = pi_world * target_inv (applied after inst_xform). */
    nanousd_mul_m4d(pi_world, target_inv, pi_world_target_inv);

    /* Per-proto concrete prefixes (for drain-time scene-mesh resolution). */
    char (*proto_concrete)[512] = (char (*)[512])calloc((size_t)n_protos, 512);
    if (!proto_concrete) {
        free(proto_indices); free(positions); free(orientations); free(scales);
        nanousd_freeprim(pi); return;
    }
    for (int p = 0; p < n_protos; p++) {
        const char* pp = nanousd_reltarget(pi, "prototypes", p);
        if (pp && pp[0])
            nested_concrete_path(pp, actual_target_path, inst_root_path,
                                 proto_concrete[p], 512);
    }
    nanousd_freeprim(pi);

    /* Count per-proto, then pack local transforms grouped by proto. */
    uint32_t* pcount = (uint32_t*)calloc((size_t)n_protos, sizeof(uint32_t));
    if (!pcount) {
        free(proto_concrete); free(proto_indices);
        free(positions); free(orientations); free(scales);
        return;
    }
    for (int i = 0; i < n_instances; i++) {
        int pidx = proto_indices[i];
        if (pidx >= 0 && pidx < n_protos) pcount[pidx]++;
    }

    NestedPiProtoGroup* groups =
        (NestedPiProtoGroup*)calloc((size_t)n_protos, sizeof(NestedPiProtoGroup));
    if (!groups) {
        free(pcount); free(proto_concrete); free(proto_indices);
        free(positions); free(orientations); free(scales);
        return;
    }
    int ngroups = 0;
    for (int p = 0; p < n_protos; p++) {
        if (pcount[p] == 0 || !proto_concrete[p][0]) continue;
        NestedPiProtoGroup* g = &groups[ngroups];
        g->proto_mesh_idx = -1;
        snprintf(g->proto_concrete, sizeof(g->proto_concrete), "%s",
                 proto_concrete[p]);
        g->local = (SceneInstanceTransform*)malloc(
            (size_t)pcount[p] * sizeof(SceneInstanceTransform));
        if (!g->local) { ngroups++; continue; }
        g->count = 0;
        /* map proto index p -> group slot via a quick lookup below */
        ngroups++;
    }
    /* Fill: for each instance, find its group (proto p) and append local. */
    for (int i = 0; i < n_instances; i++) {
        int pidx = proto_indices[i];
        if (pidx < 0 || pidx >= n_protos || pcount[pidx] == 0 ||
            !proto_concrete[pidx][0])
            continue;
        /* group slot index == prefix-count of non-empty protos up to pidx */
        int slot = -1, seen = 0;
        for (int p = 0; p <= pidx; p++) {
            if (pcount[p] && proto_concrete[p][0]) {
                if (p == pidx) { slot = seen; break; }
                seen++;
            }
        }
        if (slot < 0 || !groups[slot].local) continue;
        double inst_xform[16], tmp[16], local[16];
        pi_trs_to_affine16(&orientations[i*4], &scales[i*3], &positions[i*3],
                           inst_xform);
        /* local = inst_xform * pi_world * target_inv */
        nanousd_mul_m4d(inst_xform, pi_world_target_inv, tmp);
        memcpy(local, tmp, sizeof(local));
        affine16_to_12f(local, groups[slot].local[groups[slot].count].m);
        groups[slot].count++;
    }

    free(pcount); free(proto_concrete);
    free(proto_indices); free(positions); free(orientations); free(scales);

    /* Append to the per-root pending list. */
    if (s_nested_pending_count >= s_nested_pending_cap) {
        int nc = s_nested_pending_cap ? s_nested_pending_cap * 2 : 64;
        NestedPiPending* gp = (NestedPiPending*)realloc(
            s_nested_pending, (size_t)nc * sizeof(NestedPiPending));
        if (!gp) {
            for (int g = 0; g < ngroups; g++) free(groups[g].local);
            free(groups);
            return;
        }
        s_nested_pending = gp;
        s_nested_pending_cap = nc;
    }
    NestedPiPending* e = &s_nested_pending[s_nested_pending_count++];
    memset(e, 0, sizeof(*e));
    snprintf(e->pi_concrete, sizeof(e->pi_concrete), "%s", pi_concrete);
    memcpy(e->inst_root_world, inst_root_world, sizeof(double) * 16);
    e->groups = groups;
    e->ngroups = ngroups;
    e->cartesian = cartesian;
    s_nested_pi_unique++;

    if (getenv("NUSD_GEO_DIAG")) {
        long long leaves = 0;
        for (int g = 0; g < ngroups; g++) leaves += groups[g].count;
        fprintf(stderr,
                "scene_load:   NESTED-PI-RECORD path=%s groups=%d leaves=%lld "
                "cartesian=%d\n", pi_concrete, ngroups, leaves, cartesian);
    }
}

/* Expand all pending nested PIs (filled during this instance root's seed) into
 * the compact scene->pi_transforms / pi_batches model. Resolves each
 * prototype's concrete path to scene-mesh indices among the proto-only meshes
 * the walk just created, composes leaf_world = local * inst_root_world, and
 * emits one SceneInstanceBatch per (PI, proto sub-mesh). Consumes and resets
 * the pending list so the next root starts clean. */
static void scene_drain_nested_pis(Scene* scene, int mesh_count,
                                   uint64_t* pi_xform_cap, int* pi_batch_cap)
{
    if (!scene || !pi_xform_cap || !pi_batch_cap) return;
    int diag = getenv("NUSD_GEO_DIAG") != NULL;

    for (int pe = 0; pe < s_nested_pending_count; pe++) {
        NestedPiPending* e = &s_nested_pending[pe];
        for (int g = 0; g < e->ngroups; g++) {
            NestedPiProtoGroup* grp = &e->groups[g];
            if (!grp->local || grp->count == 0 || !grp->proto_concrete[0])
                continue;

            /* Pack this proto's leaf transforms into ONE shared slab. A
             * prototype is typically a multi-mesh Xform; every sub-mesh is
             * instanced at the SAME leaf positions, so all sub-mesh batches
             * must reference this single slab rather than each duplicating the
             * (up to 21M) transforms — that distinction is the difference
             * between ~1.3 GiB and ~9.7 GiB for Moana. */
            uint32_t base_off = (uint32_t)scene->npi_transforms;
            SceneInstanceTransform* slab = scene_pi_transforms_reserve(
                scene, pi_xform_cap, (uint64_t)grp->count);
            if (!slab) {
                scene_fail(scene, "OOM growing PI slab for nested PI '%s'",
                           e->pi_concrete);
                return;
            }
            for (uint32_t i = 0; i < grp->count; i++) {
                double localm[16], leafm[16];
                compact12_to_affine16(grp->local[i].m, localm);
                nanousd_mul_m4d(localm, e->inst_root_world, leafm);
                affine16_to_12f(leafm, slab[i].m);
            }
            s_nested_pi_emitted_leaves += grp->count;

            /* Resolve proto concrete path -> sub-mesh indices (exact or
             * prefix with a '/' boundary; a proto Xform expands to every child
             * Mesh). mesh_count is the live mesh_idx — scene->nmeshes is not
             * set until the very end of scene_load. Each matching sub-mesh
             * gets a batch that SHARES base_off/count. */
            for (int sm = 0; sm < mesh_count; sm++) {
                SceneMesh* proto_m = &scene->meshes[sm];
                if (!proto_m->path[0]) continue;
                if (!path_has_prefix(proto_m->path, grp->proto_concrete))
                    continue;
                if (!proto_m->positions || !proto_m->indices ||
                    proto_m->nvertices <= 0 || proto_m->nindices <= 0)
                    continue;
                if (!scene_pi_batches_reserve(scene, pi_batch_cap,
                                              scene->npi_batches + 1)) {
                    scene_fail(scene,
                               "OOM growing PI batch table for nested PI '%s'",
                               e->pi_concrete);
                    return;
                }
                SceneInstanceBatch* b =
                    &scene->pi_batches[scene->npi_batches++];
                b->prototype_mesh_idx     = sm;
                b->transform_offset       = base_off;
                b->transform_count        = grp->count;
                b->source_prim_idx        = -1;
                b->material_or_binding_id = -1;
                b->source_kind            = SCENE_INSTANCE_SOURCE_POINT_INSTANCER;
                /* Mark the proto mesh as proto-only so it is not also drawn
                 * as a static pile at the asset origin. */
                proto_m->is_proto_only = 1;
                s_nested_pi_emitted_batches++;
            }
        }
    }
    if (diag && s_nested_pending_count > 0) {
        fprintf(stderr,
                "scene_load:   NESTED-PI-DRAIN pending=%d emitted_leaves=%lld "
                "emitted_batches=%d npi_transforms=%llu\n",
                s_nested_pending_count, s_nested_pi_emitted_leaves,
                s_nested_pi_emitted_batches,
                (unsigned long long)scene->npi_transforms);
    }
    nested_pending_reset();
}

static int seed_asset_composition_arcs(NanousdStage owner_stage,
                                       NanousdPrim inst_prim,
                                       NanousdPrim owner_prim,
                                       const char* inst_root_path,
                                       const double inst_root_world[16],
                                       InstanceChildWork** stack,
                                       int* stack_size,
                                       int* stack_cap,
                                       NativeAssetStageList* owned_stages,
                                       int depth);

static int seed_asset_arc_item(NanousdStage owner_stage,
                               NanousdPrim inst_prim,
                               const char* owner_prim_path,
                               const char* field,
                               const char* item,
                               const char* inst_root_path,
                               const double inst_root_world[16],
                               InstanceChildWork** stack,
                               int* stack_size,
                               int* stack_cap,
                               NativeAssetStageList* owned_stages,
                               int depth)
{
    if (!owner_stage || !item || !stack || depth > 4)
        return 0;
    int diag = scene_native_replay_arc_diag_enabled();
    char asset[1024];
    char target[1024];
    if (!parse_asset_arc_item(item, asset, sizeof(asset),
                              target, sizeof(target)))
        return 0;
    if (scene_skip_arc_for_geometry_only(asset, target)) {
        if (diag) {
            fprintf(stderr,
                    "scene_load: native asset arc skip material depth=%d "
                    "field=%s asset=%s target=%s\n",
                    depth, field ? field : "?",
                    asset, target[0] ? target : "<default>");
        }
        return 0;
    }

    char anchor[2048];
    if (!find_authored_arc_anchor(owner_stage, owner_prim_path, field,
                                  anchor, sizeof(anchor))) {
        const char* root_layer = nanousd_stage_get_root_layer_path(owner_stage);
        if (!root_layer || !root_layer[0])
            return 0;
        snprintf(anchor, sizeof(anchor), "%s", root_layer);
    }

    char resolved[2048];
    if (!nanousd_resolve_asset_path(anchor, asset,
                                    resolved, sizeof(resolved)))
        return 0;
    if (diag) {
        fprintf(stderr,
                "scene_load: native asset arc depth=%d field=%s "
                "asset=%s target=%s resolved=%s\n",
                depth, field ? field : "?",
                asset, target[0] ? target : "<default>", resolved);
    }
    NanousdStage substage = nanousd_open(resolved);
    if (!substage || !nanousd_isvalid(substage)) {
        if (substage) nanousd_close(substage);
        return 0;
    }
    if (!native_asset_stage_list_add(owned_stages, substage)) {
        nanousd_close(substage);
        return 0;
    }

    apply_instance_variants_to_asset_target(inst_prim, substage, target);
    return seed_asset_stage_meshes(substage, inst_prim, target,
                                  inst_root_path, inst_root_world,
                                  stack, stack_size, stack_cap,
                                  owned_stages, depth + 1);
}

static int seed_asset_composition_arcs(NanousdStage owner_stage,
                                       NanousdPrim inst_prim,
                                       NanousdPrim owner_prim,
                                       const char* inst_root_path,
                                       const double inst_root_world[16],
                                       InstanceChildWork** stack,
                                       int* stack_size,
                                       int* stack_cap,
                                       NativeAssetStageList* owned_stages,
                                       int depth)
{
    (void)owner_stage;
    if (!owner_prim || !inst_root_path || !inst_root_world || !stack ||
        !stack_size || !stack_cap || !owned_stages || depth > 4)
        return 0;

    int diag = scene_native_replay_arc_diag_enabled();
    int seeded = 0;
    int narcs = nanousd_ncomposition_arcs(owner_prim);
    for (int i = 0; i < narcs; i++) {
        NanousdCompositionArc arc;
        memset(&arc, 0, sizeof(arc));
        if (!nanousd_composition_arc(owner_prim, i, &arc))
            continue;
        if (arc.arc_type != NANOUSD_ARC_REFERENCE &&
            arc.arc_type != NANOUSD_ARC_PAYLOAD)
            continue;
        if ((arc.flags & NANOUSD_COMPOSITION_ARC_DIRECT) == 0)
            continue;
        if (!arc.layer_path || !arc.layer_path[0] ||
            !arc.source_path || !arc.source_path[0])
            continue;

        if (scene_skip_arc_for_geometry_only(arc.layer_path,
                                             arc.source_path)) {
            if (diag) {
                fprintf(stderr,
                        "scene_load: native composition arc skip material "
                        "depth=%d source=%s layer=%s\n",
                        depth, arc.source_path, arc.layer_path);
            }
            continue;
        }
        if (diag) {
            fprintf(stderr,
                    "scene_load: native composition arc depth=%d "
                    "type=%d source=%s layer=%s flags=%u\n",
                    depth, arc.arc_type, arc.source_path,
                    arc.layer_path, arc.flags);
        }
        NanousdStage substage = nanousd_open(arc.layer_path);
        if (!substage || !nanousd_isvalid(substage)) {
            if (substage) nanousd_close(substage);
            continue;
        }
        if (!native_asset_stage_list_add(owned_stages, substage)) {
            nanousd_close(substage);
            continue;
        }

        apply_instance_variants_to_asset_target(inst_prim, substage,
                                                arc.source_path);
        seeded += seed_asset_stage_meshes(substage, inst_prim,
                                          arc.source_path,
                                          inst_root_path,
                                          inst_root_world,
                                          stack, stack_size, stack_cap,
                                          owned_stages, depth + 1);
    }
    return seeded;
}

static int seed_asset_stage_meshes(NanousdStage asset_stage,
                                  NanousdPrim inst_prim,
                                  const char* target_path,
                                  const char* inst_root_path,
                                  const double inst_root_world[16],
                                  InstanceChildWork** stack,
                                  int* stack_size,
                                  int* stack_cap,
                                  NativeAssetStageList* owned_stages,
                                  int depth)
{
    if (!asset_stage || !inst_root_path || !inst_root_world ||
        !stack || !stack_size || !stack_cap || depth > 4)
        return 0;
    int diag = scene_native_replay_diag_enabled();
    double diag_t0 = diag ? get_time_sec() : 0.0;

    NanousdPrim target = NULL;
    if (target_path && target_path[0])
        target = nanousd_primpath(asset_stage, target_path);
    if (!target)
        target = nanousd_defaultprim(asset_stage);
    if (!target)
        return 0;

    const char* actual_target_path_raw = nanousd_path(target);
    if (!actual_target_path_raw || !actual_target_path_raw[0]) {
        nanousd_freeprim(target);
        return 0;
    }
    char actual_target_path[1024];
    snprintf(actual_target_path, sizeof(actual_target_path), "%s",
             actual_target_path_raw);

    double target_world[16];
    double target_inv[16];
    compute_world_xform(target, target_world);
    int have_target_inv = invert_affine_m4d_rowvec(target_world, target_inv);
    nanousd_freeprim(target);
    if (!have_target_inv)
        return 0;

    int seeded = 0;
    size_t target_len = strlen(actual_target_path);
    int nflat = nanousd_traverse_flat(asset_stage, NULL, 0);
    NanousdFlatPrim* flat = NULL;
    if (nflat > 0) {
        flat = (NanousdFlatPrim*)calloc((size_t)nflat, sizeof(NanousdFlatPrim));
        if (flat) {
            int total = nanousd_traverse_flat(asset_stage, flat, nflat);
            if (total > 0 && total < nflat) nflat = total;
        } else {
            nflat = 0;
        }
    }

    int nscan = flat ? nflat : nanousd_nprims(asset_stage);
    for (int i = 0; i < nscan; i++) {
        const char* path = NULL;
        const char* tn = NULL;
        NanousdPrim prim = NULL;
        if (flat) {
            path = flat[i].path;
            tn = flat[i].type_name;
            if (!path) continue;
        } else {
            prim = nanousd_prim(asset_stage, i);
            if (!prim) continue;
            path = nanousd_path(prim);
            tn = nanousd_typename(prim);
        }
        if (!path_has_prefix(path, actual_target_path)) {
            if (prim) nanousd_freeprim(prim);
            continue;
        }
        if (tn && strcmp(tn, "PointInstancer") == 0) {
            nested_pi_record(asset_stage, path, actual_target_path,
                             inst_root_path, target_inv, inst_root_world);
            if (getenv("NUSD_GEO_DIAG"))
                fprintf(stderr, "scene_load:   ASSET NESTED PI recorded %s\n", path);
            if (prim) nanousd_freeprim(prim);
            continue;
        }
        if (tn && strcmp(tn, "Mesh") == 0) {
            if (!prim)
                prim = nanousd_primpath(asset_stage, path);
            if (!prim)
                continue;
            const char* prim_path = nanousd_path(prim);
            if (prim_path && prim_path[0])
                path = prim_path;
            const char* rel = path + target_len;
            while (*rel == '/') rel++;
            char concrete_path[512];
            if (rel[0]) {
                snprintf(concrete_path, sizeof(concrete_path),
                         "%s/%s", inst_root_path, rel);
            } else {
                snprintf(concrete_path, sizeof(concrete_path),
                         "%s", inst_root_path);
            }
            double raw_world[16];
            double rel_xform[16];
            double world[16];
            compute_world_xform(prim, raw_world);
            nanousd_mul_m4d(raw_world, target_inv, rel_xform);
            nanousd_mul_m4d(rel_xform, inst_root_world, world);
            if (!instance_child_work_push_override(stack, stack_size,
                                                   stack_cap, prim,
                                                   asset_stage,
                                                   concrete_path, rel,
                                                   world)) {
                nanousd_freeprim(prim);
                continue;
            }
            seeded++;
            continue;
        }
        if (prim) nanousd_freeprim(prim);
    }

    if (seeded > 0) {
        if (diag) {
            fprintf(stderr,
                    "scene_load: native asset stage seeded depth=%d "
                    "target=%s flat=%d meshes=%d time=%.1f ms "
                    "maxrss=%.1f MiB\n",
                    depth, actual_target_path, nflat, seeded,
                    (get_time_sec() - diag_t0) * 1000.0,
                    (double)scene_maxrss_bytes() / (1024.0 * 1024.0));
        }
        free(flat);
        return seeded;
    }

    for (int i = 0; i < nscan; i++) {
        const char* path = NULL;
        NanousdPrim prim = NULL;
        if (flat) {
            path = flat[i].path;
            if (!path) continue;
            if (!path_has_prefix(path, actual_target_path))
                continue;
            prim = nanousd_primpath(asset_stage, path);
        } else {
            prim = nanousd_prim(asset_stage, i);
        }
        if (!prim) continue;
        path = nanousd_path(prim);
        if (!path_has_prefix(path, actual_target_path)) {
            nanousd_freeprim(prim);
            continue;
        }
        if (depth < 4) {
            seeded += seed_asset_composition_arcs(asset_stage, inst_prim,
                                                  prim, inst_root_path,
                                                  inst_root_world,
                                                  stack, stack_size,
                                                  stack_cap,
                                                  owned_stages, depth);
            for (int f = 0; f < 2; f++) {
                const char* field = f == 0 ? "payload" : "references";
                NanousdListOp op = nanousd_prim_listop(prim, field);
                if (!op) continue;
                int nitems = nanousd_listop_nitems(op);
                for (int j = 0; j < nitems; j++) {
                    const char* item = nanousd_listop_item(op, j);
                    seeded += seed_asset_arc_item(asset_stage, inst_prim,
                                                 path, field, item,
                                                 inst_root_path,
                                                 inst_root_world,
                                                 stack, stack_size,
                                                 stack_cap,
                                                 owned_stages, depth);
                }
                nanousd_listop_free(op);
            }
        }
        nanousd_freeprim(prim);
    }
    free(flat);
    if (diag) {
        fprintf(stderr,
                "scene_load: native asset stage fallback depth=%d "
                "target=%s flat=%d seeded=%d time=%.1f ms "
                "maxrss=%.1f MiB\n",
                depth, actual_target_path, nflat, seeded,
                (get_time_sec() - diag_t0) * 1000.0,
                (double)scene_maxrss_bytes() / (1024.0 * 1024.0));
    }
    return seeded;
}

static int seed_native_instance_stack_from_asset_arcs(NanousdStage root_stage,
                                                     NanousdPrim inst_prim,
                                                     const char* inst_root_path,
                                                     const double inst_root_world[16],
                                                     InstanceChildWork** stack,
                                                     int* stack_size,
                                                     int* stack_cap,
                                                     NativeAssetStageList* owned_stages)
{
    if (!root_stage || !inst_prim || !inst_root_path || !inst_root_world)
        return 0;
    int seeded = seed_asset_composition_arcs(root_stage, inst_prim, inst_prim,
                                             inst_root_path, inst_root_world,
                                             stack, stack_size, stack_cap,
                                             owned_stages, 0);
    if (seeded > 0)
        return seeded;

    for (int f = 0; f < 2; f++) {
        const char* field = f == 0 ? "payload" : "references";
        NanousdListOp op = nanousd_prim_listop(inst_prim, field);
        if (!op) continue;
        int nitems = nanousd_listop_nitems(op);
        for (int i = 0; i < nitems; i++) {
            const char* item = nanousd_listop_item(op, i);
            seeded += seed_asset_arc_item(root_stage, inst_prim,
                                         inst_root_path, field, item,
                                         inst_root_path, inst_root_world,
                                         stack, stack_size, stack_cap,
                                         owned_stages, 0);
        }
        nanousd_listop_free(op);
    }
    return seeded;
}

static void scene_mesh_bounds_from_positions(Scene* scene, SceneMesh* mesh)
{
    if (!scene || !mesh) return;
    mesh->bounds_min[0] = FLT_MAX;
    mesh->bounds_min[1] = FLT_MAX;
    mesh->bounds_min[2] = FLT_MAX;
    mesh->bounds_max[0] = -FLT_MAX;
    mesh->bounds_max[1] = -FLT_MAX;
    mesh->bounds_max[2] = -FLT_MAX;
    for (int v = 0; v < mesh->nvertices; v++) {
        float wp[3];
        xform_point(mesh->world_xform, &mesh->positions[v * 3], wp);
        for (int k = 0; k < 3; k++) {
            if (wp[k] < mesh->bounds_min[k]) mesh->bounds_min[k] = wp[k];
            if (wp[k] > mesh->bounds_max[k]) mesh->bounds_max[k] = wp[k];
            if (wp[k] < scene->bounds_min[k]) scene->bounds_min[k] = wp[k];
            if (wp[k] > scene->bounds_max[k]) scene->bounds_max[k] = wp[k];
        }
    }
}

static int emit_native_replay_catalog(Scene* scene,
                                      int* max_meshes,
                                      int mesh_limit,
                                      int* mesh_idx,
                                      int* instanced_meshes,
                                      int* total_verts,
                                      int* total_indices,
                                      const char* inst_root_path,
                                      const double inst_root_world[16],
                                      const NativePrototypeReplayCatalog* cat)
{
    if (!scene || !max_meshes || !mesh_idx || !cat || !inst_root_world)
        return 0;
    int emitted = 0;
    for (int i = 0; i < cat->count; i++) {
        const NativePrototypeReplayEntry* e = &cat->entries[i];
        int src_idx = e->prototype_idx;
        if (src_idx < 0 || src_idx >= *mesh_idx)
            continue;
        SceneMesh* proto_m = &scene->meshes[src_idx];
        if (!proto_m->positions || !proto_m->indices ||
            proto_m->nvertices <= 0 || proto_m->nindices <= 0)
            continue;
        if (!scene_mesh_reserve(scene, max_meshes, *mesh_idx + 1,
                                mesh_limit)) {
            return -1;
        }
        SceneMesh* mesh = &scene->meshes[*mesh_idx];
        memset(mesh, 0, sizeof(*mesh));
        mesh->positions = proto_m->positions;
        mesh->normals = proto_m->normals;
        mesh->indices = proto_m->indices;
        mesh->nvertices = proto_m->nvertices;
        mesh->nindices = proto_m->nindices;
        mesh->prototype_idx = src_idx;
        mesh->has_display_color = proto_m->has_display_color;
        mesh->display_color[0] = proto_m->display_color[0];
        mesh->display_color[1] = proto_m->display_color[1];
        mesh->display_color[2] = proto_m->display_color[2];
        mesh->colors = proto_m->colors;
        mesh->texcoords = proto_m->texcoords;
        mesh->material_index = proto_m->material_index;
        mesh->is_proto_only = 0;
        nanousd_mul_m4d(e->relative_xform, inst_root_world, mesh->world_xform);
        if (inst_root_path && inst_root_path[0]) {
            if (e->relative_path[0]) {
                snprintf(mesh->path, sizeof(mesh->path), "%s/%s",
                         inst_root_path, e->relative_path);
            } else {
                snprintf(mesh->path, sizeof(mesh->path), "%s", inst_root_path);
            }
        } else {
            set_mesh_path(mesh, proto_m->path);
        }
        scene_mesh_bounds_from_positions(scene, mesh);
        mesh->vertex_offset = 0;
        mesh->index_offset = 0;
        if (total_verts) *total_verts += mesh->nvertices;
        if (total_indices) *total_indices += mesh->nindices;
        (*mesh_idx)++;
        if (instanced_meshes) (*instanced_meshes)++;
        emitted++;
    }
    return emitted;
}

static int path_is_native_usd_prototype(const char* path)
{
    if (!path) return 0;
    const char* keep = getenv("NUSD_RENDER_NATIVE_PROTOTYPES");
    if (keep && keep[0] && keep[0] != '0')
        return 0;
    return strncmp(path, "/__Prototype_", 13) == 0;
}

static NanousdPrim resolve_relative_child_prim(NanousdPrim root,
                                               const char* relative)
{
    if (!root || !relative) return NULL;
    while (*relative == '/') relative++;
    if (!*relative) return NULL;

    NanousdPrim current = root;
    int current_owned = 0;
    const char* p = relative;
    while (*p) {
        const char* slash = strchr(p, '/');
        size_t n = slash ? (size_t)(slash - p) : strlen(p);
        if (n == 0 || n >= 256) {
            if (current_owned) nanousd_freeprim(current);
            return NULL;
        }
        char name[256];
        memcpy(name, p, n);
        name[n] = '\0';

        NanousdPrim next = nanousd_childname(current, name);
        if (current_owned) nanousd_freeprim(current);
        if (!next) return NULL;
        current = next;
        current_owned = 1;
        p = slash ? slash + 1 : p + n;
    }
    return current_owned ? current : NULL;
}

static void path_prefix_list_add(PathPrefixList* list, const char* prefix)
{
    if (!list || !prefix || prefix[0] != '/') return;
    for (int i = 0; i < list->count; i++) {
        if (!strcmp(list->items[i], prefix)) return;
    }
    if (list->count >= list->cap) {
        int new_cap = list->cap ? list->cap * 2 : 64;
        char** new_items = (char**)realloc(list->items, (size_t)new_cap * sizeof(char*));
        if (!new_items) return;
        list->items = new_items;
        list->cap = new_cap;
    }
    size_t n = strlen(prefix);
    char* copy = (char*)malloc(n + 1);
    if (!copy) return;
    memcpy(copy, prefix, n + 1);
    list->items[list->count++] = copy;
}

static int path_prefix_list_contains(const PathPrefixList* list, const char* path)
{
    if (!list || !path) return 0;
    for (int i = 0; i < list->count; i++) {
        if (path_has_prefix(path, list->items[i])) return 1;
    }
    return 0;
}

static void path_prefix_list_free(PathPrefixList* list)
{
    if (!list) return;
    for (int i = 0; i < list->count; i++) free(list->items[i]);
    free(list->items);
    list->items = NULL;
    list->count = 0;
    list->cap = 0;
}

static void collect_point_instancer_proto_prefixes(NanousdStage stage,
                                                   int nprims,
                                                   PathPrefixList* out)
{
    if (!stage || !out) return;
    for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;
        if (!nanousd_isa(prim, "PointInstancer")) {
            nanousd_freeprim(prim);
            continue;
        }
        int n = nanousd_nreltargets(prim, "prototypes");
        for (int p = 0; p < n; p++) {
            const char* target = nanousd_reltarget(prim, "prototypes", p);
            path_prefix_list_add(out, target);
        }
        nanousd_freeprim(prim);
    }
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

/* Load one BasisCurves prim into a SceneCurve, with CVs baked to world
 * space. Returns 1 on success, 0 on skip/no-data. */
static int load_basiscurves(NanousdPrim prim, Arena* arena, SceneCurve* curve)
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

    /* (d) widths (fan to per-CV). Default radius 0.05 m matches Storm. */
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

    /* (f) world transform. */
    compute_world_xform(prim, curve->world_xform);

    /* (g) Bake CVs to world space (renderer is static-scene-focused; the
     * MVP avoids per-curve TLAS instance transforms). Compute world bounds
     * during bake. */
    float* world_cvs = (float*)arena_alloc(arena, (size_t)nv * 3 * sizeof(float), 16);
    if (!world_cvs) return 0;
    curve->bounds_min[0] = curve->bounds_min[1] = curve->bounds_min[2] =  FLT_MAX;
    curve->bounds_max[0] = curve->bounds_max[1] = curve->bounds_max[2] = -FLT_MAX;
    for (int v = 0; v < nv; v++) {
        xform_point(curve->world_xform, &pts[v*3], &world_cvs[v*3]);
        /* Expand bounds by per-CV width (radius) so AABB encloses tube. */
        float r = widths[v] * 0.5f;
        for (int k = 0; k < 3; k++) {
            float lo = world_cvs[v*3+k] - r, hi = world_cvs[v*3+k] + r;
            if (lo < curve->bounds_min[k]) curve->bounds_min[k] = lo;
            if (hi > curve->bounds_max[k]) curve->bounds_max[k] = hi;
        }
    }

    /* (h) Persist into SceneCurve. patch_indices is cubic-tessellation
     * machinery the gles raster path needs; the RT path generates
     * per-segment AABBs directly at GPU upload time, so we don't carry
     * patch_indices here. */
    int* cvc_copy = (int*)arena_alloc(arena, (size_t)cvc_count * sizeof(int), 16);
    if (!cvc_copy) return 0;
    memcpy(cvc_copy, cvc, (size_t)cvc_count * sizeof(int));

    curve->cvs                  = world_cvs;
    curve->widths               = widths;
    curve->colors               = colors;
    curve->nv                   = nv;
    curve->ncurves_in_prim      = cvc_count;
    curve->curve_vertex_counts  = cvc_copy;
    curve->basis                = basis;
    curve->wrap                 = wrap;
    curve->type_is_cubic        = type_is_cubic;
    curve->has_display_color    = has_dc;
    curve->material_index       = -1;
    return 1;
}

/* ----------------------------------------------------------------
 * Per-segment AABB extraction.
 *
 * For type=linear: each consecutive CV pair within a sub-curve emits
 * one cylinder/capsule segment. AABB is the axis-aligned bound of the
 * two endpoint spheres (radius = width/2).
 *
 * For type=cubic (Phase 11.B port from Vulkan scene.c): the cubic
 * patch is tessellated into N_CUBIC_SUBSEGS straight sub-segments
 * using either Catmull-Rom or Bezier evaluation. Each sub-segment
 * gets the same sphere-sphere AABB treatment. 8 sub-segs is a visual
 * sweet spot for typical S-curves used as flex hoses.
 * ---------------------------------------------------------------- */

/* Phase 11.B: cubic catmullRom curves tessellate into N_CUBIC_SUBSEGS
 * linear sub-segments per patch. 8 sub-segs is a visual sweet spot for
 * the typical S-curves used as flex hoses; below 4 the cubic look is
 * lost, above 16 the segment budget explodes without visible gain. */
#define N_CUBIC_SUBSEGS 8

/* Number of patches in a cubic sub-curve under each (basis, wrap) combo.
 * - catmullRom nonperiodic: each patch needs CVs [i, i+1, i+2, i+3] and
 *   spans CV[i+1]..CV[i+2], so N CVs → max(0, N-3) patches.
 * - catmullRom periodic: every CV starts a patch (wraps around) → N patches.
 * - bezier nonperiodic: 4-CV-per-patch, vstep=3 → (N-1)/3 patches.
 * - bspline / unknown: treated like catmullRom for now. */
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

/* Catmull-Rom evaluation at parameter t∈[0,1] for the patch
 * P[i+1]..P[i+2] using P[i] and P[i+3] as in/out tangent controls. */
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

/* Tessellate one cubic patch into N_CUBIC_SUBSEGS linear segments
 * (with their AABBs). CV indices are passed in `idx`[0..3] so the
 * same loop body handles periodic wrap. Widths linearly interpolated
 * between the two endpoint CVs of the patch (P1..P2 for catmullRom,
 * P0..P3 for bezier). The CVs are already baked to world space at
 * load_basiscurves() time, so the tessellated points inherit the
 * world transform without an extra xform_point() pass.
 *
 * Returns N_CUBIC_SUBSEGS. */
static int emit_cubic_patch(const SceneCurve* curve,
                            const int idx[4],
                            SceneCurveSegment* out_segments,
                            SceneCurveAabb*    out_aabbs,
                            float*             out_colors)
{
    const float* cvs = curve->cvs;
    const float* P0 = &cvs[idx[0] * 3];
    const float* P1 = &cvs[idx[1] * 3];
    const float* P2 = &cvs[idx[2] * 3];
    const float* P3 = &cvs[idx[3] * 3];

    /* Width along the patch: linearly interpolate between the two
     * endpoint widths (CV[1] and CV[2] for catmullRom, P0..P3 ends for
     * bezier). Cheap and visually fine for typical hoses. */
    float w_start = curve->widths[idx[1]];
    float w_end   = curve->widths[idx[2]];
    if (curve->basis == CURVE_BASIS_BEZIER) {
        w_start = curve->widths[idx[0]];
        w_end   = curve->widths[idx[3]];
    }

    float color_start[3];
    float color_end[3];
    if (curve->colors) {
        int ci0 = (curve->basis == CURVE_BASIS_BEZIER) ? idx[0] : idx[1];
        int ci1 = (curve->basis == CURVE_BASIS_BEZIER) ? idx[3] : idx[2];
        color_start[0] = curve->colors[ci0 * 3 + 0];
        color_start[1] = curve->colors[ci0 * 3 + 1];
        color_start[2] = curve->colors[ci0 * 3 + 2];
        color_end[0] = curve->colors[ci1 * 3 + 0];
        color_end[1] = curve->colors[ci1 * 3 + 1];
        color_end[2] = curve->colors[ci1 * 3 + 2];
    } else if (curve->has_display_color) {
        color_start[0] = color_end[0] = curve->display_color[0];
        color_start[1] = color_end[1] = curve->display_color[1];
        color_start[2] = color_end[2] = curve->display_color[2];
    } else {
        color_start[0] = color_end[0] = 0.7f;
        color_start[1] = color_end[1] = 0.7f;
        color_start[2] = color_end[2] = 0.7f;
    }

    /* Evaluate the curve at N+1 points (already in world space because
     * curve->cvs are baked). Then emit N segments + AABBs. */
    float pts[N_CUBIC_SUBSEGS + 1][3];
    for (int s = 0; s <= N_CUBIC_SUBSEGS; s++) {
        float t = (float)s / (float)N_CUBIC_SUBSEGS;
        if (curve->basis == CURVE_BASIS_BEZIER) {
            bezier_eval(P0, P1, P2, P3, t, pts[s]);
        } else {
            catmull_rom_eval(P0, P1, P2, P3, t, pts[s]);
        }
    }

    for (int s = 0; s < N_CUBIC_SUBSEGS; s++) {
        float t0 = (float)s         / (float)N_CUBIC_SUBSEGS;
        float t1 = (float)(s + 1)   / (float)N_CUBIC_SUBSEGS;
        float r0 = 0.5f * (w_start + (w_end - w_start) * t0);
        float r1 = 0.5f * (w_start + (w_end - w_start) * t1);
        const float* a = pts[s];
        const float* b = pts[s + 1];

        SceneCurveSegment* seg = &out_segments[s];
        seg->p0[0] = a[0]; seg->p0[1] = a[1]; seg->p0[2] = a[2];
        seg->r0   = r0;
        seg->p1[0] = b[0]; seg->p1[1] = b[1]; seg->p1[2] = b[2];
        seg->r1   = r1;

        SceneCurveAabb* aabb = &out_aabbs[s];
        aabb->min_x = (a[0] - r0 < b[0] - r1) ? a[0] - r0 : b[0] - r1;
        aabb->min_y = (a[1] - r0 < b[1] - r1) ? a[1] - r0 : b[1] - r1;
        aabb->min_z = (a[2] - r0 < b[2] - r1) ? a[2] - r0 : b[2] - r1;
        aabb->max_x = (a[0] + r0 > b[0] + r1) ? a[0] + r0 : b[0] + r1;
        aabb->max_y = (a[1] + r0 > b[1] + r1) ? a[1] + r0 : b[1] + r1;
        aabb->max_z = (a[2] + r0 > b[2] + r1) ? a[2] + r0 : b[2] + r1;

        if (out_colors) {
            float tm = 0.5f * (t0 + t1);
            out_colors[s * 3 + 0] = color_start[0] + (color_end[0] - color_start[0]) * tm;
            out_colors[s * 3 + 1] = color_start[1] + (color_end[1] - color_start[1]) * tm;
            out_colors[s * 3 + 2] = color_start[2] + (color_end[2] - color_start[2]) * tm;
        }
    }
    return N_CUBIC_SUBSEGS;
}

int scene_curve_count_segments(const SceneCurve* curve)
{
    if (!curve) return 0;
    if (curve->type_is_cubic) {
        int total = 0;
        for (int c = 0; c < curve->ncurves_in_prim; c++) {
            int n = curve->curve_vertex_counts[c];
            int p = cubic_patches(n, curve->basis, curve->wrap);
            total += p * N_CUBIC_SUBSEGS;
        }
        return total;
    }
    int total = 0;
    for (int c = 0; c < curve->ncurves_in_prim; c++) {
        int n = curve->curve_vertex_counts[c];
        if (n >= 2) total += (n - 1);
        if (curve->wrap == CURVE_WRAP_PERIODIC && n >= 2) total += 1;
    }
    return total;
}

int scene_curve_to_segments(const SceneCurve* curve,
                            SceneCurveSegment* out_segments,
                            SceneCurveAabb*    out_aabbs)
{
    return scene_curve_to_segments_colored(curve, out_segments, out_aabbs, NULL);
}

int scene_curve_to_segments_colored(const SceneCurve* curve,
                                    SceneCurveSegment* out_segments,
                                    SceneCurveAabb*    out_aabbs,
                                    float*             out_colors)
{
    if (!curve) return 0;
    if (!out_segments || !out_aabbs) return 0;

    if (curve->type_is_cubic) {
        int written = 0;
        int v_off = 0;
        int vstep = (curve->basis == CURVE_BASIS_BEZIER) ? 3 : 1;
        for (int c = 0; c < curve->ncurves_in_prim; c++) {
            int n = curve->curve_vertex_counts[c];
            int n_patches = cubic_patches(n, curve->basis, curve->wrap);
            for (int p = 0; p < n_patches; p++) {
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
                written += emit_cubic_patch(curve, idx,
                                            out_segments + written,
                                            out_aabbs    + written,
                                            out_colors ? out_colors + written * 3 : NULL);
            }
            v_off += n;
        }
        return written;
    }

    int written = 0;
    int v_off = 0;
    for (int c = 0; c < curve->ncurves_in_prim; c++) {
        int n = curve->curve_vertex_counts[c];
        if (n < 2) { v_off += n; continue; }

        int n_segs = n - 1 + (curve->wrap == CURVE_WRAP_PERIODIC ? 1 : 0);
        for (int s = 0; s < n_segs; s++) {
            int i0 = v_off + s;
            int i1 = (s == n - 1)            /* periodic wrap segment */
                     ? v_off                 /* close back to first CV */
                     : v_off + s + 1;
            const float* p0 = &curve->cvs[i0 * 3];
            const float* p1 = &curve->cvs[i1 * 3];
            float r0 = curve->widths[i0] * 0.5f;
            float r1 = curve->widths[i1] * 0.5f;

            SceneCurveSegment* seg = &out_segments[written];
            seg->p0[0] = p0[0]; seg->p0[1] = p0[1]; seg->p0[2] = p0[2];
            seg->r0   = r0;
            seg->p1[0] = p1[0]; seg->p1[1] = p1[1]; seg->p1[2] = p1[2];
            seg->r1   = r1;

            /* AABB = bound of two endpoint spheres. */
            SceneCurveAabb* a = &out_aabbs[written];
            a->min_x = (p0[0] - r0 < p1[0] - r1) ? p0[0] - r0 : p1[0] - r1;
            a->min_y = (p0[1] - r0 < p1[1] - r1) ? p0[1] - r0 : p1[1] - r1;
            a->min_z = (p0[2] - r0 < p1[2] - r1) ? p0[2] - r0 : p1[2] - r1;
            a->max_x = (p0[0] + r0 > p1[0] + r1) ? p0[0] + r0 : p1[0] + r1;
            a->max_y = (p0[1] + r0 > p1[1] + r1) ? p0[1] + r0 : p1[1] + r1;
            a->max_z = (p0[2] + r0 > p1[2] + r1) ? p0[2] + r0 : p1[2] + r1;

            if (out_colors) {
                float c0[3];
                float c1[3];
                if (curve->colors) {
                    c0[0] = curve->colors[i0 * 3 + 0];
                    c0[1] = curve->colors[i0 * 3 + 1];
                    c0[2] = curve->colors[i0 * 3 + 2];
                    c1[0] = curve->colors[i1 * 3 + 0];
                    c1[1] = curve->colors[i1 * 3 + 1];
                    c1[2] = curve->colors[i1 * 3 + 2];
                } else if (curve->has_display_color) {
                    c0[0] = c1[0] = curve->display_color[0];
                    c0[1] = c1[1] = curve->display_color[1];
                    c0[2] = c1[2] = curve->display_color[2];
                } else {
                    c0[0] = c1[0] = 0.7f;
                    c0[1] = c1[1] = 0.7f;
                    c0[2] = c1[2] = 0.7f;
                }
                out_colors[written * 3 + 0] = 0.5f * (c0[0] + c1[0]);
                out_colors[written * 3 + 1] = 0.5f * (c0[1] + c1[1]);
                out_colors[written * 3 + 2] = 0.5f * (c0[2] + c1[2]);
            }

            written++;
        }
        v_off += n;
    }
    return written;
}

/* Iterate stage and load all BasisCurves into scene->curves. */
static int scene_load_curves(NanousdStage stage, Scene* scene, Arena* arena)
{
    int nprims = nanousd_nprims(stage);

    /* Pass 1: count BasisCurves prims for allocation. */
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

    scene->curves = (SceneCurve*)arena_calloc(arena, (size_t)max_curves, sizeof(SceneCurve));
    if (!scene->curves) return -1;

    /* Pass 2: load each prim. */
    int curve_idx = 0;
    int n_cubic_tessellated = 0;
    int cull_curves = scene_curve_frustum_cull_active();
    float cull_pad = scene_point_instance_cull_pad();
    int n_curve_extent_culled = 0;
    int n_curve_bounds_culled = 0;
    for (int i = 0; i < nprims; i++) {
        NanousdPrim p = nanousd_prim(stage, i);
        if (!p) continue;
        if (!nanousd_isactive(p)) { nanousd_freeprim(p); continue; }
        const char* tn = nanousd_typename(p);
        if (!tn || strcmp(tn, "BasisCurves") != 0) {
            nanousd_freeprim(p);
            continue;
        }
        if (cull_curves) {
            int ext_count = 0;
            const float* ext = nanousd_arraydataf(p, "extent", &ext_count);
            if (ext && ext_count == 6) {
                float local_min[3] = { ext[0], ext[1], ext[2] };
                float local_max[3] = { ext[3], ext[4], ext[5] };
                float world_min[3], world_max[3];
                double world[16];
                compute_world_xform(p, world);
                world_bounds_from_local(world, local_min, local_max,
                                        world_min, world_max);
                if (bounds_is_usable(world_min, world_max) &&
                    !scene_aabb_visible_in_any_frustum_padded(world_min,
                                                              world_max,
                                                              cull_pad)) {
                    n_curve_extent_culled++;
                    nanousd_freeprim(p);
                    continue;
                }
            }
        }
        SceneCurve* curve = &scene->curves[curve_idx];
        if (load_basiscurves(p, arena, curve)) {
            if (cull_curves &&
                !scene_aabb_visible_in_any_frustum_padded(curve->bounds_min,
                                                          curve->bounds_max,
                                                          cull_pad)) {
                n_curve_bounds_culled++;
                memset(curve, 0, sizeof(*curve));
                nanousd_freeprim(p);
                continue;
            }
            /* Phase 11.B: cubic curves tessellate to N_CUBIC_SUBSEGS linear
             * sub-segments per patch via Catmull-Rom or Bezier evaluation
             * (see scene_curve_to_segments). Pre-port they were skipped at
             * BLAS-build time and silently disappeared. */
            if (curve->type_is_cubic) n_cubic_tessellated++;
            for (int k = 0; k < 3; k++) {
                if (curve->bounds_min[k] < scene->bounds_min[k]) scene->bounds_min[k] = curve->bounds_min[k];
                if (curve->bounds_max[k] > scene->bounds_max[k]) scene->bounds_max[k] = curve->bounds_max[k];
            }
            curve_idx++;
        }
        nanousd_freeprim(p);
    }
    scene->ncurves = curve_idx;
    if (n_cubic_tessellated > 0) {
        fprintf(stderr, "scene_load: %d cubic BasisCurves tessellated to %d linear sub-segs per patch\n",
                n_cubic_tessellated, N_CUBIC_SUBSEGS);
    }
    if (curve_idx > 0) {
        fprintf(stderr, "scene_load: loaded %d BasisCurves prims\n", curve_idx);
    }
    if (cull_curves && (n_curve_extent_culled > 0 || n_curve_bounds_culled > 0)) {
        fprintf(stderr,
                "scene_load: BasisCurves frustum cull skipped %d by extent, "
                "%d after load (%d cameras, pad=%.3f)\n",
                n_curve_extent_culled, n_curve_bounds_culled,
                s_pi_frustum_count, cull_pad);
    }
    return curve_idx;
}

/* ----------------------------------------------------------------
 * scene_load — open a USD file and build a Scene that owns the stage.
 * ---------------------------------------------------------------- */
Scene* scene_load(const char* filepath)
{
    scene_clear_last_error();
    NanousdStage stage = nanousd_open(filepath);
    if (!nanousd_isvalid(stage)) {
        const char* err = nanousd_error(stage);
        fprintf(stderr, "scene_load: failed to open '%s': %s\n",
                filepath, err ? err : "unknown error");
        scene_set_last_errorf("failed to open '%s': %s",
                              filepath ? filepath : "<null>",
                              err ? err : "unknown error");
        if (stage) nanousd_close(stage);
        return NULL;
    }
    Scene* scene = scene_load_from_stage((void*)stage, filepath);
    if (!scene) {
        nanousd_close(stage);
        return NULL;
    }
    /* Transfer ownership: scene_free will close the stage. */
    scene->_owns_stage = 1;
    return scene;
}

/* ----------------------------------------------------------------
 * scene_load_from_stage — build a Scene over a borrowed NanousdStage.
 * The caller keeps ownership; scene_free does NOT close the stage.
 * ---------------------------------------------------------------- */
Scene* scene_load_from_stage(void* stage_void, const char* stage_label)
{
    return scene_load_from_stage_filtered(stage_void, stage_label, NULL, 0);
}

/* Render UsdGeomPoints as one small axis-aligned box per point, sized by the
 * point's width (`widths`: per-point, uniform, or default 1.0) and centered at
 * the local point position. The boxes feed the same SceneMesh / RT-mesh path as
 * Mesh and the primitive shapes; the caller's after_mesh_attrs tail bakes the
 * prim's world transform, material binding, and (uniform) displayColor. A box
 * (vs a sphere) keeps the vertex count low; per-point displayColor is not yet
 * applied per-box (v1 uses the prim's uniform displayColor). Returns 1 on
 * success, 0 if there are no points or allocation fails. */
static int build_points_geom(NanousdPrim prim, Arena* arena, SceneMesh* m)
{
    int n3 = 0;
    const float* pts = nanousd_arraydataf(prim, "points", &n3);
    if (!pts || n3 < 3) return 0;
    int np = n3 / 3;
    /* Cap to keep memory bounded and 24*np within int range. ALab's vegetation
     * point sets are far below this; a hit is logged so truncation is never
     * silent. */
    const int NP_CAP = 2000000;
    int capped = 0;
    if (np > NP_CAP) { np = NP_CAP; capped = 1; }

    int nw = 0;
    const float* widths = nanousd_arraydataf(prim, "widths", &nw);

    /* Optional size multiplier (default 1.0). UsdGeomPoints widths can be tiny
     * relative to the scene; NUSD_POINTS_SCALE fattens the boxes for
     * visibility/diagnostics without altering authored data. */
    float pscale = 1.0f;
    { const char* e = getenv("NUSD_POINTS_SCALE");
      if (e && e[0]) { float v = (float)atof(e); if (v > 0.0f) pscale = v; } }

    const int NVP = 24, NIP = 36;
    size_t nv = (size_t)np * NVP;
    size_t ni = (size_t)np * NIP;
    m->nvertices = (int)nv;
    m->nindices  = (int)ni;
    m->positions = (float*)arena_alloc(arena, nv * 3 * sizeof(float), 16);
    m->normals   = (float*)arena_alloc(arena, nv * 3 * sizeof(float), 16);
    m->indices   = (uint32_t*)arena_alloc(arena, ni * sizeof(uint32_t), 16);
    if (!m->positions || !m->normals || !m->indices) return 0;

    static const float fn[6][3] = {
        { 1, 0, 0}, {-1, 0, 0}, {0,  1, 0}, {0, -1, 0}, {0, 0,  1}, {0, 0, -1} };
    static const float fc[6][4][3] = {
        {{ 1,-1,-1},{ 1, 1,-1},{ 1, 1, 1},{ 1,-1, 1}},
        {{-1, 1,-1},{-1,-1,-1},{-1,-1, 1},{-1, 1, 1}},
        {{ 1, 1,-1},{-1, 1,-1},{-1, 1, 1},{ 1, 1, 1}},
        {{-1,-1,-1},{ 1,-1,-1},{ 1,-1, 1},{-1,-1, 1}},
        {{-1,-1, 1},{ 1,-1, 1},{ 1, 1, 1},{-1, 1, 1}},
        {{ 1,-1,-1},{-1,-1,-1},{-1, 1,-1},{ 1, 1,-1}} };

    for (int p = 0; p < np; p++) {
        float cx = pts[p*3+0], cy = pts[p*3+1], cz = pts[p*3+2];
        float w = 1.0f;
        if (widths) { if (nw == np) w = widths[p]; else if (nw == 1) w = widths[0]; }
        float hw = w * 0.5f * pscale;
        if (hw < 1e-5f) hw = 1e-5f;
        size_t vbase = (size_t)p * NVP;
        size_t ibase = (size_t)p * NIP;
        int v = 0, idx = 0;
        for (int f = 0; f < 6; f++) {
            uint32_t base = (uint32_t)(vbase + v);
            for (int c = 0; c < 4; c++) {
                size_t o = (vbase + v) * 3;
                m->positions[o+0] = cx + fc[f][c][0]*hw;
                m->positions[o+1] = cy + fc[f][c][1]*hw;
                m->positions[o+2] = cz + fc[f][c][2]*hw;
                m->normals[o+0] = fn[f][0];
                m->normals[o+1] = fn[f][1];
                m->normals[o+2] = fn[f][2];
                v++;
            }
            m->indices[ibase + idx++] = base;
            m->indices[ibase + idx++] = base + 1;
            m->indices[ibase + idx++] = base + 2;
            m->indices[ibase + idx++] = base;
            m->indices[ibase + idx++] = base + 2;
            m->indices[ibase + idx++] = base + 3;
        }
    }
    if (capped || getenv("NUSD_LOAD_TIMING"))
        fprintf(stderr, "scene_load: UsdGeomPoints '%s' -> %d points%s (%zu verts)\n",
                nanousd_path(prim) ? nanousd_path(prim) : "?", np,
                capped ? " [CAPPED]" : "", nv);
    return 1;
}

Scene* scene_load_from_stage_filtered(void* stage_void,
                                      const char* stage_label,
                                      const unsigned char* wanted_prims,
                                      int nprims_in_bitmap)
{
    scene_clear_last_error();
    NanousdStage stage = (NanousdStage)stage_void;
    if (!nanousd_isvalid(stage)) {
        const char* err = nanousd_error(stage);
        fprintf(stderr, "scene_load_from_stage: invalid stage '%s': %s\n",
                stage_label ? stage_label : "<no label>",
                err ? err : "unknown error");
        scene_set_last_errorf("invalid stage '%s': %s",
                              stage_label ? stage_label : "<no label>",
                              err ? err : "unknown error");
        return NULL;
    }
    const char* filepath = stage_label ? stage_label : "<stage>";
    double t0 = get_time_sec();

    int nprims = nanousd_nprims(stage);
    int semantic_flat_records = 0;
    double semantic_flat_traversal_ms = 0.0;
    if (scene_no_cull_all_geometry_requested() ||
        scene_flat_native_instance_traversal_enabled()) {
        double flat_t0 = get_time_sec();
        semantic_flat_records = nanousd_traverse_flat(stage, NULL, 0);
        semantic_flat_traversal_ms = (get_time_sec() - flat_t0) * 1000.0;
    }
    if (scene_no_cull_all_geometry_requested() && wanted_prims) {
        fprintf(stderr,
                "[scene_load] NUSD_NO_CULL_ALL_GEOMETRY=1 — ignoring "
                "filtered prim bitmap and loading all prims\n");
        wanted_prims = NULL;
        nprims_in_bitmap = 0;
    }
    if (wanted_prims && nprims_in_bitmap > 0) {
        int wanted_count = 0;
        int upto = nprims_in_bitmap < nprims ? nprims_in_bitmap : nprims;
        for (int i = 0; i < upto; i++) {
            if (wanted_prims[i]) wanted_count++;
        }
        fprintf(stderr,
                "[scene_load] filter active: %d/%d prims wanted (%.1f%% cull rate)\n",
                wanted_count, upto,
                upto > 0 ? 100.0 * (double)(upto - wanted_count) / (double)upto : 0.0);
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

    /* Allocate Scene and Arena. The stage is BORROWED here —
     * scene_load() flips _owns_stage=1 after we return, so it gets closed
     * by scene_free; scene_load_from_stage callers manage stage lifetime
     * themselves. We never close `stage` from inside this function. */
    Scene* scene = (Scene*)calloc(1, sizeof(Scene));
    if (!scene) {
        scene_set_last_errorf("failed to allocate Scene");
        return NULL;
    }

    Arena* arena = (Arena*)malloc(sizeof(Arena));
    if (!arena) {
        scene_set_last_errorf("failed to allocate scene arena header");
        free(scene);
        return NULL;
    }
    /* Scale arena to scene complexity: ~4KB per prim is a reasonable heuristic
     * for mesh-heavy scenes (positions + normals + indices + colors).
     * Minimum 64MB, so small scenes don't under-allocate. */
    size_t arena_size = (size_t)nprims * 4096;
    if (arena_size < 64 * 1024 * 1024) arena_size = 64 * 1024 * 1024;
    *arena = arena_create(arena_size);
    if (!arena->head) {
        scene_set_last_errorf("failed to allocate scene arena (%zu bytes)",
                              arena_size);
        free(arena);
        free(scene);
        return NULL;
    }

    scene->_arena = arena;
    scene->_stage = stage;
    scene->_owns_stage = 0;  /* default: borrowed; scene_load() flips this. */

    const char* lazy_mesh_env = getenv("NUSD_LAZY_MESH");
    int lazy_mesh_requested =
        (lazy_mesh_env && lazy_mesh_env[0] && lazy_mesh_env[0] != '0') ? 1 : 0;

    /* --- Load materials from the stage --- */
    if (s_load_materials && !lazy_mesh_requested) {
        /* Compute scene directory for resolving relative texture paths */
        char dir_buf[1024];
        snprintf(dir_buf, sizeof(dir_buf), "%s", filepath);
        char* scene_dir = dirname(dir_buf);
        scene->materials = (void*)materials_load(stage, scene_dir);
    } else if (s_load_materials && lazy_mesh_requested) {
        fprintf(stderr,
                "scene_load: deferring material load for lazy metadata pass\n");
    }

    /* Mesh records are grown on demand. PointInstancer expansion can exceed
     * nprims by a large factor, but reserving the old worst-case array up
     * front made camera-filtered Moana loads spend hundreds of MB on slots
     * that would never become visible. Keep the old factor as a hard guard so
     * pathological instance clouds truncate instead of exhausting memory. */
    int mesh_factor = scene_no_cull_all_geometry_requested()
        ? 128
        : ((nprims > 100000) ? 4 : 16);
    {
        const char* e = getenv("NUSD_SCENE_MESH_FACTOR");
        if (e && e[0]) {
            long parsed = strtol(e, NULL, 10);
            if (parsed >= 1 && parsed <= 256) mesh_factor = (int)parsed;
        }
    }
    int mesh_limit = nprims > 0 && mesh_factor <= INT_MAX / nprims
        ? nprims * mesh_factor
        : INT_MAX / 2;
    {
        const char* e = getenv("NUSD_MAX_SCENE_MESHES");
        if (e && e[0]) {
            long parsed = strtol(e, NULL, 10);
            if (parsed > 0 && parsed <= INT_MAX)
                mesh_limit = (int)parsed;
        }
    }
    int max_meshes = nprims + 1024;
    if (max_meshes < 1024) max_meshes = 1024;
    if (max_meshes > mesh_limit) max_meshes = mesh_limit;
    scene->meshes = (SceneMesh*)calloc((size_t)max_meshes, sizeof(SceneMesh));
    if (!scene->meshes) {
        scene_set_last_errorf("failed to allocate %d scene mesh records",
                              max_meshes);
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

    int mesh_idx = 0;
    int total_verts = 0;
    int total_indices = 0;
    long long geometry_verts_loaded = 0;
    long long geometry_indices_loaded = 0;
    long long default_vertex_limit =
        scene_no_cull_all_geometry_requested() ? 0LL : 50000000LL;
    long long default_index_limit =
        scene_no_cull_all_geometry_requested() ? 0LL : 250000000LL;
    long long geometry_vertex_limit =
        scene_env_ll_limit("NUSD_MAX_GEOM_VERTICES", default_vertex_limit);
    long long geometry_index_limit =
        scene_env_ll_limit("NUSD_MAX_GEOM_INDICES", default_index_limit);
    int geometry_budget_truncated = 0;
    int instanced_meshes = 0;
    int pi_frustum_cull = scene_point_instance_cull_active();
    float pi_cull_pad = scene_point_instance_cull_pad();
    PathPrefixList pi_proto_prefixes = { NULL, 0, 0 };

    /* NUSD_LAZY_MESH=1 mirrors the Vulkan/OpenGL Tier 3 path: build a
     * metadata-only scene quickly, then let nu_extract_deferred() rerun
     * extraction on the pinned stage when geometry is actually needed. */
    {
        int lazy_mesh = lazy_mesh_requested;
        if (lazy_mesh) {
            const char* aabb_env = getenv("NUSD_LAZY_AABB");
            int lazy_aabb = !(aabb_env && aabb_env[0] == '0');
            /* Point walks defeat lazy-load interactivity on large assets:
             * Moana is only ~28K composed prims but still needs millions of
             * point reads if authored extents are missing. Default to
             * extent-only bounds for larger stages; extent-less meshes stay
             * always-visible unless explicitly opted in. */
            int lazy_points_aabb = (nprims <= 20000);
            const char* points_aabb_env = getenv("NUSD_LAZY_POINTS_AABB");
            if (points_aabb_env && points_aabb_env[0])
                lazy_points_aabb = (points_aabb_env[0] != '0');
            const char* inst_aabb_env = getenv("NUSD_LAZY_INSTANCE_AABB");
            int lazy_inst_aabb = (inst_aabb_env && inst_aabb_env[0] &&
                                  inst_aabb_env[0] != '0');
            fprintf(stderr,
                    "[scene_load] NUSD_LAZY_MESH=1 — metadata-only walk of flat prim list "
                    "(aabb=%d, points_aabb=%d, instance_aabb=%d).\n",
                    lazy_aabb, lazy_points_aabb, lazy_inst_aabb);
            int lazy_idx = 0;
            int lazy_n_mesh = 0, lazy_n_prim_shape = 0, lazy_n_pi = 0;
            int lazy_n_aabb_extent = 0, lazy_n_aabb_points = 0, lazy_n_aabb_skipped = 0;
            int lazy_n_inst = 0, lazy_n_inst_children = 0;
            double lazy_xform_stack[256][16];
            unsigned char lazy_xform_valid[256] = {0};
            memcpy(lazy_xform_stack[0], kIdentity, sizeof(kIdentity));
            lazy_xform_valid[0] = 1;
            for (int i = 0; i < nprims; i++) {
                NanousdPrim p = nanousd_prim(stage, i);
                if (!p) continue;
                /* Stage traversal already prunes inactive prims; avoid a
                 * per-prim composed active-opinion lookup in the lazy path. */
                const char* tn = nanousd_typename(p);
                if (!tn) { nanousd_freeprim(p); continue; }
                int is_mesh = !strcmp(tn, "Mesh");
                int is_prim_shape = (!strcmp(tn, "Cube") || !strcmp(tn, "Sphere") ||
                                     !strcmp(tn, "Capsule") || !strcmp(tn, "Cylinder"));
                int is_pi = !strcmp(tn, "PointInstancer");
                int is_xform_carrier = is_mesh || is_prim_shape || is_pi ||
                                       !strcmp(tn, "Xform") || !strcmp(tn, "Scope");
                int is_inst_aabb = 0;
                if (lazy_aabb && lazy_inst_aabb)
                    is_inst_aabb = nanousd_isinstance(p);

                double prim_world[16];
                if (is_xform_carrier || is_inst_aabb) {
                    compute_world_xform_cached(p, prim_world,
                                               lazy_xform_stack,
                                               lazy_xform_valid, 256);
                } else {
                    int depth = prim_path_depth(nanousd_path(p));
                    if (depth >= 0 && depth < 256)
                        lazy_xform_valid[depth] = 0;
                }

                if (is_inst_aabb) {
                    const char* inst_root_path_raw = nanousd_path(p);
                    NanousdPrim proto_root = nanousd_prototype(p);
                    const char* proto_root_path_raw = proto_root ? nanousd_path(proto_root) : NULL;
                    char inst_root_path_buf[1024];
                    char proto_root_path_buf[1024];
                    const char* inst_root_path = NULL;
                    const char* proto_root_path = NULL;
                    if (inst_root_path_raw) {
                        snprintf(inst_root_path_buf, sizeof(inst_root_path_buf), "%s",
                                 inst_root_path_raw);
                        inst_root_path = inst_root_path_buf;
                    }
                    if (proto_root_path_raw) {
                        snprintf(proto_root_path_buf, sizeof(proto_root_path_buf), "%s",
                                 proto_root_path_raw);
                        proto_root_path = proto_root_path_buf;
                    }
                    int matched_children = 0;

                    NanousdPrim stack[256];
                    int stack_size = 0;
                    int nc = nanousd_nchildren(p);
                    for (int c = nc - 1; c >= 0 && stack_size < 256; c--) {
                        NanousdPrim child = nanousd_child(p, c);
                        if (child) stack[stack_size++] = child;
                    }
                    while (stack_size > 0) {
                        NanousdPrim child = stack[--stack_size];
                        if (!child) continue;
                        int ncc = nanousd_nchildren(child);
                        for (int c = ncc - 1; c >= 0 && stack_size < 256; c--) {
                            NanousdPrim gc = nanousd_child(child, c);
                            if (gc) stack[stack_size++] = gc;
                        }

                        const char* ctn = nanousd_typename(child);
                        int cmesh = ctn && !strcmp(ctn, "Mesh");
                        int cprim_shape = ctn && (!strcmp(ctn, "Cube") || !strcmp(ctn, "Sphere") ||
                                                  !strcmp(ctn, "Capsule") || !strcmp(ctn, "Cylinder"));
                        if (!cmesh && !cprim_shape) {
                            nanousd_freeprim(child);
                            continue;
                        }
                        matched_children++;

                        float wmin[3], wmax[3];
                        int have_bounds = 0;
                        if (cmesh) {
                            int ext_count = 0;
                            const float* ext = nanousd_arraydataf(child, "extent", &ext_count);
                            if (ext && ext_count == 6) {
                                float lmin[3] = { ext[0], ext[1], ext[2] };
                                float lmax[3] = { ext[3], ext[4], ext[5] };
                                double cw[16];
                                compute_world_xform(child, cw);
                                world_bounds_from_local(cw, lmin, lmax, wmin, wmax);
                                have_bounds = bounds_is_usable(wmin, wmax);
                            }
                        }

                        if (!scene_mesh_reserve(scene, &max_meshes, lazy_idx + 1,
                                                mesh_limit)) {
                            nanousd_freeprim(child);
                            continue;
                        }
                        SceneMesh* m = &scene->meshes[lazy_idx];
                        memset(m, 0, sizeof(*m));
                        memcpy(m->world_xform, prim_world, sizeof(prim_world));
                        if (have_bounds) {
                            memcpy(m->bounds_min, wmin, sizeof(wmin));
                            memcpy(m->bounds_max, wmax, sizeof(wmax));
                            scene_expand_bounds(scene, wmin, wmax);
                            lazy_n_aabb_extent++;
                        } else {
                            bounds_set_never_cull(m->bounds_min, m->bounds_max);
                            lazy_n_aabb_skipped++;
                        }
                        m->display_color[0] = 0.7f;
                        m->display_color[1] = 0.7f;
                        m->display_color[2] = 0.7f;
                        m->material_index = -1;
                        m->prototype_idx = lazy_idx;
                        m->is_proto_only = 0;
                        m->has_display_color = 0;
                        m->lazy_prim_idx = i;
                        {
                            const char* child_path = nanousd_path(child);
                            const char* store_path = child_path;
                            char concrete_path[512];
                            if (child_path && inst_root_path && proto_root_path &&
                                path_has_prefix(child_path, proto_root_path)) {
                                size_t proto_len = strlen(proto_root_path);
                                snprintf(concrete_path, sizeof(concrete_path), "%s%s",
                                         inst_root_path, child_path + proto_len);
                                store_path = concrete_path;
                            }
                            if (store_path) {
                                size_t n = strlen(store_path);
                                if (n < sizeof(m->path)) {
                                    memcpy(m->path, store_path, n + 1);
                                }
                            }
                        }
                        lazy_idx++;
                        nanousd_freeprim(child);
                    }
                    lazy_n_inst++;
                    lazy_n_inst_children += matched_children;
                    if (proto_root) nanousd_freeprim(proto_root);
                    nanousd_freeprim(p);
                    continue;
                }

                if (!is_mesh && !is_prim_shape && !is_pi) {
                    nanousd_freeprim(p);
                    continue;
                }

                if (!scene_mesh_reserve(scene, &max_meshes, lazy_idx + 1,
                                        mesh_limit)) {
                    nanousd_freeprim(p);
                    break;
                }

                SceneMesh* m = &scene->meshes[lazy_idx];
                m->positions = NULL;
                m->normals = NULL;
                m->colors = NULL;
                m->texcoords = NULL;
                m->indices = NULL;
                /* Deferred extraction re-reads geometry later; keeping this
                 * zero avoids touching `points` just for a diagnostic count. */
                m->nvertices = 0;
                m->nindices = 0;
                memcpy(m->world_xform, prim_world, sizeof(prim_world));
                m->bounds_min[0] = m->bounds_min[1] = m->bounds_min[2] = -FLT_MAX;
                m->bounds_max[0] = m->bounds_max[1] = m->bounds_max[2] =  FLT_MAX;
                if (is_mesh) {
                    if (lazy_aabb) {
                        int ext_count = 0;
                        const float* ext = nanousd_arraydataf(p, "extent", &ext_count);
                        if (ext && ext_count == 6) {
                            float local_min[3] = { ext[0], ext[1], ext[2] };
                            float local_max[3] = { ext[3], ext[4], ext[5] };
                            world_bounds_from_local(m->world_xform, local_min, local_max,
                                                    m->bounds_min, m->bounds_max);
                            if (bounds_is_usable(m->bounds_min, m->bounds_max)) {
                                scene_expand_bounds(scene, m->bounds_min, m->bounds_max);
                                lazy_n_aabb_extent++;
                            } else {
                                bounds_set_never_cull(m->bounds_min, m->bounds_max);
                                lazy_n_aabb_skipped++;
                            }
                        } else if (lazy_points_aabb) {
                            int pts_count = 0;
                            const float* pts = nanousd_arraydataf(p, "points", &pts_count);
                            if (pts && pts_count >= 3) {
                                float local_min[3], local_max[3];
                                bounds_init(local_min, local_max);
                                int npts = pts_count / 3;
                                for (int v = 0; v < npts; v++) {
                                    const float* q = &pts[v * 3];
                                    for (int k = 0; k < 3; k++) {
                                        if (q[k] < local_min[k]) local_min[k] = q[k];
                                        if (q[k] > local_max[k]) local_max[k] = q[k];
                                    }
                                }
                                world_bounds_from_local(m->world_xform, local_min, local_max,
                                                        m->bounds_min, m->bounds_max);
                                m->nvertices = npts;
                                if (bounds_is_usable(m->bounds_min, m->bounds_max)) {
                                    scene_expand_bounds(scene, m->bounds_min, m->bounds_max);
                                    lazy_n_aabb_points++;
                                } else {
                                    bounds_set_never_cull(m->bounds_min, m->bounds_max);
                                    lazy_n_aabb_skipped++;
                                }
                            } else {
                                lazy_n_aabb_skipped++;
                            }
                        } else {
                            lazy_n_aabb_skipped++;
                        }
                    } else {
                        lazy_n_aabb_skipped++;
                    }
                }
                m->material_index = -1;
                m->vertex_offset = 0;
                m->index_offset = 0;
                m->prototype_idx = lazy_idx;
                m->is_proto_only = 0;
                m->has_display_color = 0;
                m->display_color[0] = 0.7f;
                m->display_color[1] = 0.7f;
                m->display_color[2] = 0.7f;
                m->lazy_prim_idx = i;
                if (is_mesh) lazy_n_mesh++;
                else if (is_pi) lazy_n_pi++;
                else lazy_n_prim_shape++;
                total_verts += m->nvertices;
                lazy_idx++;
                nanousd_freeprim(p);
            }
            scene->nmeshes = lazy_idx;
            mesh_idx = lazy_idx;
            if (scene->bounds_min[0] == FLT_MAX) {
                scene->bounds_min[0] = scene->bounds_min[1] = scene->bounds_min[2] = 0.0f;
                scene->bounds_max[0] = scene->bounds_max[1] = scene->bounds_max[2] = 0.0f;
            }
            fprintf(stderr,
                    "scene_load: lazy metadata captured %d records "
                    "(%d Mesh, %d primitive shapes, %d PointInstancer; "
                    "aabb extent=%d points=%d skipped=%d; "
                    "%d instance prims w/ %d matched children)\n",
                    lazy_idx, lazy_n_mesh, lazy_n_prim_shape, lazy_n_pi,
                    lazy_n_aabb_extent, lazy_n_aabb_points, lazy_n_aabb_skipped,
                    lazy_n_inst, lazy_n_inst_children);
            goto lazy_postprocess;
        }
    }

    /* --- Extract mesh data --- */
    if (pi_frustum_cull) {
        collect_point_instancer_proto_prefixes(stage, nprims, &pi_proto_prefixes);
        fprintf(stderr,
                "scene_load: PointInstancer frustum cull active "
                "(%d cameras, %d prototype prefixes, pad=%.3f)\n",
                s_pi_frustum_count, pi_proto_prefixes.count, pi_cull_pad);
    }

    /* Prototype mesh lookup: maps prim path → mesh_idx for instancing */
    ProtoMeshHash proto_mesh_table = { NULL, 0, 0 };
    int geometry_only = scene_geometry_only_fast_path();
    if (geometry_only) {
        fprintf(stderr,
                "scene_load: geometry-only extraction active "
                "(normals=%s, displayColor=%s, skip face-varying primvars, materials, subsets)\n",
                scene_geometry_only_skip_normals() ? "skipped" : "authored",
                scene_geometry_only_skip_display_color() ? "skipped" : "authored");
    }

    /* Collect instance prims for the second pass */
    int n_instance_prims = 0, inst_cap = 0;
    int n_proto_only_instance_prims = 0;
    int n_direct_instance_roots = 0;
    int n_skipped_filtered_instance_roots = 0;
    int n_instance_child_meshes = 0;
    int n_instance_child_culled = 0;
    int* instance_prim_indices = NULL;

    #define PRIM_IS_WANTED(I)                                                  \
        (!wanted_prims || nprims_in_bitmap <= 0 ||                             \
         (I) < 0 || (I) >= nprims_in_bitmap || wanted_prims[(I)] != 0)

    /* Renderable USD geometry types NOT materialized by any load pass (Mesh +
     * Cube/Sphere/Capsule/Cylinder are handled here; BasisCurves + PointInstancer
     * have their own passes). If any of these appear, geometry is being dropped
     * — count them so it is never silent. */
    int n_unhandled_geom = 0;
    char unhandled_geom_example[64] = {0};
    /* Intentional, USD-spec-correct hides — counted so they are auditable
     * rather than silent (a datacenter that tagged real geometry proxy/guide
     * would show up here). */
    int n_hide_navmesh = 0, n_hide_vis_own = 0, n_hide_vis_inherited = 0, n_hide_purpose = 0;

    /* First pass: extract meshes from flat prim list + collect instance prims */
    for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;
        int prim_wanted = PRIM_IS_WANTED(i);
        int keep_for_pi_proto = 0;
        if (pi_frustum_cull && !prim_wanted) {
            keep_for_pi_proto =
                path_prefix_list_contains(&pi_proto_prefixes, nanousd_path(prim));
        }
        if (!prim_wanted && !keep_for_pi_proto) {
            nanousd_freeprim(prim);
            continue;
        }

        /* Collect instance prims for second pass + record proto→inst mapping */
        if (nanousd_isinstance(prim)) {
            if (prim_wanted) {
                if (n_instance_prims >= inst_cap) {
                    inst_cap = inst_cap ? inst_cap * 2 : 64;
                    instance_prim_indices = (int*)realloc(instance_prim_indices, (size_t)inst_cap * sizeof(int));
                }
                instance_prim_indices[n_instance_prims++] = i;
            } else if (keep_for_pi_proto) {
                n_proto_only_instance_prims++;
            }

        }

        const char* tn = nanousd_typename(prim);
        int is_mesh = (tn && !strcmp(tn, "Mesh"));
        int is_prim_shape = (tn && (!strcmp(tn, "Cube") || !strcmp(tn, "Sphere") ||
                                    !strcmp(tn, "Capsule") || !strcmp(tn, "Cylinder")));
        int is_points = (tn && !strcmp(tn, "Points"));
        if (!nanousd_isactive(prim) || !tn || (!is_mesh && !is_prim_shape && !is_points)) {
            /* Flag active renderable geometry types that no pass materializes,
             * so dropped geometry is loud rather than silent. BasisCurves and
             * PointInstancer are intentionally excluded (handled elsewhere). */
            if (tn && nanousd_isactive(prim) &&
                (!strcmp(tn, "Cone") ||
                 !strcmp(tn, "Plane") || !strcmp(tn, "NurbsPatch") ||
                 !strcmp(tn, "NurbsCurves") || !strcmp(tn, "HermiteCurves") ||
                 !strcmp(tn, "TetMesh"))) {
                if (n_unhandled_geom == 0)
                    snprintf(unhandled_geom_example, sizeof(unhandled_geom_example),
                             "%s @ %s", tn, nanousd_path(prim) ? nanousd_path(prim) : "?");
                n_unhandled_geom++;
            }
            nanousd_freeprim(prim);
            continue;
        }

        /* Skip meshes hidden from the default render. Checks the mesh's own
         * `visibility` and `purpose`, plus a path-prefix hit on Isaac's
         * /Navmesh/ pathfinding volumes (authored without an explicit purpose
         * but never meant to render). */
        /* Skip meshes hidden from the default render (own/inherited visibility,
         * purpose, Isaac /Navmesh/), and in the SAME ancestor traversal detect a
         * PointInstancer ancestor (proto-only). Merged into one parent walk: the
         * visibility walk already climbs to the root for every kept mesh, so the
         * PointInstancer check rides along for free instead of paying a second
         * full nanousd_parent chain (each parent step interns paths — the load's
         * dominant cost). Equivalent to the old two-walk form: a hidden mesh is
         * skipped before is_proto_only is read, and a kept mesh has no invisible
         * ancestor so its visibility walk reaches the root and sees every
         * PointInstancer ancestor the old proto walk would have. */
        int pi_ancestor = 0;
        {
            int hide = 0;
            const char* ppath = nanousd_path(prim);
            if (ppath && strstr(ppath, "/Navmesh/") == ppath) { hide = 1; n_hide_navmesh++; }

            if (!hide) {
                int ok = 0;
                const char* vis = nanousd_attrib_token(prim, "visibility", &ok);
                if (ok && vis && !strcmp(vis, "invisible")) { hide = 1; n_hide_vis_own++; }
            }
            /* Single ancestor walk: visibility=invisible is inherited per the USD
             * spec (a Mesh with visibility="inherited" still hides under an
             * invisible ancestor; usdview H/Shift+H authors it on an Xform), and
             * a PointInstancer ancestor marks geometry as proto-only. */
            if (!hide) {
                int inh_hide = 0;
                NanousdPrim a = nanousd_parent(prim);
                while (a) {
                    int ok = 0;
                    const char* vis = nanousd_attrib_token(a, "visibility", &ok);
                    if (ok && vis && !strcmp(vis, "invisible")) {
                        inh_hide = 1;
                        nanousd_freeprim(a);
                        break;
                    }
                    if (!pi_ancestor) {
                        const char* atn = nanousd_typename(a);
                        if (atn && !strcmp(atn, "PointInstancer")) pi_ancestor = 1;
                    }
                    NanousdPrim parent = nanousd_parent(a);
                    nanousd_freeprim(a);
                    a = parent;
                }
                if (inh_hide) { hide = 1; n_hide_vis_inherited++; }
            }
            if (!hide) {
                int ok = 0;
                const char* pur = nanousd_attrib_token(prim, "purpose", &ok);
                if (ok && pur && (!strcmp(pur, "guide") || !strcmp(pur, "proxy"))) { hide = 1; n_hide_purpose++; }
            }

            if (hide) {
                nanousd_freeprim(prim);
                continue;
            }
        }

        if (!scene_mesh_reserve(scene, &max_meshes, mesh_idx + 1,
                                mesh_limit)) {
            if (scene_fail_on_limit()) {
                scene_fail_mesh_record_limit(scene, "root mesh extraction",
                                             mesh_idx + 1, mesh_limit,
                                             nanousd_path(prim));
            }
            fprintf(stderr,
                    "scene_load: mesh record limit reached (%d); "
                    "truncating visible mesh extraction\n",
                    mesh_limit);
            nanousd_freeprim(prim);
            break;
        }
        SceneMesh* mesh = &scene->meshes[mesh_idx];

        /* USD semantics: meshes under PointInstancer prototype subtrees and
         * native /__Prototype_N roots are geometry sources, not standalone
         * drawables. Keep them for sharing but hide the source entry. */
        mesh->is_proto_only =
            path_is_native_usd_prototype(nanousd_path(prim)) || pi_ancestor;

        /* ---- (pre) UsdGeom primitive shapes: synthesize geometry now,
         *           skip the Mesh-specific attribute readers below. ---- */
        if (is_prim_shape) {
            if (!load_primitive_geom(prim, tn, arena, mesh)) {
                nanousd_freeprim(prim);
                continue;
            }
            goto after_mesh_attrs;
        }

        /* ---- UsdGeomPoints: one small box per point (see build_points_geom). ---- */
        if (is_points) {
            if (!build_points_geom(prim, arena, mesh)) {
                nanousd_freeprim(prim);
                continue;
            }
            goto after_mesh_attrs;
        }

        /* ---- (a) Points ---- */
        int pts_count = 0;
        const float* pts_data = nanousd_arraydataf(prim, "points", &pts_count);

        if (pts_data && pts_count > 0) {
            int candidate_vertices = pts_count / 3;
            if (!scene_geometry_budget_allows(
                    geometry_verts_loaded, candidate_vertices,
                    geometry_vertex_limit,
                    geometry_indices_loaded, 0,
                    geometry_index_limit)) {
                scene_log_geometry_budget(
                    "mesh points", nanousd_path(prim),
                    geometry_verts_loaded, candidate_vertices,
                    geometry_vertex_limit,
                    geometry_indices_loaded, 0,
                    geometry_index_limit);
                if (scene_fail_on_limit())
                    scene_fail_geometry_budget(
                        scene, "mesh points", nanousd_path(prim),
                        geometry_verts_loaded, candidate_vertices,
                        geometry_vertex_limit,
                        geometry_indices_loaded, 0,
                        geometry_index_limit);
                geometry_budget_truncated = 1;
                nanousd_freeprim(prim);
                break;
            }
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
                int candidate_vertices = len / 3;
                if (!scene_geometry_budget_allows(
                        geometry_verts_loaded, candidate_vertices,
                        geometry_vertex_limit,
                        geometry_indices_loaded, 0,
                        geometry_index_limit)) {
                    scene_log_geometry_budget(
                        "mesh points", nanousd_path(prim),
                        geometry_verts_loaded, candidate_vertices,
                        geometry_vertex_limit,
                        geometry_indices_loaded, 0,
                        geometry_index_limit);
                    if (scene_fail_on_limit())
                        scene_fail_geometry_budget(
                            scene, "mesh points", nanousd_path(prim),
                            geometry_verts_loaded, candidate_vertices,
                            geometry_vertex_limit,
                            geometry_indices_loaded, 0,
                            geometry_index_limit);
                    geometry_budget_truncated = 1;
                    nanousd_freeprim(prim);
                    break;
                }
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
        if (scene_geometry_only_skip_normals()) {
            mesh->normals = NULL;
        } else {
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
            if (!scene_geometry_budget_allows(
                    geometry_verts_loaded, mesh->nvertices,
                    geometry_vertex_limit,
                    geometry_indices_loaded, tri_count,
                    geometry_index_limit)) {
                scene_log_geometry_budget(
                    "mesh indices", nanousd_path(prim),
                    geometry_verts_loaded, mesh->nvertices,
                    geometry_vertex_limit,
                    geometry_indices_loaded, tri_count,
                    geometry_index_limit);
                if (scene_fail_on_limit())
                    scene_fail_geometry_budget(
                        scene, "mesh indices", nanousd_path(prim),
                        geometry_verts_loaded, mesh->nvertices,
                        geometry_vertex_limit,
                        geometry_indices_loaded, tri_count,
                        geometry_index_limit);
                geometry_budget_truncated = 1;
                nanousd_freeprim(prim);
                break;
            }
            mesh->nindices = tri_count;
            mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)tri_count * sizeof(uint32_t), 16);
            if (mesh->indices) {
                triangulate_faces(fvi_data, fvi_count, fvc_data, fvc_count,
                                  mesh->indices, NULL);
            }
        } else {
            /* No faceVertexCounts: assume all triangles */
            if (!scene_geometry_budget_allows(
                    geometry_verts_loaded, mesh->nvertices,
                    geometry_vertex_limit,
                    geometry_indices_loaded, fvi_count,
                    geometry_index_limit)) {
                scene_log_geometry_budget(
                    "mesh indices", nanousd_path(prim),
                    geometry_verts_loaded, mesh->nvertices,
                    geometry_vertex_limit,
                    geometry_indices_loaded, fvi_count,
                    geometry_index_limit);
                if (scene_fail_on_limit())
                    scene_fail_geometry_budget(
                        scene, "mesh indices", nanousd_path(prim),
                        geometry_verts_loaded, mesh->nvertices,
                        geometry_vertex_limit,
                        geometry_indices_loaded, fvi_count,
                        geometry_index_limit);
                geometry_budget_truncated = 1;
                nanousd_freeprim(prim);
                break;
            }
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

        if (!geometry_only) {
            load_mesh_facevarying_primvars(arena, mesh, stage, prim,
                                           fvi_data, fvi_count,
                                           fvc_data, fvc_count);
        } else {
            mesh->texcoords = NULL;
        }

        after_mesh_attrs:
        if (!scene_geometry_budget_allows(
                geometry_verts_loaded, mesh->nvertices,
                geometry_vertex_limit,
                geometry_indices_loaded, mesh->nindices,
                geometry_index_limit)) {
            scene_log_geometry_budget(
                "mesh finalization", nanousd_path(prim),
                geometry_verts_loaded, mesh->nvertices,
                geometry_vertex_limit,
                geometry_indices_loaded, mesh->nindices,
                geometry_index_limit);
            if (scene_fail_on_limit())
                scene_fail_geometry_budget(
                    scene, "mesh finalization", nanousd_path(prim),
                    geometry_verts_loaded, mesh->nvertices,
                    geometry_vertex_limit,
                    geometry_indices_loaded, mesh->nindices,
                    geometry_index_limit);
            geometry_budget_truncated = 1;
            nanousd_freeprim(prim);
            break;
        }

        /* ---- (e2) Compute smooth normals if none were loaded ---- */
        if (!scene_geometry_only_skip_normals() && !mesh->normals && mesh->positions &&
            mesh->indices && mesh->nindices > 0) {
            mesh->normals = compute_smooth_normals(arena, mesh->positions, mesh->nvertices,
                                                   mesh->indices, mesh->nindices);
        }

        /* ---- (e3) UV coordinates ---- */
        /* Loaded above with face-varying expansion so UV seams are preserved. */

        /* ---- (e4) Material binding lookup ---- */
        if (geometry_only) {
            mesh->material_index = -1;
        } else {
            mesh->material_index = materials_find_binding(
                (MaterialCollection*)scene->materials, stage, prim);
        }

        if (!geometry_only && getenv("NUSD_BIND_DIAG")) {
            const char* pp = nanousd_path(prim);
            const char* needle = getenv("NUSD_BIND_DIAG");
            if (pp && (needle[0] == '*' || strstr(pp, needle))) {
                const char* mname = "<none>";
                if (mesh->material_index >= 0) {
                    MaterialCollection* mcd =
                        (MaterialCollection*)scene->materials;
                    if (mcd && mesh->material_index < mcd->nmaterials)
                        mname = mcd->materials[mesh->material_index].prim_path;
                }
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

        if (!scene_geometry_only_skip_display_color()) {
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
        }

        /* ---- (g) World transform ---- */
        compute_world_xform(prim, mesh->world_xform);

        /* ---- (g.1) Prim path for nu_get_mesh_name ---- */
        {
            const char* pp = nanousd_path(prim);
            if (pp && pp[0]) {
                size_t n = strnlen(pp, sizeof(mesh->path) - 1);
                memcpy(mesh->path, pp, n);
                mesh->path[n] = '\0';
            } else {
                mesh->path[0] = '\0';
            }
        }

        if (getenv("NUSD_XFORM_DIAG") && (mesh_idx % 100 == 0 || mesh_idx < 4)) {
            fprintf(stderr, "[xform_diag] mesh %d path=%s T=(%.2f,%.2f,%.2f)\n",
                    mesh_idx, mesh->path[0] ? mesh->path : "?",
                    mesh->world_xform[12], mesh->world_xform[13], mesh->world_xform[14]);
        }

        /* ---- Update bounds (scene + per-mesh) ---- */
        mesh->bounds_min[0] = FLT_MAX;  mesh->bounds_min[1] = FLT_MAX;  mesh->bounds_min[2] = FLT_MAX;
        mesh->bounds_max[0] = -FLT_MAX; mesh->bounds_max[1] = -FLT_MAX; mesh->bounds_max[2] = -FLT_MAX;
        for (int v = 0; v < mesh->nvertices; v++) {
            float wp[3];
            xform_point(mesh->world_xform, &mesh->positions[v * 3], wp);

            for (int k = 0; k < 3; k++) {
                if (wp[k] < mesh->bounds_min[k]) mesh->bounds_min[k] = wp[k];
                if (wp[k] > mesh->bounds_max[k]) mesh->bounds_max[k] = wp[k];
                if (wp[k] < scene->bounds_min[k]) scene->bounds_min[k] = wp[k];
                if (wp[k] > scene->bounds_max[k]) scene->bounds_max[k] = wp[k];
            }
        }
        if (getenv("NUSD_BOUNDS_DUMP")) {
            const char* pp = nanousd_path(prim);
            float dx = mesh->bounds_max[0] - mesh->bounds_min[0];
            float dy = mesh->bounds_max[1] - mesh->bounds_min[1];
            float dz = mesh->bounds_max[2] - mesh->bounds_min[2];
            float biggest = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
            /* Only log meshes whose bounds exceed N meters in any axis. */
            float threshold = (float)atof(getenv("NUSD_BOUNDS_DUMP"));
            if (threshold <= 0.0f) threshold = 50.0f;
            if (biggest > threshold) {
                fprintf(stderr, "[bounds] BIG mesh %d %s extents=(%.2f,%.2f,%.2f) min=(%.2f,%.2f,%.2f) max=(%.2f,%.2f,%.2f)\n",
                        mesh_idx, pp ? pp : "?",
                        dx, dy, dz,
                        mesh->bounds_min[0], mesh->bounds_min[1], mesh->bounds_min[2],
                        mesh->bounds_max[0], mesh->bounds_max[1], mesh->bounds_max[2]);
            }
        }

        /* ---- Offsets (zeroed, set by viewer later) ---- */
        mesh->vertex_offset = 0;
        mesh->index_offset = 0;

        /* ---- Instancing: register prototype mesh by path ---- */
        mesh->prototype_idx = mesh_idx;  /* self = unique or first instance */
        {
            const char* mesh_path = nanousd_path(prim);
            if (mesh_path) {
                proto_hash_insert(&proto_mesh_table, mesh_path, mesh_idx);
            }
        }
        geometry_verts_loaded += mesh->nvertices;
        geometry_indices_loaded += mesh->nindices;

        int emitted_meshes = geometry_only
            ? 1
            : split_mesh_by_geom_subsets(
                arena, scene, stage, prim, mesh_idx, &max_meshes,
                mesh_limit, fvc_data, fvc_count, fvi_count);
        for (int e = 0; e < emitted_meshes; e++) {
            total_verts += scene->meshes[mesh_idx + e].nvertices;
            total_indices += scene->meshes[mesh_idx + e].nindices;
        }
        mesh_idx += emitted_meshes;

        nanousd_freeprim(prim);
    }

    /* ---- Second pass: expand instance prims into instanced mesh copies ---- */
    /* Instance mesh children are NOT in the flat prim list — they are only
     * accessible via nanousd_child(). For the first instance of each prototype,
     * we fully load geometry. Subsequent instances share that geometry. */
    if (!geometry_budget_truncated && n_instance_prims > 0) {
        int flat_native_replay =
            scene_flat_native_instance_traversal_enabled() && !pi_frustum_cull;
        int native_diag = scene_native_replay_diag_enabled();
        NativePrototypeReplayCache native_replay_cache = { NULL, 0, 0 };
        int native_replay_catalogs_built = 0;
        int native_replay_roots_emitted = 0;
        int native_replay_meshes_emitted = 0;
        int native_replay_entries_recorded = 0;
        int native_replay_asset_seed_roots = 0;
        int native_replay_asset_seed_meshes = 0;
        double native_replay_asset_seed_ms = 0.0;
        fprintf(stderr,
                "scene_load: expanding %d instance prims%s...\n",
                n_instance_prims,
                flat_native_replay ? " with flat prototype replay" : "");

        /* Nested-PI bookkeeping: reset stale process-level state and record
         * the concrete paths of PIs the top-level Third Pass already owns
         * (isCoral/isNaupakaA are composed-visible), so the nested seed path
         * skips them instead of double-counting. */
        nested_pending_reset();
        toplevel_pi_paths_reset();
        s_nested_pi_emitted_leaves = 0;
        s_nested_pi_emitted_batches = 0;
        s_nested_pi_unique = 0;
        s_nested_pi_skipped_toplevel = 0;
        s_nested_pi_skipped_dup = 0;
        if (flat_native_replay) {
            for (int i = 0; i < nprims; i++) {
                if (!PRIM_IS_WANTED(i)) continue;
                NanousdPrim p = nanousd_prim(stage, i);
                if (!p) continue;
                if (nanousd_isa(p, "PointInstancer")) {
                    const char* pp = nanousd_path(p);
                    if (pp) toplevel_pi_paths_add(pp);
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
            const char* inst_root_path_raw = nanousd_path(inst_prim);
            if (!proto_root_path_raw || !inst_root_path_raw) {
                nanousd_freeprim(proto_root);
                nanousd_freeprim(inst_prim);
                continue;
            }
            char proto_root_path_buf[1024];
            char inst_root_path_buf[1024];
            snprintf(proto_root_path_buf, sizeof(proto_root_path_buf), "%s",
                     proto_root_path_raw);
            snprintf(inst_root_path_buf, sizeof(inst_root_path_buf), "%s",
                     inst_root_path_raw);
            const char* proto_root_path = proto_root_path_buf;
            const char* inst_root_path = inst_root_path_buf;
            int instance_is_pi_proto =
                pi_frustum_cull &&
                path_prefix_list_contains(&pi_proto_prefixes, inst_root_path);
            int instance_diag = pi_frustum_cull &&
                                s_visible_instance_child_path_count > 0 &&
                                !instance_is_pi_proto;
            if (native_diag)
                instance_diag = 1;
            double instance_t0 = instance_diag ? get_time_sec() : 0.0;
            double proto_root_world[16];
            double proto_root_inv[16];
            double inst_root_world[16];
            double inst_root_inv[16];
            compute_world_xform(proto_root, proto_root_world);
            int have_proto_root_inv =
                invert_affine_m4d_rowvec(proto_root_world, proto_root_inv);
            compute_world_xform(inst_prim, inst_root_world);
            int have_inst_root_inv =
                invert_affine_m4d_rowvec(inst_root_world, inst_root_inv);

            NativePrototypeReplayCatalog* replay_catalog = NULL;
            uint64_t replay_key_hash = 0;
            if (flat_native_replay && have_inst_root_inv) {
                char replay_key[768];
                native_instance_prototype_key(inst_prim, proto_root,
                                              replay_key, sizeof(replay_key));
                replay_catalog =
                    native_replay_cache_find_or_add(&native_replay_cache,
                                                    replay_key, NULL);
                if (replay_catalog) {
                    replay_catalog->roots_seen++;
                    replay_key_hash = proto_fnv1a64(replay_catalog->key);
                    if (replay_catalog->built) {
                        int emitted = emit_native_replay_catalog(
                            scene, &max_meshes, mesh_limit, &mesh_idx,
                            &instanced_meshes, &total_verts, &total_indices,
                            inst_root_path, inst_root_world, replay_catalog);
                        if (emitted < 0) {
                            if (scene_fail_on_limit()) {
                                scene_fail_mesh_record_limit(
                                    scene, "flat native instance replay",
                                    mesh_idx + 1, mesh_limit, inst_root_path);
                            }
                            fprintf(stderr,
                                    "scene_load: mesh record limit reached (%d), "
                                    "truncating flat native instance replay\n",
                                    mesh_limit);
                            nanousd_freeprim(proto_root);
                            nanousd_freeprim(inst_prim);
                            goto done_instances;
                        }
                        native_replay_roots_emitted++;
                        native_replay_meshes_emitted += emitted;
                        nanousd_freeprim(proto_root);
                        nanousd_freeprim(inst_prim);
                        continue;
                    }
                }
            }

            /* Traverse instance children recursively via a stack. During
             * filtered lazy extraction, prefer the exact visible child paths
             * collected from the lazy AABB pass so we do not recursively walk
             * a huge instance root just to discard most of its children. */
            InstanceChildWork* stack_arr = NULL;
            int stack_cap = 0;
            int stack_size = 0;
            int direct_child_path_mode = 0;
            NativeAssetStageList owned_stages = { NULL, 0, 0 };

#define ABORT_INSTANCE_EXPANSION()                                             \
            do {                                                               \
                instance_child_work_stack_free(stack_arr, stack_size);          \
                native_asset_stage_list_close(&owned_stages);                  \
                nanousd_freeprim(proto_root);                                  \
                nanousd_freeprim(inst_prim);                                   \
                goto done_instances;                                           \
            } while (0)

            if (pi_frustum_cull && !instance_is_pi_proto &&
                s_visible_instance_child_path_count > 0) {
                int matched_paths = 0;
                for (int p = 0; p < s_visible_instance_child_path_count; p++) {
                    const char* child_path = s_visible_instance_child_paths[p];
                    if (!path_has_prefix(child_path, inst_root_path))
                        continue;
                    if (child_path[strlen(inst_root_path)] != '/')
                        continue;
                    matched_paths++;
                }
                fprintf(stderr,
                        "scene_load: instance root %d/%d direct visible paths=%d path=%s\n",
                        ii + 1, n_instance_prims, matched_paths, inst_root_path);
                double direct_resolve_t0 = get_time_sec();
                for (int p = 0; p < s_visible_instance_child_path_count; p++) {
                    const char* child_path = s_visible_instance_child_paths[p];
                    if (!path_has_prefix(child_path, inst_root_path))
                        continue;
                    if (child_path[strlen(inst_root_path)] != '/')
                        continue;
                    const char* relative = child_path + strlen(inst_root_path);
                    NanousdPrim child =
                        resolve_relative_child_prim(proto_root, relative);
                    int resolved_in_proto_space = child != NULL;
                    if (!child) {
                        child = resolve_relative_child_prim(inst_prim, relative);
                        resolved_in_proto_space = 0;
                    }
                    if (child) {
                        if (!instance_child_work_push(&stack_arr, &stack_size,
                                                      &stack_cap, child,
                                                      child_path,
                                                      resolved_in_proto_space)) {
                            nanousd_freeprim(child);
                        }
                    }
                }
                if (getenv("NUSD_INSTANCE_CHILD_DIAG")) {
                    fprintf(stderr,
                            "scene_load: instance root %d/%d resolved %d direct "
                            "child prims in %.1f ms\n",
                            ii + 1, n_instance_prims, stack_size,
                            (get_time_sec() - direct_resolve_t0) * 1000.0);
                }
                direct_child_path_mode = (stack_size > 0);
            }

            if (!direct_child_path_mode && flat_native_replay &&
                have_inst_root_inv) {
                double seed_t0 = get_time_sec();
                int seeded = seed_native_instance_stack_from_asset_arcs(
                    stage, inst_prim, inst_root_path, inst_root_world,
                    &stack_arr, &stack_size, &stack_cap, &owned_stages);
                double seed_ms = (get_time_sec() - seed_t0) * 1000.0;
                native_replay_asset_seed_ms += seed_ms;
                if (seeded > 0) {
                    direct_child_path_mode = 1;
                    native_replay_asset_seed_roots++;
                    native_replay_asset_seed_meshes += seeded;
                }
                if (native_diag) {
                    fprintf(stderr,
                            "scene_load: native asset seed root %d/%d "
                            "seeded=%d stack=%d stages=%d time=%.1f ms "
                            "maxrss=%.1f MiB path=%s\n",
                            ii + 1, n_instance_prims, seeded, stack_size,
                            owned_stages.count, seed_ms,
                            (double)scene_maxrss_bytes() / (1024.0 * 1024.0),
                            inst_root_path);
                }
            }

            if (!direct_child_path_mode) {
                if (native_diag) {
                    fprintf(stderr,
                            "scene_load: native asset seed root %d/%d "
                            "falling back to child walk path=%s\n",
                            ii + 1, n_instance_prims, inst_root_path);
                }
                if (pi_frustum_cull && !instance_is_pi_proto &&
                    s_visible_instance_child_path_count > 0) {
                    n_skipped_filtered_instance_roots++;
                    native_asset_stage_list_close(&owned_stages);
                    nanousd_freeprim(proto_root);
                    nanousd_freeprim(inst_prim);
                    continue;
                }
                int nc = nanousd_nchildren(inst_prim);
                for (int c = nc - 1; c >= 0; c--) {
                    NanousdPrim child = nanousd_child(inst_prim, c);
                    if (child) {
                        if (!instance_child_work_push(&stack_arr, &stack_size,
                                                      &stack_cap, child,
                                                      NULL, 0)) {
                            nanousd_freeprim(child);
                            break;
                        }
                    }
                }
            } else {
                n_direct_instance_roots++;
            }

            while (stack_size > 0) {
                InstanceChildWork work = stack_arr[--stack_size];
                NanousdPrim child = work.prim;
                if (!child) continue;
                NanousdStage child_stage = work.stage ? work.stage : stage;

                /* Push grandchildren */
                if (!direct_child_path_mode) {
                    int ncc = nanousd_nchildren(child);
                    for (int c = ncc - 1; c >= 0; c--) {
                        NanousdPrim gc = nanousd_child(child, c);
                        if (gc) {
                            if (!instance_child_work_push(&stack_arr,
                                                          &stack_size,
                                                          &stack_cap,
                                                          gc, NULL, 0)) {
                                nanousd_freeprim(gc);
                                break;
                            }
                        }
                    }
                }

                const char* ctn = nanousd_typename(child);
                if (ctn && strcmp(ctn, "PointInstancer") == 0 && getenv("NUSD_GEO_DIAG")) {
                    int npi = nanousd_attribarraylen(child, "protoIndices");
                    int npr = nanousd_nreltargets(child, "prototypes");
                    fprintf(stderr, "scene_load:   NESTED PI %s instances=%d protos=%d\n",
                            nanousd_path(child), npi, npr);
                }
                if (!ctn || strcmp(ctn, "Mesh") != 0) {
                    nanousd_freeprim(child);
                    continue;
                }

                /* Compute prototype-relative key for geometry sharing */
                const char* child_path = nanousd_path(child);
                if (!child_path) { nanousd_freeprim(child); continue; }
                /* nanousd_path() returns a transient buffer that be_path() reuses on
                   the next call (e.g. inside compute_world_xform below). This path is
                   retained and used as a proto_mesh_table key further down, so copy it
                   into a stable buffer to avoid a heap-use-after-free (ASan-confirmed). */
                char child_path_stable[1024];
                snprintf(child_path_stable, sizeof(child_path_stable), "%s", child_path);
                child_path = child_path_stable;

                size_t inst_len = strlen(inst_root_path);
                size_t proto_len = strlen(proto_root_path);
                const char* relative = NULL;
                char relative_override[512];
                int child_in_proto_space = work.resolved_in_proto_space;
                if (work.relative_path[0]) {
                    snprintf(relative_override, sizeof(relative_override),
                             "/%s", work.relative_path);
                    relative = relative_override;
                } else if (work.instance_path[0] &&
                    path_has_prefix(work.instance_path, inst_root_path)) {
                    relative = work.instance_path + inst_len;
                } else if (path_has_prefix(child_path, inst_root_path)) {
                    relative = child_path + inst_len;
                } else if (path_has_prefix(child_path, proto_root_path)) {
                    relative = child_path + proto_len;
                    child_in_proto_space = 1;
                } else {
                    relative = "";
                }

                char proto_key[512];
                if (replay_key_hash) {
                    snprintf(proto_key, sizeof(proto_key), "%s|g=%016llx%s",
                             proto_root_path,
                             (unsigned long long)replay_key_hash,
                             relative ? relative : "");
                } else {
                    snprintf(proto_key, sizeof(proto_key), "%s%s",
                             proto_root_path, relative ? relative : "");
                }
                char instance_child_path[512];
                if (work.instance_path[0]) {
                    snprintf(instance_child_path, sizeof(instance_child_path),
                             "%s", work.instance_path);
                } else {
                    snprintf(instance_child_path, sizeof(instance_child_path),
                             "%s%s", inst_root_path, relative);
                }
                int child_diag = getenv("NUSD_INSTANCE_CHILD_DIAG") != NULL;
                double child_t0 = child_diag ? get_time_sec() : 0.0;
                if (child_diag) {
                    fprintf(stderr,
                            "scene_load: instance child begin root=%d/%d "
                            "proto_space=%d path=%s\n",
                            ii + 1, n_instance_prims, child_in_proto_space,
                            instance_child_path[0] ? instance_child_path : child_path);
                }

                n_instance_child_meshes++;
                double child_world[16];
                int have_child_world = 0;
                float child_bounds_min[3];
                float child_bounds_max[3];
                int have_child_bounds = 0;
                if (work.has_override_world) {
                    memcpy(child_world, work.override_world, sizeof(child_world));
                    have_child_world = 1;
                }
                if (pi_frustum_cull && !instance_is_pi_proto &&
                    !direct_child_path_mode) {
                    int ext_count = 0;
                    const float* ext = nanousd_arraydataf(child, "extent", &ext_count);
                    if (ext && ext_count == 6) {
                        float local_min[3] = { ext[0], ext[1], ext[2] };
                        float local_max[3] = { ext[3], ext[4], ext[5] };
                        double raw_child_world[16];
                        compute_world_xform(child, raw_child_world);
                        if (child_in_proto_space && have_proto_root_inv) {
                            double child_proto_rel[16];
                            nanousd_mul_m4d(raw_child_world, proto_root_inv, child_proto_rel);
                            nanousd_mul_m4d(child_proto_rel, inst_root_world, child_world);
                        } else {
                            memcpy(child_world, raw_child_world, sizeof(raw_child_world));
                        }
                        have_child_world = 1;
                        world_bounds_from_local(child_world, local_min, local_max,
                                                child_bounds_min, child_bounds_max);
                        if (pi_cull_pad > 0.0f) {
                            for (int k = 0; k < 3; k++) {
                                child_bounds_min[k] -= pi_cull_pad;
                                child_bounds_max[k] += pi_cull_pad;
                            }
                        }
                        have_child_bounds = bounds_is_usable(child_bounds_min,
                                                             child_bounds_max);
                        if (have_child_bounds &&
                            !scene_aabb_visible_in_any_pi_frustum(child_bounds_min,
                                                                  child_bounds_max)) {
                            n_instance_child_culled++;
                            nanousd_freeprim(child);
                            continue;
                        }
                    }
                }

                /* Look up existing prototype mesh */
                int proto_idx = proto_hash_lookup(&proto_mesh_table, proto_key);

                if (proto_idx >= 0) {
                    /* Bounds check */
                    if (!scene_mesh_reserve(scene, &max_meshes, mesh_idx + 1,
                                            mesh_limit)) {
                        if (scene_fail_on_limit()) {
                            scene_fail_mesh_record_limit(
                                scene, "native instance replay",
                                mesh_idx + 1, mesh_limit,
                                instance_child_path[0]
                                    ? instance_child_path : child_path);
                        }
                        fprintf(stderr, "scene_load: mesh record limit reached (%d), truncating instances\n",
                                mesh_limit);
                        nanousd_freeprim(child);
                        ABORT_INSTANCE_EXPANSION();
                    }
                    /* Share geometry with already-loaded prototype mesh */
                    SceneMesh* proto_m = &scene->meshes[proto_idx];
                    SceneMesh* mesh = &scene->meshes[mesh_idx];
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
                    {
                        const char* pp = instance_child_path[0]
                            ? instance_child_path : proto_m->path;
                        size_t n = strnlen(pp, sizeof(mesh->path) - 1);
                        memcpy(mesh->path, pp, n);
                        mesh->path[n] = '\0';
                    }
                    if (instance_child_path[0])
                        proto_hash_insert(&proto_mesh_table,
                                          instance_child_path, proto_idx);

                    if (have_child_world) {
                        memcpy(mesh->world_xform, child_world, sizeof(child_world));
                    } else {
                        double raw_child_world[16];
                        compute_world_xform(child, raw_child_world);
                        if (child_in_proto_space && have_proto_root_inv) {
                            double child_proto_rel[16];
                            nanousd_mul_m4d(raw_child_world, proto_root_inv, child_proto_rel);
                            nanousd_mul_m4d(child_proto_rel, inst_root_world, mesh->world_xform);
                        } else {
                            memcpy(mesh->world_xform, raw_child_world, sizeof(raw_child_world));
                        }
                    }

                    if (have_child_bounds) {
                        memcpy(mesh->bounds_min, child_bounds_min, sizeof(child_bounds_min));
                        memcpy(mesh->bounds_max, child_bounds_max, sizeof(child_bounds_max));
                        scene_expand_bounds(scene, mesh->bounds_min, mesh->bounds_max);
                    } else {
                        mesh->bounds_min[0] = FLT_MAX;  mesh->bounds_min[1] = FLT_MAX;  mesh->bounds_min[2] = FLT_MAX;
                        mesh->bounds_max[0] = -FLT_MAX; mesh->bounds_max[1] = -FLT_MAX; mesh->bounds_max[2] = -FLT_MAX;
                        for (int v = 0; v < mesh->nvertices; v++) {
                            float wp[3];
                            xform_point(mesh->world_xform, &mesh->positions[v * 3], wp);
                            for (int k = 0; k < 3; k++) {
                                if (wp[k] < mesh->bounds_min[k]) mesh->bounds_min[k] = wp[k];
                                if (wp[k] > mesh->bounds_max[k]) mesh->bounds_max[k] = wp[k];
                                if (wp[k] < scene->bounds_min[k]) scene->bounds_min[k] = wp[k];
                                if (wp[k] > scene->bounds_max[k]) scene->bounds_max[k] = wp[k];
                            }
                        }
                    }
                    if (replay_catalog && !replay_catalog->built &&
                        have_inst_root_inv) {
                        double rel_xform[16];
                        nanousd_mul_m4d(mesh->world_xform, inst_root_inv,
                                        rel_xform);
                        if (native_replay_catalog_add(replay_catalog, relative,
                                                      proto_idx, rel_xform)) {
                            native_replay_entries_recorded++;
                        }
                    }
                    mesh->vertex_offset = 0;
                    mesh->index_offset = 0;
                    total_verts += mesh->nvertices;
                    total_indices += mesh->nindices;
                    mesh_idx++;
                    instanced_meshes++;
                } else {
                    /* Bounds check */
                    if (!scene_mesh_reserve(scene, &max_meshes, mesh_idx + 1,
                                            mesh_limit)) {
                        if (scene_fail_on_limit()) {
                            scene_fail_mesh_record_limit(
                                scene, "native instance child geometry",
                                mesh_idx + 1, mesh_limit,
                                instance_child_path[0]
                                    ? instance_child_path : child_path);
                        }
                        fprintf(stderr, "scene_load: mesh record limit reached (%d), truncating instances\n",
                                mesh_limit);
                        nanousd_freeprim(child);
                        ABORT_INSTANCE_EXPANSION();
                    }
                    /* First instance of this prototype mesh — load geometry fully */
                    SceneMesh* mesh = &scene->meshes[mesh_idx];

                    /* Points */
                    int pts_count = 0;
                    const float* pts_data = nanousd_arraydataf(child, "points", &pts_count);
                    if (pts_data && pts_count > 0) {
                        int candidate_vertices = pts_count / 3;
                        if (!scene_geometry_budget_allows(
                                geometry_verts_loaded, candidate_vertices,
                                geometry_vertex_limit,
                                geometry_indices_loaded, 0,
                                geometry_index_limit)) {
                            scene_log_geometry_budget(
                                "instance child points", instance_child_path,
                                geometry_verts_loaded, candidate_vertices,
                                geometry_vertex_limit,
                                geometry_indices_loaded, 0,
                            geometry_index_limit);
                            if (scene_fail_on_limit())
                                scene_fail_geometry_budget(
                                    scene, "instance child points",
                                    instance_child_path,
                                    geometry_verts_loaded, candidate_vertices,
                                    geometry_vertex_limit,
                                    geometry_indices_loaded, 0,
                                    geometry_index_limit);
                            geometry_budget_truncated = 1;
                            nanousd_freeprim(child);
                            ABORT_INSTANCE_EXPANSION();
                        }
                        mesh->nvertices = pts_count / 3;
                        mesh->positions = (float*)arena_alloc(arena, (size_t)pts_count * sizeof(float), 16);
                        if (mesh->positions) memcpy(mesh->positions, pts_data, (size_t)pts_count * sizeof(float));
                    } else {
                        int len = nanousd_attribarraylen(child, "points");
                        if (len > 0) {
                            int candidate_vertices = len / 3;
                            if (!scene_geometry_budget_allows(
                                    geometry_verts_loaded, candidate_vertices,
                                    geometry_vertex_limit,
                                    geometry_indices_loaded, 0,
                                    geometry_index_limit)) {
                                scene_log_geometry_budget(
                                    "instance child points", instance_child_path,
                                    geometry_verts_loaded, candidate_vertices,
                                    geometry_vertex_limit,
                                    geometry_indices_loaded, 0,
                                geometry_index_limit);
                                if (scene_fail_on_limit())
                                    scene_fail_geometry_budget(
                                        scene, "instance child points",
                                        instance_child_path,
                                        geometry_verts_loaded,
                                        candidate_vertices,
                                        geometry_vertex_limit,
                                        geometry_indices_loaded, 0,
                                        geometry_index_limit);
                                geometry_budget_truncated = 1;
                                nanousd_freeprim(child);
                                ABORT_INSTANCE_EXPANSION();
                            }
                            mesh->nvertices = len / 3;
                            mesh->positions = (float*)arena_alloc(arena, (size_t)len * sizeof(float), 16);
                            if (mesh->positions) nanousd_attribarrayf(child, "points", mesh->positions, len);
                        }
                    }
                    if (mesh->nvertices == 0 || !mesh->positions) { nanousd_freeprim(child); continue; }

                    /* Normals */
                    if (scene_geometry_only_skip_normals()) {
                        mesh->normals = NULL;
                    } else {
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
                        if (!scene_geometry_budget_allows(
                                geometry_verts_loaded, mesh->nvertices,
                                geometry_vertex_limit,
                                geometry_indices_loaded, tri_count,
                                geometry_index_limit)) {
                            scene_log_geometry_budget(
                                "instance child indices", instance_child_path,
                                geometry_verts_loaded, mesh->nvertices,
                                geometry_vertex_limit,
                                geometry_indices_loaded, tri_count,
                            geometry_index_limit);
                            if (scene_fail_on_limit())
                                scene_fail_geometry_budget(
                                    scene, "instance child indices",
                                    instance_child_path,
                                    geometry_verts_loaded, mesh->nvertices,
                                    geometry_vertex_limit,
                                    geometry_indices_loaded, tri_count,
                                    geometry_index_limit);
                            geometry_budget_truncated = 1;
                            nanousd_freeprim(child);
                            ABORT_INSTANCE_EXPANSION();
                        }
                        mesh->nindices = tri_count;
                        mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)tri_count * sizeof(uint32_t), 16);
                        if (mesh->indices) {
                            triangulate_faces(fvi_data, fvi_count, fvc_data,
                                              fvc_count, mesh->indices, NULL);
                        }
                    } else {
                        if (!scene_geometry_budget_allows(
                                geometry_verts_loaded, mesh->nvertices,
                                geometry_vertex_limit,
                                geometry_indices_loaded, fvi_count,
                                geometry_index_limit)) {
                            scene_log_geometry_budget(
                                "instance child indices", instance_child_path,
                                geometry_verts_loaded, mesh->nvertices,
                                geometry_vertex_limit,
                                geometry_indices_loaded, fvi_count,
                            geometry_index_limit);
                            if (scene_fail_on_limit())
                                scene_fail_geometry_budget(
                                    scene, "instance child indices",
                                    instance_child_path,
                                    geometry_verts_loaded, mesh->nvertices,
                                    geometry_vertex_limit,
                                    geometry_indices_loaded, fvi_count,
                                    geometry_index_limit);
                            geometry_budget_truncated = 1;
                            nanousd_freeprim(child);
                            ABORT_INSTANCE_EXPANSION();
                        }
                        mesh->nindices = fvi_count;
                        mesh->indices = (uint32_t*)arena_alloc(arena, (size_t)fvi_count * sizeof(uint32_t), 16);
                        if (mesh->indices) {
                            for (int j = 0; j < fvi_count; j++) mesh->indices[j] = (uint32_t)fvi_data[j];
                        }
                    }
                    if (!mesh->indices || mesh->nindices == 0) { nanousd_freeprim(child); continue; }

                    if (!geometry_only) {
                        load_mesh_facevarying_primvars(arena, mesh, child_stage, child,
                                                       fvi_data, fvi_count,
                                                       fvc_data, fvc_count);
                    } else {
                        mesh->texcoords = NULL;
                    }

                    if (!scene_geometry_budget_allows(
                            geometry_verts_loaded, mesh->nvertices,
                            geometry_vertex_limit,
                            geometry_indices_loaded, mesh->nindices,
                            geometry_index_limit)) {
                        scene_log_geometry_budget(
                            "instance child finalization", instance_child_path,
                            geometry_verts_loaded, mesh->nvertices,
                            geometry_vertex_limit,
                            geometry_indices_loaded, mesh->nindices,
                        geometry_index_limit);
                        if (scene_fail_on_limit())
                            scene_fail_geometry_budget(
                                scene, "instance child finalization",
                                instance_child_path,
                                geometry_verts_loaded, mesh->nvertices,
                                geometry_vertex_limit,
                                geometry_indices_loaded, mesh->nindices,
                                geometry_index_limit);
                        geometry_budget_truncated = 1;
                        nanousd_freeprim(child);
                        ABORT_INSTANCE_EXPANSION();
                    }

                    /* Smooth normals if needed */
                    if (!scene_geometry_only_skip_normals() && !mesh->normals && mesh->positions && mesh->indices)
                        mesh->normals = compute_smooth_normals(arena, mesh->positions, mesh->nvertices,
                                                               mesh->indices, mesh->nindices);

                    /* UV coordinates are loaded with face-varying expansion above. */

                    /* Material binding lookup. Direct visible native-instance
                     * extraction must not call nanousd's composed-child walk on
                     * every streamed child; Moana's USDA layers already contain
                     * leaf-name binding opinions, indexed by materials_load(). */
                    if (geometry_only) {
                        mesh->material_index = -1;
                    } else {
                        MaterialCollection* mcoll =
                            (MaterialCollection*)scene->materials;
                        int mat_idx = -1;
                        if (direct_child_path_mode && instance_child_path[0])
                            mat_idx = materials_find_binding_by_path(
                                mcoll, instance_child_path);
                        if (mat_idx >= 0 ||
                            (direct_child_path_mode &&
                             !getenv("NUSD_SLOW_INSTANCE_BINDINGS"))) {
                            mesh->material_index = mat_idx;
                        } else {
                            mesh->material_index =
                                materials_find_binding(mcoll, child_stage, child);
                        }
                    }

                    /* displayColor */
                    mesh->has_display_color = 0;
                    mesh->display_color[0] = 0.5f; mesh->display_color[1] = 0.5f; mesh->display_color[2] = 0.5f;
                    mesh->colors = NULL;
                    if (!scene_geometry_only_skip_display_color()) {
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

                    /* World transform */
                    if (have_child_world) {
                        memcpy(mesh->world_xform, child_world, sizeof(child_world));
                    } else {
                        double raw_child_world[16];
                        compute_world_xform(child, raw_child_world);
                        if (child_in_proto_space && have_proto_root_inv) {
                            double child_proto_rel[16];
                            nanousd_mul_m4d(raw_child_world, proto_root_inv, child_proto_rel);
                            nanousd_mul_m4d(child_proto_rel, inst_root_world, mesh->world_xform);
                        } else {
                            memcpy(mesh->world_xform, raw_child_world, sizeof(raw_child_world));
                        }
                    }
                    mesh->prototype_idx = mesh_idx; /* self = first instance */
                    mesh->is_proto_only = 0;

                    /* Prim path for picking. Direct filtered loads may have
                     * read prototype geometry; keep the concrete instance path. */
                    {
                        const char* pp = instance_child_path[0]
                            ? instance_child_path : nanousd_path(child);
                        if (pp && pp[0]) {
                            size_t n = strnlen(pp, sizeof(mesh->path) - 1);
                            memcpy(mesh->path, pp, n);
                            mesh->path[n] = '\0';
                        } else {
                            mesh->path[0] = '\0';
                        }
                    }

                    /* Bounds */
                    if (have_child_bounds) {
                        memcpy(mesh->bounds_min, child_bounds_min, sizeof(child_bounds_min));
                        memcpy(mesh->bounds_max, child_bounds_max, sizeof(child_bounds_max));
                        scene_expand_bounds(scene, mesh->bounds_min, mesh->bounds_max);
                    } else {
                        mesh->bounds_min[0] = FLT_MAX;  mesh->bounds_min[1] = FLT_MAX;  mesh->bounds_min[2] = FLT_MAX;
                        mesh->bounds_max[0] = -FLT_MAX; mesh->bounds_max[1] = -FLT_MAX; mesh->bounds_max[2] = -FLT_MAX;
                        for (int v = 0; v < mesh->nvertices; v++) {
                            float wp[3];
                            xform_point(mesh->world_xform, &mesh->positions[v * 3], wp);
                            for (int k = 0; k < 3; k++) {
                                if (wp[k] < mesh->bounds_min[k]) mesh->bounds_min[k] = wp[k];
                                if (wp[k] > mesh->bounds_max[k]) mesh->bounds_max[k] = wp[k];
                                if (wp[k] < scene->bounds_min[k]) scene->bounds_min[k] = wp[k];
                                if (wp[k] > scene->bounds_max[k]) scene->bounds_max[k] = wp[k];
                            }
                        }
                    }
                    if (replay_catalog && !replay_catalog->built &&
                        have_inst_root_inv) {
                        double rel_xform[16];
                        nanousd_mul_m4d(mesh->world_xform, inst_root_inv,
                                        rel_xform);
                        if (native_replay_catalog_add(replay_catalog, relative,
                                                      mesh_idx, rel_xform)) {
                            native_replay_entries_recorded++;
                        }
                    }
                    mesh->vertex_offset = 0;
                    mesh->index_offset = 0;

                    /* Register as prototype for sharing */
                    proto_hash_insert(&proto_mesh_table, proto_key, mesh_idx);
                    if (instance_child_path[0])
                        proto_hash_insert(&proto_mesh_table,
                                          instance_child_path, mesh_idx);
                    if (child_path && child_path[0])
                        proto_hash_insert(&proto_mesh_table,
                                          child_path, mesh_idx);

                    total_verts += mesh->nvertices;
                    total_indices += mesh->nindices;
                    geometry_verts_loaded += mesh->nvertices;
                    geometry_indices_loaded += mesh->nindices;
                    mesh_idx++;
                }

                if (child_diag) {
                    fprintf(stderr,
                            "scene_load: instance child end root=%d/%d %.1f ms "
                            "path=%s\n",
                            ii + 1, n_instance_prims,
                            (get_time_sec() - child_t0) * 1000.0,
                            instance_child_path[0] ? instance_child_path : child_path);
                }
                if ((mesh_idx & 0x3fff) == 0 &&
                    scene_check_rss_limit(scene, "native instance expansion")) {
                    nanousd_freeprim(child);
                    ABORT_INSTANCE_EXPANSION();
                }
                nanousd_freeprim(child);
            }
            if (replay_catalog && !replay_catalog->built) {
                replay_catalog->built = 1;
                native_replay_catalogs_built++;
                if (getenv("NUSD_INSTANCE_CHILD_DIAG")) {
                    fprintf(stderr,
                            "scene_load: flat native prototype catalog built "
                            "root=%d/%d entries=%d key=%s\n",
                            ii + 1, n_instance_prims,
                            replay_catalog->count,
                            replay_catalog->key);
                }
            }
            native_asset_stage_list_close(&owned_stages);
            free(stack_arr);
#undef ABORT_INSTANCE_EXPANSION
            if (instance_diag) {
                fprintf(stderr,
                        "scene_load: instance root %d/%d finished in %.1f ms "
                        "(direct=%d, children_seen=%d, culled_total=%d)\n",
                        ii + 1, n_instance_prims,
                        (get_time_sec() - instance_t0) * 1000.0,
                        direct_child_path_mode,
                        n_instance_child_meshes,
                        n_instance_child_culled);
            }

            nanousd_freeprim(proto_root);
            nanousd_freeprim(inst_prim);
        }
done_instances:
        if (flat_native_replay) {
            int replay_entries = 0;
            for (int c = 0; c < native_replay_cache.count; c++)
                replay_entries += native_replay_cache.items[c].count;
            fprintf(stderr,
                    "scene_load: flat native instance replay: catalogs=%d "
                    "built_by_child_walk=%d entries=%d recorded=%d "
                    "replayed_roots=%d replayed_meshes=%d "
                    "asset_seed_roots=%d asset_seed_meshes=%d "
                    "asset_seed_ms=%.1f\n",
                    native_replay_cache.count, native_replay_catalogs_built,
                    replay_entries, native_replay_entries_recorded,
                    native_replay_roots_emitted, native_replay_meshes_emitted,
                    native_replay_asset_seed_roots,
                    native_replay_asset_seed_meshes,
                    native_replay_asset_seed_ms);
            if (getenv("NUSD_GEO_DIAG")) {
                for (int c = 0; c < native_replay_cache.count; c++) {
                    fprintf(stderr,
                            "scene_load:   CATALOG-ROOTS roots_seen=%d "
                            "entries=%d key=%s\n",
                            native_replay_cache.items[c].roots_seen,
                            native_replay_cache.items[c].count,
                            native_replay_cache.items[c].key);
                }
            }
        }
        native_replay_cache_free(&native_replay_cache);
    }

    if (n_proto_only_instance_prims > 0) {
        fprintf(stderr,
                "scene_load: skipped %d prototype-only instance prim expansions "
                "during filtered extraction\n",
                n_proto_only_instance_prims);
    }
    if (n_direct_instance_roots > 0 || n_skipped_filtered_instance_roots > 0) {
        fprintf(stderr,
                "scene_load: filtered USD instance expansion used %d direct roots "
                "(%d roots had no visible child paths)\n",
                n_direct_instance_roots, n_skipped_filtered_instance_roots);
    }
    if (n_instance_child_meshes > 0 && pi_frustum_cull) {
        fprintf(stderr,
                "scene_load: USD instance child frustum cull kept %d/%d meshes "
                "(%d culled)\n",
                n_instance_child_meshes - n_instance_child_culled,
                n_instance_child_meshes,
                n_instance_child_culled);
    }

    free(instance_prim_indices);
    scene_check_rss_limit(scene, "native instance expansion");

    /* Shared PI compact-storage capacities — used by both the nested-PI
     * drain (below) and the top-level Third Pass (further down) so they grow
     * one allocation rather than fighting over scene->pi_transforms. */
    uint64_t pi_xform_cap = 0;
    int      pi_batch_cap = 0;

    /* Drain nested PointInstancers recorded during native-instance seeding
     * into the compact pi_transforms/pi_batches model. The proto meshes they
     * reference were created (as ordinary rows) by the instance child walk
     * above; the drain marks them is_proto_only so they are not also drawn as
     * static piles at the asset origin. */
    scene_drain_nested_pis(scene, mesh_idx, &pi_xform_cap, &pi_batch_cap);
    scene_check_rss_limit(scene, "nested PointInstancer drain");

    /* ---- Third pass: emit compact PointInstancer batches ----
     *
     * Section 8B of docs/report/moana_no_cull_2026-05-25/moana-no-cull-memory-plan.md
     *
     * No SceneMesh row is allocated per PI clone — the prototype Mesh prims
     * already live in scene->meshes from the first pass with is_proto_only=1.
     * Per (PI, prototype, sub-mesh) we emit one SceneInstanceBatch whose
     * transform_offset/transform_count slice points into scene->pi_transforms.
     * 2.9M Moana PI instances cost ~138 MiB of compact transforms instead of
     * ~2.9 GiB of SceneMesh+RendererMesh rows.
     *
     * Acceptance gate (logged at the end of scene_load): pi_scene_mesh_clones=0.
     *
     * The frustum-cull semantics (proto-local bounds cache, per-instance AABB
     * cull) match the previous per-clone path exactly so existing camera-
     * filtered loads behave identically. */
    /* pi_xform_cap / pi_batch_cap declared above (shared with nested drain). */
    int      pi_prims_seen = 0;
    int      pi_total_batches_emitted = 0;
    int      pi_total_transforms_emitted = 0;
    int      pi_total_culled = 0;
    if (scene->load_failed) {
        fprintf(stderr,
                "scene_load: skipping PointInstancer expansion after fatal "
                "load error\n");
    } else if (geometry_budget_truncated) {
        fprintf(stderr,
                "scene_load: skipping PointInstancer expansion after geometry "
                "budget truncation\n");
    } else {
        for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;
        if (!nanousd_isa(prim, "PointInstancer")) { nanousd_freeprim(prim); continue; }
        if (!PRIM_IS_WANTED(i) && !pi_frustum_cull) {
            nanousd_freeprim(prim);
            continue;
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
        else { /* default identity quaternion in nanousd's (x, y, z, w)
                 * layout — not USD's text-form (real, imag) ordering.
                 * Pre-fix this slot wrote [1, 0, 0, 0], which under the
                 * read order downstream would have been a 180° X
                 * rotation; chess set never hit it because it ships
                 * explicit orientations on every PointInstancer. */
            for (int q = 0; q < n_instances; q++) {
                orientations[q*4+0] = 0.0f; orientations[q*4+1] = 0.0f;
                orientations[q*4+2] = 0.0f; orientations[q*4+3] = 1.0f;
            }
        }

        int nscale = nanousd_attribarraylen(prim, "scales");
        if (nscale > 0) nanousd_attribarrayf(prim, "scales", scales, nscale * 3);
        else { for (int s = 0; s < n_instances * 3; s++) scales[s] = 1.0f; }

        /* Resolve prototype paths.
         *
         * One USD `prototypes` rel target can resolve to MULTIPLE meshes —
         * e.g., the OpenChessSet's Pawn proto is an Xform with two child
         * Mesh prims (Geom_Body and Geom_Top). The earlier prefix-match
         * loop `break`d at the first hit, so each instance only carried
         * one of the two meshes (chess set rendered pawn bodies without
         * crowns, or vice versa, depending on hash order). Collect every
         * matching mesh per proto. */
        int  n_protos    = nanousd_nreltargets(prim, "prototypes");
        int** proto_lists = (int**)calloc((size_t)n_protos, sizeof(int*));
        int*  proto_counts = (int*)calloc((size_t)n_protos, sizeof(int));
        for (int p = 0; p < n_protos; p++) {
            const char* proto_path = nanousd_reltarget(prim, "prototypes", p);
            if (!proto_path) continue;

            /* Exact match first — proto resolves to a single Mesh prim. */
            int exact = proto_hash_lookup(&proto_mesh_table, proto_path);
            if (exact < 0) {
                for (int sm = 0; sm < mesh_idx; sm++) {
                    if (scene->meshes[sm].path[0] &&
                        strcmp(scene->meshes[sm].path, proto_path) == 0) {
                        exact = sm;
                        break;
                    }
                }
            }
            if (exact >= 0) {
                proto_lists[p]  = (int*)malloc(sizeof(int));
                proto_lists[p][0] = exact;
                proto_counts[p] = 1;
                continue;
            }

            /* Else: prefix match — proto is an Xform/Scope; collect every
             * mesh whose path starts with proto_path. */
            size_t proto_len = strlen(proto_path);
            int matches[64];
            int n_matches = 0;
            for (int pm = 0; pm < proto_mesh_table.cap && n_matches < 64; pm++) {
                const ProtoMeshSlot* s = &proto_mesh_table.slots[pm];
                if (s->path[0] == '\0') continue;
                if (strncmp(s->path, proto_path, proto_len) == 0 &&
                    (s->path[proto_len] == '/' ||
                     s->path[proto_len] == '\0')) {
                    matches[n_matches++] = s->mesh_idx;
                }
            }
            if (n_matches == 0) {
                for (int sm = 0; sm < mesh_idx && n_matches < 64; sm++) {
                    const char* sp = scene->meshes[sm].path;
                    if (!sp[0]) continue;
                    if (strncmp(sp, proto_path, proto_len) == 0 &&
                        (sp[proto_len] == '/' || sp[proto_len] == '\0')) {
                        matches[n_matches++] = sm;
                    }
                }
            }
            if (n_matches > 0) {
                proto_lists[p]  = (int*)malloc((size_t)n_matches * sizeof(int));
                memcpy(proto_lists[p], matches, (size_t)n_matches * sizeof(int));
                proto_counts[p] = n_matches;
            }
            if (getenv("NUSD_POINTS_DIAG") && proto_counts[p] == 0) {
                NanousdPrim pp = nanousd_primpath(stage, proto_path);
                fprintf(stderr, "[proto_unres] %s type=%s instanceable=%d isinstance=%d hasproto=%d nchild=%d\n",
                        proto_path,
                        pp ? (nanousd_typename(pp) ? nanousd_typename(pp) : "?") : "NULL",
                        pp ? nanousd_isinstanceable(pp) : -1,
                        pp ? nanousd_isinstance(pp) : -1,
                        pp ? (nanousd_prototype(pp) != NULL) : -1,
                        pp ? nanousd_nchildren(pp) : -1);
                if (pp) nanousd_freeprim(pp);
            }
        }

        /* Instancer world transform */
        double instancer_world[16];
        compute_world_xform(prim, instancer_world);

        /* Allocate proto-local bounds cache unconditionally — the compact
         * path needs it for fast scene-bounds expansion (world_bounds_from_local
         * vs per-vertex transform), not just for frustum cull. Cheap: at most
         * mesh_idx * 28 B (a few MiB even for Moana-class scenes). */
        int proto_bound_count = mesh_idx;
        PiProtoLocalBounds* proto_bounds = NULL;
        if (proto_bound_count > 0) {
            proto_bounds = (PiProtoLocalBounds*)calloc((size_t)proto_bound_count,
                                                       sizeof(PiProtoLocalBounds));
        }

        /* Pre-compose every authored instance's world-space transform once
         * per PI. Stored as a 4x4 double during cull/bounds work; packed to
         * 12-float column-major when copied into scene->pi_transforms. ~128
         * B/instance temporary peak ≈ 73 MiB for Moana's largest PI
         * (xgFlutes). Freed at end of PI body. */
        double* inst_worlds = (double*)malloc((size_t)n_instances * 16 * sizeof(double));

        /* Bucket instance indices by pidx so each (pidx, sub) batch can
         * iterate only its own instances instead of scanning all
         * n_instances and filtering. The old per-clone path was
         * O(N × S_avg); without this bucket, the new path is
         * O(P × S × N) — at Moana's 23 PIs with up to ~10 protos × ~10
         * sub-meshes × ~570k instances this turned ~2 minutes into ~34
         * minutes. Bucketing returns it to O(N × S_avg). Memory:
         * 2 × n_instances × sizeof(int) ≈ ~5 MiB peak per PI. */
        int* pidx_inst_offsets = (int*)malloc(((size_t)n_protos + 1) * sizeof(int));
        int* pidx_inst_indices = (int*)malloc((size_t)n_instances * sizeof(int));
        if (!inst_worlds || !pidx_inst_offsets || !pidx_inst_indices) {
            fprintf(stderr,
                    "scene_load: OOM allocating instance world cache for "
                    "PointInstancer '%s' (%d instances)\n",
                    nanousd_path(prim), n_instances);
            free(inst_worlds);
            free(pidx_inst_offsets);
            free(pidx_inst_indices);
            free(proto_bounds);
            free(proto_indices);
            free(positions);
            free(orientations);
            free(scales);
            for (int p = 0; p < n_protos; p++) free(proto_lists[p]);
            free(proto_lists);
            free(proto_counts);
            nanousd_freeprim(prim);
            continue;
        }
        /* Counting + prefix-sum bucket build (O(N + P)). */
        for (int p = 0; p <= n_protos; p++) pidx_inst_offsets[p] = 0;
        for (int inst = 0; inst < n_instances; inst++) {
            int pidx = proto_indices[inst];
            if (pidx < 0 || pidx >= n_protos) continue;
            pidx_inst_offsets[pidx + 1]++;
        }
        for (int p = 1; p <= n_protos; p++) {
            pidx_inst_offsets[p] += pidx_inst_offsets[p - 1];
        }
        /* Scatter — clobber+restore via a per-pidx running cursor. */
        {
            int* cursors = (int*)malloc((size_t)n_protos * sizeof(int));
            if (cursors) {
                memcpy(cursors, pidx_inst_offsets, (size_t)n_protos * sizeof(int));
                for (int inst = 0; inst < n_instances; inst++) {
                    int pidx = proto_indices[inst];
                    if (pidx < 0 || pidx >= n_protos) continue;
                    pidx_inst_indices[cursors[pidx]++] = inst;
                }
                free(cursors);
            } else {
                /* Fallback: slow scatter via re-scan. */
                int* tmp_cursors = (int*)alloca((size_t)n_protos * sizeof(int));
                memcpy(tmp_cursors, pidx_inst_offsets, (size_t)n_protos * sizeof(int));
                for (int inst = 0; inst < n_instances; inst++) {
                    int pidx = proto_indices[inst];
                    if (pidx < 0 || pidx >= n_protos) continue;
                    pidx_inst_indices[tmp_cursors[pidx]++] = inst;
                }
            }
        }

        for (int inst = 0; inst < n_instances; inst++) {
            float qi = orientations[inst*4+0];
            float qj = orientations[inst*4+1];
            float qk = orientations[inst*4+2];
            float w  = orientations[inst*4+3];
            float sx = scales[inst*3+0];
            float sy = scales[inst*3+1];
            float sz = scales[inst*3+2];
            float tx = positions[inst*3+0];
            float ty = positions[inst*3+1];
            float tz = positions[inst*3+2];

            double inst_xform[16];
            double ww=w*w, ii=qi*qi, jj=qj*qj, kk=qk*qk;
            double wi=w*qi, wj=w*qj, wk=w*qk;
            double ij=qi*qj, ik=qi*qk, jk=qj*qk;
            inst_xform[0]  = (ww+ii-jj-kk)*sx; inst_xform[1]  = 2*(ij+wk)*sx;     inst_xform[2]  = 2*(ik-wj)*sx;     inst_xform[3]  = 0;
            inst_xform[4]  = 2*(ij-wk)*sy;      inst_xform[5]  = (ww-ii+jj-kk)*sy; inst_xform[6]  = 2*(jk+wi)*sy;     inst_xform[7]  = 0;
            inst_xform[8]  = 2*(ik+wj)*sz;      inst_xform[9]  = 2*(jk-wi)*sz;     inst_xform[10] = (ww-ii-jj+kk)*sz; inst_xform[11] = 0;
            inst_xform[12] = tx;                 inst_xform[13] = ty;               inst_xform[14] = tz;               inst_xform[15] = 1;

            nanousd_mul_m4d(inst_xform, instancer_world, &inst_worlds[(size_t)inst * 16]);
        }

        /* Emit one SceneInstanceBatch per (proto, sub-mesh). Iterate via
         * pidx_inst_indices[pidx_inst_offsets[pidx]..pidx_inst_offsets[pidx+1])
         * so each batch sees only its own instances — total work is
         * O(N × S_avg), matching the legacy per-clone path's complexity. */
        int pi_emitted = 0;
        int pi_culled  = 0;
        int pi_considered = 0;
        int pi_batches_added = 0;
        int pi_abort = 0;
        for (int pidx = 0; pidx < n_protos && !pi_abort; pidx++) {
            if (proto_counts[pidx] == 0) continue;
            int inst_lo = pidx_inst_offsets[pidx];
            int inst_hi = pidx_inst_offsets[pidx + 1];
            int inst_count_for_pidx = inst_hi - inst_lo;
            if (inst_count_for_pidx <= 0) continue;
            const int* inst_list = &pidx_inst_indices[inst_lo];

            for (int sub = 0; sub < proto_counts[pidx] && !pi_abort; sub++) {
                int src_mesh_idx = proto_lists[pidx][sub];
                if (src_mesh_idx < 0 || src_mesh_idx >= mesh_idx) continue;
                SceneMesh* proto_m = &scene->meshes[src_mesh_idx];

                /* Lazy proto-local bounds cache (lifted from the legacy path
                 * but applied even in no-cull mode so scene bounds get
                 * expanded via the cheap world_bounds_from_local path rather
                 * than per-vertex transform per instance). */
                PiProtoLocalBounds* pb = NULL;
                if (proto_bounds &&
                    src_mesh_idx >= 0 && src_mesh_idx < proto_bound_count) {
                    pb = &proto_bounds[src_mesh_idx];
                    if (pb->state == 0) {
                        if (mesh_local_bounds(proto_m, pb->mn, pb->mx)) pb->state = 1;
                        else                                            pb->state = 2;
                    }
                }

                /* Survivor count for this (pidx, sub). Frustum cull is the
                 * only thing that can reject; in no-cull mode kept ==
                 * inst_count_for_pidx and we skip the per-instance check. */
                uint32_t kept = (uint32_t)inst_count_for_pidx;
                pi_considered += inst_count_for_pidx;
                if (pi_frustum_cull && pb) {
                    kept = 0;
                    for (int li = 0; li < inst_count_for_pidx; li++) {
                        int inst = inst_list[li];
                        if (pb->state == 2) { pi_culled++; continue; }
                        if (pb->state == 1) {
                            const double* iw = &inst_worlds[(size_t)inst * 16];
                            float mn[3], mx[3];
                            world_bounds_from_local(iw, pb->mn, pb->mx, mn, mx);
                            if (pi_cull_pad > 0.0f) {
                                for (int k = 0; k < 3; k++) {
                                    mn[k] -= pi_cull_pad;
                                    mx[k] += pi_cull_pad;
                                }
                            }
                            if (bounds_is_usable(mn, mx) &&
                                !scene_aabb_visible_in_any_pi_frustum(mn, mx)) {
                                pi_culled++;
                                continue;
                            }
                        }
                        kept++;
                    }
                }
                if (kept == 0) continue;

                /* Reserve the transform slab + batch slot. */
                uint32_t base_off = (uint32_t)scene->npi_transforms;
                SceneInstanceTransform* slab = scene_pi_transforms_reserve(
                    scene, &pi_xform_cap, (uint64_t)kept);
                if (!slab) {
                    scene_fail(scene,
                               "Out of memory growing PI transform slab "
                               "for '%s'",
                               nanousd_path(prim));
                    pi_abort = 1;
                    break;
                }
                if (!scene_pi_batches_reserve(scene, &pi_batch_cap,
                                              scene->npi_batches + 1)) {
                    scene_fail(scene,
                               "Out of memory growing PI batch table for '%s'",
                               nanousd_path(prim));
                    pi_abort = 1;
                    break;
                }

                /* Second scan: pack survivors + expand scene bounds. */
                uint32_t w = 0;
                for (int li = 0; li < inst_count_for_pidx && w < kept; li++) {
                    int inst = inst_list[li];
                    const double* iw = &inst_worlds[(size_t)inst * 16];

                    if (pi_frustum_cull && pb) {
                        if (pb->state == 2) continue;
                        if (pb->state == 1) {
                            float mn[3], mx[3];
                            world_bounds_from_local(iw, pb->mn, pb->mx, mn, mx);
                            if (pi_cull_pad > 0.0f) {
                                for (int k = 0; k < 3; k++) {
                                    mn[k] -= pi_cull_pad;
                                    mx[k] += pi_cull_pad;
                                }
                            }
                            if (bounds_is_usable(mn, mx) &&
                                !scene_aabb_visible_in_any_pi_frustum(mn, mx)) {
                                continue;
                            }
                        }
                    }

                    /* Pack as 12-float column-major affine (drop the
                     * always-(0,0,0,1) fourth row). Matches Metal RT instance
                     * descriptor / VkTransformMatrixKHR layout. */
                    slab[w].m[0]  = (float)iw[0];
                    slab[w].m[1]  = (float)iw[1];
                    slab[w].m[2]  = (float)iw[2];
                    slab[w].m[3]  = (float)iw[4];
                    slab[w].m[4]  = (float)iw[5];
                    slab[w].m[5]  = (float)iw[6];
                    slab[w].m[6]  = (float)iw[8];
                    slab[w].m[7]  = (float)iw[9];
                    slab[w].m[8]  = (float)iw[10];
                    slab[w].m[9]  = (float)iw[12];
                    slab[w].m[10] = (float)iw[13];
                    slab[w].m[11] = (float)iw[14];

                    /* Expand scene world-space bounds. */
                    if (pb && pb->state == 1) {
                        float mn[3], mx[3];
                        world_bounds_from_local(iw, pb->mn, pb->mx, mn, mx);
                        scene_expand_bounds(scene, mn, mx);
                    } else {
                        /* Conservative per-vertex transform — only hit when
                         * proto bounds couldn't be derived (degenerate
                         * geometry). Rare; the cost is bounded by survivor
                         * count for those few cases. */
                        for (int v = 0; v < proto_m->nvertices; v++) {
                            float wp[3];
                            xform_point(iw, &proto_m->positions[v * 3], wp);
                            for (int k = 0; k < 3; k++) {
                                if (wp[k] < scene->bounds_min[k]) scene->bounds_min[k] = wp[k];
                                if (wp[k] > scene->bounds_max[k]) scene->bounds_max[k] = wp[k];
                            }
                        }
                    }
                    w++;
                }

                SceneInstanceBatch* b = &scene->pi_batches[scene->npi_batches++];
                b->prototype_mesh_idx     = src_mesh_idx;
                b->transform_offset       = base_off;
                b->transform_count        = kept;
                b->source_prim_idx        = i;
                b->material_or_binding_id = -1;
                b->source_kind            = SCENE_INSTANCE_SOURCE_POINT_INSTANCER;
                pi_emitted += (int)kept;
                pi_batches_added++;

                /* Periodic RSS guard. Bound by number of batches per PI
                 * rather than per-instance; the per-batch peak is dominated
                 * by the transform slab realloc, not per-instance work. */
                if (scene_check_rss_limit(scene, "PointInstancer batch")) {
                    pi_abort = 1;
                    break;
                }
            }
        }

        free(inst_worlds);
        free(pidx_inst_offsets);
        free(pidx_inst_indices);

        if (pi_emitted > 0 || pi_culled > 0 || pi_batches_added > 0) {
            pi_prims_seen++;
            pi_total_batches_emitted += pi_batches_added;
            pi_total_transforms_emitted += pi_emitted;
            pi_total_culled += pi_culled;
            fprintf(stderr,
                    "scene_load: PointInstancer '%s': compact %d batches, "
                    "%d transforms (%d considered, %d culled, "
                    "%d source instances, %d protos)\n",
                    nanousd_path(prim), pi_batches_added, pi_emitted,
                    pi_considered, pi_culled, n_instances, n_protos);
        }

        free(proto_bounds);
        free(proto_indices);
        free(positions);
        free(orientations);
        free(scales);
        for (int p = 0; p < n_protos; p++) free(proto_lists[p]);
        free(proto_lists);
        free(proto_counts);
        nanousd_freeprim(prim);
        if (scene->load_failed)
            break;
        }
    }

    if (pi_prims_seen > 0 || scene->npi_batches > 0) {
        uint64_t xform_bytes = scene->npi_transforms *
                               sizeof(SceneInstanceTransform);
        uint64_t batch_bytes = (uint64_t)scene->npi_batches *
                               sizeof(SceneInstanceBatch);
        fprintf(stderr,
                "scene_load: compact PI summary: %d PIs, %d batches, "
                "%llu transforms (%llu B), batch table %llu B, "
                "%d culled, pi_scene_mesh_clones=0\n",
                pi_prims_seen,
                scene->npi_batches,
                (unsigned long long)scene->npi_transforms,
                (unsigned long long)xform_bytes,
                (unsigned long long)batch_bytes,
                pi_total_culled);
        fprintf(stderr,
                "scene_load: nested PI summary: %d unique, %lld leaves, "
                "%d batches (%.2f MiB), skipped %d top-level + %d dup\n",
                s_nested_pi_unique, s_nested_pi_emitted_leaves,
                s_nested_pi_emitted_batches,
                (double)s_nested_pi_emitted_leaves *
                    sizeof(SceneInstanceTransform) / (1024.0 * 1024.0),
                s_nested_pi_skipped_toplevel, s_nested_pi_skipped_dup);
    }

    scene_check_rss_limit(scene, "PointInstancer expansion");

    proto_hash_free(&proto_mesh_table);

    /* Set actual mesh count (over-allocated above using nprims as upper bound) */
    scene->nmeshes = mesh_idx;

    if (n_unhandled_geom > 0) {
        fprintf(stderr,
                "scene_load: WARNING %d renderable geometry prims of unsupported "
                "type were DROPPED (not rendered), e.g. %s — extend the geometry "
                "loader to cover them\n",
                n_unhandled_geom, unhandled_geom_example);
    }
    if (getenv("NUSD_LOAD_TIMING")) {
        fprintf(stderr,
                "scene_load: geometry hides (USD-spec, not rendered) — navmesh=%d, "
                "visibility:invisible(own)=%d, visibility:invisible(inherited)=%d, "
                "purpose:guide/proxy=%d\n",
                n_hide_navmesh, n_hide_vis_own, n_hide_vis_inherited, n_hide_purpose);
    }
    if (scene->load_failed) {
        fprintf(stderr, "scene_load: failing load before render attach: %s\n",
                scene->load_error[0] ? scene->load_error
                                     : "fatal scene load error");
        path_prefix_list_free(&pi_proto_prefixes);
        scene_free(scene);
        return NULL;
    }

    if (scene->nmeshes == 0) {
        fprintf(stderr, "scene_load: no Mesh prims found in '%s' (%d prims total)\n",
                filepath, nprims);
        /* Return empty scene instead of NULL — caller may add meshes via add_mesh() */
    }

    /* ---- Extract scene lights (UsdLuxRectLight, UsdLuxDistantLight, UsdLuxSphereLight) ----
     *
     * Isaac/Omniverse assets usually author UsdLux fields as inputs:*,
     * while small debug fixtures and older USDs often use legacy bare names
     * such as `float intensity`. Support both so wrapper scenes light Metal,
     * Vulkan, and OpenGL the same way. */
    {
        int nlights = 0;
        for (int i = 0; i < nprims; i++) {
            NanousdPrim p = nanousd_prim(stage, i);
            if (!p) continue;
            if (nanousd_isactive(p)) {
                const char* tn = nanousd_typename(p);
                if (tn && (!strcmp(tn, "RectLight") ||
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
                int is_dist = (tn && !strcmp(tn, "DistantLight"));
                int is_sphere = (tn && !strcmp(tn, "SphereLight"));
                if (!is_rect && !is_dist && !is_sphere) { nanousd_freeprim(p); continue; }

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

                if (is_rect) {
                    L->kind = SCENE_LIGHT_RECT;
                    float width  = nanousd_attribf(p, "inputs:width",  &ok);
                    if (!ok) width = nanousd_attribf(p, "width", &ok);
                    if (!ok) width = 1.0f;
                    float height = nanousd_attribf(p, "inputs:height", &ok);
                    if (!ok) height = nanousd_attribf(p, "height", &ok);
                    if (!ok) height = 1.0f;
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
            fprintf(stderr, "scene_load: %d lights (rect+distant+sphere) extracted\n", li);
        }
    }

    /* ---- Extract UsdLuxDomeLight ----
     * Single dome only — first found wins, matching Hydra. The HDR path
     * stays in scene->dome_hdr_path; nu_attach_scene picks it up and
     * calls gpu_load_environment to install it as IBL. Without this the
     * renderer falls back to procedural sky. Port from Vulkan ade51f3. */
    {
        scene->dome_hdr_path[0] = '\0';
        scene->has_dome_light   = 0;
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
            scene->has_dome_light = 1;
            int ok;
            if (!nanousd_attribv3f(p, "inputs:color", scene->dome_color))
                nanousd_attribv3f(p, "color", scene->dome_color);
            const char* asset = nanousd_attribasset(p, "inputs:texture:file", &ok);
            if (ok && asset && asset[0]) {
                /* USD asset paths can be @relative@ or absolute; nanousd
                 * stores them already-resolved when possible. We treat
                 * non-absolute paths as relative to the USD file's parent. */
                const char* resolved = asset;
                char buf[512];
                if (asset[0] != '/' && filepath) {
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
            if (ok) scene->dome_intensity = intensity;
            float rot = nanousd_attribf(p, "inputs:rotation", &ok);
            if (ok) scene->dome_rotation_y = rot;
            nanousd_freeprim(p);
            if (scene->dome_hdr_path[0]) {
                fprintf(stderr,
                        "scene_load: DomeLight → %s (color=%.3f %.3f %.3f, intensity=%.3f)\n",
                        scene->dome_hdr_path,
                        scene->dome_color[0], scene->dome_color[1], scene->dome_color[2],
                        scene->dome_intensity);
            } else {
                fprintf(stderr,
                        "scene_load: textureless DomeLight (color=%.3f %.3f %.3f, intensity=%.3f)\n",
                        scene->dome_color[0], scene->dome_color[1], scene->dome_color[2],
                        scene->dome_intensity);
            }
            break;
        }

        /* No authored UsdLuxDomeLight, but Omniverse/Isaac scenes commonly
         * light the world via the RTX "sky" background
         * (omni:rtx:background:source:type = "sky") with no dome prim at all.
         * Without an ambient/IBL term the renderer lights surfaces by direct
         * lights only, so the interior reads dark/cold while OVRTX/OpenGL —
         * which apply the sky — look bright. Synthesize a textureless flat
         * dome (warm-neutral, modest intensity) so has_dome_light=1 makes
         * renderer.c install it as flat-dome ambient (shader flat_dome_rgb =
         * dome.rgb * dome.w). Fires ONLY when a sky background is authored, so
         * scenes with a real DomeLight (handled above) and plain test scenes
         * with no sky hint are untouched / byte-identical. */
        if (!scene->has_dome_light) {
            int sky_authored = 0;
            for (int i = 0; i < nprims && !sky_authored; i++) {
                NanousdPrim p = nanousd_prim(stage, i);
                if (!p) continue;
                int ok = 0;
                const char* bg = nanousd_attrib_token(
                    p, "omni:rtx:background:source:type", &ok);
                if (ok && bg && !strcmp(bg, "sky")) sky_authored = 1;
                nanousd_freeprim(p);
            }
            if (sky_authored) {
                scene->has_dome_light = 1;
                scene->dome_color[0]  = 0.80f;
                scene->dome_color[1]  = 0.78f;
                scene->dome_color[2]  = 0.72f;  /* warm-neutral sky */
                scene->dome_intensity = 1.0f;
                scene->dome_hdr_path[0] = '\0';
                fprintf(stderr,
                    "scene_load: no DomeLight + RTX sky background -> "
                    "synthesized flat sky ambient (color=%.2f %.2f %.2f, "
                    "intensity=%.2f)\n",
                    scene->dome_color[0], scene->dome_color[1],
                    scene->dome_color[2], scene->dome_intensity);
            }
        }
    }

    /* ---- upAxis handling ----
     *
     * Preserve authored coordinates by default for OVRTX parity. The old
     * Z-up -> Y-up bake remains opt-in through NUSD_BAKE_ZUP_TO_YUP=1. */
    {
        int ok = 0;
        const char* up = nanousd_metadatas(stage, "upAxis", &ok);
        /* upAxis metadata: ok=%d means the value was found in the stage */

        /* Heuristic: if metadata is missing but Z range is < XY range and
         * Z_min ≈ 0, treat as Z-up (common for Omniverse / Isaac Sim assets). */
        int is_z_up = (ok && up && (up[0] == 'Z' || up[0] == 'z'));
        if (!is_z_up && !ok) {
            float dx = scene->bounds_max[0] - scene->bounds_min[0];
            float dy = scene->bounds_max[1] - scene->bounds_min[1];
            float dz = scene->bounds_max[2] - scene->bounds_min[2];
            float max_xy = dx > dy ? dx : dy;
            if (dz > 1e-3f && dz < max_xy * 0.8f && scene->bounds_min[2] > -dz * 0.1f) {
                fprintf(stderr, "scene_load: heuristic Z-up detection (dz=%.1f < max_xy=%.1f, z_min=%.1f)\n",
                        dz, max_xy, scene->bounds_min[2]);
                is_z_up = 1;
            }
        }

        const char* bake_env = getenv("NUSD_BAKE_ZUP_TO_YUP");
        int bake_zup_to_yup = bake_env && bake_env[0] && bake_env[0] != '0';

        const char* preserve = getenv("NUSD_PRESERVE_UPAXIS");
        if (preserve && *preserve && *preserve != '0') {
            bake_zup_to_yup = 0;
        }
        if (!bake_zup_to_yup)
            is_z_up = 0;

        if (is_z_up) {
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

            for (int c = 0; c < scene->ncurves; c++) {
                SceneCurve* curve = &scene->curves[c];
                double tmp[16];
                nanousd_mul_m4d(curve->world_xform, R, tmp);
                memcpy(curve->world_xform, tmp, sizeof(double) * 16);
                for (int v = 0; v < curve->nv; v++) {
                    float* p = &curve->cvs[v * 3];
                    float y = p[1];
                    p[1] = p[2];
                    p[2] = -y;
                }

                float omin[3], omax[3];
                memcpy(omin, curve->bounds_min, sizeof(float) * 3);
                memcpy(omax, curve->bounds_max, sizeof(float) * 3);
                curve->bounds_min[0] = omin[0];
                curve->bounds_min[1] = omin[2];
                curve->bounds_min[2] = -omax[1];
                curve->bounds_max[0] = omax[0];
                curve->bounds_max[1] = omax[2];
                curve->bounds_max[2] = -omin[1];
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
            for (int c = 0; c < scene->ncurves; c++) {
                SceneCurve* curve = &scene->curves[c];
                for (int k = 0; k < 3; k++) {
                    if (curve->bounds_min[k] < scene->bounds_min[k])
                        scene->bounds_min[k] = curve->bounds_min[k];
                    if (curve->bounds_max[k] > scene->bounds_max[k])
                        scene->bounds_max[k] = curve->bounds_max[k];
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
        }
    }

    /* ---- metersPerUnit handling ---- */
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

            for (int c = 0; c < scene->ncurves; c++) {
                SceneCurve* curve = &scene->curves[c];
                for (int i = 0; i < 12; i++) curve->world_xform[i] *= mpu;
                curve->world_xform[12] *= mpu;
                curve->world_xform[13] *= mpu;
                curve->world_xform[14] *= mpu;
                for (int v = 0; v < curve->nv; v++) {
                    curve->cvs[v * 3 + 0] = (float)(curve->cvs[v * 3 + 0] * mpu);
                    curve->cvs[v * 3 + 1] = (float)(curve->cvs[v * 3 + 1] * mpu);
                    curve->cvs[v * 3 + 2] = (float)(curve->cvs[v * 3 + 2] * mpu);
                    curve->widths[v] = (float)(curve->widths[v] * mpu);
                }
                for (int k = 0; k < 3; k++) {
                    curve->bounds_min[k] = (float)(curve->bounds_min[k] * mpu);
                    curve->bounds_max[k] = (float)(curve->bounds_max[k] * mpu);
                }
            }

            for (int li = 0; li < scene->nlights; li++) {
                SceneLight* L = &scene->lights[li];
                for (int k = 0; k < 3; k++) {
                    L->position[k] = (float)(L->position[k] * mpu);
                    L->u_axis[k] = (float)(L->u_axis[k] * mpu);
                    L->v_axis[k] = (float)(L->v_axis[k] * mpu);
                }
            }

            for (int k = 0; k < 3; k++) {
                scene->bounds_min[k] = (float)(scene->bounds_min[k] * mpu);
                scene->bounds_max[k] = (float)(scene->bounds_max[k] * mpu);
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

                /* DIAG: distance distribution + sample paths (set NUSD_GEO_DIAG=1) */
                if (getenv("NUSD_GEO_DIAG")) {
                    int bins[6] = {0,0,0,0,0,0}; float maxd = 0.0f;
                    for (int m = 0; m < scene->nmeshes; m++) {
                        float dx = cx[m*3+0]-median[0], dy = cx[m*3+1]-median[1], dz = cx[m*3+2]-median[2];
                        float dd = sqrtf(dx*dx+dy*dy+dz*dz); if (dd > maxd) maxd = dd;
                        int b = dd<10?0:dd<100?1:dd<1000?2:dd<5000?3:dd<50000?4:5; bins[b]++;
                    }
                    fprintf(stderr, "scene_load: DIST DIAG median=(%.1f %.1f %.1f) maxd=%.0f bins[<10=%d <100=%d <1k=%d <5k=%d <50k=%d >=50k=%d] of %d\n",
                            median[0],median[1],median[2], maxd, bins[0],bins[1],bins[2],bins[3],bins[4],bins[5], scene->nmeshes);
                    int pn = 0;
                    for (int m = 0; m < scene->nmeshes && pn < 8; m++) {
                        float dx=cx[m*3]-median[0],dy=cx[m*3+1]-median[1],dz=cx[m*3+2]-median[2];
                        if (sqrtf(dx*dx+dy*dy+dz*dz) < 10.0f) {
                            fprintf(stderr, "scene_load:   NEAR c=(%.1f %.1f %.1f) v=%d proto=%d %s\n",
                                    cx[m*3],cx[m*3+1],cx[m*3+2], scene->meshes[m].nvertices, scene->meshes[m].is_proto_only, scene->meshes[m].path);
                            pn++;
                        }
                    }
                    pn = 0;
                    for (int m = 0; m < scene->nmeshes && pn < 8; m++) {
                        float dx=cx[m*3]-median[0],dy=cx[m*3+1]-median[1],dz=cx[m*3+2]-median[2];
                        float dd=sqrtf(dx*dx+dy*dy+dz*dz);
                        if (dd > 1000.0f) {
                            fprintf(stderr, "scene_load:   FAR  c=(%.0f %.0f %.0f) d=%.0f proto=%d %s\n",
                                    cx[m*3],cx[m*3+1],cx[m*3+2], dd, scene->meshes[m].is_proto_only, scene->meshes[m].path);
                            pn++;
                        }
                    }
                }

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

    /* ---- Load BasisCurves prims (Phase 11.A) ---- */
    {
        const char* curves_env = getenv("NUSD_DISABLE_CURVES");
        if (curves_env && curves_env[0] &&
            strcmp(curves_env, "0") != 0 &&
            strcmp(curves_env, "false") != 0 &&
            strcmp(curves_env, "FALSE") != 0) {
            fprintf(stderr, "scene_load: NUSD_DISABLE_CURVES=1 — skipping BasisCurves extraction\n");
        } else {
            scene_load_curves(stage, scene, arena);
            scene_check_rss_limit(scene, "BasisCurves extraction");
        }
    }

    if (scene->load_failed) {
        fprintf(stderr, "scene_load: failing load before render attach: %s\n",
                scene->load_error[0] ? scene->load_error
                                     : "fatal scene load error");
        path_prefix_list_free(&pi_proto_prefixes);
        scene_free(scene);
        return NULL;
    }

lazy_postprocess:;
    double t1 = get_time_sec();
    if (scene->nmeshes > 0 || scene->ncurves > 0) {
        fprintf(stderr, "scene_load: '%s' loaded in %.1f ms\n"
                        "  %d meshes, %d vertices, %d indices, %d curves\n"
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
    if (geometry_budget_truncated) {
        fprintf(stderr,
                "scene_load: geometry extraction truncated by budget "
                "(loaded unique geometry %lld vertices, %lld indices; "
                "limits vertices=%lld indices=%lld)\n",
                geometry_verts_loaded, geometry_indices_loaded,
                geometry_vertex_limit, geometry_index_limit);
    }
    if (semantic_flat_records > 0 ||
        scene_no_cull_all_geometry_requested() ||
        scene_flat_native_instance_traversal_enabled()) {
        long long maxrss = scene_maxrss_bytes();
        fprintf(stderr,
                "scene_load: semantic flat traversal records=%d time=%.1f ms "
                "maxrss=%.1f MiB\n",
                semantic_flat_records, semantic_flat_traversal_ms,
                maxrss > 0 ? (double)maxrss / (1024.0 * 1024.0) : 0.0);
    }

    #undef PRIM_IS_WANTED
    path_prefix_list_free(&pi_proto_prefixes);
    return scene;
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
     * own it. Borrowed stages from scene_load_from_stage are owned by the
     * caller and must outlive the Scene. */
    if (scene->_stage && scene->_owns_stage) {
        nanousd_close((NanousdStage)scene->_stage);
    }
    scene->_stage = NULL;

    /* Destroy the arena (frees all mesh data allocated from it) */
    if (scene->_arena) {
        arena_destroy((Arena*)scene->_arena);
        free(scene->_arena);
        scene->_arena = NULL;
    }

    free(scene->meshes);
    scene->meshes = NULL;

    free(scene->pi_batches);
    scene->pi_batches = NULL;
    scene->npi_batches = 0;
    free(scene->pi_transforms);
    scene->pi_transforms = NULL;
    scene->npi_transforms = 0;

    free(scene);
}
