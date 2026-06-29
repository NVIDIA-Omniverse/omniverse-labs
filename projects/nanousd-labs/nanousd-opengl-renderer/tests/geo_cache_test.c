// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* geo_cache_test.c — verify the NUSD_GEO_CACHE geometry-cache round-trip is
 * byte-exact. scene_load() parses the USD and writes <usd>.nzgeo.gl;
 * geo_cache_try_load() reads it straight back. The two Scenes must be
 * geometrically byte-identical — the cache losslessly round-trips the parsed
 * geometry. No GL context required.
 *
 * prototype_idx is intentionally NOT compared: cook-time content dedup
 * rewrites it (folded meshes point at a content leader). The geometry-array
 * byte-compare still holds — a folded mesh's geometry equals its leader's.
 *
 * Usage: geo_cache_test <scene.usd>
 */
#include "scene.h"
#include "geo_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else         { fprintf(stderr, "ok:   %s\n", msg); } \
} while (0)

/* Byte-compare two arrays that may both be NULL (NULL == NULL passes). */
static int arr_eq(const void* a, const void* b, size_t n)
{
    if ((a == NULL) != (b == NULL)) return 0;
    if (!a || n == 0) return 1;
    return memcmp(a, b, n) == 0;
}

/* True when index buffers a and b describe the same triangles with the same
 * winding — equal up to per-triangle cyclic rotation. The meshoptimizer index
 * codec canonicalizes each triangle's vertex order, so cache-loaded indices
 * are cyclic rotations of the parse-loaded ones. Cyclic rotation preserves
 * winding (front/back face) and is render-invariant under smooth shading; a
 * pair-swap reverses winding and is correctly rejected here. */
static int idx_winding_eq(const uint32_t* a, const uint32_t* b, int n)
{
    if ((a == NULL) != (b == NULL)) return 0;
    if (!a) return 1;
    if (n < 0 || n % 3 != 0)
        return n >= 0 && memcmp(a, b, (size_t)n * sizeof(uint32_t)) == 0;
    for (int t = 0; t < n; t += 3) {
        uint32_t a0 = a[t], a1 = a[t+1], a2 = a[t+2];
        uint32_t b0 = b[t], b1 = b[t+1], b2 = b[t+2];
        int rot = (b0 == a0 && b1 == a1 && b2 == a2) ||
                  (b0 == a1 && b1 == a2 && b2 == a0) ||
                  (b0 == a2 && b1 == a0 && b2 == a1);
        if (!rot) return 0;
    }
    return 1;
}

/* Winding-canonical triangle key: rotate so the smallest vertex is first.
 * Invariant under cyclic rotation (render-preserving), distinct under a
 * pair-swap (winding flip) — so sorting these keys turns a triangle set into
 * a comparable multiset. */
typedef struct { uint32_t a, b, c; } Tri;

static Tri tri_canon(uint32_t v0, uint32_t v1, uint32_t v2)
{
    Tri t;
    if (v0 <= v1 && v0 <= v2)      { t.a = v0; t.b = v1; t.c = v2; }
    else if (v1 <= v0 && v1 <= v2) { t.a = v1; t.b = v2; t.c = v0; }
    else                           { t.a = v2; t.b = v0; t.c = v1; }
    return t;
}

static int tri_cmp(const void* x, const void* y)
{
    const Tri* p = (const Tri*)x;
    const Tri* q = (const Tri*)y;
    if (p->a != q->a) return p->a < q->a ? -1 : 1;
    if (p->b != q->b) return p->b < q->b ? -1 : 1;
    if (p->c != q->c) return p->c < q->c ? -1 : 1;
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: geo_cache_test <scene.usd>\n");
        return 2;
    }
    const char* usd = argv[1];
    setenv("NUSD_GEO_CACHE", "1", 1);

    /* Drop any stale cache so the parse below is a guaranteed MISS + write. */
    char cpath[2048];
    if (geo_cache_path_for(usd, cpath, sizeof cpath) == 0) remove(cpath);

    Scene* s1 = scene_load(usd);            /* MISS: parse USD, write cache */
    CHECK(s1 != NULL, "cold parse loaded the scene");
    if (!s1) return 1;

    Scene* s2 = geo_cache_try_load(usd);    /* warm: load straight from cache */
    CHECK(s2 != NULL, "geo_cache_try_load returned a cached scene (HIT)");
    if (!s2) { scene_free(s1); return 1; }

    CHECK(s1->nmeshes == s2->nmeshes, "nmeshes match");
    CHECK(s1->up_axis == s2->up_axis, "up_axis match");
    CHECK(memcmp(s1->bounds_min, s2->bounds_min, sizeof s1->bounds_min) == 0,
          "scene bounds_min match");
    CHECK(memcmp(s1->bounds_max, s2->bounds_max, sizeof s1->bounds_max) == 0,
          "scene bounds_max match");
    CHECK(s1->has_authored_light == s2->has_authored_light,
          "has_authored_light match");
    CHECK(strcmp(s1->dome_hdr_path, s2->dome_hdr_path) == 0,
          "DomeLight HDR path match");
    CHECK(s1->dome_intensity == s2->dome_intensity,
          "DomeLight intensity match");
    CHECK(s1->dome_rotation_y == s2->dome_rotation_y,
          "DomeLight rotation match");

    int n = (s1->nmeshes < s2->nmeshes) ? s1->nmeshes : s2->nmeshes;
    int mismatch = 0, folded = 0;
    for (int i = 0; i < n; i++) {
        const SceneMesh* a = &s1->meshes[i];
        const SceneMesh* b = &s2->meshes[i];
        if (b->prototype_idx != i) folded++;   /* content-deduped or instanced */

        size_t pos = (size_t)a->nvertices * 3u * sizeof(float);
        size_t uv  = (size_t)a->nvertices * 2u * sizeof(float);
        const char* bad = NULL;
        if      (a->nvertices != b->nvertices)            bad = "nvertices";
        else if (a->nindices  != b->nindices)             bad = "nindices";
        else if (a->material_index != b->material_index)  bad = "material_index";
        else if (a->is_proto_only  != b->is_proto_only)   bad = "is_proto_only";
        else if (a->has_display_color != b->has_display_color) bad = "has_display_color";
        else if (memcmp(a->world_xform, b->world_xform, sizeof a->world_xform)) bad = "world_xform";
        else if (memcmp(a->bounds_min, b->bounds_min, sizeof a->bounds_min)) bad = "bounds_min";
        else if (memcmp(a->bounds_max, b->bounds_max, sizeof a->bounds_max)) bad = "bounds_max";
        else if (memcmp(a->local_bounds_min, b->local_bounds_min, sizeof a->local_bounds_min)) bad = "local_bounds_min";
        else if (memcmp(a->local_bounds_max, b->local_bounds_max, sizeof a->local_bounds_max)) bad = "local_bounds_max";
        else if (memcmp(a->display_color, b->display_color, sizeof a->display_color)) bad = "display_color";
        else if (!arr_eq(a->positions, b->positions, pos)) bad = "positions";
        else if (!arr_eq(a->normals,   b->normals,   pos)) bad = "normals";
        else if (!arr_eq(a->colors,    b->colors,    pos)) bad = "colors";
        else if (!arr_eq(a->texcoords, b->texcoords, uv))  bad = "texcoords";
        /* indices: winding-equivalent — the meshopt index codec cyclically
         * rotates each triangle (render-invariant), so a byte compare is
         * wrong; idx_winding_eq accepts rotation but rejects winding flips. */
        else if (!idx_winding_eq(a->indices, b->indices, a->nindices)) bad = "indices";
        if (bad) {
            if (mismatch < 8)
                fprintf(stderr, "  mesh[%d] mismatch: %s  (nv %d ni %d)\n",
                        i, bad, a->nvertices, a->nindices);
            mismatch++;
        }
    }
    CHECK(mismatch == 0,
          "every mesh's geometry is byte-identical (cache == parse)");
    fprintf(stderr,
            "geo_cache_test: %d meshes compared, %d mismatches, "
            "%d folded/instanced in cache\n", n, mismatch, folded);

    /* ---- meshlet section (mesh-shader-native compact layout) ---- */
    /* The cold parse (s1) carries no meshlets — only the warm cache (s2)
     * does. Verify the compact layout (per-meshlet vertex table + 8-bit
     * micro-indices) is structurally sound and that each mesh's meshlets
     * re-partition exactly that mesh's triangle set. */
    CHECK(s2->nmeshlets > 0, "warm cache built meshlets");
    CHECK(s1->nmeshlets == 0, "cold parse has no meshlets (cache-only)");

    long ml_vtx_sum = 0, ml_tri_sum = 0;
    int  ml_struct_bad = 0;
    for (int j = 0; j < s2->nmeshlets; j++) {
        const SceneMeshlet* ml = &s2->meshlets[j];
        if (ml->vertex_count == 0 || ml->vertex_count > 64 ||  /* meshopt cap */
            ml->triangle_count == 0 ||
            (long)ml->vertex_offset + ml->vertex_count > s2->nmeshlet_vertices ||
            (long)ml->triangle_offset + (long)ml->triangle_count * 3
                > s2->nmeshlet_triangles)
            ml_struct_bad++;
        ml_vtx_sum += ml->vertex_count;
        ml_tri_sum += (long)ml->triangle_count * 3;
    }
    CHECK(ml_struct_bad == 0,
          "every meshlet has valid in-range vertex/triangle spans");
    CHECK(ml_vtx_sum == s2->nmeshlet_vertices,
          "meshlet vertex_counts sum to nmeshlet_vertices");
    CHECK(ml_tri_sum == s2->nmeshlet_triangles,
          "meshlet micro-index counts sum to nmeshlet_triangles");

    /* Every micro-index must be a valid local index into its vertex table. */
    int micro_oor = 0;
    for (int j = 0; j < s2->nmeshlets; j++) {
        const SceneMeshlet* ml = &s2->meshlets[j];
        uint32_t n = ml->triangle_count * 3u;
        for (uint32_t e = 0; e < n; e++)
            if (s2->meshlet_triangles[ml->triangle_offset + e] >= ml->vertex_count) {
                micro_oor++;
                break;
            }
    }
    CHECK(micro_oor == 0, "every micro-index is in range of its vertex table");

    int meshed_meshes = 0, tri_mismatch = 0, vtx_oor_meshes = 0;
    for (int i = 0; i < s2->nmeshes; i++) {
        const SceneMesh* m = &s2->meshes[i];
        if (m->meshlet_count == 0) continue;
        meshed_meshes++;

        long sum = 0;
        for (uint32_t k = 0; k < m->meshlet_count; k++)
            sum += (long)s2->meshlets[m->meshlet_offset + k].triangle_count * 3;
        if (sum != m->nindices) { tri_mismatch++; continue; }
        if (m->nindices <= 0 || m->nindices % 3 != 0 || !m->indices ||
            !s2->meshlet_vertices || !s2->meshlet_triangles) continue;

        int  ntris = m->nindices / 3;
        Tri* mt = (Tri*)malloc((size_t)ntris * sizeof(Tri));
        Tri* lt = (Tri*)malloc((size_t)ntris * sizeof(Tri));
        if (!mt || !lt) { free(mt); free(lt); continue; }

        for (int t = 0; t < ntris; t++)
            mt[t] = tri_canon(m->indices[t*3], m->indices[t*3+1],
                              m->indices[t*3+2]);

        int lo = 0, oor = 0;
        for (uint32_t k = 0; k < m->meshlet_count; k++) {
            const SceneMeshlet* ml = &s2->meshlets[m->meshlet_offset + k];
            const uint32_t*      vt = s2->meshlet_vertices  + ml->vertex_offset;
            const unsigned char* mi = s2->meshlet_triangles + ml->triangle_offset;
            for (uint32_t t = 0; t < ml->triangle_count; t++) {
                uint32_t g0 = vt[mi[t*3+0]];
                uint32_t g1 = vt[mi[t*3+1]];
                uint32_t g2 = vt[mi[t*3+2]];
                if (g0 >= (uint32_t)m->nvertices ||
                    g1 >= (uint32_t)m->nvertices ||
                    g2 >= (uint32_t)m->nvertices) oor++;
                if (lo < ntris) lt[lo] = tri_canon(g0, g1, g2);
                lo++;
            }
        }
        if (oor) vtx_oor_meshes++;
        if (oor || lo != ntris) { tri_mismatch++; free(mt); free(lt); continue; }

        qsort(mt, (size_t)ntris, sizeof(Tri), tri_cmp);
        qsort(lt, (size_t)ntris, sizeof(Tri), tri_cmp);
        if (memcmp(mt, lt, (size_t)ntris * sizeof(Tri)) != 0) tri_mismatch++;
        free(mt);
        free(lt);
    }
    CHECK(meshed_meshes > 0, "at least one mesh carries meshlets");
    CHECK(vtx_oor_meshes == 0,
          "no meshlet vertex exceeds its mesh's vertex count");
    CHECK(tri_mismatch == 0,
          "every meshed mesh's meshlets re-partition its exact triangle set");
    fprintf(stderr,
            "geo_cache_test: meshlets — %d records, %d vertices, "
            "%d micro-indices, %d meshes meshed\n",
            s2->nmeshlets, s2->nmeshlet_vertices, s2->nmeshlet_triangles,
            meshed_meshes);

    scene_free(s1);
    scene_free(s2);

    if (g_fail == 0) {
        fprintf(stderr, "PASS: geo_cache round-trip byte-exact\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d check(s) failed\n", g_fail);
    return 1;
}
