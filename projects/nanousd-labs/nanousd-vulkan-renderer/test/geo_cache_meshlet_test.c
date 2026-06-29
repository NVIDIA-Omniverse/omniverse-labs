// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* geo_cache_meshlet_test.c — verify the on-disk geometry cache round-trips.
 *
 * Parses a USD scene cold (writing the cache), warm-loads it, and asserts:
 * (1) the geometry round-trips byte-exact — every vertex attribute identical,
 *     indices winding-equivalent (the meshopt index codec rotates triangles);
 * (2) the meshlets, which geo_cache_try_load() compacts at load time into the
 *     mesh-shader-native layout (per-meshlet vertex table + 8-bit
 *     micro-indices), are structurally sound and re-partition each mesh's
 *     exact triangle set.
 *
 * Pure CPU — no Vulkan / GPU. Usage: geo_cache_meshlet_test <scene.usd>
 */
#define _POSIX_C_SOURCE 200809L

#include "scene.h"
#include "geo_cache.h"
#include "material.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else         { fprintf(stderr, "ok:   %s\n", msg); } \
} while (0)

/* Winding-canonical triangle key: rotate so the smallest vertex is first.
 * Invariant under cyclic rotation (the meshopt index codec rotates each
 * triangle), distinct under a pair-swap (winding flip) — so a sorted multiset
 * of these keys compares two triangle sets for winding-aware equality. */
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

/* Byte-compare two arrays that may both be NULL (NULL == NULL passes). */
static int arr_eq(const void* a, const void* b, size_t n)
{
    if ((a == NULL) != (b == NULL)) return 0;
    if (!a || n == 0) return 1;
    return memcmp(a, b, n) == 0;
}

/* True when two index buffers describe the same triangles with the same
 * winding — equal up to per-triangle cyclic rotation. The meshopt index codec
 * canonicalizes each triangle's vertex order, so warm-cache indices are cyclic
 * rotations of the cold-parse ones; rotation preserves winding, a pair-swap
 * (rejected here) flips it. */
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

static int materials_eq(const MaterialCollection* a, const MaterialCollection* b)
{
    if ((a == NULL) != (b == NULL)) return 0;
    if (!a) return 1;
    if (a->nmaterials != b->nmaterials || a->ntextures != b->ntextures)
        return 0;
    for (int i = 0; i < a->nmaterials; i++) {
        const SceneMaterial* ma = &a->materials[i];
        const SceneMaterial* mb = &b->materials[i];
        if (memcmp(&ma->params, &mb->params, sizeof(MaterialParams)) != 0)
            return 0;
        if (strcmp(ma->name, mb->name) != 0 ||
            strcmp(ma->prim_path, mb->prim_path) != 0 ||
            ma->shader_index != mb->shader_index)
            return 0;
    }
    for (int i = 0; i < a->ntextures; i++) {
        const MaterialTexture* ta = &a->textures[i];
        const MaterialTexture* tb = &b->textures[i];
        if (ta->width != tb->width || ta->height != tb->height ||
            ta->udim_cols != tb->udim_cols || ta->udim_rows != tb->udim_rows ||
            ta->is_srgb != tb->is_srgb || strcmp(ta->path, tb->path) != 0)
            return 0;
        size_t bytes = (ta->pixels && ta->width > 0 && ta->height > 0)
            ? (size_t)ta->width * (size_t)ta->height * 4u : 0u;
        if (!arr_eq(ta->pixels, tb->pixels, bytes)) return 0;
    }
    return 1;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: geo_cache_meshlet_test <scene.usd>\n");
        return 2;
    }
    const char* usd = argv[1];
    setenv("NUSD_GEO_CACHE", "1", 1);

    /* Drop any stale cache so the parse below is a guaranteed MISS + write. */
    char cpath[2048];
    if (geo_cache_path_for(usd, cpath, sizeof cpath) == 0) remove(cpath);

    Scene* s1 = scene_load(usd);             /* MISS: parse USD, write cache */
    CHECK(s1 != NULL, "cold parse loaded the scene");
    if (!s1) return 1;

    Scene* s = geo_cache_try_load(usd, 1);   /* warm: load straight from cache */
    CHECK(s != NULL, "geo_cache_try_load returned a cached scene (HIT)");
    if (!s) { scene_free(s1); return 1; }

    /* ---- Geometry round-trip: the warm cache must reproduce the cold parse
     * exactly. prototype_idx is not compared — cook-time content dedup
     * rewrites it (a folded mesh's geometry still equals its leader's). ---- */
    CHECK(s1->nmeshes == s->nmeshes, "nmeshes match");
    {
        int n = s1->nmeshes < s->nmeshes ? s1->nmeshes : s->nmeshes;
        int mismatch = 0, folded = 0;
        for (int i = 0; i < n; i++) {
            const SceneMesh* a = &s1->meshes[i];
            const SceneMesh* b = &s->meshes[i];
            if (b->prototype_idx != i) folded++;
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
            else if (!idx_winding_eq(a->indices, b->indices, a->nindices)) bad = "indices";
            if (bad) {
                if (mismatch < 8)
                    fprintf(stderr, "  mesh[%d] mismatch: %s (nv %d ni %d)\n",
                            i, bad, a->nvertices, a->nindices);
                mismatch++;
            }
        }
        CHECK(mismatch == 0,
              "every mesh's geometry is byte-identical (cache == parse)");
        fprintf(stderr, "geo_cache_meshlet_test: %d meshes compared, "
                "%d mismatches, %d folded/instanced\n", n, mismatch, folded);
    }

    CHECK(materials_eq((const MaterialCollection*)s1->materials,
                       (const MaterialCollection*)s->materials),
          "material params and texture pixels round-trip");

    CHECK(s->nmeshlets > 0, "warm cache built meshlets");

    /* Structural: every meshlet's vertex/triangle spans are in range, and the
     * per-meshlet counts sum to the scene totals. */
    long ml_vtx_sum = 0, ml_tri_sum = 0;
    int  ml_struct_bad = 0;
    for (int j = 0; j < s->nmeshlets; j++) {
        const SceneMeshlet* ml = &s->meshlets[j];
        if (ml->vertex_count == 0 || ml->vertex_count > 64 ||  /* meshopt cap */
            ml->triangle_count == 0 ||
            (long)ml->vertex_offset + ml->vertex_count > s->nmeshlet_vertices ||
            (long)ml->triangle_offset + (long)ml->triangle_count * 3
                > s->nmeshlet_triangles)
            ml_struct_bad++;
        ml_vtx_sum += ml->vertex_count;
        ml_tri_sum += (long)ml->triangle_count * 3;
    }
    CHECK(ml_struct_bad == 0,
          "every meshlet has valid in-range vertex/triangle spans");
    CHECK(ml_vtx_sum == s->nmeshlet_vertices,
          "meshlet vertex_counts sum to nmeshlet_vertices");
    CHECK(ml_tri_sum == s->nmeshlet_triangles,
          "meshlet micro-index counts sum to nmeshlet_triangles");

    /* Every micro-index must be a valid local index into its vertex table. */
    int micro_oor = 0;
    for (int j = 0; j < s->nmeshlets; j++) {
        const SceneMeshlet* ml = &s->meshlets[j];
        uint32_t n = ml->triangle_count * 3u;
        for (uint32_t e = 0; e < n; e++)
            if (s->meshlet_triangles[ml->triangle_offset + e] >= ml->vertex_count) {
                micro_oor++;
                break;
            }
    }
    CHECK(micro_oor == 0, "every micro-index is in range of its vertex table");

    /* Per mesh: its meshlets must re-partition exactly that mesh's triangle
     * set — reconstruct each triangle through the vertex table and compare
     * the winding-canonical multisets. */
    int meshed = 0, tri_mismatch = 0, vtx_oor_meshes = 0;
    for (int i = 0; i < s->nmeshes; i++) {
        const SceneMesh* m = &s->meshes[i];
        if (m->meshlet_count == 0) continue;
        meshed++;

        long sum = 0;
        for (uint32_t k = 0; k < m->meshlet_count; k++)
            sum += (long)s->meshlets[m->meshlet_offset + k].triangle_count * 3;
        if (sum != m->nindices) { tri_mismatch++; continue; }
        if (m->nindices <= 0 || m->nindices % 3 != 0 || !m->indices ||
            !s->meshlet_vertices || !s->meshlet_triangles) continue;

        int  ntris = m->nindices / 3;
        Tri* mt = (Tri*)malloc((size_t)ntris * sizeof(Tri));
        Tri* lt = (Tri*)malloc((size_t)ntris * sizeof(Tri));
        if (!mt || !lt) { free(mt); free(lt); continue; }

        for (int t = 0; t < ntris; t++)
            mt[t] = tri_canon(m->indices[t*3], m->indices[t*3+1],
                              m->indices[t*3+2]);

        int lo = 0, oor = 0;
        for (uint32_t k = 0; k < m->meshlet_count; k++) {
            const SceneMeshlet* ml = &s->meshlets[m->meshlet_offset + k];
            const uint32_t*      vt = s->meshlet_vertices  + ml->vertex_offset;
            const unsigned char* mi = s->meshlet_triangles + ml->triangle_offset;
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
    CHECK(meshed > 0, "at least one mesh carries meshlets");
    CHECK(vtx_oor_meshes == 0,
          "no meshlet vertex exceeds its mesh's vertex count");
    CHECK(tri_mismatch == 0,
          "every meshed mesh's meshlets re-partition its exact triangle set");
    fprintf(stderr,
            "geo_cache_meshlet_test: %d meshlets, %d vertices, "
            "%d micro-indices, %d meshes meshed\n",
            s->nmeshlets, s->nmeshlet_vertices, s->nmeshlet_triangles, meshed);

    scene_free(s1);
    scene_free(s);

    if (g_fail == 0) {
        fprintf(stderr, "PASS: geo_cache geometry + meshlet round-trip\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d check(s) failed\n", g_fail);
    return 1;
}
