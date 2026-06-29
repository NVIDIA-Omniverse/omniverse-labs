// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * geo_cache.c — on-disk geometry cache for the nanousd OpenGL renderer.
 *
 * Writes a fully-built geometry-only Scene to a sidecar <usd>.nzgeo.gl file
 * using meshoptimizer's lossless vertex/index codecs, and reloads it without
 * re-parsing USD. Gated by NUSD_GEO_CACHE=1.
 *
 * Ported from nanousd-vulkan-renderer/src/geo_cache.c. This variant carries
 * the OpenGL Scene's up_axis and DomeLight metadata. NZGO v2 serializes
 * meshoptimizer meshlets (built
 * per prototype at cook time) and bakes content-hash dedup: meshes with
 * byte-identical geometry, material_index, and displayColor data fold onto
 * one prototype, so a warm load stores each unique geometry (and its meshlets)
 * once.
 *
 * Distinct magic ("NZGO") + file suffix (".nzgeo.gl") so the cache never
 * collides with the Vulkan renderer's <usd>.nzgeo.
 *
 * Single-arch (x86-64 Linux): the record structs are written raw, so the
 * format is not endian/ABI portable. A magic / format-version / source
 * mtime+size mismatch makes geo_cache_try_load() return NULL and the caller
 * falls back to a normal USD parse.
 *
 * See docs/planning/MESHLET_GEOMETRY_CACHE_PLAN.md.
 */

#include "geo_cache.h"
#include "arena.h"

#include <meshoptimizer.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#define NZGO_FORMAT_VERSION 9u   /* v9: DomeLight color + presence metadata */

/* Meshlet limits — match the Vulkan renderer's geo_cache. */
#define GEO_MESHLET_MAX_VERTICES   64u
#define GEO_MESHLET_MAX_TRIANGLES  124u

/* Per-array blob kinds. */
enum {
    GEO_BLOB_NULL    = 0,   /* array pointer was NULL */
    GEO_BLOB_SHARED  = 1,   /* instance shares this array with its prototype */
    GEO_BLOB_RAW     = 2,   /* uncompressed bytes follow */
    GEO_BLOB_MESHOPT = 3    /* meshoptimizer-encoded bytes follow */
};

/* Defensive upper bounds so a corrupt header cannot drive a huge malloc. */
#define GEO_MAX_VERTICES  300000000
#define GEO_MAX_INDICES   900000000
#define GEO_MAX_BLOB      (1ull << 33)   /* 8 GiB */

typedef struct {
    char     magic[4];          /* "NZGO" */
    uint32_t format_version;
    int64_t  src_mtime;         /* source USD st_mtime — cache invalidation */
    int64_t  src_size;          /* source USD st_size  — cache invalidation */
    uint64_t total_geo_bytes;   /* decoded vertex+index bytes — sizes read arena */
    uint32_t nmeshes;
    int32_t  up_axis;           /* Scene.up_axis (0=X, 1=Y, 2=Z) */
    uint32_t nmeshlets;         /* total meshlet records (back-patched) */
    uint32_t nmeshlet_indices;  /* v3: total meshlet micro-index (uint8) entries */
    float    bounds_min[3];
    float    bounds_max[3];
    uint32_t nmeshlet_verts;    /* v3: total meshlet vertex-table entries */
    int32_t  has_authored_light;
    int32_t  has_dome_light;
    float    dome_color[3];
    float    dome_intensity;
    float    dome_rotation_y;
    char     dome_hdr_path[512];
    uint32_t reserved[1];
} GeoCacheHeader;

typedef struct {
    int32_t  nvertices;
    int32_t  nindices;
    double   world_xform[16];
    float    bounds_min[3];
    float    bounds_max[3];
    float    local_bounds_min[3];
    float    local_bounds_max[3];
    float    display_color[3];
    int32_t  has_display_color;
    int32_t  material_index;
    int32_t  prototype_idx;
    int32_t  is_proto_only;
    int32_t  own_index;         /* == record index; validated on load */
} GeoCacheMeshHdr;

/* ---------------------------------------------------------------- helpers */

int geo_cache_enabled(void)
{
    const char* e = getenv("NUSD_GEO_CACHE");
    if (!(e && e[0] && e[0] != '0')) return 0;

    /* Legacy scene-space mutations change cached world transforms. Keep the
     * persistent cache on the OVRTX/default authored-frame path only, so a
     * cache cooked under one camera space cannot be reused under another. */
    const char* bake = getenv("NUSD_BAKE_ZUP_TO_YUP");
    if (bake && bake[0] && bake[0] != '0') return 0;
    const char* units = getenv("NUSD_APPLY_METERS_PER_UNIT");
    if (units && units[0] && units[0] != '0') return 0;
    return 1;
}

int geo_cache_path_for(const char* usd_path, char* out_path, size_t out_cap)
{
    if (!usd_path || !out_path || out_cap == 0) return -1;
    int n = snprintf(out_path, out_cap, "%s.nzgeo.gl", usd_path);
    if (n < 0 || (size_t)n >= out_cap) return -1;
    return 0;
}

static int wr(FILE* f, const void* p, size_t n)
{
    if (n == 0) return 0;
    return fwrite(p, 1, n, f) == n ? 0 : -1;
}

/* Read cursor over the slurped .nzgeo.gl (iter 38). The whole cache is read
 * into one buffer up front; rd is a bounds-checked memcpy from it — that
 * removes the small fread calls from the warm load's Pass-1a, and lets decode
 * tasks point their encoded source straight into the slurped buffer. */
typedef struct {
    const uint8_t* base;
    size_t         size;
    size_t         pos;
} GeoReader;

static int rd(GeoReader* r, void* p, size_t n)
{
    if (n == 0) return 0;
    if (n > r->size - r->pos) return -1;   /* truncated / corrupt */
    memcpy(p, r->base + r->pos, n);
    r->pos += n;
    return 0;
}

static int wr_tagsize(FILE* f, uint32_t tag, uint64_t size)
{
    if (wr(f, &tag, sizeof tag)) return -1;
    if (wr(f, &size, sizeof size)) return -1;
    return 0;
}

/* ----------------------------------------------------------------- write */

/* Write one vertex-stream array (comp floats per element). `is_shared` marks
 * a mesh folded onto an earlier prototype — emit a SHARED marker instead of
 * the data; load pass 2 wires the array to scene->meshes[prototype_idx]. */
static int write_vtx_blob(FILE* f, const float* arr, int count, int comp,
                          int is_shared)
{
    if (is_shared)
        return wr_tagsize(f, GEO_BLOB_SHARED, 0);
    if (!arr || count <= 0)
        return wr_tagsize(f, GEO_BLOB_NULL, 0);

    size_t stride = (size_t)comp * sizeof(float);
    size_t raw    = (size_t)count * stride;
    size_t bound  = meshopt_encodeVertexBufferBound((size_t)count, stride);
    unsigned char* buf = bound ? (unsigned char*)malloc(bound) : NULL;
    size_t enc = buf ? meshopt_encodeVertexBuffer(buf, bound, arr,
                                                  (size_t)count, stride) : 0;
    int rc;
    if (enc > 0 && enc < raw) {
        rc = wr_tagsize(f, GEO_BLOB_MESHOPT, enc);
        if (!rc) rc = wr(f, buf, enc);
    } else {
        rc = wr_tagsize(f, GEO_BLOB_RAW, raw);
        if (!rc) rc = wr(f, arr, raw);
    }
    free(buf);
    return rc;
}

/* Write one triangle-list index array. */
static int write_idx_blob(FILE* f, const uint32_t* idx, int count, int nverts,
                          int is_shared)
{
    if (is_shared)
        return wr_tagsize(f, GEO_BLOB_SHARED, 0);
    if (!idx || count <= 0)
        return wr_tagsize(f, GEO_BLOB_NULL, 0);

    size_t raw = (size_t)count * sizeof(uint32_t);
    size_t enc = 0;
    unsigned char* buf = NULL;
    /* meshopt index codec needs a triangle list. */
    if (count >= 3 && (count % 3) == 0 && nverts > 0) {
        size_t bound = meshopt_encodeIndexBufferBound((size_t)count,
                                                      (size_t)nverts);
        buf = bound ? (unsigned char*)malloc(bound) : NULL;
        if (buf) enc = meshopt_encodeIndexBuffer(buf, bound, idx, (size_t)count);
    }
    int rc;
    if (enc > 0 && enc < raw) {
        rc = wr_tagsize(f, GEO_BLOB_MESHOPT, enc);
        if (!rc) rc = wr(f, buf, enc);
    } else {
        rc = wr_tagsize(f, GEO_BLOB_RAW, raw);
        if (!rc) rc = wr(f, idx, raw);
    }
    free(buf);
    return rc;
}

/* ------------------------------------------------------------- meshlets */

/* On-disk meshlet record (NZGO v3 — compact mesh-shader-native layout). */
typedef struct {
    float    center[3];
    float    radius;
    float    cone_axis[3];
    float    cone_cutoff;
    uint32_t vertex_count;      /* entries in this meshlet's vertex table  */
    uint32_t triangle_count;    /* triangles (micro-index count = ×3)      */
} GeoCacheMeshletRec;

/* Meshlets built for one mesh during a cache write — the compact
 * mesh-shader-native layout, tightly packed (NZGO v3). */
typedef struct {
    GeoCacheMeshletRec* recs;
    uint32_t            count;
    uint32_t*           vertices;   /* concatenated per-meshlet vertex tables */
    uint32_t            nvertices;
    unsigned char*      triangles;  /* concatenated 8-bit micro-indices       */
    uint32_t            ntriangles;
} GeoMeshletSet;

/* Build meshlets for one mesh's geometry. Fills *out (zeroed on entry / on
 * any failure or empty input). Returns 0 on success or benign skip, -1 only
 * on allocation failure. A meshlet's index stream is the original triangles
 * re-grouped per meshlet, expanded to global vertex indices. */
static int geo_build_meshlets(const float* positions, int nvertices,
                              const uint32_t* indices, int nindices,
                              GeoMeshletSet* out)
{
    memset(out, 0, sizeof(*out));
    if (!positions || !indices || nvertices <= 0 ||
        nindices < 3 || (nindices % 3) != 0)
        return 0;

    /* meshopt_optimizeVertexCache / buildMeshlets index into vertex-sized
     * arrays — an out-of-range index reads out of bounds and crashes. Skip
     * meshlet building for a malformed mesh. */
    for (int i = 0; i < nindices; i++)
        if (indices[i] >= (uint32_t)nvertices)
            return 0;

    size_t max_meshlets = meshopt_buildMeshletsBound(
        (size_t)nindices, GEO_MESHLET_MAX_VERTICES, GEO_MESHLET_MAX_TRIANGLES);
    if (max_meshlets == 0) return 0;

    /* Feed buildMeshlets a vertex-cache-optimized index copy for better
     * vertex reuse inside meshlets. This is a triangle reorder only — the
     * mesh's own index buffer (rendered, and stored as the geometry index
     * blob) is untouched, so the render stays byte-identical. */
    uint32_t* opt = (uint32_t*)malloc((size_t)nindices * sizeof(uint32_t));
    if (!opt) return -1;
    meshopt_optimizeVertexCache(opt, indices, (size_t)nindices,
                                (size_t)nvertices);

    struct meshopt_Meshlet* mls =
        (struct meshopt_Meshlet*)malloc(max_meshlets * sizeof(struct meshopt_Meshlet));
    unsigned int* mlv = (unsigned int*)malloc(
        max_meshlets * GEO_MESHLET_MAX_VERTICES * sizeof(unsigned int));
    unsigned char* mlt = (unsigned char*)malloc(
        max_meshlets * GEO_MESHLET_MAX_TRIANGLES * 3u * sizeof(unsigned char));
    if (!mls || !mlv || !mlt) {
        free(opt); free(mls); free(mlv); free(mlt);
        return -1;
    }

    /* cone_weight 0.25 — meshoptimizer's recommended balance between cone
     * culling and other culling forms (0.0 = no cone-cull weighting). */
    size_t mcount = meshopt_buildMeshlets(
        mls, mlv, mlt, opt, (size_t)nindices,
        positions, (size_t)nvertices, 3u * sizeof(float),
        GEO_MESHLET_MAX_VERTICES, GEO_MESHLET_MAX_TRIANGLES, 0.25f);
    free(opt);
    if (mcount == 0) { free(mls); free(mlv); free(mlt); return 0; }

    /* Sum the compact-layout totals so the tight output arrays can be sized. */
    uint32_t tot_v = 0, tot_t = 0;
    for (size_t mi = 0; mi < mcount; mi++) {
        tot_v += mls[mi].vertex_count;
        tot_t += mls[mi].triangle_count * 3u;
    }

    GeoCacheMeshletRec* recs =
        (GeoCacheMeshletRec*)malloc(mcount * sizeof(GeoCacheMeshletRec));
    uint32_t*      cv = (uint32_t*)malloc((size_t)tot_v * sizeof(uint32_t));
    unsigned char* ct = (unsigned char*)malloc((size_t)tot_t);
    if (!recs || !cv || !ct) {
        free(recs); free(cv); free(ct); free(mls); free(mlv); free(mlt);
        return -1;
    }

    /* meshopt_buildMeshlets already produced the mesh-shader-native compact
     * layout (per-meshlet vertex table + 8-bit micro-indices). Keep it
     * verbatim, repacked tightly — meshopt 4-byte-aligns triangle_offset, so
     * a verbatim copy of mlv/mlt would carry padding; the tight form makes
     * the warm load a straight read with no compaction. */
    uint32_t vcur = 0, tcur = 0;
    for (size_t mi = 0; mi < mcount; mi++) {
        const struct meshopt_Meshlet* m = &mls[mi];
        unsigned int*  mv = mlv + m->vertex_offset;
        unsigned char* mt = mlt + m->triangle_offset;
        meshopt_optimizeMeshlet(mv, mt, m->triangle_count, m->vertex_count);
        struct meshopt_Bounds b = meshopt_computeMeshletBounds(
            mv, mt, m->triangle_count,
            positions, (size_t)nvertices, 3u * sizeof(float));

        GeoCacheMeshletRec* r = &recs[mi];
        r->center[0] = b.center[0];
        r->center[1] = b.center[1];
        r->center[2] = b.center[2];
        r->radius    = b.radius;
        r->cone_axis[0] = b.cone_axis[0];
        r->cone_axis[1] = b.cone_axis[1];
        r->cone_axis[2] = b.cone_axis[2];
        r->cone_cutoff  = b.cone_cutoff;
        r->vertex_count   = m->vertex_count;
        r->triangle_count = m->triangle_count;

        memcpy(cv + vcur, mv, (size_t)m->vertex_count * sizeof(uint32_t));
        memcpy(ct + tcur, mt, (size_t)m->triangle_count * 3u);
        vcur += m->vertex_count;
        tcur += m->triangle_count * 3u;
    }

    free(mls); free(mlv); free(mlt);
    out->recs       = recs;
    out->count      = (uint32_t)mcount;
    out->vertices   = cv;
    out->nvertices  = tot_v;
    out->triangles  = ct;
    out->ntriangles = tot_t;
    return 0;
}

static void geo_meshlet_set_free(GeoMeshletSet* s)
{
    free(s->recs);
    free(s->vertices);
    free(s->triangles);
    memset(s, 0, sizeof(*s));
}

/* ----------------------------------------------------- cook-time dedup */
/*
 * NZGO v1 folds content-identical meshes at cook time. A mesh whose geometry
 * and displayColor data byte-match an earlier mesh — and shares its
 * material_index — is stored as a SHARED reference and, on load, becomes an
 * instance of that earlier prototype, so the cache (and the viewer's merged
 * buffers) hold each unique geometry once.
 *
 * A hash hit is memcmp-verified, so a 64-bit collision cannot corrupt the
 * persistent cache. material_index is a separate group key — geometry shared
 * between differing materials must not fold.
 */

/* Gate: NUSD_GEO_CACHE_DEDUP unset or non-"0" -> on; "0" -> off (A/B knob). */
static int geo_cache_dedup_enabled(void)
{
    const char* e = getenv("NUSD_GEO_CACHE_DEDUP");
    return !(e && e[0] == '0');
}

static inline uint64_t gc_fnv1a(uint64_t h, const void* data, size_t n)
{
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

/* FNV-1a-64 over a mesh's geometry and color payload. Returns 0 for a
 * degenerate mesh (the caller skips it). */
static uint64_t gc_geo_hash(const SceneMesh* m)
{
    if (!m || m->nvertices <= 0 || m->nindices <= 0 ||
        !m->positions || !m->indices)
        return 0;
    uint64_t h = 1469598103934665603ULL;          /* FNV-1a offset basis */
    int32_t nv = m->nvertices, ni = m->nindices;
    h = gc_fnv1a(h, &nv, sizeof nv);
    h = gc_fnv1a(h, &ni, sizeof ni);
    uint8_t present = (uint8_t)((m->normals ? 1 : 0) | (m->texcoords ? 2 : 0));
    h = gc_fnv1a(h, &present, sizeof present);
    h = gc_fnv1a(h, m->positions, (size_t)nv * 3u * sizeof(float));
    h = gc_fnv1a(h, m->indices,   (size_t)ni * sizeof(uint32_t));
    if (m->normals)
        h = gc_fnv1a(h, m->normals,   (size_t)nv * 3u * sizeof(float));
    if (m->texcoords)
        h = gc_fnv1a(h, m->texcoords, (size_t)nv * 2u * sizeof(float));
    int32_t has_dc = m->has_display_color ? 1 : 0;
    h = gc_fnv1a(h, &has_dc, sizeof has_dc);
    if (m->has_display_color)
        h = gc_fnv1a(h, m->display_color, 3u * sizeof(float));
    uint8_t has_colors = (uint8_t)(m->colors ? 1 : 0);
    h = gc_fnv1a(h, &has_colors, sizeof has_colors);
    if (m->colors)
        h = gc_fnv1a(h, m->colors, (size_t)nv * 3u * sizeof(float));
    if (h == 0) h = 0xdeadbeefcafebabeULL;
    return h;
}

/* True when meshes a and b have byte-identical geometry and color payload.
 * Confirms a hash hit so a collision cannot mis-fold the cache. */
static int gc_geo_equal(const SceneMesh* a, const SceneMesh* b)
{
    if (a->nvertices != b->nvertices || a->nindices != b->nindices)
        return 0;
    if (((a->normals ? 1 : 0) | (a->texcoords ? 2 : 0)) !=
        ((b->normals ? 1 : 0) | (b->texcoords ? 2 : 0)))
        return 0;
    size_t pos = (size_t)a->nvertices * 3u * sizeof(float);
    size_t idx = (size_t)a->nindices  * sizeof(uint32_t);
    if (!a->positions || !b->positions ||
        memcmp(a->positions, b->positions, pos)) return 0;
    if (!a->indices || !b->indices ||
        memcmp(a->indices, b->indices, idx)) return 0;
    if (a->normals && memcmp(a->normals, b->normals, pos)) return 0;
    if (a->texcoords &&
        memcmp(a->texcoords, b->texcoords,
               (size_t)a->nvertices * 2u * sizeof(float))) return 0;
    if ((a->has_display_color ? 1 : 0) != (b->has_display_color ? 1 : 0))
        return 0;
    if (a->has_display_color &&
        memcmp(a->display_color, b->display_color, 3u * sizeof(float)))
        return 0;
    if ((a->colors ? 1 : 0) != (b->colors ? 1 : 0))
        return 0;
    if (a->colors &&
        memcmp(a->colors, b->colors,
               (size_t)a->nvertices * 3u * sizeof(float))) return 0;
    return 1;
}

/* Open-addressing table: geometry hash -> first-occurrence mesh index. */
typedef struct { uint64_t hash; int mesh; } GcHashEntry;

/* Fill eff[0..N) so eff[i] is the mesh whose stored geometry mesh i should
 * reference: i itself when i is a prototype, an earlier mesh otherwise.
 * eff[i] <= i is guaranteed (load pass 2 needs the prototype to precede the
 * instance). Composes the scene's existing USD-instance prototype_idx with
 * cook-time content dedup; falls back to the USD-instance-only mapping when
 * dedup is disabled or scratch alloc fails. */
static void gc_compute_effective_proto(const Scene* s, int* eff)
{
    int N = s->nmeshes;

    /* Phase A — resolve each mesh to its USD geometry owner (follow
     * prototype_idx to a fixpoint; cycle-guarded). */
    for (int i = 0; i < N; i++) {
        int cur = i, guard = 0;
        while (guard++ < N) {
            int p = s->meshes[cur].prototype_idx;
            if (p < 0 || p >= N || p == cur) break;
            cur = p;
        }
        eff[i] = (cur >= 0 && cur <= i) ? cur : i;
    }

    if (!geo_cache_dedup_enabled() || N <= 1) return;

    uint32_t cap = 1024;
    while (cap < (uint32_t)N * 2u) cap <<= 1;
    GcHashEntry* tab = (GcHashEntry*)calloc(cap, sizeof(GcHashEntry));
    if (!tab) return;                          /* OOM -> USD-instance mapping */
    uint32_t mask = cap - 1u;

    /* Phase B — content-dedup the geometry-bearing meshes (eff[i]==i), in
     * index order so the first occurrence of a geometry is its leader. */
    for (int i = 0; i < N; i++) {
        if (eff[i] != i) continue;             /* USD instance — owner handled */
        const SceneMesh* m = &s->meshes[i];
        uint64_t h = gc_geo_hash(m);
        if (!h) continue;                      /* degenerate -> own leader */
        uint32_t slot = (uint32_t)h & mask;
        int leader = -1;
        while (tab[slot].hash) {
            if (tab[slot].hash == h) {
                int cand = tab[slot].mesh;
                /* material_index intentionally NOT gated — material is applied
                 * per-draw, not baked into shared geometry, so geometry-identical
                 * meshes share a leader regardless of material (gc_geo_equal still
                 * gates on the vertex-baked display_color). Matches viewer.c. */
                if (gc_geo_equal(&s->meshes[cand], m))
                    leader = cand;             /* confirmed content duplicate */
                break;                         /* one slot per hash */
            }
            slot = (slot + 1u) & mask;
        }
        if (leader >= 0)
            eff[i] = leader;
        else if (!tab[slot].hash) {
            tab[slot].hash = h;
            tab[slot].mesh = i;
        }
    }
    free(tab);

    /* Phase C — point every non-leader at the root content leader. */
    for (int i = 0; i < N; i++)
        if (eff[i] != i)
            eff[i] = eff[eff[i]];
}

int geo_cache_write(const char* usd_path, const Scene* scene)
{
    if (!geo_cache_enabled() || !usd_path || !scene) return -1;
    if (scene->nmeshes <= 0)  return -1;
    if (scene->ncurves > 0)   return -1;   /* geometry-only cache; curves re-parse */

    struct stat st;
    if (stat(usd_path, &st) != 0) return -1;

    char path[2048];
    if (geo_cache_path_for(usd_path, path, sizeof path)) return -1;

    /* Cook-time dedup mapping: eff[i] is the mesh whose geometry mesh i
     * references — i itself for a prototype, an earlier mesh for an instance
     * or a content duplicate. */
    int  nmeshes = scene->nmeshes;
    int* eff = (int*)malloc((size_t)nmeshes * sizeof(int));
    if (!eff) return -1;
    gc_compute_effective_proto(scene, eff);
    {
        int leaders = 0;
        for (int i = 0; i < nmeshes; i++) if (eff[i] == i) leaders++;
        fprintf(stderr,
                "[geo_cache] dedup: %d meshes -> %d unique geometries "
                "(%d folded)\n", nmeshes, leaders, nmeshes - leaders);
    }

    /* Pre-pass: sum decoded bytes of leader (non-shared) arrays to size the
     * read arena — folded meshes store no blob bytes. */
    uint64_t total = 0;
    for (int i = 0; i < nmeshes; i++) {
        if (eff[i] != i) continue;
        const SceneMesh* m = &scene->meshes[i];
        if (m->nvertices > 0) {
            if (m->positions) total += (uint64_t)m->nvertices * 3 * sizeof(float);
            if (m->normals)   total += (uint64_t)m->nvertices * 3 * sizeof(float);
            if (m->colors)    total += (uint64_t)m->nvertices * 3 * sizeof(float);
            if (m->texcoords) total += (uint64_t)m->nvertices * 2 * sizeof(float);
        }
        if (m->nindices > 0 && m->indices)
            total += (uint64_t)m->nindices * sizeof(uint32_t);
    }

    FILE* f = fopen(path, "wb");
    if (!f) { free(eff); return -1; }

    GeoCacheHeader h;
    memset(&h, 0, sizeof h);
    memcpy(h.magic, "NZGO", 4);
    h.format_version  = NZGO_FORMAT_VERSION;
    h.src_mtime       = (int64_t)st.st_mtime;
    h.src_size        = (int64_t)st.st_size;
    h.total_geo_bytes = total;
    h.nmeshes         = (uint32_t)scene->nmeshes;
    h.up_axis         = scene->up_axis;
    h.has_authored_light = scene->has_authored_light;
    h.has_dome_light  = scene->has_dome_light;
    memcpy(h.dome_color, scene->dome_color, sizeof h.dome_color);
    h.dome_intensity  = scene->dome_intensity;
    h.dome_rotation_y = scene->dome_rotation_y;
    memcpy(h.dome_hdr_path, scene->dome_hdr_path, sizeof h.dome_hdr_path);
    memcpy(h.bounds_min, scene->bounds_min, sizeof h.bounds_min);
    memcpy(h.bounds_max, scene->bounds_max, sizeof h.bounds_max);

    int err = wr(f, &h, sizeof h);

    for (int i = 0; i < nmeshes && !err; i++) {
        const SceneMesh* m  = &scene->meshes[i];
        int is_shared = (eff[i] != i);

        GeoCacheMeshHdr mh;
        memset(&mh, 0, sizeof mh);
        mh.nvertices = m->nvertices;
        mh.nindices  = m->nindices;
        memcpy(mh.world_xform,      m->world_xform,      sizeof mh.world_xform);
        memcpy(mh.bounds_min,       m->bounds_min,       sizeof mh.bounds_min);
        memcpy(mh.bounds_max,       m->bounds_max,       sizeof mh.bounds_max);
        memcpy(mh.local_bounds_min, m->local_bounds_min, sizeof mh.local_bounds_min);
        memcpy(mh.local_bounds_max, m->local_bounds_max, sizeof mh.local_bounds_max);
        memcpy(mh.display_color,    m->display_color,    sizeof mh.display_color);
        mh.has_display_color = m->has_display_color;
        mh.material_index    = m->material_index;
        mh.prototype_idx     = eff[i];
        mh.is_proto_only     = m->is_proto_only;
        mh.own_index         = i;
        err |= wr(f, &mh, sizeof mh);
        {
            uint32_t path_len = m->path ? (uint32_t)strlen(m->path) : 0u;
            err |= wr(f, &path_len, sizeof path_len);
            if (!err && path_len > 0u)
                err |= wr(f, m->path, (size_t)path_len);
        }

        err |= write_vtx_blob(f, m->positions, m->nvertices, 3, is_shared);
        err |= write_vtx_blob(f, m->normals,   m->nvertices, 3, is_shared);
        err |= write_vtx_blob(f, m->colors,    m->nvertices, 3, is_shared);
        err |= write_vtx_blob(f, m->texcoords, m->nvertices, 2, is_shared);
        err |= write_idx_blob(f, m->indices,   m->nindices, m->nvertices,
                              is_shared);
    }

    /* Meshlet section: one [u32 count][records][vtx blob][tri blob] group per
     * mesh. Meshlets are built and compacted per prototype; instances write
     * count 0 and inherit the prototype's range on load. The compact layout
     * is stored raw — the warm load reads it with no decode and no compaction. */
    uint32_t total_meshlets = 0, total_meshlet_verts = 0, total_meshlet_tris = 0;
    for (int i = 0; i < nmeshes && !err; i++) {
        const SceneMesh* m = &scene->meshes[i];

        GeoMeshletSet set;
        memset(&set, 0, sizeof set);
        if (eff[i] == i)
            geo_build_meshlets(m->positions, m->nvertices,
                               m->indices, m->nindices, &set);

        uint32_t mc = set.count;
        err |= wr(f, &mc, sizeof mc);
        if (!err && mc > 0) {
            err |= wr(f, set.recs, (size_t)mc * sizeof(GeoCacheMeshletRec));
            err |= wr(f, set.vertices, (size_t)set.nvertices * sizeof(uint32_t));
            err |= wr(f, set.triangles, (size_t)set.ntriangles);
            total_meshlets      += mc;
            total_meshlet_verts += set.nvertices;
            total_meshlet_tris  += set.ntriangles;
        }
        geo_meshlet_set_free(&set);
    }

    if (!err)
        fprintf(stderr, "[geo_cache] meshlets built: %u (%u verts, %u tris)\n",
                total_meshlets, total_meshlet_verts, total_meshlet_tris);

    /* Back-patch the meshlet totals into the header. */
    if (!err) {
        if (fseek(f, (long)offsetof(GeoCacheHeader, nmeshlets), SEEK_SET) != 0)
            err = 1;
        else {
            err |= wr(f, &total_meshlets, sizeof total_meshlets);
            err |= wr(f, &total_meshlet_tris, sizeof total_meshlet_tris);
        }
    }
    if (!err) {
        if (fseek(f, (long)offsetof(GeoCacheHeader, nmeshlet_verts), SEEK_SET) != 0)
            err = 1;
        else
            err |= wr(f, &total_meshlet_verts, sizeof total_meshlet_verts);
    }

    if (fclose(f) != 0) err = 1;
    free(eff);
    if (err) { remove(path); return -1; }
    return 0;
}

/* ------------------------------------------------------------------ read */

/* iter 37 (port of Vulkan iter 34): blob decode is collected as independent
 * tasks during the sequential read pass, then run in parallel — the
 * per-prototype blobs are independent. Arena allocation stays in the
 * sequential pass, so the parallel decode region touches no shared state. */
typedef struct {
    int            is_index;   /* 0 = decodeVertexBuffer, 1 = decodeIndexBuffer */
    int            is_raw;     /* GEO_BLOB_RAW → memcpy; else MESHOPT → decode  */
    void*          dst;        /* arena-allocated decode destination            */
    size_t         count;
    size_t         elem_size;  /* bytes per element                             */
    const unsigned char* enc;  /* encoded bytes — a slice of the slurped file   */
    size_t         enc_size;
    int            rc;         /* worker result: 0 = ok                         */
} DecodeTask;

typedef struct {
    DecodeTask* tasks;
    uint32_t    count;
    uint32_t    cap;
} DecodeTaskList;

static int decode_task_append(DecodeTaskList* tl, const DecodeTask* dt)
{
    if (tl->count == tl->cap) {
        uint32_t nc = tl->cap ? tl->cap * 2u : 8192u;
        DecodeTask* nt = (DecodeTask*)realloc(tl->tasks,
                                              (size_t)nc * sizeof(DecodeTask));
        if (!nt) return -1;
        tl->tasks = nt;
        tl->cap   = nc;
    }
    tl->tasks[tl->count++] = *dt;
    return 0;
}

static void decode_tasklist_free(DecodeTaskList* tl)
{
    free(tl->tasks);   /* task.enc slices the slurped file — not owned here */
    tl->tasks = NULL;
    tl->count = 0;
    tl->cap   = 0;
}

/* Decode every collected task — parallel when OpenMP is available. The tasks
 * are independent (distinct dst, distinct enc, no arena), so the parallel
 * region is data-race-free. Returns 0 iff every task succeeded. */
static int decode_tasklist_run(DecodeTaskList* tl)
{
    int n = (int)tl->count;
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int t = 0; t < n; t++) {
        DecodeTask* dt = &tl->tasks[t];
        if (dt->is_raw) {
            dt->rc = (dt->enc_size == dt->count * dt->elem_size)
                     ? (memcpy(dt->dst, dt->enc, dt->enc_size), 0) : -1;
        } else if (dt->is_index) {
            dt->rc = meshopt_decodeIndexBuffer(dt->dst, dt->count, dt->elem_size,
                                               dt->enc, dt->enc_size);
        } else {
            dt->rc = meshopt_decodeVertexBuffer(dt->dst, dt->count, dt->elem_size,
                                                dt->enc, dt->enc_size);
        }
    }
    for (int t = 0; t < n; t++)
        if (tl->tasks[t].rc != 0) return -1;
    return 0;
}

/* Read one vertex-stream blob: sequentially read the encoded bytes + arena-
 * alloc the destination, and append a decode task (the decode runs later,
 * in parallel). On SHARED, leaves *out NULL and sets *shared_flag. */
static int read_vtx_blob(GeoReader* r, Arena* a, int count, int comp,
                         float** out, unsigned char* shared_flag,
                         DecodeTaskList* tl)
{
    uint32_t tag; uint64_t size;
    *out = NULL; *shared_flag = 0;
    if (rd(r, &tag, sizeof tag) || rd(r, &size, sizeof size)) return -1;
    if (tag == GEO_BLOB_NULL)   return 0;
    if (tag == GEO_BLOB_SHARED) { *shared_flag = 1; return 0; }
    if (tag != GEO_BLOB_RAW && tag != GEO_BLOB_MESHOPT) return -1;
    if (count <= 0 || size == 0 || size > GEO_MAX_BLOB) return -1;
    if (size > r->size - r->pos) return -1;   /* blob must lie fully in-buffer */

    size_t decoded = (size_t)count * (size_t)comp * sizeof(float);
    const unsigned char* enc = r->base + r->pos;   /* slice of the slurped file */
    r->pos += (size_t)size;

    float* dst = (float*)arena_alloc(a, decoded, 8);
    if (!dst) return -1;

    DecodeTask dt;
    dt.is_index  = 0;
    dt.is_raw    = (tag == GEO_BLOB_RAW);
    dt.dst       = dst;
    dt.count     = (size_t)count;
    dt.elem_size = (size_t)comp * sizeof(float);
    dt.enc       = enc;
    dt.enc_size  = (size_t)size;
    dt.rc        = 0;
    if (decode_task_append(tl, &dt)) return -1;

    *out = dst;
    return 0;
}

static int read_idx_blob(GeoReader* r, Arena* a, int count,
                         uint32_t** out, unsigned char* shared_flag,
                         DecodeTaskList* tl)
{
    uint32_t tag; uint64_t size;
    *out = NULL; *shared_flag = 0;
    if (rd(r, &tag, sizeof tag) || rd(r, &size, sizeof size)) return -1;
    if (tag == GEO_BLOB_NULL)   return 0;
    if (tag == GEO_BLOB_SHARED) { *shared_flag = 1; return 0; }
    if (tag != GEO_BLOB_RAW && tag != GEO_BLOB_MESHOPT) return -1;
    if (count <= 0 || size == 0 || size > GEO_MAX_BLOB) return -1;
    if (size > r->size - r->pos) return -1;   /* blob must lie fully in-buffer */

    size_t decoded = (size_t)count * sizeof(uint32_t);
    const unsigned char* enc = r->base + r->pos;   /* slice of the slurped file */
    r->pos += (size_t)size;

    uint32_t* dst = (uint32_t*)arena_alloc(a, decoded, 8);
    if (!dst) return -1;

    DecodeTask dt;
    dt.is_index  = 1;
    dt.is_raw    = (tag == GEO_BLOB_RAW);
    dt.dst       = dst;
    dt.count     = (size_t)count;
    dt.elem_size = sizeof(uint32_t);
    dt.enc       = enc;
    dt.enc_size  = (size_t)size;
    dt.rc        = 0;
    if (decode_task_append(tl, &dt)) return -1;

    *out = dst;
    return 0;
}

Scene* geo_cache_try_load(const char* usd_path)
{
    if (!geo_cache_enabled() || !usd_path) return NULL;

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    char path[2048];
    if (geo_cache_path_for(usd_path, path, sizeof path)) return NULL;

    struct stat st;
    if (stat(usd_path, &st) != 0) return NULL;

    /* Slurp the whole cache into one buffer, then parse from memory: a
     * bounds-checked memcpy per read instead of many small freads, and
     * decode tasks point their encoded source straight into the buffer
     * (no per-blob malloc/copy). An mmap variant was measured and rejected
     * on the Vulkan cache (iter 35) — faulting a freshly-cooked file's
     * pages into a new process regressed the cook→load path. */
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat cst;
    if (fstat(fd, &cst) != 0 || cst.st_size < (off_t)sizeof(GeoCacheHeader)) {
        close(fd); return NULL;
    }
    size_t   filebuf_size = (size_t)cst.st_size;
    uint8_t* filebuf      = (uint8_t*)malloc(filebuf_size);
    if (!filebuf) { close(fd); return NULL; }

    /* Parallel slurp (iter 41, ports Vulkan iter 40): a single fread of the
     * whole cache dominates the warm load's setup phase — most of its cost is
     * faulting + filling ~200 MB of fresh anonymous pages single-threaded.
     * pread carries its own offset and is thread-safe, so N threads fault +
     * fill disjoint chunks at once. */
    {
        int          rc_any  = 0;
        const size_t CHUNK   = 2u << 20;   /* 2 MiB */
        long         nchunks = (long)((filebuf_size + CHUNK - 1) / CHUNK);
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 1) reduction(|:rc_any)
#endif
        for (long c = 0; c < nchunks; c++) {
            size_t off = (size_t)c * CHUNK;
            size_t len = (filebuf_size - off < CHUNK) ? filebuf_size - off
                                                      : CHUNK;
            size_t got = 0;
            while (got < len) {
                ssize_t n = pread(fd, filebuf + off + got, len - got,
                                  (off_t)(off + got));
                if (n <= 0) { rc_any = 1; break; }
                got += (size_t)n;
            }
        }
        close(fd);
        if (rc_any) { free(filebuf); filebuf = NULL; return NULL; }
    }
    GeoReader reader = { filebuf, filebuf_size, 0 };

    GeoCacheHeader h;
    if (rd(&reader, &h, sizeof h)) { free(filebuf); return NULL; }
    if (memcmp(h.magic, "NZGO", 4) != 0 ||
        h.format_version != NZGO_FORMAT_VERSION ||
        h.src_mtime != (int64_t)st.st_mtime ||
        h.src_size  != (int64_t)st.st_size) {
        free(filebuf);
        return NULL;   /* stale, foreign, or wrong-version cache */
    }
    if (h.nmeshes == 0 || h.nmeshes > (uint32_t)GEO_MAX_VERTICES ||
        h.nmeshlets > (uint32_t)GEO_MAX_INDICES ||
        h.nmeshlet_indices > (uint32_t)GEO_MAX_INDICES ||
        h.nmeshlet_verts > (uint32_t)GEO_MAX_INDICES) {
        free(filebuf);
        return NULL;
    }

    Scene*         scene  = (Scene*)calloc(1, sizeof(Scene));
    Arena*         arena  = (Arena*)malloc(sizeof(Arena));
    unsigned char* shared = NULL;
    DecodeTaskList tl     = {0};
    if (!scene || !arena) { free(scene); free(arena); free(filebuf); return NULL; }

    /* v3 stores the compact meshlet layout directly: meshlet_vertices is
     * nmeshlet_verts uint32 entries, meshlet_triangles nmeshlet_indices
     * 8-bit micro-indices. */
    size_t arena_sz = (size_t)h.total_geo_bytes
                    + (size_t)h.nmeshes * sizeof(SceneMesh)
                    + (size_t)h.nmeshlets * sizeof(SceneMeshlet)
                    + (size_t)h.nmeshlet_verts * sizeof(uint32_t)
                    + (size_t)h.nmeshlet_indices
                    + (1u << 16);
    if (arena_sz < (64u << 20)) arena_sz = 64u << 20;
    *arena = arena_create(arena_sz);
    if (!arena->head) { free(scene); free(arena); free(filebuf); return NULL; }

    struct timespec t_setup;
    clock_gettime(CLOCK_MONOTONIC, &t_setup);   /* slurp + arena setup done */

    scene->_arena       = arena;
    scene->_stage       = NULL;
    scene->_meshes_heap = 0;          /* meshes live in the arena */
    scene->curves       = NULL;
    scene->ncurves      = 0;
    scene->up_axis      = h.up_axis;
    scene->has_authored_light = h.has_authored_light;
    scene->has_dome_light = h.has_dome_light;
    memcpy(scene->dome_color, h.dome_color, sizeof scene->dome_color);
    scene->dome_intensity = h.dome_intensity;
    scene->dome_rotation_y = h.dome_rotation_y;
    memcpy(scene->dome_hdr_path, h.dome_hdr_path, sizeof scene->dome_hdr_path);
    memcpy(scene->bounds_min, h.bounds_min, sizeof scene->bounds_min);
    memcpy(scene->bounds_max, h.bounds_max, sizeof scene->bounds_max);

    scene->meshes = (SceneMesh*)arena_calloc(arena, h.nmeshes, sizeof(SceneMesh));
    if (!scene->meshes) goto fail;
    scene->nmeshes = (int)h.nmeshes;

    shared = (unsigned char*)calloc((size_t)h.nmeshes, 5);
    if (!shared) goto fail;

    /* Pass 1a — sequential: read records + encoded blobs, arena-alloc the
     * decode destinations, collect decode tasks (decode runs in 1b). */
    for (uint32_t i = 0; i < h.nmeshes; i++) {
        SceneMesh* m = &scene->meshes[i];
        GeoCacheMeshHdr mh;
        if (rd(&reader, &mh, sizeof mh)) goto fail;
        if (mh.own_index != (int)i) goto fail;
        if (mh.nvertices < 0 || mh.nvertices > GEO_MAX_VERTICES ||
            mh.nindices  < 0 || mh.nindices  > GEO_MAX_INDICES) goto fail;

        m->nvertices = mh.nvertices;
        m->nindices  = mh.nindices;
        memcpy(m->world_xform,      mh.world_xform,      sizeof m->world_xform);
        memcpy(m->bounds_min,       mh.bounds_min,       sizeof m->bounds_min);
        memcpy(m->bounds_max,       mh.bounds_max,       sizeof m->bounds_max);
        memcpy(m->local_bounds_min, mh.local_bounds_min, sizeof m->local_bounds_min);
        memcpy(m->local_bounds_max, mh.local_bounds_max, sizeof m->local_bounds_max);
        memcpy(m->display_color,    mh.display_color,    sizeof m->display_color);
        m->has_display_color = mh.has_display_color;
        m->material_index    = mh.material_index;
        m->prototype_idx     = mh.prototype_idx;
        m->is_proto_only     = mh.is_proto_only;
        m->visible           = 1;
        /* vertex_offset / index_offset / buffer_index / index_byte_offset /
         * index_type_bits / lazy_prim_idx stay 0 (arena_calloc) — viewer-set. */
        {
            uint32_t path_len = 0u;
            if (rd(&reader, &path_len, sizeof path_len)) goto fail;
            if (path_len > (1u << 20)) goto fail;
            if (path_len > 0u) {
                m->path = (char*)malloc((size_t)path_len + 1u);
                if (!m->path) goto fail;
                if (rd(&reader, m->path, (size_t)path_len)) goto fail;
                m->path[path_len] = '\0';
            }
        }

        unsigned char* sf = &shared[(size_t)i * 5];
        if (read_vtx_blob(&reader, arena, m->nvertices, 3, &m->positions, &sf[0], &tl)) goto fail;
        if (read_vtx_blob(&reader, arena, m->nvertices, 3, &m->normals,   &sf[1], &tl)) goto fail;
        if (read_vtx_blob(&reader, arena, m->nvertices, 3, &m->colors,    &sf[2], &tl)) goto fail;
        if (read_vtx_blob(&reader, arena, m->nvertices, 2, &m->texcoords, &sf[3], &tl)) goto fail;
        if (read_idx_blob(&reader, arena, m->nindices,     &m->indices,   &sf[4], &tl)) goto fail;
    }

    /* Pass 1b — decode the collected blobs in parallel (no file, no arena). */
    if (decode_tasklist_run(&tl)) goto fail;
    decode_tasklist_free(&tl);

    /* Pass 2 — wire each instance's shared arrays to its prototype. */
    for (uint32_t i = 0; i < h.nmeshes; i++) {
        SceneMesh*     m  = &scene->meshes[i];
        unsigned char* sf = &shared[(size_t)i * 5];
        int p = m->prototype_idx;
        if (p < 0 || p >= (int)h.nmeshes) {
            if (sf[0] | sf[1] | sf[2] | sf[3] | sf[4]) goto fail;
            continue;
        }
        const SceneMesh* pm = &scene->meshes[p];
        if (sf[0]) m->positions = pm->positions;
        if (sf[1]) m->normals   = pm->normals;
        if (sf[2]) m->colors    = pm->colors;
        if (sf[3]) m->texcoords = pm->texcoords;
        if (sf[4]) m->indices   = pm->indices;
    }

    /* Meshlet section (NZGO v3): the cache stores the compact mesh-shader-
     * native layout directly — per-meshlet records + a raw vertex-table blob
     * + a raw 8-bit micro-index blob. The warm load is a straight read; the
     * load-time hash-dedup compaction is gone. */
    if (h.nmeshlets > 0) {
        scene->meshlets = (SceneMeshlet*)arena_calloc(arena, h.nmeshlets,
                                                      sizeof(SceneMeshlet));
        if (!scene->meshlets) goto fail;
    }
    if (h.nmeshlet_verts > 0) {
        scene->meshlet_vertices = (uint32_t*)arena_alloc(
            arena, (size_t)h.nmeshlet_verts * sizeof(uint32_t), 8);
        if (!scene->meshlet_vertices) goto fail;
    }
    if (h.nmeshlet_indices > 0) {
        scene->meshlet_triangles = (unsigned char*)arena_alloc(
            arena, (size_t)h.nmeshlet_indices, 8);
        if (!scene->meshlet_triangles) goto fail;
    }
    {
        uint32_t ml_off = 0;   /* fill cursor into scene->meshlets          */
        uint32_t gv_off = 0;   /* fill cursor into scene->meshlet_vertices  */
        uint32_t gt_off = 0;   /* fill cursor into scene->meshlet_triangles */
        for (uint32_t i = 0; i < h.nmeshes; i++) {
            SceneMesh* m = &scene->meshes[i];
            uint32_t mc;
            if (rd(&reader, &mc, sizeof mc)) goto fail;
            if (mc == 0) continue;
            if (mc > h.nmeshlets - ml_off) goto fail;

            /* Records: cull bounds + the compact-layout vertex/triangle
             * counts. Scene-global offsets are accumulated as records arrive. */
            uint32_t mv_count = 0, mt_count = 0;
            for (uint32_t k = 0; k < mc; k++) {
                GeoCacheMeshletRec rec;
                if (rd(&reader, &rec, sizeof rec)) goto fail;
                if (rec.triangle_count == 0 || rec.vertex_count == 0) goto fail;
                SceneMeshlet* sm = &scene->meshlets[ml_off + k];
                memcpy(sm->center, rec.center, sizeof sm->center);
                sm->radius = rec.radius;
                memcpy(sm->cone_axis, rec.cone_axis, sizeof sm->cone_axis);
                sm->cone_cutoff     = rec.cone_cutoff;
                sm->triangle_count  = rec.triangle_count;
                sm->vertex_count    = rec.vertex_count;
                sm->vertex_offset   = gv_off + mv_count;
                sm->triangle_offset = gt_off + mt_count;
                mv_count += rec.vertex_count;
                mt_count += rec.triangle_count * 3u;
            }
            if (mv_count > h.nmeshlet_verts   - gv_off) goto fail;
            if (mt_count > h.nmeshlet_indices - gt_off) goto fail;

            /* Read the two raw compact blobs straight into the scene arrays —
             * no decode, no compaction. */
            if (rd(&reader, scene->meshlet_vertices + gv_off,
                   (size_t)mv_count * sizeof(uint32_t))) goto fail;
            if (rd(&reader, scene->meshlet_triangles + gt_off, (size_t)mt_count))
                goto fail;
            gv_off += mv_count;
            gt_off += mt_count;
            m->meshlet_offset = ml_off;
            m->meshlet_count  = mc;
            ml_off += mc;
        }
        /* Round-trip integrity: read totals must match the header. */
        if (ml_off != h.nmeshlets || gv_off != h.nmeshlet_verts ||
            gt_off != h.nmeshlet_indices) goto fail;
        scene->nmeshlets          = (int)h.nmeshlets;
        scene->nmeshlet_vertices  = (int)gv_off;
        scene->nmeshlet_triangles = (int)gt_off;

        /* Instances inherit their prototype's meshlet range. */
        for (uint32_t i = 0; i < h.nmeshes; i++) {
            SceneMesh* m = &scene->meshes[i];
            if (m->meshlet_count > 0) continue;
            int p = m->prototype_idx;
            if (p >= 0 && p < (int)h.nmeshes && p != (int)i) {
                m->meshlet_offset = scene->meshes[p].meshlet_offset;
                m->meshlet_count  = scene->meshes[p].meshlet_count;
            }
        }
    }

    free(shared);
    shared = NULL;
    free(filebuf);

    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_ms = (t1.tv_sec - t0.tv_sec) * 1e3
                    + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
    double setup_ms = (t_setup.tv_sec - t0.tv_sec) * 1e3
                    + (t_setup.tv_nsec - t0.tv_nsec) * 1e-6;
    fprintf(stderr, "[geo_cache] HIT: %s — warm-loaded in %.1f ms "
            "(setup %.1f ms; %u meshes, %u meshlets)\n", usd_path,
            total_ms, setup_ms, h.nmeshes, h.nmeshlets);
    return scene;

fail:
    decode_tasklist_free(&tl);
    free(shared);
    if (scene && scene->meshes) {
        for (uint32_t i = 0; i < h.nmeshes; i++) {
            free(scene->meshes[i].path);
            scene->meshes[i].path = NULL;
        }
    }
    arena_destroy(arena);
    free(arena);
    free(scene);
    free(filebuf);
    return NULL;
}

/* ------------------------------------------------------------------- cook */

int geo_cache_cook(const char* usd_path)
{
    if (!usd_path) return -1;

    /* Engage the scene_load() cache hooks for this process. */
    setenv("NUSD_GEO_CACHE", "1", 1);

    /* scene_load() returns the cached Scene on a fresh hit, or parses the USD
     * geometry and writes <usd>.nzgeo.gl on a miss — CPU-only, no GL. */
    Scene* s = scene_load(usd_path);
    if (!s) return -1;
    int ok = (s->nmeshes > 0);
    scene_free(s);
    return ok ? 0 : -1;
}
