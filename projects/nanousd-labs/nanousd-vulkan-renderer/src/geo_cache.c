// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * geo_cache.c — on-disk geometry cache for the nanousd Vulkan renderer.
 *
 * Writes a fully-built Scene to a sidecar <usd>.nzgeo file using
 * meshoptimizer's lossless vertex/index codecs, and reloads it without
 * re-parsing USD. Gated by NUSD_GEO_CACHE=1.
 *
 * NZGC v2 also preprocesses meshoptimizer meshlets: at write time each
 * prototype's geometry is run through meshopt_buildMeshlets / optimizeMeshlet /
 * computeMeshletBounds, and the meshlets (cull bounds + a re-ordered index
 * stream) are serialized. On load the expanded index stream is compacted into
 * the mesh-shader-native layout (Scene.meshlet_vertices + meshlet_triangles);
 * instances inherit their prototype's meshlet range.
 *
 * NZGC v3 bakes content-hash dedup at cook time: meshes with byte-identical
 * geometry, material_index, and displayColor data are folded onto one
 * prototype, so a warm load is deduped with no runtime NUSD_HASH_DEDUP flag
 * and the file stores each unique geometry once. See gc_compute_effective_proto.
 *
 * Scope: meshes + lights + meshlets + scene bounds + dome metadata, and
 * optionally the flat material/texture payload needed by the renderer. Scenes
 * that carry curves (Phase 1) are not cached.
 * Instances (SceneMesh.prototype_idx != own index) share their prototype's
 * geometry and meshlets; the cache stores each only once.
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
#include "material.h"

#include <dirent.h>
#include <limits.h>
#include <meshoptimizer.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#define NZGC_FORMAT_VERSION 13u  /* v13: DomeLight color + presence metadata */

/* meshlet limits — match the Metal renderer's build_meshlets_for_prototype. */
#define GEO_MESHLET_MAX_VERTICES   64u
#define GEO_MESHLET_MAX_TRIANGLES  124u

enum {
    GEO_CACHE_HAS_MATERIALS     = 1u << 0,
    GEO_CACHE_MTLX_SIDELOAD     = 1u << 1,
    GEO_CACHE_UNTRACKED_DEPS    = 1u << 2
};

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
#define GEO_MAX_MATERIALS 1000000u
#define GEO_MAX_TEXTURES  1000000u
#define GEO_MAX_TEXTURE_BYTES (1ull << 33) /* 8 GiB */

typedef struct {
    char     magic[4];          /* "NZGC" */
    uint32_t format_version;
    int64_t  src_mtime;         /* source USD st_mtime — cache invalidation */
    int64_t  src_size;          /* source USD st_size  — cache invalidation */
    uint32_t nmeshes;
    uint32_t nlights;
    uint64_t total_geo_bytes;   /* decoded vertex+index bytes — sizes read arena */
    uint32_t nmeshlets;         /* total meshlet records (back-patched) */
    uint32_t nmeshlet_indices;  /* v4: total meshlet micro-index (uint8) entries */
    float    bounds_min[3];
    float    bounds_max[3];
    uint32_t has_authored_light;
    uint32_t has_dome_light;
    float    dome_color[3];
    float    dome_intensity;
    float    dome_rotation_y;
    char     dome_hdr_path[512];
    uint32_t nmeshlet_verts;    /* v4: total meshlet vertex-table entries */
    uint32_t nmaterials;        /* v6: material records when HAS_MATERIALS */
    uint32_t ntextures;         /* v6: texture records when HAS_MATERIALS */
    uint32_t material_flags;    /* GEO_CACHE_* flags */
    uint32_t mtlx_file_count;   /* side-loaded .mtlx dependency count */
    uint64_t mtlx_fingerprint;  /* side-loaded .mtlx mtime/size/path hash */
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

/* On-disk meshlet record (NZGC v4 — compact mesh-shader-native layout). */
typedef struct {
    float    center[3];
    float    radius;
    float    cone_axis[3];
    float    cone_cutoff;
    uint32_t vertex_count;      /* entries in this meshlet's vertex table  */
    uint32_t triangle_count;    /* triangles (micro-index count = ×3)      */
} GeoCacheMeshletRec;

/* Meshlets built for one mesh during a cache write — the compact
 * mesh-shader-native layout, tightly packed (NZGC v4). */
typedef struct {
    GeoCacheMeshletRec* recs;
    uint32_t            count;
    uint32_t*           vertices;   /* concatenated per-meshlet vertex tables */
    uint32_t            nvertices;
    unsigned char*      triangles;  /* concatenated 8-bit micro-indices       */
    uint32_t            ntriangles;
} GeoMeshletSet;

typedef struct {
    MaterialParams params;
    int32_t        shader_index;
    char           name[256];
    char           prim_path[512];
} GeoCacheMaterialRec;

typedef struct {
    int32_t  width;
    int32_t  height;
    int32_t  udim_cols;
    int32_t  udim_rows;
    int32_t  is_srgb;
    int32_t  reserved0;
    int64_t  src_mtime;
    int64_t  src_size;
    uint64_t pixel_bytes;
    char     path[512];
} GeoCacheTextureRec;

/* ---------------------------------------------------------------- helpers */

int geo_cache_enabled(void)
{
    const char* e = getenv("NUSD_GEO_CACHE");
    if (!(e && e[0] && e[0] != '0')) return 0;

    /* Legacy scene-space mutation changes cached world transforms. Keep the
     * persistent cache on the OVRTX/default authored-frame path only, so a
     * cache cooked under one camera space cannot be reused under another. */
    const char* bake = getenv("NUSD_BAKE_ZUP_TO_YUP");
    if (bake && bake[0] && bake[0] != '0') return 0;
    return 1;
}

int geo_cache_path_for(const char* usd_path, char* out_path, size_t out_cap)
{
    if (!usd_path || !out_path || out_cap == 0) return -1;
    int n = snprintf(out_path, out_cap, "%s.nzgeo", usd_path);
    if (n < 0 || (size_t)n >= out_cap) return -1;
    return 0;
}

static int wr(FILE* f, const void* p, size_t n)
{
    if (n == 0) return 0;
    return fwrite(p, 1, n, f) == n ? 0 : -1;
}

/* Read cursor over the slurped .nzgeo (iter 35). The whole cache is read into
 * one buffer up front; rd is a bounds-checked memcpy from it — that removes
 * ~1.4 M small fread calls from the warm load's Pass-1a, and lets decode
 * tasks point their encoded source straight into the buffer (no malloc/copy). */
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

/* ------------------------------------------------------------- meshlets */

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
     * meshlet building for a malformed mesh (matches renderer.c's
     * optimizeVertexCache guard). */
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

/* ----------------------------------------------------- cook-time dedup */
/*
 * geo_cache v3 folds content-identical meshes at cook time. A mesh whose
 * geometry and displayColor data byte-match an earlier mesh — and shares its
 * material_index — is stored as a SHARED reference and, on load, becomes an
 * instance of that earlier prototype. This bakes the win the renderer's
 * runtime NUSD_HASH_DEDUP otherwise recomputes on every load: a warm cache
 * load is deduped with no env flag, the BLAS build sees only unique geometry,
 * and the .nzgeo file stores each geometry once.
 *
 * The hash domain replicates renderer.c's geo_hash_compute exactly
 * (nvertices, nindices, present-mask, positions, indices, normals?,
 * texcoords?), plus displayColor metadata and per-vertex colors. A hash hit
 * is memcmp-verified, so a 64-bit collision cannot corrupt the persistent
 * cache. material_index is a separate group key: the renderer bakes it into
 * vertex data, so geometry shared between differing materials must not fold.
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

/* True when meshes a and b have byte-identical geometry and color payload
 * (the gc_geo_hash domain). Confirms a hash hit so a collision cannot
 * mis-fold the cache. */
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

static int gc_env_mtlx_sideload_enabled(void)
{
    const char* env = getenv("NUSD_MTLX_SIDELOAD");
    return !env || env[0] != '0';
}

static int gc_has_suffix(const char* s, const char* suffix)
{
    if (!s || !suffix) return 0;
    size_t ns = strlen(s), nx = strlen(suffix);
    return ns >= nx && strcmp(s + ns - nx, suffix) == 0;
}

static int gc_is_usdz_path(const char* usd_path)
{
    return gc_has_suffix(usd_path, ".usdz") || gc_has_suffix(usd_path, ".USDZ");
}

static int gc_dirname_for(const char* path, char* out, size_t cap)
{
    if (!path || !out || cap == 0) return -1;
    size_t n = strlen(path);
    if (n >= cap) return -1;
    memcpy(out, path, n + 1);
    char* slash = strrchr(out, '/');
    if (!slash) {
        if (cap < 2) return -1;
        memcpy(out, ".", 2);
    } else if (slash == out) {
        out[1] = '\0';
    } else {
        *slash = '\0';
    }
    return 0;
}

static uint64_t gc_hash_one_mtlx(const char* path, const struct stat* st)
{
    uint64_t h = 1469598103934665603ULL;
    h = gc_fnv1a(h, path, strlen(path) + 1);
    int64_t mt = (int64_t)st->st_mtime;
    int64_t sz = (int64_t)st->st_size;
    h = gc_fnv1a(h, &mt, sizeof mt);
    h = gc_fnv1a(h, &sz, sizeof sz);
    return h ? h : 0x9e3779b97f4a7c15ULL;
}

static void gc_scan_mtlx_dir(const char* dir, int depth,
                             uint64_t* fingerprint,
                             uint32_t* count,
                             int* error)
{
    if (*error || depth > 6) return;
    DIR* d = opendir(dir);
    if (!d) { *error = 1; return; }

    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;
        char path[2048];
        int n = snprintf(path, sizeof path, "%s/%s", dir, ent->d_name);
        if (n < 0 || (size_t)n >= sizeof path) { *error = 1; break; }

        struct stat st;
        if (lstat(path, &st) != 0) { *error = 1; break; }
        if (S_ISDIR(st.st_mode)) {
            gc_scan_mtlx_dir(path, depth + 1, fingerprint, count, error);
            if (*error) break;
        } else if (S_ISREG(st.st_mode) && gc_has_suffix(ent->d_name, ".mtlx")) {
            uint64_t one = gc_hash_one_mtlx(path, &st);
            *fingerprint ^= one;
            *fingerprint += one * 0x9e3779b97f4a7c15ULL;
            if (*count == UINT32_MAX) { *error = 1; break; }
            (*count)++;
        }
    }
    closedir(d);
}

static int gc_mtlx_fingerprint_for(const char* usd_path,
                                   uint64_t* fingerprint,
                                   uint32_t* count)
{
    *fingerprint = 0;
    *count = 0;
    if (!gc_env_mtlx_sideload_enabled()) return 0;

    char dir[2048];
    if (gc_dirname_for(usd_path, dir, sizeof dir)) return -1;
    int err = 0;
    gc_scan_mtlx_dir(dir, 0, fingerprint, count, &err);
    return err ? -1 : 0;
}

static int gc_texture_pixel_bytes(const MaterialTexture* t, uint64_t* out)
{
    *out = 0;
    if (!t || !t->pixels) return 0;
    if (t->width <= 0 || t->height <= 0) return -1;
    uint64_t w = (uint64_t)t->width;
    uint64_t h = (uint64_t)t->height;
    if (w > UINT64_MAX / h || w * h > UINT64_MAX / 4u) return -1;
    uint64_t bytes = w * h * 4u;
    if (bytes > GEO_MAX_TEXTURE_BYTES) return -1;
    *out = bytes;
    return 0;
}

static int gc_texture_rec_for(const char* usd_path, const MaterialTexture* t,
                              GeoCacheTextureRec* rec)
{
    memset(rec, 0, sizeof *rec);
    if (!t) return -1;
    rec->width     = t->width;
    rec->height    = t->height;
    rec->udim_cols = t->udim_cols;
    rec->udim_rows = t->udim_rows;
    rec->is_srgb   = t->is_srgb;
    memcpy(rec->path, t->path, sizeof rec->path);
    rec->path[sizeof rec->path - 1] = '\0';
    if (gc_texture_pixel_bytes(t, &rec->pixel_bytes)) return -1;

    if (rec->path[0]) {
        struct stat st;
        if (stat(rec->path, &st) == 0 && S_ISREG(st.st_mode)) {
            rec->src_mtime = (int64_t)st.st_mtime;
            rec->src_size  = (int64_t)st.st_size;
        } else if (rec->pixel_bytes > 0 && !gc_is_usdz_path(usd_path)) {
            return -1; /* external texture dependency cannot be validated */
        }
    }
    return 0;
}

static int gc_material_header_fill(const char* usd_path, const Scene* scene,
                                   GeoCacheHeader* h)
{
    MaterialCollection* mc = (MaterialCollection*)scene->materials;
    if (!mc) return 0;
    if (mc->nmaterials < 0 || mc->ntextures < 0 ||
        (uint32_t)mc->nmaterials > GEO_MAX_MATERIALS ||
        (uint32_t)mc->ntextures > GEO_MAX_TEXTURES)
        return -1;

    h->material_flags |= GEO_CACHE_HAS_MATERIALS;
    if (gc_env_mtlx_sideload_enabled())
        h->material_flags |= GEO_CACHE_MTLX_SIDELOAD;
    h->nmaterials = (uint32_t)mc->nmaterials;
    h->ntextures  = (uint32_t)mc->ntextures;
    if (gc_mtlx_fingerprint_for(usd_path, &h->mtlx_fingerprint,
                                &h->mtlx_file_count))
        return -1;

    for (int i = 0; i < mc->ntextures; i++) {
        GeoCacheTextureRec tr;
        if (gc_texture_rec_for(usd_path, &mc->textures[i], &tr))
            return -1;
    }
    return 0;
}

static int gc_material_header_matches(const char* usd_path,
                                      const GeoCacheHeader* h)
{
    uint32_t cur_count = 0;
    uint64_t cur_fp = 0;
    uint32_t cur_side = gc_env_mtlx_sideload_enabled() ? GEO_CACHE_MTLX_SIDELOAD : 0u;
    if ((h->material_flags & GEO_CACHE_MTLX_SIDELOAD) != cur_side)
        return 0;
    if (gc_mtlx_fingerprint_for(usd_path, &cur_fp, &cur_count))
        return 0;
    return cur_count == h->mtlx_file_count && cur_fp == h->mtlx_fingerprint;
}

static int gc_write_materials(FILE* f, const char* usd_path,
                              const MaterialCollection* mc)
{
    if (!mc) return 0;
    for (int i = 0; i < mc->nmaterials; i++) {
        const SceneMaterial* sm = &mc->materials[i];
        GeoCacheMaterialRec mr;
        memset(&mr, 0, sizeof mr);
        memcpy(&mr.params, &sm->params, sizeof mr.params);
        mr.shader_index = sm->shader_index;
        memcpy(mr.name, sm->name, sizeof mr.name);
        memcpy(mr.prim_path, sm->prim_path, sizeof mr.prim_path);
        mr.name[sizeof mr.name - 1] = '\0';
        mr.prim_path[sizeof mr.prim_path - 1] = '\0';
        if (wr(f, &mr, sizeof mr)) return -1;
    }
    for (int i = 0; i < mc->ntextures; i++) {
        const MaterialTexture* t = &mc->textures[i];
        GeoCacheTextureRec tr;
        if (gc_texture_rec_for(usd_path, t, &tr)) return -1;
        if (wr(f, &tr, sizeof tr)) return -1;
        if (tr.pixel_bytes > 0 && wr(f, t->pixels, (size_t)tr.pixel_bytes))
            return -1;
    }
    return 0;
}

static int gc_skip_bytes(GeoReader* r, uint64_t n)
{
    if (n > r->size - r->pos) return -1;
    r->pos += (size_t)n;
    return 0;
}

static int gc_read_materials(GeoReader* reader, const char* usd_path,
                             const GeoCacheHeader* h, int want_materials,
                             Scene* scene)
{
    if (!(h->material_flags & GEO_CACHE_HAS_MATERIALS)) return 0;
    if (h->nmaterials > GEO_MAX_MATERIALS || h->ntextures > GEO_MAX_TEXTURES)
        return -1;

    MaterialCollection* mc = NULL;
    if (want_materials) {
        mc = (MaterialCollection*)calloc(1, sizeof(MaterialCollection));
        if (!mc) return -1;
        mc->nmaterials = (int)h->nmaterials;
        mc->ntextures  = (int)h->ntextures;
        if (h->nmaterials > 0) {
            mc->materials = (SceneMaterial*)calloc(h->nmaterials,
                                                   sizeof(SceneMaterial));
            if (!mc->materials) { materials_free(mc); return -1; }
        }
        if (h->ntextures > 0) {
            mc->textures = (MaterialTexture*)calloc(h->ntextures,
                                                    sizeof(MaterialTexture));
            if (!mc->textures) { materials_free(mc); return -1; }
        }
        scene->materials = mc;
    }

    for (uint32_t i = 0; i < h->nmaterials; i++) {
        GeoCacheMaterialRec mr;
        if (rd(reader, &mr, sizeof mr)) return -1;
        if (mc) {
            SceneMaterial* sm = &mc->materials[i];
            memcpy(&sm->params, &mr.params, sizeof sm->params);
            sm->shader_index = mr.shader_index;
            memcpy(sm->name, mr.name, sizeof sm->name);
            memcpy(sm->prim_path, mr.prim_path, sizeof sm->prim_path);
            sm->name[sizeof sm->name - 1] = '\0';
            sm->prim_path[sizeof sm->prim_path - 1] = '\0';
        }
    }

    for (uint32_t i = 0; i < h->ntextures; i++) {
        GeoCacheTextureRec tr;
        if (rd(reader, &tr, sizeof tr)) return -1;
        if (tr.pixel_bytes > GEO_MAX_TEXTURE_BYTES ||
            tr.pixel_bytes > reader->size - reader->pos)
            return -1;
        if (tr.width < 0 || tr.height < 0) return -1;

        if (want_materials && tr.path[0] && tr.pixel_bytes > 0 &&
            !gc_is_usdz_path(usd_path)) {
            if (tr.src_size <= 0 && tr.src_mtime <= 0)
                return -1;
            struct stat st;
            if (stat(tr.path, &st) != 0 || !S_ISREG(st.st_mode) ||
                tr.src_mtime != (int64_t)st.st_mtime ||
                tr.src_size  != (int64_t)st.st_size)
                return -1;
        }

        if (mc) {
            MaterialTexture* t = &mc->textures[i];
            t->width     = tr.width;
            t->height    = tr.height;
            t->udim_cols = tr.udim_cols;
            t->udim_rows = tr.udim_rows;
            t->is_srgb   = tr.is_srgb;
            memcpy(t->path, tr.path, sizeof t->path);
            t->path[sizeof t->path - 1] = '\0';
            if (tr.pixel_bytes > 0) {
                t->pixels = (unsigned char*)malloc((size_t)tr.pixel_bytes);
                if (!t->pixels) return -1;
                if (rd(reader, t->pixels, (size_t)tr.pixel_bytes)) return -1;
            }
        } else if (gc_skip_bytes(reader, tr.pixel_bytes)) {
            return -1;
        }
    }
    return 0;
}

/* Open-addressing table: geometry hash -> first-occurrence mesh index. */
typedef struct { uint64_t hash; int mesh; } GcHashEntry;

/* Fill eff[0..N) so eff[i] is the mesh whose stored geometry mesh i should
 * reference: i itself when i is a prototype, an earlier mesh otherwise.
 * eff[i] <= i is guaranteed (load + nu_attach_scene need the prototype to
 * precede the instance). Composes the scene's existing USD-instance
 * prototype_idx with cook-time content dedup; falls back to the
 * USD-instance-only mapping when dedup is disabled or scratch alloc fails. */
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
                if (s->meshes[cand].material_index == m->material_index &&
                    gc_geo_equal(&s->meshes[cand], m))
                    leader = cand;             /* confirmed content duplicate */
                break;                         /* one slot per hash (runtime) */
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

    /* Phase C — point every non-leader at the root content leader. After
     * phase B, eff[i] (for eff[i]!=i) names a geometry-bearing mesh whose
     * own eff[] is already a root, so a single hop flattens the chain. */
    for (int i = 0; i < N; i++)
        if (eff[i] != i)
            eff[i] = eff[eff[i]];
}

int geo_cache_write(const char* usd_path, const Scene* scene)
{
    if (!geo_cache_enabled() || !usd_path || !scene) return -1;
    if (scene->nmeshes <= 0)  return -1;
    if (scene->ncurves > 0)   return -1;   /* Phase 1 */
    for (int i = 0; i < scene->nmeshes; i++) {
        if (scene->meshes[i].ptex_tri_colors ||
            scene->meshes[i].ptex_tri_color_count > 0)
            return -1;
    }
    MaterialCollection* mc = (MaterialCollection*)scene->materials;
    if (mc) {
        for (int i = 0; i < mc->nmaterials; i++) {
            if (mc->materials[i].ptex_color_path[0] ||
                mc->materials[i].has_ptex_average_color)
                return -1;
        }
    }

    struct stat st;
    if (stat(usd_path, &st) != 0) return -1;

    char path[2048];
    if (geo_cache_path_for(usd_path, path, sizeof path)) return -1;

    /* Cook-phase profiling (iter 39): setup+dedup / encode / meshlet-build /
     * meshlet-write checkpoints — mirrors the warm load's gc_t* profile. */
    struct timespec cook_t0, cook_t1, cook_t2, cook_t3, cook_t4;
    clock_gettime(CLOCK_MONOTONIC, &cook_t0);

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
    memcpy(h.magic, "NZGC", 4);
    h.format_version  = NZGC_FORMAT_VERSION;
    h.src_mtime       = (int64_t)st.st_mtime;
    h.src_size        = (int64_t)st.st_size;
    h.nmeshes         = (uint32_t)scene->nmeshes;
    h.nlights         = (uint32_t)(scene->nlights > 0 ? scene->nlights : 0);
    h.total_geo_bytes = total;
    h.nmeshlets         = 0;   /* back-patched after the meshlet section */
    h.nmeshlet_indices  = 0;
    memcpy(h.bounds_min, scene->bounds_min, sizeof h.bounds_min);
    memcpy(h.bounds_max, scene->bounds_max, sizeof h.bounds_max);
    h.has_authored_light = scene->has_authored_light ? 1u : 0u;
    h.has_dome_light     = scene->has_dome_light ? 1u : 0u;
    memcpy(h.dome_color, scene->dome_color, sizeof h.dome_color);
    h.dome_intensity  = scene->dome_intensity;
    h.dome_rotation_y = scene->dome_rotation_y;
    memcpy(h.dome_hdr_path, scene->dome_hdr_path, sizeof h.dome_hdr_path);
    if (gc_material_header_fill(usd_path, scene, &h)) {
        fclose(f);
        free(eff);
        remove(path);
        return -1;
    }

    int err = wr(f, &h, sizeof h);

    /* Section-size profiling (iter 31): ftell checkpoints decompose the
     * .nzgeo into mesh-header / vertex / index / light / meshlet bytes. */
    long gc_p_hdr = ftell(f);
    long gc_b_meshhdr = 0, gc_b_vtx = 0, gc_b_idx = 0, gc_b_mat = 0;
    long gc_b_mrec = 0, gc_b_mvtx = 0, gc_b_mtri = 0;

    clock_gettime(CLOCK_MONOTONIC, &cook_t1);   /* setup + dedup done */
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
        long gc_m0 = ftell(f);
        err |= wr(f, &mh, sizeof mh);
        long gc_m1 = ftell(f);

        err |= write_vtx_blob(f, m->positions, m->nvertices, 3, is_shared);
        err |= write_vtx_blob(f, m->normals,   m->nvertices, 3, is_shared);
        err |= write_vtx_blob(f, m->colors,    m->nvertices, 3, is_shared);
        err |= write_vtx_blob(f, m->texcoords, m->nvertices, 2, is_shared);
        long gc_m2 = ftell(f);
        err |= write_idx_blob(f, m->indices,   m->nindices, m->nvertices,
                              is_shared);
        long gc_m3 = ftell(f);
        gc_b_meshhdr += gc_m1 - gc_m0;
        gc_b_vtx     += gc_m2 - gc_m1;
        gc_b_idx     += gc_m3 - gc_m2;
    }

    long gc_p_meshes = ftell(f);
    if (!err && scene->nlights > 0 && scene->lights)
        err |= wr(f, scene->lights, (size_t)scene->nlights * sizeof(SceneLight));
    long gc_p_lights = ftell(f);
    long gc_p_materials0 = ftell(f);
    if (!err && (h.material_flags & GEO_CACHE_HAS_MATERIALS)) {
        err |= gc_write_materials(f, usd_path,
                                  (const MaterialCollection*)scene->materials);
    }
    long gc_p_materials = ftell(f);
    gc_b_mat = gc_p_materials - gc_p_materials0;
    clock_gettime(CLOCK_MONOTONIC, &cook_t2);   /* vertex/index encode done */

    /* Meshlet section: one [u32 count][records][vtx blob][tri blob] group per
     * mesh. Meshlets are built and compacted per prototype; instances write
     * count 0 and inherit the prototype's range on load. The compact layout
     * is stored raw — the warm load reads it with no decode and no compaction.
     *
     * iter 39: the per-prototype meshlet build (buildMeshlets + optimizeMeshlet
     * + computeMeshletBounds) is the cook's dominant CPU cost and is
     * embarrassingly parallel — geo_build_meshlets is a pure function with
     * private scratch. Phase A collects the leaders; Phase B builds every
     * leader's meshlets in parallel, each into its own slot; Phase C writes the
     * groups in mesh order. The file bytes are identical to the sequential
     * cook — only the build order changes. */
    uint32_t total_meshlets = 0, total_meshlet_verts = 0, total_meshlet_tris = 0;
    GeoMeshletSet* sets    = (GeoMeshletSet*)calloc((size_t)nmeshes,
                                                    sizeof(GeoMeshletSet));
    int*           leaders = (int*)malloc((size_t)nmeshes * sizeof(int));
    int            nleaders = 0;
    if (!sets || !leaders) err = 1;

    if (!err) {
        for (int i = 0; i < nmeshes; i++)
            if (eff[i] == i) leaders[nleaders++] = i;

        /* Phase B — parallel: build each leader's meshlets into its own slot.
         * geo_build_meshlets only reads scene geometry and writes sets[i] plus
         * its own private mallocs, so the region is data-race-free. */
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int k = 0; k < nleaders; k++) {
            int i = leaders[k];
            const SceneMesh* m = &scene->meshes[i];
            geo_build_meshlets(m->positions, m->nvertices,
                               m->indices, m->nindices, &sets[i]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &cook_t3);   /* parallel meshlet build done */

    /* Phase C — sequential: write the meshlet groups in mesh order. */
    for (int i = 0; i < nmeshes && !err; i++) {
        GeoMeshletSet* set = &sets[i];
        uint32_t mc = set->count;
        err |= wr(f, &mc, sizeof mc);
        if (!err && mc > 0) {
            long gc_ml0 = ftell(f);
            err |= wr(f, set->recs, (size_t)mc * sizeof(GeoCacheMeshletRec));
            long gc_ml1 = ftell(f);
            err |= wr(f, set->vertices, (size_t)set->nvertices * sizeof(uint32_t));
            long gc_ml2 = ftell(f);
            err |= wr(f, set->triangles, (size_t)set->ntriangles);
            long gc_ml3 = ftell(f);
            gc_b_mrec += gc_ml1 - gc_ml0;
            gc_b_mvtx += gc_ml2 - gc_ml1;
            gc_b_mtri += gc_ml3 - gc_ml2;
            total_meshlets      += mc;
            total_meshlet_verts += set->nvertices;
            total_meshlet_tris  += set->ntriangles;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &cook_t4);   /* meshlet write done */

    if (sets)
        for (int i = 0; i < nmeshes; i++) geo_meshlet_set_free(&sets[i]);
    free(sets);
    free(leaders);

    if (!err)
        fprintf(stderr, "[geo_cache] meshlets built: %u (%u verts, %u tris)\n",
                total_meshlets, total_meshlet_verts, total_meshlet_tris);

    if (!err) {
        long gc_p_meshlets = ftell(f);
        fprintf(stderr,
            "[geo_cache] section bytes: header %.2f, mesh-hdr %.1f, vertex %.1f, "
            "index %.1f, lights %.2f, materials %.1f, meshlet-recs %.1f, meshlet-vtx %.1f, "
            "meshlet-tri %.1f (total %.1f MB)\n",
            gc_p_hdr / 1048576.0,
            gc_b_meshhdr / 1048576.0, gc_b_vtx / 1048576.0, gc_b_idx / 1048576.0,
            (gc_p_lights - gc_p_meshes) / 1048576.0,
            gc_b_mat / 1048576.0,
            gc_b_mrec / 1048576.0, gc_b_mvtx / 1048576.0, gc_b_mtri / 1048576.0,
            gc_p_meshlets / 1048576.0);
    }

    if (!err) {
        double d_setup = (cook_t1.tv_sec-cook_t0.tv_sec) + (cook_t1.tv_nsec-cook_t0.tv_nsec)*1e-9;
        double d_enc   = (cook_t2.tv_sec-cook_t1.tv_sec) + (cook_t2.tv_nsec-cook_t1.tv_nsec)*1e-9;
        double d_build = (cook_t3.tv_sec-cook_t2.tv_sec) + (cook_t3.tv_nsec-cook_t2.tv_nsec)*1e-9;
        double d_mlwr  = (cook_t4.tv_sec-cook_t3.tv_sec) + (cook_t4.tv_nsec-cook_t3.tv_nsec)*1e-9;
        fprintf(stderr,
            "[geo_cache] cook phases (ms): setup+dedup %.1f, vtx/idx encode %.1f, "
            "meshlet build (parallel) %.1f, meshlet write %.1f\n",
            d_setup*1e3, d_enc*1e3, d_build*1e3, d_mlwr*1e3);
    }

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

/* iter 34: blob decode is collected as independent tasks during the
 * sequential read pass, then run in parallel — meshopt vertex/index decode
 * is ~61% of the warm load and the per-prototype blobs are independent.
 * Arena allocation stays in the sequential pass, so the parallel decode
 * region touches no shared mutable state. */
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
 * alloc the destination, and append a decode task. On SHARED, leaves *out
 * NULL and sets *shared_flag (pass 2 wires it to the prototype). The actual
 * decode runs later, in parallel, via decode_tasklist_run. */
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

Scene* geo_cache_try_load(const char* usd_path, int want_materials)
{
    if (!geo_cache_enabled() || !usd_path) return NULL;

    /* Warm-load phase profiling (iter 31). */
    struct timespec gc_t0, gc_t1, gc_t2, gc_t3, gc_t4;
    double gc_dec_s = 0.0;
    DecodeTaskList tl = {0};
    GeoReader reader = {0};
    uint8_t*  filebuf = NULL;
    size_t    filebuf_size = 0;
    clock_gettime(CLOCK_MONOTONIC, &gc_t0);

    char path[2048];
    if (geo_cache_path_for(usd_path, path, sizeof path)) return NULL;

    struct stat st;
    if (stat(usd_path, &st) != 0) return NULL;

    /* Read the whole cache into one buffer; the warm load then parses from
     * memory — a bounds-checked memcpy per read instead of ~1.4 M fread
     * calls — and decode tasks point their encoded source straight into the
     * buffer (no per-blob malloc/copy). (An mmap variant was measured and
     * rejected — see the iter 35 roadmap entry: lazy faulting and
     * MAP_POPULATE both regressed the cook→load path because populating the
     * freshly-written file's pages into a new process is slow.) */
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat cst;
    if (fstat(fd, &cst) != 0 || cst.st_size < (off_t)sizeof(GeoCacheHeader)) {
        close(fd); return NULL;
    }
    filebuf_size = (size_t)cst.st_size;
    filebuf = (uint8_t*)malloc(filebuf_size);
    if (!filebuf) { close(fd); return NULL; }

    /* Parallel slurp (iter 40): a single fread of the whole cache dominates
     * the warm load's setup phase — most of its cost is faulting + filling
     * ~200 MB of fresh anonymous pages single-threaded. pread carries its own
     * offset and is thread-safe, so N threads fault + fill disjoint chunks at
     * once. */
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
    reader.base = filebuf;
    reader.size = filebuf_size;
    reader.pos  = 0;

    GeoCacheHeader h;
    if (rd(&reader, &h, sizeof h)) { free(filebuf); return NULL; }
    if (memcmp(h.magic, "NZGC", 4) != 0 ||
        h.format_version != NZGC_FORMAT_VERSION ||
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
    if (want_materials) {
        if (!(h.material_flags & GEO_CACHE_HAS_MATERIALS) ||
            !gc_material_header_matches(usd_path, &h)) {
            free(filebuf);
            return NULL;
        }
    }

    Scene*         scene  = (Scene*)calloc(1, sizeof(Scene));
    Arena*         arena  = (Arena*)malloc(sizeof(Arena));
    unsigned char* shared = NULL;
    if (!scene || !arena) { free(scene); free(arena); free(filebuf); return NULL; }

    /* v4 stores the compact meshlet layout directly: meshlet_vertices is
     * nmeshlet_verts uint32 entries, meshlet_triangles nmeshlet_indices
     * 8-bit micro-indices. */
    size_t arena_sz = (size_t)h.total_geo_bytes
                    + (size_t)h.nmeshes * sizeof(SceneMesh)
                    + (size_t)h.nlights * sizeof(SceneLight)
                    + (size_t)h.nmeshlets * sizeof(SceneMeshlet)
                    + (size_t)h.nmeshlet_verts * sizeof(uint32_t)
                    + (size_t)h.nmeshlet_indices
                    + (1u << 16);
    if (arena_sz < (64u << 20)) arena_sz = 64u << 20;
    *arena = arena_create(arena_sz);
    if (!arena->head) { free(scene); free(arena); free(filebuf); return NULL; }

    scene->_arena      = arena;
    scene->_stage      = NULL;
    scene->_owns_stage = 0;
    scene->_meshes_heap = 0;
    scene->materials   = NULL;
    scene->curves      = NULL;
    scene->ncurves     = 0;
    memcpy(scene->bounds_min, h.bounds_min, sizeof scene->bounds_min);
    memcpy(scene->bounds_max, h.bounds_max, sizeof scene->bounds_max);
    scene->has_authored_light = h.has_authored_light ? 1 : 0;
    scene->has_dome_light = h.has_dome_light ? 1 : 0;
    memcpy(scene->dome_color, h.dome_color, sizeof scene->dome_color);
    memcpy(scene->dome_hdr_path, h.dome_hdr_path, sizeof scene->dome_hdr_path);
    scene->dome_intensity  = h.dome_intensity;
    scene->dome_rotation_y = h.dome_rotation_y;

    scene->meshes = (SceneMesh*)arena_calloc(arena, h.nmeshes, sizeof(SceneMesh));
    if (!scene->meshes) goto fail;
    scene->nmeshes = (int)h.nmeshes;

    shared = (unsigned char*)calloc((size_t)h.nmeshes, 5);
    if (!shared) goto fail;

    clock_gettime(CLOCK_MONOTONIC, &gc_t1);   /* setup done */

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
        m->lazy_prim_idx     = -1;
        m->vertex_offset     = 0;
        m->index_offset      = 0;

        unsigned char* sf = &shared[(size_t)i * 5];
        if (read_vtx_blob(&reader, arena, m->nvertices, 3, &m->positions, &sf[0], &tl)) goto fail;
        if (read_vtx_blob(&reader, arena, m->nvertices, 3, &m->normals,   &sf[1], &tl)) goto fail;
        if (read_vtx_blob(&reader, arena, m->nvertices, 3, &m->colors,    &sf[2], &tl)) goto fail;
        if (read_vtx_blob(&reader, arena, m->nvertices, 2, &m->texcoords, &sf[3], &tl)) goto fail;
        if (read_idx_blob(&reader, arena, m->nindices, &m->indices, &sf[4], &tl)) goto fail;
    }

    /* Pass 1b — parallel: decode the collected blobs (no file, no arena). */
    {
        struct timespec da, db;
        clock_gettime(CLOCK_MONOTONIC, &da);
        int derr = decode_tasklist_run(&tl);
        clock_gettime(CLOCK_MONOTONIC, &db);
        gc_dec_s = (db.tv_sec - da.tv_sec) + (db.tv_nsec - da.tv_nsec) * 1e-9;
        if (derr) goto fail;
    }
    decode_tasklist_free(&tl);

    clock_gettime(CLOCK_MONOTONIC, &gc_t2);   /* pass 1 (read + parallel decode) done */

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
    free(shared);
    shared = NULL;

    if (h.nlights > 0 && h.nlights < 10000000u) {
        scene->lights = (SceneLight*)arena_calloc(arena, h.nlights,
                                                  sizeof(SceneLight));
        if (!scene->lights) goto fail;
        if (rd(&reader, scene->lights, (size_t)h.nlights * sizeof(SceneLight))) goto fail;
        scene->nlights = (int)h.nlights;
        if (scene->nlights > 0)
            scene->has_authored_light = 1;
    }
    if (gc_read_materials(&reader, usd_path, &h, want_materials, scene))
        goto fail;
    clock_gettime(CLOCK_MONOTONIC, &gc_t3);   /* pass 2 (wire) + lights done */

    /* Meshlet section (NZGC v4): the cache stores the compact mesh-shader-
     * native layout directly — per-meshlet records + a raw vertex-table blob
     * + a raw 8-bit micro-index blob. The warm load is a straight read; the
     * load-time hash-dedup compaction is gone (it ran ~107 ms on full DSX). */
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

    clock_gettime(CLOCK_MONOTONIC, &gc_t4);   /* meshlet read done */
    {
        double d_setup = (gc_t1.tv_sec-gc_t0.tv_sec) + (gc_t1.tv_nsec-gc_t0.tv_nsec)*1e-9;
        double d_pass1 = (gc_t2.tv_sec-gc_t1.tv_sec) + (gc_t2.tv_nsec-gc_t1.tv_nsec)*1e-9;
        double d_pass2 = (gc_t3.tv_sec-gc_t2.tv_sec) + (gc_t3.tv_nsec-gc_t2.tv_nsec)*1e-9;
        double d_mesh  = (gc_t4.tv_sec-gc_t3.tv_sec) + (gc_t4.tv_nsec-gc_t3.tv_nsec)*1e-9;
        fprintf(stderr,
            "[geo_cache] load phases (ms): setup %.1f, blob-read+decode %.1f "
            "(parallel-decode %.1f), instance-wire %.1f, meshlet-read %.1f\n",
            d_setup*1e3, d_pass1*1e3, gc_dec_s*1e3, d_pass2*1e3, d_mesh*1e3);
    }

    free(filebuf);
    return scene;

fail:
    decode_tasklist_free(&tl);
    free(shared);
    if (scene && scene->materials) {
        materials_free((MaterialCollection*)scene->materials);
        scene->materials = NULL;
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
     * geometry-only and writes <usd>.nzgeo on a miss — CPU-only, no GPU. */
    Scene* s = scene_load(usd_path);
    if (!s) return -1;
    int ok = (s->nmeshes > 0);
    scene_free(s);
    return ok ? 0 : -1;
}

/* Like geo_cache_cook but intentionally does NOT free the parsed Scene.
 * scene_free on a full-DSX scene is ~6.8 s — almost entirely nanousd_close,
 * the USD-stage teardown — and that is pure waste in a process about to
 * _exit. Use only for the final scene of a short-lived cook run (geo_cook):
 * the OS reclaims the leaked stage on exit. Long-lived callers must use
 * geo_cache_cook. */
int geo_cache_cook_keep(const char* usd_path)
{
    if (!usd_path) return -1;
    setenv("NUSD_GEO_CACHE", "1", 1);
    Scene* s = scene_load(usd_path);
    if (!s) return -1;
    return (s->nmeshes > 0) ? 0 : -1;   /* Scene intentionally leaked */
}
