// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * viewer.c -- Render core for the OpenGL renderer.
 *
 * NO glfw* function calls live in this file. GLFW windowing has been
 * extracted to viewer_window.c so libnusd_gles.dylib (Python-loaded via
 * ctypes by ovgear) does not statically pull in GLFW; that would
 * register a duplicate set of Cocoa classes (GLFWWindow, GLFWContentView,
 * …) when ovui's Homebrew libglfw is also in the process.
 *
 * The header <GLFW/glfw3.h> is included because handle_key uses the
 * GLFW_KEY_* / GLFW_PRESS macros, but headers don't link any library
 * — only function calls do, and there are none here.
 *
 * Stripped from the Vulkan viewer: no RT and no DLSS. Raster shadow maps are
 * implemented for authored material-light scenes.
 * Supports RASTER and MATERIAL modes only.
 *
 * Key GLES adaptation: the default raster path mixes 16-bit relative
 * indices (via glDrawElementsBaseVertex) with 32-bit absolute indices.
 * Set NUSD_GL_MIXED_INDICES=0 to force the older all-32-bit path.
 */

#include "viewer_internal.h"
#include "shaders_gles.h"
#ifdef NUSD_HAVE_PTEX
#include "ptex_material.h"
#endif
#include <nanousd/nanousdapi.h>
#include <meshoptimizer.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#ifdef __GLIBC__
#include <malloc.h>  /* malloc_trim() — lets scene_release_mesh_payloads
                     * return freed CPU pages to the kernel so VmRSS
                     * reflects the drop. */
#endif

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(MaterialParams) == sizeof(GpuMaterialParams),
               "MaterialParams and GpuMaterialParams must stay layout-compatible");
#endif

static const char* mode_names[] = { "RASTER", "MATERIAL" };

static int dome_visible_fallback_path(const char* hdr_path,
                                      char* out, size_t out_size)
{
    if (!hdr_path || !out || out_size == 0) return 0;
    const char* dot = strrchr(hdr_path, '.');
    if (!dot) return 0;
    size_t stem_len = (size_t)(dot - hdr_path);
    const char suffix[] = "VIS.png";
    if (stem_len + sizeof(suffix) > out_size) return 0;
    memcpy(out, hdr_path, stem_len);
    memcpy(out + stem_len, suffix, sizeof(suffix));
    return access(out, R_OK) == 0;
}

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

/* ---- Math helpers ---- */

static void mat4_mul(const float a[16], const float b[16], float out[16])
{
    float tmp[16];
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            tmp[r * 4 + c] = a[r * 4 + 0] * b[0 * 4 + c]
                            + a[r * 4 + 1] * b[1 * 4 + c]
                            + a[r * 4 + 2] * b[2 * 4 + c]
                            + a[r * 4 + 3] * b[3 * 4 + c];
        }
    }
    memcpy(out, tmp, sizeof(float) * 16);
}

static void mat4_d2f(const double src[16], float dst[16])
{
    for (int i = 0; i < 16; i++)
        dst[i] = (float)src[i];
}

static void mat4_transpose(const float src[16], float dst[16])
{
    float tmp[16];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            tmp[r * 4 + c] = src[c * 4 + r];
    memcpy(dst, tmp, sizeof(float) * 16);
}

static int aabb_outside_planes(const float planes[6][4],
                               const float bmin[3],
                               const float bmax[3])
{
    for (int p = 0; p < 6; p++) {
        float a = planes[p][0], b = planes[p][1], c = planes[p][2], d = planes[p][3];
        float px = a > 0 ? bmax[0] : bmin[0];
        float py = b > 0 ? bmax[1] : bmin[1];
        float pz = c > 0 ? bmax[2] : bmin[2];
        if (a*px + b*py + c*pz + d < 0.0f)
            return 1;
    }
    return 0;
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

static int mat4_inverse(const float m[16], float out[16])
{
    float inv[16];
    inv[ 0] =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15]
             + m[9]*m[7]*m[14]  + m[13]*m[6]*m[11]  - m[13]*m[7]*m[10];
    inv[ 4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14]  + m[8]*m[6]*m[15]
             - m[8]*m[7]*m[14]  - m[12]*m[6]*m[11]  + m[12]*m[7]*m[10];
    inv[ 8] =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13]  - m[8]*m[5]*m[15]
             + m[8]*m[7]*m[13]  + m[12]*m[5]*m[11]  - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13]  + m[8]*m[5]*m[14]
             - m[8]*m[6]*m[13]  - m[12]*m[5]*m[10]  + m[12]*m[6]*m[9];
    inv[ 1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14]  + m[9]*m[2]*m[15]
             - m[9]*m[3]*m[14]  - m[13]*m[2]*m[11]  + m[13]*m[3]*m[10];
    inv[ 5] =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14]  - m[8]*m[2]*m[15]
             + m[8]*m[3]*m[14]  + m[12]*m[2]*m[11]  - m[12]*m[3]*m[10];
    inv[ 9] = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13]  + m[8]*m[1]*m[15]
             - m[8]*m[3]*m[13]  - m[12]*m[1]*m[11]  + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13]  - m[8]*m[1]*m[14]
             + m[8]*m[2]*m[13]  + m[12]*m[1]*m[10]  - m[12]*m[2]*m[9];
    inv[ 2] =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]   - m[5]*m[2]*m[15]
             + m[5]*m[3]*m[14]  + m[13]*m[2]*m[7]   - m[13]*m[3]*m[6];
    inv[ 6] = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]   + m[4]*m[2]*m[15]
             - m[4]*m[3]*m[14]  - m[12]*m[2]*m[7]   + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]   - m[4]*m[1]*m[15]
             + m[4]*m[3]*m[13]  + m[12]*m[1]*m[7]   - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]   + m[4]*m[1]*m[14]
             - m[4]*m[2]*m[13]  - m[12]*m[1]*m[6]   + m[12]*m[2]*m[5];
    inv[ 3] = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]   + m[5]*m[2]*m[11]
             - m[5]*m[3]*m[10]  - m[9]*m[2]*m[7]    + m[9]*m[3]*m[6];
    inv[ 7] =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]   - m[4]*m[2]*m[11]
             + m[4]*m[3]*m[10]  + m[8]*m[2]*m[7]    - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]    + m[4]*m[1]*m[11]
             - m[4]*m[3]*m[9]   - m[8]*m[1]*m[7]    + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]    - m[4]*m[1]*m[10]
             + m[4]*m[2]*m[9]   + m[8]*m[1]*m[6]    - m[8]*m[2]*m[5];
    float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (det == 0.0f) return 0;
    det = 1.0f / det;
    for (int i = 0; i < 16; i++) out[i] = inv[i] * det;
    return 1;
}

static float vec3_dot(const float a[3], const float b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static void vec3_cross(const float a[3], const float b[3], float out[3])
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static float vec3_normalize(float v[3])
{
    float len = sqrtf(vec3_dot(v, v));
    if (len > 1e-8f) {
        v[0] /= len; v[1] /= len; v[2] /= len;
    }
    return len;
}

static void mat4_transform_point(const float m[16], const float p[3], float out[3])
{
    float x = m[0]*p[0] + m[1]*p[1] + m[2]*p[2] + m[3];
    float y = m[4]*p[0] + m[5]*p[1] + m[6]*p[2] + m[7];
    float z = m[8]*p[0] + m[9]*p[1] + m[10]*p[2] + m[11];
    float w = m[12]*p[0] + m[13]*p[1] + m[14]*p[2] + m[15];
    if (fabsf(w) > 1e-8f) {
        float inv_w = 1.0f / w;
        x *= inv_w; y *= inv_w; z *= inv_w;
    }
    out[0] = x; out[1] = y; out[2] = z;
}

static void mat4_make_lookat(const float eye[3], const float target[3],
                             const float up_in[3], float out[16])
{
    float f[3] = { target[0]-eye[0], target[1]-eye[1], target[2]-eye[2] };
    if (vec3_normalize(f) <= 1e-8f) {
        f[0] = 0.0f; f[1] = 0.0f; f[2] = -1.0f;
    }
    float up[3] = { up_in[0], up_in[1], up_in[2] };
    if (vec3_normalize(up) <= 1e-8f) {
        up[0] = 0.0f; up[1] = 0.0f; up[2] = 1.0f;
    }
    float s[3];
    vec3_cross(f, up, s);
    if (vec3_normalize(s) <= 1e-8f) {
        float alt[3] = { 1.0f, 0.0f, 0.0f };
        if (fabsf(f[0]) > 0.8f) {
            alt[0] = 0.0f; alt[1] = 1.0f; alt[2] = 0.0f;
        }
        vec3_cross(f, alt, s);
        vec3_normalize(s);
    }
    float u[3];
    vec3_cross(s, f, u);

    out[0] =  s[0];  out[1] =  s[1];  out[2]  =  s[2];  out[3]  = -vec3_dot(s, eye);
    out[4] =  u[0];  out[5] =  u[1];  out[6]  =  u[2];  out[7]  = -vec3_dot(u, eye);
    out[8] = -f[0];  out[9] = -f[1];  out[10] = -f[2];  out[11] =  vec3_dot(f, eye);
    out[12] = 0.0f;  out[13] = 0.0f;  out[14] = 0.0f;  out[15] = 1.0f;
}

static void mat4_make_ortho_bounds(float l, float r, float b, float t,
                                   float zn, float zf, float out[16])
{
    float w = r - l;
    float h = t - b;
    float d = zf - zn;
    if (fabsf(w) < 1e-5f) { l -= 0.5f; r += 0.5f; w = r - l; }
    if (fabsf(h) < 1e-5f) { b -= 0.5f; t += 0.5f; h = t - b; }
    if (fabsf(d) < 1e-5f) { zn -= 0.5f; zf += 0.5f; d = zf - zn; }
    memset(out, 0, sizeof(float) * 16);
    out[0] = 2.0f / w;
    out[3] = -(r + l) / w;
    out[5] = 2.0f / h;
    out[7] = -(t + b) / h;
    out[10] = 2.0f / d;
    out[11] = -(zf + zn) / d;
    out[15] = 1.0f;
}

/* ---- System stats helpers ---- */

static long read_rss_kb(void)
{
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    long rss = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%ld", &rss);
            break;
        }
    }
    fclose(f);
    return rss;
}

static int read_gpu_mem_mb(Gpu* gpu)
{
    return (int)(gpu_get_allocated_memory(gpu) / (1024 * 1024));
}

static int read_cpu_temp_c(void)
{
    const char* paths[] = {
        "/sys/class/thermal/thermal_zone11/temp",
        "/sys/class/thermal/thermal_zone0/temp",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        FILE* f = fopen(paths[i], "r");
        if (f) {
            int milli = 0;
            if (fscanf(f, "%d", &milli) != 1) milli = 0;
            fclose(f);
            if (milli > 0) return milli / 1000;
        }
    }
    return 0;
}


/* ---- Merged buffer builders ---- */

static unsigned char* build_geometry_needed_mask(Scene* scene)
{
    if (!scene || scene->nmeshes <= 0) return NULL;

    unsigned char* needed = (unsigned char*)calloc((size_t)scene->nmeshes, 1);
    if (!needed) return NULL;

    for (int m = 0; m < scene->nmeshes; m++) {
        SceneMesh* mesh = &scene->meshes[m];
        if (mesh->nvertices <= 0 || mesh->nindices <= 0) continue;
        if (mesh->prototype_idx == m && !mesh->is_proto_only)
            needed[m] = 1;
    }

    for (int m = 0; m < scene->nmeshes; m++) {
        SceneMesh* mesh = &scene->meshes[m];
        if (mesh->nvertices <= 0 || mesh->nindices <= 0) continue;
        if (mesh->is_proto_only) continue;
        int proto = mesh->prototype_idx;
        if (proto >= 0 && proto < scene->nmeshes && proto != m)
            needed[proto] = 1;
    }

    /* Compact-PI batch prototypes are referenced only by scene->pi_batches (not
     * by any non-proto-only SceneMesh), so mark them needed here — else their
     * geometry never lands in the merged VBO/IB and the instanced draw is empty. */
    for (int b = 0; b < scene->npi_batches; b++) {
        int p = scene->pi_batches[b].prototype_mesh_idx;
        if (p >= 0 && p < scene->nmeshes &&
            scene->meshes[p].nvertices > 0 && scene->meshes[p].nindices > 0)
            needed[p] = 1;
    }

    return needed;
}

typedef struct {
    uint32_t vertices;
    uint32_t indices;
    uint64_t index_bytes;
} RasterChunk;

typedef struct {
    float   position[3];
    int16_t normal[4];
    uint8_t color[4];
} RasterVertexPacked;

typedef struct {
    float    position[3];
    float    width;
    uint32_t color_rgba8;
} CurveVertexPacked;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(RasterVertexPacked) == 24,
               "RasterVertexPacked must remain a 24-byte GL vertex");
_Static_assert(sizeof(CurveVertexPacked) == 20,
               "CurveVertexPacked must remain a 20-byte GL vertex");
#endif

#define RASTER_CHUNK_MAX_VERTICES 32000000u
#define RASTER_CHUNK_MAX_INDEX_BYTES (768ull * 1024ull * 1024ull)

static uint64_t align_up_u64(uint64_t v, uint64_t align)
{
    return (v + align - 1u) & ~(align - 1u);
}

static int raster_mesh_index_bits(const SceneMesh* mesh)
{
    return (mesh && mesh->nvertices <= 65535) ? 16 : 32;
}

static int16_t pack_snorm16(float v)
{
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    return (int16_t)lrintf(v * 32767.0f);
}

static uint8_t pack_unorm8(float v)
{
    if (v > 1.0f) v = 1.0f;
    if (v < 0.0f) v = 0.0f;
    return (uint8_t)lrintf(v * 255.0f);
}

static uint32_t pack_rgba8(float r, float g, float b, float a)
{
    return (uint32_t)pack_unorm8(r) |
           ((uint32_t)pack_unorm8(g) << 8) |
           ((uint32_t)pack_unorm8(b) << 16) |
           ((uint32_t)pack_unorm8(a) << 24);
}

#define VIEWER_PTEX_COLOR_NONE 0xFFFFFFFFu

static int env_truthy_default(const char* name, int def)
{
    const char* e = getenv(name);
    if (!e || !e[0]) return def;
    if (e[0] == '0' || e[0] == 'f' || e[0] == 'F' ||
        e[0] == 'n' || e[0] == 'N')
        return 0;
    return 1;
}

static uint64_t env_u64_or_unlimited(const char* name, uint64_t def)
{
    const char* e = getenv(name);
    if (!e || !e[0]) return def;
    char* end = NULL;
    long long v = strtoll(e, &end, 10);
    if (end == e) return def;
    if (v < 0) return (uint64_t)UINT32_MAX;
    return (uint64_t)v;
}

static void viewer_reset_ptex_offsets(Scene* scene)
{
    if (!scene) return;
    for (int i = 0; i < scene->nmeshes; i++) {
        scene->meshes[i].ptex_color_offset = VIEWER_PTEX_COLOR_NONE;
        free(scene->meshes[i].ptex_tri_colors);
        scene->meshes[i].ptex_tri_colors = NULL;
        scene->meshes[i].ptex_tri_color_count = 0;
    }
}

static int viewer_mesh_ptex_path(const Viewer* v, const SceneMesh* mesh,
                                 const char** out_path)
{
    if (out_path) *out_path = NULL;
    if (!v || !v->materials || !mesh || !out_path) return 0;
    if (mesh->material_index < 0 ||
        mesh->material_index >= v->materials->nmaterials)
        return 0;
    const SceneMaterial* mat = &v->materials->materials[mesh->material_index];
    if (!mat->ptex_color_path[0]) return 0;
    *out_path = mat->ptex_color_path;
    return 1;
}

static int viewer_upload_scene_ptex_triangle_colors(Viewer* v)
{
    if (!v || !v->gpu || !v->scene || !v->materials) return 0;
    viewer_reset_ptex_offsets(v->scene);
    v->ptex_color_buffer_uploaded = 0;

#ifndef NUSD_HAVE_PTEX
    gpu_upload_ptex_triangle_colors(v->gpu, NULL, 0);
    return 0;
#else
    if (!env_truthy_default("NUSD_OPENGL_PTEX_TRI_COLORS", 1)) {
        gpu_upload_ptex_triangle_colors(v->gpu, NULL, 0);
        fprintf(stderr, "viewer: OpenGL Ptex triangle colors disabled by env\n");
        return 0;
    }

    uint64_t max_colors = env_u64_or_unlimited(
        "NUSD_OPENGL_PTEX_MAX_COLORS", 80000000ull);
    if (max_colors == 0) {
        gpu_upload_ptex_triangle_colors(v->gpu, NULL, 0);
        fprintf(stderr, "viewer: OpenGL Ptex triangle colors disabled (max_colors=0)\n");
        return 0;
    }
    if (max_colors > UINT32_MAX) max_colors = UINT32_MAX;

    uint64_t candidate_colors = 0;
    int candidate_meshes = 0;
    int cap_skipped_meshes = 0;
    for (int m = 0; m < v->scene->nmeshes; m++) {
        SceneMesh* mesh = &v->scene->meshes[m];
        const char* path = NULL;
        if (mesh->is_proto_only || mesh->nindices <= 0 ||
            (mesh->nindices % 3) != 0 || !mesh->indices)
            continue;
        if (!viewer_mesh_ptex_path(v, mesh, &path))
            continue;
        uint64_t need = (uint64_t)mesh->nindices;
        if (candidate_colors + need > max_colors) {
            cap_skipped_meshes++;
            continue;
        }
        candidate_colors += need;
        candidate_meshes++;
        (void)path;
    }

    if (candidate_colors == 0 || candidate_meshes == 0) {
        gpu_upload_ptex_triangle_colors(v->gpu, NULL, 0);
        if (cap_skipped_meshes > 0) {
            fprintf(stderr,
                    "viewer: OpenGL Ptex triangle colors skipped %d mesh(es) "
                    "because max_colors=%llu\n",
                    cap_skipped_meshes, (unsigned long long)max_colors);
        }
        return 0;
    }

    uint32_t* flat = (uint32_t*)malloc(
        (size_t)candidate_colors * sizeof(uint32_t));
    if (!flat) {
        gpu_upload_ptex_triangle_colors(v->gpu, NULL, 0);
        fprintf(stderr,
                "viewer: failed to allocate OpenGL Ptex triangle colors "
                "(%llu colors)\n",
                (unsigned long long)candidate_colors);
        return 0;
    }

    uint32_t cursor = 0;
    int sampled_meshes = 0;
    int failed_meshes = 0;
    int warn_count = 0;
    for (int m = 0; m < v->scene->nmeshes; m++) {
        SceneMesh* mesh = &v->scene->meshes[m];
        const char* path = NULL;
        if (mesh->is_proto_only || mesh->nindices <= 0 ||
            (mesh->nindices % 3) != 0 || !mesh->indices)
            continue;
        if (!viewer_mesh_ptex_path(v, mesh, &path))
            continue;
        uint64_t need = (uint64_t)mesh->nindices;
        if ((uint64_t)cursor + need > candidate_colors)
            continue;

        int written = nusd_ptex_sample_face_tri_colors(
            path, NULL, 0, mesh->nindices, flat + cursor, mesh->nindices);
        if (written == mesh->nindices) {
            mesh->ptex_color_offset = cursor;
            cursor += (uint32_t)mesh->nindices;
            sampled_meshes++;
        } else {
            failed_meshes++;
            if (warn_count < 16) {
                fprintf(stderr,
                        "viewer: OpenGL Ptex sampling skipped %s "
                        "(wrote %d/%d colors from %s)\n",
                        mesh->path ? mesh->path : "<mesh>",
                        written, mesh->nindices, path);
                warn_count++;
            }
        }
    }

    int ok = gpu_upload_ptex_triangle_colors(v->gpu, flat, cursor);
    free(flat);
    if (!ok || cursor == 0) {
        viewer_reset_ptex_offsets(v->scene);
        gpu_upload_ptex_triangle_colors(v->gpu, NULL, 0);
        fprintf(stderr,
                "viewer: OpenGL Ptex triangle-color upload disabled "
                "(sampled=%d failed=%d)\n",
                sampled_meshes, failed_meshes);
        return 0;
    }

    v->ptex_color_buffer_uploaded = 1;
    fprintf(stderr,
            "viewer: OpenGL Ptex triangle colors active: %u colors "
            "(%u triangles), %d/%d candidate meshes, %d failed, "
            "%d cap-skipped, max_colors=%llu\n",
            cursor, cursor / 3u, sampled_meshes, candidate_meshes,
            failed_meshes, cap_skipped_meshes,
            (unsigned long long)max_colors);
    return 1;
#endif
}

/* ---- Content-hash dedup (NUSD_HASH_DEDUP=1) ----
 *
 * Mirror the Vulkan-side fast path: meshes with byte-identical
 * (positions, normals, texcoords, indices, baked-color) share the prototype's
 * vertex_offset/index_offset in the merged GL chunk. Catches the
 * non-USD-instance duplicates (different USD paths, identical mesh
 * data — typical for DSX racks of repeated GB300 parts authored
 * without instance arcs). Vulkan census measured 70.7% dup rate on
 * full DSX; OpenGL benefits from the same pool of duplicates.
 *
 * Dedup key includes (has_display_color, display_color) — the OpenGL vertex
 * stream bakes color in, so meshes that differ in baked color must take the
 * slow path. It does NOT include material_index: material is applied per-draw
 * (gpu_cmd_bind_material by each mesh's own material_index), not baked into the
 * vertex stream, so byte-identical geometry with distinct materials can share a
 * chunk and still render correctly (each instance binds its own material). This
 * mirrors the Metal renderer's cross-material dedup. */
typedef struct {
    uint64_t hash;
    int      mesh_idx;
} GeoDedupSlot;

typedef struct {
    GeoDedupSlot* slots;
    int           cap;
    int           count;
} GeoDedupTable;

static int geo_dedup_enabled_gl(void) {
    /* Default ON. Collapses byte-identical geometry (the warehouse/DSX racks of
     * repeated parts authored without instance arcs) onto one shared GL chunk.
     * NUSD_HASH_DEDUP=0 forces the per-mesh path for A/B measurement. */
    const char* e = getenv("NUSD_HASH_DEDUP");
    return (e && e[0] == '0') ? 0 : 1;
}

static inline uint64_t fnv1a_update64(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static uint64_t geo_dedup_hash(const SceneMesh* mesh)
{
    uint64_t h = 1469598103934665603ULL;
    int nv = mesh->nvertices, ni = mesh->nindices;
    h = fnv1a_update64(h, &nv, sizeof(nv));
    h = fnv1a_update64(h, &ni, sizeof(ni));
    /* material_index intentionally excluded — applied per-draw, not vertex-baked.
     * display_color IS baked into the vertex stream, so it stays in the key. */
    int has_dc = mesh->has_display_color ? 1 : 0;
    h = fnv1a_update64(h, &has_dc, sizeof(has_dc));
    if (mesh->has_display_color)
        h = fnv1a_update64(h, mesh->display_color, 3 * sizeof(float));
    if (mesh->positions) h = fnv1a_update64(h, mesh->positions, (size_t)nv * 3 * sizeof(float));
    if (mesh->normals)   h = fnv1a_update64(h, mesh->normals,   (size_t)nv * 3 * sizeof(float));
    if (mesh->colors)    h = fnv1a_update64(h, mesh->colors,    (size_t)nv * 3 * sizeof(float));
    if (mesh->texcoords) h = fnv1a_update64(h, mesh->texcoords, (size_t)nv * 2 * sizeof(float));
    if (mesh->indices)   h = fnv1a_update64(h, mesh->indices,   (size_t)ni * sizeof(uint32_t));
    if (h == 0) h = 0xdeadbeefcafebabeULL;
    return h;
}

static int geo_mesh_equal_gl(const SceneMesh* a, const SceneMesh* b)
{
    if (!a || !b) return 0;
    if (a->nvertices != b->nvertices || a->nindices != b->nindices)
        return 0;
    /* material_index intentionally NOT compared — applied per-draw, so shared
     * geometry renders with each instance's own material. Color IS vertex-baked,
     * so it is still compared below. */
    if ((a->has_display_color ? 1 : 0) != (b->has_display_color ? 1 : 0))
        return 0;
    if (a->has_display_color &&
        memcmp(a->display_color, b->display_color, 3u * sizeof(float)) != 0)
        return 0;

    if (!a->positions || !b->positions || !a->indices || !b->indices)
        return 0;
    if (memcmp(a->positions, b->positions,
               (size_t)a->nvertices * 3u * sizeof(float)) != 0)
        return 0;
    if ((a->normals != NULL) != (b->normals != NULL))
        return 0;
    if (a->normals &&
        memcmp(a->normals, b->normals,
               (size_t)a->nvertices * 3u * sizeof(float)) != 0)
        return 0;
    if ((a->colors != NULL) != (b->colors != NULL))
        return 0;
    if (a->colors &&
        memcmp(a->colors, b->colors,
               (size_t)a->nvertices * 3u * sizeof(float)) != 0)
        return 0;
    if ((a->texcoords != NULL) != (b->texcoords != NULL))
        return 0;
    if (a->texcoords &&
        memcmp(a->texcoords, b->texcoords,
               (size_t)a->nvertices * 2u * sizeof(float)) != 0)
        return 0;
    if (memcmp(a->indices, b->indices,
               (size_t)a->nindices * sizeof(uint32_t)) != 0)
        return 0;
    return 1;
}

static void geo_dedup_grow(GeoDedupTable* t) {
    if (!t) return;
    /* Normalize the {slots, cap} pair so the rehash loop can't dereference
     * t->slots while it is NULL with nonzero t->cap. */
    if (!t->slots) t->cap = 0;
    int new_cap = t->cap ? t->cap * 2 : 1024;
    GeoDedupSlot* ns = (GeoDedupSlot*)calloc(new_cap, sizeof(GeoDedupSlot));
    if (!ns) return;
    for (int i = 0; i < t->cap; i++) {
        if (!t->slots[i].hash) continue;
        uint32_t mask = new_cap - 1;
        uint32_t s = (uint32_t)t->slots[i].hash & mask;
        while (ns[s].hash) s = (s + 1) & mask;
        ns[s] = t->slots[i];
    }
    free(t->slots);
    t->slots = ns;
    t->cap = new_cap;
}

static int geo_dedup_lookup_or_insert(GeoDedupTable* t,
                                      const Scene* scene,
                                      uint64_t h,
                                      int mesh_idx,
                                      uint64_t* collisions)
{
    if (!t->slots || t->count * 2 >= t->cap) geo_dedup_grow(t);
    if (!t->slots) return -1;
    uint32_t mask = t->cap - 1;
    uint32_t s = (uint32_t)h & mask;
    while (t->slots[s].hash) {
        if (t->slots[s].hash == h) {
            int proto = t->slots[s].mesh_idx;
            if (scene && proto >= 0 && proto < scene->nmeshes &&
                mesh_idx >= 0 && mesh_idx < scene->nmeshes &&
                geo_mesh_equal_gl(&scene->meshes[proto], &scene->meshes[mesh_idx]))
                return proto;
            if (collisions) (*collisions)++;
        }
        s = (s + 1) & mask;
    }
    t->slots[s].hash = h;
    t->slots[s].mesh_idx = mesh_idx;
    t->count++;
    return -1;
}

static void scene_apply_geo_dedup_gl(Scene* scene)
{
    if (!scene || !geo_dedup_enabled_gl()) return;

    GeoDedupTable dedup = { NULL, 0, 0 };
    uint64_t total_calls = 0, dup_calls = 0, hash_collisions = 0;

    for (int m = 0; m < scene->nmeshes; m++) {
        SceneMesh* mesh = &scene->meshes[m];
        if (mesh->nvertices <= 0 || mesh->nindices <= 0) continue;
        if (mesh->prototype_idx >= 0 &&
            mesh->prototype_idx < scene->nmeshes &&
            mesh->prototype_idx != m)
            continue;

        uint64_t h = geo_dedup_hash(mesh);
        int proto = geo_dedup_lookup_or_insert(&dedup, scene, h, m,
                                               &hash_collisions);
        total_calls++;
        if (proto >= 0 && proto < scene->nmeshes) {
            mesh->prototype_idx = proto;
            dup_calls++;
        }
    }

    if (total_calls > 0) {
        fprintf(stderr,
                "viewer: hash dedup marked %llu/%llu duplicate meshes"
                " (collisions=%llu)\n",
                (unsigned long long)dup_calls,
                (unsigned long long)total_calls,
                (unsigned long long)hash_collisions);
    }
    free(dedup.slots);
}

static int build_chunked_raster_buffers(Viewer* viewer,
                                        uint32_t* out_total_vertices,
                                        uint32_t* out_total_indices)
{
    Scene* scene = viewer ? viewer->scene : NULL;
    if (!viewer || !scene || !out_total_vertices || !out_total_indices) return 0;
    *out_total_vertices = 0;
    *out_total_indices = 0;

    unsigned char* needed = build_geometry_needed_mask(scene);
    if (!needed) {
        /* Empty or metadata-only (Tier 3 lazy) scene — no GL buffers to
         * build, but viewer_create should still succeed. Renders will
         * fail-fast with "no geometry" until step 3 lands. */
        viewer->buffer_count = 0;
        viewer->vertex_buffer = NULL;
        viewer->index_buffer = NULL;
        viewer->vertex_buffers = NULL;
        viewer->index_buffers = NULL;
        return 1;
    }

    RasterChunk* chunks = NULL;
    int nchunks = 0;
    int cap = 0;
    const char* mixed_env = getenv("NUSD_GL_MIXED_INDICES");
    int use_mixed_indices = !(mixed_env && mixed_env[0] == '0');

    /* Optional content-hash dedup: shares chunk slots across meshes with
     * byte-identical (positions, normals, texcoords, indices, baked-color, material).
     * Enabled by NUSD_HASH_DEDUP=1. Acts in addition to USD-instance
     * dedup (prototype_idx) — for DSX where ~70% of nu_add_mesh calls
     * are content-identical, this halves GL VB+IB beyond the instance-
     * skip fix. */
    int dedup_on = geo_dedup_enabled_gl();
    GeoDedupTable dedup = { NULL, 0, 0 };
    uint64_t dedup_total_calls = 0, dedup_dup_calls = 0, dedup_hash_collisions = 0;

    for (int m = 0; m < scene->nmeshes; m++) {
        SceneMesh* mesh = &scene->meshes[m];
        if (!needed[m]) continue;
        /* Skip USD-instance copies during chunk sizing. Phase 2 below
         * (lines 300-311) reassigns their buffer_index / vertex_offset /
         * index_offset to share the prototype's slot, so reserving their
         * own slot here just leaves a hole in the merged buffer. On DSX
         * (56% instance share rate) this previously inflated the GL VB+IB
         * by ~2.3× — instances each claimed chunk capacity that was
         * never read at draw time. */
        if (mesh->prototype_idx >= 0 &&
            mesh->prototype_idx < scene->nmeshes &&
            mesh->prototype_idx != m) continue;
        /* Content-hash fast path: if a previously-sized mesh is byte-
         * identical, mark this one as its instance so Phase 2 (line 300)
         * propagates the prototype's offsets and we skip chunk sizing. */
        if (dedup_on) {
            uint64_t h = geo_dedup_hash(mesh);
            int proto = geo_dedup_lookup_or_insert(&dedup, scene, h, m,
                                                   &dedup_hash_collisions);
            dedup_total_calls++;
            if (proto >= 0 && proto < scene->nmeshes) {
                mesh->prototype_idx = proto;
                dedup_dup_calls++;
                continue;
            }
        }
        uint32_t nv = (uint32_t)mesh->nvertices;
        uint32_t ni = (uint32_t)mesh->nindices;
        if (nv == 0 || ni == 0) continue;
        int index_bits = use_mixed_indices ? raster_mesh_index_bits(mesh) : 32;
        uint64_t index_align = (index_bits == 16) ? 2u : 4u;
        uint64_t index_bytes = (uint64_t)ni * (uint64_t)(index_bits / 8);
        uint64_t aligned_index_bytes = nchunks > 0
            ? align_up_u64(chunks[nchunks - 1].index_bytes, index_align)
            : 0;

        if (nchunks == 0 ||
            (chunks[nchunks - 1].vertices > 0 &&
             (chunks[nchunks - 1].vertices + nv > RASTER_CHUNK_MAX_VERTICES ||
              aligned_index_bytes + index_bytes > RASTER_CHUNK_MAX_INDEX_BYTES))) {
            if (nchunks >= cap) {
                int new_cap = cap ? cap * 2 : 8;
                RasterChunk* new_chunks = (RasterChunk*)realloc(
                    chunks, (size_t)new_cap * sizeof(RasterChunk));
                if (!new_chunks) {
                    free(chunks);
                    free(needed);
                    return 0;
                }
                chunks = new_chunks;
                cap = new_cap;
            }
            chunks[nchunks].vertices = 0;
            chunks[nchunks].indices = 0;
            chunks[nchunks].index_bytes = 0;
            nchunks++;
            aligned_index_bytes = 0;
        }

        RasterChunk* c = &chunks[nchunks - 1];
        mesh->buffer_index = (uint32_t)(nchunks - 1);
        mesh->vertex_offset = c->vertices;
        mesh->index_offset = c->indices;
        mesh->index_byte_offset = aligned_index_bytes;
        mesh->index_type_bits = (uint8_t)index_bits;
        c->vertices += nv;
        c->indices += ni;
        c->index_bytes = aligned_index_bytes + index_bytes;
        *out_total_vertices += nv;
        *out_total_indices += ni;
    }

    for (int m = 0; m < scene->nmeshes; m++) {
        SceneMesh* mesh = &scene->meshes[m];
        if (mesh->prototype_idx != m && mesh->prototype_idx >= 0 &&
            mesh->prototype_idx < scene->nmeshes) {
            SceneMesh* proto = &scene->meshes[mesh->prototype_idx];
            mesh->buffer_index = proto->buffer_index;
            mesh->vertex_offset = proto->vertex_offset;
            mesh->index_offset = proto->index_offset;
            mesh->index_byte_offset = proto->index_byte_offset;
            mesh->index_type_bits = proto->index_type_bits;
        }
    }

    if (nchunks <= 0) {
        /* All meshes filtered out (e.g. metadata-only lazy scene where
         * nindices==0 everywhere). Viewer is otherwise valid; render
         * paths just have no geometry to draw. */
        viewer->buffer_count = 0;
        viewer->vertex_buffer = NULL;
        viewer->index_buffer = NULL;
        viewer->vertex_buffers = NULL;
        viewer->index_buffers = NULL;
        free(chunks);
        free(needed);
        return 1;
    }

    viewer->vertex_buffers = (GpuBuffer*)calloc((size_t)nchunks, sizeof(GpuBuffer));
    viewer->index_buffers = (GpuBuffer*)calloc((size_t)nchunks, sizeof(GpuBuffer));
    if (!viewer->vertex_buffers || !viewer->index_buffers) {
        free(chunks);
        free(needed);
        return 0;
    }
    viewer->buffer_count = nchunks;

    for (int cidx = 0; cidx < nchunks; cidx++) {
        RasterChunk* c = &chunks[cidx];
        RasterVertexPacked* verts = (RasterVertexPacked*)malloc(
            (size_t)c->vertices * sizeof(RasterVertexPacked));
        unsigned char* indices = (unsigned char*)malloc((size_t)c->index_bytes);
        if (!verts || !indices) {
            free(verts);
            free(indices);
            free(chunks);
            free(needed);
            return 0;
        }

        for (int m = 0; m < scene->nmeshes; m++) {
            SceneMesh* mesh = &scene->meshes[m];
            if (!needed[m] || mesh->buffer_index != (uint32_t)cidx) continue;

            RasterVertexPacked* dst = verts + mesh->vertex_offset;
            const float default_gray[3] = { 0.7f, 0.7f, 0.7f };
            const float* fallback_color = mesh->has_display_color
                ? mesh->display_color : default_gray;

            for (int i = 0; i < mesh->nvertices; i++) {
                const float* src_pos = &mesh->positions[i * 3];
                dst->position[0] = src_pos[0];
                dst->position[1] = src_pos[1];
                dst->position[2] = src_pos[2];

                if (mesh->normals) {
                    const float* src_nrm = &mesh->normals[i * 3];
                    dst->normal[0] = pack_snorm16(src_nrm[0]);
                    dst->normal[1] = pack_snorm16(src_nrm[1]);
                    dst->normal[2] = pack_snorm16(src_nrm[2]);
                } else {
                    dst->normal[0] = 0;
                    dst->normal[1] = 32767;
                    dst->normal[2] = 0;
                }
                dst->normal[3] = 0;

                if (mesh->colors) {
                    const float* src_col = &mesh->colors[i * 3];
                    dst->color[0] = pack_unorm8(src_col[0]);
                    dst->color[1] = pack_unorm8(src_col[1]);
                    dst->color[2] = pack_unorm8(src_col[2]);
                } else {
                    dst->color[0] = pack_unorm8(fallback_color[0]);
                    dst->color[1] = pack_unorm8(fallback_color[1]);
                    dst->color[2] = pack_unorm8(fallback_color[2]);
                }
                dst->color[3] = 255;
                dst++;
            }

            unsigned char* idst_base = indices + mesh->index_byte_offset;
            if (mesh->index_type_bits == 16) {
                uint16_t* idst = (uint16_t*)idst_base;
                for (int i = 0; i < mesh->nindices; i++)
                    *idst++ = (uint16_t)mesh->indices[i];
            } else {
                uint32_t* idst = (uint32_t*)idst_base;
                for (int i = 0; i < mesh->nindices; i++)
                    *idst++ = mesh->indices[i] + mesh->vertex_offset;
            }
        }

        GpuBufferDesc vb_desc;
        vb_desc.usage = GPU_BUFFER_VERTEX;
        vb_desc.size = (uint64_t)c->vertices * sizeof(RasterVertexPacked);
        vb_desc.data = verts;
        fprintf(stderr, "viewer: uploading raster chunk %d/%d VB %.1f MB, IB %.1f MB...\n",
                cidx + 1, nchunks,
                (double)vb_desc.size / (1024.0 * 1024.0),
                (double)c->index_bytes / (1024.0 * 1024.0));
        viewer->vertex_buffers[cidx] = gpu_create_buffer(viewer->gpu, &vb_desc);
        free(verts);

        GpuBufferDesc ib_desc;
        ib_desc.usage = GPU_BUFFER_INDEX;
        ib_desc.size = c->index_bytes;
        ib_desc.data = indices;
        viewer->index_buffers[cidx] = gpu_create_buffer(viewer->gpu, &ib_desc);
        free(indices);

        if (!viewer->vertex_buffers[cidx] || !viewer->index_buffers[cidx]) {
            free(chunks);
            free(needed);
            return 0;
        }
    }

    viewer->vertex_buffer = viewer->vertex_buffers[0];
    viewer->index_buffer = viewer->index_buffers[0];
    fprintf(stderr, "viewer: raster buffers split into %d chunks%s\n",
            nchunks, use_mixed_indices ? " (mixed 16/32-bit indices)" : "");

    if (dedup_on && dedup_total_calls > 0) {
        fprintf(stderr, "viewer: content-hash dedup — %llu sizing calls, "
                "%llu hits (%.1f%% dup rate, %llu same-hash nonmatches, "
                "%d unique)\n",
                (unsigned long long)dedup_total_calls,
                (unsigned long long)dedup_dup_calls,
                100.0 * (double)dedup_dup_calls / (double)dedup_total_calls,
                (unsigned long long)dedup_hash_collisions,
                dedup.count);
    }
    free(dedup.slots);

    free(chunks);
    free(needed);
    return 1;
}

/* ---- Meshlet draw path (NUSD_MESHLET_DRAW=1, raster mode) --------------- */

/* 1 when NUSD_MESHLET_DRAW is set to a truthy value. */
static int meshlet_draw_enabled(void)
{
    const char* e = getenv("NUSD_MESHLET_DRAW");
    return e && e[0] && e[0] != '0';
}

/* Transform an object-space point by a USD row-major world_xform
 * (v' = v . M, translation at M[12..14]). */
static void ml_xform_point(const double* M, const float* p, float* o)
{
    for (int j = 0; j < 3; j++)
        o[j] = (float)((double)p[0]*M[j]   + (double)p[1]*M[4+j]
                     + (double)p[2]*M[8+j] + M[12+j]);
}

static void curve_transform_aabb_world(const SceneCurve* c,
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
                ml_xform_point(c->world_xform, p, wp);
                for (int k = 0; k < 3; k++) {
                    if (wp[k] < world_min[k]) world_min[k] = wp[k];
                    if (wp[k] > world_max[k]) world_max[k] = wp[k];
                }
            }
        }
    }
}

static void curve_chunk_bounds_world(const SceneCurve* c,
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
    curve_transform_aabb_world(c, obj_min, obj_max, world_min, world_max);
}

/* Largest basis-axis length of a USD world_xform — a conservative uniform
 * scale for transforming a bounding-sphere radius to world space. */
static float ml_xform_max_scale(const double* M)
{
    float best = 0.0f;
    for (int i = 0; i < 3; i++) {
        double a = M[i*4+0], b = M[i*4+1], c = M[i*4+2];
        float len = (float)sqrt(a*a + b*b + c*c);
        if (len > best) best = len;
    }
    return best;
}

/* Build the meshlet index buffer: every meshlet's triangles expanded to
 * mesh-local 32-bit indices, ordered by meshlet record. meshlet_ibo_off[r]
 * is record r's first index (prefix sums; nmeshlets+1 entries). Sets
 * v->meshlet_draw on success. Raster mode only — the meshlet vertex tables
 * index mesh-local vertices, which the raster chunk VBO stores 1:1. */
static int build_meshlet_index_buffer(Viewer* v)
{
    Scene* s = v ? v->scene : NULL;
    if (!s || s->nmeshlets <= 0 || !s->meshlets ||
        !s->meshlet_vertices || !s->meshlet_triangles)
        return 0;

    uint32_t* off = (uint32_t*)malloc((size_t)(s->nmeshlets + 1) * sizeof(uint32_t));
    if (!off) return 0;
    uint64_t total = 0;
    for (int r = 0; r < s->nmeshlets; r++) {
        off[r] = (uint32_t)total;
        total += (uint64_t)s->meshlets[r].triangle_count * 3u;
    }
    off[s->nmeshlets] = (uint32_t)total;
    if (total == 0) { free(off); return 0; }

    uint32_t* idx = (uint32_t*)malloc(total * sizeof(uint32_t));
    if (!idx) { free(off); return 0; }
    uint64_t w = 0;
    for (int r = 0; r < s->nmeshlets; r++) {
        const SceneMeshlet*  ml = &s->meshlets[r];
        const uint32_t*      vt = s->meshlet_vertices  + ml->vertex_offset;
        const unsigned char* mi = s->meshlet_triangles + ml->triangle_offset;
        uint32_t n = ml->triangle_count * 3u;
        for (uint32_t i = 0; i < n; i++)
            idx[w++] = vt[mi[i]];
    }

    GpuBufferDesc d;
    d.usage = GPU_BUFFER_INDEX;
    d.size  = total * sizeof(uint32_t);
    d.data  = idx;
    v->meshlet_index_buffer = gpu_create_buffer(v->gpu, &d);
    free(idx);
    if (!v->meshlet_index_buffer) { free(off); return 0; }

    /* Copy the per-meshlet cull data out of the arena — the draw loop reads
     * it every frame, but scene_release_mesh_payloads frees the arena that
     * backs scene->meshlets after GPU upload. */
    ViewerMeshlet* recs = (ViewerMeshlet*)malloc(
        (size_t)s->nmeshlets * sizeof(ViewerMeshlet));
    if (!recs) {
        gpu_destroy_buffer(v->gpu, v->meshlet_index_buffer);
        v->meshlet_index_buffer = NULL;
        free(off);
        return 0;
    }
    for (int r = 0; r < s->nmeshlets; r++) {
        recs[r].center[0]      = s->meshlets[r].center[0];
        recs[r].center[1]      = s->meshlets[r].center[1];
        recs[r].center[2]      = s->meshlets[r].center[2];
        recs[r].radius         = s->meshlets[r].radius;
        recs[r].triangle_count = s->meshlets[r].triangle_count;
    }

    v->meshlet_ibo_off = off;
    v->meshlet_recs    = recs;
    v->meshlet_draw = 1;
    fprintf(stderr, "viewer: meshlet draw path — %llu indices (%.1f MB IBO)\n",
            (unsigned long long)total,
            (double)(total * sizeof(uint32_t)) / (1024.0 * 1024.0));
    return 1;
}

/*
 * Build a single interleaved vertex buffer with UNIQUE prototype geometry only.
 * Layout: [pos.x, pos.y, pos.z, norm.x, norm.y, norm.z, col.r, col.g, col.b]
 * = 9 floats = 36 bytes per vertex.  Positions/normals in LOCAL space.
 */
static float* build_merged_vertex_buffer(Scene* scene, uint32_t* out_total_vertices)
{
    unsigned char* needed = build_geometry_needed_mask(scene);
    if (!needed) {
        *out_total_vertices = 0;
        return NULL;
    }

    uint32_t total = 0;
    for (int m = 0; m < scene->nmeshes; m++) {
        if (needed[m])
            total += (uint32_t)scene->meshes[m].nvertices;
    }

    if (total == 0) {
        *out_total_vertices = 0;
        free(needed);
        return NULL;
    }

    float* buf = (float*)malloc((size_t)total * 9 * sizeof(float));
    if (!buf) {
        *out_total_vertices = 0;
        free(needed);
        return NULL;
    }

    uint32_t vertex_cursor = 0;
    float* dst = buf;

    for (int m = 0; m < scene->nmeshes; m++) {
        SceneMesh* mesh = &scene->meshes[m];
        if (!needed[m])
            continue;

        mesh->vertex_offset = vertex_cursor;

        const float default_gray[3] = { 0.7f, 0.7f, 0.7f };
        const float* fallback_color = mesh->has_display_color
            ? mesh->display_color : default_gray;

        for (int i = 0; i < mesh->nvertices; i++) {
            const float* src_pos = &mesh->positions[i * 3];
            *dst++ = src_pos[0];
            *dst++ = src_pos[1];
            *dst++ = src_pos[2];

            if (mesh->normals) {
                const float* src_nrm = &mesh->normals[i * 3];
                *dst++ = src_nrm[0];
                *dst++ = src_nrm[1];
                *dst++ = src_nrm[2];
            } else {
                *dst++ = 0.0f;
                *dst++ = 1.0f;
                *dst++ = 0.0f;
            }

            if (mesh->colors) {
                const float* src_col = &mesh->colors[i * 3];
                *dst++ = src_col[0];
                *dst++ = src_col[1];
                *dst++ = src_col[2];
            } else {
                *dst++ = fallback_color[0];
                *dst++ = fallback_color[1];
                *dst++ = fallback_color[2];
            }
        }

        vertex_cursor += (uint32_t)mesh->nvertices;
    }

    /* Set instance vertex_offset to prototype's offset */
    for (int m = 0; m < scene->nmeshes; m++) {
        SceneMesh* mesh = &scene->meshes[m];
        if (mesh->prototype_idx != m && mesh->prototype_idx >= 0 &&
            mesh->prototype_idx < scene->nmeshes)
            mesh->vertex_offset = scene->meshes[mesh->prototype_idx].vertex_offset;
    }

    *out_total_vertices = total;
    free(needed);
    return buf;
}

/*
 * Build a merged index buffer with PRE-OFFSET indices.
 *
 * Unlike the Vulkan viewer (which uses glDrawElementsBaseVertex at draw time),
 * GLES 3.1 lacks baseVertex support. We bake vertex_offset into every index
 * so glDrawElements works directly.
 */
static uint32_t* build_merged_index_buffer(Scene* scene, uint32_t* out_total_indices)
{
    unsigned char* needed = build_geometry_needed_mask(scene);
    if (!needed) {
        *out_total_indices = 0;
        return NULL;
    }

    uint32_t total = 0;
    for (int m = 0; m < scene->nmeshes; m++) {
        if (needed[m])
            total += (uint32_t)scene->meshes[m].nindices;
    }

    if (total == 0) {
        *out_total_indices = 0;
        free(needed);
        return NULL;
    }

    uint32_t* buf = (uint32_t*)malloc((size_t)total * sizeof(uint32_t));
    if (!buf) {
        *out_total_indices = 0;
        free(needed);
        return NULL;
    }

    uint32_t index_cursor = 0;
    uint32_t* dst = buf;

    for (int m = 0; m < scene->nmeshes; m++) {
        SceneMesh* mesh = &scene->meshes[m];
        if (!needed[m])
            continue;

        mesh->index_offset = index_cursor;

        /* Vertex cache optimization: reorder this prototype's indices for
         * better post-T&L cache hit rate. Lossless — same triangles,
         * different order.
         *
         * meshopt_optimizeVertexCache requires a triangle-list index
         * buffer (nindices a positive multiple of 3) with every index in
         * [0, nvertices) — it indexes an nvertices-sized score table and
         * an out-of-range index segfaults. Malformed prototypes occur at
         * DSX scale; validate first and skip the (optional) optimization
         * for any mesh that doesn't satisfy the contract. */
        {
            int meshopt_safe = (mesh->nvertices > 0 &&
                                mesh->nindices >= 3 &&
                                mesh->nindices % 3 == 0);
            if (meshopt_safe) {
                uint32_t nv = (uint32_t)mesh->nvertices;
                for (int i = 0; i < mesh->nindices; i++) {
                    if (mesh->indices[i] >= nv) { meshopt_safe = 0; break; }
                }
            }
            if (meshopt_safe) {
                meshopt_optimizeVertexCache(mesh->indices, mesh->indices,
                                            (size_t)mesh->nindices,
                                            (size_t)mesh->nvertices);
            }
        }

        /* Pre-offset: add vertex_offset to each index */
        for (int i = 0; i < mesh->nindices; i++) {
            *dst++ = mesh->indices[i] + mesh->vertex_offset;
        }

        index_cursor += (uint32_t)mesh->nindices;
    }

    /* Set instance index_offset to prototype's offset */
    for (int m = 0; m < scene->nmeshes; m++) {
        SceneMesh* mesh = &scene->meshes[m];
        if (mesh->prototype_idx != m && mesh->prototype_idx >= 0 &&
            mesh->prototype_idx < scene->nmeshes)
            mesh->index_offset = scene->meshes[mesh->prototype_idx].index_offset;
    }

    *out_total_indices = total;
    free(needed);
    return buf;
}

/*
 * Build a material-aware vertex buffer with UVs and tangents (no materialId —
 * GLES uses per-draw UBO instead of dynamic SSBO indexing).
 * Layout: [pos3, nrm3, col3, uv2, tan3, tanSign1] = 15 floats = 60 bytes per vertex.
 */
#define MAT_VERTEX_STRIDE 60
#define MAT_VERTEX_FLOATS 15

static float* build_material_vertex_buffer(Scene* scene, uint32_t* out_total)
{
    unsigned char* needed = build_geometry_needed_mask(scene);
    if (!needed) {
        *out_total = 0;
        return NULL;
    }

    uint32_t total = 0;
    for (int m = 0; m < scene->nmeshes; m++) {
        if (needed[m])
            total += (uint32_t)scene->meshes[m].nvertices;
    }

    if (total == 0) {
        *out_total = 0;
        free(needed);
        return NULL;
    }

    float* buf = (float*)malloc((size_t)total * MAT_VERTEX_FLOATS * sizeof(float));
    if (!buf) {
        *out_total = 0;
        free(needed);
        return NULL;
    }

    uint32_t vertex_cursor = 0;
    float* dst = buf;

    for (int m = 0; m < scene->nmeshes; m++) {
        SceneMesh* mesh = &scene->meshes[m];
        if (!needed[m]) {
            if (mesh->prototype_idx >= 0 && mesh->prototype_idx < scene->nmeshes)
                mesh->vertex_offset = scene->meshes[mesh->prototype_idx].vertex_offset;
            continue;
        }

        mesh->vertex_offset = vertex_cursor;

        if (!mesh->positions)
            continue;

        const float default_gray[3] = { 0.7f, 0.7f, 0.7f };
        const float* fallback_color = mesh->has_display_color
            ? mesh->display_color : default_gray;

        /* Compute per-vertex tangents and bitangent signs from triangle UVs */
        float* tangents = (float*)calloc((size_t)mesh->nvertices * 3, sizeof(float));
        float* bitangents = (float*)calloc((size_t)mesh->nvertices * 3, sizeof(float));
        float* tan_signs = (float*)calloc((size_t)mesh->nvertices, sizeof(float));

        if (mesh->texcoords && mesh->indices) {
            for (int t = 0; t + 2 < mesh->nindices; t += 3) {
                uint32_t i0 = mesh->indices[t];
                uint32_t i1 = mesh->indices[t + 1];
                uint32_t i2 = mesh->indices[t + 2];

                const float* p0 = &mesh->positions[i0 * 3];
                const float* p1 = &mesh->positions[i1 * 3];
                const float* p2 = &mesh->positions[i2 * 3];
                const float* uv0 = &mesh->texcoords[i0 * 2];
                const float* uv1 = &mesh->texcoords[i1 * 2];
                const float* uv2 = &mesh->texcoords[i2 * 2];

                float e1[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
                float e2[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
                /* Negate dv to match the V-flip applied at vertex-buffer
                 * pack (uv.y stored as 1-v). The tangent / bitangent
                 * frame stays consistent with the GPU's UV convention,
                 * so normal-mapping reads through tangent space without
                 * a left-handed flip. */
                float du1 = uv1[0]-uv0[0], dv1 = -(uv1[1]-uv0[1]);
                float du2 = uv2[0]-uv0[0], dv2 = -(uv2[1]-uv0[1]);

                float det = du1*dv2 - du2*dv1;
                float r = (fabsf(det) > 1e-8f) ? 1.0f / det : 0.0f;

                float tx = r * (dv2*e1[0] - dv1*e2[0]);
                float ty = r * (dv2*e1[1] - dv1*e2[1]);
                float tz = r * (dv2*e1[2] - dv1*e2[2]);

                float bx = r * (-du2*e1[0] + du1*e2[0]);
                float by = r * (-du2*e1[1] + du1*e2[1]);
                float bz = r * (-du2*e1[2] + du1*e2[2]);

                /* Accumulate for averaging */
                tangents[i0*3+0] += tx; tangents[i0*3+1] += ty; tangents[i0*3+2] += tz;
                tangents[i1*3+0] += tx; tangents[i1*3+1] += ty; tangents[i1*3+2] += tz;
                tangents[i2*3+0] += tx; tangents[i2*3+1] += ty; tangents[i2*3+2] += tz;
                bitangents[i0*3+0] += bx; bitangents[i0*3+1] += by; bitangents[i0*3+2] += bz;
                bitangents[i1*3+0] += bx; bitangents[i1*3+1] += by; bitangents[i1*3+2] += bz;
                bitangents[i2*3+0] += bx; bitangents[i2*3+1] += by; bitangents[i2*3+2] += bz;
            }

            /* Normalize accumulated tangents and compute bitangent signs */
            for (int i = 0; i < mesh->nvertices; i++) {
                float* t = &tangents[i * 3];
                float len = sqrtf(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]);
                if (len > 1e-8f) {
                    t[0] /= len; t[1] /= len; t[2] /= len;
                } else {
                    t[0] = 1.0f; t[1] = 0.0f; t[2] = 0.0f;
                }
                /* Compute bitangent sign: dot(cross(N, T), B) */
                if (mesh->normals) {
                    const float* n = &mesh->normals[i * 3];
                    float* b = &bitangents[i * 3];
                    float cx = n[1]*t[2] - n[2]*t[1];
                    float cy = n[2]*t[0] - n[0]*t[2];
                    float cz = n[0]*t[1] - n[1]*t[0];
                    float d = cx*b[0] + cy*b[1] + cz*b[2];
                    tan_signs[i] = (d < 0.0f) ? -1.0f : 1.0f;
                } else {
                    tan_signs[i] = 1.0f;
                }
            }
        }

        for (int i = 0; i < mesh->nvertices; i++) {
            const float* src_pos = &mesh->positions[i * 3];
            *dst++ = src_pos[0];
            *dst++ = src_pos[1];
            *dst++ = src_pos[2];

            if (mesh->normals) {
                const float* src_nrm = &mesh->normals[i * 3];
                *dst++ = src_nrm[0];
                *dst++ = src_nrm[1];
                *dst++ = src_nrm[2];
            } else {
                *dst++ = 0.0f;
                *dst++ = 1.0f;
                *dst++ = 0.0f;
            }

            if (mesh->colors) {
                const float* src_col = &mesh->colors[i * 3];
                *dst++ = src_col[0];
                *dst++ = src_col[1];
                *dst++ = src_col[2];
            } else {
                *dst++ = fallback_color[0];
                *dst++ = fallback_color[1];
                *dst++ = fallback_color[2];
            }

            if (mesh->texcoords) {
                /* USD authors UVs with V=0 at the bottom (OpenGL convention),
                 * but stb_image loads textures top-down (V=0 = first row).
                 * The chess set's normal/base-color JPEGs were authored
                 * for a top-V convention — without flipping V here textures
                 * sample upside-down and wrap-stitch wrong at UV seams,
                 * which presents as the "wrong UV interpolation" the user
                 * spotted. nanousd-vulkan-renderer/src/renderer.c:635
                 * applies the same `1.0 - v` on upload. */
                *dst++ = mesh->texcoords[i * 2];
                *dst++ = 1.0f - mesh->texcoords[i * 2 + 1];
            } else {
                *dst++ = 0.0f;
                *dst++ = 0.0f;
            }

            *dst++ = tangents[i * 3 + 0];
            *dst++ = tangents[i * 3 + 1];
            *dst++ = tangents[i * 3 + 2];
            *dst++ = tan_signs ? tan_signs[i] : 1.0f;
        }

        free(tangents);
        free(bitangents);
        free(tan_signs);

        vertex_cursor += (uint32_t)mesh->nvertices;
    }

    *out_total = total;
    free(needed);
    return buf;
}

/* ---- Input handlers (shared by GLFW callbacks + viewer_inject_*) ----
 *
 * Pulled out of the GLFW callbacks so the same code path runs for real
 * mouse / keyboard events and for synthetic events injected via the
 * public viewer_inject_* API. NUSD_DEBUG_INPUT=1 in the env logs every
 * dispatched event to stderr — useful when "the camera doesn't move"
 * to find out whether the events are even arriving. */

/* ---- Timing helper ---- */

static double get_time_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}


/* ---- Public API ----
 *
 * The `headless` parameter is preserved for ABI compatibility but the
 * windowed (GLFW) path now lives in viewer_window.c — viewer_create
 * always creates or reuses a GL context. Standalone callers create a
 * GLFW window first via viewer_window_setup(), which makes its context
 * current; viewer_create then detects "GL context already current,
 * reuse it" and stays out of GLFW's way. Embed callers (ovgear via
 * gl_thread.c) get the same path: their host already provides a
 * context. With no current context, the EGL/CGL pbuffer fallback
 * kicks in. */
Viewer* viewer_create(const char* usd_path, int width, int height,
                      int max_tex_size, const char* envmap_path,
                      int headless, int enable_materials)
{
    (void)headless;  /* always headless internally; see header comment */

    Viewer* v = (Viewer*)calloc(1, sizeof(Viewer));
    if (!v) return NULL;

    v->width = width;
    v->height = height;
    v->fallback_material_index = -1;

    /* Set texture resolution cap before material loading */
    if (max_tex_size > 0)
        materials_set_max_tex_size(max_tex_size);

    /* Reuse a GL context already current on this thread (standalone:
     * GLFW just made one current; embed: ovui or gl_thread did). With
     * no current context, fall back to a fresh headless EGL/CGL
     * pbuffer — that's how `--headless` mode and standalone CI tests
     * (no display) get a context. */
    if (egl_headless_has_current()) {
        v->egl_ctx = NULL;
        fprintf(stderr, "viewer: reusing existing GL context\n");
    } else {
        v->egl_ctx = egl_headless_create(width, height);
        if (!v->egl_ctx) {
            fprintf(stderr, "viewer: failed to create headless GL context\n");
            free(v);
            return NULL;
        }
    }

    /* ---- Load scene ---- */

    double t0 = get_time_seconds();

    /* The OpenGL material path still loads MaterialCollection from the live
     * USD stage in viewer_create(), then assigns bindings against that stage.
     * A geometry-cache hit reconstructs Scene without a stage, so a
     * material-enabled cache hit would silently degrade to a materialless
     * render. Until NZGO grows a material section like Vulkan's NZGC, bypass
     * the geometry cache for material-enabled OpenGL loads. Geometry-only
     * OpenGL loads keep using the cache. */
    char* saved_geo_cache = NULL;
    int had_geo_cache = 0;
    if (enable_materials) {
        const char* env = getenv("NUSD_GEO_CACHE");
        if (env) {
            size_t env_len = strlen(env);
            saved_geo_cache = (char*)malloc(env_len + 1);
            if (saved_geo_cache)
                memcpy(saved_geo_cache, env, env_len + 1);
            if (saved_geo_cache) {
                had_geo_cache = 1;
                unsetenv("NUSD_GEO_CACHE");
                fprintf(stderr,
                        "viewer: bypassing NUSD_GEO_CACHE for material-enabled "
                        "OpenGL load (materials require live USD stage)\n");
            }
        }
    }

    scene_set_load_flags(enable_materials ? 0u : SCENE_LOAD_SKIP_TEXCOORDS);
    v->scene = scene_load(usd_path);
    scene_set_load_flags(0u);
    if (had_geo_cache) {
        setenv("NUSD_GEO_CACHE", saved_geo_cache ? saved_geo_cache : "", 1);
        free(saved_geo_cache);
    }
    if (!v->scene) {
        fprintf(stderr, "viewer: scene_load failed for: %s\n", usd_path);
        if (v->egl_ctx) egl_headless_destroy(v->egl_ctx);
        free(v);
        return NULL;
    }

    double t1 = get_time_seconds();
    fprintf(stderr, "viewer: scene loaded in %.1f ms  (%d meshes)\n",
            (t1 - t0) * 1000.0, v->scene->nmeshes);

    /* ---- Init GPU ---- */

    int fb_width = width;
    int fb_height = height;
    v->width = fb_width;
    v->height = fb_height;

    v->gpu = gpu_init(NULL, fb_width, fb_height);
    if (!v->gpu) {
        fprintf(stderr, "viewer: gpu_init failed\n");
        scene_free(v->scene);
        if (v->egl_ctx) egl_headless_destroy(v->egl_ctx);
        free(v);
        return NULL;
    }

    /* ---- Init overlay text rendering ---- */
    gpu_overlay_init(v->gpu);
    if (v->scene->nlights > 0) {
        GpuLight glights[GPU_MAX_SCENE_LIGHTS];
        int n = v->scene->nlights;
        if (n > GPU_MAX_SCENE_LIGHTS) n = GPU_MAX_SCENE_LIGHTS;
        memset(glights, 0, sizeof(glights));
        for (int i = 0; i < n; i++) {
            const SceneLight* sl = &v->scene->lights[i];
            GpuLight* gl = &glights[i];
            v->shadow_lights[i] = *sl;
            memcpy(gl->position, sl->position, 3 * sizeof(float));
            memcpy(gl->normal, sl->normal, 3 * sizeof(float));
            memcpy(gl->u_axis, sl->u_axis, 3 * sizeof(float));
            memcpy(gl->v_axis, sl->v_axis, 3 * sizeof(float));
            memcpy(gl->color, sl->color, 3 * sizeof(float));
            gl->intensity = sl->intensity;
            gl->kind = sl->kind;
            gl->normalize = sl->normalize;
            gl->angle_deg = sl->angle_deg;
        }
        v->shadow_light_count = n;
        gpu_upload_lights(v->gpu, glights, n);
        fprintf(stderr, "viewer: uploaded %d authored lights\n", n);
    } else {
        v->shadow_light_count = 0;
        gpu_upload_lights(v->gpu, NULL, 0);
    }
    gpu_set_authored_light_count(v->gpu, v->scene->nlights);
    /* Match Vulkan/OVRTX fallback policy: any authored light, including a
     * DomeLight, owns the lighting setup and disables the synthetic rig. */
    gpu_set_fallback_lighting(v->gpu, v->scene->has_authored_light ? 0 : 1);

    /* ---- Load IBL environment ---- */
    if (envmap_path) {
        if (gpu_load_environment(v->gpu, envmap_path))
            fprintf(stderr, "viewer: IBL environment loaded\n");
        else
            fprintf(stderr, "viewer: IBL load failed, using fallback lighting\n");
    } else if (v->scene->has_dome_light && v->scene->dome_hdr_path[0]) {
        const char* dome_path = v->scene->dome_hdr_path;
        float loaded_intensity = v->scene->dome_intensity;
        int loaded = gpu_load_environment_tinted_intensity(
            v->gpu, dome_path, loaded_intensity, v->scene->dome_color);
        char fallback_path[1024];
        if (!loaded &&
            dome_visible_fallback_path(v->scene->dome_hdr_path, fallback_path,
                                       sizeof(fallback_path))) {
            loaded_intensity = visible_dome_fallback_intensity();
            fprintf(stderr,
                    "viewer: trying visible dome fallback %s "
                    "(internal LDR sky intensity=%.3f)\n",
                    fallback_path, loaded_intensity);
            loaded = gpu_load_environment_tinted_intensity(
                v->gpu, fallback_path, loaded_intensity, v->scene->dome_color);
            if (loaded)
                dome_path = fallback_path;
        }
        if (loaded) {
            fprintf(stderr,
                    "viewer: dome-light IBL loaded: %s (intensity=%.3f)\n",
                    dome_path, loaded_intensity);
        } else {
            fprintf(stderr,
                    "viewer: dome-light IBL load failed for %s\n",
                    v->scene->dome_hdr_path);
        }
    } else if (v->scene->has_dome_light) {
        int loaded = gpu_load_flat_environment(v->gpu,
                                               v->scene->dome_color,
                                               v->scene->dome_intensity);
        if (loaded) {
            fprintf(stderr,
                    "viewer: flat DomeLight IBL "
                    "(color=%.3f,%.3f,%.3f intensity=%.3f)\n",
                    v->scene->dome_color[0],
                    v->scene->dome_color[1],
                    v->scene->dome_color[2],
                    v->scene->dome_intensity);
        } else {
            fprintf(stderr,
                    "viewer: flat DomeLight IBL creation failed\n");
        }
    } else {
        /* Auto-discover a default HDR environment map */
        char exe_dir[1024] = "";
        {
            char exe_path[1024];
            ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
            if (n > 0) {
                exe_path[n] = '\0';
                /* Strip binary name to get directory */
                char* last = strrchr(exe_path, '/');
                if (last) { *last = '\0'; snprintf(exe_dir, sizeof(exe_dir), "%s", exe_path); }
            }
        }
        /* Search paths: sibling Vulkan viewer build, system MaterialX */
        char search[4][1024];
        int nsearch = 0;
        if (exe_dir[0]) {
            /* Same build dir (if MaterialX fetched here) */
            snprintf(search[nsearch++], 1024,
                     "%s/_deps/materialx-src/resources/Lights/san_giuseppe_bridge.hdr", exe_dir);
            /* Sibling Vulkan renderer build */
            snprintf(search[nsearch++], 1024,
                     "%s/../../nanousd-vulkan-renderer/build/_deps/materialx-src/resources/Lights/san_giuseppe_bridge.hdr", exe_dir);
        }
        for (int i = 0; i < nsearch; i++) {
            FILE* f = fopen(search[i], "rb");
            if (f) {
                fclose(f);
                if (gpu_load_environment(v->gpu, search[i])) {
                    fprintf(stderr, "viewer: auto-loaded IBL from %s\n", search[i]);
                    break;
                }
            }
        }
    }

    /* ---- Load materials ---- */

    char scene_dir[1024] = ".";
    {
        const char* last_slash = strrchr(usd_path, '/');
        const char* last_bslash = strrchr(usd_path, '\\');
        const char* sep = last_slash;
        if (last_bslash && last_bslash > sep) sep = last_bslash;
        if (sep) {
            size_t dir_len = (size_t)(sep - usd_path);
            if (dir_len >= sizeof(scene_dir)) dir_len = sizeof(scene_dir) - 1;
            memcpy(scene_dir, usd_path, dir_len);
            scene_dir[dir_len] = '\0';
        }
    }

    /* No MaterialX init needed for GLES — we use built-in PBR shaders */

    double t_mat1 = get_time_seconds();
    if (enable_materials) {
        double t_mat0 = get_time_seconds();
        v->materials = materials_load(v->scene->_stage, scene_dir);
        t_mat1 = get_time_seconds();
        fprintf(stderr, "viewer: materials_load %.1f ms\n",
                (t_mat1 - t_mat0) * 1000.0);
    } else {
        fprintf(stderr, "viewer: materials disabled by renderer config\n");
    }

    if (enable_materials && v->materials && v->materials->nmaterials > 0) {
        materials_assign_bindings(v->materials, v->scene->_stage,
                                  v->scene->meshes, v->scene->nmeshes);
        v->has_materials = 1;
        double t_bind = get_time_seconds();
        fprintf(stderr, "viewer: material binding total %.1f ms\n",
                (t_bind - t_mat1) * 1000.0);

        /* Build mesh draw order sorted by material_index. Crucial for
         * the same-material cache in gpu_cmd_bind_material — without
         * sorting, adjacent draws hit different materials almost every
         * time and the cache misses. Counting sort: O(N + nmaterials). */
        int n = v->scene->nmeshes;
        int nmat = v->materials->nmaterials;
        v->draw_order = (int*)malloc((size_t)n * sizeof(int));
        int* counts = (int*)calloc((size_t)(nmat + 1), sizeof(int));
        if (v->draw_order && counts) {
            /* counts[0] = unmatched (-1), counts[m+1 .. nmat] = material m. */
            for (int m = 0; m < n; m++) {
                int mi = v->scene->meshes[m].material_index;
                counts[(mi >= 0 && mi < nmat) ? (mi + 1) : 0]++;
            }
            /* exclusive prefix-sum → bucket starts */
            int run = 0;
            for (int b = 0; b <= nmat; b++) {
                int c = counts[b];
                counts[b] = run;
                run += c;
            }
            /* place each mesh */
            for (int m = 0; m < n; m++) {
                int mi = v->scene->meshes[m].material_index;
                int b = (mi >= 0 && mi < nmat) ? (mi + 1) : 0;
                v->draw_order[counts[b]++] = m;
            }
        }
        free(counts);
    }

    uint32_t total_verts = 0;
    uint32_t total_idx = 0;
    if (!enable_materials) {
        double t_rb0 = get_time_seconds();
        if (!build_chunked_raster_buffers(v, &total_verts, &total_idx)) {
            fprintf(stderr, "viewer: failed to build chunked raster buffers\n");
            gpu_shutdown(v->gpu);
            scene_free(v->scene);
            if (v->egl_ctx) egl_headless_destroy(v->egl_ctx);
            free(v);
            return NULL;
        }
        v->total_vertices = total_verts;
        v->total_indices = total_idx;
        fprintf(stderr, "viewer: build_chunked_raster_buffers %.1f ms\n",
                (get_time_seconds() - t_rb0) * 1000.0);
        fprintf(stderr, "viewer: %u vertices (%.1f MB), %u indices (%u triangles)\n",
                total_verts,
                (double)total_verts * (double)sizeof(RasterVertexPacked) /
                    (1024.0 * 1024.0),
                total_idx, total_idx / 3);
        if (meshlet_draw_enabled() && !build_meshlet_index_buffer(v))
            fprintf(stderr, "viewer: NUSD_MESHLET_DRAW set but no cached "
                            "meshlets — using whole-mesh draw\n");
    } else {
        scene_apply_geo_dedup_gl(v->scene);

        /* ---- Build merged vertex buffer ---- */

        double t_vb0 = get_time_seconds();
        float* verts = build_merged_vertex_buffer(v->scene, &total_verts);
        v->total_vertices = total_verts;
        fprintf(stderr, "viewer: build_merged_vertex_buffer %.1f ms\n",
                (get_time_seconds() - t_vb0) * 1000.0);

        if (total_verts > 0 && verts) {
            GpuBufferDesc vb_desc;
            vb_desc.usage = GPU_BUFFER_VERTEX;
            vb_desc.size  = (uint64_t)total_verts * 9 * sizeof(float);
            vb_desc.data  = verts;
            fprintf(stderr, "viewer: uploading VB (%.1f MB)...\n",
                    (double)vb_desc.size / (1024.0 * 1024.0));
            v->vertex_buffer = gpu_create_buffer(v->gpu, &vb_desc);
            fprintf(stderr, "viewer: VB upload %s\n", v->vertex_buffer ? "ok" : "FAILED");
            free(verts);

            if (!v->vertex_buffer) {
                fprintf(stderr, "viewer: failed to create vertex buffer\n");
                gpu_shutdown(v->gpu);
                scene_free(v->scene);
                if (v->egl_ctx) egl_headless_destroy(v->egl_ctx);
                free(v);
                return NULL;
            }
        }

        fprintf(stderr, "viewer: %u vertices (%.1f MB)\n",
                total_verts, (double)total_verts * 36.0 / (1024.0 * 1024.0));

        /* ---- Build merged index buffer (pre-offset for GLES) ---- */

        double t_ib0 = get_time_seconds();
        uint32_t* indices = build_merged_index_buffer(v->scene, &total_idx);
        v->total_indices = total_idx;
        fprintf(stderr, "viewer: build_merged_index_buffer %.1f ms\n",
                (get_time_seconds() - t_ib0) * 1000.0);

        if (total_idx > 0 && indices) {
            GpuBufferDesc ib_desc;
            ib_desc.usage = GPU_BUFFER_INDEX;
            ib_desc.size  = (uint64_t)total_idx * sizeof(uint32_t);
            ib_desc.data  = indices;
            fprintf(stderr, "viewer: uploading IB (%.1f MB)...\n",
                    (double)ib_desc.size / (1024.0 * 1024.0));
            v->index_buffer = gpu_create_buffer(v->gpu, &ib_desc);
            fprintf(stderr, "viewer: IB upload %s\n", v->index_buffer ? "ok" : "FAILED");
            free(indices);

            if (!v->index_buffer) {
                fprintf(stderr, "viewer: failed to create index buffer\n");
                if (v->vertex_buffer)
                    gpu_destroy_buffer(v->gpu, v->vertex_buffer);
                gpu_shutdown(v->gpu);
                scene_free(v->scene);
                if (v->egl_ctx) egl_headless_destroy(v->egl_ctx);
                free(v);
                return NULL;
            }
        }

        fprintf(stderr, "viewer: %u indices (%u triangles)\n",
                total_idx, total_idx / 3);

        {
            double t_ptex0 = get_time_seconds();
            viewer_upload_scene_ptex_triangle_colors(v);
            fprintf(stderr, "viewer: OpenGL Ptex triangle-color setup %.1f ms\n",
                    (get_time_seconds() - t_ptex0) * 1000.0);
        }
    }

    /* ---- Create raster pipeline (GLSL strings, no SPIR-V) ---- */

    GpuVertexAttrib attribs[3];
    memset(attribs, 0, sizeof(attribs));
    attribs[0].location = 0;
    attribs[0].offset = 0;
    attribs[0].format = GPU_FORMAT_FLOAT3;
    attribs[1].location = 1;
    attribs[1].offset = 12;
    attribs[1].format = enable_materials ? GPU_FORMAT_FLOAT3 : GPU_FORMAT_SNORM16X4;
    attribs[2].location = 2;
    attribs[2].offset = enable_materials ? 24u : 20u;
    attribs[2].format = enable_materials ? GPU_FORMAT_FLOAT3 : GPU_FORMAT_UNORM8X4;

    GpuPipelineDesc pipe_desc;
    memset(&pipe_desc, 0, sizeof(pipe_desc));
    pipe_desc.vert_glsl          = k_mesh_vert_gles;
    pipe_desc.frag_glsl          = k_mesh_frag_gles;
    pipe_desc.push_constant_size = (uint32_t)sizeof(GpuMeshPushConstants);
    pipe_desc.vertex_stride      = enable_materials ? 36u :
                                   (uint32_t)sizeof(RasterVertexPacked);
    pipe_desc.attribs            = attribs;
    pipe_desc.nattribs           = 3;

    v->pipeline = gpu_create_pipeline(v->gpu, &pipe_desc);

    if (!v->pipeline) {
        fprintf(stderr, "viewer: failed to create raster pipeline\n");
        if (v->index_buffer)  gpu_destroy_buffer(v->gpu, v->index_buffer);
        if (v->vertex_buffer) gpu_destroy_buffer(v->gpu, v->vertex_buffer);
        gpu_shutdown(v->gpu);
        scene_free(v->scene);
        if (v->egl_ctx) egl_headless_destroy(v->egl_ctx);
        free(v);
        return NULL;
    }

    if (enable_materials && v->scene && v->scene->nlights > 0) {
        GpuVertexAttrib shadow_attrib;
        memset(&shadow_attrib, 0, sizeof(shadow_attrib));
        shadow_attrib.location = 0;
        shadow_attrib.offset = 0;
        shadow_attrib.format = GPU_FORMAT_FLOAT3;

        GpuPipelineDesc shadow_desc;
        memset(&shadow_desc, 0, sizeof(shadow_desc));
        shadow_desc.vert_glsl = k_shadow_vert_gles;
        shadow_desc.frag_glsl = k_shadow_frag_gles;
        shadow_desc.vertex_stride = 36u;
        shadow_desc.attribs = &shadow_attrib;
        shadow_desc.nattribs = 1;
        v->shadow_pipeline = gpu_create_pipeline(v->gpu, &shadow_desc);
        if (!v->shadow_pipeline)
            fprintf(stderr, "viewer: shadow pipeline creation failed; shadows disabled\n");
    }

    /* ---- BasisCurves: build per-scene VBO + IBO + pipeline ----
     *
     * Curves go through the same per-mesh UBO infra as meshes — each
     * SceneCurve gets a slot at index `nmeshes + i`. CVs are
     * interleaved (pos, width, packed color) at stride 20 bytes; patch
     * indices are pre-offset to absolute vertex space (uint32, 4 per
     * patch). The CPU draw metadata chunks large curves for frustum
     * culling before tessellation. The smoke-test
     * (NUSD_CURVES_SMOKETEST=1) still works on mesh-only scenes. */
    {
        const char* st = getenv("NUSD_CURVES_SMOKETEST");
        v->curves_smoketest = (st && *st && *st != '0') ? 1 : 0;
    }
    int build_curve_pipeline = (v->scene && v->scene->ncurves > 0)
                                || v->curves_smoketest;
    if (build_curve_pipeline) {
        /* (a) Build curve VBO + IBO. */
        if (v->scene && v->scene->ncurves > 0) {
            int patch_chunk_i = env_int_or("NUSD_GL_CURVE_PATCH_CHUNK", 8192);
            if (patch_chunk_i < 256) patch_chunk_i = 256;
            if (patch_chunk_i > 65536) patch_chunk_i = 65536;
            uint32_t patch_chunk = (uint32_t)patch_chunk_i;
            uint32_t total_cv = 0, total_idx = 0;
            uint64_t total_patches = 0;
            int ndraws = 0;
            for (int c = 0; c < v->scene->ncurves; c++) {
                SceneCurve* cu = &v->scene->curves[c];
                if (!cu->cvs || cu->nv <= 0 || !cu->patch_indices || cu->npatches <= 0)
                    continue;
                total_cv  += (uint32_t)cu->nv;
                total_idx += (uint32_t)cu->npatches * 4u;
                total_patches += (uint64_t)cu->npatches;
                ndraws += (cu->npatches + (int)patch_chunk - 1) / (int)patch_chunk;
            }
            if (total_cv > 0 && total_idx > 0 && ndraws > 0) {
                /* Interleaved buffer: pos(12) + width(4) + packed color(4) = 20 B. */
                CurveVertexPacked* vb =
                    (CurveVertexPacked*)malloc((size_t)total_cv * sizeof(CurveVertexPacked));
                uint8_t* ib = (uint8_t*)malloc((size_t)total_idx * sizeof(uint32_t));
                ViewerCurveDraw* draws =
                    (ViewerCurveDraw*)calloc((size_t)ndraws, sizeof(ViewerCurveDraw));
                if (vb && ib && draws) {
                    uint32_t voff = 0;
                    uint64_t ibyte = 0;
                    int chunks16 = 0;
                    int chunks32 = 0;
                    int draw_i = 0;
                    for (int c = 0; c < v->scene->ncurves; c++) {
                        SceneCurve* cu = &v->scene->curves[c];
                        if (!cu->cvs || cu->nv <= 0 || !cu->patch_indices || cu->npatches <= 0)
                            continue;
                        cu->vertex_offset = voff;
                        cu->index_offset  = 0;
                        for (int j = 0; j < cu->nv; j++) {
                            CurveVertexPacked* dst = &vb[(size_t)voff + (size_t)j];
                            dst->position[0] = cu->cvs[j*3+0];
                            dst->position[1] = cu->cvs[j*3+1];
                            dst->position[2] = cu->cvs[j*3+2];
                            dst->width = cu->widths ? cu->widths[j] : 0.05f;
                            if (cu->colors) {
                                dst->color_rgba8 = pack_rgba8(cu->colors[j*3+0],
                                                              cu->colors[j*3+1],
                                                              cu->colors[j*3+2],
                                                              1.0f);
                            } else {
                                dst->color_rgba8 = pack_rgba8(0.7f, 0.7f, 0.7f, 1.0f);
                            }
                        }
                        for (uint32_t patch_start = 0;
                             patch_start < (uint32_t)cu->npatches;
                             patch_start += patch_chunk) {
                            uint32_t chunk_n = (uint32_t)cu->npatches - patch_start;
                            if (chunk_n > patch_chunk) chunk_n = patch_chunk;
                            uint32_t index_start = patch_start * 4u;
                            uint32_t index_count = chunk_n * 4u;
                            uint32_t min_idx = UINT32_MAX;
                            uint32_t max_idx = 0;
                            for (uint32_t j = 0; j < index_count; j++) {
                                uint32_t idx = cu->patch_indices[index_start + j];
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
                            ibyte = align_up_u64(ibyte, index_align);

                            ViewerCurveDraw* d = &draws[draw_i++];
                            d->curve_id = (uint32_t)c;
                            d->index_byte_offset = ibyte;
                            d->npatches = chunk_n;
                            d->vertex_offset = use_u16 ? (int32_t)base_vertex64 : 0;
                            d->index_type_bits = index_bits;
                            curve_chunk_bounds_world(cu, patch_start, chunk_n,
                                                     d->bounds_min, d->bounds_max);
                            if (use_u16) {
                                uint16_t* dst = (uint16_t*)(void*)(ib + ibyte);
                                for (uint32_t j = 0; j < index_count; j++)
                                    dst[j] = (uint16_t)(cu->patch_indices[index_start + j] - min_idx);
                                ibyte += (uint64_t)index_count * sizeof(uint16_t);
                                chunks16++;
                            } else {
                                uint32_t* dst = (uint32_t*)(void*)(ib + ibyte);
                                for (uint32_t j = 0; j < index_count; j++)
                                    dst[j] = cu->patch_indices[index_start + j] + voff;
                                ibyte += (uint64_t)index_count * sizeof(uint32_t);
                                chunks32++;
                            }
                        }
                        voff += (uint32_t)cu->nv;
                    }
                    GpuBufferDesc vbd = { .usage=GPU_BUFFER_VERTEX,
                                          .size=(uint64_t)total_cv * sizeof(CurveVertexPacked), .data=vb };
                    GpuBufferDesc ibd = { .usage=GPU_BUFFER_INDEX,
                                          .size=ibyte, .data=ib };
                    v->curve_vertex_buffer = gpu_create_buffer(v->gpu, &vbd);
                    v->curve_index_buffer  = gpu_create_buffer(v->gpu, &ibd);
                    v->curve_draws = draws;
                    v->curve_draw_count = draw_i;
                    draws = NULL;
                    fprintf(stderr,
                            "viewer: curves: %d prims, %d draw chunks, "
                            "%llu patches, %d-patch chunks, %.1f MiB VBO, "
                            "%.1f MiB IBO (%d u16 chunks / %d u32 chunks)\n",
                            v->scene->ncurves,
                            v->curve_draw_count,
                            (unsigned long long)total_patches,
                            patch_chunk_i,
                            (double)vbd.size / (1024.0 * 1024.0),
                            (double)ibd.size / (1024.0 * 1024.0),
                            chunks16, chunks32);
                }
                free(draws);
                free(vb);
                free(ib);
            }
        } else if (v->curves_smoketest) {
            /* Smoke-test fallback: hardcoded NDC patch. Only used when
             * the loaded scene has zero BasisCurves prims. */
            CurveVertexPacked cvs[4] = {
                { { -0.5f, -0.5f, 0.0f }, 0.10f, 0xffff00ffu },
                { { -0.5f,  0.5f, 0.0f }, 0.10f, 0xffff00ffu },
                { {  0.5f,  0.5f, 0.0f }, 0.10f, 0xffff00ffu },
                { {  0.5f, -0.5f, 0.0f }, 0.10f, 0xffff00ffu },
            };
            uint32_t idx[4] = { 0, 1, 2, 3 };
            GpuBufferDesc vbd = { .usage=GPU_BUFFER_VERTEX, .size=sizeof(cvs), .data=cvs };
            GpuBufferDesc ibd = { .usage=GPU_BUFFER_INDEX,  .size=sizeof(idx), .data=idx };
            v->curve_vertex_buffer = gpu_create_buffer(v->gpu, &vbd);
            v->curve_index_buffer  = gpu_create_buffer(v->gpu, &ibd);
        }

        /* (b) Pipeline. */
        if (v->curve_vertex_buffer && v->curve_index_buffer) {
            GpuVertexAttrib curve_attribs[3] = {
                { .location = 0, .offset = 0,  .format = GPU_FORMAT_FLOAT3 },
                { .location = 1, .offset = 12, .format = GPU_FORMAT_FLOAT1 },
                { .location = 2, .offset = 16, .format = GPU_FORMAT_UINT },
            };
            GpuPipelineDesc cpipe_desc;
            memset(&cpipe_desc, 0, sizeof(cpipe_desc));
            cpipe_desc.vert_glsl       = k_curve_vert_gles;
            cpipe_desc.frag_glsl       = k_curve_frag_gles;
            cpipe_desc.tcs_glsl        = k_curve_tcs_gles;
            cpipe_desc.tes_glsl        = k_curve_tes_gles;
            cpipe_desc.patch_vertices  = 4;
            cpipe_desc.vertex_stride   = (uint32_t)sizeof(CurveVertexPacked);
            cpipe_desc.attribs         = curve_attribs;
            cpipe_desc.nattribs        = 3;
            v->curve_pipeline = gpu_create_pipeline(v->gpu, &cpipe_desc);
        }

        if (!v->curve_pipeline || !v->curve_vertex_buffer || !v->curve_index_buffer) {
            fprintf(stderr, "viewer: curve pipeline setup FAILED — curves disabled\n");
            if (v->curve_pipeline)      gpu_destroy_pipeline(v->gpu, v->curve_pipeline);
            if (v->curve_vertex_buffer) gpu_destroy_buffer(v->gpu, v->curve_vertex_buffer);
            if (v->curve_index_buffer)  gpu_destroy_buffer(v->gpu, v->curve_index_buffer);
            free(v->curve_draws);
            v->curve_pipeline = NULL;
            v->curve_vertex_buffer = NULL;
            v->curve_index_buffer = NULL;
            v->curve_draws = NULL;
            v->curve_draw_count = 0;
            v->curves_smoketest = 0;
        } else if (v->scene && v->scene->ncurves > 0) {
            int total_patches = 0;
            int skipped = 0;
            for (int c = 0; c < v->scene->ncurves; c++) {
                total_patches += v->scene->curves[c].npatches;
                if (v->scene->curves[c].npatches == 0) skipped++;
            }
            if (v->curve_draw_count <= 0) {
                fprintf(stderr, "viewer: curves: %d prims, %d patches",
                        v->scene->ncurves, total_patches);
                if (skipped > 0)
                    fprintf(stderr, " (%d skipped — non-bezier or non-cubic; phase 3)",
                            skipped);
                fprintf(stderr, "\n");
            }
        } else {
            fprintf(stderr, "viewer: curves smoke test enabled (1 Bezier patch)\n");
        }
    }

    /* ---- Material pipeline (if materials loaded) ---- */

    if (v->has_materials && v->materials->nmaterials > 0) {
        /* Upload material data to GPU */
        const int real_nmaterials = v->materials->nmaterials;
        const int upload_nmaterials = real_nmaterials + 1;
        GpuMaterialParams* gpu_mats = (GpuMaterialParams*)calloc(
            (size_t)upload_nmaterials, sizeof(GpuMaterialParams));

        GpuTextureData* gpu_texs = NULL;
        int mat_ok = 0;
        if (!gpu_mats) {
            fprintf(stderr, "viewer: failed to allocate material upload buffer\n");
        } else {
            for (int i = 0; i < real_nmaterials; i++) {
                memcpy(&gpu_mats[i], &v->materials->materials[i].params,
                       sizeof(GpuMaterialParams));
            }
            GpuMaterialParams* fallback = &gpu_mats[real_nmaterials];
            fallback->base_color[0] = 1.0f;
            fallback->base_color[1] = 1.0f;
            fallback->base_color[2] = 1.0f;
            fallback->base_color[3] = 1.0f;
            fallback->opacity = 1.0f;
            fallback->ior = 1.5f;
            fallback->roughness = 0.55f;
            fallback->normal_scale = 1.0f;
            fallback->occlusion = 1.0f;
            fallback->use_vertex_color = 1;
            fallback->udim_scale_u = 1.0f;
            fallback->udim_scale_v = 1.0f;
            fallback->mdl_uv_transform[0] = 1.0f;
            fallback->mdl_uv_transform[1] = 1.0f;
            fallback->roughness_tex_scale = 1.0f;
            for (int t = 0; t < MAX_MATERIAL_TEXTURES; t++)
                fallback->tex_indices[t] = -1;

            if (v->materials->ntextures > 0) {
                gpu_texs = (GpuTextureData*)calloc(v->materials->ntextures,
                                                   sizeof(GpuTextureData));
                for (int i = 0; i < v->materials->ntextures; i++) {
                    gpu_texs[i].pixels = v->materials->textures[i].pixels;
                    gpu_texs[i].width  = v->materials->textures[i].width;
                    gpu_texs[i].height = v->materials->textures[i].height;
                    gpu_texs[i].is_srgb = v->materials->textures[i].is_srgb;
                }
            }

            double t_up0 = get_time_seconds();
            mat_ok = gpu_upload_materials(v->gpu,
                gpu_mats, upload_nmaterials,
                gpu_texs, v->materials->ntextures);
            fprintf(stderr,
                    "viewer: gpu_upload_materials %.1f ms (%d textures, %d materials + fallback)\n",
                    (get_time_seconds() - t_up0) * 1000.0,
                    v->materials->ntextures, real_nmaterials);
        }

        free(gpu_mats);
        free(gpu_texs);

        if (mat_ok) {
            v->fallback_material_index = real_nmaterials;
            /* Build material vertex buffer (44 bytes: no materialId) */
            uint32_t mat_total_verts = 0;
            float* mat_verts = build_material_vertex_buffer(v->scene,
                                                             &mat_total_verts);
            if (mat_verts && mat_total_verts > 0) {
                GpuBufferDesc mvb_desc;
                mvb_desc.usage = GPU_BUFFER_VERTEX;
                mvb_desc.size  = (uint64_t)mat_total_verts * MAT_VERTEX_STRIDE;
                mvb_desc.data  = mat_verts;
                v->mat_vertex_buffer = gpu_create_buffer(v->gpu, &mvb_desc);
                free(mat_verts);
            }

            /* Create PBR pipeline using GLES shader strings */
            GpuVertexAttrib mat_attribs[6] = {
                { .location = 0, .offset = 0,  .format = GPU_FORMAT_FLOAT3 },  /* pos */
                { .location = 1, .offset = 12, .format = GPU_FORMAT_FLOAT3 },  /* normal */
                { .location = 2, .offset = 24, .format = GPU_FORMAT_FLOAT3 },  /* color */
                { .location = 3, .offset = 36, .format = GPU_FORMAT_FLOAT2 },  /* texcoord */
                { .location = 4, .offset = 44, .format = GPU_FORMAT_FLOAT3 },  /* tangent */
                { .location = 5, .offset = 56, .format = GPU_FORMAT_FLOAT1 },  /* tangent sign */
            };

            GpuPipelineDesc mat_pipe_desc;
            memset(&mat_pipe_desc, 0, sizeof(mat_pipe_desc));
            mat_pipe_desc.vert_glsl     = k_pbr_vert_gles;
            mat_pipe_desc.frag_glsl     = k_pbr_frag_gles;
            mat_pipe_desc.push_constant_size = (uint32_t)sizeof(GpuMeshPushConstants);
            mat_pipe_desc.vertex_stride = MAT_VERTEX_STRIDE;
            mat_pipe_desc.attribs       = mat_attribs;
            mat_pipe_desc.nattribs      = 6;

            v->mat_pipeline = gpu_create_material_pipeline(v->gpu, &mat_pipe_desc);
            if (v->mat_pipeline) {
                v->render_mode = MODE_MATERIAL;
                fprintf(stderr, "viewer: material pipeline created\n");
            } else {
                fprintf(stderr, "viewer: material pipeline creation failed\n");
            }

            /* Instanced variant for compact-PI batches: same per-vertex attribs
             * + PBR fragment, but the instanced vertex shader reads a per-instance
             * world matrix from locations 6-9 (set at draw time). */
            GpuPipelineDesc inst_pipe_desc = mat_pipe_desc;
            inst_pipe_desc.vert_glsl = k_pbr_instanced_vert_gles;
            v->mat_instanced_pipeline = gpu_create_material_pipeline(v->gpu, &inst_pipe_desc);
            if (v->mat_instanced_pipeline)
                fprintf(stderr, "viewer: instanced material pipeline created\n");
            else
                fprintf(stderr, "viewer: instanced material pipeline creation failed "
                                "(compact PI batches will not draw)\n");
        }
    }

    /* No camera state lives in the viewer — view/proj are caller-supplied
     * to viewer_render_to_rgba on each frame. HUD on by default; embedders
     * that want a clean viewport call viewer_set_overlay_enabled(0). */
    (void)width; (void)height;
    v->overlay_enabled = 1;
    v->ui_scale = 1.0f;
    v->fps_time = get_time_seconds();
    v->fps_frames = 0;
    v->fps_display = 0.0f;
    v->total_meshes = v->scene ? v->scene->nmeshes : 0;

    {
        const char* slash = strrchr(usd_path, '/');
        const char* bslash = strrchr(usd_path, '\\');
        const char* name = usd_path;
        if (slash && slash > name) name = slash + 1;
        if (bslash && bslash > name) name = bslash + 1;
        snprintf(v->scene_name, sizeof(v->scene_name), "%s", name);
    }

    fprintf(stderr, "viewer: bounds min [%.2f, %.2f, %.2f]  max [%.2f, %.2f, %.2f]\n",
            v->scene->bounds_min[0], v->scene->bounds_min[1], v->scene->bounds_min[2],
            v->scene->bounds_max[0], v->scene->bounds_max[1], v->scene->bounds_max[2]);

    {
        double t_rel0 = get_time_seconds();
        long rss_before_kb = read_rss_kb();
        if (scene_release_mesh_payloads(v->scene)) {
            /* Hint glibc to return the freed arena pages to the kernel
             * so VmRSS drops immediately rather than waiting for the
             * arena to be reclaimed under memory pressure. */
#ifdef __GLIBC__
            malloc_trim(0);
#endif
            long rss_after_kb = read_rss_kb();
            fprintf(stderr,
                    "viewer: released mesh payload arena %.1f ms "
                    "(RSS %ld -> %ld MB, freed %ld MB)\n",
                    (get_time_seconds() - t_rel0) * 1000.0,
                    rss_before_kb / 1024,
                    rss_after_kb / 1024,
                    (rss_before_kb - rss_after_kb) / 1024);
        }
    }

    fprintf(stderr, "\n--- Controls ---\n");
    fprintf(stderr, "  WASD       Fly camera (forward/strafe)\n");
    fprintf(stderr, "  Q/E        Fly down/up\n");
    fprintf(stderr, "  Arrows     Orbit yaw/pitch\n");
    fprintf(stderr, "  PgUp/PgDn  Zoom in/out\n");
    fprintf(stderr, "  Shift      5x speed multiplier\n");
    fprintf(stderr, "  LMB drag   Orbit\n");
    fprintf(stderr, "  MMB drag   Pan\n");
    fprintf(stderr, "  RMB drag   Zoom\n");
    fprintf(stderr, "  Scroll     Zoom\n");
    fprintf(stderr, "  R          Reset camera to launch view\n");
    fprintf(stderr, "  F          Frame scene\n");
    fprintf(stderr, "  T          Toggle: RASTER / MATERIAL\n");
    fprintf(stderr, "  P          Screenshot (PPM)\n");
    fprintf(stderr, "  H          Toggle stats overlay\n");
    fprintf(stderr, "  ESC        Quit\n");
    fprintf(stderr, "----------------\n");
    fprintf(stderr, "(macOS tip: if held WASD doesn't repeat, run\n");
    fprintf(stderr, "  defaults write -g ApplePressAndHoldEnabled -bool false\n");
    fprintf(stderr, " and relaunch — or use the arrow keys, which aren't\n");
    fprintf(stderr, " intercepted by the press-and-hold accent menu.)\n\n");
    /* Allocate the per-mesh UBO (Plan A): one slice per mesh, indexed
     * via glBindBufferRange in the per-frame draw loop. Curves take
     * additional contiguous slots starting at `curve_ubo_slot`. */
    int nmeshes = (v->scene ? v->scene->nmeshes : 0);
    int ncurves_real = (v->scene ? v->scene->ncurves : 0);
    int curve_slots  = ncurves_real > 0 ? ncurves_real
                       : (v->curves_smoketest ? 1 : 0);
    v->curve_ubo_slot = nmeshes;
    /* Compact-PI batches get one MeshBlock slot each (mvp=VP, color/ptex from
     * the prototype) after the mesh + curve slots. */
    int nbatches = (v->scene ? v->scene->npi_batches : 0);
    v->batch_ubo_slot = nmeshes + curve_slots;
    if (nmeshes + curve_slots + nbatches > 0)
        gpu_alloc_mesh_buffer(v->gpu, nmeshes + curve_slots + nbatches);

    /* Upload all PI instance world matrices once (16 floats each). */
    v->instance_buffer_ready = 0;
    if (v->scene && v->scene->npi_transforms > 0 && v->scene->pi_transforms) {
        uint64_t n = v->scene->npi_transforms;
        float* m16 = (float*)malloc((size_t)n * 16 * sizeof(float));
        if (m16) {
            for (uint64_t i = 0; i < n; i++)
                scene_instance_transform_to_model16(&v->scene->pi_transforms[i],
                                                    &m16[i * 16]);
            gpu_upload_instance_transforms(v->gpu, m16, (uint32_t)n);
            free(m16);
            v->instance_buffer_ready = 1;
            fprintf(stderr,
                    "viewer: PI instance buffer ready — %llu instances (%.1f MiB), "
                    "%d batches\n",
                    (unsigned long long)n,
                    (double)(n * 16 * sizeof(float)) / (1024.0 * 1024.0), nbatches);
        }
    }

    fprintf(stderr, "viewer: ready\n");

    /* Release the headless GL context so an embedder's own context can
     * stay current on this thread. We re-claim it inside
     * viewer_render_to_rgba and release again on the way out. The
     * standalone binary's main loop keeps the headless surface
     * current — gated by the NUSD_GLES_EMBEDDED environment variable
     * so we only yield the thread when an embedder explicitly opts
     * in. With egl_ctx == NULL the GL context was inherited from the
     * caller (standalone GLFW window or embed host); we never release
     * a context we don't own. */
    if (v->egl_ctx) {
        const char* embed = getenv("NUSD_GLES_EMBEDDED");
        if (embed && *embed && *embed != '0')
            egl_headless_release_current(v->egl_ctx);
    }

    return v;
}

/* Update FPS + system stats every 0.5s. Returns 1 if the stats just
 * refreshed (caller can then update the window title). */
void viewer_resize(Viewer* viewer, int width, int height)
{
    if (!viewer || width <= 0 || height <= 0) return;
    if (width == viewer->width && height == viewer->height) return;

    /* If we own the headless context (standalone --headless or embed
     * with no host context), resize its pbuffer. With egl_ctx == NULL
     * the context belongs to the caller (GLFW window or ovgear host)
     * and they handle their own surface resize. */
    if (viewer->egl_ctx) {
        if (!egl_headless_resize(viewer->egl_ctx, width, height))
            return;
    }
    viewer->width = width;
    viewer->height = height;
    if (viewer->gpu)
        gpu_resize(viewer->gpu, width, height);
}

void viewer_set_render_mode(Viewer* viewer, ViewerRenderMode mode)
{
    if (!viewer) return;
    int m = (mode == VIEWER_MODE_MATERIAL) ? MODE_MATERIAL : MODE_RASTER;
    if (m == MODE_MATERIAL && !viewer->mat_pipeline) m = MODE_RASTER;
    viewer->render_mode = m;
}

void viewer_set_overlay_enabled(Viewer* viewer, int enabled)
{
    if (!viewer) return;
    viewer->overlay_enabled = enabled ? 1 : 0;
}

void viewer_set_tone_mapping(Viewer* viewer, float exposure_scale,
                             float sky_scale, float white_point_scale,
                             unsigned int flags)
{
    if (!viewer || !viewer->gpu) return;
    gpu_set_tone_mapping(viewer->gpu, exposure_scale, sky_scale,
                         white_point_scale, (uint32_t)flags);
}

int viewer_get_scene_bounds(Viewer* viewer, float out_min[3], float out_max[3])
{
    if (!viewer || !out_min || !out_max) return 0;
    if (!viewer->scene) {
        out_min[0] = out_min[1] = out_min[2] = 0.0f;
        out_max[0] = out_max[1] = out_max[2] = 0.0f;
        return 0;
    }
    out_min[0] = viewer->scene->bounds_min[0];
    out_min[1] = viewer->scene->bounds_min[1];
    out_min[2] = viewer->scene->bounds_min[2];
    out_max[0] = viewer->scene->bounds_max[0];
    out_max[1] = viewer->scene->bounds_max[1];
    out_max[2] = viewer->scene->bounds_max[2];
    return 1;
}

int viewer_get_scene_up_axis(Viewer* viewer)
{
    if (!viewer || !viewer->scene) return 1;
    int up = viewer->scene->up_axis;
    return (up >= 0 && up <= 2) ? up : 1;
}

int viewer_get_scene_has_authored_light(Viewer* viewer)
{
    return (viewer && viewer->scene && viewer->scene->has_authored_light) ? 1 : 0;
}

uint64_t viewer_get_gpu_memory_used(Viewer* viewer)
{
    return (viewer && viewer->gpu) ? gpu_get_allocated_memory(viewer->gpu) : 0;
}

static void viewer_xform_point_rowvec(const double m[16], const float p[3], float out[3])
{
    out[0] = p[0] * (float)m[0] + p[1] * (float)m[4] + p[2] * (float)m[8]  + (float)m[12];
    out[1] = p[0] * (float)m[1] + p[1] * (float)m[5] + p[2] * (float)m[9]  + (float)m[13];
    out[2] = p[0] * (float)m[2] + p[1] * (float)m[6] + p[2] * (float)m[10] + (float)m[14];
}

static void viewer_recompute_mesh_bounds(SceneMesh* mesh)
{
    float bmin[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float bmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                float p[3] = {
                    ix ? mesh->local_bounds_max[0] : mesh->local_bounds_min[0],
                    iy ? mesh->local_bounds_max[1] : mesh->local_bounds_min[1],
                    iz ? mesh->local_bounds_max[2] : mesh->local_bounds_min[2],
                };
                float wp[3];
                viewer_xform_point_rowvec(mesh->world_xform, p, wp);
                for (int k = 0; k < 3; k++) {
                    if (wp[k] < bmin[k]) bmin[k] = wp[k];
                    if (wp[k] > bmax[k]) bmax[k] = wp[k];
                }
            }
        }
    }
    memcpy(mesh->bounds_min, bmin, sizeof bmin);
    memcpy(mesh->bounds_max, bmax, sizeof bmax);
}

static void viewer_recompute_scene_bounds(Scene* scene)
{
    if (!scene) return;
    float bmin[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float bmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    int any = 0;

    for (int i = 0; i < scene->nmeshes; i++) {
        SceneMesh* mesh = &scene->meshes[i];
        if (!mesh->visible || mesh->is_proto_only || mesh->nindices <= 0) continue;
        for (int k = 0; k < 3; k++) {
            if (mesh->bounds_min[k] < bmin[k]) bmin[k] = mesh->bounds_min[k];
            if (mesh->bounds_max[k] > bmax[k]) bmax[k] = mesh->bounds_max[k];
        }
        any = 1;
    }
    for (int i = 0; i < scene->ncurves; i++) {
        SceneCurve* curve = &scene->curves[i];
        for (int k = 0; k < 3; k++) {
            if (curve->bounds_min[k] < bmin[k]) bmin[k] = curve->bounds_min[k];
            if (curve->bounds_max[k] > bmax[k]) bmax[k] = curve->bounds_max[k];
        }
        any = 1;
    }

    if (!any) {
        bmin[0] = bmin[1] = bmin[2] = 0.0f;
        bmax[0] = bmax[1] = bmax[2] = 0.0f;
    }
    memcpy(scene->bounds_min, bmin, sizeof bmin);
    memcpy(scene->bounds_max, bmax, sizeof bmax);
}

int viewer_get_mesh_count(Viewer* viewer)
{
    return (viewer && viewer->scene) ? viewer->scene->nmeshes : 0;
}

int viewer_get_mesh_name(Viewer* viewer, int mesh_id, char* out_buf, int buf_cap)
{
    if (!viewer || !viewer->scene || mesh_id < 0 || mesh_id >= viewer->scene->nmeshes)
        return -1;
    const char* path = viewer->scene->meshes[mesh_id].path;
    if (!path) path = "";
    int len = (int)strlen(path);
    if (out_buf && buf_cap > 0) {
        snprintf(out_buf, (size_t)buf_cap, "%s", path);
    }
    return len;
}

int viewer_get_mesh_transform(Viewer* viewer, int mesh_id, float out_mat4x4[16])
{
    if (!viewer || !viewer->scene || !out_mat4x4 ||
        mesh_id < 0 || mesh_id >= viewer->scene->nmeshes)
        return 0;
    const SceneMesh* mesh = &viewer->scene->meshes[mesh_id];
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            out_mat4x4[row * 4 + col] = (float)mesh->world_xform[col * 4 + row];
        }
    }
    return 1;
}

int viewer_set_transforms(Viewer* viewer, const int* ids, const float* mat4x4s, int count)
{
    if (!viewer || !viewer->scene || !ids || !mat4x4s || count <= 0) return 0;
    Scene* scene = viewer->scene;
    for (int i = 0; i < count; i++) {
        int id = ids[i];
        if (id < 0 || id >= scene->nmeshes) return 0;
    }
    for (int i = 0; i < count; i++) {
        int id = ids[i];
        SceneMesh* mesh = &scene->meshes[id];
        const float* src = &mat4x4s[i * 16];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                mesh->world_xform[row * 4 + col] = (double)src[col * 4 + row];
            }
        }
        viewer_recompute_mesh_bounds(mesh);
    }
    viewer_recompute_scene_bounds(scene);
    return 1;
}

int viewer_set_colors(Viewer* viewer, const int* ids, const float* rgb, int count)
{
    if (!viewer || !viewer->scene || !ids || !rgb || count <= 0) return 0;
    Scene* scene = viewer->scene;
    for (int i = 0; i < count; i++) {
        int id = ids[i];
        if (id < 0 || id >= scene->nmeshes) return 0;
    }
    for (int i = 0; i < count; i++) {
        SceneMesh* mesh = &scene->meshes[ids[i]];
        mesh->display_color[0] = rgb[i * 3 + 0];
        mesh->display_color[1] = rgb[i * 3 + 1];
        mesh->display_color[2] = rgb[i * 3 + 2];
        mesh->has_display_color = 1;
    }
    return 1;
}

int viewer_set_visibility(Viewer* viewer, const int* ids, const int* visible, int count)
{
    if (!viewer || !viewer->scene || !ids || !visible || count <= 0) return 0;
    Scene* scene = viewer->scene;
    for (int i = 0; i < count; i++) {
        int id = ids[i];
        if (id < 0 || id >= scene->nmeshes) return 0;
    }
    for (int i = 0; i < count; i++) {
        scene->meshes[ids[i]].visible = visible[i] ? 1 : 0;
    }
    viewer_recompute_scene_bounds(scene);
    return 1;
}

int viewer_load_environment(Viewer* viewer, const char* hdr_path)
{
    if (!viewer || !viewer->gpu || !hdr_path) return 0;
    return gpu_load_environment(viewer->gpu, hdr_path) ? 1 : 0;
}

int viewer_load_environment_intensity(Viewer* viewer, const char* hdr_path, float intensity)
{
    if (!viewer || !viewer->gpu || !hdr_path) return 0;
    return gpu_load_environment_intensity(viewer->gpu, hdr_path, intensity) ? 1 : 0;
}

/* HUD overlay (FPS / RSS / GPU mem / mode / vertex count / controls).
 * Drawn into the rendered frame so embedders that consume our pixels
 * (ovgear's image bridge) get the stats overlay for free. */
static void queue_overlay(Viewer* viewer)
{
    if (!viewer->overlay_enabled) return;

    float sc = viewer->ui_scale > 0.0f ? viewer->ui_scale : 1.0f;
    float lh = 16.0f * sc;
    float w  = (float)viewer->width;
    char line[256];

    /* Stats panel — upper right. */
    {
        float sx = w - 240.0f * sc;
        float sy = 8.0f * sc;
        float r = 0.0f, g = 1.0f, b = 0.0f, a = 0.9f;
        float pad = 6.0f * sc;
        gpu_overlay_rect(viewer->gpu, sx - pad, sy - pad,
                         240.0f * sc + pad * 2, lh * 7 + pad * 2,
                         0.0f, 0.0f, 0.0f, 0.55f);

        snprintf(line, sizeof(line), "FPS: %.0f", viewer->fps_display);
        gpu_overlay_text(viewer->gpu, sx, sy, sc, r, g, b, a, line); sy += lh;
        snprintf(line, sizeof(line), "RSS: %ld MB", viewer->rss_kb / 1024);
        gpu_overlay_text(viewer->gpu, sx, sy, sc, r, g, b, a, line); sy += lh;
        snprintf(line, sizeof(line), "VRAM: %d MB", viewer->gpu_mem_mb);
        gpu_overlay_text(viewer->gpu, sx, sy, sc, r, g, b, a, line); sy += lh;
        snprintf(line, sizeof(line), "Meshes: %d", viewer->total_meshes);
        gpu_overlay_text(viewer->gpu, sx, sy, sc, r, g, b, a, line); sy += lh;
        snprintf(line, sizeof(line), "Vertices: %uK",
                 viewer->total_vertices / 1000);
        gpu_overlay_text(viewer->gpu, sx, sy, sc, r, g, b, a, line); sy += lh;
        snprintf(line, sizeof(line), "CPU: %d C", viewer->cpu_temp_c);
        gpu_overlay_text(viewer->gpu, sx, sy, sc, r, g, b, a, line); sy += lh;

        snprintf(line, sizeof(line), "%s",
                 viewer->render_mode == MODE_MATERIAL ? "MATERIAL" : "RASTER");
        gpu_overlay_text(viewer->gpu, sx, sy, sc, 1.0f, 1.0f, 0.0f, a, line);
    }

    /* Scene name banner — upper left. */
    if (viewer->scene_name[0]) {
        float sx = 8.0f * sc;
        float sy = 8.0f * sc;
        float pad = 4.0f * sc;
        size_t name_len = strlen(viewer->scene_name);
        if (name_len > 64) name_len = 64;
        float bw = 8.0f * sc * (float)name_len + pad * 2;
        gpu_overlay_rect(viewer->gpu, sx - pad, sy - pad, bw, lh + pad * 2,
                         0.0f, 0.0f, 0.0f, 0.45f);
        gpu_overlay_text(viewer->gpu, sx, sy, sc,
                         0.55f, 0.82f, 1.0f, 0.95f, viewer->scene_name);
    }
}

static void hud_tick_stats(Viewer* v)
{
    v->fps_frames++;
    double now = get_time_seconds();
    double dt = now - v->fps_time;
    if (dt < 0.5) return;
    v->fps_display = (float)v->fps_frames / (float)dt;
    v->fps_frames  = 0;
    v->fps_time    = now;
    v->rss_kb      = read_rss_kb();
    v->gpu_mem_mb  = read_gpu_mem_mb(v->gpu);
    v->cpu_temp_c  = read_cpu_temp_c();
}

static int viewer_shadow_disabled(void)
{
    const char* e = getenv("NUSD_DISABLE_OPENGL_SHADOWS");
    return (e && e[0] && e[0] != '0') ? 1 : 0;
}

static int viewer_build_shadow_light_vp_for_index(Viewer* viewer, int light_index,
                                                  float out_vp[16])
{
    if (!viewer || !viewer->scene || !out_vp) return 0;
    if (light_index < 0 || light_index >= viewer->shadow_light_count) return 0;

    const SceneLight* L = &viewer->shadow_lights[light_index];
    if (L->kind != SCENE_LIGHT_RECT && L->kind != SCENE_LIGHT_DISTANT)
        return 0;
    float center[3] = {
        (viewer->scene->bounds_min[0] + viewer->scene->bounds_max[0]) * 0.5f,
        (viewer->scene->bounds_min[1] + viewer->scene->bounds_max[1]) * 0.5f,
        (viewer->scene->bounds_min[2] + viewer->scene->bounds_max[2]) * 0.5f
    };
    float extent[3] = {
        viewer->scene->bounds_max[0] - viewer->scene->bounds_min[0],
        viewer->scene->bounds_max[1] - viewer->scene->bounds_min[1],
        viewer->scene->bounds_max[2] - viewer->scene->bounds_min[2]
    };
    float diag = sqrtf(extent[0]*extent[0] + extent[1]*extent[1] +
                       extent[2]*extent[2]);
    if (!(diag > 1e-5f)) diag = 1.0f;

    float forward[3] = { L->normal[0], L->normal[1], L->normal[2] };
    if (vec3_normalize(forward) <= 1e-8f) {
        forward[0] = center[0] - L->position[0];
        forward[1] = center[1] - L->position[1];
        forward[2] = center[2] - L->position[2];
        if (vec3_normalize(forward) <= 1e-8f) {
            forward[0] = 0.0f; forward[1] = 0.0f; forward[2] = -1.0f;
        }
    }

    float eye[3];
    float target[3];
    float up[3] = { L->v_axis[0], L->v_axis[1], L->v_axis[2] };
    if (vec3_normalize(up) <= 1e-8f || fabsf(vec3_dot(up, forward)) > 0.95f) {
        up[0] = 0.0f; up[1] = 0.0f; up[2] = 1.0f;
        if (fabsf(vec3_dot(up, forward)) > 0.95f) {
            up[0] = 0.0f; up[1] = 1.0f; up[2] = 0.0f;
        }
    }

    if (L->kind == SCENE_LIGHT_DISTANT) {
        eye[0] = center[0] - forward[0] * diag;
        eye[1] = center[1] - forward[1] * diag;
        eye[2] = center[2] - forward[2] * diag;
        target[0] = center[0];
        target[1] = center[1];
        target[2] = center[2];
    } else {
        eye[0] = L->position[0];
        eye[1] = L->position[1];
        eye[2] = L->position[2];
        target[0] = eye[0] + forward[0];
        target[1] = eye[1] + forward[1];
        target[2] = eye[2] + forward[2];
    }

    float view[16];
    mat4_make_lookat(eye, target, up, view);

    float minv[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float maxv[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (int ix = 0; ix < 2; ix++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int iz = 0; iz < 2; iz++) {
                float p[3] = {
                    ix ? viewer->scene->bounds_max[0] : viewer->scene->bounds_min[0],
                    iy ? viewer->scene->bounds_max[1] : viewer->scene->bounds_min[1],
                    iz ? viewer->scene->bounds_max[2] : viewer->scene->bounds_min[2]
                };
                float q[3];
                mat4_transform_point(view, p, q);
                for (int a = 0; a < 3; a++) {
                    if (q[a] < minv[a]) minv[a] = q[a];
                    if (q[a] > maxv[a]) maxv[a] = q[a];
                }
            }
        }
    }

    float x_margin = fmaxf((maxv[0] - minv[0]) * 0.08f, 0.05f);
    float y_margin = fmaxf((maxv[1] - minv[1]) * 0.08f, 0.05f);
    float z_margin = fmaxf((maxv[2] - minv[2]) * 0.15f, 0.10f);
    float proj[16];
    mat4_make_ortho_bounds(minv[0] - x_margin, maxv[0] + x_margin,
                           minv[1] - y_margin, maxv[1] + y_margin,
                           minv[2] - z_margin, maxv[2] + z_margin,
                           proj);
    mat4_mul(proj, view, out_vp);
    return 1;
}

static int viewer_collect_shadow_lights(Viewer* viewer,
                                        float out_vp[GPU_MAX_SHADOW_LIGHTS][16],
                                        int out_indices[GPU_MAX_SHADOW_LIGHTS])
{
    if (!viewer || !out_vp || !out_indices) return 0;
    for (int i = 0; i < GPU_MAX_SHADOW_LIGHTS; i++)
        out_indices[i] = -1;
    if (viewer_shadow_disabled()) return 0;
    if (viewer->render_mode != MODE_MATERIAL || !viewer->shadow_pipeline)
        return 0;
    if (!viewer->vertex_buffer || !viewer->index_buffer) return 0;
    if (!viewer->scene) return 0;
    if (!viewer->scene->dome_hdr_path[0]) {
        /* Many-light, no-HDR scenes such as Isaac warehouse are handled by
         * the calibrated fallback lighting path. Rendering shadow maps for a
         * couple of arbitrary lights just burns time and is not sampled. */
        if (viewer->scene->nlights > 8)
            return 0;
    }

    int count = 0;
    for (int i = 0; i < viewer->shadow_light_count &&
                    count < GPU_MAX_SHADOW_LIGHTS; i++) {
        if (viewer_build_shadow_light_vp_for_index(viewer, i, out_vp[count])) {
            out_indices[count] = i;
            count++;
        }
    }
    return count;
}

static void viewer_bind_main_mesh_pipeline(Viewer* viewer)
{
    if (!viewer) return;
    if (viewer->render_mode == MODE_MATERIAL && viewer->mat_pipeline) {
        gpu_cmd_bind_pipeline(viewer->gpu, viewer->mat_pipeline);
        if (viewer->mat_vertex_buffer)
            gpu_cmd_bind_vertex_buffer(viewer->gpu, viewer->mat_vertex_buffer);
    } else {
        gpu_cmd_bind_pipeline(viewer->gpu, viewer->pipeline);
        if (viewer->vertex_buffer)
            gpu_cmd_bind_vertex_buffer(viewer->gpu, viewer->vertex_buffer);
    }
    if (viewer->index_buffer)
        gpu_cmd_bind_index_buffer(viewer->gpu, viewer->index_buffer);
    if (viewer->render_mode == MODE_MATERIAL && viewer->mat_pipeline)
        gpu_cmd_begin_material_pass(viewer->gpu);
}

static void viewer_render_shadow_map(Viewer* viewer, int slot,
                                     const float light_vp[16], int light_index)
{
    if (!viewer || !viewer->scene || !light_vp || !viewer->shadow_pipeline)
        return;
    if (!gpu_shadow_begin(viewer->gpu, slot, 1024, light_vp, light_index))
        return;

    gpu_cmd_bind_pipeline(viewer->gpu, viewer->shadow_pipeline);
    gpu_cmd_bind_vertex_buffer(viewer->gpu, viewer->vertex_buffer);
    gpu_cmd_bind_index_buffer(viewer->gpu, viewer->index_buffer);

    for (int m = 0; m < viewer->scene->nmeshes; m++) {
        SceneMesh* sm = &viewer->scene->meshes[m];
        if (!sm->visible || sm->is_proto_only || sm->nindices == 0)
            continue;
        gpu_cmd_bind_mesh_data(viewer->gpu, m);
        gpu_cmd_draw_indexed(viewer->gpu, (uint32_t)sm->nindices,
                             sm->index_offset, 0);
    }

    gpu_shadow_end(viewer->gpu);
}

/* Render one frame using caller-supplied camera matrices, then read the
 * default-framebuffer color buffer back into out_rgba (top-left origin,
 * 4 bytes per pixel, no alpha strip). */
int viewer_render_to_rgba(Viewer* viewer, int width, int height,
                          const float view16[16], const float proj16[16],
                          const float eye3[3], unsigned char* out_rgba)
{
    if (!viewer || !view16 || !proj16 || !eye3 || !out_rgba) return 0;
    if (width <= 0 || height <= 0) return 0;

    /* Per-frame phase timing — gated by NUSD_DEBUG_FRAME_PERF=1. */
    static int g_perf_frame = 0;
    static int g_perf_enabled = -1;
    if (g_perf_enabled < 0) {
        const char* e = getenv("NUSD_DEBUG_FRAME_PERF");
        g_perf_enabled = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    double t_p0 = g_perf_enabled ? get_time_seconds() : 0.0;
    double t_p1 = 0, t_p2 = 0, t_p3 = 0, t_p4 = 0, t_p5 = 0, t_p6 = 0, t_p7 = 0, t_p8 = 0;

    /* When running embedded in another GL host (omni.ui's GLFW window),
     * the host's GL context is current on this thread. Make our headless
     * context current for the render and release it before returning so
     * the host can re-issue its own GL calls afterwards. */
    if (viewer->egl_ctx) {
        if (!egl_headless_make_current(viewer->egl_ctx)) return 0;
    }

    /* Resize the framebuffer/pbuffer if the embedder requested a new size. */
    viewer_resize(viewer, width, height);
    if (g_perf_enabled) t_p1 = get_time_seconds();

    /* HUD overlay disabled for now — the in-render text/rect rendering
     * doesn't look good against ovgear's image bridge. Code is
     * preserved (`queue_overlay` + `hud_tick_stats` below) for when we
     * revisit the overlay style. To re-enable: uncomment these two
     * calls and ensure the embedder didn't disable it via
     * viewer_set_overlay_enabled(0). */
    /* hud_tick_stats(viewer); */
    /* queue_overlay(viewer); */

    if (!gpu_begin_frame(viewer->gpu)) {
        if (viewer->egl_ctx)
            egl_headless_release_current(viewer->egl_ctx);
        return 0;
    }
    if (g_perf_enabled) t_p2 = get_time_seconds();

    float view_inv[16], proj_inv[16];
    if (mat4_inverse(view16, view_inv) && mat4_inverse(proj16, proj_inv))
        gpu_draw_env_background(viewer->gpu, view_inv, proj_inv);
    if (g_perf_enabled) t_p3 = get_time_seconds();

    float vp[16];
    mat4_mul(proj16, view16, vp);

    /* Hoist UBO base + IBL binds + sampler-unit assignments out of the
     * per-mesh loop. Saves ~12 GL calls × 1162 meshes per frame. */
    viewer_bind_main_mesh_pipeline(viewer);

    /* Frustum culling (Plan C). Extract 6 planes from VP using the
     * standard column-major derivation and reject meshes whose AABB is
     * fully outside any plane. Works in world space — scene.c already
     * sets per-mesh bounds_min/bounds_max in world coords. */
    int n_total = viewer->scene ? viewer->scene->nmeshes : 0;
    int* visible = (int*)alloca((size_t)(n_total > 0 ? n_total : 1) * sizeof(int));
    int* mesh_straddle = (int*)alloca((size_t)(n_total > 0 ? n_total : 1) * sizeof(int));
    int n_visible = 0;
    float ml_nplanes[6][4];      /* unit-normal frustum planes — meshlet cull */
    float frustum_planes[6][4];  /* unnormalized world-space planes — AABB cull */
    {
        float planes[6][4];
        /* vp is row-major (vp[row*4 + col]) after the convention fix —
         * row R is the contiguous 4-float run starting at vp[R*4]. */
        #define VP_ROW(R) {vp[(R)*4+0], vp[(R)*4+1], vp[(R)*4+2], vp[(R)*4+3]}
        float r0[4] = VP_ROW(0), r1[4] = VP_ROW(1), r2[4] = VP_ROW(2), r3[4] = VP_ROW(3);
        #undef VP_ROW
        for (int i = 0; i < 4; i++) {
            planes[0][i] = r3[i] + r0[i];  /* left   */
            planes[1][i] = r3[i] - r0[i];  /* right  */
            planes[2][i] = r3[i] + r1[i];  /* bottom */
            planes[3][i] = r3[i] - r1[i];  /* top    */
            planes[4][i] = r3[i] + r2[i];  /* near   */
            planes[5][i] = r3[i] - r2[i];  /* far    */
        }
        /* Unit-normal copies for the per-meshlet bounding-sphere test. */
        for (int p = 0; p < 6; p++) {
            float a=planes[p][0], b=planes[p][1], c=planes[p][2];
            float l = sqrtf(a*a + b*b + c*c);
            if (l < 1e-20f) l = 1.0f;
            ml_nplanes[p][0]=a/l; ml_nplanes[p][1]=b/l;
            ml_nplanes[p][2]=c/l; ml_nplanes[p][3]=planes[p][3]/l;
        }
        memcpy(frustum_planes, planes, sizeof(frustum_planes));
        for (int oi = 0; oi < n_total; oi++) {
            int m = viewer->draw_order ? viewer->draw_order[oi] : oi;
            SceneMesh* sm = &viewer->scene->meshes[m];
            if (!sm->visible) continue;
            if (sm->nindices == 0) continue;
            if (sm->is_proto_only) continue;
            /* Classify the mesh AABB: fully-outside (cull) / straddle / inside.
             * Per plane, the p-vertex behind it means fully outside; the
             * n-vertex behind it (p-vertex in front) means straddling. */
            int outside = 0, straddle = 0;
            for (int p = 0; p < 6; p++) {
                float a = planes[p][0], b = planes[p][1], c = planes[p][2], d = planes[p][3];
                float px = a > 0 ? sm->bounds_max[0] : sm->bounds_min[0];
                float py = b > 0 ? sm->bounds_max[1] : sm->bounds_min[1];
                float pz = c > 0 ? sm->bounds_max[2] : sm->bounds_min[2];
                if (a*px + b*py + c*pz + d < 0) { outside = 1; break; }
                float nx = a > 0 ? sm->bounds_min[0] : sm->bounds_max[0];
                float ny = b > 0 ? sm->bounds_min[1] : sm->bounds_max[1];
                float nz = c > 0 ? sm->bounds_min[2] : sm->bounds_max[2];
                if (a*nx + b*ny + c*nz + d < 0) straddle = 1;
            }
            if (!outside) {
                mesh_straddle[n_visible] = straddle;
                visible[n_visible++] = m;
            }
        }
    }
    if (g_perf_enabled) t_p4 = get_time_seconds();

    float shadow_vp[GPU_MAX_SHADOW_LIGHTS][16];
    int shadow_light_indices[GPU_MAX_SHADOW_LIGHTS];
    int shadow_count = viewer_collect_shadow_lights(viewer, shadow_vp,
                                                    shadow_light_indices);
    int shadow_enabled = shadow_count > 0;

    /* Plan A: write each visible mesh's mvp/model/color into the mesh
     * UBO via one big mapped write. INVALIDATE_BUFFER orphans last
     * frame's storage so the GPU keeps reading it while we write the
     * fresh frame — no stalls, no per-mesh glBufferSubData. */
    void* mesh_map = NULL;
    int has_curves = viewer->scene && viewer->scene->ncurves > 0;
    if (n_visible > 0 || shadow_enabled || has_curves || viewer->curves_smoketest)
        mesh_map = gpu_begin_mesh_writes(viewer->gpu);
    int mesh_stride = gpu_mesh_stride(viewer->gpu);
    if (mesh_map && mesh_stride > 0 && viewer->scene) {
        int write_count = shadow_enabled ? n_total : n_visible;
        for (int i = 0; i < write_count; i++) {
            int m = shadow_enabled ? i : visible[i];
            SceneMesh* sm = &viewer->scene->meshes[m];
            if (!sm->visible || sm->nindices == 0) continue;
            if (sm->is_proto_only) continue;
            GpuMeshData* md = (GpuMeshData*)((unsigned char*)mesh_map +
                                              (size_t)m * (size_t)mesh_stride);
            /* nanousd's world_xform is "USD row-major" — flat array
             * with translation at [12]/[13]/[14] (matrix-vector on the
             * right convention). For our row_major UBO + column-vector
             * shader math, we need translation at [3]/[7]/[11], so
             * transpose. mat4_mul also expects row-major operands;
             * after transpose vp/world both feed mat4_mul as row-major
             * and the result is correct. */
            float world_f[16], world_row[16];
            mat4_d2f(sm->world_xform, world_f);
            mat4_transpose(world_f, world_row);
            memcpy(md->model, world_row, sizeof(md->model));
            mat4_mul(vp, world_row, md->mvp);
            if (sm->has_display_color) {
                md->color[0] = sm->display_color[0];
                md->color[1] = sm->display_color[1];
                md->color[2] = sm->display_color[2];
                md->color[3] = 1.0f;
            } else {
                md->color[0] = 0.7f; md->color[1] = 0.7f;
                md->color[2] = 0.7f; md->color[3] = 0.0f;  /* w<=0.5 → use vertex color */
            }
            md->ptex[0] = viewer->ptex_color_buffer_uploaded
                ? sm->ptex_color_offset : VIEWER_PTEX_COLOR_NONE;
            md->ptex[1] = md->ptex[2] = md->ptex[3] = 0;
        }
    }
    if (mesh_map && mesh_stride > 0 && viewer->scene && viewer->scene->ncurves > 0) {
        /* Per-curve UBO write: model = transposed world_xform (matches
         * the row_major UBO layout, translation at [3]/[7]/[11]).
         * color = displayColor when authored. mvp is unused by the
         * curve TES (Storm's u_view + u_proj separate-matrix path). */
        for (int c = 0; c < viewer->scene->ncurves; c++) {
            SceneCurve* cu = &viewer->scene->curves[c];
            int slot = viewer->curve_ubo_slot + c;
            GpuMeshData* md = (GpuMeshData*)((unsigned char*)mesh_map +
                                              (size_t)slot * (size_t)mesh_stride);
            float world_f[16], world_row[16];
            mat4_d2f(cu->world_xform, world_f);
            mat4_transpose(world_f, world_row);
            memcpy(md->model, world_row, sizeof(md->model));
            memset(md->mvp, 0, sizeof(md->mvp));  /* unused */
            if (cu->has_display_color) {
                md->color[0] = cu->display_color[0];
                md->color[1] = cu->display_color[1];
                md->color[2] = cu->display_color[2];
                md->color[3] = 1.0f;
            } else {
                md->color[0] = 0.7f; md->color[1] = 0.7f;
                md->color[2] = 0.7f; md->color[3] = 0.0f;
            }
            md->ptex[0] = VIEWER_PTEX_COLOR_NONE;
            md->ptex[1] = md->ptex[2] = md->ptex[3] = 0;
        }
    } else if (mesh_map && mesh_stride > 0 && viewer->curves_smoketest && viewer->scene) {
        /* Smoke-test fallback: identity model + magenta override on the
         * one extra slot. Used when the scene has zero curve prims. */
        float identity[16] = {
            1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1,
        };
        GpuMeshData* md = (GpuMeshData*)((unsigned char*)mesh_map +
                                          (size_t)viewer->curve_ubo_slot *
                                          (size_t)mesh_stride);
        memcpy(md->model, identity, sizeof(md->model));
        memset(md->mvp, 0, sizeof(md->mvp));
        md->color[0] = 1.0f; md->color[1] = 0.0f;
        md->color[2] = 1.0f; md->color[3] = 1.0f;
        md->ptex[0] = VIEWER_PTEX_COLOR_NONE;
        md->ptex[1] = md->ptex[2] = md->ptex[3] = 0;
    }
    /* PI batch UBO slots: mvp = VP only (the instanced shader composes
     * instModel * mvp); color/ptex inherited from each batch's prototype. */
    if (mesh_map && mesh_stride > 0 && viewer->scene &&
        viewer->scene->npi_batches > 0 && viewer->instance_buffer_ready) {
        for (int b = 0; b < viewer->scene->npi_batches; b++) {
            const SceneInstanceBatch* pb = &viewer->scene->pi_batches[b];
            int p = pb->prototype_mesh_idx;
            if (p < 0 || p >= viewer->scene->nmeshes) continue;
            SceneMesh* proto = &viewer->scene->meshes[p];
            int slot = viewer->batch_ubo_slot + b;
            GpuMeshData* md = (GpuMeshData*)((unsigned char*)mesh_map +
                                              (size_t)slot * (size_t)mesh_stride);
            memcpy(md->mvp, vp, sizeof(md->mvp));      /* VP only */
            memset(md->model, 0, sizeof(md->model));
            if (proto->has_display_color) {
                md->color[0] = proto->display_color[0];
                md->color[1] = proto->display_color[1];
                md->color[2] = proto->display_color[2];
                md->color[3] = 1.0f;
            } else {
                md->color[0] = 0.7f; md->color[1] = 0.7f;
                md->color[2] = 0.7f; md->color[3] = 0.0f;
            }
            md->ptex[0] = viewer->ptex_color_buffer_uploaded
                ? proto->ptex_color_offset : VIEWER_PTEX_COLOR_NONE;
            md->ptex[1] = md->ptex[2] = md->ptex[3] = 0;
        }
    }
    if (mesh_map)
        gpu_end_mesh_writes(viewer->gpu);

    if (shadow_enabled) {
        for (int s = 0; s < shadow_count; s++)
            viewer_render_shadow_map(viewer, s, shadow_vp[s],
                                     shadow_light_indices[s]);
        viewer_bind_main_mesh_pipeline(viewer);
    }

    /* Eye position is per-frame, not per-mesh — set once. */
    gpu_cmd_set_eye_pos(viewer->gpu, eye3);

    int n_drawn = 0, n_mat_binds = 0;
    long long n_meshlet_drawn = 0, n_meshlet_culled = 0, n_meshlet_draws = 0;
    int bound_raster_chunk = -1;
    double t_mat_total = 0.0, t_pc_total = 0.0, t_draw_total = 0.0;
    if (viewer->scene) {
        for (int i = 0; i < n_visible; i++) {
            int m = visible[i];
            SceneMesh* sm = &viewer->scene->meshes[m];
            if (sm->is_proto_only) continue;

            double t_inner_a = g_perf_enabled ? get_time_seconds() : 0.0;
            if (viewer->render_mode == MODE_MATERIAL) {
                int material_index = sm->material_index;
                if (material_index < 0)
                    material_index = viewer->fallback_material_index;
                if (material_index >= 0) {
                    gpu_cmd_bind_material(viewer->gpu, material_index);
                    n_mat_binds++;
                }
            }
            double t_inner_b = g_perf_enabled ? get_time_seconds() : 0.0;
            if (g_perf_enabled) t_mat_total += (t_inner_b - t_inner_a);

            double t_inner_c = g_perf_enabled ? get_time_seconds() : 0.0;
            if (!(viewer->render_mode == MODE_MATERIAL && viewer->mat_pipeline) &&
                viewer->buffer_count > 0) {
                int chunk = (int)sm->buffer_index;
                if (chunk >= 0 && chunk < viewer->buffer_count &&
                    chunk != bound_raster_chunk) {
                    gpu_cmd_bind_vertex_buffer(viewer->gpu, viewer->vertex_buffers[chunk]);
                    /* Meshlet path binds one shared meshlet IBO; otherwise
                     * the chunk's own index buffer. */
                    gpu_cmd_bind_index_buffer(viewer->gpu,
                        viewer->meshlet_draw ? viewer->meshlet_index_buffer
                                             : viewer->index_buffers[chunk]);
                    bound_raster_chunk = chunk;
                }
            }
            gpu_cmd_bind_mesh_data(viewer->gpu, m);
            double t_inner_d = g_perf_enabled ? get_time_seconds() : 0.0;
            if (viewer->meshlet_draw && sm->meshlet_count > 0) {
                /* Meshlet draw path. A fully-inside mesh draws its whole
                 * meshlet range in one call; a straddling mesh culls each
                 * meshlet (world-space bounding sphere vs the frustum) and
                 * draws only the survivors. The meshlet IBO holds mesh-local
                 * indices, so vertex_offset is the GL base vertex. */
                uint32_t mo = sm->meshlet_offset, mc = sm->meshlet_count;
                if (!mesh_straddle[i]) {
                    uint32_t base = viewer->meshlet_ibo_off[mo];
                    uint32_t cnt  = viewer->meshlet_ibo_off[mo + mc] - base;
                    gpu_cmd_draw_indexed_typed(viewer->gpu, cnt,
                                               (uint64_t)base * 4u,
                                               (int32_t)sm->vertex_offset, 32);
                    n_meshlet_drawn += mc;
                    n_meshlet_draws++;
                } else {
                    /* Straddle — cull each meshlet, then draw maximal runs of
                     * consecutive survivors. Consecutive meshlet records are
                     * contiguous in the meshlet IBO, so one run is one index
                     * range — one draw call instead of one per meshlet. */
                    float msc = ml_xform_max_scale(sm->world_xform);
                    uint32_t run_start = 0;
                    int run_open = 0;
                    for (uint32_t k = 0; k < mc; k++) {
                        const ViewerMeshlet* mr = &viewer->meshlet_recs[mo + k];
                        float wc[3];
                        ml_xform_point(sm->world_xform, mr->center, wc);
                        float wr = mr->radius * msc;
                        int cull = 0;
                        for (int p = 0; p < 6; p++) {
                            float dd = ml_nplanes[p][0]*wc[0] + ml_nplanes[p][1]*wc[1]
                                     + ml_nplanes[p][2]*wc[2] + ml_nplanes[p][3];
                            if (dd < -wr) { cull = 1; break; }
                        }
                        if (!cull) {
                            if (!run_open) { run_start = k; run_open = 1; }
                            continue;
                        }
                        n_meshlet_culled++;
                        if (run_open) {
                            uint32_t base = viewer->meshlet_ibo_off[mo + run_start];
                            uint32_t cnt  = viewer->meshlet_ibo_off[mo + k] - base;
                            gpu_cmd_draw_indexed_typed(viewer->gpu, cnt,
                                                       (uint64_t)base * 4u,
                                                       (int32_t)sm->vertex_offset, 32);
                            n_meshlet_drawn += (k - run_start);
                            n_meshlet_draws++;
                            run_open = 0;
                        }
                    }
                    if (run_open) {
                        uint32_t base = viewer->meshlet_ibo_off[mo + run_start];
                        uint32_t cnt  = viewer->meshlet_ibo_off[mo + mc] - base;
                        gpu_cmd_draw_indexed_typed(viewer->gpu, cnt,
                                                   (uint64_t)base * 4u,
                                                   (int32_t)sm->vertex_offset, 32);
                        n_meshlet_drawn += (mc - run_start);
                        n_meshlet_draws++;
                    }
                }
            } else if (!(viewer->render_mode == MODE_MATERIAL && viewer->mat_pipeline) &&
                viewer->buffer_count > 0) {
                if (sm->index_type_bits == 16) {
                    gpu_cmd_draw_indexed_typed(viewer->gpu, (uint32_t)sm->nindices,
                                               sm->index_byte_offset,
                                               (int32_t)sm->vertex_offset,
                                               sm->index_type_bits);
                } else {
                    gpu_cmd_draw_indexed(viewer->gpu, (uint32_t)sm->nindices,
                                         (uint32_t)(sm->index_byte_offset / 4u), 0);
                }
            } else {
                gpu_cmd_draw_indexed(viewer->gpu, (uint32_t)sm->nindices,
                                     sm->index_offset, 0);
            }
            double t_inner_e = g_perf_enabled ? get_time_seconds() : 0.0;
            if (g_perf_enabled) {
                t_pc_total += (t_inner_d - t_inner_c);
                t_draw_total += (t_inner_e - t_inner_d);
            }
            n_drawn++;
        }
    }

    /* Compact-PointInstancer batches — draw each prototype once per placement
     * via GLES instancing (the merged mat VBO/IB supplies geometry; the
     * per-instance world matrix comes from the instance VBO). Mirrors the RT/
     * Vulkan compact-PI consumption so Moana's scattered foliage renders.
     * Opt out with NUSD_GL_DISABLE_PI=1. */
    if (viewer->render_mode == MODE_MATERIAL && viewer->mat_instanced_pipeline &&
        viewer->instance_buffer_ready && viewer->scene &&
        viewer->scene->npi_batches > 0 && viewer->mat_vertex_buffer &&
        viewer->index_buffer && getenv("NUSD_GL_DISABLE_PI") == NULL) {
        gpu_cmd_bind_pipeline(viewer->gpu, viewer->mat_instanced_pipeline);
        gpu_cmd_bind_vertex_buffer(viewer->gpu, viewer->mat_vertex_buffer);
        gpu_cmd_bind_index_buffer(viewer->gpu, viewer->index_buffer);
        gpu_cmd_begin_material_pass(viewer->gpu);
        gpu_cmd_set_eye_pos(viewer->gpu, eye3);
        for (int b = 0; b < viewer->scene->npi_batches; b++) {
            const SceneInstanceBatch* pb = &viewer->scene->pi_batches[b];
            int p = pb->prototype_mesh_idx;
            if (p < 0 || p >= viewer->scene->nmeshes) continue;
            SceneMesh* proto = &viewer->scene->meshes[p];
            if (proto->nindices <= 0 || pb->transform_count == 0) continue;
            int material_index = proto->material_index;
            if (material_index < 0) material_index = viewer->fallback_material_index;
            if (material_index >= 0)
                gpu_cmd_bind_material(viewer->gpu, material_index);
            gpu_cmd_bind_mesh_data(viewer->gpu, viewer->batch_ubo_slot + b);
            gpu_cmd_draw_instanced(viewer->gpu, (uint32_t)proto->nindices,
                                   proto->index_offset, pb->transform_count,
                                   pb->transform_offset);
            n_drawn++;
        }
    }

    if (viewer->meshlet_draw)
        fprintf(stderr, "[meshlet] frame: %lld meshlets drawn in %lld draw calls, "
                "%lld culled\n", n_meshlet_drawn, n_meshlet_draws, n_meshlet_culled);

    /* BasisCurves draw: one bind per pipeline, then per-curve bind UBO
     * slot + draw_indexed. The TES uses Storm's separate-matrix pattern
     * (u_view + u_proj) so we set those plain uniforms once after
     * binding the pipeline. */
    if (viewer->curve_pipeline) {
        gpu_cmd_bind_pipeline(viewer->gpu, viewer->curve_pipeline);
        gpu_cmd_bind_vertex_buffer(viewer->gpu, viewer->curve_vertex_buffer);
        gpu_cmd_bind_index_buffer(viewer->gpu, viewer->curve_index_buffer);
        gpu_cmd_set_eye_pos(viewer->gpu, eye3);
        gpu_cmd_set_view_proj(viewer->gpu, view16, proj16);

        if (viewer->scene && viewer->scene->ncurves > 0) {
            const char* cull_env = getenv("NUSD_GL_CURVE_CULL");
            int cull_curves = !(cull_env && cull_env[0] == '0');
            if (viewer->curve_draws && viewer->curve_draw_count > 0) {
                for (int d = 0; d < viewer->curve_draw_count; ) {
                    ViewerCurveDraw* draw = &viewer->curve_draws[d];
                    if (draw->npatches == 0 ||
                        draw->curve_id >= (uint32_t)viewer->scene->ncurves) {
                        d++;
                        continue;
                    }
                    if (cull_curves &&
                        aabb_outside_planes((const float (*)[4])frustum_planes,
                                            draw->bounds_min, draw->bounds_max)) {
                        d++;
                        continue;
                    }

                    uint32_t run_patches = draw->npatches;
                    int next = d + 1;
                    uint64_t bytes_per_index =
                        (draw->index_type_bits == 16) ? 2u : 4u;
                    while (next < viewer->curve_draw_count) {
                        ViewerCurveDraw* n = &viewer->curve_draws[next];
                        if (n->curve_id != draw->curve_id) break;
                        if (n->index_type_bits != draw->index_type_bits) break;
                        if (n->vertex_offset != draw->vertex_offset) break;
                        if (n->index_byte_offset !=
                            draw->index_byte_offset +
                            (uint64_t)run_patches * 4u * bytes_per_index) {
                            break;
                        }
                        if (cull_curves &&
                            aabb_outside_planes((const float (*)[4])frustum_planes,
                                                n->bounds_min, n->bounds_max)) {
                            break;
                        }
                        run_patches += n->npatches;
                        next++;
                    }

                    SceneCurve* cu = &viewer->scene->curves[draw->curve_id];
                    int bid = cu->type_is_cubic ? (int)cu->basis : 3;
                    gpu_cmd_set_basis_id(viewer->gpu, bid);
                    gpu_cmd_bind_mesh_data(viewer->gpu,
                                           viewer->curve_ubo_slot + (int)draw->curve_id);
                    gpu_cmd_draw_indexed_typed(viewer->gpu,
                                               run_patches * 4u,
                                               draw->index_byte_offset,
                                               draw->vertex_offset,
                                               draw->index_type_bits);
                    d = next;
                }
            } else {
                for (int c = 0; c < viewer->scene->ncurves; c++) {
                    SceneCurve* cu = &viewer->scene->curves[c];
                    if (cu->npatches == 0) continue;
                    if (cull_curves &&
                        aabb_outside_planes((const float (*)[4])frustum_planes,
                                            cu->bounds_min, cu->bounds_max))
                        continue;
                    /* Per-curve basis selector: linear curves (type=linear)
                     * always uses the linear-basis branch regardless of the
                     * authored `basis` token; cubic uses the authored basis. */
                    int bid = cu->type_is_cubic ? (int)cu->basis : 3;
                    gpu_cmd_set_basis_id(viewer->gpu, bid);
                    gpu_cmd_bind_mesh_data(viewer->gpu, viewer->curve_ubo_slot + c);
                    gpu_cmd_draw_indexed(viewer->gpu,
                                         (uint32_t)cu->npatches * 4,
                                         cu->index_offset, 0);
                }
            }
        } else if (viewer->curves_smoketest) {
            gpu_cmd_set_basis_id(viewer->gpu, 0);  /* bezier */
            gpu_cmd_bind_mesh_data(viewer->gpu, viewer->curve_ubo_slot);
            gpu_cmd_draw_indexed(viewer->gpu, 4, 0, 0);
        }
    }

    if (g_perf_enabled) t_p5 = get_time_seconds();

    gpu_end_frame(viewer->gpu);
    if (g_perf_enabled) t_p6 = get_time_seconds();

    /* Read pixels (GLES guarantees GL_RGBA + UNSIGNED_BYTE on default FBO). */
    extern void glFinish(void);
    extern void glReadPixels(int, int, int, int, unsigned int, unsigned int, void*);
    /* Use the headers' GL_* constants — the values are stable but pull
     * the official ones in via the GLES header included by gpu_opengles.c.
     * Calling through gpu_screenshot would write to a file; we have a
     * direct readback path below using the same parameters as that
     * function. */
    {
        /* GLES2 enums: GL_RGBA=0x1908, GL_UNSIGNED_BYTE=0x1401 */
        const unsigned int GL_RGBA_E = 0x1908;
        const unsigned int GL_UNSIGNED_BYTE_E = 0x1401;
        unsigned char* tmp = (unsigned char*)malloc((size_t)width * (size_t)height * 4);
        if (!tmp) return 0;
        glFinish();
        if (g_perf_enabled) t_p7 = get_time_seconds();
        glReadPixels(0, 0, width, height, GL_RGBA_E, GL_UNSIGNED_BYTE_E, tmp);
        /* glReadPixels returns bottom-up; flip into top-down out_rgba. */
        size_t row = (size_t)width * 4;
        for (int y = 0; y < height; y++) {
            memcpy(out_rgba + (size_t)y * row,
                   tmp + (size_t)(height - 1 - y) * row,
                   row);
        }
        free(tmp);
        if (g_perf_enabled) t_p8 = get_time_seconds();
    }

    if (viewer->egl_ctx)
        egl_headless_release_current(viewer->egl_ctx);

    extern int g_dbg_mat_calls, g_dbg_mat_cache_hits, g_dbg_mat_tex_binds;
    if (g_perf_enabled && (g_perf_frame % 30) == 0) {
        double draw_loop = (t_p5 - t_p4) * 1000.0;
        fprintf(stderr,
                "frame[%4d] %dx%d draws=%d mats=%d  "
                "ctx=%.2f clear=%.2f env=%.2f bind=%.2f "
                "draw_loop=%.2f (mat=%.2f pc=%.2f drawN=%.2f) "
                "end=%.2f finish=%.2f read=%.2f  "
                "matcalls=%d cache_hit=%d texbinds=%d  total=%.2f ms\n",
                g_perf_frame, width, height, n_drawn, n_mat_binds,
                (t_p1 - t_p0) * 1000.0,
                (t_p2 - t_p1) * 1000.0,
                (t_p3 - t_p2) * 1000.0,
                (t_p4 - t_p3) * 1000.0,
                draw_loop,
                t_mat_total * 1000.0,
                t_pc_total * 1000.0,
                t_draw_total * 1000.0,
                (t_p6 - t_p5) * 1000.0,
                (t_p7 - t_p6) * 1000.0,
                (t_p8 - t_p7) * 1000.0,
                g_dbg_mat_calls, g_dbg_mat_cache_hits, g_dbg_mat_tex_binds,
                (t_p8 - t_p0) * 1000.0);
    }
    g_dbg_mat_calls = g_dbg_mat_cache_hits = g_dbg_mat_tex_binds = 0;
    g_perf_frame++;
    return 1;
}

void viewer_destroy(Viewer* viewer)
{
    if (!viewer) return;

    gpu_overlay_shutdown(viewer->gpu);

    if (viewer->mat_pipeline)
        gpu_destroy_pipeline(viewer->gpu, viewer->mat_pipeline);
    if (viewer->shadow_pipeline)
        gpu_destroy_pipeline(viewer->gpu, viewer->shadow_pipeline);
    if (viewer->mat_vertex_buffer)
        gpu_destroy_buffer(viewer->gpu, viewer->mat_vertex_buffer);
    gpu_destroy_materials(viewer->gpu);
    gpu_destroy_environment(viewer->gpu);

    if (viewer->pipeline)
        gpu_destroy_pipeline(viewer->gpu, viewer->pipeline);

    if (viewer->buffer_count > 0) {
        for (int i = 0; i < viewer->buffer_count; i++) {
            if (viewer->vertex_buffers && viewer->vertex_buffers[i])
                gpu_destroy_buffer(viewer->gpu, viewer->vertex_buffers[i]);
            if (viewer->index_buffers && viewer->index_buffers[i])
                gpu_destroy_buffer(viewer->gpu, viewer->index_buffers[i]);
        }
        free(viewer->vertex_buffers);
        free(viewer->index_buffers);
        viewer->vertex_buffer = NULL;
        viewer->index_buffer = NULL;
    } else {
        if (viewer->vertex_buffer)
            gpu_destroy_buffer(viewer->gpu, viewer->vertex_buffer);

        if (viewer->index_buffer)
            gpu_destroy_buffer(viewer->gpu, viewer->index_buffer);
    }

    if (viewer->meshlet_index_buffer)
        gpu_destroy_buffer(viewer->gpu, viewer->meshlet_index_buffer);
    free(viewer->meshlet_ibo_off);
    free(viewer->meshlet_recs);

    if (viewer->curve_pipeline)
        gpu_destroy_pipeline(viewer->gpu, viewer->curve_pipeline);
    if (viewer->curve_vertex_buffer)
        gpu_destroy_buffer(viewer->gpu, viewer->curve_vertex_buffer);
    if (viewer->curve_index_buffer)
        gpu_destroy_buffer(viewer->gpu, viewer->curve_index_buffer);
    free(viewer->curve_draws);

    if (viewer->gpu)
        gpu_shutdown(viewer->gpu);

    if (viewer->materials)
        materials_free(viewer->materials);

    free(viewer->draw_order);

    if (viewer->scene)
        scene_free(viewer->scene);

    /* Only destroy the GL context if we own it. With egl_ctx == NULL
     * the context belongs to the standalone GLFW window (which main()
     * tears down via viewer_window_teardown) or to the embed host. */
    if (viewer->egl_ctx)
        egl_headless_destroy(viewer->egl_ctx);

    free(viewer);
}
