// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "ptex_material.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef NUSD_HAVE_PTEX
#include <Ptexture.h>
#endif

static int ptex_enabled(void)
{
    const char* s = std::getenv("NUSD_ENABLE_PTEX_MATERIALS");
    return !(s && s[0] == '0');
}

static float ptex_color_gamma(void)
{
    const char* s = std::getenv("NUSD_PTEX_COLOR_GAMMA");
    if (s && s[0]) {
        char* end = nullptr;
        float v = std::strtof(s, &end);
        if (end != s && std::isfinite(v) && v > 0.0f && v <= 4.0f)
            return v;
    }
    /* Moana's Ptex surface graph feeds PxrPtexture through PxrColorCorrect
     * gamma=(0.454545...). RenderMan-style gamma correction uses the inverse
     * exponent here, which linearizes the authored color textures instead of
     * brightening them into a washed-out clay range. */
    return 2.2f;
}

static float ptex_triangle_center_blend(void)
{
    const char* s = std::getenv("NUSD_PTEX_TRI_CENTER_BLEND");
    if (s && s[0]) {
        char* end = nullptr;
        float v = std::strtof(s, &end);
        if (end != s && std::isfinite(v))
            return std::clamp(v, 0.0f, 0.45f);
    }
    return 0.22f;
}

static void apply_ptex_color_transform(float rgb[3])
{
    float gamma = ptex_color_gamma();
    if (std::fabs(gamma - 1.0f) < 1e-6f)
        return;
    for (int c = 0; c < 3; ++c)
        rgb[c] = std::pow(std::max(rgb[c], 0.0f), gamma);
}

static uint32_t pack_rgba8(float r, float g, float b)
{
    r = std::clamp(r, 0.0f, 1.0f);
    g = std::clamp(g, 0.0f, 1.0f);
    b = std::clamp(b, 0.0f, 1.0f);
    uint32_t ri = (uint32_t)(r * 255.0f + 0.5f);
    uint32_t gi = (uint32_t)(g * 255.0f + 0.5f);
    uint32_t bi = (uint32_t)(b * 255.0f + 0.5f);
    return ri | (gi << 8) | (bi << 16) | (0xFFu << 24);
}

#ifdef NUSD_HAVE_PTEX
static void sample_pixel(Ptex::PtexTexture* tx, int face_id,
                         float u, float v, float out[3])
{
    const Ptex::FaceInfo& fi = tx->getFaceInfo(face_id);
    int uw = std::max(fi.res.u(), 1);
    int vh = std::max(fi.res.v(), 1);
    float pix[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    int nch = std::min(tx->numChannels(), 3);
    if (nch > 0) {
        Ptex::PtexFilter::Options opts(Ptex::PtexFilter::f_bilinear,
                                       true, 0.0f, true);
        float fw_u = 1.0f / (float)uw;
        float fw_v = 1.0f / (float)vh;
        Ptex::PtexFilter::eval(tx, opts, pix, 0, nch, face_id,
                               std::clamp(u, 0.0f, 1.0f),
                               std::clamp(v, 0.0f, 1.0f),
                               fw_u, 0.0f, 0.0f, fw_v);
    }
    out[0] = pix[0];
    out[1] = (nch >= 2) ? pix[1] : pix[0];
    out[2] = (nch >= 3) ? pix[2] : pix[0];
    apply_ptex_color_transform(out);
}
#endif

static void quad_corner_uv(int corner, float* u, float* v)
{
    const float e = 0.12f;
    switch (corner & 3) {
    case 0: *u = e;        *v = e;        break;
    case 1: *u = 1.0f - e; *v = e;        break;
    case 2: *u = 1.0f - e; *v = 1.0f - e; break;
    default:*u = e;        *v = 1.0f - e; break;
    }
}

static void tri_corner_uv(int corner, float* u, float* v)
{
    const float e = 0.12f;
    switch (corner % 3) {
    case 0: *u = e;              *v = e;              break;
    case 1: *u = 1.0f - 2.0f*e;  *v = e;              break;
    default:*u = e;              *v = 1.0f - 2.0f*e;  break;
    }
}

static void ngon_corner_uv(int vertex, int vertex_count, float* u, float* v)
{
    if (vertex_count <= 0) {
        *u = *v = 0.5f;
        return;
    }
    float angle = ((float)vertex / (float)vertex_count) * 6.28318530718f;
    *u = 0.5f + 0.43f * std::cos(angle);
    *v = 0.5f + 0.43f * std::sin(angle);
}

#ifdef NUSD_HAVE_PTEX
static void ptex_face_corner_uv(Ptex::MeshType mesh_type, int vertex_count,
                                int corner, float* u, float* v)
{
    if (vertex_count == 4) {
        quad_corner_uv(corner, u, v);
    } else if (mesh_type == Ptex::mt_triangle && vertex_count == 3) {
        tri_corner_uv(corner, u, v);
    } else {
        ngon_corner_uv(corner, vertex_count, u, v);
    }
}
#endif

int nusd_ptex_sample_face_tri_colors(const char* path,
                                     const int* face_counts,
                                     int face_count,
                                     int face_vertex_index_count,
                                     uint32_t* out_rgba8,
                                     int out_color_count)
{
    if (!path || !path[0] || !out_rgba8 || out_color_count <= 0)
        return 0;
    if (!ptex_enabled())
        return 0;

#ifndef NUSD_HAVE_PTEX
    (void)face_counts;
    (void)face_count;
    (void)face_vertex_index_count;
    return 0;
#else
    Ptex::String error;
    Ptex::PtexTexture* tx = Ptex::PtexTexture::open(path, error, true);
    if (!tx) {
        static int warned = 0;
        if (warned < 16) {
            std::fprintf(stderr, "ptex: failed to open %s: %s\n",
                         path, error.c_str());
            warned++;
        }
        return 0;
    }
    if (tx->numFaces() <= 0 || tx->numChannels() <= 0) {
        tx->release();
        return 0;
    }

    int written = 0;
    int consumed_fvi = 0;
    if (face_counts && face_count > 0) {
        int ptex_faces = tx->numFaces();
        for (int f = 0; f < face_count && written < out_color_count; ++f) {
            int n = face_counts[f];
            if (n < 3) {
                consumed_fvi += (n > 0) ? n : 0;
                continue;
            }
            int face_id = (f < ptex_faces) ? f : (ptex_faces - 1);
            int ntri = n - 2;
            float center_blend = ptex_triangle_center_blend();
            for (int t = 0; t < ntri && written + 2 < out_color_count; ++t) {
                int corners[3] = {0, t + 1, t + 2};
                float uv[3][2];
                for (int k = 0; k < 3; ++k) {
                    ptex_face_corner_uv(tx->meshType(), n, corners[k],
                                        &uv[k][0], &uv[k][1]);
                }
                float cu = (uv[0][0] + uv[1][0] + uv[2][0]) / 3.0f;
                float cv = (uv[0][1] + uv[1][1] + uv[2][1]) / 3.0f;
                for (int k = 0; k < 3; ++k) {
                    float u = uv[k][0] + (cu - uv[k][0]) * center_blend;
                    float v = uv[k][1] + (cv - uv[k][1]) * center_blend;
                    float rgb[3];
                    sample_pixel(tx, face_id, u, v, rgb);
                    out_rgba8[written++] = pack_rgba8(rgb[0], rgb[1], rgb[2]);
                }
            }
            consumed_fvi += n;
        }
    } else if (face_vertex_index_count > 0) {
        int tri_count = face_vertex_index_count / 3;
        int ptex_faces = tx->numFaces();
        for (int t = 0; t < tri_count && written + 2 < out_color_count; ++t) {
            int face_id = (t < ptex_faces) ? t : (ptex_faces - 1);
            for (int k = 0; k < 3; ++k) {
                float u, v;
                tri_corner_uv(k, &u, &v);
                float rgb[3];
                sample_pixel(tx, face_id, u, v, rgb);
                out_rgba8[written++] = pack_rgba8(rgb[0], rgb[1], rgb[2]);
            }
        }
    }

    if (face_vertex_index_count > 0 && consumed_fvi > face_vertex_index_count) {
        tx->release();
        return 0;
    }
    tx->release();
    return written;
#endif
}

int nusd_ptex_sample_average_color(const char* path,
                                   float out_rgb[3],
                                   int max_samples)
{
    if (!path || !path[0] || !out_rgb || !ptex_enabled())
        return 0;
    out_rgb[0] = out_rgb[1] = out_rgb[2] = 0.0f;

#ifndef NUSD_HAVE_PTEX
    (void)max_samples;
    return 0;
#else
    Ptex::String error;
    Ptex::PtexTexture* tx = Ptex::PtexTexture::open(path, error, true);
    if (!tx) return 0;
    int nfaces = tx->numFaces();
    if (nfaces <= 0 || tx->numChannels() <= 0) {
        tx->release();
        return 0;
    }

    int nsamp = max_samples > 0 ? max_samples : 64;
    nsamp = std::min(nsamp, nfaces);
    int step = std::max(nfaces / nsamp, 1);
    double acc[3] = {0.0, 0.0, 0.0};
    int count = 0;
    for (int f = 0; f < nfaces && count < nsamp; f += step) {
        float rgb[3];
        sample_pixel(tx, f, 0.5f, 0.5f, rgb);
        acc[0] += rgb[0];
        acc[1] += rgb[1];
        acc[2] += rgb[2];
        count++;
    }
    tx->release();
    if (count == 0) return 0;
    out_rgb[0] = (float)(acc[0] / (double)count);
    out_rgb[1] = (float)(acc[1] / (double)count);
    out_rgb[2] = (float)(acc[2] / (double)count);
    return 1;
#endif
}
