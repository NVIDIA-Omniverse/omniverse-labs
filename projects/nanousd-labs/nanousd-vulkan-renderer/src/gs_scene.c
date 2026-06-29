// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * gs_scene.c — internal Gaussian splat (VK3DGRT) scene state.
 *
 * Owns CPU-side particle arrays + config knobs. GPU acceleration
 * structures and ray-tracing pipeline live in gpu_vulkan.c (P2/P3).
 *
 * See docs/plans/VK3DGRT_PLAN.md (parent repo).
 */

#include "gs_scene.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct NuGsScene {
    /* Particle arrays — heap-owned deep copies. */
    float* positions;        /* [N * 3] */
    float* scales;           /* [N * 3] linear sigma */
    float* orientations;     /* [N * 4] wxyz, normalized */
    float* opacities;        /* [N * 1] in [0,1] */
    float* sh_coefficients;  /* [N * (sh_degree+1)^2 * 3] */
    float* kernel_scales;    /* [N * 1] precomputed adaptive scale */
    int    particle_count;
    int    particle_capacity; /* allocated slots; arrays may oversize this */
    int    sh_degree;         /* 0..3 */

    /* Per-prim transform (row-major 4x4). */
    float  prim_xform[16];

    /* Configuration knobs (defaults from gs_scene.h). */
    NuGsProxyKind   proxy;
    NuGsColorSpace  color_space;
    NuGsCameraModel camera_model;
    int             k;
    int             max_passes;
    float           min_transmittance;
    float           iso_opacity_threshold;
};

static void identity_xform(float m[16])
{
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static int sh_coeffs_per_particle(int sh_degree)
{
    int n = sh_degree + 1;
    return n * n;
}

/* ---- kernelScale (matches vk_gaussian_splatting/src/splat_set_vk.cpp::kernelScale).
 *
 * Solve density · exp(a · m^b) = KMR for m, where:
 *   b = kernel_degree, a = -4.5 / 3^b (negative; for b=2, a=-0.5).
 * With adaptive clamping (default-on per plan §5):
 *   modulatedMinResponse = min(KMR / density, 0.97)   // density = opacity
 *   m = pow(log(modulatedMinResponse) / a, 1/b)
 * Linear (b=0) case uses the closed-form from the reference. */
float gs_compute_kernel_scale(float opacity, int kernel_degree)
{
    /* Guard: vanishingly low opacity blows up modulatedMinResponse;
     * clamp once at the same 0.97 threshold the reference uses. */
    if (opacity <= 0.0f) opacity = 1e-6f;

    if (kernel_degree == 0) {
        const float min_response = NU_GS_KERNEL_MIN_RESPONSE / opacity;
        const float clamped = (min_response > 0.97f) ? 0.97f : min_response;
        return ((1.0f - clamped) / 3.0f) / -0.329630334487f;
    }

    const float b = (float)kernel_degree;
    const float a = -4.5f / powf(3.0f, b);

    float modulated = NU_GS_KERNEL_MIN_RESPONSE / opacity;
    if (modulated > 0.97f) modulated = 0.97f;

    return powf(logf(modulated) / a, 1.0f / b);
}

/* ---- Lifecycle ---- */

NuGsScene* gs_scene_create(void)
{
    NuGsScene* s = (NuGsScene*)calloc(1, sizeof(NuGsScene));
    if (!s) return NULL;
    identity_xform(s->prim_xform);
    s->proxy                 = NU_GS_PROXY_ICOSAHEDRON;
    s->color_space           = NU_GS_COLOR_LINEAR;
    s->camera_model          = NU_GS_CAMERA_PINHOLE;
    s->k                     = NU_GS_DEFAULT_K;
    s->max_passes            = NU_GS_DEFAULT_MAX_PASSES;
    s->min_transmittance     = NU_GS_DEFAULT_MIN_TRANSMITTANCE;
    s->iso_opacity_threshold = NU_GS_DEFAULT_ISO_OPACITY_THRESHOLD;
    return s;
}

void gs_scene_destroy(NuGsScene* s)
{
    if (!s) return;
    free(s->positions);
    free(s->scales);
    free(s->orientations);
    free(s->opacities);
    free(s->sh_coefficients);
    free(s->kernel_scales);
    free(s);
}

static void free_arrays(NuGsScene* s)
{
    free(s->positions);       s->positions       = NULL;
    free(s->scales);          s->scales          = NULL;
    free(s->orientations);    s->orientations    = NULL;
    free(s->opacities);       s->opacities       = NULL;
    free(s->sh_coefficients); s->sh_coefficients = NULL;
    free(s->kernel_scales);   s->kernel_scales   = NULL;
    s->particle_count    = 0;
    s->particle_capacity = 0;
    s->sh_degree         = 0;
}

NuResult gs_scene_clear_particles(NuGsScene* s)
{
    if (!s) return NU_ERROR;
    free_arrays(s);
    identity_xform(s->prim_xform);
    return NU_OK;
}

/* ---- Ingestion ---- */

NuResult gs_scene_set_particles(NuGsScene* s, const NuGsDesc* desc)
{
    if (!s || !desc) return NU_ERROR;

    /* Empty desc → treat as clear (mirrors the public API contract). */
    if (desc->particle_count <= 0) {
        return gs_scene_clear_particles(s);
    }

    if (desc->sh_degree < 0 || desc->sh_degree > 3) return NU_ERROR;
    if (!desc->positions || !desc->scales ||
        !desc->orientations || !desc->opacities ||
        !desc->sh_coefficients) return NU_ERROR;

    const int N = desc->particle_count;
    const int sh_per = sh_coeffs_per_particle(desc->sh_degree);

    /* (Re)allocate arrays. We always realloc on size change rather than
     * keeping a high-water capacity — particle counts change rarely
     * (whole-scene replace) and the savings don't outweigh complexity. */
    float* p_pos  = (float*)realloc(s->positions,       (size_t)N * 3 * sizeof(float));
    float* p_scl  = (float*)realloc(s->scales,          (size_t)N * 3 * sizeof(float));
    float* p_ori  = (float*)realloc(s->orientations,    (size_t)N * 4 * sizeof(float));
    float* p_opa  = (float*)realloc(s->opacities,       (size_t)N * 1 * sizeof(float));
    float* p_sh   = (float*)realloc(s->sh_coefficients, (size_t)N * (size_t)sh_per * 3 * sizeof(float));
    float* p_ker  = (float*)realloc(s->kernel_scales,   (size_t)N * 1 * sizeof(float));

    if (!p_pos || !p_scl || !p_ori || !p_opa || !p_sh || !p_ker) {
        /* On allocation failure, hold what realloc returned and don't
         * leak. The caller can retry. */
        if (p_pos) s->positions       = p_pos;
        if (p_scl) s->scales          = p_scl;
        if (p_ori) s->orientations    = p_ori;
        if (p_opa) s->opacities       = p_opa;
        if (p_sh)  s->sh_coefficients = p_sh;
        if (p_ker) s->kernel_scales   = p_ker;
        return NU_ERROR;
    }

    s->positions       = p_pos;
    s->scales          = p_scl;
    s->orientations    = p_ori;
    s->opacities       = p_opa;
    s->sh_coefficients = p_sh;
    s->kernel_scales   = p_ker;

    memcpy(s->positions,    desc->positions,    (size_t)N * 3 * sizeof(float));
    memcpy(s->scales,       desc->scales,       (size_t)N * 3 * sizeof(float));
    memcpy(s->orientations, desc->orientations, (size_t)N * 4 * sizeof(float));
    memcpy(s->opacities,    desc->opacities,    (size_t)N * 1 * sizeof(float));
    memcpy(s->sh_coefficients, desc->sh_coefficients,
           (size_t)N * (size_t)sh_per * 3 * sizeof(float));

    /* Precompute adaptive kernelScale per particle. */
    for (int i = 0; i < N; i++) {
        s->kernel_scales[i] = gs_compute_kernel_scale(
            desc->opacities[i], NU_GS_DEFAULT_KERNEL_DEGREE);
    }

    if (desc->prim_xform) {
        memcpy(s->prim_xform, desc->prim_xform, 16 * sizeof(float));
    } else {
        identity_xform(s->prim_xform);
    }

    s->particle_count    = N;
    s->particle_capacity = N;
    s->sh_degree         = desc->sh_degree;
    return NU_OK;
}

/* ---- Configuration ---- */

NuResult gs_scene_set_proxy(NuGsScene* s, NuGsProxyKind k)
{
    if (!s) return NU_ERROR;
    if (k != NU_GS_PROXY_ICOSAHEDRON && k != NU_GS_PROXY_AABB) return NU_ERROR;
    s->proxy = k;
    return NU_OK;
}

NuResult gs_scene_set_color_space(NuGsScene* s, NuGsColorSpace cs)
{
    if (!s) return NU_ERROR;
    if (cs != NU_GS_COLOR_LINEAR && cs != NU_GS_COLOR_SRGB) return NU_ERROR;
    s->color_space = cs;
    return NU_OK;
}

NuResult gs_scene_set_camera_model(NuGsScene* s, NuGsCameraModel cm)
{
    if (!s) return NU_ERROR;
    if (cm != NU_GS_CAMERA_PINHOLE &&
        cm != NU_GS_CAMERA_FISHEYE &&
        cm != NU_GS_CAMERA_EQUIRECT) return NU_ERROR;
    s->camera_model = cm;
    return NU_OK;
}

NuResult gs_scene_set_k(NuGsScene* s, int k)
{
    if (!s) return NU_ERROR;
    if (k != 8 && k != 16 && k != 32) return NU_ERROR;
    s->k = k;
    return NU_OK;
}

NuResult gs_scene_set_max_passes(NuGsScene* s, int max_passes)
{
    if (!s) return NU_ERROR;
    if (max_passes < 1) return NU_ERROR;
    s->max_passes = max_passes;
    return NU_OK;
}

NuResult gs_scene_set_min_transmittance(NuGsScene* s, float eps)
{
    if (!s) return NU_ERROR;
    if (!(eps >= 0.0f && eps <= 1.0f)) return NU_ERROR;  /* also rejects NaN */
    s->min_transmittance = eps;
    return NU_OK;
}

NuResult gs_scene_set_iso_opacity_threshold(NuGsScene* s, float iso)
{
    if (!s) return NU_ERROR;
    if (!(iso >= 0.0f && iso <= 1.0f)) return NU_ERROR;
    s->iso_opacity_threshold = iso;
    return NU_OK;
}

/* ---- Queries ---- */

int             gs_scene_particle_count       (const NuGsScene* s) { return s ? s->particle_count       : 0; }
int             gs_scene_sh_degree            (const NuGsScene* s) { return s ? s->sh_degree            : 0; }
NuGsProxyKind   gs_scene_proxy                (const NuGsScene* s) { return s ? s->proxy                : NU_GS_PROXY_ICOSAHEDRON; }
NuGsColorSpace  gs_scene_color_space          (const NuGsScene* s) { return s ? s->color_space          : NU_GS_COLOR_LINEAR; }
NuGsCameraModel gs_scene_camera_model         (const NuGsScene* s) { return s ? s->camera_model         : NU_GS_CAMERA_PINHOLE; }
int             gs_scene_k                    (const NuGsScene* s) { return s ? s->k                    : NU_GS_DEFAULT_K; }
int             gs_scene_max_passes           (const NuGsScene* s) { return s ? s->max_passes           : NU_GS_DEFAULT_MAX_PASSES; }
float           gs_scene_min_transmittance    (const NuGsScene* s) { return s ? s->min_transmittance    : NU_GS_DEFAULT_MIN_TRANSMITTANCE; }
float           gs_scene_iso_opacity_threshold(const NuGsScene* s) { return s ? s->iso_opacity_threshold : NU_GS_DEFAULT_ISO_OPACITY_THRESHOLD; }

const float* gs_scene_positions      (const NuGsScene* s) { return s ? s->positions       : NULL; }
const float* gs_scene_scales         (const NuGsScene* s) { return s ? s->scales          : NULL; }
const float* gs_scene_orientations   (const NuGsScene* s) { return s ? s->orientations    : NULL; }
const float* gs_scene_opacities      (const NuGsScene* s) { return s ? s->opacities       : NULL; }
const float* gs_scene_sh_coefficients(const NuGsScene* s) { return s ? s->sh_coefficients : NULL; }
const float* gs_scene_kernel_scales  (const NuGsScene* s) { return s ? s->kernel_scales   : NULL; }
const float* gs_scene_prim_xform     (const NuGsScene* s) { return s ? s->prim_xform      : NULL; }
