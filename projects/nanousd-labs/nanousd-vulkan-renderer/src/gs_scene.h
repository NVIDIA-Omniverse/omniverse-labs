// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * gs_scene.h — internal Gaussian splat (VK3DGRT) scene state.
 *
 * Owns CPU-side particle arrays + config knobs. GPU acceleration
 * structures and ray-tracing pipeline live in gpu_vulkan.c (P2/P3).
 *
 * See docs/plans/VK3DGRT_PLAN.md (parent repo).
 */

#ifndef NUSD_GS_SCENE_H
#define NUSD_GS_SCENE_H

#include "nusd_renderer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default values (see plan §6 "Constants (defaults)"). */
#define NU_GS_DEFAULT_K                       16
/* Reference vk_gaussian_splatting default is 200 (shaderio.h::maxPasses). */
#define NU_GS_DEFAULT_MAX_PASSES              200
#define NU_GS_DEFAULT_MIN_TRANSMITTANCE       0.03f
#define NU_GS_DEFAULT_ISO_OPACITY_THRESHOLD   0.5f
#define NU_GS_DEFAULT_KERNEL_DEGREE           2     /* quadratic */
#define NU_GS_KERNEL_MIN_RESPONSE             0.0113f /* exp(-0.5 * 9) */
#define NU_GS_MAX_PARTICLE_SQUARED_DISTANCE   9.0f  /* 3-sigma cutoff */
#define NU_GS_ALPHA_CLAMP                     0.99f
#define NU_GS_ALPHA_CULL_THRESHOLD            (1.0f / 255.0f)

typedef struct NuGsScene NuGsScene;

/* Lifecycle */
NuGsScene* gs_scene_create(void);
void       gs_scene_destroy(NuGsScene* scene);

/* Data ingestion */
NuResult   gs_scene_set_particles(NuGsScene* scene, const NuGsDesc* desc);
NuResult   gs_scene_clear_particles(NuGsScene* scene);

/* Configuration */
NuResult   gs_scene_set_proxy(NuGsScene* scene, NuGsProxyKind kind);
NuResult   gs_scene_set_color_space(NuGsScene* scene, NuGsColorSpace cs);
NuResult   gs_scene_set_camera_model(NuGsScene* scene, NuGsCameraModel cm);
NuResult   gs_scene_set_k(NuGsScene* scene, int k);
NuResult   gs_scene_set_max_passes(NuGsScene* scene, int max_passes);
NuResult   gs_scene_set_min_transmittance(NuGsScene* scene, float eps);
NuResult   gs_scene_set_iso_opacity_threshold(NuGsScene* scene, float iso);

/* Queries */
int             gs_scene_particle_count(const NuGsScene* scene);
int             gs_scene_sh_degree(const NuGsScene* scene);
NuGsProxyKind   gs_scene_proxy(const NuGsScene* scene);
NuGsColorSpace  gs_scene_color_space(const NuGsScene* scene);
NuGsCameraModel gs_scene_camera_model(const NuGsScene* scene);
int             gs_scene_k(const NuGsScene* scene);
int             gs_scene_max_passes(const NuGsScene* scene);
float           gs_scene_min_transmittance(const NuGsScene* scene);
float           gs_scene_iso_opacity_threshold(const NuGsScene* scene);

/* Access raw arrays for upload to GPU (P2). All pointers may be NULL when
 * particle_count == 0. Returned pointers are stable until the next
 * gs_scene_set_particles / gs_scene_clear_particles. */
const float* gs_scene_positions(const NuGsScene* scene);
const float* gs_scene_scales(const NuGsScene* scene);
const float* gs_scene_orientations(const NuGsScene* scene);
const float* gs_scene_opacities(const NuGsScene* scene);
const float* gs_scene_sh_coefficients(const NuGsScene* scene);
const float* gs_scene_kernel_scales(const NuGsScene* scene); /* per-particle adaptive scale */
const float* gs_scene_prim_xform(const NuGsScene* scene);    /* row-major 4x4 */

/* Adaptive kernelScale[i] = sqrt(log(opacity[i] / KERNEL_MIN_RESPONSE) / a),
 * where a = -4.5 / 3^kernel_degree. For default quadratic (degree 2),
 * a = -0.5, and kernelScale ~ 3.0 at typical opacity. Recomputed by
 * gs_scene_set_particles. Exposed for tests. */
float      gs_compute_kernel_scale(float opacity, int kernel_degree);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_GS_SCENE_H */
