// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_CAMERA_H
#define NUSD_CAMERA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float target[3];
    float distance;
    float yaw;       /* radians */
    float pitch;     /* radians, clamped to [-89, 89] degrees */
    float fov_y;     /* radians */
    float aspect;
    float near_z;
    float far_z;
    float projection_shift_x;
    float projection_shift_y;
    /* When `has_explicit_view` is non-zero, `view_matrix` (row-major 4x4)
     * is used as-is by camera_get_view, bypassing the spherical (yaw/pitch
     * around Y-up) reconstruction. Lets callers feed in views computed
     * under arbitrary up-axis conventions (e.g. Z-up scenes). */
    int   has_explicit_view;
    float view_matrix[16];
} Camera;

/* Initialize camera to frame the given bounding box. */
void camera_init(Camera* cam, const float bounds_min[3],
                 const float bounds_max[3], float aspect);

/* Override camera position: set eye and target directly.
 * Computes orbit parameters (distance, yaw, pitch) from the eye-target vector. */
void camera_set_eye_target(Camera* cam, const float eye[3], const float target[3]);

/* Set a fully explicit row-major 4x4 view matrix. Subsequent camera_get_view
 * calls return this matrix verbatim until camera_clear_explicit_view() runs.
 * Use when the spherical Y-up parameterization isn't a fit (e.g. Z-up scenes). */
void camera_set_explicit_view(Camera* cam,
                              const float eye[3],
                              const float target[3],
                              const float up[3],
                              float fov_y_radians,
                              float near_z, float far_z);

/* Set an off-axis normalized projection shift for USD camera aperture offsets.
 * Values are added in clip/NDC units; 0,0 is the symmetric projection. */
void camera_set_projection_shift(Camera* cam, float shift_x, float shift_y);

void camera_clear_explicit_view(Camera* cam);

/* Orbit: dx/dy in pixels (or normalized screen coords). */
void camera_orbit(Camera* cam, float dx, float dy);

/* Zoom: delta > 0 zooms in. */
void camera_zoom(Camera* cam, float delta);

/* Pan: dx/dy in screen-space. */
void camera_pan(Camera* cam, float dx, float dy);

/* Translate camera along its local axes.
 * forward > 0 = toward target, right > 0 = strafe right, up > 0 = world up. */
void camera_move(Camera* cam, float forward, float right, float up);

/* Get camera eye position in world space. */
void camera_get_eye(const Camera* cam, float out[3]);

/* Get 4x4 view matrix (row-major float[16]). */
void camera_get_view(const Camera* cam, float out[16]);

/* Get 4x4 perspective projection matrix (row-major float[16]).
 * Standard depth range [0,1] with Vulkan Y-flip. */
void camera_get_proj(const Camera* cam, float out[16]);

/* Get jittered projection matrix for DLSS temporal accumulation.
 * jitter_x/jitter_y are in pixel coordinates (typically [-0.5, +0.5]).
 * render_w/render_h are the render resolution (for NDC conversion). */
void camera_get_proj_jittered(const Camera* cam,
                              float jitter_x, float jitter_y,
                              uint32_t render_w, uint32_t render_h,
                              float out[16]);

/* Get combined view-projection matrix (row-major float[16]). */
void camera_get_vp(const Camera* cam, float out[16]);

/* Halton low-discrepancy sequence value for the given index and base.
 * Used to generate sub-pixel jitter offsets for DLSS.
 * Returns a value in [0, 1). Subtract 0.5 for centered jitter. */
float halton(uint32_t index, uint32_t base);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_CAMERA_H */
