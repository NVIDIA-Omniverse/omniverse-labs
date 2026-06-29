// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "camera.h"
#include <math.h>
#include <string.h>

static float dot3(const float a[3], const float b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void cross3(const float a[3], const float b[3], float out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static void normalize3(float v[3])
{
    float len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 1e-12f) {
        float inv = 1.0f / len;
        v[0] *= inv;
        v[1] *= inv;
        v[2] *= inv;
    }
}

/* Row-major 4x4 matrix multiply: out = a * b */
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

void camera_init(Camera* cam, const float bounds_min[3], const float bounds_max[3], float aspect)
{
    /* Check for degenerate bounds */
    float dx = bounds_max[0] - bounds_min[0];
    float dy = bounds_max[1] - bounds_min[1];
    float dz = bounds_max[2] - bounds_min[2];
    float diag = sqrtf(dx * dx + dy * dy + dz * dz);

    if (diag < 1e-6f) {
        cam->target[0] = 0.0f;
        cam->target[1] = 0.0f;
        cam->target[2] = 0.0f;
        cam->distance = 5.0f;
    } else {
        cam->target[0] = (bounds_min[0] + bounds_max[0]) * 0.5f;
        cam->target[1] = (bounds_min[1] + bounds_max[1]) * 0.5f;
        cam->target[2] = (bounds_min[2] + bounds_max[2]) * 0.5f;
        cam->distance = diag * 1.5f;
    }

    cam->yaw = 0.3f;
    cam->pitch = -0.3f;
    cam->fov_y = 60.0f * (3.14159265358979323846f / 180.0f);
    cam->aspect = aspect;
    cam->near_z = cam->distance * 0.001f;
    cam->far_z = cam->distance * 100.0f;
    cam->projection_shift_x = 0.0f;
    cam->projection_shift_y = 0.0f;
}

void camera_set_eye_target(Camera* cam, const float eye[3], const float target[3])
{
    cam->has_explicit_view = 0;
    cam->projection_shift_x = 0.0f;
    cam->projection_shift_y = 0.0f;
    cam->target[0] = target[0];
    cam->target[1] = target[1];
    cam->target[2] = target[2];

    float dx = eye[0] - target[0];
    float dy = eye[1] - target[1];
    float dz = eye[2] - target[2];
    cam->distance = sqrtf(dx * dx + dy * dy + dz * dz);
    if (cam->distance < 1e-6f) cam->distance = 1.0f;

    /* Compute yaw and pitch from the eye-to-target direction.
     * Y is up. Yaw is rotation around Y, pitch is elevation. */
    float horiz = sqrtf(dx * dx + dz * dz);
    cam->pitch = atan2f(dy, horiz);
    cam->yaw = atan2f(dx, dz);

    cam->near_z = cam->distance * 0.001f;
    cam->far_z = cam->distance * 100.0f;
}

void camera_set_explicit_view(Camera* cam,
                              const float eye[3],
                              const float target[3],
                              const float up[3],
                              float fov_y_radians,
                              float near_z, float far_z)
{
    cam->has_explicit_view = 1;
    cam->fov_y = fov_y_radians;
    cam->near_z = near_z;
    cam->far_z = far_z;
    cam->projection_shift_x = 0.0f;
    cam->projection_shift_y = 0.0f;
    /* Keep target/distance roughly sane for any orbit/zoom operations a
     * caller might still issue; the view matrix below is what actually
     * drives rendering. */
    cam->target[0] = target[0];
    cam->target[1] = target[1];
    cam->target[2] = target[2];
    float dx = eye[0] - target[0];
    float dy = eye[1] - target[1];
    float dz = eye[2] - target[2];
    cam->distance = sqrtf(dx * dx + dy * dy + dz * dz);
    if (cam->distance < 1e-6f) cam->distance = 1.0f;

    /* Build a right-handed look-at view matrix for an arbitrary up vector. */
    float forward[3] = {
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2],
    };
    normalize3(forward);

    float right[3];
    cross3(forward, up, right);
    float rn = sqrtf(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    if (rn < 1e-6f) {
        /* Forward parallel to up: pick an alternate up. */
        float alt[3] = {1.0f, 0.0f, 0.0f};
        if (fabsf(up[0]) > 0.5f) { alt[0] = 0.0f; alt[1] = 1.0f; }
        cross3(forward, alt, right);
    }
    normalize3(right);

    float v_up[3];
    cross3(right, forward, v_up);

    float* m = cam->view_matrix;
    m[0]  =  right[0];   m[1]  =  right[1];  m[2]  =  right[2];   m[3]  = -dot3(right, eye);
    m[4]  =  v_up[0];    m[5]  =  v_up[1];   m[6]  =  v_up[2];    m[7]  = -dot3(v_up, eye);
    m[8]  = -forward[0]; m[9]  = -forward[1]; m[10] = -forward[2]; m[11] =  dot3(forward, eye);
    m[12] = 0.0f;        m[13] = 0.0f;        m[14] = 0.0f;        m[15] = 1.0f;
}

void camera_set_projection_shift(Camera* cam, float shift_x, float shift_y)
{
    if (!cam) return;
    cam->projection_shift_x = isfinite(shift_x) ? shift_x : 0.0f;
    cam->projection_shift_y = isfinite(shift_y) ? shift_y : 0.0f;
}

void camera_clear_explicit_view(Camera* cam)
{
    cam->has_explicit_view = 0;
    cam->projection_shift_x = 0.0f;
    cam->projection_shift_y = 0.0f;
}

void camera_orbit(Camera* cam, float dx, float dy)
{
    cam->yaw += dx * 0.005f;
    cam->pitch += dy * 0.005f;

    /* Clamp pitch to [-89, 89] degrees in radians */
    const float limit = 89.0f * (3.14159265358979323846f / 180.0f); /* ~1.553 */
    if (cam->pitch > limit)  cam->pitch = limit;
    if (cam->pitch < -limit) cam->pitch = -limit;
}

void camera_zoom(Camera* cam, float delta)
{
    cam->distance *= (1.0f - delta * 0.1f);

    if (cam->distance < cam->near_z) cam->distance = cam->near_z;
    if (cam->distance > cam->far_z)  cam->distance = cam->far_z;
}

void camera_pan(Camera* cam, float dx, float dy)
{
    /* Compute right vector from yaw/pitch */
    float forward[3] = {
        -cosf(cam->pitch) * sinf(cam->yaw),
        -sinf(cam->pitch),
        -cosf(cam->pitch) * cosf(cam->yaw)
    };

    float world_up[3] = { 0.0f, 1.0f, 0.0f };

    float right[3];
    cross3(forward, world_up, right);
    normalize3(right);

    float up[3];
    cross3(right, forward, up);
    normalize3(up);

    float scale = cam->distance * 0.001f;
    cam->target[0] += right[0] * dx * scale + up[0] * dy * scale;
    cam->target[1] += right[1] * dx * scale + up[1] * dy * scale;
    cam->target[2] += right[2] * dx * scale + up[2] * dy * scale;
}

void camera_move(Camera* cam, float fwd, float right_amt, float up_amt)
{
    /* Forward direction (from eye toward target) projected onto XZ plane */
    float forward[3] = {
        -cosf(cam->pitch) * sinf(cam->yaw),
        0.0f,
        -cosf(cam->pitch) * cosf(cam->yaw)
    };
    normalize3(forward);

    float world_up[3] = { 0.0f, 1.0f, 0.0f };
    float right[3];
    cross3(forward, world_up, right);
    normalize3(right);

    cam->target[0] += forward[0] * fwd + right[0] * right_amt;
    cam->target[1] += up_amt;
    cam->target[2] += forward[2] * fwd + right[2] * right_amt;
}

void camera_get_eye(const Camera* cam, float out[3])
{
    if (cam->has_explicit_view) {
        /* eye = -R^T * t  where view = [R | t; 0 0 0 1] (row-major). */
        const float* m = cam->view_matrix;
        out[0] = -(m[0] * m[3] + m[4] * m[7] + m[8]  * m[11]);
        out[1] = -(m[1] * m[3] + m[5] * m[7] + m[9]  * m[11]);
        out[2] = -(m[2] * m[3] + m[6] * m[7] + m[10] * m[11]);
        return;
    }
    out[0] = cam->target[0] + cam->distance * cosf(cam->pitch) * sinf(cam->yaw);
    out[1] = cam->target[1] + cam->distance * sinf(cam->pitch);
    out[2] = cam->target[2] + cam->distance * cosf(cam->pitch) * cosf(cam->yaw);
}

void camera_get_view(const Camera* cam, float out[16])
{
    if (cam->has_explicit_view) {
        for (int i = 0; i < 16; i++) out[i] = cam->view_matrix[i];
        return;
    }
    /* Eye position from spherical coordinates */
    float eye[3];
    eye[0] = cam->target[0] + cam->distance * cosf(cam->pitch) * sinf(cam->yaw);
    eye[1] = cam->target[1] + cam->distance * sinf(cam->pitch);
    eye[2] = cam->target[2] + cam->distance * cosf(cam->pitch) * cosf(cam->yaw);

    /* Forward = normalize(target - eye) */
    float forward[3] = {
        cam->target[0] - eye[0],
        cam->target[1] - eye[1],
        cam->target[2] - eye[2]
    };
    normalize3(forward);

    /* Right = normalize(cross(forward, world_up)) */
    float world_up[3] = { 0.0f, 1.0f, 0.0f };
    float right[3];
    cross3(forward, world_up, right);
    normalize3(right);

    /* Up = cross(right, forward) */
    float up[3];
    cross3(right, forward, up);

    /* Build row-major view matrix */
    out[0]  =  right[0];
    out[1]  =  right[1];
    out[2]  =  right[2];
    out[3]  = -dot3(right, eye);

    out[4]  =  up[0];
    out[5]  =  up[1];
    out[6]  =  up[2];
    out[7]  = -dot3(up, eye);

    out[8]  = -forward[0];
    out[9]  = -forward[1];
    out[10] = -forward[2];
    out[11] =  dot3(forward, eye);

    out[12] = 0.0f;
    out[13] = 0.0f;
    out[14] = 0.0f;
    out[15] = 1.0f;
}

void camera_get_proj(const Camera* cam, float out[16])
{
    float f = 1.0f / tanf(cam->fov_y * 0.5f);
    float near_z = cam->near_z;
    float far_z = cam->far_z;
    float range = near_z - far_z;

    memset(out, 0, sizeof(float) * 16);

    out[0]  = f / cam->aspect;
    out[5]  = -f;                           /* Vulkan Y-flip */
    out[2]  = cam->projection_shift_x;
    out[6]  = cam->projection_shift_y;
    out[10] = far_z / range;
    out[11] = (near_z * far_z) / range;
    out[14] = -1.0f;
}

void camera_get_proj_jittered(const Camera* cam,
                              float jitter_x, float jitter_y,
                              uint32_t render_w, uint32_t render_h,
                              float out[16])
{
    camera_get_proj(cam, out);

    /* Apply sub-pixel jitter in NDC space.
     * Row-major layout:
     *   row 0: [f/aspect, 0,       jitter,  0]
     *   row 1: [0,       -f,       jitter,  0]
     *   row 2: [0,        0,       far/rng, near*far/rng]
     *   row 3: [0,        0,       -1,      0]
     *
     * The projection multiplies as:  clip = pos * P  (row-vector * matrix)
     * so jitter in column 2 offsets the x/y clip coordinates.
     */
    if (render_w > 0 && render_h > 0) {
        out[2] += jitter_x * 2.0f / (float)render_w;
        out[6] += jitter_y * 2.0f / (float)render_h;
    }
}

void camera_get_vp(const Camera* cam, float out[16])
{
    float view[16];
    float proj[16];
    camera_get_view(cam, view);
    camera_get_proj(cam, proj);
    mat4_mul(proj, view, out);
}

float halton(uint32_t index, uint32_t base)
{
    float f = 1.0f;
    float result = 0.0f;
    uint32_t i = index;
    while (i > 0) {
        f /= (float)base;
        result += f * (float)(i % base);
        i /= base;
    }
    return result;
}
