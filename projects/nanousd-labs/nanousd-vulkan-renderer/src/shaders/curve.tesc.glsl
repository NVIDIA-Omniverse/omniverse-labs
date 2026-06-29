// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 450

layout(vertices = 4) out;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
    vec4 color;
    vec4 eye_pos;    // xyz camera, w = int curve basis id bits
    vec4 ibl_params; // x/y = render width/height for curve LOD
    vec4 tone_params;
} pc;

layout(location = 0) in vec3  vsPos[];
layout(location = 1) in float vsWidth[];
layout(location = 2) in vec3  vsColor[];

layout(location = 0) out vec3  tcsPos[];
layout(location = 1) out float tcsWidth[];
layout(location = 2) out vec3  tcsColor[];

const float MAX_TESS = 64.0;
const float PIXEL_TO_TESS = 7.0;
const float WIDTH_PIXEL_TO_TESS = 8.0;
const float MIN_AROUND = 6.0;

void compute_basis_w(int basis_id, float u, out vec4 b) {
    float u2 = u * u;
    float u3 = u2 * u;
    float ic = 1.0 - u;
    float ic2 = ic * ic;
    float ic3 = ic2 * ic;
    if (basis_id == 1) {
        b[0] = -u3 / 6.0 + 0.5 * u2 - 0.5 * u + 1.0 / 6.0;
        b[1] = 0.5 * u3 - u2 + 2.0 / 3.0;
        b[2] = -0.5 * u3 + 0.5 * u2 + 0.5 * u + 1.0 / 6.0;
        b[3] = u3 / 6.0;
    } else if (basis_id == 2) {
        b[0] = -0.5 * u3 + u2 - 0.5 * u;
        b[1] = 1.5 * u3 - 2.5 * u2 + 1.0;
        b[2] = -1.5 * u3 + 2.0 * u2 + 0.5 * u;
        b[3] = 0.5 * u3 - 0.5 * u2;
    } else if (basis_id == 3) {
        b[0] = ic;
        b[1] = u;
        b[2] = 0.0;
        b[3] = 0.0;
    } else {
        b[0] = ic3;
        b[1] = 3.0 * u * ic2;
        b[2] = 3.0 * u2 * ic;
        b[3] = u3;
    }
}

vec2 project_to_screen(vec4 clip, vec2 screen_size) {
    float inv_w = 1.0 / max(abs(clip.w), 1.0e-4);
    vec2 ndc = clamp(clip.xy * inv_w, vec2(-1.3), vec2(1.3));
    return (ndc + vec2(1.0)) * (screen_size * 0.5);
}

float tess_length_from_screen(float pixels) {
    return clamp(pixels / PIXEL_TO_TESS, 1.0, MAX_TESS);
}

float tess_width_from_screen(float pixels) {
    return clamp(pixels / WIDTH_PIXEL_TO_TESS, MIN_AROUND, MAX_TESS);
}

float world_to_pixel_width(vec4 clip, vec2 screen_size) {
    float inv_w = 1.0 / max(abs(clip.w), 1.0e-4);
    float proj_scale = max(abs(pc.mvp[0][0]), abs(pc.mvp[1][1]));
    return abs(screen_size.x * 0.5 * proj_scale * inv_w);
}

void main() {
    tcsPos[gl_InvocationID] = vsPos[gl_InvocationID];
    tcsWidth[gl_InvocationID] = vsWidth[gl_InvocationID];
    tcsColor[gl_InvocationID] = vsColor[gl_InvocationID];

    if (gl_InvocationID == 0) {
        int basis_id = floatBitsToInt(pc.eye_pos.w);
        vec2 screen_size = max(pc.ibl_params.xy, vec2(1.0));

        vec4 clipP[4];
        vec2 scrP[4];
        for (int i = 0; i < 4; i++) {
            clipP[i] = vec4(vsPos[i], 1.0) * pc.mvp;
            scrP[i] = project_to_screen(clipP[i], screen_size);
        }

        float dist = distance(scrP[0], scrP[1]) +
                     distance(scrP[1], scrP[2]) +
                     distance(scrP[2], scrP[3]);
        float level = tess_length_from_screen(dist);

        const float width_multiplier = 2.0;
        vec4 scrW;
        for (int i = 0; i < 4; i++) {
            scrW[i] = world_to_pixel_width(clipP[i], screen_size) *
                      vsWidth[i] * width_multiplier;
        }

        vec4 b00;
        vec4 b33;
        vec4 b66;
        vec4 b10;
        compute_basis_w(basis_id, 0.0, b00);
        compute_basis_w(basis_id, 1.0 / 3.0, b33);
        compute_basis_w(basis_id, 2.0 / 3.0, b66);
        compute_basis_w(basis_id, 1.0, b10);
        float w00 = dot(b00, scrW);
        float w33 = dot(b33, scrW);
        float w66 = dot(b66, scrW);
        float w10 = dot(b10, scrW);
        float level_width_avg = 0.25 *
            (tess_width_from_screen(w00) + tess_width_from_screen(w33) +
             tess_width_from_screen(w66) + tess_width_from_screen(w10));

        gl_TessLevelOuter[0] = tess_width_from_screen(w00);
        gl_TessLevelOuter[1] = level;
        gl_TessLevelOuter[2] = tess_width_from_screen(w10);
        gl_TessLevelOuter[3] = level;
        gl_TessLevelInner[0] = level;
        gl_TessLevelInner[1] = level_width_avg;
    }
}
