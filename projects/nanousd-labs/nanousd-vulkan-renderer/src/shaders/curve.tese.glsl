// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 450

layout(quads, fractional_odd_spacing, ccw) in;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
    vec4 color;
    vec4 eye_pos;    // xyz camera, w = int curve basis id bits
    vec4 ibl_params;
    vec4 tone_params;
} pc;

layout(location = 0) in vec3  tcsPos[];
layout(location = 1) in float tcsWidth[];
layout(location = 2) in vec3  tcsColor[];

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec2 fragTexCoord;
layout(location = 4) out vec3 fragTangent;

const float TWO_PI = 6.28318530717958647692;

void compute_basis(int basis_id, float u, out vec4 b, out vec4 db) {
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
        db[0] = -0.5 * u2 + u - 0.5;
        db[1] = 1.5 * u2 - 2.0 * u;
        db[2] = -1.5 * u2 + u + 0.5;
        db[3] = 0.5 * u2;
    } else if (basis_id == 2) {
        b[0] = -0.5 * u3 + u2 - 0.5 * u;
        b[1] = 1.5 * u3 - 2.5 * u2 + 1.0;
        b[2] = -1.5 * u3 + 2.0 * u2 + 0.5 * u;
        b[3] = 0.5 * u3 - 0.5 * u2;
        db[0] = -1.5 * u2 + 2.0 * u - 0.5;
        db[1] = 4.5 * u2 - 5.0 * u;
        db[2] = -4.5 * u2 + 4.0 * u + 0.5;
        db[3] = 1.5 * u2 - u;
    } else if (basis_id == 3) {
        b[0] = ic;
        b[1] = u;
        b[2] = 0.0;
        b[3] = 0.0;
        db[0] = -1.0;
        db[1] = 1.0;
        db[2] = 0.0;
        db[3] = 0.0;
    } else {
        b[0] = ic3;
        b[1] = 3.0 * u * ic2;
        b[2] = 3.0 * u2 * ic;
        b[3] = u3;
        db[0] = -3.0 * ic2;
        db[1] = 3.0 * ic2 - 6.0 * u * ic;
        db[2] = 6.0 * u * ic - 3.0 * u2;
        db[3] = 3.0 * u2;
    }
}

void main() {
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;
    int basis_id = floatBitsToInt(pc.eye_pos.w);

    vec4 b;
    vec4 db;
    compute_basis(basis_id, u, b, db);

    vec3 P = b[0] * tcsPos[0] + b[1] * tcsPos[1] +
             b[2] * tcsPos[2] + b[3] * tcsPos[3];
    vec3 T = db[0] * tcsPos[0] + db[1] * tcsPos[1] +
             db[2] * tcsPos[2] + db[3] * tcsPos[3];
    float tlen = length(T);
    T = (tlen > 1.0e-6) ? (T / tlen) : vec3(0.0, 0.0, 1.0);

    vec3 obj_up = vec3(0.0, 1.0, 0.0);
    if (abs(dot(T, obj_up)) > 0.99) obj_up = vec3(1.0, 0.0, 0.0);
    vec3 side = normalize(cross(T, obj_up));
    vec3 up = cross(side, T);

    float radius = 0.5 * (b[0] * tcsWidth[0] + b[1] * tcsWidth[1] +
                          b[2] * tcsWidth[2] + b[3] * tcsWidth[3]);
    float theta = TWO_PI * v;
    vec3 ring_dir = cos(theta) * side + sin(theta) * up;
    vec3 os_pos = P + radius * ring_dir;

    vec4 world_pos = vec4(os_pos, 1.0) * pc.model;
    fragWorldPos = world_pos.xyz;
    fragNormal = normalize((vec4(ring_dir, 0.0) * pc.model).xyz);
    fragTangent = normalize((vec4(T, 0.0) * pc.model).xyz);
    fragTexCoord = vec2(0.0, v);
    fragColor = b[0] * tcsColor[0] + b[1] * tcsColor[1] +
                b[2] * tcsColor[2] + b[3] * tcsColor[3];
    if (pc.color.w > 0.5) fragColor = pc.color.rgb;

    gl_Position = vec4(os_pos, 1.0) * pc.mvp;
}
