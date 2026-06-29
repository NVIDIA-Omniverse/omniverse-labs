// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
    vec4 color;
    vec4 eye_pos;
    vec4 ibl_params;
    vec4 tone_params;
} pc;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragTexCoord;
layout(location = 4) in vec3 fragTangent;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(pc.eye_pos.xyz - fragWorldPos);
    vec3 L = V;
    vec3 H = normalize(V + L);
    float ndotl = max(dot(N, L), 0.0);
    float ndoth = max(dot(N, H), 0.0);
    vec3 diffuse = fragColor * ndotl;
    float spec = pow(ndoth, 32.0);
    vec3 ambient = fragColor * 0.18;
    vec3 lit = ambient + diffuse * 0.55 + vec3(spec * 0.10);
    lit *= max(pc.tone_params.x, 0.0);
    outColor = vec4(pow(clamp(lit, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
