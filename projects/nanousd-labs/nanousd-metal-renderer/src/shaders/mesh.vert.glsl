// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;   // world transform (for normals + world position)
    vec4 color;   // .w > 0.5 = override vertex color
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;

void main() {
    gl_Position = vec4(inPosition, 1.0) * pc.mvp;
    fragWorldPos = (vec4(inPosition, 1.0) * pc.model).xyz;
    fragNormal = normalize((vec4(inNormal, 0.0) * pc.model).xyz);
    fragColor = pc.color.rgb;
}
