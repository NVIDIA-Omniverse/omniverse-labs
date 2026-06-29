// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 450

/*
 * DLSS vertex shader — outputs current and previous clip positions
 * for per-pixel motion vector computation in the fragment shader.
 *
 * Push constants: 208 bytes (within NVIDIA's 256-byte limit)
 *   mvp      = current frame jittered MVP
 *   model    = world transform (for normals)
 *   color    = override color
 *   prev_mvp = previous frame unjittered MVP (for motion vectors)
 */

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
    vec4 color;
    mat4 prev_mvp;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec4 currClipPos;
layout(location = 4) out vec4 prevClipPos;

void main() {
    vec4 pos = vec4(inPosition, 1.0);

    currClipPos = pos * pc.mvp;
    prevClipPos = pos * pc.prev_mvp;
    gl_Position = currClipPos;

    fragWorldPos = (pos * pc.model).xyz;
    fragNormal = normalize((vec4(inNormal, 0.0) * pc.model).xyz);
    fragColor = pc.color.rgb;
}
