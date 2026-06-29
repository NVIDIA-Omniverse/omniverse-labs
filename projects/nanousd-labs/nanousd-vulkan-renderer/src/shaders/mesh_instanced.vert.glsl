// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Instanced raster mesh vertex shader. Mirrors mesh.vert.glsl but reads the
// per-instance world matrix from instance-rate vertex attributes (locations
// 4..7 = the four columns of the placement matrix) instead of pc.model, so a
// single prototype mesh can be drawn once per PointInstancer placement.
//
// Convention: the instance matrix columns are uploaded in the same 16-float
// layout as pc.model in the non-instanced path (column-vector row-major,
// translation in elements 3/7/11 — i.e. the vk3x4 form padded with a (0,0,0,1)
// bottom row). pc.mvp here carries the camera view-projection (NOT mvp*model),
// and the final clip position reuses the proven `vec4 * matrix` row-vector
// operator: clip = (pos * model) * VP, which equals the non-instanced
// pos * (VP*model) by associativity.

#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;     // here: camera view-projection (VP)
    mat4 model;   // unused in the instanced path
    vec4 color;   // .w packs the prototype Ptex color offset (frag reads it)
    vec4 eye_pos; // camera world position (xyz); .w = material id (int bits)
    vec4 ibl_params;
    vec4 tone_params;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in uint inMatID;
// Per-instance world matrix (instance-rate, binding 1) as four vec4 columns.
layout(location = 4) in vec4 inModel0;
layout(location = 5) in vec4 inModel1;
layout(location = 6) in vec4 inModel2;
layout(location = 7) in vec4 inModel3;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec2 fragUV;
layout(location = 4) flat out int fragMatID;
layout(location = 5) flat out vec3 fragEyePos;

void main() {
    mat4 model = mat4(inModel0, inModel1, inModel2, inModel3);
    vec4 worldPos = vec4(inPosition, 1.0) * model;
    gl_Position = vec4(worldPos.xyz, 1.0) * pc.mvp;   // pc.mvp == VP
    fragWorldPos = worldPos.xyz;
    fragNormal = normalize((vec4(inNormal, 0.0) * model).xyz);
    fragColor = pc.color.rgb;
    fragUV = inUV;
    fragMatID = int(inMatID);
    fragEyePos = pc.eye_pos.xyz;
}
