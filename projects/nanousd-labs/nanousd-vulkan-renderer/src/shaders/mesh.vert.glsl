// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;   // world transform (for normals + world position)
    vec4 color;   // .w > 0.5 = override vertex color
    vec4 eye_pos; // camera world position (xyz)
    vec4 ibl_params; // x=has_ibl, y=mip_levels, z=intensity (consumed by frag)
    vec4 tone_params;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
/* matID arrives as uint32 raw bits (packed by nu_add_mesh's memcpy) —
 * read as `uint` so the value isn't reinterpreted as a denormal float. */
layout(location = 3) in uint inMatID;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec2 fragUV;
layout(location = 4) flat out int fragMatID;
layout(location = 5) flat out vec3 fragEyePos;

void main() {
    gl_Position = vec4(inPosition, 1.0) * pc.mvp;
    fragWorldPos = (vec4(inPosition, 1.0) * pc.model).xyz;
    fragNormal = normalize((vec4(inNormal, 0.0) * pc.model).xyz);
    fragColor = pc.color.rgb;
    fragUV = inUV;
    fragMatID = int(inMatID);
    fragEyePos = pc.eye_pos.xyz;
}
