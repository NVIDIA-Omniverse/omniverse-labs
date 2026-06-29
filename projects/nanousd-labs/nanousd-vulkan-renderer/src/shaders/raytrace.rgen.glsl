// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba8) uniform image2D outputImage;

layout(push_constant) uniform PushConstants {
    mat4 viewInverse;
    mat4 projInverse;
} pc;

layout(location = 0) rayPayloadEXT vec4 hitValue;

void main()
{
    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = inUV * 2.0 - 1.0;

    vec4 origin    = vec4(0, 0, 0, 1) * pc.viewInverse;
    vec4 target    = vec4(d.x, d.y, 1, 1) * pc.projInverse;
    vec4 direction = vec4(normalize(target.xyz), 0) * pc.viewInverse;

    hitValue = vec4(0.0, 0.0, 0.0, 0.0); /* w=0: primary ray depth */

    traceRayEXT(tlas,
                gl_RayFlagsOpaqueEXT,
                0xFF,       /* cullMask */
                0,          /* sbtRecordOffset */
                0,          /* sbtRecordStride */
                0,          /* missIndex */
                origin.xyz,
                0.001,      /* tmin */
                direction.xyz,
                100000.0,   /* tmax */
                0           /* payload location */
    );

    imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(hitValue.rgb, 1.0));
}
