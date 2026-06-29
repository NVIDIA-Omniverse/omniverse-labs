// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_ray_tracing : require

/*
 * DLSS ray generation shader — writes color, depth, and motion vectors
 * to three separate storage images for DLSS Super Resolution input.
 *
 * Push constants (208 bytes):
 *   viewInverse  = current frame inverse view
 *   projInverse  = current frame inverse projection (jittered)
 *   prevVP       = previous frame's unjittered view-projection
 *   jitter       = pixel-space jitter offsets
 */

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba8) uniform image2D colorImage;
layout(set = 0, binding = 3, r32f)  uniform image2D depthImage;
layout(set = 0, binding = 4, rg16f) uniform image2D mvImage;

layout(push_constant) uniform PushConstants {
    mat4  viewInverse;
    mat4  projInverse;
    mat4  prevVP;
    vec2  jitter;
    vec2  pad;
} pc;

layout(location = 0) rayPayloadEXT vec3  hitValue;
layout(location = 1) rayPayloadEXT vec3  hitWorldPos;
layout(location = 2) rayPayloadEXT float hitT;

void main()
{
    /* Apply sub-pixel jitter to the pixel center. */
    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5) + pc.jitter;
    const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = inUV * 2.0 - 1.0;

    vec4 origin    = vec4(0, 0, 0, 1) * pc.viewInverse;
    vec4 target    = vec4(d.x, d.y, 1, 1) * pc.projInverse;
    vec4 direction = vec4(normalize(target.xyz), 0) * pc.viewInverse;

    /* Initialize payload to miss defaults. */
    hitValue    = vec3(0.0);
    hitWorldPos = vec3(0.0);
    hitT        = -1.0;

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

    ivec2 px = ivec2(gl_LaunchIDEXT.xy);

    /* Color output */
    imageStore(colorImage, px, vec4(hitValue, 1.0));

    /* Depth output: linear depth from ray t-value.
     * Negative hitT means miss (sky) — write far plane. */
    float depth = (hitT > 0.0) ? hitT : 100000.0;
    imageStore(depthImage, px, vec4(depth, 0.0, 0.0, 0.0));

    /* Motion vector: reproject world-space hit through previous VP. */
    vec2 mv = vec2(0.0);
    if (hitT > 0.0) {
        vec4 prevClip = vec4(hitWorldPos, 1.0) * pc.prevVP;
        vec2 prevNDC  = prevClip.xy / prevClip.w;

        /* Current NDC (unjittered pixel center for MV reference) */
        vec2 currCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
        vec2 currNDC = currCenter / vec2(gl_LaunchSizeEXT.xy) * 2.0 - 1.0;

        mv = (prevNDC - currNDC) * 0.5;
    }
    imageStore(mvImage, px, vec4(mv, 0.0, 0.0));
}
