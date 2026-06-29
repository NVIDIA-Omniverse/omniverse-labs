// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_scalar_block_layout : enable

/*
 * curve_aabb_gen.comp.glsl — Generate per-segment AABBs on the GPU.
 *
 * Phase 12.x: this compute pass replaces the ~825 MB host→device
 * AABB upload that Phase 11.A.2.5 used to do at scene load. Each
 * thread reads one SceneCurveSegment from the curve segment SSBO
 * (binding 0) and writes one VkAabbPositionsKHR (24 B) to the AABB
 * device buffer (binding 1), then the host issues a single memory
 * barrier before vkCmdBuildAccelerationStructuresKHR.
 *
 * The AABB bounds the tube or oriented ribbon of half-width `r0` between
 * p0 and p1. For ribbons this is conservative because it expands in all
 * axes rather than only along the Storm side vector cross(tangent, normal).
 * That keeps primary visibility correct without another per-segment buffer.
 *
 * Bindings:
 *   0 — readonly  SceneCurveSegment[]   (32 B/seg)
 *   1 — writeonly VkAabbPositionsKHR[]  (24 B/seg, std430 packed)
 *
 * Push constant: total segment count.
 */

/* Workgroup size — overridable at compile time via -DNUSD_AABB_GEN_LOCAL_X=N
 * passed to glslang (host C code uses the same macro). 64 is the
 * historical default; 128/256 may be faster on RTX 4090/5090-class
 * hardware where 34 M segments saturate the SM scheduler. The host-
 * side dispatch ceiling-divides by this same constant. */
#ifndef NUSD_AABB_GEN_LOCAL_X
#define NUSD_AABB_GEN_LOCAL_X 64
#endif
layout(local_size_x = NUSD_AABB_GEN_LOCAL_X) in;

struct CurveSegment {
    vec3  p0;
    float r0;
    vec3  p1;
    uint  mat_flags;   /* MUST match scene.h SceneCurveSegment */
};

layout(set = 0, binding = 0, scalar) readonly buffer CurveSegmentBuffer {
    CurveSegment segs[];
} curve_segs;

/* VkAabbPositionsKHR is 6 floats packed (no padding). std430 with a
 * float array gives the same memory layout. Use float[] so the
 * compiler emits 4-byte stride writes — std430 with a vec3-struct
 * would inject 4-byte tail padding to vec4 alignment. */
layout(set = 0, binding = 1, std430) writeonly buffer CurveAabbBuffer {
    float aabbs[];
} curve_aabbs;

layout(push_constant) uniform PushConstants {
    uint num_segments;
} pc;

void main()
{
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.num_segments) return;

    CurveSegment seg = curve_segs.segs[i];

    /* Tube/ribbon bounds. Storm's ribbon shader offsets by authored width;
     * any screen-space coverage/AA has to stay out of the acceleration
     * geometry or real Moana leaf/frond widths get distorted. */
    float r = seg.r0;
    vec3 a_min = min(seg.p0, seg.p1) - vec3(r);
    vec3 a_max = max(seg.p0, seg.p1) + vec3(r);

    uint base = i * 6u;
    curve_aabbs.aabbs[base + 0u] = a_min.x;
    curve_aabbs.aabbs[base + 1u] = a_min.y;
    curve_aabbs.aabbs[base + 2u] = a_min.z;
    curve_aabbs.aabbs[base + 3u] = a_max.x;
    curve_aabbs.aabbs[base + 4u] = a_max.y;
    curve_aabbs.aabbs[base + 5u] = a_max.z;
}
