// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * curve_aabb_gen.metal — GPU-side per-segment AABB computation for curves.
 *
 * Mirrors the Vulkan port's curve_aabb_gen.comp.glsl. Each thread reads
 * one CurveSegment (32 B: float3 p0; float r0; float3 p1; float r1) and
 * writes the corresponding 24-byte axis-aligned bounding box to the
 * device buffer the BLAS descriptor reads from. Bounds the constant-
 * radius cylinder of radius r0 between p0 and p1 — same geometry the
 * curve intersection function in raytrace.metal hits.
 *
 * Saves the host→device AABB upload at scene load (which can be tens of
 * MB on heavy-curve fixtures).
 *
 * Bindings:
 *   buffer(0): readonly  CurveSegment[]
 *   buffer(1): writeonly float[]            (6 floats per AABB, 24 B stride)
 *   buffer(2): constant  uint num_segments
 */

#include <metal_stdlib>
using namespace metal;

struct CurveSegment {
    packed_float3 p0;   /* packed: 32 B host stride (see raytrace.metal) */
    float         r0;
    packed_float3 p1;
    float         r1;
};

kernel void curve_aabb_gen(
    uint                          tid       [[thread_position_in_grid]],
    device const CurveSegment*    segs      [[buffer(0)]],
    device float*                 aabbs     [[buffer(1)]],
    constant uint&                num_segs  [[buffer(2)]])
{
    if (tid >= num_segs) return;

    CurveSegment s = segs[tid];

    /* Constant-radius cylinder bounds — matches curve_isect's geometry
     * (which uses mean(r0, r1), but for chess-set / linear curves r0 ==
     * r1, so r0 is the right inflation here. When varying-radius lands
     * we'll inflate by max(r0, r1) to be conservative). */
    float r = s.r0;
    float3 a_min = min(s.p0, s.p1) - float3(r);
    float3 a_max = max(s.p0, s.p1) + float3(r);

    uint base = tid * 6u;
    aabbs[base + 0u] = a_min.x;
    aabbs[base + 1u] = a_min.y;
    aabbs[base + 2u] = a_min.z;
    aabbs[base + 3u] = a_max.x;
    aabbs[base + 4u] = a_max.y;
    aabbs[base + 5u] = a_max.z;
}
