// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// gs_3dgrt.rahit.slang — VK3DGRT any-hit (K-buffer insertion-sort).
//
// rgen seeds payload.pixel_index and TraceRay's; for every icosahedron
// proxy hit, this shader insertion-sorts (t, instance_id) into the
// per-pixel SoA K-buffer at bindings 8/9. Always IgnoreHit() so
// traversal continues until the BLAS pool is exhausted — closest-hit is
// kept in the pipeline as a no-op (some Vulkan implementations gate
// any-hit invocations on a hit group having a closest-hit slot).
//
// IMPORTANT: the BLAS must NOT be marked VK_GEOMETRY_OPAQUE_BIT_KHR — the
// OPAQUE flag lets the implementation skip any-hit entirely. The host
// builds gs_blas_icosa with `geom.flags = 0`.
//
// Sort key is `RayTCurrent()` (proxy distance), not Mahalanobis² in
// canonical space. Plan §6 sorts the same way: the proxy is a
// conservative bound on the Gaussian's effective region, so t-ordering
// of proxy entries is a valid front-to-back ordering of the underlying
// Gaussians.

#include "gs_common.h.slang"

[[vk::binding(8, 0)]] RWStructuredBuffer<uint>  gs_kbuf_id;
[[vk::binding(9, 0)]] RWStructuredBuffer<float> gs_kbuf_dist;

struct GsPayload
{
    uint pixel_index;
    uint pad;
};

[shader("anyhit")]
void main(inout GsPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    const float new_t  = RayTCurrent();
    const uint  new_id = uint(InstanceID());
    const uint  base   = payload.pixel_index * GS_KBUF_K;

    // Slot K-1 is the worst kept; if our t is worse than that, drop.
    if (gs_kbuf_dist[base + (GS_KBUF_K - 1u)] <= new_t) {
        IgnoreHit();
        return;
    }

    // Insertion sort: walk down from K-1, shifting entries with t > new_t
    // one slot to the right; insert at the first position whose left
    // neighbour is <= new_t (or at slot 0 if none).
    uint k = GS_KBUF_K - 1u;
    while (k > 0u) {
        const float prev = gs_kbuf_dist[base + (k - 1u)];
        if (prev <= new_t) break;
        gs_kbuf_dist[base + k] = prev;
        gs_kbuf_id  [base + k] = gs_kbuf_id[base + (k - 1u)];
        k -= 1u;
    }
    gs_kbuf_dist[base + k] = new_t;
    gs_kbuf_id  [base + k] = new_id;

    IgnoreHit();
}
