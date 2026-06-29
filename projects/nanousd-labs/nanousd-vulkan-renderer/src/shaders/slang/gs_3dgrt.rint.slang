// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// gs_3dgrt.rint.slang — VK3DGRT intersection (AABB proxy mode only).
//
// For NU_GS_PROXY_AABB the BLAS is procedural; this shader reports a hit
// at the closest point along the ray to the origin in the particle's
// canonical/object space. Phase 2 stub: trivial accept. Phase 3 will
// implement the m^2(t) projection from plan §6.

#include "gs_common.h.slang"

struct GsHitAttribs
{
    float dummy;
};

[shader("intersection")]
void main()
{
    // Phase 2: report a hit at the ray's tMin in object space — enough to
    // exercise the AABB-proxy hit-group binding without the projection
    // math. Phase 3 replaces with:
    //   o = ObjectRayOrigin(); d = ObjectRayDirection();
    //   t = -dot(o,d) / dot(d,d);
    //   reject if outside [tMin,tMax] or m^2(t) > 9
    GsHitAttribs attr;
    attr.dummy = 0.0f;
    ReportHit(RayTMin(), 0u, attr);
}
