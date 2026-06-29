// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// gs_3dgrt.rchit.slang — VK3DGRT closest-hit (no-op for K-buffer pipeline).
//
// rgen integrates radiance directly from the K-buffer; rahit IgnoreHit's
// every candidate so traversal exhausts without accepting any hit, which
// means this shader is never invoked at runtime. We keep it in the
// pipeline because some Vulkan implementations gate any-hit invocations
// on a hit group having a closest-hit slot. The payload struct must
// match rgen / rahit / rmiss for pipeline linkage.

#include "gs_common.h.slang"

struct GsPayload
{
    uint pixel_index;
    uint pad;
};

[shader("closesthit")]
void main(inout GsPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
}
