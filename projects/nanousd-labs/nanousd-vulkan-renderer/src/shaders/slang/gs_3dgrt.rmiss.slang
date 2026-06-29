// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// gs_3dgrt.rmiss.slang — VK3DGRT miss (Chunk B: no-op).
//
// rgen handles the "all-INVALID K-buffer → composite with sky" path
// directly, so miss is just a marker. The Pragmatist gate's payload
// fields (radiance/hit_t) are gone; the K-buffer payload is opaque
// state for rahit only.

#include "gs_common.h.slang"

struct GsPayload
{
    uint pixel_index;
    uint pad;
};

[shader("miss")]
void main(inout GsPayload payload)
{
}
