// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_scalar_block_layout : enable

/* tlas_translate_partitioned.comp.glsl — Phase C
 *
 * Per-frame transform write into the per-env partitioned instance
 * buffer (gpu->tlas_arr_inst_buf). Runs in addition to the legacy
 * tlas_translate.comp dispatch — the warp kernel has already written
 * row-major 4x4 matrices into the CUDA-imported transforms buffer,
 * and the legacy translate has filled gpu->instance_buf with the
 * 12-float upper rows in legacy order. This pass copies those same
 * transforms into gpu->tlas_arr_inst_buf using a two-step indirect
 * lookup: gid → legacy_slot (via tlas_indices) → partitioned_slot
 * (via inst_to_partial).
 *
 * Bindings:
 *   set 0 binding 0 — readonly  Transforms     : float[gid*16 + 0..15]
 *   set 0 binding 1 —          PartitionedBuf : uint[partitioned_slot*16 + 0..15]
 *                     reinterpreted as VkAccelerationStructureInstanceKHR records
 *   set 0 binding 2 — readonly  TlasIndices    : int[gid]
 *   set 0 binding 3 — readonly  InstToPartial  : uint[legacy_slot]
 *
 * Push constant: total dispatch count (n_valid). */

#ifndef NUSD_TLAS_TRANSLATE_LOCAL_X
#define NUSD_TLAS_TRANSLATE_LOCAL_X 64
#endif
layout(local_size_x = NUSD_TLAS_TRANSLATE_LOCAL_X) in;

layout(set = 0, binding = 0, std430) readonly buffer TransformsBuffer {
    float in_xforms[];
} transforms;

layout(set = 0, binding = 1, std430) buffer PartitionedInstanceBuffer {
    uint partitioned_buf[];
} partitioned;

layout(set = 0, binding = 2, std430) readonly buffer TlasIndicesBuffer {
    int tlas_indices[];
} tlas_idx;

layout(set = 0, binding = 3, std430) readonly buffer InstToPartialBuffer {
    uint inst_to_partial[];
} mapping;

layout(push_constant) uniform PushConstants {
    uint count;
} pc;

void main()
{
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.count) return;

    int legacy = tlas_idx.tlas_indices[gid];
    if (legacy < 0) return;

    uint partitioned_slot = mapping.inst_to_partial[uint(legacy)];
    if (partitioned_slot == 0xFFFFFFFFu) return;

    uint src  = gid * 16u;
    uint base = partitioned_slot * 16u;

    for (uint i = 0u; i < 12u; ++i) {
        partitioned.partitioned_buf[base + i] = floatBitsToUint(transforms.in_xforms[src + i]);
    }
}
