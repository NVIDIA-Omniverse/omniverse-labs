// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_scalar_block_layout : enable

/*
 * tlas_translate.comp.glsl — Translate per-shape world transforms into
 * VkAccelerationStructureInstanceKHR records on the GPU.
 *
 * Eliminates two host-side costs that previously dominated frame time at
 * 4096 envs:
 *   1. The wp-kernel→numpy sync (`.numpy()`) used to drain GPU transforms
 *      back into a host array (~5-8 ms / frame at 4096 envs).
 *   2. The per-instance host loop in gpu_update_tlas_inline that wrote
 *      VkAccelerationStructureInstanceKHR records into `instance_buf`.
 *
 * The warp kernel writes a dense (n_valid, 16) row-major matrix array into
 * a CUDA-imported Vulkan storage buffer; this compute pass reads each
 * thread's 4x4, copies the upper 3x4 (12 floats) into `instance_buf`, and
 * leaves the per-instance metadata bytes 48..63 (customIndex, mask,
 * sbtOffset, flags, AS reference) untouched — they were written once by
 * the initial gpu_update_tlas() during scene init and remain valid.
 *
 * Layout of VkAccelerationStructureInstanceKHR (64 bytes):
 *   offset  0..47 : float[3][4] transform (row-major 3x4)
 *   offset 48..51 : uint   instanceCustomIndexAndMask  (24|8 bits)
 *   offset 52..55 : uint   sbtOffsetAndFlags           (24|8 bits)
 *   offset 56..63 : uint64 accelerationStructureReference
 *
 * Bindings:
 *   set 0 binding 0 — readonly  Transforms : float[gid*16 + 0..15]
 *                     row-major 4x4 matrices output by _compute_transforms_kernel
 *   set 0 binding 1 —          InstanceBuf : uint[tlas_idx*16 + 0..15]
 *                     reinterpreted as the 64-byte VkAccelerationStructureInstanceKHR records
 *   set 0 binding 2 — readonly  TlasIndices : int[gid]
 *                     mapping (gid → TLAS instance index) so the kernel writes
 *                     the right slot. Set once at scene init via
 *                     nu_set_transform_layout().
 *
 * Push constant: total dispatch count (n_valid).
 */

#ifndef NUSD_TLAS_TRANSLATE_LOCAL_X
#define NUSD_TLAS_TRANSLATE_LOCAL_X 64
#endif
layout(local_size_x = NUSD_TLAS_TRANSLATE_LOCAL_X) in;

layout(set = 0, binding = 0, std430) readonly buffer TransformsBuffer {
    float in_xforms[];
} transforms;

layout(set = 0, binding = 1, std430) buffer InstanceBuffer {
    uint instance_buf[];
} instance;

layout(set = 0, binding = 2, std430) readonly buffer TlasIndicesBuffer {
    int tlas_indices[];
} tlas_idx;

layout(push_constant) uniform PushConstants {
    uint count;
} pc;

void main()
{
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.count) return;

    int dst = tlas_idx.tlas_indices[gid];
    if (dst < 0) return;

    uint src = gid * 16u;
    uint base = uint(dst) * 16u;

    /* Copy 12 floats (3x4 row-major) — bytes 0..47 of the 64-byte record.
     * The kernel writes a row-major 4x4 with row 3 = (0,0,0,1); we drop
     * row 3 by stopping at index 11. floatBitsToUint preserves the bit
     * pattern through the uint-typed storage buffer. */
    for (uint i = 0u; i < 12u; ++i) {
        instance.instance_buf[base + i] = floatBitsToUint(transforms.in_xforms[src + i]);
    }
}
