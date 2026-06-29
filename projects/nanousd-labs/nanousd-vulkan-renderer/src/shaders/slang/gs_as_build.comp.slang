// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// gs_as_build.comp.slang — GPU-driven TLAS instance builder for VK3DGRT.
//
// One thread per particle. Reads from particle SSBOs (set 0) + push-constant
// header, writes a VkAccelerationStructureInstanceKHR record to the output
// instance buffer. The transform is M = T(pos) · prim_xform · R(quat)
// · diag(scale · kernelScale · proxyExtent), packed into the 3x4 row-major
// `transform` field VkAccelerationStructureInstanceKHR expects.
//
// See docs/plans/VK3DGRT_PLAN.md §5/§6. Phase 2.

#include "gs_common.h.slang"

[[vk::push_constant]] GsAsBuildPC pc;

// Vulkan's VkAccelerationStructureInstanceKHR is 64 bytes:
//   float3x4 transform                                   (48 B, row-major)
//   uint     instanceCustomIndexAndMask                  (4 B)
//   uint     instanceShaderBindingTableRecordOffsetAndFlags (4 B)
//   uint64   accelerationStructureReference              (8 B)
struct AsInstance
{
    float4 row0;
    float4 row1;
    float4 row2;
    uint   custom_index_and_mask;
    uint   sbt_offset_and_flags;
    uint2  blas_ref; // uint64 split lo/hi
};

// Per-particle SSBOs (host populates from gs_scene_*).
[[vk::binding(0, 0)]] StructuredBuffer<float3>     positions;
[[vk::binding(1, 0)]] StructuredBuffer<float3>     scales;
[[vk::binding(2, 0)]] StructuredBuffer<float4>     orientations; // wxyz
[[vk::binding(3, 0)]] StructuredBuffer<float>      opacities;
[[vk::binding(4, 0)]] StructuredBuffer<float>      kernel_scales;
[[vk::binding(5, 0)]] RWStructuredBuffer<AsInstance> instances;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint i = tid.x;
    if (i >= pc.particle_count) return;

    const float3 pos    = positions[i];
    const float3 scl    = scales[i];
    const float4 quat   = orientations[i];
    const float  kerS   = kernel_scales[i];
    const float  proxyE = (pc.proxy_kind == GS_PROXY_ICOSAHEDRON)
                          ? GS_ICOSA_VRT_SCALE
                          : GS_AABB_VRT_SCALE;

    // Build local transform: M_local = R(quat) * diag(scale * kerS * proxyE)
    const float3   diag = scl * kerS * proxyE;
    const float3x3 R    = gs_quat_to_mat3(quat);

    float3x3 M3 = float3x3(
        R[0][0] * diag.x, R[0][1] * diag.y, R[0][2] * diag.z,
        R[1][0] * diag.x, R[1][1] * diag.y, R[1][2] * diag.z,
        R[2][0] * diag.x, R[2][1] * diag.y, R[2][2] * diag.z
    );

    // Compose: world = prim_xform * T(pos) * M_local
    float4x4 local = float4x4(
        float4(M3[0], pos.x),
        float4(M3[1], pos.y),
        float4(M3[2], pos.z),
        float4(0, 0, 0, 1)
    );
    float4x4 world = mul(pc.prim_xform, local);

    AsInstance inst;
    inst.row0 = world[0];
    inst.row1 = world[1];
    inst.row2 = world[2];
    inst.custom_index_and_mask = (i & 0x00FFFFFFu) | (0xFFu << 24); // mask = 0xFF
    // VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR = 0x08 in the flags byte
    // (high byte of sbt_offset_and_flags). Reference vk_gaussian_splatting
    // sets this on every instance — it forces the driver to invoke any-hit
    // even on BLASes the implementation might otherwise treat as opaque.
    inst.sbt_offset_and_flags  = (0x08u << 24);
    inst.blas_ref = uint2(uint(pc.blas_address & 0xFFFFFFFFull),
                          uint((pc.blas_address >> 32) & 0xFFFFFFFFull));

    // The host binds `instances` to the same VkBuffer that backs the TLAS
    // VkAccelerationStructureGeometryInstancesDataKHR.data — bypassing
    // a host transfer of the per-frame instance array.
    instances[i] = inst;
}
