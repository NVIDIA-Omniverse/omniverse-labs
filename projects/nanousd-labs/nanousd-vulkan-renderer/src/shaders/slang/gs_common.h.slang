// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// gs_common.h.slang — VK3DGRT shared shader types, constants, and bindings.
//
// Phase 2/3 of docs/plans/VK3DGRT_PLAN.md. The binding layout here is
// authoritative: gpu_vulkan.c builds VkDescriptorSetLayoutBinding entries
// at the same indices.

#ifndef GS_COMMON_H_SLANG
#define GS_COMMON_H_SLANG

// ---- Constants (must match gs_scene.h) ----
static const float GS_KERNEL_MIN_RESPONSE       = 0.0113f;       // exp(-0.5 * 9)
static const float GS_MAX_PARTICLE_SQ_DISTANCE  = 9.0f;          // 3-sigma cutoff
static const float GS_ALPHA_CLAMP               = 0.99f;
static const float GS_ALPHA_CULL_THRESHOLD      = (1.0f / 255.0f);
static const int   GS_KERNEL_DEGREE_DEFAULT     = 2;             // quadratic Gaussian
static const float GS_PI                        = 3.14159265358979323846f;

// Proxy kinds (NuGsProxyKind).
static const uint GS_PROXY_ICOSAHEDRON = 0u;
static const uint GS_PROXY_AABB        = 1u;

// Color spaces (NuGsColorSpace).
static const uint GS_COLOR_LINEAR = 0u;
static const uint GS_COLOR_SRGB   = 1u;

// Camera models (NuGsCameraModel).
static const uint GS_CAMERA_PINHOLE  = 0u;
static const uint GS_CAMERA_FISHEYE  = 1u;
static const uint GS_CAMERA_EQUIRECT = 2u;

// Reserved sentinels.
static const uint GS_INVALID_ID = 0xFFFFFFFFu;

// K-buffer width (must match GS_KBUF_K in gpu_vulkan.c). Plan §6 default
// was 16; reference vk_gaussian_splatting uses PARTICLES_SPP=18 (in-payload
// K-buffer). Bumping to 18 brings us numerically in-line with the reference.
static const uint GS_KBUF_K = 18u;

// Unit-icosahedron proxy: vertex/edge mapped so max extent fits [-1,1]^3.
// Match reference vk_gaussian_splatting/shaders/particle_as_build.comp.slang.
static const float GS_ICOSA_VRT_SCALE = 1.0704662767622f;
static const float GS_AABB_VRT_SCALE  = 1.0f;

// ---- Push constants ----

// Compute (gs_as_build): builds VkAccelerationStructureInstanceKHR records
// from the particle SSBOs. One thread per particle.
struct GsAsBuildPC
{
    uint64_t blas_address;       // unit-BLAS device address (icosahedron or AABB)
    uint64_t instance_address;   // output VkAccelerationStructureInstanceKHR[N] dst
    uint     particle_count;
    uint     proxy_kind;         // GS_PROXY_*
    uint     kernel_degree;      // 0 (linear) or 2 (quadratic)
    uint     adaptive_clamping;  // 0 or 1
    float4x4 prim_xform;         // optional per-prim world transform (row-major)
};

// RT pipeline (gs_3dgrt): per-camera + per-frame controls. Sized to fit
// the 128-byte push-constant block we use elsewhere in the renderer.
struct GsRtPC
{
    float4x4 view_inverse;
    float4x4 proj_inverse;
    uint     particle_count;
    uint     sh_degree;          // 0..3
    uint     K;                  // K-buffer size (8/16/32)
    uint     max_passes;
    uint     camera_model;       // GS_CAMERA_*
    uint     color_space;        // GS_COLOR_*
    float    min_transmittance;
    float    iso_opacity_threshold;
};

// ---- Helpers ----

// Quaternion (wxyz) → 3x3 rotation. Matches the OpenUSD column-major
// orientation convention used elsewhere in the renderer.
float3x3 gs_quat_to_mat3(float4 q)
{
    q = normalize(q);
    float w = q.x, x = q.y, y = q.z, z = q.w;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return float3x3(
        1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),
        2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),
        2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)
    );
}

// Adaptive kernelScale — matches gs_compute_kernel_scale() in src/gs_scene.c
// and reference vk_gaussian_splatting/src/splat_set_vk.cpp::kernelScale.
float gs_kernel_scale(float density, float min_response, uint kernel_degree, bool adaptive)
{
    float modulation = adaptive ? density : 1.0f;
    float clamped    = min(min_response / max(modulation, 1e-6f), 0.97f);

    if (kernel_degree == 0u) {
        return ((1.0f - clamped) / 3.0f) / -0.329630334487f;
    }
    float b = float(kernel_degree);
    float a = -4.5f / pow(3.0f, b);
    return pow(log(clamped) / a, 1.0f / b);
}

// Spherical harmonics evaluation up to band 3 (16 coefficients per channel).
//
// Per-particle radiance:
//   L = 0.5 + C0 · sh[0]                      (DC, with the 3DGS bias)
//     + C1 · (-y · sh[1] + z · sh[2] - x · sh[3])               (band 1)
//     + Σ band 2 (5 terms)
//     + Σ band 3 (7 terms)
// where (x, y, z) is the (already-normalized) view direction in world
// space. Plan §6 says use `dir = normalize(particle.center - rayOrigin)`
// (camera-to-particle), i.e., the direction from which the splat is
// "looking back" at the camera. Constants match
// references/vk_gaussian_splatting/shaders/threedgs_particle_storage.h.slang.
//
// `sh_base` is the float-index into the gs_sh SSBO where this particle's
// SH block starts: `particle_id * (deg+1)^2 * 3`. We pass the buffer
// through `[[ForceInline]]` semantics by macro-ifying via Slang's
// generic `RWStructuredBuffer<float>` type — but Slang allows passing
// a StructuredBuffer as a function parameter, so we use that.
static const float GS_SH_C0    = 0.28209479177387814f;        // 1 / (2·sqrt(pi))
static const float GS_SH_C1    = 0.4886025119029199f;          // sqrt(3 / (4·pi))
static const float GS_SH_C2[5] = { 1.0925484f, -1.0925484f, 0.3153916f,
                                    -1.0925484f, 0.5462742f };
static const float GS_SH_C3[7] = { -0.5900435899266435f, 2.890611442640554f,
                                    -0.4570457994644658f, 0.3731763325901154f,
                                    -0.4570457994644658f, 1.445305721320277f,
                                    -0.5900435899266435f };

// DC-only path retained for sh_degree == 0 (simplest fast-path).
float3 gs_eval_sh_dc(float3 sh0)
{
    return float3(0.5f, 0.5f, 0.5f) + GS_SH_C0 * sh0;
}

// Full SH evaluation: bands 0..deg. `sh` is the global SSBO
// (RWStructuredBuffer because the binding is RW; reads are fine);
// `sh_base` is the starting float-index for this particle's coefficients.
// `dir` is the world-space view direction (camera → particle), already
// normalized.
float3 gs_eval_sh(uint deg, uint sh_base, float3 dir,
                  StructuredBuffer<float> sh)
{
    // Band 0 (DC).
    float3 c0 = float3(sh[sh_base + 0u], sh[sh_base + 1u], sh[sh_base + 2u]);
    float3 rgb = float3(0.5f, 0.5f, 0.5f) + GS_SH_C0 * c0;
    if (deg < 1u) return rgb;

    const float x = dir.x, y = dir.y, z = dir.z;

    // Band 1 (3 coefficients at offsets 3..11).
    float3 c1_0 = float3(sh[sh_base + 3u],  sh[sh_base + 4u],  sh[sh_base + 5u]);
    float3 c1_1 = float3(sh[sh_base + 6u],  sh[sh_base + 7u],  sh[sh_base + 8u]);
    float3 c1_2 = float3(sh[sh_base + 9u],  sh[sh_base + 10u], sh[sh_base + 11u]);
    rgb += GS_SH_C1 * (-c1_0 * y + c1_1 * z - c1_2 * x);
    if (deg < 2u) return rgb;

    // Band 2 (5 coefficients at offsets 12..26).
    const float xx = x*x, yy = y*y, zz = z*z;
    const float xy = x*y, yz = y*z, xz = x*z;
    float3 c2_0 = float3(sh[sh_base + 12u], sh[sh_base + 13u], sh[sh_base + 14u]);
    float3 c2_1 = float3(sh[sh_base + 15u], sh[sh_base + 16u], sh[sh_base + 17u]);
    float3 c2_2 = float3(sh[sh_base + 18u], sh[sh_base + 19u], sh[sh_base + 20u]);
    float3 c2_3 = float3(sh[sh_base + 21u], sh[sh_base + 22u], sh[sh_base + 23u]);
    float3 c2_4 = float3(sh[sh_base + 24u], sh[sh_base + 25u], sh[sh_base + 26u]);
    rgb += (GS_SH_C2[0] * xy)                     * c2_0
         + (GS_SH_C2[1] * yz)                     * c2_1
         + (GS_SH_C2[2] * (2.0f * zz - xx - yy))  * c2_2
         + (GS_SH_C2[3] * xz)                     * c2_3
         + (GS_SH_C2[4] * (xx - yy))              * c2_4;
    if (deg < 3u) return rgb;

    // Band 3 (7 coefficients at offsets 27..47).
    float3 c3_0 = float3(sh[sh_base + 27u], sh[sh_base + 28u], sh[sh_base + 29u]);
    float3 c3_1 = float3(sh[sh_base + 30u], sh[sh_base + 31u], sh[sh_base + 32u]);
    float3 c3_2 = float3(sh[sh_base + 33u], sh[sh_base + 34u], sh[sh_base + 35u]);
    float3 c3_3 = float3(sh[sh_base + 36u], sh[sh_base + 37u], sh[sh_base + 38u]);
    float3 c3_4 = float3(sh[sh_base + 39u], sh[sh_base + 40u], sh[sh_base + 41u]);
    float3 c3_5 = float3(sh[sh_base + 42u], sh[sh_base + 43u], sh[sh_base + 44u]);
    float3 c3_6 = float3(sh[sh_base + 45u], sh[sh_base + 46u], sh[sh_base + 47u]);
    rgb += GS_SH_C3[0] * c3_0 * (3.0f * xx - yy) * y
         + GS_SH_C3[1] * c3_1 * x * y * z
         + GS_SH_C3[2] * c3_2 * (4.0f * zz - xx - yy) * y
         + GS_SH_C3[3] * c3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy)
         + GS_SH_C3[4] * c3_4 * x * (4.0f * zz - xx - yy)
         + GS_SH_C3[5] * c3_5 * (xx - yy) * z
         + GS_SH_C3[6] * c3_6 * x * (xx - 3.0f * yy);
    return rgb;
}

#endif // GS_COMMON_H_SLANG
