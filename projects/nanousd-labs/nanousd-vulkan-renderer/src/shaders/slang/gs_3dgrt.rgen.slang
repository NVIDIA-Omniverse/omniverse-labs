// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// gs_3dgrt.rgen.slang — VK3DGRT raygen (multi-pass K-buffer trace+composite).
//
// Plan §6 algorithm (single-camera pinhole). For each pixel:
//
//   transmittance = 1
//   tMin = 0.001
//   loop pass < maxPasses, transmittance > minT, tMin < tMax:
//       clear K-buffer
//       TraceRay(tMin, tMax)               // rahit insertion-sorts hits
//       walk K-buffer front-to-back:
//           load particle from SSBOs by InstanceID
//           m² = canonical-space squared Mahalanobis from ray-closest-pt
//           skip if m² > 9
//           color = SH-DC eval (higher bands TODO)
//           α = clamp(opacity * exp(-0.5 m²), 0, alphaClamp)
//           skip if α < 1/255
//           radiance      += transmittance * color * α
//           transmittance *= 1 - α
//           lastDist       = kbuf[k].dist
//           break inner if transmittance < minT
//       if no hits this pass: break outer
//       tMin = max(tMin, lastDist) + epsT
//   composite residual transmittance with sky → output

#include "gs_common.h.slang"

[[vk::push_constant]] GsRtPC pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] RWTexture2D<float4> outImage;

[[vk::binding(2, 0)]] StructuredBuffer<float3> gs_positions;
[[vk::binding(3, 0)]] StructuredBuffer<float3> gs_scales;
[[vk::binding(4, 0)]] StructuredBuffer<float4> gs_orientations;
[[vk::binding(5, 0)]] StructuredBuffer<float>  gs_opacities;
[[vk::binding(6, 0)]] StructuredBuffer<float>  gs_kernel_scales;
[[vk::binding(7, 0)]] StructuredBuffer<float>  gs_sh;

[[vk::binding(8, 0)]]  RWStructuredBuffer<uint>  gs_kbuf_id;
[[vk::binding(9, 0)]]  RWStructuredBuffer<float> gs_kbuf_dist;
[[vk::binding(10, 0)]] RWStructuredBuffer<float> gs_depth;
[[vk::binding(11, 0)]] RWStructuredBuffer<float> gs_normal;  // 3 floats / pixel

struct GsPayload
{
    uint pixel_index;
    uint pad;
};

static const float GS_EPS_T = 1e-4f;

[shader("raygeneration")]
void main()
{
    const uint2 pid       = DispatchRaysIndex().xy;
    const uint2 dims      = DispatchRaysDimensions().xy;
    const uint  pixel_idx = pid.y * dims.x + pid.x;
    const uint  base      = pixel_idx * GS_KBUF_K;

    // Pinhole camera ray (constant across passes).
    const float2 uv  = (float2(pid) + 0.5f) / float2(dims);
    const float2 ndc = float2(uv.x * 2.0f - 1.0f, uv.y * 2.0f - 1.0f);

    const float4 origin_h  = mul(pc.view_inverse, float4(0, 0, 0, 1));
    const float4 target_v  = mul(pc.proj_inverse, float4(ndc.x, ndc.y, 1, 1));
    const float3 dir_view  = normalize(target_v.xyz / target_v.w);
    const float4 dir_world = mul(pc.view_inverse, float4(dir_view, 0));
    const float3 ray_o = origin_h.xyz;
    const float3 ray_d = normalize(dir_world.xyz);

    // Plan §11 D4: rgen accumulators run in double precision. At α=0.05,
    // (1-α)^N underflows to denormal in float32 around N=140 hits;
    // dense scenes pass that easily and silently produce black pixels
    // that no goldens flag. Per-particle math (color, alpha) stays in
    // float; only the running products promote.
    double3     radiance      = double3(0.0, 0.0, 0.0);
    double      transmittance = 1.0;
    float       t_min         = 0.001f;
    const float t_max         = 10000.0f;
    const double min_T        = max((double)pc.min_transmittance, 1e-12);
    const uint  max_passes    = max(1u, pc.max_passes);
    const uint  N             = max(1u, pc.particle_count);

    // Iso-opacity surface depth (plan §7). Record the t at which
    // accumulated opacity (1 - transmittance) first crosses
    // pc.iso_opacity_threshold. -1 = never crossed (e.g., sky pixel
    // or scene with insufficient cumulative opacity).
    //
    // Iso-opacity normal: analytic gradient of the per-particle density
    // contribution at the iso-crossing point, in world space. For a
    // Gaussian g(x) = α · exp(-0.5·m²) the gradient w.r.t. world-space
    // position is g · (-Σ⁻¹ · (x - center)), where Σ⁻¹ in world space
    // is `R · diag(1/scale²) · R^T`. Plan §7 calls this the
    // "max-density-plane normal" — for a single dominant Gaussian it
    // matches the surface normal of the level-set.
    float  depth_t            = -1.0f;
    float3 iso_normal         = float3(0.0f, 0.0f, 0.0f);
    const double iso_opacity  = (double)pc.iso_opacity_threshold;

    for (uint pass = 0u; pass < max_passes; pass++) {
        if (transmittance < min_T) break;
        if (t_min >= t_max) break;

        // Clear this pass's K-buffer slots.
        [unroll]
        for (uint k = 0u; k < GS_KBUF_K; k++) {
            gs_kbuf_id  [base + k] = GS_INVALID_ID;
            gs_kbuf_dist[base + k] = 1e30f;
        }

        RayDesc ray;
        ray.Origin    = ray_o;
        ray.Direction = ray_d;
        ray.TMin      = t_min;
        ray.TMax      = t_max;

        GsPayload payload;
        payload.pixel_index = pixel_idx;
        payload.pad         = 0;

        // RAY_FLAG_CULL_BACK_FACING_TRIANGLES: matches reference
        // vk_gaussian_splatting/threedgrt_raytrace.rgen.slang:153. Without
        // this, the icosahedron proxy is hit twice per ray (front + back
        // face), filling the K-buffer with duplicates that get
        // double-counted in the alpha composite — visible as bright
        // streak artifacts on real-world dense PLY data.
        // RAY_FLAG_SKIP_CLOSEST_HIT_SHADER: our rchit is a no-op kept only
        // to satisfy drivers that gate any-hit on closest-hit existing.
        // Skipping its invocation is a free perf win.
        const uint rt_flags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES
                            | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
        TraceRay(tlas, rt_flags, 0xFFu, 0u, 0u, 0u, ray, payload);

        // Walk the (sorted-by-t) K-buffer, α-composite. Track the
        // last *visited* (not just *accepted*) distance — reference
        // vk_gaussian_splatting/threedgrt_raytrace.rgen.slang advances
        // tMin to the max distance among ALL K-buffer entries processed
        // each pass (accepted or rejected). Without this, dense scenes
        // re-visit rejected hits in later passes, exhausting the K
        // budget before the trace reaches genuinely-new contributors.
        float last_dist = t_min;
        bool  any_kept  = false;

        for (uint k = 0u; k < GS_KBUF_K; k++) {
            const uint id = gs_kbuf_id[base + k];
            if (id == GS_INVALID_ID) break;
            // Always advance the per-pass cursor to the current entry's t,
            // even if we end up rejecting/skipping it below.
            last_dist = gs_kbuf_dist[base + k];

            // Load particle.
            const float3 center = gs_positions  [id];
            const float3 scl    = gs_scales     [id];
            const float4 quat   = gs_orientations[id];
            const float  opa    = gs_opacities  [id];
            const uint   sh_per = (pc.sh_degree + 1u) * (pc.sh_degree + 1u);
            const uint   sh_b   = id * sh_per * 3u;

            // Ray in canonical (Gaussian) space: oc = R^T · ((o - c) / scale).
            // Normalize the canonical direction before computing m² —
            // the reference renderer does this (vk_gaussian_splatting/
            // shaders/threedgrt.h.slang::particleCannonicalRay), and on
            // highly anisotropic splats (1:1000+ aspect) the unnormalized
            // |dc|² in the denominator loses precision, producing the
            // long colored streaks visible on real-world trained PLYs.
            const float3x3 R     = gs_quat_to_mat3(quat);
            const float3x3 R_T   = transpose(R);
            const float3   inv_s = float3(1.0f / max(scl.x, 1e-12f),
                                           1.0f / max(scl.y, 1e-12f),
                                           1.0f / max(scl.z, 1e-12f));
            const float3 oc = mul(R_T, ray_o - center) * inv_s;
            const float3 dc_raw = mul(R_T, ray_d)      * inv_s;
            const float3 dc = normalize(dc_raw);

            // |cross(dc_unit, oc)|² = perpendicular squared distance from
            // canonical origin to the canonical-space ray.
            const float3 cr = cross(dc, oc);
            const float  m2 = dot(cr, cr);
            if (m2 > GS_MAX_PARTICLE_SQ_DISTANCE) continue;

            // SH at-hit (plan §6 D1): view-dir = normalize(particle - rayOrigin).
            // For sh_degree == 0 we shortcut to the DC-only branch — same
            // bytes touched as before, no perf regression for the common case.
            float3 color;
            if (pc.sh_degree == 0u) {
                const float3 sh0 = float3(gs_sh[sh_b], gs_sh[sh_b + 1u], gs_sh[sh_b + 2u]);
                color = gs_eval_sh_dc(sh0);
            } else {
                const float3 view_dir = normalize(center - ray_o);
                color = gs_eval_sh(pc.sh_degree, sh_b, view_dir, gs_sh);
            }
            color = max(float3(0.0f, 0.0f, 0.0f), color);
            const float  alpha = clamp(opa * exp(-0.5f * m2),
                                       0.0f, GS_ALPHA_CLAMP);
            if (alpha < GS_ALPHA_CULL_THRESHOLD) continue;

            // Capture iso-opacity surface depth + normal (plan §7) the
            // first time accumulated opacity (1 - transmittance) crosses
            // the threshold. Multiplications run in double; SH color and
            // α stay float so per-particle work doesn't pay the
            // double-precision cost.
            const double prev_T = transmittance;
            transmittance      *= (double)(1.0f - alpha);
            radiance           += prev_T * (double3)color * (double)alpha;
            any_kept            = true;
            if (depth_t < 0.0f && (1.0 - transmittance) >= iso_opacity) {
                depth_t = last_dist;
                // Max-density-plane normal (StochasticSplats / reference
                // computeEllipsoidNormalMaxDensityPlane). Density gradient
                // in world is ∝ Σ⁻¹·(x - center), where Σ⁻¹ in world is
                // R · diag(1/σ²) · R^T. Recompute the canonical offset
                // from the world-space hit point — we no longer cache
                // `closest` in canonical units.
                const float3 hit_world  = ray_o + last_dist * ray_d;
                const float3 local_vec  = hit_world - center;
                const float3 canon_off  = mul(R_T, local_vec) * inv_s;
                const float3 grad_world = mul(R, canon_off * inv_s);
                const float3 n          = -grad_world;
                const float  n_len      = length(n);
                if (n_len > 1e-12f) {
                    iso_normal = n / n_len;
                }
            }
            if (transmittance < min_T) break;
        }

        if (!any_kept) break;
        t_min = max(t_min, last_dist) + GS_EPS_T;
    }

    // Composite residual transmittance with sky. Cast double accumulators
    // back to float32 only at the imageStore boundary.
    const double3 sky = double3(0.05, 0.07, 0.10);
    radiance += transmittance * sky;

    outImage[pid]                  = float4((float3)radiance, 1.0f);
    gs_depth[pixel_idx]            = depth_t;
    gs_normal[pixel_idx * 3u + 0u] = iso_normal.x;
    gs_normal[pixel_idx * 3u + 1u] = iso_normal.y;
    gs_normal[pixel_idx * 3u + 2u] = iso_normal.z;
}
