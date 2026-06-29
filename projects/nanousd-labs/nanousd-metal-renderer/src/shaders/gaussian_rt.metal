// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * gaussian_rt.metal — hardware-RT assisted Gaussian splat renderer.
 *
 * The host builds one AABB BLAS over world-space ellipsoid bounds. This
 * kernel ray-queries those AABBs, evaluates the analytic ellipsoid Gaussian
 * at each ray's closest approach, keeps the nearest K contributors, and
 * front-to-back composites SH color. It follows the same camera matrix
 * convention as raytrace.metal.
 */

#include <metal_stdlib>
#include <metal_raytracing>

using namespace metal;
using namespace raytracing;

struct GsPushConstants {
    float4x4 view_inv;
    float4x4 proj_inv;
    uint     particle_count;
    uint     sh_degree;
    uint     k;
    uint     max_passes;
    float    min_transmittance;
    float    iso_opacity_threshold;
    uint     color_space;
    uint     _pad0;
};

struct GsParticle {
    float4 center_opacity;
    float4 axis0_inv;
    float4 axis1_inv;
    float4 axis2_inv;
    float4 kernel_pad;
};

constant uint GS_MAX_K = 32u;
constant float GS_ALPHA_CLAMP = 0.99;
constant float GS_ALPHA_CULL = 1.0 / 255.0;

struct GsHit {
    float t;
    float m2;
    float alpha;
    uint  id;
};

inline float3 srgb_to_linear(float3 c)
{
    float3 lo = c / 12.92;
    float3 hi = pow(max((c + 0.055) / 1.055, float3(0.0)), float3(2.4));
    return select(hi, lo, c <= float3(0.04045));
}

inline float3 eval_sh(device const float* sh, uint id, uint degree, float3 dir)
{
    uint sh_per = (degree + 1u) * (degree + 1u);
    uint base = id * sh_per * 3u;
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    /* nvpro/vk_gaussian_splatting packs degree-0 color separately as
     * clamp(0.5 + C0 * f_dc, 0, 1), then adds view-dependent SH rest. */
    float3 result = clamp(
        0.5 + 0.28209479177387814 * float3(sh[base + 0u], sh[base + 1u], sh[base + 2u]),
        float3(0.0), float3(1.0));

    if (degree >= 1u) {
        uint b = base + 3u;
        result += -0.4886025119029199 * y * float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
        b += 3u;
        result +=  0.4886025119029199 * z * float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
        b += 3u;
        result += -0.4886025119029199 * x * float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
    }

    if (degree >= 2u) {
        uint b = base + 12u;
        result +=  1.0925484305920792 * x * y * float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
        b += 3u;
        result += -1.0925484305920792 * y * z * float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
        b += 3u;
        result +=  0.31539156525252005 * (2.0 * z * z - x * x - y * y) *
                   float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
        b += 3u;
        result += -1.0925484305920792 * x * z * float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
        b += 3u;
        result +=  0.5462742152960396 * (x * x - y * y) * float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
    }

    if (degree >= 3u) {
        uint b = base + 27u;
        result += -0.5900435899266435 * y * (3.0 * x * x - y * y) *
                  float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
        b += 3u;
        result +=  2.890611442640554 * x * y * z * float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
        b += 3u;
        result += -0.4570457994644658 * y * (4.0 * z * z - x * x - y * y) *
                  float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
        b += 3u;
        result +=  0.3731763325901154 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y) *
                  float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
        b += 3u;
        result += -0.4570457994644658 * x * (4.0 * z * z - x * x - y * y) *
                  float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
        b += 3u;
        result +=  1.445305721320277 * z * (x * x - y * y) *
                  float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
        b += 3u;
        result += -0.5900435899266435 * x * (x * x - 3.0 * y * y) *
                  float3(sh[b + 0u], sh[b + 1u], sh[b + 2u]);
    }

    return max(result, float3(0.0));
}

inline bool eval_gaussian_hit(const ray r,
                              device const GsParticle* particles,
                              uint id,
                              thread GsHit& out_hit)
{
    GsParticle p = particles[id];
    float3 center = p.center_opacity.xyz;
    float3 axis0 = p.axis0_inv.xyz;
    float3 axis1 = p.axis1_inv.xyz;
    float3 axis2 = p.axis2_inv.xyz;
    float inv_s0 = p.axis0_inv.w;
    float inv_s1 = p.axis1_inv.w;
    float inv_s2 = p.axis2_inv.w;
    float3 rel0 = r.origin - center;
    float d0 = dot(rel0, axis0) * inv_s0;
    float d1 = dot(rel0, axis1) * inv_s1;
    float d2 = dot(rel0, axis2) * inv_s2;
    float v0 = dot(r.direction, axis0) * inv_s0;
    float v1 = dot(r.direction, axis1) * inv_s1;
    float v2 = dot(r.direction, axis2) * inv_s2;
    float3 d = float3(d0, d1, d2);
    float3 v = float3(v0, v1, v2);
    float denom = dot(v, v);
    if (denom < 1e-12) return false;

    float t = -dot(d, v) / denom;
    if (t < r.min_distance || t > r.max_distance) return false;

    float3 vc = v * rsqrt(denom);
    float3 cr = cross(vc, d);
    float m2 = dot(cr, cr);
    if (m2 > p.kernel_pad.x) return false;

    float alpha = min(p.center_opacity.w * exp(-0.5 * m2), GS_ALPHA_CLAMP);
    if (alpha <= GS_ALPHA_CULL) return false;

    out_hit.t = t;
    out_hit.m2 = m2;
    out_hit.alpha = alpha;
    out_hit.id = id;
    return true;
}

inline void insert_hit(thread GsHit hits[GS_MAX_K], thread uint& hit_count, uint max_k, GsHit h)
{
    if (max_k == 0u) return;
    uint n = min(hit_count, max_k);
    if (n == max_k && h.t >= hits[n - 1u].t) return;

    uint pos = n;
    while (pos > 0u && h.t < hits[pos - 1u].t) {
        if (pos < max_k) hits[pos] = hits[pos - 1u];
        pos--;
    }
    if (pos < max_k) hits[pos] = h;
    if (hit_count < max_k) hit_count++;
}

inline float3 gaussian_normal(const ray r, device const GsParticle* particles, GsHit h)
{
    GsParticle p = particles[h.id];
    float3 world_p = r.origin + r.direction * h.t;
    float3 center = p.center_opacity.xyz;
    float3 axis0 = p.axis0_inv.xyz;
    float3 axis1 = p.axis1_inv.xyz;
    float3 axis2 = p.axis2_inv.xyz;
    float inv_s0 = p.axis0_inv.w;
    float inv_s1 = p.axis1_inv.w;
    float inv_s2 = p.axis2_inv.w;
    float3 rel = world_p - center;
    float l0 = dot(rel, axis0) * inv_s0;
    float l1 = dot(rel, axis1) * inv_s1;
    float l2 = dot(rel, axis2) * inv_s2;
    float3 n = axis0 * (l0 * inv_s0) +
               axis1 * (l1 * inv_s1) +
               axis2 * (l2 * inv_s2);
    n = normalize(n);
    if (dot(n, r.direction) > 0.0) n = -n;
    return n;
}

struct GsBoxIntersection
{
    bool  accept_intersection [[accept_intersection]];
    float distance            [[distance]];
};

[[intersection(bounding_box, triangle_data, instancing)]]
GsBoxIntersection gs_isect(
    float3                    origin       [[origin]],
    float3                    direction    [[direction]],
    float                     min_distance [[min_distance]],
    float                     max_distance [[max_distance]],
    uint                      primitive_id [[primitive_id]],
    device const GsParticle*  particles    [[buffer(6)]])
{
    GsBoxIntersection out;
    out.accept_intersection = false;
    out.distance = max_distance;

    GsParticle p = particles[primitive_id];
    float3 center = p.center_opacity.xyz;
    float3 axis0 = p.axis0_inv.xyz;
    float3 axis1 = p.axis1_inv.xyz;
    float3 axis2 = p.axis2_inv.xyz;
    float inv_s0 = p.axis0_inv.w;
    float inv_s1 = p.axis1_inv.w;
    float inv_s2 = p.axis2_inv.w;
    float3 rel0 = origin - center;
    float d0 = dot(rel0, axis0) * inv_s0;
    float d1 = dot(rel0, axis1) * inv_s1;
    float d2 = dot(rel0, axis2) * inv_s2;
    float v0 = dot(direction, axis0) * inv_s0;
    float v1 = dot(direction, axis1) * inv_s1;
    float v2 = dot(direction, axis2) * inv_s2;
    float3 d = float3(d0, d1, d2);
    float3 v = float3(v0, v1, v2);
    float denom = dot(v, v);
    if (denom < 1e-12) return out;

    float t = -dot(d, v) / denom;
    if (t < min_distance || t > max_distance) return out;

    float3 vc = v * rsqrt(denom);
    float3 cr = cross(vc, d);
    float m2 = dot(cr, cr);
    if (m2 > p.kernel_pad.x) return out;

    float alpha = min(p.center_opacity.w * exp(-0.5 * m2), GS_ALPHA_CLAMP);
    if (alpha <= GS_ALPHA_CULL) return out;

    out.accept_intersection = true;
    out.distance = t;
    return out;
}

kernel void gs_rt_render(
    uint2                                   tid       [[thread_position_in_grid]],
    texture2d<float, access::write>         output    [[texture(0)]],
    instance_acceleration_structure         accel     [[buffer(0)]],
    constant GsPushConstants&               pc        [[buffer(1)]],
    device const GsParticle*                particles [[buffer(2)]],
    device const float*                     sh        [[buffer(3)]],
    device float*                           depth_buf [[buffer(4)]],
    device float*                           normal_buf [[buffer(5)]],
    intersection_function_table<triangle_data, instancing, world_space_data> gs_ift [[buffer(6)]])
{
    uint W = output.get_width();
    uint H = output.get_height();
    if (tid.x >= W || tid.y >= H) return;

    uint pixel_index = tid.y * W + tid.x;

    float2 px = float2(tid) + 0.5;
    float2 uv = px / float2(W, H);
    float2 d = uv * 2.0 - 1.0;

    float4 origin_h = float4(0.0, 0.0, 0.0, 1.0) * pc.view_inv;
    float4 target = float4(d.x, d.y, 1.0, 1.0) * pc.proj_inv;
    float4 dir_h = float4(normalize(target.xyz), 0.0) * pc.view_inv;

    ray r;
    r.origin = origin_h.xyz;
    r.direction = normalize(dir_h.xyz);
    r.min_distance = 0.001;
    r.max_distance = 1.0e5;

    float3 accum = float3(0.0);
    float transmittance = 1.0;
    float depth = -1.0;
    float3 normal = float3(0.0);
    float opacity_accum = 0.0;

    intersector<triangle_data, instancing, world_space_data> isect;
    isect.force_opacity(forced_opacity::opaque);
    float next_min = r.min_distance;
    uint guard = 0u;
    while (guard < max(pc.max_passes, 1u) && transmittance > pc.min_transmittance) {
        guard++;
        r.min_distance = next_min;
        intersection_result<triangle_data, instancing, world_space_data> hit =
            isect.intersect(r, accel, 0xFFu, gs_ift);
        if (hit.type == intersection_type::none) break;

        uint id = (hit.type == intersection_type::triangle) ? hit.instance_id : hit.primitive_id;
        next_min = hit.distance + max(1e-5, hit.distance * 1e-5);
        if (id < pc.particle_count) {
            GsHit h;
            if (eval_gaussian_hit(r, particles, id, h)) {
                float alpha = h.alpha;
                float3 sh_dir = normalize(particles[h.id].center_opacity.xyz - r.origin);
                float3 c = eval_sh(sh, h.id, pc.sh_degree, sh_dir);
                if (pc.color_space != 0u) c = srgb_to_linear(saturate(c));

                accum += transmittance * alpha * c;
                transmittance *= (1.0 - alpha);
                opacity_accum = 1.0 - transmittance;

                if (depth < 0.0 && opacity_accum >= pc.iso_opacity_threshold) {
                    depth = h.t;
                    normal = gaussian_normal(r, particles, h);
                }
            }
        }
    }

    depth_buf[pixel_index] = depth;
    uint ni = pixel_index * 3u;
    normal_buf[ni + 0u] = normal.x;
    normal_buf[ni + 1u] = normal.y;
    normal_buf[ni + 2u] = normal.z;

    output.write(float4(saturate(accum), 1.0), tid);
}
