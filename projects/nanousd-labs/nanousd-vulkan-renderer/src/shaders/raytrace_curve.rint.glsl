// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : enable

/*
 * raytrace_curve.rint.glsl — Ray-vs-cylinder/ribbon intersection for
 * BasisCurves (Phase 11.A.2.4 + Phase 11.5.1 wire format).
 *
 * For each ray that hits a curve-segment AABB, this shader runs the
 * closed-form ray-vs-finite-cylinder test (algorithm pattern: Cycles
 * `cylinder_intersect`, kernels/geom/curve_intersect.h, Apache 2.0 —
 * studied, not copied; math is the textbook quadratic-in-t formulation).
 *
 * MVP assumptions for Phase 11.A:
 *   - **constant-radius per segment** (Phase 11.5.1 wire-format spec).
 *     Varying-radius input (per-CV widths) is realised at scene-load
 *     time as a series of segments each with its leading CV's radius —
 *     Storm's pattern. True varying-radius via cone-sphere arrives in
 *     Phase 11.B with a parallel `r1[]` SSBO at binding 16.
 *   - cylinder side-surface only; the spherical end-caps the canonical
 *     algorithm includes are skipped here. Adjacent segments share an
 *     endpoint, so an internal joint is doubly covered by the two
 *     side-cylinders — visible artifacts are limited to the *outermost*
 *     endpoints of a curve run, where missing caps show as flat disks.
 *
 * Hit attribute encodes the parametric u along the segment in [0, 1].
 * The closest-hit reads gl_HitAttributeEXT.x for u.
 *
 * Binding 14 (set 0): per-segment SSBO uploaded by gpu_upload_curve_data
 * (32 B/seg). Layout MUST match `SceneCurveSegment` in `src/scene.h`:
 *   { vec3 p0; float r0; vec3 p1; uint mat_flags; }
 * `mat_flags` bit 31 marks a Storm-style oriented ribbon; the low 30 bits
 * pack a 15-bit octahedral world-space ribbon normal.
 */

struct CurveSegment {
    vec3  p0;
    float r0;
    vec3  p1;
    uint  mat_flags;   /* MUST match scene.h SceneCurveSegment layout */
};

layout(set = 0, binding = 14, scalar) readonly buffer CurveSegmentBuffer {
    CurveSegment segs[];
} curve_segs;

hitAttributeEXT vec2 hitUV;   /* (u along segment, v around tube)         */

const uint CURVE_SEG_FLAG_RIBBON = 0x80000000u;
const uint CURVE_SEG_FLAG_RIBBON_JOIN_PAD = 0x40000000u;
const uint CURVE_SEG_OCT_MASK = 0x00007fffu;

float signNotZero(float v)
{
    return v < 0.0 ? -1.0 : 1.0;
}

vec3 decodeRibbonNormal(uint matFlags)
{
    float x = float(matFlags & CURVE_SEG_OCT_MASK) *
              (1.0 / 32767.0) * 2.0 - 1.0;
    float y = float((matFlags >> 15u) & CURVE_SEG_OCT_MASK) *
              (1.0 / 32767.0) * 2.0 - 1.0;
    vec3 n = vec3(x, y, 1.0 - abs(x) - abs(y));
    if (n.z < 0.0) {
        float oldX = n.x;
        n.x = (1.0 - abs(n.y)) * signNotZero(oldX);
        n.y = (1.0 - abs(oldX)) * signNotZero(n.y);
    }
    return normalize(n);
}

vec3 ribbonSideVector(vec3 axis, vec3 normal)
{
    vec3 side = cross(axis, normal);
    float len2 = dot(side, side);
    if (len2 > 1e-12) return side * inversesqrt(len2);
    vec3 fallback = abs(axis.z) < 0.9 ? vec3(0.0, 0.0, 1.0)
                                      : vec3(0.0, 1.0, 0.0);
    side = cross(axis, fallback);
    len2 = dot(side, side);
    return len2 > 1e-12 ? side * inversesqrt(len2) : vec3(1.0, 0.0, 0.0);
}

vec3 ribbonPlaneNormal(vec3 axis, vec3 side, vec3 authoredNormal)
{
    vec3 planeNormal = cross(side, axis);
    float len2 = dot(planeNormal, planeNormal);
    if (len2 <= 1e-12) return authoredNormal;
    planeNormal *= inversesqrt(len2);
    return dot(planeNormal, authoredNormal) < 0.0 ? -planeNormal : planeNormal;
}

float ribbonJoinPad(float halfWidth, float segmentLength)
{
    return min(max(halfWidth * 0.75, 0.0), max(segmentLength * 0.25, 0.0));
}

void main()
{
    CurveSegment seg = curve_segs.segs[gl_PrimitiveID];

    vec3  O = gl_WorldRayOriginEXT;
    vec3  D = gl_WorldRayDirectionEXT;

    /* Constant-radius per segment (Phase 11.5.1 wire format). */
    float r = seg.r0;
    if (r <= 0.0) return;

    vec3  axis = seg.p1 - seg.p0;
    float h    = length(axis);
    if (h < 1e-6) return;
    vec3  a    = axis / h;

    if ((seg.mat_flags & CURVE_SEG_FLAG_RIBBON) != 0u) {
        vec3 normal = decodeRibbonNormal(seg.mat_flags);
        vec3 side = ribbonSideVector(a, normal);
        vec3 planeNormal = ribbonPlaneNormal(a, side, normal);
        float denom = dot(D, planeNormal);
        if (abs(denom) < 1e-8) return;
        float t_hit = dot(seg.p0 - O, planeNormal) / denom;
        if (t_hit < gl_RayTminEXT || t_hit > gl_RayTmaxEXT) return;

        vec3 relHit = O + t_hit * D - seg.p0;
        float u = dot(relHit, a);
        float joinPad = ((seg.mat_flags & CURVE_SEG_FLAG_RIBBON_JOIN_PAD) != 0u)
            ? ribbonJoinPad(r, h) : 0.0;
        if (u < -joinPad || u > h + joinPad) return;
        float v = dot(relHit, side);
        if (abs(v) > r) return;

        float uv = h > 0.0 ? clamp(u / h, 0.0, 1.0) : 0.0;
        float vv = r > 0.0
            ? clamp(v / (2.0 * r) + 0.5, 0.0, 1.0)
            : 0.5;
        hitUV = vec2(uv, vv);
        reportIntersectionEXT(t_hit, 0);
        return;
    }

    /* Project ray and origin onto plane perpendicular to the axis.
     * |op + t * dp|^2 = r^2 → quadratic A t^2 + B t + C = 0. */
    vec3  rel = O - seg.p0;
    float oz  = dot(rel, a);
    float dz  = dot(D,   a);
    vec3  op  = rel - oz * a;
    vec3  dp  = D   - dz * a;

    float A = dot(dp, dp);
    if (A < 1e-12) return;            /* ray parallel to cylinder axis */
    float B = 2.0 * dot(op, dp);
    float C = dot(op, op) - r * r;

    float disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return;

    float sq = sqrt(disc);
    float t1 = (-B - sq) / (2.0 * A);
    float t2 = (-B + sq) / (2.0 * A);

    float tmin = gl_RayTminEXT;
    float tmax = gl_RayTmaxEXT;

    /* Inspect both roots. Front entry t1 is preferred; t2 (exit) is the
     * fallback when the ray origin is inside the tube. */
    float t_hit = -1.0;
    float u_hit = 0.0;
    if (t1 >= tmin && t1 <= tmax) {
        float u = oz + t1 * dz;
        if (u >= 0.0 && u <= h) {
            t_hit = t1;
            u_hit = u / h;
        }
    }
    if (t_hit < 0.0 && t2 >= tmin && t2 <= tmax) {
        float u = oz + t2 * dz;
        if (u >= 0.0 && u <= h) {
            t_hit = t2;
            u_hit = u / h;
        }
    }
    if (t_hit < 0.0) return;

    /* v (around-tube angle) is computed by the closest-hit shader from
     * gl_WorldRayOriginEXT + t*gl_WorldRayDirectionEXT projected onto
     * the perpendicular plane. We pack only u here; v is local to the
     * hit and recomputed without extra storage. */
    hitUV = vec2(u_hit, 0.0);
    reportIntersectionEXT(t_hit, 0);
}
