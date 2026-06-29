// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_scalar_block_layout : enable

/*
 * raytrace_curve.rchit.glsl — Closest-hit for BasisCurves tubes/ribbons
 * (Phase 11.A.2.4 / "pretty/shading" pass).
 *
 * Shading model: energy-conserving Lambert + GGX specular for rough
 * organic curve geometry. Reads the per-segment color (binding 15) as the
 * albedo; the CPU path supplies the bound material base color when
 * available and falls back to displayColor / warm twig tones.
 *
 * Lighting:
 *   * scene-defined DistantLight / RectLight / SphereLight via binding 13.
 *     DistantLights pull `inputs:angle` from the SceneLight buffer and
 *     cone-jitter their shadow ray inside that disk for soft penumbrae;
 *     RectLights / SphereLights use a single-sample-at-centre approx
 *     (matches the mesh closest-hit's evalRectLight / evalSphereLight).
 *   * Hemisphere ambient (sky/ground split) — keeps surfaces that face
 *     away from every direct light non-black, tuned for bark / roots /
 *     palm fibers instead of high-contrast metal.
 *   * Ambient occlusion: 4 deterministic short rays (one along N, three
 *     45°-tilted) — same speckle-free pattern at every pixel. Length =
 *     5% of scene diagonal (≈ 0.5 m for the showcase scenes). Big visual
 *     win for tube intersections at ~5 % extra ray traffic.
 *   * Soft shadows reuse the curve intersection math inline inside the
 *     ray query — intersection shaders aren't invoked from ray queries,
 *     so the closed-form ray-vs-finite-cylinder test is duplicated in
 *     `intersectCylinder()` below. Mesh / floor occluders are handled
 *     by the ray-query traversal automatically (triangle BLASes commit
 *     hits without our help).
 *   * ACES filmic tone-map at the end so contrast survives the UNORM
 *     quantize step in the headless readback path.
 *
 * Bindings (set 0):
 *   0:  TLAS (used for shadow + AO ray queries).
 *   13: scene lights SSBO (matches the mesh closest-hit layout).
 *   14: per-segment SceneCurveSegment[] (also read by intersection shader).
 *   15: per-segment color (RGBA8 packed into uint32; 4 B/segment).
 *
 * Color buffer layout: each element is a uint32 with byte 0 (LSB) = R,
 * byte 1 = G, byte 2 = B, byte 3 = A. Decoded via unpackUnorm4x8 which
 * places byte 0 into .x. Quantization (1/255) is below the dynamic
 * range of the lighting in the closest-hit so the rendered output is
 * visually identical to the previous vec4-float layout.
 */

/* MUST match scene.h SceneCurveSegment layout (Phase 11.5.1). */
struct CurveSegment {
    vec3  p0;
    float r0;
    vec3  p1;
    uint  mat_flags;
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout(set = 0, binding = 2, scalar) readonly buffer SceneData {
    uint  vertexStride;
    uint  hasMaterials;
    float envMipLevels;
    float envIntensity;
    vec4  domeColor;
    uint  upAxis;
    uint  _scenePad0;
    uint  _scenePad1;
    uint  _scenePad2;
} scene;

layout(set = 0, binding = 14, scalar) readonly buffer CurveSegmentBuffer {
    CurveSegment segs[];
} curve_segs;

layout(set = 0, binding = 15, std430) readonly buffer CurveColorBuffer {
    uint colors[];  /* RGBA8 packed (byte 0 = R, byte 3 = A) */
} curve_colors;

const uint CURVE_SEG_FLAG_RIBBON = 0x80000000u;
const uint CURVE_SEG_FLAG_RIBBON_JOIN_PAD = 0x40000000u;
const uint CURVE_SEG_OCT_MASK = 0x00007fffu;

/* Scene lights SSBO — same layout as in raytrace.rchit.glsl. */
#define LIGHT_KIND_RECT     0
#define LIGHT_KIND_DISTANT  1
#define LIGHT_KIND_SPHERE   2

struct GpuLight {
    vec3  position;   float intensity;
    vec3  normal;     int   kind;
    vec3  u_axis;     int   normalize;
    vec3  v_axis;     float angle_deg;
    vec3  color;      float _pad;
};

layout(set = 0, binding = 13, std430) readonly buffer LightsBuffer {
    int       nlights;
    int       _lpad[3];
    GpuLight  items[];
} sceneLights;

/* Push constants — must match GpuRtPushConstants exactly. We only consume
 * `scene_scale` (for AO ray length) and `fast_mode` (which disables the
 * shadow + AO ray-query work for RL inference sensors). */
layout(push_constant) uniform PushConstants {
    mat4  viewInverse;
    mat4  projInverse;
    float ground_y;
    float scene_scale;
    uint  fast_mode;
    uint  depth_enabled;
    uint  segmentation_enabled;
    uint  normals_enabled;
    uint  deferred_shade_enabled;
    float tone_exposure_scale;
    float tone_sky_scale;
    float tone_white_point;
    uint  tone_flags;
    float rt_ibl_fill_scale;
};

const uint TONE_FLAG_SKIP_SECONDARY_VISIBILITY = 0x20000000u;
const uint TONE_FLAG_SKIP_AO_VISIBILITY = 0x10000000u;
const uint TONE_FLAG_SKIP_DIRECT_SHADOWS = 0x08000000u;
const uint TONE_FLAG_RECT_SHARED_SHADOWS = 0x04000000u;

layout(location = 0) rayPayloadInEXT vec4 hitValue;
hitAttributeEXT vec2 hitUV;

const float PI = 3.14159265359;

float sceneUpCoord(vec3 v) {
    if (scene.upAxis == 0u) return v.x;
    if (scene.upAxis == 2u) return v.z;
    return v.y;
}

/* --- PBR helpers (same forms as the mesh closest-hit). ----------------- */
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) *
                pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

/* Pixar branchless ONB — used to build a coordinate frame around N for AO
 * cosine sampling and around -light.normal for soft DistantLight cones. */
void branchlessONB(vec3 n, out vec3 b1, out vec3 b2) {
    float sgn = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sgn + n.z);
    float b = n.x * n.y * a;
    b1 = vec3(1.0 + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
    b2 = vec3(b, sgn + n.y * n.y * a, -n.y);
}

/* Cheap per-pixel-per-call hash. Not blue-noise but better than nothing
 * for one-sample soft shadows / AO. Two channels = two independent
 * uniforms in [0, 1). */
vec2 hash2(uint seed) {
    uvec3 v = uvec3(gl_LaunchIDEXT.xy, seed);
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    return vec2(v.xy) * (1.0 / 4294967296.0);
}

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

float intersectRibbon(CurveSegment seg, vec3 O, vec3 D,
                      float tmin, float tmax)
{
    float halfWidth = seg.r0;
    if (halfWidth <= 0.0) return -1.0;
    vec3 axis = seg.p1 - seg.p0;
    float h = length(axis);
    if (h < 1e-6) return -1.0;
    vec3 a = axis / h;
    vec3 normal = decodeRibbonNormal(seg.mat_flags);
    vec3 side = ribbonSideVector(a, normal);
    vec3 planeNormal = ribbonPlaneNormal(a, side, normal);
    float denom = dot(D, planeNormal);
    if (abs(denom) < 1e-8) return -1.0;
    float t = dot(seg.p0 - O, planeNormal) / denom;
    if (t < tmin || t > tmax) return -1.0;
    vec3 relHit = O + t * D - seg.p0;
    float u = dot(relHit, a);
    float joinPad = ((seg.mat_flags & CURVE_SEG_FLAG_RIBBON_JOIN_PAD) != 0u)
        ? ribbonJoinPad(halfWidth, h) : 0.0;
    if (u < -joinPad || u > h + joinPad) return -1.0;
    float v = dot(relHit, side);
    if (abs(v) > halfWidth) return -1.0;
    return t;
}

/* Per-segment cylinder probe: closed-form ray-vs-finite-cylinder. Used
 * by shadow / AO rays to commit procedural curve intersections inside a
 * ray query (the intersection shader is not invoked from ray queries —
 * this duplicates the math from raytrace_curve.rint.glsl). Returns the
 * front-face t in [tmin, tmax], or a sentinel < tmin if no hit. */
float intersectCylinder(uint segIdx, vec3 O, vec3 D, float tmin, float tmax)
{
    CurveSegment seg = curve_segs.segs[segIdx];
    if ((seg.mat_flags & CURVE_SEG_FLAG_RIBBON) != 0u) {
        return intersectRibbon(seg, O, D, tmin, tmax);
    }
    float r = seg.r0;
    if (r <= 0.0) return -1.0;
    vec3  axis = seg.p1 - seg.p0;
    float h    = length(axis);
    if (h < 1e-6) return -1.0;
    vec3 a = axis / h;
    vec3 rel = O - seg.p0;
    float oz = dot(rel, a);
    float dz = dot(D,   a);
    vec3 op = rel - oz * a;
    vec3 dp = D   - dz * a;
    float A = dot(dp, dp);
    if (A < 1e-12) return -1.0;
    float B = 2.0 * dot(op, dp);
    float C = dot(op, op) - r * r;
    float disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return -1.0;
    float sq = sqrt(disc);
    float t1 = (-B - sq) / (2.0 * A);
    float t2 = (-B + sq) / (2.0 * A);
    if (t1 >= tmin && t1 <= tmax) {
        float u = oz + t1 * dz;
        if (u >= 0.0 && u <= h) return t1;
    }
    if (t2 >= tmin && t2 <= tmax) {
        float u = oz + t2 * dz;
        if (u >= 0.0 && u <= h) return t2;
    }
    return -1.0;
}

/* TLAS shadow probe — returns 1.0 if the point can see along `dir` for
 * `tmax` units, 0.0 if the ray is blocked. Uses ray query so the curve
 * closest-hit doesn't have to recurse through traceRayEXT.
 *
 * Ray flag note: gl_RayFlagsOpaqueEXT here makes triangle BLAS hits
 * auto-commit through the ray-query traversal so mesh occluders shadow
 * the curves. Without it, triangle candidates would surface as
 * gl_RayQueryCandidateIntersectionTriangleEXT and silently never
 * confirm — present-day showcase scenes are curve-only, but any future
 * mixed-geometry scene would lose mesh-on-curve shadows. Procedural
 * curve AABBs still need the inline cylinder math because intersection
 * shaders aren't invoked from ray queries even with OpaqueEXT. */
float traceShadow(vec3 origin, vec3 dir, float tmax)
{
    if (fast_mode != 0 ||
        (tone_flags & TONE_FLAG_SKIP_SECONDARY_VISIBILITY) != 0u ||
        (tone_flags & TONE_FLAG_SKIP_DIRECT_SHADOWS) != 0u) {
        return 1.0;
    }
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFF, origin, 0.001, dir, tmax);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) ==
            gl_RayQueryCandidateIntersectionAABBEXT) {
            uint segIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            float t = intersectCylinder(segIdx, origin, dir, 0.001, tmax);
            if (t > 0.0) {
                rayQueryGenerateIntersectionEXT(rq, t);
            }
        }
    }
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
}

/* AO ray — short tmax, identical structure. Kept separate so we can tune
 * the bias / length independently. Same OpaqueEXT semantics as the
 * shadow probe — triangles auto-commit, procedural AABBs run the inline
 * cylinder math. */
float traceAO(vec3 origin, vec3 dir, float tmax)
{
    if (fast_mode != 0 ||
        (tone_flags & TONE_FLAG_SKIP_SECONDARY_VISIBILITY) != 0u ||
        (tone_flags & TONE_FLAG_SKIP_AO_VISIBILITY) != 0u) {
        return 1.0;
    }
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFF, origin, 0.001, dir, tmax);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) ==
            gl_RayQueryCandidateIntersectionAABBEXT) {
            uint segIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            float t = intersectCylinder(segIdx, origin, dir, 0.001, tmax);
            if (t > 0.0) {
                rayQueryGenerateIntersectionEXT(rq, t);
            }
        }
    }
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
}

/* Cosine-weighted hemisphere sample around N (z-up in the local frame). */
vec3 sampleCosineHemisphere(vec3 N, vec2 xi)
{
    float phi   = 2.0 * PI * xi.x;
    float r     = sqrt(xi.y);
    float x     = r * cos(phi);
    float y     = r * sin(phi);
    float z     = sqrt(max(0.0, 1.0 - xi.y));
    vec3  T, B;
    branchlessONB(N, T, B);
    return normalize(T * x + B * y + N * z);
}

/* Pick a direction inside a cone of half-angle `cosThetaMax` (in cosine
 * space) around `axis`. Used to soften the DistantLight shadow. */
vec3 sampleCone(vec3 axis, float cosThetaMax, vec2 xi)
{
    float cosTheta = mix(cosThetaMax, 1.0, xi.x);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi      = 2.0 * PI * xi.y;
    vec3  T, B;
    branchlessONB(axis, T, B);
    return normalize(
        T * (sinTheta * cos(phi)) +
        B * (sinTheta * sin(phi)) +
        axis * cosTheta);
}

/* Cook-Torrance BRDF + soft shadow for a DistantLight. The light's
 * `angle_deg` (USD `inputs:angle`) is consumed as the cone half-angle —
 * for the showcase scenes' KeyLight that's 0.53° (sun-disc), barely
 * visible as a soft penumbra. We use the same jittered direction for
 * shadow + BRDF: at sub-1° angles the highlight stays effectively
 * crisp, and re-running the BRDF with an unjittered L doubles the
 * sin/cos cost for no perceptible change. */
vec3 evalDistantLight(GpuLight light, vec3 N, vec3 V, vec3 baseColor,
                      float metallic, float roughness, vec3 F0,
                      vec3 worldPos, vec2 jitter)
{
    vec3 L0 = normalize(-light.normal);
    float angleRad = max(light.angle_deg, 0.05) * (PI / 180.0);
    vec3 L = sampleCone(L0, cos(angleRad), jitter);

    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    float shadow = traceShadow(worldPos + N * 0.005, L, 1.0e5);
    if (shadow <= 0.0) return vec3(0.0);

    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3  F = fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, roughness);

    vec3  numerator   = D * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    vec3  specular    = numerator / denominator;

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    /* Empirical scale lifted from the mesh path's evalDistantLight to
     * match ovrtx exposure when intensity=15000. */
    return (kD * baseColor / PI + specular) *
           light.color * (light.intensity * 0.0025) *
           NdotL * shadow;
}

float areaLightSpecularRoughness(float roughness, float angularDiameter) {
    float a = clamp(angularDiameter * 2.5, 0.0, 1.0);
    return clamp(sqrt(roughness * roughness + a * a), roughness, 1.0);
}

vec3 evalRectLightSample(GpuLight light, vec3 samplePos, float radiance,
                         float area, vec3 N, vec3 V, vec3 baseColor,
                         float metallic, float roughness, vec3 F0,
                         vec3 worldPos, float sharedShadow)
{
    vec3 toLight = samplePos - worldPos;
    float dist2  = dot(toLight, toLight);
    if (dist2 < 1e-4) return vec3(0.0);
    float dist   = sqrt(dist2);
    vec3  L      = toLight / dist;

    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);
    float cosLight = dot(-L, light.normal);
    if (cosLight <= 0.0) return vec3(0.0);

    float geom = (cosLight * area) / max(dist2, 1e-4);

    float shadow = (sharedShadow >= 0.0)
        ? sharedShadow
        : traceShadow(worldPos + N * 0.005, L, dist * 0.999);
    if (shadow <= 0.0) return vec3(0.0);

    vec3 H = normalize(V + L);
    /* The rect is sampled as a few point lights; widen the specular lobe by
     * emitter angular diameter so rough floors do not get point-light glints. */
    float rectAngular = clamp(2.0 * max(length(light.u_axis), length(light.v_axis)) /
                              max(dist, 1e-3), 0.0, 1.0);
    float specRoughness = areaLightSpecularRoughness(roughness, rectAngular);
    float D = distributionGGX(N, H, specRoughness);
    float G = geometrySmith(N, V, L, specRoughness);
    vec3  F = fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, roughness);
    vec3 specular = (D * G * F) /
                    (4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    return (kD * baseColor / PI + specular) * (light.color * radiance) *
           NdotL * geom * shadow;
}

/* USD UsdLuxRectLight — deterministic 3x3 Gauss-Legendre quadrature over the
 * authored area, kept aligned with the mesh and primary RT shaders. */
vec3 evalRectLight(GpuLight light, vec3 N, vec3 V, vec3 baseColor,
                   float metallic, float roughness, vec3 F0, vec3 worldPos)
{
    float area = 4.0 * length(light.u_axis) * length(light.v_axis);
    float radiance = (light.normalize != 0)
        ? light.intensity / max(area * PI, 1e-6)
        : light.intensity / PI;
    radiance *= (scene.envIntensity < -1.0) ? 0.92 : 0.0012;

    const float q = 0.7745966692;
    const float wc = 0.1975308642;
    const float we = 0.1234567901;
    const float wk = 0.0771604938;
    vec3 center = light.position;
    vec3 U = light.u_axis;
    vec3 W = light.v_axis;
    float sharedShadow = -1.0;
    if ((tone_flags & TONE_FLAG_RECT_SHARED_SHADOWS) != 0u) {
        vec3 centerToLight = center - worldPos;
        float centerDist2 = dot(centerToLight, centerToLight);
        if (centerDist2 > 1e-4) {
            float centerDist = sqrt(centerDist2);
            vec3 centerL = centerToLight / centerDist;
            float centerNdotL = max(dot(N, centerL), 0.0);
            float centerCosLight = dot(-centerL, light.normal);
            sharedShadow = (centerNdotL > 0.0 && centerCosLight > 0.0)
                ? traceShadow(worldPos + N * 0.005, centerL, centerDist * 0.999)
                : 0.0;
        } else {
            sharedShadow = 0.0;
        }
    }
    vec3 accum = vec3(0.0);
    accum += evalRectLightSample(light, center, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sharedShadow) * wc;
    accum += evalRectLightSample(light, center + U * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sharedShadow) * we;
    accum += evalRectLightSample(light, center - U * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sharedShadow) * we;
    accum += evalRectLightSample(light, center + W * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sharedShadow) * we;
    accum += evalRectLightSample(light, center - W * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sharedShadow) * we;
    accum += evalRectLightSample(light, center + U * q + W * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sharedShadow) * wk;
    accum += evalRectLightSample(light, center + U * q - W * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sharedShadow) * wk;
    accum += evalRectLightSample(light, center - U * q + W * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sharedShadow) * wk;
    accum += evalRectLightSample(light, center - U * q - W * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sharedShadow) * wk;
    return accum;
}

vec3 evalSphereLight(GpuLight light, vec3 N, vec3 V, vec3 baseColor,
                     float metallic, float roughness, vec3 F0, vec3 worldPos)
{
    vec3 toLight = light.position - worldPos;
    float dist2 = max(dot(toLight, toLight), 1e-4);
    float dist  = sqrt(dist2);
    vec3 L      = toLight / dist;
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    float radius = max(light.u_axis.x, 1e-4);
    float area   = 4.0 * PI * radius * radius;
    float radiance = (light.normalize != 0)
        ? light.intensity / area
        : light.intensity;
    /* Keep curve SphereLight energy aligned with mesh/RT shaders. */
    radiance *= 0.00017;

    float shadow = traceShadow(worldPos + N * 0.005, L, max(dist - radius, 0.01));
    if (shadow <= 0.0) return vec3(0.0);

    vec3 H = normalize(V + L);
    float sphereAngular = clamp(2.0 * radius / max(dist, 1e-3), 0.0, 1.0);
    float specRoughness = areaLightSpecularRoughness(roughness, sphereAngular);
    float D = distributionGGX(N, H, specRoughness);
    float G = geometrySmith(N, V, L, specRoughness);
    vec3  F = fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, specRoughness);
    vec3 specular = (D * G * F) /
                    (4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    return (kD * baseColor / PI + specular) *
           light.color * radiance * NdotL * shadow / dist2;
}

/* ACES filmic tone mapping (fitted curve by Krzysztof Narkowicz). */
vec3 acesFilmic(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main()
{
    int  idx  = gl_PrimitiveID;
    vec3 base = unpackUnorm4x8(curve_colors.colors[idx]).rgb;

    /* Recompute the curve normal at the hit point. Tube segments use the
     * cylinder radial normal. Ribbon segments use Storm's oriented authored
     * normal and flip it for back-facing hits. */
    CurveSegment seg = curve_segs.segs[idx];
    vec3 axis = normalize(seg.p1 - seg.p0);
    vec3 hitP = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 N;
    bool isRibbon = (seg.mat_flags & CURVE_SEG_FLAG_RIBBON) != 0u;
    if (isRibbon) {
        N = decodeRibbonNormal(seg.mat_flags);
    } else {
        vec3 rel  = hitP - seg.p0;
        vec3 perp = rel - dot(rel, axis) * axis;
        N = normalize(perp);
    }
    if (dot(N, gl_WorldRayDirectionEXT) > 0.0) N = -N;

    vec3 V = normalize(-gl_WorldRayDirectionEXT);
    float NdotV = max(dot(N, V), 0.001);

    /* Material — rough organic dielectric. Moana's BasisCurves are mostly
     * vegetation, roots, and fine fibers; treating them as metallic pipes
     * makes the island read as black/white wiring. */
    vec3  baseColor = base;
    float metallic  = 0.0;
    float roughness = 0.86;
    vec3  F0        = mix(vec3(0.018), baseColor, metallic);

    /* Hemisphere ambient — sky/ground split lifted from the mesh shader's
     * no-IBL path. Curves are sub-pixel-dense in Moana foliage, so keep
     * them readable without washing out contact shadows. */
    vec3 skyColor    = vec3(0.58, 0.66, 0.80);
    vec3 groundColor = vec3(0.30, 0.28, 0.24);
    if (scene.envIntensity < -1.0) {
        skyColor = vec3(0.50, 0.55, 0.60);
        groundColor = vec3(0.34, 0.28, 0.20);
    }
    float hemi = clamp(sceneUpCoord(N) * 0.5 + 0.5, 0.0, 1.0);
    vec3 ambientIrradiance = mix(groundColor, skyColor, hemi) * 0.96;

    /* AO: 4 deterministic short rays — straight up along N plus three
     * 45°-tilted directions in the tangent frame. Deterministic dirs
     * avoid the temporal/spatial speckle a per-pixel hash would
     * introduce; for static head-shots that single-sample look is
     * worse than a slightly-biased clean estimate. Length = 5% of
     * scene diagonal (typical pipe-junction scale). */
    float aoLen = max(scene_scale * 0.05, 0.25);
    vec3  aoT, aoB;
    branchlessONB(N, aoT, aoB);
    /* Cone half-angle ≈ 50° around N: cos(50°) ≈ 0.64; tilted dirs at
     * 45° contribute the bulk of the irradiance integral. */
    vec3 aoDir0 = N;
    vec3 aoDir1 = normalize(N * 0.707 + aoT * 0.707);
    vec3 aoDir2 = normalize(N * 0.707 - aoT * 0.707);
    vec3 aoDir3 = normalize(N * 0.707 + aoB * 0.707);
    float ao0 = traceAO(hitP + N * 0.005, aoDir0, aoLen);
    float ao1 = traceAO(hitP + N * 0.005, aoDir1, aoLen);
    float ao2 = traceAO(hitP + N * 0.005, aoDir2, aoLen);
    float ao3 = traceAO(hitP + N * 0.005, aoDir3, aoLen);
    /* Cosine-weighted average (the up-direction sample carries the
     * largest weight under a cosine kernel; the 45° tilts each have
     * cos = 0.707). */
    float ao = (ao0 * 1.0 + (ao1 + ao2 + ao3) * 0.707) /
               (1.0 + 3.0 * 0.707);
    /* Keep AO useful for dense root/leaf intersections without crushing
     * fine plant curves to black. */
    ao = mix((scene.envIntensity < -1.0) ? 0.68 : 0.45, 1.0, ao);

    vec3 kS_amb = fresnelSchlick(NdotV, F0);
    vec3 kD_amb = (1.0 - kS_amb) * (1.0 - metallic);
    vec3 ambient = (kD_amb * ambientIrradiance * baseColor +
                    kS_amb * ambientIrradiance * 0.14) * ao;
    if (scene.envIntensity < -1.0) {
        ambient += baseColor * vec3(0.10, 0.085, 0.060) * ao;
    }

    /* Direct lighting — iterate scene lights. RectLights / SphereLights
     * use a single-sample approximation; DistantLights get cone-jittered
     * soft shadows. */
    vec3 Lo = vec3(0.0);
    if (sceneLights.nlights > 0) {
        for (int li = 0; li < sceneLights.nlights; ++li) {
            GpuLight Lt = sceneLights.items[li];
            if (Lt.kind == LIGHT_KIND_DISTANT) {
                /* New jitter pair per light so multiple distants don't
                 * correlate. */
                vec2 jD = hash2(uint(0xD15A0001u + uint(li)));
                Lo += evalDistantLight(Lt, N, V, baseColor,
                                       metallic, roughness, F0, hitP, jD);
            } else if (Lt.kind == LIGHT_KIND_RECT) {
                Lo += evalRectLight(Lt, N, V, baseColor,
                                    metallic, roughness, F0, hitP);
            } else if (Lt.kind == LIGHT_KIND_SPHERE) {
                Lo += evalSphereLight(Lt, N, V, baseColor,
                                      metallic, roughness, F0, hitP);
            }
        }
    } else {
        /* Fallback synthetic key light (matches the mesh path's no-light
         * branch). */
        vec3 syntheticKey = normalize(vec3(0.3, 1.0, 0.5));
        float NdotL = max(dot(N, syntheticKey), 0.0);
        vec3 H = normalize(V + syntheticKey);
        float D = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, syntheticKey, roughness);
        vec3  F = fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, roughness);
        vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        Lo += (kD * baseColor / PI + specular) *
              vec3(1.0, 0.95, 0.85) * 1.6 * NdotL;
    }

    vec3 result = ambient + Lo;

    /* Exposure + ACES filmic — align curve tubes with the mesh RT path when
     * authored lights are present. Moana has millions of fine vegetation
     * curves; the older 1.2x curve-only exposure blew real material colors out
     * to pale gray/white. */
    float curveExposure = (sceneLights.nlights > 0)
        ? ((scene.envIntensity < -1.0) ? 0.88 : 0.62)
        : 1.0;
    result = acesFilmic(result * curveExposure * max(tone_exposure_scale, 0.0));

    hitValue = vec4(result, 1.0);
}
