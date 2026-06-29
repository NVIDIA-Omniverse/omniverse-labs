// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * mesh.metal — raster mesh shaders, MSL port of mesh.vert.glsl + mesh.frag.glsl.
 *
 * Vertex layout matches renderer.c's per-vertex packing (12 floats):
 *   +0..2  position
 *   +3..5  normal
 *   +6..8  pad (reserved for tangent / per-vertex color)
 *   +9..10 uv
 *   +11    material id (uint bits)
 *
 * Push constants (set via [encoder setVertexBytes:length:atIndex:1] and
 * [encoder setFragmentBytes:length:atIndex:1]) match GpuMeshPushConstants
 * in gpu.h: { float mvp[16]; float model[16]; float color[4]; float eye_pos[4];
 *             int material_index; uint instanced; uint instance_base; uint pad; }
 *
 * Matrix-storage convention: the C side stores row-major mat4. Metal's
 * float4x4 is column-major. We compensate with `vector * matrix` order
 * (matching the GLSL source which does the same).
 */

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

constant int MAX_MTLX_PROC_NODES = 64;
constant int MTLX_PROC_TYPE_FLOAT = 1;
constant int MTLX_PROC_OP_CONST = 1;
constant int MTLX_PROC_OP_POSITION = 2;
constant int MTLX_PROC_OP_TEXCOORD = 3;
constant int MTLX_PROC_OP_ADD = 4;
constant int MTLX_PROC_OP_SUBTRACT = 5;
constant int MTLX_PROC_OP_MULTIPLY = 6;
constant int MTLX_PROC_OP_DIVIDE = 7;
constant int MTLX_PROC_OP_DOT = 8;
constant int MTLX_PROC_OP_SIN = 9;
constant int MTLX_PROC_OP_POWER = 10;
constant int MTLX_PROC_OP_MIX = 11;
constant int MTLX_PROC_OP_CLAMP = 12;
constant int MTLX_PROC_OP_ABS = 13;
constant int MTLX_PROC_OP_MIN = 14;
constant int MTLX_PROC_OP_MAX = 15;
constant int MTLX_PROC_OP_FRACTAL3D = 16;
constant int MTLX_PROC_OP_CONVERT = 17;
constant int MTLX_PROC_OP_COMBINE3 = 18;
constant int MTLX_PROC_OP_EXTRACT = 19;
constant int MTLX_PROC_OP_INVERT = 20;
constant int MTLX_PROC_OP_IFGREATER = 21;
constant int MTLX_PROC_OP_RAMPTB = 22;
constant int MTLX_PROC_OP_NOISE3D = 23;
constant int MTLX_PROC_OP_CELLNOISE = 24;
constant int MTLX_PROC_OP_TEXTURE = 25;
constant int MTLX_PROC_OP_RGBTOHSV = 26;
constant int MTLX_PROC_OP_HSVTORGB = 27;

struct MtlxProcNode {
    int    op;
    int    type;
    int    in0;
    int    in1;
    float4 value;
    int    in2;
    int    in3;
    int    _pad0;
    int    _pad1;
};

struct MeshPushConstants {
    float4x4 mvp;
    float4x4 model;
    float4   color;          // .w > 0.5 = override vertex color
    float4   eye_pos;        // .xyz = camera world position
    int      material_index; // -1 = no material; matches host GpuMeshPushConstants
    uint     instanced;
    uint     instance_base;
    uint     _pad0;
};

struct RasterInstanceData {
    float4x4 mvp;
    float4x4 model;
    float4   color;
    int      material_index;
    int      _pad[3];
};

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 objectPos;
    float3 normal;
    float3 color;
    float2 uv;
    float  material_index;
};

/* MaterialParams mirror — MUST be size + layout-compatible with
 * GpuMaterialParams in gpu.h. Earlier this only covered base_color +
 * texture indices + use_vertex_color (~100 B), but the host struct is
     * larger than that minimal head (Phase 7c+ Standard Surface and
 * fields trail at the end). MSL would then pack the device-buffer array
 * stride at 112 B, so mat_params[material_index] for any
 * material_index >= 1 lands at the wrong offset and reads junk —
 * visibly that produced random pinkish/blue tints across chess pieces.
 *
 * Mirror the full struct here. Raster uses the transmission/specular
 * workflow fields directly; the remaining trailing fields are still
 * present for stride parity with the host upload (gpu_upload_materials
 * writes sizeof(GpuMaterialParams) per element). */
struct RasterMaterialParams {
    float4 base_color;          // host offset   0
    float4 emissive_color;      //               16
    float  metallic;            //               32
    float  roughness;           //               36
    float  opacity;             //               40
    float  ior;                 //               44
    float  occlusion;           //               48
    float  clearcoat;           //               52
    float  clearcoat_roughness; //               56
    float  normal_scale;        //               60
    int    tex_diffuse;         //               64  (tex_indices[0])
    int    tex_normal;          //               68  ([1])
    int    tex_roughness;       //               72  ([2])
    int    tex_metallic;        //               76  ([3])
    int    tex_emissive;        //               80  ([4])
    int    tex_occlusion;       //               84  ([5])
    int    tex_opacity;         //               88  ([6])
    int    tex_displacement;    //               92  ([7])
    int    use_vertex_color;    //               96
    float  udim_scale_u;        //              100
    float  udim_scale_v;        //              104
    float  opacity_threshold;   //              108
    /* Phase 7c — Standard Surface trailing block. */
    float4 _subsurface_color;   //              112
    float4 _subsurface_radius;  //              128
    float4 transmission_color;  //              144
    float  _subsurface_weight;  //              160
    float  _subsurface_scale;   //              164
    float  transmission_weight; //              168
    float  transmission_ior;    //              172
    int    _tex_sss_weight;     //              176
    int    tex_trans_weight;    //              180
    int    _sss_color_authored; //              184
    int    use_specular_wf;     //              188
    float4 specular_color;      //              192
    float4 normal_tex_scale;    //              208
    float4 normal_tex_bias;     //              224
    float4 uv_transform;        //              240
    float4 roughness_xform;     //              256
    int    v_flip;              //              272
    int    _pad_v_flip0;        //              276
    int    _pad_v_flip1;        //              280
    int    _pad_v_flip2;        //              284
    float  base_weight;         //              288
    float  specular_weight;     //              292
    float  sheen_weight;        //              296
    float  sheen_roughness;     //              300
    float4 sheen_color;         //              304
    float  thin_film_thickness; //              320
    float  thin_film_ior;       //              324
    float  specular_anisotropy; //              328
    int    standard_surface;    //              332
    int    procedural_kind;     //              336
    int    procedural_base;     //              340
    int    procedural_sss;      //              344
    int    procedural_octaves;  //              348
    float4 procedural_color1;   //              352
    float4 procedural_color2;   //              368
    float4 procedural_params;   //              384
    int    procedural_node_count; //            400
    int    procedural_base_out;   //            404
    int    procedural_sss_out;    //            408
    int    procedural_rough_out;  //            412
    int    procedural_normal_out; //            416
    int    procedural_flags;      //            420
    int    procedural_pad0;       //            424
    int    procedural_pad1;       //            428
    MtlxProcNode procedural_nodes[MAX_MTLX_PROC_NODES]; // 432
    /* total = 3504 host bytes; MSL rounds struct alignment up to a
     * multiple of the largest member (float4 = 16) → 3504 = 219×16, so
     * stride is exact. */
};

inline float2 material_uv(RasterMaterialParams mp, float2 uv)
{
    float2 texUV = (mp.v_flip != 0) ? float2(uv.x, 1.0 - uv.y) : uv;
    float2 scale = mp.uv_transform.xy;
    if (abs(scale.x) < 1.0e-8 && abs(scale.y) < 1.0e-8)
        scale = float2(1.0);
    float2 udim = max(float2(mp.udim_scale_u, mp.udim_scale_v), float2(1.0));
    return (texUV * scale + mp.uv_transform.zw) / udim;
}

vertex VertexOut vertex_mesh(
    VertexIn in                          [[stage_in]],
    constant MeshPushConstants& pc       [[buffer(1)]],
    const device RasterInstanceData* instances [[buffer(2)]],
    uint instance_id                     [[instance_id]])
{
    float4x4 mvp = pc.mvp;
    float4x4 model = pc.model;
    float4 color = pc.color;
    int material_index = pc.material_index;
    if (pc.instanced != 0u) {
        const device RasterInstanceData& inst = instances[instance_id];
        mvp = inst.mvp;
        model = inst.model;
        color = inst.color;
        material_index = inst.material_index;
    }

    VertexOut out;
    out.position = float4(in.position, 1.0) * mvp;
    /* The proj matrix in camera.c bakes a Vulkan-style Y-flip
     * (out[5] = -f) so RT's rgen — which works from proj_inv — gets
     * the correct ray direction in Metal NDC. Raster doesn't have
     * that round-trip and would render upside-down without this
     * compensation. Negating clip-space y is mathematically the same
     * as the Vulkan VkViewport.height < 0 convention. */
    out.position.y = -out.position.y;
    out.worldPos = (float4(in.position, 1.0) * model).xyz;
    out.objectPos = in.position;
    out.normal   = normalize((float4(in.normal, 0.0) * model).xyz);
    out.color    = color.rgb;
    out.uv       = in.uv;
    out.material_index = (float)material_index;
    return out;
}

struct CurveSegment {
    packed_float3 p0;   /* packed: 32 B host stride (see raytrace.metal) */
    float         r0;
    packed_float3 p1;
    float         r1;
};

struct CurveRasterUniform {
    float4x4 vp;
    float4   eye_pos;
    float2   viewport_size;
    float    min_pixel_radius;
    float    max_pixel_radius;
    float4   dome_color;
};

struct CurveRasterOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float3 color;
};

static inline float4 curve_project(float3 p, float4x4 vp)
{
    float4 clip = float4(p, 1.0) * vp;
    clip.y = -clip.y;
    return clip;
}

vertex CurveRasterOut curve_raster_vs(
    uint vertex_id                         [[vertex_id]],
    uint instance_id                       [[instance_id]],
    device const CurveSegment* curve_segs  [[buffer(0)]],
    device const float4* curve_colors      [[buffer(1)]],
    constant CurveRasterUniform& u         [[buffer(2)]])
{
    CurveSegment seg = curve_segs[instance_id];
    float3 p0 = seg.p0;
    float3 p1 = seg.p1;
    float3 axis = p1 - p0;
    float axis_len = length(axis);
    axis = (axis_len > 1.0e-7) ? (axis / axis_len) : float3(0.0, 1.0, 0.0);

    float4 c0 = curve_project(p0, u.vp);
    float4 c1 = curve_project(p1, u.vp);
    float iw0 = 1.0 / max(abs(c0.w), 1.0e-7);
    float iw1 = 1.0 / max(abs(c1.w), 1.0e-7);
    float2 ndc0 = c0.xy * iw0;
    float2 ndc1 = c1.xy * iw1;
    float2 screen_axis = ndc1 - ndc0;
    float screen_len = length(screen_axis);
    float2 perp = (screen_len > 1.0e-7)
        ? float2(-screen_axis.y, screen_axis.x) / screen_len
        : float2(1.0, 0.0);

    bool use_p1 = (vertex_id == 1u || vertex_id == 2u || vertex_id == 4u);
    float side = (vertex_id == 2u || vertex_id == 4u || vertex_id == 5u) ? 1.0 : -1.0;
    float3 p = use_p1 ? p1 : p0;
    float radius = max(use_p1 ? seg.r1 : seg.r0, 0.0);
    float4 clip = use_p1 ? c1 : c0;
    float2 ndc = use_p1 ? ndc1 : ndc0;

    float3 mid = (p0 + p1) * 0.5;
    float3 view_dir = normalize(mid - u.eye_pos.xyz);
    float3 offset_axis = cross(view_dir, axis);
    float offset_len = length(offset_axis);
    offset_axis = (offset_len > 1.0e-7) ? (offset_axis / offset_len) : float3(1.0, 0.0, 0.0);
    float4 clip_radius = curve_project(p + offset_axis * radius, u.vp);
    float2 ndc_radius = clip_radius.xy / max(abs(clip_radius.w), 1.0e-7);
    float projected_px = length((ndc_radius - ndc) * u.viewport_size * 0.5);
    float half_px = clamp(max(projected_px, u.min_pixel_radius),
                          0.5, u.max_pixel_radius);
    float2 offset_ndc = perp * (side * half_px * 2.0 / max(u.viewport_size, float2(1.0)));
    clip.xy += offset_ndc * clip.w;

    CurveRasterOut out;
    out.position = clip;
    out.worldPos = p;
    out.normal = offset_axis;
    out.color = clamp(curve_colors[instance_id].rgb, float3(0.0), float3(1.0));
    return out;
}

fragment float4 curve_raster_fs(CurveRasterOut in [[stage_in]],
                                 constant CurveRasterUniform& u [[buffer(2)]])
{
    float3 base = max(in.color, float3(0.03));
    float3 N = normalize(in.normal);
    float3 keyDir = normalize(float3(0.3, 1.0, 0.5));
    float3 fillDir = normalize(float3(-0.5, 0.4, -0.3));
    float hemi = dot(N, float3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
    float key = abs(dot(N, keyDir));
    float fill = abs(dot(N, fillDir));
    float dome_intensity = max(u.dome_color.a, 0.0);
    float3 dome = (dome_intensity > 0.0)
        ? clamp(u.dome_color.rgb * dome_intensity, float3(0.0), float3(8.0))
        : float3(0.50, 0.58, 0.75);
    float3 sky = mix(float3(0.42, 0.56, 0.78), dome, dome_intensity > 0.0 ? 0.85 : 0.0);
    float3 keyColor = mix(float3(1.0, 0.95, 0.84), dome, dome_intensity > 0.0 ? 0.35 : 0.0);
    float3 lit = base * (0.22 + 0.78 * key) * keyColor
               + base * float3(0.62, 0.74, 1.0) * (0.24 * fill)
               + base * sky * (0.34 * hemi)
               + float3(0.015, 0.025, 0.012);
    return float4(clamp(lit, 0.0, 1.0), 1.0);
}

/* Equirectangular UV from a world-space direction. Y-up convention,
 * matching raytrace.metal:dirToEquirect so raster + RT look up the same
 * texel for the same normal. */
static inline float2 dirToEquirect(float3 d)
{
    constexpr float kDomeLongitudeOffset = 340.0 / 360.0;
    float theta = atan2(d.x, d.z);
    float phi   = asin(clamp(d.y, -1.0, 1.0));
    return float2(fract(0.5 + theta / (2.0 * 3.14159265) + kDomeLongitudeOffset),
                  0.5 - phi   / 3.14159265);
}

/* Roughness-aware Schlick Fresnel — F0..1 lerp by NdotV with a per-
 * roughness max-clamp so polished metals don't lose ambient under
 * grazing angles. Matches the raytrace.metal IBL term. */
static inline float3 fresnelSchlickRoughness(float NdotV, float3 F0, float roughness)
{
    float3 k = float3(1.0 - roughness);
    return F0 + (max(k, F0) - F0) * pow(1.0 - NdotV, 5.0);
}

/* Cheap ACES filmic tonemap — same constants raytrace.metal's IBL pass
 * uses. Only applied on the IBL branch so pre-IBL chess scenes render
 * bit-stable (no implicit gamma shift). */
static inline float3 acesFilmic(float3 x)
{
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

/* Keep raster's authored-DomeLight exposure in step with the Metal RT path.
 * The HDR env is auto-normalized at upload, so intensity=1000 should not
 * re-boost surfaces by the old 1..1000 ramp. */
static inline float rasterEnvSurfaceExposure(float intensity)
{
    intensity = max(intensity, 0.0);
    float t1 = clamp(log(1.0 + max(intensity, 1000.0)) / log(1001.0), 0.0, 1.25);
    float gate = smoothstep(1000.0, 2000.0, intensity);
    float boost1 = 1.0 + 0.4 * t1;
    float boost2 = pow(max(intensity / 1000.0, 1.0), 0.85);
    float baseExposure = (intensity > 1.0) ? 0.38 : 0.5;
    return mix(baseExposure, 0.5 * boost1 * boost2, gate);
}

static inline float rasterEnvSkyScale(float intensity)
{
    intensity = max(intensity, 0.0);
    return (intensity > 1.0) ? (intensity * 0.000882) : 1.0;
}

/* Pixar branchless ONB — derives a stable orthonormal basis (T, B) from
 * a world-space normal so we can project a tangent-space normal map back
 * into world space without per-vertex tangent inputs. Same construction
 * as raytrace.metal:shade_hit uses for the rchit's normal-map sampling. */
static inline void branchlessONB(float3 n, thread float3& t, thread float3& b)
{
    float sign_ = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (sign_ + n.z);
    float bv = n.x * n.y * a;
    t = float3(1.0 + sign_ * n.x * n.x * a, sign_ * bv, -sign_ * n.x);
    b = float3(bv, sign_ + n.y * n.y * a, -n.y);
}

/* GGX micro-facet distribution. alpha = roughness² is the standard
 * Disney/Karis convention. */
static inline float distributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom + 1e-6);
}

/* Schlick-GGX geometric occlusion (combined Smith term). */
static inline float geometrySmithSchlick(float3 N, float3 V, float3 L, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float Gv = NdotV / (NdotV * (1.0 - k) + k);
    float Gl = NdotL / (NdotL * (1.0 - k) + k);
    return Gv * Gl;
}

static inline float3 fresnelSchlick(float HdotV, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);
}

static inline float3 thinFilmTint(float NdotV, float thicknessNm, float filmIor)
{
    if (thicknessNm <= 0.0) return float3(1.0);
    float n = max(filmIor, 1.001);
    float sin2t = (1.0 - NdotV * NdotV) / (n * n);
    float cost = sqrt(max(0.0, 1.0 - sin2t));
    float optical = 4.0 * 3.14159265 * n * thicknessNm * cost;
    float3 wavelength = float3(680.0, 550.0, 440.0);
    float3 phase = optical / wavelength + float3(0.0, 2.094395, 4.188790);
    float3 bands = 0.5 + 0.5 * cos(phase);
    return mix(float3(1.0), 0.35 + 0.95 * bands, 0.75);
}

static inline float mxpNegateIf(float val, bool b)
{
    return b ? -val : val;
}

static inline float mxpTrilerp(float v0, float v1, float v2, float v3,
                               float v4, float v5, float v6, float v7,
                               float s, float t, float r)
{
    float s1 = 1.0 - s;
    float t1 = 1.0 - t;
    float r1 = 1.0 - r;
    return r1 * (t1 * (v0 * s1 + v1 * s) + t * (v2 * s1 + v3 * s)) +
           r  * (t1 * (v4 * s1 + v5 * s) + t * (v6 * s1 + v7 * s));
}

static inline float mxpGradientFloat(uint hash, float x, float y, float z)
{
    uint h = hash & 15u;
    float u = (h < 8u) ? x : y;
    float v = (h < 4u) ? y : (((h == 12u) || (h == 14u)) ? x : z);
    return mxpNegateIf(u, (h & 1u) != 0u) +
           mxpNegateIf(v, (h & 2u) != 0u);
}

static inline uint mxpRotl32(uint x, uint k)
{
    return (x << k) | (x >> (32u - k));
}

static inline void mxpBjmix(thread uint& a, thread uint& b, thread uint& c)
{
    a -= c; a ^= mxpRotl32(c,  4u); c += b;
    b -= a; b ^= mxpRotl32(a,  6u); a += c;
    c -= b; c ^= mxpRotl32(b,  8u); b += a;
    a -= c; a ^= mxpRotl32(c, 16u); c += b;
    b -= a; b ^= mxpRotl32(a, 19u); a += c;
    c -= b; c ^= mxpRotl32(b,  4u); b += a;
}

static inline uint mxpBjfinal(uint a, uint b, uint c)
{
    c ^= b; c -= mxpRotl32(b, 14u);
    a ^= c; a -= mxpRotl32(c, 11u);
    b ^= a; b -= mxpRotl32(a, 25u);
    c ^= b; c -= mxpRotl32(b, 16u);
    a ^= c; a -= mxpRotl32(c,  4u);
    b ^= a; b -= mxpRotl32(a, 14u);
    c ^= b; c -= mxpRotl32(b, 24u);
    return c;
}

static inline uint mxpHashInt(int x, int y, int z)
{
    uint len = 3u;
    uint a = uint(0xdeadbeefu) + (len << 2u) + 13u;
    uint b = a;
    uint c = a;
    a += uint(x);
    b += uint(y);
    c += uint(z);
    return mxpBjfinal(a, b, c);
}

static inline float mxpFade(float t)
{
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

static inline float mxpPerlinNoiseFloat(float3 p)
{
    int X = int(floor(p.x));
    int Y = int(floor(p.y));
    int Z = int(floor(p.z));
    float fx = p.x - float(X);
    float fy = p.y - float(Y);
    float fz = p.z - float(Z);
    float u = mxpFade(fx);
    float v = mxpFade(fy);
    float w = mxpFade(fz);
    float result = mxpTrilerp(
        mxpGradientFloat(mxpHashInt(X,     Y,     Z    ), fx,       fy,       fz),
        mxpGradientFloat(mxpHashInt(X + 1, Y,     Z    ), fx - 1.0, fy,       fz),
        mxpGradientFloat(mxpHashInt(X,     Y + 1, Z    ), fx,       fy - 1.0, fz),
        mxpGradientFloat(mxpHashInt(X + 1, Y + 1, Z    ), fx - 1.0, fy - 1.0, fz),
        mxpGradientFloat(mxpHashInt(X,     Y,     Z + 1), fx,       fy,       fz - 1.0),
        mxpGradientFloat(mxpHashInt(X + 1, Y,     Z + 1), fx - 1.0, fy,       fz - 1.0),
        mxpGradientFloat(mxpHashInt(X,     Y + 1, Z + 1), fx,       fy - 1.0, fz - 1.0),
        mxpGradientFloat(mxpHashInt(X + 1, Y + 1, Z + 1), fx - 1.0, fy - 1.0, fz - 1.0),
        u, v, w);
    return 0.9820 * result;
}

static inline float mxpFractal3dNoiseFloat(float3 p, int octaves,
                                           float lacunarity, float diminish)
{
    float result = 0.0;
    float amplitude = 1.0;
    for (int i = 0; i < octaves; ++i) {
        result += amplitude * mxpPerlinNoiseFloat(p);
        amplitude *= diminish;
        p *= lacunarity;
    }
    return result;
}

static inline float3 procAsVec3(float4 v, int type)
{
    return (type == MTLX_PROC_TYPE_FLOAT) ? float3(v.x) : v.xyz;
}

static inline float4 procFromVec3(float3 v)
{
    return float4(v, 1.0);
}

static inline float3 mxpRgbToHsv(float3 c)
{
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = (c.g < c.b) ? float4(c.bg, K.wz) : float4(c.gb, K.xy);
    float4 q = (c.r < p.x) ? float4(p.xyw, c.r) : float4(c.r, p.yzx);
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)),
                  d / (q.x + e),
                  q.x);
}

static inline float3 mxpHsvToRgb(float3 c)
{
    float3 p = abs(fract(c.xxx + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 -
                   float3(3.0));
    return c.z * mix(float3(1.0), clamp(p - float3(1.0), 0.0, 1.0),
                     clamp(c.y, 0.0, 1.0));
}

static inline float mxpHashToUnit(uint h)
{
    return float(h & 0x00ffffffu) * (1.0 / 16777215.0);
}

static inline float mxpCellnoiseFloat(float3 p)
{
    int x = int(floor(p.x));
    int y = int(floor(p.y));
    int z = int(floor(p.z));
    return mxpHashToUnit(mxpHashInt(x, y, z));
}

static inline float4 evalMtlxProcGraph(thread const RasterMaterialParams& mp,
                                       int outputIdx,
                                       float3 objectPos,
                                       float2 uv,
                                       array<texture2d<float>, 64> matTextures,
                                       sampler matSampler)
{
    float4 values[MAX_MTLX_PROC_NODES];
    int count = clamp(mp.procedural_node_count, 0, MAX_MTLX_PROC_NODES);
    for (int i = 0; i < MAX_MTLX_PROC_NODES; ++i) {
        if (i >= count) break;
        MtlxProcNode n = mp.procedural_nodes[i];
        float4 a = (n.in0 >= 0 && n.in0 < i) ? values[n.in0] : float4(0.0);
        float4 b = (n.in1 >= 0 && n.in1 < i) ? values[n.in1] : float4(0.0);
        float4 c = (n.in2 >= 0 && n.in2 < i) ? values[n.in2] : float4(0.0);
        float4 d = (n.in3 >= 0 && n.in3 < i) ? values[n.in3] : float4(0.0);
        int at = (n.in0 >= 0 && n.in0 < count) ? mp.procedural_nodes[n.in0].type : n.type;
        int bt = (n.in1 >= 0 && n.in1 < count) ? mp.procedural_nodes[n.in1].type : n.type;
        int ct = (n.in2 >= 0 && n.in2 < count) ? mp.procedural_nodes[n.in2].type : n.type;
        int dt = (n.in3 >= 0 && n.in3 < count) ? mp.procedural_nodes[n.in3].type : n.type;
        float4 r = float4(0.0, 0.0, 0.0, 1.0);

        if (n.op == MTLX_PROC_OP_CONST) {
            r = n.value;
        } else if (n.op == MTLX_PROC_OP_POSITION) {
            r = float4(objectPos, 1.0);
        } else if (n.op == MTLX_PROC_OP_TEXCOORD) {
            r = float4(uv, 0.0, 1.0);
        } else if (n.op == MTLX_PROC_OP_TEXTURE) {
            int texIdx = int(n.value.x + 0.5);
            float2 tuv = (n.value.y > 0.5) ? (uv * procAsVec3(a, at).xy)
                                           : procAsVec3(a, at).xy;
            if (texIdx >= 0 && texIdx < 64) {
                float4 texel = matTextures[texIdx].sample(matSampler, tuv);
                r = (n.type == MTLX_PROC_TYPE_FLOAT)
                  ? float4(texel.r, 0.0, 0.0, texel.a)
                  : float4(texel.rgb, texel.a);
            }
        } else if (n.op == MTLX_PROC_OP_DOT) {
            r.x = dot(procAsVec3(a, at), procAsVec3(b, bt));
        } else if (n.op == MTLX_PROC_OP_FRACTAL3D) {
            int octaves = max(1, min(int(n.value.y + 0.5), 8));
            float3 p = procAsVec3(a, at);
            float lac = max(n.value.z, 0.001);
            float dim = clamp(n.value.w, 0.0, 1.0);
            if (n.type == MTLX_PROC_TYPE_FLOAT) {
                r.x = n.value.x * mxpFractal3dNoiseFloat(p, octaves, lac, dim);
            } else {
                r = procFromVec3(n.value.x * float3(
                    mxpFractal3dNoiseFloat(p, octaves, lac, dim),
                    mxpFractal3dNoiseFloat(p + float3(19.1, 7.3, 5.7), octaves, lac, dim),
                    mxpFractal3dNoiseFloat(p + float3(3.7, 17.9, 11.2), octaves, lac, dim)));
            }
        } else if (n.op == MTLX_PROC_OP_CONVERT) {
            r = (n.type == MTLX_PROC_TYPE_FLOAT)
              ? float4(a.x, 0.0, 0.0, 1.0)
              : procFromVec3(procAsVec3(a, at));
        } else if (n.op == MTLX_PROC_OP_COMBINE3) {
            r = procFromVec3(float3(a.x, b.x, c.x));
        } else if (n.op == MTLX_PROC_OP_EXTRACT) {
            int idx = max(0, min(int(n.value.x + 0.5), 2));
            r.x = (idx == 0) ? a.x : ((idx == 1) ? a.y : a.z);
        } else if (n.op == MTLX_PROC_OP_IFGREATER) {
            bool chooseA = a.x > b.x;
            float4 chosen = chooseA ? c : d;
            int chosenType = chooseA ? ct : dt;
            r = (n.type == MTLX_PROC_TYPE_FLOAT)
              ? float4(chosen.x, 0.0, 0.0, 1.0)
              : procFromVec3(procAsVec3(chosen, chosenType));
        } else if (n.op == MTLX_PROC_OP_RAMPTB) {
            float t = clamp(procAsVec3(c, ct).y, 0.0, 1.0);
            if (n.type == MTLX_PROC_TYPE_FLOAT) {
                r.x = mix(b.x, a.x, t);
            } else {
                r = procFromVec3(mix(procAsVec3(b, bt),
                                     procAsVec3(a, at), t));
            }
        } else if (n.op == MTLX_PROC_OP_NOISE3D) {
            float3 p = procAsVec3(a, at);
            float amp = n.value.x;
            float pivot = n.value.y;
            if (n.type == MTLX_PROC_TYPE_FLOAT) {
                r.x = amp * (0.5 + 0.5 * mxpPerlinNoiseFloat(p) - pivot);
            } else {
                r = procFromVec3(amp * (float3(
                    0.5 + 0.5 * mxpPerlinNoiseFloat(p),
                    0.5 + 0.5 * mxpPerlinNoiseFloat(p + float3(19.1, 7.3, 5.7)),
                    0.5 + 0.5 * mxpPerlinNoiseFloat(p + float3(3.7, 17.9, 11.2))) -
                    float3(pivot)));
            }
        } else if (n.op == MTLX_PROC_OP_CELLNOISE) {
            float3 p = procAsVec3(a, at);
            if (n.value.x < 2.5) p.z = 0.0;
            if (n.type == MTLX_PROC_TYPE_FLOAT) {
                r.x = mxpCellnoiseFloat(p);
            } else {
                r = procFromVec3(float3(
                    mxpCellnoiseFloat(p),
                    mxpCellnoiseFloat(p + float3(17.0, 5.0, 13.0)),
                    mxpCellnoiseFloat(p + float3(3.0, 29.0, 7.0))));
            }
        } else if (n.op == MTLX_PROC_OP_RGBTOHSV) {
            r = procFromVec3(mxpRgbToHsv(procAsVec3(a, at)));
        } else if (n.op == MTLX_PROC_OP_HSVTORGB) {
            r = procFromVec3(mxpHsvToRgb(procAsVec3(a, at)));
        } else if (n.type == MTLX_PROC_TYPE_FLOAT) {
            float ax = a.x;
            float bx = b.x;
            float cx = c.x;
            if (n.op == MTLX_PROC_OP_ADD) r.x = ax + bx;
            else if (n.op == MTLX_PROC_OP_SUBTRACT) r.x = ax - bx;
            else if (n.op == MTLX_PROC_OP_MULTIPLY) r.x = ax * bx;
            else if (n.op == MTLX_PROC_OP_DIVIDE) r.x = ax / ((abs(bx) > 1e-6) ? bx : 1.0);
            else if (n.op == MTLX_PROC_OP_SIN) r.x = sin(ax);
            else if (n.op == MTLX_PROC_OP_POWER) r.x = pow(max(ax, 0.0), bx);
            else if (n.op == MTLX_PROC_OP_MIX) r.x = mix(ax, bx, cx);
            else if (n.op == MTLX_PROC_OP_CLAMP) r.x = clamp(ax, bx, cx);
            else if (n.op == MTLX_PROC_OP_ABS) r.x = abs(ax);
            else if (n.op == MTLX_PROC_OP_MIN) r.x = min(ax, bx);
            else if (n.op == MTLX_PROC_OP_MAX) r.x = max(ax, bx);
            else if (n.op == MTLX_PROC_OP_INVERT) r.x = mix(ax, 1.0 - ax, bx);
        } else {
            float3 av = procAsVec3(a, at);
            float3 bv = procAsVec3(b, bt);
            float mx = c.x;
            if (n.op == MTLX_PROC_OP_ADD) r = procFromVec3(av + bv);
            else if (n.op == MTLX_PROC_OP_SUBTRACT) r = procFromVec3(av - bv);
            else if (n.op == MTLX_PROC_OP_MULTIPLY) r = procFromVec3(av * bv);
            else if (n.op == MTLX_PROC_OP_DIVIDE) r = procFromVec3(av / max(abs(bv), float3(1e-6)));
            else if (n.op == MTLX_PROC_OP_SIN) r = procFromVec3(sin(av));
            else if (n.op == MTLX_PROC_OP_POWER) r = procFromVec3(pow(max(av, float3(0.0)), bv));
            else if (n.op == MTLX_PROC_OP_MIX) r = procFromVec3(mix(av, bv, mx));
            else if (n.op == MTLX_PROC_OP_CLAMP) r = procFromVec3(clamp(av, bv, procAsVec3(c, ct)));
            else if (n.op == MTLX_PROC_OP_ABS) r = procFromVec3(abs(av));
            else if (n.op == MTLX_PROC_OP_MIN) r = procFromVec3(min(av, bv));
            else if (n.op == MTLX_PROC_OP_MAX) r = procFromVec3(max(av, bv));
            else if (n.op == MTLX_PROC_OP_INVERT) r = procFromVec3(mix(av, float3(1.0) - av, bv));
        }
        values[i] = r;
    }
    return (outputIdx >= 0 && outputIdx < count) ? values[outputIdx] : float4(0.0);
}

static inline float3 evalMtlxMarble3d(thread const RasterMaterialParams& mp,
                                      float3 objectPos)
{
    int octaves = max(1, min(mp.procedural_octaves, 8));
    float scale1 = mp.procedural_params.x;
    float scale2 = mp.procedural_params.y;
    float powerV = max(mp.procedural_params.z, 0.001);
    float noiseAmp = mp.procedural_params.w;
    float wave = dot(objectPos, float3(1.0)) * scale1 +
                 mxpFractal3dNoiseFloat(objectPos * scale2, octaves, 2.0, 0.5) * noiseAmp;
    float t = pow(clamp(sin(wave) * 0.5 + 0.5, 0.0, 1.0), powerV);
    float3 c1 = mp.procedural_color1.rgb;
    float3 c2 = mp.procedural_color2.rgb;
    return mix(c1, c2, t);
}

/* Mesh-only TLAS visibility query for raster shadow mode. Curve instances are
 * masked out host-side with 0xFE; mesh instances are 0xFF, so mask 0x01 only
 * tests triangles. */
static inline float trace_mesh_visibility(float3 origin, float3 dir, float tmax,
                                          instance_acceleration_structure accel)
{
    intersection_query<instancing> q;
    intersection_params params;
    params.accept_any_intersection(true);
    params.force_opacity(forced_opacity::opaque);

    ray r;
    r.origin       = origin;
    r.direction    = dir;
    r.min_distance = 0.001;
    r.max_distance = tmax;

    q.reset(r, accel, 0x01u, params);
    while (q.next()) {}
    return (q.get_committed_intersection_type() == intersection_type::none) ? 1.0 : 0.0;
}

/* Per-frame fragment uniform — has_materials gate plus IBL state.
 * Mirrors the Vulkan port's pc.ibl_params (mesh.frag.glsl line 53):
 * has_ibl is a 0/1 bool deciding diffuse-/specular-IBL vs procedural
 * 3-point lighting; mip_levels is the env-map prefilter chain depth so
 * the specular tap can pick the right roughness LOD; intensity is the
 * DomeLight inputs:intensity scalar (1.0 default; auto-exposure already
 * baked into env_texture itself). Bound at fragment buffer(3) by
 * gpu_metal.mm:gpu_cmd_bind_pipeline. */
struct RasterFragUniform {
    uint  has_materials;
    uint  has_ibl;
    float env_mip_levels;
    float env_intensity;
    uint  debug_mode;      // 0=beauty, 1=basecolor, 2=metallic, 3=roughness, 4=normal, 5=diffuse IBL, 6=specular IBL
    uint3 light_pad;       // .x = authored scene-light count (host: gpu->light_count, offset 32); >8 => many-light/no-IBL Isaac-style scene. uint3 keeps dome_color 16-byte aligned at offset 48 (matches host layout).
    float4 dome_color;     // rgb * a is the flat fallback DomeLight
};

/* Shared three-point/IBL lighting kernel (factored out so both fragment_mesh
 * and the shadow variant can use it; MSL forbids calling functions
 * declared with the [[fragment]] qualifier from other shader functions). */
static inline float4 shade_mesh(VertexOut in,
                                 device const RasterMaterialParams* mat_params,
                                 array<texture2d<float>, 64> mat_textures,
                                 sampler                       mat_sampler,
                                 texture2d<float>              env_tex,
                                 texture2d<float>              irr_tex,
                                 texture2d<float>              brdf_tex,
                                 RasterFragUniform             u,
                                 int                           material_index,
                                 float3                        eye_pos)
{
    float3 N = normalize(in.normal);

    /* ---- Material albedo + scalar params ----
     * Default: per-vertex/per-mesh fragColor (USD displayColor). The
     * material slot overrides only when use_vertex_color == 0, matching
     * the RT shader. The placeholder material (uploaded for descriptor
     * wiring when scene has no UsdShade materials) has use_vertex_color=1
     * so this branch keeps fragColor as the albedo. */
    float3 albedo    = in.color;
    float  metallic  = 0.0;
    float  roughness = 0.5;
    float  alpha     = 1.0;
    float3 emissive  = float3(0.0);
    float  ao        = 1.0;
    float  ior       = 1.5;
    float  transWeight = 0.0;
    float3 transTint = float3(1.0);
    float  transIor = 1.5;
    int    useSpecularWorkflow = 0;
    float3 specularColor = float3(0.04);
    float  baseWeight = 1.0;
    float  specularWeight = 1.0;
    float3 standardSpecularColor = float3(1.0);
    float  sheenWeight = 0.0;
    float  sheenRoughness = 0.3;
    float3 sheenColor = float3(1.0);
    float  thinFilmThickness = 0.0;
    float  thinFilmIor = 1.5;
    float  specularAnisotropy = 0.0;
    bool   useStandardSurface = false;
    if (u.has_materials != 0u && material_index >= 0) {
        RasterMaterialParams mp = mat_params[material_index];
        float2 texUV = material_uv(mp, in.uv);
        if (mp.use_vertex_color == 0) {
            albedo = mp.base_color.rgb;
            if ((mp.procedural_flags & 1) != 0 && mp.procedural_base != 0) {
                albedo = evalMtlxMarble3d(mp, in.objectPos);
            } else if (mp.procedural_node_count > 0 && mp.procedural_base_out >= 0) {
                albedo = evalMtlxProcGraph(mp, mp.procedural_base_out,
                                           in.objectPos, texUV,
                                           mat_textures, mat_sampler).rgb;
            } else if (mp.procedural_kind == 1 && mp.procedural_base != 0) {
                albedo = evalMtlxMarble3d(mp, in.objectPos);
            }
        }
        metallic  = mp.metallic;
        roughness = mp.roughness;
        if (mp.procedural_node_count > 0 && mp.procedural_rough_out >= 0) {
            roughness = evalMtlxProcGraph(mp, mp.procedural_rough_out,
                                          in.objectPos, texUV,
                                          mat_textures, mat_sampler).x;
        }
        alpha     = mp.base_color.a * mp.opacity;
        emissive  = mp.emissive_color.rgb * mp.emissive_color.a;
        ior       = mp.ior;
        ao        = mp.occlusion;
        transWeight = mp.transmission_weight;
        transTint = mp.transmission_color.rgb;
        transIor = (mp.transmission_ior > 0.0) ? mp.transmission_ior : ior;
        useSpecularWorkflow = mp.use_specular_wf;
        specularColor = mp.specular_color.rgb;
        if (mp.standard_surface != 0) {
            useStandardSurface = true;
            baseWeight = clamp(mp.base_weight, 0.0, 1.0);
            specularWeight = clamp(mp.specular_weight, 0.0, 1.0);
            standardSpecularColor = max(mp.specular_color.rgb, float3(0.0));
            sheenWeight = clamp(mp.sheen_weight, 0.0, 1.0);
            sheenRoughness = clamp(mp.sheen_roughness, 0.0, 1.0);
            sheenColor = max(mp.sheen_color.rgb, float3(0.0));
            thinFilmThickness = max(mp.thin_film_thickness, 0.0);
            thinFilmIor = max(mp.thin_film_ior, 1.001);
            specularAnisotropy = clamp(abs(mp.specular_anisotropy), 0.0, 1.0);
        }
        if (mp.tex_diffuse >= 0 && mp.tex_diffuse < 64) {
            float4 t = mat_textures[mp.tex_diffuse].sample(mat_sampler, texUV);
            albedo *= t.rgb;
            alpha *= t.a;
        }
        /* Roughness from .g (ORM convention: G=Roughness). */
        if (mp.tex_roughness >= 0 && mp.tex_roughness < 64) {
            roughness = mat_textures[mp.tex_roughness].sample(mat_sampler, texUV).g
                      * mp.roughness_xform.x + mp.roughness_xform.y;
        }
        /* Metallic from .b (ORM convention: B=Metallic). */
        if (mp.tex_metallic >= 0 && mp.tex_metallic < 64) {
            metallic = mat_textures[mp.tex_metallic].sample(mat_sampler, texUV).b;
        }
        /* Normal-map perturbation, matching raytrace.metal:shade_hit's
         * geometric-ONB approach and Vulkan's UsdUVTexture scale/bias
         * remap for TEX_NORMAL. */
        bool texNormal = (mp.tex_normal >= 0 && mp.tex_normal < 64);
        bool graphNormal = (!texNormal &&
                            mp.procedural_node_count > 0 &&
                            mp.procedural_normal_out >= 0);
        if (texNormal || graphNormal) {
            float3 nm = float3(0.0, 0.0, 1.0);
            if (graphNormal) {
                nm = evalMtlxProcGraph(mp, mp.procedural_normal_out,
                                       in.objectPos, texUV,
                                       mat_textures, mat_sampler).rgb
                   * mp.normal_tex_scale.rgb + mp.normal_tex_bias.rgb;
            } else {
                nm = mat_textures[mp.tex_normal].sample(mat_sampler, texUV).rgb
                   * mp.normal_tex_scale.rgb + mp.normal_tex_bias.rgb;
            }
            float3 T, B;
            branchlessONB(N, T, B);
            N = normalize(T * nm.x * mp.normal_scale
                        + B * nm.y * mp.normal_scale
                        + N * nm.z);
        }
        /* Emissive: emissive_color * intensity, with optional texture. */
        if (mp.tex_emissive >= 0 && mp.tex_emissive < 64) {
            float3 etex = mat_textures[mp.tex_emissive].sample(mat_sampler, texUV).rgb;
            if (dot(emissive, emissive) < 0.001) {
                /* Constant emissive is black — texture IS the emissive color. */
                emissive = etex * mp.emissive_color.a;
            } else {
                emissive *= etex;
            }
        }
        /* AO from .r channel (ORM convention: R=AO). */
        if (mp.tex_occlusion >= 0 && mp.tex_occlusion < 64) {
            ao = mat_textures[mp.tex_occlusion].sample(mat_sampler, texUV).r;
        }
        if (mp.tex_opacity >= 0 && mp.tex_opacity < 64) {
            float4 opa = mat_textures[mp.tex_opacity].sample(mat_sampler, texUV);
            alpha *= (opa.r < 1.0) ? opa.r : opa.a;
        }
        if (mp.opacity_threshold > 0.0 && alpha < mp.opacity_threshold) {
            discard_fragment();
        }
        if (mp.tex_trans_weight >= 0 && mp.tex_trans_weight < 64) {
            transWeight *= mat_textures[mp.tex_trans_weight].sample(mat_sampler, texUV).r;
        }
    }

    roughness = clamp(roughness * mix(1.0, 0.72, specularAnisotropy), 0.04, 1.0);
    if (u.debug_mode == 1u) return float4(albedo, 1.0);
    if (u.debug_mode == 2u) return float4(float3(metallic), 1.0);
    if (u.debug_mode == 3u) return float4(float3(roughness), 1.0);
    if (u.debug_mode == 4u) return float4(N * 0.5 + 0.5, 1.0);

    /* ---- Cook-Torrance PBR setup ---- */
    float3 V = normalize(eye_pos - in.worldPos);
    if (dot(N, V) < 0.0) {
        N = -N;
    }
    float NdotV = max(dot(N, V), 0.001);
    float f0_dielectric = pow((ior - 1.0) / (ior + 1.0), 2.0);
    float3 F0;
    if (useSpecularWorkflow != 0) {
        metallic = 0.0;
        F0 = specularColor;
    } else if (useStandardSurface) {
        F0 = mix(float3(f0_dielectric) * specularWeight * standardSpecularColor,
                 albedo, metallic);
    } else {
        F0 = mix(float3(f0_dielectric), albedo, metallic);
    }
    if (thinFilmThickness > 0.0) {
        float3 film = thinFilmTint(NdotV, thinFilmThickness, thinFilmIor);
        F0 = clamp(F0 * film + film * 0.08, 0.0, 1.0);
    }
    float3 diffuseAlbedo = albedo * baseWeight;

    bool has_ibl = u.has_ibl != 0u;

    /* ---- Direct lighting (key/fill/rim) — always added, attenuated
     * when IBL is on. This matches the OpenGL/Vulkan raster family:
     * key+fill stay on under IBL so dielectric albedo keeps a diffuse
     * baseline instead of letting env specular wash the chess pieces out. */
    float lightScale = has_ibl ? 0.5 : 0.7;
    float fillScale  = has_ibl ? 0.2 : 0.2;
    float rimScale   = has_ibl ? 0.0 : 0.15;
    float3 Lo = float3(0.0);
    {
        /* Key light (warm, dominant). */
        float3 keyDir = normalize(float3(0.3, 1.0, 0.5));
        {
            float3 L = keyDir;
            float3 H = normalize(V + L);
            float NdotL = max(dot(N, L), 0.0);
            if (NdotL > 0.0) {
                float D = distributionGGX(N, H, roughness);
                float G = geometrySmithSchlick(N, V, L, roughness);
                float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
                float3 spec = (D * G * F) /
                              (4.0 * NdotV * NdotL + 1e-4);
                float3 kS = F;
                float3 kD = (float3(1.0) - kS) * (1.0 - metallic);
                Lo += (kD * diffuseAlbedo / 3.14159265 + spec) *
                      float3(1.0, 0.95, 0.85) * lightScale * NdotL;
            }
        }

        /* Fill light (cool, soft) — diffuse-only Lambert. */
        float3 fillDir = normalize(float3(-0.5, 0.4, -0.3));
        Lo += diffuseAlbedo * (1.0 - metallic) * float3(0.7, 0.8, 1.0) * fillScale
              * max(dot(N, fillDir), 0.0);

        if (rimScale > 0.0) {
            /* Rim light (no-IBL only — the dome already provides backlight). */
            float3 rimDir = normalize(float3(0.0, 0.3, -1.0));
            Lo += diffuseAlbedo * (1.0 - metallic) * float3(1.0, 0.95, 0.9) * rimScale
                  * max(dot(N, rimDir), 0.0);
        }
    }

    /* ---- Ambient — IBL when env loaded, procedural hemisphere otherwise --- */
    float3 ambientTerm;
    if (has_ibl) {
        /* Diffuse IBL: equirectangular irradiance map sampled at N.
         * Specular IBL: match the OpenGL/Vulkan raster block, not Metal RT.
         * OpenGL uses box-filtered env mips and blends rough reflections
         * toward irradiance/pi; Metal's env mips are GGX-prefiltered, so we
         * keep the same roughness blend and apply a Metal-specific specular
         * compensation scale below.
         *
         * Address mode: S=repeat (longitude wraps), T=clamp_to_edge
         * (latitude clamps at poles). The earlier single `address::repeat`
         * flag set BOTH axes to repeat, which made V wrap at the poles
         * — sampling near v=0 or v=1 ended up blending across the seam
         * and gave the wrong-side hemisphere's irradiance. On the chess
         * board's upward-facing surfaces this manifested as a brown
         * tint dragged in from the lower hemisphere where photo studio's
         * warm floor sits. Matching raytrace.metal:env_sampler's
         * (S=repeat, T=clamp_to_edge) fixes it. */
        constexpr sampler ibl_samp(filter::linear, mip_filter::linear,
                                   s_address::repeat, t_address::clamp_to_edge);
        constexpr sampler brdf_samp(filter::linear, mip_filter::nearest,
                                    s_address::clamp_to_edge,
                                    t_address::clamp_to_edge);
        float3 R = reflect(-V, N);
        float3 irradiance = irr_tex.sample(ibl_samp, dirToEquirect(N)).rgb;
        float3 kS_ibl = fresnelSchlickRoughness(NdotV, F0, roughness);
        float3 kD_ibl = (float3(1.0) - kS_ibl) * (1.0 - metallic);
        float3 diffuseIBL = kD_ibl * irradiance * (diffuseAlbedo / 3.14159265);

        /* Specular IBL — split-sum with the precomputed BRDF integration
         * LUT + Kulla-Conty multi-scatter compensation. The rough-surface
         * blend below mirrors the OpenGL raster path; Metal's GGX-prefiltered
         * mip chain is stronger than OpenGL's box-filtered mips, so the small
         * compensation scale keeps raster exposure comparable. */
        float lod = roughness * max(u.env_mip_levels - 1.0, 0.0);
        float3 envSpecular = env_tex.sample(ibl_samp, dirToEquirect(R), level(lod)).rgb;
        float roughBlend = smoothstep(0.3, 0.9, roughness);
        float3 prefiltered = mix(envSpecular, irradiance / 3.14159265, roughBlend);
        float2 ab = brdf_tex.sample(brdf_samp, float2(NdotV, roughness)).xy;
        float Ess = ab.x + ab.y;
        float3 energyComp = 1.0 + F0 * (1.0 / max(Ess, 0.001) - 1.0);
        float3 specularIBL = prefiltered * (kS_ibl * ab.x + ab.y) * energyComp;
        specularIBL *= mix(0.055, 0.07, metallic);

        if (u.debug_mode == 5u) return float4(acesFilmic(diffuseIBL * 0.5), 1.0);
        if (u.debug_mode == 6u) return float4(acesFilmic(specularIBL * 0.5), 1.0);

        /* Keep the old env-tap approximation out of the raster path. It was
         * intended as a fake indirect bounce, but with the chess HDR it lifted
         * the board into a beige wash and made raster diverge from OpenGL. */
        /* No env_intensity multiplier here: env_texture/irr_texture are
         * already env_scale-normalized at upload. OpenGL/Vulkan raster trust
         * those auto-exposed textures; multiplying by DomeLight intensity
         * would blow out intensity=1000 chess wrappers. */
        ambientTerm = (diffuseIBL + specularIBL) * ao;
    } else {
        /* Procedural sky/ground hemisphere — preserves chess and other
         * pre-IBL scenes bit-stable. Original parity-tuned constants.
         * Y-up world (chess set and most authored USDs); Vulkan's port
         * flipped to Z-up for IsaacLab/Newton — we keep Y-up here. */
        float dome_intensity = max(u.dome_color.a, 0.0);
        float3 flatDome = (dome_intensity > 0.0)
            ? clamp(u.dome_color.rgb * dome_intensity, float3(0.0), float3(8.0))
            : float3(0.0);
        float3 skyColor    = (dome_intensity > 0.0)
            ? mix(float3(0.50, 0.58, 0.75), flatDome, 0.85)
            : float3(0.50, 0.58, 0.75);
        float3 groundColor = (dome_intensity > 0.0)
            ? mix(float3(0.22, 0.20, 0.17), flatDome * 0.28, 0.75)
            : float3(0.22, 0.20, 0.17);
        float hemi = dot(N, float3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
        float3 ambient = mix(groundColor, skyColor, hemi) + float3(0.08, 0.08, 0.10);
        float3 kS_amb = fresnelSchlick(NdotV, F0);
        float3 kD_amb = (float3(1.0) - kS_amb) * (1.0 - metallic);
        ambientTerm = (kD_amb * ambient * diffuseAlbedo + kS_amb * ambient * 0.06) * ao;

        /* Many authored lights + no IBL (e.g. the NVIDIA Isaac Sim warehouse:
         * 41 SphereLights + a flat DomeLight, no HDRI). The high-intensity flat
         * dome saturates this procedural hemisphere ambient and blows the tall
         * interior out to white; the authored lights already carry the key
         * illumination. Tame the procedural fill — hard on high/vertical
         * structure, gently near the floor — mirroring the RT many-light path
         * (raytrace.metal) and the Vulkan/OpenGL "close warehouse lighting
         * gaps" tuning. Gated on light_count so few-light chess/apple stay
         * bit-stable; the gate implies an Isaac-style Z-up scene, so height and
         * the floor hemisphere are keyed on +Z. */
        if (u.light_pad.x > 8u) {
            float zUp = clamp(N.z, 0.0, 1.0);
            float floorShape = smoothstep(0.15, 0.85, zUp) *
                               (1.0 - smoothstep(0.05, 0.60, in.worldPos.z));
            float highInterior = smoothstep(1.2, 3.5, in.worldPos.z);
            float nonFloorScale = mix(0.11, 0.05, highInterior);
            ambientTerm *= mix(nonFloorScale, 0.13, floorShape);
        }
    }

    /* Final color. The IBL branch tracks authored DomeLight intensity through
     * the same bounded exposure curve as the Vulkan raster path: the HDRI is
     * auto-exposed during upload, while this scalar keeps intensity=1000
     * studio domes from tonemapping every BRDF at raw exposure 1.0. */
    float3 sheenTerm = float3(0.0);
    if (sheenWeight > 0.0) {
        float grazing = pow(clamp(1.0 - NdotV, 0.0, 1.0),
                            mix(5.0, 1.4, sheenRoughness));
        float3 sheenLight = has_ibl ? ambientTerm * 0.7 : float3(0.18, 0.16, 0.22);
        sheenTerm = sheenColor * sheenLight * sheenWeight * grazing;
    }
    float3 color = ambientTerm + Lo + sheenTerm + emissive;

    /* Warehouse (many authored lights + no IBL): the baked direct-light rig
     * over-lights the painted walls/floor/ceiling to near-white. Port of the
     * Vulkan/OpenGL "close warehouse lighting gaps" final-color darkening:
     * scale the composed color down — hardest on high/vertical interior
     * structure, gentler near the floor — keyed on +Z (Isaac-style Z-up). This
     * darkens direct light too, unlike the ambient-only term above. Gated on
     * light_count so few-light chess/apple are bit-stable. */
    if (u.light_pad.x > 8u) {
        float zUp = clamp(N.z, 0.0, 1.0);
        float floorShape = smoothstep(0.15, 0.85, zUp) *
                           (1.0 - smoothstep(0.05, 0.60, in.worldPos.z));
        float highInterior = smoothstep(1.2, 3.5, in.worldPos.z);
        /* Warehouse (many authored lights, no IBL): single-pass raster has no
         * path-traced occlusion, so the interior runs bright vs the OVRTX golden
         * and the painted floor washes from its authored blue toward grey. Keep
         * the warm, product-lit racks visible (cooling + crushing them with AO
         * read far worse than the golden, not better), apply only a gentle Z-up
         * exposure trim, and restore the floor's blue. Tuned by eye against the
         * OVRTX golden, not by FLIP alone. Gated to warehouse so studio scenes
         * stay bit-identical. */
        float nonFloorScale = mix(0.30, 0.20, highInterior);
        color *= mix(nonFloorScale, 0.45, floorShape);
        color *= mix(float3(1.0), float3(0.82, 1.0, 1.16), floorShape); // floor -> authored blue
    }

    /* Studio scenes (chess, apple): few authored lights + a flat dome, no IBL.
     * Single-pass with no path-traced occlusion renders the subjects
     * systematically brighter than the OVRTX golden (raster +36%..+143% luma).
     * Pull exposure down with a flat, hue-neutral scale: the per-scene hue
     * error is bidirectional (chess reads cooler, apple warmer), so a colour
     * shift would regress one set — a neutral exposure cut nets every studio
     * frame closer. The 0.16 factor is the FLIP minimum vs the OVRTX golden
     * (swept on this Mac: mean studio raster FLIP 0.721 -> 0.343, every
     * chess+apple frame improved, none regressed; below ~0.15 the dimmer
     * subjects overshoot to too-dark and FLIP climbs again). The factor is
     * this low because the 16-bit-float target lets the cut un-clip recovered
     * highlights rather than just darken. Gated to few-light/no-IBL so
     * warehouse (many-light) and IBL scenes stay bit-stable. */
    if (!has_ibl && u.light_pad.x <= 8u) {
        color *= 0.16;
    }

    float transmissionMix = clamp(max(transWeight, 1.0 - alpha), 0.0, 1.0);
    if (has_ibl && transmissionMix > 0.001) {
        constexpr sampler ibl_samp(filter::linear, mip_filter::linear,
                                   s_address::repeat, t_address::clamp_to_edge);
        bool clearGlass = (transTint.r > 0.95 &&
                           transTint.g > 0.95 &&
                           transTint.b > 0.95);
        float3 rayDir = -V;
        bool entering = dot(rayDir, N) < 0.0;
        float3 Nr = entering ? N : -N;
        float eta = entering ? (1.0 / max(transIor, 1.001)) : max(transIor, 1.001);
        float3 throughDir;
        if (clearGlass) {
            float3 T = refract(rayDir, Nr, eta);
            throughDir = (dot(T, T) < 1e-6) ? reflect(rayDir, Nr) : T;
        } else {
            throughDir = rayDir;
        }
        float3 R = reflect(rayDir, N);
        float3 transmitted = env_tex.sample(ibl_samp, dirToEquirect(throughDir), level(0.0)).rgb;
        if (transWeight > 0.001) transmitted *= transTint;
        if (alpha < 0.5) transmitted *= albedo;
        float3 reflected = env_tex.sample(ibl_samp, dirToEquirect(R), level(0.0)).rgb;
        float f0Glass = pow((transIor - 1.0) / (transIor + 1.0), 2.0);
        float fresnel = f0Glass + (1.0 - f0Glass) * pow(1.0 - clamp(NdotV, 0.0, 1.0), 5.0);
        float3 glassColor = mix(transmitted, reflected, fresnel);
        color = mix(color, glassColor, transmissionMix);
    }

    if (has_ibl) {
        float exposure = rasterEnvSurfaceExposure(u.env_intensity);
        color = acesFilmic(color * exposure);
    }
    return float4(color, 1.0);
}

fragment float4 fragment_mesh_geom(
    VertexOut                      in           [[stage_in]],
    constant MeshPushConstants&    pc           [[buffer(1)]])
{
    float3 N = normalize(in.normal);
    float3 albedo = in.color;
    float metallic = 0.0;
    float roughness = clamp(0.5, 0.04, 1.0);
    float ao = 1.0;
    float ior = 1.5;

    float3 V = normalize(pc.eye_pos.xyz - in.worldPos);
    if (dot(N, V) < 0.0) {
        N = -N;
    }
    float f0_dielectric = pow((ior - 1.0) / (ior + 1.0), 2.0);
    float3 F0 = float3(f0_dielectric);
    float NdotV = max(dot(N, V), 0.001);

    /* Outdoor "tropical island" lighting for the geometry-only/clay viz path
     * (this shader is never used when materials are on — that's fragment_mesh).
     * A strong, warm, lateral sun defines form; a cool sky bounce + warm
     * ground bounce fill the shadow side; ACES tonemap + a small saturation
     * lift turn the formerly washed-out, untonemapped output into a punchy
     * sunlit look. */
    float lightScale = 1.05;
    float fillScale  = 0.18;
    float rimScale   = 0.16;
    float3 Lo = float3(0.0);
    {
        /* Lateral afternoon sun — lower elevation reads form far better than
         * the old near-overhead (0.3,1,0.5). */
        float3 keyDir = normalize(float3(0.55, 0.62, 0.40));
        {
            float3 L = keyDir;
            float3 H = normalize(V + L);
            float NdotL = max(dot(N, L), 0.0);
            if (NdotL > 0.0) {
                float D = distributionGGX(N, H, roughness);
                float G = geometrySmithSchlick(N, V, L, roughness);
                float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
                float3 spec = (D * G * F) /
                              (4.0 * NdotV * NdotL + 1e-4);
                float3 kS = F;
                float3 kD = (float3(1.0) - kS) * (1.0 - metallic);
                Lo += (kD * albedo / 3.14159265 + spec) *
                      float3(1.0, 0.93, 0.78) * lightScale * NdotL;
            }
        }

        /* Cool sky fill from the opposite side. */
        float3 fillDir = normalize(float3(-0.5, 0.5, -0.3));
        Lo += albedo * (1.0 - metallic) * float3(0.55, 0.70, 1.0) * fillScale
              * max(dot(N, fillDir), 0.0);

        if (rimScale > 0.0) {
            float3 rimDir = normalize(float3(0.0, 0.3, -1.0));
            Lo += albedo * (1.0 - metallic) * float3(1.0, 0.95, 0.9) * rimScale
                  * max(dot(N, rimDir), 0.0);
        }
    }

    /* Hemisphere ambient — dialed back vs. the old wash, warmer ground so the
     * shadow side isn't a flat blue-gray. */
    float3 skyColor    = float3(0.34, 0.46, 0.66);
    float3 groundColor = float3(0.20, 0.17, 0.12);
    float hemi = dot(N, float3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
    float3 ambient = mix(groundColor, skyColor, hemi) + float3(0.04, 0.045, 0.05);
    float3 kS_amb = fresnelSchlick(NdotV, F0);
    float3 kD_amb = (float3(1.0) - kS_amb) * (1.0 - metallic);
    float3 ambientTerm = (kD_amb * ambient * albedo + kS_amb * ambient * 0.05) * ao;

    /* Compose, saturate slightly, expose, and ACES-tonemap (the old path
     * returned raw linear → highlights clipped to white = the washout). */
    float3 color = ambientTerm + Lo;
    float lum = dot(color, float3(0.2126, 0.7152, 0.0722));
    color = mix(float3(lum), color, 1.22);          // +22% saturation
    color = acesFilmic(color * 1.45);               // exposure + filmic rolloff
    return float4(color, 1.0);
}

fragment float4 fragment_mesh(
    VertexOut                      in           [[stage_in]],
    constant MeshPushConstants&    pc           [[buffer(1)]],
    device const RasterMaterialParams* mat_params [[buffer(2)]],
    constant RasterFragUniform&    u            [[buffer(3)]],
    array<texture2d<float>, 64>    mat_textures [[texture(0)]],
    texture2d<float>               env_tex      [[texture(64)]],
    texture2d<float>               irr_tex      [[texture(65)]],
    texture2d<float>               brdf_tex     [[texture(66)]],
    sampler                        mat_sampler  [[sampler(0)]])
{
    return shade_mesh(in, mat_params, mat_textures, mat_sampler,
                      env_tex, irr_tex, brdf_tex, u,
                      (int)round(in.material_index), pc.eye_pos.xyz);
}

/* Shadow variant: same raster BRDF as fragment_mesh, plus mesh-only TLAS
 * queries that attenuate contact/visibility regions. This keeps raster and
 * raster-shadow math aligned while giving shadow mode an actual occlusion
 * signal. */
fragment float4 fragment_mesh_shadow(
    VertexOut                      in           [[stage_in]],
    constant MeshPushConstants&    pc           [[buffer(1)]],
    device const RasterMaterialParams* mat_params [[buffer(2)]],
    constant RasterFragUniform&    u            [[buffer(3)]],
    instance_acceleration_structure accel        [[buffer(4)]],
    array<texture2d<float>, 64>    mat_textures [[texture(0)]],
    texture2d<float>               env_tex      [[texture(64)]],
    texture2d<float>               irr_tex      [[texture(65)]],
    texture2d<float>               brdf_tex     [[texture(66)]],
    sampler                        mat_sampler  [[sampler(0)]])
{
    float4 shaded = shade_mesh(in, mat_params, mat_textures, mat_sampler,
                               env_tex, irr_tex, brdf_tex, u,
                               (int)round(in.material_index), pc.eye_pos.xyz);
    if (u.debug_mode > 0u && u.debug_mode <= 6u) return shaded;

    float3 N = normalize(in.normal);
    float3 V = normalize(pc.eye_pos.xyz - in.worldPos);
    float3 R = reflect(-V, N);
    float3 origin = in.worldPos + N * 0.002;

    /* Normal-direction probe: cheap IBL/contact occlusion. On the chess
     * board's upward faces this catches pieces sitting over the square; on
     * vertical piece sides the smoothstep keeps the effect subtle. */
    float normalLit = trace_mesh_visibility(origin, N, 5.0, accel);
    float upWeight = smoothstep(0.15, 0.80, N.y);
    float contactFactor = mix(1.0, mix(0.48, 1.0, normalLit), upWeight);
    if (upWeight > 0.0) {
        float3 T, B;
        branchlessONB(N, T, B);
        float aoLit = 0.0;
        aoLit += trace_mesh_visibility(origin, normalize(N * 0.65 + T * 0.76), 0.10, accel);
        aoLit += trace_mesh_visibility(origin, normalize(N * 0.65 - T * 0.76), 0.10, accel);
        aoLit += trace_mesh_visibility(origin, normalize(N * 0.65 + B * 0.76), 0.10, accel);
        aoLit += trace_mesh_visibility(origin, normalize(N * 0.65 - B * 0.76), 0.10, accel);
        aoLit *= 0.25;
        float aoFactor = mix(0.62, 1.0, aoLit);
        contactFactor = min(contactFactor, mix(1.0, aoFactor, upWeight));
    }

    /* Reflection-direction probe: damps env/specular where a nearby mesh blocks
     * the reflected ray. Kept mild because this is applied after tonemap. */
    float reflLit = 1.0;
    if (dot(R, N) > 0.001) {
        reflLit = trace_mesh_visibility(origin, normalize(R), 5.0, accel);
    }
    float specFactor = mix(0.84, 1.0, reflLit);

    /* Directional key-light probe, matching the raster BRDF's key direction.
     * This produces the broad piece-to-board shadows that plain IBL lacks. */
    float3 keyDir = normalize(float3(0.3, 1.0, 0.5));
    float keyLit = 1.0;
    if (dot(N, keyDir) > 0.0) {
        keyLit = trace_mesh_visibility(origin, keyDir, 1.0e5, accel);
    }
    float keyFactor = mix(0.78, 1.0, keyLit);

    float shadowFactor = contactFactor * specFactor * keyFactor;
    if (u.debug_mode == 7u)  return float4(float3(contactFactor), 1.0);
    if (u.debug_mode == 8u)  return float4(float3(specFactor), 1.0);
    if (u.debug_mode == 9u)  return float4(float3(keyFactor), 1.0);
    if (u.debug_mode == 10u) return float4(float3(shadowFactor), 1.0);

    shaded.rgb *= shadowFactor;
    return shaded;
}

/* ---- Environment background (full-screen triangle) ----
 * MSL port of nanousd-opengl-renderer/src/shaders_gles.h:k_env_bg_*.
 * Reconstructs a world-space ray direction from the inverse-view + inverse-
 * proj matrices, samples the GGX-prefiltered HDR equirect map at mip 0
 * (full-resolution sky), ACES-tonemaps, and writes to the sRGB color
 * target — the format does the gamma encode, so no manual pow(., 1/2.2)
 * is needed (unlike the GLES port).
 *
 * Drawn from renderer.c's raster path right before gpu_end_frame:
 * gl_Position.z = 0.9999 puts these pixels just inside the far plane, so
 * the depth-less-equal mesh draws above clamp them away wherever real
 * geometry is in front. Without this pass nanousdview's Metal viewport
 * shows the gpu_begin_frame clear color (sky-blue) instead of the
 * authored DomeLight HDR. */

struct EnvBgVertOut {
    float4 position [[position]];
    float2 v_uv;
};

vertex EnvBgVertOut vertex_env_bg(uint vid [[vertex_id]])
{
    /* Single fullscreen triangle (vid=0,1,2 → corner positions). */
    float2 pos = float2(float((vid & 1u)) * 4.0 - 1.0,
                        float((vid >> 1u) & 1u) * 4.0 - 1.0);
    EnvBgVertOut out;
    /* Y-flip v_uv vs the GLES port: Metal's framebuffer y axis is the
     * opposite of the GLES one, so for v_uv to feed `d = v_uv*2-1` and
     * land on the same world-space ray direction the GLES env_bg
     * computes from `u_projInv * vec(d.x,d.y,1,1)`, we flip d.y at the
     * source. Without it the env reads at -y and the photo-studio
     * appears upside-down — ceiling at the top of frame, chess board
     * floating above eye-line. */
    out.v_uv     = float2(pos.x, -pos.y) * 0.5 + 0.5;
    /* z = 0.9999 — same as the GLES port, keeps depth tests happy. */
    out.position = float4(pos, 0.9999, 1.0);
    return out;
}

struct EnvBgPushConstants {
    float4x4 view_inv;
    float4x4 proj_inv;
    float4   env_params; /* x = DomeLight intensity */
};

fragment float4 fragment_env_bg(
    EnvBgVertOut                 in       [[stage_in]],
    constant EnvBgPushConstants& pc       [[buffer(1)]],
    texture2d<float>             env_tex  [[texture(0)]])
{
    /* Reconstruct world-space ray direction. C-side stores row-major; Metal
     * is column-major, so we use `vec * mat` to match the GLES port's
     * `u_projInv * vec` GLSL semantics (same effective transpose). */
    float2 d  = in.v_uv * 2.0 - 1.0;
    float4 target  = float4(d.x, d.y, 1.0, 1.0)               * pc.proj_inv;
    float3 viewDir = normalize(target.xyz);
    float3 dir     = (float4(viewDir, 0.0) * pc.view_inv).xyz;

    /* Equirect lookup using the same convention as raytrace.metal /
     * mesh.metal's dirToEquirect — keeps the env background pixel-aligned
     * with what IBL + RT pull from the env map. */
    float2 uv = dirToEquirect(dir);

    constexpr sampler env_samp(filter::linear, mip_filter::linear,
                               s_address::repeat, t_address::clamp_to_edge);
    /* Force mip 0 (full resolution) — fullscreen-triangle UV derivatives
     * are huge and would otherwise pick the lowest mip (1×1 average), per
     * the comment in shaders_gles.h:540. */
    float3 color = env_tex.sample(env_samp, uv, level(0.0)).rgb;
    color *= rasterEnvSkyScale(pc.env_params.x);
    color = acesFilmic(color);
    /* sRGB color target handles gamma — no manual pow encode. */
    return float4(color, 1.0);
}

/* ---- SSAO post-process (screen-space ambient occlusion) ----------------
 * A separate fullscreen pass run AFTER the raster pass: reads the rendered
 * color + the scene depth, estimates contact occlusion from neighbouring
 * depths, and darkens the color in crevices/contacts to ground the geometry.
 * Depth-based (no normal G-buffer) and convention-robust: it only compares
 * LINEARIZED view-space distances in a screen-space disk, so it does not
 * depend on the exact projection form. Tuned for stills (many samples, no
 * separate blur). */
struct SsaoParams {
    float near_z;
    float far_z;
    float radius;      // world-space sample radius
    float strength;    // 0..1 occlusion darkening
    float bias;        // depth bias to avoid self-occlusion
    float power;       // contrast on the AO term
    float2 inv_res;    // 1/width, 1/height
};

struct FsOut { float4 pos [[position]]; float2 uv; };

vertex FsOut vertex_ssao(uint vid [[vertex_id]])
{
    /* Fullscreen triangle. */
    float2 p = float2((vid == 2u) ? 3.0 : -1.0, (vid == 1u) ? 3.0 : -1.0);
    FsOut o;
    o.pos = float4(p, 0.0, 1.0);
    o.uv  = float2((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
    return o;
}

static inline float ssao_linearize(float d, float n, float f)
{
    /* Metal NDC depth in [0,1], standard (non-reversed) projection. */
    return (n * f) / max(f - d * (f - n), 1e-5);
}

fragment float4 fragment_ssao(
    FsOut in [[stage_in]],
    constant SsaoParams& p          [[buffer(0)]],
    texture2d<float> colorTex       [[texture(0)]],
    depth2d<float>   depthTex       [[texture(1)]])
{
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    constexpr sampler sd(filter::nearest, address::clamp_to_edge);
    float3 color = colorTex.sample(s, in.uv).rgb;

    float d = depthTex.sample(sd, in.uv);
    if (d >= 0.9999) return float4(color, 1.0);   // sky / background
    float centerLin = ssao_linearize(d, p.near_z, p.far_z);

    /* 16-tap spiral disk; screen-space radius scaled so a fixed world radius
     * projects smaller with distance. */
    const int N = 16;
    float radiusUV = clamp(p.radius / max(centerLin, 1e-3), 0.0, 0.08);
    float occ = 0.0;
    float golden = 2.39996323;
    for (int i = 0; i < N; i++) {
        float ang = float(i) * golden;
        float rad = radiusUV * sqrt((float(i) + 0.5) / float(N));
        float2 off = float2(cos(ang), sin(ang)) * rad;
        float sd_ = depthTex.sample(sd, in.uv + off);
        if (sd_ >= 0.9999) continue;
        float sampLin = ssao_linearize(sd_, p.near_z, p.far_z);
        float diff = centerLin - sampLin;           // >0 : sample is an occluder in front
        float range = smoothstep(0.0, 1.0, p.radius / max(abs(diff), 1e-4));
        if (diff > p.bias) occ += range;
    }
    occ /= float(N);
    float ao = 1.0 - clamp(occ * p.strength, 0.0, 1.0);
    ao = pow(ao, p.power);
    return float4(color * ao, 1.0);
}
