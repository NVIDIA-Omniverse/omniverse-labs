// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInEXT vec4 hitValue;
hitAttributeEXT vec2 baryCoord;

/* Binding 0: TLAS for shadow and glass continuation rays */
layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

/* Per-mesh geometry addresses + color (indexed by gl_InstanceCustomIndexEXT) */
struct MeshData {
    uint64_t vertexAddress;
    uint64_t indexAddress;
    vec4     color;          /* per-mesh display color (.w unused, always 1.0) */
    uint     tex_index;      /* runtime texture index (0xFFFFFFFF = no texture) */
    uint     source_id;      /* public renderer mesh id for probes/readback */
    uint     material_id;    /* per-instance material id */
    uint     ptex_color_offset; /* offset into binding 18, 0xFFFFFFFF = none */
};
layout(set = 0, binding = 2, scalar) buffer SceneData {
    uint     vertexStride;   /* floats per vertex (6 = basic, 12 = material) */
    uint     hasMaterials;   /* 1 if material SSBO is bound */
    float    envMipLevels;   /* 0 = no IBL, >0 = IBL environment mip count */
    float    envIntensity;   /* USD DomeLight intensity (rmiss sky multiplier) */
    vec4     domeColor;      /* fast_mode flat sky/ambient: rgb + intensity */
    uint     upAxis;         /* 0=X, 1=Y, 2=Z */
    uint     _scenePad0;
    uint     _scenePad1;
    uint     _scenePad2;
    MeshData meshes[];
} scene;

/* Material parameters SSBO (binding 3) — std430 layout must match
 * GpuMaterialParams in src/gpu.h byte-for-byte, otherwise materials[i]
 * reads from the wrong offset for i >= 1. The Phase 7c trailing block
 * (subsurface/transmission) needs to live here even though the BSDF
 * doesn't consume it yet, just to keep the array stride in sync. */
struct MaterialData {
    vec4  base_color;       /* rgb + alpha */
    vec4  emissive_color;   /* rgb + intensity */
    float metallic;
    float roughness;
    float opacity;
    float ior;
    float occlusion;
    float clearcoat;
    float clearcoat_roughness;
    float normal_scale;
    int   tex_indices[8];   /* -1 = no texture */
    int   use_vertex_color;
    float udim_scale_u;
    float udim_scale_v;
    float opacity_threshold;        /* UPS alpha-cutout threshold; 0 = disabled */
    /* Phase 7c trailing block — present in GpuMaterialParams. Inert in
     * the kernel until the SSS / transmission lobes land. */
    vec4  subsurface_color;
    vec4  subsurface_radius;
    vec4  transmission_color;
    float subsurface_weight;
    float subsurface_scale;
    float transmission_weight;
    float transmission_ior;
    int   tex_subsurface_weight;
    int   tex_transmission_weight;
    int   sss_color_authored;       /* 1 iff loader extracted a constant */
    int   use_specular_workflow;    /* 1 iff UsdPreviewSurface useSpecularWorkflow=1 */
    vec4  specular_color;           /* rgb (linear), w unused — UPS specular workflow F0 */
    vec4  normal_tex_scale;         /* TEX_NORMAL UsdUVTexture scale (default 2,2,2,1) */
    vec4  normal_tex_bias;          /* TEX_NORMAL UsdUVTexture bias  (default -1,-1,-1,0) */
    vec4  mdl_uv_transform;         /* xy scale + zw bias for Isaac MDL UVs */
    int   v_flip;                   /* 1 = sample texture V as 1-v for MDL */
    float roughness_tex_scale;
    float roughness_tex_bias;
    int   _pad_v_flip;
};

layout(set = 0, binding = 3, std430) readonly buffer MaterialBuffer {
    MaterialData materials[];
} mat_buf;

/* Binding 4: Texture array (combined image samplers) */
layout(set = 0, binding = 4) uniform sampler2D textures[];

/* Bindings 5-7: IBL (equirectangular environment, BRDF LUT, equirectangular irradiance) */
layout(set = 0, binding = 5) uniform sampler2D envMap;
layout(set = 0, binding = 6) uniform sampler2D brdfLUT;
layout(set = 0, binding = 7) uniform sampler2D irrMap;

/* Push constants — layout matches GpuRtTiledPushConstants / GpuRtPushConstants.
 * For tiled path, first 16 bytes are tile params (reinterpreted by raygen). */
/* Depth output SSBO (binding 10): one float per pixel, written by CHL and miss.
 * Guarded by depth_enabled push constant — single-camera pipeline sets 0. */
layout(set = 0, binding = 10, std430) writeonly buffer DepthOutput {
    float depths[];
} depthOut;

/* Segmentation output SSBO (binding 11): one uint32 per pixel.
 * Stores mesh_index+1 for hits, 0 for miss/sky.
 * Guarded by segmentation_enabled push constant. */
layout(set = 0, binding = 11, std430) writeonly buffer SegmentationOutput {
    uint ids[];
} segOut;

/* Normals output SSBO (binding 12): 3 floats (x,y,z) per pixel.
 * World-space surface normal. (0,0,0) for miss/sky.
 * Guarded by normals_enabled push constant. */
layout(set = 0, binding = 12, std430) writeonly buffer NormalsOutput {
    float normals[];
} normOut;

/* Scene lights SSBO (binding 13).
 * Header: nlights (uint) + 12 bytes of pad to 16-byte align the array.
 * GpuLight matches the C struct in src/gpu.h — std430 packs each
 * vec3+scalar pair into 16 bytes (5 pairs = 80 bytes per light). */
#define LIGHT_KIND_RECT     0
#define LIGHT_KIND_DISTANT  1
/* SphereLight: emitter at u_axis[0] = radius; v_axis/normal unused. */
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

/* Phase A G-buffer SSBO (binding 17): 32 bytes per pixel, indexed by
 * pixelIdx = launch_y * launchSize_x + launch_x — same scheme rgen + AOVs use.
 * Written additively alongside the existing color/depth/seg/normal writes;
 * the existing color path is untouched. Phase B will consume this buffer
 * in a deferred-shading compute pass (see DEFERRED_SHADING_PLAN.md §8).
 *
 * NOTE: plan §5 specified binding 14, but bindings 14/15 are already taken
 * by curve segments + per-segment colors (Phase 11.A.2.5). Using 17 here;
 * binding 16 is the per-env TLAS array. */
struct GBufferEntry {
    uint  inst_and_flags;  /* bit 31: hit; bits 0-23: gl_InstanceCustomIndexEXT */
    uint  primitive_id;    /* gl_PrimitiveID */
    uint  bary_packed;     /* packUnorm2x16(baryCoord) */
    float hit_t;           /* gl_HitTEXT */
    float normal_x;        /* world-space surface normal (post-flip) */
    float normal_y;
    float normal_z;
    uint  _pad;            /* reserved: octahedral pack lands here in Phase B */
};
layout(set = 0, binding = 17, std430) buffer GBuffer {
    GBufferEntry entries[];
} gbuf;

/* Binding 18: real authored Ptex surface colors, sampled once per triangle
 * corner on the CPU and packed as RGBA8 in index order. */
layout(set = 0, binding = 18, std430) readonly buffer PtexTriangleColors {
    uint colors[];
} ptexTriColors;

layout(push_constant) uniform PushConstants {
    mat4  viewInverse;
    mat4  projInverse;
    float ground_y;
    float scene_scale;
    uint  fast_mode;              /* 1 = skip shadow rays for RL sensors */
    uint  depth_enabled;          /* 1 = write depth to binding 10 SSBO */
    uint  segmentation_enabled;   /* 1 = write instance IDs to binding 11 SSBO */
    uint  normals_enabled;        /* 1 = write normals to binding 12 SSBO */
    uint  deferred_shade_enabled; /* Phase B: 1 = also write G-buffer in fast_mode */
    float tone_exposure_scale;
    float tone_sky_scale;
    float tone_white_point;
    uint  tone_flags;
    float rt_ibl_fill_scale;
};

const uint TONE_FLAG_CLAY_VIZ = 0x40000000u;
const uint TONE_FLAG_SKIP_SECONDARY_VISIBILITY = 0x20000000u;
const uint TONE_FLAG_SKIP_AO_VISIBILITY = 0x10000000u;
const uint TONE_FLAG_SKIP_DIRECT_SHADOWS = 0x08000000u;
const uint TONE_FLAG_RECT_SHARED_SHADOWS = 0x04000000u;

/* Buffer references for vertex/index access */
layout(buffer_reference, scalar) buffer Vertices { float v[]; };
layout(buffer_reference, scalar) buffer Indices  { uint  i[]; };

vec3 ptexTriangleColor(uint meshIdx, vec3 bary)
{
    uint off = scene.meshes[meshIdx].ptex_color_offset;
    if (off == 0xFFFFFFFFu)
        return vec3(-1.0);
    uint base = off + gl_PrimitiveID * 3u;
    vec3 c0 = unpackUnorm4x8(ptexTriColors.colors[base + 0u]).rgb;
    vec3 c1 = unpackUnorm4x8(ptexTriColors.colors[base + 1u]).rgb;
    vec3 c2 = unpackUnorm4x8(ptexTriColors.colors[base + 2u]).rgb;
    return c0 * bary.x + c1 * bary.y + c2 * bary.z;
}

/* ---- PBR Functions ---- */

const float PI = 3.14159265359;

/* OVRTX viewer fallback: no-light stages get a camera-attached SphereLight
 * plus a weak DomeLight authored by nanousdview. The renderer-native fallback
 * uses the same camera-relative key shape instead of requiring a USD wrapper. */
const vec3 OVRTX_FALLBACK_KEY_DIR = normalize(vec3(-0.29883624, 0.70710678, 0.64085638));
const vec3 OVRTX_FALLBACK_KEY_COLOR = vec3(1.0, 0.95, 0.9);
const float OVRTX_FALLBACK_KEY_INTENSITY = 3500.0 * 0.0006;

vec3 sceneUpVector() {
    if (scene.upAxis == 0u) return vec3(1.0, 0.0, 0.0);
    if (scene.upAxis == 2u) return vec3(0.0, 0.0, 1.0);
    return vec3(0.0, 1.0, 0.0);
}

float sceneUpCoord(vec3 v) {
    if (scene.upAxis == 0u) return v.x;
    if (scene.upAxis == 2u) return v.z;
    return v.y;
}

float sceneHeightCoord(vec3 p) {
    return sceneUpCoord(p);
}

/* Narkowicz ACES filmic tonemap (diagnostic clay-viz). */
vec3 nu_aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
/* Clay-viz override (tone_flags bit 30): warm key sun + sky/ground hemisphere
 * ambient + ACES for material diagnostics. Driven by NUSD_CLAY_VIZ=1 via
 * gpu_cmd_trace_rays. */
vec3 nu_clay_shade(vec3 N, vec3 rayDir) {
    vec3 Nf     = (dot(N, rayDir) > 0.0) ? -N : N;
    vec3 clay   = vec3(0.58, 0.40, 0.28);   /* saturated terracotta, not warm gray */
    vec3 sunDir = normalize(vec3(0.35, 0.80, 0.45));
    float ndl   = max(dot(Nf, sunDir), 0.0);
    vec3 sunCol = vec3(1.0, 0.92, 0.78) * 1.6;  /* calmer key so lit faces stay tan */
    float upf   = clamp(0.5 * (Nf.y + 1.0), 0.0, 1.0);
    vec3 amb    = mix(vec3(0.32, 0.28, 0.24), vec3(0.40, 0.50, 0.70), upf);
    vec3 lit    = nu_aces(clay * (sunCol * ndl + amb));
    float luma  = dot(lit, vec3(0.2126, 0.7152, 0.0722));
    return clamp(mix(vec3(luma), lit, 1.2), 0.0, 1.0);  /* +20% saturation */
}

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
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

/* Pixar orthonormal basis (Duff et al.) — robust on axis-aligned normals.
 * Fallback for degenerate UVs in genTangSpace. */
void branchlessONB(vec3 n, out vec3 b1, out vec3 b2) {
    float sgn = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sgn + n.z);
    float b = n.x * n.y * a;
    b1 = vec3(1.0 + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
    b2 = vec3(b, sgn + n.y * n.y * a, -n.y);
}

/* Tangent basis from triangle position + UV partial derivatives. This is the
 * standard solution of dP = T*dU + B*dV, with a stable orthonormal fallback
 * for degenerate UVs. */
void genTangSpace(vec2 uv0, vec2 uv1, vec2 uv2,
                  vec3 p0,  vec3 p1,  vec3 p2,
                  vec3 N,   out vec3 T, out vec3 B) {
    vec2 duv02 = uv0 - uv2, duv12 = uv1 - uv2;
    vec3 dp02  = p0 - p2,   dp12  = p1 - p2;
    float det = duv02.x * duv12.y - duv02.y * duv12.x;
    bool degenerate = abs(det) < 1e-8;
    T = vec3(1.0, 0.0, 0.0);
    B = vec3(0.0, 1.0, 0.0);
    if (!degenerate) {
        float inv = 1.0 / det;
        T = ( duv12.y * dp02 - duv02.y * dp12) * inv;
        B = (-duv12.x * dp02 + duv02.x * dp12) * inv;
    }
    if (degenerate || dot(cross(T, B), cross(T, B)) == 0.0) {
        branchlessONB(normalize(cross(dp02, p0 - p1)), T, B);
    }
    /* Gram-Schmidt orthonormalize T against N, then B = N × T with handedness. */
    T = normalize(T - N * dot(N, T));
    vec3 nu = cross(N, T);
    float sgn = dot(nu, B) < 0.0 ? -1.0 : 1.0;
    B = normalize(sgn * nu);
}

/* Split-sum specular IBL term. The LUT is generated in gpu_vulkan.c with a
 * GGX importance-sampled integration (Karis-style scale+bias). */
vec3 sampleEnvBRDF(vec3 F0, float roughness, float NoV) {
    vec2 brdf = texture(brdfLUT, vec2(clamp(abs(NoV), 0.0, 1.0),
                                      clamp(roughness, 0.0, 1.0))).rg;
    return F0 * brdf.x + brdf.y;
}

/* Convert a 3D direction to equirectangular UV coordinates. Match Hydra /
 * OVRTX DomeLight's lat-long convention for these assets: +Z is the north
 * pole and +Y is the zero-longitude forward axis. The longitude offset keeps
 * Vulkan's visible dome and IBL aligned with OVRTX. */
vec2 dirToEquirect(vec3 dir) {
    const float kDomeLongitudeOffset = 340.0 / 360.0;
    float u;
    float v;
    if (scene.upAxis == 1u) {
        u = fract(atan(dir.x, dir.z) * (0.5 / PI) + 0.5 + kDomeLongitudeOffset);
        v = asin(clamp(dir.y, -1.0, 1.0)) * (1.0 / PI) + 0.5;
    } else if (scene.upAxis == 0u) {
        u = fract(atan(dir.y, dir.z) * (0.5 / PI) + 0.5 + kDomeLongitudeOffset);
        v = asin(clamp(dir.x, -1.0, 1.0)) * (1.0 / PI) + 0.5;
    } else {
        u = fract(atan(dir.x, dir.y) * (0.5 / PI) + 0.5 + kDomeLongitudeOffset);
        v = asin(clamp(dir.z, -1.0, 1.0)) * (1.0 / PI) + 0.5;
    }
    return vec2(u, 1.0 - v);
}

/* Ray shaders do not have fragment derivatives, so implicit texture()
 * sampling can fall back to base mip and make high-res MaterialX normal and
 * roughness maps shimmer/over-darken in RT. Estimate the pixel footprint from
 * the ray cone and triangle UV density, then sample material maps explicitly
 * with textureLod. */
float estimateMaterialLod(int texIndex,
                          vec2 uv0, vec2 uv1, vec2 uv2,
                          vec3 p0,  vec3 p1,  vec3 p2) {
    if (texIndex < 0) return 0.0;
    ivec2 ts = textureSize(textures[nonuniformEXT(texIndex)], 0);
    float texDim = float(max(max(ts.x, ts.y), 1));

    vec3 w0 = (gl_ObjectToWorldEXT * vec4(p0, 1.0)).xyz;
    vec3 w1 = (gl_ObjectToWorldEXT * vec4(p1, 1.0)).xyz;
    vec3 w2 = (gl_ObjectToWorldEXT * vec4(p2, 1.0)).xyz;
    float e01 = max(length(w1 - w0), 1e-7);
    float e12 = max(length(w2 - w1), 1e-7);
    float e20 = max(length(w0 - w2), 1e-7);
    float uvPerWorld = max(max(length(uv1 - uv0) / e01,
                               length(uv2 - uv1) / e12),
                           length(uv0 - uv2) / e20);
    if (uvPerWorld <= 0.0) return 0.0;

    float tanHalfY = abs(projInverse[1][1]);
    float launchH = max(float(gl_LaunchSizeEXT.y), 1.0);
    float pixelWorld = max(gl_HitTEXT * (2.0 * tanHalfY / launchH),
                           scene_scale * 1e-5);
    float rho = max(pixelWorld * uvPerWorld * texDim * 0.5, 1.0);
    return clamp(log2(rho), 0.0, 12.0);
}

vec4 sampleMaterialTexture(int texIndex,
                           vec2 uv,
                           vec2 uv0, vec2 uv1, vec2 uv2,
                           vec3 p0,  vec3 p1,  vec3 p2) {
    float lod = estimateMaterialLod(texIndex, uv0, uv1, uv2, p0, p1, p2);
    return textureLod(textures[nonuniformEXT(texIndex)], uv, lod);
}

/* Shadow test via ray query — returns 0.0 if occluded, 1.0 if lit.
 * In fast_mode, always returns 1.0 (no shadow rays for RL sensors). */
float traceShadow(vec3 origin, vec3 direction, float tmax) {
    if (fast_mode != 0 ||
        (tone_flags & TONE_FLAG_SKIP_SECONDARY_VISIBILITY) != 0u ||
        (tone_flags & TONE_FLAG_SKIP_DIRECT_SHADOWS) != 0u) {
        return 1.0;
    }
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFF, origin, 0.001, direction, tmax);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT)
        return 0.0;
    return 1.0;
}

float contactVisibility(vec3 worldPos, vec3 N) {
    if (fast_mode != 0 ||
        (tone_flags & TONE_FLAG_SKIP_SECONDARY_VISIBILITY) != 0u ||
        (tone_flags & TONE_FLAG_SKIP_AO_VISIBILITY) != 0u) {
        return 1.0;
    }
    vec3 helper = (abs(N.y) < 0.95) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(helper, N));
    vec3 B = cross(N, T);
    float aoLen = clamp(scene_scale * 0.045, 0.018, 0.090);
    vec3 origin = worldPos + N * max(scene_scale * 0.0008, 0.0005);
    float v = 0.0;
    v += traceShadow(origin, N, aoLen * 0.55);
    v += traceShadow(origin, normalize(N * 0.62 + T * 0.60), aoLen);
    v += traceShadow(origin, normalize(N * 0.62 - T * 0.60), aoLen);
    v += traceShadow(origin, normalize(N * 0.62 + B * 0.60), aoLen);
    v += traceShadow(origin, normalize(N * 0.62 - B * 0.60), aoLen);
    return v * 0.2;
}

/* Wrap-lighting subsurface approximation. Shifts the cosine term so light
 * "wraps" around the silhouette, then mixes a body-color tint into the
 * diffuse lobe. Cheap stand-in for a path-traced random walk; matches the
 * visual MaterialXView shows for jade and the chess King/Queen.
 *
 *   sssWeight     — fraction of the diffuse lobe replaced by SSS (0..1)
 *   sssColor      — body color (subsurface_color from MaterialX)
 *   wrapAmount    — clamp(max(radius)*scale*K, 0, 0.5); 0 disables wrap
 *
 * With sssWeight == 0 the result is identical (bit-wise) to a plain
 * Lambert NdotL term, so chrome/copper/gold materials are unchanged. */
float computeWrappedNdotL(vec3 N, vec3 L, float wrapAmount) {
    return clamp((dot(N, L) + wrapAmount) / (1.0 + wrapAmount), 0.0, 1.0);
}

/* Evaluate one directional light with Cook-Torrance BRDF + shadow.
 * SSS contribution is added when sssWeight > 0: the diffuse lobe is
 * mixed with a wrapped + body-tinted term, the specular lobe uses the
 * unwrapped NdotL so highlights stay sharp. */
vec3 evalDirectionalLight(vec3 lightDir, vec3 lightColor, float lightIntensity,
                          vec3 N, vec3 V, vec3 baseColor, float metallic,
                          float roughness, vec3 F0, vec3 worldPos,
                          float sssWeight, vec3 sssColor, float wrapAmount) {
    vec3 L = lightDir;
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    /* When SSS is active, wrapped term may be > 0 even when NdotL == 0
     * (back of object catches a little body-color glow). Keep the gate
     * but use the wrapped value so jade/king receive light past 90°. */
    float wrappedNdotL = (sssWeight > 0.0)
        ? computeWrappedNdotL(N, L, wrapAmount)
        : NdotL;
    if (wrappedNdotL <= 0.0) return vec3(0.0);

    /* Shadow */
    float shadow = traceShadow(worldPos + N * 0.01, L, 100000.0);
    if (shadow <= 0.0) return vec3(0.0);

    /* Cook-Torrance BRDF — uses sharp NdotL so specular highlight stays
     * tight. If NdotL == 0 we still need to skip specular. */
    vec3 specular = vec3(0.0);
    if (NdotL > 0.0) {
        float D = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        vec3  F = fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, roughness);
        float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
        specular = (D * G * F) / denominator;
    }

    /* Diffuse with optional SSS body-tint mix. sssColor is already the
     * loader-resolved body tint (see main() — when the loader couldn't
     * extract a constant subsurface_color, sssColor equals baseColor
     * so the mix below collapses to no-op for the default case). */
    vec3 kD = (vec3(1.0) - fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, roughness)) * (1.0 - metallic);
    vec3 diffuseAlbedo = baseColor;
    if (sssWeight > 0.0) {
        diffuseAlbedo = mix(baseColor, sssColor, sssWeight);
    }
    vec3 diffuse = kD * diffuseAlbedo / PI * wrappedNdotL;

    /* Specular uses sharp NdotL; diffuse uses wrappedNdotL above. */
    return (diffuse + specular * NdotL) * lightColor * lightIntensity * shadow;
}

float areaLightSpecularRoughness(float roughness, float angularDiameter) {
    float a = clamp(angularDiameter * 2.5, 0.0, 1.0);
    return clamp(sqrt(roughness * roughness + a * a), roughness, 1.0);
}

vec3 evalRectLightSample(GpuLight light, vec3 samplePos, float radiance,
                         float area, vec3 N, vec3 V, vec3 baseColor,
                         float metallic, float roughness, vec3 F0,
                         vec3 worldPos, float sssWeight, vec3 sssColor,
                         float wrapAmount, float sharedShadow) {
    vec3 toLight = samplePos - worldPos;
    float dist2 = dot(toLight, toLight);
    if (dist2 < 1e-4) return vec3(0.0);
    float dist = sqrt(dist2);
    vec3 L = toLight / dist;

    float NdotL = dot(N, L);
    float wrappedNdotL = (sssWeight > 0.0)
        ? computeWrappedNdotL(N, L, wrapAmount)
        : max(NdotL, 0.0);
    if (wrappedNdotL <= 0.0) return vec3(0.0);

    float cosLight = dot(-L, light.normal);
    if (cosLight <= 0.0) return vec3(0.0);

    float geom = (cosLight * area) / max(dist2, 1e-4);
    float shadow = (sharedShadow >= 0.0)
        ? sharedShadow
        : traceShadow(worldPos + N * 0.01, L, dist * 0.999);
    if (shadow <= 0.0) return vec3(0.0);

    vec3 H = normalize(V + L);
    vec3 specular = vec3(0.0);
    float sharpNdotL = max(NdotL, 0.0);
    /* The rect is sampled as a few point lights; widen the specular lobe by
     * emitter angular diameter so rough floors do not get point-light glints. */
    float rectAngular = clamp(2.0 * max(length(light.u_axis), length(light.v_axis)) /
                              max(dist, 1e-3), 0.0, 1.0);
    float specRoughness = areaLightSpecularRoughness(roughness, rectAngular);
    if (sharpNdotL > 0.0) {
        float D = distributionGGX(N, H, specRoughness);
        float G = geometrySmith(N, V, L, specRoughness);
        vec3  F = fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, roughness);
        specular = (D * G * F) /
                   (4.0 * max(dot(N, V), 0.0) * sharpNdotL + 0.0001);
    }
    vec3 kD = (vec3(1.0) - fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, roughness)) *
              (1.0 - metallic);

    vec3 diffuseAlbedo = baseColor;
    if (sssWeight > 0.0)
        diffuseAlbedo = mix(baseColor, sssColor, sssWeight);
    vec3 diffuse = kD * diffuseAlbedo / PI * wrappedNdotL;

    return (diffuse + specular * sharpNdotL) *
           (light.color * radiance) * geom * shadow;
}

/* Evaluate a USD UsdLuxRectLight by sampling deterministic 3x3
 * Gauss-Legendre quadrature over the area. The weights sum to one because
 * evalRectLightSample already multiplies by the full light area.
 *   - light.position: rect center
 *   - light.normal:   emit direction (light shines along +normal towards scene)
 *   - light.u_axis, light.v_axis: half-extent vectors (so width=2|u|, height=2|v|)
 *   - normalize=1: intensity is power (W) → radiance L = intensity / (area * π)
 *   - normalize=0: intensity is radiance directly */
vec3 evalRectLight(GpuLight light, vec3 N, vec3 V, vec3 baseColor, float metallic,
                   float roughness, vec3 F0, vec3 worldPos,
                   float sssWeight, vec3 sssColor, float wrapAmount) {
    float area = 4.0 * length(light.u_axis) * length(light.v_axis);
    /* Treat inputs:intensity as radiance-like scene units and convert through
     * the Lambertian area-light normalization used by common PBR renderers.
     * The final scale below is calibrated against public ovrtx output, not
     * against private renderer implementation details. */
    float radiance;
    if (light.normalize != 0) {
        /* Power → radiance: P / (A·π) */
        radiance = light.intensity / max(area * PI, 1e-6);
    } else {
        /* Divide by pi for Lambertian normalization: E = pi * L for an
         * isotropic emitter under typical tone-mapping budgets. */
        radiance = light.intensity / PI;
    }
    /* Global tuning factor to match ovrtx exposure (see comparison.png).
     * The Isaac warehouse authors lights with intensities up to 4e5; the
     * ovrtx renderer's auto-exposure reduces them by ~3-4 stops. Disney
     * Moana, however, uses modest exposure-authored RectLights plus a visible
     * dome fallback (negative envIntensity); using the Isaac compression there
     * effectively disables the warm key lights. */
    radiance *= (scene.envIntensity < -1.0) ? 0.92 : 0.0012;

    const float q = 0.7745966692;
    const float wc = 0.1975308642;
    const float we = 0.1234567901;
    const float wk = 0.0771604938;
    vec3 accum = vec3(0.0);
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
            float centerNdotL = dot(N, centerL);
            float centerWrappedNdotL = (sssWeight > 0.0)
                ? computeWrappedNdotL(N, centerL, wrapAmount)
                : max(centerNdotL, 0.0);
            float centerCosLight = dot(-centerL, light.normal);
            sharedShadow = (centerWrappedNdotL > 0.0 && centerCosLight > 0.0)
                ? traceShadow(worldPos + N * 0.01, centerL, centerDist * 0.999)
                : 0.0;
        } else {
            sharedShadow = 0.0;
        }
    }
    accum += evalRectLightSample(light, center, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sssWeight, sssColor, wrapAmount,
                                 sharedShadow) * wc;
    accum += evalRectLightSample(light, center + U * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sssWeight, sssColor, wrapAmount,
                                 sharedShadow) * we;
    accum += evalRectLightSample(light, center - U * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sssWeight, sssColor, wrapAmount,
                                 sharedShadow) * we;
    accum += evalRectLightSample(light, center + W * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sssWeight, sssColor, wrapAmount,
                                 sharedShadow) * we;
    accum += evalRectLightSample(light, center - W * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sssWeight, sssColor, wrapAmount,
                                 sharedShadow) * we;
    accum += evalRectLightSample(light, center + U * q + W * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sssWeight, sssColor, wrapAmount,
                                 sharedShadow) * wk;
    accum += evalRectLightSample(light, center + U * q - W * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sssWeight, sssColor, wrapAmount,
                                 sharedShadow) * wk;
    accum += evalRectLightSample(light, center - U * q + W * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sssWeight, sssColor, wrapAmount,
                                 sharedShadow) * wk;
    accum += evalRectLightSample(light, center - U * q - W * q, radiance, area,
                                 N, V, baseColor, metallic, roughness, F0,
                                 worldPos, sssWeight, sssColor, wrapAmount,
                                 sharedShadow) * wk;
    return accum;
}

/* USD UsdLuxDistantLight — sun-like directional. light.normal is emit direction;
 * L = -light.normal. Scale is calibrated against the grey-cube OVRTX sweep:
 * intensity=3000 is already near exposure-compressed white in OVRTX. */
vec3 evalDistantLight(GpuLight light, vec3 N, vec3 V, vec3 baseColor, float metallic,
                      float roughness, vec3 F0, vec3 worldPos,
                      float sssWeight, vec3 sssColor, float wrapAmount) {
    vec3 L = normalize(-light.normal);
    return evalDirectionalLight(L, light.color, light.intensity * 0.0025,
                                N, V, baseColor, metallic, roughness, F0, worldPos,
                                sssWeight, sssColor, wrapAmount);
}

/* USD UsdLuxSphereLight — point/area emitter at light.position with radius
 * stored in light.u_axis.x (see scene.h SCENE_LIGHT_SPHERE). Treated as an
 * isotropic point with 1/r² falloff; when normalize=1 the intensity is
 * interpreted as power and divided by sphere surface area to give radiant
 * exitance.
 *
 * NOTE: the final `radiance *= 0.00017` below is an EMPIRICAL no-HDR-probe
 * brightness scale (see the inline comment), NOT the DistantLight 0.0006
 * cd→radiance factor — the two are different and unrelated (this comment
 * previously mis-stated 0.0006). The 0.00017 is a known fudge: not derived
 * from a physical/ovrtx-matched normalization, validated only by eye. A
 * principled fix must (1) match the ovrtx SphereLight radiance formula and
 * (2) be re-validated against the IsaacLab visual correctness gate (this path
 * can reach RL tensors under NU_DEFERRED_SHADE). Do not change the value
 * without that visual-gate validation. */
vec3 evalSphereLight(GpuLight light, vec3 N, vec3 V, vec3 baseColor, float metallic,
                     float roughness, vec3 F0, vec3 worldPos,
                     float sssWeight, vec3 sssColor, float wrapAmount) {
    vec3 toLight = light.position - worldPos;
    float dist2 = max(dot(toLight, toLight), 1e-4);
    float dist = sqrt(dist2);
    vec3 L = toLight / dist;

    float radius = max(light.u_axis.x, 1e-4);
    float area = 4.0 * PI * radius * radius;
    float radiance;
    if (light.normalize != 0) {
        radiance = light.intensity / area;
    } else {
        radiance = light.intensity;
    }
    /* Small no-HDR probe SphereLights need a lower radiance scale than
     * DistantLights; otherwise high-albedo USDZ assets blow out long before
     * OVRTX's auto-exposure settles. */
    radiance *= 0.00017;

    /* Shadow ray to sphere center, shortened by radius so we don't self-hit
     * the light geometry if the user models it as a real mesh. */
    float shadow = traceShadow(worldPos + N * 0.01, L, max(dist - radius, 0.01));
    if (shadow <= 0.0) return vec3(0.0);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float wrappedNdotL = (sssWeight > 0.0)
        ? computeWrappedNdotL(N, L, wrapAmount)
        : NdotL;
    if (wrappedNdotL <= 0.0 || NdotV <= 0.0) return vec3(0.0);

    vec3 H = normalize(V + L);
    vec3 specular = vec3(0.0);
    float sphereAngular = clamp(2.0 * radius / max(dist, 1e-3), 0.0, 1.0);
    float specRoughness = areaLightSpecularRoughness(roughness, sphereAngular);
    if (NdotL > 0.0) {
        float NDF = distributionGGX(N, H, specRoughness);
        float G   = geometrySmith(N, V, L, specRoughness);
        vec3  F   = fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, specRoughness);
        specular = (NDF * G * F) / (4.0 * NdotV * NdotL + 0.0001);
    }
    vec3 kD = (vec3(1.0) - fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, roughness)) * (1.0 - metallic);

    vec3 diffuseAlbedo = baseColor;
    if (sssWeight > 0.0) {
        /* sssColor is already the loader-resolved body tint (see main()
         * flag-aware assignment) — no inline (1,1,1) heuristic needed. */
        diffuseAlbedo = mix(baseColor, sssColor, sssWeight);
    }
    vec3 diffuse = kD * diffuseAlbedo / PI * wrappedNdotL;

    /* 1/dist² falloff bakes the inverse-square law for point sources. */
    return (diffuse + specular * NdotL) * light.color *
           radiance * shadow / dist2;
}

/* ACES filmic tone mapping (fitted curve by Krzysztof Narkowicz) */
vec3 acesFilmic(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main()
{
    /* Write depth for primary rays (hitValue.w < 0.5).
     * Secondary glass-refraction rays (hitValue.w = 1.0) skip to preserve
     * the first-surface depth written by the primary hit. */
    if (hitValue.w < 0.5) {
        uint pixelIdx = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
        if (depth_enabled != 0u)
            depthOut.depths[pixelIdx] = gl_HitTEXT;
        if (segmentation_enabled != 0u)
            segOut.ids[pixelIdx] = gl_InstanceCustomIndexEXT + 1u;
    }

    uint meshIdx = gl_InstanceCustomIndexEXT;
    Vertices verts = Vertices(scene.meshes[meshIdx].vertexAddress);
    Indices  idxs  = Indices(scene.meshes[meshIdx].indexAddress);
    uint stride = scene.vertexStride;

    /* gl_PrimitiveID is local to this mesh's BLAS */
    uint base = gl_PrimitiveID * 3;
    uint i0 = idxs.i[base + 0];
    uint i1 = idxs.i[base + 1];
    uint i2 = idxs.i[base + 2];

    /* Barycentric weights */
    vec3 bary = vec3(1.0 - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);

    /* Read positions: offset +0 */
    vec3 p0 = vec3(verts.v[i0*stride+0], verts.v[i0*stride+1], verts.v[i0*stride+2]);
    vec3 p1 = vec3(verts.v[i1*stride+0], verts.v[i1*stride+1], verts.v[i1*stride+2]);
    vec3 p2 = vec3(verts.v[i2*stride+0], verts.v[i2*stride+1], verts.v[i2*stride+2]);

    /* ---- Fast mode: minimal reads, face normal, simple diffuse ---- */
    if (fast_mode != 0) {
        vec3 faceN = normalize(cross(p1 - p0, p2 - p0));
        vec3 N = normalize(gl_ObjectToWorldEXT * vec4(faceN, 0.0));
        if (normals_enabled != 0u && hitValue.w < 0.5) {
            uint nIdx = (gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x) * 3u;
            normOut.normals[nIdx + 0u] = N.x;
            normOut.normals[nIdx + 1u] = N.y;
            normOut.normals[nIdx + 2u] = N.z;
        }

        /* Phase B: write G-buffer for deferred-shading compute pass. Gated by
         * deferred_shade_enabled so the legacy fast-mode path stays bytewise
         * identical when the flag is off (binding 17 is bound to a stub
         * scene_data_buf in that case — never written to). Only primary rays
         * (hitValue.w < 0.5) populate the G-buffer; secondary rays don't
         * exist in fast_mode anyway.
         *
         * The G-buffer normal is FLIPPED toward the incoming ray to match
         * the non-fast normal-flip at line 604. This is a LOCAL flip — the
         * legacy normals AOV write above and the Lambertian/ambient
         * computations below continue to use the un-flipped N, preserving
         * byte-identity of the existing fast-mode color and AOV output. */
        if (deferred_shade_enabled != 0u && hitValue.w < 0.5) {
            vec3 N_for_gbuf = (dot(N, gl_WorldRayDirectionEXT) > 0.0) ? -N : N;
            uint gbufIdx = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
            uint inst    = gl_InstanceCustomIndexEXT & 0x00FFFFFFu;
            gbuf.entries[gbufIdx].inst_and_flags = (1u << 31) | inst;
            gbuf.entries[gbufIdx].primitive_id   = gl_PrimitiveID;
            gbuf.entries[gbufIdx].bary_packed    = packUnorm2x16(baryCoord);
            gbuf.entries[gbufIdx].hit_t          = gl_HitTEXT;
            gbuf.entries[gbufIdx].normal_x       = N_for_gbuf.x;
            gbuf.entries[gbufIdx].normal_y       = N_for_gbuf.y;
            gbuf.entries[gbufIdx].normal_z       = N_for_gbuf.z;
            gbuf.entries[gbufIdx]._pad           = 0u;
        }

        vec3 baseColor = scene.meshes[meshIdx].color.rgb;
        /* Phase B: textured albedo when the mesh has a runtime texture
         * bound via nu_set_mesh_texture. Sentinel 0xFFFFFFFFu means
         * "no texture" — the early-out branch is taken and baseColor
         * stays at the flat per-mesh color (byte-identical for un-
         * textured meshes such as dexsuite-Lift's procedural primitives).
         * Stride >= 12 → UVs at offsets +9, +10 (mirrors the fat-path
         * read at line 612 onward). */
        uint tex_idx = scene.meshes[meshIdx].tex_index;
        if (scene.hasMaterials != 0u && tex_idx != 0xFFFFFFFFu && stride >= 12u) {
            vec2 uv0 = vec2(verts.v[i0*stride+9], verts.v[i0*stride+10]);
            vec2 uv1 = vec2(verts.v[i1*stride+9], verts.v[i1*stride+10]);
            vec2 uv2 = vec2(verts.v[i2*stride+9], verts.v[i2*stride+10]);
            vec2 uv  = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
            baseColor *= sampleMaterialTexture(int(tex_idx), uv, uv0, uv1, uv2, p0, p1, p2).rgb;
        }
        vec3 ptexColor = ptexTriangleColor(meshIdx, bary);
        if (ptexColor.x >= 0.0)
            baseColor = ptexColor;
        /* Z-up key light direction. */
        vec3 L = normalize(sceneUpVector() + vec3(0.3, 0.0, 0.5));
        float NdotL = max(dot(N, L), 0.0);
        /* Hemispheric ambient driven by the scene's flat DomeLight color so
         * surface shading tracks the sky written by the rmiss path (instead
         * of the old hardcoded blue/tan tints, which left a ~15/255 surface
         * delta on top of the dominant background term). The "ground tint"
         * is the dome modulated to ~50% — a cheap stand-in for the bottom
         * hemisphere that keeps the bright-sky / shaded-floor split.
         *
         * domeColor.rgb is pre-clamped on upload; intensity (.a) is folded
         * in so the rchit ambient scales with the same knob the rmiss sky
         * does — this preserves the contrast between geometry and sky. */
        vec3 dome = scene.domeColor.rgb * scene.domeColor.a;
        float h = 0.5 * (sceneUpCoord(N) + 1.0);
        vec3 ambient = mix(dome * 0.5, dome, h) * 0.5;
        if ((tone_flags & TONE_FLAG_CLAY_VIZ) != 0u) {
            hitValue = vec4(nu_clay_shade(N, gl_WorldRayDirectionEXT), 0.0);
            return;
        }
        hitValue = vec4(baseColor * (NdotL * 0.8) + baseColor * ambient, 0.0);
        return;
    }

    /* Read normals: offset +3 */
    vec3 n0 = vec3(verts.v[i0*stride+3], verts.v[i0*stride+4], verts.v[i0*stride+5]);
    vec3 n1 = vec3(verts.v[i1*stride+3], verts.v[i1*stride+4], verts.v[i1*stride+5]);
    vec3 n2 = vec3(verts.v[i2*stride+3], verts.v[i2*stride+4], verts.v[i2*stride+5]);

    /* Per-mesh color from SceneData SSBO (replaces per-vertex color) */
    vec3 meshColor = scene.meshes[meshIdx].color.rgb;

    /* Read UVs: offset +9 (when stride >= 12) */
    vec2 uv0 = vec2(0.0), uv1 = vec2(0.0), uv2 = vec2(0.0);
    vec2 uv  = vec2(0.0);
    bool hasUV = (stride >= 12);
    if (hasUV) {
        uv0 = vec2(verts.v[i0*stride+9], verts.v[i0*stride+10]);
        uv1 = vec2(verts.v[i1*stride+9], verts.v[i1*stride+10]);
        uv2 = vec2(verts.v[i2*stride+9], verts.v[i2*stride+10]);
        uv  = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    }

    /* Interpolate */
    vec3 localPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    vec3 worldPos = (gl_ObjectToWorldEXT * vec4(localPos, 1.0)).xyz;
    vec3 localNormal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);
    vec3 normal = normalize(gl_ObjectToWorldEXT * vec4(localNormal, 0.0));
    /* Flip normal to face the incoming ray — handles back-facing triangles
     * (e.g. floor tiles authored with inverted winding in Isaac scenes).
     * Without this, alternate tiles render dark because NdotL < 0 for
     * overhead lights. */
    if (dot(normal, gl_WorldRayDirectionEXT) > 0.0) {
        normal = -normal;
        localNormal = -localNormal;
    }

    /* G-buffer write — primary rays only. Secondary glass-refraction rays
     * (hitValue.w == 1.0) skip so they don't clobber the first-surface
     * record. The host enables this flag only when binding 17 is a real
     * G-buffer; tiled non-deferred RT binds a scene_data_buf stub there. */
    if (deferred_shade_enabled != 0u && hitValue.w < 0.5) {
        uint gbufIdx = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
        uint inst    = gl_InstanceCustomIndexEXT & 0x00FFFFFFu;
        gbuf.entries[gbufIdx].inst_and_flags = (1u << 31) | inst;
        gbuf.entries[gbufIdx].primitive_id   = gl_PrimitiveID;
        gbuf.entries[gbufIdx].bary_packed    = packUnorm2x16(baryCoord);
        gbuf.entries[gbufIdx].hit_t          = gl_HitTEXT;
        gbuf.entries[gbufIdx].normal_x       = normal.x;
        gbuf.entries[gbufIdx].normal_y       = normal.y;
        gbuf.entries[gbufIdx].normal_z       = normal.z;
        gbuf.entries[gbufIdx]._pad           = 0u;
    }

    vec3 color  = meshColor;

    /* Read material ID from SceneData rather than prototype vertex data:
     * instanced geometry can share vertices while carrying a distinct
     * material binding. */
    uint materialId = 0xFFFFFFFFu;
    if (scene.hasMaterials != 0) {
        materialId = scene.meshes[meshIdx].material_id;
    }

    /* PBR shading */
    vec3 baseColor = color;
    float metallic = 0.0;
    float roughness = 0.5;
    vec3 emissive = vec3(0.0);
    float ao = 1.0;
    float alpha = 1.0;
    float materialIor = 1.5;
    float clearcoat = 0.0;
    float clearcoatRoughness = 0.0;
    /* Subsurface scattering state (Phase 7c). Defaults disable SSS so
     * non-SSS materials (chrome/copper/gold/etc.) render bit-identical
     * to before this lobe was wired up — the eval functions branch on
     * sssWeight > 0 and skip every SSS code path otherwise. */
    float sssWeight   = 0.0;
    vec3  sssColor    = vec3(1.0);
    float sssWrap     = 0.0;
    /* Transmission lobe state — populated below from MaterialX Standard
     * Surface `transmission`, `transmission_color`, `specular_IOR` inputs.
     * Defaults disable transmission so non-glass materials render
     * bit-identical to before this lobe was wired. */
    float transWeight = 0.0;
    vec3  transTint   = vec3(1.0);
    float transIor    = 1.5;
    /* UsdPreviewSurface specular workflow — captured from material inside
     * the `if (scene.hasMaterials)` block; used at F0 derivation below. */
    int  useSpecularWorkflow = 0;
    vec3 specularColor       = vec3(0.0);
    float opacityThreshold   = 0.0;  /* UPS alpha cutout; 0 = disabled */

    if (scene.hasMaterials != 0 && materialId != 0xFFFFFFFFu) {
        MaterialData mat = mat_buf.materials[materialId];
        if (mat.use_vertex_color == 0) {
            baseColor = mat.base_color.rgb;
        }
        metallic = mat.metallic;
        roughness = mat.roughness;
        ao = mat.occlusion;
        alpha = mat.base_color.a * mat.opacity;
        materialIor = mat.ior;
        clearcoat = mat.clearcoat;
        clearcoatRoughness = clamp(mat.clearcoat_roughness, 0.04, 1.0);
        useSpecularWorkflow = mat.use_specular_workflow;
        specularColor = mat.specular_color.rgb;
        opacityThreshold = mat.opacity_threshold;

        /* Emissive: color * intensity, with log-scale for extreme USD intensity values */
        vec3 emissiveColor = mat.emissive_color.rgb;
        float emissiveIntensity = mat.emissive_color.a;
        if (emissiveIntensity > 100.0)
            emissiveIntensity = 1.0 + log2(emissiveIntensity / 100.0);
        emissive = emissiveColor * emissiveIntensity;

        /* UDIM UV scaling */
        vec2 tc = uv / vec2(mat.udim_scale_u, mat.udim_scale_v);
        tc = tc * mat.mdl_uv_transform.xy + mat.mdl_uv_transform.zw;
        if (mat.v_flip != 0) tc.y = 1.0 - tc.y;

        /* Sample diffuse texture if available; base_color acts as tint when
         * present (defaults to white in the loader for textured materials). */
        int diffuse_idx = mat.tex_indices[0]; /* TEX_DIFFUSE_COLOR = 0 */
        if (diffuse_idx >= 0) {
            vec4 tex_color = sampleMaterialTexture(diffuse_idx, tc, uv0, uv1, uv2, p0, p1, p2);
            baseColor = tex_color.rgb * mat.base_color.rgb;
            alpha *= tex_color.a;
        }

        /* Sample opacity texture if separately bound (UsdPreviewSurface
         * scenes that connect inputs:opacity to a UsdUVTexture distinct
         * from the diffuse map — common for foliage with separate albedo
         * + alpha sheets). The alpha-cutout path (opacityThreshold > 0)
         * needs an opacity source to be meaningful. */
        int opacity_idx = mat.tex_indices[6]; /* TEX_OPACITY = 6 */
        if (opacity_idx >= 0) {
            vec4 opa_color = sampleMaterialTexture(opacity_idx, tc, uv0, uv1, uv2, p0, p1, p2);
            /* UsdPreviewSurface convention: outputs:r drives opacity, but
             * RGBA → alpha is also common (and is what gltf+png produce).
             * Use .r if it's not a constant 1; fall back to .a otherwise. */
            float opa_sample = (opa_color.r < 1.0) ? opa_color.r : opa_color.a;
            alpha *= opa_sample;
        }

        /* Sample normal map if available.
         *
         * UsdUVTexture inputs:scale and inputs:bias remap each sampled
         * channel: out = sample * scale.rgb + bias.rgb. The defaults of
         * (2,2,2,1) / (-1,-1,-1,0) reproduce the historical implicit
         * `nm = nm * 2 - 1` mapping; scenes authoring different scale/bias
         * (e.g. raw [-1,1]-encoded normal maps with scale=1/bias=0) now
         * render spec-correctly. */
        int normal_idx = mat.tex_indices[1]; /* TEX_NORMAL = 1 */
        if (normal_idx >= 0) {
            vec3 nm = sampleMaterialTexture(normal_idx, tc, uv0, uv1, uv2, p0, p1, p2).rgb;
            nm = nm * mat.normal_tex_scale.rgb + mat.normal_tex_bias.rgb;
            vec3 localT, localB;
            if (hasUV) {
                genTangSpace(uv0, uv1, uv2, p0, p1, p2, localNormal, localT, localB);
            } else {
                branchlessONB(localNormal, localT, localB);
            }
            vec3 T = normalize(gl_ObjectToWorldEXT * vec4(localT, 0.0));
            vec3 B = normalize(gl_ObjectToWorldEXT * vec4(localB, 0.0));
            normal = normalize(T * nm.x * mat.normal_scale
                             + B * nm.y * mat.normal_scale
                             + normal * nm.z);
        }

        /* Sample roughness texture — use .g channel (ORM: G=Roughness) */
        int rough_idx = mat.tex_indices[2]; /* TEX_ROUGHNESS = 2 */
        if (rough_idx >= 0) {
            float roughSample = sampleMaterialTexture(rough_idx, tc, uv0, uv1, uv2, p0, p1, p2).g;
            float roughScale = (mat.roughness_tex_scale != 0.0) ? mat.roughness_tex_scale : 1.0;
            roughness = roughSample * roughScale + mat.roughness_tex_bias;
        }

        /* Sample metallic texture — use .b channel (ORM: B=Metallic) */
        int metal_idx = mat.tex_indices[3]; /* TEX_METALLIC = 3 */
        if (metal_idx >= 0) {
            metallic = sampleMaterialTexture(metal_idx, tc, uv0, uv1, uv2, p0, p1, p2).b;
        }

        /* Sample emissive texture if available */
        int emissive_idx = mat.tex_indices[4]; /* TEX_EMISSIVE_COLOR = 4 */
        if (emissive_idx >= 0) {
            vec3 emissive_tex = sampleMaterialTexture(emissive_idx, tc, uv0, uv1, uv2, p0, p1, p2).rgb;
            if (dot(emissiveColor, emissiveColor) < 0.001) {
                /* Emissive constant is black — texture IS the emissive color */
                emissive = emissive_tex * emissiveIntensity;
            } else {
                emissive *= emissive_tex;
            }
        }

        /* Sample occlusion texture — use .r channel (ORM: R=AO) */
        int ao_idx = mat.tex_indices[5]; /* TEX_OCCLUSION = 5 */
        if (ao_idx >= 0) {
            ao = sampleMaterialTexture(ao_idx, tc, uv0, uv1, uv2, p0, p1, p2).r;
        }

        /* === Subsurface (Phase 7c wrap-lighting approximation) ===========
         * MaterialX standard_surface defaults: subsurface_radius=(1,1,1),
         * subsurface_scale=1.0, so jade (which only authors weight + color)
         * gets a unit radius * 1 scale → wrap saturates at 0.5. Chess
         * King/Queen author scale=0.003 + radius=base_color, giving a
         * smaller, asset-appropriate wrap value.
         *
         * The 100.0 multiplier converts the (radius * scale) product into
         * a wrap fraction; tuned so the jade ball softly glows and the
         * king has a subtle marbled-glow rather than hard plastic. */
        sssWeight = mat.subsurface_weight;
        int sss_idx = mat.tex_subsurface_weight;
        if (sss_idx >= 0) {
            sssWeight *= sampleMaterialTexture(sss_idx, tc, uv0, uv1, uv2, p0, p1, p2).r;
        }
        /* If the loader extracted a constant subsurface_color, use it as
         * authored. Otherwise (chess King: nodegraph the side-loader can't
         * sample) fall back to baseColor as the body tint — matches
         * Standard Surface intent that scatter takes on the base color
         * when subsurface_color isn't explicitly set. The flag replaces
         * the previous fragile `sssColor < 0.999` heuristic which broke
         * for assets that intentionally author white SSS (milk, marble). */
        sssColor = (mat.sss_color_authored != 0)
                   ? mat.subsurface_color.rgb
                   : baseColor;
        vec3 sssRadius = mat.subsurface_radius.rgb * mat.subsurface_scale;
        float maxRadius = max(max(sssRadius.r, sssRadius.g), sssRadius.b);
        sssWrap = clamp(maxRadius * 100.0, 0.0, 0.5);

        /* === Transmission (Phase 7c continuation-ray glass lobe) =========
         * MaterialX standard_surface authors `transmission` as a 0..1
         * weight; Pawn tops + Glass.mtlx ship transmission=1. The
         * `transmission_color` is the Beer-Lambert-style tint we apply
         * to the through-ray result. `specular_IOR` (mapped to
         * params->transmission_ior in the loader) drives the Fresnel
         * mix between transmitted and reflected light. */
        transWeight = mat.transmission_weight;
        int trans_idx = mat.tex_transmission_weight;
        if (trans_idx >= 0) {
            transWeight *= sampleMaterialTexture(trans_idx, tc, uv0, uv1, uv2, p0, p1, p2).r;
        }
        transTint = mat.transmission_color.rgb;
        if (mat.transmission_ior > 0.0) transIor = mat.transmission_ior;
        else                            transIor = materialIor;
    }

    vec3 ptexColor = ptexTriangleColor(meshIdx, bary);
    if (ptexColor.x >= 0.0)
        baseColor = ptexColor;

    roughness = clamp(roughness, 0.04, 1.0);

    vec3 N = normal;
    if (normals_enabled != 0u && hitValue.w < 0.5) {
        uint nIdx = (gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x) * 3u;
        normOut.normals[nIdx + 0u] = N.x;
        normOut.normals[nIdx + 1u] = N.y;
        normOut.normals[nIdx + 2u] = N.z;
    }
    vec3 V = normalize(-gl_WorldRayDirectionEXT);
    float NdotV = max(dot(N, V), 0.001);
    bool fallbackLighting = (scene.envMipLevels > 0.0 &&
                             scene.envIntensity < -1.0);
    bool noLightNoIbl = (scene.envMipLevels == 0.0 &&
                         sceneLights.nlights == 0);
    bool displayColorFallback = ((scene.envMipLevels == 0.0 || fallbackLighting) &&
                                 sceneLights.nlights == 0 &&
                                 scene.hasMaterials == 0u);
    bool manySceneLightsNoIbl = (scene.envMipLevels == 0.0 &&
                                 sceneLights.nlights > 8);
    bool manySceneLightsWithIbl = (scene.envMipLevels > 0.0 &&
                                   sceneLights.nlights > 8);
    bool authoredKeyOnlyNoIbl = (scene.envMipLevels == 0.0 &&
                                 sceneLights.nlights > 0 &&
                                 sceneLights.nlights <= 8);
    if (displayColorFallback) {
        baseColor = clamp(baseColor * vec3(1.030, 1.012, 0.965), vec3(0.0), vec3(1.0));
    }
    float displayContact = 1.0;
    float displayFaceOcclusion = 1.0;
    float displayDarkUp = 0.0;
    float displayBrightUp = 0.0;
    float displaySkyBounce = 1.0;
    if (displayColorFallback) {
        float displayAlbedoLuma = dot(baseColor, vec3(0.2126, 0.7152, 0.0722));
        float upFace = smoothstep(0.30, 0.85, clamp(sceneUpCoord(N), 0.0, 1.0));
        float lowSurface = 1.0 - smoothstep(8.0, 60.0, sceneHeightCoord(worldPos));
        displayDarkUp = upFace * lowSurface;
        displayBrightUp = upFace * (1.0 - lowSurface) * smoothstep(0.50, 0.88, displayAlbedoLuma);
        displaySkyBounce = mix(0.08, 1.0, 1.0 - displayDarkUp);
        displayContact = contactVisibility(worldPos, N);
        float verticalFace = 1.0 - smoothstep(0.18, 0.95, abs(sceneUpCoord(N)));
        displayFaceOcclusion = clamp(1.0 - 0.06 * verticalFace, 0.92, 1.0);
    }

    /* ---- Glass / transparent material ----
     *
     * Two triggers feed this lobe:
     *   1. UsdPreviewSurface low-alpha materials (alpha < 0.5) — legacy
     *      path; tints transmitted light by baseColor.
     *   2. MaterialX Standard Surface `transmission` lobe — Pawn tops
     *      ship `transmission=1, transmission_color=(0.30, 0.50, 0.45)`,
     *      and Glass.mtlx ships the same with white tint. Tinted by
     *      `transmission_color` (Pawn tops have white baseColor and a
     *      colored tint, so baseColor would be wrong).
     *
     * Two flavours of transmission, distinguished by transmission_color:
     *
     *  (a) Clear glass — `transmission_color ≈ (1,1,1)`. We do Snell
     *      refraction with TIR fallback, allowing up to 2 bounces
     *      (front face → back face → behind), gated by the pipeline's
     *      maxPipelineRayRecursionDepth = 3 (gpu_vulkan.c). Standalone
     *      Glass.mtlx is the canonical case.
     *
     *  (b) Tinted translucent — `transmission_color != (1,1,1)`, used
     *      for wax / jade ball / Pawn-tops with
     *      transmission_color = (0.30, 0.50, 0.45). Real Snell here
     *      makes the surface look like a glass marble instead of soft
     *      translucent. We instead pass the ray straight through and
     *      tint by transmission_color — what most production renderers
     *      do for these materials when path-traced volume scattering
     *      isn't available. (A future Beer-Lambert / random-walk SSS
     *      lobe would supersede this.) */
    float rayDepth = hitValue.w;

    /* UsdPreviewSurface alpha cutout (opacityThreshold > 0).
     * Distinct from the alpha-blend / transmission path: this is a binary
     * test — the surface is either fully opaque (alpha ≥ threshold) or
     * fully invisible (alpha < threshold, recurse straight through).
     * Foliage / decals / cutouts use this. Without an any-hit shader we
     * piggyback on the existing rayDepth budget and trace a continuation
     * ray, identical in spirit to the tinted-transmission path below. */
    if (opacityThreshold > 0.0 && alpha < opacityThreshold && rayDepth < 1.5) {
        vec3 throughOrigin = worldPos + gl_WorldRayDirectionEXT * 0.001;
        hitValue = vec4(0.0, 0.0, 0.0, rayDepth + 1.0);
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0,
                    throughOrigin, 0.001, gl_WorldRayDirectionEXT, 100000.0, 0);
        return;
    }
    /* If cutout passed (alpha ≥ threshold), treat surface as fully
     * opaque — alpha-test should not also alpha-blend. */
    if (opacityThreshold > 0.0) {
        alpha = 1.0;
    }

    bool transmissionActive = (alpha < 0.5 || transWeight > 0.001) && rayDepth < 1.5;
    if (transmissionActive) {
        bool clearGlass = (transTint.r > 0.95 && transTint.g > 0.95 && transTint.b > 0.95);
        bool waterLike = (transWeight > 0.35 &&
                          transIor > 1.24 && transIor < 1.39 &&
                          transTint.g > transTint.r + 0.05 &&
                          transTint.g >= transTint.b &&
                          roughness <= 0.08);

        /* Determine entry vs exit by comparing the geometric normal to
         * the ray direction. Inside-going rays (dot < 0) are entering
         * the dielectric; outside-going (dot > 0) are exiting. The
         * normal we use for refract() must point AGAINST the ray. */
        vec3 rayDir = gl_WorldRayDirectionEXT;
        bool entering = dot(rayDir, N) < 0.0;
        vec3 normalForRefract = entering ? N : -N;
        float eta = entering ? (1.0 / transIor) : transIor;

        /* IOR-driven Fresnel — driven from the actual material IOR. */
        float f0_glass = pow((transIor - 1.0) / (transIor + 1.0), 2.0);
        float cosI = max(dot(V, N), 0.0);
        float F_glass = f0_glass + (1.0 - f0_glass) * pow(1.0 - cosI, 5.0);

        /* Through-direction: Snell for clear glass, straight for tinted. */
        vec3 throughDir;
        if (clearGlass) {
            vec3 refracted = refract(rayDir, normalForRefract, eta);
            bool tir = dot(refracted, refracted) < 1e-6;
            throughDir = tir ? reflect(rayDir, normalForRefract) : refracted;
        } else {
            throughDir = rayDir;
        }

        /* Trace continuation ray. Push origin slightly beyond the
         * surface in the direction of travel to avoid self-intersection.
         * Mark the payload's w-channel with rayDepth+1 so the next hit
         * can decide whether to recurse again. */
        vec3 throughOrigin = worldPos + throughDir * 0.001;
        hitValue = vec4(0.0, 0.0, 0.0, rayDepth + 1.0);
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0,
                    throughOrigin, 0.001, throughDir, 100000.0, 0);
        vec3 behindColor = hitValue.xyz;

        /* Tint: legacy alpha path uses baseColor; MaterialX path uses
         * transmission_color (Pawn tops, Glass.mtlx). When BOTH are set
         * — extreme corner case — multiply both. */
        vec3 transmitted = behindColor;
        if (transWeight > 0.001) transmitted *= transTint;
        if (alpha < 0.5)         transmitted *= baseColor;

        /* Specular reflection from environment */
        vec3 R = reflect(-V, N);
        vec3 envReflect;
        if (scene.envMipLevels > 0.0) {
            envReflect = textureLod(envMap, dirToEquirect(R), 0.0).rgb;
        } else {
            float rT = R.y * 0.5 + 0.5;
            if (rT < 0.5)
                envReflect = mix(vec3(0.20, 0.18, 0.15), vec3(0.72, 0.72, 0.70), rT * 2.0);
            else
                envReflect = mix(vec3(0.72, 0.72, 0.70), vec3(0.35, 0.50, 0.80), (rT - 0.5) * 2.0);
            envReflect *= 0.5;
        }

        vec3 glassResult;
        if (waterLike) {
            float tintLuma = dot(transTint, vec3(0.2126, 0.7152, 0.0722));
            vec3 waterTint = mix(vec3(tintLuma), transTint, 0.78);
            float grazing = pow(1.0 - cosI, 2.0);
            float upWater = clamp(sceneUpCoord(N), 0.0, 1.0);
            if (scene.envIntensity < -1.0) {
                float envLuma = dot(envReflect, vec3(0.2126, 0.7152, 0.0722));
                envReflect = mix(vec3(envLuma), envReflect, 0.34);
                envReflect = mix(envReflect, waterTint, 0.66);
                envReflect *= vec3(0.82, 1.02, 0.88);
            }
            vec3 softenedTransmission;
            float waterFresnel;
            if (scene.envIntensity < -1.0) {
                softenedTransmission =
                    transmitted * transTint * 0.18 +
                    waterTint * (0.24 + 0.30 * upWater);
                waterFresnel = clamp(F_glass * 1.10 + grazing * 0.09, 0.0, 0.42);
            } else {
                softenedTransmission =
                    transmitted * 0.64 + waterTint * (0.16 + 0.20 * upWater);
                waterFresnel = clamp(F_glass * 1.35 + grazing * 0.13, 0.0, 0.58);
            }
            glassResult = mix(softenedTransmission, envReflect, waterFresnel);
            glassResult += waterTint * (0.055 + 0.060 * grazing) * upWater;
        } else {
            glassResult = mix(transmitted, envReflect, F_glass);
        }

        /* No-IBL brightness floor.
         *
         * When `scene.envMipLevels == 0` (no HDR loaded) the recursive
         * through-ray often hits dark interior geometry (Microwave door
         * → dark microwave body, Blender jar → motor housing) and
         * `transmitted` returns near-zero. The procedural `envReflect`
         * is also dim (the procedural sky has `*= 0.5`), so without IBL
         * the glass face renders pure black — visually divergent from
         * raster and opengl which simply render glass as opaque grey.
         *
         * Add a small hemisphere-ambient floor tinted by `baseColor`
         * (matches the opaque-PBR no-IBL ambient at lines ~974-1014) so
         * a glass surface in an unlit scene looks like dim translucent
         * material rather than a pure-black hole. Gated on no-IBL so we
         * don't perturb chess set / Glass.mtlx (which have IBL and
         * already render correctly). */
        if (scene.envMipLevels == 0.0) {
            vec3 skyColor    = vec3(0.60, 0.68, 0.82);
            vec3 groundColor = vec3(0.32, 0.31, 0.30);
            float hemi = dot(N, sceneUpVector()) * 0.5 + 0.5;
            vec3 hemiAmbient = mix(groundColor, skyColor, hemi);
            glassResult += baseColor * hemiAmbient * 0.25 * (1.0 - F_glass);
        }

        /* Tone map glass result — sRGB swapchain handles gamma */
        float glassExposure = waterLike ? 0.58 : ((scene.envMipLevels > 0.0) ? 0.8 : 1.2);
        glassResult = acesFilmic(glassResult * glassExposure);

        hitValue = vec4(glassResult, 1.0);
        return;
    }

    /* ---- Opaque PBR lighting ---- */

    /* IOR-driven F0 (matches raster material shader).
     *
     * UsdPreviewSurface specular workflow: when useSpecularWorkflow=1,
     * F0 is artist-authored as `specularColor` (an RGB), and `metallic`
     * is undefined / typically 0. Override metallic to 0 so the
     * downstream diffuse-cancel terms (kD ∝ 1-metallic) reduce to the
     * pure dielectric case — F0 still drives the specular lobe via
     * specularColor. */
    float f0_dielectric = pow((materialIor - 1.0) / (materialIor + 1.0), 2.0);
    vec3 F0;
    if (useSpecularWorkflow != 0) {
        metallic = 0.0;
        F0 = specularColor;
    } else {
        F0 = mix(vec3(f0_dielectric), baseColor, metallic);
    }

    /* Ambient / IBL */
    vec3 ambient;
    if (scene.envMipLevels > 0.0) {
        /* IBL diffuse: irradiance map lookup (equirectangular).
         * For SSS materials, the diffuse-IBL albedo is mixed toward
         * subsurface_color, giving jade its glow under HDRI lighting
         * (where IBL dominates the ambient term). */
        vec3 irradiance = texture(irrMap, dirToEquirect(N)).rgb;
        if (fallbackLighting) {
            float irrLuma = dot(irradiance, vec3(0.2126, 0.7152, 0.0722));
            irradiance = mix(vec3(irrLuma), irradiance, 0.58) *
                         vec3(1.08, 1.00, 0.82);
        }
        vec3 kS_ibl = fresnelSchlickRoughness(NdotV, F0, roughness);
        vec3 kD_ibl = (1.0 - kS_ibl) * (1.0 - metallic);
        /* sssColor is already the loader-resolved body tint (see main()
         * sss_color_authored handling). */
        vec3 diffuseAlbedoIBL = (sssWeight > 0.0)
            ? mix(baseColor, sssColor, sssWeight)
            : baseColor;
        vec3 diffuseIBL = kD_ibl * irradiance * (diffuseAlbedoIBL / PI);
        float upFacing = smoothstep(0.0, 0.85, clamp(sceneUpCoord(N), 0.0, 1.0));
        float albedoLum = dot(baseColor, vec3(0.2126, 0.7152, 0.0722));
        float darkVertical = mix(0.68, 1.0, smoothstep(0.08, 0.55, albedoLum));
        float diffuseShape = mix(darkVertical, 1.28, upFacing);
        float specularShape = mix(0.35, 1.0, upFacing);
        float verticalOcc = 1.0 - smoothstep(0.18, 0.90, abs(sceneUpCoord(N)));
        float lowOcc = 1.0 - smoothstep(0.015, 0.145, sceneHeightCoord(worldPos));
        float darkOcc = 1.0 - smoothstep(0.10, 0.55, albedoLum);
        float localOcc = clamp(1.0 - verticalOcc *
                               (0.11 + 0.16 * lowOcc + 0.11 * darkOcc),
                               0.60, 1.0);
        diffuseIBL *= diffuseShape;

        /* IBL specular: pre-filtered env map (mip = roughness * mipCount) +
         * analytical split-sum integral. Matches ovrtx's straight mip lookup. */
        vec3 R = reflect(-V, N);
        float lod = roughness * (scene.envMipLevels - 1.0);
        vec3 prefilteredColor = textureLod(envMap, dirToEquirect(R), lod).rgb;
        if (fallbackLighting) {
            float preLuma = dot(prefilteredColor, vec3(0.2126, 0.7152, 0.0722));
            prefilteredColor = mix(vec3(preLuma), prefilteredColor, 0.62);
        }
        vec3 specularIBL = prefilteredColor * sampleEnvBRDF(F0, roughness, NdotV);
        specularIBL *= specularShape;

        if (manySceneLightsWithIbl) {
            /* Large Isaac-style interiors use practical fixtures plus a flat
             * DomeLight. Match OVRTX by keeping ceiling fill low while giving
             * low upward floors the bounce that the flat IBL otherwise lacks. */
            float upCoord = sceneUpCoord(N);
            float highInterior = smoothstep(1.0, 3.0, sceneHeightCoord(worldPos));
            float lowFloor = smoothstep(0.15, 0.85, clamp(upCoord, 0.0, 1.0)) *
                (1.0 - smoothstep(0.05, 0.50, sceneHeightCoord(worldPos)));
            float downwardFace = smoothstep(0.12, 0.72, clamp(-upCoord, 0.0, 1.0));
            float verticalFace = 1.0 - smoothstep(0.18, 0.90, abs(upCoord));
            float ceilingFace = highInterior * downwardFace *
                (1.0 - lowFloor);
            float upperVertical = highInterior * verticalFace *
                (1.0 - lowFloor);
            diffuseIBL *= mix(1.0, 0.30, max(ceilingFace, upperVertical * 0.18));
            specularIBL *= mix(1.0, 0.42, ceilingFace);
            localOcc *= mix(1.0, 0.78, ceilingFace);
        }

        float contact = mix(0.78, 1.0, contactVisibility(worldPos, N));
        ambient = (diffuseIBL + specularIBL) * ao * contact * localOcc;
        float coolBounce = upFacing * (1.0 - metallic) *
                           smoothstep(0.06, 0.45, albedoLum);
        ambient += vec3(0.035, 0.055, 0.090) *
                   coolBounce * ao * contact;
        float lowHorizontal = upFacing * (1.0 - metallic) *
                              (1.0 - smoothstep(0.006, 0.035, sceneHeightCoord(worldPos))) *
                              smoothstep(0.06, 0.42, albedoLum);
        ambient += vec3(0.070, 0.110, 0.180) *
                   lowHorizontal * ao * contact;

        /* === Subsurface back-light glow (IBL path) ====================
         * Pure wrap-NdotL has no effect on IBL since the irradiance map
         * already integrates over the hemisphere. To give jade its
         * characteristic translucent glow, sample the irradiance at the
         * back-facing normal (-N) and tint by the body color (which is
         * subsurface_color when the asset authors it explicitly, else
         * baseColor — see evalDirectionalLight comment). */
        if (sssWeight > 0.0) {
            vec3 backIrradiance = texture(irrMap, dirToEquirect(-N)).rgb;
            vec3 backGlow = backIrradiance * (sssColor / PI) * sssWeight * 0.2;
            ambient += backGlow * ao * localOcc * (1.0 - metallic);
        }

        /* When the scene has explicit lights, IBL acts as fill only —
         * dominant illumination comes from the ceiling fixtures. But since
         * we have no recursive GI, IBL must supply the bounce light that
         * reaches shelf interiors; keep it small but non-negligible. */
        if (sceneLights.nlights > 0) {
            float iblFill = max(rt_ibl_fill_scale, 0.0);
            if (fallbackLighting)
                iblFill = min(iblFill, 0.36);
            ambient *= iblFill;
            if (fallbackLighting) {
                /* Moana's public package lacks the HDR water/sky EXR path in
                 * this renderer, so authored direct lights do not get the
                 * path-traced bounce that keeps roots and cliff vegetation
                 * readable in RenderMan. Add a small albedo-preserving
                 * hemisphere fill after clamping the visible-dome IBL. */
                float hemiFill = mix(0.16, 0.32, upFacing);
                ambient += baseColor * vec3(0.95, 0.88, 0.72) *
                           hemiFill * ao * contact * localOcc;
            }
        }
        /* Boost for interior surfaces (seen through glass) */
        if (rayDepth > 0.5)
            ambient *= 1.5;
    } else {
        /* Hemisphere ambient fallback (no IBL).
         * ovrtx uses a procedural blue-ish sky dome
         * (`omni:rtx:background:source:type = "sky"`); approximate that
         * with a cool sky / warm ground split so upward-facing surfaces
         * pick up a desaturated blue tint. */
        vec3 skyColor    = vec3(0.60, 0.68, 0.82);
        vec3 groundColor = vec3(0.32, 0.31, 0.30);
        float hemi = dot(N, sceneUpVector()) * 0.5 + 0.5;
        vec3 ambientIrradiance = mix(groundColor, skyColor, hemi);
        ambientIrradiance += displayColorFallback ? vec3(0.09, 0.09, 0.105)
                                                  : vec3(0.10, 0.10, 0.12);
        if (rayDepth > 0.5)
            ambientIrradiance *= 2.0;
        vec3 kS_ambient = fresnelSchlick(NdotV, F0);
        vec3 kD_ambient = (1.0 - kS_ambient) * (1.0 - metallic);
        vec3 diffuseAlbedoAmb = baseColor;
        if (sssWeight > 0.0) {
            diffuseAlbedoAmb = mix(baseColor, sssColor, sssWeight);
        }
        ambient = kD_ambient * ambientIrradiance * diffuseAlbedoAmb * ao;
        ambient += kS_ambient * ao * 0.06;

        /* Subsurface back-light for the procedural-sky fallback: sample
         * the same hemisphere on the back side and add a small glow. */
        if (sssWeight > 0.0) {
            float hemiBack = dot(-N, sceneUpVector()) * 0.5 + 0.5;
            vec3 backIrradiance = mix(groundColor, skyColor, hemiBack);
            backIrradiance += vec3(0.10, 0.10, 0.12);
            ambient += backIrradiance * (sssColor / PI) * sssWeight * 0.2 * ao
                       * (1.0 - metallic);
        }

        /* Unmodulated sky fill for upward-facing surfaces. Approximates
         * the indirect sky bounce that ovrtx's path tracer delivers onto
         * floors: our dark floor albedo (~0.05) multiplied by any sky
         * illuminance still reads near-black, so we add a small additive
         * term independent of baseColor. Uses the authored up axis directly
         * (not a reflection vector) so normal-map perturbations don't cause alternating-tile
         * artifacts on glossy floors. */
        float skyUp = clamp(sceneUpCoord(N), 0.0, 1.0);
        float skyBounce = manySceneLightsNoIbl ? 2.35 : displaySkyBounce;
        vec3 skyBounceColor = manySceneLightsNoIbl
                            ? vec3(0.055, 0.062, 0.075)
                            : vec3(0.050, 0.060, 0.080);
        ambient += skyBounceColor * skyUp * ao * skyBounce;
        if (sceneLights.nlights > 0) {
            /* Authored-light / no-IBL scenes should not also receive the
             * renderer's old synthetic studio fill. OVRTX renders this case
             * mostly from the explicit fixtures plus a small, cool bounce;
             * the previous warm multiplier lifted cardboard closeups by
             * ~20-30 luma. */
            ambient *= manySceneLightsNoIbl ? vec3(0.78, 0.80, 0.84)
                                            : vec3(0.32, 0.34, 0.38);
        }
        if (manySceneLightsNoIbl) {
            float lowSceneHorizontal = smoothstep(0.15, 0.85, skyUp) *
                (1.0 - smoothstep(0.05, 0.45, sceneHeightCoord(worldPos)));
            ambient *= mix(0.36, 1.0, lowSceneHorizontal);
            ambient += vec3(0.045, 0.055, 0.070) * lowSceneHorizontal * ao *
                       (1.0 - metallic);
        }
        if (displayColorFallback) {
            ambient *= mix(0.58, 1.0, displayContact) *
                       displayFaceOcclusion *
                       mix(1.0, 0.38, displayDarkUp) *
                       mix(1.0, 1.08, displayBrightUp);
        }
    }
    if (authoredKeyOnlyNoIbl) {
        ambient = vec3(0.0);
    }

    /* Direct lighting */
    vec3 Lo = vec3(0.0);
    vec3 keyDir;  /* used by clearcoat — first directional light or fallback. */
    bool keyDirValid = false;

    if (sceneLights.nlights > 0) {
        /* Iterate scene-defined lights (RectLight / DistantLight). */
        for (int li = 0; li < sceneLights.nlights; ++li) {
            GpuLight L = sceneLights.items[li];
            if (L.kind == LIGHT_KIND_RECT) {
                Lo += evalRectLight(L, N, V, baseColor, metallic, roughness, F0, worldPos,
                                    sssWeight, sssColor, sssWrap);
            } else if (L.kind == LIGHT_KIND_DISTANT) {
                Lo += evalDistantLight(L, N, V, baseColor, metallic, roughness, F0, worldPos,
                                       sssWeight, sssColor, sssWrap);
                if (!keyDirValid) {
                    keyDir = normalize(-L.normal);
                    keyDirValid = true;
                }
            } else if (L.kind == LIGHT_KIND_SPHERE) {
                Lo += evalSphereLight(L, N, V, baseColor, metallic, roughness, F0, worldPos,
                                      sssWeight, sssColor, sssWrap);
                if (!keyDirValid) {
                    keyDir = normalize(L.position - worldPos);
                    keyDirValid = true;
                }
            }
        }
    } else if (scene.envMipLevels == 0.0 || scene.envIntensity < -1.0) {
        bool viewerFallbackIbl = (scene.envMipLevels > 0.0 &&
                                  scene.envIntensity < -1.0);
        if (viewerFallbackIbl) {
            Lo += evalDirectionalLight(OVRTX_FALLBACK_KEY_DIR,
                                       OVRTX_FALLBACK_KEY_COLOR,
                                       OVRTX_FALLBACK_KEY_INTENSITY,
                                       N, V, baseColor, metallic, roughness, F0,
                                       worldPos, sssWeight, sssColor, sssWrap);
            keyDir = OVRTX_FALLBACK_KEY_DIR;
        } else {
            /* nanousdview's OVRTX fallback authors a light at the camera.
             * Match that shape for raw no-light assets such as OpenChessSet. */
            vec3 syntheticKey = V;
            float keyIntensity = displayColorFallback ? 1.42 : 1.48;
            float fillIntensity = displayColorFallback ? 0.14 : 0.16;
            Lo += evalDirectionalLight(syntheticKey, vec3(1.0, 0.97, 0.92), keyIntensity,
                                       N, V, baseColor, metallic, roughness, F0, worldPos,
                                       sssWeight, sssColor, sssWrap);
            vec3 fillAlbedo = (sssWeight > 0.0)
                ? mix(baseColor, sssColor, sssWeight)
                : baseColor;
            float domeHemi = clamp(dot(N, sceneUpVector()) * 0.5 + 0.5, 0.0, 1.0);
            Lo += fillAlbedo * vec3(0.72, 0.82, 1.0) * fillIntensity * domeHemi;
            keyDir = syntheticKey;
        }
        keyDirValid = true;
    }
    if (displayColorFallback) {
        Lo *= mix(0.84, 1.0, displayContact) *
              mix(1.0, 0.45, displayDarkUp) *
              mix(1.0, 1.10, displayBrightUp);
    }

    vec3 result = ambient + Lo + emissive;
    if (manySceneLightsWithIbl) {
        float upCoord = sceneUpCoord(N);
        float lowFloor = smoothstep(0.15, 0.85, clamp(upCoord, 0.0, 1.0)) *
            (1.0 - smoothstep(0.05, 0.50, sceneHeightCoord(worldPos)));
        float highInterior = smoothstep(1.0, 3.0, sceneHeightCoord(worldPos));
        float downwardFace = smoothstep(0.12, 0.72, clamp(-upCoord, 0.0, 1.0));
        float verticalFace = 1.0 - smoothstep(0.18, 0.90, abs(upCoord));
        float ceilingFace = highInterior * downwardFace *
            (1.0 - lowFloor);
        float upperVertical = highInterior * verticalFace *
            (1.0 - lowFloor);
        float interiorScale = mix(1.0, 0.36, ceilingFace) *
                              mix(1.0, 0.84, upperVertical);
        vec3 floorLift = mix(vec3(1.0), vec3(3.14, 3.28, 3.42), lowFloor);
        result = (ambient + Lo) * interiorScale * floorLift + emissive;
    }
    if (manySceneLightsNoIbl) {
        float floorShape = smoothstep(0.15, 0.85, clamp(sceneUpCoord(N), 0.0, 1.0)) *
            (1.0 - smoothstep(0.05, 0.45, sceneHeightCoord(worldPos)));
        float highInterior = smoothstep(1.2, 3.0, sceneHeightCoord(worldPos));
        float nonFloorScale = mix(0.86, 0.58, highInterior);
        result *= mix(nonFloorScale, 1.0, floorShape);
    }

    /* Clearcoat: second GGX specular lobe with fixed IOR 1.5 (F0 = 0.04) */
    if (clearcoat > 0.0 && keyDirValid) {
        vec3 H = normalize(V + keyDir);
        float ccNdotL = max(dot(N, keyDir), 0.0);
        float ccD = distributionGGX(N, H, clearcoatRoughness);
        float ccG = geometrySmith(N, V, keyDir, clearcoatRoughness);
        vec3  ccF = fresnelSchlick(max(dot(H, V), 0.0), vec3(0.04));
        vec3  ccSpec = (ccD * ccG * ccF) / (4.0 * NdotV * ccNdotL + 0.0001);
        float ccShadow = traceShadow(worldPos + N * 0.01, keyDir, 100000.0);
        result += ccSpec * vec3(1.0, 0.95, 0.85) * ccNdotL * clearcoat * ccShadow;
    }

    /* Exposure + ACES filmic tone mapping.
     * Lower fixed exposure to compensate for IBL+direct energy; ovrtx's
     * auto-exposure settles roughly here for a warehouse lit by 4e5-nit
     * ceiling rect lights. */
    /* Surface exposure: 0.5 baseline (chess + materialx-rig calibrated
     * against MaterialXView), bumped by an intensity-dependent factor
     * when a USD DomeLight authors a high intensity. ovrtx auto-exposes
     * the whole frame; we approximate with a log-tracking soft scale.
     *
     * Calibration: intensity=1 → boost=1.0 (no effect, gated below 2),
     * intensity=1000 → boost=1.4 (sphere_one rig empirical match against
     * ovrtx — sphere 162/155, ground 160/159), intensity beyond 1000
     * tracks slowly upward (vs. the previous smoothstep which saturated
     * past intensity=100, breaking high-energy scenes). Cap at 1.5 so
     * the boost can't run away on USDs authored at intensity≥10000. */
    float intensity = (scene.envIntensity < -1.0) ? 1.0 : abs(scene.envIntensity);
    /* Phase 1: gentle log boost intensity 1→1000 (calibrated point) */
    float t1 = clamp(log(1.0 + intensity) / log(1001.0), 0.0, 1.25);
    float gate = smoothstep(1.0, 2.0, intensity);  /* keep intensity≤1 unchanged */
    float boost1 = 1.0 + 0.4 * t1 * gate;
    /* Phase 2: linear-with-intensity ramp past 1000 to track ovrtx's
     * surface saturation at high intensities. Suite measurement at
     * intensity=10000 showed our Vulkan surfaces stuck at ~165 mean
     * while ovrtx ramped to ~245 (auto-exposure + raw intensity =
     * frame-wide saturation). pow(., 0.85) is mildly sub-linear so
     * the boost doesn't run away faster than ACES can saturate. */
    float boost2 = pow(max(intensity / 1000.0, 1.0), 0.85);
    float exposure = 0.5 * boost1 * boost2;
    if (scene.envIntensity < -1.0)
        exposure *= 1.58;
    if (manySceneLightsNoIbl)
        exposure *= 1.7;
    if (manySceneLightsWithIbl)
        exposure *= 0.45;
    bool hadAnyLight = result.r > 0.0 || result.g > 0.0 || result.b > 0.0;
    if (noLightNoIbl)
        result *= vec3(1.18, 1.04, 0.90);
    else if (fallbackLighting)
        result *= vec3(1.08, 1.03, 0.92);
    result = acesFilmic(result * exposure * max(tone_exposure_scale, 0.0));
    /* UNORM rounding-floor lift.
     *
     * The rt_image is RGBA8_UNORM (`floor(x*255 + 0.5)`), so any ACES
     * output below ~0.002 rounds to byte 0 → pure black pixels on
     * surfaces that ARE hit and shaded but very dimly (synthetic
     * 3-point + hemisphere ambient on metallic / back-facing geometry
     * in unlit scenes — Microwave064's back face was the first
     * casualty; Blender003 + CoffeeMachine064 follow the same path).
     *
     * Keep exposure unchanged so well-lit IBL frames don't drift; just
     * lift the post-tonemap output to the lowest representable byte
     * when there's any positive contribution from ambient/Lo/emissive.
     * Floor at 2/255 ≈ 0.0078 — one byte above the rounding boundary,
     * still imperceptibly dark, but distinct from an actual miss. */
    if (hadAnyLight) {
        result = max(result, vec3(2.0/255.0));
    }
    /* sRGB swapchain handles gamma — no manual pow needed */

    /* Clay-viz override (NUSD_CLAY_VIZ=1 -> tone_flags bit 30): replace the
     * PBR result with a diagnostic warm-clay look. */
    if ((tone_flags & TONE_FLAG_CLAY_VIZ) != 0u) {
        result = nu_clay_shade(N, gl_WorldRayDirectionEXT);
    }
    hitValue = vec4(result, 0.0);
}
