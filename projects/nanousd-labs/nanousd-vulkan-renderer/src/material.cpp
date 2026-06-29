// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * material.cpp — MaterialX-based material system for the nanousd-vulkan-renderer.
 *
 * Standard pipeline:
 *   USD Material prims → MaterialX Document → GLSL codegen → SPIR-V (shaderc)
 *
 * Uses MaterialXGenGlsl for shader generation and shaderc for runtime
 * GLSL-to-SPIR-V compilation, which is the normal way MaterialX is used
 * with Vulkan.
 */

#include <nanousd/nanousdapi.h>
#include <cstddef>

#include "material.h"
#include "mdl_bridge.h"
#include "ptex_material.h"

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Library.h>
#include <MaterialXFormat/Environ.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXGenGlsl/GlslShaderGenerator.h>
#include <MaterialXGenGlsl/VkShaderGenerator.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/UnitSystem.h>
#include <MaterialXCore/Unit.h>

#include <shaderc/shaderc.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <initializer_list>

namespace mx = MaterialX;
namespace fs = std::filesystem;

/* Load-time instrumentation (load-time campaign Phase: confirm the ~30s split
 * decode-vs-resolve before parallelizing). Output-neutral: pure timing, no
 * pixel/binding change. Decode = stb_image; resolve = resolve_texture_path
 * (the ~1.1M-stat path-resolution suspect). Printed once at end of
 * materials_load; upload is inferred as the remainder (it lives in renderer.c
 * gpu_upload_materials). */
#include <time.h>
#include <cstdint>
static double g_tex_decode_ms = 0.0;
static long   g_tex_decode_n  = 0;
static double g_tex_resolve_ms = 0.0;   /* actual (cache-miss) resolution work */
static long   g_tex_resolve_n  = 0;     /* cache-miss resolutions */
static long   g_tex_resolve_hits = 0;   /* memoized hits (avoided re-resolve) */
/* Per-load resolution memo: resolve_texture_path is called once per texture
 * REFERENCE (~14.6k for ~156 unique paths on the warehouse) but its result is
 * deterministic per (authored path, scene_dir, stage) within a single load.
 * Memoizing collapses the ~14.6k stat-heavy resolutions to the unique set.
 * Bit-identical: same inputs -> same resolved string. Cleared at the top of
 * materials_load so it never persists stale state across loads/scenes. */
static std::unordered_map<std::string, std::string> g_resolve_cache;
static std::unordered_set<std::string> g_failed_texture_cache;
/* Parallel texture pre-decode (NUSD_PARALLEL_DECODE): textures decoded up front
 * across a thread pool land here keyed by resolved path; load_texture() consumes
 * them so the per-material readers stay unchanged. MaterialTexture is POD (no
 * destructor) — map erase transfers pixel ownership to the caller, no copy. */
static std::unordered_map<std::string, MaterialTexture> g_predecoded;
static long   g_tex_predecode_n       = 0;    /* load_texture pre-decode hits */
static double g_tex_predecode_wall_ms = 0.0;  /* parallel pre-pass wall time */
static inline double nu_load_now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}
struct NuResolveTimer {
    double t0;
    NuResolveTimer() : t0(nu_load_now_ms()) {}
    ~NuResolveTimer() { g_tex_resolve_ms += nu_load_now_ms() - t0; ++g_tex_resolve_n; }
};
/* Profiling: per-material reader cost (USD attrib reads + composite bakes) —
 * splits the residual "other_material_cpu" to see if it is read-bound (the
 * GetPrimIndex-race-blocked part) or bake-bound (race-free, parallelisable). */
static double g_reader_ms = 0.0;
static long   g_reader_n  = 0;
struct NuReaderTimer {
    double t0;
    NuReaderTimer() : t0(nu_load_now_ms()) {}
    ~NuReaderTimer() { g_reader_ms += nu_load_now_ms() - t0; ++g_reader_n; }
};
/* Bake-only timer (race-free pixel work) — reader_ms minus bake_ms ≈ the
 * attribute-read (Token/Path-interning, race-blocked) portion. Gates whether
 * a renderer-only parallel reader can win without touching nanousd core. */
static double g_bake_ms = 0.0;
static long   g_bake_n  = 0;
struct NuBakeTimer {
    double t0;
    NuBakeTimer() : t0(nu_load_now_ms()) {}
    ~NuBakeTimer() { g_bake_ms += nu_load_now_ms() - t0; ++g_bake_n; }
};

static bool is_package_identifier_path(const std::string& path)
{
    return path.find('[') != std::string::npos && !path.empty() &&
           path.back() == ']';
}

static std::string resolve_stage_package_asset(const char* path,
                                               NanousdStage stage)
{
    if (!path || !path[0]) return std::string();
    std::string authored(path);
    if (is_package_identifier_path(authored)) return authored;
    if (!stage) return std::string();

    char resolved[4096];
    if (!nanousd_stage_resolve_asset_path(stage, path, resolved,
                                          sizeof(resolved)))
        return std::string();
    std::string location(resolved);
    return is_package_identifier_path(location) ? location : std::string();
}

/* ---- Global MaterialX state ---- */

static mx::DocumentPtr g_stdlib;
static mx::GenContext*  g_gen_context = nullptr;
static mx::ShaderGeneratorPtr g_shader_gen;
static bool g_materialx_initialized = false;

/* ---- Default PBR shaders (fallback when MaterialX codegen is not needed) ---- */

static const char* k_default_vert_glsl = R"(
#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;     // world transform (for normal/position to world space)
    vec4 color;     // .w > 0.5 = override vertex color
    vec4 eye_pos;   // xyz = camera world position, w = env_mip_levels (0 = no IBL)
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in uint inMaterialId;
layout(location = 5) in vec3 inTangent;
layout(location = 6) in float inTangentSign;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec2 fragTexCoord;
layout(location = 4) flat out uint fragMaterialId;
layout(location = 5) out vec3 fragEyePos;
layout(location = 6) out vec3 fragTangent;
layout(location = 7) out float fragTangentSign;

void main() {
    gl_Position = vec4(inPosition, 1.0) * pc.mvp;
    fragWorldPos = (pc.model * vec4(inPosition, 1.0)).xyz;
    fragNormal = normalize(mat3(pc.model) * inNormal);
    fragTangent = normalize(mat3(pc.model) * inTangent);
    fragTangentSign = inTangentSign;
    fragColor = pc.color.w > 0.5 ? pc.color.rgb : inColor;
    fragTexCoord = inTexCoord;
    fragMaterialId = inMaterialId;
    fragEyePos = pc.eye_pos.xyz;
}
)";

static const char* k_default_pbr_frag_glsl = R"(
#version 450
#extension GL_EXT_nonuniform_qualifier : enable

/* Material parameters SSBO */
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
    float opacity_threshold;
    vec4  subsurface_color;
    vec4  subsurface_radius;
    vec4  transmission_color;
    float subsurface_weight;
    float subsurface_scale;
    float transmission_weight;
    float transmission_ior;
    int   tex_subsurface_weight;
    int   tex_transmission_weight;
    int   sss_color_authored;
    int   use_specular_workflow;
    vec4  specular_color;
    vec4  normal_tex_scale;
    vec4  normal_tex_bias;
    vec4  mdl_uv_transform;
    int   v_flip;
    float roughness_tex_scale;
    float roughness_tex_bias;
    int   _pad_v_flip;
};

layout(set = 0, binding = 0, std430) readonly buffer MaterialBuffer {
    MaterialData materials[];
} mat_buf;

layout(set = 0, binding = 1) uniform sampler2D textures[];
layout(set = 0, binding = 2) uniform sampler2D envMap;
layout(set = 0, binding = 3) uniform sampler2D brdfLUT;
layout(set = 0, binding = 4) uniform sampler2D irrMap;  /* SH irradiance */

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
    vec4 color;
    vec4 eye_pos;   /* xyz = camera, w = env_mip_levels (0 = no IBL) */
} pc;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragTexCoord;
layout(location = 4) flat in uint fragMaterialId;
layout(location = 5) in vec3 fragEyePos;
layout(location = 6) in vec3 fragTangent;
layout(location = 7) in float fragTangentSign;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

/* GGX/Trowbridge-Reitz normal distribution function */
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

/* Schlick-GGX geometry function */
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

/* Fresnel-Schlick */
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float r) {
    return F0 + (max(vec3(1.0 - r), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

/* Lat-long environment map lookup. Matches Hydra / OVRTX DomeLight for these
 * assets: +Z is the north pole and +Y is zero-longitude forward. */
vec2 envMapUV(vec3 dir) {
    const float kDomeLongitudeOffset = 340.0 / 360.0;
    float phi = atan(dir.x, dir.y);
    float theta = asin(clamp(dir.z, -1.0, 1.0));
    return vec2(fract(phi / (2.0 * PI) + 0.5 + kDomeLongitudeOffset),
                0.5 - theta / PI);
}

void main() {
    MaterialData mat = mat_buf.materials[fragMaterialId];
    vec3 N = normalize(fragNormal);

    /* UDIM UV scaling: divide UVs by atlas grid dimensions */
    vec2 tc = fragTexCoord / max(vec2(mat.udim_scale_u, mat.udim_scale_v), vec2(1.0));
    tc = tc * mat.mdl_uv_transform.xy + mat.mdl_uv_transform.zw;
    if (mat.v_flip != 0) tc.y = 1.0 - tc.y;

    /* Normal mapping via per-vertex TBN with bitangent handedness */
    if (mat.tex_indices[1] >= 0) {
        vec3 T = fragTangent;
        float tLen = length(T);
        if (tLen > 0.001) {
            T = T / tLen;
            /* Gram-Schmidt re-orthogonalize */
            T = normalize(T - dot(T, N) * N);
            vec3 B = cross(N, T) * fragTangentSign;
            mat3 TBN = mat3(T, B, N);
            vec3 mapN = texture(textures[nonuniformEXT(mat.tex_indices[1])], tc).rgb * 2.0 - 1.0;
            mapN.xy *= mat.normal_scale;
            N = normalize(TBN * mapN);
        }
    }

    /* Two-sided rendering: flip normals that face away from the camera.
     * Instance transforms (e.g. 180° Z-rotation for opposing chess pieces)
     * can invert normals; without this fix NdotV→0 causes Fresnel blowout. */
    {
        vec3 V_pre = normalize(fragEyePos - fragWorldPos);
        if (dot(N, V_pre) < 0.0)
            N = -N;
    }

    /* Base color: texture or parameter, fallback to vertex color */
    vec3 baseColor;
    float alpha = mat.base_color.a;
    if (mat.tex_indices[0] >= 0) {
        vec4 texColor = texture(textures[nonuniformEXT(mat.tex_indices[0])], tc);
        baseColor = texColor.rgb;
        alpha *= texColor.a;
    } else if (mat.use_vertex_color != 0) {
        baseColor = fragColor;
    } else {
        baseColor = mat.base_color.rgb;
    }

    /* Roughness / metallic */
    float roughness = mat.roughness;
    float metallic = mat.metallic;

    if (mat.tex_indices[2] >= 0) {
        float roughSample = texture(textures[nonuniformEXT(mat.tex_indices[2])], tc).g;
        float roughScale = (mat.roughness_tex_scale != 0.0) ? mat.roughness_tex_scale : 1.0;
        roughness = roughSample * roughScale + mat.roughness_tex_bias;
    }
    if (mat.tex_indices[3] >= 0) {
        metallic = texture(textures[nonuniformEXT(mat.tex_indices[3])], tc).b;
    }
    roughness = clamp(roughness, 0.04, 1.0);

    /* Emissive with proper texture-driven emission */
    vec3 emissiveConst = mat.emissive_color.rgb;
    float emissiveIntensity = mat.emissive_color.a;
    if (emissiveIntensity > 100.0)
        emissiveIntensity = 1.0 + log2(emissiveIntensity / 100.0);
    vec3 emissive = emissiveConst * emissiveIntensity;
    if (mat.tex_indices[4] >= 0) {
        vec3 emissive_tex = texture(textures[nonuniformEXT(mat.tex_indices[4])], tc).rgb;
        if (dot(emissiveConst, emissiveConst) < 0.001) {
            emissive = emissive_tex * emissiveIntensity;
        } else {
            emissive *= emissive_tex;
        }
    }

    /* Occlusion */
    float ao = mat.occlusion;
    if (mat.tex_indices[5] >= 0) {
        ao *= texture(textures[nonuniformEXT(mat.tex_indices[5])], tc).r;
    }

    /* PBR lighting */
    vec3 V = normalize(fragEyePos - fragWorldPos);
    float NdotV = max(dot(N, V), 0.001);

    /* IOR-driven F0: use material IOR instead of hardcoded 0.04 */
    float f0_dielectric = pow((mat.ior - 1.0) / (mat.ior + 1.0), 2.0);
    vec3 F0 = mix(vec3(f0_dielectric), baseColor, metallic);

    float envMipLevels = pc.eye_pos.w;
    int hasIBL = (envMipLevels > 0.5) ? 1 : 0;

    /* Ambient / IBL */
    vec3 ambient;
    if (hasIBL != 0) {
        /* Diffuse IBL: sample SH irradiance map (proper cosine-convolved) */
        vec3 irradiance = texture(irrMap, envMapUV(N)).rgb;
        vec3 F_ibl = fresnelSchlickRoughness(NdotV, F0, roughness);
        vec3 kD_ibl = (1.0 - F_ibl) * (1.0 - metallic);
        vec3 diffuseIBL = irradiance * (baseColor / PI) * kD_ibl;

        /* Specular IBL: sample env at roughness-based LOD.
         * Mips are box-filtered (not GGX pre-filtered), so blend toward
         * irradiance-based fallback for rough surfaces to reduce noise.
         * Irradiance = ∫L·cos(θ)dω ≈ L_avg·π, but the split-sum expects
         * average radiance L_avg, so divide by PI when using as fallback. */
        vec3 R = reflect(-V, N);
        float lod = roughness * (envMipLevels - 1.0);
        vec3 envSpecular = textureLod(envMap, envMapUV(R), lod).rgb;
        float roughBlend = smoothstep(0.3, 0.9, roughness);
        vec3 prefilteredColor = mix(envSpecular, irradiance / PI, roughBlend);

        /* Split-sum: BRDF integration from LUT */
        vec2 brdf = texture(brdfLUT, vec2(NdotV, roughness)).rg;
        vec3 specularIBL = prefilteredColor * (F_ibl * brdf.x + brdf.y);

        /* Directional albedo compensation (energy conservation) —
         * accounts for energy lost by the single-scattering BRDF model.
         * See Kulla & Conty (2017). */
        float Ess = brdf.x + brdf.y;  /* directional albedo at NdotV */
        vec3 energyCompensation = 1.0 + F0 * (1.0 / max(Ess, 0.001) - 1.0);
        specularIBL *= energyCompensation;

        ambient = (diffuseIBL + specularIBL) * ao;
    } else {
        /* Fallback: hemisphere ambient */
        vec3 skyColor    = vec3(0.45, 0.52, 0.68);
        vec3 groundColor = vec3(0.15, 0.13, 0.10);
        float hemi = dot(N, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
        vec3 ambientIrradiance = mix(groundColor, skyColor, hemi) + vec3(0.08, 0.08, 0.10);
        vec3 kS_amb = fresnelSchlick(NdotV, F0);
        vec3 kD_amb = (1.0 - kS_amb) * (1.0 - metallic);
        ambient = kD_amb * ambientIrradiance * baseColor * ao;
        ambient += kS_amb * ao * 0.15;
    }

    /* Analytical lights */
    vec3 Lo = vec3(0.0);
    {
        /* Key directional light — full strength without IBL, fill role with IBL */
        vec3 lightDir = normalize(vec3(0.3, 1.0, 0.5));
        vec3 lightColor = vec3(1.0, 0.95, 0.85);
        float lightScale = (hasIBL != 0) ? 0.0 : 1.0;

        vec3 L = lightDir;
        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);

        /* Cook-Torrance BRDF */
        float D = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        vec3  F = fresnelSchlickRoughness(max(dot(H, V), 0.0), F0, roughness);

        vec3 numerator = D * G * F;
        float denominator = 4.0 * NdotV * NdotL + 0.0001;
        vec3 specular = numerator / denominator;

        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        Lo = (kD * baseColor / PI + specular) * lightColor * NdotL * lightScale;

        /* Fill light from opposite side */
        vec3 fillDir = normalize(vec3(-0.5, 0.4, -0.3));
        float fillNdotL = max(dot(N, fillDir), 0.0);
        float fillScale = (hasIBL != 0) ? 0.0 : 0.15;
        Lo += baseColor * vec3(0.7, 0.8, 1.0) * fillScale * fillNdotL;
    }

    vec3 color = ambient + Lo + emissive;

    /* Clearcoat: second GGX specular lobe with fixed IOR 1.5 (F0 = 0.04) */
    if (mat.clearcoat > 0.0) {
        float ccRoughness = clamp(mat.clearcoat_roughness, 0.04, 1.0);
        if (hasIBL != 0) {
            /* IBL clearcoat: sample env at clearcoat roughness */
            vec3 ccR = reflect(-V, N);
            float ccLod = ccRoughness * (envMipLevels - 1.0);
            vec3 ccEnv = textureLod(envMap, envMapUV(ccR), ccLod).rgb;
            vec3 ccF = fresnelSchlick(NdotV, vec3(0.04));
            color += ccEnv * ccF * mat.clearcoat;
        } else {
            vec3 ccL = normalize(vec3(0.3, 1.0, 0.5));
            vec3 ccH = normalize(V + ccL);
            float ccNdotL = max(dot(N, ccL), 0.0);
            float ccD = distributionGGX(N, ccH, ccRoughness);
            float ccG = geometrySmith(N, V, ccL, ccRoughness);
            vec3  ccF = fresnelSchlick(max(dot(ccH, V), 0.0), vec3(0.04));
            vec3  ccSpec = (ccD * ccG * ccF) / (4.0 * NdotV * ccNdotL + 0.0001);
            color += ccSpec * vec3(1.0, 0.95, 0.85) * ccNdotL * mat.clearcoat;
        }
    }

    /* PBR lighting */

    /* ACES filmic tone mapping */
    float exposure = (hasIBL != 0) ? 1.0 : 1.2;
    color *= exposure;
    color = clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
    /* sRGB swapchain handles gamma — no manual pow needed */

    /* Opacity */
    if (alpha < 0.01) discard;

    outColor = vec4(color, alpha * mat.opacity);
}
)";

/* ---- Environment background shaders ---- */

static const char* k_env_bg_vert_glsl = R"(
#version 450
layout(location = 0) out vec2 v_uv;
void main() {
    vec2 pos = vec2(float(gl_VertexIndex & 1) * 4.0 - 1.0,
                    float((gl_VertexIndex >> 1) & 1) * 4.0 - 1.0);
    v_uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.9999, 1.0);
}
)";

static const char* k_env_bg_frag_glsl = R"(
#version 450
layout(location = 0) in vec2 v_uv;
layout(push_constant) uniform PushConstants {
    mat4 mvp;       /* view_inv */
    mat4 model;     /* proj_inv */
    vec4 color;
    vec4 eye_pos;
    vec4 ibl_params; /* x=has_ibl, y=mips, z=USD DomeLight intensity */
    vec4 tone_params; /* x=surface exposure scale, y=visible sky scale */
} pc;
layout(set = 0, binding = 2) uniform sampler2D envMap;
layout(location = 0) out vec4 outColor;

const float PI = 3.14159265;

void main() {
    /* Reconstruct world-space ray direction using separate view/proj inverses.
     * This matches the RT raygen shader's proven approach.
     * pc.mvp = view_inv, pc.model = proj_inv */
    vec2 d = v_uv * 2.0 - 1.0;
    vec4 target = vec4(d.x, d.y, 1.0, 1.0) * pc.model;  /* clip → view space */
    vec3 viewDir = normalize(target.xyz);
    vec3 dir = (vec4(viewDir, 0.0) * pc.mvp).xyz;         /* view → world space */

    /* Equirectangular lookup, up-axis aware (matches RT rmiss + mesh.frag).
     * Up axis is packed into tone_flags bits 24-25 by gpu_draw_env_background;
     * the Z-up branch is bit-identical to the old hardcoded path. */
    const float kDomeLongitudeOffset = 340.0 / 360.0;
    uint upAxis = (floatBitsToUint(pc.tone_params.w) >> 24) & 3u;
    float u, v;
    if (upAxis == 1u) {            // Y-up (USD default)
        u = fract(atan(dir.x, dir.z) * (0.5 / PI) + 0.5 + kDomeLongitudeOffset);
        v = asin(clamp(dir.y, -1.0, 1.0)) * (1.0 / PI) + 0.5;
    } else if (upAxis == 0u) {     // X-up
        u = fract(atan(dir.y, dir.z) * (0.5 / PI) + 0.5 + kDomeLongitudeOffset);
        v = asin(clamp(dir.x, -1.0, 1.0)) * (1.0 / PI) + 0.5;
    } else {                       // Z-up (== legacy)
        u = fract(atan(dir.x, dir.y) * (0.5 / PI) + 0.5 + kDomeLongitudeOffset);
        v = asin(clamp(dir.z, -1.0, 1.0)) * (1.0 / PI) + 0.5;
    }
    vec2 uv = vec2(u, 1.0 - v);  /* flip V for top-down convention */

    vec3 color = textureLod(envMap, uv, 0.0).rgb;
    /* Match raytrace.rmiss.glsl's visible-dome exposure. Surface BRDFs
     * sample the auto-exposed HDR; the directly visible sky additionally
     * receives the authored DomeLight intensity through a calibrated
     * photographic exposure scale before ACES. */
    const float kPhotoExposure = 0.000315;
    color *= abs(pc.ibl_params.z) * kPhotoExposure * max(pc.tone_params.y, 0.0);
    color = clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
    color = pow(color, vec3(1.0 / 2.2));
    outColor = vec4(color, 1.0);
}
)";

/* ---- SPIR-V compilation via shaderc ---- */

struct SpvResult {
    uint32_t* data;
    uint32_t  size;
    bool      ok;
    std::string error;
};

static SpvResult compile_glsl_to_spirv(const char* source,
                                       shaderc_shader_kind kind,
                                       const char* name)
{
    SpvResult result = {};
    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    if (!compiler) {
        result.error = "Failed to create shaderc compiler";
        return result;
    }

    shaderc_compile_options_t options = shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(options,
        shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    shaderc_compile_options_set_target_spirv(options, shaderc_spirv_version_1_5);
    shaderc_compile_options_set_optimization_level(options,
        shaderc_optimization_level_performance);

    shaderc_compilation_result_t res = shaderc_compile_into_spv(
        compiler, source, strlen(source), kind, name, "main", options);

    if (shaderc_result_get_compilation_status(res) != shaderc_compilation_status_success) {
        result.error = shaderc_result_get_error_message(res);
        shaderc_result_release(res);
        shaderc_compile_options_release(options);
        shaderc_compiler_release(compiler);
        return result;
    }

    size_t spv_size = shaderc_result_get_length(res);
    const char* spv_data = shaderc_result_get_bytes(res);

    result.data = (uint32_t*)malloc(spv_size);
    memcpy(result.data, spv_data, spv_size);
    result.size = (uint32_t)spv_size;
    result.ok = true;

    shaderc_result_release(res);
    shaderc_compile_options_release(options);
    shaderc_compiler_release(compiler);
    return result;
}

/* ---- MaterialX document creation from USD material properties ---- */

static mx::DocumentPtr create_mtlx_from_usd_preview_surface(
    const MaterialParams& params,
    const std::vector<std::string>& texture_paths)
{
    mx::DocumentPtr doc = mx::createDocument();
    // Note: stdlib is set as data library in generate_glsl_from_mtlx()

    // Create a node graph for the material
    auto nodegraph = doc->addNodeGraph("NG_material");

    // Create UsdPreviewSurface node
    auto surfaceNode = nodegraph->addNode("UsdPreviewSurface", "SR_material",
                                          "surfaceshader");

    // Set parameters
    surfaceNode->setInputValue("diffuseColor",
        mx::Color3(params.base_color[0], params.base_color[1], params.base_color[2]));
    surfaceNode->setInputValue("metallic", params.metallic);
    surfaceNode->setInputValue("roughness", params.roughness);
    surfaceNode->setInputValue("opacity", params.opacity);
    surfaceNode->setInputValue("ior", params.ior);
    surfaceNode->setInputValue("clearcoat", params.clearcoat);
    surfaceNode->setInputValue("clearcoatRoughness", params.clearcoat_roughness);

    if (params.emissive_color[0] > 0.0f || params.emissive_color[1] > 0.0f ||
        params.emissive_color[2] > 0.0f) {
        surfaceNode->setInputValue("emissiveColor",
            mx::Color3(params.emissive_color[0], params.emissive_color[1],
                       params.emissive_color[2]));
    }

    // Add texture nodes for bound textures
    if (params.tex_indices[TEX_DIFFUSE_COLOR] >= 0 &&
        params.tex_indices[TEX_DIFFUSE_COLOR] < (int)texture_paths.size()) {
        auto texNode = nodegraph->addNode("image", "diffuse_tex", "color3");
        texNode->setInputValue("file",
            texture_paths[params.tex_indices[TEX_DIFFUSE_COLOR]],
            "filename");
        surfaceNode->setConnectedNode("diffuseColor", texNode);
    }

    if (params.tex_indices[TEX_NORMAL] >= 0 &&
        params.tex_indices[TEX_NORMAL] < (int)texture_paths.size()) {
        auto texNode = nodegraph->addNode("image", "normal_tex", "vector3");
        texNode->setInputValue("file",
            texture_paths[params.tex_indices[TEX_NORMAL]],
            "filename");
        auto normalMap = nodegraph->addNode("normalmap", "normalmap1", "vector3");
        normalMap->setConnectedNode("in", texNode);
        surfaceNode->setConnectedNode("normal", normalMap);
    }

    // Create output
    auto output = nodegraph->addOutput("out_surface", "surfaceshader");
    output->setConnectedNode(surfaceNode);

    // Create surface material
    auto material = doc->addNode("surfacematerial", "M_material", "material");
    material->setConnectedOutput("surfaceshader", output);

    return doc;
}

/*
 * Note: MaterialX 1.39.4 fixed the previously-broken VkShaderGenerator
 * vertex-data block. Earlier versions (1.39.2 and prior) emitted
 * "layout(location=N) in vec3 fieldName;" alongside "vd.fieldName"
 * references with no backing struct, which was worked around by
 * stripping "vd." from the generated GLSL.
 *
 * 1.39.4 emits a proper interface block —
 *   layout(location=N) in Block { fields... } vd;
 * — so "vd.fieldName" is valid GLSL and no fixup is required.
 */

/* ---- Generate GLSL from MaterialX document ---- */

struct GlslShaders {
    std::string vertex;
    std::string fragment;
    bool ok;
    std::string error;
};

static GlslShaders generate_glsl_from_mtlx(mx::DocumentPtr doc)
{
    GlslShaders result = {};

    try {
        // Set stdlib as data library (avoids duplicate import issues)
        doc->setDataLibrary(g_stdlib);

        // Register shader metadata from this document.
        // Note: in MaterialX <= 1.39.2 this was preceded by an explicit
        // loadStructTypeDefs(doc) call. 1.39.4 internalised that step
        // inside generate(), so it's no longer needed here.
        g_shader_gen->registerShaderMetadata(doc, *g_gen_context);

        // Clear node implementation cache for fresh codegen
        g_gen_context->clearNodeImplementations();

        // Find renderable element: material node or surfaceshader output
        mx::TypedElementPtr renderable = nullptr;
        for (auto& node : doc->getNodes()) {
            if (node->getType() == "material") {
                renderable = node;
                break;
            }
        }
        if (!renderable) {
            for (auto& output : doc->getOutputs()) {
                if (output->getType() == "surfaceshader") {
                    renderable = output;
                    break;
                }
            }
        }
        if (!renderable) {
            result.error = "No renderable elements found in MaterialX document";
            return result;
        }

        // Generate shader
        mx::ShaderPtr shader = g_shader_gen->generate(
            renderable->getName(), renderable, *g_gen_context);

        if (!shader) {
            result.error = "MaterialX shader generation failed";
            return result;
        }

        result.vertex   = shader->getSourceCode(mx::Stage::VERTEX);
        result.fragment = shader->getSourceCode(mx::Stage::PIXEL);
        result.ok = true;

    } catch (const std::exception& e) {
        result.error = std::string("MaterialX codegen exception: ") + e.what();
    }

    return result;
}

/* ---- Texture loading ---- */

/* Filename → absolute-path cache, built on first fallback miss.
 * Keyed by scene_dir so multiple scenes don't poison each other. */
static std::unordered_map<std::string,
    std::unordered_map<std::string, std::string>> g_filename_index_cache;

static const std::unordered_map<std::string, std::string>&
get_filename_index(const std::string& scene_dir)
{
    auto it = g_filename_index_cache.find(scene_dir);
    if (it != g_filename_index_cache.end()) return it->second;

    std::unordered_map<std::string, std::string> index;
    try {
        fs::path root(scene_dir);
        if (fs::exists(root) && fs::is_directory(root)) {
            for (auto it2 = fs::recursive_directory_iterator(root,
                     fs::directory_options::skip_permission_denied);
                 it2 != fs::recursive_directory_iterator(); ++it2) {
                if (it2->is_directory() && it2->path().filename() == ".thumbs") {
                    it2.disable_recursion_pending();
                    continue;
                }
                if (it2.depth() > 16) { it2.disable_recursion_pending(); continue; }
                if (!it2->is_regular_file()) continue;
                std::string fn = it2->path().filename().string();
                /* Keep first hit — caller can always pass a more specific path. */
                index.emplace(std::move(fn), it2->path().string());
            }
        }
    } catch (...) {}
    auto& ref = g_filename_index_cache[scene_dir];
    ref = std::move(index);
    return ref;
}

static void push_unique_path(std::vector<fs::path>& out, const fs::path& p)
{
    try {
        if (p.empty() || !fs::exists(p) || !fs::is_directory(p)) return;
        fs::path c = fs::weakly_canonical(p);
        for (const fs::path& q : out)
            if (q == c) return;
        out.push_back(c);
    } catch (...) {}
}

static bool env_enabled(const char* name)
{
    const char* e = getenv(name);
    return e && e[0] && e[0] != '0' &&
           strcmp(e, "false") && strcmp(e, "False") &&
           strcmp(e, "off") && strcmp(e, "OFF") &&
           strcmp(e, "no") && strcmp(e, "NO");
}

static bool is_http_url(const std::string& s)
{
    return s.rfind("https://", 0) == 0 || s.rfind("http://", 0) == 0;
}

static bool remote_asset_fetch_allowed(const std::string& url)
{
    if (env_enabled("NUSD_DISABLE_REMOTE_ASSET_FETCH"))
        return false;
    if (env_enabled("NUSD_ENABLE_REMOTE_ASSET_FETCH"))
        return true;

    /* Opt-in only: do not trigger network requests on scene load by default,
     * not even the GB300 paint preset URL (previously auto-fetched to match
     * OVRTX's default resolver). Warn once so the opt-in is discoverable. */
    static bool warned = false;
    if (!warned && url.rfind("http", 0) == 0) {
        warned = true;
        fprintf(stderr,
                "[nusd] remote asset fetch is opt-in; set "
                "NUSD_ENABLE_REMOTE_ASSET_FETCH=1 to download remote textures "
                "(e.g. DSX GB300 paint). Skipping remote assets this run.\n");
    }
    return false;
}

static fs::path remote_asset_cache_root()
{
    const char* env = getenv("NUSD_REMOTE_ASSET_CACHE");
    if (env && env[0]) return fs::path(env);

    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0])
        return fs::path(xdg) / "nusd_renderer" / "remote_assets";

    const char* home = getenv("HOME");
    if (home && home[0])
        return fs::path(home) / ".cache" / "nusd_renderer" / "remote_assets";

    return fs::path("/tmp/nusd_renderer_remote_assets");
}

static std::string sanitize_cache_segment(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '.' || c == '_' || c == '-')
            out.push_back((char)c);
        else
            out.push_back('_');
    }
    if (out.empty() || out == "." || out == "..") out = "_";
    return out;
}

static fs::path remote_asset_cache_path(const std::string& url)
{
    size_t scheme = url.find("://");
    std::string rest = (scheme == std::string::npos) ? url : url.substr(scheme + 3);
    size_t query = rest.find_first_of("?#");
    if (query != std::string::npos) rest = rest.substr(0, query);

    fs::path rel;
    size_t pos = 0;
    while (pos <= rest.size()) {
        size_t end = rest.find('/', pos);
        std::string seg = rest.substr(pos, end == std::string::npos
                                           ? std::string::npos
                                           : end - pos);
        if (!seg.empty())
            rel /= sanitize_cache_segment(seg);
        if (end == std::string::npos) break;
        pos = end + 1;
    }
    return remote_asset_cache_root() / rel;
}

static std::string shell_quote(const std::string& s)
{
    std::string out("'");
    for (char c : s) {
        if (c == '\'')
            out += "'\\''";
        else
            out.push_back(c);
    }
    out.push_back('\'');
    return out;
}

static std::string resolve_remote_asset_to_cache(const std::string& url)
{
    fs::path cached = remote_asset_cache_path(url);
    try {
        if (fs::exists(cached) && fs::file_size(cached) > 0)
            return cached.string();
    } catch (...) {}

    if (!remote_asset_fetch_allowed(url))
        return std::string();

    std::error_code ec;
    fs::create_directories(cached.parent_path(), ec);
    if (ec) return std::string();

    fs::path tmp = cached;
    tmp += ".tmp";
    fs::remove(tmp, ec);

    std::string qtmp = shell_quote(tmp.string());
    std::string qurl = shell_quote(url);
    std::string cmd =
        "(command -v curl >/dev/null 2>&1 && "
        "curl --fail --silent --show-error --connect-timeout 10 --max-time 60 "
        "--max-filesize 8388608 "
        "-o " + qtmp + " " + qurl + ") || "
        "(command -v wget >/dev/null 2>&1 && "
        "wget -q -T 30 --max-redirect=0 -Q 8m -O " + qtmp + " " + qurl + ")";

    int rc = std::system(cmd.c_str());
    try {
        if (rc == 0 && fs::exists(tmp) && fs::file_size(tmp) > 0) {
            fs::rename(tmp, cached, ec);
            if (ec) {
                fs::remove(cached, ec);
                ec.clear();
                fs::rename(tmp, cached, ec);
            }
            if (!ec && fs::exists(cached)) {
                if (getenv("NUSD_MAT_DIAG"))
                    fprintf(stderr, "material: cached remote asset: %s -> %s\n",
                            url.c_str(), cached.string().c_str());
                return cached.string();
            }
        }
    } catch (...) {}

    fs::remove(tmp, ec);
    return std::string();
}

static void collect_env_asset_search_roots(std::vector<fs::path>& roots)
{
    const char* env = getenv("NUSD_ASSET_SEARCH_PATH");
    if (!env || !env[0]) return;

    std::string s(env);
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find(':', pos);
        std::string item = s.substr(pos, end == std::string::npos
                                           ? std::string::npos
                                           : end - pos);
        if (!item.empty())
            push_unique_path(roots, fs::path(item));
        if (end == std::string::npos) break;
        pos = end + 1;
    }
}

static void collect_materialx_resource_roots(std::vector<fs::path>& roots)
{
    auto add_from_search_entry = [&](const fs::path& entry) {
        if (entry.empty()) return;
        push_unique_path(roots, entry);
        push_unique_path(roots, entry / "resources");
        fs::path parent = entry.parent_path();
        if (!parent.empty()) {
            push_unique_path(roots, parent);
            push_unique_path(roots, parent / "resources");
        }
    };

    auto add_path_list = [&](const char* list) {
        if (!list || !list[0]) return;
        std::string s(list);
        size_t pos = 0;
        while (pos <= s.size()) {
            size_t end = s.find(':', pos);
            std::string item = s.substr(pos, end == std::string::npos
                                               ? std::string::npos
                                               : end - pos);
            if (!item.empty()) add_from_search_entry(fs::path(item));
            if (end == std::string::npos) break;
            pos = end + 1;
        }
    };

    add_path_list(getenv("MATERIALX_SEARCH_PATH"));
#ifdef MATERIALX_SEARCH_PATH
    add_path_list(MATERIALX_SEARCH_PATH);
#endif
}

static std::vector<fs::path> collect_texture_search_roots(const char* scene_dir)
{
    std::vector<fs::path> roots;
    collect_env_asset_search_roots(roots);

    if (scene_dir && scene_dir[0]) {
        const bool disable_dsx_rescue =
            env_enabled("NUSD_DISABLE_DSX_ASSET_RESCUE");
        fs::path p(scene_dir);
        for (int up = 0; up <= 4 && !p.empty(); up++, p = p.parent_path()) {
            push_unique_path(roots, p);
            push_unique_path(roots, p / "Library");
            if (!disable_dsx_rescue)
                push_unique_path(roots, p / "DSX_BP" / "Library");
        }
    }

    collect_materialx_resource_roots(roots);

    /* Drop roots that don't exist as directories. A non-existent root can never
     * contribute a resolved file: resolve_via_search_roots probes
     * fs::exists(root/suffix) (always false) and get_filename_index already
     * returns an empty index for it (it guards on fs::exists && is_directory).
     * Filtering is therefore result-identical, and it removes the per-miss
     * fs::exists storm + redundant empty-index builds over the many
     * layer/Materials and layer/Textures dirs that ALab-scale scenes (1099
     * layers) generate but that mostly don't exist. */
    {
        std::vector<fs::path> existing;
        existing.reserve(roots.size());
        for (auto& r : roots) {
            std::error_code ec;
            if (fs::is_directory(r, ec)) existing.push_back(std::move(r));
        }
        roots.swap(existing);
    }

    return roots;
}

static std::vector<std::string> candidate_asset_suffixes(const char* path)
{
    std::vector<std::string> suffixes;
    if (!path || !path[0]) return suffixes;

    auto push_suffix = [&](std::string suffix) {
        if (!suffix.empty() &&
            std::find(suffixes.begin(), suffixes.end(), suffix) == suffixes.end()) {
            suffixes.push_back(std::move(suffix));
        }
    };

    std::string s(path);
    if (s.rfind("./", 0) == 0) s = s.substr(2);
    push_suffix(s);

    std::string stripped = s;
    while (stripped.rfind("../", 0) == 0) stripped = stripped.substr(3);
    push_suffix(stripped);

    const char* materialx_example_roots[] = {
        "Materials/Examples/StandardSurface/",
        "Materials/Examples/OpenPbr/",
        "Materials/Examples/UsdPreviewSurface/",
        "Materials/Examples/GltfPbr/",
        "Materials/Examples/DisneyPrincipled/",
        "Materials/Examples/SimpleHair/",
        "Images/",
    };
    if (!stripped.empty() && !fs::path(stripped).is_absolute()) {
        for (const char* root : materialx_example_roots)
            push_suffix(std::string(root) + stripped);
    }

    const char* markers[] = {
        "/Library/Materials/",
        "/resources/Materials/",
        "/Materials/",
        "/resources/Images/",
        "/textures/",
        "/Textures/",
        "/omniverse-content-production.s3.us-west-2.amazonaws.com/",
    };
    for (const char* marker : markers) {
        size_t pos = s.find(marker);
        if (pos == std::string::npos) continue;
        std::string rel = s.substr(pos + 1);
        push_suffix(std::move(rel));
    }

    return suffixes;
}

static std::string resolve_via_search_roots(const char* path, const char* scene_dir)
{
    std::vector<fs::path> roots = collect_texture_search_roots(scene_dir);
    std::vector<std::string> suffixes = candidate_asset_suffixes(path);

    for (const fs::path& root : roots) {
        for (const std::string& suffix : suffixes) {
            fs::path rel(suffix);
            if (rel.is_absolute()) continue;
            try {
                fs::path candidate = root / rel;
                if (fs::exists(candidate))
                    return fs::weakly_canonical(candidate).string();
            } catch (...) {}
        }
    }

    return std::string();
}

static bool is_usd_layer_path(const fs::path& p)
{
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return ext == ".usd" || ext == ".usda" || ext == ".usdc";
}

static void push_unique_dir_string(std::vector<std::string>& out, const fs::path& p)
{
    try {
        if (p.empty() || !fs::exists(p) || !fs::is_directory(p)) return;
        std::string c = fs::weakly_canonical(p).string();
        if (std::find(out.begin(), out.end(), c) == out.end())
            out.push_back(std::move(c));
    } catch (...) {}
}

static void collect_layer_asset_dirs_recursive(const fs::path& layer_path,
                                               int depth,
                                               std::vector<std::string>& dirs,
                                               std::unordered_set<std::string>& seen)
{
    if (depth > 6 || layer_path.empty()) return;

    fs::path layer;
    try {
        layer = fs::weakly_canonical(layer_path);
        if (!fs::exists(layer) || !fs::is_regular_file(layer)) return;
    } catch (...) {
        return;
    }

    std::string key = layer.string();
    if (!seen.insert(key).second) return;

    fs::path layer_dir = layer.parent_path();
    push_unique_dir_string(dirs, layer_dir);
    if (depth >= 6 || !is_usd_layer_path(layer)) return;

    try {
        if (fs::file_size(layer) > 32ull * 1024ull * 1024ull)
            return;
    } catch (...) {}

    std::ifstream f(layer, std::ios::binary);
    if (!f) return;

    std::string line;
    while (std::getline(f, line)) {
        size_t pos = 0;
        while ((pos = line.find('@', pos)) != std::string::npos) {
            size_t end = line.find('@', pos + 1);
            if (end == std::string::npos) break;
            std::string asset = line.substr(pos + 1, end - pos - 1);
            pos = end + 1;

            auto angle = asset.find('<');
            if (angle != std::string::npos) asset = asset.substr(0, angle);
            while (!asset.empty() && std::isspace((unsigned char)asset.front()))
                asset.erase(asset.begin());
            while (!asset.empty() && std::isspace((unsigned char)asset.back()))
                asset.pop_back();
            if (asset.empty() || asset.rfind("anon:", 0) == 0 ||
                asset.find("://") != std::string::npos)
                continue;

            fs::path asset_path(asset);
            if (asset_path.is_relative())
                asset_path = layer_dir / asset_path;
            try {
                asset_path = fs::weakly_canonical(asset_path);
                if (!fs::exists(asset_path)) continue;
                push_unique_dir_string(dirs, asset_path.parent_path());
                if (is_usd_layer_path(asset_path))
                    collect_layer_asset_dirs_recursive(asset_path, depth + 1,
                                                       dirs, seen);
            } catch (...) {}
        }
    }
}

/* Collect candidate anchor directories for relative texture paths.
 *
 * USD resolves asset paths against the layer that authored the value. The
 * public nanousd C API does not yet expose per-attribute layer-of-origin, so
 * this renderer builds a conservative set of filesystem anchors by walking
 * the root layer's asset arcs (`subLayers`, `references`, `payloads`, and
 * texture/MDL assets expressed as @...@ tokens in text layers). That gives
 * sublayered probe scenes and generated SimReady wrappers the same asset bundle
 * roots OVRTX sees through full USD composition. */
static std::vector<std::string>
collect_root_layer_reference_dirs(NanousdStage stage)
{
    std::vector<std::string> dirs;
    if (!stage) return dirs;

    /* Cache keyed by root-layer path, NOT the NanousdStage pointer.
     * Stage handles are recycled across scenes (each nu_load_usd creates
     * and later frees one) so a pointer-keyed cache leaked the prior
     * scene's reference dirs into the next scene — caught by the
     * correctness test suite running material_tray after chess. */
    static std::unordered_map<std::string, std::vector<std::string>> cache;

    const char* root_path = nanousd_stage_get_root_layer_path(stage);
    if (!root_path || !root_path[0]) return dirs;

    auto it = cache.find(root_path);
    if (it != cache.end()) return it->second;

    std::unordered_set<std::string> seen;
    collect_layer_asset_dirs_recursive(fs::path(root_path), 0, dirs, seen);
    cache[root_path] = dirs;
    if (!dirs.empty()) {
        fprintf(stderr, "material: %zu asset anchor dir(s) collected from %s\n",
                dirs.size(), root_path);
    }
    return dirs;
}

static std::string resolve_texture_path_impl(const char* path, const char* scene_dir,
                                             NanousdStage stage)
{
    NuResolveTimer _rt;  /* load-time instrumentation (actual resolve/stat cost) */
    if (!path || !path[0]) return std::string();

    std::string authored = path ? path : "";
    if (is_http_url(authored)) {
        std::string cached = resolve_remote_asset_to_cache(authored);
        if (!cached.empty()) return cached;
        return authored;
    }

    if (stage) {
        std::string packaged = resolve_stage_package_asset(path, stage);
        if (!packaged.empty()) return packaged;
    }

    std::string resolved;
    if (path[0] == '/' || path[0] == '\\') {
        resolved = path;
    } else {
        resolved = std::string(scene_dir) + "/" + path;
    }
    try {
        resolved = fs::weakly_canonical(resolved).string();
    } catch (...) {}

    /* If file exists at the resolved path, we're done */
    if (fs::exists(resolved))
        return resolved;

    /* Absent-basename short-circuit. Every fallback below ultimately looks for a
     * file whose basename is `filename`, reachable at some shallow path under one
     * of the search roots or root-layer reference dirs. get_filename_index records
     * every regular file under a root (recursively, depth <= 16), so if `filename`
     * is absent from the index of EVERY root we consult, no fs::exists(root/suffix)
     * probe, anchor probe, basename lookup, or sibling-dir scan below can hit, and
     * returning the unresolved path here is result-identical. This collapses the
     * dominant cost on ALab-scale scenes (1099 layers -> thousands of roots; absent
     * 4k-texture refs each otherwise drive a roots x suffixes fs::exists storm
     * across Fallback -1/0/2) to a set of cached index lookups.
     *
     * The index union MUST cover every source the fallbacks read, which on Vulkan
     * is richer than Metal's: collect_root_layer_reference_dirs (Fallback 0 anchors
     * + Fallback 1's first loop) AND collect_texture_search_roots (Fallback -1,
     * Fallback 1's second loop, and — because search roots include scene_dir's
     * ancestors up to 4 levels — the Fallback 2 sibling scan, which only walks
     * subdirs of those same ancestors). Both index families are the same ones the
     * fallbacks build, so no extra tree walks are introduced; lookups short-circuit
     * on the first root that has the basename, so resolvable paths fall through. */
    {
        std::string filename = fs::path(path).filename().string();
        if (!filename.empty()) {
            bool anywhere = false;
            if (stage) {
                for (const std::string& root :
                     collect_root_layer_reference_dirs(stage)) {
                    const auto& index = get_filename_index(root);
                    if (index.find(filename) != index.end()) { anywhere = true; break; }
                }
            }
            if (!anywhere) {
                for (const fs::path& root : collect_texture_search_roots(scene_dir)) {
                    const auto& index = get_filename_index(root.string());
                    if (index.find(filename) != index.end()) { anywhere = true; break; }
                }
            }
            if (!anywhere) return resolved;
        }
    }

    /* Fallback -1: DSX and collected Omniverse bundles sometimes carry stale
     * absolute asset paths from the authoring machine. Preserve the authored
     * suffix (e.g. Materials/Base/...) but try it under the current scene's
     * package roots before falling back to expensive filename search. */
    {
        std::string rooted = resolve_via_search_roots(path, scene_dir);
        if (!rooted.empty()) return rooted;
    }

    /* Fallback 0: try anchor dirs harvested from the root layer's
     * `references` and `payload` directives. SimReady wrappers (agibot,
     * etc.) author absolute reference paths to bundled assets that have
     * their own ./textures/ directories — those need to be searched
     * before falling through to filename-index search. */
    if (stage) {
        std::vector<std::string> anchors =
            collect_root_layer_reference_dirs(stage);
        std::string rel_path = path;
        if (rel_path.substr(0, 2) == "./") rel_path = rel_path.substr(2);
        std::vector<std::string> suffixes = candidate_asset_suffixes(path);
        std::string filename = fs::path(path).filename().string();
        for (const auto& anchor : anchors) {
            for (const std::string& suffix : suffixes) {
                fs::path rel(suffix);
                if (rel.is_absolute()) continue;
                try {
                    fs::path candidate = fs::path(anchor) / rel;
                    if (fs::exists(candidate))
                        return fs::weakly_canonical(candidate).string();
                } catch (...) {}
            }
            if (!filename.empty()) {
                const fs::path base_anchor(anchor);
                const fs::path probes[] = {
                    base_anchor / filename,
                    base_anchor / "Textures" / filename,
                    base_anchor / "textures" / filename,
                    base_anchor / "Materials" / filename,
                    base_anchor / "materials" / filename,
                };
                for (const fs::path& candidate : probes) {
                    try {
                        if (fs::exists(candidate))
                            return fs::weakly_canonical(candidate).string();
                    } catch (...) {}
                }
            }
        }
    }

    /* Fallback 1: recursive filename index under scene_dir and likely asset
     * roots. Sublayered scenes reference textures relative to the sublayer
     * directory, and DSX package roots put Assembly beside Library; indexing
     * only Assembly misses all shared material textures.
     * Sublayered scenes reference textures relative to the sublayer directory,
     * which nanousd does not currently resolve. Indexing the scene tree
     * by filename handles any depth. */
    try {
        std::string filename = fs::path(path).filename().string();
        if (!filename.empty()) {
            if (stage) {
                for (const std::string& root : collect_root_layer_reference_dirs(stage)) {
                    const auto& index = get_filename_index(root);
                    auto it = index.find(filename);
                    if (it != index.end() && fs::exists(it->second))
                        return it->second;
                }
            }
            for (const fs::path& root : collect_texture_search_roots(scene_dir)) {
                const auto& index = get_filename_index(root.string());
                auto it = index.find(filename);
                if (it != index.end() && fs::exists(it->second))
                    return it->second;
            }
        }
    } catch (...) {}

    /* Fallback 2: legacy sibling-directory scan (kept for small scenes). */
    std::string rel_path = path;
    if (rel_path.substr(0, 2) == "./") rel_path = rel_path.substr(2);
    try {
        fs::path scene = fs::path(scene_dir);
        for (int up = 1; up <= 3; up++) {
            fs::path parent = scene;
            for (int i = 0; i < up; i++) parent = parent.parent_path();
            if (parent.empty() || !fs::exists(parent)) break;
            for (auto& entry : fs::directory_iterator(parent,
                     fs::directory_options::skip_permission_denied)) {
                if (!entry.is_directory()) continue;
                fs::path candidate = entry.path() / rel_path;
                if (fs::exists(candidate))
                    return fs::weakly_canonical(candidate).string();
            }
        }
    } catch (...) {}

    return resolved;
}

/* Memoizing wrapper around resolve_texture_path_impl. Within one load the
 * resolution of an authored path is deterministic (stage is parsed, not
 * mutated), so caching by (path, scene_dir, stage) is bit-identical and
 * collapses the ~14.6k stat-heavy resolutions to the unique set. The cache is
 * cleared at the top of materials_load (per-load, never stale across scenes). */
static std::string resolve_texture_path(const char* path, const char* scene_dir,
                                        NanousdStage stage = nullptr)
{
    if (!path || !path[0]) return std::string();
    std::string key;
    key.reserve(strlen(path) + (scene_dir ? strlen(scene_dir) : 0) + 32);
    key.append(path).push_back('\x1f');
    if (scene_dir) key.append(scene_dir);
    key.push_back('\x1f');
    key.append(std::to_string((uintptr_t)stage));
    auto it = g_resolve_cache.find(key);
    if (it != g_resolve_cache.end()) {
        ++g_tex_resolve_hits;
        return it->second;
    }
    std::string r = resolve_texture_path_impl(path, scene_dir, stage);
    g_resolve_cache.emplace(std::move(key), r);
    return r;
}

/*
 * UDIM atlas stitcher: replaces <UDIM> with tile numbers (1001..),
 * loads all tiles, and composites into one texture.
 *
 * UDIM layout: tile 1001 + col*1 + row*10
 *   row 0: 1001, 1002, 1003, ...
 *   row 1: 1011, 1012, 1013, ...
 */
static std::string canonical_udim_pattern_key(const std::string& pattern,
                                              const char* scene_dir,
                                              NanousdStage stage = nullptr)
{
    size_t udim_pos = pattern.find("<UDIM>");
    if (udim_pos == std::string::npos)
        return std::string();

    std::string test_path = pattern;
    test_path.replace(udim_pos, 6, "1001");
    std::string resolved_test = resolve_texture_path(test_path.c_str(),
                                                     scene_dir, stage);
    if (fs::exists(resolved_test)) {
        std::string dir = fs::path(resolved_test).parent_path().string();
        std::string filename = fs::path(pattern).filename().string();
        std::string key = dir + "/" + filename;
        try { key = fs::weakly_canonical(fs::path(key).parent_path()).string() +
                    "/" + filename; }
        catch (...) {}
        return key;
    }

    if (!pattern.empty() && (pattern[0] == '/' || pattern[0] == '\\'))
        return pattern;
    if (scene_dir && *scene_dir)
        return std::string(scene_dir) + "/" + pattern;
    return pattern;
}

static MaterialTexture load_udim_atlas(const std::string& pattern,
                                       const char* scene_dir,
                                       NanousdStage stage = nullptr)
{
    MaterialTexture tex = {};

    std::string base = canonical_udim_pattern_key(pattern, scene_dir, stage);

    size_t udim_pos = base.find("<UDIM>");
    if (udim_pos == std::string::npos) return tex;

    std::string prefix = base.substr(0, udim_pos);
    std::string suffix = base.substr(udim_pos + 6); // strlen("<UDIM>")

    // Scan for existing tiles: UDIM range 1001-1100 (10 rows x 10 cols)
    struct UdimTileCandidate {
        int col, row;
        int tile_id;
        std::string path;
    };
    struct UdimTile {
        int col, row;
        int tile_id;
        unsigned char* pixels;
        int w, h;
    };
    std::vector<UdimTileCandidate> candidates;
    std::vector<UdimTile> tiles;
    int max_col = 0, max_row = 0;

    for (int row = 0; row < 10; row++) {
        for (int col = 0; col < 10; col++) {
            int tile_id = 1001 + col + row * 10;
            std::string tile_path = prefix + std::to_string(tile_id) + suffix;

            std::error_code exists_ec;
            if (!fs::exists(tile_path, exists_ec) || exists_ec)
                continue;
            try { tile_path = fs::weakly_canonical(tile_path).string(); }
            catch (...) {}
            candidates.push_back({col, row, tile_id, tile_path});
        }
    }

    if (!candidates.empty()) {
        std::vector<UdimTile> decoded(candidates.size());
        std::vector<double> decode_ms(candidates.size(), 0.0);
        std::atomic<size_t> next{0};
        unsigned hw = std::thread::hardware_concurrency();
        unsigned nthreads = hw ? std::min(hw, 8u) : 4u;
        if ((size_t)nthreads > candidates.size())
            nthreads = (unsigned)candidates.size();
        auto worker = [&]() {
            for (;;) {
                size_t idx = next.fetch_add(1);
                if (idx >= candidates.size()) break;
                const UdimTileCandidate& cand = candidates[idx];
                int w = 0, h = 0, ch = 0;
                double _dt0 = nu_load_now_ms();
                unsigned char* px = stbi_load(cand.path.c_str(), &w, &h, &ch, 4);
                decode_ms[idx] = nu_load_now_ms() - _dt0;
                if (px)
                    decoded[idx] = {cand.col, cand.row, cand.tile_id, px, w, h};
            }
        };
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < nthreads; ++t)
            pool.emplace_back(worker);
        for (auto& th : pool)
            th.join();

        for (size_t i = 0; i < decoded.size(); ++i) {
            g_tex_decode_ms += decode_ms[i];
            ++g_tex_decode_n;
            if (!decoded[i].pixels) continue;
            tiles.push_back(decoded[i]);
            if (decoded[i].col > max_col) max_col = decoded[i].col;
            if (decoded[i].row > max_row) max_row = decoded[i].row;
        }
    }

    if (tiles.empty()) {
        fprintf(stderr, "material: UDIM no tiles found for %s\n", pattern.c_str());
        return tex;
    }

    // All tiles should be the same size; use first tile's dimensions.
    // Multi-tile 8K UDIM sets can require several GiB as a flattened atlas;
    // cap the per-tile atlas resolution so large production assets stay
    // renderable while preserving every tile and the authored UDIM layout.
    int tile_w = tiles[0].w;
    int tile_h = tiles[0].h;
    int cols = max_col + 1;
    int rows = max_row + 1;
    int dst_tile_w = tile_w;
    int dst_tile_h = tile_h;
    int max_tile_dim = 1024;
    if (const char* env = getenv("NUSD_UDIM_MAX_TILE_DIM")) {
        int v = atoi(env);
        if (v >= 64) max_tile_dim = v;
    }
    if (tiles.size() > 1 &&
        (dst_tile_w > max_tile_dim || dst_tile_h > max_tile_dim)) {
        float sx = (float)max_tile_dim / (float)dst_tile_w;
        float sy = (float)max_tile_dim / (float)dst_tile_h;
        float s = std::min(sx, sy);
        dst_tile_w = std::max(1, (int)(dst_tile_w * s + 0.5f));
        dst_tile_h = std::max(1, (int)(dst_tile_h * s + 0.5f));
    }
    int atlas_w = dst_tile_w * cols;
    int atlas_h = dst_tile_h * rows;

    size_t atlas_bytes = (size_t)atlas_w * (size_t)atlas_h * 4u;
    const size_t max_atlas_bytes = 512ull * 1024ull * 1024ull;
    if (atlas_bytes > max_atlas_bytes) {
        float s = sqrtf((float)max_atlas_bytes / (float)atlas_bytes);
        dst_tile_w = std::max(1, (int)(dst_tile_w * s));
        dst_tile_h = std::max(1, (int)(dst_tile_h * s));
        atlas_w = dst_tile_w * cols;
        atlas_h = dst_tile_h * rows;
        atlas_bytes = (size_t)atlas_w * (size_t)atlas_h * 4u;
    }

    tex.pixels = (unsigned char*)calloc((size_t)atlas_w * (size_t)atlas_h, 4);
    if (!tex.pixels) {
        for (auto& t : tiles) stbi_image_free(t.pixels);
        return tex;
    }
    tex.width = atlas_w;
    tex.height = atlas_h;
    tex.udim_cols = cols;
    tex.udim_rows = rows;
    std::vector<unsigned char> present((size_t)cols * (size_t)rows, 0);

    // Blit tiles into atlas. The shaders sample the flattened UDIM with
    // tc = authored_uv / vec2(cols, rows), without an extra V inversion,
    // so row 0 must occupy the first atlas row.
    for (auto& t : tiles) {
        int dst_x = t.col * dst_tile_w;
        int dst_y = t.row * dst_tile_h;

        if (dst_tile_w == t.w && dst_tile_h == t.h) {
            for (int y = 0; y < dst_tile_h; y++) {
                memcpy(tex.pixels + ((dst_y + y) * atlas_w + dst_x) * 4,
                       t.pixels + (y * t.w) * 4,
                       (size_t)dst_tile_w * 4u);
            }
        } else {
            for (int y = 0; y < dst_tile_h; y++) {
                int sy = std::min(t.h - 1, (int)((int64_t)y * t.h / dst_tile_h));
                for (int x = 0; x < dst_tile_w; x++) {
                    int sx = std::min(t.w - 1, (int)((int64_t)x * t.w / dst_tile_w));
                    memcpy(tex.pixels + ((dst_y + y) * atlas_w + dst_x + x) * 4,
                           t.pixels + (sy * t.w + sx) * 4,
                           4);
                }
            }
        }
        present[(size_t)t.row * (size_t)cols + (size_t)t.col] = 1;
        stbi_image_free(t.pixels);
    }

    std::string lower_pattern = pattern;
    std::transform(lower_pattern.begin(), lower_pattern.end(),
                   lower_pattern.begin(), ::tolower);
    const bool opacity_udim = (lower_pattern.find("opacity") != std::string::npos ||
                               lower_pattern.find("alpha") != std::string::npos);

    /* Sparse color/normal UDIM sets should not create transparent black atlas
     * cells. Renderers generally assume missing UDIM files are unsampled
     * authoring gaps; if coordinates do land there, edge extension is a better
     * approximation than black. Opacity is the exception: missing opacity tiles
     * are an authored transparent region in DSX MountainNevada, and copying a
     * neighboring white tile makes large invisible cutout sheets render solid. */
    int filled_missing = 0;
    if (!opacity_udim) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                size_t dst_cell = (size_t)row * (size_t)cols + (size_t)col;
                if (present[dst_cell]) continue;

                int best_col = -1;
                int best_row = -1;
                int best_dist = 1 << 30;
                for (int sr = 0; sr < rows; sr++) {
                    for (int sc = 0; sc < cols; sc++) {
                        if (!present[(size_t)sr * (size_t)cols + (size_t)sc])
                            continue;
                        int dist = abs(sc - col) + abs(sr - row);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_col = sc;
                            best_row = sr;
                        }
                    }
                }
                if (best_col < 0) continue;

                int dst_x = col * dst_tile_w;
                int dst_y = row * dst_tile_h;
                int src_x = best_col * dst_tile_w;
                int src_y = best_row * dst_tile_h;
                for (int y = 0; y < dst_tile_h; y++) {
                    memcpy(tex.pixels + ((dst_y + y) * atlas_w + dst_x) * 4,
                           tex.pixels + ((src_y + y) * atlas_w + src_x) * 4,
                           (size_t)dst_tile_w * 4u);
                }
                filled_missing++;
            }
        }
    }

    snprintf(tex.path, sizeof(tex.path), "%s", pattern.c_str());
    fprintf(stderr,
            "material: UDIM atlas %dx%d (%d tiles, %dx%d grid, tile %dx%d from %dx%d, filled %d missing) from %s\n",
            atlas_w, atlas_h, (int)tiles.size(), cols, rows,
            dst_tile_w, dst_tile_h, tile_w, tile_h, filled_missing,
            pattern.c_str());

    return tex;
}

/* Decode-only texture loader for the parallel pre-pass: takes an already
 * RESOLVED path (no resolve_texture_path -> no g_resolve_cache access), so it
 * is safe to call from worker threads. stb decode itself is reentrant. */
static MaterialTexture decode_texture_file(const std::string& resolved)
{
    MaterialTexture tex = {};
    int channels;
    tex.pixels = stbi_load(resolved.c_str(), &tex.width, &tex.height, &channels, 4);
    if (tex.pixels)
        snprintf(tex.path, sizeof(tex.path), "%s", resolved.c_str());
    return tex;
}

static MaterialTexture load_texture(const char* path, const char* scene_dir,
                                     NanousdStage stage = nullptr)
{
    MaterialTexture tex = {};
    std::string path_str(path);

    // UDIM support: detect <UDIM> placeholder
    if (path_str.find("<UDIM>") != std::string::npos) {
        return load_udim_atlas(path_str, scene_dir, stage);
    }

    std::string package_location = resolve_stage_package_asset(path, stage);
    if (!package_location.empty()) {
        unsigned char* data = nullptr;
        size_t sz = 0;
        if (nanousd_read_asset_bytes(package_location.c_str(), &data, &sz) &&
            data && sz > 0) {
            int channels;
            double _dt0 = nu_load_now_ms();
            tex.pixels = stbi_load_from_memory(data, (int)sz,
                                                &tex.width, &tex.height,
                                                &channels, 4);
            g_tex_decode_ms += nu_load_now_ms() - _dt0;
            ++g_tex_decode_n;
            nanousd_free_bytes(data);
            if (tex.pixels) {
                snprintf(tex.path, sizeof(tex.path), "%s", package_location.c_str());
                fprintf(stderr, "material: loaded package texture %dx%d: %s\n",
                        tex.width, tex.height, package_location.c_str());
                return tex;
            }
        } else if (data) {
            nanousd_free_bytes(data);
        }
    }

    std::string resolved = resolve_texture_path(path, scene_dir, stage);
    snprintf(tex.path, sizeof(tex.path), "%s", resolved.c_str());

    /* Parallel pre-decode hit: this texture was decoded up front (see the
     * pre-pass in materials_load). Take ownership of the blob and skip stb. */
    if (!g_predecoded.empty()) {
        auto it = g_predecoded.find(resolved);
        if (it != g_predecoded.end()) {
            MaterialTexture pre = it->second;  /* POD copy: carries the pixel ptr */
            g_predecoded.erase(it);            /* erase does not free (POD) */
            if (pre.pixels) {
                snprintf(pre.path, sizeof(pre.path), "%s", resolved.c_str());
                ++g_tex_predecode_n;
                return pre;
            }
        }
    }

    int channels;
    double _dt0 = nu_load_now_ms();
    tex.pixels = stbi_load(resolved.c_str(), &tex.width, &tex.height,
                           &channels, 4 /* force RGBA */);
    g_tex_decode_ms += nu_load_now_ms() - _dt0;
    ++g_tex_decode_n;

    if (!tex.pixels) {
        fprintf(stderr, "material: failed to load texture: %s\n", resolved.c_str());
    } else {
        fprintf(stderr, "material: loaded texture %dx%d: %s\n",
                tex.width, tex.height, resolved.c_str());
    }

    return tex;
}

/* Deduplicated texture loading: returns index into tex_paths/textures arrays.
 * Returns -1 if texture could not be loaded. */
static int load_texture_dedup(const char* file, const char* scene_dir,
                              std::vector<std::string>& tex_paths,
                              std::vector<MaterialTexture>& textures,
                              NanousdStage stage = nullptr,
                              NanousdPrim /* shader_prim */ = nullptr)
{
    /* Resolve to canonical path for dedup comparison. The stage handle
     * lets resolve_texture_path try anchor dirs harvested from
     * `references` / `payload` directives in the root layer — needed
     * for SimReady wrappers (e.g. agibot) where the texture lives
     * relative to the referenced asset's directory. */
    std::string file_str(file ? file : "");
    bool is_udim = file_str.find("<UDIM>") != std::string::npos;
    std::string resolved = is_udim
        ? canonical_udim_pattern_key(file_str, scene_dir, stage)
        : resolve_texture_path(file, scene_dir, stage);
    if (!resolved.empty() && g_failed_texture_cache.count(resolved))
        return -1;

    /* Check if already loaded */
    for (int i = 0; i < (int)tex_paths.size(); i++) {
        if (tex_paths[i] == resolved || tex_paths[i] == file) {
            return i;
        }
    }

    /* Load new texture; load_texture also calls resolve_texture_path
     * (with the same stage) for its own resolution, so passing `file`
     * is enough for regular images. UDIM patterns are already canonicalized
     * above so equivalent relative references share the same atlas key. */
    MaterialTexture tex = load_texture(
        is_udim && !resolved.empty() ? resolved.c_str() : file,
        scene_dir, stage);
    if (!tex.pixels) {
        if (!resolved.empty())
            g_failed_texture_cache.insert(resolved);
        return -1;
    }

    int idx = (int)textures.size();
    textures.push_back(tex);
    tex_paths.push_back(resolved);
    return idx;
}

static void propagate_udim_scale(MaterialParams* params,
                                 const MaterialTexture& tex);

static void bind_metal_black_paint_remote_textures(
    MaterialParams* params,
    std::vector<std::string>& tex_paths,
    std::vector<MaterialTexture>& textures,
    const char* scene_dir,
    NanousdStage stage)
{
    static const char* kBaseColor =
        "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
        "Materials/2023_2_1/Automotive/Surfacing/ov_metal_a_01/"
        "ov_metal_a_01_basecolor.jpg";
    static const char* kOrm =
        "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
        "Materials/2023_2_1/Automotive/Surfacing/ov_metal_a_01/"
        "ov_metal_a_01_orm.jpg";
    static const char* kNormal =
        "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
        "Materials/2023_2_1/Automotive/Surfacing/ov_metal_a_01/"
        "ov_metal_a_01_normal.jpg";

    int base_idx = load_texture_dedup(kBaseColor, scene_dir,
                                      tex_paths, textures, stage);
    if (base_idx >= 0) {
        params->tex_indices[TEX_DIFFUSE_COLOR] = base_idx;
        params->base_color[0] = 1.0f;
        params->base_color[1] = 1.0f;
        params->base_color[2] = 1.0f;
        propagate_udim_scale(params, textures[base_idx]);
    }

    int normal_idx = load_texture_dedup(kNormal, scene_dir,
                                        tex_paths, textures, stage);
    if (normal_idx >= 0) {
        params->tex_indices[TEX_NORMAL] = normal_idx;
        propagate_udim_scale(params, textures[normal_idx]);
    }

    int orm_idx = load_texture_dedup(kOrm, scene_dir,
                                     tex_paths, textures, stage);
    if (orm_idx >= 0) {
        params->tex_indices[TEX_OCCLUSION] = orm_idx;
        params->tex_indices[TEX_ROUGHNESS] = orm_idx;
        params->tex_indices[TEX_METALLIC] = orm_idx;
        propagate_udim_scale(params, textures[orm_idx]);
    }

    if (getenv("NUSD_MAT_DIAG")) {
        fprintf(stderr,
                "[mat_diag] Metal_Black_Paint remote texture fallback: "
                "base=%d normal=%d orm=%d\n",
                base_idx, normal_idx, orm_idx);
    }
}

/* ---- Phase 7c: parse MaterialX `<standard_surface>` directly ----------
 *
 * Some pipelines (e.g. the OpenChessSet) bind materials via USD references
 * to .mtlx files. nanousd's USD reader doesn't ship a MaterialX file-format
 * plugin, so those references don't resolve and no Material prims show up
 * in the stage. We compensate here: scan the scene tree for *.mtlx, parse
 * every `<surfacematerial>` we find via the MaterialX C++ API, and fold the
 * resulting Standard Surface inputs into our flat MaterialParams.
 *
 * This is a deliberately narrow implementation — only `<standard_surface>`
 * with input edges that are either constants or `nodegraph="..." output="..."`
 * references to a single `<image>` (or `<normalmap>`-wrapped `<image>`).
 * That's exactly the shape every chess-set material uses; broader graph
 * support (procedural nodes, `<mix>`, etc.) is Phase 7c-2 territory.
 */

namespace {

/* Constant + filename split for one Standard Surface input.  Either
 * is_constant is true (with v[0..nVals-1] populated) or file_path is
 * non-empty (relative to the .mtlx file's directory). */
struct SsInput {
    bool        is_constant   = false;
    bool        is_normal_map = false;   /* set when reached via <normalmap> */
    bool        has_uv_scale  = false;
    int         n_vals        = 0;
    float       v[3]          = {0, 0, 0};
    float       uv_scale[2]   = {1.0f, 1.0f};
    std::string file_path;
};

struct ImageCandidate {
    std::string file_path;
    bool        has_uv_scale = false;
    float       uv_scale[2] = {1.0f, 1.0f};
};

/* Parse a constant value string into n floats. MaterialX stores
 * vec/color values as "x, y, z"; floats as a single number. */
static int parse_value(const std::string& s, int max_vals, float* out)
{
    int n = 0;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',') && n < max_vals) {
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a == std::string::npos) continue;
        try {
            out[n++] = std::stof(tok.substr(a, b - a + 1));
        } catch (...) { /* leave as-is */ }
    }
    return n;
}

static int parse_graph_input_value(mx::NodeGraphPtr ng,
                                   mx::InputPtr in,
                                   int max_vals,
                                   float* out,
                                   int depth = 0)
{
    if (!in || depth > 6) return 0;

    std::string vs = in->getValueString();
    if (!vs.empty()) return parse_value(vs, max_vals, out);

    if (ng && in->hasInterfaceName()) {
        mx::InputPtr iface = ng->getInput(in->getInterfaceName());
        int n = parse_graph_input_value(ng, iface, max_vals, out, depth + 1);
        if (n > 0) return n;
    }

    if (ng) {
        std::string child_name = in->getNodeName();
        mx::NodePtr child = child_name.empty() ? mx::NodePtr() : ng->getNode(child_name);
        if (child) {
            mx::InputPtr value = child->getInput("value");
            int n = parse_graph_input_value(ng, value, max_vals, out, depth + 1);
            if (n > 0) return n;
            mx::InputPtr inner = child->getInput("in");
            n = parse_graph_input_value(ng, inner, max_vals, out, depth + 1);
            if (n > 0) return n;
        }
    }

    return 0;
}

static void fill_image_candidate(mx::NodeGraphPtr ng,
                                 mx::NodePtr image,
                                 ImageCandidate& out)
{
    if (!image) return;
    mx::InputPtr file_in = image->getInput("file");
    if (file_in) out.file_path = file_in->getValueString();

    mx::InputPtr uv = image->getInput("uvtiling");
    if (uv) {
        float vals[2] = {1.0f, 1.0f};
        int n = parse_graph_input_value(ng, uv, 2, vals);
        if (n > 0) {
            out.has_uv_scale = true;
            out.uv_scale[0] = vals[0];
            out.uv_scale[1] = (n > 1) ? vals[1] : vals[0];
        }
    }
}

static void apply_image_candidate(const ImageCandidate& image, SsInput& out)
{
    out.file_path = image.file_path;
    out.has_uv_scale = image.has_uv_scale;
    out.uv_scale[0] = image.uv_scale[0];
    out.uv_scale[1] = image.uv_scale[1];
}

static std::string lower_copy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return s;
}

static bool contains_any(const std::string& s,
                         std::initializer_list<const char*> needles)
{
    for (const char* n : needles) {
        if (s.find(n) != std::string::npos) return true;
    }
    return false;
}

static int texture_semantic_score(const std::string& path,
                                  const std::string& input_name)
{
    std::string p = lower_copy(path);
    std::string in = lower_copy(input_name);
    if (contains_any(in, {"normal"}))
        return contains_any(p, {"normal", "nrm", "bump"}) ? 100 : 1;
    if (contains_any(in, {"rough"}))
        return contains_any(p, {"rough"}) ? 100 :
               contains_any(p, {"mask"}) ? 15 : 1;
    if (contains_any(in, {"metal"}))
        return contains_any(p, {"metal"}) ? 100 : 1;
    if (contains_any(in, {"opacity", "alpha"}))
        return contains_any(p, {"opacity", "alpha"}) ? 100 : 1;
    if (contains_any(in, {"base_color", "diffuse", "albedo", "color"}))
        return contains_any(p, {"base", "albedo", "diffuse", "color"}) ? 100 :
               contains_any(p, {"mask"}) ? 10 : 1;
    return 1;
}

static void collect_upstream_images(mx::NodeGraphPtr ng,
                                    mx::NodePtr node,
                                    std::unordered_set<std::string>& seen,
                                    std::vector<ImageCandidate>& out,
                                    int depth)
{
    if (!ng || !node || depth > 12) return;
    std::string key = node->getName();
    if (!seen.insert(key).second) return;

    const std::string& cat = node->getCategory();
    if (cat == "image" || cat == "tiledimage") {
        ImageCandidate image;
        fill_image_candidate(ng, node, image);
        if (!image.file_path.empty()) out.push_back(image);
        return;
    }

    for (mx::InputPtr in : node->getInputs()) {
        if (!in) continue;
        std::string child_name = in->getNodeName();
        if (child_name.empty()) continue;
        collect_upstream_images(ng, ng->getNode(child_name),
                                seen, out, depth + 1);
    }
}

static ImageCandidate choose_upstream_image(mx::NodeGraphPtr ng,
                                            mx::NodePtr driver,
                                            const std::string& input_name)
{
    std::vector<ImageCandidate> candidates;
    std::unordered_set<std::string> seen;
    collect_upstream_images(ng, driver, seen, candidates, 0);
    if (candidates.empty()) return {};

    int best = 0;
    int best_score = texture_semantic_score(candidates[0].file_path, input_name);
    for (int i = 1; i < (int)candidates.size(); ++i) {
        int score = texture_semantic_score(candidates[i].file_path, input_name);
        if (score > best_score) {
            best = i;
            best_score = score;
        }
    }
    return candidates[best];
}

static void apply_interface_constant(mx::NodeGraphPtr ng,
                                     const std::string& input_name,
                                     SsInput& r)
{
    if (!ng) return;
    std::string wanted = lower_copy(input_name);
    int best_score = 0;
    float best[3] = {0, 0, 0};
    int best_n = 0;

    for (mx::InputPtr in : ng->getInputs()) {
        if (!in || in->getValueString().empty()) continue;
        std::string name = lower_copy(in->getName());
        std::string type = in->getType();
        int score = 0;
        int max_vals = 1;
        if (contains_any(wanted, {"base_color", "diffuse", "albedo", "color"}) &&
            (type == "color3" || type == "vector3") &&
            contains_any(name, {"base", "brick", "diffuse", "albedo", "color"})) {
            score = contains_any(name, {"base"}) ? 100 : 80;
            max_vals = 3;
        } else if (contains_any(wanted, {"rough"}) &&
                   (type == "float" || type == "integer") &&
                   contains_any(name, {"rough"})) {
            score = 100;
            max_vals = 1;
        }
        if (!score) continue;

        float vals[3] = {0, 0, 0};
        int n = parse_value(in->getValueString(), max_vals, vals);
        if (n > 0 && score > best_score) {
            best_score = score;
            best_n = n;
            best[0] = vals[0];
            best[1] = vals[1];
            best[2] = vals[2];
        }
    }

    if (best_score > 0) {
        r.is_constant = true;
        r.n_vals = best_n;
        r.v[0] = best[0];
        r.v[1] = best[1];
        r.v[2] = best[2];
    }
}

/* Resolve a Standard Surface input to either a constant or an upstream
 * image filename. Handles direct images plus a bounded upstream walk for
 * simple MaterialX sample graphs such as the procedural brick material. */
static SsInput resolve_ss_input(mx::NodePtr ss,
                                mx::DocumentPtr doc,
                                const std::string& input_name)
{
    SsInput r;
    if (!ss || !doc) return r;

    mx::InputPtr in = ss->getInput(input_name);
    if (!in) return r;

    /* Connection through a nodegraph output. */
    std::string ng_name  = in->getNodeGraphString();
    std::string out_name = in->getOutputString();
    if (!ng_name.empty() && !out_name.empty()) {
        mx::NodeGraphPtr ng = doc->getNodeGraph(ng_name);
        if (ng) {
            mx::OutputPtr out_elt = ng->getOutput(out_name);
            if (out_elt) {
                std::string driver_name = out_elt->getNodeName();
                mx::NodePtr driver = ng->getNode(driver_name);
                if (driver) {
                    const std::string& cat = driver->getCategory();
                    if (cat == "image" || cat == "tiledimage") {
                        ImageCandidate image;
                        fill_image_candidate(ng, driver, image);
                        apply_image_candidate(image, r);
                        return r;
                    }
                    if (cat == "normalmap") {
                        mx::InputPtr nmIn = driver->getInput("in");
                        if (nmIn) {
                            std::string inner = nmIn->getNodeName();
                            mx::NodePtr inner_node = ng->getNode(inner);
                            if (inner_node && (inner_node->getCategory() == "image" ||
                                               inner_node->getCategory() == "tiledimage")) {
                                ImageCandidate image;
                                fill_image_candidate(ng, inner_node, image);
                                apply_image_candidate(image, r);
                                r.is_normal_map = true;
                                return r;
                            }
                        }
                    }
                    ImageCandidate image = choose_upstream_image(ng, driver, input_name);
                    apply_image_candidate(image, r);
                    if (!r.file_path.empty()) {
                        apply_interface_constant(ng, input_name, r);
                        return r;
                    }
                }
            }
        }
    }

    /* Constant value. */
    std::string vs = in->getValueString();
    if (vs.empty()) return r;
    r.is_constant = true;
    int max_vals = (in->getType() == "float" || in->getType() == "integer") ? 1 : 3;
    r.n_vals = parse_value(vs, max_vals, r.v);
    return r;
}

}  // namespace

static void set_default_mdl_uv_transform(MaterialParams* params)
{
    if (!params) return;
    params->mdl_uv_transform[0] = 1.0f;
    params->mdl_uv_transform[1] = 1.0f;
    params->mdl_uv_transform[2] = 0.0f;
    params->mdl_uv_transform[3] = 0.0f;
}

static void set_mdl_y_flipped_uv_scale(MaterialParams* params, float sx, float sy)
{
    if (!params) return;
    if (sx <= 0.0f) sx = 1.0f;
    if (sy <= 0.0f) sy = 1.0f;
    params->mdl_uv_transform[0] = sx;
    params->mdl_uv_transform[1] = sy;
    params->mdl_uv_transform[2] = 0.0f;
    params->mdl_uv_transform[3] = 1.0f - sy;
    params->v_flip = 0;
}

static void read_standard_surface_mtlx(
    mx::NodePtr ss,
    mx::DocumentPtr doc,
    MaterialParams* params,
    const std::string& mtlx_dir,
    std::vector<std::string>& tex_paths,
    std::vector<MaterialTexture>& textures,
    NanousdStage stage)
{
    /* Standard Surface defaults — match the MaterialX nodedef so an
     * unfilled input lands on the spec's intended fallback. */
    params->base_color[0] = 0.8f;
    params->base_color[1] = 0.8f;
    params->base_color[2] = 0.8f;
    params->base_color[3] = 1.0f;
    params->emissive_color[0] = 0.0f;
    params->emissive_color[1] = 0.0f;
    params->emissive_color[2] = 0.0f;
    params->emissive_color[3] = 1.0f;
    params->metallic = 0.0f;
    params->roughness = 0.2f;          /* SS specular_roughness default */
    params->opacity = 1.0f;
    params->ior = 1.5f;
    params->occlusion = 1.0f;
    params->clearcoat = 0.0f;
    params->clearcoat_roughness = 0.01f;
    params->normal_scale = 1.0f;
    /* TEX_NORMAL UsdUVTexture scale/bias defaults: (2,2,2,1)/(-1,-1,-1,0)
     * reproduce the historical implicit `nm = nm * 2 - 1` remap. The
     * UPS path sets these in read_usd_preview_surface; without mirroring
     * them in the MTLX path the SceneMaterial = {} zero-init zeros the
     * bias and the shader computes `nm * 0 + 0 = 0` for every normal-
     * mapped surface — chess set rendered pure black silhouettes. */
    params->normal_tex_scale[0] = 2.0f;
    params->normal_tex_scale[1] = 2.0f;
    params->normal_tex_scale[2] = 2.0f;
    params->normal_tex_scale[3] = 1.0f;
    params->normal_tex_bias[0]  = -1.0f;
    params->normal_tex_bias[1]  = -1.0f;
    params->normal_tex_bias[2]  = -1.0f;
    params->normal_tex_bias[3]  =  0.0f;
    set_default_mdl_uv_transform(params);
    params->roughness_tex_scale = 1.0f;
    params->roughness_tex_bias = 0.0f;
    params->use_vertex_color = 0;
    params->udim_scale_u = 1.0f;
    params->udim_scale_v = 1.0f;
    params->v_flip = 0;
    params->opacity_threshold = 0.0f;
    for (int i = 0; i < MAX_MATERIAL_TEXTURES; i++) params->tex_indices[i] = -1;
    params->tex_subsurface_weight = -1;
    params->tex_transmission_weight = -1;
    params->subsurface_weight = 0.0f;
    params->subsurface_scale = 1.0f;
    params->transmission_weight = 0.0f;
    params->transmission_ior = 0.0f;
    /* By default subsurface tints stay white; the nodegraph or a constant
     * may override them below. */
    params->subsurface_color[0] = 1.0f;
    params->subsurface_color[1] = 1.0f;
    params->subsurface_color[2] = 1.0f;
    params->subsurface_color[3] = 1.0f;
    params->subsurface_radius[0] = 1.0f;
    params->subsurface_radius[1] = 1.0f;
    params->subsurface_radius[2] = 1.0f;
    params->subsurface_radius[3] = 1.0f;
    params->transmission_color[0] = 1.0f;
    params->transmission_color[1] = 1.0f;
    params->transmission_color[2] = 1.0f;
    params->transmission_color[3] = 1.0f;

    if (!ss) return;

    auto load = [&](const std::string& rel) -> int {
        if (rel.empty()) return -1;
        return load_texture_dedup(rel.c_str(), mtlx_dir.c_str(),
                                  tex_paths, textures, stage);
    };

    /* Helper: assign a constant or texture into a (vec3 + tex) slot. */
    auto bind_color3 = [&](const SsInput& in, float* out_rgb, int tex_slot) {
        if (in.has_uv_scale) {
            params->mdl_uv_transform[0] = in.uv_scale[0];
            params->mdl_uv_transform[1] = in.uv_scale[1];
        }
        if (in.is_constant && in.n_vals >= 3) {
            out_rgb[0] = in.v[0];
            out_rgb[1] = in.v[1];
            out_rgb[2] = in.v[2];
        }
        if (!in.file_path.empty()) {
            int t = load(in.file_path);
            if (t >= 0 && tex_slot >= 0) params->tex_indices[tex_slot] = t;
        }
    };

    auto bind_float = [&](const SsInput& in, float* out_f, int tex_slot) {
        if (in.has_uv_scale) {
            params->mdl_uv_transform[0] = in.uv_scale[0];
            params->mdl_uv_transform[1] = in.uv_scale[1];
        }
        if (in.is_constant && in.n_vals >= 1) {
            *out_f = in.v[0];
        }
        if (!in.file_path.empty()) {
            int t = load(in.file_path);
            if (t >= 0 && tex_slot >= 0) params->tex_indices[tex_slot] = t;
        }
    };

    /* === Standard Surface inputs we care about for OpenChessSet === */
    {
        SsInput base = resolve_ss_input(ss, doc, "base_color");
        bind_color3(base, params->base_color, TEX_DIFFUSE_COLOR);
        if (!base.is_constant && !base.file_path.empty() &&
            params->tex_indices[TEX_DIFFUSE_COLOR] >= 0) {
            params->base_color[0] = 1.0f;
            params->base_color[1] = 1.0f;
            params->base_color[2] = 1.0f;
        }
    }

    bind_float(resolve_ss_input(ss, doc, "metalness"),
               &params->metallic, TEX_METALLIC);

    bind_float(resolve_ss_input(ss, doc, "specular_roughness"),
               &params->roughness, TEX_ROUGHNESS);

    /* `normal` arrives via a <normalmap> wrapper; resolve_ss_input flags
     * is_normal_map so we know to bind to TEX_NORMAL. */
    SsInput nrm = resolve_ss_input(ss, doc, "normal");
    if (!nrm.file_path.empty()) {
        if (nrm.has_uv_scale) {
            params->mdl_uv_transform[0] = nrm.uv_scale[0];
            params->mdl_uv_transform[1] = nrm.uv_scale[1];
        }
        int t = load(nrm.file_path);
        if (t >= 0) params->tex_indices[TEX_NORMAL] = t;
    }

    /* Subsurface block — King and Queen drive `subsurface` from a
     * scattering map, color and radius from base_color. The chess-set
     * conventions match exactly here. */
    SsInput sss = resolve_ss_input(ss, doc, "subsurface");
    if (sss.is_constant && sss.n_vals >= 1) {
        params->subsurface_weight = sss.v[0];
    } else if (!sss.file_path.empty()) {
        int t = load(sss.file_path);
        if (t >= 0) {
            params->tex_subsurface_weight = t;
            params->subsurface_weight = 1.0f;  /* texture supplies the weight */
        }
    }
    {
        SsInput ssc = resolve_ss_input(ss, doc, "subsurface_color");
        bind_color3(ssc, params->subsurface_color, -1);
        /* Flag whether we actually got a constant value (vs. unresolved
         * nodegraph or input missing). Lets the shader distinguish "white
         * SSS authored intentionally" (sss_color_authored=1, sssColor=
         * (1,1,1) — milk, etc.) from "MaterialX default leaked because we
         * couldn't resolve a nodegraph" (sss_color_authored=0 — fall back
         * to baseColor for the body tint). Replaces the old fragile
         * `sssColor.r < 0.999` heuristic. */
        params->sss_color_authored = (ssc.is_constant && ssc.n_vals >= 3) ? 1 : 0;
    }
    bind_color3(resolve_ss_input(ss, doc, "subsurface_radius"),
                params->subsurface_radius, -1);
    {
        SsInput sc = resolve_ss_input(ss, doc, "subsurface_scale");
        if (sc.is_constant && sc.n_vals >= 1) params->subsurface_scale = sc.v[0];
    }

    /* Transmission block — Pawn tops set `transmission=1` constant + a
     * tinted `transmission_color`. */
    SsInput tr = resolve_ss_input(ss, doc, "transmission");
    if (tr.is_constant && tr.n_vals >= 1) {
        params->transmission_weight = tr.v[0];
    } else if (!tr.file_path.empty()) {
        int t = load(tr.file_path);
        if (t >= 0) {
            params->tex_transmission_weight = t;
            params->transmission_weight = 1.0f;
        }
    }
    bind_color3(resolve_ss_input(ss, doc, "transmission_color"),
                params->transmission_color, -1);
    {
        SsInput ti = resolve_ss_input(ss, doc, "specular_IOR");
        if (ti.is_constant && ti.n_vals >= 1) {
            params->ior = ti.v[0];
            params->transmission_ior = ti.v[0];
        }
    }

    /* Emission — chess set doesn't use it but harmless. */
    {
        SsInput ew = resolve_ss_input(ss, doc, "emission");
        SsInput ec = resolve_ss_input(ss, doc, "emission_color");
        float weight = 0.0f;
        if (ew.is_constant && ew.n_vals >= 1) weight = ew.v[0];
        if (weight > 0.0f) {
            if (ec.is_constant && ec.n_vals >= 3) {
                params->emissive_color[0] = ec.v[0] * weight;
                params->emissive_color[1] = ec.v[1] * weight;
                params->emissive_color[2] = ec.v[2] * weight;
            }
            params->emissive_color[3] = 1.0f;
        }
    }
}

/* Recursively scan a directory for *.mtlx files. Skips hidden dirs and
 * anything more than 6 levels deep, which matches the chess-set layout
 * and avoids runaway traversal on accidental symlink loops. */
static std::vector<std::string> find_mtlx_files(const std::string& root, int max_depth = 6)
{
    std::vector<std::string> out;
    try {
        if (!fs::exists(root) || !fs::is_directory(root)) return out;
        fs::recursive_directory_iterator it(root,
            fs::directory_options::skip_permission_denied);
        for (; it != fs::recursive_directory_iterator(); ++it) {
            if (it.depth() > max_depth) { it.disable_recursion_pending(); continue; }
            const auto& p = it->path();
            std::string fname = p.filename().string();
            if (!fname.empty() && fname[0] == '.') {
                if (fs::is_directory(p)) it.disable_recursion_pending();
                continue;
            }
            if (it->is_regular_file() && p.extension() == ".mtlx") {
                out.push_back(p.string());
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "material: mtlx scan failed under %s: %s\n",
                root.c_str(), e.what());
    }
    std::sort(out.begin(), out.end());
    return out;
}

/* Parse every .mtlx under scene_dir, append a SceneMaterial per
 * <surfacematerial> we find. Returns the number of materials added.
 *
 * No-op if MaterialX isn't initialised (g_stdlib unset) — caller should
 * have called materialx_init().
 */
static int load_mtlx_directory(const char* scene_dir,
                               std::vector<SceneMaterial>& materials,
                               std::vector<std::string>& tex_paths,
                               std::vector<MaterialTexture>& textures,
                               std::unordered_map<std::string, int>& mat_name_to_idx,
                               NanousdStage stage)
{
    if (!scene_dir || !*scene_dir) return 0;
    if (!g_materialx_initialized) {
        fprintf(stderr, "material: mtlx scan skipped — MaterialX not initialized\n");
        return 0;
    }

    std::vector<std::string> files = find_mtlx_files(scene_dir);
    if (files.empty()) return 0;

    fprintf(stderr, "material: scanning %zu .mtlx file(s) under %s\n",
            files.size(), scene_dir);

    int added = 0;
    for (const std::string& path : files) {
        mx::DocumentPtr doc = mx::createDocument();
        try {
            /* Intentionally no setDataLibrary(g_stdlib) — we only want
             * the nodes authored in *this* file. The stdlib carries
             * sample-asset Materials (chess_set, brass, wood) whose
             * referenced textures don't exist here, and folding it in
             * makes resolve_texture_path try every one of them against
             * the current scene dir. */
            mx::readFromXmlFile(doc, path);
        } catch (const std::exception& e) {
            fprintf(stderr, "material: failed to parse %s: %s\n",
                    path.c_str(), e.what());
            continue;
        }

        std::string mtlx_dir = fs::path(path).parent_path().string();

        /* Each <surfacematerial> becomes a SceneMaterial. We look up its
         * `surfaceshader` input and follow it to the connected
         * <standard_surface> inside the same document. `getNodes()` with
         * an empty filter is the safest cross-version way to enumerate;
         * we filter by category ourselves. */
        int file_added = 0;
        for (mx::NodePtr mat : doc->getNodes()) {
            if (mat->getCategory() != "surfacematerial") continue;
            mx::InputPtr ss_in = mat->getInput("surfaceshader");
            if (!ss_in) continue;
            std::string ss_name = ss_in->getNodeName();
            if (ss_name.empty()) continue;
            mx::NodePtr ss = doc->getNode(ss_name);
            if (!ss) continue;
            if (ss->getCategory() != "standard_surface") {
                /* Could extend to UsdPreviewSurface / OpenPBR later. */
                continue;
            }

            std::string mat_name = mat->getName();
            if (mat_name_to_idx.count(mat_name)) {
                /* Same surfacematerial name already registered — most
                 * likely two .mtlx files defined the same material. Skip
                 * the duplicate rather than shadow the first reading. */
                continue;
            }

            SceneMaterial sm = {};
            snprintf(sm.name, sizeof(sm.name), "%s", mat_name.c_str());
            /* Synthesize a prim path. The exact-path match in
             * materials_find_binding will miss; the leaf-name fallback
             * resolves correctly. */
            snprintf(sm.prim_path, sizeof(sm.prim_path),
                     "/Materials/%s", mat_name.c_str());

            read_standard_surface_mtlx(ss, doc, &sm.params, mtlx_dir,
                                       tex_paths, textures, stage);

            mat_name_to_idx[mat_name] = (int)materials.size();
            materials.push_back(sm);
            added++;
            file_added++;
        }
        if (file_added > 0) {
            fprintf(stderr, "material:   %s → %d material(s)\n",
                    fs::path(path).filename().string().c_str(), file_added);
        }
    }

    fprintf(stderr, "material: loaded %d MaterialX materials from %zu file(s)\n",
            added, files.size());
    return added;
}

/* ---- Read material properties from USD prim ---- */

static bool material_color_is_placeholder_debug(const float c[4])
{
    int hi = 0;
    int lo = 0;
    for (int i = 0; i < 3; ++i) {
        if (c[i] > 0.95f) ++hi;
        if (c[i] < 0.05f) ++lo;
    }
    return hi >= 1 && hi <= 2 && hi + lo == 3;
}

static bool material_read_asset_string(NanousdPrim prim,
                                       const char* attr,
                                       std::string& out)
{
    if (!prim || !attr) return false;
    int ok = 0;
    const char* s = nanousd_attribasset(prim, attr, &ok);
    if (!ok || !s || !s[0])
        s = nanousd_attribs(prim, attr, &ok);
    if (!ok || !s || !s[0]) return false;
    out = s;
    return true;
}

static bool material_try_resolve_ptex_asset(const std::string& asset,
                                            const char* scene_dir,
                                            NanousdStage stage,
                                            char out[512])
{
    if (asset.empty()) return false;
    std::string lower = lower_copy(asset);
    if (!contains_any(lower, {".ptx", ".ptex", "ptex"}))
        return false;

    std::string resolved = resolve_texture_path(asset.c_str(), scene_dir, stage);
    const std::string& value = resolved.empty() ? asset : resolved;
    snprintf(out, 512, "%s", value.c_str());
    return true;
}

static bool read_material_ptex_color_path(NanousdPrim material_prim,
                                          const char* scene_dir,
                                          NanousdStage stage,
                                          char out[512])
{
    if (!out) return false;
    out[0] = '\0';
    if (!material_prim) return false;

    const char* direct_attrs[] = {
        "inputs:surfaceMap",
        "inputs:baseColor",
        "inputs:diffuseColor"
    };
    for (const char* attr : direct_attrs) {
        std::string asset;
        if (material_read_asset_string(material_prim, attr, asset) &&
            material_try_resolve_ptex_asset(asset, scene_dir, stage, out)) {
            return true;
        }
    }

    int nchildren = nanousd_nchildren(material_prim);
    for (int i = 0; i < nchildren; ++i) {
        NanousdPrim child = nanousd_child(material_prim, i);
        if (!child) continue;

        const char* child_attrs[] = {
            "inputs:surfaceMap",
            "inputs:file",
            "inputs:filename",
            "inputs:texture",
            "inputs:baseColor",
            "inputs:diffuseColor"
        };
        for (const char* attr : child_attrs) {
            std::string asset;
            if (material_read_asset_string(child, attr, asset) &&
                material_try_resolve_ptex_asset(asset, scene_dir, stage, out)) {
                nanousd_freeprim(child);
                return true;
            }
        }
        nanousd_freeprim(child);
    }
    return false;
}

static void apply_material_interface_inputs(NanousdPrim material_prim,
                                            MaterialParams* params)
{
    if (!material_prim || !params) return;

    float v3[3];
    if (nanousd_attribv3f(material_prim, "inputs:baseColor", v3) ||
        nanousd_attribv3f(material_prim, "inputs:diffuseColor", v3)) {
        params->base_color[0] = v3[0];
        params->base_color[1] = v3[1];
        params->base_color[2] = v3[2];
    }

    int ok = 0;
    float f = nanousd_attribf(material_prim, "inputs:alpha", &ok);
    if (!ok) f = nanousd_attribf(material_prim, "inputs:opacity", &ok);
    if (ok) {
        params->opacity = f;
        params->base_color[3] = f;
    }

    f = nanousd_attribf(material_prim, "inputs:metallic", &ok);
    if (ok) params->metallic = f;
    f = nanousd_attribf(material_prim, "inputs:roughness", &ok);
    if (ok) params->roughness = f;
    f = nanousd_attribf(material_prim, "inputs:ior", &ok);
    if (ok) params->ior = f;
    f = nanousd_attribf(material_prim, "inputs:clearcoat", &ok);
    if (ok) params->clearcoat = f;
    f = nanousd_attribf(material_prim, "inputs:clearcoatRoughness", &ok);
    if (ok) params->clearcoat_roughness = f;
    f = nanousd_attribf(material_prim, "inputs:clearcoatGloss", &ok);
    if (ok) params->clearcoat_roughness = std::clamp(1.0f - f, 0.01f, 1.0f);
    f = nanousd_attribf(material_prim, "inputs:occlusion", &ok);
    if (ok) params->occlusion = f;

    params->metallic = std::clamp(params->metallic, 0.0f, 1.0f);
    params->roughness = std::clamp(params->roughness, 0.01f, 1.0f);
    params->opacity = std::clamp(params->opacity, 0.0f, 1.0f);
    params->ior = std::clamp(params->ior, 1.0f, 3.0f);
}

static void read_usd_preview_surface(NanousdPrim shader_prim,
                                     NanousdPrim material_prim,
                                     MaterialParams* params,
                                     std::vector<std::string>& tex_paths,
                                     std::vector<MaterialTexture>& textures,
                                     const char* scene_dir,
                                     NanousdStage stage = nullptr)
{
    // Initialize defaults matching UsdPreviewSurface spec
    params->base_color[0] = 0.18f;
    params->base_color[1] = 0.18f;
    params->base_color[2] = 0.18f;
    params->base_color[3] = 1.0f;
    params->emissive_color[0] = 0.0f;
    params->emissive_color[1] = 0.0f;
    params->emissive_color[2] = 0.0f;
    params->emissive_color[3] = 1.0f;
    params->metallic = 0.0f;
    params->roughness = 0.5f;
    params->opacity = 1.0f;
    params->ior = 1.5f;
    params->occlusion = 1.0f;
    params->clearcoat = 0.0f;
    params->clearcoat_roughness = 0.01f;
    params->normal_scale = 1.0f;
    /* See read_standard_surface_mtlx for rationale. */
    params->normal_tex_scale[0] = 2.0f;
    params->normal_tex_scale[1] = 2.0f;
    params->normal_tex_scale[2] = 2.0f;
    params->normal_tex_scale[3] = 1.0f;
    params->normal_tex_bias[0]  = -1.0f;
    params->normal_tex_bias[1]  = -1.0f;
    params->normal_tex_bias[2]  = -1.0f;
    params->normal_tex_bias[3]  =  0.0f;
    set_default_mdl_uv_transform(params);
    params->use_vertex_color = 0;
    params->udim_scale_u = 1.0f;
    params->udim_scale_v = 1.0f;
    params->v_flip = 0;
    params->use_specular_workflow = 0;
    params->specular_color[0] = 0.0f;
    params->specular_color[1] = 0.0f;
    params->specular_color[2] = 0.0f;
    params->specular_color[3] = 0.0f;
    params->opacity_threshold = 0.0f;
    /* Defaults reproduce the implicit `nm = nm * 2 - 1` historical
     * behaviour, so scenes that don't author scale/bias on the normal-
     * map UsdUVTexture render bit-identically to before this was wired. */
    params->normal_tex_scale[0] = 2.0f;
    params->normal_tex_scale[1] = 2.0f;
    params->normal_tex_scale[2] = 2.0f;
    params->normal_tex_scale[3] = 1.0f;
    params->normal_tex_bias[0]  = -1.0f;
    params->normal_tex_bias[1]  = -1.0f;
    params->normal_tex_bias[2]  = -1.0f;
    params->normal_tex_bias[3]  =  0.0f;
    set_default_mdl_uv_transform(params);

    for (int i = 0; i < MAX_MATERIAL_TEXTURES; i++)
        params->tex_indices[i] = -1;

    if (!shader_prim) {
        apply_material_interface_inputs(material_prim, params);
        return;
    }

    // Read scalar parameters
    float v3[3];
    int ok;

    bool had_constant_diffuse = false;
    if (nanousd_attribv3f(shader_prim, "inputs:diffuseColor", v3)) {
        params->base_color[0] = v3[0];
        params->base_color[1] = v3[1];
        params->base_color[2] = v3[2];
        had_constant_diffuse = true;
    }

    if (nanousd_attribv3f(shader_prim, "inputs:emissiveColor", v3)) {
        params->emissive_color[0] = v3[0];
        params->emissive_color[1] = v3[1];
        params->emissive_color[2] = v3[2];
    }

    float f;
    f = nanousd_attribf(shader_prim, "inputs:metallic", &ok);
    if (ok) params->metallic = f;

    f = nanousd_attribf(shader_prim, "inputs:roughness", &ok);
    if (ok) params->roughness = f;

    f = nanousd_attribf(shader_prim, "inputs:opacity", &ok);
    if (ok) params->opacity = f;

    f = nanousd_attribf(shader_prim, "inputs:ior", &ok);
    if (ok) params->ior = f;

    f = nanousd_attribf(shader_prim, "inputs:clearcoat", &ok);
    if (ok) params->clearcoat = f;

    f = nanousd_attribf(shader_prim, "inputs:clearcoatRoughness", &ok);
    if (ok) params->clearcoat_roughness = f;

    /* useSpecularWorkflow + specularColor (UsdPreviewSurface spec):
     *   useSpecularWorkflow=1 → F0 is artist-authored as specularColor
     *   useSpecularWorkflow=0 → metallic workflow, F0 derived from metallic
     * Authored as `int` in USDA but written via attribf works for the 0/1
     * range we care about. */
    int spec_int_ok;
    int spec_int = nanousd_attribi(shader_prim,
                                   "inputs:useSpecularWorkflow",
                                   &spec_int_ok);
    if (spec_int_ok) {
        params->use_specular_workflow = (spec_int != 0) ? 1 : 0;
    }
    if (nanousd_attribv3f(shader_prim, "inputs:specularColor", v3)) {
        params->specular_color[0] = v3[0];
        params->specular_color[1] = v3[1];
        params->specular_color[2] = v3[2];
    }

    /* opacityThreshold (UsdPreviewSurface): >0 enables the alpha-cutout
     * binary test in the rchit shader; 0 disables (default). */
    f = nanousd_attribf(shader_prim, "inputs:opacityThreshold", &ok);
    if (ok) params->opacity_threshold = f;

    /* normalScale (UsdPreviewSurface): scalar multiplier applied to the
     * tangent-space normal-map XY components. Default 1.0. The shader
     * applies this AFTER scale/bias remap, so it controls bump strength
     * after the [-1,1] mapping. */
    f = nanousd_attribf(shader_prim, "inputs:normalScale", &ok);
    if (ok) params->normal_scale = f;

    f = nanousd_attribf(shader_prim, "inputs:occlusion", &ok);
    if (ok) params->occlusion = f;

    // --- Connection-based texture resolution (like OpenUSD) ---
    // Follow explicit .connect attributes on shader inputs to find
    // upstream UsdUVTexture prims, then read their inputs:file.
    struct { const char* input; int slot; } conn_inputs[] = {
        {"inputs:diffuseColor",  TEX_DIFFUSE_COLOR},
        {"inputs:normal",        TEX_NORMAL},
        {"inputs:roughness",     TEX_ROUGHNESS},
        {"inputs:metallic",      TEX_METALLIC},
        {"inputs:emissiveColor", TEX_EMISSIVE_COLOR},
        {"inputs:occlusion",     TEX_OCCLUSION},
        {"inputs:opacity",       TEX_OPACITY},
    };

    for (auto& ci : conn_inputs) {
        int nconn = nanousd_nconnections(shader_prim, ci.input);
        if (nconn <= 0) continue;

        const char* conn_path = nanousd_connection(shader_prim, ci.input, 0);
        if (!conn_path || !conn_path[0]) continue;

        // conn_path is like: /Material/metallic_roughness_texture.outputs:b
        // Extract prim path (before the dot)
        std::string prim_path(conn_path);
        auto dot = prim_path.rfind('.');
        if (dot != std::string::npos)
            prim_path = prim_path.substr(0, dot);

        NanousdPrim tex_prim = nanousd_primpath(stage, prim_path.c_str());
        if (!tex_prim) continue;

        const char* file = nanousd_attribasset(tex_prim, "inputs:file", &ok);
        if (!ok || !file || !file[0]) {
            file = nanousd_attribs(tex_prim, "inputs:file", &ok);
        }
        bool tex_loaded = false;
        if (ok && file && file[0]) {
            int tex_idx = load_texture_dedup(file, scene_dir,
                                             tex_paths, textures, stage,
                                             tex_prim);
            if (tex_idx >= 0) {
                params->tex_indices[ci.slot] = tex_idx;
                tex_loaded = true;
                // Propagate UDIM grid dimensions
                MaterialTexture& tex = textures[tex_idx];
                if (tex.udim_cols > 0 && tex.udim_cols > (int)params->udim_scale_u)
                    params->udim_scale_u = (float)tex.udim_cols;
                if (tex.udim_rows > 0 && tex.udim_rows > (int)params->udim_scale_v)
                    params->udim_scale_v = (float)tex.udim_rows;
            }
        }
        /* UsdUVTexture inputs:fallback (color4f, default (0,0,0,1)) is
         * what the texture node outputs when the file is missing or
         * unbound. Applies the same per-slot mapping the shader would do
         * if the texture had loaded — write to the underlying constant. */
        if (!tex_loaded) {
            float fb[4];
            if (nanousd_attribv4f(tex_prim, "inputs:fallback", fb)) {
                switch (ci.slot) {
                case TEX_DIFFUSE_COLOR:
                    params->base_color[0] = fb[0];
                    params->base_color[1] = fb[1];
                    params->base_color[2] = fb[2];
                    break;
                case TEX_EMISSIVE_COLOR:
                    params->emissive_color[0] = fb[0];
                    params->emissive_color[1] = fb[1];
                    params->emissive_color[2] = fb[2];
                    break;
                case TEX_ROUGHNESS:
                    params->roughness = fb[1];  /* .g per ORM convention */
                    break;
                case TEX_METALLIC:
                    params->metallic = fb[2];   /* .b per ORM convention */
                    break;
                case TEX_OCCLUSION:
                    params->occlusion = fb[0];  /* .r */
                    break;
                case TEX_OPACITY:
                    params->opacity = fb[3];    /* alpha channel */
                    break;
                }
            }
        }
        /* For the normal-map slot, also pull inputs:scale and inputs:bias
         * off the UsdUVTexture so the shader can apply them explicitly
         * instead of the implicit `nm * 2 - 1`. Other slots ignore them
         * for now (Phase C closure: extend to diffuse/roughness/etc. by
         * generalising to per-slot scale/bias arrays). */
        if (ci.slot == TEX_NORMAL) {
            float s4[4];
            if (nanousd_attribv4f(tex_prim, "inputs:scale", s4)) {
                params->normal_tex_scale[0] = s4[0];
                params->normal_tex_scale[1] = s4[1];
                params->normal_tex_scale[2] = s4[2];
                params->normal_tex_scale[3] = s4[3];
            }
            float b4[4];
            if (nanousd_attribv4f(tex_prim, "inputs:bias", b4)) {
                params->normal_tex_bias[0] = b4[0];
                params->normal_tex_bias[1] = b4[1];
                params->normal_tex_bias[2] = b4[2];
                params->normal_tex_bias[3] = b4[3];
            }
        }
        /* For the diffuse-color slot, fold UsdUVTexture inputs:scale into
         * base_color. The rchit/raster shaders compute
         * `baseColor = tex_color.rgb * mat.base_color.rgb`, so writing
         * the scale into base_color reproduces the UsdUVTexture spec
         * `out = sample * scale + bias` for the common bias=0 case.
         *
         * When inputs:diffuseColor is *connected* but has no constant
         * authored, the upstream loader leaves base_color at the
         * UsdPreviewSurface default (0.18). That default is meant for
         * untextured surfaces — once a texture is connected, the
         * texture IS the color source, so we reset base_color to
         * (1,1,1) before applying scale. Asset that DO author a
         * constant alongside a connection (rare; not spec-compliant)
         * are still honored — caught by checking nconnections AND that
         * the constant differs from the default. */
        if (ci.slot == TEX_DIFFUSE_COLOR && tex_loaded) {
            /* When a diffuse texture is connected and no constant
             * inputs:diffuseColor was authored, the upstream nanousd
             * loader left base_color at the UPS default (0.18) — that
             * default is meant for untextured surfaces, so once a
             * texture is the color source we want to start from white
             * and let inputs:scale + texture sample drive the color.
             * Track whether the constant was actually authored at line
             * ~1524 above; sentinel-checking the value would misfire on
             * any UPS asset that authors (0.18, 0.18, 0.18) deliberately
             * alongside a connection. */
            if (!had_constant_diffuse) {
                params->base_color[0] = 1.0f;
                params->base_color[1] = 1.0f;
                params->base_color[2] = 1.0f;
            }
            float s4[4];
            if (nanousd_attribv4f(tex_prim, "inputs:scale", s4)) {
                params->base_color[0] *= s4[0];
                params->base_color[1] *= s4[1];
                params->base_color[2] *= s4[2];
                /* alpha component handled separately via opacity */
            }
            float b4[4];
            if (nanousd_attribv4f(tex_prim, "inputs:bias", b4)) {
                if (b4[0] != 0.0f || b4[1] != 0.0f || b4[2] != 0.0f) {
                    fprintf(stderr,
                        "material: warning: inputs:bias=(%.3f,%.3f,%.3f) "
                        "on diffuseColor texture not yet honored\n",
                        b4[0], b4[1], b4[2]);
                }
            }
        }
        nanousd_freeprim(tex_prim);
    }

    // Fallback: scan Material children for UsdUVTexture prims not reached
    // via connections (e.g., older assets without explicit connections)
    NanousdPrim mat_parent = nanousd_parent(shader_prim);
    if (mat_parent) {
        int nchildren = nanousd_nchildren(mat_parent);
        for (int c = 0; c < nchildren; c++) {
            NanousdPrim child = nanousd_child(mat_parent, c);
            if (!child) continue;

            const char* file = nanousd_attribasset(child, "inputs:file", &ok);
            if (!ok || !file || !file[0]) {
                file = nanousd_attribs(child, "inputs:file", &ok);
            }
            if (ok && file && file[0]) {
                int tex_idx = load_texture_dedup(file, scene_dir,
                                                 tex_paths, textures, stage);
                const char* cp = nanousd_path(child);
                if (cp && tex_idx >= 0) {
                    std::string lower(cp);
                    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                    if (params->tex_indices[TEX_DIFFUSE_COLOR] < 0 &&
                        (lower.find("diffuse") != std::string::npos ||
                         lower.find("basecolor") != std::string::npos))
                        params->tex_indices[TEX_DIFFUSE_COLOR] = tex_idx;
                    if (params->tex_indices[TEX_NORMAL] < 0 &&
                        lower.find("normal") != std::string::npos)
                        params->tex_indices[TEX_NORMAL] = tex_idx;
                    if (params->tex_indices[TEX_ROUGHNESS] < 0 &&
                        lower.find("rough") != std::string::npos)
                        params->tex_indices[TEX_ROUGHNESS] = tex_idx;
                    if (params->tex_indices[TEX_METALLIC] < 0 &&
                        lower.find("metal") != std::string::npos)
                        params->tex_indices[TEX_METALLIC] = tex_idx;
                    if (params->tex_indices[TEX_EMISSIVE_COLOR] < 0 &&
                        lower.find("emissive") != std::string::npos)
                        params->tex_indices[TEX_EMISSIVE_COLOR] = tex_idx;
                    if (params->tex_indices[TEX_OCCLUSION] < 0 &&
                        (lower.find("occlusion") != std::string::npos ||
                         lower.find("_ao") != std::string::npos))
                        params->tex_indices[TEX_OCCLUSION] = tex_idx;
                }
            }
            nanousd_freeprim(child);
        }
        nanousd_freeprim(mat_parent);
    }

    /* Moana-style RenderMan/Ptex materials author scalar controls directly on
     * the Material interface and connect preview diffuseColor to a Ptex shader
     * node. Ptex color maps are sampled later; these scalar inputs still carry
     * roughness, IOR, sheen, opacity, and non-Ptex constant colors. */
    apply_material_interface_inputs(material_prim, params);
}

/* ---- Shader type detection ---- */

enum ShaderType {
    SHADER_UNKNOWN = 0,
    SHADER_USD_PREVIEW_SURFACE,
    SHADER_PXR_DISNEY_BSDF,
    SHADER_PXR_SURFACE,
    SHADER_PXR_VOLUME,
    SHADER_MATERIALX_STANDARD_SURFACE,
    SHADER_MATERIALX_OPEN_PBR,
    SHADER_OMNIPBR,
    SHADER_OMNIGLASS,
    SHADER_OMNISURFACE,
    SHADER_MDL_GENERIC,
};

static ShaderType detect_shader_type(NanousdPrim shader_prim)
{
    if (!shader_prim) return SHADER_UNKNOWN;
    int ok;
    const char* info_id = nanousd_attrib_token(shader_prim, "info:id", &ok);
    if (ok && info_id) {
        if (strcmp(info_id, "UsdPreviewSurface") == 0) return SHADER_USD_PREVIEW_SURFACE;
        if (strcmp(info_id, "PxrDisneyBsdf") == 0) return SHADER_PXR_DISNEY_BSDF;
        if (strcmp(info_id, "PxrSurface") == 0) return SHADER_PXR_SURFACE;
        if (strcmp(info_id, "PxrVolume") == 0) return SHADER_PXR_VOLUME;
        if (strstr(info_id, "ND_standard_surface") ||
            strstr(info_id, "standard_surface_surfaceshader"))
            return SHADER_MATERIALX_STANDARD_SURFACE;
        if (strstr(info_id, "ND_open_pbr_surface") ||
            strstr(info_id, "open_pbr_surface_surfaceshader"))
            return SHADER_MATERIALX_OPEN_PBR;
        if (strstr(info_id, "OmniPBR"))     return SHADER_OMNIPBR;
        if (strstr(info_id, "OmniGlass"))   return SHADER_OMNIGLASS;
        if (strstr(info_id, "OmniSurface")) return SHADER_OMNISURFACE;
        if (strstr(info_id, "mdlMaterial")) return SHADER_MDL_GENERIC;
    }
    // Check MDL sub-identifier
    const char* mdl_sub = nanousd_attrib_token(shader_prim, "info:mdl:sourceAsset:subIdentifier", &ok);
    if (ok && mdl_sub) {
        if (strstr(mdl_sub, "OmniPBR"))     return SHADER_OMNIPBR;
        if (strstr(mdl_sub, "OmniGlass"))   return SHADER_OMNIGLASS;
        if (strstr(mdl_sub, "OmniSurface")) return SHADER_OMNISURFACE;
        return SHADER_MDL_GENERIC;
    }
    const char* mdl_asset = nanousd_attribasset(shader_prim,
                                                 "info:mdl:sourceAsset", &ok);
    if (ok && mdl_asset && mdl_asset[0]) return SHADER_MDL_GENERIC;
    return SHADER_UNKNOWN;
}

static int classify_texture_slot(const char* name_hint, const char* tex_path,
                                 bool* is_orm)
{
    if (is_orm) *is_orm = false;
    if (!tex_path || !tex_path[0]) return -1;

    std::string lower_fn(tex_path);
    std::transform(lower_fn.begin(), lower_fn.end(),
                   lower_fn.begin(), ::tolower);
    if (lower_fn.find(".mdl") != std::string::npos) return -1;

    std::string lower_hint = name_hint ? name_hint : "";
    std::transform(lower_hint.begin(), lower_hint.end(),
                   lower_hint.begin(), ::tolower);
    if (lower_hint.find("maskselection") != std::string::npos ||
        lower_hint.find("alpha_selection") != std::string::npos ||
        lower_hint.find("alphaselection") != std::string::npos ||
        lower_hint.find("sampler_masks") != std::string::npos ||
        (lower_hint.find("mask") != std::string::npos &&
         lower_hint.find("opacity") == std::string::npos &&
         lower_hint.find("cutout") == std::string::npos)) {
        return -1;
    }

    std::string base_no_ext = fs::path(lower_fn).stem().string();
    size_t last_us = base_no_ext.rfind('_');
    std::string suffix = (last_us != std::string::npos)
                             ? base_no_ext.substr(last_us + 1) : "";
    bool orm = (suffix == "orm" ||
                lower_fn.find("_orm") != std::string::npos ||
                lower_fn.find("orm.") != std::string::npos ||
                lower_hint.find("mergemap") != std::string::npos);
    if (orm) {
        if (is_orm) *is_orm = true;
        return -1;
    }

    if (suffix == "d" || suffix == "c" || suffix == "diff" || suffix == "diffuse" ||
        suffix == "albedo" || suffix == "basecolor" ||
        lower_fn.find("_c.") != std::string::npos ||
        lower_fn.find("albedo") != std::string::npos ||
        lower_fn.find("diffuse") != std::string::npos ||
        lower_fn.find("basecolor") != std::string::npos ||
        lower_fn.find("base_color") != std::string::npos ||
        lower_hint.find("diffuse") != std::string::npos ||
        lower_hint.find("albedo") != std::string::npos ||
        lower_hint.find("basecolor") != std::string::npos) {
        return TEX_DIFFUSE_COLOR;
    }
    if (suffix == "n" || suffix == "nrm" || suffix == "norm" ||
        lower_fn.find("normal") != std::string::npos ||
        lower_hint.find("normal") != std::string::npos) {
        return TEX_NORMAL;
    }
    if (suffix == "r" || suffix == "rgh" ||
        lower_fn.find("rough") != std::string::npos ||
        lower_hint.find("rough") != std::string::npos) {
        return TEX_ROUGHNESS;
    }
    const bool has_metal_hint =
        lower_fn.find("metal") != std::string::npos ||
        lower_hint.find("metal") != std::string::npos ||
        lower_hint.find("metalness") != std::string::npos;
    if (suffix == "m" && !has_metal_hint) {
        return -1;
    }
    if (suffix == "m" || suffix == "met" ||
        lower_fn.find("metal") != std::string::npos ||
        lower_hint.find("metal") != std::string::npos) {
        return TEX_METALLIC;
    }
    if (suffix == "e" || suffix == "emi" ||
        lower_fn.find("emissive") != std::string::npos ||
        lower_fn.find("emission") != std::string::npos ||
        lower_hint.find("emissive") != std::string::npos ||
        lower_hint.find("emission") != std::string::npos) {
        return TEX_EMISSIVE_COLOR;
    }
    if (suffix == "ao" || suffix == "occ" ||
        lower_fn.find("occlusion") != std::string::npos ||
        lower_fn.find("_ao") != std::string::npos ||
        lower_hint.find("occlusion") != std::string::npos) {
        return TEX_OCCLUSION;
    }
    if (suffix == "o" || suffix == "op" || suffix == "a" ||
        lower_fn.find("opacity") != std::string::npos ||
        lower_fn.find("alpha") != std::string::npos ||
        lower_hint.find("opacity") != std::string::npos) {
        return TEX_OPACITY;
    }

    return -1;
}

static void propagate_udim_scale(MaterialParams* params,
                                 const MaterialTexture& tex)
{
    if (tex.udim_cols > 0 && tex.udim_cols > (int)params->udim_scale_u)
        params->udim_scale_u = (float)tex.udim_cols;
    if (tex.udim_rows > 0 && tex.udim_rows > (int)params->udim_scale_v)
        params->udim_scale_v = (float)tex.udim_rows;
}

static std::string lower_ascii(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static std::string trim_ascii(const std::string& s)
{
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char)s[b])) b++;
    size_t e = s.size();
    while (e > b && std::isspace((unsigned char)s[e - 1])) e--;
    return s.substr(b, e - b);
}

static bool has_image_extension(const std::string& path)
{
    std::string l = lower_ascii(path);
    return l.find(".png")  != std::string::npos ||
           l.find(".jpg")  != std::string::npos ||
           l.find(".jpeg") != std::string::npos ||
           l.find(".exr")  != std::string::npos ||
           l.find(".tga")  != std::string::npos ||
           l.find(".bmp")  != std::string::npos ||
           l.find(".hdr")  != std::string::npos ||
           l.find("<udim>") != std::string::npos;
}

static float srgb_u8_to_linear(unsigned char v)
{
    float c = (float)v / 255.0f;
    if (c <= 0.04045f) return c / 12.92f;
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

static unsigned char linear_to_srgb_u8(float v)
{
    v = std::max(0.0f, std::min(1.0f, v));
    float c = (v <= 0.0031308f)
                  ? (12.92f * v)
                  : (1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f);
    int q = (int)std::lround(std::max(0.0f, std::min(1.0f, c)) * 255.0f);
    return (unsigned char)std::max(0, std::min(255, q));
}

static int bake_mdl_masked_albedo_texture(const std::string& albedo_ref,
                                          const std::string& mask_ref,
                                          const float color_albedo[3],
                                          const char* anchor_dir,
                                          NanousdStage stage,
                                          std::vector<std::string>& tex_paths,
                                          std::vector<MaterialTexture>& textures)
{
    if (!has_image_extension(albedo_ref) || !has_image_extension(mask_ref))
        return -1;
    NuBakeTimer _bt;

    std::ostringstream key_ss;
    key_ss << "mdl_baked_masked_albedo:"
           << albedo_ref << "|"
           << mask_ref << "|"
           << color_albedo[0] << ","
           << color_albedo[1] << ","
           << color_albedo[2];
    std::string key = key_ss.str();
    for (int i = 0; i < (int)tex_paths.size(); ++i) {
        if (tex_paths[i] == key) return i;
    }

    int albedo_idx = load_texture_dedup(albedo_ref.c_str(), anchor_dir,
                                        tex_paths, textures, stage);
    int mask_idx = load_texture_dedup(mask_ref.c_str(), anchor_dir,
                                      tex_paths, textures, stage);
    if (albedo_idx < 0 || mask_idx < 0) return -1;
    if (albedo_idx >= (int)textures.size() ||
        mask_idx >= (int)textures.size()) return -1;

    const MaterialTexture& albedo = textures[albedo_idx];
    const MaterialTexture& mask = textures[mask_idx];
    if (!albedo.pixels || !mask.pixels ||
        albedo.width <= 0 || albedo.height <= 0 ||
        mask.width <= 0 || mask.height <= 0) {
        return -1;
    }

    MaterialTexture baked = {};
    baked.width = albedo.width;
    baked.height = albedo.height;
    baked.pixels = (unsigned char*)malloc((size_t)baked.width *
                                          (size_t)baked.height * 4);
    if (!baked.pixels) return -1;

    // Each y-row writes a distinct output span from read-only inputs → safe to
    // parallelise; output is bit-identical to the serial path. (~half the cold
    // material-load cost is these bakes — measured 8.5s / 29 bakes.)
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < baked.height; ++y) {
        int my = std::min(mask.height - 1,
                          (int)((int64_t)y * mask.height / baked.height));
        for (int x = 0; x < baked.width; ++x) {
            int mx = std::min(mask.width - 1,
                              (int)((int64_t)x * mask.width / baked.width));
            size_t ai = ((size_t)y * baked.width + x) * 4;
            size_t mi = ((size_t)my * mask.width + mx) * 4;
            for (int c = 0; c < 3; ++c) {
                float tex_c = srgb_u8_to_linear(albedo.pixels[ai + c]);
                float mask_c = (float)mask.pixels[mi + c] / 255.0f;
                float out_c = color_albedo[c] * (1.0f - mask_c) +
                              tex_c * mask_c;
                baked.pixels[ai + c] = linear_to_srgb_u8(out_c);
            }
            baked.pixels[ai + 3] = albedo.pixels[ai + 3];
        }
    }

    snprintf(baked.path, sizeof(baked.path), "%s", key.c_str());
    int idx = (int)textures.size();
    textures.push_back(baked);
    tex_paths.push_back(key);
    fprintf(stderr, "material:   MDL baked masked albedo: %s + %s -> diffuse slot\n",
            albedo_ref.c_str(), mask_ref.c_str());
    return idx;
}

static int bake_mdl_body_masked_albedo_texture(
    const std::string& albedo_ref,
    const std::string& mask_ref,
    const float body_color[3],
    const float handle_color[3],
    const float cap_color[3],
    const char* anchor_dir,
    NanousdStage stage,
    std::vector<std::string>& tex_paths,
    std::vector<MaterialTexture>& textures)
{
    if (!has_image_extension(albedo_ref) || !has_image_extension(mask_ref))
        return -1;

    std::ostringstream key_ss;
    key_ss << "mdl_baked_body_masked_albedo:"
           << albedo_ref << "|"
           << mask_ref << "|"
           << body_color[0] << "," << body_color[1] << "," << body_color[2] << "|"
           << handle_color[0] << "," << handle_color[1] << "," << handle_color[2] << "|"
           << cap_color[0] << "," << cap_color[1] << "," << cap_color[2];
    std::string key = key_ss.str();
    for (int i = 0; i < (int)tex_paths.size(); ++i) {
        if (tex_paths[i] == key) return i;
    }

    int albedo_idx = load_texture_dedup(albedo_ref.c_str(), anchor_dir,
                                        tex_paths, textures, stage);
    int mask_idx = load_texture_dedup(mask_ref.c_str(), anchor_dir,
                                      tex_paths, textures, stage);
    if (albedo_idx < 0 || mask_idx < 0) return -1;
    if (albedo_idx >= (int)textures.size() ||
        mask_idx >= (int)textures.size()) return -1;

    const MaterialTexture& albedo = textures[albedo_idx];
    const MaterialTexture& mask = textures[mask_idx];
    if (!albedo.pixels || !mask.pixels ||
        albedo.width <= 0 || albedo.height <= 0 ||
        mask.width <= 0 || mask.height <= 0) {
        return -1;
    }

    MaterialTexture baked = {};
    baked.width = albedo.width;
    baked.height = albedo.height;
    baked.pixels = (unsigned char*)malloc((size_t)baked.width *
                                          (size_t)baked.height * 4);
    if (!baked.pixels) return -1;

    // Race-free per-row bake (see masked-albedo variant above).
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < baked.height; ++y) {
        int my = std::min(mask.height - 1,
                          (int)((int64_t)y * mask.height / baked.height));
        for (int x = 0; x < baked.width; ++x) {
            int mx = std::min(mask.width - 1,
                              (int)((int64_t)x * mask.width / baked.width));
            size_t ai = ((size_t)y * baked.width + x) * 4;
            size_t mi = ((size_t)my * mask.width + mx) * 4;

            float r = (float)mask.pixels[mi + 0] / 255.0f;
            float g = (float)mask.pixels[mi + 1] / 255.0f;
            float b = (float)mask.pixels[mi + 2] / 255.0f;
            float a = (float)mask.pixels[mi + 3] / 255.0f;

            for (int c = 0; c < 3; ++c) {
                float body_mix = body_color[c] * r;
                float handle_mix = body_mix * (1.0f - g) + handle_color[c] * g;
                float cap_mix = handle_mix * (1.0f - b) + cap_color[c] * b;
                float tex_c = srgb_u8_to_linear(albedo.pixels[ai + c]);
                float out_c = cap_mix * (1.0f - a) + tex_c * a;
                baked.pixels[ai + c] = linear_to_srgb_u8(out_c);
            }
            baked.pixels[ai + 3] = albedo.pixels[ai + 3];
        }
    }

    snprintf(baked.path, sizeof(baked.path), "%s", key.c_str());
    int idx = (int)textures.size();
    textures.push_back(baked);
    tex_paths.push_back(key);
    fprintf(stderr,
            "material:   MDL baked body-mask albedo: %s + %s -> diffuse slot\n",
            albedo_ref.c_str(), mask_ref.c_str());
    return idx;
}

static int bake_mdl_emissive_product_texture(
    const std::string& color_ref,
    const std::string& mask_ref,
    float scale,
    const char* anchor_dir,
    NanousdStage stage,
    std::vector<std::string>& tex_paths,
    std::vector<MaterialTexture>& textures)
{
    if (!has_image_extension(color_ref) || !has_image_extension(mask_ref))
        return -1;

    std::ostringstream key_ss;
    key_ss << "mdl_baked_emissive_product:"
           << color_ref << "|" << mask_ref << "|" << scale;
    std::string key = key_ss.str();
    for (int i = 0; i < (int)tex_paths.size(); ++i) {
        if (tex_paths[i] == key) return i;
    }

    int color_idx = load_texture_dedup(color_ref.c_str(), anchor_dir,
                                       tex_paths, textures, stage);
    int mask_idx = load_texture_dedup(mask_ref.c_str(), anchor_dir,
                                      tex_paths, textures, stage);
    if (color_idx < 0 || mask_idx < 0) return -1;
    if (color_idx >= (int)textures.size() ||
        mask_idx >= (int)textures.size()) return -1;

    const MaterialTexture& color = textures[color_idx];
    const MaterialTexture& mask = textures[mask_idx];
    if (!color.pixels || !mask.pixels ||
        color.width <= 0 || color.height <= 0 ||
        mask.width <= 0 || mask.height <= 0) {
        return -1;
    }

    MaterialTexture baked = {};
    baked.width = color.width;
    baked.height = color.height;
    baked.pixels = (unsigned char*)malloc((size_t)baked.width *
                                          (size_t)baked.height * 4);
    if (!baked.pixels) return -1;

    scale = std::max(scale, 0.0f);
    for (int y = 0; y < baked.height; ++y) {
        int my = std::min(mask.height - 1,
                          (int)((int64_t)y * mask.height / baked.height));
        for (int x = 0; x < baked.width; ++x) {
            int mx = std::min(mask.width - 1,
                              (int)((int64_t)x * mask.width / baked.width));
            size_t ci = ((size_t)y * baked.width + x) * 4;
            size_t mi = ((size_t)my * mask.width + mx) * 4;
            for (int c = 0; c < 3; ++c) {
                float color_c = srgb_u8_to_linear(color.pixels[ci + c]);
                float mask_c = (float)mask.pixels[mi + c] / 255.0f;
                baked.pixels[ci + c] = linear_to_srgb_u8(color_c * mask_c * scale);
            }
            baked.pixels[ci + 3] = 255;
        }
    }

    snprintf(baked.path, sizeof(baked.path), "%s", key.c_str());
    int idx = (int)textures.size();
    textures.push_back(baked);
    tex_paths.push_back(key);
    fprintf(stderr,
            "material:   MDL baked emissive product: %s * %s -> emissive slot\n",
            color_ref.c_str(), mask_ref.c_str());
    return idx;
}

static int wrap_texel_index(float t, int dim)
{
    if (dim <= 0) return 0;
    float f = t - std::floor(t);
    int i = (int)std::floor(f * (float)dim);
    if (i < 0) i = 0;
    if (i >= dim) i = dim - 1;
    return i;
}

static int bake_mdl_plastic_wrap_albedo_texture(
    const std::string& box_ref,
    const std::string& plastic_ref,
    const std::string& plastic_normal_ref,
    const float box_tint[3],
    const float plastic_tint[3],
    float plastic_opacity,
    float u_tiling,
    float v_tiling,
    const char* anchor_dir,
    NanousdStage stage,
    std::vector<std::string>& tex_paths,
    std::vector<MaterialTexture>& textures)
{
    if (!has_image_extension(box_ref) || !has_image_extension(plastic_ref))
        return -1;

    std::ostringstream key_ss;
    key_ss << "mdl_baked_plastic_wrap_albedo:"
           << box_ref << "|"
           << plastic_ref << "|"
           << plastic_normal_ref << "|"
           << box_tint[0] << "," << box_tint[1] << "," << box_tint[2] << "|"
           << plastic_tint[0] << "," << plastic_tint[1] << ","
           << plastic_tint[2] << "|"
           << plastic_opacity << "|"
           << u_tiling << "," << v_tiling;
    std::string key = key_ss.str();
    for (int i = 0; i < (int)tex_paths.size(); ++i) {
        if (tex_paths[i] == key) return i;
    }

    int box_idx = load_texture_dedup(box_ref.c_str(), anchor_dir,
                                     tex_paths, textures, stage);
    int plastic_idx = load_texture_dedup(plastic_ref.c_str(), anchor_dir,
                                         tex_paths, textures, stage);
    int normal_idx = -1;
    if (has_image_extension(plastic_normal_ref)) {
        normal_idx = load_texture_dedup(plastic_normal_ref.c_str(), anchor_dir,
                                        tex_paths, textures, stage);
    }
    if (box_idx < 0 || plastic_idx < 0) return -1;
    if (box_idx >= (int)textures.size() ||
        plastic_idx >= (int)textures.size()) return -1;

    const MaterialTexture& box = textures[box_idx];
    const MaterialTexture& plastic = textures[plastic_idx];
    const MaterialTexture* normal = nullptr;
    if (normal_idx >= 0 && normal_idx < (int)textures.size())
        normal = &textures[normal_idx];
    if (!box.pixels || !plastic.pixels ||
        box.width <= 0 || box.height <= 0 ||
        plastic.width <= 0 || plastic.height <= 0) {
        return -1;
    }

    MaterialTexture baked = {};
    baked.width = box.width;
    baked.height = box.height;
    baked.pixels = (unsigned char*)malloc((size_t)baked.width *
                                          (size_t)baked.height * 4);
    if (!baked.pixels) return -1;

    u_tiling = std::max(u_tiling, 1.0e-6f);
    v_tiling = std::max(v_tiling, 1.0e-6f);
    plastic_opacity = std::clamp(plastic_opacity, 0.0f, 1.0f);

    for (int y = 0; y < baked.height; ++y) {
        float v = ((float)y + 0.5f) / (float)baked.height;
        int py = wrap_texel_index(v * v_tiling, plastic.height);
        int ny = normal ? wrap_texel_index(v * v_tiling, normal->height) : 0;
        for (int x = 0; x < baked.width; ++x) {
            float u = ((float)x + 0.5f) / (float)baked.width;
            int px = wrap_texel_index(u * u_tiling, plastic.width);
            int nx = normal ? wrap_texel_index(u * u_tiling, normal->width) : 0;

            size_t bi = ((size_t)y * baked.width + x) * 4;
            size_t pi = ((size_t)py * plastic.width + px) * 4;
            float blend = plastic_opacity;
            if (normal && normal->pixels &&
                normal->width > 0 && normal->height > 0) {
                size_t ni = ((size_t)ny * normal->width + nx) * 4;
                float normal_z = 2.0f * ((float)normal->pixels[ni + 2] / 255.0f) - 1.0f;
                blend = std::clamp(plastic_opacity + (1.0f - normal_z),
                                   0.0f, 1.0f);
            }

            for (int c = 0; c < 3; ++c) {
                float box_c = srgb_u8_to_linear(box.pixels[bi + c]) *
                              box_tint[c];
                float plastic_c = srgb_u8_to_linear(plastic.pixels[pi + c]) *
                                  plastic_tint[c];
                float out_c = box_c * (1.0f - blend) + plastic_c * blend;
                baked.pixels[bi + c] = linear_to_srgb_u8(out_c);
            }
            baked.pixels[bi + 3] = box.pixels[bi + 3];
        }
    }

    snprintf(baked.path, sizeof(baked.path), "%s", key.c_str());
    int idx = (int)textures.size();
    textures.push_back(baked);
    tex_paths.push_back(key);
    fprintf(stderr,
            "material:   MDL baked plastic-wrap albedo: %s + %s -> diffuse slot\n",
            box_ref.c_str(), plastic_ref.c_str());
    return idx;
}

static bool material_uses_mdl_baked_masked_albedo(
    const MaterialParams* params,
    const std::vector<MaterialTexture>& textures)
{
    if (!params) return false;
    int idx = params->tex_indices[TEX_DIFFUSE_COLOR];
    if (idx < 0 || idx >= (int)textures.size()) return false;
    return strncmp(textures[idx].path, "mdl_baked_masked_albedo:",
                   strlen("mdl_baked_masked_albedo:")) == 0;
}

static bool material_uses_mdl_baked_body_masked_albedo(
    const MaterialParams* params,
    const std::vector<MaterialTexture>& textures)
{
    if (!params) return false;
    int idx = params->tex_indices[TEX_DIFFUSE_COLOR];
    if (idx < 0 || idx >= (int)textures.size()) return false;
    return strncmp(textures[idx].path, "mdl_baked_body_masked_albedo:",
                   strlen("mdl_baked_body_masked_albedo:")) == 0;
}

static bool material_uses_mdl_baked_plastic_wrap_albedo(
    const MaterialParams* params,
    const std::vector<MaterialTexture>& textures)
{
    if (!params) return false;
    int idx = params->tex_indices[TEX_DIFFUSE_COLOR];
    if (idx < 0 || idx >= (int)textures.size()) return false;
    return strncmp(textures[idx].path, "mdl_baked_plastic_wrap_albedo:",
                   strlen("mdl_baked_plastic_wrap_albedo:")) == 0;
}

static std::string mdl_material_name_from_subidentifier(const char* sub)
{
    if (!sub || !sub[0]) return std::string();
    std::string s = trim_ascii(sub);
    size_t paren = s.find('(');
    if (paren != std::string::npos) s = s.substr(0, paren);
    size_t colon = s.rfind("::");
    if (colon != std::string::npos) s = s.substr(colon + 2);
    size_t slash = s.find_last_of("/\\");
    if (slash != std::string::npos) s = s.substr(slash + 1);
    return trim_ascii(s);
}

static bool is_ident_char(char c)
{
    return std::isalnum((unsigned char)c) || c == '_';
}

static bool find_matching_paren(const std::string& s, size_t open, size_t* close_out)
{
    bool in_string = false;
    bool escape = false;
    int depth = 0;
    for (size_t i = open; i < s.size(); ++i) {
        char c = s[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '(') {
            depth++;
        } else if (c == ')') {
            depth--;
            if (depth == 0) {
                if (close_out) *close_out = i;
                return true;
            }
        }
    }
    return false;
}

static std::string find_mdl_export_material_params(const std::string& body,
                                                   const std::string& material_name,
                                                   std::string* source_out = nullptr)
{
    const std::string key = "export material";
    size_t first_open = std::string::npos;
    size_t first_close = std::string::npos;
    size_t first_section_begin = std::string::npos;
    size_t first_section_end = std::string::npos;

    for (size_t pos = 0; (pos = body.find(key, pos)) != std::string::npos;
         pos += key.size()) {
        size_t name_pos = pos + key.size();
        while (name_pos < body.size() &&
               std::isspace((unsigned char)body[name_pos])) name_pos++;
        if (name_pos >= body.size() ||
            !(std::isalpha((unsigned char)body[name_pos]) || body[name_pos] == '_'))
            continue;

        size_t name_end = name_pos + 1;
        while (name_end < body.size() && is_ident_char(body[name_end]))
            name_end++;
        std::string found_name = body.substr(name_pos, name_end - name_pos);
        size_t open = body.find('(', name_end);
        if (open == std::string::npos) continue;
        size_t close = std::string::npos;
        if (!find_matching_paren(body, open, &close)) continue;
        size_t section_end = body.find(key, close + 1);
        if (section_end == std::string::npos) section_end = body.size();

        if (first_open == std::string::npos) {
            first_open = open;
            first_close = close;
            first_section_begin = pos;
            first_section_end = section_end;
        }
        if (material_name.empty() || found_name == material_name) {
            if (source_out)
                *source_out = body.substr(pos, section_end - pos);
            return body.substr(open + 1, close - open - 1);
        }
    }

    if (first_open != std::string::npos) {
        if (source_out)
            *source_out = body.substr(first_section_begin,
                                      first_section_end - first_section_begin);
        return body.substr(first_open + 1, first_close - first_open - 1);
    }
    if (source_out) source_out->clear();
    return std::string();
}

static std::vector<std::string> split_top_level_commas(const std::string& text)
{
    std::vector<std::string> out;
    bool in_string = false;
    bool escape = false;
    int paren = 0, bracket = 0, brace = 0;
    size_t start = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (in_string) {
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') { in_string = true; continue; }
        if (c == '(') paren++;
        else if (c == ')' && paren > 0) paren--;
        else if (c == '[') bracket++;
        else if (c == ']' && bracket > 0) bracket--;
        else if (c == '{') brace++;
        else if (c == '}' && brace > 0) brace--;
        else if (c == ',' && paren == 0 && bracket == 0 && brace == 0) {
            out.push_back(trim_ascii(text.substr(start, i - start)));
            start = i + 1;
        }
    }
    std::string tail = trim_ascii(text.substr(start));
    if (!tail.empty()) out.push_back(tail);
    return out;
}

static std::string mdl_param_name_before_equal(const std::string& param)
{
    size_t eq = param.find('=');
    if (eq == std::string::npos) return std::string();
    size_t end = eq;
    while (end > 0 && std::isspace((unsigned char)param[end - 1])) end--;
    size_t start = end;
    while (start > 0 && is_ident_char(param[start - 1])) start--;
    if (start == end) return std::string();
    return param.substr(start, end - start);
}

static bool parse_first_quoted_after(const std::string& s, size_t pos,
                                     std::string* out)
{
    size_t q1 = s.find('"', pos);
    if (q1 == std::string::npos) return false;
    size_t q2 = s.find('"', q1 + 1);
    if (q2 == std::string::npos || q2 == q1 + 1) return false;
    if (out) *out = s.substr(q1 + 1, q2 - q1 - 1);
    return true;
}

static bool parse_mdl_texture_default(const std::string& param, std::string* out)
{
    size_t eq = param.find('=');
    if (eq == std::string::npos) return false;
    size_t tex = param.find("texture_2d", eq + 1);
    if (tex != std::string::npos)
        return parse_first_quoted_after(param, tex, out);
    size_t q = param.find('"', eq + 1);
    if (q != std::string::npos)
        return parse_first_quoted_after(param, q, out);
    return false;
}

static bool parse_mdl_color_default(const std::string& param, float out[3])
{
    size_t eq = param.find('=');
    if (eq == std::string::npos) return false;
    size_t cpos = param.find("color", eq + 1);
    if (cpos == std::string::npos) return false;
    size_t open = param.find('(', cpos);
    if (open == std::string::npos) return false;

    const char* p = param.c_str() + open + 1;
    char* endp = nullptr;
    float vals[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; ++i) {
        while (*p && (std::isspace((unsigned char)*p) || *p == ',')) p++;
        if (i > 0 && *p == ')') {
            vals[i] = vals[0];
            if (i == 1) vals[2] = vals[0];
            break;
        }
        vals[i] = std::strtof(p, &endp);
        if (endp == p) return false;
        p = endp;
    }
    out[0] = vals[0];
    out[1] = vals[1];
    out[2] = vals[2];
    return true;
}

static bool parse_mdl_float4_default(const std::string& param, float out[4])
{
    size_t eq = param.find('=');
    if (eq == std::string::npos) return false;
    size_t fpos = param.find("float4", eq + 1);
    if (fpos == std::string::npos) return false;
    size_t open = param.find('(', fpos);
    if (open == std::string::npos) return false;

    const char* p = param.c_str() + open + 1;
    char* endp = nullptr;
    for (int i = 0; i < 4; ++i) {
        while (*p && (std::isspace((unsigned char)*p) || *p == ',')) p++;
        out[i] = std::strtof(p, &endp);
        if (endp == p) return false;
        p = endp;
    }
    return true;
}

static bool parse_mdl_float_default(const std::string& param, float* out)
{
    size_t eq = param.find('=');
    if (eq == std::string::npos || !out) return false;
    const char* p = param.c_str() + eq + 1;
    while (*p && !(std::isdigit((unsigned char)*p) ||
                   *p == '-' || *p == '+' || *p == '.')) {
        if (*p == '"' || *p == '{') return false;
        p++;
    }
    char* endp = nullptr;
    float v = std::strtof(p, &endp);
    if (endp == p) return false;
    *out = v;
    return true;
}

static bool mdl_name_has_any(const std::string& name,
                             std::initializer_list<const char*> keys)
{
    for (const char* k : keys) {
        if (name.find(k) != std::string::npos) return true;
    }
    return false;
}

static void apply_mdl_color_param(MaterialParams* params,
                                  const std::string& lname,
                                  const float c[3])
{
    if (mdl_name_has_any(lname, {"emiss", "emission"})) {
        params->emissive_color[0] = c[0];
        params->emissive_color[1] = c[1];
        params->emissive_color[2] = c[2];
        if (params->emissive_color[3] <= 0.0f)
            params->emissive_color[3] = 1.0f;
        return;
    }
    if (mdl_name_has_any(lname, {"transmission_color", "transmission_tint"})) {
        params->transmission_color[0] = c[0];
        params->transmission_color[1] = c[1];
        params->transmission_color[2] = c[2];
        return;
    }
    if (mdl_name_has_any(lname, {"subsurface_color", "subsurface_tint"})) {
        params->subsurface_color[0] = c[0];
        params->subsurface_color[1] = c[1];
        params->subsurface_color[2] = c[2];
        params->sss_color_authored = 1;
        return;
    }
    if (mdl_name_has_any(lname, {"specular_color", "specular_tint"})) {
        params->specular_color[0] = c[0];
        params->specular_color[1] = c[1];
        params->specular_color[2] = c[2];
        params->use_specular_workflow = 1;
        return;
    }
    if (mdl_name_has_any(lname, {"base", "diffuse", "albedo", "tint", "color"})) {
        params->base_color[0] = c[0];
        params->base_color[1] = c[1];
        params->base_color[2] = c[2];
    }
}

static void apply_mdl_float_param(MaterialParams* params,
                                  const std::string& lname,
                                  float v)
{
    if (lname.find("texture") != std::string::npos ||
        lname.find("image") != std::string::npos ||
        lname.find("map") != std::string::npos ||
        lname.find("influence") != std::string::npos ||
        lname.find("enable") != std::string::npos ||
        lname.find("flip_") != std::string::npos ||
        lname.find("tiling") != std::string::npos ||
        lname.find("rotation") != std::string::npos ||
        lname.find("desaturation") != std::string::npos)
        return;

    if (mdl_name_has_any(lname, {"roughnessmin", "roughness_min",
                                 "roughnessmax", "roughness_max",
                                 "u_tiling", "v_tiling",
                                 "plasticopacity", "plasticnormalalpha"}))
        return;

    if (mdl_name_has_any(lname, {"clearcoat_roughness", "coat_roughness"}))
        params->clearcoat_roughness = v;
    else if (mdl_name_has_any(lname, {"roughness", "roughtness"}))
        params->roughness = v;
    else if (mdl_name_has_any(lname, {"metallic", "metalness", "base_metalness"}))
        params->metallic = v;
    else if (mdl_name_has_any(lname, {"specular_mdl", "specular"}) &&
             !mdl_name_has_any(lname, {"transmission_ior", "specular_ior", "ior"})) {
        float f0 = std::clamp(v * 0.08f, 0.0f, 1.0f);
        params->specular_color[0] = f0;
        params->specular_color[1] = f0;
        params->specular_color[2] = f0;
        params->specular_color[3] = 1.0f;
        params->use_specular_workflow = 1;
    }
    else if (mdl_name_has_any(lname, {"opacity_threshold", "alpha_threshold", "cutout_threshold"}))
        params->opacity_threshold = v;
    else if (mdl_name_has_any(lname, {"opacity", "alpha", "cutout"}))
        params->opacity = v;
    else if (mdl_name_has_any(lname, {"clearcoat", "coat_weight"}))
        params->clearcoat = v;
    else if (mdl_name_has_any(lname, {"transmission_ior", "specular_ior"}))
        params->transmission_ior = v;
    else if (mdl_name_has_any(lname, {"ior"}))
        params->ior = v;
    else if (mdl_name_has_any(lname, {"normal_scale", "bump_factor", "bump_scale"}))
        params->normal_scale = v;
    else if (mdl_name_has_any(lname, {"transmission", "thin_walled"}))
        params->transmission_weight = v;
    else if (mdl_name_has_any(lname, {"subsurface"}))
        params->subsurface_weight = v;
    else if (mdl_name_has_any(lname, {"emission", "emissive", "luminance", "intensity"}))
        params->emissive_color[3] = v;
}

static void apply_authored_mdl_shader_inputs(NanousdPrim shader_prim,
                                             MaterialParams* params)
{
    if (!shader_prim || !params) return;
    int na = nanousd_nattribs(shader_prim);
    for (int a = 0; a < na; ++a) {
        const char* aname = nanousd_attribname(shader_prim, a);
        if (!aname || strncmp(aname, "inputs:", 7) != 0) continue;
        if (strstr(aname, ".connect")) continue;

        std::string lname = lower_ascii(aname + 7);
        float v4[4];
        float v3[3];
        if (nanousd_attribv4f(shader_prim, aname, v4)) {
            apply_mdl_color_param(params, lname, v4);
            continue;
        }
        if (nanousd_attribv3f(shader_prim, aname, v3)) {
            apply_mdl_color_param(params, lname, v3);
            continue;
        }

        int ok = 0;
        float f = nanousd_attribf(shader_prim, aname, &ok);
        if (ok) {
            apply_mdl_float_param(params, lname, f);
            continue;
        }
        int i = nanousd_attribi(shader_prim, aname, &ok);
        if (ok) {
            apply_mdl_float_param(params, lname, (float)i);
            continue;
        }
        int b = nanousd_attribb(shader_prim, aname, &ok);
        if (ok)
            apply_mdl_float_param(params, lname, b ? 1.0f : 0.0f);
    }
}

static void assign_mdl_texture(MaterialParams* params,
                               std::vector<std::string>& tex_paths,
                               std::vector<MaterialTexture>& textures,
                               const std::string& name_hint,
                               const std::string& tex_ref,
                               const char* anchor_dir,
                               NanousdStage stage)
{
    if (!has_image_extension(tex_ref)) return;

    std::string lname = lower_ascii(name_hint);
    if (lname.find("maskselection") != std::string::npos ||
        lname.find("alpha_selection") != std::string::npos ||
        lname.find("alphaselection") != std::string::npos ||
        lname.find("sampler_masks") != std::string::npos ||
        (lname.find("mask") != std::string::npos &&
         lname.find("opacity") == std::string::npos &&
         lname.find("cutout") == std::string::npos)) {
        return;
    }

    std::string lower_ref = lower_ascii(tex_ref);
    std::string base_no_ext = fs::path(lower_ref).stem().string();
    size_t last_us = base_no_ext.rfind('_');
    std::string suffix = (last_us != std::string::npos)
                             ? base_no_ext.substr(last_us + 1) : "";
    const bool ambiguous_m_suffix =
        suffix == "m" &&
        lower_ref.find("metal") == std::string::npos &&
        lname.find("metal") == std::string::npos &&
        lname.find("metalness") == std::string::npos;

    bool is_orm = false;
    int slot = classify_texture_slot(name_hint.c_str(), tex_ref.c_str(), &is_orm);

    if (mdl_name_has_any(lname, {"transmission"}) &&
        params->tex_transmission_weight < 0) {
        int idx = load_texture_dedup(tex_ref.c_str(), anchor_dir,
                                     tex_paths, textures, stage);
        if (idx >= 0) {
            params->tex_transmission_weight = idx;
            params->transmission_weight = 1.0f;
            propagate_udim_scale(params, textures[idx]);
        }
        return;
    }
    if (mdl_name_has_any(lname, {"subsurface"}) &&
        params->tex_subsurface_weight < 0) {
        int idx = load_texture_dedup(tex_ref.c_str(), anchor_dir,
                                     tex_paths, textures, stage);
        if (idx >= 0) {
            params->tex_subsurface_weight = idx;
            params->subsurface_weight = 1.0f;
            propagate_udim_scale(params, textures[idx]);
        }
        return;
    }

    if (slot < 0 && !is_orm && !ambiguous_m_suffix &&
        params->tex_indices[TEX_DIFFUSE_COLOR] < 0 &&
        mdl_name_has_any(lname, {"base", "diffuse", "albedo", "color", "tex", "texture"}))
        slot = TEX_DIFFUSE_COLOR;

    if (is_orm) {
        int idx = load_texture_dedup(tex_ref.c_str(), anchor_dir,
                                     tex_paths, textures, stage);
        if (idx >= 0) {
            if (params->tex_indices[TEX_OCCLUSION] < 0)
                params->tex_indices[TEX_OCCLUSION] = idx;
            if (params->tex_indices[TEX_ROUGHNESS] < 0)
                params->tex_indices[TEX_ROUGHNESS] = idx;
            if (params->tex_indices[TEX_METALLIC] < 0)
                params->tex_indices[TEX_METALLIC] = idx;
            propagate_udim_scale(params, textures[idx]);
        }
    } else if (slot >= 0 && params->tex_indices[slot] < 0) {
        int idx = load_texture_dedup(tex_ref.c_str(), anchor_dir,
                                     tex_paths, textures, stage);
        if (idx >= 0) {
            params->tex_indices[slot] = idx;
            propagate_udim_scale(params, textures[idx]);
        }
    }
}

static void collect_mdl_shader_inputs(NanousdPrim shader_prim,
                                      std::vector<NusdMdlInput>& inputs,
                                      std::vector<std::string>& names)
{
    if (!shader_prim) return;
    int na = nanousd_nattribs(shader_prim);
    inputs.reserve(inputs.size() + (size_t)na);
    names.reserve(names.size() + (size_t)na);
    for (int a = 0; a < na; ++a) {
        const char* aname = nanousd_attribname(shader_prim, a);
        if (!aname || strncmp(aname, "inputs:", 7) != 0) continue;
        if (strstr(aname, ".connect")) continue;
        const char* pname = aname + 7;
        if (!pname[0]) continue;

        NusdMdlInput input;
        memset(&input, 0, sizeof(input));

        float v4[4];
        float v3[3];
        int ok = 0;
        if (nanousd_attribv4f(shader_prim, aname, v4)) {
            input.kind = NUSD_MDL_INPUT_COLOR;
            input.values[0] = v4[0];
            input.values[1] = v4[1];
            input.values[2] = v4[2];
            input.values[3] = v4[3];
        } else if (nanousd_attribv3f(shader_prim, aname, v3)) {
            input.kind = NUSD_MDL_INPUT_COLOR;
            input.values[0] = v3[0];
            input.values[1] = v3[1];
            input.values[2] = v3[2];
            input.values[3] = 1.0f;
        } else {
            int b = nanousd_attribb(shader_prim, aname, &ok);
            if (ok) {
                input.kind = NUSD_MDL_INPUT_BOOL;
                input.int_value = b ? 1 : 0;
                input.values[0] = b ? 1.0f : 0.0f;
            } else {
                int i = nanousd_attribi(shader_prim, aname, &ok);
                if (ok) {
                    input.kind = NUSD_MDL_INPUT_INT;
                    input.int_value = i;
                    input.values[0] = (float)i;
                } else {
                    float f = nanousd_attribf(shader_prim, aname, &ok);
                    if (!ok) continue;
                    input.kind = NUSD_MDL_INPUT_FLOAT;
                    input.values[0] = f;
                }
            }
        }

        names.emplace_back(pname);
        inputs.push_back(input);
        inputs.back().name = names.back().c_str();
    }
}

static std::string find_mdl_texture_literal_containing(
    const std::string& body,
    const std::string& needle_lower)
{
    size_t pos = 0;
    const std::string key = "texture_2d";
    while ((pos = body.find(key, pos)) != std::string::npos) {
        std::string tex_ref;
        if (!parse_first_quoted_after(body, pos + key.size(), &tex_ref)) {
            pos += key.size();
            continue;
        }
        if (lower_ascii(tex_ref).find(needle_lower) != std::string::npos)
            return tex_ref;
        pos += key.size();
    }
    return std::string();
}

static std::string find_first_mdl_texture_literal_for_slot(
    const std::string& body,
    int wanted_slot)
{
    size_t pos = 0;
    const std::string key = "texture_2d";
    while ((pos = body.find(key, pos)) != std::string::npos) {
        std::string tex_ref;
        if (!parse_first_quoted_after(body, pos + key.size(), &tex_ref)) {
            pos += key.size();
            continue;
        }
        bool is_orm = false;
        int slot = classify_texture_slot(nullptr, tex_ref.c_str(), &is_orm);
        if (!is_orm && slot == wanted_slot)
            return tex_ref;
        pos += key.size();
    }
    return std::string();
}

static std::string find_mdl_texture_literal_with_any_keyword(
    const std::string& body,
    std::initializer_list<const char*> keywords)
{
    size_t pos = 0;
    const std::string key = "texture_2d";
    while ((pos = body.find(key, pos)) != std::string::npos) {
        std::string tex_ref;
        if (!parse_first_quoted_after(body, pos + key.size(), &tex_ref)) {
            pos += key.size();
            continue;
        }
        std::string lower_ref = lower_ascii(tex_ref);
        size_t hint_begin = body.rfind('\n', pos);
        hint_begin = (hint_begin == std::string::npos) ? 0 : hint_begin + 1;
        size_t hint_end = body.find('\n', pos);
        if (hint_end == std::string::npos) hint_end = body.size();
        std::string lower_context =
            lower_ref + " " +
            lower_ascii(body.substr(hint_begin, hint_end - hint_begin));
        for (const char* keyword : keywords) {
            if (lower_context.find(keyword) != std::string::npos)
                return tex_ref;
        }
        pos += key.size();
    }
    return std::string();
}

static bool mdl_source_has_nonzero_emissive_color(const std::string& source)
{
    std::string lower = lower_ascii(source);
    size_t pos = 0;
    while ((pos = lower.find("emissivecolor_mdl", pos)) != std::string::npos) {
        size_t semi = lower.find(';', pos);
        size_t eq = lower.find('=', pos);
        if (eq != std::string::npos &&
            (semi == std::string::npos || eq < semi)) {
            size_t end = (semi == std::string::npos) ? lower.size() : semi;
            std::string expr = lower.substr(eq + 1, end - eq - 1);
            expr.erase(std::remove_if(expr.begin(), expr.end(),
                                      [](unsigned char c) {
                                          return std::isspace(c) != 0;
                                      }),
                       expr.end());
            return expr != "float3(0.0,0.0,0.0)" &&
                   expr != "float3(0,0,0)" &&
                   expr != "color(0.0,0.0,0.0)" &&
                   expr != "color(0,0,0)";
        }
        pos += strlen("emissivecolor_mdl");
    }
    return false;
}

static bool apply_mdl_sdk_source_material(NanousdPrim shader_prim,
                                          MaterialParams* params,
                                          const char* scene_dir,
                                          NanousdStage stage,
                                          std::vector<std::string>& tex_paths,
                                          std::vector<MaterialTexture>& textures)
{
    if (!shader_prim || !params) return false;
    int ok = 0;
    const char* mdl_asset = nanousd_attribasset(shader_prim,
                                                "info:mdl:sourceAsset", &ok);
    if (!ok || !mdl_asset || !mdl_asset[0]) return false;
    std::string mdl_asset_value(mdl_asset);
    std::string mdl_resolved = resolve_texture_path(mdl_asset_value.c_str(),
                                                    scene_dir, stage);
    const char* sub = nanousd_attrib_token(shader_prim,
                                           "info:mdl:sourceAsset:subIdentifier",
                                           &ok);
    std::string sub_value = (ok && sub) ? std::string(sub) : std::string();

    std::vector<NusdMdlInput> inputs;
    std::vector<std::string> names;
    collect_mdl_shader_inputs(shader_prim, inputs, names);

    NusdMdlDecoded decoded;
    if (!nusd_mdl_bridge_decode_with_inputs(
            mdl_resolved.empty() ? mdl_asset_value.c_str() : mdl_resolved.c_str(),
            sub_value.empty() ? nullptr : sub_value.c_str(), scene_dir,
            inputs.empty() ? nullptr : inputs.data(),
            (int)inputs.size(), &decoded))
        return false;

    if (decoded.has_base_color) {
        params->base_color[0] = decoded.base_color[0];
        params->base_color[1] = decoded.base_color[1];
        params->base_color[2] = decoded.base_color[2];
        params->base_color[3] = decoded.base_color[3];
    }
    if (decoded.has_emissive_color) {
        params->emissive_color[0] = decoded.emissive_color[0];
        params->emissive_color[1] = decoded.emissive_color[1];
        params->emissive_color[2] = decoded.emissive_color[2];
        params->emissive_color[3] = decoded.emissive_color[3];
    }
    if (decoded.has_metallic) params->metallic = decoded.metallic;
    if (decoded.has_roughness) params->roughness = decoded.roughness;
    if (decoded.has_opacity) {
        params->opacity = decoded.opacity;
        params->base_color[3] = decoded.opacity;
    }
    if (decoded.has_ior) params->ior = decoded.ior;
    if (decoded.has_clearcoat) params->clearcoat = decoded.clearcoat;
    if (decoded.has_clearcoat_roughness)
        params->clearcoat_roughness = decoded.clearcoat_roughness;
    if (decoded.has_normal_scale) params->normal_scale = decoded.normal_scale;
    if (decoded.has_transmission_color) {
        params->transmission_color[0] = decoded.transmission_color[0];
        params->transmission_color[1] = decoded.transmission_color[1];
        params->transmission_color[2] = decoded.transmission_color[2];
        params->transmission_color[3] = decoded.transmission_color[3];
    }
    if (decoded.has_transmission_weight)
        params->transmission_weight = decoded.transmission_weight;
    if (decoded.has_transmission_ior)
        params->transmission_ior = decoded.transmission_ior;
    if (decoded.has_specular_color) {
        params->specular_color[0] = decoded.specular_color[0];
        params->specular_color[1] = decoded.specular_color[1];
        params->specular_color[2] = decoded.specular_color[2];
        params->specular_color[3] = decoded.specular_color[3];
    }
    if (decoded.has_specular_workflow)
        params->use_specular_workflow = decoded.use_specular_workflow;
    for (int i = 0; i < decoded.texture_count; ++i) {
        const NusdMdlDecodedTexture& tex = decoded.textures[i];
        const char* tex_ref = tex.file_path[0] ? tex.file_path : tex.db_name;
        const char* hint = tex.name_hint[0] ? tex.name_hint : tex_ref;
        if (tex_ref && tex_ref[0]) {
            assign_mdl_texture(params, tex_paths, textures, hint, tex_ref,
                               scene_dir, stage);
        }
    }
    return true;
}

static void apply_generic_mdl_source_material(NanousdPrim shader_prim,
                                              MaterialParams* params,
                                              std::vector<std::string>& tex_paths,
                                              std::vector<MaterialTexture>& textures,
                                              const char* scene_dir,
                                              NanousdStage stage,
                                              bool apply_constant_defaults)
{
    int ok = 0;
    const char* mdl_asset = nanousd_attribasset(shader_prim,
                                                 "info:mdl:sourceAsset", &ok);
    if (!ok || !mdl_asset || !mdl_asset[0]) return;
    std::string mdl_asset_path(mdl_asset);

    std::string mdl_resolved = resolve_texture_path(mdl_asset_path.c_str(),
                                                    scene_dir, stage);
    std::ifstream mdl_file(mdl_resolved);
    if (!mdl_file.good()) {
        if (getenv("NUSD_MAT_DIAG"))
            fprintf(stderr, "[mat_diag] MDL sourceAsset '%s' resolved to '%s' but could not be opened\n",
                    mdl_asset_path.c_str(), mdl_resolved.c_str());
        return;
    }

    std::stringstream buf;
    buf << mdl_file.rdbuf();
    std::string body = buf.str();
    std::string mdl_dir;
    try {
        mdl_dir = fs::path(mdl_resolved).parent_path().string();
    } catch (...) {}
    const char* anchor = mdl_dir.empty() ? scene_dir : mdl_dir.c_str();

    const char* sub = nanousd_attrib_token(shader_prim,
                                           "info:mdl:sourceAsset:subIdentifier",
                                           &ok);
    std::string material_name = (ok && sub)
                                    ? mdl_material_name_from_subidentifier(sub)
                                    : std::string();
    std::string material_source;
    std::string params_text =
        find_mdl_export_material_params(body, material_name, &material_source);
    const std::string& pattern_source =
        material_source.empty() ? body : material_source;
    if (getenv("NUSD_MAT_DIAG")) {
        const char* pp = nanousd_path(shader_prim);
        fprintf(stderr,
                "[mat_diag] generic MDL shader=%s asset=%s material=%s params=%s\n",
                pp ? pp : "?", mdl_asset_path.c_str(),
                material_name.empty() ? "<first>" : material_name.c_str(),
                params_text.empty() ? "no" : "yes");
    }

    bool has_color_albedo = false;
    float color_albedo[3] = {0.0f, 0.0f, 0.0f};
    bool has_body_color = false;
    bool has_handle_color = false;
    bool has_cap_color = false;
    float body_color[3] = {0.0f, 0.0f, 0.0f};
    float handle_color[3] = {0.0f, 0.0f, 0.0f};
    float cap_color[3] = {0.0f, 0.0f, 0.0f};
    std::string albedo_texture_ref;
    std::string mask_selection_ref;
    std::string box_albedo_ref;
    std::string plastic_albedo_ref;
    std::string plastic_normal_ref;
    std::string plastic_orm_ref;
    float box_tint[3] = {1.0f, 1.0f, 1.0f};
    float plastic_tint[3] = {1.0f, 1.0f, 1.0f};
    float plastic_opacity = 0.0f;
    float plastic_normal_alpha = 1.0f;
    float roughness_min = 0.0f;
    float roughness_max = 1.0f;
    float u_tiling = 1.0f;
    float v_tiling = 1.0f;
    float main_tiling[2] = {1.0f, 1.0f};
    float emissive_product_scale = 1.0f;
    bool has_box_tint = false;
    bool has_plastic_tint = false;
    bool has_plastic_opacity = false;
    bool has_plastic_normal_alpha = false;
    bool has_roughness_min = false;
    bool has_roughness_max = false;
    bool has_plastic_wrap = false;
    bool has_u_tiling = false;
    bool has_v_tiling = false;
    bool has_main_tiling = false;

    if (!params_text.empty()) {
        for (const std::string& raw : split_top_level_commas(params_text)) {
            std::string name = mdl_param_name_before_equal(raw);
            if (name.empty()) continue;
            std::string lname = lower_ascii(name);

            std::string tex_ref;
            if (parse_mdl_texture_default(raw, &tex_ref)) {
                if (lname.find("albedotexture") != std::string::npos ||
                    lname.find("basecolor_texture") != std::string::npos) {
                    albedo_texture_ref = tex_ref;
                } else if (lname.find("basecolor_box") != std::string::npos) {
                    box_albedo_ref = tex_ref;
                } else if (lname.find("basecolor_plastic") != std::string::npos) {
                    plastic_albedo_ref = tex_ref;
                } else if (lname.find("multimap_plastic") != std::string::npos) {
                    plastic_orm_ref = tex_ref;
                } else if (lname.find("normalmap_plastic") != std::string::npos) {
                    plastic_normal_ref = tex_ref;
                } else if (lname.find("maskselection") != std::string::npos ||
                           lname.find("alpha_selection") != std::string::npos ||
                           lname.find("alphaselection") != std::string::npos) {
                    mask_selection_ref = tex_ref;
                }
                assign_mdl_texture(params, tex_paths, textures, name, tex_ref,
                                   anchor, stage);
                continue;
            }

            float f4[4];
            if (parse_mdl_float4_default(raw, f4)) {
                if (lname == "maintiling") {
                    main_tiling[0] = f4[0];
                    main_tiling[1] = f4[1];
                    has_main_tiling = true;
                    continue;
                }
                if (lname == "body") {
                    body_color[0] = f4[0];
                    body_color[1] = f4[1];
                    body_color[2] = f4[2];
                    has_body_color = true;
                    continue;
                }
                if (lname == "handle") {
                    handle_color[0] = f4[0];
                    handle_color[1] = f4[1];
                    handle_color[2] = f4[2];
                    has_handle_color = true;
                    continue;
                }
                if (lname == "cap") {
                    cap_color[0] = f4[0];
                    cap_color[1] = f4[1];
                    cap_color[2] = f4[2];
                    has_cap_color = true;
                    continue;
                }
                if (lname.find("basecolorbox_tint") != std::string::npos) {
                    box_tint[0] = f4[0];
                    box_tint[1] = f4[1];
                    box_tint[2] = f4[2];
                    has_box_tint = true;
                    if (apply_constant_defaults)
                        apply_mdl_color_param(params, lname, box_tint);
                    continue;
                }
                if (lname.find("basecolorplastic_tint") != std::string::npos) {
                    plastic_tint[0] = f4[0];
                    plastic_tint[1] = f4[1];
                    plastic_tint[2] = f4[2];
                    has_plastic_tint = true;
                    continue;
                }
                if (lname.find("coloralbedo") != std::string::npos ||
                    lname.find("basecolor") != std::string::npos ||
                    lname.find("diffusecolor") != std::string::npos) {
                    color_albedo[0] = f4[0];
                    color_albedo[1] = f4[1];
                    color_albedo[2] = f4[2];
                    has_color_albedo = true;
                    if (apply_constant_defaults)
                        apply_mdl_color_param(params, lname, color_albedo);
                    continue;
                }
            }

            float c[3];
            if (apply_constant_defaults && parse_mdl_color_default(raw, c)) {
                apply_mdl_color_param(params, lname, c);
                continue;
            }

            float v = 0.0f;
            if (parse_mdl_float_default(raw, &v)) {
                if (lname == "u_tiling") {
                    u_tiling = v;
                    has_u_tiling = true;
                    continue;
                }
                if (lname == "v_tiling") {
                    v_tiling = v;
                    has_v_tiling = true;
                    continue;
                }
                if (lname == "plasticopacity") {
                    plastic_opacity = v;
                    has_plastic_opacity = true;
                    continue;
                }
                if (lname == "plasticnormalalpha") {
                    plastic_normal_alpha = v;
                    has_plastic_normal_alpha = true;
                    continue;
                }
                if (lname == "roughnessmin" || lname == "roughness_min") {
                    roughness_min = v;
                    has_roughness_min = true;
                    continue;
                }
                if (lname == "roughnessmax" || lname == "roughness_max") {
                    roughness_max = v;
                    has_roughness_max = true;
                    continue;
                }
                if (lname == "param") {
                    emissive_product_scale = v;
                    continue;
                }
                if (apply_constant_defaults)
                    apply_mdl_float_param(params, lname, v);
            }
        }

        if (!box_albedo_ref.empty() && !plastic_albedo_ref.empty()) {
            if (plastic_normal_ref.empty())
                plastic_normal_ref =
                    find_mdl_texture_literal_containing(pattern_source,
                                                        "plasticwrap_n");
            if (plastic_orm_ref.empty())
                plastic_orm_ref =
                    find_mdl_texture_literal_containing(pattern_source,
                                                        "plasticwrap_orm");
            if (!has_box_tint) {
                box_tint[0] = box_tint[1] = box_tint[2] = 1.0f;
            }
            if (!has_plastic_tint) {
                plastic_tint[0] = plastic_tint[1] = plastic_tint[2] = 1.0f;
            }
            if (!has_plastic_opacity)
                plastic_opacity = 0.0f;
            has_plastic_wrap = true;
        }

        if (has_roughness_min || has_roughness_max) {
            float rmin = has_roughness_min ? roughness_min : 0.0f;
            float rmax = has_roughness_max ? roughness_max : rmin;
            if (rmax < rmin) {
                float tmp = rmin;
                rmin = rmax;
                rmax = tmp;
            }
            params->roughness = 0.5f * (rmin + rmax);
            params->roughness_tex_bias = rmin;
            params->roughness_tex_scale = rmax - rmin;
        }

        if (has_main_tiling) {
            set_mdl_y_flipped_uv_scale(params, main_tiling[0], main_tiling[1]);
        } else if (!has_plastic_wrap && (has_u_tiling || has_v_tiling)) {
            set_mdl_y_flipped_uv_scale(params, u_tiling, v_tiling);
        }

        int authored_ok = 0;
        const char* authored_tex = nanousd_attribasset(shader_prim,
                                                       "inputs:AlbedoTexture",
                                                       &authored_ok);
        if (!authored_ok || !authored_tex || !authored_tex[0])
            authored_tex = nanousd_attribs(shader_prim, "inputs:AlbedoTexture",
                                           &authored_ok);
        if (authored_ok && authored_tex && authored_tex[0])
            albedo_texture_ref = authored_tex;

        authored_ok = 0;
        authored_tex = nanousd_attribasset(shader_prim,
                                           "inputs:MaskSelection",
                                           &authored_ok);
        if (!authored_ok || !authored_tex || !authored_tex[0])
            authored_tex = nanousd_attribs(shader_prim, "inputs:MaskSelection",
                                           &authored_ok);
        if (authored_ok && authored_tex && authored_tex[0])
            mask_selection_ref = authored_tex;

        float authored_color4[4];
        float authored_color3[3];
        if (nanousd_attribv4f(shader_prim, "inputs:ColorAlbedo",
                              authored_color4) ||
            nanousd_attribv4f(shader_prim, "inputs:BaseColor_Tint",
                              authored_color4)) {
            color_albedo[0] = authored_color4[0];
            color_albedo[1] = authored_color4[1];
            color_albedo[2] = authored_color4[2];
            has_color_albedo = true;
        } else if (nanousd_attribv3f(shader_prim, "inputs:ColorAlbedo",
                                     authored_color3) ||
                   nanousd_attribv3f(shader_prim, "inputs:BaseColor_Tint",
                                     authored_color3)) {
            color_albedo[0] = authored_color3[0];
            color_albedo[1] = authored_color3[1];
            color_albedo[2] = authored_color3[2];
            has_color_albedo = true;
        }

        if (nanousd_attribv4f(shader_prim, "inputs:Body", authored_color4)) {
            body_color[0] = authored_color4[0];
            body_color[1] = authored_color4[1];
            body_color[2] = authored_color4[2];
            has_body_color = true;
        } else if (nanousd_attribv3f(shader_prim, "inputs:Body",
                                     authored_color3)) {
            body_color[0] = authored_color3[0];
            body_color[1] = authored_color3[1];
            body_color[2] = authored_color3[2];
            has_body_color = true;
        }
        if (nanousd_attribv4f(shader_prim, "inputs:Handle", authored_color4)) {
            handle_color[0] = authored_color4[0];
            handle_color[1] = authored_color4[1];
            handle_color[2] = authored_color4[2];
            has_handle_color = true;
        } else if (nanousd_attribv3f(shader_prim, "inputs:Handle",
                                     authored_color3)) {
            handle_color[0] = authored_color3[0];
            handle_color[1] = authored_color3[1];
            handle_color[2] = authored_color3[2];
            has_handle_color = true;
        }
        if (nanousd_attribv4f(shader_prim, "inputs:Cap", authored_color4)) {
            cap_color[0] = authored_color4[0];
            cap_color[1] = authored_color4[1];
            cap_color[2] = authored_color4[2];
            has_cap_color = true;
        } else if (nanousd_attribv3f(shader_prim, "inputs:Cap",
                                     authored_color3)) {
            cap_color[0] = authored_color3[0];
            cap_color[1] = authored_color3[1];
            cap_color[2] = authored_color3[2];
            has_cap_color = true;
        }

        int authored_float_ok = 0;
        float authored_param = nanousd_attribf(shader_prim, "inputs:Param",
                                               &authored_float_ok);
        if (authored_float_ok)
            emissive_product_scale = authored_param;

        bool baked_body_mask_albedo = false;
        if ((has_body_color || has_handle_color || has_cap_color) &&
            !albedo_texture_ref.empty() && !mask_selection_ref.empty()) {
            if (!has_body_color) {
                body_color[0] = body_color[1] = body_color[2] = 0.0f;
            }
            if (!has_handle_color) {
                handle_color[0] = handle_color[1] = handle_color[2] = 0.0f;
            }
            if (!has_cap_color) {
                cap_color[0] = cap_color[1] = cap_color[2] = 0.0f;
            }
            int baked_idx = bake_mdl_body_masked_albedo_texture(
                albedo_texture_ref, mask_selection_ref, body_color,
                handle_color, cap_color, anchor, stage, tex_paths, textures);
            if (baked_idx >= 0) {
                params->tex_indices[TEX_DIFFUSE_COLOR] = baked_idx;
                params->base_color[0] = 1.0f;
                params->base_color[1] = 1.0f;
                params->base_color[2] = 1.0f;
                params->v_flip = 0;
                baked_body_mask_albedo = true;
            }
        }

        if (!baked_body_mask_albedo &&
            has_color_albedo && !albedo_texture_ref.empty() &&
            !mask_selection_ref.empty()) {
            int baked_idx = bake_mdl_masked_albedo_texture(
                albedo_texture_ref, mask_selection_ref, color_albedo,
                anchor, stage, tex_paths, textures);
            if (baked_idx >= 0) {
                params->tex_indices[TEX_DIFFUSE_COLOR] = baked_idx;
                params->base_color[0] = 1.0f;
                params->base_color[1] = 1.0f;
                params->base_color[2] = 1.0f;
            }
        }

        if (has_plastic_wrap) {
            int baked_idx = bake_mdl_plastic_wrap_albedo_texture(
                box_albedo_ref, plastic_albedo_ref, plastic_normal_ref,
                box_tint, plastic_tint, plastic_opacity, u_tiling, v_tiling,
                anchor, stage, tex_paths, textures);
            if (baked_idx >= 0) {
                params->tex_indices[TEX_DIFFUSE_COLOR] = baked_idx;
                params->base_color[0] = 1.0f;
                params->base_color[1] = 1.0f;
                params->base_color[2] = 1.0f;
                params->v_flip = 0;
                if (!plastic_orm_ref.empty()) {
                    int orm_idx = load_texture_dedup(plastic_orm_ref.c_str(),
                                                     anchor, tex_paths,
                                                     textures, stage);
                    if (orm_idx >= 0) {
                        params->tex_indices[TEX_OCCLUSION] = orm_idx;
                        params->tex_indices[TEX_METALLIC] = orm_idx;
                        propagate_udim_scale(params, textures[orm_idx]);
                    }
                }
                /* Some generated plastic-wrap MDLs clamp plastic ORM.g to
                 * RoughnessMax rather than lerping the full 0..1 range.
                 * In practice the authored map is almost always above the
                 * 0.05 max, so a scalar max is closer than remapping to
                 * near-mirror values. */
                if (has_roughness_max) {
                    params->tex_indices[TEX_ROUGHNESS] = -1;
                    params->roughness = roughness_max;
                    params->roughness_tex_scale = 1.0f;
                    params->roughness_tex_bias = 0.0f;
                }
                if (!plastic_normal_ref.empty()) {
                    int normal_idx = load_texture_dedup(plastic_normal_ref.c_str(),
                                                        anchor, tex_paths,
                                                        textures, stage);
                    if (normal_idx >= 0) {
                        params->tex_indices[TEX_NORMAL] = normal_idx;
                        propagate_udim_scale(params, textures[normal_idx]);
                        if (has_plastic_normal_alpha)
                            params->normal_scale = plastic_normal_alpha;
                    }
                }
                if (getenv("NUSD_MAT_DIAG")) {
                    const char* pp = nanousd_path(shader_prim);
                    fprintf(stderr,
                            "[mat_diag] MDL plastic-wrap albedo finalised: "
                            "shader=%s tex=%d uv=(%.3f,%.3f) opacity=%.3f\n",
                            pp ? pp : "?", baked_idx, u_tiling, v_tiling,
                            plastic_opacity);
                }
            }
        }

        if (mdl_source_has_nonzero_emissive_color(pattern_source)) {
            std::string emissive_base = albedo_texture_ref.empty()
                ? find_first_mdl_texture_literal_for_slot(pattern_source,
                                                          TEX_DIFFUSE_COLOR)
                : albedo_texture_ref;
            std::string emissive_mask =
                find_mdl_texture_literal_with_any_keyword(
                    pattern_source,
                    {"emiss", "stripe", "strip", "glow", "light"});
            int baked_idx = bake_mdl_emissive_product_texture(
                emissive_base, emissive_mask, 1.0f,
                anchor, stage, tex_paths, textures);
            if (baked_idx >= 0) {
                params->tex_indices[TEX_EMISSIVE_COLOR] = baked_idx;
                params->emissive_color[0] = 0.0f;
                params->emissive_color[1] = 0.0f;
                params->emissive_color[2] = 0.0f;
                params->emissive_color[3] = emissive_product_scale;
            }
        }
    }

    size_t pos = 0;
    const std::string key = "texture_2d";
    bool saw_body_texture = false;
    while ((pos = pattern_source.find(key, pos)) != std::string::npos) {
        std::string tex_ref;
        if (!parse_first_quoted_after(pattern_source, pos + key.size(), &tex_ref)) {
            pos += key.size();
            continue;
        }
        size_t hint_begin = pattern_source.rfind('\n', pos);
        hint_begin = (hint_begin == std::string::npos) ? 0 : hint_begin + 1;
        std::string hint = pattern_source.substr(hint_begin, pos - hint_begin);
        assign_mdl_texture(params, tex_paths, textures, hint, tex_ref,
                           anchor, stage);
        if (has_image_extension(tex_ref))
            saw_body_texture = true;
        pos += key.size();
    }
    if (saw_body_texture &&
        pattern_source.find("1.0-state::texture_coordinate(0).y") !=
            std::string::npos &&
        pattern_source.find("1.0-") != std::string::npos) {
        params->v_flip = 0;
    }

    if (apply_constant_defaults)
        apply_authored_mdl_shader_inputs(shader_prim, params);
}

static bool split_connection_path(const char* conn,
                                  std::string& prim_path,
                                  std::string& prop_name)
{
    prim_path.clear();
    prop_name.clear();
    if (!conn || !conn[0]) return false;

    std::string s(conn);
    if (!s.empty() && s.front() == '<') {
        size_t end = s.find('>');
        if (end != std::string::npos)
            s = s.substr(1, end - 1);
    }

    size_t dot = s.rfind('.');
    if (dot == std::string::npos) {
        prim_path = s;
        return !prim_path.empty();
    }
    prim_path = s.substr(0, dot);
    prop_name = s.substr(dot + 1);
    return !prim_path.empty();
}

static std::string first_connection(NanousdPrim prim, const char* attr)
{
    if (!prim || !attr) return std::string();
    if (nanousd_nconnections(prim, attr) <= 0) return std::string();
    const char* c = nanousd_connection(prim, attr, 0);
    return c ? std::string(c) : std::string();
}

static bool read_attr_float(NanousdPrim prim, const char* attr, float* out)
{
    if (!prim || !attr || !out) return false;
    int ok = 0;
    float f = nanousd_attribf(prim, attr, &ok);
    if (ok) {
        *out = f;
        return true;
    }
    int i = nanousd_attribi(prim, attr, &ok);
    if (ok) {
        *out = (float)i;
        return true;
    }
    float v3[3];
    if (nanousd_attribv3f(prim, attr, v3)) {
        *out = v3[0];
        return true;
    }
    float v4[4];
    if (nanousd_attribv4f(prim, attr, v4)) {
        *out = v4[0];
        return true;
    }
    return false;
}

static bool read_attr_color3(NanousdPrim prim, const char* attr, float out[3])
{
    if (!prim || !attr || !out) return false;
    float v3[3];
    if (nanousd_attribv3f(prim, attr, v3)) {
        out[0] = v3[0];
        out[1] = v3[1];
        out[2] = v3[2];
        return true;
    }
    float v4[4];
    if (nanousd_attribv4f(prim, attr, v4)) {
        out[0] = v4[0];
        out[1] = v4[1];
        out[2] = v4[2];
        return true;
    }
    float f = 0.0f;
    if (read_attr_float(prim, attr, &f)) {
        out[0] = f;
        out[1] = f;
        out[2] = f;
        return true;
    }
    return false;
}

static bool read_connected_attr_float(NanousdStage stage,
                                      NanousdPrim shader_prim,
                                      NanousdPrim material_prim,
                                      const char* input_name,
                                      float* out)
{
    char attr[128];
    snprintf(attr, sizeof(attr), "inputs:%s", input_name);

    if (read_attr_float(shader_prim, attr, out)) return true;

    std::string conn = first_connection(shader_prim, attr);
    if (!conn.empty()) {
        std::string prim_path, prop_name;
        if (split_connection_path(conn.c_str(), prim_path, prop_name)) {
            NanousdPrim src = nanousd_primpath(stage, prim_path.c_str());
            if (src) {
                bool ok = false;
                if (!prop_name.empty())
                    ok = read_attr_float(src, prop_name.c_str(), out);
                nanousd_freeprim(src);
                if (ok) return true;
            }
        }
    }

    return read_attr_float(material_prim, attr, out);
}

static bool read_connected_attr_color3(NanousdStage stage,
                                       NanousdPrim shader_prim,
                                       NanousdPrim material_prim,
                                       const char* input_name,
                                       float out[3])
{
    char attr[128];
    snprintf(attr, sizeof(attr), "inputs:%s", input_name);

    if (read_attr_color3(shader_prim, attr, out)) return true;

    std::string conn = first_connection(shader_prim, attr);
    if (!conn.empty()) {
        std::string prim_path, prop_name;
        if (split_connection_path(conn.c_str(), prim_path, prop_name)) {
            NanousdPrim src = nanousd_primpath(stage, prim_path.c_str());
            if (src) {
                bool ok = false;
                if (!prop_name.empty())
                    ok = read_attr_color3(src, prop_name.c_str(), out);
                nanousd_freeprim(src);
                if (ok) return true;
            }
        }
    }

    return read_attr_color3(material_prim, attr, out);
}

static void init_pxr_material_defaults(MaterialParams* params)
{
    memset(params, 0, sizeof(*params));
    params->base_color[0] = 0.18f;
    params->base_color[1] = 0.18f;
    params->base_color[2] = 0.18f;
    params->base_color[3] = 1.0f;
    params->emissive_color[3] = 1.0f;
    params->metallic = 0.0f;
    params->roughness = 0.5f;
    params->opacity = 1.0f;
    params->ior = 1.5f;
    params->occlusion = 1.0f;
    params->clearcoat = 0.0f;
    params->clearcoat_roughness = 0.01f;
    params->normal_scale = 1.0f;
    for (int i = 0; i < MAX_MATERIAL_TEXTURES; i++)
        params->tex_indices[i] = -1;
    params->udim_scale_u = 1.0f;
    params->udim_scale_v = 1.0f;
    params->subsurface_color[0] = 1.0f;
    params->subsurface_color[1] = 1.0f;
    params->subsurface_color[2] = 1.0f;
    params->subsurface_color[3] = 1.0f;
    params->subsurface_radius[0] = 1.0f;
    params->subsurface_radius[1] = 1.0f;
    params->subsurface_radius[2] = 1.0f;
    params->subsurface_radius[3] = 1.0f;
    params->transmission_color[0] = 1.0f;
    params->transmission_color[1] = 1.0f;
    params->transmission_color[2] = 1.0f;
    params->transmission_color[3] = 1.0f;
    params->tex_subsurface_weight = -1;
    params->tex_transmission_weight = -1;
    params->specular_color[3] = 1.0f;
    params->normal_tex_scale[0] = 2.0f;
    params->normal_tex_scale[1] = 2.0f;
    params->normal_tex_scale[2] = 2.0f;
    params->normal_tex_scale[3] = 1.0f;
    params->normal_tex_bias[0] = -1.0f;
    params->normal_tex_bias[1] = -1.0f;
    params->normal_tex_bias[2] = -1.0f;
    params->normal_tex_bias[3] = 0.0f;
    set_default_mdl_uv_transform(params);
    params->roughness_tex_scale = 1.0f;
    params->roughness_tex_bias = 0.0f;
}

static void apply_color_correct_gamma(float rgb[3], NanousdPrim shader_prim)
{
    float gamma[3] = {1.0f, 1.0f, 1.0f};
    if (!read_attr_color3(shader_prim, "inputs:gamma", gamma))
        return;
    for (int c = 0; c < 3; ++c) {
        float g = gamma[c];
        if (std::isfinite(g) && g > 0.0f && std::fabs(g - 1.0f) > 1e-6f)
            rgb[c] = std::pow(std::max(rgb[c], 0.0f), 1.0f / g);
    }
}

static bool read_attr_color3_deep(NanousdStage stage,
                                  NanousdPrim prim,
                                  const char* attr,
                                  float out[3],
                                  int depth)
{
    if (!stage || !prim || !attr || !out || depth > 8) return false;

    if (read_attr_color3(prim, attr, out)) return true;

    int ok = 0;
    const char* info_id = nanousd_attrib_token(prim, "info:id", &ok);
    if (ok && info_id && strcmp(info_id, "PxrColorCorrect") == 0 &&
        strncmp(attr, "outputs:", 8) == 0) {
        float rgb[3];
        if (read_attr_color3_deep(stage, prim, "inputs:inputRGB", rgb,
                                  depth + 1)) {
            apply_color_correct_gamma(rgb, prim);
            out[0] = rgb[0];
            out[1] = rgb[1];
            out[2] = rgb[2];
            return true;
        }
    }

    std::string conn = first_connection(prim, attr);
    if (!conn.empty()) {
        std::string prim_path, prop_name;
        if (split_connection_path(conn.c_str(), prim_path, prop_name)) {
            NanousdPrim src = nanousd_primpath(stage, prim_path.c_str());
            if (src) {
                bool got = !prop_name.empty() &&
                           read_attr_color3_deep(stage, src,
                                                 prop_name.c_str(), out,
                                                 depth + 1);
                nanousd_freeprim(src);
                if (got) return true;
            }
        }
    }

    return false;
}

static bool read_pxr_input_color3(NanousdStage stage,
                                  NanousdPrim shader_prim,
                                  NanousdPrim material_prim,
                                  const char* input_name,
                                  float out[3])
{
    char attr[128];
    snprintf(attr, sizeof(attr), "inputs:%s", input_name);
    if (read_attr_color3_deep(stage, shader_prim, attr, out, 0)) return true;
    return read_attr_color3_deep(stage, material_prim, attr, out, 0);
}

static bool read_pxr_input_float(NanousdStage stage,
                                 NanousdPrim shader_prim,
                                 NanousdPrim material_prim,
                                 const char* input_name,
                                 float* out)
{
    return read_connected_attr_float(stage, shader_prim, material_prim,
                                     input_name, out);
}

static void read_pxr_disney_material(NanousdStage stage,
                                     NanousdPrim material_prim,
                                     NanousdPrim shader_prim,
                                     MaterialParams* params)
{
    init_pxr_material_defaults(params);

    /* Material-interface inputs are the fallback. The shader graph may route
     * those values through PxrColorCorrect, so resolve the graph afterwards and
     * let its color-space transform win when present. */
    apply_material_interface_inputs(material_prim, params);

    float c3[3];
    if (read_pxr_input_color3(stage, shader_prim, material_prim,
                              "baseColor", c3) ||
        read_pxr_input_color3(stage, shader_prim, material_prim,
                              "diffuseColor", c3)) {
        params->base_color[0] = c3[0];
        params->base_color[1] = c3[1];
        params->base_color[2] = c3[2];
    }

    int ok = 0;
    float f = 0.0f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "metallic", &f)) params->metallic = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "roughness", &f)) params->roughness = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "ior", &f)) params->ior = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "clearcoat", &f)) params->clearcoat = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "clearcoatGloss", &f))
        params->clearcoat_roughness = 1.0f - f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "opacity", &f)) {
        params->opacity = f;
        params->base_color[3] = f;
    }

    f = nanousd_attribf(material_prim, "inputs:alpha", &ok);
    if (ok) {
        params->opacity = f;
        params->base_color[3] = f;
    }

    params->metallic = std::clamp(params->metallic, 0.0f, 1.0f);
    params->roughness = std::clamp(params->roughness, 0.01f, 1.0f);
    params->opacity = std::clamp(params->opacity, 0.0f, 1.0f);
    params->base_color[3] = std::clamp(params->base_color[3], 0.0f, 1.0f);
    params->ior = std::clamp(params->ior, 1.0f, 3.0f);
    params->clearcoat = std::clamp(params->clearcoat, 0.0f, 1.0f);
    params->clearcoat_roughness =
        std::clamp(params->clearcoat_roughness, 0.01f, 1.0f);
}

static void read_pxr_surface_material(NanousdStage stage,
                                      NanousdPrim material_prim,
                                      NanousdPrim shader_prim,
                                      MaterialParams* params)
{
    init_pxr_material_defaults(params);

    float c3[3];
    bool have_diffuse = read_pxr_input_color3(stage, shader_prim, material_prim,
                                              "diffuseColor", c3);
    if (!have_diffuse)
        have_diffuse = read_pxr_input_color3(stage, shader_prim, material_prim,
                                             "baseColor", c3);
    if (have_diffuse) {
        params->base_color[0] = c3[0];
        params->base_color[1] = c3[1];
        params->base_color[2] = c3[2];
    }

    float transmit[3] = {1.0f, 1.0f, 1.0f};
    bool have_transmit = read_pxr_input_color3(stage, shader_prim, material_prim,
                                               "diffuseTransmitColor",
                                               transmit);

    float f = 0.0f;
    float diffuse_gain = 1.0f;
    float diffuse_transmit_gain = 0.0f;
    float reflection_gain = 0.0f;
    float refraction_gain = 0.0f;
    float specular_transmission = 0.0f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "diffuseGain", &f)) diffuse_gain = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "diffuseTransmitGain", &f))
        diffuse_transmit_gain = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "reflectionGain", &f)) reflection_gain = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "refractionGain", &f)) refraction_gain = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "specularTransmission", &f))
        specular_transmission = f;

    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "roughness", &f)) params->roughness = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "glassRoughness", &f)) params->roughness = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "ior", &f)) params->ior = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "glassIor", &f)) params->ior = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "clearcoat", &f)) params->clearcoat = f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "clearcoatGloss", &f))
        params->clearcoat_roughness = 1.0f - f;
    if (read_pxr_input_float(stage, shader_prim, material_prim,
                             "opacity", &f)) {
        params->opacity = f;
        params->base_color[3] = f;
    }

    float trans_weight = std::max(std::max(refraction_gain,
                                           specular_transmission),
                                  diffuse_transmit_gain);
    if (trans_weight > 0.001f) {
        params->transmission_weight = std::clamp(trans_weight, 0.0f, 1.0f);
        params->transmission_ior = params->ior;
        if (have_transmit) {
            params->transmission_color[0] = transmit[0];
            params->transmission_color[1] = transmit[1];
            params->transmission_color[2] = transmit[2];
            params->base_color[0] = transmit[0];
            params->base_color[1] = transmit[1];
            params->base_color[2] = transmit[2];
        }
        if (diffuse_gain <= 0.001f) {
            params->opacity = std::min(params->opacity, 0.35f);
            params->base_color[3] = params->opacity;
        }
        params->clearcoat = std::max(params->clearcoat,
                                     std::clamp(reflection_gain, 0.0f, 1.0f));
    }

    params->metallic = std::clamp(params->metallic, 0.0f, 1.0f);
    params->roughness = std::clamp(params->roughness, 0.0f, 1.0f);
    params->opacity = std::clamp(params->opacity, 0.0f, 1.0f);
    params->base_color[3] = std::clamp(params->base_color[3], 0.0f, 1.0f);
    params->ior = std::clamp(params->ior, 1.0f, 3.0f);
    params->clearcoat = std::clamp(params->clearcoat, 0.0f, 1.0f);
    params->clearcoat_roughness =
        std::clamp(params->clearcoat_roughness, 0.01f, 1.0f);
}

static std::string resolve_mtlx_texture_file_from_target(NanousdStage stage,
                                                         const std::string& target,
                                                         int depth,
                                                         bool* is_normal_map,
                                                         float out_uvtiling[2],
                                                         int* has_uvtiling)
{
    if (!stage || target.empty() || depth > 8) return std::string();

    std::string prim_path, prop_name;
    if (!split_connection_path(target.c_str(), prim_path, prop_name))
        return std::string();

    NanousdPrim src = nanousd_primpath(stage, prim_path.c_str());
    if (!src) return std::string();

    int ok = 0;
    const char* info_id = nanousd_attrib_token(src, "info:id", &ok);
    std::string info = (ok && info_id) ? info_id : "";

    if (info.find("normalmap") != std::string::npos) {
        if (is_normal_map) *is_normal_map = true;
        std::string inner = first_connection(src, "inputs:in");
        nanousd_freeprim(src);
        return resolve_mtlx_texture_file_from_target(stage, inner, depth + 1,
                                                     is_normal_map, out_uvtiling,
                                                     has_uvtiling);
    }

    const char* file = nanousd_attribasset(src, "inputs:file", &ok);
    if (!ok || !file || !file[0])
        file = nanousd_attribs(src, "inputs:file", &ok);
    if (ok && file && file[0]) {
        std::string out(file);
        /* USD-authored MaterialX carries per-image UV tiling as inputs:uvtiling
         * (a vector2 — the same custom image input the standalone .mtlx reader
         * consumes via image->getInput("uvtiling")). Read it off the image node
         * so the USD-authored path tiles like the standalone path instead of
         * silently ignoring it. */
        if (out_uvtiling && has_uvtiling) {
            float uvt[2] = {1.0f, 1.0f};
            if (nanousd_attribv2f(src, "inputs:uvtiling", uvt)) {
                out_uvtiling[0] = uvt[0];
                out_uvtiling[1] = uvt[1];
                *has_uvtiling = 1;
            }
        }
        nanousd_freeprim(src);
        return out;
    }

    if (!prop_name.empty()) {
        std::string next = first_connection(src, prop_name.c_str());
        if (!next.empty()) {
            nanousd_freeprim(src);
            return resolve_mtlx_texture_file_from_target(stage, next, depth + 1,
                                                         is_normal_map, out_uvtiling,
                                                         has_uvtiling);
        }
    }

    nanousd_freeprim(src);
    return std::string();
}

static int bind_mtlx_texture_input(NanousdStage stage,
                                   NanousdPrim shader_prim,
                                   const char* input_name,
                                   int slot,
                                   MaterialParams* params,
                                   std::vector<std::string>& tex_paths,
                                   std::vector<MaterialTexture>& textures,
                                   const char* scene_dir)
{
    char attr[128];
    snprintf(attr, sizeof(attr), "inputs:%s", input_name);
    std::string conn = first_connection(shader_prim, attr);
    if (conn.empty()) return -1;

    bool is_normal_map = false;
    float uvtiling[2] = {1.0f, 1.0f};
    int has_uvtiling = 0;
    std::string file = resolve_mtlx_texture_file_from_target(stage, conn, 0,
                                                             &is_normal_map,
                                                             uvtiling, &has_uvtiling);
    if (file.empty()) return -1;

    int tex_idx = load_texture_dedup(file.c_str(), scene_dir,
                                     tex_paths, textures, stage);
    if (tex_idx < 0) return -1;

    if (slot >= 0 && params->tex_indices[slot] < 0)
        params->tex_indices[slot] = tex_idx;
    if (slot == TEX_NORMAL || is_normal_map) {
        params->normal_tex_scale[0] = 2.0f;
        params->normal_tex_scale[1] = 2.0f;
        params->normal_tex_scale[2] = 2.0f;
        params->normal_tex_scale[3] = 1.0f;
        params->normal_tex_bias[0] = -1.0f;
        params->normal_tex_bias[1] = -1.0f;
        params->normal_tex_bias[2] = -1.0f;
        params->normal_tex_bias[3] = 0.0f;
    }
    propagate_udim_scale(params, textures[tex_idx]);

    /* One UV transform per material; let the base-color image define it. This is
     * a deterministic priority (vs the order-dependent last-write the standalone
     * path happens to do) — uvtiling on other slots is ignored. mdl_uv_transform
     * was initialized to identity (1,1,0,0) by init_materialx_surface_defaults,
     * so a material with no base-color uvtiling stays un-tiled. */
    if (slot == TEX_DIFFUSE_COLOR && has_uvtiling) {
        params->mdl_uv_transform[0] = uvtiling[0];
        params->mdl_uv_transform[1] = uvtiling[1];
    }
    return tex_idx;
}

static void init_materialx_surface_defaults(MaterialParams* params,
                                            bool open_pbr)
{
    params->base_color[0] = 0.8f;
    params->base_color[1] = 0.8f;
    params->base_color[2] = 0.8f;
    params->base_color[3] = 1.0f;
    params->emissive_color[0] = 0.0f;
    params->emissive_color[1] = 0.0f;
    params->emissive_color[2] = 0.0f;
    params->emissive_color[3] = 1.0f;
    params->metallic = 0.0f;
    params->roughness = open_pbr ? 0.5f : 0.2f;
    params->opacity = 1.0f;
    params->ior = 1.5f;
    params->occlusion = 1.0f;
    params->clearcoat = 0.0f;
    params->clearcoat_roughness = 0.01f;
    params->normal_scale = 1.0f;
    params->use_vertex_color = 0;
    params->udim_scale_u = 1.0f;
    params->udim_scale_v = 1.0f;
    params->opacity_threshold = 0.0f;
    params->subsurface_weight = 0.0f;
    params->subsurface_scale = 1.0f;
    params->transmission_weight = 0.0f;
    params->transmission_ior = 0.0f;
    params->tex_subsurface_weight = -1;
    params->tex_transmission_weight = -1;
    params->sss_color_authored = 0;
    params->use_specular_workflow = 0;
    params->v_flip = 0;
    for (int i = 0; i < MAX_MATERIAL_TEXTURES; i++)
        params->tex_indices[i] = -1;
    for (int i = 0; i < 4; i++) {
        params->subsurface_color[i] = (i == 3) ? 1.0f : 1.0f;
        params->subsurface_radius[i] = (i == 3) ? 1.0f : 1.0f;
        params->transmission_color[i] = (i == 3) ? 1.0f : 1.0f;
        params->specular_color[i] = 0.0f;
    }
    params->normal_tex_scale[0] = 2.0f;
    params->normal_tex_scale[1] = 2.0f;
    params->normal_tex_scale[2] = 2.0f;
    params->normal_tex_scale[3] = 1.0f;
    params->normal_tex_bias[0] = -1.0f;
    params->normal_tex_bias[1] = -1.0f;
    params->normal_tex_bias[2] = -1.0f;
    params->normal_tex_bias[3] = 0.0f;
    set_default_mdl_uv_transform(params);
}

static void read_materialx_usd_surface(NanousdStage stage,
                                       NanousdPrim material_prim,
                                       NanousdPrim shader_prim,
                                       MaterialParams* params,
                                       std::vector<std::string>& tex_paths,
                                       std::vector<MaterialTexture>& textures,
                                       const char* scene_dir,
                                       bool open_pbr)
{
    init_materialx_surface_defaults(params, open_pbr);
    if (!stage || !material_prim || !shader_prim) return;

    float c3[3];
    float f = 0.0f;

    if (read_connected_attr_color3(stage, shader_prim, material_prim,
                                   "base_color", c3)) {
        params->base_color[0] = c3[0];
        params->base_color[1] = c3[1];
        params->base_color[2] = c3[2];
    }

    if (open_pbr) {
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "base_metalness", &f))
            params->metallic = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "specular_roughness", &f))
            params->roughness = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "specular_ior", &f))
            params->ior = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "coat_weight", &f))
            params->clearcoat = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "coat_roughness", &f))
            params->clearcoat_roughness = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "geometry_opacity", &f))
            params->opacity = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "transmission_weight", &f))
            params->transmission_weight = f;
        if (read_connected_attr_color3(stage, shader_prim, material_prim,
                                       "transmission_color", c3)) {
            params->transmission_color[0] = c3[0];
            params->transmission_color[1] = c3[1];
            params->transmission_color[2] = c3[2];
        }
        if (read_connected_attr_color3(stage, shader_prim, material_prim,
                                       "emission_color", c3)) {
            params->emissive_color[0] = c3[0];
            params->emissive_color[1] = c3[1];
            params->emissive_color[2] = c3[2];
        }
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "emission_luminance", &f))
            params->emissive_color[3] = f;

        if (bind_mtlx_texture_input(stage, shader_prim, "base_color",
                                    TEX_DIFFUSE_COLOR, params, tex_paths,
                                    textures, scene_dir) >= 0) {
            params->base_color[0] = 1.0f;
            params->base_color[1] = 1.0f;
            params->base_color[2] = 1.0f;
        }
        bind_mtlx_texture_input(stage, shader_prim, "base_metalness",
                                TEX_METALLIC, params, tex_paths, textures,
                                scene_dir);
        bind_mtlx_texture_input(stage, shader_prim, "specular_roughness",
                                TEX_ROUGHNESS, params, tex_paths, textures,
                                scene_dir);
        bind_mtlx_texture_input(stage, shader_prim, "geometry_normal",
                                TEX_NORMAL, params, tex_paths, textures,
                                scene_dir);
        bind_mtlx_texture_input(stage, shader_prim, "geometry_opacity",
                                TEX_OPACITY, params, tex_paths, textures,
                                scene_dir);
        bind_mtlx_texture_input(stage, shader_prim, "emission_color",
                                TEX_EMISSIVE_COLOR, params, tex_paths,
                                textures, scene_dir);
    } else {
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "metalness", &f))
            params->metallic = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "specular_roughness", &f))
            params->roughness = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "specular_IOR", &f)) {
            params->ior = f;
            params->transmission_ior = f;
        }
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "coat", &f))
            params->clearcoat = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "coat_roughness", &f))
            params->clearcoat_roughness = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "opacity", &f))
            params->opacity = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "transmission", &f))
            params->transmission_weight = f;
        if (read_connected_attr_color3(stage, shader_prim, material_prim,
                                       "transmission_color", c3)) {
            params->transmission_color[0] = c3[0];
            params->transmission_color[1] = c3[1];
            params->transmission_color[2] = c3[2];
        }
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "subsurface", &f))
            params->subsurface_weight = f;
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "subsurface_scale", &f))
            params->subsurface_scale = f;
        if (read_connected_attr_color3(stage, shader_prim, material_prim,
                                       "subsurface_color", c3)) {
            params->subsurface_color[0] = c3[0];
            params->subsurface_color[1] = c3[1];
            params->subsurface_color[2] = c3[2];
            params->sss_color_authored = 1;
        }
        if (read_connected_attr_color3(stage, shader_prim, material_prim,
                                       "subsurface_radius", c3)) {
            params->subsurface_radius[0] = c3[0];
            params->subsurface_radius[1] = c3[1];
            params->subsurface_radius[2] = c3[2];
        }
        if (read_connected_attr_float(stage, shader_prim, material_prim,
                                      "emission", &f))
            params->emissive_color[3] = f;
        if (read_connected_attr_color3(stage, shader_prim, material_prim,
                                       "emission_color", c3)) {
            params->emissive_color[0] = c3[0] * params->emissive_color[3];
            params->emissive_color[1] = c3[1] * params->emissive_color[3];
            params->emissive_color[2] = c3[2] * params->emissive_color[3];
        }

        if (bind_mtlx_texture_input(stage, shader_prim, "base_color",
                                    TEX_DIFFUSE_COLOR, params, tex_paths,
                                    textures, scene_dir) >= 0) {
            params->base_color[0] = 1.0f;
            params->base_color[1] = 1.0f;
            params->base_color[2] = 1.0f;
        }
        bind_mtlx_texture_input(stage, shader_prim, "metalness",
                                TEX_METALLIC, params, tex_paths, textures,
                                scene_dir);
        bind_mtlx_texture_input(stage, shader_prim, "specular_roughness",
                                TEX_ROUGHNESS, params, tex_paths, textures,
                                scene_dir);
        bind_mtlx_texture_input(stage, shader_prim, "normal",
                                TEX_NORMAL, params, tex_paths, textures,
                                scene_dir);
        bind_mtlx_texture_input(stage, shader_prim, "opacity",
                                TEX_OPACITY, params, tex_paths, textures,
                                scene_dir);
        bind_mtlx_texture_input(stage, shader_prim, "emission_color",
                                TEX_EMISSIVE_COLOR, params, tex_paths,
                                textures, scene_dir);

        int sss_tex = bind_mtlx_texture_input(stage, shader_prim, "subsurface",
                                              -1, params, tex_paths, textures,
                                              scene_dir);
        if (sss_tex >= 0) {
            params->tex_subsurface_weight = sss_tex;
            params->subsurface_weight = 1.0f;
        }
        int tr_tex = bind_mtlx_texture_input(stage, shader_prim, "transmission",
                                             -1, params, tex_paths, textures,
                                             scene_dir);
        if (tr_tex >= 0) {
            params->tex_transmission_weight = tr_tex;
            params->transmission_weight = 1.0f;
        }
    }

    params->metallic = std::clamp(params->metallic, 0.0f, 1.0f);
    params->roughness = std::clamp(params->roughness, 0.0f, 1.0f);
    params->opacity = std::clamp(params->opacity, 0.0f, 1.0f);

    fprintf(stderr,
            "material: read MaterialX USD %s params: base=(%.2f,%.2f,%.2f) "
            "metal=%.2f rough=%.2f opacity=%.2f slots D=%d N=%d R=%d M=%d E=%d OP=%d\n",
            open_pbr ? "OpenPBR" : "StandardSurface",
            params->base_color[0], params->base_color[1], params->base_color[2],
            params->metallic, params->roughness, params->opacity,
            params->tex_indices[TEX_DIFFUSE_COLOR],
            params->tex_indices[TEX_NORMAL],
            params->tex_indices[TEX_ROUGHNESS],
            params->tex_indices[TEX_METALLIC],
            params->tex_indices[TEX_EMISSIVE_COLOR],
            params->tex_indices[TEX_OPACITY]);
}

/* ---- Read OmniPBR / MDL material properties ---- */

static void read_omnipbr_material(NanousdPrim shader_prim,
                                  MaterialParams* params,
                                  std::vector<std::string>& tex_paths,
                                  std::vector<MaterialTexture>& textures,
                                  const char* scene_dir,
                                  NanousdStage stage = nullptr)
{
    // Initialize defaults similar to OmniPBR defaults.
    // base_color defaults to white so it acts as a no-op multiplier when a
    // diffuse texture is bound; for textureless materials, the diffuse_color_*
    // input below overrides it with the surface color.
    params->base_color[0] = 1.0f;
    params->base_color[1] = 1.0f;
    params->base_color[2] = 1.0f;
    params->base_color[3] = 1.0f;
    params->emissive_color[0] = 0.0f;
    params->emissive_color[1] = 0.0f;
    params->emissive_color[2] = 0.0f;
    params->emissive_color[3] = 0.0f; // intensity=0 means no emission
    params->metallic = 0.0f;
    params->roughness = 0.5f;
    params->opacity = 1.0f;
    params->ior = 1.5f;
    params->occlusion = 1.0f;
    params->clearcoat = 0.0f;
    params->clearcoat_roughness = 0.01f;
    params->normal_scale = 1.0f;
    /* See read_standard_surface_mtlx for rationale. */
    params->normal_tex_scale[0] = 2.0f;
    params->normal_tex_scale[1] = 2.0f;
    params->normal_tex_scale[2] = 2.0f;
    params->normal_tex_scale[3] = 1.0f;
    params->normal_tex_bias[0]  = -1.0f;
    params->normal_tex_bias[1]  = -1.0f;
    params->normal_tex_bias[2]  = -1.0f;
    params->normal_tex_bias[3]  =  0.0f;
    set_default_mdl_uv_transform(params);
    params->use_vertex_color = 0;
    params->udim_scale_u = 1.0f;
    params->udim_scale_v = 1.0f;
    params->opacity_threshold = 0.0f;
    params->subsurface_weight = 0.0f;
    params->subsurface_scale = 1.0f;
    params->transmission_weight = 0.0f;
    params->transmission_ior = 0.0f;
    params->tex_subsurface_weight = -1;
    params->tex_transmission_weight = -1;
    params->sss_color_authored = 0;
    params->use_specular_workflow = 0;
    for (int i = 0; i < 4; i++) {
        params->subsurface_color[i] = 1.0f;
        params->subsurface_radius[i] = 1.0f;
        params->transmission_color[i] = 1.0f;
        params->specular_color[i] = 0.0f;
    }

    for (int i = 0; i < MAX_MATERIAL_TEXTURES; i++)
        params->tex_indices[i] = -1;

    if (!shader_prim) return;

    {
        int mdl_asset_ok = 0;
        const char* mdl_asset = nanousd_attribasset(shader_prim,
                                                    "info:mdl:sourceAsset",
                                                    &mdl_asset_ok);
        if (mdl_asset_ok && mdl_asset && mdl_asset[0])
            params->v_flip = 1;
    }

    /* When the MDL SDK is not linked (default build), apply_mdl_sdk_source_material
     * collects shader inputs (many per-material nanousd_attrib reads) then hits the
     * mdl_bridge stub which discards them and returns failure — pure wasted CPU in
     * the ~14.7s material-load phase. Skip it; the generic path below produces the
     * identical material (mdl_sdk_applied is invariantly false here, so it is called
     * with apply_constant_defaults=true exactly as before). Bit-identical.
     * See docs/plans/VULKAN_RT_LOAD_TIME_PLAN.md (iter4). */
#ifdef NUSD_HAVE_MDL_SDK
    bool mdl_sdk_applied = apply_mdl_sdk_source_material(shader_prim,
                                                         params,
                                                         scene_dir, stage,
                                                         tex_paths, textures);
#else
    bool mdl_sdk_applied = false;
#endif

    int ok;
    float v3[3];

    // --- Diffuse / base color ---
    // OmniPBR: inputs:diffuse_color_constant (color3f)
    if (nanousd_attribv3f(shader_prim, "inputs:diffuse_color_constant", v3)) {
        params->base_color[0] = v3[0];
        params->base_color[1] = v3[1];
        params->base_color[2] = v3[2];
    }
    // OmniSurface: inputs:diffuse_reflection_color
    else if (nanousd_attribv3f(shader_prim, "inputs:diffuse_reflection_color", v3)) {
        params->base_color[0] = v3[0];
        params->base_color[1] = v3[1];
        params->base_color[2] = v3[2];
    }
    // Fallback: inputs:diffuseColor (some MDL materials use this)
    else if (nanousd_attribv3f(shader_prim, "inputs:diffuseColor", v3)) {
        params->base_color[0] = v3[0];
        params->base_color[1] = v3[1];
        params->base_color[2] = v3[2];
    }
    /* Generated MDL tint inputs (BaseColor_Tint, ColorAlbedo) modulate an
     * albedo texture. We read them into base_color so the shader can
     * multiply: baseColor = tex * mat.base_color. Skip pure-zero "sentinel"
     * tints to avoid rendering black. */
    {
        float t4[4];
        if (nanousd_attribv4f(shader_prim, "inputs:BaseColor_Tint", t4) ||
            nanousd_attribv4f(shader_prim, "inputs:ColorAlbedo", t4)) {
            if (t4[0] + t4[1] + t4[2] > 0.001f) {
                params->base_color[0] = t4[0];
                params->base_color[1] = t4[1];
                params->base_color[2] = t4[2];
            }
        } else if (nanousd_attribv3f(shader_prim, "inputs:BaseColor_Tint", v3) ||
                   nanousd_attribv3f(shader_prim, "inputs:ColorAlbedo", v3)) {
            if (v3[0] + v3[1] + v3[2] > 0.001f) {
                params->base_color[0] = v3[0];
                params->base_color[1] = v3[1];
                params->base_color[2] = v3[2];
            }
        }
    }

    // --- Metallic ---
    float f;
    f = nanousd_attribf(shader_prim, "inputs:metallic_constant", &ok);
    if (ok) params->metallic = f;
    else {
        f = nanousd_attribf(shader_prim, "inputs:metallic", &ok);
        if (ok) params->metallic = f;
    }

    // --- Roughness ---
    f = nanousd_attribf(shader_prim, "inputs:reflection_roughness_constant", &ok);
    if (ok) params->roughness = f;
    else {
        f = nanousd_attribf(shader_prim, "inputs:roughness", &ok);
        if (ok) params->roughness = f;
    }
    /* Generated MDL: RoughnessMin/Max remaps ORM.g:
     *   roughness = RoughnessMin + MergeMap.g * (RoughnessMax - RoughnessMin)
     * Keep a scalar fallback for materials without a roughness texture. */
    {
        float rmin, rmax;
        int okmin, okmax;
        rmin = nanousd_attribf(shader_prim, "inputs:RoughnessMin", &okmin);
        rmax = nanousd_attribf(shader_prim, "inputs:RoughnessMax", &okmax);
        if (okmin && okmax) {
            params->roughness = 0.5f * (rmin + rmax);
            params->roughness_tex_bias = rmin;
            params->roughness_tex_scale = rmax - rmin;
        } else if (okmax) {
            params->roughness = rmax;
        } else if (okmin) {
            params->roughness = rmin;
        }
    }
    /* Generated MDL float "Roughness" and "Metallic" */
    f = nanousd_attribf(shader_prim, "inputs:Roughness", &ok);
    if (ok) params->roughness = f;
    f = nanousd_attribf(shader_prim, "inputs:Metallic", &ok);
    if (ok) params->metallic = f;

    // --- Emissive ---
    if (nanousd_attribv3f(shader_prim, "inputs:emissive_color", v3)) {
        params->emissive_color[0] = v3[0];
        params->emissive_color[1] = v3[1];
        params->emissive_color[2] = v3[2];
    }
    // Check enable_emission toggle (OmniPBR)
    int bok = 0;
    int b = nanousd_attribb(shader_prim, "inputs:enable_emission", &bok);
    f = nanousd_attribf(shader_prim, "inputs:enable_emission", &ok);
    bool emission_enabled = bok ? (b != 0) : (!ok || f > 0.5f);
    f = nanousd_attribf(shader_prim, "inputs:emissive_intensity", &ok);
    if (!emission_enabled) {
        params->emissive_color[3] = 0.0f;
    } else if (ok) {
        params->emissive_color[3] = f;
    } else if (emission_enabled &&
               (params->emissive_color[0] > 0.0f ||
                params->emissive_color[1] > 0.0f ||
                params->emissive_color[2] > 0.0f)) {
        params->emissive_color[3] = 1.0f; // Default intensity
    }

    // --- Opacity ---
    f = nanousd_attribf(shader_prim, "inputs:opacity_constant", &ok);
    if (ok) params->opacity = f;
    else {
        f = nanousd_attribf(shader_prim, "inputs:opacity", &ok);
        if (ok) params->opacity = f;
    }
    f = nanousd_attribf(shader_prim, "inputs:geometry_opacity_threshold", &ok);
    if (ok) params->opacity_threshold = f;

    // --- IOR ---
    f = nanousd_attribf(shader_prim, "inputs:ior", &ok);
    if (!ok) f = nanousd_attribf(shader_prim, "inputs:glass_ior", &ok);
    if (ok) params->ior = f;

    // --- Clearcoat (OmniPBR_ClearCoat) ---
    f = nanousd_attribf(shader_prim, "inputs:clearcoat_weight", &ok);
    if (!ok) f = nanousd_attribf(shader_prim, "inputs:clearcoat", &ok);
    if (ok) params->clearcoat = f;

    f = nanousd_attribf(shader_prim, "inputs:clearcoat_roughness", &ok);
    if (ok) params->clearcoat_roughness = f;

    // --- AO / Occlusion ---
    f = nanousd_attribf(shader_prim, "inputs:ao_to_diffuse", &ok);
    if (ok) params->occlusion = f;

    // --- Normal scale ---
    f = nanousd_attribf(shader_prim, "inputs:bump_factor", &ok);
    if (!ok) f = nanousd_attribf(shader_prim, "inputs:normal_scale", &ok);
    if (ok) params->normal_scale = f;

    // --- Texture maps ---
    // OmniPBR uses asset-typed attributes directly on the shader prim.
    // Generated MDL often uses names such as AlbedoTexture and MergeMapInput.
    struct { const char* input; int slot; } tex_inputs[] = {
        {"inputs:diffuse_texture",           TEX_DIFFUSE_COLOR},
        {"inputs:reflectionroughness_texture", TEX_ROUGHNESS},
        {"inputs:metallic_texture",          TEX_METALLIC},
        {"inputs:normalmap_texture",         TEX_NORMAL},
        {"inputs:emissive_mask_texture",     TEX_EMISSIVE_COLOR},
        {"inputs:ao_texture",               TEX_OCCLUSION},
        {"inputs:opacity_texture",           TEX_OPACITY},
        /* OmniSurface / OmniPBRBase MDL */
        {"inputs:diffuse_reflection_color_image", TEX_DIFFUSE_COLOR},
        {"inputs:geometry_normal_image",     TEX_NORMAL},
        {"inputs:geometry_opacity_image",    TEX_OPACITY},
        {"inputs:specular_reflection_roughness_image", TEX_ROUGHNESS},
        {"inputs:metalness_image",           TEX_METALLIC},
        {"inputs:emission_color_image",      TEX_EMISSIVE_COLOR},
        /* Generated MDL texture inputs */
        {"inputs:AlbedoTexture",             TEX_DIFFUSE_COLOR},
        {"inputs:BaseColor_Texture",         TEX_DIFFUSE_COLOR},
        {"inputs:TextureSelection",          TEX_DIFFUSE_COLOR},
        {"inputs:Text",                      TEX_DIFFUSE_COLOR},
        {"inputs:MainNormalInput",           TEX_NORMAL},
        {"inputs:MergeMapInput",             TEX_ROUGHNESS},  // ORM packed; shared across slots below
    };

    bool generated_mdl_texture_inputs = false;
    for (auto& ti : tex_inputs) {
        const char* file = nanousd_attribasset(shader_prim, ti.input, &ok);
        if (getenv("NUSD_MAT_DIAG")) {
            const char* pp = nanousd_path(shader_prim);
            fprintf(stderr, "[mat_diag] shader=%s %s ok=%d file='%s'\n",
                    pp ? pp : "?", ti.input, ok, file ? file : "");
        }
        if (ok && file && file[0] != '\0') {
            if (strcmp(ti.input, "inputs:AlbedoTexture") == 0 ||
                strcmp(ti.input, "inputs:BaseColor_Texture") == 0 ||
                strcmp(ti.input, "inputs:TextureSelection") == 0 ||
                strcmp(ti.input, "inputs:Text") == 0 ||
                strcmp(ti.input, "inputs:MainNormalInput") == 0 ||
                strcmp(ti.input, "inputs:MergeMapInput") == 0) {
                generated_mdl_texture_inputs = true;
            }
            int tex_idx = load_texture_dedup(file, scene_dir, tex_paths,
                                             textures, stage, shader_prim);
            if (tex_idx >= 0 && params->tex_indices[ti.slot] < 0) {
                params->tex_indices[ti.slot] = tex_idx;
                // Propagate UDIM grid dimensions
                MaterialTexture& tex = textures[tex_idx];
                if (tex.udim_cols > 0 && tex.udim_cols > (int)params->udim_scale_u)
                    params->udim_scale_u = (float)tex.udim_cols;
                if (tex.udim_rows > 0 && tex.udim_rows > (int)params->udim_scale_v)
                    params->udim_scale_v = (float)tex.udim_rows;
            }
        }
    }
    if (generated_mdl_texture_inputs)
        params->v_flip = 0;

    // --- Fallback: scan ALL asset-type attributes for texture paths ---
    // MDL materials in USDC often store texture paths as attribute names/values
    // instead of the standard inputs:*_texture naming. Detect these by file
    // extension and use filename heuristics to assign to the correct slot.
    {
        int na = nanousd_nattribs(shader_prim);
        for (int a = 0; a < na; a++) {
            const char* aname = nanousd_attribname(shader_prim, a);
            if (!aname) continue;

            // Try to get the value as an asset path
            int aok;
            const char* aval = nanousd_attribasset(shader_prim, aname, &aok);
            const char* tex_path = nullptr;
            if (aok && aval && aval[0] != '\0') {
                // Check if this looks like a texture file
                std::string val(aval);
                std::string lower_val = val;
                std::transform(lower_val.begin(), lower_val.end(),
                               lower_val.begin(), ::tolower);
                if (lower_val.find(".png") != std::string::npos ||
                    lower_val.find(".jpg") != std::string::npos ||
                    lower_val.find(".jpeg") != std::string::npos ||
                    lower_val.find(".exr") != std::string::npos ||
                    lower_val.find(".tga") != std::string::npos ||
                    lower_val.find(".bmp") != std::string::npos ||
                    lower_val.find(".hdr") != std::string::npos) {
                    tex_path = aval;
                }
            }
            if (!tex_path) continue;

            // Determine slot by filename heuristics
            std::string fn(tex_path);
            std::string lower_fn = fn;
            std::transform(lower_fn.begin(), lower_fn.end(),
                           lower_fn.begin(), ::tolower);

            // Skip MDL source asset references
            if (lower_fn.find(".mdl") != std::string::npos) continue;

            int slot = -1;
            bool is_orm = false;

            /* Generated game-asset naming often uses suffixes: _D, _N, _ORM.
             * Extract the basename's final _X token to classify. */
            std::string base_no_ext = fs::path(lower_fn).stem().string();
            size_t last_us = base_no_ext.rfind('_');
            std::string suffix = (last_us != std::string::npos)
                                     ? base_no_ext.substr(last_us + 1) : "";
            /* Also check attribute name for explicit hints */
            std::string lower_attr(aname);
            std::transform(lower_attr.begin(), lower_attr.end(),
                           lower_attr.begin(), ::tolower);
            if (lower_attr.find("maskselection") != std::string::npos ||
                lower_attr.find("alpha_selection") != std::string::npos ||
                lower_attr.find("alphaselection") != std::string::npos ||
                lower_attr.find("sampler_masks") != std::string::npos ||
                (lower_attr.find("mask") != std::string::npos &&
                 lower_attr.find("opacity") == std::string::npos &&
                 lower_attr.find("cutout") == std::string::npos)) {
                continue;
            }

            if (suffix == "orm" ||
                lower_fn.find("_orm") != std::string::npos ||
                lower_fn.find("orm.") != std::string::npos ||
                lower_attr.find("mergemap") != std::string::npos) {
                is_orm = true;
            } else if (suffix == "d" || suffix == "c" || suffix == "diff" || suffix == "diffuse" ||
                       suffix == "albedo" || suffix == "basecolor" ||
                       lower_fn.find("_c.") != std::string::npos ||
                       lower_fn.find("albedo") != std::string::npos ||
                       lower_fn.find("diffuse") != std::string::npos ||
                       lower_fn.find("basecolor") != std::string::npos ||
                       lower_fn.find("base_color") != std::string::npos ||
                       lower_attr.find("albedo") != std::string::npos ||
                       lower_attr.find("basecolor") != std::string::npos) {
                slot = TEX_DIFFUSE_COLOR;
            } else if (suffix == "n" || suffix == "nrm" || suffix == "norm" ||
                       lower_fn.find("normal") != std::string::npos ||
                       lower_attr.find("normal") != std::string::npos) {
                slot = TEX_NORMAL;
            } else if (suffix == "r" || suffix == "rgh" ||
                       lower_fn.find("rough") != std::string::npos ||
                       lower_attr.find("rough") != std::string::npos) {
                slot = TEX_ROUGHNESS;
            } else if ((suffix == "m" &&
                        (lower_fn.find("metal") != std::string::npos ||
                         lower_attr.find("metal") != std::string::npos ||
                         lower_attr.find("metalness") != std::string::npos)) ||
                       suffix == "met" ||
                       lower_fn.find("metal") != std::string::npos ||
                       lower_attr.find("metal") != std::string::npos) {
                slot = TEX_METALLIC;
            } else if (suffix == "e" || suffix == "emi" ||
                       lower_fn.find("emissive") != std::string::npos ||
                       lower_fn.find("emission") != std::string::npos) {
                slot = TEX_EMISSIVE_COLOR;
            } else if (suffix == "ao" || suffix == "occ" ||
                       lower_fn.find("occlusion") != std::string::npos ||
                       lower_fn.find("_ao") != std::string::npos) {
                slot = TEX_OCCLUSION;
            } else if (suffix == "o" || suffix == "op" || suffix == "a" ||
                       lower_fn.find("opacity") != std::string::npos ||
                       lower_fn.find("alpha") != std::string::npos) {
                slot = TEX_OPACITY;
            }

            // Only assign if the slot is empty (don't override explicit bindings)
            if (is_orm) {
                int tex_idx = load_texture_dedup(tex_path, scene_dir,
                                                  tex_paths, textures, stage);
                if (tex_idx >= 0) {
                    if (params->tex_indices[TEX_OCCLUSION] < 0)
                        params->tex_indices[TEX_OCCLUSION] = tex_idx;
                    if (params->tex_indices[TEX_ROUGHNESS] < 0)
                        params->tex_indices[TEX_ROUGHNESS] = tex_idx;
                    if (params->tex_indices[TEX_METALLIC] < 0)
                        params->tex_indices[TEX_METALLIC] = tex_idx;
                    fprintf(stderr, "material:   ORM texture: %s → slots 2,3,5\n",
                            tex_path);
                }
            } else if (slot >= 0 && params->tex_indices[slot] < 0) {
                int tex_idx = load_texture_dedup(tex_path, scene_dir,
                                                  tex_paths, textures, stage);
                if (tex_idx >= 0) {
                    params->tex_indices[slot] = tex_idx;
                    MaterialTexture& tex = textures[tex_idx];
                    if (tex.udim_cols > 0 && tex.udim_cols > (int)params->udim_scale_u)
                        params->udim_scale_u = (float)tex.udim_cols;
                    if (tex.udim_rows > 0 && tex.udim_rows > (int)params->udim_scale_v)
                        params->udim_scale_v = (float)tex.udim_rows;
                }
            }
        }
    }

    /* Generic MDL sourceAsset support. This reads the public MDL source
     * parameter defaults and obvious texture_2d references into the same PBR
     * fields used by Vulkan raster and RT. It is intentionally renderer-local:
     * no wrapper-side USD rewrites, and no hard dependency on the MDL SDK for
     * the default path. */
    apply_generic_mdl_source_material(shader_prim, params, tex_paths, textures,
                                      scene_dir, stage, !mdl_sdk_applied);

    /* OmniSurface/OmniPBR graphs often store file_texture nodes as
     * Material children with `inputs:texture` instead of UsdUVTexture
     * `inputs:file`, and connect the surface inputs through blend/multiply
     * nodegraphs. Until we evaluate the full MDL graph, bind the obvious
     * leaf texture nodes by filename/child-name heuristics. This is enough
     * for DSX MountainNevada: albedo/normal/opacity are direct child
     * texture nodes, while the utility map remains a graph-only mask. */
    {
        NanousdPrim mat_parent = nanousd_parent(shader_prim);
        if (mat_parent) {
            int nchildren = nanousd_nchildren(mat_parent);
            for (int c = 0; c < nchildren; c++) {
                NanousdPrim child = nanousd_child(mat_parent, c);
                if (!child) continue;

                const char* file = nanousd_attribasset(child, "inputs:texture", &ok);
                if (!ok || !file || !file[0])
                    file = nanousd_attribasset(child, "inputs:file", &ok);
                if (!ok || !file || !file[0]) {
                    nanousd_freeprim(child);
                    continue;
                }

                std::string hint;
                const char* child_path = nanousd_path(child);
                if (child_path) hint = child_path;
                hint += " ";
                hint += file;

                bool is_orm = false;
                int slot = classify_texture_slot(hint.c_str(), file, &is_orm);
                if (is_orm) {
                    int tex_idx = load_texture_dedup(file, scene_dir,
                                                     tex_paths, textures,
                                                     stage, child);
                    if (tex_idx >= 0) {
                        if (params->tex_indices[TEX_OCCLUSION] < 0)
                            params->tex_indices[TEX_OCCLUSION] = tex_idx;
                        if (params->tex_indices[TEX_ROUGHNESS] < 0)
                            params->tex_indices[TEX_ROUGHNESS] = tex_idx;
                        if (params->tex_indices[TEX_METALLIC] < 0)
                            params->tex_indices[TEX_METALLIC] = tex_idx;
                        fprintf(stderr,
                                "material:   graph ORM texture: %s -> AO/R/M slots\n",
                                file);
                    }
                } else if (slot >= 0 && params->tex_indices[slot] < 0) {
                    int tex_idx = load_texture_dedup(file, scene_dir,
                                                     tex_paths, textures,
                                                     stage, child);
                    if (tex_idx >= 0) {
                        params->tex_indices[slot] = tex_idx;
                        propagate_udim_scale(params, textures[tex_idx]);
                        fprintf(stderr,
                                "material:   graph texture: %s -> slot %d\n",
                                file, slot);
                    }
                }

                nanousd_freeprim(child);
            }
            nanousd_freeprim(mat_parent);
        }
    }

    // --- ORM packed texture detection ---
    // NVIDIA assets commonly use a single _ORM texture: R=AO, G=Roughness, B=Metallic.
    // If we loaded a roughness texture whose path contains "ORM" or "orm", share it
    // across the roughness, metallic, and occlusion slots so the shader can decode
    // the packed channels (R→AO, G→Roughness, B→Metallic).
    {
        int orm_idx = -1;
        // Check roughness slot first (most common binding for ORM in OmniPBR)
        if (params->tex_indices[TEX_ROUGHNESS] >= 0) {
            int idx = params->tex_indices[TEX_ROUGHNESS];
            if (idx < (int)textures.size()) {
                std::string tp(textures[idx].path);
                std::string lower_tp = tp;
                std::transform(lower_tp.begin(), lower_tp.end(),
                               lower_tp.begin(), ::tolower);
                if (lower_tp.find("orm") != std::string::npos ||
                    lower_tp.find("_orm") != std::string::npos) {
                    orm_idx = idx;
                }
            }
        }
        // Also check any texture slot that has an ORM path
        if (orm_idx < 0) {
            for (int s = 0; s < MAX_MATERIAL_TEXTURES; s++) {
                if (params->tex_indices[s] < 0) continue;
                int idx = params->tex_indices[s];
                if (idx >= (int)textures.size()) continue;
                std::string tp(textures[idx].path);
                std::string lower_tp = tp;
                std::transform(lower_tp.begin(), lower_tp.end(),
                               lower_tp.begin(), ::tolower);
                if (lower_tp.find("_orm") != std::string::npos) {
                    orm_idx = idx;
                    break;
                }
            }
        }
        if (orm_idx >= 0) {
            // Share the ORM texture across all three channel slots
            if (params->tex_indices[TEX_OCCLUSION] < 0)
                params->tex_indices[TEX_OCCLUSION] = orm_idx;
            if (params->tex_indices[TEX_ROUGHNESS] < 0)
                params->tex_indices[TEX_ROUGHNESS] = orm_idx;
            if (params->tex_indices[TEX_METALLIC] < 0)
                params->tex_indices[TEX_METALLIC] = orm_idx;
            fprintf(stderr, "material: ORM packed texture detected (idx=%d): "
                    "AO→slot5, Rough→slot2, Metal→slot3\n", orm_idx);
        }
    }

    if (material_uses_mdl_baked_masked_albedo(params, textures) ||
        material_uses_mdl_baked_body_masked_albedo(params, textures)) {
        /* The baked texture already contains the MDL ColorAlbedo/BaseColor
         * tint and MaskSelection lerp. Keep the shader from multiplying that
         * result by the original tint a second time. Generated MDLs
         * also double-flip their V coordinate before texture lookup, so the
         * baked texture should be sampled with the normal USD UV convention. */
        params->base_color[0] = 1.0f;
        params->base_color[1] = 1.0f;
        params->base_color[2] = 1.0f;
        params->v_flip = 0;
        if (getenv("NUSD_MAT_DIAG")) {
            int idx = params->tex_indices[TEX_DIFFUSE_COLOR];
            fprintf(stderr,
                    "[mat_diag] MDL baked masked albedo finalised: tex=%d "
                    "base=(1,1,1) v_flip=%d path=%s\n",
                    idx, params->v_flip,
                    (idx >= 0 && idx < (int)textures.size())
                        ? textures[idx].path : "?");
        }
    }
    if (material_uses_mdl_baked_plastic_wrap_albedo(params, textures)) {
        /* The baked texture folds generated cardboard base,
         * plastic-wrap base, plastic opacity, and plastic normal-mask
         * blend. Do not multiply it by the intermediate tint again. */
        params->base_color[0] = 1.0f;
        params->base_color[1] = 1.0f;
        params->base_color[2] = 1.0f;
        params->v_flip = 0;
        if (getenv("NUSD_MAT_DIAG")) {
            int idx = params->tex_indices[TEX_DIFFUSE_COLOR];
            fprintf(stderr,
                    "[mat_diag] MDL plastic-wrap albedo finalised: tex=%d "
                    "base=(1,1,1) v_flip=%d path=%s\n",
                    idx, params->v_flip,
                    (idx >= 0 && idx < (int)textures.size())
                        ? textures[idx].path : "?");
        }
    }

    /* If a diffuse texture was bound, the MDL "tint" base_color would
     * double-darken the texture sample. Clamp base_color up to white when a
     * diffuse texture is present, preserving only tints above gray. */
    if (params->tex_indices[TEX_DIFFUSE_COLOR] >= 0) {
        float max_c = params->base_color[0];
        if (params->base_color[1] > max_c) max_c = params->base_color[1];
        if (params->base_color[2] > max_c) max_c = params->base_color[2];
        if (max_c < 0.5f) {
            params->base_color[0] = 1.0f;
            params->base_color[1] = 1.0f;
            params->base_color[2] = 1.0f;
        }
    }

    {
        int eb_ok = 0;
        int eb = nanousd_attribb(shader_prim, "inputs:enable_emission", &eb_ok);
        if (eb_ok && !eb) {
            params->emissive_color[0] = 0.0f;
            params->emissive_color[1] = 0.0f;
            params->emissive_color[2] = 0.0f;
            params->emissive_color[3] = 0.0f;
        }
    }

    {
        int mok = 0, sub_ok = 0;
        const char* mdl_asset = nanousd_attribasset(shader_prim,
                                                    "info:mdl:sourceAsset",
                                                    &mok);
        const char* mdl_sub = nanousd_attrib_token(
            shader_prim, "info:mdl:sourceAsset:subIdentifier", &sub_ok);
        std::string mdl_key;
        if (mok && mdl_asset) mdl_key += lower_ascii(mdl_asset);
        mdl_key += " ";
        if (sub_ok && mdl_sub) mdl_key += lower_ascii(mdl_sub);
        bool has_any_texture = false;
        for (int s = 0; s < MAX_MATERIAL_TEXTURES; ++s) {
            if (params->tex_indices[s] >= 0) {
                has_any_texture = true;
                break;
            }
        }
        if (!has_any_texture &&
            mdl_key.find("metal_black_paint") != std::string::npos) {
            /* DSX GB300 references NVIDIA's remote Automotive/Imperfect
             * Metal_Black_Paint.mdl without packaging the MDL body or any USD
             * inputs. Prefer OVRTX's remote texture set; if the network/cache
             * path is unavailable, keep the scalar painted-metal preset so
             * unresolved remote presets fail dark and glossy.
             * NOTE: this hardcoded scalar preset masks an MDL asset-resolution
             * failure, so warn once instead of silently substituting it. */
            static bool s_warned_mbp = false;
            if (!s_warned_mbp) {
                s_warned_mbp = true;
                fprintf(stderr,
                        "[nusd] WARNING: Metal_Black_Paint MDL '%s' resolved no "
                        "textures; applying an empirical scalar painted-metal "
                        "fallback (real asset/remote textures unavailable). This "
                        "is a known hardcoded preset, not the authored material.\n",
                        (mok && mdl_asset) ? mdl_asset : "<unknown>");
            }
            params->base_color[0] = 0.012f;
            params->base_color[1] = 0.014f;
            params->base_color[2] = 0.016f;
            params->metallic = 0.0f;
            params->roughness = 0.42f;
            params->clearcoat = 0.45f;
            params->clearcoat_roughness = 0.22f;
            bind_metal_black_paint_remote_textures(params, tex_paths, textures,
                                                   scene_dir, stage);
        }
    }

    fprintf(stderr, "material: read OmniPBR params: base=(%.2f,%.2f,%.2f) "
            "metal=%.2f rough=%.2f emissive=(%.2f,%.2f,%.2f)*%.1f "
            "opacity=%.2f cut=%.2f udim=%.0fx%.0f "
            "slots D=%d N=%d R=%d M=%d E=%d AO=%d OP=%d\n",
            params->base_color[0], params->base_color[1], params->base_color[2],
            params->metallic, params->roughness,
            params->emissive_color[0], params->emissive_color[1],
            params->emissive_color[2], params->emissive_color[3],
            params->opacity, params->opacity_threshold,
            params->udim_scale_u, params->udim_scale_v,
            params->tex_indices[TEX_DIFFUSE_COLOR],
            params->tex_indices[TEX_NORMAL],
            params->tex_indices[TEX_ROUGHNESS],
            params->tex_indices[TEX_METALLIC],
            params->tex_indices[TEX_EMISSIVE_COLOR],
            params->tex_indices[TEX_OCCLUSION],
            params->tex_indices[TEX_OPACITY]);
}

/* ---- Read OmniGlass material properties ---- */

static void read_omniglass_material(NanousdPrim shader_prim,
                                    MaterialParams* params,
                                    std::vector<std::string>& tex_paths,
                                    std::vector<MaterialTexture>& textures,
                                    const char* scene_dir,
                                    NanousdStage stage = nullptr)
{
    // Glass defaults: transparent, smooth, dielectric
    params->base_color[0] = 1.0f;
    params->base_color[1] = 1.0f;
    params->base_color[2] = 1.0f;
    params->base_color[3] = 1.0f;
    params->emissive_color[0] = 0.0f;
    params->emissive_color[1] = 0.0f;
    params->emissive_color[2] = 0.0f;
    params->emissive_color[3] = 0.0f;
    params->metallic = 0.0f;
    params->roughness = 0.0f;
    params->opacity = 0.1f; // Semi-transparent by default for glass
    params->ior = 1.5f;
    params->occlusion = 1.0f;
    params->clearcoat = 0.0f;
    params->clearcoat_roughness = 0.01f;
    params->normal_scale = 1.0f;
    params->use_vertex_color = 0;
    params->udim_scale_u = 1.0f;
    params->udim_scale_v = 1.0f;
    set_default_mdl_uv_transform(params);

    for (int i = 0; i < MAX_MATERIAL_TEXTURES; i++)
        params->tex_indices[i] = -1;

    if (!shader_prim) return;

    int ok;
    float v3[3];
    float f;

    // Glass color (tint)
    if (nanousd_attribv3f(shader_prim, "inputs:glass_color", v3)) {
        params->base_color[0] = v3[0];
        params->base_color[1] = v3[1];
        params->base_color[2] = v3[2];
    }

    // Glass IOR
    f = nanousd_attribf(shader_prim, "inputs:glass_ior", &ok);
    if (!ok) f = nanousd_attribf(shader_prim, "inputs:ior", &ok);
    if (ok) params->ior = f;

    // Roughness
    f = nanousd_attribf(shader_prim, "inputs:frosting_roughness", &ok);
    if (!ok) f = nanousd_attribf(shader_prim, "inputs:roughness", &ok);
    if (ok) params->roughness = f;

    // Thin-walled glass is more transparent
    f = nanousd_attribf(shader_prim, "inputs:thin_walled", &ok);
    if (ok && f > 0.5f) {
        params->opacity = 0.05f;
    }

    fprintf(stderr, "material: read OmniGlass params: color=(%.2f,%.2f,%.2f) "
            "ior=%.2f rough=%.2f opacity=%.2f\n",
            params->base_color[0], params->base_color[1], params->base_color[2],
            params->ior, params->roughness, params->opacity);
}

/* ---- Find surface shader prim under a Material ---- */

static NanousdPrim find_surface_shader(NanousdStage stage, NanousdPrim material_prim)
{
    // Check surface relationship/connection targets. MaterialX USD authors
    // outputs:mtlx:surface as an attribute connection, while UsdPreviewSurface
    // and MDL scenes may use either relationships or output connections.
    const char* surface_outputs[] = {
        "outputs:mtlx:surface",
        "outputs:mdl:surface",
        "outputs:ri:surface",
        "outputs:surface",
        "outputs:glslfx:surface"
    };
    for (const char* output_name : surface_outputs) {
        int ntargets = nanousd_nreltargets(material_prim, output_name);
        if (ntargets > 0) {
            const char* target = nanousd_reltarget(material_prim, output_name, 0);
            if (target) {
                // Target may be like /Materials/Mat1/Shader.outputs:surface
                // We want the prim part (before the property)
                std::string target_str(target);
                size_t dot = target_str.find('.');
                if (dot != std::string::npos) {
                    target_str = target_str.substr(0, dot);
                }
                NanousdPrim shader = nanousd_primpath(stage, target_str.c_str());
                if (shader) return shader;
            }
        }

        int nconn = nanousd_nconnections(material_prim, output_name);
        if (nconn > 0) {
            const char* target = nanousd_connection(material_prim, output_name, 0);
            std::string prim_path, prop_name;
            if (split_connection_path(target, prim_path, prop_name)) {
                NanousdPrim shader = nanousd_primpath(stage, prim_path.c_str());
                if (shader) return shader;
            }
        }
    }

    // Fallback: search children for Shader prims
    // Accept UsdPreviewSurface, OmniPBR, OmniGlass, and other MDL materials
    int nchildren = nanousd_nchildren(material_prim);
    NanousdPrim best_mdl = nullptr;
    for (int i = 0; i < nchildren; i++) {
        NanousdPrim child = nanousd_child(material_prim, i);
        if (!child) continue;

        const char* type = nanousd_typename(child);
        if (type && strcmp(type, "Shader") == 0) {
            int ok;
            const char* info_id = nanousd_attrib_token(child, "info:id", &ok);
            if (ok && info_id) {
                if (strcmp(info_id, "UsdPreviewSurface") == 0) {
                    if (best_mdl) nanousd_freeprim(best_mdl);
                    return child; // Prefer UsdPreviewSurface
                }
                if (strcmp(info_id, "PxrDisneyBsdf") == 0 ||
                    strcmp(info_id, "PxrSurface") == 0 ||
                    strcmp(info_id, "PxrVolume") == 0) {
                    if (best_mdl) nanousd_freeprim(best_mdl);
                    return child;
                }
                if (strstr(info_id, "ND_standard_surface") ||
                    strstr(info_id, "ND_open_pbr_surface")) {
                    if (best_mdl) nanousd_freeprim(best_mdl);
                    return child;
                }
                // Check for MDL material (info:id or mdl:sourceAsset:subIdentifier)
                if (strstr(info_id, "mdlMaterial") ||
                    strstr(info_id, "OmniPBR") ||
                    strstr(info_id, "OmniGlass") ||
                    strstr(info_id, "OmniSurface")) {
                    if (!best_mdl) {
                        best_mdl = child;
                        continue; // Don't free — keep as fallback
                    }
                }
            }
            // Also check info:mdl:sourceAsset:subIdentifier
            const char* mdl_sub = nanousd_attrib_token(child, "info:mdl:sourceAsset:subIdentifier", &ok);
            if (ok && mdl_sub && !best_mdl) {
                best_mdl = child;
                continue;
            }
            const char* mdl_asset = nanousd_attribasset(child, "info:mdl:sourceAsset", &ok);
            if (ok && mdl_asset && mdl_asset[0] && !best_mdl) {
                best_mdl = child;
                continue;
            }
        }
        if (child != best_mdl) nanousd_freeprim(child);
    }

    return best_mdl; // May be nullptr or an MDL shader prim
}

/* ---- Public API ---- */

extern "C" int materialx_init(void)
{
    if (g_materialx_initialized) return 1;

    try {
        // Create the standard library
        g_stdlib = mx::createDocument();

        // Build search path for MaterialX libraries
        const char* mtlx_env = getenv("MATERIALX_SEARCH_PATH");
        mx::FileSearchPath searchPath;

        if (mtlx_env) {
            searchPath.append(mx::FilePath(mtlx_env));
        }

#ifdef MATERIALX_SEARCH_PATH
        searchPath.append(mx::FilePath(MATERIALX_SEARCH_PATH));
        {
            mx::FilePath libPath(MATERIALX_SEARCH_PATH);
            searchPath.append(libPath.getParentPath());
        }
#endif

        searchPath.append(mx::FilePath("/usr/share/MaterialX/libraries"));
        searchPath.append(mx::FilePath("/usr/local/share/MaterialX/libraries"));
        searchPath.append(mx::FilePath("libraries"));

        // Load standard libraries
        mx::StringSet loadedLibs;
        try {
            loadedLibs = mx::loadLibraries({}, searchPath, g_stdlib);
        } catch (...) {}

        if (loadedLibs.empty()) {
            try {
                loadedLibs = mx::loadLibraries({"libraries"}, searchPath, g_stdlib);
            } catch (...) {}
        }

        fprintf(stderr, "materialx: loaded %zu standard library definitions\n",
                loadedLibs.size());

        // Create Vulkan GLSL shader generator
        g_shader_gen = mx::VkShaderGenerator::create();

        // Set up Color Management System
        auto cms = mx::DefaultColorManagementSystem::create(
            g_shader_gen->getTarget());
        if (cms) {
            g_shader_gen->setColorManagementSystem(cms);
            cms->loadLibrary(g_stdlib);
            fprintf(stderr, "materialx: color management system initialized\n");
        }

        // Set up Unit System (distance + angle)
        auto unitSystem = mx::UnitSystem::create(g_shader_gen->getTarget());
        if (unitSystem) {
            g_shader_gen->setUnitSystem(unitSystem);
            unitSystem->loadLibrary(g_stdlib);
            unitSystem->setUnitConverterRegistry(mx::UnitConverterRegistry::create());

            mx::UnitTypeDefPtr distTypeDef = g_stdlib->getUnitTypeDef("distance");
            if (distTypeDef) {
                unitSystem->getUnitConverterRegistry()->addUnitConverter(
                    distTypeDef, mx::LinearUnitConverter::create(distTypeDef));
            }
            mx::UnitTypeDefPtr angleTypeDef = g_stdlib->getUnitTypeDef("angle");
            if (angleTypeDef) {
                unitSystem->getUnitConverterRegistry()->addUnitConverter(
                    angleTypeDef, mx::LinearUnitConverter::create(angleTypeDef));
            }
            fprintf(stderr, "materialx: unit system initialized\n");
        }

        // Load struct type definitions from stdlib

        // Create GenContext
        g_gen_context = new mx::GenContext(g_shader_gen);

        // Configure options
        g_gen_context->getOptions().targetColorSpaceOverride = "lin_rec709";
        g_gen_context->getOptions().fileTextureVerticalFlip = true;
        g_gen_context->getOptions().targetDistanceUnit = "meter";

        // Register source code search paths
        g_gen_context->registerSourceCodeSearchPath(searchPath);

        // Register shader metadata
        g_shader_gen->registerShaderMetadata(g_stdlib, *g_gen_context);

        // Bind light shader NodeDefs from the standard library.
        // This is critical for generating the sampleLightSource() function.
        // The stdlib has ND_point_light, ND_directional_light, ND_spot_light.
        const char* lightDefNames[] = {
            "ND_point_light", "ND_directional_light", "ND_spot_light"
        };
        unsigned int boundLights = 0;
        for (unsigned int i = 0; i < 3; i++) {
            mx::NodeDefPtr nodedef = g_stdlib->getNodeDef(lightDefNames[i]);
            if (nodedef) {
                mx::HwShaderGenerator::bindLightShader(*nodedef, i + 1,
                                                       *g_gen_context);
                fprintf(stderr, "materialx: bound light '%s' (id=%u)\n",
                        lightDefNames[i], i + 1);
                boundLights++;
            }
        }

        if (boundLights > 0) {
            g_gen_context->getOptions().hwMaxActiveLightSources = boundLights;
            fprintf(stderr, "materialx: %u light sources registered\n",
                    boundLights);
        } else {
            g_gen_context->getOptions().hwMaxActiveLightSources = 1;
            fprintf(stderr, "materialx: warning - no light nodedefs found\n");
        }

        g_materialx_initialized = true;
        fprintf(stderr, "materialx: initialized successfully\n");
        return 1;

    } catch (const std::exception& e) {
        fprintf(stderr, "materialx: init failed: %s\n", e.what());
        return 0;
    }
}

extern "C" void materialx_shutdown(void)
{
    if (!g_materialx_initialized) return;

    delete g_gen_context;
    g_gen_context = nullptr;
    g_shader_gen = nullptr;
    g_stdlib = nullptr;
    g_materialx_initialized = false;
}

static int extract_quoted_name_after(const std::string& line,
                                     const char* keyword,
                                     char out[256])
{
    size_t k = line.find(keyword);
    if (k == std::string::npos) return 0;
    size_t q0 = line.find('"', k + strlen(keyword));
    if (q0 == std::string::npos) return 0;
    size_t q1 = line.find('"', q0 + 1);
    if (q1 == std::string::npos || q1 <= q0 + 1) return 0;
    size_t n = q1 - q0 - 1;
    if (n >= 256) n = 255;
    memcpy(out, line.data() + q0 + 1, n);
    out[n] = '\0';
    return 1;
}

static void build_binding_hints_from_usda_layers(MaterialCollection* mc,
                                                 NanousdStage stage)
{
    if (!mc || !stage || mc->nmaterials <= 0) return;
    const char* disable = getenv("NUSD_BINDING_HINTS");
    if (disable && disable[0] == '0') return;

    std::unordered_map<std::string, int> material_by_path;
    std::unordered_map<std::string, int> material_by_name;
    material_by_path.reserve((size_t)mc->nmaterials * 2u);
    material_by_name.reserve((size_t)mc->nmaterials * 2u);
    for (int i = 0; i < mc->nmaterials; i++) {
        if (mc->materials[i].prim_path[0] &&
            material_by_path.find(mc->materials[i].prim_path) == material_by_path.end()) {
            material_by_path.emplace(mc->materials[i].prim_path, i);
        }
        if (mc->materials[i].name[0] &&
            material_by_name.find(mc->materials[i].name) == material_by_name.end()) {
            material_by_name.emplace(mc->materials[i].name, i);
        }
    }

    std::vector<MaterialBindingHint> hints;
    int n_layers = nanousd_stage_n_layers(stage);
    for (int li = 0; li < n_layers; li++) {
        const char* layer_path = nanousd_stage_layer_path(stage, li);
        if (!layer_path || !*layer_path) continue;
        std::string path(layer_path);
        if (path.size() < 5 || path.substr(path.size() - 5) != ".usda")
            continue;

        std::ifstream in(path);
        if (!in) continue;

        char current_prim[256] = {0};
        std::string line;
        while (std::getline(in, line)) {
            char name[256];
            if (extract_quoted_name_after(line, "over ", name) ||
                extract_quoted_name_after(line, "def Mesh ", name) ||
                extract_quoted_name_after(line, "def Xform ", name)) {
                memcpy(current_prim, name, sizeof(current_prim));
                current_prim[sizeof(current_prim) - 1] = '\0';
            }

            if (current_prim[0] == '\0')
                continue;
            if (line.find("material:binding") == std::string::npos)
                continue;

            size_t lt = line.find('<');
            size_t gt = line.find('>', lt == std::string::npos ? 0 : lt + 1);
            if (lt == std::string::npos || gt == std::string::npos || gt <= lt + 1)
                continue;

            std::string target = line.substr(lt + 1, gt - lt - 1);
            int mat_idx = -1;
            auto exact = material_by_path.find(target);
            if (exact != material_by_path.end()) {
                mat_idx = exact->second;
            } else {
                const char* slash = strrchr(target.c_str(), '/');
                const char* target_name = slash ? slash + 1 : target.c_str();
                auto by_name = material_by_name.find(target_name);
                if (by_name != material_by_name.end())
                    mat_idx = by_name->second;
            }
            if (mat_idx < 0) continue;

            MaterialBindingHint hint = {};
            snprintf(hint.prim_name, sizeof(hint.prim_name), "%s", current_prim);
            hint.material_index = mat_idx;
            hints.push_back(hint);
        }
    }

    if (hints.empty()) return;
    std::sort(hints.begin(), hints.end(),
              [](const MaterialBindingHint& a, const MaterialBindingHint& b) {
                  int c = strcmp(a.prim_name, b.prim_name);
                  if (c != 0) return c < 0;
                  return a.material_index < b.material_index;
              });
    hints.erase(std::unique(hints.begin(), hints.end(),
                            [](const MaterialBindingHint& a,
                               const MaterialBindingHint& b) {
                                return strcmp(a.prim_name, b.prim_name) == 0;
                            }),
                hints.end());

    mc->binding_hints = (MaterialBindingHint*)calloc(
        hints.size(), sizeof(MaterialBindingHint));
    if (!mc->binding_hints) return;
    memcpy(mc->binding_hints, hints.data(),
           hints.size() * sizeof(MaterialBindingHint));
    mc->nbinding_hints = (int)hints.size();
    fprintf(stderr, "material: %d fast binding hints loaded from USDA layers\n",
            mc->nbinding_hints);
}

static void assign_default_material_shader(SceneMaterial& sm,
                                           const SpvResult& default_vert,
                                           const SpvResult& default_frag)
{
    if (!default_vert.ok || !default_frag.ok) return;
    sm.shader.vert_spv = (uint32_t*)malloc(default_vert.size);
    if (sm.shader.vert_spv) {
        memcpy(sm.shader.vert_spv, default_vert.data, default_vert.size);
        sm.shader.vert_size = default_vert.size;
    }
    sm.shader.frag_spv = (uint32_t*)malloc(default_frag.size);
    if (sm.shader.frag_spv) {
        memcpy(sm.shader.frag_spv, default_frag.data, default_frag.size);
        sm.shader.frag_size = default_frag.size;
    }
}

static bool is_usda_material_layer(const fs::path& path)
{
    if (path.extension() != ".usda") return false;
    std::string leaf = path.filename().string();
    if (leaf.rfind("._", 0) == 0) return false;
    return leaf == "materials.usda" || leaf.find("_materials") != std::string::npos;
}

static std::vector<std::string>
collect_usda_material_layer_paths(NanousdStage stage, const char* scene_dir)
{
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    auto add_path = [&](const fs::path& p) {
        if (!is_usda_material_layer(p)) return;
        std::string s = p.string();
        if (s.empty() || !seen.insert(s).second) return;
        out.push_back(s);
    };

    int n_layers = stage ? nanousd_stage_n_layers(stage) : 0;
    for (int li = 0; li < n_layers; ++li) {
        const char* layer_path = nanousd_stage_layer_path(stage, li);
        if (layer_path && *layer_path) add_path(fs::path(layer_path));
    }

    if (scene_dir && *scene_dir) {
        std::error_code ec;
        fs::recursive_directory_iterator it(scene_dir, ec), end;
        for (; !ec && it != end; it.increment(ec)) {
            if (it->is_regular_file(ec))
                add_path(it->path());
        }
    }
    return out;
}

static void collect_missing_usda_material_targets(
    const std::vector<std::string>& layer_paths,
    const std::unordered_set<std::string>& existing_paths,
    std::unordered_set<std::string>& wanted_paths)
{
    for (const std::string& layer_path : layer_paths) {
        std::ifstream in(layer_path);
        if (!in) continue;
        std::string line;
        while (std::getline(in, line)) {
            if (line.find("material:binding") == std::string::npos)
                continue;
            size_t lt = line.find('<');
            size_t gt = line.find('>', lt == std::string::npos ? 0 : lt + 1);
            if (lt == std::string::npos || gt == std::string::npos || gt <= lt + 1)
                continue;
            std::string target = line.substr(lt + 1, gt - lt - 1);
            if (!target.empty() && existing_paths.find(target) == existing_paths.end())
                wanted_paths.insert(target);
        }
    }
}

static bool parse_usda_prim_header(const std::string& line,
                                   std::string& type,
                                   std::string& name)
{
    size_t pos = line.find("def ");
    if (pos == std::string::npos) pos = line.find("over ");
    if (pos == std::string::npos) pos = line.find("class ");
    if (pos == std::string::npos) return false;

    size_t quote = line.find('"', pos);
    if (quote == std::string::npos) return false;
    size_t quote2 = line.find('"', quote + 1);
    if (quote2 == std::string::npos) return false;
    name = line.substr(quote + 1, quote2 - quote - 1);

    std::string head = line.substr(pos, quote - pos);
    std::istringstream hs(head);
    std::string specifier;
    hs >> specifier >> type;
    if (specifier == "over") type = "over";
    return !name.empty();
}

static std::string usda_path_from_stack(
    const std::vector<std::pair<std::string, int>>& stack,
    const std::string& leaf)
{
    std::string path;
    for (const auto& frame : stack) {
        if (frame.first.empty()) continue;
        path += "/";
        path += frame.first;
    }
    path += "/";
    path += leaf;
    return path;
}

static bool parse_float_after_equal(const std::string& text, size_t eq, float* out)
{
    if (!out || eq == std::string::npos || eq + 1 >= text.size()) return false;
    const char* p = text.c_str() + eq + 1;
    while (*p && std::isspace((unsigned char)*p)) ++p;
    char* end = nullptr;
    float v = std::strtof(p, &end);
    if (end == p || !std::isfinite(v)) return false;
    *out = v;
    return true;
}

static bool usda_find_float_input(const std::string& block,
                                  const char* name,
                                  float* out)
{
    std::string key = std::string("inputs:") + name;
    size_t pos = 0;
    while ((pos = block.find(key, pos)) != std::string::npos) {
        size_t line_end = block.find('\n', pos);
        size_t eq = block.find('=', pos);
        if (eq != std::string::npos &&
            (line_end == std::string::npos || eq < line_end) &&
            parse_float_after_equal(block, eq, out)) {
            return true;
        }
        pos += key.size();
    }
    return false;
}

static bool usda_parse_color_tuple(const std::string& block,
                                   size_t open,
                                   float out[3])
{
    if (!out || open == std::string::npos || open + 1 >= block.size())
        return false;
    const char* p = block.c_str() + open + 1;
    for (int c = 0; c < 3; ++c) {
        while (*p && (std::isspace((unsigned char)*p) || *p == ',')) ++p;
        char* end = nullptr;
        float v = std::strtof(p, &end);
        if (end == p || !std::isfinite(v)) return false;
        out[c] = v;
        p = end;
    }
    return true;
}

static bool usda_find_color_input(const std::string& block,
                                  const char* name,
                                  float out[3])
{
    std::string key = std::string("inputs:") + name;
    size_t pos = 0;
    while ((pos = block.find(key, pos)) != std::string::npos) {
        size_t line_end = block.find('\n', pos);
        size_t eq = block.find('=', pos);
        size_t open = block.find('(', eq == std::string::npos ? pos : eq);
        if (eq != std::string::npos && open != std::string::npos &&
            (line_end == std::string::npos || (eq < line_end && open < line_end)) &&
            usda_parse_color_tuple(block, open, out)) {
            return true;
        }
        pos += key.size();
    }
    return false;
}

static void apply_usda_moana_gamma(float rgb[3])
{
    const float gamma = 0.45454547f;
    for (int c = 0; c < 3; ++c)
        rgb[c] = std::pow(std::max(rgb[c], 0.0f), 1.0f / gamma);
}

static bool usda_find_connected_color_correct_input(const std::string& block,
                                                    const char* name,
                                                    float out[3])
{
    if (!name || !out) return false;
    std::string key = std::string("inputs:") + name;
    size_t pos = 0;
    while ((pos = block.find(key, pos)) != std::string::npos) {
        size_t line_end = block.find('\n', pos);
        size_t lt = block.find('<', pos);
        size_t gt = block.find('>', lt == std::string::npos ? pos : lt + 1);
        if (lt == std::string::npos || gt == std::string::npos ||
            (line_end != std::string::npos && lt > line_end)) {
            pos += key.size();
            continue;
        }
        std::string target = block.substr(lt + 1, gt - lt - 1);
        size_t slash = target.find_last_of('/');
        size_t dot = target.find(".outputs", slash == std::string::npos ? 0 : slash);
        if (slash == std::string::npos || dot == std::string::npos || dot <= slash + 1) {
            pos += key.size();
            continue;
        }

        std::string shader_name = target.substr(slash + 1, dot - slash - 1);
        std::string header = std::string("def Shader \"") + shader_name + "\"";
        size_t h = block.find(header);
        if (h == std::string::npos) {
            pos += key.size();
            continue;
        }
        size_t open = block.find('{', h);
        if (open == std::string::npos) {
            pos += key.size();
            continue;
        }

        int depth = 1;
        size_t end = open + 1;
        for (; end < block.size() && depth > 0; ++end) {
            if (block[end] == '{') depth++;
            else if (block[end] == '}') depth--;
        }
        if (depth != 0 || end <= open + 1) {
            pos += key.size();
            continue;
        }

        std::string shader_block = block.substr(open + 1, end - open - 2);
        if (shader_block.find("PxrColorCorrect") != std::string::npos &&
            usda_find_color_input(shader_block, "inputRGB", out)) {
            apply_usda_moana_gamma(out);
            return true;
        }
        pos += key.size();
    }
    return false;
}

static std::string resolve_usda_layer_asset(const fs::path& layer_path,
                                            const std::string& asset)
{
    if (asset.empty()) return std::string();
    fs::path p(asset);
    if (p.is_relative())
        p = layer_path.parent_path() / p;
    p = p.lexically_normal();
    std::error_code ec;
    fs::path canon = fs::weakly_canonical(p, ec);
    return ec ? p.string() : canon.string();
}

static std::string usda_find_surface_ptx(const std::string& block,
                                         const fs::path& layer_path)
{
    std::vector<std::string> candidates;
    size_t pos = 0;
    while ((pos = block.find('@', pos)) != std::string::npos) {
        size_t end = block.find('@', pos + 1);
        if (end == std::string::npos) break;
        std::string asset = block.substr(pos + 1, end - pos - 1);
        std::string lower = asset;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char ch) { return (char)std::tolower(ch); });
        if (lower.size() >= 4 &&
            lower.find(".ptx") != std::string::npos &&
            lower.find("._") == std::string::npos) {
            candidates.push_back(asset);
        }
        pos = end + 1;
    }

    auto is_color = [](const std::string& s) {
        std::string lower = s;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char ch) { return (char)std::tolower(ch); });
        return lower.find("/color/") != std::string::npos &&
               lower.find("/normal/") == std::string::npos &&
               lower.find("/displacement/") == std::string::npos;
    };
    for (const std::string& c : candidates)
        if (is_color(c)) return resolve_usda_layer_asset(layer_path, c);
    for (const std::string& c : candidates) {
        std::string lower = c;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char ch) { return (char)std::tolower(ch); });
        if (lower.find("normal") == std::string::npos &&
            lower.find("displacement") == std::string::npos)
            return resolve_usda_layer_asset(layer_path, c);
    }
    return std::string();
}

static void fill_material_from_usda_text(const std::string& block,
                                         const fs::path& layer_path,
                                         SceneMaterial& sm)
{
    init_pxr_material_defaults(&sm.params);

    float c3[3];
    bool have_c3 = usda_find_connected_color_correct_input(block, "diffuseColor", c3) ||
                   usda_find_connected_color_correct_input(block, "baseColor", c3);
    if (!have_c3) {
        have_c3 = usda_find_color_input(block, "diffuseColor", c3) ||
                  usda_find_color_input(block, "baseColor", c3);
        if (have_c3)
            apply_usda_moana_gamma(c3);
    }
    if (have_c3) {
        sm.params.base_color[0] = c3[0];
        sm.params.base_color[1] = c3[1];
        sm.params.base_color[2] = c3[2];
    }

    float transmit[3];
    bool have_transmit =
        usda_find_connected_color_correct_input(block, "diffuseTransmitColor", transmit);
    if (!have_transmit) {
        have_transmit = usda_find_color_input(block, "diffuseTransmitColor", transmit);
        if (have_transmit)
            apply_usda_moana_gamma(transmit);
    }
    if (have_transmit) {
        sm.params.transmission_color[0] = transmit[0];
        sm.params.transmission_color[1] = transmit[1];
        sm.params.transmission_color[2] = transmit[2];
    }

    float f = 0.0f;
    if (usda_find_float_input(block, "metallic", &f)) sm.params.metallic = f;
    if (usda_find_float_input(block, "roughness", &f)) sm.params.roughness = f;
    if (usda_find_float_input(block, "glassRoughness", &f)) sm.params.roughness = f;
    if (usda_find_float_input(block, "ior", &f)) sm.params.ior = f;
    if (usda_find_float_input(block, "glassIor", &f)) sm.params.ior = f;
    if (usda_find_float_input(block, "clearcoat", &f)) sm.params.clearcoat = f;
    if (usda_find_float_input(block, "clearcoatGloss", &f))
        sm.params.clearcoat_roughness = 1.0f - f;
    if (usda_find_float_input(block, "opacity", &f) ||
        usda_find_float_input(block, "alpha", &f)) {
        sm.params.opacity = f;
        sm.params.base_color[3] = f;
    }

    float diffuse_gain = 1.0f;
    float transmit_gain = 0.0f;
    float reflection_gain = 0.0f;
    float refraction_gain = 0.0f;
    float spec_trans = 0.0f;
    usda_find_float_input(block, "diffuseGain", &diffuse_gain);
    usda_find_float_input(block, "diffuseTransmitGain", &transmit_gain);
    usda_find_float_input(block, "reflectionGain", &reflection_gain);
    usda_find_float_input(block, "refractionGain", &refraction_gain);
    usda_find_float_input(block, "specularTransmission", &spec_trans);

    float trans_weight = std::max(std::max(refraction_gain, spec_trans),
                                  transmit_gain);
    if (trans_weight > 0.001f) {
        sm.params.transmission_weight = std::clamp(trans_weight, 0.0f, 1.0f);
        sm.params.transmission_ior = sm.params.ior;
        if (have_transmit) {
            sm.params.base_color[0] = transmit[0];
            sm.params.base_color[1] = transmit[1];
            sm.params.base_color[2] = transmit[2];
        }
        if (diffuse_gain <= 0.001f) {
            sm.params.opacity = std::min(sm.params.opacity, 0.35f);
            sm.params.base_color[3] = sm.params.opacity;
        }
        sm.params.clearcoat = std::max(sm.params.clearcoat,
                                       std::clamp(reflection_gain, 0.0f, 1.0f));
    }

    std::string ptex = usda_find_surface_ptx(block, layer_path);
    if (!ptex.empty())
        snprintf(sm.ptex_color_path, sizeof(sm.ptex_color_path), "%s", ptex.c_str());

    sm.params.metallic = std::clamp(sm.params.metallic, 0.0f, 1.0f);
    sm.params.roughness = std::clamp(sm.params.roughness, 0.0f, 1.0f);
    sm.params.opacity = std::clamp(sm.params.opacity, 0.0f, 1.0f);
    sm.params.base_color[3] = std::clamp(sm.params.base_color[3], 0.0f, 1.0f);
    sm.params.ior = std::clamp(sm.params.ior, 1.0f, 3.0f);
    sm.params.clearcoat = std::clamp(sm.params.clearcoat, 0.0f, 1.0f);
    sm.params.clearcoat_roughness =
        std::clamp(sm.params.clearcoat_roughness, 0.01f, 1.0f);
}

static int append_usda_sidecar_materials(
    NanousdStage stage,
    const char* scene_dir,
    std::vector<SceneMaterial>& materials,
    const SpvResult& default_vert,
    const SpvResult& default_frag)
{
    std::unordered_set<std::string> existing_paths;
    existing_paths.reserve(materials.size() * 2 + 1);
    for (const SceneMaterial& m : materials)
        if (m.prim_path[0]) existing_paths.insert(m.prim_path);

    std::vector<std::string> layer_paths =
        collect_usda_material_layer_paths(stage, scene_dir);
    std::unordered_set<std::string> wanted_paths;
    collect_missing_usda_material_targets(layer_paths, existing_paths,
                                          wanted_paths);
    if (wanted_paths.empty()) return 0;

    int appended = 0;
    for (const std::string& layer_path_str : layer_paths) {
        fs::path layer_path(layer_path_str);
        std::ifstream in(layer_path_str);
        if (!in) continue;

        std::vector<std::pair<std::string, int>> stack;
        std::string pending_type, pending_name;
        bool have_pending = false;
        bool capturing = false;
        int capture_depth = 0;
        std::string capture_path;
        std::string capture_block;
        int depth = 0;
        std::string line;

        while (std::getline(in, line)) {
            std::string typ, nam;
            if (parse_usda_prim_header(line, typ, nam)) {
                pending_type = typ;
                pending_name = nam;
                have_pending = true;
            }

            if (capturing) {
                capture_block += line;
                capture_block += '\n';
            }

            int opens = (int)std::count(line.begin(), line.end(), '{');
            int closes = (int)std::count(line.begin(), line.end(), '}');
            for (int oi = 0; oi < opens; ++oi) {
                depth++;
                if (have_pending) {
                    if (pending_type == "Material") {
                        std::string mat_path =
                            usda_path_from_stack(stack, pending_name);
                        if (wanted_paths.find(mat_path) != wanted_paths.end() &&
                            existing_paths.find(mat_path) == existing_paths.end()) {
                            capturing = true;
                            capture_depth = depth;
                            capture_path = mat_path;
                            capture_block.clear();
                            capture_block += line;
                            capture_block += '\n';
                        }
                    }
                    stack.push_back({pending_name, depth});
                    have_pending = false;
                }
            }

            for (int ci = 0; ci < closes; ++ci) {
                depth--;
                while (!stack.empty() && stack.back().second > depth)
                    stack.pop_back();
            }

            if (capturing && depth < capture_depth) {
                SceneMaterial sm = {};
                const char* slash = strrchr(capture_path.c_str(), '/');
                snprintf(sm.name, sizeof(sm.name), "%s", slash ? slash + 1 : capture_path.c_str());
                snprintf(sm.prim_path, sizeof(sm.prim_path), "%s", capture_path.c_str());
                fill_material_from_usda_text(capture_block, layer_path, sm);

                if (sm.ptex_color_path[0]) {
                    float avg[3];
                    if (nusd_ptex_sample_average_color(sm.ptex_color_path,
                                                       avg, 64)) {
                        sm.ptex_average_color[0] = avg[0];
                        sm.ptex_average_color[1] = avg[1];
                        sm.ptex_average_color[2] = avg[2];
                        sm.has_ptex_average_color = 1;
                        sm.params.base_color[0] = avg[0];
                        sm.params.base_color[1] = avg[1];
                        sm.params.base_color[2] = avg[2];
                        sm.params.use_vertex_color = 0;
                    }
                }
                if (!sm.has_ptex_average_color &&
                    material_color_is_placeholder_debug(sm.params.base_color)) {
                    sm.params.use_vertex_color = 1;
                }

                assign_default_material_shader(sm, default_vert, default_frag);
                materials.push_back(sm);
                existing_paths.insert(capture_path);
                appended++;
                capturing = false;
                capture_path.clear();
                capture_block.clear();
            }
        }
    }

    if (appended > 0) {
        fprintf(stderr,
                "material: side-loaded %d USDA layer material(s) from %zu "
                "bound target(s) missing in composed stage\n",
                appended, wanted_paths.size());
    }
    return appended;
}

static int material_binding_is_stronger_than_descendants(NanousdPrim prim,
                                                         const char* rel_name)
{
    int ok = 0;
    const char* strength =
        nanousd_rel_metadatas(prim, rel_name, "bindMaterialAs", &ok);
    return ok && strength &&
           strcmp(strength, "strongerThanDescendants") == 0;
}

static bool material_filter_renderable_prim(NanousdPrim prim)
{
    if (!prim || !nanousd_isactive(prim)) return false;
    const char* tn = nanousd_typename(prim);
    return tn && (!strcmp(tn, "Mesh") || !strcmp(tn, "Sphere") ||
                  !strcmp(tn, "Cylinder") || !strcmp(tn, "Capsule") ||
                  !strcmp(tn, "Cube") || !strcmp(tn, "Cone"));
}

static void material_filter_add_rel_targets(NanousdPrim prim,
                                            std::unordered_set<std::string>& targets)
{
    if (!prim) return;
    const char* rel_names[] = {"material:binding", "rel:material:binding"};
    for (const char* rel_name : rel_names) {
        int ntargets = nanousd_nreltargets(prim, rel_name);
        for (int i = 0; i < ntargets; ++i) {
            const char* target = nanousd_reltarget(prim, rel_name, i);
            if (target && target[0])
                targets.insert(target);
        }
    }
}

static void material_filter_collect_sibling_scope_materials(
    NanousdPrim mesh_prim,
    std::unordered_set<std::string>& targets)
{
    NanousdPrim cur = mesh_prim;
    while (cur) {
        NanousdPrim parent = nanousd_parent(cur);
        if (!parent) {
            if (cur != mesh_prim) nanousd_freeprim(cur);
            break;
        }
        int nch = nanousd_nchildren(parent);
        for (int c = 0; c < nch; ++c) {
            NanousdPrim sibling = nanousd_child(parent, c);
            if (!sibling) continue;
            const char* tn = nanousd_typename(sibling);
            if (!tn || strcmp(tn, "Scope") != 0) {
                nanousd_freeprim(sibling);
                continue;
            }
            int nmat = nanousd_nchildren(sibling);
            for (int m = 0; m < nmat; ++m) {
                NanousdPrim mat = nanousd_child(sibling, m);
                if (!mat) continue;
                const char* mt = nanousd_typename(mat);
                if (mt && strcmp(mt, "Material") == 0) {
                    const char* path = nanousd_path(mat);
                    if (path && path[0])
                        targets.insert(path);
                }
                nanousd_freeprim(mat);
            }
            nanousd_freeprim(sibling);
        }
        if (cur != mesh_prim) nanousd_freeprim(cur);
        cur = parent;
    }
}

static std::unordered_set<std::string> material_filter_collect_paths(
    NanousdStage stage,
    const unsigned char* wanted_prims,
    int nprims_in_bitmap)
{
    std::unordered_set<std::string> targets;
    if (!wanted_prims || nprims_in_bitmap <= 0)
        return targets;
    bool scan_sibling_scopes = false;
    if (const char* env = getenv("NUSD_FILTER_SIBLING_MATERIALS")) {
        scan_sibling_scopes = !(env[0] == '0' || env[0] == 'f' || env[0] == 'F' ||
                                env[0] == 'n' || env[0] == 'N');
    }
    std::unordered_set<std::string> visited_binding_prims;
    int nprims = nanousd_nprims(stage);
    int upto = nprims_in_bitmap < nprims ? nprims_in_bitmap : nprims;
    for (int i = 0; i < upto; ++i) {
        if (!wanted_prims[i]) continue;
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!material_filter_renderable_prim(prim)) {
            if (prim) nanousd_freeprim(prim);
            continue;
        }

        NanousdPrim cur = prim;
        while (cur) {
            const char* cur_path = nanousd_path(cur);
            if (cur_path && cur_path[0] &&
                !visited_binding_prims.insert(cur_path).second) {
                if (cur != prim) nanousd_freeprim(cur);
                break;
            }
            material_filter_add_rel_targets(cur, targets);
            NanousdPrim next = nanousd_parent(cur);
            if (cur != prim) nanousd_freeprim(cur);
            cur = next;
        }
        if (scan_sibling_scopes)
            material_filter_collect_sibling_scope_materials(prim, targets);
        nanousd_freeprim(prim);
    }
    return targets;
}

static int material_index_for_binding_target(MaterialCollection* mc,
                                             NanousdPrim binding_prim,
                                             const char* target)
{
    if (!mc || !binding_prim || !target || !target[0]) return -1;

    for (int i = 0; i < mc->nmaterials; i++) {
        if (strcmp(mc->materials[i].prim_path, target) == 0)
            return i;
    }

    const char* target_name = strrchr(target, '/');
    if (!target_name) return -1;
    target_name++;

    const char* mesh_path = nanousd_path(binding_prim);
    for (int i = 0; i < mc->nmaterials; i++) {
        if (strcmp(mc->materials[i].name, target_name) != 0)
            continue;
        if (mesh_path && strstr(mc->materials[i].prim_path, mesh_path) == NULL) {
            const char* mp = mc->materials[i].prim_path;
            const char* slash1 = strchr(mp + 1, '/');
            const char* slash2 = mesh_path ? strchr(mesh_path + 1, '/') : NULL;
            if (slash1 && slash2) {
                size_t len1 = (size_t)(slash1 - mp);
                size_t len2 = (size_t)(slash2 - mesh_path);
                if (len1 == len2 && memcmp(mp, mesh_path, len1) == 0)
                    return i;
            }
        } else {
            return i;
        }
    }

    for (int i = 0; i < mc->nmaterials; i++) {
        if (strcmp(mc->materials[i].name, target_name) == 0)
            return i;
    }
    return -1;
}

extern "C" MaterialCollection* materials_load_filtered(
    void* stage_handle,
    const char* scene_dir,
    const unsigned char* wanted_prims,
    int nprims_in_bitmap)
{
    NanousdStage stage = (NanousdStage)stage_handle;
    double _materials_load_t0 = nu_load_now_ms();  /* load-time instrumentation */
    g_resolve_cache.clear();  /* per-load resolution memo (bit-identical, never stale) */
    g_failed_texture_cache.clear();

    auto* mc = (MaterialCollection*)calloc(1, sizeof(MaterialCollection));
    if (!mc) return nullptr;

    int nprims = nanousd_nprims(stage);
    bool filter_materials = false;
    if (wanted_prims && nprims_in_bitmap > 0) {
        const char* env = getenv("NUSD_FILTER_VISIBLE_MATERIALS");
        filter_materials = env && env[0] &&
            !(env[0] == '0' || env[0] == 'f' || env[0] == 'F' ||
              env[0] == 'n' || env[0] == 'N');
    }
    std::unordered_set<std::string> wanted_material_paths;
    if (filter_materials) {
        double _mf_t0 = nu_load_now_ms();
        wanted_material_paths = material_filter_collect_paths(
            stage, wanted_prims, nprims_in_bitmap);
        fprintf(stderr,
                "material: visible filter collected %zu material target path(s) "
                "in %.1f ms\n",
                wanted_material_paths.size(), nu_load_now_ms() - _mf_t0);
        if (wanted_material_paths.empty())
            filter_materials = false;
    }

    // Collect all Material prims
    struct MatInfo {
        std::string path;
        std::string name;
        NanousdPrim   prim;
    };
    std::vector<MatInfo> mat_prims;
    size_t total_material_prims = 0;

    for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;

        if (nanousd_isactive(prim) && nanousd_isa(prim, "Material")) {
            total_material_prims++;
            MatInfo info;
            const char* path = nanousd_path(prim);
            const char* name = nanousd_name(prim);
            info.path = path ? path : "";
            info.name = name ? name : "unnamed";
            if (filter_materials && !wanted_material_paths.count(info.path)) {
                nanousd_freeprim(prim);
                continue;
            }
            info.prim = prim;
            mat_prims.push_back(info);
        } else {
            nanousd_freeprim(prim);
        }
    }

    if (filter_materials) {
        fprintf(stderr, "material: found %zu/%zu Material prims after visible filter\n",
                mat_prims.size(), total_material_prims);
    } else {
        fprintf(stderr, "material: found %zu Material prims\n", mat_prims.size());
    }

    /* Phase 7c gate (perf): MaterialX init (~125 ms) + default PBR shader
     * compile (~185 ms) are both unconditional setup costs. Skip them
     * entirely when there are no USD Material prims AND no .mtlx files
     * under scene_dir to side-load. Saves ~310 ms/load on curve-only or
     * mesh-without-materials scenes (e.g. showcase
     * grids). find_mtlx_files is a cheap recursive_directory_iterator
     * walk (max depth 6) and is called again inside load_mtlx_directory
     * later — the OS dir cache makes the second call effectively free.
     *
     * NUSD_MTLX_SIDELOAD=0 disables the side-loader entirely. Useful when
     * scene_dir is a system path (/tmp, /var) that contains unrelated
     * .mtlx files (e.g. a MaterialX upstream checkout under /tmp/mtlx-src),
     * which the recursive walk would otherwise pick up and load. */
    bool sideload_enabled = true;
    if (const char* env = getenv("NUSD_MTLX_SIDELOAD"))
        sideload_enabled = (env[0] != '0');
    std::vector<std::string> mtlx_side_files;
    if (sideload_enabled && scene_dir && *scene_dir) {
        mtlx_side_files = find_mtlx_files(scene_dir);
    }
    if (mat_prims.empty() && mtlx_side_files.empty()) {
        fprintf(stderr,
            "material: no Material prims, no .mtlx files — "
            "skipping MaterialX init + default-PBR shader compile\n");
        return mc;
    }

    /* MaterialX library access is needed both for the GenMsl codegen path
     * and for parsing the standalone .mtlx files we just discovered.
     * Initialize lazily — no-op if already done. */
    materialx_init();

    // Process each USD material prim independently. Large production scenes
    // often reuse leaf names under different element paths, and those prims can
    // carry different colors, roughness, Ptex maps, or water/transmission knobs.
    std::vector<SceneMaterial> materials;
    std::vector<MaterialTexture> all_textures;
    std::vector<std::string> all_tex_paths;

    // Map MaterialX side-loaded material name -> index of first occurrence.
    std::unordered_map<std::string, int> mat_name_to_idx;

    // Compile default PBR shaders once
    SpvResult default_vert = compile_glsl_to_spirv(
        k_default_vert_glsl, shaderc_vertex_shader, "default_pbr.vert");
    SpvResult default_frag = compile_glsl_to_spirv(
        k_default_pbr_frag_glsl, shaderc_fragment_shader, "default_pbr.frag");
    if (!default_vert.ok || !default_frag.ok) {
        fprintf(stderr, "material: default PBR shader compile failed!\n");
        if (!default_vert.ok) fprintf(stderr, "  vert: %s\n", default_vert.error.c_str());
        if (!default_frag.ok) fprintf(stderr, "  frag: %s\n", default_frag.error.c_str());
    }

    int unique_count = 0;
    /* ---- Parallel texture pre-decode -------------------------------------
     * PNG decode is ~9.8s of cold load and is embarrassingly parallel (pure
     * stbi, no USD/GetPrimIndex access). Scan materials + shader descendants
     * for image-asset inputs, resolve serially (the resolve memo isn't
     * thread-safe), then decode the unique set across a thread pool. The
     * readers below are unchanged — load_texture() consumes results from
     * g_predecoded and hits. Bit-identical (same stb decode of same bytes).
     * NUSD_PARALLEL_DECODE=0 disables. Paths a reader resolves via a different
     * anchor dir (or <UDIM>/package assets) simply miss and decode serially. */
    {
        const char* pdenv = getenv("NUSD_PARALLEL_DECODE");
        bool pd_on = !(pdenv && (pdenv[0]=='0'||pdenv[0]=='f'||pdenv[0]=='F'||
                                 pdenv[0]=='n'||pdenv[0]=='N'));
        for (auto& kv : g_predecoded)
            if (kv.second.pixels) stbi_image_free(kv.second.pixels);
        g_predecoded.clear();
        if (pd_on && !mat_prims.empty()) {
            double _pd_t0 = nu_load_now_ms();
            std::vector<std::string> todo;
            std::unordered_set<std::string> seen;
            std::vector<NanousdPrim> work, to_free;
            for (auto& mi : mat_prims) work.push_back(mi.prim);
            for (size_t wi = 0; wi < work.size(); ++wi) {
                NanousdPrim prim = work[wi];
                if (!prim) continue;
                int na = nanousd_nattribs(prim);
                for (int a = 0; a < na; ++a) {
                    const char* an = nanousd_attribname(prim, a);
                    if (!an) continue;
                    int ok = 0;
                    const char* av = nanousd_attribasset(prim, an, &ok);
                    if (ok && av && *av && !strstr(av, "<UDIM>") &&
                        has_image_extension(av)) {
                        std::string r = resolve_texture_path(av, scene_dir, stage);
                        if (!r.empty() && seen.insert(r).second)
                            todo.push_back(r);
                    }
                }
                int nc = nanousd_nchildren(prim);
                for (int c = 0; c < nc; ++c) {
                    NanousdPrim ch = nanousd_child(prim, c);
                    if (ch) { work.push_back(ch); to_free.push_back(ch); }
                }
            }
            for (NanousdPrim p : to_free) nanousd_freeprim(p);
            if (!todo.empty()) {
                std::vector<MaterialTexture> out(todo.size());
                unsigned hw = std::thread::hardware_concurrency();
                unsigned nthreads = hw ? (hw < 32u ? hw : 32u) : 4u;
                if ((size_t)nthreads > todo.size()) nthreads = (unsigned)todo.size();
                std::atomic<size_t> next{0};
                auto worker = [&]() {
                    for (;;) {
                        size_t i = next.fetch_add(1);
                        if (i >= todo.size()) break;
                        out[i] = decode_texture_file(todo[i]);
                    }
                };
                std::vector<std::thread> pool;
                for (unsigned t = 0; t < nthreads; ++t) pool.emplace_back(worker);
                for (auto& th : pool) th.join();
                int okc = 0;
                for (size_t i = 0; i < todo.size(); ++i)
                    if (out[i].pixels) { g_predecoded[todo[i]] = out[i]; ++okc; }
                g_tex_predecode_wall_ms = nu_load_now_ms() - _pd_t0;
                fprintf(stderr,
                        "material: PARALLEL-DECODE %d/%zu textures, %u threads, "
                        "%.1f ms wall\n",
                        okc, todo.size(), nthreads, g_tex_predecode_wall_ms);
            }
        }
    }

    for (auto& mi : mat_prims) {
        SceneMaterial sm = {};
        snprintf(sm.name, sizeof(sm.name), "%s", mi.name.c_str());
        snprintf(sm.prim_path, sizeof(sm.prim_path), "%s", mi.path.c_str());
        read_material_ptex_color_path(mi.prim, scene_dir, stage, sm.ptex_color_path);

        {
            // Read properties for this exact material prim/path.
            NuReaderTimer _rt;
            NanousdPrim shader = find_surface_shader(stage, mi.prim);
            ShaderType stype = detect_shader_type(shader);

            switch (stype) {
            case SHADER_PXR_DISNEY_BSDF:
                read_pxr_disney_material(stage, mi.prim, shader, &sm.params);
                break;
            case SHADER_PXR_SURFACE:
                read_pxr_surface_material(stage, mi.prim, shader, &sm.params);
                break;
            case SHADER_PXR_VOLUME:
                read_pxr_surface_material(stage, mi.prim, shader, &sm.params);
                sm.params.opacity = std::min(sm.params.opacity, 0.35f);
                sm.params.base_color[3] = sm.params.opacity;
                break;
            case SHADER_MATERIALX_STANDARD_SURFACE:
                fprintf(stderr, "material: '%s' -> MaterialX Standard Surface reader\n",
                        sm.name);
                read_materialx_usd_surface(stage, mi.prim, shader, &sm.params,
                                           all_tex_paths, all_textures,
                                           scene_dir, false);
                break;
            case SHADER_MATERIALX_OPEN_PBR:
                fprintf(stderr, "material: '%s' -> MaterialX OpenPBR reader\n",
                        sm.name);
                read_materialx_usd_surface(stage, mi.prim, shader, &sm.params,
                                           all_tex_paths, all_textures,
                                           scene_dir, true);
                break;
            case SHADER_OMNIPBR:
            case SHADER_OMNISURFACE:
            case SHADER_MDL_GENERIC:
                fprintf(stderr, "material: '%s' → OmniPBR/MDL reader\n", sm.name);
                read_omnipbr_material(shader, &sm.params,
                                      all_tex_paths, all_textures, scene_dir, stage);
                break;
            case SHADER_OMNIGLASS:
                fprintf(stderr, "material: '%s' → OmniGlass reader\n", sm.name);
                read_omniglass_material(shader, &sm.params,
                                        all_tex_paths, all_textures, scene_dir, stage);
                break;
            case SHADER_USD_PREVIEW_SURFACE:
            default:
                read_usd_preview_surface(shader, mi.prim, &sm.params,
                                         all_tex_paths, all_textures, scene_dir, stage);
                break;
            }

            // If no textures and no explicit color, use vertex color
            bool has_any_tex = false;
            for (int t = 0; t < MAX_MATERIAL_TEXTURES; t++) {
                if (sm.params.tex_indices[t] >= 0) {
                    has_any_tex = true;
                    break;
                }
            }
            if (!has_any_tex) {
                float dc = sm.params.base_color[0] + sm.params.base_color[1] +
                           sm.params.base_color[2];
                if (dc < 0.001f) {
                    sm.params.use_vertex_color = 1;
                }
            }

            if (shader) nanousd_freeprim(shader);
            unique_count++;
        }

        if (sm.ptex_color_path[0]) {
            float avg[3];
            if (nusd_ptex_sample_average_color(sm.ptex_color_path, avg, 64)) {
                sm.ptex_average_color[0] = avg[0];
                sm.ptex_average_color[1] = avg[1];
                sm.ptex_average_color[2] = avg[2];
                sm.has_ptex_average_color = 1;
                sm.params.base_color[0] = avg[0];
                sm.params.base_color[1] = avg[1];
                sm.params.base_color[2] = avg[2];
                sm.params.use_vertex_color = 0;
            }
        }
        if (!sm.has_ptex_average_color &&
            material_color_is_placeholder_debug(sm.params.base_color)) {
            sm.params.use_vertex_color = 1;
        }

        // Use pre-compiled default PBR shaders
        assign_default_material_shader(sm, default_vert, default_frag);

        materials.push_back(sm);
        nanousd_freeprim(mi.prim);
    }

    append_usda_sidecar_materials(stage, scene_dir, materials,
                                  default_vert, default_frag);

    /* Free pre-decoded textures the readers did not consume (inputs a specific
     * reader skipped, or anchor-dir resolve mismatches that decoded serially
     * in the walk instead). Consumed entries were erased from g_predecoded and
     * their pixels now live in all_textures. */
    for (auto& kv : g_predecoded)
        if (kv.second.pixels) stbi_image_free(kv.second.pixels);
    g_predecoded.clear();

    free(default_vert.data);
    free(default_frag.data);

    fprintf(stderr, "material: read %d material parameter blocks (from %zu total prims)\n",
            unique_count, mat_prims.size());

    /* Phase 7c — fold in any standalone MaterialX .mtlx files under
     * the scene directory. Useful for assets like OpenChessSet whose USD
     * `references = @file.mtlx@` aren't resolvable without a usdMtlx
     * file-format plugin. Adds materials with synthesised
     * /Materials/<name> prim paths; the leaf-name fallback in
     * materials_find_binding resolves bindings from real USD prims.
     *
     * Honor NUSD_MTLX_SIDELOAD=0 (see gate above). */
    if (sideload_enabled && scene_dir && *scene_dir) {
        load_mtlx_directory(scene_dir, materials, all_tex_paths,
                            all_textures, mat_name_to_idx, stage);
    }

    // Build the collection
    mc->nmaterials = (int)materials.size();
    mc->materials = (SceneMaterial*)calloc(mc->nmaterials, sizeof(SceneMaterial));
    memcpy(mc->materials, materials.data(),
           mc->nmaterials * sizeof(SceneMaterial));

    mc->ntextures = (int)all_textures.size();
    if (mc->ntextures > 0) {
        mc->textures = (MaterialTexture*)calloc(mc->ntextures,
                                                sizeof(MaterialTexture));
        memcpy(mc->textures, all_textures.data(),
               mc->ntextures * sizeof(MaterialTexture));

        /* Decide each texture's color space by looking at every slot that
         * references it across all materials.  Color slots
         * (DIFFUSE, EMISSIVE) want sRGB; data slots (NORMAL, ROUGHNESS,
         * METALLIC, OCCLUSION) want linear.  Mixed-use textures default
         * to linear — sampling sRGB-encoded data as linear renders too
         * dark, but sampling linear-encoded data as sRGB corrupts
         * normals + PBR parameters, which is much worse.
         *
         * Default: sRGB.  Downgrade to linear if any data-slot use is
         * found (or if the only references are data-slot). */
        const bool color_slot[MAX_MATERIAL_TEXTURES] = {
            /* TEX_DIFFUSE_COLOR  = 0 */ true,
            /* TEX_NORMAL         = 1 */ false,
            /* TEX_ROUGHNESS      = 2 */ false,
            /* TEX_METALLIC       = 3 */ false,
            /* TEX_EMISSIVE_COLOR = 4 */ true,
            /* TEX_OCCLUSION      = 5 */ false,
            /* TEX_OPACITY        = 6 */ false,  /* separate opacity maps are scalar data */
            /* slot 7 reserved          */ true,
        };

        /* For each texture: srgb_votes / data_votes from slot usage. */
        for (int t = 0; t < mc->ntextures; ++t) {
            int srgb_votes = 0, data_votes = 0;
            for (int m = 0; m < mc->nmaterials; ++m) {
                const int* tex_indices = mc->materials[m].params.tex_indices;
                for (int s = 0; s < MAX_MATERIAL_TEXTURES; ++s) {
                    if (tex_indices[s] != t) continue;
                    if (color_slot[s]) ++srgb_votes;
                    else               ++data_votes;
                }
            }
            /* Linear wins on tie or if any data-slot reference exists. */
            mc->textures[t].is_srgb = (data_votes == 0 && srgb_votes > 0) ? 1 : 0;
        }
    }

    build_binding_hints_from_usda_layers(mc, stage);

    fprintf(stderr, "material: %d materials, %d textures loaded\n",
            mc->nmaterials, mc->ntextures);

    /* Material-object dedup: collapse byte-identical materials to one canonical
     * index. MaterialParams is POD and the array is calloc'd (padding zeroed),
     * so a raw-byte hash is exact and never merges genuinely different
     * materials. Omniverse authors one MaterialInstanceDynamic per prim, so
     * thousands of materials are byte-identical; sharing the index lets the
     * geometry compaction (B6 guard) fuse same-geometry/same-material instances
     * that currently bind distinct-but-identical materials. */
    if (mc->nmaterials > 1) {
        mc->dedup_remap = (int*)malloc(sizeof(int) * (size_t)mc->nmaterials);
        if (mc->dedup_remap) {
            std::unordered_map<uint64_t, std::vector<int>> buckets;
            buckets.reserve((size_t)mc->nmaterials);
            int unique = 0;
            for (int i = 0; i < mc->nmaterials; i++) {
                const SceneMaterial* mi = &mc->materials[i];
                const MaterialParams* p = &mi->params;
                uint64_t h = 1469598103934665603ULL;
                const unsigned char* b = (const unsigned char*)p;
                for (size_t k = 0; k < sizeof(MaterialParams); k++) {
                    h ^= b[k]; h *= 1099511628211ULL;
                }
                int canon = -1;
                std::vector<int>& cand = buckets[h];
                for (size_t c = 0; c < cand.size(); c++) {
                    const SceneMaterial* mj = &mc->materials[cand[c]];
                    /* params is the coarse hash bucket, but SceneMaterial also
                     * carries render-affecting Ptex fields NOT in params. Ptex
                     * materials frequently share default params and differ ONLY
                     * in ptex_color_path, so the exact-equality test must widen
                     * to them or distinct Ptex textures would over-merge. */
                    if (memcmp(&mj->params, p, sizeof(MaterialParams)) == 0 &&
                        strcmp(mj->ptex_color_path, mi->ptex_color_path) == 0 &&
                        mj->has_ptex_average_color == mi->has_ptex_average_color &&
                        memcmp(mj->ptex_average_color, mi->ptex_average_color,
                               sizeof(mi->ptex_average_color)) == 0) {
                        canon = cand[c]; break;
                    }
                }
                if (canon < 0) { canon = i; cand.push_back(i); unique++; }
                mc->dedup_remap[i] = canon;
            }
            fprintf(stderr,
                    "material: object-dedup — %d materials -> %d unique "
                    "(shared index unblocks geometry compaction)\n",
                    mc->nmaterials, unique);
        }
    }
    {
        double _total = nu_load_now_ms() - _materials_load_t0;
        fprintf(stderr,
                "material: LOAD-SPLIT materials_load_total=%.1f ms | "
                "predecode=%.1f ms wall (%ld hits) | decode=%.1f ms (%ld imgs) | "
                "resolve=%.1f ms (%ld miss + %ld memo-hit) | "
                "other_material_cpu=%.1f ms (reader[reads+bakes]=%.1f ms / %ld mats; "
                "bake=%.1f ms / %ld bakes; reads=%.1f ms) "
                "[+ GPU upload + first render are outside materials_load]\n",
                _total, g_tex_predecode_wall_ms, g_tex_predecode_n,
                g_tex_decode_ms, g_tex_decode_n,
                g_tex_resolve_ms, g_tex_resolve_n, g_tex_resolve_hits,
                _total - g_tex_predecode_wall_ms - g_tex_decode_ms - g_tex_resolve_ms,
                g_reader_ms, g_reader_n,
                g_bake_ms, g_bake_n, g_reader_ms - g_bake_ms);
    }

    if (getenv("NUSD_MAT_DUMP")) {
        const char* dump_filter = getenv("NUSD_MAT_DUMP_FILTER");
        auto dump_matches = [dump_filter](const SceneMaterial& m) -> bool {
            if (!dump_filter || !dump_filter[0]) return true;
            const char* p = dump_filter;
            while (*p) {
                while (*p == ',' || *p == ';' || *p == ' ') ++p;
                const char* start = p;
                while (*p && *p != ',' && *p != ';') ++p;
                const char* end = p;
                while (end > start && end[-1] == ' ') --end;
                if (end > start) {
                    std::string term(start, (size_t)(end - start));
                    if (strstr(m.name, term.c_str()) ||
                        strstr(m.prim_path, term.c_str()) ||
                        strstr(m.ptex_color_path, term.c_str()))
                        return true;
                }
            }
            return false;
        };
        int no_diffuse = 0;
        int dumped = 0;
        for (int i = 0; i < mc->nmaterials; i++) {
            SceneMaterial& m = mc->materials[i];
            if (!dump_matches(m)) {
                if (m.params.tex_indices[0] < 0) no_diffuse++;
                continue;
            }
            fprintf(stderr,
                    "  mat[%d] '%s' path=%s ptex=%s diff=%d norm=%d "
                    "roughTex=%d metalTex=%d emi=%d occ=%d opaTex=%d "
                    "baseC=(%.3f,%.3f,%.3f,%.3f) rough=%.3f metal=%.3f "
                    "opacity=%.3f cut=%.3f trans=%.3f transC=(%.3f,%.3f,%.3f) "
                    "ior=%.3f clearcoat=%.3f useVC=%d ptexAvg=%d\n",
                    i, m.name, m.prim_path,
                    m.ptex_color_path[0] ? m.ptex_color_path : "<none>",
                    m.params.tex_indices[0], m.params.tex_indices[1],
                    m.params.tex_indices[2], m.params.tex_indices[3],
                    m.params.tex_indices[4], m.params.tex_indices[5],
                    m.params.tex_indices[6],
                    m.params.base_color[0], m.params.base_color[1],
                    m.params.base_color[2], m.params.base_color[3],
                    m.params.roughness, m.params.metallic,
                    m.params.opacity, m.params.opacity_threshold,
                    m.params.transmission_weight,
                    m.params.transmission_color[0],
                    m.params.transmission_color[1],
                    m.params.transmission_color[2],
                    m.params.ior, m.params.clearcoat,
                    m.params.use_vertex_color,
                    m.has_ptex_average_color);
            dumped++;
            if (m.params.tex_indices[0] < 0) no_diffuse++;
        }
        fprintf(stderr,
                "material: %d / %d materials have NO diffuse texture; dumped %d%s%s\n",
                no_diffuse, mc->nmaterials, dumped,
                (dump_filter && dump_filter[0]) ? " matching " : "",
                (dump_filter && dump_filter[0]) ? dump_filter : "");
    }

    return mc;
}

extern "C" MaterialCollection* materials_load(void* stage_handle,
                                              const char* scene_dir)
{
    return materials_load_filtered(stage_handle, scene_dir, NULL, 0);
}

static int materials_find_binding_impl(MaterialCollection* mc,
                                       void* stage_handle,
                                       void* mesh_prim_handle)
{
    NanousdPrim mesh_prim = (NanousdPrim)mesh_prim_handle;
    NanousdStage stage = (NanousdStage)stage_handle;
    (void)stage;
    if (!mc || !mesh_prim || mc->nmaterials == 0) return -1;

    // 1. Walk up the prim hierarchy looking for material:binding.
    // A normal child binding wins by namespace depth, but an ancestor
    // relationship tagged bindMaterialAs="strongerThanDescendants" must
    // override descendant placeholder bindings.
    int closest_idx = -1;
    NanousdPrim cur = mesh_prim;
    while (cur) {
        int ntargets = nanousd_nreltargets(cur, "material:binding");
        const char* rel_name = "material:binding";
        if (ntargets <= 0) {
            ntargets = nanousd_nreltargets(cur, "rel:material:binding");
            rel_name = "rel:material:binding";
        }
        if (ntargets > 0) {
            const char* target = nanousd_reltarget(cur, rel_name, 0);
            if (target) {
                int idx = material_index_for_binding_target(mc, cur, target);
                if (idx >= 0) {
                    if (material_binding_is_stronger_than_descendants(cur, rel_name)) {
                        if (cur != mesh_prim) nanousd_freeprim(cur);
                        return idx;
                    }
                    if (closest_idx < 0)
                        closest_idx = idx;
                }
            }
        }
        NanousdPrim next = nanousd_parent(cur);
        if (cur != mesh_prim) nanousd_freeprim(cur);
        cur = next;
    }
    if (closest_idx >= 0)
        return closest_idx;

    // 2. Determine variant selection for material preference.
    char variant_sel_buf[64] = {0};
    const char* variant_sel = NULL;
    cur = mesh_prim;
    while (cur) {
        const char* sel = nanousd_variantselection(cur, "shadingVariant");
        if (sel && sel[0]) {
            snprintf(variant_sel_buf, sizeof(variant_sel_buf), "%s", sel);
            variant_sel = variant_sel_buf;
            if (cur != mesh_prim) nanousd_freeprim(cur);
            break;
        }
        NanousdPrim next = nanousd_parent(cur);
        if (cur != mesh_prim) nanousd_freeprim(cur);
        cur = next;
    }

    // 3. Fallback: walk up and find a sibling "Materials" scope.
    // Use variant selection to pick the correct material when multiple exist.
    cur = mesh_prim;
    while (cur) {
        NanousdPrim parent = nanousd_parent(cur);
        if (!parent) {
            if (cur != mesh_prim) nanousd_freeprim(cur);
            break;
        }
        int nch = nanousd_nchildren(parent);
        int found_idx = -1;
        for (int c = 0; c < nch && found_idx < 0; c++) {
            NanousdPrim sibling = nanousd_child(parent, c);
            if (!sibling) continue;
            const char* tn = nanousd_typename(sibling);
            if (!tn || strcmp(tn, "Scope") != 0) {
                nanousd_freeprim(sibling);
                continue;
            }
            int nmat = nanousd_nchildren(sibling);
            int first_match = -1;
            for (int m = 0; m < nmat; m++) {
                NanousdPrim mat_prim = nanousd_child(sibling, m);
                if (!mat_prim) continue;
                const char* mt = nanousd_typename(mat_prim);
                if (!mt || strcmp(mt, "Material") != 0) {
                    nanousd_freeprim(mat_prim);
                    continue;
                }
                const char* mat_path = nanousd_path(mat_prim);
                if (!mat_path) {
                    nanousd_freeprim(mat_prim);
                    continue;
                }

                int mat_idx = -1;
                for (int i = 0; i < mc->nmaterials; i++) {
                    if (strcmp(mc->materials[i].prim_path, mat_path) == 0) {
                        mat_idx = i;
                        break;
                    }
                }
                if (mat_idx < 0) {
                    nanousd_freeprim(mat_prim);
                    continue;
                }

                if (first_match < 0) first_match = mat_idx;

                // If we have a variant selection, prefer matching material
                if (variant_sel) {
                    const char* mn = nanousd_name(mat_prim);
                    if (mn) {
                        const char* suffix = (strcmp(variant_sel, "Black") == 0) ? "_B" :
                                             (strcmp(variant_sel, "White") == 0) ? "_W" : NULL;
                        if (suffix) {
                            size_t mn_len = strlen(mn);
                            size_t sf_len = strlen(suffix);
                            if (mn_len >= sf_len &&
                                strcmp(mn + mn_len - sf_len, suffix) == 0) {
                                found_idx = mat_idx;
                                nanousd_freeprim(mat_prim);
                                break;
                            }
                        }
                    }
                }
                nanousd_freeprim(mat_prim);
            }
            nanousd_freeprim(sibling);
            if (found_idx < 0 && first_match >= 0) found_idx = first_match;
        }
        if (found_idx >= 0) {
            if (cur != mesh_prim) nanousd_freeprim(cur);
            nanousd_freeprim(parent);
            return found_idx;
        }
        if (cur != mesh_prim) nanousd_freeprim(cur);
        cur = parent;
    }

    return -1;
}

/* Canonicalize a raw material index through the dedup remap. */
static inline int materials_canonical_index(MaterialCollection* mc, int idx)
{
    return (idx >= 0 && mc && mc->dedup_remap) ? mc->dedup_remap[idx] : idx;
}

/* Public entry: resolve the binding, then return the CANONICAL (deduped) index
 * so byte-identical per-prim materials share one index — which lets the geometry
 * compaction (B6 guard) fuse same-geometry instances that bind distinct-but-
 * identical materials. */
extern "C" int materials_find_binding(MaterialCollection* mc,
                                      void* stage_handle,
                                      void* mesh_prim_handle)
{
    return materials_canonical_index(
        mc, materials_find_binding_impl(mc, stage_handle, mesh_prim_handle));
}

extern "C" int materials_find_binding_by_path(MaterialCollection* mc,
                                              const char* mesh_path)
{
    if (!mc || !mesh_path || !*mesh_path || !mc->binding_hints ||
        mc->nbinding_hints <= 0) {
        return -1;
    }

    const char* leaf = strrchr(mesh_path, '/');
    leaf = leaf ? leaf + 1 : mesh_path;
    char key[256];
    size_t n = strcspn(leaf, "#");
    if (n >= sizeof(key)) n = sizeof(key) - 1;
    memcpy(key, leaf, n);
    key[n] = '\0';
    if (!key[0]) return -1;

    int lo = 0;
    int hi = mc->nbinding_hints;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        int c = strcmp(mc->binding_hints[mid].prim_name, key);
        if (c < 0) lo = mid + 1;
        else       hi = mid;
    }
    if (lo < mc->nbinding_hints &&
        strcmp(mc->binding_hints[lo].prim_name, key) == 0) {
        return mc->binding_hints[lo].material_index;
    }
    return -1;
}

extern "C" void materials_free(MaterialCollection* mc)
{
    if (!mc) return;

    // Free generated shaders
    for (int i = 0; i < mc->nmaterials; i++) {
        free(mc->materials[i].shader.vert_spv);
        free(mc->materials[i].shader.frag_spv);
    }
    free(mc->materials);

    // Free textures
    for (int i = 0; i < mc->ntextures; i++) {
        if (mc->textures[i].pixels)
            stbi_image_free(mc->textures[i].pixels);
    }
    free(mc->textures);

    free(mc->unique_shaders);
    free(mc->binding_hints);
    free(mc->dedup_remap);
    free(mc);
}

/* ---- Environment background SPIR-V compilation ---- */

extern "C" int env_bg_compile_shaders(uint32_t** vert_spv, uint32_t* vert_size,
                                       uint32_t** frag_spv, uint32_t* frag_size)
{
    SpvResult vert = compile_glsl_to_spirv(k_env_bg_vert_glsl,
                                           shaderc_vertex_shader, "env_bg.vert");
    SpvResult frag = compile_glsl_to_spirv(k_env_bg_frag_glsl,
                                           shaderc_fragment_shader, "env_bg.frag");
    if (!vert.ok || !frag.ok) {
        fprintf(stderr, "env_bg: shader compile failed\n");
        if (!vert.ok) fprintf(stderr, "  vert: %s\n", vert.error.c_str());
        if (!frag.ok) fprintf(stderr, "  frag: %s\n", frag.error.c_str());
        free(vert.data);
        free(frag.data);
        return 0;
    }
    *vert_spv = vert.data;   *vert_size = vert.size;
    *frag_spv = frag.data;   *frag_size = frag.size;
    return 1;
}
