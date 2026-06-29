// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * material.cpp — MaterialX-based material system for the nanousd-metal-renderer.
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
#include <cstdint>

#include "material.h"
#include "mdl_bridge.h"

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Library.h>
#include <MaterialXFormat/Environ.h>
#include <MaterialXFormat/Util.h>
/* Phase 7b: Metal port — swap GLSL/Vulkan codegen for MSL. The two
 * generators expose identical interfaces (HwShaderGenerator subclass);
 * only the include + create() target change. */
#include <MaterialXGenMsl/MslShaderGenerator.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/UnitSystem.h>
#include <MaterialXCore/Unit.h>

/* shaderc removed — Phase 7b stashes MSL source bytes instead of
 * SPIR-V output. The Metal backend's gpu_create_material_pipeline
 * compiles the MSL source via [device newLibraryWithSource:]. */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <initializer_list>
#include <cstdint>

namespace mx = MaterialX;
namespace fs = std::filesystem;

/* ============================================================================
 * Load-time instrumentation + CPU parallelism
 * Ported from nanousd-vulkan-renderer/src/material.cpp (83fee92, 3918b3b,
 * 169b54c). Metal uses std::thread (NOT OpenMP) for the bake + decode
 * parallelism: AppleClang ships no libomp by default, find_package(OpenMP)
 * would need a brew install + CMake hints and would silently no-op the pragmas
 * if libomp is missing, and the decode pre-pass already uses std::thread — one
 * primitive keeps material.cpp internally consistent. Enable timing with
 * NUSD_LOAD_TIMING=1. ========================================================*/
#include <thread>
#include <atomic>
#include <time.h>

static inline double nu_load_now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* Dynamic parallel-for over [0, n): worker threads pull item indices off a
 * shared atomic counter and call fn(i). Dynamic scheduling balances both
 * variable-cost items (texture decode, where image sizes differ) and uniform
 * ones (bake rows) from one primitive. Serial fallback for small n so tiny
 * bakes/decodes don't pay thread-spawn cost. All threads join before return,
 * so fn (and anything it captures by reference) outlives every worker; fn(i)
 * must be safe to call concurrently for distinct i. Up to min(hw, 32) threads. */
template <class Fn>
static void nu_parallel_for(int n, Fn fn) {
    if (n <= 0) return;
    /* NUSD_PARALLEL_OFF=1 forces the serial path — lets the same binary be A/B
     * measured (parallel decode + bake wins) against an otherwise-identical run. */
    static const bool g_par_off = [](){ const char* e = getenv("NUSD_PARALLEL_OFF");
                                        return e && e[0] && e[0] != '0'; }();
    if (g_par_off || n < 64) { for (int i = 0; i < n; ++i) fn(i); return; }
    unsigned hw = std::thread::hardware_concurrency();
    unsigned nt = hw ? (hw < 32u ? hw : 32u) : 4u;
    if ((int)nt > n) nt = (unsigned)n;
    std::atomic<int> next{0};
    auto worker = [&]() {
        for (;;) {
            int i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;
            fn(i);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (unsigned t = 0; t < nt; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
}

/* Load-time counters (printed when NUSD_LOAD_TIMING=1). Bit-identical: the
 * timers only measure; they never change material output. */
static double g_tex_decode_ms         = 0.0;  static long g_tex_decode_n       = 0;
static double g_tex_resolve_ms        = 0.0;  static long g_tex_resolve_n      = 0;
static long   g_tex_resolve_hits      = 0;
static double g_reader_ms             = 0.0;  static long g_reader_n           = 0;
static double g_bake_ms               = 0.0;  static long g_bake_n             = 0;
static double g_tex_predecode_wall_ms = 0.0;  static long g_tex_predecode_n    = 0;
struct NuResolveTimer { double t0; NuResolveTimer():t0(nu_load_now_ms()){}
    ~NuResolveTimer(){ g_tex_resolve_ms += nu_load_now_ms()-t0; ++g_tex_resolve_n; } };
struct NuReaderTimer  { double t0; NuReaderTimer():t0(nu_load_now_ms()){}
    ~NuReaderTimer(){ g_reader_ms += nu_load_now_ms()-t0; ++g_reader_n; } };
struct NuBakeTimer    { double t0; NuBakeTimer():t0(nu_load_now_ms()){}
    ~NuBakeTimer(){ g_bake_ms += nu_load_now_ms()-t0; ++g_bake_n; } };

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
    int   _pad[1];
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

/* Lat-long environment map lookup (Vulkan top-left origin) */
vec2 envMapUV(vec3 dir) {
    float phi = atan(dir.z, dir.x);
    float theta = asin(clamp(dir.y, -1.0, 1.0));
    return vec2(phi / (2.0 * PI) + 0.5, 0.5 - theta / PI);
}

void main() {
    MaterialData mat = mat_buf.materials[fragMaterialId];
    vec3 N = normalize(fragNormal);

    /* UDIM UV scaling: divide UVs by atlas grid dimensions */
    vec2 tc = fragTexCoord / max(vec2(mat.udim_scale_u, mat.udim_scale_v), vec2(1.0));

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
        roughness = texture(textures[nonuniformEXT(mat.tex_indices[2])], tc).g;
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
        float lightScale = (hasIBL != 0) ? 0.5 : 1.0;

        vec3 L = lightDir;
        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);

        /* Cook-Torrance BRDF */
        float D = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        vec3  F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 numerator = D * G * F;
        float denominator = 4.0 * NdotV * NdotL + 0.0001;
        vec3 specular = numerator / denominator;

        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        Lo = (kD * baseColor / PI + specular) * lightColor * NdotL * lightScale;

        /* Fill light from opposite side */
        vec3 fillDir = normalize(vec3(-0.5, 0.4, -0.3));
        float fillNdotL = max(dot(N, fillDir), 0.0);
        float fillScale = (hasIBL != 0) ? 0.2 : 0.15;
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

    /* Equirectangular lookup (matching RT miss shader convention) */
    float u = atan(dir.z, dir.x) * (0.5 / PI) + 0.5;
    float v = asin(clamp(dir.y, -1.0, 1.0)) * (1.0 / PI) + 0.5;
    vec2 uv = vec2(u, 1.0 - v);  /* flip V for top-down convention */

    vec3 color = texture(envMap, uv).rgb;
    /* ACES tone map — sRGB swapchain handles gamma */
    color = clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
    outColor = vec4(color, 1.0);
}
)";

/* ---- Phase 7b: stash MSL source bytes (no compile here) ----
 *
 * The Metal backend ports the Vulkan shaderc → SPIR-V bytes path to
 * "MSL source string copied into a malloc'd byte buffer." MaterialShader's
 * vert_spv / frag_spv fields are reinterpreted as MSL source bytes
 * with the same `data + size` semantics — gpu_create_material_pipeline
 * (in gpu_metal.mm) reads them as UTF-8 source and compiles via
 * [device newLibraryWithSource:options:error:]. This keeps the C struct
 * layout identical to the Vulkan port so other clients (renderer.c,
 * nu_renderer.h) don't need any change.
 *
 * The "kind" and "name" args from the old shaderc API are unused here;
 * we keep parity in callers via a compatible enum + a no-op signature
 * to minimize the diff size in callers below. */

struct SpvResult {
    uint32_t* data;
    uint32_t  size;
    bool      ok;
    std::string error;
};

enum shaderc_shader_kind {  // legacy enum kept so call sites compile unchanged
    shaderc_vertex_shader,
    shaderc_fragment_shader,
};

/* Copy an MSL source string into a heap-allocated byte buffer with
 * data+size semantics matching the old SPIR-V output. Caller frees data. */
static SpvResult compile_glsl_to_spirv(const char* source,
                                       shaderc_shader_kind /*kind*/,
                                       const char* /*name*/)
{
    SpvResult result = {};
    if (!source) { result.error = "null MSL source"; return result; }
    size_t len = strlen(source);
    if (len == 0) { result.error = "empty MSL source"; return result; }
    /* +4 for trailing null + 4-byte alignment headroom; the consumer
     * pads as needed when calling [device newLibraryWithSource:]. */
    size_t bytes = len + 4;
    result.data = (uint32_t*)malloc(bytes);
    if (!result.data) { result.error = "OOM"; return result; }
    memcpy(result.data, source, len);
    ((char*)result.data)[len] = '\0';
    result.size = (uint32_t)len;
    result.ok = true;
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
 * Keyed by search root so multiple scenes don't poison each other. */
static std::unordered_map<std::string,
    std::unordered_map<std::string, std::string>> g_filename_index_cache;

static const std::unordered_map<std::string, std::string>&
get_filename_index(const std::string& search_root)
{
    auto it = g_filename_index_cache.find(search_root);
    if (it != g_filename_index_cache.end()) return it->second;

    std::unordered_map<std::string, std::string> index;
    try {
        fs::path root(search_root);
        const char* home = getenv("HOME");
        if (root == fs::path("/") ||
            root == fs::path("/Users") ||
            root == fs::path("/home") ||
            (home && home[0] && root == fs::path(home))) {
            auto& ref = g_filename_index_cache[search_root];
            ref = std::move(index);
            return ref;
        }
        if (fs::exists(root) && fs::is_directory(root)) {
            double _idx_t0 = nu_load_now_ms();
            long _idx_n = 0;
            for (auto it2 = fs::recursive_directory_iterator(root,
                     fs::directory_options::skip_permission_denied);
                 it2 != fs::recursive_directory_iterator(); ++it2) {
                if (it2.depth() > 8) { it2.disable_recursion_pending(); continue; }
                if (!it2->is_regular_file()) continue;
                std::string fn = it2->path().filename().string();
                /* Keep first hit — caller can always pass a more specific path. */
                index.emplace(std::move(fn), it2->path().string());
                ++_idx_n;
            }
            if (getenv("NUSD_LOAD_TIMING"))
                fprintf(stderr, "material: PROFILE filename-index walk %.0f ms, "
                        "%ld files, root=%s\n",
                        nu_load_now_ms() - _idx_t0, _idx_n, search_root.c_str());
        }
    } catch (...) {}
    auto& ref = g_filename_index_cache[search_root];
    ref = std::move(index);
    return ref;
}

static void push_unique_path(std::vector<fs::path>& paths, const fs::path& path)
{
    if (path.empty()) return;
    fs::path normalized = path;
    try { normalized = fs::weakly_canonical(path); } catch (...) {}
    for (const auto& existing : paths) {
        if (existing == normalized) return;
    }
    try {
        if (fs::exists(normalized) && fs::is_directory(normalized))
            paths.push_back(std::move(normalized));
    } catch (...) {}
}

static void collect_env_asset_search_roots(std::vector<fs::path>& roots)
{
    const char* env = getenv("NUSD_ASSET_ROOTS");
    if (!env || !env[0]) return;

    std::string s(env);
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find(':', pos);
        std::string item = s.substr(pos, end == std::string::npos
                                           ? std::string::npos
                                           : end - pos);
        if (!item.empty()) push_unique_path(roots, fs::path(item));
        if (end == std::string::npos) break;
        pos = end + 1;
    }
}

static void collect_materialx_resource_roots(std::vector<fs::path>& roots)
{
    /* This intentionally mirrors nanousd-vulkan-renderer/src/material.cpp.
     * Texture resolution is CPU-side renderer policy, not a Metal/Vulkan GPU
     * detail, so keeping these search roots aligned prevents the same
     * MaterialX graph from finding different files on Linux and macOS. */
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

static std::vector<fs::path>
collect_texture_search_roots(const char* scene_dir, NanousdStage stage)
{
    std::string cache_key = std::string(scene_dir ? scene_dir : "") + "|" +
        std::to_string((uintptr_t)stage);
    static std::unordered_map<std::string, std::vector<fs::path>> cache;
    auto cached = cache.find(cache_key);
    if (cached != cache.end()) return cached->second;

    std::vector<fs::path> roots;
    collect_env_asset_search_roots(roots);

    if (scene_dir && scene_dir[0]) {
        fs::path p(scene_dir);
        const char* home = getenv("HOME");
        for (int up = 0; up <= 4 && !p.empty(); up++, p = p.parent_path()) {
            if (p == fs::path("/") ||
                p == fs::path("/Users") ||
                p == fs::path("/home") ||
                (home && home[0] && p == fs::path(home)))
                break;
            push_unique_path(roots, p);
            push_unique_path(roots, p / "Library");
            push_unique_path(roots, p / "DSX_BP" / "Library");
        }
    }

    if (stage) {
        int n_layers = nanousd_stage_n_layers(stage);
        for (int i = 0; i < n_layers; ++i) {
            const char* lp = nanousd_stage_layer_path(stage, i);
            if (!lp || !lp[0]) continue;
            try {
                fs::path dir = fs::path(lp).parent_path();
                push_unique_path(roots, dir);
                push_unique_path(roots, dir / "Materials");
                push_unique_path(roots, dir / "Textures");
            } catch (...) {}
        }
    }

    collect_materialx_resource_roots(roots);

    /* Drop roots that don't exist as directories. A non-existent root can never
     * contain a resolved file (fs::exists(root/suffix) is always false, and
     * get_filename_index already returns an empty index for it), so filtering is
     * result-identical. It removes the per-miss fs::exists storm and the
     * redundant index builds over the many layer/Materials and layer/Textures
     * dirs that ALab-scale scenes (1099 layers) generate but that mostly don't
     * exist — the dominant cost of texture-path resolution on such scenes. */
    {
        std::vector<fs::path> existing;
        existing.reserve(roots.size());
        for (auto& r : roots) {
            std::error_code ec;
            if (fs::is_directory(r, ec)) existing.push_back(std::move(r));
        }
        roots.swap(existing);
    }

    auto inserted = cache.emplace(std::move(cache_key), std::move(roots));
    return inserted.first->second;
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
    if (s.size() >= 2 && s.front() == '@' && s.back() == '@')
        s = s.substr(1, s.size() - 2);
    if (s.rfind("./", 0) == 0) s = s.substr(2);
    push_suffix(s);

    std::string stripped = s;
    while (stripped.rfind("../", 0) == 0) stripped = stripped.substr(3);
    push_suffix(stripped);

    const char* materialx_example_roots[] = {
        /* Same suffix rescue as Vulkan: official MaterialX examples often
         * author filenames relative to their resource package rather than to
         * the loaded scene layer. */
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

static std::string resolve_via_search_roots(const char* path,
                                            const char* scene_dir,
                                            NanousdStage stage)
{
    std::vector<fs::path> roots = collect_texture_search_roots(scene_dir, stage);
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

/* Per-load resolution memo (port of vulkan 83fee92): resolve_texture_path is
 * called once per texture REFERENCE (~13k for ~157 unique paths on the
 * warehouse) but its result is deterministic per (authored path, scene_dir,
 * stage) within a single load. Memoizing collapses the stat-heavy resolutions
 * to the unique set. Bit-identical: same inputs -> same resolved string.
 * Cleared at the top of materials_load so it never persists across scenes. */
static std::unordered_map<std::string, std::string> g_resolve_cache;

static std::string resolve_texture_path_impl(const char* path, const char* scene_dir,
                                             NanousdStage stage)
{
    NuResolveTimer _rt;  /* load-time instrumentation (actual resolve/stat cost) */
    std::string cleaned_path(path ? path : "");
    if (cleaned_path.size() >= 2 && cleaned_path.front() == '@' &&
        cleaned_path.back() == '@') {
        cleaned_path = cleaned_path.substr(1, cleaned_path.size() - 2);
    }
    path = cleaned_path.c_str();

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

    /* Absent-basename short-circuit: every fallback below ultimately looks for a
     * file whose basename is `filename`, at a shallow relative path under some
     * search root. If that basename does not appear in ANY root's recursive
     * index, no root/suffix probe and no basename lookup can succeed, so the
     * whole fallback cascade is guaranteed to miss. Returning the unresolved
     * path here is result-identical for the shallow suffixes we generate, and it
     * collapses the dominant cost on ALab-scale scenes (1099 layers -> ~2600
     * roots; ~26 absent 4k-texture refs each otherwise drive a roots x suffixes
     * fs::exists storm) to a set of cached index lookups. The per-root index is
     * the same one Fallback 1 builds, so no extra tree walks are introduced for
     * paths that miss; lookups short-circuit on the first root that has the
     * basename, so resolvable paths still fall through to Fallback 0 below. */
    {
        std::string filename = fs::path(path).filename().string();
        if (!filename.empty()) {
            bool anywhere = false;
            for (const fs::path& root :
                 collect_texture_search_roots(scene_dir, stage)) {
                const auto& index = get_filename_index(root.string());
                if (index.find(filename) != index.end()) { anywhere = true; break; }
            }
            if (!anywhere) return resolved;
        }
    }

    /* Fallback 0: try package/layer roots before the basename index.
     * Isaac SimReady assets author many values as `../Materials/foo`
     * inside referenced prop layers. The wrapper's scene_dir is too high
     * in the tree for those relative paths, so use contributing layer
     * directories as anchors. */
    {
        std::string rooted = resolve_via_search_roots(path, scene_dir, stage);
        if (!rooted.empty()) return rooted;
    }

    /* Fallback 1: recursive filename index under scene_dir.
     * Sublayered scenes reference textures relative to the sublayer directory,
     * which nanousd does not currently resolve. Indexing the scene tree
     * by filename handles any depth. */
    try {
        std::string filename = fs::path(path).filename().string();
        if (!filename.empty()) {
            for (const fs::path& root :
                 collect_texture_search_roots(scene_dir, stage)) {
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
        const char* home = getenv("HOME");
        for (int up = 1; up <= 3; up++) {
            fs::path parent = scene;
            for (int i = 0; i < up; i++) parent = parent.parent_path();
            if (parent.empty() || !fs::exists(parent)) break;
            if (parent == fs::path("/") ||
                parent == fs::path("/Users") ||
                parent == fs::path("/home") ||
                (home && home[0] && parent == fs::path(home)))
                break;
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

/* Memoizing wrapper around resolve_texture_path_impl (vulkan 83fee92). Within
 * one load the resolution of an authored path is deterministic (the stage is
 * parsed, not mutated), so caching by (path, scene_dir, stage) is bit-identical
 * and collapses the ~13k stat-heavy resolutions to the unique set. The \x1f
 * separators keep the composite key unambiguous. */
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
static MaterialTexture load_udim_atlas(const std::string& pattern,
                                       const char* scene_dir,
                                       NanousdStage stage = nullptr)
{
    MaterialTexture tex = {};

    // Resolve base path (with <UDIM> still in it)
    // First, try to resolve the pattern without <UDIM> by replacing it with
    // "1001" and using resolve_texture_path for the fallback search.
    std::string test_path = pattern;
    {
        size_t pos = test_path.find("<UDIM>");
        if (pos != std::string::npos)
            test_path.replace(pos, 6, "1001");
    }
    std::string resolved_test = resolve_texture_path(test_path.c_str(),
                                                     scene_dir, stage);
    // Extract the base directory from the resolved test path
    std::string base;
    if (fs::exists(resolved_test)) {
        // Found! Use the resolved directory as base
        std::string resolved_dir = fs::path(resolved_test).parent_path().string();
        std::string rel = pattern;
        if (rel.substr(0, 2) == "./") rel = rel.substr(2);
        std::string filename_part = fs::path(rel).filename().string();
        base = resolved_dir + "/" + filename_part;
    } else if (pattern[0] == '/' || pattern[0] == '\\') {
        base = pattern;
    } else {
        base = std::string(scene_dir) + "/" + pattern;
    }

    size_t udim_pos = base.find("<UDIM>");
    if (udim_pos == std::string::npos) return tex;

    std::string prefix = base.substr(0, udim_pos);
    std::string suffix = base.substr(udim_pos + 6); // strlen("<UDIM>")

    // Scan for existing tiles: UDIM range 1001-1100 (10 rows x 10 cols)
    struct UdimTile {
        int col, row;
        int tile_id;
        unsigned char* pixels;
        int w, h;
    };
    std::vector<UdimTile> tiles;
    int max_col = 0, max_row = 0;

    for (int row = 0; row < 10; row++) {
        for (int col = 0; col < 10; col++) {
            int tile_id = 1001 + col + row * 10;
            std::string tile_path = prefix + std::to_string(tile_id) + suffix;

            // Normalize
            try { tile_path = fs::weakly_canonical(tile_path).string(); }
            catch (...) {}

            int w, h, ch;
            unsigned char* px = stbi_load(tile_path.c_str(), &w, &h, &ch, 4);
            if (px) {
                tiles.push_back({col, row, tile_id, px, w, h});
                if (col > max_col) max_col = col;
                if (row > max_row) max_row = row;
            }
        }
    }

    if (tiles.empty()) {
        fprintf(stderr, "material: UDIM no tiles found for %s\n", pattern.c_str());
        return tex;
    }

    // All tiles should be the same size; use first tile's dimensions
    int tile_w = tiles[0].w;
    int tile_h = tiles[0].h;
    int cols = max_col + 1;
    int rows = max_row + 1;
    int atlas_w = tile_w * cols;
    int atlas_h = tile_h * rows;

    tex.pixels = (unsigned char*)calloc(atlas_w * atlas_h, 4);
    if (!tex.pixels) {
        for (auto& t : tiles) stbi_image_free(t.pixels);
        return tex;
    }
    tex.width = atlas_w;
    tex.height = atlas_h;
    tex.udim_cols = cols;
    tex.udim_rows = rows;

    // Blit tiles into atlas. UDIM row 0 is at the bottom in UV space,
    // but our texture is top-down, so flip vertically.
    for (auto& t : tiles) {
        int dst_x = t.col * tile_w;
        int dst_y = (rows - 1 - t.row) * tile_h; // flip rows

        // Handle mismatched tile sizes gracefully
        int copy_w = std::min(t.w, tile_w);
        int copy_h = std::min(t.h, tile_h);

        for (int y = 0; y < copy_h; y++) {
            memcpy(tex.pixels + ((dst_y + y) * atlas_w + dst_x) * 4,
                   t.pixels + (y * t.w) * 4,
                   copy_w * 4);
        }
        stbi_image_free(t.pixels);
    }

    snprintf(tex.path, sizeof(tex.path), "%s", pattern.c_str());
    fprintf(stderr, "material: UDIM atlas %dx%d (%d tiles, %dx%d grid) from %s\n",
            atlas_w, atlas_h, (int)tiles.size(), cols, rows, pattern.c_str());

    return tex;
}

/* Parallel texture pre-decode (port of vulkan 3918b3b): textures decoded up
 * front across the thread pool land here keyed by resolved path; load_texture()
 * consumes them so the per-material readers stay unchanged. MaterialTexture is
 * POD (no destructor) — map erase transfers pixel ownership to the caller with
 * no copy/free. NUSD_PARALLEL_DECODE=0 disables the pre-pass. */
static std::unordered_map<std::string, MaterialTexture> g_predecoded;

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
                              NanousdStage stage = nullptr)
{
    /* Resolve to canonical path for dedup comparison */
    std::string resolved = resolve_texture_path(file, scene_dir, stage);

    /* Check if already loaded */
    for (int i = 0; i < (int)tex_paths.size(); i++) {
        if (tex_paths[i] == resolved || tex_paths[i] == file) {
            return i;
        }
    }

    /* Load new texture */
    MaterialTexture tex = load_texture(file, scene_dir, stage);
    if (!tex.pixels) return -1;

    int idx = (int)textures.size();
    textures.push_back(tex);
    tex_paths.push_back(resolved);
    return idx;
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
 * This is a deliberately narrow implementation: constants, image /
 * normalmap image nodegraphs, and a compact MaterialX procedural graph IR
 * for common arithmetic, ramp and noise nodes.
 */

namespace {

/* Constant + filename split for one Standard Surface input.  Either
 * is_constant is true (with v[0..nVals-1] populated) or file_path is
 * non-empty (relative to the .mtlx file's directory). */
struct SsInput {
    bool        is_constant   = false;
    bool        is_normal_map = false;   /* set when reached via <normalmap> */
    int         n_vals        = 0;
    float       v[3]          = {0, 0, 0};
    std::string file_path;
    std::string nodegraph_name;
    std::string output_name;
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

/* Resolve a Standard Surface input to a constant, upstream image filename,
 * or unresolved nodegraph reference. Walks at most one level of nodegraph
 * indirection for image/normalmap, while preserving the graph handle for
 * procedural pattern detection. */
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
        r.nodegraph_name = ng_name;
        r.output_name = out_name;
        mx::NodeGraphPtr ng = doc->getNodeGraph(ng_name);
        if (ng) {
            mx::OutputPtr out_elt = ng->getOutput(out_name);
            if (out_elt) {
                std::string driver_name = out_elt->getNodeName();
                mx::NodePtr driver = ng->getNode(driver_name);
                if (driver) {
                    const std::string& cat = driver->getCategory();
                    if (cat == "image" || cat == "tiledimage") {
                        mx::InputPtr fileIn = driver->getInput("file");
                        if (fileIn) r.file_path = fileIn->getValueString();
                        return r;
                    }
                    if (cat == "normalmap") {
                        mx::InputPtr nmIn = driver->getInput("in");
                        if (nmIn) {
                            std::string inner = nmIn->getNodeName();
                            mx::NodePtr inner_node = ng->getNode(inner);
                            if (inner_node && (inner_node->getCategory() == "image" ||
                                               inner_node->getCategory() == "tiledimage")) {
                                mx::InputPtr fileIn = inner_node->getInput("file");
                                if (fileIn) r.file_path = fileIn->getValueString();
                                r.is_normal_map = true;
                                return r;
                            }
                        }
                    }
                    /* Other node categories are retained as nodegraph refs for
                     * procedural detection below. */
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

static bool read_input_values(mx::InputPtr in,
                              mx::NodeGraphPtr ng,
                              int max_vals,
                              float* out)
{
    if (!in || !out || max_vals <= 0) return false;

    const std::string& iface = in->getInterfaceName();
    if (!iface.empty() && ng) {
        mx::InputPtr graph_in = ng->getInput(iface);
        if (graph_in) {
            std::string vs = graph_in->getValueString();
            if (!vs.empty() && parse_value(vs, max_vals, out) > 0) return true;
        }
    }

    std::string vs = in->getValueString();
    return !vs.empty() && parse_value(vs, max_vals, out) > 0;
}

static float read_graph_float(mx::NodeGraphPtr ng,
                              const char* name,
                              float fallback)
{
    if (!ng || !name) return fallback;
    mx::InputPtr in = ng->getInput(name);
    float v = fallback;
    if (read_input_values(in, ng, 1, &v)) return v;
    return fallback;
}

static int read_graph_int(mx::NodeGraphPtr ng,
                          const char* name,
                          int fallback)
{
    float v = (float)fallback;
    if (ng && name && read_input_values(ng->getInput(name), ng, 1, &v))
        return std::max(1, (int)(v + 0.5f));
    return fallback;
}

static void read_graph_color(mx::NodeGraphPtr ng,
                             const char* name,
                             const float fallback[3],
                             float out[4])
{
    out[0] = fallback[0];
    out[1] = fallback[1];
    out[2] = fallback[2];
    out[3] = 1.0f;
    if (!ng || !name) return;
    mx::InputPtr in = ng->getInput(name);
    float v[3] = {out[0], out[1], out[2]};
    if (read_input_values(in, ng, 3, v)) {
        out[0] = v[0];
        out[1] = v[1];
        out[2] = v[2];
    }
}

static void read_color_from_node_input(mx::NodeGraphPtr ng,
                                       mx::InputPtr in,
                                       const float fallback[3],
                                       float out[4])
{
    out[0] = fallback[0];
    out[1] = fallback[1];
    out[2] = fallback[2];
    out[3] = 1.0f;
    float v[3] = {out[0], out[1], out[2]};
    if (read_input_values(in, ng, 3, v)) {
        out[0] = v[0];
        out[1] = v[1];
        out[2] = v[2];
    }
}

static float read_float_from_node_input(mx::NodeGraphPtr ng,
                                        mx::InputPtr in,
                                        float fallback)
{
    float v = fallback;
    if (read_input_values(in, ng, 1, &v)) return v;
    return fallback;
}

struct ProcGraphBuild {
    mx::NodeGraphPtr ng;
    MaterialParams* params;
    const std::string* mtlx_dir = nullptr;
    std::vector<std::string>* tex_paths = nullptr;
    std::vector<MaterialTexture>* textures = nullptr;
    std::unordered_map<std::string, int> node_to_index;
    bool ok = true;
};

static int proc_type_from_mtlx(const std::string& type)
{
    if (type == "float" || type == "integer")
        return MTLX_PROC_TYPE_FLOAT;
    return MTLX_PROC_TYPE_VEC3;
}

static int append_proc_node(ProcGraphBuild& gb,
                            int op,
                            int type,
                            int in0,
                            int in1,
                            int in2,
                            int in3,
                            const float value[4])
{
    if (!gb.params || gb.params->procedural_node_count >= MAX_MTLX_PROC_NODES) {
        gb.ok = false;
        return -1;
    }
    int idx = gb.params->procedural_node_count++;
    MtlxProcNode& n = gb.params->procedural_nodes[idx];
    memset(&n, 0, sizeof(n));
    n.op = op;
    n.type = type;
    n.in0 = in0;
    n.in1 = in1;
    n.in2 = in2;
    n.in3 = in3;
    n.value[0] = value ? value[0] : 0.0f;
    n.value[1] = value ? value[1] : 0.0f;
    n.value[2] = value ? value[2] : 0.0f;
    n.value[3] = value ? value[3] : 0.0f;
    return idx;
}

static int append_proc_const(ProcGraphBuild& gb,
                             int type,
                             const float value[4])
{
    return append_proc_node(gb, MTLX_PROC_OP_CONST, type,
                            -1, -1, -1, -1, value);
}

static bool read_proc_input_value(mx::InputPtr in,
                                  mx::NodeGraphPtr ng,
                                  int type,
                                  const float fallback[4],
                                  float out[4])
{
    out[0] = fallback ? fallback[0] : 0.0f;
    out[1] = fallback ? fallback[1] : 0.0f;
    out[2] = fallback ? fallback[2] : 0.0f;
    out[3] = fallback ? fallback[3] : 1.0f;
    if (!in) return false;
    int n = (type == MTLX_PROC_TYPE_FLOAT) ? 1 : 3;
    return read_input_values(in, ng, n, out);
}

static int compile_proc_node(ProcGraphBuild& gb, mx::NodePtr node);

static int compile_proc_input(ProcGraphBuild& gb,
                              mx::NodePtr node,
                              const char* input_name,
                              int fallback_type,
                              const float fallback[4])
{
    if (!node || !input_name) return append_proc_const(gb, fallback_type, fallback);

    mx::InputPtr in = node->getInput(input_name);
    if (!in) return append_proc_const(gb, fallback_type, fallback);

    std::string upstream_name = in->getNodeName();
    if (!upstream_name.empty()) {
        mx::NodePtr upstream = gb.ng ? gb.ng->getNode(upstream_name) : nullptr;
        if (upstream) return compile_proc_node(gb, upstream);
    }

    int type = proc_type_from_mtlx(in->getType().empty()
                                  ? node->getType()
                                  : in->getType());
    if (fallback_type != MTLX_PROC_TYPE_FLOAT &&
        type == MTLX_PROC_TYPE_FLOAT &&
        in->getValueString().empty() &&
        in->getInterfaceName().empty())
        type = fallback_type;

    float v[4];
    read_proc_input_value(in, gb.ng, type, fallback, v);
    return append_proc_const(gb, type, v);
}

static int append_proc_position(ProcGraphBuild& gb)
{
    float zero[4] = {0, 0, 0, 1};
    return append_proc_node(gb, MTLX_PROC_OP_POSITION, MTLX_PROC_TYPE_VEC3,
                            -1, -1, -1, -1, zero);
}

static int load_proc_texture(ProcGraphBuild& gb, mx::NodePtr node)
{
    if (!gb.mtlx_dir || !gb.tex_paths || !gb.textures || !node) {
        gb.ok = false;
        return -1;
    }
    mx::InputPtr file_in = node->getInput("file");
    if (!file_in || file_in->getValueString().empty()) {
        gb.ok = false;
        return -1;
    }
    int tex = load_texture_dedup(file_in->getValueString().c_str(),
                                 gb.mtlx_dir->c_str(),
                                 *gb.tex_paths, *gb.textures, nullptr);
    if (tex < 0) gb.ok = false;
    else if (node->getType() == "color3" && tex < (int)gb.textures->size())
        (*gb.textures)[tex].is_srgb = 1;
    return tex;
}

static int compile_proc_node(ProcGraphBuild& gb, mx::NodePtr node)
{
    if (!node) { gb.ok = false; return -1; }
    auto it = gb.node_to_index.find(node->getName());
    if (it != gb.node_to_index.end()) return it->second;

    const std::string& cat = node->getCategory();
    int type = proc_type_from_mtlx(node->getType());
    float zero[4] = {0, 0, 0, 1};
    float one[4] = {1, 1, 1, 1};
    int idx = -1;

    if (cat == "constant") {
        float v[4];
        read_proc_input_value(node->getInput("value"), gb.ng, type, zero, v);
        idx = append_proc_const(gb, type, v);
    } else if (cat == "position") {
        idx = append_proc_position(gb);
    } else if (cat == "texcoord") {
        idx = append_proc_node(gb, MTLX_PROC_OP_TEXCOORD, MTLX_PROC_TYPE_VEC3,
                               -1, -1, -1, -1, zero);
    } else if (cat == "image" || cat == "tiledimage") {
        int tex = load_proc_texture(gb, node);
        if (tex < 0) return -1;

        int uv_idx = -1;
        float v[4] = {(float)tex, 1.0f, 0.0f, 0.0f};
        if (node->getInput("uvtiling")) {
            uv_idx = compile_proc_input(gb, node, "uvtiling",
                                        MTLX_PROC_TYPE_VEC3, one);
            v[1] = 1.0f; /* input is tiling; multiply current UV by it */
        } else if (node->getInput("texcoord")) {
            uv_idx = compile_proc_input(gb, node, "texcoord",
                                        MTLX_PROC_TYPE_VEC3, zero);
            v[1] = 0.0f; /* input is already a texture coordinate */
        } else {
            uv_idx = append_proc_const(gb, MTLX_PROC_TYPE_VEC3, one);
            v[1] = 1.0f;
        }
        idx = append_proc_node(gb, MTLX_PROC_OP_TEXTURE, type,
                               uv_idx, -1, -1, -1, v);
    } else if (cat == "add" || cat == "subtract" ||
               cat == "multiply" || cat == "divide" ||
               cat == "min" || cat == "max") {
        const float* rhs_default = (cat == "multiply" || cat == "divide") ? one : zero;
        int a = compile_proc_input(gb, node, "in1", type, zero);
        int b = compile_proc_input(gb, node, "in2", type, rhs_default);
        int op = MTLX_PROC_OP_ADD;
        if (cat == "subtract") op = MTLX_PROC_OP_SUBTRACT;
        else if (cat == "multiply") op = MTLX_PROC_OP_MULTIPLY;
        else if (cat == "divide") op = MTLX_PROC_OP_DIVIDE;
        else if (cat == "min") op = MTLX_PROC_OP_MIN;
        else if (cat == "max") op = MTLX_PROC_OP_MAX;
        idx = append_proc_node(gb, op, type, a, b, -1, -1, zero);
    } else if (cat == "dotproduct") {
        int a = compile_proc_input(gb, node, "in1", MTLX_PROC_TYPE_VEC3, zero);
        int b = compile_proc_input(gb, node, "in2", MTLX_PROC_TYPE_VEC3, one);
        idx = append_proc_node(gb, MTLX_PROC_OP_DOT, MTLX_PROC_TYPE_FLOAT,
                               a, b, -1, -1, zero);
    } else if (cat == "sin" || cat == "abs") {
        int a = compile_proc_input(gb, node, "in", type, zero);
        int op = (cat == "sin") ? MTLX_PROC_OP_SIN : MTLX_PROC_OP_ABS;
        idx = append_proc_node(gb, op, type, a, -1, -1, -1, zero);
    } else if (cat == "power") {
        int a = compile_proc_input(gb, node, "in1", type, zero);
        int b = compile_proc_input(gb, node, "in2", MTLX_PROC_TYPE_FLOAT, one);
        idx = append_proc_node(gb, MTLX_PROC_OP_POWER, type, a, b, -1, -1, zero);
    } else if (cat == "mix") {
        int bg = compile_proc_input(gb, node, "bg", type, zero);
        int fg = compile_proc_input(gb, node, "fg", type, one);
        int mxv = compile_proc_input(gb, node, "mix", MTLX_PROC_TYPE_FLOAT, zero);
        idx = append_proc_node(gb, MTLX_PROC_OP_MIX, type, bg, fg, mxv, -1, zero);
    } else if (cat == "clamp") {
        int in_idx = compile_proc_input(gb, node, "in", type, zero);
        int low = compile_proc_input(gb, node, "low", type, zero);
        int high = compile_proc_input(gb, node, "high", type, one);
        idx = append_proc_node(gb, MTLX_PROC_OP_CLAMP, type, in_idx, low, high, -1, zero);
    } else if (cat == "fractal3d") {
        int pos = compile_proc_input(gb, node, "position",
                                     MTLX_PROC_TYPE_VEC3, zero);
        float amplitude = 1.0f;
        float octaves = 3.0f;
        float lacunarity = 2.0f;
        float diminish = 0.5f;
        read_input_values(node->getInput("amplitude"), gb.ng, 1, &amplitude);
        read_input_values(node->getInput("octaves"), gb.ng, 1, &octaves);
        read_input_values(node->getInput("lacunarity"), gb.ng, 1, &lacunarity);
        read_input_values(node->getInput("diminish"), gb.ng, 1, &diminish);
        float v[4] = {amplitude, octaves, lacunarity, diminish};
        idx = append_proc_node(gb, MTLX_PROC_OP_FRACTAL3D, type,
                               pos, -1, -1, -1, v);
    } else if (cat == "convert") {
        int a = compile_proc_input(gb, node, "in", type, zero);
        idx = append_proc_node(gb, MTLX_PROC_OP_CONVERT, type,
                               a, -1, -1, -1, zero);
    } else if (cat == "combine3") {
        int a = compile_proc_input(gb, node, "in1", MTLX_PROC_TYPE_FLOAT, zero);
        int b = compile_proc_input(gb, node, "in2", MTLX_PROC_TYPE_FLOAT, zero);
        int c = compile_proc_input(gb, node, "in3", MTLX_PROC_TYPE_FLOAT, zero);
        idx = append_proc_node(gb, MTLX_PROC_OP_COMBINE3, MTLX_PROC_TYPE_VEC3,
                               a, b, c, -1, zero);
    } else if (cat == "extract") {
        int a = compile_proc_input(gb, node, "in", MTLX_PROC_TYPE_VEC3, zero);
        float index = 0.0f;
        read_input_values(node->getInput("index"), gb.ng, 1, &index);
        float v[4] = {index, 0.0f, 0.0f, 0.0f};
        idx = append_proc_node(gb, MTLX_PROC_OP_EXTRACT, MTLX_PROC_TYPE_FLOAT,
                               a, -1, -1, -1, v);
    } else if (cat == "invert") {
        int a = compile_proc_input(gb, node, "in", type, zero);
        int amount = compile_proc_input(gb, node, "amount", type, one);
        idx = append_proc_node(gb, MTLX_PROC_OP_INVERT, type,
                               a, amount, -1, -1, zero);
    } else if (cat == "ifgreater") {
        int value1 = compile_proc_input(gb, node, "value1",
                                        MTLX_PROC_TYPE_FLOAT, zero);
        int value2 = compile_proc_input(gb, node, "value2",
                                        MTLX_PROC_TYPE_FLOAT, zero);
        int in1 = compile_proc_input(gb, node, "in1", type, zero);
        int in2 = compile_proc_input(gb, node, "in2", type, zero);
        idx = append_proc_node(gb, MTLX_PROC_OP_IFGREATER, type,
                               value1, value2, in1, in2, zero);
    } else if (cat == "ramptb") {
        int top = compile_proc_input(gb, node, "valuet", type, one);
        int bottom = compile_proc_input(gb, node, "valueb", type, zero);
        int tc = compile_proc_input(gb, node, "texcoord",
                                    MTLX_PROC_TYPE_VEC3, zero);
        idx = append_proc_node(gb, MTLX_PROC_OP_RAMPTB, type,
                               top, bottom, tc, -1, zero);
    } else if (cat == "noise3d") {
        int pos = compile_proc_input(gb, node, "position",
                                     MTLX_PROC_TYPE_VEC3, zero);
        float amplitude = 1.0f;
        float pivot = 0.0f;
        read_input_values(node->getInput("amplitude"), gb.ng, 1, &amplitude);
        read_input_values(node->getInput("pivot"), gb.ng, 1, &pivot);
        float v[4] = {amplitude, pivot, 0.0f, 0.0f};
        idx = append_proc_node(gb, MTLX_PROC_OP_NOISE3D, type,
                               pos, -1, -1, -1, v);
    } else if (cat == "cellnoise2d" || cat == "cellnoise3d") {
        const char* coord_name = (cat == "cellnoise2d") ? "texcoord" : "position";
        int pos = compile_proc_input(gb, node, coord_name,
                                     MTLX_PROC_TYPE_VEC3, zero);
        float v[4] = {(cat == "cellnoise2d") ? 2.0f : 3.0f,
                      0.0f, 0.0f, 0.0f};
        idx = append_proc_node(gb, MTLX_PROC_OP_CELLNOISE, type,
                               pos, -1, -1, -1, v);
    } else if (cat == "rgbtohsv" || cat == "hsvtorgb") {
        int a = compile_proc_input(gb, node, "in", MTLX_PROC_TYPE_VEC3, zero);
        int op = (cat == "rgbtohsv") ? MTLX_PROC_OP_RGBTOHSV : MTLX_PROC_OP_HSVTORGB;
        idx = append_proc_node(gb, op, MTLX_PROC_TYPE_VEC3,
                               a, -1, -1, -1, zero);
    } else if (cat == "normalmap") {
        int a = compile_proc_input(gb, node, "in", MTLX_PROC_TYPE_VEC3, zero);
        idx = append_proc_node(gb, MTLX_PROC_OP_CONVERT, MTLX_PROC_TYPE_VEC3,
                               a, -1, -1, -1, zero);
    } else {
        gb.ok = false;
        return -1;
    }

    if (idx >= 0) gb.node_to_index[node->getName()] = idx;
    return idx;
}

static int compile_proc_graph(const SsInput& in,
                              mx::DocumentPtr doc,
                              MaterialParams* params,
                              const std::string& mtlx_dir,
                              std::vector<std::string>& tex_paths,
                              std::vector<MaterialTexture>& textures)
{
    if (!doc || !params || in.nodegraph_name.empty() || in.output_name.empty())
        return -1;
    mx::NodeGraphPtr ng = doc->getNodeGraph(in.nodegraph_name);
    if (!ng) return -1;
    mx::OutputPtr out = ng->getOutput(in.output_name);
    if (!out) return -1;
    mx::NodePtr driver = ng->getNode(out->getNodeName());
    if (!driver) return -1;

    int start_count = params->procedural_node_count;
    ProcGraphBuild gb;
    gb.ng = ng;
    gb.params = params;
    gb.mtlx_dir = &mtlx_dir;
    gb.tex_paths = &tex_paths;
    gb.textures = &textures;
    int idx = compile_proc_node(gb, driver);
    if (!gb.ok || idx < 0) {
        for (int i = start_count; i < params->procedural_node_count; ++i)
            memset(&params->procedural_nodes[i], 0, sizeof(params->procedural_nodes[i]));
        params->procedural_node_count = start_count;
        return -1;
    }
    params->procedural_kind = 2;
    return idx;
}

enum {
    MTLX_PROC_SLOT_BASE_COLOR = 1,
    MTLX_PROC_SLOT_SUBSURFACE_COLOR = 2,
};

static bool detect_marble3d_graph(const SsInput& in,
                                  mx::DocumentPtr doc,
                                  MaterialParams* params,
                                  int slot)
{
    if (!doc || !params || in.nodegraph_name.empty() || in.output_name.empty())
        return false;

    mx::NodeGraphPtr ng = doc->getNodeGraph(in.nodegraph_name);
    if (!ng) return false;

    mx::OutputPtr out = ng->getOutput(in.output_name);
    if (!out) return false;

    mx::NodePtr driver = ng->getNode(out->getNodeName());
    if (!driver || driver->getCategory() != "mix") return false;

    bool has_fractal3d = false;
    bool has_sin = false;
    for (mx::NodePtr node : ng->getNodes()) {
        if (!node) continue;
        const std::string& cat = node->getCategory();
        if (cat == "fractal3d") has_fractal3d = true;
        if (cat == "sin") has_sin = true;
    }
    if (!has_fractal3d || !has_sin) return false;

    static const float c1_default[3] = {0.8f, 0.8f, 0.8f};
    static const float c2_default[3] = {0.1f, 0.1f, 0.3f};
    float color1[4], color2[4];
    read_graph_color(ng, "base_color_1", c1_default, color1);
    read_graph_color(ng, "base_color_2", c2_default, color2);
    read_color_from_node_input(ng, driver->getInput("bg"), c1_default, color1);
    read_color_from_node_input(ng, driver->getInput("fg"), c2_default, color2);

    float scale1 = read_graph_float(ng, "noise_scale_1", 6.0f);
    float scale2 = read_graph_float(ng, "noise_scale_2", 4.0f);
    float power  = read_graph_float(ng, "noise_power", 3.0f);
    int octaves  = read_graph_int(ng, "noise_octaves", 3);
    float amp    = 3.0f;
    if (mx::NodePtr scale_noise = ng->getNode("scale_noise")) {
        if (scale_noise->getCategory() == "multiply") {
            amp = read_float_from_node_input(ng, scale_noise->getInput("in2"), amp);
        }
    }

    params->procedural_kind = 1;
    if (slot == MTLX_PROC_SLOT_BASE_COLOR)
        params->procedural_base_color = 1;
    if (slot == MTLX_PROC_SLOT_SUBSURFACE_COLOR)
        params->procedural_subsurface_color = 1;
    params->procedural_octaves = std::max(1, std::min(octaves, 8));
    memcpy(params->procedural_color1, color1, sizeof(color1));
    memcpy(params->procedural_color2, color2, sizeof(color2));
    params->procedural_params[0] = scale1;
    params->procedural_params[1] = scale2;
    params->procedural_params[2] = std::max(power, 0.001f);
    params->procedural_params[3] = amp;
    return true;
}

}  // namespace

static void init_mdl_texture_transforms(MaterialParams* params);

static void read_standard_surface_mtlx(
    mx::NodePtr ss,
    mx::DocumentPtr doc,
    MaterialParams* params,
    const std::string& mtlx_dir,
    std::vector<std::string>& tex_paths,
    std::vector<MaterialTexture>& textures)
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
    params->normal_tex_scale[0] = 2.0f;
    params->normal_tex_scale[1] = 2.0f;
    params->normal_tex_scale[2] = 2.0f;
    params->normal_tex_scale[3] = 1.0f;
    params->normal_tex_bias[0]  = -1.0f;
    params->normal_tex_bias[1]  = -1.0f;
    params->normal_tex_bias[2]  = -1.0f;
    params->normal_tex_bias[3]  =  0.0f;
    init_mdl_texture_transforms(params);
    params->v_flip = 0;
    params->use_vertex_color = 0;
    params->udim_scale_u = 1.0f;
    params->udim_scale_v = 1.0f;
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
    params->standard_surface_lobes = 1;
    params->base_weight = 1.0f;
    params->specular_weight = 1.0f;
    params->specular_color[0] = 1.0f;
    params->specular_color[1] = 1.0f;
    params->specular_color[2] = 1.0f;
    params->specular_color[3] = 1.0f;
    params->sheen_weight = 0.0f;
    params->sheen_roughness = 0.3f;
    params->sheen_color[0] = 1.0f;
    params->sheen_color[1] = 1.0f;
    params->sheen_color[2] = 1.0f;
    params->sheen_color[3] = 1.0f;
    params->thin_film_thickness = 0.0f;
    params->thin_film_ior = 1.5f;
    params->specular_anisotropy = 0.0f;
    params->procedural_kind = 0;
    params->procedural_base_color = 0;
    params->procedural_subsurface_color = 0;
    params->procedural_octaves = 0;
    params->procedural_color1[0] = 0.8f;
    params->procedural_color1[1] = 0.8f;
    params->procedural_color1[2] = 0.8f;
    params->procedural_color1[3] = 1.0f;
    params->procedural_color2[0] = 0.1f;
    params->procedural_color2[1] = 0.1f;
    params->procedural_color2[2] = 0.3f;
    params->procedural_color2[3] = 1.0f;
    params->procedural_params[0] = 6.0f;
    params->procedural_params[1] = 4.0f;
    params->procedural_params[2] = 3.0f;
    params->procedural_params[3] = 3.0f;
    params->procedural_node_count = 0;
    params->procedural_base_color_output = -1;
    params->procedural_subsurface_color_output = -1;
    params->procedural_roughness_output = -1;
    params->procedural_normal_output = -1;
    params->procedural_graph_flags = 0;
    params->procedural_graph_pad0 = 0;
    params->procedural_graph_pad1 = 0;
    memset(params->procedural_nodes, 0, sizeof(params->procedural_nodes));

    if (!ss) return;

    auto load = [&](const std::string& rel) -> int {
        if (rel.empty()) return -1;
        return load_texture_dedup(rel.c_str(), mtlx_dir.c_str(),
                                  tex_paths, textures, nullptr);
    };

    /* Helper: assign a constant or texture into a (vec3 + tex) slot. */
    auto bind_color3 = [&](const SsInput& in, float* out_rgb, int tex_slot) {
        if (in.is_constant && in.n_vals >= 3) {
            out_rgb[0] = in.v[0];
            out_rgb[1] = in.v[1];
            out_rgb[2] = in.v[2];
        } else if (!in.file_path.empty()) {
            int t = load(in.file_path);
            if (t >= 0 && tex_slot >= 0) params->tex_indices[tex_slot] = t;
        }
    };

    auto bind_float = [&](const SsInput& in, float* out_f, int tex_slot) {
        if (in.is_constant && in.n_vals >= 1) {
            *out_f = in.v[0];
        } else if (!in.file_path.empty()) {
            int t = load(in.file_path);
            if (t >= 0 && tex_slot >= 0) params->tex_indices[tex_slot] = t;
        }
    };

    /* === Standard Surface inputs === */
    bind_float(resolve_ss_input(ss, doc, "base"),
               &params->base_weight, -1);
    bind_float(resolve_ss_input(ss, doc, "specular"),
               &params->specular_weight, -1);
    bind_color3(resolve_ss_input(ss, doc, "specular_color"),
                params->specular_color, -1);

    {
        SsInput base = resolve_ss_input(ss, doc, "base_color");
        bind_color3(base, params->base_color, TEX_DIFFUSE_COLOR);
        if (!base.file_path.empty() && params->tex_indices[TEX_DIFFUSE_COLOR] >= 0) {
            params->base_color[0] = 1.0f;
            params->base_color[1] = 1.0f;
            params->base_color[2] = 1.0f;
        } else if (!base.is_constant && base.file_path.empty()) {
            int out = compile_proc_graph(base, doc, params, mtlx_dir,
                                         tex_paths, textures);
            if (out >= 0) {
                params->procedural_base_color_output = out;
                if (detect_marble3d_graph(base, doc, params,
                                          MTLX_PROC_SLOT_BASE_COLOR)) {
                    params->procedural_kind = 2;
                    params->procedural_graph_flags |= 1;
                }
            } else {
                detect_marble3d_graph(base, doc, params, MTLX_PROC_SLOT_BASE_COLOR);
            }
        }
    }

    bind_float(resolve_ss_input(ss, doc, "metalness"),
               &params->metallic, TEX_METALLIC);

    {
        SsInput rough = resolve_ss_input(ss, doc, "specular_roughness");
        bind_float(rough, &params->roughness, TEX_ROUGHNESS);
        if (!rough.is_constant && rough.file_path.empty()) {
            int out = compile_proc_graph(rough, doc, params, mtlx_dir,
                                         tex_paths, textures);
            if (out >= 0) params->procedural_roughness_output = out;
        }
    }
    bind_float(resolve_ss_input(ss, doc, "specular_anisotropy"),
               &params->specular_anisotropy, -1);
    bind_float(resolve_ss_input(ss, doc, "coat"),
               &params->clearcoat, -1);
    bind_float(resolve_ss_input(ss, doc, "coat_roughness"),
               &params->clearcoat_roughness, -1);
    {
        /* Standard Surface applies coat_color as attenuation to the base
         * layer under the dielectric coat. Reuse the base-color slot for
         * direct coat-color textures; this covers the official tiled brass
         * sample without adding another GPU texture index. */
        SsInput coat_color = resolve_ss_input(ss, doc, "coat_color");
        if (params->clearcoat > 0.0f) {
            float coat_mix = std::max(0.0f, std::min(params->clearcoat, 1.0f));
            if (coat_color.is_constant && coat_color.n_vals >= 3) {
                for (int i = 0; i < 3; ++i) {
                    float atten = (1.0f - coat_mix) + coat_color.v[i] * coat_mix;
                    params->base_color[i] *= atten;
                }
            } else if (!coat_color.file_path.empty() &&
                       params->tex_indices[TEX_DIFFUSE_COLOR] < 0) {
                int t = load(coat_color.file_path);
                if (t >= 0) {
                    params->tex_indices[TEX_DIFFUSE_COLOR] = t;
                    if (t < (int)textures.size()) textures[t].is_srgb = 1;
                }
            }
        }
    }
    bind_float(resolve_ss_input(ss, doc, "sheen"),
               &params->sheen_weight, -1);
    bind_color3(resolve_ss_input(ss, doc, "sheen_color"),
                params->sheen_color, -1);
    bind_float(resolve_ss_input(ss, doc, "sheen_roughness"),
               &params->sheen_roughness, -1);
    bind_float(resolve_ss_input(ss, doc, "thin_film_thickness"),
               &params->thin_film_thickness, -1);
    bind_float(resolve_ss_input(ss, doc, "thin_film_IOR"),
               &params->thin_film_ior, -1);
    bind_float(resolve_ss_input(ss, doc, "opacity"),
               &params->opacity, -1);

    /* `normal` often arrives via a <normalmap> wrapper. Prefer compiling the
     * graph so tiledimage uvtiling is preserved; fall back to the legacy direct
     * texture slot if a broader graph shape is unsupported. */
    SsInput nrm = resolve_ss_input(ss, doc, "normal");
    if (!nrm.is_constant && !nrm.nodegraph_name.empty()) {
        int out = compile_proc_graph(nrm, doc, params, mtlx_dir,
                                     tex_paths, textures);
        if (out >= 0) params->procedural_normal_output = out;
    }
    if (!nrm.file_path.empty()) {
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
    /* subsurface_color: chess King + Queen wire this through a
     * nodegraph that resolves to the same texture as base_color. We
     * don't have a dedicated texture slot for SSS color, so unauthored
     * / nodegraph-driven inputs leave the constant at the (1,1,1)
     * default; the shader uses the new sss_color_authored flag to fall
     * back to baseColor in that case (port from Vulkan 8056f2e —
     * replaces the earlier (>0.99) sentinel which false-positived on
     * author-specified white SSS like milk or marble). */
    {
        SsInput sc_in = resolve_ss_input(ss, doc, "subsurface_color");
        if (sc_in.is_constant && sc_in.n_vals >= 3) {
            params->subsurface_color[0] = sc_in.v[0];
            params->subsurface_color[1] = sc_in.v[1];
            params->subsurface_color[2] = sc_in.v[2];
            params->sss_color_authored = 1;
        } else if (sc_in.file_path.empty()) {
            int out = compile_proc_graph(sc_in, doc, params, mtlx_dir,
                                         tex_paths, textures);
            if (out >= 0) {
                params->procedural_subsurface_color_output = out;
                if (detect_marble3d_graph(sc_in, doc, params,
                                          MTLX_PROC_SLOT_SUBSURFACE_COLOR)) {
                    params->procedural_kind = 2;
                    params->procedural_graph_flags |= 1;
                }
            } else {
                detect_marble3d_graph(sc_in, doc, params,
                                      MTLX_PROC_SLOT_SUBSURFACE_COLOR);
            }
        }
        /* nodegraph case: sss_color_authored stays 0, shader → baseColor */
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
    /* Same shape as subsurface_color above. transmission_color *is*
     * read by the shader (raytrace.metal:737, the trans_radiance tint),
     * so silently dropping a texture-driven authoring would render
     * tinted-glass assets as untinted white. When the input resolves
     * through a nodegraph (file_path set), zero the constant so the
     * shader's runtime `> 0.001 ? c : float3(1)` fallback kicks in
     * and falls back to no-tint instead of incorrectly using stale
     * default white * baseColor. */
    {
        SsInput tc_in = resolve_ss_input(ss, doc, "transmission_color");
        if (tc_in.is_constant && tc_in.n_vals >= 3) {
            params->transmission_color[0] = tc_in.v[0];
            params->transmission_color[1] = tc_in.v[1];
            params->transmission_color[2] = tc_in.v[2];
        } else if (!tc_in.file_path.empty()) {
            params->transmission_color[0] = 0.0f;
            params->transmission_color[1] = 0.0f;
            params->transmission_color[2] = 0.0f;
        }
    }
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
        if (!fs::exists(root)) return out;
        if (fs::is_regular_file(root) && fs::path(root).extension() == ".mtlx") {
            out.push_back(root);
            return out;
        }
        if (!fs::is_directory(root)) return out;
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
                               std::unordered_map<std::string, int>& mat_name_to_idx)
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
                                       tex_paths, textures);

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

static void read_usd_preview_surface(NanousdPrim shader_prim,
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
    params->normal_tex_scale[0] = 2.0f;
    params->normal_tex_scale[1] = 2.0f;
    params->normal_tex_scale[2] = 2.0f;
    params->normal_tex_scale[3] = 1.0f;
    params->normal_tex_bias[0]  = -1.0f;
    params->normal_tex_bias[1]  = -1.0f;
    params->normal_tex_bias[2]  = -1.0f;
    params->normal_tex_bias[3]  =  0.0f;
    init_mdl_texture_transforms(params);
    params->v_flip = 0;
    params->use_vertex_color = 0;
    params->udim_scale_u = 1.0f;
    params->udim_scale_v = 1.0f;
    params->use_specular_workflow = 0;
    params->specular_color[0] = 0.0f;
    params->specular_color[1] = 0.0f;
    params->specular_color[2] = 0.0f;
    params->specular_color[3] = 0.0f;
    params->opacity_threshold = 0.0f;

    for (int i = 0; i < MAX_MATERIAL_TEXTURES; i++)
        params->tex_indices[i] = -1;

    if (!shader_prim) return;

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
     * Authored as `int` in USDA but attribf works for the {0,1} range.
     * Port from Vulkan 301d753. */
    {
        int spec_ok;
        int spec_int = nanousd_attribi(shader_prim,
                                       "inputs:useSpecularWorkflow",
                                       &spec_ok);
        if (spec_ok) {
            params->use_specular_workflow = (spec_int != 0) ? 1 : 0;
        }
    }
    if (nanousd_attribv3f(shader_prim, "inputs:specularColor", v3)) {
        params->specular_color[0] = v3[0];
        params->specular_color[1] = v3[1];
        params->specular_color[2] = v3[2];
    }

    /* opacityThreshold (UsdPreviewSurface): >0 enables the alpha-cutout
     * binary test in the shader; 0 disables (default). */
    f = nanousd_attribf(shader_prim, "inputs:opacityThreshold", &ok);
    if (ok) params->opacity_threshold = f;

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
                                             tex_paths, textures, stage);
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
        /* For the normal-map slot, also pull inputs:scale and inputs:bias
         * off the UsdUVTexture so the shader can apply them explicitly
         * instead of assuming every normal texture is [0,1] encoded. */
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
        /* UsdUVTexture inputs:fallback (color4f, default (0,0,0,1)) is
         * what the texture node outputs when the file is missing or
         * unbound. Apply the same per-slot mapping the shader would do
         * if the texture had loaded — write to the underlying constant.
         * Skip TEX_NORMAL — a missing normal map should leave the
         * surface flat, not bias the geometry.
         * (Vulkan port: 56dab0f.) */
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
                default:
                    break;
                }
            }
        }
        /* For the diffuse-color slot, fold UsdUVTexture inputs:scale into
         * base_color. The shader computes
         * `baseColor = tex_color.rgb * mat.base_color.rgb`, so writing
         * the scale into base_color reproduces the UsdUVTexture spec
         * `out = sample * scale + bias` for the common bias=0 case.
         *
         * When inputs:diffuseColor is *connected* but has no constant
         * authored, the upstream loader leaves base_color at the
         * UsdPreviewSurface default (0.18). That default is meant for
         * untextured surfaces — once a texture is connected, the
         * texture IS the color source, so we reset base_color to
         * (1,1,1) before applying scale. */
        if (ci.slot == TEX_DIFFUSE_COLOR && tex_loaded) {
            /* When a diffuse texture is connected and no constant
             * inputs:diffuseColor was authored, the upstream nanousd
             * loader left base_color at the UPS default (0.18) — that
             * default is meant for untextured surfaces, so once a
             * texture is the color source we want to start from white
             * and let inputs:scale + texture sample drive the color.
             * Track whether the constant was actually authored above;
             * sentinel-checking the value would misfire on any UPS
             * asset that authors (0.18, 0.18, 0.18) deliberately
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

}

/* ---- Shader type detection ---- */

enum ShaderType {
    SHADER_UNKNOWN = 0,
    SHADER_USD_PREVIEW_SURFACE,
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
        if (strcmp(info_id, "UsdPreviewSurface") == 0 ||
            strstr(info_id, "ND_UsdPreviewSurface"))
            return SHADER_USD_PREVIEW_SURFACE;
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

static std::string lower_ascii(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static int classify_texture_slot(const char* name_hint, const char* tex_path,
                                 bool* is_orm)
{
    if (is_orm) *is_orm = false;
    if (!tex_path || !tex_path[0]) return -1;

    std::string lower_fn = lower_ascii(tex_path);
    if (lower_fn.find(".mdl") != std::string::npos) return -1;

    std::string lower_hint = lower_ascii(name_hint ? name_hint : "");
    if (lower_hint.find("maskselection") != std::string::npos ||
        lower_hint.find("sampler_masks") != std::string::npos ||
        (lower_hint.find("mask") != std::string::npos &&
         lower_hint.find("opacity") == std::string::npos &&
         lower_hint.find("alpha") == std::string::npos &&
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
    if (suffix == "m" && !has_metal_hint) return -1;
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
        lower_hint.find("opacitymask") != std::string::npos ||
        lower_hint.find("alpha_selection") != std::string::npos ||
        lower_hint.find("alphaselection") != std::string::npos ||
        lower_hint.find("sampler_alpha") != std::string::npos ||
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
    baked.is_srgb = 1;
    baked.pixels = (unsigned char*)malloc((size_t)baked.width *
                                          (size_t)baked.height * 4);
    if (!baked.pixels) return -1;

    NuBakeTimer _bt;  /* race-free per-row bake; reads stay serial (vulkan 169b54c) */
    /* Each y-row writes a distinct output span from read-only inputs → safe to
     * parallelise; output is bit-identical to the serial path. */
    nu_parallel_for(baked.height, [&](int y) {
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
    });

    snprintf(baked.path, sizeof(baked.path), "%s", key.c_str());
    int idx = (int)textures.size();
    textures.push_back(baked);
    tex_paths.push_back(key);
    fprintf(stderr,
            "material:   MDL baked masked albedo: %s + %s -> diffuse slot\n",
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
    baked.is_srgb = 1;
    baked.pixels = (unsigned char*)malloc((size_t)baked.width *
                                          (size_t)baked.height * 4);
    if (!baked.pixels) return -1;

    NuBakeTimer _bt;  /* race-free per-row bake; reads stay serial (vulkan 169b54c) */
    nu_parallel_for(baked.height, [&](int y) {
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
    });

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
    baked.is_srgb = 1;
    baked.pixels = (unsigned char*)malloc((size_t)baked.width *
                                          (size_t)baked.height * 4);
    if (!baked.pixels) return -1;

    NuBakeTimer _bt;  /* race-free per-row bake; reads stay serial (vulkan 169b54c) */
    scale = std::max(scale, 0.0f);
    nu_parallel_for(baked.height, [&](int y) {
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
                baked.pixels[ci + c] =
                    linear_to_srgb_u8(color_c * mask_c * scale);
            }
            baked.pixels[ci + 3] = 255;
        }
    });

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
    baked.is_srgb = 1;
    baked.pixels = (unsigned char*)malloc((size_t)baked.width *
                                          (size_t)baked.height * 4);
    if (!baked.pixels) return -1;

    NuBakeTimer _bt;  /* race-free per-row bake; reads stay serial (vulkan 169b54c) */
    u_tiling = std::max(u_tiling, 1.0e-6f);
    v_tiling = std::max(v_tiling, 1.0e-6f);
    plastic_opacity = std::clamp(plastic_opacity, 0.0f, 1.0f);

    nu_parallel_for(baked.height, [&](int y) {
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
                float normal_z = 2.0f *
                    ((float)normal->pixels[ni + 2] / 255.0f) - 1.0f;
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
    });

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

static void init_mdl_texture_transforms(MaterialParams* params)
{
    if (!params) return;
    params->uv_transform[0] = 1.0f;
    params->uv_transform[1] = 1.0f;
    params->uv_transform[2] = 0.0f;
    params->uv_transform[3] = 0.0f;
    params->roughness_tex_transform[0] = 1.0f;
    params->roughness_tex_transform[1] = 0.0f;
    params->roughness_tex_transform[2] = 0.0f;
    params->roughness_tex_transform[3] = 0.0f;
}

static void apply_mdl_uv_tiling(MaterialParams* params,
                                float u_scale, float v_scale,
                                float u_offset = 0.0f,
                                float v_offset = 0.0f)
{
    if (!params) return;
    if (std::isfinite(u_scale) && std::fabs(u_scale) > 1.0e-8f)
        params->uv_transform[0] = u_scale;
    if (std::isfinite(v_scale) && std::fabs(v_scale) > 1.0e-8f) {
        params->uv_transform[1] = v_scale;
        if (std::fabs(v_offset) < 1.0e-8f && params->v_flip == 0)
            v_offset = 1.0f - v_scale;
    }
    if (std::isfinite(u_offset))
        params->uv_transform[2] = u_offset;
    if (std::isfinite(v_offset))
        params->uv_transform[3] = v_offset;
}

static void apply_mdl_roughness_range(MaterialParams* params,
                                      float rmin, float rmax)
{
    if (!params || !std::isfinite(rmin) || !std::isfinite(rmax)) return;
    /* Vulkan stores the same remap as roughness_tex_scale/bias. Metal packs
     * it into roughness_tex_transform.xy because that is the shader ABI here. */
    params->roughness_tex_transform[0] = rmax - rmin;
    params->roughness_tex_transform[1] = rmin;
}

static void assign_texture_to_material(MaterialParams* params,
                                       std::vector<std::string>& tex_paths,
                                       std::vector<MaterialTexture>& textures,
                                       const char* name_hint,
                                       const char* tex_ref,
                                       const char* scene_dir,
                                       NanousdStage stage)
{
    bool is_orm = false;
    int slot = classify_texture_slot(name_hint, tex_ref, &is_orm);
    if (!is_orm && slot < 0) return;

    int tex_idx = load_texture_dedup(tex_ref, scene_dir, tex_paths, textures, stage);
    if (tex_idx < 0 || tex_idx >= (int)textures.size()) return;

    if (is_orm) {
        if (params->tex_indices[TEX_OCCLUSION] < 0)
            params->tex_indices[TEX_OCCLUSION] = tex_idx;
        if (params->tex_indices[TEX_ROUGHNESS] < 0)
            params->tex_indices[TEX_ROUGHNESS] = tex_idx;
        if (params->tex_indices[TEX_METALLIC] < 0)
            params->tex_indices[TEX_METALLIC] = tex_idx;
    } else if (params->tex_indices[slot] < 0) {
        params->tex_indices[slot] = tex_idx;
    }
    propagate_udim_scale(params, textures[tex_idx]);
}

static std::string trim_ascii(const std::string& s)
{
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char)s[b])) b++;
    size_t e = s.size();
    while (e > b && std::isspace((unsigned char)s[e - 1])) e--;
    return s.substr(b, e - b);
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
    if (!params) return;
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
        if (c[0] + c[1] + c[2] > 0.001f) {
            params->base_color[0] = c[0];
            params->base_color[1] = c[1];
            params->base_color[2] = c[2];
        }
    }
}

static void apply_mdl_float_param(MaterialParams* params,
                                  const std::string& lname,
                                  float v)
{
    if (!params || !std::isfinite(v)) return;
    if (mdl_name_has_any(lname, {"roughnessmin", "roughness_min"})) {
        params->roughness_tex_transform[1] = v;
    } else if (mdl_name_has_any(lname, {"roughnessmax", "roughness_max"})) {
        params->roughness_tex_transform[0] = v - params->roughness_tex_transform[1];
        params->roughness = v;
    } else if (mdl_name_has_any(lname, {"rough", "roughtness"})) {
        params->roughness = v;
    } else if (mdl_name_has_any(lname, {"metal", "metalness"})) {
        params->metallic = v;
    } else if (mdl_name_has_any(lname, {"opacity_threshold", "alpha_threshold", "cutout_threshold"})) {
        params->opacity_threshold = v;
    } else if (mdl_name_has_any(lname, {"opacity", "alpha", "cutout"})) {
        params->opacity = v;
    } else if (mdl_name_has_any(lname, {"clearcoat_roughness", "coat_roughness"})) {
        params->clearcoat_roughness = v;
    } else if (mdl_name_has_any(lname, {"clearcoat", "coat_weight"})) {
        params->clearcoat = v;
    } else if (mdl_name_has_any(lname, {"transmission_ior", "specular_ior"})) {
        params->transmission_ior = v;
    } else if (mdl_name_has_any(lname, {"ior"})) {
        params->ior = v;
    } else if (mdl_name_has_any(lname, {"normal_scale", "bump_factor", "bump_scale"})) {
        params->normal_scale = v;
    } else if (mdl_name_has_any(lname, {"u_tiling"})) {
        apply_mdl_uv_tiling(params, v, params->uv_transform[1]);
    } else if (mdl_name_has_any(lname, {"v_tiling"})) {
        apply_mdl_uv_tiling(params, params->uv_transform[0], v);
    } else if (mdl_name_has_any(lname, {"transmission", "thin_walled"})) {
        params->transmission_weight = v;
    } else if (mdl_name_has_any(lname, {"subsurface"})) {
        params->subsurface_weight = v;
    } else if (mdl_name_has_any(lname, {"emissivestrength", "emission_strength",
                                        "emission", "emissive", "luminance", "intensity"})) {
        params->emissive_color[3] = v;
    }
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
            if (lname.find("maintiling") != std::string::npos) {
                apply_mdl_uv_tiling(params, v4[0], v4[1], v4[2], v4[3]);
            } else {
                apply_mdl_color_param(params, lname, v4);
            }
            continue;
        }
        if (nanousd_attribv3f(shader_prim, aname, v3)) {
            if (lname.find("maintiling") != std::string::npos)
                apply_mdl_uv_tiling(params, v3[0], v3[1], v3[2], 0.0f);
            else
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

static bool find_matching_paren(const std::string& s, size_t open,
                                size_t* close_out)
{
    bool in_string = false;
    bool escape = false;
    int depth = 0;
    for (size_t i = open; i < s.size(); ++i) {
        char c = s[i];
        if (in_string) {
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '(') depth++;
        else if (c == ')') {
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

static const std::string* load_mdl_source_cached(const std::string& mdl_path)
{
    static std::unordered_map<std::string, std::string> cache;
    auto it = cache.find(mdl_path);
    if (it != cache.end()) return &it->second;

    std::ifstream mdl_file(mdl_path);
    if (!mdl_file.good()) return nullptr;

    std::stringstream buf;
    buf << mdl_file.rdbuf();
    auto inserted = cache.emplace(mdl_path, buf.str());
    return &inserted.first->second;
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

    std::string mdl_resolved = resolve_texture_path(mdl_asset, scene_dir, stage);
    const std::string* cached_body = load_mdl_source_cached(mdl_resolved);
    if (!cached_body) {
        if (getenv("NUSD_MAT_DIAG")) {
            fprintf(stderr,
                    "[mat_diag] MDL sourceAsset '%s' resolved to '%s' but could not be opened\n",
                    mdl_asset, mdl_resolved.c_str());
        }
        return;
    }
    const std::string& body = *cached_body;
    if (body.find("CustomizedUV0_mdl") != std::string::npos &&
        body.find("1.0-state::texture_coordinate(0).y") != std::string::npos) {
        params->v_flip = 0;
    }

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
    /* Keep this path in lockstep with nanousd-vulkan-renderer: IsaacSim MDL
     * files often contain several `export material` blocks, and scanning the
     * whole file lets sibling material textures leak into this material. */
    std::string material_source;
    std::string params_text =
        find_mdl_export_material_params(body, material_name, &material_source);
    const std::string& pattern_source =
        material_source.empty() ? body : material_source;

    float rmin = 0.0f, rmax = 1.0f;
    bool have_rmin = false, have_rmax = false;
    float u_tiling = params->uv_transform[0];
    float v_tiling = params->uv_transform[1];
    bool have_u = false, have_v = false;
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
    float main_tiling[2] = {1.0f, 1.0f};
    float emissive_product_scale = 1.0f;
    bool has_box_tint = false;
    bool has_plastic_tint = false;
    bool has_plastic_opacity = false;
    bool has_plastic_normal_alpha = false;
    bool has_plastic_wrap = false;
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
                    u_tiling = f4[0];
                    v_tiling = f4[1];
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
                } else if (lname.find("coloralbedo") != std::string::npos ||
                           lname.find("basecolor") != std::string::npos ||
                           lname.find("diffusecolor") != std::string::npos) {
                    color_albedo[0] = f4[0];
                    color_albedo[1] = f4[1];
                    color_albedo[2] = f4[2];
                    has_color_albedo = true;
                    if (apply_constant_defaults)
                        apply_mdl_color_param(params, lname, color_albedo);
                } else if (apply_constant_defaults) {
                    apply_mdl_color_param(params, lname, f4);
                }
                continue;
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
                    have_u = true;
                    continue;
                }
                if (lname == "v_tiling") {
                    v_tiling = v;
                    have_v = true;
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
                if (lname == "roughnessmin" ||
                    lname == "roughness_min") {
                    rmin = v;
                    have_rmin = true;
                    continue;
                }
                if (lname == "roughnessmax" ||
                    lname == "roughness_max") {
                    rmax = v;
                    have_rmax = true;
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

        if (have_rmin || have_rmax) {
            float rrmin = have_rmin ? rmin : 0.0f;
            float rrmax = have_rmax ? rmax : rrmin;
            if (rrmax < rrmin) {
                float tmp = rrmin;
                rrmin = rrmax;
                rrmax = tmp;
            }
            rmin = rrmin;
            rmax = rrmax;
            params->roughness = 0.5f * (rmin + rmax);
            apply_mdl_roughness_range(params, rmin, rmax);
        }

        if (has_main_tiling) {
            apply_mdl_uv_tiling(params, main_tiling[0], main_tiling[1]);
        } else if (!has_plastic_wrap && (have_u || have_v)) {
            apply_mdl_uv_tiling(params, u_tiling, v_tiling);
        }
    }

    {
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
    }

    bool baked_body_mask_albedo = false;
    /* Mirrors the Vulkan fallback: IsaacSim assets like traffic cones encode
     * Body/Handle/Cap colors plus an RGBA selection mask instead of a single
     * authored base-color texture. Baking here gives Metal and Vulkan the same
     * compact texture contract before either backend shades the material. */
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
            if (have_rmax) {
                params->tex_indices[TEX_ROUGHNESS] = -1;
                params->roughness = rmax;
                params->roughness_tex_transform[0] = 1.0f;
                params->roughness_tex_transform[1] = 0.0f;
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

    /* Vulkan has the same product bake. The heuristic is intentionally
     * pre-shader: generated MDLs often multiply albedo by a stripe/glow mask
     * in MDL code, which neither backend executes in its fallback renderer. */
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

    if (pattern_source.find("OpacityMask_mdl") != std::string::npos &&
        pattern_source.find("0.3333") != std::string::npos) {
        params->opacity_threshold = 0.3333f;
    }
    if (apply_constant_defaults)
        apply_authored_mdl_shader_inputs(shader_prim, params);
    if (pattern_source.find("OmniUe4Base") != std::string::npos &&
        pattern_source.find("emissive_color") != std::string::npos &&
        (params->emissive_color[0] > 0.0f ||
         params->emissive_color[1] > 0.0f ||
         params->emissive_color[2] > 0.0f) &&
        params->emissive_color[3] > 0.0f) {
        params->emissive_color[3] *= 2560.0f;
    }
}

static bool apply_mdl_sdk_source_material(NanousdPrim shader_prim,
                                          MaterialParams* params,
                                          const char* scene_dir,
                                          NanousdStage stage,
                                          std::vector<std::string>& tex_paths,
                                          std::vector<MaterialTexture>& textures)
{
    if (!shader_prim || !params || !nusd_mdl_bridge_available()) return false;

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
            sub_value.empty() ? nullptr : sub_value.c_str(),
            scene_dir,
            inputs.empty() ? nullptr : inputs.data(),
            (int)inputs.size(), &decoded)) {
        return false;
    }

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
            assign_texture_to_material(params, tex_paths, textures,
                                       hint, tex_ref, scene_dir, stage);
        }
    }

    return true;
}

/* ---- Read OmniPBR / MDL material properties ---- */

static void read_omnipbr_material(NanousdPrim shader_prim,
                                  MaterialParams* params,
                                  std::vector<std::string>& tex_paths,
                                  std::vector<MaterialTexture>& textures,
                                  const char* scene_dir,
                                  NanousdStage stage = nullptr)
{
    NuReaderTimer _rt;  /* per-material reader cost (attribute reads + bakes) */
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
    params->normal_tex_scale[0] = 2.0f;
    params->normal_tex_scale[1] = 2.0f;
    params->normal_tex_scale[2] = 2.0f;
    params->normal_tex_scale[3] = 1.0f;
    params->normal_tex_bias[0]  = -1.0f;
    params->normal_tex_bias[1]  = -1.0f;
    params->normal_tex_bias[2]  = -1.0f;
    params->normal_tex_bias[3]  =  0.0f;
    init_mdl_texture_transforms(params);
    params->v_flip = 0;
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
        params->sheen_color[i] = 1.0f;
    }
    params->standard_surface_lobes = 0;
    params->base_weight = 1.0f;
    params->specular_weight = 1.0f;
    params->sheen_weight = 0.0f;
    params->sheen_roughness = 0.3f;
    params->thin_film_thickness = 0.0f;
    params->thin_film_ior = 1.5f;
    params->specular_anisotropy = 0.0f;

    for (int i = 0; i < MAX_MATERIAL_TEXTURES; i++)
        params->tex_indices[i] = -1;

    if (!shader_prim) return;

    int ok;
    float v3[3];
    {
        int mdl_asset_ok = 0;
        const char* mdl_asset = nanousd_attribasset(shader_prim,
                                                    "info:mdl:sourceAsset",
                                                    &mdl_asset_ok);
        if (mdl_asset_ok && mdl_asset && mdl_asset[0])
            params->v_flip = 1;
    }

    bool mdl_sdk_applied = apply_mdl_sdk_source_material(shader_prim, params,
                                                         scene_dir, stage,
                                                         tex_paths, textures);

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
    /* Isaac/UE MDL tint inputs (BaseColor_Tint, ColorAlbedo) modulate an
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
    /* Isaac MDL: RoughnessMin/Max remaps ORM.g before final roughness clamp. */
    {
        float rmin, rmax;
        int okmin, okmax;
        rmin = nanousd_attribf(shader_prim, "inputs:RoughnessMin", &okmin);
        rmax = nanousd_attribf(shader_prim, "inputs:RoughnessMax", &okmax);
        if (okmin && okmax) {
            params->roughness = 0.5f * (rmin + rmax);
            apply_mdl_roughness_range(params, rmin, rmax);
        }
        else if (okmax)     params->roughness = rmax;
        else if (okmin)     params->roughness = rmin;
    }
    /* Isaac MDL float "Roughness" and "Metallic" */
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
    f = nanousd_attribf(shader_prim, "inputs:enable_emission", &ok);
    bool emission_enabled = (!ok || f > 0.5f); // Default: enabled if emissive_color set
    f = nanousd_attribf(shader_prim, "inputs:emissive_intensity", &ok);
    if (ok && emission_enabled) {
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
    if (!ok) f = nanousd_attribf(shader_prim, "inputs:opacityThreshold", &ok);
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
    // Isaac/UE MDL uses a different naming convention (AlbedoTexture etc.).
    struct { const char* input; int slot; } tex_inputs[] = {
        {"inputs:diffuse_texture",           TEX_DIFFUSE_COLOR},
        {"inputs:reflectionroughness_texture", TEX_ROUGHNESS},
        {"inputs:metallic_texture",          TEX_METALLIC},
        {"inputs:normalmap_texture",         TEX_NORMAL},
        {"inputs:emissive_mask_texture",     TEX_EMISSIVE_COLOR},
        {"inputs:ao_texture",               TEX_OCCLUSION},
        {"inputs:opacity_texture",           TEX_OPACITY},
        {"inputs:diffuse_reflection_color_image", TEX_DIFFUSE_COLOR},
        {"inputs:geometry_normal_image",     TEX_NORMAL},
        {"inputs:geometry_opacity_image",    TEX_OPACITY},
        {"inputs:specular_reflection_roughness_image", TEX_ROUGHNESS},
        {"inputs:metalness_image",           TEX_METALLIC},
        {"inputs:emission_color_image",      TEX_EMISSIVE_COLOR},
        /* Isaac/UE MDL */
        {"inputs:AlbedoTexture",             TEX_DIFFUSE_COLOR},
        {"inputs:BaseColor_Texture",         TEX_DIFFUSE_COLOR},
        {"inputs:BaseColor_Box",             TEX_DIFFUSE_COLOR},
        {"inputs:BaseColor_Plastic",         TEX_DIFFUSE_COLOR},
        {"inputs:TextureSelection",          TEX_DIFFUSE_COLOR},  // Isaac MI_Sign* materials
        {"inputs:Text",                      TEX_DIFFUSE_COLOR},  // Isaac M_AisleSign per-instance text
        {"inputs:MainNormalInput",           TEX_NORMAL},
        {"inputs:NormalMap_Box",             TEX_NORMAL},
        {"inputs:MergeMapInput",             TEX_ROUGHNESS},  // ORM packed; shared across slots below
        {"inputs:MultiMap_Box",              TEX_ROUGHNESS},
        {"inputs:MultiMap_Plastic",          TEX_ROUGHNESS},
        {"inputs:AlphaSelection",            TEX_OPACITY},
    };

    bool isaac_mdl_texture_inputs = false;
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
                strcmp(ti.input, "inputs:BaseColor_Box") == 0 ||
                strcmp(ti.input, "inputs:BaseColor_Plastic") == 0 ||
                strcmp(ti.input, "inputs:TextureSelection") == 0 ||
                strcmp(ti.input, "inputs:Text") == 0 ||
                strcmp(ti.input, "inputs:MainNormalInput") == 0 ||
                strcmp(ti.input, "inputs:NormalMap_Box") == 0 ||
                strcmp(ti.input, "inputs:MergeMapInput") == 0 ||
                strcmp(ti.input, "inputs:MultiMap_Box") == 0 ||
                strcmp(ti.input, "inputs:MultiMap_Plastic") == 0 ||
                strcmp(ti.input, "inputs:AlphaSelection") == 0) {
                isaac_mdl_texture_inputs = true;
            }
            int tex_idx = load_texture_dedup(file, scene_dir, tex_paths, textures, stage);
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
    if (isaac_mdl_texture_inputs)
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

            bool is_orm = false;
            int slot = classify_texture_slot(aname, tex_path, &is_orm);

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
                    if (getenv("NUSD_MAT_DIAG"))
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

    apply_generic_mdl_source_material(shader_prim, params, tex_paths, textures,
                                      scene_dir, stage, !mdl_sdk_applied);

    /* OmniSurface/OmniPBR graphs often store file_texture nodes as
     * Material children with `inputs:texture` instead of UsdUVTexture
     * `inputs:file`, and connect the surface inputs through utility nodes.
     * Bind obvious leaf texture nodes by filename/child-name heuristics. */
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
                assign_texture_to_material(params, tex_paths, textures,
                                           hint.c_str(), file, scene_dir, stage);
                nanousd_freeprim(child);
            }
            nanousd_freeprim(mat_parent);
        }
    }

    /* --- MDL body parsing fallback for Isaac/UE4 custom materials ---
     * M_TrafficCone, M_WetFloorSign, M_AisleSign, etc. reference their
     * textures only via MDL body expressions like
     *   tex::lookup_float4(texture_2d("./Textures/T_Foo_D.png", ...), ...)
     * and NOT as USD attributes, so the scans above find nothing.
     * When diffuse is still missing, open the referenced .mdl file, scan
     * for texture_2d("...") literals, classify by _D / _N / _ORM / _M
     * suffix, and load via load_texture_dedup. */
    if (params->tex_indices[TEX_DIFFUSE_COLOR] < 0) {
        int mok;
        const char* mdl_asset = nanousd_attribasset(shader_prim,
                                                      "info:mdl:sourceAsset", &mok);
        if (mok && mdl_asset && mdl_asset[0]) {
            /* Resolve MDL file path using the same rules as textures. */
            std::string mdl_resolved = resolve_texture_path(mdl_asset, scene_dir, stage);
            std::ifstream mdl_file(mdl_resolved);
            if (mdl_file.good()) {
                std::stringstream buf;
                buf << mdl_file.rdbuf();
                std::string body = buf.str();

                /* Collect texture_2d declarations. The filename usually
                 * carries the slot hint (_D/_N/_ORM), but generic MDLs can
                 * hide that in the variable name before '='. */
                struct MdlTexRef {
                    std::string name_hint;
                    std::string path;
                };
                std::vector<MdlTexRef> tex_refs;
                const std::string key = "texture_2d(";
                size_t pos = 0;
                while ((pos = body.find(key, pos)) != std::string::npos) {
                    size_t q1 = body.find('"', pos + key.size());
                    if (q1 == std::string::npos) break;
                    size_t q2 = body.find('"', q1 + 1);
                    if (q2 == std::string::npos) break;
                    std::string path = body.substr(q1 + 1, q2 - q1 - 1);
                    std::string name_hint;
                    size_t eq = body.rfind('=', pos);
                    size_t semi = body.rfind(';', pos);
                    size_t line = body.rfind('\n', pos);
                    size_t scope = std::max(semi == std::string::npos ? 0 : semi + 1,
                                            line == std::string::npos ? 0 : line + 1);
                    if (eq != std::string::npos && eq >= scope && pos - eq < 512) {
                        size_t end = eq;
                        while (end > scope &&
                               std::isspace((unsigned char)body[end - 1]))
                            --end;
                        size_t start = end;
                        while (start > scope) {
                            unsigned char ch = (unsigned char)body[start - 1];
                            if (!(std::isalnum(ch) || ch == '_' || ch == ':'))
                                break;
                            --start;
                        }
                        if (start < end)
                            name_hint = body.substr(start, end - start);
                    }
                    if (!path.empty()) tex_refs.push_back({name_hint, path});
                    pos = q2 + 1;
                }

                /* The MDL directory becomes the anchor for "./Textures/..."
                 * paths. We pass the MDL's parent directory as scene_dir
                 * so relative refs resolve correctly. */
                std::string mdl_dir;
                try {
                    mdl_dir = fs::path(mdl_resolved).parent_path().string();
                } catch (...) {}
                const char* anchor = mdl_dir.empty() ? scene_dir : mdl_dir.c_str();

                for (const MdlTexRef& ref : tex_refs) {
                    const std::string& tp = ref.path;
                    std::string lower_tp = tp;
                    std::transform(lower_tp.begin(), lower_tp.end(),
                                   lower_tp.begin(), ::tolower);
                    /* skip non-image files */
                    if (lower_tp.find(".png") == std::string::npos &&
                        lower_tp.find(".jpg") == std::string::npos &&
                        lower_tp.find(".jpeg") == std::string::npos &&
                        lower_tp.find(".tga") == std::string::npos &&
                        lower_tp.find(".exr") == std::string::npos)
                        continue;

                    bool is_orm = false;
                    std::string hint = ref.name_hint;
                    hint += " ";
                    hint += tp;
                    int slot = classify_texture_slot(hint.c_str(), tp.c_str(), &is_orm);
                    /* skip detail/mask-only textures like _Stripes, _Mask */
                    if (!is_orm && slot < 0) continue;

                    int idx = load_texture_dedup(tp.c_str(), anchor,
                                                  tex_paths, textures, stage);
                    if (idx < 0) continue;

                    if (is_orm) {
                        if (params->tex_indices[TEX_OCCLUSION] < 0)
                            params->tex_indices[TEX_OCCLUSION] = idx;
                        if (params->tex_indices[TEX_ROUGHNESS] < 0)
                            params->tex_indices[TEX_ROUGHNESS] = idx;
                        if (params->tex_indices[TEX_METALLIC] < 0)
                            params->tex_indices[TEX_METALLIC] = idx;
                        if (getenv("NUSD_MAT_DIAG"))
                            fprintf(stderr,
                                    "material:   MDL-body ORM: %s → AO/R/M slots\n",
                                    tp.c_str());
                    } else if (slot >= 0 && params->tex_indices[slot] < 0) {
                        params->tex_indices[slot] = idx;
                        if (getenv("NUSD_MAT_DIAG"))
                            fprintf(stderr,
                                    "material:   MDL-body tex: %s → slot %d\n",
                                    tp.c_str(), slot);
                    }
                }
            } else if (getenv("NUSD_MAT_DIAG")) {
                fprintf(stderr,
                        "[mat_diag] could not open MDL file '%s' (resolved '%s')\n",
                        mdl_asset, mdl_resolved.c_str());
            }
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
            if (getenv("NUSD_MAT_DIAG"))
                fprintf(stderr, "material: ORM packed texture detected (idx=%d): "
                        "AO→slot5, Rough→slot2, Metal→slot3\n", orm_idx);
        }
    }

    if (material_uses_mdl_baked_masked_albedo(params, textures) ||
        material_uses_mdl_baked_body_masked_albedo(params, textures)) {
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

    if (getenv("NUSD_MAT_DIAG")) {
        fprintf(stderr, "material: read OmniPBR params: base=(%.2f,%.2f,%.2f) "
                "metal=%.2f rough=%.2f emissive=(%.2f,%.2f,%.2f)*%.1f udim=%.0fx%.0f "
                "slots D=%d N=%d R=%d M=%d E=%d AO=%d OP=%d\n",
                params->base_color[0], params->base_color[1], params->base_color[2],
                params->metallic, params->roughness,
                params->emissive_color[0], params->emissive_color[1],
                params->emissive_color[2], params->emissive_color[3],
                params->udim_scale_u, params->udim_scale_v,
                params->tex_indices[TEX_DIFFUSE_COLOR],
                params->tex_indices[TEX_NORMAL],
                params->tex_indices[TEX_ROUGHNESS],
                params->tex_indices[TEX_METALLIC],
                params->tex_indices[TEX_EMISSIVE_COLOR],
                params->tex_indices[TEX_OCCLUSION],
                params->tex_indices[TEX_OPACITY]);
    }
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
    params->normal_tex_scale[0] = 2.0f;
    params->normal_tex_scale[1] = 2.0f;
    params->normal_tex_scale[2] = 2.0f;
    params->normal_tex_scale[3] = 1.0f;
    params->normal_tex_bias[0]  = -1.0f;
    params->normal_tex_bias[1]  = -1.0f;
    params->normal_tex_bias[2]  = -1.0f;
    params->normal_tex_bias[3]  =  0.0f;
    init_mdl_texture_transforms(params);
    params->v_flip = 0;
    params->use_vertex_color = 0;
    params->udim_scale_u = 1.0f;
    params->udim_scale_v = 1.0f;

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
    // Check surface relationship/connection targets. MaterialX USD authored
    // by Reality Composer Pro commonly uses outputs:mtlx:surface.connect,
    // while UsdPreviewSurface and MDL scenes may use outputs:surface or
    // outputs:mdl:surface.
    const char* surface_outputs[] = {
        "outputs:surface",
        "outputs:mtlx:surface",
        "outputs:mdl:surface"
    };
    for (const char* output_name : surface_outputs) {
        int ntargets = nanousd_nreltargets(material_prim, output_name);
        if (ntargets > 0) {
            const char* target = nanousd_reltarget(material_prim, output_name, 0);
            if (target) {
                // Target may be like /Materials/Mat1/Shader.outputs:surface
                // We want the prim part (before the property).
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
            if (target) {
                std::string target_str(target);
                size_t dot = target_str.find('.');
                if (dot != std::string::npos) {
                    target_str = target_str.substr(0, dot);
                }
                NanousdPrim shader = nanousd_primpath(stage, target_str.c_str());
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
                if (strcmp(info_id, "UsdPreviewSurface") == 0 ||
                    strstr(info_id, "ND_UsdPreviewSurface")) {
                    if (best_mdl) nanousd_freeprim(best_mdl);
                    return child; // Prefer UsdPreviewSurface
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
        }
        if (child != best_mdl) nanousd_freeprim(child);
    }

    return best_mdl; // May be nullptr or an MDL shader prim
}

/* ---- Public API ---- */

extern "C" int materials_backend_available(void)
{
    return 1;
}

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

        // Create Metal Shading Language shader generator (Phase 7b).
        // GenMsl reached parity with GenGlsl in MaterialX 1.39.4.
        g_shader_gen = mx::MslShaderGenerator::create();

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

static int material_index_for_target_name(const MaterialCollection* mc,
                                          const char* target)
{
    if (!mc || !target || !*target) return -1;
    for (int i = 0; i < mc->nmaterials; i++) {
        if (strcmp(mc->materials[i].prim_path, target) == 0)
            return i;
    }
    const char* name = strrchr(target, '/');
    name = name ? name + 1 : target;
    for (int i = 0; i < mc->nmaterials; i++) {
        if (strcmp(mc->materials[i].name, name) == 0)
            return i;
    }
    return -1;
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

extern "C" MaterialCollection* materials_load(void* stage_handle,
                                              const char* scene_dir)
{
    NanousdStage stage = (NanousdStage)stage_handle;
    double _materials_load_t0 = nu_load_now_ms();  /* NUSD_LOAD_TIMING */
    g_resolve_cache.clear();  /* per-load resolution memo (bit-identical, never stale) */

    auto* mc = (MaterialCollection*)calloc(1, sizeof(MaterialCollection));
    if (!mc) return nullptr;

    /* materialx_init() (~125 ms) + default-PBR shader compile (~185 ms)
     * are both unconditional setup costs. Defer them until we know the
     * scene actually has materials — see the early-out gate below.
     * (Vulkan port: 97713ec.) */

    int nprims = nanousd_nprims(stage);

    // Collect all Material prims
    struct MatInfo {
        std::string path;
        std::string name;
        NanousdPrim   prim;
    };
    std::vector<MatInfo> mat_prims;

    for (int i = 0; i < nprims; i++) {
        NanousdPrim prim = nanousd_prim(stage, i);
        if (!prim) continue;

        if (nanousd_isactive(prim) && nanousd_isa(prim, "Material")) {
            MatInfo info;
            const char* path = nanousd_path(prim);
            const char* name = nanousd_name(prim);
            info.path = path ? path : "";
            info.name = name ? name : "unnamed";
            info.prim = prim;
            mat_prims.push_back(info);
        } else {
            nanousd_freeprim(prim);
        }
    }

    fprintf(stderr, "material: found %zu Material prims\n", mat_prims.size());

    /* Phase 7c gate (perf): MaterialX init + default-PBR shader compile
     * are skipped entirely when there are no USD Material prims AND no
     * .mtlx files found anywhere reachable (scene_dir + every contributing
     * layer's directory). Saves ~310 ms/load on materials-less scenes
     * (curve grids, geometry-only USDAs, animated xform demos).
     *
     * NUSD_MTLX_SIDELOAD=0 disables the side-loader entirely — useful when
     * scene_dir is a system path (/tmp, /var) that picks up unrelated
     * .mtlx files via the recursive walk. Off by default = enabled.
     *
     * Vulkan port: 97713ec. Metal extension: also probes the stage's
     * contributing layer dirs (the path my earlier change scans for the
     * cross-file-reference chess-set case), so the gate stays correct
     * when scene_dir has no .mtlx but a referenced asset does. */
    bool sideload_enabled = true;
    if (const char* env = getenv("NUSD_MTLX_SIDELOAD"))
        sideload_enabled = (env[0] != '0');
    bool have_mtlx_anywhere = false;
    if (sideload_enabled && mat_prims.empty()) {
        if (scene_dir && *scene_dir) {
            if (!find_mtlx_files(scene_dir).empty()) have_mtlx_anywhere = true;
        }
        if (!have_mtlx_anywhere && stage) {
            int n_layers = nanousd_stage_n_layers(stage);
            for (int i = 0; i < n_layers && !have_mtlx_anywhere; i++) {
                const char* lp = nanousd_stage_layer_path(stage, i);
                if (!lp || !*lp) continue;
                try {
                    std::string dir = fs::path(lp).parent_path().string();
                    if (!dir.empty() && !find_mtlx_files(dir).empty())
                        have_mtlx_anywhere = true;
                } catch (...) {}
            }
        }
        if (!have_mtlx_anywhere) {
            if (const char* extra = getenv("NUSD_MTLX_DIRS")) {
                const char* p = extra;
                while (*p && !have_mtlx_anywhere) {
                    const char* sep = strchr(p, ':');
                    std::string dir = sep ? std::string(p, sep - p)
                                          : std::string(p);
                    if (!dir.empty() && !find_mtlx_files(dir).empty())
                        have_mtlx_anywhere = true;
                    if (!sep) break;
                    p = sep + 1;
                }
            }
        }
    }
    if (mat_prims.empty() && !have_mtlx_anywhere) {
        fprintf(stderr,
            "material: no Material prims, no .mtlx files — "
            "skipping MaterialX init + default-PBR shader compile\n");
        return mc;
    }

    /* Now we know we'll need MaterialX — initialize lazily (no-op if
     * already done). */
    materialx_init();

    // Process each material — with deduplication by name for params/textures
    // but keeping all prim_paths for correct binding lookup.
    std::vector<SceneMaterial> materials;
    std::vector<MaterialTexture> all_textures;
    std::vector<std::string> all_tex_paths;

    // Map material name → index of first occurrence (for copying params)
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

    /* ---- Parallel texture pre-decode (port of vulkan 3918b3b) -------------
     * PNG decode is the largest single CPU cost in material load (~5.2s on the
     * warehouse) and is embarrassingly parallel (pure stbi, no USD/core access).
     * Scan materials + shader descendants for image-asset inputs, resolve them
     * serially (the resolve memo is not thread-safe), then decode the unique set
     * across the thread pool. The readers below are unchanged — load_texture()
     * consumes results from g_predecoded and hits. Bit-identical (same stb
     * decode of the same bytes). NUSD_PARALLEL_DECODE=0 disables. Paths a reader
     * resolves via a different anchor dir (or <UDIM>/package assets) simply miss
     * and decode serially in the walk. */
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
                nu_parallel_for((int)todo.size(), [&](int i) {
                    out[i] = decode_texture_file(todo[i]);
                });
                int okc = 0;
                for (size_t i = 0; i < todo.size(); ++i)
                    if (out[i].pixels) { g_predecoded[todo[i]] = out[i]; ++okc; }
                g_tex_predecode_wall_ms = nu_load_now_ms() - _pd_t0;
                fprintf(stderr,
                        "material: PARALLEL-DECODE %d/%zu textures, %.1f ms wall\n",
                        okc, todo.size(), g_tex_predecode_wall_ms);
            }
        }
    }

    int unique_count = 0;
    for (auto& mi : mat_prims) {
        SceneMaterial sm = {};
        snprintf(sm.name, sizeof(sm.name), "%s", mi.name.c_str());
        snprintf(sm.prim_path, sizeof(sm.prim_path), "%s", mi.path.c_str());

        // Check if a material with this name was already processed
        auto it = mat_name_to_idx.find(mi.name);
        if (it != mat_name_to_idx.end()) {
            // Copy params from the first occurrence
            sm.params = materials[it->second].params;
        } else {
            // New unique material — read properties
            NanousdPrim shader = find_surface_shader(stage, mi.prim);
            ShaderType stype = detect_shader_type(shader);

            switch (stype) {
            case SHADER_OMNIPBR:
            case SHADER_OMNISURFACE:
            case SHADER_MDL_GENERIC:
                if (getenv("NUSD_MAT_DIAG"))
                    fprintf(stderr, "material: '%s' → OmniPBR/MDL reader\n", sm.name);
                read_omnipbr_material(shader, &sm.params,
                                      all_tex_paths, all_textures, scene_dir, stage);
                break;
            case SHADER_OMNIGLASS:
                if (getenv("NUSD_MAT_DIAG"))
                    fprintf(stderr, "material: '%s' → OmniGlass reader\n", sm.name);
                read_omniglass_material(shader, &sm.params,
                                        all_tex_paths, all_textures, scene_dir, stage);
                break;
            case SHADER_USD_PREVIEW_SURFACE:
            default:
                read_usd_preview_surface(shader, &sm.params,
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
            mat_name_to_idx[mi.name] = (int)materials.size();
            unique_count++;
        }

        // Use pre-compiled default PBR shaders
        if (default_vert.ok && default_frag.ok) {
            sm.shader.vert_spv = (uint32_t*)malloc(default_vert.size);
            memcpy(sm.shader.vert_spv, default_vert.data, default_vert.size);
            sm.shader.vert_size = default_vert.size;
            sm.shader.frag_spv = (uint32_t*)malloc(default_frag.size);
            memcpy(sm.shader.frag_spv, default_frag.data, default_frag.size);
            sm.shader.frag_size = default_frag.size;
        }

        materials.push_back(sm);
        nanousd_freeprim(mi.prim);
    }

    /* Free pre-decoded textures the readers did not consume (a reader skipped
     * the input, or resolved it via a different anchor dir and decoded serially
     * in the walk). Consumed entries were erased from g_predecoded; their pixels
     * now live in all_textures. */
    for (auto& kv : g_predecoded)
        if (kv.second.pixels) stbi_image_free(kv.second.pixels);
    g_predecoded.clear();

    free(default_vert.data);
    free(default_frag.data);

    fprintf(stderr, "material: %d unique materials (from %zu total prims)\n",
            unique_count, mat_prims.size());

    /* Phase 7c — fold in any standalone MaterialX .mtlx files under
     * the scene directory. Useful for assets like OpenChessSet whose USD
     * `references = @file.mtlx@` aren't resolvable without a usdMtlx
     * file-format plugin. Adds materials with synthesised
     * /Materials/<name> prim paths; the leaf-name fallback in
     * materials_find_binding resolves bindings from real USD prims. */
    if (sideload_enabled && mat_prims.empty() && scene_dir && *scene_dir) {
        load_mtlx_directory(scene_dir, materials, all_tex_paths,
                            all_textures, mat_name_to_idx);
    }
    /* NUSD_MTLX_DIRS — colon-separated extra roots for the .mtlx scan.
     * Needed when a wrapper USD lives in a directory that doesn't itself
     * contain the asset's .mtlx files (e.g., comparison harnesses that
     * write per-piece variant wrappers separate from the asset tree).
     * Each root is treated like scene_dir: the recursive find walks it
     * for *.mtlx and feeds them through the same parser. */
    if (const char* extra = getenv("NUSD_MTLX_DIRS")) {
        const char* p = extra;
        while (*p) {
            const char* sep = strchr(p, ':');
            std::string dir = sep ? std::string(p, sep - p) : std::string(p);
            if (!dir.empty()) {
                load_mtlx_directory(dir.c_str(), materials, all_tex_paths,
                                    all_textures, mat_name_to_idx);
            }
            if (!sep) break;
            p = sep + 1;
        }
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
            int srgb_votes = mc->textures[t].is_srgb ? 1 : 0;
            int data_votes = 0;
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

    if (getenv("NUSD_LOAD_TIMING")) {
        double _total = nu_load_now_ms() - _materials_load_t0;
        fprintf(stderr,
                "material: LOAD-SPLIT materials_load_total=%.1f ms | "
                "predecode=%.1f ms wall (%ld hits) | decode=%.1f ms (%ld imgs) | "
                "resolve=%.1f ms (%ld miss + %ld memo-hit) | "
                "reader[reads+bakes]=%.1f ms / %ld mats; bake=%.1f ms / %ld bakes; "
                "reads=%.1f ms [+ GPU upload outside]\n",
                _total, g_tex_predecode_wall_ms, g_tex_predecode_n,
                g_tex_decode_ms, g_tex_decode_n,
                g_tex_resolve_ms, g_tex_resolve_n, g_tex_resolve_hits,
                g_reader_ms, g_reader_n, g_bake_ms, g_bake_n,
                g_reader_ms - g_bake_ms);
    }

    if (getenv("NUSD_MAT_DUMP")) {
        int no_diffuse = 0;
        for (int i = 0; i < mc->nmaterials; i++) {
            SceneMaterial& m = mc->materials[i];
            fprintf(stderr, "  mat[%d] '%s' path=%s diff=%d norm=%d rough=%d metal=%d emi=%d occ=%d opa=%d baseC=(%.2f,%.2f,%.2f)\n",
                    i, m.name, m.prim_path,
                    m.params.tex_indices[0], m.params.tex_indices[1],
                    m.params.tex_indices[2], m.params.tex_indices[3],
                    m.params.tex_indices[4], m.params.tex_indices[5],
                    m.params.tex_indices[6],
                    m.params.base_color[0], m.params.base_color[1], m.params.base_color[2]);
            if (m.params.tex_indices[0] < 0) no_diffuse++;
        }
        fprintf(stderr, "material: %d / %d materials have NO diffuse texture\n",
                no_diffuse, mc->nmaterials);
    }

    return mc;
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

static int material_index_for_binding_target(MaterialCollection* mc,
                                             NanousdPrim binding_prim,
                                             const char* target)
{
    if (!mc || !binding_prim || !target || !target[0]) return -1;

    // Exact path match.
    for (int i = 0; i < mc->nmaterials; i++) {
        if (strcmp(mc->materials[i].prim_path, target) == 0)
            return i;
    }

    // Path-remapping fallback: match by material name (last component).
    const char* target_name = strrchr(target, '/');
    if (!target_name) return -1;
    target_name++;  // skip the '/'

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

    // If ancestor matching failed, just match by name.
    for (int i = 0; i < mc->nmaterials; i++) {
        if (strcmp(mc->materials[i].name, target_name) == 0)
            return i;
    }
    return -1;
}

extern "C" int materials_find_binding(MaterialCollection* mc,
                                      void* stage_handle,
                                      void* mesh_prim_handle)
{
    NanousdPrim mesh_prim = (NanousdPrim)mesh_prim_handle;
    NanousdStage stage = (NanousdStage)stage_handle;
    (void)stage;
    if (!mc || !mesh_prim || mc->nmaterials == 0) return -1;

    // 1. Walk up the prim hierarchy looking for material:binding.
    // OVRTX/Hydra and Vulkan resolve USD relationship metadata here:
    // a normal child binding wins by namespace depth, but an ancestor
    // relationship tagged bindMaterialAs="strongerThanDescendants" must
    // override descendant placeholder bindings. Metal keeps the same rule
    // so Linux/Vulkan and macOS/Metal choose the same material.
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
