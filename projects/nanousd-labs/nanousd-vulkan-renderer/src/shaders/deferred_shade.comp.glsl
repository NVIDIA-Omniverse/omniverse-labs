// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout  : enable
#extension GL_EXT_buffer_reference2    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_ray_tracing          : require
#extension GL_EXT_ray_query            : require

/* Phase C.4 deferred-shading compute pass — direct lighting + inline
 * ray-query shadows. Mode 3 now reaches visual parity with the rchit's
 * opaque PBR path (rchit:1004-1167) by combining IBL ambient (C.3),
 * scene-light direct contribution (rect / distant / sphere), inline
 * ray-query shadow visibility (rchit:281-291), the *=0.15 IBL scaler
 * when scene lights are present (rchit:1067-1068), the
 * useSpecularWorkflow F0 path (rchit:1014-1019), and the SSS body-tint
 * mix (rchit:343-351, etc.).
 *
 * Modes (push constant `deferred_debug_mode`):
 *   0 = base color only (Phase C.1 path, byte-identical to commit cdf2d7f)
 *   1 = world-space shading normal as RGB (debug normals viz — Gate 2)
 *   2 = pbr packed: vec3(metallic, roughness, ao)  (visual debug for C.3)
 *   3 = FULL PBR (C.4): IBL + direct lights + shadows + emissive,
 *       exposure + ACES + UNORM lift. Visual parity with rchit opaque path.
 *
 * Mode 0/1/2 paths are byte-identical to Phase C.2 (Gate 3 contract).
 *
 * Phase C.4 deliberately still does NOT add:
 *   - Clearcoat second specular lobe (rchit:1172-1181) — defer to E
 *   - Synthetic 3-point fallback (rchit:1144-1167) — defer to E
 *   - SSS back-light glow on IBL/sky (rchit:1057-1061, 1097-1102) — defer
 *   - Glass / cutout / alpha-blend (recursion required) — never
 *
 * Bindings — MUST mirror the tiled RT pipeline. C.4 adds 0 (tlas) and
 * 13 (sceneLights) on top of C.3's set (1/2/3/4/5/6/7/8/9/17). See
 * gpu_build_deferred_pipeline() in gpu_vulkan.c. */

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

/* ---------------------------------------------------------------------- */
/* Bindings — must align byte-for-byte with the RT pipeline's binding map */
/* ---------------------------------------------------------------------- */

/* Binding 0 (Phase C.4): TLAS. Used by inline ray-query shadow tests
 * inside the direct-light evaluators. Mirrors rchit's binding 0 exactly
 * (raytrace.rchit.glsl:13). Bound to gpu->tlas (the legacy single TLAS),
 * matching the rchit's traceShadow() call site at rchit:284. The compute
 * pipeline doesn't need the per-env tlas_arr at binding 16 — shadow rays
 * use the same global TLAS the rchit's primary rays committed against. */
layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

/* Binding 2: SceneData. Layout MUST mirror the rchit declaration at
 * raytrace.rchit.glsl:16-27 — scalar layout, MeshData stride is fixed by
 * (2 * uint64 + 1 * vec4) = 32 B, so any drift breaks the meshes[i].color
 * lookup. */
struct MeshData {
    uint64_t vertexAddress;
    uint64_t indexAddress;
    vec4     color;          /* per-mesh display color; .a unused */
    uint     tex_index;      /* runtime texture index (0xFFFFFFFF = none) */
    uint     source_id;
    uint     material_id;    /* per-instance material id */
    uint     ptex_color_offset;
};
layout(set = 0, binding = 2, scalar) readonly buffer SceneData {
    uint     vertexStride;
    uint     hasMaterials;
    float    envMipLevels;
    float    envIntensity;
    vec4     domeColor;
    uint     upAxis;
    uint     _scenePad0;
    uint     _scenePad1;
    uint     _scenePad2;
    MeshData meshes[];
} scene;

/* Binding 3: MaterialBuffer. Layout MUST mirror raytrace.rchit.glsl:34-66
 * byte-for-byte — std430 stride drift would corrupt materials[i] for i>=1.
 * Phase C.2 reads many more fields than C.1 (metallic/roughness/occlusion/
 * emissive/opacity/sss/transmission/normal_scale/normal_tex_scale/bias,
 * tex_subsurface_weight, tex_transmission_weight) — the FULL struct must
 * match GpuMaterialParams in src/gpu.h. */
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
    /* Phase 7c trailing block — present in GpuMaterialParams. */
    vec4  subsurface_color;
    vec4  subsurface_radius;
    vec4  transmission_color;
    float subsurface_weight;
    float subsurface_scale;
    float transmission_weight;
    float transmission_ior;
    int   tex_subsurface_weight;     /* SEPARATE field — NOT in tex_indices[] */
    int   tex_transmission_weight;   /* SEPARATE field — NOT in tex_indices[] */
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

layout(set = 0, binding = 3, std430) readonly buffer MaterialBuffer {
    MaterialData materials[];
} mat_buf;

/* Binding 4: Texture array (combined image samplers). Phase C.2 samples
 * tex_indices[0..6] (TEX_DIFFUSE_COLOR..TEX_OPACITY) plus the standalone
 * tex_subsurface_weight / tex_transmission_weight indices. */
layout(set = 0, binding = 4) uniform sampler2D textures[];

/* Bindings 5-7 (Phase C.3): IBL — equirectangular environment,
 * pre-integrated split-sum BRDF LUT, and equirectangular irradiance map. The
 * descriptor-set layout binds these only when `gpu->env_image_view != NULL`
 * at pipeline-build time (gpu_build_deferred_pipeline in gpu_vulkan.c).
 * The shader gates every sampler use on `scene.envMipLevels > 0.0` so
 * unbound declarations never reach a fetch instruction at runtime. */
layout(set = 0, binding = 5) uniform sampler2D envMap;
layout(set = 0, binding = 6) uniform sampler2D brdfLUT;
layout(set = 0, binding = 7) uniform sampler2D irrMap;

/* Binding 8 (Phase C.3): Camera SSBO. cameras[cam_idx*2+0] = view_inverse,
 * cameras[cam_idx*2+1] = proj_inverse. We need this in mode 3 to
 * reconstruct the per-pixel ray direction so we can compute V (view dir)
 * for the specular reflection R = reflect(-V, N). The G-buffer is fixed
 * at 32 B and doesn't carry a per-pixel ray direction — recomputing from
 * the camera matrices uses the same math as raytrace_tiled.rgen.glsl
 * lines 140-145.
 *
 * Layout is identical to the rgen's binding 8 declaration (rgen lines
 * 51-53). Only mode 3 dereferences this; modes 0/1/2 don't read it. */
layout(set = 0, binding = 8) readonly buffer CameraSSBO {
    mat4 cameras[];
};

/* Binding 1: tiled storage image — used when the rgen writes via imageStore
 * (useDirectOut == 0). The compute pass writes the same target so the Phase B
 * output flows through whichever readback path is active. */
layout(set = 0, binding = 1, rgba8) uniform image2D outputImage;

/* Binding 9: DirectOutput — the same SSBO raytrace_tiled.rgen.glsl writes
 * when useDirectOut != 0. We overwrite the per-pixel uint32 the rgen
 * already produced. */
layout(set = 0, binding = 9, std430) writeonly buffer DirectOutput {
    uint pixels[];
} directOut;

/* Binding 13 (Phase C.4): scene lights SSBO. Layout MUST mirror
 * raytrace.rchit.glsl:106-123 byte-for-byte — std430 packs each
 * vec3+scalar pair into 16 bytes (5 pairs = 80 bytes per light). The
 * descriptor write falls back to tiled_camera_buf when no lights are
 * uploaded; the shader gates the dispatch loop on `nlights > 0` so the
 * fallback case never reads past the 16-byte header. */
#define LIGHT_KIND_RECT     0
#define LIGHT_KIND_DISTANT  1
#define LIGHT_KIND_SPHERE   2

const vec3 OVRTX_FALLBACK_KEY_DIR_DEF = normalize(vec3(-0.29883624, 0.70710678, 0.64085638));
const vec3 OVRTX_FALLBACK_KEY_COLOR_DEF = vec3(1.0, 0.95, 0.9);
const float OVRTX_FALLBACK_KEY_INTENSITY_DEF = 3500.0 * 0.0006;

struct GpuLightDef {
    vec3  position;   float intensity;
    vec3  normal;     int   kind;
    vec3  u_axis;     int   normalize;
    vec3  v_axis;     float angle_deg;
    vec3  color;      float _pad;
};

layout(set = 0, binding = 13, std430) readonly buffer LightsBuffer {
    int          nlights;
    int          _lpad[3];
    GpuLightDef  items[];
} sceneLights;

/* Binding 17: G-buffer entries written by rchit (hit) + rmiss (primary
 * miss). Layout matches raytrace.rchit.glsl:131-146 byte-for-byte
 * (32 bytes). */
struct GBufferEntry {
    uint  inst_and_flags;  /* bit 31: hit; bits 0-23: gl_InstanceCustomIndexEXT */
    uint  primitive_id;
    uint  bary_packed;
    float hit_t;
    float normal_x;
    float normal_y;
    float normal_z;
    uint  _pad;
};
layout(set = 0, binding = 17, std430) readonly buffer GBuffer {
    GBufferEntry entries[];
} gbuf;

/* Buffer references for vertex/index access — mirror raytrace.rchit.glsl:161-162. */
layout(buffer_reference, scalar) buffer Vertices { float v[]; };
layout(buffer_reference, scalar) buffer Indices  { uint  i[]; };

/* ---------------------------------------------------------------------- */
/* Push constants — same struct GpuRtTiledPushConstants the RT pipeline    */
/* pushes. We re-decode the tile params from the first 16 bytes of         */
/* viewInverse and the layout flags from viewInverse[1] so the compute     */
/* dispatch's pixel<->camera mapping matches the rgen exactly.             */
/*                                                                         */
/* Phase C.2 adds `deferred_debug_mode` decoded from viewInverse[1][3]     */
/* (offset 28 in the host struct: GpuRtTiledPushConstants._pad[3]). The    */
/* rgen does not consume this slot so the change is safe.                  */
/* ---------------------------------------------------------------------- */
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

/* sRGB encode — identical to raytrace_tiled.rgen.glsl:75-84. */
vec3 linear_to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    vec3 lo = 12.92 * c;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    bvec3 use_hi = greaterThan(c, vec3(0.0031308));
    return vec3(use_hi.x ? hi.x : lo.x,
                use_hi.y ? hi.y : lo.y,
                use_hi.z ? hi.z : lo.z);
}

/* Pixar branchless ONB (Duff et al.). Builds a robust orthonormal basis
 * around an arbitrary unit vector. Verbatim algorithm of rchit:199-205,
 * re-implemented (per workspace policy: read for the algorithm, write our
 * own line-by-line — feedback_match_not_copy applies even within-repo). */
void branchless_onb(vec3 n, out vec3 b1, out vec3 b2) {
    float sgn = n.z >= 0.0 ? 1.0 : -1.0;
    float a   = -1.0 / (sgn + n.z);
    float b   = n.x * n.y * a;
    b1 = vec3(1.0 + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
    b2 = vec3(b, sgn + n.y * n.y * a, -n.y);
}

void gen_tangent_space(vec2 uv0, vec2 uv1, vec2 uv2,
                       vec3 p0, vec3 p1, vec3 p2,
                       vec3 N, out vec3 T, out vec3 B) {
    vec2 duv02 = uv0 - uv2;
    vec2 duv12 = uv1 - uv2;
    vec3 dp02 = p0 - p2;
    vec3 dp12 = p1 - p2;
    float det = duv02.x * duv12.y - duv02.y * duv12.x;
    if (abs(det) >= 1e-8) {
        float inv = 1.0 / det;
        T = ( duv12.y * dp02 - duv02.y * dp12) * inv;
        B = (-duv12.x * dp02 + duv02.x * dp12) * inv;
    } else {
        branchless_onb(N, T, B);
        return;
    }
    T = normalize(T - N * dot(N, T));
    vec3 nu = cross(N, T);
    float sgn = dot(nu, B) < 0.0 ? -1.0 : 1.0;
    B = normalize(sgn * nu);
}

/* ---------------------------------------------------------------------- */
/* Phase C.3 IBL helpers — algorithms ported from raytrace.rchit.glsl     */
/* (re-implemented; per feedback_match_not_copy: read for the math, write */
/* our own).                                                              */
/* ---------------------------------------------------------------------- */

const float DEF_PI = 3.14159265359;

/* Roughness-aware Schlick Fresnel — mirrors rchit:193-195. Used by the
 * IBL diffuse-cancel term (kS_ibl) so the rough-surface diffuse lobe
 * doesn't dim toward zero at grazing angles. */
vec3 fresnelSchlickRoughness_def(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0)
              * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

/* Split-sum specular IBL term. brdfLUT is generated by gpu_vulkan.c from
 * a GGX importance-sampled integration, matching the OpenGL raster path. */
vec3 sampleEnvBRDF_def(vec3 F0, float roughness, float NoV) {
    vec2 brdf = texture(brdfLUT, vec2(clamp(abs(NoV), 0.0, 1.0),
                                      clamp(roughness, 0.0, 1.0))).rg;
    return F0 * brdf.x + brdf.y;
}

/* Equirectangular projection matching Hydra / OVRTX DomeLight for these
 * assets: +Z is the north pole and +Y is zero-longitude forward. The CPU
 * GGX prefilter / SH projection path uses the same convention. */
vec2 dirToEquirect_def(vec3 dir) {
    const float kDomeLongitudeOffset = 340.0 / 360.0;
    float u = fract(atan(dir.x, dir.y) * (0.5 / DEF_PI) + 0.5 + kDomeLongitudeOffset);
    float v = asin(clamp(dir.z, -1.0, 1.0)) * (1.0 / DEF_PI) + 0.5;
    return vec2(u, 1.0 - v);
}

/* ACES filmic tone mapping (Krzysztof Narkowicz fitted curve).
 * Mirrors rchit:509-516. */
vec3 acesFilmic_def(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

/* ---------------------------------------------------------------------- */
/* Phase C.4 PBR helpers — Cook-Torrance GGX + Schlick Fresnel.            */
/* Algorithms ported from rchit:166-205; re-implemented per                */
/* feedback_match_not_copy. Used by the direct-light evaluators below.    */
/* ---------------------------------------------------------------------- */

/* Trowbridge-Reitz GGX normal distribution. Mirror rchit:168-175. */
float distributionGGX_def(vec3 N, vec3 H, float roughness) {
    float a    = roughness * roughness;
    float a2   = a * a;
    float NdH  = max(dot(N, H), 0.0);
    float NdH2 = NdH * NdH;
    float denom = NdH2 * (a2 - 1.0) + 1.0;
    return a2 / (DEF_PI * denom * denom + 0.0001);
}

/* Smith GGX shadowing-masking, separable form. Mirror rchit:177-187. */
float geometrySchlickGGX_def(float NdV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdV / (NdV * (1.0 - k) + k);
}

float geometrySmith_def(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdV = max(dot(N, V), 0.0);
    float NdL = max(dot(N, L), 0.0);
    return geometrySchlickGGX_def(NdV, roughness)
         * geometrySchlickGGX_def(NdL, roughness);
}

/* Schlick Fresnel — sharp form (no roughness gate). Mirror rchit:189-191. */
vec3 fresnelSchlick_def(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

/* SSS wrap term — shifts the cosine so light "wraps" around the silhouette.
 * Mirror rchit:304-306. With wrapAmount == 0 collapses to clamped NdotL. */
float computeWrappedNdotL_def(vec3 N, vec3 L, float wrapAmount) {
    return clamp((dot(N, L) + wrapAmount) / (1.0 + wrapAmount), 0.0, 1.0);
}

/* Inline shadow ray query — returns 0.0 if any opaque hit blocks the
 * light, 1.0 otherwise. Mirror rchit:281-291. The cull mask 0xFF matches
 * the rchit's traceShadow() exactly; the per-env tlas_arr at binding 16
 * is NOT consulted here — shadows ride the global tlas just like the
 * rchit does. The caller passes the fast_mode flag (push constant) since
 * GLSL only allows one layout(push_constant) block per shader; we can't
 * forward-reference fields from a not-yet-declared block. */
float traceShadow_def(vec3 origin, vec3 direction, float tmax, uint fastMode) {
    if (fastMode != 0u) return 1.0;
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFFu, origin, 0.001, direction, tmax);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true)
        != gl_RayQueryCommittedIntersectionNoneEXT)
        return 0.0;
    return 1.0;
}

float contactVisibility_def(vec3 worldPos, vec3 N, uint fastMode) {
    if (fastMode != 0u) return 1.0;
    vec3 helper = (abs(N.y) < 0.95) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(helper, N));
    vec3 B = cross(N, T);
    float aoLen = clamp(scene_scale * 0.040, 0.016, 0.080);
    vec3 origin = worldPos + N * max(scene_scale * 0.0008, 0.0005);
    float v = 0.0;
    v += traceShadow_def(origin, N, aoLen * 0.55, fastMode);
    v += traceShadow_def(origin, normalize(N * 0.62 + T * 0.60), aoLen, fastMode);
    v += traceShadow_def(origin, normalize(N * 0.62 - T * 0.60), aoLen, fastMode);
    v += traceShadow_def(origin, normalize(N * 0.62 + B * 0.60), aoLen, fastMode);
    v += traceShadow_def(origin, normalize(N * 0.62 - B * 0.60), aoLen, fastMode);
    return v * 0.2;
}

/* ---------------------------------------------------------------------- */
/* PbrMaterial — packed per-pixel material inputs C.3/C.4 consume.         */
/* C.4 extends the struct with the fields the direct-light evaluators     */
/* need: material IOR, the useSpecularWorkflow flag + specular F0, and    */
/* the loader-resolved SSS body tint.                                     */
/* ---------------------------------------------------------------------- */
struct PbrMaterial {
    vec3  base_color;            /* same as C.1 albedo */
    vec3  shading_normal;        /* world-space, post normal-map */
    vec3  geom_normal;           /* world-space geometric (G-buffer normal) */
    float metallic;
    float roughness;             /* clamped [0.04, 1.0] */
    float ao;
    vec3  emissive;
    float opacity;
    float sss_weight;
    float transmission_weight;
    /* Phase C.4 additions. */
    float ior;                   /* material IOR (default 1.5) */
    int   use_specular_workflow; /* 1 = UPS specular workflow */
    vec3  specular_color;        /* UPS specular workflow F0 (when set) */
    vec3  sss_color;             /* loader-resolved body tint */
    float sss_wrap;              /* clamp(maxRadius * 100.0, 0.0, 0.5) */
};

/* ---------------------------------------------------------------------- */
/* Phase C.4 light evaluators — port the three rchit eval functions       */
/* (rect / distant / sphere). Each returns the BRDF*L*NdotL contribution  */
/* of a single light, with inline shadow ray. Algorithms mirror the rchit */
/* but the GLSL is re-implemented per feedback_match_not_copy.            */
/* ---------------------------------------------------------------------- */

/* Cook-Torrance evaluator core — diffuse + specular for a unit light dir
 * already known. Used by all three light types after they compute L,
 * radiance, and shadow. Wrapped NdotL handles SSS body-tint. */
vec3 evalCookTorranceCore_def(vec3 L, vec3 N, vec3 V, vec3 baseColor,
                               float metallic, float roughness, vec3 F0,
                               float sssWeight, vec3 sssColor, float wrapAmount) {
    vec3  H        = normalize(V + L);
    float NdotL    = max(dot(N, L), 0.0);
    float wrappedN = (sssWeight > 0.0)
                   ? computeWrappedNdotL_def(N, L, wrapAmount)
                   : NdotL;
    if (wrappedN <= 0.0) return vec3(0.0);

    /* Specular uses sharp NdotL — keeps highlights tight even with SSS. */
    vec3 specular = vec3(0.0);
    if (NdotL > 0.0) {
        float D = distributionGGX_def(N, H, roughness);
        float G = geometrySmith_def(N, V, L, roughness);
        vec3  F = fresnelSchlickRoughness_def(max(dot(H, V), 0.0), F0, roughness);
        float denom = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
        specular = (D * G * F) / denom;
    }

    vec3 kD = (vec3(1.0)
            - fresnelSchlickRoughness_def(max(dot(H, V), 0.0), F0, roughness))
           * (1.0 - metallic);
    vec3 diffuseAlbedo = baseColor;
    if (sssWeight > 0.0) {
        diffuseAlbedo = mix(baseColor, sssColor, sssWeight);
    }
    vec3 diffuse = kD * diffuseAlbedo / DEF_PI * wrappedN;
    return diffuse + specular * NdotL;
}

float areaLightSpecularRoughness_def(float roughness, float angularDiameter) {
    float a = clamp(angularDiameter * 2.5, 0.0, 1.0);
    return clamp(sqrt(roughness * roughness + a * a), roughness, 1.0);
}

vec3 evalRectLightSample_def(GpuLightDef light, PbrMaterial pm,
                              vec3 samplePos, float radiance, float area,
                              vec3 N, vec3 V, vec3 F0, vec3 worldPos,
                              uint fastMode) {
    vec3  toLight = samplePos - worldPos;
    float dist2   = dot(toLight, toLight);
    if (dist2 < 1e-4) return vec3(0.0);
    float dist = sqrt(dist2);
    vec3  L    = toLight / dist;

    float NdotL    = dot(N, L);
    float wrappedN = (pm.sss_weight > 0.0)
                   ? computeWrappedNdotL_def(N, L, pm.sss_wrap)
                   : max(NdotL, 0.0);
    if (wrappedN <= 0.0) return vec3(0.0);

    /* One-sided emission: cosθ between -L and lightNormal. */
    float cosLight = dot(-L, light.normal);
    if (cosLight <= 0.0) return vec3(0.0);

    float geom = (cosLight * area) / max(dist2, 1e-4);

    float shadow = traceShadow_def(worldPos + N * 0.01, L, dist * 0.999, fastMode);
    if (shadow <= 0.0) return vec3(0.0);

    /* The rect is sampled as a few point lights; widen the specular lobe by
     * emitter angular diameter so rough floors do not get point-light glints. */
    float rectAngular = clamp(2.0 * max(length(light.u_axis), length(light.v_axis)) /
                              max(dist, 1e-3), 0.0, 1.0);
    float specRoughness = areaLightSpecularRoughness_def(pm.roughness, rectAngular);
    vec3 brdf = evalCookTorranceCore_def(L, N, V, pm.base_color,
                                          pm.metallic, specRoughness, F0,
                                          pm.sss_weight, pm.sss_color, pm.sss_wrap);
    return brdf * (light.color * radiance) * geom * shadow;
}

/* USD UsdLuxRectLight — deterministic 3x3 Gauss-Legendre quadrature.
 * Mirrors the rchit evaluator so tiled/deferred RT sees the same partial
 * occlusion from warehouse shelves and nearby box stacks. */
vec3 evalRectLight_def(GpuLightDef light, PbrMaterial pm,
                       vec3 N, vec3 V, vec3 F0, vec3 worldPos,
                       uint fastMode) {
    float area = 4.0 * length(light.u_axis) * length(light.v_axis);
    float radiance;
    if (light.normalize != 0) {
        radiance = light.intensity / max(area * DEF_PI, 1e-6);
    } else {
        radiance = light.intensity / DEF_PI;
    }
    /* Same empirical exposure scale the rchit uses (rchit:403). */
    radiance *= 0.0012;

    const float q = 0.7745966692;
    const float wc = 0.1975308642;
    const float we = 0.1234567901;
    const float wk = 0.0771604938;
    vec3 accum = vec3(0.0);
    vec3 center = light.position;
    vec3 U = light.u_axis;
    vec3 W = light.v_axis;
    accum += evalRectLightSample_def(light, pm, center, radiance, area,
                                     N, V, F0, worldPos, fastMode) * wc;
    accum += evalRectLightSample_def(light, pm, center + U * q, radiance, area,
                                     N, V, F0, worldPos, fastMode) * we;
    accum += evalRectLightSample_def(light, pm, center - U * q, radiance, area,
                                     N, V, F0, worldPos, fastMode) * we;
    accum += evalRectLightSample_def(light, pm, center + W * q, radiance, area,
                                     N, V, F0, worldPos, fastMode) * we;
    accum += evalRectLightSample_def(light, pm, center - W * q, radiance, area,
                                     N, V, F0, worldPos, fastMode) * we;
    accum += evalRectLightSample_def(light, pm, center + U * q + W * q,
                                     radiance, area, N, V, F0, worldPos,
                                     fastMode) * wk;
    accum += evalRectLightSample_def(light, pm, center + U * q - W * q,
                                     radiance, area, N, V, F0, worldPos,
                                     fastMode) * wk;
    accum += evalRectLightSample_def(light, pm, center - U * q + W * q,
                                     radiance, area, N, V, F0, worldPos,
                                     fastMode) * wk;
    accum += evalRectLightSample_def(light, pm, center - U * q - W * q,
                                     radiance, area, N, V, F0, worldPos,
                                     fastMode) * wk;
    return accum;
}

/* USD UsdLuxDistantLight — sun-like directional. Matches the authored
 * DistantLight scale in raytrace.rchit.glsl. */
vec3 evalDistantLight_def(GpuLightDef light, PbrMaterial pm,
                          vec3 N, vec3 V, vec3 F0, vec3 worldPos,
                          uint fastMode) {
    vec3  L = normalize(-light.normal);
    float NdotL    = max(dot(N, L), 0.0);
    float wrappedN = (pm.sss_weight > 0.0)
                   ? computeWrappedNdotL_def(N, L, pm.sss_wrap)
                   : NdotL;
    if (wrappedN <= 0.0) return vec3(0.0);

    /* Shadow first — long ray to "infinity". rchit:328 uses tmax=100000. */
    float shadow = traceShadow_def(worldPos + N * 0.01, L, 100000.0, fastMode);
    if (shadow <= 0.0) return vec3(0.0);

    float distantIntensity = light.intensity * 0.0025;
    vec3 brdf = evalCookTorranceCore_def(L, N, V, pm.base_color,
                                          pm.metallic, pm.roughness, F0,
                                          pm.sss_weight, pm.sss_color, pm.sss_wrap);
    return brdf * (light.color * distantIntensity) * shadow;
}

/* USD UsdLuxSphereLight — point/area emitter at light.position with
 * radius in light.u_axis.x. Mirror rchit:455-506. */
vec3 evalSphereLight_def(GpuLightDef light, PbrMaterial pm,
                         vec3 N, vec3 V, vec3 F0, vec3 worldPos,
                         uint fastMode) {
    vec3  toLight = light.position - worldPos;
    float dist2   = max(dot(toLight, toLight), 1e-4);
    float dist    = sqrt(dist2);
    vec3  L       = toLight / dist;

    float radius = max(light.u_axis.x, 1e-4);
    float area   = 4.0 * DEF_PI * radius * radius;
    float radiance;
    if (light.normalize != 0) {
        radiance = light.intensity / area;
    } else {
        radiance = light.intensity;
    }
    /* Keep deferred SphereLight energy aligned with mesh/RT shaders. */
    radiance *= 0.00017;

    /* Shorten by radius so we don't self-hit user-modeled light geom. */
    float shadow = traceShadow_def(worldPos + N * 0.01, L,
                                    max(dist - radius, 0.01), fastMode);
    if (shadow <= 0.0) return vec3(0.0);

    float NdotL    = max(dot(N, L), 0.0);
    float NdotV    = max(dot(N, V), 0.0);
    float wrappedN = (pm.sss_weight > 0.0)
                   ? computeWrappedNdotL_def(N, L, pm.sss_wrap)
                   : NdotL;
    if (wrappedN <= 0.0 || NdotV <= 0.0) return vec3(0.0);

    float sphereAngular = clamp(2.0 * radius / max(dist, 1e-3), 0.0, 1.0);
    float specRoughness = areaLightSpecularRoughness_def(pm.roughness, sphereAngular);
    vec3 brdf = evalCookTorranceCore_def(L, N, V, pm.base_color,
                                          pm.metallic, specRoughness, F0,
                                          pm.sss_weight, pm.sss_color, pm.sss_wrap);
    /* 1/dist² inverse-square fall-off bakes the point-source convention. */
    return brdf * light.color * radiance * shadow / dist2;
}

void main()
{
    /* Decode tile params (mirror rgen lines 91-94). */
    uint tile_w      = floatBitsToUint(viewInverse[0][0]);
    uint tile_h      = floatBitsToUint(viewInverse[0][1]);
    uint num_cols    = floatBitsToUint(viewInverse[0][2]);
    uint num_cameras = floatBitsToUint(viewInverse[0][3]);

    uint useDirectOut    = floatBitsToUint(viewInverse[1][0]);
    uint usePerEnvLayout = floatBitsToUint(viewInverse[1][1]);
    uint useSrgb         = floatBitsToUint(viewInverse[1][2]);
    /* Phase C.2 — runtime debug-mode selector, packed at offset 28.
     * Default 0 → preserves Phase C.1 behaviour bit-for-bit. */
    uint debug_mode      = floatBitsToUint(viewInverse[1][3]);

    /* NU_TILE_RES output stride (see raytrace_tiled.rgen.glsl). The host
     * populates these with the UNSCALED tile dims when NU_TILE_RES != 64
     * so the per-env stride matches the caller-allocated output buffer.
     * Defaults to tile_w/tile_h when zero (legacy host) for byte-identical
     * behaviour. */
    uint out_tile_w = floatBitsToUint(viewInverse[2][0]);
    uint out_tile_h = floatBitsToUint(viewInverse[2][1]);
    if (out_tile_w == 0u) out_tile_w = tile_w;
    if (out_tile_h == 0u) out_tile_h = tile_h;

    /* Diagnostic linear-radiance probe gain (host packs NU_DEFERRED_LINEAR_PROBE
     * here). >0 makes the full-lit path emit pre-tonemap radiance * gain instead
     * of the ACES-tonemapped color, for light-unit calibration vs ovrtx HdrColor. */
    float linear_probe_gain = viewInverse[2][2];

    /* Pixel coord. Dispatch is (W/16, H/16, 1) where W/H = tiled image
     * dimensions, matching the rgen launch grid 1:1. */
    uvec2 pix = gl_GlobalInvocationID.xy;
    uint  W   = num_cols * tile_w;
    uint  num_rows = (num_cameras + num_cols - 1u) / num_cols;
    uint  H   = num_rows * tile_h;
    if (pix.x >= W || pix.y >= H) return;

    /* pixel -> tile -> cam_idx (mirror rgen lines 107-114). */
    uint tile_x  = pix.x / tile_w;
    uint tile_y  = pix.y / tile_h;
    uint cam_idx = tile_y * num_cols + tile_x;
    uint local_x = pix.x % tile_w;
    uint local_y = pix.y % tile_h;

    /* Output offset — IDENTICAL to rgen lines 119-121 / 243-246. */
    uint pixelIdx_tiled = pix.y * W + pix.x;
    uint ofs = (usePerEnvLayout != 0u)
        ? cam_idx * out_tile_w * out_tile_h + local_y * out_tile_w + local_x
        : pixelIdx_tiled;

    /* Out-of-range tiles: same sentinel the rgen writes. The shaders
     * never invoked CHS/RMISS for these pixels, so the G-buffer slot is
     * untouched. Mirror the sentinel for byte-identical output. */
    if (cam_idx >= num_cameras) {
        if (useDirectOut != 0u) {
            if (usePerEnvLayout != 0u) {
                /* Per-env-layout block-fill (mirrors raytrace_tiled.rgen.glsl). */
                uint x_lo = (local_x * out_tile_w) / tile_w;
                uint x_hi = ((local_x + 1u) * out_tile_w) / tile_w;
                uint y_lo = (local_y * out_tile_h) / tile_h;
                uint y_hi = ((local_y + 1u) * out_tile_h) / tile_h;
                uint env_base = cam_idx * out_tile_w * out_tile_h;
                for (uint y = y_lo; y < y_hi; ++y) {
                    uint row = y * out_tile_w;
                    for (uint x = x_lo; x < x_hi; ++x) {
                        directOut.pixels[env_base + row + x] = 0xFF000000u;
                    }
                }
            } else {
                directOut.pixels[ofs] = 0xFF000000u;
            }
        } else {
            imageStore(outputImage, ivec2(pix), vec4(0.0, 0.0, 0.0, 1.0));
        }
        return;
    }

    /* G-buffer is indexed by the rgen's tiled launch coord (the rchit
     * line 616 uses gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + .x), NOT by
     * the per-env `ofs`. */
    GBufferEntry g = gbuf.entries[pixelIdx_tiled];

    bool hit  = (g.inst_and_flags & 0x80000000u) != 0u;
    uint inst = (g.inst_and_flags & 0x00FFFFFFu);

    vec3 outRgb;

    /* ================================================================== */
    /* Mode 0 — Phase C.1 byte-identical path. DO NOT reorder ops here.  */
    /* Mode 3 (full IBL-lit) splits off after C.3; do not lump them here. */
    /* ================================================================== */
    if (debug_mode == 0u) {
        if (!hit) {
            outRgb = vec3(0.0);
        } else {
            vec3 baseColor = scene.meshes[inst].color.rgb;

            uint stride = scene.vertexStride;
            uint materialId = scene.meshes[inst].material_id;
            if (scene.hasMaterials != 0u && materialId != 0xFFFFFFFFu && stride >= 12u) {
                Vertices verts = Vertices(scene.meshes[inst].vertexAddress);
                Indices  idxs  = Indices(scene.meshes[inst].indexAddress);

                uint base = g.primitive_id * 3u;
                uint i0 = idxs.i[base + 0u];
                uint i1 = idxs.i[base + 1u];
                uint i2 = idxs.i[base + 2u];

                vec2 baryCoord = unpackUnorm2x16(g.bary_packed);
                vec3 bary = vec3(1.0 - baryCoord.x - baryCoord.y,
                                 baryCoord.x, baryCoord.y);

                MaterialData mat = mat_buf.materials[materialId];

                if (mat.use_vertex_color == 0) {
                    baseColor = mat.base_color.rgb;
                }

                vec2 uv0 = vec2(verts.v[i0 * stride + 9u], verts.v[i0 * stride + 10u]);
                vec2 uv1 = vec2(verts.v[i1 * stride + 9u], verts.v[i1 * stride + 10u]);
                vec2 uv2 = vec2(verts.v[i2 * stride + 9u], verts.v[i2 * stride + 10u]);
                vec2 uv  = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

                vec2 tc = uv / vec2(mat.udim_scale_u, mat.udim_scale_v);
                tc = tc * mat.mdl_uv_transform.xy + mat.mdl_uv_transform.zw;
                if (mat.v_flip != 0) tc.y = 1.0 - tc.y;

                int diffuse_idx = mat.tex_indices[0];
                if (diffuse_idx >= 0) {
                    vec4 tex_color = texture(textures[nonuniformEXT(diffuse_idx)], tc);
                    baseColor = tex_color.rgb * mat.base_color.rgb;
                }
            }
            outRgb = baseColor;
        }

        if (useSrgb != 0u) outRgb = linear_to_srgb(outRgb);

        if (useDirectOut != 0u) {
            uvec4 c = uvec4(clamp(vec4(outRgb, 1.0), 0.0, 1.0) * 255.0 + 0.5);
            uint packed = c.r | (c.g << 8u) | (c.b << 16u) | (c.a << 24u);
            if (usePerEnvLayout != 0u) {
                /* Per-env-layout block-fill (mirrors raytrace_tiled.rgen.glsl). */
                uint x_lo = (local_x * out_tile_w) / tile_w;
                uint x_hi = ((local_x + 1u) * out_tile_w) / tile_w;
                uint y_lo = (local_y * out_tile_h) / tile_h;
                uint y_hi = ((local_y + 1u) * out_tile_h) / tile_h;
                uint env_base = cam_idx * out_tile_w * out_tile_h;
                for (uint y = y_lo; y < y_hi; ++y) {
                    uint row = y * out_tile_w;
                    for (uint x = x_lo; x < x_hi; ++x) {
                        directOut.pixels[env_base + row + x] = packed;
                    }
                }
            } else {
                directOut.pixels[ofs] = packed;
            }
        } else {
            imageStore(outputImage, ivec2(pix), vec4(outRgb, 1.0));
        }
        return;
    }

    /* ================================================================== */
    /* Modes 1/2/3 — Phase C.2 PBR sampling, then debug viz (1/2) or full */
    /* IBL-lit color (3, Phase C.3).                                      */
    /* ================================================================== */
    PbrMaterial pm;
    pm.base_color            = vec3(0.0);
    pm.shading_normal        = vec3(0.0, 1.0, 0.0);
    pm.geom_normal           = vec3(0.0, 1.0, 0.0);
    pm.metallic              = 0.0;
    pm.roughness             = 0.5;
    pm.ao                    = 1.0;
    pm.emissive              = vec3(0.0);
    pm.opacity               = 1.0;
    pm.sss_weight            = 0.0;
    pm.transmission_weight   = 0.0;
    /* Phase C.4 defaults — match rchit:672, 691-693, 680-681. */
    pm.ior                   = 1.5;
    pm.use_specular_workflow = 0;
    pm.specular_color        = vec3(0.0);
    pm.sss_color             = vec3(1.0);
    pm.sss_wrap              = 0.0;

    if (!hit) {
        /* Miss handling.
         *
         * Modes 1/2 (debug viz): write black — the debug visualizations
         *   target hit pixels only. Same as C.2.
         * Mode 3 (IBL-lit): reproduce the rmiss sky so the visible sky
         *   behind unhit geometry stays consistent with the OFF reference.
         *   The compute pass overwrites binding 9 wholesale, so we MUST
         *   write the sky here or it'd be clobbered to black. Mirror
         *   raytrace.rmiss.glsl:231-254. */
        if (debug_mode == 3u && scene.envMipLevels > 0.0) {
            /* Reconstruct the per-pixel ray direction the rgen used so
             * we can sample the env map. Same math as the rgen; same
             * math as below for hit-pixel V derivation. */
            vec2 localPixel = vec2(local_x, local_y) + vec2(0.5);
            vec2 inUV       = localPixel / vec2(float(tile_w), float(tile_h));
            vec2 d_ndc      = inUV * 2.0 - 1.0;
            mat4 camViewInv = cameras[cam_idx * 2u + 0u];
            mat4 camProjInv = cameras[cam_idx * 2u + 1u];
            vec4 target_h   = vec4(d_ndc.x, d_ndc.y, 1.0, 1.0) * camProjInv;
            vec4 dir_h      = vec4(normalize(target_h.xyz), 0.0) * camViewInv;
            vec3 ray_dir    = normalize(dir_h.xyz);

            /* Match the miss shader's directly-visible dome path: apply
             * calibrated photographic exposure, then ACES-compress. */
            vec3 sky = texture(envMap, dirToEquirect_def(ray_dir)).rgb;
            const float kPhotoExposure = 0.000315;
            sky *= abs(scene.envIntensity) * kPhotoExposure * max(tone_sky_scale, 0.0);
            sky  = acesFilmic_def(sky);
            sky  = pow(sky, vec3(1.0 / 2.2));
            outRgb = sky;
        } else {
            outRgb = vec3(0.0);
        }
    } else {
        /* Geometric world normal — pre-flipped by the rchit at line 632
         * to face the incoming ray. We use it directly. */
        vec3 N_world = vec3(g.normal_x, g.normal_y, g.normal_z);
        float n_len  = length(N_world);
        if (n_len > 1e-6) N_world = N_world / n_len;
        else              N_world = vec3(0.0, 1.0, 0.0);
        pm.geom_normal     = N_world;
        pm.shading_normal  = N_world;

        vec3 baseColor = scene.meshes[inst].color.rgb;

        uint stride = scene.vertexStride;
        uint materialId  = scene.meshes[inst].material_id;
        if (scene.hasMaterials != 0u && materialId != 0xFFFFFFFFu && stride >= 12u) {
            /* Reconstruct triangle vertex data from G-buffer prim+bary. */
            Vertices verts = Vertices(scene.meshes[inst].vertexAddress);
            Indices  idxs  = Indices(scene.meshes[inst].indexAddress);

            uint base = g.primitive_id * 3u;
            uint i0 = idxs.i[base + 0u];
            uint i1 = idxs.i[base + 1u];
            uint i2 = idxs.i[base + 2u];

            vec2 baryCoord = unpackUnorm2x16(g.bary_packed);
            vec3 bary = vec3(1.0 - baryCoord.x - baryCoord.y,
                             baryCoord.x, baryCoord.y);

            MaterialData mat = mat_buf.materials[materialId];

            /* Material defaults — mirror rchit:666-693. */
            if (mat.use_vertex_color == 0) {
                baseColor = mat.base_color.rgb;
            }
            float metallic   = mat.metallic;
            float roughness  = mat.roughness;
            float ao         = mat.occlusion;
            float opacity    = mat.base_color.a * mat.opacity;
            vec3  emissive   = vec3(0.0);
            /* Phase C.4 — capture material IOR + UPS specular workflow flags
             * before texture sampling can override metallic/roughness. */
            float matIor      = mat.ior;
            int   uspecWf     = mat.use_specular_workflow;
            vec3  specColor   = mat.specular_color.rgb;

            /* Emissive intensity log-scale clamp — mirror rchit:711-716. */
            vec3  emissiveColorMat = mat.emissive_color.rgb;
            float emissiveIntensity = mat.emissive_color.a;
            if (emissiveIntensity > 100.0)
                emissiveIntensity = 1.0 + log2(emissiveIntensity / 100.0);
            emissive = emissiveColorMat * emissiveIntensity;

            vec3 p0 = vec3(verts.v[i0 * stride + 0u], verts.v[i0 * stride + 1u], verts.v[i0 * stride + 2u]);
            vec3 p1 = vec3(verts.v[i1 * stride + 0u], verts.v[i1 * stride + 1u], verts.v[i1 * stride + 2u]);
            vec3 p2 = vec3(verts.v[i2 * stride + 0u], verts.v[i2 * stride + 1u], verts.v[i2 * stride + 2u]);

            /* UV interpolation — mirror rchit:617-620. */
            vec2 uv0 = vec2(verts.v[i0 * stride + 9u], verts.v[i0 * stride + 10u]);
            vec2 uv1 = vec2(verts.v[i1 * stride + 9u], verts.v[i1 * stride + 10u]);
            vec2 uv2 = vec2(verts.v[i2 * stride + 9u], verts.v[i2 * stride + 10u]);
            vec2 uv  = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

            vec2 tc = uv / vec2(mat.udim_scale_u, mat.udim_scale_v);
            tc = tc * mat.mdl_uv_transform.xy + mat.mdl_uv_transform.zw;
            if (mat.v_flip != 0) tc.y = 1.0 - tc.y;

            /* TEX_DIFFUSE_COLOR — mirror rchit:723-728. Drives baseColor +
             * applies UPS opacity from the diffuse alpha channel. */
            int diffuse_idx = mat.tex_indices[0];
            if (diffuse_idx >= 0) {
                vec4 tex_color = texture(textures[nonuniformEXT(diffuse_idx)], tc);
                baseColor = tex_color.rgb * mat.base_color.rgb;
                opacity  *= tex_color.a;
            }

            /* TEX_OPACITY — mirror rchit:735-743. Use .r unless it's
             * constant-1.0 (UNORM-clamped to 1.0), then fall back to .a. */
            int opacity_idx = mat.tex_indices[6];
            if (opacity_idx >= 0) {
                vec4 opa_color = texture(textures[nonuniformEXT(opacity_idx)], tc);
                float opa_sample = (opa_color.r < 1.0) ? opa_color.r : opa_color.a;
                opacity *= opa_sample;
            }

            /* TEX_NORMAL — mirror rchit:753-770. Apply the UsdUVTexture
             * scale/bias (defaults reproduce the historic nm*2-1 mapping),
             * then build a UV-derived tangent basis. The deferred pass does
             * not have the object-to-world matrix in its G-buffer, but for
             * current unskinned USD meshes this object/local basis tracks the
             * rchit result much more closely than a purely geometric ONB. */
            int normal_idx = mat.tex_indices[1];
            if (normal_idx >= 0) {
                vec3 nm_raw = texture(textures[nonuniformEXT(normal_idx)], tc).rgb;
                vec3 nm     = nm_raw * mat.normal_tex_scale.rgb
                            + mat.normal_tex_bias.rgb;
                vec3 T, B;
                gen_tangent_space(uv0, uv1, uv2, p0, p1, p2, N_world, T, B);
                vec3 N_shade = normalize(T * nm.x * mat.normal_scale
                                       + B * nm.y * mat.normal_scale
                                       + N_world * nm.z);
                pm.shading_normal = N_shade;
            }

            /* TEX_ROUGHNESS — .g channel, mirror rchit:773-776. */
            int rough_idx = mat.tex_indices[2];
            if (rough_idx >= 0) {
                float roughSample = texture(textures[nonuniformEXT(rough_idx)], tc).g;
                float roughScale = (mat.roughness_tex_scale != 0.0) ? mat.roughness_tex_scale : 1.0;
                roughness = roughSample * roughScale + mat.roughness_tex_bias;
            }

            /* TEX_METALLIC — .b channel, mirror rchit:779-782. */
            int metal_idx = mat.tex_indices[3];
            if (metal_idx >= 0) {
                metallic = texture(textures[nonuniformEXT(metal_idx)], tc).b;
            }

            /* TEX_EMISSIVE — black-override + multiply, mirror rchit:785-794. */
            int emissive_idx = mat.tex_indices[4];
            if (emissive_idx >= 0) {
                vec3 emissive_tex = texture(textures[nonuniformEXT(emissive_idx)], tc).rgb;
                if (dot(emissiveColorMat, emissiveColorMat) < 0.001) {
                    emissive = emissive_tex * emissiveIntensity;
                } else {
                    emissive *= emissive_tex;
                }
            }

            /* TEX_OCCLUSION — .r channel, mirror rchit:797-800. */
            int ao_idx = mat.tex_indices[5];
            if (ao_idx >= 0) {
                ao = texture(textures[nonuniformEXT(ao_idx)], tc).r;
            }

            /* SSS weight — guarded so we don't waste a sample for the
             * 99% of materials with subsurface_weight = 0. Mirror
             * rchit:812-816. */
            float sssWeight = mat.subsurface_weight;
            if (sssWeight > 0.0) {
                int sss_idx = mat.tex_subsurface_weight;
                if (sss_idx >= 0) {
                    sssWeight *= texture(textures[nonuniformEXT(sss_idx)], tc).r;
                }
            }
            /* Phase C.4 — SSS body tint + wrap amount.
             * sssColor uses the loader's authored flag (mirror rchit:824-826):
             *   sss_color_authored != 0 → mat.subsurface_color.rgb
             *   else                    → baseColor (collapses SSS mix to no-op
             *                              for materials that didn't author it).
             * sssWrap = clamp(maxRadius * 100.0, 0, 0.5) — mirror rchit:827-829. */
            vec3  sssColorVal = (mat.sss_color_authored != 0)
                              ? mat.subsurface_color.rgb
                              : baseColor;
            vec3  sssRadius = mat.subsurface_radius.rgb * mat.subsurface_scale;
            float sssMaxR   = max(max(sssRadius.r, sssRadius.g), sssRadius.b);
            float sssWrapVal = clamp(sssMaxR * 100.0, 0.0, 0.5);

            /* Transmission weight — guarded for the same reason. Mirror
             * rchit:838-842. */
            float transWeight = mat.transmission_weight;
            if (transWeight > 0.0) {
                int trans_idx = mat.tex_transmission_weight;
                if (trans_idx >= 0) {
                    transWeight *= texture(textures[nonuniformEXT(trans_idx)], tc).r;
                }
            }

            /* Roughness clamp — mirror rchit:848. */
            roughness = clamp(roughness, 0.04, 1.0);

            pm.base_color            = baseColor;
            pm.metallic              = metallic;
            pm.roughness             = roughness;
            pm.ao                    = ao;
            pm.emissive              = emissive;
            pm.opacity               = opacity;
            pm.sss_weight            = sssWeight;
            pm.transmission_weight   = transWeight;
            pm.ior                   = matIor;
            pm.use_specular_workflow = uspecWf;
            pm.specular_color        = specColor;
            pm.sss_color             = sssColorVal;
            pm.sss_wrap              = sssWrapVal;
        } else {
            /* No material — fall back to per-mesh display color, identical
             * to the C.1 fallback path. */
            pm.base_color = baseColor;
            pm.opacity    = 1.0;
        }

        /* Debug visualization (modes 1/2) or full IBL-lit color (mode 3). */
        if (debug_mode == 1u) {
            /* World-space shading normal as RGB: 0.5 * (N + 1).
             * Visualizes normal-map TBN reconstruction. Without a normal
             * texture this displays the geometric normal — six distinct
             * faces on a cube, smooth gradients on a sphere. */
            outRgb = pm.shading_normal * 0.5 + 0.5;
        } else if (debug_mode == 2u) {
            /* PBR-packed visualization. metallic in R, roughness in G,
             * ao in B. */
            outRgb = vec3(pm.metallic, pm.roughness, pm.ao);
        } else {
            /* Mode 3 — Phase C.4 full PBR. IBL + direct lights + inline
             * shadow ray queries + emissive, then the same exposure +
             * ACES + UNORM-floor-lift tonemap as the rchit at lines
             * 1183-1229. Glass / cutout / clearcoat / synthetic 3-point
             * deferred per the C.4 brief. */

            /* ----- Per-pixel ray direction ---------------------------- */
            /* Reconstruct exactly as the rgen (raytrace_tiled.rgen.glsl
             * lines 134-145). ray_dir points from the camera into the
             * pixel; V (view direction at the surface) = -ray_dir. */
            vec2 localPixel = vec2(local_x, local_y) + vec2(0.5);
            vec2 inUV       = localPixel / vec2(float(tile_w), float(tile_h));
            vec2 d_ndc      = inUV * 2.0 - 1.0;
            mat4 camViewInv = cameras[cam_idx * 2u + 0u];
            mat4 camProjInv = cameras[cam_idx * 2u + 1u];
            vec4 target_h   = vec4(d_ndc.x, d_ndc.y, 1.0, 1.0) * camProjInv;
            vec4 dir_h      = vec4(normalize(target_h.xyz), 0.0) * camViewInv;
            vec3 ray_dir    = normalize(dir_h.xyz);
            vec3 V          = -ray_dir;
            vec3 N          = pm.shading_normal;
            float NdotV     = clamp(dot(N, V), 0.0, 1.0);

            /* ----- World position recompute (Phase C.4) -------------- */
            /* Required for shadow ray origins and rect/sphere light
             * distance terms. The G-buffer carries hit_t; ray_origin is
             * recovered the same way the rgen does (row-vector convention,
             * see raytrace_tiled.rgen.glsl:143):
             *   origin = vec4(0,0,0,1) * camViewInverse
             * which extracts the translation row of camViewInverse. */
            vec4 origin_h   = vec4(0.0, 0.0, 0.0, 1.0) * camViewInv;
            vec3 ray_origin = origin_h.xyz;
            vec3 worldPos   = ray_origin + ray_dir * g.hit_t;

            /* ----- F0 derivation (mirror rchit:1004-1019) ------------- */
            /* IOR-driven dielectric F0 + UPS specular-workflow override.
             * useSpecularWorkflow=1 forces metallic=0 and uses
             * specular_color as F0 directly — matches rchit:1014-1018. */
            float f0_dielectric = pow((pm.ior - 1.0) / (pm.ior + 1.0), 2.0);
            vec3  F0;
            float metallicEff = pm.metallic;
            if (pm.use_specular_workflow != 0) {
                metallicEff = 0.0;
                F0 = pm.specular_color;
            } else {
                F0 = mix(vec3(f0_dielectric), pm.base_color, pm.metallic);
            }
            bool manySceneLightsWithIbl = (scene.envMipLevels > 0.0 &&
                                           sceneLights.nlights > 8);
            bool authoredKeyOnlyNoIbl = (scene.envMipLevels == 0.0 &&
                                         sceneLights.nlights > 0 &&
                                         sceneLights.nlights <= 8);

            /* ----- Ambient / IBL (mirror rchit:1021-1071) ------------- */
            vec3 ambient = vec3(0.0);
            if (scene.envMipLevels > 0.0) {
                /* Diffuse IBL: irradiance lookup at N. SSS body-tint mix
                 * here (rchit:1033-1036) — for non-SSS materials the mix
                 * collapses to baseColor since sss_weight == 0 short-
                 * circuits the alternate branch. */
                vec3 irradiance = texture(irrMap, dirToEquirect_def(N)).rgb;
                vec3 kS_ibl = fresnelSchlickRoughness_def(NdotV, F0, pm.roughness);
                vec3 kD_ibl = (1.0 - kS_ibl) * (1.0 - metallicEff);
                vec3 diffuseAlbedoIBL = (pm.sss_weight > 0.0)
                    ? mix(pm.base_color, pm.sss_color, pm.sss_weight)
                    : pm.base_color;
                vec3 diffuseIBL = kD_ibl * irradiance * (diffuseAlbedoIBL / DEF_PI);
                float upFacing = smoothstep(0.0, 0.85, clamp(N.z, 0.0, 1.0));
                float albedoLum = dot(pm.base_color, vec3(0.2126, 0.7152, 0.0722));
                float darkVertical = mix(0.55, 1.0, smoothstep(0.10, 0.55, albedoLum));
                float diffuseShape = mix(darkVertical, 1.20, upFacing);
                float specularShape = mix(0.35, 1.0, upFacing);
                diffuseIBL *= diffuseShape;

                /* Specular IBL: pre-filtered env at R, mip = roughness *
                 * (mips - 1), weighted by the split-sum BRDF LUT. */
                vec3  R    = reflect(-V, N);
                float lod  = pm.roughness * (scene.envMipLevels - 1.0);
                vec3  prefiltered = textureLod(envMap, dirToEquirect_def(R), lod).rgb;
                vec3  specularIBL = prefiltered * sampleEnvBRDF_def(F0, pm.roughness, NdotV);
                specularIBL *= specularShape;
                if (manySceneLightsWithIbl) {
                    /* Large Isaac-style interiors use practical fixtures plus
                     * a flat DomeLight. Match OVRTX by lowering ceiling fill. */
                    float highInterior = smoothstep(1.0, 3.0, worldPos.z);
                    float lowFloor = smoothstep(0.15, 0.85, clamp(N.z, 0.0, 1.0)) *
                        (1.0 - smoothstep(0.05, 0.50, worldPos.z));
                    float downwardFace = smoothstep(0.12, 0.72, clamp(-N.z, 0.0, 1.0));
                    float verticalFace = 1.0 - smoothstep(0.18, 0.90, abs(N.z));
                    float ceilingFace = highInterior * downwardFace * (1.0 - lowFloor);
                    float upperVertical = highInterior * verticalFace * (1.0 - lowFloor);
                    diffuseIBL *= mix(1.0, 0.30, max(ceilingFace, upperVertical * 0.18));
                    specularIBL *= mix(1.0, 0.42, ceilingFace);
                }

                float contact = mix(0.68, 1.0,
                    contactVisibility_def(worldPos, N, fast_mode));
                ambient = (diffuseIBL + specularIBL) * pm.ao * contact;

                /* Phase C.4 IBL fill scaler — when scene lights are
                 * present, IBL acts as fill only (rchit:1067-1068). */
                if (sceneLights.nlights > 0)
                    ambient *= max(rt_ibl_fill_scale, 0.0);
            } else {
                /* No-IBL fallback: hemisphere ambient (mirror rchit:1072-
                 * 1113). Z-up sky/ground tints. */
                vec3 skyColor    = vec3(0.60, 0.68, 0.82);
                vec3 groundColor = vec3(0.32, 0.31, 0.30);
                float hemi  = dot(N, vec3(0.0, 0.0, 1.0)) * 0.5 + 0.5;
                vec3 hemiAmbient = mix(groundColor, skyColor, hemi);
                /* Diffuse-only fallback — no specular without an env map.
                 * Mirror rchit:1086-1093 (Lambertian diffuse + tiny
                 * specular fill). SSS body-tint mix matches rchit:1088-1091. */
                vec3 kS_ambient = fresnelSchlick_def(NdotV, F0);
                vec3 kD_ambient = (1.0 - kS_ambient) * (1.0 - metallicEff);
                vec3 diffuseAlbedoAmb = (pm.sss_weight > 0.0)
                    ? mix(pm.base_color, pm.sss_color, pm.sss_weight)
                    : pm.base_color;
                ambient = kD_ambient * hemiAmbient * diffuseAlbedoAmb * pm.ao;
                ambient += kS_ambient * pm.ao * 0.06;

                /* Sky-fill for upward-facing surfaces (mirror rchit:1112-
                 * 1113) so dark albedos still show against light. */
                bool manySceneLightsNoIbl = (scene.envMipLevels == 0.0 &&
                                             sceneLights.nlights > 8);
                float skyUp = clamp(N.z, 0.0, 1.0);
                float skyBounce = manySceneLightsNoIbl ? 2.6 : 1.0;
                ambient += vec3(0.05, 0.06, 0.08) * skyUp * pm.ao * skyBounce;
                if (sceneLights.nlights > 0) {
                    ambient *= manySceneLightsNoIbl ? vec3(0.72, 0.78, 0.90)
                                                    : vec3(0.32, 0.34, 0.38);
                }
                if (manySceneLightsNoIbl) {
                    float lowSceneHorizontal = smoothstep(0.15, 0.85, skyUp) *
                        (1.0 - smoothstep(0.05, 0.45, worldPos.z));
                    ambient *= mix(0.36, 1.0, lowSceneHorizontal);
                    ambient += vec3(0.035, 0.055, 0.080) * lowSceneHorizontal *
                               pm.ao * (1.0 - metallicEff);
                }
            }
            if (authoredKeyOnlyNoIbl) {
                ambient = vec3(0.0);
            }

            /* ----- Direct lighting (Phase C.4) ----------------------- */
            /* Mirror rchit:1116-1167. Each light type dispatches to its
             * own evaluator, which evaluates BSDF, traces an inline
             * shadow ray, and accumulates. Use a thin wrapper material
             * struct that already carries IOR/SSS/etc. to avoid passing
             * five extra parameters down. */
            PbrMaterial pmDirect = pm;
            pmDirect.metallic = metallicEff;  /* useSpecularWorkflow path */
            vec3 Lo = vec3(0.0);
            if (sceneLights.nlights > 0) {
                int n = sceneLights.nlights;
                for (int li = 0; li < n; ++li) {
                    GpuLightDef Ll = sceneLights.items[li];
                    if (Ll.kind == LIGHT_KIND_RECT) {
                        Lo += evalRectLight_def(Ll, pmDirect, N, V, F0,
                                                worldPos, fast_mode);
                    } else if (Ll.kind == LIGHT_KIND_DISTANT) {
                        Lo += evalDistantLight_def(Ll, pmDirect, N, V, F0,
                                                   worldPos, fast_mode);
                    } else if (Ll.kind == LIGHT_KIND_SPHERE) {
                        Lo += evalSphereLight_def(Ll, pmDirect, N, V, F0,
                                                  worldPos, fast_mode);
                    }
                }
            }
            if (sceneLights.nlights == 0 && scene.envMipLevels > 0.0 &&
                scene.envIntensity < -1.0) {
                float shadow = traceShadow_def(worldPos + N * 0.01,
                                               OVRTX_FALLBACK_KEY_DIR_DEF,
                                               100000.0, fast_mode);
                if (shadow > 0.0) {
                    vec3 brdf = evalCookTorranceCore_def(
                        OVRTX_FALLBACK_KEY_DIR_DEF, N, V, pmDirect.base_color,
                        pmDirect.metallic, pmDirect.roughness, F0,
                        pmDirect.sss_weight, pmDirect.sss_color,
                        pmDirect.sss_wrap);
                    Lo += brdf * OVRTX_FALLBACK_KEY_COLOR_DEF *
                          OVRTX_FALLBACK_KEY_INTENSITY_DEF * shadow;
                }
            }

            /* ----- Combine + emissive (mirror rchit:1169) ------------- */
            vec3 result = ambient + Lo + pm.emissive;
            if (manySceneLightsWithIbl) {
                float highInterior = smoothstep(1.0, 3.0, worldPos.z);
                float lowFloor = smoothstep(0.15, 0.85, clamp(N.z, 0.0, 1.0)) *
                    (1.0 - smoothstep(0.05, 0.50, worldPos.z));
                float downwardFace = smoothstep(0.12, 0.72, clamp(-N.z, 0.0, 1.0));
                float verticalFace = 1.0 - smoothstep(0.18, 0.90, abs(N.z));
                float ceilingFace = highInterior * downwardFace * (1.0 - lowFloor);
                float upperVertical = highInterior * verticalFace * (1.0 - lowFloor);
                float interiorScale = mix(1.0, 0.36, ceilingFace) *
                                      mix(1.0, 0.84, upperVertical);
                vec3 floorLift = mix(vec3(1.0), vec3(3.14, 3.28, 3.42), lowFloor);
                result = (ambient + Lo) * interiorScale * floorLift + pm.emissive;
            }
            if (scene.envMipLevels == 0.0 && sceneLights.nlights > 8) {
                float floorShape = smoothstep(0.15, 0.85, clamp(N.z, 0.0, 1.0)) *
                    (1.0 - smoothstep(0.05, 0.45, worldPos.z));
                result *= mix(0.70, 1.0, floorShape);
            }

            /* ----- Exposure + ACES + UNORM lift (mirror rchit:1183-1229) */
            float intensity = (scene.envIntensity < -1.0) ? 1.0 : abs(scene.envIntensity);
            float t1   = clamp(log(1.0 + intensity) / log(1001.0), 0.0, 1.25);
            float gate = smoothstep(1.0, 2.0, intensity);
            float boost1 = 1.0 + 0.4 * t1 * gate;
            float boost2 = pow(max(intensity / 1000.0, 1.0), 0.85);
            float exposure = 0.5 * boost1 * boost2;
            if (scene.envIntensity < -1.0)
                exposure *= 0.92;
            if (scene.envMipLevels == 0.0 && sceneLights.nlights > 8)
                exposure *= 1.7;
            if (manySceneLightsWithIbl)
                exposure *= 0.45;
            bool hadAnyLight = result.r > 0.0 || result.g > 0.0 || result.b > 0.0;
            if (linear_probe_gain > 0.0) {
                /* Diagnostic: emit pre-tonemap LINEAR radiance (no exposure/ACES/
                 * UNORM-lift) so callers can read true radiance for light-unit
                 * calibration against ovrtx's linear HdrColor AOV. */
                outRgb = clamp(result * linear_probe_gain, vec3(0.0), vec3(1.0));
            } else {
                result = acesFilmic_def(result * exposure * max(tone_exposure_scale, 0.0));
                if (hadAnyLight) {
                    result = max(result, vec3(2.0 / 255.0));
                }
                outRgb = result;
            }
        }
    }

    /* Keep the linear-radiance probe linear (no sRGB transfer). */
    if (useSrgb != 0u && linear_probe_gain <= 0.0) outRgb = linear_to_srgb(outRgb);

    if (useDirectOut != 0u) {
        uvec4 c = uvec4(clamp(vec4(outRgb, 1.0), 0.0, 1.0) * 255.0 + 0.5);
        uint packed = c.r | (c.g << 8u) | (c.b << 16u) | (c.a << 24u);
        if (usePerEnvLayout != 0u) {
            /* NU_TILE_RES block-fill (mirrors raytrace_tiled.rgen.glsl). */
            uint scale_x = out_tile_w / tile_w;
            uint scale_y = out_tile_h / tile_h;
            if (scale_x == 0u) scale_x = 1u;
            if (scale_y == 0u) scale_y = 1u;
            uint base_x = local_x * scale_x;
            uint base_y = local_y * scale_y;
            uint env_base = cam_idx * out_tile_w * out_tile_h;
            for (uint dy = 0u; dy < scale_y; ++dy) {
                uint row = (base_y + dy) * out_tile_w;
                for (uint dx = 0u; dx < scale_x; ++dx) {
                    directOut.pixels[env_base + row + base_x + dx] = packed;
                }
            }
        } else {
            directOut.pixels[ofs] = packed;
        }
    } else {
        imageStore(outputImage, ivec2(pix), vec4(outRgb, 1.0));
    }
}
