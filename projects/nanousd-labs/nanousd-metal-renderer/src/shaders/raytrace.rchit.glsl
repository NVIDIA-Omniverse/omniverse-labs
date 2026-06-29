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
};
layout(set = 0, binding = 2, scalar) buffer SceneData {
    uint     vertexStride;   /* floats per vertex (6 = basic, 12 = material) */
    uint     hasMaterials;   /* 1 if material SSBO is bound */
    float    envMipLevels;   /* 0 = no IBL, >0 = IBL environment mip count */
    uint     _pad;
    MeshData meshes[];
} scene;

/* Material parameters SSBO (binding 3) */
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

layout(push_constant) uniform PushConstants {
    mat4  viewInverse;
    mat4  projInverse;
    float ground_y;
    float scene_scale;
    uint  fast_mode;              /* 1 = skip shadow rays for RL sensors */
    uint  depth_enabled;          /* 1 = write depth to binding 10 SSBO */
    uint  segmentation_enabled;   /* 1 = write instance IDs to binding 11 SSBO */
    uint  normals_enabled;        /* 1 = write normals to binding 12 SSBO */
};

/* Buffer references for vertex/index access */
layout(buffer_reference, scalar) buffer Vertices { float v[]; };
layout(buffer_reference, scalar) buffer Indices  { uint  i[]; };

/* ---- PBR Functions ---- */

const float PI = 3.14159265359;

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

/* Split-sum specular IBL term. The LUT is generated by the renderer from a
 * GGX importance-sampled integration (Karis-style scale+bias). */
vec3 sampleEnvBRDF(vec3 F0, float roughness, float NoV) {
    vec2 brdf = texture(brdfLUT, vec2(clamp(abs(NoV), 0.0, 1.0),
                                      clamp(roughness, 0.0, 1.0))).rg;
    return F0 * brdf.x + brdf.y;
}

/* Convert a 3D direction to equirectangular UV coordinates */
vec2 dirToEquirect(vec3 dir) {
    float u = atan(dir.z, dir.x) * (0.5 / PI) + 0.5;
    float v = asin(clamp(dir.y, -1.0, 1.0)) * (1.0 / PI) + 0.5;
    return vec2(u, 1.0 - v);  /* flip V for top-down convention */
}

/* Shadow test via ray query — returns 0.0 if occluded, 1.0 if lit.
 * In fast_mode, always returns 1.0 (no shadow rays for RL sensors). */
float traceShadow(vec3 origin, vec3 direction, float tmax) {
    if (fast_mode != 0) return 1.0;
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFF, origin, 0.001, direction, tmax);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT)
        return 0.0;
    return 1.0;
}

/* Evaluate one directional light with Cook-Torrance BRDF + shadow */
vec3 evalDirectionalLight(vec3 lightDir, vec3 lightColor, float lightIntensity,
                          vec3 N, vec3 V, vec3 baseColor, float metallic,
                          float roughness, vec3 F0, vec3 worldPos) {
    vec3 L = lightDir;
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    /* Shadow */
    float shadow = traceShadow(worldPos + N * 0.01, L, 100000.0);
    if (shadow <= 0.0) return vec3(0.0);

    /* Cook-Torrance BRDF */
    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3  F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator = D * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    vec3 specular = numerator / denominator;

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    return (kD * baseColor / PI + specular) * lightColor * lightIntensity * NdotL * shadow;
}

/* Evaluate a USD UsdLuxRectLight by sampling its center.
 *   - light.position: rect center
 *   - light.normal:   emit direction (light shines along +normal towards scene)
 *   - light.u_axis, light.v_axis: half-extent vectors (so width=2|u|, height=2|v|)
 *   - normalize=1: intensity is power (W) → radiance L = intensity / (area * π)
 *   - normalize=0: intensity is radiance directly
 * Uses single-sample Monte Carlo at center: dE = L · cosθ_l · cosθ_n · A / d² */
vec3 evalRectLight(GpuLight light, vec3 N, vec3 V, vec3 baseColor, float metallic,
                   float roughness, vec3 F0, vec3 worldPos) {
    vec3 toLight = light.position - worldPos;
    float dist2 = dot(toLight, toLight);
    if (dist2 < 1e-4) return vec3(0.0);
    float dist = sqrt(dist2);
    vec3 L = toLight / dist;

    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return vec3(0.0);

    /* Light only emits along +normal (one-sided): cosθ between -L and lightNormal. */
    float cosLight = dot(-L, light.normal);
    if (cosLight <= 0.0) return vec3(0.0);

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
     * ovrtx renderer's auto-exposure reduces them by ~3-4 stops. This factor
     * approximates that compression for our fixed-exposure tone mapper. */
    radiance *= 0.0012;

    /* Geometric form factor for area-to-point. */
    float geom = (cosLight * area) / max(dist2, 1e-4);

    /* Shadow ray to light center (slightly short to avoid self-hit at light geom). */
    float shadow = traceShadow(worldPos + N * 0.01, L, dist * 0.999);
    if (shadow <= 0.0) return vec3(0.0);

    /* Cook-Torrance */
    vec3 H = normalize(V + L);
    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3  F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (D * G * F) /
                    (4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    return (kD * baseColor / PI + specular) * (light.color * radiance) *
           NdotL * geom * shadow;
}

/* USD UsdLuxDistantLight — sun-like directional. light.normal is emit direction;
 * L = -light.normal. Isaac authors distant lights at ~3000 (cd-style units);
 * scale to match ovrtx render. */
vec3 evalDistantLight(GpuLight light, vec3 N, vec3 V, vec3 baseColor, float metallic,
                      float roughness, vec3 F0, vec3 worldPos) {
    vec3 L = normalize(-light.normal);
    return evalDirectionalLight(L, light.color, light.intensity * 0.0006,
                                N, V, baseColor, metallic, roughness, F0, worldPos);
}

/* USD UsdLuxSphereLight — point/area emitter at light.position with radius
 * stored in light.u_axis.x (see scene.h SCENE_LIGHT_SPHERE). Treated as an
 * isotropic point with 1/r² falloff; when normalize=1 the intensity is
 * interpreted as power and divided by sphere surface area to give radiant
 * exitance. The 0.0006 scale factor mirrors evalDistantLight's empirical
 * cd→shader-radiance match against the ovrtx reference render. */
vec3 evalSphereLight(GpuLight light, vec3 N, vec3 V, vec3 baseColor, float metallic,
                     float roughness, vec3 F0, vec3 worldPos) {
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
    radiance *= 0.0006;

    /* Shadow ray to sphere center, shortened by radius so we don't self-hit
     * the light geometry if the user models it as a real mesh. */
    float shadow = traceShadow(worldPos + N * 0.01, L, max(dist - radius, 0.01));
    if (shadow <= 0.0) return vec3(0.0);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    if (NdotL <= 0.0 || NdotV <= 0.0) return vec3(0.0);

    vec3 H = normalize(V + L);
    float NDF = distributionGGX(N, H, roughness);
    float G   = geometrySmith(N, V, L, roughness);
    vec3  F   = fresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (NDF * G * F) / (4.0 * NdotV * NdotL + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    /* 1/dist² falloff bakes the inverse-square law for point sources. */
    return (kD * baseColor / PI + specular) * light.color *
           radiance * NdotL * shadow / dist2;
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
        vec3 baseColor = scene.meshes[meshIdx].color.rgb;
        vec3 L = normalize(vec3(0.3, 1.0, 0.5));
        float NdotL = max(dot(N, L), 0.0);
        hitValue = vec4(baseColor * (NdotL * 0.8 + 0.2), 0.0);
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
    vec3 color  = meshColor;

    /* Read material ID if available (offset +11 in 12-float stride, stored as uint bits) */
    uint materialId = 0;
    if (scene.hasMaterials != 0 && stride >= 12) {
        uint m0 = floatBitsToUint(verts.v[i0*stride+11]);
        materialId = m0;
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

    if (scene.hasMaterials != 0) {
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

        /* Emissive: color * intensity, with log-scale for extreme USD intensity values */
        vec3 emissiveColor = mat.emissive_color.rgb;
        float emissiveIntensity = mat.emissive_color.a;
        if (emissiveIntensity > 100.0)
            emissiveIntensity = 1.0 + log2(emissiveIntensity / 100.0);
        emissive = emissiveColor * emissiveIntensity;

        /* UDIM UV scaling */
        vec2 tc = uv / vec2(mat.udim_scale_u, mat.udim_scale_v);

        /* Sample diffuse texture if available; base_color acts as tint when
         * present (defaults to white in the loader for textured materials). */
        int diffuse_idx = mat.tex_indices[0]; /* TEX_DIFFUSE_COLOR = 0 */
        if (diffuse_idx >= 0) {
            vec4 tex_color = texture(textures[nonuniformEXT(diffuse_idx)], tc);
            baseColor = tex_color.rgb * mat.base_color.rgb;
            alpha *= tex_color.a;
        }

        /* Sample normal map if available */
        int normal_idx = mat.tex_indices[1]; /* TEX_NORMAL = 1 */
        if (normal_idx >= 0) {
            vec3 nm = texture(textures[nonuniformEXT(normal_idx)], tc).rgb;
            nm = nm * 2.0 - 1.0;
            /* Use a geometric orthonormal basis (Pixar branchless ONB).
             * We tried UV-derived genTangSpace but adjacent tiles with
             * opposite UV winding produce mirrored tangents → alternating
             * bright/dark floor tiles in the warehouse. A geometric basis
             * is consistent across tiles; for isotropic normal maps
             * (concrete, wood, brick) it looks identical to a UV basis. */
            vec3 localT, localB;
            branchlessONB(localNormal, localT, localB);
            vec3 T = normalize(gl_ObjectToWorldEXT * vec4(localT, 0.0));
            vec3 B = normalize(gl_ObjectToWorldEXT * vec4(localB, 0.0));
            normal = normalize(T * nm.x * mat.normal_scale
                             + B * nm.y * mat.normal_scale
                             + normal * nm.z);
        }

        /* Sample roughness texture — use .g channel (ORM: G=Roughness) */
        int rough_idx = mat.tex_indices[2]; /* TEX_ROUGHNESS = 2 */
        if (rough_idx >= 0) {
            roughness = texture(textures[nonuniformEXT(rough_idx)], tc).g;
        }

        /* Sample metallic texture — use .b channel (ORM: B=Metallic) */
        int metal_idx = mat.tex_indices[3]; /* TEX_METALLIC = 3 */
        if (metal_idx >= 0) {
            metallic = texture(textures[nonuniformEXT(metal_idx)], tc).b;
        }

        /* Sample emissive texture if available */
        int emissive_idx = mat.tex_indices[4]; /* TEX_EMISSIVE_COLOR = 4 */
        if (emissive_idx >= 0) {
            vec3 emissive_tex = texture(textures[nonuniformEXT(emissive_idx)], tc).rgb;
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
            ao = texture(textures[nonuniformEXT(ao_idx)], tc).r;
        }
    }

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

    /* ---- Glass / transparent material ---- */
    float rayDepth = hitValue.w;
    if (alpha < 0.5 && rayDepth < 0.5) {
        /* Fresnel reflectance for dielectric glass */
        float cosI = max(dot(V, N), 0.0);
        float F_glass = 0.04 + 0.96 * pow(1.0 - cosI, 5.0);

        /* Trace continuation ray through the glass surface */
        vec3 throughOrigin = worldPos + gl_WorldRayDirectionEXT * 0.01;
        hitValue = vec4(0.0, 0.0, 0.0, 1.0); /* mark as secondary */
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0,
                    throughOrigin, 0.001, gl_WorldRayDirectionEXT, 100000.0, 0);
        vec3 behindColor = hitValue.xyz;

        /* Tint the transmitted light by glass color */
        vec3 transmitted = behindColor * baseColor;

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

        vec3 glassResult = mix(transmitted, envReflect, F_glass);

        /* Tone map glass result — sRGB swapchain handles gamma */
        glassResult = acesFilmic(glassResult * 1.2);

        hitValue = vec4(glassResult, 1.0);
        return;
    }

    /* ---- Opaque PBR lighting ---- */

    /* IOR-driven F0 (matches raster material shader) */
    float f0_dielectric = pow((materialIor - 1.0) / (materialIor + 1.0), 2.0);
    vec3 F0 = mix(vec3(f0_dielectric), baseColor, metallic);

    /* Ambient / IBL */
    vec3 ambient;
    if (scene.envMipLevels > 0.0) {
        /* IBL diffuse: irradiance map lookup (equirectangular) */
        vec3 irradiance = texture(irrMap, dirToEquirect(N)).rgb;
        vec3 kS_ibl = fresnelSchlickRoughness(NdotV, F0, roughness);
        vec3 kD_ibl = (1.0 - kS_ibl) * (1.0 - metallic);
        vec3 diffuseIBL = kD_ibl * irradiance * (baseColor / PI);

        /* IBL specular: pre-filtered env map (mip = roughness * mipCount)
         * weighted by the split-sum BRDF LUT. */
        vec3 R = reflect(-V, N);
        float lod = roughness * (scene.envMipLevels - 1.0);
        vec3 prefilteredColor = textureLod(envMap, dirToEquirect(R), lod).rgb;
        vec3 specularIBL = prefilteredColor * sampleEnvBRDF(F0, roughness, NdotV);

        ambient = (diffuseIBL + specularIBL) * ao;
        /* When the scene has explicit lights, IBL acts as fill only —
         * dominant illumination comes from the ceiling fixtures. But since
         * we have no recursive GI, IBL must supply the bounce light that
         * reaches shelf interiors; keep it small but non-negligible. */
        if (sceneLights.nlights > 0)
            ambient *= 0.15;
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
        float hemi = dot(N, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
        vec3 ambientIrradiance = mix(groundColor, skyColor, hemi);
        ambientIrradiance += vec3(0.10, 0.10, 0.12);
        if (rayDepth > 0.5)
            ambientIrradiance *= 2.0;
        vec3 kS_ambient = fresnelSchlick(NdotV, F0);
        vec3 kD_ambient = (1.0 - kS_ambient) * (1.0 - metallic);
        ambient = kD_ambient * ambientIrradiance * baseColor * ao;
        ambient += kS_ambient * ao * 0.06;

        /* Unmodulated sky fill for upward-facing surfaces. Approximates
         * the indirect sky bounce that ovrtx's path tracer delivers onto
         * floors: our dark floor albedo (~0.05) multiplied by any sky
         * illuminance still reads near-black, so we add a small additive
         * term independent of baseColor. Uses N.y directly (not a reflection
         * vector) so normal-map perturbations don't cause alternating-tile
         * artifacts on glossy floors. */
        float skyUp = clamp(N.y, 0.0, 1.0);
        ambient += vec3(0.05, 0.06, 0.08) * skyUp * ao;
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
                Lo += evalRectLight(L, N, V, baseColor, metallic, roughness, F0, worldPos);
            } else if (L.kind == LIGHT_KIND_DISTANT) {
                Lo += evalDistantLight(L, N, V, baseColor, metallic, roughness, F0, worldPos);
                if (!keyDirValid) {
                    keyDir = normalize(-L.normal);
                    keyDirValid = true;
                }
            } else if (L.kind == LIGHT_KIND_SPHERE) {
                Lo += evalSphereLight(L, N, V, baseColor, metallic, roughness, F0, worldPos);
                if (!keyDirValid) {
                    keyDir = normalize(L.position - worldPos);
                    keyDirValid = true;
                }
            }
        }
    } else {
        /* Synthetic 3-point fallback (used if scene has no lights). */
        vec3 syntheticKey  = normalize(vec3(0.3, 1.0, 0.5));
        vec3 syntheticFill = normalize(vec3(-0.5, 0.4, -0.3));
        vec3 syntheticRim  = normalize(vec3(0.0, 0.3, -1.0));
        float keyScale = (scene.envMipLevels > 0.0) ? 0.5 : 1.8;
        Lo += evalDirectionalLight(syntheticKey, vec3(1.0, 0.95, 0.85), keyScale,
                                   N, V, baseColor, metallic, roughness, F0, worldPos);
        float fillScale = (scene.envMipLevels > 0.0) ? 0.2 : 0.40;
        Lo += baseColor * vec3(0.7, 0.8, 1.0) * fillScale *
              max(dot(N, syntheticFill), 0.0);
        Lo += evalDirectionalLight(syntheticRim, vec3(1.0, 0.95, 0.9), 0.2,
                                   N, V, baseColor, metallic, roughness, F0, worldPos);
        keyDir = syntheticKey;
        keyDirValid = true;
    }

    vec3 result = ambient + Lo + emissive;

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
    float exposure = 0.5;
    result = acesFilmic(result * exposure);
    /* sRGB swapchain handles gamma — no manual pow needed */

    hitValue = vec4(result, 0.0);
}
