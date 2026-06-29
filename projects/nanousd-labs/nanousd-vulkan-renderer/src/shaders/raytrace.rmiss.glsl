// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(location = 0) rayPayloadInEXT vec4 hitValue;

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

/* Scene data for envMipLevels check */
struct MeshData {
    uint64_t vertexAddress;
    uint64_t indexAddress;
    vec4     color;
    uint     tex_index;
    uint     source_id;
    uint     material_id;
    uint     ptex_color_offset;
};
layout(set = 0, binding = 2, scalar) buffer SceneData {
    uint     vertexStride;
    uint     hasMaterials;
    float    envMipLevels;
    float    envIntensity;   /* USD DomeLight intensity (sky multiplier) */
    vec4     domeColor;      /* fast_mode flat sky/ambient: rgb + intensity */
    uint     upAxis;         /* 0=X, 1=Y, 2=Z */
    uint     _scenePad0;
    uint     _scenePad1;
    uint     _scenePad2;
    MeshData meshes[];
} scene;

/* IBL environment map (equirectangular) */
layout(set = 0, binding = 5) uniform sampler2D envMap;

/* Depth output SSBO (binding 10): one float per pixel.
 * Miss writes -1.0 (no hit) for primary rays. */
layout(set = 0, binding = 10, std430) writeonly buffer DepthOutput {
    float depths[];
} depthOut;

/* Segmentation output SSBO (binding 11): one uint32 per pixel.
 * Miss writes 0 (no mesh hit) for sky and ground. */
layout(set = 0, binding = 11, std430) writeonly buffer SegmentationOutput {
    uint ids[];
} segOut;

/* Normals output SSBO (binding 12): 3 floats per pixel.
 * Miss writes (0,1,0) for ground, (0,0,0) for sky. */
layout(set = 0, binding = 12, std430) writeonly buffer NormalsOutput {
    float normals[];
} normOut;

/* Scene lights SSBO (binding 13). Miss shading only needs the count so
 * authored-light/no-dome scenes do not inherit the renderer's neutral
 * studio background. */
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

/* Phase A G-buffer SSBO (binding 17): see raytrace.rchit.glsl for the
 * full layout. Miss path writes inst_and_flags = 0 (hit bit clear) for
 * primary rays so the compute consumer in Phase B sees a clean miss. */
struct GBufferEntry {
    uint  inst_and_flags;
    uint  primitive_id;
    uint  bary_packed;
    float hit_t;
    float normal_x;
    float normal_y;
    float normal_z;
    uint  _pad;
};
layout(set = 0, binding = 17, std430) buffer GBuffer {
    GBufferEntry entries[];
} gbuf;

layout(push_constant) uniform PushConstants {
    mat4 view_inv;
    mat4 proj_inv;
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

const float PI = 3.14159265359;

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

/* Shadow test via ray query — returns 0.0 if occluded, 1.0 if lit */
float traceShadow(vec3 origin, vec3 direction, float tmax) {
    if (fast_mode != 0) return 1.0;
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFF, origin, 0.01, direction, tmax);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT)
        return 0.0;
    return 1.0;
}

void main()
{
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    vec3 origin = gl_WorldRayOriginEXT;

    /* Sky palette — consistent across all shaders (matched to raster fallback) */
    const vec3 SKY_HORIZON = vec3(0.95, 0.88, 0.78);
    const vec3 SKY_ZENITH  = vec3(0.70, 0.70, 0.82);
    const vec3 SKY_NADIR   = vec3(0.32, 0.28, 0.22);

    /* Sensor output helpers: primary rays only (hitValue.w < 0.5) */
    uint pixelIdx = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
    bool isPrimary = (hitValue.w < 0.5);
    bool writeMissDepth = (depth_enabled != 0u && isPrimary);
    bool writeMissSeg   = (segmentation_enabled != 0u && isPrimary);
    bool writeMissNorm  = (normals_enabled != 0u && isPrimary);

    /* Phase A G-buffer miss write (primary rays only). Phase B's compute
     * consumer reads inst_and_flags == 0 as the miss/sky/ground signal and
     * branches to its background path.
     *
     * The host sets deferred_shade_enabled only when binding 17 points at a
     * real G-buffer. Single-camera RT keeps it enabled for diagnostics; tiled
     * RT leaves it disabled unless the deferred compute path allocated and
     * rebound the per-launch tiled G-buffer. Never infer this from fast_mode:
     * tiled non-fast renders otherwise write into the scene_data_buf stub. */
    bool gbufWrite = isPrimary && (deferred_shade_enabled != 0u);
    if (gbufWrite) {
        gbuf.entries[pixelIdx].inst_and_flags = 0u;
        gbuf.entries[pixelIdx].primitive_id   = 0u;
        gbuf.entries[pixelIdx].bary_packed    = 0u;
        gbuf.entries[pixelIdx].hit_t          = -1.0;
        gbuf.entries[pixelIdx].normal_x       = 0.0;
        gbuf.entries[pixelIdx].normal_y       = 0.0;
        gbuf.entries[pixelIdx].normal_z       = 0.0;
        gbuf.entries[pixelIdx]._pad           = 0u;
    }

    /* Z-up world: vertical axis is Z; ground plane lives at Z = ground_y
     * (the push-constant name is historical — we now interpret it as the
     * vertical-axis ground for Z-up scenes used by Newton/IsaacLab). */
    if (fast_mode != 0) {
        float dirUp = sceneUpCoord(dir);
        if (dirUp < -0.0001) {
            float t_ground = (ground_y - sceneUpCoord(origin)) / dirUp;
            if (t_ground > 0.0 && t_ground < 500000.0) {
                if (writeMissDepth) depthOut.depths[pixelIdx] = t_ground;
                if (writeMissSeg)   segOut.ids[pixelIdx] = 0u;
                if (writeMissNorm)  {
                    vec3 n = sceneUpVector();
                    uint ni = pixelIdx * 3u;
                    normOut.normals[ni] = n.x;
                    normOut.normals[ni+1u] = n.y;
                    normOut.normals[ni+2u] = n.z;
                }
                hitValue = vec4(vec3(0.45, 0.44, 0.42), 0.0);
                return;
            }
        }
        if (writeMissDepth) depthOut.depths[pixelIdx] = -1.0;
        if (writeMissSeg)   segOut.ids[pixelIdx] = 0u;
        if (writeMissNorm)  { uint ni = pixelIdx * 3u; normOut.normals[ni] = 0.0; normOut.normals[ni+1u] = 0.0; normOut.normals[ni+2u] = 0.0; }
        /* Flat near-white DomeLight-driven sky — matches Newton's 0xEEEEEE
         * clear color (and ovrtx's auto-exposed flat-gray DomeLight) so the
         * Vulkan-vs-Newton background term collapses on RL-vision scenes.
         * Defaults to (0.93,0.93,0.93)*1 from gpu_init() when no scene
         * DomeLight is plumbed; nu_set_dome_color() pushes scene-authored
         * `inputs:color * inputs:intensity` from the IsaacLab adapter. */
        vec3 sky = clamp(scene.domeColor.rgb * scene.domeColor.a, 0.0, 1.0);
        hitValue = vec4(sky, 0.0);
        return;
    }

    /* Sky miss: no geometry hit — write -1.0 depth, 0 segmentation, zero normal */
    if (writeMissDepth) depthOut.depths[pixelIdx] = -1.0;
    if (writeMissSeg)   segOut.ids[pixelIdx] = 0u;
    if (writeMissNorm)  { uint ni = pixelIdx * 3u; normOut.normals[ni] = 0.0; normOut.normals[ni+1u] = 0.0; normOut.normals[ni+2u] = 0.0; }

    /* Sky: use IBL environment map if available, otherwise gradient.
     * USD-authored DomeLight intensity is applied here (not at env load
     * time) so surface lighting in rchit stays normalized via env_scale
     * auto-exposure, while the visible sky follows the high-luminance dome
     * behavior observed through the public ovrtx renderer API. */
    if (scene.envMipLevels > 0.0) {
        vec3 sky = texture(envMap, dirToEquirect(dir)).rgb;
        /* Apply the photographic exposure scale used for the directly
         * visible sky so the HDR backdrop matches public ovrtx output:
         *   exposure = (responsivity * filmIso) /
         *              (100 * shutter * fNumber^2)
         *            = 1.1027 / (100 * 50 * 25)
         *            ≈ 0.000882
         *   pixel = ACES(intensity * exposure * radiance)
         * For chess (intensity=1000, HDR avg ~2.6), this lands the sky
         * at ACES(0.882 * 2.6) ≈ ACES(2.29) ≈ 0.7-0.9 — bright but not
         * saturated, matching ovrtx's photographic look. The previous
         * sqrt(intensity) hack gave 31.6× / saturated white which
         * dominated every nanousdview 3-way capture vs ovrtx.
         * Surface lighting in rchit keeps its own calibrated exposure
         * formula (intensity=1000 → ~0.7×) — this only affects the
         * directly-visible dome. */
        const float kPhotoExposure = 0.000315;
        sky *= abs(scene.envIntensity) * kPhotoExposure * max(tone_sky_scale, 0.0);
        float a2 = 2.51, b2 = 0.03, c2 = 2.43, d2 = 0.59, e2 = 0.14;
        sky = clamp((sky*(a2*sky+b2))/(sky*(c2*sky+d2)+e2), 0.0, 1.0);
        sky = pow(sky, vec3(1.0 / 2.2));
        hitValue = vec4(sky, 0.0);
    } else {
        /* Authored-light scenes without a DomeLight should not get the
         * renderer's no-light studio background; OVRTX shows miss pixels black
         * when nanousdview default lighting is disabled. Keep the neutral
         * background only for genuinely no-light assets. */
        vec3 bg = (sceneLights.nlights > 0) ? vec3(0.0) : vec3(0.39);
        hitValue = vec4(bg, 0.0);
    }
}
