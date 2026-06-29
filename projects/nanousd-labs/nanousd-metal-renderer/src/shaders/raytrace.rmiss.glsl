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
};
layout(set = 0, binding = 2, scalar) buffer SceneData {
    uint     vertexStride;
    uint     hasMaterials;
    float    envMipLevels;
    uint     _pad;
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

layout(push_constant) uniform PushConstants {
    mat4 view_inv;
    mat4 proj_inv;
    float ground_y;
    float scene_scale;
    uint  fast_mode;              /* 1 = skip shadow rays for RL sensors */
    uint  depth_enabled;          /* 1 = write depth to binding 10 SSBO */
    uint  segmentation_enabled;   /* 1 = write instance IDs to binding 11 SSBO */
    uint  normals_enabled;        /* 1 = write normals to binding 12 SSBO */
};

const float PI = 3.14159265359;

vec2 dirToEquirect(vec3 dir) {
    float u = atan(dir.z, dir.x) * (0.5 / PI) + 0.5;
    float v = asin(clamp(dir.y, -1.0, 1.0)) * (1.0 / PI) + 0.5;
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

    /* Fast mode: simple sky gradient, flat ground — no shadow rays or tonemap */
    if (fast_mode != 0) {
        if (dir.y < -0.0001) {
            float t_ground = (ground_y - origin.y) / dir.y;
            if (t_ground > 0.0 && t_ground < 500000.0) {
                if (writeMissDepth) depthOut.depths[pixelIdx] = t_ground;
                if (writeMissSeg)   segOut.ids[pixelIdx] = 0u;
                if (writeMissNorm)  { uint ni = pixelIdx * 3u; normOut.normals[ni] = 0.0; normOut.normals[ni+1u] = 1.0; normOut.normals[ni+2u] = 0.0; }
                hitValue = vec4(vec3(0.45, 0.44, 0.42), 0.0);
                return;
            }
        }
        if (writeMissDepth) depthOut.depths[pixelIdx] = -1.0;
        if (writeMissSeg)   segOut.ids[pixelIdx] = 0u;
        if (writeMissNorm)  { uint ni = pixelIdx * 3u; normOut.normals[ni] = 0.0; normOut.normals[ni+1u] = 0.0; normOut.normals[ni+2u] = 0.0; }
        float t = dir.y * 0.5 + 0.5;
        vec3 sky = (t < 0.5) ? mix(SKY_NADIR, SKY_HORIZON, t * 2.0)
                              : mix(SKY_HORIZON, SKY_ZENITH, (t - 0.5) * 2.0);
        hitValue = vec4(sky, 0.0);
        return;
    }

    /* Infinite ground plane at Y = ground_y */
    if (dir.y < -0.0001) {
        float t_ground = (ground_y - origin.y) / dir.y;
        if (t_ground > 0.0 && t_ground < 500000.0) {
            vec3 hitPos = origin + dir * t_ground;

            /* Checkerboard pattern scaled to scene */
            float scale = max(scene_scale * 0.15, 1.0);
            float cx = floor(hitPos.x / scale);
            float cz = floor(hitPos.z / scale);
            float checker = mod(cx + cz, 2.0);
            vec3 baseGround = mix(vec3(0.42, 0.41, 0.39), vec3(0.52, 0.51, 0.48), checker);

            /* Three-light setup matching closest-hit key/fill/rim */
            vec3 N = vec3(0.0, 1.0, 0.0);
            vec3 keyDir  = normalize(vec3(0.3, 1.0, 0.5));
            vec3 fillDir = normalize(vec3(-0.5, 0.4, -0.3));
            float keyNdotL  = max(dot(N, keyDir), 0.0);
            float fillNdotL = max(dot(N, fillDir), 0.0);

            /* Shadow rays from ground plane */
            float keyShadow  = traceShadow(hitPos + N * 0.1, keyDir, 100000.0);
            float fillShadow = traceShadow(hitPos + N * 0.1, fillDir, 100000.0);

            vec3 groundLight = baseGround * (
                vec3(1.0, 0.95, 0.85) * 1.2 * keyNdotL * keyShadow +
                vec3(0.9, 0.85, 0.75) * 0.40 * fillNdotL * fillShadow +
                vec3(0.35, 0.32, 0.26) * 1.2  /* warm ambient (always present) */
            );

            /* ACES tone map — sRGB swapchain handles gamma */
            float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
            groundLight = clamp((groundLight*(a*groundLight+b))/(groundLight*(c*groundLight+d)+e), 0.0, 1.0);

            /* Fog fade to sky at distance */
            float dist = length(hitPos - origin);
            float fogRange = scene_scale * 6.0;
            float fog = clamp(dist / fogRange, 0.0, 1.0);
            fog = fog * fog;

            groundLight = mix(groundLight, SKY_HORIZON, fog);

            if (writeMissDepth) depthOut.depths[pixelIdx] = t_ground;
            if (writeMissSeg)   segOut.ids[pixelIdx] = 0u;
            if (writeMissNorm)  { uint ni = pixelIdx * 3u; normOut.normals[ni] = 0.0; normOut.normals[ni+1u] = 1.0; normOut.normals[ni+2u] = 0.0; }
            hitValue = vec4(groundLight, 0.0);
            return;
        }
    }

    /* Sky miss: no geometry hit — write -1.0 depth, 0 segmentation, zero normal */
    if (writeMissDepth) depthOut.depths[pixelIdx] = -1.0;
    if (writeMissSeg)   segOut.ids[pixelIdx] = 0u;
    if (writeMissNorm)  { uint ni = pixelIdx * 3u; normOut.normals[ni] = 0.0; normOut.normals[ni+1u] = 0.0; normOut.normals[ni+2u] = 0.0; }

    /* Sky: use IBL environment map if available, otherwise gradient */
    if (scene.envMipLevels > 0.0) {
        vec3 sky = texture(envMap, dirToEquirect(dir)).rgb;
        /* ACES tone map — sRGB swapchain handles gamma */
        float a2 = 2.51, b2 = 0.03, c2 = 2.43, d2 = 0.59, e2 = 0.14;
        sky = clamp((sky*(a2*sky+b2))/(sky*(c2*sky+d2)+e2), 0.0, 1.0);
        hitValue = vec4(sky, 0.0);
    } else {
        float t = dir.y * 0.5 + 0.5;
        vec3 sky;
        if (t < 0.5) {
            sky = mix(SKY_NADIR, SKY_HORIZON, t * 2.0);
        } else {
            sky = mix(SKY_HORIZON, SKY_ZENITH, (t - 0.5) * 2.0);
        }
        hitValue = vec4(sky, 0.0);
    }
}
