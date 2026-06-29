// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require

layout(location = 0) rayPayloadInEXT vec3  hitValue;
layout(location = 1) rayPayloadInEXT vec3  hitWorldPos;
layout(location = 2) rayPayloadInEXT float hitT;

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout(push_constant) uniform PushConstants {
    mat4 view_inv;
    mat4 proj_inv;
    mat4 prev_vp;
    vec2 jitter;
    float ground_y;
    float scene_scale;
};

/* Shadow test via ray query — returns 0.0 if occluded, 1.0 if lit */
float traceShadow(vec3 origin, vec3 direction, float tmax) {
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

    /* Sky palette — consistent across all shaders */
    const vec3 SKY_HORIZON = vec3(0.72, 0.72, 0.70);
    const vec3 SKY_ZENITH  = vec3(0.35, 0.50, 0.80);
    const vec3 SKY_NADIR   = vec3(0.20, 0.18, 0.15);

    /* Infinite ground plane at Y = ground_y */
    if (dir.y < -0.0001) {
        float t_ground = (ground_y - origin.y) / dir.y;
        if (t_ground > 0.0 && t_ground < 500000.0) {
            vec3 hitPos = origin + dir * t_ground;

            /* Checkerboard pattern scaled to scene */
            float sc = max(scene_scale * 0.15, 1.0);
            float cx = floor(hitPos.x / sc);
            float cz = floor(hitPos.z / sc);
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
                vec3(0.7, 0.8,  1.0)  * 0.35 * fillNdotL * fillShadow +
                vec3(0.12, 0.12, 0.14)  /* ambient (always present) */
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

            hitValue = groundLight;
            hitWorldPos = hitPos;
            hitT = t_ground;
            return;
        }
    }

    /* Gradient sky — smooth blend from nadir through horizon to zenith */
    float t = dir.y * 0.5 + 0.5;
    vec3 sky;
    if (t < 0.5) {
        sky = mix(SKY_NADIR, SKY_HORIZON, t * 2.0);
    } else {
        sky = mix(SKY_HORIZON, SKY_ZENITH, (t - 0.5) * 2.0);
    }
    hitValue = sky;
    hitWorldPos = vec3(0.0);
    hitT = -1.0;
}
