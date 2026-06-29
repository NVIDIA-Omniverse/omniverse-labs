// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_ray_query : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 N = normalize(fragNormal);

    /* Dome (hemisphere) ambient — matches sky palette */
    vec3 skyColor    = vec3(0.50, 0.58, 0.75);
    vec3 groundColor = vec3(0.22, 0.20, 0.17);
    float hemi = dot(N, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
    vec3 ambient = mix(groundColor, skyColor, hemi) + vec3(0.08, 0.08, 0.10);

    /* Three-point light setup matching RT closest-hit */
    vec3 keyDir  = normalize(vec3(0.3, 1.0, 0.5));
    vec3 fillDir = normalize(vec3(-0.5, 0.4, -0.3));
    vec3 rimDir  = normalize(vec3(0.0, 0.3, -1.0));

    float keyNdotL  = max(dot(N, keyDir), 0.0);
    float fillNdotL = max(dot(N, fillDir), 0.0);
    float rimNdotL  = max(dot(N, rimDir), 0.0);

    /* Shadow ray for key light */
    float shadow = 1.0;
    if (keyNdotL > 0.0) {
        vec3 origin = fragWorldPos + N * 0.002;
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, tlas,
            gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
            0xFF, origin, 0.001, keyDir, 100000.0);
        while (rayQueryProceedEXT(rq)) {}
        if (rayQueryGetIntersectionTypeEXT(rq, true)
            != gl_RayQueryCommittedIntersectionNoneEXT)
            shadow = 0.0;
    }

    vec3 key  = vec3(1.0, 0.95, 0.85) * 0.7 * keyNdotL * shadow;
    vec3 fill = vec3(0.7, 0.8, 1.0) * 0.2 * fillNdotL;
    vec3 rim  = vec3(1.0, 0.95, 0.9) * 0.15 * rimNdotL;

    vec3 color = fragColor * (ambient + key + fill + rim);

    outColor = vec4(color, 1.0);
}
