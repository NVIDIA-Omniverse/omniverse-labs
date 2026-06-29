// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 450

layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);

    /* Dome (hemisphere) ambient — matches sky palette */
    vec3 skyColor    = vec3(0.50, 0.58, 0.75);
    vec3 groundColor = vec3(0.22, 0.20, 0.17);
    float hemi = dot(N, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
    vec3 ambient = mix(groundColor, skyColor, hemi) + vec3(0.08, 0.08, 0.10);

    /* Three-point light setup matching RT closest-hit */
    /* Key light — warm directional (sun) */
    vec3 keyDir = normalize(vec3(0.3, 1.0, 0.5));
    float keyNdotL = max(dot(N, keyDir), 0.0);
    vec3 key = vec3(1.0, 0.95, 0.85) * 0.7 * keyNdotL;

    /* Fill light — cool, softer, from opposite side */
    vec3 fillDir = normalize(vec3(-0.5, 0.4, -0.3));
    float fillNdotL = max(dot(N, fillDir), 0.0);
    vec3 fill = vec3(0.7, 0.8, 1.0) * 0.2 * fillNdotL;

    /* Rim / back light — highlights edges */
    vec3 rimDir = normalize(vec3(0.0, 0.3, -1.0));
    float rimNdotL = max(dot(N, rimDir), 0.0);
    vec3 rim = vec3(1.0, 0.95, 0.9) * 0.15 * rimNdotL;

    vec3 color = fragColor * (ambient + key + fill + rim);

    outColor = vec4(color, 1.0);
}
