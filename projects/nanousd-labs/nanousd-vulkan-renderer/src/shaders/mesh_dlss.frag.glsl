// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 450

/*
 * DLSS fragment shader — MRT output:
 *   location 0: color (RGBA8)
 *   location 1: motion vector (RG16F, screen-space pixels)
 *
 * Motion vector convention: MV = previous_pixel - current_pixel
 * (maps current pixel to where it was in the previous frame).
 */

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec4 currClipPos;
layout(location = 4) in vec4 prevClipPos;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotionVector;

void main() {
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

    vec3 key  = vec3(1.0, 0.95, 0.85) * 0.7 * keyNdotL;
    vec3 fill = vec3(0.7, 0.8, 1.0) * 0.2 * fillNdotL;
    vec3 rim  = vec3(1.0, 0.95, 0.9) * 0.15 * rimNdotL;

    vec3 color = fragColor * (ambient + key + fill + rim);
    outColor = vec4(color, 1.0);

    /* Motion vector: previous NDC - current NDC, in pixel space.
     * NDC ranges [-1, 1], multiply by 0.5 * resolution to get pixels. */
    vec2 currNDC = currClipPos.xy / currClipPos.w;
    vec2 prevNDC = prevClipPos.xy / prevClipPos.w;
    outMotionVector = (prevNDC - currNDC) * 0.5;
    /* Note: this is in normalized [0,1] units. The DLSS wrapper sets
     * MV scale to render_width/render_height to convert to pixels. */
}
