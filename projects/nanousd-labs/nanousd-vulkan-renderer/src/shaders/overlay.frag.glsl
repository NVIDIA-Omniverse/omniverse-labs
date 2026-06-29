// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 450

layout(set = 0, binding = 0) uniform sampler2D fontTex;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    if (fragUV.x < 0.0) {
        /* Solid fill (background rectangle) */
        outColor = fragColor;
    } else {
        /* Anti-aliased text */
        float d = texture(fontTex, fragUV).r;
        float aaw = fwidth(d) * 0.75;
        float alpha = smoothstep(0.25 - aaw, 0.25 + aaw, d);
        outColor = vec4(fragColor.rgb, fragColor.a * alpha);
    }
}
