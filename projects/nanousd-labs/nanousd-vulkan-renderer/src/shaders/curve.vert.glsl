// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 450

layout(location = 0) in vec3  inPosition;
layout(location = 1) in float inWidth;
layout(location = 2) in uint   inColorPacked;

layout(location = 0) out vec3  vsPos;
layout(location = 1) out float vsWidth;
layout(location = 2) out vec3  vsColor;

vec3 unpack_rgba8(uint c) {
    return vec3(float(c & 255u),
                float((c >> 8u) & 255u),
                float((c >> 16u) & 255u)) / 255.0;
}

void main() {
    vsPos = inPosition;
    vsWidth = inWidth;
    vsColor = unpack_rgba8(inColorPacked);
}
