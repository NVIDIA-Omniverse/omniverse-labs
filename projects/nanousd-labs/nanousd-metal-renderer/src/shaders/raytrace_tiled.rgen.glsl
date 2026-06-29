// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba8) uniform image2D outputImage;

/* Direct output buffer: when bound, writes pixels to this SSBO instead of
 * the storage image, eliminating the vkCmdCopyImageToBuffer step. */
layout(set = 0, binding = 9, std430) writeonly buffer DirectOutput {
    uint pixels[];
} directOut;

/* Depth output buffer: one float per pixel (ray T value, -1.0 = miss/sky). */
layout(set = 0, binding = 10, std430) writeonly buffer DepthOutput {
    float depths[];
} depthOut;

/* Segmentation output buffer: one uint32 per pixel (mesh_id+1, 0 = miss). */
layout(set = 0, binding = 11, std430) writeonly buffer SegmentationOutput {
    uint ids[];
} segOut;

/* Normals output buffer: 3 floats (x,y,z) per pixel. */
layout(set = 0, binding = 12, std430) writeonly buffer NormalsOutput {
    float normals[];
} normOut;

/* Camera SSBO: array of (view_inverse, proj_inverse) pairs.
 * Each camera is 2 mat4 = 32 floats = 128 bytes. */
layout(set = 0, binding = 8) readonly buffer CameraSSBO {
    mat4 cameras[];  /* cameras[cam_idx * 2 + 0] = view_inv, [cam_idx * 2 + 1] = proj_inv */
};

/* Push constant layout matches the single-camera RT pipeline so that the
 * shared miss/hit shaders can read ground_y and scene_scale at the same
 * offsets (128 and 132).  The first 16 bytes of viewInverse are reinterpreted
 * as tile parameters via floatBitsToUint(). */
layout(push_constant) uniform PushConstants {
    mat4  viewInverse;   /* bytes 0-63:  first 16 bytes hold tile params */
    mat4  projInverse;   /* bytes 64-127: unused by tiled raygen */
    float ground_y;      /* byte 128 */
    float scene_scale;   /* byte 132 */
    uint  fast_mode;     /* byte 136 */
    uint  depth_enabled; /* byte 140: 1 = write depth to binding 10 */
    uint  segmentation_enabled; /* byte 144: 1 = write IDs to binding 11 */
    uint  normals_enabled; /* byte 148: 1 = write normals to binding 12 */
};

layout(location = 0) rayPayloadEXT vec4 hitValue;

/* Standard sRGB transfer function (IEC 61966-2-1). Applied per-channel before
 * the UNORM quantize so the raytracer's color output is gamma-encoded ready
 * for display. Python side skips its CPU LUT pass when this is enabled. */
vec3 linear_to_srgb(vec3 c)
{
    c = clamp(c, 0.0, 1.0);
    vec3 lo = 12.92 * c;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    bvec3 use_hi = greaterThan(c, vec3(0.0031308));
    return vec3(use_hi.x ? hi.x : lo.x,
                use_hi.y ? hi.y : lo.y,
                use_hi.z ? hi.z : lo.z);
}

void main()
{
    /* Extract tile parameters packed into the first 16 bytes of viewInverse.
     * GLSL mat4 is column-major, so column 0 holds the first 4 floats of the
     * row-major C struct: {tile_w, tile_h, num_cols, num_cameras} as uint bits. */
    uint tile_w      = floatBitsToUint(viewInverse[0][0]);
    uint tile_h      = floatBitsToUint(viewInverse[0][1]);
    uint num_cols    = floatBitsToUint(viewInverse[0][2]);
    uint num_cameras = floatBitsToUint(viewInverse[0][3]);

    /* Direct buffer output flag: packed in padding at offset 16 */
    uint useDirectOut = floatBitsToUint(viewInverse[1][0]);
    /* Per-env layout flag: packed in padding at offset 20.
     * When set, SSBO writes use per-env contiguous layout [env, H, W] instead
     * of tiled layout [tiled_H, tiled_W], eliminating CUDA de-tiling. */
    uint usePerEnvLayout = floatBitsToUint(viewInverse[1][1]);
    /* sRGB encode flag: packed in padding at offset 24.
     * When set, hitValue.rgb is gamma-encoded before UNORM quantize. Lets
     * callers skip a ~16 ms CPU LUT pass per tiled frame. */
    uint useSrgb = floatBitsToUint(viewInverse[1][2]);

    /* Compute which tile (camera) this pixel belongs to */
    uint tile_x = gl_LaunchIDEXT.x / tile_w;
    uint tile_y = gl_LaunchIDEXT.y / tile_h;
    uint cam_idx = tile_y * num_cols + tile_x;

    /* Local pixel coordinate within this tile (needed for both bounds check and ray gen) */
    uint local_x = gl_LaunchIDEXT.x % tile_w;
    uint local_y = gl_LaunchIDEXT.y % tile_h;

    /* Pixels outside valid camera range: write black + invalid depth */
    if (cam_idx >= num_cameras) {
        if (useDirectOut != 0u) {
            uint ofs = (usePerEnvLayout != 0u)
                ? cam_idx * tile_w * tile_h + local_y * tile_w + local_x
                : gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
            directOut.pixels[ofs] = 0xFF000000u;
        } else {
            imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(0, 0, 0, 1));
        }
        uint sensorOfs = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
        if (depth_enabled != 0u) depthOut.depths[sensorOfs] = -1.0;
        if (segmentation_enabled != 0u) segOut.ids[sensorOfs] = 0u;
        if (normals_enabled != 0u) { uint ni = sensorOfs * 3u; normOut.normals[ni] = 0.0; normOut.normals[ni+1u] = 0.0; normOut.normals[ni+2u] = 0.0; }
        return;
    }

    /* Local pixel coordinate within this tile */
    vec2 localPixel = vec2(local_x, local_y) + vec2(0.5);
    vec2 inUV = localPixel / vec2(float(tile_w), float(tile_h));
    vec2 d = inUV * 2.0 - 1.0;

    /* Fetch this camera's inverse matrices from the SSBO */
    mat4 camViewInverse = cameras[cam_idx * 2 + 0];
    mat4 camProjInverse = cameras[cam_idx * 2 + 1];

    vec4 origin    = vec4(0, 0, 0, 1) * camViewInverse;
    vec4 target    = vec4(d.x, d.y, 1, 1) * camProjInverse;
    vec4 direction = vec4(normalize(target.xyz), 0) * camViewInverse;

    hitValue = vec4(0.0, 0.0, 0.0, 0.0);

    traceRayEXT(tlas,
                gl_RayFlagsOpaqueEXT,
                0xFF,
                0, 0, 0,
                origin.xyz,
                0.001,
                direction.xyz,
                100000.0,
                0);

    vec3 outRgb = (useSrgb != 0u) ? linear_to_srgb(hitValue.rgb) : hitValue.rgb;

    if (useDirectOut != 0u) {
        /* Pack RGBA8 into uint32 — same byte order as VK_FORMAT_R8G8B8A8_UNORM */
        uvec4 c = uvec4(clamp(vec4(outRgb, 1.0), 0.0, 1.0) * 255.0 + 0.5);
        uint ofs = (usePerEnvLayout != 0u)
            ? cam_idx * tile_w * tile_h + local_y * tile_w + local_x
            : gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
        directOut.pixels[ofs] = c.r | (c.g << 8u) | (c.b << 16u) | (c.a << 24u);
    } else {
        imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(outRgb, 1.0));
    }
}
