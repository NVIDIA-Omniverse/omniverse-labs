// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
/* Shader Execution Reordering (SER) — VK_NV_ray_tracing_invocation_reorder.
 *
 * When the build emits this shader with -DSER_ENABLED=1 (CMake target
 * `raytrace_tiled_ser.rgen.spv`), the compiled SPV uses hitObjectNV +
 * reorderThreadNV to cluster threads by hit instance before invoking the
 * shared closest-hit shader. The host loads this variant only when the
 * device exposes VK_NV_ray_tracing_invocation_reorder; otherwise it loads
 * the non-SER `raytrace_tiled.rgen.spv` (compiled without the define) so
 * that pipeline creation never depends on the optional extension.
 *
 * `enable` (not `require`) keeps the include below non-fatal on glslang
 * builds that recognise the extension symbol — the SER_ENABLED guard is
 * what actually toggles emission. */
#extension GL_NV_shader_invocation_reorder : enable

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

/* Phase C: per-env TLAS array. When isolation is on (cam_env_plus1 > 0)
 * each tile traces against tlas_arr[cam_env_idx] which contains only
 * that env's instances — true alias-free isolation regardless of the
 * 8-bit cullMask. */
layout(set = 0, binding = 16) uniform accelerationStructureEXT tlas_arr[];
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
    uint  deferred_shade_enabled; /* byte 152: Phase B G-buffer + compute */
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

    /* NU_TILE_RES output stride: per-env layout writes use these as the
     * output buffer's per-env stride. Packed in viewInverse[2][0..1] (offset
     * 32-39 of push constants = _pad[4..5]). When NU_TILE_RES is unset / =64
     * the host populates these with tile_w/tile_h so the shader's offset is
     * unchanged from before the probe. With NU_TILE_RES=32 the host puts the
     * UNSCALED tile_w/tile_h here so per-env strides match the caller's
     * (CudaOwnedInterop-allocated) buffer layout — the rgen still launches
     * at scaled dims and writes only the first (tile_w * tile_h) pixels of
     * each unscaled-stride env slot. The padding region is pre-cleared to
     * opaque-black by the host's vkCmdFillBuffer before the dispatch.
     *
     * Fallback: if either out_tile is zero (legacy host that doesn't
     * populate these slots), fall back to using tile_w/tile_h. */
    uint out_tile_w = floatBitsToUint(viewInverse[2][0]);
    uint out_tile_h = floatBitsToUint(viewInverse[2][1]);
    if (out_tile_w == 0u) out_tile_w = tile_w;
    if (out_tile_h == 0u) out_tile_h = tile_h;

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
            if (usePerEnvLayout != 0u) {
                /* Same per-env-layout block-fill as the hit/miss output below. */
                uint x_lo = (local_x * out_tile_w) / tile_w;
                uint x_hi = ((local_x + 1u) * out_tile_w) / tile_w;
                uint y_lo = (local_y * out_tile_h) / tile_h;
                uint y_hi = ((local_y + 1u) * out_tile_h) / tile_h;
                uint env_base = cam_idx * out_tile_w * out_tile_h;
                for (uint y = y_lo; y < y_hi; ++y) {
                    uint row = y * out_tile_w;
                    for (uint x = x_lo; x < x_hi; ++x) {
                        directOut.pixels[env_base + row + x] = 0xFF000000u;
                    }
                }
            } else {
                uint ofs = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
                directOut.pixels[ofs] = 0xFF000000u;
            }
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

    /* Per-tile env isolation. The IsaacLab vulkan adapter packs a sentinel
     * into the view_inverse [3][3] slot (normally 1.0 and only affects
     * origin.w which is discarded). The sign distinguishes the two paths:
     *
     *   sentinel >  0 (Phase A — default): use the legacy single TLAS with
     *                cullMask = 1 << ((sentinel - 1) & 7). Each instance
     *                was tagged with its env's bit at TLAS-build time, so
     *                the BVH prunes wrong-env subtrees at traversal.
     *   sentinel <  0 (Phase C — opt-in):  trace against
     *                tlas_arr[-sentinel - 1] which contains only that env's
     *                instances; cullMask = 0xFF (every instance in the
     *                per-env BVH belongs to this env by construction).
     *   sentinel == 0:  no isolation (visualizer / single-cam RT). Use
     *                legacy tlas with cullMask 0xFF. */
    int sentinel = int(camViewInverse[3][3]);
#ifdef SER_ENABLED
    /* SER path: trace into a hitObjectNV, hint by instance custom index so
     * threads hitting the same instance (and therefore the same material +
     * texture sampling pattern in raytrace.rchit) cluster together before
     * the closest-hit shader runs. The single-CHS scene has no SBT-table
     * dispatch divergence, so the win comes entirely from the rchit's
     * material lookup + texture indexing being more coherent post-reorder. */
    hitObjectNV hit;
    hitObjectRecordEmptyNV(hit);
    if (sentinel < 0) {
        /* Phase C path. */
        uint env_idx = uint(-sentinel - 1);
        hitObjectTraceRayNV(hit,
                            tlas_arr[nonuniformEXT(env_idx)],
                            gl_RayFlagsOpaqueEXT,
                            0xFFu,
                            0, 0, 0,
                            origin.xyz,
                            0.001,
                            direction.xyz,
                            100000.0,
                            0);
    } else {
        /* Phase A path (or no isolation when sentinel == 0). */
        uint cull_mask = (sentinel > 0)
            ? (1u << uint((sentinel - 1) & 7))
            : 0xFFu;
        hitObjectTraceRayNV(hit,
                            tlas,
                            gl_RayFlagsOpaqueEXT,
                            cull_mask,
                            0, 0, 0,
                            origin.xyz,
                            0.001,
                            direction.xyz,
                            100000.0,
                            0);
    }
    /* Coherence hint: instance id (low bits) clusters threads that hit the
     * same instance → same material → same texture sampling pattern. Misses
     * collapse to bucket 0 so they reorder together too. 16-bit hint = up
     * to 65k buckets, enough for our ~4096-instance scenes. */
    uint hint = hitObjectIsHitNV(hit) ? uint(hitObjectGetInstanceCustomIndexNV(hit)) : 0u;
    reorderThreadNV(hit, hint, 16);
    hitObjectExecuteShaderNV(hit, 0);
#else
    if (sentinel < 0) {
        /* Phase C path. */
        uint env_idx = uint(-sentinel - 1);
        traceRayEXT(tlas_arr[nonuniformEXT(env_idx)],
                    gl_RayFlagsOpaqueEXT,
                    0xFFu,
                    0, 0, 0,
                    origin.xyz,
                    0.001,
                    direction.xyz,
                    100000.0,
                    0);
    } else {
        /* Phase A path (or no isolation when sentinel == 0). */
        uint cull_mask = (sentinel > 0)
            ? (1u << uint((sentinel - 1) & 7))
            : 0xFFu;
        traceRayEXT(tlas,
                    gl_RayFlagsOpaqueEXT,
                    cull_mask,
                    0, 0, 0,
                    origin.xyz,
                    0.001,
                    direction.xyz,
                    100000.0,
                    0);
    }
#endif

    vec3 outRgb = (useSrgb != 0u) ? linear_to_srgb(hitValue.rgb) : hitValue.rgb;

    if (useDirectOut != 0u) {
        /* Pack RGBA8 into uint32 — same byte order as VK_FORMAT_R8G8B8A8_UNORM */
        uvec4 c = uvec4(clamp(vec4(outRgb, 1.0), 0.0, 1.0) * 255.0 + 0.5);
        uint packed = c.r | (c.g << 8u) | (c.b << 16u) | (c.a << 24u);
        if (usePerEnvLayout != 0u) {
            /* NU_TILE_RES per-env-layout block-fill — each rgen invocation
             * owns a contiguous range of output pixels in the unscaled
             * per-env stride. Using exclusive integer floor yields a
             * proper non-overlapping nearest-neighbour upsampling that
             * tiles the entire (out_tile_w x out_tile_h) per-env slot for
             * any out_tile / tile ratio (including non-integer scales like
             * NU_TILE_RES=48 with caller stride 64). At default
             * NU_TILE_RES (out_tile == tile) this collapses to the
             * single-pixel write below — byte-identical pre-fix. */
            uint x_lo = (local_x * out_tile_w) / tile_w;
            uint x_hi = ((local_x + 1u) * out_tile_w) / tile_w;
            uint y_lo = (local_y * out_tile_h) / tile_h;
            uint y_hi = ((local_y + 1u) * out_tile_h) / tile_h;
            uint env_base = cam_idx * out_tile_w * out_tile_h;
            for (uint y = y_lo; y < y_hi; ++y) {
                uint row = y * out_tile_w;
                for (uint x = x_lo; x < x_hi; ++x) {
                    directOut.pixels[env_base + row + x] = packed;
                }
            }
        } else {
            uint ofs = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
            directOut.pixels[ofs] = packed;
        }
    } else {
        imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), vec4(outRgb, 1.0));
    }
}
