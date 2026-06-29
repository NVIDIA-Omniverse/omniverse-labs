// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

/*
 * DLSS closest-hit shader — expanded payload:
 *   hitValue.rgb  = shaded color
 *   hitWorldPos   = world-space hit position (for MV reprojection)
 *   hitT          = ray t-value (for depth buffer)
 */

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 1) rayPayloadInEXT vec3 hitWorldPos;
layout(location = 2) rayPayloadInEXT float hitT;
hitAttributeEXT vec2 baryCoord;

/* Per-mesh geometry addresses (indexed by gl_InstanceCustomIndexEXT) */
struct MeshAddrs {
    uint64_t vertexAddress;
    uint64_t indexAddress;
};
layout(set = 0, binding = 2, scalar) buffer SceneData {
    MeshAddrs meshes[];
} scene;

/* Buffer references for vertex/index access */
layout(buffer_reference, scalar) buffer Vertices { float v[]; };
layout(buffer_reference, scalar) buffer Indices  { uint  i[]; };

void main()
{
    uint meshIdx = gl_InstanceCustomIndexEXT;
    Vertices verts = Vertices(scene.meshes[meshIdx].vertexAddress);
    Indices  idxs  = Indices(scene.meshes[meshIdx].indexAddress);

    /* gl_PrimitiveID is local to this mesh's BLAS */
    uint base = gl_PrimitiveID * 3;
    uint i0 = idxs.i[base + 0];
    uint i1 = idxs.i[base + 1];
    uint i2 = idxs.i[base + 2];

    /* Barycentric weights */
    vec3 bary = vec3(1.0 - baryCoord.x - baryCoord.y, baryCoord.x, baryCoord.y);

    /* Read positions: offset +0 in 9-float stride */
    vec3 p0 = vec3(verts.v[i0*9+0], verts.v[i0*9+1], verts.v[i0*9+2]);
    vec3 p1 = vec3(verts.v[i1*9+0], verts.v[i1*9+1], verts.v[i1*9+2]);
    vec3 p2 = vec3(verts.v[i2*9+0], verts.v[i2*9+1], verts.v[i2*9+2]);

    /* Read normals: offset +3 in 9-float stride */
    vec3 n0 = vec3(verts.v[i0*9+3], verts.v[i0*9+4], verts.v[i0*9+5]);
    vec3 n1 = vec3(verts.v[i1*9+3], verts.v[i1*9+4], verts.v[i1*9+5]);
    vec3 n2 = vec3(verts.v[i2*9+3], verts.v[i2*9+4], verts.v[i2*9+5]);

    /* Read colors: offset +6 in 9-float stride */
    vec3 c0 = vec3(verts.v[i0*9+6], verts.v[i0*9+7], verts.v[i0*9+8]);
    vec3 c1 = vec3(verts.v[i1*9+6], verts.v[i1*9+7], verts.v[i1*9+8]);
    vec3 c2 = vec3(verts.v[i2*9+6], verts.v[i2*9+7], verts.v[i2*9+8]);

    /* Interpolate local-space position */
    vec3 localPos = p0 * bary.x + p1 * bary.y + p2 * bary.z;

    /* Transform to world space via TLAS instance transform */
    hitWorldPos = (gl_ObjectToWorldEXT * vec4(localPos, 1.0)).xyz;
    hitT = gl_HitTEXT;

    /* Interpolate and transform normal */
    vec3 localNormal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);
    vec3 normal = normalize(gl_ObjectToWorldEXT * vec4(localNormal, 0.0));
    vec3 color  = c0 * bary.x + c1 * bary.y + c2 * bary.z;

    /* Dome (hemisphere) ambient — matches sky palette */
    vec3 skyColor    = vec3(0.50, 0.58, 0.75);
    vec3 groundColor = vec3(0.22, 0.20, 0.17);
    float hemi = dot(normal, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
    vec3 ambient = mix(groundColor, skyColor, hemi) + vec3(0.08, 0.08, 0.10);

    /* Three-point light setup matching RT closest-hit */
    /* Key light — warm directional (sun) */
    vec3 keyDir = normalize(vec3(0.3, 1.0, 0.5));
    float keyNdotL = max(dot(normal, keyDir), 0.0);
    vec3 key = vec3(1.0, 0.95, 0.85) * 0.7 * keyNdotL;

    /* Fill light — cool, softer, from opposite side */
    vec3 fillDir = normalize(vec3(-0.5, 0.4, -0.3));
    float fillNdotL = max(dot(normal, fillDir), 0.0);
    vec3 fill = vec3(0.7, 0.8, 1.0) * 0.2 * fillNdotL;

    /* Rim / back light — highlights edges */
    vec3 rimDir = normalize(vec3(0.0, 0.3, -1.0));
    float rimNdotL = max(dot(normal, rimDir), 0.0);
    vec3 rim = vec3(1.0, 0.95, 0.9) * 0.15 * rimNdotL;

    hitValue = color * (ambient + key + fill + rim);
}
