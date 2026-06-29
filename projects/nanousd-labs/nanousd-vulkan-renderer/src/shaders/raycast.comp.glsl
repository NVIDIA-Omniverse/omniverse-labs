// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#version 460
#extension GL_EXT_ray_query : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

/* Input: split layout — origins[num_rays*3] then directions[num_rays*3] */
layout(set = 0, binding = 1, std430) readonly buffer RayInput {
    float data[];
} rayIn;

/* Output: split layout — distances[num_rays], meshIds[num_rays],
 * normals[num_rays*3], then hitPositions[num_rays*3].
 * distance = -1.0 and meshId = -1.0 mean no hit. */
layout(set = 0, binding = 2, std430) writeonly buffer RayOutput {
    float data[];
} rayOut;

/* Scene data: vertex/index buffer device addresses per mesh.
 * Layout matches SceneData in raytrace.rchit.glsl. */
struct MeshData {
    uint64_t vertexAddress;
    uint64_t indexAddress;
    vec4     color;
    uint     tex_index;
    uint     source_id;
    uint     material_id;
    uint     ptex_color_offset;
};
layout(set = 0, binding = 3, scalar) buffer SceneData {
    uint     vertexStride;    /* floats per vertex */
    uint     hasMaterials;
    float    envMipLevels;
    float    envIntensity;    /* USD DomeLight intensity (rmiss sky) */
    vec4     domeColor;       /* fast_mode flat sky/ambient: rgb + intensity */
    uint     upAxis;
    uint     _scenePad0;
    uint     _scenePad1;
    uint     _scenePad2;
    MeshData meshes[];
} scene;

/* Buffer references for fetching vertex positions + triangle indices */
layout(buffer_reference, scalar) buffer Vertices { float v[]; };
layout(buffer_reference, scalar) buffer Indices  { uint  i[]; };

layout(push_constant) uniform PushConstants {
    uint  num_rays;
    float max_distance;
};

void main()
{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_rays) return;

    /* Split layout: origins at [0..num_rays*3), directions at [num_rays*3..num_rays*6) */
    uint o = idx * 3;
    uint d = num_rays * 3 + idx * 3;
    vec3 origin    = vec3(rayIn.data[o + 0], rayIn.data[o + 1], rayIn.data[o + 2]);
    vec3 direction = vec3(rayIn.data[d + 0], rayIn.data[d + 1], rayIn.data[d + 2]);

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF,           /* cullMask — hit all geometry */
        origin, 0.001,  /* tmin */
        direction,
        max_distance);  /* tmax */

    while (rayQueryProceedEXT(rq)) {}

    /* Split output layout: distances[num_rays], meshIds[num_rays],
     * normals[num_rays*3], hitPos[num_rays*3]. */
    uint dist_off = idx;
    uint id_off   = num_rays + idx;
    uint norm_off = num_rays * 2u + idx * 3u;
    uint hit_off  = num_rays * 2u + num_rays * 3u + idx * 3u;

    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
        float t = rayQueryGetIntersectionTEXT(rq, true);
        vec3 hitPos = origin + t * direction;

        /* Fetch the real triangle geometry normal from the hit BLAS.
         *   instanceCustomIndex  → compact mesh index in the SceneData SSBO
         *   primitiveIndex       → triangle index inside that mesh's BLAS
         *   scene.meshes[].source_id → public renderer mesh id
         * Each vertex has scene.vertexStride floats, with position at offset 0.
         * Face normal from (v1-v0) x (v2-v0); transform through the TLAS
         * instance object-to-world rotation to get a world-space normal.
         * If the ray hit the back face, flip the normal toward the ray origin
         * (the raster path is double-sided; grazing physics cares about |dot|). */
        int meshIdx  = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
        int primIdx  = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        uint stride  = scene.vertexStride;
        Vertices verts = Vertices(scene.meshes[meshIdx].vertexAddress);
        Indices  idxs  = Indices(scene.meshes[meshIdx].indexAddress);

        uint base = uint(primIdx) * 3u;
        uint i0 = idxs.i[base + 0u];
        uint i1 = idxs.i[base + 1u];
        uint i2 = idxs.i[base + 2u];

        vec3 p0 = vec3(verts.v[i0*stride+0u], verts.v[i0*stride+1u], verts.v[i0*stride+2u]);
        vec3 p1 = vec3(verts.v[i1*stride+0u], verts.v[i1*stride+1u], verts.v[i1*stride+2u]);
        vec3 p2 = vec3(verts.v[i2*stride+0u], verts.v[i2*stride+1u], verts.v[i2*stride+2u]);

        vec3 faceN_local = normalize(cross(p1 - p0, p2 - p0));

        /* Transform by TLAS instance object-to-world (3x4 affine). The normal
         * transform is strictly inverse-transpose, but TLAS transforms here
         * are rotation + translation + uniform scale — inverse-transpose on the
         * rotation part is the rotation itself, so object-to-world on the
         * normal (as a direction) followed by normalize is correct. */
        mat4x3 o2w = rayQueryGetIntersectionObjectToWorldEXT(rq, true);
        vec3 normal = normalize(mat3(o2w) * faceN_local);

        /* Flip back-face hits so |direction . normal| reflects the true
         * grazing angle regardless of triangle winding. */
        bool frontFace = rayQueryGetIntersectionFrontFaceEXT(rq, true);
        if (!frontFace) normal = -normal;

        rayOut.data[dist_off]     = t;
        rayOut.data[id_off]       = float(scene.meshes[meshIdx].source_id);
        rayOut.data[norm_off + 0] = normal.x;
        rayOut.data[norm_off + 1] = normal.y;
        rayOut.data[norm_off + 2] = normal.z;
        rayOut.data[hit_off + 0]  = hitPos.x;
        rayOut.data[hit_off + 1]  = hitPos.y;
        rayOut.data[hit_off + 2]  = hitPos.z;
    } else {
        rayOut.data[dist_off]     = -1.0;
        rayOut.data[id_off]       = -1.0;
        rayOut.data[norm_off + 0] = 0.0;
        rayOut.data[norm_off + 1] = 0.0;
        rayOut.data[norm_off + 2] = 0.0;
        rayOut.data[hit_off + 0]  = 0.0;
        rayOut.data[hit_off + 1]  = 0.0;
        rayOut.data[hit_off + 2]  = 0.0;
    }
}
