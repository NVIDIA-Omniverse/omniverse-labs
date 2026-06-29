// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_PRIMITIVE_GEOM_H
#define NUSD_PRIMITIVE_GEOM_H

/*
 * primitive_geom.h — Tessellation for UsdGeom* primitive shape prims
 * (Cube / Sphere / Capsule / Cylinder). Synthesizes positions + indices
 * + sharp normals so these prims feed the same SceneMesh path that
 * UsdGeomMesh prims use.
 *
 * Caller owns nothing — all allocations land in the supplied arena.
 */

#include "arena.h"
#include "scene.h"
#include "nanousd/nanousdapi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * If `type_name` is "Cube" / "Sphere" / "Capsule" / "Cylinder", reads
 * the prim's USD attributes, tessellates into triangles, allocates
 * positions / normals / indices in `arena`, and writes them to
 * `out_mesh` (nvertices, nindices, positions, normals, indices).
 *
 * Returns 1 on success, 0 if `type_name` isn't a known primitive shape
 * or the tessellation failed (out-of-memory).
 *
 * Other SceneMesh fields (world_xform, display_color, material_index,
 * texcoords, ...) are left untouched — the caller fills them after.
 */
int load_primitive_geom(NanousdPrim prim,
                        const char* type_name,
                        Arena* arena,
                        SceneMesh* out_mesh);

#ifdef __cplusplus
}
#endif

#endif
