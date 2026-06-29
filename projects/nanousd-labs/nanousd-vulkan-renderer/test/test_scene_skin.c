// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* test_scene_skin.c — end-to-end check of the REAL apply_usdskel_skinning normal
 * path (task #14). Loads a synthetic skinned mesh whose single joint carries a
 * non-uniform scale (2,1,0.5) and whose vertex normal is diagonal (1,1,0). With
 * the correct inverse-transpose the skinned normal is (0.5,1,0) -> (0.447,0.894,0);
 * the old direct matrix apply gave (2,1,0) -> (0.894,0.447,0). This exercises the
 * actual scene.c wiring (not the mirrored unit-test copy in test_normal_matrix.c),
 * so it fails if the fix is reverted OR mis-wired. */
#include "../src/scene.h"
#include <math.h>
#include <stdio.h>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <skinned_nonuniform.usda>\n", argv[0]); return 2; }
    Scene* scene = scene_load(argv[1]);
    if (!scene) { fprintf(stderr, "FAIL: scene_load returned NULL\n"); return 1; }
    if (scene->nmeshes < 1) { fprintf(stderr, "FAIL: no meshes loaded\n"); return 1; }
    SceneMesh* m = &scene->meshes[0];
    if (!m->normals || m->nvertices < 1) { fprintf(stderr, "FAIL: mesh has no normals\n"); return 1; }

    float nx = m->normals[0], ny = m->normals[1], nz = m->normals[2];
    printf("loaded skinned normal[0] = (%.5f, %.5f, %.5f)\n", nx, ny, nz);

    /* Correct (inverse-transpose) result; direction only, so allow sign flip. */
    const float ex = 0.44721f, ey = 0.89443f, ez = 0.0f;
    const float tol = 2e-3f;
    int ok = (fabsf(nx - ex) < tol && fabsf(ny - ey) < tol && fabsf(nz - ez) < tol) ||
             (fabsf(-nx - ex) < tol && fabsf(-ny - ey) < tol && fabsf(nz - ez) < tol);
    if (ok) { printf("test_scene_skin: PASS (skinned normal uses inverse-transpose)\n"); return 0; }

    /* Diagnose the classic regression explicitly. */
    const float bx = 0.89443f, by = 0.44721f;
    if (fabsf(fabsf(nx) - bx) < tol && fabsf(fabsf(ny) - by) < tol)
        fprintf(stderr, "FAIL: normal matches the OLD direct-apply bug (0.894,0.447,0); "
                        "inverse-transpose not applied\n");
    else if (fabsf(fabsf(nx) - 0.70711f) < tol && fabsf(fabsf(ny) - 0.70711f) < tol)
        fprintf(stderr, "FAIL: normal unchanged (0.707,0.707,0) — skinning did not run on this mesh\n");
    else
        fprintf(stderr, "FAIL: expected ~(%.3f,%.3f,%.3f), got (%.5f,%.5f,%.5f)\n", ex, ey, ez, nx, ny, nz);
    return 1;
}
