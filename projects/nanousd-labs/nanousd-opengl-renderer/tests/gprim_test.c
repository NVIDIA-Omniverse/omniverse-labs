// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* test_gprim.c — UsdGeomCube + UsdGeomCone render gate. Cube size=2 occupies
 * +-1 on all axes; Cone radius=1 height=2 occupies +-1 radial and +-1 axial.
 * Both must load as non-empty meshes (they were skipped before). */
#include "../src/scene.h"
#include <math.h>
#include <stdio.h>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "tests/assets/gprim_cube_cone.usda";
    Scene* scene = scene_load(path);
    if (!scene) { fprintf(stderr, "FAIL: scene_load NULL\n"); return 1; }
    if (scene->nmeshes != 2) { fprintf(stderr, "FAIL: expected 2 meshes (Cube+Cone), got %d\n", scene->nmeshes); return 1; }
    int fail = 0;
    for (int i = 0; i < scene->nmeshes; i++) {
        SceneMesh* m = &scene->meshes[i];
        if (!m->positions || m->nvertices < 8) { fprintf(stderr, "FAIL: mesh %d empty (%d verts)\n", i, m->nvertices); fail = 1; continue; }
        float mn[3] = {1e9f,1e9f,1e9f}, mx[3] = {-1e9f,-1e9f,-1e9f};
        for (int v = 0; v < m->nvertices; v++)
            for (int k = 0; k < 3; k++) { float c = m->positions[v*3+k]; if (c<mn[k]) mn[k]=c; if (c>mx[k]) mx[k]=c; }
        printf("mesh %d (%s): x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f] %d verts\n",
               i, m->path?m->path:"?", mn[0],mx[0],mn[1],mx[1],mn[2],mx[2], m->nvertices);
        for (int k = 0; k < 3; k++)
            if (fabsf(mx[k]-1.0f) > 0.05f || fabsf(mn[k]+1.0f) > 0.05f) {
                fprintf(stderr, "FAIL: mesh %d axis %d extent != +-1\n", i, k); fail = 1;
            }
    }
    if (!fail) { printf("gprim_test: PASS (cube + cone)\n"); return 0; }
    return 1;
}
