// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* test_capsule.c — UsdGeomCapsule true-scale tessellation gate (Newton robot
 * limbs are capsules). A capsule radius=0.5 height=2.0 axis=Z must occupy
 * object-space x,y in [-0.5,0.5] and z in [-1.5,1.5] (z extent = height/2 +
 * radius on each end). The old unit-primitive+nonuniform-scale path (the bug)
 * would scale the caps with the cylinder and give z in [-1.0,1.0] with flattened
 * caps. Exercises the real scene_load -> load_capsule path. */
#include "../src/scene.h"
#include <math.h>
#include <stdio.h>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "tests/assets/capsule_z.usda";
    Scene* scene = scene_load(path);
    if (!scene) { fprintf(stderr, "FAIL: scene_load NULL\n"); return 1; }
    if (scene->nmeshes < 1) { fprintf(stderr, "FAIL: capsule produced no mesh (skipped?)\n"); return 1; }
    SceneMesh* m = &scene->meshes[0];
    if (!m->positions || m->nvertices < 8) { fprintf(stderr, "FAIL: capsule mesh empty (%d verts)\n", m->nvertices); return 1; }

    float mn[3] = { 1e9f, 1e9f, 1e9f }, mx[3] = { -1e9f, -1e9f, -1e9f };
    for (int v = 0; v < m->nvertices; v++)
        for (int k = 0; k < 3; k++) {
            float c = m->positions[v*3+k];
            if (c < mn[k]) mn[k] = c;
            if (c > mx[k]) mx[k] = c;
        }
    printf("capsule object bounds: x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f] (%d verts)\n",
           mn[0], mx[0], mn[1], mx[1], mn[2], mx[2], m->nvertices);
    int fail = 0;
    if (fabsf(mx[0] - 0.5f) > 0.02f || fabsf(mn[0] + 0.5f) > 0.02f) { fprintf(stderr, "FAIL: x extent != +-0.5 (radius)\n"); fail = 1; }
    if (fabsf(mx[1] - 0.5f) > 0.02f || fabsf(mn[1] + 0.5f) > 0.02f) { fprintf(stderr, "FAIL: y extent != +-0.5 (radius)\n"); fail = 1; }
    /* z extent must be +-1.5 (height/2 + radius), NOT +-1.0 (the unit+scale bug). */
    if (fabsf(mx[2] - 1.5f) > 0.02f || fabsf(mn[2] + 1.5f) > 0.02f) {
        fprintf(stderr, "FAIL: z extent != +-1.5 (caps not true-scale: got [%.3f,%.3f])\n", mn[2], mx[2]); fail = 1;
    }
    if (!fail) { printf("test_capsule: PASS (true-scale capsule, spherical caps)\n"); return 0; }
    printf("test_capsule: FAIL\n");
    return 1;
}
