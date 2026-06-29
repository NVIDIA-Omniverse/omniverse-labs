// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* mtlx_usd_uvtiling_test.c — task #15 (OpenGL): UV tiling on MaterialX authored
 * directly in USD. Mirror of the Vulkan test. Loads a material whose base-color
 * image has inputs:uvtiling=(4,2) and asserts it reaches MaterialParams.
 * mdl_uv_transform; the orthogonality case (no uvtiling) must stay identity.
 * Note: OpenGL treats an all-zero mdl_uv_transform as identity (material.h), so
 * the no-tiling case accepts either (0,0,0,0) or (1,1,0,0). */
#include "material.h"
#include <nanousd/nanousdapi.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } } while (0)

static SceneMaterial* find_material(MaterialCollection* mc, const char* name) {
    for (int i = 0; i < mc->nmaterials; i++)
        if (strcmp(mc->materials[i].name, name) == 0) return &mc->materials[i];
    return NULL;
}

static int is_identity_uv(const float* t) {
    int all_zero = t[0] == 0.0f && t[1] == 0.0f && t[2] == 0.0f && t[3] == 0.0f;
    int unit = fabsf(t[0] - 1.0f) < 1e-4f && fabsf(t[1] - 1.0f) < 1e-4f;
    return all_zero || unit;
}

int main(int argc, char** argv) {
    const char* usd = argc > 1 ? argv[1] : "tests/fixtures/mtlx_usd_uvtiling.usda";
    const char* scene_dir = argc > 2 ? argv[2] : "tests/fixtures";

    NanousdStage stage = nanousd_open(usd);
    CHECK(stage && nanousd_isvalid(stage), "nanousd_open uvtiling fixture");
    if (!stage || !nanousd_isvalid(stage)) return 1;

    MaterialCollection* mc = materials_load(stage, scene_dir);
    CHECK(mc != NULL, "materials_load");
    if (!mc) { nanousd_close(stage); return 1; }

    SceneMaterial* tiled = find_material(mc, "TiledMat");
    CHECK(tiled != NULL, "TiledMat exists");
    if (tiled) {
        const float* t = tiled->params.mdl_uv_transform;
        printf("TiledMat mdl_uv_transform = (%.3f, %.3f, %.3f, %.3f)\n", t[0], t[1], t[2], t[3]);
        CHECK(fabsf(t[0] - 4.0f) < 1e-4f, "TiledMat uv scale x == 4 (from inputs:uvtiling)");
        CHECK(fabsf(t[1] - 2.0f) < 1e-4f, "TiledMat uv scale y == 2 (from inputs:uvtiling)");
    }

    SceneMaterial* plain = find_material(mc, "PlainMat");
    CHECK(plain != NULL, "PlainMat exists");
    if (plain) {
        const float* t = plain->params.mdl_uv_transform;
        printf("PlainMat mdl_uv_transform = (%.3f, %.3f, %.3f, %.3f)\n", t[0], t[1], t[2], t[3]);
        CHECK(is_identity_uv(t), "PlainMat stays identity (no tiling)");
    }

    nanousd_close(stage);
    if (g_fail == 0) { printf("mtlx_usd_uvtiling_test: PASS\n"); return 0; }
    printf("mtlx_usd_uvtiling_test: FAIL (%d checks)\n", g_fail);
    return 1;
}
