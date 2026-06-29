// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* scene_test.c — load each test USDA via scene_load() and verify
 * counts, topology, and that the bounds box is non-degenerate.
 * No GL context required. Run from the repo root or with the test
 * scene paths passed via argv. */
#include "scene.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); g_fail++; } \
    else         { fprintf(stderr, "ok:   %s\n", msg); } \
} while (0)

static int load_and_check(const char* path, int min_meshes, const char* label) {
    fprintf(stderr, "\n--- %s (%s) ---\n", label, path);
    Scene* s = scene_load(path);
    if (!s) {
        fprintf(stderr, "FAIL: scene_load returned NULL for %s\n", path);
        g_fail++;
        return 0;
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "%s: nmeshes >= %d", label, min_meshes);
    CHECK(s->nmeshes >= min_meshes, buf);

    /* Bounds non-degenerate: max > min on at least one axis. */
    float dx = s->bounds_max[0] - s->bounds_min[0];
    float dy = s->bounds_max[1] - s->bounds_min[1];
    float dz = s->bounds_max[2] - s->bounds_min[2];
    snprintf(buf, sizeof(buf), "%s: bounds non-degenerate (dx=%.2f dy=%.2f dz=%.2f)",
             label, dx, dy, dz);
    CHECK(dx > 1e-6f || dy > 1e-6f || dz > 1e-6f, buf);

    /* Each mesh has at least 3 vertices and a multiple-of-3 index count. */
    for (int m = 0; m < s->nmeshes; m++) {
        SceneMesh* mesh = &s->meshes[m];
        snprintf(buf, sizeof(buf), "%s: mesh[%d] has positions", label, m);
        CHECK(mesh->positions != NULL && mesh->nvertices >= 3, buf);

        snprintf(buf, sizeof(buf), "%s: mesh[%d] indices triangulated", label, m);
        CHECK(mesh->indices != NULL && mesh->nindices > 0 && mesh->nindices % 3 == 0, buf);

        /* Indices must reference valid vertices. */
        uint32_t worst_idx = 0;
        int bad_at = -1;
        for (int i = 0; i < mesh->nindices; i++) {
            if (mesh->indices[i] >= (uint32_t)mesh->nvertices) {
                if (bad_at < 0) { bad_at = i; worst_idx = mesh->indices[i]; }
            }
        }
        if (bad_at >= 0) {
            fprintf(stderr, "  detail: mesh[%d] nvertices=%d, indices[%d]=%u\n",
                    m, mesh->nvertices, bad_at, worst_idx);
        }
        snprintf(buf, sizeof(buf), "%s: mesh[%d] indices in-range "
                 "(nvertices=%d nindices=%d)",
                 label, m, mesh->nvertices, mesh->nindices);
        CHECK(bad_at < 0, buf);
    }

    scene_free(s);
    return 1;
}

static void check_authored_lights(const char* path) {
    fprintf(stderr, "\n--- authored lights (%s) ---\n", path);
    Scene* s = scene_load(path);
    if (!s) {
        fprintf(stderr, "FAIL: scene_load returned NULL for authored lights\n");
        g_fail++;
        return;
    }

    CHECK(s->has_authored_light != 0, "authored_lights: has_authored_light");
    CHECK(s->nlights == 3, "authored_lights: extracted 3 supported lights");

    int saw_dist = 0, saw_rect = 0, saw_sphere = 0;
    for (int i = 0; i < s->nlights; i++) {
        SceneLight* L = &s->lights[i];
        if (L->kind == SCENE_LIGHT_DISTANT) {
            saw_dist = 1;
            CHECK(fabsf(L->intensity - 1400.0f) < 1e-3f,
                  "authored_lights: DistantLight intensity/exposure");
            CHECK(fabsf(L->color[0] - 1.0f) < 1e-5f &&
                  fabsf(L->color[1] - 0.8f) < 1e-5f &&
                  fabsf(L->color[2] - 0.6f) < 1e-5f,
                  "authored_lights: DistantLight bare color");
        } else if (L->kind == SCENE_LIGHT_RECT) {
            saw_rect = 1;
            CHECK(fabsf(L->position[0] - 1.0f) < 1e-5f &&
                  fabsf(L->position[1] - 2.0f) < 1e-5f &&
                  fabsf(L->position[2] - 3.0f) < 1e-5f,
                  "authored_lights: RectLight transform position");
            CHECK(fabsf(L->u_axis[0] - 1.0f) < 1e-5f,
                  "authored_lights: RectLight half width");
            CHECK(fabsf(L->v_axis[1] - 0.5f) < 1e-5f,
                  "authored_lights: RectLight half height");
            CHECK(L->normalize == 1, "authored_lights: RectLight normalize");
        } else if (L->kind == SCENE_LIGHT_SPHERE) {
            saw_sphere = 1;
            CHECK(fabsf(L->u_axis[0] - 0.25f) < 1e-5f,
                  "authored_lights: SphereLight radius");
        }
    }

    CHECK(saw_dist, "authored_lights: saw DistantLight");
    CHECK(saw_rect, "authored_lights: saw RectLight");
    CHECK(saw_sphere, "authored_lights: saw SphereLight");
    scene_free(s);
}

int main(int argc, char** argv) {
    /* Allow the caller to pass in absolute paths to the test USDA files
     * (used by ctest from the build directory). Fall back to repo-root
     * relative paths for direct invocation. */
    const char* cube  = (argc > 1) ? argv[1] : "test_cube.usda";
    const char* mats  = (argc > 2) ? argv[2] : "test_materials.usda";
    const char* pbr   = (argc > 3) ? argv[3] : "test_pbr_materials.usda";
    const char* lights = (argc > 4) ? argv[4] : "tests/fixtures/authored_lights.usda";
    const char* refwrap = (argc > 5) ? argv[5] : "tests/fixtures/reference_wrapper.usda";

    load_and_check(cube, 1, "test_cube");
    load_and_check(mats, 1, "test_materials");
    load_and_check(pbr,  1, "test_pbr_materials");
    check_authored_lights(lights);
    load_and_check(refwrap, 1, "reference_wrapper");

    /* Negative test: nonexistent file must return NULL gracefully (no crash). */
    fprintf(stderr, "\n--- negative: nonexistent file ---\n");
    Scene* nope = scene_load("/this/path/does/not/exist.usda");
    CHECK(nope == NULL, "scene_load(missing) returns NULL");
    if (nope) scene_free(nope);

    fprintf(stderr, "\n%s: %d failures\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
