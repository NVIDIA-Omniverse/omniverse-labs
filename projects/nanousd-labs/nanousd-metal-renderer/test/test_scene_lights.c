// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../src/scene.h"

#include <math.h>
#include <stdio.h>

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", msg); \
        failures++; \
    } \
} while (0)

int main(int argc, char** argv)
{
    int failures = 0;
    const char* path = (argc > 1) ? argv[1] : "test/fixtures/authored_lights.usda";

    Scene* s = scene_load(path);
    if (!s) {
        fprintf(stderr, "FAIL: scene_load returned NULL for %s\n", path);
        return 1;
    }

    CHECK(s->nlights == 3, "extracted 3 supported lights");

    int saw_dist = 0;
    int saw_rect = 0;
    int saw_sphere = 0;
    for (int i = 0; i < s->nlights; i++) {
        SceneLight* L = &s->lights[i];
        if (L->kind == SCENE_LIGHT_DISTANT) {
            saw_dist = 1;
            CHECK(fabsf(L->intensity - 1400.0f) < 1e-3f,
                  "DistantLight legacy intensity/exposure");
            CHECK(fabsf(L->color[0] - 1.0f) < 1e-5f &&
                  fabsf(L->color[1] - 0.8f) < 1e-5f &&
                  fabsf(L->color[2] - 0.6f) < 1e-5f,
                  "DistantLight legacy color");
            CHECK(fabsf(L->angle_deg - 2.0f) < 1e-5f,
                  "DistantLight legacy angle");
        } else if (L->kind == SCENE_LIGHT_RECT) {
            saw_rect = 1;
            CHECK(fabsf(L->position[0] - 1.0f) < 1e-5f &&
                  fabsf(L->position[1] - 2.0f) < 1e-5f &&
                  fabsf(L->position[2] - 3.0f) < 1e-5f,
                  "RectLight authored transform position");
            CHECK(fabsf(L->u_axis[0] - 1.0f) < 1e-5f,
                  "RectLight half width");
            CHECK(fabsf(L->v_axis[1] - 0.5f) < 1e-5f,
                  "RectLight authored half height");
            CHECK(L->normalize == 1, "RectLight normalize");
        } else if (L->kind == SCENE_LIGHT_SPHERE) {
            saw_sphere = 1;
            CHECK(fabsf(L->u_axis[0] - 0.25f) < 1e-5f,
                  "SphereLight radius");
        }
    }

    CHECK(saw_dist, "saw DistantLight");
    CHECK(saw_rect, "saw RectLight");
    CHECK(saw_sphere, "saw SphereLight");

    scene_free(s);
    if (failures) return 1;
    printf("PASS: scene lights test\n");
    return 0;
}
