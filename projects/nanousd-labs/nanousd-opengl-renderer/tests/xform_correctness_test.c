// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * xform_correctness_test.c — CPU scene-loader transform correctness checks.
 */

#include "scene.h"
#include <math.h>
#include <stdio.h>

static int nearf4(float a, float b)
{
    return fabsf(a - b) <= 1e-4f;
}

static int check_reset_xform(const char* path)
{
    Scene* scene = scene_load(path);
    if (!scene) {
        fprintf(stderr, "FAIL: scene_load returned NULL for %s\n", path);
        return 1;
    }

    int fail = 0;
    if (scene->nmeshes != 1) {
        fprintf(stderr, "FAIL: reset expected 1 mesh, got %d\n", scene->nmeshes);
        fail = 1;
    }

    if (!nearf4(scene->bounds_min[0], 1.0f) ||
        !nearf4(scene->bounds_max[0], 2.0f) ||
        !nearf4(scene->bounds_min[1], 0.0f) ||
        !nearf4(scene->bounds_max[1], 1.0f)) {
        fprintf(stderr,
                "FAIL: resetXformStack bounds = "
                "(%.3f, %.3f, %.3f) - (%.3f, %.3f, %.3f), "
                "expected x=[1,2] y=[0,1]\n",
                scene->bounds_min[0], scene->bounds_min[1], scene->bounds_min[2],
                scene->bounds_max[0], scene->bounds_max[1], scene->bounds_max[2]);
        fail = 1;
    }

    scene_free(scene);
    return fail;
}

static int check_metrics(const char* path)
{
    Scene* scene = scene_load(path);
    if (!scene) {
        fprintf(stderr, "FAIL: scene_load returned NULL for %s\n", path);
        return 1;
    }

    int fail = 0;
    if (!nearf4(scene->bounds_min[0], 0.0f) ||
        !nearf4(scene->bounds_max[0], 1.0f) ||
        !nearf4(scene->bounds_min[1], 0.0f) ||
        !nearf4(scene->bounds_max[1], 1.0f) ||
        !nearf4(scene->bounds_min[2], 0.0f) ||
        !nearf4(scene->bounds_max[2], 0.0f)) {
        fprintf(stderr,
                "FAIL: metrics/upAxis bounds = "
                "(%.3f, %.3f, %.3f) - (%.3f, %.3f, %.3f), "
                "expected x=[0,1] y=[0,1] z=[0,0]\n",
                scene->bounds_min[0], scene->bounds_min[1], scene->bounds_min[2],
                scene->bounds_max[0], scene->bounds_max[1], scene->bounds_max[2]);
        fail = 1;
    }

    scene_free(scene);
    return fail;
}

static int check_facevarying_uv(const char* path)
{
    Scene* scene = scene_load(path);
    if (!scene) {
        fprintf(stderr, "FAIL: scene_load returned NULL for %s\n", path);
        return 1;
    }

    int fail = 0;
    if (scene->nmeshes != 1) {
        fprintf(stderr, "FAIL: uv seam expected 1 mesh, got %d\n", scene->nmeshes);
        fail = 1;
    } else {
        SceneMesh* mesh = &scene->meshes[0];
        if (mesh->nvertices != 6 || mesh->nindices != 6) {
            fprintf(stderr,
                    "FAIL: uv seam expected expanded 6 vertices/6 indices, got %d/%d\n",
                    mesh->nvertices, mesh->nindices);
            fail = 1;
        }
        if (!mesh->texcoords) {
            fprintf(stderr, "FAIL: uv seam mesh has no texcoords\n");
            fail = 1;
        } else if (!nearf4(mesh->texcoords[0], 0.0f) ||
                   !nearf4(mesh->texcoords[1], 0.0f) ||
                   !nearf4(mesh->texcoords[6], 0.25f) ||
                   !nearf4(mesh->texcoords[7], 0.25f)) {
            fprintf(stderr,
                    "FAIL: uv seam expected split vertex UVs (0,0) and (0.25,0.25), "
                    "got (%.3f,%.3f) and (%.3f,%.3f)\n",
                    mesh->texcoords[0], mesh->texcoords[1],
                    mesh->texcoords[6], mesh->texcoords[7]);
            fail = 1;
        }
    }

    scene_free(scene);
    return fail;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <reset_xform_stack.usda> "
                "<metrics_zup_centimeters.usda> <face_varying_uv_seam.usda>\n",
                argv[0]);
        return 1;
    }

    int fail = 0;
    fail |= check_reset_xform(argv[1]);
    fail |= check_metrics(argv[2]);
    fail |= check_facevarying_uv(argv[3]);
    if (fail) return 1;
    printf("PASS: scene transform, metrics, and UV seam checks\n");
    return 0;
}
