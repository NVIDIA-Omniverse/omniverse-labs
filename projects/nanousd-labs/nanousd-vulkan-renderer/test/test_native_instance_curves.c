// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * test_native_instance_curves.c
 *
 * Regression for instance-proxy BasisCurves. OpenUSD Storm sees curves under
 * native instance proxies; the Vulkan scene loader must replay those curves
 * from asset arcs just like it replays native-instance meshes.
 */

#include "../src/scene.h"
#include <stdio.h>
#include <stdlib.h>

static void test_setenv(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void test_unsetenv(const char* name)
{
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <native_instance_curve_root.usda>\n", argv[0]);
        return 1;
    }

    test_setenv("NUSD_FLAT_NATIVE_INSTANCE_TRAVERSAL", "1");
    test_setenv("NUSD_RENDER_PI_BATCHES", "1");
    test_unsetenv("NUSD_NATIVE_CURVES");

    Scene* scene = scene_load(argv[1]);
    if (!scene) {
        fprintf(stderr, "FAIL: scene_load returned NULL\n");
        return 1;
    }

    int total_segments = 0;
    for (int i = 0; i < scene->ncurves; i++) {
        total_segments += scene_curve_count_segments(&scene->curves[i]);
    }

    printf("native_instance_curves: meshes=%d curves=%d segments=%d\n",
           scene->nmeshes, scene->ncurves, total_segments);

    if (scene->ncurves != 2) {
        fprintf(stderr, "FAIL: expected 2 replayed native instance curves, got %d\n",
                scene->ncurves);
        scene_free(scene);
        return 1;
    }
    if (total_segments != 4) {
        fprintf(stderr, "FAIL: expected 4 replayed curve segments, got %d\n",
                total_segments);
        scene_free(scene);
        return 1;
    }

    scene_free(scene);
    printf("test_native_instance_curves: PASS\n");
    return 0;
}
