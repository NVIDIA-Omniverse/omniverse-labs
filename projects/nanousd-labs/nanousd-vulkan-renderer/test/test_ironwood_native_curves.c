// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * test_ironwood_native_curves.c
 *
 * Regression for the real Moana Ironwood failure: Storm sees 56
 * instance-proxy BasisCurves under two instanceable Ironwood prims. The
 * nanousd Vulkan scene path must load the same authored curve prims by
 * replaying native-instance asset arcs.
 */

#include "../src/scene.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int file_exists(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

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
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <ironwood_wrapper.usda> <expected_curves> "
                "<expected_segments> [required_asset]\n",
                argv[0]);
        return 1;
    }

    const char* usd_path = argv[1];
    int expected_curves = atoi(argv[2]);
    long long expected_segments = atoll(argv[3]);

    if (!file_exists(usd_path)) {
        printf("SKIP: Ironwood wrapper missing: %s\n", usd_path);
        return 77;
    }
    for (int i = 4; i < argc; i++) {
        if (!file_exists(argv[i])) {
            printf("SKIP: Ironwood dependency missing: %s\n", argv[i]);
            return 77;
        }
    }

    test_setenv("NUSD_FLAT_NATIVE_INSTANCE_TRAVERSAL", "1");
    test_setenv("NUSD_RENDER_PI_BATCHES", "1");
    test_setenv("NUSD_APPLY_METERS_PER_UNIT", "0");
    test_unsetenv("NUSD_NATIVE_CURVES");

    Scene* scene = scene_load(usd_path);
    if (!scene) {
        fprintf(stderr, "FAIL: scene_load returned NULL for %s\n", usd_path);
        return 1;
    }

    long long total_segments = 0;
    long long total_widths = 0;
    double width_sum = 0.0;
    float width_min = FLT_MAX;
    float width_max = -FLT_MAX;
    int bspline_curves = 0;
    int display_colored_curves = 0;
    int ribbon_curves = 0;
    for (int i = 0; i < scene->ncurves; i++) {
        const SceneCurve* c = &scene->curves[i];
        total_segments += scene_curve_count_segments(&scene->curves[i]);
        if (c->basis == CURVE_BASIS_BSPLINE) bspline_curves++;
        if (c->has_display_color) display_colored_curves++;
        if (c->has_normals) ribbon_curves++;
        if (!c->widths) {
            fprintf(stderr, "FAIL: Ironwood curve %d has no widths\n", i);
            scene_free(scene);
            return 1;
        }
        for (int v = 0; v < c->nv; v++) {
            float w = c->widths[v];
            if (w < width_min) width_min = w;
            if (w > width_max) width_max = w;
            width_sum += (double)w;
            total_widths++;
        }
    }
    double width_mean = total_widths ? width_sum / (double)total_widths : 0.0;

    printf("ironwood_native_curves: meshes=%d curves=%d segments=%lld widths=%lld "
           "width_min=%.8g width_max=%.8g width_mean=%.12g bspline=%d "
           "displayColor=%d ribbons=%d\n",
           scene->nmeshes, scene->ncurves, total_segments, total_widths,
           (double)width_min, (double)width_max, width_mean, bspline_curves,
           display_colored_curves, ribbon_curves);

    if (scene->ncurves != expected_curves) {
        fprintf(stderr,
                "FAIL: expected %d Ironwood curves visible to Storm, got %d\n",
                expected_curves, scene->ncurves);
        scene_free(scene);
        return 1;
    }

    if (total_segments != expected_segments) {
        fprintf(stderr,
                "FAIL: expected %lld Ironwood curve segments, got %lld\n",
                expected_segments, total_segments);
        scene_free(scene);
        return 1;
    }

    if (bspline_curves != expected_curves) {
        fprintf(stderr,
                "FAIL: expected all %d Ironwood curves to use basis=bspline, got %d\n",
                expected_curves, bspline_curves);
        scene_free(scene);
        return 1;
    }

    if (display_colored_curves != expected_curves) {
        fprintf(stderr,
                "FAIL: expected all %d Ironwood curves to carry authored displayColor, got %d\n",
                expected_curves, display_colored_curves);
        scene_free(scene);
        return 1;
    }

    if (ribbon_curves != 0) {
        fprintf(stderr,
                "FAIL: Ironwood widths-only curves should stay tubes, got %d ribbon curves\n",
                ribbon_curves);
        scene_free(scene);
        return 1;
    }

    if (total_widths != 6870612LL ||
        fabs((double)width_min - 0.0) > 1.0e-7 ||
        fabs((double)width_max - 3.25) > 1.0e-6 ||
        fabs(width_mean - 0.0247535655693) > 1.0e-8) {
        fprintf(stderr,
                "FAIL: Ironwood width stats changed: samples=%lld min=%.8g "
                "max=%.8g mean=%.12g\n",
                total_widths, (double)width_min, (double)width_max, width_mean);
        scene_free(scene);
        return 1;
    }

    scene_free(scene);
    printf("test_ironwood_native_curves: PASS\n");
    return 0;
}
