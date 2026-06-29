// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * test_scene_curves.c — Phase 11.A loader smoke test.
 *
 * Verifies the BasisCurves scene loader without exercising the GPU
 * pipeline. Loads a USD file, prints per-curve summary, and dumps the
 * first few extracted segments + AABBs.
 *
 * Then constructs a NuRenderer and calls nu_load_usd to verify the
 * renderer-side curve extraction path matches the scene loader's count.
 *
 * Usage: test_scene_curves <path.usda>
 */

#include "../src/scene.h"
#include "../include/nusd_renderer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <curves.usda>\n", argv[0]);
        return 1;
    }

    Scene* scene = scene_load(argv[1]);
    if (!scene) {
        fprintf(stderr, "FAIL: scene_load returned NULL\n");
        return 1;
    }

    printf("scene_load: '%s'\n", argv[1]);
    printf("  meshes:  %d\n", scene->nmeshes);
    printf("  curves:  %d\n", scene->ncurves);
    printf("  bounds:  (%.3f, %.3f, %.3f) - (%.3f, %.3f, %.3f)\n",
           scene->bounds_min[0], scene->bounds_min[1], scene->bounds_min[2],
           scene->bounds_max[0], scene->bounds_max[1], scene->bounds_max[2]);

    if (scene->ncurves == 0) {
        printf("WARN: no BasisCurves prims found\n");
        scene_free(scene);
        return 0;
    }

    /* Per-curve summary + segment extraction. */
    int total_segments = 0;
    for (int i = 0; i < scene->ncurves; i++) {
        const SceneCurve* c = &scene->curves[i];
        const char* basis_str =
            (c->basis == CURVE_BASIS_BEZIER)     ? "bezier" :
            (c->basis == CURVE_BASIS_BSPLINE)    ? "bspline" :
            (c->basis == CURVE_BASIS_CATMULLROM) ? "catmullRom" :
            (c->basis == CURVE_BASIS_LINEAR)     ? "linear" : "?";
        const char* wrap_str =
            (c->wrap == CURVE_WRAP_NONPERIODIC) ? "nonperiodic" :
            (c->wrap == CURVE_WRAP_PERIODIC)    ? "periodic" :
            (c->wrap == CURVE_WRAP_PINNED)      ? "pinned" : "?";

        int n_segs = scene_curve_count_segments(c);
        total_segments += n_segs;

        printf("  curve[%d]: nv=%d sub=%d type=%s basis=%s wrap=%s segs=%d\n",
               i, c->nv, c->ncurves_in_prim,
               c->type_is_cubic ? "cubic" : "linear",
               basis_str, wrap_str, n_segs);
        printf("            bounds (%.3f, %.3f, %.3f) - (%.3f, %.3f, %.3f)\n",
               c->bounds_min[0], c->bounds_min[1], c->bounds_min[2],
               c->bounds_max[0], c->bounds_max[1], c->bounds_max[2]);

        if (n_segs > 0) {
            SceneCurveSegment* segs  = (SceneCurveSegment*)malloc(n_segs * sizeof(*segs));
            SceneCurveAabb*    aabbs = (SceneCurveAabb*)malloc(n_segs * sizeof(*aabbs));
            int written = scene_curve_to_segments(c, segs, aabbs);
            if (written != n_segs) {
                fprintf(stderr, "    FAIL: expected %d segments, got %d\n", n_segs, written);
            }
            int n_show = (n_segs < 3) ? n_segs : 3;
            for (int s = 0; s < n_show; s++) {
                printf("    seg[%d]: p0=(%.3f,%.3f,%.3f) r0=%.3f -> p1=(%.3f,%.3f,%.3f) r1=%.3f\n",
                       s,
                       segs[s].p0[0], segs[s].p0[1], segs[s].p0[2], segs[s].r0,
                       segs[s].p1[0], segs[s].p1[1], segs[s].p1[2], segs[s].r1);
                printf("    aabb[%d]: (%.3f,%.3f,%.3f) - (%.3f,%.3f,%.3f)\n",
                       s,
                       aabbs[s].min_x, aabbs[s].min_y, aabbs[s].min_z,
                       aabbs[s].max_x, aabbs[s].max_y, aabbs[s].max_z);
            }
            if (n_segs > n_show) {
                printf("    ... and %d more\n", n_segs - n_show);
            }
            free(segs);
            free(aabbs);
        }
    }

    printf("TOTAL: %d curves, %d linear segments\n", scene->ncurves, total_segments);
    scene_free(scene);

    /* Round-trip through NuRenderer to verify the renderer-side curve
     * registry sees the same segment count. Render mode is irrelevant —
     * we never call nu_render. */
    NuRendererConfig cfg = {0};
    cfg.width  = 64;
    cfg.height = 64;
    cfg.enable_rt = 1;
    NuRenderer* r = nu_renderer_create(&cfg);
    if (!r) {
        fprintf(stderr, "FAIL: nu_renderer_create returned NULL\n");
        return 1;
    }
    int loaded = nu_load_usd(r, argv[1]);
    int rseg = nu_get_curve_segment_count(r);
    printf("RENDERER: nu_load_usd returned %d, curve segments = %d\n", loaded, rseg);
    if (rseg != total_segments) {
        fprintf(stderr, "FAIL: renderer extracted %d segments, scene loader extracted %d\n",
                rseg, total_segments);
        nu_renderer_destroy(r);
        return 1;
    }

    /* Force GPU upload via nu_build_accel — exercises gpu_upload_curve_data. */
    NuResult build_res = nu_build_accel(r);
    if (build_res != NU_OK) {
        fprintf(stderr, "FAIL: nu_build_accel returned %d: %s\n",
                build_res, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    printf("RENDERER: nu_build_accel OK (curve segments uploaded to GPU)\n");
    nu_renderer_destroy(r);

    printf("PASS: scene curves test\n");
    return 0;
}
