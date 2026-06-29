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
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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
            int written = scene_curve_to_segments(c, segs);
            if (written != n_segs) {
                fprintf(stderr, "    FAIL: expected %d segments, got %d\n", n_segs, written);
            }
            int n_show = (n_segs < 3) ? n_segs : 3;
            for (int s = 0; s < n_show; s++) {
                /* Phase 12.x: AABBs are GPU-generated at BLAS-build time
                 * from (p0, r0, p1). Test displays the same cylinder
                 * bounds the GPU shader will derive (constant-radius). */
                float r = segs[s].r0;
                float min_x = (segs[s].p0[0] < segs[s].p1[0] ? segs[s].p0[0] : segs[s].p1[0]) - r;
                float min_y = (segs[s].p0[1] < segs[s].p1[1] ? segs[s].p0[1] : segs[s].p1[1]) - r;
                float min_z = (segs[s].p0[2] < segs[s].p1[2] ? segs[s].p0[2] : segs[s].p1[2]) - r;
                float max_x = (segs[s].p0[0] > segs[s].p1[0] ? segs[s].p0[0] : segs[s].p1[0]) + r;
                float max_y = (segs[s].p0[1] > segs[s].p1[1] ? segs[s].p0[1] : segs[s].p1[1]) + r;
                float max_z = (segs[s].p0[2] > segs[s].p1[2] ? segs[s].p0[2] : segs[s].p1[2]) + r;
                printf("    seg[%d]: p0=(%.3f,%.3f,%.3f) r=%.3f -> p1=(%.3f,%.3f,%.3f) mat_flags=0x%08x\n",
                       s,
                       segs[s].p0[0], segs[s].p0[1], segs[s].p0[2], segs[s].r0,
                       segs[s].p1[0], segs[s].p1[1], segs[s].p1[2], segs[s].mat_flags);
                printf("    aabb[%d] (GPU-derived): (%.3f,%.3f,%.3f) - (%.3f,%.3f,%.3f)\n",
                       s, min_x, min_y, min_z, max_x, max_y, max_z);
            }
            if (n_segs > n_show) {
                printf("    ... and %d more\n", n_segs - n_show);
            }
            free(segs);
        }
    }

    printf("TOTAL: %d curves, %d linear segments\n", scene->ncurves, total_segments);

    /* Stash scene bounds for camera framing before freeing. */
    float scene_min[3] = { scene->bounds_min[0], scene->bounds_min[1], scene->bounds_min[2] };
    float scene_max[3] = { scene->bounds_max[0], scene->bounds_max[1], scene->bounds_max[2] };
    scene_free(scene);

    /* Round-trip through NuRenderer + render the scene end-to-end. */
    NuRendererConfig cfg = {0};
    cfg.width  = 512;
    cfg.height = 512;
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
    double t_build0 = now_sec();
    NuResult build_res = nu_build_accel(r);
    double t_build = now_sec() - t_build0;
    if (build_res != NU_OK) {
        fprintf(stderr, "FAIL: nu_build_accel returned %d: %s\n",
                build_res, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    printf("RENDERER: nu_build_accel OK in %.1f ms\n", t_build * 1000.0);

    /* Auto-frame camera on scene bounds. Look down + back at the
     * scene from above its center, far enough that the bounding
     * sphere fits the FOV. */
    float cx = 0.5f * (scene_min[0] + scene_max[0]);
    float cy = 0.5f * (scene_min[1] + scene_max[1]);
    float cz = 0.5f * (scene_min[2] + scene_max[2]);
    float ex = scene_max[0] - scene_min[0];
    float ey = scene_max[1] - scene_min[1];
    float ez = scene_max[2] - scene_min[2];
    float diag = sqrtf(ex*ex + ey*ey + ez*ez);
    if (diag < 1e-3f) diag = 1.0f;
    /* Distance: half-diag / tan(fov/2); fov=60° → tan=0.577 → dist≈0.87*diag.
     * We use 1.2x for headroom + offset so corners aren't clipped. */
    float dist = diag * 1.2f;
    NuCameraDesc cam = {0};
    cam.eye[0] = cx + dist * 0.5f;
    cam.eye[1] = cy + dist * 0.7f;
    cam.eye[2] = cz + dist * 0.7f;
    cam.target[0] = cx; cam.target[1] = cy; cam.target[2] = cz;
    cam.fov_degrees = 60.0f;
    cam.near_clip = 0.01f * diag;
    cam.far_clip  = 10.0f * diag;
    printf("CAMERA: eye=(%.2f, %.2f, %.2f) target=(%.2f, %.2f, %.2f) diag=%.2f\n",
           cam.eye[0], cam.eye[1], cam.eye[2],
           cam.target[0], cam.target[1], cam.target[2], diag);
    nu_set_camera(r, 0, &cam);

    /* Time first render (includes any lazy compilation). */
    double t_first0 = now_sec();
    NuResult render_res = nu_render(r, 0, NU_RENDER_RT);
    double t_first = now_sec() - t_first0;
    if (render_res != NU_OK) {
        fprintf(stderr, "FAIL: nu_render returned %d: %s\n",
                render_res, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }

    /* Compute steady-state ms/frame.
     *
     * The first version of this test divided wall-clock by 5, which
     * under-reported per-frame time because the GPU work was queued
     * deeper than 5 frames (Vulkan submits return immediately, and the
     * test program never blocked on GPU completion). Now we run a
     * larger batch (n_warm) and force a device-idle synchronisation
     * via nu_save_ppm at the end, so the wall-clock interval truly
     * encloses the GPU work for all n_warm frames. */
    int n_warm = 64;
    /* Save PPM with input filename as suffix. */
    const char* base = strrchr(argv[1], '/');
    base = base ? base + 1 : argv[1];
    char ppm_path[512];
    snprintf(ppm_path, sizeof(ppm_path), "render_%s.ppm", base);
    /* strip .usdc/.usda extension and replace */
    char* dot = strrchr(ppm_path, '.');
    if (dot) memcpy(dot, ".ppm", 5);

    /* Warm-up sync (drains any queued frames from t_first). */
    nu_save_ppm(r, ppm_path);

    double t_warm0 = now_sec();
    for (int i = 0; i < n_warm; i++) {
        nu_render(r, 0, NU_RENDER_RT);
    }
    /* nu_save_ppm calls vkDeviceWaitIdle internally — guarantees the
     * GPU has finished all n_warm frames before we stop the timer. */
    nu_save_ppm(r, ppm_path);
    double t_warm = now_sec() - t_warm0;
    double per_frame_ms = (t_warm / n_warm) * 1000.0;

    printf("RENDER: first=%.1f ms, steady=%.1f ms/frame (%.1f fps), gpu=%lu MB → %s\n",
           t_first * 1000.0, per_frame_ms, 1000.0 / per_frame_ms,
           (unsigned long)(nu_get_gpu_memory_used(r) / (1024 * 1024)),
           ppm_path);

    nu_renderer_destroy(r);

    printf("PASS: scene curves test\n");
    return 0;
}
