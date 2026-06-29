// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * test_headless_render.c — Smoke test for nusd_renderer.
 *
 * Usage: test_headless_render [path.usd] [output.ppm]
 *
 * If no USD path given, creates a simple triangle scene programmatically.
 */

#include "nusd_renderer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static int test_triangle(const char* output_path)
{
    printf("Test: rendering a triangle...\n");

    NuRendererConfig config = {0};
    config.width  = 800;
    config.height = 600;
    config.enable_rt = 0; /* raster only for simple test */

    NuRenderer* r = nu_renderer_create(&config);
    if (!r) {
        fprintf(stderr, "FAIL: nu_renderer_create returned NULL\n");
        return 1;
    }

    printf("  Renderer created (%dx%d, RT=%s)\n",
           config.width, config.height,
           nu_rt_available(r) ? "yes" : "no");

    /* Add a simple triangle */
    float positions[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f,
    };
    float normals[] = {
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
    };
    uint32_t indices[] = { 0, 1, 2 };

    NuMeshDesc desc = {0};
    desc.positions = positions;
    desc.normals   = normals;
    desc.indices   = indices;
    desc.nvertices = 3;
    desc.nindices  = 3;
    desc.display_color[0] = 0.2f;
    desc.display_color[1] = 0.6f;
    desc.display_color[2] = 1.0f;
    desc.name = "triangle";

    int mesh_id = nu_add_mesh(r, &desc);
    if (mesh_id < 0) {
        fprintf(stderr, "FAIL: nu_add_mesh returned %d\n", mesh_id);
        nu_renderer_destroy(r);
        return 1;
    }
    printf("  Added mesh '%s' (id=%d)\n", desc.name, mesh_id);

    /* Set camera looking at the triangle */
    NuCameraDesc cam = {0};
    cam.eye[0] = 0.0f; cam.eye[1] = 0.0f; cam.eye[2] = 3.0f;
    cam.target[0] = 0.0f; cam.target[1] = 0.0f; cam.target[2] = 0.0f;
    cam.fov_degrees = 45.0f;
    cam.near_clip = 0.01f;
    cam.far_clip = 100.0f;
    nu_set_camera(r, 0, &cam);

    /* Render raster frame */
    NuResult res = nu_render(r, 0, NU_RENDER_RASTER);
    if (res != NU_OK) {
        fprintf(stderr, "FAIL: nu_render returned %d: %s\n",
                res, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    printf("  Raster frame rendered\n");

    /* Save output */
    res = nu_save_ppm(r, output_path);
    if (res != NU_OK) {
        fprintf(stderr, "FAIL: nu_save_ppm returned %d: %s\n",
                res, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    printf("  Saved to %s\n", output_path);

    printf("  GPU memory used: %.1f MB\n",
           (double)nu_get_gpu_memory_used(r) / (1024.0 * 1024.0));

    nu_renderer_destroy(r);
    printf("PASS: triangle test\n");
    return 0;
}

static int test_usd_file(const char* usd_path, const char* output_path)
{
    printf("Test: rendering USD file %s...\n", usd_path);

    NuRendererConfig config = {0};
    config.width  = 1920;
    config.height = 1080;
    config.enable_rt = 1;

    double t0 = now_ms();
    NuRenderer* r = nu_renderer_create(&config);
    if (!r) {
        fprintf(stderr, "FAIL: nu_renderer_create returned NULL\n");
        return 1;
    }
    double t_create = now_ms();

    printf("  Renderer created (%dx%d, RT=%s)\n",
           config.width, config.height,
           nu_rt_available(r) ? "yes" : "no");

    int nmeshes = nu_load_usd(r, usd_path);
    if (nmeshes < 0) {
        fprintf(stderr, "FAIL: nu_load_usd returned %d: %s\n",
                nmeshes, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    double t_load = now_ms();
    printf("  Loaded %d meshes\n", nmeshes);

    /* Try RT first, fall back to raster */
    NuRenderMode mode = NU_RENDER_RT;
    if (!nu_rt_available(r)) {
        printf("  RT not available, falling back to raster\n");
        mode = NU_RENDER_RASTER;
    }

    NuResult res = nu_render(r, 0, mode);
    if (res != NU_OK) {
        fprintf(stderr, "FAIL: nu_render returned %d: %s\n",
                res, nu_get_last_error(r));
        /* Try raster fallback */
        if (mode == NU_RENDER_RT) {
            printf("  RT failed, trying raster...\n");
            res = nu_render(r, 0, NU_RENDER_RASTER);
        }
        if (res != NU_OK) {
            nu_renderer_destroy(r);
            return 1;
        }
    }
    double t_render = now_ms();
    printf("  Frame rendered (mode=%s)\n",
           mode == NU_RENDER_RT ? "RT" : "raster");

    res = nu_save_ppm(r, output_path);
    if (res != NU_OK) {
        fprintf(stderr, "FAIL: nu_save_ppm returned %d: %s\n",
                res, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    printf("  Saved to %s\n", output_path);

    /* Test nu_fetch_pixels (direct readback, no temp file) */
    {
        int w = config.width, h = config.height;
        uint8_t* pixels = (uint8_t*)malloc((size_t)w * h * 4);
        if (!pixels) {
            fprintf(stderr, "FAIL: malloc for pixel readback\n");
            nu_renderer_destroy(r);
            return 1;
        }
        res = nu_fetch_pixels(r, pixels, NU_PIXEL_RGBA8);
        if (res != NU_OK) {
            fprintf(stderr, "FAIL: nu_fetch_pixels returned %d: %s\n",
                    res, nu_get_last_error(r));
            free(pixels);
            nu_renderer_destroy(r);
            return 1;
        }
        /* Sanity check: some non-zero pixel data */
        int has_nonzero = 0;
        for (int i = 0; i < w * h * 4 && !has_nonzero; i++)
            if (pixels[i]) has_nonzero = 1;
        if (!has_nonzero) {
            fprintf(stderr, "FAIL: nu_fetch_pixels returned all-zero buffer\n");
            free(pixels);
            nu_renderer_destroy(r);
            return 1;
        }
        printf("  nu_fetch_pixels: OK (direct readback verified)\n");
        free(pixels);
    }
    double t_fetch = now_ms();

    printf("  GPU memory used: %.1f MB\n",
           (double)nu_get_gpu_memory_used(r) / (1024.0 * 1024.0));

    nu_renderer_destroy(r);
    double t_destroy = now_ms();

    /* Phase profile (iter 43): where the headless render's wall time goes.
     * nu_load_usd includes the geo-cache scene load + BLAS/TLAS + RT setup. */
    printf("  [phases ms] renderer_create %.0f, load_usd %.0f, render %.0f, "
           "save+fetch %.0f, destroy %.0f, total %.0f\n",
           t_create - t0, t_load - t_create, t_render - t_load,
           t_fetch - t_render, t_destroy - t_fetch, t_destroy - t0);

    printf("PASS: USD file test\n");
    return 0;
}

int main(int argc, char** argv)
{
    const char* usd_path = NULL;
    const char* output_path = "render_output.ppm";

    if (argc >= 2) usd_path = argv[1];
    if (argc >= 3) output_path = argv[2];

    if (usd_path) {
        return test_usd_file(usd_path, output_path);
    } else {
        return test_triangle(output_path);
    }
}
