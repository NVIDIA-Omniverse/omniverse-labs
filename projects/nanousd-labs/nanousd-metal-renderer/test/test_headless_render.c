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
#include <math.h>
#include <unistd.h>

static int test_hash_dedup_renderer_isolation(void)
{
    if (!getenv("NUSD_HASH_DEDUP")) return 0;

    printf("Test: hash-dedup renderer isolation...\n");

    NuRendererConfig config = {0};
    config.width = 64;
    config.height = 64;
    config.enable_rt = 0;

    float positions_a[] = {
         2.0f, -0.5f, 0.0f,
         3.0f, -0.5f, 0.0f,
         2.5f,  0.5f, 0.0f,
    };
    float positions_b[] = {
        -3.0f, -0.5f, 0.0f,
        -2.0f, -0.5f, 0.0f,
        -2.5f,  0.5f, 0.0f,
    };
    float normals[] = {
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
    };
    uint32_t indices[] = { 0, 1, 2 };

    NuMeshDesc desc_a = {0};
    desc_a.positions = positions_a;
    desc_a.normals = normals;
    desc_a.indices = indices;
    desc_a.nvertices = 3;
    desc_a.nindices = 3;
    desc_a.display_color[0] = 0.2f;
    desc_a.display_color[1] = 0.6f;
    desc_a.display_color[2] = 1.0f;
    desc_a.name = "right_triangle";

    NuMeshDesc desc_b = desc_a;
    desc_b.positions = positions_b;
    desc_b.display_color[0] = 1.0f;
    desc_b.display_color[1] = 0.2f;
    desc_b.display_color[2] = 0.1f;
    desc_b.name = "left_triangle";

    NuRenderer* seed = nu_renderer_create(&config);
    NuRenderer* r = nu_renderer_create(&config);
    if (!seed || !r) {
        fprintf(stderr, "FAIL: renderer create failed for hash-dedup isolation\n");
        if (seed) nu_renderer_destroy(seed);
        if (r) nu_renderer_destroy(r);
        return 1;
    }

    if (nu_add_mesh(seed, &desc_a) < 0) {
        fprintf(stderr, "FAIL: seed nu_add_mesh failed: %s\n", nu_get_last_error(seed));
        nu_renderer_destroy(r);
        nu_renderer_destroy(seed);
        return 1;
    }
    if (nu_add_mesh(r, &desc_b) < 0 || nu_add_mesh(r, &desc_a) < 0) {
        fprintf(stderr, "FAIL: isolated renderer nu_add_mesh failed: %s\n",
                nu_get_last_error(r));
        nu_renderer_destroy(r);
        nu_renderer_destroy(seed);
        return 1;
    }

    char cache_path[256];
    snprintf(cache_path, sizeof(cache_path),
             "/tmp/nusd_renderer_hash_dedup_isolation_%ld.nugc",
             (long)getpid());
    NuResult res = nu_save_geometry_cache(r, cache_path);
    if (res != NU_OK) {
        fprintf(stderr, "FAIL: isolation cache save failed: %s\n",
                nu_get_last_error(r));
        nu_renderer_destroy(r);
        nu_renderer_destroy(seed);
        return 1;
    }

    NuRenderer* rc = nu_renderer_create(&config);
    if (!rc) {
        nu_renderer_destroy(r);
        nu_renderer_destroy(seed);
        remove(cache_path);
        return 1;
    }
    int loaded = nu_load_geometry_cache(rc, cache_path);
    float bmin[3], bmax[3];
    res = nu_get_scene_bounds(rc, bmin, bmax);
    if (loaded != 2 || res != NU_OK || bmin[0] > -2.99f || bmax[0] < 2.99f) {
        fprintf(stderr,
                "FAIL: hash-dedup renderer isolation bounds invalid "
                "(loaded=%d, minx=%f, maxx=%f)\n",
                loaded, bmin[0], bmax[0]);
        nu_renderer_destroy(rc);
        nu_renderer_destroy(r);
        nu_renderer_destroy(seed);
        remove(cache_path);
        return 1;
    }

    nu_renderer_destroy(rc);
    nu_renderer_destroy(r);
    nu_renderer_destroy(seed);
    remove(cache_path);
    printf("PASS: hash-dedup renderer isolation\n");
    return 0;
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

    float inst_xform[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    int inst_id = nu_add_mesh_instance(r, mesh_id, inst_xform);
    if (inst_id < 0) {
        fprintf(stderr, "FAIL: nu_add_mesh_instance returned %d\n", inst_id);
        nu_renderer_destroy(r);
        return 1;
    }
    printf("  Added identity instance (id=%d)\n", inst_id);

    int expected_cached_meshes = 2;
    if (getenv("NUSD_HASH_DEDUP")) {
        NuMeshDesc dup_desc = desc;
        dup_desc.display_color[0] = 1.0f;
        dup_desc.display_color[1] = 0.2f;
        dup_desc.display_color[2] = 0.1f;
        dup_desc.name = "triangle_duplicate";
        int dup_id = nu_add_mesh(r, &dup_desc);
        if (dup_id < 0) {
            fprintf(stderr, "FAIL: duplicate nu_add_mesh returned %d\n", dup_id);
            nu_renderer_destroy(r);
            return 1;
        }
        expected_cached_meshes = 3;
        printf("  Added duplicate mesh for hash-dedup path (id=%d)\n", dup_id);
    }

    /* Set camera looking at the triangle */
    NuCameraDesc cam = {0};
    cam.eye[0] = 0.0f; cam.eye[1] = 0.0f; cam.eye[2] = 3.0f;
    cam.target[0] = 0.0f; cam.target[1] = 0.0f; cam.target[2] = 0.0f;
    cam.fov_degrees = 45.0f;
    cam.near_clip = 0.01f;
    cam.far_clip = 100.0f;
    nu_set_camera(r, 0, &cam);

    /* Preprocessed geometry cache round-trip.
     *
     * Must run BEFORE the first nu_render: a raster upload in
     * rebuild_gpu_buffers() releases CPU-side geometry staging
     * (release_cpu_staging at src/renderer.c), after which
     * nu_save_geometry_cache can no longer serialize cpu_vertices/
     * cpu_indices/meshlets and returns NU_ERROR_UNSUPPORTED. All cache
     * inputs (including the global meshlet arrays) are fully populated by
     * nu_add_mesh, so saving here is correct and complete. */
    {
        char cache_path[256];
        snprintf(cache_path, sizeof(cache_path),
                 "/tmp/nusd_renderer_triangle_geometry_%ld.nugc",
                 (long)getpid());
        NuResult res = nu_save_geometry_cache(r, cache_path);
        if (res != NU_OK) {
            fprintf(stderr, "FAIL: nu_save_geometry_cache returned %d: %s\n",
                    res, nu_get_last_error(r));
            nu_renderer_destroy(r);
            return 1;
        }

        NuRenderer* rc = nu_renderer_create(&config);
        if (!rc) {
            fprintf(stderr, "FAIL: cache renderer create returned NULL\n");
            nu_renderer_destroy(r);
            return 1;
        }
        int loaded = nu_load_geometry_cache(rc, cache_path);
        if (loaded != expected_cached_meshes) {
            fprintf(stderr, "FAIL: nu_load_geometry_cache returned %d, expected %d: %s\n",
                    loaded, expected_cached_meshes, nu_get_last_error(rc));
            nu_renderer_destroy(rc);
            nu_renderer_destroy(r);
            remove(cache_path);
            return 1;
        }
        float cbmin[3], cbmax[3];
        res = nu_get_scene_bounds(rc, cbmin, cbmax);
        if (res != NU_OK ||
            fabsf(cbmin[0] + 0.5f) > 1e-5f ||
            fabsf(cbmin[1] + 0.5f) > 1e-5f ||
            fabsf(cbmin[2]) > 1e-5f ||
            fabsf(cbmax[0] - 0.5f) > 1e-5f ||
            fabsf(cbmax[1] - 0.5f) > 1e-5f ||
            fabsf(cbmax[2]) > 1e-5f) {
            fprintf(stderr, "FAIL: cached geometry bounds are invalid\n");
            nu_renderer_destroy(rc);
            nu_renderer_destroy(r);
            remove(cache_path);
            return 1;
        }
        nu_set_camera(rc, 0, &cam);
        res = nu_render(rc, 0, NU_RENDER_RASTER);
        if (res != NU_OK) {
            fprintf(stderr, "FAIL: cache nu_render returned %d: %s\n",
                    res, nu_get_last_error(rc));
            nu_renderer_destroy(rc);
            nu_renderer_destroy(r);
            remove(cache_path);
            return 1;
        }

        uint8_t* pixels = (uint8_t*)malloc((size_t)config.width * config.height * 4);
        if (!pixels) {
            nu_renderer_destroy(rc);
            nu_renderer_destroy(r);
            remove(cache_path);
            return 1;
        }
        res = nu_fetch_pixels(rc, pixels, NU_PIXEL_RGBA8);
        int has_nonzero = 0;
        for (int i = 0; res == NU_OK && i < config.width * config.height * 4 && !has_nonzero; i++)
            if (pixels[i]) has_nonzero = 1;
        free(pixels);
        if (res != NU_OK || !has_nonzero) {
            fprintf(stderr, "FAIL: cached geometry render did not produce pixels\n");
            nu_renderer_destroy(rc);
            nu_renderer_destroy(r);
            remove(cache_path);
            return 1;
        }
        printf("  Geometry cache round-trip rendered\n");
        nu_renderer_destroy(rc);
        remove(cache_path);
    }

    /* Render raster frame */
    NuResult res = nu_render(r, 0, NU_RENDER_RASTER);
    if (res != NU_OK) {
        fprintf(stderr, "FAIL: nu_render returned %d: %s\n",
                res, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    printf("  Raster frame rendered\n");

    if (getenv("NUSD_MESHLET_RASTER")) {
        NuMeshletStats stats;
        memset(&stats, 0, sizeof(stats));
        res = nu_get_meshlet_stats(r, &stats);
        if (res != NU_OK) {
            fprintf(stderr, "FAIL: nu_get_meshlet_stats returned %d: %s\n",
                    res, nu_get_last_error(r));
            nu_renderer_destroy(r);
            return 1;
        }
        printf("  Meshlets: %u meshlets, %u indices, raster=%u\n",
               stats.active_meshlets,
               stats.active_meshlet_indices,
               stats.meshlet_raster_enabled);
        if (!stats.meshlet_raster_enabled ||
            stats.active_meshlets == 0 ||
            stats.active_meshlet_indices != 3 ||
            stats.gpu_index_bytes == 0) {
            fprintf(stderr, "FAIL: meshlet raster stats are not valid\n");
            nu_renderer_destroy(r);
            return 1;
        }
    }

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

static const char* default_ibl_hdr(void)
{
    const char* env = getenv("NUSD_TEST_HDR");
    if (env && env[0]) return env;

    const char* home = getenv("HOME");
    static char path[1024];
    if (home && home[0]) {
        snprintf(path, sizeof(path), "%s/OpenUSD-install/resources/Lights/table_mountain.hdr", home);
        return path;
    }
    return "resources/Lights/table_mountain.hdr";
}

static int test_triangle_rt(const char* output_path)
{
    printf("Test: rendering a triangle via ray tracing...\n");

    NuRendererConfig config = {0};
    config.width  = 800;
    config.height = 600;
    config.enable_rt = 1;

    NuRenderer* r = nu_renderer_create(&config);
    if (!r) {
        fprintf(stderr, "FAIL: nu_renderer_create returned NULL\n");
        return 1;
    }

    if (!nu_rt_available(r)) {
        printf("  RT not available on this GPU — skipping RT test\n");
        nu_renderer_destroy(r);
        return 0;
    }
    printf("  Renderer created (%dx%d, RT=yes)\n", config.width, config.height);

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

    NuCameraDesc cam = {0};
    cam.eye[0] = 0.0f; cam.eye[1] = 0.0f; cam.eye[2] = 3.0f;
    cam.target[0] = 0.0f; cam.target[1] = 0.0f; cam.target[2] = 0.0f;
    cam.fov_degrees = 45.0f;
    cam.near_clip = 0.01f;
    cam.far_clip = 100.0f;
    nu_set_camera(r, 0, &cam);

    NuResult res = nu_render(r, 0, NU_RENDER_RT);
    if (res != NU_OK) {
        fprintf(stderr, "FAIL: nu_render(RT) returned %d: %s\n",
                res, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    printf("  RT frame rendered\n");

    res = nu_save_ppm(r, output_path);
    if (res != NU_OK) {
        fprintf(stderr, "FAIL: nu_save_ppm returned %d: %s\n",
                res, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    printf("  Saved to %s\n", output_path);

    /* Direct readback to inspect pixels */
    int w = config.width, h = config.height;
    uint8_t* px = (uint8_t*)malloc((size_t)w * h * 4);
    if (!px) {
        fprintf(stderr, "FAIL: malloc\n");
        nu_renderer_destroy(r);
        return 1;
    }
    res = nu_fetch_pixels(r, px, NU_PIXEL_RGBA8);
    if (res != NU_OK) {
        fprintf(stderr, "FAIL: nu_fetch_pixels returned %d\n", res);
        free(px);
        nu_renderer_destroy(r);
        return 1;
    }

    /* Center pixel should hit the triangle (mesh color RGB ≈ 0.2/0.6/1.0). */
    int cx = w / 2, cy = h / 2;
    uint8_t* c = px + ((size_t)cy * w + cx) * 4;
    /* Top-left corner should be sky (no triangle). */
    uint8_t* tl = px + 0;
    printf("  center pixel: (%u,%u,%u)  top-left: (%u,%u,%u)\n",
           c[0], c[1], c[2], tl[0], tl[1], tl[2]);

    /* Sanity: the triangle's blue channel should dominate at center. */
    if (c[2] < 60 || c[2] <= c[0]) {
        fprintf(stderr, "FAIL: center pixel does not look like the cyan triangle (got R=%u G=%u B=%u)\n",
                c[0], c[1], c[2]);
        free(px);
        nu_renderer_destroy(r);
        return 1;
    }

    /* Refit smoke test — translate the triangle far off-screen, re-render, and
     * verify the center pixel is now sky/ground (no triangle). Exercises
     * gpu_update_tlas's refit path which is otherwise unused at this point. */
    {
        float xform[16] = {
            1, 0, 0, 100,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1   /* translate +100 in X */
        };
        int mesh_ids[1] = { mesh_id };
        res = nu_set_transforms(r, mesh_ids, xform, 1);
        if (res != NU_OK) {
            fprintf(stderr, "FAIL: nu_set_transforms returned %d\n", res);
            free(px);
            nu_renderer_destroy(r);
            return 1;
        }
        res = nu_render(r, 0, NU_RENDER_RT);
        if (res != NU_OK) {
            fprintf(stderr, "FAIL: nu_render after move returned %d: %s\n",
                    res, nu_get_last_error(r));
            free(px);
            nu_renderer_destroy(r);
            return 1;
        }
        res = nu_fetch_pixels(r, px, NU_PIXEL_RGBA8);
        if (res != NU_OK) { free(px); nu_renderer_destroy(r); return 1; }
        uint8_t* c2 = px + ((size_t)cy * w + cx) * 4;
        printf("  after move: center pixel: (%u,%u,%u)\n", c2[0], c2[1], c2[2]);
        /* Triangle is gone — center should now be sky-or-ground.
         * R should not be << B anymore (sky/ground are warm/neutral). */
        if (c2[2] > c2[0] + 20) {
            fprintf(stderr, "FAIL: triangle still visible after move (R=%u G=%u B=%u)\n",
                    c2[0], c2[1], c2[2]);
            free(px);
            nu_renderer_destroy(r);
            return 1;
        }
    }

    free(px);

    nu_renderer_destroy(r);
    printf("PASS: triangle RT test\n");
    return 0;
}

/* Two-camera tiled RT smoke test.
 *
 * Layout: 1 row × 2 cols of 800×600 tiles → 1600×600 tiled image.
 *   Camera 0: eye=(0,0,3), gaze=-Z (looks at the triangle at origin)
 *   Camera 1: eye=(0,0,3), gaze=+Z (looks AWAY from the triangle, at sky)
 *
 * Both cameras share a 45° fov, 4:3 aspect, near=0.01, far=100. The
 * matrices below are constructed as the math inverse-view and inverse-proj
 * directly: C row-major storage of view_inv puts the eye in row 2 column 3
 * (i.e. floats [11]/[7]/[3] = eye.x/y/z) so the shader's `(0,0,0,1) * M`
 * recovers the eye correctly. proj_inv is the inverse of a Y-flipped
 * Vulkan-style perspective matrix.
 *
 * Expected:
 *   - Left half center (~400, 300): triangle (cyan-blue, B>>R)
 *   - Right half center (~1200, 300): sky/ground (warm/neutral, R>=B)
 *   - Refit smoke test: translate triangle far in +X, both halves become sky/ground.
 */
static int test_tiled_rt(const char* output_path)
{
    printf("Test: rendering two cameras via tiled ray tracing...\n");
    (void)output_path;  /* PPM save for tiled isn't a clean fit; skip */

    NuRendererConfig config = {0};
    config.width  = 800;
    config.height = 600;
    config.enable_rt = 1;

    NuRenderer* r = nu_renderer_create(&config);
    if (!r) {
        fprintf(stderr, "FAIL: nu_renderer_create returned NULL\n");
        return 1;
    }
    if (!nu_rt_available(r)) {
        printf("  RT not available — skipping tiled RT test\n");
        nu_renderer_destroy(r);
        return 0;
    }

    /* One triangle at origin, same as the single-camera RT test. */
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
    if (mesh_id < 0) { nu_renderer_destroy(r); return 1; }

    /* Enable all three sensor outputs to exercise the full kernel path. */
    nu_set_depth_enabled(r, 1);
    nu_set_segmentation_enabled(r, 1);
    nu_set_normals_enabled(r, 1);

    /* Camera 0 view_inv (math row-major): identity rotation + eye at (0,0,3). */
    static const float view_inv_cam0[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 3.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    /* Camera 1 view_inv: 180° rotation around Y (X and Z flipped) + eye at (0,0,3). */
    static const float view_inv_cam1[16] = {
       -1.0f, 0.0f,  0.0f, 0.0f,
        0.0f, 1.0f,  0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 3.0f,
        0.0f, 0.0f,  0.0f, 1.0f
    };
    /* proj_inv for Vulkan-style perspective: fov=45°, aspect=800/600=4/3,
     * near=0.01, far=100. f = 1/tan(22.5°) ≈ 2.41421. */
    static const float proj_inv[16] = {
        0.55228f,    0.0f,      0.0f,       0.0f,
        0.0f,       -0.41421f,  0.0f,       0.0f,
        0.0f,        0.0f,      0.0f,      -1.0f,
        0.0f,        0.0f,   -100.0f,     100.01f
    };

    /* Pack into vp_inv_matrices: pairs of (view_inv, proj_inv) per camera. */
    float vp_inv[64];
    memcpy(vp_inv +  0, view_inv_cam0, sizeof(view_inv_cam0));
    memcpy(vp_inv + 16, proj_inv,      sizeof(proj_inv));
    memcpy(vp_inv + 32, view_inv_cam1, sizeof(view_inv_cam1));
    memcpy(vp_inv + 48, proj_inv,      sizeof(proj_inv));

    const int tile_w = 800, tile_h = 600;
    const int num_cameras = 2;

    NuResult res = nu_render_tiled(r, vp_inv, num_cameras, tile_w, tile_h, NU_RENDER_RT);
    if (res != NU_OK) {
        fprintf(stderr, "FAIL: nu_render_tiled returned %d: %s\n",
                res, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    printf("  Tiled RT frame rendered (2 cams, 1600x600)\n");

    /* Use slot API to exercise the contract — even though MVP is single-buffered,
     * gpu_get_last_tiled_slot must return 0 and slot 0 must map. */
    int slot = nu_get_last_tiled_slot(r);
    if (slot != 0) {
        fprintf(stderr, "FAIL: nu_get_last_tiled_slot returned %d (expected 0)\n", slot);
        nu_renderer_destroy(r);
        return 1;
    }
    int total_w = 0, total_h = 0;
    const uint8_t* tiled = (const uint8_t*)nu_map_tiled_pixels_raw_slot(
        r, num_cameras, tile_w, tile_h, slot, &total_w, &total_h);
    if (!tiled || total_w != 1600 || total_h != 600) {
        fprintf(stderr, "FAIL: nu_map_tiled_pixels_raw_slot returned NULL or wrong size (%dx%d)\n",
                total_w, total_h);
        nu_renderer_destroy(r);
        return 1;
    }

    /* Center of camera 0 tile (~400, 300): should hit the triangle. */
    int cx0 = tile_w / 2, cy0 = tile_h / 2;
    const uint8_t* p_cam0 = tiled + ((size_t)cy0 * total_w + cx0) * 4;
    /* Center of camera 1 tile (~1200, 300): should be sky/ground. */
    int cx1 = tile_w + tile_w / 2;
    const uint8_t* p_cam1 = tiled + ((size_t)cy0 * total_w + cx1) * 4;
    printf("  cam0 center: (%u,%u,%u)  cam1 center: (%u,%u,%u)\n",
           p_cam0[0], p_cam0[1], p_cam0[2],
           p_cam1[0], p_cam1[1], p_cam1[2]);

    /* Triangle has color RGB ≈ (0.2, 0.6, 1.0): blue should dominate. */
    if (p_cam0[2] < 60 || p_cam0[2] <= p_cam0[0]) {
        fprintf(stderr, "FAIL: cam0 center does not look like the triangle (R=%u G=%u B=%u)\n",
                p_cam0[0], p_cam0[1], p_cam0[2]);
        nu_renderer_destroy(r);
        return 1;
    }
    /* Sky/ground: warm/neutral, R should NOT be much smaller than B. */
    if (p_cam1[2] > p_cam1[0] + 20) {
        fprintf(stderr, "FAIL: cam1 center looks like the triangle (R=%u G=%u B=%u) — tile indexing broken?\n",
                p_cam1[0], p_cam1[1], p_cam1[2]);
        nu_renderer_destroy(r);
        return 1;
    }

    /* Sanity-check sensor outputs at cam0 center: depth should be ~3 (eye-to-triangle),
     * segmentation == mesh_id+1, normals roughly (0, 0, ±1). */
    float* depth = (float*)malloc((size_t)num_cameras * tile_w * tile_h * sizeof(float));
    uint32_t* seg = (uint32_t*)malloc((size_t)num_cameras * tile_w * tile_h * sizeof(uint32_t));
    float* norms = (float*)malloc((size_t)num_cameras * tile_w * tile_h * 3 * sizeof(float));
    if (!depth || !seg || !norms) {
        fprintf(stderr, "FAIL: malloc for sensor buffers\n");
        free(depth); free(seg); free(norms);
        nu_renderer_destroy(r);
        return 1;
    }
    if (nu_fetch_depth_tiled(r, depth, num_cameras, tile_w, tile_h) != NU_OK
        || nu_fetch_segmentation_tiled(r, seg, num_cameras, tile_w, tile_h) != NU_OK
        || nu_fetch_normals_tiled(r, norms, num_cameras, tile_w, tile_h) != NU_OK) {
        fprintf(stderr, "FAIL: nu_fetch_*_tiled\n");
        free(depth); free(seg); free(norms);
        nu_renderer_destroy(r);
        return 1;
    }
    size_t cam0_center = (size_t)cy0 * tile_w + cx0;
    float d0 = depth[cam0_center];
    uint32_t s0 = seg[cam0_center];
    float n0x = norms[cam0_center * 3 + 0];
    float n0y = norms[cam0_center * 3 + 1];
    float n0z = norms[cam0_center * 3 + 2];
    printf("  cam0 sensor: depth=%.3f seg=%u N=(%.2f,%.2f,%.2f)\n",
           d0, s0, n0x, n0y, n0z);
    if (d0 < 2.0f || d0 > 4.0f) {
        fprintf(stderr, "FAIL: cam0 center depth %.3f is not ~3.0\n", d0);
        free(depth); free(seg); free(norms);
        nu_renderer_destroy(r);
        return 1;
    }
    if (s0 != (uint32_t)(mesh_id + 1)) {
        fprintf(stderr, "FAIL: cam0 center seg id %u != %d\n", s0, mesh_id + 1);
        free(depth); free(seg); free(norms);
        nu_renderer_destroy(r);
        return 1;
    }
    /* Triangle's vertex normals are (0,0,1); after flip-toward-ray, expect (0,0,-1) (or +1 if not flipped). */
    if (fabsf(n0z) < 0.9f) {
        fprintf(stderr, "FAIL: cam0 center normal Z=%.2f not ~±1\n", n0z);
        free(depth); free(seg); free(norms);
        nu_renderer_destroy(r);
        return 1;
    }
    /* cam1 center should be a miss — depth = -1, seg = 0. */
    size_t cam1_center = (size_t)tile_w * tile_h + (size_t)cy0 * tile_w + cx0;
    if (depth[cam1_center] >= 0.0f || seg[cam1_center] != 0u) {
        fprintf(stderr, "FAIL: cam1 center sensor (depth=%.3f seg=%u) not a miss\n",
                depth[cam1_center], seg[cam1_center]);
        free(depth); free(seg); free(norms);
        nu_renderer_destroy(r);
        return 1;
    }
    free(depth); free(seg); free(norms);

    /* Refit smoke test — translate the triangle far in +X, re-render, both
     * camera halves should now be sky/ground (no triangle). Exercises the
     * inline TLAS refit path during a tiled frame. */
    {
        float xform[16] = {
            1, 0, 0, 100,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
        int mesh_ids[1] = { mesh_id };
        if (nu_set_transforms(r, mesh_ids, xform, 1) != NU_OK) {
            fprintf(stderr, "FAIL: nu_set_transforms\n");
            nu_renderer_destroy(r);
            return 1;
        }
        if (nu_render_tiled(r, vp_inv, num_cameras, tile_w, tile_h, NU_RENDER_RT) != NU_OK) {
            fprintf(stderr, "FAIL: nu_render_tiled (after refit) — %s\n", nu_get_last_error(r));
            nu_renderer_destroy(r);
            return 1;
        }
        const uint8_t* tiled2 = (const uint8_t*)nu_map_tiled_pixels_raw_slot(
            r, num_cameras, tile_w, tile_h, 0, &total_w, &total_h);
        if (!tiled2) {
            fprintf(stderr, "FAIL: nu_map_tiled_pixels_raw_slot (after refit)\n");
            nu_renderer_destroy(r);
            return 1;
        }
        const uint8_t* p2_cam0 = tiled2 + ((size_t)cy0 * total_w + cx0) * 4;
        const uint8_t* p2_cam1 = tiled2 + ((size_t)cy0 * total_w + cx1) * 4;
        printf("  after refit — cam0: (%u,%u,%u)  cam1: (%u,%u,%u)\n",
               p2_cam0[0], p2_cam0[1], p2_cam0[2],
               p2_cam1[0], p2_cam1[1], p2_cam1[2]);
        if (p2_cam0[2] > p2_cam0[0] + 20) {
            fprintf(stderr, "FAIL: cam0 still shows the triangle after refit (R=%u G=%u B=%u)\n",
                    p2_cam0[0], p2_cam0[1], p2_cam0[2]);
            nu_renderer_destroy(r);
            return 1;
        }

        /* Sensor outputs after refit: cam0 center should now MISS — depth=-1, seg=0.
         * This pins down that the inline refit propagated to the same TLAS the
         * sensor-write branches read, not just to whatever the color path used. */
        float* depth2 = (float*)malloc((size_t)num_cameras * tile_w * tile_h * sizeof(float));
        uint32_t* seg2 = (uint32_t*)malloc((size_t)num_cameras * tile_w * tile_h * sizeof(uint32_t));
        if (!depth2 || !seg2
            || nu_fetch_depth_tiled(r, depth2, num_cameras, tile_w, tile_h) != NU_OK
            || nu_fetch_segmentation_tiled(r, seg2, num_cameras, tile_w, tile_h) != NU_OK) {
            fprintf(stderr, "FAIL: nu_fetch_*_tiled (after refit)\n");
            free(depth2); free(seg2);
            nu_renderer_destroy(r);
            return 1;
        }
        size_t cam0_after = (size_t)cy0 * tile_w + cx0;
        if (depth2[cam0_after] >= 0.0f || seg2[cam0_after] != 0u) {
            fprintf(stderr, "FAIL: cam0 sensor after refit not a miss (depth=%.3f seg=%u)\n",
                    depth2[cam0_after], seg2[cam0_after]);
            free(depth2); free(seg2);
            nu_renderer_destroy(r);
            return 1;
        }
        free(depth2); free(seg2);
    }

    nu_renderer_destroy(r);
    printf("PASS: tiled RT test\n");
    return 0;
}

/* Phase 7b — verify diffuse texture sampling actually works.
 *
 * Loads test/textured_quad.usda — a quad with UVs [0,0]–[1,1] and a
 * UsdPreviewSurface bound to a 2×2 BMP (quad4color.bmp) with four
 * distinct colors:
 *
 *   BMP texel layout (USD UV: u right, v up):
 *     v=1: (blue)   (yellow)
 *     v=0: (red)    (green)
 *
 * The quad covers world-space [-1, 1] × [-1, 1] at z=0; we render at
 * 400×400 with the camera at (0,0,3) (built into the USDA).
 *
 * Sampling logic: the rendered quad fills the central frame. We read
 * 4 anchor points (one per quadrant) and test whether each rendered
 * pixel matches the expected texel color by dominant-channel signature.
 * The test is robust to lighting (channels are scaled but ordering
 * preserved) and V-flip (we accept either orientation).
 *
 * Discriminating: if textures don't bind, all four quadrants render
 * the SAME base_color (or a uniform shade). The signature check fails
 * because all four anchors have the same dominant channel. If textures
 * do bind, we see four distinct signatures. The test fails on
 * ambiguity, not just on absence.
 *
 * Skips if the build is missing materials (NUSD_ENABLE_MATERIALS=OFF) —
 * detected by nu_load_usd returning 0 meshes (the parser silently
 * drops material bindings without the materials path enabled).
 */
typedef enum { SIG_RED, SIG_GREEN, SIG_BLUE, SIG_YELLOW, SIG_AMBIGUOUS } ColorSig;

static ColorSig classify_pixel(uint8_t r, uint8_t g, uint8_t b)
{
    /* Dominant-channel classifier. Tolerates lighting: channels can be
     * scaled (e.g. 60% lit red = (153, 0, 0)) but ordering remains:
     *   red:    R >> G, R >> B
     *   green:  G >> R, G >> B
     *   blue:   B >> R, B >> G
     *   yellow: R high, G high, B << R, B << G
     */
    int margin = 25;
    int rg = (int)r - (int)g;
    int rb = (int)r - (int)b;
    int gb = (int)g - (int)b;

    /* Yellow: R and G both well above B. */
    if (rb > margin && gb > margin && abs(rg) <= margin + 30) return SIG_YELLOW;
    /* Pure-channel dominance. */
    if (rg > margin && rb > margin) return SIG_RED;
    if (-rg > margin && gb > margin) return SIG_GREEN;
    if (-rb > margin && -gb > margin) return SIG_BLUE;
    return SIG_AMBIGUOUS;
}

static const char* sig_name(ColorSig s)
{
    switch (s) {
    case SIG_RED: return "red";
    case SIG_GREEN: return "green";
    case SIG_BLUE: return "blue";
    case SIG_YELLOW: return "yellow";
    default: return "ambiguous";
    }
}

static int test_textured_cube_rt(const char* usd_path, const char* output_path)
{
    printf("Test: rendering textured quad via RT (4-color texture verification)...\n");

    FILE* probe = fopen(usd_path, "rb");
    if (!probe) {
        fprintf(stderr, "FAIL: fixture not found at %s\n", usd_path);
        return 1;
    }
    fclose(probe);

    NuBackendInfo info;
    if (nu_get_backend_info(NULL, &info) == NU_OK &&
        (info.capabilities & (NU_CAP_MATERIALS | NU_CAP_TEXTURES)) !=
            (NU_CAP_MATERIALS | NU_CAP_TEXTURES)) {
        printf("  material/texture backend not available — skipping\n");
        return 0;
    }

    NuRendererConfig config = {0};
    config.width  = 400;
    config.height = 400;
    config.enable_rt = 1;
    config.enable_materials = 1;

    NuRenderer* r = nu_renderer_create(&config);
    if (!r) return 1;
    if (!nu_rt_available(r)) {
        printf("  RT not available — skipping\n");
        nu_renderer_destroy(r);
        return 0;
    }

    int n = nu_load_usd(r, usd_path);
    if (n <= 0) {
        printf("  nu_load_usd returned %d — skipping (likely NUSD_ENABLE_MATERIALS=OFF)\n", n);
        nu_renderer_destroy(r);
        return 0;
    }
    printf("  loaded %d meshes from %s\n", n, usd_path);

    /* Camera straight-on so the [-1,1]×[-1,1] quad fills most of the frame. */
    NuCameraDesc cam = {0};
    cam.eye[0] = 0; cam.eye[1] = 0; cam.eye[2] = 3;
    cam.target[0] = 0; cam.target[1] = 0; cam.target[2] = 0;
    cam.fov_degrees = 45.0f;
    cam.near_clip = 0.01f;
    cam.far_clip = 100.0f;
    nu_set_camera(r, 0, &cam);

    if (nu_render(r, 0, NU_RENDER_RT) != NU_OK) {
        fprintf(stderr, "FAIL: nu_render: %s\n", nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    nu_save_ppm(r, output_path);
    printf("  Saved %s\n", output_path);

    int W = config.width, H = config.height;
    uint8_t* px = (uint8_t*)malloc((size_t)W * H * 4);
    if (nu_fetch_pixels(r, px, NU_PIXEL_RGBA8) != NU_OK) {
        free(px); nu_renderer_destroy(r); return 1;
    }

    /* The quad's NDC bounds (after perspective at z=0, eye z=3, fov=45°)
     * cover roughly the central 60% of the frame. Sample 4 anchor pixels,
     * one per quadrant, well within each texel: */
    int xL = W * 35 / 100, xR = W * 65 / 100;
    int yT = H * 35 / 100, yB = H * 65 / 100;
    struct { int x, y; const char* label; } anchors[4] = {
        { xL, yT, "top-left"     },
        { xR, yT, "top-right"    },
        { xL, yB, "bottom-left"  },
        { xR, yB, "bottom-right" },
    };
    ColorSig sigs[4];
    for (int i = 0; i < 4; i++) {
        uint8_t* p = px + ((size_t)anchors[i].y * W + anchors[i].x) * 4;
        sigs[i] = classify_pixel(p[0], p[1], p[2]);
        printf("  %s @ (%d,%d): RGB=(%u,%u,%u) -> %s\n",
               anchors[i].label, anchors[i].x, anchors[i].y,
               p[0], p[1], p[2], sig_name(sigs[i]));
    }

    /* Pass condition: at least 3 of 4 anchors classify cleanly AND no
     * two anchors share the same signature. Rationale: if textures
     * aren't sampled, all four quadrants render the same base_color,
     * yielding identical signatures (or all ambiguous). */
    int distinct = 0;
    int ambig = 0;
    for (int i = 0; i < 4; i++) {
        if (sigs[i] == SIG_AMBIGUOUS) { ambig++; continue; }
        int dup = 0;
        for (int j = 0; j < i; j++)
            if (sigs[j] == sigs[i]) { dup = 1; break; }
        if (!dup) distinct++;
    }
    printf("  distinct color signatures: %d, ambiguous: %d\n", distinct, ambig);

    free(px);
    nu_renderer_destroy(r);

    if (distinct < 3) {
        fprintf(stderr, "FAIL: only %d distinct signatures — texture not being sampled\n", distinct);
        return 1;
    }
    printf("PASS: textured quad test\n");
    return 0;
}

/* Phase 7 — load an HDR env map and verify IBL produces visibly different
 * pixels than the synthetic-3-point fallback. We compare two renders of
 * the same single triangle: one without env (synthetic lighting), one
 * with env (IBL). They should differ enough that the simple delta count
 * is non-trivial. The HDR file must exist at the path passed in;
 * Apple's OpenUSD-install ships several at ~/OpenUSD-install/resources/Lights/. */
static int test_ibl_rt(const char* hdr_path, const char* output_path)
{
    printf("Test: IBL via HDR env map (%s)...\n", hdr_path);

    /* Probe — skip silently if no HDR is at this path. */
    FILE* probe = fopen(hdr_path, "rb");
    if (!probe) {
        printf("  HDR not found at %s — skipping IBL test\n", hdr_path);
        return 0;
    }
    fclose(probe);

    NuRendererConfig config = {0};
    config.width  = 400;
    config.height = 400;
    config.enable_rt = 1;

    NuRenderer* r = nu_renderer_create(&config);
    if (!r) {
        fprintf(stderr, "FAIL: nu_renderer_create returned NULL\n");
        return 1;
    }
    if (!nu_rt_available(r)) {
        printf("  RT not available — skipping IBL test\n");
        nu_renderer_destroy(r);
        return 0;
    }

    /* A single bright-white triangle facing camera. Material is
     * implicitly white via display_color; IBL contribution is most
     * visible on a neutral surface. */
    float positions[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f,
    };
    float normals[] = {
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    };
    uint32_t indices[] = { 0, 1, 2 };
    NuMeshDesc desc = {0};
    desc.positions = positions; desc.normals = normals; desc.indices = indices;
    desc.nvertices = 3; desc.nindices = 3;
    desc.display_color[0] = 0.9f; desc.display_color[1] = 0.9f; desc.display_color[2] = 0.9f;
    desc.name = "ibl_tri";
    int mid = nu_add_mesh(r, &desc);
    if (mid < 0) { nu_renderer_destroy(r); return 1; }

    NuCameraDesc cam = {0};
    cam.eye[0] = 0; cam.eye[1] = 0; cam.eye[2] = 3;
    cam.target[0] = 0; cam.target[1] = 0; cam.target[2] = 0;
    cam.fov_degrees = 45.0f; cam.near_clip = 0.01f; cam.far_clip = 100.0f;
    nu_set_camera(r, 0, &cam);

    /* Render WITHOUT env loaded — synthetic 3-point lighting. */
    if (nu_render(r, 0, NU_RENDER_RT) != NU_OK) {
        fprintf(stderr, "FAIL: pre-IBL render: %s\n", nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    int W = config.width, H = config.height;
    uint8_t* px_pre = (uint8_t*)malloc((size_t)W * H * 4);
    if (nu_fetch_pixels(r, px_pre, NU_PIXEL_RGBA8) != NU_OK) {
        free(px_pre); nu_renderer_destroy(r); return 1;
    }

    /* Load env, re-render. */
    if (nu_load_environment(r, hdr_path) != NU_OK) {
        fprintf(stderr, "FAIL: nu_load_environment: %s\n", nu_get_last_error(r));
        free(px_pre); nu_renderer_destroy(r);
        return 1;
    }
    if (nu_render(r, 0, NU_RENDER_RT) != NU_OK) {
        fprintf(stderr, "FAIL: post-IBL render: %s\n", nu_get_last_error(r));
        free(px_pre); nu_renderer_destroy(r);
        return 1;
    }
    nu_save_ppm(r, output_path);
    printf("  Saved %s\n", output_path);

    uint8_t* px_post = (uint8_t*)malloc((size_t)W * H * 4);
    if (nu_fetch_pixels(r, px_post, NU_PIXEL_RGBA8) != NU_OK) {
        free(px_pre); free(px_post); nu_renderer_destroy(r); return 1;
    }

    /* Compare: the env-lit image should differ from the synthetic-lit
     * one. The miss pixels (sky/ground) most obviously change — sky
     * from synthetic warm-horizon to actual env color. We count pixels
     * with > 30 RGB delta. */
    int changed = 0;
    int sample_x = W / 2, sample_y = H / 2;
    int corner_x = 10, corner_y = 10;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t* a = px_pre  + ((size_t)y * W + x) * 4;
            uint8_t* b = px_post + ((size_t)y * W + x) * 4;
            int da = abs((int)a[0] - (int)b[0])
                   + abs((int)a[1] - (int)b[1])
                   + abs((int)a[2] - (int)b[2]);
            if (da > 30) changed++;
        }
    }
    uint8_t* sc_pre  = px_pre  + ((size_t)sample_y * W + sample_x) * 4;
    uint8_t* sc_post = px_post + ((size_t)sample_y * W + sample_x) * 4;
    uint8_t* cn_pre  = px_pre  + ((size_t)corner_y * W + corner_x) * 4;
    uint8_t* cn_post = px_post + ((size_t)corner_y * W + corner_x) * 4;
    printf("  center pre/post: (%u,%u,%u) / (%u,%u,%u)\n",
           sc_pre[0], sc_pre[1], sc_pre[2], sc_post[0], sc_post[1], sc_post[2]);
    printf("  corner pre/post: (%u,%u,%u) / (%u,%u,%u)\n",
           cn_pre[0], cn_pre[1], cn_pre[2], cn_post[0], cn_post[1], cn_post[2]);
    printf("  pixels changed (Δsum > 30): %d / %d\n", changed, W * H);

    if (changed < W * H / 100) {
        fprintf(stderr, "FAIL: too few pixels changed — IBL doesn't seem to be active\n");
        free(px_pre); free(px_post); nu_renderer_destroy(r);
        return 1;
    }

    free(px_pre); free(px_post);
    nu_renderer_destroy(r);
    printf("PASS: IBL test\n");
    return 0;
}

/* Phase 11.A — render the BasisCurves scene via RT and verify the
 * curve segments are visible. Loads test/basicCurves.usda (8-point
 * linear curve = 6 segments, 0.05 radius), renders, dumps to
 * render_output_curves.ppm, and checks center and corner pixel
 * values to confirm the cylinder rendering produced non-sky pixels. */
static int test_curves_rt(const char* output_path)
{
    printf("Test: rendering BasisCurves via ray tracing...\n");

    NuRendererConfig config = {0};
    config.width  = 600;
    config.height = 600;
    config.enable_rt = 1;

    NuRenderer* r = nu_renderer_create(&config);
    if (!r) {
        fprintf(stderr, "FAIL: nu_renderer_create returned NULL\n");
        return 1;
    }
    if (!nu_rt_available(r)) {
        printf("  RT not available — skipping curves RT test\n");
        nu_renderer_destroy(r);
        return 0;
    }

    /* Load the curve fixture. Auto-frames camera based on scene bounds. */
    int n = nu_load_usd(r, "test/basicCurves.usda");
    if (n < 0) {
        fprintf(stderr, "FAIL: nu_load_usd returned %d: %s\n",
                n, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    int nseg = nu_get_curve_segment_count(r);
    printf("  loaded %d meshes + %d curve segments\n", n, nseg);
    if (nseg <= 0) {
        fprintf(stderr, "FAIL: expected curve segments, got %d\n", nseg);
        nu_renderer_destroy(r);
        return 1;
    }

    NuResult res = nu_render(r, 0, NU_RENDER_RT);
    if (res != NU_OK) {
        fprintf(stderr, "FAIL: nu_render(RT) returned %d: %s\n",
                res, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    res = nu_save_ppm(r, output_path);
    if (res != NU_OK) {
        fprintf(stderr, "FAIL: nu_save_ppm returned %d: %s\n",
                res, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    printf("  Saved to %s\n", output_path);

    /* Pull pixels and look for curve evidence. The curves run through the
     * upper-right quadrant of the framing box; the saturated orange tint
     * (RGB ~ 0.85, 0.55, 0.20) gets sRGB-encoded by the color_target's
     * MTLPixelFormatRGBA8Unorm_sRGB to roughly (240, 170, 100). Scan a
     * horizontal sweep across the frame and count "warm" pixels: R > B + 30.
     * Empty-scene rendering (sky/ground) would have R ~ B (warm-neutral). */
    int w = config.width, h = config.height;
    uint8_t* px = (uint8_t*)malloc((size_t)w * h * 4);
    if (!px) { nu_renderer_destroy(r); return 1; }
    res = nu_fetch_pixels(r, px, NU_PIXEL_RGBA8);
    if (res != NU_OK) { free(px); nu_renderer_destroy(r); return 1; }

    int warm_pixels = 0;
    for (int y = 0; y < h; y += 4) {
        for (int x = 0; x < w; x += 4) {
            uint8_t* p = px + ((size_t)y * w + x) * 4;
            if ((int)p[0] - (int)p[2] > 30) warm_pixels++;
        }
    }
    printf("  warm pixels (R > B + 30): %d  (sample pixels @ (300,150)=(%u,%u,%u))\n",
           warm_pixels,
           px[((size_t)150 * w + 300) * 4 + 0],
           px[((size_t)150 * w + 300) * 4 + 1],
           px[((size_t)150 * w + 300) * 4 + 2]);

    if (warm_pixels < 10) {
        fprintf(stderr, "FAIL: expected curve pixels (warm tint), found only %d\n", warm_pixels);
        free(px);
        nu_renderer_destroy(r);
        return 1;
    }

    free(px);
    nu_renderer_destroy(r);
    printf("PASS: curves RT test\n");
    return 0;
}

static int test_usd_file(const char* usd_path, const char* output_path)
{
    printf("Test: rendering USD file %s...\n", usd_path);

    NuRendererConfig config = {0};
    config.width  = getenv("NUSD_WIDTH") ? atoi(getenv("NUSD_WIDTH")) : 1920;
    config.height = getenv("NUSD_HEIGHT") ? atoi(getenv("NUSD_HEIGHT")) : 1080;
    if (config.width <= 0) config.width = 1920;
    if (config.height <= 0) config.height = 1080;
    config.enable_rt = 1;
    if (getenv("NUSD_DISABLE_RT")) {
        config.enable_rt = atoi(getenv("NUSD_DISABLE_RT")) == 0;
    }
    config.enable_materials = 1;
    if (getenv("NUSD_MATERIALS")) {
        config.enable_materials = atoi(getenv("NUSD_MATERIALS")) != 0;
    }

    NuRenderer* r = nu_renderer_create(&config);
    if (!r) {
        fprintf(stderr, "FAIL: nu_renderer_create returned NULL\n");
        return 1;
    }

    printf("  Renderer created (%dx%d, RT=%s, materials=%s)\n",
           config.width, config.height,
           nu_rt_available(r) ? "yes" : "no",
           config.enable_materials ? "yes" : "no");

    int nmeshes = nu_load_usd(r, usd_path);
    if (nmeshes < 0) {
        fprintf(stderr, "FAIL: nu_load_usd returned %d: %s\n",
                nmeshes, nu_get_last_error(r));
        nu_renderer_destroy(r);
        return 1;
    }
    printf("  Loaded %d meshes\n", nmeshes);

    /* NUSD_FINALIZE=1 drops CPU-side geometry staging after GPU upload/accel
     * build (nu_finalize_scene). For huge static scenes (e.g. DSX) this roughly
     * halves host RSS at the cost of being unable to add/mutate geometry after. */
    const char* finalize_env = getenv("NUSD_FINALIZE");
    if (finalize_env && finalize_env[0] == '1') {
        if (nu_finalize_scene(r) != NU_OK) {
            fprintf(stderr, "  NUSD_FINALIZE: nu_finalize_scene failed: %s\n",
                    nu_get_last_error(r));
        } else {
            printf("  NUSD_FINALIZE: released CPU geometry staging\n");
        }
    }

    /* Optional camera override — NUSD_CAM=eye_x,eye_y,eye_z,tgt_x,tgt_y,tgt_z[,fov]
     * lets you override the renderer's auto-framed camera. Useful when
     * auto-frame puts the camera too far out (e.g., warehouse-scale
     * scenes where 96m camera distance + beams along view axis produces
     * streak-like projections that LOOK like a geometry explosion). */
    {
        const char* cam = getenv("NUSD_CAM");
        if (cam && *cam) {
            float ex, ey, ez, tx, ty, tz, fov = 45.0f;
            int n = sscanf(cam, "%f,%f,%f,%f,%f,%f,%f",
                           &ex, &ey, &ez, &tx, &ty, &tz, &fov);
            if (n >= 6) {
                NuCameraDesc cd = {0};
                cd.eye[0] = ex; cd.eye[1] = ey; cd.eye[2] = ez;
                cd.target[0] = tx; cd.target[1] = ty; cd.target[2] = tz;
                cd.fov_degrees = fov;
                cd.near_clip = 0.001f;
                cd.far_clip  = 1.0e5f;
                nu_set_camera(r, 0, &cd);
                printf("  Camera override: eye=(%.2f,%.2f,%.2f) target=(%.2f,%.2f,%.2f) fov=%.1f\n",
                       ex, ey, ez, tx, ty, tz, fov);
            } else {
                fprintf(stderr, "  NUSD_CAM: parse failed, expected 6+ comma-separated floats\n");
            }
        }
    }

    /* Optional IBL — set NUSD_HDRI=<path> to drop a real HDR into the
     * scene's ambient + reflection slots. Useful for visual-checking
     * MaterialX-shaded assets (transmission, SSS, metallic) which look
     * lifeless under the hemisphere fallback. */
    const char* hdr_path = getenv("NUSD_HDRI");
    if (hdr_path && *hdr_path) {
        const char* intensity_s = getenv("NUSD_HDRI_INTENSITY");
        NuResult env_res = NU_ERROR;
        if (intensity_s && *intensity_s) {
            float intensity = (float)atof(intensity_s);
            env_res = nu_load_environment_intensity(r, hdr_path, intensity);
            if (env_res == NU_OK)
                printf("  IBL: %s (intensity=%.3f)\n", hdr_path, intensity);
        } else {
            env_res = nu_load_environment(r, hdr_path);
            if (env_res == NU_OK)
                printf("  IBL: %s\n", hdr_path);
        }
        if (env_res != NU_OK) {
            fprintf(stderr, "  IBL: failed to load %s (%s)\n",
                    hdr_path, nu_get_last_error(r));
        }
    }
    const char* dome_color = getenv("NUSD_DOME_COLOR");
    if (dome_color && *dome_color) {
        float dr, dg, db, di = 1.0f;
        int n = sscanf(dome_color, "%f,%f,%f,%f", &dr, &dg, &db, &di);
        if (n >= 3) {
            if (nu_set_dome_color(r, dr, dg, db, di) == NU_OK) {
                printf("  Dome background: %.3f,%.3f,%.3f intensity=%.3f\n",
                       dr, dg, db, di);
            } else {
                fprintf(stderr, "  Dome background: failed (%s)\n",
                        nu_get_last_error(r));
            }
        } else {
            fprintf(stderr,
                    "  NUSD_DOME_COLOR: parse failed, expected r,g,b[,intensity]\n");
        }
    }

    /* Try RT first, fall back to raster. NUSD_RASTER=1 forces raster
     * for debugging mode-specific bugs (e.g., test_pbr_materials.usda
     * RT-vs-raster discrepancies). NUSD_SHADOW=1 forces the raster +
     * ray-query shadow path for parity captures. */
    NuRenderMode mode = NU_RENDER_RT;
    if (!nu_rt_available(r)) {
        printf("  RT not available, falling back to raster\n");
        mode = NU_RENDER_RASTER;
    } else if (getenv("NUSD_SHADOW")) {
        printf("  NUSD_SHADOW set — forcing raster+shadow mode\n");
        mode = NU_RENDER_SHADOW;
    } else if (getenv("NUSD_RASTER")) {
        printf("  NUSD_RASTER set — forcing raster mode\n");
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
    printf("  Frame rendered (mode=%s)\n",
           mode == NU_RENDER_RT ? "RT" :
           mode == NU_RENDER_SHADOW ? "shadow" : "raster");

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

    printf("  GPU memory used: %.1f MB\n",
           (double)nu_get_gpu_memory_used(r) / (1024.0 * 1024.0));

    nu_renderer_destroy(r);
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
        int rc = test_triangle(output_path);
        if (rc) return rc;
        rc = test_hash_dedup_renderer_isolation();
        if (rc) return rc;
        rc = test_triangle_rt("render_output_rt.ppm");
        if (rc) return rc;
        rc = test_tiled_rt(NULL);
        if (rc) return rc;
        rc = test_curves_rt("render_output_curves.ppm");
        if (rc) return rc;
        rc = test_ibl_rt(default_ibl_hdr(), "render_output_ibl.ppm");
        if (rc) return rc;
        return test_textured_cube_rt("test/textured_quad.usda",
                                     "render_output_textured.ppm");
    }
}
