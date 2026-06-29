// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* geo_cache_render_smoke.c — one-time empirical floor under the geometry
 * cache's rotation-invariance assumption.
 *
 * The meshoptimizer index codec canonicalizes each triangle's vertex order,
 * so warm-cache index buffers are per-triangle cyclic rotations of the
 * cold-parse ones. Cyclic rotation preserves winding, so it is render-
 * invariant under smooth shading — but that is an assumption about the
 * renderer. This test renders the same scene twice — once from the cold USD
 * parse, once from the warm geometry cache (whose indices carry the codec's
 * rotation) — and asserts the framebuffers are bit-identical.
 *
 * This is a one-time confirmation, not the per-scene gate: geo_cache_test
 * (rotation-aware, pure CPU) is the gate run on every scene.
 *
 * Usage: geo_cache_render_smoke <scene.usd>
 */
#include "viewer.h"
#include "geo_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void make_view(float view[16])
{
    memset(view, 0, sizeof(float) * 16);
    view[0] = view[5] = view[10] = 1.0f;
    view[14] = -5.0f;                 /* eye at z=+5 looking down -Z */
    view[15] = 1.0f;
}

static void make_proj(float proj[16], float aspect)
{
    float f = 1.0f / tanf(0.5f * 45.0f * 3.14159265358979323846f / 180.0f);
    memset(proj, 0, sizeof(float) * 16);
    proj[0]  = f / aspect;
    proj[5]  = f;
    proj[10] = -(100.0f + 0.1f) / (100.0f - 0.1f);
    proj[11] = -1.0f;
    proj[14] = -(2.0f * 100.0f * 0.1f) / (100.0f - 0.1f);
}

/* Render `usd` headless into `out` (W*H*4 RGBA). Returns 1 on success. */
static int render_once(const char* usd, int W, int H, unsigned char* out)
{
    Viewer* v = viewer_create(usd, W, H, 64, NULL, /*headless=*/1,
                              /*enable_materials=*/0);
    if (!v) return 0;
    viewer_set_overlay_enabled(v, 0);
    float view[16], proj[16], eye[3] = { 0, 0, 5 };
    make_view(view);
    make_proj(proj, (float)W / (float)H);
    int ok = viewer_render_to_rgba(v, W, H, view, proj, eye, out);
    viewer_destroy(v);
    return ok;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: geo_cache_render_smoke <scene.usd>\n");
        return 2;
    }
    const char* usd = argv[1];
    const int W = 256, H = 256;
    size_t nbytes = (size_t)W * H * 4;

    setenv("NUSD_GEO_CACHE", "1", 1);
    char cpath[2048];
    if (geo_cache_path_for(usd, cpath, sizeof cpath) == 0) remove(cpath);

    unsigned char* a = (unsigned char*)malloc(nbytes);   /* render of cold parse */
    unsigned char* b = (unsigned char*)malloc(nbytes);   /* render of warm cache */
    if (!a || !b) { fprintf(stderr, "FAIL: alloc\n"); free(a); free(b); return 1; }

    /* Render 1: cache MISS — the viewer loads the cold USD parse and writes
     * <usd>.nzgeo.gl as a side effect. */
    if (!render_once(usd, W, H, a)) {
        fprintf(stderr, "SKIP: viewer_create/render failed (no headless GL?)\n");
        free(a); free(b);
        return 77;
    }
    /* Render 2: cache HIT — the viewer loads the warm cache, whose index
     * buffers carry the meshopt codec's per-triangle rotation. */
    if (!render_once(usd, W, H, b)) {
        fprintf(stderr, "FAIL: second (warm-cache) render failed\n");
        free(a); free(b);
        return 1;
    }

    int nonzero = 0;
    for (size_t i = 0; i < nbytes; i++)
        if (a[i] > 0) { nonzero++; if (nonzero > 100) break; }

    int diff = memcmp(a, b, nbytes);
    if (diff != 0) {
        for (size_t i = 0; i < nbytes; i++)
            if (a[i] != b[i]) {
                fprintf(stderr, "first pixel-byte diff at %zu: parse=%d cache=%d\n",
                        i, a[i], b[i]);
                break;
            }
    }
    free(a); free(b);

    if (nonzero <= 100) {
        fprintf(stderr, "FAIL: framebuffer all-black — render did not run\n");
        return 1;
    }
    if (diff != 0) {
        fprintf(stderr, "FAIL: warm-cache render != cold-parse render — "
                        "the index rotation is NOT render-invariant here\n");
        return 1;
    }
    fprintf(stderr, "PASS: warm-cache render == cold-parse render "
                    "(meshopt index rotation is render-invariant)\n");
    return 0;
}
