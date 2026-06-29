// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* determinism_test.c — render the same scene twice with identical
 * camera parameters and assert the framebuffers are bit-identical.
 * Catches accidentally introduced state-leak / per-frame jitter. */
#include "viewer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); g_fail++; } \
    else         { fprintf(stderr, "ok:   %s\n", msg); } \
} while (0)

/* Identity-ish view + perspective projection, hand-built so this test
 * doesn't depend on Camera internals. Looks down -Z. */
static void make_view(float view[16]) {
    memset(view, 0, sizeof(float) * 16);
    view[0]  = 1.0f;
    view[5]  = 1.0f;
    view[10] = 1.0f;
    view[14] = -5.0f;  /* eye at z=+5 looking at origin */
    view[15] = 1.0f;
}

static void make_proj(float proj[16], float aspect) {
    /* GL clip space, fov_y = 45deg, near=0.1, far=100 */
    float f = 1.0f / tanf(0.5f * 45.0f * 3.14159265358979323846f / 180.0f);
    memset(proj, 0, sizeof(float) * 16);
    proj[0]  = f / aspect;
    proj[5]  = f;
    proj[10] = -(100.0f + 0.1f) / (100.0f - 0.1f);
    proj[11] = -1.0f;
    proj[14] = -(2.0f * 100.0f * 0.1f) / (100.0f - 0.1f);
}

int main(int argc, char** argv) {
    const char* usd = (argc > 1) ? argv[1] : "test_pbr_materials.usda";
    const int W = 256, H = 256;

    Viewer* v = viewer_create(usd, W, H, 64, NULL, /*headless=*/1, 1);
    if (!v) {
        fprintf(stderr, "SKIP: viewer_create(headless) failed\n");
        return 77;
    }
    /* Disable HUD overlay — the FPS counter changes per frame and
     * would defeat determinism. */
    viewer_set_overlay_enabled(v, 0);

    float view[16], proj[16], eye[3] = { 0, 0, 5 };
    make_view(view);
    make_proj(proj, (float)W / (float)H);

    unsigned char* a = (unsigned char*)malloc((size_t)W * H * 4);
    unsigned char* b = (unsigned char*)malloc((size_t)W * H * 4);
    CHECK(a && b, "buffer alloc");

    int ok1 = viewer_render_to_rgba(v, W, H, view, proj, eye, a);
    int ok2 = viewer_render_to_rgba(v, W, H, view, proj, eye, b);
    CHECK(ok1 && ok2, "two render_to_rgba calls succeed");

    int diff = memcmp(a, b, (size_t)W * H * 4);
    if (diff != 0) {
        /* Find first differing pixel for debugging. */
        size_t total = (size_t)W * H * 4;
        for (size_t i = 0; i < total; i++) {
            if (a[i] != b[i]) {
                fprintf(stderr, "first diff at byte %zu: a=%d b=%d\n", i, a[i], b[i]);
                break;
            }
        }
    }
    CHECK(diff == 0, "consecutive renders are bit-identical");

    /* All-black framebuffer would mean rendering didn't actually run.
     * Any modest non-zero pixel count is sufficient. */
    int nonzero = 0;
    size_t total = (size_t)W * H * 4;
    for (size_t i = 0; i < total; i++) if (a[i] > 0) { nonzero++; if (nonzero > 100) break; }
    CHECK(nonzero > 100, "framebuffer is not all-black");

    free(a); free(b);
    viewer_destroy(v);

    fprintf(stderr, "\n%s: %d failures\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
