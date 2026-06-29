// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* curves_smoketest.c — render one frame and dump PPM. Same matrices as
 * determinism_test.c (which is known to render visible geometry). When
 * NUSD_CURVES_SMOKETEST=1 is set, the BasisCurves smoke-test patch
 * should appear on top of the scene. */
#include "viewer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void make_view(float view[16], float dist) {
    memset(view, 0, sizeof(float) * 16);
    view[0]  = 1.0f;
    view[5]  = 1.0f;
    view[10] = 1.0f;
    view[14] = -dist;
    view[15] = 1.0f;
}

static void make_proj(float proj[16], float aspect, float near, float far) {
    float f = 1.0f / tanf(0.5f * 45.0f * 3.14159265358979323846f / 180.0f);
    memset(proj, 0, sizeof(float) * 16);
    proj[0]  = f / aspect;
    proj[5]  = f;
    proj[10] = -(far + near) / (far - near);
    proj[11] = -1.0f;
    proj[14] = -(2.0f * far * near) / (far - near);
}

int main(int argc, char** argv) {
    const char* usd = (argc > 1) ? argv[1] : "test_pbr_materials.usda";
    const char* out = (argc > 2) ? argv[2] : "/tmp/curves_test.ppm";
    float dist     = (argc > 3) ? (float)atof(argv[3]) : 5.0f;
    const int W = 512, H = 512;

    Viewer* v = viewer_create(usd, W, H, 64, NULL, 1, 1);
    if (!v) { fprintf(stderr, "FAIL viewer_create\n"); return 1; }
    viewer_set_overlay_enabled(v, 0);
    viewer_set_render_mode(v, VIEWER_MODE_RASTER);

    float view[16], proj[16], eye[3] = { 0, 0, dist };
    make_view(view, dist);
    make_proj(proj, (float)W / (float)H, 0.1f, 1000.0f);

    unsigned char* px = (unsigned char*)malloc((size_t)W * H * 4);
    if (!viewer_render_to_rgba(v, W, H, view, proj, eye, px)) {
        fprintf(stderr, "FAIL render\n"); return 1;
    }
    FILE* fp = fopen(out, "wb");
    if (!fp) { fprintf(stderr, "FAIL open %s\n", out); return 1; }
    fprintf(fp, "P6\n%d %d\n255\n", W, H);
    /* Strip alpha and write RGB rows. */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            unsigned char* p = &px[(y * W + x) * 4];
            unsigned char rgb[3] = { p[0], p[1], p[2] };
            fwrite(rgb, 1, 3, fp);
        }
    }
    fclose(fp);
    fprintf(stderr, "wrote %s\n", out);
    free(px);
    viewer_destroy(v);
    return 0;
}
