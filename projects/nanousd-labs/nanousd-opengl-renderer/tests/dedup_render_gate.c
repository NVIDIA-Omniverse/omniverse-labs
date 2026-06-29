// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* dedup_render_gate.c — correctness gate for content-hash dedup.
 *
 * Renders <scene.usd> headless (materials on, framed to the scene bounds) TWICE
 * — once with dedup on (default) and once with NUSD_HASH_DEDUP=0 — and asserts
 * the framebuffers are bit-identical. Because dedup shares byte-identical
 * geometry across differing materials (material is applied per-draw, not baked
 * into the shared vertex stream; display_color stays in the dedup key because it
 * IS vertex-baked), the rendered frame must not change.
 *
 * Note: requires headless GL that actually rasterizes geometry (Linux EGL). On
 * platforms whose headless context only clears (some macOS CGL pbuffers) both
 * renders degrade to the same background and the gate passes trivially — it
 * never produces a false failure.
 *
 * Usage: dedup_render_gate <scene.usd> [width height]
 */
#include "viewer.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void make_proj(float proj[16], float aspect, float nearp, float farp) {
    float fy = 1.0f / tanf(0.5f * 45.0f * 3.14159265358979323846f / 180.0f);
    memset(proj, 0, sizeof(float)*16);
    proj[0]  = fy / aspect;
    proj[5]  = fy;
    proj[10] = -(farp + nearp) / (farp - nearp);
    proj[11] = -1.0f;
    proj[14] = -(2.0f * farp * nearp) / (farp - nearp);
}

/* Render `usd` into `out` (W*H*4). dedup_on toggles NUSD_HASH_DEDUP. mats
 * enables materials. Returns 1 on success, 0 on render failure, -1 if the GL
 * context could not be created (caller treats as SKIP). */
static int render_frame(const char* usd, int W, int H, int mats, int dedup_on,
                        unsigned char* out) {
    setenv("NUSD_HASH_DEDUP", dedup_on ? "1" : "0", 1);
    Viewer* v = viewer_create(usd, W, H, 64, NULL, /*headless=*/1, mats);
    if (!v) return -1;
    viewer_set_overlay_enabled(v, 0);
    if (mats) viewer_set_render_mode(v, VIEWER_MODE_MATERIAL);

    float mn[3], mx[3];
    if (!viewer_get_scene_bounds(v, mn, mx)) { viewer_destroy(v); return 0; }
    float ctr[3] = { 0.5f*(mn[0]+mx[0]), 0.5f*(mn[1]+mx[1]), 0.5f*(mn[2]+mx[2]) };
    float dx=mx[0]-mn[0], dy=mx[1]-mn[1], dz=mx[2]-mn[2];
    float radius = 0.5f * sqrtf(dx*dx + dy*dy + dz*dz);
    if (radius < 1e-4f) radius = 1.0f;
    float dist = radius * 3.0f;
    /* Proven convention from geo_cache_render_smoke: camera at eye looking down
     * world -Z (view = translate(-eye)), centered on the scene bounds. Exact
     * framing is irrelevant — it just needs to be identical for on/off. */
    float eye[3] = { ctr[0], ctr[1], ctr[2] + dist };
    float view[16];
    memset(view, 0, sizeof(view));
    view[0] = view[5] = view[10] = view[15] = 1.0f;
    view[12] = -eye[0]; view[13] = -eye[1]; view[14] = -eye[2];
    float proj[16];
    make_proj(proj, (float)W/(float)H, radius*0.01f + 0.01f, dist + radius*4.0f);

    int ok = viewer_render_to_rgba(v, W, H, view, proj, eye, out);
    viewer_destroy(v);
    return ok ? 1 : 0;
}

static uint64_t fnv64(const unsigned char* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: dedup_render_gate <scene.usd> [W H]\n"); return 2; }
    const char* usd = argv[1];
    int W = (argc >= 4) ? atoi(argv[2]) : 512;
    int H = (argc >= 4) ? atoi(argv[3]) : 512;
    int mats = 1;
    const char* me = getenv("NUSD_MATERIALS");
    if (me && me[0] == '0') mats = 0;

    size_t nbytes = (size_t)W*H*4;
    unsigned char* on  = (unsigned char*)malloc(nbytes);
    unsigned char* off = (unsigned char*)malloc(nbytes);
    if (!on || !off) { fprintf(stderr, "FAIL: alloc\n"); free(on); free(off); return 1; }

    int r1 = render_frame(usd, W, H, mats, /*dedup_on=*/1, on);
    if (r1 < 0) { fprintf(stderr, "SKIP: viewer_create failed (no headless GL?)\n"); free(on); free(off); return 77; }
    int r2 = render_frame(usd, W, H, mats, /*dedup_on=*/0, off);
    if (r1 != 1 || r2 != 1) { fprintf(stderr, "FAIL: render failed (on=%d off=%d)\n", r1, r2); free(on); free(off); return 1; }

    /* Diagnostic: dump the dedup-on render + report distinct colors so we can
     * confirm geometry actually rasterized (not just a flat clear). */
    const char* ppm = getenv("NUSD_GATE_PPM");
    if (ppm) {
        FILE* f = fopen(ppm, "wb");
        if (f) {
            fprintf(f, "P6\n%d %d\n255\n", W, H);
            for (size_t i = 0; i < (size_t)W*H; i++) fwrite(on + i*4, 1, 3, f);
            fclose(f);
            fprintf(stderr, "wrote %s\n", ppm);
        }
    }
    { /* count distinct RGB values (cap at 4096) as a "is there structure?" probe */
        unsigned seen = 0; static unsigned char tbl[1<<12]; memset(tbl,0,sizeof tbl);
        for (size_t i = 0; i < (size_t)W*H && seen < 4096; i++) {
            unsigned key = ((on[i*4]>>4)<<8) | ((on[i*4+1]>>4)<<4) | (on[i*4+2]>>4);
            if (!tbl[key]) { tbl[key]=1; seen++; }
        }
        fprintf(stderr, "distinct ~colors (4-bit/chan, dedup-on render): %u\n", seen);
    }

    int diff = memcmp(on, off, nbytes);
    long nonzero = 0;
    for (size_t i = 0; i < nbytes; i++) if (on[i]) nonzero++;
    printf("dedup_render_gate: %dx%d hash_on=%016llx hash_off=%016llx nonzero=%ld materials=%d\n",
           W, H, (unsigned long long)fnv64(on, nbytes), (unsigned long long)fnv64(off, nbytes), nonzero, mats);
    free(on); free(off);

    if (diff != 0) {
        fprintf(stderr, "FAIL: dedup-on render != dedup-off render — cross-material dedup changed pixels\n");
        return 1;
    }
    fprintf(stderr, "PASS: dedup-on render == dedup-off render (cross-material dedup is bit-identical)\n");
    return 0;
}
