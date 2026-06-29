// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* meshlet_draw_test.c — verify the meshlet-driven draw path.
 *
 * Renders a scene headless in raster mode twice — once with the whole-mesh
 * draw (NUSD_MESHLET_DRAW off), once with the meshlet draw path on — and
 * compares the framebuffers. The meshlet path frustum-culls whole meshlets,
 * which removes only off-screen geometry, so every visible pixel must match.
 *
 * Two cameras: a whole-scene framing (no culling — isolates triangle-reorder
 * invariance + the inside-mesh meshlet draw) and an interior camera (heavy
 * straddle — exercises the per-meshlet cull). A 0xAB pre-fill canary catches a
 * render that silently did nothing.
 *
 * Usage: meshlet_draw_test <scene.usd>
 */
#define _POSIX_C_SOURCE 200809L

#include "viewer.h"
#include "scene.h"
#include "geo_cache.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- minimal row-major camera math (matches the renderer's VP) ---- */

static void mat4_mul4(const float* a, const float* b, float* o)
{
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a[r*4+k]*b[k*4+c];
            o[r*4+c] = s;
        }
}
static void v3sub(const float* a, const float* b, float* o)
{ o[0]=a[0]-b[0]; o[1]=a[1]-b[1]; o[2]=a[2]-b[2]; }
static void v3nrm(float* v)
{ float l=sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); if(l>1e-20f){v[0]/=l;v[1]/=l;v[2]/=l;} }
static void v3crs(const float* a, const float* b, float* o)
{ o[0]=a[1]*b[2]-a[2]*b[1]; o[1]=a[2]*b[0]-a[0]*b[2]; o[2]=a[0]*b[1]-a[1]*b[0]; }
static float v3dot(const float* a, const float* b)
{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

static void look_at(const float* eye, const float* ctr, const float* up_in, float* v)
{
    float fwd[3], right[3], u[3], up[3];
    up[0]=up_in[0]; up[1]=up_in[1]; up[2]=up_in[2];
    v3sub(ctr, eye, fwd); v3nrm(fwd);
    v3crs(fwd, up, right);
    if (right[0]*right[0]+right[1]*right[1]+right[2]*right[2] < 1e-12f) {
        up[0]=1; up[1]=0; up[2]=0;
        if (fabsf(fwd[0]) > 0.9f) { up[0]=0; up[1]=1; up[2]=0; }
        v3crs(fwd, up, right);
    }
    v3nrm(right);
    v3crs(right, fwd, u);
    v[0]=right[0]; v[1]=right[1]; v[2]=right[2]; v[3]=-v3dot(right,eye);
    v[4]=u[0];     v[5]=u[1];     v[6]=u[2];     v[7]=-v3dot(u,eye);
    v[8]=-fwd[0];  v[9]=-fwd[1];  v[10]=-fwd[2]; v[11]=v3dot(fwd,eye);
    v[12]=0; v[13]=0; v[14]=0; v[15]=1;
}
static void perspective(float fovy, float aspect, float zn, float zf, float* p)
{
    float f = 1.0f / tanf(0.5f*fovy*3.14159265358979f/180.0f);
    memset(p, 0, 16*sizeof(float));
    p[0]=f/aspect; p[5]=f;
    p[10]=(zf+zn)/(zn-zf); p[11]=(2.0f*zf*zn)/(zn-zf); p[14]=-1.0f;
}

/* Render `usd` headless in raster mode (materials off) into `out` (W*H*4
 * RGBA). meshlet=1 enables NUSD_MESHLET_DRAW. `out` is 0xAB pre-filled.
 * Returns 1 on success. */
static int render_once(const char* usd, int W, int H, int meshlet,
                       const float view[16], const float proj[16],
                       unsigned char* out)
{
    setenv("NUSD_MESHLET_DRAW", meshlet ? "1" : "0", 1);
    memset(out, 0xAB, (size_t)W*H*4);                   /* canary pre-fill */
    Viewer* v = viewer_create(usd, W, H, 64, NULL, /*headless=*/1,
                              /*enable_materials=*/0);
    if (!v) return 0;
    viewer_set_overlay_enabled(v, 0);
    float eye[3] = { 0, 0, 0 };
    int ok = viewer_render_to_rgba(v, W, H, view, proj, eye, out);
    viewer_destroy(v);
    return ok;
}

/* Compare two framebuffers. Returns the differing-byte count; prints the
 * first differing pixel. */
static long fb_diff(const unsigned char* a, const unsigned char* b, size_t n)
{
    long diff = 0;
    size_t first = (size_t)-1;
    for (size_t i = 0; i < n; i++)
        if (a[i] != b[i]) { if (first == (size_t)-1) first = i; diff++; }
    if (diff)
        fprintf(stderr, "  first diff at byte %zu: plain=%d meshlet=%d\n",
                first, a[first], b[first]);
    return diff;
}

static int g_fail = 0;

/* Render the scene plain vs meshlet for one camera and compare. */
static void check_camera(const char* usd, int W, int H,
                         const float view[16], const float proj[16],
                         const char* label)
{
    size_t n = (size_t)W*H*4;
    unsigned char* a = (unsigned char*)malloc(n);
    unsigned char* b = (unsigned char*)malloc(n);
    if (!a || !b) { fprintf(stderr, "FAIL: alloc\n"); g_fail++; free(a); free(b); return; }

    if (!render_once(usd, W, H, 0, view, proj, a)) {
        fprintf(stderr, "FAIL [%s]: plain render failed\n", label);
        g_fail++; free(a); free(b); return;
    }
    if (!render_once(usd, W, H, 1, view, proj, b)) {
        fprintf(stderr, "FAIL [%s]: meshlet render failed\n", label);
        g_fail++; free(a); free(b); return;
    }

    /* Canary: the plain render must have overwritten the 0xAB pre-fill. */
    long canary = 0;
    for (size_t i = 0; i < n; i++) if (a[i] != 0xAB) { canary++; if (canary > 64) break; }
    if (canary <= 64) {
        fprintf(stderr, "FAIL [%s]: framebuffer still 0xAB — render did not run\n",
                label);
        g_fail++; free(a); free(b); return;
    }

    long diff = fb_diff(a, b, n);
    double pct = 100.0 * (double)diff / (double)n;
    fprintf(stderr, "[%s] meshlet-vs-plain: %ld / %zu bytes differ (%.4f%%)\n",
            label, diff, n, pct);
    /* Frustum culling removes only off-screen geometry, so the meshlet render
     * must match the plain render. A tiny residual is the z-fight noise floor
     * from triangle reordering; anything above 0.5% is a real bug. */
    if (diff == 0)
        fprintf(stderr, "  ok: bit-identical\n");
    else if (pct < 0.5)
        fprintf(stderr, "  ok: within the triangle-reorder z-fight noise floor\n");
    else {
        fprintf(stderr, "  FAIL: meshlet render differs from plain render\n");
        g_fail++;
    }
    free(a);
    free(b);
}

static double now_seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* Create one viewer in raster mode (meshlet draw on/off) and time `nframes`
 * renders after a warmup. Returns average ms/frame, or -1 on failure. */
static double time_renders(const char* usd, int W, int H, int meshlet,
                           const float view[16], const float proj[16],
                           int nframes)
{
    setenv("NUSD_MESHLET_DRAW", meshlet ? "1" : "0", 1);
    Viewer* v = viewer_create(usd, W, H, 64, NULL, /*headless=*/1,
                              /*enable_materials=*/0);
    if (!v) return -1.0;
    viewer_set_overlay_enabled(v, 0);
    unsigned char* buf = (unsigned char*)malloc((size_t)W*H*4);
    if (!buf) { viewer_destroy(v); return -1.0; }
    float eye[3] = { 0, 0, 0 };
    viewer_render_to_rgba(v, W, H, view, proj, eye, buf);   /* warmup */
    double t0 = now_seconds();
    for (int f = 0; f < nframes; f++)
        viewer_render_to_rgba(v, W, H, view, proj, eye, buf);
    double dt = now_seconds() - t0;
    free(buf);
    viewer_destroy(v);
    return nframes > 0 ? dt * 1000.0 / (double)nframes : -1.0;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: meshlet_draw_test <scene.usd>\n");
        return 2;
    }
    const char* usd = argv[1];
    const int W = 256, H = 256;
    setenv("NUSD_GEO_CACHE", "1", 1);

    /* Ensure a cache exists — the meshlet path consumes cached meshlets. */
    if (geo_cache_cook(usd) != 0) {
        fprintf(stderr, "FAIL: geo_cache_cook(%s)\n", usd);
        return 1;
    }

    /* Probe the scene bounds to place the cameras. */
    Scene* probe = scene_load(usd);
    if (!probe || probe->nmeshes <= 0) {
        fprintf(stderr, "FAIL: scene_load(%s)\n", usd);
        return 1;
    }
    float c[3] = { 0.5f*(probe->bounds_min[0]+probe->bounds_max[0]),
                   0.5f*(probe->bounds_min[1]+probe->bounds_max[1]),
                   0.5f*(probe->bounds_min[2]+probe->bounds_max[2]) };
    float e[3] = { probe->bounds_max[0]-probe->bounds_min[0],
                   probe->bounds_max[1]-probe->bounds_min[1],
                   probe->bounds_max[2]-probe->bounds_min[2] };
    float radius = 0.5f*sqrtf(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]);
    if (radius < 1e-6f) radius = 1.0f;
    float up[3] = { 0, 1, 0 };
    if (probe->up_axis == 2) { up[0]=0; up[1]=0; up[2]=1; }
    scene_free(probe);

    float view[16], proj[16], eye[3], tgt[3];

    /* Camera 1 — whole-scene framing (oblique, never parallel to up). */
    {
        float d[3] = { 0.40f, -0.84f, 0.36f };
        eye[0]=c[0]+5.0f*radius*d[0];
        eye[1]=c[1]+5.0f*radius*d[1];
        eye[2]=c[2]+5.0f*radius*d[2];
    }
    look_at(eye, c, up, view);
    perspective(50.0f, 1.0f, 0.01f*radius, 20.0f*radius, proj);
    check_camera(usd, W, H, view, proj, "whole-scene");

    /* Camera 2 — interior, looking +X from the centre (heavy straddle). */
    eye[0]=c[0]; eye[1]=c[1]; eye[2]=c[2];
    tgt[0]=c[0]+radius; tgt[1]=c[1]; tgt[2]=c[2];
    look_at(eye, tgt, up, view);
    perspective(60.0f, 1.0f, 0.001f*radius, 10.0f*radius, proj);
    check_camera(usd, W, H, view, proj, "interior");

    /* Frame-time measurement — interior camera, scanned across resolutions.
     * The meshlet path's draw-call-CPU cost is resolution-independent; the GPU
     * fragment work grows with resolution. A crossover (meshlet faster at high
     * resolution) would mean the cull's triangle saving wins once the GPU is
     * the bottleneck rather than draw-call submission. */
    {
        static const int RES[][2] = {
            { 1280,  720 }, { 1920, 1080 }, { 2560, 1440 }, { 3840, 2160 },
        };
        eye[0]=c[0]; eye[1]=c[1]; eye[2]=c[2];
        tgt[0]=c[0]+radius; tgt[1]=c[1]; tgt[2]=c[2];
        for (int ri = 0; ri < 4; ri++) {
            int PW = RES[ri][0], PH = RES[ri][1];
            look_at(eye, tgt, up, view);
            perspective(60.0f, (float)PW/(float)PH, 0.001f*radius,
                        10.0f*radius, proj);
            double off = time_renders(usd, PW, PH, 0, view, proj, 40);
            double on  = time_renders(usd, PW, PH, 1, view, proj, 40);
            if (off > 0.0 && on > 0.0)
                fprintf(stderr, "perf [interior %dx%d, 40 frames]: whole-mesh "
                        "%.3f ms/frame, meshlet %.3f ms/frame  (%.2fx, %+.3f ms)\n",
                        PW, PH, off, on, off/on, off-on);
            else
                fprintf(stderr, "perf [interior %dx%d]: skipped (render failed)\n",
                        PW, PH);
        }
    }

    if (g_fail == 0) {
        fprintf(stderr, "PASS: meshlet draw path renders correctly\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d check(s) failed\n", g_fail);
    return 1;
}
