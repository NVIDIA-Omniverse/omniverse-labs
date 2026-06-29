// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * gl_thread.c — Worker thread that owns the headless EGL context.
 *
 * One worker for the whole library. The thread is started on demand on
 * the first request and serves all subsequent requests from a single
 * mutex-protected slot. Each public nusd_gl_* function fills the slot,
 * signals the worker, and waits for the response. This serializes all
 * viewer operations through one GL context — which is exactly what the
 * standalone OpenGL ES driver supports.
 *
 * The slot uses a typed Request union so we don't allocate per call.
 */

#include "gl_thread.h"
#include "viewer.h"
#include "egl_headless.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Single global headless GL context owned by the worker thread. Created
 * on the first warmup or viewer_create. Once created, all viewer_create
 * / viewer_resize / viewer_destroy operations on the worker thread
 * reuse it via egl_headless_has_current() in viewer.c. The backend
 * underneath is platform-specific: EGL pbuffer on Linux, CGL + FBO on
 * macOS — the egl_headless.h API is the same on both. */
static EglHeadless* g_egl_ctx = NULL;
static int          g_egl_w = 1280;
static int          g_egl_h = 720;

typedef enum {
    OP_NONE = 0,
    OP_WARMUP,
    OP_CREATE,
    OP_DESTROY,
    OP_RESIZE,
    OP_RENDER,
    OP_SET_MODE,
    OP_SET_OVERLAY,
} Op;

typedef struct {
    Op op;

    /* CREATE */
    const char* path;
    const char* envmap;
    int max_tex;

    /* shared (DESTROY, RESIZE, RENDER, SET_*) */
    Viewer* viewer;
    int w, h;

    /* RENDER */
    const float* view;     /* 16 floats */
    const float* proj;     /* 16 floats */
    const float* eye;      /* 3 floats  */
    unsigned char* out_rgba;

    /* SET_MODE / SET_OVERLAY */
    int param;

    /* response */
    Viewer* result_viewer;
    int     result_ok;
    int     done;
} Request;

static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cv_req = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_cv_resp = PTHREAD_COND_INITIALIZER;
static Request*        g_pending = NULL;
static int             g_thread_started = 0;
static pthread_t       g_tid;

static int ensure_egl(int width, int height)
{
    if (g_egl_ctx) return 1;
    g_egl_ctx = egl_headless_create(width, height);
    if (g_egl_ctx) {
        g_egl_w = width;
        g_egl_h = height;
        return 1;
    }
    return 0;
}

static void execute(Request* r)
{
    switch (r->op) {
        case OP_WARMUP:
            r->result_ok = ensure_egl(r->w, r->h);
            break;
        case OP_CREATE:
            /* Make sure the worker has a headless context current before
             * viewer_create; viewer.c will then reuse it instead of
             * creating its own (which on Linux fails after Vulkan
             * init — NVIDIA's driver refuses fresh EGL setup once
             * Vulkan-WSI has loaded). */
            if (!ensure_egl(r->w, r->h)) {
                r->result_ok = 0;
                break;
            }
            /* Resize the existing surface to the requested size so
             * gpu_init's glViewport and subsequent glReadPixels are
             * sized correctly. */
            if (r->w != g_egl_w || r->h != g_egl_h) {
                if (egl_headless_resize(g_egl_ctx, r->w, r->h)) {
                    g_egl_w = r->w;
                    g_egl_h = r->h;
                }
            }
            r->result_viewer = viewer_create(r->path, r->w, r->h,
                                             r->max_tex, r->envmap, 1, 1);
            r->result_ok = (r->result_viewer != NULL);
            break;
        case OP_DESTROY:
            viewer_destroy(r->viewer);
            r->result_ok = 1;
            break;
        case OP_RESIZE:
            /* Resize both the global headless surface and the viewer's viewport. */
            if (g_egl_ctx) {
                if (egl_headless_resize(g_egl_ctx, r->w, r->h)) {
                    g_egl_w = r->w;
                    g_egl_h = r->h;
                }
            }
            viewer_resize(r->viewer, r->w, r->h);
            r->result_ok = 1;
            break;
        case OP_RENDER:
            /* Make sure the surface matches the requested render size. */
            if (g_egl_ctx && (r->w != g_egl_w || r->h != g_egl_h)) {
                if (egl_headless_resize(g_egl_ctx, r->w, r->h)) {
                    g_egl_w = r->w;
                    g_egl_h = r->h;
                }
            }
            r->result_ok = viewer_render_to_rgba(
                r->viewer, r->w, r->h, r->view, r->proj, r->eye, r->out_rgba);
            break;
        case OP_SET_MODE:
            viewer_set_render_mode(r->viewer, (ViewerRenderMode)r->param);
            r->result_ok = 1;
            break;
        case OP_SET_OVERLAY:
            viewer_set_overlay_enabled(r->viewer, r->param);
            r->result_ok = 1;
            break;
        default:
            r->result_ok = 0;
            break;
    }
}

static void* worker_main(void* arg)
{
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_mtx);
        while (g_pending == NULL)
            pthread_cond_wait(&g_cv_req, &g_mtx);
        Request* r = g_pending;
        pthread_mutex_unlock(&g_mtx);

        execute(r);

        pthread_mutex_lock(&g_mtx);
        r->done = 1;
        g_pending = NULL;
        pthread_cond_broadcast(&g_cv_resp);
        pthread_mutex_unlock(&g_mtx);
    }
    return NULL;
}

static void ensure_worker(void)
{
    pthread_mutex_lock(&g_mtx);
    if (!g_thread_started) {
        g_thread_started = 1;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        if (pthread_create(&g_tid, &attr, worker_main, NULL) != 0) {
            fprintf(stderr, "gl_thread: pthread_create failed\n");
            g_thread_started = 0;
        }
        pthread_attr_destroy(&attr);
    }
    pthread_mutex_unlock(&g_mtx);
}

static int submit(Request* r)
{
    ensure_worker();
    if (!g_thread_started) return 0;

    pthread_mutex_lock(&g_mtx);
    while (g_pending != NULL)
        pthread_cond_wait(&g_cv_resp, &g_mtx);
    r->done = 0;
    g_pending = r;
    pthread_cond_signal(&g_cv_req);
    while (!r->done)
        pthread_cond_wait(&g_cv_resp, &g_mtx);
    pthread_mutex_unlock(&g_mtx);
    return r->result_ok;
}

int nusd_gl_warmup(int width, int height)
{
    Request r;
    memset(&r, 0, sizeof(r));
    r.op = OP_WARMUP;
    r.w = width > 0 ? width : 1280;
    r.h = height > 0 ? height : 720;
    submit(&r);
    return r.result_ok;
}

Viewer* nusd_gl_viewer_create(const char* usd_path, int width, int height,
                              int max_tex_size, const char* envmap_path,
                              int headless)
{
    (void)headless; /* embedders always headless */
    Request r;
    memset(&r, 0, sizeof(r));
    r.op = OP_CREATE;
    r.path = usd_path;
    r.w = width;
    r.h = height;
    r.max_tex = max_tex_size;
    r.envmap = envmap_path;
    submit(&r);
    return r.result_viewer;
}

void nusd_gl_viewer_destroy(Viewer* viewer)
{
    if (!viewer) return;
    Request r;
    memset(&r, 0, sizeof(r));
    r.op = OP_DESTROY;
    r.viewer = viewer;
    submit(&r);
}

void nusd_gl_viewer_resize(Viewer* viewer, int width, int height)
{
    if (!viewer) return;
    Request r;
    memset(&r, 0, sizeof(r));
    r.op = OP_RESIZE;
    r.viewer = viewer;
    r.w = width;
    r.h = height;
    submit(&r);
}

void nusd_gl_viewer_set_render_mode(Viewer* viewer, ViewerRenderMode mode)
{
    if (!viewer) return;
    Request r;
    memset(&r, 0, sizeof(r));
    r.op = OP_SET_MODE;
    r.viewer = viewer;
    r.param = (int)mode;
    submit(&r);
}

void nusd_gl_viewer_set_overlay_enabled(Viewer* viewer, int enabled)
{
    if (!viewer) return;
    Request r;
    memset(&r, 0, sizeof(r));
    r.op = OP_SET_OVERLAY;
    r.viewer = viewer;
    r.param = enabled ? 1 : 0;
    submit(&r);
}

int nusd_gl_viewer_render_to_rgba(Viewer* viewer, int width, int height,
                                  const float view16[16],
                                  const float proj16[16],
                                  const float eye3[3],
                                  unsigned char* out_rgba)
{
    if (!viewer) return 0;
    Request r;
    memset(&r, 0, sizeof(r));
    r.op = OP_RENDER;
    r.viewer = viewer;
    r.w = width;
    r.h = height;
    r.view = view16;
    r.proj = proj16;
    r.eye = eye3;
    r.out_rgba = out_rgba;
    submit(&r);
    return r.result_ok;
}
