// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_GL_THREAD_H
#define NUSD_GL_THREAD_H

/*
 * gl_thread.h — Single-worker pthread that owns the headless EGL context.
 *
 * Embedders (omni.ui, ovgear) already have a GL context current on the
 * main thread. EGL refuses to switch context bindings on a thread that
 * has another driver's GL context bound, so all viewer GL work happens
 * on a dedicated worker thread instead. The functions in this header
 * marshal a request to the worker and block until it returns.
 *
 * The worker thread is created lazily on the first nusd_gl_submit_*
 * call and torn down at process exit (no explicit shutdown — the
 * pthread is detached).
 */

#include "viewer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Pre-warm the worker thread and its EGL context. NVIDIA's driver gives
 * up on EGL after Vulkan initialization (via Vulkan-WSI or otherwise),
 * so embedders that load Vulkan eagerly (omni.ui's ui.init()) must call
 * this **before** their Vulkan setup. After warmup the worker thread
 * holds a current GLES context that subsequent viewer_create calls
 * inherit. Returns 1 on success, 0 on failure. */
int     nusd_gl_warmup(int width, int height);

/* Threaded equivalents of the public viewer_* API. Each blocks the
 * caller until the worker thread completes the request. */

Viewer* nusd_gl_viewer_create(const char* usd_path, int width, int height,
                              int max_tex_size, const char* envmap_path,
                              int headless);

void    nusd_gl_viewer_destroy(Viewer* viewer);

void    nusd_gl_viewer_resize(Viewer* viewer, int width, int height);

void    nusd_gl_viewer_set_render_mode(Viewer* viewer, ViewerRenderMode mode);

void    nusd_gl_viewer_set_overlay_enabled(Viewer* viewer, int enabled);

int     nusd_gl_viewer_render_to_rgba(Viewer* viewer, int width, int height,
                                      const float view16[16],
                                      const float proj16[16],
                                      const float eye3[3],
                                      unsigned char* out_rgba);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_GL_THREAD_H */
