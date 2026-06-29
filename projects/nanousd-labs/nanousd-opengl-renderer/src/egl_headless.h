// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_EGL_HEADLESS_H
#define NUSD_EGL_HEADLESS_H

/*
 * egl_headless.h — Headless EGL context via NVIDIA GPU device.
 *
 * Uses EGL_EXT_device_enumeration + EGL_EXT_platform_device to create
 * an OpenGL ES 3.1 context on an NVIDIA GPU without any X11 display.
 * Falls back to the default EGL display if no NVIDIA device is found.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EglHeadless EglHeadless;

/* Create a headless EGL context with the given pbuffer dimensions.
 * Returns NULL on failure. */
EglHeadless* egl_headless_create(int width, int height);

/* Destroy the headless EGL context. */
void egl_headless_destroy(EglHeadless* ctx);

/* Resize the pbuffer. Destroys the old surface and allocates a new one
 * with the requested dimensions, then re-makes the context current.
 * Returns 1 on success, 0 on failure (context retains old surface). */
int egl_headless_resize(EglHeadless* ctx, int width, int height);

/* Make this context current on the calling thread. Useful for embedders
 * that share the process with another GL context (e.g. omni.ui running
 * its own GLFW window). Returns 1 on success. */
int egl_headless_make_current(EglHeadless* ctx);

/* Release this context if it is current on the calling thread, leaving
 * EGL_NO_CONTEXT current. Lets a sibling library (e.g. omni.ui) re-make
 * its own context current. */
void egl_headless_release_current(EglHeadless* ctx);

/* Returns 1 if some GL context (ours, the host's, anyone's) is already
 * current on the calling thread. Used by viewer_create to decide
 * whether to spin up a fresh headless context or piggy-back on one
 * the worker thread already pre-warmed. Portable wrapper around
 * eglGetCurrentContext / CGLGetCurrentContext. */
int egl_headless_has_current(void);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_EGL_HEADLESS_H */
