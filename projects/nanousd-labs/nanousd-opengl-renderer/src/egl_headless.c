// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * egl_headless.c — Headless EGL context via NVIDIA GPU device.
 *
 * Uses EGL_EXT_device_enumeration + EGL_EXT_platform_device to create
 * an OpenGL ES 3.1 context backed by an NVIDIA GPU, without X11.
 * The pbuffer surface acts as the default framebuffer for glReadPixels.
 */

#include "egl_headless.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef EGL_EXT_platform_device
#define EGL_PLATFORM_DEVICE_EXT 0x313F
#endif

struct EglHeadless {
    EGLDisplay display;
    EGLContext context;
    EGLSurface surface;
    EGLConfig  config;
    int width;
    int height;
};

EglHeadless* egl_headless_create(int width, int height)
{
    EglHeadless* ctx = (EglHeadless*)calloc(1, sizeof(EglHeadless));
    if (!ctx) return NULL;
    ctx->width = width;
    ctx->height = height;

    /* Check client extensions for device enumeration */
    const char* client_exts = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    if (!client_exts ||
        !strstr(client_exts, "EGL_EXT_device_enumeration") ||
        !strstr(client_exts, "EGL_EXT_platform_device")) {
        fprintf(stderr, "egl_headless: required EGL extensions not available\n");
        free(ctx);
        return NULL;
    }

    /* Load extension functions */
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    PFNEGLQUERYDEVICESTRINGEXTPROC eglQueryDeviceStringEXT =
        (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress("eglQueryDeviceStringEXT");

    if (!eglQueryDevicesEXT || !eglGetPlatformDisplayEXT) {
        fprintf(stderr, "egl_headless: failed to load EGL extension functions\n");
        free(ctx);
        return NULL;
    }

    /* Enumerate EGL devices */
    EGLDeviceEXT devices[16];
    EGLint num_devices = 0;
    if (!eglQueryDevicesEXT(16, devices, &num_devices) || num_devices == 0) {
        fprintf(stderr, "egl_headless: no EGL devices found\n");
        free(ctx);
        return NULL;
    }

    fprintf(stderr, "egl_headless: found %d EGL device(s)\n", num_devices);

    /* Find an NVIDIA device (prefer it over Mesa/llvmpipe) */
    EGLDeviceEXT nvidia_device = EGL_NO_DEVICE_EXT;
    EGLDeviceEXT fallback_device = EGL_NO_DEVICE_EXT;

    for (EGLint i = 0; i < num_devices; i++) {
        const char* exts = eglQueryDeviceStringEXT
            ? eglQueryDeviceStringEXT(devices[i], EGL_EXTENSIONS)
            : NULL;
        fprintf(stderr, "egl_headless: device %d extensions: %s\n", i,
                exts ? exts : "(none)");

        /* Try to get a display from this device to check the renderer */
        EGLDisplay test_dpy = eglGetPlatformDisplayEXT(
            EGL_PLATFORM_DEVICE_EXT, devices[i], NULL);
        if (test_dpy != EGL_NO_DISPLAY) {
            EGLint major, minor;
            if (eglInitialize(test_dpy, &major, &minor)) {
                /* Bind OpenGL ES API before querying configs */
                eglBindAPI(EGL_OPENGL_ES_API);

                /* Check if this device supports GLES rendering */
                EGLint config_attribs[] = {
                    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                    EGL_RED_SIZE, 8,
                    EGL_GREEN_SIZE, 8,
                    EGL_BLUE_SIZE, 8,
                    EGL_ALPHA_SIZE, 8,
                    EGL_DEPTH_SIZE, 24,
                    EGL_NONE
                };

                EGLConfig config;
                EGLint num_configs;
                if (eglChooseConfig(test_dpy, config_attribs, &config, 1,
                                    &num_configs) && num_configs > 0) {
                    const char* vendor = eglQueryString(test_dpy, EGL_VENDOR);
                    fprintf(stderr, "egl_headless: device %d vendor: %s\n",
                            i, vendor ? vendor : "(unknown)");

                    if (vendor && strstr(vendor, "NVIDIA")) {
                        nvidia_device = devices[i];
                        eglTerminate(test_dpy);
                        break;
                    }
                    if (fallback_device == EGL_NO_DEVICE_EXT)
                        fallback_device = devices[i];
                }
                eglTerminate(test_dpy);
            }
        }
    }

    EGLDeviceEXT chosen = (nvidia_device != EGL_NO_DEVICE_EXT)
        ? nvidia_device : fallback_device;
    if (chosen == EGL_NO_DEVICE_EXT) {
        fprintf(stderr, "egl_headless: no suitable EGL device found\n");
        free(ctx);
        return NULL;
    }

    /* Create display from chosen device */
    ctx->display = eglGetPlatformDisplayEXT(
        EGL_PLATFORM_DEVICE_EXT, chosen, NULL);
    if (ctx->display == EGL_NO_DISPLAY) {
        fprintf(stderr, "egl_headless: eglGetPlatformDisplay failed\n");
        free(ctx);
        return NULL;
    }

    EGLint major, minor;
    if (!eglInitialize(ctx->display, &major, &minor)) {
        fprintf(stderr, "egl_headless: eglInitialize failed (0x%x)\n",
                eglGetError());
        free(ctx);
        return NULL;
    }
    fprintf(stderr, "egl_headless: EGL %d.%d initialized\n", major, minor);

    eglBindAPI(EGL_OPENGL_ES_API);

    /* Choose config — request 4x MSAA for smoother curve silhouettes.
     * If the driver can't satisfy the multisample request we fall back
     * to a 1-sample config below. */
    EGLint msaa_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_SAMPLE_BUFFERS, 1,
        EGL_SAMPLES, 4,
        EGL_NONE
    };
    EGLint plain_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_NONE
    };

    EGLConfig config;
    EGLint num_configs = 0;
    int msaa = 0;
    if (eglChooseConfig(ctx->display, msaa_attribs, &config, 1, &num_configs)
            && num_configs > 0) {
        msaa = 1;
        fprintf(stderr, "egl_headless: 4x MSAA enabled\n");
    } else if (!eglChooseConfig(ctx->display, plain_attribs, &config, 1,
                                &num_configs) || num_configs == 0) {
        fprintf(stderr, "egl_headless: eglChooseConfig failed\n");
        eglTerminate(ctx->display);
        free(ctx);
        return NULL;
    }

    ctx->config = config;

    /* Create pbuffer surface */
    EGLint pbuf_attribs[] = {
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_NONE
    };
    ctx->surface = eglCreatePbufferSurface(ctx->display, config, pbuf_attribs);
    if (ctx->surface == EGL_NO_SURFACE) {
        fprintf(stderr, "egl_headless: eglCreatePbufferSurface failed (0x%x)\n",
                eglGetError());
        eglTerminate(ctx->display);
        free(ctx);
        return NULL;
    }

    /* Create OpenGL ES 3.2 context (tessellation + geometry shaders are
     * core in 3.2, required by the curve tube renderer). */
    EGLint ctx_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 2,
        EGL_NONE
    };
    ctx->context = eglCreateContext(ctx->display, config, EGL_NO_CONTEXT,
                                    ctx_attribs);
    if (ctx->context == EGL_NO_CONTEXT) {
        fprintf(stderr, "egl_headless: eglCreateContext failed (0x%x)\n",
                eglGetError());
        eglDestroySurface(ctx->display, ctx->surface);
        eglTerminate(ctx->display);
        free(ctx);
        return NULL;
    }

    /* If some other library (e.g. omni.ui's GLFW) already has a GL
     * context current on this thread, eglMakeCurrent below fails with
     * EGL_BAD_ACCESS. Releasing the thread's prior EGL state first
     * lets our pbuffer become current without disturbing the host. */
    eglReleaseThread();
    eglBindAPI(EGL_OPENGL_ES_API);

    fprintf(stderr,
        "egl_headless: pre-makeCurrent: display=%p surface=%p context=%p api=0x%x\n",
        (void*)ctx->display, (void*)ctx->surface, (void*)ctx->context,
        (unsigned)eglQueryAPI());

    /* Make current */
    EGLBoolean mc_ok = eglMakeCurrent(ctx->display, ctx->surface, ctx->surface, ctx->context);
    EGLint mc_err = eglGetError();
    if (!mc_ok) {
        fprintf(stderr, "egl_headless: eglMakeCurrent failed (rv=%d, err=0x%x)\n",
                (int)mc_ok, (unsigned)mc_err);
        eglDestroyContext(ctx->display, ctx->context);
        eglDestroySurface(ctx->display, ctx->surface);
        eglTerminate(ctx->display);
        free(ctx);
        return NULL;
    }

    fprintf(stderr, "egl_headless: context created (%dx%d)\n", width, height);
    return ctx;
}

void egl_headless_destroy(EglHeadless* ctx)
{
    if (!ctx) return;
    eglMakeCurrent(ctx->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(ctx->display, ctx->context);
    eglDestroySurface(ctx->display, ctx->surface);
    eglTerminate(ctx->display);
    free(ctx);
}

int egl_headless_make_current(EglHeadless* ctx)
{
    if (!ctx) return 0;
    if (!eglMakeCurrent(ctx->display, ctx->surface, ctx->surface, ctx->context)) {
        fprintf(stderr, "egl_headless: make_current failed (0x%x)\n",
                eglGetError());
        return 0;
    }
    return 1;
}

void egl_headless_release_current(EglHeadless* ctx)
{
    if (!ctx) return;
    eglMakeCurrent(ctx->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
}

int egl_headless_has_current(void)
{
    return eglGetCurrentContext() != EGL_NO_CONTEXT;
}

int egl_headless_resize(EglHeadless* ctx, int width, int height)
{
    if (!ctx || width <= 0 || height <= 0) return 0;
    if (width == ctx->width && height == ctx->height) return 1;

    EGLint pbuf_attribs[] = {
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_NONE
    };
    EGLSurface new_surface = eglCreatePbufferSurface(ctx->display, ctx->config, pbuf_attribs);
    if (new_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "egl_headless: resize eglCreatePbufferSurface failed (0x%x)\n",
                eglGetError());
        return 0;
    }
    if (!eglMakeCurrent(ctx->display, new_surface, new_surface, ctx->context)) {
        fprintf(stderr, "egl_headless: resize eglMakeCurrent failed (0x%x)\n",
                eglGetError());
        eglDestroySurface(ctx->display, new_surface);
        return 0;
    }
    eglDestroySurface(ctx->display, ctx->surface);
    ctx->surface = new_surface;
    ctx->width = width;
    ctx->height = height;
    return 1;
}
