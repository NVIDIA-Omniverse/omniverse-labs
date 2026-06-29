// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * cgl_headless.c — Headless OpenGL context on macOS via CGL.
 *
 * macOS counterpart to egl_headless.c: creates a CGL OpenGL 4.1 Core
 * context with no drawable, then attaches an FBO (RGBA8 colour +
 * DEPTH24_STENCIL8) sized to the requested dimensions and binds it as
 * the active framebuffer. The viewer / gpu code reads from
 * GL_FRAMEBUFFER 0-bound at this scope, so glReadPixels in
 * viewer_render_to_rgba pulls bytes back from the FBO transparently.
 *
 * CGL is deprecated since macOS 10.14 but remains the only
 * documented way to obtain a headless GL context without a window.
 * GL_SILENCE_DEPRECATION suppresses the warning spam.
 */

#define GL_SILENCE_DEPRECATION 1

#include "egl_headless.h"

#include <OpenGL/OpenGL.h>
#include <OpenGL/gl3.h>
#include <stdio.h>
#include <stdlib.h>

struct EglHeadless {
    CGLContextObj ctx;
    CGLPixelFormatObj pix;
    GLuint fbo;
    GLuint color_rb;
    GLuint depth_stencil_rb;
    int width;
    int height;
};

static int rebuild_renderbuffers(EglHeadless* h, int w, int h_)
{
    glBindFramebuffer(GL_FRAMEBUFFER, h->fbo);

    glBindRenderbuffer(GL_RENDERBUFFER, h->color_rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h_);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_RENDERBUFFER, h->color_rb);

    glBindRenderbuffer(GL_RENDERBUFFER, h->depth_stencil_rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h_);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                              GL_RENDERBUFFER, h->depth_stencil_rb);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "cgl_headless: FBO incomplete after resize (0x%x)\n",
                (unsigned)status);
        return 0;
    }
    glViewport(0, 0, w, h_);
    return 1;
}

EglHeadless* egl_headless_create(int width, int height)
{
    if (width <= 0 || height <= 0) return NULL;

    EglHeadless* h = (EglHeadless*)calloc(1, sizeof(*h));
    if (!h) return NULL;

    CGLPixelFormatAttribute attrs[] = {
        kCGLPFAAccelerated,
        kCGLPFAOpenGLProfile,
        (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core,
        kCGLPFAColorSize, (CGLPixelFormatAttribute)32,
        kCGLPFADepthSize, (CGLPixelFormatAttribute)24,
        (CGLPixelFormatAttribute)0,
    };
    GLint npix = 0;
    CGLError err = CGLChoosePixelFormat(attrs, &h->pix, &npix);
    if (err != kCGLNoError || h->pix == NULL || npix == 0) {
        fprintf(stderr, "cgl_headless: CGLChoosePixelFormat failed (%d)\n", err);
        free(h);
        return NULL;
    }

    err = CGLCreateContext(h->pix, NULL, &h->ctx);
    if (err != kCGLNoError || h->ctx == NULL) {
        fprintf(stderr, "cgl_headless: CGLCreateContext failed (%d)\n", err);
        CGLDestroyPixelFormat(h->pix);
        free(h);
        return NULL;
    }

    err = CGLSetCurrentContext(h->ctx);
    if (err != kCGLNoError) {
        fprintf(stderr, "cgl_headless: CGLSetCurrentContext failed (%d)\n", err);
        CGLDestroyContext(h->ctx);
        CGLDestroyPixelFormat(h->pix);
        free(h);
        return NULL;
    }

    glGenFramebuffers(1, &h->fbo);
    glGenRenderbuffers(1, &h->color_rb);
    glGenRenderbuffers(1, &h->depth_stencil_rb);

    if (!rebuild_renderbuffers(h, width, height)) {
        CGLSetCurrentContext(NULL);
        CGLDestroyContext(h->ctx);
        CGLDestroyPixelFormat(h->pix);
        free(h);
        return NULL;
    }

    h->width = width;
    h->height = height;
    fprintf(stderr, "cgl_headless: context created (%dx%d)\n", width, height);
    return h;
}

void egl_headless_destroy(EglHeadless* h)
{
    if (!h) return;
    if (h->ctx) {
        CGLSetCurrentContext(h->ctx);
        if (h->fbo)              glDeleteFramebuffers(1, &h->fbo);
        if (h->color_rb)         glDeleteRenderbuffers(1, &h->color_rb);
        if (h->depth_stencil_rb) glDeleteRenderbuffers(1, &h->depth_stencil_rb);
        CGLSetCurrentContext(NULL);
        CGLDestroyContext(h->ctx);
    }
    if (h->pix) CGLDestroyPixelFormat(h->pix);
    free(h);
}

int egl_headless_make_current(EglHeadless* h)
{
    if (!h || !h->ctx) return 0;
    CGLError err = CGLSetCurrentContext(h->ctx);
    if (err != kCGLNoError) {
        fprintf(stderr, "cgl_headless: make_current failed (%d)\n", err);
        return 0;
    }
    /* Ensure our FBO is the active draw target — a sibling library may
     * have rebound the default framebuffer while we yielded the thread. */
    glBindFramebuffer(GL_FRAMEBUFFER, h->fbo);
    glViewport(0, 0, h->width, h->height);
    return 1;
}

void egl_headless_release_current(EglHeadless* h)
{
    (void)h;
    CGLSetCurrentContext(NULL);
}

int egl_headless_resize(EglHeadless* h, int width, int height)
{
    if (!h || width <= 0 || height <= 0) return 0;
    if (width == h->width && height == h->height) return 1;
    if (CGLSetCurrentContext(h->ctx) != kCGLNoError) return 0;
    if (!rebuild_renderbuffers(h, width, height)) return 0;
    h->width = width;
    h->height = height;
    return 1;
}

int egl_headless_has_current(void)
{
    return CGLGetCurrentContext() != NULL;
}
