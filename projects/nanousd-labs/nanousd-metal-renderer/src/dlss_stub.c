// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * dlss_stub.c — Simulated DLSS when SDK is unavailable.
 *
 * Returns success from all functions so the viewer exercises the full
 * DLSS render pipeline (offscreen targets, MRT, motion vectors, blit upscale).
 * The actual upscaling is a bilinear blit in gpu_vulkan.c rather than the
 * NVIDIA neural network. Plug in dlss_wrapper.cpp + NGX SDK for real DLSS.
 */

#include "dlss.h"
#include <stdlib.h>
#include <stdio.h>

/* Dummy context — just needs to be non-NULL. */
struct DlssContext {
    int dummy;
};

/* Render scale factors per quality mode. */
static const float quality_scales[] = {
    0.50f,  /* DLSS_QUALITY_PERFORMANCE   */
    0.58f,  /* DLSS_QUALITY_BALANCED      */
    0.67f,  /* DLSS_QUALITY_QUALITY       */
    0.77f,  /* DLSS_QUALITY_ULTRA_QUALITY  */
    1.00f,  /* DLSS_QUALITY_DLAA           */
};

uint32_t dlss_get_instance_extensions(const char** out_names, uint32_t max_count)
{
    (void)out_names;
    (void)max_count;
    return 0;
}

uint32_t dlss_get_device_extensions(void* vk_instance, void* vk_physical_device,
                                     const char** out_names, uint32_t max_count)
{
    (void)vk_instance;
    (void)vk_physical_device;
    (void)out_names;
    (void)max_count;
    return 0;
}

DlssContext* dlss_init(const DlssInitInfo* info)
{
    (void)info;
    DlssContext* ctx = (DlssContext*)calloc(1, sizeof(DlssContext));
    if (ctx)
        fprintf(stderr, "dlss_stub: DLSS simulated (bilinear upscale, no NGX SDK)\n");
    return ctx;
}

int dlss_sr_available(DlssContext* ctx)
{
    return ctx != NULL;
}

int dlss_get_optimal_settings(DlssContext* ctx, DlssOptimalSettings* settings)
{
    if (!ctx || !settings) return 0;

    int q = settings->quality;
    if (q < 0 || q >= DLSS_QUALITY_COUNT) q = DLSS_QUALITY_BALANCED;

    float scale = quality_scales[q];
    settings->render_width  = (uint32_t)(settings->display_width  * scale + 0.5f);
    settings->render_height = (uint32_t)(settings->display_height * scale + 0.5f);

    /* Ensure even dimensions and at least 2. */
    if (settings->render_width  < 2) settings->render_width  = 2;
    if (settings->render_height < 2) settings->render_height = 2;
    settings->render_width  &= ~1u;
    settings->render_height &= ~1u;

    settings->sharpness = 0.0f;
    return 1;
}

int dlss_create_feature(DlssContext* ctx, const DlssOptimalSettings* settings, void* cmd)
{
    (void)ctx;
    (void)settings;
    (void)cmd;
    return 1;
}

int dlss_evaluate(DlssContext* ctx, const DlssEvalParams* params)
{
    (void)ctx;
    (void)params;
    /* No-op: gpu_vulkan.c blits the offscreen color to the swapchain. */
    return 1;
}

void dlss_release_feature(DlssContext* ctx)
{
    (void)ctx;
}

void dlss_shutdown(DlssContext* ctx)
{
    free(ctx);
}
