// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_DLSS_H
#define NUSD_DLSS_H

/*
 * dlss.h — Pure C interface to NVIDIA DLSS (via NGX SDK)
 *
 * All NGX types are hidden behind the opaque DlssContext.
 * When compiled without HAS_DLSS, dlss_stub.c provides no-op implementations.
 * When compiled with HAS_DLSS, dlss_wrapper.cpp calls the real NGX API.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DlssContext DlssContext;

typedef enum {
    DLSS_QUALITY_PERFORMANCE   = 0,   /* ~50% render scale */
    DLSS_QUALITY_BALANCED      = 1,   /* ~58% render scale */
    DLSS_QUALITY_QUALITY       = 2,   /* ~67% render scale */
    DLSS_QUALITY_ULTRA_QUALITY = 3,   /* ~77% render scale */
    DLSS_QUALITY_DLAA          = 4,   /* 100% render scale (anti-aliasing only) */
    DLSS_QUALITY_COUNT         = 5,
} DlssQualityMode;

static const char* dlss_quality_names[] = {
    "Performance", "Balanced", "Quality", "Ultra Quality", "DLAA"
};

/* Vulkan handles needed by NGX init (passed as void* so this header stays C). */
typedef struct {
    void*    vk_instance;
    void*    vk_physical_device;
    void*    vk_device;
    uint32_t queue_family;
    void*    vk_queue;
} DlssInitInfo;

/* Query result from dlss_get_optimal_settings. */
typedef struct {
    uint32_t         display_width;
    uint32_t         display_height;
    DlssQualityMode  quality;
    /* Filled in by dlss_get_optimal_settings(): */
    uint32_t         render_width;
    uint32_t         render_height;
    float            sharpness;
} DlssOptimalSettings;

/* Per-frame evaluation parameters.
 * All VkImage / VkImageView / VkCommandBuffer handles passed as void*. */
typedef struct {
    void*    color_image;       /* VkImage,     render res, RGBA8/RGBA16F     */
    void*    color_view;        /* VkImageView                                */
    void*    depth_image;       /* VkImage,     render res, D32F or R32F      */
    void*    depth_view;        /* VkImageView                                */
    int      depth_is_d32;      /* 1 = D32_SFLOAT depth attachment, 0 = R32F  */
    void*    mv_image;          /* VkImage,     render res, RG16_SFLOAT       */
    void*    mv_view;           /* VkImageView                                */
    void*    output_image;      /* VkImage,     display res, RGBA8            */
    void*    output_view;       /* VkImageView                                */
    void*    cmd;               /* VkCommandBuffer (must be recording)        */
    uint32_t render_width;
    uint32_t render_height;
    uint32_t display_width;
    uint32_t display_height;
    float    jitter_x;          /* sub-pixel jitter in pixel coordinates      */
    float    jitter_y;
    float    delta_time_ms;
    int      reset;             /* 1 = reset temporal history this frame      */
} DlssEvalParams;

/* ---- Vulkan extension queries (call BEFORE vkCreateInstance / vkCreateDevice) ---- */

/* Query Vulkan instance extensions required by NGX.
 * Fills out_names (up to max_count) with static extension name strings.
 * Returns number of required extensions (0 on error or stub). */
uint32_t dlss_get_instance_extensions(const char** out_names, uint32_t max_count);

/* Query Vulkan device extensions required by NGX.
 * Call after vkCreateInstance but before vkCreateDevice.
 * Returns number of required extensions (0 on error or stub). */
uint32_t dlss_get_device_extensions(void* vk_instance, void* vk_physical_device,
                                     const char** out_names, uint32_t max_count);

/* ---- Lifecycle ---- */

/* Initialize NGX runtime. Returns NULL if DLSS is not available
 * (non-NVIDIA GPU, driver too old, missing runtime libraries). */
DlssContext* dlss_init(const DlssInitInfo* info);

/* Returns 1 if DLSS Super Resolution is supported on this GPU. */
int dlss_sr_available(DlssContext* ctx);

/* Query the optimal render resolution for a given display size and quality mode.
 * Fills in settings->render_width, render_height, sharpness. Returns 1 on success. */
int dlss_get_optimal_settings(DlssContext* ctx, DlssOptimalSettings* settings);

/* Create (or re-create) the DLSS-SR feature at the specified resolutions.
 * cmd must be a command buffer in recording state (used for internal resource init).
 * Returns 1 on success. */
int dlss_create_feature(DlssContext* ctx, const DlssOptimalSettings* settings, void* cmd);

/* Run DLSS upscaling. Inserts compute work into params->cmd.
 * The command buffer must be in recording state. Returns 1 on success. */
int dlss_evaluate(DlssContext* ctx, const DlssEvalParams* params);

/* Release the current DLSS feature (keeps NGX alive for re-creation). */
void dlss_release_feature(DlssContext* ctx);

/* Full shutdown of NGX. Frees ctx. */
void dlss_shutdown(DlssContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_DLSS_H */
