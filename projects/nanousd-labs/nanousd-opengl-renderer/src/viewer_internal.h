// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_VIEWER_INTERNAL_H
#define NUSD_VIEWER_INTERNAL_H

/*
 * viewer_internal.h — private struct for viewer.c.
 *
 * No camera state, no input state: view/proj/eye are caller-supplied
 * on each call to viewer_render_to_rgba. The interactive viewer is
 * Python (python/nusd_gles/viewer.py).
 */

#include "viewer.h"
#include "egl_headless.h"
#include "gpu.h"
#include "scene.h"
#include "material.h"

#include <stdint.h>

enum {
    MODE_RASTER   = 0,
    MODE_MATERIAL = 1,
    MODE_COUNT    = 2,
};

/* Per-meshlet cull data the draw path reads each frame. Copied out of the
 * arena-backed Scene by build_meshlet_index_buffer, because
 * scene_release_mesh_payloads frees that arena after GPU upload. */
typedef struct {
    float    center[3];
    float    radius;
    uint32_t triangle_count;
} ViewerMeshlet;

typedef struct {
    uint32_t curve_id;
    uint64_t index_byte_offset;
    uint32_t npatches;
    int32_t  vertex_offset;
    int      index_type_bits;
    float    bounds_min[3];
    float    bounds_max[3];
} ViewerCurveDraw;

struct Viewer {
    Gpu*        gpu;
    Scene*      scene;

    GpuBuffer   vertex_buffer;
    GpuBuffer   index_buffer;
    GpuBuffer*  vertex_buffers;
    GpuBuffer*  index_buffers;
    int         buffer_count;
    GpuPipeline pipeline;

    /* Meshlet draw path (NUSD_MESHLET_DRAW=1, raster mode). meshlet_index_buffer
     * holds every meshlet's mesh-local triangles, meshlet-record ordered;
     * meshlet_ibo_off[r] is record r's first index (prefix sums, nmeshlets+1
     * entries). meshlet_draw is 1 when the path is built and active. */
    GpuBuffer      meshlet_index_buffer;
    uint32_t*      meshlet_ibo_off;
    ViewerMeshlet* meshlet_recs;
    int            meshlet_draw;

    /* Material system */
    MaterialCollection* materials;
    GpuPipeline         mat_pipeline;
    GpuPipeline         mat_instanced_pipeline;  /* compact-PI batches (instanced) */
    GpuBuffer           mat_vertex_buffer;
    int                 batch_ubo_slot;          /* first mesh-UBO slot for PI batches */
    int                 instance_buffer_ready;   /* gpu instance VBO uploaded */
    int                 has_materials;
    int                 fallback_material_index;
    GpuPipeline         shadow_pipeline;
    SceneLight          shadow_lights[GPU_MAX_SCENE_LIGHTS];
    int                 shadow_light_count;
    int                 ptex_color_buffer_uploaded;

    /* Mesh draw order — sorted by material_index so the per-frame loop
     * issues consecutive same-material draws together, letting
     * gpu_cmd_bind_material's "same as last" cache short-circuit ~95%
     * of the UBO uploads + texture binds on warehouse-style scenes. */
    int*                draw_order;

    int width, height;
    int render_mode;
    int overlay_enabled;   /* HUD on/off — set by viewer_set_overlay_enabled */

    uint32_t total_vertices;
    uint32_t total_indices;
    int      total_meshes;

    /* HUD stats (refreshed every 0.5 s inside viewer_render_to_rgba). */
    char     scene_name[256];
    double   fps_time;
    int      fps_frames;
    float    fps_display;
    long     rss_kb;
    int      gpu_mem_mb;
    int      cpu_temp_c;
    float    ui_scale;     /* HUD scale; 1.0 default, embedders may override */

    /* BasisCurves smoke test — gated by NUSD_CURVES_SMOKETEST env var.
     * Renders one hardcoded 4-CV Bezier patch each frame to validate the
     * tess pipeline + curve shaders before phase 2 (scene loader) lands.
     * Reuses the per-mesh UBO at slot `scene->nmeshes` (one extra slot
     * is allocated when this is enabled). */
    int         curves_smoketest;          /* 0 = off, 1 = on */
    GpuPipeline curve_pipeline;
    GpuBuffer   curve_vertex_buffer;
    GpuBuffer   curve_index_buffer;
    ViewerCurveDraw* curve_draws;
    int         curve_draw_count;
    int         curve_ubo_slot;            /* index into mesh UBO */

    /* Headless GL context. NULL when the GL context was inherited
     * from the caller (e.g. ovgear's window). */
    EglHeadless* egl_ctx;
};

#endif /* NUSD_VIEWER_INTERNAL_H */
