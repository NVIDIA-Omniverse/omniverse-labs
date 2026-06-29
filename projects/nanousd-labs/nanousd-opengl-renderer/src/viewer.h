// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_VIEWER_H
#define NUSD_VIEWER_H

/*
 * viewer.h — Embeddable USD scene renderer.
 *
 * Public API: open a USD stage on a headless GL context, render frames
 * with caller-supplied view/proj matrices, read pixels back as RGBA8.
 *
 * The standalone interactive viewer is implemented in Python on top of
 * this API (see python/nusd_gles/viewer.py). There is no native
 * interactive entry point.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Viewer Viewer;

/* Create viewer, load USD file, initialize GPU.
 *   max_tex_size — caps texture resolution (0 → default 512).
 *   envmap_path  — optional HDR equirectangular for IBL (NULL → none).
 *   headless     — currently always required; reserved for future use.
 * Returns NULL on failure. */
Viewer* viewer_create(const char* usd_path, int width, int height,
                      int max_tex_size, const char* envmap_path,
                      int headless, int enable_materials);

/* Resize the framebuffer. For inherited GL contexts the embedder is
 * responsible for the surface; we just update viewport state. */
void    viewer_resize(Viewer* viewer, int width, int height);

/* Render one frame using caller-supplied camera matrices and read
 * back RGBA8 pixels (top-left origin, vertically flipped from GL's
 * native bottom-up). view16 / proj16 are column-major 4×4 matrices.
 * Returns 1 on success. */
int     viewer_render_to_rgba(Viewer* viewer, int width, int height,
                              const float view16[16], const float proj16[16],
                              const float eye3[3], unsigned char* out_rgba);

/* Renderer mode used by subsequent renders. */
typedef enum {
    VIEWER_MODE_RASTER   = 0,
    VIEWER_MODE_MATERIAL = 1,
} ViewerRenderMode;
void    viewer_set_render_mode(Viewer* viewer, ViewerRenderMode mode);

/* Suppress the on-render text overlay (HUD). Embedders that draw
 * their own UI typically disable it. */
void    viewer_set_overlay_enabled(Viewer* viewer, int enabled);

/* OVRTX-style exposure/tonemap scale for built-in shaders. */
void    viewer_set_tone_mapping(Viewer* viewer, float exposure_scale,
                                float sky_scale, float white_point_scale,
                                unsigned int flags);

/* Return the loaded scene's world-space AABB. Both arrays receive 3
 * floats. Returns 1 on success; on failure (no scene loaded) the
 * arrays are zeroed and 0 is returned. */
int     viewer_get_scene_bounds(Viewer* viewer, float out_min[3], float out_max[3]);

/* Authored stage up axis: 0 = X, 1 = Y, 2 = Z. Returns 1 (Y-up) if no
 * scene is loaded. */
int     viewer_get_scene_up_axis(Viewer* viewer);
int     viewer_get_scene_has_authored_light(Viewer* viewer);
uint64_t viewer_get_gpu_memory_used(Viewer* viewer);

/* OVRTX-style live mesh mutation API. Transforms are row-major
 * column-vector matrices (translation in [3], [7], [11]), matching
 * the Vulkan backend's nu_set_transforms convention. */
int     viewer_get_mesh_count(Viewer* viewer);
int     viewer_get_mesh_name(Viewer* viewer, int mesh_id, char* out_buf, int buf_cap);
int     viewer_get_mesh_transform(Viewer* viewer, int mesh_id, float out_mat4x4[16]);
int     viewer_set_transforms(Viewer* viewer, const int* ids, const float* mat4x4s, int count);
int     viewer_set_colors(Viewer* viewer, const int* ids, const float* rgb, int count);
int     viewer_set_visibility(Viewer* viewer, const int* ids, const int* visible, int count);

/* Load (or replace) the IBL environment HDR. Returns 1 on success,
 * 0 if the file can't be read or the GPU is unavailable. Safe to call
 * after viewer_create with a different path; the previous env is
 * released by gpu_load_environment. */
int     viewer_load_environment(Viewer* viewer, const char* hdr_path);
int     viewer_load_environment_intensity(Viewer* viewer, const char* hdr_path, float intensity);

void    viewer_destroy(Viewer* viewer);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_VIEWER_H */
