// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_GEO_CACHE_H
#define NUSD_GEO_CACHE_H

/*
 * geo_cache.h — on-disk geometry cache for the nanousd OpenGL renderer.
 *
 * Serializes a fully-built geometry-only Scene to a sidecar <usd>.nzgeo.gl
 * file and reloads it without re-parsing USD. Gated by NUSD_GEO_CACHE=1.
 *
 * Ported from nanousd-vulkan-renderer/src/geo_cache.c (the "NZGC" format).
 * This OpenGL variant ("NZGO") carries the Scene's up_axis and DomeLight
 * metadata, and uses a distinct magic + ".nzgeo.gl" suffix so it never
 * collides with the Vulkan renderer's
 * <usd>.nzgeo. NZGO serializes meshoptimizer meshlets; the load path compacts
 * them into the mesh-shader-native layout (vertex table + 8-bit micro-indices).
 */

#include <stddef.h>
#include "scene.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when NUSD_GEO_CACHE is set to a truthy value. */
int    geo_cache_enabled(void);

/* Derive the cache file path (<usd_path>.nzgeo.gl) into out_path[out_cap].
 * Returns 0 on success, -1 if the result would not fit. */
int    geo_cache_path_for(const char* usd_path, char* out_path, size_t out_cap);

/* Serialize `scene` to the cache file for `usd_path`. Best-effort: returns 0
 * on success, -1 on any failure (the caller already has a valid Scene from
 * USD parse, so a failed write is non-fatal). Refuses scenes that carry
 * curves — the cache is geometry-only. */
int    geo_cache_write(const char* usd_path, const Scene* scene);

/* Load a Scene from the cache for `usd_path`. Returns a heap Scene owned like
 * a scene_load() result (_stage = NULL, meshes in the arena) on a valid hit,
 * or NULL on miss / stale source / version mismatch / corruption. */
Scene* geo_cache_try_load(const char* usd_path);

/* Pre-warm the cache: parse `usd_path` and write its <usd>.nzgeo.gl when the
 * cache is missing or stale. CPU-only: no GL context required. Returns 0 on
 * success (a usable cache exists afterwards), -1 on failure. */
int    geo_cache_cook(const char* usd_path);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_GEO_CACHE_H */
