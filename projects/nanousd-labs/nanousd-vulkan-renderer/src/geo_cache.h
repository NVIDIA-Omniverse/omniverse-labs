// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NUSD_GEO_CACHE_H
#define NUSD_GEO_CACHE_H

/*
 * geo_cache.h — Phase 0 on-disk geometry cache (meshoptimizer-encoded).
 *
 * Serializes a fully-built geometry-only Scene to a sidecar <usd>.nzgeo file
 * and reloads it without re-parsing USD. Gated by NUSD_GEO_CACHE=1.
 *
 * See docs/planning/MESHLET_GEOMETRY_CACHE_PLAN.md.
 */

#include <stddef.h>
#include "scene.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when NUSD_GEO_CACHE is set to a truthy value. */
int    geo_cache_enabled(void);

/* Derive the cache file path (<usd_path>.nzgeo) into out_path[out_cap].
 * Returns 0 on success, -1 if the result would not fit. */
int    geo_cache_path_for(const char* usd_path, char* out_path, size_t out_cap);

/* Serialize `scene` to the cache file for `usd_path`. Best-effort: returns 0
 * on success, -1 on any failure (the caller already has a valid Scene from
 * USD parse, so a failed write is non-fatal). Refuses scenes that carry
 * curves. Material-enabled scenes write a material section; material-enabled
 * cache reads reject geometry-only sidecars. */
int    geo_cache_write(const char* usd_path, const Scene* scene);

/* Load a Scene from the cache for `usd_path`. Returns a heap Scene owned like
 * a scene_load() result (_stage = NULL) on a valid hit, or NULL on miss /
 * stale source / version mismatch / corruption. When want_materials is true,
 * the sidecar must carry a valid material section. */
Scene* geo_cache_try_load(const char* usd_path, int want_materials);

/* Pre-warm the cache: parse `usd_path` and write its <usd>.nzgeo when the
 * cache is missing or stale. Idempotent — a fresh cache is left untouched.
 * CPU-only: no renderer or GPU is required. Returns 0 on success (a usable
 * cache exists afterwards), -1 on failure. */
int    geo_cache_cook(const char* usd_path);

/* Like geo_cache_cook but skips the parsed Scene's teardown (a multi-second
 * nanousd_close on a large scene). The Scene is leaked — intended only for the
 * final scene of a short-lived cook process that _exit()s immediately, letting
 * the OS reclaim it. */
int    geo_cache_cook_keep(const char* usd_path);

#ifdef __cplusplus
}
#endif

#endif /* NUSD_GEO_CACHE_H */
