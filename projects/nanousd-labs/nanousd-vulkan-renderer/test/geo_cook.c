// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * geo_cook — pre-warm the NUSD geometry cache for one or more USD scenes.
 *
 * Parses each USD geometry-only and writes a <usd>.nzgeo sidecar so a later
 * renderer load takes the fast cache path instead of re-parsing USD. CPU-only:
 * no GPU, no display — runnable on a build/CI box.
 *
 * Usage: geo_cook <scene.usd> [scene2.usd ...]
 *
 * See docs/planning/MESHLET_GEOMETRY_CACHE_PLAN.md.
 */

#include "geo_cache.h"

#include <stdio.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <scene.usd> [scene2.usd ...]\n", argv[0]);
        return 2;
    }

    int failures = 0;
    for (int i = 1; i < argc; i++) {
        /* The final scene uses geo_cache_cook_keep — it skips the multi-second
         * scene_free (USD-stage teardown), pure waste before the _exit below.
         * Earlier scenes free normally so a multi-scene run does not
         * accumulate leaked stages. */
        int rc = (i == argc - 1) ? geo_cache_cook_keep(argv[i])
                                 : geo_cache_cook(argv[i]);
        if (rc == 0) {
            printf("geo_cook: OK   %s.nzgeo\n", argv[i]);
        } else {
            fprintf(stderr, "geo_cook: FAIL %s\n", argv[i]);
            failures++;
        }
    }
    /* Fast exit: every cache file is already fclose'd into the page cache, so
     * _exit skips the leaked-stage + C-runtime teardown the OS does anyway. */
    fflush(stdout);
    fflush(stderr);
    _exit(failures ? 1 : 0);
}
