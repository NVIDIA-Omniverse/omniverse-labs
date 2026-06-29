// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Microbenchmark: isolate Populate() and DefinePrim costs
//
// Build:
//   cd nanousd
//   g++ -std=c++17 -O2 -o build/populate_perf_test tests/populate_perf_test.cpp \
//       -I include -L build -L build/Release -lnanousdapi -lnanousd_core \
//       -Wl,-rpath,$PWD/build:$PWD/build/Release

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <nanousd/nanousdapi.h>

using Clock = std::chrono::high_resolution_clock;

int main() {
    printf("%-8s  %12s  %12s  %12s\n",
           "N", "DefinePrim", "Total(ms)", "Per-prim(us)");
    printf("%-8s  %12s  %12s  %12s\n",
           "---", "---", "---", "---");

    for (int n : {1, 5, 10, 25, 50, 100, 200}) {
        NanousdStage stage = nanousd_create();

        // Define /Root
        NanousdPrim root = nanousd_define_prim(stage, "/Root", "Xform");
        nanousd_freeprim(root);

        // Time creating N bodies, each with a geom and joint child
        auto t0 = Clock::now();

        for (int i = 0; i < n; i++) {
            char bodyPath[128], geomPath[128], jointPath[128];
            snprintf(bodyPath, sizeof(bodyPath), "/Root/link%d", i);
            snprintf(geomPath, sizeof(geomPath), "/Root/link%d/geom", i);
            snprintf(jointPath, sizeof(jointPath), "/Root/link%d/joint", i);

            NanousdPrim body = nanousd_define_prim(stage, bodyPath, "Xform");
            nanousd_freeprim(body);
            NanousdPrim geom = nanousd_define_prim(stage, geomPath, "Sphere");
            nanousd_freeprim(geom);
            NanousdPrim joint = nanousd_define_prim(stage, jointPath, "PhysicsRevoluteJoint");
            nanousd_freeprim(joint);
        }

        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int prims = n * 3;
        double per_prim_us = (ms * 1000.0) / prims;

        printf("%-8d  %12d  %12.1f  %12.1f\n", n, prims, ms, per_prim_us);
        nanousd_close(stage);
    }

    // Now test: what if we could skip Populate?
    // Simulate by just doing the layer spec creation + primIndex insertion
    // without the full stage machinery
    printf("\n--- Comparison: raw layer writes (no Populate) ---\n");
    printf("%-8s  %12s  %12s  %12s\n",
           "N", "Specs", "Total(ms)", "Per-spec(us)");
    printf("%-8s  %12s  %12s  %12s\n",
           "---", "---", "---", "---");

    for (int n : {1, 5, 10, 25, 50, 100, 200}) {
        // Use a fresh stage, create all prims, measure just the C API overhead
        // without the Populate by timing individual attribute sets instead
        // Actually, let's just measure N sequential define_prim calls
        // and see the growth pattern
        NanousdStage stage = nanousd_create();
        NanousdPrim root = nanousd_define_prim(stage, "/Root", "Xform");
        nanousd_freeprim(root);

        // Pre-create all body paths, then time adding geoms only
        // This way Populate already has the bodies and we measure incremental cost
        for (int i = 0; i < n; i++) {
            char bodyPath[128];
            snprintf(bodyPath, sizeof(bodyPath), "/Root/link%d", i);
            NanousdPrim body = nanousd_define_prim(stage, bodyPath, "Xform");
            nanousd_freeprim(body);
        }

        // Now time adding 1 more prim at the end (worst case — largest stage)
        char lastPath[128];
        snprintf(lastPath, sizeof(lastPath), "/Root/link%d/extra", n - 1);

        auto t0 = Clock::now();
        int reps = 100;
        for (int r = 0; r < reps; r++) {
            // Re-define same path (DefinePrim handles existing specs)
            NanousdPrim p = nanousd_define_prim(stage, lastPath, "Xform");
            nanousd_freeprim(p);
        }
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;

        printf("%-8d  %12d  %12s  %12.1f\n", n, n + 1, "-", us);
        nanousd_close(stage);
    }

    return 0;
}
