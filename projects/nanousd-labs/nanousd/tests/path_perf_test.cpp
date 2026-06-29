// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Microbenchmark: isolate costs in the path lookup pipeline.
//
// Build:
//   cd nanousd
//   g++ -std=c++17 -O2 -o build/path_perf_test tests/path_perf_test.cpp \
//       -I include -L build/Release -lnanousd_core -lnanousdapi \
//       -Wl,-rpath,$PWD/build/Release
//
// Run:
//   ./build/path_perf_test

#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nanousd/path.h>
#include <nanousd/types.h>
#include <nanousd/nanousdapi.h>

using namespace nanousd;
using Clock = std::chrono::high_resolution_clock;

// Generate N path strings like "/Root/linkK/geom"
static std::vector<std::string> GeneratePaths(int n) {
    std::vector<std::string> paths;
    paths.reserve(n * 3);
    for (int i = 0; i < n; i++) {
        std::string body = "/Root/link" + std::to_string(i);
        paths.push_back(body);
        paths.push_back(body + "/geom");
        paths.push_back(body + "/joint");
    }
    return paths;
}

// 1) Raw std::map<string,...>::find (what specs_ and primIndices use)
static void BenchStdMapFind(const std::vector<std::string>& paths, int iters) {
    std::map<std::string, int> m;
    for (const auto& p : paths) m[p] = 1;

    auto t0 = Clock::now();
    int found = 0;
    for (int i = 0; i < iters; i++) {
        for (const auto& p : paths) {
            found += (m.find(p) != m.end());
        }
    }
    auto t1 = Clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    int total = iters * paths.size();
    printf("  std::map<string> find:        %8.1f ns/op  (%d)\n", ns / total, found);
}

// 2) std::unordered_map<string,...>::find
static void BenchUnorderedMapFind(const std::vector<std::string>& paths, int iters) {
    std::unordered_map<std::string, int> m;
    for (const auto& p : paths) m[p] = 1;

    auto t0 = Clock::now();
    int found = 0;
    for (int i = 0; i < iters; i++) {
        for (const auto& p : paths) {
            found += (m.find(p) != m.end());
        }
    }
    auto t1 = Clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    int total = iters * paths.size();
    printf("  unordered_map<string> find:   %8.1f ns/op  (%d)\n", ns / total, found);
}

// 3) Token intern (already-interned path — measures hash + find in pool)
static void BenchTokenIntern(const std::vector<std::string>& paths, int iters) {
    // Pre-intern all
    for (const auto& p : paths) { Token t(p); }

    auto t0 = Clock::now();
    volatile const char* sink;
    for (int i = 0; i < iters; i++) {
        for (const auto& p : paths) {
            Token t(p);
            sink = t.GetText();
        }
    }
    auto t1 = Clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    int total = iters * paths.size();
    printf("  Token intern (existing):      %8.1f ns/op\n", ns / total);
}

// 4) Path::Parse only (no lookup — measures allocation + parsing cost)
static void BenchPathParse(const std::vector<std::string>& paths, int iters) {
    auto t0 = Clock::now();
    volatile bool sink;
    for (int i = 0; i < iters; i++) {
        for (const auto& p : paths) {
            Path path = Path::Parse(p);
            sink = path.IsEmpty();
        }
    }
    auto t1 = Clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    int total = iters * paths.size();
    printf("  Path::Parse:                  %8.1f ns/op\n", ns / total);
}

// 5) Path::Parse + GetParentPath (measures the ancestor chain cost)
static void BenchPathParseAndParent(const std::vector<std::string>& paths, int iters) {
    auto t0 = Clock::now();
    volatile bool sink;
    for (int i = 0; i < iters; i++) {
        for (const auto& p : paths) {
            Path path = Path::Parse(p);
            Path parent = path.GetParentPath();
            while (!parent.IsEmpty() && !parent.IsAbsoluteRoot()) {
                parent = parent.GetParentPath();
            }
            sink = parent.IsEmpty();
        }
    }
    auto t1 = Clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    int total = iters * paths.size();
    printf("  Parse + ancestor walk:        %8.1f ns/op\n", ns / total);
}

// 6) nanousd_primpath — full C API path (backend dispatch + all internal work)
static void BenchNanousdPrimpath(const std::vector<std::string>& paths, int iters) {
    NanousdStage stage = nanousd_create();
    // Define /Root
    NanousdPrim root = nanousd_define_prim(stage, "/Root", "Xform");
    nanousd_freeprim(root);
    // Define all paths
    for (const auto& p : paths) {
        NanousdPrim prim = nanousd_define_prim(stage, p.c_str(), "Xform");
        nanousd_freeprim(prim);
    }

    auto t0 = Clock::now();
    int found = 0;
    for (int i = 0; i < iters; i++) {
        for (const auto& p : paths) {
            NanousdPrim prim = nanousd_primpath(stage, p.c_str());
            if (prim) {
                found++;
                nanousd_freeprim(prim);
            }
        }
    }
    auto t1 = Clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    int total = iters * paths.size();
    printf("  nanousd_primpath (full C API):  %8.1f ns/op  (%d)\n", ns / total, found);

    nanousd_close(stage);
}

// 7) nanousd_primpath for paths that DON'T exist (the GetAvailablePrimName case)
static void BenchNanousdPrimpathMiss(const std::vector<std::string>& paths, int iters) {
    NanousdStage stage = nanousd_create();
    NanousdPrim root = nanousd_define_prim(stage, "/Root", "Xform");
    nanousd_freeprim(root);
    // Don't create the paths — we're testing miss performance

    auto t0 = Clock::now();
    int found = 0;
    for (int i = 0; i < iters; i++) {
        for (const auto& p : paths) {
            NanousdPrim prim = nanousd_primpath(stage, p.c_str());
            if (prim) {
                found++;
                nanousd_freeprim(prim);
            }
        }
    }
    auto t1 = Clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    int total = iters * paths.size();
    printf("  nanousd_primpath (miss):        %8.1f ns/op  (%d)\n", ns / total, found);

    nanousd_close(stage);
}

// 8) Simulate what GetAvailablePrimName does: parse + check + freeprim, for hits
static void BenchGetAvailableNamePattern(const std::vector<std::string>& paths, int iters) {
    NanousdStage stage = nanousd_create();
    NanousdPrim root = nanousd_define_prim(stage, "/Root", "Xform");
    nanousd_freeprim(root);
    for (const auto& p : paths) {
        NanousdPrim prim = nanousd_define_prim(stage, p.c_str(), "Xform");
        nanousd_freeprim(prim);
    }

    // This is what GetAvailablePrimName does: check if path exists, if so try _1, _2...
    // We simulate the common case: name doesn't collide (first check is a miss)
    // by checking paths that don't exist
    std::vector<std::string> miss_paths;
    for (const auto& p : paths) {
        miss_paths.push_back(p + "_NOTEXIST");
    }

    auto t0 = Clock::now();
    int found = 0;
    for (int i = 0; i < iters; i++) {
        for (const auto& mp : miss_paths) {
            NanousdPrim test = nanousd_primpath(stage, mp.c_str());
            if (test) {
                found++;
                nanousd_freeprim(test);
            }
        }
    }
    auto t1 = Clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    int total = iters * miss_paths.size();
    printf("  GetAvailableName pattern:     %8.1f ns/op  (%d)\n", ns / total, found);

    nanousd_close(stage);
}

int main() {
    for (int n : {5, 25, 50, 100}) {
        auto paths = GeneratePaths(n);
        int iters = (n <= 10) ? 10000 : (n <= 25) ? 2000 : (n <= 50) ? 500 : 200;

        printf("\n=== N=%d bodies (%zu paths, %d iters) ===\n",
               n, paths.size(), iters);

        BenchStdMapFind(paths, iters);
        BenchUnorderedMapFind(paths, iters);
        BenchTokenIntern(paths, iters);
        BenchPathParse(paths, iters);
        BenchPathParseAndParent(paths, iters);
        BenchNanousdPrimpath(paths, iters);
        BenchNanousdPrimpathMiss(paths, iters);
        BenchGetAvailableNamePattern(paths, iters);
    }
    return 0;
}
