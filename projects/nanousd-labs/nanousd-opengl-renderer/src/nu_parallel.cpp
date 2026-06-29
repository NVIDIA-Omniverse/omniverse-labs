// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* nu_parallel.cpp — std::thread-backed parallel-for, C-callable.
 *
 * Replaces OpenMP for the OpenGL renderer's CPU parallelism. OpenMP is the
 * least portable option in this workspace: AppleClang ships no libomp (so the
 * pragmas silently became no-ops → serial on macOS), and MSVC's /openmp is
 * version-limited. std::thread is C++11 stdlib — identical and dependency-free
 * on Linux, macOS, and Windows. Exposed via extern "C" so the C renderer
 * (material.c, etc.) can call it without becoming C++. */
#include "nu_parallel.h"

#include <thread>
#include <atomic>
#include <vector>

extern "C" void nu_parallel_for(int n, void (*body)(int, void *), void *ctx)
{
    if (n <= 0 || !body) return;
    /* Small n: serial — thread spawn would dominate. */
    if (n < 64) {
        for (int i = 0; i < n; ++i) body(i, ctx);
        return;
    }
    unsigned hw = std::thread::hardware_concurrency();
    unsigned nt = hw ? (hw < 32u ? hw : 32u) : 4u;
    if ((int)nt > n) nt = (unsigned)n;

    std::atomic<int> next{0};
    auto worker = [&]() {
        for (;;) {
            int i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;
            body(i, ctx);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (unsigned t = 0; t < nt; ++t) pool.emplace_back(worker);
    for (auto &th : pool) th.join();
}
