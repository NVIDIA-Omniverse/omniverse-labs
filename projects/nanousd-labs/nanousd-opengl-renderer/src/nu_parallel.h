// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NU_PARALLEL_H
#define NU_PARALLEL_H

#ifdef __cplusplus
extern "C" {
#endif

/* Cross-platform parallel-for (Linux / macOS / Windows) backed by std::thread —
 * no OpenMP/libomp dependency, so it actually runs on AppleClang (which ships
 * no libomp) and on MSVC without /openmp. body(i, ctx) is invoked for every i
 * in [0, n) across up to min(hardware_concurrency, 32) worker threads using
 * dynamic (atomic work-stealing) scheduling — good for both uniform work
 * (bake rows) and variable-cost work (texture decode). body MUST be safe to
 * call concurrently for distinct i; n < 64 runs serially to avoid spawn cost.
 * All threads join before return, so anything ctx points at outlives the run. */
void nu_parallel_for(int n, void (*body)(int i, void *ctx), void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* NU_PARALLEL_H */
