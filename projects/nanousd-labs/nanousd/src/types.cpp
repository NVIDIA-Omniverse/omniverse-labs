// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/types.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace nanousd {

// Out-of-line so the function-local static lives inside nanousd.dll
// exactly once per process. An inline definition in the header would give
// each DLL/EXE translation unit its own intern pool on Windows, breaking
// Token pointer-identity (two Tokens built from the same string in
// different DLLs would have different ptr_ values, and operator== — which
// is a pointer compare — would return false).
std::array<Token::Shard, 16>& Token::GetShards() {
    static std::array<Shard, kNumShards> shards;
    return shards;
}

const std::string& Token::EmptyString() {
    static const std::string empty;
    return empty;
}

// Lock-contention counters for Token::Intern, parallel to PathPool's.
// Used to confirm whether the 16-shard design actually distributes
// contention or if specific tokens cluster on a few shards.
namespace {
std::atomic<bool> g_token_trace{false};
std::atomic<long long> g_token_calls{0};
std::atomic<long long> g_token_wait_ns{0};   // wait across all shards
std::atomic<long long> g_token_held_ns{0};   // held across all shards
std::atomic<long long> g_token_per_shard_calls[16]{};

struct TokenTraceInit {
    TokenTraceInit() {
        const char* v = std::getenv("NANOUSD_TRACE_TOKEN");
        if (v && v[0] == '1')
            g_token_trace.store(true, std::memory_order_relaxed);
    }
    ~TokenTraceInit() {
        if (!g_token_trace.load(std::memory_order_relaxed)) return;
        long long calls = g_token_calls.load();
        if (calls == 0) return;
        std::fprintf(stderr,
            "[token] intern calls=%lld   wait=%.1fms (%.0fns/call)   "
            "held=%.1fms (%.0fns/call)\n",
            calls,
            g_token_wait_ns.load() / 1.0e6,
            calls ? double(g_token_wait_ns.load()) / calls : 0.0,
            g_token_held_ns.load() / 1.0e6,
            calls ? double(g_token_held_ns.load()) / calls : 0.0);
        // Per-shard call distribution — flag clustering if some shards
        // see WAY more calls than others (sign that hashing isn't
        // distributing the workload).
        std::fprintf(stderr, "[token] per-shard calls:");
        for (size_t i = 0; i < 16; ++i) {
            std::fprintf(stderr, " %lld",
                g_token_per_shard_calls[i].load());
        }
        std::fprintf(stderr, "\n");
    }
};
TokenTraceInit g_token_trace_init;
} // anonymous

std::atomic<bool>& Token::TraceEnabledFlag() { return g_token_trace; }

const std::string& Token::InternTraced(const std::string& s) {
    using ns = std::chrono::nanoseconds;
    size_t h = std::hash<std::string>{}(s);
    size_t shardIdx = h & (kNumShards - 1);
    auto& shard = GetShards()[shardIdx];

    auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(shard.mutex);
    auto t1 = std::chrono::steady_clock::now();

    auto it = shard.pool.find(s);
    const std::string* result;
    if (it != shard.pool.end()) {
        result = &*it;
    } else {
        result = &*shard.pool.insert(s).first;
    }
    auto t2 = std::chrono::steady_clock::now();

    g_token_calls.fetch_add(1, std::memory_order_relaxed);
    g_token_per_shard_calls[shardIdx].fetch_add(1, std::memory_order_relaxed);
    g_token_wait_ns.fetch_add(
        std::chrono::duration_cast<ns>(t1 - t0).count(),
        std::memory_order_relaxed);
    g_token_held_ns.fetch_add(
        std::chrono::duration_cast<ns>(t2 - t1).count(),
        std::memory_order_relaxed);
    return *result;
}

} // namespace nanousd
