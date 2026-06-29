// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "path_pool.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace nanousd {

// Lock-contention counters for PathPool::Intern. Kept as a permanent
// diagnostic so we can re-confirm contention levels if the prefetch
// thread count or workload shape ever changes. Gated on
// NANOUSD_TRACE_PATHPOOL=1; printed at process exit.
//
// Hot-path overhead when disabled: one relaxed atomic load per intern
// call. The env var is read once at startup; the bool stays in the
// branch predictor the rest of the run.
namespace {
std::atomic<bool> g_trace_enabled{false};
std::atomic<long long> g_intern_calls{0};
std::atomic<long long> g_intern_wait_ns{0};   // time blocked acquiring the mutex
std::atomic<long long> g_intern_held_ns{0};   // time spent under the mutex

struct PathPoolTraceInit {
    PathPoolTraceInit() {
        const char* v = std::getenv("NANOUSD_TRACE_PATHPOOL");
        if (v && v[0] == '1')
            g_trace_enabled.store(true, std::memory_order_relaxed);
    }
    ~PathPoolTraceInit() {
        if (!g_trace_enabled.load(std::memory_order_relaxed)) return;
        long long calls = g_intern_calls.load();
        long long wait = g_intern_wait_ns.load();
        long long held = g_intern_held_ns.load();
        if (calls == 0) return;
        std::fprintf(stderr,
            "[pathpool] intern calls=%lld   wait=%.1fms (%.0fns/call)   "
            "held=%.1fms (%.0fns/call)\n",
            calls,
            wait / 1.0e6, calls ? double(wait) / calls : 0.0,
            held / 1.0e6, calls ? double(held) / calls : 0.0);
    }
};
PathPoolTraceInit g_trace_init;
} // anonymous

namespace {
constexpr std::size_t kHashSeed = 0x9e3779b97f4a7c15ULL;

inline void HashCombine(std::size_t& h, std::size_t v) {
    h ^= v + kHashSeed + (h << 6) + (h >> 2);
}

} // anonymous

std::size_t PathPool::KeyHash::operator()(const Key& k) const noexcept {
    Token::Hash th;
    std::size_t h = std::hash<const void*>{}(k.parent);
    HashCombine(h, th(k.step));
    HashCombine(h, th(k.variantValue));
    HashCombine(h, static_cast<std::size_t>(k.kind));
    return h;
}

PathPool& PathPool::Instance() {
    static PathPool inst;
    return inst;
}

PathPool::PathPool() {
    // Root lives in shard 0.
    auto* r = ArenaAlloc(shards_[0]);
    r->parent       = nullptr;
    r->kind         = PathStepKind::Child;  // root is conceptually a "Child" of nothing
    r->depth        = 0;
    r->hasVariantSelection = false;
    // step/variantValue stay default-empty Tokens.
    root_ = r;
}

PathData* PathPool::ArenaAlloc(Shard& s) {
    if (s.arenaCursor >= kArenaSlabCount) {
        s.arenaSlabs.emplace_back(new PathData[kArenaSlabCount]{});
        s.arenaCursor = 0;
    }
    return &s.arenaSlabs.back()[s.arenaCursor++];
}

void PathPool::ArenaResetShard(Shard& s) {
    const std::size_t used =
        s.arenaSlabs.empty() ? 0
                             : (s.arenaSlabs.size() - 1) * kArenaSlabCount + s.arenaCursor;
    std::size_t seen = 0;
    for (auto& slab : s.arenaSlabs) {
        for (std::size_t i = 0; i < kArenaSlabCount && seen < used; ++i, ++seen) {
            auto& pd = slab[i];
            delete pd.elementsCache.load(std::memory_order_relaxed);
            delete pd.textCache.load(std::memory_order_relaxed);
        }
    }
    s.arenaSlabs.clear();
    s.arenaCursor = kArenaSlabCount;
}

const PathData* PathPool::Root() { return root_; }

const PathData* PathPool::Intern(const PathData* parent,
                                  Token step,
                                  Token variantValue,
                                  PathStepKind kind) {
    Key k{parent, step, variantValue, kind};
    const std::size_t h = KeyHash{}(k);
    Shard& s = shards_[h & kShardMask];
    if (!g_trace_enabled.load(std::memory_order_relaxed)) {
        std::lock_guard<Spinlock> lock(s.mutex);
        // try_emplace probes once and either returns the existing entry
        // or inserts (nullptr placeholder, which we backfill below). The
        // alternative — find() then emplace() on miss — hashes + probes
        // twice. Hash + probe is ~3% of cycles each.
        auto [it, inserted] = s.entries.try_emplace(k, nullptr);
        if (!inserted) return it->second;
        auto* d = ArenaAlloc(s);
        d->parent = parent;
        d->step = step;
        d->variantValue = variantValue;
        d->kind = kind;
        d->depth = static_cast<std::uint8_t>(
            parent ? std::min<int>(parent->depth + 1, 255) : 0);
        d->hasVariantSelection =
            kind == PathStepKind::VariantSelection ||
            (parent && parent->hasVariantSelection);
        it->second = d;
        return d;
    }
    // Traced path: time wait-on-lock vs work-under-lock separately.
    using ns = std::chrono::nanoseconds;
    auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<Spinlock> lock(s.mutex);
    auto t1 = std::chrono::steady_clock::now();
    auto [it, inserted] = s.entries.try_emplace(k, nullptr);
    const PathData* result;
    if (!inserted) {
        result = it->second;
    } else {
        auto* d = ArenaAlloc(s);
        d->parent = parent;
        d->step = step;
        d->variantValue = variantValue;
        d->kind = kind;
        d->depth = static_cast<std::uint8_t>(
            parent ? std::min<int>(parent->depth + 1, 255) : 0);
        d->hasVariantSelection =
            kind == PathStepKind::VariantSelection ||
            (parent && parent->hasVariantSelection);
        it->second = d;
        result = d;
    }
    auto t2 = std::chrono::steady_clock::now();
    g_intern_calls.fetch_add(1, std::memory_order_relaxed);
    g_intern_wait_ns.fetch_add(
        std::chrono::duration_cast<ns>(t1 - t0).count(),
        std::memory_order_relaxed);
    g_intern_held_ns.fetch_add(
        std::chrono::duration_cast<ns>(t2 - t1).count(),
        std::memory_order_relaxed);
    return result;
}

std::size_t PathPool::Size() const {
    std::size_t total = 1;  // +1 for root
    for (auto& s : shards_) {
        std::lock_guard<Spinlock> lock(s.mutex);
        total += s.entries.size();
    }
    return total;
}

void PathPool::ReserveAdditional(std::size_t count) {
    // Split the reservation across shards. Path distribution across
    // shards is hash-uniform, so each shard receives ~count/kShardCount.
    const std::size_t perShard = (count + kShardCount - 1) / kShardCount;
    for (auto& s : shards_) {
        std::lock_guard<Spinlock> lock(s.mutex);
        s.entries.reserve(s.entries.size() + perShard);
    }
}

void PathPool::Clear() {
    Token rootStep         = root_->step;
    Token rootVariantValue = root_->variantValue;
    PathStepKind rootKind  = root_->kind;
    std::uint8_t rootDepth = root_->depth;
    bool rootHasVariantSelection = root_->hasVariantSelection;
    delete root_->elementsCache.load(std::memory_order_relaxed);
    delete root_->textCache.load(std::memory_order_relaxed);
    for (auto& s : shards_) {
        std::lock_guard<Spinlock> lock(s.mutex);
        ArenaResetShard(s);
        s.entries.clear();
    }
    auto* r = ArenaAlloc(shards_[0]);
    r->parent       = nullptr;
    r->step         = rootStep;
    r->variantValue = rootVariantValue;
    r->kind         = rootKind;
    r->depth        = rootDepth;
    r->hasVariantSelection = rootHasVariantSelection;
    root_ = r;
}

} // namespace nanousd
