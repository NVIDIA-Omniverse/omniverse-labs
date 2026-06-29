// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// path_pool.h — internal interning machinery for `Path`.
//
// Not part of the public API. The single producer is `Path` itself
// (and `RelativePath::Anchor` via Path's mutators). Consumers see
// `Path` as an 8-byte handle; this file owns the storage behind it.

#pragma once

#include "nanousd/api.h"
#include "nanousd/external/unordered_dense.h"
#include "nanousd/relative_path.h"  // PathElement, VariantSelection
#include "nanousd/types.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#endif

namespace nanousd {

class Path;  // declared in nanousd/path.h — fwd-decl avoids the public-header dep

// What kind of step does a `PathData` node represent in the chain
// rooted at the absolute-root singleton? Each `PathData` is exactly
// one step beyond its parent.
enum class PathStepKind : std::uint8_t {
    Child,             // a prim element (e.g. /Foo)
    Property,          // a property suffix (e.g. .attr)
    VariantSelection,  // a {set=value} selection between prim elements
};

// One node in the parent-pointer tree of interned absolute paths.
// Public surface (`Path`) only ever exposes a `const PathData*`.
//
// The core path identity (parent/step/variantValue/kind/depth) is
// always populated. The `elementsCache_` and `textCache_` pointers
// are nullptr by default and only allocate on first `GetElements`/
// `GetText` query against the path. Hot paths (RemapPath in compose)
// query GetElements once per unique path; the cache amortises the
// allocation cost across subsequent queries.
struct PathData {
    // Identity (set at insert time, never mutated):
    const PathData* parent;        // nullptr only for the root singleton
    Token           step;          // child name | property name | variant set name
    Token           variantValue;  // variant value (only when kind==VariantSelection)
    PathStepKind    kind;
    std::uint8_t    depth;         // distance from root, ≤ 255 (USD scenes don't approach this)
    bool            hasVariantSelection;
    // 5 bytes padding here.

    // Lazy materialised views. Built lock-free on first query via
    // CAS; freed when the pool clears.
    mutable std::atomic<const std::vector<PathElement>*> elementsCache{nullptr};
    mutable std::atomic<const std::string*>              textCache{nullptr};
};
static_assert(sizeof(PathData) <= 64,
              "PathData should fit in a single cache line on 64-bit targets");

// Process-wide pool. Stage-singleton ownership is *not* used: paths
// from a discarded stage stay interned, same convention as Token's
// process-singleton intern pool. Tooling that loads many scenes can
// call `PathPool::Clear()` between runs.
class PathPool {
public:
    static PathPool& Instance();

    // Return the singleton root entry ("/"). Always non-null.
    const PathData* Root();

    // Look up or insert (parent, step, variantValue, kind). Identity
    // is determined by the four-tuple compared by Token pointer
    // equality and parent pointer equality. Returns a stable pointer
    // valid for the pool's lifetime.
    const PathData* Intern(const PathData* parent,
                           Token step,
                           Token variantValue,
                           PathStepKind kind);

    // Reserve buckets for an additional `count` interns (parser hint —
    // USDC knows up-front how many paths it will reconstruct, so we can
    // skip the ~log2(n) rehash cycle that hits during a bulk load).
    // The reservation is *additive*: it expands beyond the pool's
    // current size so other callers' interns aren't disturbed.
    void ReserveAdditional(std::size_t count);

    // Diagnostic: number of pool entries (including the root).
    std::size_t Size() const;

    // Free all entries except the root. Mainly for tests/benchmarks
    // that load many scenes and want to start fresh; production use
    // can ignore this.
    void Clear();

private:
    PathPool();

    struct Key {
        const PathData* parent;
        Token           step;
        Token           variantValue;
        PathStepKind    kind;

        bool operator==(const Key& o) const {
            return parent == o.parent && step == o.step
                && variantValue == o.variantValue && kind == o.kind;
        }
    };
    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept;
    };

    // 16-way sharded entry table + per-shard arena.
    //
    // Earlier benchmarks of sharding (across 4 and 16 shards) didn't
    // show a clear win — that was on top of std::unordered_map's
    // per-call rehash overhead, which dominated everything else. With
    // ankerl flat maps + arena allocator in place, the residual cost
    // in this mutex is pure wait-on-lock contention from the 3-thread
    // prefetch pool. NANOUSD_TRACE_PATHPOOL shows ~2.5 s wait + ~1.9 s
    // held across 8.3 M intern calls on the DSX sketch scene; sharding
    // by `KeyHash(k) & 15` parallelises both portions across shards
    // with the only added cost being one cheap hash compute per call.
    //
    // PathData is arena-allocated per shard so the slab cursor is
    // never contended.
    static constexpr std::size_t kShardCount = 16;
    static constexpr std::size_t kShardMask  = kShardCount - 1;
    static constexpr std::size_t kArenaSlabCount = 1024;  // ~64 KB / slab

    // Adaptive spinlock for shard critical sections. PathPool ops
    // measure ~240 ns (one ankerl lookup + maybe one arena alloc +
    // emplace), well under a futex syscall round-trip (~3 µs). Spin
    // beats sleep at this granularity for the 3-thread prefetch pool.
    class Spinlock {
        std::atomic<bool> flag_{false};
    public:
        void lock() noexcept {
            for (int spins = 0;; ++spins) {
                if (!flag_.exchange(true, std::memory_order_acquire))
                    return;
                while (flag_.load(std::memory_order_relaxed)) {
#if defined(__x86_64__) || defined(__i386__)
                    _mm_pause();
#endif
                    if (spins > 64) { std::this_thread::yield(); spins = 0; }
                    ++spins;
                }
            }
        }
        void unlock() noexcept {
            flag_.store(false, std::memory_order_release);
        }
    };

    struct Shard {
        mutable Spinlock mutex;
        ankerl::unordered_dense::map<Key, const PathData*, KeyHash> entries;
        std::vector<std::unique_ptr<PathData[]>> arenaSlabs;
        std::size_t arenaCursor = kArenaSlabCount;  // forces new slab on first alloc
    };
    std::array<Shard, kShardCount> shards_;
    const PathData* root_ = nullptr;  // owned by shards_[0]

    PathData* ArenaAlloc(Shard& s);
    void      ArenaResetShard(Shard& s);
};

// Internal const-reference accessor for the cached PathElement view.
// Same data as Path::GetElements() but returns a reference into the
// per-PathData cache — no per-call vector copy. Declared in this
// private (src/) path_pool.h header so it's only callable from inside
// libnanousd. Callers must not retain the reference across operations
// that could clear the pool (e.g. PathPool::Clear()); within a single
// composition pass this is always safe.
NANOUSD_CORE_API const std::vector<PathElement>&
GetCachedPathElements(const Path& p);

} // namespace nanousd
