// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/compose.h"
#include "nanousd/usd_parser.h"
#include "nanousd/usdz_package.h"
#include "path_pool.h"   // for GetCachedPathElements (internal, no-copy element access)
#include "uri.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nanousd {

namespace {
using steady_clock = std::chrono::steady_clock;

inline bool TimingEnabled() {
    static bool enabled = [] {
        const char* v = std::getenv("NANOUSD_TIMING");
        return v && v[0] == '1';
    }();
    return enabled;
}

struct ComposePhaseTimer {
    steady_clock::time_point t0 = steady_clock::now();
    void lap(const char* label) {
        if (!TimingEnabled()) { t0 = steady_clock::now(); return; }
        auto t1 = steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "    [compose] %-26s %9.1f ms\n", label, ms);
        t0 = t1;
    }
};

// Per-resolve-pass accumulator. Not thread-safe (composition runs on one thread).
struct ResolveStats {
    long long ns_entry_scan_refs = 0;
    long long ns_entry_scan_payloads = 0;
    long long ns_cached_parse = 0;
    long long ns_cached_parse_hit = 0;
    long long ns_cached_parse_miss = 0;
    long long ns_gather_layer_stack = 0;
    long long ns_index_descendants = 0;
    long long ns_resolver_calls = 0;
    long long ns_resolve_refs_total = 0;
    long long ns_resolve_payloads_total = 0;
    int queue_iterations = 0;
    int ext_refs = 0;
    int int_refs = 0;
    int ext_payloads = 0;
    int int_payloads = 0;
    int index_descendant_calls = 0;
    int prims_queued = 0;
    int cache_hits = 0;
    int cache_misses = 0;
    int gather_hits = 0;
    int gather_misses = 0;
    int leaf_layer_reuse_hits = 0;
    int leaf_layer_reuse_created = 0;
    void reset() { *this = ResolveStats{}; }
};
static ResolveStats g_stats;

struct AccumTimer {
    long long& bucket_ns;
    steady_clock::time_point t0;
    bool on;
    explicit AccumTimer(long long& b) : bucket_ns(b), on(TimingEnabled()) {
        if (on) t0 = steady_clock::now();
    }
    ~AccumTimer() {
        if (on) {
            bucket_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                steady_clock::now() - t0).count();
        }
    }
};
} // namespace

static Diagnostic MakeInvalidRetimingScaleDiagnostic(
        const Retiming& retiming,
        ArcType arcType,
        const std::string& primPath,
        const std::string& layerPath,
        const std::string& assetPath) {
    std::string message = "Invalid ";
    message += ArcTypeToString(arcType);
    message += " retiming scale ";
    message += std::to_string(retiming.scale);
    message += "; using default offset/scale";
    return {DiagSeverity::Error,
            DiagCategory::InvalidRetimingScale,
            std::move(message),
            primPath,
            layerPath,
            assetPath,
            arcType};
}

static Retiming NormalizeRetimingScale(
        const Retiming& retiming,
        ArcType arcType,
        const std::string& primPath,
        const std::string& layerPath,
        const std::string& assetPath,
        DiagnosticCollector* diagnostics) {
    if (retiming.scale > 0.0) return retiming;
    if (diagnostics) {
        diagnostics->Add(MakeInvalidRetimingScaleDiagnostic(
            retiming, arcType, primPath, layerPath, assetPath));
    }
    return Retiming{0.0, 1.0};
}

static Retiming NormalizeRetimingScale(
        const Retiming& retiming,
        ArcType arcType,
        const std::string& primPath,
        const std::string& layerPath,
        const std::string& assetPath,
        std::vector<Diagnostic>& diagnostics) {
    if (retiming.scale > 0.0) return retiming;
    diagnostics.push_back(MakeInvalidRetimingScaleDiagnostic(
        retiming, arcType, primPath, layerPath, assetPath));
    return Retiming{0.0, 1.0};
}

// ============================================================
// PathMapping
// ============================================================

// Structural path remapping: check if inputPath starts with prefixPath's
// elements, and if so, replace that prefix with basePath and append the
// remaining elements + property. Uses Token pointer comparison (O(1) per
// element) instead of string operations + Path::Parse.
static bool HasPrimPathPrefix(const Path& inputPath, const Path& prefixPath,
                              size_t* prefixLen = nullptr) {
    if (inputPath.IsEmpty() || prefixPath.IsEmpty()) return false;

    if (!inputPath.HasVariantSelections() &&
        !prefixPath.HasVariantSelections() &&
        !prefixPath.HasProperty()) {
        const PathData* inputData = inputPath.GetData_();
        const PathData* prefixData = prefixPath.GetData_();
        if (!inputData || !prefixData) return false;
        const PathData* inputPrimData =
            inputPath.HasProperty() ? inputData->parent : inputData;
        if (!inputPrimData) return false;
        if (inputPrimData->depth < prefixData->depth) return false;

        const PathData* candidate = inputPrimData;
        while (candidate && candidate->depth > prefixData->depth)
            candidate = candidate->parent;
        if (candidate != prefixData) return false;
        if (prefixLen) *prefixLen = prefixData->depth;
        return true;
    }

    const auto& inputElems = GetCachedPathElements(inputPath);
    const auto& prefixElems = GetCachedPathElements(prefixPath);
    if (inputElems.size() < prefixElems.size()) return false;
    for (size_t i = 0; i < prefixElems.size(); ++i) {
        if (inputElems[i].name != prefixElems[i].name) return false;
    }
    if (prefixLen) *prefixLen = prefixElems.size();
    return true;
}

static Path RemapPath(const Path& inputPath, const Path& prefixPath,
                      const Path& basePath) {
    if (!inputPath.HasVariantSelections() &&
        !prefixPath.HasVariantSelections() &&
        !basePath.HasVariantSelections() &&
        !prefixPath.HasProperty() &&
        !basePath.HasProperty()) {
        const PathData* inputData = inputPath.GetData_();
        const PathData* prefixData = prefixPath.GetData_();
        if (!inputData || !prefixData) return {};
        const PathData* inputPrimData =
            inputPath.HasProperty() ? inputData->parent : inputData;
        if (!inputPrimData || inputPrimData->depth < prefixData->depth)
            return {};

        std::vector<Token> suffix;
        suffix.reserve(inputPrimData->depth - prefixData->depth);
        const PathData* d = inputPrimData;
        while (d && d->depth > prefixData->depth) {
            suffix.push_back(d->step);
            d = d->parent;
        }
        if (d != prefixData) return {};

        Path result = basePath;
        for (auto it = suffix.rbegin(); it != suffix.rend(); ++it) {
            result = result.AppendChild(*it);
            if (result.IsEmpty()) return {};
        }
        if (inputPath.HasProperty()) {
            result = result.AppendProperty(inputPath.GetPropertyName());
        }
        return result;
    }

    size_t prefixLen = 0;
    if (!HasPrimPathPrefix(inputPath, prefixPath, &prefixLen)) return {};
    const auto& inputElems = GetCachedPathElements(inputPath);

    // Exact match (no suffix elements)
    if (inputElems.size() == prefixLen && !inputPath.HasProperty()) {
        return basePath;
    }

    // Build result: base + remaining child elements + optional property
    Path result = basePath;
    for (size_t i = prefixLen; i < inputElems.size(); ++i) {
        result = result.AppendChild(inputElems[i].name);
    }
    if (inputPath.HasProperty()) {
        result = result.AppendProperty(inputPath.GetPropertyName());
    }
    return result;
}

static Path OpinionSourcePath(const OpinionEntry& entry,
                              const Path& primPath) {
    return entry.sourcePath.IsEmpty()
        ? entry.pathMapping->MapToSource(primPath)
        : entry.sourcePath;
}

static bool EntryCanContributeInstancingKey(const OpinionEntry& entry,
                                            const Path& primPath) {
    return !entry.pathMapping->targetPrimPath.IsEmpty() &&
           entry.pathMapping->targetPrimPath == primPath;
}

static const PathMappingEntry* BestSourceMapping(
        const std::vector<PathMappingEntry>& mappings,
        const Path& sourcePath) {
    const PathMappingEntry* best = nullptr;
    size_t bestLen = 0;
    for (const auto& entry : mappings) {
        size_t len = 0;
        if (!HasPrimPathPrefix(sourcePath, entry.sourcePrimPath, &len))
            continue;
        if (!best || len > bestLen) {
            best = &entry;
            bestLen = len;
        }
    }
    return best;
}

static const PathMappingEntry* BestTargetMapping(
        const std::vector<PathMappingEntry>& mappings,
        const Path& targetPath) {
    const PathMappingEntry* best = nullptr;
    size_t bestLen = 0;
    for (const auto& entry : mappings) {
        if (entry.targetPrimPath.IsEmpty()) continue;
        size_t len = 0;
        if (!HasPrimPathPrefix(targetPath, entry.targetPrimPath, &len))
            continue;
        if (!best || len > bestLen) {
            best = &entry;
            bestLen = len;
        }
    }
    return best;
}

Path PathMapping::MapToSource(const Path& targetPath) const {
    if (isIdentity) return targetPath;
    if (const auto* extra = BestTargetMapping(extraMappings, targetPath)) {
        Path mapped = RemapPath(targetPath, extra->targetPrimPath,
                                extra->sourcePrimPath);
        if (!mapped.IsEmpty()) return mapped;
    }
    Path mapped = RemapPath(targetPath, targetPrimPath, sourcePrimPath);
    if (!mapped.IsEmpty()) return mapped;
    return fallbackIdentity ? targetPath : Path();
}

Path PathMapping::MapToTarget(const Path& sourcePath) const {
    if (isIdentity) return sourcePath;
    if (const auto* extra = BestSourceMapping(extraMappings, sourcePath)) {
        if (extra->targetPrimPath.IsEmpty()) return Path();
        Path mapped = RemapPath(sourcePath, extra->sourcePrimPath,
                                extra->targetPrimPath);
        if (!mapped.IsEmpty()) return mapped;
    }
    Path mapped = RemapPath(sourcePath, sourcePrimPath, targetPrimPath);
    if (!mapped.IsEmpty()) {
        if (BestTargetMapping(extraMappings, mapped)) return Path();
        return mapped;
    }
    return fallbackIdentity ? sourcePath : Path();
}

bool PathMapping::UsesExtraSourceMapping(const Path& sourcePath) const {
    return BestSourceMapping(extraMappings, sourcePath) != nullptr;
}

const PathMappingPtr& IdentityPathMappingPtr() {
    // Default-constructed PathMapping: isIdentity=true, empty source/target.
    // One allocation, shared across every identity-mapping OpinionEntry —
    // the common case for the root sublayer stack and for descendants of
    // identity arcs.
    static const PathMappingPtr kIdentity = std::make_shared<const PathMapping>();
    return kIdentity;
}

PathMappingPtr MakePathMapping(
        Path sourcePrimPath,
        Path targetPrimPath,
        bool fallbackIdentity,
        std::vector<PathMappingEntry> extraMappings) {
    auto m = std::make_shared<PathMapping>();
    const bool identity = (sourcePrimPath == targetPrimPath) &&
                          extraMappings.empty() &&
                          (fallbackIdentity ||
                           sourcePrimPath.IsAbsoluteRoot());
    m->sourcePrimPath = std::move(sourcePrimPath);
    m->targetPrimPath = std::move(targetPrimPath);
    m->isIdentity = identity;
    m->fallbackIdentity = fallbackIdentity;
    m->extraMappings = std::move(extraMappings);
    return m;
}

static constexpr size_t kInvalidLayerStackId =
    static_cast<size_t>(-1);

static size_t LayerStackIdForLayer(const CompositionGraph& graph,
                                   size_t layerIndex) {
    if (layerIndex >= graph.layerStackIds.size())
        return kInvalidLayerStackId;
    return graph.layerStackIds[layerIndex];
}

static const std::vector<Relocate>& LayerStackRelocates(
        const CompositionGraph& graph,
        size_t stackId) {
    static const std::vector<Relocate> kEmpty;
    if (stackId >= graph.layerStacks.size()) return kEmpty;
    return graph.layerStacks[stackId].relocates;
}

static bool IsRelocateSourcePath(const CompositionGraph& graph,
                                 size_t stackId,
                                 const Path& path) {
    for (const auto& r : LayerStackRelocates(graph, stackId)) {
        if (HasPrimPathPrefix(path, r.sourcePath)) return true;
    }
    return false;
}

using StrengthComponent = OpinionEntry::StrengthComponent;
using StrengthChain = std::vector<StrengthComponent>;

static const StrengthChain& EmptyStrengthChain() {
    static const StrengthChain empty;
    return empty;
}

static const StrengthChain& EntryStrengthChain(const OpinionEntry& entry) {
    return entry.strengthChain ? *entry.strengthChain : EmptyStrengthChain();
}

static StrengthChain EntryStrengthChainCopy(const OpinionEntry& entry) {
    const auto& chain = EntryStrengthChain(entry);
    return StrengthChain(chain.begin(), chain.end());
}

static std::shared_ptr<const StrengthChain> ShareStrengthChain(StrengthChain chain) {
    if (chain.empty()) return nullptr;
    struct Hash {
        size_t operator()(const StrengthChain& value) const noexcept {
            size_t h = value.size();
            for (const auto& component : value) {
                auto combine = [&](auto v) {
                    h ^= std::hash<decltype(v)>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
                };
                combine(static_cast<int>(component.arcType));
                combine(component.namespaceDepth);
                combine(component.implied);
                combine(component.siblingOrder);
            }
            return h;
        }
    };
    thread_local static std::unordered_map<
        StrengthChain,
        std::shared_ptr<const StrengthChain>,
        Hash> cache;
    auto found = cache.find(chain);
    if (found != cache.end()) return found->second;
    StrengthChain key = chain;
    auto stored = std::make_shared<const StrengthChain>(std::move(chain));
    auto [it, _] = cache.emplace(std::move(key), std::move(stored));
    return it->second;
}

static void SetStrengthChain(OpinionEntry& entry, StrengthChain chain) {
    entry.strengthChain = ShareStrengthChain(std::move(chain));
}

static uint32_t PrimPathDepth(const Path& path) {
    if (path.IsEmpty() || path.IsAbsoluteRoot()) return 0;
    return static_cast<uint32_t>(GetCachedPathElements(path).size());
}

static StrengthComponent MakeStrengthComponent(
        ArcType arcType,
        const Path& arcAuthorPath = Path(),
        bool implied = false,
        uint32_t siblingOrder = 0) {
    StrengthComponent component;
    component.arcType = arcType;
    component.namespaceDepth = PrimPathDepth(arcAuthorPath);
    component.implied = implied;
    component.siblingOrder = siblingOrder;
    return component;
}

static bool SameStrengthComponent(const StrengthComponent& a,
                                  const StrengthComponent& b) {
    return a.arcType == b.arcType &&
           a.namespaceDepth == b.namespaceDepth &&
           a.implied == b.implied &&
           a.siblingOrder == b.siblingOrder;
}

static StrengthChain StrengthPrefixForAuthoredArc(
        const OpinionEntry& entry) {
    StrengthChain result = EntryStrengthChainCopy(entry);
    if (result.empty()) result.push_back(MakeStrengthComponent(entry.arcType));
    if (!result.empty() && result.back().arcType == ArcType::Local)
        result.pop_back();
    return result;
}

static StrengthChain MakeStrengthChain(
        const StrengthChain& prefix,
        ArcType arcType,
        const Path& arcAuthorPath = Path(),
        bool implied = false,
        uint32_t siblingOrder = 0) {
    StrengthChain result = prefix;
    result.push_back(MakeStrengthComponent(
        arcType, arcAuthorPath, implied, siblingOrder));
    result.push_back(MakeStrengthComponent(ArcType::Local));
    return result;
}

static void ReplaceInnermostArc(StrengthChain& chain,
                                ArcType arcType) {
    if (chain.empty()) {
        chain.push_back(MakeStrengthComponent(arcType));
        chain.push_back(MakeStrengthComponent(ArcType::Local));
        return;
    }
    size_t idx = chain.size() - 1;
    if (chain.back().arcType == ArcType::Local) {
        if (chain.size() == 1) {
            chain.insert(chain.begin(), MakeStrengthComponent(arcType));
            return;
        }
        idx = chain.size() - 2;
    }
    chain[idx].arcType = arcType;
}

static bool OpinionEntryStronger(const OpinionEntry& a,
                                 const OpinionEntry& b) {
    auto chainSize = [](const OpinionEntry& e) {
        const auto& chain = EntryStrengthChain(e);
        return chain.empty() ? size_t{1} : chain.size();
    };
    auto chainAt = [](const OpinionEntry& e, size_t i) {
        const auto& chain = EntryStrengthChain(e);
        return chain.empty() ?
            MakeStrengthComponent(e.arcType) :
            chain[i];
    };
    const size_t asz = chainSize(a);
    const size_t bsz = chainSize(b);
    const size_t n = std::min(asz, bsz);
    for (size_t i = 0; i < n; ++i) {
        StrengthComponent ac = chainAt(a, i);
        StrengthComponent bc = chainAt(b, i);
        int as = ArcTypeStrength(ac.arcType);
        int bs = ArcTypeStrength(bc.arcType);
        if (as != bs) return as < bs;
        if (ac.namespaceDepth != bc.namespaceDepth)
            return ac.namespaceDepth > bc.namespaceDepth;
        if (ac.implied != bc.implied)
            return !ac.implied;
        if (ac.siblingOrder != bc.siblingOrder)
            return ac.siblingOrder < bc.siblingOrder;
    }
    return asz < bsz;
}

// ============================================================
// CompositionGraph
// ============================================================

Layer* CompositionGraph::FindLayer(const std::string& sourcePath) {
    for (size_t i = 0; i < layerPaths.size(); ++i) {
        if (layerPaths[i] == sourcePath) return layers[i].get();
    }
    return nullptr;
}

const Layer* CompositionGraph::FindLayer(const std::string& sourcePath) const {
    for (size_t i = 0; i < layerPaths.size(); ++i) {
        if (layerPaths[i] == sourcePath) return layers[i].get();
    }
    return nullptr;
}

const PrimIndex* CompositionGraph::GetPrimIndex(const Path& path) const {
    auto it = primIndices.find(path);
    if (it != primIndices.end()) return &it->second;
    // Lazy fallback: arc-untouched prims aren't seeded by the eager
    // build, so compute on first query and cache. Empty-result misses
    // are NOT cached — re-querying a non-existent path is rare and
    // caching every miss would grow the map unboundedly under bad
    // queries.
    PrimIndex computed = ComposePrimIndex(*this, path);
    if (computed.entries.empty()) return nullptr;
    auto [insertedIt, _] = primIndices.try_emplace(path, std::move(computed));
    return &insertedIt->second;
}

bool CompositionGraph::HasPrimIndex(const Path& path) const {
    // Cache-only check: "has someone already pushed an opinion for this
    // path into our graph's view?" — used by Stage::DefinePrim to decide
    // whether to incrementally append a new opinion entry vs. trust an
    // existing one. Doesn't trigger lazy compose. Use GetPrimIndex if
    // you want "is there ANY opinion in the layer stack for this path."
    return primIndices.count(path) > 0;
}

PrimIndex& CompositionGraph::EnsurePrimIndex(const Path& path) {
    return primIndices[path];
}

void CompositionGraph::SetPrimIndex(const Path& path, PrimIndex pi) {
    primIndices[path] = std::move(pi);
}

const PrimDefinition& CompositionGraph::GetOrBuildPrimDef(
        const std::string& /*primPath — unused; see key note below*/,
        const std::string& typeName,
        const std::vector<std::string>& apiSchemas) const {
    if (!primDefCache_) primDefCache_ = std::make_unique<PrimDefCache>();

    // Cache key = typeName + apiSchemas joined by NUL. The PrimDefinition
    // returned by BuildFullPrimDefinition is purely a function of (typeName,
    // apiSchemas); keying on primPath (as we used to) built a separate copy
    // per prim, so a scene with 45k Xform prims rebuilt Xform's primdef 45k
    // times. Keying on (type, schemas) collapses that to one build per
    // unique combination — a few dozen entries for a typical scene.
    std::string key;
    size_t keyBytes = typeName.size() + apiSchemas.size();
    for (const auto& s : apiSchemas) keyBytes += s.size();
    key.reserve(keyBytes);
    key.append(typeName);
    for (const auto& s : apiSchemas) {
        key.push_back('\0');
        key.append(s);
    }

    auto& registry = SchemaRegistry::GetInstance();
    // Retry guard: if schemas are registered while we build a PrimDefinition
    // outside the lock, discard that stale build and try again.
    for (;;) {
        uint64_t currentGen = registry.GetGeneration();

        // Fast path: generation still matches and the entry is already cached.
        // Keep cache-hit traversal read-only; taking the unique lock here showed
        // up as futex contention in multi-threaded attribute-count scans.
        {
            std::shared_lock lock(primDefCache_->mutex);
            if (primDefCache_->schemaGeneration == currentGen) {
                auto it = primDefCache_->entries.find(key);
                if (it != primDefCache_->entries.end()) return it->second;
            }
        }

        {
            std::unique_lock lock(primDefCache_->mutex);
            currentGen = registry.GetGeneration();
            if (primDefCache_->schemaGeneration != currentGen) {
                primDefCache_->entries.clear();
                primDefCache_->schemaGeneration = currentGen;
            }
            auto it = primDefCache_->entries.find(key);
            if (it != primDefCache_->entries.end()) return it->second;
        }

        // Slow path: build outside lock, then insert under exclusive lock.
        auto def = registry.BuildFullPrimDefinition(typeName, apiSchemas);
        std::unique_lock lock(primDefCache_->mutex);
        if (registry.GetGeneration() != currentGen) {
            continue;
        }
        if (primDefCache_->schemaGeneration != currentGen) {
            primDefCache_->entries.clear();
            primDefCache_->schemaGeneration = currentGen;
        }
        // Double-check: another thread may have inserted while we built.
        auto [it, inserted] = primDefCache_->entries.emplace(key, std::move(def));
        return it->second;
    }
}

void CompositionGraph::InvalidatePrimDefCache(const std::string& /*primPath*/) const {
    // With (type, schemas) keying, a schema change on one prim is captured
    // automatically: its next GetPrimDefinition lookup uses a different key
    // and builds a fresh entry, leaving other prims with the old (type,
    // schemas) combo still hitting their valid cache entry. No explicit
    // invalidation is needed.
}

void CompositionGraph::AddToChildIndex(const Path& path) {
    if (path.IsEmpty() || path.IsAbsoluteRoot()) return;
    Path parent = path.GetParentPath();
    Token name = path.GetName();
    if (name.IsEmpty()) return;
    auto& children = composedChildren_[parent];
    const PrimIndex* primIndex = nullptr;
    auto pit = primIndices.find(path);
    if (pit != primIndices.end()) primIndex = &pit->second;
    for (size_t i = 0; i < children.names.size(); ++i) {
        if (children.names[i] == name) {
            if (i >= children.paths.size()) children.paths.push_back(path);
            if (i >= children.primIndices.size())
                children.primIndices.push_back(primIndex);
            if (children.childHasChildren.size() < children.names.size())
                children.childHasChildren.resize(children.names.size(), 0);
            return;
        }
    }
    children.names.push_back(name);
    children.paths.push_back(path);
    children.primIndices.push_back(primIndex);
    children.childHasChildren.push_back(0);
    ++composedChildEntryCount_;

    if (!parent.IsAbsoluteRoot() && !parent.IsEmpty()) {
        Path grandParent = parent.GetParentPath();
        auto parentIt = composedChildren_.find(grandParent);
        if (parentIt != composedChildren_.end()) {
            auto& parentChildren = parentIt->second;
            if (parentChildren.childHasChildren.size() < parentChildren.paths.size()) {
                parentChildren.childHasChildren.resize(parentChildren.paths.size(), 0);
            }
            for (size_t i = 0; i < parentChildren.paths.size(); ++i) {
                if (parentChildren.paths[i] == parent) {
                    parentChildren.childHasChildren[i] = 1;
                    break;
                }
            }
        }
    }
}

namespace {
struct ChildIndexPair {
    Token name;
    Path path;
    const PrimIndex* primIndex;
};

void AppendChildIndexPair(PathMap<std::vector<ChildIndexPair>>& pairsByParent,
                          const Path& path,
                          const PrimIndex* primIndex = nullptr) {
    if (path.IsEmpty() || path.IsAbsoluteRoot()) return;
    Token name = path.GetName();
    if (name.IsEmpty()) return;
    Path parent = path.GetParentPath();
    pairsByParent[parent].push_back({name, path, primIndex});
}

void AppendChildIndexPairsFromOpinionSources(
        const CompositionGraph& graph,
        const Path& targetParent,
        const PrimIndex& primIndex,
        PathMap<std::vector<ChildIndexPair>>& pairsByParent) {
    for (const auto& entry : primIndex.entries) {
        Path sourceParent = OpinionSourcePath(entry, targetParent);
        if (sourceParent.IsEmpty()) continue;

        const Layer& layer = graph.GetLayer(entry.layerIndex);
        const auto& childNames = layer.GetChildNames(sourceParent);
        if (childNames.empty()) continue;

        const auto& childPaths =
            detail::GetLayerChildPaths(layer, sourceParent);
        const bool useChildPaths = childPaths.size() == childNames.size();
        const size_t stackId = LayerStackIdForLayer(graph, entry.layerIndex);
        for (size_t i = 0; i < childNames.size(); ++i) {
            const auto& childName = childNames[i];
            Path childSource = useChildPaths
                ? childPaths[i]
                : sourceParent.AppendChild(childName);
            if (childSource.IsEmpty()) continue;
            if (IsRelocateSourcePath(graph, stackId, childSource)) continue;

            Path childTarget = entry.pathMapping->MapToTarget(childSource);
            if (childTarget.IsEmpty()) continue;
            AppendChildIndexPair(pairsByParent, childTarget);
        }
    }
}

void DeduplicateChildIndexPairsPreservingOrder(
        std::vector<ChildIndexPair>& childPairs) {
    if (childPairs.size() < 2) return;

    std::vector<ChildIndexPair> ordered;
    ordered.reserve(childPairs.size());
    std::unordered_map<Token, size_t, Token::Hash> seen;
    seen.reserve(childPairs.size());

    for (const auto& pair : childPairs) {
        auto [it, inserted] = seen.emplace(pair.name, ordered.size());
        if (inserted) {
            ordered.push_back(pair);
            continue;
        }

        // Keep the first-authored position, but prefer a cached PrimIndex
        // pointer when a later duplicate came from the eager prim-index set.
        ChildIndexPair& existing = ordered[it->second];
        if (!existing.primIndex && pair.primIndex)
            existing.primIndex = pair.primIndex;
        if (existing.path.IsEmpty())
            existing.path = pair.path;
    }

    childPairs = std::move(ordered);
}
} // namespace

void CompositionGraph::BuildChildIndex() {
    composedChildren_.clear();
    composedChildEntryCount_ = 0;
    PathMap<std::vector<ChildIndexPair>> childPairsByParent;
    childPairsByParent.reserve(primIndices.size() / 4 + 1);

    // Lazy-eligible paths aren't in primIndices yet (they populate on
    // first GetPrimIndex query), so walk the sublayer-stack layers'
    // own child indices to discover them in authored order.
    for (size_t li = 0; li < arcOriginLayerCount; ++li) {
        const Layer& layer = *layers[li];
        const size_t stackId = LayerStackIdForLayer(*this, li);
        const bool checkRelocates =
            !LayerStackRelocates(*this, stackId).empty();
        detail::ForEachLayerChildIndex(layer,
            [this, stackId, checkRelocates, &childPairsByParent](
                    const Path& parent,
                    const std::vector<Token>& names,
                    const std::vector<Path>& paths) {
                const bool usePaths = paths.size() == names.size();
                auto& childPairs = childPairsByParent[parent];
                childPairs.reserve(childPairs.size() + names.size());
                for (size_t i = 0; i < names.size(); ++i) {
                    Path childPath = usePaths
                        ? paths[i]
                        : parent.AppendChild(names[i]);
                    if (checkRelocates &&
                        IsRelocateSourcePath(*this, stackId, childPath))
                        continue;
                    childPairs.push_back({names[i], childPath, nullptr});
                }
            });
    }

    // Arc-derived children need the source layer's authored sibling order too.
    // Walk each cached parent's opinion stack in strength order and map ordered
    // source children into the composed namespace. The prim-index map pass below
    // remains as a pointer attachment/fallback path, not the ordering source.
    for (const auto& [path, primIndex] : primIndices) {
        AppendChildIndexPairsFromOpinionSources(
            *this, path, primIndex, childPairsByParent);
    }

    // Arc-touched paths come from the primIndices cache: eager-built
    // entries for arc-bearing prims plus arc-derived descendants seeded
    // by IndexDescendantPrims during ResolveGraphReferences. Append these after
    // ordered layer/opinion children so dedupe keeps the authored position while
    // still attaching the cached PrimIndex pointer.
    for (const auto& [path, primIndex] : primIndices) {
        AppendChildIndexPair(childPairsByParent, path, &primIndex);
    }

    for (auto& [parent, childPairs] : childPairsByParent) {
        DeduplicateChildIndexPairsPreservingOrder(childPairs);
    }

    composedChildren_.reserve(childPairsByParent.size());
    for (auto& [parent, childPairs] : childPairsByParent) {
        composedChildEntryCount_ += childPairs.size();
        auto [it, _] = composedChildren_.try_emplace(parent);
        auto* currentChildren = &it->second.names;
        auto* currentChildPaths = &it->second.paths;
        auto* currentChildPrimIndices = &it->second.primIndices;
        auto* currentChildHasChildren = &it->second.childHasChildren;
        currentChildren->reserve(childPairs.size());
        currentChildPaths->reserve(childPairs.size());
        currentChildPrimIndices->reserve(childPairs.size());
        currentChildHasChildren->reserve(childPairs.size());
        for (const auto& pair : childPairs) {
            currentChildren->push_back(pair.name);
            currentChildPaths->push_back(pair.path);
            currentChildPrimIndices->push_back(pair.primIndex);
            currentChildHasChildren->push_back(0);
        }
    }

    // Mark childHasChildren from the set of parent keys. This avoids a
    // hash lookup for every child edge; only paths that are themselves
    // parents need to find and mark their slot under the grandparent.
    for (const auto& [parent, _] : composedChildren_) {
        if (parent.IsEmpty() || parent.IsAbsoluteRoot()) continue;
        Path grandParent = parent.GetParentPath();
        auto grandIt = composedChildren_.find(grandParent);
        if (grandIt == composedChildren_.end()) continue;
        auto& siblings = grandIt->second;
        const Token name = parent.GetName();
        for (size_t i = 0; i < siblings.names.size(); ++i) {
            if (siblings.names[i] != name) continue;
            if (i >= siblings.paths.size() || siblings.paths[i] == parent)
                siblings.childHasChildren[i] = 1;
            break;
        }
    }
}

static const std::vector<Token> s_emptyComposedChildNames;
static const std::vector<Path> s_emptyComposedChildPaths;
static const std::vector<const PrimIndex*> s_emptyComposedChildPrimIndices;
static const std::vector<std::uint8_t> s_emptyComposedChildHasChildren;

const std::vector<Token>& CompositionGraph::GetComposedChildNames(
    const Path& parentPath) const {
    auto it = composedChildren_.find(parentPath);
    return it != composedChildren_.end() ? it->second.names : s_emptyComposedChildNames;
}

const std::vector<Path>& CompositionGraph::GetComposedChildPaths(
    const Path& parentPath) const {
    auto it = composedChildren_.find(parentPath);
    return it != composedChildren_.end() ? it->second.paths : s_emptyComposedChildPaths;
}

const std::vector<const PrimIndex*>& CompositionGraph::GetComposedChildPrimIndices(
    const Path& parentPath) const {
    auto it = composedChildren_.find(parentPath);
    return it != composedChildren_.end()
        ? it->second.primIndices
        : s_emptyComposedChildPrimIndices;
}

CompositionGraph::ComposedChildView CompositionGraph::GetComposedChildren(
    const Path& parentPath) const {
    auto it = composedChildren_.find(parentPath);
    if (it == composedChildren_.end()) {
        return {&s_emptyComposedChildNames,
                &s_emptyComposedChildPaths,
                &s_emptyComposedChildPrimIndices,
                &s_emptyComposedChildHasChildren};
    }
    return {&it->second.names, &it->second.paths, &it->second.primIndices,
            &it->second.childHasChildren};
}

// ============================================================
// Internal helpers
// ============================================================

namespace {

// Normalize a resolved layer location for cache keys and cycle detection.
// Filesystem paths use weakly_canonical so aliases on disk dedupe. URI
// locations must not go through std::filesystem: paths like
// "https://example.com/a.usda" are otherwise treated as ordinary POSIX text
// and collapsed to "https:/example.com/a.usda".
std::string NormalizePath(const std::string& path) {
    if (path.empty()) return {};

    std::string packagePath;
    std::string entryPath;
    if (SplitPackageIdentifier(path, &packagePath, &entryPath)) {
        return MakePackageIdentifier(NormalizePath(packagePath), entryPath);
    }

    if (!detail::HasWindowsDrivePrefix(path)) {
        detail::UriParseResult parsed = detail::ParseUriReference(path);
        if (parsed.success && parsed.uri.scheme) {
            parsed.uri.path = detail::RemoveUriDotSegments(parsed.uri.path);
            return detail::FormatUriReference(parsed.uri);
        }
        if (!parsed.success && detail::HasUriSchemePrefix(path)) {
            return path;
        }
    }

    std::error_code ec;
    auto canonical = std::filesystem::weakly_canonical(path, ec);
    return ec ? path : canonical.string();
}

std::string LayerSourcePathForAsset(const std::string& resolvedPath) {
    if (!IsPackageIdentifier(resolvedPath) && HasUsdzExtension(resolvedPath)) {
        std::string rootEntry = GetUsdzRootLayerPath(resolvedPath);
        if (!rootEntry.empty())
            return MakePackageIdentifier(resolvedPath, rootEntry);
    }
    return resolvedPath;
}

// ============================================================
// Composition context
// ============================================================

struct ParseCache {
    std::mutex mutex;
    // Keyed by normalized layer location. Filesystem locations use
    // weakly_canonical to dedupe aliases on disk; URI locations preserve URI
    // syntax and only normalize URI path dot segments.
    std::unordered_map<std::string, std::shared_ptr<Layer>> entries;
    // Memoization of NormalizePath: raw resolved path -> cached layer. Most
    // call sites pass the exact same resolvedPath string repeatedly (same
    // asset+anchor pair), so this short-circuits weakly_canonical on 96%+
    // of CachedParseUsdFile calls, which is otherwise the hot-path cost.
    std::unordered_map<std::string, std::shared_ptr<Layer>> rawPathEntries;
};

struct GatherResult;  // forward decl — cached by resolvedPath with identity parent retiming

struct GatherCache {
    std::mutex mutex;
    // Keyed by the resolved raw layer location. Entries are gathered with
    // identity parent retiming; callers reapply their own parent retiming
    // when reading. Only populated from gathers that enter with an empty
    // cycle-detection layerStack, so the cached result is context-free.
    std::unordered_map<std::string, std::shared_ptr<GatherResult>> entries;
};

struct ResolverCache {
    std::mutex mutex;
    // Memoizes the deterministic (anchorPath, assetPath) -> resolvedPath
    // mapping the asset resolver computes. The same anchor+asset pair is
    // typically hit many times while processing refs/payloads that share
    // a layer; caching avoids redoing the resolver lookup.
    // Key is "anchor\0asset" to avoid pair/hash boilerplate.
    std::unordered_map<std::string, std::string> entries;
};

struct InstanceArcKey {
    const Layer* layer = nullptr;
    Path         sourcePath;  // pool-interned; pointer-compare equality, std::hash<Path> hashes the handle
    ArcType arcType = ArcType::None;
    Retiming retiming;
    std::vector<OpinionEntry::StrengthComponent> strengthChain;

    bool operator==(const InstanceArcKey& o) const {
        return layer == o.layer &&
               sourcePath == o.sourcePath &&
               arcType == o.arcType &&
               retiming.offset == o.retiming.offset &&
               retiming.scale == o.retiming.scale &&
               StrengthChainEqual(strengthChain, o.strengthChain);
    }

private:
    static bool StrengthChainEqual(
            const std::vector<OpinionEntry::StrengthComponent>& a,
            const std::vector<OpinionEntry::StrengthComponent>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].arcType != b[i].arcType ||
                a[i].namespaceDepth != b[i].namespaceDepth ||
                a[i].implied != b[i].implied ||
                a[i].siblingOrder != b[i].siblingOrder)
                return false;
        }
        return true;
    }
};

struct InstanceArcKeyHash {
    size_t operator()(const InstanceArcKey& k) const {
        size_t h = 0;
        HashCombine(h, k.layer);
        HashCombine(h, k.sourcePath);
        HashCombine(h, static_cast<int>(k.arcType));
        HashCombine(h, k.retiming.offset);
        HashCombine(h, k.retiming.scale);
        for (const auto& component : k.strengthChain) {
            HashCombine(h, static_cast<int>(component.arcType));
            HashCombine(h, component.namespaceDepth);
            HashCombine(h, component.implied);
            HashCombine(h, component.siblingOrder);
        }
        return h;
    }

private:
    template <typename T>
    static void HashCombine(size_t& seed, const T& value) {
        constexpr size_t kHashCombineGoldenRatio = 0x9e3779b9;
        seed ^= std::hash<T>{}(value) + kHashCombineGoldenRatio +
                (seed << 6) + (seed >> 2);
    }
};

struct InstanceArcRepresentativeCache {
    std::unordered_map<InstanceArcKey, Path, InstanceArcKeyHash> representativeByKey;
};

struct SourceDescendantNode {
    Path sourcePath;
    const Spec* spec = nullptr;
    size_t parentIndex = static_cast<size_t>(-1);
    Token childName;
};

struct SourceDescendantKey {
    const Layer* layer = nullptr;
    Path sourceRoot;
    bool operator==(const SourceDescendantKey& o) const {
        return layer == o.layer && sourceRoot == o.sourceRoot;
    }
};

struct SourceDescendantKeyHash {
    size_t operator()(const SourceDescendantKey& k) const noexcept {
        size_t h = std::hash<const void*>{}(k.layer);
        h ^= Path::Hash{}(k.sourceRoot) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct SourceDescendantCache {
    ankerl::unordered_dense::map<SourceDescendantKey,
                                 std::unique_ptr<std::vector<SourceDescendantNode>>,
                                 SourceDescendantKeyHash> entries;
    std::vector<Path> targetPathScratch;
    std::vector<std::uint8_t> activeScratch;
};

struct ReusableLeafLayerKey {
    const Layer* layer = nullptr;
    Retiming retiming;
    bool operator==(const ReusableLeafLayerKey& o) const {
        return layer == o.layer &&
               retiming.offset == o.retiming.offset &&
               retiming.scale == o.retiming.scale;
    }
};

struct ReusableLeafLayerKeyHash {
    size_t operator()(const ReusableLeafLayerKey& k) const noexcept {
        size_t h = std::hash<const void*>{}(k.layer);
        h ^= std::hash<double>{}(k.retiming.offset) + 0x9e3779b9 +
             (h << 6) + (h >> 2);
        h ^= std::hash<double>{}(k.retiming.scale) + 0x9e3779b9 +
             (h << 6) + (h >> 2);
        return h;
    }
};

struct ReusableLeafLayerUse {
    size_t layerIndex = kInvalidLayerStackId;
};

struct ReusableLeafLayerCache {
    ankerl::unordered_dense::map<ReusableLeafLayerKey,
                                 ReusableLeafLayerUse,
                                 ReusableLeafLayerKeyHash> entries;
};

struct ComposeContext {
    AssetResolver resolver;
    std::unordered_set<std::string> layerStack;  // per-branch cycle detection
    // Layer parse cache: avoids re-parsing the same file when referenced by
    // multiple prims or sublayer entries. Keyed by normalized layer location.
    std::shared_ptr<ParseCache> parseCache;
    // Sublayer-stack gather cache: avoids re-walking a referenced file's
    // sublayer tree on each ref/payload that targets it.
    std::shared_ptr<GatherCache> gatherCache;
    // Asset resolver memoization: (anchor, asset) -> resolvedPath.
    std::shared_ptr<ResolverCache> resolverCache;
    // Composition-scoped sharing for repeated instanceable external arcs.
    std::shared_ptr<InstanceArcRepresentativeCache> instanceArcCache;
    // Composition-scoped source subtree cache for repeated arc expansion.
    std::shared_ptr<SourceDescendantCache> sourceDescendantCache;
    // Composition-scoped layer-use sharing for arc leaf subtrees. This is
    // intentionally narrower than layer-stack sharing: only subtrees with no
    // composition arcs can reuse a graph layer index without affecting
    // downstream layer-stack parentage.
    std::shared_ptr<ReusableLeafLayerCache> reusableLeafLayerCache;
    DiagnosticCollector* diagnostics = nullptr;   // non-fatal diagnostic sink
};

static bool IsRootOrRootPrimPath(const Path& path) {
    return path.IsEmpty() || path.IsAbsoluteRoot() ||
           GetCachedPathElements(path).size() <= 1;
}

static std::vector<Relocate> CollectValidatedLayerRelocates(
        const CompositionGraph& graph,
        size_t beginLayerIndex,
        size_t endLayerIndex,
        DiagnosticCollector* diagnostics) {
    struct Candidate {
        Relocate relocate;
        std::string layerPath;
    };
    std::vector<Candidate> basic;

    auto emit = [&](const std::string& message, const Relocate& r,
                    const std::string& layerPath) {
        if (!diagnostics) return;
        diagnostics->Add({
            DiagSeverity::Error,
            DiagCategory::InvalidRelocate,
            message,
            r.sourcePath.GetText(),
            layerPath,
            r.targetPath ? r.targetPath->GetText() : std::string{},
            ArcType::Relocate});
    };

    endLayerIndex = std::min(endLayerIndex, graph.GetNumLayers());
    for (size_t li = beginLayerIndex; li < endLayerIndex; ++li) {
        const Layer& layer = graph.GetLayer(li);
        const Value* field =
            layer.GetLayerSpec().GetField(FieldNames::layerRelocates);
        if (!field) continue;
        const auto* relocates = field->Get<std::vector<Relocate>>();
        if (!relocates) continue;

        const std::string& layerPath = graph.layerPaths[li];
        for (const auto& r : *relocates) {
            if (IsRootOrRootPrimPath(r.sourcePath) ||
                !r.sourcePath.IsPrimPath()) {
                emit("Invalid relocate source path", r, layerPath);
                continue;
            }
            if (r.targetPath) {
                if (IsRootOrRootPrimPath(*r.targetPath) ||
                    !r.targetPath->IsPrimPath()) {
                    emit("Invalid relocate target path", r, layerPath);
                    continue;
                }
                if (r.sourcePath == *r.targetPath) {
                    emit("Relocate source and target paths must differ",
                         r, layerPath);
                    continue;
                }
                if (HasPrimPathPrefix(*r.targetPath, r.sourcePath) ||
                    HasPrimPathPrefix(r.sourcePath, *r.targetPath)) {
                    emit("Relocate source and target paths may not be ancestors of each other",
                         r, layerPath);
                    continue;
                }
            }
            basic.push_back({r, layerPath});
        }
    }

    std::vector<Candidate> ancestral;
    ancestral.reserve(basic.size());
    for (size_t i = 0; i < basic.size(); ++i) {
        const Relocate& r = basic[i].relocate;
        bool invalid = false;
        for (size_t j = 0; j < basic.size(); ++j) {
            if (i == j) continue;
            const Relocate& ancestor = basic[j].relocate;
            if (r.sourcePath == ancestor.sourcePath) continue;
            if (!HasPrimPathPrefix(r.sourcePath, ancestor.sourcePath))
                continue;
            emit("Relocate source path must use ancestral relocated path",
                 r, basic[i].layerPath);
            invalid = true;
            break;
        }
        if (!invalid) ancestral.push_back(basic[i]);
    }

    std::vector<Relocate> result;
    PathSet sources;
    PathSet targets;
    for (const auto& candidate : ancestral) {
        const Relocate& r = candidate.relocate;
        if (sources.count(r.sourcePath)) {
            emit("Duplicate relocate source path", r, candidate.layerPath);
            continue;
        }
        if (r.targetPath && targets.count(*r.targetPath)) {
            emit("Duplicate relocate target path", r, candidate.layerPath);
            continue;
        }
        sources.insert(r.sourcePath);
        if (r.targetPath) targets.insert(*r.targetPath);
        result.push_back(r);
    }
    return result;
}

static size_t StartLayerStack(CompositionGraph& graph) {
    graph.layerStacks.emplace_back();
    return graph.layerStacks.size() - 1;
}

static void FinishLayerStack(CompositionGraph& graph,
                             size_t stackId,
                             size_t beginLayerIndex,
                             size_t endLayerIndex,
                             DiagnosticCollector* diagnostics) {
    if (stackId >= graph.layerStacks.size()) return;
    auto& stack = graph.layerStacks[stackId];
    stack.beginLayerIndex = beginLayerIndex;
    stack.endLayerIndex = endLayerIndex;
    stack.relocates = CollectValidatedLayerRelocates(
        graph, beginLayerIndex, endLayerIndex, diagnostics);
}

static size_t NormalizeLayerStackId(const CompositionGraph& graph,
                                    size_t stackId) {
    if (stackId < graph.layerStacks.size()) return stackId;
    if (!graph.layerStackIds.empty()) return graph.layerStackIds.front();
    return kInvalidLayerStackId;
}

static StrengthChain AppendStrengthPrefix(
        const StrengthChain& prefix,
        ArcType arcType,
        const Path& arcAuthorPath,
        uint32_t siblingOrder = 0) {
    StrengthChain result = prefix;
    result.push_back(MakeStrengthComponent(
        arcType, arcAuthorPath, /*implied=*/false, siblingOrder));
    return result;
}

static void SetLayerStackIntroduction(
        CompositionGraph& graph,
        size_t stackId,
        size_t parentStackId,
        PathMappingPtr impliedInheritMapping,
        StrengthChain strengthPrefix) {
    if (stackId >= graph.layerStacks.size()) return;
    auto& stack = graph.layerStacks[stackId];
    stack.parentStackId = NormalizeLayerStackId(graph, parentStackId);
    stack.impliedInheritMapping = impliedInheritMapping ?
        std::move(impliedInheritMapping) :
        IdentityPathMappingPtr();
    stack.strengthPrefix = std::move(strengthPrefix);
}

static Path MapRelocateSourceThroughAncestralTargets(
        const std::vector<Relocate>& relocates,
        const Path& path) {
    Path current = path;
    PathSet visited;
    while (!current.IsEmpty() && visited.insert(current).second) {
        const Relocate* best = nullptr;
        size_t bestLen = 0;
        for (const auto& r : relocates) {
            if (!r.targetPath) continue;
            size_t len = 0;
            if (!HasPrimPathPrefix(current, *r.targetPath, &len))
                continue;
            if (!best || len > bestLen) {
                best = &r;
                bestLen = len;
            }
        }
        if (!best) break;
        Path mapped = RemapPath(current, *best->targetPath,
                                best->sourcePath);
        if (mapped.IsEmpty() || mapped == current) break;
        current = std::move(mapped);
    }
    return current;
}

static std::vector<PathMappingEntry> BuildRelocateMappingsForArc(
        const CompositionGraph& graph,
        size_t relocateStackId,
        const Path& sourcePrimPath,
        const Path& arcAuthorSourcePath,
        const PathMappingPtr& authorToTargetMapping) {
    std::vector<PathMappingEntry> entries;
    const auto& relocates = LayerStackRelocates(graph, relocateStackId);
    if (relocates.empty()) return entries;
    if (arcAuthorSourcePath.IsEmpty()) return entries;

    PathMappingPtr localArcMapping =
        MakePathMapping(sourcePrimPath, arcAuthorSourcePath);

    for (const auto& r : relocates) {
        if (!HasPrimPathPrefix(r.sourcePath, arcAuthorSourcePath)) continue;
        Path stackSourcePath =
            MapRelocateSourceThroughAncestralTargets(relocates,
                                                     r.sourcePath);
        Path sourceSide = localArcMapping->MapToSource(stackSourcePath);
        if (sourceSide.IsEmpty()) continue;

        PathMappingEntry e;
        e.sourcePrimPath = sourceSide;
        if (r.targetPath) {
            e.targetPrimPath = authorToTargetMapping ?
                authorToTargetMapping->MapToTarget(*r.targetPath) :
                *r.targetPath;
        }
        entries.push_back(std::move(e));
    }
    return entries;
}

static PathMappingPtr MakeArcPathMapping(
        const CompositionGraph& graph,
        Path sourcePrimPath,
        Path targetPrimPath,
        size_t relocateStackId,
        const Path& arcAuthorSourcePath,
        const PathMappingPtr& authorToTargetMapping,
        bool fallbackIdentity = false) {
    PathMappingPtr base = MakePathMapping(sourcePrimPath, targetPrimPath,
                                          fallbackIdentity);
    auto extra = BuildRelocateMappingsForArc(
        graph, relocateStackId, sourcePrimPath, arcAuthorSourcePath,
        authorToTargetMapping);
    if (extra.empty()) return base;
    return MakePathMapping(std::move(sourcePrimPath), std::move(targetPrimPath),
                           fallbackIdentity, std::move(extra));
}

// Helper: resolve (anchor, asset) through ctx.resolver, memoizing in
// ctx.resolverCache. Returns the resolved path (possibly empty on failure).
inline std::string ResolveAssetCached(const ComposeContext& ctx,
                                       const std::string& anchor,
                                       const std::string& asset) {
    if (!ctx.resolverCache) return ctx.resolver(anchor, asset);

    std::string key;
    key.reserve(anchor.size() + 1 + asset.size());
    key.append(anchor);
    key.push_back('\0');
    key.append(asset);

    {
        std::lock_guard<std::mutex> lock(ctx.resolverCache->mutex);
        auto it = ctx.resolverCache->entries.find(key);
        if (it != ctx.resolverCache->entries.end()) return it->second;
    }
    std::string resolved = ctx.resolver(anchor, asset);
    {
        std::lock_guard<std::mutex> lock(ctx.resolverCache->mutex);
        ctx.resolverCache->entries.try_emplace(std::move(key), resolved);
    }
    return resolved;
}

// ============================================================
// Gathered layer info (from phase 1: sublayer gathering)
// ============================================================

struct GatheredLayer {
    std::shared_ptr<Layer> layer;
    Retiming retiming;    // accumulated offset/scale
    std::string sourcePath;
};

struct GatherResult {
    bool success = false;
    std::string error;                   // non-empty only for fatal errors
    std::vector<GatheredLayer> layers;   // strength order (root first)
    std::vector<Diagnostic> diagnostics; // non-fatal issues from this subtree
};

// Forward declarations
GatherResult GatherLayerStack(std::shared_ptr<Layer> rootLayer,
                              const std::string& anchorPath,
                              const Retiming& parentRetiming,
                              ComposeContext ctx);

// Minimum number of arcs required to justify thread overhead.
constexpr int kParallelThreshold = 2;

// Result of a cached parse — carries a shared layer pointer.
struct CachedParseOutput {
    bool success = false;
    std::string error;
    std::shared_ptr<Layer> layer;
};

// Parse a USD file with caching — avoids re-parsing the same file when
// it's referenced by multiple sublayer entries or prim references.
CachedParseOutput CachedParseUsdFile(const std::string& resolvedPath,
                                      const ComposeContext& ctx) {
    if (ctx.parseCache) {
        // Fast path: the raw resolvedPath matched something we've seen
        // before — skip NormalizePath entirely.
        {
            std::lock_guard<std::mutex> lock(ctx.parseCache->mutex);
            auto rit = ctx.parseCache->rawPathEntries.find(resolvedPath);
            if (rit != ctx.parseCache->rawPathEntries.end()) {
                if (TimingEnabled()) g_stats.cache_hits++;
                return {true, {}, rit->second};
            }
        }
        // Slow path: NormalizePath collapses aliases (different anchors
        // that produce different resolved-path strings pointing at the
        // same file) to a single canonical key.
        std::string key = NormalizePath(resolvedPath);
        if (!key.empty()) {
            {
                std::lock_guard<std::mutex> lock(ctx.parseCache->mutex);
                auto it = ctx.parseCache->entries.find(key);
                if (it != ctx.parseCache->entries.end()) {
                    // Remember the raw path so next time we skip NormalizePath.
                    ctx.parseCache->rawPathEntries.emplace(resolvedPath, it->second);
                    if (TimingEnabled()) g_stats.cache_hits++;
                    return {true, {}, it->second};
                }
                if (TimingEnabled()) g_stats.cache_misses++;
            }  // release lock before parsing — lets parallel prefetchers
               // actually run in parallel rather than serializing here.
            auto result = ParseUsdFile(resolvedPath);
            if (result.success) {
                auto sp = std::make_shared<Layer>(std::move(result.layer));
                std::lock_guard<std::mutex> lock(ctx.parseCache->mutex);
                // Another thread may have raced us and inserted the same
                // key while we were parsing — try_emplace keeps the first.
                auto [it, inserted] = ctx.parseCache->entries.try_emplace(key, sp);
                ctx.parseCache->rawPathEntries.try_emplace(resolvedPath, it->second);
                return {true, {}, it->second};
            }
            return {false, result.error, nullptr};
        }
    }
    auto result = ParseUsdFile(resolvedPath);
    if (result.success) {
        return {true, {}, std::make_shared<Layer>(std::move(result.layer))};
    }
    return {false, result.error, nullptr};
}

// ============================================================
// Phase 1: GatherLayerStack (sublayers)
// ============================================================

struct SublayerGatherResult {
    bool success = false;
    std::string error;                   // non-empty only for fatal errors (cycles)
    std::vector<GatheredLayer> layers;
    std::vector<Diagnostic> diagnostics; // non-fatal issues from this branch
    int index;  // original position for ordering
};

SublayerGatherResult GatherSublayer(
        const std::string& sublayerAssetPath,
        const std::string& anchorPath,
        const Retiming& parentRetiming,
        const Retiming& sublayerRetiming,
        int index,
        ComposeContext ctx)
{
    SublayerGatherResult result;
    result.index = index;

    Retiming effectiveSublayerRetiming = NormalizeRetimingScale(
        sublayerRetiming,
        ArcType::Sublayer,
        /*primPath=*/"",
        anchorPath,
        sublayerAssetPath,
        result.diagnostics);

    std::string resolvedPath = ctx.resolver(anchorPath, sublayerAssetPath);
    if (resolvedPath.empty()) {
        // Non-fatal: skip this sublayer, collect diagnostic
        result.diagnostics.push_back({
            DiagSeverity::Error, DiagCategory::MissingSublayer,
            "Failed to resolve sublayer: " + sublayerAssetPath,
            /*primPath=*/"", /*layerPath=*/anchorPath,
            /*assetPath=*/sublayerAssetPath, ArcType::Sublayer});
        result.success = true;
        return result;
    }

    std::string normalized = NormalizePath(resolvedPath);
    if (!normalized.empty() && ctx.layerStack.count(normalized)) {
        // Fatal: cycles are non-recoverable
        result.error = "Sublayer cycle detected: " + resolvedPath;
        return result;
    }

    auto parseResult = CachedParseUsdFile(resolvedPath, ctx);
    if (!parseResult.success) {
        // Non-fatal: skip this sublayer, collect diagnostic
        result.diagnostics.push_back({
            DiagSeverity::Error, DiagCategory::SublayerParseFail,
            "Failed to parse sublayer " + sublayerAssetPath + ": " + parseResult.error,
            /*primPath=*/"", /*layerPath=*/anchorPath,
            /*assetPath=*/sublayerAssetPath, ArcType::Sublayer});
        result.success = true;
        return result;
    }

    // Accumulate retiming: composed_offset = parent_offset + child_offset * parent_scale
    // composed_scale = parent_scale * child_scale
    Retiming accumulated;
    accumulated.offset = parentRetiming.offset +
        effectiveSublayerRetiming.offset * parentRetiming.scale;
    accumulated.scale = parentRetiming.scale * effectiveSublayerRetiming.scale;

    auto gathered = GatherLayerStack(parseResult.layer,
                                     LayerSourcePathForAsset(resolvedPath),
                                     accumulated,
                                     std::move(ctx));
    if (!gathered.success) {
        // Fatal error from deeper in the stack (cycle)
        result.error = gathered.error;
        return result;
    }

    result.layers = std::move(gathered.layers);
    for (auto& diag : gathered.diagnostics) {
        result.diagnostics.push_back(std::move(diag));
    }
    result.success = true;
    return result;
}

GatherResult GatherLayerStack(std::shared_ptr<Layer> rootLayer,
                               const std::string& anchorPath,
                               const Retiming& parentRetiming,
                               ComposeContext ctx)
{
    GatherResult result;

    // Add to this branch's layer stack for cycle detection
    std::string normalizedPath = NormalizePath(anchorPath);
    if (!normalizedPath.empty()) {
        if (ctx.layerStack.count(normalizedPath)) {
            result.error = "Layer cycle detected: " + normalizedPath;
            return result;
        }
        ctx.layerStack.insert(normalizedPath);
    }

    // The root layer itself is the strongest
    GatheredLayer rootGathered;
    rootGathered.layer = rootLayer;  // share, no copy
    rootGathered.retiming = parentRetiming;
    rootGathered.sourcePath = anchorPath;
    result.layers.push_back(std::move(rootGathered));

    // Gather sublayers (weaker than root, in declaration order = strength order)
    const auto* slField = rootLayer->GetLayerSpec().GetField(FieldNames::subLayers);
    if (slField) {
        const auto* sl = slField->Get<SubLayerPaths>();
        if (sl && !sl->paths.empty()) {
            const int n = static_cast<int>(sl->paths.size());

            if (n >= kParallelThreshold) {
                std::vector<std::future<SublayerGatherResult>> futures;
                futures.reserve(n);

                for (int i = 0; i < n; ++i) {
                    futures.push_back(std::async(std::launch::async,
                        GatherSublayer,
                        sl->paths[i], anchorPath, parentRetiming, sl->offsets[i], i,
                        ctx));
                }

                std::vector<SublayerGatherResult> subResults(n);
                for (int i = 0; i < n; ++i) {
                    subResults[i] = futures[i].get();
                }

                for (int i = 0; i < n; ++i) {
                    // Merge diagnostics from this branch
                    for (auto& d : subResults[i].diagnostics) {
                        result.diagnostics.push_back(std::move(d));
                    }
                    if (!subResults[i].success) {
                        // Fatal error (cycle) — propagate
                        result.error = subResults[i].error;
                        return result;
                    }
                    for (auto& gl : subResults[i].layers) {
                        result.layers.push_back(std::move(gl));
                    }
                }
            } else {
                for (int i = 0; i < n; ++i) {
                    auto subResult = GatherSublayer(
                        sl->paths[i], anchorPath, parentRetiming, sl->offsets[i], i, ctx);
                    // Merge diagnostics from this branch
                    for (auto& d : subResult.diagnostics) {
                        result.diagnostics.push_back(std::move(d));
                    }
                    if (!subResult.success) {
                        // Fatal error (cycle) — propagate
                        result.error = subResult.error;
                        return result;
                    }
                    for (auto& gl : subResult.layers) {
                        result.layers.push_back(std::move(gl));
                    }
                }
            }
        }
    }

    result.success = true;
    return result;
}

// Cached entry point used from ResolvePrim*{References,Payloads}. Callers
// always enter with an empty layerStack (cycle-detection context) because
// reference/payload resolution starts a fresh branch, so the cached result
// is reusable regardless of the parent retiming — which we reapply here.
//
// On a miss, we gather with identity parent, store that, then rewrite the
// caller's view with their actual parent retiming. This means every cache
// hit avoids the sublayer walk + CachedParseUsdFile calls that full gather
// would otherwise do.
GatherResult GatherLayerStackCached(std::shared_ptr<Layer> rootLayer,
                                     const std::string& anchorPath,
                                     const Retiming& parentRetiming,
                                     const ComposeContext& ctx) {
    const std::string sourceAnchorPath = LayerSourcePathForAsset(anchorPath);
    // Apply parent retiming to a base GatherResult gathered with identity.
    auto retimeCopy = [&](const GatherResult& base) {
        GatherResult out;
        out.success = base.success;
        out.error = base.error;
        out.layers.reserve(base.layers.size());
        for (const auto& gl : base.layers) {
            GatheredLayer copy;
            copy.layer = gl.layer;                // immutable, share
            copy.sourcePath = gl.sourcePath;
            copy.retiming.offset =
                parentRetiming.offset + gl.retiming.offset * parentRetiming.scale;
            copy.retiming.scale = parentRetiming.scale * gl.retiming.scale;
            out.layers.push_back(std::move(copy));
        }
        out.diagnostics = base.diagnostics;
        return out;
    };

    if (ctx.gatherCache && !sourceAnchorPath.empty()) {
        {
            std::lock_guard<std::mutex> lock(ctx.gatherCache->mutex);
            auto it = ctx.gatherCache->entries.find(sourceAnchorPath);
            if (it != ctx.gatherCache->entries.end()) {
                if (TimingEnabled()) g_stats.gather_hits++;
                return retimeCopy(*it->second);
            }
        }
        if (TimingEnabled()) g_stats.gather_misses++;
        // Miss: gather under identity parent so we can cache the un-biased result.
        // GatherLayerStack's `ctx` parameter is by-value (it mutates layerStack
        // per-branch), so passing the outer ctx triggers the copy there.
        Retiming identity{0.0, 1.0};
        auto base = GatherLayerStack(std::move(rootLayer), sourceAnchorPath, identity, ctx);
        auto stored = std::make_shared<GatherResult>(std::move(base));
        {
            std::lock_guard<std::mutex> lock(ctx.gatherCache->mutex);
            // try_emplace: another thread may have raced us; keep whichever landed first
            ctx.gatherCache->entries.try_emplace(sourceAnchorPath, stored);
        }
        return retimeCopy(*stored);
    }

    // No cache available — fall through to uncached gather.
    return GatherLayerStack(std::move(rootLayer), sourceAnchorPath, parentRetiming, ctx);
}

// ============================================================
// Phase 2: Build prim indices from gathered layers
// ============================================================

// True iff the spec carries a `references` or `payload` listOp field.
// Used by BuildPrimIndices/IndexDescendantPrims to set hasArcOpinions so
// the resolve loop can skip prims that have no arcs to process.
//
// Important: GetField may return non-null for absent fields when deferred
// decode has cached a "checked but absent" sentinel (an empty Value).
// Treat that as not-authored. Without this, every prim looks like it has
// arc opinions and we queue 250k useless work items into ResolveGraphReferences.
inline bool SpecHasArcOpinion(const Spec& spec) {
    return spec.HasArcOpinion();
}

// Composition arcs (references, payloads, inherits, specializes) can be
// authored on both Prim and Variant specs per spec §11.2 — a variant body
// carrying `references = @...@` is valid and must have those references
// resolved when the variant is selected. This predicate captures that
// superset for the resolve-side scans.
inline bool SpecCanCarryArcs(const Spec& spec) {
    return spec.GetType() == SpecType::Prim ||
           spec.GetType() == SpecType::Variant;
}

// ============================================================
// Build prim indices with retiming info
// ============================================================
//
// Phase 4 made this two-pass and sparse: instead of seeding a PrimIndex
// for every Prim spec across every layer (the old O(prim-spec-count)
// per-spec map insert + vector grow that dominated compose time on
// flat scenes), we identify which prim paths actually need eager
// scaffolding — namely, paths where some opinion authors a composition
// arc — and seed entries only for those. Lazy-eligible paths populate
// the primIndices cache on first GetPrimIndex query via
// ComposePrimIndex.
//
// Pass 1 walks the sublayer-stack layers and collects the set of
// arc-bearing prim paths. Pass 2 walks the same layers again (in
// strength order so OpinionEntry order matches LIVRPS) and writes
// entries only for those paths — but writes ALL of their layer
// opinions, including non-arc layers that override the same path,
// because ResolveGraphReferences and the LIVRPS combine downstream
// expect a complete entry list per arc-touched prim.
//
// The two passes both call SpecHasArcOpinion on every Prim spec, so
// the field-decode cost is unchanged from the single-pass version. The
// win is the per-spec map insert + vector grow that's now skipped for
// every non-arc-bearing path.

void BuildPrimIndicesWithRetiming(CompositionGraph& graph) {
    // Single-layer stacks do not need the two-pass "which weaker layers also
    // contribute to this arc-bearing path" dance: there are no sibling layers.
    // Seed only prims that actually author arcs and let all other prim indices
    // populate lazily on demand.
    if (graph.GetNumLayers() == 1) {
        const size_t layerIdx = 0;
        const Layer& layer = graph.GetLayer(layerIdx);
        for (const Path& specPath : detail::GetLayerArcOpinionPrimPaths(layer)) {
            if (IsRelocateSourcePath(
                    graph, LayerStackIdForLayer(graph, layerIdx), specPath))
                continue;
            const Spec* spec = layer.GetPrimSpec(specPath);
            if (!spec || spec->GetType() != SpecType::Prim) continue;

            OpinionEntry entry;
            entry.layerIndex = layerIdx;
            entry.sourcePath = specPath;
            entry.retiming = graph.layerRetimings[layerIdx];
            auto& pi = graph.primIndices[specPath];
            if (pi.entries.empty()) pi.entries.reserve(2);
            pi.entries.push_back(std::move(entry));
            pi.hasArcOpinions = true;
        }
        return;
    }

    // Pass 1: collect paths whose spec carries an arc opinion in any
    // sublayer-stack layer. SpecHasArcOpinion forces the spec's
    // fieldset to fully decode; we'd pay that cost in resolve anyway.
    PathSet arcBearingPaths;
    for (size_t layerIdx = 0; layerIdx < graph.GetNumLayers(); ++layerIdx) {
        const Layer& layer = graph.GetLayer(layerIdx);
        for (const Path& specPath : detail::GetLayerArcOpinionPrimPaths(layer)) {
            if (IsRelocateSourcePath(
                    graph, LayerStackIdForLayer(graph, layerIdx), specPath))
                continue;
            arcBearingPaths.insert(specPath);
        }
    }

    if (arcBearingPaths.empty()) return;  // nothing for resolve to do

    // Pass 2: walk layers in strength order; for each Prim spec at an
    // arc-bearing path, seed an OpinionEntry. Non-arc-bearing paths
    // are skipped entirely — they'll lazy-populate via GetPrimIndex.
    for (size_t layerIdx = 0; layerIdx < graph.GetNumLayers(); ++layerIdx) {
        const Layer& layer = graph.GetLayer(layerIdx);
        layer.ForEachSpec([&](const Path& specPath, const Spec& spec) {
            if (spec.GetType() != SpecType::Prim) return;
            if (IsRelocateSourcePath(
                    graph, LayerStackIdForLayer(graph, layerIdx), specPath))
                return;
            if (!arcBearingPaths.count(specPath)) return;

            OpinionEntry entry;
            entry.layerIndex = layerIdx;
            // pathMapping defaults to the shared identity singleton.
            entry.sourcePath = specPath;
            entry.retiming = graph.layerRetimings[layerIdx];
            auto& pi = graph.primIndices[specPath];
            if (pi.entries.empty()) pi.entries.reserve(2);
            pi.entries.push_back(std::move(entry));
            if (SpecHasArcOpinion(spec)) pi.hasArcOpinions = true;
        });
    }
}

// ============================================================
// Phase 3: Resolve references
// ============================================================

// Tracks composition arcs that have already been expanded so we don't
// need to mutate layers to strip consumed fields.
struct ConsumedArcs {
    struct Key {
        size_t layerIndex;
        Path   primPath;   // pool-interned handle; pointer-compare equality
        bool operator==(const Key& o) const {
            return layerIndex == o.layerIndex && primPath == o.primPath;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            size_t h = std::hash<size_t>{}(k.layerIndex);
            h ^= Path::Hash{}(k.primPath) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    ankerl::unordered_dense::set<Key, KeyHash> references;
    ankerl::unordered_dense::set<Key, KeyHash> payloads;
};

// ============================================================
// Arc collection
// ============================================================
//
// `references` and `payload` are both authored as `ListOp<Reference>` —
// per LIVRPS the resolve loop has to walk the prim's opinions in strength
// order, decode the listOp from each, and combine them into a single
// authored sequence to expand. Pre-hoist this so phase 2's per-prim
// ComposePrimIndex can call the same code path that resolve does today,
// rather than duplicating the scan a third time.
//
// Variant selections (the `variantSelection` Dictionary) live in
// ResolveVariantSelections — different shape, intentionally not bundled.

struct CollectedListOpArcs {
    bool found = false;                       // true iff at least one opinion authored an arc
    std::optional<ListOp<Reference>> combined; // LIVRPS-combined items
    std::string anchorLayer;                  // layer path of the FIRST opinion that authored an arc
                                              // (used as the asset-resolution base — see Spec §10.1.1)
    size_t relocateStackId = kInvalidLayerStackId;
    Path arcAuthorSourcePath;
    PathMappingPtr arcAuthorMapping = IdentityPathMappingPtr();
    StrengthChain arcStrengthPrefix;
};

inline CollectedListOpArcs CollectListOpArcs(
        const CompositionGraph& graph,
        const PrimIndex& idx,
        const Path& primPath,
        size_t startIdx,
        const Token& fieldName,
        const ankerl::unordered_dense::set<ConsumedArcs::Key, ConsumedArcs::KeyHash>& alreadyConsumed,
        const std::string& fallbackAnchor) {
    CollectedListOpArcs out;
    out.anchorLayer = fallbackAnchor;
    if (startIdx >= idx.entries.size()) return out;

    for (size_t ei = startIdx; ei < idx.entries.size(); ++ei) {
        const auto& entry = idx.entries[ei];
        const Layer& layer = graph.GetLayer(entry.layerIndex);
        Path sourcePath = OpinionSourcePath(entry, primPath);
        if (sourcePath.IsEmpty()) continue;

        // Skip arcs already consumed in a previous pass. Belt-and-suspenders
        // with the startIdx skip: startIdx avoids re-scanning old entries,
        // and ConsumedArcs catches the (rare) case where the same (layer,
        // source) pair appears in both an old and a new entry.
        if (alreadyConsumed.count({entry.layerIndex, sourcePath})) continue;

        const auto* spec = layer.GetSpec(sourcePath);
        if (!spec || !SpecCanCarryArcs(*spec)) continue;

        auto* field = spec->GetField(fieldName);
        if (!field) continue;
        auto* listOp = field->Get<ListOp<Reference>>();
        if (!listOp) continue;

        if (!out.found) {
            out.anchorLayer = graph.layerPaths[entry.layerIndex];
            out.relocateStackId =
                LayerStackIdForLayer(graph, entry.layerIndex);
            out.arcAuthorSourcePath = sourcePath;
            out.arcAuthorMapping = entry.pathMapping;
            out.arcStrengthPrefix = StrengthPrefixForAuthoredArc(entry);
        }
        out.found = true;
        if (!out.combined) {
            out.combined = *listOp;
        } else {
            out.combined = out.combined->Combine(*listOp);
        }
    }
    return out;
}

// `inheritPaths` and `specializes` are authored as `ListOp<Path>` per
// Spec §10.5.1 / §10.5.5 — the targets are absolute prim paths in the
// same layer stack, no asset paths involved. Sibling to CollectListOpArcs
// for the path-typed flavour. Same scan / consumed-skip / strength-order
// combine logic, just a different value type.
struct CollectedListOpPathArcs {
    bool found = false;
    std::optional<ListOp<Path>> combined;
    size_t relocateStackId = kInvalidLayerStackId;
    Path arcAuthorSourcePath;
    PathMappingPtr arcAuthorMapping = IdentityPathMappingPtr();
    StrengthChain arcStrengthPrefix;
};

inline std::vector<CollectedListOpPathArcs> CollectListOpPathArcGroups(
        const CompositionGraph& graph,
        const PrimIndex& idx,
        const Path& primPath,
        size_t startIdx,
        const Token& fieldName,
        const ankerl::unordered_dense::set<ConsumedArcs::Key, ConsumedArcs::KeyHash>& alreadyConsumed) {
    std::vector<CollectedListOpPathArcs> groups;
    if (startIdx >= idx.entries.size()) return groups;

    for (size_t ei = startIdx; ei < idx.entries.size(); ++ei) {
        const auto& entry = idx.entries[ei];
        const Layer& layer = graph.GetLayer(entry.layerIndex);
        Path sourcePath = OpinionSourcePath(entry, primPath);
        if (sourcePath.IsEmpty()) continue;
        if (alreadyConsumed.count({entry.layerIndex, sourcePath})) continue;

        const auto* spec = layer.GetSpec(sourcePath);
        if (!spec || !SpecCanCarryArcs(*spec)) continue;

        auto* field = spec->GetField(fieldName);
        if (!field) continue;
        auto* listOp = field->Get<ListOp<Path>>();
        if (!listOp) continue;

        size_t stackId = LayerStackIdForLayer(graph, entry.layerIndex);
        auto it = std::find_if(groups.begin(), groups.end(),
            [&](const CollectedListOpPathArcs& group) {
                return group.relocateStackId == stackId &&
                       group.arcAuthorSourcePath == sourcePath;
            });
        if (it == groups.end()) {
            CollectedListOpPathArcs group;
            group.found = true;
            group.relocateStackId = stackId;
            group.arcAuthorSourcePath = sourcePath;
            group.arcAuthorMapping = entry.pathMapping;
            group.arcStrengthPrefix = StrengthPrefixForAuthoredArc(entry);
            group.combined = *listOp;
            groups.push_back(std::move(group));
        } else {
            if (!it->combined) {
                it->combined = *listOp;
            } else {
                it->combined = it->combined->Combine(*listOp);
            }
        }
    }
    return groups;
}

// Walk the primChildren_ index recursively from sourcePath, creating opinion
// entries that map each descendant prim into the target namespace.
// Uses Layer child indices for O(K) descent (K = descendants under source)
// instead of O(N) full-spec scan.
// If newPrims is non-null, target paths for which a primIndex did not already
// exist are appended — letting the caller queue only freshly-introduced prims.
// sourceSpec: the already-fetched Spec at sourcePath in layer, or nullptr.
// Passing it in lets the parent recursion reuse the spec lookup it just did,
// and lets us set hasArcOpinions without a second GetSpec call per insertion.
// Each entry points the queue worker at the (primPath) to resolve and the
// first opinion-entry index in that prim's PrimIndex that still needs
// scanning. Newly-created prim indices are pushed with startIdx=0 (scan
// all entries); prim indices that gained an entry via an ancestor's arc
// are pushed with startIdx=<count-before-push> so the worker only scans
// the newly-attached entry, not the entries already processed in prior
// visits.
struct QueuedArcWork {
    Path primPath;
    size_t startIdx;
};

bool IndexSinglePrimFromArc(const Layer& layer, const Path& sourcePath,
                            const Path* knownMappedTarget,
                            const OpinionEntry& templateEntry,
                            const Spec* sourceSpec,
                            CompositionGraph& graph,
                            std::vector<QueuedArcWork>* newPrims,
                            bool queueArcOpinions,
                            bool seedLocalOpinions) {
    if (IsRelocateSourcePath(
            graph,
            LayerStackIdForLayer(graph, templateEntry.layerIndex),
            sourcePath))
        return false;

    Path mappedTarget = knownMappedTarget
        ? *knownMappedTarget
        : templateEntry.pathMapping->MapToTarget(sourcePath);
    if (mappedTarget.IsEmpty()) return false;

    OpinionEntry entryForPath = templateEntry;
    if (templateEntry.pathMapping->UsesExtraSourceMapping(sourcePath)) {
        entryForPath.arcType = ArcType::Relocate;
        StrengthChain relocatedChain = EntryStrengthChainCopy(entryForPath);
        ReplaceInnermostArc(relocatedChain, ArcType::Relocate);
        SetStrengthChain(entryForPath, std::move(relocatedChain));
    }

    auto [it, inserted] = graph.primIndices.try_emplace(mappedTarget);
    // Phase 4 lazy seed: the eager BuildPrimIndices skips paths that
    // aren't arc-bearing, so a path can become arc-touched here for the
    // first time. Its sublayer-stack Local opinions (overrides at this
    // path) MUST land in the cache *before* the arc-derived templateEntry
    // — composition expects strongest-first ordering and a Local `over`
    // outranks the inbound payload/reference. Without this seed, e.g.
    // /Set3/Prop/PropScope's local `over` value would be missed and the
    // payload value would leak through.
    if (inserted && seedLocalOpinions) {
        PrimIndex& seeded = it->second;
        for (size_t li = 0; li < graph.arcOriginLayerCount; ++li) {
            const Layer& sub = graph.GetLayer(li);
            const Spec* sublocal = sub.GetSpec(mappedTarget);
            if (!sublocal || sublocal->GetType() != SpecType::Prim) continue;
            OpinionEntry local;
            local.layerIndex = li;
            local.sourcePath = mappedTarget;
            local.retiming = graph.layerRetimings[li];
            local.arcType = ArcType::Local;
            seeded.entries.push_back(std::move(local));
            if (SpecHasArcOpinion(*sublocal)) seeded.hasArcOpinions = true;
        }
    }
    // startIdx points AT the templateEntry we're about to append, so
    // the resolve worker re-scans only the newly-attached arc entry on
    // subsequent visits (seeded Local entries don't carry refs/payloads
    // by construction — those paths would have been in arcBearingPaths
    // and seeded by BuildPrimIndicesWithRetiming instead).
    const size_t startIdx = it->second.entries.size();
    entryForPath.sourcePath = sourcePath;
    if (EntryCanContributeInstancingKey(entryForPath, mappedTarget))
        it->second.hasInstancingKeyEntries = true;
    it->second.entries.push_back(std::move(entryForPath));

    // Propagate hasArcOpinions from the source spec, which the caller already
    // fetched — avoids the redundant GetSpec(sourcePath) we'd otherwise need.
    const bool specHasArc = sourceSpec && SpecHasArcOpinion(*sourceSpec);
    if (!it->second.hasArcOpinions && specHasArc)
        it->second.hasArcOpinions = true;

    // Queue only newly-attached opinions that can introduce more arcs. The
    // source-subtree walk still visits every descendant, so a child source spec
    // with refs/payloads/inherits/specializes/variants is queued when that
    // child is indexed. Queueing arc-free descendants just makes
    // ResolveGraphReferences hash-look them up later and skip them.
    //
    // Chained references (root -> A -> B) are still covered: when A's source
    // spec has an arc opinion, specHasArc is true and the new opinion is
    // queued with startIdx pointing at that appended entry.
    if (newPrims && queueArcOpinions && specHasArc)
        newPrims->push_back({mappedTarget, startIdx});

    return true;
}

bool OriginStackMayHaveLocalDescendants(const CompositionGraph& graph,
                                        const Path& targetRoot) {
    if (targetRoot.IsEmpty() || targetRoot.IsAbsoluteRoot())
        return true;
    for (size_t li = 0; li < graph.arcOriginLayerCount; ++li) {
        if (!LayerStackRelocates(
                graph, LayerStackIdForLayer(graph, li)).empty()) {
            return true;
        }
        const Layer& layer = graph.GetLayer(li);
        if (!layer.GetChildNames(targetRoot).empty())
            return true;
    }
    return false;
}

const std::vector<SourceDescendantNode>& GetSourceDescendants(
        const Layer& layer,
        const Path& sourceRoot,
        const Spec* sourceSpec,
        SourceDescendantCache* cache) {
    static const std::vector<SourceDescendantNode> kEmpty;
    if (!cache) return kEmpty;

    SourceDescendantKey key{&layer, sourceRoot};
    auto it = cache->entries.find(key);
    if (it != cache->entries.end()) return *it->second;

    auto nodes = std::make_unique<std::vector<SourceDescendantNode>>();
    nodes->push_back({sourceRoot, sourceSpec, static_cast<size_t>(-1), Token()});

    for (size_t i = 0; i < nodes->size(); ++i) {
        const Path& parentSource = (*nodes)[i].sourcePath;
        const auto& childNames = layer.GetChildNames(parentSource);
        const auto& childPaths = detail::GetLayerChildPaths(layer, parentSource);
        const bool useChildPaths = childPaths.size() == childNames.size();
        for (size_t childI = 0; childI < childNames.size(); ++childI) {
            const auto& childName = childNames[childI];
            Path childSource = useChildPaths
                ? childPaths[childI]
                : parentSource.AppendChild(childName);
            const auto* childSpec = layer.GetSpec(childSource);
            if (!childSpec || childSpec->GetType() != SpecType::Prim) continue;
            nodes->push_back({childSource, childSpec, i, childName});
        }
    }

    auto [inserted, _] = cache->entries.emplace(std::move(key), std::move(nodes));
    return *inserted->second;
}

bool LayerHasAuthoredRelocates(const Layer& layer) {
    const Value* field =
        layer.GetLayerSpec().GetField(FieldNames::layerRelocates);
    if (!field) return false;
    const auto* relocates = field->Get<std::vector<Relocate>>();
    return relocates && !relocates->empty();
}

bool SourceSubtreeHasArcOpinions(const Layer& layer,
                                 const Path& sourceRoot,
                                 const Spec* sourceSpec,
                                 SourceDescendantCache* sourceCache) {
    if (!sourceCache) return true;  // cannot prove this is a leaf subtree
    const auto& nodes =
        GetSourceDescendants(layer, sourceRoot, sourceSpec, sourceCache);
    if (nodes.empty()) return true;
    for (const auto& node : nodes) {
        if (node.spec && SpecHasArcOpinion(*node.spec)) return true;
    }
    return false;
}

std::optional<size_t> TryGetReusableLeafLayerIndex(
        CompositionGraph& graph,
        const GatherResult& gathered,
        const Layer& layer,
        const Path& sourceRoot,
        const Spec* sourceSpec,
        const ComposeContext& ctx) {
    if (!ctx.reusableLeafLayerCache) return std::nullopt;
    if (gathered.layers.size() != 1) return std::nullopt;

    const GatheredLayer& gl = gathered.layers.front();
    if (!gl.layer || gl.layer.get() != &layer) return std::nullopt;
    if (LayerHasAuthoredRelocates(layer)) return std::nullopt;
    if (SourceSubtreeHasArcOpinions(
            layer, sourceRoot, sourceSpec, ctx.sourceDescendantCache.get())) {
        return std::nullopt;
    }

    ReusableLeafLayerKey key{gl.layer.get(), gl.retiming};
    auto& cache = ctx.reusableLeafLayerCache->entries;
    auto found = cache.find(key);
    if (found != cache.end()) {
        if (TimingEnabled()) g_stats.leaf_layer_reuse_hits++;
        return found->second.layerIndex;
    }

    size_t stackId = StartLayerStack(graph);
    size_t begin = graph.GetNumLayers();
    size_t newIdx = graph.GetNumLayers();
    graph.layers.push_back(gl.layer);
    graph.layerPaths.push_back(gl.sourcePath);
    graph.layerRetimings.push_back(gl.retiming);
    graph.layerStackIds.push_back(stackId);
    FinishLayerStack(graph, stackId, begin, graph.GetNumLayers(),
                     ctx.diagnostics);
    cache.emplace(std::move(key), ReusableLeafLayerUse{newIdx});
    if (TimingEnabled()) g_stats.leaf_layer_reuse_created++;
    return newIdx;
}

constexpr size_t kMinCachedSourceDescendantNodes = 64;

void IndexDescendantPrimsImpl(const Layer& layer, const Path& sourcePath,
                              const Path& targetPath, const OpinionEntry& templateEntry,
                              const Spec* sourceSpec,
                              CompositionGraph& graph,
                              std::vector<QueuedArcWork>* newPrims,
                              bool seedLocalOpinions,
                              bool seedDescendantLocalOpinions) {
    const bool simpleSubtreeMapping =
        templateEntry.pathMapping &&
        templateEntry.pathMapping->extraMappings.empty() &&
        !templateEntry.pathMapping->fallbackIdentity;
    const Path* knownTarget = simpleSubtreeMapping ? &targetPath : nullptr;
    if (!IndexSinglePrimFromArc(layer, sourcePath, knownTarget, templateEntry,
                                sourceSpec, graph, newPrims, true,
                                seedLocalOpinions))
        return;

    const auto& childNames = layer.GetChildNames(sourcePath);
    const auto& childPaths = detail::GetLayerChildPaths(layer, sourcePath);
    const bool useChildPaths = childPaths.size() == childNames.size();
    for (size_t i = 0; i < childNames.size(); ++i) {
        const auto& childName = childNames[i];
        Path childSource = useChildPaths
            ? childPaths[i]
            : sourcePath.AppendChild(childName);
        Path childTarget = simpleSubtreeMapping
            ? targetPath.AppendChild(childName)
            : templateEntry.pathMapping->MapToTarget(childSource);
        if (childTarget.IsEmpty()) continue;

        const auto* childSpec = layer.GetSpec(childSource);
        if (!childSpec || childSpec->GetType() != SpecType::Prim) continue;

        IndexDescendantPrimsImpl(layer, childSource, childTarget,
                                 templateEntry, childSpec, graph, newPrims,
                                 seedDescendantLocalOpinions,
                                 seedDescendantLocalOpinions);
    }
}

void IndexDescendantPrims(const Layer& layer, const Path& sourcePath,
                          const Path& targetPath, const OpinionEntry& templateEntry,
                          CompositionGraph& graph,
                          SourceDescendantCache* sourceCache,
                          std::vector<QueuedArcWork>* newPrims = nullptr) {
    AccumTimer _t(g_stats.ns_index_descendants);
    g_stats.index_descendant_calls++;
    // One GetSpec at the top level; recursion reuses pre-fetched child specs.
    const Spec* sourceSpec = layer.GetSpec(sourcePath);
    if (sourceCache) {
        const auto& nodes =
            GetSourceDescendants(layer, sourcePath, sourceSpec, sourceCache);
        if (nodes.size() >= kMinCachedSourceDescendantNodes) {
            const bool simpleSubtreeMapping =
                templateEntry.pathMapping &&
                templateEntry.pathMapping->extraMappings.empty() &&
                !templateEntry.pathMapping->fallbackIdentity;
            const bool seedDescendantLocalOpinions =
                !simpleSubtreeMapping ||
                OriginStackMayHaveLocalDescendants(graph, targetPath);
            auto& targetPaths = sourceCache->targetPathScratch;
            auto& active = sourceCache->activeScratch;
            targetPaths.clear();
            targetPaths.resize(nodes.size());
            active.assign(nodes.size(), 0);

            for (size_t i = 0; i < nodes.size(); ++i) {
                if (i == 0) {
                    targetPaths[i] = targetPath;
                    active[i] = !targetPaths[i].IsEmpty();
                } else {
                    const size_t parent = nodes[i].parentIndex;
                    if (parent >= nodes.size() || !active[parent]) continue;
                    targetPaths[i] = simpleSubtreeMapping
                        ? targetPaths[parent].AppendChild(nodes[i].childName)
                        : templateEntry.pathMapping->MapToTarget(nodes[i].sourcePath);
                    active[i] = !targetPaths[i].IsEmpty();
                }
                if (!active[i]) continue;
                const bool seedLocalOpinions =
                    i == 0 || seedDescendantLocalOpinions;
                if (!IndexSinglePrimFromArc(layer, nodes[i].sourcePath,
                                            &targetPaths[i], templateEntry,
                                            nodes[i].spec, graph, newPrims, true,
                                            seedLocalOpinions)) {
                    active[i] = 0;
                }
            }
            return;
        }
    }
    const bool simpleSubtreeMapping =
        templateEntry.pathMapping &&
        templateEntry.pathMapping->extraMappings.empty() &&
        !templateEntry.pathMapping->fallbackIdentity;
    const bool seedDescendantLocalOpinions =
        !simpleSubtreeMapping ||
        OriginStackMayHaveLocalDescendants(graph, targetPath);
    IndexDescendantPrimsImpl(layer, sourcePath, targetPath,
                             templateEntry, sourceSpec, graph, newPrims,
                             /*seedLocalOpinions=*/true,
                             seedDescendantLocalOpinions);
}

std::optional<bool> ReadInstanceableOpinion(const Spec* spec) {
    if (!spec) return std::nullopt;
    const Value* v = spec->GetField(FieldNames::instanceable);
    if (!v) return std::nullopt;
    if (const auto* b = v->Get<bool>()) return *b;
    if (const auto* i = v->Get<Int>()) return *i != 0;
    return std::nullopt;
}

bool IsAuthoredField(const Value* value) {
    return value && !value->IsEmpty();
}

bool SpecHasVariantMetadata(const Spec* spec) {
    if (!spec) return false;
    return IsAuthoredField(spec->GetField(FieldNames::variantSelection)) ||
           IsAuthoredField(spec->GetField(FieldNames::variantSetNames));
}

bool PrimIndexHasVariantMetadata(const CompositionGraph& graph,
                                 const PrimIndex& idx,
                                 const Path& primPath) {
    for (const auto& entry : idx.entries) {
        const Layer& layer = graph.GetLayer(entry.layerIndex);
        Path sourcePath = entry.pathMapping->isIdentity ? primPath :
            OpinionSourcePath(entry, primPath);
        if (sourcePath.IsEmpty()) continue;
        if (SpecHasVariantMetadata(layer.GetPrimSpec(sourcePath))) return true;
    }
    return false;
}

std::optional<bool> GetComposedInstanceableOpinion(
        const CompositionGraph& graph,
        const PrimIndex& idx,
        const Path& primPath) {
    for (const auto& entry : idx.entries) {
        const Layer& layer = graph.GetLayer(entry.layerIndex);
        Path sourcePath = entry.pathMapping->isIdentity ? primPath :
            OpinionSourcePath(entry, primPath);
        if (sourcePath.IsEmpty()) continue;
        const Spec* spec = layer.GetPrimSpec(sourcePath);
        auto opinion = ReadInstanceableOpinion(spec);
        if (opinion) return opinion;
    }
    return std::nullopt;
}

bool WouldComposeInstanceableTrueAfterAppend(const CompositionGraph& graph,
                                             const Path& primPath,
                                             const Spec* appendedSourceSpec) {
    if (const PrimIndex* current = graph.GetPrimIndex(primPath)) {
        auto existing = GetComposedInstanceableOpinion(graph, *current, primPath);
        if (existing) return *existing;
    }
    return ReadInstanceableOpinion(appendedSourceSpec).value_or(false);
}

bool CanShareInstanceableExternalArc(const CompositionGraph& graph,
                                     const CollectedListOpArcs& collected) {
    // Relocates can change source->target namespace mapping per authoring
    // layer stack. Keep this slice conservative until the representative key
    // fingerprints relocate mappings explicitly.
    if (collected.relocateStackId != kInvalidLayerStackId &&
        !LayerStackRelocates(graph, collected.relocateStackId).empty())
        return false;
    if (collected.arcAuthorMapping &&
        !collected.arcAuthorMapping->extraMappings.empty())
        return false;
    return true;
}

bool TryUseRepresentativeForInstanceableArc(CompositionGraph& graph,
                                            const Path& primPath,
                                            const Layer& layer,
                                            const Path& sourcePath,
                                            const Spec* sourceSpec,
                                            const OpinionEntry& entry,
                                            const ComposeContext& ctx,
                                            ArcType arcType) {
    if (!ctx.instanceArcCache) return false;
    if (!WouldComposeInstanceableTrueAfterAppend(graph, primPath, sourceSpec))
        return false;
    // Variant selections need a later resolve pass to materialize selected
    // variant bodies. Share only non-variant instanceable arcs in this slice.
    if (SpecHasVariantMetadata(sourceSpec)) return false;
    if (const PrimIndex* current = graph.GetPrimIndex(primPath)) {
        if (PrimIndexHasVariantMetadata(graph, *current, primPath)) return false;
    }

    InstanceArcKey key;
    // The cache is composition-scoped, so layer object identity is stable
    // and avoids relying on path string normalization for equivalence.
    key.layer = &layer;
    key.sourcePath = sourcePath;
    key.arcType = arcType;
    key.retiming = entry.retiming;
    key.strengthChain = EntryStrengthChainCopy(entry);

    auto [it, inserted] =
        ctx.instanceArcCache->representativeByKey.emplace(std::move(key), primPath);
    if (inserted || it->second == primPath) return false;

    IndexSinglePrimFromArc(layer, sourcePath, nullptr, entry, sourceSpec,
                           graph, nullptr, false,
                           /*seedLocalOpinions=*/true);
    graph.instanceRepresentatives[primPath] = it->second;
    return true;
}

// Spec §11.2.5 (ancestral variant selection): when a composition arc
// targets a subroot prim path such as /A/B/C, any variant selections
// authored on the STRICT ancestors (/A, /A/B) in the SOURCE layer must
// be folded into the effective source path so the target's variant-body
// content composes in. The target prim itself (/A/B/C) is NOT decorated
// here — its own variantSelection is composed by the regular variant
// arc on the target-side prim (ExpandVariantArcs) and decorating it
// here would double-apply.
//
// Walk strict-ancestor prefixes from the root. At each prefix, read
// the Prim spec at that prefix (in the source layer) and, if it
// authors a variantSelection Dictionary, decorate the current path
// element with those {setName=variantName} selections so the
// subsequent prefix lookup hits the variant body's namespace (e.g.
// /A with v=y becomes /A{v=y}, which then lets /A{v=y}/B find the B
// spec authored inside that variant body).
//
// Returns the variant-decorated path. If no strict ancestor authors
// any variantSelection, returns a path structurally equal to the input.
static Path ApplyAncestralVariantSelections(const Layer& layer,
                                             const Path& sourcePath) {
    if (!sourcePath.IsAbsolute()) return sourcePath;
    const auto& elems = GetCachedPathElements(sourcePath);
    if (elems.size() < 2) return sourcePath;  // no strict ancestors

    Path cur = Path::AbsoluteRoot();
    // Decorate ancestors only — stop before the last element.
    for (size_t i = 0; i + 1 < elems.size(); ++i) {
        cur = cur.AppendChild(elems[i].name);
        const Spec* spec = layer.GetPrimSpec(cur);
        if (!spec) continue;
        const auto* field = spec->GetField(FieldNames::variantSelection);
        if (!field) continue;
        const auto* dict = field->Get<Dictionary>();
        if (!dict) continue;
        for (const auto& [setNameStr, valueField] : *dict) {
            Token variantName;
            if (const auto* tok = valueField.Get<Token>()) {
                variantName = *tok;
            } else if (const auto* s = valueField.Get<String>()) {
                variantName = Token(*s);
            }
            if (variantName.IsEmpty()) continue;
            cur = cur.AppendVariantSelection(Token(setNameStr), variantName);
        }
    }
    // Append the target element unchanged.
    cur = cur.AppendChild(elems.back().name);
    return cur;
}

// Resolve references for a single prim. Returns true if any new layers were added.
// startIdx = first opinion entry to scan; earlier entries were already scanned
// on a previous visit (their arcs, if any, were marked consumed). Each entry
// is scanned at most once across all visits of a given prim.
// Walk the prim's opinion entries strongest-first and compose the
// variantSelection Dictionary per spec §11.2. Stronger layers' keys
// override weaker. Returns a setName -> variantName map. Returns an
// empty map when no selections are authored.
using VariantSelectionMap =
    std::unordered_map<Token, Token, Token::Hash>;

static VariantSelectionMap ResolveVariantSelections(
        const CompositionGraph& graph,
        const PrimIndex* primIndex,
        const Path& primPath) {
    VariantSelectionMap selections;
    if (!primIndex) return selections;

    for (const auto& entry : primIndex->entries) {
        const Layer& layer = graph.GetLayer(entry.layerIndex);
        Path sourcePath = OpinionSourcePath(entry, primPath);
        if (sourcePath.IsEmpty()) continue;
        const auto* spec = layer.GetPrimSpec(sourcePath);
        if (!spec) continue;
        const auto* field = spec->GetField(FieldNames::variantSelection);
        if (!field) continue;
        const auto* dict = field->Get<Dictionary>();
        if (!dict) continue;
        for (const auto& [setNameStr, valueField] : *dict) {
            Token setName(setNameStr);
            if (selections.count(setName)) continue;  // stronger wins
            // The selected variant name may be authored as String or Token.
            if (const auto* tok = valueField.Get<Token>()) {
                if (!tok->IsEmpty()) selections.emplace(setName, *tok);
            } else if (const auto* s = valueField.Get<String>()) {
                if (!s->empty()) selections.emplace(setName, Token(*s));
            }
        }
    }
    return selections;
}

// For each selected (setName, variantName) on primPath, locate any Variant
// spec at /primPath{setName=variantName} in the prim's current opinion
// layers, and merge its descendant content into primPath's namespace via
// IndexDescendantPrims tagged ArcType::Variant.
//
// Variant opinions are stronger than References/Payload but weaker than
// Local opinions (LIVRPS §10.5). Current insertion order places Local
// opinions first (built in BuildPrimIndices), then any Variant opinions
// added here, then References/Payload brought in by ResolvePrim* below —
// so the arcType-stable-sort at the end of ResolveGraphReferences ends
// up ordering entries correctly for query.
//
// Returns true if any new opinion entries were added.
static bool ExpandVariantArcs(
        CompositionGraph& graph,
        const Path& primPath,
        size_t startIdx,
        const VariantSelectionMap& selections,
        const ComposeContext& ctx,
        std::vector<QueuedArcWork>* newPrims) {
    if (selections.empty()) return false;
    PrimIndex* idx = const_cast<PrimIndex*>(graph.GetPrimIndex(primPath));
    if (!idx) return false;

    // Snapshot the "source" entries to iterate — new entries added by the
    // expansion below would otherwise recurse into themselves. The snapshot
    // is shallow: we only need the indices to look up entries in the idx
    // vector after new appends, but entries may reallocate, so copy the
    // OpinionEntry values we care about.
    const size_t entriesBefore = idx->entries.size();

    bool added = false;
    for (size_t ei = startIdx; ei < entriesBefore; ++ei) {
        const OpinionEntry sourceEntry = idx->entries[ei];
        const Layer& layer = graph.GetLayer(sourceEntry.layerIndex);
        Path sourcePath = OpinionSourcePath(sourceEntry, primPath);
        if (sourcePath.IsEmpty()) continue;

        for (const auto& [setName, variantName] : selections) {
            // Variant specs live at source-layer paths of the form
            // /SourcePrim{setName=variantName}. The prefix-matching logic in
            // RemapPath ignores variant selections on elements, so this
            // mapping remaps /SourcePrim{set=var}/Child back to the target
            // /primPath/Child cleanly.
            Path variantSource = sourcePath.AppendVariantSelection(setName, variantName);
            const auto* variantSpec = layer.GetSpec(variantSource);
            if (!variantSpec || variantSpec->GetType() != SpecType::Variant) continue;

            OpinionEntry variantEntry;
            variantEntry.layerIndex = sourceEntry.layerIndex;
            variantEntry.pathMapping = MakeArcPathMapping(
                graph,
                variantSource, primPath,
                LayerStackIdForLayer(graph, sourceEntry.layerIndex),
                sourcePath,
                sourceEntry.pathMapping,
                /*fallbackIdentity=*/true);
            variantEntry.retiming = sourceEntry.retiming;
            variantEntry.arcType = ArcType::Variant;
            SetStrengthChain(
                variantEntry,
                MakeStrengthChain(
                    StrengthPrefixForAuthoredArc(sourceEntry), ArcType::Variant,
                    sourcePath));

            IndexDescendantPrims(layer, variantSource, primPath,
                                 variantEntry, graph,
                                 ctx.sourceDescendantCache.get(), newPrims);
            added = true;
        }
    }
    return added;
}

static bool RejectVariantSelectionArcTarget(
        const std::optional<Path>& targetPath,
        DiagnosticCollector* diagnostics,
        DiagSeverity severity,
        DiagCategory category,
        ArcType arcType,
        const Path& primPath,
        const std::string& layerPath) {
    if (!targetPath || !targetPath->HasVariantSelections()) return false;
    if (diagnostics) {
        const char* arcName =
            arcType == ArcType::Payload ? "Payload" : "Reference";
        diagnostics->Add({
            severity,
            category,
            std::string(arcName) +
                " target path may not contain variant selections: " +
                targetPath->GetText(),
            primPath.GetText(),
            layerPath,
            targetPath->GetText(),
            arcType});
    }
    return true;
}

template <typename T, typename Fn>
bool ForEachListOpItemNoCopy(const ListOp<T>& op, Fn&& fn) {
    uint32_t order = 0;
    bool any = false;
    if (op.IsExplicit()) {
        for (const auto& item : op.GetExplicitItems()) {
            any = true;
            if (!fn(item, order++)) return any;
        }
        return any;
    }

    const auto& appended = op.GetAppendedItems();
    for (const auto& item : op.GetPrependedItems()) {
        if (std::find(appended.begin(), appended.end(), item) != appended.end())
            continue;
        any = true;
        if (!fn(item, order++)) return any;
    }
    for (const auto& item : appended) {
        any = true;
        if (!fn(item, order++)) return any;
    }
    return any;
}

bool ResolvePrimReferences(CompositionGraph& graph,
                           const Path& primPath,
                           const std::string& anchorPath,
                           const ComposeContext& ctx,
                           ConsumedArcs& consumed,
                           std::string& error,
                           size_t startIdx = 0,
                           std::vector<QueuedArcWork>* newPrims = nullptr) {
    AccumTimer _tot(g_stats.ns_resolve_refs_total);
    const PrimIndex* idx = graph.GetPrimIndex(primPath);
    if (!idx) return false;
    if (startIdx >= idx->entries.size()) return false;

    CollectedListOpArcs collected;
    {
        AccumTimer _s(g_stats.ns_entry_scan_refs);
        collected = CollectListOpArcs(graph, *idx, primPath, startIdx,
                                       FieldNames::references,
                                       consumed.references, anchorPath);
    }
    if (!collected.found || !collected.combined) return false;

    const std::string& refAnchorPath = collected.anchorLayer;

    // Mark references as consumed from startIdx forward. Earlier entries were
    // already marked on the visit that scanned them; re-marking is idempotent
    // but the per-entry MapToSource+GetText isn't free, so skip it.
    for (size_t ei = startIdx; ei < idx->entries.size(); ++ei) {
        const auto& entry = idx->entries[ei];
        Path sourcePath = OpinionSourcePath(entry, primPath);
        if (!sourcePath.IsEmpty())
            consumed.references.insert({entry.layerIndex, sourcePath});
    }

    bool addedLayers = false;

    bool hasRefs = ForEachListOpItemNoCopy(
        *collected.combined,
        [&](const Reference& ref, uint32_t siblingOrder) -> bool {
        if (RejectVariantSelectionArcTarget(
                ref.primPath, ctx.diagnostics, DiagSeverity::Error,
                DiagCategory::InvalidReferenceTarget, ArcType::Reference,
                primPath, refAnchorPath)) {
            return true;
        }
        if (ref.assetPath && !ref.assetPath->empty()) {
            g_stats.ext_refs++;
            Retiming refRetiming = NormalizeRetimingScale(
                Retiming{ref.offset, ref.scale},
                ArcType::Reference,
                primPath.GetText(),
                refAnchorPath,
                *ref.assetPath,
                ctx.diagnostics);

            // External reference — resolve relative to the layer that contains it
            std::string resolvedPath;
            {
                AccumTimer _r(g_stats.ns_resolver_calls);
                resolvedPath = ResolveAssetCached(ctx, refAnchorPath, *ref.assetPath);
            }
            if (resolvedPath.empty()) {
                if (ctx.diagnostics) {
                    ctx.diagnostics->Add({
                        DiagSeverity::Error, DiagCategory::MissingReference,
                        "Failed to resolve reference asset: " + *ref.assetPath,
                        primPath.GetText(), refAnchorPath,
                        *ref.assetPath, ArcType::Reference});
                }
                return true;
            }

            CachedParseOutput parseResult;
            {
                AccumTimer _p(g_stats.ns_cached_parse);
                parseResult = CachedParseUsdFile(resolvedPath, ctx);
            }
            if (!parseResult.success) {
                if (ctx.diagnostics) {
                    ctx.diagnostics->Add({
                        DiagSeverity::Error, DiagCategory::ReferenceParseFail,
                        "Failed to parse reference " + *ref.assetPath + ": " + parseResult.error,
                        primPath.GetText(), refAnchorPath,
                        *ref.assetPath, ArcType::Reference});
                }
                return true;
            }

            // Gather the referenced layer stack (resolves sublayers recursively)
            GatherResult gathered;
            {
                AccumTimer _g(g_stats.ns_gather_layer_stack);
                gathered = GatherLayerStackCached(parseResult.layer, resolvedPath,
                                                  refRetiming, ctx);
            }
            if (!gathered.success) {
                // Fatal error from gather (cycle) — propagate
                error = gathered.error;
                return false;
            }
            // Merge any non-fatal diagnostics from the gathered stack
            if (ctx.diagnostics && !gathered.diagnostics.empty()) {
                ctx.diagnostics->Merge(std::move(gathered.diagnostics));
            }

            // Determine source prim path
            Path sourcePrimPath;
            if (ref.primPath) {
                sourcePrimPath = *ref.primPath;
            } else {
                if (!gathered.layers.empty()) {
                    auto dp = gathered.layers[0].layer->GetLayerSpec().GetDefaultPrim();
                    if (dp.GetString().empty()) {
                        if (ctx.diagnostics) {
                            ctx.diagnostics->Add({
                                DiagSeverity::Error, DiagCategory::MissingDefaultPrim,
                                "Reference has no target path and referenced layer has no defaultPrim",
                                primPath.GetText(), resolvedPath,
                                *ref.assetPath, ArcType::Reference});
                        }
                        return true;
                    }
                    sourcePrimPath = Path::AbsoluteRoot().AppendChild(dp);
                }
            }

            // Add referenced layers to graph as one layer stack.
            std::vector<size_t> newLayerIndices;
            if (gathered.layers.size() == 1 && gathered.layers[0].layer) {
                const Layer& refLayer = *gathered.layers[0].layer;
                Path effectiveSource = ApplyAncestralVariantSelections(
                    refLayer, sourcePrimPath);
                const Spec* sourceSpec = refLayer.GetSpec(effectiveSource);
                const bool preferInstanceableRepresentative =
                    ReadInstanceableOpinion(sourceSpec).value_or(false);
                if (!preferInstanceableRepresentative) {
                    if (auto reusedIdx = TryGetReusableLeafLayerIndex(
                            graph, gathered, refLayer, effectiveSource,
                            sourceSpec, ctx)) {
                        newLayerIndices.push_back(*reusedIdx);
                    }
                }
            }
            if (newLayerIndices.empty()) {
                size_t referencedStackId = StartLayerStack(graph);
                size_t referencedStackBegin = graph.GetNumLayers();
                newLayerIndices.reserve(gathered.layers.size());
                for (auto& gl : gathered.layers) {
                    size_t newIdx = graph.GetNumLayers();
                    graph.layers.push_back(std::move(gl.layer));
                    graph.layerPaths.push_back(gl.sourcePath);
                    graph.layerRetimings.push_back(gl.retiming);
                    graph.layerStackIds.push_back(referencedStackId);
                    newLayerIndices.push_back(newIdx);
                    addedLayers = true;
                }
                FinishLayerStack(graph, referencedStackId, referencedStackBegin,
                                 graph.GetNumLayers(), ctx.diagnostics);
                SetLayerStackIntroduction(
                    graph,
                    referencedStackId,
                    collected.relocateStackId,
                    MakeArcPathMapping(graph, sourcePrimPath, primPath,
                                       collected.relocateStackId,
                                       collected.arcAuthorSourcePath,
                                       collected.arcAuthorMapping,
                                       /*fallbackIdentity=*/true),
                    AppendStrengthPrefix(collected.arcStrengthPrefix,
                                         ArcType::Reference,
                                         collected.arcAuthorSourcePath,
                                         siblingOrder));
            }

            // Create opinion entries for each layer in the referenced stack.
            for (size_t newIdx : newLayerIndices) {
                const Layer& refLayer = graph.GetLayer(newIdx);
                // Spec §11.2.5: subroot references inherit variant
                // selections from their source ancestors. Decorate the
                // source path with those selections so descendants land
                // on the correct variant-body spec chain.
                Path effectiveSource = ApplyAncestralVariantSelections(
                    refLayer, sourcePrimPath);
                OpinionEntry entry;
                entry.layerIndex = newIdx;
                entry.pathMapping =
                    MakeArcPathMapping(graph, effectiveSource, primPath,
                                       collected.relocateStackId,
                                       collected.arcAuthorSourcePath,
                                       collected.arcAuthorMapping);
                entry.retiming = graph.layerRetimings[newIdx];
                entry.arcType = ArcType::Reference;
                SetStrengthChain(
                    entry,
                    MakeStrengthChain(
                        collected.arcStrengthPrefix, ArcType::Reference,
                        collected.arcAuthorSourcePath, /*implied=*/false,
                        siblingOrder));
                const Spec* sourceSpec = refLayer.GetSpec(effectiveSource);
                if (CanShareInstanceableExternalArc(graph, collected) &&
                    TryUseRepresentativeForInstanceableArc(
                        graph, primPath, refLayer, effectiveSource,
                        sourceSpec, entry, ctx, ArcType::Reference)) {
                    continue;
                }
                IndexDescendantPrims(refLayer, effectiveSource, primPath,
                                     entry, graph,
                                     ctx.sourceDescendantCache.get(), newPrims);
            }
        } else {
            g_stats.int_refs++;
            // Internal reference
            Path sourcePrimPath;
            if (ref.primPath) {
                sourcePrimPath = *ref.primPath;
            } else {
                return true;
            }

            Retiming refRetiming = NormalizeRetimingScale(
                Retiming{ref.offset, ref.scale},
                ArcType::Reference,
                primPath.GetText(),
                refAnchorPath,
                /*assetPath=*/"",
                ctx.diagnostics);

            const PrimIndex* sourceIdx = graph.GetPrimIndex(sourcePrimPath);
            if (!sourceIdx) return true;

            auto sourceEntries = sourceIdx->entries;
            // Reuse one PathMapping across all per-layer entries of the
            // internal ref — they all share the same arc source/target.
            PathMappingPtr mapping = MakeArcPathMapping(
                graph,
                sourcePrimPath, primPath,
                collected.relocateStackId,
                collected.arcAuthorSourcePath,
                collected.arcAuthorMapping,
                /*fallbackIdentity=*/true);
            for (const auto& srcEntry : sourceEntries) {
                const Layer& layer = graph.GetLayer(srcEntry.layerIndex);

                OpinionEntry entry;
                entry.layerIndex = srcEntry.layerIndex;
                entry.pathMapping = mapping;
                entry.retiming.offset = srcEntry.retiming.offset +
                    refRetiming.offset * srcEntry.retiming.scale;
                entry.retiming.scale = srcEntry.retiming.scale * refRetiming.scale;
                entry.arcType = ArcType::Reference;
                SetStrengthChain(
                    entry,
                    MakeStrengthChain(
                        collected.arcStrengthPrefix, ArcType::Reference,
                        collected.arcAuthorSourcePath, /*implied=*/false,
                        siblingOrder));
                IndexDescendantPrims(layer, sourcePrimPath, primPath,
                                     entry, graph,
                                     ctx.sourceDescendantCache.get(), newPrims);
            }
        }
        return true;
    });
    if (!hasRefs) return false;
    if (!error.empty()) return false;

    return addedLayers;
}

// Resolve payloads for a single prim. Same logic as ResolvePrimReferences
// but reads the "payload" field. Per LIVRPS, payloads are weaker than references.
// See ResolvePrimReferences for the startIdx contract.
bool ResolvePrimPayloads(CompositionGraph& graph,
                         const Path& primPath,
                         const std::string& anchorPath,
                         const ComposeContext& ctx,
                         ConsumedArcs& consumed,
                         std::string& error,
                         size_t startIdx = 0,
                         std::vector<QueuedArcWork>* newPrims = nullptr) {
    AccumTimer _tot(g_stats.ns_resolve_payloads_total);
    const PrimIndex* idx = graph.GetPrimIndex(primPath);
    if (!idx) return false;
    if (startIdx >= idx->entries.size()) return false;

    CollectedListOpArcs collected;
    {
        AccumTimer _s(g_stats.ns_entry_scan_payloads);
        collected = CollectListOpArcs(graph, *idx, primPath, startIdx,
                                       FieldNames::payload,
                                       consumed.payloads, anchorPath);
    }
    if (!collected.found || !collected.combined) return false;

    const std::string& plAnchorPath = collected.anchorLayer;

    // Mark payloads as consumed from startIdx forward (earlier entries were
    // marked on the visit that scanned them).
    for (size_t ei = startIdx; ei < idx->entries.size(); ++ei) {
        const auto& entry = idx->entries[ei];
        Path sourcePath = OpinionSourcePath(entry, primPath);
        if (!sourcePath.IsEmpty())
            consumed.payloads.insert({entry.layerIndex, sourcePath});
    }

    bool addedLayers = false;

    bool hasPayloads = ForEachListOpItemNoCopy(
        *collected.combined,
        [&](const Reference& pl, uint32_t siblingOrder) -> bool {
        if (RejectVariantSelectionArcTarget(
                pl.primPath, ctx.diagnostics, DiagSeverity::Warning,
                DiagCategory::InvalidPayloadTarget, ArcType::Payload,
                primPath, plAnchorPath)) {
            return true;
        }
        if (pl.assetPath && !pl.assetPath->empty()) {
            g_stats.ext_payloads++;
            Retiming plRetiming = NormalizeRetimingScale(
                Retiming{pl.offset, pl.scale},
                ArcType::Payload,
                primPath.GetText(),
                plAnchorPath,
                *pl.assetPath,
                ctx.diagnostics);

            std::string resolvedPath;
            {
                AccumTimer _r(g_stats.ns_resolver_calls);
                resolvedPath = ResolveAssetCached(ctx, plAnchorPath, *pl.assetPath);
            }
            if (resolvedPath.empty()) {
                if (ctx.diagnostics) {
                    ctx.diagnostics->Add({
                        DiagSeverity::Warning, DiagCategory::MissingPayload,
                        "Failed to resolve payload asset: " + *pl.assetPath,
                        primPath.GetText(), plAnchorPath,
                        *pl.assetPath, ArcType::Payload});
                }
                return true;
            }

            CachedParseOutput parseResult;
            {
                AccumTimer _p(g_stats.ns_cached_parse);
                parseResult = CachedParseUsdFile(resolvedPath, ctx);
            }
            if (!parseResult.success) {
                if (ctx.diagnostics) {
                    ctx.diagnostics->Add({
                        DiagSeverity::Warning, DiagCategory::PayloadParseFail,
                        "Failed to parse payload " + *pl.assetPath + ": " + parseResult.error,
                        primPath.GetText(), plAnchorPath,
                        *pl.assetPath, ArcType::Payload});
                }
                return true;
            }

            GatherResult gathered;
            {
                AccumTimer _g(g_stats.ns_gather_layer_stack);
                gathered = GatherLayerStackCached(parseResult.layer, resolvedPath,
                                                  plRetiming, ctx);
            }
            if (!gathered.success) {
                // Fatal from gather (cycle) — propagate
                error = gathered.error;
                return false;
            }
            // Merge any non-fatal diagnostics from the gathered stack
            if (ctx.diagnostics && !gathered.diagnostics.empty()) {
                ctx.diagnostics->Merge(std::move(gathered.diagnostics));
            }

            Path sourcePrimPath;
            if (pl.primPath) {
                sourcePrimPath = *pl.primPath;
            } else {
                if (!gathered.layers.empty()) {
                    auto dp = gathered.layers[0].layer->GetLayerSpec().GetDefaultPrim();
                    if (dp.GetString().empty()) {
                        if (ctx.diagnostics) {
                            ctx.diagnostics->Add({
                                DiagSeverity::Warning, DiagCategory::MissingDefaultPrim,
                                "Payload has no target path and referenced layer has no defaultPrim",
                                primPath.GetText(), resolvedPath,
                                *pl.assetPath, ArcType::Payload});
                        }
                        return true;
                    }
                    sourcePrimPath = Path::AbsoluteRoot().AppendChild(dp);
                }
            }

            std::vector<size_t> newLayerIndices;
            if (gathered.layers.size() == 1 && gathered.layers[0].layer) {
                const Layer& plLayer = *gathered.layers[0].layer;
                Path effectiveSource = ApplyAncestralVariantSelections(
                    plLayer, sourcePrimPath);
                const Spec* sourceSpec = plLayer.GetSpec(effectiveSource);
                const bool preferInstanceableRepresentative =
                    ReadInstanceableOpinion(sourceSpec).value_or(false);
                if (!preferInstanceableRepresentative) {
                    if (auto reusedIdx = TryGetReusableLeafLayerIndex(
                            graph, gathered, plLayer, effectiveSource,
                            sourceSpec, ctx)) {
                        newLayerIndices.push_back(*reusedIdx);
                    }
                }
            }
            if (newLayerIndices.empty()) {
                size_t payloadStackId = StartLayerStack(graph);
                size_t payloadStackBegin = graph.GetNumLayers();
                newLayerIndices.reserve(gathered.layers.size());
                for (auto& gl : gathered.layers) {
                    size_t newIdx = graph.GetNumLayers();
                    graph.layers.push_back(std::move(gl.layer));
                    graph.layerPaths.push_back(gl.sourcePath);
                    graph.layerRetimings.push_back(gl.retiming);
                    graph.layerStackIds.push_back(payloadStackId);
                    newLayerIndices.push_back(newIdx);
                    addedLayers = true;
                }
                FinishLayerStack(graph, payloadStackId, payloadStackBegin,
                                 graph.GetNumLayers(), ctx.diagnostics);
                SetLayerStackIntroduction(
                    graph,
                    payloadStackId,
                    collected.relocateStackId,
                    MakeArcPathMapping(graph, sourcePrimPath, primPath,
                                       collected.relocateStackId,
                                       collected.arcAuthorSourcePath,
                                       collected.arcAuthorMapping,
                                       /*fallbackIdentity=*/true),
                    AppendStrengthPrefix(collected.arcStrengthPrefix,
                                         ArcType::Payload,
                                         collected.arcAuthorSourcePath,
                                         siblingOrder));
            }

            for (size_t newIdx : newLayerIndices) {
                const Layer& plLayer = graph.GetLayer(newIdx);
                // Spec §11.2.5: subroot payloads inherit variant
                // selections from their source ancestors, same as refs.
                Path effectiveSource = ApplyAncestralVariantSelections(
                    plLayer, sourcePrimPath);
                OpinionEntry entry;
                entry.layerIndex = newIdx;
                entry.pathMapping =
                    MakeArcPathMapping(graph, effectiveSource, primPath,
                                       collected.relocateStackId,
                                       collected.arcAuthorSourcePath,
                                       collected.arcAuthorMapping);
                entry.retiming = graph.layerRetimings[newIdx];
                entry.arcType = ArcType::Payload;
                SetStrengthChain(
                    entry,
                    MakeStrengthChain(
                        collected.arcStrengthPrefix, ArcType::Payload,
                        collected.arcAuthorSourcePath, /*implied=*/false,
                        siblingOrder));
                const Spec* sourceSpec = plLayer.GetSpec(effectiveSource);
                if (CanShareInstanceableExternalArc(graph, collected) &&
                    TryUseRepresentativeForInstanceableArc(
                        graph, primPath, plLayer, effectiveSource,
                        sourceSpec, entry, ctx, ArcType::Payload)) {
                    continue;
                }
                IndexDescendantPrims(plLayer, effectiveSource, primPath,
                                     entry, graph,
                                     ctx.sourceDescendantCache.get(), newPrims);
            }
        } else if (pl.primPath) {
            g_stats.int_payloads++;
            // Internal payload
            Path sourcePrimPath = *pl.primPath;
            Retiming plRetiming = NormalizeRetimingScale(
                Retiming{pl.offset, pl.scale},
                ArcType::Payload,
                primPath.GetText(),
                plAnchorPath,
                /*assetPath=*/"",
                ctx.diagnostics);

            const PrimIndex* sourceIdx = graph.GetPrimIndex(sourcePrimPath);
            if (!sourceIdx) return true;

            auto sourceEntries = sourceIdx->entries;
            // One shared PathMapping for all per-layer entries of the payload.
            PathMappingPtr mapping = MakeArcPathMapping(
                graph,
                sourcePrimPath, primPath,
                collected.relocateStackId,
                collected.arcAuthorSourcePath,
                collected.arcAuthorMapping,
                /*fallbackIdentity=*/true);
            for (const auto& srcEntry : sourceEntries) {
                const Layer& layer = graph.GetLayer(srcEntry.layerIndex);
                OpinionEntry entry;
                entry.layerIndex = srcEntry.layerIndex;
                entry.pathMapping = mapping;
                entry.retiming.offset = srcEntry.retiming.offset +
                    plRetiming.offset * srcEntry.retiming.scale;
                entry.retiming.scale = srcEntry.retiming.scale * plRetiming.scale;
                entry.arcType = ArcType::Payload;
                SetStrengthChain(
                    entry,
                    MakeStrengthChain(
                        collected.arcStrengthPrefix, ArcType::Payload,
                        collected.arcAuthorSourcePath, /*implied=*/false,
                        siblingOrder));
                IndexDescendantPrims(layer, sourcePrimPath, primPath,
                                     entry, graph,
                                     ctx.sourceDescendantCache.get(), newPrims);
            }
        }
        return true;
    });
    if (!hasPayloads) return false;
    if (!error.empty()) return false;

    return addedLayers;
}

struct PathArcLayerStackTarget {
    size_t stackId = kInvalidLayerStackId;
    Path targetPath;
    bool authored = false;
};

static PathMappingPtr LayerStackImpliedPathArcMapping(
        const CompositionGraph& graph,
        size_t stackId) {
    if (stackId < graph.layerStacks.size() &&
        graph.layerStacks[stackId].impliedInheritMapping) {
        return graph.layerStacks[stackId].impliedInheritMapping;
    }
    return IdentityPathMappingPtr();
}

static StrengthChain LayerStackStrengthPrefix(
        const CompositionGraph& graph,
        size_t stackId) {
    if (stackId < graph.layerStacks.size()) {
        return graph.layerStacks[stackId].strengthPrefix;
    }
    return {};
}

static StrengthChain StripLayerStackStrengthPrefix(
        const StrengthChain& arcStrengthPrefix,
        const StrengthChain& stackStrengthPrefix) {
    if (stackStrengthPrefix.empty()) return arcStrengthPrefix;
    auto stripAt = [&](size_t offset) -> std::optional<StrengthChain> {
        if (offset + stackStrengthPrefix.size() > arcStrengthPrefix.size()) {
            return std::nullopt;
        }
        for (size_t i = 0; i < stackStrengthPrefix.size(); ++i) {
            if (!SameStrengthComponent(
                    arcStrengthPrefix[offset + i],
                    stackStrengthPrefix[i])) {
                return std::nullopt;
            }
        }
        StrengthChain result;
        result.reserve(arcStrengthPrefix.size() - stackStrengthPrefix.size());
        result.insert(result.end(),
                      arcStrengthPrefix.begin(),
                      arcStrengthPrefix.begin() + offset);
        result.insert(result.end(),
                      arcStrengthPrefix.begin() + offset +
                          stackStrengthPrefix.size(),
                      arcStrengthPrefix.end());
        return result;
    };

    if (auto stripped = stripAt(0)) return *stripped;

    size_t afterLeadingSpecializes = 0;
    while (afterLeadingSpecializes < arcStrengthPrefix.size() &&
           arcStrengthPrefix[afterLeadingSpecializes].arcType ==
               ArcType::Specialize) {
        ++afterLeadingSpecializes;
    }
    if (afterLeadingSpecializes > 0) {
        if (auto stripped = stripAt(afterLeadingSpecializes)) {
            return *stripped;
        }
    }

    if (stackStrengthPrefix.size() > arcStrengthPrefix.size()) {
        return arcStrengthPrefix;
    }
    return arcStrengthPrefix;
}

static StrengthChain ImpliedPathArcStrengthPrefix(
        const CompositionGraph& graph,
        size_t impliedStackId,
        const StrengthChain& nestedStrengthPrefix) {
    StrengthChain result = LayerStackStrengthPrefix(graph, impliedStackId);
    result.insert(result.end(),
                  nestedStrengthPrefix.begin(),
                  nestedStrengthPrefix.end());
    return result;
}

static StrengthChain ImpliedSpecializeStrengthPrefix(
        const CompositionGraph& graph,
        size_t impliedStackId,
        const StrengthChain& nestedStrengthPrefix) {
    StrengthChain result = nestedStrengthPrefix;
    size_t insertAt = 0;
    while (insertAt < result.size() &&
           result[insertAt].arcType == ArcType::Specialize) {
        ++insertAt;
    }
    auto stackPrefix = LayerStackStrengthPrefix(graph, impliedStackId);
    result.insert(result.begin() + insertAt,
                  stackPrefix.begin(),
                  stackPrefix.end());
    return result;
}

static StrengthChain MakeSpecializeStrengthChain(
        const StrengthChain& prefix,
        const Path& arcAuthorPath,
        bool implied = false,
        uint32_t siblingOrder = 0) {
    StrengthChain result = prefix;
    size_t insertAt = 0;
    while (insertAt < result.size() &&
           result[insertAt].arcType == ArcType::Specialize) {
        ++insertAt;
    }
    result.insert(result.begin() + insertAt,
                  MakeStrengthComponent(
                      ArcType::Specialize, arcAuthorPath, implied,
                      siblingOrder));
    result.push_back(MakeStrengthComponent(ArcType::Local));
    return result;
}

static bool LayerStackLayerRange(const CompositionGraph& graph,
                                 size_t stackId,
                                 size_t& begin,
                                 size_t& end) {
    stackId = NormalizeLayerStackId(graph, stackId);
    if (stackId >= graph.layerStacks.size()) return false;
    const auto& stack = graph.layerStacks[stackId];
    begin = std::min(stack.beginLayerIndex, graph.GetNumLayers());
    end = std::min(stack.endLayerIndex, graph.GetNumLayers());
    return begin < end;
}

static std::string LayerStackAnchorPath(const CompositionGraph& graph,
                                        size_t stackId) {
    size_t begin = 0, end = 0;
    if (LayerStackLayerRange(graph, stackId, begin, end) &&
        begin < graph.layerPaths.size()) {
        return graph.layerPaths[begin];
    }
    return graph.layerPaths.empty() ? std::string{} : graph.layerPaths.front();
}

static std::vector<PathArcLayerStackTarget> BuildPathArcLayerStackTargets(
        const CompositionGraph& graph,
        size_t authoredStackId,
        const Path& authoredTarget) {
    std::vector<PathArcLayerStackTarget> targets;
    authoredStackId = NormalizeLayerStackId(graph, authoredStackId);
    if (authoredStackId >= graph.layerStacks.size()) return targets;

    targets.push_back({authoredStackId, authoredTarget, true});

    PathMappingPtr authoredStackMapping =
        LayerStackImpliedPathArcMapping(graph, authoredStackId);
    Path composedTarget = authoredStackMapping->MapToTarget(authoredTarget);
    if (composedTarget.IsEmpty()) return targets;

    std::unordered_set<size_t> visited;
    visited.insert(authoredStackId);
    size_t parentStackId = graph.layerStacks[authoredStackId].parentStackId;
    while (parentStackId < graph.layerStacks.size() &&
           visited.insert(parentStackId).second) {
        PathMappingPtr parentMapping =
            LayerStackImpliedPathArcMapping(graph, parentStackId);
        Path impliedTarget = parentMapping->MapToSource(composedTarget);
        if (!impliedTarget.IsEmpty()) {
            targets.push_back({parentStackId, impliedTarget, false});
        }
        parentStackId = graph.layerStacks[parentStackId].parentStackId;
    }
    return targets;
}

// Resolve inherits opinions for one prim. Spec §10.5.1: inheritPaths is a
// ListOp<ObjectPath> whose entries name absolute prim paths in the layer
// stack where the inherit was authored. For each authored target, walk the
// authored layer stack and every upstream layer stack that introduced it via
// reference/payload arcs. Upstream targets are implied inherit arcs: their
// target paths are computed through layer-stack namespace mappings and
// missing implied targets are ignored.
//
// If no layer in the authored stack has a spec at the target, emit a
// composition-error diagnostic per spec lines 459–461.
//
// Cycle handling: spec is silent on inherits cycles (the cycle language at
// composition.md lines 88–89 is sublayer-only). The sandbox golden's
// `direct_cycle` case prescribes an explicit error; nanousd's iterative
// resolve loop instead converges via the existing startIdx / depth-limit
// machinery and emits no error. Both are spec-permissible
// implementation conventions; the divergence is documented for the future
// AOUSD WG spec-gap report mentioned in proof/sandboxes/inherits/
// spec-ambiguity-notes.md.
bool ResolvePrimInherits(CompositionGraph& graph,
                         const Path& primPath,
                         const ComposeContext& ctx,
                         size_t startIdx = 0,
                         std::vector<QueuedArcWork>* newPrims = nullptr) {
    const PrimIndex* idx = graph.GetPrimIndex(primPath);
    if (!idx) return false;
    if (startIdx >= idx->entries.size()) return false;

    // Same-layer inherit chains must be allowed to re-scan a source path
    // when that path is encountered through a different composition chain.
    // A global (layer,path) consumed key would conflate "/B was scanned while
    // composing /B" with "/B was scanned while composing /A -> /B" and drop
    // /A -> /B -> /C opinions. startIdx bounds re-scans to newly-attached
    // entries; kMaxReferenceDepth bounds cycles.
    static const ankerl::unordered_dense::set<ConsumedArcs::Key, ConsumedArcs::KeyHash>
        kEmptyConsumed;
    auto collectedGroups = CollectListOpPathArcGroups(
        graph, *idx, primPath, startIdx,
        FieldNames::inheritPaths, kEmptyConsumed);
    if (collectedGroups.empty()) return false;

    bool addedAnything = false;
    for (const auto& collected : collectedGroups) {
        if (!collected.found || !collected.combined) continue;

        auto targets = collected.combined->GetItems();
        if (targets.empty()) continue;

        size_t authoredStackId =
            NormalizeLayerStackId(graph, collected.relocateStackId);
        auto nestedStrengthPrefix = StripLayerStackStrengthPrefix(
            collected.arcStrengthPrefix,
            LayerStackStrengthPrefix(graph, authoredStackId));

        for (size_t targetIndex = 0;
             targetIndex < targets.size();
             ++targetIndex) {
            const auto& target = targets[targetIndex];
            if (target.IsEmpty()) continue;
            uint32_t siblingOrder = static_cast<uint32_t>(targetIndex);

            bool foundAuthoredLayer = false;
            auto stackTargets = BuildPathArcLayerStackTargets(
                graph, authoredStackId, target);
            for (const auto& stackTarget : stackTargets) {
                size_t begin = 0, end = 0;
                if (!LayerStackLayerRange(
                        graph, stackTarget.stackId, begin, end)) {
                    continue;
                }

                PathMappingPtr authorMapping = stackTarget.authored ?
                    collected.arcAuthorMapping :
                    LayerStackImpliedPathArcMapping(graph, stackTarget.stackId);
                Path authorSourcePath = stackTarget.authored ?
                    collected.arcAuthorSourcePath :
                    authorMapping->MapToSource(primPath);
                if (authorSourcePath.IsEmpty()) {
                    authorSourcePath = primPath;
                }

                StrengthChain inheritStrengthPrefix =
                    stackTarget.authored ?
                    collected.arcStrengthPrefix :
                    ImpliedPathArcStrengthPrefix(
                        graph, stackTarget.stackId, nestedStrengthPrefix);

                PathMappingPtr mapping = MakeArcPathMapping(
                    graph,
                    stackTarget.targetPath, primPath,
                    stackTarget.stackId,
                    authorSourcePath,
                    authorMapping,
                    /*fallbackIdentity=*/true);

                for (size_t li = begin; li < end; ++li) {
                    const Layer& layer = graph.GetLayer(li);
                    const Spec* targetSpec =
                        layer.GetSpec(stackTarget.targetPath);
                    if (!targetSpec ||
                        targetSpec->GetType() != SpecType::Prim) {
                        continue;
                    }
                    if (stackTarget.authored) foundAuthoredLayer = true;

                    OpinionEntry entry;
                    entry.layerIndex = li;
                    entry.pathMapping = mapping;
                    entry.retiming = graph.layerRetimings[li];
                    entry.arcType = ArcType::Inherits;
                    SetStrengthChain(
                        entry,
                        MakeStrengthChain(
                            inheritStrengthPrefix, ArcType::Inherits,
                            authorSourcePath, /*implied=*/!stackTarget.authored,
                            siblingOrder));
                    IndexDescendantPrims(layer, stackTarget.targetPath,
                                         primPath, entry, graph,
                                         ctx.sourceDescendantCache.get(), newPrims);
                    addedAnything = true;
                }
            }
            if (!foundAuthoredLayer && ctx.diagnostics) {
                ctx.diagnostics->Add({
                    DiagSeverity::Error,
                    DiagCategory::MissingInheritTarget,
                    "Inherit target has no specs in the layer stack: " +
                        target.GetText(),
                    primPath.GetText(),
                    LayerStackAnchorPath(graph, authoredStackId),
                    target.GetText(),
                    ArcType::Inherits});
            }
        }
    }
    return addedAnything;
}

// Resolve specializes opinions for one prim. Spec §10.5.5: the specializes
// arc composes "exactly the same as inherits, except that the list of
// specializes are computed using the `specializes` field" (composition.md
// line 535), but opinions introduced by specializes are globally weaker
// than opinions for the specializing prim.
//
// Strength position: Specializes is the WEAKEST arc per LIVRPS (line 836)
// and is *globally* weakest per §10.5.5 lines 861-868 (an opinion
// introduced by specializes outranked by every other opinion, including
// across reference chains). MakeSpecializeStrengthChain encodes this by
// inserting each new specializes arc before any non-specializes prefix
// components, so a ref/payload opinion for A remains stronger than
// specialized opinions for B.
//
// Cycle handling: spec is silent (the cycle language at composition.md
// 88-89 is sublayer-only); nanousd's resolve loop converges via the
// startIdx mechanism + kMaxReferenceDepth rather than emitting an
// explicit error. Both are spec-permissible.
//
// Dedup: unlike ResolvePrimReferences/Payloads, this resolver does NOT
// use a (layerIdx, sourcePath) consumed-set. That key conflates "/B's
// specializes was read while composing /A's chain" with "/B's specializes
// was read while composing /B itself" — fine for multi-layer reference
// chains where each chain hop creates a new layer index, but wrong for
// single-layer specializes/inherits chains. Re-scanning is bounded by
// the `startIdx` parameter (each prim visit only processes entries
// added since the last visit), so dropping the dedup just bounds the
// cycle work by kMaxReferenceDepth instead of a hash-set lookup.
bool ResolvePrimSpecializes(CompositionGraph& graph,
                             const Path& primPath,
                             const ComposeContext& ctx,
                             size_t startIdx = 0,
                             std::vector<QueuedArcWork>* newPrims = nullptr) {
    const PrimIndex* idx = graph.GetPrimIndex(primPath);
    if (!idx) return false;
    if (startIdx >= idx->entries.size()) return false;

    static const ankerl::unordered_dense::set<ConsumedArcs::Key, ConsumedArcs::KeyHash>
        kEmptyConsumed;
    auto collectedGroups = CollectListOpPathArcGroups(
        graph, *idx, primPath, startIdx,
        FieldNames::specializes, kEmptyConsumed);
    if (collectedGroups.empty()) return false;

    bool addedAnything = false;
    for (const auto& collected : collectedGroups) {
        if (!collected.found || !collected.combined) continue;

        auto targets = collected.combined->GetItems();
        if (targets.empty()) continue;

        size_t authoredStackId =
            NormalizeLayerStackId(graph, collected.relocateStackId);
        auto nestedStrengthPrefix = StripLayerStackStrengthPrefix(
            collected.arcStrengthPrefix,
            LayerStackStrengthPrefix(graph, authoredStackId));

        for (size_t targetIndex = 0;
             targetIndex < targets.size();
             ++targetIndex) {
            const auto& target = targets[targetIndex];
            if (target.IsEmpty()) continue;
            uint32_t siblingOrder = static_cast<uint32_t>(targetIndex);

            bool foundAuthoredLayer = false;
            auto stackTargets = BuildPathArcLayerStackTargets(
                graph, authoredStackId, target);
            for (const auto& stackTarget : stackTargets) {
                size_t begin = 0, end = 0;
                if (!LayerStackLayerRange(
                        graph, stackTarget.stackId, begin, end)) {
                    continue;
                }

                PathMappingPtr authorMapping = stackTarget.authored ?
                    collected.arcAuthorMapping :
                    LayerStackImpliedPathArcMapping(graph, stackTarget.stackId);
                Path authorSourcePath = stackTarget.authored ?
                    collected.arcAuthorSourcePath :
                    authorMapping->MapToSource(primPath);
                if (authorSourcePath.IsEmpty()) {
                    authorSourcePath = primPath;
                }

                StrengthChain specializeStrengthPrefix =
                    stackTarget.authored ?
                    collected.arcStrengthPrefix :
                    ImpliedSpecializeStrengthPrefix(
                        graph, stackTarget.stackId, nestedStrengthPrefix);

                PathMappingPtr mapping = MakeArcPathMapping(
                    graph,
                    stackTarget.targetPath, primPath,
                    stackTarget.stackId,
                    authorSourcePath,
                    authorMapping,
                    /*fallbackIdentity=*/true);

                for (size_t li = begin; li < end; ++li) {
                    const Layer& layer = graph.GetLayer(li);
                    const Spec* targetSpec =
                        layer.GetSpec(stackTarget.targetPath);
                    if (!targetSpec ||
                        targetSpec->GetType() != SpecType::Prim) {
                        continue;
                    }
                    if (stackTarget.authored) foundAuthoredLayer = true;

                    OpinionEntry entry;
                    entry.layerIndex = li;
                    entry.pathMapping = mapping;
                    entry.retiming = graph.layerRetimings[li];
                    entry.arcType = ArcType::Specialize;
                    SetStrengthChain(
                        entry,
                        MakeSpecializeStrengthChain(
                            specializeStrengthPrefix, authorSourcePath,
                            /*implied=*/!stackTarget.authored, siblingOrder));
                    IndexDescendantPrims(layer, stackTarget.targetPath,
                                         primPath, entry, graph,
                                         ctx.sourceDescendantCache.get(), newPrims);
                    addedAnything = true;
                }
            }
            if (!foundAuthoredLayer && ctx.diagnostics) {
                ctx.diagnostics->Add({
                    DiagSeverity::Error,
                    DiagCategory::MissingSpecializeTarget,
                    "Specialize target has no specs in the layer stack: " +
                        target.GetText(),
                    primPath.GetText(),
                    LayerStackAnchorPath(graph, authoredStackId),
                    target.GetText(),
                    ArcType::Specialize});
            }
        }
    }
    return addedAnything;
}

constexpr int kMaxReferenceDepth = 32;

// Warm the ParseCache ahead of the sequential ResolveGraphReferences pass
// so ref/payload lookups hit the cache instead of blocking on per-file I/O.
//
// Wave 1 scans the graph's initial opinion entries for ref/payload asset
// paths. Subsequent waves scan each newly-parsed layer's prim specs (and
// sublayer metadata) for new targets. The process stops when a wave adds
// no new files. Parses within a wave run in parallel; the number of waves
// equals the longest reference chain from the root.
void PrefetchArcAssets(CompositionGraph& graph, ComposeContext& ctx) {
    if (!ctx.parseCache) return;

    std::unordered_set<std::string> processedLayers;
    std::unordered_set<std::string> pending;

    auto collectArcs = [&](const Spec& spec, const std::string& anchor) {
        auto grab = [&](const Token& fieldName) {
            const auto* f = spec.GetField(fieldName);
            if (!f) return;
            const auto* op = f->Get<ListOp<Reference>>();
            if (!op) return;
            auto addAll = [&](const std::vector<Reference>& v) {
                for (const auto& r : v) {
                    if (!r.assetPath || r.assetPath->empty()) continue;
                    std::string resolved = ResolveAssetCached(ctx, anchor, *r.assetPath);
                    if (!resolved.empty()) pending.insert(std::move(resolved));
                }
            };
            if (op->IsExplicit()) addAll(op->GetExplicitItems());
            else { addAll(op->GetPrependedItems()); addAll(op->GetAppendedItems()); }
        };
        grab(FieldNames::references);
        grab(FieldNames::payload);
    };

    // Seed the pending set from the initial graph's opinion entries.
    for (const auto& [primPath, primIdx] : graph.primIndices) {
        if (!primIdx.hasArcOpinions) continue;
        for (const auto& entry : primIdx.entries) {
            const Layer& layer = graph.GetLayer(entry.layerIndex);
            Path sourcePath = OpinionSourcePath(entry, primPath);
            if (sourcePath.IsEmpty()) continue;
            const auto* spec = layer.GetSpec(sourcePath);
            if (!spec || !SpecCanCarryArcs(*spec)) continue;
            collectArcs(*spec, graph.layerPaths[entry.layerIndex]);
        }
    }
    // Graph layers are already parsed — don't re-scan them in later waves.
    for (const auto& p : graph.layerPaths) processedLayers.insert(p);

    // Seed the queue from wave-1 asset paths discovered in the graph scan.
    std::deque<std::string> queue;
    for (auto& p : pending) {
        if (processedLayers.insert(p).second) queue.push_back(std::move(p));
    }
    pending.clear();
    if (queue.empty()) return;
    // A single initial target usually means a serial dependency chain. The
    // producer/consumer prefetcher cannot expose parallelism there, and it
    // duplicates the resolve pass's arc scan. Let resolve parse on demand.
    if (queue.size() == 1) return;

    // Cap default prefetch parallelism by hardware and by queued work. Beyond
    // roughly four workers, PathPool interning contention can erase the parse
    // parallelism on the benchmark matrix. Workloads with very large layer
    // counts can still override via NANOUSD_PREFETCH_THREADS=N.
    constexpr unsigned kDefaultMaxThreads = 4;
    unsigned hw = std::thread::hardware_concurrency();
    unsigned threadCount = std::min(hw ? hw : 1u, kDefaultMaxThreads);
    if (const char* p = std::getenv("NANOUSD_PREFETCH_THREADS"))
        threadCount = static_cast<unsigned>(std::max(1, std::atoi(p)));
    threadCount = std::min<unsigned>(
        threadCount, static_cast<unsigned>(queue.size()));

    // Producer/consumer model: workers draw the next path from a shared
    // queue, parse it, scan the parsed layer for its own arcs/sublayers,
    // and enqueue newly-discovered paths. This removes wave boundaries —
    // threads never block at a sync point waiting for the slowest file
    // in a wave to finish before starting the next batch.
    std::mutex qMutex;
    std::condition_variable qCond;
    int activeWorkers = 0;
    bool shuttingDown = false;
    std::atomic<int> totalPrefetched{0};

    auto worker = [&]() {
        for (;;) {
            std::string path;
            {
                std::unique_lock<std::mutex> lk(qMutex);
                qCond.wait(lk, [&] {
                    return shuttingDown || !queue.empty();
                });
                if (queue.empty()) return;  // shutdown signaled, nothing left
                path = std::move(queue.front());
                queue.pop_front();
                ++activeWorkers;
            }

            auto res = CachedParseUsdFile(path, ctx);
            std::vector<std::string> newTargets;

            if (res.success && res.layer) {
                // Collect arcs + sublayers into a local list; commit under
                // lock to keep shared set mutations brief.
                for (const Path& arcPath : detail::GetLayerArcOpinionPrimPaths(*res.layer)) {
                    const Spec* spec = res.layer->GetPrimSpec(arcPath);
                    if (!spec) continue;
                    auto grabField = [&](const Token& fieldName) {
                        const auto* f = spec->GetField(fieldName);
                        if (!f) return;
                        const auto* op = f->Get<ListOp<Reference>>();
                        if (!op) return;
                        auto addAll = [&](const std::vector<Reference>& v) {
                            for (const auto& r : v) {
                                if (!r.assetPath || r.assetPath->empty()) continue;
                                std::string resolved =
                                    ResolveAssetCached(ctx, path, *r.assetPath);
                                if (!resolved.empty())
                                    newTargets.push_back(std::move(resolved));
                            }
                        };
                        if (op->IsExplicit()) addAll(op->GetExplicitItems());
                        else {
                            addAll(op->GetPrependedItems());
                            addAll(op->GetAppendedItems());
                        }
                    };
                    grabField(FieldNames::references);
                    grabField(FieldNames::payload);
                }
                const auto* slField =
                    res.layer->GetLayerSpec().GetField(FieldNames::subLayers);
                if (slField) {
                    if (const auto* sl = slField->Get<SubLayerPaths>()) {
                        for (const auto& p : sl->paths) {
                            std::string resolved = ResolveAssetCached(ctx, path, p);
                            if (!resolved.empty())
                                newTargets.push_back(std::move(resolved));
                        }
                    }
                }
            }

            totalPrefetched.fetch_add(1, std::memory_order_relaxed);

            // Commit newly-discovered targets and wake a worker for each.
            {
                std::unique_lock<std::mutex> lk(qMutex);
                int added = 0;
                for (auto& t : newTargets) {
                    if (processedLayers.insert(t).second) {
                        queue.push_back(std::move(t));
                        ++added;
                    }
                }
                --activeWorkers;
                // If the queue is now empty and no worker is mid-parse,
                // nobody will ever enqueue more work — tell everyone to go.
                if (queue.empty() && activeWorkers == 0) {
                    shuttingDown = true;
                    qCond.notify_all();
                } else if (added > 0) {
                    // Wake up to `added` waiters to pick up the new work.
                    for (int i = 0; i < added; ++i) qCond.notify_one();
                }
            }
        }
    };

    std::vector<std::future<void>> workers;
    workers.reserve(threadCount);
    for (unsigned t = 0; t < threadCount; ++t)
        workers.push_back(std::async(std::launch::async, worker));
    for (auto& w : workers) w.wait();

    if (TimingEnabled())
        std::fprintf(stderr,
            "    [prefetch] files=%d (producer/consumer, threads=%u)\n",
            totalPrefetched.load(), threadCount);
}

void ResolveGraphReferences(CompositionGraph& graph,
                             const std::string& anchorPath,
                             const ComposeContext& ctx,
                             std::string& error) {
    g_stats.reset();
    ConsumedArcs consumed;

    // Work queue of (primPath, depth, startIdx). Seed with the initial prims
    // from the sublayer stack at depth 0, startIdx 0 (scan all entries).
    // Chained arcs append new entries to existing prim indices; those pushes
    // carry a startIdx that points past the old entries, so re-visits scan
    // only the newly-attached opinion instead of the full list each time.
    struct QueueEntry { Path primPath; int depth; size_t startIdx; };
    std::vector<QueueEntry> workQueue;
    const size_t seedWorkCount = graph.primIndices.size();
    workQueue.reserve(seedWorkCount + seedWorkCount / 2 + 1);
    graph.layers.reserve(graph.layers.size() + seedWorkCount + 1);
    graph.layerPaths.reserve(graph.layerPaths.size() + seedWorkCount + 1);
    graph.layerRetimings.reserve(graph.layerRetimings.size() + seedWorkCount + 1);
    graph.layerStackIds.reserve(graph.layerStackIds.size() + seedWorkCount + 1);
    graph.layerStacks.reserve(graph.layerStacks.size() + seedWorkCount + 1);
    for (const auto& [path, primIndex] : graph.primIndices) {
        workQueue.push_back({path, 0, 0});
    }
    // graph.primIndices is an unordered_map, so the iteration above yields
    // a run-to-run non-deterministic seed order. Composition results are
    // order-sensitive for equally-strong same-arc-type opinions (stable
    // LIVRPS sort preserves insertion order), and insertion order in turn
    // depends on visit order — if /A is visited before /A/B, opinions
    // reaching /A/B via /A's arcs land before /A/B's own arc opinions.
    // Sort the seed lexicographically so ancestors precede descendants and
    // every run composes the same graph.
    std::sort(workQueue.begin(), workQueue.end(),
              [](const QueueEntry& a, const QueueEntry& b) {
                  return a.primPath < b.primPath;
              });

    std::vector<QueuedArcWork> scratch;

    for (size_t i = 0; i < workQueue.size(); ++i) {
        // Copy by value — workQueue may be reallocated by subsequent push_backs.
        Path primPath = workQueue[i].primPath;
        int depth = workQueue[i].depth;
        size_t startIdx = workQueue[i].startIdx;

        if (depth >= kMaxReferenceDepth) continue;

        // Fast-skip prims that have no arc opinions at all. hasArcOpinions
        // is set during index build / IndexDescendantPrims when an entry's
        // source spec carries a references or payload field.
        auto pit = graph.primIndices.find(primPath);
        if (pit == graph.primIndices.end() || !pit->second.hasArcOpinions) continue;

        scratch.clear();
        g_stats.queue_iterations++;

        // Variants first (LIVRPS §10.5: Variants are stronger than References
        // and Payloads, and variant body opinions may themselves carry
        // references or payloads that the subsequent ResolvePrim* calls
        // need to see). Selection is composed strongest-first from this
        // prim's current opinions (already in LIVRPS order since Local
        // entries precede any arc-introduced entries at this point).
        VariantSelectionMap selections =
            ResolveVariantSelections(graph, &pit->second, primPath);
        ExpandVariantArcs(graph, primPath, startIdx, selections,
                          ctx, &scratch);

        // Inherits sit at LIVRPS strength 1 (between Local=0 and
        // Variants=2). Resolve before references/payloads so the
        // entries land in the right strength order before the final
        // stable-sort. See spec §10.5.1.
        ResolvePrimInherits(graph, primPath, ctx, startIdx, &scratch);

        ResolvePrimReferences(graph, primPath, anchorPath,
                              ctx, consumed, error, startIdx, &scratch);
        if (!error.empty()) return;

        // Payloads are weaker than references per LIVRPS, but that affects
        // opinion strength ordering (enforced by entry order), not traversal
        // order — it is safe to resolve both arc kinds for this prim here.
        ResolvePrimPayloads(graph, primPath, anchorPath,
                            ctx, consumed, error, startIdx, &scratch);
        if (!error.empty()) return;

        // Specializes is the weakest LIVRPS arc. Resolve LAST;
        // ResolvePrimSpecializes builds its global-weakest strength chain.
        ResolvePrimSpecializes(graph, primPath, ctx, startIdx, &scratch);

        int childDepth = depth + 1;
        for (const auto& np : scratch) {
            workQueue.push_back({np.primPath, childDepth, np.startIdx});
            g_stats.prims_queued++;
        }
    }

    // LIVRPS stable-sort pass (spec §10.5). During resolution, opinion
    // entries are appended in arrival order (Local first from the sublayer
    // stack, then Variant / Reference / Payload as arcs are expanded). A
    // single variant's content may bring in references that append
    // Reference-typed entries AFTER earlier arc entries, which could leave
    // Reference-before-Variant on a prim whose variant content was
    // processed mid-stream. Sort each prim's entries by LIVRPS strength so
    // query code can read strongest-first without caring about insertion
    // order. Stable so insertion order within an arc type (which
    // represents layer strength within the same arc) is preserved.
    for (auto& [primPath, idx] : graph.primIndices) {
        if (idx.entries.size() < 2) continue;
        std::stable_sort(idx.entries.begin(), idx.entries.end(),
            [](const OpinionEntry& a, const OpinionEntry& b) {
                return OpinionEntryStronger(a, b);
            });
    }

    if (TimingEnabled()) {
        auto ms = [](long long ns) { return ns / 1e6; };
        std::fprintf(stderr,
            "      [resolve] queue iterations: %d   ext-refs: %d   int-refs: %d   "
            "ext-payloads: %d   int-payloads: %d\n",
            g_stats.queue_iterations, g_stats.ext_refs, g_stats.int_refs,
            g_stats.ext_payloads, g_stats.int_payloads);
        std::fprintf(stderr,
            "      [resolve] index-descendants calls: %d   prims queued: %d\n",
            g_stats.index_descendant_calls, g_stats.prims_queued);
        std::fprintf(stderr,
            "      [resolve] entry scan (refs):       %9.1f ms\n", ms(g_stats.ns_entry_scan_refs));
        std::fprintf(stderr,
            "      [resolve] entry scan (payloads):   %9.1f ms\n", ms(g_stats.ns_entry_scan_payloads));
        std::fprintf(stderr,
            "      [resolve] resolver (asset lookup): %9.1f ms\n", ms(g_stats.ns_resolver_calls));
        std::fprintf(stderr,
            "      [resolve] CachedParseUsdFile:      %9.1f ms   (hits=%d misses=%d)\n",
            ms(g_stats.ns_cached_parse), g_stats.cache_hits, g_stats.cache_misses);
        std::fprintf(stderr,
            "      [resolve] GatherLayerStack:        %9.1f ms   (hits=%d misses=%d)\n",
            ms(g_stats.ns_gather_layer_stack), g_stats.gather_hits, g_stats.gather_misses);
        std::fprintf(stderr,
            "      [resolve] leaf layer-use reuse: created=%d hits=%d\n",
            g_stats.leaf_layer_reuse_created, g_stats.leaf_layer_reuse_hits);
        std::fprintf(stderr,
            "      [resolve] IndexDescendantPrims:    %9.1f ms\n", ms(g_stats.ns_index_descendants));
        std::fprintf(stderr,
            "      [resolve] ResolvePrimReferences:   %9.1f ms (total, including above)\n",
            ms(g_stats.ns_resolve_refs_total));
        std::fprintf(stderr,
            "      [resolve] ResolvePrimPayloads:     %9.1f ms (total, including above)\n",
            ms(g_stats.ns_resolve_payloads_total));
    }
}

// ============================================================
// Build the resolved layer spec used for stage metadata.
// ============================================================

void BuildComposedLayerSpec(CompositionGraph& graph) {
    if (graph.GetNumLayers() == 0) {
        graph.composedLayerSpec = Spec(SpecType::Layer);
        return;
    }

    // AOUSD layer metadata resolves from the root layer spec only. Sublayers
    // and reference/payload layers do not provide weaker metadata opinions.
    graph.composedLayerSpec = graph.GetLayer(0).GetLayerSpec();

    // Composition fields are consumed by the composition engine and are not
    // exposed as resolved stage metadata.
    graph.composedLayerSpec.RemoveField(FieldNames::subLayers);
    graph.composedLayerSpec.RemoveField(FieldNames::subLayerOffsets);
}

} // anonymous namespace

// ============================================================
// Public API
// ============================================================

// ============================================================
// ComposePrimIndex (phase 2) — from-scratch per-prim PrimIndex builder
// ============================================================
//
// Builds a PrimIndex for `primPath` by walking the graph's layers and
// finding Prim specs at primPath. Phase 2 v1 only handles the sublayer-
// stack opinion case: layers in [0, arcOriginLayerCount) with identity
// path mapping. Arc-derived entries (from references/payloads bringing
// in target-layer opinions) are out of scope here — phase 4 will extend
// this to cover arc resolution per-prim. Until then, the equivalence
// check below skips arc-touched prims.
//
// This is the per-prim accessor that phase 4 will swap eager
// `BuildPrimIndices` for; everything compose does today must eventually
// be expressible as ComposePrimIndex(path) called lazily.

PrimIndex ComposePrimIndex(const CompositionGraph& graph, const Path& primPath) {
    PrimIndex result;
    for (size_t li = 0; li < graph.arcOriginLayerCount; ++li) {
        if (IsRelocateSourcePath(
                graph, LayerStackIdForLayer(graph, li), primPath))
            continue;
        const Layer& layer = graph.GetLayer(li);
        const Spec* spec = layer.GetSpec(primPath);
        if (!spec || spec->GetType() != SpecType::Prim) continue;

        OpinionEntry entry;
        entry.layerIndex = li;
        // pathMapping defaults to the shared identity singleton.
        entry.sourcePath = primPath;
        entry.retiming = graph.layerRetimings[li];
        entry.arcType = ArcType::Local;
        result.entries.push_back(std::move(entry));
        if (SpecHasArcOpinion(*spec)) result.hasArcOpinions = true;
    }
    return result;
}

// True iff every entry in `idx` is a sublayer-stack Local opinion with
// identity mapping — i.e. no arc resolution touched this prim. These are
// the prims phase 2 v1 can validate against ComposePrimIndex.
static bool AllEntriesAreSublayerStackLocal(const PrimIndex& idx,
                                              size_t arcOriginLayerCount) {
    for (const auto& e : idx.entries) {
        if (e.layerIndex >= arcOriginLayerCount) return false;
        if (e.arcType != ArcType::Local) return false;
        if (!e.pathMapping || !e.pathMapping->isIdentity) return false;
    }
    return true;
}

// Compare two PrimIndex objects for equivalence. Used by the validation
// pass below. We compare: entry count, per-entry layerIndex/arcType/
// retiming offset+scale, identity-mapping flag, and hasArcOpinions.
// Returns empty string on match, otherwise a short reason.
static std::string PrimIndexDiffReason(const PrimIndex& a, const PrimIndex& b) {
    if (a.hasArcOpinions != b.hasArcOpinions)
        return "hasArcOpinions mismatch";
    if (a.entries.size() != b.entries.size())
        return "entry count " + std::to_string(a.entries.size())
             + " vs " + std::to_string(b.entries.size());
    for (size_t i = 0; i < a.entries.size(); ++i) {
        const auto& ae = a.entries[i];
        const auto& be = b.entries[i];
        if (ae.layerIndex != be.layerIndex) return "layerIndex differs at entry " + std::to_string(i);
        if (ae.arcType    != be.arcType)    return "arcType differs at entry "    + std::to_string(i);
        if (ae.retiming.offset != be.retiming.offset
            || ae.retiming.scale != be.retiming.scale)
            return "retiming differs at entry " + std::to_string(i);
        bool aId = !ae.pathMapping || ae.pathMapping->isIdentity;
        bool bId = !be.pathMapping || be.pathMapping->isIdentity;
        if (aId != bId) return "pathMapping identity differs at entry " + std::to_string(i);
    }
    return {};
}

void PrimIndexEquivalenceCheck(const CompositionGraph& graph) {
    static const bool enabled = [] {
        const char* v = std::getenv("NANOUSD_PRIMINDEX_EQUIV");
        return v && v[0] == '1';
    }();
    if (!enabled) return;

    size_t total = 0, validated = 0, skipped = 0, mismatches = 0;
    std::vector<std::string> sample;  // up to 5 mismatch details for diagnostics
    for (const auto& [path, eagerIdx] : graph.primIndices) {
        ++total;
        if (!AllEntriesAreSublayerStackLocal(eagerIdx, graph.arcOriginLayerCount)) {
            ++skipped;
            continue;
        }
        PrimIndex lazyIdx = ComposePrimIndex(graph, path);
        std::string reason = PrimIndexDiffReason(eagerIdx, lazyIdx);
        if (!reason.empty()) {
            ++mismatches;
            if (sample.size() < 5) {
                sample.push_back(path.GetText() + ": " + reason);
            }
        } else {
            ++validated;
        }
    }
    std::fprintf(stderr,
        "[primindex-equiv] total=%zu validated=%zu skipped=%zu mismatches=%zu\n",
        total, validated, skipped, mismatches);
    for (const auto& s : sample) {
        std::fprintf(stderr, "[primindex-equiv]   %s\n", s.c_str());
    }
}

namespace {

ComposeResult ComposeWithRootLayer(std::shared_ptr<Layer> rootLayerPtr,
                                   const std::string& anchorPath,
                                   AssetResolver resolver) {
    ComposeResult result;
    ComposeContext ctx;
    if (!resolver) resolver = DefaultResolve;
    ctx.resolver = std::move(resolver);
    ctx.parseCache = std::make_shared<ParseCache>();
    ctx.gatherCache = std::make_shared<GatherCache>();
    ctx.resolverCache = std::make_shared<ResolverCache>();
    ctx.instanceArcCache = std::make_shared<InstanceArcRepresentativeCache>();
    ctx.sourceDescendantCache = std::make_shared<SourceDescendantCache>();
    ctx.reusableLeafLayerCache = std::make_shared<ReusableLeafLayerCache>();
    ctx.diagnostics = &result.diagnostics;

    ComposePhaseTimer timer;
    const std::string rootAnchorPath = LayerSourcePathForAsset(anchorPath);

    // Phase 1: Gather sublayer stack
    Retiming identity{0.0, 1.0};
    auto gathered = GatherLayerStack(rootLayerPtr, rootAnchorPath, identity, ctx);
    if (!gathered.success) {
        // Fatal error (cycle)
        result.error = gathered.error;
        return result;
    }
    // Merge non-fatal diagnostics from sublayer gathering
    result.diagnostics.Merge(std::move(gathered.diagnostics));
    timer.lap("gather sublayer stack");

    // Build the graph from gathered root layer-stack layers
    size_t rootStackId = StartLayerStack(result.graph);
    size_t rootStackBegin = result.graph.GetNumLayers();
    for (auto& gl : gathered.layers) {
        result.graph.layers.push_back(std::move(gl.layer));
        result.graph.layerPaths.push_back(std::move(gl.sourcePath));
        result.graph.layerRetimings.push_back(gl.retiming);
        result.graph.layerStackIds.push_back(rootStackId);
    }
    FinishLayerStack(result.graph, rootStackId, rootStackBegin,
                     result.graph.GetNumLayers(), ctx.diagnostics);

    // Record where sublayer stack ends (for instancing key computation)
    result.graph.arcOriginLayerCount = result.graph.GetNumLayers();

    // Build prim indices
    BuildPrimIndicesWithRetiming(result.graph);
    timer.lap("build prim indices");

    // Build resolved layer metadata
    BuildComposedLayerSpec(result.graph);
    timer.lap("build layer metadata");

    // Prefetch arc assets in parallel: warms ParseCache so the sequential
    // resolve pass hits the cache instead of blocking on per-file parse I/O.
    PrefetchArcAssets(result.graph, ctx);
    timer.lap("prefetch arc assets");

    // Phase 2: Resolve references (non-fatal issues go to ctx.diagnostics)
    std::string error;
    ResolveGraphReferences(result.graph, rootAnchorPath, ctx, error);
    if (!error.empty()) {
        // Fatal error (cycle in reference chain)
        result.error = error;
        return result;
    }
    timer.lap("resolve references/payloads");

    result.graph.resolver = ctx.resolver;
    result.graph.BuildChildIndex();
    timer.lap("build child index");

    // Phase 2 of the lazy-PrimIndex work: validate that a from-scratch
    // ComposePrimIndex(path) reproduces the eager primIndices entries for
    // the trivial subset (sublayer-stack-only opinions, identity mapping,
    // Local arcType). Arc-derived entries are out of scope for v1 — those
    // prims are skipped. Gated on NANOUSD_PRIMINDEX_EQUIV=1 and printed to
    // stderr so benchmark runs can validate without paying the cost on
    // normal opens.
    PrimIndexEquivalenceCheck(result.graph);

    result.success = true;
    return result;
}

} // anonymous namespace

ComposeResult Compose(const Layer& rootLayer,
                      const std::string& anchorPath,
                      AssetResolver resolver) {
    return ComposeWithRootLayer(std::make_shared<Layer>(rootLayer),
                                anchorPath, std::move(resolver));
}

} // namespace nanousd
