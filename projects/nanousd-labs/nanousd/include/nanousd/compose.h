// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "arc.h"
#include "asset_resolver.h"
#include "diagnostic.h"
#include "layer.h"
#include "schema.h"

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nanousd {

class ClipLayerCache;

// ============================================================
// Composition graph types (spec Section 10.2)
// ============================================================

// Maps a prim path from one namespace to another (e.g., for references).
struct PathMappingEntry {
    Path sourcePrimPath;
    // Empty target means the source subtree is removed by relocates.
    Path targetPrimPath;
};

struct PathMapping {
    Path sourcePrimPath;   // e.g., /Template
    Path targetPrimPath;   // e.g., /MyPrim
    std::vector<PathMappingEntry> extraMappings;
    // True only when this mapping preserves every path unchanged.
    bool isIdentity = true;
    // Some composition arcs carry an additional identity mapping [(/,/)]:
    // internal references/payloads, inherits/specializes, and variants.
    // Paths outside sourcePrimPath map to themselves for those arcs.
    bool fallbackIdentity = false;

    // Map a target-namespace path back to source-layer space.
    Path MapToSource(const Path& targetPath) const;

    // Map a source-layer path into target namespace.
    Path MapToTarget(const Path& sourcePath) const;

    // True when sourcePath is affected by an extra relocate mapping.
    bool UsesExtraSourceMapping(const Path& sourcePath) const;
};

// Shared, immutable PathMapping handle. OpinionEntry holds a shared_ptr
// because one arc produces the SAME mapping for every descendant of its
// target; sharing turns per-descendant Path copies into a refcount bump.
// Always non-null — default-constructed OpinionEntries reference a process-
// wide identity singleton (see IdentityPathMappingPtr()).
using PathMappingPtr = std::shared_ptr<const PathMapping>;

// Returns a shared, immutable default-constructed PathMapping
// (isIdentity=true, empty source/target). Used for identity opinions that
// don't need their own allocation.
const PathMappingPtr& IdentityPathMappingPtr();

// Build a fresh non-identity PathMapping on the heap.
PathMappingPtr MakePathMapping(
    Path sourcePrimPath,
    Path targetPrimPath,
    bool fallbackIdentity = false,
    std::vector<PathMappingEntry> extraMappings = {});

// One layer's opinion for a given prim.
struct OpinionEntry {
    size_t layerIndex;             // index into CompositionGraph::layers
    PathMappingPtr pathMapping;    // shared; default-constructed = identity
    Path sourcePath;               // cached source prim path for this target, if known
    Retiming retiming;             // accumulated offset/scale (non-destructive)
    ArcType arcType = ArcType::Local;
    struct StrengthComponent {
        ArcType arcType = ArcType::Local;
        uint32_t namespaceDepth = 0;
        bool implied = false;
        uint32_t siblingOrder = 0;

        StrengthComponent() = default;
        StrengthComponent(ArcType t) : arcType(t) {}
        bool operator==(const StrengthComponent& o) const {
            return arcType == o.arcType &&
                   namespaceDepth == o.namespaceDepth &&
                   implied == o.implied &&
                   siblingOrder == o.siblingOrder;
        }
    };
    std::shared_ptr<const std::vector<StrengthComponent>> strengthChain;

    OpinionEntry()
        : layerIndex(0)
        , pathMapping(IdentityPathMappingPtr()) {}
};

// Strength-ordered opinion stack for a single prim.
struct PrimIndex {
    std::vector<OpinionEntry> entries;  // strongest first
    // Set true when at least one source spec for an entry carries a
    // `references` or `payload` field. Lets ResolveGraphReferences skip
    // the entry-scan loop on prims that have no arc opinions at all.
    bool hasArcOpinions = false;
    // Set true when at least one entry can contribute to the instancing key
    // for this prim. Most descendants inherit an ancestor arc entry; that
    // entry is needed for value resolution but MakeInstancingKey will ignore
    // it because the arc target is above the descendant.
    bool hasInstancingKeyEntries = false;
};

struct LayerStackRecord {
    size_t beginLayerIndex = 0;
    size_t endLayerIndex = 0;
    size_t parentStackId = static_cast<size_t>(-1);
    // Maps paths in this layer stack to the composed/root namespace for
    // implied inherit target propagation. Unlike ordinary external
    // reference/payload mappings, this preserves paths outside the arc
    // source subtree via fallback identity, matching inherit namespace
    // mapping rules.
    PathMappingPtr impliedInheritMapping = IdentityPathMappingPtr();
    // Strength prefix for local opinions in this layer stack. Root stack is
    // empty; a stack introduced by a reference from root is [Reference], etc.
    std::vector<OpinionEntry::StrengthComponent> strengthPrefix;
    std::vector<Relocate> relocates;
};

// The retained composition graph.
struct CompositionGraph {
    std::vector<std::shared_ptr<Layer>> layers;    // layer[0] = root (strongest), COW shared
    std::vector<std::string> layerPaths;           // parallel: source file path per layer
    std::vector<Retiming> layerRetimings;          // parallel: accumulated offset/scale per layer
                                                    // (identity for the root sublayer stack;
                                                    // retiming applies as arcs bring in new layers)
    std::vector<size_t> layerStackIds;              // parallel: owning layer stack per layer
    std::vector<LayerStackRecord> layerStacks;      // gathered layer stacks and their relocates
    // Repeated instance roots whose descendant expansion was skipped can
    // reuse a fully-expanded representative instance root when prototypes
    // are built during stage population. Rebuilt by Compose(), so Recompose()
    // naturally drops stale mappings.
    PathMap<Path> instanceRepresentatives;
    // Per-prim opinion stacks. Phase 4 made this a sparse cache: the
    // eager build only seeds entries for arc-touched paths (prims with
    // arc opinions, plus descendants brought in by IndexDescendantPrims
    // during arc resolution). Other paths populate on demand via
    // GetPrimIndex's lazy fallback. `mutable` because the const
    // accessor inserts on miss.
    mutable PathMap<PrimIndex> primIndices;
    Spec composedLayerSpec{SpecType::Layer};        // resolved root layer metadata
    size_t arcOriginLayerCount = 0;  // layers[0..N-1] are sublayer stack; N+ are from refs/payloads
    AssetResolver resolver;                        // retained for value resolution after composition

    // Shared across value resolution queries per spec §12.3.4. Lazy-
    // initialized on first clip-contributing attribute Get(); mutable so
    // const queries can populate it. shared_ptr so CompositionGraph
    // remains copyable without slicing the cache.
    mutable std::shared_ptr<ClipLayerCache> clipLayerCache;

    // --- Layer access ---
    // COW: GetMutableLayer detaches shared layers on first write
    Layer& GetRootLayer() {
        return GetMutableLayer(0);
    }
    const Layer& GetRootLayer() const { return *layers[0]; }
    Layer& GetMutableLayer(size_t index) {
        auto& sp = layers[index];
        if (sp.use_count() > 1)  // shared with cache or other graph
            sp = std::make_shared<Layer>(*sp);  // COW detach
        return *sp;
    }
    const Layer& GetLayer(size_t index) const { return *layers[index]; }
    size_t GetNumLayers() const              { return layers.size(); }

    // Find layer by source file path (returns nullptr if not found)
    Layer* FindLayer(const std::string& sourcePath);
    const Layer* FindLayer(const std::string& sourcePath) const;

    NANOUSD_CORE_API const PrimIndex* GetPrimIndex(const Path& path) const;

    Path GetInstanceRepresentative(const Path& path) const {
        auto it = instanceRepresentatives.find(path);
        return it == instanceRepresentatives.end() ? Path() : it->second;
    }

    // Mutating accessors. All writes to the prim-index storage should go
    // through one of these so phase 4's lazy switchover has a single set
    // of hooks: today they mutate `primIndices` directly; lazy will
    // invalidate the cache and let the next read recompose.
    bool HasPrimIndex(const Path& path) const;
    PrimIndex& EnsurePrimIndex(const Path& path);    // existing-or-default
    void SetPrimIndex(const Path& path, PrimIndex pi); // overwrite

    // --- Composed child index ---

    // Build parent→child-name index from primIndices (call after composition).
    void BuildChildIndex();

    // Incrementally add a path to the child index.
    void AddToChildIndex(const Path& path);

    size_t GetComposedChildEntryCount() const { return composedChildEntryCount_; }

    // O(1) lookup of direct child prim names under a composed parent path.
    const std::vector<Token>& GetComposedChildNames(const Path& parentPath) const;

    // O(1) lookup of direct child prim paths under a composed parent path.
    const std::vector<Path>& GetComposedChildPaths(const Path& parentPath) const;

    // O(1) lookup of direct child prim indices under a composed parent path.
    // Entries can be null for lazy-eligible children; callers must fall back
    // to GetPrimIndex in that case.
    const std::vector<const PrimIndex*>& GetComposedChildPrimIndices(
        const Path& parentPath) const;

    struct ComposedChildView {
        const std::vector<Token>* names = nullptr;
        const std::vector<Path>* paths = nullptr;
        const std::vector<const PrimIndex*>* primIndices = nullptr;
        const std::vector<std::uint8_t>* childHasChildren = nullptr;
    };
    ComposedChildView GetComposedChildren(const Path& parentPath) const;

    // --- PrimDefinition cache (thread-safe) ---

    // Returns cached PrimDefinition or builds and caches one.
    const PrimDefinition& GetOrBuildPrimDef(
        const std::string& primPath,
        const std::string& typeName,
        const std::vector<std::string>& apiSchemas) const;

    // Invalidate cached PrimDefinition for a given prim path.
    void InvalidatePrimDefCache(const std::string& primPath) const;

private:
    struct ComposedChildren {
        std::vector<Token> names;
        std::vector<Path> paths;
        std::vector<const PrimIndex*> primIndices;
        std::vector<std::uint8_t> childHasChildren;
    };

    struct PrimDefCache {
        std::shared_mutex mutex;
        std::unordered_map<std::string, PrimDefinition> entries;
        uint64_t schemaGeneration = 0;
    };
    // unique_ptr so CompositionGraph remains copyable/movable
    // (shared_mutex is neither). Copy/move starts with a fresh cache.
    mutable std::unique_ptr<PrimDefCache> primDefCache_ =
        std::make_unique<PrimDefCache>();

    // Composed parent path → child prim names and paths (built from
    // primIndices). Paths let hot traversal/populate loops avoid rebuilding
    // the same child paths via AppendChild.
    PathMap<ComposedChildren> composedChildren_;
    size_t composedChildEntryCount_ = 0;
};

// Build a PrimIndex for `primPath` from scratch by walking the graph's
// layer stack + arc graph. The lazy compose path: GetPrimIndex's cache
// miss calls this, the result is cached for subsequent queries. v1
// covers sublayer-stack Local opinions (identity mapping); arc-derived
// expansion is the next slice and lives in the eager seed for now.
NANOUSD_CORE_API PrimIndex ComposePrimIndex(const CompositionGraph& graph,
                                             const Path& primPath);

// Result of composition
struct ComposeResult {
    bool success = false;
    std::string error;              // Non-empty only for fatal errors (cycles, root parse)
    CompositionGraph graph;
    DiagnosticCollector diagnostics; // Non-fatal issues collected during composition
};

// Compose a root layer, resolving sublayers and references.
// anchorPath: filesystem path of the root layer (for relative asset resolution)
// resolver: optional custom asset resolver; if null, uses default filesystem resolution
NANOUSD_CORE_API ComposeResult Compose(const Layer& rootLayer,
                      const std::string& anchorPath,
                      AssetResolver resolver = nullptr);

} // namespace nanousd
