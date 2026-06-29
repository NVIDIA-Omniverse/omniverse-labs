// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"
#include "asset_resolver.h"
#include "path.h"
#include "value.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nanousd { class Layer; class Spec; }

namespace nanousd {

// Materialized clip set — the result of taking a raw `clips` dict
// entry (as authored in layer metadata per spec §12.3.4) and
// resolving its fields into typed, ready-to-consume values. Asset
// paths are resolved against the anchor layer; a template-form clip
// set is expanded into the explicit assetPaths / active / times
// arrays per §12.3.4.1.3 during materialization so downstream value
// resolution only has to understand one shape.
//
// This is an intermediate representation — consumers of spec §12.3.4
// value resolution (UsdAttribute::Get) will look up entries of this
// form during time-based queries. Phase 2 of the clips work lands
// this struct and the materialization helpers; Phase 3+ wires them
// into attribute resolution.
struct NANOUSD_CORE_API ClipSetEntry {
    std::string name;  // clip set key as it appeared in `clips`

    // --- Common metadata (§12.3.4.1.1) ---

    // Path prefix substituted for the stage prim's path when looking
    // up attribute values in clip layers. Per spec: "if clip metadata
    // is authored on prim /Prim_1, and primPath is /Prim, the
    // attribute /Prim_1.size would resolve to /Prim.size."
    Path primPath;

    // Resolved asset path to the manifest layer that declares which
    // attributes the clip set provides. Empty when no manifestAssetPath
    // was authored.
    std::string manifestAssetPath;

    // Spec §12.3.4.7. Default false per §13.2.1 fallback conventions.
    bool interpolateMissingClipValues = false;

    // --- Explicit sequence (§12.3.4.1.2 / §12.3.4.1.3) ---
    // Populated directly from authored `assetPaths` / `active` / `times`
    // on the explicit form, or synthesised from template metadata on
    // the template form. Downstream value resolution sees only the
    // explicit shape.

    // Resolved asset paths, in order. Indexed by `active` entries'
    // assetIndex.
    std::vector<std::string> assetPaths;

    // (stageTime, assetIndex) pairs defining when each clip is active.
    // Spec stores as double2[]; assetIndex is conceptually an integer
    // and is stored as such here after materialization.
    std::vector<std::pair<double, int>> active;

    // (stageTime, clipTime) pairs defining the timing curve. Spec
    // stores as double2[].
    std::vector<std::pair<double, double>> times;

    // --- Provenance ---
    // The prim path the clip set was authored on (the "anchor point"
    // per §12.3.4.5) and the composition-graph layer index of the
    // authoring layer. Used by strength-ordering logic in a later PR.
    Path anchorPrimPath;
    size_t anchorLayerIndex = ~size_t(0);
};

// Read a raw `clips` dict entry keyed by `clipSetName` and build a
// ClipSetEntry. `clipsDict` is the composed `clips` metadata
// (typically from UsdPrim::GetPrimMetadata). `anchorLayerPath` is the
// authoring layer's source path — used to resolve relative asset
// paths via `resolver`.
//
// When the clip set authors BOTH explicit (`assetPaths`) and template
// (`templateAssetPath`) metadata, explicit wins per spec §12.3.4.1
// ("When both explicit and template clip metadata is authored, explicit
// will be chosen."). Template expansion only runs when `assetPaths`
// is absent.
//
// Returns std::nullopt when `clipSetName` is not in `clipsDict` or
// when the entry is malformed in a way that makes it unusable (e.g.
// template path with no `#` placeholder groups).
NANOUSD_CORE_API std::optional<ClipSetEntry> MaterializeClipSet(
        const Dictionary& clipsDict,
        const std::string& clipSetName,
        const std::string& anchorLayerPath,
        const AssetResolver& resolver);

// Populate the explicit-sequence fields of `out` (assetPaths, active,
// times) from template metadata per spec §12.3.4.1.3.
//
// `templateAssetPath` is a regex-style path string with either one or
// two `#` groups:
//   "path/clipname.###.usd"       — integer stage times only
//   "path/clipname.###.###.usd"   — integer + sub-integer times
// Two groups must be adjacent separated by a single `.`; the number
// of `#` characters in each group is the zero-padding width.
//
// Returns false when the template path is malformed or when
// `activeOffset` exceeds `abs(stride)` per §12.3.4.1.3.5.
NANOUSD_CORE_API bool ExpandTemplateClipSet(
        const std::string& templateAssetPath,
        double startTime,
        double endTime,
        double stride,
        std::optional<double> activeOffset,
        const std::string& anchorLayerPath,
        const AssetResolver& resolver,
        ClipSetEntry& out);

// Cache of opened clip layers, keyed by resolved asset path.
// Lazy-opens each layer on first access via ParseUsdFile and keeps
// it alive for the cache's lifetime so subsequent lookups are O(1).
//
// Per spec §12.3.4, a single stage can reference hundreds of clip
// layers (one per clip); and each clip's asset path may be shared
// across clip sets. Caching by resolved path avoids reparsing the
// same file. Failed opens are negatively cached so we don't retry
// I/O on a known-missing file.
//
// Phase 3 scope: single-threaded. A mutex can be added if/when
// clips participate in multi-threaded value resolution.
class NANOUSD_CORE_API ClipLayerCache {
public:
    using PathSet = ::nanousd::PathSet;

    ClipLayerCache() = default;

    // Retrieve the layer for `resolvedPath`, opening it on first
    // access. Returns nullptr on open failure; failures are cached
    // so subsequent calls with the same path also return nullptr
    // without touching the filesystem.
    std::shared_ptr<const Layer> GetOrOpen(const std::string& resolvedPath);

    // Resolve the set of attribute paths the given clip set provides
    // values for (§12.3.4.3). When `entry.manifestAssetPath` is
    // authored, reads that layer. When empty, falls back to the
    // auto-manifest (§12.3.4.3: "A manifest may be automatically
    // generated by reading all clip layers"), unioning attribute
    // specs from every clip in `entry.assetPaths`.
    //
    // Result is cached by manifest identity — authored paths use the
    // path itself; auto-manifests are keyed by the concatenation of
    // the clip asset paths, so two clip sets with the same clip
    // layers share a cached manifest.
    const PathSet& GetOrResolveManifest(const ClipSetEntry& entry);

    // Diagnostic accessors.
    bool Contains(const std::string& resolvedPath) const;
    size_t Size() const { return entries_.size(); }
    void Clear() {
        entries_.clear();
        manifestCache_.clear();
    }

private:
    struct Entry {
        std::shared_ptr<const Layer> layer;  // nullptr on failure
    };
    std::unordered_map<std::string, Entry> entries_;

    // Manifest attribute sets, keyed by authored manifest path or a
    // synthetic key built from clip asset paths for auto-manifests.
    std::unordered_map<std::string, PathSet> manifestCache_;
};

// Open the manifest layer declared by `entry.manifestAssetPath` and
// return the paths of every attribute spec it contains per §12.3.4.3
// ("A manifest... declares which attributes in the clip layers the
// clips dict provides values for. A manifest is a SdfLayer containing
// only attribute specs...").
//
// Returns an empty vector when:
//  - `entry.manifestAssetPath` is empty (no manifest authored) — use
//    `ClipLayerCache::GetOrResolveManifest` for the auto-manifest
//    fallback (§12.3.4.3)
//  - the manifest layer fails to open
//  - the manifest contains no attribute specs
NANOUSD_CORE_API std::vector<Path> GetManifestAttributePaths(
        const ClipSetEntry& entry,
        ClipLayerCache& cache);

// Read a single layer's locally-authored `clips` dict + `clipSets`
// listop (NOT composed across the opinion stack) and materialize the
// clip sets in `clipSets` order (or dict-sorted order when `clipSets`
// is absent). Used by per-opinion attribute value resolution per
// spec §12.3.4.5 ("clip data is just weaker than Local").
//
// `layerPath` anchors asset path resolution for this layer's clip
// entries. `layerIndex` is stamped onto each materialized entry's
// `anchorLayerIndex` for provenance.
NANOUSD_CORE_API std::vector<ClipSetEntry> MaterializeClipSetsForLayer(
        const Spec& primSpec,
        const std::string& layerPath,
        size_t layerIndex,
        const AssetResolver& resolver);

// Evaluate a materialized clip set at `stageTime` for the attribute
// `attrPathOnStage`. Returns the value sampled from the clip layer
// active at that time, or nullopt when:
//   - the manifest (authored or auto-derived) does not declare this
//     attribute for this clip set — the clip set is skipped entirely
//     (§12.3.4.3)
//   - no asset is active at `stageTime` (empty active + assetPaths)
//   - the active clip layer fails to open via `cache`
//   - the clip declares the attribute but has no value at the
//     mapped clip-time, AND neither `interpolateMissingClipValues`
//     nor the hold-nearest fallback can recover a value from
//     neighbouring clips (§12.3.4.6, §12.3.4.7)
//
// Per §12.3.4.5 the stage attribute path is rewritten to sit under
// `entry.primPath` before lookup: e.g. if clips are authored on
// /Prim_1 with primPath=/Prim, then /Prim_1.size resolves to
// /Prim.size in the clip layer. `stagePrimPath` is the stage prim
// the clips are authored on (used to compute the relative prefix
// against `attrPathOnStage`).
NANOUSD_CORE_API std::optional<Value> EvaluateClipSetAtTime(
        const ClipSetEntry& entry,
        const Path& attrPathOnStage,
        const Path& stagePrimPath,
        double stageTime,
        ClipLayerCache& cache);

} // namespace nanousd
