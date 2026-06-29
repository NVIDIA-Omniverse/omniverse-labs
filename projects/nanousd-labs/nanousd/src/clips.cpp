// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/clips.h"

#include "nanousd/layer.h"
#include "nanousd/spec.h"
#include "nanousd/stage.h"      // ResolveTimeSample
#include "nanousd/usd_parser.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>

namespace nanousd {

namespace {

// Extract a String-or-Token value from a dict entry into a std::string.
// Spec doesn't rigidly pin whether `primPath`, `manifestAssetPath`
// etc. are authored as `string` or `token` — accept either.
bool ReadStringLike(const Value& v, std::string& out) {
    if (auto* s = v.Get<String>())      { out = *s;             return true; }
    if (auto* t = v.Get<Token>())       { out = t->GetString(); return true; }
    if (v.GetTypeId() == TypeId::Asset) {
        if (auto* s = v.Get<String>())  { out = *s;             return true; }
    }
    return false;
}

bool ReadBool(const Value& v, bool& out) {
    if (auto* b = v.Get<Bool>()) { out = *b; return true; }
    if (auto* i = v.Get<Int>())  { out = (*i != 0); return true; }
    return false;
}

bool ReadDouble(const Value& v, double& out) {
    if (auto* d = v.Get<Double>()) { out = *d;                      return true; }
    if (auto* f = v.Get<Float>())  { out = static_cast<double>(*f); return true; }
    if (auto* i = v.Get<Int>())    { out = static_cast<double>(*i); return true; }
    return false;
}

// Helper: look up `key` in `dict`, return pointer if present or
// nullptr otherwise.
const Value* Find(const Dictionary& dict, const char* key) {
    auto it = dict.find(key);
    return (it != dict.end()) ? &it->second : nullptr;
}

// Read a double2[] array into (stageTime, clipTime) pairs. The spec
// types `active` and `times` as double2[], stored as vector<GfVec2d>.
bool ReadDouble2Array(const Value& v, std::vector<std::pair<double, double>>& out) {
    auto* arr = v.Get<std::vector<GfVec2d>>();
    if (!arr) return false;
    out.clear();
    out.reserve(arr->size());
    for (const auto& p : *arr) {
        out.emplace_back(p[0], p[1]);
    }
    return true;
}

// Convert (stageTime, assetIndex) double-pair array — second element
// rounds to int per spec ("assetIndex is a non-negative integer").
bool ReadActivePairs(const Value& v, std::vector<std::pair<double, int>>& out) {
    std::vector<std::pair<double, double>> raw;
    if (!ReadDouble2Array(v, raw)) return false;
    out.clear();
    out.reserve(raw.size());
    for (const auto& p : raw) {
        int idx = static_cast<int>(std::llround(p.second));
        if (idx < 0) idx = 0;
        out.emplace_back(p.first, idx);
    }
    return true;
}

// Read an asset[] or string[] array into resolved asset paths.
// Per spec §5 (asset path resolution) each element is resolved against
// the anchor layer via the provided resolver.
std::vector<std::string> ReadResolvedAssetPaths(const Value& v,
        const std::string& anchorLayerPath,
        const AssetResolver& resolver) {
    std::vector<std::string> out;
    auto tryAsset = v.Get<std::vector<std::string>>();
    if (!tryAsset) return out;
    out.reserve(tryAsset->size());
    for (const auto& raw : *tryAsset) {
        if (resolver) {
            auto resolved = resolver(anchorLayerPath, raw);
            out.push_back(resolved.empty() ? raw : resolved);
        } else {
            out.push_back(raw);
        }
    }
    return out;
}

// --- Template-path parsing --------------------------------------------------
//
// Spec §12.3.4.1.3.1: a templateAssetPath is either `path/name.###.usd`
// (integer form) or `path/name.###.###.usd` (integer + sub-integer
// form). Exactly one or two `#` groups; if two, separated by a single
// `.`. Each `#` in a group is one digit of zero-padding.

struct TemplateParse {
    std::string prefix;     // text before first `#`
    int intPad = 0;         // width of first `#` group
    int fracPad = 0;        // width of second `#` group (0 if integer-only form)
    std::string suffix;     // text after last `#`
    bool subInteger = false;
};

bool ParseTemplatePath(const std::string& tpl, TemplateParse& out) {
    size_t first = tpl.find('#');
    if (first == std::string::npos) return false;

    out.prefix = tpl.substr(0, first);
    size_t i = first;
    while (i < tpl.size() && tpl[i] == '#') { ++out.intPad; ++i; }

    // Integer form: no more `#`s.
    if (tpl.find('#', i) == std::string::npos) {
        out.suffix = tpl.substr(i);
        out.subInteger = false;
        return true;
    }

    // Sub-integer form: next char must be '.', followed by another
    // `#` group, and no more `#`s after that.
    if (i >= tpl.size() || tpl[i] != '.') return false;  // groups not adjacent
    ++i;  // skip the separating '.'
    if (i >= tpl.size() || tpl[i] != '#') return false;  // no second group
    size_t second = i;
    while (i < tpl.size() && tpl[i] == '#') { ++out.fracPad; ++i; }
    if (tpl.find('#', i) != std::string::npos) return false;  // more than two groups

    // Reconstruct prefix to include the first group and its dot.
    // The rule: prefix was "prefix1" up to the first `#`, then after
    // first group we have `.` then second group. The output format is
    // `prefix1 + intFormatted + . + fracFormatted + suffix`.
    out.subInteger = true;
    out.suffix = tpl.substr(i);
    // intPad stays as width of first group.
    // fracPad is set to width of second group.
    (void)second;
    return true;
}

// Format a single integer-form asset path.
std::string FormatIntegerPath(const TemplateParse& tpl, long long value) {
    char pad[64];
    std::snprintf(pad, sizeof(pad), "%0*lld", tpl.intPad, value);
    return tpl.prefix + pad + tpl.suffix;
}

// Format a sub-integer asset path. Spec §12.3.4.1.3.2: if the template
// is integer-only, fractional components get truncated. That case
// doesn't reach here (intPad-only path goes through FormatIntegerPath).
std::string FormatSubIntegerPath(const TemplateParse& tpl, double value) {
    long long intPart = static_cast<long long>(std::floor(value));
    double frac = value - static_cast<double>(intPart);
    // Scale frac to fracPad-many digits.
    long long fracInt = static_cast<long long>(std::llround(
        frac * std::pow(10.0, static_cast<double>(tpl.fracPad))));
    char intStr[64], fracStr[64];
    std::snprintf(intStr,  sizeof(intStr),  "%0*lld", tpl.intPad,  intPart);
    std::snprintf(fracStr, sizeof(fracStr), "%0*lld", tpl.fracPad, fracInt);
    return tpl.prefix + intStr + "." + fracStr + tpl.suffix;
}

// Iterate from startTime to endTime stepping by stride (inclusive of
// the endpoint when the stride lands exactly on it). Per spec §12.3.4.1.3.4
// example: start=12, end=25, stride=6 → 12, 18, 24 (stops before 30).
// stride must be positive.
std::vector<double> GenerateStepTimes(double startTime, double endTime,
                                       double stride) {
    std::vector<double> result;
    if (stride <= 0.0) return result;
    // Use a small epsilon to guard against float accumulation error
    // when the end lands exactly on a stride step.
    const double eps = stride * 1e-9;
    for (double t = startTime; t <= endTime + eps; t += stride) {
        result.push_back(t);
    }
    return result;
}

} // anonymous namespace

bool ExpandTemplateClipSet(
        const std::string& templateAssetPath,
        double startTime, double endTime, double stride,
        std::optional<double> activeOffset,
        const std::string& anchorLayerPath,
        const AssetResolver& resolver,
        ClipSetEntry& out) {
    if (templateAssetPath.empty()) return false;
    if (stride <= 0.0) return false;

    TemplateParse tpl;
    if (!ParseTemplatePath(templateAssetPath, tpl)) return false;

    if (activeOffset) {
        // Spec §12.3.4.1.3.5: templateActiveOffset cannot exceed
        // abs(templateStride).
        if (std::abs(*activeOffset) > std::abs(stride)) return false;
    }

    auto steps = GenerateStepTimes(startTime, endTime, stride);
    if (steps.empty()) return false;

    // Build assetPaths — one entry per step.
    out.assetPaths.clear();
    out.assetPaths.reserve(steps.size());
    for (double t : steps) {
        std::string rawPath = tpl.subInteger
            ? FormatSubIntegerPath(tpl, t)
            : FormatIntegerPath(tpl, static_cast<long long>(t));
        std::string resolved = resolver
            ? resolver(anchorLayerPath, rawPath)
            : rawPath;
        out.assetPaths.push_back(resolved.empty() ? rawPath : resolved);
    }

    // Build times + active arrays.
    out.times.clear();
    out.active.clear();
    if (activeOffset) {
        double off = *activeOffset;
        // Spec §12.3.4.1.3.5: prepend an edge knot at (start-off,
        // start-off), then per-step (t, t) entries, then an edge knot
        // at (end+off, end+off). Active entries are at (t+off, i).
        out.times.reserve(steps.size() + 2);
        out.times.emplace_back(steps.front() - off, steps.front() - off);
        for (double t : steps) out.times.emplace_back(t, t);
        out.times.emplace_back(steps.back() + off, steps.back() + off);

        out.active.reserve(steps.size());
        for (size_t i = 0; i < steps.size(); ++i) {
            out.active.emplace_back(steps[i] + off, static_cast<int>(i));
        }
    } else {
        out.times.reserve(steps.size());
        out.active.reserve(steps.size());
        for (size_t i = 0; i < steps.size(); ++i) {
            out.times.emplace_back(steps[i], steps[i]);
            out.active.emplace_back(steps[i], static_cast<int>(i));
        }
    }

    return true;
}

std::optional<ClipSetEntry> MaterializeClipSet(
        const Dictionary& clipsDict,
        const std::string& clipSetName,
        const std::string& anchorLayerPath,
        const AssetResolver& resolver) {
    auto it = clipsDict.find(clipSetName);
    if (it == clipsDict.end()) return std::nullopt;
    const auto* setDict = it->second.Get<Dictionary>();
    if (!setDict) return std::nullopt;

    ClipSetEntry entry;
    entry.name = clipSetName;

    // --- Common fields ---
    if (const auto* v = Find(*setDict, "primPath")) {
        std::string s;
        if (ReadStringLike(*v, s)) entry.primPath = Path::Parse(s);
    }
    if (const auto* v = Find(*setDict, "manifestAssetPath")) {
        std::string s;
        if (ReadStringLike(*v, s)) {
            entry.manifestAssetPath = resolver
                ? resolver(anchorLayerPath, s)
                : s;
            if (entry.manifestAssetPath.empty()) entry.manifestAssetPath = s;
        }
    }
    if (const auto* v = Find(*setDict, "interpolateMissingClipValues")) {
        bool b = false;
        if (ReadBool(*v, b)) entry.interpolateMissingClipValues = b;
    }

    // --- Explicit form — takes priority if both explicit and template
    // are authored (spec §12.3.4.1 last line).
    bool hasExplicit = false;
    if (const auto* v = Find(*setDict, "assetPaths")) {
        entry.assetPaths = ReadResolvedAssetPaths(*v, anchorLayerPath, resolver);
        hasExplicit = !entry.assetPaths.empty();
    }
    if (hasExplicit) {
        if (const auto* v = Find(*setDict, "active")) {
            ReadActivePairs(*v, entry.active);
        }
        if (const auto* v = Find(*setDict, "times")) {
            ReadDouble2Array(*v, entry.times);
        }
        return entry;
    }

    // --- Template form fallback ---
    const auto* tplV = Find(*setDict, "templateAssetPath");
    if (!tplV) return std::nullopt;  // no usable clip sequence

    std::string tplPath;
    if (!ReadStringLike(*tplV, tplPath)) return std::nullopt;

    double startT = 0, endT = 0, stride = 0;
    if (const auto* v = Find(*setDict, "templateStartTime")) ReadDouble(*v, startT);
    if (const auto* v = Find(*setDict, "templateEndTime"))   ReadDouble(*v, endT);
    if (const auto* v = Find(*setDict, "templateStride"))    ReadDouble(*v, stride);

    std::optional<double> off;
    if (const auto* v = Find(*setDict, "templateActiveOffset")) {
        double d;
        if (ReadDouble(*v, d)) off = d;
    }

    if (!ExpandTemplateClipSet(tplPath, startT, endT, stride, off,
                                anchorLayerPath, resolver, entry)) {
        return std::nullopt;
    }
    return entry;
}

// ============================================================
// ClipLayerCache
// ============================================================

std::shared_ptr<const Layer> ClipLayerCache::GetOrOpen(const std::string& resolvedPath) {
    if (resolvedPath.empty()) return nullptr;
    auto it = entries_.find(resolvedPath);
    if (it != entries_.end()) return it->second.layer;

    Entry e;
    auto result = ParseUsdFile(resolvedPath);
    if (result.success) {
        e.layer = std::make_shared<const Layer>(std::move(result.layer));
    }
    // Failed opens are cached as {nullptr} to suppress retry.
    auto [ins, _] = entries_.emplace(resolvedPath, std::move(e));
    return ins->second.layer;
}

bool ClipLayerCache::Contains(const std::string& resolvedPath) const {
    return entries_.find(resolvedPath) != entries_.end();
}

// ============================================================
// Manifest (§12.3.4.3)
// ============================================================

namespace {

// Collect every attribute spec path authored in `layer` into `out`.
void UnionAttributePathsInto(const Layer& layer,
                              PathSet& out) {
    layer.ForEachSpec([&](const Path& path, const Spec& spec) {
        if (spec.GetType() == SpecType::Attribute) {
            out.insert(path);
        }
    });
}

// Build a stable key for caching a derived manifest: concatenate the
// clip set's asset paths separated by '\0' (safe since paths don't
// contain NULs). Two clip sets that reference the same clip layers
// share a cached manifest.
std::string DerivedManifestKey(const ClipSetEntry& entry) {
    std::string key = "derived:";
    for (const auto& p : entry.assetPaths) {
        key.append(p);
        key.push_back('\0');
    }
    return key;
}

} // anonymous namespace

const ClipLayerCache::PathSet& ClipLayerCache::GetOrResolveManifest(
        const ClipSetEntry& entry) {
    // Key: authored path directly, or a derived-manifest key when
    // the clip set has no authored manifest.
    std::string key = entry.manifestAssetPath.empty()
        ? DerivedManifestKey(entry)
        : entry.manifestAssetPath;

    auto it = manifestCache_.find(key);
    if (it != manifestCache_.end()) return it->second;

    PathSet paths;
    if (!entry.manifestAssetPath.empty()) {
        if (auto layer = GetOrOpen(entry.manifestAssetPath)) {
            UnionAttributePathsInto(*layer, paths);
        }
    } else {
        // §12.3.4.3 auto-manifest: union of attribute specs across
        // every clip layer in the set.
        for (const auto& assetPath : entry.assetPaths) {
            if (auto layer = GetOrOpen(assetPath)) {
                UnionAttributePathsInto(*layer, paths);
            }
        }
    }
    auto [ins, _] = manifestCache_.emplace(std::move(key), std::move(paths));
    return ins->second;
}

std::vector<Path> GetManifestAttributePaths(const ClipSetEntry& entry,
                                             ClipLayerCache& cache) {
    std::vector<Path> out;
    if (entry.manifestAssetPath.empty()) return out;

    auto layer = cache.GetOrOpen(entry.manifestAssetPath);
    if (!layer) return out;

    layer->ForEachSpec([&](const Path& path, const Spec& spec) {
        if (spec.GetType() == SpecType::Attribute) {
            out.push_back(path);
        }
    });
    return out;
}

// ============================================================
// Per-opinion materialization + evaluation (§12.3.4.5)
// ============================================================

namespace {

// Piecewise-constant lookup: find the index `i` with the largest
// active[i].first <= stageTime. Below the first knot we hold at knot
// 0 (preserves the first-asset value across the hold region). When
// multiple knots share the same stageTime, the rightmost wins —
// right-continuous, matching the jump-discontinuity convention for
// `times` (§12.3.4.8).
size_t ActiveAssetIndex(const std::vector<std::pair<double, int>>& active,
                         double stageTime) {
    if (active.empty()) return 0;
    if (stageTime < active.front().first) {
        return static_cast<size_t>(active.front().second);
    }
    for (size_t i = 0; i + 1 < active.size(); ++i) {
        if (stageTime >= active[i].first && stageTime < active[i + 1].first) {
            return static_cast<size_t>(active[i].second);
        }
    }
    return static_cast<size_t>(active.back().second);
}

// Piecewise-linear stage→clip time mapping through the `times` array.
// Empty `times` means identity mapping. Right-continuous at
// coincident stageTime knots (§12.3.4.8 jump discontinuities): when
// two knots share the same stageTime, the later one wins at that
// stageTime. This lets authors express loops / resets by repeating a
// stageTime with a different clipTime.
double MapStageToClipTime(const std::vector<std::pair<double, double>>& times,
                           double stageTime) {
    if (times.empty()) return stageTime;
    if (stageTime < times.front().first) return times.front().second;
    if (stageTime > times.back().first)  return times.back().second;

    // Find the largest index `i` such that times[i].first <= stageTime.
    size_t i = 0;
    for (size_t k = 0; k + 1 < times.size(); ++k) {
        if (times[k].first <= stageTime) i = k;
    }
    // Advance through coincident knots (same stageTime) so the
    // rightmost wins at a jump boundary.
    while (i + 1 < times.size() && times[i + 1].first == times[i].first) ++i;

    if (i + 1 >= times.size()) return times[i].second;
    double t0 = times[i].first,     t1 = times[i + 1].first;
    double c0 = times[i].second,    c1 = times[i + 1].second;
    if (t1 == t0) return c1;  // safety — should be unreachable after advance
    if (stageTime <= t0) return c0;
    double alpha = (stageTime - t0) / (t1 - t0);
    return c0 + alpha * (c1 - c0);
}

// Rebuild a stage attribute path into its clip-layer-space path by
// swapping the `stagePrimPath` prefix for `clipPrimPath`. Per spec
// §12.3.4.5: `/Prim_1.size` with primPath=/Prim → `/Prim.size` in
// the clip. Returns empty when the stage attr path is not actually
// under `stagePrimPath` (shouldn't happen in practice but guards
// against malformed input).
Path RemapAttributePathToClip(const Path& stageAttrPath,
                               const Path& stagePrimPath,
                               const Path& clipPrimPath) {
    if (stageAttrPath.GetPrimPath() != stagePrimPath) return {};
    if (clipPrimPath.IsEmpty()) {
        // No primPath authored — the clip shares the stage namespace.
        return stageAttrPath;
    }
    return clipPrimPath.AppendProperty(stageAttrPath.GetPropertyName());
}

} // anonymous namespace

std::vector<ClipSetEntry> MaterializeClipSetsForLayer(
        const Spec& primSpec,
        const std::string& layerPath,
        size_t layerIndex,
        const AssetResolver& resolver) {
    std::vector<ClipSetEntry> out;

    const auto* clipsField = primSpec.GetField(FieldNames::clips);
    if (!clipsField) return out;
    const auto* clipsDict = clipsField->Get<Dictionary>();
    if (!clipsDict || clipsDict->empty()) return out;

    // Determine iteration order: authored `clipSets` listop (single-
    // layer, so .GetItems() reduces to the effective sequence for
    // this opinion) takes precedence. When absent, iterate dict keys
    // in sorted order for deterministic behavior.
    std::vector<std::string> order;
    if (const auto* setsField = primSpec.GetField(FieldNames::clipSets)) {
        if (const auto* lop = setsField->Get<ListOp<std::string>>()) {
            order = lop->GetItems();
        }
    }
    if (order.empty()) {
        std::set<std::string> keys;
        for (const auto& kv : *clipsDict) keys.insert(kv.first);
        order.assign(keys.begin(), keys.end());
    }

    out.reserve(order.size());
    for (const auto& name : order) {
        if (clipsDict->find(name) == clipsDict->end()) continue;
        auto entry = MaterializeClipSet(*clipsDict, name, layerPath, resolver);
        if (!entry) continue;
        entry->anchorLayerIndex = layerIndex;
        out.push_back(std::move(*entry));
    }
    return out;
}

namespace {

// Read an attribute value from a specific clip layer at a specific
// clip-time. Returns nullopt when the clip doesn't open, doesn't
// declare the attribute, or has no timeSamples/default that yield a
// value. Priority per §12.3.4.6: timeSamples > default.
std::optional<Value> ReadAttrAtClipTime(ClipLayerCache& cache,
                                         const std::string& assetPath,
                                         const Path& clipAttrPath,
                                         double clipTime) {
    auto layer = cache.GetOrOpen(assetPath);
    if (!layer) return std::nullopt;
    const Spec* spec = layer->GetAttributeSpec(clipAttrPath);
    if (!spec) return std::nullopt;

    if (const auto* ts = spec->GetField(FieldNames::timeSamples)) {
        if (ts->IsBlock()) return std::nullopt;
        if (const auto* dict = ts->Get<Dictionary>()) {
            if (!dict->empty()) {
                auto r = ResolveTimeSample(*dict, clipTime,
                                           UsdInterpolationType::Linear);
                if (r.found) return r.value;
            }
        }
    }
    if (const auto* def = spec->GetField(FieldNames::defaultValue)) {
        if (def->IsBlock()) return std::nullopt;
        return *def;
    }
    return std::nullopt;
}

// Linear interpolation for the scalar/vector types that clips
// commonly carry. Unknown types fall through to nullopt so the
// caller can fall back to held. Deliberately narrower than stage.cpp
// `LinearInterpolate` — we only need the shapes attribute values
// actually take.
std::optional<Value> LerpClipValue(const Value& a, const Value& b, double t) {
    if (auto* fa = a.Get<Float>()) {
        if (auto* fb = b.Get<Float>())
            return Value(static_cast<Float>(*fa + (*fb - *fa) * t));
    }
    if (auto* da = a.Get<Double>()) {
        if (auto* db = b.Get<Double>())
            return Value(*da + (*db - *da) * t);
    }
    if (auto* va = a.Get<GfVec3f>()) {
        if (auto* vb = b.Get<GfVec3f>()) {
            GfVec3f r{};
            for (int i = 0; i < 3; ++i)
                r[i] = static_cast<Float>((*va)[i] + ((*vb)[i] - (*va)[i]) * t);
            return Value(r);
        }
    }
    if (auto* va = a.Get<GfVec3d>()) {
        if (auto* vb = b.Get<GfVec3d>()) {
            GfVec3d r{};
            for (int i = 0; i < 3; ++i)
                r[i] = (*va)[i] + ((*vb)[i] - (*va)[i]) * t;
            return Value(r);
        }
    }
    return std::nullopt;
}

// Recover a value for `clipAttrPath` at `stageTime` when the
// currently-active clip has no value but the manifest declares the
// attribute (§12.3.4.6, §12.3.4.7). Walks every active knot, queries
// its clip at that knot's stageTime→clipTime mapping, and uses the
// nearest-before / nearest-after providing clips to hold or
// interpolate per `entry.interpolateMissingClipValues`.
std::optional<Value> RecoverMissingClipValue(
        const ClipSetEntry& entry,
        const Path& clipAttrPath,
        double stageTime,
        ClipLayerCache& cache) {
    // Enumerate (stageTimeKnot, value) pairs for each active knot
    // whose clip actually provides this attribute. Knots are already
    // sorted ascending by authoring convention.
    std::vector<std::pair<double, Value>> samples;
    samples.reserve(entry.active.size());
    for (const auto& [knotStage, assetIdx] : entry.active) {
        if (assetIdx < 0 ||
            static_cast<size_t>(assetIdx) >= entry.assetPaths.size())
            continue;
        double knotClipTime = MapStageToClipTime(entry.times, knotStage);
        auto v = ReadAttrAtClipTime(cache, entry.assetPaths[assetIdx],
                                     clipAttrPath, knotClipTime);
        if (v) samples.emplace_back(knotStage, std::move(*v));
    }
    if (samples.empty()) return std::nullopt;

    // Find nearest-before and nearest-after stageTime.
    const std::pair<double, Value>* before = nullptr;
    const std::pair<double, Value>* after = nullptr;
    for (const auto& s : samples) {
        if (s.first <= stageTime) before = &s;       // last match wins
        else if (!after)          after = &s;         // first match wins
    }

    if (!before && !after) return std::nullopt;
    if (!before) return after->second;
    if (!after)  return before->second;

    if (entry.interpolateMissingClipValues) {
        double dt = after->first - before->first;
        if (dt > 0.0) {
            double alpha = (stageTime - before->first) / dt;
            if (auto v = LerpClipValue(before->second, after->second, alpha))
                return v;
            // Unknown type → fall through to held below.
        }
    }
    // Held: nearest neighbour wins.
    double dBefore = stageTime - before->first;
    double dAfter  = after->first - stageTime;
    return (dBefore <= dAfter) ? before->second : after->second;
}

} // anonymous namespace

std::optional<Value> EvaluateClipSetAtTime(
        const ClipSetEntry& entry,
        const Path& attrPathOnStage,
        const Path& stagePrimPath,
        double stageTime,
        ClipLayerCache& cache) {
    Path clipAttrPath = RemapAttributePathToClip(attrPathOnStage, stagePrimPath,
                                                  entry.primPath);
    if (clipAttrPath.IsEmpty()) return std::nullopt;

    // §12.3.4.3 manifest gating. When the resolved manifest is
    // non-empty and doesn't declare this attribute, skip this clip
    // set — per spec the manifest is authoritative about what the
    // clip set provides. An empty resolved manifest is treated as
    // "unknown": we fall through to best-effort resolution rather
    // than silently hiding all clip values.
    const auto& manifestPaths = cache.GetOrResolveManifest(entry);
    if (!manifestPaths.empty() && !manifestPaths.count(clipAttrPath)) {
        return std::nullopt;
    }

    if (entry.assetPaths.empty()) return std::nullopt;

    // Primary path: active clip at stageTime.
    size_t idx = ActiveAssetIndex(entry.active, stageTime);
    if (idx < entry.assetPaths.size()) {
        double clipTime = MapStageToClipTime(entry.times, stageTime);
        if (auto v = ReadAttrAtClipTime(cache, entry.assetPaths[idx],
                                         clipAttrPath, clipTime)) {
            return v;
        }
    }

    // Missing from the active clip but declared in the manifest — try
    // neighbouring clips per §12.3.4.6 / §12.3.4.7. If there is no
    // manifest (auto-manifest failed + no authored manifest), we also
    // attempt recovery since reaching this point means the clip set
    // could still provide the attribute from a neighbour.
    return RecoverMissingClipValue(entry, clipAttrPath, stageTime, cache);
}

} // namespace nanousd
