// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/nanousd_backend.h"
#include "nanousd/nanousdapi.h"
#include "nanousd/nanousd.h"
#include "nanousd/asset_resolver.h"
#include "nanousd/geom_metrics.h"
#include "nanousd/resource.h"
#include "nanousd/usdc_writer.h"
#include "nanousd/usda_writer.h"
#include "nanousd/usdz_package.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace nanousd;

// ============================================================
// Internal handle structures (backend-owned)
// ============================================================

struct CachedFlatPrim {
    std::string path;
    std::string typeName;
    int parentIndex = -1;
    int depth = 0;
    int flags = 0;
};

struct CachedCompositionArc {
    int arcType = NANOUSD_ARC_NONE;
    int flags = 0;
    int layerIndex = -1;
    double offset = 0.0;
    double scale = 1.0;
    std::string layerPath;
    std::string sourcePath;
    std::string targetPath;
};

struct NanousdStage_s {
    std::optional<Stage> stage;
    std::string error;
    std::vector<UsdPrim> cachedTraversal;
    bool traversalDirty = true;
    std::vector<CachedFlatPrim> cachedFlatTraversal;
    bool flatTraversalDirty = true;

    bool IsValid() const { return stage && stage->IsValid(); }

    void EnsureTraversal() {
        if (traversalDirty && IsValid()) {
            cachedTraversal = stage->Traverse();
            traversalDirty = false;
            flatTraversalDirty = true;
        }
    }

    void EnsureFlatTraversal() {
        EnsureTraversal();
        if (!flatTraversalDirty || !IsValid()) return;

        cachedFlatTraversal.clear();
        cachedFlatTraversal.reserve(cachedTraversal.size());

        auto appendProxyPrim = [&](auto&& self, const UsdPrim& prim,
                                   const Path& displayPath,
                                   int parentIndex) -> void {
            if (!prim.IsValid() || displayPath.IsEmpty()) return;
            UsdPrim recordPrim = stage->GetPrimAtPath(displayPath);
            if (!recordPrim.IsValid()) recordPrim = prim;

            CachedFlatPrim rec;
            rec.path = displayPath.GetText();
            rec.typeName = recordPrim.GetTypeName().GetString();
            rec.parentIndex = parentIndex;
            rec.depth = static_cast<int>(displayPath.GetElements().size());
            rec.flags |= NANOUSD_FLAT_PRIM_INSTANCE_PROXY;

            const int selfIndex = static_cast<int>(cachedFlatTraversal.size());
            cachedFlatTraversal.push_back(std::move(rec));

            auto children = prim.GetChildren();
            for (const auto& child : children) {
                self(self, child, displayPath.AppendChild(child.GetName()), selfIndex);
            }
        };

        PathMap<int> indexByPath;
        indexByPath.reserve(cachedTraversal.size());
        for (const auto& prim : cachedTraversal) {
            const Path path = prim.GetPath();
            CachedFlatPrim rec;
            rec.path = path.GetText();
            rec.typeName = prim.GetTypeName().GetString();
            rec.depth = static_cast<int>(path.GetElements().size());

            Path parentPath = path.GetParentPath();
            auto parentIt = indexByPath.find(parentPath);
            rec.parentIndex = parentIt == indexByPath.end() ? -1 : parentIt->second;

            if (prim.IsInstance()) rec.flags |= NANOUSD_FLAT_PRIM_INSTANCE;
            if (prim.IsPrototype()) rec.flags |= NANOUSD_FLAT_PRIM_PROTOTYPE;
            if (prim.IsInPrototype()) rec.flags |= NANOUSD_FLAT_PRIM_IN_PROTOTYPE;

            const int selfIndex = static_cast<int>(cachedFlatTraversal.size());
            indexByPath[path] = selfIndex;
            cachedFlatTraversal.push_back(std::move(rec));

            if (!prim.IsInstance()) continue;
            auto proto = prim.GetPrototype();
            if (!proto.IsValid()) continue;
            auto children = proto.GetChildren();
            for (const auto& child : children) {
                appendProxyPrim(appendProxyPrim, child,
                                path.AppendChild(child.GetName()), selfIndex);
            }
        }
        flatTraversalDirty = false;
    }
};

struct NanousdPrim_s {
    UsdPrim prim;
    /* API-visible path of this handle, retained independently of `prim`.
     * This preserves display paths for instance proxies and keeps enough
     * identity to reactivate handles whose prim was masked inactive. */
    Path path;
    size_t layerIndex = 0;
    NanousdStage_s* stage = nullptr;  // back-pointer for cache invalidation

    // Resolve layer index through the graph (COW-safe)
    Layer& GetMutableLayer() {
        return stage->stage->GetGraph().GetMutableLayer(layerIndex);
    }

    mutable std::string cachedPath;
    mutable std::string cachedName;
    mutable std::string cachedTypeName;
    mutable std::string cachedKind;
    mutable std::vector<std::string> cachedPropNames;
    mutable bool propNamesCached = false;
    mutable std::vector<std::string> cachedAuthoredAttrNames;
    mutable bool authoredAttrNamesCached = false;
    mutable std::string cachedStringVal;
    mutable std::vector<std::string> cachedStringArray;
    mutable std::vector<std::string> cachedConnectionPaths;
    mutable std::string cachedCollectionInstance;
    mutable std::vector<std::string> cachedCollectionPaths;
    mutable bool collectionCached = false;
    mutable std::string cachedInstanceKey;
    mutable std::vector<CachedCompositionArc> cachedCompositionArcs;
    mutable bool compositionArcsCached = false;

    void EnsurePropNames() const {
        if (!propNamesCached) {
            auto tokens = prim.GetPropertyNames();
            cachedPropNames.clear();
            cachedPropNames.reserve(tokens.size());
            for (const auto& t : tokens) cachedPropNames.push_back(t.GetString());
            propNamesCached = true;
        }
    }

    void EnsureCollection(const char* instanceName) const {
        std::string instance = instanceName ? instanceName : "";
        if (collectionCached && cachedCollectionInstance == instance) return;

        cachedCollectionInstance = std::move(instance);
        cachedCollectionPaths.clear();
        auto members = prim.ComputeCollectionMembership(
            Token(cachedCollectionInstance));
        cachedCollectionPaths.reserve(members.size());
        for (const auto& path : members) {
            cachedCollectionPaths.push_back(path.GetText());
        }
        collectionCached = true;
    }

    void EnsureAuthoredAttrNames() const {
        if (!authoredAttrNamesCached) {
            auto tokens = prim.GetAuthoredAttributeNames();
            cachedAuthoredAttrNames.clear();
            cachedAuthoredAttrNames.reserve(tokens.size());
            for (const auto& t : tokens) cachedAuthoredAttrNames.push_back(t.GetString());
            authoredAttrNamesCached = true;
        }
    }

    void InvalidateCache() {
        propNamesCached = false;
        collectionCached = false;
        cachedPropNames.clear();
        cachedCollectionPaths.clear();
        authoredAttrNamesCached = false;
        cachedAuthoredAttrNames.clear();
        compositionArcsCached = false;
        cachedCompositionArcs.clear();
        if (stage) {
            stage->traversalDirty = true;
            stage->flatTraversalDirty = true;
        }
    }
};

struct NanousdPath_s {
    Path path;
    mutable std::string cachedText;
    mutable std::string cachedName;
};

struct NanousdListOp_s {
    ListOp<std::string> listop;
    mutable std::vector<std::string> cachedItems;
    mutable bool itemsCached = false;

    void EnsureItems() const {
        if (!itemsCached) {
            cachedItems = listop.GetItems();
            itemsCached = true;
        }
    }
};

// ============================================================
// Helpers
// ============================================================

// Returns a copy of the default value to avoid dangling pointers.
// UsdAttribute::GetDefault() can return a pointer into a temporary's
// schema fallback member; copying the value keeps it alive for callers.
static std::optional<Value> GetAttrDefault(NanousdPrim_s* p, const char* name) {
    if (!p || !name) return std::nullopt;
    auto attr = p->prim.GetAttribute(name);
    if (!attr.IsValid()) return std::nullopt;
    const Value* v = attr.GetDefault();
    if (!v) return std::nullopt;
    return *v;
}

// Returns a raw pointer to the default value — used ONLY by the zero-copy
// bulk array access functions (arraydataf/d/i) which return pointers into
// the Value's internal vector storage.
//
// LIMITATIONS (callers must be aware):
//
// 1. The pointer is ONLY stable for authored values that live in layer spec
//    storage. If the attribute has no authored value and GetDefault() falls
//    back to the schema definition, the returned pointer points into a
//    temporary UsdAttribute's member and is dangling after this function
//    returns. In practice bulk arrays are always authored, but any new
//    caller of this function must account for this.
//
// 2. The pointer is invalidated if the underlying value is modified (e.g.
//    via a set_attrib* call). Schema registration does NOT invalidate it
//    because registration doesn't touch layer storage.
//
// The root cause is that UsdAttribute copies SchemaPropertyDef into a local
// member rather than holding a stable pointer. Fixing that properly would
// require the PrimDefinition cache to guarantee pointer stability across
// invalidation, which it currently does not.
static const Value* GetAttrDefaultPtr(NanousdPrim_s* p, const char* name) {
    if (!p || !name) return nullptr;
    auto attr = p->prim.GetAttribute(name);
    if (!attr.IsValid()) return nullptr;
    return attr.GetDefault();
}

static NanousdPrim MakePrimHandleAtPath(NanousdStage_s* stage, UsdPrim prim,
                                        const Path& displayPath,
                                        size_t layerIndex = 0) {
    if (!stage || !prim.IsValid()) return nullptr;
    auto* h = new NanousdPrim_s;
    h->prim = std::move(prim);
    h->path = displayPath.IsEmpty() ? h->prim.GetPath() : displayPath;
    h->layerIndex = layerIndex;
    h->stage = stage;
    return h;
}

static NanousdPrim MakePrimHandle(NanousdStage_s* stage, UsdPrim prim,
                                  size_t layerIndex = 0) {
    if (!stage || !prim.IsValid()) return nullptr;
    Path displayPath = prim.GetPath();
    return MakePrimHandleAtPath(stage, std::move(prim), displayPath, layerIndex);
}

static void RefreshPrimHandle(NanousdPrim prim, const Path& path) {
    if (!prim) return;
    prim->path = path;
    prim->prim = (prim->stage && prim->stage->stage)
        ? prim->stage->stage->GetPrimAtPath(path)
        : UsdPrim();
    prim->layerIndex = 0;
    prim->InvalidateCache();
}

static int CopyStringToBuffer(const std::string& s, char* out, size_t out_size) {
    if (out && out_size > 0) {
        const size_t n = std::min(s.size(), out_size - 1);
        std::memcpy(out, s.data(), n);
        out[n] = '\0';
    }
    return s.size() > static_cast<size_t>(INT_MAX) ? INT_MAX
                                                   : static_cast<int>(s.size());
}

static bool PathIsAncestorOrEqual(const Path& maybeAncestor, const Path& desc) {
    if (maybeAncestor.IsEmpty() || desc.IsEmpty()) return false;
    for (Path cur = desc; !cur.IsEmpty(); cur = cur.GetParentPath()) {
        if (cur == maybeAncestor) return true;
    }
    return false;
}

static bool PathIsStrictAncestor(const Path& maybeAncestor, const Path& desc) {
    return maybeAncestor != desc && PathIsAncestorOrEqual(maybeAncestor, desc);
}

static Path SourcePrimPathForEntry(const OpinionEntry& entry, const Path& primPath) {
    return entry.sourcePath.IsEmpty()
        ? entry.pathMapping->MapToSource(primPath)
        : entry.sourcePath;
}

static Path RemapDescendantPath(const Path& input, const Path& oldRoot,
                                const Path& newRoot) {
    if (input == oldRoot) return newRoot;
    if (!PathIsAncestorOrEqual(oldRoot, input)) return {};

    auto inputElems = input.GetElements();
    auto oldElems = oldRoot.GetElements();
    if (oldElems.size() > inputElems.size()) return {};

    Path result = newRoot;
    for (size_t i = oldElems.size(); i < inputElems.size(); ++i) {
        if (!inputElems[i].variantSelections.empty()) return {};
        result = result.AppendChild(inputElems[i].name);
    }
    return result;
}

static bool FindInstanceRootForPath(const Stage& stage, const Path& path,
                                    Path* instanceRoot, Path* prototypeRoot) {
    if (path.IsEmpty()) return false;
    for (Path cur = path.GetParentPath();
         !cur.IsEmpty() && !cur.IsAbsoluteRoot();
         cur = cur.GetParentPath()) {
        Path proto = stage.GetPrototypePath(cur);
        if (proto.IsEmpty()) continue;
        if (instanceRoot) *instanceRoot = cur;
        if (prototypeRoot) *prototypeRoot = proto;
        return true;
    }
    return false;
}

static Path PrimHandlePath(const NanousdPrim_s* prim) {
    if (!prim) return {};
    if (!prim->path.IsEmpty()) return prim->path;
    return prim->prim.IsValid() ? prim->prim.GetPath() : Path();
}

static bool PathIsInPrototypeNamespace(const Stage& stage, const Path& path) {
    if (path.IsEmpty()) return false;
    for (const auto& protoPath : stage.GetPrototypePaths()) {
        if (PathIsAncestorOrEqual(protoPath, path)) return true;
    }
    return false;
}

static NanousdPrim MakePrimHandleForPath(NanousdStage_s* stage,
                                         const Path& displayPath,
                                         size_t layerIndex = 0) {
    if (!stage || !stage->stage || displayPath.IsEmpty()) return nullptr;

    auto prim = stage->stage->GetPrimAtPath(displayPath);
    if (prim.IsValid()) {
        return MakePrimHandleAtPath(stage, std::move(prim), displayPath,
                                    layerIndex);
    }

    Path instanceRoot;
    Path protoRoot;
    if (!FindInstanceRootForPath(*stage->stage, displayPath, &instanceRoot,
                                 &protoRoot)) {
        return nullptr;
    }

    Path protoPath = RemapDescendantPath(displayPath, instanceRoot, protoRoot);
    if (protoPath.IsEmpty()) return nullptr;
    prim = stage->stage->GetPrimAtPath(protoPath);
    return MakePrimHandleAtPath(stage, std::move(prim), displayPath,
                                layerIndex);
}

static int ArcTypeToInt(ArcType type) {
    return static_cast<int>(type);
}

static const char* ArcTypeName(ArcType type) {
    switch (type) {
        case ArcType::Sublayer:   return "sublayer";
        case ArcType::Reference:  return "reference";
        case ArcType::Payload:    return "payload";
        case ArcType::None:       return "none";
        case ArcType::Local:      return "local";
        case ArcType::Inherits:   return "inherits";
        case ArcType::Variant:    return "variant";
        case ArcType::Specialize: return "specialize";
        case ArcType::Relocate:   return "relocate";
    }
    return "unknown";
}

static bool SafeParseTime(const std::string& key, double& out) {
    try { out = std::stod(key); return true; } catch (...) { return false; }
}

static const Dictionary* GetTimeSamples(NanousdPrim_s* p, const char* name) {
    if (!p || !name) return nullptr;
    auto attr = p->prim.GetAttribute(name);
    if (!attr.IsValid()) return nullptr;
    auto* tsField = attr.GetTimeSamplesField();
    if (!tsField) return nullptr;
    return tsField->Get<Dictionary>();
}

template <typename OutT, typename InT>
static OutT ConvertNumeric(InT value) {
    return static_cast<OutT>(value);
}

template <typename OutT>
static OutT ConvertNumeric(Half value) {
    return static_cast<OutT>(static_cast<float>(value));
}

template <typename VecT, size_t N, typename OutT>
static int ReadVec(NanousdPrim_s* prim, const char* name, OutT* out) {
    auto val = GetAttrDefault(prim, name);
    if (!val) return 0;
    auto* v = val->Get<VecT>();
    if (!v) return 0;
    for (size_t i = 0; i < N; ++i)
        out[i] = ConvertNumeric<OutT>((*v)[i]);
    return 1;
}

template <typename VecT, size_t N, typename OutT>
static int ReadSampleVec(NanousdPrim_s* prim, const char* name, double time, OutT* out) {
    auto* dict = GetTimeSamples(prim, name);
    if (!dict) return 0;
    auto resolved = ResolveTimeSample(*dict, time, UsdInterpolationType::Linear);
    if (!resolved.found) return 0;
    auto* v = resolved.value.Get<VecT>();
    if (!v) return 0;
    for (size_t i = 0; i < N; ++i) out[i] = ConvertNumeric<OutT>((*v)[i]);
    return 1;
}

// ============================================================
// Backend function implementations
// ============================================================

static NanousdStage be_open(const char* filepath) {
    if (!filepath) return nullptr;
    auto* s = new NanousdStage_s;
    s->stage.emplace(Stage::Open(filepath));
    if (!s->IsValid()) {
        s->error = s->stage->GetError();
    }
    return s;
}

static NanousdStage be_invalid_stage(std::string error) {
    auto* s = new NanousdStage_s;
    s->error = std::move(error);
    return s;
}

static bool be_parse_population_mask(const char* const* maskPaths,
                                     int maskPathCount,
                                     std::vector<Path>& out,
                                     std::string& error) {
    if (maskPathCount < 0) {
        error = "population mask path count is negative";
        return false;
    }
    if (maskPathCount > 0 && !maskPaths) {
        error = "population mask path array is null";
        return false;
    }
    out.clear();
    out.reserve(static_cast<size_t>(maskPathCount));
    for (int i = 0; i < maskPathCount; ++i) {
        if (!maskPaths[i]) {
            error = "population mask path at index " + std::to_string(i) +
                    " is null";
            return false;
        }
        Path path = Path::Parse(maskPaths[i]);
        if (path.IsEmpty() || !path.IsPrimPath()) {
            error = "population mask path at index " + std::to_string(i) +
                    " is not an absolute prim path: " + maskPaths[i];
            return false;
        }
        out.push_back(path);
    }
    return true;
}

static NanousdStage be_open_masked(const char* filepath,
                                   const char* const* maskPaths,
                                   int maskPathCount) {
    if (!filepath) return nullptr;

    std::vector<Path> mask;
    std::string error;
    if (!be_parse_population_mask(maskPaths, maskPathCount, mask, error))
        return be_invalid_stage(std::move(error));

    auto* s = new NanousdStage_s;
    s->stage.emplace(Stage::OpenMasked(filepath, mask));
    if (!s->IsValid()) {
        s->error = s->stage->GetError();
    }
    return s;
}

static void be_close(NanousdStage stage) { delete stage; }

static int be_isvalid(NanousdStage stage) {
    return (stage && stage->IsValid()) ? 1 : 0;
}

static const char* be_error(NanousdStage stage) {
    if (!stage) return "null stage handle";
    return stage->error.c_str();
}

static double be_timecodes_per_second(NanousdStage stage) {
    return stage ? stage->stage->GetTimeCodesPerSecond() : 24.0;
}

static double be_frames_per_second(NanousdStage stage) {
    return stage ? stage->stage->GetFramesPerSecond() : 24.0;
}

static double be_start_time(NanousdStage stage) {
    return stage ? stage->stage->GetStartTimeCode() : 0.0;
}

static double be_end_time(NanousdStage stage) {
    return stage ? stage->stage->GetEndTimeCode() : 0.0;
}

static int be_nprims(NanousdStage stage) {
    if (!stage || !stage->IsValid()) return 0;
    stage->EnsureTraversal();
    return static_cast<int>(stage->cachedTraversal.size());
}

static NanousdPrim be_prim(NanousdStage stage, int index) {
    if (!stage || !stage->IsValid()) return nullptr;
    stage->EnsureTraversal();
    if (index < 0 || index >= static_cast<int>(stage->cachedTraversal.size()))
        return nullptr;
    return MakePrimHandle(stage, stage->cachedTraversal[index]);
}

static NanousdPrim be_primpath(NanousdStage stage, const char* path) {
    if (!stage || !path || !stage->IsValid()) return nullptr;
    return MakePrimHandleForPath(stage, Path::Parse(path));
}

static NanousdPrim be_defaultprim(NanousdStage stage) {
    if (!stage || !stage->IsValid()) return nullptr;
    auto prim = stage->stage->GetDefaultPrim();
    return MakePrimHandle(stage, std::move(prim));
}

static int be_traverse_flat(NanousdStage stage, NanousdFlatPrim* out,
                            int max_count) {
    if (!stage || !stage->IsValid()) return 0;
    stage->EnsureFlatTraversal();
    const int total = static_cast<int>(stage->cachedFlatTraversal.size());
    if (!out || max_count <= 0) return total;

    const int n = std::min(total, max_count);
    for (int i = 0; i < n; ++i) {
        const auto& src = stage->cachedFlatTraversal[static_cast<size_t>(i)];
        out[i].struct_size = static_cast<int>(sizeof(NanousdFlatPrim));
        out[i].path = src.path.c_str();
        out[i].type_name = src.typeName.c_str();
        out[i].parent_index = src.parentIndex;
        out[i].depth = src.depth;
        out[i].flags = src.flags;
    }
    return total;
}

static int be_nchildren(NanousdPrim prim) {
    if (!prim || !prim->prim.IsValid()) return 0;
    return static_cast<int>(prim->prim.GetChildren().size());
}

static NanousdPrim be_child(NanousdPrim prim, int index) {
    if (!prim || !prim->prim.IsValid()) return nullptr;
    auto children = prim->prim.GetChildren();
    if (index < 0 || index >= static_cast<int>(children.size())) return nullptr;
    Path displayPath = PrimHandlePath(prim);
    if (!displayPath.IsEmpty())
        displayPath = displayPath.AppendChild(children[index].GetName());
    return MakePrimHandleAtPath(prim->stage, std::move(children[index]),
                                displayPath, prim->layerIndex);
}

static NanousdPrim be_childname(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->prim.IsValid()) return nullptr;
    auto child = prim->prim.GetChild(name);
    if (!child.IsValid()) {
        auto children = prim->prim.GetChildren();
        for (auto& candidate : children) {
            if (candidate.GetName().GetString() == name) {
                child = std::move(candidate);
                break;
            }
        }
    }
    if (!child.IsValid()) return nullptr;
    Path displayPath = PrimHandlePath(prim);
    if (!displayPath.IsEmpty())
        displayPath = displayPath.AppendChild(child.GetName());
    return MakePrimHandleAtPath(prim->stage, std::move(child), displayPath,
                                prim->layerIndex);
}

static const char* be_path(NanousdPrim prim) {
    if (!prim) return "";
    Path path = PrimHandlePath(prim);
    prim->cachedPath = path.GetText();
    return prim->cachedPath.c_str();
}

static const char* be_name(NanousdPrim prim) {
    if (!prim) return "";
    Path path = PrimHandlePath(prim);
    prim->cachedName = path.GetName().GetString();
    return prim->cachedName.c_str();
}

static const char* be_typename(NanousdPrim prim) {
    if (!prim) return "";
    prim->cachedTypeName = prim->prim.GetTypeName().GetString();
    return prim->cachedTypeName.c_str();
}

static const char* be_kind(NanousdPrim prim) {
    if (!prim) return "";
    prim->cachedKind = prim->prim.GetKind().GetString();
    return prim->cachedKind.c_str();
}

static int be_isactive(NanousdPrim prim) {
    return (prim && prim->prim.IsActive()) ? 1 : 0;
}

static int be_isdefined(NanousdPrim prim) {
    return (prim && prim->prim.IsDefined()) ? 1 : 0;
}

static int be_isabstract(NanousdPrim prim) {
    return (prim && prim->prim.IsAbstract()) ? 1 : 0;
}

static int be_isinstanceable(NanousdPrim prim) {
    return (prim && prim->prim.IsInstanceable()) ? 1 : 0;
}

static int be_prim_isvalid(NanousdPrim prim) {
    return (prim && prim->prim.IsValid()) ? 1 : 0;
}

static int be_nattribs(NanousdPrim prim) {
    if (!prim || !prim->prim.IsValid()) return 0;
    return static_cast<int>(detail::GetPrimAttributeCount(prim->prim));
}

static const char* be_attribname(NanousdPrim prim, int index) {
    if (!prim || !prim->prim.IsValid()) return "";
    prim->EnsurePropNames();
    int count = 0;
    for (const auto& n : prim->cachedPropNames) {
        if (prim->prim.HasAttribute(n)) {
            if (count == index) {
                prim->cachedStringVal = n;
                return prim->cachedStringVal.c_str();
            }
            ++count;
        }
    }
    return "";
}

static int be_nauthored_attribs(NanousdPrim prim) {
    if (!prim || !prim->prim.IsValid()) return 0;
    prim->EnsureAuthoredAttrNames();
    return static_cast<int>(prim->cachedAuthoredAttrNames.size());
}

static const char* be_authored_attribname(NanousdPrim prim, int index) {
    if (!prim || !prim->prim.IsValid()) return "";
    prim->EnsureAuthoredAttrNames();
    if (index < 0 || index >= static_cast<int>(prim->cachedAuthoredAttrNames.size())) {
        return "";
    }
    prim->cachedStringVal = prim->cachedAuthoredAttrNames[static_cast<size_t>(index)];
    return prim->cachedStringVal.c_str();
}

static int be_hasattrib(NanousdPrim prim, const char* name) {
    return (prim && name && prim->prim.HasAttribute(name)) ? 1 : 0;
}

static const char* be_attribtype(NanousdPrim prim, const char* name) {
    if (!prim || !name) return "";
    auto attr = prim->prim.GetAttribute(name);
    if (!attr.IsValid()) return "";
    prim->cachedStringVal = attr.GetTypeName().GetString();
    return prim->cachedStringVal.c_str();
}

static int be_nproperties(NanousdPrim prim) {
    if (!prim || !prim->prim.IsValid()) return 0;
    prim->EnsurePropNames();
    return static_cast<int>(prim->cachedPropNames.size());
}

static const char* be_propertyname(NanousdPrim prim, int index) {
    if (!prim || !prim->prim.IsValid()) return "";
    prim->EnsurePropNames();
    if (index < 0 || index >= static_cast<int>(prim->cachedPropNames.size()))
        return "";
    return prim->cachedPropNames[static_cast<size_t>(index)].c_str();
}

static int be_property_is_attribute(NanousdPrim prim, const char* name) {
    return be_hasattrib(prim, name);
}

static int be_property_is_relationship(NanousdPrim prim, const char* name) {
    return (prim && name && prim->prim.HasRelationship(name)) ? 1 : 0;
}

static float be_attribf(NanousdPrim prim, const char* name, int* ok) {
    auto val = GetAttrDefault(prim, name);
    if (val) {
        if (auto* h = val->Get<Half>()) { if (ok) *ok = 1; return static_cast<float>(*h); }
        if (auto* f = val->Get<Float>()) { if (ok) *ok = 1; return *f; }
        if (auto* d = val->Get<Double>()) { if (ok) *ok = 1; return static_cast<float>(*d); }
    }
    if (ok) *ok = 0;
    return 0.0f;
}

static double be_attribd(NanousdPrim prim, const char* name, int* ok) {
    auto val = GetAttrDefault(prim, name);
    if (val) {
        if (auto* d = val->Get<Double>()) { if (ok) *ok = 1; return *d; }
        if (auto* f = val->Get<Float>()) { if (ok) *ok = 1; return static_cast<double>(*f); }
        if (auto* h = val->Get<Half>()) { if (ok) *ok = 1; return static_cast<double>(static_cast<float>(*h)); }
    }
    if (ok) *ok = 0;
    return 0.0;
}

static int be_attribi(NanousdPrim prim, const char* name, int* ok) {
    auto val = GetAttrDefault(prim, name);
    if (val) {
        if (auto* i = val->Get<Int>()) { if (ok) *ok = 1; return *i; }
    }
    if (ok) *ok = 0;
    return 0;
}

static const char* be_attribs(NanousdPrim prim, const char* name, int* ok) {
    auto val = GetAttrDefault(prim, name);
    if (val) {
        if (auto* s = val->Get<String>()) {
            if (ok) *ok = 1;
            prim->cachedStringVal = *s;
            return prim->cachedStringVal.c_str();
        }
    }
    if (ok) *ok = 0;
    return "";
}

static int be_attribv2f(NanousdPrim prim, const char* name, float out[2]) {
    if (ReadVec<GfVec2f, 2>(prim, name, out)) return 1;
    return ReadVec<GfVec2h, 2>(prim, name, out);
}
static int be_attribv3f(NanousdPrim prim, const char* name, float out[3]) {
    if (ReadVec<GfVec3f, 3>(prim, name, out)) return 1;
    return ReadVec<GfVec3h, 3>(prim, name, out);
}
static int be_attribv4f(NanousdPrim prim, const char* name, float out[4]) {
    if (ReadVec<GfVec4f, 4>(prim, name, out)) return 1;
    return ReadVec<GfVec4h, 4>(prim, name, out);
}
static int be_attribv2d(NanousdPrim prim, const char* name, double out[2]) {
    if (ReadVec<GfVec2d, 2>(prim, name, out)) return 1;
    return ReadVec<GfVec2h, 2>(prim, name, out);
}
static int be_attribv3d(NanousdPrim prim, const char* name, double out[3]) {
    if (ReadVec<GfVec3d, 3>(prim, name, out)) return 1;
    return ReadVec<GfVec3h, 3>(prim, name, out);
}
static int be_attribv4d(NanousdPrim prim, const char* name, double out[4]) {
    if (ReadVec<GfVec4d, 4>(prim, name, out)) return 1;
    return ReadVec<GfVec4h, 4>(prim, name, out);
}
static int be_attribv2i(NanousdPrim prim, const char* name, int out[2]) {
    return ReadVec<GfVec2i, 2>(prim, name, out);
}
static int be_attribv3i(NanousdPrim prim, const char* name, int out[3]) {
    return ReadVec<GfVec3i, 3>(prim, name, out);
}
static int be_attribv4i(NanousdPrim prim, const char* name, int out[4]) {
    return ReadVec<GfVec4i, 4>(prim, name, out);
}

static int be_attribm4d(NanousdPrim prim, const char* name, double out[16]) {
    auto val = GetAttrDefault(prim, name);
    if (!val) return 0;
    auto* m = val->Get<GfMatrix4d>();
    if (!m) return 0;
    for (int i = 0; i < 16; ++i) out[i] = m->data[i];
    return 1;
}

static int be_attribarraylen(NanousdPrim prim, const char* name) {
    auto val = GetAttrDefault(prim, name);
    if (!val) return -1;
    if (val->IsArray()) return static_cast<int>(val->ArraySize());
    // Scalar GfVec2i/3i/4i — return component count so callers can read via attribarrayi
    if (val->Get<GfVec2i>()) return 2;
    if (val->Get<GfVec3i>()) return 3;
    if (val->Get<GfVec4i>()) return 4;
    return -1;
}

static int be_attribarrayf(NanousdPrim prim, const char* name, float* out, int maxlen) {
    auto val = GetAttrDefault(prim, name);
    if (!val || !out || maxlen <= 0 || !val->IsArray()) return -1;
    // Typed float array — direct memcpy
    if (auto* v = val->Get<std::vector<Float>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        std::memcpy(out, v->data(), n * sizeof(float));
        return n;
    }
    // Typed half array — convert
    if (auto* v = val->Get<std::vector<Half>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) out[i] = static_cast<float>((*v)[i]);
        return n;
    }
    // Typed double array — convert
    if (auto* v = val->Get<std::vector<Double>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) out[i] = static_cast<float>((*v)[i]);
        return n;
    }
    // Vec2f array (e.g. texCoord2f[]) — flatten to float buffer
    if (auto* v = val->Get<std::vector<GfVec2f>>()) {
        int total = static_cast<int>(v->size()) * 2;
        int n = std::min(maxlen, total);
        std::memcpy(out, v->data(), n * sizeof(float));
        return n;
    }
    // Vec3f array (e.g. point3f[], normal3f[]) — flatten to float buffer
    if (auto* v = val->Get<std::vector<GfVec3f>>()) {
        int total = static_cast<int>(v->size()) * 3;
        int n = std::min(maxlen, total);
        std::memcpy(out, v->data(), n * sizeof(float));
        return n;
    }
    // Vec4f array — flatten to float buffer
    if (auto* v = val->Get<std::vector<GfVec4f>>()) {
        int total = static_cast<int>(v->size()) * 4;
        int n = std::min(maxlen, total);
        std::memcpy(out, v->data(), n * sizeof(float));
        return n;
    }
    // GfQuatf array — flatten to float buffer (4 floats per quat: i, j, k, r)
    if (auto* v = val->Get<std::vector<GfQuatf>>()) {
        int total = static_cast<int>(v->size()) * 4;
        int n = std::min(maxlen, total);
        for (int i = 0; i < static_cast<int>(v->size()) && i * 4 < n; ++i) {
            int rem = std::min(4, n - i * 4);
            std::memcpy(out + i * 4, (*v)[i].data.data(), rem * sizeof(float));
        }
        return n;
    }
    // GfVec3h (half3) array — convert half to float
    if (auto* v = val->Get<std::vector<GfVec3h>>()) {
        int total = static_cast<int>(v->size()) * 3;
        int n = std::min(maxlen, total);
        for (int i = 0; i < static_cast<int>(v->size()) && i * 3 < n; ++i) {
            out[i*3]   = static_cast<float>((*v)[i][0]);
            out[i*3+1] = static_cast<float>((*v)[i][1]);
            out[i*3+2] = static_cast<float>((*v)[i][2]);
        }
        return n;
    }
    // GfVec2h (half2) array — convert half to float
    if (auto* v = val->Get<std::vector<GfVec2h>>()) {
        int total = static_cast<int>(v->size()) * 2;
        int n = std::min(maxlen, total);
        for (int i = 0; i < static_cast<int>(v->size()) && i * 2 < n; ++i) {
            out[i*2]   = static_cast<float>((*v)[i][0]);
            out[i*2+1] = static_cast<float>((*v)[i][1]);
        }
        return n;
    }
    // GfVec4h (half4) array — convert half to float
    if (auto* v = val->Get<std::vector<GfVec4h>>()) {
        int total = static_cast<int>(v->size()) * 4;
        int n = std::min(maxlen, total);
        for (int i = 0; i < static_cast<int>(v->size()) && i * 4 < n; ++i) {
            out[i*4]   = static_cast<float>((*v)[i][0]);
            out[i*4+1] = static_cast<float>((*v)[i][1]);
            out[i*4+2] = static_cast<float>((*v)[i][2]);
            out[i*4+3] = static_cast<float>((*v)[i][3]);
        }
        return n;
    }
    // GfQuath (half quaternion) array — convert half to float
    if (auto* v = val->Get<std::vector<GfQuath>>()) {
        int total = static_cast<int>(v->size()) * 4;
        int n = std::min(maxlen, total);
        for (int i = 0; i < static_cast<int>(v->size()) && i * 4 < n; ++i) {
            out[i*4]   = static_cast<float>((*v)[i][0]);
            out[i*4+1] = static_cast<float>((*v)[i][1]);
            out[i*4+2] = static_cast<float>((*v)[i][2]);
            out[i*4+3] = static_cast<float>((*v)[i][3]);
        }
        return n;
    }
    // Legacy vector<Value> fallback
    if (auto* v = val->Get<std::vector<Value>>()) {
        // Check for GfQuatf elements
        if (!v->empty() && (*v)[0].Get<GfQuatf>()) {
            int nQ = static_cast<int>(v->size());
            int total = nQ * 4;
            int n = std::min(maxlen, total);
            for (int i = 0; i < nQ && i * 4 < n; ++i) {
                if (auto* q = (*v)[i].Get<GfQuatf>()) {
                    std::memcpy(out + i * 4, q->data.data(), 4 * sizeof(float));
                }
            }
            return n;
        }
        // Check for GfQuath elements
        if (!v->empty() && (*v)[0].Get<GfQuath>()) {
            int nQ = static_cast<int>(v->size());
            int total = nQ * 4;
            int n = std::min(maxlen, total);
            for (int i = 0; i < nQ && i * 4 < n; ++i) {
                if (auto* q = (*v)[i].Get<GfQuath>()) {
                    out[i*4]   = static_cast<float>((*q)[0]);
                    out[i*4+1] = static_cast<float>((*q)[1]);
                    out[i*4+2] = static_cast<float>((*q)[2]);
                    out[i*4+3] = static_cast<float>((*q)[3]);
                }
            }
            return n;
        }
        // Check for GfVec3h elements
        if (!v->empty() && (*v)[0].Get<GfVec3h>()) {
            int nH = static_cast<int>(v->size());
            int total = nH * 3;
            int n = std::min(maxlen, total);
            for (int i = 0; i < nH && i * 3 < n; ++i) {
                if (auto* h = (*v)[i].Get<GfVec3h>()) {
                    out[i*3]   = static_cast<float>((*h)[0]);
                    out[i*3+1] = static_cast<float>((*h)[1]);
                    out[i*3+2] = static_cast<float>((*h)[2]);
                }
            }
            return n;
        }
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) {
            if (auto* f = (*v)[i].Get<Float>()) out[i] = *f;
            else if (auto* d = (*v)[i].Get<Double>()) out[i] = static_cast<float>(*d);
            else if (auto* h = (*v)[i].Get<Half>()) out[i] = static_cast<float>(*h);
            else return -1;
        }
        return n;
    }
    return -1;
}

static int be_attribarrayi(NanousdPrim prim, const char* name, int* out, int maxlen) {
    auto val = GetAttrDefault(prim, name);
    if (!val || !out || maxlen <= 0) return -1;
    // Scalar Vec2i/Vec3i/Vec4i (e.g. int2 resolution)
    if (!val->IsArray()) {
        if (auto* v = val->Get<GfVec2i>()) {
            int n = std::min(maxlen, 2);
            for (int i = 0; i < n; ++i) out[i] = (*v)[i];
            return n;
        }
        if (auto* v = val->Get<GfVec3i>()) {
            int n = std::min(maxlen, 3);
            for (int i = 0; i < n; ++i) out[i] = (*v)[i];
            return n;
        }
        if (auto* v = val->Get<GfVec4i>()) {
            int n = std::min(maxlen, 4);
            for (int i = 0; i < n; ++i) out[i] = (*v)[i];
            return n;
        }
        return -1;
    }
    // Typed int array — direct memcpy
    if (auto* v = val->Get<std::vector<Int>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        std::memcpy(out, v->data(), n * sizeof(int));
        return n;
    }
    // Legacy vector<Value> fallback
    if (auto* v = val->Get<std::vector<Value>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) {
            if (auto* ii = (*v)[i].Get<Int>()) out[i] = *ii;
            else return -1;
        }
        return n;
    }
    return -1;
}

static int be_hassamples(NanousdPrim prim, const char* name) {
    if (!prim || !name) return 0;
    auto attr = prim->prim.GetAttribute(name);
    return attr.HasTimeSamples() ? 1 : 0;
}

static int be_nsamplekeys(NanousdPrim prim, const char* name) {
    auto* dict = GetTimeSamples(prim, name);
    return dict ? static_cast<int>(dict->size()) : 0;
}

static double be_samplekey(NanousdPrim prim, const char* name, int index) {
    auto* dict = GetTimeSamples(prim, name);
    if (!dict || index < 0 || index >= static_cast<int>(dict->size())) return 0.0;
    auto it = dict->begin();
    std::advance(it, index);
    return std::stod(it->first);
}

static float be_samplef(NanousdPrim prim, const char* name, double time, int* ok) {
    auto* dict = GetTimeSamples(prim, name);
    if (!dict) { if (ok) *ok = 0; return 0.0f; }
    auto resolved = ResolveTimeSample(*dict, time, UsdInterpolationType::Linear);
    if (!resolved.found) { if (ok) *ok = 0; return 0.0f; }
    if (auto* h = resolved.value.Get<Half>()) { if (ok) *ok = 1; return static_cast<float>(*h); }
    if (auto* f = resolved.value.Get<Float>()) { if (ok) *ok = 1; return *f; }
    if (auto* d = resolved.value.Get<Double>()) { if (ok) *ok = 1; return static_cast<float>(*d); }
    if (ok) *ok = 0;
    return 0.0f;
}

static int be_samplev3f(NanousdPrim prim, const char* name, double time, float out[3]) {
    if (ReadSampleVec<GfVec3f, 3>(prim, name, time, out)) return 1;
    return ReadSampleVec<GfVec3h, 3>(prim, name, time, out);
}

static int be_samplev3d(NanousdPrim prim, const char* name, double time, double out[3]) {
    if (ReadSampleVec<GfVec3d, 3>(prim, name, time, out)) return 1;
    return ReadSampleVec<GfVec3h, 3>(prim, name, time, out);
}

static int be_hasrel(NanousdPrim prim, const char* name) {
    return (prim && name && prim->prim.HasRelationship(name)) ? 1 : 0;
}

static int be_rel_authored(NanousdPrim prim, const char* name) {
    if (!prim || !name) return 0;
    auto rel = prim->prim.GetRelationship(name);
    return (rel.IsValid() && rel.IsAuthored()) ? 1 : 0;
}

static void be_freeprim(NanousdPrim prim) { delete prim; }

// ============================================================
// Extensions (appended to preserve v1 ABI)
// ============================================================

// --- Root layer path ---

static const char* be_stage_get_root_layer_path(NanousdStage stage) {
    if (!stage || !stage->IsValid()) return "";
    return stage->stage->GetRootLayerPath().c_str();
}

// --- Composed-layer enumeration ---

static int be_stage_n_layers(NanousdStage stage) {
    if (!stage || !stage->IsValid()) return 0;
    return static_cast<int>(stage->stage->GetGraph().GetNumLayers());
}

static const char* be_stage_layer_path(NanousdStage stage, int index) {
    if (!stage || !stage->IsValid()) return "";
    const auto& paths = stage->stage->GetGraph().layerPaths;
    if (index < 0 || static_cast<size_t>(index) >= paths.size()) return "";
    return paths[index].c_str();
}

static int copy_string_to_c_buffer(const std::string& value,
                                   char* out,
                                   size_t out_size) {
    if (!out || out_size == 0 || value.empty()) return 0;
    if (value.size() + 1 > out_size) {
        out[0] = '\0';
        return 0;
    }
    std::memcpy(out, value.c_str(), value.size() + 1);
    return 1;
}

static int be_resolve_asset_path(const char* anchorLayerPath,
                                  const char* assetPath,
                                  char* out,
                                  size_t out_size) {
    if (out && out_size > 0) out[0] = '\0';
    // An empty or null anchor is a valid, unanchored resolve: Stage::Open
    // resolves root paths with an empty anchor (DefaultResolve passes URIs,
    // absolute paths, and unresolvable relatives through unchanged). Only
    // the asset path is required.
    if (!assetPath || !assetPath[0]) return 0;
    return copy_string_to_c_buffer(
        DefaultResolve(anchorLayerPath ? anchorLayerPath : "", assetPath),
        out, out_size);
}

static int be_stage_resolve_asset_path(NanousdStage stage,
                                       const char* assetPath,
                                       char* out,
                                       size_t out_size) {
    if (out && out_size > 0) out[0] = '\0';
    if (!stage || !stage->IsValid() || !assetPath || !assetPath[0]) return 0;
    return copy_string_to_c_buffer(
        DefaultResolve(stage->stage->GetRootLayerPath(), assetPath),
        out, out_size);
}

static int be_read_asset_bytes(const char* resolvedLocation,
                               unsigned char** out_data,
                               size_t* out_size) {
    if (out_data) *out_data = nullptr;
    if (out_size) *out_size = 0;
    if (!resolvedLocation || !resolvedLocation[0] || !out_data || !out_size)
        return 0;

    ResourceReadResult resource = ReadResource(std::string(resolvedLocation));
    if (!resource.success) return 0;

    size_t size = resource.bytes.size();
    unsigned char* bytes = static_cast<unsigned char*>(std::malloc(size ? size : 1));
    if (!bytes) return 0;
    if (size) std::memcpy(bytes, resource.bytes.data(), size);

    *out_data = bytes;
    *out_size = size;
    return 1;
}

static void be_free_bytes(void* data) {
    std::free(data);
}

// --- Per-layer spec / opinion queries (usdview panel parity) -----
//
// Each takes the same `layerIdx` enumerated by stage_n_layers and a
// composed prim path, walking the same Graph the composed reads use.
//
// A prim pulled in through a reference / payload / inherit / specialize arc
// is authored under a different namespace in its source layer than the
// composed path it lands at (composed /World/Robot may be authored as /Robot
// in the referenced layer). Per-layer queries must follow the arc's path
// mapping to that authored source path; querying the composed path directly
// misses every opinion behind a composition arc.

// Visit every authored source prim path in `layerIdx` that contributes to a
// composed prim path. A single layer can contribute more than one source path,
// e.g. a local prim spec plus a selected variant-body spec in the same layer.
// Mirrors OpinionSourcePrimPath in stage.cpp: the cached sourcePath when
// known, else the arc path-mapping. The composed path is also tried as a
// fallback so identity opinions and specs absent from the prim-index cache
// still resolve, preserving the original direct-lookup behavior.
template <typename Fn>
static bool VisitLayerSourcePrimPaths(const CompositionGraph& graph,
                                      const Path& composedPrimPath,
                                      int layerIdx,
                                      Fn&& fn) {
    const PrimIndex* pi = graph.GetPrimIndex(composedPrimPath);
    std::vector<Path> seen;
    if (pi) {
        for (const auto& entry : pi->entries) {
            if (entry.layerIndex != static_cast<size_t>(layerIdx)) continue;
            Path src = entry.sourcePath.IsEmpty()
                ? entry.pathMapping->MapToSource(composedPrimPath)
                : entry.sourcePath;
            if (src.IsEmpty()) continue;
            if (std::find(seen.begin(), seen.end(), src) != seen.end())
                continue;
            seen.push_back(src);
            if (fn(src)) return true;
        }
    }
    if (std::find(seen.begin(), seen.end(), composedPrimPath) == seen.end()) {
        if (fn(composedPrimPath)) return true;
    }
    return false;
}

static int be_layer_has_prim_spec(NanousdStage stage, int layerIdx,
                                   const char* primPath) {
    if (!stage || !stage->IsValid() || !primPath) return 0;
    const auto& graph = stage->stage->GetGraph();
    if (layerIdx < 0 || static_cast<size_t>(layerIdx) >= graph.GetNumLayers())
        return 0;
    Path p = Path::Parse(primPath);
    if (p.IsEmpty()) return 0;
    const Layer& layer = graph.GetLayer(layerIdx);
    return VisitLayerSourcePrimPaths(graph, p, layerIdx,
        [&](const Path& src) {
            return layer.GetPrimSpec(src) != nullptr;
        }) ? 1 : 0;
}

static int be_layer_has_attr_opinion(NanousdStage stage, int layerIdx,
                                      const char* primPath, const char* attrName) {
    if (!stage || !stage->IsValid() || !primPath || !attrName) return 0;
    const auto& graph = stage->stage->GetGraph();
    if (layerIdx < 0 || static_cast<size_t>(layerIdx) >= graph.GetNumLayers())
        return 0;
    Path p = Path::Parse(primPath);
    if (p.IsEmpty()) return 0;
    const Layer& layer = graph.GetLayer(layerIdx);
    Token attrTok(attrName);
    return VisitLayerSourcePrimPaths(graph, p, layerIdx,
        [&](const Path& src) {
            return layer.HasSpec(src.AppendProperty(attrTok));
        }) ? 1 : 0;
}

static int be_layer_attr_nsamples(NanousdStage stage, int layerIdx,
                                   const char* primPath, const char* attrName) {
    if (!stage || !stage->IsValid() || !primPath || !attrName) return 0;
    const auto& graph = stage->stage->GetGraph();
    if (layerIdx < 0 || static_cast<size_t>(layerIdx) >= graph.GetNumLayers())
        return 0;
    Path p = Path::Parse(primPath);
    if (p.IsEmpty()) return 0;
    const Layer& layer = graph.GetLayer(layerIdx);
    Token attrTok(attrName);
    int nsamples = 0;
    VisitLayerSourcePrimPaths(graph, p, layerIdx,
        [&](const Path& src) {
            const auto* spec = layer.GetSpec(src.AppendProperty(attrTok));
            if (!spec) return false;
            auto* val = spec->GetField(FieldNames::timeSamples);
            if (!val) return false;
            // timeSamples are stored as a Dictionary keyed by stringified
            // time (see GetTimeSamples / the usda+usdc parsers), one entry
            // per sample -- NOT as a TimeSamples value.
            auto* ts = val->Get<Dictionary>();
            if (!ts) return false;
            nsamples = static_cast<int>(ts->size());
            return true;
        });
    return nsamples;
}

static NanousdListOp be_layer_prim_listop(NanousdStage stage, int layerIdx,
                                          const char* primPath, const char* field) {
    if (!stage || !stage->IsValid() || !primPath || !field) return nullptr;
    const auto& graph = stage->stage->GetGraph();
    if (layerIdx < 0 || static_cast<size_t>(layerIdx) >= graph.GetNumLayers())
        return nullptr;
    Path p = Path::Parse(primPath);
    if (p.IsEmpty()) return nullptr;
    Token fieldTok(field);
    const Layer& layer = graph.GetLayer(layerIdx);
    const Value* val = nullptr;
    VisitLayerSourcePrimPaths(graph, p, layerIdx,
        [&](const Path& src) {
            const auto* spec = layer.GetPrimSpec(src);
            if (!spec) return false;
            val = spec->GetField(fieldTok);
            return val != nullptr;
        });
    if (!val) return nullptr;

    // Same string-coercion logic as be_prim_listop: tokens / refs /
    // paths all flatten to "@asset@<primPath>" or "</PrimPath>" form
    // so the C boundary stays uniform.
    auto tokensToStrings = [](const ListOp<Token>& src) {
        ListOp<std::string> out;
        auto conv = [](const std::vector<Token>& in) {
            std::vector<std::string> r; r.reserve(in.size());
            for (const auto& t : in) r.push_back(t.GetString());
            return r;
        };
        if (src.IsExplicit()) out.SetExplicitItems(conv(src.GetExplicitItems()));
        else {
            out.SetPrependedItems(conv(src.GetPrependedItems()));
            out.SetAppendedItems(conv(src.GetAppendedItems()));
            out.SetDeletedItems(conv(src.GetDeletedItems()));
        }
        return out;
    };
    auto refToString = [](const Reference& r) -> std::string {
        std::string s;
        if (r.assetPath) { s += "@"; s += *r.assetPath; s += "@"; }
        if (r.primPath && !r.primPath->IsEmpty()) {
            s += "<"; s += r.primPath->GetText(); s += ">";
        }
        return s;
    };
    auto refsToStrings = [&](const ListOp<Reference>& src) {
        ListOp<std::string> out;
        auto conv = [&](const std::vector<Reference>& in) {
            std::vector<std::string> r; r.reserve(in.size());
            for (const auto& ref : in) r.push_back(refToString(ref));
            return r;
        };
        if (src.IsExplicit()) out.SetExplicitItems(conv(src.GetExplicitItems()));
        else {
            out.SetPrependedItems(conv(src.GetPrependedItems()));
            out.SetAppendedItems(conv(src.GetAppendedItems()));
            out.SetDeletedItems(conv(src.GetDeletedItems()));
        }
        return out;
    };
    auto pathsToStrings = [](const ListOp<Path>& src) {
        ListOp<std::string> out;
        auto conv = [](const std::vector<Path>& in) {
            std::vector<std::string> r; r.reserve(in.size());
            for (const auto& p : in) r.push_back(p.GetText());
            return r;
        };
        if (src.IsExplicit()) out.SetExplicitItems(conv(src.GetExplicitItems()));
        else {
            out.SetPrependedItems(conv(src.GetPrependedItems()));
            out.SetAppendedItems(conv(src.GetAppendedItems()));
            out.SetDeletedItems(conv(src.GetDeletedItems()));
        }
        return out;
    };

    std::optional<ListOp<std::string>> result;
    if (auto* lop = val->Get<ListOp<std::string>>()) result = *lop;
    else if (auto* lop = val->Get<ListOp<Token>>())     result = tokensToStrings(*lop);
    else if (auto* lop = val->Get<ListOp<Reference>>()) result = refsToStrings(*lop);
    else if (auto* lop = val->Get<ListOp<Path>>())      result = pathsToStrings(*lop);
    if (!result) return nullptr;
    auto* h = new NanousdListOp_s;
    h->listop = std::move(*result);
    return h;
}

// --- Sublayer enumeration & per-layer time offset -----
//
// CompositionGraph tracks the flat list (layerPaths) but not the
// parent/child relationship between sublayers. Recover it by reading
// each layer's authored `subLayers` field — the same data the parser
// used to add it to the graph in the first place.

namespace {
const SubLayerPaths* GetSublayerPaths(const Layer& layer) {
    const Spec& layerSpec = layer.GetLayerSpec();
    auto* val = layerSpec.GetField(FieldNames::subLayers);
    if (!val) return nullptr;
    return val->Get<SubLayerPaths>();
}
} // namespace

static int be_layer_n_sublayers(NanousdStage stage, int layerIdx) {
    if (!stage || !stage->IsValid()) return 0;
    const auto& graph = stage->stage->GetGraph();
    if (layerIdx < 0 || static_cast<size_t>(layerIdx) >= graph.GetNumLayers())
        return 0;
    const auto* slp = GetSublayerPaths(graph.GetLayer(layerIdx));
    return slp ? static_cast<int>(slp->paths.size()) : 0;
}

static const char* be_layer_sublayer_path(NanousdStage stage, int layerIdx,
                                           int subIdx) {
    if (!stage || !stage->IsValid()) return "";
    const auto& graph = stage->stage->GetGraph();
    if (layerIdx < 0 || static_cast<size_t>(layerIdx) >= graph.GetNumLayers())
        return "";
    const auto* slp = GetSublayerPaths(graph.GetLayer(layerIdx));
    if (!slp || subIdx < 0 || static_cast<size_t>(subIdx) >= slp->paths.size())
        return "";
    return slp->paths[subIdx].c_str();
}

static int be_layer_offset(NanousdStage stage, int layerIdx,
                            double* offset, double* scale) {
    if (!stage || !stage->IsValid()) return 0;
    const auto& graph = stage->stage->GetGraph();
    if (layerIdx < 0 || static_cast<size_t>(layerIdx) >= graph.GetNumLayers())
        return 0;
    if (static_cast<size_t>(layerIdx) >= graph.layerRetimings.size())
        return 0;
    const auto& r = graph.layerRetimings[layerIdx];
    if (offset) *offset = r.offset;
    if (scale)  *scale  = r.scale;
    return 1;
}

// --- Recomposition trigger -----
//
// Variant-selection writes (and other authoring ops that affect
// composition) only become visible to consumers after the graph is
// re-resolved. Rebuild by re-running ResolveStage on the current
// root layer.

static int be_recompose(NanousdStage stage) {
    if (!stage || !stage->IsValid()) return 0;
    try {
        if (!stage->stage->Recompose()) return 0;
        stage->traversalDirty = true;
        stage->flatTraversalDirty = true;
        stage->cachedTraversal.clear();
        stage->cachedFlatTraversal.clear();
        return 1;
    } catch (...) {
        return 0;
    }
}

// --- Generic stage metadata ---

static double be_metadatad(NanousdStage stage, const char* key, int* ok) {
    if (!stage || !key || !stage->IsValid()) { if (ok) *ok = 0; return 0.0; }
    auto& fields = stage->stage->GetComposedLayerSpec().GetFields();
    auto* v = fields.Get(Token(key));
    if (!v) { if (ok) *ok = 0; return 0.0; }
    if (auto* d = v->Get<Double>()) { if (ok) *ok = 1; return *d; }
    if (auto* i = v->Get<Int>()) { if (ok) *ok = 1; return static_cast<double>(*i); }
    if (auto* f = v->Get<Float>()) { if (ok) *ok = 1; return static_cast<double>(*f); }
    if (ok) *ok = 0;
    return 0.0;
}

static const char* be_metadatas(NanousdStage stage, const char* key, int* ok) {
    if (!stage || !key || !stage->IsValid()) { if (ok) *ok = 0; return ""; }
    auto& fields = stage->stage->GetComposedLayerSpec().GetFields();
    auto* v = fields.Get(Token(key));
    if (!v) { if (ok) *ok = 0; return ""; }
    if (auto* s = v->Get<String>()) { if (ok) *ok = 1; stage->error = *s; return stage->error.c_str(); }
    if (auto* t = v->Get<Token>()) { if (ok) *ok = 1; stage->error = t->GetString(); return stage->error.c_str(); }
    if (ok) *ok = 0;
    return "";
}

static int be_set_stage_metadatad(NanousdStage stage, const char* key, double value) {
    if (!stage || !key || !stage->IsValid()) return 0;
    stage->stage->GetMutableComposedLayerSpec().SetField(Token(key), Value(Double(value)));
    stage->stage->GetMutableLayer().GetLayerSpec().SetField(Token(key), Value(Double(value)));
    return 1;
}

static int be_set_stage_metadatas(NanousdStage stage, const char* key, const char* value) {
    if (!stage || !key || !value || !stage->IsValid()) return 0;
    stage->stage->GetMutableComposedLayerSpec().SetField(Token(key), Value(String(value)));
    stage->stage->GetMutableLayer().GetLayerSpec().SetField(Token(key), Value(String(value)));
    return 1;
}

static int be_set_stage_metadata_token(NanousdStage stage, const char* key, const char* value) {
    if (!stage || !key || !value || !stage->IsValid()) return 0;
    stage->stage->GetMutableComposedLayerSpec().SetField(Token(key), Value(Token(value)));
    stage->stage->GetMutableLayer().GetLayerSpec().SetField(Token(key), Value(Token(value)));
    return 1;
}

// --- Schema queries ---

static int be_isa(NanousdPrim prim, const char* typeName) {
    return (prim && typeName && prim->prim.IsA(typeName)) ? 1 : 0;
}

static int be_hasapi(NanousdPrim prim, const char* apiName) {
    return (prim && apiName && prim->prim.HasAPI(apiName)) ? 1 : 0;
}

// --- Additional scalar reads ---

static int64_t be_attribi64(NanousdPrim prim, const char* name, int* ok) {
    auto val = GetAttrDefault(prim, name);
    if (val) {
        if (auto* i = val->Get<Int>()) { if (ok) *ok = 1; return static_cast<int64_t>(*i); }
        if (auto* i64 = val->Get<int64_t>()) { if (ok) *ok = 1; return *i64; }
    }
    if (ok) *ok = 0;
    return 0;
}

static int be_attribb(NanousdPrim prim, const char* name, int* ok) {
    auto val = GetAttrDefault(prim, name);
    if (val) {
        if (auto* b = val->Get<bool>()) { if (ok) *ok = 1; return *b ? 1 : 0; }
        if (auto* i = val->Get<Int>()) { if (ok) *ok = 1; return *i ? 1 : 0; }
    }
    if (ok) *ok = 0;
    return 0;
}

// --- Additional array reads ---

static int be_attribarrayd(NanousdPrim prim, const char* name, double* out, int maxlen) {
    auto val = GetAttrDefault(prim, name);
    if (!val || !out || maxlen <= 0 || !val->IsArray()) return -1;
    // Typed double array — direct memcpy
    if (auto* v = val->Get<std::vector<Double>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        std::memcpy(out, v->data(), n * sizeof(double));
        return n;
    }
    // Typed float array — convert
    if (auto* v = val->Get<std::vector<Float>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) out[i] = static_cast<double>((*v)[i]);
        return n;
    }
    // Typed half array — convert
    if (auto* v = val->Get<std::vector<Half>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i)
            out[i] = static_cast<double>(static_cast<float>((*v)[i]));
        return n;
    }
    // GfMatrix4d array — flatten 4x4 matrices to double buffer (16 doubles each)
    if (auto* v = val->Get<std::vector<GfMatrix4d>>()) {
        int total = static_cast<int>(v->size()) * 16;
        int n = std::min(maxlen, total);
        for (int i = 0; i < static_cast<int>(v->size()) && i * 16 < n; ++i) {
            int rem = std::min(16, n - i * 16);
            std::memcpy(out + i * 16, (*v)[i].data.data(), rem * sizeof(double));
        }
        return n;
    }
    // Legacy vector<Value> fallback
    if (auto* v = val->Get<std::vector<Value>>()) {
        // Check if elements are Matrix4d
        if (!v->empty() && (*v)[0].Get<GfMatrix4d>()) {
            int nMats = static_cast<int>(v->size());
            int total = nMats * 16;
            int n = std::min(maxlen, total);
            for (int i = 0; i < nMats && i * 16 < n; ++i) {
                if (auto* m = (*v)[i].Get<GfMatrix4d>()) {
                    int rem = std::min(16, n - i * 16);
                    std::memcpy(out + i * 16, m->data.data(), rem * sizeof(double));
                }
            }
            return n;
        }
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) {
            if (auto* d = (*v)[i].Get<Double>()) out[i] = *d;
            else if (auto* f = (*v)[i].Get<Float>()) out[i] = static_cast<double>(*f);
            else if (auto* h = (*v)[i].Get<Half>()) out[i] = static_cast<double>(static_cast<float>(*h));
            else return -1;
        }
        return n;
    }
    return -1;
}

static int be_attribarrayi64(NanousdPrim prim, const char* name, int64_t* out, int maxlen) {
    auto val = GetAttrDefault(prim, name);
    if (!val || !out || maxlen <= 0 || !val->IsArray()) return -1;
    // Typed int64 array — direct memcpy
    if (auto* v = val->Get<std::vector<Int64>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        std::memcpy(out, v->data(), n * sizeof(int64_t));
        return n;
    }
    // Typed int array — widening conversion
    if (auto* v = val->Get<std::vector<Int>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) out[i] = static_cast<int64_t>((*v)[i]);
        return n;
    }
    // Legacy vector<Value> fallback
    if (auto* v = val->Get<std::vector<Value>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) {
            if (auto* i64 = (*v)[i].Get<Int64>()) out[i] = *i64;
            else if (auto* i32 = (*v)[i].Get<Int>()) out[i] = static_cast<int64_t>(*i32);
            else return -1;
        }
        return n;
    }
    return -1;
}

// --- Additional time sample reads ---

static double be_sampled(NanousdPrim prim, const char* name, double time, int* ok) {
    auto* dict = GetTimeSamples(prim, name);
    if (!dict) { if (ok) *ok = 0; return 0.0; }
    auto resolved = ResolveTimeSample(*dict, time, UsdInterpolationType::Linear);
    if (!resolved.found) { if (ok) *ok = 0; return 0.0; }
    if (auto* d = resolved.value.Get<Double>()) { if (ok) *ok = 1; return *d; }
    if (auto* f = resolved.value.Get<Float>()) { if (ok) *ok = 1; return static_cast<double>(*f); }
    if (auto* h = resolved.value.Get<Half>()) { if (ok) *ok = 1; return static_cast<double>(static_cast<float>(*h)); }
    if (ok) *ok = 0;
    return 0.0;
}

// --- Array time sample reads ---

static int be_samplev2f(NanousdPrim prim, const char* name, double time, float out[2]) {
    if (ReadSampleVec<GfVec2f, 2>(prim, name, time, out)) return 1;
    return ReadSampleVec<GfVec2h, 2>(prim, name, time, out);
}

static int be_samplearrayf(NanousdPrim prim, const char* name, double time,
                            float* out, int maxlen) {
    if (!out || maxlen <= 0) return -1;
    auto* dict = GetTimeSamples(prim, name);
    if (!dict) return -1;
    auto resolved = ResolveTimeSample(*dict, time, UsdInterpolationType::Held);
    if (!resolved.found) return 0;
    const auto& val = resolved.value;
    if (!val.IsArray()) return -1;
    if (auto* v = val.Get<std::vector<Float>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        std::memcpy(out, v->data(), n * sizeof(float));
        return n;
    }
    if (auto* v = val.Get<std::vector<Half>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) out[i] = static_cast<float>((*v)[i]);
        return n;
    }
    if (auto* v = val.Get<std::vector<Double>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) out[i] = static_cast<float>((*v)[i]);
        return n;
    }
    if (auto* v = val.Get<std::vector<GfVec2f>>()) {
        int total = static_cast<int>(v->size()) * 2;
        int n = std::min(maxlen, total);
        std::memcpy(out, v->data(), n * sizeof(float));
        return n;
    }
    if (auto* v = val.Get<std::vector<GfVec3f>>()) {
        int total = static_cast<int>(v->size()) * 3;
        int n = std::min(maxlen, total);
        std::memcpy(out, v->data(), n * sizeof(float));
        return n;
    }
    if (auto* v = val.Get<std::vector<GfVec4f>>()) {
        int total = static_cast<int>(v->size()) * 4;
        int n = std::min(maxlen, total);
        std::memcpy(out, v->data(), n * sizeof(float));
        return n;
    }
    if (auto* v = val.Get<std::vector<GfQuatf>>()) {
        int total = static_cast<int>(v->size()) * 4;
        int n = std::min(maxlen, total);
        for (int i = 0; i < static_cast<int>(v->size()) && i * 4 < n; ++i) {
            int rem = std::min(4, n - i * 4);
            std::memcpy(out + i * 4, (*v)[i].data.data(),
                        rem * sizeof(float));
        }
        return n;
    }
    if (auto* v = val.Get<std::vector<GfVec2h>>()) {
        int total = static_cast<int>(v->size()) * 2;
        int n = std::min(maxlen, total);
        for (int i = 0; i < static_cast<int>(v->size()) && i * 2 < n; ++i) {
            out[i * 2 + 0] = static_cast<float>((*v)[i][0]);
            if (i * 2 + 1 < n) out[i * 2 + 1] = static_cast<float>((*v)[i][1]);
        }
        return n;
    }
    if (auto* v = val.Get<std::vector<GfVec3h>>()) {
        int total = static_cast<int>(v->size()) * 3;
        int n = std::min(maxlen, total);
        for (int i = 0; i < static_cast<int>(v->size()) && i * 3 < n; ++i) {
            out[i * 3 + 0] = static_cast<float>((*v)[i][0]);
            if (i * 3 + 1 < n) out[i * 3 + 1] = static_cast<float>((*v)[i][1]);
            if (i * 3 + 2 < n) out[i * 3 + 2] = static_cast<float>((*v)[i][2]);
        }
        return n;
    }
    if (auto* v = val.Get<std::vector<GfVec4h>>()) {
        int total = static_cast<int>(v->size()) * 4;
        int n = std::min(maxlen, total);
        for (int i = 0; i < static_cast<int>(v->size()) && i * 4 < n; ++i) {
            out[i * 4 + 0] = static_cast<float>((*v)[i][0]);
            if (i * 4 + 1 < n) out[i * 4 + 1] = static_cast<float>((*v)[i][1]);
            if (i * 4 + 2 < n) out[i * 4 + 2] = static_cast<float>((*v)[i][2]);
            if (i * 4 + 3 < n) out[i * 4 + 3] = static_cast<float>((*v)[i][3]);
        }
        return n;
    }
    if (auto* v = val.Get<std::vector<GfQuath>>()) {
        int total = static_cast<int>(v->size()) * 4;
        int n = std::min(maxlen, total);
        for (int i = 0; i < static_cast<int>(v->size()) && i * 4 < n; ++i) {
            out[i * 4 + 0] = static_cast<float>((*v)[i][0]);
            if (i * 4 + 1 < n) out[i * 4 + 1] = static_cast<float>((*v)[i][1]);
            if (i * 4 + 2 < n) out[i * 4 + 2] = static_cast<float>((*v)[i][2]);
            if (i * 4 + 3 < n) out[i * 4 + 3] = static_cast<float>((*v)[i][3]);
        }
        return n;
    }
    if (auto* v = val.Get<std::vector<Value>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) {
            if (auto* f = (*v)[i].Get<Float>()) out[i] = *f;
            else if (auto* d = (*v)[i].Get<Double>()) out[i] = static_cast<float>(*d);
            else if (auto* h = (*v)[i].Get<Half>()) out[i] = static_cast<float>(*h);
            else if (auto* ii = (*v)[i].Get<Int>()) out[i] = static_cast<float>(*ii);
            else return -1;
        }
        return n;
    }
    return -1;
}

static int be_samplearrayd(NanousdPrim prim, const char* name, double time,
                            double* out, int maxlen) {
    if (!out || maxlen <= 0) return -1;
    auto* dict = GetTimeSamples(prim, name);
    if (!dict) return -1;
    auto resolved = ResolveTimeSample(*dict, time, UsdInterpolationType::Held);
    if (!resolved.found) return 0;
    const auto& val = resolved.value;
    if (!val.IsArray()) return -1;
    if (auto* v = val.Get<std::vector<Double>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        std::memcpy(out, v->data(), n * sizeof(double));
        return n;
    }
    if (auto* v = val.Get<std::vector<Float>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) out[i] = static_cast<double>((*v)[i]);
        return n;
    }
    if (auto* v = val.Get<std::vector<Half>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i)
            out[i] = static_cast<double>(static_cast<float>((*v)[i]));
        return n;
    }
    // Typed double vector/quat/matrix arrays (Vec*d, Quatd, Matrix*d). These
    // are contiguous std::array<double,N> (see types.h), so they flatten by
    // memcpy. Without these, double-precision time-sampled arrays (matrix4d,
    // quatd, point3d, vector3d - common xformOp/skel animation) fell through
    // to -1 and were silently dropped (be_samplearrayf has the float twins).
    if (auto* v = val.Get<std::vector<GfVec2d>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()) * 2);
        std::memcpy(out, v->data(), n * sizeof(double));
        return n;
    }
    if (auto* v = val.Get<std::vector<GfVec3d>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()) * 3);
        std::memcpy(out, v->data(), n * sizeof(double));
        return n;
    }
    if (auto* v = val.Get<std::vector<GfVec4d>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()) * 4);
        std::memcpy(out, v->data(), n * sizeof(double));
        return n;
    }
    if (auto* v = val.Get<std::vector<GfQuatd>>()) {
        int total = static_cast<int>(v->size()) * 4;
        int n = std::min(maxlen, total);
        for (int i = 0; i < static_cast<int>(v->size()) && i * 4 < n; ++i) {
            int rem = std::min(4, n - i * 4);
            std::memcpy(out + i * 4, (*v)[i].data.data(), rem * sizeof(double));
        }
        return n;
    }
    if (auto* v = val.Get<std::vector<GfMatrix2d>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()) * 4);
        std::memcpy(out, v->data(), n * sizeof(double));
        return n;
    }
    if (auto* v = val.Get<std::vector<GfMatrix3d>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()) * 9);
        std::memcpy(out, v->data(), n * sizeof(double));
        return n;
    }
    if (auto* v = val.Get<std::vector<GfMatrix4d>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()) * 16);
        std::memcpy(out, v->data(), n * sizeof(double));
        return n;
    }
    if (auto* v = val.Get<std::vector<Value>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) {
            if (auto* d = (*v)[i].Get<Double>()) out[i] = *d;
            else if (auto* f = (*v)[i].Get<Float>()) out[i] = static_cast<double>(*f);
            else if (auto* h = (*v)[i].Get<Half>()) out[i] = static_cast<double>(static_cast<float>(*h));
            else if (auto* ii = (*v)[i].Get<Int>()) out[i] = static_cast<double>(*ii);
            else return -1;
        }
        return n;
    }
    return -1;
}

static int be_samplearrayi(NanousdPrim prim, const char* name, double time,
                            int* out, int maxlen) {
    if (!out || maxlen <= 0) return -1;
    auto* dict = GetTimeSamples(prim, name);
    if (!dict) return -1;
    auto resolved = ResolveTimeSample(*dict, time, UsdInterpolationType::Held);
    if (!resolved.found) return 0;
    const auto& val = resolved.value;
    if (!val.IsArray()) return -1;
    if (auto* v = val.Get<std::vector<Int>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        std::memcpy(out, v->data(), n * sizeof(int));
        return n;
    }
    if (auto* v = val.Get<std::vector<Value>>()) {
        int n = std::min(maxlen, static_cast<int>(v->size()));
        for (int i = 0; i < n; ++i) {
            if (auto* ii = (*v)[i].Get<Int>()) out[i] = *ii;
            else return -1;
        }
        return n;
    }
    return -1;
}

// --- Relationship targets ---

static int be_nreltargets(NanousdPrim prim, const char* name) {
    if (!prim || !name) return 0;
    auto rel = prim->prim.GetRelationship(name);
    if (!rel.IsValid()) return 0;
    return static_cast<int>(rel.GetTargets().size());
}

static const char* be_reltarget(NanousdPrim prim, const char* name, int index) {
    if (!prim || !name) return "";
    auto rel = prim->prim.GetRelationship(name);
    if (!rel.IsValid()) return "";
    auto targets = rel.GetTargets();
    if (index < 0 || index >= static_cast<int>(targets.size())) return "";
    prim->cachedStringVal = targets[index].GetText();
    return prim->cachedStringVal.c_str();
}

static const char* be_rel_metadatas(NanousdPrim prim, const char* relName,
                                    const char* key, int* ok) {
    if (ok) *ok = 0;
    if (!prim || !relName || !key || !prim->stage || !prim->stage->stage)
        return "";

    bool found = false;
    auto read_field = [&](const Spec* spec) -> const char* {
        if (!spec || spec->GetType() != SpecType::Relationship)
            return "";
        const Value* field = spec->GetField(Token(key));
        if (!field)
            return "";
        if (auto* s = field->Get<String>()) {
            prim->cachedStringVal = *s;
            found = true;
            if (ok) *ok = 1;
            return prim->cachedStringVal.c_str();
        }
        if (auto* t = field->Get<Token>()) {
            prim->cachedStringVal = t->GetString();
            found = true;
            if (ok) *ok = 1;
            return prim->cachedStringVal.c_str();
        }
        return "";
    };

    const Path primPath = prim->prim.GetPath();
    const Path propPath = primPath.AppendProperty(Token(relName));
    const CompositionGraph& graph = prim->stage->stage->GetGraph();
    if (const PrimIndex* idx = graph.GetPrimIndex(primPath)) {
        for (const auto& entry : idx->entries) {
            const Layer& layer = graph.GetLayer(entry.layerIndex);
            Path sourcePrimPath = SourcePrimPathForEntry(entry, primPath);
            if (sourcePrimPath.IsEmpty())
                continue;
            Path sourcePath = sourcePrimPath.AppendProperty(Token(relName));
            if (sourcePath.IsEmpty())
                continue;
            const char* value = read_field(layer.GetRelationshipSpec(sourcePath));
            if (found)
                return value;
        }
    }

    const char* value =
        read_field(prim->stage->stage->GetComposedLayer().GetSpec(propPath));
    if (found)
        return value;
    return "";
}

// --- Collections ---

static int be_collection_nmembers(NanousdPrim prim, const char* instanceName) {
    if (!prim || !instanceName || !*instanceName) return 0;
    prim->EnsureCollection(instanceName);
    return static_cast<int>(prim->cachedCollectionPaths.size());
}

static const char* be_collection_member(NanousdPrim prim,
                                        const char* instanceName,
                                        int index) {
    if (!prim || !instanceName || !*instanceName) return "";
    prim->EnsureCollection(instanceName);
    if (index < 0 ||
        index >= static_cast<int>(prim->cachedCollectionPaths.size())) {
        return "";
    }
    return prim->cachedCollectionPaths[index].c_str();
}

static int be_collection_contains(NanousdPrim prim,
                                  const char* instanceName,
                                  const char* pathText) {
    if (!prim || !instanceName || !*instanceName || !pathText) return 0;
    Path path = Path::Parse(pathText);
    if (path.IsEmpty()) return 0;
    prim->EnsureCollection(instanceName);
    const std::string text = path.GetText();
    return std::find(prim->cachedCollectionPaths.begin(),
                     prim->cachedCollectionPaths.end(),
                     text) != prim->cachedCollectionPaths.end()
        ? 1
        : 0;
}

// --- Paths ---

static NanousdPath be_path_parse(const char* text) {
    if (!text) return nullptr;
    auto p = Path::Parse(text);
    if (p.IsEmpty() && std::string_view(text) != "/") return nullptr;
    auto* h = new NanousdPath_s;
    h->path = std::move(p);
    return h;
}

static const char* be_path_str(NanousdPath path) {
    if (!path) return "";
    path->cachedText = path->path.GetText();
    return path->cachedText.c_str();
}

static NanousdPath be_path_append_child(NanousdPath parent, const char* child) {
    if (!parent || !child) return nullptr;
    auto p = parent->path.AppendChild(child);
    if (p.IsEmpty()) return nullptr;
    auto* h = new NanousdPath_s;
    h->path = std::move(p);
    return h;
}

static NanousdPath be_path_append_property(NanousdPath prim, const char* prop) {
    if (!prim || !prop) return nullptr;
    auto p = prim->path.AppendProperty(prop);
    if (p.IsEmpty()) return nullptr;
    auto* h = new NanousdPath_s;
    h->path = std::move(p);
    return h;
}

static NanousdPath be_path_parent(NanousdPath path) {
    if (!path) return nullptr;
    auto p = path->path.GetParentPath();
    if (p.IsEmpty()) return nullptr;
    auto* h = new NanousdPath_s;
    h->path = std::move(p);
    return h;
}

static const char* be_path_name(NanousdPath path) {
    if (!path) return "";
    path->cachedName = path->path.GetName().GetString();
    return path->cachedName.c_str();
}

static int be_path_is_absolute(NanousdPath path) {
    return (path && path->path.IsAbsolute()) ? 1 : 0;
}

static int be_path_is_root(NanousdPath path) {
    return (path && path->path.IsAbsoluteRoot()) ? 1 : 0;
}

static int be_path_is_property(NanousdPath path) {
    return (path && path->path.IsPropertyPath()) ? 1 : 0;
}

static int be_path_equal(NanousdPath a, NanousdPath b) {
    if (!a || !b) return (a == b) ? 1 : 0;
    return (a->path == b->path) ? 1 : 0;
}

static void be_path_free(NanousdPath path) { delete path; }

// --- ListOps ---

static NanousdListOp be_listop_create_explicit(const char** items, int count) {
    std::vector<std::string> v;
    for (int i = 0; i < count; ++i) {
        v.push_back(items[i] ? items[i] : "");
    }
    auto* h = new NanousdListOp_s;
    h->listop = ListOp<std::string>::CreateExplicit(std::move(v));
    return h;
}

static NanousdListOp be_listop_create(const char** prepend, int nprepend,
                                     const char** append, int nappend,
                                     const char** delete_, int ndelete) {
    auto toVec = [](const char** arr, int n) {
        std::vector<std::string> v;
        for (int i = 0; i < n; ++i) v.push_back(arr[i] ? arr[i] : "");
        return v;
    };
    auto* h = new NanousdListOp_s;
    h->listop = ListOp<std::string>::CreateComposable(
        toVec(prepend, nprepend),
        toVec(append, nappend),
        toVec(delete_, ndelete));
    return h;
}

static void be_listop_free(NanousdListOp op) { delete op; }

static int be_listop_is_explicit(NanousdListOp op) {
    return (op && op->listop.IsExplicit()) ? 1 : 0;
}

static int be_listop_nitems(NanousdListOp op) {
    if (!op) return 0;
    op->EnsureItems();
    return static_cast<int>(op->cachedItems.size());
}

static const char* be_listop_item(NanousdListOp op, int index) {
    if (!op) return "";
    op->EnsureItems();
    if (index < 0 || index >= static_cast<int>(op->cachedItems.size())) return "";
    return op->cachedItems[index].c_str();
}

static int be_listop_nprepended(NanousdListOp op) {
    return op ? static_cast<int>(op->listop.GetPrependedItems().size()) : 0;
}

static const char* be_listop_prepended(NanousdListOp op, int index) {
    if (!op) return "";
    auto& items = op->listop.GetPrependedItems();
    if (index < 0 || index >= static_cast<int>(items.size())) return "";
    return items[index].c_str();
}

static int be_listop_nappended(NanousdListOp op) {
    return op ? static_cast<int>(op->listop.GetAppendedItems().size()) : 0;
}

static const char* be_listop_appended(NanousdListOp op, int index) {
    if (!op) return "";
    auto& items = op->listop.GetAppendedItems();
    if (index < 0 || index >= static_cast<int>(items.size())) return "";
    return items[index].c_str();
}

static int be_listop_ndeleted(NanousdListOp op) {
    return op ? static_cast<int>(op->listop.GetDeletedItems().size()) : 0;
}

static const char* be_listop_deleted(NanousdListOp op, int index) {
    if (!op) return "";
    auto& items = op->listop.GetDeletedItems();
    if (index < 0 || index >= static_cast<int>(items.size())) return "";
    return items[index].c_str();
}

static NanousdListOp be_listop_combine(NanousdListOp stronger, NanousdListOp weaker) {
    if (!stronger || !weaker) return nullptr;
    auto* h = new NanousdListOp_s;
    h->listop = stronger->listop.Combine(weaker->listop);
    return h;
}

static NanousdListOp be_prim_listop(NanousdPrim prim, const char* field) {
    if (!prim || !field) return nullptr;
    Token fieldTok(field);

    // The C API's opaque NanousdListOp always exposes items as
    // strings. Internally a spec's listop may be stored as
    // ListOp<Token> (e.g. apiSchemas per §13.2.1.2) or ListOp<string>
    // (e.g. clipSets, variantSetNames). Read either and collapse
    // tokens to strings at this boundary.
    auto tokensToStrings = [](const ListOp<Token>& src) {
        ListOp<std::string> out;
        auto conv = [](const std::vector<Token>& in) {
            std::vector<std::string> r; r.reserve(in.size());
            for (const auto& t : in) r.push_back(t.GetString());
            return r;
        };
        if (src.IsExplicit()) {
            out.SetExplicitItems(conv(src.GetExplicitItems()));
        } else {
            out.SetPrependedItems(conv(src.GetPrependedItems()));
            out.SetAppendedItems(conv(src.GetAppendedItems()));
            out.SetDeletedItems(conv(src.GetDeletedItems()));
        }
        return out;
    };
    // References / payloads serialize as "@assetPath@<primPath>" so the
    // C boundary stays uniform (string per item). Empty primPath omits
    // the angle-bracket suffix. Composition tab in nuView consumes
    // these strings directly.
    auto refToString = [](const Reference& r) -> std::string {
        std::string s;
        if (r.assetPath) { s += "@"; s += *r.assetPath; s += "@"; }
        if (r.primPath && !r.primPath->IsEmpty()) {
            s += "<"; s += r.primPath->GetText(); s += ">";
        }
        return s;
    };
    auto refsToStrings = [&](const ListOp<Reference>& src) {
        ListOp<std::string> out;
        auto conv = [&](const std::vector<Reference>& in) {
            std::vector<std::string> r; r.reserve(in.size());
            for (const auto& ref : in) r.push_back(refToString(ref));
            return r;
        };
        if (src.IsExplicit()) {
            out.SetExplicitItems(conv(src.GetExplicitItems()));
        } else {
            out.SetPrependedItems(conv(src.GetPrependedItems()));
            out.SetAppendedItems(conv(src.GetAppendedItems()));
            out.SetDeletedItems(conv(src.GetDeletedItems()));
        }
        return out;
    };
    auto pathsToStrings = [](const ListOp<Path>& src) {
        ListOp<std::string> out;
        auto conv = [](const std::vector<Path>& in) {
            std::vector<std::string> r; r.reserve(in.size());
            for (const auto& p : in) r.push_back(p.GetText());
            return r;
        };
        if (src.IsExplicit()) {
            out.SetExplicitItems(conv(src.GetExplicitItems()));
        } else {
            out.SetPrependedItems(conv(src.GetPrependedItems()));
            out.SetAppendedItems(conv(src.GetAppendedItems()));
            out.SetDeletedItems(conv(src.GetDeletedItems()));
        }
        return out;
    };
    auto readOpinion = [&](const Spec* spec) -> std::optional<ListOp<std::string>> {
        if (!spec) return std::nullopt;
        auto* val = spec->GetField(fieldTok);
        if (!val) return std::nullopt;
        if (auto* lop = val->Get<ListOp<std::string>>()) return *lop;
        if (auto* lop = val->Get<ListOp<Token>>())        return tokensToStrings(*lop);
        if (auto* lop = val->Get<ListOp<Reference>>())    return refsToStrings(*lop);
        if (auto* lop = val->Get<ListOp<Path>>())         return pathsToStrings(*lop);
        return std::nullopt;
    };

    // If the prim is graph-aware, combine ListOps across the opinion stack
    if (prim->prim.GetPath().IsEmpty()) return nullptr;
    const auto& graph = prim->stage->stage->GetGraph();
    const auto* primIndex = graph.GetPrimIndex(prim->prim.GetPath());
    if (primIndex) {
        std::optional<ListOp<std::string>> combined;
        for (const auto& entry : primIndex->entries) {
            const auto& layer = graph.GetLayer(entry.layerIndex);
            Path sourcePath = entry.pathMapping->isIdentity ?
                prim->prim.GetPath() :
                entry.pathMapping->MapToSource(prim->prim.GetPath());
            if (sourcePath.IsEmpty()) continue;
            auto contribution = readOpinion(layer.GetPrimSpec(sourcePath));
            if (!contribution) continue;
            if (!combined) combined = std::move(*contribution);
            else           combined = combined->Combine(*contribution);
        }
        if (!combined) return nullptr;
        auto* h = new NanousdListOp_s;
        h->listop = *combined;
        return h;
    }

    // Fallback: single-layer mode
    auto contribution = readOpinion(
        prim->stage ? prim->GetMutableLayer().GetSpec(prim->prim.GetPath()) : nullptr);
    if (!contribution) return nullptr;
    auto* h = new NanousdListOp_s;
    h->listop = std::move(*contribution);
    return h;
}

// --- Vec/Matrix/Quaternion utilities ---

static float be_dot3f(const float a[3], const float b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static double be_dot3d(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static float be_length3f(const float v[3]) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

static double be_length3d(const double v[3]) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

static void be_normalize3f(const float v[3], float out[3]) {
    float len = be_length3f(v);
    if (len > 0.0f) {
        float inv = 1.0f / len;
        out[0] = v[0] * inv; out[1] = v[1] * inv; out[2] = v[2] * inv;
    } else {
        out[0] = out[1] = out[2] = 0.0f;
    }
}

static void be_normalize3d(const double v[3], double out[3]) {
    double len = be_length3d(v);
    if (len > 0.0) {
        double inv = 1.0 / len;
        out[0] = v[0] * inv; out[1] = v[1] * inv; out[2] = v[2] * inv;
    } else {
        out[0] = out[1] = out[2] = 0.0;
    }
}

static void be_cross3f(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static void be_cross3d(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static void be_mul_m4d(const double a[16], const double b[16], double out[16]) {
    // Row-major 4x4 matrix multiply
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k)
                sum += a[r*4+k] * b[k*4+c];
            out[r*4+c] = sum;
        }
    }
}

static void be_transform_point3d(const double m[16], const double p[3], double out[3]) {
    // Transform point by row-major 4x4 matrix (assumes w=1)
    for (int r = 0; r < 3; ++r) {
        out[r] = m[r*4+0]*p[0] + m[r*4+1]*p[1] + m[r*4+2]*p[2] + m[r*4+3];
    }
}

static void be_quat_slerp(const double a[4], const double b[4], double t, double out[4]) {
    // q = [real, i, j, k]
    double dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    double bSign[4] = { b[0], b[1], b[2], b[3] };
    if (dot < 0.0) {
        dot = -dot;
        for (int i = 0; i < 4; ++i) bSign[i] = -bSign[i];
    }
    if (dot > 0.9995) {
        // Linear interpolation for very close quaternions
        for (int i = 0; i < 4; ++i)
            out[i] = a[i] + t * (bSign[i] - a[i]);
        double len = std::sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2] + out[3]*out[3]);
        if (len > 0.0) for (int i = 0; i < 4; ++i) out[i] /= len;
        return;
    }
    double theta = std::acos(dot);
    double sinTheta = std::sin(theta);
    double wa = std::sin((1.0 - t) * theta) / sinTheta;
    double wb = std::sin(t * theta) / sinTheta;
    for (int i = 0; i < 4; ++i)
        out[i] = wa * a[i] + wb * bSign[i];
}

static void be_quat_to_matrix(const double q[4], double out[16]) {
    // q = [real, i, j, k] → row-major 4x4
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double xx = x*x, yy = y*y, zz = z*z;
    double xy = x*y, xz = x*z, yz = y*z;
    double wx = w*x, wy = w*y, wz = w*z;

    out[ 0] = 1.0 - 2.0*(yy+zz); out[ 1] = 2.0*(xy-wz);       out[ 2] = 2.0*(xz+wy);       out[ 3] = 0.0;
    out[ 4] = 2.0*(xy+wz);        out[ 5] = 1.0 - 2.0*(xx+zz); out[ 6] = 2.0*(yz-wx);        out[ 7] = 0.0;
    out[ 8] = 2.0*(xz-wy);        out[ 9] = 2.0*(yz+wx);        out[10] = 1.0 - 2.0*(xx+yy); out[11] = 0.0;
    out[12] = 0.0;                 out[13] = 0.0;                 out[14] = 0.0;                 out[15] = 1.0;
}

// ============================================================
// Write operations
// ============================================================

// Helper: get UsdAttribute for writing, creating spec if needed
static UsdAttribute GetWritableAttr(NanousdPrim_s* p, const char* name) {
    if (!p || !name || !p->stage) return {};
    auto attr = p->prim.GetAttribute(name);
    if (attr.IsValid()) return attr;
    // Attribute doesn't exist yet — caller must use create_attrib first
    return {};
}

static int be_set_attribf(NanousdPrim prim, const char* name, float value) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid()) return 0;
    if (!attr.Set(Value(Float(value)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribd(NanousdPrim prim, const char* name, double value) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid()) return 0;
    if (!attr.Set(Value(Double(value)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribi(NanousdPrim prim, const char* name, int value) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid()) return 0;
    if (!attr.Set(Value(Int(value)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribs(NanousdPrim prim, const char* name, const char* value) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !value) return 0;
    if (!attr.Set(Value(String(value)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribb(NanousdPrim prim, const char* name, int value) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid()) return 0;
    if (!attr.Set(Value(Bool(value != 0)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribi64(NanousdPrim prim, const char* name, int64_t value) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid()) return 0;
    if (!attr.Set(Value(Int64(value)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribv2f(NanousdPrim prim, const char* name, const float v[2]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.Set(Value(GfVec2f{v[0], v[1]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribv3f(NanousdPrim prim, const char* name, const float v[3]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.Set(Value(GfVec3f{v[0], v[1], v[2]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribv4f(NanousdPrim prim, const char* name, const float v[4]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.Set(Value(GfVec4f{v[0], v[1], v[2], v[3]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribv2d(NanousdPrim prim, const char* name, const double v[2]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.Set(Value(GfVec2d{v[0], v[1]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribv3d(NanousdPrim prim, const char* name, const double v[3]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.Set(Value(GfVec3d{v[0], v[1], v[2]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribv4d(NanousdPrim prim, const char* name, const double v[4]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.Set(Value(GfVec4d{v[0], v[1], v[2], v[3]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribv2i(NanousdPrim prim, const char* name, const int v[2]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.Set(Value(GfVec2i{v[0], v[1]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribv3i(NanousdPrim prim, const char* name, const int v[3]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.Set(Value(GfVec3i{v[0], v[1], v[2]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribv4i(NanousdPrim prim, const char* name, const int v[4]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.Set(Value(GfVec4i{v[0], v[1], v[2], v[3]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribm4d(NanousdPrim prim, const char* name, const double v[16]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    GfMatrix4d m;
    for (int i = 0; i < 16; ++i) m.data[i] = v[i];
    if (!attr.Set(Value(m))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribarrayf(NanousdPrim prim, const char* name, const float* data, int count) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !data || count < 0) return 0;
    std::vector<Float> arr(data, data + count);
    if (!attr.Set(Value(Value::ArrayTag{}, TypeId::Float, std::move(arr)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribarrayd(NanousdPrim prim, const char* name, const double* data, int count) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !data || count < 0) return 0;
    std::vector<Double> arr(data, data + count);
    if (!attr.Set(Value(Value::ArrayTag{}, TypeId::Double, std::move(arr)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribarrayi(NanousdPrim prim, const char* name, const int* data, int count) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !data || count < 0) return 0;
    std::vector<Int> arr(data, data + count);
    if (!attr.Set(Value(Value::ArrayTag{}, TypeId::Int, std::move(arr)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_samplef(NanousdPrim prim, const char* name, double time, float value) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid()) return 0;
    if (!attr.SetTimeSample(time, Value(Float(value)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_sampled(NanousdPrim prim, const char* name, double time, double value) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid()) return 0;
    if (!attr.SetTimeSample(time, Value(Double(value)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_samplev3f(NanousdPrim prim, const char* name, double time, const float v[3]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.SetTimeSample(time, Value(GfVec3f{v[0], v[1], v[2]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_samplev3d(NanousdPrim prim, const char* name, double time, const double v[3]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.SetTimeSample(time, Value(GfVec3d{v[0], v[1], v[2]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_samplev4f(NanousdPrim prim, const char* name, double time, const float v[4]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.SetTimeSample(time, Value(GfVec4f{v[0], v[1], v[2], v[3]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_sampleqf(NanousdPrim prim, const char* name, double time, const float v[4]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.SetTimeSample(time, Value(GfQuatf{v[0], v[1], v[2], v[3]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_sample_token(NanousdPrim prim, const char* name, double time, const char* value) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !value) return 0;
    if (!attr.SetTimeSample(time, Value(Token(value)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_samplearrayf(NanousdPrim prim, const char* name, double time, const float* data, int count) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !data || count < 0) return 0;
    std::vector<Float> arr(data, data + count);
    if (!attr.SetTimeSample(time, Value(Value::ArrayTag{}, TypeId::Float, std::move(arr)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_samplearrayi(NanousdPrim prim, const char* name, double time, const int* data, int count) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !data || count < 0) return 0;
    std::vector<Int> arr(data, data + count);
    if (!attr.SetTimeSample(time, Value(Value::ArrayTag{}, TypeId::Int, std::move(arr)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_samplearrayv3f(NanousdPrim prim, const char* name, double time, const float* data, int count) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !data || count < 0) return 0;
    std::vector<GfVec3f> arr(count);
    std::memcpy(arr.data(), data, count * sizeof(GfVec3f));
    if (!attr.SetTimeSample(time, Value(Value::ArrayTag{}, TypeId::Float3, std::move(arr)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_samplev2d(NanousdPrim prim, const char* name, double time, const double v[2]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.SetTimeSample(time, Value(GfVec2d{v[0], v[1]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_samplev4d(NanousdPrim prim, const char* name, double time, const double v[4]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    if (!attr.SetTimeSample(time, Value(GfVec4d{v[0], v[1], v[2], v[3]}))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_samplem4d(NanousdPrim prim, const char* name, double time, const double v[16]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    GfMatrix4d m;
    for (int i = 0; i < 16; ++i) m.data[i] = v[i];
    if (!attr.SetTimeSample(time, Value(m))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_samplearrayd(NanousdPrim prim, const char* name, double time, const double* data, int count) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !data || count < 0) return 0;
    std::vector<Double> arr(data, data + count);
    if (!attr.SetTimeSample(time, Value(Value::ArrayTag{}, TypeId::Double, std::move(arr)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_samplearrayv3d(NanousdPrim prim, const char* name, double time, const double* data, int count) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !data || count < 0) return 0;
    std::vector<GfVec3d> arr(count);
    std::memcpy(arr.data(), data, count * sizeof(GfVec3d));
    if (!attr.SetTimeSample(time, Value(Value::ArrayTag{}, TypeId::Double3, std::move(arr)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_clear_default(NanousdPrim prim, const char* name) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid()) return 0;
    if (!attr.ClearDefault()) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_clear_samples(NanousdPrim prim, const char* name) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid()) return 0;
    if (!attr.ClearTimeSamples()) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_block_attrib(NanousdPrim prim, const char* name) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid()) return 0;
    if (!attr.Block()) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_create_attrib(NanousdPrim prim, const char* name, const char* typeName) {
    if (!prim || !name || !typeName || !prim->stage) return 0;
    auto attr = prim->prim.CreateAttribute(name, Token(typeName));
    if (!attr.IsValid()) return 0;
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// Bulk array access (GPU-friendly zero-copy)
// ============================================================

static const float* be_arraydataf(NanousdPrim prim, const char* name, int* count) {
    auto* val = GetAttrDefaultPtr(prim, name);
    if (!val || !val->IsArray()) { if (count) *count = 0; return nullptr; }
    if (auto* v = val->Get<std::vector<Float>>()) {
        if (count) *count = static_cast<int>(v->size());
        return v->data();
    }
    // Vec2f array (e.g. texCoord2f[]) — zero-copy as flat float buffer
    if (auto* v = val->Get<std::vector<GfVec2f>>()) {
        if (count) *count = static_cast<int>(v->size()) * 2;
        return reinterpret_cast<const float*>(v->data());
    }
    // Vec3f array — zero-copy as flat float buffer
    if (auto* v = val->Get<std::vector<GfVec3f>>()) {
        if (count) *count = static_cast<int>(v->size()) * 3;
        return reinterpret_cast<const float*>(v->data());
    }
    // Vec4f array — zero-copy as flat float buffer
    if (auto* v = val->Get<std::vector<GfVec4f>>()) {
        if (count) *count = static_cast<int>(v->size()) * 4;
        return reinterpret_cast<const float*>(v->data());
    }
    if (count) *count = 0;
    return nullptr;
}

static const double* be_arraydatad(NanousdPrim prim, const char* name, int* count) {
    auto* val = GetAttrDefaultPtr(prim, name);
    if (!val || !val->IsArray()) { if (count) *count = 0; return nullptr; }
    if (auto* v = val->Get<std::vector<Double>>()) {
        if (count) *count = static_cast<int>(v->size());
        return v->data();
    }
    if (count) *count = 0;
    return nullptr;
}

static const int* be_arraydatai(NanousdPrim prim, const char* name, int* count) {
    auto* val = GetAttrDefaultPtr(prim, name);
    if (!val || !val->IsArray()) { if (count) *count = 0; return nullptr; }
    if (auto* v = val->Get<std::vector<Int>>()) {
        if (count) *count = static_cast<int>(v->size());
        return v->data();
    }
    if (count) *count = 0;
    return nullptr;
}

static int be_attribarrayv3f(NanousdPrim prim, const char* name, float* out, int maxcount) {
    auto val = GetAttrDefault(prim, name);
    if (!val || !out || maxcount <= 0 || !val->IsArray()) return -1;
    if (auto* v = val->Get<std::vector<GfVec3f>>()) {
        int n = std::min(maxcount, static_cast<int>(v->size()));
        std::memcpy(out, v->data(), n * sizeof(GfVec3f));
        return n;
    }
    return -1;
}

static int be_attribarrayv3d(NanousdPrim prim, const char* name, double* out, int maxcount) {
    auto val = GetAttrDefault(prim, name);
    if (!val || !out || maxcount <= 0 || !val->IsArray()) return -1;
    if (auto* v = val->Get<std::vector<GfVec3d>>()) {
        int n = std::min(maxcount, static_cast<int>(v->size()));
        std::memcpy(out, v->data(), n * sizeof(GfVec3d));
        return n;
    }
    return -1;
}

static int be_set_attribarrayv3f(NanousdPrim prim, const char* name, const float* data, int count) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !data || count < 0) return 0;
    std::vector<GfVec3f> arr(count);
    std::memcpy(arr.data(), data, count * sizeof(GfVec3f));
    if (!attr.Set(Value(Value::ArrayTag{}, TypeId::Float3, std::move(arr)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribarrayv3d(NanousdPrim prim, const char* name, const double* data, int count) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !data || count < 0) return 0;
    std::vector<GfVec3d> arr(count);
    std::memcpy(arr.data(), data, count * sizeof(GfVec3d));
    if (!attr.Set(Value(Value::ArrayTag{}, TypeId::Double3, std::move(arr)))) return 0;
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// P0 physics prerequisites
// ============================================================

// Quaternion read: internal storage is {i,j,k,r}, C API uses {w,i,j,k}
static int be_attribqf(NanousdPrim prim, const char* name, float out[4]) {
    auto val = GetAttrDefault(prim, name);
    if (!val) return 0;
    if (auto* q = val->Get<GfQuatf>()) {
        out[0] = (*q)[3];  // w (real)
        out[1] = (*q)[0];  // i
        out[2] = (*q)[1];  // j
        out[3] = (*q)[2];  // k
        return 1;
    }
    return 0;
}

static int be_attribqd(NanousdPrim prim, const char* name, double out[4]) {
    auto val = GetAttrDefault(prim, name);
    if (!val) return 0;
    if (auto* q = val->Get<GfQuatd>()) {
        out[0] = (*q)[3];  // w (real)
        out[1] = (*q)[0];  // i
        out[2] = (*q)[1];  // j
        out[3] = (*q)[2];  // k
        return 1;
    }
    // Try float quat and convert
    if (auto* q = val->Get<GfQuatf>()) {
        out[0] = static_cast<double>((*q)[3]);
        out[1] = static_cast<double>((*q)[0]);
        out[2] = static_cast<double>((*q)[1]);
        out[3] = static_cast<double>((*q)[2]);
        return 1;
    }
    return 0;
}

static int be_set_attribqf(NanousdPrim prim, const char* name, const float v[4]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid()) return 0;
    GfQuatf q;
    q[0] = v[1]; q[1] = v[2]; q[2] = v[3]; q[3] = v[0];  // i,j,k,r from w,i,j,k
    if (!attr.Set(Value(q))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribqd(NanousdPrim prim, const char* name, const double v[4]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid()) return 0;
    GfQuatd q;
    q[0] = v[1]; q[1] = v[2]; q[2] = v[3]; q[3] = v[0];  // i,j,k,r from w,i,j,k
    if (!attr.Set(Value(q))) return 0;
    prim->InvalidateCache();
    return 1;
}

// Relationship write
static int be_create_rel(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->stage) return 0;
    auto rel = prim->prim.CreateRelationship(name);
    if (!rel.IsValid()) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_reltargets(NanousdPrim prim, const char* name,
                              const char** targets, int count) {
    if (!prim || !name || !targets) return 0;
    auto rel = prim->prim.GetRelationship(name);
    if (!rel.IsValid()) return 0;
    std::vector<Path> paths;
    paths.reserve(count);
    for (int i = 0; i < count; ++i) {
        if (targets[i]) paths.push_back(Path::Parse(targets[i]));
    }
    return rel.SetTargets(paths) ? 1 : 0;
}

static int be_add_reltarget(NanousdPrim prim, const char* name, const char* target) {
    if (!prim || !name || !target) return 0;
    auto rel = prim->prim.GetRelationship(name);
    if (!rel.IsValid()) return 0;
    return rel.AddTarget(Path::Parse(target)) ? 1 : 0;
}

static int be_clear_reltargets(NanousdPrim prim, const char* name) {
    if (!prim || !name) return 0;
    auto rel = prim->prim.GetRelationship(name);
    if (!rel.IsValid()) return 0;
    return rel.ClearTargets() ? 1 : 0;
}

// Stage creation
static NanousdStage be_create(void) {
    auto* s = new NanousdStage_s;
    s->stage.emplace(Stage::CreateInMemory());
    return s;
}

// Prim creation
static NanousdPrim be_define_prim(NanousdStage stage, const char* path,
                                 const char* typeName) {
    if (!stage || !path || !stage->IsValid()) return nullptr;
    Token tn = (typeName && typeName[0]) ? Token(typeName) : Token();
    auto primObj = stage->stage->DefinePrim(Path::Parse(path), tn);
    stage->traversalDirty = true;
    stage->flatTraversalDirty = true;
    return MakePrimHandle(stage, std::move(primObj));
}

static Specifier SpecifierFromString(const char* s) {
    if (s) {
        if (strcmp(s, "over")  == 0) return Specifier::Over;
        if (strcmp(s, "class") == 0) return Specifier::Class;
    }
    return Specifier::Def;
}

static int be_set_specifier(NanousdPrim prim, const char* specifier) {
    if (!prim || !specifier || !prim->stage) return 0;
    auto* spec = const_cast<Spec*>(prim->GetMutableLayer().GetPrimSpec(prim->prim.GetPath()));
    if (!spec) return 0;
    spec->SetSpecifier(SpecifierFromString(specifier));
    if (prim->stage) prim->stage->traversalDirty = true;
    return 1;
}

static NanousdPrim be_define_prim_s(NanousdStage stage, const char* path,
                                   const char* typeName, const char* specifier) {
    NanousdPrim p = be_define_prim(stage, path, typeName);
    if (p && specifier) be_set_specifier(p, specifier);
    return p;
}

// Schema application
static int be_apply_api(NanousdPrim prim, const char* name) {
    if (!prim || !name) return 0;
    return prim->prim.ApplyAPI(name) ? 1 : 0;
}

// --- Composition arc write operations ---

static int be_add_reference(NanousdPrim prim, const char* assetPath,
                             const char* primPath) {
    if (!prim || !prim->stage || !prim->stage) return 0;
    if (!prim->stage->IsValid()) return 0;

    Path path = prim->prim.GetPath();
    auto* spec = const_cast<Spec*>(prim->GetMutableLayer().GetPrimSpec(path));
    if (!spec) return 0;

    // Build the reference
    Reference ref;
    if (assetPath && assetPath[0] != '\0') {
        ref.assetPath = Asset{std::string(assetPath)};
    }
    if (primPath && primPath[0] != '\0') {
        ref.primPath = Path::Parse(primPath);
    }

    // Get existing references listop or start fresh
    ListOp<Reference> listOp;
    const Value* existing = spec->GetField(FieldNames::references);
    if (existing) {
        if (auto* p = existing->Get<ListOp<Reference>>()) {
            listOp = *p;
        }
    }

    // Prepend the new reference
    auto items = listOp.GetPrependedItems();
    items.insert(items.begin(), ref);
    listOp.SetPrependedItems(std::move(items));
    spec->SetField(FieldNames::references, Value(std::move(listOp)));

    // Recompose to resolve the new arc
    if (!prim->stage->stage->Recompose()) return 0;

    // Refresh the prim handle — old graph pointers are stale.
    RefreshPrimHandle(prim, path);

    return 1;
}

// --- payload (same Reference shape; lazy-loaded arc per spec §6.3.5) ---

static int be_add_payload(NanousdPrim prim, const char* assetPath,
                           const char* primPath) {
    if (!prim || !prim->stage || !prim->stage->IsValid()) return 0;

    Path path = prim->prim.GetPath();
    auto* spec = const_cast<Spec*>(prim->GetMutableLayer().GetPrimSpec(path));
    if (!spec) return 0;

    Reference ref;  // Payload == Reference (per value.h)
    if (assetPath && assetPath[0] != '\0')
        ref.assetPath = Asset{std::string(assetPath)};
    if (primPath && primPath[0] != '\0')
        ref.primPath = Path::Parse(primPath);

    ListOp<Reference> listOp;
    if (auto* existing = spec->GetField(FieldNames::payload)) {
        if (auto* p = existing->Get<ListOp<Reference>>()) listOp = *p;
    }
    auto items = listOp.GetPrependedItems();
    items.insert(items.begin(), ref);
    listOp.SetPrependedItems(std::move(items));
    spec->SetField(FieldNames::payload, Value(std::move(listOp)));

    if (!prim->stage->stage->Recompose()) return 0;
    RefreshPrimHandle(prim, path);
    return 1;
}

// --- inherits / specializes (both ListOp<Path>) ---

static int be_add_listop_path(NanousdPrim prim, const Token& field,
                               const char* primPath) {
    if (!prim || !prim->stage || !prim->stage->IsValid()) return 0;
    if (!primPath || primPath[0] == '\0') return 0;

    Path here = prim->prim.GetPath();
    auto* spec = const_cast<Spec*>(prim->GetMutableLayer().GetPrimSpec(here));
    if (!spec) return 0;

    Path target = Path::Parse(primPath);
    if (!target.IsAbsolute()) return 0;

    ListOp<Path> listOp;
    if (auto* existing = spec->GetField(field)) {
        if (auto* p = existing->Get<ListOp<Path>>()) listOp = *p;
    }
    auto items = listOp.GetPrependedItems();
    items.insert(items.begin(), target);
    listOp.SetPrependedItems(std::move(items));
    spec->SetField(field, Value(std::move(listOp)));

    if (!prim->stage->stage->Recompose()) return 0;
    RefreshPrimHandle(prim, here);
    return 1;
}

static int be_add_inherit(NanousdPrim prim, const char* primPath) {
    return be_add_listop_path(prim, FieldNames::inheritPaths, primPath);
}

static int be_add_specialize(NanousdPrim prim, const char* primPath) {
    return be_add_listop_path(prim, FieldNames::specializes, primPath);
}

// --- remove_listop_item (generalised remove across all listop fields) ---

namespace {
template <typename T>
bool RemoveFromListOp(ListOp<T>& op, int kind, int index) {
    auto erase_at = [index](std::vector<T>& v) -> bool {
        if (index < 0 || index >= static_cast<int>(v.size())) return false;
        v.erase(v.begin() + index);
        return true;
    };
    switch (kind) {
        case 0: { auto v = op.GetExplicitItems();  if (!erase_at(v)) return false;
                  op = ListOp<T>::CreateExplicit(std::move(v)); return true; }
        case 1: { auto v = op.GetPrependedItems(); if (!erase_at(v)) return false;
                  op.SetPrependedItems(std::move(v)); return true; }
        case 2: { auto v = op.GetAppendedItems();  if (!erase_at(v)) return false;
                  op.SetAppendedItems(std::move(v)); return true; }
        case 3: { auto v = op.GetDeletedItems();   if (!erase_at(v)) return false;
                  op.SetDeletedItems(std::move(v)); return true; }
        default: return false;
    }
}
}  // namespace

static int be_remove_listop_item(NanousdPrim prim, const char* field,
                                  int listOpKind, int index) {
    if (!prim || !prim->stage || !prim->stage->IsValid() || !field) return 0;
    if (listOpKind < 0 || listOpKind > 3 || index < 0) return 0;

    Path path = prim->prim.GetPath();
    auto* spec = const_cast<Spec*>(prim->GetMutableLayer().GetPrimSpec(path));
    if (!spec) return 0;

    Token fieldTok(field);
    auto* val = spec->GetField(fieldTok);
    if (!val) return 0;

    bool ok = false;
    if (auto* lop = val->Get<ListOp<Reference>>()) {
        ListOp<Reference> copy = *lop;
        if (RemoveFromListOp(copy, listOpKind, index)) {
            spec->SetField(fieldTok, Value(std::move(copy)));
            ok = true;
        }
    } else if (auto* lop = val->Get<ListOp<Path>>()) {
        ListOp<Path> copy = *lop;
        if (RemoveFromListOp(copy, listOpKind, index)) {
            spec->SetField(fieldTok, Value(std::move(copy)));
            ok = true;
        }
    } else if (auto* lop = val->Get<ListOp<Token>>()) {
        ListOp<Token> copy = *lop;
        if (RemoveFromListOp(copy, listOpKind, index)) {
            spec->SetField(fieldTok, Value(std::move(copy)));
            ok = true;
        }
    } else if (auto* lop = val->Get<ListOp<std::string>>()) {
        ListOp<std::string> copy = *lop;
        if (RemoveFromListOp(copy, listOpKind, index)) {
            spec->SetField(fieldTok, Value(std::move(copy)));
            ok = true;
        }
    }
    if (!ok) return 0;

    if (!prim->stage->stage->Recompose()) return 0;
    RefreshPrimHandle(prim, path);
    return 1;
}

// --- Prim-state writers ---

static int be_set_active(NanousdPrim prim, int active) {
    if (!prim || !prim->stage || !prim->stage->IsValid()) return 0;
    /* Prefer the live prim's path, but fall back to the retained handle path:
     * once a prim is deactivated it is masked from GetPrimAtPath, so prim->prim
     * is invalid and only prim->path still names the spec to reactivate. */
    Path path = prim->prim.IsValid() ? prim->prim.GetPath() : prim->path;
    if (path.IsEmpty()) return 0;
    auto* spec = const_cast<Spec*>(prim->GetMutableLayer().GetPrimSpec(path));
    if (!spec) return 0;
    spec->SetActive(active != 0);
    if (!prim->stage->stage->Recompose()) return 0;
    RefreshPrimHandle(prim, path);
    return 1;
}

static int be_set_instanceable(NanousdPrim prim, int instanceable) {
    if (!prim || !prim->stage || !prim->stage->IsValid()) return 0;
    Path path = prim->prim.GetPath();
    auto* spec = const_cast<Spec*>(prim->GetMutableLayer().GetPrimSpec(path));
    if (!spec) return 0;
    spec->SetInstanceable(instanceable != 0);
    if (!prim->stage->stage->Recompose()) return 0;
    RefreshPrimHandle(prim, path);
    return 1;
}

static int be_remove_api(NanousdPrim prim, const char* schemaName) {
    if (!prim || !prim->stage || !schemaName || !schemaName[0]) return 0;
    if (!prim->stage->IsValid()) return 0;
    Path path = prim->prim.GetPath();
    auto* spec = const_cast<Spec*>(prim->GetMutableLayer().GetPrimSpec(path));
    if (!spec) return 0;

    auto* val = spec->GetField(FieldNames::apiSchemas);
    if (!val) return 0;
    auto* lop = val->Get<ListOp<Token>>();
    if (!lop) return 0;
    ListOp<Token> copy = *lop;
    Token target(schemaName);

    bool changed = false;
    auto strip = [&](std::vector<Token>& v) {
        auto end = std::remove(v.begin(), v.end(), target);
        if (end != v.end()) { changed = true; v.erase(end, v.end()); }
    };
    auto exp = copy.GetExplicitItems();    strip(exp);
    auto pre = copy.GetPrependedItems();   strip(pre);
    auto app = copy.GetAppendedItems();    strip(app);
    if (!changed) return 0;  // schema not applied here
    if (copy.IsExplicit()) {
        copy = ListOp<Token>::CreateExplicit(std::move(exp));
    } else {
        copy.SetPrependedItems(std::move(pre));
        copy.SetAppendedItems(std::move(app));
    }
    spec->SetField(FieldNames::apiSchemas, Value(std::move(copy)));

    if (!prim->stage->stage->Recompose()) return 0;
    RefreshPrimHandle(prim, path);
    return 1;
}

static int be_remove_prim(NanousdPrim prim) {
    if (!prim || !prim->stage || !prim->stage->IsValid()) return 0;
    Path path = prim->prim.GetPath();
    if (path.IsAbsoluteRoot()) return 0;  // refuse to remove pseudo-root

    auto& layer = prim->GetMutableLayer();

    // Collect this spec + all descendants (child prims, properties,
    // variant-selection sub-paths). Path doesn't expose a HasPrefix
    // helper, so test via the textual representation: a descendant of
    // /Foo has text starting with "/Foo" followed by '/', '.', or '{'.
    const std::string prefix = path.GetText();
    std::vector<Path> doomed;
    layer.ForEachSpec([&](const Path& p, const Spec&) {
        if (p == path) { doomed.push_back(p); return; }
        const std::string& t = p.GetText();
        if (t.size() <= prefix.size()) return;
        if (t.compare(0, prefix.size(), prefix) != 0) return;
        char next = t[prefix.size()];
        if (next == '/' || next == '.' || next == '{')
            doomed.push_back(p);
    });
    if (doomed.empty()) return 0;

    for (auto& p : doomed) {
        layer.RemoveSpec(p);
    }
    if (!prim->stage->stage->Recompose()) return 0;

    // The prim is gone — invalidate the handle.
    prim->path = path;
    prim->prim = UsdPrim();
    prim->layerIndex = 0;
    prim->InvalidateCache();
    prim->stage->traversalDirty = true;
    return 1;
}

// --- Variant set authoring ---

static int be_create_variantset(NanousdPrim prim, const char* setName) {
    if (!prim || !prim->stage || !setName || !setName[0]) return 0;
    if (!prim->stage->IsValid()) return 0;

    Path path = prim->prim.GetPath();
    auto* spec = const_cast<Spec*>(prim->GetMutableLayer().GetPrimSpec(path));
    if (!spec) return 0;

    // variantSetNames is authored as ListOp<std::string>::CreateExplicit per
    // spec §11.2 and the existing parser convention.
    std::vector<std::string> names;
    if (auto* val = spec->GetField(FieldNames::variantSetNames)) {
        if (auto* lop = val->Get<ListOp<std::string>>()) {
            names = lop->GetExplicitItems();
            if (names.empty()) {
                // Tolerate listOp opinion stored as prepend/append (defensive).
                auto pre = lop->GetPrependedItems();
                auto app = lop->GetAppendedItems();
                names.insert(names.end(), pre.begin(), pre.end());
                names.insert(names.end(), app.begin(), app.end());
            }
        }
    }
    if (std::find(names.begin(), names.end(), setName) == names.end()) {
        names.emplace_back(setName);
        spec->SetField(FieldNames::variantSetNames,
                       Value(ListOp<std::string>::CreateExplicit(std::move(names))));
    }

    if (!prim->stage->stage->Recompose()) return 0;
    RefreshPrimHandle(prim, path);
    return 1;
}

static int be_create_variant(NanousdPrim prim, const char* setName,
                              const char* variantName) {
    if (!prim || !prim->stage || !setName || !setName[0] ||
        !variantName || !variantName[0]) return 0;
    if (!prim->stage->IsValid()) return 0;

    if (!be_create_variantset(prim, setName)) return 0;

    auto& layer = prim->GetMutableLayer();
    Path primPath = prim->prim.GetPath();
    Path varPath = primPath.AppendVariantSelection(Token(setName), Token(variantName));

    if (!layer.HasSpec(varPath)) {
        Spec varSpec(SpecType::Variant);
        varSpec.SetSpecifier(Specifier::Def);
        layer.SetSpec(varPath, std::move(varSpec));
    }

    if (!prim->stage->stage->Recompose()) return 0;
    RefreshPrimHandle(prim, primPath);
    return 1;
}

// --- P1: Matrix3d read/write ---

static int be_attribm3d(NanousdPrim prim, const char* name, double out[9]) {
    auto val = GetAttrDefault(prim, name);
    if (!val) return 0;
    auto* m = val->Get<GfMatrix3d>();
    if (!m) return 0;
    for (int i = 0; i < 9; ++i) out[i] = m->data[i];
    return 1;
}

static int be_set_attribm3d(NanousdPrim prim, const char* name, const double v[9]) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !v) return 0;
    GfMatrix3d m;
    for (int i = 0; i < 9; ++i) m.data[i] = v[i];
    if (!attr.Set(Value(m))) return 0;
    prim->InvalidateCache();
    return 1;
}

// --- P1: String/token array read/write ---

static int be_attribarrays_len(NanousdPrim prim, const char* name) {
    auto val = GetAttrDefault(prim, name);
    if (!val || !val->IsArray()) return -1;
    if (auto* v = val->Get<std::vector<std::string>>()) return static_cast<int>(v->size());
    return -1;
}

static const char* be_attribarrays(NanousdPrim prim, const char* name, int index) {
    auto val = GetAttrDefault(prim, name);
    if (!val || !val->IsArray()) return nullptr;
    if (auto* v = val->Get<std::vector<std::string>>()) {
        if (index < 0 || index >= static_cast<int>(v->size())) return nullptr;
        prim->cachedStringVal = (*v)[index];
        return prim->cachedStringVal.c_str();
    }
    return nullptr;
}

static const char* be_attribarrays_elem(NanousdPrim prim, const char* name, int index) {
    return be_attribarrays(prim, name, index);
}

static int be_set_attribarrays(NanousdPrim prim, const char* name,
                                const char** strings, int count) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !strings || count < 0) return 0;
    std::vector<std::string> arr;
    arr.reserve(count);
    for (int i = 0; i < count; ++i) {
        arr.emplace_back(strings[i] ? strings[i] : "");
    }
    // Determine type from attribute's type name
    auto typeName = attr.GetTypeName().GetString();
    TypeId tid = TypeId::String;
    if (typeName.find("token") != std::string::npos) tid = TypeId::Token;
    else if (typeName.find("asset") != std::string::npos) tid = TypeId::Asset;
    if (!attr.Set(Value(Value::ArrayTag{}, tid, std::move(arr)))) return 0;
    prim->InvalidateCache();
    return 1;
}

// --- Token/Asset attribute setters ---

static int be_set_attrib_token(NanousdPrim prim, const char* name, const char* value) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !value) return 0;
    if (!attr.Set(Value(Token(value)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attrib_asset(NanousdPrim prim, const char* name, const char* value) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !value) return 0;
    if (!attr.Set(Value(Value::AssetTag{}, std::string(value)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribarraytokens(NanousdPrim prim, const char* name,
                                     const char** values, int count) {
    auto attr = GetWritableAttr(prim, name);
    if (!attr.IsValid() || !values || count < 0) return 0;
    std::vector<Token> arr;
    arr.reserve(count);
    for (int i = 0; i < count; ++i) {
        arr.emplace_back(values[i] ? values[i] : "");
    }
    if (!attr.Set(Value(Value::ArrayTag{}, TypeId::Token, std::move(arr)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_attribarraytokens_len(NanousdPrim prim, const char* name) {
    auto val = GetAttrDefault(prim, name);
    if (!val || !val->IsArray()) return -1;
    if (auto* v = val->Get<std::vector<Token>>()) return static_cast<int>(v->size());
    return -1;
}

static const char* be_attribarraytokens(NanousdPrim prim, const char* name, int index) {
    auto val = GetAttrDefault(prim, name);
    if (!val || !val->IsArray()) return nullptr;
    auto* v = val->Get<std::vector<Token>>();
    if (!v || index < 0 || index >= static_cast<int>(v->size())) return nullptr;
    prim->cachedStringVal = (*v)[index].GetString();
    return prim->cachedStringVal.c_str();
}

// --- P1: Asset path read ---

static const char* be_attribasset(NanousdPrim prim, const char* name, int* ok) {
    if (ok) *ok = 0;
    if (!prim || !name) return nullptr;
    auto attr = prim->prim.GetAttribute(name);
    if (!attr.IsValid()) return nullptr;
    auto resolved = attr.Get();
    if (!resolved.found) return nullptr;
    auto* s = resolved.value.Get<std::string>();
    if (!s) return nullptr;
    if (ok) *ok = 1;
    prim->cachedStringVal = *s;
    return prim->cachedStringVal.c_str();
}

// --- XformOp: composed local transform ---

static int be_get_local_transform(NanousdPrim prim, double time, double out[16],
                                   int* resetXformStack) {
    if (!prim || !out) return 0;
    if (resetXformStack) *resetXformStack = 0;

    UsdTimeCode tc = std::isnan(time) ? UsdTimeCode::Default() : UsdTimeCode(time);
    bool reset = false;
    GfMatrix4d m = ComputeLocalTransform(prim->prim, tc, &reset);
    for (int i = 0; i < 16; ++i) out[i] = m.data[i];
    if (resetXformStack) *resetXformStack = reset ? 1 : 0;
    return 1;
}

static int be_write_usdc(NanousdStage stage, const char* filepath) {
    if (!stage || !stage->IsValid() || !filepath) return 0;
    Layer flat = FlattenStage(*stage->stage);
    return WriteUsdcFile(flat, filepath) ? 1 : 0;
}

static int be_write_usdz(NanousdStage stage, const char* filepath) {
    if (!stage || !stage->IsValid() || !filepath) return 0;
    Layer flat = FlattenStage(*stage->stage);
    return WriteUsdzFile(flat, filepath) ? 1 : 0;
}

static int be_write_usda(NanousdStage stage, const char* filepath) {
    if (!stage || !stage->IsValid() || !filepath) return 0;
    Layer flat = FlattenStage(*stage->stage);
    return WriteUsdaFile(flat, filepath) ? 1 : 0;
}

static const char* be_write_usda_string(NanousdStage stage) {
    if (!stage || !stage->IsValid()) return nullptr;
    Layer flat = FlattenStage(*stage->stage);
    std::string usda = WriteUsda(flat);
    if (usda.empty()) return nullptr;
    char* result = static_cast<char*>(malloc(usda.size() + 1));
    if (!result) return nullptr;
    memcpy(result, usda.c_str(), usda.size() + 1);
    return result;
}

// Prim metadata — generic getters/setters for scalar prim metadata fields.
// Token values are stored as Value(Token) per the spec (e.g. kind, purpose).

static const char* be_prim_metadatas(NanousdPrim prim, const char* key, int* ok) {
    if (!prim || !key) { if (ok) *ok = 0; return ""; }
    auto val = prim->prim.GetPrimMetadata(Token(key));
    if (!val) { if (ok) *ok = 0; return ""; }
    if (auto* s = val->Get<String>()) { if (ok) *ok = 1; prim->cachedStringVal = *s; return prim->cachedStringVal.c_str(); }
    if (auto* t = val->Get<Token>()) { if (ok) *ok = 1; prim->cachedStringVal = t->GetString(); return prim->cachedStringVal.c_str(); }
    if (ok) *ok = 0;
    return "";
}

static double be_prim_metadatad(NanousdPrim prim, const char* key, int* ok) {
    if (!prim || !key) { if (ok) *ok = 0; return 0.0; }
    auto val = prim->prim.GetPrimMetadata(Token(key));
    if (!val) { if (ok) *ok = 0; return 0.0; }
    if (auto* d = val->Get<Double>()) { if (ok) *ok = 1; return *d; }
    if (auto* i = val->Get<Int>()) { if (ok) *ok = 1; return static_cast<double>(*i); }
    if (auto* f = val->Get<Float>()) { if (ok) *ok = 1; return static_cast<double>(*f); }
    if (ok) *ok = 0;
    return 0.0;
}

static int be_set_prim_metadatas(NanousdPrim prim, const char* key, const char* value) {
    if (!prim || !key || !value || !prim->stage) return 0;
    auto* spec = const_cast<Spec*>(prim->GetMutableLayer().GetPrimSpec(prim->prim.GetPath()));
    if (!spec) return 0;
    spec->SetField(Token(key), Value(String(value)));
    prim->InvalidateCache();
    return 1;
}

static int be_set_prim_metadatad(NanousdPrim prim, const char* key, double value) {
    if (!prim || !key || !prim->stage) return 0;
    auto* spec = const_cast<Spec*>(prim->GetMutableLayer().GetPrimSpec(prim->prim.GetPath()));
    if (!spec) return 0;
    spec->SetField(Token(key), Value(Double(value)));
    prim->InvalidateCache();
    return 1;
}

static int be_set_prim_metadata_token(NanousdPrim prim, const char* key, const char* value) {
    if (!prim || !key || !value || !prim->stage) return 0;
    auto* spec = const_cast<Spec*>(prim->GetMutableLayer().GetPrimSpec(prim->prim.GetPath()));
    if (!spec) return 0;
    spec->SetField(Token(key), Value(Token(value)));
    prim->InvalidateCache();
    return 1;
}

// Token scalar reader
static const char* be_attrib_token(NanousdPrim prim, const char* name, int* ok) {
    auto val = GetAttrDefault(prim, name);
    if (val) {
        if (auto* t = val->Get<Token>()) {
            if (ok) *ok = 1;
            prim->cachedStringVal = t->GetString();
            return prim->cachedStringVal.c_str();
        }
    }
    if (ok) *ok = 0;
    return "";
}

// Schema registration
static int be_register_schemas_json(const char* json) {
    if (!json) return 0;
    std::string error;
    if (!SchemaRegistry::GetInstance().LoadFromJSON(json, &error)) {
        return 0;
    }
    return 1;
}

// ============================================================
// Attribute metadata & connections
// ============================================================

static const Spec* GetAttrSpec(NanousdPrim_s* p, const char* name) {
    if (!p || !name || !p->stage || !p->stage->stage) return nullptr;
    auto attr = p->prim.GetAttribute(Token(name));
    if (!attr.IsValid()) return nullptr;
    return attr.GetAuthoredSpec();
}

static const char* be_attrib_interpolation(NanousdPrim prim, const char* name) {
    if (!prim || !name) return nullptr;
    auto* spec = GetAttrSpec(prim, name);
    if (!spec) return nullptr;
    static const Token interpToken("interpolation");
    auto* field = spec->GetField(interpToken);
    if (!field) return nullptr;
    if (auto* s = field->Get<String>()) {
        prim->cachedStringVal = *s;
        return prim->cachedStringVal.c_str();
    }
    if (auto* t = field->Get<Token>()) {
        prim->cachedStringVal = t->GetString();
        return prim->cachedStringVal.c_str();
    }
    return nullptr;
}

static int be_attrib_authored(NanousdPrim prim, const char* name) {
    if (!prim || !name) return 0;
    auto attr = prim->prim.GetAttribute(Token(name));
    return attr.IsValid() && attr.HasAuthoredSpec() ? 1 : 0;
}

static const char* be_attrib_colorspace(NanousdPrim prim, const char* name, int* ok) {
    if (ok) *ok = 0;
    if (!prim || !name) return "";
    auto attr = prim->prim.GetAttribute(Token(name));
    if (!attr.IsValid() || !attr.HasColorSpace()) return "";
    Token colorSpace = attr.GetColorSpace();
    if (ok) *ok = 1;
    prim->cachedStringVal = colorSpace.GetString();
    return prim->cachedStringVal.c_str();
}

static const char* be_attrib_resolved_colorspace(NanousdPrim prim, const char* name) {
    if (!prim || !name) return "";
    auto attr = prim->prim.GetAttribute(Token(name));
    if (!attr.IsValid()) return "";
    prim->cachedStringVal = attr.ComputeColorSpaceName().GetString();
    return prim->cachedStringVal.c_str();
}

static int be_set_attrib_colorspace(NanousdPrim prim, const char* name,
                                    const char* colorSpace) {
    if (!prim || !name || !colorSpace) return 0;
    auto attr = prim->prim.GetAttribute(Token(name));
    if (!attr.IsValid()) return 0;
    if (!attr.SetColorSpace(Token(colorSpace))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_clear_attrib_colorspace(NanousdPrim prim, const char* name) {
    if (!prim || !name) return 0;
    auto attr = prim->prim.GetAttribute(Token(name));
    if (!attr.IsValid()) return 0;
    if (!attr.ClearColorSpace()) return 0;
    prim->InvalidateCache();
    return 1;
}

static const char* be_prim_resolved_colorspace(NanousdPrim prim) {
    if (!prim) return "";
    prim->cachedStringVal = prim->prim.ComputeColorSpaceName().GetString();
    return prim->cachedStringVal.c_str();
}

static const std::vector<std::string>& ResolveConnectionPaths(
    NanousdPrim_s* prim, const char* name)
{
    static const std::vector<std::string> empty;
    if (!prim || !name) return empty;

    // Goes through UsdAttribute::GetConnections so opinion-stack listop
    // combining (spec §12.2.6) is applied uniformly. The text cache is
    // refreshed each call — connection lists are typically small.
    auto attr = prim->prim.GetAttribute(name);
    if (!attr.IsValid()) return empty;
    auto paths = attr.GetConnections();
    prim->cachedConnectionPaths.clear();
    prim->cachedConnectionPaths.reserve(paths.size());
    for (const auto& p : paths) prim->cachedConnectionPaths.push_back(p.GetText());
    return prim->cachedConnectionPaths;
}

static int be_hasconnections(NanousdPrim prim, const char* name) {
    if (!prim || !name) return 0;
    auto attr = prim->prim.GetAttribute(name);
    return (attr.IsValid() && attr.HasConnections()) ? 1 : 0;
}

static int be_nconnections(NanousdPrim prim, const char* name) {
    auto& paths = ResolveConnectionPaths(prim, name);
    return static_cast<int>(paths.size());
}

static const char* be_connection(NanousdPrim prim, const char* name, int index) {
    auto& paths = ResolveConnectionPaths(prim, name);
    if (index < 0 || index >= static_cast<int>(paths.size())) return "";
    return paths[index].c_str();
}

static NanousdPrim be_parent(NanousdPrim prim) {
    if (!prim || !prim->stage) return nullptr;
    auto parentPath = PrimHandlePath(prim).GetParentPath();
    if (parentPath.IsEmpty() || parentPath.IsAbsoluteRoot()) return nullptr;
    auto* stagePtr = prim->stage;
    if (!stagePtr->stage.has_value()) return nullptr;
    return MakePrimHandleForPath(stagePtr, parentPath, prim->layerIndex);
}

// ============================================================
// Instancing
// ============================================================

static int be_stage_nprototypes(NanousdStage stage) {
    if (!stage || !stage->IsValid()) return 0;
    const auto& prototypes = stage->stage->GetPrototypePaths();
    return static_cast<int>(prototypes.size());
}

static NanousdPrim be_stage_prototype(NanousdStage stage, int index) {
    if (!stage || !stage->IsValid()) return nullptr;
    const auto& prototypes = stage->stage->GetPrototypePaths();
    if (index < 0 || index >= static_cast<int>(prototypes.size())) return nullptr;
    auto proto = stage->stage->GetPrimAtPath(prototypes[static_cast<size_t>(index)]);
    return MakePrimHandle(stage, std::move(proto));
}

static int be_isinstance(NanousdPrim prim) {
    return (prim && prim->prim.IsInstance()) ? 1 : 0;
}

static int be_isprototype(NanousdPrim prim) {
    if (!prim || !prim->stage || !prim->stage->stage) return 0;
    const Path path = PrimHandlePath(prim);
    for (const auto& protoPath : prim->stage->stage->GetPrototypePaths()) {
        if (path == protoPath) return 1;
    }
    return 0;
}

static int be_isinprototype(NanousdPrim prim) {
    if (!prim || !prim->stage || !prim->stage->stage) return 0;
    return PathIsInPrototypeNamespace(*prim->stage->stage, PrimHandlePath(prim))
        ? 1 : 0;
}

static int be_isinstanceproxy(NanousdPrim prim) {
    if (!prim || !prim->stage || !prim->stage->stage) return 0;
    const Path path = PrimHandlePath(prim);
    if (path.IsEmpty()) return 0;
    if (PathIsInPrototypeNamespace(*prim->stage->stage, path)) return 0;
    Path instanceRoot;
    return FindInstanceRootForPath(*prim->stage->stage, path, &instanceRoot, nullptr)
        ? 1 : 0;
}

static NanousdPrim be_prototype(NanousdPrim prim) {
    if (!prim || !prim->stage) return nullptr;
    auto proto = prim->prim.GetPrototype();
    return MakePrimHandle(prim->stage, std::move(proto), prim->layerIndex);
}

static NanousdPrim be_priminprototype(NanousdPrim prim) {
    if (!prim || !prim->stage || !prim->stage->stage) return nullptr;

    auto* stagePtr = prim->stage;
    const Stage& stage = *stagePtr->stage;
    const Path path = PrimHandlePath(prim);
    if (path.IsEmpty()) return nullptr;

    if (PathIsInPrototypeNamespace(stage, path)) {
        auto proto = stage.GetPrimAtPath(path);
        if (!proto.IsValid()) proto = prim->prim;
        return MakePrimHandle(stagePtr, std::move(proto), prim->layerIndex);
    }

    Path protoRoot = stage.GetPrototypePath(path);
    if (!protoRoot.IsEmpty()) {
        auto proto = stage.GetPrimAtPath(protoRoot);
        return MakePrimHandle(stagePtr, std::move(proto), prim->layerIndex);
    }

    Path instanceRoot;
    if (!FindInstanceRootForPath(stage, path, &instanceRoot, &protoRoot))
        return nullptr;

    Path protoPath = RemapDescendantPath(path, instanceRoot, protoRoot);
    if (protoPath.IsEmpty()) return nullptr;
    auto proto = stage.GetPrimAtPath(protoPath);
    return MakePrimHandle(stagePtr, std::move(proto), prim->layerIndex);
}

static int be_ninstances(NanousdPrim prim) {
    if (!prim) return 0;
    auto instances = prim->prim.GetInstances();
    return static_cast<int>(instances.size());
}

static NanousdPrim be_instance(NanousdPrim prim, int index) {
    if (!prim || !prim->stage) return nullptr;
    auto instances = prim->prim.GetInstances();
    if (index < 0 || index >= static_cast<int>(instances.size())) return nullptr;
    return MakePrimHandle(prim->stage, std::move(instances[index]), prim->layerIndex);
}

static int be_instance_key(NanousdPrim prim, char* out, size_t out_size) {
    if (!prim || !prim->stage || !prim->stage->stage || !prim->prim.IsValid()) {
        if (out && out_size > 0) out[0] = '\0';
        return 0;
    }
    const Stage& stage = *prim->stage->stage;
    const auto& graph = stage.GetGraph();

    Path instancePath = prim->prim.GetPath();
    if (stage.GetPrototypePath(instancePath).IsEmpty()) {
        if (out && out_size > 0) out[0] = '\0';
        return 0;
    }

    Path representative = graph.GetInstanceRepresentative(instancePath);
    if (!representative.IsEmpty()) instancePath = representative;

    const PrimIndex* primIndex = graph.GetPrimIndex(instancePath);
    if (!primIndex) {
        if (out && out_size > 0) out[0] = '\0';
        return 0;
    }

    std::ostringstream key;
    bool first = true;
    for (const auto& entry : primIndex->entries) {
        if (entry.pathMapping->targetPrimPath.IsEmpty()) continue;
        if (!PathIsAncestorOrEqual(instancePath, entry.pathMapping->targetPrimPath))
            continue;
        Path sourcePath = SourcePrimPathForEntry(entry, instancePath);
        if (sourcePath.IsEmpty()) continue;
        if (!first) key << ';';
        first = false;
        const std::string layerPath = entry.layerIndex < graph.layerPaths.size()
            ? graph.layerPaths[entry.layerIndex]
            : std::string();
        key << ArcTypeName(entry.arcType) << '|'
            << layerPath << '|'
            << sourcePath.GetText();
    }
    prim->cachedInstanceKey = key.str();
    return CopyStringToBuffer(prim->cachedInstanceKey, out, out_size);
}

static void EnsureCompositionArcs(NanousdPrim prim) {
    if (!prim || prim->compositionArcsCached) return;
    prim->cachedCompositionArcs.clear();
    prim->compositionArcsCached = true;
    if (!prim->stage || !prim->stage->stage || !prim->prim.IsValid()) return;

    const Stage& stage = *prim->stage->stage;
    const auto& graph = stage.GetGraph();
    const Path queryPath = prim->prim.GetPath();
    if (queryPath.IsEmpty()) return;

    const PrimIndex* primIndex = graph.GetPrimIndex(queryPath);
    if (!primIndex) return;

    prim->cachedCompositionArcs.reserve(primIndex->entries.size());
    for (const auto& entry : primIndex->entries) {
        CachedCompositionArc arc;
        arc.arcType = ArcTypeToInt(entry.arcType);
        arc.layerIndex = entry.layerIndex > static_cast<size_t>(INT_MAX)
            ? INT_MAX
            : static_cast<int>(entry.layerIndex);
        arc.offset = entry.retiming.offset;
        arc.scale = entry.retiming.scale;

        const Path targetPath = entry.pathMapping->targetPrimPath;
        const Path sourcePath = SourcePrimPathForEntry(entry, queryPath);

        if (!targetPath.IsEmpty()) {
            if (targetPath == queryPath)
                arc.flags |= NANOUSD_COMPOSITION_ARC_DIRECT;
            else if (PathIsStrictAncestor(targetPath, queryPath))
                arc.flags |= NANOUSD_COMPOSITION_ARC_ANCESTRAL;
            arc.targetPath = targetPath.GetText();
        }
        if (entry.pathMapping->isIdentity)
            arc.flags |= NANOUSD_COMPOSITION_ARC_IDENTITY_MAPPING;
        if (entry.pathMapping->fallbackIdentity)
            arc.flags |= NANOUSD_COMPOSITION_ARC_FALLBACK_IDENTITY;

        if (!sourcePath.IsEmpty()) {
            arc.sourcePath = sourcePath.GetText();
            if (entry.layerIndex < graph.GetNumLayers() &&
                graph.GetLayer(entry.layerIndex).GetPrimSpec(sourcePath)) {
                arc.flags |= NANOUSD_COMPOSITION_ARC_HAS_SOURCE_SPEC;
            }
        }
        if (entry.layerIndex < graph.layerPaths.size())
            arc.layerPath = graph.layerPaths[entry.layerIndex];

        prim->cachedCompositionArcs.push_back(std::move(arc));
    }
}

static int be_ncomposition_arcs(NanousdPrim prim) {
    EnsureCompositionArcs(prim);
    return prim ? static_cast<int>(prim->cachedCompositionArcs.size()) : 0;
}

static int be_composition_arc(NanousdPrim prim, int index,
                              NanousdCompositionArc* out) {
    if (!out) return 0;
    EnsureCompositionArcs(prim);
    if (!prim || index < 0 ||
        index >= static_cast<int>(prim->cachedCompositionArcs.size())) {
        return 0;
    }
    const auto& src = prim->cachedCompositionArcs[static_cast<size_t>(index)];
    out->struct_size = static_cast<int>(sizeof(NanousdCompositionArc));
    out->arc_type = src.arcType;
    out->flags = src.flags;
    out->layer_index = src.layerIndex;
    out->offset = src.offset;
    out->scale = src.scale;
    out->layer_path = src.layerPath.c_str();
    out->source_path = src.sourcePath.c_str();
    out->target_path = src.targetPath.c_str();
    return 1;
}

// ============================================================
// Diagnostics
// ============================================================

static NanousdDiagnostic* be_diagnostics(NanousdStage stage, int* count) {
    if (count) *count = 0;
    if (!stage || !stage->stage) return nullptr;

    const auto& diags = stage->stage->GetDiagnostics().GetAll();
    const int n = static_cast<int>(diags.size());
    if (count) *count = n;
    if (n == 0) return nullptr;

    auto* arr = static_cast<NanousdDiagnostic*>(
        malloc(sizeof(NanousdDiagnostic) * static_cast<size_t>(n)));
    if (!arr) { if (count) *count = 0; return nullptr; }

    for (int i = 0; i < n; ++i) {
        const auto& d = diags[static_cast<size_t>(i)];
        arr[i].severity   = static_cast<int>(d.severity);
        arr[i].category   = static_cast<int>(d.category);
        arr[i].message    = strdup(d.message.c_str());
        arr[i].prim_path  = strdup(d.primPath.c_str());
        arr[i].layer_path = strdup(d.layerPath.c_str());
        arr[i].asset_path = strdup(d.assetPath.c_str());
        arr[i].arc_type   = static_cast<int>(d.arcType);
    }
    return arr;
}

static void be_free_diagnostics(NanousdDiagnostic* diags, int count) {
    if (!diags) return;
    for (int i = 0; i < count; ++i) {
        free(const_cast<char*>(diags[i].message));
        free(const_cast<char*>(diags[i].prim_path));
        free(const_cast<char*>(diags[i].layer_path));
        free(const_cast<char*>(diags[i].asset_path));
    }
    free(diags);
}

static const char* be_diagnostics_json(NanousdStage stage) {
    if (!stage || !stage->stage) {
        char* empty = static_cast<char*>(malloc(3));
        if (empty) { memcpy(empty, "[]", 3); }
        return empty;
    }
    std::string json = stage->stage->GetDiagnostics().ToJson();
    char* result = static_cast<char*>(malloc(json.size() + 1));
    if (!result) return nullptr;
    memcpy(result, json.c_str(), json.size() + 1);
    return result;
}

// ============================================================
// Variants (spec §11.2)
// ============================================================

static int be_nvariantsets(NanousdPrim prim) {
    if (!prim || !prim->prim.IsValid()) return 0;
    return static_cast<int>(prim->prim.GetVariantSetNames().size());
}

static const char* be_variantsetname(NanousdPrim prim, int index) {
    if (!prim || !prim->prim.IsValid()) return "";
    auto names = prim->prim.GetVariantSetNames();
    if (index < 0 || index >= static_cast<int>(names.size())) return "";
    prim->cachedStringVal = names[static_cast<size_t>(index)].GetString();
    return prim->cachedStringVal.c_str();
}

static int be_hasvariantset(NanousdPrim prim, const char* setName) {
    if (!prim || !prim->prim.IsValid() || !setName) return 0;
    return prim->prim.HasVariantSet(Token(setName)) ? 1 : 0;
}

static int be_nvariants(NanousdPrim prim, const char* setName) {
    if (!prim || !prim->prim.IsValid() || !setName) return 0;
    return static_cast<int>(prim->prim.GetVariantNames(Token(setName)).size());
}

static const char* be_variantname(NanousdPrim prim, const char* setName, int index) {
    if (!prim || !prim->prim.IsValid() || !setName) return "";
    auto names = prim->prim.GetVariantNames(Token(setName));
    if (index < 0 || index >= static_cast<int>(names.size())) return "";
    prim->cachedStringVal = names[static_cast<size_t>(index)].GetString();
    return prim->cachedStringVal.c_str();
}

static const char* be_variantselection(NanousdPrim prim, const char* setName) {
    if (!prim || !prim->prim.IsValid() || !setName) return "";
    Token sel = prim->prim.GetVariantSelection(Token(setName));
    prim->cachedStringVal = sel.GetString();
    return prim->cachedStringVal.c_str();
}

static int be_setvariantselection(NanousdPrim prim, const char* setName,
                                   const char* variantName, int layerIndex) {
    if (!prim || !prim->prim.IsValid() || !setName) return 0;
    if (!prim->stage || !prim->stage->IsValid()) return 0;
    if (layerIndex < 0) return 0;
    Path path = prim->prim.GetPath();
    Token variant = variantName ? Token(variantName) : Token();
    if (!prim->prim.SetVariantSelection(Token(setName), variant,
                                        static_cast<size_t>(layerIndex)))
        return 0;
    // Recompose internally so the new selection takes effect on the composed
    // stage. Mirrors the mutate-spec / recompose / refresh-handle pattern of
    // be_add_reference — callers never invoke recompose themselves.
    if (!prim->stage->stage->Recompose()) return 0;
    RefreshPrimHandle(prim, path);
    return 1;
}

// ============================================================
// Backend vtable singleton
// ============================================================

static NanousdBackend_v1 s_backend = {
    /* --- Original v1 functions (do not reorder) --- */
    be_open,
    be_close,
    be_isvalid,
    be_error,
    be_timecodes_per_second,
    be_frames_per_second,
    be_start_time,
    be_end_time,
    be_nprims,
    be_prim,
    be_primpath,
    be_defaultprim,
    be_nchildren,
    be_child,
    be_childname,
    be_path,
    be_name,
    be_typename,
    be_kind,
    be_isactive,
    be_isdefined,
    be_isabstract,
    be_isinstanceable,
    be_prim_isvalid,
    be_nattribs,
    be_attribname,
    be_hasattrib,
    be_attribtype,
    be_attribf,
    be_attribd,
    be_attribi,
    be_attribs,
    be_attribv2f,
    be_attribv3f,
    be_attribv4f,
    be_attribv2d,
    be_attribv3d,
    be_attribv4d,
    be_attribv2i,
    be_attribv3i,
    be_attribv4i,
    be_attribm4d,
    be_attribarraylen,
    be_attribarrayf,
    be_attribarrayi,
    be_hassamples,
    be_nsamplekeys,
    be_samplekey,
    be_samplef,
    be_samplev3f,
    be_samplev3d,
    be_hasrel,
    be_freeprim,

    /* --- Extensions (appended to preserve v1 ABI) --- */
    be_metadatad,
    be_metadatas,
    be_set_stage_metadatad,
    be_set_stage_metadatas,
    be_isa,
    be_hasapi,
    be_attribi64,
    be_attribb,
    be_attribarrayd,
    be_sampled,
    be_nreltargets,
    be_reltarget,
    be_path_parse,
    be_path_str,
    be_path_append_child,
    be_path_append_property,
    be_path_parent,
    be_path_name,
    be_path_is_absolute,
    be_path_is_root,
    be_path_is_property,
    be_path_equal,
    be_path_free,
    be_listop_create_explicit,
    be_listop_create,
    be_listop_free,
    be_listop_is_explicit,
    be_listop_nitems,
    be_listop_item,
    be_listop_nprepended,
    be_listop_prepended,
    be_listop_nappended,
    be_listop_appended,
    be_listop_ndeleted,
    be_listop_deleted,
    be_listop_combine,
    be_prim_listop,
    be_dot3f,
    be_dot3d,
    be_length3f,
    be_length3d,
    be_normalize3f,
    be_normalize3d,
    be_cross3f,
    be_cross3d,
    be_mul_m4d,
    be_transform_point3d,
    be_quat_slerp,
    be_quat_to_matrix,

    /* --- Write operations --- */
    be_set_attribf,
    be_set_attribd,
    be_set_attribi,
    be_set_attribs,
    be_set_attribb,
    be_set_attribi64,
    be_set_attribv2f,
    be_set_attribv3f,
    be_set_attribv4f,
    be_set_attribv2d,
    be_set_attribv3d,
    be_set_attribv4d,
    be_set_attribv2i,
    be_set_attribv3i,
    be_set_attribv4i,
    be_set_attribm4d,
    be_set_attribarrayf,
    be_set_attribarrayd,
    be_set_attribarrayi,
    be_set_samplef,
    be_set_sampled,
    be_set_samplev3f,
    be_set_samplev3d,
    be_clear_default,
    be_clear_samples,
    be_block_attrib,
    be_create_attrib,

    /* --- Bulk array access (GPU-friendly) --- */
    be_arraydataf,
    be_arraydatad,
    be_arraydatai,
    be_attribarrayv3f,
    be_attribarrayv3d,
    be_set_attribarrayv3f,
    be_set_attribarrayv3d,

    /* --- P0 physics prerequisites --- */
    be_attribqf,
    be_attribqd,
    be_set_attribqf,
    be_set_attribqd,
    be_set_reltargets,
    be_add_reltarget,
    be_clear_reltargets,
    be_create,
    be_define_prim,
    be_apply_api,
    // P1 extensions
    be_attribm3d,
    be_set_attribm3d,
    be_attribarrays_len,
    be_attribarrays,
    be_set_attribarrays,
    be_attribasset,
    be_get_local_transform,

    /* --- Binary write --- */
    be_write_usdc,

    /* --- Array time sample reads (appended for ABI compat) --- */
    be_samplev2f,
    be_samplearrayf,
    be_samplearrayd,
    be_samplearrayi,

    /* --- Stage root layer path (appended for ABI compat) --- */
    be_stage_get_root_layer_path,

    /* --- Prim specifier write (appended for ABI compat) --- */
    be_define_prim_s,
    be_set_specifier,

    /* --- USDA text write (appended for ABI compat) --- */
    be_write_usda,
    be_write_usda_string,

    /* --- Token/Asset attribute setters (appended for ABI compat) --- */
    be_set_attrib_token,
    be_set_attrib_asset,
    be_set_attribarraytokens,
    be_attribarraytokens_len,
    be_attribarraytokens,

    /* --- Composition arc write operations (appended for ABI compat) --- */
    be_add_reference,

    /* --- Relationship creation (appended for ABI compat) --- */
    be_create_rel,

    /* --- Schema registration (appended for ABI compat) --- */
    be_register_schemas_json,

    /* --- Prim metadata (appended for ABI compat) --- */
    be_prim_metadatas,
    be_prim_metadatad,
    be_set_prim_metadatas,
    be_set_prim_metadatad,
    be_set_prim_metadata_token,
    be_attrib_token,

    /* --- Attribute metadata & connections (appended for ABI compat) --- */
    be_attrib_interpolation,
    be_attrib_authored,
    be_hasconnections,
    be_nconnections,
    be_connection,
    be_parent,
    be_attribarrayi64,
    be_attribarrays_elem,

    /* --- Time sample setters: extended types --- */
    be_set_samplev4f,
    be_set_sampleqf,
    be_set_sample_token,
    be_set_samplearrayf,
    be_set_samplearrayi,
    be_set_samplearrayv3f,

    /* --- Double-precision time sample setters --- */
    be_set_samplev2d,
    be_set_samplev4d,
    be_set_samplem4d,
    be_set_samplearrayd,
    be_set_samplearrayv3d,

    /* --- Instancing (appended for ABI compat) --- */
    be_isinstance,
    be_isprototype,
    be_isinprototype,
    be_prototype,
    be_ninstances,
    be_instance,

    /* --- Diagnostics (appended for ABI compat) --- */
    be_diagnostics,
    be_free_diagnostics,
    be_diagnostics_json,

    /* --- Variants (appended for ABI compat) --- */
    be_nvariantsets,
    be_variantsetname,
    be_hasvariantset,
    be_nvariants,
    be_variantname,
    be_variantselection,
    be_setvariantselection,

    /* --- Spec-correct typed stage metadata (appended for ABI compat) --- */
    be_set_stage_metadata_token,

    /* --- Composed-layer enumeration (appended for ABI compat) --- */
    be_stage_n_layers,
    be_stage_layer_path,

    /* --- Masked stage open (appended for ABI compat) --- */
    be_open_masked,

    /* --- Color-space resolution (appended for ABI compat) --- */
    be_attrib_colorspace,
    be_attrib_resolved_colorspace,
    be_set_attrib_colorspace,
    be_clear_attrib_colorspace,
    be_prim_resolved_colorspace,

    /* --- USDZ package write (appended for ABI compat) --- */
    be_write_usdz,

    /* --- CollectionAPI evaluation (appended for ABI compat) --- */
    be_collection_nmembers,
    be_collection_member,
    be_collection_contains,

    /* --- Property enumeration (appended for ABI compat) --- */
    be_nproperties,
    be_propertyname,
    be_property_is_attribute,
    be_property_is_relationship,

    /* --- Relationship authored-state query (appended for ABI compat) --- */
    be_rel_authored,

    /* --- Per-layer spec / opinion queries (appended for ABI compat) --- */
    be_layer_has_prim_spec,
    be_layer_has_attr_opinion,
    be_layer_attr_nsamples,
    be_layer_prim_listop,

    /* --- Sublayer enumeration & per-layer offset (appended for ABI compat) --- */
    be_layer_n_sublayers,
    be_layer_sublayer_path,
    be_layer_offset,

    /* --- Recomposition (appended for ABI compat) --- */
    be_recompose,

    /* --- Composition-arc authoring (appended for ABI compat) --- */
    be_add_payload,
    be_add_inherit,
    be_add_specialize,
    be_remove_listop_item,

    /* --- Prim-state writers (appended for ABI compat) --- */
    be_set_active,
    be_set_instanceable,
    be_remove_api,
    be_remove_prim,

    /* --- Variant set authoring (appended for ABI compat) --- */
    be_create_variantset,
    be_create_variant,

    /* --- Asset resolution and resource reads (appended for ABI compat) --- */
    be_resolve_asset_path,
    be_stage_resolve_asset_path,
    be_read_asset_bytes,
    be_free_bytes,

    /* --- Relationship metadata (appended for ABI compat) --- */
    be_rel_metadatas,

    /* --- Authored attribute enumeration (appended for ABI compat) --- */
    be_nauthored_attribs,
    be_authored_attribname,

    /* --- Flat traversal, instancing, and composition diagnostics --- */
    be_traverse_flat,
    be_stage_nprototypes,
    be_stage_prototype,
    be_isinstanceproxy,
    be_priminprototype,
    be_instance_key,
    be_ncomposition_arcs,
    be_composition_arc,
};

// ============================================================
// Entry point
// ============================================================

extern "C" NANOUSD_BACKEND_API NanousdBackend_v1* nanousd_create_backend_v1(void) {
    return &s_backend;
}
