// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/nanousd_backend.h"
#include "nanousd/nanousdapi.h"

// OpenUSD headers
#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/relationship.h>
#include <pxr/usd/usd/modelAPI.h>
#include <pxr/usd/usd/schemaRegistry.h>
#include <pxr/usd/usd/variantSets.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/listOp.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/gf/vec3i.h>
#include <pxr/base/gf/vec4i.h>
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/value.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usd/references.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/sdf/reference.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/base/tf/diagnostic.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

// ============================================================
// Internal handle structures (backend-owned, wrapping OpenUSD types)
// ============================================================

// Captured diagnostic from TfDiagnosticMgr during stage open.
struct CapturedDiag {
    int severity;   // 0=Info, 1=Warning, 2=Error
    std::string message;
};

struct NanousdStage_s {
    UsdStageRefPtr stage;
    std::string error;
    std::vector<UsdPrim> cachedTraversal;
    bool traversalDirty = true;

    mutable std::string cachedRootLayerPath;
    std::vector<CapturedDiag> diagnostics;  // from TfErrorMark during Open

    bool IsValid() const { return static_cast<bool>(stage); }

    void EnsureTraversal() {
        if (traversalDirty && IsValid()) {
            cachedTraversal.clear();
            for (auto prim : stage->Traverse()) {
                cachedTraversal.push_back(prim);
            }
            traversalDirty = false;
        }
    }
};

struct NanousdPrim_s {
    UsdPrim prim;
    NanousdStage_s* stage = nullptr;  // back-pointer for cache invalidation

    mutable std::string cachedPath;
    mutable std::string cachedName;
    mutable std::string cachedTypeName;
    mutable std::string cachedKind;
    mutable std::vector<std::string> cachedPropNames;
    mutable bool propNamesCached = false;
    mutable std::vector<std::string> cachedAuthoredAttrNames;
    mutable bool authoredAttrNamesCached = false;
    mutable std::string cachedStringVal;

    // GPU-friendly: cached array data for zero-copy pointer access
    mutable std::vector<float> cachedFloats;
    mutable std::vector<double> cachedDoubles;
    mutable std::vector<int> cachedInts;

    // Cached string array for attribarrays/attribarraytokens
    mutable std::vector<std::string> cachedStrings;

    // Cached connection paths for attribute connections
    mutable std::vector<std::string> cachedConnectionPaths;

    void EnsurePropNames() const {
        if (!propNamesCached) {
            cachedPropNames.clear();
            for (auto& prop : prim.GetProperties()) {
                cachedPropNames.push_back(prop.GetName().GetString());
            }
            propNamesCached = true;
        }
    }

    void EnsureAuthoredAttrNames() const {
        if (!authoredAttrNamesCached) {
            cachedAuthoredAttrNames.clear();
            for (auto& prop : prim.GetAuthoredProperties()) {
                if (prop.Is<UsdAttribute>()) {
                    cachedAuthoredAttrNames.push_back(prop.GetName().GetString());
                }
            }
            authoredAttrNamesCached = true;
        }
    }

    void InvalidateCache() {
        propNamesCached = false;
        cachedPropNames.clear();
        authoredAttrNamesCached = false;
        cachedAuthoredAttrNames.clear();
        if (stage) stage->traversalDirty = true;
    }
};

struct NanousdPath_s {
    SdfPath path;
    mutable std::string cachedText;
    mutable std::string cachedName;
};

struct NanousdListOp_s {
    SdfTokenListOp listop;
    mutable std::vector<std::string> cachedItems;
    mutable bool itemsCached = false;
    mutable std::string cachedStringVal;

    void EnsureItems() const {
        if (!itemsCached) {
            cachedItems.clear();
            // Apply the listop to get the resolved list
            SdfTokenListOp::ItemVector result;
            listop.ApplyOperations(&result);
            for (const auto& t : result) {
                cachedItems.push_back(t.GetString());
            }
            itemsCached = true;
        }
    }
};

// ============================================================
// Helpers
// ============================================================

static UsdAttribute FindAttr(NanousdPrim_s* p, const char* name) {
    if (!p || !name) return UsdAttribute();
    return p->prim.GetAttribute(TfToken(name));
}

// ============================================================
// Backend function implementations — Original v1
// ============================================================

static NanousdStage be_open(const char* filepath) {
    if (!filepath) return nullptr;
    auto* s = new NanousdStage_s;

    // Capture any TF diagnostics emitted during Open (missing sublayers,
    // unresolvable references, etc.)
    TfErrorMark mark;
    s->stage = UsdStage::Open(filepath);
    if (!s->IsValid()) {
        s->error = "Failed to open stage: " + std::string(filepath);
    }

    // Collect all TF errors/warnings that fired during Open
    for (auto it = mark.GetBegin(); it != mark.GetEnd(); ++it) {
        CapturedDiag d;
        if (it->GetDiagnosticCode() == TF_DIAGNOSTIC_WARNING_TYPE) {
            d.severity = 1;  // Warning
        } else {
            d.severity = 2;  // Error
        }
        d.message = it->GetCommentary();
        s->diagnostics.push_back(std::move(d));
    }
    mark.Clear();

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
    return (stage && stage->IsValid()) ? stage->stage->GetTimeCodesPerSecond() : 24.0;
}

static double be_frames_per_second(NanousdStage stage) {
    return (stage && stage->IsValid()) ? stage->stage->GetFramesPerSecond() : 24.0;
}

static double be_start_time(NanousdStage stage) {
    return (stage && stage->IsValid()) ? stage->stage->GetStartTimeCode() : 0.0;
}

static double be_end_time(NanousdStage stage) {
    return (stage && stage->IsValid()) ? stage->stage->GetEndTimeCode() : 0.0;
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
    auto* p = new NanousdPrim_s;
    p->prim = stage->cachedTraversal[index];
    p->stage = stage;
    return p;
}

static NanousdPrim be_primpath(NanousdStage stage, const char* path) {
    if (!stage || !path || !stage->IsValid()) return nullptr;
    auto prim = stage->stage->GetPrimAtPath(SdfPath(path));
    if (!prim.IsValid() || !prim.IsDefined() || !prim.IsActive()) return nullptr;
    auto* p = new NanousdPrim_s;
    p->prim = prim;
    p->stage = stage;
    return p;
}

static NanousdPrim be_defaultprim(NanousdStage stage) {
    if (!stage || !stage->IsValid()) return nullptr;
    auto prim = stage->stage->GetDefaultPrim();
    if (!prim.IsValid()) return nullptr;
    auto* p = new NanousdPrim_s;
    p->prim = prim;
    p->stage = stage;
    return p;
}

static int be_nchildren(NanousdPrim prim) {
    if (!prim || !prim->prim.IsValid()) return 0;
    int count = 0;
    for (auto child : prim->prim.GetChildren()) {
        (void)child;
        ++count;
    }
    return count;
}

static NanousdPrim be_child(NanousdPrim prim, int index) {
    if (!prim || !prim->prim.IsValid()) return nullptr;
    int i = 0;
    for (auto child : prim->prim.GetChildren()) {
        if (i == index) {
            auto* c = new NanousdPrim_s;
            c->prim = child;
            c->stage = prim->stage;
            return c;
        }
        ++i;
    }
    return nullptr;
}

static NanousdPrim be_childname(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->prim.IsValid()) return nullptr;
    auto child = prim->prim.GetChild(TfToken(name));
    if (!child.IsValid()) return nullptr;
    auto* c = new NanousdPrim_s;
    c->prim = child;
    c->stage = prim->stage;
    return c;
}

static const char* be_path(NanousdPrim prim) {
    if (!prim) return "";
    prim->cachedPath = prim->prim.GetPath().GetString();
    return prim->cachedPath.c_str();
}

static const char* be_name(NanousdPrim prim) {
    if (!prim) return "";
    prim->cachedName = prim->prim.GetName().GetString();
    return prim->cachedName.c_str();
}

static const char* be_typename(NanousdPrim prim) {
    if (!prim) return "";
    prim->cachedTypeName = prim->prim.GetTypeName().GetString();
    return prim->cachedTypeName.c_str();
}

static const char* be_kind(NanousdPrim prim) {
    if (!prim) return "";
    TfToken kindToken;
    UsdModelAPI(prim->prim).GetKind(&kindToken);
    prim->cachedKind = kindToken.GetString();
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
    return static_cast<int>(prim->prim.GetAttributes().size());
}

static const char* be_attribname(NanousdPrim prim, int index) {
    if (!prim || !prim->prim.IsValid()) return "";
    auto attrs = prim->prim.GetAttributes();
    if (index < 0 || index >= static_cast<int>(attrs.size())) return "";
    prim->cachedStringVal = attrs[index].GetName().GetString();
    return prim->cachedStringVal.c_str();
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
    if (!prim || !name) return 0;
    return prim->prim.HasAttribute(TfToken(name)) ? 1 : 0;
}

static const char* be_attribtype(NanousdPrim prim, const char* name) {
    if (!prim || !name) return "";
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return "";
    prim->cachedStringVal = attr.GetTypeName().GetAsToken().GetString();
    return prim->cachedStringVal.c_str();
}

static float be_attribf(NanousdPrim prim, const char* name, int* ok) {
    auto attr = FindAttr(prim, name);
    if (attr.IsValid()) {
        VtValue val;
        if (attr.Get(&val, UsdTimeCode::Default())) {
            if (val.IsHolding<float>()) {
                if (ok) *ok = 1;
                return val.UncheckedGet<float>();
            }
            if (val.IsHolding<double>()) {
                if (ok) *ok = 1;
                return static_cast<float>(val.UncheckedGet<double>());
            }
        }
    }
    if (ok) *ok = 0;
    return 0.0f;
}

static double be_attribd(NanousdPrim prim, const char* name, int* ok) {
    auto attr = FindAttr(prim, name);
    if (attr.IsValid()) {
        VtValue val;
        if (attr.Get(&val, UsdTimeCode::Default())) {
            if (val.IsHolding<double>()) {
                if (ok) *ok = 1;
                return val.UncheckedGet<double>();
            }
            if (val.IsHolding<float>()) {
                if (ok) *ok = 1;
                return static_cast<double>(val.UncheckedGet<float>());
            }
        }
    }
    if (ok) *ok = 0;
    return 0.0;
}

static int be_attribi(NanousdPrim prim, const char* name, int* ok) {
    auto attr = FindAttr(prim, name);
    if (attr.IsValid()) {
        VtValue val;
        if (attr.Get(&val, UsdTimeCode::Default())) {
            if (val.IsHolding<int>()) {
                if (ok) *ok = 1;
                return val.UncheckedGet<int>();
            }
        }
    }
    if (ok) *ok = 0;
    return 0;
}

static const char* be_attribs(NanousdPrim prim, const char* name, int* ok) {
    auto attr = FindAttr(prim, name);
    if (attr.IsValid()) {
        VtValue val;
        if (attr.Get(&val, UsdTimeCode::Default())) {
            if (val.IsHolding<std::string>()) {
                if (ok) *ok = 1;
                prim->cachedStringVal = val.UncheckedGet<std::string>();
                return prim->cachedStringVal.c_str();
            }
        }
    }
    if (ok) *ok = 0;
    return "";
}

template <typename VecT, int N, typename OutT>
static int ReadVecAttr(NanousdPrim_s* prim, const char* name, OutT* out) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return 0;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return 0;
    if (!val.IsHolding<VecT>()) return 0;
    const auto& v = val.UncheckedGet<VecT>();
    for (int i = 0; i < N; ++i)
        out[i] = static_cast<OutT>(v[i]);
    return 1;
}

static int be_attribv2f(NanousdPrim prim, const char* name, float out[2]) {
    return ReadVecAttr<GfVec2f, 2>(prim, name, out);
}
static int be_attribv3f(NanousdPrim prim, const char* name, float out[3]) {
    return ReadVecAttr<GfVec3f, 3>(prim, name, out);
}
static int be_attribv4f(NanousdPrim prim, const char* name, float out[4]) {
    return ReadVecAttr<GfVec4f, 4>(prim, name, out);
}
static int be_attribv2d(NanousdPrim prim, const char* name, double out[2]) {
    return ReadVecAttr<GfVec2d, 2>(prim, name, out);
}
static int be_attribv3d(NanousdPrim prim, const char* name, double out[3]) {
    return ReadVecAttr<GfVec3d, 3>(prim, name, out);
}
static int be_attribv4d(NanousdPrim prim, const char* name, double out[4]) {
    return ReadVecAttr<GfVec4d, 4>(prim, name, out);
}
static int be_attribv2i(NanousdPrim prim, const char* name, int out[2]) {
    return ReadVecAttr<GfVec2i, 2>(prim, name, out);
}
static int be_attribv3i(NanousdPrim prim, const char* name, int out[3]) {
    return ReadVecAttr<GfVec3i, 3>(prim, name, out);
}
static int be_attribv4i(NanousdPrim prim, const char* name, int out[4]) {
    return ReadVecAttr<GfVec4i, 4>(prim, name, out);
}

static int be_attribm4d(NanousdPrim prim, const char* name, double out[16]) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return 0;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return 0;
    if (!val.IsHolding<GfMatrix4d>()) return 0;
    const auto& m = val.UncheckedGet<GfMatrix4d>();
    const double* d = m.GetArray();
    for (int i = 0; i < 16; ++i) out[i] = d[i];
    return 1;
}

static int be_attribarraylen(NanousdPrim prim, const char* name) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return -1;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return -1;
    if (val.IsHolding<VtArray<float>>())
        return static_cast<int>(val.UncheckedGet<VtArray<float>>().size());
    if (val.IsHolding<VtArray<int>>())
        return static_cast<int>(val.UncheckedGet<VtArray<int>>().size());
    if (val.IsHolding<VtArray<double>>())
        return static_cast<int>(val.UncheckedGet<VtArray<double>>().size());
    return -1;
}

static int be_attribarrayf(NanousdPrim prim, const char* name, float* out, int maxlen) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid() || !out || maxlen <= 0) return -1;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return -1;
    if (val.IsHolding<VtArray<float>>()) {
        const auto& arr = val.UncheckedGet<VtArray<float>>();
        int n = std::min(maxlen, static_cast<int>(arr.size()));
        for (int i = 0; i < n; ++i) out[i] = arr[i];
        return n;
    }
    return -1;
}

static int be_attribarrayi(NanousdPrim prim, const char* name, int* out, int maxlen) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid() || !out || maxlen <= 0) return -1;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return -1;
    if (val.IsHolding<VtArray<int>>()) {
        const auto& arr = val.UncheckedGet<VtArray<int>>();
        int n = std::min(maxlen, static_cast<int>(arr.size()));
        for (int i = 0; i < n; ++i) out[i] = arr[i];
        return n;
    }
    return -1;
}

static int be_hassamples(NanousdPrim prim, const char* name) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return 0;
    return attr.GetNumTimeSamples() > 0 ? 1 : 0;
}

static int be_nsamplekeys(NanousdPrim prim, const char* name) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return 0;
    std::vector<double> times;
    attr.GetTimeSamples(&times);
    return static_cast<int>(times.size());
}

static double be_samplekey(NanousdPrim prim, const char* name, int index) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return 0.0;
    std::vector<double> times;
    attr.GetTimeSamples(&times);
    if (index < 0 || index >= static_cast<int>(times.size())) return 0.0;
    return times[index];
}

static float be_samplef(NanousdPrim prim, const char* name, double time, int* ok) {
    auto attr = FindAttr(prim, name);
    if (attr.IsValid()) {
        VtValue val;
        if (attr.Get(&val, UsdTimeCode(time))) {
            if (val.IsHolding<float>()) {
                if (ok) *ok = 1;
                return val.UncheckedGet<float>();
            }
            if (val.IsHolding<double>()) {
                if (ok) *ok = 1;
                return static_cast<float>(val.UncheckedGet<double>());
            }
        }
    }
    if (ok) *ok = 0;
    return 0.0f;
}

template <typename VecT, int N, typename OutT>
static int ReadSampleVecAttr(NanousdPrim_s* prim, const char* name, double time, OutT* out) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return 0;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode(time))) return 0;
    if (!val.IsHolding<VecT>()) return 0;
    const auto& v = val.UncheckedGet<VecT>();
    for (int i = 0; i < N; ++i) out[i] = static_cast<OutT>(v[i]);
    return 1;
}

static int be_samplev3f(NanousdPrim prim, const char* name, double time, float out[3]) {
    return ReadSampleVecAttr<GfVec3f, 3>(prim, name, time, out);
}

static int be_samplev3d(NanousdPrim prim, const char* name, double time, double out[3]) {
    return ReadSampleVecAttr<GfVec3d, 3>(prim, name, time, out);
}

static int be_hasrel(NanousdPrim prim, const char* name) {
    if (!prim || !name) return 0;
    return prim->prim.HasRelationship(TfToken(name)) ? 1 : 0;
}

static void be_freeprim(NanousdPrim prim) { delete prim; }

// ============================================================
// Extensions — Generic stage metadata
// ============================================================

static double be_metadatad(NanousdStage stage, const char* key, int* ok) {
    if (!stage || !key || !stage->IsValid()) { if (ok) *ok = 0; return 0.0; }
    VtValue val;
    if (stage->stage->GetMetadata(TfToken(key), &val)) {
        if (val.IsHolding<double>()) { if (ok) *ok = 1; return val.UncheckedGet<double>(); }
        if (val.IsHolding<int>()) { if (ok) *ok = 1; return static_cast<double>(val.UncheckedGet<int>()); }
        if (val.IsHolding<float>()) { if (ok) *ok = 1; return static_cast<double>(val.UncheckedGet<float>()); }
    }
    if (ok) *ok = 0;
    return 0.0;
}

static const char* be_metadatas(NanousdStage stage, const char* key, int* ok) {
    if (!stage || !key || !stage->IsValid()) { if (ok) *ok = 0; return ""; }
    VtValue val;
    if (stage->stage->GetMetadata(TfToken(key), &val)) {
        if (val.IsHolding<std::string>()) {
            if (ok) *ok = 1;
            stage->error = val.UncheckedGet<std::string>();
            return stage->error.c_str();
        }
        if (val.IsHolding<TfToken>()) {
            if (ok) *ok = 1;
            stage->error = val.UncheckedGet<TfToken>().GetString();
            return stage->error.c_str();
        }
    }
    if (ok) *ok = 0;
    return "";
}

// ============================================================
// Extensions — Set stage metadata
// ============================================================

static int be_set_stage_metadatad(NanousdStage stage, const char* key, double value) {
    if (!stage || !key || !stage->IsValid()) return 0;
    auto layer = stage->stage->GetRootLayer();
    if (!layer) return 0;
    layer->SetField(SdfPath::AbsoluteRootPath(), TfToken(key), VtValue(value));
    return 1;
}

static int be_set_stage_metadatas(NanousdStage stage, const char* key, const char* value) {
    if (!stage || !key || !value || !stage->IsValid()) return 0;
    auto layer = stage->stage->GetRootLayer();
    if (!layer) return 0;
    // Try as token first (upAxis etc.), then string
    layer->SetField(SdfPath::AbsoluteRootPath(), TfToken(key), VtValue(TfToken(value)));
    return 1;
}

// ============================================================
// Extensions — Schema queries
// ============================================================

static int be_isa(NanousdPrim prim, const char* typeName) {
    if (!prim || !typeName || !prim->prim.IsValid()) return 0;
    // Check if this prim's type derives from the given type
    TfType primType = UsdSchemaRegistry::GetTypeFromName(TfToken(prim->prim.GetTypeName()));
    TfType queryType = UsdSchemaRegistry::GetTypeFromName(TfToken(typeName));
    if (primType.IsUnknown() || queryType.IsUnknown()) {
        // Fallback: exact match on type name string
        return (prim->prim.GetTypeName() == TfToken(typeName)) ? 1 : 0;
    }
    return primType.IsA(queryType) ? 1 : 0;
}

static int be_hasapi(NanousdPrim prim, const char* apiName) {
    if (!prim || !apiName || !prim->prim.IsValid()) return 0;
    // Check applied schemas
    auto schemas = prim->prim.GetAppliedSchemas();
    TfToken apiToken(apiName);
    for (const auto& schema : schemas) {
        if (schema == apiToken) return 1;
        // Multi-apply: check prefix before ':'
        std::string s = schema.GetString();
        auto colonPos = s.find(':');
        if (colonPos != std::string::npos && s.substr(0, colonPos) == apiName)
            return 1;
    }
    return 0;
}

// ============================================================
// Extensions — Additional scalar reads
// ============================================================

static int64_t be_attribi64(NanousdPrim prim, const char* name, int* ok) {
    auto attr = FindAttr(prim, name);
    if (attr.IsValid()) {
        VtValue val;
        if (attr.Get(&val, UsdTimeCode::Default())) {
            if (val.IsHolding<int64_t>()) { if (ok) *ok = 1; return val.UncheckedGet<int64_t>(); }
            if (val.IsHolding<int>()) { if (ok) *ok = 1; return static_cast<int64_t>(val.UncheckedGet<int>()); }
        }
    }
    if (ok) *ok = 0;
    return 0;
}

static int be_attribb(NanousdPrim prim, const char* name, int* ok) {
    auto attr = FindAttr(prim, name);
    if (attr.IsValid()) {
        VtValue val;
        if (attr.Get(&val, UsdTimeCode::Default())) {
            if (val.IsHolding<bool>()) { if (ok) *ok = 1; return val.UncheckedGet<bool>() ? 1 : 0; }
            if (val.IsHolding<int>()) { if (ok) *ok = 1; return val.UncheckedGet<int>() ? 1 : 0; }
        }
    }
    if (ok) *ok = 0;
    return 0;
}

// ============================================================
// Extensions — Additional array reads
// ============================================================

static int be_attribarrayd(NanousdPrim prim, const char* name, double* out, int maxlen) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid() || !out || maxlen <= 0) return -1;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return -1;
    if (val.IsHolding<VtArray<double>>()) {
        const auto& arr = val.UncheckedGet<VtArray<double>>();
        int n = std::min(maxlen, static_cast<int>(arr.size()));
        for (int i = 0; i < n; ++i) out[i] = arr[i];
        return n;
    }
    return -1;
}

// ============================================================
// Extensions — Additional time sample reads
// ============================================================

static double be_sampled(NanousdPrim prim, const char* name, double time, int* ok) {
    auto attr = FindAttr(prim, name);
    if (attr.IsValid()) {
        VtValue val;
        if (attr.Get(&val, UsdTimeCode(time))) {
            if (val.IsHolding<double>()) { if (ok) *ok = 1; return val.UncheckedGet<double>(); }
            if (val.IsHolding<float>()) { if (ok) *ok = 1; return static_cast<double>(val.UncheckedGet<float>()); }
        }
    }
    if (ok) *ok = 0;
    return 0.0;
}

// ============================================================
// Extensions — Relationship targets
// ============================================================

static int be_nreltargets(NanousdPrim prim, const char* name) {
    if (!prim || !name) return 0;
    auto rel = prim->prim.GetRelationship(TfToken(name));
    if (!rel.IsValid()) return 0;
    SdfPathVector targets;
    rel.GetTargets(&targets);
    return static_cast<int>(targets.size());
}

static const char* be_reltarget(NanousdPrim prim, const char* name, int index) {
    if (!prim || !name) return "";
    auto rel = prim->prim.GetRelationship(TfToken(name));
    if (!rel.IsValid()) return "";
    SdfPathVector targets;
    rel.GetTargets(&targets);
    if (index < 0 || index >= static_cast<int>(targets.size())) return "";
    prim->cachedStringVal = targets[index].GetString();
    return prim->cachedStringVal.c_str();
}

// ============================================================
// Extensions — Paths
// ============================================================

static NanousdPath be_path_parse(const char* text) {
    if (!text) return nullptr;
    SdfPath p(text);
    if (p.IsEmpty() && std::string(text) != "/") return nullptr;
    auto* h = new NanousdPath_s;
    h->path = std::move(p);
    return h;
}

static const char* be_path_str(NanousdPath path) {
    if (!path) return "";
    path->cachedText = path->path.GetString();
    return path->cachedText.c_str();
}

static NanousdPath be_path_append_child(NanousdPath parent, const char* child) {
    if (!parent || !child) return nullptr;
    auto p = parent->path.AppendChild(TfToken(child));
    if (p.IsEmpty()) return nullptr;
    auto* h = new NanousdPath_s;
    h->path = std::move(p);
    return h;
}

static NanousdPath be_path_append_property(NanousdPath prim, const char* prop) {
    if (!prim || !prop) return nullptr;
    auto p = prim->path.AppendProperty(TfToken(prop));
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
    path->cachedName = path->path.GetName();
    return path->cachedName.c_str();
}

static int be_path_is_absolute(NanousdPath path) {
    return (path && path->path.IsAbsolutePath()) ? 1 : 0;
}

static int be_path_is_root(NanousdPath path) {
    return (path && path->path.IsAbsoluteRootPath()) ? 1 : 0;
}

static int be_path_is_property(NanousdPath path) {
    return (path && path->path.IsPropertyPath()) ? 1 : 0;
}

static int be_path_equal(NanousdPath a, NanousdPath b) {
    if (!a || !b) return (a == b) ? 1 : 0;
    return (a->path == b->path) ? 1 : 0;
}

static void be_path_free(NanousdPath path) { delete path; }

// ============================================================
// Extensions — ListOps
// ============================================================

static NanousdListOp be_listop_create_explicit(const char** items, int count) {
    SdfTokenListOp::ItemVector v;
    for (int i = 0; i < count; ++i) {
        v.push_back(TfToken(items[i] ? items[i] : ""));
    }
    auto* h = new NanousdListOp_s;
    h->listop.SetExplicitItems(v);
    return h;
}

static NanousdListOp be_listop_create(const char** prepend, int nprepend,
                                     const char** append, int nappend,
                                     const char** delete_, int ndelete) {
    auto toVec = [](const char** arr, int n) {
        SdfTokenListOp::ItemVector v;
        for (int i = 0; i < n; ++i) v.push_back(TfToken(arr[i] ? arr[i] : ""));
        return v;
    };
    auto* h = new NanousdListOp_s;
    if (nprepend > 0) h->listop.SetPrependedItems(toVec(prepend, nprepend));
    if (nappend > 0)  h->listop.SetAppendedItems(toVec(append, nappend));
    if (ndelete > 0)  h->listop.SetDeletedItems(toVec(delete_, ndelete));
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
    op->cachedStringVal = items[index].GetString();
    return op->cachedStringVal.c_str();
}

static int be_listop_nappended(NanousdListOp op) {
    return op ? static_cast<int>(op->listop.GetAppendedItems().size()) : 0;
}

static const char* be_listop_appended(NanousdListOp op, int index) {
    if (!op) return "";
    auto& items = op->listop.GetAppendedItems();
    if (index < 0 || index >= static_cast<int>(items.size())) return "";
    op->cachedStringVal = items[index].GetString();
    return op->cachedStringVal.c_str();
}

static int be_listop_ndeleted(NanousdListOp op) {
    return op ? static_cast<int>(op->listop.GetDeletedItems().size()) : 0;
}

static const char* be_listop_deleted(NanousdListOp op, int index) {
    if (!op) return "";
    auto& items = op->listop.GetDeletedItems();
    if (index < 0 || index >= static_cast<int>(items.size())) return "";
    op->cachedStringVal = items[index].GetString();
    return op->cachedStringVal.c_str();
}

static NanousdListOp be_listop_combine(NanousdListOp stronger, NanousdListOp weaker) {
    if (!stronger || !weaker) return nullptr;
    auto* h = new NanousdListOp_s;
    if (stronger->listop.IsExplicit()) {
        // Explicit list in stronger layer completely overrides weaker
        h->listop = stronger->listop;
    } else {
        // Resolve by applying weaker first, then stronger on top
        SdfTokenListOp::ItemVector result;
        weaker->listop.ApplyOperations(&result);
        stronger->listop.ApplyOperations(&result);
        h->listop.SetExplicitItems(result);
    }
    return h;
}

static NanousdListOp be_prim_listop(NanousdPrim prim, const char* field) {
    if (!prim || !field || !prim->prim.IsValid()) return nullptr;
    // Combine ListOps across all opinion layers (strongest first)
    auto primStack = prim->prim.GetPrimStack();
    if (primStack.empty()) return nullptr;
    TfToken fieldTok(field);
    // Resolve by applying each layer's listop from weakest to strongest
    SdfTokenListOp::ItemVector result;
    bool found = false;
    for (auto it = primStack.rbegin(); it != primStack.rend(); ++it) {
        if (!(*it)->HasField(fieldTok)) continue;
        VtValue val = (*it)->GetField(fieldTok);
        if (!val.IsHolding<SdfTokenListOp>()) continue;
        auto listOp = val.UncheckedGet<SdfTokenListOp>();
        if (listOp.IsExplicit()) {
            // Explicit replaces everything seen so far
            result.clear();
        }
        listOp.ApplyOperations(&result);
        found = true;
    }
    if (!found) return nullptr;
    auto* h = new NanousdListOp_s;
    h->listop.SetExplicitItems(result);
    return h;
}

// ============================================================
// Extensions — Vec/Matrix/Quaternion utilities
// (These are pure math — identical to nanousd_backend.cpp)
// ============================================================

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
        out[0] = v[0]*inv; out[1] = v[1]*inv; out[2] = v[2]*inv;
    } else {
        out[0] = out[1] = out[2] = 0.0f;
    }
}

static void be_normalize3d(const double v[3], double out[3]) {
    double len = be_length3d(v);
    if (len > 0.0) {
        double inv = 1.0 / len;
        out[0] = v[0]*inv; out[1] = v[1]*inv; out[2] = v[2]*inv;
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
    for (int r = 0; r < 3; ++r) {
        out[r] = m[r*4+0]*p[0] + m[r*4+1]*p[1] + m[r*4+2]*p[2] + m[r*4+3];
    }
}

static void be_quat_slerp(const double a[4], const double b[4], double t, double out[4]) {
    double dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    double bSign[4] = { b[0], b[1], b[2], b[3] };
    if (dot < 0.0) {
        dot = -dot;
        for (int i = 0; i < 4; ++i) bSign[i] = -bSign[i];
    }
    if (dot > 0.9995) {
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

// Helper: set a typed value on a UsdAttribute via OpenUSD
template <typename T>
static int SetAttrTyped(NanousdPrim_s* p, const char* name, const T& value) {
    if (!p || !name || !p->prim.IsValid()) return 0;
    auto attr = p->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    if (!attr.Set(VtValue(value), UsdTimeCode::Default())) return 0;
    p->InvalidateCache();
    return 1;
}

static int be_set_attribf(NanousdPrim prim, const char* name, float value) {
    return SetAttrTyped(prim, name, value);
}
static int be_set_attribd(NanousdPrim prim, const char* name, double value) {
    return SetAttrTyped(prim, name, value);
}
static int be_set_attribi(NanousdPrim prim, const char* name, int value) {
    return SetAttrTyped(prim, name, value);
}
static int be_set_attribs(NanousdPrim prim, const char* name, const char* value) {
    if (!value) return 0;
    return SetAttrTyped(prim, name, std::string(value));
}
static int be_set_attribb(NanousdPrim prim, const char* name, int value) {
    return SetAttrTyped(prim, name, value != 0);
}
static int be_set_attribi64(NanousdPrim prim, const char* name, int64_t value) {
    return SetAttrTyped(prim, name, value);
}

template <typename VecT, int N>
static int SetVecAttr(NanousdPrim_s* prim, const char* name, const typename VecT::ScalarType* v) {
    if (!prim || !name || !v || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    VecT vec;
    for (int i = 0; i < N; ++i) vec[i] = v[i];
    if (!attr.Set(VtValue(vec), UsdTimeCode::Default())) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribv2f(NanousdPrim prim, const char* name, const float v[2]) {
    return SetVecAttr<GfVec2f, 2>(prim, name, v);
}
static int be_set_attribv3f(NanousdPrim prim, const char* name, const float v[3]) {
    return SetVecAttr<GfVec3f, 3>(prim, name, v);
}
static int be_set_attribv4f(NanousdPrim prim, const char* name, const float v[4]) {
    return SetVecAttr<GfVec4f, 4>(prim, name, v);
}
static int be_set_attribv2d(NanousdPrim prim, const char* name, const double v[2]) {
    return SetVecAttr<GfVec2d, 2>(prim, name, v);
}
static int be_set_attribv3d(NanousdPrim prim, const char* name, const double v[3]) {
    return SetVecAttr<GfVec3d, 3>(prim, name, v);
}
static int be_set_attribv4d(NanousdPrim prim, const char* name, const double v[4]) {
    return SetVecAttr<GfVec4d, 4>(prim, name, v);
}
static int be_set_attribv2i(NanousdPrim prim, const char* name, const int v[2]) {
    return SetVecAttr<GfVec2i, 2>(prim, name, v);
}
static int be_set_attribv3i(NanousdPrim prim, const char* name, const int v[3]) {
    return SetVecAttr<GfVec3i, 3>(prim, name, v);
}
static int be_set_attribv4i(NanousdPrim prim, const char* name, const int v[4]) {
    return SetVecAttr<GfVec4i, 4>(prim, name, v);
}

static int be_set_attribm4d(NanousdPrim prim, const char* name, const double v[16]) {
    if (!prim || !name || !v || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    GfMatrix4d m;
    // GfMatrix4d::Set takes double[4][4], reinterpret the flat array
    m.Set(reinterpret_cast<const double(*)[4]>(v));
    if (!attr.Set(VtValue(m), UsdTimeCode::Default())) return 0;
    prim->InvalidateCache();
    return 1;
}

template <typename T>
static int SetArrayAttr(NanousdPrim_s* prim, const char* name, const T* data, int count) {
    if (!prim || !name || !data || count < 0 || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    VtArray<T> arr(count);
    for (int i = 0; i < count; ++i) arr[i] = data[i];
    if (!attr.Set(VtValue(arr), UsdTimeCode::Default())) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribarrayf(NanousdPrim prim, const char* name, const float* data, int count) {
    return SetArrayAttr(prim, name, data, count);
}
static int be_set_attribarrayd(NanousdPrim prim, const char* name, const double* data, int count) {
    return SetArrayAttr(prim, name, data, count);
}
static int be_set_attribarrayi(NanousdPrim prim, const char* name, const int* data, int count) {
    return SetArrayAttr(prim, name, data, count);
}

template <typename T>
static int SetTimeSampleTyped(NanousdPrim_s* p, const char* name, double time, const T& value) {
    if (!p || !name || !p->prim.IsValid()) return 0;
    auto attr = p->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    if (!attr.Set(VtValue(value), UsdTimeCode(time))) return 0;
    p->InvalidateCache();
    return 1;
}

static int be_set_samplef(NanousdPrim prim, const char* name, double time, float value) {
    return SetTimeSampleTyped(prim, name, time, value);
}
static int be_set_sampled(NanousdPrim prim, const char* name, double time, double value) {
    return SetTimeSampleTyped(prim, name, time, value);
}
static int be_set_samplev3f(NanousdPrim prim, const char* name, double time, const float v[3]) {
    if (!v) return 0;
    return SetTimeSampleTyped(prim, name, time, GfVec3f(v[0], v[1], v[2]));
}
static int be_set_samplev3d(NanousdPrim prim, const char* name, double time, const double v[3]) {
    if (!v) return 0;
    return SetTimeSampleTyped(prim, name, time, GfVec3d(v[0], v[1], v[2]));
}

static int be_clear_default(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    attr.Clear();
    prim->InvalidateCache();
    return 1;
}

static int be_clear_samples(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    attr.ClearAtTime(UsdTimeCode::Default());
    // OpenUSD doesn't have a single "clear all time samples" — clear each
    std::vector<double> times;
    attr.GetTimeSamples(&times);
    for (double t : times) {
        attr.ClearAtTime(UsdTimeCode(t));
    }
    prim->InvalidateCache();
    return 1;
}

static int be_block_attrib(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    attr.Block();
    prim->InvalidateCache();
    return 1;
}

static int be_create_attrib(NanousdPrim prim, const char* name, const char* typeName) {
    if (!prim || !name || !typeName || !prim->prim.IsValid()) return 0;
    SdfValueTypeName sdfType = SdfSchema::GetInstance().FindType(typeName);
    if (!sdfType) return 0;
    auto attr = prim->prim.CreateAttribute(TfToken(name), sdfType, true);
    if (!attr.IsValid()) return 0;
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// Bulk array access (GPU-friendly)
// ============================================================

// OpenUSD stores VtArray which may COW — we need to cache the resolved value
// so the pointer remains valid. We stash it in the prim handle.

static const float* be_arraydataf(NanousdPrim prim, const char* name, int* count) {
    if (!prim || !name) { if (count) *count = 0; return nullptr; }
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) { if (count) *count = 0; return nullptr; }
    VtValue val;
    if (!attr.Get(&val, PXR_NS::UsdTimeCode::Default())) { if (count) *count = 0; return nullptr; }
    if (val.IsHolding<VtArray<float>>()) {
        const auto& arr = val.UncheckedGet<VtArray<float>>();
        int n = static_cast<int>(arr.size());
        // VtArray data may be invalidated, so copy into prim cache
        prim->cachedFloats.assign(arr.cdata(), arr.cdata() + n);
        if (count) *count = n;
        return prim->cachedFloats.data();
    }
    if (count) *count = 0;
    return nullptr;
}

static const double* be_arraydatad(NanousdPrim prim, const char* name, int* count) {
    if (!prim || !name) { if (count) *count = 0; return nullptr; }
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) { if (count) *count = 0; return nullptr; }
    VtValue val;
    if (!attr.Get(&val, PXR_NS::UsdTimeCode::Default())) { if (count) *count = 0; return nullptr; }
    if (val.IsHolding<VtArray<double>>()) {
        const auto& arr = val.UncheckedGet<VtArray<double>>();
        int n = static_cast<int>(arr.size());
        prim->cachedDoubles.assign(arr.cdata(), arr.cdata() + n);
        if (count) *count = n;
        return prim->cachedDoubles.data();
    }
    if (count) *count = 0;
    return nullptr;
}

static const int* be_arraydatai(NanousdPrim prim, const char* name, int* count) {
    if (!prim || !name) { if (count) *count = 0; return nullptr; }
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) { if (count) *count = 0; return nullptr; }
    VtValue val;
    if (!attr.Get(&val, PXR_NS::UsdTimeCode::Default())) { if (count) *count = 0; return nullptr; }
    if (val.IsHolding<VtArray<int>>()) {
        const auto& arr = val.UncheckedGet<VtArray<int>>();
        int n = static_cast<int>(arr.size());
        prim->cachedInts.assign(arr.cdata(), arr.cdata() + n);
        if (count) *count = n;
        return prim->cachedInts.data();
    }
    if (count) *count = 0;
    return nullptr;
}

static int be_attribarrayv3f(NanousdPrim prim, const char* name, float* out, int maxcount) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid() || !out || maxcount <= 0) return -1;
    VtValue val;
    if (!attr.Get(&val, PXR_NS::UsdTimeCode::Default())) return -1;
    if (val.IsHolding<VtArray<PXR_NS::GfVec3f>>()) {
        const auto& arr = val.UncheckedGet<VtArray<PXR_NS::GfVec3f>>();
        int n = std::min(maxcount, static_cast<int>(arr.size()));
        std::memcpy(out, arr.cdata(), n * sizeof(PXR_NS::GfVec3f));
        return n;
    }
    return -1;
}

static int be_attribarrayv3d(NanousdPrim prim, const char* name, double* out, int maxcount) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid() || !out || maxcount <= 0) return -1;
    VtValue val;
    if (!attr.Get(&val, PXR_NS::UsdTimeCode::Default())) return -1;
    if (val.IsHolding<VtArray<PXR_NS::GfVec3d>>()) {
        const auto& arr = val.UncheckedGet<VtArray<PXR_NS::GfVec3d>>();
        int n = std::min(maxcount, static_cast<int>(arr.size()));
        std::memcpy(out, arr.cdata(), n * sizeof(PXR_NS::GfVec3d));
        return n;
    }
    return -1;
}

static int be_set_attribarrayv3f(NanousdPrim prim, const char* name, const float* data, int count) {
    if (!prim || !name || !data || count < 0 || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    VtArray<PXR_NS::GfVec3f> arr(count);
    std::memcpy(arr.data(), data, count * sizeof(PXR_NS::GfVec3f));
    if (!attr.Set(VtValue(arr), PXR_NS::UsdTimeCode::Default())) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribarrayv3d(NanousdPrim prim, const char* name, const double* data, int count) {
    if (!prim || !name || !data || count < 0 || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    VtArray<PXR_NS::GfVec3d> arr(count);
    std::memcpy(arr.data(), data, count * sizeof(PXR_NS::GfVec3d));
    if (!attr.Set(VtValue(arr), PXR_NS::UsdTimeCode::Default())) return 0;
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// P0 physics prerequisites — Quaternions
// ============================================================

static int be_attribqf(NanousdPrim prim, const char* name, float out[4]) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid() || !out) return 0;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return 0;
    if (val.IsHolding<GfQuatf>()) {
        const auto& q = val.UncheckedGet<GfQuatf>();
        out[0] = q.GetReal();
        const auto& im = q.GetImaginary();
        out[1] = im[0]; out[2] = im[1]; out[3] = im[2];
        return 1;
    }
    if (val.IsHolding<GfQuath>()) {
        const auto& q = val.UncheckedGet<GfQuath>();
        out[0] = static_cast<float>(q.GetReal());
        const auto& im = q.GetImaginary();
        out[1] = static_cast<float>(im[0]);
        out[2] = static_cast<float>(im[1]);
        out[3] = static_cast<float>(im[2]);
        return 1;
    }
    return 0;
}

static int be_attribqd(NanousdPrim prim, const char* name, double out[4]) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid() || !out) return 0;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return 0;
    if (val.IsHolding<GfQuatd>()) {
        const auto& q = val.UncheckedGet<GfQuatd>();
        out[0] = q.GetReal();
        const auto& im = q.GetImaginary();
        out[1] = im[0]; out[2] = im[1]; out[3] = im[2];
        return 1;
    }
    if (val.IsHolding<GfQuatf>()) {
        const auto& q = val.UncheckedGet<GfQuatf>();
        out[0] = static_cast<double>(q.GetReal());
        const auto& im = q.GetImaginary();
        out[1] = static_cast<double>(im[0]);
        out[2] = static_cast<double>(im[1]);
        out[3] = static_cast<double>(im[2]);
        return 1;
    }
    return 0;
}

static int be_set_attribqf(NanousdPrim prim, const char* name, const float v[4]) {
    if (!prim || !name || !v || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    // C API: v = {w, i, j, k} -> GfQuatf(real=w, imaginary=(i,j,k))
    GfQuatf q(v[0], GfVec3f(v[1], v[2], v[3]));
    if (!attr.Set(VtValue(q), UsdTimeCode::Default())) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribqd(NanousdPrim prim, const char* name, const double v[4]) {
    if (!prim || !name || !v || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    // C API: v = {w, i, j, k} -> GfQuatd(real=w, imaginary=(i,j,k))
    GfQuatd q(v[0], GfVec3d(v[1], v[2], v[3]));
    if (!attr.Set(VtValue(q), UsdTimeCode::Default())) return 0;
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// Relationship write
// ============================================================

static int be_set_reltargets(NanousdPrim prim, const char* name,
                              const char** targets, int count) {
    if (!prim || !name || !prim->prim.IsValid()) return 0;
    auto rel = prim->prim.GetRelationship(TfToken(name));
    if (!rel.IsValid()) return 0;
    SdfPathVector pathVec;
    for (int i = 0; i < count; ++i) {
        if (targets[i]) pathVec.push_back(SdfPath(targets[i]));
    }
    if (!rel.SetTargets(pathVec)) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_add_reltarget(NanousdPrim prim, const char* name, const char* target) {
    if (!prim || !name || !target || !prim->prim.IsValid()) return 0;
    auto rel = prim->prim.GetRelationship(TfToken(name));
    if (!rel.IsValid()) return 0;
    if (!rel.AddTarget(SdfPath(target))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_clear_reltargets(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->prim.IsValid()) return 0;
    auto rel = prim->prim.GetRelationship(TfToken(name));
    if (!rel.IsValid()) return 0;
    if (!rel.ClearTargets(true)) return 0;
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// Stage creation
// ============================================================

static NanousdStage be_create(void) {
    auto* s = new NanousdStage_s;
    s->stage = UsdStage::CreateInMemory();
    if (!s->IsValid()) {
        s->error = "Failed to create in-memory stage";
    }
    return s;
}

// ============================================================
// Prim creation
// ============================================================

static NanousdPrim be_define_prim(NanousdStage stage, const char* path, const char* typeName) {
    if (!stage || !path || !stage->IsValid()) return nullptr;
    TfToken typeToken(typeName ? typeName : "");
    auto usdPrim = stage->stage->DefinePrim(SdfPath(path), typeToken);
    if (!usdPrim.IsValid()) return nullptr;
    stage->traversalDirty = true;
    auto* p = new NanousdPrim_s;
    p->prim = usdPrim;
    p->stage = stage;
    return p;
}

// ============================================================
// Schema application
// ============================================================

static int be_apply_api(NanousdPrim prim, const char* schemaName) {
    if (!prim || !schemaName || !prim->prim.IsValid()) return 0;
    // Use the generic single-apply API schema application
    TfToken apiToken(schemaName);
    if (!prim->prim.ApplyAPI(apiToken)) return 0;
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// P1 extensions — Matrix3d read/write
// ============================================================

static int be_attribm3d(NanousdPrim prim, const char* name, double out[9]) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid() || !out) return 0;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return 0;
    if (!val.IsHolding<GfMatrix3d>()) return 0;
    const auto& m = val.UncheckedGet<GfMatrix3d>();
    const double* d = m.GetArray();
    for (int i = 0; i < 9; ++i) out[i] = d[i];
    return 1;
}

static int be_set_attribm3d(NanousdPrim prim, const char* name, const double v[9]) {
    if (!prim || !name || !v || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    GfMatrix3d m;
    m.Set(v[0], v[1], v[2],
          v[3], v[4], v[5],
          v[6], v[7], v[8]);
    if (!attr.Set(VtValue(m), UsdTimeCode::Default())) return 0;
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// P1 extensions — String/token array read/write
// ============================================================

static int be_attribarrays_len(NanousdPrim prim, const char* name) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return -1;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return -1;
    if (val.IsHolding<VtArray<std::string>>())
        return static_cast<int>(val.UncheckedGet<VtArray<std::string>>().size());
    return -1;
}

static const char* be_attribarrays(NanousdPrim prim, const char* name, int index) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return nullptr;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return nullptr;
    if (val.IsHolding<VtArray<std::string>>()) {
        const auto& arr = val.UncheckedGet<VtArray<std::string>>();
        if (index < 0 || index >= static_cast<int>(arr.size())) return nullptr;
        prim->cachedStringVal = arr[index];
        return prim->cachedStringVal.c_str();
    }
    return nullptr;
}

static int be_set_attribarrays(NanousdPrim prim, const char* name,
                                const char** strings, int count) {
    if (!prim || !name || !strings || count < 0 || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    VtArray<std::string> arr(count);
    for (int i = 0; i < count; ++i) arr[i] = strings[i] ? strings[i] : "";
    if (!attr.Set(VtValue(arr), UsdTimeCode::Default())) return 0;
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// P1 extensions — Asset path read
// ============================================================

static const char* be_attribasset(NanousdPrim prim, const char* name, int* ok) {
    auto attr = FindAttr(prim, name);
    if (attr.IsValid()) {
        VtValue val;
        if (attr.Get(&val, UsdTimeCode::Default())) {
            if (val.IsHolding<SdfAssetPath>()) {
                const auto& ap = val.UncheckedGet<SdfAssetPath>();
                if (ok) *ok = 1;
                // Prefer resolved path, fall back to asset path
                const std::string& resolved = ap.GetResolvedPath();
                prim->cachedStringVal = resolved.empty() ? ap.GetAssetPath() : resolved;
                return prim->cachedStringVal.c_str();
            }
        }
    }
    if (ok) *ok = 0;
    return nullptr;
}

// ============================================================
// P1 extensions — XformOp local transform
// ============================================================

static int be_get_local_transform(NanousdPrim prim, double time, double out[16],
                                   int* resetXformStack) {
    if (!prim || !out || !prim->prim.IsValid()) return 0;
    UsdGeomXformable xformable(prim->prim);
    if (!xformable) return 0;
    GfMatrix4d matrix;
    bool resetsStack = false;
    UsdTimeCode tc = std::isnan(time) ? UsdTimeCode::Default() : UsdTimeCode(time);
    if (!xformable.GetLocalTransformation(&matrix, &resetsStack, tc)) return 0;
    const double* d = matrix.GetArray();
    for (int i = 0; i < 16; ++i) out[i] = d[i];
    if (resetXformStack) *resetXformStack = resetsStack ? 1 : 0;
    return 1;
}

// ============================================================
// Binary write — USDC
// ============================================================

static int be_write_usdc(NanousdStage stage, const char* filepath) {
    if (!stage || !filepath || !stage->IsValid()) return 0;
    auto layer = stage->stage->GetRootLayer();
    if (!layer) return 0;
    return layer->Export(filepath) ? 1 : 0;
}

// ============================================================
// Array time sample reads
// ============================================================

static int be_samplev2f(NanousdPrim prim, const char* name, double time, float out[2]) {
    return ReadSampleVecAttr<GfVec2f, 2>(prim, name, time, out);
}

template <typename T>
static int ReadSampleArray(NanousdPrim_s* prim, const char* name, double time, T* out, int maxlen) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid() || !out || maxlen <= 0) return -1;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode(time))) return 0;
    if (!val.IsHolding<VtArray<T>>()) return -1;
    const auto& arr = val.UncheckedGet<VtArray<T>>();
    int n = std::min(maxlen, static_cast<int>(arr.size()));
    for (int i = 0; i < n; ++i) out[i] = arr[i];
    return n;
}

static int be_samplearrayf(NanousdPrim prim, const char* name, double time, float* out, int maxlen) {
    return ReadSampleArray(prim, name, time, out, maxlen);
}

static int be_samplearrayd(NanousdPrim prim, const char* name, double time, double* out, int maxlen) {
    return ReadSampleArray(prim, name, time, out, maxlen);
}

static int be_samplearrayi(NanousdPrim prim, const char* name, double time, int* out, int maxlen) {
    return ReadSampleArray(prim, name, time, out, maxlen);
}

// ============================================================
// Stage root layer path
// ============================================================

static const char* be_stage_get_root_layer_path(NanousdStage stage) {
    if (!stage || !stage->IsValid()) return "";
    auto layer = stage->stage->GetRootLayer();
    if (!layer) return "";
    stage->cachedRootLayerPath = layer->GetRealPath();
    return stage->cachedRootLayerPath.c_str();
}

// ============================================================
// Prim specifier write
// ============================================================

static SdfSpecifier ParseSpecifier(const char* spec) {
    if (!spec) return SdfSpecifierDef;
    std::string s(spec);
    if (s == "over") return SdfSpecifierOver;
    if (s == "class") return SdfSpecifierClass;
    return SdfSpecifierDef;
}

static NanousdPrim be_define_prim_s(NanousdStage stage, const char* path, const char* typeName,
                                   const char* specifier) {
    if (!stage || !path || !stage->IsValid()) return nullptr;
    TfToken typeToken(typeName ? typeName : "");
    SdfSpecifier spec = ParseSpecifier(specifier);

    UsdPrim usdPrim;
    if (spec == SdfSpecifierOver) {
        usdPrim = stage->stage->OverridePrim(SdfPath(path));
    } else if (spec == SdfSpecifierClass) {
        usdPrim = stage->stage->CreateClassPrim(SdfPath(path));
    } else {
        usdPrim = stage->stage->DefinePrim(SdfPath(path), typeToken);
    }
    if (!usdPrim.IsValid()) return nullptr;

    // Set type name for non-def specifiers if provided
    if (spec != SdfSpecifierDef && typeName && typeName[0]) {
        auto primSpec = stage->stage->GetRootLayer()->GetPrimAtPath(usdPrim.GetPath());
        if (primSpec) {
            primSpec->SetTypeName(typeName);
        }
    }

    stage->traversalDirty = true;
    auto* p = new NanousdPrim_s;
    p->prim = usdPrim;
    p->stage = stage;
    return p;
}

static int be_set_specifier(NanousdPrim prim, const char* specifier) {
    if (!prim || !specifier || !prim->prim.IsValid() || !prim->stage) return 0;
    auto primSpec = prim->stage->stage->GetRootLayer()->GetPrimAtPath(prim->prim.GetPath());
    if (!primSpec) return 0;
    primSpec->SetSpecifier(ParseSpecifier(specifier));
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// USDA text write
// ============================================================

static int be_write_usda(NanousdStage stage, const char* filepath) {
    if (!stage || !filepath || !stage->IsValid()) return 0;
    auto layer = stage->stage->GetRootLayer();
    if (!layer) return 0;
    return layer->Export(filepath) ? 1 : 0;
}

static const char* be_write_usda_string(NanousdStage stage) {
    if (!stage || !stage->IsValid()) return nullptr;
    // Match nanousd's semantics: return the FLATTENED stage text, not
    // the root layer's raw text. Without this, multi-layer stages emit
    // their sublayer / reference arcs verbatim and downstream string
    // checks looking for composed values from weaker layers fail.
    auto flat = stage->stage->Flatten();
    if (!flat) return nullptr;
    std::string result;
    if (!flat->ExportToString(&result)) return nullptr;
    return strdup(result.c_str());
}

// ============================================================
// Token/Asset attribute setters
// ============================================================

static int be_set_attrib_token(NanousdPrim prim, const char* name, const char* value) {
    if (!prim || !name || !value || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    if (!attr.Set(VtValue(TfToken(value)), UsdTimeCode::Default())) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attrib_asset(NanousdPrim prim, const char* name, const char* value) {
    if (!prim || !name || !value || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    if (!attr.Set(VtValue(SdfAssetPath(value)), UsdTimeCode::Default())) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_attribarraytokens(NanousdPrim prim, const char* name,
                                     const char** values, int count) {
    if (!prim || !name || !values || count < 0 || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    VtArray<TfToken> arr(count);
    for (int i = 0; i < count; ++i) arr[i] = TfToken(values[i] ? values[i] : "");
    if (!attr.Set(VtValue(arr), UsdTimeCode::Default())) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_attribarraytokens_len(NanousdPrim prim, const char* name) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return -1;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return -1;
    if (val.IsHolding<VtArray<TfToken>>())
        return static_cast<int>(val.UncheckedGet<VtArray<TfToken>>().size());
    return -1;
}

static const char* be_attribarraytokens(NanousdPrim prim, const char* name, int index) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return nullptr;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return nullptr;
    if (val.IsHolding<VtArray<TfToken>>()) {
        const auto& arr = val.UncheckedGet<VtArray<TfToken>>();
        if (index < 0 || index >= static_cast<int>(arr.size())) return nullptr;
        prim->cachedStringVal = arr[index].GetString();
        return prim->cachedStringVal.c_str();
    }
    return nullptr;
}

// ============================================================
// Composition arc write — References
// ============================================================

static int be_add_reference(NanousdPrim prim, const char* assetPath, const char* primPath) {
    if (!prim || !prim->prim.IsValid()) return 0;
    std::string asset = assetPath ? assetPath : "";
    SdfPath pPath = primPath ? SdfPath(primPath) : SdfPath();
    if (!prim->prim.GetReferences().AddReference(SdfReference(asset, pPath))) return 0;
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// Relationship creation
// ============================================================

static int be_create_rel(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->prim.IsValid()) return 0;
    auto rel = prim->prim.CreateRelationship(TfToken(name));
    if (!rel.IsValid()) return 0;
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// Schema registration (no-op for OpenUSD backend)
// ============================================================

static int be_register_schemas_json(const char* /*json*/) {
    // Not applicable for OpenUSD backend — schemas come from OpenUSD's own registry
    return 0;
}

// ============================================================
// Prim metadata
// ============================================================

static const char* be_prim_metadatas(NanousdPrim prim, const char* key, int* ok) {
    if (!prim || !key || !prim->prim.IsValid()) { if (ok) *ok = 0; return ""; }
    VtValue val;
    if (prim->prim.GetMetadata(TfToken(key), &val)) {
        if (val.IsHolding<std::string>()) {
            if (ok) *ok = 1;
            prim->cachedStringVal = val.UncheckedGet<std::string>();
            return prim->cachedStringVal.c_str();
        }
        if (val.IsHolding<TfToken>()) {
            if (ok) *ok = 1;
            prim->cachedStringVal = val.UncheckedGet<TfToken>().GetString();
            return prim->cachedStringVal.c_str();
        }
    }
    if (ok) *ok = 0;
    return "";
}

static double be_prim_metadatad(NanousdPrim prim, const char* key, int* ok) {
    if (!prim || !key || !prim->prim.IsValid()) { if (ok) *ok = 0; return 0.0; }
    VtValue val;
    if (prim->prim.GetMetadata(TfToken(key), &val)) {
        if (val.IsHolding<double>()) { if (ok) *ok = 1; return val.UncheckedGet<double>(); }
        if (val.IsHolding<float>()) { if (ok) *ok = 1; return static_cast<double>(val.UncheckedGet<float>()); }
        if (val.IsHolding<int>()) { if (ok) *ok = 1; return static_cast<double>(val.UncheckedGet<int>()); }
    }
    if (ok) *ok = 0;
    return 0.0;
}

static int be_set_prim_metadatas(NanousdPrim prim, const char* key, const char* value) {
    if (!prim || !key || !value || !prim->prim.IsValid()) return 0;
    if (!prim->prim.SetMetadata(TfToken(key), VtValue(std::string(value)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_prim_metadatad(NanousdPrim prim, const char* key, double value) {
    if (!prim || !key || !prim->prim.IsValid()) return 0;
    if (!prim->prim.SetMetadata(TfToken(key), VtValue(value))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_set_prim_metadata_token(NanousdPrim prim, const char* key, const char* value) {
    if (!prim || !key || !value || !prim->prim.IsValid()) return 0;
    if (!prim->prim.SetMetadata(TfToken(key), VtValue(TfToken(value)))) return 0;
    prim->InvalidateCache();
    return 1;
}

// Token scalar reader
static const char* be_attrib_token(NanousdPrim prim, const char* name, int* ok) {
    auto attr = FindAttr(prim, name);
    if (attr.IsValid()) {
        VtValue val;
        if (attr.Get(&val, UsdTimeCode::Default())) {
            if (val.IsHolding<TfToken>()) {
                if (ok) *ok = 1;
                prim->cachedStringVal = val.UncheckedGet<TfToken>().GetString();
                return prim->cachedStringVal.c_str();
            }
        }
    }
    if (ok) *ok = 0;
    return "";
}

// ============================================================
// Attribute metadata & connections
// ============================================================

static const char* be_attrib_interpolation(NanousdPrim prim, const char* name) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return nullptr;
    VtValue val;
    if (!attr.GetMetadata(TfToken("interpolation"), &val)) return nullptr;
    if (val.IsHolding<TfToken>()) {
        prim->cachedStringVal = val.UncheckedGet<TfToken>().GetString();
        return prim->cachedStringVal.c_str();
    }
    return nullptr;
}

static int be_attrib_authored(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    return attr.IsValid() && attr.HasAuthoredValue() ? 1 : 0;
}

static bool PrimHasAppliedAPI(const UsdPrim& prim, const TfToken& apiName) {
    auto schemas = prim.GetAppliedSchemas();
    for (const auto& schema : schemas) {
        if (schema == apiName) return true;
    }
    return false;
}

static const char* be_attrib_colorspace(NanousdPrim prim, const char* name, int* ok) {
    if (ok) *ok = 0;
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return "";
    VtValue val;
    if (!attr.GetMetadata(TfToken("colorSpace"), &val)) return "";
    if (!val.IsHolding<TfToken>()) return "";
    if (ok) *ok = 1;
    prim->cachedStringVal = val.UncheckedGet<TfToken>().GetString();
    return prim->cachedStringVal.c_str();
}

static TfToken ComputePrimColorSpaceName(const UsdPrim& prim) {
    static const TfToken colorSpaceAPI("ColorSpaceAPI");
    static const TfToken colorSpaceName("colorSpace:name");
    for (UsdPrim cur = prim; cur.IsValid(); cur = cur.GetParent()) {
        if (!PrimHasAppliedAPI(cur, colorSpaceAPI)) continue;
        auto attr = cur.GetAttribute(colorSpaceName);
        if (!attr.IsValid() || !attr.HasAuthoredValue()) continue;
        VtValue val;
        if (!attr.Get(&val, UsdTimeCode::Default())) return TfToken();
        if (val.IsHolding<TfToken>()) return val.UncheckedGet<TfToken>();
        return TfToken();
    }
    return TfToken();
}

static const char* be_attrib_resolved_colorspace(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->prim.IsValid()) return "";
    int ok = 0;
    const char* authored = be_attrib_colorspace(prim, name, &ok);
    if (ok) return authored;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return "";
    prim->cachedStringVal = ComputePrimColorSpaceName(prim->prim).GetString();
    return prim->cachedStringVal.c_str();
}

static int be_set_attrib_colorspace(NanousdPrim prim, const char* name,
                                    const char* colorSpace) {
    if (!prim || !name || !colorSpace || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    if (!attr.SetMetadata(TfToken("colorSpace"), VtValue(TfToken(colorSpace)))) return 0;
    prim->InvalidateCache();
    return 1;
}

static int be_clear_attrib_colorspace(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    if (!attr.ClearMetadata(TfToken("colorSpace"))) return 0;
    prim->InvalidateCache();
    return 1;
}

static const char* be_prim_resolved_colorspace(NanousdPrim prim) {
    if (!prim || !prim->prim.IsValid()) return "";
    prim->cachedStringVal = ComputePrimColorSpaceName(prim->prim).GetString();
    return prim->cachedStringVal.c_str();
}

static int be_hasconnections(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    SdfPathVector sources;
    attr.GetConnections(&sources);
    return sources.empty() ? 0 : 1;
}

static int be_nconnections(NanousdPrim prim, const char* name) {
    if (!prim || !name || !prim->prim.IsValid()) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    SdfPathVector sources;
    attr.GetConnections(&sources);
    prim->cachedConnectionPaths.clear();
    for (const auto& p : sources)
        prim->cachedConnectionPaths.push_back(p.GetString());
    return static_cast<int>(prim->cachedConnectionPaths.size());
}

static const char* be_connection(NanousdPrim prim, const char* name, int index) {
    if (!prim || !name) return "";
    // Must call nconnections first to populate cache
    if (index < 0 || index >= static_cast<int>(prim->cachedConnectionPaths.size()))
        return "";
    return prim->cachedConnectionPaths[index].c_str();
}

// ============================================================
// Parent prim traversal
// ============================================================

static NanousdPrim be_parent(NanousdPrim prim) {
    if (!prim || !prim->prim.IsValid()) return nullptr;
    UsdPrim parent = prim->prim.GetParent();
    if (!parent.IsValid()) return nullptr;
    auto* h = new NanousdPrim_s;
    h->prim = parent;
    h->stage = prim->stage;
    return h;
}

// ============================================================
// Int64 array read
// ============================================================

static int be_attribarrayi64(NanousdPrim prim, const char* name, int64_t* out, int maxlen) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid() || !out || maxlen <= 0) return -1;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return -1;
    if (val.IsHolding<VtArray<int64_t>>()) {
        const auto& arr = val.UncheckedGet<VtArray<int64_t>>();
        int n = std::min(maxlen, static_cast<int>(arr.size()));
        for (int i = 0; i < n; ++i) out[i] = arr[i];
        return n;
    }
    if (val.IsHolding<VtArray<int>>()) {
        const auto& arr = val.UncheckedGet<VtArray<int>>();
        int n = std::min(maxlen, static_cast<int>(arr.size()));
        for (int i = 0; i < n; ++i) out[i] = static_cast<int64_t>(arr[i]);
        return n;
    }
    return -1;
}

// ============================================================
// Token-or-string array element reader
// ============================================================

static const char* be_attribarrays_elem(NanousdPrim prim, const char* name, int index) {
    auto attr = FindAttr(prim, name);
    if (!attr.IsValid()) return nullptr;
    VtValue val;
    if (!attr.Get(&val, UsdTimeCode::Default())) return nullptr;
    if (val.IsHolding<VtArray<TfToken>>()) {
        const auto& arr = val.UncheckedGet<VtArray<TfToken>>();
        if (index < 0 || index >= static_cast<int>(arr.size())) return nullptr;
        prim->cachedStringVal = arr[index].GetString();
        return prim->cachedStringVal.c_str();
    }
    if (val.IsHolding<VtArray<std::string>>()) {
        const auto& arr = val.UncheckedGet<VtArray<std::string>>();
        if (index < 0 || index >= static_cast<int>(arr.size())) return nullptr;
        prim->cachedStringVal = arr[index];
        return prim->cachedStringVal.c_str();
    }
    return nullptr;
}

// ============================================================
// Extended time sample setters
// ============================================================

static int be_set_samplev4f(NanousdPrim prim, const char* name, double time, const float v[4]) {
    if (!v) return 0;
    return SetTimeSampleTyped(prim, name, time, GfVec4f(v[0], v[1], v[2], v[3]));
}

static int be_set_sampleqf(NanousdPrim prim, const char* name, double time, const float v[4]) {
    if (!v) return 0;
    return SetTimeSampleTyped(prim, name, time, GfQuatf(v[0], GfVec3f(v[1], v[2], v[3])));
}

static int be_set_sample_token(NanousdPrim prim, const char* name, double time, const char* value) {
    if (!value) return 0;
    return SetTimeSampleTyped(prim, name, time, TfToken(value));
}

template <typename T>
static int SetTimeSampleArray(NanousdPrim_s* p, const char* name, double time,
                               const T* data, int count) {
    if (!p || !name || !p->prim.IsValid() || !data || count < 0) return 0;
    auto attr = p->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    VtArray<T> arr(count);
    for (int i = 0; i < count; ++i) arr[i] = data[i];
    if (!attr.Set(VtValue(arr), UsdTimeCode(time))) return 0;
    p->InvalidateCache();
    return 1;
}

static int be_set_samplearrayf(NanousdPrim prim, const char* name, double time,
                                const float* data, int count) {
    return SetTimeSampleArray(prim, name, time, data, count);
}

static int be_set_samplearrayi(NanousdPrim prim, const char* name, double time,
                                const int* data, int count) {
    return SetTimeSampleArray(prim, name, time, data, count);
}

static int be_set_samplearrayv3f(NanousdPrim prim, const char* name, double time,
                                  const float* data, int count) {
    if (!prim || !name || !prim->prim.IsValid() || !data || count < 0) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    VtArray<GfVec3f> arr(count);
    std::memcpy(arr.data(), data, count * sizeof(GfVec3f));
    if (!attr.Set(VtValue(arr), UsdTimeCode(time))) return 0;
    prim->InvalidateCache();
    return 1;
}

// --- Double-precision time sample setters ---

static int be_set_samplev2d(NanousdPrim prim, const char* name, double time, const double v[2]) {
    if (!v) return 0;
    return SetTimeSampleTyped(prim, name, time, GfVec2d(v[0], v[1]));
}

static int be_set_samplev4d(NanousdPrim prim, const char* name, double time, const double v[4]) {
    if (!v) return 0;
    return SetTimeSampleTyped(prim, name, time, GfVec4d(v[0], v[1], v[2], v[3]));
}

static int be_set_samplem4d(NanousdPrim prim, const char* name, double time, const double v[16]) {
    if (!v) return 0;
    GfMatrix4d m;
    const double* d = v;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            m[i][j] = d[i * 4 + j];
    return SetTimeSampleTyped(prim, name, time, m);
}

static int be_set_samplearrayd(NanousdPrim prim, const char* name, double time,
                                const double* data, int count) {
    return SetTimeSampleArray(prim, name, time, data, count);
}

static int be_set_samplearrayv3d(NanousdPrim prim, const char* name, double time,
                                  const double* data, int count) {
    if (!prim || !name || !prim->prim.IsValid() || !data || count < 0) return 0;
    auto attr = prim->prim.GetAttribute(TfToken(name));
    if (!attr.IsValid()) return 0;
    VtArray<GfVec3d> arr(count);
    std::memcpy(arr.data(), data, count * sizeof(GfVec3d));
    if (!attr.Set(VtValue(arr), UsdTimeCode(time))) return 0;
    prim->InvalidateCache();
    return 1;
}

// ============================================================
// Instancing
// ============================================================

static int be_isinstance(NanousdPrim prim) {
    return (prim && prim->prim.IsInstance()) ? 1 : 0;
}

static int be_isprototype(NanousdPrim prim) {
    return (prim && prim->prim.IsPrototype()) ? 1 : 0;
}

static int be_isinprototype(NanousdPrim prim) {
    return (prim && prim->prim.IsInPrototype()) ? 1 : 0;
}

static NanousdPrim be_prototype(NanousdPrim prim) {
    if (!prim || !prim->stage) return nullptr;
    auto proto = prim->prim.GetPrototype();
    if (!proto.IsValid()) return nullptr;

    auto* h = new NanousdPrim_s;
    h->prim = proto;
    h->stage = prim->stage;
    return h;
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

    auto* h = new NanousdPrim_s;
    h->prim = instances[index];
    h->stage = prim->stage;
    return h;
}

// ============================================================
// Diagnostics (stubs — OpenUSD backend has no diagnostic collector)
// ============================================================

static NanousdDiagnostic* be_diagnostics(NanousdStage stage, int* count) {
    if (count) *count = 0;
    if (!stage) return nullptr;

    const int n = static_cast<int>(stage->diagnostics.size());
    if (count) *count = n;
    if (n == 0) return nullptr;

    auto* arr = static_cast<NanousdDiagnostic*>(
        malloc(sizeof(NanousdDiagnostic) * static_cast<size_t>(n)));
    if (!arr) { if (count) *count = 0; return nullptr; }

    for (int i = 0; i < n; ++i) {
        const auto& d = stage->diagnostics[static_cast<size_t>(i)];
        arr[i].severity   = d.severity;
        arr[i].category   = 7;  // Other — no structured category from pxr
        arr[i].message    = strdup(d.message.c_str());
        arr[i].prim_path  = strdup("");
        arr[i].layer_path = strdup("");
        arr[i].asset_path = strdup("");
        arr[i].arc_type   = 3;  // None
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
    if (!stage || stage->diagnostics.empty()) {
        char* empty = static_cast<char*>(malloc(3));
        if (empty) memcpy(empty, "[]", 3);
        return empty;
    }

    // Hand-build JSON array
    std::string json = "[";
    for (size_t i = 0; i < stage->diagnostics.size(); ++i) {
        if (i > 0) json += ',';
        const auto& d = stage->diagnostics[i];
        const char* sev = d.severity >= 2 ? "error" : (d.severity == 1 ? "warning" : "info");
        json += "{\"severity\":\"";
        json += sev;
        json += "\",\"category\":\"other\",\"message\":\"";
        // Escape the message for JSON
        for (char c : d.message) {
            switch (c) {
                case '"':  json += "\\\""; break;
                case '\\': json += "\\\\"; break;
                case '\n': json += "\\n";  break;
                case '\r': json += "\\r";  break;
                case '\t': json += "\\t";  break;
                default:   json += c; break;
            }
        }
        json += "\",\"primPath\":\"\",\"layerPath\":\"\",\"assetPath\":\"\",\"arcType\":\"none\"}";
    }
    json += ']';

    char* result = static_cast<char*>(malloc(json.size() + 1));
    if (!result) return nullptr;
    memcpy(result, json.c_str(), json.size() + 1);
    return result;
}

// ============================================================
// Variants (spec §11.2)
// Implemented by delegating to pxr's UsdVariantSets/UsdVariantSet.
// setvariantselection's layerIndex is ignored in this backend — writes
// go through the stage's current edit target (typically the root
// layer). nanousd's multi-layer variant of set-by-index isn't exposed
// through pxr without a UsdEditTarget dance; acceptable here because
// the OpenUSD backend is for cross-implementation parity, not full
// write-into-any-layer fidelity.
// ============================================================

static int be_nvariantsets(NanousdPrim prim) {
    if (!prim || !prim->prim.IsValid()) return 0;
    return static_cast<int>(prim->prim.GetVariantSets().GetNames().size());
}

static const char* be_variantsetname(NanousdPrim prim, int index) {
    if (!prim || !prim->prim.IsValid()) return "";
    auto names = prim->prim.GetVariantSets().GetNames();
    if (index < 0 || index >= static_cast<int>(names.size())) return "";
    prim->cachedStringVal = names[static_cast<size_t>(index)];
    return prim->cachedStringVal.c_str();
}

static int be_hasvariantset(NanousdPrim prim, const char* setName) {
    if (!prim || !prim->prim.IsValid() || !setName) return 0;
    auto names = prim->prim.GetVariantSets().GetNames();
    std::string target(setName);
    for (const auto& n : names) {
        if (n == target) return 1;
    }
    return 0;
}

static int be_nvariants(NanousdPrim prim, const char* setName) {
    if (!prim || !prim->prim.IsValid() || !setName) return 0;
    if (!be_hasvariantset(prim, setName)) return 0;
    auto vset = prim->prim.GetVariantSet(setName);
    return static_cast<int>(vset.GetVariantNames().size());
}

static const char* be_variantname(NanousdPrim prim, const char* setName, int index) {
    if (!prim || !prim->prim.IsValid() || !setName) return "";
    if (!be_hasvariantset(prim, setName)) return "";
    auto vset = prim->prim.GetVariantSet(setName);
    auto names = vset.GetVariantNames();
    if (index < 0 || index >= static_cast<int>(names.size())) return "";
    prim->cachedStringVal = names[static_cast<size_t>(index)];
    return prim->cachedStringVal.c_str();
}

static const char* be_variantselection(NanousdPrim prim, const char* setName) {
    if (!prim || !prim->prim.IsValid() || !setName) return "";
    if (!be_hasvariantset(prim, setName)) return "";
    auto vset = prim->prim.GetVariantSet(setName);
    prim->cachedStringVal = vset.GetVariantSelection();
    return prim->cachedStringVal.c_str();
}

static int be_setvariantselection(NanousdPrim prim, const char* setName,
                                   const char* variantName, int layerIndex) {
    (void)layerIndex;  // see header comment
    if (!prim || !prim->prim.IsValid() || !setName) return 0;
    auto vset = prim->prim.GetVariantSet(setName);
    bool ok = vset.SetVariantSelection(variantName ? variantName : "");
    if (ok) prim->InvalidateCache();
    return ok ? 1 : 0;
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

    /* --- P1 extensions --- */
    be_attribm3d,
    be_set_attribm3d,
    be_attribarrays_len,
    be_attribarrays,
    be_set_attribarrays,
    be_attribasset,
    be_get_local_transform,

    /* --- Binary write --- */
    be_write_usdc,

    /* --- Array time sample reads --- */
    be_samplev2f,
    be_samplearrayf,
    be_samplearrayd,
    be_samplearrayi,

    /* --- Stage root layer path --- */
    be_stage_get_root_layer_path,

    /* --- Prim specifier write --- */
    be_define_prim_s,
    be_set_specifier,

    /* --- USDA text write --- */
    be_write_usda,
    be_write_usda_string,

    /* --- Token/Asset attribute setters --- */
    be_set_attrib_token,
    be_set_attrib_asset,
    be_set_attribarraytokens,
    be_attribarraytokens_len,
    be_attribarraytokens,

    /* --- Composition arc write --- */
    be_add_reference,

    /* --- Relationship creation --- */
    be_create_rel,

    /* --- Schema registration --- */
    be_register_schemas_json,

    /* --- Prim metadata --- */
    be_prim_metadatas,
    be_prim_metadatad,
    be_set_prim_metadatas,
    be_set_prim_metadatad,
    be_set_prim_metadata_token,
    be_attrib_token,

    /* --- Attribute metadata & connections --- */
    be_attrib_interpolation,
    be_attrib_authored,
    be_hasconnections,
    be_nconnections,
    be_connection,

    /* --- Parent prim traversal --- */
    be_parent,

    /* --- Int64 array read --- */
    be_attribarrayi64,

    /* --- Token-or-string array element reader --- */
    be_attribarrays_elem,

    /* --- Extended time sample setters --- */
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

    /* The OpenUSD reference backend implements a contiguous prefix plus the
     * authored-attribute pair. Every unsupported slot between the prefix and
     * that pair is spelled out as null so the pair lands in its ABI-correct
     * position. Later appended semantic-traversal slots are left as
     * zero-initialized trailing nulls. The public C API null-guards them. */
    /* set_stage_metadata_token */ nullptr,
    /* stage_n_layers           */ nullptr,
    /* stage_layer_path         */ nullptr,
    /* open_masked              */ nullptr,

    /* --- Color-space resolution (appended for ABI compat) --- */
    be_attrib_colorspace,
    be_attrib_resolved_colorspace,
    be_set_attrib_colorspace,
    be_clear_attrib_colorspace,
    be_prim_resolved_colorspace,

    /* write_usdz               */ nullptr,
    /* collection_nmembers      */ nullptr,
    /* collection_member        */ nullptr,
    /* collection_contains      */ nullptr,
    /* nproperties              */ nullptr,
    /* propertyname             */ nullptr,
    /* property_is_attribute    */ nullptr,
    /* property_is_relationship */ nullptr,
    /* rel_authored             */ nullptr,
    /* layer_has_prim_spec      */ nullptr,
    /* layer_has_attr_opinion   */ nullptr,
    /* layer_attr_nsamples      */ nullptr,
    /* layer_prim_listop        */ nullptr,
    /* layer_n_sublayers        */ nullptr,
    /* layer_sublayer_path      */ nullptr,
    /* layer_offset             */ nullptr,
    /* recompose                */ nullptr,
    /* add_payload              */ nullptr,
    /* add_inherit              */ nullptr,
    /* add_specialize           */ nullptr,
    /* remove_listop_item       */ nullptr,
    /* set_active               */ nullptr,
    /* set_instanceable         */ nullptr,
    /* remove_api               */ nullptr,
    /* remove_prim              */ nullptr,
    /* create_variantset        */ nullptr,
    /* create_variant           */ nullptr,
    /* resolve_asset_path       */ nullptr,
    /* stage_resolve_asset_path */ nullptr,
    /* read_asset_bytes         */ nullptr,
    /* free_bytes               */ nullptr,
    /* rel_metadatas            */ nullptr,

    /* --- Authored attribute enumeration (appended for ABI compat) --- */
    be_nauthored_attribs,
    be_authored_attribname,
};

// ============================================================
// Entry point
// ============================================================

extern "C" NANOUSD_BACKEND_API NanousdBackend_v1* nanousd_create_backend_v1(void) {
    return &s_backend;
}
